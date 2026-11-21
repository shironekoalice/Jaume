"""Jaume-Loop: a decoder-only LM whose depth can be looped, and whose interior
is observable.

Why this module exists
----------------------
DeepSeek's recent looped-transformer work (LoopMTP 2608.03624, ECHO
2610.24766, Looped Latent Attention 2607.15456) shows that reusing one stack
of layers across T iterations buys effective depth at fixed parameter count,
and LoopMTP reports up to +8.1% average accuracy with stability to 15 loops.
All three USE the loop; none of them MEASURE what happens inside it.

LoopMTP supervises the loop with a latent soft alignment but says nothing
about how the representation is organised across rounds.  LLA shows the
per-loop KV cache is low-rank across loops, but not what the low-rank part
is.  Jaume has spent three years building representation probes (memory-slot
diagnostics, probe-validity checks, cross-layer path execution), which is
exactly the instrument this gap needs.

So the module is built to be OBSERVABLE first and capable second:

  * `return_loop_states=True` hands back the hidden state after every loop
    iteration, so cross-round subspace overlap, which coordinates stay put
    and which move, and where the rounds stop changing can be measured
    directly instead of inferred;
  * `num_loops` is a first-class knob, not a training trick, so loop depth
    can be swept at inference time;
  * `loop_position_mode` decides what a round means for positional
    encoding, and both options are kept because the choice is itself a
    research variable (see below).

Loop forms
----------
  full    every layer is applied num_loops times.  Effective depth
          = num_layers * num_loops, parameters unchanged.  This is the form
          LoopMTP describes ("reusing one stack of layers across T
          iterations").
  tail    the first `num_layers - loop_layers` layers run once and only the
          last `loop_layers` are looped.  This is the cheaper "recurrent
          depth" arrangement and keeps the early layers single-shot.

Position handling inside a loop
-------------------------------
  repeat   every round sees the same positions (the round index is not
           encoded in the position).  Relative positions inside the context
           stay truthful, and the only thing that changes between rounds is
           the computation.
  increment each round advances the position by the context length, i.e. the
           rounds behave like one longer sequence.  This makes the rounds
           distinguishable to RoPE but pushes positions past the trained
           range, which is why it is not the default.

The default is `repeat`.  Whether the model *should* be able to tell rounds
apart is an open question, and keeping both options makes it measurable
rather than assumed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .backbone import RMSNorm
from .backbone_causal import CausalEncoderLayer


class LoopLMBackbone(nn.Module):
    """Causal backbone with an optional looped (recurrent depth) schedule."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.num_loops = max(1, int(getattr(config, "num_loops", 1)))
        self.loop_form = getattr(config, "loop_form", "full")
        self.loop_layers = int(getattr(config, "loop_layers", 0) or 0)
        self.loop_position_mode = getattr(config, "loop_position_mode", "repeat")
        # Break-the-contraction knobs.  See the 2026-11-07 probe finding in
        # docs/loop-spec.md 8.3: the plain loop converges to a fixed point
        # (round-to-round distance decays geometrically), so extra rounds do
        # no work.  A per-round gate lets the model re-weight each round's
        # update; an optional per-round positional offset lets RoPE tell the
        # rounds apart.  Both default OFF so the plain loop stays the
        # baseline.
        self.loop_gate = bool(getattr(config, "loop_gate", False))
        self.loop_gate_min = float(getattr(config, "loop_gate_min", 0.0))
        self.loop_pos_offset = bool(getattr(config, "loop_pos_offset", False))

        if self.loop_form == "tail":
            if self.loop_layers <= 0 or self.loop_layers >= self.num_layers:
                raise ValueError(
                    "loop_form='tail' needs 0 < loop_layers < num_layers, got "
                    f"loop_layers={self.loop_layers} num_layers={self.num_layers}")
            n_unique = self.num_layers
            n_looped = self.loop_layers
        else:
            n_unique = self.num_layers
            n_looped = self.num_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [CausalEncoderLayer(config) for _ in range(n_unique)])
        if self.loop_gate:
            # [num_loops] scalars, one per round, shared across layers.
            # Round 0 is pinned to 1.0: at initialisation the gated loop is
            # EXACTLY the plain loop, so the intervention can only change
            # behaviour if training moves it -- it cannot start worse.
            #
            # The pin is done by EXCLUDING round 0 from the parameter (a
            # registered buffer would still receive optimizer updates via a
            # hook, and a hook-based pin breaks on NPU because the hook
            # receives gradients on the accelerator while the mask lives on
            # the CPU).  self.loop_gate_param therefore has num_loops-1
            # entries covering rounds 1..T-1, and round 0 is handled in
            # forward() as a hard-coded 1.0.
            n = max(1, self.num_loops)
            self.loop_gate_param = nn.Parameter(
                torch.ones(max(1, n - 1))) if n > 1 else None
        else:
            self.loop_gate_param = None
        self.norm = RMSNorm(config.hidden_size)
        self._gradient_checkpointing = False

    # -- introspection ------------------------------------------------
    @property
    def effective_depth(self):
        if self.loop_form == "tail":
            return (self.num_layers - self.loop_layers) + self.loop_layers * self.num_loops
        return self.num_layers * self.num_loops

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def _run_layer(self, layer, x, attention_mask):
        if self._gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                layer, x, attention_mask, use_reentrant=False)
        return layer(x, attention_mask)

    def _apply_layers(self, layer_ids, hidden_states, attention_mask):
        aux_total = None
        for li in layer_ids:
            hidden_states, aux = self._run_layer(self.layers[li], hidden_states,
                                                attention_mask)
            if aux is not None:
                aux_total = aux if aux_total is None else aux_total + aux
        return hidden_states, aux_total

    def forward(self, input_ids, attention_mask=None, return_loop_states=False):
        """Returns (hidden, aux_total) or (hidden, aux_total, loop_states).

        loop_states is a list with one entry per loop round (each
        [B, L, D]); it is the handle the cross-round probe needs.
        """
        hidden_states = self.embed_tokens(input_ids)
        aux_total = torch.tensor(0.0, device=input_ids.device)
        loop_states = []
        # Optional per-round positional offset: round t shifts its positions
        # by t*L... NO -- see the config knob loop_pos_offset: shifting by a
        # CONSTANT per round distinguishes rounds without leaving the trained
        # positional range.  Implemented in attention_causal via the
        # position_offset attribute set here per round.
        pos_offset = getattr(self, "_round_pos_offset", 0)

        if self.loop_form == "tail":
            n_plain = self.num_layers - self.loop_layers
            hidden_states, aux = self._apply_layers(
                list(range(n_plain)), hidden_states, attention_mask)
            if aux is not None:
                aux_total = aux_total + aux
            looped_ids = list(range(n_plain, self.num_layers))
        else:
            looped_ids = list(range(self.num_layers))

        for t in range(self.num_loops):
            if self.loop_pos_offset:
                self._round_pos_offset = t
            h_before = hidden_states
            hidden_states, aux = self._apply_layers(
                looped_ids, hidden_states, attention_mask)
            if self.loop_gate_param is not None and t > 0:
                # scale the ROUND'S UPDATE, not the state: this keeps the
                # residual stream intact and only re-weights what this round
                # contributes on top of it.  loop_gate_param[t-1] is round t
                # because round 0 is pinned and not stored.
                update = hidden_states - h_before
                g = self.loop_gate_param[t - 1].clamp(
                    min=self.loop_gate_min).to(hidden_states.dtype)
                hidden_states = h_before + g * update
            if aux is not None:
                aux_total = aux_total + aux
            if return_loop_states:
                loop_states.append(self.norm(hidden_states))

        hidden_states = self.norm(hidden_states)
        if return_loop_states:
            return hidden_states, aux_total, loop_states
        return hidden_states, aux_total


class LoopLM(nn.Module):
    """Decoder-only language model on top of the looped backbone.

    The LM head is optionally tied to the token embedding.  It is NOT tied by
    default: Jaume's embedding matrix was trained with a masked-LM objective
    and the [MASK] row of a tied head would then be shared between a token
    nobody predicts and a token everybody does.  Tying is available for
    parameter-count parity experiments.
    """

    def __init__(self, config, tie_lm_head=False):
        super().__init__()
        self.config = config
        self.backbone = LoopLMBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size,
                                 bias=False)
        if tie_lm_head:
            self.lm_head.weight = self.backbone.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None, return_loop_states=False):
        out = self.backbone(input_ids, attention_mask,
                            return_loop_states=return_loop_states)
        if return_loop_states:
            hidden, aux, loop_states = out
            logits = self.lm_head(hidden)
            return logits, aux, loop_states
        hidden, aux = out
        return self.lm_head(hidden), aux

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
