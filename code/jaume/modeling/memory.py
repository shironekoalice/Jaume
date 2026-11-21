"""Jaume-Memory: latent memory-slot pooling (JM-S) and cross-input memory (JM-R).

Plan: docs/memory-plan.md

JM-S  latent_memory            : N learned latent slots cross-attend over the
                                 encoder hidden states; the slot states are
                                 pooled into the sentence vector.
JM-R  recurrent_memory[_gate]  : the slot state produced for chunk t is carried
                                 into chunk t+1 as the slot initialisation,
                                 optionally through a learned update gate.

Every pooler in this module shares one signature::

    pooled, slot_state = pooler(hidden_states, attention_mask, memory_state)

`pooled` is [B, D] and feeds the projection head; `slot_state` is [B, N, D] for
memory arms and None for the classic poolers, so all arms run through exactly
the same training / evaluation code path.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import RMSNorm
from .pooling import MeanPooling, CLSPooling, AttentionPooling


def _pad_mask(attention_mask, dtype):
    """[B, T] -> [B, 1, 1, T] additive mask for SDPA (None when no padding)."""
    if attention_mask is None:
        return None
    return (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min


class _ClassicWrapper(nn.Module):
    """Adapts a classic pooler to the (pooled, slot_state) signature."""

    def __init__(self, pooler):
        super().__init__()
        self.pooler = pooler

    def forward(self, hidden_states, attention_mask=None, memory_state=None):
        return self.pooler(hidden_states, attention_mask), None


class AttentionMLPPooling(nn.Module):
    """Attention pooling with an MLP scorer.

    Used as the parameter-matched B2 control: plain AttentionPooling has only
    `hidden_size` parameters, while JM-S adds a cross-attention block. Widening
    the scorer to `mlp_width` brings B2 up to a comparable parameter budget so
    that "more learned pooling parameters" is not confounded with "memory
    slots".
    """

    def __init__(self, hidden_size, mlp_width=2048):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.score = nn.Sequential(
            nn.Linear(hidden_size, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, 1),
        )
        self.scale = hidden_size ** -0.5

    def forward(self, hidden_states, attention_mask=None, memory_state=None):
        h = hidden_states * self.scale
        scores = self.score(h).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1), None


class LatentMemoryPooling(nn.Module):
    """JM-S: N learned latent slots cross-attend over the hidden states.

    Two attention normalisations are available, and the choice is the whole
    ballgame for whether slots specialise (see docs/memory-results.md):

    `slot_competition=False` (v1, failed) -- softmax over TOKENS, computed
    independently per slot. Nothing stops every slot from reading the same
    tokens, and in practice they converge to one another (measured: slot-state
    cosine 0.87, JS divergence 0.04).

    `slot_competition=True` (Slot Attention, Locatello et al. 2020,
    arXiv:2006.15055) -- softmax over SLOTS: each token's attention mass is
    split *competitively* among slots, so slots must divide the input between
    them. This is what forces specialisation.

    `slot_iterations>1` adds the paper's iterative refinement: the slots are
    updated over several rounds against the same tokens, which is what lets
    the competitive assignment stabilise (a single pass has no chance to
    re-assign after the first read).

    Ablations:
      E1 num_slots            : effective capacity of the slot bank
      E2 learn_slots=False    : gain from learned slots vs the attention
                                structure alone (slots become a fixed random
                                buffer, still projected)
      E4 slot_interaction     : whether slots may talk to each other
    """

    def __init__(
        self,
        hidden_size,
        num_slots=8,
        num_heads=8,
        learn_slots=True,
        slot_interaction=False,
        slot_ffn=True,
        slot_pool="mean",
        init_scale=0.02,
        slot_competition=False,
        slot_iterations=1,
        slot_gru=True,
        slot_eps=1e-8,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} not divisible by {num_heads} heads")
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.learn_slots = learn_slots
        self.slot_interaction = slot_interaction
        self.slot_pool = slot_pool
        self.slot_competition = slot_competition
        self.slot_iterations = int(slot_iterations)
        self.slot_eps = slot_eps
        self.scale = self.head_dim ** -0.5

        slots = torch.randn(num_slots, hidden_size) * init_scale
        if learn_slots:
            self.slots = nn.Parameter(slots)
        else:
            # E2: fixed random slots (not optimised) -- keeps the attention
            # structure, removes the learned-slot hypothesis.
            self.register_buffer("slots", slots)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_slots = RMSNorm(hidden_size)
        self.norm_hidden = RMSNorm(hidden_size)

        # Slot Attention's GRU-style refinement: decide how much of the freshly
        # read state to keep vs the previous one. Used when iterations > 1.
        self.slot_gru = slot_gru
        if slot_gru and self.slot_iterations > 1:
            self.gru_gate = nn.Linear(hidden_size, hidden_size)
            self.gru_cand = nn.Linear(hidden_size, hidden_size)

        if slot_interaction:
            self.slot_attn_norm = RMSNorm(hidden_size)
            self.slot_self_attn = nn.MultiheadAttention(
                hidden_size, num_heads, batch_first=True
            )
            self.slot_ffn = None
            if slot_ffn:
                self.slot_ffn_norm = RMSNorm(hidden_size)
                self.slot_ffn = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 2),
                    nn.GELU(),
                    nn.Linear(hidden_size * 2, hidden_size),
                )

        if slot_pool == "attention":
            self.pool_query = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.pool_scale = hidden_size ** -0.5

    # -- core -------------------------------------------------------------
    def _kv(self, hidden):
        """Projects hidden states to keys/values once per input.

        With slot_iterations > 1 the tokens are re-read on every round; the key
        and value tensors do not change, so projecting them inside the loop
        wastes ~2/3 of the compute. Cached here instead.
        """
        B, T, _ = hidden.shape
        H, hd = self.num_heads, self.head_dim
        hv = self.norm_hidden(hidden)
        k = self.k_proj(hv).view(B, T, H, hd).transpose(1, 2)
        v = self.v_proj(hv).view(B, T, H, hd).transpose(1, 2)
        return k, v

    def _cross_attend(self, slots, hidden, attention_mask, kv=None):
        """Returns (read_out, attn) where attn is [B, H, N, T] for diagnostics."""
        B, N, _ = slots.shape
        H, hd = self.num_heads, self.head_dim
        T = hidden.size(1)
        q = self.q_proj(self.norm_slots(slots)).view(B, N, H, hd).transpose(1, 2)

        if not self.slot_competition:
            # v1: independent per-slot softmax over tokens. Fast path via the
            # fused SDPA kernel.
            k, v = kv if kv is not None else self._kv(hidden)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=_pad_mask(attention_mask, q.dtype)
            )
            out = out.transpose(1, 2).contiguous().view(B, N, self.hidden_size)
            return self.o_proj(out), None

        # Slot Attention: softmax over the SLOT axis, so slots compete for each
        # token. Padding is applied as an additive mask on the token axis.
        k, v = kv if kv is not None else self._kv(hidden)
        logits = (q @ k.transpose(-2, -1)) * self.scale          # [B,H,N,T]
        if attention_mask is not None:
            logits = logits + _pad_mask(attention_mask, logits.dtype)
        attn = torch.softmax(logits, dim=-2)                      # over slots
        # renormalise over tokens so each slot performs a weighted mean
        if attention_mask is not None:
            attn = attn * attention_mask[:, None, None, :].to(attn.dtype)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + self.slot_eps)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, self.hidden_size)
        return self.o_proj(out), attn

    def _refine(self, slots, read):
        """GRU-style slot update (Slot Attention). Falls back to a plain
        residual when the GRU is disabled."""
        if not (self.slot_gru and self.slot_iterations > 1):
            return slots + read
        gate = torch.sigmoid(self.gru_gate(read))
        cand = torch.tanh(self.gru_cand(read))
        return gate * slots + (1.0 - gate) * cand

    def slot_pass(self, hidden, attention_mask=None, init_slots=None):
        """Runs the slot bank over one input; returns the new slot states."""
        B = hidden.size(0)
        if init_slots is None:
            slots = self.slots.unsqueeze(0).expand(B, -1, -1).contiguous()
        else:
            slots = init_slots
        kv = self._kv(hidden)   # projected once, reused across refinement rounds
        last_attn = None
        iters = max(1, self.slot_iterations)
        for i in range(iters):
            read, last_attn = self._cross_attend(slots, hidden, attention_mask, kv)
            slots = self._refine(slots, read)
        if self.slot_interaction:
            h = self.slot_attn_norm(slots)
            attn_out, _ = self.slot_self_attn(h, h, h, need_weights=False)
            slots = slots + attn_out
            if self.slot_ffn is not None:
                slots = slots + self.slot_ffn(self.slot_ffn_norm(slots))
        self.last_attn = last_attn
        return slots

    def pool_slots(self, slots):
        if self.slot_pool == "mean":
            return slots.mean(dim=1)
        scores = (slots @ self.pool_query.transpose(-2, -1)).squeeze(-1) * self.pool_scale
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (slots * weights).sum(dim=1)

    def forward(self, hidden_states, attention_mask=None, memory_state=None):
        slots = self.slot_pass(hidden_states, attention_mask, memory_state)
        return self.pool_slots(slots), slots


class GatedSlotUpdate(nn.Module):
    """JM-R gate: per-slot, per-dimension interpolation between old and new."""

    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, prev_slots, new_slots):
        g = torch.sigmoid(self.gate(torch.cat([prev_slots, new_slots], dim=-1)))
        return g * new_slots + (1.0 - g) * prev_slots


class RecurrentMemoryPooling(nn.Module):
    """JM-R: slot state carried across chunks of the same document.

    B5 (use_gate=False): the previous slot state is used directly as the slot
    initialisation for the next chunk; the update is whatever the cross-
    attention residual does.
    B6 (use_gate=True) : an explicit gated interpolation between the carried
    state and the freshly read state (ablation E5).

    `memory_state=None` (or an explicit reset at a document boundary) makes the
    pooler fall back to the learned slot bank, i.e. exactly JM-S on one chunk.
    """

    def __init__(self, hidden_size, use_gate=True, **slot_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_gate = use_gate
        self.core = LatentMemoryPooling(hidden_size, **slot_kwargs)
        if use_gate:
            self.gate = GatedSlotUpdate(hidden_size)

    @property
    def num_slots(self):
        return self.core.num_slots

    def forward(self, hidden_states, attention_mask=None, memory_state=None):
        B = hidden_states.size(0)
        if memory_state is None:
            init = None
        else:
            init = memory_state
        new_slots = self.core.slot_pass(hidden_states, attention_mask, init)
        if memory_state is not None and self.use_gate:
            carry = self.gate(memory_state, new_slots)
        else:
            carry = new_slots
        return self.core.pool_slots(new_slots), carry

    def initial_state(self, batch_size, device=None):
        """Fresh (reset) state for the start of a document."""
        return None


class SlotDiversityLoss(nn.Module):
    """Anti-collapse regulariser on slot states (E4 follow-up).

    The negative result showed all slots converge to the same read (slot-state
    cosine 0.87). Competition (Slot Attention) is the architectural fix; this
    is the explicit loss-based one: penalise the mean squared off-diagonal
    cosine between slot states so slots are pushed toward orthogonal
    (i.e. non-redundant) directions. Follows the diversity regularisation
    common in disentanglement / object-centric work.

    Returns 0 when there is no slot state (classic pooling arms).
    """

    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, slot_states):
        if slot_states is None or not self.weight:
            return torch.zeros((), device=slot_states.device if slot_states is not None else "cpu")
        s = F.normalize(slot_states.float(), dim=-1)          # [B, N, D]
        gram = s @ s.transpose(-1, -2)                        # [B, N, N]
        n = s.size(1)
        if n < 2:
            return torch.zeros((), device=s.device)
        eye = torch.eye(n, dtype=torch.bool, device=s.device)
        off = gram[:, ~eye]
        return self.weight * (off ** 2).mean()


def build_memory_pooling(config):
    """Builds the pooler named by config.pooling (memory arms + controls)."""
    name = getattr(config, "pooling", "mean")
    hidden = config.hidden_size
    slots = int(getattr(config, "num_slots", 8))
    heads = int(getattr(config, "slot_heads", 8))
    slot_kwargs = dict(
        num_slots=slots,
        num_heads=heads,
        learn_slots=bool(getattr(config, "learn_slots", True)),
        slot_interaction=bool(getattr(config, "slot_interaction", False)),
        slot_ffn=bool(getattr(config, "slot_ffn", True)),
        slot_pool=getattr(config, "slot_pool", "mean"),
        slot_competition=bool(getattr(config, "slot_competition", False)),
        slot_iterations=int(getattr(config, "slot_iterations", 1)),
        slot_gru=bool(getattr(config, "slot_gru", True)),
    )

    if name == "mean":
        return _ClassicWrapper(MeanPooling())
    if name == "cls":
        return _ClassicWrapper(CLSPooling())
    if name == "attention":
        return _ClassicWrapper(AttentionPooling(hidden))
    if name == "attention_mlp":
        return AttentionMLPPooling(hidden, mlp_width=int(getattr(config, "attn_mlp_width", 2048)))
    if name == "latent_memory":
        return LatentMemoryPooling(hidden, **slot_kwargs)
    if name == "recurrent_memory":
        return RecurrentMemoryPooling(hidden, use_gate=False, **slot_kwargs)
    if name == "recurrent_memory_gate":
        return RecurrentMemoryPooling(hidden, use_gate=True, **slot_kwargs)
    raise ValueError(f"unknown pooling: {name}")
