import io
p = "code/repo/jaume/modeling/loop_lm.py"
src = io.open(p, encoding="utf-8").read()

# ---- config knobs on the backbone ---------------------------------------
old = """        self.loop_position_mode = getattr(config, "loop_position_mode", "repeat")"""
new = """        self.loop_position_mode = getattr(config, "loop_position_mode", "repeat")
        # Break-the-contraction knobs.  See the 2026-11-07 probe finding in
        # docs/loop-spec.md 8.3: the plain loop converges to a fixed point
        # (round-to-round distance decays geometrically), so extra rounds do
        # no work.  A per-round gate lets the model re-weight each round's
        # update; an optional per-round positional offset lets RoPE tell the
        # rounds apart.  Both default OFF so the plain loop stays the
        # baseline.
        self.loop_gate = bool(getattr(config, "loop_gate", False))
        self.loop_gate_min = float(getattr(config, "loop_gate_min", 0.0))
        self.loop_pos_offset = bool(getattr(config, "loop_pos_offset", False))"""
assert old in src
src = src.replace(old, new)

# ---- the gate parameter --------------------------------------------------
old = """        self.norm = RMSNorm(config.hidden_size)
        self._gradient_checkpointing = False"""
new = """        if self.loop_gate:
            # [num_loops] scalars, one per round, shared across layers.
            # Round 0 is pinned to 1.0: at initialisation the gated loop is
            # EXACTLY the plain loop, so the intervention can only change
            # behaviour if training moves it -- it cannot start worse.
            g = torch.ones(max(1, self.num_loops))
            self.loop_gate_param = nn.Parameter(g)
            self.loop_gate_param.register_hook(
                lambda grad: grad * torch.tensor(
                    [0.0 if i == 0 else 1.0 for i in range(g.shape[0])]))
        else:
            self.loop_gate_param = None
        self.norm = RMSNorm(config.hidden_size)
        self._gradient_checkpointing = False"""
assert old in src
src = src.replace(old, new)

# ---- apply the gate inside the loop --------------------------------------
old = """        for _ in range(self.num_loops):
            hidden_states, aux = self._apply_layers(
                looped_ids, hidden_states, attention_mask)
            if aux is not None:
                aux_total = aux_total + aux
            if return_loop_states:
                loop_states.append(self.norm(hidden_states))"""
new = """        for t in range(self.num_loops):
            h_before = hidden_states
            hidden_states, aux = self._apply_layers(
                looped_ids, hidden_states, attention_mask)
            if self.loop_gate_param is not None and t > 0:
                # scale the ROUND'S UPDATE, not the state: this keeps the
                # residual stream intact and only re-weights what this round
                # contributes on top of it.
                update = hidden_states - h_before
                g = self.loop_gate_param[t].clamp(
                    min=self.loop_gate_min).to(hidden_states.dtype)
                hidden_states = h_before + g * update
            if aux is not None:
                aux_total = aux_total + aux
            if return_loop_states:
                loop_states.append(self.norm(hidden_states))"""
assert old in src
src = src.replace(old, new)

# ---- positional offset per round (RoPE-level) -----------------------------
old = """    def forward(self, input_ids, attention_mask=None, return_loop_states=False):
        \"\"\"Returns (hidden, aux_total) or (hidden, aux_total, loop_states).

        loop_states is a list with one entry per loop round (each
        [B, L, D]); it is the handle the cross-round probe needs.
        \"\"\"
        hidden_states = self.embed_tokens(input_ids)
        aux_total = torch.tensor(0.0, device=input_ids.device)
        loop_states = []"""
new = """    def forward(self, input_ids, attention_mask=None, return_loop_states=False):
        \"\"\"Returns (hidden, aux_total) or (hidden, aux_total, loop_states).

        loop_states is a list with one entry per loop round (each
        [B, L, D]); it is the handle the cross-round probe needs.
        \"\"\"
        hidden_states = self.embed_tokens(input_ids)
        aux_total = torch.tensor(0.0, device=input_ids.device)
        loop_states = []
        # Optional per-round positional offset: round t shifts its positions
        # by t*L... NO -- see the config knob loop_pos_offset: shifting by a
        # CONSTANT per round distinguishes rounds without leaving the trained
        # positional range.  Implemented in attention_causal via the
        # position_offset attribute set here per round.
        pos_offset = getattr(self, "_round_pos_offset", 0)"""
assert old in src
src = src.replace(old, new)

old = """        for t in range(self.num_loops):
            h_before = hidden_states"""
new = """        for t in range(self.num_loops):
            if self.loop_pos_offset:
                self._round_pos_offset = t
            h_before = hidden_states"""
assert old in src
src = src.replace(old, new)

io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("patched loop_lm")
