import io
p = "code/repo/jaume/modeling/loop_lm.py"
src = io.open(p, encoding="utf-8").read()
old = """        if self.loop_gate:
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
            self.loop_gate_param = None"""
new = """        if self.loop_gate:
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
            self.loop_gate_param = None"""
assert old in src
src = src.replace(old, new)

old = """            if self.loop_gate_param is not None and t > 0:
                # scale the ROUND'S UPDATE, not the state: this keeps the
                # residual stream intact and only re-weights what this round
                # contributes on top of it.
                update = hidden_states - h_before
                g = self.loop_gate_param[t].clamp(
                    min=self.loop_gate_min).to(hidden_states.dtype)
                hidden_states = h_before + g * update"""
new = """            if self.loop_gate_param is not None and t > 0:
                # scale the ROUND'S UPDATE, not the state: this keeps the
                # residual stream intact and only re-weights what this round
                # contributes on top of it.  loop_gate_param[t-1] is round t
                # because round 0 is pinned and not stored.
                update = hidden_states - h_before
                g = self.loop_gate_param[t - 1].clamp(
                    min=self.loop_gate_min).to(hidden_states.dtype)
                hidden_states = h_before + g * update"""
assert old in src
src = src.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("patched: pin by exclusion")
