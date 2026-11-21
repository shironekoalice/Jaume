"""Break the contraction: round-dependent computation for the looped LM.

The 2026-11-07 probe finding
----------------------------
The stage-1 loop failed the preregistered same-wall-clock criterion (C=5.93
vs base=4.94), and the interior probe showed why.  Round-to-round distance
decays geometrically (0.55, 0.36, 0.21, 0.14, 0.10, ...), and the
consecutive-round principal subspace overlap converges to 0.99: at this
scale the loop is approximately a CONTRACTION MAPPING.  The extra rounds
converge to a fixed point instead of computing anything new, so spending
4x the wall-clock on a converged update is worth less than spending it on
more data.

What this module adds
---------------------
Three minimal interventions, each targeting a different way to stop the
rounds from being repetitions of the same map:

  gate    a learned per-round scalar gate on the residual update of every
          layer.  Round t applies scale[t] to what the layer changes.
          If the contraction is a property of the weights, letting the
          model re-weight each round gives it the tools to NOT converge.
          Cheapest of the three; no extra forward passes.

  offset  per-round loop-position offsets: round t sees positions shifted
          by t, so RoPE distinguishes the rounds.  The 'increment' mode
          already existed but shifted by the whole context length; a
          per-round *constant* offset distinguishes rounds without pushing
          positions out of the trained range.

  hmm     (not a mode) the honest baseline is gate: it is the smallest
          change that makes 'round 2' a different function from 'round 1'.

The gate is zero-initialised per round with the FIRST round forced to 1.0,
so at initialisation the gated model is exactly the ungated one -- the
intervention starts as a no-op and can only matter if training moves it.
"""
