# K-trajectory exact-policy parity closure

The Human-13 K-trajectory RP-crossover unit closed at its v5 admission gate.
Authoritative evidence and interpretation live in
`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-14-human13-k-trajectory-rp-crossover-screen/results.md`.

Continuity only:

- fp32/SDPA exact-history replay reduced the v4 cross-surface mismatch by about
  26x in mean error, proving the prior BF16/FA2 surface dominated the drift;
- the remaining vLLM fp32 versus HF fp32/SDPA mismatch still failed the frozen
  exact-policy gate at `rp=1.0`;
- no optimizer or owner outcome exists, and the matrix is retired;
- do not resume by widening tolerance or by running `rp=1.10` in this unit;
- the next research fork is user-owned: one shared sampler/gradient surface, or
  an explicitly approximate/off-policy objective with its bias made visible.
