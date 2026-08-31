# Protected-null output distillation implementation plan

- [x] Hash-bind the terminal witness, r32 checkpoint/readback, prompt, image,
  surfaces, five intervention blocks, route, and owner ledger.
- [x] Capture aligned output-head hidden states and raw logits for the frozen
  ordinary and controlled routes; verify zero-residual parity.
- [x] Build the protected-state projector, positive basis, selected-token
  linear constraints, minimum-norm solver, cap, and frozen FP64 numerical-null
  diagnostics (`max_abs <= 1e-10`, protected-matrix hash, warm/cold replay).
- [x] Install a persistent sparse output residual module and serialize its
  payload plus identity metadata without mutating existing model weights.
- [x] Execute the first staged solve with at most two active-competitor or
  earliest-drift closures per stage and ordinary greedy evaluation only.
- [ ] On first warm match-level success, load the payload in a fresh process
  and require warm/cold route, ledger, EOS, residual, and frozen-surface parity.
- [x] Seal one authoritative receipt, results, and final unit/index disposition.

The implemented mechanics are receipt binding, protected-null projection,
three feasible minimum-norm solves, persistent residual serialization, and
full-vocabulary recheck. Cold success was not reached: stage 1 stopped after
two exhausted closures at position 301, with zero warm candidates. A
target-only cold-success attempt later executed as a separately owned phase;
see [Target-Only Protected-Null Distillation](../2026-08-30-image2299-target-only-protected-null-distillation/results.md).
No general inference API or reusable architecture change
is required for this single-image research unit.
