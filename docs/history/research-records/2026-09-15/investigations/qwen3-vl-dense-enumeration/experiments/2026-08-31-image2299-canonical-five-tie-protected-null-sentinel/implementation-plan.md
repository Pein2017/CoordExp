# Canonical five-tie protected-null sentinel implementation plan

- [x] Bind the parent receipt, parent residual, target library, exact anchor,
  five canonical rows, and static 415-token 46/46 route.
- [x] Load parent augmentation and prove exact zero-child anchor reproduction.
- [x] Capture positive/protected states for the canonical suffix and build the
  target-only child nullspace/constraint system.
- [x] Run one repaired minimum-norm solve, enforce child cap `9/8`, runtime
  full-vocabulary margins, and protected correction `<=1e-10`.
- [x] Compose one union payload and run one ordinary greedy warm candidate.
- [x] Persist only warm 46/46 success and require fresh-subprocess cold parity.
- [x] Seal the authoritative receipt, result, and final unit/index disposition.

Reuse the accepted parent helpers and payload format; add one runner and one
focused test only.
