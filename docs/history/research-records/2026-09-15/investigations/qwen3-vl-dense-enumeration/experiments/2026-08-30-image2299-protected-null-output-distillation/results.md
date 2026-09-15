---
title: Image2299 protected-null output distillation results
type: investigation-result
unit_id: 2026-08-30-image2299-protected-null-output-distillation
status: stage_exhausted
evidence_status: mechanically_valid_bounded_negative
updated: 2026-08-30
---

# Result

The authoritative receipt is
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-30-image2299-protected-null-output-distillation/20260830T-image2299-protected-null-output-distillation-v1/receipt.json`.
Its independently verified SHA-256 is
`d41d0418b6f035a3124334107de86e436f5e94072d832042e59cb430e122e3da`.
Receipt status is `stage_exhausted`; stop reason is
`ProtectedNullHold: stage 1 full-vocabulary closure exhausted`.

The frozen controlled route is 370 tokens with SHA-256
`c89c6f9900303227cf781caf4a39dc203a09a53b0d319930754cda184529530e`.
The ordinary baseline route is 307 tokens with SHA-256
`5df0ac25aa871ddc0550298e70cd6012ce4dc3b6c6068b97dfa0cddca2f02a67`.
Capture found 12 positive states at positions
`301, 324, 328, 342, 346, 351, 355, 360, 364, 365, 367, 369`, with rank 12,
and 664 protected states. Raw argmax matched ordinary generation and the
zero-wrapper hidden/logit parity check was exact.

## Stage-1 frontier

All three attempted systems were feasible and numerically null on the
protected states. Normalized minimum-norm residual increased from
`0.4104686089` to `0.4144821808` (approximately `0.41047 -> 0.41448`), while
the protected maximum absolute correction stayed around `3.1e-13`, below the
frozen `1e-10` tolerance. The three solves accumulated 13, 14, and 15
constraints. Full-vocabulary recheck then exposed two active competitors for
the target token at position 301: competitor token `152583`, followed by
`152584`. Both closures were exhausted, so no warm candidate was evaluated
(`warm_candidates=0`) and no cold augmented-model greedy load occurred.

Runtime was 91.4330311343 seconds, one model load, three solves, one GPU,
world size 1, and peak CUDA reserved memory 10,815,012,864 bytes. The r32
surface, frozen surface, and resource bounds remained unchanged; no training,
checkpoint promotion, or existing-weight mutation occurred.

## Interpretation and claim boundary

Active-competitor rows created or continued a whack-a-mole frontier. This v1
closes exactly the two-closure target+competitor family at position 301; it is
not evidence that output-only capacity is exhausted in general. A target-only
successor later executed as a distinct phase; see [its result](../2026-08-30-image2299-target-only-protected-null-distillation/results.md).
No claim is made **by this unit**
for ordinary greedy 38-person output, transfer, general enumeration learning,
base-r32 improvement, or tie recovery.
