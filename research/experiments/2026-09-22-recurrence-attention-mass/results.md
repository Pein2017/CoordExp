# Accepted live-query mass/profile counterexample

2026-09-22. Scientific status: the advance mass-following global-winner
prediction is rejected in both fixed cases. Technical status: accepted.
The cells below name MASS then PROFILE; OO maps to the predecessor pure-phase
NO, with new pre-keys and native V throughout.

| Case | Cell | Global coordinate winner | Top-two gap | Fixed old-minus-new margin |
|---|---|---:|---:|---:|
| val | NN | 999 | 0.006084442 | -0.006084442 |
| val | OO | 38 | 0.331842422 | 0.756042480 |
| val | ON | 38 | 0.342863083 | 0.653097153 |
| val | NO | 38 | 0.310939789 | 0.336606979 |
| train | NN | 348 | 0.028825760 | -1.098085403 |
| train | OO | 591 | 0.011472702 | 0.670390129 |
| train | ON | 592 | 0.018914223 | 0.686938286 |
| train | NO | 599 | 0.226969719 | -0.435474396 |

The fixed margins are val z38-z999 and train z591-z348. Train350 remains in
the full-vector readback as the failed earlier restoration prediction.

On val, either old mass with native profile or native mass with old profile
changes the native999 winner to38. Both native components are required for999
among these four tested conditions. This is compatible with ordinary margin
shifts across a near tie; it does not establish a special conjunction circuit.
On train, mass strongly changes the named591-versus348 margin, yet the mixed
conditions select592 and599. A two-candidate margin cannot stand in for the
global decision. The descriptive larger mass-response norms did not establish
winner sufficiency.

| Case | Mass effect / new profile | Mass effect / old profile | Profile effect / new mass | Profile effect / old mass | Interaction |
|---|---:|---:|---:|---:|---:|
| val | 0.659181595 | 0.419435501 | 0.342691422 | 0.102945328 | -0.239746094 |
| train | 1.785023689 | 1.105864525 | 0.662611008 | -0.016548157 | -0.679159164 |

Effects use old minus new for the named component. The strongest local account
is phase-dependent redistribution through both row weight and within-row
allocation, followed by adaptive current-state propagation. This is a
head/query-specific attention-rule intervention, not one scalar gate, matched
normalized probabilities across different live queries, or a natural mediation
fraction. Static common-shift RoPE invariance also prevents interpreting this
one-row phase intervention as evidence of an absolute-position clock.

## Qualification and evidence

Root independently checked79 source/artifact bindings, all eight full-vocabulary
vectors,28 layers per cell, actual masks/phases/native V and companion invariance,
and FP64 QK/row-logsumexp/profile formulas. All four diagonal vectors reproduce
the accepted NN/NO references exactly. Maximum independent score error is
5.28e-6; live logsumexp error5.96e-6; profile identity error3.56e-15. Maximum
actual SDPA head-output reconstruction error1.3352e-4 remains below the frozen
2e-4 bound. The installed-Qwen caller selfcheck was rerun by root successfully.

Ten model/two vision calls,54.2307s model-package execution,61.3499s elapsed,
14,640,218,112 bytes peak reserved. The GPU process is terminal. No retry or
substituted case. A root CPU inspection initially assumed a list where the
train readback uses a layer-index dictionary; correcting that indexing required
no model rerun and changed no artifact.

- [Frozen protocol](unit.md)
- [Original result](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-attention-mass/attempt-001/result.json)
- [Independent readback including absolute probabilities and full-vocabulary competitors](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-attention-mass/lead-checks/mass-readback.json)
- [Lead acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-attention-mass/lead-acceptance.json)

The next distinct discriminator is whether the preceding row's whole contextual
K/V cache, moved to the correct new-row phase, can replace freshly computed
row-specific K/V. It does not revisit the rejected mass-only prediction.
