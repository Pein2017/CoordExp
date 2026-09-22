---
title: Weak-correction early-dose result
description: Neither saved early checkpoint meets the fixed confirmation gate.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-09-07-coco-weak-correction-dose
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-07
---

# Earlier stopping did not recover the desired trade-off

**Round1 complete; neither dose eligible; round2 not launched.**
The [unit](unit.md) owns the fixed rule and two-round ceiling. The compact
[result JSON](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-weak-correction-dose/dose-v1/round1-results.json)
owns per-image matching, paired sets, uncertainty and output monitors.

## Same 512-image dose-selection panel / 3,759 annotated owners

Coverage cells are matched owner counts, not percentages. Source and Rweak64
are reused from the preceding experiment; only Rweak16/32 required new decode.

| Checkpoint | IoU50 | IoU60 | IoU80 | Strict duplicate candidates | Duplicate-affected images | Invalid | Capped images |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source | 2225 | 2055 | 1534 | 21 | 8 | 2 | 11 |
| Rweak16 | 2240 | 2065 | 1517 | 27 | 11 | 1 | 15 |
| Rweak32 | 2250 | 2070 | 1519 | 343 | 12 | 1 | 13 |
| Rweak64 | 2310 | 2102 | 1540 | 276 | 9 | 2 | 8 |

| Early dose vs Source | IoU50 gains / losses / net | Image-bootstrap 95% total-equivalent interval | IoU60 net | IoU80 net |
|---|---|---|---:|---:|
| 16 | 94 / 79 / +15 | [-18,46] | +10 | -17 |
| 32 | 116 / 91 / +25 | [-16,69] | +15 | -15 |

Both early doses fail the predeclared rule: IoU80 decreases, strict duplicates
increase, and more images hit the decode cap. Positive IoU50 point estimates
alone are insufficient, and their paired image intervals include zero.

## Interpretation and decision

- Earlier stopping at these two tested doses did not preserve the 64-step
  coverage benefit while restoring Source-level stability. This does not prove
  that every possible intermediate dose would fail; no further checkpoint
  search is justified by the current bounded result.
- Repetition is not confined to the final checkpoint. At step32, image566923
  contributes 313 of 343 strict duplicate candidates, versus 249 of 276 at
  step64. At step16, no comparable single-image loop dominates (largest count
  11), but localization and cap outcomes still fail the joint criterion.
- These are geometry-derived duplicate candidates; unmatched predictions are
  not labeled hallucinations. No repetitive image or capped output was removed.
- This reused512 panel is now a dose-selection set, not a fresh confirmation
  set. The separately frozen confirmation512 / 3,886-owner panel was not
  decoded because neither candidate qualified. There is no confirmation result.

## Execution and proportionate verification

Rweak16 and Rweak32 are existing checkpoints from the same training trajectory;
no new training occurred. They decoded concurrently on four GPUs each with
native greedy FP32 SDPA, RP1.0, cap3084 and per-device batch4. The actual128
global four-image groups were checked to be identical to the reused two-rank
Source/64 grouping; only rank placement changed.

Both new panel exits are0, spanning `17:31:53Z–18:21:23Z` (49m30s). The saved
analysis checks completed summaries, checkpoint arm/step, decoder settings,
all512 raw rows against the input, owner matching and the fixed eligibility
predicate. Lead ran it successfully and checked every stored gain/loss/net
count against its owner set. No extra hash audit or independent matcher was
introduced. The driver has exited.

**STOP after one round:** no second-round GPU work, other dose, retraining,
decoder change, promotion, commit, push or archive. Confirmation inputs remain
available but do not authorize a later run.
