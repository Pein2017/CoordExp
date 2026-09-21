# KL10 dev112 concentration diagnosis

Status: candidate advisory diagnostic. Native raw-text parsing and category-consistent global one-to-one owner scoring were independently replayed on all112 Source/candidate pairs (224 parses and224 scores). No model was loaded and no training or arm was selected. All prior artifacts remain unchanged.

## Outcome and concentration

Source -> KL10: TP50 558 -> 576, FP50 343 -> 466, valid predictions 901 -> 1042, strict later pixel-IoU>.95 repeats 68 -> 109. F1@50 0.655314 -> 0.624729. Every output reaches natural EOS; no cap.

FP excess is concentrated, but neither solely one runaway image nor a uniform collapse across images. Exact FP decomposition is repeated-FP +41 and non-repeat-FP +82. Net prediction increase +141 is an output-count change, not141 newly verified physical instances.

Concentration below uses gross positive increases, avoiding cancellation by improved images.

| Quantity | Positive / negative / net | Images worse | Top1 | Top3 | Top5 | Top10 |
|---|---:|---:|---:|---:|---:|---:|
| FP50 | 134 / -11 / 123 | 32 | 36.6% | 54.5% | 65.7% | 78.4% |
| Strict repeats | 45 / -4 / 41 | 5 | 75.6% | 93.3% | 100.0% | 100.0% |
| Non-repeat FP50 | 94 / -12 / 82 | 32 | 16.0% | 39.4% | 51.1% | 70.2% |
| Lost IoU50 owners | 10 / 0 / 10 | 9 | 20.0% | 40.0% | 60.0% | 100.0% |

Top FP contributions:226097 +49;167952 +15;131580 +9;73724 +8;460339 +7. Their combined88 accounts for71.5% of net FP increase, or65.7% of gross-positive increase. Image226097 alone adds34 repeats; only five images increase strict repeats. FP increases on32/112 images, is unchanged on72, and decreases on8.

As a descriptive sensitivity only, excluding the same top-FP images from BOTH models leaves these F1 differences (not corrected evaluation or a proposal to filter development):

| Excluded top images | Remaining images | KL10 minus Source F1@50 |
|---|---:|---:|
| 1 | 111 | -0.015663 |
| 10 | 102 | +0.005590 |
| 3 | 109 | -0.009837 |
| 5 | 107 | -0.002004 |

## Parser repair does not explain the FP increase

| Drop-count group | Images | Delta valid predictions | Delta TP50 | Delta FP50 | Delta strict repeats | Delta tokens |
|---|---:|---:|---:|---:|---:|---:|
| improved | 4 | +11 | +8 | +3 | -4 | -328 |
| unchanged | 106 | +128 | +11 | +117 | +45 | +1238 |
| worse | 2 | +2 | -1 | +3 | +0 | +39 |

Source drops51 -> KL10 drops4. Four drop-improved images remove49 drops; two others add2. The repaired-image group contributes only net FP+3 (2.4% of total+123), while drop-unchanged images contribute+117. Image39654 accounts for46 removed drops, TP+5/FP+2, repeats-4 and333 fewer tokens. This is not evidence that46 malformed spans became46 correct objects: the generated histories changed.

## Owner and category checks

| IoU | Owners gained | Owners lost | Net |
|---|---:|---:|---:|
| 50 | 28 | 10 | +18 |
| 60 | 18 | 16 | +2 |
| 80 | 18 | 14 | +4 |

The10 IoU50 losses occur on9 images (image392606 loses2). Every lost owner lacks a candidate same-category box meeting0.5; none is a global-assignment-only loss. Owner1331687 on460339 falls from best IoU0.716 to0.460; other losses include much larger support changes. One IoU50 gain,511299:1801594, already had above-threshold Source direct support and therefore is not clean evidence of newly appearing instance support. All claims retain the global matching result rather than reclassifying it.

Net FP categories: spoons+48, bottles+18, people+17, books+16, carrots+14. Repeated-FP changes are spoons+34, books+11 and bananas-4. The largest loop is not literal repetition of either trained x1 label: frequent spoon x1 values include479 and486.

## Exactly three inspected Source/candidate comparisons

- **226097, kitchen:** Source already emits61 spoon rows; KL10 emits107. Exact repeated spoon boxes are concentrated near the window/counter, with additional spoon boxes elsewhere. This is amplification of an existing local failure, not proof that all49 additional predictions are hallucinations.
- **167952, food:** carrot rows12 ->26 with no strict repeat increase; TP50 stays10 through one gain and one loss. The original image contains many overlapping thin vegetable pieces. Additional non-strict boxes are not automatically distinct owners, but uncertain extent/instance granularity also prevents treating them all as physically false.
- **460339, snowboarding crowd:** person rows17 ->23, no strict repeats, TP50 4 ->3. The image visibly contains a dense crowd beyond the13 annotated people. Some extra predictions may be real unannotated people; none of the six additions is individually certified here. The specific annotated loss still exists under the frozen IoU metric.

The native visual renderer uses its own greedy overlay matcher; caption duplicate-pair hints are not our strict-later-repeat count or research global matching. Only the independently recomputed JSON owns the reported metrics. No GT or prediction geometry was edited.

## Implication for a possible support expansion

The cheapest useful distinction is coverage of long, dense, same-category Source histories versus preservation only around the two trained examples. Current evidence makes an exclusively repeat-focused remedy incomplete: two thirds of net FP excess is outside the strict-repeat definition, and owner losses are distributed across9 images. Broader Source-train-only preservation support can test this without selecting failed development images or introducing unknown-as-negative targets. Judge its effects separately on the concentrated repeat tail, non-repeat FP, owner gain/loss and parser repair—not merely on output length.

The strongest counterhypothesis is that some expanded enumeration is physically useful but annotation-relative precision penalizes incomplete labels or uncertain extents. A support expansion that merely suppresses more output could improve F1 while sacrificing useful recall. These observational counts do not choose an arm, identify a ledger, or prove physical precision deterioration.

First18 context differs: train2 improves from TP16/FP2 to18/2; guard16 preserves TP56 while FP falls146 ->135. That small panel did not predict the dev112 tail. The dev112 diagnosis is development evidence, not untouched confirmation.

## Reproduction and evidence

- `analysis.json`: every per-image contribution, source/candidate scores, repeat-FP decomposition, owner-loss best IoUs, concentrations and input SHA256 values.
- `reproduce.md`: one bounded CPU query; also recreates or verifies the same three unchanged visual projections.
- `visual/comparison/manifest.json`: ordered three-image Source/KL10 comparison provenance.

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from pathlib import Path
p=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/dev112-diagnosis/reproduce.md')
code=p.read_text().split("```python\n",1)[1].split("\n```",1)[0]
exec(compile(code,str(p),"exec"),{"__file__":str(p),"__name__":"__diagnosis__"})
PY
```
