# Human-13 N13 K4/K8 factorial result

Last verified: 2026-08-24.

## Why this result exists

The preceding one-image standalone update proved that the simple
trajectory-credit plus Source-owner preservation loss could execute, but it was
a clean greedy null. N=4 then produced a small RP1.0 gain driven only by image
1584. The N=13 matrix tested whether broader image aggregation reduces gradient
noise and whether K8 supplies a more useful direction than the exact K4 prefix
of the same sampling banks.

Frozen contrast: all 13 Human-13 images, paired K4/K8 banks, replicates r0-r3,
four sequential AdamW updates at LR `3e-6`, milestones `0/1/2/4`, and clean
greedy evaluation at RP1.0 and RP1.10. Matching is physical: actual-pixel,
class-aware, global one-to-one IoU >= 0.5.

## Evidence owner

Immutable root:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-23-human13-n13-k4k8-corrected-geometry-probe/v3`

Bound correction commit:

`a904e3ae38405cf018bccbc176c3497be29313c9`

Rebase file SHA256:

`57697ef4c234149484ca618fde97a9b7445d11e0a0b2d6b23e1ef1bdfe4356b5`

Each of the eight cell directories contains its canonical `result.json` and
terminal receipt. The independent final audit replayed 16,874 assertions:
eight successful cells, 32 finite optimizer updates, persistent per-cell
AdamW step order, exact rollback 8/8, and exact Source dual-RP token
reproduction 208/208.

## Decision-bearing result

Milestone 4, aggregated over four paired replicates:

| Decode | K4 TP/FP/FN | K8 TP/FP/FN | Paired aggregate delta |
|---|---:|---:|---:|
| RP1.0 | 670 / 323 / 898 | 672 / 313 / 896 | TP +2, FP -10, FN -2 |
| RP1.10 | 622 / 308 / 946 | 603 / 324 / 965 | TP -19, FP +16, FN +19 |

At RP1.0, the K8 net-owner outcome was better/equal/worse in 1/2/1
replicates. At RP1.10 it was better/equal/worse in 1/0/3 replicates. K8 added
some H in most replicates, but RP1.10 Source-owner loss worsened in 4/4.

The effect is owner churn, not a stable shared improvement. Final RP1.0 gains
were driven mainly by image 16228, with smaller positives on 10707, 13923, and
14038; losses appeared on 1584, 2299, 2685, and 7511. RP1.10 degradation was
concentrated on 1584, 2685, and 4134, with G loss spread across more images.

## Interpretation and rejected continuation

More samples provide more reachable-owner information, but the current mean
credit direction does not combine it safely. K8 magnifies both useful H credit
and destructive interference with existing TP. Image breadth N=13 is useful
for exposing that conflict, but it does not by itself denoise the update into a
universally helpful direction.

Therefore:

- prefer K4 for the next bounded discovery experiment;
- do not spend more compute widening K under the unchanged loss;
- do not interpret the RP1.0 aggregate alone as success;
- retain dual-RP physical TP/FP/FN as the decision criterion;
- change credit fusion or preservation before further scaling.

This result is restricted to the exact Human-13 cohort, Source checkpoint,
DoRA surface, LR, four-update horizon, sampling banks, and dual-RP evaluation.
It does not establish a general training recipe or broader-distribution result.
