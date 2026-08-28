---
title: Human-13 Owner-Credit N/K Factorial Results
type: investigation
role: results
authority: non_normative_research
unit_id: 2026-08-22-human13-owner-credit-nk-factorial
topic: qwen3-vl-dense-enumeration
status: complete
updated: 2026-08-28
---

# Human-13 Owner-Credit N/K Factorial Results

This document reproduces, verbatim in its factual and interpretive content,
`memories/notes/2026-08-24-human13-n13-k4-k8-factorial-result.md` ("Human-13
N13 K4/K8 factorial result", last verified 2026-08-24), reorganized into the
research-graph closeout shape. No number in this document was recomputed or
altered; it is copied from that note and from the corresponding paragraphs of
`memories/current.md` (both current as of 2026-08-24/2026-08-28). That note
remains available as the secondary, original-form provenance for this result.

## Disposition

`COMPLETE`. The N=13 K4/K8 matrix is valid scientific evidence, not a runtime
smoke: all eight cells completed four updates; all losses, gradients, and
parameter deltas were finite; exact adapter rollback passed 8/8; dual-RP
Source token reproduction passed 208/208. Independent replay passed 16,874
checks.

K8 is not supported over K4 under the required dual-RP criterion. K8 often
discovers additional reachable H owners, but it also increases owner
exchange. The small RP1.0 aggregate improvement is not replicate-robust; at
RP1.10, K8 is worse in net owners in 3/4 replicates and loses materially more
Source owners. Sampling changes are widespread owner churn rather than a
broad, stable coverage shift.

## Why this result exists

The preceding one-image standalone update proved that the simple
trajectory-credit plus Source-owner preservation loss could execute, but it
was a clean greedy null. N=4 then produced a small RP1.0 gain driven only by
image 1584. The N=13 matrix tested whether broader image aggregation reduces
gradient noise and whether K8 supplies a more useful direction than the exact
K4 prefix of the same sampling banks.

Frozen contrast: all 13 Human-13 images, paired K4/K8 banks, replicates
r0-r3, four sequential AdamW updates at LR `3e-6`, milestones `0/1/2/4`, and
clean greedy evaluation at RP1.0 and RP1.10. Matching is physical:
actual-pixel, class-aware, global one-to-one IoU >= 0.5.

## Frozen evidence

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

On-disk verification at close of the `reclaim-research-probes-lifecycle`
change (2026-08-28): the `v3` artifact root exists and contains `cells/`,
`logs/`, `preflight/`, and `receipts/` subdirectories. The predecessor N=1/N=4
root
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-21-human13-standalone-owner-credit-probe/`
also exists and contains `full-gpu1.json` (14.5K), `sentinel-gpu0.json`
(56.4K), and `multi-image-n4-k16-t4-gpu1.json` (373.4K).

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
were driven mainly by image 16228, with smaller positives on 10707, 13923,
and 14038; losses appeared on 1584, 2299, 2685, and 7511. RP1.10 degradation
was concentrated on 1584, 2685, and 4134, with G loss spread across more
images.

## Observed

- All eight (N13, K4/K8, replicate) cells completed four sequential AdamW
  updates from exact Source with a fresh optimizer each; all losses,
  gradients, and parameter deltas were finite.
- Exact adapter rollback passed 8/8 and exact Source dual-RP token
  reproduction passed 208/208; independent replay reproduced 16,874
  assertions.
- The milestone-4 paired K4/K8 dual-RP table above is the primary readout.
- Owner-level attribution: RP1.0 gains concentrate on image 16228 (with
  smaller gains on 10707, 13923, 14038) and losses on 1584, 2299, 2685, 7511;
  RP1.10 degradation concentrates on 1584, 2685, 4134, with G loss spread
  across more images.

## Supported

- More sampled trajectories (K8 vs K4) provide more reachable-owner
  information, but the current mean credit direction does not combine it
  safely.
- K8 magnifies both useful H credit and destructive interference with
  existing TP (true-positive Source owners).
- Image breadth N=13 is useful for exposing that conflict — it makes the
  K4-vs-K8 trade-off visible across a wider cohort than N=4 did.

## Ruled out

- K8 as a safe, replicate-robust improvement over K4 under the required
  dual-RP criterion: K8 is worse in net owners in 3/4 replicates at RP1.10
  and loses materially more Source owners there.
- N=13 image breadth alone denoising the update into a universally helpful
  direction: it does not, by itself, make the current credit direction
  coherent.
- The RP1.0 aggregate alone as a success signal: it is not
  replicate-robust and is contradicted by the RP1.10 result.

## Unresolved

- Whether image breadth alone (holding K fixed) denoises the update into a
  universally helpful direction is not established either way beyond what
  N=4-vs-N=13 already shows; it does not by itself resolve the conflict.
- The exact mechanism of gradient conflict between newly credited H owners
  and previously retained TP owners under RP1.10 greedy decoding.

## Interpretation and rejected continuation

More samples provide more reachable-owner information, but the current mean
credit direction does not combine it safely. K8 magnifies both useful H
credit and destructive interference with existing TP. Image breadth N=13 is
useful for exposing that conflict, but it does not by itself denoise the
update into a universally helpful direction.

Therefore:

- prefer K4 for the next bounded discovery experiment;
- do not spend more compute widening K under the unchanged loss;
- do not interpret the RP1.0 aggregate alone as success;
- retain dual-RP physical TP/FP/FN as the decision criterion;
- change credit fusion or preservation before further scaling.

## Not claimed

This result is restricted to the exact Human-13 cohort, Source checkpoint,
DoRA surface, LR, four-update horizon, sampling banks, and dual-RP
evaluation. It does not establish a general training recipe or broader-
distribution result.

## Next decision

Reproduced from `memories/current.md` ("Decision and next discriminator",
last verified 2026-08-24), as the standing authorized continuation state for
this route:

Do not enlarge K again for this objective. K4 is the safer discovery setting.
The next scientific successor should change the credit-fusion or
preservation objective, not add more trajectories or resume the retired
production runner. The main unresolved mechanism is gradient conflict: extra
sampled-owner credit can raise new H while moving probability away from
existing TP, particularly under RP1.10 greedy decoding.

A next unit, if the user authorizes it, should restart from the same Source
and compare a small number of K4 treatments that directly address this
conflict, for example owner-balanced/conflict-aware trajectory aggregation
and stronger physical-owner preservation. Dual-RP clean greedy physical
TP/FP/FN remains the decision owner. No follow-on GPU run is authorized
merely by this memory or by this records-only return.

## Provenance note

The probe's `contract.md` and CPU-only code (branch
`codex/human13-nk-factorial-probe`, tip `a904e3ae3`) were not returned to
`research-probes` by this lifecycle change; the frozen design is preserved in
[unit.md](unit.md) and the lane is retired at tag
`probe-final/human13-nk-factorial-probe`. Recovery of the source code, if
ever needed, is `git branch <name> probe-final/human13-nk-factorial-probe^{}`.
