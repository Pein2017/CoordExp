# Current Project Memory

Last verified for the Human-13 owner-credit route: 2026-08-24.

## Authority boundary

This file is continuity, not scientific authority. For the latest executed
route, the immutable artifacts and their bound commits own the evidence:

- N=1 standalone result:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-21-human13-standalone-owner-credit-probe/full-gpu1.json`
  at commit `32bc918d468baa41fbe218dd81998f86eb5eb226`
- N=4/K16/T4 standalone result:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-21-human13-standalone-owner-credit-probe/multi-image-n4-k16-t4-gpu1.json`
  at commit `b36216f10b89f7a585b2c08dcaed6fe997b0ff9d`
- N=13, paired K4/K8, four-replicate matrix:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-23-human13-n13-k4k8-corrected-geometry-probe/v3`
  at correction commit `a904e3ae38405cf018bccbc176c3497be29313c9`

The interpretation and integrity boundary are owned by the research unit
[2026-08-22 Human-13 Owner-Credit N/K Factorial](../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-22-human13-owner-credit-nk-factorial/results.md)
(secondary provenance:
`memories/notes/2026-08-24-human13-n13-k4-k8-factorial-result.md`).
Earlier formal Human-13 units remain evidence for their own bounded questions,
but they do not override this newer standalone route.

## Current research question

Can information from multiple sampled trajectories be combined into a scalable
training signal that improves clean greedy physical-owner coverage, while
retaining owners already found by the Source model?

The current standalone treatment is deliberately simple: sampling-derived
trajectory credit plus masked CE preservation of current Source owners, one
fresh AdamW instance per cell, LR `3e-6`, milestones `0/1/2/4`, and dual
greedy evaluation at repetition penalty `1.0` and `1.10`. Physical owners are
scored with actual-pixel, class-aware, global one-to-one IoU >= 0.5 matching.

## Latest durable conclusion

The N=13 K4/K8 matrix is valid scientific evidence, not a runtime smoke. All
eight cells completed four updates; all losses, gradients, and parameter deltas
were finite; exact adapter rollback passed 8/8; dual-RP Source token
reproduction passed 208/208. Independent replay passed 16,874 checks.

At milestone 4, aggregated across four paired replicates:

| Decode | K4 physical TP/FP/FN | K8 physical TP/FP/FN | K8 minus K4 |
|---|---:|---:|---:|
| RP 1.0 | 670 / 323 / 898 | 672 / 313 / 896 | TP +2, FP -10, FN -2 |
| RP 1.10 | 622 / 308 / 946 | 603 / 324 / 965 | TP -19, FP +16, FN +19 |

K8 is therefore not supported over K4 under the required dual-RP criterion.
K8 often discovers additional reachable H owners, but it also increases owner
exchange. The small RP1.0 aggregate improvement is not replicate-robust; at
RP1.10, K8 is worse in net owners in 3/4 replicates and loses materially more
Source owners. Sampling changes are widespread owner churn rather than a broad,
stable coverage shift.

This narrows the earlier direction:

- N=1/K16/T1 was a clean null with exact G preservation.
- N=4/K16/T1 first crossed a greedy boundary, but the positive evidence was
  localized to image 1584 and did not establish shared cross-image learning.
- N=13 shows that greater image breadth alone does not make the current credit
  direction coherent, and increasing K from 4 to 8 does not solve it.

## Decision and next discriminator

Do not enlarge K again for this objective. K4 is the safer discovery setting.
The next scientific successor should change the credit-fusion or preservation
objective, not add more trajectories or resume the retired production runner.
The main unresolved mechanism is gradient conflict: extra sampled-owner credit
can raise new H while moving probability away from existing TP, particularly
under RP1.10 greedy decoding.

A next unit, if the user authorizes it, should restart from the same Source and
compare a small number of K4 treatments that directly address this conflict,
for example owner-balanced/conflict-aware trajectory aggregation and stronger
physical-owner preservation. Dual-RP clean greedy physical TP/FP/FN remains the
decision owner. No follow-on GPU run is authorized merely by this memory.

## Retired and retained context

- The long all-HF production-vertical route repeatedly exposed infrastructure
  ownership and admission seams before science. It is superseded for this
  question by the standalone probe and should not be repaired or resumed.
- The 2026-08-13 on-policy first-bottleneck study remains evidence that native H
  rows are trainable but fixed-dose updates systematically exchange owners; see
  `memories/notes/2026-08-13-human13-on-policy-first-bottleneck-result.md`.
- The 2026-08-12 static Human-13 screen remains evidence that one-exposure DoRA
  can increase same-panel recall but repeated suffix CE develops owner exchange
  and burden; see
  `memories/notes/2026-08-12-human13-k-union-execution-result.md`.
- The 2026-08-14 scalable-trajectory note preserves the original motivation;
  its optimizer details are historical hypotheses, not current authorization.

## Operational continuation

The v3 run root is complete and immutable; no cells remain running. The
producer-PID plus terminal-receipt wake monitor worked for the multi-hour smoke
and final seven-cell matrix. A native depth-2 subtask transcribed two checksums
incorrectly, so identity tables must be emitted mechanically or independently
replayed by the lead; the final scientific calculations were independently
verified.
