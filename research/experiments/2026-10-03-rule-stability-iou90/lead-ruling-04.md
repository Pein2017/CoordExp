# Completed A/B contrast: accepted mixed evidence

The lead accepts the finite primary package as technically complete. Both arms
ran 16 updates from independent copies of the original anchor and fresh AdamW;
each retained all 17 greedy versions and checkpoints 0/1/4/8/16. Native A,
native B and the single saved comparison exited 0. All owned jobs are settled,
and the worker explicitly released the frozen source holder. The final saved
consumer check is `outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/lead-primary-consumer-check-01.json`.
It verifies the 18-image/570-owner denominator, per-image totals, exact endpoint
retained/gained/lost sets, B-minus-A differences, and terminal receipts without
new model work, rescoring, or repeated readback. [Results](results.md) retain
all versions, runtime, numerical limits, exact source and raw evidence.

| Endpoint | Anchor | A: positive + geometry | B: A + sampled duplicate credit |
|---|---:|---:|---:|
| Matched annotations / 570 | 255 | 286 | 288 |
| Strict IoU > .9 duplicate events | 5 | 562 | 375 |
| Geometry-invalid rows | 4 | 125 | 10 |
| Capped images | 0 | 2 | 1 |
| Longest duplicate burst | 2 | 153 | 174 |

A retains 223 baseline owners, gains 63 and loses 32; B retains 230, gains 58
and loses 25. B improves several endpoint burdens relative to A, but both
substantially increase repetition and lose previously matched owners. B's
longest burst worsens. No fixed coverage/retention threshold is introduced,
no checkpoint is promoted, and these exposed-label results do not establish
physical-FN recovery or generalization.

B's samples contain only 2 duplicate events while its greedy training references
contain 5187. The resulting credit can reinforce sampled alternatives to a
repetitive greedy mode; sparse sampled penalties and a large greedy baseline
also permit high-variance updates. This is an explanatory hypothesis, not an
identified cause or proof that the objective is ineffective. Cached/full replay
remains numerically approximate; the active policy gap reaches .953699 and
fixed-advantage loss differences do not bound gradient bias. Full-horizon
sampled backward remains unmeasured (largest actual B sample: 1082 actions).

Image 351017 ends at 1/49 matches and the 3084-action cap in both arms. In A,
the first person row is unchanged, but the next category switches from person
to bottle at the first update, before any repetitive history. The sorted teacher
sequence also asks for bottle next, but follows a first-person box differing
by three coordinate tokens. This supports a cheap new diagnostic: saved A0/A16
crossed with generated versus same-owner GT first-row history. It can distinguish
conditional prefix sensitivity from failure under both histories, not prove
natural recovery. The user's autonomous-next-round instruction authorizes a
separate bounded unit; this completed unit receives no new calls or extension.
