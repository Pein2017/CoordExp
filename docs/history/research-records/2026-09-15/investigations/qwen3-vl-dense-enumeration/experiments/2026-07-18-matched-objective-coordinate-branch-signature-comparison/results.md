---
title: Matched-Objective Coordinate-Branch Signature Comparison Results
description: Case-level evidence, including an exact current-sampler replication, that the Gaussian-smoothed coordinate and Ranked Probability Score checkpoint changes several coordinate modes while some native autoregressive geometry failures remain shared.
type: investigation-result
role: evidence-summary
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-07-18-matched-objective-coordinate-branch-signature-comparison
topic: qwen3-vl-dense-enumeration
status: completed
evidence_status: bounded_case_level_decision_grade
updated: 2026-07-18
---

# Matched-Objective Coordinate-Branch Signature Comparison Results

## Decision

Do not launch the proposed complete-row consistency training screen on top of
the Gaussian-smoothed coordinate and Ranked Probability Score checkpoint.

For the reviewed bowl, umbrella, and bus controls, the scoring checkpoint has
more influence on the coordinate mode than the checkpoint that generated the
prefix. The pure cross-entropy checkpoint is also better on the already
available matched validation-200 benchmark. The strongest previously observed
same-owner truncation is therefore not adequate evidence of an
objective-independent defect that needs another loss.

Use the pure cross-entropy plus token-type-gate checkpoint as the primary
baseline for the next mechanism study. Keep the Gaussian-smoothed and Ranked
Probability Score checkpoint as an ablation that demonstrates an objective
tradeoff, not as the default checkpoint to repair.

## What was executed

The comparison used the same six reviewed images, the same sixteen sampling
seeds per image, temperature 0.4, top-p 0.95, repetition penalty 1.0, and a
512-token maximum generation horizon.

- Pure cross-entropy native trajectory root:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-matched-objective-coordinate-branch-signature-comparison/native-pure-cross-entropy-bfloat16/`
- Gaussian and Ranked Probability Score exact current-sampler trajectory root:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-matched-objective-coordinate-branch-signature-comparison/native-gaussian-current-sampler-bfloat16/`
- Exact current-sampler comparison summary:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-matched-objective-coordinate-branch-signature-comparison/native-current-sampler-objective-comparison-v1.json`
- Joined native inventory:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-matched-objective-coordinate-branch-signature-comparison/native-trajectory-inventory-v1.json`
- Enlarged row-zero overlays:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-matched-objective-coordinate-branch-signature-comparison/native-row0-crop-overlays-v1/`
- Exact current-sampler row-zero overlays:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-matched-objective-coordinate-branch-signature-comparison/native-current-sampler-row0-overlays-v1/`
- Pure-native admitted cases: `pure-cross-entropy-native-branch-cases.json`
- Pure-native prefixes scored by pure cross-entropy:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-matched-objective-coordinate-branch-signature-comparison/native-pure-source-scored-by-pure-bfloat16-v1/results.json`
- Pure-native prefixes scored by Gaussian smoothing plus Ranked Probability
  Score:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-matched-objective-coordinate-branch-signature-comparison/native-pure-source-scored-by-gaussian-bfloat16-v1/results.json`
- Full-model float32 controls are stored in the adjacent
  `native-pure-source-scored-by-*-float32-v1/` directories.

The scorer's conclusion-critical contract passed an independent read-only
audit. Exact prompt tokens, image media hashes, selected native rows, coordinate
token positions, branch prefixes, and explicit counterfactual model
composition were checked. The audit found no conclusion-blocking defect.

## Denominators before selected examples

Every source contributed exactly sixteen naturally closed trajectories per
image. The Gaussian replication used the same current sampler implementation
and exact ordered seed list as pure cross-entropy. Its first-seed replay check
passed for every image. The table reports all valid rows of the reviewed
category, not only rows close to the official annotation.

| Image | Reviewed category | Pure cross-entropy rows | Gaussian and Ranked Probability Score rows | Gaussian dropped rows of this category |
|---:|---|---:|---:|---:|
| 15335 | person | 102 | 114 | 0 |
| 16451 | umbrella | 16 | 16 | 0 |
| 7574 | bowl | 42 | 39 | 0 |
| 12670 | person | 295 | 284 | 0 |
| 6471 | person | 215 | 215 | 0 |
| 15517 | bus | 169 | 162 | 0 |

Across all categories on these six images, pure cross-entropy produced 1,460
valid rows and the exact current-sampler Gaussian replication produced 1,360.
Neither run dropped a prediction. These are exact execution-contract totals,
but they remain selected-case behavior rather than a population estimate.
The noncanonical two-seed smoke file `image-12670-test.json` in the Gaussian
artifact directory is excluded; only the six canonical `image-<id>.json`
artifacts enter the comparison summary.

## Native row-zero behavior

The crop overlays show a consistent but bounded pattern.

### Image 7574, white bowl

Pure cross-entropy keeps the reviewed bowl's lower coordinate within bins
181-193, with median 187 and population standard deviation 2.63. The exact
current-sampler Gaussian replication spans bins 157-196, with median 171 and
standard deviation 11.56, while `x1`, `y1`, and `x2` remain comparatively
tight. Several boxes are visibly truncated. Mean intersection over union with
the accepted reference is 0.615 for pure cross-entropy and 0.346 for the
Gaussian checkpoint. This endpoint-specific difference survives removal of
the historical sampler confound.

### Image 12670, red-shirt person

Pure cross-entropy produces thirteen row-zero person boxes tightly attached to
the red-shirt person; mean intersection over union is 0.915. All sixteen exact
Gaussian row-zero predictions say `person` and most bind the same red-shirt
person, but one clear owner switch or partial-person trajectory reaches the
people above the target. That single event expands `y1` from 0 to 112 and `y2`
from 158 to 568. Mean intersection over union remains 0.862. This is evidence
of an occasional objective-associated branch failure, not a claim that the
Gaussian checkpoint generally lacks person binding.

### Image 15517, center-right bus

Pure cross-entropy keeps row-zero `x1` within bins 503-532, with mean 518.4.
The exact current-sampler Gaussian replication is also coherent but uses a
systematically earlier left boundary: bins 479-506, with mean 493.3. Its other
three coordinates remain comparatively tight. Mean intersection over union
with the official box is 0.584 versus 0.455. Visual review shows a stable
extent disagreement around the same visible bus, not the broad neighboring-bus
mixture seen in the historical sampler run. The checkpoint-specific `x1` mode
is real; the earlier claim of broad native bus owner instability was partly a
sampler or runtime-path effect and is not retained.

### Image 16451, umbrella

Both checkpoints describe the same umbrella in all sixteen row-zero samples.
Gaussian predictions have better overlap with the official canopy annotation,
while many pure cross-entropy boxes include more of the pole and lower physical
extent. Mean official-box intersection over union is 0.627 versus 0.530. This
is an ontology-sensitive counterexample: smoothing can move predictions toward
an official boundary even when that boundary is not the only defensible
physical extent.

Images 15335 and 6471 are not used for a same-owner coordinate conclusion.
Their first-row person predictions often select different nearby people or an
edge-clipped person, so coordinate spread and instance selection cannot be
separated without a different intervention.

## Trajectory source by scoring checkpoint

The table reports bfloat16-forward coordinate argmaxes. Each cell contains the
clean/degraded branch modes under the named scoring checkpoint.

| Reviewed decision | Gaussian-native prefix, Gaussian scorer | Gaussian-native prefix, pure scorer | Pure-native prefix, Gaussian scorer | Pure-native prefix, pure scorer |
|---|---:|---:|---:|---:|
| Bowl y2 | 188 / 172 | 188 / 186 | 171 / 171 | 186 / 186 |
| Umbrella y2 | 283 / 283 | 303 / 303 | 283 / 283 | 297 / 303 |
| Bus x1 | 492 / 492 | 527 / 527 | 492 / 492 | 527 / 527 |

This is the central result. Changing prefix source does not remove the learned
mode. Changing the scoring checkpoint does.

For the pure-native bowl prefix that shares description, x1, y1, and x2:

- pure cross-entropy has y2 argmax 186, accepted-reference plus-or-minus-four
  raw coordinate mass 0.408, and sampling-policy mass 0.711;
- the Gaussian-smoothed checkpoint has y2 argmax 171, accepted-reference
  plus-or-minus-four raw mass 0.101, and sampling-policy mass 0.013.

The full-model float32 control changes the Gaussian argmax only from 171 to
172 and leaves the pure cross-entropy argmax at 186. The mechanism conclusion
is therefore not a bfloat16 boundary artifact.

## Existing matched validation-200 result

No new benchmark inference was needed. The two existing current Swift runs use
the same validation data, prompt fingerprint, decoding policy, batch size,
geometry-sorted format, repetition penalty 1.10, maximum generation horizon,
and evaluator.

| Adapter | Average precision | Average recall at 100 predictions | Full precision at intersection-over-union 0.30 | Full recall at intersection-over-union 0.30 | Predictions |
|---|---:|---:|---:|---:|---:|
| Pure cross-entropy plus token-type gate | 0.4156 | 0.5022 | 0.8017 | 0.6524 | 1,294 |
| Gaussian smoothing plus Ranked Probability Score and token-type gate | 0.4043 | 0.4945 | 0.7623 | 0.6440 | 1,336 |

Pure cross-entropy is better on every reported primary column despite producing
fewer predictions. The duplicate guard suppresses 175 pure-cross-entropy
predictions and 222 Gaussian predictions. Gaussian smoothing therefore does
not buy higher official recall in this matched run; it produces more errors and
more duplicate candidates.

Official COCO annotations remain incomplete, so this benchmark is not an
exhaustive estimate of physical object discovery. It is nevertheless a valid
relative comparison because both checkpoints use the same annotation and
inference contract.

## Interpretation

The evidence rejects three overly broad explanations for these selected cases.

1. **Off-policy prefix mismatch is not the main explanation.** The bowl,
   umbrella, and bus modes follow the scoring checkpoint even when the prefix
   comes from the other checkpoint.
2. **The strongest bowl truncation is not an unavoidable pure autoregressive
   defect.** Pure cross-entropy assigns the correct y2 mode under both Gaussian
   and pure-native prefixes.
3. **The historical sampler is not the sole explanation, but it did amplify
   one claim.** The bowl endpoint spread and the image-12670 outlier survive
   exact current-sampler replication. The historical bus-wide spread does not;
   only a stable checkpoint-specific left-boundary shift remains.

The supported, narrower explanation is that the combined Gaussian-smoothed
coordinate and Ranked Probability Score checkpoint learns different
coordinate preferences, produces more variable sampled extents in some cases,
and shifts coordinate modes. It clearly broadens the bowl's lower boundary,
admits one person branch failure, and shifts the bus left boundary, while
improving official overlap for the umbrella. The image-15335 person failure
remains severe under both objectives, so smoothing amplifies a native
autoregressive weakness rather than creating the whole problem. This
experiment cannot isolate the Gaussian term from the Ranked Probability Score
term because the trained checkpoints differ in their complete learned
parameter states.

## Stopping decision and next research direction

This unit is complete. The predeclared final control used 96 additional
Gaussian rollouts under the exact current sampler and removed the largest
remaining execution-path confound. Do not stack a new complete-row consistency
loss on the Gaussian checkpoint to repair these examples. That would risk
treating an objective-sensitive and case-dependent failure as a universal
architectural defect.

The next mechanism unit should return to the original unresolved question using
the pure cross-entropy checkpoint: why can sampling retrieve a valid object that
greedy decoding misses, and which prefix transition makes that object become
reachable? Coordinate smoothing should remain an ablation in that unit, not
the model from which cases are selected.

If coordinate smoothing is revisited later, it needs a matched four-arm training
comparison against pure cross-entropy and should be conditioned on coordinate
phase or axis rather than applied as an assumed universal improvement. Its
promotion gate must include complete-row owner coherence and full rollout
behavior, not only local probability mass around a coordinate target.

## Limitations

- The six images were selected for mechanism discrimination, not population
  prevalence.
- Three pure-native branch pairs were manually admitted; the umbrella case is
  explicitly ontology-sensitive.
- Matching the sampler implementation and seeds controls execution semantics,
  but checkpoint-dependent token distributions still make a seed an aligned
  random stream rather than a token-for-token counterfactual pair.
- The validation-200 comparison uses repetition penalty 1.10. It supports a
  matched relative conclusion, not an absolute claim about repetition penalty
  1.0.
- Current result artifacts persist configured model paths, but not complete live
  adapter and embedding payload hashes. The independent audit classified this
  as a non-blocking provenance improvement for publication-grade evidence.
