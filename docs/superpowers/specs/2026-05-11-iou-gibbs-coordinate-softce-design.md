# IoU-Gibbs Coordinate SoftCE Design

Date: 2026-05-11

Status: refined proposal after first audit pass. Implementation has not started.

Owner surface: latest compact recursive detection stack under `src/detection/*`.

Primary launch target: A5, extending
`configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml`
with coordinate-token soft supervision while keeping the A2 training/data setup
unchanged.

## Purpose

Replace the old fixed Gaussian coordinate-token softCE shape with a
data-calibrated, geometry-aware, smooth continuous target family for
`compact_full` recursive detection training.

The latest recursive detection stack should treat the new continuous target
family as canonical. The old fixed Gaussian/deviation/truncation surface
(`sigma`, `target_sigma`, `truncate`, `target_truncate`, fixed token-radius
windows) is deprecated for latest recursive detection. Legacy modules may stay
available for old configs, but A5 must not route through them.

The target should make coordinate supervision less arbitrary without creating a
new hyperparameter garden. It must stay within the current autoregressive token
loss surface:

```text
compact_full ET-RMP recursive CE
  + unchanged A2 support/balance coefficients
  + geometry-aware soft target mass over <|coord_0|> ... <|coord_999|>
  + dense geometry-valid coordinate support
  + full-vocab probability normalization
```

This design intentionally does not add a detection head, decoded-box-only loss,
RL signal, beam-search training objective, or upstream Qwen/HF model change.

## Baseline And Data Scope

The production comparison baseline is the current A2/support+balance ET-RMP
config:

```text
configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml
```

Relevant fixed surface:

| Field | Value |
|---|---|
| `detection_template.id` | `compact_full` |
| `detection_template.coordinate_surface` | `coord_token` |
| `detection_template.bbox_format` | `xyxy` |
| `data.train_jsonl` | `public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl` |
| `data.val_jsonl` | `public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl` |
| `data.object_ordering` | `random_permutation` |
| coord tokens | `<|coord_0|>` through `<|coord_999|>` |
| support weight | `2.0` |
| balance weight | `1.0` |

The A5 YAML must extend this config and only change:

- run/artifact identity;
- `objective.coord_soft_ce`;
- non-claiming experiment metadata (`surface: ablation`, `claim_scope: none`).

No optimizer, data, prompt/template, base model, token-row, LoRA, epoch, batch,
packing/cache, eval cadence, or recursive non-coordinate behavior should drift.

## Empirical Geometry Summary

The current training JSONL contains:

| Quantity | Value |
|---|---:|
| rows | `117247` |
| objects / bboxes | `848656` |
| degenerate token boxes | `0` |
| median objects per image | `4` |
| p75 objects per image | `10` |
| p99 objects per image | `32` |

Box dimensions in coordinate-token bins:

| Statistic | Width | Height |
|---|---:|---:|
| p5 | `14` | `22` |
| p10 | `21` | `31` |
| p25 | `40` | `60` |
| median | `93` | `134` |
| p75 | `218` | `309` |
| p90 | `455` | `590` |
| p95 | `674` | `771` |

Additional shape observations:

- Median bbox area fraction is `0.0121` of the norm-1000 image plane.
- p75 bbox area fraction is `0.0616`; p95 is `0.4114`.
- Aspect ratio `width / height` has p25 `0.41`, median `0.69`, p75 `1.15`,
  and p95 `2.62`.
- `min(width, height) <= 32` covers about `23.1%` of boxes.
- `min(width, height) <= 50` covers about `36.4%` of boxes.
- `min(width, height) <= 100` covers about `58.3%` of boxes.
- Exact image-boundary contact appears in about `15.1%` of boxes.

One-token coordinate sensitivity:

| Statistic | `1 - IoU` over valid one-coordinate-token perturbations |
|---|---:|
| median | `0.0090909091` |
| mean | `0.016641985` |
| p75 | `0.0204082` |
| p90 | `0.04` |
| p95 | `0.0588235` |
| p99 | `0.111111` |

Per-box worst valid one-token perturbation has median `0.01299`, matching
`1 / median_min_side` because median `min(width, height)` is about `77`.

The empirical and analytic conclusion is:

```text
one-token x-edge sensitivity ~= 1 / width
one-token y-edge sensitivity ~= 1 / height
```

That makes a fixed token-space Gaussian target systematically too soft for
small boxes and too sharp for large boxes.

## Calibration Contract

`tau_data = 0.0090909091` is the `train_one_token_iou_median_v0` statistic.

The reproducible definition is:

```text
For every training bbox and every slot r in {x1, y1, x2, y2},
evaluate delta in {-1, +1}. Include the perturbation only if replacing
r by r + delta keeps 0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999.
The statistic is the median of 1 - IoU over this pooled perturbation set.
```

The implementation plan must include a calibration script/artifact that records:

- train JSONL path;
- train JSONL content hash or size/mtime fallback if hashing is too expensive;
- row count;
- object count;
- included perturbation count;
- skipped invalid perturbation count;
- median, mean, p75, p90, p95, and p99 of `1 - IoU`;
- the resolved `tau_data` used by A5.

This value must not be tuned from eval AP.

## Locked Target Distribution

Use a data-calibrated IoU-Gibbs target distribution for coordinate-token
positions.

For a ground-truth bbox in coordinate-token units:

```text
b = (x1, y1, x2, y2)
```

For coordinate slot:

```text
r in {x1, y1, x2, y2}
```

and candidate coordinate token:

```text
k in {0, 1, ..., 999}
```

define:

```text
b_r(k) = b with only coordinate r replaced by k
```

Do not canonicalize invalid candidate boxes by swapping edges. Candidate `k` is
valid only when the resulting box satisfies:

```text
0 <= x1 < x2 <= 999
0 <= y1 < y2 <= 999
```

The target energy is:

```text
E_r(k; b) = 1 - IoU(b_r(k), b)
```

The target probability is:

```text
q_r(k | b) =
  exp(-E_r(k; b) / tau_data)
  / sum_{l in V_r(b)} exp(-E_r(l; b) / tau_data)
```

where `V_r(b)` is the valid candidate set for slot `r`. Invalid candidates have
probability zero.

## Loss Semantics

A5 replaces hard coordinate supervision with an IoU-Gibbs soft target while
preserving the A2 recursive support/balance coefficients. It intentionally does
not preserve the old hard/discrete coordinate support cardinality: coordinate
support becomes the dense geometry-valid coordinate-token set for the slot.

This means A5 is best interpreted as:

```text
same A2 training setup and support/balance coefficients
+ continuous geometry-valid coordinate support
+ IoU-Gibbs balance target over that support
```

not as a pure target-shape change over the old hard support set.

For a coordinate position, let:

```text
S_geom = {coord_token_id(k) where k in V_r(b)}
log_p(k) = log softmax_full_vocab(logits)[coord_token_id(k)]
log_m = logsumexp_{token_id in S_geom} log_p(token_id)
```

Define:

```text
support = -log_m
balance = -sum_{k in V_r(b)} q(k) * (log_p(coord_token_id(k)) - log_m)
L_coord = trie_support_weight * support + trie_balance_weight * balance
```

For A5:

```text
trie_support_weight = 2.0
trie_balance_weight = 1.0
```

When `trie_support_weight = 1` and `trie_balance_weight = 1`, this reduces to
ordinary full-vocab softCE:

```text
L_coord = -sum_k q(k) log_p(k)
```

The phrase "pure softCE" in A5 means:

- no hard one-hot coordinate CE is optimized for coordinate positions;
- no W1 term is optimized;
- no decoded CIoU/SmoothL1 bbox loss is optimized;
- the optimized coordinate target is the IoU-Gibbs soft distribution.

It does not mean dropping A2's recursive support/balance coefficients.

The implementation must not derive `S_geom` from `q > 0`, because tiny but
valid IoU-Gibbs probabilities may underflow in finite precision. The target
builder/helper must carry an explicit geometry-valid mask or support token set
separately from the normalized probability vector.

For non-coordinate positions, keep the existing recursive detection CE behavior:

- `hard_ce` positions use hard one-token CE.
- `trie_multi_positive` positions use the existing support/balance loss.
- type-gate and boundary/EOS policies remain unchanged.

## ET-RMP Support Semantics

For singleton coordinate positions:

```text
q(k) = q_r(k | teacher_box)
```

For coordinate trie positions with multiple valid remaining objects, use an
object-uniform support mixture over the active trie support:

```text
q_state(k) = sum_o pi_o q_r(k | box_o)
pi_o = 1 / number_of_active_support_objects_at_this_trie_node
```

This preserves the existing object-multiplicity semantics of ET-RMP. If several
objects share the same candidate coordinate token, their probability mass
naturally accumulates.

Coordinate-softCE eligibility must be driven by explicit target metadata, not
by `SemanticRole.BBOX_COORD` alone. In the current target builder a coordinate
trie divergence can keep semantic role `ENTRY_TRIE_DECISION` for normalization
and metric bucket purposes. A5 may still replace that position's optimized loss
when the target carries coordinate metadata.

Required target metadata:

```text
target.coord_slot_name in {x1, y1, x2, y2}
target.coord_soft_targets = tuple of active object bbox candidates
```

Each `_TrieObjectInstance` must carry:

```text
bbox_xyxy
coord_slot_by_trie_offset
coord_token_by_slot
```

Before creating a support mixture, target construction must prove:

- every active instance at the trie node has the same non-null coordinate slot
  for that trie offset;
- the teacher slot and active instance slots match;
- each active instance's next child token equals the serialized coordinate token
  for that instance and slot.

If any invariant fails while `coord_soft_ce` is enabled, target construction
must raise a context-rich `ValueError`. It must not silently fall back to hard CE.

## Why IoU-Gibbs Instead Of Fixed Gaussian

The fixed Gaussian/deviation target has arbitrary knobs:

```text
sigma
truncate radius
temperature
```

Geometry-aware axis Gaussian reduces one failure mode but still needs:

```text
size fraction
sigma floor
sigma cap
truncate policy
```

IoU-Gibbs has one research scalar:

```text
tau_data
```

and that scalar is calibrated from the training data, not hand tuned. The
formula automatically gives:

- sharp targets for tiny objects;
- broader targets for large objects;
- x-slot adaptation by bbox width;
- y-slot adaptation by bbox height;
- anisotropic behavior for elongated objects;
- valid-range and edge-order awareness through candidate masking;
- no hard truncation radius.

Locally, IoU-Gibbs behaves like a relative-edge-error Laplace target:

```text
q_x(d) ~= exp(-|d| / (tau_data * width))
q_y(d) ~= exp(-|d| / (tau_data * height))
```

but it remains globally tied to actual IoU of the one-coordinate candidate box.

## Target-Shape Launch Gate

The formula is intentionally sharp for small objects. That is mathematically
aligned with IoU sensitivity, but it is still a launch risk because roughly
23.1% of boxes have `min(width, height) <= 32`.

Before production-scale A5 training, run a no-training target-shape audit over
the train JSONL and report:

- entropy;
- perplexity;
- peak probability;
- effective support size;
- valid candidate count;
- effective support size;
- target std;
- each statistic by min-side decile, area decile, coordinate slot, and boundary
  contact flag.
- explicit fixed small-object buckets for `min_side <= 32` and `min_side <= 50`.

This audit is not a tau tuning loop. It is a safety check that the target family
is not unintentionally hard-label-like for an unacceptable slice.

## Why Not CIoU First

CIoU-Gibbs is a valid follow-up ablation because the repo already has CIoU
geometry machinery. It is not the recommended first A5 target.

For the first softCE replacement, plain IoU-Gibbs is cleaner because:

- it is directly aligned with COCO localization semantics;
- it is easier to explain as maximum-entropy mass ordered by one-coordinate box
  overlap;
- it avoids extra center/aspect effects that make attribution harder;
- it provides the scale adaptation needed for this ablation without adding a
  second geometry objective.

If A5 improves or gives useful diagnostics, a later A6 can compare
`ciou_gibbs_v0` against `iou_gibbs_v0`.

## Config Surface

Add an objectized latest-schema section under `objective`:

```yaml
objective:
  id: recursive_detection_ce
  variant: random_permutation_et_rmp_ce
  trie_support_weight: 2.0
  trie_balance_weight: 1.0
  state_weighting: uniform_permutation
  normalization: semantic_image_bucket_balanced
  coord_soft_ce:
    enabled: true
    target_distribution: iou_gibbs_v0
    tau: 0.0090909091
    tau_source: train_one_token_iou_median_v0
    weighting: preserve_recursive_support_balance
    replace_coord_hard_ce: true
    apply_to_multi_positive: support_mixture
```

The latest schema must reject fixed Gaussian knobs under `objective.coord_soft_ce`
(`sigma`, `truncate`, `target_sigma`, `target_truncate`, `window`, `radius`).

The A5 production-scale ablation config extends the A2 support2 config and must
keep all other training/data/model settings unchanged.

Experiment metadata should be non-claiming until results exist:

```yaml
experiment:
  surface: ablation
  ablation_id: A5-iou-gibbs-softce-support2
  claim_scope: none
```

## Required Diagnostics

Canonical event namespace:

```text
detection_sequence/objective/recursive_detection_ce/coord_soft_ce/*
```

Trainer reporter metrics may additionally expose flat aliases:

```text
recursive_detection_ce/coord_soft_ce/*
```

Required finite diagnostics:

| Metric suffix | Meaning |
|---|---|
| `enabled` | `1.0` when active |
| `tau` | resolved `tau_data` |
| `token_count` | supervised coordinate target count |
| `support_loss` | support term before support weight |
| `support_mass` | `exp(-support_loss)`, full-vocab probability mass on dense valid coordinate support |
| `outside_support_mass` | `1 - support_mass` |
| `balance_loss` | soft target balance term before balance weight |
| `weighted_loss` | final optimized coordinate contribution |
| `pure_soft_ce_equiv` | `support_loss + balance_loss`, for interpretability |
| `target_entropy` | entropy of `q` |
| `kl_like` | `pure_soft_ce_equiv - target_entropy` |
| `peak_prob` | mean target peak mass |
| `perplexity` | mean `exp(entropy)` |
| `effective_support_size` | inverse participation ratio or documented equivalent over `q` |
| `valid_candidate_count` | number of geometry-valid coordinate candidates |
| `target_std` | mean coordinate-bin target std |
| `support_mixture_fraction` | fraction using multi-object mixture |
| `missing_geometry_count` | must remain zero for active objective |

The standard hard-teacher coord-token CE/accuracy summary can remain as a
diagnostic, but it must not be confused with the optimized coordinate loss.

## Deprecation Contract For Fixed Gaussian SoftCE

Latest recursive detection should not add or accept new Gaussian/truncation
configuration. The implementation plan should:

- document `src/tokens/coord/soft_ce_w1.py` and
  `src/trainers/losses/coord_soft_ce_w1.py` as legacy compatibility surfaces;
- keep old configs working unless they are already blocked by latest recursive
  detection guards;
- reject old fixed-shape knobs in the new `objective.coord_soft_ce` schema;
- avoid routing A5 through `custom.coord_soft_ce_w1` or
  `CoordSoftCEW1LossMixin`;
- avoid adding new latest docs that recommend `sigma`/`truncate` coordinate
  targets.

Deleting legacy code is out of scope for A5 unless a separate compatibility
audit proves there are no remaining supported references.

## Non-Goals

- Do not add W1 to the first A5 objective.
- Do not add decoded bbox CIoU or SmoothL1 to the first A5 objective.
- Do not use legacy `custom.coord_soft_ce_w1` or the legacy coord-softCE mixin.
- Do not expose fixed Gaussian `sigma`/`truncate` knobs in latest recursive
  detection.
- Do not change the dataset, prompt, compact-full chat template, object order,
  optimizer, LoRA setup, epoch count, batch-size semantics, or base model.
- Do not turn this into a Stage-2 objective or rollout objective.
- Do not tune `tau_data` from eval AP. It is a data-derived target-shape
  constant for this ablation.

## Acceptance Criteria

- The tau calibration script/artifact reproduces `train_one_token_iou_median_v0`.
- The target-shape audit is available before a production-scale launch.
- The A5 YAML parses under the latest strict schema.
- The A5 YAML extends A2/support2 and differs only in run identity,
  `objective.coord_soft_ce`, and non-claiming experiment metadata.
- The A5 config metadata uses `surface: ablation` and `claim_scope: none`.
- Coordinate softCE is computed through recursive CE, not through legacy trainer
  mixins.
- Coordinate targets use full-vocab softmax probabilities over coord token IDs.
- Coordinate optimized loss preserves A2 support/balance coefficients while
  adopting dense geometry-valid coordinate support.
- Non-coordinate recursive CE behavior remains unchanged.
- Hard CE coordinate positions are replaced by IoU-Gibbs soft targets.
- Coordinate trie multi-positive positions use support-mixture IoU-Gibbs only
  when same-slot geometry metadata is proven.
- Target entropy, KL-like softCE, support/balance terms, peak probability, and
  support-mixture metrics are logged.
- Targeted tests cover tiny, medium, large, boundary-touching, invalid-candidate,
  singleton, multi-positive, same-slot, and slot-mismatch fail-fast cases.
