# coordexp-swift-geometry-augmentation Specification

## Purpose
TBD - created by archiving change add-coordexp-swift-geometry-flip-augmentation. Update Purpose after archive.
## Requirements
### Requirement: Train-only geometry flip config
CoordExp-Swift SHALL support train-only geometry flip augmentation through the
resolved config path `data.augmentation.train.geometry_flips`.

The config MUST accept `enabled`, `horizontal_prob`, and `vertical_prob`.
Probabilities MUST be finite numbers in `[0.0, 1.0]`. Unknown fields MUST fail
config validation. The default behavior MUST be equivalent to augmentation
disabled.

#### Scenario: Disabled augmentation preserves behavior
- **WHEN** a training config omits `data.augmentation` or sets `data.augmentation.train.geometry_flips.enabled: false`
- **THEN** raw training examples pass into rendering without sampled geometry transforms
- **AND** eval and inference inputs remain unaugmented

#### Scenario: Invalid probability fails config validation
- **WHEN** `horizontal_prob` or `vertical_prob` is less than `0.0`, greater than `1.0`, non-finite, or not numeric
- **THEN** config resolution MUST fail before dataset materialization

#### Scenario: Eval does not accept augmentation
- **WHEN** an eval or inference config attempts to enable geometry flip augmentation
- **THEN** the system MUST reject the eval or inference augmentation config before row materialization
- **AND** a train-only augmentation block in a training config MUST NOT apply transforms to eval.forward rows

### Requirement: Static stochastic presentation materialization
When geometry flips are enabled for training, CoordExp-Swift SHALL materialize
exactly one presentation per source training example during pack-cache
materialization.

Horizontal and vertical flip decisions MUST be sampled independently from the
resolved probabilities. The realized transform id MUST be one of `identity`,
`hflip`, `vflip`, or `hvflip`. Same resolved config, same seed, same source
example identity, same policy version, and same code identity MUST produce the
same presentation sequence.

#### Scenario: Output count equals input count
- **WHEN** a train split contains `N` source examples and geometry flips are enabled
- **THEN** the augmentation processor MUST emit exactly `N` training presentations

#### Scenario: Independent horizontal and vertical decisions compose
- **WHEN** both sampled Bernoulli decisions are true for one source example
- **THEN** the realized transform id MUST be `hvflip`
- **AND** no additional h-only or v-only duplicate presentation is emitted

#### Scenario: Same seed reuses sampled view
- **WHEN** the same dataset, resolved config, runtime seed, policy version, and code identity are materialized twice
- **THEN** every presentation id, transform id, transformed bbox, and object order seed/key MUST match across the two materializations

#### Scenario: Cache hit does not resample
- **WHEN** a valid pack cache exists for the same augmentation cache identity
- **THEN** the training pipeline MUST load the cached sampled presentations and packed micro-steps
- **AND** it MUST NOT resample geometry flip decisions on cache hit

### Requirement: Matrix-based bbox transforms

CoordExp-Swift SHALL define geometry flips as affine transforms over the
inclusive `0..999` coordinate-bin plane. The implementation MUST transform all
four corners of each bbox and canonicalize the result to `x1,y1,x2,y2` by
min/max. Transformed bboxes MUST pass the same validation contract as loaded
`RawObject` bboxes. Realized transform identity MUST stay attached to the
in-memory presentation and current cache materialization metadata.

#### Scenario: Horizontal flip maps boundary boxes correctly

- **WHEN** a bbox has coordinates `(x1, y1, x2, y2)` and the realized transform
  is `hflip`
- **THEN** the transformed bbox MUST equal
  `(999 - x2, y1, 999 - x1, y2)`
- **AND** boundary coordinates at `0` and `999` MUST remain valid.

#### Scenario: Vertical flip maps boundary boxes correctly

- **WHEN** a bbox has coordinates `(x1, y1, x2, y2)` and the realized transform
  is `vflip`
- **THEN** the transformed bbox MUST equal
  `(x1, 999 - y2, x2, 999 - y1)`
- **AND** boundary coordinates at `0` and `999` MUST remain valid.

#### Scenario: Composed flip uses one matrix-equivalent transform

- **WHEN** the realized transform is `hvflip`
- **THEN** the transformed bbox MUST equal
  `(999 - x2, 999 - y2, 999 - x1, 999 - y1)`
- **AND** presentation/cache metadata MUST identify the transform as `hvflip`.

#### Scenario: Invalid transformed bbox fails

- **WHEN** a transformed bbox is degenerate or outside the inclusive
  `0..999` range
- **THEN** materialization MUST fail with a data contract error rather than
  silently dropping or clipping the object.

### Requirement: In-memory Qwen image transform alignment
CoordExp-Swift SHALL apply the realized geometry transform to image pixels in
memory before Qwen image processing, using the same transform id that was used
for bbox transformation.

The system MUST NOT materialize flipped image files in V1. Flips MUST preserve
the no-resize Qwen image contract: image width, height, `image_grid_thw`, merged
visual token count, and packing image-token cost remain unchanged by the flip.

#### Scenario: Image plan records logical transform
- **WHEN** a training presentation uses `hflip`, `vflip`, or `hvflip`
- **THEN** the Qwen image plan or receipt MUST record the original image path and the logical transform id
- **AND** the original image path MUST remain the storage source

#### Scenario: No-resize geometry remains stable
- **WHEN** a flipped presentation is encoded for Qwen with `do_resize=False`
- **THEN** the image grid and visual token count MUST match the unflipped source image grid and visual token count

#### Scenario: Pixel view matches coordinate transform
- **WHEN** a presentation uses a realized transform id
- **THEN** the in-memory pixel transform MUST correspond to the same coordinate-plane transform id recorded for bbox transformation

### Requirement: Object ordering remains explicit and reproducible

CoordExp-Swift SHALL preserve the existing object-ordering contract while
supporting augmentation. For `source_order`, the processor MUST preserve
source object tuple order after bbox transform. For `geo_sorted`, the
processor MUST transform bboxes first and then prepare the presentation object
tuple in top-to-bottom then left-to-right order before renderer assertion. For
`random`, the processor MUST support augmentation and random order together
and MUST retain a deterministic object-order seed/key or equivalent handle in
the cached presentation metadata. Normal training MUST NOT persist a separate
per-presentation ordering receipt family.

#### Scenario: Source order is preserved after transform

- **WHEN** object ordering is `source_order` and geometry flips are enabled
- **THEN** the rendered presentation MUST see objects in original source tuple
  order after bbox transformation.

#### Scenario: Geo sorted is asserted after transform

- **WHEN** object ordering is `geo_sorted` and geometry flips are enabled
- **THEN** bboxes MUST be transformed before the presentation object tuple is
  sorted
- **AND** the renderer's existing `geo_sorted` assertion MUST pass on the
  transformed presentation.

#### Scenario: Random ordering records seed key

- **WHEN** object ordering is `random` and geometry flips are enabled
- **THEN** the transform MUST be sampled before randomized object ordering
- **AND** cached presentation metadata MUST retain the object-order seed/key
  needed to reproduce rendered order.

#### Scenario: Worker order does not affect random ordering

- **WHEN** examples are materialized with different worker completion orders
- **THEN** presentation transform ids and randomized object-order seed/keys
  MUST remain identical for the same resolved config and seed.

### Requirement: Packing cache identity includes augmentation semantics
CoordExp-Swift SHALL include augmentation semantics in the packing-cache
semantic fingerprint.

The fingerprint MUST include resolved augmentation config, augmentation seed
source, augmentation policy version, augmentation code identity, and relevant
Qwen image transform/materialization code identity. Worker count MUST be
recorded only as provenance and MUST NOT affect the semantic fingerprint.

#### Scenario: Probability change invalidates cache
- **WHEN** `horizontal_prob` or `vertical_prob` changes in the resolved config
- **THEN** the packing-cache fingerprint MUST change

#### Scenario: Seed change invalidates cache
- **WHEN** the runtime seed used for geometry flip sampling changes
- **THEN** the packing-cache fingerprint MUST change

#### Scenario: Worker count does not invalidate cache
- **WHEN** only the internal pack-cache materialization worker count changes
- **THEN** the packing-cache fingerprint MUST remain unchanged

#### Scenario: Augmentation code change invalidates cache
- **WHEN** the augmentation geometry, processor, factory, or relevant Qwen image materialization code identity changes
- **THEN** the packing-cache fingerprint MUST change

### Requirement: Augmentation Provenance Stays In Cache State

When train-only geometry augmentation is enabled, the current-version packing
cache manifest SHALL contain compact materialization provenance sufficient to
explain and reproduce the sampled presentation set. It MUST include mode,
split, policy and probabilities, seed source, input/output example counts,
transform counts, dataset identity, and reproducibility handles for randomized
object order. Normal training MUST NOT copy presentation-level augmentation
receipts into every run directory.

#### Scenario: Augmented cache is materialized

- **WHEN** a cache miss materializes augmented training presentations
- **THEN** the cache manifest MUST record counts for `identity`, `hflip`,
  `vflip`, and `hvflip`
- **AND** their sum MUST equal both input and output example count.

#### Scenario: Random object order is used

- **WHEN** an augmented presentation uses randomized object ordering
- **THEN** cache materialization metadata MUST retain the deterministic
  seed/key or equivalent reproducibility handle.

#### Scenario: Cache is reused by training

- **WHEN** training consumes a matching complete augmented cache
- **THEN** the run MUST reuse its sampled presentations without resampling
- **AND** MUST NOT duplicate the cache's presentation provenance as a run
  receipt family.
