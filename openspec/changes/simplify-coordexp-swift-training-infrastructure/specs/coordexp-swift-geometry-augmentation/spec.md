## ADDED Requirements

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

## MODIFIED Requirements

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

#### Scenario: Random ordering retains seed key

- **WHEN** object ordering is `random` and geometry flips are enabled
- **THEN** the transform MUST be sampled before randomized object ordering
- **AND** cached presentation metadata MUST retain the object-order seed/key
  needed to reproduce rendered order.

#### Scenario: Worker order does not affect random ordering

- **WHEN** examples are materialized with different worker completion orders
- **THEN** presentation transform ids and randomized object-order seed/keys
  MUST remain identical for the same resolved config and seed.

## REMOVED Requirements

### Requirement: Augmentation receipts are compact and sufficient

**Reason**: Reproducibility belongs to the semantic cache identity and cache
materialization metadata. A separate per-run augmentation receipt duplicates
derived cache state and encourages presentation-level artifact growth.

**Migration**: Record compact augmentation summary and deterministic ordering
handles in the current cache manifest. Validate presentation-level behavior in
focused tests; do not create a training-run receipt family.
