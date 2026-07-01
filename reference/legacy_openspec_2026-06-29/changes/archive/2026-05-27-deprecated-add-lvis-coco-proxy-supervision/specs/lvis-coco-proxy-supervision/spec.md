# lvis-coco-proxy-supervision Specification (Delta)

## ADDED Requirements

### Requirement: Augmented COCO training records may append LVIS-derived proxy objects
The system SHALL support an offline augmented COCO training artifact that keeps
the canonical record contract while appending LVIS-derived recovered objects.

Normative behavior:

- each augmented record MUST remain structurally compatible with the existing
  JSONL / CoordJSON contract,
- proxy objects MUST be represented as ordinary object entries with standard
  `desc` and `bbox_2d` fields,
- the exporter MUST NOT inject proxy metadata into rendered object text,
- the exporter MUST support at least the object supervision tiers:
  - `real`
  - `strict`
  - `plausible`

#### Scenario: Augmented record preserves canonical object syntax
- **WHEN** an augmented COCO training record is exported
- **THEN** each object entry still renders as ordinary CoordJSON
- **AND** proxy provenance does not appear inside the rendered `objects[]`
  payload text.

### Requirement: Exported proxy objects require a determined mapping strategy
The augmentation pipeline SHALL determine an evidence-backed proxy strategy
before assigning supervision tiers to LVIS-derived objects.

Normative behavior:

- each candidate LVIS->COCO mapping MUST be classified as exactly one of:
  - `same_extent_proxy`
  - `cue_only_proxy`
  - `reject`
- the determination step MUST materialize a stable mapping artifact before
  dataset export,
- the export step MUST consume that determined mapping artifact rather than
  re-deriving strict/plausible policy implicitly from raw evidence tables,
- `same_extent_proxy` indicates the LVIS and COCO annotations usually target
  the same object extent,
- `cue_only_proxy` indicates the LVIS annotation is still a useful objectness or
  recall cue for the COCO class but the extent is systematically different,
- `reject` indicates the mapping is too ambiguous, weak, or geometrically
  unstable to export,
- supervision tiers MUST follow strategy:
  - `same_extent_proxy -> strict`
  - `cue_only_proxy -> plausible`
  - `reject -> not exported`

The determined mapping artifact MUST expose enough fields to audit the
decision, including at least:

- `lvis_category_id`
- `lvis_category_name`
- `mapped_coco_category_id`
- `mapped_coco_category_name`
- `determination_tier`
- `mapping_class`
- `decision_rule_version`
- `n_match`
- `n_images`
- `precision_like`
- `coverage_like`
- `mean_iou`
- `median_iou`
- `iou_ge_075_rate`

#### Scenario: Same-extent and cue-only strategies diverge
- **WHEN** the analysis compares mappings such as `mug -> cup` and
  `tablecloth -> dining table`
- **THEN** the pipeline may assign `mug -> cup` to `same_extent_proxy`
- **AND** it may assign `tablecloth -> dining table` to `cue_only_proxy`
- **AND** the two mappings do not need to share the same supervision tier just
  because both have semantic evidence.

#### Scenario: Export consumes an explicit determined mapping CSV
- **WHEN** a val2017 proxy determination pass writes
  `openspec/changes/add-lvis-coco-proxy-supervision/artifacts/determined_proxy_mappings_val2017.csv`
- **THEN** dataset export may consume that file as the semantic proxy source of
  truth
- **AND** the file contains the audited evidence fields needed to explain each
  `strict`, `plausible`, or `reject` decision.

### Requirement: Proxy strategy uses geometry-compatibility evidence, not semantic evidence alone
The augmentation pipeline SHALL use geometry-compatibility metrics in addition
to semantic matched-pair evidence when assigning proxy strategy.

Normative behavior:

- the strategy analysis MUST consider overlap-quality metrics such as IoU,
- the strategy analysis MUST also consider extent-compatibility metrics such as:
  - `intersection_over_lvis`
  - `intersection_over_coco`
  - `area_ratio`
  - one-way containment rates
  - normalized center offset
- semantic evidence alone MUST NOT be sufficient to assign
  `same_extent_proxy`,
- weak or contradictory geometry evidence MUST bias the mapping toward
  `cue_only_proxy` or `reject`.

#### Scenario: Strong semantic evidence with asymmetric geometry becomes cue-only
- **WHEN** a candidate mapping shows good semantic evidence but the LVIS boxes
  are systematically contained within the COCO box extent
- **THEN** the mapping is not promoted to `same_extent_proxy`
- **AND** it is classified as `cue_only_proxy` or `reject`.

### Requirement: Proxy supervision metadata aligns one-to-one with final object order
Augmented records SHALL carry top-level proxy supervision metadata aligned
exactly to the final exported object list.

Normative behavior:

- top-level metadata MUST include a stable proxy-supervision namespace,
- that namespace MUST contain one supervision entry per final object,
- each supervision entry MUST include:
  - `source`
  - `proxy_tier`
  - `mapping_class`
  - `desc_ce_weight`
  - `coord_weight`
- if final object ordering is re-sorted, the metadata entries MUST be re-sorted
  to the same final order,
- metadata/object count mismatches MUST fail fast.

#### Scenario: Metadata count mismatch fails fast
- **WHEN** an augmented record has `len(object_supervision) != len(objects)`
- **THEN** export or runtime validation fails fast
- **AND** training does not proceed with misaligned proxy metadata.

### Requirement: Final object ordering remains deterministic after proxy injection
The augmented export path SHALL preserve deterministic final object ordering.

Normative behavior:

- final objects MUST follow the repo's canonical `(minY, minX)` ordering
  invariant,
- injected proxy objects MUST participate in the same ordering rule as base
  COCO objects,
- proxy supervision metadata MUST remain aligned to that final order.

#### Scenario: Injected proxy objects follow canonical ordering
- **WHEN** recovered LVIS objects are merged into a COCO record
- **THEN** the final `objects[]` list remains deterministically ordered
- **AND** repeated exports with the same inputs produce the same object order.
