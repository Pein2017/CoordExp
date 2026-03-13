# gt-vs-pred-visualization Specification (Delta)

## ADDED Requirements

### Requirement: Canonical visualization resources are gt-vs-pred compatible at the top level and explicit at the object level
The system SHALL define one canonical single-view visualization-resource
contract that stays compatible with the existing `gt_vs_pred.jsonl` worldview at
the top level while using one explicit bbox-only object schema for review.

Normative behavior:

- the canonical single-view resource MUST remain centered on:
  - `gt`
  - `pred`
  - `image`
  - `width`
  - `height`
  - `coord_mode`
- canonical visualization records MUST additionally carry:
  - `schema_version`
  - `source_kind`
  - `record_idx`
- canonical visualization records MUST normalize to `coord_mode: "pixel"`,
- each object in both `gt` and `pred` MUST use the canonical visualization
  object schema:
  - `index` (int; stable within the record),
  - `desc` (string),
  - `bbox_2d` (list[int] of length 4 in pixel space as `[x1, y1, x2, y2]`),
- richer source-specific payloads MUST be carried additively under stable
  namespaces rather than as bespoke top-level keys:
  - `matching`
  - `stats`
  - `provenance`
  - `debug`
- `image_id`, `images`, and `file_name` MAY be included when available for
  auditability and cross-run joins,
- source-native geometry MAY be preserved under `debug` / `provenance`, but the
  canonical visualization object schema remains bbox-only for label and
  localization review,
- when materialized as a standalone artifact, a canonical visualization resource
  MUST NOT overwrite or path-alias the raw inference artifact
  `<run_dir>/gt_vs_pred.jsonl`,
- the default materialized location for a canonical visualization resource
  SHOULD be `<run_dir>/vis_resources/gt_vs_pred.jsonl`.

#### Scenario: Monitor dump sample normalizes into the canonical resource
- **GIVEN** a monitor-dump sample containing `gt_objects`, `pred_objects`,
  `match`, `stats`, and rollout text
- **WHEN** an upstream normalization step materializes a compatible
  `gt_vs_pred.jsonl` for visualization
- **THEN** the normalized record uses canonical `gt` / `pred` top-level keys
- **AND** each canonical object contains only `index`, `desc`, and pixel-space
  `bbox_2d`
- **AND** matching, stats, provenance, and debug payloads are preserved under the
  stable namespaces above.

### Requirement: Canonical GT ordering is deterministic across producers
Canonical visualization resources SHALL use one deterministic GT ordering rule so
`canonical_gt_index` means the same thing across producers.

Normative behavior:

- adapters MUST normalize GT objects into canonical pixel-space bbox-only objects
  before assigning canonical GT indices,
- canonical GT objects MUST be sorted lexicographically by:
  - `bbox_2d[0]`,
  - `bbox_2d[1]`,
  - `bbox_2d[2]`,
  - `bbox_2d[3]`,
  - then `desc`,
- canonical `gt[*].index` MUST be assigned from that sorted order,
- adapters that ingest source-local GT indices MUST remap them into
  `canonical_gt_index` before populating `matching`,
- adapters and renderers MUST NOT preserve arbitrary source GT ordering when it
  conflicts with the canonical GT ordering rule.

#### Scenario: Two producers normalize the same GT scene to the same GT order
- **GIVEN** two different source artifacts that describe the same GT boxes and
  descriptions for one scene
- **WHEN** both are normalized into canonical visualization records
- **THEN** both produce the same canonical `gt` array order
- **AND** any source-local GT indices are remapped into that same
  `canonical_gt_index` domain.

### Requirement: Canonical visualization normalization inverse-scales norm1000 sources before rendering
Canonical visualization resources SHALL normalize `norm1000`-style and
coord-token sources into pixel-space bounding boxes before any renderer consumes
them.

Normative behavior:

- if a source carries numeric coordinates in the `0..999` domain, the adapter
  MUST inverse-scale them to pixel space using the record’s `width` and
  `height`,
- if a source carries coord tokens in the same `0..999` domain, the adapter MUST
  resolve them through shared geometry helpers before deriving the canonical
  `bbox_2d`,
- if a source already carries pixel-space geometry, the adapter MUST normalize /
  clamp it without applying an extra scale,
- renderers MUST consume only the canonical pixel-space `bbox_2d` objects and
  MUST NOT implement their own source-specific `norm1000` scaling logic.

#### Scenario: Offline and monitor norm1000 payloads normalize to the same pixel box
- **GIVEN** one source record with offline-style geometry and one source record
  with monitor-style `points_norm1000`
- **AND** both describe the same `0..999` bbox under the same `width` /
  `height`
- **WHEN** both are normalized into canonical visualization records
- **THEN** both produce the same pixel-space canonical `bbox_2d`
- **AND** no renderer-local scaling branch is required to draw them correctly.

### Requirement: Canonical matching is required and normalized into one explicit sub-schema
The system SHALL require canonical visualization records used by the shared
GT-vs-Pred review renderer to carry one explicit canonical matching sub-schema
so renderers can color and label scenes consistently across sources.

Normative behavior:

- canonical shared-review records MUST include `matching`,
- `matching` MUST use normalized canonical index domains rather than
  source-specific ones,
- `matching.pred_index_domain` MUST equal `canonical_pred_index`,
- `matching.gt_index_domain` MUST equal `canonical_gt_index`,
- `matching` MUST include:
  - `match_source`,
  - `match_policy`,
  - `matched_pairs`,
  - `fn_gt_indices`,
  - `fp_pred_indices`,
- `matching` MAY additionally include:
  - `iou_thr`,
  - `pred_scope`,
  - `ignored_pred_indices`,
  - `unmatched_pred_indices`,
  - `unmatched_gt_indices`,
  - `gating_rejections`,
- adapters MUST normalize source-specific match payloads into this sub-schema
  before a scene is treated as canonical for shared review rendering,
- if a source does not already carry matching, canonical matching MUST be
  materialized in an explicit preprocessing step before rendering,
- renderers MUST NOT recompute matching as a fallback,
- attempts to render a shared-review canonical scene without canonical matching
  MUST fail fast.

#### Scenario: Evaluator and monitor matching normalize to the same index domains
- **GIVEN** one evaluator-selected scene and one monitor-dump scene that both
  carry precomputed matching
- **WHEN** each is normalized into a canonical visualization record
- **THEN** both records expose `matching.pred_index_domain=canonical_pred_index`
- **AND** both records expose `matching.gt_index_domain=canonical_gt_index`
- **AND** the renderer can interpret `matched_pairs`, `fn_gt_indices`, and
  `fp_pred_indices` without source-specific index translation.

#### Scenario: Shared review rendering fails fast when canonical matching is missing
- **GIVEN** a materialized visualization record that lacks canonical `matching`
- **WHEN** the shared GT-vs-Pred review renderer is invoked
- **THEN** rendering fails fast with explicit diagnostics instead of attempting
  fallback matching.

### Requirement: Prediction order is preserved for visualization tracing
Canonical visualization resources SHALL preserve prediction order so qualitative
review can trace model interpretation order rather than a post-hoc sorted order.

Normative behavior:

- the `pred` array MUST preserve the original relative order of surviving
  predicted objects from the source artifact,
- adapters and renderers MUST NOT sort predictions by score, description, area,
  IoU, or error status for presentation convenience,
- canonical records MUST preserve stable per-object indices so matching and
  debug references remain stable across derived artifacts,
- if invalid predictions are dropped during normalization, the surviving
  predictions MUST still preserve their original relative order and dropped-object
  diagnostics MAY be preserved under `debug`.

#### Scenario: Invalid prediction is dropped without reordering the survivors
- **GIVEN** a parsed prediction sequence with objects at source positions
  `0, 1, 2`
- **AND** object `1` is dropped during validation
- **WHEN** the scene is normalized
- **THEN** the surviving predictions appear in the same relative order as source
  positions `0, 2`
- **AND** the adapter does not renumber them by applying a score- or class-based
  sort.

### Requirement: Default GT-vs-Pred renderer semantics are shared and error-focused
The default reusable GT-vs-Pred renderer SHALL use one shared review-panel
semantics across offline and online workflows.

Normative behavior:

- the default layout MUST be `1x2`,
- GT MUST appear on the left,
- Pred MUST appear on the right,
- regular GT objects MUST render in green,
- GT false negatives MUST render in orange,
- matched predictions MUST render in green,
- false-positive predictions MUST render in red,
- object description labels MUST be error-focused by default:
  - `FN` and `FP` objects are labeled by default,
  - matched objects MAY remain unlabeled by default for readability,
- label placement MUST attempt to minimize overlap by using deterministic
  outside-box placement and conflict resolution before falling back to truncation.

#### Scenario: Default renderer highlights false negatives and false positives
- **GIVEN** a canonical visualization record with one matched prediction and one
  false-positive prediction
- **WHEN** the default renderer draws the scene
- **THEN** the matched prediction is green
- **AND** the false-positive prediction is red
- **AND** any GT false negative is orange on the GT panel
- **AND** the scene is drawn as GT-left / Pred-right.

### Requirement: Comparison workflows require GT equivalence after candidate alignment
Pairwise or multi-run visualization workflows SHALL be built by aligning multiple
canonical single-view resources rather than redefining a compare-only per-object
schema, but candidate join keys alone are insufficient proof of scene identity.

Normative behavior:

- comparison members MUST each preserve the canonical single-view object schema,
- candidate alignment MUST use stable join keys:
  - `record_idx`
  - plus `image_id` and/or `file_name` / `image` when available,
- after candidate alignment, the compositor MUST verify GT equivalence before
  composing a multi-run scene,
- GT equivalence verification MUST include:
  - exact `width` match,
  - exact `height` match,
  - exact equality of the canonical normalized `gt` arrays,
- comparison composers MUST preserve each member’s ordered `pred` array,
- comparison composition MUST fail fast when aligned members disagree on the
  canonical GT scene,
- optional comparison manifests MAY reference multiple labeled canonical members,
  but they MUST NOT redefine the underlying per-object schema.

#### Scenario: Backend comparison preserves ordered predictions from both members
- **GIVEN** two normalized resources for the same `record_idx`, one from `hf`
  and one from `vllm`
- **WHEN** a comparison scene is composed
- **THEN** each member keeps its own ordered `pred` list
- **AND** the comparison uses stable join keys rather than a custom compare-only
  object format.

#### Scenario: Composition fails fast on GT mismatch
- **GIVEN** two candidate-aligned records that share `record_idx` and `image`
- **AND** their normalized `gt` arrays differ
- **WHEN** a comparison scene is composed
- **THEN** the compositor fails fast instead of silently treating them as the
  same scene.

#### Scenario: Materialized visualization resource does not reuse raw inference path
- **GIVEN** a run directory that already contains the raw inference artifact
  `<run_dir>/gt_vs_pred.jsonl`
- **WHEN** a canonical visualization resource is materialized as a standalone
  artifact
- **THEN** it is written to a distinct derived path such as
  `<run_dir>/vis_resources/gt_vs_pred.jsonl`
- **AND** the raw inference artifact path is left unchanged.

### Requirement: Shared GT-vs-Pred visualization consumes explicit input and output paths and fails fast
The shared GT-vs-Pred visualization entrypoint SHALL accept an explicit input
`gt_vs_pred.jsonl` path and an explicit output path, and it SHALL fail fast on
unexpected contract violations.

Normative behavior:

- the visualizer MUST require:
  - an explicit input `gt_vs_pred.jsonl` path,
  - an explicit output path,
- the visualizer MUST NOT discover or rewrite `monitor_dumps` paths,
- the visualizer MUST fail fast when required canonical fields are missing,
  malformed, or inconsistent,
- implementation MAY use targeted exception handling only for uncontrolled input
  parsing surfaces such as raw sequence / token parsing in upstream
  normalization, but shared review rendering itself MUST remain fail-fast.

#### Scenario: Visualization invocation fails fast on missing canonical fields
- **GIVEN** an input `gt_vs_pred.jsonl` record missing canonical `matching`
- **WHEN** the shared visualization entrypoint is invoked
- **THEN** the visualization run fails fast with explicit diagnostics
- **AND** it does not attempt source discovery, fallback reconstruction, or
  partial rendering.
