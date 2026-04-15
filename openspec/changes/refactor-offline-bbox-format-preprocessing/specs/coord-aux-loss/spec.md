## ADDED Requirements

### Requirement: Stage-1 non-canonical bbox experiments require provenance-matched prepared datasets
When Stage-1 coord auxiliary supervision is used with a non-canonical bbox
format such as `cxcy_logw_logh` or `cxcywh`, the training dataset SHALL be an
offline-prepared bbox-format branch whose provenance matches the authored
training config.

Normative behavior:
- when `custom.bbox_format=cxcy_logw_logh`, training MUST consume dataset
  artifacts stamped with:
  - `prepared_bbox_format=cxcy_logw_logh`,
  - `prepared_bbox_slot_order=cxcy_logw_logh`,
  - canonical source lineage,
  - supported bbox-format conversion version,
- when `custom.bbox_format=cxcywh`, training MUST consume dataset artifacts
  stamped with:
  - `prepared_bbox_format=cxcywh`,
  - `prepared_bbox_slot_order=cxcywh`,
  - canonical source lineage,
  - supported bbox-format conversion version,
- canonical `xyxy` datasets MUST be rejected for that experiment unless the
  authored bbox format is also canonical `xyxy`,
- if train and val artifacts disagree on prepared bbox-format provenance,
  startup MUST fail fast.

#### Scenario: Matching prepared bbox-format branch is accepted
- **GIVEN** `custom.bbox_format=cxcy_logw_logh`
- **AND** the training dataset metadata declares
  `prepared_bbox_format=cxcy_logw_logh`
- **AND** the training dataset metadata declares
  `prepared_bbox_slot_order=cxcy_logw_logh`
- **WHEN** training startup validates dataset provenance
- **THEN** startup succeeds
- **AND** coord auxiliary supervision uses the prepared branch semantics.

#### Scenario: Matching prepared cxcywh branch is accepted
- **GIVEN** `custom.bbox_format=cxcywh`
- **AND** the training dataset metadata declares
  `prepared_bbox_format=cxcywh`
- **AND** the training dataset metadata declares
  `prepared_bbox_slot_order=cxcywh`
- **WHEN** training startup validates dataset provenance
- **THEN** startup succeeds
- **AND** coord auxiliary supervision uses the prepared branch semantics.

#### Scenario: Canonical dataset is rejected for non-canonical Stage-1 training
- **GIVEN** `custom.bbox_format=cxcy_logw_logh`
- **AND** the training dataset lacks matching non-canonical prepared-bbox
  provenance
- **WHEN** training startup validates dataset provenance
- **THEN** startup fails fast
- **AND** it reports that a matching offline-prepared bbox-format branch is
  required.

#### Scenario: Canonical dataset is rejected for cxcywh Stage-1 training
- **GIVEN** `custom.bbox_format=cxcywh`
- **AND** the training dataset lacks matching non-canonical prepared-bbox
  provenance
- **WHEN** training startup validates dataset provenance
- **THEN** startup fails fast
- **AND** it reports that a matching offline-prepared bbox-format branch is
  required.

### Requirement: Runtime bbox-format conversion is unsupported for offline-prepared experiments
For offline-prepared non-canonical bbox-format experiments, runtime dataset and
builder layers SHALL NOT perform an additional bbox-format conversion.

Normative behavior:
- when a dataset artifact already declares a non-canonical prepared bbox format,
  runtime rendering MUST preserve the prepared model-facing bbox values,
- the runtime MUST fail fast if it detects both:
  - a non-canonical prepared bbox-format dataset, and
  - a request to reinterpret the same dataset as canonical `xyxy` for
    model-facing conversion,
- supported bbox-format behavior for these experiments is:
  - offline derivation owns conversion,
  - runtime validation owns provenance checking,
  - prompt/loss surfaces consume the prepared branch directly.

#### Scenario: Prepared branch is rendered without a second conversion
- **GIVEN** a dataset artifact stamped `prepared_bbox_format=cxcy_logw_logh`
- **WHEN** the runtime dataset/builder path renders an assistant payload for
  training
- **THEN** the emitted bbox slots match the prepared dataset values
- **AND** the runtime does not apply a second bbox-format transform.

#### Scenario: Prepared cxcywh branch is rendered without a second conversion
- **GIVEN** a dataset artifact stamped `prepared_bbox_format=cxcywh`
- **WHEN** the runtime dataset/builder path renders an assistant payload for
  training
- **THEN** the emitted bbox slots match the prepared dataset values
- **AND** the runtime does not apply a second bbox-format transform.

#### Scenario: Mixed conversion intent fails fast
- **GIVEN** a dataset artifact stamped with a non-canonical prepared bbox format
- **AND** runtime logic would otherwise attempt to convert canonical `xyxy`
  geometry into the same non-canonical format
- **WHEN** training startup or rendering validation runs
- **THEN** execution fails fast
- **AND** it reports the mixed conversion ownership error explicitly.
