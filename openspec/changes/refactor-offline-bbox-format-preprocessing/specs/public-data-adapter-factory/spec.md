## ADDED Requirements

### Requirement: Adapters remain canonical producers for bbox-format branching
Dataset adapters SHALL emit canonical raw records only and SHALL NOT directly
author model-facing non-canonical bbox parameterizations.

Normative behavior:
- the adapter-to-pipeline boundary remains canonical pixel-space `xyxy`
  geometry for `bbox_2d`,
- adapters MUST NOT emit `cxcy_logw_logh` or other non-canonical bbox charts
  in raw outputs,
- non-canonical bbox-format derivation SHALL occur only in shared offline
  preprocessing stages after adapter-owned ingestion completes.

#### Scenario: Adapter emits canonical raw bbox records
- **WHEN** an adapter converts source annotations into raw CoordExp records
- **THEN** `bbox_2d` remains canonical `xyxy`
- **AND** the adapter does not emit a model-facing non-canonical bbox chart.

#### Scenario: New bbox-format branch does not require adapter-specific logic
- **WHEN** a new non-canonical bbox-format branch is added in shared
  preprocessing
- **THEN** existing adapters continue to emit the same canonical raw records
- **AND** adapter implementations do not need dataset-specific bbox-format code.

### Requirement: Shared pipeline planning owns bbox-format derivation stages
The adapter factory / shared pipeline boundary SHALL treat bbox-format
derivation as a shared offline stage rather than adapter-specific ingestion
logic.

Normative behavior:
- shared stage planning MAY insert a bbox-format derivation stage after
  canonical rescale preparation for workflows that request it,
- adapter contracts MUST remain limited to source ingestion and canonical
  intermediate-record production,
- adding a bbox-format-specific training branch MUST NOT require editing shared
  orchestration to introduce dataset-specific branching.

#### Scenario: Bbox-format derivation is shared across datasets
- **WHEN** canonical COCO and LVIS preset artifacts both request the same
  offline bbox-format branch
- **THEN** the shared derivation stage applies the format semantics uniformly
- **AND** adapters remain dataset-specific only at ingestion time.

#### Scenario: Unknown bbox-format derivation request fails before adapter logic changes
- **WHEN** a workflow requests an unsupported bbox-format branch
- **THEN** shared planning fails fast with an actionable error
- **AND** no adapter implementation changes are required to surface the error.
