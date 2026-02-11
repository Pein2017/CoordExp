# inference-pipeline Spec Delta

This is a delta spec for change `refactor-src-modernization`.

## ADDED Requirements

### Requirement: Pipeline config resolution is separated from stage execution
Inference-pipeline SHALL separate config parsing/resolution (including artifact path precedence and overrides) from stage execution logic.
Resolved stage and artifact decisions MUST be materialized before stage execution begins.

#### Scenario: Stage execution consumes pre-resolved artifact paths
- **GIVEN** a pipeline config with explicit artifact overrides
- **WHEN** the pipeline runs
- **THEN** stage handlers consume pre-resolved artifact paths
- **AND** stage handlers do not recompute precedence rules.

### Requirement: `resolved_config.json` is the canonical resolved manifest
Inference-pipeline SHALL persist resolved run metadata in `resolved_config.json` in the run directory.
The file SHALL include:
- active stage toggles,
- resolved artifact paths,
- a redacted config snapshot,
- `schema_version` as an integer major version (initial major `1` for this contract).

Within a given `schema_version` major, key evolution MUST be additive-only (existing stable keys must not be renamed/removed).
Consumers MUST explicitly reject unsupported major schema versions.
The pipeline SHALL NOT introduce a second parallel manifest artifact for the same contract.

#### Scenario: `resolved_config.json` captures resolved stage/artifact contract
- **WHEN** a pipeline run starts with infer/eval/vis toggles
- **THEN** the run directory includes `resolved_config.json` recording stage toggles and resolved artifact paths
- **AND** downstream stages/tools can consume this file as the single resolved-config source of truth.

#### Scenario: Unsupported manifest major version is rejected explicitly
- **GIVEN** a consumer reads `resolved_config.json` with an unsupported `schema_version` major
- **WHEN** contract validation runs
- **THEN** the consumer rejects the manifest with explicit version-mismatch diagnostics
- **AND** does not silently proceed with partial assumptions.

### Requirement: Relative image-root fallback behavior is explicit and shared across eval/vis
Inference-pipeline SHALL keep compatibility fallback behavior for relative image path resolution while making fallback activation explicit in logs.
The resolved root behavior SHALL be shared consistently for evaluation and visualization consumers so they do not diverge in relative-path handling.
For overlapping active deltas, detailed helper semantics are authoritative in `src-ambiguity-cleanup-2026-02-11`; this change MUST remain consistent with that contract.
Root resolution precedence SHALL be explicit and deterministic: `ROOT_IMAGE_DIR` env override > config root (`run.root_image_dir`) > `infer.gt_jsonl` parent directory > none.
The resolved image-root decision MUST be recorded in `resolved_config.json` with at least `root_image_dir` and `root_image_dir_source` (`env`, `gt_parent`, `config`, or `none`).
This requirement MUST preserve existing downstream contract compatibility for eval/vis flows.

#### Scenario: Eval/vis run with relative paths uses one explicit root-resolution contract
- **GIVEN** an eval+vis run and relative image paths in artifact records
- **WHEN** root fallback is required for path resolution
- **THEN** fallback activation is explicit
- **AND** both evaluation and visualization resolve image paths under the same root-resolution contract
- **AND** `resolved_config.json` records both the chosen `root_image_dir` and its `root_image_dir_source`
- **AND** stage outputs preserve unchanged artifact semantics.

### Requirement: `resolved_config.json` stable keys are explicit and `cfg` snapshot is opaque
Within `schema_version` major `1`, the stable top-level contract key set SHALL be:
- `schema_version`,
- `config_path`,
- `root_image_dir`,
- `root_image_dir_source`,
- `stages`,
- `artifacts`.

The redacted `cfg` snapshot MAY be persisted for diagnostics, but SHALL be treated as opaque/non-contract by downstream consumers.

#### Scenario: Downstream readers validate only stable manifest keys
- **GIVEN** a `resolved_config.json` containing both stable keys and a large redacted `cfg` snapshot
- **WHEN** a downstream consumer validates compatibility
- **THEN** compatibility decisions are based on stable keys and schema version
- **AND** changes inside `cfg` do not break the manifest contract.
