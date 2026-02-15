# fusion-dataset Specification

## Purpose
Define the fusion-config dataset contract (`custom.fusion_config`) for building train/eval datasets from multiple JSONL sources under a single experiment.

## Requirements
### Requirement: Fusion Config Overrides Standard JSONL Paths
When `custom.fusion_config` is set, the system SHALL build train/eval datasets from the fusion config and SHALL ignore `custom.train_jsonl` and `custom.val_jsonl`.

#### Scenario: Fusion Config Takes Priority
- **WHEN** a training YAML sets `custom.fusion_config: configs/fusion/variants/example.yaml`
- **AND** the same YAML also sets `custom.train_jsonl` and/or `custom.val_jsonl`
- **THEN** training and evaluation datasets SHALL be determined by the fusion config
- **AND** `custom.train_jsonl` / `custom.val_jsonl` SHALL NOT change the fused datasets.

### Requirement: Fusion Config File Schema (Qwen3-VL Compatible Containers)
The fusion config file SHALL be a mapping (YAML/JSON) with:
- `targets`: list of dataset entries
- `sources`: optional list of dataset entries (**accepted for compatibility**; treated the same as `targets`)
- `extends`: optional string or list of strings (inheritance)

At least one dataset entry across `targets` and `sources` SHALL be provided.

#### Scenario: Targets And Sources Are Both Accepted
- **WHEN** a fusion config defines both `targets` and `sources`
- **THEN** the system SHALL treat all listed datasets as training datasets (no target/source semantic differences).

### Requirement: Dataset Entry Schema + Template Validation
Each dataset entry SHALL include:
- `dataset` (string; dataset wrapper key)
- `train_jsonl` (string path)
- `template` (string; MUST be a known template ID in CoordExp)

Each dataset entry MAY include:
- `name` (string; optional identifier; if omitted, the dataset ID defaults to `dataset`)
- `val_jsonl` (string path or null; null/missing means "skip eval for this dataset")
- `ratio` (float; defaults to `1.0`)

Dataset IDs (defined as `name` if provided, otherwise `dataset`) SHALL be unique across all dataset entries in the effective fusion config. Duplicate dataset IDs SHALL raise an error.

Unknown `template` values SHALL raise an error.

#### Scenario: Unknown Template Errors
- **WHEN** a dataset entry sets `template: some_unknown_template`
- **THEN** fusion config loading SHALL fail with a clear error message.

### Requirement: Extends Merge Semantics
When using `extends`, dataset entries SHALL be merged by dataset ID (use `name` if provided, otherwise `dataset`).
- If both base and override define the same dataset ID, the effective entry SHALL be deep-merged (override keys take precedence).
- Dataset ordering SHALL be preserved (base order first, then new override-only entries appended).

#### Scenario: Extends Merges Entries By Dataset ID
- **WHEN** a fusion config uses `extends` and both base and override define the same dataset ID
- **THEN** the effective dataset entry SHALL be the deep-merged mapping.

### Requirement: Per-Dataset Ratio Quotas (No Target/Source Semantics)
For every dataset entry in the fusion config, the system SHALL compute a per-epoch quota:
`quota_i = round(len(pool_i) * ratio_i)` where `ratio_i` defaults to `1.0`.

#### Scenario: Ratio Downsamples
- **WHEN** a dataset entry has `ratio: 0.1`
- **THEN** the dataset SHALL contribute approximately 10% of its pool per epoch (rounded).

### Requirement: Eval Dataset Uses Any Non-Null val_jsonl
The eval dataset built from a fusion config SHALL include all datasets whose `val_jsonl` is a non-null path.
Datasets with `val_jsonl: null` (or missing `val_jsonl`) SHALL be excluded from eval.

#### Scenario: val_jsonl Null Skips Eval For That Dataset
- **WHEN** a dataset entry sets `val_jsonl: null`
- **THEN** that dataset SHALL NOT contribute to the eval dataset.

### Requirement: Dense-Caption Only (v1)
Fusion datasets in CoordExp SHALL run in dense-caption mode. Other task types (e.g., referring/grounding-only datasets) are out of scope for this change.

#### Scenario: Fusion Training Uses Dense Caption Encoding
- **WHEN** fusion is enabled for training
- **THEN** samples SHALL be encoded using the dense-caption data path.

### Requirement: Compatibility With Coord-Token Mode And Packing
Fusion datasets SHALL remain compatible with CoordExp defaults:
- coord-token supervision (`custom.coord_tokens.enabled: true`)
- dataset packing wrapper (`training.packing: true`)

#### Scenario: Packed Fusion Dataset Iteration
- **WHEN** `training.packing: true` and fusion is enabled
- **THEN** the packed dataset iterator SHALL yield packed groups of encoded samples
- **AND** no fusion-specific runtime error SHALL occur during dataset iteration.
