# inference-pipeline Spec Delta

This is a delta spec for change `2026-02-11-src-ambiguity-cleanup`.

## ADDED Requirements

This change is the authoritative helper-contract delta for image-root resolution precedence/provenance across active overlaps.

### Requirement: Image-path resolution logic is shared across infer stages
Pipeline stages that resolve image paths for reading (inference engine, visualization) SHALL reuse shared image-path resolution helpers.

The implementation MUST preserve each stage’s intended strictness:
- inference SHOULD resolve paths best-effort (environment/config root + JSONL-relative fallback),
- visualization SHOULD resolve paths strictly (returning no-path / skipping when the image cannot be found).

Root image-dir precedence SHALL be explicit and deterministic:
- `ROOT_IMAGE_DIR` (env override) >
- `run.root_image_dir` (config root) >
- `infer.gt_jsonl` parent (`gt_parent`) >
- `none` (no resolved root).

The chosen root decision SHALL be recorded in `resolved_config.json` using:
- `root_image_dir`
- `root_image_dir_source` in `{env, config, gt_parent, none}`.

#### Scenario: ROOT_IMAGE_DIR is respected when resolving relative paths
- **GIVEN** a relative image path `images/0001.jpg`
- **AND** the environment variable `ROOT_IMAGE_DIR` points to a dataset root
- **WHEN** the infer stage resolves the image path for loading
- **THEN** it resolves under `ROOT_IMAGE_DIR` rather than the JSONL directory.

#### Scenario: Visualization skips missing images without crashing
- **GIVEN** a record whose `image` path cannot be found under any resolution rule
- **WHEN** the visualization stage runs
- **THEN** it skips that record/image deterministically without crashing.

#### Scenario: Infer/eval/vis consume one resolved root decision
- **GIVEN** a pipeline run with relative image paths
- **WHEN** root resolution is computed once for the run
- **THEN** infer/eval/vis use the same resolved root decision
- **AND** `resolved_config.json` preserves both root path and root source for reproducibility.
