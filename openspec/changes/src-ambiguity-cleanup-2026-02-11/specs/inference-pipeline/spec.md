# inference-pipeline Spec Delta

This is a delta spec for change `src-ambiguity-cleanup-2026-02-11`.

## ADDED Requirements

### Requirement: Image-path resolution logic is shared across infer stages
Pipeline stages that resolve image paths for reading (inference engine, visualization) SHALL reuse shared image-path resolution helpers.

The implementation MUST preserve each stage’s intended strictness:
- inference SHOULD resolve paths best-effort (environment/config root + JSONL-relative fallback),
- visualization SHOULD resolve paths strictly (returning no-path / skipping when the image cannot be found).

#### Scenario: ROOT_IMAGE_DIR is respected when resolving relative paths
- **GIVEN** a relative image path `images/0001.jpg`
- **AND** the environment variable `ROOT_IMAGE_DIR` points to a dataset root
- **WHEN** the infer stage resolves the image path for loading
- **THEN** it resolves under `ROOT_IMAGE_DIR` rather than the JSONL directory.

#### Scenario: Visualization skips missing images without crashing
- **GIVEN** a record whose `image` path cannot be found under any resolution rule
- **WHEN** the visualization stage runs
- **THEN** it skips that record/image deterministically without crashing.
