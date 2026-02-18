## Why

The current `public_data` flow is functionally useful but structurally fragmented: dataset-specific logic is split across shell plugins and large one-off scripts, while shared transforms are partially duplicated. This increases maintenance cost, makes behavior drift more likely across datasets (`lvis`, `coco`, `vg`), and slows down onboarding new public sources.

We need an official, spec-backed refactor that introduces a modular adapter/factory pipeline while preserving the existing JSONL contract and Qwen3-VL coordinate-token compatibility.

## What Changes

- Introduce a registry/factory abstraction for public dataset ingestion that instantiates dataset adapters by id (`lvis`, `coco`, `vg`, extensible).
- Define a pluggable adapter interface for source-specific responsibilities:
  - raw image/annotation download,
  - source parsing,
  - source-level normalization into canonical intermediate records.
- Refactor shared post-ingestion processing into composable pipeline stages:
  - image extraction/path normalization,
  - smart resize enforcement under pixel budget (`32 * 32 * 768`),
  - optional max-object filtering stage (configurable, e.g., `max_objects=60`),
  - coordinate normalization/token expansion,
  - standardized output writing with explicit artifact/coordinate-space mapping:
    - `train.raw.jsonl` / `val.raw.jsonl` (pixel-space),
    - `train.norm.jsonl` / `val.norm.jsonl` (norm1000 ints),
    - `train.coord.jsonl` / `val.coord.jsonl` (coord tokens).
- Extract reusable output formatting/writing utilities so path-relativization, manifest/stats, and deterministic ordering are centrally owned.
- Keep current CLI entrypoints and reproducible directory conventions stable during migration (`public_data/run.sh`, `public_data/<dataset>/raw`, `public_data/<dataset>/<preset>`).
- Keep max-object suffixing reproducible while avoiding path forks with existing artifacts:
  - canonical token remains `max_{N}`,
  - legacy `max{N}` names are recognized for compatibility and reuse.
- Add parity tests and migration guards so refactor is behavior-preserving for existing pipelines unless explicitly configured otherwise.

## Capabilities

### New Capabilities
- `public-data-adapter-factory`: Registry-based dataset adapter creation and composable stage pipeline execution for public data preparation.

### Modified Capabilities
- `public-data-pipeline`: Expand the existing unified runner/pipeline contract to include factory-owned orchestration, adapter plugin contracts, and first-class configurable transform stages without hard-coded dataset logic in core orchestration.

## Impact

- Affected code areas:
  - `public_data/run.sh`
  - `public_data/datasets/*.sh`
  - `public_data/converters/*.py`
  - `public_data/scripts/*.py` (converter/rescale/filter/coord/validate paths)
  - new modular pipeline package under `public_data/` (exact module layout finalized in design/tasks)
  - tests under `public_data/tests/`
- No training contract changes are intended:
  - output remains compliant with `docs/data/JSONL_CONTRACT.md`,
  - Qwen3-VL coord-token-expanded artifacts remain first-class outputs.
- Reproducibility requirements remain strict:
  - deterministic artifact paths,
  - stable stage ordering,
  - explicit config for any optional filtering behavior.
