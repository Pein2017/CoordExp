## Why

The current `cxcy_logw_logh` experiment mixes canonical `xyxy` data, runtime
dataset-side conversion, builder-side conversion, prompt branching, cache
fingerprinting, and inference-time decode into one heavily branched path. The
recent double-conversion failure shows that this layering is too subtle for a
research-critical geometry contract and can silently corrupt training targets
while leaving text behavior apparently healthy.

We need a simpler and safer architecture: generate the relevant bbox-format
dataset artifacts offline inside `public_data/`, keep canonical raw data
explicit, and require training to consume a fully separated prepared-data
branch instead of reconstructing model-facing geometry online.

## What Changes

- Add a first-class offline preprocessing capability that derives
  bbox-format-specific training artifacts from canonical raw/prepared datasets,
  with initial supported non-canonical branches:
  - `cxcy_logw_logh = [cx, cy, logw, logh]`
  - `cxcywh = [cx, cy, w, h]`
- Define a fully separate prepared-data workflow branch for non-canonical
  bbox parameterizations so `xyxy`, `cxcy_logw_logh`, and `cxcywh` datasets are
  generated, stored, and referenced independently from the start.
- Keep canonical raw and evaluation-facing artifacts in `xyxy`, but require
  model-facing non-canonical training datasets to be authored offline in
  `public_data/` rather than synthesized inside the runtime dataset/builder
  path.
- Require each non-canonical prepared-data branch to export split-local JSONL
  artifacts for both train and val when those splits exist:
  - `train.jsonl` / `val.jsonl` for the offline-prepared norm1000 integer branch,
  - `train.coord.jsonl` / `val.coord.jsonl` for the tokenized training branch,
  - plus branch provenance metadata.
- Scope the first offline branch to `bbox_2d`-only presets and fail fast on
  `poly` or mixed-geometry sources instead of carrying forward the current
  mixed-geometry branching.
- **BREAKING**: remove the supported Stage-1 online bbox-format conversion path
  for non-canonical bbox formats; configs that request `cxcy_logw_logh` or
  `cxcywh` must point to a matching offline-prepared dataset branch and MUST
  fail fast when the dataset provenance or geometry contract does not match.
- Add strict safety guards and provenance:
  - prepared dataset metadata must record canonical source artifact, emitted
    bbox format, coord-token contract, and conversion version,
  - prepared dataset metadata must also record the model-facing slot-order
    contract so each branch remains unambiguous, including:
    - `cxcy_logw_logh = [cx, cy, logw, logh]`
    - `cxcywh = [cx, cy, w, h]`,
  - training/runtime must reject mixed or ambiguous surfaces,
  - cache identity and run metadata must include prepared bbox-format branch
    provenance.
- Refactor the public-data pipeline so raw dataset generation and derived
  bbox-format artifact generation are explicit offline stages rather than
  implicit runtime behavior.

## Capabilities

### New Capabilities
- `offline-bbox-format-branches`: defines offline derivation, naming,
  provenance, and safety rules for prepared bbox-format-specific dataset
  branches such as `cxcy_logw_logh` and `cxcywh`.

### Modified Capabilities
- `public-data-pipeline`: add a first-class offline bbox-format derivation
  branch and deterministic artifact workflow for prepared datasets.
- `public-data-adapter-factory`: clarify that adapters emit canonical raw
  records and that non-canonical bbox parameterization derivation happens only
  in shared offline preprocessing stages.
- `coord-aux-loss`: require Stage-1 non-canonical bbox-format training
  (`cxcy_logw_logh`, `cxcywh`) to consume offline-prepared datasets and reject
  runtime bbox-format conversion.
- `encoded-training-cache`: require cache provenance/fingerprints to capture the
  prepared bbox-format branch identity and reject ambiguous mixed surfaces.

## Impact

- Affected systems:
  - `public_data/` runner, planners, dataset plugins, and writers
  - dataset prep scripts that currently stop at canonical `xyxy` / coord-token
    artifacts
  - Stage-1 dataset rendering and builder paths in `src/datasets/`
  - training config validation and cache provenance surfaces
- Expected code touch points include:
  - `public_data/run.sh`
  - `public_data/scripts/`
  - `src/datasets/dense_caption.py`
  - `src/datasets/builders/jsonlines.py`
  - `src/config/schema.py`
  - `src/config/loader.py`
  - encoded-sample cache helpers and run metadata writers
- Reproducibility impact:
  - bbox-format conversion becomes explicit, offline, and auditable
  - runtime branching and redundant conversion logic are reduced
- research comparisons between `xyxy`, `cxcy_logw_logh`, and `cxcywh` become
    cleaner because each run starts from a distinct prepared-data branch
