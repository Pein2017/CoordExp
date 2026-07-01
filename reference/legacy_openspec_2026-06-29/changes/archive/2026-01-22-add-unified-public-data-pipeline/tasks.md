## 1. Implementation
- [x] Add `public_data/run.sh` unified entrypoint with subcommands (`download`, `convert`, `rescale`, `coord`, `validate`, `all`, `help`).
- [x] Enforce that `public_data/run.sh` is executed from the repo root (fail fast with a helpful message otherwise).
- [x] Define the shell dataset plugin contract (`public_data/datasets/<dataset>.sh`) as an executable script (required subcommands + required flags), and create `public_data/datasets/vg.sh` and `public_data/datasets/lvis.sh`.
- [x] Update dataset plugin contract to avoid runner-exported environment variables (execute plugins with explicit flags instead of sourcing).
- [x] Implement preset resolution for `all|validate`: `--preset` overrides plugin default preset; error if neither is available.
- [x] Ensure the runner uses `conda run -n <env> python ...` (default `ms`) and supports explicit `--conda-env`.
- [x] Implement consistent logging (stage banners + invoked commands) and fail-fast behavior (`set -euo pipefail`).
- [x] Wire `rescale` and `coord` steps to shared scripts:
  - `public_data/scripts/rescale_jsonl.py`
  - `public_data/scripts/convert_to_coord_tokens.py`
- [x] Add `validate` step that:
  - validates BOTH raw and preset outputs by default (support `--raw-only` and `--preset-only`)
  - supports `--skip-image-check`
  - validates both `bbox_2d` and `poly` geometries per `docs/data/JSONL_CONTRACT.md`
  - runs `scripts/inspect_chat_template.py --index 0` for `.coord.jsonl` outputs as a fast prompt/template sanity check.

## 2. Documentation
- [x] Add a short guide in `docs/data/INTAKE_PIPELINE.md` (and `public_data/README.md`) documenting the unified runner commands.
- [x] Update `docs/data/VISUAL_GENOME.md` to reference the unified runner as the preferred path (keep direct-script instructions as fallback).

## 3. Tests / Validation
- [x] Add a lightweight smoke test that runs the runner against the existing `public_data/vg/sample/` and a tiny LVIS sample JSONL (no internet required) to verify:
  - command parsing
  - output paths
  - rescale + coord-token conversion calls succeed
- [x] Run `openspec validate 2026-01-22-add-unified-public-data-pipeline --strict`.
