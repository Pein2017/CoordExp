## 1. Implementation
- [x] 1.1 Remove `line`/`line_points` from prompts and user-facing docs:
  - [x] Update `src/config/prompts.py` to remove all `line` mentions and examples.
  - [x] Update `docs/DATA_JSONL_CONTRACT.md`, `README.md`, and `AGENTS.md` to state the contract is `bbox_2d|poly` only.
- [x] 1.2 Remove `line` from shared schemas and geometry utilities:
  - [x] Update `src/common/schemas.py` and `src/common/coord_standardizer.py` to drop `line` and `line_points`.
  - [x] Update `src/common/geometry/*` to remove polyline helpers (e.g., `line_to_bbox`) and exports.
- [x] 1.3 Remove `line` handling from dataset pipeline:
  - [x] Update `src/datasets/utils.py`, `src/datasets/geometry.py`, `src/datasets/builders/jsonlines.py` to remove polyline support and fail fast if `line` is present.
  - [x] Update preprocessors/augmentation (`src/datasets/preprocessors/*`, `src/datasets/augmentation/*`) to remove polyline branches and config knobs (e.g., `skip_if_line`).
- [x] 1.4 Remove `line` support from inference/eval/vis:
  - [x] Update parsers and validators to accept only `bbox_2d|poly` (`src/eval/parsing.py`, `src/infer/engine.py`, `vis_tools/vis_coordexp.py`, etc.).
  - [x] Ensure evaluators do not reference `line` and treat any `line` as invalid geometry with a clear counter.
- [x] 1.5 Update tests/fixtures:
  - [x] Remove/adjust tests that include `line` geometry (e.g., coord-token augmentation roundtrip) to cover `bbox_2d|poly` only.
  - [x] Remove config examples that mention `skip_if_line` (e.g., `configs/debug.yaml`).

## 2. Specs (OpenSpec)
- [x] 2.1 Update OpenSpec specs to reflect `bbox_2d|poly` only:
  - [x] `coord-token-mode`
  - [x] `coord-utils`
  - [x] `detection-evaluator`
  - [x] `inference-engine`
  - [x] `inference-pipeline`

## 3. Validation
- [x] 3.1 Run `openspec validate 2026-01-19-remove-line-geometry --strict`.
- [x] 3.2 Run unit tests (`conda run -n ms python -m pytest -q`), at least covering datasets + eval parsing paths.

