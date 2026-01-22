# Design: Unified Public-Data Pipeline Runner (Shell + Dataset Plugins)

## Goals
- Provide a single, consistent entrypoint to prepare public datasets under `public_data/`:
  - dataset-specific: internet download + raw-to-contract conversion
  - shared: rescale + coord-token conversion + validation
- Keep outputs reproducible and paper-ready by standardizing directory layout and validation.
- Prefer thin wrappers around existing scripts over refactors.

## Non-Goals
- No changes to `docs/DATA_JSONL_CONTRACT.md`.
- No new training-time fusion defaults; dataset mixing remains offline (`public_data/scripts/merge_jsonl.py`).
- No attempt to unify dataset *data sources* (mirrors/auth/etc. remain dataset-specific).

## User-Facing CLI
The runner is a shell script:

```bash
./public_data/run.sh <dataset> <command> [runner-flags] [-- <passthrough-args>]
```

### Working Directory
The runner is intentionally “repo-root anchored”:
- Users run it from the CoordExp repo root (so relative paths and `PYTHONPATH=.` are consistent).
- The runner fails fast if invoked from a different directory.

### Flag Parsing Philosophy
- Parse only a small set of runner flags (`--preset`, `--conda-env`, validation toggles).
- Forward everything after `--` to the underlying step:
  - `download` / `convert` -> dataset plugin functions
  - `rescale` -> `public_data/scripts/rescale_jsonl.py`
  - `coord` -> `public_data/scripts/convert_to_coord_tokens.py`
  - `all` -> forward passthrough args to plugin steps only (download/convert) to avoid ambiguity.

This keeps dataset-specific knobs and script options available without growing a new flag surface area.

## Standard Directory Layout
For a dataset id `<ds>`:

- Raw artifacts:
  - `public_data/<ds>/raw/annotations/` (optional; dataset-specific)
  - `public_data/<ds>/raw/images/` (optional but typical)
  - `public_data/<ds>/raw/train.jsonl` (required output of convert)
  - `public_data/<ds>/raw/val.jsonl` (optional; if dataset defines val split)

- Preset outputs (shared preprocessing):
  - `public_data/<ds>/<preset>/images/`
  - `public_data/<ds>/<preset>/train.jsonl`
  - `public_data/<ds>/<preset>/val.jsonl` (optional)
  - `public_data/<ds>/<preset>/train.coord.jsonl`
  - `public_data/<ds>/<preset>/val.coord.jsonl` (optional)

This matches the contract requirement that image paths in JSONL are relative to the JSONL directory.

## Dataset Plugin Contract (Shell, Executable)
Each dataset provides an **executable** plugin script:
`public_data/datasets/<dataset>.sh`

The runner executes the plugin (it does **not** `source` it). This makes the plugin self-contained and prevents implicit coupling via exported environment variables.

### Required Subcommands
- `default-preset`
  - Prints a preset name (e.g., `rescale_32_768_bbox`) and exits 0.
  - Exits non-zero if the dataset has no default preset.
- `download`
  - Downloads internet artifacts into the dataset raw directory.
  - Exits non-zero on failure.
- `convert`
  - Reads raw artifacts and writes the CoordExp contract JSONLs:
    - `raw/train.jsonl` (required)
    - `raw/val.jsonl` (optional)
  - Exits non-zero on failure.

### Explicit Flags (No Runner Env Vars)
The runner passes all required locations/settings as explicit flags:
- `--repo-root <abs>`
- `--dataset <id>`
- `--dataset-dir <abs>`
- `--raw-dir <abs>`
- `--raw-image-dir <abs>`
- `--raw-train-jsonl <abs>`
- `--raw-val-jsonl <abs>`
- `--conda-env <name>` (default `ms`)

All args after `--` are forwarded verbatim as passthrough args to the plugin (dataset-specific options).

### Error Handling and Logging
- The runner prints stage banners and invoked commands.
- Plugins SHOULD use `set -euo pipefail` and surface progress output for long downloads (e.g., `wget` progress bars).
- Plugins SHOULD run python via `PYTHONPATH=. conda run -n <conda-env> python ...` using the passed `--conda-env`.

## Shared Steps

### Rescale
Implements:
- input: `$RAW_TRAIN_JSONL` (+ `$RAW_VAL_JSONL` if present)
- output: `$PRESET_TRAIN_JSONL` / `$PRESET_VAL_JSONL` and `$PRESET_IMAGE_DIR`

Uses the existing shared script:
- `public_data/scripts/rescale_jsonl.py`

### Coord Tokens
Implements:
- input: `$PRESET_TRAIN_JSONL` (+ `$PRESET_VAL_JSONL` if present)
- output: `$PRESET_TRAIN_COORD_JSONL` / `$PRESET_VAL_COORD_JSONL`

Uses the existing shared script:
- `public_data/scripts/convert_to_coord_tokens.py`

### Validate (Raw + Preset)
Validate is intended to prevent “silent bad data”:
- Validate raw JSONLs (`raw/train.jsonl`, optional `raw/val.jsonl`)
- Validate preset JSONLs and coord-token JSONLs for a specific preset
- Optional: run `scripts/inspect_chat_template.py` on `train.coord.jsonl` to confirm prompt/template rendering

Implementation note:
- `public_data/scripts/validate_jsonl.py` is currently bbox-only, so the runner should use (or add) a contract validator that supports both `bbox_2d` and `poly` plus coord-token values per `docs/DATA_JSONL_CONTRACT.md`.
- Support `--skip-image-check` for annotation-only workflows (e.g., validating JSONL structure before images finish downloading).

## Geometry Normalization Notes
Different sources use different geometry field names and conventions.
The unified pipeline treats the CoordExp JSONL contract as the boundary:
- Plugins normalize any source-specific keys (e.g., `bbox`, `coords`, `segmentation`) into contract keys (`bbox_2d` or `poly`).
- `bbox_2d` is the required minimum geometry that every dataset plugin must be able to emit.
- `poly` is optional and opt-in: bbox-only is the default geometry mode even when polygons exist; plugins emit polygons only when explicitly enabled (via dataset-specific passthrough options).
- If a source provides polygon-only annotations, plugins should derive `bbox_2d` from the polygon when bbox output is required.

## Example Workflows

### Visual Genome (VG)
```bash
./public_data/run.sh vg all --preset rescale_32_768_bbox -- --objects-version 1.2.0
```

### LVIS
```bash
./public_data/run.sh lvis all --preset rescale_32_768_bbox
```

### LVIS (poly opt-in)
```bash
./public_data/run.sh lvis all --preset rescale_32_768_poly_20 -- --use-polygon --poly-max-points 20
```

## Tuning Shared Steps
`all` runs shared steps (`rescale`, `coord`) with runner defaults. If you need to tune
rescale/coord options (e.g., `--image-factor`, `--max-pixels`), run those steps separately:

```bash
./public_data/run.sh <dataset> rescale --preset <preset> -- --image-factor 32 --max-pixels $((32*32*768))
./public_data/run.sh <dataset> coord --preset <preset>
```

Note: `--assume-normalized` (from `public_data/scripts/convert_to_coord_tokens.py`) is only for inputs that are already norm1000 ints/tokens
(e.g., converting `*.norm.jsonl` → `*.coord.jsonl`). It SHOULD NOT be used for the common pixel → rescale → coord flow.

## Compatibility Notes
- The unified runner standardizes *runner outputs* (under `public_data/<dataset>/raw/`), even if underlying converters have different defaults.
- The runner does not change training configs; you still point YAML `custom.train_jsonl` to the produced `*.coord.jsonl`.
