# Public Data Pipeline (Unified Runner)

This folder contains dataset preparation for public detection/grounding datasets
(currently COCO, LVIS, and Visual Genome), and shared preprocessing utilities.

The preferred interface is the unified runner:
`./public_data/run.sh <dataset> <command> ...`

## Scope & Prereqs
- Repo root: `.` (run `./public_data/run.sh` from here).
- Python env: activate the environment you want (e.g., `conda activate ms`) so `python` points to it.
- Tools: dataset-dependent (`wget`, `unzip`, etc.); disk requirements depend on dataset.

## Directory Layout (Runner Outputs)
For a dataset id `<ds>`:

- Raw outputs (dataset plugin writes these):
  - `public_data/<ds>/raw/train.jsonl`
  - `public_data/<ds>/raw/val.jsonl` (optional; if the dataset defines a val split)
  - `public_data/<ds>/raw/images/...` (typical; dataset-specific)

- Preset outputs (shared stages write these):
  - `public_data/<ds>/<preset>/train.jsonl` (pixel-space canonical artifact)
  - `public_data/<ds>/<preset>/val.jsonl` (optional)
  - `public_data/<ds>/<preset>/train.norm.jsonl` (norm1000 integer artifact)
  - `public_data/<ds>/<preset>/val.norm.jsonl` (optional)
  - `public_data/<ds>/<preset>/train.coord.jsonl` (coord-token artifact)
  - `public_data/<ds>/<preset>/val.coord.jsonl` (optional)
  - `public_data/<ds>/<preset>/images/...`

Image paths in JSONL are a contract requirement: they MUST be relative to the JSONL directory
(`docs/data/JSONL_CONTRACT.md`).

## Quick Start (Unified Runner)

Show help / command grammar:
```bash
./public_data/run.sh <dataset> help
# Example:
./public_data/run.sh lvis help
```

LVIS end-to-end (download -> convert -> rescale -> coord -> validate):
```bash
./public_data/run.sh lvis all --preset rescale_32_768_bbox
```

Visual Genome end-to-end (passes dataset-specific args after `--`):
```bash
./public_data/run.sh vg all --preset rescale_32_768_bbox -- --objects-version 1.2.0
```

Validate only raw JSONL structure (no images required):
```bash
./public_data/run.sh lvis validate --raw-only --skip-image-check
```

Tune shared preprocessing options (run steps separately; args after `--` go to the python scripts):
```bash
./public_data/run.sh <dataset> rescale --preset <preset> -- --image-factor 32 --max-pixels $((32*32*768))
./public_data/run.sh <dataset> coord   --preset <preset>
./public_data/run.sh <dataset> validate --preset <preset> --skip-image-check
```

Notes:
- For `all`, args after `--` are forwarded only to dataset-specific steps (`download`/`convert`) to avoid ambiguity.
- `validate` also runs a best-effort prompt/template sanity check on `train.coord.jsonl` (it warns+skips if no cached model is available under `model_cache/`).

## Unified Internal Architecture
`public_data/run.sh` keeps the same external command grammar, while shared internals are now routed through a modular pipeline/factory:

- Source adapter layer:
  - registry-based adapters for `coco`, `lvis`, `vg`, and `vg_ref`
  - adapter boundary owns source-ingestion normalization only
- Deterministic shared stage plan:
  - always-on structural preflight checks
  - smart-resize
  - optional max-object filtering
  - norm1000 numeric normalization
  - coord-token expansion
  - writer + manifest emission
  - optional in-plan validation stage
- Output layer:
  - standardized `*.jsonl`, `*.norm.jsonl`, `*.coord.jsonl`
  - relative image-path preservation
  - reproducibility manifest: `pipeline_manifest.json`

## Max-Object Filtering (`max{N}`)
Max-object filtering is **off by default**.

Enable it during the **coord step** by setting:
```bash
PUBLIC_DATA_MAX_OBJECTS=60 ./public_data/run.sh <dataset> coord --preset <preset>
```

Behavior:
- Filtering policy is drop-only: samples with `len(objects) > N` are removed (never truncated).
- Effective preset naming appends canonical suffix `_max{N}` (for example `rescale_32_768_bbox_max60`).
- Legacy `_max_<N>` naming is rejected with an actionable rename/rebuild hint.
- Strict fail-fast: using `PUBLIC_DATA_MAX_OBJECTS` with `rescale`, `validate`, or `all` is rejected. Use two-step flow (`all`/`rescale` first, then `coord` with max filtering).

## Rescale Safety (Fail-Fast)
- `rescale`/`full` require a **fresh preset target**.
- If target preset already contains prior artifacts (for example `images/`, `train.jsonl`, `pipeline_manifest.json`), execution fails fast.
- To rebuild, use a new preset name or deliberately delete the entire preset directory first.

## Dataset Plugins (Download/Convert)
Datasets implement a small executable plugin contract:
`public_data/datasets/<dataset>.sh`

The runner executes the plugin (it does NOT `source` it). Plugins receive required paths/settings via explicit flags and MUST NOT rely on runner-exported environment variables for those values.

Existing plugins:
- `public_data/datasets/lvis.sh`
- `public_data/datasets/vg.sh`
- `public_data/datasets/vg_ref.sh`
- `public_data/datasets/coco.sh`

Tip: standard proxy env vars like `http_proxy`/`https_proxy` are still honored by underlying tools (wget/huggingface), but they are not part of the runner/plugin *contract*.

## LVIS Geometry Ablations (BBox-only vs Poly-prefer)
In addition to the unified runner, we maintain **dataset-fixed** LVIS exports for
geometry-format ablations:
- `bbox_only`: every instance emits `bbox_2d`
- `poly_prefer_semantic`: prefer `poly` (single segment) when feasible; fallback to `bbox_2d`
  when the visible-mask polygon is semantically unreliable (occlusion edge cases) or a capped poly
  cannot be formed.

Reproducer script:
```bash
# (On a fresh machine/node, run `./public_data/run.sh lvis download` first.)
bash public_data/scripts/export_lvis_bbox_poly_prefer_semantic_max60.sh
```

Details, rationale, and output paths:
- `public_data/lvis/README.md`

## JSONL Contract (Produced Here)
See `docs/data/JSONL_CONTRACT.md` for the authoritative schema.

Produced records look like:
```json
{
  "images": ["images/000000000001.jpg"],
  "objects": [
    {"bbox_2d": [x1, y1, x2, y2], "desc": "person"},
    {"poly": [x1, y1, ..., xn, yn], "poly_points": n, "desc": "car"}
  ],
  "width": 640,
  "height": 480
}
```
- `*.jsonl` uses pixel coordinates.
- `*.norm.jsonl` uses normalized integers in [0,999].
- `*.coord.jsonl` uses `<|coord_k|>` tokens.
- Exactly one geometry per object (`bbox_2d` OR `poly`).

## Validation
### Validator CLI
`public_data/scripts/validate_jsonl.py` validates the contract for both `bbox_2d` and `poly`:
```bash
python public_data/scripts/validate_jsonl.py public_data/<ds>/raw/train.jsonl
python public_data/scripts/validate_jsonl.py public_data/<ds>/<preset>/train.coord.jsonl --skip-image-check
```

Behavior:
- Enforces `images[0]` is a relative path (errors on absolute paths).
- Validates geometry values are numeric OR coord tokens (`<|coord_k|>` with k in 0..999).
- Malformed/out-of-range coord tokens are reported as validation errors (the validator must not crash).
- Rejects legacy/unsupported geometry keys (`bbox`, `polygon`, `line`, `line_points`).
- Bounds image-open checks with deterministic first-N sampling per file (default `N=64`) instead of opening every image.

### Runner Validation
`./public_data/run.sh <ds> validate ...` runs the validator for raw and/or preset artifacts and (best-effort) runs `scripts/tools/inspect_chat_template.py` on a coord-token JSONL sample.

## Tests
LVIS converter tests (no images required):
```bash
bash public_data/tests/run_tests.sh
```

## COCO 2017 (Instances / 80 classes)
Reproducer and details:
- `public_data/coco/README.md`

## Extra Utilities
- COIG-CQIA (ModelScope) downloader: `public_data/scripts/download_coig_cqia.py`
- JSONL formatting: `public_data/scripts/format_coig_cqia.py`
- Sampling: `public_data/scripts/sample_dataset.py`
- Merging JSONLs: `public_data/scripts/merge_jsonl.py` (rewrites relative image paths to keep them resolvable)

## Troubleshooting
- If `validate` fails on missing images, re-run with `--skip-image-check` to validate structure first.
- If `all` fails because `scripts/tools/inspect_chat_template.py` cannot run (no cached model), it will warn+skip; the JSONL contract validation still runs.
- If `validate` fails on `*.coord.jsonl` with `x2 <= x1` / `y2 <= y1`, it is typically caused by very thin (1px) boxes collapsing under norm1000 quantization in older coord conversion logic. Re-run `./public_data/run.sh <ds> coord --preset <preset>` after updating `public_data/scripts/convert_to_coord_tokens.py`.
- Disk pressure: use `public_data/scripts/sample_dataset.py` to produce smaller JSONLs before training.
- Migration-time dataset rollback (operator, parity-gate failure):
  1. Find the last parity-passing commit for the affected dataset (`<legacy_sha>`).
  2. Create a dataset-scoped rollback worktree: `git worktree add temp/rollback-<ds> <legacy_sha>`.
  3. Re-run only that dataset from the rollback worktree (for example `./public_data/run.sh <ds> all --preset <preset>`).
  4. Keep other datasets on the current branch; only the failing dataset is temporarily pinned to legacy internals.
  5. Remove the temporary worktree after parity is restored: `git worktree remove temp/rollback-<ds>`.
- Full-cutover rollback (operator): if a regression is discovered after global cutover, revert the cutover commit in VCS (for example `git revert <cutover_sha>`) to restore legacy behavior globally.
