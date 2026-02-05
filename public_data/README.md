# Public Data Pipeline (Unified Runner)

This folder contains dataset preparation for public detection/grounding datasets
(currently LVIS + Visual Genome), and shared preprocessing utilities.

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

- Preset outputs (shared steps write these):
  - `public_data/<ds>/<preset>/train.jsonl`
  - `public_data/<ds>/<preset>/val.jsonl` (optional)
  - `public_data/<ds>/<preset>/images/...`
  - `public_data/<ds>/<preset>/train.coord.jsonl`
  - `public_data/<ds>/<preset>/val.coord.jsonl` (optional)

Image paths in JSONL are a contract requirement: they MUST be relative to the JSONL directory
(`docs/data/JSONL_CONTRACT.md`).

## Quick Start (Unified Runner)

Show help / command grammar:
```bash
./public_data/run.sh help
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

## Dataset Plugins (Download/Convert)
Datasets implement a small executable plugin contract:
`public_data/datasets/<dataset>.sh`

The runner executes the plugin (it does NOT `source` it). Plugins receive required paths/settings via explicit flags and MUST NOT rely on runner-exported environment variables for those values.

Existing plugins:
- `public_data/datasets/lvis.sh`
- `public_data/datasets/vg.sh`
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
- Pixel coordinates (not normalized); templates normalize to norm1000 at encode time.
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

### Runner Validation
`./public_data/run.sh <ds> validate ...` runs the validator for raw and/or preset artifacts and (best-effort) runs `scripts/inspect_chat_template.py` on a coord-token JSONL sample.

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
- If `all` fails because `scripts/inspect_chat_template.py` cannot run (no cached model), it will warn+skip; the JSONL contract validation still runs.
- If `validate` fails on `*.coord.jsonl` with `x2 <= x1` / `y2 <= y1`, it is typically caused by very thin (1px) boxes collapsing under norm1000 quantization in older coord conversion logic. Re-run `./public_data/run.sh <ds> coord --preset <preset>` after updating `public_data/scripts/convert_to_coord_tokens.py`.
- Disk pressure: use `public_data/scripts/sample_dataset.py` to produce smaller JSONLs before training.
