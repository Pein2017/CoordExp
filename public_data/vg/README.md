# Visual Genome (VG)

This folder documents how to prepare **Visual Genome** into the CoordExp
JSONL contract (`docs/DATA_JSONL_CONTRACT.md`) using the unified public-data
runner (`public_data/run.sh`).

We support two closely related dataset flavors:
- `vg` (**objects**): `desc = object name`, `bbox_2d = object box`
- `vg_ref` (**region phrases**): `desc = region phrase`, `bbox_2d = region box`

VG does **not** ship official train/val/test splits. We generate deterministic
splits from `image_id` (configurable; see below).

## Quick start (recommended)

From the repo root:
```bash
# Download + convert (objects) -> smart-resize -> coord tokens -> validate
./public_data/run.sh vg all --preset rescale_32_768_bbox -- --objects-version 1.2.0

# Region phrases variant (reuses images from `vg` when available)
./public_data/run.sh vg_ref all --preset rescale_32_768_bbox -- --val-mod 100
```

### Important defaults (and how to change them)

- **Objects annotations version**: default `1.2.0` (pass `--objects-version 1.0.0|1.2.0` after `--`)
- **Deterministic split**: default `--val-mod 5` (~20% val; image_id % val_mod == 0)
- **Checksums**: downloads verify sha256 by default (opt-out: `--no-verify-checksums`)
- **Per-image dedupe**: exact duplicate `(desc, bbox_2d)` objects are removed by default (opt-out: `--no-dedupe-objects`)
- **Junk labels**: high-confidence placeholder labels like `this/that/it` are dropped by default (opt-out: `--no-filter-junk-descs`)

To audit how often junk labels appear in your converted JSONL, run:
```bash
conda run -n ms python public_data/vg/collect_junk_descs.py --jsonl public_data/vg/raw/train.jsonl --top-k 50
```

If you need to tune shared resize settings (pixel budget / factor), run stages
separately:
```bash
./public_data/run.sh vg download -- --objects-version 1.2.0
./public_data/run.sh vg convert  -- --objects-version 1.2.0
./public_data/run.sh vg rescale  --preset rescale_32_768_bbox -- --image-factor 32 --max-pixels $((32*32*768)) --num-workers 8
./public_data/run.sh vg coord    --preset rescale_32_768_bbox
./public_data/run.sh vg validate --preset rescale_32_768_bbox
```

## Outputs (runner layout)

For `vg` and `vg_ref`, the unified runner follows the standard layout described
in `public_data/README.md`:

- Raw (dataset plugin output):
  - `public_data/<ds>/raw/train.jsonl`
  - `public_data/<ds>/raw/val.jsonl`
  - `public_data/<ds>/raw/images/...`
  - `public_data/<ds>/raw/annotations/...`

- Preset (shared output):
  - `public_data/<ds>/<preset>/train.jsonl`
  - `public_data/<ds>/<preset>/val.jsonl`
  - `public_data/<ds>/<preset>/train.coord.jsonl`
  - `public_data/<ds>/<preset>/val.coord.jsonl`
  - `public_data/<ds>/<preset>/images/...`

## Config-first defaults

The recommended, reproducible parameters for VG preparation live in:
- `configs/public_data/vg.yaml`

This config is intended as the “single place” to record default choices
(VG objects version, split policy, filters, and resize preset).

## Smoke test (no download required)

For a tiny end-to-end run that does **not** download anything (writes synthetic
VG-style annotations + images under `temp/` and validates outputs):
```bash
PYTHONPATH=. conda run -n ms python public_data/vg/smoke_test.py
```
