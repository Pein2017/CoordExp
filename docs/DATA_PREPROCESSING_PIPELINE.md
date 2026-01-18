# Data Preprocessing & Intake (Annotation → Train/Eval JSONL)

This is the single flow we use for all detection/grounding datasets (LVIS, COCO, Objects365, etc.) to produce:
- Pixel JSONL (train/val)
- Resized JSONL + resized images (budget + grid-aligned)
- Tiny smoke splits
- Coord-token JSONL (strict 0–999)

## End-to-end pipeline
```
Raw annotations/images
  → dataset converter (per-dataset, emits pixel JSONL)
  → smart resize (public_data/scripts/rescale_jsonl.py)
  → tiny subset (public_data/scripts/sample_dataset.py)
  → coord tokens (public_data/scripts/convert_to_coord_tokens.py)
  → training (custom.train_jsonl / custom.val_jsonl)
```

For LVIS, the converter is `public_data/scripts/convert_lvis.py`. For other datasets, add a matching converter that outputs the same JSONL contract (see docs/DATA_JSONL_CONTRACT.md), then feed it into the steps below.

## Smart resize (budget + grid)
```bash
PYTHONPATH=. /root/miniconda3/envs/ms/bin/python public_data/scripts/rescale_jsonl.py \
  --input-jsonl path/to/raw/train.jsonl \
  --output-jsonl path/to/out/train.jsonl \
  --output-images path/to/out/images \
  --image-factor 32 \
  --max-pixels $((32*32*768)) \
  --min-pixels $((32*32*4)) \
  --num-workers 8 \
  --relative-images
```
- Uses `SmartResizePreprocessor` (shared with training) to resize images + geometry together.
- Dimensions snap to `image_factor`; pixel budget enforced by `max_pixels` / `min_pixels`.
- Image paths can be relativized to the output JSONL directory.

## Tiny subset (smoke)
```bash
PYTHONPATH=. /root/miniconda3/envs/ms/bin/python public_data/scripts/sample_dataset.py \
  --input path/to/out/train.jsonl \
  --output path/to/out/train_tiny.jsonl \
  --num_samples 256 \
  --strategy random
```
Use `--strategy stratified` for long-tail sets like LVIS.

## Coord-token conversion (strict 0–999)
```bash
PYTHONPATH=. /root/miniconda3/envs/ms/bin/python public_data/scripts/convert_to_coord_tokens.py \
  --input path/to/out/train.jsonl \
  --output path/to/out/train.coord.jsonl
```
- Maps pixel domain `[0, w-1]/[0, h-1]` → `[0,999]`; any out-of-bounds raises (no clamping).
- Repeat for `val.jsonl` and tiny splits.
- Train with `custom.coord_tokens.enabled: true` and `custom.coord_tokens.skip_bbox_norm: true`.

## One-shot wrapper
`public_data/scripts/pipeline_rescale_tokenize.sh` runs resize → tiny → coord-token in one go. Configure via env vars:
```
DATASET_JSONL=/path/to/raw/train.jsonl
OUTPUT_ROOT=/path/to/out/rescale_32_768
FACTOR=32 MAX_BLOCKS=768 MIN_BLOCKS=4 NUM_WORKERS=8 TINY=256
bash public_data/scripts/pipeline_rescale_tokenize.sh
```
Outputs:
- `${OUTPUT_ROOT}/train.jsonl` (resized) + `images/`
- `${OUTPUT_ROOT}/train_tiny.jsonl`
- `${OUTPUT_ROOT}/train.coord.jsonl`
- `${OUTPUT_ROOT}/train_tiny.coord.jsonl`

## Quality checklist (before training)
- JSONL matches `docs/DATA_JSONL_CONTRACT.md` (width/height present; one geometry field per object).
- No coord tokens outside 0–999.
- Resized images exist and match paths in JSONL.
- Tiny splits load without errors.

## Handoff to training
- Point `custom.train_jsonl` / `custom.val_jsonl` to the resized or coord-token JSONL.
- For LVIS, we typically train on the **filtered** JSONLs:
  - `public_data/lvis/rescale_32_768_poly_20/train.filtered_max100_dense50_u3_t095.coord.jsonl`
  - `public_data/lvis/rescale_32_768_poly_20/val.filtered_max100_dense50_u3_t095.coord.jsonl`
- Multi-dataset training:
  - Preferred: merge JSONLs offline (see `public_data/scripts/merge_jsonl.py`).
  - Optional (legacy/experimental): set `custom.fusion_config` to a fusion YAML/JSON (see `docs/data/FUSION_DATASET.md`).
