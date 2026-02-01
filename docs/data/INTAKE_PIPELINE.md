# Data Preprocessing & Intake Pipeline

This is the unified flow used for all detection/grounding datasets (LVIS, COCO, Objects365, etc.) to produce:
- Pixel JSONL (train/val)
- Resized JSONL + resized images (budget + grid-aligned)
- Tiny smoke splits
- Coord-token JSONL (strict 0–999)

---

## End-to-end Pipeline Overview

```
Raw annotations/images
  → dataset converter (per-dataset, emits pixel JSONL)
  → smart resize (public_data/scripts/rescale_jsonl.py)
  → tiny subset (public_data/scripts/sample_dataset.py)
  → coord tokens (public_data/scripts/convert_to_coord_tokens.py)
  → (optional) image-level filter (public_data/scripts/filter_low_diversity_images.py)
  → training (custom.train_jsonl / custom.val_jsonl)
```

For LVIS, the converter is `public_data/scripts/convert_lvis.py`. For other datasets, add a matching converter that outputs the same JSONL contract (see [`JSONL_CONTRACT.md`](JSONL_CONTRACT.md)).

---

## Unified Runner (Recommended for Public Data)

The preferred way to prepare public datasets is the unified runner:

```bash
./public_data/run.sh <dataset> <command> [runner-flags] [-- <passthrough-args>]
```

**Key Commands**:
- `all`: download + convert + rescale + coord + validate (using a preset).
- `validate`: check annotation structure and optionally image presence.

**Examples**:
```bash
# VG (download+convert+rescale+coord+validate)
./public_data/run.sh vg all --preset rescale_32_768_bbox

# LVIS polygons (segmentation → poly)
./public_data/run.sh lvis all --preset rescale_32_768_poly_20 -- --use-polygon --poly-max-points 20
```

---

## Manual Pipeline Steps

### 1. Smart Resize (Budget + Grid)

```bash
PYTHONPATH=. conda run -n ms python public_data/scripts/rescale_jsonl.py \
  --input-jsonl path/to/raw/train.jsonl \
  --output-jsonl path/to/out/train.jsonl \
  --output-images path/to/out \
  --image-factor 32 \
  --max-pixels $((32*32*768)) \
  --min-pixels $((32*32*4)) \
  --relative-images
```
- Uses `SmartResizePreprocessor` to resize images and geometry together.
- Dimensions snap to `image_factor`; pixel budget enforced by `max_pixels`.

### 2. Tiny Subset (Smoke Tests)

```bash
PYTHONPATH=. conda run -n ms python public_data/scripts/sample_dataset.py \
  --input path/to/out/train.jsonl \
  --output path/to/out/train_tiny.jsonl \
  --num_samples 256 \
  --strategy random
```
Use `--strategy stratified` for long-tail datasets like LVIS.

### 3. Coord-Token Conversion (Strict 0–999)

```bash
PYTHONPATH=. conda run -n ms python public_data/scripts/convert_to_coord_tokens.py \
  --input path/to/out/train.raw.jsonl \
  --output-norm path/to/out/train.jsonl \
  --output-tokens path/to/out/train.coord.jsonl
```
- Converts pixel coords into **norm1000 integer coords** (0..999) and/or `<|coord_k|>` tokens.
- Pixel -> norm scaling clamps into range and ensures `bbox_2d` stays strictly valid after rounding (no collapse).
- Train with `custom.emit_norm: none`. If you use coord tokens, also set `custom.coord_tokens.enabled: true` and keep `custom.coord_tokens.skip_bbox_norm: true`.

---

## Record-Level Filtering

### Image-Level Filtering (Semantic Diversity)

Some datasets (LVIS) contain images with many repeated instances but low semantic diversity. We filter these at the record/image level to keep the dataset semantically rich.

```bash
PYTHONPATH=. conda run -n ms python public_data/scripts/filter_low_diversity_images.py \
  --input  path/to/train.coord.jsonl \
  --output path/to/train.filtered.coord.jsonl \
  --hard_max_objects 101 \
  --min_objects 50 \
  --max_unique 3 \
  --min_top1_ratio 0.95
```
Tip: add `--stats_json output/<name>.json` to record filter statistics for reproducibility.

### Object-Count Cap (Transparency)

If you want simple, transparent control over sequence length, cap objects per image:
```bash
PYTHONPATH=. conda run -n ms python public_data/scripts/filter_jsonl_max_objects.py \
  --input  train.raw.jsonl \
  --output train.max60.raw.jsonl \
  --max-objects 60
```

---

## Automation: One-Shot Wrapper

Preferred automation is the dataset runner, which wires together download/convert/rescale/coord/validate in a preset:

```bash
# VG (download+convert+rescale+coord+validate)
./public_data/run.sh vg all --preset rescale_32_768_bbox

# LVIS polygons (segmentation -> poly)
./public_data/run.sh lvis all --preset rescale_32_768_poly_20 -- --use-polygon --poly-max-points 20
```

For LVIS, there is also a legacy single-script baseline:

```bash
# End-to-end LVIS pipeline: convert + smart-resize + coord + tiny
bash public_data/scripts/lvis_full_pipeline.sh
```

---

## Quality & Visualization

- **Validation**: `python public_data/scripts/validate_jsonl.py <path.jsonl>`
- **Visualization**: `python public_data/vis_tools/visualize_lvis.py --num_samples 3 --mode both --save`
- **Chat template inspection**: `python scripts/inspect_chat_template.py --jsonl <path> --index 0`

---

## LVIS Geometry Ablations (Dataset-Fixed)

For dataset-fixed geometry experiments (bbox-only vs polygon with semantic fallback), use:

```bash
bash public_data/scripts/export_lvis_bbox_poly_prefer_semantic_max60.sh
```

This exports both `bbox_only` and `poly_prefer_semantic` train/val JSONLs. See `public_data/lvis/README.md` for details and output paths.

---

## Quality Checklist (Before Training)

- JSONL matches `JSONL_CONTRACT.md` (width/height present; one geometry field per object).
- No coord tokens outside 0-999 (coord-token JSONLs).
- Resized images exist and match paths in JSONL.
- Tiny splits load without errors (use a quick smoke run before launching large training).

---

## Handoff to Training

- Point `custom.train_jsonl` / `custom.val_jsonl` to the resized or coord-token JSONL.
- For LVIS, pick a dataset-fixed variant that matches your ablation goal:
  - Geometry ablations: use `public_data/scripts/export_lvis_bbox_poly_prefer_semantic_max60.sh` outputs under `public_data/lvis/`.
  - Sequence-length control: apply a simple record-level `max_objects` cap (e.g., 60).
  - Optional: apply low-diversity filtering (`filter_low_diversity_images.py`) if you want to drop dense repetitive scenes.
- Multi-dataset training:
  - Preferred: merge JSONLs offline (see `public_data/scripts/merge_jsonl.py`).
  - Optional (legacy/experimental): set `custom.fusion_config` (see `docs/data/FUSION_DATASET.md`).

---

## See Also

- **JSONL Contract**: [`JSONL_CONTRACT.md`](JSONL_CONTRACT.md)
- **Public Data Submodule**: `public_data/README.md`
- **LVIS Geometry Ablations**: `public_data/lvis/README.md`
