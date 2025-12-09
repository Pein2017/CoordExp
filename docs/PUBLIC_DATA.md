# Public Data Module (`public_data/`) Overview

This document introduces the **public data** submodule under `public_data/` at the repo root.

The goal of this module is to provide **geometry-aware, tested pipelines** for turning public detection/segmentation datasets (starting with **LVIS**, expanding to COCO/Objects365/OpenImages) into JSONL files that match the CoordExp training contract and can be used as **primary datasets** in training. Multi-dataset fusion is currently disabled.

---

## Scope and Responsibilities

`public_data/` is a self-contained mini-project focused on:

- **Dataset engineering** for public vision datasets (LVIS now; Objects365 / Open Images later).
- **Geometry-aware conversion** from source formats (e.g., COCO-style bbox + segmentation) to Qwen3-VL JSONL with:
  - `bbox_2d`: `[x1, y1, x2, y2]` in **pixel coordinates**.
- `poly`: `[..., xn, yn]` + `poly_points` for N-point polygons. Simplify polygons during conversion (e.g., `--poly-max-points N` to downgrade oversized polygons to `bbox_2d`); the dataloader consumes geometry as written.
- **Validation & tests** to catch schema or geometry errors early.
- **Visualization tools** to visually inspect bounding boxes and polygons.

`public_data/` deliberately does **not** contain training code; it produces JSONL files that are then consumed by the main training stack under `configs/` and `src/`.

---

## How `public_data/` Fits into Qwen3-VL

At the project level, `public_data/` plays three roles:

- **Producer of training data**: converts public datasets (LVIS now; COCO/Objects365 next) into JSONL that matches the CoordExp dense-caption schema.
- **Geometry bridge**: exposes `bbox_2d` and N-point polygon (`poly` + `poly_points`) geometries in **pixel space**, ready for downstream normalization to `norm1000` in templates.
- **Quality gate**: provides tests and validation scripts to catch schema / geometry issues before training.

In training configs under `configs/`, these JSONL files are referenced via `custom.train_jsonl` / `custom.val_jsonl`. Multi-dataset fusion is disabled for now; ignore `fusion_config` references in older docs.

---

## Where to Find Operational Details

This document is a **high-level overview** for the main repo. For concrete commands, directory layout, and troubleshooting, see:

- `public_data/README.md` – the single source of truth for:
  - LVIS download and conversion commands
  - JSONL schema details (bbox + polygon)
  - Sampling / validation workflows
  - Integration examples and common issues

As new public datasets are added (Objects365, Open Images, ...), they should follow the same pattern inside `public_data/`, with this file remaining the entry point for how the submodule relates to the rest of Qwen3-VL.

## Smart-resize (shared preprocessor)

- `public_data/scripts/convert_lvis.py --smart-resize` invokes the shared `SmartResizePreprocessor` (pixel budget + grid alignment) to rewrite images and geometry. Outputs default to `public_data/lvis/resized_<factor>_<blocks>/`.
- Datasets loaded by `DenseCaptionDataset` resolve relative image paths against the JSONL parent and can optionally apply the same smart-resize guard via env (`SMART_RESIZE_GUARD=true`, `SMART_RESIZE_GUARD_OUTPUT_DIR=<dir>`), keeping paths portable regardless of the working directory.

---

## Unified pipeline (all datasets)

1) **Convert raw annotations → pixel JSONL**
   - LVIS: `public_data/scripts/convert_lvis.py --split train --use-polygon --smart-resize ...`
   - COCO/Objects365: add a dataset-specific converter that emits the same JSONL contract (see `docs/DATA_JSONL_CONTRACT.md`).

2) **Smart resize (budget + grid, dataset-agnostic)**
   ```bash
   PYTHONPATH=. /root/miniconda3/envs/ms/bin/python public_data/scripts/rescale_jsonl.py \
     --input-jsonl public_data/lvis/raw/train.jsonl \
     --output-jsonl public_data/lvis/rescale_32_768/train.jsonl \
     --output-images public_data/lvis/rescale_32_768/images \
     --image-factor 32 \
     --max-pixels $((32*32*768)) \
     --min-pixels $((32*32*4)) \
     --num-workers 8 \
     --relative-images
   ```

3) **Tiny subset (smoke tests)**
   ```bash
   PYTHONPATH=. /root/miniconda3/envs/ms/bin/python public_data/scripts/sample_dataset.py \
     --input public_data/lvis/rescale_32_768/train.jsonl \
     --output public_data/lvis/rescale_32_768/train_tiny.jsonl \
     --num_samples 256 \
     --strategy random
   ```

4) **Coord-token conversion (strict 0–999)**
   ```bash
   PYTHONPATH=. /root/miniconda3/envs/ms/bin/python public_data/scripts/convert_to_coord_tokens.py \
     --input public_data/lvis/rescale_32_768/train.jsonl \
     --output public_data/lvis/rescale_32_768/train.coord.jsonl
   ```
   - Repeat for val and tiny splits.
   - Training: set `custom.coord_tokens.enabled: true` and `custom.coord_tokens.skip_bbox_norm: true`.

5) **One-shot wrapper (resize → tiny → coord tokens)**
   ```bash
   DATASET_JSONL=public_data/lvis/raw/train.jsonl \
   OUTPUT_ROOT=public_data/lvis/rescale_32_768 \
   FACTOR=32 MAX_BLOCKS=768 MIN_BLOCKS=4 NUM_WORKERS=8 TINY=256 \
   bash public_data/scripts/pipeline_rescale_tokenize.sh
   ```
   Outputs:
   - `${OUTPUT_ROOT}/train.jsonl`, `images/`
   - `${OUTPUT_ROOT}/train_tiny.jsonl`
   - `${OUTPUT_ROOT}/train.coord.jsonl`
   - `${OUTPUT_ROOT}/train_tiny.coord.jsonl`

6) **Validation (bbox-focused)**
   ```bash
   conda run -n ms python public_data/scripts/validate_jsonl.py \
     public_data/lvis/rescale_32_768/train.jsonl
   ```

7) **Visualization**
   ```bash
   conda run -n ms python public_data/vis_tools/visualize_lvis.py --num_samples 3 --mode both --save
   ```
