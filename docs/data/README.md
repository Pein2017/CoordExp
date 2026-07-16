---
doc_id: docs.data.index
layer: docs
doc_type: router
status: canonical
domain: data
summary: Router for dataset contracts, offline preparation, packing, and dataset-specific notes.
tags: [data, datasets, jsonl]
updated: 2026-07-16
---

# Data & Datasets

Open this folder when you need the current dataset contract, offline preparation
workflow, packing policy, or dataset-specific intake caveats.

## Read Order

1. [CONTRACT.md](CONTRACT.md)
2. [PREPARATION.md](PREPARATION.md)
3. [PACKING.md](PACKING.md)
4. [COCO_REFINEMENT_RUNBOOK.md](COCO_REFINEMENT_RUNBOOK.md) when operating the localhost COCO-80 bbox correction workflow
5. [VISUAL_GENOME.md](VISUAL_GENOME.md) only for VG-specific split, naming, and region-phrase notes

## Page Roles

- [CONTRACT.md](CONTRACT.md)
  - authoritative JSONL, CoordJSON, geometry, norm1000, ordering, and bbox-format branch contracts
- [PREPARATION.md](PREPARATION.md)
  - offline conversion, resizing, validation, tiny splits, coord-token conversion, and handoff to training
- [PACKING.md](PACKING.md)
  - surface-specific packing support, hard-cap behavior, cache policy, and runtime tradeoffs
- [COCO_REFINEMENT_RUNBOOK.md](COCO_REFINEMENT_RUNBOOK.md)
  - loopback Label Studio setup, exact max_len12000 train/val bootstrap, Draft/batch operation, ROI profiles, recovery, validation, and runtime-only rollback
- [VISUAL_GENOME.md](VISUAL_GENOME.md)
  - Visual Genome object / region-phrase intake notes that are not already covered by the shared preparation workflow

## Current Dataset Policy

- Offline-prepared single-dataset JSONL is the default training/eval surface.
- Runtime transforms should stay minimal; image resize and geometry conversion happen offline.
- Canonical raw and preset JSONL remain model-independent `xyxy` surfaces.
- Non-canonical bbox charts such as `cxcy_logw_logh` and `cxcywh` must be authored as offline sibling preset roots and converted back to canonical `xyxy` before inference/eval/visualization boundaries.
- Runtime fusion config authoring has been removed from the supported training surface. If multi-dataset training is needed now, prepare each dataset independently, merge JSONLs offline, and point `custom.train_jsonl` / `custom.val_jsonl` at the merged artifact.

## Use This Router For

- "What shape must the JSONL have?"
- "How are images resized and validated?"
- "What are the current packing defaults for this training surface?"
- "What happens if a raw JSONL sample exceeds `global_max_length`?"
- "How should I handle multi-dataset mixing after runtime fusion removal?"
- "How do I launch, operate, recover, or roll back COCO refinement?"

## Primary current code handles

- `src/data/`
- `src/data/jsonl.py`
- `src/data/examples.py`
- `src/data/geometry.py`
- `src/data/images.py`
- `src/templates/`
- `src/qwen/encoding.py`
- `src/packing/`
- `src/config/loader.py`

Older `src/datasets/` and `src/config/schema.py` references belong to the
historical MS-Swift/mainline path and are not current Swift ownership.
