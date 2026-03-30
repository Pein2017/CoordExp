---
doc_id: docs.data.index
layer: docs
doc_type: router
status: canonical
domain: data
summary: Router for dataset contracts, offline preparation, packing, and dataset-specific notes.
tags: [data, datasets, jsonl]
updated: 2026-03-30
---

# Data & Datasets

Open this folder when you need the current dataset contract or the offline preparation workflow.

## Read Order

1. [CONTRACT.md](CONTRACT.md)
2. [PREPARATION.md](PREPARATION.md)
3. [PACKING.md](PACKING.md)
4. [FUSION.md](FUSION.md) only for dormant legacy fusion notes
5. [VISUAL_GENOME.md](VISUAL_GENOME.md) only for dataset-specific notes

## Page Roles

- [CONTRACT.md](CONTRACT.md)
  - authoritative JSONL, geometry, and runtime assumptions
- [PREPARATION.md](PREPARATION.md)
  - offline conversion, resizing, validation, and data-building flow
- [PACKING.md](PACKING.md)
  - sequence-packing policy and runtime tradeoffs
- [FUSION.md](FUSION.md)
  - dormant legacy fusion surface kept in-tree for future reactivation
- [VISUAL_GENOME.md](VISUAL_GENOME.md)
  - Visual Genome-specific intake notes

## Use This Router For

- "What shape must the JSONL have?"
- "How are images resized and validated?"
- "What are the current packing defaults?"
- "How should I handle multi-dataset mixing while runtime fusion is temporarily disabled?"

## Primary Code Handles

- `src/datasets/`
- `src/datasets/geometry.py`
- `src/datasets/builders/jsonlines.py`
- `src/config/schema.py`
