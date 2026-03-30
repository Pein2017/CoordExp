---
doc_id: docs.data.fusion
layer: docs
doc_type: reference
status: historical
domain: data
summary: Dormant legacy fusion surface kept in-tree for future reactivation.
updated: 2026-03-30
---

# Fusion Dataset (Dormant Legacy Surface)

CoordExp keeps its legacy fusion assets in-tree so multi-dataset runtime mixing
can be reactivated later, but the authored training surface is currently
disabled.

Current status:

- `custom.fusion_config` fails fast in the active training schema.
- canonical training/eval flows assume the single-dataset hierarchy.
- if you need multi-dataset training today, merge JSONLs offline first.

Why this file still exists:

- preserve the old config shape and example assets for future reactivation
- keep parser/module maintenance discoverable
- document where the dormant examples live in the repo

## What Is Still Kept In-Tree

- fusion config parser/utilities:
  - `src/datasets/fusion.py`
  - `src/datasets/unified_fusion_dataset.py`
  - `src/datasets/fusion_types.py`
- dormant example configs:
  - `configs/fusion/examples/lvis_vg.yaml`
  - `configs/fusion/lvis_bbox_only_vs_poly_prefer_1to1.yaml`
  - `configs/fusion/sft_lvis_vg.yaml`

These files are archival reference assets right now, not active runbook entry
points.

## How To Handle Multi-Dataset Mixing Today

Use offline merging instead:

```bash
PYTHONPATH=. conda run -n ms python public_data/scripts/merge_jsonl.py --help
```

Recommended flow:

1. prepare each dataset JSONL independently
2. merge them into one stable training JSONL offline
3. point `custom.train_jsonl` / `custom.val_jsonl` at that merged artifact

## Dormant Fusion Config Shape

The legacy fusion format is still useful as a reference for future
reactivation:

- top-level keys:
  - `targets`
  - optional `sources`
  - optional `extends`
- per-dataset entry keys commonly used:
  - `dataset`
  - `name`
  - `train_jsonl`
  - `val_jsonl`
  - `template`
  - `ratio`
  - optional prompt overrides such as `user_prompt` / `system_prompt`

For concrete examples, open the configs listed above under `configs/fusion/`.
