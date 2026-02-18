# Public Data Module (`public_data/`)

Purpose:
- Provides geometry-aware, tested pipelines to turn public datasets (LVIS first; expanding) into JSONLs that match the CoordExp training contract.

Entry points:
- High-level overview: `public_data/README.md`.
- Operational docs + dataset-specific notes: `public_data/README.md`.

Unified runner (recommended):
- `./public_data/run.sh <dataset> <command> [runner-flags] [-- <passthrough-args>]`
- Dataset plugins live under `public_data/datasets/<dataset>.sh`.

Shared preprocessing steps (dataset-agnostic):
- `public_data/scripts/rescale_jsonl.py` (smart resize / pixel budget + geometry rewrite).
- `public_data/scripts/convert_to_coord_tokens.py` (convert norm1000 numeric coords to `<|coord_k|>` tokens).
- `public_data/scripts/validate_jsonl.py` (schema + geometry validation).

Output expectations:
- Pipelines typically emit `train.jsonl` / `val.jsonl` plus a coord-token version `*.coord.jsonl`.
- Training consumes them via `custom.train_jsonl`/`custom.val_jsonl` or via `custom.fusion_config`.
