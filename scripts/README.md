# Scripts

This directory intentionally contains only stable, user-facing entrypoints.

## Core entrypoints

- Training (YAML-first): `scripts/train.sh` (wraps `python -m src.sft --config ...`).
- Unified inference pipeline (YAML-first): `scripts/run_infer.py`.
- Offline detection evaluation (YAML-first): `scripts/evaluate_detection.py`.
- Visualization wrapper: `scripts/run_vis.sh` (calls `vis_tools/vis_coordexp.py`).

## Shared helpers

- `scripts/_lib/backbone.sh`: shared bash helpers (repo root resolution, `ensure_required`, python runner).

## Legacy scripts

Non-core analysis and one-off utilities have been moved under:
- `scripts/_legacy/analysis/`
- `scripts/_legacy/tools/`
- `scripts/_legacy/pipelines/`

The original paths under `scripts/` remain as small forwarders for backward
compatibility and print a deprecation notice.
