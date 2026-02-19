# Scripts

This directory contains stable, user-facing entrypoints plus a small set of
organized utilities used by docs and research workflows.

## Core entrypoints

- Training (YAML-first): `scripts/train.sh` (wraps `python -m src.sft --config ...`).
- Stage-2 AB server-mode launcher (vLLM server + multi-GPU learner): `scripts/train_stage2.sh`.
  Compatibility wrapper (deprecated): `scripts/stage2_ab_server_train.sh`.
- Unified inference pipeline (YAML-first): `scripts/run_infer.py`.
- Offline detection evaluation (YAML-first): `scripts/evaluate_detection.py`.
- Visualization wrapper: `scripts/run_vis.sh` (calls `vis_tools/vis_coordexp.py`).
- Export helper (merge LoRA + coord offsets): `scripts/merge_coord.sh`.

## Shared helpers

- `scripts/_lib/backbone.sh`: shared bash helpers (repo root resolution, `ensure_required`, python runner).

## Utilities (organized)

- Analysis helpers: `scripts/analysis/`
- Tooling helpers: `scripts/tools/` (incl. `scripts/tools/workspace_gc.sh`)
- Small pipelines / workflow wrappers: `scripts/pipelines/`

## Deprecated

Deprecated wrappers are removed. Prefer:
- `scripts/run_vis.sh`
- `vis_tools/vis_coordexp.py`
