# Scripts

This directory contains stable, user-facing entrypoints plus compatibility
wrappers and historical diagnostics. Treat YAML-first entrypoints as the
reportable path; shell wrappers that take only environment variables are
manual/debug.

## Stable entrypoints

- Training (YAML-first): `scripts/train.sh` (wraps `python -m src.sft --config ...`).
- Stage-2 rollout-correction server-mode launcher (vLLM server + multi-GPU learner): `scripts/train_stage2.sh`.
- Unified inference pipeline (YAML-first): `scripts/run_infer.py`.
- Confidence scoring / score materialization (YAML-first): `scripts/postop_confidence.py`.
- Offline detection evaluation (YAML-first): `scripts/evaluate_detection.py`.
- One-run proxy bundle evaluation (YAML-first): `scripts/evaluate_proxy_detection_bundle.py`.
- Export helper (merge LoRA + coord offsets): `scripts/merge_coord.sh`.

## Compatibility / debug wrappers

- `scripts/run_infer_eval.sh`: legacy environment-variable convenience wrapper
  for quick inference plus debug evaluation. It defaults to raw/F1-ish debug
  scope and must not be used for official-looking COCO/LVIS/both claims from
  raw `gt_vs_pred.jsonl`. It refuses COCO/LVIS/both metrics entirely; for those
  metrics, use the YAML-first infer -> score -> eval flow.
- `scripts/run_vis.sh`: manual/debug visualization wrapper for an explicitly
  supplied prediction artifact and image root. Prefer evaluator overlays or
  `vis_resources/` artifacts tied to resolved pipeline provenance for
  reportable evidence.

## Shared helpers

- `scripts/_lib/backbone.sh`: shared bash helpers (repo root resolution, `ensure_required`, python runner).

## External transfer helpers

Baidu Netdisk upload/download helpers live in the repo-local Codex skill:

- `.codex/skills/baidupcsgo-upload/scripts/upload_dir.sh`
- `.codex/skills/baidupcsgo-upload/scripts/download_dir.sh`

Use them for `output/` backups under `/CoordExp/output/`. Do not use Baidu
Netdisk as the default sync surface for `model_cache/`, raw `public_data/`, or
processed `public_data/` contents.

## Utilities (organized)

- Analysis helpers: `scripts/analysis/`
- Tooling helpers: `scripts/tools/`
- Small pipelines / workflow wrappers and diagnostics: `scripts/pipelines/`
  - tmux queue manager for sequential training jobs: `scripts/pipelines/train_task_manager.sh`
    (Python core: `scripts/pipelines/train_task_manager.py`)
  - historical rollout parser/stability diagnostic:
    `scripts/pipelines/run_rollout_stability_probe.sh`; this delegates to the
    legacy/debug `run_infer_eval.sh` wrapper and is not a stable benchmark
    pipeline.

## Deprecated

Deprecated wrappers are removed. Prefer stable YAML-first entrypoints for
inference, scoring, evaluation, and reportable visualization artifacts.
