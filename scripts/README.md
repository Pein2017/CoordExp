# Scripts

This directory contains user-facing entrypoints plus compatibility wrappers and
historical diagnostics. The canonical training/inference implementation is
owned by `src/train.py`, `src/infer.py`, and `src/inference/`; see
`docs/COORDEXP_SWIFT.md` and `docs/BRANCH_AND_WORKTREE_POLICY.md`.

## Stable entrypoints

- Training (canonical Swift): `python -m src.train --config configs/coordexp_swift/...`.
- Inference (canonical Swift): `python -m src.infer --config configs/coordexp_swift/infer/...`.
- Offline CoordExp-Swift detection evaluation (direct artifact reducer):
  `scripts/evaluate_detection.py --artifact-dir ... --out-dir ...`.
- Export helper (merge LoRA + token-embeddings adapter offsets): `scripts/merge_coord.sh`.

## Compatibility / debug wrappers

- `scripts/train.sh`, `scripts/train_stage2.sh`, `scripts/run_infer.py`,
  `scripts/postop_confidence.py`, and
  `scripts/evaluate_proxy_detection_bundle.py`: legacy/mainline wrappers.
  They are not the canonical Swift entrypoints and should be used only for
  explicit compatibility or historical reproduction.
- `scripts/run_infer_eval.sh`: legacy/mainline environment-variable wrapper.
- `scripts/run_vis.sh`: manual/debug visualization wrapper for an explicitly
  supplied prediction artifact and image root. Prefer evaluator overlays or
  `vis_resources/` artifacts tied to resolved pipeline provenance for
  reportable evidence.

## Shared helpers

- `scripts/_lib/backbone.sh`: shared bash helpers (repo root resolution, `ensure_required`, python runner).

## External transfer helpers

No current transfer helper is owned by this directory or a first-party skill.
For an explicitly authorized outputs transfer, follow
`docs/standards/OUTPUT_SYNC_AND_DATA_PROVENANCE.md`, inspect the installed
BaiduPCS-Go client, and bind exact local and remote roots before acting. Do not
use Baidu Netdisk as the default sync surface for `model_cache/`, raw
`public_data/`, or processed `public_data/` contents.

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
