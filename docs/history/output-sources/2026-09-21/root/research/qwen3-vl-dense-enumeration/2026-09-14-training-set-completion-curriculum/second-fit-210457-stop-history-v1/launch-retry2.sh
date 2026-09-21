#!/usr/bin/env bash
exec >> /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/second-fit-210457-stop-history-v1/phase-retry2.log 2>&1
set -euo pipefail
export CONDA_PREFIX=/root/miniconda3/envs/ms
export PATH=/root/miniconda3/envs/ms/bin:/data/CoordExp/.codex/bin:$PATH
export PYTHONPATH=/data/CoordExp/.worktrees/research-probes
export CUDA_VISIBLE_DEVICES=0
printf 'started_utc=%s shell_pid=%s python=%s conda_prefix=%s cuda_visible_devices=%s\n' "$(date -u +%FT%TZ)" "$$" "$(command -v python)" "$CONDA_PREFIX" "$CUDA_VISIBLE_DEVICES"
exec python /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/second-fit-210457-stop-history-v1/run.py run --manifest /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/second-fit-210457-stop-history-v1/manifest.json
