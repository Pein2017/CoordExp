#!/usr/bin/env bash
exec >> /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/second-fit-210457-stop-history-v1/phase.log 2>&1
set -euo pipefail
export CONDA_PREFIX=/root/miniconda3/envs/ms
export PATH=/root/miniconda3/envs/ms/bin:/data/CoordExp/.codex/bin:$PATH
export PYTHONPATH=/data/CoordExp/.worktrees/research-probes
printf 'started_utc=%s shell_pid=%s python=%s conda_prefix=%s\n' "$(date -u +%FT%TZ)" "$$" "$(command -v python)" "$CONDA_PREFIX"
exec python /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/second-fit-210457-stop-history-v1/run.py run --manifest /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/second-fit-210457-stop-history-v1/manifest.json
