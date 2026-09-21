#!/usr/bin/env bash
set -euo pipefail
retry_root=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/stage03-single-owner-repair-v1-retry2
export CONDA_PREFIX=/root/miniconda3/envs/ms
export PATH=/root/miniconda3/envs/ms/bin:/data/CoordExp/.codex/bin:$PATH
mkdir -p "$retry_root/logs"
python_path=$(command -v python)
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "launcher_pid=$$" "python_path=$python_path" "conda_prefix=$CONDA_PREFIX" > "$retry_root/phase.log"
printf '{"schema":"training_set_completion.stage03_single_owner_repair.v1.controller_start.v1","launcher_pid":%s,"python_path":"%s","conda_prefix":"%s"}\n' "$$" "$python_path" "$CONDA_PREFIX" > "$retry_root/controller-start.json"
cd /data/CoordExp/.worktrees/research-probes
exec python -m probes.training_set_completion.repair controller --manifest "$retry_root/manifest.json" --output "$retry_root" >> "$retry_root/logs/controller.log" 2>&1
