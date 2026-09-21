#!/usr/bin/env bash
set -euo pipefail
cd /data/CoordExp/.worktrees/research-probes
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
R=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-output-ranking-repair
Q="$R/runtime/qualification-v1"
trap 'rc=$?; echo "RANKING_READBACK_RETRY_FAILED exit=$rc"; exit "$rc"' ERR
echo "$$" > "$Q/readback-retry.pid"
python -m probes.training_set_completion.source256_ranking_evaluation worker --manifest "$R/preparation/R-qualification.json" --terminal "$Q/training/terminal.json" --qualification --output "$Q/readback.json" > "$Q/readback-retry.log" 2>&1
echo RANKING_READBACK_RETRY_COMPLETED
