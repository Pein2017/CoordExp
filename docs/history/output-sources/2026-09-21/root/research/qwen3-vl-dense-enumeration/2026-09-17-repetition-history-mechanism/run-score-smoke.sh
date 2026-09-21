#!/bin/bash
cd /data/CoordExp/.worktrees/research-probes || exit 90
R=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism
export CUDA_VISIBLE_DEVICES=0
python -m probes.training_set_completion.repetition_history_scores --panel "$R/stage-a-panel.json" --image 309264 --out "$R/scores/309264-smoke" > "$R/logs/score-smoke.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$R/logs/score-smoke.exit"
exit "$rc"
