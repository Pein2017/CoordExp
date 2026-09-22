#!/bin/bash
cd /data/CoordExp/.worktrees/research-probes || exit 90
R=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism
CUDA_VISIBLE_DEVICES=5 python -m probes.training_set_completion.repetition_history_scores --panel "$R/stage-a-alt-panel.json" --image 309264 --out "$R/scores/309264-alt" > "$R/logs/score-alt.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$R/logs/score-alt.exit"
if [ "$rc" -eq 0 ]; then
 CUDA_VISIBLE_DEVICES=5 python -m probes.training_set_completion.repetition_history_runtime --panel "$R/panels/A/309264-AA_alt.json" --case 309264 --mode prefix --output-root "$R/runtime/A/AA_alt" > "$R/logs/A-309264-AA_alt.log" 2>&1
 rc=$?
 printf '%s\n' "$rc" > "$R/logs/A-309264-AA_alt.exit"
fi
tmux wait-for -S repetition-history-alt
exit "$rc"
