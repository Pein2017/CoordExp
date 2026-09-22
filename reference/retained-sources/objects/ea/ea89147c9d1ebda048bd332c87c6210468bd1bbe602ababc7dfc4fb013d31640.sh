#!/usr/bin/env bash
set -u
cd /data/CoordExp/.worktrees/research-probes
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
R=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-owner-recurrence-row-branch
echo $$ > "$R/bird-branches-shell.pid"
(CUDA_VISIBLE_DEVICES=4 python "$R/producer-v2.py" --panel "$R/panel.json" --case 309264 --arm same --output-root "$R/runtime" > "$R/logs/309264-same.log" 2>&1; echo $? > "$R/logs/309264-same.exit") &
p1=$!
(CUDA_VISIBLE_DEVICES=5 python "$R/producer-v2.py" --panel "$R/panel.json" --case 309264 --arm distinct --output-root "$R/runtime" > "$R/logs/309264-distinct.log" 2>&1; echo $? > "$R/logs/309264-distinct.exit") &
p2=$!
wait "$p1" "$p2"
tmux wait-for -S owner-recurrence-bird-done
