#!/usr/bin/env bash
set -u
cd /data/CoordExp/.worktrees/research-probes
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
R=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-owner-recurrence-row-branch
echo $$ > "$R/book-branches-shell.pid"
(CUDA_VISIBLE_DEVICES=2 python "$R/producer-v2.py" --panel "$R/panel.json" --case 386313 --arm same --output-root "$R/runtime" > "$R/logs/386313-same.log" 2>&1; echo $? > "$R/logs/386313-same.exit") &
p1=$!
(CUDA_VISIBLE_DEVICES=3 python "$R/producer-v2.py" --panel "$R/panel.json" --case 386313 --arm distinct --output-root "$R/runtime" > "$R/logs/386313-distinct.log" 2>&1; echo $? > "$R/logs/386313-distinct.exit") &
p2=$!
wait "$p1" "$p2"
tmux wait-for -S owner-recurrence-book-done
