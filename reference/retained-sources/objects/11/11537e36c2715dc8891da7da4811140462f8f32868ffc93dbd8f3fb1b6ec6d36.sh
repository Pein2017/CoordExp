#!/usr/bin/env bash
set -u
cd /data/CoordExp/.worktrees/research-probes
RUN=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-successful-row-mechanism
mkdir -p "$RUN/logs"
date +%s > "$RUN/logs/SS.start"
CUDA_VISIBLE_DEVICES=0 python -m probes.training_set_completion.repetition_history_runtime --panel "$RUN/stage1/panels/309264-SS.json" --case 309264 --mode prefix --output-root "$RUN/stage1/runtime/SS" > "$RUN/logs/SS.log" 2>&1 &
producer=$!
echo "$producer" > "$RUN/logs/SS.pid"
wait "$producer"
status=$?
echo "$status" > "$RUN/logs/SS.exit"
date +%s > "$RUN/logs/SS.end"
tmux wait-for -S successful-row-ss-done
exit "$status"
