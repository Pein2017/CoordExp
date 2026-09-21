#!/usr/bin/env bash
set -u
cd /data/CoordExp/.worktrees/research-probes
RUN=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-successful-row-mechanism
date +%s > "$RUN/logs/component-capture.start"
CUDA_VISIBLE_DEVICES=2 python -m probes.training_set_completion.successful_row_components --panel "$RUN/stage2/panels/S.json" --output "$RUN/stage2/component-capture" > "$RUN/logs/component-capture.log" 2>&1 &
producer=$!; echo "$producer" > "$RUN/logs/component-capture.pid"
wait "$producer"; status=$?; echo "$status" > "$RUN/logs/component-capture.exit"; date +%s > "$RUN/logs/component-capture.end"
tmux wait-for -S successful-row-component-capture-done
exit "$status"
