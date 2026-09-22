#!/usr/bin/env bash
set -u
cd /data/CoordExp/.worktrees/research-probes
RUN=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-successful-row-mechanism
date +%s > "$RUN/logs/native-S.start"
CUDA_VISIBLE_DEVICES=0 python -m probes.training_set_completion.successful_row_state --panel "$RUN/stage2/panels/S.json" --condition native-S --output "$RUN/stage2/runtime/native-S" > "$RUN/logs/native-S.log" 2>&1 &
producer=$!; echo "$producer" > "$RUN/logs/native-S.pid"
wait "$producer"; status=$?; echo "$status" > "$RUN/logs/native-S.exit"
date +%s > "$RUN/logs/native-S.end"
tmux wait-for -S successful-row-native-s-done
exit "$status"
