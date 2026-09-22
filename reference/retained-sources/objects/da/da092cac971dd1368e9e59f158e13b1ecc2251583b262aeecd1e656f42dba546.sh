#!/usr/bin/env bash
set -u
cd /data/CoordExp/.worktrees/research-probes
RUN=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-successful-row-mechanism
(
 date +%s > "$RUN/logs/native-F.start"
 CUDA_VISIBLE_DEVICES=1 python -m probes.training_set_completion.successful_row_state --panel "$RUN/stage2/panels/F.json" --condition native-F --output "$RUN/stage2/runtime/native-F" > "$RUN/logs/native-F.log" 2>&1 &
 producer=$!; echo "$producer" > "$RUN/logs/native-F.pid"
 wait "$producer"; status=$?; echo "$status" > "$RUN/logs/native-F.exit"; date +%s > "$RUN/logs/native-F.end"
 exit "$status"
) &
p1=$!
(
 date +%s > "$RUN/logs/self-S.start"
 CUDA_VISIBLE_DEVICES=0 python -m probes.training_set_completion.successful_row_state --panel "$RUN/stage2/panels/S.json" --condition self-S --donor-states "$RUN/stage2/runtime/native-S/states.pt" --output "$RUN/stage2/runtime/self-S" > "$RUN/logs/self-S.log" 2>&1 &
 producer=$!; echo "$producer" > "$RUN/logs/self-S.pid"
 wait "$producer"; status=$?; echo "$status" > "$RUN/logs/self-S.exit"; date +%s > "$RUN/logs/self-S.end"
 exit "$status"
) &
p2=$!
status=0
wait "$p1" || status=1
wait "$p2" || status=1
echo "$status" > "$RUN/logs/native-f-self-s.exit"
tmux wait-for -S successful-row-native-f-self-s-done
exit "$status"
