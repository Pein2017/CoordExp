#!/usr/bin/env bash
set -u
cd /data/CoordExp/.worktrees/research-probes
RUN=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-successful-row-mechanism
pids=()
for pair in FF:1 SX_FY:2 FX_SY:3; do
  name=${pair%:*}; gpu=${pair#*:}
  (
    date +%s > "$RUN/logs/$name.start"
    CUDA_VISIBLE_DEVICES=$gpu python -m probes.training_set_completion.repetition_history_runtime --panel "$RUN/stage1/panels/309264-$name.json" --case 309264 --mode prefix --output-root "$RUN/stage1/runtime/$name" > "$RUN/logs/$name.log" 2>&1 &
    producer=$!; echo "$producer" > "$RUN/logs/$name.pid"
    wait "$producer"; status=$?; echo "$status" > "$RUN/logs/$name.exit"
    date +%s > "$RUN/logs/$name.end"
    exit "$status"
  ) &
  pids+=("$!")
done
(
 date +%s > "$RUN/logs/scores.start"
 CUDA_VISIBLE_DEVICES=4 python -m probes.training_set_completion.repetition_history_scores --panel "$RUN/stage1/score-panel.json" --image 309264 --out "$RUN/stage1/score-runtime" > "$RUN/logs/scores.log" 2>&1 &
 producer=$!; echo "$producer" > "$RUN/logs/scores.pid"
 wait "$producer"; status=$?; echo "$status" > "$RUN/logs/scores.exit"
 date +%s > "$RUN/logs/scores.end"
 exit "$status"
) &
pids+=("$!")
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "$status" > "$RUN/logs/stage1-rest.exit"
tmux wait-for -S successful-row-stage1-done
exit "$status"
