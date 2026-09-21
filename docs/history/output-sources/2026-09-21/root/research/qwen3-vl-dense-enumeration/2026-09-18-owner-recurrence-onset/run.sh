#!/bin/bash
cd /data/CoordExp/.worktrees/research-probes
R=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-owner-recurrence-onset
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
python -m probes.training_set_completion.owner_recurrence_onset --root "$R" --mode native > "$R/native.log" 2>&1 &
PID=$!
echo "$PID" > "$R/native.pid"
wait "$PID"
EXIT=$?
echo "$EXIT" > "$R/native.exit"
if [ "$EXIT" != 0 ]; then exit "$EXIT"; fi
export CUDA_VISIBLE_DEVICES=1
python -m probes.training_set_completion.owner_recurrence_onset --root "$R" --mode scores > "$R/scores.log" 2>&1 &
PID=$!
echo "$PID" > "$R/scores.pid"
wait "$PID"
EXIT=$?
echo "$EXIT" > "$R/scores.exit"
exit "$EXIT"
