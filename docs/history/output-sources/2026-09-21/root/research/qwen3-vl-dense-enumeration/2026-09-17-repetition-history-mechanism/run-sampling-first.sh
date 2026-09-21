#!/bin/bash
cd /data/CoordExp/.worktrees/research-probes
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=3
python /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/sampling-producer.py --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/C/309264-T0.7-seed19.json --out /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/C/T0.7-seed19 > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/C-T0.7-seed19.log 2>&1 &
pid=$!
echo "$pid" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/C-T0.7-seed19.pid
wait "$pid"
status=$?
echo "$status" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/C-T0.7-seed19.exit
touch /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/sampling-first.settled
tmux wait-for -S repetition-history-sampling-first
exit "$status"
