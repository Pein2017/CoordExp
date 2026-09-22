#!/bin/bash
cd /data/CoordExp/.worktrees/research-probes
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
failed=0
python /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/sampling-producer.py --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/C/309264-T0.1-seed19.json --out /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/C/T0.1-seed19 > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/C-T0.1-seed19.log 2>&1 &
pid=$!
echo "$pid" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/C-T0.1-seed19.pid
wait "$pid"
status=$?
echo "$status" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/C-T0.1-seed19.exit
if [ "$status" -ne 0 ]; then failed=1; fi
python /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/sampling-producer.py --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/C/309264-T0.3-seed19.json --out /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/C/T0.3-seed19 > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/C-T0.3-seed19.log 2>&1 &
pid=$!
echo "$pid" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/C-T0.3-seed19.pid
wait "$pid"
status=$?
echo "$status" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/C-T0.3-seed19.exit
if [ "$status" -ne 0 ]; then failed=1; fi
python /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/sampling-producer.py --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/C/309264-T0.7-seed20.json --out /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/C/T0.7-seed20 > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/C-T0.7-seed20.log 2>&1 &
pid=$!
echo "$pid" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/C-T0.7-seed20.pid
wait "$pid"
status=$?
echo "$status" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/C-T0.7-seed20.exit
if [ "$status" -ne 0 ]; then failed=1; fi
echo "$failed" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/sampling-worker-0.exit
exit "$failed"
