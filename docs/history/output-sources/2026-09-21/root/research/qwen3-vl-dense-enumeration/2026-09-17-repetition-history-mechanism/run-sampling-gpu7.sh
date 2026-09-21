#!/bin/bash
cd /data/CoordExp/.worktrees/research-probes
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=7
failed=0
python /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/sampling-producer.py --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/C/309264-T0.1-seed26.json --out /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/C/T0.1-seed26 > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/C-T0.1-seed26.log 2>&1 &
pid=$!
echo "$pid" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/C-T0.1-seed26.pid
wait "$pid"
status=$?
echo "$status" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/C-T0.1-seed26.exit
if [ "$status" -ne 0 ]; then failed=1; fi
python /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/sampling-producer.py --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/C/309264-T0.3-seed26.json --out /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/C/T0.3-seed26 > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/C-T0.3-seed26.log 2>&1 &
pid=$!
echo "$pid" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/C-T0.3-seed26.pid
wait "$pid"
status=$?
echo "$status" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/C-T0.3-seed26.exit
if [ "$status" -ne 0 ]; then failed=1; fi
echo "$failed" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/sampling-worker-7.exit
exit "$failed"
