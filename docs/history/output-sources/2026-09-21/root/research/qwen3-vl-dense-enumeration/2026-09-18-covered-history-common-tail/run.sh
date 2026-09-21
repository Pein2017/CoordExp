#!/bin/bash
cd /data/CoordExp/.worktrees/research-probes
R=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-covered-history-common-tail
export PYTHONPATH=$PWD OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0 python -m probes.training_set_completion.covered_history_common_tail --root "$R" --condition A1 > "$R/A1.log" 2>&1 &
FIRST=$!
echo "$FIRST" > "$R/A1.pid"
wait "$FIRST"
CODE=$?
echo "$CODE" > "$R/A1.exit"
if [ "$CODE" != 0 ]; then exit "$CODE"; fi
python -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="passed"' "$R/runtime/A1/parity.json" || exit 1
CUDA_VISIBLE_DEVICES=1 python -m probes.training_set_completion.covered_history_common_tail --root "$R" --condition A2 > "$R/A2.log" 2>&1 &
A2=$!
echo "$A2" > "$R/A2.pid"
CUDA_VISIBLE_DEVICES=2 python -m probes.training_set_completion.covered_history_common_tail --root "$R" --condition B1 > "$R/B1.log" 2>&1 &
B1=$!
echo "$B1" > "$R/B1.pid"
CUDA_VISIBLE_DEVICES=3 python -m probes.training_set_completion.covered_history_common_tail --root "$R" --condition B2 > "$R/B2.log" 2>&1 &
B2=$!
echo "$B2" > "$R/B2.pid"
wait "$A2"
echo "$?" > "$R/A2.exit"
wait "$B1"
echo "$?" > "$R/B1.exit"
wait "$B2"
echo "$?" > "$R/B2.exit"
