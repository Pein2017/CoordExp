#!/bin/bash
cd /data/CoordExp/.worktrees/research-probes
r=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128
start=$(date +%s)
printf '%s\n' "$$" > "$r/admit.pid"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python probes/training_set_completion/readout_norm_fresh.py --panel "$r/panel.json" --output "$r/runtime" --groups shard-02-batch0 > "$r/admit.log" 2>&1
status=$?
printf '%s\n' "$status" > "$r/admit.exit"
printf '%s %s\n' "$start" "$(date +%s)" > "$r/admit.times"
exit "$status"
