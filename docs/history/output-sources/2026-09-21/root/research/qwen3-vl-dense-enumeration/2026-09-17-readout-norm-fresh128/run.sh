#!/bin/bash
cd /data/CoordExp/.worktrees/research-probes
r=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128
date +%s > "$r/main.start"
echo $$ > "$r/main.pid"
pids=()
(
date +%s > "$r/shard-0.start"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. timeout --kill-after=30 4400 python probes/training_set_completion/readout_norm_fresh.py --panel "$r/panel.json" --output "$r/runtime" --groups fresh-00,fresh-08,fresh-16,fresh-24,shard-01-batch5 > "$r/shard-0.log" 2>&1
status=$?
echo "$status" > "$r/shard-0.exit"
date +%s > "$r/shard-0.end"
exit "$status"
) &
pids+=("$!")
echo "$!" > "$r/shard-0.pid"
(
date +%s > "$r/shard-1.start"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. timeout --kill-after=30 4400 python probes/training_set_completion/readout_norm_fresh.py --panel "$r/panel.json" --output "$r/runtime" --groups fresh-01,fresh-09,fresh-17,fresh-25,shard-05-batch6 > "$r/shard-1.log" 2>&1
status=$?
echo "$status" > "$r/shard-1.exit"
date +%s > "$r/shard-1.end"
exit "$status"
) &
pids+=("$!")
echo "$!" > "$r/shard-1.pid"
(
date +%s > "$r/shard-2.start"
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. timeout --kill-after=30 4400 python probes/training_set_completion/readout_norm_fresh.py --panel "$r/panel.json" --output "$r/runtime" --groups fresh-02,fresh-10,fresh-18,fresh-26,shard-07-batch5 > "$r/shard-2.log" 2>&1
status=$?
echo "$status" > "$r/shard-2.exit"
date +%s > "$r/shard-2.end"
exit "$status"
) &
pids+=("$!")
echo "$!" > "$r/shard-2.pid"
(
date +%s > "$r/shard-3.start"
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. timeout --kill-after=30 4400 python probes/training_set_completion/readout_norm_fresh.py --panel "$r/panel.json" --output "$r/runtime" --groups fresh-03,fresh-11,fresh-19,fresh-27 > "$r/shard-3.log" 2>&1
status=$?
echo "$status" > "$r/shard-3.exit"
date +%s > "$r/shard-3.end"
exit "$status"
) &
pids+=("$!")
echo "$!" > "$r/shard-3.pid"
(
date +%s > "$r/shard-4.start"
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. timeout --kill-after=30 4400 python probes/training_set_completion/readout_norm_fresh.py --panel "$r/panel.json" --output "$r/runtime" --groups fresh-04,fresh-12,fresh-20,fresh-28 > "$r/shard-4.log" 2>&1
status=$?
echo "$status" > "$r/shard-4.exit"
date +%s > "$r/shard-4.end"
exit "$status"
) &
pids+=("$!")
echo "$!" > "$r/shard-4.pid"
(
date +%s > "$r/shard-5.start"
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. timeout --kill-after=30 4400 python probes/training_set_completion/readout_norm_fresh.py --panel "$r/panel.json" --output "$r/runtime" --groups fresh-05,fresh-13,fresh-21,fresh-29 > "$r/shard-5.log" 2>&1
status=$?
echo "$status" > "$r/shard-5.exit"
date +%s > "$r/shard-5.end"
exit "$status"
) &
pids+=("$!")
echo "$!" > "$r/shard-5.pid"
(
date +%s > "$r/shard-6.start"
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. timeout --kill-after=30 4400 python probes/training_set_completion/readout_norm_fresh.py --panel "$r/panel.json" --output "$r/runtime" --groups fresh-06,fresh-14,fresh-22,fresh-30 > "$r/shard-6.log" 2>&1
status=$?
echo "$status" > "$r/shard-6.exit"
date +%s > "$r/shard-6.end"
exit "$status"
) &
pids+=("$!")
echo "$!" > "$r/shard-6.pid"
(
date +%s > "$r/shard-7.start"
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. timeout --kill-after=30 4400 python probes/training_set_completion/readout_norm_fresh.py --panel "$r/panel.json" --output "$r/runtime" --groups fresh-07,fresh-15,fresh-23,fresh-31 > "$r/shard-7.log" 2>&1
status=$?
echo "$status" > "$r/shard-7.exit"
date +%s > "$r/shard-7.end"
exit "$status"
) &
pids+=("$!")
echo "$!" > "$r/shard-7.pid"
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "$status" > "$r/main.exit"
date +%s > "$r/main.end"
exit "$status"
