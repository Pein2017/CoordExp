#!/bin/bash
cd /data/CoordExp/.worktrees/research-probes
root=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-natural-readout-norm
pids=()
(
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 timeout 800 python "$root/producer.py" shard-01-batch5 identity > "$root/shard-01-batch5-identity.log" 2>&1
c=$?
printf "%s\n" "$c" > "$root/shard-01-batch5-identity.exit"
if [ "$c" -ne 0 ]; then exit "$c"; fi
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 timeout 800 python "$root/producer.py" shard-01-batch5 norm > "$root/shard-01-batch5-norm.log" 2>&1
c=$?
printf "%s\n" "$c" > "$root/shard-01-batch5-norm.exit"
exit "$c"
) &
pids+=($!)
(
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 timeout 800 python "$root/producer.py" shard-02-batch0 identity > "$root/shard-02-batch0-identity.log" 2>&1
c=$?
printf "%s\n" "$c" > "$root/shard-02-batch0-identity.exit"
if [ "$c" -ne 0 ]; then exit "$c"; fi
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 timeout 800 python "$root/producer.py" shard-02-batch0 norm > "$root/shard-02-batch0-norm.log" 2>&1
c=$?
printf "%s\n" "$c" > "$root/shard-02-batch0-norm.exit"
exit "$c"
) &
pids+=($!)
(
PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 timeout 800 python "$root/producer.py" shard-05-batch6 identity > "$root/shard-05-batch6-identity.log" 2>&1
c=$?
printf "%s\n" "$c" > "$root/shard-05-batch6-identity.exit"
if [ "$c" -ne 0 ]; then exit "$c"; fi
PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 timeout 800 python "$root/producer.py" shard-05-batch6 norm > "$root/shard-05-batch6-norm.log" 2>&1
c=$?
printf "%s\n" "$c" > "$root/shard-05-batch6-norm.exit"
exit "$c"
) &
pids+=($!)
(
PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 timeout 800 python "$root/producer.py" shard-07-batch5 identity > "$root/shard-07-batch5-identity.log" 2>&1
c=$?
printf "%s\n" "$c" > "$root/shard-07-batch5-identity.exit"
if [ "$c" -ne 0 ]; then exit "$c"; fi
PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 timeout 800 python "$root/producer.py" shard-07-batch5 norm > "$root/shard-07-batch5-norm.log" 2>&1
c=$?
printf "%s\n" "$c" > "$root/shard-07-batch5-norm.exit"
exit "$c"
) &
pids+=($!)
code=0
for p in "${pids[@]}"; do wait "$p" || code=1; done
printf "%s\n" "$code" > "$root/run.exit"
exit "$code"
