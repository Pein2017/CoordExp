#!/bin/bash
cd /data/CoordExp/.worktrees/research-probes
root=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-image-history
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 timeout 900 python "$root/producer.py" I00 > "$root/I00.log" 2>&1
code=$?
printf "%s\n" "$code" > "$root/I00.exit"
if [ "$code" -ne 0 ]; then printf "%s\n" "$code" > "$root/run.exit"; exit "$code"; fi
python "$root/admit.py" > "$root/admit.log" 2>&1
code=$?
if [ "$code" -ne 0 ]; then printf "%s\n" "$code" > "$root/run.exit"; exit "$code"; fi
(PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 timeout 900 python "$root/producer.py" D00 > "$root/D00.log" 2>&1; c=$?; printf "%s\n" "$c" > "$root/D00.exit"; exit "$c") &
p0=$!
(PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 timeout 900 python "$root/producer.py" D10 > "$root/D10.log" 2>&1; c=$?; printf "%s\n" "$c" > "$root/D10.exit"; exit "$c") &
p1=$!
code=0
wait "$p0" || code=1
wait "$p1" || code=1
printf "%s\n" "$code" > "$root/run.exit"
exit "$code"
