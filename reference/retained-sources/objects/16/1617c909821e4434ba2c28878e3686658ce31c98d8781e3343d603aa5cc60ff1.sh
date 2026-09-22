#!/bin/bash
cd /data/CoordExp/.worktrees/research-probes
root=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-readout-state
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 timeout 900 python "$root/producer.py" Bnormalized64 477415 > "$root/Bnormalized64-477415.log" 2>&1
first=$?
printf "%s\n" "$first" > "$root/Bnormalized64-477415.exit"
if [ "$first" -ne 0 ]; then printf "%s\n" "$first" > "$root/run.exit"; exit "$first"; fi
pids=()
(PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 timeout 900 python "$root/producer.py" Bnormalized64 351017 > "$root/Bnormalized64-351017.log" 2>&1; c=$?; printf "%s\n" "$c" > "$root/Bnormalized64-351017.exit"; exit "$c") &
pids+=($!)
(PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 timeout 900 python "$root/producer.py" Bnormalized64 417044 > "$root/Bnormalized64-417044.log" 2>&1; c=$?; printf "%s\n" "$c" > "$root/Bnormalized64-417044.exit"; exit "$c") &
pids+=($!)
(PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 timeout 900 python "$root/producer.py" P16 477415 > "$root/P16-477415.log" 2>&1; c=$?; printf "%s\n" "$c" > "$root/P16-477415.exit"; exit "$c") &
pids+=($!)
(PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 timeout 900 python "$root/producer.py" P16 351017 > "$root/P16-351017.log" 2>&1; c=$?; printf "%s\n" "$c" > "$root/P16-351017.exit"; exit "$c") &
pids+=($!)
(PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 timeout 900 python "$root/producer.py" P16 417044 > "$root/P16-417044.log" 2>&1; c=$?; printf "%s\n" "$c" > "$root/P16-417044.exit"; exit "$c") &
pids+=($!)
(PYTHONPATH=. CUDA_VISIBLE_DEVICES=5 timeout 900 python "$root/producer.py" R16 477415 > "$root/R16-477415.log" 2>&1; c=$?; printf "%s\n" "$c" > "$root/R16-477415.exit"; exit "$c") &
pids+=($!)
(PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 timeout 900 python "$root/producer.py" R16 351017 > "$root/R16-351017.log" 2>&1; c=$?; printf "%s\n" "$c" > "$root/R16-351017.exit"; exit "$c") &
pids+=($!)
(PYTHONPATH=. CUDA_VISIBLE_DEVICES=7 timeout 900 python "$root/producer.py" R16 417044 > "$root/R16-417044.log" 2>&1; c=$?; printf "%s\n" "$c" > "$root/R16-417044.exit"; exit "$c") &
pids+=($!)
code=0
for p in "${pids[@]}"; do wait "$p" || code=1; done
printf "%s\n" "$code" > "$root/run.exit"
exit "$code"
