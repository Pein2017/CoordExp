#!/bin/bash
cd /data/CoordExp/.worktrees/research-probes
root=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-corner-loop-y1-one-bin-control
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python "$root/producer.py" Y1_998 > "$root/Y1_998.log" 2>&1
code=$?
printf "%s\n" "$code" > "$root/run.exit"
exit "$code"
