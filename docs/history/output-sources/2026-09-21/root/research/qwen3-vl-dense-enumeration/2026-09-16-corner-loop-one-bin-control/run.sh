#!/usr/bin/env bash
set -u
cd /data/CoordExp/.worktrees/research-probes
root=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-corner-loop-one-bin-control
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python "$root/producer.py" Y998 > "$root/Y998.log" 2>&1
code=$?
echo "$code" > "$root/run.exit"
exit "$code"
