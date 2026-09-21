#!/usr/bin/env bash
bash /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/source0-v1/run.sh
code=$?
printf "%s\n" "$code" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/source0-v1/exit.txt
tmux wait-for -S coco22-source0-done
exit "$code"
