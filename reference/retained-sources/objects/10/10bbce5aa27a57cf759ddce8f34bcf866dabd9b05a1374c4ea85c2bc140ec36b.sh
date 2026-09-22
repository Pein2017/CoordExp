#!/usr/bin/env bash
python -m probes.training_set_completion.coco22_trial qualification-controller --plan /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/training-qualification-v1/plan.json --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/training-qualification-v1 > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/training-qualification-v1/controller.log 2>&1
code=$?
printf "%s\n" "$code" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/training-qualification-v1/controller-exit.txt
tmux wait-for -S coco22-training-qualification-done
exit "$code"
