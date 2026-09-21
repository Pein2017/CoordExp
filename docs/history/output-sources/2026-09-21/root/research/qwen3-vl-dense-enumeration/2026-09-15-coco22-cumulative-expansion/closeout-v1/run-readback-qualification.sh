#!/usr/bin/env bash
python -m probes.training_set_completion.coco22_readback qualification-controller --training-manifest /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/training-qualification-v1/manifests/mb1.json --training-terminal /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/closeout-v1/source0-terminal.json --adapter /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco227-ce-normalization/trial-v1/S/training/checkpoints/step-00256/adapter --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/readback-qualification-v1 > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/readback-qualification-v1/controller.log 2>&1
code=$?
printf "%s\n" "$code" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/readback-qualification-v1/controller-exit.txt
tmux wait-for -S coco22-readback-qualification-done
exit "$code"
