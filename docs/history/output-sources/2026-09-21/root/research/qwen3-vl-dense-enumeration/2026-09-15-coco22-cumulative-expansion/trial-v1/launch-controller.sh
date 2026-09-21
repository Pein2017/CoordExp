#!/usr/bin/env bash
python -m probes.training_set_completion.coco22_trial controller --trial /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/trial-v1/trial.json --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/trial-v1 --release /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/runtime-repair-v1/main-release.json >> /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/trial-v1/controller.log 2>&1
code=$?
printf "COCO22_MAIN_EXIT=%s\n" "$code" >> /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/trial-v1/controller-event.log
printf "%s\n" "$code" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/trial-v1/controller-exit.txt
exit "$code"
