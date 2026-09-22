#!/usr/bin/env bash
# One fixed three-arm invocation. No retries or scientific recipe changes.
set -uo pipefail
cd /data/CoordExp/.worktrees/coco-gt-correction-portfolio || exit 1
pilot=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1
bank=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1/bank/manifest.json
logs="$pilot/launch-training"
mkdir -p "$logs"
for arm in R M Rweak; do
  if [[ -e "$pilot/$arm" || -e "$logs/$arm.exit" ]]; then
    echo "FOCUS_ARM_FAILED: refusing existing $arm target"
    exit 1
  fi
done
printf '%s\n' "$$" > "$logs/driver.pid"
date -u +%FT%TZ > "$logs/started-utc.txt"
sha256sum scripts/research/train_coco_gt_correction.py scripts/research/coco_gt_correction_bank.py > "$logs/code-sha256.txt"
run_arm() {
  local arm="$1" devices="$2" status
  timeout --signal=TERM --kill-after=60s 12h env CUDA_VISIBLE_DEVICES="$devices" \
    conda run --no-capture-output -n ms torchrun --standalone --nproc_per_node=2 \
    scripts/research/train_coco_gt_correction.py \
    --experiment-id 2026-09-07-coco-owner-focus-ablation \
    --bank-manifest "$bank" --arm "$arm" --output-root "$pilot/$arm" \
    --max-updates 64 --global-batch-size 32 --microbatch-images 2 \
    --seed 20260908 > "$logs/$arm.log" 2>&1
  status=$?
  printf '%s\n' "$status" > "$logs/$arm.exit"
  if [[ "$status" -eq 0 ]]; then
    echo "FOCUS_ARM_COMPLETED: $arm"
  else
    echo "FOCUS_ARM_FAILED: $arm exit=$status"
  fi
  return "$status"
}
# GPU 2/3 remain with the evaluation owner, then the Source baseline.
run_arm R 0,1 & p_r=$!
run_arm M 4,5 & p_m=$!
run_arm Rweak 6,7 & p_w=$!
failed=0
for job in "$p_r" "$p_m" "$p_w"; do
  wait "$job" || failed=1
done
date -u +%FT%TZ > "$logs/finished-utc.txt"
if [[ "$failed" -eq 0 ]]; then
  echo FOCUS_ALL_TRAINING_COMPLETED
else
  echo FOCUS_TRAINING_FAILED
fi
exit "$failed"
