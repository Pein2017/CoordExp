#!/usr/bin/env bash
# Fixed pilot launcher; no retries, extra updates, or inter-arm state reuse.
set -uo pipefail
cd /data/CoordExp/.worktrees/coco-gt-correction-portfolio || exit 1
pilot=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1
logs="$pilot/launch"
mkdir -p "$logs"
for arm in R B M W; do
  if [[ -e "$pilot/$arm" || -e "$logs/$arm.exit" ]]; then
    echo "COCO_ARM_FAILED: refusing existing $arm target"
    exit 1
  fi
done
printf '%s\n' "$$" > "$logs/driver.pid"
date -u +%FT%TZ > "$logs/started-utc.txt"
run_arm() {
  local arm="$1" devices="$2" status
  timeout --signal=TERM --kill-after=60s 12h env CUDA_VISIBLE_DEVICES="$devices" \
    conda run --no-capture-output -n ms torchrun --standalone --nproc_per_node=2 \
    scripts/research/train_coco_gt_correction.py \
    --bank-manifest "$pilot/bank/manifest.json" --arm "$arm" \
    --output-root "$pilot/$arm" --max-updates 64 --global-batch-size 32 \
    --seed 20260907 > "$logs/$arm.log" 2>&1
  status=$?
  printf '%s\n' "$status" > "$logs/$arm.exit"
  if [[ "$status" -eq 0 ]]; then
    echo "COCO_ARM_COMPLETED: $arm"
  else
    echo "COCO_ARM_FAILED: $arm exit=$status"
  fi
  return "$status"
}
run_arm R 0,1 & p_r=$!
run_arm B 2,3 & p_b=$!
run_arm M 4,5 & p_m=$!
run_arm W 6,7 & p_w=$!
failed=0
for job in "$p_r" "$p_b" "$p_m" "$p_w"; do
  wait "$job" || failed=1
done
date -u +%FT%TZ > "$logs/finished-utc.txt"
if [[ "$failed" -eq 0 ]]; then
  echo COCO_ALL_COMPLETED
else
  echo COCO_PORTFOLIO_FAILED
fi
exit "$failed"
