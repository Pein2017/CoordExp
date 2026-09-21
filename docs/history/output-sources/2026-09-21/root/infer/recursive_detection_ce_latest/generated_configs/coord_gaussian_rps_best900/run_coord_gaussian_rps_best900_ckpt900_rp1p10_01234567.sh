#!/usr/bin/env bash
set -euo pipefail

cd /data/CoordExp/.worktrees/coord-shape-gaussian-rps

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29631
export PYTHONPATH=/data/CoordExp/.worktrees/coord-shape-gaussian-rps
export TOKENIZERS_PARALLELISM=false

LOG=/data/CoordExp/.worktrees/coord-shape-gaussian-rps/outputs/infer/recursive_detection_ce_latest/coord_gaussian_rps_best900_ckpt900_rp1p10_8gpu.log
mkdir -p "$(dirname "$LOG")"

echo "[$(date -Is)] coord_gaussian_rps_best900 rp=1.10 infer/eval start" | tee -a "$LOG"
/root/miniconda3/envs/ms/bin/python3.12 -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29631 \
  scripts/run_infer.py \
  --config /data/CoordExp/.worktrees/coord-shape-gaussian-rps/temp/infer/recursive_detection_ce_latest/compact_full_prefix_rollin_balance2_coord_gaussian_rps_best900_ckpt900_val200_bsz8_temp0_rp1p10_max3084_chatfix_8gpu.yaml \
  2>&1 | tee -a "$LOG"
echo "[$(date -Is)] coord_gaussian_rps_best900 rp=1.10 infer/eval finished" | tee -a "$LOG"
