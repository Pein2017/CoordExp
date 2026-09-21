#!/usr/bin/env bash
set -euo pipefail
cd /data/CoordExp/.worktrees/coordexp-infras
OUT=/data/CoordExp/outputs/infra_base/untied-axis-val200-20260918
exec >> "$OUT/terminal.log" 2>&1
echo "$$" > "$OUT/launcher.pid"
trap 'rc=$?; echo "$rc" > "$OUT/exit-code"; echo "QUEUE_TERMINAL exit_code=$rc time=$(date -u +%FT%TZ)"' EXIT
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 FLASH_ATTENTION_DETERMINISTIC=1
for RUN in rp110 rp100; do
 echo "INFERENCE_STARTED run=$RUN time=$(date -u +%FT%TZ)"
 python -m src.infer --config "$OUT/$RUN.yaml" > "$OUT/$RUN-infer.log" 2>&1
 echo "EVALUATION_STARTED run=$RUN time=$(date -u +%FT%TZ)"
 python -m scripts.evaluate_detection --artifact-dir "$OUT/$RUN" --out-dir "$OUT/$RUN/evaluation" > "$OUT/$RUN-eval.log" 2>&1
 echo "RUN_COMPLETED run=$RUN time=$(date -u +%FT%TZ)"
done
