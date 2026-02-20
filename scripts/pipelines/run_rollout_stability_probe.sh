#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper to run a small rollout → eval → stability report loop.
#
# Example:
#   ckpt=output/12-24/coord_loss-merged/ckpt-3106 \
#   gt_jsonl=public_data/lvis/rescale_32_768_poly_20/val.coord.jsonl \
#   bash scripts/pipelines/run_rollout_stability_probe.sh
#
# You can also pass env vars (override defaults):
#   device=cuda:0 limit=200 temp=0 maxtok=2048 overlay=0 bash scripts/pipelines/run_rollout_stability_probe.sh

PYTHON_BIN="/root/miniconda3/envs/ms/bin/python"

if [[ $# -gt 0 ]]; then
  echo "[ERROR] scripts/pipelines/run_rollout_stability_probe.sh accepts environment variables only (no positional args)." >&2
  echo "[ERROR] Example: ckpt=output/.../checkpoint-1234 gt_jsonl=public_data/.../val.coord.jsonl bash scripts/pipelines/run_rollout_stability_probe.sh" >&2
  exit 2
fi

CKPT="${ckpt:-${CKPT:-}}"
GT_JSONL="${gt_jsonl:-${GT_JSONL:-public_data/lvis/rescale_32_768_poly_20/val.coord.jsonl}}"
OUTPUT_BASE_DIR="${output_base_dir:-${OUTPUT_BASE_DIR:-output/infer/rollout_probe_ckpt_$(date +%Y%m%d_%H%M%S)}}"

DEVICE="${device:-${DEVICE:-cuda:0}}"
MODE="${mode:-${MODE:-coord}}"
PRED_COORD_MODE="${pred_coord_mode:-${PRED_COORD_MODE:-auto}}"

# Keep defaults aligned with scripts/run_infer_eval.sh but favor parsability checks.
LIMIT="${limit:-${LIMIT:-200}}"
TEMP="${temp:-${TEMP:-0}}"       # 0 => greedy decoding (do_sample=false)
TOPP="${topp:-${TOPP:-0.95}}"
MAXTOK="${maxtok:-${MAXTOK:-2048}}"  # high enough to avoid truncation-caused JSON breakage
REPPEN="${reppen:-${REPPEN:-1.05}}"
SEED="${seed:-${SEED:-42}}"

# Evaluator knobs (fast by default)
UNKNOWN_POLICY="${unknown_policy:-${UNKNOWN_POLICY:-bucket}}"
STRICT_PARSE="${strict_parse:-${STRICT_PARSE:-0}}"
USE_SEGM="${use_segm:-${USE_SEGM:-1}}"
OVERLAY="${overlay:-${OVERLAY:-0}}"
OVERLAY_K="${overlay_k:-${OVERLAY_K:-12}}"
NUM_WORKERS="${num_workers:-${NUM_WORKERS:-0}}"

if [[ -z "$CKPT" ]]; then
  echo "ERROR: ckpt must be set." >&2
  echo "Example: ckpt=output/.../checkpoint-1234 bash scripts/pipelines/run_rollout_stability_probe.sh" >&2
  exit 1
fi

echo "Rollout stability probe"
echo "  CKPT:            $CKPT"
echo "  GT_JSONL:        $GT_JSONL"
echo "  OUTPUT_BASE_DIR: $OUTPUT_BASE_DIR"
echo "  DEVICE:          $DEVICE"
echo "  MODE:            $MODE"
echo "  LIMIT:           $LIMIT"
echo "  TEMP:            $TEMP"
echo "  MAXTOK:          $MAXTOK"
echo "  OVERLAY:         $OVERLAY"

ckpt="$CKPT" \
gt_jsonl="$GT_JSONL" \
output_base_dir="$OUTPUT_BASE_DIR" \
device="$DEVICE" \
mode="$MODE" \
pred_coord_mode="$PRED_COORD_MODE" \
limit="$LIMIT" \
temp="$TEMP" \
topp="$TOPP" \
maxtok="$MAXTOK" \
reppen="$REPPEN" \
seed="$SEED" \
unknown_policy="$UNKNOWN_POLICY" \
strict_parse="$STRICT_PARSE" \
use_segm="$USE_SEGM" \
overlay="$OVERLAY" \
overlay_k="$OVERLAY_K" \
num_workers="$NUM_WORKERS" \
bash scripts/run_infer_eval.sh

echo ""
echo "Stability report:"
PYTHONPATH=. "$PYTHON_BIN" scripts/report_rollout_stability.py \
  --pred_jsonl "$OUTPUT_BASE_DIR/gt_vs_pred.jsonl" \
  --summary_json "$OUTPUT_BASE_DIR/summary.json" \
  --eval_metrics_json "$OUTPUT_BASE_DIR/eval/metrics.json"

echo ""
echo "Artifacts:"
echo "  pred_jsonl:   $OUTPUT_BASE_DIR/gt_vs_pred.jsonl"
echo "  infer_summary:$OUTPUT_BASE_DIR/summary.json"
echo "  eval_metrics: $OUTPUT_BASE_DIR/eval/metrics.json"
