#!/usr/bin/env bash
set -euo pipefail

# Historical/debug wrapper to run a small rollout -> eval -> stability report
# loop. This delegates to scripts/run_infer_eval.sh, which is a legacy/debug
# wrapper. Treat outputs as parser/rollout health diagnostics, not stable
# benchmark metrics.
#
# Example:
#   ckpt=output/12-24/coord_loss-merged/ckpt-3106 \
#   gt_jsonl=public_data/lvis/rescale_32_768_poly_20/val.coord.jsonl \
#   bash scripts/pipelines/run_rollout_stability_probe.sh
#
# You can also pass env vars (override defaults):
#   device=cuda:0 limit=200 temp=0 maxtok=2048 overlay=0 bash scripts/pipelines/run_rollout_stability_probe.sh

if [[ $# -gt 0 ]]; then
  echo "[ERROR] scripts/pipelines/run_rollout_stability_probe.sh accepts environment variables only (no positional args)." >&2
  echo "[ERROR] Example: ckpt=output/.../checkpoint-1234 gt_jsonl=public_data/.../val.coord.jsonl bash scripts/pipelines/run_rollout_stability_probe.sh" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_lib/backbone.sh"

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
EVAL_METRICS="${eval_metrics:-${EVAL_METRICS:-f1ish}}"

if [[ -z "$CKPT" ]]; then
  echo "ERROR: ckpt must be set." >&2
  echo "Example: ckpt=output/.../checkpoint-1234 bash scripts/pipelines/run_rollout_stability_probe.sh" >&2
  exit 1
fi

echo "[WARN] scripts/pipelines/run_rollout_stability_probe.sh is a historical/debug diagnostic."
echo "[WARN] It delegates to legacy/debug scripts/run_infer_eval.sh; do not treat outputs as benchmark claims."
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
  echo "  EVAL_METRICS:    $EVAL_METRICS"

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
eval_metrics="$EVAL_METRICS" \
bash "$REPO_ROOT/scripts/run_infer_eval.sh"

echo ""
echo "Stability report:"
PYTHONPATH="$REPO_ROOT" "${COORDEXP_PYTHON[@]}" "$REPO_ROOT/scripts/report_rollout_stability.py" \
  --pred_jsonl "$OUTPUT_BASE_DIR/gt_vs_pred.jsonl" \
  --summary_json "$OUTPUT_BASE_DIR/summary.json" \
  --eval_metrics_json "$OUTPUT_BASE_DIR/eval/metrics.json"

echo ""
echo "Artifacts:"
echo "  pred_jsonl:   $OUTPUT_BASE_DIR/gt_vs_pred.jsonl"
echo "  infer_summary:$OUTPUT_BASE_DIR/summary.json"
echo "  eval_metrics: $OUTPUT_BASE_DIR/eval/metrics.json"
