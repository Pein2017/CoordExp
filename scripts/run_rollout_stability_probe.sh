#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper to run a small rollout → eval → stability report loop.
#
# Example:
#   bash scripts/run_rollout_stability_probe.sh \
#     output/12-24/coord_loss-merged/ckpt-3106 \
#     public_data/lvis/rescale_32_768_poly_20/val.coord.jsonl
#
# You can also pass env vars (override defaults):
#   DEVICE=cuda:0 LIMIT=200 TEMP=0 MAXTOK=2048 OVERLAY=0 bash scripts/run_rollout_stability_probe.sh <ckpt>

PYTHON_BIN="/root/miniconda3/envs/ms/bin/python"

CKPT="${CKPT:-${1:-}}"
GT_JSONL="${GT_JSONL:-${2:-public_data/lvis/rescale_32_768_poly_20/val.coord.jsonl}}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-output/infer/rollout_probe_ckpt_$(date +%Y%m%d_%H%M%S)}"

DEVICE="${DEVICE:-cuda:0}"
MODE="${MODE:-coord}"
PRED_COORD_MODE="${PRED_COORD_MODE:-auto}"

# Keep defaults aligned with scripts/run_infer_eval.sh but favor parsability checks.
LIMIT="${LIMIT:-200}"
TEMP="${TEMP:-0}"       # 0 => greedy decoding (do_sample=false)
TOPP="${TOPP:-0.95}"
MAXTOK="${MAXTOK:-2048}"  # high enough to avoid truncation-caused JSON breakage
REPPEN="${REPPEN:-1.05}"
SEED="${SEED:-42}"

# Evaluator knobs (fast by default)
UNKNOWN_POLICY="${UNKNOWN_POLICY:-bucket}"
STRICT_PARSE="${STRICT_PARSE:-0}"
USE_SEGM="${USE_SEGM:-1}"
OVERLAY="${OVERLAY:-0}"
OVERLAY_K="${OVERLAY_K:-12}"
NUM_WORKERS="${NUM_WORKERS:-0}"

if [[ -z "$CKPT" ]]; then
  echo "Usage: $0 <ckpt_dir> [gt_jsonl]" >&2
  echo "Or set CKPT=... as env var." >&2
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

CKPT="$CKPT" \
GT_JSONL="$GT_JSONL" \
OUTPUT_BASE_DIR="$OUTPUT_BASE_DIR" \
DEVICE="$DEVICE" \
MODE="$MODE" \
PRED_COORD_MODE="$PRED_COORD_MODE" \
LIMIT="$LIMIT" \
TEMP="$TEMP" \
TOPP="$TOPP" \
MAXTOK="$MAXTOK" \
REPPEN="$REPPEN" \
SEED="$SEED" \
UNKNOWN_POLICY="$UNKNOWN_POLICY" \
STRICT_PARSE="$STRICT_PARSE" \
USE_SEGM="$USE_SEGM" \
OVERLAY="$OVERLAY" \
OVERLAY_K="$OVERLAY_K" \
NUM_WORKERS="$NUM_WORKERS" \
bash scripts/run_infer_eval.sh

echo ""
echo "Stability report:"
PYTHONPATH=. "$PYTHON_BIN" scripts/report_rollout_stability.py \
  --pred_jsonl "$OUTPUT_BASE_DIR/pred.jsonl" \
  --summary_json "$OUTPUT_BASE_DIR/summary.json" \
  --eval_metrics_json "$OUTPUT_BASE_DIR/eval/metrics.json"

echo ""
echo "Artifacts:"
echo "  pred_jsonl:   $OUTPUT_BASE_DIR/pred.jsonl"
echo "  infer_summary:$OUTPUT_BASE_DIR/summary.json"
echo "  eval_metrics: $OUTPUT_BASE_DIR/eval/metrics.json"

