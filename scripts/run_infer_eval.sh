#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  echo "[ERROR] scripts/run_infer_eval.sh accepts environment variables only (no positional args)." >&2
  echo "[ERROR] Example: ckpt=output/... gt_jsonl=public_data/... output_base_dir=output/infer/my_run bash scripts/run_infer_eval.sh" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib/backbone.sh"

# ----------- Declare runtime configuration -----------
# Required (no default)
CKPT="${ckpt:-${CKPT:-output/12-4/coord_merged_ck1632}}"
GT_JSONL="${gt_jsonl:-${GT_JSONL:-public_data/lvis/rescale_32_768_poly_20/train.coord.jsonl}}"
OUTPUT_BASE_DIR="${output_base_dir:-${OUTPUT_BASE_DIR:-output/infer/coord_new}}"

# Inference defaults
DEVICE="${device:-${DEVICE:-cuda:1}}"
MODE="${mode:-${MODE:-coord}}"                        # coord | text
PRED_COORD_MODE="${pred_coord_mode:-${PRED_COORD_MODE:-auto}}"  # auto | pixel | norm1000
LIMIT="${limit:-${LIMIT:-50}}"
TEMP="${temp:-${TEMP:-0.01}}"
TOPP="${topp:-${TOPP:-0.95}}"
MAXTOK="${maxtok:-${MAXTOK:-1024}}"
REPPEN="${reppen:-${REPPEN:-1.05}}"
SEED="${seed:-${SEED:-}}"

# Evaluation defaults
EVAL_METRICS="${eval_metrics:-${EVAL_METRICS:-both}}"          # coco | f1ish | both
STRICT_PARSE="${strict_parse:-${STRICT_PARSE:-0}}"             # 1 to enable --strict-parse
USE_SEGM="${use_segm:-${USE_SEGM:-1}}"                     # 0 to disable segmentation metrics
IOU_THRS="${iou_thrs:-${IOU_THRS:-}}"
OVERLAY="${overlay:-${OVERLAY:-1}}"                       # 1 to enable overlay rendering
OVERLAY_K="${overlay_k:-${OVERLAY_K:-12}}"
NUM_WORKERS="${num_workers:-${NUM_WORKERS:-0}}"

# Semantic desc matching is always on (map-or-drop). Legacy UNKNOWN_POLICY /
# SEMANTIC_FALLBACK knobs are deprecated and ignored by the evaluator.
SEMANTIC_MODEL="${semantic_model:-${SEMANTIC_MODEL:-sentence-transformers/all-MiniLM-L6-v2}}"
SEMANTIC_THRESHOLD="${semantic_threshold:-${SEMANTIC_THRESHOLD:-0.6}}"
SEMANTIC_DEVICE="${semantic_device:-${SEMANTIC_DEVICE:-}}"            # auto|cpu|cuda[:N]
SEMANTIC_BATCH_SIZE="${semantic_batch_size:-${SEMANTIC_BATCH_SIZE:-64}}"

if [[ -z "$SEMANTIC_DEVICE" ]]; then
  # Prefer the same device as inference when it's a CUDA device; otherwise auto.
  if [[ "${DEVICE:-}" == cuda* ]]; then
    SEMANTIC_DEVICE="$DEVICE"
  else
    SEMANTIC_DEVICE="auto"
  fi
fi

PRED_JSONL="$OUTPUT_BASE_DIR/gt_vs_pred.jsonl"
SUMMARY_JSON="$OUTPUT_BASE_DIR/summary.json"
EVAL_OUT_DIR="$OUTPUT_BASE_DIR/eval"

ensure_required "CKPT" "$CKPT"
ensure_required "GT_JSONL" "$GT_JSONL"
ensure_required "OUTPUT_BASE_DIR" "$OUTPUT_BASE_DIR"

mkdir -p "$OUTPUT_BASE_DIR" "$EVAL_OUT_DIR"

echo "Inference → Evaluation workflow"
echo "  checkpoint:         $CKPT"
echo "  GT JSONL:           $GT_JSONL"
echo "  OUTPUT_BASE_DIR:    $OUTPUT_BASE_DIR"
echo "  MODE:               $MODE"
echo "  PRED_COORD_MODE:    $PRED_COORD_MODE"
echo "  DEVICE:             $DEVICE"
echo "  LIMIT:              $LIMIT"
echo "  EVAL_METRICS:       $EVAL_METRICS"
echo "  Eval out dir:       $EVAL_OUT_DIR"

cd "$REPO_ROOT"

CMD_INF=(
  "${COORDEXP_PYTHON[@]}" "$REPO_ROOT/scripts/run_infer.py"
  --gt_jsonl "$GT_JSONL"
  --model_checkpoint "$CKPT"
  --mode "$MODE"
  --pred-coord-mode "$PRED_COORD_MODE"
  --out "$PRED_JSONL"
  --summary "$SUMMARY_JSON"
  --device "$DEVICE"
  --limit "$LIMIT"
  --temperature "$TEMP"
  --top_p "$TOPP"
  --max_new_tokens "$MAXTOK"
  --repetition_penalty "$REPPEN"
)

if [[ -n "$SEED" ]]; then
  CMD_INF+=(--seed "$SEED")
fi

echo "Running inference..."
printf '%q ' "${CMD_INF[@]}"
echo
PYTHONPATH="$REPO_ROOT" "${CMD_INF[@]}"

echo "Inference finished, predictions saved to $PRED_JSONL"

CMD_EVAL=(
  "${COORDEXP_PYTHON[@]}" "$REPO_ROOT/scripts/evaluate_detection.py"
  --pred_jsonl "$PRED_JSONL"
  --out_dir "$EVAL_OUT_DIR"
  --metrics "$EVAL_METRICS"
  --overlay-k "$OVERLAY_K"
  --num-workers "$NUM_WORKERS"
  --semantic-model "$SEMANTIC_MODEL"
  --semantic-threshold "$SEMANTIC_THRESHOLD"
  --semantic-device "$SEMANTIC_DEVICE"
  --semantic-batch-size "$SEMANTIC_BATCH_SIZE"
)

if [[ "$STRICT_PARSE" =~ ^(1|true|yes)$ ]]; then
  CMD_EVAL+=(--strict-parse)
fi

if [[ "$USE_SEGM" =~ ^(0|false|no)$ ]]; then
  CMD_EVAL+=(--no-segm)
fi

if [[ -n "$OVERLAY" && "$OVERLAY" =~ ^(1|true|yes)$ ]]; then
  CMD_EVAL+=(--overlay)
fi

if [[ -n "$IOU_THRS" ]]; then
  CMD_EVAL+=(--iou-thrs $IOU_THRS)
fi

echo "Running evaluation..."
printf '%q ' "${CMD_EVAL[@]}"
echo
PYTHONPATH="$REPO_ROOT" "${CMD_EVAL[@]}"

echo "Finished evaluation — metrics saved in $EVAL_OUT_DIR"
