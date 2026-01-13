#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/ms/bin/python"

# ----------- Declare runtime configuration -----------
# Required (no default)
CKPT="${CKPT:-output/12-4/coord_merged_ck1632}"
GT_JSONL="${GT_JSONL:-public_data/lvis/rescale_32_768_poly_20/train.coord.jsonl}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-output/infer/coord_new}"

# Inference defaults
DEVICE="${DEVICE:-cuda:1}"
MODE="${MODE:-coord}"                        # coord | text
PRED_COORD_MODE="${PRED_COORD_MODE:-auto}"  # auto | pixel | norm1000
LIMIT="${LIMIT:-50}"
TEMP="${TEMP:-0.01}"
TOPP="${TOPP:-0.95}"
MAXTOK="${MAXTOK:-1024}"
REPPEN="${REPPEN:-1.05}"
SEED="${SEED:-}"

# Evaluation defaults
UNKNOWN_POLICY="${UNKNOWN_POLICY:-semantic}"  # bucket | drop | semantic
STRICT_PARSE="${STRICT_PARSE:-0}"             # 1 to enable --strict-parse
USE_SEGM="${USE_SEGM:-1}"                     # 0 to disable segmentation metrics
IOU_THRS="${IOU_THRS:-}"
OVERLAY="${OVERLAY:-1}"                       # 1 to enable overlay rendering
OVERLAY_K="${OVERLAY_K:-12}"
NUM_WORKERS="${NUM_WORKERS:-0}"

# Optional semantic desc matching (only used when UNKNOWN_POLICY=semantic)
SEMANTIC_MODEL="${SEMANTIC_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
SEMANTIC_THRESHOLD="${SEMANTIC_THRESHOLD:-0.6}"
SEMANTIC_FALLBACK="${SEMANTIC_FALLBACK:-bucket}"  # bucket | drop
SEMANTIC_DEVICE="${SEMANTIC_DEVICE:-}"            # auto|cpu|cuda[:N]
SEMANTIC_BATCH_SIZE="${SEMANTIC_BATCH_SIZE:-64}"

if [[ -z "$SEMANTIC_DEVICE" ]]; then
  # Prefer the same device as inference when it's a CUDA device; otherwise auto.
  if [[ "${DEVICE:-}" == cuda* ]]; then
    SEMANTIC_DEVICE="$DEVICE"
  else
    SEMANTIC_DEVICE="auto"
  fi
fi

PRED_JSONL="$OUTPUT_BASE_DIR/pred.jsonl"
SUMMARY_JSON="$OUTPUT_BASE_DIR/summary.json"
EVAL_OUT_DIR="$OUTPUT_BASE_DIR/eval"

ensure_required() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    echo "ERROR: $name must be set."
    exit 1
  fi
}

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
echo "  Eval out dir:       $EVAL_OUT_DIR"

cd "$REPO_ROOT"

CMD_INF=(
  "$PYTHON_BIN" "$REPO_ROOT/scripts/run_infer.py"
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
  "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_detection.py"
  --pred_jsonl "$PRED_JSONL"
  --out_dir "$EVAL_OUT_DIR"
  --unknown-policy "$UNKNOWN_POLICY"
  --overlay-k "$OVERLAY_K"
  --num-workers "$NUM_WORKERS"
)

if [[ "$UNKNOWN_POLICY" == "semantic" ]]; then
  CMD_EVAL+=(
    --semantic-model "$SEMANTIC_MODEL"
    --semantic-threshold "$SEMANTIC_THRESHOLD"
    --semantic-fallback "$SEMANTIC_FALLBACK"
    --semantic-device "$SEMANTIC_DEVICE"
    --semantic-batch-size "$SEMANTIC_BATCH_SIZE"
  )
fi

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
