#!/usr/bin/env bash
set -euo pipefail

# Fixed-config smoke inference script (no CLI flags).
# Edit values below if needed.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/ms/bin/python"

# Choose which preset to run: coord (norm1000) or text (pixel).
# Override with: PRESET=coord ./scripts/infer.sh
PRESET="${PRESET:-text}"   # coord | text

# Common inputs
GT="public_data/lvis/rescale_32_768_poly_20/train.jsonl"
CKPT="output/12-4/text_only_merged_ck1632"
DEVICE="cuda:2"
LIMIT=20
TEMP=0.1
TOPP=0.95
MAXTOK=1024
REPPEN=1.05
SEED=""                    # set to an int for deterministic runs

if [[ "$PRESET" == "coord" ]]; then
  # Coord-mode preset: GT and preds expected in norm1000; denorm to pixels.
  MODE="coord"
  PRED_COORD_MODE="norm1000"
  OUT="output/infer/coord/pred.jsonl"
  SUMMARY="output/infer/coord/summary.json"
else
  # Text-mode preset: GT in pixel space; force preds interpreted as pixel to
  # avoid norm1000 denorm issues when visualizing text-only models.
  MODE="text"
  PRED_COORD_MODE="pixel"
  OUT="output/infer/text/pred.jsonl"
  SUMMARY="output/infer/text/summary.json"
fi

OUT_DIR="$(dirname "$OUT")"
SUMMARY_DIR="$(dirname "$SUMMARY")"
mkdir -p "$OUT_DIR" "$SUMMARY_DIR"

cmd=(
  "$PYTHON_BIN" "$REPO_ROOT/scripts/run_infer.py"
  --gt_jsonl "$GT"
  --model_checkpoint "$CKPT"
  --mode "$MODE"
  --pred-coord-mode "$PRED_COORD_MODE"
  --out "$OUT"
  --summary "$SUMMARY"
  --device "$DEVICE"
  --limit "$LIMIT"
  --temperature "$TEMP"
  --top_p "$TOPP"
  --max_new_tokens "$MAXTOK"
  --repetition_penalty "$REPPEN"
)

if [[ -n "$SEED" ]]; then
  cmd+=(--seed "$SEED")
fi

cd "$REPO_ROOT"
echo "Running preset=$PRESET mode=$MODE pred_coord_mode=$PRED_COORD_MODE"
exec "${cmd[@]}"
