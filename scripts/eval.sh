#!/usr/bin/env bash
# Fixed-config evaluation script (no CLI flags).
# Edit values below if needed.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/ms/bin/python"

PRED="output/infer/text/pred.jsonl"
OUT_DIR="output/evaluation/text"
UNKNOWN="bucket"
STRICT=""              # set to "--strict-parse" to enable
USE_SEGM=""            # set to "--no-segm" to disable segmentation
IOU_THRS=""            # set to space-separated values like "0.5 0.75" or leave empty
OVERLAY=""             # set to "--overlay" to enable
OVERLAY_K=12
NUM_WORKERS=8

CMD=("$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_detection.py"
  --pred_jsonl "$PRED"
  --out_dir "$OUT_DIR"
  --unknown-policy "$UNKNOWN"
  --overlay-k "$OVERLAY_K"
  --num-workers "$NUM_WORKERS"
)

if [[ -n "$STRICT" ]]; then CMD+=($STRICT); fi
if [[ -n "$USE_SEGM" ]]; then CMD+=($USE_SEGM); fi
if [[ -n "$OVERLAY" ]]; then CMD+=($OVERLAY); fi
if [[ -n "$IOU_THRS" ]]; then CMD+=(--iou-thrs $IOU_THRS); fi

echo "Running: ${CMD[*]}"
cd "$REPO_ROOT"
"${CMD[@]}"
