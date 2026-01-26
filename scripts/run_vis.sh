#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib/backbone.sh"

# ----------- Declare runtime configuration -----------
PRED_JSONL="${PRED_JSONL:-}"
SAVE_DIR="${SAVE_DIR:-}"
ROOT_IMAGE_DIR="${ROOT_IMAGE_DIR:-}"
LIMIT="${LIMIT:-20}"

ensure_required "PRED_JSONL" "$PRED_JSONL"
ensure_required "SAVE_DIR" "$SAVE_DIR"
ensure_required "ROOT_IMAGE_DIR" "$ROOT_IMAGE_DIR"

mkdir -p "$SAVE_DIR"

echo "Visualization workflow"
echo "  predictions:      $PRED_JSONL"
echo "  save dir:         $SAVE_DIR"
echo "  root image dir:   $ROOT_IMAGE_DIR"
echo "  limit:            $LIMIT"

cd "$REPO_ROOT"

CMD=(
  "${COORDEXP_PYTHON[@]}" "$REPO_ROOT/vis_tools/vis_coordexp.py"
  --pred_jsonl "$PRED_JSONL"
  --save_dir "$SAVE_DIR"
  --limit "$LIMIT"
)

echo "Running visualization..."
printf '%q ' "${CMD[@]}"
echo
ROOT_IMAGE_DIR="$ROOT_IMAGE_DIR" PYTHONPATH="$REPO_ROOT" "${CMD[@]}"

echo "Visualization complete — outputs in $SAVE_DIR"
