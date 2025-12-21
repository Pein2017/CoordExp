#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/ms/bin/python"

# ----------- Declare runtime configuration -----------
PRED_JSONL="${PRED_JSONL:-}"
SAVE_DIR="${SAVE_DIR:-}"
ROOT_IMAGE_DIR="${ROOT_IMAGE_DIR:-}"
LIMIT="${LIMIT:-20}"

ensure_required() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    echo "ERROR: $name must be set."
    exit 1
  fi
}

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
  "$PYTHON_BIN" "$REPO_ROOT/vis_tools/vis_coordexp.py"
  --pred_jsonl "$PRED_JSONL"
  --save_dir "$SAVE_DIR"
  --limit "$LIMIT"
)

echo "Running visualization..."
printf '%q ' "${CMD[@]}"
echo
ROOT_IMAGE_DIR="$ROOT_IMAGE_DIR" PYTHONPATH="$REPO_ROOT" "${CMD[@]}"

echo "Visualization complete — outputs in $SAVE_DIR"

