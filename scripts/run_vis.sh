#!/usr/bin/env bash
set -euo pipefail

# Manual/debug renderer for an explicitly supplied artifact and image root.
# For reportable evaluation provenance, prefer evaluator overlays or
# vis_resources artifacts tied to resolved_config.json.

if [[ $# -gt 0 ]]; then
  echo "[ERROR] scripts/run_vis.sh accepts environment variables only (no positional args)." >&2
  echo "[ERROR] Example: pred_jsonl=output/.../gt_vs_pred.jsonl save_dir=vis_out root_image_dir=public_data/... bash scripts/run_vis.sh" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib/backbone.sh"

# ----------- Declare runtime configuration -----------
PRED_JSONL="${pred_jsonl:-${PRED_JSONL:-}}"
SAVE_DIR="${save_dir:-${SAVE_DIR:-}}"
ROOT_IMAGE_DIR="${root_image_dir:-${ROOT_IMAGE_DIR:-}}"
LIMIT="${limit:-${LIMIT:-20}}"

ensure_required "PRED_JSONL" "$PRED_JSONL"
ensure_required "SAVE_DIR" "$SAVE_DIR"
ensure_required "ROOT_IMAGE_DIR" "$ROOT_IMAGE_DIR"

mkdir -p "$SAVE_DIR"

echo "[WARN] scripts/run_vis.sh is a manual/debug visualization wrapper."
echo "[WARN] It renders the explicit artifact and image root provided by env vars; it does not recover full pipeline scoring/config provenance."
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
