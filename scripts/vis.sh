#!/usr/bin/env bash
# Fixed-config visualization script (no CLI flags).
# Edit values below if needed.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/ms/bin/python"

# PRED="output/infer/coord/pred.jsonl"
PRED="output/infer/text/pred.jsonl"
SAVE_DIR="vis_out/text"
LIMIT=20                # 0 = all samples
# Directory containing the images referenced in pred.jsonl (for relative paths).
# Set this to the dataset root used during inference.
ROOT_IMAGE_DIR="public_data/lvis/rescale_32_768_poly_20"

CMD=("$PYTHON_BIN" "$REPO_ROOT/vis_tools/vis_coordexp.py"
  --pred_jsonl "$PRED"
  --save_dir "$SAVE_DIR"
  --limit "$LIMIT"
)

echo "Running: ${CMD[*]}"
cd "$REPO_ROOT"
ROOT_IMAGE_DIR="$ROOT_IMAGE_DIR" "${CMD[@]}"
