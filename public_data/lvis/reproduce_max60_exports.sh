#!/usr/bin/env bash
# Reproduce the two LVIS max60 geometry-ablation datasets on a fresh machine/node:
# - public_data/lvis/rescale_32_768_bbox_max60/
# - public_data/lvis/rescale_32_768_poly_prefer_semantic_max60/
#
# This script is intentionally "safe by default":
# - It runs the ~25GB LVIS/COCO2017 download only if raw assets are missing.
# - It delegates export details (caps/guards/tokenlen) to the canonical exporter script.
#
# Usage (from repo root):
#   bash public_data/lvis/reproduce_max60_exports.sh
#
# Tip (tmux):
#   tmux new -s lvis_reexport 'bash public_data/lvis/reproduce_max60_exports.sh'

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RAW_ANN_DIR="public_data/lvis/raw/annotations"
RAW_IMG_DIR="public_data/lvis/raw/images"

need_download=0
if [ ! -f "${RAW_ANN_DIR}/lvis_v1_train.json" ] || [ ! -f "${RAW_ANN_DIR}/lvis_v1_val.json" ]; then
  need_download=1
fi
if [ ! -d "${RAW_IMG_DIR}/train2017" ] || [ ! -d "${RAW_IMG_DIR}/val2017" ]; then
  need_download=1
fi

if [ "${need_download}" -eq 1 ]; then
  echo "[lvis] raw assets missing; downloading LVIS annotations + COCO2017 images..."
  ./public_data/run.sh lvis download
else
  echo "[lvis] raw assets exist; skip download"
fi

echo "[lvis] exporting bbox-only + poly-prefer-semantic max60 datasets..."
bash public_data/scripts/export_lvis_bbox_poly_prefer_semantic_max60.sh

