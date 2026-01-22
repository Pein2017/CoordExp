#!/usr/bin/env bash
# End-to-end LVIS pipeline: convert raw annotations -> JSONL, smart-resize images/geometry,
# cap polygon vertices, emit coord-token JSONLs, and create tiny subsets.
#
# Run after public_data/scripts/download_lvis.py
# Override defaults via env vars, e.g.
#   MAX_BLOCKS=1024 TINY=512 bash public_data/scripts/lvis_full_pipeline.sh

set -euo pipefail

RAW_ROOT=${RAW_ROOT:-public_data/lvis/raw}
OUTPUT_BASE=${OUTPUT_BASE:-public_data/lvis}
FACTOR=${FACTOR:-32}
MAX_BLOCKS=${MAX_BLOCKS:-768}   # set 1024 for larger budget
MIN_BLOCKS=${MIN_BLOCKS:-4}
POLY_MAX_POINTS=${POLY_MAX_POINTS:-20}
TINY=${TINY:-256}
NUM_WORKERS=${NUM_WORKERS:-32}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/ms/bin/python}
SPLITS=${SPLITS:-"train val"}   # space-separated list

OUT_ROOT="${OUTPUT_BASE}/rescale_${FACTOR}_${MAX_BLOCKS}_poly_${POLY_MAX_POINTS}"
MAX_PIXELS=$((FACTOR * FACTOR * MAX_BLOCKS))
MIN_PIXELS=$((FACTOR * FACTOR * MIN_BLOCKS))

echo "=== LVIS full pipeline ==="
echo "raw:        ${RAW_ROOT}"
echo "output:     ${OUT_ROOT}"
echo "factor:     ${FACTOR}"
echo "max_blocks: ${MAX_BLOCKS}"
echo "min_blocks: ${MIN_BLOCKS}"
echo "poly cap:   ${POLY_MAX_POINTS}"
echo "tiny size:  ${TINY}"
echo "splits:     ${SPLITS}"
mkdir -p "${OUT_ROOT}/images"

for SPLIT in ${SPLITS}; do
  ANNO="${RAW_ROOT}/annotations/lvis_v1_${SPLIT}.json"
  IMG_ROOT="${RAW_ROOT}/images"
  RAW_JSONL="${OUT_ROOT}/${SPLIT}.raw.jsonl"          # pixel-space intermediate
  OUT_JSONL="${OUT_ROOT}/${SPLIT}.jsonl"               # norm1000 ints (final)
  TINY_JSONL="${OUT_ROOT}/${SPLIT}_tiny.jsonl"         # norm1000 ints (final)
  OUT_COORD="${OUT_ROOT}/${SPLIT}.coord.jsonl"         # coord tokens (final)
  TINY_COORD="${OUT_ROOT}/${SPLIT}_tiny.coord.jsonl"   # coord tokens (final)

  if [ ! -f "${ANNO}" ]; then
    echo "Missing annotation file: ${ANNO}" >&2
    exit 1
  fi
  if [ ! -d "${IMG_ROOT}" ]; then
    echo "Missing images directory: ${IMG_ROOT}" >&2
    exit 1
  fi

  echo "---- ${SPLIT}: convert + smart-resize (pixel -> raw) ----"
  ${PYTHON_BIN} public_data/scripts/convert_lvis.py \
    --split "${SPLIT}" \
    --annotation "${ANNO}" \
    --image_root "${IMG_ROOT}" \
    --use-polygon \
    --poly-max-points "${POLY_MAX_POINTS}" \
    --smart-resize \
    --image_factor "${FACTOR}" \
    --max_pixels "${MAX_PIXELS}" \
    --min_pixels "${MIN_PIXELS}" \
    --resize_output_root "${OUT_ROOT}" \
    --output "${RAW_JSONL}" \
    --num_workers "${NUM_WORKERS}"

  echo "---- ${SPLIT}: normalize (pixel -> norm ints + tokens) ----"
  PYTHONPATH=. ${PYTHON_BIN} public_data/scripts/convert_to_coord_tokens.py \
    --input "${RAW_JSONL}" \
    --output-norm "${OUT_JSONL}" \
    --output-tokens "${OUT_COORD}"

  echo "---- ${SPLIT}: tiny subset (${TINY}) from norm ints ----"
  PYTHONPATH=. ${PYTHON_BIN} public_data/scripts/sample_dataset.py \
    --input "${OUT_JSONL}" \
    --output "${TINY_JSONL}" \
    --num_samples "${TINY}" \
    --strategy random

  echo "---- ${SPLIT}: coord tokens (tiny, already normalized) ----"
  PYTHONPATH=. ${PYTHON_BIN} public_data/scripts/convert_to_coord_tokens.py \
    --input "${TINY_JSONL}" \
    --assume-normalized \
    --output "${TINY_COORD}"

  echo "✓ ${SPLIT} done"
done

echo "All outputs under: ${OUT_ROOT}"
echo "  - {train,val}.jsonl (norm1000 ints)"
echo "  - {train,val}.coord.jsonl (coord tokens)"
echo "  - {train,val}_tiny.jsonl (norm1000 ints)"
echo "  - {train,val}_tiny.coord.jsonl (coord tokens)"
