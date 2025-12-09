#!/bin/bash
# Wrapper to run vis_tools/vis_coordexp.py.
# Tune the constants below before execution; no CLI args are parsed.

set -euo pipefail

# Fixed python interpreter
PYTHON_BIN="/root/miniconda3/envs/ms/bin/python"

# --------- Edit these constants as needed ---------
CUDA_DEVICES="1"
JSONL="public_data/lvis/rescale_32_768_poly_20/val.coord.jsonl"
SAVE_DIR="output/eval/12-4/det_eval_coord_ck1632"
CKPT="output/12-4/coord_merged_ck1632"
MODE="coord"  # coord | text
# --------------------------------------------------

echo "CUDA devices : $CUDA_DEVICES"
echo "JSONL        : $JSONL"
echo "Save dir     : $SAVE_DIR"
echo "Checkpoint   : $CKPT"

TMP_SCRIPT=$(mktemp /tmp/vis_coordexp.XXXXXX.py)
cat > "$TMP_SCRIPT" <<PY
from vis_tools import vis_coordexp as vc

vc.CONFIG = vc.Config(
    ckpt="${CKPT}",
    jsonl="${JSONL}",
    device="cuda:0",  # index within CUDA_VISIBLE_DEVICES
    limit=0,
    temperature=0.001,
    repetition_penalty=1.05,
    save_dir="${SAVE_DIR}",
    mode="${MODE}",
)

vc.main()
PY

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" "$TMP_SCRIPT"
rm -f "$TMP_SCRIPT"

echo "Done. Outputs in ${SAVE_DIR}"
