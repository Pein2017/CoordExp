#!/usr/bin/env bash
set -euo pipefail

# NOTE: This wrapper historically ran inference+vis for coord-mode checkpoints.
# It drifted over time; the unified pipeline supersedes it.
#
# Replacement (recommended):
#   python scripts/run_infer.py --config configs/infer/pipeline.yaml
#
# Or, for vis-only from an existing artifact:
#   PRED_JSONL=<run_dir>/gt_vs_pred.jsonl SAVE_DIR=<out_dir> ROOT_IMAGE_DIR=<img_root> scripts/run_vis.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "[DEPRECATED] scripts/run_vis_coord.sh is superseded by the unified YAML pipeline." >&2

exec "$SCRIPT_DIR/run_vis.sh"
