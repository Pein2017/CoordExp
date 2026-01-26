#!/usr/bin/env bash
set -euo pipefail

# NOTE: This script used to be a fixed-config visualization helper.
# It is superseded by the unified pipeline and the vis CLI wrapper.
#
# Recommended:
#   PYTHONPATH=. conda run -n ms python vis_tools/vis_coordexp.py \
#     --pred_jsonl <run_dir>/gt_vs_pred.jsonl \
#     --save_dir  <run_dir>/vis \
#     --limit 20 \
#     --root_image_dir <image_root>

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "[DEPRECATED] scripts/vis.sh is superseded by vis_tools/vis_coordexp.py or scripts/run_vis.sh" >&2

exec "$SCRIPT_DIR/run_vis.sh"
