#!/usr/bin/env bash
# Backwards-compatible wrapper for the Stage-2 AB server-mode launcher.
#
# Canonical entrypoint: `scripts/train_stage2.sh`
# This wrapper exists to avoid breaking older docs and external run scripts.

set -euo pipefail

echo "[WARN] scripts/stage2_ab_server_train.sh is deprecated; use scripts/train_stage2.sh instead." >&2
if [[ $# -gt 0 ]]; then
  echo "[ERROR] scripts/stage2_ab_server_train.sh accepts environment variables only (no positional args)." >&2
  echo "[ERROR] Example: server_gpus=0,1,2,3,4,5,6 train_gpus=7 config=configs/stage2_ab/prod/a_only.yaml bash scripts/stage2_ab_server_train.sh" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/train_stage2.sh"
