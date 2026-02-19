#!/usr/bin/env bash
# Backwards-compatible wrapper for the Stage-2 AB server-mode launcher.
#
# Canonical entrypoint: `scripts/train_stage2.sh`
# This wrapper exists to avoid breaking older docs and external run scripts.

set -euo pipefail

echo "[WARN] scripts/stage2_ab_server_train.sh is deprecated; use scripts/train_stage2.sh instead." >&2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/train_stage2.sh" "$@"

