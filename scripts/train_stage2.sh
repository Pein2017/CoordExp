#!/usr/bin/env bash
# Stage-2 AB combined launcher entrypoint (operator-facing shim).
#
# This script is intentionally thin:
# - Selects the YAML config path.
# - Selects the GPU split (server vs learner).
# - Delegates orchestration + validation to a repo-owned Python launcher.
#
# Example (single node, 8 GPUs; default 7 actors / 1 learner split):
#   server_gpus=0,1,2,3,4,5,6 \
#   train_gpus=7 \
#   config=configs/stage2_two_channel/smoke/ab_mixed_20steps.yaml \
#   bash scripts/train_stage2.sh

set -euo pipefail

if [[ $# -gt 0 ]]; then
  echo "[ERROR] scripts/train_stage2.sh accepts environment variables only (no positional args)." >&2
  echo "[ERROR] Example: server_gpus=0,1,2,3,4,5,6 train_gpus=7 config=configs/stage2_two_channel/prod/ab_mixed.yaml bash scripts/train_stage2.sh" >&2
  exit 2
fi

# Defaults (override via env vars)
export SERVER_GPUS="${server_gpus:-${SERVER_GPUS:-0,1,2,3,4,5,6}}"
export TRAIN_GPUS="${train_gpus:-${TRAIN_GPUS:-7}}"
export WAIT_TIMEOUT="${wait_timeout:-${WAIT_TIMEOUT:-900}}"
export WAIT_INTERVAL="${wait_interval:-${WAIT_INTERVAL:-2}}"
export CONFIG="${config:-${CONFIG:-configs/stage2_two_channel/smoke/ab_mixed_20steps.yaml}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}"

cd "${REPO_DIR}"
exec python -m src.launchers.stage2_vllm_server
