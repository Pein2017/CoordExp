#!/usr/bin/env bash
set -euo pipefail

# Thin bash wrapper around the Python task manager.
#
# Queue format: JSONL (blank lines and lines starting with `#` are ignored).
# Each row is a JSON object.
#
# Backward-compatible simple form:
#   {"name":"watch-current","session":"a_only","launch":false,"config":"configs/stage2_two_channel/prod/a_only.yaml","stop_after_step":300,"require_eval":true}
#   {"name":"next-task","session":"a_only","config":"configs/stage2_two_channel/prod/a_only_iter2.yaml","gpus":"0","stop_after_step":300,"require_eval":true}
#
# Extended flexible form:
#   {
#     "name": "watch-current",
#     "session": "a_only",
#     "launch": false,
#     "config": "configs/stage2_two_channel/prod/a_only.yaml",
#     "monitor_after_seconds": 7200,
#     "poll_seconds": 600,
#     "criteria_mode": "all",
#     "criteria": [
#       {"type": "global_step", "value": 300, "require_eval": true}
#     ],
#     "post_stop_wait_seconds": 15,
#     "gpu_zero": {
#       "enabled": true,
#       "devices": "all",
#       "timeout_seconds": 300,
#       "poll_seconds": 5,
#       "memory_threshold_mb": 0
#     }
#   }
#   {
#     "name": "next-task",
#     "session": "a_only",
#     "command": "config=configs/stage2_two_channel/prod/a_only_iter2.yaml gpus=0 bash scripts/train.sh"
#   }
#
# Usage:
#   queue_file=temp/train_queue.jsonl bash scripts/pipelines/train_task_manager.sh

if [[ $# -gt 0 ]]; then
  echo "[ERROR] scripts/pipelines/train_task_manager.sh accepts environment variables only (no positional args)." >&2
  echo "[ERROR] Example: queue_file=temp/train_queue.jsonl bash scripts/pipelines/train_task_manager.sh" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_lib/backbone.sh"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

QUEUE_FILE_RAW="${queue_file:-${QUEUE_FILE:-}}"
ensure_required "queue_file" "${QUEUE_FILE_RAW}"

POLL_SECONDS="${poll_seconds:-${POLL_SECONDS:-15}}"
RUN_DIR_TIMEOUT_S="${run_dir_timeout_s:-${RUN_DIR_TIMEOUT_S:-300}}"
STOP_TIMEOUT_S="${stop_timeout_s:-${STOP_TIMEOUT_S:-180}}"
POST_STOP_WAIT_S="${post_stop_wait_s:-${POST_STOP_WAIT_S:-0}}"
GPU_ZERO_TIMEOUT_S="${gpu_zero_timeout_s:-${GPU_ZERO_TIMEOUT_S:-300}}"
GPU_ZERO_POLL_SECONDS="${gpu_zero_poll_seconds:-${GPU_ZERO_POLL_SECONDS:-5}}"

"${COORDEXP_PYTHON[@]}" "${SCRIPT_DIR}/train_task_manager.py" \
  --queue-file "${QUEUE_FILE_RAW}" \
  --default-poll-seconds "${POLL_SECONDS}" \
  --default-run-dir-timeout-s "${RUN_DIR_TIMEOUT_S}" \
  --default-stop-timeout-s "${STOP_TIMEOUT_S}" \
  --default-post-stop-wait-s "${POST_STOP_WAIT_S}" \
  --default-gpu-zero-timeout-s "${GPU_ZERO_TIMEOUT_S}" \
  --default-gpu-zero-poll-seconds "${GPU_ZERO_POLL_SECONDS}"
