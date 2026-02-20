#!/usr/bin/env bash
# Pure config-driven training script
# All hyperparameters are in YAML config files
# This script only handles runtime settings

set -euo pipefail

if [[ $# -gt 0 ]]; then
  echo "[ERROR] scripts/train.sh accepts environment variables only (no positional args)." >&2
  echo "[ERROR] Example: config=configs/stage2_ab/prod/a_only.yaml gpus=0,1 bash scripts/train.sh" >&2
  exit 2
fi

# CUDA / NCCL runtime defaults (can be overridden by caller)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-67108864}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# Proxy hygiene: local vLLM server-mode uses localhost HTTP endpoints (health + infer).
# A global http(s)_proxy can route localhost traffic and break training.
DISABLE_PROXY="${disable_proxy:-${DISABLE_PROXY:-true}}"
if [[ "${DISABLE_PROXY}" == "true" || "${DISABLE_PROXY}" == "1" ]]; then
  # Merge existing NO_PROXY/no_proxy and ensure localhost entries are present.
  _np_raw="${NO_PROXY:-${no_proxy:-}}"
  IFS=',' read -r -a _np_split <<< "${_np_raw}"
  _np_tokens=()
  for _tok in "${_np_split[@]}"; do
    _tok="${_tok//[[:space:]]/}"
    [[ -n "${_tok}" ]] && _np_tokens+=("${_tok}")
  done
  for _need in "127.0.0.1" "localhost"; do
    _found="false"
    for _tok in "${_np_tokens[@]}"; do
      if [[ "${_tok}" == "${_need}" ]]; then
        _found="true"
        break
      fi
    done
    if [[ "${_found}" == "false" ]]; then
      _np_tokens+=("${_need}")
    fi
  done
  export NO_PROXY="$(IFS=','; echo "${_np_tokens[*]}")"
  export no_proxy="${NO_PROXY}"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
fi

# Resolve repository root from this script's location and set PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Use the currently active environment's Python binaries.
PYTHON_BIN=(python)
TORCHRUN_BIN=(torchrun)
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}"

# ============================================================================
# Runtime Settings (NOT training hyperparameters)
# ============================================================================

CONFIG_RAW="${config:-${CONFIG:-debug}}"
DEBUG="${debug:-${DEBUG:-false}}"

# GPU configuration
GPU_DEVICES="${gpus:-${GPU_DEVICES:-0}}"
CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"

# Derive number of GPUs (ignore empty/whitespace tokens)
IFS=',' read -r -a _raw_gpu_array <<< "${CUDA_VISIBLE_DEVICES}"
gpu_array=()
for _dev in "${_raw_gpu_array[@]}"; do
  [[ -n "${_dev// }" ]] && gpu_array+=("${_dev}")
done
NUM_GPUS="${#gpu_array[@]}"

# ============================================================================
# Build Command
# ============================================================================

## Resolve CONFIG_RAW to absolute path or repo-relative
if [[ "${CONFIG_RAW}" = /* ]]; then
  CONFIG_PATH="${CONFIG_RAW}"
elif [[ "${CONFIG_RAW}" == *.yaml ]]; then
  CONFIG_PATH="${REPO_DIR}/${CONFIG_RAW}"
else
  CONFIG_PATH="${REPO_DIR}/configs/${CONFIG_RAW}.yaml"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# ============================================================================
# CPU-Only Data Guards (Hard Errors)
# ============================================================================

PY_PRECHECK_CODE=$(cat <<'PY'
import os
import shlex
import sys

from src.config.loader import ConfigLoader
from src.trainers.rollout_matching.preflight import _resolve_path_for_config


def die(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    sys.exit(1)


def emit(name: str, value: object) -> None:
    print(f"{name}={shlex.quote(str(value))}")


config_path = os.environ.get("CONFIG_PATH")
if not config_path:
    die("CONFIG_PATH is required")

cfg = ConfigLoader.load_materialized_training_config(config_path)

train_jsonl_raw = getattr(cfg.custom, "train_jsonl", None)
val_jsonl_raw = getattr(cfg.custom, "val_jsonl", None)
if not isinstance(train_jsonl_raw, str) or not train_jsonl_raw.strip():
    die("custom.train_jsonl must be set for CPU JSONL validation")
if not isinstance(val_jsonl_raw, str) or not val_jsonl_raw.strip():
    die("custom.val_jsonl must be set for CPU JSONL validation")

train_jsonl = _resolve_path_for_config(train_jsonl_raw.strip(), config_path)
val_jsonl = _resolve_path_for_config(val_jsonl_raw.strip(), config_path)

max_pixels_raw = cfg.template.get("max_pixels")
if max_pixels_raw is None:
    die("template.max_pixels must be set (we treat it as a hard input constraint)")
try:
    max_pixels = int(max_pixels_raw)
except Exception as exc:
    die(f"template.max_pixels must be an int, got: {max_pixels_raw!r} ({exc})")
if max_pixels <= 0:
    die(f"template.max_pixels must be > 0, got: {max_pixels}")

emit("TRAIN_JSONL_RESOLVED", train_jsonl)
emit("VAL_JSONL_RESOLVED", val_jsonl)
emit("TEMPLATE_MAX_PIXELS", max_pixels)
PY
)

PRECHECK_VARS="$(
  CONFIG_PATH="${CONFIG_PATH}" "${PYTHON_BIN[@]}" -c "${PY_PRECHECK_CODE}"
)" || {
  echo "[ERROR] Failed to resolve JSONL precheck inputs from ${CONFIG_PATH}" >&2
  exit 1
}

eval "${PRECHECK_VARS}"

echo "========================================================================"
echo "[PRECHECK] Validating JSONL contracts + max_pixels before launching GPUs"
echo "========================================================================"
echo "[PRECHECK] train_jsonl: ${TRAIN_JSONL_RESOLVED}"
echo "[PRECHECK] val_jsonl: ${VAL_JSONL_RESOLVED}"
echo "[PRECHECK] max_pixels: ${TEMPLATE_MAX_PIXELS} (expect 768*32*32=786432)"
echo "[PRECHECK] multiple_of: 32"
echo "========================================================================"

"${PYTHON_BIN[@]}" "${REPO_DIR}/scripts/tools/validate_jsonl_max_pixels.py" \
  --jsonl "${TRAIN_JSONL_RESOLVED}" \
  --max-pixels "${TEMPLATE_MAX_PIXELS}" \
  --multiple-of 32 \
  --image-check-mode exists \
  --image-check-n 0

# Spot-check open+size alignment on train to catch any meta/image mismatch.
"${PYTHON_BIN[@]}" "${REPO_DIR}/scripts/tools/validate_jsonl_max_pixels.py" \
  --jsonl "${TRAIN_JSONL_RESOLVED}" \
  --max-pixels "${TEMPLATE_MAX_PIXELS}" \
  --multiple-of 32 \
  --image-check-mode open \
  --image-check-n 256

# Validate val with full open+size checks (usually small enough).
"${PYTHON_BIN[@]}" "${REPO_DIR}/scripts/tools/validate_jsonl_max_pixels.py" \
  --jsonl "${VAL_JSONL_RESOLVED}" \
  --max-pixels "${TEMPLATE_MAX_PIXELS}" \
  --multiple-of 32 \
  --image-check-mode open \
  --image-check-n 0

declare -a RUN_CMD=()
# Ensure MASTER_ADDR/MASTER_PORT are exported even for single-GPU runs so
# DeepSpeed never falls back to MPI discovery (libmpi/mpi4py may be absent).
if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT=$((10000 + RANDOM % 55536))
fi
if [[ -z "${MASTER_ADDR:-}" ]]; then
  MASTER_ADDR="127.0.0.1"
fi
export MASTER_ADDR MASTER_PORT

RUN_CMD=(
  "${TORCHRUN_BIN[@]}"
  --nproc_per_node="${NUM_GPUS}"
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT}"
  -m src.sft
  --config "${CONFIG_PATH}"
)


if [[ "${DEBUG}" == "true" ]]; then
  RUN_CMD+=(--debug)
fi

# ============================================================================
# Display Info & Execute
# ============================================================================

echo "========================================================================"
echo "  MS-Swift Training with YAML Configuration"
echo "========================================================================"
echo "[INFO] Config file: ${CONFIG_PATH}"
echo "[INFO] GPUs: ${CUDA_VISIBLE_DEVICES} (num=${NUM_GPUS})"
echo "[INFO] Master port: ${MASTER_PORT}"
echo "[INFO] Python: python (active environment)"
echo "[INFO] Debug mode: ${DEBUG}"
echo "[INFO] disable_proxy: ${DISABLE_PROXY}"
echo "========================================================================"
echo ""
printf '[RUN] (cwd=%s) CUDA_VISIBLE_DEVICES=%s ' "${REPO_DIR}" "${CUDA_VISIBLE_DEVICES}"
printf '%q ' "${RUN_CMD[@]}"
echo ""
echo ""

cd "${REPO_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${RUN_CMD[@]}"
