#!/usr/bin/env bash
# Pure config-driven training script
# All hyperparameters are in YAML config files
# This script only handles runtime settings

set -euo pipefail

# CUDA / NCCL runtime defaults (can be overridden by caller)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-67108864}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# Proxy hygiene: local vLLM server-mode uses localhost HTTP endpoints (health + infer).
# A global http(s)_proxy can route localhost traffic and break training.
DISABLE_PROXY="${disable_proxy:-true}"
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
PYTHON_BIN="/root/miniconda3/envs/ms/bin/python"
TORCHRUN_BIN="/root/miniconda3/envs/ms/bin/torchrun"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}"

# ============================================================================
# Runtime Settings (NOT training hyperparameters)
# ============================================================================

CONFIG_RAW="${config:-debug}"
DEBUG="${debug:-false}"

# GPU configuration
CUDA_VISIBLE_DEVICES="${gpus:-0}"

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

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  # Generate random port in range [10000, 65535] if not already set
  if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT=$((10000 + RANDOM % 55536))
  fi
  # Ensure MASTER_ADDR/MASTER_PORT are exported so DeepSpeed (used by ms-swift in some trainers)
  # does not fall back to MPI discovery (which requires libmpi/mpi4py on the system).
  if [[ -z "${MASTER_ADDR:-}" ]]; then
    MASTER_ADDR="127.0.0.1"
  fi
  export MASTER_ADDR MASTER_PORT

  CMD="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${TORCHRUN_BIN} --nproc_per_node=${NUM_GPUS} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} -m src.sft --config ${CONFIG_PATH}"
else
  CMD="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${PYTHON_BIN} -m src.sft --config ${CONFIG_PATH}"
fi


if [[ "${DEBUG}" == "true" ]]; then
  CMD+=" --debug"
fi

# ============================================================================
# Display Info & Execute
# ============================================================================

echo "========================================================================"
echo "  MS-Swift Training with YAML Configuration"
echo "========================================================================"
echo "[INFO] Config file: ${CONFIG_PATH}"
echo "[INFO] GPUs: ${CUDA_VISIBLE_DEVICES} (num=${NUM_GPUS})"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
echo "[INFO] Master port: ${MASTER_PORT}"
fi
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Debug mode: ${DEBUG}"
echo "[INFO] disable_proxy: ${DISABLE_PROXY}"
echo "========================================================================"
echo ""
echo "[RUN] (cwd=${REPO_DIR}) ${CMD}"
echo ""

cd "${REPO_DIR}"
eval "${CMD}"
