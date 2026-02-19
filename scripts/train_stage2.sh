#!/usr/bin/env bash
# Launch vLLM rollout server (swift rollout) + Stage-2 AB learner training in one entrypoint.
#
# Example (single node, 8 GPUs; default 6 actors / 2 learners split):
#   bash scripts/train_stage2.sh \
#     server_gpus=0,1,2,3,4,5 train_gpus=6,7 \
#     config=configs/stage2_ab/smoke/ab_mixed.yaml

set -euo pipefail

# Allow passing key=value pairs as positional args (common launcher convention).
for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "[ERROR] Unknown argument: ${arg} (expected key=value)" >&2
    exit 2
  fi
  key="${arg%%=*}"
  if [[ ! "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    echo "[ERROR] Invalid argument name in: ${arg}" >&2
    exit 2
  fi
  export "${arg}"
done

# Defaults (override via env vars)
CONDA_ENV="${CONDA_ENV:-ms}"
SERVER_GPUS="${server_gpus:-0,1,2,3,4,5,6}"
TRAIN_GPUS="${train_gpus:-7}"
WAIT_TIMEOUT="${wait_timeout:-900}"
WAIT_INTERVAL="${wait_interval:-2}"
CONFIG_RAW="${config:-configs/stage2_ab/smoke/ab_mixed.yaml}"
DEBUG="${debug:-false}"
TRAIN_ENV="${train_env:-}"
DISABLE_PROXY="${disable_proxy:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${CONFIG_RAW}" = /* ]]; then
  CONFIG_PATH="${CONFIG_RAW}"
elif [[ "${CONFIG_RAW}" == configs/* ]]; then
  CONFIG_PATH="${REPO_DIR}/${CONFIG_RAW}"
elif [[ "${CONFIG_RAW}" == *.yaml ]]; then
  CONFIG_PATH="${REPO_DIR}/configs/${CONFIG_RAW}"
else
  CONFIG_PATH="${REPO_DIR}/configs/${CONFIG_RAW}.yaml"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

_maybe_disable_proxy() {
  if [[ "${DISABLE_PROXY}" == "true" || "${DISABLE_PROXY}" == "1" ]]; then
    # Ensure localhost bypass even if caller uses a global proxy.
    local np_raw np_out
    np_raw="${NO_PROXY:-${no_proxy:-}}"
    np_out="${np_raw}"
    if [[ -n "${np_out}" && "${np_out}" != *, ]]; then
      np_out+=","
    fi
    if [[ "${np_out}" != *"127.0.0.1"* ]]; then
      np_out+="127.0.0.1,"
    fi
    if [[ "${np_out}" != *"localhost"* ]]; then
      np_out+="localhost,"
    fi
    np_out="${np_out%,}"
    export NO_PROXY="${np_out}"
    export no_proxy="${NO_PROXY}"
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
  fi
}

_maybe_disable_proxy

PY_CODE=$(cat <<'PY'
import json
import os
import shlex
import sys
from src.trainers.rollout_matching.preflight import resolve_stage2_launcher_preflight


def die(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    sys.exit(1)


def emit(name: str, value: object) -> None:
    print(f"{name}={shlex.quote(str(value))}")


config_path = os.environ.get("CONFIG_PATH")
if not config_path:
    die("CONFIG_PATH is required")

try:
    preflight = resolve_stage2_launcher_preflight(config_path)
except Exception as exc:
    die(f"Failed to resolve Stage-2 launcher preflight payload: {exc}")

required_schema = {
    "rollout_backend": str,
    "vllm_mode": (str, type(None)),
    "server_base_urls": list,
}
for key, expected_type in required_schema.items():
    if key not in preflight:
        die(f"Preflight payload missing required key: {key}")
    if not isinstance(preflight[key], expected_type):
        die(
            f"Preflight key {key} has invalid type {type(preflight[key]).__name__}; "
            f"expected {expected_type}."
        )
if not all(isinstance(url, str) for url in preflight["server_base_urls"]):
    die("Preflight key server_base_urls must be a list of strings.")

rollout_contract = {
    "rollout_backend": preflight["rollout_backend"],
    "vllm_mode": preflight["vllm_mode"],
    "server_base_urls": preflight["server_base_urls"],
}
emit(
    "ROLLOUT_CONTRACT_JSON",
    json.dumps(rollout_contract, ensure_ascii=False, separators=(",", ":")),
)

emit("SERVER_MODEL", preflight["server_model"])
emit("ROOT_IMAGE_DIR_RESOLVED", preflight["root_image_dir_resolved"])
emit("VLLM_MAX_MODEL_LEN", int(preflight["vllm_max_model_len"]))
emit("VLLM_ENABLE_LORA", "true" if bool(preflight["vllm_enable_lora"]) else "false")
PY
)

CONFIG_VARS="$(
  CONFIG_PATH="${CONFIG_PATH}" conda run -n "${CONDA_ENV}" python -c "${PY_CODE}"
)" || {
  echo "[ERROR] Failed to resolve vLLM server settings from ${CONFIG_PATH}" >&2
  exit 1
}

eval "${CONFIG_VARS}"

SERVER_HOST_PORT="$(
  ROLLOUT_CONTRACT_JSON="${ROLLOUT_CONTRACT_JSON:-}" python - <<'PY'
import json
import os
import shlex
import sys
from urllib.parse import urlparse

def die(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    sys.exit(1)

contract_raw = os.environ.get("ROLLOUT_CONTRACT_JSON")
if not contract_raw:
    die("ROLLOUT_CONTRACT_JSON is missing; rollout resolver failed.")
try:
    contract = json.loads(contract_raw)
except Exception as exc:
    die(f"Failed to parse rollout contract JSON: {exc}")

backend = contract.get("rollout_backend")
mode = contract.get("vllm_mode")
urls = contract.get("server_base_urls")
if backend != "vllm":
    die(f"rollout_backend must be 'vllm', got: {backend!r}")
if mode != "server":
    die(f"vllm_mode must be 'server', got: {mode!r}")
if not isinstance(urls, list) or not urls:
    die("server_base_urls must be a non-empty list.")

first_url = urls[0]
if not isinstance(first_url, str):
    die("server_base_urls entries must be strings.")
first_url = first_url.strip()
if not first_url:
    die("server_base_urls entries must be non-empty.")

parsed = urlparse(first_url)
if parsed.scheme not in ("http", "https"):
    die(f"base_url must be http(s), got: {first_url!r}")
host = parsed.hostname
port = parsed.port
if not host or not port:
    die(f"base_url must include host and port, got: {first_url!r}")

def emit(name: str, value: object) -> None:
    print(f"{name}={shlex.quote(str(value))}")

emit("SERVER_HOST", host)
emit("SERVER_PORT", port)
emit("PRIMARY_SERVER_BASE_URL", first_url)
PY
)"
eval "${SERVER_HOST_PORT}"

# Resolve model dir to an absolute path (ms-swift treats missing local paths as hub IDs).
if [[ "${SERVER_MODEL}" != /* ]]; then
  SERVER_MODEL="${REPO_DIR}/${SERVER_MODEL}"
fi
if [[ ! -d "${SERVER_MODEL}" ]]; then
  echo "[ERROR] Server model directory not found: ${SERVER_MODEL}" >&2
  exit 1
fi

# Derive data-parallel size from SERVER_GPUS unless explicitly overridden.
IFS=',' read -r -a _server_gpu_array <<< "${SERVER_GPUS}"
server_gpu_array=()
for _dev in "${_server_gpu_array[@]}"; do
  [[ -n "${_dev// }" ]] && server_gpu_array+=("${_dev}")
done

# Derive learner world size from TRAIN_GPUS.
IFS=',' read -r -a _train_gpu_array <<< "${TRAIN_GPUS}"
train_gpu_array=()
for _dev in "${_train_gpu_array[@]}"; do
  [[ -n "${_dev// }" ]] && train_gpu_array+=("${_dev}")
done
TRAIN_WORLD_SIZE="${#train_gpu_array[@]}"
if [[ "${TRAIN_WORLD_SIZE}" -le 0 ]]; then
  echo "[ERROR] train_gpus must contain at least one device id. train_gpus=${TRAIN_GPUS}" >&2
  exit 2
fi

# Validate that server and learner GPU sets are disjoint (strict role split).
declare -A _gpu_seen=()
for _dev in "${server_gpu_array[@]}"; do
  _gpu_seen[$_dev]="server"
done
for _dev in "${train_gpu_array[@]}"; do
  if [[ -n "${_gpu_seen[$_dev]+x}" ]]; then
    echo "[ERROR] server_gpus and train_gpus must be disjoint. Overlap on GPU ${_dev}." >&2
    echo "[ERROR] server_gpus=${SERVER_GPUS} train_gpus=${TRAIN_GPUS}" >&2
    exit 2
  fi
  _gpu_seen[$_dev]="train"
done

SERVER_DP="${server_dp:-${#server_gpu_array[@]}}"
SERVER_TP="${server_tp:-1}"

# Server runtime knobs (kept config-free; affects only server launch)
SERVER_TORCH_DTYPE="${server_torch_dtype:-bfloat16}"
SERVER_VLLM_ENFORCE_EAGER="${server_vllm_enforce_eager:-true}"

# Default server parallelism: **data-parallel first** (tp=1, dp=#gpus).
# Rationale: on nodes where a single GPU can fit the full model, DP maximizes
# rollout throughput and keeps per-request latency stable.
#
# If caller sets `server_tp>1` (and leaves `server_dp` unset), we derive
# `server_dp = n_gpus / server_tp` so TP+DP combinations still work without
# extra knobs.
if [[ -z "${server_dp:-}" ]]; then
  if [[ "${SERVER_TP}" -le 0 ]]; then
    echo "[ERROR] server_tp must be >= 1. Got server_tp=${SERVER_TP}" >&2
    exit 2
  fi
  if (( ${#server_gpu_array[@]} % SERVER_TP != 0 )); then
    echo "[ERROR] server_gpus count must be divisible by server_tp when server_dp is not set." >&2
    echo "[ERROR] server_gpus=${SERVER_GPUS} (n=${#server_gpu_array[@]}) server_tp=${SERVER_TP}" >&2
    exit 2
  fi
  SERVER_DP=$(( ${#server_gpu_array[@]} / SERVER_TP ))
fi

# Default lower utilization for stability (avoid borderline OOM on busy nodes).
VLLM_GPU_MEMORY_UTILIZATION="${vllm_gpu_memory_utilization:-0.75}"

echo "========================================================================"
echo "  Stage-2 AB vLLM Server + Learner Launcher"
echo "========================================================================"
echo "[INFO] Config:      ${CONFIG_PATH}"
echo "[INFO] Server GPUs: ${SERVER_GPUS} (dp=${SERVER_DP}, tp=${SERVER_TP})"
echo "[INFO] Train GPUs:  ${TRAIN_GPUS} (world_size=${TRAIN_WORLD_SIZE})"
echo "[INFO] Server:      ${SERVER_HOST}:${SERVER_PORT}"
echo "[INFO] Model:       ${SERVER_MODEL}"
echo "[INFO] ROOT_IMAGE_DIR:${ROOT_IMAGE_DIR_RESOLVED}"
echo "[INFO] torch_dtype: ${SERVER_TORCH_DTYPE}"
echo "[INFO] eager:       ${SERVER_VLLM_ENFORCE_EAGER}"
echo "[INFO] max_model_len:${VLLM_MAX_MODEL_LEN}"
echo "[INFO] enable_lora: ${VLLM_ENABLE_LORA}"
echo "[INFO] disable_proxy:${DISABLE_PROXY}"
echo "========================================================================"

SERVER_ENV=(CUDA_VISIBLE_DEVICES="${SERVER_GPUS}")
if [[ -n "${NO_PROXY:-}" ]]; then
  SERVER_ENV+=(NO_PROXY="${NO_PROXY}" no_proxy="${NO_PROXY}")
fi
SERVER_ENV+=(ROOT_IMAGE_DIR="${ROOT_IMAGE_DIR_RESOLVED}")

SERVER_CMD=(conda run -n "${CONDA_ENV}" swift rollout \
  --model "${SERVER_MODEL}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --infer_backend vllm \
  --torch_dtype "${SERVER_TORCH_DTYPE}" \
  --vllm_data_parallel_size "${SERVER_DP}" \
  --vllm_tensor_parallel_size "${SERVER_TP}" \
  --vllm_enforce_eager "${SERVER_VLLM_ENFORCE_EAGER}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
  --vllm_enable_lora "${VLLM_ENABLE_LORA}")

echo "[RUN] ${SERVER_ENV[*]} ${SERVER_CMD[*]}"
if command -v setsid >/dev/null 2>&1; then
  setsid env "${SERVER_ENV[@]}" "${SERVER_CMD[@]}" &
  SERVER_PID=$!
  SERVER_PGID="$(ps -o pgid= "${SERVER_PID}" | tr -d ' ')"
else
  env "${SERVER_ENV[@]}" "${SERVER_CMD[@]}" &
  SERVER_PID=$!
  SERVER_PGID=""
fi

cleanup() {
  if [[ -n "${SERVER_PGID}" ]]; then
    echo "[INFO] Stopping vLLM server process group (pgid ${SERVER_PGID})"
    kill -TERM "-${SERVER_PGID}" >/dev/null 2>&1 || true
    sleep 3
    kill -KILL "-${SERVER_PGID}" >/dev/null 2>&1 || true
  elif [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "[INFO] Stopping vLLM server (pid ${SERVER_PID})"
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" || true
  fi
}
trap cleanup EXIT INT TERM

HEALTH_HOST="${SERVER_HOST}"
if [[ "${HEALTH_HOST}" == "0.0.0.0" ]]; then
  HEALTH_HOST="127.0.0.1"
fi
HEALTH_URL="http://${HEALTH_HOST}:${SERVER_PORT}/health/"
WORLD_URL="http://${HEALTH_HOST}:${SERVER_PORT}/get_world_size/"

echo "[INFO] Waiting for vLLM server readiness: ${HEALTH_URL}"
start_ts=$(date +%s)
while true; do
  http_code=$(curl --noproxy "*" -s -o /dev/null -w '%{http_code}' "${HEALTH_URL}" || true)
  if [[ "${http_code}" == "200" ]]; then
    break
  fi
  now_ts=$(date +%s)
  if (( now_ts - start_ts > WAIT_TIMEOUT )); then
    echo "[ERROR] vLLM server did not become ready within ${WAIT_TIMEOUT}s. Last HTTP code: ${http_code}" >&2
    exit 1
  fi
  sleep "${WAIT_INTERVAL}"
done

echo "[INFO] vLLM server is ready. world_size: $(curl --noproxy \"*\" -s \"${WORLD_URL}\" || true)"

STAGE2_META_ENV_VARS=(
  "COORDEXP_STAGE2_LAUNCHER=scripts/train_stage2.sh"
  "COORDEXP_STAGE2_SERVER_BASE_URL=${PRIMARY_SERVER_BASE_URL}"
  "COORDEXP_STAGE2_SERVER_MODEL=${SERVER_MODEL}"
  "COORDEXP_STAGE2_SERVER_TORCH_DTYPE=${SERVER_TORCH_DTYPE}"
  "COORDEXP_STAGE2_SERVER_DP=${SERVER_DP}"
  "COORDEXP_STAGE2_SERVER_TP=${SERVER_TP}"
  "COORDEXP_STAGE2_SERVER_ENFORCE_EAGER=${SERVER_VLLM_ENFORCE_EAGER}"
  "COORDEXP_STAGE2_SERVER_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}"
  "COORDEXP_STAGE2_SERVER_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}"
  "COORDEXP_STAGE2_SERVER_ENABLE_LORA=${VLLM_ENABLE_LORA}"
  "COORDEXP_STAGE2_SERVER_GPUS=${SERVER_GPUS}"
  "COORDEXP_STAGE2_LEARNER_GPUS=${TRAIN_GPUS}"
)

TRAIN_CMD="config=${CONFIG_PATH} gpus=${TRAIN_GPUS} bash ${REPO_DIR}/scripts/train.sh"
if [[ "${DEBUG}" == "true" ]]; then
  TRAIN_CMD="debug=true ${TRAIN_CMD}"
fi
if [[ -n "${TRAIN_ENV}" ]]; then
  TRAIN_CMD="${TRAIN_ENV} ${TRAIN_CMD}"
fi
TRAIN_CMD="${STAGE2_META_ENV_VARS[*]} ${TRAIN_CMD}"

echo "[RUN] ${TRAIN_CMD}"
eval "${TRAIN_CMD}"
