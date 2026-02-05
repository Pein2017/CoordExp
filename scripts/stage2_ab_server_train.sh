#!/usr/bin/env bash
# Launch vLLM rollout server (swift rollout) + Stage-2 AB learner training in one entrypoint.
#
# Example (single node, 4 GPUs):
#   bash scripts/stage2_ab_server_train.sh \
#     server_gpus=0,1,2 train_gpus=3 \
#     config=configs/stage2_ab/smoke/bbox_max60_ckpt1516_ab_mixed_vllm_server.yaml

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
SERVER_GPUS="${server_gpus:-0,1,2}"
TRAIN_GPUS="${train_gpus:-3}"
WAIT_TIMEOUT="${wait_timeout:-900}"
WAIT_INTERVAL="${wait_interval:-2}"
CONFIG_RAW="${config:-configs/stage2_ab/smoke/bbox_max60_ckpt1516_ab_mixed_vllm_server.yaml}"
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
import os
import shlex
import sys
from pathlib import Path
from urllib.parse import urlparse

try:
    import yaml
except Exception as exc:  # pragma: no cover
    print("[ERROR] PyYAML is required to parse training configs.", file=sys.stderr)
    raise


def die(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    sys.exit(1)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        die(f"Top-level YAML config must be a mapping: {path}")
    return data


def normalize_to_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def merge_configs(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            merged[key] = merge_configs(existing, value)
        else:
            merged[key] = value
    return merged


def load_with_extends(path: Path, visited: set[Path] | None = None) -> dict:
    abs_path = path.resolve()
    visited = set() if visited is None else visited
    if abs_path in visited:
        die(f"Cyclic config inheritance detected at: {abs_path}")
    visited.add(abs_path)

    config = load_yaml(abs_path)
    extends_value = None
    if isinstance(config, dict):
        extends_value = config.pop("extends", None)
        if extends_value is None:
            extends_value = config.pop("inherit", None)

    merged_base: dict = {}
    for base_ref in normalize_to_list(extends_value):
        base_path = Path(str(base_ref))
        if not base_path.is_absolute():
            base_path = (abs_path.parent / base_path).resolve()
        merged_base = merge_configs(merged_base, load_with_extends(base_path, visited))

    return merge_configs(merged_base, config)


config_path = os.environ.get("CONFIG_PATH")
if not config_path:
    die("CONFIG_PATH is required")

cfg = load_with_extends(Path(config_path))
model = cfg.get("model") or {}
custom = cfg.get("custom") or {}
extra = (custom.get("extra") or {}) if isinstance(custom, dict) else {}
rm = (extra.get("rollout_matching") or {}) if isinstance(extra, dict) else {}
vllm = (rm.get("vllm") or {}) if isinstance(rm, dict) else {}
server = (vllm.get("server") or {}) if isinstance(vllm, dict) else {}
servers = server.get("servers") if isinstance(server, dict) else None

if rm.get("rollout_backend") != "vllm":
    die("custom.extra.rollout_matching.rollout_backend must be 'vllm' for server-mode launch.")
if vllm.get("mode") != "server":
    die("custom.extra.rollout_matching.vllm.mode must be 'server' for server-mode launch.")
if not isinstance(servers, list) or not servers:
    die("custom.extra.rollout_matching.vllm.server.servers must be a non-empty list.")

base_url = servers[0].get("base_url") if isinstance(servers[0], dict) else None
if not base_url:
    die("custom.extra.rollout_matching.vllm.server.servers[0].base_url is required.")

parsed = urlparse(str(base_url))
if parsed.scheme not in ("http", "https"):
    die(f"base_url must be http(s), got: {base_url!r}")
host = parsed.hostname
port = parsed.port
if not host or not port:
    die(f"base_url must include host and port, got: {base_url!r}")

model_path = model.get("model")
if not model_path:
    die("model.model must be set (rollout model path).")

max_model_len = None
try:
    if isinstance(vllm, dict):
        max_model_len = vllm.get("max_model_len")
except Exception:
    max_model_len = None

# Fallback for legacy configs that did not set vLLM max_model_len explicitly.
if max_model_len is None:
    max_model_len = cfg.get("global_max_length") or 8192

try:
    max_model_len = int(max_model_len)
except Exception as exc:
    die(f"vllm.max_model_len (or global_max_length fallback) must be an int: {exc}")

enable_lora = vllm.get("enable_lora", False)
enable_lora = bool(enable_lora)


def emit(name: str, value: object) -> None:
    print(f"{name}={shlex.quote(str(value))}")


emit("SERVER_HOST", host)
emit("SERVER_PORT", port)
emit("SERVER_MODEL", model_path)
emit("VLLM_MAX_MODEL_LEN", max_model_len)
emit("VLLM_ENABLE_LORA", "true" if enable_lora else "false")
PY
)

CONFIG_VARS="$(
  CONFIG_PATH="${CONFIG_PATH}" conda run -n "${CONDA_ENV}" python -c "${PY_CODE}"
)" || {
  echo "[ERROR] Failed to resolve vLLM server settings from ${CONFIG_PATH}" >&2
  exit 1
}

eval "${CONFIG_VARS}"

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

TRAIN_CMD="config=${CONFIG_PATH} gpus=${TRAIN_GPUS} bash ${REPO_DIR}/scripts/train.sh"
if [[ "${DEBUG}" == "true" ]]; then
  TRAIN_CMD="debug=true ${TRAIN_CMD}"
fi
if [[ -n "${TRAIN_ENV}" ]]; then
  TRAIN_CMD="${TRAIN_ENV} ${TRAIN_CMD}"
fi

echo "[RUN] ${TRAIN_CMD}"
eval "${TRAIN_CMD}"
