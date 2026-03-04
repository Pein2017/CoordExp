#!/usr/bin/env bash
# Launch vLLM rollout server (swift rollout) + Stage-2 AB learner training in one entrypoint.
# Assumes the target environment is already activated.
#
# Example (single node, 8 GPUs; default 7 actors / 1 learner split):
#   server_gpus=0,1,2,3,4,5,6 \
#   train_gpus=7 \
#   config=configs/stage2_two_channel/smoke/ab_mixed.yaml \
#   bash scripts/train_stage2.sh

set -euo pipefail

if [[ $# -gt 0 ]]; then
  echo "[ERROR] scripts/train_stage2.sh accepts environment variables only (no positional args)." >&2
  echo "[ERROR] Example: server_gpus=0,1,2,3,4,5,6 train_gpus=7 config=configs/stage2_two_channel/prod/ab_mixed.yaml bash scripts/train_stage2.sh" >&2
  exit 2
fi

# Defaults (override via env vars)
SERVER_GPUS="${server_gpus:-${SERVER_GPUS:-0,1,2,3,4,5,6}}"
TRAIN_GPUS="${train_gpus:-${TRAIN_GPUS:-7}}"
WAIT_TIMEOUT="${wait_timeout:-${WAIT_TIMEOUT:-900}}"
WAIT_INTERVAL="${wait_interval:-${WAIT_INTERVAL:-2}}"
CONFIG_RAW="${config:-${CONFIG:-configs/stage2_two_channel/smoke/ab_mixed.yaml}}"
DEBUG="${debug:-${DEBUG:-false}}"
TRAIN_ENV="${train_env:-${TRAIN_ENV:-}}"
DISABLE_PROXY="${disable_proxy:-${DISABLE_PROXY:-true}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}"

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
    "eval_rollout_backend": str,
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
    "eval_rollout_backend": preflight["eval_rollout_backend"],
    "vllm_mode": preflight["vllm_mode"],
    "server_base_urls": preflight["server_base_urls"],
}
emit(
    "ROLLOUT_CONTRACT_JSON",
    json.dumps(rollout_contract, ensure_ascii=False, separators=(",", ":")),
)

emit("SERVER_MODEL", preflight["server_model"])
emit("TRAIN_JSONL_RESOLVED", preflight["train_jsonl_resolved"])
emit("VAL_JSONL_RESOLVED", preflight.get("val_jsonl_resolved") or "")
emit("TEMPLATE_MAX_PIXELS", int(preflight["template_max_pixels"]))
emit("SERVER_TEMPLATE", preflight.get("server_template") or "")
server_max_length = preflight.get("server_max_length")
emit("SERVER_MAX_LENGTH", "" if server_max_length is None else int(server_max_length))
emit("SERVER_TRUNCATION_STRATEGY", preflight.get("server_truncation_strategy") or "")
emit("ROOT_IMAGE_DIR_RESOLVED", preflight["root_image_dir_resolved"])
emit("VLLM_TENSOR_PARALLEL_SIZE_CFG", int(preflight["vllm_tensor_parallel_size"]))
emit(
    "VLLM_ENFORCE_EAGER_CFG",
    "true" if bool(preflight["vllm_enforce_eager"]) else "false",
)
emit("VLLM_MAX_MODEL_LEN", int(preflight["vllm_max_model_len"]))
emit("VLLM_ENABLE_LORA", "true" if bool(preflight["vllm_enable_lora"]) else "false")
engine_kwargs = preflight.get("vllm_engine_kwargs") or {}
emit(
    "VLLM_ENGINE_KWARGS_JSON",
    ""
    if not engine_kwargs
    else json.dumps(engine_kwargs, ensure_ascii=False, separators=(",", ":")),
)
gpu_mem = preflight.get("vllm_gpu_memory_utilization")
emit("VLLM_GPU_MEMORY_UTILIZATION_CFG", "" if gpu_mem is None else gpu_mem)
emit("SERVER_TORCH_DTYPE_CFG", preflight.get("server_torch_dtype") or "")
PY
)

CONFIG_VARS="$(
  CONFIG_PATH="${CONFIG_PATH}" python -c "${PY_CODE}"
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
eval_backend = contract.get("eval_rollout_backend")
mode = contract.get("vllm_mode")
urls = contract.get("server_base_urls")
if backend not in ("hf", "vllm"):
    die(f"rollout_backend must be 'hf' or 'vllm', got: {backend!r}")
if eval_backend != "vllm":
    die(f"eval_rollout_backend must be 'vllm', got: {eval_backend!r}")
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

# vLLM server launch knobs are config-derived (YAML is the single source of truth).
SERVER_TP="${VLLM_TENSOR_PARALLEL_SIZE_CFG:-1}"
SERVER_VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER_CFG:-false}"
SERVER_TORCH_DTYPE="${SERVER_TORCH_DTYPE_CFG:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION_CFG:-0.75}"

# Resolve model dir to an absolute path (ms-swift treats missing local paths as hub IDs).
if [[ "${SERVER_MODEL}" != /* ]]; then
  SERVER_MODEL="${REPO_DIR}/${SERVER_MODEL}"
fi
if [[ ! -d "${SERVER_MODEL}" ]]; then
  echo "[ERROR] Server model directory not found: ${SERVER_MODEL}" >&2
  exit 1
fi

# ============================================================================
# CPU-Only Data Guards (Hard Errors)
# ============================================================================

if [[ -z "${TRAIN_JSONL_RESOLVED:-}" ]]; then
  echo "[ERROR] custom.train_jsonl is required (resolved empty). Check ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ -z "${VAL_JSONL_RESOLVED:-}" ]]; then
  echo "[ERROR] custom.val_jsonl is required (resolved empty). Check ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ -z "${TEMPLATE_MAX_PIXELS:-}" ]]; then
  echo "[ERROR] template.max_pixels is required (resolved empty). Check ${CONFIG_PATH}" >&2
  exit 1
fi

echo "========================================================================"
echo "[PRECHECK] Validating JSONL contracts + max_pixels before launching GPUs"
echo "========================================================================"
echo "[PRECHECK] train_jsonl: ${TRAIN_JSONL_RESOLVED}"
echo "[PRECHECK] val_jsonl: ${VAL_JSONL_RESOLVED}"
echo "[PRECHECK] max_pixels: ${TEMPLATE_MAX_PIXELS} (expect 768*32*32=786432)"
echo "[PRECHECK] multiple_of: 32"
echo "========================================================================"

python "${REPO_DIR}/public_data/scripts/validate_jsonl.py" \
  "${TRAIN_JSONL_RESOLVED}" \
  --max-pixels "${TEMPLATE_MAX_PIXELS}" \
  --multiple-of 32 \
  --image-check-mode exists \
  --enforce-rescale-images-real-dir \
  --image-check-n 0

# Spot-check open+size alignment on train to catch any meta/image mismatch.
python "${REPO_DIR}/public_data/scripts/validate_jsonl.py" \
  "${TRAIN_JSONL_RESOLVED}" \
  --max-pixels "${TEMPLATE_MAX_PIXELS}" \
  --multiple-of 32 \
  --image-check-mode open \
  --enforce-rescale-images-real-dir \
  --image-check-n 256

# Validate val with full open+size checks (usually small enough).
python "${REPO_DIR}/public_data/scripts/validate_jsonl.py" \
  "${VAL_JSONL_RESOLVED}" \
  --max-pixels "${TEMPLATE_MAX_PIXELS}" \
  --multiple-of 32 \
  --image-check-mode open \
  --enforce-rescale-images-real-dir \
  --image-check-n 0

# Derive vLLM server data-parallel size from SERVER_GPUS and tensor-parallel size.
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

# Default server parallelism: **data-parallel first** (tp=1, dp=#gpus).
# Rationale: on nodes where a single GPU can fit the full model, DP maximizes
# rollout throughput and keeps per-request latency stable.
if [[ "${SERVER_TP}" -le 0 ]]; then
  echo "[ERROR] rollout_matching.vllm.tensor_parallel_size must be >= 1. Got ${SERVER_TP}" >&2
  exit 2
fi
if (( ${#server_gpu_array[@]} % SERVER_TP != 0 )); then
  echo "[ERROR] server_gpus count must be divisible by rollout_matching.vllm.tensor_parallel_size." >&2
  echo "[ERROR] server_gpus=${SERVER_GPUS} (n=${#server_gpu_array[@]}) tensor_parallel_size=${SERVER_TP}" >&2
  exit 2
fi
SERVER_DP=$(( ${#server_gpu_array[@]} / SERVER_TP ))

# Fail-fast NCCL watchdog defaults (can be overridden by caller env).
# These guard against indefinite DDP hangs when one learner rank is stuck.
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-180}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-67108864}"

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
echo "[INFO] enable_lora: ${VLLM_ENABLE_LORA} (full-sync-only; adapter sync unsupported)"
if [[ -n "${VLLM_ENGINE_KWARGS_JSON:-}" ]]; then
  echo "[INFO] engine_kwargs:${VLLM_ENGINE_KWARGS_JSON}"
fi
echo "[INFO] disable_proxy:${DISABLE_PROXY}"
echo "[INFO] nccl_monitor:${TORCH_NCCL_ENABLE_MONITORING} heartbeat_s:${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC} dump_on_timeout:${TORCH_NCCL_DUMP_ON_TIMEOUT}"
echo "========================================================================"

SERVER_ENV=(CUDA_VISIBLE_DEVICES="${SERVER_GPUS}")
if [[ -n "${NO_PROXY:-}" ]]; then
  SERVER_ENV+=(NO_PROXY="${NO_PROXY}" no_proxy="${NO_PROXY}")
fi
SERVER_ENV+=(ROOT_IMAGE_DIR="${ROOT_IMAGE_DIR_RESOLVED}")

SERVER_CMD=(swift rollout \
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
if [[ -n "${VLLM_ENGINE_KWARGS_JSON:-}" ]]; then
  SERVER_CMD+=(--vllm_engine_kwargs "${VLLM_ENGINE_KWARGS_JSON}")
fi

# Keep server prompt encoding identical to learner template.encode:
# - `--template` pins the exact chat template (e.g. qwen3_vl).
# - `--max_pixels` must match training to avoid vision token count drift.
# - `--max_length` aligns the template max_length with global_max_length.
# - truncation_strategy: ms-swift CLI uses `delete` to mean Template(..., truncation_strategy='raise').
if [[ -n "${SERVER_TEMPLATE:-}" ]]; then
  SERVER_CMD+=(--template "${SERVER_TEMPLATE}")
fi
if [[ -n "${SERVER_MAX_LENGTH:-}" ]]; then
  SERVER_CMD+=(--max_length "${SERVER_MAX_LENGTH}")
fi
if [[ -n "${SERVER_TRUNCATION_STRATEGY:-}" ]]; then
  SERVER_CMD+=(--truncation_strategy "${SERVER_TRUNCATION_STRATEGY}")
fi
SERVER_CMD+=(--max_pixels "${TEMPLATE_MAX_PIXELS}")

SCRIPT_PGID="$(ps -o pgid= "$$" | tr -d ' ')"
SERVER_PID=""
SERVER_PGID=""
TRAIN_PID=""
TRAIN_PGID=""
CLEANUP_DONE=0
SERVER_STARTED=0
TRAIN_STARTED=0

_collect_descendants() {
  local root_pid="$1"
  local -a queue=("${root_pid}")
  local -a descendants=()
  local current child
  while (( ${#queue[@]} > 0 )); do
    current="${queue[0]}"
    queue=("${queue[@]:1}")
    while read -r child; do
      child="${child//[[:space:]]/}"
      [[ -z "${child}" ]] && continue
      descendants+=("${child}")
      queue+=("${child}")
    done < <(ps -o pid= --ppid "${current}" 2>/dev/null || true)
  done
  if (( ${#descendants[@]} > 0 )); then
    printf '%s\n' "${descendants[@]}"
  fi
}

_collect_server_pids() {
  local -a raw_pids=()
  local pid
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    raw_pids+=("${SERVER_PID}")
    while read -r pid; do
      [[ -n "${pid}" ]] && raw_pids+=("${pid}")
    done < <(_collect_descendants "${SERVER_PID}" || true)
  fi
  if [[ -n "${SERVER_PGID:-}" ]]; then
    while read -r pid; do
      pid="${pid//[[:space:]]/}"
      [[ -n "${pid}" ]] && raw_pids+=("${pid}")
    done < <(ps -o pid= -g "${SERVER_PGID}" 2>/dev/null || true)
  fi
  if command -v lsof >/dev/null 2>&1; then
    while read -r pid; do
      pid="${pid//[[:space:]]/}"
      [[ -n "${pid}" ]] && raw_pids+=("${pid}")
    done < <(lsof -t -iTCP:"${SERVER_PORT}" -sTCP:LISTEN 2>/dev/null || true)
  fi
  if command -v netstat >/dev/null 2>&1; then
    while read -r pid; do
      pid="${pid//[[:space:]]/}"
      [[ -n "${pid}" ]] && raw_pids+=("${pid}")
    done < <(
      netstat -tulpn 2>/dev/null \
        | awk -v port=":${SERVER_PORT}" '$4 ~ port && $6 == "LISTEN" {split($7,a,"/"); if (a[1] ~ /^[0-9]+$/) print a[1]}' \
        || true
    )
  fi

  local -A seen=()
  local -a unique_pids=()
  for pid in "${raw_pids[@]}"; do
    [[ -z "${pid}" ]] && continue
    [[ ! "${pid}" =~ ^[0-9]+$ ]] && continue
    if [[ -z "${seen[$pid]+x}" ]]; then
      seen[$pid]=1
      unique_pids+=("${pid}")
    fi
  done

  if (( ${#unique_pids[@]} > 0 )); then
    printf '%s\n' "${unique_pids[@]}"
  fi
}

_kill_pid_list() {
  local label="$1"
  shift || true
  local -a pids=("$@")
  local pid
  if (( ${#pids[@]} == 0 )); then
    return 0
  fi
  echo "[INFO] Stopping ${label} pids: ${pids[*]}"
  kill -TERM "${pids[@]}" >/dev/null 2>&1 || true
  sleep 3
  local -a alive=()
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      alive+=("${pid}")
    fi
  done
  if (( ${#alive[@]} > 0 )); then
    echo "[WARN] Force-killing lingering ${label} pids: ${alive[*]}"
    kill -KILL "${alive[@]}" >/dev/null 2>&1 || true
  fi
}

_stop_train_processes() {
  if [[ -n "${TRAIN_PGID:-}" && "${TRAIN_PGID}" != "${SCRIPT_PGID}" ]]; then
    echo "[INFO] Stopping learner process group (pgid ${TRAIN_PGID})"
    kill -TERM "-${TRAIN_PGID}" >/dev/null 2>&1 || true
  elif [[ -n "${TRAIN_PID:-}" ]] && kill -0 "${TRAIN_PID}" >/dev/null 2>&1; then
    echo "[INFO] Stopping learner process (pid ${TRAIN_PID})"
    kill -TERM "${TRAIN_PID}" >/dev/null 2>&1 || true
  fi
}

_stop_server_processes() {
  if [[ -n "${SERVER_PGID:-}" ]]; then
    echo "[INFO] Stopping vLLM server process group (pgid ${SERVER_PGID})"
    kill -TERM "-${SERVER_PGID}" >/dev/null 2>&1 || true
    sleep 2
  fi
  local -a server_pids=()
  mapfile -t server_pids < <(_collect_server_pids || true)
  _kill_pid_list "vLLM server" "${server_pids[@]}"
}

cleanup() {
  if [[ "${CLEANUP_DONE}" -eq 1 ]]; then
    return
  fi
  CLEANUP_DONE=1
  trap - EXIT INT TERM HUP QUIT PIPE

  # Cleanup must be best-effort (never let `set -e` abort teardown).
  set +e

  if [[ "${TRAIN_STARTED}" -eq 1 ]]; then
    _stop_train_processes
  fi
  if [[ "${SERVER_STARTED}" -eq 1 ]]; then
    _stop_server_processes
  fi

  # Reap children if they are still ours.
  if [[ -n "${TRAIN_PID:-}" ]]; then
    wait "${TRAIN_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${SERVER_PID:-}" ]]; then
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi

  # Ensure the rollout server port is actually released before returning, so reruns
  # don't immediately fail the "port already in use" precheck after Ctrl+C.
  if [[ "${SERVER_STARTED}" -eq 1 && -n "${SERVER_PORT:-}" ]]; then
    local host="${SERVER_HOST:-127.0.0.1}"
    if [[ "${host}" == "0.0.0.0" ]]; then
      host="127.0.0.1"
    fi
    local port="${SERVER_PORT}"
    local start_ts now_ts
    start_ts=$(date +%s)
    while true; do
      if python - "${host}" "${port}" <<'PY'
import socket
import sys

host = str(sys.argv[1]).strip() or "127.0.0.1"
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((host, port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
      then
        break
      fi
      now_ts=$(date +%s)
      if (( now_ts - start_ts >= 15 )); then
        echo "[WARN] Rollout server port still busy after cleanup timeout: ${host}:${port}" >&2
        if command -v netstat >/dev/null 2>&1; then
          netstat -tulpn 2>/dev/null | grep -E ":${port}\\b" >&2 || true
        fi
        break
      fi
      sleep 0.5
    done
  fi
}

on_interrupt() {
  local sig="$1"
  echo "[INFO] Received ${sig}; shutting down learner + vLLM server..."
  cleanup
  case "${sig}" in
    INT) exit 130 ;;
    TERM) exit 143 ;;
    HUP) exit 129 ;;
    QUIT) exit 131 ;;
    PIPE) exit 141 ;;
    *) exit 130 ;;
  esac
}

_assert_server_port_free() {
  local host="$1"
  local port="$2"

  if python - "${host}" "${port}" <<'PY'
import socket
import sys

host = str(sys.argv[1]).strip() or "127.0.0.1"
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((host, port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
  then
    return 0
  fi

  echo "[ERROR] Requested vLLM server endpoint is already in use: ${host}:${port}" >&2
  if command -v lsof >/dev/null 2>&1; then
    local -a listener_pids=()
    mapfile -t listener_pids < <(lsof -t -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null | sort -u || true)
    if (( ${#listener_pids[@]} > 0 )); then
      echo "[ERROR] Existing LISTEN pid(s) on port ${port}: ${listener_pids[*]}" >&2
      ps -fp "${listener_pids[@]}" >&2 || true
    fi
  fi
  if command -v netstat >/dev/null 2>&1; then
    echo "[ERROR] netstat listeners on port ${port} (if any):" >&2
    netstat -tulpn 2>/dev/null | grep -E ":${port}\\b" >&2 || true
  fi
  echo "[ERROR] Refusing to continue: rollout server may auto-shift to another port, which desynchronizes trainer/server endpoints." >&2
  exit 2
}

# From this point onward, the launcher owns process lifecycle (server + learner) and must
# terminate everything on error or Ctrl+C to avoid leaking a `swift rollout` server.
trap cleanup EXIT
trap 'on_interrupt INT' INT
trap 'on_interrupt TERM' TERM
trap 'on_interrupt HUP' HUP
trap 'on_interrupt QUIT' QUIT
trap 'on_interrupt PIPE' PIPE

echo "[RUN] ${SERVER_ENV[*]} ${SERVER_CMD[*]}"
HEALTH_HOST="${SERVER_HOST}"
if [[ "${HEALTH_HOST}" == "0.0.0.0" ]]; then
  HEALTH_HOST="127.0.0.1"
fi
_assert_server_port_free "${HEALTH_HOST}" "${SERVER_PORT}"

if command -v setsid >/dev/null 2>&1; then
  setsid env "${SERVER_ENV[@]}" "${SERVER_CMD[@]}" &
  SERVER_PID=$!
  SERVER_STARTED=1
  SERVER_PGID="$(ps -o pgid= "${SERVER_PID}" | tr -d ' ')"
else
  env "${SERVER_ENV[@]}" "${SERVER_CMD[@]}" &
  SERVER_PID=$!
  SERVER_STARTED=1
  SERVER_PGID=""
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

SERVER_WORLD_RAW="$(curl --noproxy "*" -s "${WORLD_URL}" || true)"
SERVER_WORLD_SIZE="$(
  SERVER_WORLD_RAW="${SERVER_WORLD_RAW}" python - <<'PY'
import json
import os
import sys

raw = str(os.environ.get("SERVER_WORLD_RAW", "")).strip()
if not raw:
    raise SystemExit(2)

try:
    payload = json.loads(raw)
except Exception:
    payload = raw

world_size = None
if isinstance(payload, int):
    world_size = int(payload)
elif isinstance(payload, dict):
    if "world_size" in payload:
        world_size = int(payload["world_size"])
elif isinstance(payload, str) and payload.strip().isdigit():
    world_size = int(payload.strip())

if world_size is None:
    raise SystemExit(3)
print(world_size)
PY
)" || {
  echo "[ERROR] Failed to parse rollout server world_size from ${WORLD_URL}. Raw payload: ${SERVER_WORLD_RAW:-<empty>}" >&2
  exit 1
}

EXPECTED_SERVER_WORLD_SIZE=$(( SERVER_DP * SERVER_TP ))
if [[ "${SERVER_WORLD_SIZE}" -ne "${EXPECTED_SERVER_WORLD_SIZE}" ]]; then
  echo "[ERROR] Unexpected rollout server world_size at ${WORLD_URL}: got=${SERVER_WORLD_SIZE} expected=${EXPECTED_SERVER_WORLD_SIZE}" >&2
  echo "[ERROR] This usually indicates a stale server or a port mismatch; aborting before learner launch." >&2
  exit 1
fi

echo "[INFO] vLLM server is ready. world_size: ${SERVER_WORLD_SIZE}"

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
if command -v setsid >/dev/null 2>&1; then
  setsid bash -lc "${TRAIN_CMD}" &
else
  bash -lc "${TRAIN_CMD}" &
fi
TRAIN_PID=$!
TRAIN_STARTED=1
TRAIN_PGID="$(ps -o pgid= "${TRAIN_PID}" | tr -d ' ')"

set +e
wait "${TRAIN_PID}"
TRAIN_RC=$?
set -e

exit "${TRAIN_RC}"
