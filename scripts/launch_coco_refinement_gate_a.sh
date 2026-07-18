#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
RUNTIME_ROOT="${REPO_ROOT}/outputs/coco_refinement/gate-a-20260717"
HOST="127.0.0.1"
PORT="53662"
BIND_URL="http://${HOST}:${PORT}/"
BROWSER_URL="http://localhost:${PORT}/"

print_config() {
  printf 'repo_root=%s\n' "${REPO_ROOT}"
  printf 'runtime_root=%s\n' "${RUNTIME_ROOT}"
  printf 'bind_url=%s\n' "${BIND_URL}"
  printf 'browser_url=%s\n' "${BROWSER_URL}"
}

usage() {
  printf 'Usage: bash %s [--print-config|--help]\n' "${BASH_SOURCE[0]}"
  printf 'Starts the COCO refinement Gate A server in the foreground.\n'
  printf 'The direct browser port is fixed at %s; stop with Ctrl-C.\n' "${PORT}"
}

listener_lines() {
  local port="$1"
  ss -H -ltnp "( sport = :${port} )"
}

listener_pids() {
  local port="$1"
  local lines
  lines="$(listener_lines "${port}")"
  if [[ -n "${lines}" ]]; then
    printf '%s\n' "${lines}" \
      | grep -oE 'pid=[0-9]+' \
      | cut -d= -f2 \
      | sort -u || true
  fi
}

any_pid_alive() {
  local pid
  local state
  for pid in "$@"; do
    state="$(ps -o stat= -p "${pid}" 2>/dev/null || true)"
    state="${state//[[:space:]]/}"
    if [[ -n "${state}" && "${state:0:1}" != "Z" ]]; then
      return 0
    fi
  done
  return 1
}

wait_for_port_release() {
  local port="$1"
  shift
  local attempt
  for attempt in {1..100}; do
    if [[ -z "$(listener_lines "${port}")" ]] && ! any_pid_alive "$@"; then
      return 0
    fi
    sleep 0.1
  done
  return 1
}

release_listeners_on_port() {
  local port="$1"
  local lines
  lines="$(listener_lines "${port}")"
  [[ -n "${lines}" ]] || return 0

  local -a pids=()
  mapfile -t pids < <(listener_pids "${port}")
  if (( ${#pids[@]} == 0 )); then
    printf 'error: port %s is occupied, but its listener PID is unavailable\n' "${port}" >&2
    printf '%s\n' "${lines}" >&2
    return 1
  fi

  printf 'Port %s is occupied; replacing the existing listener:\n' "${port}"
  local joined
  joined="$(IFS=,; printf '%s' "${pids[*]}")"
  ps -o pid=,args= -p "${joined}" || true
  kill -TERM "${pids[@]}" 2>/dev/null || true

  if wait_for_port_release "${port}" "${pids[@]}"; then
    printf 'Port %s was released cleanly.\n' "${port}"
    return 0
  fi

  local -a current_pids=()
  mapfile -t current_pids < <(listener_pids "${port}")
  printf 'Port %s is still occupied after 10 seconds; forcing release.\n' "${port}" >&2
  kill -KILL "${pids[@]}" "${current_pids[@]}" 2>/dev/null || true

  if ! wait_for_port_release "${port}"; then
    printf 'error: failed to release port %s\n' "${port}" >&2
    listener_lines "${port}" >&2 || true
    return 1
  fi
  printf 'Port %s was forcibly released.\n' "${port}"
}

main() {
  case "${1:-}" in
    "")
      ;;
    --print-config)
      if (( $# != 1 )); then
        printf 'error: --print-config accepts no additional arguments\n' >&2
        exit 2
      fi
      print_config
      exit 0
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'error: Gate A port is fixed at %s; unsupported argument: %s\n' "${PORT}" "$1" >&2
      usage >&2
      exit 2
      ;;
  esac

  local conda_bin
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    conda_bin="${CONDA_EXE}"
  elif command -v conda >/dev/null 2>&1; then
    conda_bin="$(command -v conda)"
  elif [[ -x /root/miniconda3/bin/conda ]]; then
    conda_bin="/root/miniconda3/bin/conda"
  else
    printf 'error: conda was not found; the required environment is ms\n' >&2
    exit 1
  fi
  if ! command -v ss >/dev/null 2>&1; then
    printf 'error: ss is required to enforce the fixed-port replacement policy\n' >&2
    exit 1
  fi

  release_listeners_on_port "${PORT}"

  cd -- "${REPO_ROOT}"
  printf 'COCO refinement Gate A\n'
  printf 'Browser: %s\n' "${BROWSER_URL}"
  printf 'Runtime: %s\n' "${RUNTIME_ROOT}"
  printf 'Port policy: latest launcher wins and replaces any listener on %s.\n' "${PORT}"
  printf 'Startup validates the full workspace before binding the port (about 1-2 minutes).\n'
  printf 'VS Code can offer Forward/Open after Uvicorn reports that it is running.\n'
  printf 'Stop cleanly with Ctrl-C.\n'

  exec "${conda_bin}" run --no-capture-output -n ms \
    python -u scripts/run_coco_refinement.py \
    --repo-root "${REPO_ROOT}" \
    --runtime-root "${RUNTIME_ROOT}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --browser-origin "http://localhost:${PORT}" \
    --allow-browser-port-remap \
    --startup-timeout 300 \
    --shutdown-timeout 10
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
