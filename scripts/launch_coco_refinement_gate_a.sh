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

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  CONDA_BIN="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [[ -x /root/miniconda3/bin/conda ]]; then
  CONDA_BIN="/root/miniconda3/bin/conda"
else
  printf 'error: conda was not found; the required environment is ms\n' >&2
  exit 1
fi

cd -- "${REPO_ROOT}"
printf 'COCO refinement Gate A\n'
printf 'Browser: %s\n' "${BROWSER_URL}"
printf 'Runtime: %s\n' "${RUNTIME_ROOT}"
printf 'Stop cleanly with Ctrl-C.\n'

exec "${CONDA_BIN}" run --no-capture-output -n ms \
  python -u scripts/run_coco_refinement.py \
  --repo-root "${REPO_ROOT}" \
  --runtime-root "${RUNTIME_ROOT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --browser-origin "http://localhost:${PORT}" \
  --startup-timeout 300 \
  --shutdown-timeout 10
