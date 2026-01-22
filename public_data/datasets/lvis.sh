#!/usr/bin/env bash
# LVIS dataset plugin for public_data/run.sh.
#
# Contract: this script is executed (not sourced). It MUST NOT rely on runner
# environment variables; the runner passes all paths explicitly as flags.

set -euo pipefail

die() {
  echo "[lvis][error] $*" >&2
  exit 1
}

_resolve_conda_exe() {
  if command -v conda >/dev/null 2>&1; then
    echo "conda"
    return 0
  fi
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    echo "${CONDA_EXE}"
    return 0
  fi
  die "Missing 'conda' on PATH (and CONDA_EXE is unset); required for python steps."
}

run_py() {
  local conda_exe
  conda_exe="$(_resolve_conda_exe)"
  echo "+ PYTHONPATH=. ${conda_exe} run -n ${CONDA_ENV} python $*" >&2
  PYTHONPATH=. "${conda_exe}" run -n "${CONDA_ENV}" python "$@"
}

usage() {
  cat <<'EOF'
LVIS plugin (public_data/datasets/lvis.sh)

Usage:
  lvis.sh default-preset
  lvis.sh download --repo-root <abs> --dataset-dir <abs> --raw-dir <abs> --conda-env <name> [-- <passthrough>]
  lvis.sh convert  --repo-root <abs> --raw-image-dir <abs> --raw-train-jsonl <abs> --raw-val-jsonl <abs> --conda-env <name> [-- <passthrough>]

Notes:
  - Passthrough args after `--` are forwarded to the underlying python scripts:
    - download: public_data/scripts/download_lvis.py
    - convert:  public_data/scripts/convert_lvis.py (for both train + val)
EOF
}

SUBCMD="${1:-}"
shift $(( $# > 0 ? 1 : 0 )) || true

case "${SUBCMD}" in
  default-preset)
    echo "rescale_32_768_bbox"
    exit 0
    ;;
  download|convert)
    ;;
  ""|help|-h|--help)
    usage
    exit 0
    ;;
  *)
    die "Unknown subcommand: ${SUBCMD}"
    ;;
esac

REPO_ROOT=""
DATASET_DIR=""
RAW_DIR=""
RAW_IMAGE_DIR=""
RAW_TRAIN_JSONL=""
RAW_VAL_JSONL=""
CONDA_ENV="ms"
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --)
      shift
      PASSTHROUGH=("$@")
      break
      ;;
    --repo-root)
      shift
      REPO_ROOT="${1:-}"
      shift
      ;;
    --dataset-dir)
      shift
      DATASET_DIR="${1:-}"
      shift
      ;;
    --raw-dir)
      shift
      RAW_DIR="${1:-}"
      shift
      ;;
    --raw-image-dir)
      shift
      RAW_IMAGE_DIR="${1:-}"
      shift
      ;;
    --raw-train-jsonl)
      shift
      RAW_TRAIN_JSONL="${1:-}"
      shift
      ;;
    --raw-val-jsonl)
      shift
      RAW_VAL_JSONL="${1:-}"
      shift
      ;;
    --conda-env)
      shift
      CONDA_ENV="${1:-}"
      shift
      ;;
    # Accept and ignore additional runner-provided flags to keep the interface forward-compatible.
    --dataset)
      shift
      shift
      ;;
    *)
      die "Unknown flag: $1 (use -- to pass through args)"
      ;;
  esac
done

[[ -n "${REPO_ROOT}" ]] || die "Missing required flag: --repo-root"
[[ -n "${CONDA_ENV}" ]] || die "Missing required flag: --conda-env"

cd "${REPO_ROOT}"

case "${SUBCMD}" in
  download)
    [[ -n "${RAW_DIR}" ]] || die "Missing required flag: --raw-dir"
    mkdir -p "${RAW_DIR}"
    # download_lvis.py expects --output_dir to be the parent directory that contains public_data/lvis/.
    run_py public_data/scripts/download_lvis.py \
      "${PASSTHROUGH[@]}" \
      --output_dir "${REPO_ROOT}/public_data"
    ;;
  convert)
    [[ -n "${RAW_IMAGE_DIR}" ]] || die "Missing required flag: --raw-image-dir"
    [[ -n "${RAW_TRAIN_JSONL}" ]] || die "Missing required flag: --raw-train-jsonl"
    [[ -n "${RAW_VAL_JSONL}" ]] || die "Missing required flag: --raw-val-jsonl"
    # convert_lvis.py needs base_dir pointing at public_data/ for default annotation paths.
    run_py public_data/scripts/convert_lvis.py \
      "${PASSTHROUGH[@]}" \
      --split train \
      --output "${RAW_TRAIN_JSONL}" \
      --base_dir "${REPO_ROOT}/public_data" \
      --image_root "${RAW_IMAGE_DIR}"

    run_py public_data/scripts/convert_lvis.py \
      "${PASSTHROUGH[@]}" \
      --split val \
      --output "${RAW_VAL_JSONL}" \
      --base_dir "${REPO_ROOT}/public_data" \
      --image_root "${RAW_IMAGE_DIR}"
    ;;
  *)
    die "Unhandled subcommand: ${SUBCMD}"
    ;;
esac

