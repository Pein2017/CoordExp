#!/usr/bin/env bash
# Visual Genome (VG) dataset plugin for public_data/run.sh.
#
# Contract: this script is executed (not sourced). It MUST NOT rely on runner
# environment variables; the runner passes all paths explicitly as flags.

set -euo pipefail

die() {
  echo "[vg][error] $*" >&2
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
VG plugin (public_data/datasets/vg.sh)

Usage:
  vg.sh default-preset
  vg.sh download --repo-root <abs> --dataset-dir <abs> --raw-dir <abs> --conda-env <name> [-- <passthrough>]
  vg.sh convert  --repo-root <abs> --dataset-dir <abs> --raw-dir <abs> --conda-env <name> [-- <passthrough>]

Notes:
  - Passthrough args after `--` are forwarded to public_data/scripts/prepare_visual_genome.py.
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
    --conda-env)
      shift
      CONDA_ENV="${1:-}"
      shift
      ;;
    # Accept and ignore additional runner-provided flags to keep the interface forward-compatible.
    --dataset|--raw-image-dir|--raw-train-jsonl|--raw-val-jsonl)
      shift
      shift
      ;;
    *)
      die "Unknown flag: $1 (use -- to pass through args)"
      ;;
  esac
done

[[ -n "${REPO_ROOT}" ]] || die "Missing required flag: --repo-root"
[[ -n "${DATASET_DIR}" ]] || die "Missing required flag: --dataset-dir"
[[ -n "${RAW_DIR}" ]] || die "Missing required flag: --raw-dir"
[[ -n "${CONDA_ENV}" ]] || die "Missing required flag: --conda-env"

cd "${REPO_ROOT}"
mkdir -p "${RAW_DIR}"

case "${SUBCMD}" in
  download)
    run_py public_data/scripts/prepare_visual_genome.py \
      "${PASSTHROUGH[@]}" \
      --output-root "${DATASET_DIR}" \
      --download \
      --download-only
    ;;
  convert)
    run_py public_data/scripts/prepare_visual_genome.py \
      "${PASSTHROUGH[@]}" \
      --output-root "${DATASET_DIR}"
    ;;
  *)
    die "Unhandled subcommand: ${SUBCMD}"
    ;;
esac

