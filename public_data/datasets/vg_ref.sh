#!/usr/bin/env bash
# Visual Genome (VG) region descriptions ("reference"/phrase regions) plugin for public_data/run.sh.
#
# Produces a VG-flavored dataset where:
#   - desc = region phrase (free-form text)
#   - bbox_2d = region box (x,y,width,height -> x1,y1,x2,y2)
#
# Contract: this script is executed (not sourced). It MUST NOT rely on runner
# environment variables; the runner passes all paths explicitly as flags.

set -euo pipefail

die() {
  echo "[vg_ref][error] $*" >&2
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
VG region descriptions plugin (public_data/datasets/vg_ref.sh)

Usage:
  vg_ref.sh default-preset
  vg_ref.sh download --repo-root <abs> --dataset-dir <abs> --raw-dir <abs> --conda-env <name> [-- <passthrough>]
  vg_ref.sh convert  --repo-root <abs> --dataset-dir <abs> --raw-dir <abs> --conda-env <name> [-- <passthrough>]

Notes:
  - Passthrough args after `--` are forwarded to public_data/scripts/prepare_visual_genome.py.
  - If public_data/vg/raw/images is present, this plugin will automatically:
      - pass --skip-images during download to avoid duplicate downloads, and
      - create symlinks for VG_100K and VG_100K_2 under this dataset's raw/images/.
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

SHARED_VG_IMAGES="${REPO_ROOT}/public_data/vg/raw/images"
USE_SHARED_IMAGES="false"
if [[ -d "${SHARED_VG_IMAGES}/VG_100K" || -d "${SHARED_VG_IMAGES}/VG_100K_2" ]]; then
  USE_SHARED_IMAGES="true"
fi

ensure_shared_image_symlinks() {
  if [[ "${USE_SHARED_IMAGES}" != "true" ]]; then
    return 0
  fi

  mkdir -p "${RAW_DIR}/images"
  if [[ -d "${SHARED_VG_IMAGES}/VG_100K" && ! -e "${RAW_DIR}/images/VG_100K" ]]; then
    ln -s "${SHARED_VG_IMAGES}/VG_100K" "${RAW_DIR}/images/VG_100K"
  fi
  if [[ -d "${SHARED_VG_IMAGES}/VG_100K_2" && ! -e "${RAW_DIR}/images/VG_100K_2" ]]; then
    ln -s "${SHARED_VG_IMAGES}/VG_100K_2" "${RAW_DIR}/images/VG_100K_2"
  fi
}

case "${SUBCMD}" in
  download)
    EXTRA_ARGS=()
    if [[ "${USE_SHARED_IMAGES}" == "true" ]]; then
      EXTRA_ARGS+=(--skip-images)
    fi
    run_py public_data/scripts/prepare_visual_genome.py \
      "${PASSTHROUGH[@]}" \
      --mode regions \
      --output-root "${DATASET_DIR}" \
      --download \
      --download-only \
      "${EXTRA_ARGS[@]}"
    ensure_shared_image_symlinks
    ;;
  convert)
    ensure_shared_image_symlinks
    run_py public_data/scripts/prepare_visual_genome.py \
      "${PASSTHROUGH[@]}" \
      --mode regions \
      --output-root "${DATASET_DIR}"
    ;;
  *)
    die "Unhandled subcommand: ${SUBCMD}"
    ;;
esac

