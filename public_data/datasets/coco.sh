#!/usr/bin/env bash
# COCO 2017 dataset plugin for public_data/run.sh.
#
# Contract: this script is executed (not sourced). It MUST NOT rely on runner
# environment variables; the runner passes all paths explicitly as flags.

set -euo pipefail

die() {
  echo "[coco][error] $*" >&2
  exit 1
}

run_py() {
  echo "+ PYTHONPATH=. python $*" >&2
  PYTHONPATH=. python "$@"
}

usage() {
  cat <<'EOF'
COCO plugin (public_data/datasets/coco.sh)

Target:
  COCO 2017 (80-class instances): train2017/val2017 images + instances_{train,val}2017.json

Usage:
  coco.sh default-preset
  coco.sh download --repo-root <abs> --dataset-dir <abs> --raw-dir <abs> [-- <passthrough>]
  coco.sh convert  --repo-root <abs> --raw-image-dir <abs> --raw-train-jsonl <abs> --raw-val-jsonl <abs> [-- <passthrough>]

Notes:
  - Passthrough args after `--` are forwarded to:
    - download: public_data/scripts/download_coco2017.py
    - convert:  public_data/scripts/convert_coco2017_instances.py (for both train + val)
  - Converter outputs JSONL that follows docs/data/JSONL_CONTRACT.md (images/objects/width/height),
    with extra COCO provenance fields (image_id/file_name/category_id/category_name) that downstream
    code should safely ignore.
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

cd "${REPO_ROOT}"

case "${SUBCMD}" in
  download)
    [[ -n "${RAW_DIR}" ]] || die "Missing required flag: --raw-dir"
    mkdir -p "${RAW_DIR}"
    run_py public_data/scripts/download_coco2017.py \
      "${PASSTHROUGH[@]}" \
      --raw_dir "${RAW_DIR}"
    ;;
  convert)
    [[ -n "${RAW_IMAGE_DIR}" ]] || die "Missing required flag: --raw-image-dir"
    [[ -n "${RAW_TRAIN_JSONL}" ]] || die "Missing required flag: --raw-train-jsonl"
    [[ -n "${RAW_VAL_JSONL}" ]] || die "Missing required flag: --raw-val-jsonl"

    run_py public_data/scripts/convert_coco2017_instances.py \
      "${PASSTHROUGH[@]}" \
      --split train \
      --raw_dir "${DATASET_DIR}/raw" \
      --image_dir_name "train2017" \
      --output "${RAW_TRAIN_JSONL}"

    run_py public_data/scripts/convert_coco2017_instances.py \
      "${PASSTHROUGH[@]}" \
      --split val \
      --raw_dir "${DATASET_DIR}/raw" \
      --image_dir_name "val2017" \
      --output "${RAW_VAL_JSONL}"
    ;;
  *)
    die "Unhandled subcommand: ${SUBCMD}"
    ;;
esac
