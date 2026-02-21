#!/usr/bin/env bash
# Export LVIS JSONLs for two-mode training:
# - bbox-only dataset (max60 objects/image)
# - poly-prefer dataset (single poly via convex hull + semantic guard + vertex cap), max60 objects/image
#
# Outputs (raw pixels + norm ints + coord tokens) are written to:
# - public_data/lvis/rescale_32_768_bbox_max60/
# - public_data/lvis/rescale_32_768_poly_prefer_semantic_max60/
#
# Usage (from repo root):
#   bash public_data/scripts/export_lvis_bbox_poly_prefer_semantic_max60.sh
#
# Force rebuild (overwrite outputs even if they already exist):
#   bash public_data/scripts/export_lvis_bbox_poly_prefer_semantic_max60.sh --force
#   # or
#   FORCE_REBUILD=1 bash public_data/scripts/export_lvis_bbox_poly_prefer_semantic_max60.sh
#
# Or in tmux:
#   tmux new -s lvis_reexport 'bash public_data/scripts/export_lvis_bbox_poly_prefer_semantic_max60.sh --force'

set -euo pipefail

FORCE_REBUILD="${FORCE_REBUILD:-0}"
JOBS="${LVIS_EXPORT_JOBS:-${JOBS:-1}}"

# Args (keep defaults safe; no parallelism unless user opts in).
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      echo "Usage: bash public_data/scripts/export_lvis_bbox_poly_prefer_semantic_max60.sh [--force] [--jobs N]"
      echo "  --force: overwrite outputs even if they already exist (or set FORCE_REBUILD=1)"
      echo "  --jobs N: run N dataset splits concurrently (or set LVIS_EXPORT_JOBS=N)"
      exit 0
      ;;
    --force)
      FORCE_REBUILD=1
      shift
      ;;
    --jobs)
      JOBS="${2:-}"
      shift 2
      ;;
    *)
      echo "[error] unexpected arg: $1"
      exit 2
      ;;
  esac
done

if ! [[ "${JOBS}" =~ ^[0-9]+$ ]]; then
  echo "[error] --jobs must be an integer, got: ${JOBS}"
  exit 2
fi
if [ "${JOBS}" -lt 1 ]; then
  JOBS=1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

mkdir -p output
RAW_LVIS_JSON_DIR="${ROOT}/public_data/lvis/raw/annotations"
RAW_IMG_DIR="${ROOT}/public_data/lvis/raw/images"
MODEL_CKPT="model_cache/Qwen3-VL-4B-Instruct-coordexp"

# Semantic guard thresholds (Balanced). Only used by poly-prefer policy.
GUARD_ARGS=( \
  --force-bbox-hull-iou 0.25 \
  --force-bbox-fill 0.06 \
  --force-bbox-min-bbox-frac 0.02 \
  --force-bbox-min-area 500 \
  --force-bbox-max-aspect 4.0 \
  --force-bbox-min-parts 3 \
  --force-bbox-hull-iou-multipart 0.35 \
  --force-bbox-thin-aspect 4.0 \
  --force-bbox-thin-min-hull-iou 0.2 \
)

# Disable the legacy "low-diversity dense image" drop rule; keep only max60 filtering.
DROP_ARGS=( --drop-min-objects 999999 )

LOG="output/tmux_lvis_bbox_poly_prefer_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "[run_lvis_bbox_poly_prefer_pipeline] start: $(date)"
echo "[run_lvis_bbox_poly_prefer_pipeline] log:   ${LOG}"
if [ "${FORCE_REBUILD}" -eq 1 ]; then
  echo "[run_lvis_bbox_poly_prefer_pipeline] FORCE_REBUILD=1 (will overwrite existing outputs)"
fi

if [ ! -f "${RAW_LVIS_JSON_DIR}/lvis_v1_train.json" ] || [ ! -f "${RAW_LVIS_JSON_DIR}/lvis_v1_val.json" ]; then
  echo "[error] Missing LVIS annotation JSONs under: ${RAW_LVIS_JSON_DIR}"
  echo "        Expected: lvis_v1_train.json and lvis_v1_val.json"
  echo ""
  echo "To download+extract LVIS/COCO2017 raw data, run (from repo root):"
  echo "  ./public_data/run.sh lvis download"
  exit 2
fi
if [ ! -d "${RAW_IMG_DIR}/train2017" ] || [ ! -d "${RAW_IMG_DIR}/val2017" ]; then
  echo "[error] Missing COCO2017 image dirs under: ${RAW_IMG_DIR}"
  echo "        Expected: train2017/ and val2017/"
  echo ""
  echo "To download+extract LVIS/COCO2017 raw data, run (from repo root):"
  echo "  ./public_data/run.sh lvis download"
  exit 2
fi

ensure_images_link() {
  local out_dir="$1"
  mkdir -p "${out_dir}"
  # Keep datasets self-contained: JSONLs reference `images/...` within the dataset dir.
  # Point at the downloaded raw COCO2017 tree containing train2017/ and val2017/.
  if [ -L "${out_dir}/images" ]; then
    ln -sfn ../raw/images "${out_dir}/images"
  elif [ ! -e "${out_dir}/images" ]; then
    ln -s ../raw/images "${out_dir}/images"
  else
    echo "[warn] ${out_dir}/images exists and is not a symlink; keeping as-is"
  fi
}

# Lightweight parallel job runner (bash-only; no external deps).
# NOTE: stdout/stderr from concurrent jobs will interleave in the shared log.
_pids=()
_names=()

_running_jobs() {
  jobs -pr | wc -l | tr -d ' '
}

_wait_for_slot() {
  local max_jobs="$1"
  while [ "$(_running_jobs)" -ge "${max_jobs}" ]; do
    sleep 0.2
  done
}

_run_bg() {
  local name="$1"; shift
  _wait_for_slot "${JOBS}"
  echo "[job] start ${name}"
  (
    set -euo pipefail
    "$@"
  ) &
  _pids+=("$!")
  _names+=("${name}")
}

_wait_all() {
  local fail=0
  for i in "${!_pids[@]}"; do
    local pid="${_pids[$i]}"
    local name="${_names[$i]}"
    if wait "${pid}"; then
      echo "[job] done ${name}"
    else
      echo "[job] FAIL ${name} (pid=${pid})"
      fail=1
    fi
  done
  return "${fail}"
}

last_json_ok() {
  local path="$1"
  python - "$path" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists() or p.stat().st_size <= 0:
    sys.exit(1)
chunk = 32768
with p.open("rb") as f:
    f.seek(max(0, p.stat().st_size - chunk))
    tail = f.read().splitlines()
for raw in reversed(tail):
    if raw.strip():
        json.loads(raw.decode("utf-8"))
        sys.exit(0)
sys.exit(1)
PY
}

convert_if_needed() {
  local raw_in="$1"
  local norm_out="$2"
  local coord_out="$3"
  local needs=0
  if [ "${FORCE_REBUILD}" -eq 1 ]; then
    needs=1
  fi
  if [ ! -s "${norm_out}" ] || [ ! -s "${coord_out}" ]; then
    needs=1
  else
    if ! last_json_ok "${norm_out}" || ! last_json_ok "${coord_out}"; then
      needs=1
    fi
  fi
  if [ "${needs}" -eq 1 ]; then
    if [ "${FORCE_REBUILD}" -eq 1 ]; then
      rm -f "${norm_out}" "${coord_out}"
    fi
    echo "[convert] ${raw_in} -> ${norm_out} + ${coord_out}"
    PYTHONPATH=. conda run -n ms python public_data/scripts/convert_to_coord_tokens.py \
      --input "${raw_in}" \
      --output-norm "${norm_out}" \
      --output-tokens "${coord_out}"
  else
    echo "[convert] skip existing (looks valid): ${coord_out}"
  fi
}

build_bbox_only_split() {
  local split="$1"  # train|val
  local out_dir="public_data/lvis/rescale_32_768_bbox_max60"
  ensure_images_link "${out_dir}"

  local lvis_json="${RAW_LVIS_JSON_DIR}/lvis_v1_${split}.json"
  local pixel_out="${out_dir}/${split}.bbox_only.jsonl"
  local build_stats="${out_dir}/${split}.bbox_only.build_stats.json"
  local max60_pixel="${out_dir}/${split}.bbox_only.max60.jsonl"
  local max60_stats="${out_dir}/${split}.bbox_only.max60.filter_stats.json"
  local norm_out="${out_dir}/${split}.bbox_only.max60.norm.jsonl"
  local coord_out="${out_dir}/${split}.bbox_only.max60.coord.jsonl"

  if [ "${FORCE_REBUILD}" -eq 1 ]; then
      rm -f \
      "${pixel_out}" \
      "${build_stats}" \
      "${max60_pixel}" \
      "${max60_stats}" \
      "${norm_out}" \
      "${coord_out}"
  fi

  if [ "${FORCE_REBUILD}" -eq 1 ] || [ ! -s "${pixel_out}" ]; then
    echo ""
    echo "[build:bbox_only] ${split}"
    PYTHONPATH=. conda run -n ms python public_data/scripts/build_lvis_hull_mix.py \
      --lvis-json "${lvis_json}" \
      --images-dir "${out_dir}/images" \
      --output-jsonl "${pixel_out}" \
      --geometry-policy bbox_only \
      --poly-cap 20 \
      "${DROP_ARGS[@]}" \
      --stats-json "${build_stats}"
  else
    echo "[build:bbox_only] skip existing: ${pixel_out}"
  fi

  if [ "${FORCE_REBUILD}" -eq 1 ] || [ ! -s "${max60_pixel}" ]; then
    echo "[filter] ${split} bbox_only max_objects=60"
    PYTHONPATH=. conda run -n ms python public_data/scripts/filter_jsonl_max_objects.py \
      --input "${pixel_out}" \
      --output "${max60_pixel}" \
      --max-objects 60 \
      --stats-json "${max60_stats}"
  else
    echo "[filter] skip existing: ${max60_pixel}"
  fi

  convert_if_needed "${max60_pixel}" "${norm_out}" "${coord_out}"
}

build_poly_prefer_split() {
  local split="$1"  # train|val
  local cap="$2"    # 10|20
  local out_dir="public_data/lvis/rescale_32_768_poly_prefer_semantic_max60"
  ensure_images_link "${out_dir}"

  local lvis_json="${RAW_LVIS_JSON_DIR}/lvis_v1_${split}.json"
  local pixel_out="${out_dir}/${split}.poly_prefer_semantic_cap${cap}.jsonl"
  local build_stats="${out_dir}/${split}.poly_prefer_semantic_cap${cap}.build_stats.json"
  local max60_pixel="${out_dir}/${split}.poly_prefer_semantic_cap${cap}.max60.jsonl"
  local max60_stats="${out_dir}/${split}.poly_prefer_semantic_cap${cap}.max60.filter_stats.json"
  local norm_out="${out_dir}/${split}.poly_prefer_semantic_cap${cap}.max60.norm.jsonl"
  local coord_out="${out_dir}/${split}.poly_prefer_semantic_cap${cap}.max60.coord.jsonl"

  if [ "${FORCE_REBUILD}" -eq 1 ]; then
      rm -f \
      "${pixel_out}" \
      "${build_stats}" \
      "${max60_pixel}" \
      "${max60_stats}" \
      "${norm_out}" \
      "${coord_out}"
  fi

  if [ "${FORCE_REBUILD}" -eq 1 ] || [ ! -s "${pixel_out}" ]; then
    echo ""
    echo "[build:poly_prefer] ${split} cap=${cap}"
    PYTHONPATH=. conda run -n ms python public_data/scripts/build_lvis_hull_mix.py \
      --lvis-json "${lvis_json}" \
      --images-dir "${out_dir}/images" \
      --output-jsonl "${pixel_out}" \
      --geometry-policy poly_prefer_semantic \
      --poly-cap "${cap}" \
      "${DROP_ARGS[@]}" \
      "${GUARD_ARGS[@]}" \
      --stats-json "${build_stats}"
  else
    echo "[build:poly_prefer] skip existing: ${pixel_out}"
  fi

  if [ "${FORCE_REBUILD}" -eq 1 ] || [ ! -s "${max60_pixel}" ]; then
    echo "[filter] ${split} cap=${cap} max_objects=60"
    PYTHONPATH=. conda run -n ms python public_data/scripts/filter_jsonl_max_objects.py \
      --input "${pixel_out}" \
      --output "${max60_pixel}" \
      --max-objects 60 \
      --stats-json "${max60_stats}"
  else
    echo "[filter] skip existing: ${max60_pixel}"
  fi

  convert_if_needed "${max60_pixel}" "${norm_out}" "${coord_out}"
}

echo ""
echo "== Build bbox-only (max60) =="
_run_bg bbox_only_train build_bbox_only_split train
_run_bg bbox_only_val build_bbox_only_split val

echo ""
echo "== Build poly-prefer semantic (max60) cap10/cap20 =="
_run_bg poly_prefer_cap10_train build_poly_prefer_split train 10
_run_bg poly_prefer_cap10_val build_poly_prefer_split val 10
_run_bg poly_prefer_cap20_train build_poly_prefer_split train 20
_run_bg poly_prefer_cap20_val build_poly_prefer_split val 20

echo ""
echo "== Wait for all jobs (JOBS=${JOBS}) =="
if ! _wait_all; then
  echo "[error] one or more export jobs failed"
  exit 1
fi

echo ""
echo "== Materialize resized images (no symlinks) =="
# CoordExp forbids runtime ms-swift resizing; therefore the dataset directory must
# contain real resized images matching JSONL width/height (no `images -> ../raw/images`).
SRC_POOL="public_data/coco/rescale_32_768_bbox_max60"
if [ ! -d "${SRC_POOL}/images" ]; then
  echo "[error] Missing rescaled COCO image pool under: ${SRC_POOL}/images"
  echo "        This export expects a pre-built rescaled COCO pool (768*32*32 max_pixels, factor=32)."
  echo "        Please build it first, then re-run this script."
  exit 2
fi

conda run -n ms python scripts/tools/materialize_rescaled_images_from_jsonl.py \
  --jsonl public_data/lvis/rescale_32_768_bbox_max60/train.bbox_only.max60.coord.jsonl \
  --jsonl public_data/lvis/rescale_32_768_bbox_max60/val.bbox_only.max60.coord.jsonl \
  --src-root "${SRC_POOL}" \
  --dst-root public_data/lvis/rescale_32_768_bbox_max60 \
  --write

conda run -n ms python scripts/tools/materialize_rescaled_images_from_jsonl.py \
  --jsonl public_data/lvis/rescale_32_768_poly_prefer_semantic_max60/train.poly_prefer_semantic_cap10.max60.coord.jsonl \
  --jsonl public_data/lvis/rescale_32_768_poly_prefer_semantic_max60/val.poly_prefer_semantic_cap10.max60.coord.jsonl \
  --jsonl public_data/lvis/rescale_32_768_poly_prefer_semantic_max60/train.poly_prefer_semantic_cap20.max60.coord.jsonl \
  --jsonl public_data/lvis/rescale_32_768_poly_prefer_semantic_max60/val.poly_prefer_semantic_cap20.max60.coord.jsonl \
  --src-root "${SRC_POOL}" \
  --dst-root public_data/lvis/rescale_32_768_poly_prefer_semantic_max60 \
  --write

# Quick spot-check: open+size alignment (full JSONL scan for width/height constraints).
conda run -n ms python public_data/scripts/validate_jsonl.py \
  public_data/lvis/rescale_32_768_bbox_max60/val.bbox_only.max60.coord.jsonl \
  --max-pixels $((768*32*32)) \
  --multiple-of 32 \
  --image-check-mode open \
  --enforce-rescale-images-real-dir \
  --image-check-n 64

conda run -n ms python public_data/scripts/validate_jsonl.py \
  public_data/lvis/rescale_32_768_poly_prefer_semantic_max60/val.poly_prefer_semantic_cap20.max60.coord.jsonl \
  --max-pixels $((768*32*32)) \
  --multiple-of 32 \
  --image-check-mode open \
  --enforce-rescale-images-real-dir \
  --image-check-n 64

echo ""
echo "== Token length sanity (GT assistant, coord jsonl) =="
if [ -d "${MODEL_CKPT}" ]; then
  PYTHONPATH=. conda run -n ms python scripts/analysis/measure_gt_max_new_tokens.py \
    --train_jsonl public_data/lvis/rescale_32_768_bbox_max60/train.bbox_only.max60.coord.jsonl \
    --val_jsonl public_data/lvis/rescale_32_768_bbox_max60/val.bbox_only.max60.coord.jsonl \
    --checkpoint "${MODEL_CKPT}" \
    --batch_size 128 --topk 10 \
    --save_json public_data/lvis/rescale_32_768_bbox_max60/length.bbox_only.max60.assistant_tokens.json

  PYTHONPATH=. conda run -n ms python scripts/analysis/measure_gt_max_new_tokens.py \
    --train_jsonl public_data/lvis/rescale_32_768_poly_prefer_semantic_max60/train.poly_prefer_semantic_cap10.max60.coord.jsonl \
    --val_jsonl public_data/lvis/rescale_32_768_poly_prefer_semantic_max60/val.poly_prefer_semantic_cap10.max60.coord.jsonl \
    --checkpoint "${MODEL_CKPT}" \
    --batch_size 128 --topk 10 \
    --save_json public_data/lvis/rescale_32_768_poly_prefer_semantic_max60/length.poly_prefer_semantic_cap10.max60.assistant_tokens.json

  PYTHONPATH=. conda run -n ms python scripts/analysis/measure_gt_max_new_tokens.py \
    --train_jsonl public_data/lvis/rescale_32_768_poly_prefer_semantic_max60/train.poly_prefer_semantic_cap20.max60.coord.jsonl \
    --val_jsonl public_data/lvis/rescale_32_768_poly_prefer_semantic_max60/val.poly_prefer_semantic_cap20.max60.coord.jsonl \
    --checkpoint "${MODEL_CKPT}" \
    --batch_size 128 --topk 10 \
    --save_json public_data/lvis/rescale_32_768_poly_prefer_semantic_max60/length.poly_prefer_semantic_cap20.max60.assistant_tokens.json
else
  echo "[warn] skip token-length sanity: missing checkpoint dir: ${MODEL_CKPT}"
fi

echo ""
echo "[run_lvis_bbox_poly_prefer_pipeline] done: $(date)"
