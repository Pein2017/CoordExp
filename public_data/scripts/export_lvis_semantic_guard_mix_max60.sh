#!/usr/bin/env bash
# Run the LVIS "semantic-guard" export pipeline in the background (tmux-friendly).
#
# This produces:
# - cap10/cap20 train+val JSONLs (raw pixels + norm ints + coord tokens), max60 objects/image
# - assistant token-length stats JSONs (GT-only) for cap10/cap20
# - a few visualization grids for typical failure modes (seg vs hull vs bbox)
#
# Usage (from repo root):
#   bash temp/run_lvis_semantic_pipeline.sh
#
# Or in tmux:
#   tmux new -s lvis_semantic 'bash temp/run_lvis_semantic_pipeline.sh'

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

OUT="public_data/lvis/rescale_32_768_mix_hull_semantic_max60"
mkdir -p "${OUT}" "output"

# Keep image dir consistent with existing LVIS rescale outputs.
if [ ! -e "${OUT}/images" ]; then
  ln -s ../rescale_32_768_poly_20/images "${OUT}/images"
fi

LOG="output/tmux_lvis_semantic_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "[run_lvis_semantic_pipeline] start: $(date)"
echo "[run_lvis_semantic_pipeline] out:   ${OUT}"
echo "[run_lvis_semantic_pipeline] log:   ${LOG}"

COMMON_ARGS=( \
  --images-dir public_data/lvis/rescale_32_768_poly_20/images \
  --lambda-iou-per-extra-point 0.01 \
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

build_split() {
  local split="$1"  # train|val
  local cap="$2"    # 10|20
  local lvis_json="public_data/lvis/raw/annotations/lvis_v1_${split}.json"
  local raw_out="${OUT}/${split}.mix_hull_semantic_cap${cap}.raw.jsonl"
  local build_stats="${OUT}/${split}.mix_hull_semantic_cap${cap}.build_stats.json"
  local max60_raw="${OUT}/${split}.mix_hull_semantic_cap${cap}.max60.raw.jsonl"
  local max60_stats="${OUT}/${split}.mix_hull_semantic_cap${cap}.max60.filter_stats.json"
  local norm_out="${OUT}/${split}.mix_hull_semantic_cap${cap}.max60.jsonl"
  local coord_out="${OUT}/${split}.mix_hull_semantic_cap${cap}.max60.coord.jsonl"

  if [ ! -s "${raw_out}" ]; then
    echo ""
    echo "[build] ${split} cap=${cap}"
    PYTHONPATH=. conda run -n ms python public_data/scripts/build_lvis_hull_mix.py \
      --lvis-json "${lvis_json}" \
      --output-jsonl "${raw_out}" \
      --poly-cap "${cap}" \
      --stats-json "${build_stats}" \
      "${COMMON_ARGS[@]}"
  else
    echo "[build] skip existing: ${raw_out}"
  fi

  if [ ! -s "${max60_raw}" ]; then
    echo ""
    echo "[filter] ${split} cap=${cap} max_objects=60"
    PYTHONPATH=. conda run -n ms python public_data/scripts/filter_jsonl_max_objects.py \
      --input "${raw_out}" \
      --output "${max60_raw}" \
      --max-objects 60 \
      --stats-json "${max60_stats}"
  else
    echo "[filter] skip existing: ${max60_raw}"
  fi

  # The convert step can be interrupted (partial last line -> invalid JSON).
  # We do a cheap "last line parses" check before skipping.
  needs_convert=0
  if [ ! -s "${coord_out}" ] || [ ! -s "${norm_out}" ]; then
    needs_convert=1
  else
    if ! python - "${coord_out}" "${norm_out}" <<'PY'
import json
import sys
from pathlib import Path

def _last_json_ok(path: str) -> bool:
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        return False
    # Read the last ~32KB and parse the last non-empty line.
    chunk = 32768
    with p.open("rb") as f:
        f.seek(max(0, p.stat().st_size - chunk))
        tail = f.read().splitlines()
    for raw in reversed(tail):
        if raw.strip():
            json.loads(raw.decode("utf-8"))
            return True
    return False

ok = _last_json_ok(sys.argv[1]) and _last_json_ok(sys.argv[2])
sys.exit(0 if ok else 1)
PY
    then
      needs_convert=1
    fi
  fi

  if [ "${needs_convert}" -eq 1 ]; then
    echo ""
    echo "[convert] ${split} cap=${cap} -> norm + coord"
    PYTHONPATH=. conda run -n ms python public_data/scripts/convert_to_coord_tokens.py \
      --input "${max60_raw}" \
      --output-norm "${norm_out}" \
      --output-tokens "${coord_out}"
  else
    echo "[convert] skip existing (looks valid): ${coord_out}"
  fi
}

# Build missing pieces only (idempotent).
build_split train 10
build_split val 10
build_split train 20
build_split val 20

echo ""
echo "[length] measuring GT assistant token lengths (coord jsonl)"
PYTHONPATH=. conda run -n ms python scripts/measure_gt_max_new_tokens.py \
  --train_jsonl "${OUT}/train.mix_hull_semantic_cap20.max60.coord.jsonl" \
  --val_jsonl "${OUT}/val.mix_hull_semantic_cap20.max60.coord.jsonl" \
  --checkpoint model_cache/Qwen3-VL-8B-Instruct-coordexp \
  --batch_size 128 --topk 10 \
  --save_json "${OUT}/length.mix_hull_semantic_cap20.max60.assistant_tokens.json"

PYTHONPATH=. conda run -n ms python scripts/measure_gt_max_new_tokens.py \
  --train_jsonl "${OUT}/train.mix_hull_semantic_cap10.max60.coord.jsonl" \
  --val_jsonl "${OUT}/val.mix_hull_semantic_cap10.max60.coord.jsonl" \
  --checkpoint model_cache/Qwen3-VL-8B-Instruct-coordexp \
  --batch_size 128 --topk 10 \
  --save_json "${OUT}/length.mix_hull_semantic_cap10.max60.assistant_tokens.json"

echo ""
echo "[vis] raw seg vs lcc vs hull (+bbox) for a few categories"
PYTHONPATH=. conda run -n ms python public_data/scripts/visualize_lvis_seg_vs_lcc_vs_hull.py \
  --lvis-json public_data/lvis/raw/annotations/lvis_v1_val.json \
  --images-dir public_data/lvis/rescale_32_768_poly_20/images \
  --out-dir output/vis_lvis/raw_vs_lcc_vs_hull_val_semantic_guard \
  --cats "plate,tray,table,place_mat" \
  --num 24 --min-parts 2 --min-area 1000 --pick mix

echo ""
echo "[vis] low-fill edge cases (bbox vs visible mask)"
PYTHONPATH=. conda run -n ms python public_data/scripts/find_lvis_semantic_edge_cases.py \
  --lvis-json public_data/lvis/raw/annotations/lvis_v1_val.json \
  --images-dir public_data/lvis/rescale_32_768_poly_20/images \
  --out-dir output/vis_lvis/edge_cases_low_fill_val_semantic_guard \
  --topk 24 --max-per-cat 3 \
  --fill-thresh 0.06 --min-bbox-frac 0.02

echo ""
echo "[run_lvis_semantic_pipeline] done: $(date)"
