#!/usr/bin/env bash
set -euo pipefail

# Merge LoRA + coord_offset into a single checkpoint.
#
# Coord-offset note (tie-head):
# - Newer runs default to a tie-head coord_offset adapter (single shared offset table),
#   which is compatible with Qwen-family tied embeddings.
# - Older/ablation runs may have separate embed_offset/head_offset; in that case the
#   injector may materialize lm_head.weight and disable tying in the merged config
#   to preserve distinct embedding vs head updates.
#
# Preferred usage (env vars):
#   ADAPTERS=output/.../checkpoint-XXX \
#   OUTPUT_DIR=output/.../merged_ckXXX \
#   GPU_DEVICES=0 \
#   bash scripts/merge_coord.sh
#
# Positional args are also supported for convenience:
#   bash scripts/merge_coord.sh <adapter_dir> <output_dir> [gpu_devices]
#
# Safety:
#   This script will NOT delete an existing OUTPUT_DIR unless ALLOW_OVERWRITE=1.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib/backbone.sh"

usage() {
  cat >&2 <<'EOF'
Usage:
  ADAPTERS=<adapter_dir> OUTPUT_DIR=<output_dir> [GPU_DEVICES=0] bash scripts/merge_coord.sh
  bash scripts/merge_coord.sh <adapter_dir> <output_dir> [gpu_devices]

Env vars:
  ADAPTERS           Adapter checkpoint directory (must contain adapter_config.json)
  OUTPUT_DIR         Output directory for the merged checkpoint
  GPU_DEVICES        CUDA_VISIBLE_DEVICES for swift (default: 0)
  ALLOW_OVERWRITE    Set to 1 to delete an existing OUTPUT_DIR (default: 0)
  MAX_SHARD_SIZE     Passed to swift export (default: 5GB)
  PYTHON_BIN         Optional override for python
  SWIFT_BIN          Optional override for swift
  CONDA_ENV          Conda env name if using conda run (default: ms)
EOF
  exit 1
}

ADAPTERS="${ADAPTERS:-${1:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${2:-}}"
GPU_DEVICES="${GPU_DEVICES:-${3:-0}}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-5GB}"

ensure_required "ADAPTERS" "$ADAPTERS"
ensure_required "OUTPUT_DIR" "$OUTPUT_DIR"

if [[ ! -f "$ADAPTERS/adapter_config.json" ]]; then
  echo "adapter_config.json not found under $ADAPTERS" >&2
  exit 1
fi

# Prefer conda run for swift, but allow overrides.
COORDEXP_SWIFT=()
if [[ -n "${SWIFT_BIN:-}" ]]; then
  COORDEXP_SWIFT=("${SWIFT_BIN}")
elif command -v conda >/dev/null 2>&1; then
  COORDEXP_SWIFT=(conda run -n "${CONDA_ENV:-ms}" swift)
else
  COORDEXP_SWIFT=(swift)
fi

# Extract base model path from adapter_config.json
BASE_MODEL=$("${COORDEXP_PYTHON[@]}" -c "import json; cfg=json.load(open('${ADAPTERS}/adapter_config.json')); print(cfg.get('base_model_name_or_path',''))")

if [[ -z "${BASE_MODEL}" ]]; then
  echo "ERROR: Failed to read base_model_name_or_path from: ${ADAPTERS}/adapter_config.json" >&2
  echo "Hint: If you're running under conda, this script requires python execution to work under conda run." >&2
  exit 1
fi

echo "Base model : $BASE_MODEL"
echo "Adapters   : $ADAPTERS"
echo "GPU        : $GPU_DEVICES"
echo "Output dir : $OUTPUT_DIR"

# Refuse to delete outputs by default.
if [[ -e "$OUTPUT_DIR" ]]; then
  if [[ "$ALLOW_OVERWRITE" != "1" ]]; then
    echo "ERROR: OUTPUT_DIR already exists: $OUTPUT_DIR" >&2
    echo "Set ALLOW_OVERWRITE=1 to delete it (DANGEROUS) or pick a new OUTPUT_DIR." >&2
    exit 2
  fi
  echo "[WARN] Removing existing OUTPUT_DIR because ALLOW_OVERWRITE=1: $OUTPUT_DIR" >&2
  rm -rf "$OUTPUT_DIR"
fi

# 1) Merge LoRA with swift export
CUDA_VISIBLE_DEVICES=$GPU_DEVICES "${COORDEXP_SWIFT[@]}" export \
  --model "$BASE_MODEL" \
  --adapters "$ADAPTERS" \
  --merge_lora true \
  --output_dir "$OUTPUT_DIR" \
  --safe_serialization true \
  --max_shard_size "$MAX_SHARD_SIZE"

# Keep CoordExp-specific token metadata if available (swift export may not copy it).
if [[ -f "$BASE_MODEL/coord_tokens.json" && ! -f "$OUTPUT_DIR/coord_tokens.json" ]]; then
  cp "$BASE_MODEL/coord_tokens.json" "$OUTPUT_DIR/coord_tokens.json"
fi

# 2) Optionally bake coord_offset into embed_tokens/lm_head if present
HAS_COORD_OFFSETS=$("${COORDEXP_PYTHON[@]}" -c $'import sys\nfrom pathlib import Path\ntry:\n    from safetensors import safe_open\nexcept Exception:\n    print(\"no\")\n    sys.exit(0)\n\nadapter_dir = Path(sys.argv[1])\nweights = adapter_dir / \"adapter_model.safetensors\"\nif not weights.exists():\n    print(\"no\")\n    sys.exit(0)\n\ntry:\n    with safe_open(str(weights), framework=\"pt\", device=\"cpu\") as f:\n        has = any(\"coord_offset_adapter\" in k for k in f.keys())\nexcept Exception:\n    has = False\n\nprint(\"yes\" if has else \"no\")\n' "$ADAPTERS")

if [[ "$HAS_COORD_OFFSETS" == "yes" ]]; then
  echo "Injecting coord_offset offsets into merged weights..."
  "${COORDEXP_PYTHON[@]}" "$REPO_ROOT/scripts/tools/inject_coord_offsets.py" \
    --merged_dir "$OUTPUT_DIR" \
    --adapter_dir "$ADAPTERS"
else
  echo "No coord_offset tensors found in adapter; skipping coord-token injection."
fi

echo "Merged model (with coord offsets) saved to: $OUTPUT_DIR"
echo "Test inference:"
echo "CUDA_VISIBLE_DEVICES=${GPU_DEVICES%%,*} ${COORDEXP_SWIFT[*]} infer --model $OUTPUT_DIR --stream true"
