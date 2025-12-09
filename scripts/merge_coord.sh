#!/bin/bash
# Merge LoRA + coord_offset into a single checkpoint.
# Usage:
#   bash scripts/merge_coord.sh <adapter_dir> <output_dir> [gpu_ids]
# Example:
#   bash scripts/merge_coord.sh \
#     output/debug/coord/v0-20251203-084935/epoch_30-dlora-lrs_4_2_8-sft_base/checkpoint-48 \
#     output/debug/coord_merged_ck48 \
#     0

set -euo pipefail

usage() {
  echo "Usage: $0 <adapter_dir> <output_dir> [gpu_ids]" >&2
  exit 1
}

PYTHON_BIN="/root/miniconda3/envs/ms/bin/python"
SWIFT_BIN="/root/miniconda3/envs/ms/bin/swift"

ADAPTERS="output/12-4/text_only/v0-20251206-025652/epoch_4-dlora-lrs_2_1_4-sorted-text_only/checkpoint-1632"
OUTPUT_DIR="output/12-4/text_only_merged_ck1632"
GPU_DEVICES="${3:-0}"

if [[ -z "$ADAPTERS" || -z "$OUTPUT_DIR" ]]; then
  usage
fi

if [[ ! -f "$ADAPTERS/adapter_config.json" ]]; then
  echo "adapter_config.json not found under $ADAPTERS" >&2
  exit 1
fi

# Extract base model path from adapter_config.json
BASE_MODEL=$("$PYTHON_BIN" - <<PY
import json, sys
cfg = json.load(open("${ADAPTERS}/adapter_config.json"))
print(cfg["base_model_name_or_path"])
PY
)

echo "Base model : $BASE_MODEL"
echo "Adapters   : $ADAPTERS"
echo "GPU        : $GPU_DEVICES"
echo "Output dir : $OUTPUT_DIR"

# Clean existing output to satisfy swift export
if [ -d "$OUTPUT_DIR" ]; then
  echo "Removing existing output dir: $OUTPUT_DIR"
  rm -rf "$OUTPUT_DIR"
fi

# 1) Merge LoRA with swift export
CUDA_VISIBLE_DEVICES=$GPU_DEVICES "$SWIFT_BIN" export \
  --model "$BASE_MODEL" \
  --adapters "$ADAPTERS" \
  --merge_lora true \
  --output_dir "$OUTPUT_DIR" \
  --safe_serialization true \
  --max_shard_size 5GB

# 2) Optionally bake coord_offset into embed_tokens/lm_head if present
HAS_COORD_OFFSETS=$("$PYTHON_BIN" - <<'PY' "$ADAPTERS"
import sys
from pathlib import Path
try:
    from safetensors import safe_open
except Exception:
    print("no")
    sys.exit(0)

adapter_dir = Path(sys.argv[1])
weights = adapter_dir / "adapter_model.safetensors"
if not weights.exists():
    print("no")
    sys.exit(0)

try:
    with safe_open(weights, framework="pt", device="cpu") as f:
        has = any("coord_offset_adapter" in k for k in f.keys())
except Exception:
    has = False

print("yes" if has else "no")
PY
)

if [[ "$HAS_COORD_OFFSETS" == "yes" ]]; then
  echo "Injecting coord_offset offsets into merged weights..."
  "$PYTHON_BIN" scripts/inject_coord_offsets.py \
    --merged_dir "$OUTPUT_DIR" \
    --adapter_dir "$ADAPTERS"
else
  echo "No coord_offset tensors found in adapter; skipping coord-token injection."
fi

echo "Merged model (with coord offsets) saved to: $OUTPUT_DIR"
echo "Test inference:"
echo "CUDA_VISIBLE_DEVICES=${GPU_DEVICES%%,*} ${SWIFT_BIN} infer --model $OUTPUT_DIR --stream true"
