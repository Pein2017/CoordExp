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

ADAPTERS="output/debug/coord/v0-20251204-020002/epoch_30-dlora-lrs_4_2_8-sft_base/checkpoint-400"
OUTPUT_DIR="output/debug/coord_merged_ck400"
GPU_DEVICES="${3:-0}"

if [[ -z "$ADAPTERS" || -z "$OUTPUT_DIR" ]]; then
  usage
fi

if [[ ! -f "$ADAPTERS/adapter_config.json" ]]; then
  echo "adapter_config.json not found under $ADAPTERS" >&2
  exit 1
fi

# Initialize conda if needed
if ! command -v conda &> /dev/null; then
    if [ -f "${CONDA_BASE:-}/etc/profile.d/conda.sh" ]; then
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        source "/root/miniconda3/etc/profile.d/conda.sh"
    fi
fi

# Extract base model path from adapter_config.json
BASE_MODEL=$(python3 - <<PY
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
CUDA_VISIBLE_DEVICES=$GPU_DEVICES conda run -n ms swift export \
  --model "$BASE_MODEL" \
  --adapters "$ADAPTERS" \
  --merge_lora true \
  --output_dir "$OUTPUT_DIR" \
  --safe_serialization true \
  --max_shard_size 5GB

# 2) Bake coord_offset into embed_tokens/lm_head of the merged weights
echo "Injecting coord_offset offsets into merged weights..."
conda run -n ms python scripts/inject_coord_offsets.py \
  --merged_dir "$OUTPUT_DIR" \
  --adapter_dir "$ADAPTERS"

echo "Merged model (with coord offsets) saved to: $OUTPUT_DIR"
echo "Test inference:"
echo "CUDA_VISIBLE_DEVICES=${GPU_DEVICES%%,*} conda run -n ms swift infer --model $OUTPUT_DIR --stream true"
