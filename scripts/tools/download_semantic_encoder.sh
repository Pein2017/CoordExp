#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-${CONDA_ENV:-ms}}"
export CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_NAME}}"

source "${REPO_DIR}/scripts/_lib/backbone.sh"
cd "${REPO_ROOT}"

MODEL_ID="${MODEL_ID:-sentence-transformers/all-MiniLM-L6-v2}"
REVISION="${REVISION:-}"
CACHE_DIR="${CACHE_DIR:-}"

PYTHON_RUNNER=("${COORDEXP_PYTHON[@]}")
if [[ "${PYTHON_RUNNER[0]}" == "conda" ]]; then
  PYTHON_RUNNER=(conda run --no-capture-output -n "${CONDA_ENV_NAME}" python)
fi

echo "[INFO] Downloading semantic encoder: ${MODEL_ID}"
echo "[INFO] Python runner: ${PYTHON_RUNNER[*]}"
if [[ -n "${REVISION}" ]]; then
  echo "[INFO] Revision: ${REVISION}"
fi
if [[ -n "${CACHE_DIR}" ]]; then
  echo "[INFO] Cache dir override: ${CACHE_DIR}"
fi

MODEL_ID="${MODEL_ID}" \
REVISION="${REVISION}" \
CACHE_DIR="${CACHE_DIR}" \
"${PYTHON_RUNNER[@]}" - <<'PY'
import os

from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer


model_id = os.environ["MODEL_ID"]
revision = os.environ.get("REVISION") or None
cache_dir = os.environ.get("CACHE_DIR") or None

download_path = snapshot_download(
    repo_id=model_id,
    revision=revision,
    cache_dir=cache_dir,
    allow_patterns=[
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "merges.txt",
        "sentencepiece.bpe.model",
        "spiece.model",
        "added_tokens.json",
        "model.safetensors",
    ],
)
print(f"[INFO] snapshot_download path: {download_path}")

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    revision=revision,
    cache_dir=cache_dir,
    local_files_only=True,
)
model = AutoModel.from_pretrained(
    model_id,
    revision=revision,
    cache_dir=cache_dir,
    local_files_only=True,
)

print(f"[INFO] tokenizer loaded: {tokenizer.__class__.__name__}")
print(f"[INFO] model loaded: {model.__class__.__name__}")
print("[INFO] Semantic encoder is available for offline eval.")
PY
