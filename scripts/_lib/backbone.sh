#!/usr/bin/env bash
set -euo pipefail

# Shared helpers for scripts/*.sh to avoid copy/paste drift.
# - Resolves repo root
# - Provides ensure_required
# - Provides a python runner that prefers `conda run -n ms python`

BACKBONE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd -- "${BACKBONE_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${SCRIPTS_DIR}/.." && pwd)"

ensure_required() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    echo "ERROR: $name must be set." >&2
    exit 1
  fi
}

# Use an array so callers can safely append args without eval/quoting issues.
COORDEXP_PYTHON=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
  COORDEXP_PYTHON=("${PYTHON_BIN}")
elif command -v conda >/dev/null 2>&1; then
  COORDEXP_PYTHON=(conda run -n "${CONDA_ENV:-ms}" python)
else
  COORDEXP_PYTHON=(python)
fi

