#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

DO_IT="${DO_IT:-0}"
PURGE_MODEL_CACHE="${PURGE_MODEL_CACHE:-0}"

paths=(
  "output"
  "tb"
  "vis_out"
  "temp"
  "tmp"
  "result"
  ".pytest_cache"
  ".ruff_cache"
)

if [[ "$PURGE_MODEL_CACHE" == "1" ]]; then
  paths+=("model_cache")
fi

echo "[workspace_gc] repo_root=$ROOT"
echo "[workspace_gc] mode=$([[ "$DO_IT" == "1" ]] && echo delete || echo dry-run)"
echo "[workspace_gc] purge_model_cache=$PURGE_MODEL_CACHE"
echo

for p in "${paths[@]}"; do
  if [[ -e "$p" ]]; then
    echo "[workspace_gc] target: $p"
  fi
done

echo
echo "[workspace_gc] also pruning Python caches under repo (excluding .git and .worktrees)"

if [[ "$DO_IT" != "1" ]]; then
  echo
  echo "Dry-run only."
  echo "To delete: DO_IT=1 bash scripts/tools/workspace_gc.sh"
  echo "To also purge model cache: DO_IT=1 PURGE_MODEL_CACHE=1 bash scripts/tools/workspace_gc.sh"
  exit 0
fi

for p in "${paths[@]}"; do
  if [[ -e "$p" ]]; then
    rm -rf -- "$p"
  fi
done

find . \
  -type d -name "__pycache__" \
  -not -path "./.git/*" -not -path "./.worktrees/*" \
  -prune -exec rm -rf -- {} + \
  >/dev/null

find . \
  -type f \( -name "*.pyc" -o -name "*.pyo" \) \
  -not -path "./.git/*" -not -path "./.worktrees/*" \
  -delete \
  >/dev/null

echo "[workspace_gc] done"
