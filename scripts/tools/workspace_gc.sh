#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

DO_IT="${DO_IT:-0}"
PURGE_MODEL_CACHE="${PURGE_MODEL_CACHE:-0}"

paths=(
  "tb"
  "vis_out"
  "temp"
  "tmp"
  "result"
  ".pytest_cache"
  ".ruff_cache"
)

PURGE_OUTPUT="${PURGE_OUTPUT:-0}"

if [[ "$PURGE_OUTPUT" == "1" ]]; then
  echo "[workspace_gc] refusing: output/ contains checkpoints/logs and must not be deleted" >&2
  echo "[workspace_gc] (if you truly need this, delete manually after backing up)" >&2
  exit 2
fi

if [[ "$PURGE_MODEL_CACHE" == "1" ]]; then
  paths+=("model_cache")
fi

echo "[workspace_gc] repo_root=$ROOT"
echo "[workspace_gc] mode=$([[ "$DO_IT" == "1" ]] && echo delete || echo dry-run)"
echo "[workspace_gc] purge_model_cache=$PURGE_MODEL_CACHE"
echo "[workspace_gc] purge_output=$PURGE_OUTPUT (blocked)"
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
