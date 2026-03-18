#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: rtk_auto.sh '<command>'" >&2
  echo "   or: rtk_auto.sh <arg1> <arg2> ..." >&2
  exit 2
fi

if [[ $# -eq 1 ]]; then
  cmd="$1"
else
  printf -v cmd '%q ' "$@"
  cmd="${cmd% }"
fi

if command -v rtk >/dev/null 2>&1; then
  if rewritten="$(rtk rewrite "$cmd" 2>/dev/null)"; then
    exec bash -lc "$rewritten"
  fi
fi

exec bash -lc "$cmd"
