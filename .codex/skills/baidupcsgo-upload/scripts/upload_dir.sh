#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 LOCAL_DIR REMOTE_DIR [BAIDUPCS_BIN]" >&2
  exit 2
fi

LOCAL_DIR="$1"
REMOTE_DIR="$2"
BAIDUPCS_BIN="${3:-${BAIDUPCS_BIN:-BaiduPCS-Go}}"

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "Local directory not found: $LOCAL_DIR" >&2
  exit 1
fi

if [[ "$REMOTE_DIR" != /* ]]; then
  echo "REMOTE_DIR must be an absolute Netdisk path like /output/stage1_2b/foo" >&2
  exit 1
fi

if ! command -v "$BAIDUPCS_BIN" >/dev/null 2>&1 && [[ ! -x "$BAIDUPCS_BIN" ]]; then
  echo "BaiduPCS-Go binary not found: $BAIDUPCS_BIN" >&2
  exit 1
fi

bin="$BAIDUPCS_BIN"

mkdir_chain() {
  local path="$1"
  local cur=""
  local part=""
  IFS='/' read -r -a parts <<< "$path"
  for part in "${parts[@]}"; do
    [[ -z "$part" ]] && continue
    cur="$cur/$part"
    "$bin" mkdir "$cur" >/dev/null 2>&1 || true
  done
}

mkdir_chain "$REMOTE_DIR"
parent_dir="$(dirname "$REMOTE_DIR")"

exec "$bin" upload \
  "$LOCAL_DIR" \
  "$parent_dir" \
  --policy overwrite \
  --norapid \
  -p 1 \
  -l 1 \
  --retry 8
