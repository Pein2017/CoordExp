#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: upload_dir.sh LOCAL_DIR REMOTE_DIR [BAIDUPCS_BIN]

Environment overrides:
  BAIDUPCS_UPLOAD_FILE_THREADS     per-file upload threads, default: 1
  BAIDUPCS_UPLOAD_PARALLEL_FILES   number of files uploaded concurrently, default: 1
  BAIDUPCS_UPLOAD_RETRY            retry count, default: 8
  BAIDUPCS_UPLOAD_POLICY           overwrite policy, default: overwrite
  BAIDUPCS_UPLOAD_NO_RAPID         1 keeps --norapid (default), 0 allows rapid upload

Notes:
  - With BAIDUPCS_UPLOAD_NO_RAPID=1, BaiduPCS-Go documents that single-file threading
    is effectively limited to one thread. You can still raise BAIDUPCS_UPLOAD_PARALLEL_FILES
    to upload multiple files at once.
  - Set BAIDUPCS_UPLOAD_NO_RAPID=0 only when you want higher per-file concurrency and can
    accept the rapid-upload/md5 tradeoff described by BaiduPCS-Go.
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

LOCAL_DIR="$1"
REMOTE_DIR="$2"
BAIDUPCS_BIN="${3:-${BAIDUPCS_BIN:-BaiduPCS-Go}}"
UPLOAD_FILE_THREADS="${BAIDUPCS_UPLOAD_FILE_THREADS:-1}"
UPLOAD_PARALLEL_FILES="${BAIDUPCS_UPLOAD_PARALLEL_FILES:-1}"
UPLOAD_RETRY="${BAIDUPCS_UPLOAD_RETRY:-8}"
UPLOAD_POLICY="${BAIDUPCS_UPLOAD_POLICY:-overwrite}"
UPLOAD_NO_RAPID="${BAIDUPCS_UPLOAD_NO_RAPID:-1}"

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

upload_args=(
  "$bin" upload
  "$LOCAL_DIR"
  "$parent_dir"
  --policy "$UPLOAD_POLICY"
  -p "$UPLOAD_FILE_THREADS"
  -l "$UPLOAD_PARALLEL_FILES"
  --retry "$UPLOAD_RETRY"
)

if [[ "$UPLOAD_NO_RAPID" != "0" ]]; then
  upload_args+=(--norapid)
fi

exec "${upload_args[@]}"
