#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: download_dir.sh REMOTE_PATH LOCAL_PARENT_DIR [BAIDUPCS_BIN]

This downloads REMOTE_PATH into LOCAL_PARENT_DIR while preserving the remote folder name.

Environment overrides:
  BAIDUPCS_DOWNLOAD_THREADS         per-file download threads, default: 4
  BAIDUPCS_DOWNLOAD_PARALLEL_FILES  number of files downloaded concurrently, default: 2
  BAIDUPCS_DOWNLOAD_RETRY           retry count, default: 8
  BAIDUPCS_DOWNLOAD_MODE            download mode, default: locate
  BAIDUPCS_DOWNLOAD_NOCHECK         1 disables checksum verification, default: 0
  BAIDUPCS_DOWNLOAD_MTIME           1 preserves server mtime, default: 1
  BAIDUPCS_DOWNLOAD_OVERWRITE       1 passes --ow, default: 1
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

REMOTE_PATH="$1"
LOCAL_PARENT_DIR="$2"
BAIDUPCS_BIN="${3:-${BAIDUPCS_BIN:-BaiduPCS-Go}}"
DOWNLOAD_THREADS="${BAIDUPCS_DOWNLOAD_THREADS:-4}"
DOWNLOAD_PARALLEL_FILES="${BAIDUPCS_DOWNLOAD_PARALLEL_FILES:-2}"
DOWNLOAD_RETRY="${BAIDUPCS_DOWNLOAD_RETRY:-8}"
DOWNLOAD_MODE="${BAIDUPCS_DOWNLOAD_MODE:-locate}"
DOWNLOAD_NOCHECK="${BAIDUPCS_DOWNLOAD_NOCHECK:-0}"
DOWNLOAD_MTIME="${BAIDUPCS_DOWNLOAD_MTIME:-1}"
DOWNLOAD_OVERWRITE="${BAIDUPCS_DOWNLOAD_OVERWRITE:-1}"

if [[ "$REMOTE_PATH" != /* ]]; then
  echo "REMOTE_PATH must be an absolute Netdisk path like /output/stage1/foo" >&2
  exit 1
fi

mkdir -p "$LOCAL_PARENT_DIR"
LOCAL_PARENT_DIR="$(cd "$LOCAL_PARENT_DIR" && pwd)"

if ! command -v "$BAIDUPCS_BIN" >/dev/null 2>&1 && [[ ! -x "$BAIDUPCS_BIN" ]]; then
  echo "BaiduPCS-Go binary not found: $BAIDUPCS_BIN" >&2
  exit 1
fi

bin="$BAIDUPCS_BIN"

"$bin" config set -savedir "$LOCAL_PARENT_DIR" >/dev/null

download_args=(
  "$bin" download
  "$REMOTE_PATH"
  --fullpath
  --mode "$DOWNLOAD_MODE"
  -p "$DOWNLOAD_THREADS"
  -l "$DOWNLOAD_PARALLEL_FILES"
  --retry "$DOWNLOAD_RETRY"
)

if [[ "$DOWNLOAD_OVERWRITE" != "0" ]]; then
  download_args+=(--ow)
fi

if [[ "$DOWNLOAD_NOCHECK" != "0" ]]; then
  download_args+=(--nocheck)
fi

if [[ "$DOWNLOAD_MTIME" != "0" ]]; then
  download_args+=(--mtime)
fi

exec "${download_args[@]}"
