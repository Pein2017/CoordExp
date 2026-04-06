#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-v4.0.1}"
ROOT_DIR_INPUT="${2:-$PWD/baidupcsgo}"
ROOT_DIR="$(mkdir -p "$ROOT_DIR_INPUT" && cd "$ROOT_DIR_INPUT" && pwd)"
ZIP_NAME="BaiduPCS-Go-${VERSION}-linux-amd64.zip"
URL="https://github.com/qjfoidnh/BaiduPCS-Go/releases/download/${VERSION}/${ZIP_NAME}"

cd "$ROOT_DIR"

curl -L -o "$ZIP_NAME" "$URL"
unzip -o "$ZIP_NAME"

BIN_DIR="$ROOT_DIR/BaiduPCS-Go-${VERSION}-linux-amd64"
BIN_PATH="$BIN_DIR/BaiduPCS-Go"

if [[ ! -x "$BIN_PATH" ]]; then
  echo "BaiduPCS-Go binary not found after install: $BIN_PATH" >&2
  exit 1
fi

echo "$BIN_PATH"
