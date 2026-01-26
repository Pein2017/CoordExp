#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEGACY="${SCRIPT_DIR}/_legacy/tools/merge_coord.sh"

echo "[DEPRECATED] scripts/merge_coord.sh moved to ${LEGACY}" >&2
exec "${LEGACY}" "$@"
