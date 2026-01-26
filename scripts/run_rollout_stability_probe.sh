#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEGACY="${SCRIPT_DIR}/_legacy/pipelines/run_rollout_stability_probe.sh"

echo "[DEPRECATED] scripts/run_rollout_stability_probe.sh moved to ${LEGACY}" >&2
exec "${LEGACY}" "$@"
