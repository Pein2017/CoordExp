#!/usr/bin/env bash
set -uo pipefail

worktree=/data/CoordExp/.worktrees/research-probes
artifact_dir=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/shared-geometry-implementation-v1
config_path="$artifact_dir/shared-entry-smoke.yaml"
receipt_path="$artifact_dir/exit-receipt.json"
signal_name=shared_geometry_entry_smoke_gpu6_done

cd "$worktree" || exit 125
started_at="$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
set +e
/usr/bin/timeout --signal=TERM --kill-after=30s 600s \
  env CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 \
  python -m src.train --config "$config_path"
exit_code=$?
set -e
finished_at="$(date --utc +%Y-%m-%dT%H:%M:%SZ)"

python - "$exit_code" "$started_at" "$finished_at" "$receipt_path" <<'PY'
import json
import sys
from pathlib import Path

exit_code, started_at, finished_at, output_path = sys.argv[1:]
payload = {
    "schema_version": 1,
    "status": "passed" if int(exit_code) == 0 else "failed",
    "exit_code": int(exit_code),
    "started_at": started_at,
    "finished_at": finished_at,
    "gpu": 6,
    "tmux_session": "shared-geometry-entry-smoke-gpu6",
    "timeout_seconds": 600,
    "worktree": "/data/CoordExp/.worktrees/research-probes",
    "entrypoint": "python -m src.train",
    "config": "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/shared-geometry-implementation-v1/shared-entry-smoke.yaml",
    "log": "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/shared-geometry-implementation-v1/shared-entry-smoke.log",
}
Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

tmux wait-for -S "$signal_name"
exit "$exit_code"
