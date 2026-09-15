#!/usr/bin/env bash
set -euo pipefail
cd /data/CoordExp/.worktrees/research-probes
training_root=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput/training
launch_root="$training_root/detached-cold-B-v1"
test ! -e "$training_root/full-B-v2/cold"
test ! -e "$training_root/full-B-v2/cold-check.json"
mkdir "$launch_root"
started=$(date --iso-8601=seconds)
finish() {
    status=$?
    trap - EXIT
    python - "$launch_root/terminal.json" "$started" "$status" <<'PY'
import datetime, json, sys
path, started, status = sys.argv[1:]
with open(path, 'x') as stream:
    json.dump({'schema': 'owner_successor_scale.detached_command.v1',
               'stage': 'B_v2_cold_check', 'started': started,
               'ended': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'exit_status': int(status),
               'status': 'completed' if status == '0' else 'failed'}, stream, indent=2)
    stream.write('\n')
PY
    exit "$status"
}
trap finish EXIT
python -c 'import sys; print(sys.executable)' > "$launch_root/runtime.txt"
timeout --signal=TERM --kill-after=60s 86400s env CUDA_VISIBLE_DEVICES=0 \
    python -m probes.owner_successor_scale.training cold-check \
    --input "$training_root/inputs-sealed-v1.json" \
    --output-root "$training_root/full-B-v2" > "$launch_root/command.log" 2>&1
