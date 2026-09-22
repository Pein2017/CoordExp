#!/usr/bin/env bash
set -u
set -o noclobber
cd /data/CoordExp/.worktrees/research-probes || exit 125
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
run_root=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-round1-greedy-realization
started=$(date -u +%s)
printf 'ROUND1_STARTED unix=%s\n' "$started"
timeout --signal=INT --kill-after=60s 3600s bash -c '
  printf "%s\n" "$$" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-round1-greedy-realization/producer.pid
  exec python -m src.infer --config /data/CoordExp/.worktrees/research-probes/configs/coordexp_swift/infer/source256-rloo-round1-train256-realization-v1.yaml
'
status=$?
ended=$(date -u +%s)
printf '{"schema_version":"round1_inference_terminal.v1","exit_code":%d,"started_unix":%d,"ended_unix":%d,"wall_seconds":%d}\n' "$status" "$started" "$ended" "$((ended-started))" > "$run_root/terminal.json"
if [ "$status" -eq 0 ]; then
  printf 'ROUND1_SUCCESS exit_code=0\n'
else
  printf 'ROUND1_FAILURE exit_code=%s\n' "$status"
fi
exit "$status"
