#!/usr/bin/env bash
set +e
ROOT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime
OUT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime/untied-gpu6
LOG=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime/untied-gpu6.log
PID=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime/untied-gpu6.pid
EXIT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime/untied-gpu6.exit
printf '%s\n' "$$" > "$PID"
exec > >(tee -a "$LOG") 2>&1
printf 'launch_pid=%s\n' "$$"
printf 'gpu=6 model=untied shard=untied-gpu6\n'
printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ -e "$OUT" ]; then
  printf 'output_exists=%s\n' "$OUT"
  rc=2
else
  cd /data/CoordExp/.worktrees/research-probes
  env CUDA_VISIBLE_DEVICES=6 python probes/training_set_completion/readout_component/runtime.py run --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback/panel.json --plan /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/execution-plan.json \
  --cell-id untied-14038-failure--shared \
  --cell-id untied-14038-healthy--shared \
  --cell-id untied-309264-healthy--shared \
  --cell-id untied-417044-failure--shared \
  --cell-id untied-417044-healthy--shared \
  --cell-id untied-5586-failure--shared \
  --cell-id untied-5586-healthy--shared \
  --cell-id untied-632-failure--shared \
  --cell-id untied-632-healthy--shared \
  --cell-id untied-7511-failure--shared \
  --cell-id untied-7511-healthy--shared \
  --cell-id untied-885-healthy--shared \
  --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime/untied-gpu6 \
  --device cuda:0 --max-forwards 9000 --max-seconds 3000 --max-bytes 1932735283
  rc=$?
fi
printf 'finished_utc=%s\nexit=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" | tee "$EXIT"
exit "$rc"
