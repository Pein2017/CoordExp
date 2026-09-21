#!/usr/bin/env bash
set +e
ROOT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime
OUT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime/untied-gpu7
LOG=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime/untied-gpu7.log
PID=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime/untied-gpu7.pid
EXIT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime/untied-gpu7.exit
printf '%s\n' "$$" > "$PID"
exec > >(tee -a "$LOG") 2>&1
printf 'launch_pid=%s\n' "$$"
printf 'gpu=7 model=untied shard=untied-gpu7\n'
printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ -e "$OUT" ]; then
  printf 'output_exists=%s\n' "$OUT"
  rc=2
else
  cd /data/CoordExp/.worktrees/research-probes
  env CUDA_VISIBLE_DEVICES=7 python probes/training_set_completion/readout_component/runtime.py run --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback/panel.json --plan /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/execution-plan.json \
  --cell-id untied-14038-failure--centered \
  --cell-id untied-14038-healthy--centered \
  --cell-id untied-309264-healthy--centered \
  --cell-id untied-417044-failure--centered \
  --cell-id untied-417044-healthy--centered \
  --cell-id untied-5586-failure--centered \
  --cell-id untied-5586-healthy--centered \
  --cell-id untied-632-failure--centered \
  --cell-id untied-632-healthy--centered \
  --cell-id untied-7511-failure--centered \
  --cell-id untied-7511-healthy--centered \
  --cell-id untied-885-healthy--centered \
  --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime/untied-gpu7 \
  --device cuda:0 --max-forwards 9000 --max-seconds 3000 --max-bytes 1932735283
  rc=$?
fi
printf 'finished_utc=%s\nexit=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" | tee "$EXIT"
exit "$rc"
