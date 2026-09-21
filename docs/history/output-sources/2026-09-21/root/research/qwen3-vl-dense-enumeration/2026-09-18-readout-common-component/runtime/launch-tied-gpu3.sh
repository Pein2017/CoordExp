#!/usr/bin/env bash
set +e
ROOT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime
OUT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime/tied-gpu3
LOG=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime/tied-gpu3.log
PID=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime/tied-gpu3.pid
EXIT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime/tied-gpu3.exit
printf '%s\n' "$$" > "$PID"
exec > >(tee -a "$LOG") 2>&1
printf 'launch_pid=%s\n' "$$"
printf 'gpu=3 model=tied shard=tied-gpu3\n'
printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ -e "$OUT" ]; then
  printf 'output_exists=%s\n' "$OUT"
  rc=2
else
  cd /data/CoordExp/.worktrees/research-probes
  env CUDA_VISIBLE_DEVICES=3 python probes/training_set_completion/readout_component/runtime.py run --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback/panel.json --plan /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/execution-plan.json \
  --cell-id tied-14038-failure--centered \
  --cell-id tied-14038-healthy--centered \
  --cell-id tied-309264-healthy--centered \
  --cell-id tied-417044-failure--centered \
  --cell-id tied-417044-healthy--centered \
  --cell-id tied-5586-failure--centered \
  --cell-id tied-5586-healthy--centered \
  --cell-id tied-632-failure--centered \
  --cell-id tied-632-healthy--centered \
  --cell-id tied-7511-healthy--centered \
  --cell-id tied-885-failure--centered \
  --cell-id tied-885-healthy--centered \
  --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime/tied-gpu3 \
  --device cuda:0 --max-forwards 9000 --max-seconds 3000 --max-bytes 1932735283
  rc=$?
fi
printf 'finished_utc=%s\nexit=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" | tee "$EXIT"
exit "$rc"
