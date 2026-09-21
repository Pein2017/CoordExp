#!/usr/bin/env bash
set +e
ROOT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime
OUT="$ROOT/qualification"
LOG="$ROOT/qualification.log"
PID="$ROOT/qualification.pid"
EXIT="$ROOT/qualification.exit"
printf '%s\n' "$$" > "$PID"
exec > >(tee -a "$LOG") 2>&1
printf 'launch_pid=%s\n' "$$"
printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ -e "$OUT" ]; then
  printf 'output_exists=%s\n' "$OUT"
  rc=2
else
  cd /data/CoordExp/.worktrees/research-probes
env CUDA_VISIBLE_DEVICES=0 python probes/training_set_completion/readout_component/runtime.py run \
  --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback/panel.json \
  --plan /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/execution-plan.json \
  --cell-id untied-885-failure--original \
  --cell-id untied-885-failure--full \
  --cell-id untied-885-failure--shared \
  --cell-id untied-885-failure--centered \
  --output "$OUT" \
  --device cuda:0 \
  --max-forwards 100000 \
  --max-seconds 28800 \
  --max-bytes 17179869184
  rc=$?
fi
printf 'finished_utc=%s\nexit=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" | tee "$EXIT"
exit "$rc"
