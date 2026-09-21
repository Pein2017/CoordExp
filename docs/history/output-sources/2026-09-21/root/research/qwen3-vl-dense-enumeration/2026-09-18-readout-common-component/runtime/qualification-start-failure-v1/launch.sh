#!/usr/bin/env bash
set +e
RUN=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime/qualification
printf '%s\n' "$$" > "$RUN/qualification.pid"
exec > >(tee -a "$RUN/qualification.log") 2>&1
printf 'launch_pid=%s\n' "$$"
printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd /data/CoordExp/.worktrees/research-probes
env CUDA_VISIBLE_DEVICES=0 python probes/training_set_completion/readout_component/runtime.py run \
  --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback/panel.json \
  --plan /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/execution-plan.json \
  --cell-id untied-885-failure--original \
  --cell-id untied-885-failure--full \
  --cell-id untied-885-failure--shared \
  --cell-id untied-885-failure--centered \
  --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/runtime/qualification \
  --device cuda:0 \
  --max-forwards 100000 \
  --max-seconds 28800 \
  --max-bytes 17179869184
rc=$?
printf 'finished_utc=%s\nexit=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" | tee "$RUN/qualification.exit"
exit "$rc"
