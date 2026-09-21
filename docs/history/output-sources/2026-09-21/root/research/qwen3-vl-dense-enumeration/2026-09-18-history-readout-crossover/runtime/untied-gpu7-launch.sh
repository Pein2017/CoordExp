#!/usr/bin/env bash
set -u
ROOT=/data/CoordExp/.worktrees/research-probes
RUNTIME_ROOT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-history-readout-crossover/runtime
OUT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-history-readout-crossover/runtime/untied-gpu7
PLAN=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-history-readout-crossover/execution-plan.json
PANEL=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback/panel.json
printf '%s\n' "$$" > "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-history-readout-crossover/runtime/untied-gpu7.pid"
set +e
cd "$ROOT"
CUDA_VISIBLE_DEVICES=7 python3 probes/training_set_completion/history_readout/runtime.py run \
  --panel "$PANEL" \
  --plan "$PLAN" \
  --output "$OUT" \
  --device cuda:0 \
  --max-forwards 9000 \
  --max-seconds 3000 \
  --max-bytes 1932735283 \
  --cell-id untied-885-failure--cut8--history-full--future-full \
  --cell-id untied-5586-failure--cut8--history-original--future-full \
  --cell-id untied-5586-failure--cut8--history-full--future-full \
  --cell-id untied-7511-failure--cut8--history-original--future-full \
  --cell-id untied-7511-failure--cut8--history-full--future-full \
  --cell-id untied-14038-failure--cut8--history-original--future-full \
  --cell-id untied-14038-failure--cut8--history-full--future-full \
  --cell-id untied-632-failure--cut8--history-original--future-full \
  --cell-id untied-632-failure--cut8--history-full--future-full \
  --cell-id untied-417044-failure--cut8--history-original--future-full \
  --cell-id untied-417044-failure--cut8--history-full--future-full
rc=$?
printf '%s\n' "$rc" > "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-history-readout-crossover/runtime/untied-gpu7.exit"
exit "$rc"
