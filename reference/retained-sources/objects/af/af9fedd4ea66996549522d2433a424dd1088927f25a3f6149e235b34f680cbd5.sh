#!/usr/bin/env bash
set -u
ROOT=/data/CoordExp/.worktrees/research-probes
RUNTIME_ROOT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-history-readout-crossover/runtime
OUT=$RUNTIME_ROOT/qualification
PLAN=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-history-readout-crossover/execution-plan.json
PANEL=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback/panel.json
printf '%s\n' "$$" > "$RUNTIME_ROOT/qualification.pid"
set +e
cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 python3 probes/training_set_completion/history_readout/runtime.py run \
  --panel "$PANEL" \
  --plan "$PLAN" \
  --output "$OUT" \
  --device cuda:0 \
  --max-forwards 9000 \
  --max-seconds 3000 \
  --max-bytes 1932735283 \
  --cell-id untied-885-failure--cut4--history-original--future-original \
  --cell-id untied-885-failure--cut4--history-original--future-full \
  --cell-id untied-885-failure--cut4--history-full--future-original \
  --cell-id untied-885-failure--cut4--history-full--future-full
rc=$?
printf '%s\n' "$rc" > "$RUNTIME_ROOT/qualification.exit"
exit "$rc"
