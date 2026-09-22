#!/usr/bin/env bash
set -u
ROOT=/data/CoordExp/.worktrees/research-probes
OUT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-history-readout-crossover/runtime/qualification
set +e
cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 python3 probes/training_set_completion/history_readout/runtime.py run \
  --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-history-readout-crossover/execution-plan.json \
  --plan /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-history-readout-crossover/execution-plan.json \
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
printf '%s\n' "$rc" > "$OUT/qualification.exit"
exit "$rc"
