#!/usr/bin/env bash
# Eight fixed, independent native consumers. Run only after the root launch grant.
set -euo pipefail
cd /data/CoordExp/.worktrees/research-probes
run=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/data-flywheel/endpoint-v1
mkdir "$run"
pids=()
for shard in 0 1 2 3 4 5 6 7; do
  (
    set +e
    CUDA_VISIBLE_DEVICES="$shard" python -m probes.parallel_owner_research.data_flywheel endpoint-rank \
      --shard "$shard" --physical-gpu "$shard" --output "$run/shard-$shard" \
      >"$run/shard-$shard.log" 2>&1
    status=$?
    printf '%s\n' "$status" >"$run/shard-$shard.exit"
    exit "$status"
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then :; else failed=1; fi
done
exit "$failed"
