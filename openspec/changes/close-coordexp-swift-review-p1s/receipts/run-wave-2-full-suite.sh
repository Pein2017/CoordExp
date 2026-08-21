#!/bin/bash
set -u
cd /data/CoordExp/.worktrees/CoordExp-swift
export PYTHONDONTWRITEBYTECODE=1
unset COORDEXP_SWIFT_PACK_CACHE_ROOT
unset COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE
unset COORDEXP_SWIFT_EVAL_REDUCTION_MODE
unset COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS
unset RANK
unset LOCAL_RANK
unset WORLD_SIZE
unset MASTER_ADDR
unset MASTER_PORT

echo "START=$(date -Iseconds)"
echo "ARGV=conda run -n ms pytest tests/config tests/losses tests/runtime tests/training tests/artifacts tests/eval -q"
SECONDS=0
conda run -n ms pytest tests/config tests/losses tests/runtime tests/training tests/artifacts tests/eval -q
code=$?
echo "WALL_SECONDS=$SECONDS"
echo "END=$(date -Iseconds)"
echo "EXIT=$code"
