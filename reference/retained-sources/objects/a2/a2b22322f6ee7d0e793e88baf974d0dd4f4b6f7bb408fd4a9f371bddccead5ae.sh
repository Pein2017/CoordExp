#!/usr/bin/env bash
set -euo pipefail
cd /data/CoordExp/.worktrees/research-probes
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
R=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-output-ranking-repair
Q="$R/runtime/qualification-v1"
trap 'rc=$?; echo "RANKING_QUAL_FAILED exit=$rc"; exit "$rc"' ERR
echo "$$" > "$Q/producer.pid"
python -m torch.distributed.run --standalone --nproc-per-node=4 --module probes.training_set_completion.source256_ranking_training --manifest "$R/preparation/R-qualification.json" --cache --output "$R/preparation/reference-scores.json" > "$Q/cache.log" 2>&1
python -m torch.distributed.run --standalone --nproc-per-node=4 --module probes.training_set_completion.source256_ranking_training --manifest "$R/preparation/R-qualification.json" --output "$Q/training" > "$Q/training.log" 2>&1
CUDA_VISIBLE_DEVICES=0 python -m probes.training_set_completion.source256_ranking_evaluation worker --manifest "$R/preparation/R-qualification.json" --terminal "$Q/training/terminal.json" --qualification --output "$Q/readback.json" > "$Q/readback.log" 2>&1
echo RANKING_QUAL_COMPLETED
