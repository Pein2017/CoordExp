#!/usr/bin/env bash
set -euo pipefail
cd /data/CoordExp/.worktrees/research-probes
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
R=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-output-ranking-repair
M="$R/runtime/main-v1"
test -f "$R/lead/qualification-acceptance.json"
trap 'rc=$?; echo "RANKING_MAIN_FAILED exit=$rc"; exit "$rc"' ERR
echo "$$" > "$M/producer.pid"
mkdir -p "$M/logs"
wait_all() {
  local bad=0
  for child in "$@"; do wait "$child" || bad=1; done
  return "$bad"
}
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc-per-node=4 --module probes.training_set_completion.source256_ranking_training --manifest "$R/preparation/P-main.json" --output "$M/P/training" > "$M/logs/P-training.log" 2>&1 &
p=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --standalone --nproc-per-node=4 --module probes.training_set_completion.source256_ranking_training --manifest "$R/preparation/R-main.json" --output "$M/R/training" > "$M/logs/R-training.log" 2>&1 &
r=$!
wait_all "$p" "$r"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc-per-node=4 --module probes.training_set_completion.source256_ranking_training --manifest "$R/preparation/P-main.json" --likelihood --checkpoint "$M/P/training/checkpoints/step-00016" --output "$M/P/likelihood.json" > "$M/logs/P-likelihood.log" 2>&1 &
p=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --standalone --nproc-per-node=4 --module probes.training_set_completion.source256_ranking_training --manifest "$R/preparation/R-main.json" --likelihood --checkpoint "$M/R/training/checkpoints/step-00016" --output "$M/R/likelihood.json" > "$M/logs/R-likelihood.log" 2>&1 &
r=$!
wait_all "$p" "$r"
for arm in P R; do
 for split in train dev; do
  jobs=()
  mkdir -p "$M/$arm/readback/$split"
  for shard in 0 1 2 3 4 5 6 7; do
   shard_name=$(printf '%02d' "$shard")
   CUDA_VISIBLE_DEVICES="$shard" python -m probes.training_set_completion.source256_ranking_evaluation worker --manifest "$R/preparation/$arm-main.json" --terminal "$M/$arm/training/terminal.json" --split "$split" --shard "$shard" --output "$M/$arm/readback/$split/shard-$shard_name.json" > "$M/logs/$arm-$split-$shard_name.log" 2>&1 &
   jobs+=("$!")
  done
  wait_all "${jobs[@]}"
 done
done
python -m probes.training_set_completion.source256_ranking_evaluation reduce --p-manifest "$R/preparation/P-main.json" --p-terminal "$M/P/training/terminal.json" --p-readback-root "$M/P/readback" --p-likelihood "$M/P/likelihood.json" --r-manifest "$R/preparation/R-main.json" --r-terminal "$M/R/training/terminal.json" --r-readback-root "$M/R/readback" --r-likelihood "$M/R/likelihood.json" --output "$M/result.json" > "$M/logs/reduce.log" 2>&1
echo RANKING_MAIN_COMPLETED
