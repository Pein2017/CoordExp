#!/usr/bin/env bash
set -u

readonly WORKDIR="/data/CoordExp/.worktrees/research-probes"
readonly ROOT="/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-completion-ce-normalization"
readonly CONTROL_REUSE="$ROOT/preparation/control-reuse-v2.json"
readonly PLAN="/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-fixed-prefix-completion/runtime/main-v1/readback-plan.json"
readonly BATCH4_QUALIFICATION="/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-fixed-prefix-completion/runtime/qualification-mechanics-v3/readback-batch4-qualification.json"
readonly MANIFEST="$ROOT/runtime/qualification-normalized-v1/B-normalized/training-manifest.json"
readonly TERMINAL="$ROOT/runtime/qualification-normalized-v1/B-normalized/training/terminal.json"
readonly OUTPUT="$ROOT/runtime/qualification-normalized-v1/B-normalized/readback/qualification-checkpoint-batch4.json"
readonly LOG="$ROOT/runtime/qualification-normalized-v1/B-normalized/readback/qualification-checkpoint-batch4.log"

mkdir -p "$(dirname "$OUTPUT")"
{
  printf '%s\n' 'SOURCE256_NORMALIZED_QUALIFICATION_READBACK_STARTED'
  cd "$WORKDIR"
  CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
    python -m probes.training_set_completion.source256_normalized_readback \
      qualification-checkpoint-worker \
      --control-reuse "$CONTROL_REUSE" \
      --plan "$PLAN" \
      --qualification "$BATCH4_QUALIFICATION" \
      --training-manifest "$MANIFEST" \
      --terminal "$TERMINAL" \
      --output "$OUTPUT" \
      --device cuda:0
  status=$?
  if [ "$status" -eq 0 ]; then
    printf '%s\n' 'SOURCE256_NORMALIZED_QUALIFICATION_READBACK_COMPLETED'
  else
    printf 'SOURCE256_NORMALIZED_QUALIFICATION_READBACK_FAILED exit=%s\n' "$status"
  fi
  exit "$status"
} >>"$LOG" 2>&1
