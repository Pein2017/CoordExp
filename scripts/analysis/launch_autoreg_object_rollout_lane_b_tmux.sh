#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-autoreg_lane_b_ckpt3664}"
NUM_SHARDS="${NUM_SHARDS:-8}"
ROOT="${ROOT:-/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
REPO_ROOT="${REPO_ROOT:-/data/CoordExp}"

CHECKPOINT="/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664"
CONFIG="configs/stage1/detection_teacher_forcing/prod/compact_full_support2.yaml"
DECODE_ARTIFACT="/data/CoordExp/outputs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu/gt_vs_pred.jsonl"
TRACE_ARTIFACT="/data/CoordExp/outputs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu/pred_token_trace.jsonl"

LOG_DIR="$ROOT/logs/lane_b"
SHARDS_DIR="$ROOT/prefix_boundary/shards"
MERGED_DIR="$ROOT/prefix_boundary"
COMMAND_FILE="${COMMAND_FILE:-$LOG_DIR/${SESSION}_commands.sh}"

existing_outputs=()
for output_path in \
  "$MERGED_DIR/per_case.jsonl" \
  "$MERGED_DIR/summary.json" \
  "$MERGED_DIR/merge_summary.json"
do
  if [[ -e "$output_path" ]]; then
    existing_outputs+=("$output_path")
  fi
done

if [[ -d "$SHARDS_DIR" ]]; then
  for shard_dir in "$SHARDS_DIR"/shard_*; do
    if [[ -d "$shard_dir" ]]; then
      existing_outputs+=("$shard_dir")
    fi
  done
fi

if (( ${#existing_outputs[@]} > 0 )); then
  if [[ "$ALLOW_OVERWRITE" != "1" ]]; then
    echo "existing Lane B outputs found under $MERGED_DIR; set ALLOW_OVERWRITE=1 to remove stale outputs before launch." >&2
    printf '  %s\n' "${existing_outputs[@]}" >&2
    exit 1
  fi
  rm -f \
    "$MERGED_DIR/per_case.jsonl" \
    "$MERGED_DIR/summary.json" \
    "$MERGED_DIR/merge_summary.json"
  if [[ -d "$SHARDS_DIR" ]]; then
    for shard_dir in "$SHARDS_DIR"/shard_*; do
      if [[ -d "$shard_dir" ]]; then
        rm -rf "$shard_dir"
      fi
    done
  fi
fi

mkdir -p "$LOG_DIR" "$SHARDS_DIR"

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n\n'
  printf 'cd %q\n' "$REPO_ROOT"
  printf 'mkdir -p %q %q %q\n\n' "$LOG_DIR" "$SHARDS_DIR" "$MERGED_DIR"
  printf 'pids=()\n'
  for ((i = 0; i < NUM_SHARDS; i += 1)); do
    label="$(printf "shard_%03d-of-%03d" "$i" "$NUM_SHARDS")"
    shard_dir="$SHARDS_DIR/$label"
    log_file="$LOG_DIR/$label.log"
    printf 'mkdir -p %q\n' "$shard_dir"
    printf '(\n'
    printf '  CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q python -m src.analysis.prefix_rollin_teacher_forced_diagnostic \\\n' "$i" "$REPO_ROOT"
    printf '    --config %q \\\n' "$CONFIG"
    printf '    --checkpoint %q \\\n' "$CHECKPOINT"
    printf '    --output-dir %q \\\n' "$shard_dir"
    printf '    --split val \\\n'
    printf '    --limit 200 \\\n'
    printf '    --prefix-modes %q \\\n' "gt_prefix,generated_prefix"
    printf '    --k-values every \\\n'
    printf '    --decode-artifact %q \\\n' "$DECODE_ARTIFACT"
    printf '    --trace-artifact %q \\\n' "$TRACE_ARTIFACT"
    printf '    --device cuda:0 \\\n'
    printf '    --shard-index %q \\\n' "$i"
    printf '    --num-shards %q\n' "$NUM_SHARDS"
    printf ') > %q 2>&1 &\n' "$log_file"
    printf 'pids+=("$!")\n\n'
  done
  printf 'status=0\n'
  printf 'for pid in "${pids[@]}"; do\n'
  printf '  wait "$pid" || status=1\n'
  printf 'done\n'
  printf 'if [[ "$status" -ne 0 ]]; then\n'
  printf '  echo "one or more Lane B shards failed; inspect %s" >&2\n' "$LOG_DIR"
  printf '  exit "$status"\n'
  printf 'fi\n\n'
  printf 'PYTHONPATH=%q python -m src.analysis.prefix_rollin_teacher_forced_diagnostic \\\n' "$REPO_ROOT"
  printf '  --merge-shards \\\n'
  printf '  --shards-dir %q \\\n' "$SHARDS_DIR"
  printf '  --output-dir %q \\\n' "$MERGED_DIR"
  printf '  --expected-shards %q > %q 2>&1\n' "$NUM_SHARDS" "$LOG_DIR/merge.log"
} > "$COMMAND_FILE"
chmod +x "$COMMAND_FILE"

echo "Lane B tmux command file: $COMMAND_FILE"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1; not starting tmux session."
  cat "$COMMAND_FILE"
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" "bash '$COMMAND_FILE'"

echo "Started tmux session: $SESSION"
echo "Attach with:"
echo "  tmux attach -t $SESSION"
echo "Inspect recent pane output with:"
echo "  tmux capture-pane -pt $SESSION:0 -S -200"
echo "Tail first shard log with:"
echo "  tail -f $LOG_DIR/shard_000-of-$(printf "%03d" "$NUM_SHARDS").log"
