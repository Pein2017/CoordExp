#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-autoreg_attention_val200_ckpt3664}"
NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
CONFIG="${CONFIG:-configs/analysis/autoreg_attention_evidence_routing/ckpt3664_feasibility_val200.yaml}"
REPO_ROOT="${REPO_ROOT:-/data/CoordExp}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
RESUME_SELECT_CASES="${RESUME_SELECT_CASES:-0}"
RUN_FEASIBILITY_ONLY="${RUN_FEASIBILITY_ONLY:-0}"
SKIP_SELECT_CASES="${SKIP_SELECT_CASES:-0}"
SKIP_MERGE_REPORT="${SKIP_MERGE_REPORT:-0}"
ROOT="${ROOT:-}"

CONFIG_ROOT="$(
  CONFIG_PATH="$CONFIG" REPO_ROOT="$REPO_ROOT" python - <<'PY'
import os
from pathlib import Path

import yaml

repo = Path(os.environ["REPO_ROOT"]).expanduser().resolve()
config_path = Path(os.environ["CONFIG_PATH"]).expanduser()
if not config_path.is_absolute():
    config_path = repo / config_path
payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
paths = payload.get("paths") if isinstance(payload, dict) else None
if not isinstance(paths, dict) or not paths.get("artifact_root"):
    raise SystemExit(f"config missing paths.artifact_root: {config_path}")
artifact_root = Path(str(paths["artifact_root"])).expanduser()
if not artifact_root.is_absolute():
    artifact_root = repo / artifact_root
print(artifact_root.resolve(strict=False))
PY
)"

if [[ -z "$ROOT" ]]; then
  ROOT="$CONFIG_ROOT"
else
  ROOT="$(
    ROOT_PATH="$ROOT" REPO_ROOT="$REPO_ROOT" python - <<'PY'
import os
from pathlib import Path

repo = Path(os.environ["REPO_ROOT"]).expanduser().resolve()
root = Path(os.environ["ROOT_PATH"]).expanduser()
if not root.is_absolute():
    root = repo / root
print(root.resolve(strict=False))
PY
  )"
  if [[ "$ROOT" != "$CONFIG_ROOT" ]]; then
    echo "Attention ROOT mismatch: ROOT=$ROOT but CONFIG paths.artifact_root=$CONFIG_ROOT" >&2
    exit 1
  fi
fi

if [[ ! "$NUM_SHARDS" =~ ^[0-9]+$ ]] || [[ "$NUM_SHARDS" -le 0 ]]; then
  echo "NUM_SHARDS must be a positive integer, got $NUM_SHARDS" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if (( ${#GPUS[@]} == 0 )); then
  echo "GPU_LIST must contain at least one GPU id" >&2
  exit 1
fi
seen_gpus=" "
for gpu in "${GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "GPU_LIST entries must be integers, got $gpu from $GPU_LIST" >&2
    exit 1
  fi
  if [[ "$seen_gpus" == *" $gpu "* ]]; then
    echo "GPU_LIST contains duplicate GPU id: $gpu" >&2
    exit 1
  fi
  seen_gpus="$seen_gpus$gpu "
done
if (( ${#GPUS[@]} > NUM_SHARDS )); then
  echo "GPU_LIST length (${#GPUS[@]}) must be <= NUM_SHARDS ($NUM_SHARDS)" >&2
  exit 1
fi

LOG_DIR="$ROOT/logs"
SHARDS_DIR="$ROOT/shards"
COMMAND_FILE="${COMMAND_FILE:-$LOG_DIR/${SESSION}_commands.sh}"
RUNNER="$REPO_ROOT/scripts/analysis/run_autoreg_attention_evidence_routing.py"

if [[ ! -f "$RUNNER" ]]; then
  echo "attention runner not found: $RUNNER" >&2
  exit 1
fi

existing_outputs=()
for output_path in \
  "$ROOT/selected_cases.jsonl" \
  "$ROOT/candidate_region_rows.jsonl" \
  "$ROOT/feasibility_rows.jsonl" \
  "$ROOT/attention_region_rows.jsonl" \
  "$ROOT/decision_context_rows.jsonl" \
  "$ROOT/summary.json" \
  "$ROOT/merge_summary.json" \
  "$ROOT/report.md" \
  "$ROOT/shards_manifest.json" \
  "$COMMAND_FILE"
do
  if [[ -e "$output_path" ]]; then
    existing_outputs+=("$output_path")
  fi
done
if [[ -d "$SHARDS_DIR" ]]; then
  for shard_dir in "$SHARDS_DIR"/shard_[0-9][0-9][0-9]-of-[0-9][0-9][0-9]; do
    if [[ -d "$shard_dir" ]]; then
      existing_outputs+=("$shard_dir")
    fi
  done
fi

if (( ${#existing_outputs[@]} > 0 )); then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1; existing attention outputs found under $ROOT; no files will be removed." >&2
    printf '  %s\n' "${existing_outputs[@]}" >&2
  elif [[ "$ALLOW_OVERWRITE" != "1" && "$RESUME_SELECT_CASES" != "1" ]]; then
    echo "existing attention outputs found under $ROOT; set RESUME_SELECT_CASES=1 to reuse select-case shards, or ALLOW_OVERWRITE=1 to remove known stale outputs." >&2
    printf '  %s\n' "${existing_outputs[@]}" >&2
    exit 1
  fi
fi

if (( ${#existing_outputs[@]} > 0 )) && [[ "$DRY_RUN" != "1" && "$ALLOW_OVERWRITE" == "1" ]]; then
  rm -f \
    "$ROOT/selected_cases.jsonl" \
    "$ROOT/candidate_region_rows.jsonl" \
    "$ROOT/feasibility_rows.jsonl" \
    "$ROOT/attention_region_rows.jsonl" \
    "$ROOT/decision_context_rows.jsonl" \
    "$ROOT/summary.json" \
    "$ROOT/merge_summary.json" \
    "$ROOT/report.md" \
    "$ROOT/shards_manifest.json" \
    "$COMMAND_FILE"
  if [[ -d "$SHARDS_DIR" ]]; then
    for shard_dir in "$SHARDS_DIR"/shard_[0-9][0-9][0-9]-of-[0-9][0-9][0-9]; do
      if [[ -d "$shard_dir" ]]; then
        rm -rf "$shard_dir"
      fi
    done
  fi
fi

mkdir -p "$LOG_DIR" "$SHARDS_DIR"

GPU_STAGE="attention_atlas"
if [[ "$RUN_FEASIBILITY_ONLY" == "1" ]]; then
  GPU_STAGE="feasibility"
fi

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n\n'
  printf 'cd %q\n' "$REPO_ROOT"
  printf 'mkdir -p %q %q\n\n' "$LOG_DIR" "$SHARDS_DIR"
  printf '# This is analysis forward-pass sharding, not production training.\n'
  if [[ "$SKIP_SELECT_CASES" != "1" ]]; then
    printf '# Serial selected-case materialization avoids concurrent shard coordination writes.\n'
    for ((i = 0; i < NUM_SHARDS; i += 1)); do
      label="$(printf "shard_%03d-of-%03d" "$i" "$NUM_SHARDS")"
      select_log="$LOG_DIR/${label}_select_cases.log"
      printf 'CUDA_VISIBLE_DEVICES= PYTHONPATH=%q python %q \\\n' "$REPO_ROOT" "$RUNNER"
      printf '  --config %q \\\n' "$CONFIG"
      printf '  --stages select_cases \\\n'
      printf '  --shard-index %q \\\n' "$i"
      printf '  --num-shards %q > %q 2>&1\n\n' "$NUM_SHARDS" "$select_log"
    done
  fi
  printf 'pids=()\n'
  printf 'status=0\n'
  for ((i = 0; i < NUM_SHARDS; i += 1)); do
    gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    label="$(printf "shard_%03d-of-%03d" "$i" "$NUM_SHARDS")"
    stage_log="$LOG_DIR/${label}_${GPU_STAGE}.log"
    printf '(\n'
    printf '  CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q python %q \\\n' "$gpu" "$REPO_ROOT" "$RUNNER"
    printf '    --config %q \\\n' "$CONFIG"
    printf '    --stages %q \\\n' "$GPU_STAGE"
    printf '    --shard-index %q \\\n' "$i"
    printf '    --num-shards %q\n' "$NUM_SHARDS"
    printf ') > %q 2>&1 &\n' "$stage_log"
    printf 'pids+=("$!")\n'
    if (( (i + 1) % ${#GPUS[@]} == 0 || i + 1 == NUM_SHARDS )); then
      printf 'for pid in "${pids[@]}"; do\n'
      printf '  wait "$pid" || status=1\n'
      printf 'done\n'
      printf 'pids=()\n'
      printf 'if [[ "$status" -ne 0 ]]; then\n'
      printf '  echo "one or more attention shards failed; inspect %s" >&2\n' "$LOG_DIR"
      printf '  exit "$status"\n'
      printf 'fi\n\n'
    fi
  done
  if [[ "$SKIP_MERGE_REPORT" != "1" ]]; then
    printf 'PYTHONPATH=%q python %q \\\n' "$REPO_ROOT" "$RUNNER"
    printf '  --config %q \\\n' "$CONFIG"
    printf '  --stages merge,report \\\n'
    printf '  --merge-shards \\\n'
    printf '  --num-shards %q > %q 2>&1\n' "$NUM_SHARDS" "$LOG_DIR/merge.log"
  fi
} > "$COMMAND_FILE"
chmod +x "$COMMAND_FILE"

echo "Attention tmux command file: $COMMAND_FILE"
echo "NUM_SHARDS=$NUM_SHARDS GPU_LIST=$GPU_LIST GPU_STAGE=$GPU_STAGE"

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
echo "  tail -f $LOG_DIR/shard_000-of-$(printf "%03d" "$NUM_SHARDS")_${GPU_STAGE}.log"
