#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
NUM_SHARDS="${NUM_SHARDS:-8}"
CONFIG="${CONFIG:-configs/analysis/autoreg_fn_rescue_continuation/ckpt3664_val200_linked.yaml}"
SESSION="${SESSION:-autoreg_fn_rescue_ckpt3664}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
ROOT="${ROOT:-}"
SKIP_MERGE_REPORT="${SKIP_MERGE_REPORT:-0}"

ROOT="$(
  CONFIG_PATH="$CONFIG" ROOT_OVERRIDE="$ROOT" REPO_ROOT="$REPO_ROOT" python - <<'PY'
import os
from pathlib import Path

import yaml

repo = Path(os.environ["REPO_ROOT"]).expanduser().resolve()
config_path = Path(os.environ["CONFIG_PATH"]).expanduser()
if not config_path.is_absolute():
    config_path = repo / config_path
config_path = config_path.resolve(strict=False)
payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
paths = payload.get("paths") if isinstance(payload, dict) else None
if not isinstance(paths, dict):
    raise SystemExit(f"config missing paths mapping: {config_path}")

def resolve_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = repo / path
    return path.resolve(strict=False)

artifact_root_value = paths.get("artifact_root")
if not artifact_root_value:
    raise SystemExit(f"config missing paths.artifact_root: {config_path}")
artifact_root = resolve_path(artifact_root_value)
root_override = os.environ.get("ROOT_OVERRIDE", "").strip()
if root_override:
    override_root = resolve_path(root_override)
    if override_root != artifact_root:
        raise SystemExit(
            f"FN-rescue ROOT mismatch: ROOT={override_root} but CONFIG paths.artifact_root={artifact_root}"
        )
root = artifact_root

source_fields = (
    "attention_atlas_root",
    "source_selected_cases",
    "source_candidate_regions",
    "rollout_anatomy_per_row",
    "gt_vs_pred_scored",
    "pred_token_trace",
    "infer_resolved_config",
    "dataset_jsonl",
    "checkpoint",
)
source_paths: list[tuple[str, Path]] = []
for field_name in source_fields:
    value = paths.get(field_name)
    if value is None:
        continue
    source_paths.append((field_name, resolve_path(value)))

for field_name, source_path in source_paths:
    if root == source_path:
        raise SystemExit(
            f"artifact_root must not equal source path {field_name}: {source_path}"
        )
    if root in source_path.parents:
        raise SystemExit(
            f"artifact_root must not be a parent of source path {field_name}: {root} -> {source_path}"
        )
    if source_path in root.parents:
        raise SystemExit(
            f"artifact_root must not be inside source path {field_name}: {root} inside {source_path}"
        )

print(root)
PY
)"

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

LOG_DIR="$ROOT/logs"
SHARDS_DIR="$ROOT/shards"
COMMAND_FILE="${COMMAND_FILE:-$LOG_DIR/${SESSION}_commands.sh}"
RUNNER="$REPO_ROOT/scripts/analysis/run_autoreg_fn_rescue_continuation.py"

if [[ ! -f "$RUNNER" ]]; then
  echo "FN-rescue runner not found: $RUNNER" >&2
  exit 1
fi

existing_outputs=()
for output_path in \
  "$ROOT/selected_rescue_cases.jsonl" \
  "$ROOT/rescue_rows.jsonl" \
  "$ROOT/rescue_generation_rows.jsonl" \
  "$ROOT/rescue_replay_prefix_rows.jsonl" \
  "$ROOT/rescue_attention_region_rows.jsonl" \
  "$ROOT/rescue_decision_context_rows.jsonl" \
  "$ROOT/rescue_candidate_region_rows.jsonl" \
  "$ROOT/wrong_control_rows.jsonl" \
  "$ROOT/summary.json" \
  "$ROOT/merge_summary.json" \
  "$ROOT/report.md" \
  "$ROOT/gallery" \
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
if [[ -d "$LOG_DIR" ]]; then
  for log_path in \
    "$LOG_DIR"/shard_[0-9][0-9][0-9]-of-[0-9][0-9][0-9]_select_cases.log \
    "$LOG_DIR"/shard_[0-9][0-9][0-9]-of-[0-9][0-9][0-9]_feasibility.log \
    "$LOG_DIR"/shard_[0-9][0-9][0-9]-of-[0-9][0-9][0-9]_rescue_decode.log \
    "$LOG_DIR"/shard_[0-9][0-9][0-9]-of-[0-9][0-9][0-9]_attention_replay.log \
    "$LOG_DIR"/merge.log
  do
    if [[ -e "$log_path" ]]; then
      existing_outputs+=("$log_path")
    fi
  done
fi

if (( ${#existing_outputs[@]} > 0 )); then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1; existing FN-rescue outputs found under $ROOT; no files will be removed." >&2
    printf '  %s\n' "${existing_outputs[@]}" >&2
  elif [[ "$ALLOW_OVERWRITE" != "1" ]]; then
    echo "existing FN-rescue outputs found under $ROOT; set ALLOW_OVERWRITE=1 to remove known stale outputs." >&2
    printf '  %s\n' "${existing_outputs[@]}" >&2
    exit 1
  fi
fi

if (( ${#existing_outputs[@]} > 0 )) && [[ "$DRY_RUN" != "1" && "$ALLOW_OVERWRITE" == "1" ]]; then
  rm -f \
    "$ROOT/selected_rescue_cases.jsonl" \
    "$ROOT/rescue_rows.jsonl" \
    "$ROOT/rescue_generation_rows.jsonl" \
    "$ROOT/rescue_replay_prefix_rows.jsonl" \
    "$ROOT/rescue_attention_region_rows.jsonl" \
    "$ROOT/rescue_decision_context_rows.jsonl" \
    "$ROOT/rescue_candidate_region_rows.jsonl" \
    "$ROOT/wrong_control_rows.jsonl" \
    "$ROOT/summary.json" \
    "$ROOT/merge_summary.json" \
    "$ROOT/report.md" \
    "$COMMAND_FILE" \
    "$LOG_DIR/merge.log"
  rm -rf "$ROOT/gallery"
  if [[ -d "$SHARDS_DIR" ]]; then
    for shard_dir in "$SHARDS_DIR"/shard_[0-9][0-9][0-9]-of-[0-9][0-9][0-9]; do
      if [[ -d "$shard_dir" ]]; then
        rm -rf "$shard_dir"
      fi
    done
  fi
  if [[ -d "$LOG_DIR" ]]; then
    rm -f \
      "$LOG_DIR"/shard_[0-9][0-9][0-9]-of-[0-9][0-9][0-9]_select_cases.log \
      "$LOG_DIR"/shard_[0-9][0-9][0-9]-of-[0-9][0-9][0-9]_feasibility.log \
      "$LOG_DIR"/shard_[0-9][0-9][0-9]-of-[0-9][0-9][0-9]_rescue_decode.log \
      "$LOG_DIR"/shard_[0-9][0-9][0-9]-of-[0-9][0-9][0-9]_attention_replay.log
  fi
fi

mkdir -p "$LOG_DIR" "$SHARDS_DIR"

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n\n'
  printf 'cd %q\n' "$REPO_ROOT"
  printf 'mkdir -p %q %q\n\n' "$LOG_DIR" "$SHARDS_DIR"
  printf '# This is analysis sharding, not production training.\n'
  printf '# Select-case materialization stays serial to avoid shard write coordination.\n'
  for ((i = 0; i < NUM_SHARDS; i += 1)); do
    label="$(printf "shard_%03d-of-%03d" "$i" "$NUM_SHARDS")"
    select_log="$LOG_DIR/${label}_select_cases.log"
    printf 'CUDA_VISIBLE_DEVICES= PYTHONPATH=%q python %q \\\n' "$REPO_ROOT" "$RUNNER"
    printf '  --config %q \\\n' "$CONFIG"
    printf '  --stages select_cases \\\n'
    printf '  --shard-index %q \\\n' "$i"
    printf '  --num-shards %q > %q 2>&1\n\n' "$NUM_SHARDS" "$select_log"
  done
  for stage_name in feasibility rescue_decode attention_replay; do
    printf 'pids=()\n'
    printf 'status=0\n'
    for ((i = 0; i < NUM_SHARDS; i += 1)); do
      gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
      label="$(printf "shard_%03d-of-%03d" "$i" "$NUM_SHARDS")"
      stage_log="$LOG_DIR/${label}_${stage_name}.log"
      printf '(\n'
      printf '  CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q python %q \\\n' "$gpu" "$REPO_ROOT" "$RUNNER"
      printf '    --config %q \\\n' "$CONFIG"
      printf '    --stages %q \\\n' "$stage_name"
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
        printf '  echo "one or more FN-rescue %s shards failed; inspect %s" >&2\n' "$stage_name" "$LOG_DIR"
        printf '  exit "$status"\n'
        printf 'fi\n\n'
      fi
    done
  done
  if [[ "$SKIP_MERGE_REPORT" != "1" ]]; then
    printf 'PYTHONPATH=%q python %q \\\n' "$REPO_ROOT" "$RUNNER"
    printf '  --config %q \\\n' "$CONFIG"
    printf '  --stages merge,report,gallery \\\n'
    printf '  --merge-shards \\\n'
    printf '  --num-shards %q > %q 2>&1\n' "$NUM_SHARDS" "$LOG_DIR/merge.log"
  fi
} > "$COMMAND_FILE"
chmod +x "$COMMAND_FILE"

echo "FN-rescue tmux command file: $COMMAND_FILE"
echo "NUM_SHARDS=$NUM_SHARDS GPU_LIST=$GPU_LIST"

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
echo "  tail -f $LOG_DIR/shard_000-of-$(printf "%03d" "$NUM_SHARDS")_rescue_decode.log"
