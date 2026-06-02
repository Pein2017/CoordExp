#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-autoreg_lane_c_ckpt3664}"
NUM_SHARDS="${NUM_SHARDS:-8}"
CONFIG="${CONFIG:-configs/analysis/hard_ce_coord_logit_locality/ckpt3664_lane_c_val200.yaml}"
ROOT="${ROOT:-/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/x1_basin_attribution}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
REPO_ROOT="${REPO_ROOT:-/data/CoordExp}"

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
ROOT="$(ROOT_PATH="$ROOT" REPO_ROOT="$REPO_ROOT" python - <<'PY'
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
  echo "Lane C ROOT mismatch: ROOT=$ROOT but CONFIG paths.artifact_root=$CONFIG_ROOT" >&2
  echo "Set ROOT to the YAML artifact_root or update CONFIG before launch." >&2
  exit 1
fi

LOG_DIR="$ROOT/logs"
SHARDS_DIR="$ROOT/shards"
COMMAND_FILE="${COMMAND_FILE:-$LOG_DIR/${SESSION}_commands.sh}"

existing_outputs=()
for output_path in \
  "$ROOT/per_slot.jsonl" \
  "$ROOT/per_case.jsonl" \
  "$ROOT/summary.json" \
  "$ROOT/merge_summary.json" \
  "$ROOT/run_summary.json"
do
  if [[ -e "$output_path" ]]; then
    existing_outputs+=("$output_path")
  fi
done

if [[ -d "$SHARDS_DIR" ]]; then
  for shard_dir in "$SHARDS_DIR"/*; do
    if [[ -d "$shard_dir" ]]; then
      existing_outputs+=("$shard_dir")
    fi
  done
fi

if (( ${#existing_outputs[@]} > 0 )); then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1; existing Lane C outputs found under $ROOT; no files will be removed." >&2
    printf '  %s\n' "${existing_outputs[@]}" >&2
  elif [[ "$ALLOW_OVERWRITE" != "1" ]]; then
    echo "existing Lane C outputs found under $ROOT; set ALLOW_OVERWRITE=1 to remove stale outputs before launch." >&2
    printf '  %s\n' "${existing_outputs[@]}" >&2
    exit 1
  fi
fi

if (( ${#existing_outputs[@]} > 0 )) && [[ "$DRY_RUN" != "1" ]]; then
  if [[ "$ALLOW_OVERWRITE" != "1" ]]; then
    echo "existing Lane C outputs found under $ROOT; set ALLOW_OVERWRITE=1 to remove stale outputs before launch." >&2
    printf '  %s\n' "${existing_outputs[@]}" >&2
    exit 1
  fi
  rm -f \
    "$ROOT/per_slot.jsonl" \
    "$ROOT/per_case.jsonl" \
    "$ROOT/summary.json" \
    "$ROOT/merge_summary.json" \
    "$ROOT/run_summary.json"
  if [[ -d "$SHARDS_DIR" ]]; then
    for shard_dir in "$SHARDS_DIR"/*; do
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
  printf 'mkdir -p %q %q\n\n' "$LOG_DIR" "$SHARDS_DIR"
  printf 'pids=()\n'
  for ((i = 0; i < NUM_SHARDS; i += 1)); do
    label="$(printf "lane_c_shard_%03d-of-%03d" "$i" "$NUM_SHARDS")"
    shard_dir="$SHARDS_DIR/$label"
    log_file="$LOG_DIR/$label.log"
    printf 'mkdir -p %q\n' "$shard_dir"
    printf '(\n'
    printf '  CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q python %q \\\n' "$i" "$REPO_ROOT" "$REPO_ROOT/scripts/analysis/run_hard_ce_coord_logit_locality.py"
    printf '    --config %q \\\n' "$CONFIG"
    printf '    --stages x1_basin_attribution \\\n'
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
  printf '  echo "one or more Lane C shards failed; inspect %s" >&2\n' "$LOG_DIR"
  printf '  exit "$status"\n'
  printf 'fi\n\n'
  printf 'PYTHONPATH=%q python %q \\\n' "$REPO_ROOT" "$REPO_ROOT/scripts/analysis/run_hard_ce_coord_logit_locality.py"
  printf '  --config %q \\\n' "$CONFIG"
  printf '  --stages x1_basin_attribution \\\n'
  printf '  --merge-shards \\\n'
  printf '  --num-shards %q > %q 2>&1\n' "$NUM_SHARDS" "$LOG_DIR/merge.log"
} > "$COMMAND_FILE"
chmod +x "$COMMAND_FILE"

echo "Lane C tmux command file: $COMMAND_FILE"

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
echo "  tail -f $LOG_DIR/lane_c_shard_000-of-$(printf "%03d" "$NUM_SHARDS").log"
