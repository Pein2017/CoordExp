#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
CONFIG="${CONFIG:-configs/analysis/autoreg_fn_rescue_attention_guided_causal_binding/ckpt3664_val200.yaml}"
SESSION="${SESSION:-autoreg_fn_rescue_phase3_ckpt3664}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
  if [[ "$REPO_ROOT" != "$DEFAULT_REPO_ROOT" ]]; then
    echo "REPO_ROOT override must equal launcher worktree root: REPO_ROOT=$REPO_ROOT DEFAULT_REPO_ROOT=$DEFAULT_REPO_ROOT" >&2
    exit 1
  fi
else
  REPO_ROOT="$DEFAULT_REPO_ROOT"
fi
ROOT="${ROOT:-}"

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
artifact_root_value = paths.get("artifact_root")
if not artifact_root_value:
    raise SystemExit(f"config missing paths.artifact_root: {config_path}")
artifact_root = Path(str(artifact_root_value)).expanduser()
if not artifact_root.is_absolute():
    artifact_root = repo / artifact_root
artifact_root = artifact_root.resolve(strict=False)
root_override = os.environ.get("ROOT_OVERRIDE", "").strip()
if root_override:
    override_root = Path(root_override).expanduser()
    if not override_root.is_absolute():
        override_root = repo / override_root
    override_root = override_root.resolve(strict=False)
    if override_root != artifact_root:
        raise SystemExit(
            f"Phase-3 ROOT mismatch: ROOT={override_root} but CONFIG paths.artifact_root={artifact_root}"
        )
print(artifact_root)
PY
)"

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
COMMAND_FILE="${COMMAND_FILE:-$LOG_DIR/${SESSION}_commands.sh}"
RUNNER="$REPO_ROOT/scripts/analysis/run_autoreg_fn_rescue_attention_guided_causal_binding.py"

if [[ ! -f "$RUNNER" ]]; then
  echo "Phase-3 runner not found: $RUNNER" >&2
  exit 1
fi

existing_outputs=()
for output_path in \
  "$ROOT/target_mask" \
  "$ROOT/competitor_source" \
  "$ROOT/sink_triage" \
  "$ROOT/case_linked" \
  "$ROOT/summary.json" \
  "$ROOT/manifest.json" \
  "$ROOT/report.md" \
  "$COMMAND_FILE" \
  "$LOG_DIR/target_mask.log" \
  "$LOG_DIR/competitor_source.log" \
  "$LOG_DIR/sink_case_report.log"
do
  if [[ -e "$output_path" ]]; then
    existing_outputs+=("$output_path")
  fi
done

if (( ${#existing_outputs[@]} > 0 )) && [[ "$DRY_RUN" == "1" ]]; then
  echo "existing Phase-3 outputs present; dry-run will not remove them:" >&2
  printf '  %s\n' "${existing_outputs[@]}" >&2
elif (( ${#existing_outputs[@]} > 0 )) && [[ "$ALLOW_OVERWRITE" != "1" ]]; then
  echo "Refusing to overwrite existing Phase-3 outputs under $ROOT:" >&2
  printf '  %s\n' "${existing_outputs[@]}" >&2
  echo "Set ALLOW_OVERWRITE=1 to rerun and replace these outputs." >&2
  exit 1
fi

if [[ "$ALLOW_OVERWRITE" == "1" && "$DRY_RUN" != "1" ]]; then
  rm -rf \
    "$ROOT/target_mask" \
    "$ROOT/competitor_source" \
    "$ROOT/sink_triage" \
    "$ROOT/case_linked"
  rm -f \
    "$ROOT/summary.json" \
    "$ROOT/manifest.json" \
    "$ROOT/report.md" \
    "$COMMAND_FILE" \
    "$LOG_DIR/target_mask.log" \
    "$LOG_DIR/competitor_source.log" \
    "$LOG_DIR/sink_case_report.log"
fi

mkdir -p "$LOG_DIR"

target_gpu="${GPUS[0]}"
competitor_gpu="${GPUS[$(( ${#GPUS[@]} > 1 ? 1 : 0 ))]}"

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n\n'
  printf 'cd %q\n' "$REPO_ROOT"
  printf 'mkdir -p %q\n\n' "$LOG_DIR"
  printf '# Phase-3 ScriptMaster: analysis orchestration, not production training.\n'
  printf '# GPU lanes are decode/intervention probes; CPU lanes materialize sink/case/report artifacts.\n\n'
  printf 'CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q python %q \\\n' "$target_gpu" "$REPO_ROOT" "$RUNNER"
  printf '  --config %q \\\n' "$CONFIG"
  printf '  --stages target_mask > %q 2>&1\n\n' "$LOG_DIR/target_mask.log"
  printf 'CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q python %q \\\n' "$competitor_gpu" "$REPO_ROOT" "$RUNNER"
  printf '  --config %q \\\n' "$CONFIG"
  printf '  --stages competitor_source > %q 2>&1\n\n' "$LOG_DIR/competitor_source.log"
  printf 'CUDA_VISIBLE_DEVICES= PYTHONPATH=%q python %q \\\n' "$REPO_ROOT" "$RUNNER"
  printf '  --config %q \\\n' "$CONFIG"
  printf '  --stages sink_triage,case_linked,report > %q 2>&1\n' "$LOG_DIR/sink_case_report.log"
} > "$COMMAND_FILE"
chmod +x "$COMMAND_FILE"

echo "Wrote Phase-3 ScriptMaster command file: $COMMAND_FILE"
echo "Artifact root: $ROOT"
echo "Target-mask GPU: $target_gpu"
echo "Competitor/source GPU: $competitor_gpu"
echo "Available GPU pool: $GPU_LIST"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "--- command file: $COMMAND_FILE ---"
  cat "$COMMAND_FILE"
  echo "DRY_RUN=1; not starting tmux session."
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" "bash '$COMMAND_FILE'"
echo "Started tmux session: $SESSION"
