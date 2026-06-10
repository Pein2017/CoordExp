#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-/data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/prefix_state_transition_tomography/ckpt3664_et_vs_purece_phase_a3_4096.yaml}"
SESSION="${SESSION:-prefix_state_transition_a3_ckpt3664}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHONPATH="${PYTHONPATH:-$ROOT}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/$SESSION}"
export PYTHONPATH

config_value() {
  python - "$CONFIG" "$1" <<'PY'
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = config
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(config_value artifact_root)}"

gpu_preflight() {
  if [[ "${SKIP_GPU_PREFLIGHT:-0}" == "1" ]]; then
    echo "[prefix-state] SKIP_GPU_PREFLIGHT=1; not checking existing GPU compute apps"
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[prefix-state] nvidia-smi not found; cannot verify GPU availability" >&2
    return 1
  fi
  local busy=0
  local gpu
  for gpu in $GPU_IDS; do
    local apps
    apps="$(nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "$apps" ]]; then
      busy=1
      echo "[prefix-state] GPU $gpu already has compute apps:" >&2
      echo "$apps" | sed 's/^/[prefix-state]   /' >&2
    fi
  done
  if [[ "$busy" != "0" ]]; then
    echo "[prefix-state] refusing to launch because one or more requested GPUs are busy" >&2
    echo "[prefix-state] set SKIP_GPU_PREFLIGHT=1 only if you intentionally want to share/override GPUs" >&2
    return 1
  fi
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN prefix-state transition tomography, not production training"
  python "$ROOT/scripts/analysis/prefix_state_transition_tomography/run.py" \
    --config "$CONFIG" \
    --stages prefix_state_index,validate \
    --dry-run
  echo "artifact_root=$ARTIFACT_ROOT"
  echo "log_root=$LOG_ROOT"
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

COMMAND_FILE="$(mktemp -p /tmp prefix_state_tmux_XXXXXX.sh)"
cat > "$COMMAND_FILE" <<EOF
set -euo pipefail
cd "$ROOT"
export PYTHONPATH="$PYTHONPATH"
LOG_ROOT="$LOG_ROOT"
mkdir -p "\$LOG_ROOT"
echo "[prefix-state] building prefix-state index and validating gate, not production training"
if [[ "\${REUSE_READY_INDEX:-1}" == "1" && -f "$ARTIFACT_ROOT/prefix_state_index_summary.json" ]]; then
  if python scripts/analysis/prefix_state_transition_tomography/run.py --config "$CONFIG" --stages validate --allow-overwrite; then
    echo "[prefix-state] reusing existing validated prefix-state index"
  else
    echo "[prefix-state] existing index failed validation; rebuilding"
    python scripts/analysis/prefix_state_transition_tomography/run.py --config "$CONFIG" --stages prefix_state_index,validate --allow-overwrite
  fi
else
  python scripts/analysis/prefix_state_transition_tomography/run.py --config "$CONFIG" --stages prefix_state_index,validate --allow-overwrite
fi
python - "$ARTIFACT_ROOT/prefix_state_index_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("launch_eligible") is not True:
    print("[prefix-state] launch gate failed:", summary.get("failed_launch_gates"), file=sys.stderr)
    raise SystemExit(1)
print("[prefix-state] launch gate passed")
PY
if [[ "\${SKIP_GPU_PREFLIGHT:-0}" == "1" ]]; then
  echo "[prefix-state] SKIP_GPU_PREFLIGHT=1; not checking existing GPU compute apps"
elif ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[prefix-state] nvidia-smi not found; cannot verify GPU availability" >&2
  exit 1
else
  busy=0
  for gpu in $GPU_IDS; do
    apps="\$(nvidia-smi -i "\$gpu" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "\$apps" ]]; then
      busy=1
      echo "[prefix-state] GPU \$gpu already has compute apps:" >&2
      echo "\$apps" | sed 's/^/[prefix-state]   /' >&2
    fi
  done
  if [[ "\$busy" != "0" ]]; then
    echo "[prefix-state] refusing to launch because one or more requested GPUs are busy" >&2
    echo "[prefix-state] set SKIP_GPU_PREFLIGHT=1 only if you intentionally want to share/override GPUs" >&2
    exit 1
  fi
fi
pids=()
: > "\$LOG_ROOT/shard_pids.tsv"
: > "\$LOG_ROOT/shard_status.log"
idx=0
for gpu in $GPU_IDS; do
  echo "[prefix-state] launching paired checkpoint probe shard \$idx on GPU \$gpu"
  log_file="\$LOG_ROOT/prefix_state_paired_probe_shard_\$(printf '%02d' "\$idx").log"
  CUDA_VISIBLE_DEVICES="\$gpu" python scripts/analysis/prefix_state_transition_tomography/run.py --config "$CONFIG" --stages paired_checkpoint_probe --shard-id "\$idx" --allow-overwrite > "\$log_file" 2>&1 &
  pid="\$!"
  pids+=("\$pid")
  printf '%s\t%s\t%s\t%s\n' "\$idx" "\$gpu" "\$pid" "\$log_file" >> "\$LOG_ROOT/shard_pids.tsv"
  idx=\$((idx + 1))
done
status=0
for i in "\${!pids[@]}"; do
  pid="\${pids[\$i]}"
  if wait "\$pid"; then
    echo "[prefix-state] shard \$i pid \$pid exit 0" | tee -a "\$LOG_ROOT/shard_status.log"
  else
    code="\$?"
    echo "[prefix-state] shard \$i pid \$pid exit \$code" | tee -a "\$LOG_ROOT/shard_status.log" >&2
    status=1
  fi
done
if [[ "\$status" != "0" ]]; then
  echo "[prefix-state] one or more paired probe shards failed" >&2
  exit "\$status"
fi
echo "[prefix-state] merging/reporting/gallery"
python scripts/analysis/prefix_state_transition_tomography/run.py --config "$CONFIG" --stages merge,report,gallery,validate --allow-overwrite
echo "[prefix-state] complete"
EOF
chmod +x "$COMMAND_FILE"

tmux new-session -d -s "$SESSION" "bash '$COMMAND_FILE'; exec bash"
echo "Started analysis tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
echo "Status: python $ROOT/scripts/analysis/prefix_state_transition_tomography/status.py --artifact-root '$ARTIFACT_ROOT' --log-root '$LOG_ROOT'"
