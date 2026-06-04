#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-/data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/candidate_field_cardinality_tomography/ckpt3664_smoke.yaml}"
SESSION="${SESSION:-candidate_field_cardinality_ckpt3664}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHONPATH="${PYTHONPATH:-$ROOT}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/$SESSION}"
export PYTHONPATH

gpu_preflight() {
  if [[ "${SKIP_GPU_PREFLIGHT:-0}" == "1" ]]; then
    echo "[candidate-field] SKIP_GPU_PREFLIGHT=1; not checking existing GPU compute apps"
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[candidate-field] nvidia-smi not found; cannot verify GPU availability" >&2
    return 1
  fi
  local busy=0
  local gpu
  for gpu in $GPU_IDS; do
    local apps
    apps="$(nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "$apps" ]]; then
      busy=1
      echo "[candidate-field] GPU $gpu already has compute apps:" >&2
      echo "$apps" | sed 's/^/[candidate-field]   /' >&2
    fi
  done
  if [[ "$busy" != "0" ]]; then
    echo "[candidate-field] refusing to launch because one or more requested GPUs are busy" >&2
    echo "[candidate-field] set SKIP_GPU_PREFLIGHT=1 only if you intentionally want to share/override GPUs" >&2
    return 1
  fi
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN candidate-field tomography, not production training"
  python "$ROOT/scripts/analysis/candidate_field_cardinality_tomography/run.py" \
    --config "$CONFIG" \
    --stages case_index,probe_plan,validate \
    --dry-run
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

gpu_preflight

COMMAND_FILE="$(mktemp -p /tmp candidate_field_tmux_XXXXXX.sh)"
cat > "$COMMAND_FILE" <<EOF
set -euo pipefail
cd "$ROOT"
export PYTHONPATH="$PYTHONPATH"
LOG_ROOT="$LOG_ROOT"
mkdir -p "\$LOG_ROOT"
echo "[candidate-field] building denominator/probe plan, not production training"
python scripts/analysis/candidate_field_cardinality_tomography/run.py --config "$CONFIG" --stages case_index,probe_plan --allow-overwrite
pids=()
: > "\$LOG_ROOT/shard_pids.tsv"
: > "\$LOG_ROOT/shard_status.log"
idx=0
for gpu in $GPU_IDS; do
  echo "[candidate-field] launching x1 shard \$idx on GPU \$gpu"
  log_file="\$LOG_ROOT/candidate_field_x1_shard_\$(printf '%03d' "\$idx").log"
  CUDA_VISIBLE_DEVICES="\$gpu" python scripts/analysis/candidate_field_cardinality_tomography/run.py --config "$CONFIG" --stages x1_candidate_field --shard-id "\$idx" --allow-overwrite > "\$log_file" 2>&1 &
  pid="\$!"
  pids+=("\$pid")
  printf '%s\t%s\t%s\t%s\n' "\$idx" "\$gpu" "\$pid" "\$log_file" >> "\$LOG_ROOT/shard_pids.tsv"
  idx=\$((idx + 1))
done
status=0
for i in "\${!pids[@]}"; do
  pid="\${pids[\$i]}"
  if wait "\$pid"; then
    echo "[candidate-field] shard \$i pid \$pid exit 0" | tee -a "\$LOG_ROOT/shard_status.log"
  else
    code="\$?"
    echo "[candidate-field] shard \$i pid \$pid exit \$code" | tee -a "\$LOG_ROOT/shard_status.log" >&2
    status=1
  fi
done
if [[ "\$status" != "0" ]]; then
  echo "[candidate-field] one or more x1 shards failed" >&2
  exit "\$status"
fi
echo "[candidate-field] merging and validating"
python scripts/analysis/candidate_field_cardinality_tomography/run.py --config "$CONFIG" --stages merge,taxonomy,validate,report --allow-overwrite
echo "[candidate-field] complete"
EOF
chmod +x "$COMMAND_FILE"

tmux new-session -d -s "$SESSION" "bash '$COMMAND_FILE'; exec bash"
echo "Started analysis tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
