#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
RUN_PY="$REPO_ROOT/scripts/analysis/sorted_random_no_newline_phenotype/run.py"
STATUS_PY="$REPO_ROOT/scripts/analysis/sorted_random_no_newline_phenotype/status.py"
DEFAULT_CONFIG="$REPO_ROOT/configs/analysis/sorted_random_no_newline_phenotype/fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2.yaml"

CONFIG="${CONFIG:-$DEFAULT_CONFIG}"
SESSION="${SESSION:-sorted_random_no_newline_a3_2_ckpt3668}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS:-0}"
SKIP_GPU_PREFLIGHT="${SKIP_GPU_PREFLIGHT:-0}"
ENABLE_A3_2_REAL_GPU_RUNTIME="${ENABLE_A3_2_REAL_GPU_RUNTIME:-0}"
REAL_GPU_RUNTIME_CMD="${REAL_GPU_RUNTIME_CMD:-}"

if [[ -n "${ARTIFACT_ROOT:-}" ]]; then
  echo "ARTIFACT_ROOT override is refused; artifact_root must come from CONFIG." >&2
  exit 2
fi

if [[ "$CONFIG" != /* ]]; then
  CONFIG="$REPO_ROOT/$CONFIG"
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "CONFIG does not exist: $CONFIG" >&2
  exit 2
fi

PLAN_JSON="$("$PYTHON_BIN" "$RUN_PY" --config "$CONFIG" --dry-run)"

json_get() {
  local expr="$1"
  "$PYTHON_BIN" -c '
import json
import sys

data = json.load(sys.stdin)
value = data
for part in sys.argv[1].split("."):
    value = value[part]
if isinstance(value, list):
    print(" ".join(str(item) for item in value))
elif isinstance(value, bool):
    print("true" if value else "false")
else:
    print(value)
' "$expr" <<<"$PLAN_JSON"
}

PROJECT_ID="$(json_get project_id)"
PHASE_ID="$(json_get phase_id)"
RUN_ID="$(json_get run_id)"
ARTIFACT_ROOT_FROM_CONFIG="$(json_get artifact_root)"
CHECKPOINT_ROLES="$(json_get checkpoint_roles)"
NUM_SHARDS="$(json_get sampling.num_shards)"

GPU_IDS_NORMALIZED="${GPU_IDS//,/ }"
read -r -a GPU_ARRAY <<<"$GPU_IDS_NORMALIZED"
if [[ "${#GPU_ARRAY[@]}" -ne "$NUM_SHARDS" ]]; then
  echo "GPU_IDS count (${#GPU_ARRAY[@]}) must equal sampling.num_shards ($NUM_SHARDS)." >&2
  exit 2
fi

LOG_ROOT="${LOG_ROOT:-$ARTIFACT_ROOT_FROM_CONFIG/logs/tmux_$SESSION}"
SHARD_PIDS="$ARTIFACT_ROOT_FROM_CONFIG/shard_pids.tsv"
SHARD_STATUS_LOG="$ARTIFACT_ROOT_FROM_CONFIG/shard_status.log"

print_plan() {
  echo "DRY_RUN=1 A3.2 tmux launch plan"
  echo "config=$CONFIG"
  echo "project_id=$PROJECT_ID"
  echo "phase_id=$PHASE_ID"
  echo "run_id=$RUN_ID"
  echo "artifact_root=$ARTIFACT_ROOT_FROM_CONFIG"
  echo "checkpoint_roles=$CHECKPOINT_ROLES"
  echo "num_shards=$NUM_SHARDS"
  echo "gpu_ids=${GPU_ARRAY[*]}"
  echo "session=$SESSION"
  echo "planned_shard_pids=$SHARD_PIDS"
  echo "planned_shard_status_log=$SHARD_STATUS_LOG"
  echo "planned_log_root=$LOG_ROOT"
  for shard_id in "${!GPU_ARRAY[@]}"; do
    local gpu_id="${GPU_ARRAY[$shard_id]}"
    echo "planned_stage=paired_checkpoint_probe shard_id=$shard_id gpu_id=$gpu_id log=$LOG_ROOT/paired_checkpoint_probe_shard_${shard_id}.log"
  done
  echo "planned_stage=native_rollout launch_context=required log=$LOG_ROOT/native_rollout.log"
  for shard_id in "${!GPU_ARRAY[@]}"; do
    local gpu_id="${GPU_ARRAY[$shard_id]}"
    echo "planned_stage=fn_hint_probe shard_id=$shard_id gpu_id=$gpu_id log=$LOG_ROOT/fn_hint_probe_shard_${shard_id}.log"
  done
}

if [[ "$DRY_RUN" == "1" ]]; then
  print_plan
  exit 0
fi

STATUS_JSON="$("$PYTHON_BIN" "$STATUS_PY" --artifact-root "$ARTIFACT_ROOT_FROM_CONFIG")"
STATUS_VALUE="$("$PYTHON_BIN" -c 'import json, sys; print(json.load(sys.stdin)["status"])' <<<"$STATUS_JSON")"
if [[ "$STATUS_VALUE" != "index_ready_pending_gpu" ]]; then
  echo "Status gate refused launch: expected index_ready_pending_gpu, got $STATUS_VALUE." >&2
  echo "$STATUS_JSON" >&2
  exit 2
fi

if [[ "$ENABLE_A3_2_REAL_GPU_RUNTIME" != "1" ]]; then
  echo "A3.2 real GPU runtime is not wired in this launcher yet; refusing to start placeholder tmux jobs." >&2
  echo "Use DRY_RUN=1 for planning, or wire the dedicated real runtime and set ENABLE_A3_2_REAL_GPU_RUNTIME=1." >&2
  exit 2
fi

if [[ -z "$REAL_GPU_RUNTIME_CMD" ]]; then
  echo "A3.2 real GPU runtime command is not configured; refusing to launch generic placeholder jobs." >&2
  exit 2
fi

if [[ "$ALLOW_OVERWRITE" != "1" ]]; then
  if [[ -e "$SHARD_PIDS" || -e "$SHARD_STATUS_LOG" || -d "$LOG_ROOT" ]]; then
    echo "Launch metadata/logs already exist; refusing to overwrite without ALLOW_OVERWRITE=1." >&2
    echo "shard_pids=$SHARD_PIDS" >&2
    echo "shard_status_log=$SHARD_STATUS_LOG" >&2
    echo "log_root=$LOG_ROOT" >&2
    exit 2
  fi
else
  if [[ -d "$LOG_ROOT" ]]; then
    echo "Log root already exists; refusing recursive cleanup. Use a fresh SESSION/LOG_ROOT or remove it manually." >&2
    echo "log_root=$LOG_ROOT" >&2
    exit 2
  fi
fi

if [[ "$ALLOW_BUSY_GPUS" != "1" && "$SKIP_GPU_PREFLIGHT" != "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; set SKIP_GPU_PREFLIGHT=1 to override explicitly." >&2
    exit 2
  fi
  COMPUTE_APPS="$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "${COMPUTE_APPS//[[:space:]]/}" ]]; then
    echo "GPU compute apps are present; refusing conservative launch." >&2
    echo "$COMPUTE_APPS" >&2
    echo "Set ALLOW_BUSY_GPUS=1 only if this launch is intentionally sharing GPUs." >&2
    exit 2
  fi
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for real launch." >&2
  exit 2
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT" "$ARTIFACT_ROOT_FROM_CONFIG"
printf "shard_id\tstage\tgpu_id\twindow\tlog_path\n" >"$SHARD_PIDS"
printf "session=%s\nartifact_root=%s\n" "$SESSION" "$ARTIFACT_ROOT_FROM_CONFIG" >"$SHARD_STATUS_LOG"

ALLOW_ARG=""
if [[ "$ALLOW_OVERWRITE" == "1" ]]; then
  ALLOW_ARG="--allow-overwrite"
fi
CONTROLLER_SCRIPT="$LOG_ROOT/controller.sh"
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "cd '$REPO_ROOT'"
  echo "echo '[controller] A3.2 run started at '\"\$(date -Is)\" >>'$SHARD_STATUS_LOG'"
  echo "echo '[controller] prefix readout wave starting' >>'$SHARD_STATUS_LOG'"
  echo "pids=()"
  for shard_id in "${!GPU_ARRAY[@]}"; do
    gpu_id="${GPU_ARRAY[$shard_id]}"
    log_path="$LOG_ROOT/paired_checkpoint_probe_shard_${shard_id}.log"
    echo "CUDA_VISIBLE_DEVICES='$gpu_id' PYTHONDONTWRITEBYTECODE=1 $REAL_GPU_RUNTIME_CMD --config '$CONFIG' --stages paired_checkpoint_probe --shard-id '$shard_id' --launch-context $ALLOW_ARG >'$log_path' 2>&1 &"
    echo "pids+=(\"\$!\")"
  done
  echo "for pid in \"\${pids[@]}\"; do wait \"\$pid\"; done"
  echo "echo '[controller] prefix CPU merge/report/gallery starting' >>'$SHARD_STATUS_LOG'"
  echo "PYTHONDONTWRITEBYTECODE=1 '$PYTHON_BIN' '$RUN_PY' --config '$CONFIG' --stages prefix_merge,prefix_report,prefix_gallery $ALLOW_ARG >>'$LOG_ROOT/prefix_cpu_merge_report_gallery.log' 2>&1"
  echo "echo '[controller] native rollout starting' >>'$SHARD_STATUS_LOG'"
  echo "CUDA_VISIBLE_DEVICES='${GPU_ARRAY[0]}' PYTHONDONTWRITEBYTECODE=1 $REAL_GPU_RUNTIME_CMD --config '$CONFIG' --stages native_rollout --launch-context $ALLOW_ARG >>'$LOG_ROOT/native_rollout.log' 2>&1"
  echo "echo '[controller] rollout phenotype and FN case index starting' >>'$SHARD_STATUS_LOG'"
  echo "PYTHONDONTWRITEBYTECODE=1 '$PYTHON_BIN' '$RUN_PY' --config '$CONFIG' --stages rollout_phenotype,fn_case_index $ALLOW_ARG >>'$LOG_ROOT/rollout_phenotype_fn_case_index.log' 2>&1"
  echo "echo '[controller] FN hint wave starting' >>'$SHARD_STATUS_LOG'"
  echo "pids=()"
  for shard_id in "${!GPU_ARRAY[@]}"; do
    gpu_id="${GPU_ARRAY[$shard_id]}"
    log_path="$LOG_ROOT/fn_hint_probe_shard_${shard_id}.log"
    echo "CUDA_VISIBLE_DEVICES='$gpu_id' PYTHONDONTWRITEBYTECODE=1 $REAL_GPU_RUNTIME_CMD --config '$CONFIG' --stages fn_hint_probe --shard-id '$shard_id' --launch-context $ALLOW_ARG >>'$log_path' 2>&1 &"
    echo "pids+=(\"\$!\")"
  done
  echo "for pid in \"\${pids[@]}\"; do wait \"\$pid\"; done"
  echo "echo '[controller] FN merge/report/gallery/finalize starting' >>'$SHARD_STATUS_LOG'"
  echo "PYTHONDONTWRITEBYTECODE=1 '$PYTHON_BIN' '$RUN_PY' --config '$CONFIG' --stages fn_merge,fn_report,fn_gallery,finalize $ALLOW_ARG >>'$LOG_ROOT/fn_merge_report_gallery_finalize.log' 2>&1"
  echo "echo '[controller] A3.2 run finished at '\"\$(date -Is)\" >>'$SHARD_STATUS_LOG'"
} >"$CONTROLLER_SCRIPT"
chmod +x "$CONTROLLER_SCRIPT"

for shard_id in "${!GPU_ARRAY[@]}"; do
  gpu_id="${GPU_ARRAY[$shard_id]}"
  log_path="$LOG_ROOT/paired_checkpoint_probe_shard_${shard_id}.log"
  window_name="controller"
  printf "%s\t%s\t%s\t%s\t%s\n" "$shard_id" "paired_checkpoint_probe" "$gpu_id" "$window_name" "$log_path" >>"$SHARD_PIDS"
done

native_log="$LOG_ROOT/native_rollout.log"
printf "all\tnative_rollout\t%s\t%s\t%s\n" "${GPU_ARRAY[0]}" "controller" "$native_log" >>"$SHARD_PIDS"

for shard_id in "${!GPU_ARRAY[@]}"; do
  gpu_id="${GPU_ARRAY[$shard_id]}"
  log_path="$LOG_ROOT/fn_hint_probe_shard_${shard_id}.log"
  window_name="controller"
  printf "%s\t%s\t%s\t%s\t%s\n" "$shard_id" "fn_hint_probe" "$gpu_id" "$window_name" "$log_path" >>"$SHARD_PIDS"
done

tmux new-session -d -s "$SESSION" -n controller "$CONTROLLER_SCRIPT"

echo "Launched $SESSION"
echo "shard_pids=$SHARD_PIDS"
echo "shard_status_log=$SHARD_STATUS_LOG"
echo "log_root=$LOG_ROOT"
echo "controller_script=$CONTROLLER_SCRIPT"
