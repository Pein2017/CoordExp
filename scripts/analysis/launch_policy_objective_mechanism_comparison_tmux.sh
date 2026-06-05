#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-smoke}"
SESSION="${SESSION:-policy_objective_mechanism_${MODE}}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-1}"
DRY_RUN="${DRY_RUN:-0}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
POST_X1_REAL_SLOT_LIMIT="${POST_X1_REAL_SLOT_LIMIT:-}"

ROOT="/data/CoordExp"
RUN_ROOT="$ROOT/outputs/analysis/autoreg_object_rollout/policy_objective_mechanism_comparison/fullobj_5ckpt_ckpt3668"
LOG_ROOT="$RUN_ROOT/logs/${SESSION}"
PYTHONPATH="$ROOT"
export PYTHONPATH

case "$MODE" in
  smoke)
    A32_CONFIG="$ROOT/configs/analysis/sorted_random_no_newline_phenotype/fullobj_policy_objective_4ckpt_ckpt3668_phase_a3_2_smoke.yaml"
    PREFIX_CONFIG="$ROOT/configs/analysis/prefix_state_transition_tomography/fullobj_policy_objective_5ckpt_ckpt3668_phase_a3_smoke.yaml"
    POST_X1_CONFIG="$ROOT/configs/analysis/post_x1_instance_basin_tomography/five_ckpt_policy_objective_ckpt3668_phase_a3_3_smoke.yaml"
    ;;
  full)
    A32_CONFIG="$ROOT/configs/analysis/sorted_random_no_newline_phenotype/fullobj_policy_objective_4ckpt_ckpt3668_phase_a3_2.yaml"
    PREFIX_CONFIG="$ROOT/configs/analysis/prefix_state_transition_tomography/fullobj_policy_objective_5ckpt_ckpt3668_phase_a3.yaml"
    POST_X1_CONFIG="$ROOT/configs/analysis/post_x1_instance_basin_tomography/five_ckpt_policy_objective_ckpt3668_phase_a3_3.yaml"
    ;;
  *)
    echo "MODE must be smoke or full, got: $MODE" >&2
    exit 2
    ;;
esac

AGG_CONFIG="$ROOT/configs/analysis/policy_objective_mechanism_comparison/fullobj_5ckpt_ckpt3668.yaml"
OVERWRITE_FLAG=()
if [[ "$ALLOW_OVERWRITE" == "1" ]]; then
  OVERWRITE_FLAG=(--allow-overwrite)
fi

mkdir -p "$LOG_ROOT"

run_logged() {
  local name="$1"
  shift
  local log="$LOG_ROOT/${name}.log"
  echo "[$(date -Is)] START $name"
  echo "COMMAND: $*" > "$log"
  if "$@" >> "$log" 2>&1; then
    echo "[$(date -Is)] OK $name"
  else
    local code=$?
    echo "[$(date -Is)] FAIL $name code=$code; log=$log" >&2
    tail -80 "$log" >&2 || true
    exit "$code"
  fi
}

run_logged_bg() {
  local name="$1"
  shift
  local log="$LOG_ROOT/${name}.log"
  (
    set -euo pipefail
    echo "[$(date -Is)] START $name"
    echo "COMMAND: $*" > "$log"
    "$@" >> "$log" 2>&1
    echo "[$(date -Is)] OK $name"
  ) &
  RUN_LOGGED_BG_PID="$!"
}

wait_logged_bg() {
  local name="$1"
  local pid="$2"
  local log="$LOG_ROOT/${name}.log"
  if wait "$pid"; then
    echo "[$(date -Is)] OK $name"
  else
    local code=$?
    echo "[$(date -Is)] FAIL $name code=$code; log=$log" >&2
    tail -80 "$log" >&2 || true
    exit "$code"
  fi
}

run_gpu_shards() {
  local label="$1"
  local command_kind="$2"
  local -a pids=()
  local shard=0
  for gpu in $GPU_LIST; do
    local log="$LOG_ROOT/${label}_shard_${shard}_gpu_${gpu}.log"
    (
      set -euo pipefail
      export CUDA_VISIBLE_DEVICES="$gpu"
      export A33_GPU_ID="$gpu"
      export A32_GPU_ID="$gpu"
      if [[ -n "$POST_X1_REAL_SLOT_LIMIT" ]]; then
        export A33_REAL_SLOT_LIMIT="$POST_X1_REAL_SLOT_LIMIT"
      fi
      echo "[$(date -Is)] START ${label} shard=${shard} gpu=${gpu}"
      case "$command_kind" in
        prefix_probe)
          python "$ROOT/scripts/analysis/prefix_state_transition_tomography/run.py" \
            --config "$PREFIX_CONFIG" \
            --stages paired_checkpoint_probe \
            --shard-id "$shard" \
            "${OVERWRITE_FLAG[@]}"
          ;;
        post_x1_slot)
          python "$ROOT/scripts/analysis/run_post_x1_instance_basin_tomography.py" \
            --config "$POST_X1_CONFIG" \
            --stages slot_posterior \
            --real-runtime \
            --shard-id "$shard" \
            "${OVERWRITE_FLAG[@]}"
          ;;
        *)
          echo "unknown command_kind: $command_kind" >&2
          exit 2
          ;;
      esac
      echo "[$(date -Is)] OK ${label} shard=${shard} gpu=${gpu}"
    ) > "$log" 2>&1 &
    pids+=("$!")
    shard=$((shard + 1))
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "[$(date -Is)] FAIL $label; logs under $LOG_ROOT" >&2
    for log in "$LOG_ROOT"/"${label}"_shard_*.log; do
      echo "--- $log" >&2
      tail -40 "$log" >&2 || true
    done
    exit 1
  fi
  echo "[$(date -Is)] OK $label all shards"
}

main() {
  cd "$ROOT"
  echo "SESSION=$SESSION"
  echo "MODE=$MODE"
  echo "LOG_ROOT=$LOG_ROOT"
  echo "GPU_LIST=$GPU_LIST"
  echo "ALLOW_OVERWRITE=$ALLOW_OVERWRITE"

  run_logged_bg a32_all_cpu_mocked \
    python "$ROOT/scripts/analysis/sorted_random_no_newline_phenotype/run.py" \
      --config "$A32_CONFIG" \
      --stages data_root_audit,prefix_state_index,validate,paired_checkpoint_probe,prefix_merge,prefix_report,prefix_gallery,native_rollout,rollout_phenotype,fn_case_index,fn_hint_probe,fn_merge,fn_report,fn_gallery,finalize \
      --launch-context \
      --mock-runtime \
      "${OVERWRITE_FLAG[@]}"
  a32_pid="$RUN_LOGGED_BG_PID"

  run_logged_bg prefix_index \
    python "$ROOT/scripts/analysis/prefix_state_transition_tomography/run.py" \
      --config "$PREFIX_CONFIG" \
      --stages prefix_state_index,validate \
      "${OVERWRITE_FLAG[@]}"
  prefix_index_pid="$RUN_LOGGED_BG_PID"

  run_logged_bg post_x1_cpu_inputs \
    python "$ROOT/scripts/analysis/run_post_x1_instance_basin_tomography.py" \
      --config "$POST_X1_CONFIG" \
      --stages data_root_audit,case_universe,prefix_states \
      "${OVERWRITE_FLAG[@]}"
  post_x1_cpu_pid="$RUN_LOGGED_BG_PID"

  wait_logged_bg post_x1_cpu_inputs "$post_x1_cpu_pid"
  run_gpu_shards post_x1_slot post_x1_slot

  run_logged post_x1_downstream \
    python "$ROOT/scripts/analysis/run_post_x1_instance_basin_tomography.py" \
      --config "$POST_X1_CONFIG" \
      --stages slot_merge,trajectory,attraction_matrix,prefix_sensitivity,greedy_continuation,report,gallery \
      "${OVERWRITE_FLAG[@]}"

  wait_logged_bg prefix_index "$prefix_index_pid"
  run_gpu_shards prefix_probe prefix_probe

  run_logged prefix_merge_report_gallery \
    python "$ROOT/scripts/analysis/prefix_state_transition_tomography/run.py" \
      --config "$PREFIX_CONFIG" \
      --stages merge,report,gallery \
      "${OVERWRITE_FLAG[@]}"

  wait_logged_bg a32_all_cpu_mocked "$a32_pid"

  run_logged aggregate \
    python "$ROOT/scripts/analysis/policy_objective_mechanism_comparison/run.py" \
      --config "$AGG_CONFIG" \
      "${OVERWRITE_FLAG[@]}"

  echo "[$(date -Is)] COMPLETE policy objective mechanism comparison $MODE"
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1"
  echo "Would start tmux session: $SESSION"
  echo "Mode: $MODE"
  echo "Logs: $LOG_ROOT"
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 2
fi

COMMAND_FILE="$(mktemp -p /tmp policy_objective_mechanism_XXXXXX.sh)"
{
  declare -f run_logged
  declare -f run_logged_bg
  declare -f wait_logged_bg
  declare -f run_gpu_shards
  declare -f main
  printf 'set -euo pipefail\n'
  printf 'MODE=%q\n' "$MODE"
  printf 'SESSION=%q\n' "$SESSION"
  printf 'ALLOW_OVERWRITE=%q\n' "$ALLOW_OVERWRITE"
  printf 'GPU_LIST=%q\n' "$GPU_LIST"
  printf 'POST_X1_REAL_SLOT_LIMIT=%q\n' "$POST_X1_REAL_SLOT_LIMIT"
  printf 'ROOT=%q\n' "$ROOT"
  printf 'RUN_ROOT=%q\n' "$RUN_ROOT"
  printf 'LOG_ROOT=%q\n' "$LOG_ROOT"
  printf 'PYTHONPATH=%q\n' "$PYTHONPATH"
  printf 'A32_CONFIG=%q\n' "$A32_CONFIG"
  printf 'PREFIX_CONFIG=%q\n' "$PREFIX_CONFIG"
  printf 'POST_X1_CONFIG=%q\n' "$POST_X1_CONFIG"
  printf 'AGG_CONFIG=%q\n' "$AGG_CONFIG"
  printf 'OVERWRITE_FLAG=(%s)\n' "${OVERWRITE_FLAG[*]}"
  printf 'export PYTHONPATH\n'
  printf 'mkdir -p "$LOG_ROOT"\n'
  printf 'main\n'
} > "$COMMAND_FILE"

tmux new-session -d -s "$SESSION" "bash '$COMMAND_FILE'; exec bash"
echo "Started tmux session: $SESSION"
echo "Command file: $COMMAND_FILE"
echo "Logs: $LOG_ROOT"
echo "Attach: tmux attach -t $SESSION"
echo "Tail: tmux capture-pane -pt $SESSION:0 -S -200"
