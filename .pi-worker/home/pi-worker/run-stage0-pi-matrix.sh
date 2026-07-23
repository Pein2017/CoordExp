#!/usr/bin/env bash
set -euo pipefail

worktree_root=/data/CoordExp/.worktrees/research-probes
local_root="$worktree_root/.pi-worker"
base_root="$local_root/sandbox-base-v2"
fixture_root=/data/CoordExp/outputs/research/pi-lightweight-worker-ablation/2026-07-22-stage0-frozen-task-harness-screen/fixtures-v1
auth_source="$local_root/agent/auth.json"
state_path=/data/CoordExp/.worktrees/research-probes/.pi-worker

test -s "$auth_source"
test -x "$base_root/opt/node/bin/node"

source "$local_root/home/.bashrc"

# Pi's provider route requires the host proxy in this environment. Preserve
# only proxy variables across the otherwise-clean environment boundary. Do
# not print this array because its values are environment-local credentials.
proxy_env_args=()
for name in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy; do
  if [[ -v "$name" ]]; then
    proxy_env_args+=("$name=${!name}")
  fi
done
test "${#proxy_env_args[@]}" -gt 0

task_dir() {
  case "$1" in
    01) printf '%s\n' task-01-artifact-inventory ;;
    02) printf '%s\n' task-02-source-routing ;;
    03) printf '%s\n' task-03-claim-audit ;;
    04) printf '%s\n' task-04-mechanical-aggregation ;;
    *) return 2 ;;
  esac
}

run_one() {
  task="$1"
  model="$2"
  level="$3"
  original_run_id="pi-${model}-t${task}-${level}"
  run_id="${original_run_id}-proxy9090-rerun1"
  run_parent="$local_root/runs/$run_id"
  run_root="$run_parent/root"
  result_dir="$local_root/results/$run_id"
  worker_dir="$fixture_root/$(task_dir "$task")/worker"

  test ! -e "$run_parent"
  test ! -e "$result_dir"
  mkdir -p "$run_parent" "$result_dir"
  cp -al "$base_root" "$run_root"

  cp -a "$worker_dir/." "$run_root/workspace/"
  chown -R 65534:65534 "$run_root/workspace"
  chmod -R u+rwX "$run_root/workspace"

  install -o 65534 -g 65534 -m 0600 \
    "$auth_source" \
    "$run_root$state_path/agent/auth.json"
  chown -R 65534:65534 \
    "$run_root$state_path/home" \
    "$run_root$state_path/agent" \
    "$run_root$state_path/sessions" \
    "$run_root/tmp"

  start_epoch=$(date +%s)
  set +e
  timeout --signal=TERM --kill-after=30s 1200s \
    /usr/bin/env -i \
      "${proxy_env_args[@]}" \
      HOME="$state_path/home" \
      PI_CODING_AGENT_DIR="$state_path/agent" \
      PI_CODING_AGENT_SESSION_DIR="$state_path/sessions" \
      PI_TELEMETRY=0 \
      PATH=/opt/node/bin:/usr/bin:/bin \
      TERM=dumb \
      /usr/sbin/chroot --userspec=65534:65534 "$run_root" \
      /bin/bash -lc \
      "cd /workspace && exec /opt/node/bin/node /opt/pi/node_modules/@earendil-works/pi-coding-agent/dist/cli.js --mode json --print --no-session --provider openai-codex --model gpt-5.6-${model} --thinking ${level} @task.md 'Follow task.md exactly and stop after its required output.'" \
      >"$result_dir/events.jsonl" \
      2>"$result_dir/stderr.log"
  exit_code=$?
  set -e
  end_epoch=$(date +%s)

  if jq -e -s '
    ([.[] | select(.type == "message_end" and .message.role == "assistant")] | last) as $event
    | $event != null
      and $event.message.stopReason != "error"
      and (($event.message.usage.totalTokens // 0) > 0)
  ' "$result_dir/events.jsonl" >/dev/null; then
    valid_model_response=yes
  else
    valid_model_response=no
  fi

  {
    printf 'run_id=%s\n' "$run_id"
    printf 'infrastructure_retry_of=%s\n' "$original_run_id"
    printf 'proxy_route=127.0.0.1:9090\n'
    printf 'task=%s\n' "$task"
    printf 'harness=pi-0.81.1\n'
    printf 'provider=openai-codex\n'
    printf 'model=gpt-5.6-%s\n' "$model"
    printf 'reasoning=%s\n' "$level"
    printf 'start_epoch=%s\n' "$start_epoch"
    printf 'end_epoch=%s\n' "$end_epoch"
    printf 'wall_seconds=%s\n' "$((end_epoch - start_epoch))"
    printf 'exit_code=%s\n' "$exit_code"
    printf 'valid_model_response=%s\n' "$valid_model_response"
    printf 'events_sha256='; sha256sum "$result_dir/events.jsonl" | cut -d' ' -f1
    printf 'stderr_sha256='; sha256sum "$result_dir/stderr.log" | cut -d' ' -f1
  } >"$result_dir/receipt.env"
  chmod 0600 "$result_dir"/*

  printf '%s exit=%s wall=%ss\n' "$run_id" "$exit_code" "$((end_epoch - start_epoch))"
}

run_one 01 luna medium &
p1=$!
run_one 01 terra high &
p2=$!
run_one 01 sol xhigh &
p3=$!
wait "$p1" "$p2" "$p3"

run_one 02 luna high &
p1=$!
run_one 02 terra xhigh &
p2=$!
run_one 02 sol medium &
p3=$!
wait "$p1" "$p2" "$p3"

run_one 03 luna xhigh &
p1=$!
run_one 03 terra medium &
p2=$!
run_one 03 sol high &
p3=$!
wait "$p1" "$p2" "$p3"

run_one 04 luna medium &
p1=$!
run_one 04 terra high &
p2=$!
run_one 04 sol xhigh &
p3=$!
wait "$p1" "$p2" "$p3"
