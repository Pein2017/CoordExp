#!/usr/bin/env bash
set -euo pipefail

R=/data/CoordExp/.worktrees/research-probes
U="$R/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-13-owner-successor-scale-throughput"
O=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput
SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
SESSION=coordexp-owner-blind32-v1
LAUNCH_ROOT="$O/evaluation/detached-blind32-recovery-v1"
CONSUMER_ROOT="$O/evaluation/paired-consumer-v1"
QUEUE="$CONSUMER_ROOT/blind-review-queue.jsonl"
OUTPUT="$CONSUMER_ROOT/blind32-review-preparation-v1"
BLIND_REVIEW="$U/evaluation/blind_review.py"
RED_LOG="$O/evaluation/detached-paired-v1/blind32_prepare.log"
QUEUE_SHA256=b5f781b8edd5f5d87a48127e42525323e802888821bb4accd889481749cd8766
BLIND_REVIEW_SHA256=1ae7da233a3ee8e29d03bb46ed158c0feed159d95fac571821283fa38144f3eb
REPO_PYTHONPATH="$R${PYTHONPATH:+:$PYTHONPATH}"

fail() {
    printf 'detached-blind32-recovery-v1: %s\n' "$*" >&2
    exit 1
}

require_sha256() {
    local path=$1 expected=$2 actual
    [[ -f "$path" ]] || fail "required file missing: $path"
    actual=$(sha256sum "$path")
    actual=${actual%% *}
    [[ "$actual" == "$expected" ]] || fail "SHA256 mismatch: $path"
}

write_terminal() {
    local status=$1 stage=$2 ended final_status
    ended=$(date --iso-8601=seconds)
    if [[ "$status" -eq 0 ]]; then
        final_status=completed
    else
        final_status=failed
    fi
    printf '{"schema":"owner_successor_scale.detached_blind32_recovery.v1","session":"%s","status":"%s","started":"%s","ended":"%s","exit_status":%d,"terminal_stage":"%s"}\n' \
        "$SESSION" "$final_status" "$DETACHED_STARTED" "$ended" "$status" "$stage" > "$LAUNCH_ROOT/terminal.json"
}

run_detached() {
    cd "$R"
    CURRENT_STAGE=blind32_prepare
    finish() {
        local status=$?
        trap - EXIT
        write_terminal "$status" "$CURRENT_STAGE"
        exit "$status"
    }
    trap finish EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM

    command_path="$LAUNCH_ROOT/blind32_prepare.command.sh"
    log_path="$LAUNCH_ROOT/blind32_prepare.log"
    started=$(date --iso-8601=seconds)
    command=(env "PYTHONPATH=$REPO_PYTHONPATH" python "$BLIND_REVIEW" prepare --queue "$QUEUE" --output "$OUTPUT")
    {
        printf '#!/usr/bin/env bash\n'
        printf 'cd %q\n' "$R"
        printf '%q ' "${command[@]}"
        printf '\n'
    } > "$command_path"
    chmod 0444 "$command_path"

    set +e
    "${command[@]}" > "$log_path" 2>&1
    status=$?
    set -e
    ended=$(date --iso-8601=seconds)
    printf '{"schema":"owner_successor_scale.detached_stage.v1","stage":"blind32_prepare","started":"%s","ended":"%s","exit_status":%d,"log":"%s","command":"%s"}\n' \
        "$started" "$ended" "$status" "$log_path" "$command_path" > "$LAUNCH_ROOT/blind32_prepare.exit.json"
    [[ "$status" -eq 0 ]] || return "$status"
    CURRENT_STAGE=complete
}

if [[ "${1-}" == __run ]]; then
    run_detached
    exit 0
fi
[[ $# -eq 0 ]] || fail "this launcher takes no arguments"

cd "$R"
command -v tmux >/dev/null || fail "tmux is unavailable"
command -v python >/dev/null || fail "bare python is unavailable"
require_sha256 "$QUEUE" "$QUEUE_SHA256"
require_sha256 "$BLIND_REVIEW" "$BLIND_REVIEW_SHA256"
[[ -f "$RED_LOG" ]] || fail "preserved import-failure log missing: $RED_LOG"
grep -Fq "ModuleNotFoundError: No module named 'probes'" "$RED_LOG" || fail "preserved RED identity changed"
[[ ! -e "$OUTPUT" ]] || fail "output path occupied: $OUTPUT"
[[ ! -e "$LAUNCH_ROOT" ]] || fail "launcher record path occupied: $LAUNCH_ROOT"
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    fail "tmux session occupied: $SESSION"
fi

EXPECTED_PYTHON=$(python -c 'import sys; print(sys.executable)')
RUNTIME_ENV=$(python -c 'from pathlib import Path; import sys; print(Path(sys.prefix).name)')
[[ "$RUNTIME_ENV" == ms ]] || fail "bare python is not the ms runtime: $EXPECTED_PYTHON"
DETACHED_STARTED=$(date --iso-8601=seconds)
mkdir "$LAUNCH_ROOT"
printf 'python=%s\nenvironment=%s\nPYTHONPATH=%s\n' "$EXPECTED_PYTHON" "$RUNTIME_ENV" "$REPO_PYTHONPATH" > "$LAUNCH_ROOT/runtime.txt"

help_started=$(date --iso-8601=seconds)
set +e
env "PYTHONPATH=$REPO_PYTHONPATH" python "$BLIND_REVIEW" --help > "$LAUNCH_ROOT/entry-help.log" 2>&1
help_status=$?
set -e
help_ended=$(date --iso-8601=seconds)
printf '{"schema":"owner_successor_scale.detached_stage.v1","stage":"entry_help","started":"%s","ended":"%s","exit_status":%d,"log":"%s"}\n' \
    "$help_started" "$help_ended" "$help_status" "$LAUNCH_ROOT/entry-help.log" > "$LAUNCH_ROOT/entry-help.exit.json"
if [[ "$help_status" -ne 0 ]]; then
    write_terminal "$help_status" entry_help
    exit "$help_status"
fi

printf '{"schema":"owner_successor_scale.detached_blind32_submission.v1","session":"%s","status":"submitted","submitted":"%s","python":"%s","queue_sha256":"%s","renderer_sha256":"%s","repair":"explicit_repo_PYTHONPATH_only","red_log":"%s"}\n' \
    "$SESSION" "$DETACHED_STARTED" "$EXPECTED_PYTHON" "$QUEUE_SHA256" "$BLIND_REVIEW_SHA256" "$RED_LOG" > "$LAUNCH_ROOT/submission.json"

printf -v detached_command '%q __run' "$SCRIPT"
set +e
tmux new-session -d -s "$SESSION" -c "$R" \
    -e "PATH=$PATH" \
    -e "CONDA_PREFIX=${CONDA_PREFIX-}" \
    -e "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV-}" \
    -e "PYTHONPATH=${PYTHONPATH-}" \
    -e "LD_LIBRARY_PATH=${LD_LIBRARY_PATH-}" \
    -e "DETACHED_STARTED=$DETACHED_STARTED" \
    "$detached_command"
submit_status=$?
set -e
if [[ "$submit_status" -ne 0 ]]; then
    write_terminal "$submit_status" tmux_submission
    exit "$submit_status"
fi

observed=$(date --iso-8601=seconds)
if pane_pid=$(tmux display-message -p -t "=$SESSION" '#{pane_pid}' 2>/dev/null); then
    printf '{"schema":"owner_successor_scale.detached_observation.v1","session":"%s","status":"live_process_observed_and_detached","observed":"%s","pane_pid":%d}\n' \
        "$SESSION" "$observed" "$pane_pid" > "$LAUNCH_ROOT/detached.json"
    printf 'detached session=%s pane_pid=%s records=%s\n' "$SESSION" "$pane_pid" "$LAUNCH_ROOT"
elif [[ -f "$LAUNCH_ROOT/terminal.json" ]]; then
    terminal_status=$(python -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["exit_status"]))' "$LAUNCH_ROOT/terminal.json")
    printf 'terminal-before-observation exit_status=%s records=%s\n' "$terminal_status" "$LAUNCH_ROOT"
    exit "$terminal_status"
else
    fail "session vanished without terminal record: $SESSION"
fi
