#!/usr/bin/env bash
set -euo pipefail

R=/data/CoordExp/.worktrees/research-probes
U="$R/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-13-owner-successor-scale-throughput"
O=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput
SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
SESSION=coordexp-owner-paired-v1
LAUNCH_ROOT="$O/evaluation/detached-paired-v1"
PACKET="$O/evaluation/paired-preparation/packet.json"
BOUND_PACKET="$O/evaluation/paired-preparation/packet-bound.json"
ROOT_ACCEPTANCE="$O/evaluation/root-endpoint-acceptance-v1.json"
A_RECEIPT="$O/training/full-A-v2/receipt.json"
A_COLD="$O/training/full-A-v2/cold-check.json"
B_RECEIPT="$O/training/full-B-v2/receipt.json"
B_COLD="$O/training/full-B-v2/cold-check.json"
A_OUTPUT="$O/evaluation/paired-natural-A-v1"
B_OUTPUT="$O/evaluation/paired-natural-B-v1"
CONSUMER_OUTPUT="$O/evaluation/paired-consumer-v1"
BLIND_OUTPUT="$CONSUMER_OUTPUT/blind32-review-preparation-v1"
BLIND_REVIEW="$U/evaluation/blind_review.py"
PAIRED_SOURCE="$R/probes/owner_successor_scale/paired_evaluation.py"
PACKET_SHA256=fa7e156f8d0b19ec36a34c05548258fe443e1a6365079d60e29be610b30e7232
BOUND_PACKET_SHA256=269fe422507738a7ff27549466480dcaa7ed4762ed7283626ce419c440206e1a
ROOT_ACCEPTANCE_SHA256=688f253e67f0c461bfb9bd8efb772339f5da3996ede4ddb9d6415a7c2de5600c
PAIRED_SOURCE_SHA256=c0b22bd0effe8c930f0f9063ae1ae7c15819f12766ceb91e3457d6321cdd94e9

fail() {
    printf 'detached-paired-v1: %s\n' "$*" >&2
    exit 1
}

require_unoccupied() {
    local path=$1
    [[ ! -e "$path" ]] || fail "output path occupied: $path"
}

require_sha256() {
    local path=$1 expected=$2 actual
    [[ -f "$path" ]] || fail "required file missing: $path"
    actual=$(sha256sum "$path")
    actual=${actual%% *}
    [[ "$actual" == "$expected" ]] || fail "SHA256 mismatch: $path"
}

run_stage() {
    local name=$1
    shift
    local started ended status command_path log_path record_path
    started=$(date --iso-8601=seconds)
    CURRENT_STAGE=$name
    command_path="$LAUNCH_ROOT/${name}.command.sh"
    log_path="$LAUNCH_ROOT/${name}.log"
    record_path="$LAUNCH_ROOT/${name}.exit.json"
    {
        printf '#!/usr/bin/env bash\n'
        printf 'cd %q\n' "$R"
        printf '%q ' "$@"
        printf '\n'
    } > "$command_path"
    chmod 0444 "$command_path"

    set +e
    "$@" > "$log_path" 2>&1
    status=$?
    set -e
    ended=$(date --iso-8601=seconds)
    printf '{"schema":"owner_successor_scale.detached_stage.v1","stage":"%s","started":"%s","ended":"%s","exit_status":%d,"log":"%s","command":"%s"}\n' \
        "$name" "$started" "$ended" "$status" "$log_path" "$command_path" > "$record_path"
    cat "$record_path" >> "$LAUNCH_ROOT/stages.jsonl"
    [[ "$status" -eq 0 ]] || return "$status"
}

run_detached() {
    cd "$R"
    CURRENT_STAGE=runtime_and_bound_preflight

    finish() {
        local status=$? ended final_status
        trap - EXIT
        ended=$(date --iso-8601=seconds)
        if [[ "$status" -eq 0 ]]; then
            final_status=completed
        else
            final_status=failed
        fi
        printf '{"schema":"owner_successor_scale.detached_paired.v1","session":"%s","status":"%s","started":"%s","ended":"%s","exit_status":%d,"terminal_stage":"%s"}\n' \
            "$SESSION" "$final_status" "$DETACHED_STARTED" "$ended" "$status" "$CURRENT_STAGE" > "$LAUNCH_ROOT/terminal.json"
        exit "$status"
    }
    trap finish EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM

    run_stage runtime_and_bound_preflight python -c '
import sys
from pathlib import Path
from probes.owner_successor_scale import paired_evaluation as pe

bound_path, base_path, receipt_a, cold_a, receipt_b, cold_b, root_acceptance_path, a_output, b_output, consumer_output, expected_python, expected_source_sha = map(Path, sys.argv[1:13])
pe.require(Path(sys.executable) == expected_python, "tmux changed bare-python runtime")
pe.require(Path(sys.prefix).name == "ms", "bare python is not the ms runtime")
pe.require(pe.binding(Path(pe.__file__).resolve())["sha256"] == str(expected_source_sha), "paired evaluator source SHA changed")
bound = pe.read(bound_path)
pe._validate_bound(bound)
pe.require(bound.get("base_packet") == pe.binding(base_path), "bound packet does not bind the immutable base packet")
for arm, receipt, cold in (("A", receipt_a, cold_a), ("B", receipt_b, cold_b)):
    endpoint = bound["endpoints"][arm]
    pe.require(endpoint["receipt"] == pe.binding(receipt), f"arm {arm} is not bound to the actual full256 receipt")
    pe.require(endpoint["cold_check"] == pe.binding(cold), f"arm {arm} is not bound to the actual cold check")
grant = pe.read(root_acceptance_path)
pe.require(grant.get("schema") == "owner_successor_scale.paired_evaluation.root_launch_grant.v1", "root acceptance schema changed")
pe.require(grant.get("status") == "endpoints_lead_accepted_launcher_review_pending", "root endpoint acceptance status changed")
pe.require(grant.get("packet") == {"path": str(bound_path), "sha256": pe.binding(bound_path)["sha256"]}, "root acceptance packet changed")
pe.require(grant.get("producer_sha256") == str(expected_source_sha), "root acceptance producer changed")
pe.require(grant.get("arms", {}).get("A", {}).get("output") == str(a_output), "root-accepted A output changed")
pe.require(grant.get("arms", {}).get("B", {}).get("output") == str(b_output), "root-accepted B output changed")
pe.require(grant.get("consumer_output") == str(consumer_output), "root-accepted consumer output changed")
pe.require(grant.get("gpus") == list(range(8)), "root-accepted GPU assignment changed")
pe.require(grant.get("schedule") == ["A launch", "A merge", "B launch", "B merge", "paired cold consume", "blind32 CPU prepare"], "root-accepted schedule changed")
pe.require(grant.get("attempts_per_arm") == 1, "root-accepted attempt count changed")
pe.require(grant.get("wall_limit_seconds_per_arm") == 86400, "root-accepted wall limit changed")
pe.require(grant.get("images_per_arm") == 896 and grant.get("shards_per_arm") == 8 and grant.get("images_per_shard") == 112, "root-accepted image denominator changed")
pe.require(grant.get("max_output_tokens_per_image") == 3084 and grant.get("max_generated_tokens_per_arm") == 2763264, "root-accepted token bound changed")
pe.require(grant.get("model_loads_per_arm") == 8, "root-accepted model-load denominator changed")
pe.require(grant.get("retry_policy") == "none; preserve partial evidence and stop on any failed gate", "root-accepted retry policy changed")
contract = bound.get("launch_contract", {})
pe.require(contract.get("workers_per_arm") == 8, "worker denominator changed")
pe.require(contract.get("images_per_arm") == 896, "image denominator changed")
pe.require(contract.get("max_new_tokens_per_image") == 3084, "token cap changed")
pe.require(contract.get("model_loads_per_arm") == 8, "model-load denominator changed")
pe.require(contract.get("retry_policy") == "none; preserve partial shards", "retry policy changed")
print(sys.executable)
print("bound_packet_sha256=" + pe.binding(bound_path)["sha256"])
' "$BOUND_PACKET" "$PACKET" "$A_RECEIPT" "$A_COLD" "$B_RECEIPT" "$B_COLD" "$ROOT_ACCEPTANCE" "$A_OUTPUT" "$B_OUTPUT" "$CONSUMER_OUTPUT" "$EXPECTED_PYTHON" "$PAIRED_SOURCE_SHA256"

    run_stage launch_A \
        timeout --signal=TERM --kill-after=60s 86400s \
        python -m probes.owner_successor_scale.paired_evaluation launch \
        --packet "$BOUND_PACKET" --arm A --gpus 0,1,2,3,4,5,6,7 --output "$A_OUTPUT"
    run_stage merge_A \
        python -m probes.owner_successor_scale.paired_evaluation merge \
        --packet "$BOUND_PACKET" --arm A --output "$A_OUTPUT"
    run_stage launch_B \
        timeout --signal=TERM --kill-after=60s 86400s \
        python -m probes.owner_successor_scale.paired_evaluation launch \
        --packet "$BOUND_PACKET" --arm B --gpus 0,1,2,3,4,5,6,7 --output "$B_OUTPUT"
    run_stage merge_B \
        python -m probes.owner_successor_scale.paired_evaluation merge \
        --packet "$BOUND_PACKET" --arm B --output "$B_OUTPUT"
    run_stage consume \
        python -m probes.owner_successor_scale.paired_evaluation consume \
        --packet "$BOUND_PACKET" --rows-a "$A_OUTPUT/rows.jsonl" --rows-b "$B_OUTPUT/rows.jsonl" \
        --output "$CONSUMER_OUTPUT"
    run_stage blind32_prepare \
        python "$BLIND_REVIEW" prepare \
        --queue "$CONSUMER_OUTPUT/blind-review-queue.jsonl" --output "$BLIND_OUTPUT"
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
require_sha256 "$PACKET" "$PACKET_SHA256"
require_sha256 "$BOUND_PACKET" "$BOUND_PACKET_SHA256"
require_sha256 "$ROOT_ACCEPTANCE" "$ROOT_ACCEPTANCE_SHA256"
require_sha256 "$PAIRED_SOURCE" "$PAIRED_SOURCE_SHA256"
for required in "$A_RECEIPT" "$A_COLD" "$B_RECEIPT" "$B_COLD" "$BLIND_REVIEW"; do
    [[ -f "$required" ]] || fail "required file missing: $required"
done
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    fail "tmux session occupied: $SESSION"
fi
for output in "$LAUNCH_ROOT" "$A_OUTPUT" "$B_OUTPUT" "$CONSUMER_OUTPUT" "$BLIND_OUTPUT"; do
    require_unoccupied "$output"
done

EXPECTED_PYTHON=$(python -c 'import sys; print(sys.executable)')
RUNTIME_ENV=$(python -c 'from pathlib import Path; import sys; print(Path(sys.prefix).name)')
[[ "$RUNTIME_ENV" == ms ]] || fail "bare python is not the ms runtime: $EXPECTED_PYTHON"
DETACHED_STARTED=$(date --iso-8601=seconds)
mkdir "$LAUNCH_ROOT"
printf 'python=%s\nenvironment=%s\npath=%s\n' "$EXPECTED_PYTHON" "$RUNTIME_ENV" "$PATH" > "$LAUNCH_ROOT/runtime.txt"
printf '{"schema":"owner_successor_scale.detached_submission.v1","session":"%s","status":"submitted","submitted":"%s","python":"%s","base_packet_sha256":"%s","bound_packet_sha256":"%s","root_acceptance_sha256":"%s","paired_source_sha256":"%s"}\n' \
    "$SESSION" "$DETACHED_STARTED" "$EXPECTED_PYTHON" "$PACKET_SHA256" "$BOUND_PACKET_SHA256" "$ROOT_ACCEPTANCE_SHA256" "$PAIRED_SOURCE_SHA256" > "$LAUNCH_ROOT/submission.json"

printf -v detached_command '%q __run' "$SCRIPT"
set +e
tmux new-session -d -s "$SESSION" -c "$R" \
    -e "PATH=$PATH" \
    -e "CONDA_PREFIX=${CONDA_PREFIX-}" \
    -e "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV-}" \
    -e "PYTHONPATH=${PYTHONPATH-}" \
    -e "LD_LIBRARY_PATH=${LD_LIBRARY_PATH-}" \
    -e "EXPECTED_PYTHON=$EXPECTED_PYTHON" \
    -e "DETACHED_STARTED=$DETACHED_STARTED" \
    "$detached_command"
submit_status=$?
set -e
if [[ "$submit_status" -ne 0 ]]; then
    ended=$(date --iso-8601=seconds)
    printf '{"schema":"owner_successor_scale.detached_paired.v1","session":"%s","status":"failed_to_detach","started":"%s","ended":"%s","exit_status":%d,"terminal_stage":"tmux_submission"}\n' \
        "$SESSION" "$DETACHED_STARTED" "$ended" "$submit_status" > "$LAUNCH_ROOT/terminal.json"
    exit "$submit_status"
fi
if ! pane_pid=$(tmux display-message -p -t "=$SESSION" '#{pane_pid}' 2>/dev/null); then
    fail "detached session exited before live-process observation; inspect $LAUNCH_ROOT"
fi
detached_at=$(date --iso-8601=seconds)
printf '{"schema":"owner_successor_scale.detached_observation.v1","session":"%s","status":"live_process_observed_and_detached","observed":"%s","pane_pid":%d}\n' \
    "$SESSION" "$detached_at" "$pane_pid" > "$LAUNCH_ROOT/detached.json"
printf 'detached session=%s records=%s\n' "$SESSION" "$LAUNCH_ROOT"
