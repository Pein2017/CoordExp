#!/usr/bin/env bash
# Fixed one-round launch; native process waits, no polling or scientific retry.
set -uo pipefail
cd /data/CoordExp/.worktrees/dora-prox-linear-n2 || exit 1
RUN=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-row-boundary-continuous-feedback/pilot-v1
SCRIPT=/data/CoordExp/.worktrees/dora-prox-linear-n2/scripts/research/run_row_boundary_feedback_pilot.py
printf '%s\n' "$$" > "$RUN/launch/driver.pid"
sha256sum --check "$RUN/launch/runtime.sha256" || exit 1

run_step() {
    local name="$1" gpu="$2" seconds="$3" rc
    shift 3
    printf 'STEP_START name=%s gpu=%s utc=%s\n' "$name" "$gpu" "$(date -u +%FT%TZ)"
    timeout --signal=INT --kill-after=60s "${seconds}s" /usr/bin/env "CUDA_VISIBLE_DEVICES=$gpu" \
        conda run --no-capture-output -n ms python "$SCRIPT" "$@" > "$RUN/launch/$name.log" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$RUN/launch/$name.exit"
    if [ "$rc" -ne 0 ]; then
        printf 'STEP_FAILED name=%s exit=%s utc=%s\n' "$name" "$rc" "$(date -u +%FT%TZ)"
    else
        printf 'STEP_COMPLETE name=%s utc=%s\n' "$name" "$(date -u +%FT%TZ)"
    fi
    return "$rc"
}

train_chain() {
    local arm="$1" gpu="$2"
    run_step "$arm-train" "$gpu" 14340 train-arm --arm "$arm" --output-root "$RUN/$arm" &&
    run_step "$arm-rp110" "$gpu" 2340 evaluate --arm "$arm" --payload-root "$RUN/$arm/payload" --repetition-penalty 1.1 --output-root "$RUN/eval/$arm-rp110" &&
    run_step "$arm-rp100" "$gpu" 2340 evaluate --arm "$arm" --payload-root "$RUN/$arm/payload" --repetition-penalty 1.0 --output-root "$RUN/eval/$arm-rp100"
}

source_chain() {
    run_step source-rp110 2 2340 evaluate --arm source --repetition-penalty 1.1 --output-root "$RUN/eval/source-rp110" &&
    run_step source-rp100 2 2340 evaluate --arm source --repetition-penalty 1.0 --output-root "$RUN/eval/source-rp100"
}

train_chain control 0 & control_pid=$!
train_chain feedback 1 & feedback_pid=$!
source_chain & source_pid=$!
status=0
wait "$control_pid" || status=1
wait "$feedback_pid" || status=1
wait "$source_pid" || status=1
if [ "$status" -eq 0 ]; then
    conda run --no-capture-output -n ms python \
        research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-row-boundary-continuous-feedback/reduce_results.py \
        --eval-root "$RUN/eval" --output "$RUN/results.json" > "$RUN/launch/reduction.log" 2>&1 || status=1
fi
printf '%s\n' "$status" > "$RUN/launch/driver.exit"
printf 'FEEDBACK_PILOT_TERMINAL exit=%s utc=%s\n' "$status" "$(date -u +%FT%TZ)"
exit "$status"
