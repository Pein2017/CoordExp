#!/usr/bin/env bash
set -u
cd /data/CoordExp/.worktrees/research-probes
RUN=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-successful-row-mechanism
run_cell() {
 name=$1; gpu=$2; route=$3; donor=$4
 date +%s > "$RUN/logs/$name.start"
 CUDA_VISIBLE_DEVICES=$gpu python -m probes.training_set_completion.successful_row_state --panel "$RUN/stage2/panels/$route.json" --condition "$name" --donor-states "$RUN/stage2/runtime/native-$donor/states.pt" --output "$RUN/stage2/runtime/$name" > "$RUN/logs/$name.log" 2>&1 &
 producer=$!; echo "$producer" > "$RUN/logs/$name.pid"
 wait "$producer"; status=$?; echo "$status" > "$RUN/logs/$name.exit"; date +%s > "$RUN/logs/$name.end"
 return "$status"
}
pids=()
# The same-source native and S self-patch gates have passed. F self-control
# runs alongside independent patches; no causal conclusion is admitted until it passes.
(run_cell self-F 1 F F) & pids+=("$!")
(run_cell head-S-to-F 0 F S && run_cell head-F-to-S 0 S F) & pids+=("$!")
(run_cell residual-S-to-F-layer6 2 F S) & pids+=("$!")
(run_cell residual-F-to-S-layer6 3 S F) & pids+=("$!")
(run_cell residual-S-to-F-layer13 4 F S) & pids+=("$!")
(run_cell residual-F-to-S-layer13 5 S F) & pids+=("$!")
(run_cell residual-S-to-F-layer20 6 F S) & pids+=("$!")
(run_cell residual-F-to-S-layer20 7 S F) & pids+=("$!")
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "$status" > "$RUN/logs/patches.exit"
tmux wait-for -S successful-row-patches-done
exit "$status"
