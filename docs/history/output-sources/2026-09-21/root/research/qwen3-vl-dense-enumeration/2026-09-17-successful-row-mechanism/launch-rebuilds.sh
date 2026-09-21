#!/usr/bin/env bash
set -u
cd /data/CoordExp/.worktrees/research-probes
RUN=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-successful-row-mechanism
pids=()
for tuple in rebuild-S-to-F:0:residual-S-to-F-layer20 rebuild-control-F:1:native-F; do
 IFS=: read -r name gpu source <<< "$tuple"
 (
 date +%s > "$RUN/logs/$name.start"
 CUDA_VISIBLE_DEVICES=$gpu python -m probes.training_set_completion.successful_row_state --panel "$RUN/stage2/panels/F.json" --condition rebuild-S-to-F --rebuild-from "$RUN/stage2/runtime/$source/raw.json" --output "$RUN/stage2/runtime/$name" > "$RUN/logs/$name.log" 2>&1 &
 producer=$!; echo "$producer" > "$RUN/logs/$name.pid"
 wait "$producer"; status=$?; echo "$status" > "$RUN/logs/$name.exit"; date +%s > "$RUN/logs/$name.end"
 exit "$status"
 ) & pids+=("$!")
done
status=0; for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "$status" > "$RUN/logs/rebuilds.exit"
tmux wait-for -S successful-row-rebuilds-done
exit "$status"
