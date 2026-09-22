#!/usr/bin/env bash
set -u
cd /data/CoordExp/.worktrees/research-probes
RUN=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-successful-row-mechanism
pids=()
for pair in attention:2 mlp:3; do
 component=${pair%:*}; gpu=${pair#*:}; name=$component-S-to-F-layer20
 (
 date +%s > "$RUN/logs/$name.start"
 CUDA_VISIBLE_DEVICES=$gpu python -m probes.training_set_completion.successful_row_components --panel "$RUN/stage2/panels/F.json" --component "$component" --donor "$RUN/stage2/component-capture/components.pt" --output "$RUN/stage2/runtime/$name" > "$RUN/logs/$name.log" 2>&1 &
 producer=$!; echo "$producer" > "$RUN/logs/$name.pid"
 wait "$producer"; status=$?; echo "$status" > "$RUN/logs/$name.exit"; date +%s > "$RUN/logs/$name.end"
 exit "$status"
 ) & pids+=("$!")
done
status=0; for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "$status" > "$RUN/logs/components.exit"
tmux wait-for -S successful-row-components-done
exit "$status"
