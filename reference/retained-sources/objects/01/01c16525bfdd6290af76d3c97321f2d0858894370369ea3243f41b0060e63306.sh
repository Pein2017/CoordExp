#!/usr/bin/env bash
set -u
cd /data/CoordExp/.worktrees/research-probes
RUN=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-history-rereading-mechanism
pids=()
for spec in extract-S:S:extract:0 extract-F:F:extract:1 native-F:F:full:2; do
 IFS=: read -r name route mode gpu <<< "$spec"
 (
 date +%s > "$RUN/logs/$name.start"
 CUDA_VISIBLE_DEVICES=$gpu python -m probes.training_set_completion.history_rereading --panel "$RUN/panels/$route.json" --mode "$mode" --output "$RUN/runtime/$name" > "$RUN/logs/$name.log" 2>&1 &
 p=$!; echo "$p" > "$RUN/logs/$name.pid";wait "$p";s=$?;echo "$s" > "$RUN/logs/$name.exit";date +%s > "$RUN/logs/$name.end";exit "$s"
 ) & pids+=("$!")
done
s=0;for p in "${pids[@]}";do wait "$p" || s=1;done
echo "$s" > "$RUN/logs/admission.exit"
tmux wait-for -S rereading-admission-done
exit "$s"
