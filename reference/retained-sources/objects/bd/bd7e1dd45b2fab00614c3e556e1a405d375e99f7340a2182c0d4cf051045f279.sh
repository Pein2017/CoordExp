#!/usr/bin/env bash
set -u
cd /data/CoordExp/.worktrees/research-probes
RUN="/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-wrapper-history-factorial"
PREV="/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-history-rereading-mechanism"
pids=()
for spec in native-F:0 self-F:1; do
 IFS=: read -r name gpu <<< "$spec"
 args=(); if [[ "$name" == self-F ]]; then args=(--donor "$PREV/runtime/extract-F/capture.pt" --residual --cache-patch --wrapper-patch); fi
 (
 date +%s > "$RUN/logs/$name.start"
 CUDA_VISIBLE_DEVICES=$gpu python -m probes.training_set_completion.wrapper_history --panel "$RUN/panels/F.json" --output "$RUN/runtime/$name" "${args[@]}" > "$RUN/logs/$name.log" 2>&1 &
 p=$!; echo "$p" > "$RUN/logs/$name.pid";wait "$p";s=$?;echo "$s" > "$RUN/logs/$name.exit";date +%s > "$RUN/logs/$name.end";exit "$s"
 ) & pids+=("$!")
done
s=0;for p in "${pids[@]}";do wait "$p" || s=1;done
echo "$s" > "$RUN/logs/admission.exit"
tmux wait-for -S wrapper-admission-done
exit "$s"
