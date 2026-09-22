#!/usr/bin/env bash
set -u
cd /data/CoordExp/.worktrees/research-probes
RUN=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-history-rereading-mechanism
pids=()
for spec in self-F:F:3 residual-only:S:4 cache-only:S:5 joint:S:6;do
 IFS=: read -r name donor gpu <<< "$spec"
 flags=();case "$name" in self-F|joint)flags=(--residual --cache-patch);; residual-only)flags=(--residual);; cache-only)flags=(--cache-patch);;esac
 (
 date +%s > "$RUN/logs/$name.start"
 CUDA_VISIBLE_DEVICES=$gpu python -m probes.training_set_completion.history_rereading --panel "$RUN/panels/F.json" --donor "$RUN/runtime/extract-$donor/capture.pt" "${flags[@]}" --output "$RUN/runtime/$name" > "$RUN/logs/$name.log" 2>&1 &
 p=$!;echo "$p" > "$RUN/logs/$name.pid";wait "$p";s=$?;echo "$s" > "$RUN/logs/$name.exit";date +%s > "$RUN/logs/$name.end";exit "$s"
 ) & pids+=("$!")
done
s=0;for p in "${pids[@]}";do wait "$p" || s=1;done
echo "$s" > "$RUN/logs/factorial.exit";tmux wait-for -S rereading-factorial-done;exit "$s"
