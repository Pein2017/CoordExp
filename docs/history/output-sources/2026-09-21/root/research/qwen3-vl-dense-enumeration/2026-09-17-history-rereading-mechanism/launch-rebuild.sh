#!/usr/bin/env bash
set -u
cd /data/CoordExp/.worktrees/research-probes
RUN=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-history-rereading-mechanism
name=rebuild-cache-only
date +%s > "$RUN/logs/$name.start"
CUDA_VISIBLE_DEVICES=0 python -m probes.training_set_completion.history_rereading --panel "$RUN/panels/F.json" --rebuild "$RUN/runtime/cache-only/raw.json" --output "$RUN/runtime/$name" > "$RUN/logs/$name.log" 2>&1 &
p=$!;echo "$p" > "$RUN/logs/$name.pid";wait "$p";s=$?;echo "$s" > "$RUN/logs/$name.exit";date +%s > "$RUN/logs/$name.end"
tmux wait-for -S rereading-rebuild-done;exit "$s"
