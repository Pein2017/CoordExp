#!/bin/bash
cd /data/CoordExp/.worktrees/research-probes || exit 90
(
CUDA_VISIBLE_DEVICES=0 python -m probes.training_set_completion.repetition_history_runtime --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/B/309264-native.json --case 309264 --mode prefix --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/B/native > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-native.log 2>&1
rc=$?
printf '%s\n' "$rc" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-native.exit
exit "$rc"
) &
printf '%s\n' "$!" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-native.pid
(
CUDA_VISIBLE_DEVICES=1 python -m probes.training_set_completion.repetition_history_runtime --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/A/309264-AA.json --case 309264 --mode prefix --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/A/AA > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/A-309264-AA.log 2>&1
rc=$?
printf '%s\n' "$rc" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/A-309264-AA.exit
exit "$rc"
) &
printf '%s\n' "$!" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/A-309264-AA.pid
(
CUDA_VISIBLE_DEVICES=2 python -m probes.training_set_completion.repetition_history_runtime --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/A/309264-BB.json --case 309264 --mode prefix --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/A/BB > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/A-309264-BB.log 2>&1
rc=$?
printf '%s\n' "$rc" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/A-309264-BB.exit
exit "$rc"
) &
printf '%s\n' "$!" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/A-309264-BB.pid
(
CUDA_VISIBLE_DEVICES=3 python -m probes.training_set_completion.repetition_history_runtime --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/B/386313-native.json --case 386313 --mode prefix --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/B/native > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-386313-native.log 2>&1
rc=$?
printf '%s\n' "$rc" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-386313-native.exit
exit "$rc"
) &
printf '%s\n' "$!" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-386313-native.pid
wait
touch /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/qualification.settled
tmux wait-for -S repetition-history-qualification
