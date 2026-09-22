#!/bin/bash
cd /data/CoordExp/.worktrees/research-probes || exit 90
(
CUDA_VISIBLE_DEVICES=0 python -m probes.training_set_completion.repetition_history_runtime --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/B/309264-full_norm.json --case 309264 --mode sustained --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/B/full_norm > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-full_norm.log 2>&1
rc=$?
printf '%s\n' "$rc" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-full_norm.exit
exit "$rc"
) &
printf '%s\n' "$!" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-full_norm.pid
(
CUDA_VISIBLE_DEVICES=1 python -m probes.training_set_completion.repetition_history_runtime --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/B/309264-pre_entry_one.json --case 309264 --mode pulse1 --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/B/pre_entry_one > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-pre_entry_one.log 2>&1
rc=$?
printf '%s\n' "$rc" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-pre_entry_one.exit
exit "$rc"
) &
printf '%s\n' "$!" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-pre_entry_one.pid
(
CUDA_VISIBLE_DEVICES=2 python -m probes.training_set_completion.repetition_history_runtime --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/B/309264-established_one.json --case 309264 --mode pulse1 --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/B/established_one > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-established_one.log 2>&1
rc=$?
printf '%s\n' "$rc" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-established_one.exit
exit "$rc"
) &
printf '%s\n' "$!" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-established_one.pid
(
CUDA_VISIBLE_DEVICES=3 python -m probes.training_set_completion.repetition_history_runtime --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/B/309264-established_four.json --case 309264 --mode pulse4 --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/B/established_four > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-established_four.log 2>&1
rc=$?
printf '%s\n' "$rc" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-established_four.exit
exit "$rc"
) &
printf '%s\n' "$!" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-established_four.pid
(
CUDA_VISIBLE_DEVICES=4 python -m probes.training_set_completion.repetition_history_runtime --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/B/309264-late_sustained.json --case 309264 --mode sustained --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/B/late_sustained > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-late_sustained.log 2>&1
rc=$?
printf '%s\n' "$rc" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-late_sustained.exit
exit "$rc"
) &
printf '%s\n' "$!" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-late_sustained.pid
(
CUDA_VISIBLE_DEVICES=5 python -m probes.training_set_completion.repetition_history_runtime --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/B/386313-pre_entry_one_replay.json --case 386313 --mode prefix --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/B/pre_entry_one_replay > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-386313-pre_entry_one_replay.log 2>&1
rc=$?
printf '%s\n' "$rc" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-386313-pre_entry_one_replay.exit
exit "$rc"
) &
printf '%s\n' "$!" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-386313-pre_entry_one_replay.pid
(
CUDA_VISIBLE_DEVICES=6 python -m probes.training_set_completion.repetition_history_runtime --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/B/386313-established_one_replay.json --case 386313 --mode prefix --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/B/established_one_replay > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-386313-established_one_replay.log 2>&1
rc=$?
printf '%s\n' "$rc" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-386313-established_one_replay.exit
exit "$rc"
) &
printf '%s\n' "$!" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-386313-established_one_replay.pid
(
CUDA_VISIBLE_DEVICES=7 python -m probes.training_set_completion.repetition_history_runtime --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/B/386313-established_four_replay.json --case 386313 --mode prefix --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/B/established_four_replay > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-386313-established_four_replay.log 2>&1
rc=$?
printf '%s\n' "$rc" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-386313-established_four_replay.exit
exit "$rc"
) &
printf '%s\n' "$!" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-386313-established_four_replay.pid
wait
touch /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/wave2.settled
tmux wait-for -S repetition-history-wave2
