#!/bin/bash
cd /data/CoordExp/.worktrees/research-probes || exit 90
(
CUDA_VISIBLE_DEVICES=0 python -m probes.training_set_completion.repetition_history_runtime --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/B/309264-pre_entry_one_replay.json --case 309264 --mode prefix --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/B/pre_entry_one_replay > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-pre_entry_one_replay.log 2>&1
rc=$?
printf '%s\n' "$rc" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-pre_entry_one_replay.exit
exit "$rc"
) &
printf '%s\n' "$!" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-pre_entry_one_replay.pid
(
CUDA_VISIBLE_DEVICES=1 python -m probes.training_set_completion.repetition_history_runtime --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/B/309264-established_one_replay.json --case 309264 --mode prefix --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/B/established_one_replay > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-established_one_replay.log 2>&1
rc=$?
printf '%s\n' "$rc" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-established_one_replay.exit
exit "$rc"
) &
printf '%s\n' "$!" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-established_one_replay.pid
(
CUDA_VISIBLE_DEVICES=2 python -m probes.training_set_completion.repetition_history_runtime --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/panels/B/309264-established_four_replay.json --case 309264 --mode prefix --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/runtime/B/established_four_replay > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-established_four_replay.log 2>&1
rc=$?
printf '%s\n' "$rc" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-established_four_replay.exit
exit "$rc"
) &
printf '%s\n' "$!" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/B-309264-established_four_replay.pid
wait
touch /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/bird-replay.settled
tmux wait-for -S repetition-history-bird-replay
