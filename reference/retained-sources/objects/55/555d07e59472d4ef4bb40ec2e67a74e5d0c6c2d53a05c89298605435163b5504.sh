#!/bin/bash
failed=0
pids=()
bash /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/run-sampling-gpu0.sh &
pids+=("$!")
bash /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/run-sampling-gpu1.sh &
pids+=("$!")
bash /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/run-sampling-gpu2.sh &
pids+=("$!")
bash /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/run-sampling-gpu3.sh &
pids+=("$!")
bash /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/run-sampling-gpu4.sh &
pids+=("$!")
bash /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/run-sampling-gpu5.sh &
pids+=("$!")
bash /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/run-sampling-gpu6.sh &
pids+=("$!")
bash /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/run-sampling-gpu7.sh &
pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
echo "$failed" > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/sampling-rest.exit
touch /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism/logs/sampling-rest.settled
tmux wait-for -S repetition-history-sampling-rest
exit "$failed"
