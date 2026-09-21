#!/usr/bin/env bash
set -u
cd /data/CoordExp/.worktrees/research-probes
root=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-corner-loop-bridge-factorial
CUDA_VISIBLE_DEVICES=0 python -m probes.training_set_completion.corner_loop_bridge_factorial C00 > "$root/C00.log" 2>&1
code=$?
echo "$code" > "$root/C00.exit"
if [ "$code" -ne 0 ]; then echo "$code" > "$root/run.exit"; exit "$code"; fi
pids=()
launch() {
 (CUDA_VISIBLE_DEVICES="$1" python -m probes.training_set_completion.corner_loop_bridge_factorial "$2" > "$root/$2.log" 2>&1
 code=$?
 echo "$code" > "$root/$2.exit"
 exit "$code") &
 pids+=("$!")
}
launch 1 C10
launch 2 C01
launch 3 C11
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "$status" > "$root/run.exit"
exit "$status"
