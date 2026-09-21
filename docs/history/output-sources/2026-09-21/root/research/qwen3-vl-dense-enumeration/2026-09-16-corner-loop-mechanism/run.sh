#!/usr/bin/env bash
set -u
cd /data/CoordExp/.worktrees/research-probes
root=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-corner-loop-mechanism
pids=()
launch() {
 local gpu=$1 label=$2 image=$3
 (CUDA_VISIBLE_DEVICES="$gpu" python -m probes.training_set_completion.corner_loop_phase1 "$label" "$image" > "$root/$label-$image.log" 2>&1
 code=$?
 echo "$code" > "$root/$label-$image.exit"
 exit "$code") &
 pids+=("$!")
}
launch 0 Bnormalized64 477415
launch 1 P16 477415
launch 2 R16 477415
launch 3 P16 351017
launch 4 R16 351017
launch 5 Bnormalized64 417044
launch 6 P16 417044
launch 7 R16 417044
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "$status" > "$root/run.exit"
exit "$status"
