#!/bin/bash
cd /data/CoordExp/.worktrees/research-probes
R=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-native-coordinate-branch-completion
CUDA_VISIBLE_DEVICES=0 python -m probes.training_set_completion.native_coordinate_branch --root "$R" --coordinate 0 > "$R/branch0.log" 2>&1
s=$?
echo "$s" > "$R/branch0.exit"
if [ "$s" != 0 ]; then exit "$s"; fi
CUDA_VISIBLE_DEVICES=1 python -m probes.training_set_completion.native_coordinate_branch --root "$R" --coordinate 52 > "$R/branch52.log" 2>&1 &
a=$!
CUDA_VISIBLE_DEVICES=2 python -m probes.training_set_completion.native_coordinate_branch --root "$R" --coordinate 30 > "$R/branch30.log" 2>&1 &
b=$!
wait "$a"; x=$?; echo "$x" > "$R/branch52.exit"
wait "$b"; y=$?; echo "$y" > "$R/branch30.exit"
echo "$x $y" > "$R/settled"
