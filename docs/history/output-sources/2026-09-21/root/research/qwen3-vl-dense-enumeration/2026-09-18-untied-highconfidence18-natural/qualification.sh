#!/bin/bash
cd /data/CoordExp/.worktrees/research-probes
R=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural
conditions=(tied-original tied-normalized untied-original untied-normalized)
pids=()
for i in 0 1 2 3; do
 CUDA_VISIBLE_DEVICES=$i python -m probes.training_set_completion.untied_natural --condition ${conditions[$i]} --groups fresh-00 --qualify > "$R/qual-${conditions[$i]}.log" 2>&1 &
 pids+=($!)
done
for i in 0 1 2 3; do wait ${pids[$i]}; echo $? > "$R/qual-${conditions[$i]}.exit"; done
