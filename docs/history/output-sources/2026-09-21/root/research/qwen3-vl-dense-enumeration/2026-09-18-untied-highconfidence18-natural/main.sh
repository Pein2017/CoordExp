#!/bin/bash
cd /data/CoordExp/.worktrees/research-probes
R=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural
pids=()
CUDA_VISIBLE_DEVICES=0 python -m probes.training_set_completion.untied_natural --condition tied-original --groups refined-00 refined-02 refined-04 fresh-02 fresh-04 fresh-06 fresh-08 fresh-10 fresh-12 fresh-14 fresh-16 fresh-18 fresh-20 fresh-22 fresh-24 fresh-26 fresh-28 fresh-30 > "$R/main-0.log" 2>&1 &
pids+=($!)
echo $! > "$R/main-0.pid"
CUDA_VISIBLE_DEVICES=1 python -m probes.training_set_completion.untied_natural --condition tied-original --groups refined-01 refined-03 fresh-01 fresh-03 fresh-05 fresh-07 fresh-09 fresh-11 fresh-13 fresh-15 fresh-17 fresh-19 fresh-21 fresh-23 fresh-25 fresh-27 fresh-29 fresh-31 > "$R/main-1.log" 2>&1 &
pids+=($!)
echo $! > "$R/main-1.pid"
CUDA_VISIBLE_DEVICES=2 python -m probes.training_set_completion.untied_natural --condition tied-normalized --groups refined-00 refined-01 refined-02 refined-03 refined-04 fresh-01 fresh-02 fresh-03 fresh-04 fresh-05 fresh-06 fresh-07 fresh-08 fresh-09 fresh-10 fresh-11 fresh-12 fresh-13 fresh-14 fresh-15 fresh-16 fresh-17 fresh-18 fresh-19 fresh-20 fresh-21 fresh-22 fresh-23 fresh-24 fresh-25 fresh-26 fresh-27 fresh-28 fresh-29 fresh-30 fresh-31 > "$R/main-2.log" 2>&1 &
pids+=($!)
echo $! > "$R/main-2.pid"
CUDA_VISIBLE_DEVICES=3 python -m probes.training_set_completion.untied_natural --condition untied-original --groups refined-00 refined-02 refined-04 fresh-02 fresh-04 fresh-06 fresh-08 fresh-10 fresh-12 fresh-14 fresh-16 fresh-18 fresh-20 fresh-22 fresh-24 fresh-26 fresh-28 fresh-30 > "$R/main-3.log" 2>&1 &
pids+=($!)
echo $! > "$R/main-3.pid"
CUDA_VISIBLE_DEVICES=4 python -m probes.training_set_completion.untied_natural --condition untied-original --groups refined-01 refined-03 fresh-01 fresh-03 fresh-05 fresh-07 fresh-09 fresh-11 fresh-13 fresh-15 fresh-17 fresh-19 fresh-21 fresh-23 fresh-25 fresh-27 fresh-29 fresh-31 > "$R/main-4.log" 2>&1 &
pids+=($!)
echo $! > "$R/main-4.pid"
CUDA_VISIBLE_DEVICES=5 python -m probes.training_set_completion.untied_natural --condition untied-normalized --groups refined-00 refined-01 refined-02 refined-03 refined-04 fresh-01 fresh-02 fresh-03 fresh-04 fresh-05 fresh-06 fresh-07 fresh-08 fresh-09 fresh-10 fresh-11 fresh-12 fresh-13 fresh-14 fresh-15 fresh-16 fresh-17 fresh-18 fresh-19 fresh-20 fresh-21 fresh-22 fresh-23 fresh-24 fresh-25 fresh-26 fresh-27 fresh-28 fresh-29 fresh-30 fresh-31 > "$R/main-5.log" 2>&1 &
pids+=($!)
echo $! > "$R/main-5.pid"
for i in 0 1 2 3 4 5; do wait ${pids[$i]}; echo $? > "$R/main-$i.exit"; done
date -u +%FT%TZ > "$R/main.settled"
