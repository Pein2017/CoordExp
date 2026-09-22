#!/usr/bin/env bash
cd /data/CoordExp/.worktrees/research-probes
timeout --signal=TERM --kill-after=60s 86400s python -m probes.owner_successor_scale.paired_evaluation launch --packet /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput/evaluation/paired-preparation/packet-bound.json --arm A --gpus 0\,1\,2\,3\,4\,5\,6\,7 --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput/evaluation/paired-natural-A-v1 
