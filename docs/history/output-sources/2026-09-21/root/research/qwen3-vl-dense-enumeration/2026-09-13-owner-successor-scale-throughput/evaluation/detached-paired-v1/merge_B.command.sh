#!/usr/bin/env bash
cd /data/CoordExp/.worktrees/research-probes
python -m probes.owner_successor_scale.paired_evaluation merge --packet /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput/evaluation/paired-preparation/packet-bound.json --arm B --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput/evaluation/paired-natural-B-v1 
