#!/usr/bin/env bash
set -u
set -o pipefail
BASE=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source
cd /data/CoordExp/.worktrees/research-probes
echo "START tied-train-269858-failure"
python3 -m probes.training_set_completion.recurrence_spatial.producer --model tied --device cuda:2 --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-train-269858-failure.json --result-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/runtime/tied-train-269858-failure.json > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/logs/tied-train-269858-failure-rerun3.log 2>&1
rc=$?
echo "END tied-train-269858-failure rc=$rc"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
echo "START untied-train-269858-failure"
python3 -m probes.training_set_completion.recurrence_spatial.producer --model untied --device cuda:2 --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/untied-train-269858-failure.json --result-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/runtime/untied-train-269858-failure.json > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/logs/untied-train-269858-failure-rerun3.log 2>&1
rc=$?
echo "END untied-train-269858-failure rc=$rc"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
echo "START tied-train-523815-healthy"
python3 -m probes.training_set_completion.recurrence_spatial.producer --model tied --device cuda:2 --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-train-523815-healthy.json --result-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/runtime/tied-train-523815-healthy.json > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/logs/tied-train-523815-healthy-rerun3.log 2>&1
rc=$?
echo "END tied-train-523815-healthy rc=$rc"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
exit 0
