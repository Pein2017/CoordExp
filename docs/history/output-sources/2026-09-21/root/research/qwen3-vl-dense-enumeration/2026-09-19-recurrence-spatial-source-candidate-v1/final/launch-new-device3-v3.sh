#!/usr/bin/env bash
set -u
set -o pipefail
BASE=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source
cd /data/CoordExp/.worktrees/research-probes
echo "START untied-train-169872-failure"
python3 -m probes.training_set_completion.recurrence_spatial.producer --model untied --device cuda:3 --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/untied-train-169872-failure.json --result-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/runtime/untied-train-169872-failure.json > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/logs/untied-train-169872-failure-rerun3.log 2>&1
rc=$?
echo "END untied-train-169872-failure rc=$rc"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
echo "START tied-train-185502-healthy"
python3 -m probes.training_set_completion.recurrence_spatial.producer --model tied --device cuda:3 --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-train-185502-healthy.json --result-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/runtime/tied-train-185502-healthy.json > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/logs/tied-train-185502-healthy-rerun3.log 2>&1
rc=$?
echo "END tied-train-185502-healthy rc=$rc"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
echo "START tied-train-59540-healthy"
python3 -m probes.training_set_completion.recurrence_spatial.producer --model tied --device cuda:3 --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-train-59540-healthy.json --result-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/runtime/tied-train-59540-healthy.json > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/logs/tied-train-59540-healthy-rerun3.log 2>&1
rc=$?
echo "END tied-train-59540-healthy rc=$rc"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
exit 0
