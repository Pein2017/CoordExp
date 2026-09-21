#!/usr/bin/env bash
set -u
set -o pipefail
BASE=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source
cd /data/CoordExp/.worktrees/research-probes
echo "START untied-train-301827-failure"
python3 -m probes.training_set_completion.recurrence_spatial.producer --model untied --device cuda:5 --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/untied-train-301827-failure.json --result-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/runtime/untied-train-301827-failure.json > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/logs/untied-train-301827-failure-rerun3.log 2>&1
rc=$?
echo "END untied-train-301827-failure rc=$rc"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
echo "START tied-train-477785-healthy"
python3 -m probes.training_set_completion.recurrence_spatial.producer --model tied --device cuda:5 --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-train-477785-healthy.json --result-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/runtime/tied-train-477785-healthy.json > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/logs/tied-train-477785-healthy-rerun3.log 2>&1
rc=$?
echo "END tied-train-477785-healthy rc=$rc"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
echo "START tied-train-131490-healthy"
python3 -m probes.training_set_completion.recurrence_spatial.producer --model tied --device cuda:5 --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-train-131490-healthy.json --result-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/runtime/tied-train-131490-healthy.json > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/logs/tied-train-131490-healthy-rerun3.log 2>&1
rc=$?
echo "END tied-train-131490-healthy rc=$rc"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
exit 0
