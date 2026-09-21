#!/usr/bin/env bash
set -u
set -o pipefail
BASE=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source
cd /data/CoordExp/.worktrees/research-probes
echo "START tied-train-322768-failure"
python3 -m probes.training_set_completion.recurrence_spatial.producer --model tied --device cuda:0 --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-train-322768-failure.json --result-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/runtime/tied-train-322768-failure.json > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/logs/tied-train-322768-failure-rerun3.log 2>&1
rc=$?
echo "END tied-train-322768-failure rc=$rc"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
echo "START untied-train-196924-failure"
python3 -m probes.training_set_completion.recurrence_spatial.producer --model untied --device cuda:0 --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/untied-train-196924-failure.json --result-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/runtime/untied-train-196924-failure.json > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/logs/untied-train-196924-failure-rerun3.log 2>&1
rc=$?
echo "END untied-train-196924-failure rc=$rc"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
echo "START tied-train-328462-healthy"
python3 -m probes.training_set_completion.recurrence_spatial.producer --model tied --device cuda:0 --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-train-328462-healthy.json --result-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/runtime/tied-train-328462-healthy.json > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/logs/tied-train-328462-healthy-rerun3.log 2>&1
rc=$?
echo "END tied-train-328462-healthy rc=$rc"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
echo "START tied-train-354063-healthy"
python3 -m probes.training_set_completion.recurrence_spatial.producer --model tied --device cuda:0 --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-train-354063-healthy.json --result-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/runtime/tied-train-354063-healthy.json > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/logs/tied-train-354063-healthy-rerun3.log 2>&1
rc=$?
echo "END tied-train-354063-healthy rc=$rc"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
exit 0
