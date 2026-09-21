#!/usr/bin/env bash
set -euo pipefail

test -f /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-label-vs-compilation/diagnostic/execution-packet-v2.json
test -f /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-label-vs-compilation/diagnostic/root-launch-grant-v1.json
test ! -e /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-label-vs-compilation/diagnostic/execution-v1
test ! -e /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-label-vs-compilation/diagnostic/execution-v1.launcher.log
! tmux has-session -t label-vs-compilation-diag-v1 2>/dev/null

tmux new-session -d -s label-vs-compilation-diag-v1 \
  "bash -lc 'set -euo pipefail; set -o noclobber; cd /data/CoordExp/.worktrees/research-probes; exec env PYTHONPATH=. python research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-14-label-vs-compilation/diagnostic/runner.py launch --packet /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-label-vs-compilation/diagnostic/execution-packet-v2.json --grant /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-label-vs-compilation/diagnostic/root-launch-grant-v1.json --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-label-vs-compilation/diagnostic/execution-v1 > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-label-vs-compilation/diagnostic/execution-v1.launcher.log 2>&1'"
