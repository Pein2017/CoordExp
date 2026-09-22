#!/usr/bin/env bash
set -euo pipefail
retry_root=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/stage03-single-owner-repair-v1-retry1
mkdir -p "$retry_root/logs"
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  printf 'launcher_pid=%s\n' "$$"
  command -v python
  python -c 'import sys; print(sys.executable)'
  printf 'controller_pid=%s\n' "$$"
} > "$retry_root/phase.log"
exec python -m probes.training_set_completion.repair controller --manifest "$retry_root/manifest.json" --output "$retry_root" >> "$retry_root/logs/controller.log" 2>&1
