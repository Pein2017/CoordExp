#!/usr/bin/env bash
exec >> /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/second-fit-readback-recovery-v1/phase.log 2>&1
set -euo pipefail
export CONDA_PREFIX=/root/miniconda3/envs/ms
export PATH=/root/miniconda3/envs/ms/bin:/data/CoordExp/.codex/bin:$PATH
out=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/second-fit-readback-recovery-v1
printf 'started_utc=%s shell_pid=%s python=%s conda_prefix=%s\n' "$(date -u +%FT%TZ)" "$$" "$(command -v python)" "$CONDA_PREFIX"
python - "$out/controller-start.json" <<'PY'
import json, os, sys, time
from pathlib import Path
path = Path(sys.argv[1])
value = {"schema": "training_set_completion.second_fit_readback_recovery.v1.controller_start", "status": "running", "pid": os.getpid(), "python": sys.executable, "started_unix": time.time()}
tmp = path.with_suffix('.tmp')
with tmp.open('x') as stream:
    stream.write(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n')
    stream.flush()
    os.fsync(stream.fileno())
os.replace(tmp, path)
PY
cd /data/CoordExp/.worktrees/research-probes
exec python -m probes.training_set_completion.recover_readback controller --manifest "$out/manifest.json" --output "$out"
