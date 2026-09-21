#!/usr/bin/env bash
set -uo pipefail
LAUNCH='/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout/runtime/launch/qualification-gpu0'
WORKTREE='/data/CoordExp/.worktrees/research-probes'
OUT='/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout/runtime/qualification/gpu0-untied'
LOG='/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout/runtime/launch/qualification-gpu0/producer.log'
LIVE='/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout/runtime/launch/qualification-gpu0/live.json'
EXIT='/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout/runtime/launch/qualification-gpu0/exit-status.json'
python3 - "$LIVE" "$OUT" "$LOG" <<'PYLIVE'
import json, os, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
data = {
    "schema": "coordinate_pair.qualification_live.v1",
    "state": "started",
    "wrapper_pid": os.getpid(),
    "output": sys.argv[2],
    "log": sys.argv[3],
    "started_at_unix": time.time(),
}
path.write_text(json.dumps(data, indent=2) + "\n")
PYLIVE
CUDA_VISIBLE_DEVICES=0 python3 "$WORKTREE/probes/training_set_completion/coordinate_pair/runtime.py" run \
  --panel "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback/panel.json" \
  --plan "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout/execution-plan.json" \
  --output "$OUT" \
  --device cuda:0 \
  --max-forwards 3500 \
  --max-seconds 900 \
  --max-bytes 1073741824 \
  --cell-id untied-417044-failure--original \
  --cell-id untied-417044-failure--sign23 \
  --cell-id untied-417044-failure--only0 \
  --cell-id untied-417044-failure--only1 \
  --cell-id untied-417044-failure--pair01 \
  --cell-id untied-417044-failure--except01 \
  >"$LOG" 2>&1
RC=$?
python3 - "$LIVE" "$EXIT" "$RC" <<'PYEXIT'
import json, pathlib, sys, time
live = pathlib.Path(sys.argv[1])
exit_path = pathlib.Path(sys.argv[2])
rc = int(sys.argv[3])
data = json.loads(live.read_text()) if live.exists() else {}
data.update({"schema": "coordinate_pair.qualification_live.v1", "state": "ended", "ended_at_unix": time.time(), "wait_status_observed": True, "exit_code": rc})
live.write_text(json.dumps(data, indent=2) + "\n")
exit_path.write_text(json.dumps({
    "schema": "coordinate_pair.qualification_exit_status.v1",
    "wrapper_pid": data.get("wrapper_pid"),
    "runtime_pid": data.get("runtime_pid"),
    "exit_code": rc,
    "exit_status_observed": True,
    "state": "ended",
    "live": str(live),
    "log": "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout/runtime/launch/qualification-gpu0/producer.log",
}, indent=2) + "\n")
PYEXIT
exit "$RC"
