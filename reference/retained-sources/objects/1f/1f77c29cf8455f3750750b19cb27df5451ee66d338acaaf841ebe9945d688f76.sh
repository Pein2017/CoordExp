#!/usr/bin/env bash
set -uo pipefail
BASE="/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-first-coordinate-mediation"
WORKTREE="/data/CoordExp/.worktrees/research-probes"
OUT="$BASE/runtime/qualification/gpu0-untied"
LOG="$BASE/runtime/qualification/producer.log"
LIVE="$BASE/runtime/qualification/live.json"
EXIT="$BASE/runtime/qualification/exit-status.json"
mkdir -p "$BASE/runtime/qualification"
python3 - "$LIVE" <<'PYLIVE'
import json, os, time, pathlib
path=pathlib.Path(__import__('sys').argv[1])
path.write_text(json.dumps({"schema":"first_coordinate_mediation.live.v1","wrapper_pid":os.getpid(),"state":"started","started_at_unix":time.time()},indent=2)+"\n")
PYLIVE
CUDA_VISIBLE_DEVICES=0 python3 "$WORKTREE/probes/training_set_completion/first_coordinate/runtime.py" run \
  --panel "$BASE/../2026-09-18-numerical-recurrence-feedback/panel.json" \
  --plan "$BASE/execution-plan.json" \
  --output "$OUT" \
  --device cuda:0 \
  --max-forwards 2500 \
  --max-seconds 900 \
  --max-bytes 1073741824 \
  --cell-id untied-417044-failure--original--x1-0 \
  --cell-id untied-417044-failure--original--x1-1 \
  --cell-id untied-417044-failure--sign20--x1-0 \
  --cell-id untied-417044-failure--sign20--x1-1 \
  >"$LOG" 2>&1 &
CHILD=$!
python3 - "$LIVE" "$CHILD" <<'PYLIVE2'
import json, os, sys, time, pathlib
path=pathlib.Path(sys.argv[1])
data=json.loads(path.read_text())
data.update({"runtime_pid":int(sys.argv[2]),"runtime_command":"CUDA_VISIBLE_DEVICES=0 python3 first_coordinate/runtime.py run qualification-four-cells","state":"running","runtime_started_at_unix":time.time()})
path.write_text(json.dumps(data,indent=2)+"\n")
PYLIVE2
wait "$CHILD"
RC=$?
python3 - "$LIVE" "$EXIT" "$RC" <<'PYEXIT'
import json, os, sys, time, pathlib
live=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); rc=int(sys.argv[3])
data=json.loads(live.read_text())
data.update({"state":"ended","ended_at_unix":time.time(),"wait_status_observed":True,"exit_code":rc})
live.write_text(json.dumps(data,indent=2)+"\n")
out.write_text(json.dumps({"schema":"first_coordinate_mediation.exit_status.v1","wrapper_pid":os.getpid(),"runtime_pid":data.get("runtime_pid"),"exit_code":rc,"exit_status_observed":True,"state":"ended","live":str(live)},indent=2)+"\n")
PYEXIT
exit "$RC"
