#!/usr/bin/env bash
set -uo pipefail
BASE="/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-first-coordinate-mediation"
WORKTREE="/data/CoordExp/.worktrees/research-probes"
GPU="5"
OUT="$BASE/runtime/scaleout-gpu5-untied"
LAUNCH_DIR="$BASE/runtime/launch/scaleout-gpu5-untied"
LOG="$LAUNCH_DIR/producer.log"
LIVE="$LAUNCH_DIR/live.json"
EXIT="$LAUNCH_DIR/exit-status.json"
mkdir -p "$LAUNCH_DIR"
python3 - "$LIVE" "$BASHPID" <<'PYLIVE'
import json, os, sys, time, pathlib
path=pathlib.Path(sys.argv[1])
path.write_text(json.dumps({"schema":"first_coordinate_mediation.live.v1","wrapper_shell_pid":int(sys.argv[2]),"gpu":5,"model":"untied","state":"started","started_at_unix":time.time()},indent=2)+"\n")
PYLIVE
CUDA_VISIBLE_DEVICES="$GPU" python3 "$WORKTREE/probes/training_set_completion/first_coordinate/runtime.py" run \
  --panel "$BASE/../2026-09-18-numerical-recurrence-feedback/panel.json" \
  --plan "$BASE/execution-plan.json" \
  --output "$OUT" \
  --device cuda:0 \
  --max-forwards 3000 \
  --max-seconds 1500 \
  --max-bytes 786432000 \
  --cell-id untied-417044-failure--sign19--x1-1 \
  --cell-id untied-417044-failure--sign22--x1-1 \
  --cell-id untied-417044-failure--sign24--x1-1 \
  --cell-id untied-417044-failure--sign26--x1-1 \
  >"$LOG" 2>&1 &
CHILD=$!
python3 - "$LIVE" "$CHILD" <<'PYLIVE2'
import json, sys, time, pathlib
path=pathlib.Path(sys.argv[1]); data=json.loads(path.read_text())
data.update({"runtime_pid":int(sys.argv[2]),"runtime_command":"CUDA_VISIBLE_DEVICES=5 python3 first_coordinate/runtime.py run scaleout-gpu5-untied","state":"running","runtime_started_at_unix":time.time()})
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
out.write_text(json.dumps({"schema":"first_coordinate_mediation.exit_status.v1","wrapper_shell_pid":data.get("wrapper_shell_pid"),"runtime_pid":data.get("runtime_pid"),"gpu":data.get("gpu"),"model":data.get("model"),"exit_code":rc,"exit_status_observed":True,"state":"ended","live":str(live)},indent=2)+"\n")
PYEXIT
exit "$RC"
