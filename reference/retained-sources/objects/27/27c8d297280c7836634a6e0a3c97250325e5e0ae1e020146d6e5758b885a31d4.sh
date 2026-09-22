#!/usr/bin/env bash
set -uo pipefail
WORKTREE=/data/CoordExp/.worktrees/research-probes
OUT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout/runtime/scaleout-gpu0-tied
LOG=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout/runtime/launch/scaleout-gpu0-tied/producer.log
LIVE=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout/runtime/launch/scaleout-gpu0-tied/live.json
EXIT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout/runtime/launch/scaleout-gpu0-tied/exit-status.json
python3 - "$LIVE" "$OUT" "$LOG" <<'PYLIVE'
import json, os, pathlib, sys, time
path=pathlib.Path(sys.argv[1])
path.write_text(json.dumps({"schema":"coordinate_pair.scaleout_live.v1","state":"started","wrapper_pid":os.getpid(),"output":sys.argv[2],"log":sys.argv[3],"started_at_unix":time.time()},indent=2)+"\n")
PYLIVE
CUDA_VISIBLE_DEVICES=0 python3 /data/CoordExp/.worktrees/research-probes/probes/training_set_completion/coordinate_pair/runtime.py run --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback/panel.json --plan /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout/execution-plan.json --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout/runtime/scaleout-gpu0-tied --device cuda:0 --max-forwards 5500 --max-seconds 1400 --max-bytes 1288490188 --cell-id tied-885-failure--original --cell-id tied-885-failure--pair01 --cell-id tied-5586-failure--only0 --cell-id tied-14038-failure--original --cell-id tied-14038-failure--pair01 --cell-id tied-632-failure--only0 --cell-id tied-417044-failure--original --cell-id tied-417044-failure--pair01 >"$LOG" 2>&1 &
CHILD=$!
python3 - "$LIVE" "$CHILD" <<'PYRUNNING'
import json, pathlib, sys, time
path=pathlib.Path(sys.argv[1]); data=json.loads(path.read_text())
data.update({"state":"running","runtime_pid":int(sys.argv[2]),"runtime_started_at_unix":time.time(),"runtime_command":"python3 /data/CoordExp/.worktrees/research-probes/probes/training_set_completion/coordinate_pair/runtime.py run --panel /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback/panel.json --plan /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout/execution-plan.json --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout/runtime/scaleout-gpu0-tied --device cuda:0 --max-forwards 5500 --max-seconds 1400 --max-bytes 1288490188 --cell-id tied-885-failure--original --cell-id tied-885-failure--pair01 --cell-id tied-5586-failure--only0 --cell-id tied-14038-failure--original --cell-id tied-14038-failure--pair01 --cell-id tied-632-failure--only0 --cell-id tied-417044-failure--original --cell-id tied-417044-failure--pair01"})
path.write_text(json.dumps(data,indent=2)+"\n")
PYRUNNING
wait "$CHILD"
RC=$?
python3 - "$LIVE" "$EXIT" "$RC" <<'PYEXIT'
import json, pathlib, sys, time
live=pathlib.Path(sys.argv[1]); exitp=pathlib.Path(sys.argv[2]); rc=int(sys.argv[3])
data=json.loads(live.read_text()) if live.exists() else {}
data.update({"schema":"coordinate_pair.scaleout_live.v1","state":"ended","ended_at_unix":time.time(),"wait_status_observed":True,"exit_code":rc})
live.write_text(json.dumps(data,indent=2)+"\n")
exitp.write_text(json.dumps({"schema":"coordinate_pair.scaleout_exit_status.v1","wrapper_pid":data.get("wrapper_pid"),"runtime_pid":data.get("runtime_pid"),"exit_code":rc,"exit_status_observed":True,"state":"ended","live":str(live),"log":"/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout/runtime/launch/scaleout-gpu0-tied/producer.log"},indent=2)+"\n")
PYEXIT
exit "$RC"
