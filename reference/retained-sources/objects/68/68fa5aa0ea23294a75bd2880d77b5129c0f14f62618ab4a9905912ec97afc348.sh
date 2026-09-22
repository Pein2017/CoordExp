#!/usr/bin/env bash
set -uo pipefail
BASE="/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-first-coordinate-mediation"
MASTER_LIVE="$BASE/runtime/scaleout-master-live.json"
MASTER_EXIT="$BASE/runtime/scaleout-master-exit.json"
WRAPPERS=()
WRAPPERS+=('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-first-coordinate-mediation/runtime/launch/scaleout-gpu0-tied/run.sh')
WRAPPERS+=('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-first-coordinate-mediation/runtime/launch/scaleout-gpu1-tied/run.sh')
WRAPPERS+=('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-first-coordinate-mediation/runtime/launch/scaleout-gpu2-tied/run.sh')
WRAPPERS+=('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-first-coordinate-mediation/runtime/launch/scaleout-gpu3-tied/run.sh')
WRAPPERS+=('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-first-coordinate-mediation/runtime/launch/scaleout-gpu4-untied/run.sh')
WRAPPERS+=('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-first-coordinate-mediation/runtime/launch/scaleout-gpu5-untied/run.sh')
WRAPPERS+=('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-first-coordinate-mediation/runtime/launch/scaleout-gpu6-untied/run.sh')
WRAPPERS+=('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-first-coordinate-mediation/runtime/launch/scaleout-gpu7-untied/run.sh')
PIDS=()
for wrapper in "${WRAPPERS[@]}"; do
  bash "$wrapper" >"${wrapper%/*}/launcher.log" 2>&1 &
  PIDS+=("$!")
done
python3 - "$MASTER_LIVE" "${PIDS[@]}" <<'PYLIVE'
import json,sys,time,pathlib
path=pathlib.Path(sys.argv[1]); pids=[int(x) for x in sys.argv[2:]]
path.write_text(json.dumps({"schema":"first_coordinate_mediation.scaleout_master_live.v1","state":"running","wrapper_pids":pids,"started_at_unix":time.time()},indent=2)+"\n")
PYLIVE
set +e
RCS=()
for pid in "${PIDS[@]}"; do
  wait "$pid"; RCS+=("$?")
done
python3 - "$MASTER_LIVE" "$MASTER_EXIT" "${RCS[@]}" <<'PYEXIT'
import json,sys,time,pathlib
live=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); rcs=[int(x) for x in sys.argv[3:]]
data=json.loads(live.read_text()); data.update({"state":"ended","ended_at_unix":time.time(),"wait_status_observed":True,"wrapper_exit_codes":rcs}); live.write_text(json.dumps(data,indent=2)+"\n")
out.write_text(json.dumps({"schema":"first_coordinate_mediation.scaleout_master_exit.v1","state":"ended","wait_status_observed":True,"wrapper_pids":data["wrapper_pids"],"wrapper_exit_codes":rcs,"live":str(live)},indent=2)+"\n")
PYEXIT
for rc in "${RCS[@]}"; do if [ "$rc" -ne 0 ]; then exit "$rc"; fi; done
exit 0
