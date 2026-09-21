#!/usr/bin/env bash
set -u
set -o pipefail
BROAD="/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1"
RUN_START=$(date -u +%FT%TZ)
RUN_EPOCH=$(date +%s.%N)
printf 'run_start=%s epoch=%s\n' "$RUN_START" "$RUN_EPOCH" > "$BROAD/workers/run-start.txt"
pids=()
for device in 0 1 2 3 4 5 6 7; do
  "$BROAD/worker-device-$device.sh" > "$BROAD/workers/device-$device.stdout.log" 2>&1 &
  pids+=("$!")
  printf '%s %s\n' "$device" "$!" >> "$BROAD/workers/run-pids.txt"
done
rc_total=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc_total=1
done
RUN_END=$(date -u +%FT%TZ)
RUN_END_EPOCH=$(date +%s.%N)
printf 'run_end=%s epoch=%s rc=%s\n' "$RUN_END" "$RUN_END_EPOCH" "$rc_total" > "$BROAD/workers/run-end.txt"
python3 - "$BROAD" "$RUN_START" "$RUN_END" "$RUN_EPOCH" "$RUN_END_EPOCH" "$rc_total" <<'PY2'
import json,sys
from pathlib import Path
(root,start,end,epoch_start,epoch_end,rc)=sys.argv[1:]
rows=[]
for p in sorted((Path(root)/'workers').glob('device-*.json')):
 try: rows.append(json.loads(p.read_text()))
 except Exception: pass
out={'schema':'recurrence_spatial_source.broad_v1_run.v1','status':'closed' if int(rc)==0 and len(rows)==8 else 'partial','run_start_utc':start,'run_end_utc':end,'run_start_epoch':float(epoch_start),'run_end_epoch':float(epoch_end),'wall_seconds':float(epoch_end)-float(epoch_start),'returncode':int(rc),'devices':rows,'device_count':len(rows),'states':sum(int(row.get('state_count',0)) for row in rows),'model_forwards':sum(int(row.get('model_forwards',0)) for row in rows),'vision_forwards':sum(int(row.get('vision_forwards',0)) for row in rows),'gpu_seconds':None,'owned_processes_remaining':False}
(Path(root)/'run-receipt.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':out['status'],'devices':len(rows),'states':out['states'],'model_forwards':out['model_forwards'],'vision_forwards':out['vision_forwards'],'returncode':int(rc)}))
PY2
exit "$rc_total"
