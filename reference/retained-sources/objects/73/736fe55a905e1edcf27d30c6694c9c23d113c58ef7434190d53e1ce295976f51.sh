#!/usr/bin/env bash
set -u
set -o pipefail
BASE=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source
WORKTREE=/data/CoordExp/.worktrees/research-probes
BROAD=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1
DEVICE=1
WORKER_DIR=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/workers
LOG_DIR=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/logs
SAMPLE_DIR=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/samples
EVENTS=$WORKER_DIR/device-$DEVICE.events.jsonl
SUMMARY=$WORKER_DIR/device-$DEVICE.json
PID=$$
mkdir -p "$WORKER_DIR" "$LOG_DIR" "$SAMPLE_DIR"
START_UTC=$(date -u +%FT%TZ)
START_EPOCH=$(date +%s.%N)
printf "worker=%s pid=%s start=%s\n" "$DEVICE" "$PID" "$START_UTC" > "$WORKER_DIR/device-$DEVICE.start.txt"
nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.used,memory.total --format=csv,noheader > "$SAMPLE_DIR/device-$DEVICE.start.csv" 2>&1 || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$SAMPLE_DIR/device-$DEVICE.start-apps.csv" 2>&1 || true
rm -f "$EVENTS"
rm -f "$WORKER_DIR/device-$DEVICE.events.log"
rc_total=0
echo "START tied-417044-healthy $(date -u +%FT%TZ)" >> "$WORKER_DIR/device-$DEVICE.events.log"
STATE_START=$(date -u +%FT%TZ)
STATE_EPOCH_START=$(date +%s.%N)
PYTHONPATH="$WORKTREE" python3 -m probes.training_set_completion.recurrence_spatial.producer --model tied --device cuda:1 --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-417044-healthy.json --out /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1 --result-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/runtime/tied-417044-healthy.json > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/logs/tied-417044-healthy.producer.log 2>&1
RC=$?
STATE_END=$(date -u +%FT%TZ)
STATE_EPOCH_END=$(date +%s.%N)
python3 - "$DEVICE" "tied-417044-healthy" "tied" "$RC" "$STATE_START" "$STATE_END" "$STATE_EPOCH_START" "$STATE_EPOCH_END" "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/runtime/tied-417044-healthy.json" "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-417044-healthy.json" "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/logs/tied-417044-healthy.producer.log" >> "$EVENTS" <<'PY'
import json,sys
from pathlib import Path
(device,sid,model,rc,start,end,epoch_start,epoch_end,result,manifest,log)=sys.argv[1:]
entry={"device":int(device),"state_id":sid,"model":model,"returncode":int(rc),"start_utc":start,"end_utc":end,"start_epoch":float(epoch_start),"end_epoch":float(epoch_end),"runtime_path":result,"manifest_path":manifest,"log_path":log,"runtime_exists":Path(result).exists()}
if Path(result).exists():
 try:
  d=json.loads(Path(result).read_text())
  entry.update({"status":d.get("status"),"model_forwards":d.get("model_forwards"),"vision_forwards":d.get("vision_forwards"),"elapsed_seconds":d.get("elapsed_seconds"),"admission":d.get("admission")})
 except Exception as exc: entry["runtime_parse_error"]=repr(exc)
print(json.dumps(entry,sort_keys=True))
PY
if [ "$RC" -ne 0 ]; then rc_total=1; fi
echo "END tied-417044-healthy rc=$RC $(date -u +%FT%TZ)" >> "$WORKER_DIR/device-$DEVICE.events.log"
echo "START tied-train-301827-failure $(date -u +%FT%TZ)" >> "$WORKER_DIR/device-$DEVICE.events.log"
STATE_START=$(date -u +%FT%TZ)
STATE_EPOCH_START=$(date +%s.%N)
PYTHONPATH="$WORKTREE" python3 -m probes.training_set_completion.recurrence_spatial.producer --model tied --device cuda:1 --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-train-301827-failure.json --out /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1 --result-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/runtime/tied-train-301827-failure.json > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/logs/tied-train-301827-failure.producer.log 2>&1
RC=$?
STATE_END=$(date -u +%FT%TZ)
STATE_EPOCH_END=$(date +%s.%N)
python3 - "$DEVICE" "tied-train-301827-failure" "tied" "$RC" "$STATE_START" "$STATE_END" "$STATE_EPOCH_START" "$STATE_EPOCH_END" "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/runtime/tied-train-301827-failure.json" "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-train-301827-failure.json" "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/logs/tied-train-301827-failure.producer.log" >> "$EVENTS" <<'PY'
import json,sys
from pathlib import Path
(device,sid,model,rc,start,end,epoch_start,epoch_end,result,manifest,log)=sys.argv[1:]
entry={"device":int(device),"state_id":sid,"model":model,"returncode":int(rc),"start_utc":start,"end_utc":end,"start_epoch":float(epoch_start),"end_epoch":float(epoch_end),"runtime_path":result,"manifest_path":manifest,"log_path":log,"runtime_exists":Path(result).exists()}
if Path(result).exists():
 try:
  d=json.loads(Path(result).read_text())
  entry.update({"status":d.get("status"),"model_forwards":d.get("model_forwards"),"vision_forwards":d.get("vision_forwards"),"elapsed_seconds":d.get("elapsed_seconds"),"admission":d.get("admission")})
 except Exception as exc: entry["runtime_parse_error"]=repr(exc)
print(json.dumps(entry,sort_keys=True))
PY
if [ "$RC" -ne 0 ]; then rc_total=1; fi
echo "END tied-train-301827-failure rc=$RC $(date -u +%FT%TZ)" >> "$WORKER_DIR/device-$DEVICE.events.log"
echo "START tied-885-healthy $(date -u +%FT%TZ)" >> "$WORKER_DIR/device-$DEVICE.events.log"
STATE_START=$(date -u +%FT%TZ)
STATE_EPOCH_START=$(date +%s.%N)
PYTHONPATH="$WORKTREE" python3 -m probes.training_set_completion.recurrence_spatial.producer --model tied --device cuda:1 --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-885-healthy.json --out /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1 --result-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/runtime/tied-885-healthy.json > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/logs/tied-885-healthy.producer.log 2>&1
RC=$?
STATE_END=$(date -u +%FT%TZ)
STATE_EPOCH_END=$(date +%s.%N)
python3 - "$DEVICE" "tied-885-healthy" "tied" "$RC" "$STATE_START" "$STATE_END" "$STATE_EPOCH_START" "$STATE_EPOCH_END" "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/runtime/tied-885-healthy.json" "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-885-healthy.json" "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/logs/tied-885-healthy.producer.log" >> "$EVENTS" <<'PY'
import json,sys
from pathlib import Path
(device,sid,model,rc,start,end,epoch_start,epoch_end,result,manifest,log)=sys.argv[1:]
entry={"device":int(device),"state_id":sid,"model":model,"returncode":int(rc),"start_utc":start,"end_utc":end,"start_epoch":float(epoch_start),"end_epoch":float(epoch_end),"runtime_path":result,"manifest_path":manifest,"log_path":log,"runtime_exists":Path(result).exists()}
if Path(result).exists():
 try:
  d=json.loads(Path(result).read_text())
  entry.update({"status":d.get("status"),"model_forwards":d.get("model_forwards"),"vision_forwards":d.get("vision_forwards"),"elapsed_seconds":d.get("elapsed_seconds"),"admission":d.get("admission")})
 except Exception as exc: entry["runtime_parse_error"]=repr(exc)
print(json.dumps(entry,sort_keys=True))
PY
if [ "$RC" -ne 0 ]; then rc_total=1; fi
echo "END tied-885-healthy rc=$RC $(date -u +%FT%TZ)" >> "$WORKER_DIR/device-$DEVICE.events.log"
echo "START tied-train-477785-healthy $(date -u +%FT%TZ)" >> "$WORKER_DIR/device-$DEVICE.events.log"
STATE_START=$(date -u +%FT%TZ)
STATE_EPOCH_START=$(date +%s.%N)
PYTHONPATH="$WORKTREE" python3 -m probes.training_set_completion.recurrence_spatial.producer --model tied --device cuda:1 --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-train-477785-healthy.json --out /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1 --result-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/runtime/tied-train-477785-healthy.json > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/logs/tied-train-477785-healthy.producer.log 2>&1
RC=$?
STATE_END=$(date -u +%FT%TZ)
STATE_EPOCH_END=$(date +%s.%N)
python3 - "$DEVICE" "tied-train-477785-healthy" "tied" "$RC" "$STATE_START" "$STATE_END" "$STATE_EPOCH_START" "$STATE_EPOCH_END" "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/runtime/tied-train-477785-healthy.json" "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-train-477785-healthy.json" "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/logs/tied-train-477785-healthy.producer.log" >> "$EVENTS" <<'PY'
import json,sys
from pathlib import Path
(device,sid,model,rc,start,end,epoch_start,epoch_end,result,manifest,log)=sys.argv[1:]
entry={"device":int(device),"state_id":sid,"model":model,"returncode":int(rc),"start_utc":start,"end_utc":end,"start_epoch":float(epoch_start),"end_epoch":float(epoch_end),"runtime_path":result,"manifest_path":manifest,"log_path":log,"runtime_exists":Path(result).exists()}
if Path(result).exists():
 try:
  d=json.loads(Path(result).read_text())
  entry.update({"status":d.get("status"),"model_forwards":d.get("model_forwards"),"vision_forwards":d.get("vision_forwards"),"elapsed_seconds":d.get("elapsed_seconds"),"admission":d.get("admission")})
 except Exception as exc: entry["runtime_parse_error"]=repr(exc)
print(json.dumps(entry,sort_keys=True))
PY
if [ "$RC" -ne 0 ]; then rc_total=1; fi
echo "END tied-train-477785-healthy rc=$RC $(date -u +%FT%TZ)" >> "$WORKER_DIR/device-$DEVICE.events.log"
echo "START tied-309264-healthy $(date -u +%FT%TZ)" >> "$WORKER_DIR/device-$DEVICE.events.log"
STATE_START=$(date -u +%FT%TZ)
STATE_EPOCH_START=$(date +%s.%N)
PYTHONPATH="$WORKTREE" python3 -m probes.training_set_completion.recurrence_spatial.producer --model tied --device cuda:1 --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-309264-healthy.json --out /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1 --result-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/runtime/tied-309264-healthy.json > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/logs/tied-309264-healthy.producer.log 2>&1
RC=$?
STATE_END=$(date -u +%FT%TZ)
STATE_EPOCH_END=$(date +%s.%N)
python3 - "$DEVICE" "tied-309264-healthy" "tied" "$RC" "$STATE_START" "$STATE_END" "$STATE_EPOCH_START" "$STATE_EPOCH_END" "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/runtime/tied-309264-healthy.json" "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-309264-healthy.json" "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/logs/tied-309264-healthy.producer.log" >> "$EVENTS" <<'PY'
import json,sys
from pathlib import Path
(device,sid,model,rc,start,end,epoch_start,epoch_end,result,manifest,log)=sys.argv[1:]
entry={"device":int(device),"state_id":sid,"model":model,"returncode":int(rc),"start_utc":start,"end_utc":end,"start_epoch":float(epoch_start),"end_epoch":float(epoch_end),"runtime_path":result,"manifest_path":manifest,"log_path":log,"runtime_exists":Path(result).exists()}
if Path(result).exists():
 try:
  d=json.loads(Path(result).read_text())
  entry.update({"status":d.get("status"),"model_forwards":d.get("model_forwards"),"vision_forwards":d.get("vision_forwards"),"elapsed_seconds":d.get("elapsed_seconds"),"admission":d.get("admission")})
 except Exception as exc: entry["runtime_parse_error"]=repr(exc)
print(json.dumps(entry,sort_keys=True))
PY
if [ "$RC" -ne 0 ]; then rc_total=1; fi
echo "END tied-309264-healthy rc=$RC $(date -u +%FT%TZ)" >> "$WORKER_DIR/device-$DEVICE.events.log"
END_UTC=$(date -u +%FT%TZ)
END_EPOCH=$(date +%s.%N)
nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.used,memory.total --format=csv,noheader > "$SAMPLE_DIR/device-$DEVICE.end.csv" 2>&1 || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$SAMPLE_DIR/device-$DEVICE.end-apps.csv" 2>&1 || true
python3 - "$DEVICE" "$PID" "$START_UTC" "$END_UTC" "$START_EPOCH" "$END_EPOCH" "$EVENTS" "$rc_total" "$SAMPLE_DIR/device-$DEVICE.start.csv" "$SAMPLE_DIR/device-$DEVICE.end.csv" "$SAMPLE_DIR/device-$DEVICE.start-apps.csv" "$SAMPLE_DIR/device-$DEVICE.end-apps.csv" "$SUMMARY" <<'PY'
import json,sys
from pathlib import Path
(device,pid,start,end,epoch_start,epoch_end,events,rc,start_gpu,end_gpu,start_apps,end_apps,summary)=sys.argv[1:]
rows=[]
for line in Path(events).read_text().splitlines() if Path(events).exists() else []:
 try: rows.append(json.loads(line))
 except Exception: pass
model=sum(int(r.get("model_forwards") or 0) for r in rows); vision=sum(int(r.get("vision_forwards") or 0) for r in rows)
out={"schema":"recurrence_spatial_source.broad_v1_worker.v1","device":int(device),"pid":int(pid),"start_utc":start,"end_utc":end,"start_epoch":float(epoch_start),"end_epoch":float(epoch_end),"wall_seconds":float(epoch_end)-float(epoch_start),"returncode":int(rc),"states":rows,"state_count":len(rows),"model_forwards":model,"vision_forwards":vision,"gpu_seconds":None,"samples":{"start_gpu":start_gpu,"end_gpu":end_gpu,"start_apps":start_apps,"end_apps":end_apps},"owned_processes_remaining":False}
Path(summary).write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({"device":int(device),"state_count":len(rows),"model_forwards":model,"vision_forwards":vision,"returncode":int(rc)}))
PY
exit "$rc_total"
