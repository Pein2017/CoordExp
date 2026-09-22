#!/usr/bin/env bash
set -euo pipefail
WORKTREE=/data/CoordExp/.worktrees/research-probes
ROOT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/corrected-pilot-v2
CELLS=10-,10+,01-,01+,11-,11+
START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
START_EPOCH=$(date +%s.%N)
cd "$WORKTREE"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits > "$ROOT/logs/gpu0-transforms-before.csv"
python3 -u -m probes.training_set_completion.recurrence_spatial.producer \
  --model tied --device cuda:0 --cells "$CELLS" --skip-gate \
  --out "$ROOT" \
  --manifest-path "$ROOT/inputs/manifests/tied-417044-failure.json" \
  --result-path "$ROOT/raw/tied-transforms-runtime.json" > "$ROOT/logs/tied-transforms-producer.log" 2>&1
TIED_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
TIED_END_EPOCH=$(date +%s.%N)
python3 -u -m probes.training_set_completion.recurrence_spatial.producer \
  --model untied --device cuda:0 --cells "$CELLS" --skip-gate \
  --out "$ROOT" \
  --manifest-path "$ROOT/inputs/manifests/untied-417044-failure.json" \
  --result-path "$ROOT/raw/untied-transforms-runtime.json" > "$ROOT/logs/untied-transforms-producer.log" 2>&1
END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
END_EPOCH=$(date +%s.%N)
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits > "$ROOT/logs/gpu0-transforms-after.csv"
START_UTC="$START_UTC" START_EPOCH="$START_EPOCH" TIED_END_UTC="$TIED_END_UTC" TIED_END_EPOCH="$TIED_END_EPOCH" END_UTC="$END_UTC" END_EPOCH="$END_EPOCH" python3 - "$ROOT" <<'PY'
import json,os,sys
from pathlib import Path
root=Path(sys.argv[1])
def f(name): return {'path':str(root/'raw'/name),'exists':(root/'raw'/name).exists()}
r={
 'schema':'recurrence_spatial_source.corrected_pilot_v2_transform_launch.v1',
 'attempt_id':'corrected-pilot-v2','device':'cuda:0','device_index':0,
 'cells':['10-','10+','01-','01+','11-','11+'],'models':['tied','untied'],
 'start_utc':os.environ['START_UTC'],'tied_end_utc':os.environ['TIED_END_UTC'],'end_utc':os.environ['END_UTC'],
 'start_epoch':float(os.environ['START_EPOCH']),'tied_end_epoch':float(os.environ['TIED_END_EPOCH']),'end_epoch':float(os.environ['END_EPOCH']),
 'wall_seconds_total':float(os.environ['END_EPOCH'])-float(os.environ['START_EPOCH']),
 'wall_seconds_tied':float(os.environ['TIED_END_EPOCH'])-float(os.environ['START_EPOCH']),
 'wall_seconds_untied':float(os.environ['END_EPOCH'])-float(os.environ['TIED_END_EPOCH']),
 'skip_gate':True,'00_rerun':False,
 'raw_results':{'tied':f('tied-transforms-runtime.json'),'untied':f('untied-transforms-runtime.json')},
 'logs':{'tied':str(root/'logs/tied-transforms-producer.log'),'untied':str(root/'logs/untied-transforms-producer.log'),'before':str(root/'logs/gpu0-transforms-before.csv'),'after':str(root/'logs/gpu0-transforms-after.csv')},
 'gpu_seconds':'not instrumented; enclosing wall interval retained',
}
(root/'transform-launch-receipt.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
PY
