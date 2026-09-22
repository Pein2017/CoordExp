#!/usr/bin/env bash
set -u
cd /data/CoordExp/.worktrees/research-probes
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
R=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-owner-recurrence-row-branch
mkdir -p "$R/logs"
echo $$ > "$R/native-shell.pid"
(CUDA_VISIBLE_DEVICES=0 python "$R/producer.py" --panel "$R/panel.json" --case 309264 --arm native --output-root "$R/runtime" > "$R/logs/309264-native.log" 2>&1; echo $? > "$R/logs/309264-native.exit") &
p1=$!
(CUDA_VISIBLE_DEVICES=1 python "$R/producer.py" --panel "$R/panel.json" --case 386313 --arm native --output-root "$R/runtime" > "$R/logs/386313-native.log" 2>&1; echo $? > "$R/logs/386313-native.exit") &
p2=$!
wait "$p1" "$p2"
python - <<'CHECK'
import json
from pathlib import Path
r=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-owner-recurrence-row-branch')
x={str(i):int((r/f'logs/{i}-native.exit').read_text()) for i in [309264,386313]}
(r/'native-stage.json').write_text(json.dumps({'exit_codes':x,'status':'complete' if all(v==0 for v in x.values()) else 'failed'},indent=2)+'\n')
CHECK
