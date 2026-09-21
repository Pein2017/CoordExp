import json
from pathlib import Path
R=Path(__file__).resolve().parent
paths=list((R/'stage1/runtime').glob('*/309264/prefix/receipt.json'))+list((R/'stage2/runtime').glob('*/receipt.json'))
score=R/'stage1/score-runtime/receipt.json'
if score.exists():paths.append(score)
extraction=R/'stage2/component-capture/receipt.json'
if extraction.exists():paths.append(extraction)
rows=[]
for p in paths:
 d=json.loads(p.read_text());rows.append(dict(path=str(p),status=d['status'],model_forwards=d.get('model_forwards'),elapsed_seconds=d.get('elapsed_seconds'),peak_allocated_bytes=d.get('peak_allocated_bytes'),pid=d.get('pid')))
intervals=[];live=[];exits={}
for p in sorted((R/'logs').glob('*.pid')):
 name=p.stem;pid=int(p.read_text());proc=Path(f'/proc/{pid}/cmdline')
 if proc.exists():
  cmd=proc.read_bytes().replace(b'\x00',b' ').decode(errors='replace')
  if 'successful-row-mechanism' in cmd:live.append(dict(pid=pid,command=cmd))
 exit_path=p.with_suffix('.exit');exits[name]=int(exit_path.read_text()) if exit_path.exists() else None
 start=p.with_suffix('.start');end=p.with_suffix('.end')
 if start.exists() and end.exists():intervals.append((int(start.read_text()),int(end.read_text())))
merged=[]
for a,b in sorted(intervals):
 if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
 else:merged.append([a,b])
result=dict(status='candidate_cost',receipts=rows,native_batches=len(paths)-int(score.exists())-int(extraction.exists()),score_executions=int(score.exists()),prefix_extraction_executions=int(extraction.exists()),model_forwards=sum(x['model_forwards'] or 0 for x in rows),unknown_forward_receipts=[x['path'] for x in rows if x['model_forwards'] is None],measured_runtime_seconds=sum(x['elapsed_seconds'] or 0 for x in rows),allocated_process_seconds=sum(b-a for a,b in intervals),execution_interval_union_seconds=sum(b-a for a,b in merged),execution_window_seconds=(max(b for a,b in intervals)-min(a for a,b in intervals)) if intervals else 0,peak_allocated_bytes=max((x['peak_allocated_bytes'] or 0 for x in rows),default=0),model_exits=exits,owned_live_jobs=live,artifact_bytes=sum(p.stat().st_size for p in R.rglob('*') if p.is_file()),limits=dict(native_batches=24,model_forwards=74144,allocated_gpu_seconds=28800,elapsed_execution_seconds=14400,artifact_bytes=8589934592))
(R/'cost.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['receipts','model_exits']}))
