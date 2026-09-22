import json
from pathlib import Path
R=Path(__file__).resolve().parent
rows=[]
for p in sorted((R/'runtime').glob('*/receipt.json')):
 d=json.loads(p.read_text());rows.append(dict(path=str(p),condition=p.parent.name,mode=d['mode'],status=d['status'],model_forwards=d.get('model_forwards'),elapsed_seconds=d.get('elapsed_seconds'),peak_allocated_bytes=d.get('peak_allocated_bytes'),pid=d.get('pid')))
intervals=[];live=[];exits={}
for p in sorted((R/'logs').glob('*.pid')):
 name=p.stem;pid=int(p.read_text());proc=Path(f'/proc/{pid}/cmdline')
 if proc.exists():
  cmd=proc.read_bytes().replace(b'\x00',b' ').decode(errors='replace')
  if 'history_rereading' in cmd:live.append(dict(pid=pid,command=cmd))
 e=p.with_suffix('.exit');exits[name]=int(e.read_text()) if e.exists() else None
 a=p.with_suffix('.start');b=p.with_suffix('.end')
 if a.exists() and b.exists():intervals.append((int(a.read_text()),int(b.read_text())))
merged=[]
for a,b in sorted(intervals):
 if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
 else:merged.append([a,b])
d=dict(status='candidate_cost',receipts=rows,native_batches=sum(x['mode']=='full' for x in rows),extractions=sum(x['mode']=='extract' for x in rows),model_forwards=sum(x['model_forwards'] or 0 for x in rows),unknown_forward_receipts=[x['path'] for x in rows if x['model_forwards'] is None],allocated_process_seconds=sum(b-a for a,b in intervals),execution_interval_union_seconds=sum(b-a for a,b in merged),execution_window_seconds=max((b for a,b in intervals),default=0)-min((a for a,b in intervals),default=0),peak_allocated_bytes=max((x['peak_allocated_bytes'] or 0 for x in rows),default=0),model_exits=exits,owned_live_jobs=live,artifact_bytes=sum(p.stat().st_size for p in R.rglob('*') if p.is_file()),limits=dict(native_batches=10,model_forwards=31096,allocated_gpu_seconds=14400,elapsed_execution_seconds=7200,artifact_bytes=3221225472))
(R/'cost.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({k:v for k,v in d.items() if k not in ['receipts','model_exits']}))
