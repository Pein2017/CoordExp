import json,os,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
rows=[]
for p in sorted((R/'runtime').glob('*/receipt.json')):
 r=json.loads(p.read_text());n=p.parent.name
 exit_path=R/f'logs/{n}.exit';assert exit_path.exists() and exit_path.read_text().strip()=='0',(n,'unsettled')
 pid=int((R/f'logs/{n}.pid').read_text());live=Path(f'/proc/{pid}').exists()
 if live:
  command=Path(f'/proc/{pid}/cmdline').read_bytes();assert b'wrapper_history' not in command,(n,pid,'live owned producer')
 rows.append(dict(cell=n,mode=r['mode'],pid=pid,pid_absent=not live,status=r['status'],forwards=r['model_forwards'],seconds=r['elapsed_seconds'],start=int((R/f'logs/{n}.start').read_text()),end=int((R/f'logs/{n}.end').read_text()),peak_allocated_bytes=r['peak_allocated_bytes']))
assert rows
out=dict(cells=rows,batches=len(rows),model_forwards=sum(r['forwards'] for r in rows),allocated_GPU_seconds=sum(r['seconds'] for r in rows),elapsed_execution_seconds=max(r['end'] for r in rows)-min(r['start'] for r in rows),tensor_bytes=sum(p.stat().st_size for p in (R/'runtime').rglob('*.pt')),owned_live_jobs=[])
assert sum(r['mode']=='full' for r in rows)<=10 and sum(r['forwards'] for r in rows if r['mode']=='extract')<=256 and out['model_forwards']<=31096 and out['allocated_GPU_seconds']<=14400 and out['elapsed_execution_seconds']<=7200 and out['tensor_bytes']<=3*1024**3
(R/'cost.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
