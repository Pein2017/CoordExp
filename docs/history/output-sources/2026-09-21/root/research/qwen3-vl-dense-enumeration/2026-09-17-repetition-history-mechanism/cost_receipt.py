"""Count executed receipts and conservative process wall allocations, no GPU queries."""
import json,hashlib,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
receipts=sorted((R/'runtime').glob('**/receipt.json'))+sorted((R/'scores').glob('*/receipt.json'))
rows=[];live=[]
for p in receipts:
 a=read(p);pid=a.get('pid');proc=Path(f'/proc/{pid}/cmdline')
 if proc.exists():
  cmd=proc.read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if 'repetition_history' in cmd or 'sampling-producer' in cmd:live.append(dict(pid=pid,command=cmd))
 rows.append(dict(receipt=bind(p),status=a['status'],pid=pid,model_forwards=a.get('model_forwards',0),measured_runtime_seconds=a.get('elapsed_seconds',0),peak_allocated_bytes=a.get('peak_allocated_bytes',0)))
logs=[]
for p in sorted((R/'logs').glob('*.exit')):
 log=p.with_suffix('.log')
 if not log.exists():continue
 born=int(subprocess.check_output(['stat','-c','%W',str(log)],text=True));end=p.stat().st_mtime
 assert born>0 and end>=born
 logs.append(dict(log=str(log),exit=int(p.read_text()),start_lower_bound_unix=born,end_unix=end,allocated_process_seconds_upper_bound=end-born+1))
merged=[]
for x in sorted(logs,key=lambda x:x['start_lower_bound_unix']):
 start=x['start_lower_bound_unix'];end=x['end_unix']+1
 if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
 else:merged.append([start,end])
# Log opening precedes process launch; one GPU per recorded producer. Seconds include imports/load.
out=dict(status='candidate',receipts=rows,execution_logs=logs,live_owned_jobs=live,
 totals=dict(receipt_executions=len(rows),native_batch_executions=sum('/runtime/' in x['receipt']['path'] for x in rows),score_executions=sum('/scores/' in x['receipt']['path'] for x in rows),model_forwards=sum(x['model_forwards'] for x in rows),measured_runtime_seconds=sum(x['measured_runtime_seconds'] for x in rows),allocated_process_seconds_upper_bound=sum(x['allocated_process_seconds_upper_bound'] for x in logs),peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in rows),elapsed_execution_seconds_upper_bound=sum(b-a for a,b in merged),execution_interval_union=merged),
 bounds=dict(native_batches=64,model_forwards=201472,allocated_gpu_seconds=43200,elapsed_execution_seconds=21600),
 cost_note='Per-receipt runtime timers differ in load inclusion. Allocation upper bound uses durable log birth to exit-file timestamp plus one second; idle agent gaps are not GPU execution. One GPU per process; no model calls in saved-logit RNG verification.')
assert out['totals']['model_forwards']<=201472 and out['totals']['native_batch_executions']<=64
assert out['totals']['allocated_process_seconds_upper_bound']<=43200
assert out['totals']['elapsed_execution_seconds_upper_bound']<=21600
(R/'cost.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(totals=out['totals'],live=live)))
