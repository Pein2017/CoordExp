import hashlib,json
from pathlib import Path
b=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1')
cache={}; refs=[];bad=[]
def walk(x):
 if isinstance(x,dict):
  if isinstance(x.get('path'),str) and isinstance(x.get('sha256'),str):
   p=Path(x['path']); key=str(p)
   if key not in cache:cache[key]=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
   refs.append(key)
   if cache[key]!=x['sha256']:bad.append({'path':key,'expected':x['sha256'],'actual':cache[key]})
  for v in x.values():walk(v)
 elif isinstance(x,list):
  for v in x:walk(v)
for name in ['result.json','artifact-map.json','cost-receipt.json','job-closure.json']:
 walk(json.loads((b/name).read_text()))
out={'status':'pass' if not bad else 'mismatch','references':len(refs),'unique_files':len(cache),'mismatches':bad}
Path(__file__).with_name('spatial-bindings-check.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
