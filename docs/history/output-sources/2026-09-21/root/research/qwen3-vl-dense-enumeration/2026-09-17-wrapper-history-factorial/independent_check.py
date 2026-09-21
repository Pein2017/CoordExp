import json,hashlib,collections
from pathlib import Path
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
bind=lambda p:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
a=read(R/'reduction.json');assert a==read(R/'independent-reduction.json')
checks=[]
for name,c in a['cells'].items():
 raw=read(Path(c['raw']['path']))['rows'][1];tokens=raw['token_ids'][72:];boxes=[]
 for i,t in enumerate(tokens):
  if t==151648 and i+5<len(tokens) and tokens[i+5]==151649 and all(151670<=v<=152669 for v in tokens[i+1:i+5]):boxes.append(tuple(v-151670 for v in tokens[i+1:i+5]))
 v=c['versions']['current']['free'];b=v['burden'];assert len(boxes)==b['complete_rows'];assert sum(not(x[0]<x[2] and x[1]<x[3]) for x in boxes)==b['invalid']
 assert all(r['description']=='bird' for r in v['complete_rows'])
 assert sum(n-1 for n in collections.Counter(boxes).values())==b['literal_repeats']
 owners=v['matches']['covered_owner_ids'];assert len(owners)==len(set(owners));assert set(owners)<={x['owner_id'] for x in a['banks']['current']}
 assert set(c['primary']['free'])==set(owners)-set(c['primary']['excluded_supplied_union'])
 checks.append(dict(cell=name,raw=bind(Path(c['raw']['path'])),free_complete_rows=len(boxes),free_invalid=b['invalid'],literal_repeats=b['literal_repeats'],positive_ids=owners))
out=dict(status='passed',JSON_exact_replay=True,independent_token_geometry_and_literal_repeat_counts=True,cells=checks,bindings=[bind(R/'reduction.json'),bind(R/'independent-reduction.json'),bind(Path(__file__))])
(R/'independent-output-check.json').write_text(json.dumps(out,indent=2)+'\n');print('passed',len(checks))
