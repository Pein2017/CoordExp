import json
from pathlib import Path
import torch
R=Path(__file__).resolve().parent;torch.set_num_threads(2)
read=lambda p:json.loads(p.read_text())
names=['native-F','residual-only','cache-only','joint']
payload={n:torch.load(R/'runtime'/n/'capture.pt',map_location='cpu',weights_only=True)['tensors'] for n in names}
s=torch.load(R/'runtime/extract-S/capture.pt',map_location='cpu',weights_only=True)['tensors']
m={n:read(R/'runtime'/n/'receipt.json')['fork']['coord131_minus_coord0'] for n in names}
rows=[]
for n,t in payload.items():
 states={str(l):t['residual'][67][str(l)]['after'][1] for l in [13,14,17,20]};states['head']=t['head'][67][1]
 refs={str(l):s['residual'][67][str(l)]['after'][1] for l in [13,14,17,20]};refs['head']=s['head'][67][1]
 dist={l:dict(delta_to_S=float((v-refs[l]).norm()),norm=float(v.norm()),cosine_to_S=float(torch.nn.functional.cosine_similarity(v,refs[l],dim=0))) for l,v in states.items()}
 rows.append(dict(condition=n,margin=m[n],descriptive_state_distance=dist))
seqs={n:read(R/'runtime'/n/'raw.json')['rows'][1]['token_ids'] for n in names}
pairs=[]
for a,b in [('residual-only','native-F'),('cache-only','native-F'),('joint','residual-only'),('joint','cache-only')]:
 x,y=seqs[a],seqs[b];i=next((i for i,(p,q) in enumerate(zip(x,y)) if p!=q),None)
 pairs.append(dict(a=a,b=b,exact=x==y,first_fork=i,tokens=None if i is None else [x[i],y[i]],
    row_one_based=None if i is None else i//9+1,field=None if i is None else ['opener','category','category_end','box_start','x1','y1','x2','y2','box_end'][i%9]))
d=dict(status='candidate',margin_effects=dict(residual=m['residual-only']-m['native-F'],history_KV=m['cache-only']-m['native-F'],joint=m['joint']-m['native-F'],interaction=m['joint']-m['residual-only']-m['cache-only']+m['native-F']),cells=rows,exact_trajectory_comparisons=pairs,limits='State distances and factorial margin interaction are descriptive at this selected intervention; no layer logit lens, causal percentages or unique information-origin claim.')
(R/'factorial-summary.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(dict(margin_effects=d['margin_effects'],pairs=pairs)))
