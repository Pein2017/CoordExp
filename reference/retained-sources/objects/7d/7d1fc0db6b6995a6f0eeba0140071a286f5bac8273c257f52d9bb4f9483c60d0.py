import json,hashlib
from pathlib import Path
import torch
R=Path(__file__).resolve().parent;torch.set_num_threads(2)
rows=[]
def collect(x,p=''):
 if isinstance(x,torch.Tensor):return [dict(key=p,shape=list(x.shape),dtype=str(x.dtype),bytes=x.numel()*x.element_size())]
 if isinstance(x,dict):return [v for k,y in x.items() for v in collect(y,p+'/'+str(k))]
 if isinstance(x,(list,tuple)):return [v for i,y in enumerate(x) for v in collect(y,p+'/'+str(i))]
 return []
for p in sorted((R/'runtime').glob('*/capture.pt')):
 rows.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),file_bytes=p.stat().st_size,tensors=collect(torch.load(p,map_location='cpu',weights_only=True))))
d=dict(status='complete',files=len(rows),file_bytes=sum(x['file_bytes'] for x in rows),records=rows,
 absent=['complete historical KV','attention probability matrices','per-head intervention scan','new sampling RNG','layer logit-lens archive'])
(R/'tensor-inventory.json').write_text(json.dumps(d,indent=2)+'\n');print(dict(files=d['files'],file_bytes=d['file_bytes']))
