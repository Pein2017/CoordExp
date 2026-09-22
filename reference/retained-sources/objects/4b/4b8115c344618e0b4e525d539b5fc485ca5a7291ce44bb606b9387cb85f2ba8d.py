import hashlib,json
from pathlib import Path
import torch
R=Path(__file__).resolve().parent
torch.set_num_threads(2)
paths=[]
for folder in ['stage1/runtime','stage1/score-runtime','stage2/runtime','stage2/component-capture']:
 paths.extend((R/folder).rglob('*.pt'))
records=[]
def tensors(value,path=''):
 if isinstance(value,torch.Tensor):return [dict(key=path,shape=list(value.shape),dtype=str(value.dtype),bytes=value.numel()*value.element_size())]
 if isinstance(value,dict):return [z for k,v in value.items() for z in tensors(v,path+'.'+str(k))]
 if isinstance(value,(list,tuple)):return [z for i,v in enumerate(value) for z in tensors(v,path+'.'+str(i))]
 return []
for p in sorted(paths):
 value=torch.load(p,map_location='cpu',weights_only=False)
 records.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),file_bytes=p.stat().st_size,tensors=tensors(value)))
result=dict(status='complete',files=len(records),file_bytes=sum(x['file_bytes'] for x in records),records=records,absent=['complete attention weights','complete historical KV archive','layer/head sweep','new sampling RNG draws','direct layer20 incoming-residual snapshot'],present='selected full logits/head inputs, layers6/13/20, literal positions, KV last positions67/68, layer20 component outputs; see per-file tensor keys')
(R/'tensor-inventory.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}))
