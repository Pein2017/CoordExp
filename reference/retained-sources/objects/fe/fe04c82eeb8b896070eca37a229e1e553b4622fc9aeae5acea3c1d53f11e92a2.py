import json,hashlib
from pathlib import Path
import torch
R=Path(__file__).resolve().parent
torch.set_num_threads(2)
read=lambda p:json.loads(p.read_text())
bind=lambda p:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
get=lambda n:read(R/f'stage2/runtime/{n}/raw.json')
checks=[]
for name,reference in [('rebuild-S-to-F','residual-S-to-F-layer20'),('rebuild-control-F','native-F')]:
 folder=R/'stage2/runtime'/name;rec=read(folder/'receipt.json');assert rec['status']=='candidate_complete';assert int((R/f'logs/{name}.exit').read_text())==0
 for k in ['raw','cache','states','logits','producer','panel']:assert rec[k]==bind(Path(rec[k]['path']))
 actual=get(name);expected=get(reference)
 for a,b in zip(actual['rows'],expected['rows']):assert a['token_ids']==b['token_ids'] and a['stop']==b['stop']
 assert actual['prefix']['token_ids']==expected['rows'][1]['token_ids'][:68]
 cache=torch.load(folder/'cache.pt',map_location='cpu',weights_only=False)['last_position']
 native=torch.load(R/'stage2/runtime/native-F/cache.pt',map_location='cpu',weights_only=False)['last_position']
 for layer,kv in cache['67'].items():
  for k,v in kv.items():assert torch.equal(v,native['67'][layer][k])
 patch=torch.load(R/'stage2/runtime/residual-S-to-F-layer20/cache.pt',map_location='cpu',weights_only=False)['last_position']
 changed=[int(layer) for layer,kv in cache['67'].items() if any(not torch.equal(v[1],patch['67'][layer][k][1]) for k,v in kv.items())]
 checks.append(dict(condition=name,receipt=bind(folder/'receipt.json'),full_tokens_and_stops_exact_to=reference,unpatched_fork_cache_exact_to_native_F=True,KV_layers_different_from_retained_patch=changed,forward_count=rec['model_forwards']))
assert checks[0]['KV_layers_different_from_retained_patch']==list(range(21,28))
out=R/'rebuild-verification.json';out.write_text(json.dumps({'status':'passed','checks':checks,'claim':'For selected S-to-F layer20 patch, the entire continuation is reproduced by the emitted fork token with original rebuilt cache; extra retained patch KV is not required for this exact route. Not proof all cache effects are irrelevant.'},indent=2)+'\n');print(json.dumps(checks))
