import hashlib,json
from pathlib import Path
import torch
R=Path(__file__).resolve().parent
torch.set_num_threads(2)
read=lambda p:json.loads(p.read_text())
bind=lambda p:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
donor=torch.load(R/'stage2/component-capture/components.pt',map_location='cpu',weights_only=True)
capture=read(R/'stage2/component-capture/receipt.json');assert capture['status']=='candidate_complete' and capture['model_forwards']==68 and capture['fork_logits_exact'];assert capture['tensors']==bind(R/'stage2/component-capture/components.pt')
base=read(R/'stage2/runtime/native-F/raw.json');basecache=torch.load(R/'stage2/runtime/native-F/cache.pt',map_location='cpu',weights_only=False)['last_position']['67'];checks=[]
for component in ['attention','mlp']:
 name=f'{component}-S-to-F-layer20';folder=R/'stage2/runtime'/name;rec=read(folder/'receipt.json');assert rec['status']=='candidate_complete';assert int((R/f'logs/{name}.exit').read_text())==0
 for k in ['producer','panel','raw','native_receipt','tensors','donor']:assert rec[k]==bind(Path(rec[k]['path']))
 raw=read(Path(rec['raw']['path']));payload=torch.load(folder/'components.pt',map_location='cpu',weights_only=True)
 assert len(rec['patch'])==1 and rec['patch'][0]['kind']==component
 assert torch.equal(payload['states'][component+'_after'][1],donor['states'][component][1])
 assert torch.equal(payload['states'][component+'_after'][[0,2,3]],payload['states'][component][[0,2,3]])
 assert raw['rows'][1]['token_ids'][:67]==base['rows'][1]['token_ids'][:67]
 for i in [0,2,3]:assert raw['rows'][i]['token_ids']==base['rows'][i]['token_ids'] and raw['rows'][i]['stop']==base['rows'][i]['stop']
 changed=[]
 for layer,kv in payload['cache']['67'].items():
  for k,v in kv.items():assert torch.equal(v[[0,2,3]],basecache[layer][k][[0,2,3]])
  if any(not torch.equal(v[1],basecache[layer][k][1]) for k,v in kv.items()):changed.append(int(layer))
 assert all(x>20 for x in changed)
 v=payload['logits'][67][1];top=v.topk(2)
 checks.append(dict(condition=name,receipt=bind(folder/'receipt.json'),argmax=int(v.argmax()),margin=float(top.values[0]-top.values[1]),S_minus_F=float(v[151801]-v[151670]),S_rank=1+int((v>v[151801]).sum()),F_rank=1+int((v>v[151670]).sum()),EOS_rank=1+int((v>v[151645]).sum()),EOS_logit=float(v[151645]),fork_changed_KV_layers=sorted(changed),native_prefix_and_companions_exact=True,target_component_exact_to_donor=True,output_stop=raw['rows'][1]['stop'],tokens=len(raw['rows'][1]['token_ids'])))
(R/'component-verification.json').write_text(json.dumps({'status':'passed','prefix_extraction':bind(R/'stage2/component-capture/receipt.json'),'checks':checks},indent=2)+'\n');print(json.dumps(checks))
