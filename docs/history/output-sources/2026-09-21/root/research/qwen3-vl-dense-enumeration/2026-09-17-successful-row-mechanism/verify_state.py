import hashlib,json,sys
from pathlib import Path
import torch
R=Path(__file__).resolve().parent
torch.set_num_threads(2)
read=lambda p:json.loads(p.read_text())
bind=lambda p:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
checks=[]
for name in sys.argv[1:]:
 folder=R/'stage2/runtime'/name;rec=read(folder/'receipt.json');raw=read(folder/'raw.json');route=rec['route'];target=rec['target_position'];panel=read(Path(rec['panel']['path']))
 assert rec['status']=='candidate_complete'
 for key in ['raw','states','logits','cache','producer','panel']:
  b=rec[key];assert bind(Path(b['path']))==b,(name,key)
 for b in rec['installed_source_bindings']:assert bind(Path(b['path']))==b,(name,b['path'])
 assert int((R/f'logs/{name}.exit').read_text())==0
 prior=read(Path(panel['cases'][0]['source_sampled_raw']['path']))
 assert raw['rows'][target]['token_ids'][:63]==prior['rows'][target]['token_ids'][:63]
 for i,row in enumerate(raw['rows']):
  if i!=target or rec['mode'] in ['native','self']:
   assert row['token_ids']==prior['rows'][i]['token_ids'] and row['stop']==prior['rows'][i]['stop'],(name,i)
 states=torch.load(folder/'states.pt',map_location='cpu',weights_only=False);logits=torch.load(folder/'logits.pt',map_location='cpu',weights_only=False);cache=torch.load(folder/'cache.pt',map_location='cpu',weights_only=False)
 v=logits['raw_logits'][67][target].float();top=v.topk(2)
 item=dict(condition=name,receipt=bind(folder/'receipt.json'),fork_argmax=int(v.argmax()),margin=float(top.values[0]-top.values[1]),S_minus_F=float(v[151801]-v[151670]),S_rank=1+int((v>v[151801]).sum()),F_rank=1+int((v>v[151670]).sum()),EOS_rank=1+int((v>v[151645]).sum()),EOS_logit=float(v[151645]),model_forwards=rec['model_forwards'],tokens=len(raw['rows'][target]['token_ids']),stop=raw['rows'][target]['stop'])
 assert set(map(int,states['captures']))=={63,67,68,76,103}
 assert set(states['captures']['67']['layers'])=={'6','13','20'}
 if rec['mode']=='native':
  stage1=R/f"stage1/runtime/{'SS' if route=='S' else 'FF'}/309264/prefix"
  saved=torch.load(stage1/'pulse-captures.pt',map_location='cpu',weights_only=True)['captures'][67]
  item['stage1_logit_max_error']=float((saved['before_logits']-v).abs().max());assert item['stage1_logit_max_error']==0
  assert torch.equal(saved['lm_head_input'],states['captures']['67']['head_before'][target])
 if rec['mode']=='self':
  original=R/'stage2/runtime'/f'native-{route}'
  base=torch.load(original/'logits.pt',map_location='cpu',weights_only=False)['raw_logits'][67]
  item['self_logit_max_error']=float((base-logits['raw_logits'][67]).abs().max());assert item['self_logit_max_error']==0
  for p in rec['patch_records']:assert p['before']['sha256']==p['after']['sha256']
  base_cache=torch.load(original/'cache.pt',map_location='cpu',weights_only=False)['last_position']
  for offset,ls in cache['last_position'].items():
   for layer,kv in ls.items():
    for k,t in kv.items():assert torch.equal(t,base_cache[offset][layer][k]),(name,offset,layer,k)
  item['self_all_sampled_cache_slices_exact']=True
 if rec['mode'] in ['head','residual']:
  original=R/'stage2/runtime'/f'native-{route}'
  basecache=torch.load(original/'cache.pt',map_location='cpu',weights_only=False)['last_position']['67']
  changed=[]
  for layer,kv in cache['last_position']['67'].items():
   for k,t in kv.items():
    assert torch.equal(t[[i for i in range(4) if i!=target]],basecache[layer][k][[i for i in range(4) if i!=target]])
   if any(not torch.equal(t[target],basecache[layer][k][target]) for k,t in kv.items()):changed.append(int(layer))
  item['fork_changed_KV_layers']=sorted(changed)
  if rec['mode']=='head':assert not changed
  if rec['mode']=='residual':assert all(x>rec['depth'] for x in changed)
  donor=rec['donor_route'];donor_logits=torch.load(R/'stage2/runtime'/f'native-{donor}'/'logits.pt',map_location='cpu',weights_only=False)['raw_logits'][67][target]
  item['donor_argmax']=int(donor_logits.argmax());item['donor_argmax_transfer']=item['fork_argmax']==item['donor_argmax']
  if rec['mode']=='head':assert torch.equal(v,donor_logits)
 checks.append(item)
out=R/('state-verification-'+'-'.join(sys.argv[1:])+'.json');out.write_text(json.dumps({'status':'passed','checks':checks},indent=2)+'\n');print(json.dumps(checks))
