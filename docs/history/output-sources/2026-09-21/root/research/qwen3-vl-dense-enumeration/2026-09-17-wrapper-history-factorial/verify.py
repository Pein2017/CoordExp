import json,hashlib,sys
from pathlib import Path
import torch
R=Path(__file__).resolve().parent;P=R.parent/'2026-09-17-successful-row-mechanism'
torch.set_num_threads(2)
read=lambda p:json.loads(p.read_text())
bind=lambda p:dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
checks=[]
for name in sys.argv[1:]:
 r=R/'runtime'/name;rec=read(r/'receipt.json');raw=read(r/'raw.json');d=torch.load(r/'capture.pt',map_location='cpu',weights_only=True);t=d['tensors']
 assert rec['status']=='candidate_complete';assert int((R/f'logs/{name}.exit').read_text())==0
 for k in ['producer','panel','raw','capture']:
  b=rec[k];assert bind(Path(b['path']))==b,(name,k)
 for b in rec['installed_sources']:assert bind(Path(b['path']))==b
 panel=read(Path(rec['panel']['path']));route=panel['route'];base=P/f'stage2/runtime/native-{route}'
 assert rec['loaded_identity']==read(base/'receipt.json')['loaded_model_identity']
 old=read(Path(panel['cases'][0]['source_sampled_raw']['path']))['rows']
 for i,row in enumerate(raw['rows']):
  if rec['mode']=='extract':assert row['token_ids']==old[i]['token_ids'][:68]
  elif i!=1 or name in ['native-F','self-F']:
   assert row['token_ids']==old[i]['token_ids'] and row['stop']==old[i]['stop']
 assert d['cache_range']==[d['width']+54,d['width']+63]
 assert d['prefix'][54:63]==d['row_tokens']
 for x in d['trace']:
  assert x['cache_position']==[d['width']+x['offset']-1]
  if x['offset']<=63:assert x['ids'][1][0]==d['prefix'][x['offset']-1]
  elif x['offset']<=67:assert x['ids'][1][0]==old[1]['token_ids'][x['offset']-1]
 assert t['positions'][67].shape==(3,4,1)
 changed=[]
 basecache=torch.load(base/'cache.pt',map_location='cpu',weights_only=False)['last_position']['67']
 for l,kv in t['cache'][67].items():
  for k,v in kv.items():assert torch.equal(v[[0,2,3]],basecache[l][k][[0,2,3]])
  if any(not torch.equal(v[1],basecache[l][k][1]) for k,v in kv.items()):changed.append(int(l))
 donor=torch.load(rec['donor']['path'],map_location='cpu',weights_only=True) if 'donor' in rec else None
 for l,h in t['history'].items():
  rr=t['restore'][l]
  for kind in ['key','value']:
   assert torch.equal(rr['restored_'+kind],h[kind]) and rr['all_history_restored']
   assert torch.equal(rr['applied_'+kind][[0,2,3]],h[kind][[0,2,3]])
   if rec['cache_patch']:assert torch.equal(rr['applied_'+kind][1],donor['tensors']['history'][l][kind][1])
   else:assert torch.equal(rr['applied_'+kind],h[kind])
 wd=torch.load(rec['wrapper_donor']['path'],map_location='cpu',weights_only=True) if rec.get('wrapper_patch') else None
 for l,h in t['history'].items():
  rr=t['restore'][l]
  for kind in ['key','value']:
   k='wrapper_'+kind
   assert torch.equal(rr['restored_'+k],h[k])
   assert torch.equal(rr['applied_'+k][[0,2,3]],h[k][[0,2,3]])
   expect=wd['tensors']['history'][l][k][1] if wd else h[k][1]
   assert torch.equal(rr['applied_'+k][1],expect)
 if rec['residual']:
  rd=torch.load(rec.get('residual_donor',rec['donor'])['path'],map_location='cpu',weights_only=True)
  assert torch.equal(t['residual'][67]['13']['after'][1],rd['tensors']['residual'][67]['13']['before'][1])
 if name in ['extract-S','extract-F','native-F','self-F']:
  oldlog=torch.load(base/'logits.pt',map_location='cpu',weights_only=False)['raw_logits'][67]
  oldstate=torch.load(base/'states.pt',map_location='cpu',weights_only=False)['captures']['67']
  assert torch.equal(t['logits'][67],oldlog)
  assert torch.equal(t['head'][67],oldstate['head_before'])
  assert torch.equal(t['residual'][67]['13']['after'],oldstate['layers']['13']['before'])
  assert changed==[]
 if name=='residual-only':
  oldr=P/'stage2/runtime/residual-S-to-F-layer13'
  prior=read(oldr/'raw.json')['rows'];assert all(a['token_ids']==b['token_ids'] for a,b in zip(raw['rows'],prior))
  oldlog=torch.load(oldr/'logits.pt',map_location='cpu',weights_only=False)['raw_logits'][67]
  assert torch.equal(t['logits'][67],oldlog)
 v=t['logits'][67][1];assert int(v.argmax())==rec['fork']['argmax']
 checks.append(dict(condition=name,fork=rec['fork'],changed_current_KV_layers=changed,history_restored=True,
                    target_tokens=len(raw['rows'][1]['token_ids']),stop=raw['rows'][1]['stop'],forwards=rec['model_forwards'],receipt=bind(r/'receipt.json')))
out=R/('verify-'+'-'.join(sys.argv[1:])+'.json');out.write_text(json.dumps(dict(status='passed',checks=checks),indent=2)+'\n')
print(json.dumps([dict(condition=c['condition'],argmax=c['fork']['argmax'],margin=c['fork']['coord131_minus_coord0'],tokens=c['target_tokens'],stop=c['stop'],changed_current_KV_layers=c['changed_current_KV_layers']) for c in checks]))
