import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;OLD=R.parent/'2026-09-16-corner-loop-bridge-factorial';Y2=R.parent/'2026-09-16-corner-loop-one-bin-control'
read=lambda p:json.loads(p.read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
p=read(R/'panel.json');d=read(R/'result.json');r=read(R/'runtime/Y1_998/receipt.json');raw=read(R/'runtime/Y1_998/raw.json')
for b in p['sources']+d['bindings']:assert bind(Path(b['path']))==b
assert (R/'run.exit').read_text().strip()=='0' and r['status']=='candidate_complete' and not r['mechanical_forks']
assert r['producer']==bind(R/'producer.py')==p['derivative']['executed']
c00=read(OLD/'runtime/C00/raw.json'); y2=read(Y2/'runtime/Y998/raw.json')
assert raw['rows'][1]['token_ids'][:1224]==c00['rows'][1]['token_ids'][:1224]
delta=[dict(action_offset=1224+i,old=b,new=a) for i,(a,b) in enumerate(zip(raw['supplied137_token_ids'],c00['supplied137_token_ids'])) if a!=b]
assert delta==[dict(action_offset=1229,old=152669,new=152668)]
for i in [0,2,3]:assert raw['rows'][i]==c00['rows'][i]==y2['rows'][i]
assert [(x['action_offset'],x['supplied_token']) for x in r['seam_logits'] if x['supplied_token'] is not None]==list(enumerate(p['cells']['Y1_998']['tokens'],1224))
for x in r['seam_logits']:
 if x['action_offset']<=1229:assert x['top5'][0]['id']==x['original_token']
 if x['action_offset']>=1233:assert x['supplied_token'] is None
seams={}
for name,path in [('C00',OLD/'runtime/C00/receipt.json'),('Y998',Y2/'runtime/Y998/receipt.json'),('Y1_998',R/'runtime/Y1_998/receipt.json')]:
 seams[name]=[{ 'offset':x['action_offset'],'argmax':x['top5'][0]['id'],'text':x['top5'][0]['text'],'margin':x['top5'][0]['logit']-x['top5'][1]['logit']} for x in read(path)['seam_logits'] if x['action_offset']>=1233]
identity={name:raw['free_token_ids']==v['free_token_ids'] for name,v in [('C00',c00),('Y998',y2)]}
live=[]
for pid in [r['pid'],read(R/'launch.json')['pid']]:
 f=Path(f'/proc/{pid}/cmdline')
 if f.exists() and str(R).encode() in f.read_bytes():live.append(pid)
assert not live
edit=dict(status='verified',single_literal_delta=delta,common_prefix_tokens=1224,supplied_tokens=9,mask=dict(batch_position=1,start=1224,end_exclusive=1233,subsequent_logits_unrestricted=True),all_companions_exact=True,free_suffix_exact=identity,seam_comparison=seams,live_owned_jobs=live)
(R/'edit-receipt.json').write_text(json.dumps(edit,indent=2)+'\n')
print(json.dumps(dict(suffix_identity=identity,seams={k:v[:6] for k,v in seams.items()},burden=d['cell']['burden'],coverage=d['cell']['joint_match_excluding_supplied137']['matched_count'],new=d['cell']['new_covered_owner_ids'],recurrence=d['cell']['later_recurrence']),indent=2))
