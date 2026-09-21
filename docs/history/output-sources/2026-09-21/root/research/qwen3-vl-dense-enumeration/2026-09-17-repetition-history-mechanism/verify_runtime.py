"""Independent saved-token/transform checks, without model calls."""
import json,hashlib
from pathlib import Path
import torch
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def bind(p):p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
m=read(R/'runtime-manifest.json');checks=[];seen=set()
for e in m['cells']:
 if e['stage']=='C':continue
 folder=Path(e['reuse_from']) if 'reuse_from' in e else Path(e['output_root'])/str(e['image_id'])/e['mode']
 if folder in seen or not(folder/'raw.json').exists():continue
 seen.add(folder);rec=read(folder/'receipt.json');assert rec['status']=='candidate_complete';assert rec['raw']==bind(folder/'raw.json')
 panel=read(Path(e['reuse_receipt_panel']['path'] if 'reuse_from' in e else e['panel']['path']));c=panel['cases'][0];raw=read(folder/'raw.json');old=read(Path(c['saved_raw']['path']));t=c['target_position'];ids=raw['rows'][t]['token_ids'];pref=c['target_prefix_token_ids'];assert ids[:len(pref)]==pref
 for j,row in enumerate(raw['rows']):
  if j!=t or c.get('native_identity'):assert row['token_ids']==old['rows'][j]['token_ids'] and row['stop']==old['rows'][j]['stop']
 assert raw['input_identity']==read(Path(c['saved_receipt']['path']))['input_identity']
 n=raw['norm_pulse'];steps=n['steps'];start=n['start_offset'];last=n['last_active_offset']
 if steps:
  assert [x['offset'] for x in steps]==list(range(start,last+1))
  assert all(ids[x['offset']]==x['transformed_winner_token_id'] for x in steps)
  ends=ids[start:last+1].count(151649);assert ends==n['complete_rows']
  if n['row_limit'] is not None and n['complete']:assert ends==n['row_limit'] and ids[last]==151649
 captures=torch.load(folder/'pulse-captures.pt',map_location='cpu',weights_only=True)
 co=torch.tensor(panel['coordinate_ids']);fac=torch.load(panel['coefficients']['path'],map_location='cpu',weights_only=True)['factors'];mask=torch.ones(152670,dtype=torch.bool);mask[co]=False
 for offset,v in captures['captures'].items():
  before=v['before_logits'];after=v['after_logits'];assert torch.isfinite(before).all();assert torch.isfinite(v['lm_head_input']).all()
  expected=before.clone()
  if start is not None and last is not None and start<=offset<=last:expected[co]=(before[co].double()*fac).to(before.dtype)
  # Only declared capture windows after forcing are compared here.
  if offset>=len(pref):assert torch.equal(expected,after),(folder,offset)
  if steps and offset in {s['offset'] for s in steps}:
   s=next(s for s in steps if s['offset']==offset);assert int(before.argmax())==s['native_winner_token_id'];assert int(after.argmax())==s['transformed_winner_token_id']
 checks.append(dict(path=str(folder),raw=bind(folder/'raw.json'),receipt=bind(folder/'receipt.json'),captures=len(captures['captures']),model_forwards=rec['model_forwards']))
result=dict(status='candidate_saved_checks_pass',checks=checks,completed_distinct_runs=len(checks),consumer=bind(__file__))
(R/'runtime-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],runs=len(checks),captures=sum(x['captures'] for x in checks))))
