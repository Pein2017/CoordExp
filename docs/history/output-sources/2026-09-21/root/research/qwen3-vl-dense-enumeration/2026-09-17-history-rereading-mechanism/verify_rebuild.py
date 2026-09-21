import json,hashlib
from pathlib import Path
import torch
R=Path(__file__).resolve().parent;torch.set_num_threads(2)
read=lambda p:json.loads(p.read_text())
bind=lambda p:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
names=['native-F','cache-only','rebuild-cache-only']
raw={n:read(R/'runtime'/n/'raw.json') for n in names}
t={n:torch.load(R/'runtime'/n/'capture.pt',map_location='cpu',weights_only=True)['tensors'] for n in names}
for i in range(4):
 assert raw['native-F']['rows'][i]['token_ids']==raw['rebuild-cache-only']['rows'][i]['token_ids']
 assert raw['native-F']['rows'][i]['stop']==raw['rebuild-cache-only']['rows'][i]['stop']
assert raw['cache-only']['rows'][1]['token_ids'][:68]==raw['native-F']['rows'][1]['token_ids'][:68]
assert raw['cache-only']['rows'][1]['token_ids']!=raw['rebuild-cache-only']['rows'][1]['token_ids']
for off,ls in t['native-F']['cache'].items():
 for l,kv in ls.items():
  for k,v in kv.items():assert torch.equal(v,t['rebuild-cache-only']['cache'][off][l][k])
assert torch.equal(t['native-F']['logits'][67],t['rebuild-cache-only']['logits'][67])
assert not torch.equal(t['native-F']['logits'][67],t['cache-only']['logits'][67])
d=dict(status='passed',selected='cache-only',same_emitted_first68=True,rebuild_equals_native_all_four_full_sequences=True,rebuild_current_KV_equals_native_at67_68=True,retained_patch_differs_after114=True,
 claim='History row KV restored immediately and emitted token at67 unchanged; retained current-position KV carries an unhelpful later trajectory difference, removed by native rebuild. This is not useful recovery or general memory/circuit attribution.',
 receipts=[bind(R/'runtime'/n/'receipt.json') for n in names])
(R/'rebuild-verification.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({k:v for k,v in d.items() if k!='receipts'}))
