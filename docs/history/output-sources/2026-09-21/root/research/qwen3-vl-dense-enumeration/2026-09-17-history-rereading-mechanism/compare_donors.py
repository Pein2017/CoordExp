import json
from pathlib import Path
import torch
R=Path(__file__).resolve().parent;torch.set_num_threads(2)
a=torch.load(R/'runtime/extract-S/capture.pt',map_location='cpu',weights_only=True)
b=torch.load(R/'runtime/extract-F/capture.pt',map_location='cpu',weights_only=True)
assert a['width']==b['width'] and a['input_identity']==b['input_identity']
assert a['prefix'][:54]==b['prefix'][:54]
assert set(a['tensors']['positions'])==set(b['tensors']['positions'])
assert all(torch.equal(v,b['tensors']['positions'][k]) for k,v in a['tensors']['positions'].items())
rows=[]
for l,x in a['tensors']['history'].items():
 y=b['tensors']['history'][l];out={'layer':int(l)}
 for kind in ['key','value','wrapper_key','wrapper_value']:
  assert torch.equal(x[kind][[0,2,3]],y[kind][[0,2,3]])
  av=x[kind][1];bv=y[kind][1];delta=av-bv
  out[kind]=dict(S_norm=float(av.norm()),F_norm=float(bv.norm()),delta_norm=float(delta.norm()),
      per_position_delta_norm=delta.permute(1,0,2).flatten(1).norm(dim=1).tolist())
 rows.append(out)
d=dict(status='passed',same_positions=True,companions_exact=True,cache_actions=[54,63],wrapper_actions=[63,66],
       notes='Paired post-RoPE K/native V. Wrapper states may differ despite identical token IDs; no wrapper patch or causal claim from norms.',layers=rows)
(R/'donor-comparison.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(dict(status=d['status'],same_positions=True,layers=len(rows),layer14=rows[0])))
