"""CPU readback of bounded native decision evidence; no model calls."""
import json,hashlib
from pathlib import Path
import torch
R=Path(__file__).resolve().parent
p=json.loads((R/'panel.json').read_text());result={}
for c in p['cases']:
 iid=str(c['image_id']);cells={};nr=R/f'runtime/{iid}/native/raw.json'
 if not nr.exists():continue
 original=json.loads(nr.read_text())['rows'][c['target_position']]['token_ids']
 for arm in c['arms']:
  folder=R/f'runtime/{iid}/{arm}'
  if not (folder/'raw.json').exists():continue
  raw=json.loads((folder/'raw.json').read_text());tokens=raw['rows'][c['target_position']]['token_ids'];t=torch.load(folder/'sparse-logits.pt',map_location='cpu',weights_only=True);slots=raw['complete_row_native_logprobs'];x=dict(complete_row_logprob_sum=slots['sum'],complete_row_logprob_mean=slots['mean'],slots=slots['slots'],captured=[])
  for off in t['offsets']:
   logits=t['logits'][off];top=torch.topk(logits,2);h=t['lm_head_input'][off]
   x['captured'].append(dict(offset=off,history_sha256=hashlib.sha256(json.dumps(tokens[:off],separators=(',',':')).encode()).hexdigest(),winner=int(top.indices[0]),full_vocab_top2_margin=float(top.values[0]-top.values[1]),observed_token=tokens[off],lm_head_shape=list(h.shape),finite=bool(torch.isfinite(logits).all() and torch.isfinite(h).all())))
  changes=c['arms'][arm]['changed_offsets']
  if changes:
   first=min(changes);assert tokens[:first]==original[:first];logits=t['logits'][first];a=original[first];b=tokens[first];lp=torch.log_softmax(logits.float(),-1)
   x['first_fork']=dict(offset=first,identical_supplied_history=True,original_token=a,supplied_token=b,original_rank=1+int((logits>logits[a]).sum()),supplied_rank=1+int((logits>logits[b]).sum()),supplied_minus_original_logprob=float(lp[b]-lp[a]),scope='same-history local candidate decision; complete-row scores after fork use candidate-specific histories')
  cells[arm]=x
 result[iid]=cells
(R/'decision-scores.json').write_text(json.dumps(dict(status='candidate',images=result,scope='local fork and full-row conditional likelihood; not owner credit or proof of reinforcement'),indent=2)+'\n')
print(json.dumps({i:{a:{k:v for k,v in z.items() if k in ['complete_row_logprob_sum','complete_row_logprob_mean','first_fork']} for a,z in x.items()} for i,x in result.items()},indent=2))
