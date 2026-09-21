import json,math,hashlib
from pathlib import Path
import torch
R=Path(__file__).parent
rows=[];checks=[];receipts=[]
for b in [0,52,30]:
 d=R/'runtime'/str(b);t=json.loads((d/'tree.json').read_text());n=torch.load(d/'nodes.pt',weights_only=True);r=torch.load(d/'root.pt',weights_only=True);s=torch.load(d/'rescored.pt',weights_only=True)
 rec=json.loads((d/'receipt.json').read_text());assert rec['status']=='candidate_complete';receipts.append(rec)
 common=sum(float(r['logits'][i].double().log_softmax(-1)[x['token']]) for i,x in enumerate(t['common_prelude_terms']));forced=float(r['logits'][-1].double().log_softmax(-1)[151670+b])
 for step in t['ledger']:
  for exp in step['expanded']:
   key=','.join(map(str,exp['parent']));z=n[key]['logits'];choices=torch.argsort(z,descending=True,stable=True)[:4].tolist();assert choices==[x['tokens'][-1] for x in exp['children']]
   for ch in exp['children']:assert abs(ch['suffix_logprob']-exp['parent_logprob']-float(z.double().log_softmax(-1)[ch['tokens'][-1]]))<1e-10
  pool=[ch for exp in step['expanded'] for ch in exp['children']]
  if step['depth']>1:pool += [x for x in t['ledger'][step['depth']-2]['kept'] if x['stop']]
  assert sorted(pool,key=lambda x:(-x['suffix_logprob'],tuple(x['tokens'])))[:4]==step['kept']
  assert abs(sum(math.exp(x['suffix_logprob']) for x in step['kept'])+step['cumulative_discarded_mass']-1)<1e-10
 for p in t['paths']:
  vals=s[p['id']]['logits'];total=sum(float(vals[i].double().log_softmax(-1)[token]) for i,token in enumerate(p['row_token_ids']));assert abs(total-p['rescored_full_row_logprob'])<1e-10
  for i,token in enumerate(p['row_token_ids']):
   z=vals[i];top=z.topk(2);other=float(top.values[1] if int(top.indices[0])==token else top.values[0]);saved=p['token_scores'][i];assert saved['rank']==1+int((z>z[token]).sum());assert saved['argmax']==int(z.argmax());assert abs(saved['chosen_minus_best_other']-(float(z[token])-other))<1e-12
  cond=0
  for i,token in enumerate(p['suffix_token_ids']):
   z=n[','.join(map(str,p['suffix_token_ids'][:i]))]['logits'];cond+=float(z.double().log_softmax(-1)[token])
  assert abs(cond-p['suffix_logprob'])<1e-10;assert abs(common+forced+cond-p['full_row_logprob'])<1e-10;assert abs(common+cond-p['full_row_logprob'])>1 # forced-score omission must fail
  p.update(branch=b,box=[x-151670 for x in p['row_token_ids'][5:9]] if p['stop']=='complete' else None);rows.append(p)
 checks.append(dict(branch=b,nodes=len(n),paths=len(t['paths']),mass=t['beam_conditional_retained_mass'],discarded=t['beam_conditional_discarded_mass'],maximum_rescore_error=max(p['rescore_error'] for p in t['paths'])))
out=dict(status='independent_saved_output_passed',paths=rows,checks=checks,forwards=sum(x['model_forwards'] for x in receipts),gpu_seconds=sum(x['elapsed_seconds'] for x in receipts),tensor_bytes=sum(p.stat().st_size for p in R.rglob('*.pt')))
(R/'reduction.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='paths'},indent=2))
for p in rows:print(p['id'],p['box'],p['provenance'],p['full_row_logprob'])
