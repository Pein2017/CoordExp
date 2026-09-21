"""CPU-only reconstruction and native-state accounting; no model calls."""
import json,hashlib,collections
from pathlib import Path
import torch
from probes.training_set_completion.untied_readout import reduce_capture
R=Path(__file__).parent
torch.set_num_threads(2)
rows=[];checks=[];layer_norms=collections.defaultdict(list)
for model in ['tied','untied']:
 p=R/'native-captures'/f'{model}-original';receipt=json.loads((p/'receipt.json').read_text());weights=torch.load(p/'weights.pt',map_location='cpu',weights_only=False)
 for c in receipt['captures']:
  f=Path(c['tensors']['path']);assert hashlib.sha256(f.read_bytes()).hexdigest()==c['tensors']['sha256']
  packet=torch.load(f,map_location='cpu',weights_only=False);packet.update(weights);r=reduce_capture(packet);saved=torch.load(str(f)+'.reduced.pt',map_location='cpu',weights_only=False)
  for k,v in r.items():
   if isinstance(v,torch.Tensor):assert torch.equal(v,saved[k]),k
   elif isinstance(v,float):assert abs(v-saved[k])<=1e-12,k
   else:assert v==saved[k],k
  
  for key,value in packet.items():
   if key.startswith('layer'):layer_norms[(model,key)].append(float(value.float().norm()))
  e=c['event'];ids=packet['coordinate_ids'];z=packet['logits'][ids].double();n=r['immediate_readout_counterfactual_coordinate_logits'].double();raw=int(z.argmax());norm=int(n.argmax());best=int(z[1:-1].argmax())+1
  full=packet['logits'].clone();full[ids]=n.to(full.dtype)
  coordinate_selected=e['selected_token'] in ids.tolist();endpoint=raw in (0,999)
  margin=r['endpoint_margins'][str(raw)] if endpoint else None
  rows.append(dict(model=model,image=e['image_key'],offset=e['offset'],role=e['role'],recurrence_candidate=e['recurrence_candidate'],coordinate_selected=coordinate_selected,raw_coordinate_winner=raw,normalized_coordinate_winner=norm,endpoint_win=endpoint,endpoint_survives=endpoint and norm==raw,raw_full_winner=int(packet['logits'].argmax()),normalized_full_winner=int(full.argmax()),native_margin=c['native_margin'],counterfactual_margin=float(n.topk(2).values[0]-n.topk(2).values[1]),rank_claim_admissible=c['rank_change_claim_admissible'],endpoint_margin=margin,affine_r2=float(1-r['residual'].square().sum()/((z-z.mean()).square().sum())),coordinate_reconstruction_error=r['coordinate_reconstruction_max_abs'],final_norm_error=r['norm_reconstruction_max_abs'],parity_error=c['top2_max_abs_error'],tensor=str(f)))
summary={}
for model in ['tied','untied']:
 allrows=[x for x in rows if x['model']==model];rr=[x for x in allrows if x['coordinate_selected']];ep=[x for x in rr if x['endpoint_win']]
 summary[model]=dict(events=len(allrows),coordinate_decisions=len(rr),endpoint_coordinate_wins=len(ep),same_endpoint_after_equal_norm=sum(x['endpoint_survives'] for x in ep),endpoint_to_interior=sum(x['normalized_coordinate_winner'] not in (0,999) for x in ep),all_coordinate_winner_changes=sum(x['raw_coordinate_winner']!=x['normalized_coordinate_winner'] for x in rr),slope_opposes_winning_endpoint=sum(x['endpoint_margin']['slope_contribution']<0 for x in ep),max_affine_r2=max(x['affine_r2'] for x in rr),median_affine_r2=float(torch.tensor([x['affine_r2'] for x in rr]).median()),max_coordinate_reconstruction_error=max(x['coordinate_reconstruction_error'] for x in allrows),max_norm_error=max(x['final_norm_error'] for x in allrows),max_parity_error=max(x['parity_error'] for x in allrows),native_near_ties=sum(not x['rank_claim_admissible'] for x in allrows))
(R/'layer-summary.json').write_text(json.dumps({model:{key:dict(n=len(values),min=min(values),median=float(torch.tensor(values).median()),max=max(values)) for (m,key),values in layer_norms.items() if m==model} for model in ['tied','untied']},indent=2)+'\n')
(R/'summary.json').write_text(json.dumps(dict(status='candidate',models=summary,events=rows,verification='442 saved tensors hashed and CPU reductions reproduced; tensors exact, scalars <=1e-12',limits='Selected correlated slots, not population trials; recurrence is a geometric proxy unless separately reviewed. Counterfactual only at same native h.'),indent=2)+'\n')
print(json.dumps(summary,indent=2))
