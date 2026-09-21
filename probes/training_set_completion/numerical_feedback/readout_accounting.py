"""Optional saved-state centering arithmetic; no model or new decoding operator."""
from pathlib import Path
import json,hashlib,torch
R=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback');B=R.parent/'2026-09-18-untied-active-readout-geometry'
def main():
 torch.set_num_threads(2);out=[]
 for model in ['tied','untied']:
  p=B/'native-captures'/f'{model}-original';w=torch.load(p/'weights.pt',map_location='cpu',weights_only=False);W=w['coordinate_weights'].double();mu=W.mean(0);n=W.norm(dim=1);alpha=n.median()/n
  receipt=json.loads((p/'receipt.json').read_text())
  for c in receipt['captures']:
   f=Path(c['tensors']['path']);assert hashlib.sha256(f.read_bytes()).hexdigest()==c['tensors']['sha256'];s=torch.load(f,map_location='cpu',weights_only=False);z=s['logits'][w['coordinate_ids']].double();h=s['post_norm'].double();b=float(mu@h);d=z-b;common=(alpha-1)*b;centered=(alpha-1)*d;change=(alpha-1)*z;err=float((change-common-centered).abs().max());assert err<1e-12
   raw=int(z.argmax());norm=int((z*alpha).argmax());comp=norm if norm!=raw else int(z.topk(2).indices[1]);fulltoken=int(s['logits'].argmax());coordinate=fulltoken in w['coordinate_ids'].tolist()
   out.append(dict(model=model,event={k:c['event'][k] for k in ['image_key','row_index','offset','role']},coordinate_decision=coordinate,common_logit=b,raw_winner=raw,equalnorm_winner=norm,comparison=comp,original_margin=float(z[raw]-z[comp]),common_margin_change=float(common[raw]-common[comp]),centered_margin_change=float(centered[raw]-centered[comp]),normalized_margin=float(z[raw]*alpha[raw]-z[comp]*alpha[comp]),reconstruction_max_abs=err,source=c['tensors']))
 summaries={}
 for m in ['tied','untied']:
  rr=[x for x in out if x['model']==m and x['coordinate_decision']];fl=[x for x in rr if x['raw_winner']!=x['equalnorm_winner']]
  summaries[m]=dict(coordinate_decisions=len(rr),flips=len(fl),flips_common_term_alone_reverses_pair=sum(x['original_margin']+x['common_margin_change']<0 for x in fl),flips_centered_term_alone_reverses_pair=sum(x['original_margin']+x['centered_margin_change']<0 for x in fl),max_reconstruction_error=max(x['reconstruction_max_abs'] for x in rr))
 (R/'readout-accounting.json').write_text(json.dumps(dict(status='candidate_optional_CPU',convention='mu=mean of1000 effective output rows; z from actual native logits; d=z-mu dot h',models=summaries,events=out,limits='Pair arithmetic can overlap; no percentage causal credit, no new operator, no history change, no recurrence-origin claim.',model_calls=0),indent=2)+'\n');print(json.dumps(summaries))
if __name__=='__main__':main()
