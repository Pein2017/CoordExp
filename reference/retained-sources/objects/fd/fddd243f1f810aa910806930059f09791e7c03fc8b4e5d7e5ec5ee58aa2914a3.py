"""Static sign23-support controls; all generated rows are free."""
import argparse,json,hashlib,itertools
from pathlib import Path
from probes.training_set_completion.numerical_feedback.metrics import release_metrics
from probes.training_set_completion.numerical_feedback.select import rows,same

def bind(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def reduce(root):
 plan=json.loads((root/'execution-plan.json').read_text());boundaries={b['id']:b for b in json.loads(Path(plan['selection']).read_text())['boundaries']};cells=[];traces={};bindings=[bind(root/'execution-plan.json')]
 for c in plan['cells']:
  ps=list((root/'runtime').rglob(c['id']+'/release.json'));assert len(ps)==1,(c['id'],ps)
  p=ps[0];r=json.loads(p.read_text());t=r['target']['token_ids'];m=release_metrics(t,boundaries[c['boundary_id']]['source_row']);rr=rows(t);pairs=list(itertools.combinations(range(len(rr)),2));m['all_pairs_exact']=sum(same(rr[i],rr[j],0) for i,j in pairs);m['all_pairs_near']=sum(same(rr[i],rr[j],8) for i,j in pairs);m['all_pairs_denominator']=len(pairs);stop='eos' if m['eos'] else 'rows' if m['complete_rows']==32 else 'cap';assert stop==r['target']['stop']['reason'];assert len(t)<=512 and m['complete_rows']<=32;assert stop!='cap' or len(t)==512
  control_equal=None
  if 'control_source' in c:
   cp=Path(c['control_source']['path']);assert bind(cp)['sha256']==c['control_source']['sha256'];control_equal=t==json.loads(cp.read_text())['target']['token_ids'];assert control_equal,c['id']
  cells.append(dict(id=c['id'],boundary_id=c['boundary_id'],model=c['model'],image_id=boundaries[c['boundary_id']]['image_id'],policy=c['policy'],tokens=t,metrics=m,stop=stop,control_equal=control_equal));traces[c['id']]=r['trace']['steps'];bindings.append(bind(p))
 comparisons=[];first_forks=[]
 for bid in boundaries:
  cc={c['policy']:c for c in cells if c['boundary_id']==bid};original=cc['original'];full=cc['sign23']
  for policy in plan['policies'][1:]:
   c=cc[policy];t=c['tokens'];o=original['tokens'];fork=next((i for i,(a,b) in enumerate(zip(o,t)) if a!=b),None)
   if fork is None and len(o)!=len(t):fork=min(len(o),len(t))
   shadow=traces[original['id']][fork]['operators'] if fork is not None and fork<len(o) else None
   if shadow:assert shadow[policy]['top2'][0]['token_id']==t[fork]
   first_forks.append(dict(boundary_id=bid,policy=policy,offset=fork,same_original_state=shadow))
   comparisons.append(dict(boundary_id=bid,policy=policy,minus_original={k:c['metrics'][k]-original['metrics'][k] for k in ['longest_exact_run','longest_near_run','invalid_rows','malformed_openers','complete_rows']},minus_sign23={k:c['metrics'][k]-full['metrics'][k] for k in ['longest_exact_run','longest_near_run','invalid_rows','malformed_openers','complete_rows']},stop=c['stop'],original_stop=original['stop'],sign23_stop=full['stop'],tokens_equal_original=t==o,tokens_equal_sign23=t==full['tokens']))
 summary=[]
 for model in ['tied','untied']:
  for policy in plan['policies']:
   cs=[c for c in cells if c['model']==model and c['policy']==policy];ds=[x for x in comparisons if x['policy']==policy and boundaries[x['boundary_id']]['model']==model]
   summary.append(dict(model=model,policy=policy,n=len(cs),shorter_near_run_than_original=sum(x['minus_original']['longest_near_run']<0 for x in ds),longer_near_run_than_original=sum(x['minus_original']['longest_near_run']>0 for x in ds),invalid_rows=sum(c['metrics']['invalid_rows'] for c in cs),malformed_openers=sum(c['metrics']['malformed_openers'] for c in cs),eos=sum(c['stop']=='eos' for c in cs),native_return=sum(bool(c['metrics']['native_near_return_rows']) for c in cs),alternate_repeat=sum(bool(c['metrics']['alternate_repeat_starts']) for c in cs)))
 return dict(status='candidate',cells=cells,comparisons=comparisons,first_forks=first_forks,summary=summary,bindings=bindings,limits='Static supportablation at original11failureprefixes. Allrowsfree; magnitudeintentionallydiffers. Allpairs includeinvalid literalboxes; numericalrecurrence is notphysicalidentity. No weightedqualityscore or physicalrecovery claim.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();d=reduce(a.root);a.out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(dict(cells=len(d['cells']),comparisons=len(d['comparisons']))))
