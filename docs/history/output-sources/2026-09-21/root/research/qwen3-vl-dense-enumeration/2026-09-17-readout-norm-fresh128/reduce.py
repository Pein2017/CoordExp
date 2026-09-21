"""Independent saved-output reduction; never invokes a model."""
import collections, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
OLD=R.parent/'2026-09-16-endpoint-loop-natural-readout-norm'
spec=importlib.util.spec_from_file_location('accepted_norm_consumer',OLD/'reduce.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read=lambda p:json.loads(Path(p).read_text())
bind=old.bind

def interval(values):
 a=np.asarray(values,dtype=float);rng=np.random.default_rng(19)
 means=np.array([a[rng.integers(0,len(a),len(a))].mean() for _ in range(10000)])
 return dict(mean=float(a.mean()),mean_ci95=np.quantile(means,[.025,.975]).tolist(),sum=float(a.sum()),n=len(a),unit='image',method='paired image bootstrap10000 seed19')

def shadow_stats(trace,raw):
 assert len(trace)==len(raw['token_ids'])
 assert [x['transformed_argmax'] for x in trace]==raw['token_ids']
 disagreements=[x for x in trace if x['original_full_vocab_argmax']!=x['transformed_argmax']]
 return dict(active_steps=len(trace),disagreements=len(disagreements),first=disagreements[0] if disagreements else None,last=disagreements[-1] if disagreements else None,roles=dict(collections.Counter(x['role'] for x in disagreements)),transitions=dict(collections.Counter(('coordinate' if 151670<=x['original_full_vocab_argmax']<152670 else 'noncoordinate')+'->'+('coordinate' if 151670<=x['transformed_argmax']<152670 else 'noncoordinate') for x in disagreements)),single_action_literal_certificate=len(disagreements)==1,no_intervention_needed_for_recorded_path=len(disagreements)==0,unchanged_tail_steps=len(trace)-(disagreements[-1]['offset']+1) if disagreements else len(trace),scope='exact recorded suffix under deterministic same execution; capped horizon is not EOS or physical coverage guarantee')

def main(fresh_only=False):
 p=read(R/'panel.json');images={};receipts=[]
 for g in p['groups']:
  if fresh_only and g['cohort']!='fresh':continue
  root=R/'runtime'/('group-'+g['key'])
  rec={k:read(root/k/'receipt.json') for k in ['O','N']};raw={k:read(root/k/'raw.json') for k in rec}
  for k,r in rec.items():
   assert r['status']=='candidate_complete',(g['key'],k,r['status'])
   assert r['panel']==bind(R/'panel.json')
   assert r['producer']['sha256']==p['producer']['sha256'] and Path(r['producer']['path']).resolve()==Path(p['producer']['path']).resolve()
   assert r['raw']==bind(root/k/'raw.json')
   receipts.append(dict(path=str(root/k/'receipt.json'),**r))
  for key in ['input_identity','prefill_mrope_sha256','readout','generate_settings']:
   assert rec['O'][key]==rec['N'][key],(g['key'],key)
  for j,case in enumerate(g['cases']):
   iid=case['input_record']['image_id']
   if iid not in g['focus_ids']:continue
   a=raw['O']['rows'][j];b=raw['N']['rows'][j]
   assert a['image_id']==b['image_id']==iid
   O=old.score(a,case,p['banks'][str(iid)]);N=old.score(b,case,p['banks'][str(iid)])
   oldids=set(O['matches']['covered_owner_ids']);newids=set(N['matches']['covered_owner_ids'])
   first=next((i for i,(x,y) in enumerate(zip(a['token_ids'],b['token_ids'])) if x!=y),None)
   if first is None and len(a['token_ids'])!=len(b['token_ids']):first=min(len(a['token_ids']),len(b['token_ids']))
   border={str(x['owner_id']) for x in p['banks'][str(iid)] if any(v in (0,999) for v in x['reference_coord_bins_1000'])}
   healthy=O['stop']=='im_end' and all(O['burden'][k]==0 for k in ['invalid','malformed','strict_valid_repeats'])
   images[str(iid)]=dict(cohort=g['cohort'],group=g['key'],batch_index=j,baseline=O,treated=N,G=sorted(newids-oldids),L=sorted(oldids-newids),retained=sorted(oldids&newids),delta=len(newids)-len(oldids),baseline_structurally_healthy=healthy,border_baseline=sorted(oldids&border),border_losses=sorted((oldids-newids)&border),border_gains=sorted((newids-oldids)&border),shadow=shadow_stats(rec['N']['shadow']['samples'][str(iid)]['steps'],b),first_policy_divergence=first,raw_bindings={k:bind(root/k/'raw.json') for k in rec})
  # Trace schema is checked at integration; no model rerun required.
 fresh={k:v for k,v in images.items() if v['cohort']=='fresh'};assert len(fresh)==128
 def aggregate(items):
  v=list(items)
  return dict(images=len(v),known_bank=sum(x['baseline']['matches']['target_count'] for x in v),O_matches=sum(x['baseline']['matches']['matched_count'] for x in v),N_matches=sum(x['treated']['matches']['matched_count'] for x in v),G=sum(len(x['G']) for x in v),L=sum(len(x['L']) for x in v),improved=sum(x['delta']>0 for x in v),worse=sum(x['delta']<0 for x in v),unchanged=sum(x['delta']==0 for x in v),paired=interval([x['delta'] for x in v]) if v else None,border_losses=sum(len(x['border_losses']) for x in v),debt={arm:{k:sum(x[arm]['burden'][k] for x in v) for k in ['invalid','malformed','strict_valid_repeats','literal_repeats','literal_invalid_repeats','unknown','cap','eos']} for arm in ['baseline','treated']},new_debt_images={k:[iid for iid,x in fresh.items() if x in v and x['baseline']['burden'][k]==0 and x['treated']['burden'][k]>0] for k in ['cap','invalid','malformed','strict_valid_repeats']})
 result=dict(status='candidate',panel=bind(R/'panel.json'),consumer=bind(__file__),accepted_score_consumer=bind(OLD/'reduce.py'),images=images,fresh=aggregate(fresh.values()),healthy=aggregate(x for x in fresh.values() if x['baseline_structurally_healthy']),unhealthy=aggregate(x for x in fresh.values() if not x['baseline_structurally_healthy']),diagnostic=aggregate(x for x in images.values() if x['cohort']=='diagnostic'),cost=dict(batch_executions=len(receipts),model_forwards=sum(x['model_forwards'] for x in receipts),generation_gpu_seconds=sum(x['elapsed_seconds'] for x in receipts)),receipts=[bind(x['path']) for x in receipts])
 (R/('fresh-result.json' if fresh_only else 'result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['fresh','healthy','unhealthy']},indent=2))
if __name__=='__main__':
 import sys
 main('--fresh-only' in sys.argv)
