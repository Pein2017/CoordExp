"""Independent CPU reduction of frozen numerical-feedback states and free releases."""
import json,hashlib,statistics,argparse,collections
from pathlib import Path
import torch
from probes.training_set_completion.numerical_feedback.metrics import release_metrics,crossed_margin
R=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback')
def binding(p):p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
def verify(b):assert binding(b['path'])==b,b['path']
def reduce(output):
 torch.set_num_threads(2);selection=json.loads((R/'selection.json').read_text());boundaries={x['id']:x for x in selection['boundaries']};seen={};cells=[];contrasts=[];duplicates=[];source_cache={}
 for f in sorted(R.rglob('paired.json')):
  info=json.loads(f.read_text())
  if info.get('status')!='candidate_complete':continue
  verify(info['replay']);verify(info['release']);p=torch.load(info['replay']['path'],map_location='cpu',weights_only=False);b=p['boundary'];c=p['candidate'];key=(b['id'],c['role'],c['kind'],c['value']);assert b==boundaries[b['id']]
  if key in seen:
   duplicates.append(dict(key=key,paths=[seen[key],str(f)]));continue
  seen[key]=str(f);assert p['parity']['passed'];a=c['original_value'];v=c['value'];fixed=p['fixed_suffix'];modified=p['candidate_fixed_suffix'];assert [x['token_id'] for x in fixed['tokens']]==[x['token_id'] for x in modified['tokens']]
  for suffix in [fixed,modified]:assert abs(sum(x['logprob'] for x in suffix['tokens'])-suffix['logprob_sum'])<1e-10
  for keyslot,old in p['native_slots'].items():
   if c.get('baseline_mode')=='native_noop':continue
   new=p['candidate_slots'][keyslot];offset=old['offset'];target=next(t for t in b['target_slots'] if t['offset']==offset);z=torch.tensor(old['coordinate_logits'],dtype=torch.float64);zz=torch.tensor(new['coordinate_logits'],dtype=torch.float64);delta=torch.tensor(new['coordinate_logit_diff'],dtype=torch.float64);assert (zz-z-delta).abs().max()<1e-8
   for mode,k in [('original','coordinate_logits'),('equalnorm_shadow','normalized_coordinate_logits')]:
    K=crossed_margin(old[k],new[k],a,v);contrasts.append(dict(boundary_id=b['id'],model=b['model'],image_id=b['image_id'],episode=b['kind'],stratum=b['episode_stratum'],candidate_kind=c['kind'],source_role=c['role'],target_role=target['role'],same_role=c['role']==target['role'],delay=target['row_delay'],readout=mode,a=a,b=v,K=K,above_conservative_logit_bound=abs(K)>.0008,native_winner=int(torch.tensor(old[k]).argmax()),modified_winner=int(torch.tensor(new[k]).argmax()),native_a_minus_b=old[k][a]-old[k][v],modified_a_minus_b=new[k][a]-new[k][v],eos_probability_before=old['eos_probability'],eos_probability_after=new['eos_probability'],validity_preserved=c['validity_preserved'],ordering_preserved=c['ordering_preserved']))
  manifest=json.loads(Path(info['manifest']['path']).read_text());entry=next(e for e in manifest['boundaries'] if e['boundary']['id']==b['id']);capture_path=entry['source_capture']['path']
  if capture_path not in source_cache:
   verify(entry['source_capture']);source_cache[capture_path]=torch.load(capture_path,map_location='cpu',weights_only=False)
  slot=source_cache[capture_path]['slots'][f"source_{c['role']}@{c['source_offset']}"];candidate_logprob=slot['coordinate_logits'][v]-slot['native_logit']+slot['native_logprob']
  release=json.loads(Path(info['release']['path']).read_text());tokens=release['target']['token_ids'];assert release['mutation']['old_token_id']==151670+a and release['mutation']['new_token_id']==151670+v
  metrics=release_metrics(tokens,b['source_row'],c['role'],v);assert len(tokens)<=512 and metrics['complete_rows']<=32
  cells.append(dict(boundary_id=b['id'],model=b['model'],image_id=b['image_id'],episode=b['kind'],candidate=c,candidate_native_full_vocab_logprob=candidate_logprob,free=metrics,stop=release['stop'],fixed_suffix_native_logprob=fixed['logprob_sum'],fixed_suffix_modified_logprob=modified['logprob_sum'],fixed_suffix_delta=modified['logprob_sum']-fixed['logprob_sum'],source=binding(f),release=info['release'],replay=info['replay']))
 summaries=[]
 groups=collections.defaultdict(list)
 for x in contrasts:groups[(x['boundary_id'],x['readout'],x['same_role'],x['delay'])].append(x)
 for (bid,mode,same,delay),xs in sorted(groups.items()):
  values=[x['K'] for x in xs];summaries.append(dict(boundary_id=bid,readout=mode,same_role=same,delay=delay,n=len(xs),median_K=statistics.median(values),min_K=min(values),max_K=max(values),positive_above_bound=sum(x>.0008 for x in values),negative_below_bound=sum(x<-.0008 for x in values),within_bound=sum(abs(x)<=.0008 for x in values)))
 baselines=[]
 for b in boundaries.values():
  tokens=b['native_tokens'][b['source_row']['end']:][:512]
  from probes.training_set_completion.numerical_feedback.select import rows
  rr=rows(tokens)
  if len(rr)>=32:tokens=tokens[:rr[31]['end']]
  if 151645 in tokens:tokens=tokens[:tokens.index(151645)+1]
  baselines.append(dict(boundary_id=b['id'],provenance='reused actual natural suffix; original observed horizon, no new forward',token_ids=tokens,metrics=release_metrics(tokens,b['source_row'])))
 out=dict(baselines=baselines,status='candidate_cpu_reduction',boundaries_selected=len(boundaries),unique_candidate_releases=len(cells),contrasts=contrasts,episode_summaries=summaries,cells=cells,duplicate_provenance=duplicates,selection_exclusions=selection['exclusions'],interpretation='Positive K is relative movement toward substituted numeric value; healthy controls and spatial/order/validity mismatches limit recurrence specificity. Correlated slots are not statistical replicates. Numerical return or escape is not physical owner recovery.')
 output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(cells=len(cells),contrasts=len(contrasts),duplicates=len(duplicates),boundaries=len(boundaries))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=R/'reduction.json');a=p.parse_args();reduce(a.output)
