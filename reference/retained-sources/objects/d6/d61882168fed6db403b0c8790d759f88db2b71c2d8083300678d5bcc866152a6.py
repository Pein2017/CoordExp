import json,importlib.util,collections,argparse
from pathlib import Path
import numpy as np
R=Path(__file__).parent
s=importlib.util.spec_from_file_location('frozen_score',R.parent/'2026-09-16-endpoint-loop-natural-readout-norm/reduce.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def main(partial=False,output=None):
 p=json.loads((R/'panel.json').read_text());cells={};receipts=[];events=[];shadow={}
 for condition in p['conditions']:
  byimage={};shadow[condition]=collections.Counter()
  for group in p['groups']:
   d=R/'runtime'/condition/group['key']
   if not (d/'receipt.json').exists():continue
   rec=json.loads((d/'receipt.json').read_text())
   if rec['status']!='candidate_complete':continue
   assert m.bind(d/'raw.json')==rec['raw'];raw=json.loads((d/'raw.json').read_text());trace=json.loads((d/'trace.json').read_text())['steps'];receipts.append(rec)
   for j,(x,c) in enumerate(zip(raw['rows'],group['cases'])):
    iid=str(x['image_id']);assert [step['chosen'][j] for step in trace[:len(x['token_ids'])]]==x['token_ids'];views={}
    for step in trace[:len(x['token_ids'])]:
     raw=step['raw_winners'][j];chosen=step['chosen'][j];shadow[condition]['active_steps']+=1;shadow[condition]['winner_changes']+=int(raw!=chosen);shadow[condition]['family_switches']+=int((151670<=raw<=152669)!=(151670<=chosen<=152669));shadow[condition]['EOS_choice_changes']+=int((raw==151645)!=(chosen==151645))
    for view,bank in [('refined',p['refined_banks'].get(iid)),('sentinel',p['sentinel_banks'].get(iid))]:
     if bank is None:continue
     z=m.score(x,c,bank);z['same_category']=[m.match._ledger_image([o for o in bank if o['normalized_description']==label],[v for v in z['valid_predictions'] if v['description'].strip().lower()==label],threshold=.5) for label in sorted({o['normalized_description'] for o in bank})]
     z['iou80']=m.match._ledger_image(bank,z['valid_predictions'],threshold=.8)
     # Separate label-compatible matches, not replacement of the primary class-agnostic ledger.
     z['exact_valid_repeats']=sum(v-1 for v in collections.Counter((v['description'],tuple(v['coord_bins_1000'])) for v in z['valid_predictions']).values())
     views[view]=z
    byimage[iid]=dict(group=group['key'],batch_index=j,raw=m.bind(d/'raw.json'),views=views)
  cells[condition]=byimage
 if not partial:assert all(len(v)==145 for v in cells.values()),{k:len(v) for k,v in cells.items()}
 summaries={};comparisons={}
 for label,ids,view in [('human13',[str(x) for x in p['refined_source_order'][:13]],'refined'),('refined5',[str(x) for x in p['refined_source_order'][13:]],'refined'),('sentinel',list(p['sentinel_banks']),'sentinel')]:
  summaries[label]={}
  for condition,cc in cells.items():
   eligible=[cc[i]['views'][view] for i in ids if i in cc];totals=collections.Counter()
   for v in eligible:
    totals.update({k:n for k,n in v['burden'].items() if isinstance(n,(int,float))});totals['matches']+=v['matches']['matched_count'];totals['FN']+=len(v['matches']['missing_owner_ids']);totals['tokens']+=v['token_count'];totals['exact_valid_repeats']+=v['exact_valid_repeats'];totals['iou80_matches']+=v['iou80']['matched_count'];totals['same_category_matches']+=sum(z['matched_count'] for z in v['same_category']);totals['label_compatible_matches']+=len(v['matches']['label_string_compatible_matches'])
   summaries[label][condition]=dict(images=len(eligible),**dict(totals))
  comparisons[label]={}
  for model,left,right in [('tied','tied-original','tied-normalized'),('untied','untied-original','untied-normalized'),('package_original','tied-original','untied-original'),('package_normalized','tied-normalized','untied-normalized')]:
   a=cells[left];b=cells[right];pairs=[]
   for iid in ids:
    if iid not in a or iid not in b:continue
    x=a[iid]['views'][view];y=b[iid]['views'][view];old=set(x['matches']['covered_owner_ids']);new=set(y['matches']['covered_owner_ids']);healthy=x['burden']['eos']==1 and x['burden']['invalid']==0 and x['burden']['malformed']==0 and x['burden']['strict_valid_repeats']==0
    bank=p['refined_banks'][iid] if view=='refined' else p['sentinel_banks'][iid];border={o['owner_id'] for o in bank if any(v in [0,999] for v in o['reference_coord_bins_1000'])}
    pairs.append(dict(image_id=int(iid),gain=sorted(new-old),loss=sorted(old-new),retained=sorted(old&new),match_delta=len(new)-len(old),baseline_healthy=healthy,legal_border_baseline=sorted(old&border),legal_border_retained=sorted(old&new&border),legal_border_lost=sorted((old-new)&border)))
   item=dict(left=left,right=right,pairs=pairs)
   if label=='sentinel' and len(pairs)==128:
    values=np.array([x['match_delta'] for x in pairs]);rng=np.random.default_rng(19);boot=values[rng.integers(0,128,(10000,128))].sum(axis=1);item['paired_total_delta_95percentile_interval']=np.quantile(boot,[.025,.975]).tolist()
   comparisons[label][model]=item
 cost=dict(forwards=sum(x['model_forwards'] for x in receipts),gpu_seconds=sum(x['elapsed_seconds'] for x in receipts),vision_forwards=sum(x['vision_forwards'] for x in receipts),active_tokens=sum(x['active_tokens'] for x in receipts),padded_token_work=sum(x['padded_token_work'] for x in receipts),completed_batches=len(receipts),allocated_gpu_seconds_completed_workers=sum(json.loads(f.read_text())['elapsed_seconds'] for f in R.glob('worker-*.json')))
 out=dict(status='partial' if partial else 'candidate',cells=cells,summaries=summaries,comparisons=comparisons,shadow={k:dict(v) for k,v in shadow.items()},cost=cost)
 (output or R/('partial-reduction.json' if partial else 'reduction.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(cost=cost,summaries=summaries),indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--partial',action='store_true');a.add_argument('--output',type=Path);x=a.parse_args();main(x.partial,x.output)
