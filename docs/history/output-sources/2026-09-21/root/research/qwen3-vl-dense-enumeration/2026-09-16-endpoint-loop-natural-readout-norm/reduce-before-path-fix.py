import collections,hashlib,json,re
from pathlib import Path
from src.inference.parsing import parse_compact_object_box_closed
from probes.training_set_completion import paired_evaluation as match,source256_evaluation as metrics
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
PAT=re.compile(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>'+r'<\|coord_(\d+)\|>'*4+r'<\|box_end\|>')
def score(raw,case,bank):
 native=parse_compact_object_box_closed(raw['text'],image_width=case['image_width'],image_height=case['image_height'],row_id=case['row_id'],row_index=case['row_index']).to_artifact_dict();valid,drops=match._matchable_rows_with_geometry_debt({**native,'pred':native['predictions']})
 for x in valid:x['prediction_id']=f"P{x['generated_order']}"
 ledger=match._ledger_image(bank,valid,threshold=.5);owners={str(x['owner_id']):x for x in bank};preds={x['prediction_id']:x for x in valid};ledger['label_string_compatible_matches']=[dict(owner_id=m['reference_owner_id'],prediction_id=m['prediction_id']) for m in ledger['matches'] if owners[m['reference_owner_id']]['description'].strip().lower()==preds[m['prediction_id']]['description'].strip().lower()]
 spans=[dict(row=j+1,description=m[1],box=list(map(int,m.groups()[1:]))) for j,m in enumerate(PAT.finditer(raw['text']))];counts=collections.Counter((x['description'],tuple(x['box'])) for x in spans);seen=set();first_repeat=None;runs=[]
 for x in spans:
  key=(x['description'],tuple(x['box']))
  if key in seen and first_repeat is None:first_repeat=x
  seen.add(key)
  if runs and (runs[-1]['description'],runs[-1]['box'])==(x['description'],x['box']):runs[-1]['length']+=1
  else:runs.append(dict(start_row=x['row'],length=1,description=x['description'],box=x['box']))
 repeats=metrics._strict_repeat_rows(valid);reasons=collections.Counter(x.get('drop_reason',x.get('reason')) for x in drops);coord=[x-151670 for x in raw['token_ids'] if 151670<=x<=152669]
 return dict(matches=ledger,token_count=len(raw['token_ids']),stop=raw['stop'],burden=dict(complete_rows=len(spans),valid=len(valid),invalid=sum(not(x['box'][0]<x['box'][2] and x['box'][1]<x['box'][3]) for x in spans),malformed=sum(n for k,n in reasons.items() if 'geometry' not in str(k) and 'bbox' not in str(k)),drop_reasons=dict(reasons),strict_valid_repeats=len(repeats),literal_repeats=sum(n-1 for n in counts.values()),literal_invalid_repeats=sum(n-1 for (desc,b),n in counts.items() if not(b[0]<b[2] and b[1]<b[3])),unknown=len(ledger['annotation_unmatched_prediction_ids']),cap=int(raw['stop']=='length'),eos=int(raw['stop']=='im_end')),endpoint_occupancy=dict(total=len(coord),zero=coord.count(0),last=coord.count(999),fraction=sum(x in [0,999] for x in coord)/len(coord) if coord else None,per_role={role:dict(total=len(spans),zero=sum(x['box'][j]==0 for x in spans),last=sum(x['box'][j]==999 for x in spans)) for j,role in enumerate(['x1','y1','x2','y2'])}),first_literal_repeat=first_repeat,first_strict_repeat=repeats[0] if repeats else None,longest_exact_run=max(runs,key=lambda x:x['length']) if runs else None,exact_runs=runs,complete_rows=spans,native_parse=native,valid_predictions=valid,drops=drops)
def main():
 p=read(R/'panel.json');assert (R/'run.exit').read_text().strip()=='0'
 for b in p['sources']:assert bind(b['path'])==b,b['path']
 cells={};receipts=[];checks=[]
 for group in p['groups']:
  gkey=group['key'];rs={policy:read(R/f'runtime/{gkey}-{policy}/receipt.json') for policy in ['identity','norm']};raws={policy:read(R/f'runtime/{gkey}-{policy}/raw.json') for policy in rs}
  for policy,rec in rs.items():
   assert rec['status']=='candidate_complete';assert rec['panel']==bind(R/'panel.json') and rec['producer']==bind(R/'producer.py');assert rec['raw']==bind(R/f'runtime/{gkey}-{policy}/raw.json');assert rec['no_parameter_mutation'];receipts.append(rec);checks+=rec['offline_checks']
  for k in ['loaded_identity','batch_shape','prompt_position_sha256','generate_settings','readout']:assert rs['identity'][k]==rs['norm'][k],(gkey,k)
  for b,saved in enumerate(group['rows']):
   base=raws['identity']['rows'][b];assert base['token_ids']==saved['generated_token_ids'] and base['stop']==saved['decode_stop_reason']
   iid=saved['image_id']
   if iid not in p['focus_images']:continue
   treated=raws['norm']['rows'][b];a=score(base,group['cases'][b],p['banks'][str(iid)]);z=score(treated,group['cases'][b],p['banks'][str(iid)]);old=set(a['matches']['covered_owner_ids']);new=set(z['matches']['covered_owner_ids']);first=next((i for i,(x,y) in enumerate(zip(base['token_ids'],treated['token_ids'])) if x!=y),min(len(base['token_ids']),len(treated['token_ids'])) if len(base['token_ids'])!=len(treated['token_ids']) else None);fork=rs['norm']['first_forks'].get(str(iid));assert (fork['offset'] if fork else None)==first
   cells[str(iid)]=dict(group=gkey,batch_position=b,baseline=a,treated=z,gained_owner_ids=sorted(new-old),lost_incumbent_owner_ids=sorted(old-new),retained_incumbent_owner_ids=sorted(old&new),matched_count_delta=len(new)-len(old),first_changed_token=first,first_fork=fork,full_output_exact=base['token_ids']==treated['token_ids'],raw_bindings={policy:bind(R/f'runtime/{gkey}-{policy}/raw.json') for policy in rs})
 assert cells['7116']['baseline']['matches']['matched_count']==4 and cells['7116']['baseline']['stop']=='im_end'
 cost=dict(batch_executions=len(receipts),model_forwards=sum(x['model_forwards'] for x in receipts),vision_forwards=sum(x['vision_forwards'] for x in receipts),allocated_gpu_seconds=sum(x['elapsed_seconds'] for x in receipts),peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in receipts));assert cost['batch_executions']==8 and cost['model_forwards']<=25000 and cost['allocated_gpu_seconds']<=7200;assert len(cells)==4
 result=dict(status='candidate',panel=bind(R/'panel.json'),consumer=bind(Path(__file__)),images=cells,cost=cost,offline_checks=checks,receipt_bindings=[bind(R/f"runtime/{x['group']}-{x['policy']}/receipt.json") for x in receipts],scope='Selected four natural focus images only; companion changes are not scientific cohort expansion')
 (R/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(images={i:dict(baseline_matches=x['baseline']['matches']['matched_count'],treated_matches=x['treated']['matches']['matched_count'],G=x['gained_owner_ids'],L=x['lost_incumbent_owner_ids'],baseline_burden=x['baseline']['burden'],treated_burden=x['treated']['burden'],baseline_tokens=x['baseline']['token_count'],treated_tokens=x['treated']['token_count'],first_fork=x['first_changed_token'],field=x['first_fork']['field'] if x['first_fork'] else None,treated_stop=x['treated']['stop']) for i,x in cells.items()},cost=cost,offline_checks=len(checks)),indent=2))
if __name__=='__main__':main()
