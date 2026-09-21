import collections,hashlib,json,re
from pathlib import Path
from transformers import AutoTokenizer
from src.inference.parsing import parse_compact_object_box_closed
from probes.training_set_completion import paired_evaluation as match,source256_evaluation as metrics
R=Path(__file__).resolve().parent;OLD=R.parent/'2026-09-16-corner-loop-bridge-factorial'
read=lambda p:json.loads(p.read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
PAT=re.compile(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>'+r'<\|coord_(\d+)\|>'*4+r'<\|box_end\|>')
def score(ids,stop,iid,p,tok):
 text=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False);context={**p['parse_context'],'row_id':f'free_{iid}','row_index':0};native=parse_compact_object_box_closed(text,**context).to_artifact_dict();valid,drops=match._matchable_rows_with_geometry_debt({**native,'pred':native['predictions']})
 for x in valid:x['prediction_id']=f"F{x['generated_order']}"
 banks={'477415':p['original_targets'],str(p['donor']['image_id']):p['donor']['targets']};ledgers={k:match._ledger_image(bank,valid,threshold=.5) for k,bank in banks.items()}
 for k,ledger in ledgers.items():
  owners={str(o['owner_id']):o for o in banks[k]};preds={x['prediction_id']:x for x in valid};ledger['matched_labels']=[dict(prediction_id=m['prediction_id'],owner_id=m['reference_owner_id'],prediction=preds[m['prediction_id']]['description'],owner=owners[m['reference_owner_id']]['description'],class_correct=match._class_correct(target=owners[m['reference_owner_id']],prediction=preds[m['prediction_id']]),label_string_compatible=preds[m['prediction_id']]['description'].strip().lower()==owners[m['reference_owner_id']]['description'].strip().lower()) for m in ledger['matches']]
  ledger['label_string_compatible_count']=sum(x['label_string_compatible'] for x in ledger['matched_labels'])
 spans=[dict(row_1based=j+1,description=m[1],box=list(map(int,m.groups()[1:]))) for j,m in enumerate(PAT.finditer(text))];counter=collections.Counter((x['description'],tuple(x['box'])) for x in spans);coords=[x-151670 for x in ids if 151670<=x<=152669];roles={role:dict(zero=sum(x['box'][j]==0 for x in spans),last=sum(x['box'][j]==999 for x in spans),total=len(spans)) for j,role in enumerate(['x1','y1','x2','y2'])}
 runs=[]
 for x in spans:
  key=(x['description'],x['box'])
  if runs and (runs[-1]['description'],runs[-1]['box'])==key:runs[-1]['length']+=1
  else:runs.append(dict(start_row=x['row_1based'],length=1,description=x['description'],box=x['box']))
 old=[x['row_1based'] for x in spans if x['box']==[0,999,999,999]];noncorner=next((x for x in spans if x['box']!=[0,999,999,999]),None);invalid_repeats=sum(n-1 for (desc,b),n in counter.items() if not(b[0]<b[2] and b[1]<b[3]));reason=collections.Counter(x.get('drop_reason',x.get('reason')) for x in drops)
 return dict(image_id=iid,free_token_count=len(ids),stop=stop,primary_free_only=ledgers[str(iid)],diagnostic_both_banks=ledgers,valid_predictions=valid,native_parse=native,drops=drops,complete_rows=spans,immediate_next_row=spans[0] if spans else None,first_noncorner=noncorner,first_nonzero_x1=next((x for x in spans if x['box'][0]!=0),None),original_corner_rows=old,recurrence_after_exit=[j for j in old if noncorner and j>noncorner['row_1based']],all999_rows=[x['row_1based'] for x in spans if x['box']==[999,999,999,999]],exact_runs=runs,burden=dict(complete_rows=len(spans),valid=len(valid),invalid_complete=sum(not(x['box'][0]<x['box'][2] and x['box'][1]<x['box'][3]) for x in spans),drop_reasons=dict(reason),malformed=sum(v for k,v in reason.items() if 'geometry' not in str(k) and 'bbox' not in str(k)),strict_valid_repeats=len(metrics._strict_repeat_rows(valid)),literal_repeats_free=sum(n-1 for n in counter.values()),literal_invalid_repeats_free=invalid_repeats,unknown=len(ledgers[str(iid)]['annotation_unmatched_prediction_ids']),cap=int(stop=='length'),eos=int(stop=='im_end')),endpoint_occupancy=dict(coordinate_tokens=len(coords),zero=coords.count(0),last=coords.count(999),fraction=sum(c in [0,999] for c in coords)/len(coords) if coords else None,complete_row_roles=roles))
def main():
 p=read(R/'panel.json');tok=AutoTokenizer.from_pretrained(p['config']['model']['base_model'],local_files_only=True)
 for b in p['sources']:assert bind(Path(b['path']))==b
 assert read(R/'admission.json')['status']=='passed';assert (R/'run.exit').read_text().strip()=='0'
 recs={c:read(R/f'runtime/{c}/receipt.json') for c in ['I00','D00','D10']};base=read(OLD/'runtime/C00/receipt.json');originalraw=read(OLD/'runtime/C00/raw.json')
 for c,rec in recs.items():
  assert rec['status']=='candidate_complete' and rec['producer']==bind(R/'producer.py') and rec['panel']==bind(R/'panel.json');assert rec['raw']==bind(R/f'runtime/{c}/raw.json');assert not rec['mechanical_forks']
  for k in ['loaded_identity','generate_settings','prompt_positions_sha256','positions']:assert rec[k]==base[k],(c,k)
  shape={k:v for k,v in rec['batch_shape'].items() if k!='image_ids'};assert shape=={k:v for k,v in base['batch_shape'].items() if k!='image_ids'}
  raw=read(R/f'runtime/{c}/raw.json');assert raw['rows'][1]['token_ids'][:1233]==p['cells'][c]['history_ids'];assert raw['history_supply_interval']==[0,1233]
  for j in [0,2,3]:assert raw['rows'][j]==originalraw['rows'][j]
  assert all(x['supplied_token'] is None for x in rec['seam_logits'] if x['action_offset']>=1233)
 cells={};raws={}
 for c,path,iid in [('O00',OLD/'runtime/C00/raw.json',477415),('O10',OLD/'runtime/C10/raw.json',477415),('D00',R/'runtime/D00/raw.json',p['donor']['image_id']),('D10',R/'runtime/D10/raw.json',p['donor']['image_id'])]:
  raw=read(path);raws[c]=raw;cells[c]=score(raw['free_token_ids'],raw['rows'][1]['stop'],iid,p,tok);cells[c]['raw']=bind(path)
  # This reduction never passes supplied tokens to parser/matcher.
  assert raw['free_token_ids']==raw['rows'][1]['token_ids'][1233:]
 comparisons={}
 for donor,orig in [('D00','O00'),('D10','O10')]:
  a=raws[donor]['free_token_ids'];b=raws[orig]['free_token_ids'];first=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)) if len(a)!=len(b) else None)
  comparisons[donor]=dict(original=orig,exact_suffix=a==b,first_changed_free_token_zero_based=first,donor_token=a[first] if first is not None and first<len(a) else None,original_token=b[first] if first is not None and first<len(b) else None,matching_leading_tokens=first if first is not None else len(a),early_64_token_equal=a[:64]==b[:64],early_256_token_equal=a[:256]==b[:256])
 # Credit falsification: a supplied full-image row matches an artificial target only if incorrectly included.
 fixture=tok.encode('<|object_ref_start|>person<|object_ref_end|><|box_start|><|coord_0|><|coord_0|><|coord_999|><|coord_999|><|box_end|>',add_special_tokens=False);ctx=p['parse_context'];native=parse_compact_object_box_closed(tok.decode(fixture),**ctx).to_artifact_dict();v,_=match._matchable_rows_with_geometry_debt({**native,'pred':native['predictions']})
 for x in v:x['prediction_id']='SUPPLIED'
 target=[metrics._target(image_id=477415,owner_id='credit-fixture',description='person',coord_bins=[0,0,999,999])];assert match._ledger_image(target,v,threshold=.5)['matched_count']==1 and match._ledger_image(target,[],threshold=.5)['matched_count']==0
 counts=sum(r['counts']['model_forwards'] for r in recs.values());seconds=sum(r['elapsed_seconds'] for r in recs.values());assert counts<=10000 and seconds<=7200
 result=dict(status='candidate',panel=bind(R/'panel.json'),consumer=bind(Path(__file__)),cells=cells,paired_comparisons=comparisons,baseline_donor=p['donor']['baseline_ledger'],baseline_donor_saved_output=score(p['donor']['generation']['generated_token_ids'],p['donor']['generation']['decode_stop_reason'],p['donor']['image_id'],p,tok),credit_exclusion_sensitivity=[1,0],cost=dict(model_forwards=counts,allocated_gpu_seconds=seconds,scientific_cells=2,identity_cells=1),runtime_bindings={c:bind(R/f'runtime/{c}/receipt.json') for c in recs})
 (R/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(cells={c:dict(matches=x['primary_free_only']['matched_count'],FN=x['primary_free_only']['fn_count'],labels=x['primary_free_only']['label_string_compatible_count'],burden=x['burden'],stop=x['stop'],tokens=x['free_token_count'],next=x['immediate_next_row'],cross_original=x['diagnostic_both_banks']['477415']['matched_count']) for c,x in cells.items()},comparisons=comparisons,cost=result['cost']),indent=2))
if __name__=='__main__':main()
