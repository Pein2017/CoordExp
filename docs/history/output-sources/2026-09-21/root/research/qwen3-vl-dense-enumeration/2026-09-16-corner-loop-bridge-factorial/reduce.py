"""Independent saved-token consumer; supplied row137 is never matched for recovery."""
import collections,hashlib,json,re
from pathlib import Path
from transformers import AutoTokenizer
from src.inference.parsing import parse_compact_object_box_closed
from probes.training_set_completion import paired_evaluation as match
from probes.training_set_completion import source256_evaluation as metrics
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
PAT=re.compile(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>'+r'<\|coord_(\d+)\|>'*4+r'<\|box_end\|>')
def parse(text,context):
 native=parse_compact_object_box_closed(text,**context).to_artifact_dict();valid,dropped=match._matchable_rows_with_geometry_debt({**native,'pred':native['predictions']})
 for p in valid:
  order=p['generated_order'];p['prediction_id']=f'H{order}' if order<136 else 'SUPPLIED137' if order==136 else f'F{order-137}'
 return native,valid,dropped

def consume(panel,raw,tok):
 ids=raw['rows'][1]['token_ids'];text=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False);assert text==raw['rows'][1]['text'];assert ids[:1224]==panel['common_history_ids'];assert ids[1224:1233]==panel['cells'][raw['cell']]['tokens'];assert ids[1233:]==raw['free_token_ids']
 native,valid,dropped=parse(text,panel['parse_context']);history=[x for x in valid if x['generated_order']<136];free=[x for x in valid if x['generated_order']>=137];supplied=[x for x in valid if x['generated_order']==136]
 admitted=history+free;assert all(x['prediction_id']!='SUPPLIED137' for x in admitted)
 targets=panel['owner_targets'];histmatch=match._ledger_image(targets,history,threshold=.5);freematch=match._ledger_image(targets,free,threshold=.5);joint=match._ledger_image(targets,admitted,threshold=.5)
 covered=set(joint['covered_owner_ids']);old=set(histmatch['covered_owner_ids']);joint_free=[x for x in joint['matches'] if x['prediction_id'].startswith('F')];joint_h=[x for x in joint['matches'] if x['prediction_id'].startswith('H')]
 oldassign={x['prediction_id']:x['reference_owner_id'] for x in histmatch['matches']};newassign={x['prediction_id']:x['reference_owner_id'] for x in joint_h};swaps=[{'prediction_id':p,'before':owner,'after':newassign.get(p)} for p,owner in oldassign.items() if newassign.get(p)!=owner]
 repeats=metrics._strict_repeat_rows(valid);free_repeats=[x for x in repeats if x['prediction_id'].startswith('F')];free_only_repeats=metrics._strict_repeat_rows(free)
 free_drops=[x for x in dropped if int(x['generated_order'])>=137]
 free_spans=[]
 for n,m in enumerate(PAT.finditer(raw['free_text'])):
  box=list(map(int,m.groups()[1:]));free_spans.append({'free_complete_row_1based':n+1,'description':m[1],'box':box,'char_start':m.start(),'char_end':m.end(),'geometry_valid':box[0]<box[2] and box[1]<box[3]})
 corner=lambda x:x['box']==[0,999,999,999]
 leave=next((x for x in free_spans if not corner(x)),None);recur=[x['free_complete_row_1based'] for x in free_spans if leave is not None and x['free_complete_row_1based']>leave['free_complete_row_1based'] and corner(x)]
 nonzero=next((x for x in free_spans if x['box'][0]!=0),None);valid_row=next((x for x in free_spans if x['geometry_valid']),None)
 sustained_invalid=next((x['free_complete_row_1based'] for j,x in enumerate(free_spans) if not x['geometry_valid'] and all(not y['geometry_valid'] for y in free_spans[j:])),None)
 endpoint_rows=[x for x in free_spans if not x['geometry_valid'] and all(v in [0,999] for v in x['box'])]
 southeast=[x for x in free_spans if x['box']==[999,999,999,999]]
 recurrence={'original_corner_definition':[0,999,999,999],'first_sustained_invalid_row':sustained_invalid,'endpoint_degenerate_rows':len(endpoint_rows),'first_all999_row':southeast[0]['free_complete_row_1based'] if southeast else None,'all999_rows':len(southeast),'dominant_exact_rows':[{'description':key[0],'box':list(key[1]),'count':n} for key,n in collections.Counter((x['description'],tuple(x['box'])) for x in free_spans).most_common(3)]}
 reasons=collections.Counter(str(x.get('drop_reason',x.get('reason'))) for x in free_drops)
 geometrydrops=sum(v for k,v in reasons.items() if 'geometry' in k or 'bbox' in k)
 # Count unmatched text after complete boxes, preserving EOS separately from fragments.
 remainder=PAT.sub('',raw['free_text']).replace('<|im_end|>','')
 return {'cell':raw['cell'],'stop':raw['rows'][1]['stop'],'free_token_count':len(raw['free_token_ids']),'total_action_tokens':len(ids),'known_owner_count':len(targets),'history_match':histmatch,'free_only_match':freematch,'joint_match_excluding_supplied137':joint,'free_matches_in_joint':joint_free,'new_covered_owner_ids':sorted(covered-old),'lost_history_owner_ids':sorted(old-covered),'history_assignment_changes':swaps,'burden':{'free_complete_rows':len(free_spans),'free_valid_rows':len(free),'free_geometry_invalid_complete_rows':sum(not x['geometry_valid'] for x in free_spans),'free_native_parser_drops':len(free_drops),'free_geometry_drops':geometrydrops,'free_malformed_non_geometry_drops':len(free_drops)-geometrydrops,'free_drop_reasons':dict(reasons),'free_strict_repeat_rows_with_supplied_history':len(free_repeats),'free_strict_repeat_rows_without_history':len(free_only_repeats),'free_annotation_unmatched_unknown_joint':sum(p.startswith('F') for p in joint['annotation_unmatched_prediction_ids']),'free_unparsed_fragment_text':remainder,'cap_debt':int(raw['rows'][1]['stop']=='length'),'eos_debt':int(raw['rows'][1]['stop']!='im_end')},'immediate_next_complete_row':free_spans[0] if free_spans else None,'first_non_corner_complete_row':leave,'first_nonzero_x1_complete_row':nonzero,'first_geometry_valid_complete_row':valid_row,'corner_recurrence_rows':recur,'later_recurrence':recurrence,'free_complete_rows':free_spans,'free_repeats_with_history':free_repeats,'native_parse':native,'matching_valid_rows':admitted,'supplied137_rows_excluded':supplied,'free_dropped_rows':free_drops}

def main():
 p=read(R/'panel.json');tok=AutoTokenizer.from_pretrained(p['config']['model']['base_model'],local_files_only=True)
 # Falsification: a full-image supplied person WOULD match a synthetic owner if admitted.
 # This is a CPU credit-exclusion fixture, never a scientific label/bank mutation.
 fixture=tok.decode(p['cells']['C11']['history_ids'],skip_special_tokens=False);_,v,_=parse(fixture,p['parse_context']);synthetic=[metrics._target(image_id=477415,owner_id='synthetic-credit-test',description='person',coord_bins=[0,0,999,999])]
 assert match._ledger_image(synthetic,v,threshold=.5)['matched_count']==1
 assert match._ledger_image(synthetic,[x for x in v if x['generated_order']!=136],threshold=.5)['matched_count']==0
 cases={};receipts={}
 for cell in ['C00','C10','C01','C11']:
  receipt=read(R/f'runtime/{cell}/receipt.json');assert receipt['status']=='candidate_complete';assert receipt['panel']==bind(R/'panel.json');assert receipt['raw']==bind(R/f'runtime/{cell}/raw.json');receipts[cell]=receipt
  raw=read(R/f'runtime/{cell}/raw.json');cases[cell]=consume(p,raw,tok)
  for m in cases[cell]['joint_match_excluding_supplied137']['matches']:assert m['prediction_id']!='SUPPLIED137'
 control=cases['C00'];controlowners=set(control['joint_match_excluding_supplied137']['covered_owner_ids'])
 for c in cases.values():
  owners=set(c['joint_match_excluding_supplied137']['covered_owner_ids']);c['vs_C00']={'gained':sorted(owners-controlowners),'lost':sorted(controlowners-owners),'FN':27-len(owners)}
 counts={k:sum(v['counts'][k] for v in receipts.values()) for k in receipts['C00']['counts']};assert counts['model_forwards']<=20000
 result={'schema':'corner_loop.bridge_factorial.reduction.v1','status':'candidate','panel':bind(R/'panel.json'),'consumer':bind(Path(__file__)),'cells':cases,'technical':{'C00_full_original_batch_identity':receipts['C00']['full_original_batch_identity'],'companion_identity_all':all(x['companion_identity'] for x in receipts.values()),'positions_equal_across_cells':len({x['prompt_positions_sha256'] for x in receipts.values()})==1,'credit_exclusion_sensitivity':'synthetic full-image owner matches1 with supplied row,0 without; scientific bank unchanged','counts':counts,'allocated_gpu_seconds':sum(x['elapsed_seconds'] for x in receipts.values()),'max_cell_seconds':max(x['elapsed_seconds'] for x in receipts.values())},'receipt_bindings':{c:bind(R/f'runtime/{c}/receipt.json') for c in cases}}
 assert result['technical']['positions_equal_across_cells'];(R/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 for c,x in cases.items():print(c,x['stop'],x['free_token_count'],'coverage',x['joint_match_excluding_supplied137']['matched_count'],'new',x['new_covered_owner_ids'],'vsC00',x['vs_C00'],'burden',x['burden'],'next',x['immediate_next_complete_row'])
 print('TECHNICAL',result['technical'])
if __name__=='__main__':main()
