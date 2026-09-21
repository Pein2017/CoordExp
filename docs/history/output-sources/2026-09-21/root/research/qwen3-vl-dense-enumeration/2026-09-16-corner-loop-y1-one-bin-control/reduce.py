"""One saved-output reduction through the exact accepted public consume function."""
import hashlib,importlib.util,json
from pathlib import Path
from transformers import AutoTokenizer
R=Path(__file__).resolve().parent;OLD=R.parent/'2026-09-16-corner-loop-bridge-factorial'
read=lambda p:json.loads(p.read_text())
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}
def main():
 p=read(R/'panel.json');prior=read(OLD/'panel.json')
 for b in p['sources']:assert bind(Path(b['path']))==b
 for key in ['config','config_sha256','batch_group','cases','common_history_ids','common_history_sha256','owner_targets','owner_bank_sha256','parse_context','original_action_cap','supplied_tokens','free_budget','eos_id']:
  assert p[key]==prior[key],key
 spec=importlib.util.spec_from_file_location('accepted_bridge_consumer',OLD/'reduce.py');consumer=importlib.util.module_from_spec(spec);spec.loader.exec_module(consumer)
 tok=AutoTokenizer.from_pretrained(p['config']['model']['base_model'],local_files_only=True)
 fixture=tok.decode(p['cells']['C11']['history_ids'],skip_special_tokens=False);_,v,_=consumer.parse(fixture,p['parse_context']);synthetic=[consumer.metrics._target(image_id=477415,owner_id='synthetic-credit-test',description='person',coord_bins=[0,0,999,999])]
 include=consumer.match._ledger_image(synthetic,v,threshold=.5)['matched_count'];exclude=consumer.match._ledger_image(synthetic,[x for x in v if x['generated_order']!=136],threshold=.5)['matched_count'];assert (include,exclude)==(1,0)
 receipt=read(R/'runtime/Y1_998/receipt.json');assert receipt['status']=='candidate_complete';assert receipt['panel']==bind(R/'panel.json') and receipt['raw']==bind(R/'runtime/Y1_998/raw.json');assert receipt['counts']['model_forwards']<=5000
 c00=read(OLD/'runtime/C00/receipt.json')
 for key in ['loaded_identity','batch_shape','generate_settings','prompt_positions_sha256','positions']:
  assert receipt[key]==c00[key],key
 raw=read(R/'runtime/Y1_998/raw.json');assert [i for i,(a,b) in enumerate(zip(raw['supplied137_token_ids'],prior['cells']['C00']['tokens'])) if a!=b]==[5];assert raw['supplied137_token_ids'][5]==tok.convert_tokens_to_ids('<|coord_998|>')==152668
 cell=consumer.consume(p,raw,tok);controls=dict(read(OLD/'result.json')['cells']);controls['Y998']=read(R.parent/'2026-09-16-corner-loop-one-bin-control/result.json')['cell'];coverage=set(cell['joint_match_excluding_supplied137']['covered_owner_ids']);new=set(cell['new_covered_owner_ids']);reference16=set(controls['C01']['new_covered_owner_ids']);assert len(reference16)==16
 comparisons={}
 for name,c in controls.items():
  old=set(c['joint_match_excluding_supplied137']['covered_owner_ids']);comparisons[name]={'gained_owner_ids':sorted(coverage-old),'lost_owner_ids':sorted(old-coverage),'matched_count_delta':len(coverage)-len(old)}
 result={'schema':'corner_loop.y1_one_bin_control.result.v1','status':'candidate','panel':bind(R/'panel.json'),'cell':cell,'inherited_controls_result':bind(OLD/'result.json'),'comparisons':comparisons,'recovered16_comparison':{'equal':new==reference16,'contains_all16':new>=reference16,'shared':sorted(new&reference16),'missing':sorted(reference16-new),'different':sorted(new-reference16)},'technical':{'new_scientific_cells':1,'control_GPU_reruns':0,'native_control_fields_equal':['loaded_identity','batch_shape','generate_settings','prompt_positions_sha256','positions'],'exact_delta':receipt['edits'],'common_history_and_companions_exact':receipt['common136_identity'] and receipt['companion_identity'],'credit_exclusion_sensitivity':[include,exclude],'counts':receipt['counts'],'elapsed_gpu_seconds':receipt['elapsed_seconds'],'free_tokens':receipt['free_tokens'],'stop':receipt['stop']},'bindings':[bind(R/'runtime/Y1_998/receipt.json'),bind(R/'runtime/Y1_998/raw.json'),bind(OLD/'reduce.py'),bind(Path(__file__))]}
 (R/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'coverage':cell['joint_match_excluding_supplied137']['matched_count'],'new_owners':cell['new_covered_owner_ids'],'lost_history':cell['lost_history_owner_ids'],'assignment_changes':cell['history_assignment_changes'],'next':cell['immediate_next_complete_row'],'burden':cell['burden'],'recurrence':cell['later_recurrence'],'recovered16':result['recovered16_comparison'],'technical':result['technical']},indent=2))
if __name__=='__main__':main()
