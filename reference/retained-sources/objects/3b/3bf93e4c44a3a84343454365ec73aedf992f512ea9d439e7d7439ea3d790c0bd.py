import hashlib,json,math
from pathlib import Path
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}
panel=read(R/'panel.json');cells=[]
for label in panel['checkpoints']:
 for iid in [477415,351017,417044]:
  p=R/'runtime'/f'{label}-{iid}'/'receipt.json';d=read(p);assert d['status'] in ['candidate_complete','blocked_material_execution_parity'];assert d['panel_sha256']==bind(R/'panel.json')['sha256'];assert d['producer_sha256']==bind(R/'producer.py')['sha256'];cells.append(d)
assert len({c['coordinate_input_sha256'] for c in cells})==1
assert len({json.dumps(c['readout_parameter_hashes'],sort_keys=True) for c in cells})==1
for iid in [477415,351017,417044]:assert len({c['prompt_position_sha256'] for c in cells if c['image_id']==iid})==1
slots=[s for c in cells for b in c['boundaries'] for r in b['routes'] for s in r['slots']]
counts={k:sum(c['counts'][k] for c in cells) for k in cells[0]['counts']};assert counts['model_forwards']<=6500 and counts['new_generated_tokens']==0
parity={'material_failures':sum(s['material_parity_failure'] for s in slots),'cache_full_argmax_flips':sum(s['cached']['argmax']!=s['full']['argmax'] for s in slots),'bs4_full_argmax_flips':sum(s['bs4_full']['argmax']!=s['full']['argmax'] for s in slots),'max_slot_cache_full_abs':max(s['max_abs_cache_full'] for s in slots),'max_twice_error_over_margin':max(s['twice_error_over_margin'] for s in slots),'full_noop_max_abs':max(r['full_noop_max_abs'] for c in cells for b in c['boundaries'] for r in b['routes']),'max_bs4_full_abs':max(r['bs4_full_max_abs'] for c in cells for b in c['boundaries'] for r in b['routes']),'slot_count':len(slots),'coordinate_input_sha256':cells[0]['coordinate_input_sha256'],'readout_parameter_hashes':cells[0]['readout_parameter_hashes']}
# Compact every case/slot with exact source pointers; raw vectors stay on disk.
table=[]
for c in cells:
 for b in c['boundaries']:
  for route in b['routes']:
   for s in route['slots']:
    f=s['full'];zero=next((x for x in f['candidates'] if x['token']=='<|coord_0|>'),None);eos=next(x for x in f['candidates'] if x['token']=='<|im_end|>')
    table.append({'checkpoint':c['checkpoint'],'image_id':c['image_id'],'row':b['row_1based'],'route':route['role'],'slot':s['slot'],'argmax':f['argmax_text'],'top_margin':f['top_margin'],'coord0_rank':zero['rank'] if zero else None,'coord0_gap':zero['gap_to_top'] if zero else None,'eos_rank':eos['rank'],'eos_gap':eos['gap_to_top'],'parity_max_abs':s['max_abs_cache_full'],'cache_same_argmax':f['argmax']==s['cached']['argmax'],'saved_observed_token':route['tokens'][s['row_token_offset']],'receipt':str(R/'runtime'/f"{c['checkpoint']}-{c['image_id']}"/'receipt.json')})
result={'status':'candidate_complete' if not parity['material_failures'] else 'blocked_execution_parity','acceptance':'execution owner checked; independent lead acceptance pending','panel':bind(R/'panel.json'),'producer':bind(R/'producer.py'),'reducer':bind(Path(__file__)),'launch':bind(R/'launch.json'),'counts':counts,'allocated_gpu_seconds_sum':sum(c['elapsed_seconds'] for c in cells),'max_cell_seconds':max(c['elapsed_seconds'] for c in cells),'parity':parity,'table':table,'cells':[bind(R/'runtime'/f"{c['checkpoint']}-{c['image_id']}"/'receipt.json') for c in cells],'logit_files':[bind(p) for p in sorted((R/'runtime').glob('*/*-logits.pt'))],'limitations':['Same P history across models; off-policy after divergence.','Candidate rows scored under their own coherent internal prefixes; no free suffix or causal intervention.','bs4 diagnostic duplicates identical image/history; it does not recreate heterogeneous natural batch companions/padding.','No exhaustive owner ledger or physical duplication inference; raw invalid geometry preserved.']}
(R/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in result.items() if k in ['status','counts','allocated_gpu_seconds_sum','max_cell_seconds','parity']},indent=2))
for iid in [477415,351017,417044]:
 for row in panel['boundary_rows'][str(iid)]:
  for label in panel['checkpoints']:
   subset=[x for x in table if x['image_id']==iid and x['row']==row and x['checkpoint']==label and x['route']=='observed'];print(iid,row,label,[(x['slot'],x['argmax'],round(x['top_margin'],4),x['coord0_rank']) for x in subset])
