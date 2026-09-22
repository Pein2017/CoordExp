import json,hashlib,shutil
from pathlib import Path
import torch
from probes.training_set_completion.recurrence_spatial import producer as p
b=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source')
out=b.parent/'2026-09-19-recurrence-distribution-census/integration/spatial-counterexample'
out.mkdir(exist_ok=True)
for name in ('prepare','producer','reduce'):
 src=Path('probes/training_set_completion/recurrence_spatial')/(name+'.py');shutil.copy2(src,out/(name+'.py'))
m=json.loads((b/'transform-manifest-tied.json').read_text())
y=[]
for cell,c in m['cells'].items():
 for i,row in enumerate(c['history_boxes']):
  source=row.get('source_bins');mapped=row.get('mapped_bins')
  if source and mapped and (source[1]!=mapped[1] or source[3]!=mapped[3]):
   y.append({'cell':cell,'row':i,'source':source,'mapped':mapped});break
row=[151646,123,151647,151648,151671,151672,151673,151674,151649]
tokens=torch.tensor([row*31+[151646,123,151647]])
scores=torch.ones(1,152670)
changed=p.RowLimitEOS(baseline_end_count=0,max_rows=32,eos=151645)(tokens,scores.clone())
stop={'complete_box_ends':int((tokens==151649).sum()),'description_ends':int((tokens==151647).sum()),'eos_forced':bool(torch.isneginf(changed[0,0]))}
def rr(i,box,valid):return {'row_index':i,'status':'valid' if valid else 'invalid','source_geometry_valid':valid,'description':'x','coord_bins_source':box}
invalid=p._runs([rr(i,[0,0,0,0],False) for i in range(3)],near=False)
drift=p._runs([rr(i,[8*i,0,100+8*i,100],True) for i in range(3)],near=True)
r=json.loads((b/'runtime-result-tied-transforms.json').read_text())
windows={k:v['boundary']['forced_description_x1']['windows'] for k,v in r['cells'].items() if k.startswith('10')}
result={'status':'blocking_counterexamples','horizontal_transform_changes_y':y,'premature_row_stop':stop,'three_literal_invalid_rows_reported_runs':len(invalid),'non_pairwise_three_row_drift_reported_runs':len(drift),'visual_only_windows':windows,'affected_lane':'B only','model_calls':0,'source_hashes':{n:hashlib.sha256((out/(n+'.py')).read_bytes()).hexdigest() for n in ('prepare','producer','reduce')}}
(out/'counterexamples.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
