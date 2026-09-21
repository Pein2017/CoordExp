import json, math, hashlib
from pathlib import Path
from transformers import AutoTokenizer
from probes.training_set_completion.recurrence_spatial import producer, reduce
root=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/corrected-pilot-v2')
out=Path(__file__).parent/'spatial-pilot-recomputed';out.mkdir(exist_ok=True)
tok=AutoTokenizer.from_pretrained(str(reduce.BASE),use_fast=False)
checks=[]
for model in ['tied','untied']:
 manifest=json.loads((root/'inputs/manifests'/f'{model}-417044-failure.json').read_text())
 for suffix in ['', '-transforms']:
  raw=json.loads((root/'raw'/f'{model}{suffix}-runtime.json').read_text())
  for key,c in raw['cells'].items():
   free=c['free']; parsed=producer.parse_rows(free['token_ids'],tok,cell=manifest['cells'][key],geometry=manifest['geometry'])
   assert parsed==free['parse'],(model,key,'parse')
   cap=free['row_limit']; assert cap['last_free_complete_rows']==[parsed['complete_rows']]
   if cap['injected_eos']: assert parsed['complete_rows']==32 and free['token_ids'][-1]==151645 and cap['injection_reason']=='row_cap'
   f=c['boundary']['forced_description_x1']; assert len(f['coordinate_log_probs'])==1000
   for sign,w in f['windows_by_sign'].items():
    for field in ['old_window','moved_window']:
     win=w[field]; lo,hi=win['lo'],win['hi'];v=f['coordinate_log_probs'][lo:hi+1];z=max(v);lm=z+math.log(sum(math.exp(x-z) for x in v))
     assert abs(lm-win['log_mass'])<1e-5,(model,key,sign,field,lm,win['log_mass'])
   checks.append({'model':model,'cell':key,'complete_rows':parsed['complete_rows'],'failure':parsed['failure_predicate'],'termination':'row_cap' if cap['injected_eos'] else free['stop_reason'],'windows_recomputed':True})
assert len(checks)==14
p=Path(__file__).parent/'spatial-pilot-check.json';p.write_text(json.dumps({'status':'pass','cells':checks,'model_calls':0,'scope':'saved token parse and full-vocabulary window-mass reconstruction; native source parity separate'},indent=2)+'\n')
print('PASS 14 saved cells, 56 window sums')
