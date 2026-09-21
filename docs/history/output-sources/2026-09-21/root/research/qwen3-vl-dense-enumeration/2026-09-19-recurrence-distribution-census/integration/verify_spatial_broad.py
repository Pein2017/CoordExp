"""Replay corrected Lane B saved outputs; no model calls."""
import hashlib,json,math
from pathlib import Path
from transformers import AutoTokenizer
from probes.training_set_completion.recurrence_spatial import producer, reduce
base=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1')
out=Path(__file__).parent/'spatial-broad-recomputed';out.mkdir(exist_ok=True)
tok=AutoTokenizer.from_pretrained(str(reduce.BASE),use_fast=False)
# The native reducer loads the same immutable tokenizer; reuse it in this CPU process.
reduce.AutoTokenizer.from_pretrained=lambda *a,**kw:tok
checks=[]; bindings={};windows=0;cells=0
for p in sorted((base/'reduced').glob('*.json')):
 expected=json.loads(p.read_text()); src=expected['source']; rp=Path(src['runtime_result']['path']);mp=Path(src['transform_manifest']['path'])
 actual=reduce.reduce(reduce.OUT,expected['model'],rp,out/p.name,mp)
 assert actual==expected,(p.name,'reducer mismatch')
 raw=json.loads(rp.read_text()); manifest=json.loads(mp.read_text())
 for b in src.values():
  if isinstance(b,dict) and 'path' in b and 'sha256' in b:
   q=Path(b['path']);assert hashlib.sha256(q.read_bytes()).hexdigest()==b['sha256'],q;bindings[str(q)]=b['sha256']
 for key,c in raw['cells'].items():
  free=c['free'];parsed=producer.parse_rows(free['token_ids'],tok,cell=manifest['cells'][key],geometry=manifest['geometry'])
  assert parsed==free['parse'],(p.name,key,'parse')
  cap=free['row_limit'];assert cap['last_free_complete_rows']==[parsed['complete_rows']]
  assert len(free['token_ids'])<=512
  if cap['injected_eos']: assert parsed['complete_rows']==32 and free['token_ids'][-1]==151645 and cap['injection_reason']=='row_cap'
  f=c['boundary']['forced_description_x1'];assert len(f['coordinate_log_probs'])==1000
  for sign,w in f['windows_by_sign'].items():
   for field in ['old_window','moved_window']:
    win=w[field];v=f['coordinate_log_probs'][win['lo']:win['hi']+1];z=max(v);lm=z+math.log(sum(math.exp(x-z) for x in v))
    assert abs(lm-win['log_mass'])<1e-5,(p.name,key,sign,field)
    # Fixed window support is identical across all cells at the same source/sign.
    ref=raw['cells']['00']['boundary']['forced_description_x1']['windows_by_sign'][sign][field]
    assert (win['lo'],win['hi'])==(ref['lo'],ref['hi'])
    windows+=1
  cells+=1
 checks.append({'state':p.stem,'cells':len(raw['cells']),'admission':raw['admission'],'forwards':raw['model_forwards']})
assert len(checks)==43
receipt={'status':'pass','states':checks,'state_count':len(checks),'cell_count':cells,'full_vocabulary_window_sums':windows,'unique_bindings':len(bindings),'reduction_json_exact':True,'model_calls':0}
(Path(__file__).parent/'spatial-broad-check.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='states'}))
