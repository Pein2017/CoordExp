"""Independent saved-output replay; never rewrites frozen selection or runtime."""
import hashlib
import json
from pathlib import Path
import numpy as np
from probes.training_set_completion.recurrence_census import reduce_new as r
from probes.training_set_completion.numerical_feedback.select import token_hash

out=r.OUT
panel=json.loads((out/'panel.json').read_text())
saved=json.loads((out/'new-census.json').read_text())
scorer=r._old_reduce_module()
cells={}
for condition in panel['conditions']:
    cells[condition]={}
    for group in panel['groups']:
        runtime=out/'runtime'/condition/group['key']
        assert json.loads((runtime/'receipt.json').read_text())['status']=='candidate_complete'
        raw=json.loads((runtime/'raw.json').read_text())
        assert len(raw['rows'])==len(group['cases'])==4
        for j,case in enumerate(group['cases']):
            key=f"{case['input_record']['metadata']['split']}:{case['input_record']['image_id']}"
            assert key not in cells[condition]
            cells[condition][key]=r._cell(condition,group,j,raw,case,scorer,panel['new_banks'][key])
assert cells==saved['cells']
for condition,by_image in cells.items():
    values=list(by_image.values());summary=r._aggregate(values)
    rng=np.random.default_rng(19);n=len(values);draws=rng.integers(0,n,(10000,n))
    bootstrap={'unit':'image','resamples':10000}
    for prefix,field in [('exact','literal_exact_repeat_rows'),('near8','near8_same_description_repeat_rows')]:
        indicator=np.array([v['output'][field]>0 for v in values],dtype=float)
        bootstrap[f'{prefix}_exposure_rate_ci95']=np.quantile(indicator[draws].mean(axis=1),[.025,.975]).tolist()
    summary['image_bootstrap_seed19']=bootstrap
    assert summary==saved['summary'][condition]
shared=json.loads((out/'shared-panel.json').read_text());seen=set();bindings=0
for b in shared['all_boundaries']:
    key=(b['split'],b['image_id'],b['model'],b['kind']);assert key not in seen;seen.add(key)
    raw=json.loads(Path(b['raw_path']).read_text())['rows'][b['batch_index']]
    assert raw['token_ids']==b['native_tokens']
    assert token_hash(b['native_tokens'][:b['source_row']['end']])==b['source_prefix_hash']
    for ref in b['bindings'].values():
        if isinstance(ref,dict) and 'path' in ref and 'sha256' in ref:
            assert hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()==ref['sha256'];bindings+=1
receipt={'status':'passed','cells_json_exact':sum(map(len,cells.values())),'summary_and_bootstrap_json_exact':True,'panel_states':len(seen),'checked_panel_bindings':bindings,'panel_sha256':hashlib.sha256((out/'shared-panel.json').read_bytes()).hexdigest()}
(out/'integration/new-census-check.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
