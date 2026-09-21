import hashlib,json,re
from pathlib import Path
from probes.training_set_completion.successful_row_reduce import reduce
R=Path(__file__).resolve().parent;P=R.parent/'2026-09-17-successful-row-mechanism'
bind=lambda p:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
expected=json.loads((R/'reduction.json').read_text());assert reduce(R/'reduction-manifest.json')==expected
checks=[]
for name,c in expected['cells'].items():
 raw=json.loads(Path(c['raw']['path']).read_text())['rows'][1];ids=raw['token_ids']
 assert (ids[-1]==151645)==(raw['stop']=='im_end')
 if raw['stop']=='length':assert len(ids)==3084
 boxes=re.findall(r'<\|box_start\|>((?:<\|coord_\d+\|>){4})<\|box_end\|>',raw['text'])
 boxes=[list(map(int,re.findall(r'coord_(\d+)',s))) for s in boxes]
 assert len(boxes)==c['full']['burden']['complete_rows']
 assert sum(a>=c_ or b>=d for a,b,c_,d in boxes)==c['full']['burden']['invalid']
 assert c['release']==72 and c['symmetric_known_accounting']['excluded_supplied_union']==[]
 checks.append(dict(condition=name,rows=len(boxes),stop=raw['stop'],primary_start=72,empty_known_retention_denominator=c['symmetric_known_accounting']['retention_denominator']==0))
cost=json.loads((R/'cost.json').read_text());assert cost['model_forwards']==18640 and not cost['owned_live_jobs'] and all(v==0 for v in cost['model_exits'].values())
control=P/'consumer-check/exclusion-check.json'
# Same unchanged pure consumer: reuse accepted nonzero exclusion falsification.
prior_inventory=json.loads((P/'artifact-bindings.json').read_text())['records'];old=next(b for b in prior_inventory if b['path']==str(control));assert bind(control)==old
result=dict(status='passed',JSON_exact_reduction=True,raw_geometry_and_stop_checks=checks,new_full_cells=6,reused_S_reference=1,forwards=18640,all_model_exits_zero=True,consumer=bind(Path('probes/training_set_completion/successful_row_reduce.py').resolve()),reused_exclusion_falsification=bind(control),scope='Saved-output CPU checks only; human-positive rescoring is a separate versioned surface.')
(R/'independent-output-check.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['raw_geometry_and_stop_checks','consumer','reused_exclusion_falsification']})
