import json,re,hashlib
from pathlib import Path
from probes.training_set_completion.successful_row_reduce import reduce
R=Path(__file__).resolve().parent
bind=lambda p:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
checks=[]
for stage,manifest,result in [('stage1','stage1-reduction-manifest.json','stage1-reduction.json'),('stage2','stage2-final-manifest.json','stage2-final-reduction.json')]:
 expected=json.loads((R/result).read_text());actual=reduce(R/manifest);assert actual==expected
 for name,c in expected['cells'].items():
  raw=json.loads(Path(c['raw']['path']).read_text());row=raw['rows'][1];ids=row['token_ids'];assert row['image_id']==309264
  assert (ids[-1]==151645)==(row['stop']=='im_end');assert len(ids)<=3084
  if row['stop']=='length':assert len(ids)==3084
  spans=re.findall(r'<\|object_ref_start\|>.*?<\|object_ref_end\|><\|box_start\|>((?:<\|coord_\d+\|>){4})<\|box_end\|>',row['text'])
  boxes=[list(map(int,re.findall(r'coord_(\d+)',s))) for s in spans]
  assert len(boxes)==c['full']['burden']['complete_rows']
  assert sum(x1>=x2 or y1>=y2 for x1,y1,x2,y2 in boxes)==c['full']['burden']['invalid']
  assert c['symmetric_known_accounting']['excluded_supplied_union']==[]
  checks.append(dict(stage=stage,condition=name,rows=len(boxes),stop=row['stop'],release=c['release']))
receipts=list((R/'stage1/runtime').glob('*/309264/prefix/receipt.json'))+list((R/'stage2/runtime').glob('*/receipt.json'))+[R/'stage1/score-runtime/receipt.json',R/'stage2/component-capture/receipt.json']
identities=[]
for p in receipts:
 d=json.loads(p.read_text());assert d['status']=='candidate_complete'
 identities.append(d.get('loaded_model_identity',d.get('loaded_identity')))
assert all(x==identities[0] for x in identities)
result=dict(status='passed',JSON_exact_consumer_replays=2,cells=len(checks),raw_checks=checks,model_identity_exact_across_receipts=len(receipts),manifests=[bind(R/'stage1-reduction-manifest.json'),bind(R/'stage2-final-manifest.json')],consumer=bind(Path(__file__)),source_grounded_credit_falsification=bind(R/'consumer-check/exclusion-check.json'),scope='Independent saved-output/identity checks; no model calls or extra physical labels.')
(R/'independent-output-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['raw_checks','manifests','consumer','source_grounded_credit_falsification']}))
