"""Bind fixed seven-image numerical feedback panel; never mutate source records."""
import json,hashlib,copy,os
from pathlib import Path
from PIL import Image
R=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback')
A=R.parent/'2026-09-18-untied-highconfidence18-natural'
def bind(p):p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
def main():
 p=json.loads((A/'panel.json').read_text());ids=[885,5586,7511,14038,632,417044,309264];groups=[copy.deepcopy(g) for g in p['groups'] if any(c['input_record']['image_id'] in ids for c in g['cases'])];saved=[]
 for g in groups:
  g['focus_ids']=[c['input_record']['image_id'] for c in g['cases'] if c['input_record']['image_id'] in ids]
  for condition in p['conditions']:
   base=A/'runtime'/condition/g['key'];saved.append(dict(condition=condition,group=g['key'],raw=bind(base/'raw.json'),trace=bind(base/'trace.json'),receipt=bind(base/'receipt.json')))
 source=Path('/data/CoordExp/public_data/coco/rescale_32_1024_bbox/val.jsonl');records={x['image_id']:x for x in map(json.loads,source.open()) if x['image_id'] in [885,5586,632]};cases=[];runtime=R/'new-three.runtime.jsonl'
 for i,iid in enumerate([885,5586,632]):
  row=copy.deepcopy(records[iid]);image=source.parent/row['images'][0];im=Image.open(image);row['images']=[os.path.relpath(image,R)]
  row['objects']=sorted(row['objects'],key=lambda o:(o['bbox_2d'][0],o['bbox_2d'][1]))
  for obj in row['objects']:obj['bbox_2d']=[f'<|coord_{v}|>' for v in obj['bbox_2d']]
  cases.append(dict(row_id=f'coco2017_val_{iid:012d}',row_index=i,input_record=row,image_path=str(image),image_width=im.width,image_height=im.height,image_binding=bind(image)))
 runtime.write_text(''.join(json.dumps(c['input_record'])+'\n' for c in cases));groups.append(dict(key='val-extra',cohort='numerical-feedback',cases=cases,focus_ids=[885,5586,632],input_jsonl=str(runtime)))
 out=dict(case_order=ids,configs=p['configs'],conditions=p['conditions'],groups=groups,saved_sources=saved,new_group='val-extra',sources=[bind(A/'panel.json'),bind(A/'shared-gate.json'),bind(source),bind(Path('src/templates/renderer.py')),bind(Path('probes/training_set_completion/untied_shared.py'))],tolerances=p['tolerances'],bounds=dict(model_forwards=400000,gpu_seconds=57600,tensor_bytes=32*1024**3,natural_new_outputs=12,max_boundaries=28,max_releases=364),frozen_rules=dict(delays=[1,2,4],episode='earliest start of >=3 exact description+coords rows, otherwise earliest3 near run all pairwise coordinate differences<=8',healthy='nonrecurrent complete row with following native row; minimize validity mismatch, total coordinate distance to failure row, then prefix token length distance, then row index',candidate_preference='validity+previous-row lexicographic x1,y1 order-stratum match preferred; relax ordering then validity only if required; record changes',near_threshold=8,release_rows=32,release_tokens=512),runtime_difference='New three-image native bs3 group; other four focus images retain existing heterogeneous groups. Production backend/dtype/grouping not equated. Other group members are companions, not new cohort targets.')
 (R/'panel.json').write_text(json.dumps(out,indent=2)+'\n');print('groups',len(groups),'reused focus outputs16; new focus outputs12')
if __name__=='__main__':main()
