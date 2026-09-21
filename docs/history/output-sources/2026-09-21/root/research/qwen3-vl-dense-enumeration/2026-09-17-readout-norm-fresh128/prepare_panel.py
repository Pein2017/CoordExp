import copy,hashlib,json,math,shutil,os
from pathlib import Path
from transformers import AutoProcessor
from src.data.examples import raw_example_from_jsonl_row
from src.config.models import TemplateConfig
from src.inference.prompt import build_prompt_record
from PIL import Image
R=Path(__file__).resolve().parent;OLD=R.parent/'2026-09-16-endpoint-loop-natural-readout-norm'
read=lambda p:json.loads(Path(p).read_text())
def bind(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
def main():
 old=read(OLD/'panel.json');cfg=copy.deepcopy(old['config']);cohort=read(R/'cohort/cohort_manifest.json');data=Path(cohort['source_population']['processed_root'])/'train.coord.jsonl';records=[json.loads(x) for x in (R/'cohort/exact128records.jsonl').read_text().splitlines()]
 native_records=[]
 for row in records:
  row=copy.deepcopy(row);row['images']=[os.path.relpath((data.parent/row['images'][0]).resolve(),R)];row['objects'].sort(key=lambda obj:tuple(int(v[len('<|coord_'):-2]) for v in obj['bbox_2d'][:2]));native_records.append(row)
 native=R/'runtime-records.jsonl';native.write_text(''.join(json.dumps(row)+'\n' for row in native_records));records=native_records;data=native
 processor=AutoProcessor.from_pretrained(cfg['model']['base_model']);ps=processor.image_processor.patch_size;merge=processor.image_processor.merge_size;assert ps==16 and merge==2
 tc=TemplateConfig(**{k:cfg['template'][k] for k in ['object_field_order','object_ordering','assistant_format','prompt']});cases=[];sources=list(old['sources'])+[bind(native)]+[bind(OLD/'lead-acceptance.json'),bind(R/'cohort/cohort_manifest.json'),bind(R/'cohort/receipt.json'),bind(R/'cohort/known_owner_targets.json'),bind(R/'cohort/exact128records.jsonl')]
 for i,row in enumerate(records):
  raw=raw_example_from_jsonl_row(row,jsonl_path=data,row_number=i+1,raw_line=json.dumps(row));path=raw.image.path;w,h=Image.open(path).size;assert (w,h)==(row['width'],row['height']) and w%(ps*merge)==h%(ps*merge)==0;grid=[1,h//ps,w//ps];visual=math.prod(grid)//4
  prompt=build_prompt_record(raw,tc,processor=processor,row_index=i,merged_visual_tokens=visual,object_order_seed=None)
  cases.append(dict(row_id=str(raw.example_id),row_index=i,input_record=row,image_path=str(path),image_width=w,image_height=h,image_plan=dict(backend_prompt_token_count=len(prompt.expected_executed_prompt_token_ids),image_content_sha256=bind(path)['sha256'],logical_transform_id='identity',merged_visual_tokens=visual,observed_image_grid_thw=grid)))
  sources.append(bind(path))
 groups=[dict(key=f'fresh-{j//4:02d}',cohort='fresh',cases=cases[j:j+4],focus_ids=[c['input_record']['image_id'] for c in cases[j:j+4]],input_jsonl=str(data)) for j in range(0,128,4)]
 for g in old['groups']:
  g=copy.deepcopy(g);g['cohort']='diagnostic';g['saved_norm_rows']=read(OLD/f"runtime/{g['key']}-norm/raw.json")['rows'];groups.append(g)
 banks={k:v for bysplit in read(R/'cohort/known_owner_targets.json').values() for k,v in bysplit.items()};banks.update(old['banks']);shutil.copyfile(OLD/'coefficients.pt',R/'coefficients.pt')
 panel=dict(schema='fresh128_readout_norm.v1',config=cfg,coordinate_ids=old['coordinate_ids'],coefficients=bind(R/'coefficients.pt'),groups=groups,banks=banks,sources=list({x['path']:x for x in sources}.values()),producer=bind('probes/training_set_completion/readout_norm_fresh.py'),cohort=bind(R/'cohort/cohort_manifest.json'),bounds=dict(planned_batch_executions=72,model_forward_ceiling=250000,allocated_gpu_seconds=36000,elapsed_seconds=7200,max_new_tokens=3084),review=dict(images=32,seed=19,strata=['loss','gain','burden_only','stable']))
 (R/'panel.json').write_text(json.dumps(panel,indent=2)+'\n');print(json.dumps(dict(fresh_cases=len(cases),groups=len(groups),banks=len(banks),source_count=len(panel['sources']),prompt_counts=sorted({c['image_plan']['backend_prompt_token_count'] for c in cases}))))
if __name__=='__main__':main()
