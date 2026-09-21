import json,hashlib,copy,importlib.util
from pathlib import Path
from PIL import Image
from src.config.inference import InferConfig
from probes.training_set_completion import paired_evaluation as match,source256_evaluation as metrics
from src.inference.parsing import parse_compact_object_box_closed
from src.qwen.runtime_loading import QwenLoadOptions,load_qwen_components_from_options
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import prepare_native_inputs
B=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration');OLD=B/'2026-09-16-corner-loop-bridge-factorial';R=B/'2026-09-16-endpoint-loop-image-history';R.mkdir(exist_ok=False)
read=lambda p:json.loads(p.read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
p=read(OLD/'panel.json');datafile=Path(p['config']['data']['input_jsonl']);data=[json.loads(x) for x in datafile.read_text().splitlines()];records={x['image_id']:(i,x) for i,x in enumerate(data)};prepfile=B/'2026-09-16-source256-fixed-prefix-completion/preparation/source256-admitted-v1/preparation.json';bank={x['image_id']:x for x in read(prepfile)['bank']['records']}
def targets(i):return [metrics._target(image_id=i,owner_id=o['owner_id'],description=o['description'],coord_bins=o['coord_bins']) for o in bank[i]['owners']]
allrows=[];sources=[]
for f in sorted((B/'2026-09-16-source256-output-ranking-repair/runtime/main-v1/R/readback/train').glob('shard-*.json')):
 sources.append(bind(f));allrows += [(f,x) for x in read(f)['generation']['rows']]
assert len(allrows)==256;excluded={x['image_id'] for x in p['batch_group']};audits=[];eligible=[]
for f,g in sorted(allrows,key=lambda a:a[1]['image_id']):
 i=g['image_id'];ri,row=records[i];reason=[]
 if i in excluded:reason.append('original_or_companion')
 if g['observed_image_grid_thw']!=[1,52,78]:reason.append('grid')
 if g['prompt_token_ids']!=p['batch_group'][1]['prompt_token_ids']:reason.append('prompt_ids')
 if g['decode_stop_reason']!='im_end':reason.append('not_EOS')
 path=(datafile.parent/row['images'][0]).resolve();dims=Image.open(path).size
 if dims!=(1248,832):reason.append('executed_dimensions_under_no_resize')
 context=dict(image_width=row['width'],image_height=row['height'],row_id=g['example_id'],row_index=ri)
 native=parse_compact_object_box_closed(g['raw_decode_text'],**context).to_artifact_dict();valid,drops=match._matchable_rows_with_geometry_debt({**native,'pred':native['predictions']})
 for n,x in enumerate(valid):x['prediction_id']=f'baseline-{n}'
 ledger=match._ledger_image(targets(i),valid,threshold=.5);repeats=metrics._strict_repeat_rows(valid)
 if drops:reason.append('parser_or_geometry_debt')
 if repeats:reason.append('strict_repeat')
 if ledger['matched_count']<2:reason.append('less_than2_matched')
 audits.append(dict(image_id=i,eligible=not reason,reasons=reason,grid=g['observed_image_grid_thw'],dimensions=list(dims),matched=ledger['matched_count'],drop_count=len(drops),repeat_count=len(repeats),stop=g['decode_stop_reason']))
 if not reason:eligible.append((f,g,row,ri,path,ledger))
qualification=dict(selection='smallest eligible image_id; no inventory preference or intervention outcome search',candidates=256,eligible_count=len(eligible),eligible_ids=[x[1]['image_id'] for x in eligible],audit=audits,sources=sources+[bind(datafile),bind(prepfile)])
(R/'qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
if not eligible:print('BLOCKED no eligible donors');raise SystemExit(0)
f,g,row,ri,path,ledger=eligible[0];donor=copy.deepcopy(p['cases'][1]);donor.update(row_id=g['example_id'],row_index=ri,input_record=row,image_path=str(path),image_width=row['width'],image_height=row['height']);donor['image_plan'].update(image_content_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),executed_media_sha256=g['executed_media_sha256'])
# CPU-only real processor geometry and prompt check, no model forwards.
cfg=p['config'];q=load_qwen_components_from_options(QwenLoadOptions(base_model=cfg['model']['base_model'],dtype='fp32',attn_implementation='sdpa',patch_embed_linearization=cfg['backend']['hf']['patch_embed_linearization'],load_model=False));cases=copy.deepcopy(p['cases']);cases[1]=donor;req,_=build_bound_native_requests(q,cfg,cases);batch=prepare_native_inputs(q.processor,req,device='cpu',record_media_identity=True)
assert [list(x) for x in batch.prompt_token_ids]==[x['prompt_token_ids'] for x in p['batch_group']];assert list(batch.image_grids[1])==[1,52,78];assert batch.media_sha256[1]==g['executed_media_sha256'];assert batch.inputs['input_ids'].shape== (4,1362)
packet=copy.deepcopy(p);packet['schema']='endpoint_image_history.v1';packet['donor']=dict(image_id=g['image_id'],case=donor,generation=g,baseline_ledger=ledger,targets=targets(g['image_id']),source=bind(f),image=bind(path),processor_pixel_values_shape=list(batch.inputs['pixel_values'].shape));packet['original_cases']=p['cases'];packet['donor_cases']=cases;packet['cells']={k:p['cells'][k] for k in ['C00','C10']};packet['original_targets']=p['owner_targets'];packet['sources']+=sources+[bind(datafile),bind(prepfile),bind(path),bind(OLD/'panel.json'),bind(OLD/'producer.py'),bind(OLD/'reduce.py'),bind(OLD/'lead-acceptance.json'),bind(B/'2026-09-16-endpoint-loop-readout-state/lead-acceptance.json')]+[bind(OLD/f'runtime/{c}/{f}.json') for c in ['C00','C10'] for f in ['raw','receipt']];packet['ceilings']=dict(scientific_cells=2,identity_cells=1,model_forwards=10000,intended_model_forwards=9252,allocated_gpu_seconds=7200,elapsed_seconds=7200,total_cap=3084,supplied_tokens=1233,free_tokens=1851);packet['qualification']=bind(R/'qualification.json');packet['credit']='PRIMARY free suffix ONLY, own frozen bank; score both banks diagnostically with class compatibility; supplied history zero credit'
(R/'panel.json').write_text(json.dumps(packet,indent=2)+'\n');(R/'qualify.py').write_text(Path(__file__).read_text());print('qualified',len(eligible),'selected',g['image_id'],'baseline',ledger['matched_count'],'owners',len(targets(g['image_id'])),'classes',sorted({x['description'] for x in targets(g['image_id'])}))
