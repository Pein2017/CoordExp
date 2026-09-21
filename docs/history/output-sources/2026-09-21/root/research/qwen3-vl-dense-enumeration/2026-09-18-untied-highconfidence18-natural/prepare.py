from pathlib import Path
import json,copy,re,hashlib,os
from probes.training_set_completion.untied_shared import ROOT,TIED,UNTIED,BASE,SOURCE
from probes.training_set_completion.readout_norm_fresh import _binding
from src.qwen.runtime_loading import load_qwen_components_from_options,QwenLoadOptions
from src.config.inference import InferConfig
from src.data.examples import raw_example_from_jsonl_row
from src.inference.inputs import plan_examples
R=ROOT
oldpath=R.parent/'2026-09-17-readout-norm-fresh128/panel.json';old=json.loads(oldpath.read_text())
hpath=R.parent/'2026-08-05-static-dynamic-owner-interface-crossover/inputs/human-refined-13.geo_sorted_xy.coord.jsonl';fpath=R.parent/'2026-09-17-history-rereading-mechanism/human-evaluation/annotation-snapshot-v1/working.norm.jsonl'
assert _binding(oldpath)['sha256']=='1521feb4b4a138f4bc2460059a40c3ac77c65a931f0b6b40308b57f2382afba1'
assert _binding(hpath)['sha256']=='5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23'
assert _binding(fpath)['sha256']=='1d8d7c6d63e982f2d5060fa96a7cbc825c0314246276b181aaefff9ee80fed85'
records=[];banks={};membership={};sources=[_binding(x) for x in [oldpath,hpath,fpath,SOURCE]]
for label,path,ids in [('human13',hpath,[1584,2299,2685,4134,5001,6040,7511,10707,13348,13923,14038,14439,16228]),('refined5',fpath,[7116,309264,351017,417044,477415])]:
 ds={int(d['image_id']):d for d in map(json.loads,path.read_text().splitlines())}
 for iid in ids:
  d=copy.deepcopy(ds[iid]);image=Path('/data/CoordExp/public_data/coco/rescale_32_1024_bbox')/d['file_name'];assert image.exists();d['images']=[os.path.relpath(image.resolve(),R)]
  for o in d['objects']:o['bbox_2d']=[int(re.fullmatch(r'<\|coord_(\d+)\|>',v)[1]) if isinstance(v,str) else v for v in o['bbox_2d']]
  d['objects'].sort(key=lambda o:(o['bbox_2d'][0],o['bbox_2d'][1]));records.append(d);key=str(iid);membership[key]=label
  banks[key]=[dict(image_id=iid,owner_id=str(o['coco_ann_id']),description=o['desc'],normalized_description=o['desc'].strip().lower(),reference_coord_bins_1000=o['bbox_2d']) for o in d['objects']]
  sources.append(_binding(image))
assert len(records)==18 and sum(len(d['objects']) for d in records)==570
for d in records:
 for o in d['objects']:o['bbox_2d']=[f'<|coord_{v}|>' for v in o['bbox_2d']]
inp=R/'refined18.runtime.jsonl';inp.write_text(''.join(json.dumps(d)+'\n' for d in records));configs={}
for key,ck in [('tied',TIED),('untied',UNTIED)]:
 c=copy.deepcopy(old['config']);c['adapter']['path']=str(ck/'adapter');c['embedding_delta']['path']=str(ck/'special_token_embeddings');c['data']['input_jsonl']=str(inp);c['generation']['batch_size']=4;c['run']['artifact_root']=str(R);configs[key]=c
 for sub in ['adapter','special_token_embeddings']:
  sources += [_binding(x) for x in sorted((ck/sub).iterdir()) if x.is_file()]
 if (ck/'inference_payload_manifest.json').exists():sources.append(_binding(ck/'inference_payload_manifest.json'))
q=load_qwen_components_from_options(QwenLoadOptions(base_model=BASE,dtype='fp32',attn_implementation='sdpa',patch_embed_linearization='enabled',load_model=False));c=InferConfig.model_validate(configs['tied']);cases=[];teachers={}
for i,d in enumerate(records):
 raw=raw_example_from_jsonl_row(d,jsonl_path=inp,row_number=i+1,raw_line=json.dumps(d));plan,=plan_examples([raw],config=c,components=q,row_indices=[i]);im=plan.image
 case=dict(row_id=str(raw.example_id),row_index=i,input_record=d,image_path=str(im.image_path),image_width=d['width'],image_height=d['height'],image_plan=dict(backend_prompt_token_count=len(plan.prompt.expected_executed_prompt_token_ids),image_content_sha256=im.image_content_sha256,logical_transform_id=im.logical_transform_id,merged_visual_tokens=im.merged_visual_tokens,observed_image_grid_thw=list(im.expected_image_grid_thw)))
 cases.append(case);text=''.join('<|object_ref_start|>'+o['desc']+'<|object_ref_end|><|box_start|>'+''.join(o['bbox_2d'])+'<|box_end|>' for o in d['objects']);tokens=q.tokenizer.encode(text,add_special_tokens=False)+[151645];teachers[str(d['image_id'])]=dict(text=text,token_ids=tokens,objects=d['objects'],case=case)
groups=copy.deepcopy([g for g in old['groups'] if g['cohort']=='fresh']);assert len(groups)==32
existing={int(g['cases'][j]['input_record']['image_id']) for g in groups for j in range(len(g['cases']))};assert set(map(lambda d:int(d['image_id']),records))&existing=={309264}
remaining=[c for c in cases if int(c['input_record']['image_id'])!=309264]
for i in range(0,len(remaining),4):groups.append(dict(key=f'refined-{i//4:02d}',cohort='refined',input_jsonl=str(inp),cases=remaining[i:i+4]))
assert sum(len(g['cases']) for g in groups)==145 and len(groups)==37
panel=dict(schema='untied_18_sentinel.v1',configs=configs,groups=groups,refined_cases=cases,teachers=teachers,refined_banks=banks,sentinel_banks={str(c['input_record']['image_id']):old['banks'][str(c['input_record']['image_id'])] for g in groups[:32] for c in g['cases']},memberships=membership,refined_source_order=[d['image_id'] for d in records],sources=sources,conditions=['tied-original','tied-normalized','untied-original','untied-normalized'],bounds=dict(images=145,outputs=580,batches=148,qualification_batches=8,forwards=500000,gpu_hours=40),tolerances=dict(logit_atol=0.0002,rank_claim='2*max_abs_error < relevant margin',no_op='bitwise',payload='bitwise'),input_jsonl=str(inp))
assert sum(map(len,panel['sentinel_banks'].values()))==919
(R/'panel.json').write_text(json.dumps(panel,indent=2)+'\n');print('panel',len(groups),len(cases),len(sources))
