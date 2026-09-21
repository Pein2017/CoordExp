import json,hashlib,math
from pathlib import Path
import torch
from PIL import Image
from probes.training_set_completion import source256_evaluation as metrics
B=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration');OLD=B/'2026-09-16-corner-loop-bridge-factorial';READ=B/'2026-09-16-endpoint-loop-readout-state';R=B/'2026-09-16-endpoint-loop-natural-readout-norm';R.mkdir(exist_ok=False)
read=lambda p:json.loads(p.read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
p=read(OLD/'panel.json');cfg=p['config'];datafile=Path(cfg['data']['input_jsonl']);rows=[json.loads(x) for x in datafile.read_text().splitlines()];records={x['image_id']:(i,x) for i,x in enumerate(rows)};prep=B/'2026-09-16-source256-fixed-prefix-completion/preparation/source256-admitted-v1/preparation.json';bank={x['image_id']:x for x in read(prep)['bank']['records']};focus=[351017,417044,477415,7116];groups=[];sources=list(p['sources'])+[bind(prep),bind(datafile),bind(READ/'lead-acceptance.json'),bind(B/'2026-09-16-endpoint-loop-image-history/lead-acceptance.json')]
for f in sorted((B/'2026-09-16-source256-output-ranking-repair/runtime/main-v1/R/readback/train').glob('shard-*.json')):
 shard=read(f);gen=shard['generation']['rows']
 for bi in sorted({x['batch_index'] for x in gen if x['image_id'] in focus}):
  g=[x for x in gen if x['batch_index']==bi];assert len(g)==4;cases=[]
  for x in g:
   idx,row=records[x['image_id']];path=(datafile.parent/row['images'][0]).resolve();w,h=Image.open(path).size;assert (w,h)==(row['width'],row['height']);grid=x['observed_image_grid_thw'];cases.append(dict(row_id=x['example_id'],row_index=idx,input_record=row,image_path=str(path),image_width=w,image_height=h,image_plan=dict(backend_prompt_token_count=len(x['prompt_token_ids']),executed_media_sha256=x['executed_media_sha256'],image_content_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),logical_transform_id='identity',merged_visual_tokens=math.prod(grid)//4,observed_image_grid_thw=grid)));sources.append(bind(path))
  groups.append(dict(key=f'{f.stem}-batch{bi}',rows=g,cases=cases,focus_ids=[x['image_id'] for x in g if x['image_id'] in focus],source=bind(f)));sources.append(bind(f))
assert len(groups)==4
anchor=READ/'runtime/R16-351017/tensors.pt';tensor=torch.load(anchor,map_location='cpu',weights_only=True);W=tensor['output_rows'];assert W.dtype==torch.float32 and torch.equal(W,tensor['input_rows']) and torch.count_nonzero(tensor['bias'])==0;norms=W.double().norm(dim=1);median=norms.median();factors=median/norms;torch.save(dict(norms=norms,median=median,factors=factors,effective_rows_sha256=hashlib.sha256(W.contiguous().numpy().tobytes()).hexdigest()),R/'coefficients.pt');sources.append(bind(anchor));firstrows={}
readpanel=read(READ/'panel.json')
for c in readpanel['cells']:
 row=next(x for x in c['rows'] if x['row']==1 and x['key']=='P-row1');firstrows[str(c['image_id'])]=row;path=READ/f"runtime/R16-{c['image_id']}/tensors.pt";sources.append(bind(path))
packet=dict(schema='endpoint_natural_readout_norm.v1',focus_images=focus,config=cfg,groups=groups,sources=sources,coefficients=bind(R/'coefficients.pt'),coefficients_summary=dict(median=float(median),minimum_factor=float(factors.min()),maximum_factor=float(factors.max()),endpoint_factors={str(i):float(factors[i]) for i in [0,999]},effective_rows_sha256=hashlib.sha256(W.contiguous().numpy().tobytes()).hexdigest()),coordinate_ids=list(range(151670,152670)),banks={str(i):[metrics._target(image_id=i,owner_id=o['owner_id'],description=o['description'],coord_bins=o['coord_bins']) for o in bank[i]['owners']] for i in focus},prior_first_rows=firstrows,bounds=dict(batch_executions=8,max_forwards_per_batch=3084,max_total_forwards=25000,allocated_gpu_seconds=7200,elapsed_seconds=7200,max_new_tokens=3084),policy='For ALL steps/samples: coordinate logits z_c * FP64 median(norm)/norm, rounded back to native logits dtype; non-coordinate logits bitwise untouched. No parameter/input mutation. Empty-prefix natural greedy.')
(R/'panel.json').write_text(json.dumps(packet,indent=2)+'\n');(R/'prepare.py').write_text(Path(__file__).read_text())
U=Path('research/experiments')/R.name;U.mkdir();(U/'unit.md').write_text('''# Natural rollout with equalized coordinate output norms

From fixed R16 on351017,417044,477415 and healthy7116 with EMPTY PREFIX, does removing measured coordinate OUTPUT-row norm structure improve harmful natural loops while retaining incumbent known owners and avoiding healthy-image damage?

## Frozen policy and panel

Exactly4 existing images,4 unique original saved heterogeneous bs4 groups,each paired identity/normalized policy. Original companions/order/left padding/prompt/media/native MRoPE/FP32-SDPA/RP1/max3084 are preserved; no rebatching. Saved group identities and every input are in panel.json. Treatment affects all batch samples, but only4 frozen focus images support scientific conclusions.

At every step and sample scale ALL1000 coordinate logits by m/||W_c||, with effective FP32 W=base+shared_delta, FP64 norms and fixed torch median. Fixed coefficients.pt computed BEFORE GPU execution from accepted R16 readout tensors. Bias must be absent, live effective rows and coefficient identity reverified. No endpoint-only ban, parser role gating, strength grid or parameter mutation. Non-coordinate logits, including EOS, are exactly unchanged at the seam. This changes coordinate-vs-noncoordinate competition as well as relative coordinate scores. Inputs stay fixed as parameters; differing generated tokens naturally change subsequent hidden states. It is a diagnostic deployed policy, not learned repair or proof of embedding innocence.

## Admission and evidence

Each original-policy batch must reproduce ALL saved member tokens/stops exactly before its normalized pair is accepted/launched. The first real identity/scaling executions belong to these8 cells,not extra smoke. Narrow native logits-processor seam only; no model/checkpoint/input embedding/DoRA/vision/KV mutation. Verify tied base/shared delta effective rows before/after and all parameter version counters; executable non-coordinate identity and factor correctness checks. At directly available SAME first-row prefixes compare native/scaled coordinate decisions with accepted offline R16 tensors,without extra replay forwards. Save first policy fork with full-vocabulary logits before/after,EOS/ranks/field and history identity.

Primary full-natural-output class-agnostic IoU50 own-bank matches,FN,incumbent G/L/identity retention; include strict valid duplicates AND literal invalid repeats,UNKNOWN,invalid/malformed,length/EOS/cap,first exact divergence/field,longest repeated runs,all-coordinate/per-role endpoint occupancy. Healthy7116 baseline4/6/EOS is mandatory. Label-string diagnostics stay separate. No physical truth inferred from unmatched outputs.

## Bounds, interpretation and stop

At most8 batch executions,each≤3084 calls,hard25000 total including mechanical work;≤7200 allocated GPU-seconds and≤7200 elapsed. No retries beyond this ceiling or baseline change; technical-invalid stays unanswered. No new images/checkpoints/seeds/layers,selective endpoint or input-only arm,strength tuning,training,labels/architecture changes,agents or review expansion. Stop after4-image reduction either outcome.

Useful gains WITH retained incumbents and less loop debt support norm structure contribution on this selected cohort. Persistence or healthy-border/owner damage is material. Endpoint rate,shorter output or valid-box count alone is not success. No inference of training origin,generalization,unique causal circuit or ordinary-parameter learned repair. Any further question is a separate lead decision,not another normalization variant.

Predecessors: accepted token edits gave selective transient recovery; both one-bin controls failed; offline norm removal changed17/69 endpoint wins while52 survived; natural-image substitution changed loop form without donor-known recovery. This removes the foreign-history confound by evaluating the single readout policy from native empty prefixes.

Immutable coefficient/group/model/scorer bindings: '''+str(R/'panel.json')+'\n')
s=dict(schema_version=1,unit_id=R.name,lifecycle='ready',evidence='none',disposition='frozen_four_image_natural_readout_norm',state_as_of='2026-09-16',protocol=str(U/'unit.md'),result=None,state_source=str(U/'unit.md'),boundary='4 original images,4 original native groups,identity+single normalized policy,no supplied history',not_authorized=['normalization variants','training','new images','parameter/KV interventions','agents','automatic successor'],next_action='Execute at most8 frozen batch cells,reduce and stop');(U/'state.json').write_text(json.dumps(s,indent=2)+'\n')
cat=Path('research/experiments/catalog.jsonl');cat.write_text(cat.read_text()+json.dumps(dict(id=R.name,title='Four-image natural coordinate-readout norm policy',kind='experiment',topics=['history-repetition-stopping'],record_root=str(U),protocols=[str(U/'unit.md')],result_records=[],reading_entry=str(U/'unit.md'),tracking='current',state=str(U/'state.json')))+'\n');idx=Path('research/index.md');idx.write_text(idx.read_text()+f'\nNatural norm-policy [unit](experiments/{R.name}/unit.md), [state](experiments/{R.name}/state.json):4 original images,identity and one equal-norm policy; stop after panel.\n');print(json.dumps(dict(groups=[(g['key'],g['focus_ids']) for g in groups],coefficients=packet['coefficients_summary']),indent=2))
