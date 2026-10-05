"""One frozen transfer image: ordinary trajectory, literal entrance, fixed queries."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from probes import box_continuity as continuity, coordinate_readout as readout
from probes.coordinate_diagnostics import spatial_grid as grid, visual_state as visual
from probes.rule_stability import artifacts as a, objectives

ROOT = a.ROOT
OUTPUT = ROOT / 'outputs/research/physical-fn-recovery/2026-10-04/spatial-grid-transfer'
UNIT = ROOT / 'research/experiments/2026-10-04-spatial-grid-transfer'
PRIOR = grid.OUTPUT / 'prepared-01/input-packet.json'
SOURCE = Path('/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000_xy_sorted/val.coord.jsonl')
EXCLUSION = ROOT / 'research/experiments/2026-10-02-full-label-self-rollout-fit/inputs/full-labels.json'
HISTORICAL = '365cd55d169e5292b60bb75b672c0a65813df5b0'
SCHEMA = 'spatial-grid-transfer-v1'
SOURCE_PATHS = ['probes/coordinate_diagnostics/spatial_transfer.py',
                'tests/probes/coordinate_diagnostics/test_spatial_transfer.py']
GRID, COORD_START, COORD_IDS = grid.GRID, grid.COORD_START, grid.COORD_IDS
binding, revision, cpu_diagnostics = grid.binding, grid.revision, grid.cpu_diagnostics
BOUNDS = dict(grid.BOUNDS, requests=34, actions=5311, forced_actions=2094, free_actions=3217,
    prefills=34, cached_predictions=5277, maximum_actions_per_request=3084,
    maximum_context_tokens=4608, wall_seconds=1200, gallery_seconds=300, retained_bytes=512*1024**2)
HEADER = '<|object_ref_start|>bottle<|object_ref_end|><|box_start|>'


def producer_identity():
    diff = subprocess.check_output(['git', 'diff', 'HEAD', '--', *SOURCE_PATHS], cwd=ROOT)
    return dict(commit=revision(), diff_sha256=hashlib.sha256(diff).hexdigest(),
                files={p:a.digest(ROOT/p) for p in SOURCE_PATHS})


def selection():
    excluded = sorted(int(i['image_id']) for i in a.load(EXCLUSION))
    eligible, selected = [], None
    with SOURCE.open() as stream:
        for line, text in enumerate(stream, 1):
            row = json.loads(text); image = int(row['image_id'])
            count = sum(o['category_name']=='bottle' for o in row['objects'])
            if image not in excluded and count >= 10:
                eligible.append(dict(image_id=image, line_one_based=line, bottle_annotations=count))
                if selected is None or image < selected[0]['image_id']:selected = (eligible[-1], row)
    eligible.sort(key=lambda e:e['image_id'])
    if (len(eligible)!=17 or selected[0]!=dict(image_id=25394,line_one_based=248,bottle_annotations=13)
            or len(selected[1]['objects'])!=24 or len(excluded)!=18):
        raise ValueError('frozen category-count/exclusion selection differs')
    return dict(eligible=eligible, excluded_image_ids=excluded, selected=selected[0]), selected[1]


def image_record(row):
    objects = [dict(o, bbox_2d=[int(re.fullmatch(r'<\|coord_(\d+)\|>',v).group(1)) for v in o['bbox_2d']])
               for o in row['objects']]
    path = (SOURCE.parent/row['images'][0]).resolve()
    if (row['image_id']!=25394 or (row['width'],row['height'])!=(864,1152) or
            str(path)!='/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/val2017/000000025394.jpg'):
        raise ValueError('selected source/frame differs')
    return dict(image_id=25394, image_path=str(path), width=864, height=1152, objects=objects)


def definitions(anchor):
    return [grid.definitions(anchor)[0], dict(condition='baseline',image_id=25394,budget=3084,
        forced_actions={},observations=[],input_region='clean',query_index=None,x1=None),
        *[dict(condition=f'grid-{j:02d}',image_id=25394,query_index=j,x1=x) for j,x in enumerate(GRID)]]


def first_header(record, tokenizer):
    """Same causal boundary certification as maintained geometry, before any x1."""
    boundaries, _, mapping = objectives._literal_action_boundaries(record,tokenizer,[])
    dispositions=[]
    for match in re.finditer(re.escape(HEADER),record['text']):
        end=boundaries[match.end()]
        prefix=tokenizer.decode(record['token_ids'][:end+1],skip_special_tokens=False)
        if prefix!=record['text'][:match.end()]:
            dispositions.append(dict(char_start=match.start(),reason='header_action_crosses_frame'))
            continue
        start=boundaries[match.start()+1];t=end+1
        return dict(status='eligible' if t<=64 else 'unavailable',reason=None if t<=64 else 'first_header_late',
            position=t,row_start=start,header_char_span=[match.start(),match.end()],
            prefix_token_ids=record['token_ids'][:t],mapping=mapping,dispositions=dispositions,
            next_native_action=record['token_ids'][t] if t<len(record['token_ids']) else None)
    return dict(status='unavailable',reason='no_certified_literal_bottle_header',position=None,
                row_start=None,prefix_token_ids=None,mapping=mapping,dispositions=dispositions)


def query_cell(slot, entrance):
    if entrance['status']!='eligible':raise ValueError('no eligible native entrance')
    t=entrance['position'];prefix=entrance['prefix_token_ids']
    if len(prefix)!=t or not 0<=t<=64:raise ValueError('dynamic entrance boundary differs')
    return dict(slot,budget=t+5,position=t,row_start=entrance['row_start'],expected_ids=prefix,
        input_region='clean',observations=list(range(t,t+4)),
        forced_actions={**{str(i):token for i,token in enumerate(prefix)},str(t):COORD_START+slot['x1']})


def frontend_batch(q, packet, key):
    from probes import hidden_human_recovery as recovery
    request=packet['requests'][key]
    batch=recovery.native_request(request,a.load(packet['bindings']['policy']['path']),q.processor)
    if visual.batch_identity(batch)!=packet['processor'][key]:raise ValueError('executed prompt/media/grid/tensors differ')
    return batch


def prepare(directory):
    from PIL import Image
    from probes import rollout_row_credit as retained
    from probes import hidden_human_recovery as recovery
    from probes.rule_stability.__main__ import runtime_identity
    old=a.load(PRIOR)
    if old['runtime']!=runtime_identity():raise ValueError('qualified runtime changed')
    for path,sha in old['cached_pipeline_files'].items():
        if a.digest(ROOT/path)!=sha:raise ValueError('qualified helper changed:'+path)
    continuity.check_payloads(old['payloads'])
    inventory,row=selection();image=image_record(row)
    with Image.open(image['image_path']) as rgb:
        if rgb.mode!='RGB' or rgb.size!=(864,1152):raise ValueError('original RGB frame differs')
    request=dict(image_id=25394,request_id='spatial-transfer:25394:clean',image_path=image['image_path'],
        image_sha256=a.digest(image['image_path']),crop=[0,0,864,1152],view_scale=1,width=864,height=1152)
    q=retained.frontend()
    if q.model is not None:raise ValueError('CPU preparation loaded a model')
    policy=a.load(old['bindings']['policy']['path'])
    batch=recovery.native_request(request,policy,q.processor);identity=visual.batch_identity(batch)
    request.update({k:identity[k] for k in ('prompt_token_ids','image_grid_thw','media_sha256')})
    if len(identity['prompt_token_ids'])+3084>4608:raise ValueError('new prompt exceeds total context bound')
    anchor=a.load(old['bindings']['anchor']['path'])
    bindings={k:old['bindings'][k] for k in ('anchor','anchor_scores','policy','checkpoint')}
    bindings.update(protocol=binding(UNIT/'unit.md'),prior_packet=binding(PRIOR),
                    selection_source=binding(SOURCE),exclusion=binding(EXCLUSION),image25394=binding(image['image_path']))
    exposure={}
    for name in ('untied_axis.yaml','untied.yaml','tied.yaml'):
        path='configs/train/geo_sorted_xy/'+name
        text=subprocess.check_output(['git','show',HISTORICAL+':'+path],cwd=ROOT,text=True)
        exposure[path]=dict(revision=HISTORICAL,sha256=hashlib.sha256(text.encode()).hexdigest(),text=text)
    pipeline=dict(old['cached_pipeline_files'])
    for path in ('probes/hidden_human_recovery.py','probes/coordinate_diagnostics/spatial_grid.py'):
        pipeline[path]=a.digest(ROOT/path)
    packet=dict(schema=SCHEMA,bounds=BOUNDS,bindings=bindings,conditions=definitions(anchor),selection=inventory,
        original_source_row=row,exposure=exposure,images={'351017':old['images']['351017'],'25394':image},
        requests={'351017':old['requests']['351017']['clean'],'25394':request},
        processor={'351017':old['processor']['351017']['clean'],'25394':identity},
        payloads=old['payloads'],runtime=old['runtime'],producer=producer_identity(),cached_pipeline_files=pipeline,
        model_loaded=False,**{k:old[k] for k in ('coordinate_ids','eos_id','object_start_id','vocabulary_size','summary_convention')})
    frontend_batch(q,packet,'351017');verify_inputs(packet)
    directory=Path(directory).resolve()
    if not directory.is_relative_to(OUTPUT):raise ValueError('preparation outside unit owner')
    directory.mkdir(parents=True,exist_ok=False);a.write(directory/'input-packet.json',packet)
    proposal=dict(schema=SCHEMA,released=False,source_revision=revision(),producer_files=packet['producer']['files'],
        input_packet=binding(directory/'input-packet.json'),runtime=packet['runtime'],bounds=BOUNDS,
        output=str(OUTPUT/'native-01'),retry='no_automatic_relaunch')
    a.write(directory/'native-proposal.json',proposal)
    return proposal


def verify_inputs(packet):
    if (packet['schema']!=SCHEMA or packet['bounds']!=BOUNDS or packet['coordinate_ids']!=COORD_IDS or
            packet['summary_convention']!='detached-original-FP32-vector_CPU-single-thread-FP32' or packet['model_loaded']):
        raise ValueError('frozen observation/resource schema differs')
    for item in packet['bindings'].values():
        if a.digest(item['path'])!=item['sha256']:raise ValueError('bound input changed:'+item['path'])
    old=a.load(packet['bindings']['prior_packet']['path']);anchor=a.load(packet['bindings']['anchor']['path'])
    if (packet['bindings']['prior_packet']!=binding(PRIOR) or
            any(packet['bindings'][k]!=old['bindings'][k] for k in ('anchor','anchor_scores','policy','checkpoint')) or
            packet['payloads']!=old['payloads'] or packet['runtime']!=old['runtime'] or
            any(packet[k]!=old[k] for k in ('eos_id','object_start_id','vocabulary_size'))):
        raise ValueError('qualified payload/anchor/runtime identity differs')
    inventory,row=selection()
    if packet['selection']!=inventory or packet['original_source_row']!=row or packet['images']['25394']!=image_record(row):
        raise ValueError('selection/ordered full source annotations differ')
    if packet['conditions']!=definitions(anchor):raise ValueError('frozen anchor/grid/action roles differ')
    if (packet['images']['351017']!=old['images']['351017'] or
            packet['requests']['351017']!=old['requests']['351017']['clean'] or
            packet['processor']['351017']!=old['processor']['351017']['clean']):raise ValueError('old anchor inputs differ')
    request=packet['requests']['25394'];identity=packet['processor']['25394'];image=packet['images']['25394']
    expected=dict(image_id=25394,request_id='spatial-transfer:25394:clean',image_path=image['image_path'],
        image_sha256=packet['bindings']['image25394']['sha256'],crop=[0,0,864,1152],view_scale=1,width=864,height=1152,
        **{k:identity[k] for k in ('prompt_token_ids','image_grid_thw','media_sha256')})
    if request!=expected or len(identity['prompt_token_ids'])+3084>4608:raise ValueError('new prompt/request/media/context differs')
    for path,item in packet['exposure'].items():
        text=subprocess.check_output(['git','show',HISTORICAL+':'+path],cwd=ROOT,text=True)
        if item!=dict(revision=HISTORICAL,sha256=hashlib.sha256(text.encode()).hexdigest(),text=text):
            raise ValueError('documented historical exposure differs')


def load_packet(path,*,cpu=False):
    from probes.rule_stability.__main__ import runtime_identity
    config=a.load(path)
    if (config['schema']!=SCHEMA or config['bounds']!=BOUNDS or config['retry']!='no_automatic_relaunch' or
            a.digest(config['input_packet']['path'])!=config['input_packet']['sha256']):raise ValueError('release/packet changed')
    packet=a.load(config['input_packet']['path']);verify_inputs(packet)
    if config['runtime']!=runtime_identity() or config['runtime']!=packet['runtime']:raise ValueError('current runtime changed')
    if config['producer_files']!=producer_identity()['files'] or config['producer_files']!=packet['producer']['files']:
        raise ValueError('producer bytes changed')
    for file,sha in packet['cached_pipeline_files'].items():
        if a.digest(ROOT/file)!=sha:raise ValueError('helper changed:'+file)
    continuity.check_payloads(packet['payloads'])
    if not cpu:
        if config['released'] is not True or config['source_revision']!=revision():raise ValueError('exact lead release required')
        if subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=ROOT):raise ValueError('native source dirty')
        subprocess.run(['git','ls-files','--error-unmatch',*SOURCE_PATHS],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('one serial GPU0 required')
    return config,packet


class NativeSession:
    def __init__(self,checkpoint,packet):
        from probes.rule_stability.runner import native_components
        from src.qwen.coordinate_policy import MedianPolicy
        self.q,self.delta,self.composition=native_components(checkpoint)
        self.q.model.eval().requires_grad_(False);self.packet=packet
        self.norm=MedianPolicy(self.q.model,COORD_IDS)
        self.batches={key:frontend_batch(self.q,packet,key) for key in packet['requests']}
        self.failure_evidence=None

    def generate(self,cell):
        import torch
        from src.qwen.generation import generate_continuations,NativeGenerationPolicy
        batch=self.batches[str(cell['image_id'])];begin=time.monotonic();ordinary=cell['condition']=='baseline'
        raw=median=None
        if not ordinary:
            raw=grid.Observer(cell,self.packet,len(batch.prompt_token_ids[0]),channel='raw')
            median=grid.Observer(cell,self.packet,len(batch.prompt_token_ids[0]),raw=raw,channel='median')
        processor=self.norm.generation_transform()
        self.failure_evidence=None
        try:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                result=generate_continuations(self.q.model,batch,extensions=[()],budgets=[cell['budget']],
                    eos_token_id=self.packet['eos_id'],pad_token_id=self.q.tokenizer.pad_token_id,
                    policy=NativeGenerationPolicy(temperature=0,top_p=1,top_k=0,repetition_penalty=1,use_model_defaults=False),
                    trace='raw_and_policy',allow_pad_tokens=True,
                    logits_processor=[processor] if ordinary else [raw,processor,median])[0]
        except Exception:
            self.failure_evidence=dict(condition=cell['condition'],selected_unconfirmed=raw.emitted if raw else None,
                raw_steps=raw.steps if raw else None,median_steps=median.steps if median else None)
            raise
        image=self.packet['images'][str(cell['image_id'])]
        record=dict(request_id=batch.request_ids[0],width=image['width'],height=image['height'],
            token_ids=list(result.token_ids),text=self.q.tokenizer.decode(result.token_ids,skip_special_tokens=False),
            stop_reason=result.stop_reason,raw_logprobs=list(result.raw_logprobs),policy_logprobs=list(result.policy_logprobs),
            generation_seconds=time.monotonic()-begin,empty_extension=True)
        if ordinary:record.update(forced_actions={},score_retention='selected-action scalars only')
        else:record.update(raw_steps=raw.steps,median_steps=median.steps,raw_observations=raw.observations,
                           median_observations=median.observations,_tensors=dict(raw.vectors,**median.vectors))
        return record

    close=readout.NativeSession.close


def validate_record(record,cell,packet,tokenizer):
    import torch
    from safetensors.torch import load_file
    image=packet['images'][str(cell['image_id'])];request=packet['requests'][str(cell['image_id'])]
    if (record.get('schema')!=SCHEMA or record.get('condition')!=cell['condition'] or
            record.get('image_id')!=cell['image_id'] or record.get('request_id')!=request['request_id'] or
            any(record.get(k)!=image[k] for k in ('width','height')) or not record.get('empty_extension') or
            tokenizer.decode(record['token_ids'],skip_special_tokens=False)!=record['text']):
        raise ValueError('record/action/text/request/frame identity differs')
    if cell['condition']=='baseline':
        ids=record['token_ids'];eos=packet['eos_id']
        if (not ids or len(ids)>3084 or eos in ids[:-1] or
                record['stop_reason']!=('im_end' if ids[-1]==eos else 'length') or
                (len(ids)!=3084 and ids[-1]!=eos) or record['forced_actions']!={} or
                record['score_retention']!='selected-action scalars only' or
                any(k in record for k in ('score_tensors','raw_steps','median_steps','raw_observations','median_observations'))):
            raise ValueError('ordinary baseline budget/stop/scalar retention differs')
        if any(len(record[k])!=len(ids) or not all(math.isfinite(v) for v in record[k])
               for k in ('raw_logprobs','policy_logprobs')):raise ValueError('baseline selected-action scalar alignment differs')
        return
    continuity.validate_record(record,cell,packet)
    if a.digest(record['score_tensors']['path'])!=record['score_tensors']['sha256']:raise ValueError('original vector bytes changed')
    vectors=load_file(record['score_tensors']['path'],device='cpu')
    if {k:visual.tensor_identity(v) for k,v in vectors.items()}!=record['tensor_identities']:raise ValueError('vector identity changed')
    keys=set()
    for channel in ('raw','median'):
        for obs in record[channel+'_observations']:
            key=f'{channel}_{obs["position"]}';keys.add(key)
            if obs['vector_key']!=key or key not in vectors:raise ValueError('score/action association differs')
            if any(obs.get(k)!=v for k,v in grid.canonical(vectors[key],packet).items()):raise ValueError('canonical CPU summary differs')
            if 'target_metrics' in obs:raise ValueError('target/owner scores forbidden')
    if keys!=set(vectors) or any(v.dtype!=torch.float32 for v in vectors.values()):raise ValueError('score vector set differs')


def overlaps(box,valid,packet):
    from src.eval.saved_rows import iou_xyxy
    return [dict(annotation_id=o['coco_ann_id'],iou=float(iou_xyxy(box,o['bbox_2d'])))
            for o in packet['images']['25394']['objects'] if o['desc']=='bottle'] if valid else None


def baseline_analysis(record,packet,tokenizer):
    parsed=objectives.trajectory_analysis(record,tokenizer);groups={}
    # The maintained parser gives malformed character spans, not action spans.
    # Attribute them to original causal actions, including actual EOS/cap tails.
    if parsed['malformed']:
        boundaries,_,_=objectives._literal_action_boundaries(record,tokenizer,[])
        targets=sorted({v for m in parsed['malformed'] for v in (m['char_start']+1,m['char_end']) if v not in boundaries})
        pending=0
        for position in range(len(record['token_ids'])) if targets else ():
            prefix=tokenizer.decode(record['token_ids'][:position+1],skip_special_tokens=False)
            while pending<len(targets) and len(prefix)>=targets[pending] and prefix.startswith(record['text'][:targets[pending]]):
                boundaries[targets[pending]]=position;pending+=1
            if pending==len(targets):break
        if pending!=len(targets):raise ValueError('uncertified malformed original action span')
        for malformed in parsed['malformed']:
            start,end=boundaries[malformed['char_start']+1],boundaries[malformed['char_end']]
            malformed.update(positions=list(range(start,end+1)),original_token_ids=record['token_ids'][start:end+1],
                             box=None,action_alignment='earliest original causal prefix completion')
    for row in parsed['rows']:
        key=(row['description'],*row['bbox'])
        if key not in groups:
            groups[key]=dict(group_index=len(groups),description=row['description'],box=row['bbox'],valid=row['valid'],occurrences=[],
                bottle_annotation_overlaps=overlaps(row['bbox'],row['valid'],packet),
                overlap_unavailable_reason=None if row['valid'] else 'invalid_original_geometry')
        groups[key]['occurrences'].append({k:row[k] for k in ('order','positions','coordinate_positions','completion_position','raw_span_sha256')})
    for group in groups.values():group['multiplicity']=len(group['occurrences'])
    return dict(groups=list(groups.values()),complete_rows=len(parsed['rows']),categories=dict(Counter(r['description'] for r in parsed['rows'])),
        malformed=parsed['malformed'],burdens=parsed['burdens'],action_mapping=parsed['action_mapping'],
        all_categories_retained=True,parsed=parsed)


def analyze(record,cell,packet,tokenizer):
    parsed=objectives.trajectory_analysis(record,tokenizer)
    row=next((r for r in parsed['rows'] if r['positions'][0]==cell['row_start']),None);selected=None
    if row is not None:
        selected=dict(description=row['description'],box=row['bbox'],valid=row['valid'],order=row['order'],
            positions=row['positions'],coordinate_positions=row['coordinate_positions'],completion_position=row['completion_position'],
            width=row['bbox'][2]-row['bbox'][0],height=row['bbox'][3]-row['bbox'][1],
            bottle_annotation_overlaps=overlaps(row['bbox'],row['valid'],packet) if cell['image_id']==25394 else None,
            overlap_unavailable_reason=None if row['valid'] else 'invalid_original_geometry',physical_recovery_credit=False)
    roles=grid.reused.causal_roles(record,cell)
    outcome='incomplete_or_structural_escape' if row is None else ('complete_valid' if row['valid'] else 'complete_invalid_geometry')
    scores={ch:[dict(position=o['position'],role=roles.get(str(o['position'])),prefix_token_ids=o['prefix_token_ids'],
        **continuity.score_summary(o,record[ch+'_steps'][o['position']])) for o in record[ch+'_observations']] for ch in ('raw','median')}
    return dict(selected_row=selected,outcome=outcome,rows=parsed['rows'],malformed=parsed['malformed'],burdens=parsed['burdens'],
        causal_roles=roles,scores=scores,actual_forced_actions=sum(str(i) in cell['forced_actions'] for i in range(len(record['token_ids']))),
        actual_free_actions=sum(str(i) not in cell['forced_actions'] for i in range(len(record['token_ids']))),
        x1_annotation_coincidences=[o['coco_ann_id'] for o in packet['images']['25394']['objects']
            if cell['x1'] is not None and o['desc']=='bottle' and o['bbox_2d'][0]==cell['x1']],physical_recovery_credit=False)


@cpu_diagnostics()
def consume(output,packet,tokenizer):
    verify_inputs(packet);output=Path(output);entries=a.load(output/'conditions.json');rows=[];anchor=False;entrance=None;baseline=None;files=set()
    if [e['condition'] for e in entries]!=[c['condition'] for c in packet['conditions']]:raise ValueError('omitted/duplicated/reordered slots')
    for entry,slot in zip(entries,packet['conditions'],strict=True):
        name=slot['condition'];base=dict(slot=slot,status=entry['status'],condition=name)
        if entry['status'] in ('skipped-HOLD','unavailable'):
            reason=entry['reason']
            if entry['status']=='unavailable':
                if name in ('anchor','baseline') or entrance is None or entrance['status']!='unavailable' or reason!=entrance['reason']:
                    raise ValueError('unjustified unavailable slot')
            elif not ((name!='anchor' and not anchor and reason=='anchor') or reason.startswith(('resource_limit:','technical_failure:'))):
                raise ValueError('unjustified HOLD slot')
            rows.append(dict(base,reason=reason,analysis=None,artifact=None));continue
        if entry['status']=='attempted-invalid':
            if not (output/entry['failure']).is_file():raise ValueError('missing failure evidence')
            files.update(entry[k] for k in ('filename','failure') if k in entry)
            rows.append(dict(base,failure=binding(output/entry['failure']),analysis=None,artifact=None));continue
        if entry['status']!='completed' or entry['filename']!=name+'.json' or (name!='anchor' and not anchor):
            raise ValueError('invalid cell/dependency publication')
        cell=slot if name in ('anchor','baseline') else query_cell(slot,entrance)
        record=a.load(output/entry['filename']);files.add(entry['filename']);validate_record(record,cell,packet,tokenizer)
        if name=='anchor':
            control=grid.fidelity(record,cell,packet);anchor=control['qualified'];analysis=analyze(record,cell,packet,tokenizer)
        elif name=='baseline':
            control=None;baseline=baseline_analysis(record,packet,tokenizer);entrance=first_header(record,tokenizer);analysis=None
        else:control=None;analysis=analyze(record,cell,packet,tokenizer)
        rows.append(dict(base,cell=cell,control=control,analysis=analysis,artifact=binding(output/entry['filename']),
                         token_ids=record['token_ids'],raw_logprobs=record['raw_logprobs'],policy_logprobs=record['policy_logprobs']))
    actual={p.name for stem in ('anchor','baseline','grid-') for p in output.glob(stem+'*.json')}
    if files!=actual:raise ValueError('extra/unconsumed cell or partial artifact')
    forced=sum(r['analysis']['actual_forced_actions'] for r in rows if r['analysis'])
    actions=sum(len(r['token_ids']) for r in rows if r['status']=='completed')
    return dict(schema=SCHEMA,complete=anchor and baseline is not None and all(r['status'] in ('completed','unavailable') for r in rows),
        anchor_qualified=anchor,entrance=entrance,baseline=baseline,conditions=rows,requested_cells=34,requested_candidates=32,
        requested_denominators=dict(anchors=1,ordinary_baselines=1,grid_candidates=32),
        attempted_requests=sum(r['status'] in ('completed','attempted-invalid') for r in rows),
        completed_requests=sum(r['status']=='completed' for r in rows),generated_actions=actions,
        actual_forced_actions=forced,actual_free_actions=actions-forced,primary_new_physical_entities=None,
        visual_ledger_status='pending_Root',limitations='One selected validation case outside documented optimization stages, not wholly unseen; supplied native category/x1. All baseline categories retained. No physical recovery credit before Root visual adjudication.')


def resources(output,begin,native):
    values=readout.usage(output,native=native)
    return values,[k for k,v in values.items() if v>BOUNDS[k]]+(['wall_seconds'] if time.monotonic()-begin>BOUNDS['wall_seconds'] else [])


def run(config_path,output,*,cpu_factory=None):
    import torch
    from safetensors.torch import save_file
    output=Path(output).resolve()
    if not output.is_relative_to(OUTPUT):raise ValueError('invocation outside unit owner')
    output.mkdir(parents=True,exist_ok=False);begin=time.monotonic();session=packet=None;entries=[];anchor=False;entrance=None;native=cpu_factory is None;code=2
    counts=dict(checkpoint_loads=0,fixture_sessions=0,attempted_requests=0,completed_requests=0,generated_actions=0,
        optimizer=0,backward=0,replay=0,training=0,warmup=0,exports=0)
    a.write(output/'invocation.json',dict(schema=SCHEMA,config=binding(config_path),pid=os.getpid(),pgid=os.getpgid(0),started=time.time(),
        cpu_threads_incoming=torch.get_num_threads(),compute='native' if native else 'CPU_FIXTURE'))
    try:
        config,packet=load_packet(config_path,cpu=not native)
        if native and str(output)!=config['output']:raise ValueError('output differs from exact release')
        a.write(output/'qualification.json',dict(source_revision=config['source_revision'],producer=config['producer_files'],
            runtime=config['runtime'],inputs=packet['bindings'],helpers=packet['cached_pipeline_files'],payloads=packet['payloads']))
        if resources(output,begin,native)[1]:raise ValueError('resource excess before load')
        start=time.monotonic();counts['checkpoint_loads' if native else 'fixture_sessions']=1
        session=(cpu_factory or NativeSession)(readout.PREVIOUS/'checkpoint-16',packet)
        a.write(output/'load-B16.json',dict(seconds=time.monotonic()-start,composition=session.composition))
        for slot in packet['conditions']:
            name=slot['condition'];excess=resources(output,begin,native)[1]
            reason='anchor' if name!='anchor' and not anchor else ''
            if excess:reason='resource_limit:'+','.join(excess)
            if reason:entries.append(dict(condition=name,status='skipped-HOLD',reason=reason));continue
            if name.startswith('grid-') and entrance['status']=='unavailable':
                entries.append(dict(condition=name,status='unavailable',reason=entrance['reason']));continue
            cell=slot if name in ('anchor','baseline') else query_cell(slot,entrance)
            counts['attempted_requests']+=1
            try:
                record=session.generate(cell);record.update(schema=SCHEMA,condition=name,image_id=cell['image_id'])
                if name!='baseline':
                    tensors=record.pop('_tensors');path=output/(name+'.safetensors');save_file(tensors,str(path))
                    record.update(score_tensors=binding(path),tensor_identities={k:visual.tensor_identity(v) for k,v in tensors.items()})
                a.write(output/(name+'.json'),record);counts['generated_actions']+=len(record['token_ids'])
                validate_record(record,cell,packet,session.q.tokenizer)
            except Exception as error:
                failure=name+'-failure.json';a.write(output/failure,dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
                entry=dict(condition=name,status='attempted-invalid',failure=failure)
                if (output/(name+'.json')).exists():entry['filename']=name+'.json'
                entries.append(entry);raise
            counts['completed_requests']+=1;entries.append(dict(condition=name,status='completed',filename=name+'.json'))
            if name=='anchor':
                control=grid.fidelity(record,cell,packet);anchor=control['qualified'];a.write(output/'control.json',control)
            if name=='baseline':
                entrance=first_header(record,session.q.tokenizer);a.write(output/'entrance.json',entrance)
            a.write(output/f'resource-{counts["attempted_requests"]:02d}.json',dict(seconds=time.monotonic()-begin,counts=counts,usage=resources(output,begin,native)[0]))
        a.write(output/'conditions.json',entries);tokenizer=session.q.tokenizer;session.close();session=None
        report=consume(output,packet,tokenizer);report['compute']='native' if native else 'CPU_FIXTURE'
        a.write(output/'readback.json',report);load_packet(config_path,cpu=not native)
        if report['complete'] and not resources(output,begin,native)[1]:code=0
    except Exception as error:
        a.write(output/'error.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        if session is not None and session.failure_evidence is not None:a.write(output/'partial-generation.json',session.failure_evidence)
        if packet is not None and not (output/'conditions.json').exists():
            done={e['condition'] for e in entries}
            entries.extend(dict(condition=c['condition'],status='skipped-HOLD',reason='technical_failure:'+type(error).__name__)
                           for c in packet['conditions'] if c['condition'] not in done)
            a.write(output/'conditions.json',entries)
    finally:
        if session is not None:
            try:session.close()
            except Exception as error:a.write(output/'cleanup-error.json',dict(error=str(error)));code=2
        usage,excess=resources(output,begin,native)
        a.write(output/'terminal.json',dict(exit_code=code,status='complete' if code==0 else 'technical_HOLD',counts=counts,
            seconds=time.monotonic()-begin,usage=usage,resource_excess=excess,
            readback=binding(output/'readback.json') if (output/'readback.json').exists() else None))
    return code


def visual_row(slot,packet,name):
    from src.vis.normalization import VisualObject,VisualRow
    image=packet['images']['25394'];w,h=image['width'],image['height']
    def obj(i,description,box,valid,metadata):
        pixel=tuple(round(v*extent/1000) for v,extent in zip(box,(w,h,w,h)))
        return VisualObject(i,description,description,pixel,tuple(box),'coord_bins',tuple(box),valid,metadata)
    gt=tuple(obj(i,o['desc'],o['bbox_2d'],True,dict(coco_ann_id=o['coco_ann_id'],full_label_order=i)) for i,o in enumerate(image['objects']))
    pred=() if slot is None else (obj(0,slot['description'],slot['box'],slot['valid'],dict(source=name)),)
    return VisualRow(name,0,Path(image['image_path']),image['image_path'],w,h,gt,pred,dict(matching_is_geometry_only=True))


def galleries(output,packet):
    from PIL import Image,ImageDraw
    from src.vis.matching import match_row
    from src.vis.rendering import prepare_render_view,render_gt_vs_prediction_png,render_view_manifest,PANEL_RIGHT_X
    output=Path(output);terminal=a.load(output/'terminal.json');report=a.load(output/'readback.json');verify_inputs(packet)
    if binding(output/'readback.json')!=terminal['readback']:raise ValueError('saved report changed before gallery')
    if [r['condition'] for r in report['conditions']]!=[c['condition'] for c in packet['conditions']]:raise ValueError('gallery slot identities differ')
    directory=output/'gallery';directory.mkdir(exist_ok=False);begin=time.monotonic();items=[];overviews=[]
    labels_path=directory/'full-labels.json';a.write(labels_path,dict(source=packet['bindings']['selection_source'],
        selected_line=248,image=packet['images']['25394'],original_source_row=packet['original_source_row']))
    with Image.open(packet['images']['25394']['image_path']) as im:im.convert('RGB').save(directory/'full-scene.png')
    baseline=report['baseline'];baseline_slots=[]
    if baseline:
        baseline_slots=[(f'baseline-{g["group_index"]:03d}',g,dict(group_index=g['group_index'],multiplicity=g['multiplicity'])) for g in baseline['groups']]
        baseline_slots += [(f'baseline-unavailable-{i:03d}',None,dict(malformed_index=i,status='malformed_or_censored',raw_entry=m)) for i,m in enumerate(baseline['malformed'])]
    for family,slots in [('candidates',[(r['condition'],(r['analysis'] or {}).get('selected_row'),dict(query_index=r['slot']['query_index'],
                x1=r['slot']['x1'],status=r['status'],outcome=(r['analysis'] or {}).get('outcome'))) for r in report['conditions'][2:]]),
                         ('baseline',baseline_slots)]:
        page=[];page_number=0
        def publish_page():
            nonlocal page_number
            if not page:return
            overview=Image.new('RGB',(1800,math.ceil(len(page)/4)*230),'white');draw=ImageDraw.Draw(overview)
            for i,path in enumerate(page):
                with Image.open(path) as im:
                    tile=im.convert('RGB');tile.thumbnail((450,205));overview.paste(tile,((i%4)*450,(i//4)*230+20))
                draw.text(((i%4)*450+4,(i//4)*230+2),path.stem,fill='black')
            path=directory/f'{family}-overview-{page_number:02d}.png';overview.save(path);overviews.append(binding(path));page_number+=1;page.clear()
        for name,selected,meta in slots:
            if time.monotonic()-begin>BOUNDS['gallery_seconds']:raise ValueError('gallery render-boundary wall excess')
            if readout.usage(output,native=False)['retained_bytes']>BOUNDS['retained_bytes']:raise ValueError('combined payload bound exceeded')
            row=visual_row(selected,packet,name);crop=grid.contextual_crop(row);path=directory/(name+'.png');view_manifest=None;matching=None
            if crop is None:
                png=Image.new('RGB',(900,830),'white');ImageDraw.Draw(png).text((20,35),name+' '+meta.get('status','unavailable')+'; no invented prediction. See full-scene.png',fill='black');png.save(path)
            else:
                match=match_row(row,match_iou_threshold=.5,duplicate_iou_threshold=.3)
                view=prepare_render_view(row,crop=crop,focus_pred_indices=[0],focus_gt_indices=[])
                title=f'{name} {selected["description"]} box={selected["box"]} valid={selected["valid"]} x1={meta.get("x1")} multiplicity={meta.get("multiplicity")}'
                # Keep the enlarged prediction/context panel, not a duplicate GT panel.
                fullpath=directory/(name+'.render.png');render_gt_vs_prediction_png(row=row,match=match,output_path=fullpath,title=title,view=view)
                with Image.open(fullpath) as im:png=im.crop((900,0,1800,830))
                caption=ImageDraw.Draw(png);caption.rectangle((0,0,900,40),fill='white')
                caption.text((12,6),title,fill='black');caption.text((12,23),'source image25394; native original geometry; see bound manifest',fill='black')
                png.save(path)
                fullpath.unlink();view_manifest=render_view_manifest(row,view,panel_x=PANEL_RIGHT_X-900);matching=match.to_manifest()
            manifest=dict(schema=SCHEMA,row_id=name,family=family,metadata=meta,
                full_label_pool=binding(labels_path),raw_sources=dict(readback=terminal['readback'],
                    baseline=report['conditions'][1]['artifact']),source_image=packet['bindings']['image25394'],
                frame=[864,1152],prediction=[o.to_manifest() for o in row.pred],render_view=view_manifest,
                match=matching,matching_claim='illustrative geometry only; no physical identity',image=binding(path))
            mpath=directory/(name+'.json');a.write(mpath,manifest);items.append(binding(mpath));page.append(path)
            if len(page)==32:publish_page()
        publish_page()
    usage=readout.usage(output,native=False)
    result=dict(schema=SCHEMA,readback=terminal['readback'],source=packet['producer'],full_scene=binding(directory/'full-scene.png'),
        full_label_pool=binding(labels_path),candidate_slots=32,baseline_groups=len(baseline['groups']) if baseline else 0,
        baseline_occurrences=baseline['complete_rows'] if baseline else 0,baseline_categories=baseline['categories'] if baseline else {},
        malformed_entries=len(baseline['malformed']) if baseline else 0,manifests=items,overviews=overviews,
        seconds=time.monotonic()-begin,usage=usage,physical_adjudication='pending_Root')
    a.write(directory/'index.json',result)
    if usage['retained_bytes']>BOUNDS['retained_bytes'] or result['seconds']>BOUNDS['gallery_seconds']:
        raise ValueError('gallery terminal resource excess; retained in-flight overshoot')
    return result


def main(argv=None,*,cpu_factory=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True)
    p=sub.add_parser('run');p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    for name in ('readback','gallery'):
        p=sub.add_parser(name);p.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':print(json.dumps(prepare(args.output),sort_keys=True));return 0
    if args.command=='run':return run(args.config,args.output,cpu_factory=cpu_factory)
    import torch
    from probes import rollout_row_credit as retained
    print('spatial_transfer_cpu_threads_incoming='+str(torch.get_num_threads()),file=sys.stderr)
    output=Path(args.output);inv=a.load(output/'invocation.json')
    if binding(inv['config']['path'])!=inv['config']:raise ValueError('saved config changed')
    _,packet=load_packet(inv['config']['path'],cpu=inv['compute']=='CPU_FIXTURE')
    if args.command=='gallery':print(json.dumps(galleries(output,packet),sort_keys=True));return 0
    q=retained.frontend()
    if q.model is not None:raise ValueError('saved consumer loaded a model')
    print(json.dumps(consume(output,packet,q.tokenizer),sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
