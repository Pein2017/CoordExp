"""Frozen independent spatial queries and exhaustive saved-data presentation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from probes import box_continuity as continuity, coordinate_readout as readout
from probes.coordinate_diagnostics import cued_visual as reused, visual_state as visual
from probes.rule_stability import artifacts as a

ROOT = a.ROOT
OUTPUT = ROOT / 'outputs/research/physical-fn-recovery/2026-10-04/spatial-candidate-grid'
UNIT = ROOT / 'research/experiments/2026-10-04-spatial-candidate-grid'
PRIOR = ROOT / 'outputs/research/physical-fn-recovery/2026-10-04/cue-region-specificity'
SCHEMA = 'spatial-candidate-grid-v1'
SOURCE_PATHS = ['probes/coordinate_diagnostics/spatial_grid.py',
                'tests/probes/coordinate_diagnostics/test_spatial_grid.py']
GRID = [((2*j+1)*1000)//64 for j in range(32)]
BOUNDS = dict(reused.BOUNDS, requests=33, actions=627, maximum_actions_per_request=19,
              maximum_context_tokens=1381, wall_seconds=300, retained_bytes=256*1024**2)
COORD_START, COORD_IDS = readout.COORD_START, readout.COORD_IDS
binding, revision, cpu_diagnostics = readout.binding, readout.revision, reused.cpu_diagnostics


def producer_identity():
    diff = subprocess.check_output(['git', 'diff', 'HEAD', '--', *SOURCE_PATHS], cwd=ROOT)
    return dict(commit=revision(), diff_sha256=hashlib.sha256(diff).hexdigest(),
                files={p: a.digest(ROOT/p) for p in SOURCE_PATHS})


def definitions(anchor):
    common = dict(image_id=351017, input_region='clean', position=14, row_start=9,
        budget=19, observations=[14,15,16,17], expected_ids=anchor['token_ids'][:14],
        forced_actions={str(i):t for i,t in enumerate(anchor['token_ids'][:14])})
    cells = [dict(common, condition='anchor', query_index=None, x1=None,
        expected_ids=anchor['token_ids'], reference_median_winners=[s['argmax'] for s in anchor['median_steps']],
        reference_raw_logprobs=anchor['raw_logprobs'],
        reference_median_logprobs=[s['pre_force_logprob'] for s in anchor['median_steps']])]
    for j,x in enumerate(GRID):
        cells.append(dict(common, condition=f'grid-{j:02d}', query_index=j, x1=x,
                          forced_actions=dict(common['forced_actions'], **{'14':COORD_START+x})))
    return cells


def baseline(packet):
    from src.eval.saved_rows import iou_xyxy
    raw = a.load(packet['bindings']['raw-351017']['path'])
    analysis = a.load(packet['bindings']['analysis-351017']['path'])
    rows = analysis['rows']; bottles = [r for r in rows if r['description']=='bottle']
    groups = {}
    for row in bottles:
        box = row['bbox']; positions = row['coordinate_positions']
        if (not row['valid'] or box[:2] != [0,0] or len(positions)!=4 or
                [raw['token_ids'][i]-COORD_START for i in positions] != box or
                row['raw_text'] != raw['text'][row['char_start']:row['char_end']]):
            raise ValueError('saved baseline row/token/source association differs')
        key = (row['description'], *box)
        if key not in groups:
            groups[key] = dict(group_index=len(groups), description=row['description'], box=box,
                valid=row['valid'], occurrences=[])
        groups[key]['occurrences'].append(dict(order=row['order'], positions=row['positions'],
            coordinate_positions=positions, completion_position=row['completion_position'],
            raw_span_sha256=row['raw_span_sha256']))
    tail = analysis['malformed']
    if (len(raw['token_ids'])!=3084 or len(rows)!=308 or len(bottles)!=307 or len(groups)!=14 or
            len(tail)!=1 or tail[0]['generated_order']!=308 or tail[0]['censored'] is not True or
            tail[0]['raw_text']!='<|object_ref_start|>bottle<|object_ref_end|><|box_start|>' or
            max(r['bbox'][2] for r in bottles)!=53 or max(r['bbox'][3] for r in bottles)!=86):
        raise ValueError('frozen full baseline counts/censored tail differ')
    for group in groups.values():
        group['same_category_overlaps']=[dict(annotation_id=o['coco_ann_id'],iou=float(iou_xyxy(group['box'],o['bbox_2d'])))
            for o in packet['images']['351017']['objects'] if o['desc']=='bottle']
    tail_positions=list(range(rows[-1]['completion_position']+1,len(raw['token_ids'])))
    return dict(raw=packet['bindings']['raw-351017'], analysis=packet['bindings']['analysis-351017'],
        tokens=3084, complete_rows=308, bottle_occurrences=307, unique_bottle_geometries=14,
        groups=[dict(g,multiplicity=len(g['occurrences'])) for g in groups.values()],
        unavailable_tail=dict(tail[0], status='unavailable_censored', box=None,positions=tail_positions,
            token_ids=[raw['token_ids'][i] for i in tail_positions]),
        freshness='historical_full_trajectory; fresh_anchor_qualifies_only_19_actions')


def verify_inputs(packet):
    import numpy as np
    from PIL import Image
    from src.qwen.images import rgb_image_sha256
    if (packet['schema']!=SCHEMA or packet['bounds']!=BOUNDS or packet['model_loaded'] is not False or
            packet['coordinate_ids']!=COORD_IDS or
            packet['summary_convention']!='detached-original-FP32-vector_CPU-single-thread-FP32'):
        raise ValueError('frozen matrix/observation contract differs')
    for item in packet['bindings'].values():
        if a.digest(item['path'])!=item['sha256']:raise ValueError('bound input changed:'+item['path'])
    old = a.load(packet['bindings']['prior_packet']['path'])
    anchor = a.load(packet['bindings']['anchor']['path'])
    raw = a.load(packet['bindings']['raw-351017']['path'])
    if packet['conditions']!=definitions(anchor) or anchor['token_ids']!=raw['token_ids'][:19]:
        raise ValueError('frozen grid/native history/action role differs')
    if (packet['images']!={'351017':old['images']['351017']} or
            packet['original_media']!={'351017':old['original_media']['351017']} or
            packet['requests']!={'351017':{'clean':old['requests']['351017']['clean']}} or
            packet['processor']!={'351017':{'clean':old['processor']['351017']['clean']}} or
            packet['pixels']!={k:old['pixels']['clean'][k] for k in ('image','array','pixel_sha256')}):
        raise ValueError('frozen clean image/request/prompt identity differs')
    image = packet['images']['351017']
    labels = next(x for x in a.load(packet['bindings']['labels']['path']) if x['image_id']==351017)
    if image!=labels or (image['width'],image['height'])!=(1248,832):
        raise ValueError('complete ordered annotation/frame identity differs')
    for item in (packet['pixels']['image'],packet['pixels']['array']):
        if a.digest(item['path'])!=item['sha256']:raise ValueError('clean pixels changed')
    with Image.open(packet['pixels']['image']['path']) as im:
        rgb = np.array(im.convert('RGB'))
        if (rgb_image_sha256(im)!=packet['pixels']['pixel_sha256'] or
                not np.array_equal(rgb,np.load(packet['pixels']['array']['path'],allow_pickle=False))):
            raise ValueError('lossless clean RGB identity differs')
    identity = packet['processor']['351017']['clean']
    if len(identity['prompt_token_ids'])!=1362 or identity['image_grid_thw']!=[1,52,78]:
        raise ValueError('prompt/context/grid bound differs')
    baseline(packet)


def prepare(directory):
    from probes import iterative_positive as p, rollout_row_credit as retained
    from probes.rule_stability.__main__ import runtime_identity
    oldpath = PRIOR/'prepared-01/input-packet.json'; old = a.load(oldpath)
    if old['runtime']!=runtime_identity():raise ValueError('qualified runtime changed')
    for path,sha in old['cached_pipeline_files'].items():
        if a.digest(ROOT/path)!=sha:raise ValueError('qualified helper changed:'+path)
    continuity.check_payloads(old['payloads'])
    q = retained.frontend()
    if q.model is not None:raise ValueError('CPU preparation loaded a model')
    request = old['requests']['351017']['clean']
    actual = visual.batch_identity(p.native_request(request,p.load(p.POLICY),q.processor))
    if actual!=old['processor']['351017']['clean']:raise ValueError('real frontend encoding differs')
    anchorpath = reused.OUTPUT/'native-01/bottle-uncued-clean.json'; anchor = a.load(anchorpath)
    bindings = {k:old['bindings'][k] for k in ('raw-351017','analysis-351017','labels','manifest',
        'image-351017','checkpoint','policy','prior_qualification')}
    bindings.update(protocol=binding(UNIT/'unit.md'),prior_packet=binding(oldpath),anchor=binding(anchorpath),
                    anchor_scores=anchor['score_tensors'])
    pipeline = dict(old['cached_pipeline_files'])
    for path in ('probes/iterative_positive.py','probes/rollout_row_credit.py','probes/rule_stability/artifacts.py',
        'probes/rule_stability/objectives.py','probes/rule_stability/data.py','probes/rule_stability/__main__.py',
        'src/eval/saved_rows.py','src/eval/detection_categories.py','src/vis/normalization.py',
        'src/vis/matching.py','src/vis/rendering.py','src/data/geometry.py','src/qwen/runtime_loading.py'):
        pipeline[path]=a.digest(ROOT/path)
    packet = dict(schema=SCHEMA,bounds=BOUNDS,bindings=bindings,conditions=definitions(anchor),
        producer=producer_identity(),cached_pipeline_files=pipeline,runtime=old['runtime'],payloads=old['payloads'],
        model_loaded=False,pixels={k:old['pixels']['clean'][k] for k in ('image','array','pixel_sha256')},
        requests={'351017':{'clean':request}},processor={'351017':{'clean':actual}},
        images={'351017':old['images']['351017']},original_media={'351017':old['original_media']['351017']},
        **{k:old[k] for k in ('coordinate_ids','eos_id','object_start_id','vocabulary_size','summary_convention')})
    verify_inputs(packet)
    directory=Path(directory).resolve()
    if not directory.is_relative_to(OUTPUT):raise ValueError('preparation outside task owner')
    directory.mkdir(parents=True,exist_ok=False)
    a.write(directory/'input-packet.json',packet)
    proposal=dict(schema=SCHEMA,released=False,source_revision=revision(),producer_files=packet['producer']['files'],
        input_packet=binding(directory/'input-packet.json'),runtime=packet['runtime'],bounds=BOUNDS,
        output=str(OUTPUT/'native-01'),retry='no_automatic_relaunch')
    a.write(directory/'native-proposal.json',proposal)
    return proposal


def load_packet(path, *, cpu=False):
    from probes.rule_stability.__main__ import runtime_identity
    config=a.load(path)
    if config['schema']!=SCHEMA or config['bounds']!=BOUNDS or config['retry']!='no_automatic_relaunch':
        raise ValueError('release/resource contract differs')
    if a.digest(config['input_packet']['path'])!=config['input_packet']['sha256']:raise ValueError('packet changed')
    packet=a.load(config['input_packet']['path']);verify_inputs(packet)
    if config['runtime']!=runtime_identity() or config['runtime']!=packet['runtime']:raise ValueError('runtime changed')
    if config['producer_files']!=producer_identity()['files'] or config['producer_files']!=packet['producer']['files']:
        raise ValueError('producer changed')
    for file,sha in packet['cached_pipeline_files'].items():
        if a.digest(ROOT/file)!=sha:raise ValueError('helper changed:'+file)
    continuity.check_payloads(packet['payloads'])
    if not cpu:
        if config['released'] is not True or config['source_revision']!=revision():raise ValueError('exact lead release required')
        if subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=ROOT):
            raise ValueError('native source dirty')
        subprocess.run(['git','ls-files','--error-unmatch',*SOURCE_PATHS],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('single GPU0 required')
    return config,packet


@cpu_diagnostics()
def canonical(vector,packet):
    import torch
    if vector.device.type!='cpu' or vector.dtype!=torch.float32 or vector.shape!=(packet['vocabulary_size'],):
        raise ValueError('original detached CPU FP32 vector required')
    return readout.capture_scores(vector[None],packet)


class Observer(continuity.ContinuationProcessor):
    def __init__(self,*args,channel,**kwargs):
        super().__init__(*args,**kwargs);self.channel=channel;self.vectors={}

    def __call__(self,input_ids,scores):
        position=input_ids.shape[1]-self.width
        vector=scores[0].detach().float().cpu().clone() if position in self.cell['observations'] else None
        result=super().__call__(input_ids,scores)
        if vector is not None:
            old=self.observations[-1];key=f'{self.channel}_{position}';self.vectors[key]=vector
            self.observations[-1]=dict(position=position,prefix_token_ids=old['prefix_token_ids'],vector_key=key,
                native_device_full_log_normalizer=old['full_log_normalizer'],native_device=str(scores.device),
                **canonical(vector,self.packet))
        return result


class NativeSession(visual.NativeSession):
    def generate(self,cell):
        import torch
        from src.qwen.generation import generate_continuations, NativeGenerationPolicy
        batch=self.batches[('351017','clean')];width=len(batch.prompt_token_ids[0]);begin=time.monotonic()
        raw=Observer(cell,self.packet,width,channel='raw')
        median=Observer(cell,self.packet,width,raw=raw,channel='median');self.failure_evidence=None
        try:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                result=generate_continuations(self.q.model,batch,extensions=[()],budgets=[19],
                    eos_token_id=self.packet['eos_id'],pad_token_id=self.q.tokenizer.pad_token_id,
                    policy=NativeGenerationPolicy(temperature=0,top_p=1,top_k=0,repetition_penalty=1,use_model_defaults=False),
                    trace='raw_and_policy',allow_pad_tokens=True,
                    logits_processor=[raw,self.norm.generation_transform(),median])[0]
        except Exception:
            self.failure_evidence=dict(condition=cell['condition'],selected_unconfirmed=raw.emitted,
                raw_steps=raw.steps,median_steps=median.steps,raw_observations=raw.observations,
                median_observations=median.observations)
            raise
        image=self.packet['images']['351017']
        return dict(request_id=batch.request_ids[0],width=image['width'],height=image['height'],
            token_ids=list(result.token_ids),text=self.q.tokenizer.decode(result.token_ids,skip_special_tokens=False),
            stop_reason=result.stop_reason,raw_logprobs=list(result.raw_logprobs),policy_logprobs=list(result.policy_logprobs),
            raw_steps=raw.steps,median_steps=median.steps,raw_observations=raw.observations,
            median_observations=median.observations,generation_seconds=time.monotonic()-begin,
            _tensors=dict(raw.vectors,**median.vectors))


def validate_record(record,cell,packet):
    from safetensors.torch import load_file
    continuity.validate_record(record,cell,packet)
    image=packet['images']['351017'];request=packet['requests']['351017']['clean']
    if (record.get('schema')!=SCHEMA or record.get('condition')!=cell['condition'] or record.get('image_id')!=351017 or
            record.get('request_id')!=request['request_id'] or any(record.get(k)!=image[k] for k in ('width','height'))):
        raise ValueError('record/request/frame identity differs')
    if a.digest(record['score_tensors']['path'])!=record['score_tensors']['sha256']:raise ValueError('score vector bytes changed')
    vectors=load_file(record['score_tensors']['path'],device='cpu')
    if {k:visual.tensor_identity(v) for k,v in vectors.items()}!=record['tensor_identities']:
        raise ValueError('original score identities differ')
    keys=set()
    for channel in ('raw','median'):
        for obs in record[channel+'_observations']:
            key=f'{channel}_{obs["position"]}';keys.add(key)
            if obs['vector_key']!=key or key not in vectors:raise ValueError('score/action association differs')
            if any(obs.get(k)!=v for k,v in canonical(vectors[key],packet).items()):raise ValueError('canonical CPU summary differs')
            if 'target_metrics' in obs:raise ValueError('designated target metadata is outside this experiment')
    if keys!=set(vectors):raise ValueError('extra/missing score vectors')


@cpu_diagnostics()
def fidelity(record,cell,packet):
    mismatch=[i for i in range(19) if i>=len(record['token_ids']) or record['token_ids'][i]!=cell['expected_ids'][i] or
              record['median_steps'][i]['argmax']!=cell['reference_median_winners'][i]]
    anchor=a.load(packet['bindings']['anchor']['path'])
    differences={c:[dict(position=o['position'],**continuity.score_difference(old,o)) for o in record[c+'_observations']
        for old in anchor[c+'_observations'] if old['position']==o['position']] for c in ('raw','median')}
    return dict(qualified=not mismatch,mismatch_positions=mismatch,numeric_comparisons_descriptive=True,
        raw_logprob_max_abs_difference=max((abs(x-y) for x,y in zip(record['raw_logprobs'],cell['reference_raw_logprobs'])),default=None),
        median_logprob_max_abs_difference=max((abs(s['pre_force_logprob']-y) for s,y in zip(record['median_steps'],cell['reference_median_logprobs'])),default=None),
        saved_coordinate_scores=differences)


def analyze(record,cell,packet,tokenizer):
    from probes.rule_stability.objectives import trajectory_analysis
    from src.eval.saved_rows import iou_xyxy
    parsed=trajectory_analysis(record,tokenizer)
    row=next((r for r in parsed['rows'] if r['positions'][0]==9),None)
    selected=None
    if row is not None:
        box=row['bbox'];valid=row['valid']
        selected=dict(description=row['description'],box=box,valid=valid,complete=True,order=row['order'],
            positions=row['positions'],coordinate_positions=row['coordinate_positions'],completion_position=row['completion_position'],
            width=box[2]-box[0],height=box[3]-box[1],
            same_category_overlaps=[dict(annotation_id=o['coco_ann_id'],iou=float(iou_xyxy(box,o['bbox_2d'])))
                for o in packet['images']['351017']['objects'] if o['desc']=='bottle'] if valid else None,
            overlaps_unavailable_reason=None if valid else 'invalid_original_geometry',physical_recovery_credit=False)
    roles=reused.causal_roles(record,cell)
    if row is None: outcome='incomplete_or_structural_escape'
    elif row['description']!='bottle':outcome='category_or_structure_drift'
    else:outcome='complete_valid' if row['valid'] else 'complete_invalid_geometry'
    return dict(selected_row=selected,outcome=outcome,unavailable_reason=None if selected else 'no_complete_row_at_original_start_action_9',
        rows=parsed['rows'],malformed=parsed['malformed'],burdens=parsed['burdens'],causal_roles=roles,
        actual_forced_actions=sum(str(i) in cell['forced_actions'] for i in range(len(record['token_ids']))),
        actual_free_actions=sum(str(i) not in cell['forced_actions'] for i in range(len(record['token_ids']))),
        x1_annotation_coincidences=[o['coco_ann_id'] for o in packet['images']['351017']['objects']
            if cell['x1'] is not None and o['desc']=='bottle' and o['bbox_2d'][0]==cell['x1']],
        correspondence_and_entity_support='pending_Root_visual_adjudication')


@cpu_diagnostics()
def consume(output,packet,tokenizer):
    verify_inputs(packet);output=Path(output);index=a.load(output/'conditions.json');rows=[];anchor=False;files=set()
    if [e['condition'] for e in index]!=[c['condition'] for c in packet['conditions']]:raise ValueError('omitted/duplicated/reordered query slots')
    for entry,cell in zip(index,packet['conditions'],strict=True):
        name=cell['condition'];base=dict(condition=name,query_index=cell['query_index'],x1=cell['x1'])
        if entry['status']=='skipped-HOLD':
            if not ((name!='anchor' and not anchor and entry['reason']=='anchor') or
                    entry['reason'].startswith(('resource_limit:','technical_failure:'))):raise ValueError('unjustified HOLD')
            rows.append(dict(base,**{k:v for k,v in entry.items() if k!='condition'},analysis=None));continue
        if entry['status']=='attempted-invalid':
            failure=output/entry['failure']
            if not failure.is_file():raise ValueError('missing failure evidence')
            files.update(entry[k] for k in ('filename','failure') if k in entry)
            rows.append(dict(base,**{k:v for k,v in entry.items() if k!='condition'},analysis=None));continue
        if entry['status']!='completed' or entry['filename']!=name+'.json' or (name!='anchor' and not anchor):
            raise ValueError('published invalid cell/dependency')
        files.add(entry['filename']);record=a.load(output/entry['filename']);validate_record(record,cell,packet)
        if tokenizer.decode(record['token_ids'],skip_special_tokens=False)!=record['text']:raise ValueError('full decode corrupted')
        control=fidelity(record,cell,packet) if name=='anchor' else dict(qualified=True,quality_gate=False)
        if name=='anchor':anchor=control['qualified']
        analysis=analyze(record,cell,packet,tokenizer);roles=analysis['causal_roles']
        scores={ch:[dict(position=o['position'],role=roles.get(str(o['position'])),prefix_token_ids=o['prefix_token_ids'],
            **continuity.score_summary(o,record[ch+'_steps'][o['position']])) for o in record[ch+'_observations']] for ch in ('raw','median')}
        rows.append(dict(base,status='completed',control=control,analysis=analysis,scores=scores,record=record,
                         artifact=binding(output/entry['filename'])))
    actual={p.name for stem in ('anchor','grid-') for p in output.glob(stem+'*.json')}
    if files!=actual:raise ValueError('extra/unconsumed cell or partial artifact')
    actions=sum(len(r['record']['token_ids']) for r in rows if r['status']=='completed')
    return dict(schema=SCHEMA,complete=anchor and all(e['status']=='completed' for e in index),anchor_qualified=anchor,
        conditions=rows,baseline=baseline(packet),requested_cells=33,requested_candidates=32,
        requested_denominators=dict(anchors=1,grid_candidates=32),attempted_requests=sum(e['status']!='skipped-HOLD' for e in index),
        completed_requests=sum(e['status']=='completed' for e in index),generated_actions=actions,
        actual_forced_actions=sum(r['analysis']['actual_forced_actions'] for r in rows if r['analysis']),
        actual_free_actions=sum(r['analysis']['actual_free_actions'] for r in rows if r['analysis']),
        primary_new_physical_entities=None,visual_ledger_status='pending_Root',
        limitations='Supplied native bottle header/x1, full-label training image; query selection only is GT-coordinate-free. Historical full baseline; short fresh fidelity only. No natural category/physical recovery claim.')


def visual_row(slot,packet,*,baseline_group=False):
    from src.vis.normalization import VisualObject,VisualRow
    image=packet['images']['351017'];w,h=image['width'],image['height']
    def obj(index,description,box,valid,metadata):
        pixel=tuple(round(v*extent/1000) for v,extent in zip(box,(w,h,w,h)))
        return VisualObject(index,description,description,pixel,tuple(box),'coord_bins',tuple(box),valid,metadata)
    gt=tuple(obj(i,o['desc'],o['bbox_2d'],True,dict(coco_ann_id=o['coco_ann_id'],full_label_order=i))
        for i,o in enumerate(image['objects']))
    selected=slot if baseline_group else (slot.get('analysis') or {}).get('selected_row')
    pred=() if selected is None else (obj(0,selected['description'],selected['box'],selected['valid'],
        dict(slot_identity=slot.get('condition'),query_index=slot.get('query_index'),x1=slot.get('x1'),
             positions=selected.get('positions'),occurrences=slot.get('occurrences'))),)
    row_id=f'baseline-{slot["group_index"]:02d}' if baseline_group else slot['condition']
    return VisualRow(row_id,slot.get('group_index',slot.get('query_index',0)),Path(packet['pixels']['image']['path']),
        image['image_path'],w,h,gt,pred,dict(slot=slot,clean_image=packet['pixels']['image'],
        full_label_source=packet['bindings']['labels'],matching_is_geometry_only=True))


def contextual_crop(row):
    if not row.pred:return None
    box=row.pred[0].bbox_pixel_xyxy
    x0,x1=min(box[0],box[2]),max(box[0],box[2]);y0,y1=min(box[1],box[3]),max(box[1],box[3])
    # Viewing extent only: original directed endpoints stay unchanged in VisualObject.
    dx=max(80,(x1-x0)*.75);dy=max(80,(y1-y0)*.5)
    return (max(0,x0-dx),max(0,y0-dy),min(row.image_width,x1+dx),min(row.image_height,y1+dy))


def galleries(output,packet):
    from PIL import Image,ImageDraw
    from src.vis.matching import match_row
    from src.vis.rendering import prepare_render_view,render_gt_vs_prediction_png,render_view_manifest,PANEL_LEFT_X,PANEL_RIGHT_X
    output=Path(output);terminal=a.load(output/'terminal.json')
    if binding(output/'readback.json')!=terminal['readback']:raise ValueError('saved report bytes changed before gallery')
    report=a.load(output/'readback.json');verify_inputs(packet)
    if [r['condition'] for r in report['conditions']]!=[c['condition'] for c in packet['conditions']]:
        raise ValueError('gallery omitted/duplicated slot')
    if report['baseline']!=baseline(packet):raise ValueError('gallery baseline differs')
    directory=output/'gallery';directory.mkdir(exist_ok=False);begin=time.monotonic()
    with Image.open(packet['pixels']['image']['path']) as original:original.convert('RGB').save(directory/'full-scene.png')
    items=[]
    for group,slots in [('candidates',report['conditions'][1:]),('baseline',report['baseline']['groups'])]:
        paths=[]
        for slot in slots:
            row=visual_row(slot,packet,baseline_group=group=='baseline');crop=contextual_crop(row)
            name=row.row_id;path=directory/(name+'.png')
            if crop is None:
                png=Image.new('RGB',(900,400),'white');ImageDraw.Draw(png).text((20,20),name+' '+slot['status']+'; no complete box. See full-scene.png',fill='black');png.save(path)
                views=None;matching=None
            else:
                match=match_row(row,match_iou_threshold=.5,duplicate_iou_threshold=.3)
                view=prepare_render_view(row,crop=crop,focus_pred_indices=[0],focus_gt_indices=[])
                title=f'{name} x1={slot.get("x1")} box={list(row.pred[0].source_bbox)} valid={row.pred[0].geometry_valid}'
                if group=='baseline':title+=f' multiplicity={slot["multiplicity"]}'
                render_gt_vs_prediction_png(row=row,match=match,output_path=path,title=title,view=view)
                views=[render_view_manifest(row,view,panel_x=x) for x in (PANEL_LEFT_X,PANEL_RIGHT_X)]
                matching=match.to_manifest()
            manifest=dict(row_id=name,status=slot.get('status','historical_complete'),query_index=slot.get('query_index'),
                x1=slot.get('x1'),original_slot=slot,source_image=packet['pixels']['image'],frame=[row.image_width,row.image_height],
                gt=[o.to_manifest() for o in row.gt],pred=[o.to_manifest() for o in row.pred],
                render_views=views,match=matching,matching_claim='illustrative geometry only; no physical identity',
                unavailable_crop_reason='no_complete_original_box' if crop is None else None,image=binding(path))
            a.write(directory/(name+'.json'),manifest);items.append(binding(directory/(name+'.json')));paths.append(path)
            if readout.usage(output,native=False)['retained_bytes']>BOUNDS['retained_bytes']:raise ValueError('combined native/gallery retained bound exceeded')
        overview=Image.new('RGB',(1800,math.ceil(len(paths)/4)*230),'white');draw=ImageDraw.Draw(overview)
        for i,path in enumerate(paths):
            with Image.open(path) as im:
                tile=im.convert('RGB');tile.thumbnail((450,205));overview.paste(tile,((i%4)*450,(i//4)*230+20))
            draw.text(((i%4)*450+6,(i//4)*230+2),path.stem,fill='black')
        overview.save(directory/(group+'-overview.png'))
    result=dict(schema=SCHEMA,readback=binding(output/'readback.json'),source=packet['producer'],
        full_scene=binding(directory/'full-scene.png'),candidate_slots=32,baseline_groups=14,baseline_occurrences=307,
        unavailable_baseline_tail=report['baseline']['unavailable_tail'],manifests=items,
        overviews=[binding(directory/(g+'-overview.png')) for g in ('candidates','baseline')],
        seconds=time.monotonic()-begin,physical_adjudication='pending_Root',usage=readout.usage(output,native=False))
    a.write(directory/'index.json',result)
    if readout.usage(output,native=False)['retained_bytes']>BOUNDS['retained_bytes']:raise ValueError('combined retained bound exceeded')
    return result


def resources(output,begin,native):
    values=readout.usage(output,native=native)
    return values,[k for k,v in values.items() if v>BOUNDS[k]]+(['wall_seconds'] if time.monotonic()-begin>BOUNDS['wall_seconds'] else [])


def run(config_path,output,*,cpu_factory=None):
    import torch
    from safetensors.torch import save_file
    output=Path(output).resolve()
    if not output.is_relative_to(OUTPUT):raise ValueError('invocation outside task owner')
    output.mkdir(parents=True,exist_ok=False);begin=time.monotonic();session=None;packet=None;native=cpu_factory is None
    code=2;entries=[];anchor=False
    counts=dict(checkpoint_loads=0,fixture_sessions=0,attempted_requests=0,completed_requests=0,generated_actions=0,
                optimizer=0,backward=0,replay=0,training=0,warmup=0,exports=0)
    a.write(output/'invocation.json',dict(schema=SCHEMA,config=binding(config_path),pid=os.getpid(),pgid=os.getpgid(0),started=time.time(),
        cpu_threads_incoming=torch.get_num_threads(),compute='native' if native else 'CPU_FIXTURE'))
    try:
        config,packet=load_packet(config_path,cpu=not native)
        if native and str(output)!=config['output']:raise ValueError('output differs from release')
        a.write(output/'qualification.json',dict(source_revision=config['source_revision'],producer=config['producer_files'],
            runtime=config['runtime'],bindings=packet['bindings'],helpers=packet['cached_pipeline_files'],payloads=packet['payloads']))
        if resources(output,begin,native)[1]:raise ValueError('resource excess before load')
        start=time.monotonic();counts['checkpoint_loads' if native else 'fixture_sessions']=1
        session=(cpu_factory or NativeSession)(readout.PREVIOUS/'checkpoint-16',packet)
        a.write(output/'load-B16.json',dict(seconds=time.monotonic()-start,composition=session.composition))
        for cell in packet['conditions']:
            name=cell['condition'];excess=resources(output,begin,native)[1]
            reason='anchor' if name!='anchor' and not anchor else ''
            if excess:reason='resource_limit:'+','.join(excess)
            if reason:entries.append(dict(condition=name,status='skipped-HOLD',reason=reason));continue
            counts['attempted_requests']+=1
            try:
                record=session.generate(cell);record.update(schema=SCHEMA,condition=name,image_id=351017)
                tensors=record.pop('_tensors');path=output/(name+'.safetensors');save_file(tensors,str(path))
                record.update(score_tensors=binding(path),tensor_identities={k:visual.tensor_identity(v) for k,v in tensors.items()})
                a.write(output/(name+'.json'),record);counts['generated_actions']+=len(record['token_ids'])
                validate_record(record,cell,packet)
            except Exception as error:
                failure=name+'-failure.json';a.write(output/failure,dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
                entry=dict(condition=name,status='attempted-invalid',failure=failure)
                if (output/(name+'.json')).exists():entry['filename']=name+'.json'
                entries.append(entry);raise
            counts['completed_requests']+=1
            if name=='anchor':
                check=fidelity(record,cell,packet);anchor=check['qualified'];a.write(output/'control.json',check)
            entries.append(dict(condition=name,status='completed',filename=name+'.json'))
            a.write(output/f'resource-{counts["attempted_requests"]:02d}.json',dict(seconds=time.monotonic()-begin,counts=counts,usage=resources(output,begin,native)[0]))
        a.write(output/'conditions.json',entries);tokenizer=session.q.tokenizer;session.close();session=None
        report=consume(output,packet,tokenizer);report['compute']='native' if native else 'CPU_FIXTURE'
        a.write(output/'readback.json',report);load_packet(config_path,cpu=not native)
        if report['complete'] and not resources(output,begin,native)[1]:code=0
    except Exception as error:
        a.write(output/'error.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        if session is not None and getattr(session,'failure_evidence',None) is not None:a.write(output/'partial-generation.json',session.failure_evidence)
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
    print('spatial_grid_cpu_threads_incoming='+str(torch.get_num_threads()),file=sys.stderr)
    output=Path(args.output);inv=a.load(output/'invocation.json')
    if binding(inv['config']['path'])!=inv['config']:raise ValueError('saved config changed')
    _,packet=load_packet(inv['config']['path'],cpu=inv['compute']=='CPU_FIXTURE')
    if args.command=='gallery':print(json.dumps(galleries(output,packet),sort_keys=True));return 0
    q=retained.frontend()
    if q.model is not None:raise ValueError('saved consumer loaded a model')
    print(json.dumps(consume(output,packet,q.tokenizer),sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
