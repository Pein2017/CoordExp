"""Bounded hidden-annotation acquisition; prepare/plan/review are CPU-only.

Only prepare and evaluate read truth. Acquisition consumes an allowlisted visible
view and frozen plan. Annotation matching is a proxy, never physical admission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import time

HUMAN = Path('/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/inputs/human13/human-refined-13.geo_sorted_xy.coord.jsonl')
REFINED = Path('/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/inputs/refined5')
CHECKPOINT = Path('/data/CoordExp/outputs/shared/checkpoints/untied-axis001-step2444/payload')
BASE = Path('/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent')
IMAGE_ROOT = Path('/data/CoordExp/public_data/coco/rescale_32_1024_bbox')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write(path, value):
    with Path(path).open('x') as f:
        f.write(canonical(value) + '\n')


def normalized(obj):
    box = [int(v.removeprefix('<|coord_').removesuffix('|>')) if isinstance(v, str) else v for v in obj['bbox_2d']]
    if len(box) != 4 or any(type(v) is not int or not 0 <= v <= 999 for v in box) or not (box[0] < box[2] and box[1] < box[3]):
        raise ValueError('invalid reference geometry')
    return {**obj, 'bbox_2d': box}


def split_views(records):
    """Project only current visible objects; no truth metadata survives projection."""
    visible, truth = [], []
    for r in records:
        public = {k: r[k] for k in ('image_id', 'image_path', 'image_sha256', 'width', 'height')}
        public['objects'] = [o for o in r['objects'] if o['coco_ann_id'] >= 0]
        visible.append(public)
        truth.append({**r, 'hidden_objects': [o for o in r['objects'] if o['coco_ann_id'] < 0]})
    return visible, truth


def prepare(output):
    from PIL import Image
    output.mkdir(parents=True, exist_ok=False)
    (output / 'acquisition').mkdir()
    (output / 'evaluator').mkdir(mode=0o700)
    if digest(HUMAN) != '5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23':
        raise ValueError('Human13 identity changed')
    human_receipt_path = HUMAN.with_name('human-refined-13.geo_sorted_xy.coord.receipt.json')
    human_receipt = load(human_receipt_path)
    if human_receipt['derived_sha256'] != digest(HUMAN):
        raise ValueError('Human13 receipt mismatch')
    human_images = {r['image_id']: r['images'][0] for r in human_receipt['images_manifest']}
    receipt = load(REFINED / 'snapshot-receipt.json')
    for name, spec in receipt['files'].items():
        if digest(REFINED / name) != spec['sha256']:
            raise ValueError(f'refined snapshot identity changed: {name}')
    if digest(REFINED / 'manifest.json') != receipt['manifest']['sha256']:
        raise ValueError('refined manifest identity changed')
    lineage = rows(REFINED / 'object_provenance.jsonl')
    crosswalk = {(r['image']['image_id'], r['annotation']['coco_ann_id']): r['annotation'] for r in lineage}
    source = {r['image_id']: r for r in rows(REFINED / 'source.norm.jsonl')}
    bindings = load(REFINED / 'manifest.json')['image_bindings']
    records, provenance, summaries = [], [], []
    for cohort, bank in [('human13', rows(HUMAN)), ('refined5', rows(REFINED / 'working.norm.jsonl'))]:
        for r in bank:
            image = IMAGE_ROOT / r['file_name']
            if cohort == 'refined5' and (str(image) != bindings[str(r['image_id'])]['resolved_path'] or digest(image) != bindings[str(r['image_id'])]['sha256']):
                raise ValueError('refined image binding changed')
            with Image.open(image) as im:
                if im.size != (r['width'], r['height']):
                    raise ValueError('image dimensions disagree')
            if cohort == 'human13' and (str(image) != human_images[r['image_id']]['resolved_path'] or digest(image) != human_images[r['image_id']]['sha256']):
                raise ValueError('Human13 image binding changed')
            objects = [normalized(o) for o in r['objects']]
            if len({o['coco_ann_id'] for o in objects}) != len(objects):
                raise ValueError('duplicate annotation ID')
            records.append(dict(image_id=r['image_id'], cohort=cohort, image_path=str(image), image_sha256=digest(image), width=r['width'], height=r['height'], objects=objects))
            counts = dict(image_id=r['image_id'], cohort=cohort, total=len(objects), added=0, retained=0, edited=0 if cohort=='refined5' else None, removed=0 if cohort=='refined5' else None)
            for o in objects:
                ann = o['coco_ann_id']
                if cohort == 'refined5':
                    entry = crosswalk[(r['image_id'], ann)]
                    if entry['working_object'] != o:
                        raise ValueError('crosswalk mismatch')
                    status = entry['record_status']
                    old = entry['source_object']
                    counts['edited'] += int(old is not None and old != o)
                else:
                    status = 'human_added_negative_id' if ann < 0 else 'retained_id_possibly_human_refined'
                    old = None
                counts['added' if ann < 0 else 'retained'] += 1
                provenance.append(dict(image_id=r['image_id'], annotation_id=ann, cohort=cohort, status=status, current=o, source=old))
            if cohort == 'refined5':
                current_ids = {o['coco_ann_id'] for o in objects}
                for old in source[r['image_id']]['objects']:
                    if old['coco_ann_id'] not in current_ids:
                        if crosswalk[(r['image_id'], old['coco_ann_id'])]['record_status'] != 'removed_source_id':
                            raise ValueError('removed crosswalk mismatch')
                        counts['removed'] += 1
                        provenance.append(dict(image_id=r['image_id'], annotation_id=old['coco_ann_id'], cohort=cohort, status='removed_source_id', current=None, source=old))
            summaries.append(counts)
    if len(records) != 18 or len({r['image_id'] for r in records}) != 18:
        raise ValueError('cohort identity differs')
    visible, truth = split_views(records)
    write(output / 'acquisition/visible.json', visible)
    write(output / 'evaluator/truth.json', truth)
    write(output / 'evaluator/provenance.json', provenance)
    config_path = CHECKPOINT / 'training_state/resolved_config.json'
    config = load(config_path)['config']
    manifest_path = CHECKPOINT / 'inference_payload_manifest.json'
    manifest = load(manifest_path)
    payload, missing = {}, []
    for section in ('adapter', 'special_token_embedding_delta'):
        for spec in manifest[section]['files']:
            path = CHECKPOINT / manifest[section]['relative_root'] / spec['relative_path']
            if not path.is_file():
                missing.append(str(path))
                continue
            actual = digest(path)
            if actual != spec['sha256']:
                raise ValueError(f'payload changed: {path}')
            payload[str(path)] = actual
    metadata = load(CHECKPOINT / 'special_token_embeddings/special_token_embeddings.json')
    if metadata['tie_word_embeddings'] is not False:
        raise ValueError('requested untied anchor payload differs')
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json', 'chat_template.json']:
        if (BASE / name).is_file():
            payload[str(BASE / name)] = digest(BASE / name)
    for path in sorted(BASE.glob('*.safetensors')):
        payload[str(path)] = digest(path)
    # Missing documentation is reported; executable bytes must all exist.
    if any(not p.endswith('/README.md') for p in missing):
        raise ValueError('executable payload missing')
    policy = dict(base_model=str(BASE), checkpoint=str(CHECKPOINT), payload_sha256=payload,
                  dtype='fp32', attention='sdpa', patch_embed_linearization='enabled',
                  prompt=config['template']['prompt'], max_new_tokens=3084,
                  temperature=0.7, top_p=1.0, top_k=0, repetition_penalty=1.0,
                  use_model_defaults=False, seeds=[92601, 92602, 92603, 92604], verification='same_model_saved_query_agreement; retain_all; no_auto_admission',
                  policy_status='schedule_frozen_awaiting_lead_release', crop='four overlapping 5/8 views; snap outer boundaries to 32px; no resize')
    write(output / 'acquisition/policy.json', policy)
    write(output / 'evaluator/data_manifest.json', dict(status='candidate', images=summaries,
        sources={str(p): digest(p) for p in [HUMAN, human_receipt_path, REFINED / 'working.norm.jsonl', REFINED / 'source.norm.jsonl', REFINED / 'object_provenance.jsonl', manifest_path, config_path]},
        historical_manifest_missing=missing, physical_identity_gap='New region IDs do not resolve possible redraws of deleted source owners; no deleted-to-added physical crosswalk exists in snapshot.',
        files={str(p.relative_to(output)): digest(p) for p in output.rglob('*.json')}))


def validate_visible(visible):
    for r in visible:
        if set(r) != {'image_id', 'image_path', 'image_sha256', 'width', 'height', 'objects'}:
            raise ValueError('visible view has forbidden or missing fields')
        for o in r['objects']:
            if set(o) != {'coco_ann_id','bbox_2d','category_id','category_name','desc'} or type(o['coco_ann_id']) is not int or o['coco_ann_id'] < 0:
                raise ValueError('visible object has forbidden fields or hidden ID')
            normalized(o)


def request_plan(visible, policy):
    validate_visible(visible)
    result = []
    for r in visible:
        w, h = r['width'], r['height']
        if w % 32 or h % 32:
            raise ValueError('native dimensions must be multiples of 32')
        x, y = 32 * math.ceil(w * 5 / 8 / 32), 32 * math.ceil(h * 5 / 8 / 32)
        boxes = [(0, 0, x, y), (w-x, 0, w, y), (0, h-y, x, h), (w-x, h-y, w, h)]
        for arm, boxes_for_arm in [('greedy', [(0,0,w,h)]), ('full', [(0,0,w,h)]*4), ('region', boxes)]:
            for i, box in enumerate(boxes_for_arm):
                result.append(dict(request_id=f"{r['image_id']}:{arm}:{i}", image_id=r['image_id'], image_path=r['image_path'], image_sha256=r['image_sha256'], width=w, height=h, arm=arm, crop=list(box), seed=None if arm=='greedy' else policy['seeds'][i], temperature=0.0 if arm=='greedy' else policy['temperature']))
    return result


def native_request(item, policy, processor):
    from PIL import Image
    from src.qwen.native import NativeRequest, prepare_native_inputs
    if digest(item['image_path']) != item['image_sha256']:
        raise ValueError('image bytes changed')
    with Image.open(item['image_path']) as source:
        image = source.convert('RGB').crop(item['crop'])
    if item.get('view_scale', 1) == 2:
        resized = image.resize((image.width * 2, image.height * 2), Image.Resampling.BICUBIC)
        image.close()
        image = resized
    elif item.get('view_scale', 1) != 1:
        raise ValueError('unsupported local view scale')
    messages = [{'role':'system','content':policy['prompt']['system']}, {'role':'user','content':[{'type':'image'}, {'type':'text','text':policy['prompt']['user']}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        batch = prepare_native_inputs(processor, [NativeRequest(item['request_id'], chat, image)], record_media_identity=True)
    finally:
        image.close()
    return batch


def execute(visible_path, policy_path, output, image_ids, generate=False, local_raw=None):
    visible, policy = load(visible_path), load(policy_path)
    if generate and policy.get('policy_status') != 'lead_released':
        raise ValueError('GPU acquisition policy is not released')
    from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
    selected = [r for r in visible if not image_ids or r['image_id'] in image_ids]
    if image_ids and {r['image_id'] for r in selected} != set(image_ids):
        raise ValueError('unknown image selection')
    if local_raw is None:
        plan = request_plan(selected, policy)
    else:
        local = local_plan(visible, read_frozen(local_raw))
        plan = local['requests']
        if image_ids:
            chosen = {min((p for p in local['selected'] if p['image_id']==i), key=local_hash)['prediction_id'] for i in image_ids}
            plan = [r for r in plan if r['target_prediction_id'] in chosen]
    global_request_ids = [r['request_id'] for r in plan]
    rank, world = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
    if generate and world > 1:
        from src.inference.data_parallel import plan_data_parallel_shards
        shards = plan_data_parallel_shards(row_ids=global_request_ids, per_device_batch_size=1, visible_cuda_tokens=[str(i) for i in range(world)])
        if rank >= shards.active_ranks:
            return
        plan = [plan[i] for i in shards.ranks[rank].row_indices]
        output = output / shards.ranks[rank].shard_dir_name
    for p, expected in policy['payload_sha256'].items():
        if digest(p) != expected:
            raise ValueError(f'model bytes changed: {p}')
    output.mkdir(parents=True, exist_ok=False)
    source_identity = None
    if generate:
        from src.artifacts.git_identity import capture_source_identity
        source_paths = ['probes/hidden_human_recovery.py', *sorted(str(p) for p in Path('src').rglob('*.py'))]
        source_identity = capture_source_identity(source_paths)
    if generate:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('GPU execution requires CUDA')
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    start = time.monotonic()
    qwen = load_qwen_components_from_options(QwenLoadOptions(policy['base_model'], policy['dtype'], policy['attention'], policy['patch_embed_linearization'], load_model=generate))
    if generate:
        import torch
        from src.adapters.dora import attach_dora_adapter
        from src.qwen.untied_embeddings import load_inference_embedding_delta
        from src.qwen.generation import generate_continuations, NativeGenerationPolicy
        if not torch.cuda.is_available():
            raise RuntimeError('GPU execution requires CUDA')
        checkpoint = Path(policy['checkpoint'])
        adapter_receipt = attach_dora_adapter(qwen.model, adapter_path=checkpoint/'adapter', base_model_path=qwen.base_model_path)
        embedding_receipt = load_inference_embedding_delta(config=SimpleNamespace(embedding_delta=SimpleNamespace(path=str(checkpoint/'special_token_embeddings'), source_gate_root=None)), qwen=qwen)
        qwen.model.to('cuda').eval()
        write(output/'composition.json', dict(adapter=adapter_receipt, embedding=embedding_receipt))
    load_seconds = time.monotonic() - start
    results = []
    for item in plan:
        start = time.monotonic()
        batch = native_request(item, policy, qwen.processor)
        grid = batch.image_grids[0]
        record = {**item, 'image_grid_thw':grid, 'visual_tokens':math.prod(grid)//qwen.processor_identity.merge_size**2,
                  'actual_pixels':(item['crop'][2]-item['crop'][0])*(item['crop'][3]-item['crop'][1]),
                  'prompt_token_ids':list(batch.prompt_token_ids[0]), 'media_sha256':batch.media_sha256[0], 'prepare_seconds':time.monotonic()-start}
        if local_raw is not None:
            roi_w, roi_h = item['crop'][2]-item['crop'][0], item['crop'][3]-item['crop'][1]
            record.update(original_roi_pixels=roi_w*roi_h, resized_dimensions=[roi_w*item['view_scale'],roi_h*item['view_scale']],
                          actual_pixels=roi_w*roi_h*item['view_scale']**2,
                          processed_dimensions=[grid[1]*qwen.processor_identity.patch_size,grid[2]*qwen.processor_identity.patch_size],
                          processed_pixels=grid[1]*grid[2]*qwen.processor_identity.patch_size**2)
        if generate:
            torch.cuda.synchronize()
            start = time.monotonic()
            result = generate_continuations(qwen.model, batch, extensions=[()], budgets=[policy['max_new_tokens']], eos_token_id=qwen.tokenizer.convert_tokens_to_ids('<|im_end|>'), pad_token_id=qwen.tokenizer.pad_token_id,
                policy=NativeGenerationPolicy(temperature=item['temperature'], top_p=policy['top_p'], top_k=policy['top_k'], repetition_penalty=policy['repetition_penalty'], use_model_defaults=policy['use_model_defaults']), seed=item['seed'])[0]
            torch.cuda.synchronize()
            record.update(token_ids=list(result.token_ids), text=qwen.tokenizer.decode(result.token_ids, skip_special_tokens=False), stop_reason=result.stop_reason, generated_tokens=len(result.token_ids), model_calls=1, generation_seconds=time.monotonic()-start)
        write(output/f"{item['request_id'].replace(':','-')}.json", record)
        results.append(record)
    if generate:
        from src.artifacts.git_identity import verify_source_identity
        verify_source_identity(source_identity, required_paths=source_paths)
    write(output/'frozen.json', dict(local_raw=str(local_raw) if local_raw else None, status='generated' if generate else 'cpu_plan', source_identity=source_identity,
        visible_sha256=digest(visible_path), policy_sha256=digest(policy_path), load_seconds=load_seconds, global_request_ids=global_request_ids,
        requests={r['request_id']:digest(output/f"{r['request_id'].replace(':','-')}.json") for r in results}))


def read_frozen(root):
    if not (root/'frozen.json').is_file():
        parts = sorted(root.glob('rank-*/frozen.json'))
        if not parts:
            raise ValueError('no frozen output')
        manifests = [load(p) for p in parts]
        for key in ('global_request_ids', 'visible_sha256', 'policy_sha256', 'source_identity'):
            if any(m[key] != manifests[0][key] for m in manifests):
                raise ValueError('shard identities differ')
        result = [r for p in parts for r in read_frozen(p.parent)]
        ids = [r['request_id'] for r in result]
        if len(ids) != len(set(ids)) or set(ids) != set(manifests[0]['global_request_ids']):
            raise ValueError('incomplete or duplicate shards')
        by_id = {r['request_id']:r for r in result}
        return [by_id[i] for i in manifests[0]['global_request_ids']]
    frozen = load(root/'frozen.json')
    if frozen['status'] != 'generated':
        raise ValueError('scoring requires frozen generated outputs')
    result = []
    for request_id, expected in frozen['requests'].items():
        p = root/f"{request_id.replace(':','-')}.json"
        if digest(p) != expected:
            raise ValueError('frozen output changed')
        result.append(load(p))
    return result


def candidates(records):
    from src.inference.parsing import parse_compact_object_box_closed
    valid, invalid = [], []
    for r in records:
        x1,y1,x2,y2 = r['crop']
        parsed = parse_compact_object_box_closed(r['text'], row_id=r['request_id'], row_index=0, image_width=x2-x1, image_height=y2-y1)
        for o in parsed.predictions:
            b = o['coord_bins']
            # Match src.data.geometry norm1000, without its pixel rounding.
            mapped = [(x1*1000+b[0]*(x2-x1))/r['width'], (y1*1000+b[1]*(y2-y1))/r['height'],
                      (x1*1000+b[2]*(x2-x1))/r['width'], (y1*1000+b[3]*(y2-y1))/r['height']]
            valid.append(dict(prediction_id=f"{r['request_id']}:p{o['generated_order']}", generated_order=len(valid), image_id=r['image_id'], arm=r['arm'], description=o['description'], coord_bins_1000=mapped, crop_boundary=any(v in (0,999) for v in b) and r['arm']=='region'))
        invalid.extend(dict(image_id=r['image_id'], arm=r['arm'], request_id=r['request_id'], **o) for o in parsed.dropped_predictions)
    return valid, invalid


def admission_inputs(visible, records):
    validate_visible(visible)
    from src.eval.saved_rows import iou_xyxy
    valid, invalid = candidates(records)
    known = {r['image_id']:r['objects'] for r in visible}
    unique = {}
    for p in valid:
        if p['arm'] == 'greedy':
            continue
        key = (p['image_id'],p['arm'],p['description'],tuple(p['coord_bins_1000']))
        if key in unique:
            unique[key]['literal_repeat_ids'].append(p['prediction_id'])
        else:
            unique[key] = {**p, 'literal_repeat_ids':[], 'known_iou50_proxy':any(iou_xyxy(o['bbox_2d'],p['coord_bins_1000'])>=0.5 for o in known[p['image_id']])}
    # Scores are correlated query agreement, never physical-owner identity.
    for p in unique.values():
        support = {}
        for other in valid:
            if other['image_id'] != p['image_id'] or other['arm'] != p['arm'] or other['description'] != p['description']:
                continue
            query = other['prediction_id'].rsplit(':p', 1)[0]
            if query == p['prediction_id'].rsplit(':p', 1)[0]:
                continue
            support[query] = max(support.get(query,0.0), iou_xyxy(other['coord_bins_1000'],p['coord_bins_1000']))
        p['verification_scores'] = dict(other_query_max_iou=support, other_query_iou50_support=sum(v>=0.5 for v in support.values()))
        p['verification_reason'] = 'same-model repeated-query geometry/category agreement; correlated screening only'
    return dict(candidates=valid, unique_candidates=list(unique.values()), invalid=invalid,
                admission='No automatic positive admission; retain all scores and reasons')



def evaluate(truth, records, reviews):
    from src.eval.saved_rows import one_to_one_matches, pairwise_iou95
    valid, invalid = candidates(records)
    review = {r['prediction_id']:r for r in reviews}
    if len(review) != len(reviews) or not set(review).issubset({p['prediction_id'] for p in valid}):
        raise ValueError('invalid review IDs')
    allowed = {'accepted','false_object','wrong_class','wrong_extent','duplicate','group','part','uncertain','out_of_scope'}
    if any(r['status'] not in allowed or not isinstance(r.get('seconds'), (int,float)) or not math.isfinite(r['seconds']) or r['seconds']<0 for r in reviews):
        raise ValueError('review needs explicit disposition and finite time')
    reports = []
    executed = {p['image_id'] for p in records}
    for r in truth:
        if r['image_id'] not in executed:
            continue
        refs = [dict(owner_id=str(o['coco_ann_id']), reference_coord_bins_1000=o['bbox_2d']) for o in r['objects']]
        hidden = {str(o['coco_ann_id']) for o in r['hidden_objects']}
        baseline = [p for p in valid if p['image_id']==r['image_id'] and p['arm']=='greedy']
        base_ids = {m['reference_owner_id'] for m in one_to_one_matches(refs,baseline,0.5)}
        for arm in ('greedy','full','region'):
            pool = [p for p in valid if p['image_id']==r['image_id'] and p['arm']==arm]
            for stage in ('raw','admitted'):
                chosen = pool if stage=='raw' else [p for p in pool if review.get(p['prediction_id'],{}).get('status')=='accepted']
                matches = one_to_one_matches(refs,chosen,0.5)
                chosen_by_id = {p['prediction_id']:p for p in chosen}
                for m in matches:
                    m['category_agrees'] = r['objects'][m['reference_index']]['desc'] == chosen_by_id[m['prediction_id']]['description']
                ids = {m['reference_owner_id'] for m in matches}
                matched_predictions = {m['prediction_id'] for m in matches}
                reports.append(dict(image_id=r['image_id'], cohort=r['cohort'], arm=arm, stage=stage, hidden_denominator=len(hidden), hidden_recovered=sorted(ids&hidden), new_hidden_beyond_greedy=sorted((ids&hidden)-base_ids), known_recovered=sorted(ids-hidden), matches=matches,
                    unmatched=[p['prediction_id'] for p in chosen if p['prediction_id'] not in matched_predictions], iou95_pairs=pairwise_iou95(chosen), crop_boundary_candidates=[p['prediction_id'] for p in chosen if p['crop_boundary']], invalid=[p for p in invalid if p['image_id']==r['image_id'] and p['arm']==arm],
                    unreviewed=[p['prediction_id'] for p in pool if p['prediction_id'] not in review], review_dispositions=[review[p['prediction_id']] for p in pool if p['prediction_id'] in review]))
    curves = []
    for k in range(1,5):
        for r in truth:
            if r['image_id'] not in executed:
                continue
            refs = [dict(owner_id=str(o['coco_ann_id']),reference_coord_bins_1000=o['bbox_2d']) for o in r['objects']]
            hidden = {str(o['coco_ann_id']) for o in r['hidden_objects']}
            baseline = [p for p in valid if p['image_id']==r['image_id'] and p['arm']=='greedy']
            baseline_ids = {m['reference_owner_id'] for m in one_to_one_matches(refs,baseline,0.5)}
            for arm in ('full','region'):
                prefix = [x for x in records if x['image_id']==r['image_id'] and x['arm']==arm and int(x['request_id'].split(':')[-1]) < k]
                if len(prefix) != k:
                    continue
                visible = [{key:r[key] for key in ('image_id','image_path','image_sha256','width','height')} | {'objects':[o for o in r['objects'] if o['coco_ann_id']>=0]}]
                screening = admission_inputs(visible,prefix)
                for support in range(k):
                    chosen = [p for p in screening['unique_candidates'] if p['verification_scores']['other_query_iou50_support']>=support]
                    matches = one_to_one_matches(refs,chosen,0.5)
                    ids = {m['reference_owner_id'] for m in matches}
                    by_id = {p['prediction_id']:p for p in chosen}
                    wrong_class = sum(r['objects'][m['reference_index']]['desc'] != by_id[m['prediction_id']]['description'] for m in matches)
                    curves.append(dict(image_id=r['image_id'],cohort=r['cohort'],arm=arm,k=k,min_other_query_support=support,screened_candidates=len(chosen),hidden_recovered=len(ids&hidden),new_hidden_beyond_greedy=len((ids&hidden)-baseline_ids),annotation_unmatched=len(chosen)-len(matches),matched_category_disagreements=wrong_class,invalid=len(screening['invalid']),visual_tokens=sum(x['visual_tokens'] for x in prefix),generated_tokens=sum(x['generated_tokens'] for x in prefix),generation_seconds=sum(x['generation_seconds'] for x in prefix)))
    return dict(status='annotation_proxy_only', curves=curves, unexecuted_image_ids=[r['image_id'] for r in truth if r['image_id'] not in executed], reports=reports, verification_seconds=sum(r['seconds'] for r in reviews), costs=[{k:r[k] for k in ('request_id','actual_pixels','visual_tokens','generated_tokens','model_calls','generation_seconds','prepare_seconds','stop_reason')} for r in records])


def local_hash(candidate):
    return hashlib.sha256(('926-local-v1:' + candidate['prediction_id']).encode()).hexdigest()


def local_plan(visible, records):
    """Prediction/visible-only diagnostic strata; never consume evaluation truth."""
    bank = admission_inputs(visible, records)['unique_candidates']
    images = {r['image_id']: r for r in visible}
    cells = {}
    for p in bank:
        cell = (p['image_id'], p['arm'], p['known_iou50_proxy'],
                p['verification_scores']['other_query_iou50_support'] > 0, p['crop_boundary'])
        if cell not in cells or local_hash(p) < local_hash(cells[cell]):
            cells[cell] = p
    selected = sorted(cells.values(), key=lambda p:(p['image_id'],local_hash(p)))
    requests = []
    for p in selected:
        r = images[p['image_id']]
        w,h = r['width'],r['height']
        b = p['coord_bins_1000']
        pixel = [b[0]*w/1000,b[1]*h/1000,b[2]*w/1000,b[3]*h/1000]
        cx,cy = (pixel[0]+pixel[2])/2,(pixel[1]+pixel[3])/2
        sx,sy = max(192,2*(pixel[2]-pixel[0])),max(192,2*(pixel[3]-pixel[1]))
        roi = [max(0,32*math.floor((cx-sx/2)/32)),max(0,32*math.floor((cy-sy/2)/32)),
               min(w,32*math.ceil((cx+sx/2)/32)),min(h,32*math.ceil((cy+sy/2)/32))]
        for scale in (1,2):
            requests.append({**{k:r[k] for k in ('image_id','image_path','image_sha256','width','height')},
                'request_id':f"{p['prediction_id']}:local:{scale}", 'arm':'local',
                'target_prediction_id':p['prediction_id'], 'view_scale':scale,
                'crop':roi, 'seed':None, 'temperature':0.0})
    if len(selected)>12*len(visible):
        raise ValueError('local cell bound exceeded')
    return dict(bank=bank,selected=selected,requests=requests)


def aligned_invalid(record, drop, encoded, tokenizer):
    """Reject text/token/row drift before exposing an own-prefix illegal slot."""
    import re
    text, ids = record['text'],record['token_ids']
    if list(encoded['input_ids']) != ids or tokenizer.decode(ids,skip_special_tokens=False) != text:
        raise ValueError('token alignment differs from saved text')
    a,z = drop['char_start'],drop['char_end']
    starts = [m.start() for m in re.finditer(re.escape('<|object_ref_start|>'),text)]
    order = drop['generated_order']
    if drop['row_id'] != record['request_id'] or order is None or order>=len(starts) or starts[order]!=a or text[a:z]!=drop['raw_text']:
        raise ValueError('row alignment differs from parser span')
    spans = drop['coord_token_spans']
    if len(spans)!=4 or not drop['raw_text'].endswith('<|box_end|>'):
        raise ValueError('complete invalid evidence requires four coordinates and box end')
    positions, bins = [],[]
    offsets = [tuple(x) for x in encoded['offset_mapping']]
    for sp in spans:
        matches = [i for i,off in enumerate(offsets) if off==(sp['char_start'],sp['char_end'])]
        if len(matches)!=1 or ids[matches[0]]!=tokenizer.convert_tokens_to_ids(sp['text']):
            raise ValueError('coordinate token alignment differs')
        positions.append(matches[0]); bins.append(int(sp['text'][8:-2]))
    if positions != list(range(positions[0],positions[0]+4)) or ids[positions[-1]+1]!=tokenizer.convert_tokens_to_ids('<|box_end|>'):
        raise ValueError('coordinate row token alignment differs')
    illegal = [j for j in (2,3) if bins[j]<=bins[j-2]]
    if not illegal:
        raise ValueError('valid boundary contact is not invalid geometry')
    dead = [j for j in (0,1) if bins[j]==999]
    return dict(coordinate_bins=bins,coordinate_token_positions=positions,
                coordinate_token_ids=[ids[i] for i in positions],own_prefix=dict(x1=bins[0],y1=bins[1]),
                first_illegal_slot=illegal[0],first_illegal_token_position=positions[illegal[0]],
                illegal_slots=illegal,empty_legal_set_predecessor_slots=dead,
                earlier_dead_end_token_positions=[positions[j] for j in dead])


def negative_evidence(records, tokenizer):
    from src.inference.parsing import parse_compact_object_box_closed
    complete, other = [],[]
    for r in records:
        encoded = tokenizer(r['text'],add_special_tokens=False,return_offsets_mapping=True)
        if list(encoded['input_ids'])!=r['token_ids'] or tokenizer.decode(r['token_ids'],skip_special_tokens=False)!=r['text']:
            raise ValueError('saved request token alignment differs')
        parsed = parse_compact_object_box_closed(r['text'],row_id=r['request_id'],row_index=0,
            image_width=r['crop'][2]-r['crop'][0],image_height=r['crop'][3]-r['crop'][1])
        for d in parsed.dropped_predictions:
            entry = dict(request_id=r['request_id'],image_id=r['image_id'],arm=r['arm'],crop=r['crop'],
                         stop_reason=r['stop_reason'],parser_drop=d)
            if d['reason']=='geometry_invalid':
                entry.update(aligned_invalid(r,d,encoded,tokenizer))
                entry['literal_repeat_key'] = hashlib.sha256(canonical([r['image_id'],r['arm'],d['raw_text']]).encode()).hexdigest()
                complete.append(entry)
            else:
                entry['censored'] = r['stop_reason'] != 'eos' and d['char_end']==len(r['text']) and not d['raw_text'].endswith('<|box_end|>')
                other.append(entry)
    counts = {}
    for d in complete:
        key=d['literal_repeat_key'];counts[key]=counts.get(key,0)+1
    for d in complete:
        d['literal_repeat_count']=counts[d['literal_repeat_key']]
    return dict(complete_geometry_invalid=complete,malformed_or_censored=other,literal_repeat_counts=counts,
                scope='Own-prefix evidence only; no gradients, invalid-GT construction, or automatic positives')


def local_readback(visible, pilot, local_records):
    from src.eval.saved_rows import iou_xyxy
    plan = local_plan(visible,pilot)
    bank = {p['prediction_id']:p for p in plan['bank']}
    predictions, invalid = candidates(local_records)
    result=[]
    for r in local_records:
        target=bank[r['target_prediction_id']]
        detections=[p for p in predictions if p['prediction_id'].startswith(r['request_id']+':p')]
        peers=[p for p in predictions if p['image_id']==r['image_id'] and p['prediction_id'].startswith(r['target_prediction_id']+':local:') and not p['prediction_id'].startswith(r['request_id']+':p')]
        result.append(dict(request_id=r['request_id'],target_prediction_id=target['prediction_id'],detections=[
            dict(prediction=p,target_iou=iou_xyxy(target['coord_bins_1000'],p['coord_bins_1000']),category_agrees=target['description']==p['description'],
                 same_category_competitors=[dict(prediction_id=q['prediction_id'],iou=iou_xyxy(q['coord_bins_1000'],p['coord_bins_1000'])) for q in bank.values() if q['image_id']==p['image_id'] and q['prediction_id']!=target['prediction_id'] and q['description']==p['description']],
                 cross_view=[dict(prediction_id=q['prediction_id'],category_agrees=q['description']==p['description'],iou=iou_xyxy(q['coord_bins_1000'],p['coord_bins_1000'])) for q in peers]) for p in detections]))
    return dict(evidence=result,invalid=invalid,admission='none; localization/competition evidence only')


def compact_local(plan, records):
    """Frozen prediction-only scores, one paired ROI at a time, no edge ledger."""
    from collections import Counter
    from src.eval.saved_rows import iou_xyxy
    expected = {r['request_id']:r for r in plan['requests']}
    actual = {r['request_id']:r for r in records}
    if len(actual)!=len(records) or set(actual)!=set(expected):
        raise ValueError('missing/duplicate/unexpected local requests')
    for rid,r in actual.items():
        if any(r.get(k)!=v for k,v in expected[rid].items()):
            raise ValueError('target/view request pairing changed')
    result=[]
    for t in plan['selected']:
        pools=[];burdens=[]
        for scale in (1,2):
            rid=f"{t['prediction_id']}:local:{scale}"
            if rid not in actual:
                raise ValueError('selected target lacks paired views')
            r=actual[rid]; valid,invalid=candidates([r]);unique={}
            for d in sorted(valid,key=lambda d:d['prediction_id']):
                unique.setdefault((d['description'],tuple(d['coord_bins_1000'])),d)
            pool=[d for d in unique.values() if d['description']==t['description']]
            pools.append(pool)
            burdens.append(dict(request_id=rid,stop_reason=r['stop_reason'],generated_tokens=r['generated_tokens'],
                valid_occurrences=len(valid),literal_unique=len(unique),same_category_unique=len(pool),
                invalid=len(invalid),invalid_reasons=dict(Counter(d['reason'] for d in invalid))))
        overlaps=[[iou_xyxy(t['coord_bins_1000'],d['coord_bins_1000']) for d in pool] for pool in pools]
        U,A=0.0,0.0;pair=None
        for i,a in enumerate(pools[0]):
            for j,b in enumerate(pools[1]):
                u=iou_xyxy(a['coord_bins_1000'],b['coord_bins_1000']);U=max(U,u)
                score=min(overlaps[0][i],overlaps[1][j],u)
                if pair is None or score>A:
                    A=score;pair=(a,b)
        witnesses=[]
        for d in pair or ():
            competitors=sorted((q for q in plan['bank'] if q['image_id']==t['image_id'] and q['description']==t['description'] and q['prediction_id']!=t['prediction_id']),key=lambda q:q['prediction_id'])
            best=None;best_iou=0.0
            for q in competitors:
                v=iou_xyxy(q['coord_bins_1000'],d['coord_bins_1000'])
                if best is None or v>best_iou:
                    best=q['prediction_id'];best_iou=v
            target_iou=iou_xyxy(t['coord_bins_1000'],d['coord_bins_1000'])
            witnesses.append(dict(prediction={k:d[k] for k in ('prediction_id','image_id','description','coord_bins_1000')},target_iou=target_iou,
                strongest_competitor_id=best,strongest_competitor_iou=best_iou,target_minus_competitor=target_iou-best_iou))
        result.append(dict(target_prediction_id=t['prediction_id'],image_id=t['image_id'],
            stratum=dict(arm=t['arm'],known=t['known_iou50_proxy'],support_nonzero=t['verification_scores']['other_query_iou50_support']>0,boundary=t['crop_boundary']),
            B=t['verification_scores']['other_query_iou50_support'],L1=max(overlaps[0],default=0.0),L2=max(overlaps[1],default=0.0),U=U,A=A,witnesses=witnesses,views=burdens))
    return dict(status='prediction_only_no_admission',targets=result)


def annotation_proxy(target, witnesses, reference):
    """Offline same-category annotation sets; never consumed by scoring."""
    from src.eval.saved_rows import iou_xyxy
    def G(box):
        return [dict(annotation_id=str(o['coco_ann_id']),hidden=o['coco_ann_id']<0,cohort=reference['cohort'])
                for o in reference['objects'] if o['desc']==box['description'] and iou_xyxy(o['bbox_2d'],box['coord_bins_1000'])>=0.5]
    groups=[G(target)]+[G(w['prediction']) for w in witnesses]
    groups+= [[] for _ in range(3-len(groups))]
    sets=[{o['annotation_id'] for o in g} for g in groups];gt,ga,gb=sets
    if any(len(g)>1 for g in sets):
        outcome='ambiguous_multiple'
    elif len(gt)==len(ga)==len(gb)==1 and gt==ga==gb:
        outcome='same_singleton_target'
    elif len(ga)==len(gb)==1 and ga==gb and ga!=gt:
        outcome='same_singleton_neighbor'
    elif len(ga)==len(gb)==1 and ga!=gb:
        outcome='witness_disagreement'
    else:
        outcome='unsupported_or_incomplete'
    return dict(G_target=groups[0],G_native=groups[1],G_double=groups[2],empty=[not x for x in sets],multiple=[len(x)>1 for x in sets],
                outcome=outcome,possible_repair_addition=not gt and bool(ga or gb))


def original_coverage(truth, predictions):
    """Existing class-agnostic matcher, category agreement only after assignment."""
    from src.eval.saved_rows import one_to_one_matches
    per_image=[]
    for r in truth:
        pool=[p for p in predictions if p['image_id']==r['image_id']]
        refs=[dict(owner_id=str(o['coco_ann_id']),reference_coord_bins_1000=o['bbox_2d']) for o in r['objects']]
        matched=one_to_one_matches(refs,pool,0.5);by={p['prediction_id']:p for p in pool}
        row=dict(image_id=r['image_id'],cohort=r['cohort'],selected=len(pool),hidden_denominator=sum(o['coco_ann_id']<0 for o in r['objects']),visible_denominator=sum(o['coco_ann_id']>=0 for o in r['objects']),hidden=0,visible=0,category_agreeing_hidden=0,category_agreeing_visible=0,category_disagreements=0,unmatched=len(pool)-len(matched))
        for m in matched:
            o=r['objects'][m['reference_index']];kind='hidden' if o['coco_ann_id']<0 else 'visible';row[kind]+=1
            if o['desc']==by[m['prediction_id']]['description']:row['category_agreeing_'+kind]+=1
            else:row['category_disagreements']+=1
        per_image.append(row)
    fields=[k for k in per_image[0] if k not in ('image_id','cohort')]
    def total(rows):return {k:sum(r[k] for r in rows) for k in fields}
    return dict(combined=total(per_image),cohorts={c:total([r for r in per_image if r['cohort']==c]) for c in sorted({r['cohort'] for r in truth})},images=per_image)


def verification_diagnostics(plan, scores, records, truth):
    """Offline annotation proxies and all tied thresholds, no operating selection."""
    from collections import Counter
    targets={p['prediction_id']:p for p in plan['selected']};refs={r['image_id']:r for r in truth}
    if set(targets)!={x['target_prediction_id'] for x in scores['targets']}:
        raise ValueError('score/target identity mismatch')
    outcomes={x['target_prediction_id']:annotation_proxy(targets[x['target_prediction_id']],x['witnesses'],refs[x['image_id']]) for x in scores['targets']}
    strata={canonical(x['stratum']) for x in scores['targets']}
    def summarize(chosen, compact=False):
        ids=[x['target_prediction_id'] for x in chosen]
        coverage=original_coverage(truth,[targets[i] for i in ids])
        result=dict(retained=len(ids),outcomes=dict(Counter(outcomes[i]['outcome'] for i in ids)),possible_repair_addition=sum(outcomes[i]['possible_repair_addition'] for i in ids),original_coverage=coverage['combined'] if compact else coverage)
        if not compact:
            result['outcomes_by_image']={str(r['image_id']):dict(Counter(outcomes[i]['outcome'] for i in ids if targets[i]['image_id']==r['image_id'])) for r in truth}
            result['outcomes_by_cohort']={c:dict(Counter(outcomes[i]['outcome'] for i in ids if refs[targets[i]['image_id']]['cohort']==c)) for c in sorted({r['cohort'] for r in truth})}
        return result
    curves=[]
    for name in ('B','L1','L2','A'):
        for threshold in sorted({x[name] for x in scores['targets']},reverse=True):
            chosen=[x for x in scores['targets'] if x[name]>=threshold]
            curves.append(dict(score=name,threshold=threshold,tie_count=sum(x[name]==threshold for x in scores['targets']),
                **summarize(chosen),strata={key:summarize([x for x in chosen if canonical(x['stratum'])==key],compact=True) for key in sorted(strata)}))
    # Coverage uses all literal occurrences and the existing matcher, never IoU clusters.
    local,invalid=candidates(records)
    scale={r['request_id']:r['view_scale'] for r in records}
    local_coverage={str(s):original_coverage(truth,[p for p in local if s=='union' or scale[p['prediction_id'].rsplit(':p',1)[0]]==s]) for s in (1,2,'union')}
    return dict(status='offline_annotation_proxy_no_admission',outcome_rule='Multiple sets take ambiguous precedence; otherwise same singleton target, same singleton neighbor, disagreeing singleton witnesses, then unsupported/incomplete. Empty-target supported-witness repair flag is separate.',
        targets=[dict(target_prediction_id=x['target_prediction_id'],image_id=x['image_id'],cohort=refs[x['image_id']]['cohort'],stratum=x['stratum'],**outcomes[x['target_prediction_id']]) for x in scores['targets']],
        selected_baseline=summarize(scores['targets']),curves=curves,local_output_coverage=local_coverage,local_invalid=len(invalid),
        scope='Fixed diagnostic strata; annotation IDs/extent proxies, not physical precision. Refined5 redraw identity unresolved. No threshold chosen.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare','plan','acquire','review','evaluate','local-plan','local-acquire','local-select','negative','local-review','local-score','local-diagnostics'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--visible', type=Path)
    p.add_argument('--policy', type=Path)
    p.add_argument('--raw', type=Path)
    p.add_argument('--truth', type=Path)
    p.add_argument('--pilot', type=Path)
    p.add_argument('--selection', type=Path)
    p.add_argument('--scores', type=Path)
    p.add_argument('--reviews', type=Path)
    p.add_argument('--image-ids', type=int, nargs='*')
    a = p.parse_args()
    if a.mode=='prepare':
        prepare(a.output)
    elif a.mode in ('local-plan','local-acquire'):
        execute(a.visible,a.policy,a.output,a.image_ids,generate=a.mode=='local-acquire',local_raw=a.pilot)
    elif a.mode=='local-select':
        write(a.output,local_plan(load(a.visible),read_frozen(a.pilot)))
    elif a.mode=='negative':
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
        write(a.output,negative_evidence(read_frozen(a.pilot),tokenizer))
    elif a.mode=='local-score':
        write(a.output,compact_local(load(a.selection),read_frozen(a.raw)))
    elif a.mode=='local-diagnostics':
        write(a.output,verification_diagnostics(load(a.selection),load(a.scores),read_frozen(a.raw),load(a.truth)))
    elif a.mode=='local-review':
        write(a.output,local_readback(load(a.visible),read_frozen(a.pilot),read_frozen(a.raw)))
    elif a.mode in ('plan','acquire'):
        execute(a.visible,a.policy,a.output,a.image_ids,generate=a.mode=='acquire')
    elif a.mode=='review':
        write(a.output,admission_inputs(load(a.visible),read_frozen(a.raw)))
    else:
        write(a.output,evaluate(load(a.truth),read_frozen(a.raw),load(a.reviews)))


if __name__=='__main__':
    main()
