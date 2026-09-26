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

HUMAN = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-05-static-dynamic-owner-interface-crossover/inputs/human-refined-13.geo_sorted_xy.coord.jsonl')
REFINED = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-history-rereading-mechanism/human-evaluation/annotation-snapshot-v1')
CHECKPOINT = Path('/data/CoordExp/outputs/infra_base/train/qwen3-vl-2b-geo-sorted-xy-untied-axis001-ebs24-4epoch/checkpoints/step-2444')
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
    messages = [{'role':'system','content':policy['prompt']['system']}, {'role':'user','content':[{'type':'image'}, {'type':'text','text':policy['prompt']['user']}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        batch = prepare_native_inputs(processor, [NativeRequest(item['request_id'], chat, image)], record_media_identity=True)
    finally:
        image.close()
    return batch


def execute(visible_path, policy_path, output, image_ids, generate=False):
    visible, policy = load(visible_path), load(policy_path)
    if generate and policy.get('policy_status') != 'lead_released':
        raise ValueError('GPU acquisition policy is not released')
    from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
    selected = [r for r in visible if not image_ids or r['image_id'] in image_ids]
    if image_ids and {r['image_id'] for r in selected} != set(image_ids):
        raise ValueError('unknown image selection')
    plan = request_plan(selected, policy)
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
    write(output/'frozen.json', dict(status='generated' if generate else 'cpu_plan', source_identity=source_identity,
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare','plan','acquire','review','evaluate'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--visible', type=Path)
    p.add_argument('--policy', type=Path)
    p.add_argument('--raw', type=Path)
    p.add_argument('--truth', type=Path)
    p.add_argument('--reviews', type=Path)
    p.add_argument('--image-ids', type=int, nargs='*')
    a = p.parse_args()
    if a.mode=='prepare':
        prepare(a.output)
    elif a.mode in ('plan','acquire'):
        execute(a.visible,a.policy,a.output,a.image_ids,generate=a.mode=='acquire')
    elif a.mode=='review':
        write(a.output,admission_inputs(load(a.visible),read_frozen(a.raw)))
    else:
        write(a.output,evaluate(load(a.truth),read_frozen(a.raw),load(a.reviews)))


if __name__=='__main__':
    main()
