"""Bounded Stable50 two-row conditional witness acquisition; no training."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import time
import traceback

from probes.dora_owner_learning.candidate_opportunity import digest, file_hash, indexed, require, score
from probes.dora_owner_learning.route_access import CONFIG, checkpoint_config, publish, checked_ids
from probes.source_rweak_row_cross.run import build_requests, native_record
from probes.source_rweak_row_cross.owner_row_robustness import native_rows
from probes.dora_owner_learning.reward_rows import _gt_objects, _pred_objects
from src.data.geometry import iou_xyxy

OUT = Path(__file__).parent
ANCHOR = OUT.parent / '2026-09-10-selective-owner-learning-autonomous/positive7-support50-81'
EOS, CAP = 151645, 3084
PROTECTED = {368, 7116, 252411, 465695, 529411, 538814, 540567}


def target_incidence(parsed, owner):
    gi = next(i for i,g in enumerate(parsed['gt']) if str(g['object_id']) == owner)
    category, box = _gt_objects(parsed, row_id=parsed['row_id'])[gi]
    pred,_ = _pred_objects(parsed)
    all_edges = [dict(pred_index=i, category=c, iou=iou_xyxy(box,b)) for i,(c,b) in enumerate(pred)]
    same = [e for e in all_edges if e['category'] == category]
    best_same = max(same,key=lambda e:e['iou'],default=None)
    best_any = max(all_edges,key=lambda e:e['iou'],default=None)
    return dict(owner=owner, same_class_prediction_count=len(same), best_same_class=best_same, best_any_class=best_any,
        compatible_IoU50_edges=[e for e in same if e['iou'] >= .5],
        interpretation='Geometry/category incidence only; omission, class error and localization need visual adjudication.')


def target_row(gt, tok):
    box = gt['bbox']
    require(len(box) == 4 and all(type(v) is int and 0 <= v <= 999 for v in box), 'GT must use native coordinate bins')
    require(box[0] < box[2] and box[1] < box[3], 'invalid target box')
    text = '<|object_ref_start|>' + gt['description'] + '<|object_ref_end|><|box_start|>'
    text += ''.join(f'<|coord_{v}|>' for v in box) + '<|box_end|>'
    ids = tok.encode(text, add_special_tokens=False).ids
    rows = native_rows(ids, tok)
    require(len(rows) == 1 and rows[0]['coords'] == box and tok.decode(ids, skip_special_tokens=False) == text, 'target row serialization')
    return dict(owner=str(gt['object_id']), bbox=box, description=gt['description'], ids=ids, text=text)


def insertion(ids, target, tok, *, after=0):
    """Insert at a whole-row boundary; never replace or splice a suffix."""
    require(ids and ids[-1] == EOS and 0 <= after < len(ids), 'insertion needs a complete trajectory')
    rows = native_rows(ids, tok)
    boundaries = [r['start'] for r in rows] + [len(ids)-1]
    require(after in boundaries, 'after must be the completed first-row boundary')
    index = next((r['start'] for r in rows if r['start'] >= after and tuple(r['coords'][:2]) >= tuple(target['bbox'][:2])), len(ids)-1)
    prefix = ids[:index]
    extension = prefix + target['ids']
    require(EOS not in extension and len(extension) < CAP, 'nonterminal bounded extension')
    return dict(index=index, prefix_ids=prefix, prefix_sha256=digest(prefix), forced_ids=target['ids'],
                extension_ids=extension, free_start=len(extension), remaining_budget=CAP-len(extension))


def preserving(old, new, stop):
    reasons = []
    if not set(old['50']['owners']) <= set(new['50']['owners']): reasons.append('old_owner_loss')
    if new['50']['fp'] > old['50']['fp']: reasons.append('annotation_relative_FP_increase')
    for key in ('strict_repeats', 'parser_drops', 'invalid_predictions'):
        if new[key] > old[key]: reasons.append(key + '_increase')
    if stop != 'im_end': reasons.append('not_native_EOS')
    return reasons


def consume(raw, frozen, tok):
    require(raw['example_id'] == frozen['example_id'], 'raw case identity')
    ids = raw['action_ids']
    checked_ids(ids, raw['stop_reason'])
    require(ids == raw['prefix_ids'] + raw['forced_ids'] + raw['free_ids'], 'replayed/forced/free partition')
    require(len(raw['prefix_ids']) + len(raw['forced_ids']) + raw['remaining_budget'] == CAP and len(raw['free_ids']) <= raw['remaining_budget'], 'total action budget')
    require(raw['prefix_sha256'] == digest(raw['prefix_ids']) and tok.decode(ids, skip_special_tokens=False) == raw['text'], 'prefix/text identity')
    parsed = native_record(raw['text'], frozen['case'], frozen['baseline'], raw['stop_reason'])
    require(parsed == raw['parsed'], 'cold parser identity')
    card = score(parsed, seed=None, length=len(ids), stop=raw['stop_reason'])
    return dict(raw, score=card)


def forced_assigned(row, boundary, owner, tok):
    if row['score']['invalid_predictions'] or row['score']['parser_drops']:
        return False
    try:
        rr = native_rows(row['action_ids'], tok)
    except (RuntimeError, ValueError):
        return False
    require(len(rr) == len(row['parsed']['pred']), 'native token/parser row alignment')
    indices = [i for i, r in enumerate(rr) if r['start'] == boundary['index'] and r['ids'] == boundary['forced_ids']]
    require(len(indices) == 1, 'forced row missing from actual action stream')
    return any(m['owner'] == owner and m['pred_index'] == indices[0] for m in row['score']['50']['matches'])


def reduce_case(records, frozen, tok):
    require(records and records[0]['stage'] == 'natural', 'missing natural baseline')
    cold = [consume(r, frozen, tok) for r in records]
    base = cold[0]
    require(base['action_ids'] == frozen['stable_ids'] and not base['prefix_ids'] and not base['forced_ids'], 'current Stable natural reproduction')
    require(base['score'] == frozen['stable_score'], 'current Stable score identity')
    require(len(cold) >= 2 and cold[1]['stage'] == 'first', 'missing first intervention')
    first = cold[1]; b, d = frozen['targets']
    first_boundary = insertion(base['action_ids'], b, tok)
    require(first['prefix_ids'] == first_boundary['prefix_ids'] and first['forced_ids'] == b['ids'], 'first intervention boundary')
    first_reasons = preserving(base['score'], first['score'], first['stop_reason'])
    if not forced_assigned(first, first_boundary, b['owner'], tok): first_reasons.append('first_forced_row_not_assigned')
    if first_reasons:
        require(len(cold) == 2, 'second intervention after failed first')
        status, reasons = 'first_rejected', first_reasons
    elif d['owner'] in first['score']['50']['owners']:
        require(len(cold) == 2, 'second intervention after single-correction closure')
        status, reasons = 'one_correction_closure', []
    else:
        require(len(cold) == 3 and cold[2]['stage'] == 'second', 'missing second intervention')
        second = cold[2]
        boundary = insertion(first['action_ids'], d, tok, after=first_boundary['free_start'])
        require(second['prefix_ids'] == boundary['prefix_ids'] and second['forced_ids'] == d['ids'], 'second must use actual first suffix')
        require(second['prefix_ids'][:first_boundary['free_start']] == first_boundary['extension_ids'], 'first intervention must remain in second history')
        reasons = preserving(base['score'], second['score'], second['stop_reason'])
        if not forced_assigned(second, boundary, d['owner'], tok): reasons.append('second_forced_row_not_assigned')
        if not {b['owner'], d['owner']} <= set(second['score']['50']['owners']): reasons.append('target_pair_not_recovered')
        if second['score']['50']['fn']: reasons.append('incomplete_annotated_set')
        status = 'second_rejected' if reasons else 'two_correction_closure'
    return dict(example_id=frozen['example_id'], image_id=frozen['image_id'], split=frozen['split'],
                targets=[t['owner'] for t in frozen['targets']], status=status, reasons=reasons,
                first_insertion_at_EOS=first_boundary['index'] == len(base['action_ids'])-1,
                second_insertion_at_EOS=(boundary['index'] == len(first['action_ids'])-1) if len(cold)==3 else None,
                first_forced_row_assigned_in_final=forced_assigned(cold[-1],first_boundary,b['owner'],tok),
                natural_target_incidence=[target_incidence(base['parsed'],t['owner']) for t in frozen['targets']],
                assisted_rows_in_final_trajectory=len(cold)-1,
                assisted_tokens_in_final_trajectory=sum(len(t['ids']) for t in frozen['targets'][:len(cold)-1]),
                stages=[dict(stage=r['stage'], score=r['score'], prefix_tokens=len(r['prefix_ids']),
                    forced_tokens=len(r['forced_ids']), free_tokens=len(r['free_ids']),
                    prefix_sha256=r['prefix_sha256']) for r in cold])


def prepare():
    from tokenizers import Tokenizer
    from probes.dora_owner_learning.selective_preservation_stable_eval import locked_checkpoint
    require(not (OUT/'inputs.json').exists(), 'occupied packet')
    selection = json.loads((OUT/'candidate-inventory.json').read_text())
    manifest = json.loads((ANCHOR/'evaluation/manifest.json').read_text())
    current = indexed(json.loads((ANCHOR/'evaluation/consumer.json').read_text()), 'example_id')
    source = indexed(manifest['records'], 'example_id')
    receipt, adapter, receipt_sha = locked_checkpoint()
    tok = Tokenizer.from_file(manifest['source_model']['base_model_path']+'/tokenizer.json')
    selected = selection['candidates']
    require(8 <= len(selected) <= 24 and len({x['example_id'] for x in selected}) == len(selected), 'fixed 8..24 unique candidate records')
    frozen = []
    for candidate in selected:
        eid = candidate['example_id']; old = source[eid]; cur = current[eid]; s = cur['score']
        require(old['split'] != 'dev128' and old['image_id'] not in PROTECTED, 'train-only new positive pool')
        require(s['50']['fn'] == 2 and not any(s[k] for k in ('strict_repeats','parser_drops','invalid_predictions','cap')), 'exact2 missing / clean stable eligibility')
        require(cur['stop_reason'] == 'im_end' and 3 <= len(cur['parsed']['gt']) <= 12 and len(cur['action_ids']) <= 256, 'bounded moderate panel')
        miss = [g for g in cur['parsed']['gt'] if str(g['object_id']) not in s['50']['owners']]
        require(len(miss) == 2 and set(map(str,candidate['missing_owner_ids'])) == {str(g['object_id']) for g in miss}, 'independent inventory missing-owner agreement')
        miss.sort(key=lambda g: (*g['bbox'], int(g['object_id'])))
        targets = [target_row(g, tok) for g in miss]
        clean_score = score(cur['parsed'], seed=None, length=len(cur['action_ids']), stop=cur['stop_reason'])
        frozen.append(dict(example_id=eid, image_id=old['image_id'], selection_rank=candidate['selection_rank'], split=old['split'], case=old['case'], baseline=old['baseline'],
            prompt_token_ids=old['prompt_token_ids'], stable_ids=cur['action_ids'], stable_score=clean_score, targets=targets))
    frozen.sort(key=lambda c:c['image_id'])
    paths = [OUT/'candidate-inventory.json', Path(__file__), OUT/'test_admission.py', ANCHOR/'evaluation/manifest.json', ANCHOR/'evaluation/consumer.json',
             ANCHOR/'training/receipt.json', CONFIG, Path(manifest['source_model']['base_model_path'])/'tokenizer.json']
    sources = {str(p):file_hash(p) for p in paths}
    for identity in (adapter, receipt['source_embedding']):
        for f in identity['files']:
            p = Path(identity['root'])/f['relative_path']; require(file_hash(p) == f['sha256'], 'model payload changed'); sources[str(p)] = f['sha256']
    for c in frozen:
        p = Path(c['case']['image_path']); h = c['case']['image_plan']['image_content_sha256']
        require(file_hash(p) == h, 'selected image changed'); sources[str(p)] = h
    packet = dict(schema='recursive_owner_admission.v1', candidates=frozen, config=manifest['config'], adapter=adapter,
        source_model=manifest['source_model'], training_receipt_sha256=receipt_sha, source_files=sources,
        shards=[[c['example_id'] for c in frozen[i::8]] for i in range(8)],
        limits=dict(gpus=8, continuations=3*len(frozen), total_action_cap=CAP, seconds_per_shard=900),
        disposition='Conditional witness supply only. No training, GT edit, architecture change or autonomous success claim.')
    publish(OUT/'inputs.json', packet)
    return dict(images=len(frozen), ids=[c['image_id'] for c in frozen], shards=list(map(len,packet['shards'])), input_sha256=file_hash(OUT/'inputs.json'))


def execute(shard):
    import torch
    from tokenizers import Tokenizer
    from src.config.inference import load_research_infer_config
    from src.qwen.native import prepare_native_inputs
    from src.qwen.generation import generate_continuations, NativeGenerationPolicy
    from probes.dora_owner_learning.runtime import load_policy
    p = json.loads((OUT/'inputs.json').read_text()); require(type(shard) is int and 0 <= shard < 8, 'valid shard')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(shard) and torch.cuda.device_count() == 1, 'assigned GPU only')
    for path,h in p['source_files'].items(): require(file_hash(path) == h, 'frozen source changed: '+path)
    run = OUT/f'shard-{shard}'; run.mkdir(exist_ok=False)
    terminal = dict(status='running', shard=shard, pid=os.getpid(), input_sha256=file_hash(OUT/'inputs.json'),
                    model_loads=0, continuations=0, new_tokens=0, model_forwards=0, image_forwards=0)
    publish(run/'launch.json', terminal); start = time.monotonic()
    def expired(*_): raise TimeoutError('900 second shard bound')
    signal.signal(signal.SIGALRM, expired); signal.alarm(900)
    try:
        cfg = checkpoint_config(load_research_infer_config(CONFIG).config, p['adapter']['root'])
        require(str(cfg.model.base_model) == p['source_model']['base_model_path'] and str(cfg.embedding_delta.path) == p['source_model']['source_embedding']['root'], 'base/embedding composition')
        qwen, identity = load_policy(cfg, device=torch.device('cuda:0')); terminal['model_loads'] = 1
        require(identity['effective_settings']['observed_model_dtype']['parameter_dtype_names'] == ['torch.float32'] and identity['effective_settings']['observed_attn_implementation'] == 'sdpa', 'observed frozen numerics')
        require(identity['model_identity']['adapter']['adapter_path'] == p['adapter']['root'] and not identity['model_identity']['adapter']['merged_adapters'], 'observed Stable adapter')
        publish(run/'model.json', identity); publish(run/'config.json', cfg.model_dump(mode='json'))
        qwen.model.eval()
        for parameter in qwen.model.parameters(): parameter.requires_grad_(False)
        tok = Tokenizer.from_file(p['source_model']['base_model_path']+'/tokenizer.json')
        policy = NativeGenerationPolicy(temperature=0., top_p=1., repetition_penalty=1., top_k=0, use_model_defaults=False)
        assigned = indexed(p['candidates'], 'example_id'); n = len(p['shards'][shard])
        def model_count(*_):
            terminal['model_forwards'] += 1; require(terminal['model_forwards'] <= n*3*(CAP+1), 'model forward bound')
        def image_count(*_):
            terminal['image_forwards'] += 1; require(terminal['image_forwards'] <= n*3, 'image forward bound')
        qwen.model.register_forward_pre_hook(model_count)
        visual = [m for name,m in qwen.model.named_modules() if name.endswith('visual')]
        require(len(visual) == 1, 'visual module identity'); visual[0].register_forward_pre_hook(image_count)
        torch.cuda.reset_peak_memory_stats()
        for eid in p['shards'][shard]:
            c = assigned[eid]; case_rows = []
            requests,_ = build_requests(qwen, p['config'], [c['case']])
            batch = prepare_native_inputs(qwen.processor, requests, device=torch.device('cuda:0'), record_media_identity=True)
            plan = c['case']['image_plan']
            require(list(batch.prompt_token_ids[0]) == c['prompt_token_ids'] and batch.media_sha256[0] == plan['executed_media_sha256'] and list(batch.image_grids[0]) == plan['observed_image_grid_thw'], 'live prompt/media/grid identity')
            def generate(stage, boundary):
                extension = boundary['prefix_ids'] + boundary['forced_ids']; budget = CAP-len(extension)
                with torch.inference_mode():
                    result = generate_continuations(qwen.model, batch, extensions=[extension], budgets=[budget], eos_token_id=EOS,
                        pad_token_id=qwen.tokenizer.pad_token_id, policy=policy, trace='none')[0]
                require(result.request_id == eid, 'request association')
                free = list(result.token_ids); ids = extension+free; text = qwen.tokenizer.decode(ids, skip_special_tokens=False)
                r = dict(example_id=eid, stage=stage, action_ids=ids, prefix_ids=boundary['prefix_ids'], forced_ids=boundary['forced_ids'],
                    prefix_sha256=digest(boundary['prefix_ids']), free_ids=free, remaining_budget=budget, text=text, stop_reason=result.stop_reason,
                    parsed=native_record(text,c['case'],c['baseline'],result.stop_reason))
                publish(run/f'{c["image_id"]}-{stage}.json', r); case_rows.append(r)
                terminal['continuations'] += 1; terminal['new_tokens'] += len(free)
                require(terminal['continuations'] <= n*3 and terminal['new_tokens'] <= n*3*CAP, 'generation bound')
                return consume(r,c,tok)
            base = generate('natural', dict(prefix_ids=[],forced_ids=[]))
            require(base['action_ids'] == c['stable_ids'] and base['score'] == c['stable_score'], 'cold Stable natural reproduction')
            b,d = c['targets']; boundary = insertion(base['action_ids'],b,tok)
            first = generate('first',boundary)
            reasons = preserving(base['score'],first['score'],first['stop_reason'])
            if not forced_assigned(first,boundary,b['owner'],tok): reasons.append('first_forced_row_not_assigned')
            if not reasons and d['owner'] not in first['score']['50']['owners']:
                generate('second',insertion(first['action_ids'],d,tok,after=boundary['free_start']))
            result = reduce_case(case_rows,c,tok); publish(run/f'{c["image_id"]}-reduction.json',result)
            print(json.dumps(dict(image_id=c['image_id'],status=result['status'],reasons=result['reasons'])),flush=True)
        require(terminal['image_forwards'] == terminal['continuations'], 'one image forward per continuation')
        terminal['status'] = 'completed'
    except BaseException as exc:
        terminal.update(status='failed',error=repr(exc),traceback=traceback.format_exc()); raise
    finally:
        signal.alarm(0); terminal.update(elapsed_seconds=time.monotonic()-start,
            rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0)
        publish(run/'terminal.json',terminal)


def merge(verify=False):
    from tokenizers import Tokenizer
    from collections import Counter
    p=json.loads((OUT/'inputs.json').read_text()); tok=Tokenizer.from_file(p['source_model']['base_model_path']+'/tokenizer.json')
    exits=json.loads((OUT/'process-exits.json').read_text())['results']
    require(len(exits)==8 and {e['shard'] for e in exits}==set(range(8)) and all(e['exit_code']==0 for e in exits), 'eight successful worker exits')
    byid=indexed(p['candidates'],'example_id'); results=[]; terminals=[]
    for i in range(8):
        run=OUT/f'shard-{i}'; t=json.loads((run/'terminal.json').read_text()); terminals.append(t)
        require(t['status']=='completed' and t['shard']==i and t['input_sha256']==file_hash(OUT/'inputs.json') and t['model_loads']==1,'complete shard identity')
        count=tokens=0
        for eid in p['shards'][i]:
            c=byid[eid]; records=[]
            for stage in ('natural','first','second'):
                path=run/f'{c["image_id"]}-{stage}.json'
                if path.exists(): records.append(json.loads(path.read_text()))
            result=reduce_case(records,c,tok); require(result==json.loads((run/f'{c["image_id"]}-reduction.json').read_text()),'cold result differs')
            results.append(result); count+=len(records); tokens+=sum(len(r['free_ids']) for r in records)
        require(t['continuations']==t['image_forwards']==count and t['new_tokens']==tokens,'cold counters differ')
    results.sort(key=lambda c:c['image_id'])
    reduction=dict(schema='recursive_owner_admission.reduction.v1', images=len(results), counts=dict(Counter(r['status'] for r in results)),
        results=results, total_continuations=sum(t['continuations'] for t in terminals), new_tokens=sum(t['new_tokens'] for t in terminals),
        allocated_gpu_seconds=sum(t['elapsed_seconds'] for t in terminals), model_loads=8, training_steps=0,
        input_sha256=file_hash(OUT/'inputs.json'), claim='Conditional intervention evidence only; no autonomous recovery or trainability established.')
    if verify: require(reduction==json.loads((OUT/'reduction.json').read_text()),'merged readback changed')
    else: publish(OUT/'reduction.json',reduction)
    return {k:v for k,v in reduction.items() if k!='results'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['prepare','execute','merge','verify']); parser.add_argument('--shard',type=int)
    a=parser.parse_args()
    if a.command=='prepare': print(json.dumps(prepare()))
    elif a.command=='execute': execute(a.shard)
    else: print(json.dumps(merge(verify=a.command=='verify')))
