"""Frozen successor supply: reuse N16 rows, bounded native acquisition, no fit."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

from probes.native_owner_scale import evaluation as e
from probes.native_owner_scale import scale
from probes.parallel_owner_research.history import complete_rows, continuation_ledger
from src.inference.bound_requests import build_bound_native_requests as build_requests
from src.eval.native_rows import native_detection_record as native_record
from src.data.geometry import iou_xyxy

ROOT = e.ROOT.parent.parent / '2026-09-13-owner-successor-scale-throughput' / 'supply'
OLD = e.ROOT
TRAIN = Path('/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000_xy_sorted/train.coord.jsonl')
GPUS = (2, 3, 4, 5)
CAP = 3084
SALT = 'owner-successor-supply4096-density-v1:'


def density(count):
    return 'sparse_1_4' if count <= 4 else ('medium_5_9' if count <= 9 else ('dense_10_19' if count <= 19 else 'very_dense_20_plus'))


def select_pool(rows, reused_ids, excluded, size=4096):
    by_id = {int(row['image_id']): row for row in rows}
    e.require(len(by_id) == len(rows), 'duplicate train identity')
    reuse = sorted(set(reused_ids) - set(excluded))
    e.require(set(reuse) <= set(by_id), 'reused rows outside train source')
    buckets = {name: [] for name in ('sparse_1_4', 'medium_5_9', 'dense_10_19', 'very_dense_20_plus')}
    for image_id, row in by_id.items():
        if image_id not in excluded and image_id not in reuse:
            buckets[density(len(row['objects']))].append(image_id)
    for values in buckets.values():
        values.sort(key=lambda image_id: hashlib.sha256(f'{SALT}{image_id}'.encode()).hexdigest())
    selected = list(reuse)
    # Equal round-robin density allocation, deterministic within each stratum.
    positions = dict.fromkeys(buckets, 0)
    while len(selected) < size:
        before = len(selected)
        for name, values in buckets.items():
            if len(selected) < size and positions[name] < len(values):
                selected.append(values[positions[name]])
                positions[name] += 1
        e.require(len(selected) > before, 'insufficient fixed train population')
    return selected, reuse, positions


def freeze(output=ROOT):
    output = Path(output)
    old_path = OLD / 'candidate-panel-bound-v11.json'
    old = e.read(old_path)
    prior = e.read(e.PRIOR_SELECTION)
    refs = set(old['strata']['reference54']['image_ids'])
    independent = set(old['strata']['fresh256']['image_ids']) | set(prior['image_ids'])
    excluded = refs | independent
    old_records = {r['image_id']: r for r in old['records']}
    old_rows = e.read_jsonl(OLD / 'candidate-natural-v11/rows.jsonl')
    by_old = {r['image_id']: r for r in old_rows}
    raw = e.read_jsonl(TRAIN)
    exposed = {r['image_id'] for r in old['records'] if r['image_id'] not in old['strata']['fresh256']['image_ids']}
    selected, reused, fill_counts = select_pool(raw, exposed, excluded)
    e.require(len(reused) == 330 and len(refs) == 54, 'exposed minus protected reference denominator')
    indexed = {int(r['image_id']): (i, r) for i, r in enumerate(raw)}
    items = []
    for image_id in selected:
        index, row = indexed[image_id]
        path = (TRAIN.parent / row['images'][0]).resolve()
        item = {'image_id': image_id, 'row_index': index, 'density_stratum': density(len(row['objects'])), 'object_count': len(row['objects']), 'image_path': str(path), 'image_sha256': e.file_hash(path), 'raw_row': row, 'source': 'reused_n16' if image_id in reused else 'new_train'}
        if image_id in reused:
            item['frozen'] = old_records[image_id]
            item['natural'] = by_old[image_id]
        items.append(item)
    pool = {'schema': 'owner_successor_scale.supply_pool.v1', 'status': 'frozen_before_new_rollouts', 'source': e.binding(TRAIN), 'previous_packet': e.binding(old_path), 'previous_rows': e.binding(OLD / 'candidate-natural-v11/rows.jsonl'), 'prior_independent_selection': e.binding(e.PRIOR_SELECTION), 'protected_reference_ids': sorted(refs), 'independent_panel_ids': sorted(independent), 'excluded_ids': sorted(excluded), 'image_ids': selected, 'image_ids_sha256': e.digest(selected), 'reused_ids': reused, 'new_ids': [i for i in selected if i not in reused], 'stratum_fill_counts': fill_counts, 'stratum_total_counts': dict(Counter(x['density_stratum'] for x in items)), 'selection_rule': 'Reuse all eligible exposed N16 rows; equal round-robin density bins 1-4/5-9/10-19/20+, SHA256 salted ID order within bin. No outcome-adaptive replacement.', 'config': copy.deepcopy(old['config']), 'adapter': old['scaled_terminal']['adapter'], 'generation': old['generation'], 'items': items, 'bounds': {'images': 4096, 'reused_natural': len(reused), 'new_natural': 4096-len(reused), 'max_histories_per_image': 2, 'max_candidates_per_history': 2, 'max_conditional_continuations': 4096*4, 'max_new_natural_forwards': (4096-len(reused))*CAP, 'max_conditional_forwards': 4096*4*CAP, 'package_cap': 128, 'aim_images': 64, 'floor_packages': 32, 'floor_images': 16}, 'claim_boundary': 'Machine nomination only; c, immediate w and first repeated earlier owner need physical review. No EOS labels or GT-negative inference.'}
    pool['config']['data']['input_jsonl'] = str(TRAIN)
    e.publish(output / 'pool.json', pool)
    return pool


def repeat_histories(ids, frozen, tokenizer):
    """First two actual later-row strict events, exact preceding native tokens."""
    if not ids or ids[0] != scale.ROW_START or scale.ROW_END not in ids:
        return []
    body=list(ids[:-1] if ids and ids[-1] == e.EOS else ids)
    # A malformed leading fragment has no admissible literal history. Its raw
    # natural output remains in images.jsonl; it is not repaired or dropped.
    first_end=body.index(scale.ROW_END)
    if scale.ROW_START in body[1:first_end] or first_end+1<8:
        return []
    prefix, _ = scale._maximal_complete_prefix(body)
    literal = complete_rows(prefix)
    seen, histories, offset = [], [], 0
    for ordinal, tokens in enumerate(literal):
        parsed = native_record(tokenizer.decode(tokens, skip_special_tokens=False), frozen['case'], frozen['golden'], 'supplied_prefix')
        if len(parsed['pred']) == 1 and not parsed['dropped_predictions']:
            box = parsed['pred'][0]['bbox']
            earlier = [x for x in seen if iou_xyxy(box, x['bbox']) > 0.95]
            if earlier:
                first = earlier[0]
                histories.append({'history_index': len(histories), 'later_row_ordinal': ordinal, 'h_token_ids': prefix[:offset], 'repeat_token_ids': tokens, 'repeat_bbox': box, 'first_owner': first, 'history_boundary': 'literal_before_later_strict_repeat'})
                if len(histories) == 2:
                    break
            seen.append({'row_ordinal': ordinal, 'bbox': box, 'token_ids': tokens, 'description': parsed['pred'][0]['description']})
        offset += len(tokens)
    return histories


def nominate(frozen, natural, tokenizer):
    jobs, held = [], []
    for history in repeat_histories(natural['action_ids'], frozen, tokenizer):
        h = history['h_token_ids']
        h_text = tokenizer.decode(h, skip_special_tokens=False)
        parsed_h = native_record(h_text, frozen['case'], frozen['golden'], 'supplied_prefix')
        seen = [p['bbox'] for p in parsed_h['pred']]
        candidates = []
        for gt in frozen['golden']['gt']:
            text = scale._candidate_text(gt)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            parsed = native_record(text, frozen['case'], frozen['golden'], 'supplied_prefix')
            e.require(len(parsed['pred']) == 1 and len(complete_rows(tokens)) == 1, 'literal GT c shape')
            c_score=scale.score(parsed,seed=None,length=len(tokens),stop='supplied_prefix')
            if c_score['50']['owners'] != [str(gt['object_id'])]:
                held.append({'image_id':frozen['image_id'],'history_index':history['history_index'],'owner_id':str(gt['object_id']),'status':'neutral_literal_c_owner_ambiguous'})
                continue
            box = parsed['pred'][0]['bbox']
            overlap = max((iou_xyxy(box, old) for old in seen), default=0.0)
            if overlap > scale.ABSENCE_MAX_IOU or len(h)+len(tokens) >= CAP:
                held.append({'image_id': frozen['image_id'], 'history_index': history['history_index'], 'owner_id': str(gt['object_id']), 'status': 'neutral_absence_or_budget_not_clear'})
                continue
            candidates.append({'owner_id': str(gt['object_id']), 'c_text': text, 'c_token_ids': tokens, 'c_bbox': box, 'c_description': gt['description'], 'c_source': gt, 'max_any_class_history_iou': overlap})
        candidates.sort(key=lambda c: (-(c['c_bbox'][2]-c['c_bbox'][0])*(c['c_bbox'][3]-c['c_bbox'][1]), c['owner_id']))
        for ci, candidate in enumerate(candidates[:2]):
            jobs.append({**history, **candidate, 'image_id': frozen['image_id'], 'example_id': frozen['example_id'], 'job_id': f"{frozen['image_id']}:h{history['history_index']}:c{ci}", 'candidate_index': ci, 'h_text': h_text, 'remaining_budget': CAP-len(h)-len(candidate['c_token_ids']), 'status': 'pre_nominated_physical_review_pending'})
    return jobs, held


def prepare_slice(output=ROOT):
    from transformers import AutoTokenizer
    output=Path(output)
    pool=e.read(output/'pool.json')
    tokenizer=AutoTokenizer.from_pretrained(pool['config']['model']['base_model'],local_files_only=True)
    reused_jobs=[]
    for item in pool['items']:
        if item['source']=='reused_n16':
            jobs,_=nominate(item['frozen'],item['natural'],tokenizer)
            reused_jobs.extend(jobs)
    e.publish(output/'reused-nominations-v2.json',{'pool':e.binding(output/'pool.json'),'jobs':reused_jobs})
    chosen=[]
    for job in reused_jobs:
        if job['candidate_index']==0 and job['image_id'] not in {j['image_id'] for j in chosen}:
            chosen.append(job)
        if len(chosen)==2:break
    e.require(len(chosen)==2,'two reused representative c histories unavailable')
    ids=pool['new_ids'][:2]+[j['image_id'] for j in chosen]
    packet={'schema':'owner_successor_scale.supply_launch.v1','status':'frozen_cpu_gate_pending','pool':e.binding(output/'pool.json'),'producer':e.binding(Path(__file__)),'stage':'slice','image_ids':ids,'new_natural_image_ids':pool['new_ids'][:2],'conditional_job_ids':[j['job_id'] for j in chosen],'conditional_jobs':chosen,'physical_gpus':list(GPUS),'call_ids':[f'natural:{i}' for i in pool['new_ids'][:2]]+[j['job_id'] for j in chosen],'bounds':{'max_calls':4,'max_model_forwards':4*CAP,'max_image_forwards':4,'max_rank_model_loads':1,'max_total_model_loads':4,'max_peak_cuda_bytes_per_rank':80*1024**3,'max_peak_rss_bytes_per_rank':96*1024**3},'remaining_rule':'Never rerun these natural IDs or completed conditional job IDs; remaining acquisition needs root release.'}
    e.publish(output/'slice-packet-v2.json',packet)
    return packet


def prepare_remainder(output=ROOT):
    output=Path(output)
    pool=e.read(output/'pool.json')
    prior=e.read(output/'slice-v2/consumer.json')
    e.require(prior['completed_call_ids']==['natural:469913','natural:135785','39654:h0:c0','70033:h0:c0'],'exact accepted slice call IDs')
    packet={'schema':'owner_successor_scale.supply_launch.v1','status':'root_authorized_remainder','stage':'remainder','pool':e.binding(output/'pool.json'),'producer':e.binding(Path(__file__)),'image_ids':pool['image_ids'],'physical_gpus':list(GPUS),'stage1_consumer':e.binding(output/'slice-v2/consumer.json'),'stage1_images':[e.binding(output/f'slice-v2/shard-{s}/images.jsonl') for s in range(4)],'excluded_completed_call_ids':prior['completed_call_ids'],'excluded_conditional_job_ids':[x for x in prior['completed_call_ids'] if not x.startswith('natural:')],'conditional_job_ids':None,'bounds':{**pool['bounds'],'new_natural':3764,'reused_natural':332,'max_conditional_continuations':4096*4-2,'max_model_loads':4,'max_new_natural_forwards':3764*CAP,'max_conditional_forwards':(4096*4-2)*CAP},'claim_boundary':'Same immutable pool and nomination rule. Preserve both accepted-slice no-w outcomes; no outcome-based substitution, natural rerun, automatic retry, physical admission or training.'}
    e.publish(output/'remainder-packet-v1.json',packet)
    return packet


def run_items(pool, packet):
    items={x['image_id']:x for x in pool['items']}
    if packet.get('stage1_images'):
        e.require(e.binding(packet['stage1_consumer']['path'])==packet['stage1_consumer'],'accepted slice receipt identity')
        for source in packet['stage1_images']:
            e.require(e.binding(source['path'])==source,'accepted slice image rows identity')
            for record in e.read_jsonl(source['path']):
                if record['source']=='new_train':
                    image_id=record['image_id']
                    e.require(f'natural:{image_id}' in packet['excluded_completed_call_ids'],'unregistered natural reuse')
                    items[image_id]={**items[image_id],'source':'reused_stage1','frozen':record['frozen'],'natural':record['natural']}
        e.require(sum(x['source']=='new_train' for x in items.values())==3764,'remaining natural denominator')
    return items


def materialize(item, config, qwen, device='cuda:0'):
    from probes.dora_owner_learning.runtime import build_request
    from src.config.inference import InferConfig
    from src.data.examples import raw_example_from_jsonl_row
    from src.qwen.native import prepare_native_inputs
    e.require(e.file_hash(Path(item['image_path'])) == item['image_sha256'], 'bound image bytes')
    if item['source'] != 'new_train':
        frozen = item['frozen']
        requests, _ = build_requests(qwen, config, [e._candidate_materialized_case(frozen['case'], config)])
        batch = prepare_native_inputs(qwen.processor, requests, device=device, record_media_identity=True)
        plan = frozen['case']['image_plan']
        e.require(list(batch.prompt_token_ids[0]) == frozen['prompt_token_ids'] and batch.media_sha256[0] == plan['executed_media_sha256'] and list(batch.image_grids[0]) == plan['observed_image_grid_thw'], 'reused frozen native identity')
        return frozen, batch
    raw = raw_example_from_jsonl_row(item['raw_row'], jsonl_path=TRAIN, row_number=item['row_index']+1, raw_line=json.dumps(item['raw_row']))
    request, plan, _ = build_request(raw, config=InferConfig.model_validate(config), qwen=qwen, row_index=item['row_index'])
    batch = prepare_native_inputs(qwen.processor, [request], device=device, record_media_identity=True)
    plan = plan.to_artifact_dict()
    plan.update(observed_image_grid_thw=list(batch.image_grids[0]), executed_media_sha256=batch.media_sha256[0], backend_prompt_token_count=len(batch.prompt_token_ids[0]), backend_projection_evidence_kind='hf_executed_tensors')
    case = {'row_id': str(raw.example_id), 'row_index': item['row_index'], 'image_path': str(raw.image.path), 'image_width': raw.image.width, 'image_height': raw.image.height, 'input_record': item['raw_row'], 'image_plan': plan}
    golden = {k: v for k, v in case.items() if k not in ('input_record', 'image_plan')}
    golden.update(example_id=str(raw.example_id), gt=[obj.to_artifact_dict() for obj in raw.objects])
    return {'image_id': item['image_id'], 'example_id': str(raw.example_id), 'case': case, 'golden': golden, 'prompt_token_ids': list(batch.prompt_token_ids[0])}, batch


def worker(packet_path, output, shard):
    import torch
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    packet, output = e.read(packet_path), Path(output)
    e.require(e.binding(Path(__file__))==packet['producer'],'frozen supply producer identity')
    pool = e.read(packet['pool']['path'])
    e.require(e.binding(packet['pool']['path']) == packet['pool'], 'pool binding')
    e.require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(GPUS[shard]) and torch.cuda.device_count() == 1, 'supply GPU identity')
    e.require(not output.exists(), 'no supply shard retry')
    output.mkdir(parents=True)
    ids = packet['image_ids'][shard::len(packet['physical_gpus'])]
    items = run_items(pool, packet)
    terminal = {'status': 'running', 'packet': e.binding(packet_path), 'shard': shard, 'physical_gpu': GPUS[shard], 'image_ids': ids, 'model_loads': 0, 'model_forwards': 0, 'image_forwards': 0, 'new_natural': 0, 'reused_natural': 0, 'conditional': 0, 'new_tokens': 0}
    e.publish(output/'launch.json', terminal)
    started, handles = time.monotonic(), []
    try:
        for value in pool['adapter']['files']:
            e.require(e.file_hash(Path(pool['adapter']['root'])/value['relative_path'])==value['sha256'],'N16 adapter file identity')
        config = checkpoint_config(InferConfig.model_validate(pool['config']), pool['adapter']['root'])
        qwen, identity = load_policy(config, device=torch.device('cuda:0'))
        live = identity['model_identity']['adapter']
        e.require(live['adapter_path'] == pool['adapter']['root'] and live['merged_adapters'] == [], 'N16 unmerged identity')
        effective = identity['effective_settings']
        e.require(effective['observed_model_dtype']['parameter_dtype_names'] == ['torch.float32'] and effective['observed_attn_implementation'] == 'sdpa', 'FP32 SDPA')
        terminal['model_loads'] = 1
        e.publish(output/'model.json', identity)
        e.publish(output/'source.json', {'pool': packet['pool'], 'image_ids': ids, 'images': [{'image_id': i, 'image_path': items[i]['image_path'], 'image_sha256': items[i]['image_sha256'], 'source': items[i]['source']} for i in ids]})
        qwen.model.eval()
        for p in qwen.model.parameters(): p.requires_grad_(False)
        handles.append(qwen.model.register_forward_pre_hook(lambda *_: terminal.__setitem__('model_forwards', terminal['model_forwards']+1)))
        visual = [m for name,m in qwen.model.named_modules() if name.endswith('visual')]
        e.require(len(visual)==1, 'visual hook identity')
        handles.append(visual[0].register_forward_pre_hook(lambda *_: terminal.__setitem__('image_forwards', terminal['image_forwards']+1)))
        policy = NativeGenerationPolicy(temperature=0.,top_p=1.,top_k=0,repetition_penalty=1.,use_model_defaults=False)
        def generate(batch, prefix):
            with torch.inference_mode():
                value = generate_continuations(qwen.model, batch, extensions=[prefix], budgets=[CAP-len(prefix)], eos_token_id=e.EOS, pad_token_id=qwen.tokenizer.pad_token_id, policy=policy, trace='none')[0]
            ids = list(value.token_ids)
            e.accepted._checked_action(ids, value.stop_reason, CAP-len(prefix))
            terminal['new_tokens'] += len(ids)
            return ids, qwen.tokenizer.decode(ids,skip_special_tokens=False), value.stop_reason
        with (output/'images.jsonl').open('x') as images, (output/'jobs.jsonl').open('x') as jobs_stream, (output/'rows.jsonl').open('x') as rows:
            for image_id in ids:
                item = items[image_id]
                frozen,batch = materialize(item,pool['config'],qwen)
                if item['source']!='new_train':
                    natural = item['natural']; terminal['reused_natural'] += 1
                else:
                    token_ids,text,stop = generate(batch,[])
                    natural = {'image_id':image_id,'example_id':frozen['example_id'],'action_ids':token_ids,'text':text,'stop_reason':stop,'parsed':native_record(text,frozen['case'],frozen['golden'],stop),'prefix_ids':[],'forced_ids':[],'remaining_budget':CAP}
                    terminal['new_natural'] += 1
                jobs,held = nominate(frozen,natural,qwen.tokenizer)
                image_record = {'image_id':image_id,'frozen':frozen,'natural':natural,'source':item['source'],'held':held,'nominations':len(jobs)}
                images.write(json.dumps(image_record)+'\n'); images.flush(); os.fsync(images.fileno())
                # Persist all per-image nominations before observing any c outcome.
                for job in jobs: jobs_stream.write(json.dumps(job)+'\n')
                jobs_stream.flush(); os.fsync(jobs_stream.fileno())
                for job in jobs:
                    if job['job_id'] in packet.get('excluded_conditional_job_ids',[]):
                        continue
                    if packet.get('conditional_job_ids') is not None and job['job_id'] not in packet['conditional_job_ids']:
                        continue
                    prefix = job['h_token_ids']+job['c_token_ids']
                    token_ids,text,stop = generate(batch,prefix)
                    ledger = continuation_ledger(job['h_text']+job['c_text'],text,frozen,len(prefix),len(token_ids),stop)
                    seen = [p['bbox'] for p in native_record(job['h_text']+job['c_text'],frozen['case'],frozen['golden'],'supplied_prefix')['pred']]
                    result = {'job_id':job['job_id'],'image_id':image_id,'packet':e.binding(packet_path),'free_token_ids':token_ids,'free_text':text,'stop_reason':stop,'prefix_token_count':len(prefix),'remaining_budget':CAP-len(prefix),**ledger,'local_w':scale._literal_first_w(token_ids,text,ledger['free_parsed'],seen,qwen.tokenizer)}
                    rows.write(json.dumps(result)+'\n');rows.flush();os.fsync(rows.fileno())
                    terminal['conditional'] += 1
        terminal.update(status='completed',exit_code=0)
    except BaseException as exc:
        terminal.update(status='failed',exit_code=1,error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        for handle in handles:handle.remove()
        terminal.update(elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        e.publish(output/'terminal.json',terminal)


def launch(packet_path, output):
    packet,output=e.read(packet_path),Path(output)
    e.require(not output.exists(),'no automatic supply retry');output.mkdir(parents=True)
    processes=[]
    for shard,gpu in enumerate(packet['physical_gpus']):
        log=(output/f'shard-{shard}.log').open('x')
        command=[sys.executable,'-m','probes.owner_successor_scale.data','worker','--packet',str(packet_path),'--output',str(output/f'shard-{shard}'),'--shard',str(shard)]
        process=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false'),stdout=log,stderr=subprocess.STDOUT)
        processes.append((shard,process,log))
    exits=[]
    for shard,process,log in processes:
        exits.append({'shard':shard,'exit_code':process.wait()});log.close()
    e.publish(output/'outer-exits.json',exits)
    e.require(all(x['exit_code']==0 for x in exits),'supply failure; preserve output, no retry')


def consume(packet_path, output):
    """Cold full-budget literal replay and review cards; never physical admission."""
    from transformers import AutoTokenizer
    from PIL import Image,ImageDraw
    from src.vis.rendering import _font,_load_image,_save
    packet,output=e.read(packet_path),Path(output)
    pool=e.read(packet['pool']['path'])
    tokenizer=AutoTokenizer.from_pretrained(pool['config']['model']['base_model'],local_files_only=True)
    records,jobs,results,terminals=[],[],[],[]
    exits=e.read(output/'outer-exits.json')
    e.require(len(exits)==len(packet['physical_gpus']) and all(x['exit_code']==0 for x in exits),'all worker exits')
    for shard in range(len(packet['physical_gpus'])):
        run=output/f'shard-{shard}'
        terminal=e.read(run/'terminal.json');terminals.append(terminal)
        e.require(terminal['status']=='completed' and terminal['packet']==e.binding(packet_path),'terminal identity')
        records.extend(e.read_jsonl(run/'images.jsonl'));jobs.extend(e.read_jsonl(run/'jobs.jsonl'));results.extend(e.read_jsonl(run/'rows.jsonl'))
    e.require(len(records)==len(packet['image_ids']) and {x['image_id'] for x in records}==set(packet['image_ids']),'exact image denominator')
    frozen={x['image_id']:x['frozen'] for x in records};by_job={j['job_id']:j for j in jobs}
    e.require(len(by_job)==len(jobs),'unique nominated jobs')
    expected=set(packet['conditional_job_ids']) if packet.get('conditional_job_ids') is not None else set(by_job)-set(packet.get('excluded_conditional_job_ids',[]))
    e.require(len(results)==len(expected) and {r['job_id'] for r in results}==expected,'exact conditional denominator')
    cards=[]
    for record in records:
        natural=record['natural'];ids=natural['action_ids']
        e.accepted._checked_action(ids,natural['stop_reason'],CAP)
        text=tokenizer.decode(ids,skip_special_tokens=False)
        e.require(text==natural['text'],'natural text/token identity')
        parsed=native_record(text,record['frozen']['case'],record['frozen']['golden'],natural['stop_reason'])
        e.require(parsed==natural['parsed'],'cold natural parser identity')
    for result in results:
        job=by_job[result['job_id']];f=frozen[result['image_id']]
        ids=result['free_token_ids'];prefix=job['h_token_ids']+job['c_token_ids']
        e.require(result['remaining_budget']==CAP-len(prefix),'full total3084 conditional budget')
        e.accepted._checked_action(ids,result['stop_reason'],CAP-len(prefix))
        text=tokenizer.decode(ids,skip_special_tokens=False)
        e.require(text==result['free_text'],'conditional text/token identity')
        ledger=continuation_ledger(job['h_text']+job['c_text'],text,f,len(prefix),len(ids),result['stop_reason'])
        e.require(all(result[k]==v for k,v in ledger.items()),'cold conditional ledger')
        source=_load_image(Path(f['case']['image_path']))
        canvas=Image.new('RGB',(source.width*2+20,source.height+65),'white');canvas.paste(source,(0,45));canvas.paste(source,(source.width+20,45))
        draw=ImageDraw.Draw(canvas);font=_font('regular',16)
        draw.text((5,8),job['job_id']+' / literal original',fill='black',font=font)
        draw.text((source.width+25,8),'cyan c; green immediate w; orange first owner; red repeat',fill='black',font=font)
        boxes=[(job['c_bbox'],'cyan','c'),(job['first_owner']['bbox'],'orange','first'),(job['repeat_bbox'],'red','repeat')]
        if 'w_bbox_xyxy_pixels' in result['local_w']:boxes.append((result['local_w']['w_bbox_xyxy_pixels'],'green','w'))
        for box,color,label in boxes:
            x1,y1,x2,y2=box;xy=(x1+source.width+20,y1+45,x2+source.width+20,y2+45)
            draw.rectangle(xy,outline=color,width=4);draw.text(xy[:2],label,fill=color,stroke_width=1,stroke_fill='black',font=font)
        path=output/'cards'/f"{job['job_id'].replace(':','-')}.png";_save(canvas,path)
        cards.append({'job_id':job['job_id'],'image_id':job['image_id'],'card':e.binding(path),'c_bbox':job['c_bbox'],'first_owner':job['first_owner'],'repeat_bbox':job['repeat_bbox'],'local_w':result['local_w'],'physical_review_status':'pending','first_repeat_owner_review_status':'pending'})
    receipt={'schema':'owner_successor_scale.supply_consumer.v1','status':'candidate_cold_verified_physical_review_pending','packet':e.binding(packet_path),'images':len(records),'new_natural':sum(t['new_natural'] for t in terminals),'reused_natural':sum(t['reused_natural'] for t in terminals),'nominations':len(jobs),'conditional':len(results),'prior_completed_calls':packet.get('excluded_completed_call_ids',[]),'prior_consumer':packet.get('stage1_consumer'),'cost':{k:sum(t[k] for t in terminals) for k in ['model_loads','model_forwards','image_forwards','new_tokens']},'max_worker_elapsed_seconds':max(t['elapsed_seconds'] for t in terminals),'completed_call_ids':[f"natural:{r['image_id']}" for r in records if r['source']=='new_train']+[r['job_id'] for r in results],'cards':cards,'claim_boundary':pool['claim_boundary']}
    e.require(not set(receipt['completed_call_ids']) & set(receipt['prior_completed_calls']),'completed slice call rerun')
    e.publish(output/'consumer.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','prepare-slice','prepare-remainder','launch','worker','consume']);parser.add_argument('--packet');parser.add_argument('--output',default=str(ROOT));parser.add_argument('--shard',type=int)
    args=parser.parse_args()
    if args.command=='freeze':freeze(Path(args.output))
    elif args.command=='prepare-slice':prepare_slice(Path(args.output))
    elif args.command=='prepare-remainder':prepare_remainder(Path(args.output))
    elif args.command=='consume':consume(Path(args.packet),Path(args.output))
    elif args.command=='launch':launch(Path(args.packet),Path(args.output))
    else:worker(Path(args.packet),Path(args.output),args.shard)


if __name__=='__main__':main()
