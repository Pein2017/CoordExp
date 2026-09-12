"""Finite history-only learning contrast; exact CPU inputs and endpoint ledger.

Training is owned by ``training.py``. This lane never changes accepted producers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time
import traceback

from probes.dora_owner_learning.candidate_opportunity import digest, file_hash, require, score
from probes.source_rweak_row_cross.run import native_record

BASE = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration')
ROOT = BASE / '2026-09-12-parallel-owner-research/history'
SOURCE_MANIFEST = BASE / '2026-09-11-positive-branch-vs-repeat-event/input-preparation/candidate_manifest.json'
SOURCE_BANK = BASE / '2026-09-11-native-escape-witness/candidate_manifest.json'
ENDPOINT_SOURCE = BASE / '2026-09-11-positive-progress-matched-control/endpoint-preparation/packet.json'
ANCHOR_INPUT = BASE / '2026-09-11-positive-branch-vs-repeat-event/trainer-preparation/inputs.json'
MARGIN_INPUT = BASE / '2026-09-11-margin-preserved-positive-branch/margin-preparation/inputs.json'
SELECTED = ('417044-c01', '477415-c02')
TARGET_OWNERS = {'417044-c01': '1083135', '477415-c02': '1583762'}
ROW_START, ROW_END, EOS = 151646, 151649, 151645
CAP = 3084


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    return {'path': str(path), 'sha256': file_hash(path)}


def publish(path, value):
    from src.artifacts import publish_json_exclusive
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_json_exclusive(path, value)


def complete_rows(ids):
    """Split stored IDs, never decode/re-tokenize or silently drop a tail."""
    require(isinstance(ids, list) and ids and all(type(x) is int for x in ids), 'literal token IDs')
    require(EOS not in ids, 'EOS cannot enter a complete-row prefix')
    result, start = [], 0
    for index, token in enumerate(ids):
        if index == start:
            require(token == ROW_START, 'row must begin at object_ref_start')
        elif token == ROW_START:
            raise ValueError('nested/incomplete object row')
        if token == ROW_END:
            row = ids[start:index + 1]
            require(len(row) >= 8, 'incomplete row grammar')
            result.append(row)
            start = index + 1
    require(start == len(ids), 'incomplete trailing row')
    return result


def earlier_swap(ids):
    rows = complete_rows(ids)
    require(len(rows) >= 3, 'insufficient_history: final row leaves fewer than two earlier rows')
    require(rows[0] != rows[1], 'first-two-row swap must be nontrivial')
    perm = [1, 0, *range(2, len(rows))]
    changed = [token for index in perm for token in rows[index]]
    validate_history_pair(ids, changed)
    return changed, perm


def validate_history_pair(p, q):
    a, b = complete_rows(p), complete_rows(q)
    require(p != q, 'history perturbation must not be identity')
    require(len(p) == len(q) and len(a) == len(b), 'history token/row count changed')
    require(a[-1] == b[-1], 'final row changed')
    require(Counter(map(tuple, a)) == Counter(map(tuple, b)), 'complete row multiset changed')
    require(b == [a[1], a[0], *a[2:]], 'unregistered earlier-row permutation')


def text_rows(text, count):
    rows = re.findall(r'<\|object_ref_start\|>.*?<\|box_end\|>', text)
    require(len(rows) == count and ''.join(rows) == text, 'source text row framing changed')
    return rows


def exposure_arms(case_ids=SELECTED, updates=32):
    require(type(updates) is int and updates > 0 and updates % 2 == 0, 'balanced even update count')
    return {
        arm: {'steps': [
            [{'record_id': f'{cid}.{"Q" if arm == "mixed_PQ" and step % 2 else "P"}', 'weight': 1.0}
             for cid in case_ids]
            for step in range(updates)]}
        for arm in ('fixed_P', 'mixed_PQ')
    }


def validate_exposure(arms, records, updates=32):
    by_id = {r['record_id']: r for r in records}
    require(len(by_id) == len(records), 'duplicate literal record ID')
    expected = exposure_arms(updates=updates)
    require(arms == expected, 'exposure schedule differs from frozen P/PQ design')
    result = {}
    for arm, item in arms.items():
        counts = Counter()
        tokens = Counter()
        for step in item['steps']:
            for exposure in step:
                record = by_id[exposure['record_id']]
                counts[record['record_id']] += 1
                tokens[record['case_id']] += len(record['target_token_ids'])
        result[arm] = {'record_exposures': dict(counts), 'case_target_token_exposures': dict(tokens)}
    require(result['fixed_P']['case_target_token_exposures'] == result['mixed_PQ']['case_target_token_exposures'],
            'total target-token exposure changed')
    for cid in SELECTED:
        require(by_id[f'{cid}.P']['target_token_ids'] == by_id[f'{cid}.Q']['target_token_ids'],
                'same owner is insufficient: literal complete target row changed')
        validate_history_pair(by_id[f'{cid}.P']['prefix_token_ids'], by_id[f'{cid}.Q']['prefix_token_ids'])
    return result


def annotation_read(text, frozen, token_count):
    parsed = native_record(text, frozen['case'], frozen['golden'], 'supplied_prefix')
    return score(parsed, seed=None, length=token_count, stop='supplied_prefix')


def build_packet():
    from probes.dora_owner_learning.repeat_recovery_train import validate_manifest
    manifest = validate_manifest(read(SOURCE_MANIFEST))
    bank, endpoint = read(SOURCE_BANK), read(ENDPOINT_SOURCE)
    frozen_by_image = {str(r['image_id']): r for r in endpoint['eval_records']}
    accepted = {r['candidate_id']: r for r in manifest['positives']}
    census = []
    for case in bank['cases']:
        for candidate in case['candidates']:
            cid = candidate['candidate_id']
            status = 'HOLD_no_fixed_accepted_c_w_package'
            if cid in accepted:
                status = 'admit' if cid in SELECTED else 'HOLD_insufficient_history'
            census.append({'candidate_id': cid, 'history_rows': case['h_complete_row_count'], 'status': status})
    cases, positives, conditionals = [], [], []
    for cid in SELECTED:
        source = accepted[cid]
        frozen = frozen_by_image[source['case_id']]
        p, c, w = (source[k]['token_ids'] for k in ('h', 'c', 'w'))
        q, perm = earlier_swap(p)
        row_texts = text_rows(source['h']['text'], len(perm))
        texts = {'P': source['h']['text'], 'Q': ''.join(row_texts[i] for i in perm)}
        history_reads = {label: annotation_read(text, frozen, len(p)) for label, text in texts.items()}
        target = annotation_read(source['c']['text'], frozen, len(c))
        require(target['50']['owners'] == [TARGET_OWNERS[cid]], 'trusted target annotation identity changed')
        for threshold in ('50', '60', '80'):
            require(set(history_reads['P'][threshold]['owners']) == set(history_reads['Q'][threshold]['owners']),
                    'annotation-relative covered-owner set changed')
            require(TARGET_OWNERS[cid] not in history_reads['P'][threshold]['owners'], 'target already covered')
        common = {
            'case_id': cid, 'example_id': source['image']['row_id'],
            'image': copy.deepcopy(source['image']),
            'prompt_token_ids': source['prompt']['token_ids'],
            'prompt_token_ids_sha256': source['prompt']['ids_sha256'],
        }
        for label, prefix in [('P', p), ('Q', q)]:
            positives.append({**common, 'record_id': f'{cid}.{label}',
                              'prefix_token_ids': prefix, 'target_token_ids': c,
                              'target_owner': TARGET_OWNERS[cid], 'history': label})
        conditionals.append({**common, 'record_id': f'{cid}.fixed_P_c_w',
                             'prefix_token_ids': p + c, 'target_token_ids': w,
                             'kl_positions': list(range(len(w))), 'unknown_mask_policy': 'literal_positions_only'})
        cases.append({
            'case_id': cid, 'image_id': source['case_id'], 'example_id': common['example_id'],
            'target_owner': TARGET_OWNERS[cid], 'target_token_ids': c,
            'target_text': source['c']['text'], 'target_score': target,
            'histories': {label: {'token_ids': prefix, 'ids_sha256': digest(prefix), 'text': texts[label],
                                 'annotation_read': history_reads[label], 'row_count': len(perm)}
                          for label, prefix in [('P', p), ('Q', q)]},
            'permutation': perm, 'source_visual_admission': source['visual_admission'],
            'protection_annotation_read': annotation_read(source['w']['text'], frozen, len(w)),
        })
    arms = exposure_arms()
    ledger = validate_exposure(arms, positives)
    return {
        'schema': 'parallel_owner_history.packet.v1', 'status': 'candidate_cpu_verified',
        'sources': {name: binding(path) for name, path in {
            'accepted_c_w_manifest': SOURCE_MANIFEST, 'finite_candidate_bank': SOURCE_BANK,
            'endpoint_source': ENDPOINT_SOURCE, 'anchor_input': ANCHOR_INPUT,
            'margin_input': MARGIN_INPUT, 'producer': Path(__file__).resolve()}.items()},
        'census': census, 'counts': {'candidate_bank': len(census), 'accepted_c_w': len(accepted), 'selected': len(cases)},
        'cases': cases, 'positive_records': positives, 'conditional_records': conditionals,
        'normal_keys': [r['key'] for r in manifest['normals']['cases']],
        'weights': {'positive': 1.0, 'conditional_kl': 10.0, 'normal_kl': 100.0, 'margin': 10.0},
        'denominators': {'positive': 2.0, 'conditional_kl': 2.0, 'normal_kl': 56.0, 'margin': 56.0},
        'optimizer': {'lr': 1e-5, 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.0, 'foreach': False},
        'clip_gradient_norm': 1.0, 'arms': arms, 'exposure_ledger': ledger,
        'endpoint': {'cap': CAP, 'eos': EOS, 'natural_count': 384, 'conditional_count_per_model': 4,
                     'models': ['Stable50', 'fixed_P', 'mixed_PQ'], 'stable_natural': 'retained_source_bound',
                     'claim_boundary': 'two-case exposed history-learning contrast; Q counterfactual; annotation-relative owners'},
    }


def validate_packet(packet, verify_sources=True):
    require(packet['schema'] == 'parallel_owner_history.packet.v1', 'history schema')
    require([r['case_id'] for r in packet['cases']] == list(SELECTED), 'frozen selected targets')
    ledger = validate_exposure(packet['arms'], packet['positive_records'])
    require(ledger == packet['exposure_ledger'], 'exposure receipt differs')
    require(packet['weights'] == {'positive': 1.0, 'conditional_kl': 10.0, 'normal_kl': 100.0, 'margin': 10.0},
            'common protection changed')
    records = {r['record_id']: r for r in packet['positive_records']}
    require(len(packet['conditional_records']) == 2, 'fixed conditional protection denominator')
    for c in packet['conditional_records']:
        p = records[f"{c['case_id']}.P"]
        require(c['prefix_token_ids'] == p['prefix_token_ids'] + p['target_token_ids'], 'conditional protection left original P+c')
        require(c['kl_positions'] == list(range(len(c['target_token_ids']))), 'conditional literal row mask')
    if verify_sources:
        for source in packet['sources'].values():
            require(file_hash(source['path']) == source['sha256'], f"changed source {source['path']}")
    return packet


def continuation_ledger(prefix_text, free_text, frozen, prefix_tokens, free_tokens, stop):
    """Retain forced rows but never award them free recovery credit."""
    from src.data.geometry import iou_xyxy
    from probes.dora_owner_learning.reward_rows import _pred_objects
    from probes.dora_owner_learning.geometric_dedup_eval import overlap_counts

    prefix = native_record(prefix_text, frozen['case'], frozen['golden'], 'supplied_prefix')
    free = native_record(free_text, frozen['case'], frozen['golden'], stop)
    full = native_record(prefix_text + free_text, frozen['case'], frozen['golden'], stop)
    earlier, _ = _pred_objects(prefix)
    later, _ = _pred_objects(free)
    seen = [box for _, box in earlier]
    strict_free_repeats = 0
    for _, box in later:
        strict_free_repeats += int(any(iou_xyxy(box, old) > .95 for old in seen))
        seen.append(box)
    drops = free['dropped_predictions']
    invalid = sum(row.get('reason') == 'geometry_invalid' for row in drops)
    return {
        'free_parsed': free, 'full_parsed': full,
        'free_score': score(free, seed=None, length=free_tokens, stop=stop),
        'full_score': score(full, seed=None, length=prefix_tokens + free_tokens, stop=stop),
        'full_overlap_counts': overlap_counts(full),
        'burden': {'free_row_starts': free_text.count('<|object_ref_start|>'),
                   'free_valid_rows': len(free['pred']), 'free_strict_repeats_including_history': strict_free_repeats,
                   'free_geometry_invalid': invalid, 'free_other_malformed': len(drops) - invalid,
                   'supplied_row_starts': prefix_text.count('<|object_ref_start|>'),
                   'cap': int(stop == 'length'), 'eos': int(stop == 'im_end')},
    }


def endpoint_jobs(packet, arm, shard):
    require(arm in ('Stable50', 'fixed_P', 'mixed_PQ') and shard in (0, 1), 'endpoint arm/shard')
    source = read(packet['sources']['endpoint_source']['path'])
    jobs = []
    if arm != 'Stable50':
        jobs.extend({'job_id': f"natural:{r['example_id']}", 'example_id': r['example_id'],
                     'case_id': None, 'history': 'natural', 'prefix_token_ids': [], 'prefix_text': '',
                     'target_owner': None, 'kind': 'natural'} for r in source['eval_records'])
    for case in packet['cases']:
        for label in ('P', 'Q'):
            h = case['histories'][label]
            jobs.append({'job_id': f"{case['case_id']}:{label}", 'example_id': case['example_id'],
                         'case_id': case['case_id'], 'history': label, 'prefix_token_ids': h['token_ids'],
                         'prefix_text': h['text'], 'target_owner': case['target_owner'], 'kind': 'conditional'})
    return jobs[shard::2]


def endpoint_rank(endpoint_packet_path, output, shard, physical_gpu):
    """Single visible-GPU native endpoint; no trainer or compatibility fallback."""
    import torch
    from probes.dora_owner_learning.runtime import load_policy
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.source_rweak_row_cross.run import build_requests
    from src.adapters.dora import inspect_dora_adapter_payload
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import prepare_native_inputs

    ep = read(endpoint_packet_path)
    require(ep['schema'] == 'parallel_owner_history.endpoint.v1', 'endpoint schema')
    for source in ep['sources'].values():
        require(file_hash(source['path']) == source['sha256'], f"endpoint changed source {source['path']}")
    packet = validate_packet(read(ep['sources']['history_packet']['path']))
    source = read(packet['sources']['endpoint_source']['path'])
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(physical_gpu) and torch.cuda.device_count() == 1,
            'exact assigned single visible GPU')
    require(physical_gpu in (2, 3) and physical_gpu == 2 + shard, 'history GPU reservation')
    jobs = endpoint_jobs(packet, ep['arm'], shard)
    require(inspect_dora_adapter_payload(ep['adapter']['root'], source['model']['base_model_path']) == ep['adapter'],
            'endpoint adapter payload changed')
    run = Path(output) / f'shard-{shard}'
    require(not run.exists(), 'endpoint shard output collision')
    run.mkdir(parents=True)
    started = time.monotonic()
    terminal = {'status': 'running', 'arm': ep['arm'], 'shard': shard, 'physical_gpu': physical_gpu,
                'pid': os.getpid(), 'model_loads': 0, 'continuations': 0, 'new_tokens': 0,
                'model_forwards': 0, 'image_forwards': 0, 'endpoint_packet_sha256': file_hash(endpoint_packet_path)}
    publish(run / 'launch.json', terminal)
    handles = []
    old_alarm = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('history endpoint rank wall bound')))
        signal.alarm(ep['bounds']['max_rank_seconds'])
        config = checkpoint_config(InferConfig.model_validate(source['config']), ep['adapter']['root'])
        torch.cuda.reset_peak_memory_stats()
        qwen, identity = load_policy(config, device=torch.device('cuda:0'))
        terminal['model_loads'] = 1
        live = identity['model_identity']['adapter']
        require(live['adapter_path'] == ep['adapter']['root'] and live['merged_adapters'] == [], 'loaded adapter identity')
        observed = identity['effective_settings']
        require(observed['observed_model_dtype']['parameter_dtype_names'] == ['torch.float32']
                and observed['observed_attn_implementation'] == 'sdpa', 'endpoint FP32 SDPA')
        publish(run / 'model.json', {'identity': identity, 'adapter': ep['adapter']})
        qwen.model.eval()
        for parameter in qwen.model.parameters():
            parameter.requires_grad_(False)

        def model_forward(*_):
            require(terminal['model_forwards'] < ep['bounds']['max_model_forwards_per_rank'], 'endpoint forward bound')
            terminal['model_forwards'] += 1

        def image_forward(*_):
            terminal['image_forwards'] += 1

        handles.append(qwen.model.register_forward_pre_hook(model_forward))
        visuals = [m for name, m in qwen.model.named_modules() if name.endswith('visual')]
        require(len(visuals) == 1, 'single visual module for cost counter')
        handles.append(visuals[0].register_forward_pre_hook(image_forward))
        by_id = {r['example_id']: r for r in source['eval_records']}
        policy = NativeGenerationPolicy(temperature=0., top_p=1., top_k=0, repetition_penalty=1., use_model_defaults=False)
        with (run / 'rows.jsonl').open('x') as stream:
            for job in jobs:
                frozen = by_id[job['example_id']]
                requests, _ = build_requests(qwen, source['config'], [frozen['case']])
                batch = prepare_native_inputs(qwen.processor, requests, device=torch.device('cuda:0'), record_media_identity=True)
                plan = frozen['case']['image_plan']
                require(list(batch.prompt_token_ids[0]) == frozen['prompt_token_ids']
                        and batch.media_sha256[0] == plan['executed_media_sha256']
                        and list(batch.image_grids[0]) == plan['observed_image_grid_thw'], 'endpoint prompt/media/grid identity')
                prefix = job['prefix_token_ids']
                require(qwen.tokenizer.decode(prefix, skip_special_tokens=False) == job['prefix_text'], 'literal prefix text/ID identity')
                budget = CAP - len(prefix)
                with torch.inference_mode():
                    generated = generate_continuations(qwen.model, batch, extensions=[prefix], budgets=[budget],
                                                       eos_token_id=EOS, pad_token_id=qwen.tokenizer.pad_token_id,
                                                       policy=policy, trace='none')[0]
                ids = list(generated.token_ids)
                require(generated.request_id == job['example_id'] and ids and len(ids) <= budget, 'native request/budget')
                require((generated.stop_reason == 'im_end' and ids[-1] == EOS and EOS not in ids[:-1])
                        or (generated.stop_reason == 'length' and len(ids) == budget and EOS not in ids), 'native terminal identity')
                free_text = qwen.tokenizer.decode(ids, skip_special_tokens=False)
                row = {**job, 'schema': 'parallel_owner_history.endpoint_row.v1', 'arm': ep['arm'], 'shard': shard,
                       'endpoint_packet_sha256': file_hash(endpoint_packet_path), 'free_token_ids': ids,
                       'free_token_ids_sha256': digest(ids), 'free_text': free_text, 'stop_reason': generated.stop_reason,
                       'remaining_budget': budget, 'image_id': str(frozen['image_id']), 'split': frozen['split'],
                       'prompt_token_ids_sha256': digest(frozen['prompt_token_ids']),
                       'executed_media_sha256': batch.media_sha256[0], 'observed_image_grid_thw': list(batch.image_grids[0]),
                       **continuation_ledger(job['prefix_text'], free_text, frozen, len(prefix), len(ids), generated.stop_reason)}
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                terminal['continuations'] += 1
                terminal['new_tokens'] += len(ids)
        require(terminal['continuations'] == len(jobs), 'all endpoint jobs completed')
        terminal.update(status='completed', exit_code=0)
    except BaseException as exc:
        terminal.update(status='failed', exit_code=1, error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        for handle in handles:
            handle.remove()
        terminal.update(elapsed_seconds=time.monotonic() - started,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
                        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        if terminal['status'] == 'completed':
            for key in ('peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes', 'peak_rss_bytes'):
                if terminal[key] > ep['bounds']['max_memory_bytes']:
                    terminal.update(status='failed', exit_code=1, error=f'{key} exceeded bound')
        publish(run / 'terminal.json', terminal)
    require(terminal['status'] == 'completed', 'endpoint terminal acceptance')


def endpoint_launch(endpoint_packet_path, output):
    output = Path(output)
    require(not output.exists(), 'endpoint launch output collision')
    output.mkdir(parents=True)
    launches, running = [], []
    for shard, gpu in enumerate((2, 3)):
        command = [sys.executable, '-m', 'probes.parallel_owner_research.history', 'endpoint-rank', '--endpoint-packet', str(endpoint_packet_path),
                   '--output', str(output), '--shard', str(shard), '--physical-gpu', str(gpu)]
        log = (output / f'shard-{shard}.log').open('x')
        process = subprocess.Popen(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}, stdout=log, stderr=subprocess.STDOUT)
        launches.append({'shard': shard, 'gpu': gpu, 'pid': process.pid, 'command': command})
        running.append((process, log))
    publish(output / 'process-launch.json', {'launches': launches, 'endpoint_packet': binding(endpoint_packet_path)})
    results = []
    for launch, (process, log) in zip(launches, running):
        results.append({**launch, 'exit_code': process.wait()})
        log.close()
    publish(output / 'process-exits.json', {'results': results})
    require(all(r['exit_code'] == 0 for r in results), 'endpoint subprocess failure; full logs retained')


def prepare_endpoint(packet_path, arm, output_path, trained_root=None):
    from src.adapters.dora import inspect_dora_adapter_payload

    packet = validate_packet(read(packet_path))
    source = read(packet['sources']['endpoint_source']['path'])
    sources = {'history_packet': binding(packet_path), 'producer': binding(Path(__file__).resolve())}
    if arm == 'Stable50':
        anchor = read(packet['sources']['accepted_c_w_manifest']['path'])
        adapter_root = anchor['model_identity']['effective_adapter']['path']
        adapter = inspect_dora_adapter_payload(adapter_root, source['model']['base_model_path'])
    else:
        from probes.parallel_owner_research.training import verify_receipt
        require(arm in ('fixed_P', 'mixed_PQ') and trained_root is not None, 'trained endpoint arm/root')
        trained_root = Path(trained_root)
        receipt = verify_receipt(trained_root)
        require(receipt['arm'] == arm and receipt['updates'] == 32, 'fixed final training endpoint')
        require(receipt['status'] == 'technically_completed_cold_pending', 'training mechanical admission')
        cold = read(trained_root / 'cold-check.json')
        require(cold['schema'] == 'parallel_owner_training.cold_check.v1' and cold['status'] == 'passed', 'cold reload passed')
        require(cold['training_receipt'] == binding(trained_root / 'receipt.json')
                and cold['saved_adapter'] == receipt['saved_adapter'], 'cold checkpoint/receipt identity')
        sources.update(training_receipt=binding(trained_root / 'receipt.json'), cold_check=binding(trained_root / 'cold-check.json'),
                       training_input=receipt['input'])
        adapter = receipt['saved_adapter']
    ep = {'schema': 'parallel_owner_history.endpoint.v1', 'arm': arm, 'sources': sources, 'adapter': adapter,
          'bounds': {'max_rank_seconds': 6000, 'max_memory_bytes': 24 * 1024**3,
                     'max_model_forwards_per_rank': 194 * CAP if arm != 'Stable50' else 2 * CAP},
          'jobs_per_rank': [len(endpoint_jobs(packet, arm, shard)) for shard in (0, 1)]}
    publish(output_path, ep)
    return ep


def merge_endpoint(endpoint_packet_path, output):
    """Read back actual parser/owner/burden values and exact job denominators."""
    ep, output = read(endpoint_packet_path), Path(output)
    packet = validate_packet(read(ep['sources']['history_packet']['path']))
    source = read(packet['sources']['endpoint_source']['path'])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(source['model']['base_model_path'], local_files_only=True)
    frozen_by_id = {r['example_id']: r for r in source['eval_records']}
    exits = read(output / 'process-exits.json')['results']
    require(len(exits) == 2 and {r['shard'] for r in exits} == {0, 1} and all(r['exit_code'] == 0 for r in exits), 'endpoint outer exits')
    rows, terminals = [], []
    for shard in (0, 1):
        terminal = read(output / f'shard-{shard}/terminal.json')
        require(terminal['status'] == 'completed' and terminal['exit_code'] == 0, 'endpoint terminal completion')
        expected = endpoint_jobs(packet, ep['arm'], shard)
        actual = [json.loads(line) for line in (output / f'shard-{shard}/rows.jsonl').read_text().splitlines() if line.strip()]
        require(len(actual) == len(expected) == terminal['continuations'], 'endpoint job count')
        for row, job in zip(actual, expected):
            require(all(row[k] == v for k, v in job.items()), 'endpoint job/forced-prefix identity')
            require(row['arm'] == ep['arm'] and row['shard'] == shard, 'endpoint arm/shard identity')
            require(row['endpoint_packet_sha256'] == file_hash(endpoint_packet_path), 'endpoint packet identity')
            require(row['free_token_ids_sha256'] == digest(row['free_token_ids'])
                    and row['remaining_budget'] == CAP - len(row['prefix_token_ids']), 'exact free tokens/budget')
            require(tokenizer.decode(row['free_token_ids'], skip_special_tokens=False) == row['free_text'], 'consumer raw token/text identity')
            frozen = frozen_by_id[row['example_id']]
            fresh = continuation_ledger(row['prefix_text'], row['free_text'], frozen,
                                        len(row['prefix_token_ids']), len(row['free_token_ids']), row['stop_reason'])
            require(all(row[key] == value for key, value in fresh.items()), 'fresh endpoint parser/owner/burden consumer differs')
        rows.extend(actual)
        terminals.append(terminal)
    natural = [r for r in rows if r['kind'] == 'natural']
    conditional = [r for r in rows if r['kind'] == 'conditional']
    require(len(natural) == (0 if ep['arm'] == 'Stable50' else 384) and len(conditional) == 4, 'endpoint final scientific denominator')
    summaries = {}
    if natural:
        panel_ids = {'union384': set(frozen_by_id), 'targets': {c['example_id'] for c in packet['cases']}}
        panel_ids['reference56'] = {r['example_id'] for r in natural if r['split'] == 'reference56'}
        panel_ids['train256'] = {r['example_id'] for r in natural if r['split'] != 'dev128'}
        panel_ids['dev128'] = {r['example_id'] for r in natural if r['split'] == 'dev128'}
        require([len(panel_ids[name]) for name in ('reference56', 'train256', 'dev128')] == [56, 256, 128],
                'source-bound exposed/reference panel identities')
        panel_ids['other_than_reference'] = panel_ids['union384'] - panel_ids['reference56']
        for name, ids in panel_ids.items():
            subset = [r for r in natural if r['example_id'] in ids]
            require(len(subset) == len(ids), f'{name} natural denominator')
            summary = {'images': len(subset), 'owners': {}, 'burden': {}}
            for threshold in ('50', '60', '80'):
                gained, lost, retained = [], [], []
                tp = fp = fn = 0
                for r in subset:
                    old = set(frozen_by_id[r['example_id']]['stable_score'][threshold]['owners'])
                    new = set(r['free_score'][threshold]['owners'])
                    gained.extend(f"{r['image_id']}:{owner}" for owner in sorted(new - old))
                    lost.extend(f"{r['image_id']}:{owner}" for owner in sorted(old - new))
                    retained.extend(f"{r['image_id']}:{owner}" for owner in sorted(old & new))
                    s = r['free_score'][threshold]
                    tp += s['tp']; fp += s['fp']; fn += s['fn']
                summary['owners'][threshold] = {'gained': gained, 'lost': lost, 'retained': retained,
                                                'tp': tp, 'fp': fp, 'fn': fn, 'f1': 2 * tp / (2 * tp + fp + fn) if tp else 0.0}
            for key in subset[0]['burden']:
                summary['burden'][key] = sum(r['burden'][key] for r in subset)
            summaries[name] = summary
    targets = []
    for case in packet['cases']:
        natural_row = next((r for r in natural if r['example_id'] == case['example_id']), None)
        for label in ('natural', 'P', 'Q'):
            r = natural_row if label == 'natural' else next(r for r in conditional if r['case_id'] == case['case_id'] and r['history'] == label)
            s = r['free_score'] if r else frozen_by_id[case['example_id']]['stable_score']
            targets.append({'case_id': case['case_id'], 'history': label, 'owner': case['target_owner'],
                            'recovered': {t: case['target_owner'] in s[t]['owners'] for t in ('50', '60', '80')},
                            'score': s, 'burden': r['burden'] if r else None,
                            'source': 'actual_free' if r else 'retained_Stable50_natural'})
    result = {'schema': 'parallel_owner_history.endpoint_result.v1', 'status': 'candidate_cpu_verified',
              'arm': ep['arm'], 'endpoint_packet': binding(endpoint_packet_path), 'natural_count': len(natural),
              'conditional_count': len(conditional), 'panels': summaries, 'targets': targets,
              'allocated_gpu_hours': sum(t['elapsed_seconds'] for t in terminals) / 3600,
              'model_forwards': sum(t['model_forwards'] for t in terminals),
              'image_forwards': sum(t['image_forwards'] for t in terminals),
              'row_sources': [binding(output / f'shard-{s}/rows.jsonl') for s in (0, 1)]}
    publish(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'verify', 'endpoint-prepare', 'endpoint-rank', 'endpoint-launch', 'endpoint-merge'])
    parser.add_argument('--packet', type=Path, default=ROOT / 'preparation/packet.json')
    parser.add_argument('--endpoint-packet', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--shard', type=int)
    parser.add_argument('--physical-gpu', type=int)
    parser.add_argument('--arm', choices=['Stable50', 'fixed_P', 'mixed_PQ'])
    parser.add_argument('--trained-root', type=Path)
    args = parser.parse_args()
    if args.command == 'endpoint-rank':
        endpoint_rank(args.endpoint_packet, args.output, args.shard, args.physical_gpu)
        return
    if args.command == 'endpoint-launch':
        endpoint_launch(args.endpoint_packet, args.output)
        return
    if args.command == 'endpoint-prepare':
        ep = prepare_endpoint(args.packet, args.arm, args.endpoint_packet, args.trained_root)
        print(json.dumps({'arm': ep['arm'], 'jobs_per_rank': ep['jobs_per_rank']}))
        return
    if args.command == 'endpoint-merge':
        result = merge_endpoint(args.endpoint_packet, args.output)
        print(json.dumps({k: result[k] for k in ('status', 'arm', 'natural_count', 'conditional_count', 'allocated_gpu_hours')}))
        return
    if args.command == 'prepare':
        packet = validate_packet(build_packet())
        publish(args.packet, packet)
    else:
        packet = validate_packet(read(args.packet))
    print(json.dumps({'status': packet['status'], 'packet': str(args.packet), 'sha256': file_hash(args.packet),
                      'counts': packet['counts'], 'exposure': packet['exposure_ledger']}, sort_keys=True))


if __name__ == '__main__':
    main()
