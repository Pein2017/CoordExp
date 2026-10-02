"""Frozen six-view geometry ranking; CPU preparation grants no runtime authority."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import resource
import signal
import struct
import subprocess
import time

from probes import greedy_prefix_branching as b
from probes import iterative_positive as p
from probes.full_label_fit.recipe import full_label_learning_rates
from src.artifacts.git_identity import capture_source_identity, verify_source_identity
from src.qwen.generation import trim_suffix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/research/physical-fn-recovery/2026-10-03/prefix-exposure-ranking-04'
HISTORY = Path('/data/CoordExp/.worktrees/greedy-prefix-native-01/outputs/research/physical-fn-recovery/2026-10-03')
ACCEPTANCE = HISTORY / 'completed-row-crossover-03/lead-acceptance-01.json'
ACCEPTANCE_SHA = 'ec559aac3ed02be82513bd98ed7fa4f21861a7c9879ea01cabef6e942feb0d26'
ORIGINAL = HISTORY / 'greedy-prefix-branching-01/released-contract-01.json'
ORIGINAL_SHA = '1dbfb7d38fced4e986d8cf909412ba617405903379427f5deece3a87c6768bdc'
WEIGHT = '07ea98e90220a9126042a27b3a76f60e9955320fed4bf77c2a8803ac9d52001a'
SITES = [('7511-626', 620, 638, 'y2'), ('351017-1507', 1499, 915, 'y1')]
OFFSETS = [0, -1, 1, -2, 2]
ARMS = ['R-single', 'R-multiple']
PHASES = ['native-anchor', 'train-R-single', 'native-R-single',
          'train-R-multiple', 'native-R-multiple']
OPTIMIZER = dict(kind='AdamW', language_lr=1e-5, delta_lr=5e-6,
                 betas=[.9, .999], eps=1e-8, weight_decay=0, clip=1,
                 seed=92711, schedule='constant', fresh_per_arm=True)
BOUNDS = dict(gpus=1, ranks=1, concurrent_sequences=1, optimizer_steps=32,
              training_replays=192, HF_diagnostics=30, score_requests=30,
              continuation_requests=54, requests=84, generated_tokens=166566,
              context=4456, kv_cache_bytes=2*1024**3, phase_seconds=1800,
              cleanup_seconds=30, active_seconds=9000)
FIELDS = ['generation', 'norm', 'checkpoint', 'checkpoint_files', 'checkpoint_acceptance',
          'weight_identity', 'base_model', 'base_config_sha256', 'tokenizer_sha256',
          'coordinate_ids', 'vocab_size', 'runtime', 'input_path', 'label_path', 'raw_paths']


def source_paths():
    return b.source_paths() + ['probes/prefix_exposure_ranking.py',
                              'tests/probes/test_prefix_exposure_ranking.py']


def historical():
    assert b.sha(ACCEPTANCE) == ACCEPTANCE_SHA, 'round03 acceptance drift'
    assert b.sha(ORIGINAL) == ORIGINAL_SHA, 'original release drift'
    accepted = b.load(ACCEPTANCE)
    assert accepted['status'] == 'lead-accepted' and accepted['cleanup_verified']
    assert accepted['native_exit'] == accepted['readback_exit'] == 0
    bindings = {str(ACCEPTANCE): ACCEPTANCE_SHA, str(ORIGINAL): ORIGINAL_SHA}
    for key in ['worker_candidate', 'released_contract', 'readback', 'terminal']:
        item = accepted[key]
        bindings[item['path']] = item['sha256']
    previous = b.load(accepted['released_contract']['path'])
    old = b.load(ORIGINAL)
    for key in ['checkpoint', 'checkpoint_files', 'weight_identity', 'input_path', 'label_path']:
        assert previous[key] == old[key], key
    assert old['weight_identity'] == WEIGHT and old['norm'] == 'off'
    bindings.update(old['bindings'])
    for path, expected in bindings.items():
        assert b.sha(Path(path)) == expected, path
    return old, bindings


def tensor_headers(path):
    with path.open('rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        assert 0 < size < 10**7
        header = json.loads(f.read(size))
    return {k: dict(dtype=v['dtype'], shape=v['shape'])
            for k, v in header.items() if k != '__metadata__'}


def training_binding(c):
    checkpoint = Path(c['checkpoint'])
    assert b.load(checkpoint/'identity.json') == c['checkpoint_files']
    for name, expected in c['checkpoint_files'].items():
        assert b.sha(checkpoint/name) == expected, name
    adapter = b.load(checkpoint/'adapter/adapter_config.json')
    assert adapter['use_dora'] and adapter['r'] == 16 and adapter['lora_alpha'] == 32
    assert adapter['lora_dropout'] == 0
    tensors = tensor_headers(checkpoint/'adapter/adapter_model.safetensors')
    assert len(tensors) == 588 and all(v['dtype'] == 'F32' for v in tensors.values())
    assert all('language_model' in k and 'lora_' in k and 'visual' not in k for k in tensors)
    deltas = tensor_headers(checkpoint/'special_token_embeddings/special_token_embeddings.safetensors')
    assert deltas == {k: dict(dtype='F32', shape=[1004, 2048])
                      for k in ['input_embed_delta', 'output_embed_delta']}
    metadata = b.load(checkpoint/'special_token_embeddings/special_token_embeddings.json')
    assert metadata['tie_word_embeddings'] is False and metadata['tensor_dtype'] == 'float32'
    assert metadata['base_config_sha256'] == c['base_config_sha256']
    assert metadata['tokenizer_sha256'] == c['tokenizer_sha256']
    assert len(metadata['token_ids']) == 1004
    assert full_label_learning_rates(None, 1) == full_label_learning_rates(None, 16) == [1e-5, 5e-6]
    return dict(adapter=tensors, deltas=deltas, rank=16, alpha=32, dropout=0,
                base='BF16', attention='flash_attention_2', attention_dropout=0,
                activation_checkpointing=False, frozen=['base', 'vision', 'projector'],
                optimizer=OPTIMIZER, updates_per_arm=16, view_weight=1/6)


def contexts(c, tokenizer):
    """Derive the first eligible neighboring earlier row from parser and literal IDs."""
    result = []
    for name, changed_position, original, coordinate in SITES:
        image, position = map(int, name.split('-'))
        raw = b.load(c['raw_paths'][str(image)])
        assert raw['raw_identity'] == b.online.seal(raw, raw['producer'])['raw_identity']
        rows, _ = b.online.observations(dict(raw, arm='greedy'), tokenizer)
        target = next(r for r in rows if r['coordinate_positions'][0] == position)
        assert target['bbox'][0] == 999 and not target['valid']
        chosen = None
        for row in reversed([r for r in rows if r['valid'] and r['positions'][-1] < target['positions'][0]]):
            for slot in [3, 2, 1, 0]:
                values = row['bbox']
                neighbors = [values[:slot] + [values[slot]+d] + values[slot+1:] for d in [-2, 2]]
                if all(all(0 <= v <= 999 for v in a) and a[0] < a[2] and a[1] < a[3] for a in neighbors):
                    chosen = row, slot
                    break
            if chosen:
                break
        assert chosen is not None
        row, slot = chosen
        assert (row['coordinate_positions'][slot], row['bbox'][slot], ['x1','y1','x2','y2'][slot]) == (changed_position, original, coordinate), 'frozen neighborhood disagreement'
        for offset in OFFSETS:
            prefix = list(raw['token_ids'][:position])  # target excluded
            prefix[changed_position] = c['coordinate_ids'][original+offset]
            assert len(prefix) == position and prefix[target['positions'][0]:] == raw['token_ids'][target['positions'][0]:position]
            differences = [i for i, (a,z) in enumerate(zip(prefix, raw['token_ids'][:position], strict=True)) if a != z]
            assert differences == ([changed_position] if offset else [])
            processed = raw['prompt_token_ids'] + prefix
            result.append(dict(context_id=f'{name}/{offset:+d}', site_id=name, image_id=image,
                offset=offset, split='original' if offset == 0 else 'training_neighbor' if abs(offset) == 1 else 'held_out',
                changed_position=changed_position, coordinate=coordinate, base_bin=original, changed_bin=original+offset,
                earlier_row=row, target_position=position, historical_target_token_id=raw['token_ids'][position],
                extension=prefix, processed_prompt_token_ids=processed,
                input_prefix_identity=b.online.identity(processed), causal_logits_position=len(processed)-1,
                raw_identity=raw['raw_identity'], support=list(c['coordinate_ids'][:999]), budget=1))
    return result


def frontend(c):
    q = b.rows.frontend()
    assert q.model is None and str(q.base_model_path) == c['base_model']
    assert q.base_config_sha256 == c['base_config_sha256'] and q.tokenizer_sha256 == c['tokenizer_sha256']
    assert q.config.text_config.vocab_size == c['vocab_size']
    assert [q.tokenizer.convert_tokens_to_ids(f'<|coord_{i}|>') for i in range(1000)] == c['coordinate_ids']
    inputs = b.load(c['input_path'])
    requests, reports = {}, []
    for item in inputs:
        image = item['image_id']
        assert b.sha(Path(item['image_path'])) == item['image_sha256']
        batch = b.online.native_batch(q, item)
        request = b.online.vllm_requests(q, {image:item}, [image])[0]
        chat = q.tokenizer.encode(request.chat_text, add_special_tokens=False)
        from vllm.multimodal.processing.processor import PromptReplacement, _apply_token_matches_with_placeholders
        visual = math.prod(item['image_grid_thw']) // q.processor.image_processor.merge_size**2
        image_token = q.processor.image_token_id
        update = {'image': [[PromptReplacement(modality='image', target=[image_token], replacement=[image_token]*visual).resolve(0)]]}
        expanded, matches, placeholders = _apply_token_matches_with_placeholders(chat, update)
        assert expanded == item['prompt_token_ids'] and matches == {'image':[0]}
        assert placeholders['image'][0].length == visual and batch.media_sha256[0] == item['media_sha256']
        reports.append(dict(image_id=image, input_identity=b.online.identity(item),
            prompt_tokens=len(expanded), processed_prompt_identity=b.online.identity(expanded),
            unexpanded_chat_token_ids=chat, media_sha256=item['media_sha256'],
            image_sha256=item['image_sha256'], image_grid_thw=item['image_grid_thw'], visual_tokens=visual))
        requests[image] = request
    assert len(inputs) == 18 and len(requests) == 18
    if 'images' in c: assert c['images'] == [i['image_id'] for i in inputs], 'original image order drift'
    if 'eos_token_id' in c: assert c['eos_token_id'] == q.tokenizer.convert_tokens_to_ids('<|im_end|>')
    assert max(r['prompt_tokens'] for r in reports) == 1372 and next(r for r in reports if r['image_id']==1584)['prompt_tokens'] == 1372
    if 'input_reports' in c:
        assert c['input_reports'] == reports, 'input/prompt/media/order drift'
    if 'contexts' in c:
        assert c['contexts'] == contexts(c, q.tokenizer), 'frozen context drift'
        for context in c['contexts']:
            item = next(i for i in inputs if i['image_id']==context['image_id'])
            assert context['processed_prompt_token_ids'] == item['prompt_token_ids'] + context['extension']
            # Each image has its own visual expansion; reconstruct it for this context.
            count = math.prod(item['image_grid_thw']) // q.processor.image_processor.merge_size**2
            own_update = {'image': [[PromptReplacement(modality='image', target=[image_token], replacement=[image_token]*count).resolve(0)]]}
            expanded, _, _ = _apply_token_matches_with_placeholders(
                next(r['unexpanded_chat_token_ids'] for r in reports if r['image_id']==context['image_id'])+context['extension'], own_update)
            assert expanded == context['processed_prompt_token_ids']
            assert len(expanded)+1 <= BOUNDS['context']
    return q, {i['image_id']:i for i in inputs}, requests, reports


def views(c, arm):
    assert arm in ARMS, 'unknown arm'
    offsets = [0,0,0] if arm == 'R-single' else [0,-1,1]
    return [next(x for x in c['contexts'] if x['site_id']==site and x['offset']==offset)
            for site, *_ in SITES for offset in offsets]


def phase_commands(contract, output):
    result = []
    for phase in PHASES:
        kind, arm = phase.split('-', 1)
        result.append(['python', '-m', 'probes.prefix_exposure_ranking', kind,
            '--contract', str(contract), '--contract-sha256', 'EXACT_RELEASE_SHA256',
            '--output', str(output), '--arm', arm])
    return result


def prepare(output):
    old, bindings = historical()
    c = {k:old[k] for k in FIELDS}
    c.update(schema='prefix-exposure-ranking-v1', source=None, native_released=False,
        execution_checkout=None, output_root=str(OUT), evidence_bindings=bindings,
        bounds=BOUNDS, arms=ARMS, phases=PHASES, offsets=OFFSETS, annotation_denominator=570,
        objective='standalone_mean_six_unweighted_Gmax', gt_role='evaluator_only',
        acquisition_role='prediction_selected_structural_neighbors', training=None)
    c['training'] = training_binding(c)
    q, inputs, _, reports = frontend(c)
    c['images'] = list(inputs)
    c['input_reports'] = reports
    c['contexts'] = contexts(c, q.tokenizer)
    c['eos_token_id'] = q.tokenizer.convert_tokens_to_ids('<|im_end|>')
    labels = b.load(c['label_path'])
    assert len(labels) == 18 and sum(len(i['objects']) for i in labels) == 570
    assert {i['image_id'] for i in labels} == set(c['images'])
    c['proposed_commands'] = phase_commands('EXACT_RELEASE_CONTRACT', 'SELECTED_EXECUTION_OUTPUT')
    output.mkdir(parents=True, exist_ok=False)
    b.write(output/'manifest.json', c)
    frontend(c)
    b.write(output/'cpu-layout.json', dict(native_launched=False, model_loaded=False,
        contexts=[{k:v for k,v in x.items() if k not in ['extension','processed_prompt_token_ids','support','earlier_row']} for x in c['contexts']],
        inputs=reports, maximum_prompt=1372, maximum_context=4456, training=c['training'], bounds=BOUNDS))
    return c


def validate(path, require_release=False, current_source=True):
    c = b.load(path)
    assert c['schema'] == 'prefix-exposure-ranking-v1'
    assert c['bounds'] == BOUNDS and c['arms'] == ARMS and c['phases'] == PHASES and c['offsets'] == OFFSETS
    assert c['objective'] == 'standalone_mean_six_unweighted_Gmax' and c['gt_role'] == 'evaluator_only'
    assert c['acquisition_role'] == 'prediction_selected_structural_neighbors'
    assert c['annotation_denominator'] == 570
    assert c['generation'] == dict(temperature=0,top_p=1,top_k=-1,repetition_penalty=1,min_tokens=0,cap=3084,seed=92711)
    old, bindings = historical()
    assert c['evidence_bindings'] == bindings
    for key in FIELDS:
        assert c[key] == old[key], key
    assert c['training'] == training_binding(c), 'training binding drift'
    for package, expected in c['runtime'].items():
        assert version(package) == expected, package
    assert c['proposed_commands'] == phase_commands('EXACT_RELEASE_CONTRACT', 'SELECTED_EXECUTION_OUTPUT')
    if current_source:
        verify_source_identity(c['source'], required_paths=source_paths(), root=ROOT)
    if require_release:
        assert c['native_released'] is True, 'exact lead release required'
        assert Path(c['execution_checkout']).resolve() == ROOT.resolve()
        assert Path(c['output_root']).resolve() == OUT.resolve()
    return c


def qualify(path, destination):
    c = validate(path, current_source=False)
    frontend(c)
    assert c['source'] is None and c['native_released'] is False
    c.update(source=capture_source_identity(source_paths(), root=ROOT),
             execution_checkout=str(ROOT), output_root=str(OUT))
    b.write(destination, c)


def entry(path, output, expected_sha, phase):
    assert b.sha(path) == expected_sha, 'released manifest drift'
    c = validate(path, require_release=True)
    assert output.resolve().parent == Path(c['output_root']).resolve(), 'output owner drift'
    index = PHASES.index(phase)
    for earlier in PHASES[:index]:
        receipt = b.load(output/earlier/'complete.json')
        assert receipt['phase'] == earlier and receipt['contract_sha256'] == expected_sha
    return c


def weight_binding(c, output, arm):
    if arm == 'anchor':
        return Path(c['checkpoint']), c['weight_identity']
    assert arm in ARMS
    receipt = b.load(output/f'train-{arm}/complete.json')
    checkpoint = output/f'train-{arm}/checkpoint-16'
    assert receipt['arm'] == arm and receipt['anchor_weight_identity'] == WEIGHT
    assert receipt['checkpoint'] == str(checkpoint) and receipt['checkpoint_files'] == b.load(checkpoint/'identity.json')
    for name, sha in receipt['checkpoint_files'].items():
        assert b.sha(checkpoint/name) == sha, 'endpoint checkpoint drift'
    identity = b.online.identity(receipt['checkpoint_files'])
    assert receipt['weight_identity'] == identity
    return checkpoint, identity


def score_measure(c, context, acquired, identity):
    assert acquired['processed_prompt_token_ids'] == context['processed_prompt_token_ids']
    assert len(acquired['token_ids']) == 1 and acquired['vocab_size'] == c['vocab_size']
    assert acquired['score_semantics'] == 'native_raw_logprobs_full_vocabulary'
    scores = {int(k):float(v) for k,v in acquired['full_scores'].items()}
    assert set(scores) == set(range(c['vocab_size'])), 'full vocabulary required'
    assert all(math.isfinite(v) for v in scores.values()), 'nonfinite score'
    assert abs(sum(math.exp(v) for v in scores.values())-1) < 1e-4
    token = acquired['token_ids'][0]
    assert type(token) is int and token in scores and scores[token] == max(scores.values()), 'literal greedy evidence'
    support = set(context['support'])
    assert context['support'] == c['coordinate_ids'][:999]
    margin = max(scores[t] for t in support)-max(v for t,v in scores.items() if t not in support)
    assert acquired['stop_reason'] == ('im_end' if token == c['eos_token_id'] else 'length')
    return dict(context_id=context['context_id'], site_id=context['site_id'], split=context['split'],
        offset=context['offset'], input_prefix_identity=context['input_prefix_identity'], weight_identity=identity,
        emitted_token_id=token, emitted_bin=c['coordinate_ids'].index(token) if token in c['coordinate_ids'] else None,
        literal_legal=token in support, legal_mass=sum(math.exp(scores[t]) for t in support),
        margin=margin, rounded_zero_tie=round(margin, 6)==0, natural_recovery_credit=False)


def natural_measure(c, item, acquired, tokenizer, label, identity):
    assert acquired['request_id'] == item['request_id'] and acquired['weight_identity'] == identity
    tokens, reason = trim_suffix(acquired['token_ids'], budget=3084,
        eos_token_id=c['eos_token_id'], pad_token_id=tokenizer.pad_token_id)
    assert list(tokens) == acquired['token_ids'] and reason == acquired['stop_reason']
    assert all(type(t) is int and 0 <= t < c['vocab_size'] for t in tokens)
    raw = b.online.seal(dict(item, arm='greedy', token_ids=list(tokens), text=tokenizer.decode(tokens,skip_special_tokens=False),
        generated_tokens=len(tokens), stop_reason=reason), dict(kind='prefix_ranking_natural',
        source=c['source']['commit'], weight_identity=identity, norm='off', empty_history=True))
    measurement = b.rows.assess_outputs([label], [], [raw])[0]
    return dict(raw=raw, measurement=measurement, supplied_history=False,
                annotation_unmatched_is_physical_negative=False)


def resources(started, gpu=False):
    result = dict(seconds=time.monotonic()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  reaped_children_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                  cuda_peak_allocated=None, cuda_peak_reserved=None)
    if gpu:
        import torch
        result.update(cuda_peak_allocated=torch.cuda.max_memory_allocated(), cuda_peak_reserved=torch.cuda.max_memory_reserved())
    return result


def native(path, output, expected_sha, arm):
    phase = f'native-{arm}'
    c = entry(path, output, expected_sha, phase)
    q, inputs, requests, reports = frontend(c)
    assert q.tokenizer.convert_tokens_to_ids('<|im_end|>') == c['eos_token_id']
    labels = {i['image_id']:i for i in b.load(c['label_path'])}  # evaluator only
    checkpoint, identity = weight_binding(c, output, arm)
    from src.qwen.vllm_rollout import VllmDoraRollout
    directory = output/phase
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    artifacts = {}
    counters = dict(score_requests=0, continuation_requests=0, requests=0, generated_tokens=0)
    with VllmDoraRollout(base_model=c['base_model'], checkpoint=checkpoint, identity=identity,
            log_path=directory/'vllm.log', device=0, trainer_rank=0, max_model_len=4456,
            max_num_seqs=1, max_logprobs=-1, kv_cache_memory_bytes=2*1024**3, seed=92711, timeout=1800) as engine:
        engine.configure_coordinate_output_norm('off', c['coordinate_ids'], identity=identity)
        for context in c['contexts']:
            assert time.monotonic()-started < 1800, 'phase bound'
            image = context['image_id']
            acquired = engine.generate_exact([requests[image]], chat_token_ids=[next(r['unexpanded_chat_token_ids'] for r in reports if r['image_id']==image)],
                extensions=[context['extension']], budgets=[1], eos_token_id=c['eos_token_id'],
                pad_token_id=q.tokenizer.pad_token_id, identity=identity, vocab_size=c['vocab_size'], full_scores=True)[0]
            assert acquired['request_id'] == inputs[image]['request_id']
            value = dict(context=context, acquisition=acquired, measurement=score_measure(c,context,acquired,identity))
            filename = f'score-{len(artifacts):02d}.json'
            b.write(directory/filename, value); artifacts[filename] = b.sha(directory/filename)
            counters['score_requests'] += 1; counters['requests'] += 1; counters['generated_tokens'] += 1
        for image in c['images']:
            assert time.monotonic()-started < 1800, 'phase bound'
            acquired = engine.generate([requests[image]], budgets=[3084], eos_token_id=c['eos_token_id'],
                pad_token_id=q.tokenizer.pad_token_id, identity=identity)[0]
            evidence = dict(request_id=acquired.request_id, token_ids=list(acquired.token_ids),
                stop_reason=acquired.stop_reason, weight_identity=identity)
            value = dict(acquisition=evidence, **natural_measure(c,inputs[image],evidence,q.tokenizer,labels[image],identity))
            filename = f'natural-{image}.json'
            b.write(directory/filename,value); artifacts[filename] = b.sha(directory/filename)
            counters['continuation_requests'] += 1; counters['requests'] += 1
            counters['generated_tokens'] += len(acquired.token_ids)
        startup, operations = engine.startup, list(engine.receipts)
        active_seconds = time.monotonic()-started
    assert not engine._process.is_alive(), 'native child survived cleanup'
    assert counters['score_requests']==10 and counters['continuation_requests']==18 and counters['generated_tokens']<=55522
    b.write(directory/'complete.json',dict(phase=phase, arm=arm, contract_sha256=expected_sha,
        checkpoint=str(checkpoint), weight_identity=identity, artifacts=artifacts, counters=counters,
        startup=startup, operations=operations, engine_closed_before_publish=True,
        owned_child_absent=True, active_seconds=active_seconds, resources=resources(started),
        artifact_bytes=sum((directory/name).stat().st_size for name in artifacts),
        native_cuda_memory='unmeasured_child_engine', native_internal_startup_capture_forwards=None))


def replay(q, batch, context):
    import torch
    from src.qwen.native import exact_history_inputs
    kwargs = exact_history_inputs(q.model, batch.inputs, [context['processed_prompt_token_ids']],
        pad_token_id=q.tokenizer.pad_token_id, logits_to_keep=1)
    assert kwargs['use_cache'] is False and kwargs['logits_to_keep'] == 1
    assert kwargs['input_ids'].shape[-1]-1 == context['causal_logits_position']
    device = kwargs['input_ids'].device.type
    with torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda' else nullcontext():
        logits = q.model(**kwargs).logits
    assert logits.shape[:2] == (1,1)
    z = logits[0,-1].float()
    assert torch.isfinite(z).all(), 'nonfinite HF logits'
    return z


def hf_measure(z, context):
    import torch
    mask = torch.ones(z.numel(), dtype=torch.bool, device=z.device)
    mask[context['support']] = False
    logp = z.log_softmax(0)
    return dict(context_id=context['context_id'], input_prefix_identity=context['input_prefix_identity'],
        margin=float((z[context['support']].max()-z[mask].max()).detach()),
        legal_mass=float(logp[context['support']].exp().sum().detach()),
        logits_identity=p.tensor_hash(z.detach()), causal_logits_position=context['causal_logits_position'])


def live_training_binding(q, delta, c):
    import torch
    params = {n:t for n,t in q.model.named_parameters() if t.requires_grad}
    deltas = list(delta.delta_tensors().values())
    delta_ids = {id(t) for t in deltas}
    dora = [t for t in params.values() if id(t) not in delta_ids]
    assert len(params)==590 and len(dora)==588 and len(deltas)==2
    assert all(t.dtype==torch.float32 for t in params.values())
    assert all('lora_' in n or id(t) in delta_ids for n,t in params.items())
    assert not any('visual' in n or 'projector' in n for n in params)
    assert all(list(t.shape)==[1004,2048] for t in deltas)
    assert q.model.config.text_config.attention_dropout == 0
    assert all(getattr(module,'p',0)==0 for name,module in q.model.named_modules() if 'lora_dropout' in name)
    assert not q.model.is_gradient_checkpointing
    assert q.model.config._attn_implementation == 'flash_attention_2'
    assert all(t.dtype==torch.bfloat16 for n,t in q.model.named_parameters() if not t.requires_grad and t.is_floating_point())
    return list(params.values()), dora, deltas


def train_updates(q, delta, batches, c, arm, directory, started):
    import torch
    params, dora, deltas = live_training_binding(q,delta,c)
    optimizer = torch.optim.AdamW([dict(params=dora,lr=1e-5),dict(params=deltas,lr=5e-6)],
        betas=(.9,.999),eps=1e-8,weight_decay=0)
    assert not optimizer.state
    q.model.train()
    schedule = views(c,arm)
    updates = []
    for step in range(1,17):
        assert time.monotonic()-started < 1800, 'phase bound'
        optimizer.zero_grad(set_to_none=True)
        terms = []
        before = {n:p.tensor_hash(t.detach()) for n,t in q.model.named_parameters() if t.requires_grad}
        for context in schedule:
            z = replay(q,batches[context['image_id']],context)
            assert z.numel()==c['vocab_size'], 'HF full vocabulary required'
            loss = b.online.max_geometry_margin(z, context['support'])
            assert torch.isfinite(loss)
            (loss/6).backward()
            terms.append(dict(weight=1/6, loss=float(loss.detach()), **hf_measure(z,context)))
        norms = {n:float(t.grad.float().norm()) if t.grad is not None else None
                 for n,t in q.model.named_parameters() if t.requires_grad}
        assert all(v is not None and math.isfinite(v) for v in norms.values()), 'missing/nonfinite gradient'
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True))
        assert [g['lr'] for g in optimizer.param_groups] == full_label_learning_rates(None,step)
        optimizer.step()
        assert all(torch.isfinite(t).all() for t in params), 'nonfinite parameter'
        after = {n:p.tensor_hash(t.detach()) for n,t in q.model.named_parameters() if t.requires_grad}
        changed = [n for n in before if before[n] != after[n]]
        value = dict(step=step,arm=arm,terms=terms,mean_loss=sum(t['loss'] for t in terms)/6,
            global_clip_count=1,optimizer_step_count=1,gradient_norm=gradient_norm,gradient_norms=norms,
            lrs=[g['lr'] for g in optimizer.param_groups],changed_parameters=changed,
            before_parameter_identity=b.online.identity(before),after_parameter_identity=b.online.identity(after))
        b.write(directory/f'update-{step:02d}.json',value); updates.append(value)
    return updates


def train(path, output, expected_sha, arm):
    import torch
    c = entry(path,output,expected_sha,f'train-{arm}')
    directory = output/f'train-{arm}'; directory.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    torch.manual_seed(92711); torch.cuda.manual_seed_all(92711)
    q, delta, composition = p.compose(Path(c['checkpoint']),evaluation=False)
    b.write(directory/'composition.json',composition)
    b.online.set_checkpointing(q.model,False)
    live_training_binding(q,delta,c)
    p.save_checkpoint(q,delta,directory/'checkpoint-0')
    b.online.verify_start_export(Path(c['checkpoint']),directory/'checkpoint-0')
    inputs = {i['image_id']:i for i in b.load(c['input_path'])}
    batches = {i:b.online.native_batch(q,inputs[i]) for i in [7511,351017]}
    q.model.eval()
    anchor = []
    if arm=='R-single':
        with torch.no_grad():
            anchor = [hf_measure(replay(q,batches[x['image_id']],x),x) for x in c['contexts']]
        b.write(directory/'hf-anchor.json',anchor)
    updates = train_updates(q,delta,batches,c,arm,directory,started)
    p.save_checkpoint(q,delta,directory/'checkpoint-16')
    q.model.eval()
    with torch.no_grad():
        endpoint = [hf_measure(replay(q,batches[x['image_id']],x),x) for x in c['contexts']]
    b.write(directory/'hf-endpoint.json',endpoint)
    checkpoint = directory/'checkpoint-16'
    files = b.load(checkpoint/'identity.json')
    artifacts = {f'update-{step:02d}.json':b.sha(directory/f'update-{step:02d}.json') for step in range(1,17)}
    artifacts['hf-endpoint.json'] = b.sha(directory/'hf-endpoint.json')
    artifacts['composition.json'] = b.sha(directory/'composition.json')
    if anchor: artifacts['hf-anchor.json'] = b.sha(directory/'hf-anchor.json')
    stats = resources(started,gpu=True)
    del q, delta, batches
    torch.cuda.empty_cache()
    assert stats['seconds'] < 1800
    b.write(directory/'complete.json',dict(phase=f'train-{arm}',arm=arm,contract_sha256=expected_sha,
        anchor_checkpoint=c['checkpoint'],anchor_weight_identity=WEIGHT,independent_anchor_reload=True,
        fresh_optimizer=True,checkpoint=str(checkpoint),checkpoint_files=files,weight_identity=b.online.identity(files),
        artifacts=artifacts,counters=dict(optimizer_steps=16,training_replays=96,HF_diagnostics=len(anchor)+len(endpoint)),
        resources=stats,active_seconds=stats['seconds'],HF_closed_before_publish=True,
        artifact_bytes=sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())))


def contrast(c, scores):
    result = {}
    for arm in ARMS:
        groups = {}
        for site,*_ in SITES:
            for split in ['original','training_neighbor','held_out']:
                indices = [i for i,x in enumerate(c['contexts']) if x['site_id']==site and x['split']==split]
                illegal = [i for i in indices if not scores['anchor'][i]['literal_legal']]
                legal = [i for i in indices if scores['anchor'][i]['literal_legal']]
                groups[site+'/'+split] = dict(baseline_illegal_denominator=len(illegal),
                    repaired=[c['contexts'][i]['context_id'] for i in illegal if scores[arm][i]['literal_legal']],
                    baseline_legal_denominator=len(legal),
                    retained=[c['contexts'][i]['context_id'] for i in legal if scores[arm][i]['literal_legal']],
                    lost=[c['contexts'][i]['context_id'] for i in legal if not scores[arm][i]['literal_legal']],
                    margin_changes=[dict(context_id=c['contexts'][i]['context_id'],delta=scores[arm][i]['margin']-scores['anchor'][i]['margin']) for i in indices])
        result[arm] = groups
    result['original_error_contrast_present'] = any(not scores['anchor'][i]['literal_legal'] for i,x in enumerate(c['contexts']) if x['offset']==0)
    return result


def consume_native(c, output, arm, expected_sha, q, inputs):
    directory = output/f'native-{arm}'
    receipt = b.load(directory/'complete.json')
    checkpoint, identity = weight_binding(c,output,arm)
    assert receipt['phase']==f'native-{arm}' and receipt['arm']==arm and receipt['contract_sha256']==expected_sha
    assert receipt['checkpoint']==str(checkpoint) and receipt['weight_identity']==identity
    assert receipt['engine_closed_before_publish'] is True
    assert receipt['owned_child_absent'] is True
    assert receipt['native_internal_startup_capture_forwards'] is None
    expected = [f'score-{i:02d}.json' for i in range(10)] + [f'natural-{i}.json' for i in c['images']]
    assert set(receipt['artifacts']) == set(expected)
    for name, sha in receipt['artifacts'].items(): assert b.sha(directory/name)==sha
    scores, natural = [], []
    labels = {i['image_id']:i for i in b.load(c['label_path'])}
    for i, context in enumerate(c['contexts']):
        value = b.load(directory/f'score-{i:02d}.json')
        assert value['context']==context and value['acquisition']['request_id']==inputs[context['image_id']]['request_id']
        measured = score_measure(c,context,value['acquisition'],identity)
        assert value['measurement']==measured, 'false conditional credit/score'
        scores.append(measured)
    for image in c['images']:
        value = b.load(directory/f'natural-{image}.json')
        measured = natural_measure(c,inputs[image],value['acquisition'],q.tokenizer,labels[image],identity)
        assert value==dict(acquisition=value['acquisition'],**measured), 'false natural credit'
        natural.append(measured['measurement'])
    counters = dict(score_requests=10,continuation_requests=18,requests=28,
        generated_tokens=10+sum(x['burdens']['generated_tokens'] for x in natural))
    assert receipt['counters']==counters and counters['generated_tokens']<=55522
    assert receipt['active_seconds']<=1800 and receipt['resources']['seconds']<=1830
    from src.qwen.vllm_rollout import validate_device_receipt
    validate_device_receipt(receipt['startup']['device'],receipt['startup']['device']['requested'])
    operations=receipt['operations']
    assert [x['operation'] for x in operations]==['coordinate_output_norm']+['generate_exact']*10+['generate']*18
    assert receipt['startup']['identity']==identity
    for op in operations:
        assert op['identity']==identity
        norm=op['coordinate_output_norm']
        assert norm['mode']=='off' and norm['identity']==identity and norm['coordinate_ids']==c['coordinate_ids']
        if op['operation'] != 'coordinate_output_norm':
            assert norm['calls']>0 and norm['first_call']['scaling_active'] is False and norm['first_call']['non_coordinate_unchanged'] is True
    return scores,natural,counters


def consume_train(c, output, arm, expected_sha):
    directory = output/f'train-{arm}'
    receipt = b.load(directory/'complete.json')
    assert receipt['phase']==f'train-{arm}' and receipt['arm']==arm and receipt['contract_sha256']==expected_sha
    assert receipt['anchor_checkpoint']==c['checkpoint'] and receipt['anchor_weight_identity']==WEIGHT
    assert receipt['independent_anchor_reload'] is True and receipt['fresh_optimizer'] is True and receipt['HF_closed_before_publish'] is True
    weight_binding(c,output,arm)
    b.online.verify_start_export(Path(c['checkpoint']),directory/'checkpoint-0')
    expected = [f'update-{i:02d}.json' for i in range(1,17)] + ['hf-endpoint.json','composition.json']
    if arm=='R-single': expected += ['hf-anchor.json']
    assert set(receipt['artifacts'])==set(expected)
    for name, sha in receipt['artifacts'].items(): assert b.sha(directory/name)==sha
    schedule = views(c,arm)
    for step in range(1,17):
        value = b.load(directory/f'update-{step:02d}.json')
        assert value['step']==step and value['arm']==arm
        assert value['global_clip_count']==value['optimizer_step_count']==1
        assert value['lrs']==[1e-5,5e-6] and len(value['terms'])==6
        for term,context in zip(value['terms'],schedule,strict=True):
            assert term['context_id']==context['context_id'] and term['input_prefix_identity']==context['input_prefix_identity']
            assert term['weight']==1/6 and term['causal_logits_position']==context['causal_logits_position']
            assert math.isfinite(term['loss']) and math.isfinite(term['margin'])
            assert math.isclose(term['loss'], math.log1p(math.exp(1-term['margin'])) if 1-term['margin']<700 else 1-term['margin'],rel_tol=2e-6,abs_tol=2e-6), 'false Gmax term'
        assert math.isclose(value['mean_loss'],sum(t['loss'] for t in value['terms'])/6,rel_tol=1e-8)
        assert math.isfinite(value['gradient_norm']) and all(v is not None and math.isfinite(v) for v in value['gradient_norms'].values())
    diagnostics = {}
    for name in ['hf-endpoint.json'] + (['hf-anchor.json'] if arm=='R-single' else []):
        data = b.load(directory/name)
        assert len(data)==10
        for value, context in zip(data,c['contexts'],strict=True):
            assert value['context_id']==context['context_id'] and value['input_prefix_identity']==context['input_prefix_identity']
            assert math.isfinite(value['margin']) and 0<=value['legal_mass']<=1.00001
        diagnostics[name]=data
    counters=dict(optimizer_steps=16,training_replays=96,HF_diagnostics=20 if arm=='R-single' else 10)
    assert receipt['counters']==counters and receipt['resources']['seconds']<=1800
    return diagnostics,counters


def readback(path, output, expected_sha):
    c = entry(path,output,expected_sha,PHASES[-1])
    q, inputs, _, _ = frontend(c)
    scores, natural, counters, diagnostics = {}, {}, {k:0 for k in ['optimizer_steps','training_replays','HF_diagnostics','score_requests','continuation_requests','requests','generated_tokens']}, {}
    for arm in ['anchor']+ARMS:
        scores[arm],natural[arm],counts=consume_native(c,output,arm,expected_sha,q,inputs)
        for k,v in counts.items(): counters[k]+=v
    for arm in ARMS:
        diagnostics[arm],counts=consume_train(c,output,arm,expected_sha)
        for k,v in counts.items(): counters[k]+=v
    assert all(counters[k]==BOUNDS[k] for k in counters if k!='generated_tokens') and counters['generated_tokens']<=BOUNDS['generated_tokens']
    transitions = {}
    for arm in ARMS:
        rows = []
        for before,after in zip(natural['anchor'],natural[arm],strict=True):
            assert before['image_id']==after['image_id']
            owners = b.transitions({m:before['ids'][m]['retained'] for m in ['raw','category']},
                                   {m:after['ids'][m]['retained'] for m in ['raw','category']})
            rows.append(dict(image_id=before['image_id'],owners=owners,
                burden_deltas={k:after['burdens'][k]-before['burdens'][k] for k in before['burdens']}))
        transitions[arm]=dict(images=rows,owner_counts={m:{k:sum(len(r['owners'][m][k]) for r in rows) for k in ['gained','lost','retained']} for m in ['raw','category']},
            burden_deltas={k:sum(r['burden_deltas'][k] for r in rows) for k in rows[0]['burden_deltas']})
    active_seconds=sum(b.load(output/phase/'complete.json')['active_seconds'] for phase in PHASES)
    assert active_seconds<=9000
    result=dict(status='candidate',contract_sha256=expected_sha,source=c['source'],phases=PHASES,
        counters=counters,active_seconds=active_seconds,scores=scores,conditional=contrast(c,scores),
        HF_diagnostics=diagnostics,natural=natural,natural_transitions=transitions,
        annotation_unmatched_is_physical_negative=False,conditional_natural_recovery_credit=False,
        scientific_acceptance=False,next_unit_scheduled=False)
    # Readback recomputes all decision fields; re-signing claimed results cannot admit false credit.
    terminal=output/'complete.json'
    if terminal.exists(): assert b.load(terminal)==result, 're-signed false terminal'
    else: b.write(terminal,result)
    readback_value=dict(status='candidate',terminal_sha256=b.sha(terminal),counters=counters)
    if (output/'readback.json').exists(): assert b.load(output/'readback.json')==readback_value
    else: b.write(output/'readback.json',readback_value)
    return result


def package(path, output, expected_sha):
    c=entry(path,output,expected_sha,PHASES[0])
    output.mkdir(parents=True,exist_ok=False)
    commands=phase_commands(path,output)
    commands=[[(expected_sha if x=='EXACT_RELEASE_SHA256' else x) for x in command] for command in commands]
    outcomes=[]
    for phase,command in zip(PHASES,commands,strict=True):
        with (output/f'{phase}.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=process.wait(timeout=1830)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL); process.wait()
                code=124
        outcomes.append(dict(phase=phase,command=command,exit=code))
        b.write(output/f'package-status-{len(outcomes)}.json',dict(status='partial' if code else 'unreviewed',phases=outcomes))
        if code: raise RuntimeError(f'{phase} failed ({code}); preserve partial evidence, no replacement')
    readback(path,output,expected_sha)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','qualify','validate','native','train','package','readback'])
    parser.add_argument('--contract',type=Path)
    parser.add_argument('--contract-sha256')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=['anchor']+ARMS)
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.output)
    elif args.command=='qualify': qualify(args.contract,args.output)
    elif args.command=='validate': validate(args.contract)
    elif args.command=='native': native(args.contract,args.output,args.contract_sha256,args.arm)
    elif args.command=='train': train(args.contract,args.output,args.contract_sha256,args.arm)
    elif args.command=='package': package(args.contract,args.output,args.contract_sha256)
    else: readback(args.contract,args.output,args.contract_sha256)


if __name__=='__main__':
    main()
