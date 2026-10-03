#!/usr/bin/env python3
"""Bounded technical HF/vLLM DoRA acquisition, replay, update and refresh smoke."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

QUALIFICATION_IMAGES = (1584, 2299, 2685)
QUALIFICATION_BUDGETS = (64, 64, 32)
ANCHOR = Path('/data/CoordExp/outputs/shared/checkpoints/untied-axis001-step2444/payload')
PHASES = ('hf_v0_greedy', 'hf_v0_sample', 'vllm_v0_greedy', 'vllm_v0_sample',
          'vllm_v1_greedy', 'vllm_v1_sample', 'vllm_restore_greedy')


def snapshot(model):
    from probes.iterative_positive import tensor_hash
    tensors = {name: tensor_hash(value) for name, value in model.named_parameters() if value.requires_grad}
    return hashlib.sha256(json.dumps(tensors, sort_keys=True).encode()).hexdigest(), len(tensors)


def _write(path, receipt):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + '\n')
    temporary.replace(path)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-ids', type=int, nargs='+')
    parser.add_argument('--max-new-tokens', type=int)
    parser.add_argument('--eager', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--learning-step', action='store_true')
    parser.add_argument('--qualification', action='store_true',
                        help='bounded two-rank median-policy greedy/sample/update/restore qualification')
    return parser


def _native_runtime(args, items, rank, local_rank, world, receipt):
    import torch
    import torch.distributed as dist
    from importlib.metadata import version
    from probes import iterative_positive as p, online_row_credit as o
    from src.artifacts.git_identity import capture_source_identity, verify_source_identity
    from src.qwen.coordinate_policy import MedianPolicy
    from src.qwen.native import NativeRequest
    from src.qwen.vllm_rollout import VllmDoraRollout, validate_device_assignments
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group('nccl')
    sources = sorted(set(o.source_paths() + [
        'scripts/probes/coordexp_infras/vllm_dora_rollout.py',
        'probes/rule_stability/data.py', 'probes/rule_stability/runner.py',
        'probes/full_label_fit/experiment.py', 'probes/full_label_fit/rollout.py']))
    source = capture_source_identity(sources, root=REPO)
    began = time.monotonic()
    q, delta, composition = p.compose(args.checkpoint, evaluation=False)
    receipt.update(source=source, composition=composition, hf_startup_seconds=time.monotonic()-began,
                   versions={name: version(name) for name in ('torch', 'transformers', 'peft', 'vllm', 'flash-attn')})
    batches = [o.native_batch(q, row) for row in items]
    prompt = p.load(p.POLICY)['prompt']
    messages = [{'role': 'system', 'content': prompt['system']}, {'role': 'user', 'content':
                [{'type': 'image'}, {'type': 'text', 'text': prompt['user']}]}]
    chat = q.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    requests = [NativeRequest(row['request_id'], chat, row['image_path'],
        expected_token_ids=tuple(row['prompt_token_ids']), expected_image_grid=tuple(row['image_grid_thw']),
        expected_image_size=(row['width'], row['height']), image_sha256=row['image_sha256']) for row in items]
    coordinate_ids = tuple(q.tokenizer.convert_tokens_to_ids(f'<|coord_{i}|>') for i in range(1000))
    return SimpleNamespace(q=q, delta=delta, norm=MedianPolicy(q.model, coordinate_ids) if args.qualification else None,
        coordinate_ids=coordinate_ids, batches=batches, requests=requests, dist=dist,
        sync=torch.cuda.synchronize, autocast=lambda: torch.autocast('cuda', dtype=torch.bfloat16),
        engine_factory=VllmDoraRollout, validate_devices=validate_device_assignments,
        verify_source=lambda: verify_source_identity(source, required_paths=sources, root=REPO),
        peak_memory=lambda: dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved()))


def _load_records(args):
    from probes import iterative_positive as p, online_row_credit as o
    if args.qualification:
        from probes.rule_stability.data import load_inputs, request_records, FULL_LABEL_PATH, INPUT_MANIFEST_PATH
        images, manifest = load_inputs()
        records = request_records(images, manifest)
        input_paths = [str(FULL_LABEL_PATH), str(INPUT_MANIFEST_PATH)]
    else:
        records, input_paths = p.load(o.INPUTS), [str(o.INPUTS)]
    by_id = {row['image_id']: row for row in records}
    if len(by_id) != len(records) or not set(args.image_ids) <= by_id.keys():
        raise ValueError('requested image ownership is missing or duplicated')
    chosen = [by_id[i] for i in args.image_ids]
    if any(row['crop'] != [0, 0, row['width'], row['height']] for row in chosen):
        raise ValueError('this smoke only supports whole-image inputs')
    return chosen, input_paths


def _gather(runtime, value, world):
    rows = [None] * world
    if world > 1:
        runtime.dist.all_gather_object(rows, value)
    else:
        rows[0] = value
    return rows


def _policy(sample):
    from src.qwen.generation import NativeGenerationPolicy
    return NativeGenerationPolicy(temperature=1 if sample else 0, top_p=1, top_k=0,
                                  repetition_penalty=1, use_model_defaults=False)


def _acquire(runtime, engine, items, budgets, *, backend, version, sample, identity, receipt):
    import torch
    from probes.rule_stability.runner import seed_for
    from src.qwen.generation import generate_continuations
    q = runtime.q
    seeds = [seed_for(version, row['image_id']) for row in items] if sample else [None] * len(items)
    runtime.sync()
    began = time.monotonic()
    policy = _policy(sample)
    if backend == 'hf':
        q.model.eval()
        with torch.inference_mode(), runtime.autocast():
            results = tuple(generate_continuations(q.model, batch, extensions=[()], budgets=[budget],
                eos_token_id=q.tokenizer.convert_tokens_to_ids('<|im_end|>'), pad_token_id=q.tokenizer.pad_token_id,
                policy=policy, trace='raw_and_policy', seed=seed, allow_pad_tokens=True,
                logits_processor=[runtime.norm.generation_transform()] if runtime.norm else None)[0]
                for batch, budget, seed in zip(runtime.batches, budgets, seeds, strict=True))
        rpc = None
    else:
        results = engine.generate(runtime.requests, budgets=budgets,
            eos_token_id=q.tokenizer.convert_tokens_to_ids('<|im_end|>'), pad_token_id=q.tokenizer.pad_token_id,
            identity=identity, policy=policy, seeds=seeds if sample else None,
            trace=True, allow_pad_tokens=True)
        rpc = engine.receipts[-1]
        if rpc.get('identity') != identity:
            raise ValueError('generation acknowledgement has a stale snapshot')
    runtime.sync()
    # Publish partial evidence before checking it, so a failed channel retains its source.
    rows = [dict(image_id=row['image_id'], request_id=result.request_id, budget=budget,
        seed=seed, snapshot_id=identity, token_ids=list(result.token_ids), stop_reason=result.stop_reason,
        eos_token_id=q.tokenizer.convert_tokens_to_ids('<|im_end|>'),
        raw_logprobs=None if result.raw_logprobs is None else list(result.raw_logprobs),
        policy_logprobs=None if result.policy_logprobs is None else list(result.policy_logprobs),
        trace_token_ids=None if result.trace is None else list(result.trace.token_ids))
        for row, result, budget, seed in zip(items, results, budgets, seeds, strict=True)]
    receipt.update(parent_seconds=time.monotonic()-began, rows=rows, rpc=rpc)
    _validate_rows(rows, items, budgets, identity)
    if rpc is not None and runtime.norm:
        _validate_rpc(rpc, rows, identity)
    return results


def _validate_rows(rows, items, budgets, identity):
    if len(rows) != len(items):
        raise ValueError('generation omitted request ownership')
    for row, item, budget in zip(rows, items, budgets, strict=True):
        ids = row['token_ids']
        if row['request_id'] != item['request_id'] or row['image_id'] != item['image_id']:
            raise ValueError('generation reordered request ownership')
        if row['snapshot_id'] != identity or row['budget'] != budget:
            raise ValueError('generation has a stale snapshot or budget')
        if (not ids or len(ids) > budget or any(type(t) is not int or t < 0 for t in ids)
                or row['stop_reason'] not in ('im_end', 'length')):
            raise ValueError('invalid generation actions or stop reason')
        if row['trace_token_ids'] != ids:
            raise ValueError('trace actions do not align with emitted tokens')
        eos = row['eos_token_id']
        if (row['stop_reason'] == 'im_end' and (ids[-1] != eos or eos in ids[:-1])) \
                or (row['stop_reason'] == 'length' and (len(ids) != budget or eos in ids)):
            raise ValueError('stop reason differs from literal action evidence')
        for name in ('raw_logprobs', 'policy_logprobs'):
            values = row[name]
            if values is None or len(values) != len(ids) or not all(math.isfinite(v) for v in values):
                raise ValueError(f'missing, misaligned or nonfinite {name}')


def _validate_rpc(rpc, rows, identity):
    norm, trace = rpc.get('coordinate_output_norm', {}), rpc.get('paired_trace', {})
    if norm.get('mode') != 'median' or norm.get('identity') != identity or trace.get('snapshot_id') != identity:
        raise ValueError('median/trace acknowledgement has a stale snapshot')
    evidence = trace.get('requests', [])
    if [r.get('request_id') for r in evidence] != [r['request_id'] for r in rows]:
        raise ValueError('paired trace receipt omitted or reordered request ownership')
    for row, observed in zip(rows, evidence, strict=True):
        if observed.get('emitted_actions') != len(row['token_ids']):
            raise ValueError('paired trace receipt action count differs')
        for name in ('excluded_async_suffix', 'discarded_prefill_actions', 'dropped_budget_actions'):
            if type(observed.get(name)) is not int or observed[name] < 0:
                raise ValueError('paired trace receipt lacks exclusion accounting')


def _replay(runtime, item, batch, result, denominator, learning):
    import torch
    from src.qwen.native import exact_history_inputs, prompt_only_placeholder_masks
    from probes.rule_stability.policy import replay_difference
    chosen = result.token_ids[:32]
    if not chosen:
        raise ValueError('no literal sampled action available for replay')
    prompt = item['prompt_token_ids']
    inputs = exact_history_inputs(runtime.q.model, batch.inputs, [(*prompt, *chosen)],
        pad_token_id=runtime.q.tokenizer.pad_token_id, prompt_only_media=True)
    positions = list(range(len(prompt)-1, len(prompt)+len(chosen)-1))
    device = inputs['input_ids'].device
    inputs['logits_to_keep'] = torch.tensor(positions, device=device)
    targets = torch.tensor(chosen, device=device)
    runtime.sync()
    began = time.monotonic()
    with (nullcontext() if learning else torch.inference_mode()), runtime.autocast(), \
            prompt_only_placeholder_masks(runtime.q.model, prompt):
        logits = runtime.q.model(**inputs).logits
        if logits.shape[:2] != (1, len(chosen)):
            raise ValueError('replay logits do not cover the exact causal action positions')
        raw = logits[0].float().log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
        policy_logits = runtime.norm.transform_replay(logits)[0] if runtime.norm else logits[0].float()
        selected = policy_logits.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
        loss = -selected.mean()
    runtime.sync()
    forward_seconds = time.monotonic()-began
    comparison = replay_difference(request_id=item['request_id'], token_ids=chosen,
        behavior_raw_logprobs=result.raw_logprobs[:len(chosen)],
        behavior_policy_logprobs=result.policy_logprobs[:len(chosen)],
        replay_raw_logprobs=raw, replay_policy_logprobs=selected)
    began = time.monotonic()
    if learning:
        (loss / denominator).backward()
        runtime.sync()
    comparison.update(image_id=item['image_id'], causal_positions=positions, loss=float(loss.detach()),
        scaled_loss=float(loss.detach())/denominator, backward_scale=1/denominator,
        hf_forward_seconds=forward_seconds, hf_backward_seconds=time.monotonic()-began if learning else 0,
        forwards=1, backwards=int(learning))
    return comparison


def _refresh(runtime, engine, identity, receipt):
    began = time.monotonic()
    engine.refresh(runtime.q.model, runtime.delta, identity=identity)
    rpc = engine.receipts[-1]
    receipt.update(parent_seconds=time.monotonic()-began, rpc=rpc)
    if rpc.get('identity') != identity or engine.identity != identity:
        raise ValueError('refresh acknowledgement has a stale snapshot')
    if runtime.norm:
        norm = rpc.get('coordinate_output_norm', {})
        if norm.get('mode') != 'median' or norm.get('identity') != identity:
            raise ValueError('refresh did not acknowledge current median factors')


def _copy(parameters, values):
    import torch
    with torch.no_grad():
        for parameter, value in zip(parameters, values, strict=True):
            parameter.copy_(value)


def _run(args, runtime, items, budgets, receipt, world, rank, local_rank):
    import torch
    from probes.rule_stability.runner import synchronize_gradients
    initial, count = snapshot(runtime.q.model)
    receipt.update(initial_snapshot=initial, trainable_tensors=count, acquisitions={}, replay=[])
    engine, updated_values, parameters, updated = None, None, None, None
    began = time.monotonic()
    try:
        engine = runtime.engine_factory(base_model=runtime.q.base_model_path, checkpoint=args.checkpoint,
            identity=initial, log_path=args.output/'vllm-worker.log', device=local_rank, trainer_rank=rank,
            max_num_seqs=2, enforce_eager=args.eager)
        receipt['vllm_startup'] = dict(**engine.startup, parent_seconds=time.monotonic()-began,
            requested_warmups=0, internal_profiling='included in startup, excluded from acquisition counters')
        devices = _gather(runtime, dict(rank=rank, request=engine.device_request, startup=engine.startup), world)
        runtime.validate_devices(devices, list(range(world)) if world > 1 else [local_rank])
        receipt['vllm_devices'] = devices
        if runtime.norm:
            norm = engine.configure_coordinate_output_norm('median', runtime.coordinate_ids, identity=initial)
            receipt['coordinate_output_norm'] = norm
            if norm.get('identity') != initial or norm.get('mode') != 'median':
                raise ValueError('initial median policy was not acknowledged')
        result_sets = {}
        for backend in ('hf', 'vllm'):
            for sample in ((False, True) if args.qualification else (False,)):
                phase = f'{backend}_v0_' + ('sample' if sample else 'greedy')
                stage = receipt['acquisitions'][phase] = {}
                result_sets[phase] = _acquire(runtime, engine, items, budgets, backend=backend,
                    version=0, sample=sample, identity=initial, receipt=stage)
        hf, resident = result_sets['hf_v0_greedy'], result_sets['vllm_v0_greedy']
        receipt['prefix_agreement'] = [dict(image_id=row['image_id'], shared_tokens=next(
            (j for j, (a, b) in enumerate(zip(h.token_ids, v.token_ids)) if a != b),
            min(len(h.token_ids), len(v.token_ids))), hf_tokens=len(h.token_ids), vllm_tokens=len(v.token_ids))
            for row, h, v in zip(items, hf, resident, strict=True)]
        chosen_results = result_sets['vllm_v0_sample'] if args.qualification else resident
        parameters = [value for value in runtime.q.model.parameters() if value.requires_grad]
        if args.learning_step:
            began = time.monotonic()
            original_values = [value.detach().cpu().clone() for value in parameters]
            receipt['original_snapshot_materialization_seconds'] = time.monotonic()-began
            delta_ids = {id(value) for value in runtime.delta.delta_tensors().values()}
            optimizer = torch.optim.AdamW([
                dict(params=[value for value in parameters if id(value) not in delta_ids], lr=1e-5),
                dict(params=list(runtime.delta.delta_tensors().values()), lr=5e-6)],
                betas=(.9, .999), eps=1e-8, weight_decay=0)
            optimizer.zero_grad(set_to_none=True)
            runtime.q.model.train()
        for item, batch, result in zip(items, runtime.batches, chosen_results, strict=True):
            receipt['replay'].append(_replay(runtime, item, batch, result, len(args.image_ids), args.learning_step))
        if args.learning_step:
            began = time.monotonic()
            if world > 1:
                synchronize_gradients(parameters)
            norms = [float(value.grad.float().norm()) for value in parameters if value.grad is not None]
            if len(norms) != len(parameters) or not all(math.isfinite(x) for x in norms) or not any(x > 0 for x in norms):
                raise ValueError('missing, nonfinite or zero global gradients')
            total_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True))
            optimizer.step()
            runtime.sync()
            receipt['learning_step'] = dict(gradient_reduction='SUM', backward_scale=1/len(args.image_ids),
                sum_calls=int(world > 1), optimizer_steps=1, clip_norm=1, total_norm=total_norm,
                gradient_count=len(norms), gradient_nonzero=sum(x > 0 for x in norms),
                sum_clip_step_seconds=time.monotonic()-began)
            began = time.monotonic()
            updated, _ = snapshot(runtime.q.model)
            updated_values = [value.detach().cpu().clone() for value in parameters]
            receipt['updated_snapshot_materialization_seconds'] = time.monotonic()-began
            snapshots = _gather(runtime, updated, world)
            receipt['synchronized_snapshots'] = snapshots
            if updated == initial or len(set(snapshots)) != 1:
                raise ValueError('optimizer did not update one identical snapshot across ranks')
            receipt['learning_step']['updated_snapshot'] = updated
            _refresh(runtime, engine, updated, receipt.setdefault('updated_refresh', {}))
            for sample in ((False, True) if args.qualification else (False,)):
                phase = 'vllm_v1_' + ('sample' if sample else 'greedy')
                _acquire(runtime, engine, items, budgets, backend='vllm', version=1, sample=sample,
                    identity=updated, receipt=receipt['acquisitions'].setdefault(phase, {}))
            began = time.monotonic()
            _copy(parameters, original_values)
            if snapshot(runtime.q.model)[0] != initial:
                raise ValueError('original HF trainable bytes were not restored')
            receipt['restore_original_hf_seconds'] = time.monotonic()-began
            _refresh(runtime, engine, initial, receipt.setdefault('original_refresh', {}))
            restore_budgets = [16 if args.qualification else budget for budget in budgets]
            restored = _acquire(runtime, engine, items, restore_budgets, backend='vllm', version=0,
                sample=False, identity=initial, receipt=receipt['acquisitions'].setdefault('vllm_restore_greedy', {}))
            exact = [dict(image_id=item['image_id'], tokens=min(budget, len(original.token_ids)),
                token_ids=list(original.token_ids[:budget]) == list(check.token_ids),
                raw_logprobs=list(original.raw_logprobs[:budget]) == list(check.raw_logprobs),
                policy_logprobs=list(original.policy_logprobs[:budget]) == list(check.policy_logprobs))
                for item, original, check, budget in zip(items, resident, restored, restore_budgets, strict=True)]
            receipt['restore_check'] = exact
            if not all(all(row[k] for k in ('token_ids', 'raw_logprobs', 'policy_logprobs')) for row in exact):
                raise ValueError('restored greedy prefix differs in actions or either likelihood channel')
        runtime.verify_source()
    finally:
        try:
            if updated_values is not None:
                began = time.monotonic()
                _copy(parameters, updated_values)
                receipt['restore_updated_hf_seconds'] = time.monotonic()-began
                receipt['final_hf_snapshot'] = snapshot(runtime.q.model)[0]
        finally:
            if engine is not None:
                try:
                    engine.close()
                finally:
                    receipt['rpc_receipts'] = engine.receipts
                    receipt['child_settlement'] = getattr(engine, 'shutdown', None)
            receipt['peak_memory'] = runtime.peak_memory()
            receipt['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if updated is not None and receipt['final_hf_snapshot'] != updated:
        raise ValueError('updated HF trainable bytes were not preserved after restore')


def validate_qualification(receipts):
    """Final consumer: exact workload/ownership and evidence, no numerical parity gate."""
    if [r.get('rank') for r in receipts] != [0, 1]:
        raise ValueError('qualification rank ownership is missing or reordered')
    snapshots, originals, distinct = [], [], False
    requests, actions, forwards, backwards, targets, refreshes = 0, 0, 0, 0, 0, 0
    for rank, receipt in enumerate(receipts):
        expected = list(QUALIFICATION_IMAGES[rank::2])
        if receipt.get('image_ids') != expected or receipt.get('world_size') != 2 or receipt.get('status') != 'complete':
            raise ValueError('qualification image ownership or terminal state differs')
        if list(receipt.get('phase_order', [])) != list(PHASES) or set(receipt['acquisitions']) != set(PHASES):
            raise ValueError('qualification acquisition sequence differs')
        initial, updated = receipt['initial_snapshot'], receipt['learning_step']['updated_snapshot']
        snapshots.append(updated)
        originals.append(initial)
        coordinate = set(receipt['coordinate_ids'])
        local_items = receipt['inputs']
        local_budgets = [QUALIFICATION_BUDGETS[QUALIFICATION_IMAGES.index(i)] for i in expected]
        for phase in PHASES:
            stage = receipt['acquisitions'][phase]
            identity = updated if '_v1_' in phase else initial
            budgets = [16]*len(expected) if phase == 'vllm_restore_greedy' else local_budgets
            _validate_rows(stage['rows'], local_items, budgets, identity)
            if phase.startswith('vllm'):
                _validate_rpc(stage['rpc'], stage['rows'], identity)
            for row in stage['rows']:
                requests += 1
                actions += len(row['token_ids'])
                if phase.startswith('vllm'):
                    distinct |= any(token not in coordinate and raw != policy for token, raw, policy in
                        zip(row['token_ids'], row['raw_logprobs'], row['policy_logprobs'], strict=True))
        replays = receipt['replay']
        if [r['image_id'] for r in replays] != expected:
            raise ValueError('replay image ownership differs')
        for replay, behavior in zip(replays, receipt['acquisitions']['vllm_v0_sample']['rows'], strict=True):
            if replay['token_ids'] != behavior['token_ids'][:32] or replay['backward_scale'] != 1/3 \
                    or not math.isclose(replay['scaled_loss'], replay['loss']/3, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('replay literal actions or image-mean backward scale differs')
            selected = replay['policy']['replay_logprobs']
            if not selected or not all(math.isfinite(x) for x in selected) \
                    or not math.isclose(replay['loss'], -sum(selected)/len(selected), rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError('replay loss differs from selected policy likelihoods')
            forwards += replay['forwards']; backwards += replay['backwards']; targets += replay['actions']
        step = receipt['learning_step']
        if step['gradient_reduction'] != 'SUM' or step['sum_calls'] != 1 or step['optimizer_steps'] != 1 \
                or step['backward_scale'] != 1/3 or step['clip_norm'] != 1:
            raise ValueError('logical SUM/clip/update contract differs')
        if receipt['synchronized_snapshots'] != [updated, updated] or receipt['final_hf_snapshot'] != updated:
            raise ValueError('updated trainable snapshots differ')
        for key, identity in (('updated_refresh', updated), ('original_refresh', initial)):
            if receipt[key]['rpc']['identity'] != identity:
                raise ValueError('refresh snapshot sequence differs')
            refreshes += 1
        if [row['image_id'] for row in receipt['restore_check']] != expected or not all(
                all(row[k] for k in ('token_ids', 'raw_logprobs', 'policy_logprobs')) for row in receipt['restore_check']):
            raise ValueError('restore prefix evidence differs')
        settled = receipt['child_settlement']
        if not settled or not settled.get('settled') or settled.get('exitcode') != 0 or settled.get('terminated'):
            raise ValueError('resident child did not settle normally')
    if len(set(snapshots)) != 1 or len(set(originals)) != 1 or not distinct:
        raise ValueError('snapshot disagreement or no observed non-coordinate raw/policy difference')
    if requests != 21 or actions > 1008 or forwards != 3 or backwards != 3 or targets > 96 or refreshes != 4:
        raise ValueError('qualification exceeded or omitted bounded work')
    return dict(generation_requests=requests, realized_actions=actions, replay_forwards=forwards,
                replay_backwards=backwards, replay_target_actions=targets, logical_updates=1, refreshes=refreshes,
                noncoordinate_channel_difference_observed=distinct)


def main(argv=None, *, runtime_factory=None):
    parser = _parser()
    args = parser.parse_args(argv)
    world, rank, local_rank = (int(os.environ.get(key, default)) for key, default in
                              (('WORLD_SIZE', '1'), ('RANK', '0'), ('LOCAL_RANK', '0')))
    if args.qualification:
        if world != 2 or args.eager or (args.image_ids is not None and args.image_ids != list(QUALIFICATION_IMAGES)) \
                or args.max_new_tokens is not None or args.checkpoint.resolve() != ANCHOR:
            parser.error('qualification requires two ranks, current shared step2444 anchor, decode graphs and fixed budgets/images')
        args.image_ids, budgets, args.learning_step = list(QUALIFICATION_IMAGES), list(QUALIFICATION_BUDGETS), True
    else:
        args.image_ids = args.image_ids or [1584, 2299]
        args.max_new_tokens = 128 if args.max_new_tokens is None else args.max_new_tokens
        if not 1 <= len(args.image_ids) <= 2 or len(set(args.image_ids)) != len(args.image_ids) \
                or not 1 <= args.max_new_tokens <= 128 or world not in (1, 2) \
                or (world == 2 and (len(args.image_ids) != 2 or not args.learning_step)):
            parser.error('choose one/two distinct images and 1..128 tokens; two ranks require --learning-step')
        budgets = [args.max_new_tokens]*len(args.image_ids)
    if not args.checkpoint.is_dir() or not 0 <= rank < world or not 0 <= local_rank < world:
        parser.error('checkpoint or rank assignment is invalid')
    if args.qualification and not args.output.resolve().is_relative_to(REPO/'outputs'):
        parser.error('qualification output must belong to this worktree outputs')
    root = args.output.resolve()
    if args.qualification and (root/'receipt.json').exists():
        parser.error('qualification output already has a terminal receipt')
    args.output = root/f'rank-{rank}' if world > 1 else root
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(checkpoint=str(args.checkpoint.resolve()), world_size=world, rank=rank, local_rank=local_rank,
        pid=os.getpid(), output=str(args.output), source_root=str(REPO), argv=list(argv or sys.argv[1:]),
        status='running', actual_exit_status=None, qualification=args.qualification)
    runtime, summary, began = None, None, time.monotonic()
    try:
        records, input_paths = _load_records(args)
        items = records[rank::world]
        local_budgets = budgets[rank::world]
        receipt.update(image_ids=[r['image_id'] for r in items], input_paths=input_paths,
            inputs=[{key: row[key] for key in ('image_id', 'request_id', 'prompt_token_ids', 'image_grid_thw',
                     'media_sha256', 'image_sha256', 'image_path')} for row in items], budgets=local_budgets,
            policy=dict(temperature=[0, 1] if args.qualification else [0], top_p=1, top_k=0,
                        repetition_penalty=1, use_model_defaults=False, coordinate_norm='median' if args.qualification else 'off'),
            engine_options=dict(enforce_eager=args.eager, tensor_parallel_size=1, max_num_seqs=2,
                max_model_len=16000, kv_cache_memory_bytes=2*1024**3, gpu_memory_utilization=.2,
                cudagraph_mode='FULL_DECODE_ONLY' if not args.eager else None))
        runtime = (runtime_factory or _native_runtime)(args, items, rank, local_rank, world, receipt)
        receipt['coordinate_ids'] = list(runtime.coordinate_ids)
        _run(args, runtime, items, local_budgets, receipt, world, rank, local_rank)
        receipt.update(status='complete', phase_order=list(receipt['acquisitions']), actual_exit_status=0,
                       wall_seconds=time.monotonic()-began)
        _write(args.output/'receipt.json', receipt)
        if args.qualification:
            receipts = _gather(runtime, receipt, world)
            summary = validate_qualification(receipts)
            summary.update(status='complete', rank_receipts=[str(root/f'rank-{i}/receipt.json') for i in range(world)],
                rank_wall_seconds=[r['wall_seconds'] for r in receipts])
            summary['rank_skew_seconds'] = max(summary['rank_wall_seconds'])-min(summary['rank_wall_seconds'])
            summary['acquisition_costs'] = {phase: dict(
                critical_path_seconds=max(r['acquisitions'][phase]['parent_seconds'] for r in receipts),
                rank_seconds=[r['acquisitions'][phase]['parent_seconds'] for r in receipts],
                realized_actions=sum(len(row['token_ids']) for r in receipts for row in r['acquisitions'][phase]['rows']))
                for phase in PHASES}
            summary['backend_acquisition_totals'] = {backend: dict(
                critical_path_seconds=max(sum(r['acquisitions'][phase]['parent_seconds'] for phase in PHASES
                    if phase.startswith(backend) and 'restore' not in phase) for r in receipts),
                realized_actions=sum(len(row['token_ids']) for r in receipts for phase in PHASES
                    if phase.startswith(backend) and 'restore' not in phase for row in r['acquisitions'][phase]['rows']))
                for backend in ('hf', 'vllm')}
    except BaseException as exc:
        receipt.update(status='failed', actual_exit_status=1, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        receipt['wall_seconds'] = time.monotonic()-began
        try:
            if world > 1:
                if runtime is not None:
                    runtime.dist.destroy_process_group()
                    receipt['distributed_settlement'] = 'destroyed'
                elif 'torch.distributed' in sys.modules and sys.modules['torch.distributed'].is_initialized():
                    sys.modules['torch.distributed'].destroy_process_group()
                    receipt['distributed_settlement'] = 'destroyed after setup failure'
        except BaseException as exc:
            receipt.update(status='failed', actual_exit_status=1, cleanup_error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            _write(args.output/'receipt.json', receipt)
    if summary is not None and rank == 0:
        _write(root/'receipt.json', summary)


if __name__ == '__main__':
    main()
