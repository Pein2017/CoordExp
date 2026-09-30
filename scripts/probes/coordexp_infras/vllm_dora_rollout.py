#!/usr/bin/env python3
"""Bounded, technical HF/vLLM DoRA rollout and refresh smoke; no efficacy claim."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def snapshot(model):
    from probes.iterative_positive import tensor_hash

    tensors = {name: tensor_hash(value) for name, value in model.named_parameters() if value.requires_grad}
    return hashlib.sha256(json.dumps(tensors, sort_keys=True).encode()).hexdigest(), len(tensors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-ids', type=int, nargs='+', default=[1584, 2299])
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--eager', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--learning-step', action='store_true')
    args = parser.parse_args()
    if not 1 <= len(args.image_ids) <= 2 or len(set(args.image_ids)) != len(args.image_ids):
        parser.error('choose one or two distinct image IDs')
    if not 1 <= args.max_new_tokens <= 128:
        parser.error('max-new-tokens must be in 1..128 for this smoke')
    if not args.checkpoint.is_dir():
        parser.error('checkpoint directory does not exist')
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if world not in (1, 2) or (world == 2 and (len(args.image_ids) != 2 or not args.learning_step)):
        parser.error('distributed smoke requires two ranks, two images and --learning-step')
    if world == 2:
        args.output = args.output / f'rank-{rank}'
        args.image_ids = args.image_ids[rank::world]
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    from importlib.metadata import version
    from probes import iterative_positive as p
    from probes import online_row_credit as o
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import NativeRequest, prepare_replay
    from src.qwen.vllm_rollout import VllmDoraRollout, validate_device_assignments
    from src.artifacts.git_identity import capture_source_identity, verify_source_identity

    sources = [str(path.relative_to(REPO)) for path in (REPO/'src').rglob('*.py')]
    sources += ['probes/online_row_credit.py', 'probes/rollout_row_credit.py',
                'probes/iterative_positive.py', 'probes/hidden_human_recovery.py',
                'scripts/probes/coordexp_infras/vllm_dora_rollout.py']
    source_identity = capture_source_identity(sources, root=REPO)

    source = {row['image_id']: row for row in p.load(o.INPUTS)}
    if not set(args.image_ids) <= source.keys():
        raise ValueError('requested image ID is absent from frozen inputs')
    items = [source[i] for i in args.image_ids]
    if any(row['crop'] != [0, 0, row['width'], row['height']] for row in items):
        raise ValueError('this smoke only supports whole-image inputs')
    policy = p.load(p.POLICY)
    torch.cuda.set_device(local_rank)
    if world > 1:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel
        dist.init_process_group('nccl')
    started = time.monotonic()
    q, delta, composition = p.compose(args.checkpoint, evaluation=False)
    train_model = (DistributedDataParallel(q.model, device_ids=[local_rank], broadcast_buffers=False)
                   if world > 1 else q.model)
    hf_startup = time.monotonic() - started
    batches = [o.native_batch(q, row) for row in items]
    messages = [{'role': 'system', 'content': policy['prompt']['system']},
                {'role': 'user', 'content': [{'type': 'image'},
                                            {'type': 'text', 'text': policy['prompt']['user']}]}]
    chat = q.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    requests = [NativeRequest(row['request_id'], chat, row['image_path'],
                              expected_token_ids=tuple(row['prompt_token_ids']),
                              expected_image_grid=tuple(row['image_grid_thw']),
                              expected_image_size=(row['width'], row['height']),
                              image_sha256=row['image_sha256']) for row in items]
    initial, count = snapshot(q.model)
    receipt = dict(checkpoint=str(args.checkpoint.resolve()), input_path=str(o.INPUTS),
                   world_size=world, rank=rank, local_rank=local_rank,
                   source=source_identity,
                   versions={name: version(name) for name in ('torch','transformers','peft','vllm','flash-attn')},
                   image_ids=args.image_ids, max_new_tokens=args.max_new_tokens,
                   composition=composition, initial_snapshot=initial,
                   trainable_tensors=count, hf_startup_seconds=hf_startup,
                   policy=dict(temperature=0, top_p=1, top_k=0,
                               repetition_penalty=1, use_model_defaults=False),
                   engine_options=dict(enforce_eager=args.eager, max_model_len=16000,
                                       max_num_seqs=2, kv_cache_memory_bytes=2*1024**3,
                                       gpu_memory_utilization=0.2),
                   inputs=[dict(image_id=row['image_id'], request_id=row['request_id'],
                                prompt_tokens=len(row['prompt_token_ids']),
                                image_grid_thw=row['image_grid_thw'], media_sha256=row['media_sha256'],
                                image_sha256=row['image_sha256']) for row in items])
    eos = q.tokenizer.convert_tokens_to_ids('<|im_end|>')
    generation_policy = NativeGenerationPolicy(temperature=0, top_p=1, top_k=0,
                                               repetition_penalty=1, use_model_defaults=False)

    def hf_generate(indices):
        q.model.eval()
        torch.cuda.synchronize()
        began = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            results = [generate_continuations(q.model, batches[i], extensions=[()],
                       budgets=[args.max_new_tokens], eos_token_id=eos,
                       pad_token_id=q.tokenizer.pad_token_id, policy=generation_policy,
                       trace='none', seed=None)[0] for i in indices]
        torch.cuda.synchronize()
        return results, time.monotonic() - began

    try:
        began = time.monotonic()
        with VllmDoraRollout(base_model=q.base_model_path, checkpoint=args.checkpoint,
                             identity=initial, log_path=args.output/'vllm-worker.log',
                             device=local_rank, trainer_rank=rank,
                             max_num_seqs=2, enforce_eager=args.eager) as engine:
            receipt['vllm_startup'] = dict(**engine.startup,
                                           parent_seconds=time.monotonic()-began)
            local_device = dict(rank=rank, request=engine.device_request, startup=engine.startup)
            devices = [None]*world
            if world > 1:
                dist.all_gather_object(devices, local_device)
            else:
                devices[0] = local_device
            validate_device_assignments(devices, [local_rank] if world == 1 else list(range(world)))
            receipt['vllm_devices'] = devices
            hf_warm, hf_warm_seconds = hf_generate([0])
            receipt['hf_warmup'] = dict(seconds=hf_warm_seconds,
                                        token_counts=[len(x.token_ids) for x in hf_warm])
            hf, hf_seconds = hf_generate(range(len(items)))
            receipt['hf_measured'] = dict(seconds=hf_seconds,
                                          token_counts=[len(x.token_ids) for x in hf],
                                          token_ids=[list(x.token_ids) for x in hf],
                                          stop_reasons=[x.stop_reason for x in hf])
            call = dict(budgets=[args.max_new_tokens]*len(items), eos_token_id=eos,
                        pad_token_id=q.tokenizer.pad_token_id, identity=initial)
            vllm_warm = engine.generate(requests[:1], **dict(call, budgets=call['budgets'][:1]))
            receipt['vllm_warmup'] = dict(rpc=engine.receipts[-1],
                                          token_counts=[len(x.token_ids) for x in vllm_warm])
            began = time.monotonic()
            generated = engine.generate(requests, trace=True, **call)
            receipt['vllm_measured'] = dict(parent_seconds=time.monotonic()-began,
                                           rpc=engine.receipts[-1],
                                           token_counts=[len(x.token_ids) for x in generated],
                                           token_ids=[list(x.token_ids) for x in generated],
                                           raw_logprobs=[list(x.raw_logprobs) for x in generated],
                                           stop_reasons=[x.stop_reason for x in generated])
            receipt['prefix_agreement'] = [dict(image_id=row['image_id'],
                shared_tokens=next((j for j, (a, b) in enumerate(zip(h.token_ids, v.token_ids))
                                    if a != b), min(len(h.token_ids), len(v.token_ids))),
                hf_tokens=len(h.token_ids), vllm_tokens=len(v.token_ids))
                for row, h, v in zip(items, hf, generated, strict=True)]

            replay = []
            for row, batch, result in zip(items, batches, generated, strict=True):
                chosen = result.token_ids[:32]
                if not chosen:
                    replay.append(dict(image_id=row['image_id'], tokens=0))
                    continue
                prepared = prepare_replay(q.model, batch.inputs,
                    prompt_token_ids=row['prompt_token_ids'], continuation_token_ids=chosen)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = prepared.aligned_logits(q.model(**prepared.inputs).logits)
                chosen_hf = logits.float().log_softmax(-1).gather(
                    -1, prepared.target_ids[:, None]).squeeze(-1).tolist()
                chosen_vllm = result.raw_logprobs[:len(chosen)]
                differences = [a-b for a, b in zip(chosen_hf, chosen_vllm, strict=True)]
                replay.append(dict(image_id=row['image_id'], tokens=len(chosen),
                    hf_logprobs=chosen_hf, vllm_logprobs=chosen_vllm,
                    max_abs_difference=max(map(abs, differences)),
                    mean_abs_difference=sum(map(abs, differences))/len(differences)))
            receipt['teacher_forced_first32'] = replay

            if args.learning_step:
                q.model.train()
                params = [value for value in q.model.parameters() if value.requires_grad]
                original_params = [value.detach().cpu().clone() for value in params]
                delta_ids = {id(value) for value in delta.delta_tensors().values()}
                optimizer = torch.optim.AdamW([
                    dict(params=[value for value in params if id(value) not in delta_ids], lr=1e-5),
                    dict(params=list(delta.delta_tensors().values()), lr=5e-6)],
                    betas=(.9, .999), eps=1e-8, weight_decay=0)
                optimizer.zero_grad(set_to_none=True)
                began = time.monotonic()
                losses = []
                for row, batch, result in zip(items, batches, generated, strict=True):
                    chosen = result.token_ids[:32]
                    if not chosen:
                        continue
                    prepared = prepare_replay(q.model, batch.inputs,
                        prompt_token_ids=row['prompt_token_ids'], continuation_token_ids=chosen)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits = prepared.aligned_logits(train_model(**prepared.inputs).logits)
                        loss = torch.nn.functional.cross_entropy(logits.float(), prepared.target_ids)
                    (loss/len(items)).backward()
                    losses.append(float(loss.detach()))
                if not losses:
                    raise RuntimeError('no emitted suffix available for the learning smoke')
                norms = [float(value.grad.float().norm()) for value in params if value.grad is not None]
                if len(norms) != len(params) or not all(math.isfinite(x) for x in norms) or not any(x > 0 for x in norms):
                    raise RuntimeError('learning smoke has missing, nonfinite or zero gradients')
                total_norm = float(torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True))
                optimizer.step()
                torch.cuda.synchronize()
                updated, _ = snapshot(q.model)
                if updated == initial:
                    raise RuntimeError('AdamW did not change the trainable snapshot')
                updated_params = [value.detach().cpu().clone() for value in params]
                if world > 1:
                    snapshots = [None] * world
                    dist.all_gather_object(snapshots, updated)
                    if len(set(snapshots)) != 1:
                        raise RuntimeError('DDP ranks diverged after the learning step')
                    receipt['synchronized_snapshots'] = snapshots
                receipt['learning_step'] = dict(seconds=time.monotonic()-began,
                    forwards=len(losses), losses=losses, gradient_nonzero=sum(x > 0 for x in norms),
                    gradient_count=len(norms), total_norm=total_norm, updated_snapshot=updated)
                began = time.monotonic()
                engine.refresh(q.model, delta, identity=updated)
                receipt['updated_refresh'] = dict(parent_seconds=time.monotonic()-began,
                                                  rpc=engine.receipts[-1])
                began = time.monotonic()
                after = engine.generate(requests, trace=True, **dict(call, identity=updated))
                receipt['post_refresh'] = dict(parent_seconds=time.monotonic()-began,
                    rpc=engine.receipts[-1], token_counts=[len(x.token_ids) for x in after],
                    token_ids=[list(x.token_ids) for x in after],
                    stop_reasons=[x.stop_reason for x in after],
                    raw_logprobs=[list(x.raw_logprobs) for x in after])
                changed = [a.token_ids != b.token_ids or a.raw_logprobs != b.raw_logprobs
                           for a, b in zip(generated, after, strict=True)]
                receipt['post_refresh']['changed_from_initial'] = changed

                try:
                    with torch.no_grad():
                        for value, original in zip(params, original_params, strict=True):
                            value.copy_(original)
                    if snapshot(q.model)[0] != initial:
                        raise RuntimeError('original HF snapshot was not restored for vLLM check')
                    began = time.monotonic()
                    engine.refresh(q.model, delta, identity=initial)
                    receipt['original_refresh'] = dict(parent_seconds=time.monotonic()-began,
                                                       rpc=engine.receipts[-1])
                    restored = engine.generate(requests, trace=True, **call)
                    exact = [a.token_ids == b.token_ids and a.raw_logprobs == b.raw_logprobs
                             for a, b in zip(generated, restored, strict=True)]
                    receipt['restore_check'] = dict(exact=exact,
                        token_ids=[list(x.token_ids) for x in restored],
                        raw_logprobs=[list(x.raw_logprobs) for x in restored],
                        rpc=engine.receipts[-1])
                    if not all(exact):
                        raise RuntimeError('restored vLLM generation differs from initial tokens or raw logprobs')
                finally:
                    with torch.no_grad():
                        for value, trained in zip(params, updated_params, strict=True):
                            value.copy_(trained)
                    if snapshot(q.model)[0] != updated:
                        raise RuntimeError('trained HF snapshot was not preserved after restore check')
                if not any(changed):
                    raise RuntimeError('updated vLLM weights did not change emitted tokens or raw logprobs')
            receipt['rpc_receipts'] = engine.receipts
        verify_source_identity(source_identity, required_paths=sources, root=REPO)
        receipt['status'] = 'complete'
    except BaseException as exc:
        receipt['status'] = 'failed'
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True,
            default=str)+'\n')
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
