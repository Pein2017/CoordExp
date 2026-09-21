"""One fixed Source-started wide31 preservation arm; original two losses unchanged."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time

from src.artifacts import load_canonical_json
from .candidate_opportunity import digest, file_hash, indexed, require, rows
from .route_access import CONFIG, ROOT, checked_ids, publish
from .branch_bridge import summarize_logits
from .entrance_ce import OPTIMIZER
from .selective_preservation import (
    OUTPUT as FIRST_ARM, selective_loss, validate_receipt as validate_local_receipt,
)

AUTONOMOUS = ROOT / '2026-09-10-selective-owner-learning-autonomous'
OUTPUT = AUTONOMOUS / 'soft-preservation-wide31/training'
PREFLIGHT = AUTONOMOUS / 'support32-preflight.json'
SUPPORT_COUNT, SUPPORT_WEIGHT, UPDATES = 31, 10., 23


def support_loss(logits, target_ids, action_ids, reference_logp):
    """Full-state reference KL, mean within image then weight10/31; no CE labels."""
    import torch
    checked_ids(action_ids, 'im_end')
    require(logits.ndim == 2 and logits.shape[0] == len(action_ids) and target_ids.tolist() == action_ids,
            'support causal target alignment')
    require(reference_logp.shape == logits.shape and logits.dtype == reference_logp.dtype == torch.float32,
            'support reference shape/precision')
    ref = reference_logp.detach()
    mean_kl = (ref.exp() * (ref - torch.log_softmax(logits, -1))).sum(-1).mean()
    return (SUPPORT_WEIGHT / SUPPORT_COUNT) * mean_kl, mean_kl


def work_items(packet):
    result = []
    for case in packet['cases']:
        t = packet['trajectories'][case['case_id']]
        result.append(dict(key=case['case_id'], kind='entrance', case=case,
                           action_ids=t['action_ids'], positions=t['preservation_positions'], suffix_start=t['suffix_start']))
    for case in packet['support_cases']:
        result.append(dict(key=case['example_id'], kind='support', case=case,
                           action_ids=case['action_ids'], positions=list(range(len(case['action_ids'])))))
    require(len(result) == 33 and len({w['key'] for w in result}) == 33, 'all33 unique losses required')
    return result


def image_objective(logits, targets, item, reference_logp):
    if item['kind'] == 'entrance':
        loss, ce, kl = selective_loss(logits, targets, item['case']['entrance'], item['action_ids'],
                item['suffix_start'], item['positions'], reference_logp)
        return loss, ce, kl
    require(item['kind'] == 'support' and item['positions'] == list(range(len(item['action_ids']))), 'support full-state mask')
    loss, kl = support_loss(logits, targets, item['action_ids'], reference_logp)
    return loss, None, kl


def validate_receipt(receipt):
    require(receipt['schema_version'] == 'selective_preservation_wide.training.v1' and
            receipt['lambda_kl'] == 10 and receipt['lambda_support_kl'] == 10 and
            len(receipt['support_image_ids']) == 31 and len(set(receipt['support_image_ids'])) == 31 and
            '417044' not in receipt['support_image_ids'], 'wide31 checkpoint objective/support identity')
    # The standard file/Source/entry and fixed23 checks have the same semantics.
    validate_local_receipt({**receipt, 'schema_version': 'selective_preservation.training.v1'})
    require(receipt['resources']['reference_cache_bytes'] == 1064415240 and
            receipt['resources']['actual_model_forwards'] == 840, 'wide31 checkpoint execution completeness')


def prepare(output):
    require(not output.exists(), 'occupied training root')
    preflight = load_canonical_json(PREFLIGHT)
    old = load_canonical_json(FIRST_ARM / 'inputs.json')
    all_selected = preflight['original_selected32']
    require(len(all_selected) == 32 and [c['rank'] for c in all_selected] == list(range(1, 33)), 'original rank32 audit trail')
    keep = [c for c in all_selected if c['image_id'] != '417044']
    excluded = [c for c in all_selected if c['image_id'] == '417044']
    require(len(excluded) == 1 and excluded[0]['rank'] == 8 and excluded[0]['stop_reason'] == 'length', 'explicit sole exclusion')
    require(len(keep) == 31 and [c['image_id'] for c in keep] == preflight['proposed_support31']['retained_image_ids'], 'no-backfill support31 identity')
    for path, sha in {**old['source_files'], **preflight['source_files']}.items():
        require(file_hash(path) == sha, f'frozen input changed: {path}')
    source_plan = load_canonical_json(ROOT / '2026-09-10-fixed-witness-route-access/inputs.json')['plan']
    groups = indexed(source_plan['population']['groups'], 'example_id')
    dev_ids = {str(int(Path(r['image_path']).stem)) for r in rows(Path(preflight['dev_root']) / 'gt_vs_pred.jsonl')}
    require(not ({c['image_id'] for c in keep} & (dev_ids | {'368', '7116'})), 'support/dev/target overlap')
    support = []
    for card in keep:
        ids, group = checked_ids(card['action_ids'], 'im_end'), groups[card['example_id']]
        require(card['stop_reason'] == 'im_end' and card['parser_drops'] == 0 and card['parse_status'] == 'accepted' and
                len(ids) == card['action_length'] and digest(ids) == card['action_ids_sha256'], 'admitted support action identity')
        require(file_hash(card['image_path']) == card['image_content_sha256'] == group['image_content_sha256'] and
                card['executed_media_sha256'] == group['executed_media_sha256'] and
                card['prompt_ids_sha256'] == digest(group['prompt_token_ids']), 'support prompt/media identity')
        support.append(dict(example_id=card['example_id'], image_id=card['image_id'], original_rank=card['rank'],
                selection_sha256=card['selection_sha256'], action_ids=ids, prompt_token_ids=group['prompt_token_ids'], group=group))
    require(sum(len(c['action_ids']) for c in support) == 1568 and max(len(c['action_ids']) for c in support) == 155 and
            max(len(c['prompt_token_ids']) + len(c['action_ids']) for c in support) == 1472, 'support length bounds')
    packet = dict(schema_version='selective_preservation_wide.inputs.v1', cases=old['cases'], trajectories=old['trajectories'],
            support_cases=support, support_preflight_sha256=file_hash(PREFLIGHT), support_image_ids=[c['image_id'] for c in support],
            explicit_exclusion=excluded[0], no_backfill=True, model=old['model'], train_source=old['train_source'],
            qualifications=old['qualifications'], optimizer=OPTIMIZER, clip_gradient_norm=1., updates=UPDATES,
            lambda_kl=10., lambda_support_kl=SUPPORT_WEIGHT,
            source_files={**old['source_files'], **preflight['source_files'], str(PREFLIGHT): file_hash(PREFLIGHT),
                          str(FIRST_ARM / 'inputs.json'): file_hash(FIRST_ARM / 'inputs.json')},
            objective='Original mean2CE+10mean2KL unchanged, plus10mean31(mean_state full-vocabulary KL(Source||student)); no support CE, one33-loss accumulation/clip/step.',
            bounds=dict(reference_forwards=33, train_forwards=759, entry_score_forwards=48, actual_model_forwards=840,
                        reference_cache_bytes=1064415240, model_loads=1, model_seconds=3600))
    work_items(packet)
    output.mkdir(parents=True, exist_ok=False)
    publish(output / 'inputs.json', packet)
    files = [Path(__file__), Path(__file__).with_name('tests') / 'test_selective_preservation_wide.py', CONFIG,
             Path(__file__).with_name('train.py'), Path(__file__).with_name('runtime.py'),
             Path(__file__).with_name('entrance_ce.py'), Path(__file__).with_name('selective_preservation.py'),
             Path(__file__).with_name('branch_bridge.py')]
    files += list(Path('src/qwen').glob('*.py')) + [Path('src/losses/token_scores.py'), Path('src/adapters/dora.py')]
    records = []
    for path in files:
        target = output / 'effective_code' / str(path.resolve()).lstrip('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        records.append(dict(path=str(path.resolve()), staged=str(target), sha256=file_hash(target)))
    publish(output / 'code_identity.json', dict(files=records,
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            git_status=subprocess.check_output(['git', 'status', '--short'], text=True)))
    return packet


def execute(output):
    import numpy as np
    import torch
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_research_infer_config
    from src.data import load_raw_examples
    from src.inference.runtime import assemble_frontend
    from src.qwen.native import prepare_replay
    from src.adapters.dora import select_dora_parameters
    from .runtime import load_policy
    from .train import (EXPECTED_TRAINABLE_TENSORS, EXPECTED_TRAINABLE_SCALARS,
                        _materialize_group, _parameter_layout, _tensor_state_hash, _save_adapter_only)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'GPU0 only')
    packet = load_canonical_json(output / 'inputs.json')
    require(not (output / 'launch.json').exists() and not (output / 'adapter').exists(), 'occupied execution')
    for path, sha in packet['source_files'].items():
        require(file_hash(path) == sha, f'frozen source changed: {path}')
    config = load_research_infer_config(CONFIG).config
    require(str(config.model.base_model) == packet['model']['base_model_path'] and
            str(config.adapter.path) == packet['model']['current_adapter']['root'] and
            str(config.embedding_delta.path) == packet['model']['source_embedding']['root'], 'original Source identity')
    require(config.model.dtype == 'fp32' and config.backend.hf.attn_implementation == 'sdpa' and
            config.backend.hf.patch_embed_linearization == 'enabled', 'frozen numerics')
    require(file_hash(config.data.input_jsonl) == packet['train_source']['sha256'], 'dataset identity')
    for identity in (packet['model']['current_adapter'], packet['model']['source_embedding']):
        for f in identity['files']:
            require(file_hash(Path(identity['root']) / f['relative_path']) == f['sha256'], 'Source payload identity')
    frontend = assemble_frontend(config, generation_config_fingerprint=sha256_json(config.generation.model_dump(mode='json')))
    raw = {str(r.example_id): r for r in load_raw_examples(config.data.input_jsonl)}
    publish(output / 'launch.json', dict(pid=os.getpid(), time=time.time(), visible_devices='0',
            processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader'], text=True)))
    started = time.monotonic()
    counters = dict(model_loads=0, reference_forwards=0, train_forwards=0, entry_score_forwards=0,
                    actual_model_forwards=0, updates=0, supervised_target_tokens=0, support_KL_trajectories=0)
    status, error = 'failed', None
    def expired(*_):
        raise TimeoutError('3600-second invocation ceiling')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(3600)
    try:
        qwen, identity = load_policy(config, device=torch.device('cuda:0'))
        counters['model_loads'] = 1
        model = qwen.model
        model.eval()
        require(identity['effective_settings']['observed_model_dtype']['parameter_dtype_names'] == ['torch.float32'] and
                identity['effective_settings']['observed_attn_implementation'] == 'sdpa' and
                identity['model_identity']['adapter']['merged_adapters'] == [], 'loaded execution identity')
        publish(output / 'loaded_model.json', identity)
        publish(output / 'effective_config.json', config.model_dump(mode='json'))
        def count_forward(*_):
            counters['actual_model_forwards'] += 1
            require(counters['actual_model_forwards'] <= 840, 'actual model forward cap')
        model.register_forward_pre_hook(count_forward)
        for p in model.parameters():
            p.requires_grad_(False)
        named = select_dora_parameters(model, towers=('language',), adapter_name='default')
        require(len(named) == EXPECTED_TRAINABLE_TENSORS == 588 and sum(p.numel() for _, p in named) == EXPECTED_TRAINABLE_SCALARS == 18006016 and
                all('language_model' in n and not any(x in n for x in ('visual', 'merger', 'embed_tokens', 'lm_head')) for n, _ in named), 'selected training surface')
        for _, p in named:
            p.requires_grad_(True)
        selected = {id(p) for _, p in named}
        frozen = [(n, p) for n, p in model.named_parameters() if id(p) not in selected]
        versions = [(p, p._version) for _, p in frozen]
        frozen_hash, source_hash = _tensor_state_hash(frozen), _tensor_state_hash(named)
        original = [p.detach().clone() for _, p in named]
        layout = _parameter_layout(named)
        publish(output / 'trainable_layout.json', layout)
        optimizer = torch.optim.AdamW([p for _, p in named], **OPTIMIZER)
        require(not optimizer.state, 'optimizer must be fresh')
        materialized = []
        for item in work_items(packet):
            case = item['case']
            prompt, inputs, grid = _materialize_group(qwen=qwen, frontend=frontend, config=config,
                    raw=raw[case['example_id']], group=case['group'])
            materialized.append((item, prompt, {**inputs, 'image_grid_thw': grid}))
        torch.cuda.reset_peak_memory_stats()
        def forward(item, prompt, inputs, *, isolated=False):
            case = item['case']
            ids = case['state_ids'] + [case['target_token_id']] if isolated else item['action_ids']
            replay = prepare_replay(model, inputs, prompt_token_ids=prompt, continuation_token_ids=ids)
            require(replay.target_ids.tolist() == ids, 'native target identity')
            return replay.aligned_logits(model(**replay.inputs).logits), replay.target_ids
        def save_array(path, values):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                np.save(stream, values, allow_pickle=False)
            require(np.array_equal(values, np.load(path, allow_pickle=False)), 'array publication/reload')
        def entry_scores(step):
            scores = {}
            for item, prompt, inputs in materialized[:2]:
                case = item['case']
                require(counters['entry_score_forwards'] < 48, 'isolated score budget')
                counters['entry_score_forwards'] += 1
                with torch.inference_mode():
                    logits, targets = forward(item, prompt, inputs, isolated=True)
                    require(logits.shape[0] == case['action_index'] + 1 and int(targets[-1]) == case['target_token_id'], 'isolated entry alignment')
                    values = logits[-1].detach().cpu().numpy().copy()
                    del logits, targets
                score = summarize_logits(values, case['entrance'])
                score['target_id'] = case['target_token_id']
                scores[case['case_id']] = score
                save_array(output / 'scores' / f"step-{step:02d}-{case['image_id']}.npy", values)
            publish(output / 'scores' / f'step-{step:02d}.json', scores)
            return scores
        references, reference_versions, reference_cards = {}, {}, []
        for item, prompt, inputs in materialized:
            counters['reference_forwards'] += 1
            with torch.no_grad():
                logits, targets = forward(item, prompt, inputs)
                logp = torch.log_softmax(logits[item['positions']], -1).detach().clone()
                _, ce, kl = image_objective(logits, targets, item, logp)
                require(abs(float(kl)) <= 1e-6 and not logp.requires_grad and logp.grad_fn is None, 'initial detached-reference KL')
                references[item['key']] = logp
                reference_versions[item['key']] = logp._version
                values = logp.cpu().numpy().copy()
                del logits, targets
            path = output / 'references' / f"{item['case']['image_id']}-logp.npy"
            save_array(path, values)
            reference_cards.append(dict(key=item['key'], kind=item['kind'], shape=list(logp.shape),
                    cache_bytes=logp.numel() * logp.element_size(), sha256=file_hash(path), initial_KL=float(kl),
                    initial_CE=float(ce) if ce is not None else None, detached=True))
        cache_bytes = sum(r['cache_bytes'] for r in reference_cards)
        require(cache_bytes == 1064415240 and counters['reference_forwards'] == 33, 'all33 reference cache bounds')
        publish(output / 'references.json', reference_cards)
        initial_scores = entry_scores(0)
        dose = []
        for step in range(1, UPDATES + 1):
            require(time.monotonic() - started < 3600 and counters['train_forwards'] + 33 <= 759, 'training budget')
            optimizer.zero_grad(set_to_none=True)
            before_step = [p.detach().clone() for _, p in named]
            losses = []
            for item, prompt, inputs in materialized:
                counters['train_forwards'] += 1
                logits, targets = forward(item, prompt, inputs)
                loss, ce, kl = image_objective(logits, targets, item, references[item['key']])
                require(bool(torch.isfinite(loss)) and float(kl.detach()) >= -1e-6, 'nonfinite/negative loss')
                if step == 1:
                    require(abs(float(kl.detach())) <= 1e-6, 'update1 reference/student mismatch')
                loss.backward()
                losses.append(dict(key=item['key'], kind=item['kind'], weighted_loss=float(loss.detach()),
                        CE=float(ce.detach()) if ce is not None else None, mean_KL=float(kl.detach()),
                        preserved_states=len(item['positions'])))
                counters['supervised_target_tokens'] += int(item['kind'] == 'entrance')
                counters['support_KL_trajectories'] += int(item['kind'] == 'support')
                del loss, ce, kl, logits, targets
            require(len(losses) == 33 and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _, p in named), 'all33 finite selected gradients')
            require(all(p.grad is None and not p.requires_grad for _, p in frozen), 'frozen surface gradients')
            raw_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in named], 1., error_if_nonfinite=True, foreach=False))
            require(math.isfinite(raw_norm) and raw_norm > 0, 'zero/nonfinite joint gradient')
            clipped_norm = math.sqrt(sum(float(p.grad.double().square().sum()) for _, p in named))
            require(clipped_norm <= 1.000001, 'global clip bound')
            optimizer.step()
            counters['updates'] = step
            require(all(p._version == v for p, v in versions), 'frozen parameters changed')
            require(all(not r.requires_grad and r.grad_fn is None and r._version == reference_versions[k] for k, r in references.items()), 'reference cache mutated')
            movement = math.sqrt(sum(float((p.detach() - old).double().square().sum()) for (_, p), old in zip(named, before_step)))
            total_movement = math.sqrt(sum(float((p.detach() - old).double().square().sum()) for (_, p), old in zip(named, original)))
            require(movement > 0 and math.isfinite(total_movement), 'zero/nonfinite adapter movement')
            del before_step
            final_scores = entry_scores(step)
            row = dict(update=step, loss=sum(x['weighted_loss'] for x in losses), image_losses=losses,
                    raw_gradient_norm=raw_norm, clipped_gradient_norm=clipped_norm, step_parameter_delta_l2=movement,
                    source_parameter_delta_l2=total_movement, final_scores=final_scores, counters=dict(counters),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), cumulative_seconds=time.monotonic() - started)
            dose.append(row)
            publish(output / f'update-{step:02d}.json', row)
            if step == 1:
                require(_tensor_state_hash(frozen) == frozen_hash, 'update1 frozen bytes changed')
                require(any(final_scores[c]['target_logit'] != initial_scores[c]['target_logit'] for c in final_scores), 'update1 score movement absent')
                publish(output / 'vertical_smoke.json', dict(update=1, all33_initial_KL_near_zero=True,
                        selected_gradients_finite_nonzero=True, frozen_bytes_unchanged=True, reference_cache_bytes=cache_bytes,
                        longest_total_sequence=max(len(pr) + len(it['action_ids']) for it, pr, _ in materialized),
                        parameter_movement_l2=movement, score_publication_reload=True, counters=dict(counters),
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024))
                print('VERTICAL_SMOKE_COMPLETED', flush=True)
            print(json.dumps(dict(update=step, loss=row['loss'], old2_loss=sum(x['weighted_loss'] for x in losses if x['kind']=='entrance'),
                    support31_loss=sum(x['weighted_loss'] for x in losses if x['kind']=='support'),
                    margins={c: s['A_vs_best_other_margin'] for c, s in final_scores.items()}, seconds=time.monotonic() - started)), flush=True)
        require(_tensor_state_hash(frozen) == frozen_hash, 'final frozen bytes changed')
        require({int(s['step']) for s in optimizer.state.values()} == {UPDATES}, 'optimizer step count')
        final_hash = _tensor_state_hash(named)
        adapter = _save_adapter_only(model, source_root=Path(config.adapter.path), output=output / 'adapter')
        require(final_hash != source_hash and adapter['fingerprint'] != packet['model']['current_adapter']['fingerprint'], 'unchanged adapter')
        for identity in (packet['model']['current_adapter'], packet['model']['source_embedding']):
            for f in identity['files']:
                require(file_hash(Path(identity['root']) / f['relative_path']) == f['sha256'], 'Source payload mutated')
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        require(elapsed < 3600 and counters['actual_model_forwards'] == 840 and counters['supervised_target_tokens'] == 46 and counters['support_KL_trajectories'] == 713, 'final execution accounting')
        receipt = dict(schema_version='selective_preservation_wide.training.v1', status='completed', adapter=adapter,
                source_adapter=packet['model']['current_adapter'], source_embedding=packet['model']['source_embedding'],
                config=config.model_dump(mode='json'), cases=packet['cases'], final_scores=final_scores, initial_scores=initial_scores,
                updates=23, stop_reason='fixed_steps', optimizer=OPTIMIZER, clip_gradient_norm=1., lambda_kl=10., lambda_support_kl=10.,
                support_preflight_sha256=packet['support_preflight_sha256'], support_image_ids=packet['support_image_ids'], objective=packet['objective'],
                trainable_layout=layout, adapter_tensor_hash_before=source_hash, adapter_tensor_hash_after=final_hash,
                frozen_tensor_hash_before=frozen_hash, frozen_tensor_hash_after=frozen_hash, dose=dose, references=reference_cards,
                inputs_sha256=file_hash(output / 'inputs.json'), code_identity_sha256=file_hash(output / 'code_identity.json'),
                resources=dict(**counters, cumulative_model_seconds=elapsed, reference_cache_bytes=cache_bytes,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    artifact_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())),
                cold_reload_status='Pending independent evaluator; saved adapter bytes reloaded equal to live materialized tensors.')
        validate_receipt(receipt)
        publish(output / 'receipt.json', receipt)
        status = 'completed'
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        signal.alarm(0)
        publish(output / 'terminal.json', dict(status=status, error=error, counters=counters,
                cumulative_model_seconds=time.monotonic() - started,
                receipt_sha256=file_hash(output / 'receipt.json') if (output / 'receipt.json').exists() else None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'execute', 'verify'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == 'prepare':
        p = prepare(args.output)
        print(json.dumps(dict(entrances=len(p['cases']), support=len(p['support_cases']), bounds=p['bounds'])))
    elif args.command == 'execute':
        execute(args.output)
    else:
        r = load_canonical_json(args.output / 'receipt.json')
        validate_receipt(r)
        print(json.dumps(dict(status=r['status'], updates=r['updates'], final_margins={c:s['A_vs_best_other_margin'] for c,s in r['final_scores'].items()}, resources=r['resources'])))


if __name__ == '__main__':
    main()
