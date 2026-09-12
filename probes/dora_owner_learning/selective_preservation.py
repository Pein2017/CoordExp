"""Fixed Source-started entrance CE plus full-vocabulary soft preservation."""
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
from .candidate_opportunity import file_hash, require, rows
from .route_access import CONFIG, ROOT, checked_ids, publish
from .branch_bridge import summarize_logits
from .entrance_ce import OPTIMIZER, EXPECTED

OUTPUT = ROOT / '2026-09-10-selective-owner-learning-autonomous/soft-preservation-10/training'
ENTRANCE_INPUTS = ROOT / '2026-09-10-native-entrance-ce-feasibility/training/inputs.json'
REFERENCE_ROWS = ROOT / '2026-09-10-owner-row-continuation-robustness/execution/rows.jsonl'
LAMBDA, UPDATES = 10., 23


def preservation_positions(action_ids, entry, suffix_start):
    checked_ids(action_ids, 'im_end')
    index = entry['action_index']
    require(len(entry['state_ids']) == index and action_ids[:index] == entry['state_ids'] and
            action_ids[index] == entry['A_id'], 'reference entrance identity/alignment')
    require(index < suffix_start < len(action_ids), 'supplied row boundary')
    return [i for i in range(len(action_ids)) if i < index or i >= suffix_start]


def selective_loss(logits, target_ids, entry, action_ids, suffix_start, positions, reference_logp):
    """Half an image objective; caller accumulates both images before one step."""
    import torch
    from src.losses import aligned_token_logprobs
    expected = preservation_positions(action_ids, entry, suffix_start)
    require(positions == expected and logits.ndim == 2 and logits.shape[0] == len(action_ids) and
            target_ids.tolist() == action_ids, 'CE/KL target or mask alignment')
    require(reference_logp.shape == (len(positions), logits.shape[1]), 'reference distribution shape')
    require(reference_logp.dtype == logits.dtype == torch.float32, 'FP32 reference/student scores')
    index = entry['action_index']
    ce = -aligned_token_logprobs(logits[index:index + 1], target_ids[index:index + 1]).sum()
    ref = reference_logp.detach()
    student_logp = torch.log_softmax(logits[positions], dim=-1)
    per_state = (ref.exp() * (ref - student_logp)).sum(dim=-1)
    kl = per_state.mean()
    return .5 * (ce + LAMBDA * kl), ce, kl


def validate_receipt(receipt):
    require(receipt['schema_version'] == 'selective_preservation.training.v1' and receipt['status'] == 'completed' and
            receipt['updates'] == UPDATES and receipt['stop_reason'] == 'fixed_steps', 'incomplete fixed-step checkpoint')
    require(len(receipt['cases']) == 2 and set(receipt['final_scores']) == {c['case_id'] for c in receipt['cases']},
            'checkpoint case/score coverage')
    for case in receipt['cases']:
        require(receipt['final_scores'][case['case_id']]['target_id'] == case['target_token_id'], 'checkpoint target identity')
    for identity in (receipt['adapter'], receipt['source_embedding']):
        require(identity['files'] and len({f['relative_path'] for f in identity['files']}) == len(identity['files']), 'checkpoint file identities')
        for f in identity['files']:
            path = Path(identity['root']) / f['relative_path']
            require(path.is_file() and file_hash(path) == f['sha256'], 'checkpoint missing/corrupt bytes')
    require({'adapter_config.json', 'adapter_model.safetensors'} <= {f['relative_path'] for f in receipt['adapter']['files']},
            'incomplete standard adapter')


def prepare(output):
    require(not output.exists(), 'occupied training root')
    old = load_canonical_json(ENTRANCE_INPUTS)
    refs = [r for r in rows(REFERENCE_ROWS) if r['arm'] == 'A']
    require(len(refs) == 2 and len({r['case_id'] for r in refs}) == 2, 'reference A population')
    reference_by_case = {r['case_id']: r for r in refs}
    trajectories = {}
    for case in old['cases']:
        index, target, owner = EXPECTED[case['image_id']]
        require(case['action_index'] == index and case['target_token_id'] == target and case['owner_id'] == owner,
                'frozen entrance constants')
        ref = reference_by_case[case['case_id']]
        action = ref['action_ids']
        require(action == ref['prefix_ids'] + ref['forced_ids'] + ref['suffix_ids'] and ref['stop_reason'] == 'im_end',
                'complete reference trajectory identity')
        suffix_start = len(ref['prefix_ids']) + len(ref['forced_ids'])
        positions = preservation_positions(action, case['entrance'], suffix_start)
        require(len(ref['forced_ids']) == 9 and suffix_start == index + 5 and
                len(action) == {'368': 139, '7116': 46}[case['image_id']], 'frozen A row/trajectory bounds')
        trajectories[case['case_id']] = dict(action_ids=action, prefix_ids=ref['prefix_ids'], forced_ids=ref['forced_ids'],
                suffix_ids=ref['suffix_ids'], suffix_start=suffix_start, preservation_positions=positions,
                preservation_count=len(positions), excluded_positions=list(range(index, suffix_start)),
                reference_row=ref)
    require(set(trajectories) == {c['case_id'] for c in old['cases']}, 'missing reference cells')
    sources = {**old['source_files'], str(ENTRANCE_INPUTS): file_hash(ENTRANCE_INPUTS), str(REFERENCE_ROWS): file_hash(REFERENCE_ROWS)}
    for path, sha in sources.items():
        require(file_hash(path) == sha, f'input bytes changed: {path}')
    packet = dict(schema_version='selective_preservation.inputs.v1', cases=old['cases'], trajectories=trajectories,
            model=old['model'], train_source=old['train_source'], qualifications=old['qualifications'], source_files=sources,
            optimizer=OPTIMIZER, clip_gradient_norm=1., lambda_kl=LAMBDA, updates=UPDATES,
            loss='Mean_image[CE(original x1) + 10*mean_preserved_state KL(Source||student)]. Prefix before x1 and natural suffix through EOS preserved; supplied x1/remaining A row excluded from KL.',
            bounds=dict(model_loads=1, reference_forwards=2, train_forwards=46, entry_score_forwards=48, total_model_forwards=96, model_seconds=3600))
    output.mkdir(parents=True, exist_ok=False)
    publish(output / 'inputs.json', packet)
    files = [Path(__file__), Path(__file__).with_name('tests') / 'test_selective_preservation.py', CONFIG,
             Path(__file__).with_name('train.py'), Path(__file__).with_name('runtime.py'),
             Path(__file__).with_name('entrance_ce.py'), Path(__file__).with_name('branch_bridge.py')]
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
    require(file_hash(config.data.input_jsonl) == packet['train_source']['sha256'], 'dataset bytes changed')
    for identity in (packet['model']['current_adapter'], packet['model']['source_embedding']):
        for f in identity['files']:
            require(file_hash(Path(identity['root']) / f['relative_path']) == f['sha256'], 'Source payload changed')
    frontend = assemble_frontend(config, generation_config_fingerprint=sha256_json(config.generation.model_dump(mode='json')))
    raw = {str(r.example_id): r for r in load_raw_examples(config.data.input_jsonl)}
    publish(output / 'launch.json', dict(pid=os.getpid(), time=time.time(), visible_devices='0',
            processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader'], text=True)))
    started = time.monotonic()
    counters = dict(model_loads=0, reference_forwards=0, train_forwards=0, entry_score_forwards=0,
                    actual_model_forwards=0, updates=0, supervised_target_tokens=0)
    status, error = 'failed', None
    def expired(*_):
        raise TimeoutError('3600-second model invocation bound')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(3600)
    try:
        qwen, model_identity = load_policy(config, device=torch.device('cuda:0'))
        counters['model_loads'] = 1
        model = qwen.model
        model.eval()
        require(model_identity['effective_settings']['observed_model_dtype']['parameter_dtype_names'] == ['torch.float32'] and
                model_identity['effective_settings']['observed_attn_implementation'] == 'sdpa' and
                model_identity['model_identity']['adapter']['merged_adapters'] == [], 'loaded numerical/adapter identity')
        publish(output / 'loaded_model.json', model_identity)
        publish(output / 'effective_config.json', config.model_dump(mode='json'))
        def count_forward(*_):
            counters['actual_model_forwards'] += 1
            require(counters['actual_model_forwards'] <= 96, 'actual model forward cap')
        model.register_forward_pre_hook(count_forward)
        for p in model.parameters():
            p.requires_grad_(False)
        named = select_dora_parameters(model, towers=('language',), adapter_name='default')
        require(len(named) == EXPECTED_TRAINABLE_TENSORS == 588 and sum(p.numel() for _, p in named) == EXPECTED_TRAINABLE_SCALARS == 18006016 and
                all('language_model' in n and not any(x in n for x in ('visual', 'merger', 'embed_tokens', 'lm_head')) for n, _ in named), 'training surface')
        for _, p in named:
            p.requires_grad_(True)
        selected = {id(p) for _, p in named}
        frozen = [(n, p) for n, p in model.named_parameters() if id(p) not in selected]
        frozen_versions = [(p, p._version) for _, p in frozen]
        frozen_hash = _tensor_state_hash(frozen)
        source_hash = _tensor_state_hash(named)
        original = [p.detach().clone() for _, p in named]
        layout = _parameter_layout(named)
        publish(output / 'trainable_layout.json', layout)
        optimizer = torch.optim.AdamW([p for _, p in named], **OPTIMIZER)
        require(not optimizer.state, 'fresh optimizer required')
        materialized = []
        for case in packet['cases']:
            prompt, inputs, grid = _materialize_group(qwen=qwen, frontend=frontend, config=config,
                    raw=raw[case['example_id']], group=case['group'])
            materialized.append((case, prompt, {**inputs, 'image_grid_thw': grid}))
        torch.cuda.reset_peak_memory_stats()
        def forward(case, prompt, inputs, *, complete):
            ids = packet['trajectories'][case['case_id']]['action_ids'] if complete else case['state_ids'] + [case['target_token_id']]
            replay = prepare_replay(model, inputs, prompt_token_ids=prompt, continuation_token_ids=ids)
            require(replay.target_ids.tolist() == ids, 'native causal target identity')
            return replay.aligned_logits(model(**replay.inputs).logits), replay.target_ids
        def save_array(path, values):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                np.save(stream, values, allow_pickle=False)
            require(np.array_equal(values, np.load(path, allow_pickle=False)), 'array publication/reload')
        def entry_scores(step):
            scores = {}
            for case, prompt, inputs in materialized:
                require(counters['entry_score_forwards'] < 48, 'entry score budget')
                counters['entry_score_forwards'] += 1
                with torch.inference_mode():
                    logits, targets = forward(case, prompt, inputs, complete=False)
                    require(logits.shape[0] == case['action_index'] + 1 and int(targets[-1]) == case['target_token_id'], 'isolated entry alignment')
                    values = logits[-1].detach().cpu().numpy().copy()
                    del logits, targets
                result = summarize_logits(values, case['entrance'])
                result['target_id'] = case['target_token_id']
                scores[case['case_id']] = result
                save_array(output / 'scores' / f"step-{step:02d}-{case['image_id']}.npy", values)
            publish(output / 'scores' / f'step-{step:02d}.json', scores)
            return scores
        references, reference_cards = {}, []
        for case, prompt, inputs in materialized:
            trajectory = packet['trajectories'][case['case_id']]
            counters['reference_forwards'] += 1
            # no_grad rather than inference_mode: detached cache remains safe in later autograd graphs.
            with torch.no_grad():
                logits, targets = forward(case, prompt, inputs, complete=True)
                logp = torch.log_softmax(logits[trajectory['preservation_positions']], -1).detach().clone()
                _, ce, kl = selective_loss(logits, targets, case['entrance'], trajectory['action_ids'],
                        trajectory['suffix_start'], trajectory['preservation_positions'], logp)
                require(abs(float(kl)) <= 1e-6 and not logp.requires_grad and logp.grad_fn is None, 'initial KL or detached-reference failure')
                references[case['case_id']] = logp
                values = logp.cpu().numpy().copy()
                del logits, targets
            path = output / 'references' / f"{case['image_id']}-logp.npy"
            save_array(path, values)
            reference_cards.append(dict(case_id=case['case_id'], shape=list(logp.shape), cache_bytes=logp.numel() * logp.element_size(),
                    sha256=file_hash(path), initial_KL=float(kl), initial_CE=float(ce), detached=True))
        publish(output / 'references.json', reference_cards)
        initial_scores = entry_scores(0)
        dose = []
        for step in range(1, UPDATES + 1):
            require(time.monotonic() - started < 3600 and counters['train_forwards'] + 2 <= 46, 'training invocation budget')
            optimizer.zero_grad(set_to_none=True)
            before_step = [p.detach().clone() for _, p in named]
            losses = []
            for case, prompt, inputs in materialized:
                counters['train_forwards'] += 1
                trajectory = packet['trajectories'][case['case_id']]
                logits, targets = forward(case, prompt, inputs, complete=True)
                loss, ce, kl = selective_loss(logits, targets, case['entrance'], trajectory['action_ids'],
                        trajectory['suffix_start'], trajectory['preservation_positions'], references[case['case_id']])
                require(bool(torch.isfinite(loss)) and float(kl.detach()) >= -1e-6, 'nonfinite/negative KL loss')
                if step == 1:
                    require(abs(float(kl.detach())) <= 1e-6, 'initial student/reference replay differs')
                loss.backward()
                losses.append(dict(case_id=case['case_id'], half_weighted_loss=float(loss.detach()),
                        CE=float(ce.detach()), mean_KL=float(kl.detach()), preserved_states=trajectory['preservation_count']))
                counters['supervised_target_tokens'] += 1
                del loss, ce, kl, logits, targets
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _, p in named), 'selected gradients missing/nonfinite')
            require(all(p.grad is None and not p.requires_grad for _, p in frozen), 'frozen surface gradients')
            raw_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in named], 1., error_if_nonfinite=True, foreach=False))
            require(math.isfinite(raw_norm) and raw_norm > 0, 'joint zero/nonfinite gradient')
            clipped_norm = math.sqrt(sum(float(p.grad.double().square().sum()) for _, p in named))
            require(clipped_norm <= 1.000001, 'gradient clip bound')
            optimizer.step()
            counters['updates'] = step
            require(all(p._version == v for p, v in frozen_versions), 'frozen parameter mutation')
            movement = math.sqrt(sum(float((p.detach() - old).double().square().sum()) for (_, p), old in zip(named, before_step)))
            total_movement = math.sqrt(sum(float((p.detach() - old).double().square().sum()) for (_, p), old in zip(named, original)))
            require(movement > 0 and math.isfinite(total_movement), 'zero/nonfinite adapter movement')
            del before_step
            final_scores = entry_scores(step)
            row = dict(update=step, loss=sum(x['half_weighted_loss'] for x in losses), image_losses=losses,
                    raw_gradient_norm=raw_norm, clipped_gradient_norm=clipped_norm, step_parameter_delta_l2=movement,
                    source_parameter_delta_l2=total_movement, final_scores=final_scores, counters=dict(counters),
                    cumulative_seconds=time.monotonic() - started)
            dose.append(row)
            publish(output / f'update-{step:02d}.json', row)
            if step == 1:
                require(_tensor_state_hash(frozen) == frozen_hash, 'update1 frozen bytes changed')
                require(any(final_scores[c]['target_logit'] != initial_scores[c]['target_logit'] for c in final_scores), 'update1 score did not move')
                publish(output / 'vertical_smoke.json', dict(update=1, initial_KL_near_zero=True, detached_reference=True,
                        selected_gradients_finite_nonzero=True, frozen_bytes_unchanged=True, parameter_movement_l2=movement,
                        score_publication_reload=True, counters=dict(counters), image_losses=losses, final_scores=final_scores))
                print('VERTICAL_SMOKE_COMPLETED', flush=True)
            print(json.dumps(dict(update=step, losses=losses, margins={c: s['A_vs_best_other_margin'] for c, s in final_scores.items()},
                    seconds=time.monotonic() - started)), flush=True)
        require(_tensor_state_hash(frozen) == frozen_hash, 'final frozen bytes changed')
        require(all(not r.requires_grad and r.grad_fn is None for r in references.values()), 'reference acquired gradient graph')
        require({int(s['step']) for s in optimizer.state.values()} == {UPDATES}, 'optimizer step count')
        final_hash = _tensor_state_hash(named)
        adapter = _save_adapter_only(model, source_root=Path(config.adapter.path), output=output / 'adapter')
        require(final_hash != source_hash and adapter['fingerprint'] != packet['model']['current_adapter']['fingerprint'], 'unchanged final adapter')
        for identity in (packet['model']['current_adapter'], packet['model']['source_embedding']):
            for f in identity['files']:
                require(file_hash(Path(identity['root']) / f['relative_path']) == f['sha256'], 'Source payload mutated')
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        require(elapsed < 3600 and counters['actual_model_forwards'] == 96, 'fixed-step execution count/budget')
        receipt = dict(schema_version='selective_preservation.training.v1', status='completed', adapter=adapter,
                source_adapter=packet['model']['current_adapter'], source_embedding=packet['model']['source_embedding'],
                config=config.model_dump(mode='json'), cases=packet['cases'], final_scores=final_scores, initial_scores=initial_scores,
                updates=UPDATES, stop_reason='fixed_steps', optimizer=OPTIMIZER, clip_gradient_norm=1., lambda_kl=LAMBDA,
                trainable_layout=layout, adapter_tensor_hash_before=source_hash, adapter_tensor_hash_after=final_hash,
                frozen_tensor_hash_before=frozen_hash, frozen_tensor_hash_after=frozen_hash, dose=dose, references=reference_cards,
                inputs_sha256=file_hash(output / 'inputs.json'), code_identity_sha256=file_hash(output / 'code_identity.json'),
                resources=dict(**counters, cumulative_model_seconds=elapsed,
                    reference_cache_bytes=sum(r['cache_bytes'] for r in reference_cards),
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
        print(json.dumps({c: dict(length=len(t['action_ids']), preserved=t['preservation_count'], suffix_start=t['suffix_start']) for c, t in p['trajectories'].items()}))
    elif args.command == 'execute':
        execute(args.output)
    else:
        r = load_canonical_json(args.output / 'receipt.json')
        validate_receipt(r)
        print(json.dumps(dict(status=r['status'], updates=r['updates'], stop_reason=r['stop_reason'],
                final_margins={c: s['A_vs_best_other_margin'] for c, s in r['final_scores'].items()}, resources=r['resources'])))


if __name__ == '__main__':
    main()
