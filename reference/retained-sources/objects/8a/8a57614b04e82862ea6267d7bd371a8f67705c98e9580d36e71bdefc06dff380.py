"""One Source-started, two-entrance last-token CE feasibility update sequence."""
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
from .candidate_opportunity import file_hash, indexed, require
from .route_access import CONFIG, ROOT, publish
from .branch_bridge import entrance, summarize_logits

BRIDGE = ROOT / '2026-09-10-verified-branch-update-bridge'
OUTPUT = ROOT / '2026-09-10-native-entrance-ce-feasibility/training'
EXPECTED = {'368': (97, 152125, '2022537'), '7116': (22, 151961, '181378')}
OPTIMIZER = dict(lr=1e-5, betas=[0.9, 0.999], eps=1e-8, weight_decay=0., foreach=False)


def last_target_loss(logits, target_ids, entry):
    """Only the final causal target row contributes; all prefix rows have zero weight."""
    from src.losses import aligned_token_logprobs
    require(logits.ndim == 2 and target_ids.ndim == 1 and logits.shape[0] == len(entry['state_ids']) + 1,
            'last-target causal alignment')
    require(target_ids.tolist() == entry['state_ids'] + [entry['A_id']], 'last-target literal token identity')
    return -0.5 * aligned_token_logprobs(logits[-1:], target_ids[-1:]).sum()


def stop_reason(step, scores):
    require(1 <= step <= 32 and len(scores) == 2, 'joint stop coverage')
    margins = [s['A_vs_best_other_margin'] for s in scores.values()]
    require(all(math.isfinite(v) for v in margins), 'nonfinite stop score')
    if all(v >= 0.1 for v in margins):
        return 'both_fixed_prefix_margins_at_least_0.1'
    return '32_update_limit' if step == 32 else None


def validate_completed_receipt(receipt):
    require(receipt['status'] == 'completed' and receipt['schema_version'] == 'native_entrance_ce.training.v1',
            'checkpoint incomplete')
    require(len(receipt['cases']) == 2 and len({c['case_id'] for c in receipt['cases']}) == 2,
            'checkpoint case identity')
    require(set(receipt['final_scores']) == {c['case_id'] for c in receipt['cases']}, 'checkpoint score coverage')
    require(stop_reason(receipt['updates'], receipt['final_scores']) == receipt['stop_reason'], 'checkpoint stop identity')
    for c in receipt['cases']:
        s = receipt['final_scores'][c['case_id']]
        require(s['target_id'] == c['target_token_id'] and c['state_ids'] == c['prefix_token_ids'] and
                len(c['state_ids']) == c['action_index'], 'checkpoint target/prefix identity')
    for identity in (receipt['adapter'], receipt['source_embedding']):
        files = identity['files']
        require(files and len({f['relative_path'] for f in files}) == len(files), 'checkpoint file identity')
        for f in files:
            path = Path(identity['root']) / f['relative_path']
            require(path.is_file() and file_hash(path) == f['sha256'], 'checkpoint missing/corrupt bytes')
    require({'adapter_config.json', 'adapter_model.safetensors'} <= {f['relative_path'] for f in receipt['adapter']['files']},
            'standard adapter incomplete')


def prepare(output):
    require(not output.exists(), 'occupied training root')
    bridge = load_canonical_json(BRIDGE / 'inputs.json')
    acceptance = json.loads((BRIDGE / 'lead-acceptance.json').read_text())
    require(acceptance['status'] == 'lead-accepted', 'bridge not accepted')
    for path, sha in bridge['source_files'].items():
        require(file_hash(path) == sha, 'bridge frozen source changed')
    bindings = bridge['bindings']
    require(len(bindings) == 2 and {b['case']['image_id'] for b in bindings} == set(EXPECTED), 'frozen two-image population')
    cases = []
    for binding in bindings:
        c, ent = binding['case'], entrance(binding['case'])
        require(ent == binding['entrance'], 'bridge entrance changed')
        index, target, owner = EXPECTED[c['image_id']]
        require(ent['action_index'] == index and ent['A_id'] == target and c['case_id'].endswith(':' + owner),
                'frozen entrance target changed')
        require(ent['state_ids'] == c['source_ids'][:index], 'not original Source prefix')
        cases.append(dict(case_id=c['case_id'], example_id=c['example_id'], image_id=c['image_id'], owner_id=owner,
                          action_index=index, target_token_id=target, state_ids=ent['state_ids'], prefix_token_ids=ent['state_ids'],
                          prompt_token_ids=binding['group']['prompt_token_ids'], entrance=ent, group=binding['group']))
    packet = dict(schema_version='native_entrance_ce.inputs.v1', model=bridge['model'], train_source=bridge['train_source'],
                  cases=cases, source_files={str(BRIDGE / 'inputs.json'): file_hash(BRIDGE / 'inputs.json'),
                                           str(BRIDGE / 'lead-acceptance.json'): file_hash(BRIDGE / 'lead-acceptance.json'),
                                           str(CONFIG): file_hash(CONFIG), **bridge['source_files']},
                  qualifications=acceptance['source_qualification'], optimizer=OPTIMIZER, clip_grad_norm=1.,
                  max_updates=32, joint_margin_threshold=0.1,
                  loss='-0.5*(logP(target368|exact_Source_state368)+logP(target7116|exact_Source_state7116)); no other supervised positions')
    output.mkdir(parents=True, exist_ok=False)
    publish(output / 'inputs.json', packet)
    sources = [Path(__file__), Path(__file__).with_name('tests') / 'test_entrance_ce.py', CONFIG,
               Path(__file__).with_name('train.py'), Path(__file__).with_name('runtime.py'),
               Path(__file__).with_name('branch_bridge.py'), Path(__file__).with_name('route_access.py')]
    sources += list(Path('src/qwen').glob('*.py')) + [Path('src/losses/token_scores.py'), Path('src/adapters/dora.py')]
    records = []
    for path in sources:
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
        require(file_hash(path) == sha, f'frozen training source changed: {path}')
    config = load_research_infer_config(CONFIG).config
    require(str(config.model.base_model) == packet['model']['base_model_path'] and
            str(config.adapter.path) == packet['model']['current_adapter']['root'] and
            str(config.embedding_delta.path) == packet['model']['source_embedding']['root'], 'original Source initialization identity')
    require(config.model.dtype == 'fp32' and config.backend.hf.attn_implementation == 'sdpa' and
            config.backend.hf.patch_embed_linearization == 'enabled', 'frozen execution numerics')
    require(file_hash(config.data.input_jsonl) == packet['train_source']['sha256'], 'training dataset identity')
    for identity in (packet['model']['current_adapter'], packet['model']['source_embedding']):
        for f in identity['files']:
            require(file_hash(Path(identity['root']) / f['relative_path']) == f['sha256'], 'Source checkpoint bytes changed')
    frontend = assemble_frontend(config, generation_config_fingerprint=sha256_json(config.generation.model_dump(mode='json')))
    raw = {str(r.example_id): r for r in load_raw_examples(config.data.input_jsonl)}
    publish(output / 'launch.json', dict(pid=os.getpid(), time=time.time(), visible_devices='0',
            processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader'], text=True)))
    started = time.monotonic()
    counters = dict(model_loads=0, train_forwards=0, initial_score_forwards=0, post_score_forwards=0, updates=0,
                    actual_model_forwards=0, scored_target_tokens=0, supervised_target_tokens=0)
    status, error, completed = 'failed', None, None
    def expired(*_):
        raise TimeoutError('900 second cumulative model-execution budget')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(900)
    try:
        qwen, model_receipt = load_policy(config, device=torch.device('cuda:0'))
        counters['model_loads'] += 1
        model = qwen.model
        model.eval()
        require(model_receipt['effective_settings']['observed_model_dtype']['parameter_dtype_names'] == ['torch.float32'] and
                model_receipt['effective_settings']['observed_attn_implementation'] == 'sdpa' and
                model_receipt['model_identity']['adapter']['merged_adapters'] == [], 'loaded FP32 SDPA unmerged identity')
        publish(output / 'loaded_model.json', model_receipt)
        publish(output / 'effective_config.json', config.model_dump(mode='json'))
        def count_forward(*_):
            counters['actual_model_forwards'] += 1
        model.register_forward_pre_hook(count_forward)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        named = select_dora_parameters(model, towers=('language',), adapter_name='default')
        require(len(named) == EXPECTED_TRAINABLE_TENSORS == 588 and
                sum(p.numel() for _, p in named) == EXPECTED_TRAINABLE_SCALARS == 18006016 and
                all('language_model' in n and not any(x in n for x in ('visual', 'merger', 'embed_tokens', 'lm_head')) for n, _ in named),
                'language DoRA training surface')
        for _, parameter in named:
            parameter.requires_grad_(True)
        ids = {id(p) for _, p in named}
        frozen = [(n, p) for n, p in model.named_parameters() if id(p) not in ids]
        require(all(not p.requires_grad for _, p in frozen), 'nonadapter surface not frozen')
        frozen_versions = [(p, p._version) for _, p in frozen]
        frozen_before = _tensor_state_hash(frozen)
        adapter_before = _tensor_state_hash(named)
        initial_parameters = [p.detach().clone() for _, p in named]
        layout = _parameter_layout(named)
        publish(output / 'trainable_layout.json', layout)
        optimizer = torch.optim.AdamW([p for _, p in named], **OPTIMIZER)
        require(not optimizer.state, 'optimizer is not fresh')
        materialized = []
        for case in packet['cases']:
            prompt, inputs, grid = _materialize_group(qwen=qwen, frontend=frontend, config=config,
                                                     raw=raw[case['example_id']], group=case['group'])
            require(prompt == case['prompt_token_ids'], 'training prompt identity')
            materialized.append((case, prompt, {**inputs, 'image_grid_thw': grid}))
        torch.cuda.reset_peak_memory_stats()
        def replay_logits(case, prompt, inputs):
            entry = case['entrance']
            replay = prepare_replay(model, inputs, prompt_token_ids=prompt,
                                    continuation_token_ids=entry['state_ids'] + [entry['A_id']])
            require(replay.target_ids.tolist() == entry['state_ids'] + [entry['A_id']], 'native target alignment')
            logits = replay.aligned_logits(model(**replay.inputs).logits)
            return logits, replay.target_ids
        def read_scores(step):
            scores = {}
            for case, prompt, inputs in materialized:
                key = 'initial_score_forwards' if step == 0 else 'post_score_forwards'
                require(counters[key] < (4 if step == 0 else 64), 'score forward budget')
                counters[key] += 1
                with torch.inference_mode():
                    logits, targets = replay_logits(case, prompt, inputs)
                    require(logits.shape[0] == case['action_index'] + 1 and int(targets[-1]) == case['target_token_id'], 'score target alignment')
                    values = logits[-1].detach().cpu().numpy().copy()
                    del logits, targets
                score = summarize_logits(values, case['entrance'])
                score['target_id'] = case['target_token_id']
                scores[case['case_id']] = score
                path = output / 'scores' / f"step-{step:02d}-{case['image_id']}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    np.save(stream, values, allow_pickle=False)
                require(np.array_equal(values, np.load(path, allow_pickle=False)), 'score disk reload changed')
                counters['scored_target_tokens'] += 1
            publish(output / 'scores' / f'step-{step:02d}.json', scores)
            return scores
        initial_scores = read_scores(0)
        dose = []
        for step in range(1, 33):
            require(time.monotonic() - started < 900 and counters['train_forwards'] + 2 <= 64, 'training budget')
            optimizer.zero_grad(set_to_none=True)
            losses = []
            before_step = [p.detach().clone() for _, p in named]
            for case, prompt, inputs in materialized:
                counters['train_forwards'] += 1
                logits, targets = replay_logits(case, prompt, inputs)
                loss = last_target_loss(logits, targets, case['entrance'])
                require(bool(torch.isfinite(loss)), 'nonfinite target loss')
                loss.backward()
                losses.append(float(loss.detach()))
                counters['supervised_target_tokens'] += 1
                del loss, logits, targets
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _, p in named), 'missing/nonfinite selected gradient')
            require(all(p.grad is None and not p.requires_grad for _, p in frozen), 'frozen surface received gradients')
            raw_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in named], 1., error_if_nonfinite=True, foreach=False))
            require(math.isfinite(raw_norm) and raw_norm > 0, 'zero/nonfinite joint gradient')
            clipped_norm = math.sqrt(sum(float(p.grad.double().square().sum()) for _, p in named))
            require(clipped_norm <= 1.000001, 'gradient clipping bound')
            optimizer.step()
            counters['updates'] = step
            require(all(p._version == version for p, version in frozen_versions), 'frozen parameter mutation')
            delta_norm = math.sqrt(sum(float((p.detach() - old).double().square().sum()) for (_, p), old in zip(named, before_step)))
            total_delta = math.sqrt(sum(float((p.detach() - old).double().square().sum()) for (_, p), old in zip(named, initial_parameters)))
            require(delta_norm > 0 and math.isfinite(total_delta), 'zero/nonfinite parameter movement')
            del before_step
            final_scores = read_scores(step)
            reason = stop_reason(step, final_scores)
            row = dict(update=step, loss=sum(losses), equal_image_half_losses=losses, raw_gradient_norm=raw_norm,
                       clipped_gradient_norm=clipped_norm, step_parameter_delta_l2=delta_norm,
                       source_parameter_delta_l2=total_delta, final_scores=final_scores,
                       counters=dict(counters), cumulative_seconds=time.monotonic() - started, stop_reason=reason)
            dose.append(row)
            publish(output / f'update-{step:02d}.json', row)
            if step == 1:
                require(_tensor_state_hash(frozen) == frozen_before, 'update1 frozen bytes changed')
                require(any(final_scores[c]['target_logit'] != initial_scores[c]['target_logit'] for c in final_scores), 'update1 no actual score movement')
                publish(output / 'vertical_smoke.json', dict(update=1, labels_per_image=1, equal_image_accumulation=True,
                        gradients_finite_nonzero=True, frozen_bytes_unchanged=True, parameter_movement_l2=delta_norm,
                        score_publication_reload=True, scores=final_scores, counters=dict(counters)))
            print(json.dumps(dict(update=step, loss=row['loss'], margins={c: s['A_vs_best_other_margin'] for c, s in final_scores.items()},
                                  seconds=time.monotonic() - started, stop_reason=reason)), flush=True)
            if reason:
                break
        require(_tensor_state_hash(frozen) == frozen_before, 'final frozen parameter bytes changed')
        require({int(s['step']) for s in optimizer.state.values()} == {step}, 'optimizer update count')
        adapter_after = _tensor_state_hash(named)
        require(adapter_after != adapter_before, 'adapter did not change')
        adapter = _save_adapter_only(model, source_root=Path(config.adapter.path), output=output / 'adapter')
        require(adapter['fingerprint'] != packet['model']['current_adapter']['fingerprint'], 'saved adapter unchanged')
        for identity in (packet['model']['current_adapter'], packet['model']['source_embedding']):
            for f in identity['files']:
                require(file_hash(Path(identity['root']) / f['relative_path']) == f['sha256'], 'Source checkpoint mutated')
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        require(elapsed < 900 and counters['actual_model_forwards'] == counters['train_forwards'] + counters['initial_score_forwards'] + counters['post_score_forwards'], 'final execution accounting')
        completed = dict(schema_version='native_entrance_ce.training.v1', status='completed', adapter=adapter,
                         source_adapter=packet['model']['current_adapter'], source_embedding=packet['model']['source_embedding'],
                         config=config.model_dump(mode='json'), cases=packet['cases'], final_scores=final_scores, initial_scores=initial_scores,
                         updates=step, stop_reason=reason, optimizer=OPTIMIZER, clip_gradient_norm=1., trainable_layout=layout,
                         adapter_tensor_hash_before=adapter_before, adapter_tensor_hash_after=adapter_after,
                         frozen_tensor_hash_before=frozen_before, frozen_tensor_hash_after=frozen_before,
                         inputs_sha256=file_hash(output / 'inputs.json'), code_identity_sha256=file_hash(output / 'code_identity.json'),
                         dose=dose, resources=dict(**counters, cumulative_model_seconds=elapsed,
                         peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                         peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                         artifact_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())),
                         cold_reload_status='Pending independent evaluator; saved tensor bytes independently reloaded and equal to live adapter.')
        validate_completed_receipt(completed)
        publish(output / 'receipt.json', completed)
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
        packet = prepare(args.output)
        print(json.dumps(dict(cases=len(packet['cases']), prepared=True)))
    elif args.command == 'execute':
        execute(args.output)
    else:
        receipt = load_canonical_json(args.output / 'receipt.json')
        validate_completed_receipt(receipt)
        print(json.dumps(dict(status=receipt['status'], updates=receipt['updates'], stop_reason=receipt['stop_reason'],
                              final_margins={c: s['A_vs_best_other_margin'] for c, s in receipt['final_scores'].items()}, resources=receipt['resources'])))


if __name__ == '__main__':
    main()
