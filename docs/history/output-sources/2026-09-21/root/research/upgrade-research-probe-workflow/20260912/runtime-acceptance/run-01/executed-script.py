"""Bounded mechanical Source-policy acceptance; never a scientific experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def publish(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=900)
    parser.add_argument('--start-gpu-authorized', action='store_true')
    args = parser.parse_args()
    if not args.start_gpu_authorized:
        parser.error('lead START_GPU authorization is required')
    if not 1 <= args.max_seconds <= 900:
        parser.error('wall budget must be positive and at most 900 seconds')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        parser.error('launch must explicitly expose only CUDA_VISIBLE_DEVICES=0')
    source, baseline, output = (p.resolve() for p in (args.source, args.baseline, args.output))
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(source))
    started = time.monotonic()
    counters = dict(model_loads=0, native_materializations=0, model_forwards=0,
                    image_forwards=0, updates=0, generated_tokens=0)
    receipt = dict(schema_version='probe_workflow_mechanical_acceptance.v1', status='failed',
        scope='One fixture, annotated exact histories, unchanged profile CE/KL and AdamW; no scientific claim or natural-generation result.',
        pid=os.getpid(), source_root=str(source), script_sha256=digest_file(__file__),
        bounds=dict(simultaneous_gpus=1, max_wall_seconds=args.max_seconds,
                    max_model_loads=1, max_model_forwards=4, max_image_forwards=4,
                    max_updates=1, max_generated_tokens=0, checkpoint_saves=0), counters=counters)
    torch = None
    source_hashes = {}
    def timeout(*_):
        raise TimeoutError('mechanical runtime wall/GPU reservation exhausted')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    try:
        source_hashes = {str(p.resolve()): digest_file(p)
            for root in ('src', 'probes') for p in (source / root).rglob('*.py')}
        config_path = source / 'probes/dora_owner_learning/configs/source256.yaml'
        fixture = source / 'tests/fixtures/smoke/qwen3_vl_single_image_pack/examples.jsonl'
        old_path = baseline / 'old/source256.json'
        receipt['input_files'] = {str(p): digest_file(p) for p in (config_path, fixture, old_path,
            baseline / 'source-manifest.json', baseline / 'capture_inputs.py')}
        receipt['git_head'] = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        receipt['git_status'] = subprocess.check_output(['git', '-C', str(source), 'status', '--short'], text=True)
        publish(output / 'launch.json', receipt)

        import numpy as np
        import torch as torch_module
        torch = torch_module
        from src.config.inference import load_research_infer_config
        from src.data import load_raw_examples
        from src.inference.runtime import assemble_frontend
        from src.inference.inputs import plan_examples
        from src.config.fingerprint import sha256_json
        from src.qwen.native import prepare_native_inputs, prepare_replay
        from probes.dora_owner_learning.inspect import select_examples
        from probes.dora_owner_learning.runtime import load_policy, bind_source256_language_dora
        from probes.dora_owner_learning.selective_preservation import selective_loss, preservation_positions
        from probes.dora_owner_learning.entrance_ce import OPTIMIZER
        from probes.dora_owner_learning.route_access import score_logits
        from probes.dora_owner_learning.branch_bridge import summarize_logits
        from probes.dora_owner_learning.train import _tensor_state_hash, _parameter_layout
        from probes.logit_lens.base import _stage_source_gate

        torch.manual_seed(0)
        torch.set_num_threads(4)
        old = json.loads(old_path.read_text())['rows'][0]
        population = load_raw_examples(fixture)
        selected = select_examples(population, example_ids=[old['example_id']])
        config = load_research_infer_config(config_path).config
        frontend = assemble_frontend(config,
            generation_config_fingerprint=sha256_json(config.generation.model_dump(mode='json')))
        assert frontend.qwen.model is None
        plan = plan_examples(selected, config=config, components=frontend.qwen,
                             row_indices=[0], target_max_length=12000)[0]
        target = plan.target
        assert target is not None
        start, end = target.supervised_token_spans[0].physical_token_start, target.supervised_token_spans[-1].physical_token_end
        action = list(target.input_ids[start:end])
        prompt = list(plan.prompt.expected_executed_prompt_token_ids)
        assert prompt == old['prompt_token_ids'] == list(target.input_ids[:start])
        assert action == old['action_token_ids'] and list(target.input_ids) == old['full_target_token_ids']
        assert [s.to_artifact_dict() for s in target.supervised_token_spans] == old['supervised_spans']
        assert [s.to_artifact_dict() for s in target.ignored_token_spans] == old['ignored_spans']
        assert plan.prompt.chat_text == old['chat_text'] and target.chat_text == old['target_chat_text']
        assert plan.image.image_content_sha256 == old['image_file_sha256']
        assert action[-1] == old['terminal_eos_id'] == 151645
        batch = prepare_native_inputs(frontend.qwen.processor, (plan.request,), device='cpu', record_media_identity=True)
        counters['native_materializations'] += 1
        def tensor_identity(value):
            data = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(data).hexdigest())
        native_hashes = {k: tensor_identity(v) for k, v in batch.inputs.items() if isinstance(v, torch.Tensor)}
        assert native_hashes == old['native_tensors']
        assert batch.media_sha256[0] == old['executed_rgb_sha256']
        assert list(batch.image_grids[0]) == old['grid']
        receipt['input_parity'] = dict(example_id=old['example_id'], selected_count=1,
            prompt_tokens=len(prompt), action_tokens=len(action), native_tensors=native_hashes,
            old_record=str(old_path), all_exact=True, model_weights_loaded_during_planning=False)
        gate_root, gate = _stage_source_gate(output)
        config = config.model_copy(update={'embedding_delta': config.embedding_delta.model_copy(update={'source_gate_root': gate_root})})
        receipt['source_gate'] = gate
        publish(output / 'effective_config.json', config.model_dump(mode='json'))
        for p in (config.model.base_model, config.adapter.path, config.embedding_delta.path):
            assert Path(p).is_dir(), str(p)
        receipt['cpu_preparation_seconds'] = time.monotonic() - started
        gpu_started = time.monotonic()
        receipt['gpu_region_started_offset_seconds'] = gpu_started - started
        qwen, identity = load_policy(config, device=torch.device('cuda:0'))
        counters['model_loads'] = 1
        publish(output / 'loaded_model.json', identity)
        assert identity['effective_settings']['observed_attn_implementation'] == 'sdpa'
        assert identity['effective_settings']['observed_model_dtype']['parameter_dtype_names'] == ['torch.float32']
        assert identity['model_identity']['adapter']['merged_adapters'] == []
        model = qwen.model
        model.eval()
        named, frozen = bind_source256_language_dora(model, expected_tensor_count=588,
            expected_scalar_count=18006016, adapter_name=config.adapter.name)
        assert all(p.requires_grad for _, p in named) and all(not p.requires_grad for _, p in frozen)
        frozen_versions = [(n, p, p._version) for n, p in frozen]
        frozen_before, selected_before = _tensor_state_hash(frozen), _tensor_state_hash(named)
        original = [p.detach().cpu().clone() for _, p in named]
        publish(output / 'trainable_layout.json', _parameter_layout(named))
        def model_hook(*_):
            counters['model_forwards'] += 1
            assert counters['model_forwards'] <= 4
        def image_hook(*_):
            counters['image_forwards'] += 1
            assert counters['image_forwards'] <= 4
        model.register_forward_pre_hook(model_hook)
        visual = [(n, m) for n, m in model.named_modules() if n == 'visual' or n.endswith('.visual')]
        assert len(visual) == 1, [n for n, _ in visual]
        visual[0][1].register_forward_pre_hook(image_hook)
        receipt['image_forward_owner'] = visual[0][0]
        native = {k: v.to('cuda:0') if isinstance(v, torch.Tensor) else v for k, v in batch.inputs.items()}
        pixels_ptr = native['pixel_values'].data_ptr()
        torch.cuda.reset_peak_memory_stats()
        def forward(prefix, continuation):
            replay = prepare_replay(model, native, prompt_token_ids=prefix, continuation_token_ids=continuation)
            assert replay.inputs['pixel_values'].data_ptr() == pixels_ptr
            assert replay.target_ids.tolist() == continuation
            result = replay.aligned_logits(model(**replay.inputs).logits)
            return result, replay.target_ids
        with torch.no_grad():
            old_logits, old_targets = forward(old['prompt_token_ids'], old['action_token_ids'])
            old_score = score_logits(old_logits, old_targets, prompt_length=len(prompt))
            old_values = old_logits.cpu().numpy().copy()
            del old_logits, old_targets
            logits, targets = forward(prompt, action)
            new_values = logits.cpu().numpy().copy()
            new_score = score_logits(logits, targets, prompt_length=len(prompt))
            assert np.array_equal(old_values, new_values) and old_score == new_score
            decoded = frontend.qwen.tokenizer.convert_ids_to_tokens(action)
            index = next(i for i, text in enumerate(decoded) if text.startswith('<|coord_'))
            other = int(logits[index].argmax())
            if other == action[index]:
                values = logits[index].clone(); values[other] = -torch.inf
                other = int(values.argmax())
            entry = dict(action_index=index, state_ids=action[:index], A_id=action[index], B_id=other)
            suffix_start = index + 5
            positions = preservation_positions(action, entry, suffix_start)
            reference = torch.log_softmax(logits[positions], -1).detach().clone()
            initial_loss, initial_ce, initial_kl = selective_loss(logits, targets, entry, action, suffix_start, positions, reference)
            assert abs(float(initial_kl)) <= 1e-6
            before_branch = summarize_logits(new_values[index], entry)
            del logits, targets
        np.save(output / 'old-zero-logits.npy', old_values, allow_pickle=False)
        np.save(output / 'new-zero-logits.npy', new_values, allow_pickle=False)
        publish(output / 'zero-score.json', dict(old=old_score, new=new_score, exact_parity=True))
        optimizer = torch.optim.AdamW([p for _, p in named], **OPTIMIZER)
        assert not optimizer.state
        optimizer.zero_grad(set_to_none=True)
        logits, targets = forward(prompt, action)
        loss, ce, kl = selective_loss(logits, targets, entry, action, suffix_start, positions, reference)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None for _, p in frozen)
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _, p in named)
        gradient_norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0)
        assert torch.isfinite(gradient_norm) and float(gradient_norm) > 0
        optimizer.step()
        counters['updates'] = 1
        selected_after = _tensor_state_hash(named)
        changed = [n for (n, p), before in zip(named, original, strict=True) if not torch.equal(p.detach().cpu(), before)]
        assert changed and selected_after != selected_before
        assert all(p._version == version for _, p, version in frozen_versions)
        frozen_after = _tensor_state_hash(frozen)
        assert frozen_after == frozen_before
        receipt['update'] = dict(objective='unchanged selective_preservation.selective_loss on one mechanical fixture',
            entry=entry, suffix_start=suffix_start, preservation_positions=positions,
            optimizer=OPTIMIZER, clip_gradient_norm=1.0, gradient_norm=float(gradient_norm),
            loss=float(loss.detach()), ce=float(ce.detach()), kl=float(kl.detach()),
            selected_tensor_count=len(named), selected_scalar_count=sum(p.numel() for _, p in named),
            changed_tensor_count=len(changed), changed_tensor_names=changed,
            selected_before_sha256=selected_before, selected_after_sha256=selected_after,
            frozen_before_sha256=frozen_before, frozen_after_sha256=frozen_after,
            frozen_versions_unchanged=True, frozen_gradients_absent=True)
        del logits, targets, loss, ce, kl
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            logits, targets = forward(prompt, action)
            after_score = score_logits(logits, targets, prompt_length=len(prompt))
            after_values = logits.cpu().numpy().copy()
            after_branch = summarize_logits(after_values[index], entry)
        np.save(output / 'after-update-logits.npy', after_values, allow_pickle=False)
        publish(output / 'paired-diagnostics.json', dict(
            scope='conditional annotated-history branch diagnostics; no natural generation or recovery claim',
            branch_scorer='probes.dora_owner_learning.branch_bridge.summarize_logits',
            token_scorer='probes.dora_owner_learning.route_access.score_logits',
            entry=entry, before=before_branch, after=after_branch,
            token_scores_before=new_score, token_scores_after=after_score))
        assert counters == dict(model_loads=1, native_materializations=1, model_forwards=4,
                                image_forwards=4, updates=1, generated_tokens=0)
        torch.cuda.synchronize()
        receipt['gpu_region_seconds'] = time.monotonic() - gpu_started
        receipt['status'] = 'completed'
    except BaseException as exc:
        receipt['error'] = dict(type=type(exc).__name__, message=str(exc))
        traceback.print_exc()
    finally:
        signal.alarm(0)
        receipt['wall_seconds'] = time.monotonic() - started
        receipt['gpu_minutes_conservative'] = receipt['wall_seconds'] / 60
        receipt['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if torch is not None and torch.cuda.is_initialized():
            receipt['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
            receipt['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
            receipt['cuda_device'] = torch.cuda.get_device_name(0)
        try:
            used = sorted({str(Path(m.__file__).resolve()) for m in list(sys.modules.values())
                           if getattr(m, '__file__', None) and str(Path(m.__file__).resolve()) in source_hashes})
            files = []
            for path in used:
                assert digest_file(path) == source_hashes[path], f'executed source changed: {path}'
                relative = Path(path).relative_to(source)
                staged = output / 'effective_code' / relative
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, staged)
                files.append(dict(path=path, staged=str(staged), sha256=source_hashes[path]))
            assert files
            publish(output / 'executed-source.json', dict(files=files, script_sha256=digest_file(__file__)))
            receipt['executed_source_file_count'] = len(files)
        except BaseException as exc:
            receipt['status'] = 'failed'
            receipt['source_identity_error'] = str(exc)
            traceback.print_exc()
        receipt['process_exit_code'] = 0 if receipt['status'] == 'completed' else 1
        publish(output / 'receipt.json', receipt)
        print(json.dumps({k: receipt[k] for k in ('status', 'process_exit_code', 'wall_seconds', 'counters')}, allow_nan=False), flush=True)
    return receipt['process_exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
