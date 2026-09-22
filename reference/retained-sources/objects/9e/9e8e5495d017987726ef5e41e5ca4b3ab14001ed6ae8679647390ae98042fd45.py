"""Four-cell exact-history next-token diagnostic for second-fit image 210457."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import signal
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

B = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum')
ROOT = B / 'second-fit-210457-stop-history-v1'
PREPARATION = B / 'second-fit-preparation-v1/manifest.json'
RECOVERY = B / 'second-fit-readback-recovery-v1'
SOURCE32 = B / 'first-fit-v1/readback-step-32.json'
IMAGE_ID, EOS, ROW_OPEN, HISTORY_LENGTH, WALL = 210457, 151645, 151646, 47, 600
SCHEMA = 'training_set_completion.second_fit_210457_stop_history.v1'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def binding(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    require(path.is_file(), f'not a file: {path}')
    return {'path': str(path), 'sha256': file_hash(path), 'size_bytes': path.stat().st_size}


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def publish(path: str | Path, value: Any) -> None:
    path = Path(path)
    require(not path.exists(), f'refusing overwrite: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value)
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    require(path.read_bytes() == data, f'publication readback: {path}')


def save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f'refusing overwrite: {path}')
    with path.open('xb') as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _row(path: Path) -> dict[str, Any]:
    value = read(path)
    return next(row for row in value['rows'] if int(row['image_id']) == IMAGE_ID)


def _adapter(step: int) -> Path:
    return B / 'second-fit-v1/training/checkpoints' / f'step-{step:05d}' / 'adapter'


def _score(values: np.ndarray) -> dict[str, Any]:
    require(values.ndim == 1 and values.shape[0] > ROW_OPEN and np.isfinite(values).all(), 'finite full-vocabulary logits')
    order = np.argsort(-values, kind='stable')[:10]
    eos, row = float(values[EOS]), float(values[ROW_OPEN])
    return {
        'eos_logit': eos,
        'object_ref_start_logit': row,
        'margin_eos_minus_object_ref_start': eos - row,
        'eos_rank': int(1 + np.sum(values > eos)),
        'object_ref_start_rank': int(1 + np.sum(values > row)),
        'top1_token_id': int(order[0]),
        'top10': [{'token_id': int(token), 'logit': float(values[token])} for token in order],
    }


def prepare(output: Path = ROOT) -> dict[str, Any]:
    from src.adapters.dora import inspect_dora_adapter_payload

    require(not (output / 'manifest.json').exists(), 'manifest collision')
    prepared = read(PREPARATION)
    record = next(route for route in prepared['routes'] if int(route['image_id']) == IMAGE_ID)
    second16_path = RECOVERY / 'readback-step-16.json'
    second32_path = RECOVERY / 'readback-step-32.json'
    step16 = _row(second16_path)
    clean = _row(SOURCE32)
    second32 = _row(second32_path)
    history16, history_clean = step16['generated_token_ids'][:HISTORY_LENGTH], clean['generated_token_ids'][:-1]
    require(len(step16['generated_token_ids']) == 3084 and step16['decode_stop_reason'] == 'length', 'second-fit step16 source route')
    require(len(clean['generated_token_ids']) == 48 and clean['generated_token_ids'][-1] == EOS and clean['decode_stop_reason'] == 'im_end', 'clean source route')
    require(second32['generated_token_ids'] == clean['generated_token_ids'], 'recovered second-fit step32 clean parity')
    require(len(history16) == len(history_clean) == HISTORY_LENGTH and EOS not in history16 and EOS not in history_clean, 'complete prefix without EOS')
    differences = [{'position': index, 'second_fit16_token_id': left, 'clean_token_id': right} for index, (left, right) in enumerate(zip(history16, history_clean)) if left != right]
    require(differences == [
        {'position': 35, 'second_fit16_token_id': 152295, 'clean_token_id': 152297},
        {'position': 36, 'second_fit16_token_id': 152272, 'clean_token_id': 152273},
    ], 'literal five-owner difference')
    models = []
    for step in (16, 32):
        adapter = _adapter(step)
        models.append({'step': step, 'adapter': inspect_dora_adapter_payload(adapter, prepared['model_config']['model']['base_model'])})
    value = {
        'schema': SCHEMA,
        'status': 'frozen_ready',
        'sources': {
            'second_fit_preparation': binding(PREPARATION),
            'second_fit_step16_readback': binding(second16_path),
            'first_fit_clean_step32_readback': binding(SOURCE32),
            'second_fit_step32_readback': binding(second32_path),
            'producer': binding(Path(__file__)),
        },
        'record': record,
        'models': models,
        'histories': [
            {'name': 'second_fit16_first5_rows', 'token_ids': history16, 'token_ids_sha256': digest(history16), 'source': {'readback': binding(second16_path), 'route_id': step16['route_id'], 'source_span': [0, HISTORY_LENGTH - 1]}},
            {'name': 'clean_source32_first5_rows_without_eos', 'token_ids': history_clean, 'token_ids_sha256': digest(history_clean), 'source': {'readback': binding(SOURCE32), 'route_id': clean['route_id'], 'source_span': [0, HISTORY_LENGTH - 1], 'excluded_eos_position': HISTORY_LENGTH}},
        ],
        'exact_history_difference': differences,
        'execution': {'physical_gpu': 0, 'cuda_visible_devices': '0', 'max_wall_seconds': WALL, 'model_loads': 2, 'score_cells': 4, 'natural_cells': 4, 'natural_budget_per_cell': 1, 'precision': 'fp32', 'attention': 'sdpa', 'empty_assistant_prefix': False},
        'policy': {'temperature': 0.0, 'top_p': 1.0, 'top_k': 0, 'repetition_penalty': 1.0, 'use_model_defaults': False},
        'interpretation_limit': 'The only intervention is the supplied literal complete-prefix token sequence. This does not identify an independent KV/cache cause or a natural full-completion effect.',
        'content_sha256': None,
    }
    value['content_sha256'] = digest({key: item for key, item in value.items() if key != 'content_sha256'})
    publish(output / 'manifest.json', value)
    return value


def validate(manifest: Mapping[str, Any], manifest_path: Path) -> None:
    require(manifest.get('schema') == SCHEMA and manifest.get('status') == 'frozen_ready', 'schema/status')
    require(manifest.get('content_sha256') == digest({key: item for key, item in manifest.items() if key != 'content_sha256'}), 'manifest hash')
    for source in manifest['sources'].values():
        require(binding(source['path']) == source, f'source changed: {source["path"]}')
    require([model['step'] for model in manifest['models']] == [16, 32], 'checkpoint set')
    require([history['name'] for history in manifest['histories']] == ['second_fit16_first5_rows', 'clean_source32_first5_rows_without_eos'], 'history set')
    require(all(len(history['token_ids']) == HISTORY_LENGTH and digest(history['token_ids']) == history['token_ids_sha256'] and EOS not in history['token_ids'] for history in manifest['histories']), 'history identity')
    require(manifest['exact_history_difference'] == [{'position': 35, 'second_fit16_token_id': 152295, 'clean_token_id': 152297}, {'position': 36, 'second_fit16_token_id': 152272, 'clean_token_id': 152273}], 'history difference')
    require(manifest['execution']['max_wall_seconds'] == WALL and manifest['execution']['score_cells'] == manifest['execution']['natural_cells'] == 4, 'execution bound')
    require(Path(manifest_path).resolve() == ROOT / 'manifest.json', 'manifest path')


def run(manifest_path: Path) -> None:
    import torch
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from probes.native_owner_scale.evaluation import _candidate_materialized_case
    from probes.source_rweak_row_cross.run import build_requests
    from src.adapters.dora import inspect_dora_adapter_payload
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import prepare_native_inputs, prepare_replay

    manifest = read(manifest_path)
    validate(manifest, manifest_path)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.device_count() == 1, 'GPU0 isolation')
    terminal_path = ROOT / 'terminal.json'
    require(not terminal_path.exists(), 'terminal collision')
    started, phase = time.monotonic(), 'preflight'
    counters = {'model_loads': 0, 'teacher_forced_cells': 0, 'natural_cells': 0, 'model_forwards': 0, 'image_forwards': 0}
    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('diagnostic wall bound')))
        signal.alarm(WALL)
        torch.cuda.set_device('cuda:0')
        torch.cuda.reset_peak_memory_stats()
        config0 = InferConfig.model_validate(manifest['record']['case'].get('model_config', manifest['record'].get('model_config', read(PREPARATION)['model_config'])))
        # `record` is an immutable training route; its model configuration is bound at the manifest top level source.
        config0 = InferConfig.model_validate(read(PREPARATION)['model_config'])
        case = _candidate_materialized_case(manifest['record']['case'], read(PREPARATION)['model_config'])
        for model_spec in manifest['models']:
            step = model_spec['step']
            phase = f'load_step{step}'
            adapter = Path(model_spec['adapter']['root'])
            require(inspect_dora_adapter_payload(adapter, config0.model.base_model) == model_spec['adapter'], 'adapter identity')
            config = checkpoint_config(config0, str(adapter))
            qwen, identity = load_policy(config, device=torch.device('cuda:0'))
            counters['model_loads'] += 1
            require(identity['model_identity']['adapter']['adapter_path'] == str(adapter) and identity['model_identity']['adapter']['merged_adapters'] == [], 'live adapter')
            require(identity['effective_settings']['observed_model_dtype']['parameter_dtype_names'] == ['torch.float32'] and identity['effective_settings']['observed_attn_implementation'] == 'sdpa', 'FP32 SDPA')
            publish(ROOT / f'model-step-{step:05d}.json', identity)
            qwen.model.eval()
            model = qwen.model
            forward_handle = model.register_forward_pre_hook(lambda *_: counters.__setitem__('model_forwards', counters['model_forwards'] + 1))
            visuals = [module for name, module in model.named_modules() if name.endswith('visual')]
            require(len(visuals) == 1, 'visual')
            image_handle = visuals[0].register_forward_pre_hook(lambda *_: counters.__setitem__('image_forwards', counters['image_forwards'] + 1))
            requests, _ = build_requests(qwen, read(PREPARATION)['model_config'], [case])
            batch = prepare_native_inputs(qwen.processor, requests, device='cuda:0', record_media_identity=True)
            record = manifest['record']
            require(list(batch.prompt_token_ids[0]) == record['prompt_token_ids'], 'prompt identity')
            require(batch.media_sha256[0] == record['image_identity']['executed_media_sha256'] and list(batch.image_grids[0]) == record['image_identity']['observed_image_grid_thw'], 'media/grid identity')
            policy = NativeGenerationPolicy(**manifest['policy'])
            for history in manifest['histories']:
                phase = f'step{step}_{history["name"]}'
                continuation = list(history['token_ids']) + [EOS]
                tick = time.monotonic()
                with torch.inference_mode():
                    replay = prepare_replay(model, batch.inputs, prompt_token_ids=record['prompt_token_ids'], continuation_token_ids=continuation)
                    require(replay.target_ids.tolist() == continuation, 'teacher-forced literal history')
                    aligned = replay.aligned_logits(model(**replay.inputs).logits)
                    require(aligned.shape[0] == HISTORY_LENGTH + 1, 'aligned target length')
                    values = aligned[-1].detach().float().cpu().numpy().copy()
                    del aligned, replay
                torch.cuda.synchronize()
                score = _score(values)
                logit_path = ROOT / 'logits' / f'step-{step:05d}-{history["name"]}.npy'
                save_npy(logit_path, values)
                counters['teacher_forced_cells'] += 1
                with torch.inference_mode():
                    generated, = generate_continuations(model, batch, extensions=[history['token_ids']], budgets=[1], eos_token_id=EOS, pad_token_id=qwen.tokenizer.pad_token_id, policy=policy, trace='none', seed=None)
                native_ids = list(generated.token_ids)
                require(len(native_ids) == 1, 'one-token natural cross-check')
                counters['natural_cells'] += 1
                row = {
                    'schema': SCHEMA + '.cell', 'checkpoint_step': step, 'history_name': history['name'], 'history_token_ids_sha256': history['token_ids_sha256'],
                    'next_logit_array': binding(logit_path), 'score': score,
                    'natural_next_token_id': native_ids[0], 'natural_stop_reason': generated.stop_reason,
                    'teacher_forced_top1_equals_natural_next': score['top1_token_id'] == native_ids[0],
                    'elapsed_seconds': time.monotonic() - tick,
                }
                publish(ROOT / 'cells' / f'step-{step:05d}-{history["name"]}.json', row)
            forward_handle.remove(); image_handle.remove()
            del batch, requests, model, qwen
            gc.collect(); torch.cuda.empty_cache()
        require(counters['model_loads'] == 2 and counters['teacher_forced_cells'] == counters['natural_cells'] == 4, 'cell denominator')
        cells = [read(ROOT / 'cells' / f'step-{step:05d}-{history["name"]}.json') for step in (16, 32) for history in manifest['histories']]
        result = {
            'schema': SCHEMA + '.result',
            'status': 'candidate_completed' if all(cell['teacher_forced_top1_equals_natural_next'] for cell in cells) else 'technical_score_generate_drift',
            'manifest': binding(manifest_path), 'cell_count': len(cells), 'cells': cells, 'counters': counters,
            'teacher_forced_natural_top1_agreement': all(cell['teacher_forced_top1_equals_natural_next'] for cell in cells),
            'interpretation_limit': manifest['interpretation_limit'],
        }
        publish(ROOT / 'result.json', result)
        terminal = {'schema': SCHEMA + '.terminal', 'status': 'completed', 'manifest': binding(manifest_path), 'result': binding(ROOT / 'result.json'), 'phase': 'complete', 'counters': counters, 'elapsed_seconds': time.monotonic() - started}
    except BaseException as error:
        terminal = {'schema': SCHEMA + '.terminal', 'status': 'failed', 'manifest': binding(manifest_path), 'phase': phase, 'counters': counters, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc(), 'elapsed_seconds': time.monotonic() - started}
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
        terminal.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0, peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        publish(terminal_path, terminal)


def verify(manifest_path: Path, output: Path) -> dict[str, Any]:
    manifest = read(manifest_path); validate(manifest, manifest_path)
    terminal, result = read(ROOT / 'terminal.json'), read(ROOT / 'result.json')
    require(terminal['status'] == 'completed' and result['status'] in ('candidate_completed', 'technical_score_generate_drift'), 'terminal/result')
    require(result['cell_count'] == 4 and result['counters']['model_loads'] == 2 and result['counters']['teacher_forced_cells'] == result['counters']['natural_cells'] == 4, 'result denominator')
    for cell in result['cells']:
        require(binding(cell['next_logit_array']['path']) == cell['next_logit_array'], 'logit artifact changed')
        observed = _score(np.load(cell['next_logit_array']['path'], allow_pickle=False))
        require(observed == cell['score'], 'score replay')
    value = {'schema': SCHEMA + '.verification', 'status': 'candidate_cpu_verified', 'manifest': binding(manifest_path), 'terminal': binding(ROOT / 'terminal.json'), 'result': binding(ROOT / 'result.json'), 'cell_count': 4, 'exact_score_native_agreement': result['teacher_forced_natural_top1_agreement']}
    publish(output, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run', 'verify'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(ROOT), indent=2))
    elif args.command == 'run':
        require(args.manifest is not None, 'manifest required'); run(args.manifest)
    else:
        require(args.manifest is not None and args.output is not None, 'verify args'); print(json.dumps(verify(args.manifest, args.output), indent=2))


if __name__ == '__main__':
    main()
