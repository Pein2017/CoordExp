"""Capture exactly two tied source-route readbacks for parent parity consumer.

This producer performs no generation and no comparison. It saves complete
single-position logits plus input identity tensors before the parent consumer
runs.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from probes.training_set_completion.untied_shared import load_model
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import exact_history_inputs, prepare_native_inputs

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration')
UNIT = ROOT / '2026-09-19-recurrence-spatial-source'
OUT = UNIT / 'final/technical-gate-v3'
WORKTREE = Path('/data/CoordExp/.worktrees/research-probes')
PANEL = ROOT / '2026-09-18-untied-highconfidence18-natural/panel.json'
SELECTION = ROOT / '2026-09-18-numerical-recurrence-feedback/selection.json'
STATE_ROOT = UNIT / 'final/corrected-pilot-v2/inputs/manifests'
PARENT_CONSUMER = WORKTREE / 'probes/training_set_completion/recurrence_spatial/parity_readback.py'
PRODUCER_V2 = UNIT / 'final/technical-gate-v2/run_gate.py'
TIED_STATE = STATE_ROOT / 'tied-417044-failure.json'
TRACE_INDEX = 119
TARGET_INDEX = 3
PAD = 0
OBJ_END = 151647


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path), 'size_bytes': path.stat().st_size}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def source_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    selection = load(SELECTION)
    boundary = next(x for x in selection['boundaries'] if x['id'] == 'tied-417044-failure')
    state = load(TIED_STATE)
    panel = load(PANEL)
    group = next(x for x in panel['groups'] if x['key'] == boundary['group'])
    target = next(x for x in group['cases'] if x['row_id'] == 'coco2017_train_000000417044')
    raw = load(Path(boundary['raw_path']))
    return boundary, state, {'panel': panel, 'group': group, 'target': target}, raw


def raw_prefix(raw: list[dict[str, Any]], end: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for item in raw:
        tokens = [int(v) for v in item['token_ids'][:end]]
        if len(tokens) < end:
            if 151645 not in tokens:
                tokens.append(151645)
            else:
                tokens = tokens[:tokens.index(151645) + 1]
            tokens.extend([PAD] * (end - len(tokens)))
        rows.append(tokens[:end])
    if len(rows) != 4 or any(len(row) != end for row in rows):
        raise RuntimeError('source histories are not exactly four equal-width 119-token prefixes')
    return rows


def snapshot(boundary: dict[str, Any], c_reuse_path: Path | None = None) -> dict[str, Any]:
    paths = [
        Path(__file__), PARENT_CONSUMER, PRODUCER_V2, PANEL, SELECTION, TIED_STATE,
        UNIT / 'final/corrected-pilot-v2/source-snapshot.json',
        UNIT / 'final/corrected-pilot-v2/pilot-receipt.json',
        WORKTREE / 'probes/training_set_completion/untied_shared.py',
        WORKTREE / 'src/inference/bound_requests.py',
        WORKTREE / 'src/qwen/native.py',
        Path(boundary['raw_path']), Path(boundary['trace_path']), Path(boundary['receipt_path']),
    ]
    if c_reuse_path is not None:
        paths.append(c_reuse_path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        path = path.resolve(strict=True)
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return {
        'schema': 'recurrence_spatial_source.technical_gate_v3_capture_snapshot.v1',
        'captured_before_model_calls': True,
        'model_calls_before_snapshot': 0,
        'frozen': {
            'model': 'tied',
            'state_id': 'tied-417044-failure',
            'source_row_end': TRACE_INDEX,
            'target_batch_index': TARGET_INDEX,
            'source_group_case_count': 4,
            'target_only_batch_size': 1,
            'dtype': 'fp32',
            'attention': 'sdpa',
            'logits_to_keep': 1,
            'no_generation': True,
            'expected_source_shape': [4, 1, 152670],
            'expected_target_shape': [1, 1, 152670],
            'comparison_position': -1,
            'tolerance': 2e-4,
        },
        'source_files': [bind(path) for path in unique],
    }


def capture(q: Any, native: Any, histories: list[list[int]], device: str, label: str) -> tuple[dict[str, Any], float]:
    prompts = [list(row) for row in native.prompt_token_ids]
    full_histories = [prompt + prefix for prompt, prefix in zip(prompts, histories, strict=True)]
    inputs = exact_history_inputs(q.model, native.inputs, full_histories, pad_token_id=PAD, logits_to_keep=1)
    if int(inputs['input_ids'].shape[0]) != len(histories):
        raise RuntimeError(f'{label}: batch cardinality drift')
    with torch.inference_mode():
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = q.model(**inputs)
        end.record()
        torch.cuda.synchronize(device)
        gpu_ms = float(start.elapsed_time(end))
    logits = output.logits.detach().float().cpu()
    expected = (len(histories), 1, 152670)
    if tuple(logits.shape) != expected:
        raise RuntimeError(f'{label}: logits shape {tuple(logits.shape)} != {expected}')
    position_ids = inputs.get('position_ids')
    if not isinstance(position_ids, torch.Tensor):
        raise RuntimeError(f'{label}: position_ids missing')
    capture_obj = {
        'schema': 'recurrence_spatial_source.full_position_capture.v1',
        'route': label,
        'logits': logits,
        'consumed_actions': TRACE_INDEX,
        'input_ids': inputs['input_ids'].detach().cpu(),
        'attention_mask': inputs['attention_mask'].detach().cpu(),
        'position_ids': position_ids.detach().cpu(),
        'prompt_token_ids': prompts,
        'media_sha256': list(native.media_sha256) if native.media_sha256 is not None else None,
        'image_grids': [list(x) for x in native.image_grids],
        'history_token_ids': [list(x) for x in histories],
        'history_token_digests': [digest(x) for x in histories],
        'input_shape': list(inputs['input_ids'].shape),
        'attention_shape': list(inputs['attention_mask'].shape),
        'position_shape': list(position_ids.shape),
        'logits_shape': list(logits.shape),
        'dtype': 'float32',
        'device': device,
        'logits_to_keep': 1,
        'generation_calls': 0,
        'gpu_ms': gpu_ms,
    }
    return capture_obj, gpu_ms


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    boundary, state, info, raw_payload = source_inputs()
    histories = raw_prefix(raw_payload['rows'], TRACE_INDEX)
    target_history = histories[TARGET_INDEX]
    c_reuse = OUT.parent / 'technical-gate-v2/c-qualification-reuse.json'
    snapshot_obj = snapshot(boundary, c_reuse if c_reuse.exists() else None)
    (OUT / 'capture-snapshot.json').write_text(json.dumps(snapshot_obj, indent=2) + '\n')

    config = dict(info['panel']['configs']['tied'])
    config['data'] = dict(input_jsonl=info['group']['input_jsonl'])
    device = 'cuda:0'
    started = time.monotonic()
    q, identity = load_model('tied', device)
    versions = {name: parameter._version for name, parameter in q.model.named_parameters()}
    requests, _ = build_bound_native_requests(q, config, info['group']['cases'])
    native_group = prepare_native_inputs(q.processor, requests, device=device, record_media_identity=True)
    target_requests, _ = build_bound_native_requests(q, config, [info['target']])
    native_one = prepare_native_inputs(q.processor, target_requests, device=device, record_media_identity=True)

    if tuple(native_group.prompt_token_ids[TARGET_INDEX]) != tuple(native_one.prompt_token_ids[0]):
        raise RuntimeError('target prompt identity mismatch before model calls')
    if native_group.image_grids[TARGET_INDEX] != native_one.image_grids[0]:
        raise RuntimeError('target image grid mismatch before model calls')
    if native_group.media_sha256 is None or native_one.media_sha256 is None or native_group.media_sha256[TARGET_INDEX] != native_one.media_sha256[0]:
        raise RuntimeError('target media identity mismatch before model calls')
    target_native = [int(v) for v in raw_payload['rows'][TARGET_INDEX]['token_ids']]
    if digest(target_native[:TRACE_INDEX]) != boundary['prefix_hash']:
        raise RuntimeError('target source prefix hash mismatch')
    if target_history != target_native[:TRACE_INDEX]:
        raise RuntimeError('target history differs from exact native prefix')

    source_capture, source_gpu_ms = capture(q, native_group, histories, device, 'source_batch4')
    # Save the first full capture before the second model call and before any comparison.
    torch.save(source_capture, OUT / 'source.pt')
    target_capture, target_gpu_ms = capture(q, native_one, [target_history], device, 'target_only_batch1')
    # Save the second full capture before any comparison or consumer import.
    torch.save(target_capture, OUT / 'target.pt')

    trace_payload = load(Path(boundary['trace_path']))
    trace_step = trace_payload['steps'][TRACE_INDEX]
    trace = {
        'schema': 'recurrence_spatial_source.saved_source_action_trace.v1',
        'action_index': TRACE_INDEX,
        'token_id': int(trace_step['raw_winners'][TARGET_INDEX]),
        'chosen_raw_logit': float(trace_step['chosen_raw_logits'][TARGET_INDEX]),
        'prefix_token_ids': target_native[:TRACE_INDEX],
        'source_trace_path': boundary['trace_path'],
        'source_trace_sha256': sha(Path(boundary['trace_path'])),
        'source_trace_step_fields': sorted(trace_step),
    }
    (OUT / 'trace.json').write_text(json.dumps(trace, indent=2) + '\n')

    current_versions = {name: parameter._version for name, parameter in q.model.named_parameters()}
    if current_versions != versions:
        raise RuntimeError('model parameters mutated during capture')
    elapsed = time.monotonic() - started
    receipt = {
        'schema': 'recurrence_spatial_source.technical_gate_v3_capture_receipt.v1',
        'status': 'captured_before_comparison',
        'model': 'tied',
        'state_id': 'tied-417044-failure',
        'source_row_end': TRACE_INDEX,
        'target_batch_index': TARGET_INDEX,
        'native_model_forwards': 2,
        'native_generation_calls': 0,
        'capture_files': {
            'source': bind(OUT / 'source.pt'),
            'target': bind(OUT / 'target.pt'),
            'trace': bind(OUT / 'trace.json'),
            'snapshot': bind(OUT / 'capture-snapshot.json'),
        },
        'identity': {
            'source_target_prompt_equal_target_only': True,
            'source_target_grid_equal_target_only': True,
            'source_target_media_equal_target_only': True,
            'source_target_prefix_digest': digest(target_history),
            'source_target_native_prefix_hash': boundary['prefix_hash'],
            'target_prompt_token_count_source_batch': len(source_capture['prompt_token_ids'][TARGET_INDEX]),
            'target_prompt_token_count_target_only': len(target_capture['prompt_token_ids'][0]),
            'source_image_grid': source_capture['image_grids'][TARGET_INDEX],
            'target_image_grid': target_capture['image_grids'][0],
            'source_media_sha256': source_capture['media_sha256'][TARGET_INDEX],
            'target_media_sha256': target_capture['media_sha256'][0],
        },
        'shapes': {
            'source_logits': source_capture['logits_shape'],
            'target_logits': target_capture['logits_shape'],
            'source_input_ids': source_capture['input_shape'],
            'target_input_ids': target_capture['input_shape'],
            'source_position_ids': source_capture['position_shape'],
            'target_position_ids': target_capture['position_shape'],
        },
        'gpu_ms': {'source_batch4': source_gpu_ms, 'target_only_batch1': target_gpu_ms, 'total': source_gpu_ms + target_gpu_ms},
        'elapsed_seconds_including_load_and_prepare': elapsed,
        'runtime_identity': identity,
        'consumer_pending': str(PARENT_CONSUMER),
        'comparison_not_run_by_producer': True,
    }
    (OUT / 'capture-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
