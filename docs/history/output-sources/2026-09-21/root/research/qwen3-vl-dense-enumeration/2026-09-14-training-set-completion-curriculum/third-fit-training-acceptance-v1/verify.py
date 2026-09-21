"""Read-only audit of the bounded third-fit masked-route training artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open

WORKTREE = Path('/data/CoordExp/.worktrees/research-probes')
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from probes.training_set_completion.training import inspect_dora_adapter_payload, validate_manifest
from src.qwen.special_token_embeddings import inspect_special_token_embedding_delta_payload

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum')
TRAINING = ROOT / 'third-fit-v1/training'
MANIFEST = ROOT / 'third-fit-preparation-v1/manifest.json'
OUT = ROOT / 'third-fit-training-acceptance-v1'
CHECKPOINT_STEPS = (16, 32, 64)
DOSE_STEPS = (1, 16, 32, 64)
EXPECTED_TENSORS = 588
EXPECTED_IMAGES = 11
EXPECTED_UPDATES = 64


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {'path': str(path), 'sha256': file_hash(path), 'size_bytes': path.stat().st_size}


def publish(path: Path, value: Any) -> None:
    require(not path.exists(), f'output collision: {path}')
    content = canonical(value)
    with path.open('xb') as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    require(path.read_bytes() == content, f'write readback: {path}')


def finite_number(value: Any, name: str) -> float:
    value = float(value)
    require(math.isfinite(value), f'non-finite {name}')
    return value


def adapter_finite(adapter: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    observed = inspect_dora_adapter_payload(adapter, expected['semantic_identity']['base_model_name_or_path'])
    require(observed == expected, 'saved adapter identity')
    tensor_path = adapter / 'adapter_model.safetensors'
    finite = 0
    with safe_open(str(tensor_path), framework='pt', device='cpu') as payload:
        keys = list(payload.keys())
        require(len(keys) == EXPECTED_TENSORS, 'adapter tensor count')
        for key in keys:
            tensor = payload.get_tensor(key)
            require(bool(torch.isfinite(tensor).all()), f'non-finite adapter tensor: {key}')
            finite += 1
    return {'adapter': observed, 'finite_tensor_count': finite}


def checkpoint_summary(step: int, manifest: Mapping[str, Any], expected_layout: list[dict[str, Any]]) -> dict[str, Any]:
    root = TRAINING / 'checkpoints' / f'step-{step:05d}'
    state_path = root / 'state.pt'
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    require(state['schema'].endswith('.checkpoint.v1') and state['step'] == step, 'checkpoint step/schema')
    require(state['manifest'] == binding(MANIFEST), 'checkpoint manifest binding')
    require(state['source_adapter'] == manifest['source_adapter'], 'checkpoint source adapter')
    require(state['optimizer'] == manifest['optimizer'], 'checkpoint optimizer identity')
    require(state['parameter_layout'] == expected_layout and len(expected_layout) == EXPECTED_TENSORS, 'checkpoint parameter layout')
    adapter = adapter_finite(root / 'adapter', state['saved_adapter'])
    optimizer = state['optimizer_state_dict']
    parameter_state = optimizer['state']
    groups = optimizer['param_groups']
    require(len(groups) == 1 and len(groups[0]['params']) == EXPECTED_TENSORS and len(parameter_state) == EXPECTED_TENSORS, 'optimizer tensor state count')
    require({int(item) for item in groups[0]['params']} == set(range(EXPECTED_TENSORS)), 'optimizer parameter IDs')
    step_values: set[int] = set()
    finite_state_tensors = 0
    for parameter_id, value in parameter_state.items():
        require(int(parameter_id) in range(EXPECTED_TENSORS), 'optimizer parameter outside layout')
        require({'step', 'exp_avg', 'exp_avg_sq'} <= set(value), 'optimizer AdamW state fields')
        step_value = int(value['step'].item() if isinstance(value['step'], torch.Tensor) else value['step'])
        step_values.add(step_value)
        for name in ('exp_avg', 'exp_avg_sq'):
            require(bool(torch.isfinite(value[name]).all()), f'non-finite optimizer {name}')
            finite_state_tensors += 1
    require(step_values == {step}, 'optimizer step counters')
    return {'step': step, 'state': binding(state_path), 'adapter_fingerprint': adapter['adapter']['fingerprint'], 'adapter_tensor_count': adapter['finite_tensor_count'], 'optimizer_state_count': len(parameter_state), 'optimizer_finite_tensors': finite_state_tensors, 'optimizer_step_values': sorted(step_values)}


def compact_adapter(identity: Mapping[str, Any]) -> dict[str, Any]:
    semantic = identity['semantic_identity']
    return {'root': identity['root'], 'fingerprint': identity['fingerprint'], 'files': identity['files'], 'semantic_identity': {key: semantic[key] for key in ('base_model_name_or_path', 'lora_A_count', 'lora_B_count', 'lora_magnitude_vector_count', 'tensor_key_count', 'use_dora', 'r')}}


def compact_embedding(identity: Mapping[str, Any]) -> dict[str, Any]:
    semantic = identity['semantic_identity']
    return {'root': identity['root'], 'fingerprint': identity['fingerprint'], 'files': identity['files'], 'semantic_identity': {'base_model_path': semantic['base_model_path'], 'base_config_sha256': semantic['base_config_sha256'], 'tokenizer_sha256': semantic['tokenizer_sha256'], 'tensor_shape': semantic['tensor_shape'], 'tensor_dtype': semantic['tensor_dtype'], 'token_count': len(semantic['token_ids']), 'semantics': semantic['semantics']}}


def update_summary(path: Path) -> dict[str, Any]:
    update = json.loads(path.read_text())
    step = int(update['step'])
    require(update['schema'].endswith('.update.v1') and update['image_count'] == EXPECTED_IMAGES and update['forwards'] == step * EXPECTED_IMAGES, 'update counters')
    require(len(update['routes']) == EXPECTED_IMAGES, 'update route count')
    objective = finite_number(update['objective_mean_over_images'], 'objective')
    norm = finite_number(update['gradient_norm_before_clip'], 'gradient norm')
    route_total = [finite_number(row['total'], 'route total') for row in update['routes']]
    route_ce = [finite_number(row['ce'], 'route ce') for row in update['routes']]
    route_hinge = [finite_number(row['raw_axis_validity_hinge'], 'route hinge') for row in update['routes']]
    require(abs(sum(route_total) / EXPECTED_IMAGES - objective) < 2e-6, 'objective reduction')
    return {'step': step, 'logical_image_forwards': int(update['forwards']), 'objective_mean_over_images': objective, 'mean_masked_ce': sum(route_ce) / EXPECTED_IMAGES, 'mean_raw_axis_validity_hinge': sum(route_hinge) / EXPECTED_IMAGES, 'gradient_norm_before_clip': norm}


def audit(*, publish_outputs: bool = True) -> dict[str, Any]:
    manifest_raw = json.loads(MANIFEST.read_text())
    manifest = validate_manifest(manifest_raw)
    require(manifest_raw == manifest and manifest['runtime']['updates'] == EXPECTED_UPDATES and manifest['runtime']['max_model_forwards'] == EXPECTED_UPDATES * EXPECTED_IMAGES, 'manifest runtime')
    require(manifest['runtime']['checkpoint_steps'] == list(CHECKPOINT_STEPS), 'manifest checkpoint schedule')
    require(manifest['source_adapter']['root'].endswith('first-fit-v1/training/checkpoints/step-00016/adapter'), 'first-fit16 parent adapter')
    terminal = json.loads((TRAINING / 'terminal.json').read_text())
    require(terminal['status'] == 'completed' and terminal['optimizer_mode'] == 'fresh', 'terminal completion/fresh optimizer')
    require(terminal['manifest'] == binding(MANIFEST), 'terminal manifest binding')
    require(terminal['updates'] == EXPECTED_UPDATES and terminal['model_forwards'] == EXPECTED_UPDATES * EXPECTED_IMAGES, 'terminal counters')
    require(len(terminal['trainable_surface']) == EXPECTED_TENSORS, 'terminal trainable surface')
    require(terminal['loaded_model']['effective_settings']['observed_attn_implementation'] == 'sdpa', 'runtime attention implementation')
    require(terminal['loaded_model']['effective_settings']['observed_model_dtype']['parameter_dtype_names'] == ['torch.float32'], 'runtime dtype')
    model_identity = terminal['loaded_model']['model_identity']
    require(model_identity['base']['path'] == manifest['model_config']['model']['base_model'], 'frozen base path')
    require(model_identity['adapter']['adapter_path'] == manifest['source_adapter']['root'], 'loaded parent adapter path')
    embedding = inspect_special_token_embedding_delta_payload(manifest['model_config']['embedding_delta']['path'], manifest['model_config']['model']['base_model'])
    require(model_identity['embedding_delta']['identity']['delta_path'] == embedding['root'], 'loaded embedding delta path')
    require(model_identity['embedding_delta']['identity']['metadata'] == embedding['semantic_identity'], 'loaded embedding delta identity')

    updates = [update_summary(TRAINING / 'updates' / f'step-{step:05d}.json') for step in range(1, EXPECTED_UPDATES + 1)]
    require([row['step'] for row in updates] == list(range(1, EXPECTED_UPDATES + 1)), 'complete update sequence')
    checkpoints = [checkpoint_summary(step, manifest, terminal['trainable_surface']) for step in CHECKPOINT_STEPS]
    terminal_steps = [int(item['step']) for item in terminal['checkpoints']]
    require(terminal_steps == list(CHECKPOINT_STEPS), 'terminal checkpoint schedule')
    for checkpoint, terminal_checkpoint in zip(checkpoints, terminal['checkpoints'], strict=True):
        require(checkpoint['state'] == terminal_checkpoint['state'] and checkpoint['adapter_fingerprint'] == terminal_checkpoint['adapter']['fingerprint'], 'terminal checkpoint binding')

    curve = [next(row for row in updates if row['step'] == step) for step in DOSE_STEPS]
    observation = {'objective_change_step1_to_step64': curve[-1]['objective_mean_over_images'] - curve[0]['objective_mean_over_images'], 'objective_change_step32_to_step64': curve[-1]['objective_mean_over_images'] - curve[-2]['objective_mean_over_images'], 'optimization_observation': 'objective continues declining with no observed plateau; convergence and native generalization are unestablished'}
    curve_receipt = {'schema': 'third_fit.optimization_curve.v1', 'status': 'candidate_ready', 'training_terminal': binding(TRAINING / 'terminal.json'), 'manifest': binding(MANIFEST), 'selected_doses': curve, 'gradient_norm_range': {'min': min(row['gradient_norm_before_clip'] for row in updates), 'max': max(row['gradient_norm_before_clip'] for row in updates), 'all_finite': True}, 'observation': observation}
    if publish_outputs:
        publish(OUT / 'optimization-curve.json', curve_receipt)
    else:
        require(json.loads((OUT / 'optimization-curve.json').read_text()) == curve_receipt, 'existing curve differs from replay')
    return {'schema': 'third_fit.training_acceptance_audit.v1', 'status': 'candidate_ready', 'scope': 'execution audit only; no native readback or scientific-quality claim', 'manifest': binding(MANIFEST), 'training_terminal': binding(TRAINING / 'terminal.json'), 'runtime': {'updates': EXPECTED_UPDATES, 'images': EXPECTED_IMAGES, 'logical_image_forwards': EXPECTED_UPDATES * EXPECTED_IMAGES, 'optimizer_mode': terminal['optimizer_mode'], 'checkpoint_steps': list(CHECKPOINT_STEPS)}, 'initialization': {'parent_adapter': compact_adapter(manifest['source_adapter']), 'frozen_base_path': manifest['model_config']['model']['base_model'], 'embedding_delta': compact_embedding(embedding), 'observed_attention': 'sdpa', 'observed_dtype': 'torch.float32'}, 'checkpoints': checkpoints, 'optimization_curve': binding(OUT / 'optimization-curve.json'), 'observation': observation, 'producer': binding(Path(__file__))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-existing', action='store_true', help='replay the read-only audit and compare the published curve')
    args = parser.parse_args()
    if args.verify_existing:
        require((OUT / 'receipt.json').exists() and (OUT / 'optimization-curve.json').exists(), 'published audit is absent')
        receipt = audit(publish_outputs=False)
        published = json.loads((OUT / 'receipt.json').read_text())
        require(receipt == published, 'existing receipt differs from replay')
        print(json.dumps({'status': 'verified_existing', 'receipt': binding(OUT / 'receipt.json')}, sort_keys=True))
        return
    require(not (OUT / 'receipt.json').exists() and not (OUT / 'optimization-curve.json').exists(), 'audit output exists')
    receipt = audit()
    publish(OUT / 'receipt.json', receipt)


if __name__ == '__main__':
    main()
