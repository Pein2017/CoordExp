"""Narrow, explicit bridge from a frozen training manifest to a runtime extension."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from probes.training_set_completion import training

B = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum')
OLD_MANIFEST = B / 'third-fit-preparation-v1/manifest.json'
RESUME = B / 'third-fit-v1/training/checkpoints/step-00064'
ROOT = B / 'fourth-fit-preparation-v1'
ALLOWED_RUNTIME_FIELDS = frozenset({'updates', 'checkpoint_steps', 'wall_seconds', 'max_model_forwards'})
EXPECTED_START_STEP = 64
EXPECTED_TENSORS = 588
EXPECTED_IMAGES = 11


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def binding(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': file_hash(path), 'size_bytes': path.stat().st_size}


def publish(path: Path, value: Any) -> None:
    require(not path.exists(), f'output collision: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical(value)
    with path.open('xb') as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    require(path.read_bytes() == content, f'write readback: {path}')


def _state(path: Path) -> dict[str, Any]:
    return torch.load(path / 'state.pt', map_location='cpu', weights_only=False)


def _rng_identity(state: Mapping[str, Any]) -> dict[str, str]:
    return {'torch_rng_state_sha256': hashlib.sha256(bytes(state['torch_rng_state'].tolist())).hexdigest(), 'python_random_state_sha256': hashlib.sha256(pickle.dumps(state['python_random_state'], protocol=4)).hexdigest()}


def _optimizer_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    values = state['optimizer_state_dict']['state']
    require(len(values) == EXPECTED_TENSORS, 'predecessor optimizer state count')
    steps = set()
    for item in values.values():
        require({'step', 'exp_avg', 'exp_avg_sq'} <= set(item), 'predecessor AdamW fields')
        value = item['step']
        steps.add(int(value.item() if isinstance(value, torch.Tensor) else value))
    require(steps == {EXPECTED_START_STEP}, 'predecessor AdamW counters')
    return {'state_count': len(values), 'step_values': sorted(steps), **_rng_identity(state)}


def _without_continuation(value: Mapping[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(dict(value))
    copied.pop('content_sha256', None)
    copied.pop('continuation', None)
    return copied


def _validate_extension(old: Mapping[str, Any], new: Mapping[str, Any]) -> None:
    old_core, new_core = _without_continuation(old), _without_continuation(new)
    require(set(new['sources']) == {'reviewed_routes', 'producer'} and new['sources'] == old['sources'], 'source bindings changed')
    require(new_core.keys() == old_core.keys(), 'unlisted top-level continuation change')
    for key in old_core:
        if key == 'runtime':
            continue
        require(new_core[key] == old_core[key], f'continuation changed frozen field: {key}')
    old_runtime, new_runtime = old_core['runtime'], new_core['runtime']
    require(set(new_runtime) == set(old_runtime), 'unlisted runtime key')
    changed = {key for key in new_runtime if new_runtime[key] != old_runtime[key]}
    require(changed <= ALLOWED_RUNTIME_FIELDS and changed, 'runtime extension changed unlisted field')
    require(new_runtime['eos_token_id'] == old_runtime['eos_token_id'], 'EOS changed')
    require(new_runtime['updates'] == 256 and new_runtime['checkpoint_steps'] == [128, 192, 256] and new_runtime['wall_seconds'] == 4500, 'declared continuation runtime')
    require(new_runtime['max_model_forwards'] == new_runtime['updates'] * len(new['routes']) == 2816, 'cumulative runtime forward bound')
    provenance = new.get('continuation')
    require(isinstance(provenance, Mapping) and set(provenance) == {'schema', 'predecessor_manifest', 'predecessor_checkpoint', 'predecessor_state', 'wrapper', 'allowed_runtime_fields', 'remaining_logical_image_forwards', 'cumulative_runtime_forward_bound'}, 'continuation provenance fields')
    require(provenance['predecessor_manifest'] == binding(OLD_MANIFEST), 'predecessor manifest provenance')
    require(provenance['predecessor_checkpoint'] == str(RESUME.resolve()) and provenance['predecessor_state'] == binding(RESUME / 'state.pt'), 'predecessor checkpoint provenance')
    require(provenance['wrapper'] == binding(Path(__file__)), 'wrapper provenance')
    require(provenance['allowed_runtime_fields'] == sorted(ALLOWED_RUNTIME_FIELDS), 'allowed runtime provenance')
    require(provenance['remaining_logical_image_forwards'] == (256 - EXPECTED_START_STEP) * EXPECTED_IMAGES == 2112, 'remaining forwards provenance')
    require(provenance['cumulative_runtime_forward_bound'] == 2816, 'cumulative forwards provenance')


def validate_predecessor(old: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    require(state['schema'] == f'{training.SCHEMA}.checkpoint.v1' and state['step'] == EXPECTED_START_STEP, 'wrong predecessor checkpoint')
    require(state['manifest'] == binding(OLD_MANIFEST), 'old state manifest binding')
    require(state['source_adapter'] == old['source_adapter'] and state['optimizer'] == old['optimizer'], 'predecessor source/optimizer identity')
    observed_adapter = training.inspect_dora_adapter_payload(RESUME / 'adapter', old['model_config']['model']['base_model'])
    require(observed_adapter == state['saved_adapter'], 'predecessor adapter changed')
    layout = state['parameter_layout']
    require(len(layout) == EXPECTED_TENSORS and all(isinstance(item.get('name'), str) and item.get('numel', 0) > 0 for item in layout), 'predecessor DoRA layout')
    return {'checkpoint_step': EXPECTED_START_STEP, 'parameter_layout_count': len(layout), 'optimizer': _optimizer_summary(state)}


def build_manifest(*, old_path: Path = OLD_MANIFEST, resume: Path = RESUME) -> dict[str, Any]:
    require(old_path.resolve() == OLD_MANIFEST.resolve() and resume.resolve() == RESUME.resolve(), 'only declared predecessor is supported')
    old = training.validate_manifest(json.loads(old_path.read_text()))
    state = _state(resume)
    predecessor = validate_predecessor(old, state)
    new = copy.deepcopy(old)
    new['runtime'] = {**old['runtime'], 'updates': 256, 'checkpoint_steps': [128, 192, 256], 'wall_seconds': 4500, 'max_model_forwards': 2816}
    new['continuation'] = {'schema': 'training_set_completion.continuation.v1', 'predecessor_manifest': binding(old_path), 'predecessor_checkpoint': str(resume.resolve()), 'predecessor_state': binding(resume / 'state.pt'), 'wrapper': binding(Path(__file__)), 'allowed_runtime_fields': sorted(ALLOWED_RUNTIME_FIELDS), 'remaining_logical_image_forwards': 2112, 'cumulative_runtime_forward_bound': 2816}
    new.pop('content_sha256')
    new['content_sha256'] = training.digest(new)
    training.validate_manifest(new)
    _validate_extension(old, new)
    return new


def prepare(*, output: Path = ROOT) -> dict[str, Any]:
    require(not output.exists() or not any(output.iterdir()), 'continuation output exists')
    manifest = build_manifest()
    state = _state(RESUME)
    predecessor = validate_predecessor(training.validate_manifest(json.loads(OLD_MANIFEST.read_text())), state)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    publish(manifest_path, manifest)
    receipt = {'schema': 'training_set_completion.continuation_preparation.v1', 'status': 'candidate_ready', 'manifest': binding(manifest_path), 'predecessor': {'manifest': binding(OLD_MANIFEST), 'checkpoint': str(RESUME), 'state': binding(RESUME / 'state.pt'), **predecessor}, 'runtime': manifest['runtime'], 'cost_estimate': {'observed_seconds_per_update': 1160.6090087592602 / 64, 'remaining_updates': 192, 'estimated_remaining_seconds': (1160.6090087592602 / 64) * 192, 'wall_seconds': 4500}, 'launch': 'not authorized by this producer; root owns launch'}
    publish(output / 'receipt.json', receipt)
    return receipt


def run(manifest_path: Path, *, output: Path, device: str) -> dict[str, Any]:
    """Invoke unchanged training.run while retaining predecessor restore identity."""
    old = training.validate_manifest(json.loads(OLD_MANIFEST.read_text()))
    manifest = training.validate_manifest(json.loads(manifest_path.read_text()))
    _validate_extension(old, manifest)
    state = _state(RESUME)
    validate_predecessor(old, state)
    original_restore = training._restore

    def restore_predecessor(resume: Path, *, manifest_path: Path, manifest: Mapping[str, Any], optimizer: torch.optim.Optimizer, named: Sequence[tuple[str, torch.nn.Parameter]]) -> int:
        require(resume.resolve() == RESUME.resolve(), 'wrong resume checkpoint')
        _validate_extension(old, manifest)
        # Original restore remains responsible for adapter, optimizer, layout, and RNG checks.
        return original_restore(resume, manifest_path=OLD_MANIFEST, manifest=old, optimizer=optimizer, named=named)

    try:
        training._restore = restore_predecessor
        return training.run(manifest_path, output=output, device=device, resume=RESUME)
    finally:
        training._restore = original_restore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--output', type=Path, default=ROOT)
    runner = sub.add_parser('run')
    runner.add_argument('--manifest', type=Path, default=ROOT / 'manifest.json')
    runner.add_argument('--output', type=Path, required=True)
    runner.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(output=args.output)
    else:
        run(args.manifest, output=args.output, device=args.device)


if __name__ == '__main__':
    main()
