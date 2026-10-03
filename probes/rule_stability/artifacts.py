"""Exclusive publication and source/checkpoint readback for this finite unit."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNIT = ROOT / 'research/experiments/2026-10-03-rule-stability-iou90'
OUTPUT = ROOT / 'outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90'
ANCHOR = Path('/data/CoordExp/outputs/shared/checkpoints/untied-axis001-step2444/payload')
ANCHOR_DIGEST = '3b168b98f23f5e42b00b6aa7ad8ca5438767bcb8c05f4cc4ce97087d800e0403'
LEAD = '01a1016a-c440-77e2-b258-f3e8f860ede7'
SCHEMA = 'rule-stability-iou90-v1'


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    """Publish a fully serialized file exclusively, never replace existing evidence."""
    path = Path(path)
    content = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.partial-{os.getpid()}')
    try:
        with temporary.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def source_paths():
    # Include this producer and every new module, plus maintained import owners.
    probes = ['probes/__init__.py', 'probes/iterative_positive.py',
              'probes/hidden_human_recovery.py', 'probes/rollout_row_credit.py',
              'probes/online_row_credit.py', 'probes/owner_exchange.py']
    probes += [str(p.relative_to(ROOT)) for p in (ROOT / 'probes/rule_stability').glob('*.py')]
    probes += [str(p.relative_to(ROOT)) for p in (ROOT / 'probes/full_label_fit').glob('*.py')]
    return sorted(set(p for p in probes if (ROOT / p).is_file()) |
                  {str(p.relative_to(ROOT)) for p in (ROOT / 'src').rglob('*.py')})


def candidate_source():
    """CPU candidate identity permits concurrent records; native uses clean binding."""
    paths = source_paths()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(['git', 'diff', 'HEAD', '--', *paths], cwd=ROOT)
    return dict(kind='cpu_candidate', commit=revision, diff_sha256=hashlib.sha256(diff).hexdigest(),
                files={p: digest(ROOT / p) for p in paths})


def verify_candidate_source(source):
    if set(source['files']) != set(source_paths()):
        raise ValueError('source manifest does not cover the current module closure')
    for path, expected in source['files'].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'stale source: {path}')


def verify_anchor(path=ANCHOR, *, hash_payload=True):
    path = Path(path)
    manifest = load(path / 'inference_payload_manifest.json')
    if (manifest.get('schema') != 'coordexp-infras-inference-checkpoint-payload-manifest'
            or manifest.get('schema_version') != 1 or manifest.get('aggregate_digest') != ANCHOR_DIGEST):
        raise ValueError('anchor manifest identity/schema differs')
    for section in ('adapter', 'special_token_embedding_delta'):
        block = manifest[section]
        if block['status'] != 'present':
            raise ValueError('anchor component unavailable')
        for item in block['files']:
            file = path / block['relative_root'] / item['relative_path']
            if not file.resolve().is_relative_to(path.resolve()):
                raise ValueError('anchor file escapes payload')
            if file.stat().st_size != item['size_bytes'] or (hash_payload and digest(file) != item['sha256']):
                raise ValueError('anchor payload changed')
    return dict(path=str(path), aggregate_digest=ANCHOR_DIGEST,
                manifest_sha256=digest(path / 'inference_payload_manifest.json'))


def file_manifest(directory, excluded=()):
    directory = Path(directory)
    return {str(p.relative_to(directory)): dict(sha256=digest(p), size_bytes=p.stat().st_size)
            for p in sorted(directory.rglob('*')) if p.is_file() and str(p.relative_to(directory)) not in excluded}


def verify_manifest(directory, files, *, excluded=()):
    directory = Path(directory)
    observed = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} - set(excluded)
    if observed != set(files):
        raise ValueError('artifact inventory differs')
    for name, identity in files.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('artifact path escapes directory')
        if path.stat().st_size != identity['size_bytes'] or digest(path) != identity['sha256']:
            raise ValueError(f'artifact changed: {name}')


def seal_checkpoint(directory, *, arm, version, engine, parameter_schema):
    write(Path(directory) / 'checkpoint.json', dict(schema=SCHEMA, arm=arm, version=version,
        engine=engine, optimizer_updates=version, optimizer_continuous=True,
        parameter_schema=parameter_schema, files=file_manifest(directory)))


def checkpoint_readback(directory, *, arm, version, engine):
    """Inspect new checkpoint bytes, optimizer continuity and export schema on CPU."""
    import torch
    from safetensors import safe_open
    directory = Path(directory)
    meta = load(directory / 'checkpoint.json')
    expected = dict(schema=SCHEMA, arm=arm, version=version, engine=engine,
                    optimizer_updates=version, optimizer_continuous=True)
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError('checkpoint schema/arm/version drift')
    verify_manifest(directory, meta['files'], excluded=('checkpoint.json',))
    with safe_open(str(directory / 'special_token_embeddings/special_token_embeddings.safetensors'),
                   framework='pt', device='cpu') as handle:
        if set(handle.keys()) != {'input_embed_delta', 'output_embed_delta'}:
            raise ValueError('checkpoint lost independent deltas')
        tensors = {k: (handle.get_slice(k).get_shape(), handle.get_slice(k).get_dtype()) for k in handle.keys()}
    if len({tuple(v[0]) for v in tensors.values()}) != 1 or any(v[1] != 'F32' for v in tensors.values()):
        raise ValueError('delta shape/dtype drift')
    embedding = load(directory / 'special_token_embeddings/special_token_embeddings.json')
    if embedding['tie_word_embeddings'] is not False:
        raise ValueError('checkpoint retied independent deltas')
    if (embedding.get('tensor_dtype') != 'float32' or
            any(value[0] != embedding.get('tensor_shape') for value in tensors.values())):
        raise ValueError('delta metadata disagrees with payload shape/dtype')
    for tensor, role in [('input_embed_delta', 'input_delta'), ('output_embed_delta', 'output_delta')]:
        entries = [spec for spec in meta['parameter_schema'] if spec.get('role') == role]
        if (len(entries) != 1 or entries[0]['shape'] != tensors[tensor][0] or
                entries[0]['dtype'] != 'torch.float32'):
            raise ValueError('delta parameter tensor shape/dtype disagrees with export')
    optimizer = torch.load(directory / 'optimizer.pt', map_location='cpu', weights_only=True)
    if optimizer['completed_updates'] != version or optimizer['arm'] != arm:
        raise ValueError('optimizer continuity drift')
    state = optimizer['state_dict']
    if [g['lr'] for g in state['param_groups']] != [1e-5, 5e-6, 5e-6]:
        raise ValueError('optimizer group learning rates differ')
    ids = [i for group in state['param_groups'] for i in group['params']]
    if len(ids) != len(meta['parameter_schema']) or len(set(ids)) != len(ids):
        raise ValueError('optimizer parameter schema differs')
    if version and (set(state['state']) != set(ids) or
                    any(int(s['step']) != version for s in state['state'].values())):
        raise ValueError('optimizer steps differ from checkpoint version')
    for index, spec in zip(ids, meta['parameter_schema'], strict=True):
        if version and any(list(state['state'][index][key].shape) != spec['shape']
                           for key in ('exp_avg', 'exp_avg_sq')):
            raise ValueError(f"optimizer tensor shape differs: {spec['name']}")
    with safe_open(str(directory / 'adapter/adapter_model.safetensors'), framework='pt', device='cpu') as handle:
        keys = list(handle.keys())
        if engine == 'native' and (len(keys) != 588 or sum('lora_magnitude' in k for k in keys) != 196):
            raise ValueError('adapter export schema differs')
    return dict(status='complete', version=version, engine=engine, files=len(meta['files']),
                bytes=sum(x['size_bytes'] for x in meta['files'].values()))


def validate_packet(packet, output, *, mode):
    if mode not in ('qualification', 'primary'):
        raise ValueError('unsupported native release mode')
    expected_updates = 16 if mode == 'primary' else 1
    if (packet.get('schema') != SCHEMA or packet.get('released') is not True
            or packet.get('lead_thread') != LEAD or packet.get('mode') != mode
            or packet.get('arm') not in ('A', 'B') or packet.get('updates') != expected_updates
            or packet.get('world_size') != 8 or packet.get('output') != str(Path(output).resolve())
            or packet.get('anchor', {}).get('aggregate_digest') != ANCHOR_DIGEST):
        raise ValueError('exact source-bound lead release absent or mismatched')
    if not Path(output).resolve().is_relative_to(OUTPUT):
        raise ValueError('native output outside unit ownership')
    if packet.get('protocol') != dict(path=str((UNIT / 'unit.md').relative_to(ROOT)), sha256=digest(UNIT / 'unit.md')):
        raise ValueError('frozen protocol changed')
    if packet.get('anchor', {}).get('path') != str(ANCHOR):
        raise ValueError('anchor path differs')
    from src.artifacts.git_identity import verify_source_identity
    verify_source_identity(packet['source'], required_paths=source_paths(), root=ROOT)
    if packet['frozen'] != dict(horizon=3084, coordinate_norm='median', images=18, labels=570,
                               sample_temperature=1, duplicate_comparator='>', duplicate_iou=.9):
        raise ValueError('packet changed the frozen experiment')
    if packet.get('retry') != 'no_automatic_relaunch' or not packet.get('operational_observation_seconds'):
        raise ValueError('packet needs retry/observation ownership')
