"""Authenticate each cache and compare complete decoded and pickle contents."""
from __future__ import annotations

from collections import Counter
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import itertools
import json
import pickletools
from pathlib import Path, PurePath
import struct
import sys

import torch

sys.path.insert(0, str(Path.cwd()))
from src.training.pack_cache import load_all_micro_steps_from_cache, load_cache_manifest

ROOT = Path('/data/CoordExp/outputs/infra_base/optimization-20260912')
OUT = ROOT / 'verification/packing'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def emit(digest, payload):
    digest.update(len(payload).to_bytes(8, 'little'))
    digest.update(payload)


def scalar_bytes(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, float):
        return struct.pack('>d', value)
    return repr(value).encode()


def compare_decoded(left, right):
    digest = hashlib.sha256()
    counts = Counter()
    left_objects, right_objects = {}, {}
    left_storages, right_storages = {}, {}

    def visit(a, b, path):
        require(type(a) is type(b), f'{path}: type mismatch {type(a)} != {type(b)}')
        typename = f'{type(a).__module__}.{type(a).__qualname__}'
        emit(digest, typename.encode())
        counts[typename] += 1
        if a is None or isinstance(a, (str, int, float, bool, bytes)):
            av, bv = scalar_bytes(a), scalar_bytes(b)
            require(av == bv, f'{path}: scalar differs')
            emit(digest, av)
            return
        if isinstance(a, (Enum, torch.device)):
            require(a == b, f'{path}: enum/device differs')
            emit(digest, repr(a).encode())
            return
        if isinstance(a, PurePath):
            require(a.parts == b.parts, f'{path}: filesystem path differs')
            emit(digest, repr(a.parts).encode())
            return
        aid, bid = id(a), id(b)
        if aid in left_objects or bid in right_objects:
            require(left_objects.get(aid) == right_objects.get(bid), f'{path}: object alias differs')
            emit(digest, f'reference:{left_objects[aid]}'.encode())
            return
        ordinal = len(left_objects)
        left_objects[aid] = ordinal
        right_objects[bid] = ordinal
        if isinstance(a, torch.Tensor):
            require(a.layout == torch.strided, f'{path}: unsupported tensor layout')
            def properties(t):
                return (str(t.dtype), str(t.device), str(t.layout), tuple(t.shape),
                        tuple(t.stride()), t.storage_offset(), t.requires_grad,
                        t.is_conj(), t.is_neg(), t.untyped_storage().nbytes())
            require(properties(a) == properties(b), f'{path}: tensor properties differ')
            emit(digest, repr(properties(a)).encode())
            ak, bk = a.untyped_storage()._cdata, b.untyped_storage()._cdata
            if ak not in left_storages and bk not in right_storages:
                ordinal = len(left_storages)
                left_storages[ak] = ordinal
                right_storages[bk] = ordinal
            require(left_storages.get(ak) == right_storages.get(bk), f'{path}: storage alias differs')
            emit(digest, repr(left_storages[ak]).encode())
            def tensor_bytes(t):
                return t.detach().resolve_conj().resolve_neg().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            av, bv = tensor_bytes(a), tensor_bytes(b)
            require(av == bv, f'{path}: tensor values differ')
            counts['tensor_value_bytes'] += len(av)
            emit(digest, av)
        elif isinstance(a, (tuple, list)):
            require(len(a) == len(b), f'{path}: sequence length differs')
            emit(digest, repr(len(a)).encode())
            for index, (av, bv) in enumerate(zip(a, b)):
                visit(av, bv, f'{path}[{index}]')
        elif isinstance(a, dict):
            require(len(a) == len(b), f'{path}: mapping size differs')
            emit(digest, repr(len(a)).encode())
            for (ak, av), (bk, bv) in zip(a.items(), b.items()):
                visit(ak, bk, f'{path}.key')
                visit(av, bv, f'{path}[{ak!r}]')
        elif isinstance(a, (set, frozenset)):
            require(all(isinstance(v, (str, int, float, bool, bytes)) for v in a | b), f'{path}: unsupported set items')
            require(a == b, f'{path}: set differs')
            for value in sorted(a, key=lambda v: (type(v).__name__, repr(v))):
                visit(value, value, f'{path}.member')
        elif hasattr(a, '__dict__'):
            visit(vars(a), vars(b), f'{path}.__dict__')
        elif is_dataclass(a):
            for field in fields(a):
                visit(getattr(a, field.name), getattr(b, field.name), f'{path}.{field.name}')
        else:
            raise TypeError(f'{path}: unhandled type {typename}')

    visit(left, right, 'micro_steps')
    return {'canonical_decoded_sha256': digest.hexdigest(), 'counts': dict(counts),
            'object_alias_groups': len(left_objects), 'storage_alias_groups': len(left_storages)}


def storage_segments(payload):
    offset, segments = 0, []
    for _ in range(5):
        ops = list(pickletools.genops(payload[offset:]))
        segments.append([(op.name, arg) for op, arg, _ in ops])
        offset += ops[-1][2] + 1
    return segments, payload[offset:]


def compare_storage(left, right):
    a, av = storage_segments(left)
    b, bv = storage_segments(right)
    require(a[:3] == b[:3], 'torch storage header differs')
    require(av == bv, 'raw tensor storage bytes differ')
    # The legacy storage object's persistent tuple and key list carry the
    # same process-local storage key. Prove that these exact fields, and no
    # other field or byte, account for the differing blob.
    akeys = [arg for op, arg in a[4] if op == 'BINUNICODE']
    bkeys = [arg for op, arg in b[4] if op == 'BINUNICODE']
    require(len(akeys) == len(bkeys) == 1, 'unexpected torch storage key list')
    ak, bk = akeys[0], bkeys[0]
    require(ak.isdigit() and bk.isdigit(), 'unexpected nonnumeric storage key')
    changes = []
    for segment in (3, 4):
        require(len(a[segment]) == len(b[segment]), 'storage opcode count differs')
        for index, ((aop, aa), (bop, ba)) in enumerate(zip(a[segment], b[segment])):
            if (aop, aa) != (bop, ba):
                require(aop == bop == 'BINUNICODE' and aa == ak and ba == bk,
                        f'unexpected storage difference at {segment}:{index}')
                changes.append([segment, index])
    require(len(changes) == 2, f'expected exactly two storage key differences: {changes}')
    return {'baseline_key': ak, 'candidate_key': bk, 'changed_opcode_locations': changes,
            'identical_storage_bytes': len(av), 'identical_storage_sha256': sha(av)}


def compare_pickle(left, right):
    counts = Counter()
    storages = []
    for index, pair in enumerate(itertools.zip_longest(pickletools.genops(left), pickletools.genops(right))):
        a, b = pair
        require(a is not None and b is not None, f'outer opcode count differs at {index}')
        ao, aa, _ = a
        bo, ba, _ = b
        require(ao.name == bo.name, f'outer opcode differs at {index}')
        counts['outer_opcode_count'] += 1
        if aa == ba:
            continue
        if ao.name == 'FRAME':
            counts['frame_length_differences'] += 1
        elif ao.name in ('BINBYTES', 'SHORT_BINBYTES', 'BINBYTES8'):
            storages.append(compare_storage(aa, ba))
            counts['storage_key_blobs_differing'] += 1
        else:
            raise AssertionError(f'outer value differs at opcode {index} {ao.name}')
    return {'counts': dict(counts), 'changed_storage_keys': storages,
            'only_differences': ['process-local storage key repeated in torch legacy object and key-list pickles',
                                 'outer frame length if storage key byte length changes']}


def main():
    receipts = {}
    for arm, filename in [('baseline', 'baseline-prepare-frozen.json'), ('candidate', 'candidate-prepare.json')]:
        path = ROOT / filename
        receipt = json.loads(path.read_text())
        require(receipt['terminal_status'] == 'completed', f'{arm} preparation not complete')
        receipts[arm] = receipt
    result = {'status': 'verified', 'method': 'own-manifest authentication, own-chunk restricted loading, exact decoded traversal and complete pickle opcode/storage comparison',
              'changed_semantic_fields': [], 'splits': {}}
    for split in ('train', 'eval'):
        manifests, steps, caches = {}, {}, {}
        for arm in ('baseline', 'candidate'):
            entry = receipts[arm]['result'][split]
            cache = Path(entry['cache_dir'])
            caches[arm] = cache
            manifest_bytes = Path(entry['manifest_path']).read_bytes()
            require(sha(manifest_bytes) == entry['manifest_sha256'], f'{arm}/{split} receipt manifest hash mismatch')
            manifests[arm] = load_cache_manifest(cache, cache_root=cache.parent.parent,
                                                expected_fingerprint=entry['fingerprint'], level='manifest')
            # This existing reader verifies every own chunk digest, restricted
            # pickle payload and declared pack count before returning objects.
            steps[arm] = load_all_micro_steps_from_cache(cache, cache_root=cache.parent.parent,
                                                       expected_fingerprint=entry['fingerprint'])
            require(len(steps[arm]) == entry['micro_step_count'], f'{arm}/{split} receipt count mismatch')
        a, b = manifests['baseline'], manifests['candidate']
        require(a.keys() == b.keys(), 'manifest fields differ')
        for key in a:
            if key != 'chunks':
                require(a[key] == b[key], f'{split} manifest field differs: {key}')
        require(len(a['chunks']) == len(b['chunks']), 'chunk counts differ')
        chunks = []
        for ac, bc in zip(a['chunks'], b['chunks']):
            require({k:v for k,v in ac.items() if k != 'sha256'} == {k:v for k,v in bc.items() if k != 'sha256'},
                    'chunk metadata besides own SHA differs')
            ab = (caches['baseline'] / ac['path']).read_bytes()
            bb = (caches['candidate'] / bc['path']).read_bytes()
            require(sha(ab) == ac['sha256'] and sha(bb) == bc['sha256'], 'own raw chunk authentication failed')
            chunks.append({'path': ac['path'], 'baseline_bytes': len(ab), 'candidate_bytes': len(bb),
                           'baseline_sha256': sha(ab), 'candidate_sha256': sha(bb),
                           'pickle_difference_proof': compare_pickle(ab, bb)})
        decoded = compare_decoded(steps['baseline'], steps['candidate'])
        result['splits'][split] = {'micro_step_count': len(steps['baseline']), 'fingerprint': a['fingerprint'],
                                   'materialization': a['materialization'], 'decoded': decoded, 'chunks': chunks}
        print(json.dumps({'split': split, 'count': len(steps['baseline']), 'decoded': decoded,
                          'pickle_difference_counts': [c['pickle_difference_proof']['counts'] for c in chunks]}), flush=True)
    result['script_sha256'] = sha(Path(__file__).read_bytes())
    (ROOT / 'preparation-parity.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        (OUT / 'preparation-parity-failure.json').write_text(json.dumps({'status': 'failed', 'type': type(exc).__name__, 'error': str(exc)}, indent=2) + '\n')
        raise
