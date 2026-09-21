#!/usr/bin/env python
"""Task-local exact comparison of authenticated eight-rank step-4 states."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

REPO = Path('/data/CoordExp/.worktrees/coordexp-infras')
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from src.artifacts.training_state import (
    DecodedRankTrainingState, TrainingStateExpectations,
    admit_training_state, load_training_state_manifest,
)

MAX_PATHS = 64
EXPECTED_STEP = 4
EXPECTED_WORLD = 8
EXCLUSIONS = {
    'manifest.parent_run_id,parent_segment_id,continuation_index':
        'Different uninterrupted and resumed publication provenance.',
    'manifest.resolved_config and identities.resolved_config':
        'Run/resume selection differs; repository admission authenticates both '
        'and requires identical resume-compatibility identity and projection.',
    'manifest.aggregate_digest and serialized file digests':
        'Each is independently authenticated; decoded state is compared exactly '
        'rather than requiring identical archive serialization or provenance.',
}


class ExactComparison:
    def __init__(self):
        self.counts = Counter()
        self.mismatches = []

    def mismatch(self, path, reason):
        self.counts['mismatches'] += 1
        if len(self.mismatches) < MAX_PATHS:
            self.mismatches.append({'path': path[:256], 'reason': reason})

    def compare(self, left, right, path):
        self.counts['nodes'] += 1
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            self.counts['tensor_pairs'] += 1
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                return self.mismatch(path, 'tensor_type')
            if left.dtype != right.dtype or left.shape != right.shape or left.layout != right.layout:
                return self.mismatch(path, 'tensor_dtype_shape_or_layout')
            self.counts['tensor_elements'] += left.numel()
            if not torch.equal(left, right):
                self.mismatch(path, 'tensor_values')
        elif isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            self.counts['array_pairs'] += 1
            if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
                return self.mismatch(path, 'array_type')
            if left.dtype != right.dtype or left.shape != right.shape:
                return self.mismatch(path, 'array_dtype_or_shape')
            if not np.array_equal(left, right):
                self.mismatch(path, 'array_values')
        elif isinstance(left, Mapping) or isinstance(right, Mapping):
            if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                return self.mismatch(path, 'mapping_type')
            if set(left) != set(right):
                self.mismatch(path, 'mapping_keys')
            for key in left:
                if key in right:
                    self.compare(left[key], right[key], f'{path}.{key}')
        elif isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            if type(left) is not type(right):
                return self.mismatch(path, 'sequence_type')
            if len(left) != len(right):
                self.mismatch(path, 'sequence_length')
            for index, (a, b) in enumerate(zip(left, right)):
                self.compare(a, b, f'{path}[{index}]')
        else:
            self.counts['scalar_pairs'] += 1
            if type(left) is not type(right) or left != right:
                self.mismatch(path, 'scalar_type_or_value')


def compare_checkpoints(left_path, right_path):
    result = {
        'schema': 'coordexp-task-exact-resume-comparison-v1',
        'status': 'error', 'expected_step': EXPECTED_STEP,
        'expected_world_size': EXPECTED_WORLD,
        'left': str(left_path), 'right': str(right_path),
        'equality': 'Exact decoded values; tensor dtype, shape and layout included.',
        'exclusions': EXCLUSIONS,
        'started_at': datetime.now(timezone.utc).isoformat(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'training_state_owner_sha256': hashlib.sha256(
            (REPO / 'src/artifacts/training_state.py').read_bytes()).hexdigest(),
        'compared_rank_pairs': 0, 'admission_calls': 0,
    }
    comparator = ExactComparison()
    try:
        left_manifest = load_training_state_manifest(left_path)
        right_manifest = load_training_state_manifest(right_path)
        result['manifest_digests'] = [left_manifest.aggregate_digest, right_manifest.aggregate_digest]
        expectations = TrainingStateExpectations(
            checkpoint_step=EXPECTED_STEP, world_size=EXPECTED_WORLD,
            identities=left_manifest.identities,
            scheduler_applicable=left_manifest.scheduler_applicable,
            scaler_applicable=left_manifest.scaler_applicable,
        )
        for name in ('schema', 'schema_version', 'artifact_type', 'commit_status',
                     'save_boundary', 'optimizer_applicable'):
            comparator.compare(getattr(left_manifest, name), getattr(right_manifest, name), f'manifest.{name}')
        for rank in range(EXPECTED_WORLD):
            left = admit_training_state(left_path, expectations, current_rank=rank)
            result['admission_calls'] += 1
            right = admit_training_state(right_path, expectations, current_rank=rank)
            result['admission_calls'] += 1
            # Freeze each independently authenticated publication across all calls.
            if left.manifest.aggregate_digest != left_manifest.aggregate_digest or right.manifest.aggregate_digest != right_manifest.aggregate_digest:
                raise RuntimeError('manifest_changed_during_comparison')
            if rank == 0:
                comparator.compare(left.resume_compatibility, right.resume_compatibility, 'resume_compatibility')
            for field in fields(DecodedRankTrainingState):
                comparator.compare(getattr(left.decoded_rank, field.name),
                                   getattr(right.decoded_rank, field.name),
                                   f'ranks[{rank}].{field.name}')
            result['compared_rank_pairs'] += 1
            del left, right
        result['status'] = 'mismatch' if comparator.counts['mismatches'] else 'equal'
    except Exception as exc:
        result['error'] = {'type': type(exc).__name__, 'code': getattr(exc, 'code', None)}
    result.update(counts=dict(comparator.counts), mismatches=comparator.mismatches,
                  omitted_mismatch_paths=max(0, comparator.counts['mismatches'] - MAX_PATHS),
                  completed_at=datetime.now(timezone.utc).isoformat())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('uninterrupted_checkpoint', type=Path)
    parser.add_argument('resumed_checkpoint', type=Path)
    parser.add_argument('--receipt', type=Path, required=True, help='Must not exist.')
    args = parser.parse_args()
    # Exclusive reservation happens before checkpoint IO; never overwrite evidence.
    with args.receipt.open('x', encoding='utf-8') as output:
        result = compare_checkpoints(args.uninterrupted_checkpoint.resolve(),
                                     args.resumed_checkpoint.resolve())
        output.write(json.dumps(result, indent=2) + '\n')
        output.flush()
    code = {'equal': 0, 'mismatch': 1, 'error': 2}[result['status']]
    print(json.dumps({'status': result['status'], 'exit_code': code,
                      'compared_rank_pairs': result['compared_rank_pairs'],
                      'mismatch_count': result['counts'].get('mismatches', 0),
                      'receipt': str(args.receipt.resolve())}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
