"""CPU acceptance reducer for the bounded source-route gate.

It verifies the frozen receipt and records the concrete counterexample.  It
never loads a model and never changes the scientific panel or pilot outputs.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final')
CUR = ROOT / 'technical-gate-v2'
PRIOR = ROOT / 'technical-gate-v1'
TOL = 2e-4

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def bind(path: Path) -> dict:
    return {'path': str(path.resolve()), 'sha256': sha(path), 'size_bytes': path.stat().st_size}

def load(path: Path):
    return json.loads(path.read_text())

def main() -> None:
    receipt = load(CUR / 'gate-receipt.json')
    comparison = load(CUR / 'comparison-tied.json')
    prior_receipt = load(PRIOR / 'gate-receipt.json')
    prior_comparison = load(PRIOR / 'comparison-tied.json')
    c_reuse = load(CUR / 'c-qualification-reuse.json')
    assert receipt['status'] == 'failed'
    assert prior_receipt['status'] == 'failed'
    assert receipt['model_forwards'] == 4
    assert receipt['prior_attempt']['aggregate_model_forwards'] == 8
    assert comparison['model'] == 'tied'
    assert comparison['status'] == 'failed'
    assert comparison['conditioning']['dtype'] == 'fp32'
    assert comparison['conditioning']['attention'] == 'sdpa'
    assert comparison['conditioning']['generation'] == 'none'
    assert comparison['conditioning']['positions'] == {'row_boundary_target_delta': 0, 'forced_x1_target_delta': 0}
    boundary = comparison['stages']['row_boundary']
    forced = comparison['stages']['forced_description_x1']
    assert boundary['full_vocab_max_abs_delta'] > TOL
    assert boundary['winner_exact_match'] is False
    assert boundary['passed'] is False
    assert forced['full_vocab_max_abs_delta'] <= TOL
    assert forced['winner_exact_match'] is True
    assert forced['passed'] is True
    assert boundary['source']['saved_trace']['passed'] is True
    assert forced['source']['saved_trace']['passed'] is True
    assert comparison['forwards']['native_model_forwards'] == 4
    assert prior_comparison['forwards']['native_model_forwards'] == 4
    assert c_reuse['status'] == 'passed'
    assert comparison['source']['source_group_case_count'] == 4
    assert comparison['source']['target_batch_index'] == 3
    assert comparison['source']['source_row_end'] == 119
    assert comparison['source']['target_native_token_hash'] == comparison['source']['boundary']['native_token_hash']
    assert comparison['source']['target_prefix_hash'] == comparison['source']['boundary']['prefix_hash']
    assert comparison['source']['native_group']['media_sha256'][3] == comparison['target_only']['native']['media_sha256']
    assert comparison['source']['native_group']['image_grids'][3] == comparison['target_only']['native']['image_grid']
    assert comparison['source']['native_group']['prompt_token_counts'][3] == comparison['target_only']['native']['prompt_token_count']
    assert not any('generate' in str(item).lower() for item in receipt.get('errors', []))
    gpu_ms = prior_comparison['forwards']['source_heterogeneous_gpu_ms'] + prior_comparison['forwards']['target_only_gpu_ms'] + comparison['forwards']['source_heterogeneous_gpu_ms'] + comparison['forwards']['target_only_gpu_ms']
    wall = prior_comparison['forwards']['elapsed_seconds_including_load_and_prepare'] + comparison['forwards']['elapsed_seconds_including_load_and_prepare']
    result = {
        'schema': 'recurrence_spatial_source.technical_gate_cpu_acceptance.v1',
        'status': 'verified_counterexample',
        'scientific_interpretation': 'The original-image target-only route and original heterogeneous batch-4 route agree at the forced-description x1 boundary but fail the required free row-boundary parity for tied 417044. Broad Lane-B launch is blocked pending parent/root decision.',
        'counterexample': {
            'model': 'tied',
            'state_id': 'tied-417044-failure',
            'source_row_end': 119,
            'comparison': 'original-image target-only batch1 versus original heterogeneous source batch4',
            'boundary_full_vocab_max_abs_delta': boundary['full_vocab_max_abs_delta'],
            'boundary_source_winner': boundary['source']['winner_token_id'],
            'boundary_target_only_winner': boundary['target_only']['winner_token_id'],
            'boundary_winner_exact_match': boundary['winner_exact_match'],
            'forced_x1_full_vocab_max_abs_delta': forced['full_vocab_max_abs_delta'],
            'forced_x1_winner_exact_match': forced['winner_exact_match'],
            'saved_source_trace_checks_passed': {'row_boundary': boundary['source']['saved_trace']['passed'], 'forced_x1': forced['source']['saved_trace']['passed']},
            'target_position_deltas': comparison['conditioning']['positions'],
        },
        'conditioning_integrity': {
            'c_reuse_status': c_reuse['status'],
            'source_group_case_count': comparison['source']['source_group_case_count'],
            'target_batch_index': comparison['source']['target_batch_index'],
            'source_row_end': comparison['source']['source_row_end'],
            'target_prefix_hash': comparison['source']['target_prefix_hash'],
            'target_native_token_hash': comparison['source']['target_native_token_hash'],
            'target_prompt_count_source_batch': comparison['source']['native_group']['prompt_token_counts'][3],
            'target_prompt_count_target_only': comparison['target_only']['native']['prompt_token_count'],
            'target_grid_source_batch': comparison['source']['native_group']['image_grids'][3],
            'target_grid_target_only': comparison['target_only']['native']['image_grid'],
            'target_media_source_batch': comparison['source']['native_group']['media_sha256'][3],
            'target_media_target_only': comparison['target_only']['native']['media_sha256'],
            'dtype': comparison['conditioning']['dtype'],
            'attention': comparison['conditioning']['attention'],
            'generation': comparison['conditioning']['generation'],
            'tolerance': TOL,
        },
        'cost': {
            'model_forwards_attempt_v1': prior_comparison['forwards']['native_model_forwards'],
            'model_forwards_attempt_v2': comparison['forwards']['native_model_forwards'],
            'model_forwards_total': 8,
            'native_generation_calls': 0,
            'forward_gpu_ms_total': gpu_ms,
            'forward_gpu_seconds_total': gpu_ms / 1000.0,
            'forward_gpu_hours_total': gpu_ms / 1000.0 / 3600.0,
            'enclosing_wall_seconds_attempts': wall,
        },
        'denominator': {
            'panel_states': 45,
            'pilot_states_preserved': 2,
            'new_spatial_cells': 0,
            'broad_states_run': 0,
            'replacement_count': 0,
            'rerun_00_count': 0,
        },
        'artifacts': {
            'current_receipt': bind(CUR / 'gate-receipt.json'),
            'current_comparison': bind(CUR / 'comparison-tied.json'),
            'prior_receipt': bind(PRIOR / 'gate-receipt.json'),
            'prior_comparison': bind(PRIOR / 'comparison-tied.json'),
            'c_reuse': bind(CUR / 'c-qualification-reuse.json'),
            'precall_snapshot': bind(CUR / 'precall-snapshot.json'),
            'pilot_receipt': bind(ROOT / 'corrected-pilot-v2' / 'pilot-receipt.json'),
        },
        'acceptance_command': 'PYTHONPATH=/data/CoordExp/.worktrees/research-probes python3 /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/technical-gate-v2/reduce_gate.py',
        'no_broad_release': True,
    }
    out = CUR / 'cpu-acceptance.json'
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
