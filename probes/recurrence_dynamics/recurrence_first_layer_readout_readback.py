"""Independent CPU acceptance of the five frozen layer0 readout cells."""
import json
from pathlib import Path

import torch

from src.artifacts.utf8_json import literal_binding
from probes.recurrence_dynamics.recurrence_transition_readback import verified
from src.artifacts.source_provenance import preserve_source
from src.qwen.input_identity import tensor_hash


ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-first-layer-readout')
OUT = ROOT / 'attempt-001'
PRIOR = ROOT.parent / '2026-09-22-recurrence-first-layer-groups'
ORDER = ['native', '11', '01', '10', '00']
ATOL = 2e-4


def main():
    checks = ROOT / 'lead-checks'
    checks.mkdir(exist_ok=True)
    read = lambda path: json.loads(path.read_text())
    selection = read(ROOT / 'selection.json')
    receipt = read(OUT / 'receipt.json')
    manifest = read(OUT / 'source-to-cell.json')
    records = read(OUT / 'readback.json')
    result = read(OUT / 'result.json')
    predecessor = read(PRIOR / 'lead-acceptance.json')
    bindings = {}

    def walk(obj):
        if isinstance(obj, dict):
            if {'path', 'sha256', 'size_bytes'} <= obj.keys():
                path = verified(obj)
                bindings[str(path)] = literal_binding(path)
            else:
                for value in obj.values():
                    walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    for obj in (selection, receipt, manifest, records, result, predecessor):
        walk(obj)
    assert predecessor['status'] == 'lead-accepted'
    assert manifest['status'] == 'frozen_before_forward'
    assert receipt['status'] == 'candidate_complete'
    assert receipt['cell_order'] == receipt['completed_cells'] == ORDER
    assert [call['cell'] for call in receipt['calls']] == ORDER
    for obj in (receipt, records, result):
        assert obj['model_forwards'] == obj['vision_forwards'] == 5
    assert receipt['elapsed_seconds'] <= 600
    prior_record = read(PRIOR / 'attempt-001/readback.json')
    prior_manifest = read(PRIOR / 'attempt-001/source-to-cell.json')
    assert manifest['model_identity'] == prior_manifest['model_identity']
    assert manifest['case']['native_input_hashes'] == records['native_checks']['input_hashes'] == selection['native_input_hashes'] == prior_record['native_input_hashes']
    assert (selection['target'], selection['query_position'], selection['raw_action_offset']) == (2, 2126, 807)
    assert len(manifest['case']['query_metadata']) == 1
    query = manifest['case']['query_metadata'][0]
    aliases = {'query_input_token': 'expected_query_input_token', 'next_token': 'chosen_token',
               'trace_top2_tokens': 'top2_tokens', 'trace_top2_logits': 'top2_logits'}
    assert {aliases.get(key, key): value for key, value in query.items()} == selection['source_query']
    factorial = torch.load(verified(selection['sources']['factorial']), map_location='cpu', weights_only=True)['cells']
    accepted = torch.load(verified(selection['sources']['native_capture']), map_location='cpu', weights_only=True)
    tensors = {key: torch.load(OUT / f'{key}.pt', map_location='cpu', weights_only=True) for key in ORDER}
    native = tensors['native']['logits']
    expected_head = accepted['head_output'][1].float().reshape(-1)
    assert tensor_hash(expected_head) == selection['native_head_flat_hash']
    native_error = float((native[2] - accepted['full_logits'][1]).abs().max())
    sham_error = float((tensors['11']['logits'] - native).abs().max())
    assert native_error <= ATOL and sham_error <= ATOL
    expected_visibility = tensor_hash(torch.arange(2127)[None, :] <= 2126)
    summaries = {}
    for label in ORDER:
        x, record = tensors[label], records['cells'][label]
        logits = x['logits']
        assert logits.shape == (4, 152670) and logits.dtype == torch.float32 and torch.isfinite(logits).all()
        assert x['query_position'].tolist() == [2126] and x['target_batch'].tolist() == [2]
        expected = expected_head if label == 'native' else factorial[label].float().reshape(-1)
        expected_hash = selection['native_head_flat_hash'] if label == 'native' else selection['replacement_fp32_flat_hash'][label]
        assert tensor_hash(expected) == expected_hash
        for field in ('consumed_o_proj_target', 'replacement_o_proj_target'):
            assert torch.equal(x[field], expected) and x[field].shape == (2048,)
        assert torch.equal(x['before_o_proj_target'], expected_head)
        for field, saved in [('before_o_proj_target', 'before_target_hash'), ('consumed_o_proj_target', 'consumed_target_hash'), ('replacement_o_proj_target', 'replacement_target_hash')]:
            assert tensor_hash(x[field]) == record[saved]
        assert record['consumer'] == {'second_hook_seen': True, 'tensor_identity_exact': True, 'off_target_exact': True, 'off_target_max_abs': 0.0}
        assert record['query_consumer'] == {'shape': [4, 1, 2048], 'physical_indices': [2126], 'exact': True}
        assert record['embedding_calls'] == record['rotary_calls'] == 1 and record['cache_length'] == 2127
        assert len(record['attention']) == 28 and record['attention'] == records['native_checks']['attention']
        for current, prior in zip(record['attention'], prior_record['attention'], strict=True):
            for field in ('mask_shape', 'mask_hash', 'cache_slots_hash', 'phase_shapes'):
                assert current[field] == prior[field]
            assert current['query_mask_hash'] == expected_visibility and current['query_visibility_exact'] and current['cache_is_native_empty_at_entry']
        companions = float((logits[[0, 1, 3]] - native[[0, 1, 3]]).abs().max())
        assert companions <= ATOL
        vector = logits[2].double()
        values, tokens = vector.topk(5)
        probability = vector.softmax(-1)
        summaries[label] = {
            'winner_token': int(tokens[0]), 'winner_bin': int(tokens[0]) - 151670,
            'top5_tokens': tokens.tolist(), 'top5_logits': values.tolist(),
            'top1_top2_gap': float(values[0] - values[1]),
            'd_z38_minus_z999': float(vector[151708] - vector[152669]),
            'p38': float(probability[151708]), 'p999': float(probability[152669]),
            'target_full_vector_max_abs_vs_native': float((logits[2] - native[2]).abs().max()),
            'companion_max_abs': companions,
        }
    assert summaries['native']['winner_token'] == summaries['11']['winner_token'] == 152669
    d = {label: summaries[label]['d_z38_minus_z999'] for label in ORDER}
    interaction = d['11'] - d['10'] - d['01'] + d['00']
    assert interaction == records['d_interaction_11_minus_10_minus_01_plus_00']
    payload_bytes = sum((OUT / f'{label}.pt').stat().st_size for label in ORDER)
    assert payload_bytes <= 64 * 1024**2
    source_hash = literal_binding(Path(__file__))['sha256']
    capture = preserve_source(Path(__file__), run_root=ROOT, relative_name=f'readout_readback-{source_hash[:12]}.py')
    report = {
        'status': 'lead-qualified-local-intervention',
        'primary_prediction': {'predicted_token': 151708, 'observed_token': summaries['00']['winner_token'], 'passed': summaries['00']['winner_token'] == 151708},
        'native_full_vector_max_abs': native_error, 'sham_full_vectors_max_abs': sham_error,
        'cells': summaries,
        'effects': {'undo_anchor_at_new_pool': d['01'] - d['11'], 'undo_pool_at_new_anchor': d['10'] - d['11'], 'undo_both': d['00'] - d['11'], 'interaction': interaction,
                    'symmetric_anchor': .5 * (d['10'] - d['00'] + d['11'] - d['01']), 'symmetric_pool': .5 * (d['01'] - d['00'] + d['11'] - d['10'])},
        'model_forwards': 5, 'vision_forwards': 5, 'tensor_payload_bytes': payload_bytes,
        'verified_bindings': list(bindings.values()), 'source': literal_binding(capture),
        'scope': 'One original native exit query; all-head layer0 intervention with downstream recomputation. No temporal or physical recovery prediction.'}
    (checks / 'readout-readback.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('verified_bindings', 'source', 'cells')}))
    print(json.dumps({'winners': {k: v['winner_bin'] for k, v in summaries.items()}, 'verified_binding_count': len(bindings)}))


if __name__ == '__main__':
    main()
