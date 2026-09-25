"""Independent CPU readback of early-recipient layer0 reverse transfer."""
import json
from pathlib import Path

import torch

from src.artifacts.utf8_json import literal_binding
from probes.recurrence_dynamics.recurrence_transition_readback import verified
from src.artifacts.source_provenance import preserve_source
from src.qwen.input_identity import tensor_hash


ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-first-layer-reverse-transfer')
PRIOR = ROOT.parent / '2026-09-22-recurrence-first-layer-groups'
ORDER = ['native', 'old_sham', 'late']


def main():
    read = lambda path: json.loads(path.read_text())
    out = ROOT / 'attempt-001'
    checks = ROOT / 'lead-checks'
    checks.mkdir(exist_ok=True)
    selection, receipt, manifest, records, result = [read(p) for p in (
        ROOT / 'selection.json', out / 'receipt.json', out / 'source-to-cell.json', out / 'readback.json', out / 'result.json')]
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

    for obj in (selection, receipt, manifest, records, result):
        walk(obj)
    for name in ('groups_acceptance', 'readout_acceptance'):
        acceptance = read(verified(selection['sources'][name]))
        assert acceptance['status'] == 'lead-accepted'
        walk(acceptance)
    assert receipt['status'] == 'candidate_complete' and manifest['status'] == 'frozen_before_forward'
    assert receipt['cell_order'] == receipt['completed_cells'] == ORDER
    assert [x['cell'] for x in receipt['calls']] == ORDER
    for obj in (receipt, records, result):
        assert obj['model_forwards'] == obj['vision_forwards'] == 3
    assert receipt['elapsed_seconds'] <= 600 and not Path(f"/proc/{receipt['pid']}").exists()
    old_records = read(PRIOR / 'attempt-001/readback.json')
    assert manifest['model_identity'] == read(PRIOR / 'attempt-001/source-to-cell.json')['model_identity']
    assert manifest['case']['native_input_hashes'] == records['native_checks']['input_hashes'] == selection['native_input_hashes'] == old_records['native_input_hashes']
    aliases = {'query_input_token': 'expected_query_input_token', 'next_token': 'chosen_token', 'trace_top2_tokens': 'top2_tokens', 'trace_top2_logits': 'top2_logits'}
    for name, key in [('recipient', 'source_query'), ('donor', 'donor_query')]:
        query = manifest['case'][name]['query_metadata']
        assert {aliases.get(k, k): v for k, v in query.items()} == selection[key]
    assert (selection['source_query']['physical_query_index'], selection['source_query']['raw_action_offset']) == (1703, 384)
    assert (selection['donor_query']['physical_query_index'], selection['donor_query']['raw_action_offset']) == (2126, 807)
    accepted = torch.load(verified(selection['sources']['native_capture']), map_location='cpu', weights_only=True)
    tensors = {k: torch.load(out / f'{k}.pt', map_location='cpu', weights_only=True) for k in ORDER}
    native = tensors['native']['logits']
    native_error = float((native[2] - accepted['full_logits'][0]).abs().max())
    sham_error = float((tensors['old_sham']['logits'] - native).abs().max())
    assert native_error <= 2e-4 and sham_error <= 2e-4
    early = accepted['head_output'][0].float().reshape(-1)
    visibility = tensor_hash(torch.arange(2127)[None, :] <= 1703)
    summaries = {}
    for label in ORDER:
        x, record = tensors[label], records['cells'][label]
        logits = x['logits']
        assert logits.shape == (4, 152670) and logits.dtype == torch.float32 and torch.isfinite(logits).all()
        assert x['query_position'].tolist() == [1703] and x['target_batch'].tolist() == [2]
        expected = accepted['head_output'][int(label == 'late')].float().reshape(-1)
        assert torch.equal(x['before_o_proj_target'], early)
        for field in ('consumed_o_proj_target', 'replacement_o_proj_target'):
            assert torch.equal(x[field], expected) and x[field].shape == (2048,)
        expected_hash = selection['native_head_flat_hash'] if label == 'native' else selection['replacement_fp32_flat_hash'][label]
        assert tensor_hash(expected) == record['consumed_target_hash'] == record['replacement_target_hash'] == expected_hash
        assert tensor_hash(early) == record['before_target_hash']
        assert record['consumer'] == {'second_hook_seen': True, 'tensor_identity_exact': True, 'off_target_exact': True, 'off_target_max_abs': 0.0}
        assert record['query_consumer'] == {'shape': [4, 1, 2048], 'physical_indices': [1703], 'exact': True}
        assert record['embedding_calls'] == record['rotary_calls'] == 1 and record['cache_length'] == 2127
        assert len(record['attention']) == 28 and record['attention'] == records['native_checks']['attention']
        for current, old in zip(record['attention'], old_records['attention'], strict=True):
            for field in ('mask_shape', 'mask_hash', 'cache_slots_hash', 'phase_shapes'):
                assert current[field] == old[field]
            assert current['query_mask_hash'] == visibility and current['query_visibility_exact'] and current['cache_is_native_empty_at_entry']
        companion_error = float((logits[[0, 1, 3]] - native[[0, 1, 3]]).abs().max())
        assert companion_error <= 2e-4
        z = logits[2].double()
        values, tokens = z.topk(5)
        probability = z.softmax(-1)
        summaries[label] = {'winner': int(tokens[0]), 'winner_bin': int(tokens[0]) - 151670, 'top5_tokens': tokens.tolist(), 'top5_logits': values.tolist(), 'gap': float(values[0] - values[1]), 'd38_999': float(z[151708] - z[152669]), 'p38': float(probability[151708]), 'p999': float(probability[152669]), 'target_max_abs': float((logits[2] - native[2]).abs().max()), 'companion_max_abs': companion_error}
    assert summaries['native']['winner'] == summaries['old_sham']['winner'] == 151708
    payload_bytes = sum((out / f'{k}.pt').stat().st_size for k in ORDER)
    assert payload_bytes <= 32 * 1024**2
    source_hash = literal_binding(Path(__file__))['sha256']
    capture = preserve_source(Path(__file__), run_root=ROOT, relative_name=f'reverse_transfer_readback-{source_hash[:12]}.py')
    report = {'status': 'lead-qualified-local-reverse-transfer', 'primary': {'predicted': 151708, 'observed': summaries['late']['winner'], 'passed': summaries['late']['winner'] == 151708}, 'native_error': native_error, 'sham_error': sham_error, 'cells': summaries, 'late_minus_native_d': summaries['late']['d38_999'] - summaries['native']['d38_999'], 'model_forwards': 3, 'vision_forwards': 3, 'tensor_payload_bytes': payload_bytes, 'verified_bindings': list(bindings.values()), 'source': literal_binding(capture), 'scope': 'Exit-associated layer0 head state sufficiency at one earlier native landmark; no natural onset or exit-time prediction.'}
    (checks / 'reverse-transfer-readback.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('verified_bindings', 'source')}))
    print(json.dumps({'verified_binding_count': len(bindings)}))


if __name__ == '__main__':
    main()
