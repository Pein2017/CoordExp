"""Independent CPU readback for matched-state, coherent Q/self-K phase transfer."""
import json
from pathlib import Path

import torch

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion.recurrence_transition_readback import verified
from src.artifacts.source_provenance import preserve_source
from src.qwen.input_identity import tensor_hash


ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-downstream-query-phase')
OUT = ROOT / 'attempt-002'
ORDER = ['native', 'held_native_phase', 'held_late_phase']
ATOL = 2e-4


def complex_rotate(value, cos, sin):
    half = value.shape[-1] // 2
    z = torch.complex(value[..., :half].double(), value[..., half:].double())
    phase = torch.complex(cos[..., :half].double(), sin[..., :half].double())
    result = z * phase[:, None, :]
    return torch.cat((result.real, result.imag), dim=-1)


def main():
    read = lambda p: json.loads(p.read_text())
    checks = ROOT / 'lead-checks'
    checks.mkdir(exist_ok=True)
    selection = read(ROOT / 'selection.json')
    receipt, manifest, records, result = [read(OUT / p) for p in ('receipt.json', 'source-to-cell.json', 'readback.json', 'result.json')]
    failed = read(ROOT / 'attempt-001/receipt.json')
    failed_manifest = read(ROOT / 'attempt-001/source-to-cell.json')
    old_source = failed_manifest['producer_capture']
    verified(old_source)
    assert old_source['sha256'] == failed_manifest['producer']['sha256']
    assert failed['status'] == 'technical_invalid' and failed['model_forwards'] == failed['vision_forwards'] == 1
    assert not failed['completed_cells'] and 'Qwen3VLVisionAttention' in failed['error']
    assert not Path(f"/proc/{failed['pid']}").exists()
    bindings = {}

    def walk(obj):
        if isinstance(obj, dict):
            if {'path', 'sha256', 'size_bytes'} <= obj.keys():
                key = (obj['path'], obj['sha256'], obj['size_bytes'])
                if key not in bindings:
                    path = verified(obj)
                    bindings[key] = literal_binding(path)
            else:
                for value in obj.values():
                    walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    for obj in (selection, receipt, manifest, records, result, failed, old_source):
        walk(obj)
    for name in ('groups_acceptance', 'reverse_acceptance', 'trajectory_acceptance'):
        acceptance = read(verified(selection['sources'][name]))
        assert acceptance['status'] == 'lead-accepted'
    assert manifest['status'] == 'frozen_before_forward' and receipt['status'] == 'candidate_complete'
    assert receipt['cell_order'] == receipt['completed_cells'] == ORDER
    for obj in (receipt, records, result):
        assert obj['model_forwards'] == obj['vision_forwards'] == 3
    assert receipt['elapsed_seconds'] <= 600 and not Path(f"/proc/{receipt['pid']}").exists()
    prior_root = ROOT.parent / '2026-09-22-recurrence-first-layer-groups'
    prior = read(prior_root / 'attempt-001/readback.json')
    assert manifest['model_identity'] == read(prior_root / 'attempt-001/source-to-cell.json')['model_identity']
    assert manifest['case']['native_input_hashes'] == selection['native_input_hashes'] == prior['native_input_hashes']
    captures = {k: torch.load(verified(selection['sources'][k]), map_location='cpu', weights_only=True)
                for k in ('native_capture', 'native_trajectory', 'reverse_late_logits')}
    group = captures['native_capture']
    trajectory = captures['native_trajectory']
    assert trajectory['query_positions'][[15, 62]].tolist() == [1703, 2126]
    tensors = {k: torch.load(OUT / f'{k}.pt', map_location='cpu', weights_only=True) for k in ORDER}
    native = tensors['native']['full_logits']
    native_error = float((native[2] - group['full_logits']).abs().max())
    control_error = float((tensors['held_native_phase']['full_logits'][2, 0] - captures['reverse_late_logits']['logits'][2]).abs().max())
    assert native_error <= ATOL and control_error <= ATOL
    incoming = trajectory['residual_input'][1][[15, 62]]
    visibility = tensor_hash(torch.arange(2127)[None, :] <= torch.tensor([1703, 2126])[:, None])
    summaries, gates = {}, {}
    for label in ORDER:
        x, record = tensors[label], records['cells'][label]
        logits = x['full_logits']
        assert logits.shape == (4, 2, 152670) and logits.dtype == torch.float32 and torch.isfinite(logits).all()
        assert x['query_positions'].tolist() == [1703, 2126]
        assert record['query_consumer'] == {'shape': [4, 2, 2048], 'physical_indices': [1703, 2126], 'exact': True}
        assert record['native_input_hashes'] == selection['native_input_hashes']
        assert record['embedding_calls'] == record['rotary_calls'] == 1 and record['cache_length'] == 2127
        assert record['attention'] == prior['attention'] and len(record['attention']) == 28
        assert all(a['query_mask_hash'] == visibility and a['query_visibility_exact'] for a in record['attention'])
        expected_head = group['head_output'][0 if label == 'native' else 1].flatten()
        assert torch.equal(x['layer0_before'], group['head_output'][0].flatten())
        assert torch.equal(x['layer0_consumed'], expected_head) and torch.equal(x['layer0_expected'], expected_head)
        assert record['layer0_consumer'] == {'second_hook_seen': True, 'off_target_exact': True, 'off_target_max_abs': 0.0}
        state_error = float((x['layer1_input'][0] - incoming[0 if label == 'native' else 1]).abs().max())
        assert state_error <= ATOL
        if label == 'native':
            assert float((x['layer1_input'][1] - incoming[1]).abs().max()) <= ATOL
        companions = float((logits[[0, 1, 3]] - native[[0, 1, 3]]).abs().max())
        assert companions <= ATOL
        phase_index = int(label == 'held_late_phase')
        cos, sin = group['query_cos'][phase_index].expand(27, -1), group['query_sin'][phase_index].expand(27, -1)
        assert torch.equal(x['used_phase_cos'], cos) and torch.equal(x['used_phase_sin'], sin)
        assert x['pre_Q'].shape == (27, 16, 128) and x['pre_K'].shape == (27, 8, 128)
        before_cos, before_sin = group['query_cos'][0].expand(27, -1), group['query_sin'][0].expand(27, -1)
        errors = {}
        for key, expected in (
            ('native_Q_target', complex_rotate(x['pre_Q'], before_cos, before_sin)),
            ('native_K_target_after_gqa', complex_rotate(x['pre_K'], before_cos, before_sin).repeat_interleave(2, dim=1)),
            ('consumed_Q_target', complex_rotate(x['pre_Q'], cos, sin)),
            ('consumed_K_target_after_gqa', complex_rotate(x['pre_K'], cos, sin).repeat_interleave(2, dim=1))):
            assert x[key].shape == (27, 16, 128)
            errors[key] = float((x[key].double() - expected).abs().max())
            assert errors[key] <= ATOL
        q0, q1 = x['native_Q_target'].double(), x['consumed_Q_target'].double()
        k0, k1 = x['native_K_target_after_gqa'].double(), x['consumed_K_target_after_gqa'].double()
        errors['q_norm'] = float((q0.norm(dim=-1) - q1.norm(dim=-1)).abs().max())
        errors['k_norm'] = float((k0.norm(dim=-1) - k1.norm(dim=-1)).abs().max())
        errors['self_score'] = float((((q0*k0).sum(-1) - (q1*k1).sum(-1)) / (128**.5)).abs().max())
        assert max(errors.values()) <= ATOL
        for layer in range(1, 28):
            metric = record['layer_metrics'][str(layer)]
            baseline = records['cells']['native']['layer_metrics'][str(layer)]
            assert metric['q_k_off_target_exact'] and metric['gqa_expansion_exact'] and metric['o_proj_consumer_exact']
            assert metric['sdpa_calls'] == 1 and metric['cache_gqa_heads'] == 16 and metric['gqa_repetitions'] == 2
            for key in ('historical_K_hash', 'historical_V_hash', 'mask_hash', 'cache_slots_hash'):
                assert metric[key] == baseline[key]
        z = logits[2, 0].double()
        scores, tokens = z.topk(5)
        probability = z.softmax(-1)
        summaries[label] = {'winner': int(tokens[0]), 'winner_bin': int(tokens[0])-151670, 'top5_tokens': tokens.tolist(), 'top5_logits': scores.tolist(), 'gap': float(scores[0]-scores[1]), 'd38_999': float(z[151708]-z[152669]), 'p38': float(probability[151708]), 'p999': float(probability[152669])}
        gates[label] = {'layer1_input_error': state_error, 'companion_error': companions, 'phase_errors': errors}
    assert native[2].argmax(-1).tolist() == [151708, 152669]
    assert summaries['held_native_phase']['winner'] == 151708
    primary = summaries['held_late_phase']
    changed = float((tensors['held_late_phase']['consumed_Q_target'] - tensors['held_late_phase']['native_Q_target']).abs().max())
    assert changed > 1e-6
    payload = sum((OUT / f'{label}.pt').stat().st_size for label in ORDER)
    assert payload <= 32 * 1024**2
    capture = preserve_source(Path(__file__), run_root=ROOT, relative_name='downstream_query_phase_readback.py')
    report = {'status': 'lead-qualified-coherent-query-phase', 'primary': {'predicted': 151708, 'observed': primary['winner'], 'passed': primary['winner'] == 151708, 'late_native_decision_reproduced': primary['winner'] == 152669}, 'cells': summaries, 'gates': gates, 'native_error': native_error, 'held_control_error': control_error, 'phase_effect_d38_999': primary['d38_999']-summaries['held_native_phase']['d38_999'], 'actual_Q_treatment_max_abs': changed, 'successful_model_forwards': 3, 'successful_vision_forwards': 3, 'total_model_invocations': 4, 'total_vision_invocations': 4, 'tensor_bytes': payload, 'failed_source_capture': old_source, 'verified_bindings': list(bindings.values()), 'source': literal_binding(capture), 'scope': 'Matched incoming state and query/self-key phases at one earlier visible history; not timing or physical recovery.'}
    (checks / 'phase-readback.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('verified_bindings', 'source', 'failed_source_capture', 'gates')}))
    print(json.dumps({'verified_binding_count': len(bindings), 'gates': gates}))


if __name__ == '__main__':
    main()
