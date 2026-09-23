"""Independent raw readback; preserve the producer's pre-enrichment hash defect."""
import json
from pathlib import Path

import torch

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion.recurrence_transition_readback import verified
from src.artifacts.source_provenance import preserve_source, source_snapshot_path
from src.qwen.input_identity import tensor_hash

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-local-prefix-phase')
OUT = ROOT / 'attempt-002'
ORDER = ['query_only', 'identity_recompute', 'prefix_coherent']
ATOL = 2e-4


def rotate(value, cos, sin):
    half = value.shape[-1] // 2
    z = torch.complex(value[..., :half].double(), value[..., half:].double())
    phase = torch.complex(cos[..., :half].double(), sin[..., :half].double())
    rotated = z * phase
    return torch.cat((rotated.real, rotated.imag), dim=-1)


def main():
    read = lambda p: json.loads(p.read_text())
    selection = read(ROOT / 'selection.json')
    receipt, manifest, records, result = [read(OUT / name) for name in
        ('receipt.json', 'source-to-cell.json', 'readback.json', 'result.json')]
    failed = read(ROOT / 'attempt-001/receipt.json')
    assert failed['status'] == 'technical_invalid' and failed['model_forwards'] == failed['vision_forwards'] == 0
    assert not failed['completed_cells'] and not Path(f"/proc/{failed['pid']}").exists()
    failed_source = source_snapshot_path(ROOT / 'attempt-001', 'recurrence_local_prefix_phase.py')
    assert literal_binding(failed_source)['sha256'] == '9e7f344e37859776e4724dc3517cb43033edb31e8455945cfe53a0f042d26896'
    assert receipt['status'] == 'candidate_complete' and records['status'] == result['status'] == 'candidate'
    assert receipt['cell_order'] == receipt['completed_cells'] == ORDER
    assert receipt['model_forwards'] == receipt['vision_forwards'] == 3 and receipt['elapsed_seconds'] <= 600
    assert not Path(f"/proc/{receipt['pid']}").exists()
    assert manifest['status'] == 'frozen_before_forward'
    tensors = {k: torch.load(OUT / k / 'held_late_phase.pt', map_location='cpu', weights_only=True) for k in ORDER}
    enriched_paths = {str((OUT / k / 'held_late_phase.pt').resolve()) for k in ORDER}
    bindings, stale = {}, {}

    def walk(obj):
        if isinstance(obj, dict):
            if {'path', 'sha256', 'size_bytes'} <= obj.keys():
                actual = literal_binding(Path(obj['path']))
                if actual != obj:
                    if obj == manifest['producer']:
                        verified(manifest['producer_capture'])
                        assert obj['sha256'] == manifest['producer_capture']['sha256']
                    else:
                        assert obj['path'] in enriched_paths, ('unexpected binding mismatch', obj)
                        stale[obj['path']] = {'producer_pre_enrichment_binding': obj, 'fresh_enriched_binding': actual}
                else:
                    verified(obj)
                bindings[obj['path']] = actual
            else:
                for value in obj.values():
                    walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    for obj in (selection, receipt, manifest, records, result):
        walk(obj)
    assert len(stale) == 3, 'Expected only the three audited post-enrichment tensor binding defects'
    assert manifest['producer']['sha256'] == manifest['producer_capture']['sha256']
    old_selection = read(verified(selection['predecessor_selection']))
    assert read(verified(selection['predecessor_acceptance']))['status'] == 'lead-accepted'
    old_manifest = read(ROOT.parent / '2026-09-22-recurrence-downstream-query-phase/attempt-002/source-to-cell.json')
    old_records = read(ROOT.parent / '2026-09-22-recurrence-downstream-query-phase/attempt-002/readback.json')
    assert manifest['model_identity'] == old_manifest['model_identity']
    assert manifest['native_input_hashes'] == old_selection['native_input_hashes']
    old_logits = torch.load(verified(selection['predecessor_logits']), map_location='cpu', weights_only=True)['full_logits']
    trajectory = torch.load(verified(old_selection['sources']['native_trajectory']), map_location='cpu', weights_only=True)
    group = torch.load(verified(old_selection['sources']['native_capture']), map_location='cpu', weights_only=True)
    baseline = tensors['query_only']['full_logits']
    errors = {'baseline_vs_prior': float((baseline - old_logits).abs().max()),
              'identity_vs_baseline': float((tensors['identity_recompute']['full_logits'] - baseline).abs().max())}
    assert max(errors.values()) <= ATOL
    assert baseline[2, 0].argmax().item() == tensors['identity_recompute']['full_logits'][2, 0].argmax().item() == 152249
    summaries, gates = {}, {}
    for label in ORDER:
        x, rec = tensors[label], records['cells'][label]
        logits, local = x['full_logits'], x['local_prefix']
        assert logits.shape == (4, 2, 152670) and logits.dtype == torch.float32 and torch.isfinite(logits).all()
        assert rec['native_input_hashes'] == old_selection['native_input_hashes']
        assert rec['attention'] == old_records['cells']['held_late_phase']['attention']
        assert rec['embedding_calls'] == rec['rotary_calls'] == 1 and rec['cache_length'] == 2127
        assert rec['query_consumer'] == {'shape': [4, 2, 2048], 'physical_indices': [1703, 2126], 'exact': True}
        assert rec['text_full_sdpa_calls'] == 28 and rec['selected_sdpa_calls'] == (0 if label == 'query_only' else 27)
        assert torch.equal(x['layer0_before'], group['head_output'][0].flatten())
        assert torch.equal(x['layer0_consumed'], group['head_output'][1].flatten())
        incoming = float((x['layer1_input'][0] - trajectory['residual_input'][1][62]).abs().max())
        companion = float((logits[[0, 1, 3]] - baseline[[0, 1, 3]]).abs().max())
        assert incoming <= ATOL and companion <= ATOL
        raw_saved = torch.load(OUT / label / 'raw-prefix.pt', map_location='cpu', weights_only=True)
        maximum_rotation, maximum_score, maximum_sham = 0., 0., 0.
        for j in range(27):
            i = str(j + 1)
            m, lm, raw = rec['layer_metrics'][i], rec['local_prefix']['layers'][i], local['layers'][i]
            ref = old_records['cells']['held_late_phase']['layer_metrics'][i]
            for key in ('historical_K_hash', 'historical_V_hash', 'mask_hash', 'cache_slots_hash'):
                assert m[key] == ref[key]
            for key in ('q_k_off_target_exact', 'gqa_expansion_exact', 'o_proj_consumer_exact'):
                assert m[key]
            assert m['sdpa_calls'] == 1 and lm['full_input_unchanged'] and lm['off_target_output_exact']
            assert lm['full_call_k_hash'] == lm['full_input_k_hash_before'] == lm['full_input_k_hash_after']
            for key, value in raw_saved['layers'][i].items():
                assert torch.equal(value, raw[key]), ('raw/enriched mismatch', label, i, key)
            for key in ('pre_K', 'early_cos', 'early_sin', 'late_cos', 'late_sin'):
                assert torch.equal(raw[key], tensors['query_only']['local_prefix']['layers'][i][key])
            q_late = rotate(x['pre_Q'][j], x['used_phase_cos'][j], x['used_phase_sin'][j])
            k_self = rotate(x['pre_K'][j], x['used_phase_cos'][j], x['used_phase_sin'][j]).repeat_interleave(2, dim=0)
            assert float((q_late - x['consumed_Q_target'][j]).abs().max()) <= ATOL
            assert float((k_self - x['consumed_K_target_after_gqa'][j]).abs().max()) <= ATOL
            for key, value in m.items():
                if key.startswith('fp64_') or key.endswith('max_abs') and key != 'phase_delta_max_abs':
                    assert value <= ATOL, (label, i, key, value)
            if label == 'query_only':
                assert not lm['selected_call']
                continue
            assert lm['selected_call'] and all(lm[key] for key in
                ('selected_nonprefix_k_exact', 'selected_v_exact', 'selected_mask_exact', 'full_inputs_unchanged_after_selected'))
            expected_mask = (torch.arange(2127) <= 1703).reshape(1, 1, 1, 2127)
            assert torch.equal(raw['selected_mask'], expected_mask) and tensor_hash(expected_mask) == lm['selected_mask_hash']
            assert torch.equal(raw['selected_q_target'], x['consumed_Q_target'][j])
            early_k = rotate(raw['pre_K'], raw['early_cos'], raw['early_sin']).repeat_interleave(2, dim=0)
            used_k = rotate(raw['pre_K'], raw['late_cos'], raw['late_sin']).repeat_interleave(2, dim=0) if label == 'prefix_coherent' else early_k
            error = max(float((early_k - raw['selected_k_prefix_before']).abs().max()),
                        float((used_k - raw['selected_k_prefix_used']).abs().max()))
            maximum_rotation = max(maximum_rotation, error)
            assert error <= ATOL
            if label == 'identity_recompute':
                assert torch.equal(raw['selected_k_prefix_before'], raw['selected_k_prefix_used'])
                maximum_sham = max(maximum_sham, float((raw['selected_output_target'] - raw['full_output_target']).abs().max()))
                assert maximum_sham <= ATOL
            else:
                early_q = rotate(x['pre_Q'][j], group['query_cos'][0], group['query_sin'][0])
                old_scores = torch.einsum('hd,hpd->hp', early_q, early_k) / 128 ** .5
                new_scores = torch.einsum('hd,hpd->hp', raw['selected_q_target'].double(), raw['selected_k_prefix_used'].double()) / 128 ** .5
                maximum_score = max(maximum_score, float((old_scores - new_scores).abs().max()))
                assert maximum_score <= ATOL
        y = logits[2, 0].double()
        probability = y.softmax(0)
        values, indices = y.topk(10)
        summaries[label] = {'winner_token': int(indices[0]), 'winner_bin': int(indices[0]) - 151670,
            'top10': [{'token': int(token), 'bin': int(token) - 151670, 'logit': float(value),
                       'probability': float(probability[token])} for token, value in zip(indices, values)],
            'd38_minus_999': float(y[151708] - y[152669]),
            'probes': {str(bin): {'probability': float(probability[151670 + bin]),
                               'rank': int((y > y[151670 + bin]).sum()) + 1} for bin in (38, 999, 579)}}
        assert summaries[label]['winner_token'] == rec['summary']['queries'][0]['winner']
        gates[label] = {'incoming_state_error': incoming, 'companion_error': companion,
                       'prefix_rotation_error': maximum_rotation, 'relative_phase_score_error': maximum_score,
                       'identity_attention_error': maximum_sham}
    tensor_bytes = sum(p.stat().st_size for p in OUT.rglob('*.pt'))
    assert tensor_bytes <= 48 << 20
    checks = ROOT / 'lead-checks/accepted-readback'
    checks.mkdir(exist_ok=True)
    capture = preserve_source(Path(__file__), run_root=checks, relative_name=Path(__file__).name)
    out = {'status': 'root-independent-readback-qualified', 'attempt': str(OUT),
           'source_capture': literal_binding(capture), 'verified_bindings': len(bindings),
           'corrected_tensor_bindings': {label: literal_binding(OUT / label / 'held_late_phase.pt') for label in ORDER},
           'producer_binding_defect': stale, 'parity_errors': errors, 'gates': gates, 'summaries': summaries,
           'model_forwards': 3, 'vision_forwards': 3, 'failed_attempt_model_forwards': 0,
           'full_text_sdpa_calls': 84, 'selected_sdpa_calls': 54, 'tensor_bytes': tensor_bytes,
           'elapsed_seconds': receipt['elapsed_seconds'], 'peak_reserved_bytes': receipt['peak_reserved_bytes']}
    (checks / 'prefix-readback.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'status': out['status'], 'errors': errors, 'gates': gates,
                      'winners': {k: v['winner_bin'] for k, v in summaries.items()}, 'tensor_bytes': tensor_bytes}))


if __name__ == '__main__':
    main()
