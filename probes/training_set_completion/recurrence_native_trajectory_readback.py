"""Independent CPU readback of the frozen two-case native trajectory assay."""
from pathlib import Path
import json
import math

import torch

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion.recurrence_transition_readback import verified
from src.artifacts.source_provenance import preserve_source
from src.qwen.input_identity import tensor_hash


ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-native-trajectory')
ATTEMPTS = {'val': ROOT/'attempt-002', 'train': ROOT/'attempt-003'}
CHECKS = ROOT / 'lead-checks'
FIXED = ROOT.parent / '2026-09-22-recurrence-fixed-template/attempt-001'
ATOL = 2e-4


def load(path):
    return json.loads(Path(path).read_text())


def phase_rotate(x, cos, sin):
    # Independent FP64 complex-pair expression, with captured native coefficients.
    a, b = x.double().chunk(2, dim=-1)
    c, s = cos.double()[..., :a.shape[-1]], sin.double()[..., :a.shape[-1]]
    return torch.cat((a*c-b*s, b*c+a*s), dim=-1)


def movement(x):
    x = x.double().flatten(1)
    delta = (x[1:] - x[:-1]).norm(dim=1)
    total = float(delta.sum())
    threshold = 64 * torch.finfo(torch.float32).eps * float(x.norm(dim=1).max())
    chord = x[-1]-x[0]
    norm = float(chord.norm())
    geometry = None
    if norm > threshold:
        displacement = x-x[0]
        projection = displacement@chord/(norm*norm)
        perpendicular = (displacement-projection[:, None]*chord).norm(dim=1)/norm
        geometry = {'chord_to_path': norm/total, 'projection': projection.tolist(),
                    'perpendicular_over_chord': perpendicular.tolist(),
                    'projection_backsteps_below_minus_1e4': int(((projection[1:]-projection[:-1]) < -1e-4).sum())}
    return dict(total=total, threshold=threshold, active=total > threshold, geometry=geometry,
                delta=delta.tolist(), late_fraction=float(delta[15:].sum()/total)
                if len(delta) == 61 and total > threshold else None)


def readback():
    # Frozen metric sensitivity: entry-only and uniformly distributed changes differ.
    jump = torch.ones(62, 1); jump[0] = 0
    assert movement(jump)['late_fraction'] == 0
    assert abs(movement(torch.arange(62.)[:, None])['late_fraction']-46/61) < 1e-12
    assert not movement(torch.ones(62, 1))['active']
    receipts = [load(ROOT/f'attempt-{i:03d}/receipt.json') for i in (1, 2, 3)]
    assert [r['status'] for r in receipts] == ['technical_invalid', 'technical_invalid', 'candidate_complete']
    assert [r['model_forwards'] for r in receipts] == [0, 1, 1]
    assert [r['vision_forwards'] for r in receipts] == [0, 1, 1]
    bindings = {}

    def walk(x):
        if isinstance(x, dict):
            if {'path', 'sha256', 'size_bytes'} <= x.keys():
                p = verified(x); bindings[str(p)] = literal_binding(p)
            else:
                for value in x.values(): walk(value)
        elif isinstance(x, list):
            for value in x: walk(value)

    oracle = load(CHECKS/'selection-oracle.json')
    prediction_meta = load(CHECKS/'first-layer-preregistered-prediction.json')
    for x in (*receipts, oracle, prediction_meta): walk(x)
    result = {'status': 'independently-verified', 'cases': {}, 'cost': {
        'model_forwards': 2, 'vision_forwards': 2,
        'attempt_elapsed_seconds': [r['elapsed_seconds'] for r in receipts],
        'reported_train_peak_reserved_bytes': receipts[-1]['peak_reserved_bytes'],
        'qualified_memory_peak_bytes': None,
        'memory_telemetry_limit': 'Producer queried default CUDA device rather than cuda:4; reported zero is not an actual run memory peak.'},
        'recovery': 'val retained native capture from technical-invalid attempt002; wrong first/penultimate postprocessing rechecked independently on CPU; original receipts unchanged.'}
    plot = {}
    for name, count, target in (('val', 63, 2), ('train', 5, 1)):
        out = ATTEMPTS[name]
        manifest = load(out/'source-to-cell.json')
        # Mutable maintained source may now implement the next bounded attempt.
        # Qualify the exact producer captured for THIS invocation instead.
        assert manifest['producer']['sha256'] == manifest['producer_capture']['sha256']
        walk({k: v for k, v in manifest.items() if k != 'producer'})
        attested = load(out/f'{name}-readback.json'); walk(attested)
        x = torch.load(out/f'{name}-trajectory.pt', map_location='cpu', weights_only=True)
        logits = x['full_logits'].double()
        assert logits.shape == (count, 152670) and torch.isfinite(logits).all()
        assert attested['target_batch'] == target
        assert attested['query_consumer']['exact']
        assert x['query_positions'].tolist() == [q['physical_query_index'] for q in oracle['cases'][name]['queries']]
        assert attested['query_consumer']['physical_indices'] == x['query_positions'].tolist()
        assert len(attested['attention']) == len(attested['cache']) == 28
        assert all(a['query_visibility_exact'] and a['cache_is_native_empty_at_entry'] for a in attested['attention'])
        assert attested['cache_gate_max_abs'] <= ATOL
        for k, expected in oracle['cases'][name]['native_input_hashes'].items():
            assert attested['native_input_hashes'][k] == expected
        packet = torch.load(manifest['cases'][name]['accepted_paths']['phase_packet']['path'], map_location='cpu', weights_only=True)
        phase_error = 0.; complex_error = 0.; layers = []
        for i in range(28):
            key, value = x['pre_rope_K'][i], x['V'][i]
            assert key.shape == value.shape == (count-1, 8, 9, 128)
            flat_value = value.permute(1, 0, 2, 3).reshape(8, -1, 128)
            assert tensor_hash(flat_value) == attested['cache'][i]['stored_V_hash']
            # Exact FP32 native expression independently reconstructs cached K.
            half = key.shape[-1]//2
            rotated = key*x['row_cos'][:, None] + torch.cat((-key[..., half:], key[..., :half]), -1)*x['row_sin'][:, None]
            flat_key = rotated.permute(1, 0, 2, 3).reshape(8, -1, 128)
            assert tensor_hash(flat_key) == attested['cache'][i]['stored_postK_hash']
            for index, pre, post in ((-2, 'pre_old', 'post_old'), (-1, 'pre_new', 'post_new')):
                phase_error = max(phase_error, float((key[index]-packet['layers'][i][pre]).abs().max()),
                                  float((rotated[index]-packet['layers'][i][post]).abs().max()))
                independent = phase_rotate(key[index], x['row_cos'][index], x['row_sin'][index])
                complex_error = max(complex_error, float((independent-packet['layers'][i][post]).abs().max()))
            phase_error = max(phase_error, float((value[-1]-packet['layers'][i]['native_V_dest']).abs().max()))
            layers.append({'layer': i, 'K': movement(key), 'V': movement(value)})
        assert phase_error <= ATOL and complex_error <= ATOL
        top = logits.topk(5, dim=-1); probs = logits.softmax(-1)
        trace_error = 0.; points = []
        bins = (38, 999) if name == 'val' else (350, 348, 591)
        for i, q in enumerate(oracle['cases'][name]['queries']):
            assert int(top.indices[i, 0]) == q['chosen_token']
            assert top.indices[i, :2].tolist() == q['top2_tokens']
            error = float((top.values[i, :2]-torch.tensor(q['top2_logits'], dtype=torch.float64)).abs().max())
            trace_error = max(error, trace_error)
            points.append({'row': q['row'], 'winner_bin': int(top.indices[i, 0])-151670,
                           'gap': float(top.values[i, 0]-top.values[i, 1]),
                           'top5': [{'token': int(t), 'bin': int(t)-151670, 'logit': float(v), 'P': float(probs[i, t])}
                                    for t, v in zip(top.indices[i], top.values[i])],
                           'bin_logits': {str(b): float(logits[i, 151670+b]) for b in bins},
                           'bin_P': {str(b): float(probs[i, 151670+b]) for b in bins}})
        assert trace_error <= ATOL
        nn = torch.load(FIXED/name/'NN.pt', map_location='cpu', weights_only=True).double()
        final_error = float((logits[-1]-nn).abs().max()); assert final_error <= ATOL
        fixed = torch.load(FIXED/name/'first-template-and-phases.pt', map_location='cpu', weights_only=True)
        first_error = max(float((x[key][i][0]-fixed[ref][i]).abs().max())
                          for key, ref in (('pre_rope_K', 'first_preK'), ('V', 'first_V')) for i in range(28))
        assert first_error <= ATOL
        layer0 = {key: float((x[key][0]-x[key][0][0]).abs().max()) for key in
                  ('pre_rope_K', 'V', 'pre_rope_Q', 'residual_input', 'residual_output')}
        settling = {}
        if name == 'val':
            for key in ('K', 'V'):
                active = [l[key] for l in layers if l[key]['active']]
                n = sum(l['late_fraction'] <= .10 for l in active)
                settling[key] = {'active_layers': len(active), 'settled_layers': n,
                                 'criterion_pass': bool(active) and n/len(active) >= .9,
                                 'late_fraction_range': [min(l['late_fraction'] for l in active), max(l['late_fraction'] for l in active)]}
        result['cases'][name] = dict(layers=layers, points=points, settling=settling,
            layer0_constancy_max_abs=layer0, trace_top2_max_abs=trace_error,
            final_vector_max_abs=final_error, first_template_max_abs=first_error,
            corrected_penultimate_last_phase_error=phase_error, independent_complex_phase_error=complex_error,
            tensor_bytes=(out/f'{name}-trajectory.pt').stat().st_size)
        plot[name] = (x, logits)
    x, _ = plot['val']
    logz = torch.full((63, 16), -torch.inf, dtype=torch.float64)
    for n in range(1, 63):
        q = phase_rotate(x['pre_rope_Q'][0][n], x['query_cos'][n], x['query_sin'][n])
        k = phase_rotate(x['pre_rope_K'][0][:n], x['row_cos'][:n, None], x['row_sin'][:n, None])
        k = k.permute(1, 0, 2, 3).reshape(8, n*9, 128).repeat_interleave(2, dim=0)
        scores = torch.einsum('hd,hkd->hk', q, k)/math.sqrt(128)
        logz[n] = scores.logsumexp(-1)
    pred = torch.load(CHECKS/'first-layer-preregistered-prediction.pt', map_location='cpu', weights_only=True)
    error = float((logz[1:]-pred['logZ_direct_from_native_coefficients'][1:]).abs().max())
    result['first_layer_prediction'] = {'logZ_max_abs_vs_preregistered': error,
        'source_constancy_satisfied': all(result['cases']['val']['layer0_constancy_max_abs'][k] == 0 for k in ('pre_rope_K','V','pre_rope_Q')),
        'heads': 16, 'nonempty_queries': 62, 'observed_FP32_coefficient_logZ_decreases': int((logz[2:]-logz[1:-1] < 0).sum()),
        'scope': 'Unnormalized repeated-row score mass only; not final readout or normalized attention.'}
    torch.save({'logZ': logz}, CHECKS/'first-layer-observed.pt')
    draw(result, logz)
    source_hash = literal_binding(Path(__file__))['sha256']
    capture = preserve_source(Path(__file__), run_root=ROOT, relative_name=f'recurrence_native_trajectory_readback-{source_hash[:12]}.py')
    result['source'] = literal_binding(capture)
    result['verified_bindings'] = list(bindings.values())
    (CHECKS/'trajectory-readback.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: {'settling': v['settling'], 'layer0': v['layer0_constancy_max_abs'],
                         'trace_error': v['trace_top2_max_abs'], 'final_error': v['final_vector_max_abs']}
                      for k, v in result['cases'].items()}))
    print(json.dumps(result['first_layer_prediction']))


def draw(result, logz):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    p = result['cases']['val']['points']; rows = [q['row'] for q in p]
    axes[0, 0].plot(rows, [q['bin_logits']['38']-q['bin_logits']['999'] for q in p], label='z38 - z999')
    axes[0, 0].axhline(0, color='gray', lw=.8); axes[0, 0].axvline(89, color='red', ls=':', label='numeric exit')
    axes[0, 0].set(title='val7511 native x2 readout', xlabel='Zero-based output row', ylabel='Logit margin')
    axes[0, 0].legend()
    for k, c in (('K', 'tab:blue'), ('V', 'tab:orange')):
        for l in result['cases']['val']['layers']:
            if l[k]['active']:
                d = torch.tensor(l[k]['delta'], dtype=torch.float64)
                axes[0, 1].plot(range(28, 89), d.cumsum(0)/d.sum(), color=c, alpha=.25)
        axes[0, 1].plot([], [], color=c, label=k)
    axes[0, 1].axvline(42, color='gray', ls=':'); axes[0, 1].axhline(.9, color='gray', ls=':')
    axes[0, 1].set(title='Each active layer: accumulated geometric movement', xlabel='Destination row of transition', ylabel='Fraction of total path length')
    axes[0, 1].legend()
    for h in range(16): axes[1, 0].plot(range(1, 63), logz[1:, h], alpha=.65)
    axes[1, 0].set(title='Layer 0: complete repeated-row score mass, 16 heads', xlabel='Number of previous complete repeats', ylabel='log Z (unnormalized)')
    p = result['cases']['train']['points']; rows = [q['row'] for q in p]
    for b in ('350', '591'):
        axes[1, 1].plot(rows, [q['bin_logits'][b]-q['bin_logits']['348'] for q in p], marker='o', label=f'z{b} - z348')
    axes[1, 1].axhline(0, color='gray', lw=.8); axes[1, 1].legend()
    axes[1, 1].set(title='train269858 native x1 readout (short sequence)', xlabel='Zero-based output row', ylabel='Logit margin', xticks=rows)
    fig.suptitle('Native trajectory; descriptive evidence, fixed mature untied+axis step2444')
    for suffix in ('png', 'pdf'): fig.savefig(CHECKS/f'native-trajectory.{suffix}', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    readback()
