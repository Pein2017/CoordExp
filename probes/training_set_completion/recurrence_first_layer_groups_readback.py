"""CPU factorial readback for the frozen layer0 anchor/pool contrast."""
from pathlib import Path
import json
import math

import torch

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion.recurrence_transition_readback import verified
from src.artifacts.source_provenance import preserve_source
from src.qwen.input_identity import tensor_hash


ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-first-layer-groups')
OUT = ROOT/'attempt-001'
PRIOR = ROOT.parent/'2026-09-22-recurrence-native-trajectory'
ATOL = 2e-4


def attend(groups):
    scores = torch.cat([g[0] for g in groups], dim=-1)
    values = torch.cat([g[1] for g in groups], dim=-2)
    weights = scores.softmax(-1)
    return torch.einsum('hk,hkd->hd', weights, values)


def decompose(a, r, c):
    cells = {f'{i}{j}': attend((a[i], r[j], c)) for i in range(2) for j in range(2)}
    delta = cells['11']-cells['00']
    anchor = .5*(cells['10']-cells['00']+cells['11']-cells['01'])
    pool = .5*(cells['01']-cells['00']+cells['11']-cells['10'])
    interaction = cells['11']-cells['10']-cells['01']+cells['00']
    assert torch.allclose(anchor+pool, delta, atol=1e-12, rtol=1e-12)
    norm = float(delta.norm())
    threshold = 64*torch.finfo(torch.float32).eps*max(float(cells['00'].norm()), float(cells['11'].norm()))
    primary = {'delta_norm': norm, 'active_threshold': threshold, 'active': norm > threshold}
    for name, vector in (('anchor', anchor), ('pool', pool)):
        projection = float((vector*delta).sum())/(norm*norm) if norm > threshold else None
        primary[name] = {'norm': float(vector.norm()), 'signed_projection': projection,
                         'perpendicular_norm': float((vector-projection*delta).norm()) if projection is not None else None}
    primary['interaction_norm'] = float(interaction.norm())
    primary['prediction_pass'] = primary['anchor']['signed_projection'] > .5 if norm > threshold else None
    return cells, primary, {'delta': delta, 'anchor': anchor, 'pool': pool, 'interaction': interaction}


def selfcheck():
    zero = torch.zeros(1, 1, dtype=torch.float64); one = torch.ones_like(zero)
    av = torch.tensor([[[1., 0.]]], dtype=torch.float64)
    rv = torch.tensor([[[0., 1.]]], dtype=torch.float64)
    c = (zero, torch.zeros_like(av))
    for a, r, expected in ((((zero,av),(one,av)), ((zero,rv),(zero,rv)), 1.),
                           (((zero,av),(zero,av)), ((zero,rv),(one,rv)), 0.)):
        _, primary, _ = decompose(a, r, c)
        assert abs(primary['anchor']['signed_projection']-expected) < 1e-12


def main():
    selfcheck()
    checks = ROOT/'lead-checks'; checks.mkdir(exist_ok=True)
    receipt = json.loads((OUT/'receipt.json').read_text())
    manifest = json.loads((OUT/'source-to-cell.json').read_text())
    readback = json.loads((OUT/'readback.json').read_text())
    assert receipt['status'] == 'candidate_complete'
    assert receipt['model_forwards'] == receipt['vision_forwards'] == 1
    bindings = {}
    def walk(x):
        if isinstance(x, dict):
            if {'path','sha256','size_bytes'} <= x.keys():
                p = verified(x); bindings[str(p)] = literal_binding(p)
            else:
                for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    for obj in (receipt, manifest, readback): walk(obj)
    prior_acceptance = json.loads((PRIOR/'lead-acceptance.json').read_text())
    assert prior_acceptance['status'] == 'lead-accepted'
    walk(prior_acceptance)
    prior_consumer = json.loads((PRIOR/'attempt-002/val-readback.json').read_text())
    assert readback['native_input_hashes'] == prior_consumer['native_input_hashes']
    assert readback['target_batch'] == 2 and readback['query_rows'] == [42,89]
    assert readback['query_consumer']['exact'] and readback['query_consumer']['physical_indices'] == [1703,2126]
    assert readback['cache_length'] == 2127 and len(readback['attention']) == 28
    expected_mask_hash = tensor_hash(torch.arange(2127)[None,:] <= torch.tensor([1703,2126])[:,None])
    for current, prior in zip(readback['attention'],prior_consumer['attention'],strict=True):
        assert current['mask_hash'] == prior['mask_hash'] and current['cache_slots_hash'] == prior['cache_slots_hash']
        assert current['query_mask_hash'] == expected_mask_hash and current['query_visibility_exact']
    x = torch.load(OUT/'native.pt', map_location='cpu', weights_only=True)
    prev = torch.load(PRIOR/'attempt-002/val-trajectory.pt', map_location='cpu', weights_only=True)
    assert x['post_K'].shape == x['V'].shape == (8,2127,128)
    assert x['pre_Q'].shape == (2,16,128) and x['query_positions'].tolist() == [1703,2126]
    assert torch.equal(x['query_positions'],prev['query_positions'][[15,62]])
    assert torch.equal(x['pre_Q'],prev['pre_rope_Q'][0][[15,62]])
    assert torch.equal(x['query_cos'],prev['query_cos'][[15,62]])
    assert torch.equal(x['query_sin'],prev['query_sin'][[15,62]])
    assert torch.equal(x['pre_Q'][0], x['pre_Q'][1])
    assert x['full_logits'].shape == (2,152670)
    native_error = float((x['full_logits']-prev['full_logits'][[15,62]]).abs().max())
    assert native_error <= ATOL
    assert x['full_logits'].argmax(-1).tolist() == [151708,152669]
    q = x['pre_Q']*x['query_cos'][:,None]+torch.cat((-x['pre_Q'][...,64:],x['pre_Q'][...,:64]),-1)*x['query_sin'][:,None]
    k = x['post_K'].double().repeat_interleave(2,dim=0)
    v = x['V'].double().repeat_interleave(2,dim=0)
    s = torch.einsum('nhd,hkd->nhk',q.double(),k)/math.sqrt(128)
    native = torch.stack([attend(((s[i,:,:end],v[:,:end]),)) for i,end in enumerate((1704,2127))])
    head_error = float((native-x['head_output']).abs().max())
    assert head_error <= ATOL
    repeated_v = x['V'][:,1563:2121].reshape(8,62,9,128)
    assert torch.equal(repeated_v,repeated_v[:,:1].expand_as(repeated_v))
    assert torch.equal(x['V'][:,1698:1704],x['V'][:,2121:2127])
    a = [(s[i,:,:1563],v[:,:1563]) for i in range(2)]
    r = [(s[i,:,1563:end],v[:,1563:end]) for i,end in enumerate((1698,2121))]
    c = [(s[i,:,start:end],v[:,start:end]) for i,(start,end) in enumerate(((1698,1704),(2121,2127)))]
    cells, primary, components = decompose(a,r,c[1])
    gates = {'full_logits_max_abs': native_error,'native_head_output_max_abs':head_error,
             'old_corner_vs_actual_old_head':float((cells['00']-x['head_output'][0]).abs().max()),
             'new_corner_vs_actual_new_head':float((cells['11']-x['head_output'][1]).abs().max()),
             'current_prefix_only_effect_max_abs':float((cells['00']-native[0]).abs().max())}
    assert max(gates.values()) <= ATOL
    per_head = []
    for h in range(16):
        norm = float(components['delta'][h].square().sum())
        per_head.append({'head':h, 'delta_norm': math.sqrt(norm),
                         'anchor_projection':float((components['anchor'][h]*components['delta'][h]).sum())/norm if norm else None})
    cell_summary = {}
    for i in range(2):
        for j in range(2):
            weights = torch.cat((a[i][0],r[j][0],c[1][0]),-1).softmax(-1)
            cell_summary[f'{i}{j}']={'distance_from_old':float((cells[f'{i}{j}']-cells['00']).norm()),
                                    'distance_from_new':float((cells[f'{i}{j}']-cells['11']).norm()),
                                    'anchor_mass_by_head':weights[:,:1563].sum(-1).tolist(),
                                    'repeat_mass_by_head':weights[:,1563:-6].sum(-1).tolist(),
                                    'prefix_mass_by_head':weights[:,-6:].sum(-1).tolist()}
    torch.save({'cells':cells,'components':components,'native_head_outputs':native},checks/'factorial.pt')
    draw(cells,components,primary,checks)
    source_hash = literal_binding(Path(__file__))['sha256']
    capture = preserve_source(Path(__file__),run_root=ROOT,relative_name=f'groups_readback-{source_hash[:12]}.py')
    result={'status':'lead-qualified-computational-contrast','gates':gates,'primary':primary,
            'cells':cell_summary,'per_head':per_head,'verified_bindings':list(bindings.values()),
            'source':literal_binding(capture),'tensors':literal_binding(checks/'factorial.pt'),
            'scope':'Symmetric decomposition of layer0 head-output movement with current-prefix scores fixed; not final-logit causality.'}
    (checks/'groups-readback.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'gates':gates,'primary':primary}))


def draw(cells, components, primary, checks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    order = ['00','10','01','11']; position = torch.arange(4).numpy()
    axes[0].bar(position-.18,[float((cells[k]-cells['00']).norm()) for k in order],width=.36,label='Distance from old/old')
    axes[0].bar(position+.18,[float((cells[k]-cells['11']).norm()) for k in order],width=.36,label='Distance from new/new')
    axes[0].set(xticks=position,xticklabels=['old A\nold R','new A\nold R','old A\nnew R','new A\nnew R'],
                ylabel='Head-output vector L2 distance',title='Four cells; current-prefix scores fixed')
    axes[0].legend(fontsize=8)
    total = components['delta'].square().sum()
    position = torch.arange(16).numpy()
    for k,shift,color in (('anchor',-.2,'tab:blue'),('pool',.2,'tab:orange')):
        contribution = (components[k]*components['delta']).sum(-1)/total
        axes[1].bar(position+shift,contribution.numpy(),width=.4,color=color,label=k)
    axes[1].axhline(0,color='gray',lw=.8)
    axes[1].set(xticks=position,xlabel='Head (all 16 shown)',ylabel='Signed contribution to total displacement',
                title=f"Sum: anchor {primary['anchor']['signed_projection']:.1%}, pool {primary['pool']['signed_projection']:.1%}")
    axes[1].legend(fontsize=8)
    fig.suptitle('val7511, layer0, native row42 to row89: local computational decomposition')
    for suffix in ('png','pdf'): fig.savefig(checks/f'first-layer-groups.{suffix}',dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
