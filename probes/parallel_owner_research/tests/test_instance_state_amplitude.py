import pytest
import torch

from probes.parallel_owner_research.instance_state_amplitude import (ARMS,branch_slices,flatten_slices,norm64,scaled_delta)


def fixture():
    native=[(torch.arange(16,dtype=torch.float32).reshape(1,2,2,4)+10*i,
             torch.arange(16,dtype=torch.float32).reshape(1,2,2,4)+20*i) for i in range(3)]
    return {'native':native,
            'owner_a':[(k+4*(i+1),v-3) for i,(k,v) in enumerate(native)],
            'owner_b':[(k+.25,v+.5) for k,v in native],
            'background':[(k-.0625,v+.125) for k,v in native]}


def test_frozen_global_norm_matching_and_random_reproducibility():
    c=fixture();base=flatten_slices(c['native']);an=norm64(flatten_slices(c['owner_a'])-base);bn=norm64(flatten_slices(c['owner_b'])-base)
    for arm in ['b_to_a','background_to_a','random_to_a','a_to_b']:
        value,stats=branch_slices(c,arm)
        assert norm64(flatten_slices(value)-base)==pytest.approx(bn if arm=='a_to_b' else an,rel=2e-6)
        assert stats['norm_rule'].startswith('single global')
    a,_=branch_slices(c,'random_to_a');b,_=branch_slices(c,'random_to_a')
    assert torch.equal(flatten_slices(a),flatten_slices(b))
    # Different input dimensions must not silently rescale each layer separately.
    b,stats=branch_slices(c,'b_to_a')
    delta=flatten_slices(b)-base
    assert delta[:16].mean()==pytest.approx(delta[32:48].mean())


def test_exact_original_self_and_component_bytes():
    c=fixture()
    for arm,source in [('native_self','native'),('original_a','owner_a')]:
        value,_=branch_slices(c,arm)
        assert torch.equal(flatten_slices(value),flatten_slices(c[source]))
    k,_=branch_slices(c,'a_k_only');v,_=branch_slices(c,'a_v_only')
    for i in range(3):
        assert torch.equal(k[i][0],c['owner_a'][i][0]) and torch.equal(k[i][1],c['native'][i][1])
        assert torch.equal(v[i][1],c['owner_a'][i][1]) and torch.equal(v[i][0],c['native'][i][0])
    assert len(ARMS)==8


def test_zero_norm_fails_closed_without_fallback():
    with pytest.raises(ValueError,match='zero-norm'):
        scaled_delta(torch.zeros(8),1.)
    c=fixture();c['owner_b']=c['native']
    with pytest.raises(ValueError,match='zero-norm'):
        branch_slices(c,'b_to_a')
