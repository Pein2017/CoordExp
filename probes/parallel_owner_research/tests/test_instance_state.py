import pytest
import torch
from transformers.cache_utils import DynamicCache

from probes.parallel_owner_research.instance_state import block_grounding, cache_slices, region_indices, transplant


def test_mask_changes_only_named_past_image_entries_and_not_original():
    base = torch.zeros(1,1,8,8).masked_fill(~torch.ones(8,8,dtype=torch.bool).tril(),torch.finfo(torch.float32).min)
    copy = base.clone()
    result = block_grounding(base,[5,6],[1,3],length=8,device='cpu',dtype=torch.float32)
    changed = torch.nonzero(result != base).tolist()
    assert changed == [[0,0,5,1],[0,0,5,3],[0,0,6,1],[0,0,6,3]]
    assert torch.equal(base,copy)
    assert torch.equal(result,block_grounding(None,[5,6],[1,3],length=8,device='cpu',dtype=torch.float32))
    with pytest.raises(ValueError,match='precede'):
        block_grounding(None,[2],[3],length=8,device='cpu',dtype=torch.float32)


def test_transplant_exact_locality_and_self_noop():
    cache = DynamicCache()
    for layer in range(2):
        key = torch.arange(48,dtype=torch.float32).reshape(1,2,6,4)+layer
        cache.update(key,key+100,layer)
    before = [(layer.keys.clone(),layer.values.clone()) for layer in cache.layers]
    self = cache_slices(cache,[2,4])
    assert transplant(cache,self,[2,4])['changed_scalars'] == 0
    donors = [(key+7,value-5) for key,value in self]
    assert transplant(cache,donors,[2,4])['changed_scalars'] == 64
    for layer,(key,value) in zip(cache.layers,before):
        for now,old in [(layer.keys,key),(layer.values,value)]:
            assert torch.equal(now[...,[0,1,3,5],:],old[...,[0,1,3,5],:])
    assert cache.get_seq_length() == 6


def test_visual_grid_mapping_and_equal_count_control():
    prompt = [1,9,9,9,9,2]
    assert region_indices(prompt,[1,4,4],9,[0,0,500,999]) == [1,3]
    assert len(region_indices(prompt,[1,4,4],9,[0,0,999,999],2)) == 2
    with pytest.raises(ValueError,match='match target'):
        region_indices(prompt,[1,4,4],9,[0,0,500,500],2)
