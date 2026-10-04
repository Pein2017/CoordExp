import pytest
import torch

from probes.readout_norm import median_norm_factors, rescale_selected_logits


def test_lower_median_and_exact_nonselected_columns():
    rows = torch.tensor([[3., 4.], [0., 10.], [6., 8.], [0., 20.]])
    before = rows.clone()
    factors = median_norm_factors(rows)
    assert factors.tolist() == [2., 1., 1., .5]
    scores = torch.arange(30, dtype=torch.float32).reshape(3, 10)
    original = scores.clone()
    ids = torch.tensor([0, 2, 5, 9])
    result = rescale_selected_logits(scores, ids, factors)
    assert torch.equal(result[:, ids], (scores[:, ids].double() * factors).float())
    other = [1, 3, 4, 6, 7, 8]
    assert torch.equal(result[:, other], scores[:, other])
    assert torch.equal(scores, original) and torch.equal(rows, before)


def test_identity_factors_are_exact_and_input_rows_are_not_read():
    scores = torch.randn(2, 3, 20)
    result = rescale_selected_logits(scores, torch.tensor([3, 8]), torch.ones(2))
    assert torch.equal(result, scores)
    assert result.data_ptr() != scores.data_ptr()


@pytest.mark.parametrize("rows", [torch.zeros(2, 3), torch.tensor([[float('nan')]]), torch.ones(2, dtype=torch.float32), torch.ones(2, 3, dtype=torch.int64)])
def test_invalid_effective_rows_fail(rows):
    with pytest.raises(ValueError):
        median_norm_factors(rows)


@pytest.mark.parametrize("ids,factors", [([1, 1], [1., 1.]), ([-1, 2], [1., 1.]), ([1, 5], [1., 1.]), ([1, 2], [0., 1.]), ([1, 2], [1., float('nan')]), ([1, 2], [1.])])
def test_invalid_columns_or_factors_fail(ids, factors):
    with pytest.raises(ValueError):
        rescale_selected_logits(torch.zeros(2, 5), torch.tensor(ids), torch.tensor(factors))
