"""CPU characterization of the shared coordinate policy's arithmetic and graph."""

import math
from types import SimpleNamespace

import pytest
import torch

from probes.rule_stability.policy import MedianPolicy
from src.losses.token_scores import aligned_token_logprobs


def _fixture():
    # Neither coordinate order nor selected-delta row order is the vocabulary order.
    ids = torch.arange(1, 1001).roll(37)
    rows = torch.arange(2, 1002).flip(0)
    weight = torch.zeros(1003, 2, dtype=torch.bfloat16)
    weight[:, 0] = 1
    weight[ids[17], 0] = 2
    delta = torch.nn.Parameter(torch.zeros(1002, 2))
    with torch.no_grad():
        delta[rows, 0] = torch.arange(1000).div(250, rounding_mode="floor").float()
        # FP32 effective addition loses this increment; FP64 addition would retain it.
        delta[rows[:250], 0] = 2.0 ** -25
        # The FP64 norm retains this increment; an FP32 norm would round to one.
        delta[rows[42], 1] = 2.0 ** -12
        delta[:2] = 123
    selected = torch.empty(1002, dtype=torch.long)
    selected[:2] = torch.tensor([0, 1002])
    selected[rows] = ids
    head = SimpleNamespace(weight=weight, shared_embed_delta=delta,
                           selected_token_ids=selected, bias=None)
    model = SimpleNamespace(get_output_embeddings=lambda: head,
        get_input_embeddings=lambda: SimpleNamespace(shared_embed_delta=torch.zeros_like(delta)))
    return model, weight, delta, ids, rows


def _expected_norms():
    expected = torch.arange(1000).div(250, rounding_mode="floor").double() + 1
    expected[17] = 2
    expected[42] = math.sqrt(1 + 2.0 ** -24)
    return expected


def test_frozen_median_precision_and_selected_row_mapping():
    model, _, _, ids, _ = _fixture()
    factors = MedianPolicy(model, ids).factors()
    expected = 2 / _expected_norms()
    assert factors.dtype == torch.float64
    assert torch.equal(factors, expected)
    assert factors[0] == 2
    assert factors[42] < 2
    assert factors[499] == 1


def test_shared_norm_values_keep_frozen_arithmetic_and_tensor_results():
    from src.qwen.coordinate_policy import coordinate_norm_values

    _, weight, delta, ids, rows = _fixture()
    values = coordinate_norm_values(weight, delta, ids, rows)
    assert set(values) == {"factors", "norms", "median"}
    for value in values.values():
        assert value.dtype == torch.float64 and value.device == weight.device
        assert value.requires_grad and value.grad_fn is not None
    assert torch.equal(values["norms"], _expected_norms())
    assert values["median"].shape == () and values["median"] == 2
    assert torch.equal(values["factors"], 2 / _expected_norms())


def test_shared_norm_current_factor_gradient_and_detachment_counterexample():
    from src.qwen.coordinate_policy import coordinate_norm_values, scale_coordinate_logits

    _, weight, delta, ids, rows = _fixture()
    # Distinct norms make the current lower-median row and its derivative unique.
    with torch.no_grad():
        weight[:, 0] = 1
        delta[rows, 0] = torch.arange(1000) / 1000
        delta[rows, 1] = 0
    raw = torch.linspace(-2, 2, 1003)[None].requires_grad_()
    target = ids[799].reshape(1)
    values = coordinate_norm_values(weight, delta, ids, rows)
    policy = scale_coordinate_logits(raw, ids, values["factors"])
    objective = aligned_token_logprobs(policy, target).sum()
    gradient, raw_gradient = torch.autograd.grad(objective, (delta, raw))
    assert gradient[rows[799], 0].abs() > 0.1
    assert gradient[rows[499], 0].abs() > 0.1
    assert torch.count_nonzero(gradient[:2]) == 0
    assert torch.isfinite(raw_gradient).all() and raw_gradient[0, target].abs() > 0
    detached = scale_coordinate_logits(raw, ids, values["factors"].detach())
    assert torch.equal(detached, policy)
    detached_objective = aligned_token_logprobs(detached, target).sum()
    assert torch.autograd.grad(detached_objective, delta, allow_unused=True)[0] is None
    step = 0.001
    for position in (799, 499):
        with torch.no_grad():
            original = delta[rows[position], 0].clone()
            samples = []
            for change in (step, -step):
                delta[rows[position], 0] = original + change
                current = coordinate_norm_values(weight, delta, ids, rows)
                scores = scale_coordinate_logits(raw, ids, current["factors"])
                samples.append(float(aligned_token_logprobs(scores, target).sum()))
            delta[rows[position], 0] = original
        finite_difference = (samples[0] - samples[1]) / (2 * step)
        assert float(gradient[rows[position], 0]) == pytest.approx(
            finite_difference, rel=0.003, abs=0.001)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("inplace", [False, True])
def test_shared_scale_cast_order_and_raw_logits_ownership(dtype, inplace):
    from src.qwen.coordinate_policy import scale_coordinate_logits

    ids = torch.arange(1, 1001).roll(37)
    factors = torch.linspace(0.51, 1.53, 1000, dtype=torch.float64)
    logits = torch.linspace(-2, 2, 2006, dtype=dtype).reshape(2, 1003)
    saved = logits.clone()
    expected = logits.clone()
    expected[..., ids] = (logits[..., ids].double() * factors).to(dtype)
    # This fixture distinguishes multiplying after an early factor cast.
    rounded = (logits[..., ids] * factors.to(dtype)).to(dtype)
    assert not torch.equal(rounded, expected[..., ids])
    actual = scale_coordinate_logits(logits, ids, factors, inplace=inplace)
    assert torch.equal(actual, expected) and actual.dtype == dtype
    assert torch.equal(actual[..., [0, 1001, 1002]], saved[..., [0, 1001, 1002]])
    if inplace:
        assert actual is logits
    else:
        assert actual.data_ptr() != logits.data_ptr()
        assert torch.equal(logits, saved)


def test_resident_scaling_extracts_no_device_scalars(monkeypatch):
    from src.qwen.coordinate_policy import scale_coordinate_logits

    ids = torch.arange(1, 1001)
    factors = torch.linspace(0.51, 1.53, 1000, dtype=torch.float64)
    scores = torch.ones(1, 1003)
    expected = scale_coordinate_logits(scores, ids, factors)

    def forbidden(*args, **kwargs):
        raise AssertionError("resident scaling extracted a tensor scalar")

    with monkeypatch.context() as patch:
        for name in ("__int__", "__float__", "__bool__", "item", "tolist", "cpu", "numpy"):
            patch.setattr(torch.Tensor, name, forbidden)
        actual = scale_coordinate_logits(scores, ids, factors, inplace=True)
    assert actual is scores and torch.equal(actual, expected)


@pytest.mark.parametrize("inplace", [False, True])
def test_scaling_fails_closed_on_outside_vocabulary_ids(inplace):
    from src.qwen.coordinate_policy import scale_coordinate_logits

    ids = torch.arange(1000)
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        scale_coordinate_logits(torch.ones(1, 999), ids, torch.ones(1000), inplace=inplace)


@pytest.mark.parametrize("fault", ["base-dtype", "base-rank", "delta-dtype", "delta-width",
    "nonfinite-delta", "zero-norm", "nonfinite-base", "ids-dtype", "ids-rank",
    "ids-count", "ids-bounds", "row-dtype", "row-shape", "missing-row", "row-bounds"])
def test_shared_norm_rejects_invalid_effective_rows(fault):
    from src.qwen.coordinate_policy import coordinate_norm_values

    _, weight, delta, ids, rows = _fixture()
    if fault == "base-dtype":
        weight = weight.float()
    elif fault == "base-rank":
        weight = weight[None]
    elif fault == "delta-dtype":
        delta = delta.bfloat16()
    elif fault == "delta-width":
        delta = delta[:, :1]
    elif fault == "nonfinite-delta":
        delta = delta.detach().clone()
        delta[0, 0] = torch.nan  # Even an unselected delta row must remain finite.
    elif fault in ("zero-norm", "nonfinite-base"):
        weight[ids[0]] = 0 if fault == "zero-norm" else torch.inf
        delta = delta.detach().clone()
        delta[rows[0]] = 0
    elif fault == "ids-dtype":
        ids = ids.float()
    elif fault == "ids-rank":
        ids = ids[None]
    elif fault == "ids-count":
        ids, rows = ids[:-1], rows[:-1]
    elif fault == "ids-bounds":
        ids[0] = weight.shape[0]
    elif fault == "row-dtype":
        rows = rows.float()
    elif fault == "row-shape":
        rows = rows[:-1]
    elif fault == "missing-row":
        rows[0] = -1
    else:
        rows[0] = delta.shape[0]
    with pytest.raises(ValueError):
        coordinate_norm_values(weight, delta, ids, rows)


def test_probe_reexports_shared_policy_and_singleton_placeholder_scope():
    from probes.rule_stability.policy import _coordinate_ids, _scale_coordinates
    from probes.rule_stability.replay import prompt_only_placeholder_masks as old_masks
    from src.qwen.coordinate_policy import MedianPolicy as shared_policy
    from src.qwen.coordinate_policy import _coordinate_ids as shared_ids
    from src.qwen.coordinate_policy import scale_coordinate_logits
    from src.qwen.native import prompt_only_placeholder_masks

    assert MedianPolicy is shared_policy
    assert _coordinate_ids is shared_ids
    assert _scale_coordinates is scale_coordinate_logits
    assert old_masks is prompt_only_placeholder_masks
