from __future__ import annotations

import pytest
import torch

from probes.training_set_completion.coordinate_address_readout.bridge import (
    CoordinateAddressReadout,
    address_permutation,
    coordinate_role_from_prefix,
    grid_cell_addresses,
)


IDS = tuple(range(1000, 2000))
GRAMMAR = dict(
    coordinate_token_ids=IDS,
    object_ref_start_id=10,
    object_ref_end_id=11,
    box_start_id=12,
    box_end_id=13,
    eos_token_id=14,
)


def test_grid_centers_and_wrong_address_order_are_rectangular() -> None:
    addresses = grid_cell_addresses(2, 3)
    assert torch.allclose(addresses[0], torch.tensor((1 / 6, 1 / 4)))
    assert torch.allclose(addresses[-1], torch.tensor((5 / 6, 3 / 4)))
    permutation = address_permutation(2, 3)
    assert permutation.tolist() == [5, 0, 1, 2, 3, 4]
    assert not torch.equal(addresses, addresses.index_select(0, permutation))


def test_role_parser_accepts_only_canonical_coordinate_positions() -> None:
    prefix = [10, 77, 11, 12]
    assert coordinate_role_from_prefix(prefix, **GRAMMAR) == 0
    assert coordinate_role_from_prefix(prefix + [1000], **GRAMMAR) == 1
    assert coordinate_role_from_prefix(prefix + [1000, 1001, 1002], **GRAMMAR) == 3
    assert coordinate_role_from_prefix(prefix + [1000, 1001, 1002, 1003], **GRAMMAR) == -1
    assert coordinate_role_from_prefix(prefix + [1000, 13], **GRAMMAR) == -1
    assert coordinate_role_from_prefix([10, 11, 12], **GRAMMAR) == -1
    assert coordinate_role_from_prefix([999, *prefix], **GRAMMAR) == -1
    assert coordinate_role_from_prefix([10, 12], **GRAMMAR) == -1


def test_future_suffix_does_not_change_an_earlier_role() -> None:
    row = [10, 77, 11, 12, 1000, 1001, 1002, 1003, 13]
    mutated = [10, 77, 11, 12, 1000, 1001, 777, 1003, 13]
    for end in range(4, 7):
        assert coordinate_role_from_prefix(row[:end], **GRAMMAR) == coordinate_role_from_prefix(mutated[:end], **GRAMMAR)
    assert coordinate_role_from_prefix(row[:6], **GRAMMAR) == 2
    assert coordinate_role_from_prefix(mutated[:7], **GRAMMAR) == -1


def test_sidecar_conserves_coordinate_mass_and_noncoordinate_logits() -> None:
    torch.manual_seed(3)
    coordinate_ids = tuple(range(3, 1003))
    bridge = CoordinateAddressReadout(coordinate_ids)
    logits = torch.randn(17, 1100)
    hidden = torch.randn(17, 2048)
    visual = torch.randn(6, 2048)
    roles = torch.tensor([-1, 0, 1, 3] + [index % 4 for index in range(13)])
    output = bridge(logits, hidden, visual, roles, 2, 3)
    assert torch.equal(output[0], logits[0])
    mask = torch.ones(1100, dtype=torch.bool)
    mask[list(coordinate_ids)] = False
    assert torch.equal(output[:, mask], logits[:, mask])
    before = torch.logsumexp(logits[:, list(coordinate_ids)], dim=-1)
    after = torch.logsumexp(output[:, list(coordinate_ids)], dim=-1)
    assert torch.allclose(before, after, atol=1e-6, rtol=0)


def test_gain_gradient_then_qk_and_role_gradients() -> None:
    torch.manual_seed(4)
    bridge = CoordinateAddressReadout(tuple(range(1000)))
    logits = torch.randn(3, 1000)
    hidden = torch.randn(3, 2048)
    visual = torch.randn(4, 2048)
    roles = torch.tensor([0, 1, 2])
    output = bridge(logits, hidden, visual, roles, 2, 2)
    loss = output[:, [0, 7, 99]].sum()
    loss.backward()
    assert bridge.gain.grad is not None and torch.isfinite(bridge.gain.grad)
    assert bridge.gain.grad.abs() > 0
    assert bridge.q_proj.weight.grad is not None and bridge.q_proj.weight.grad.abs().sum() == 0
    assert bridge.k_proj.weight.grad is not None and bridge.k_proj.weight.grad.abs().sum() == 0
    assert bridge.role_embeddings.weight.grad is not None and bridge.role_embeddings.weight.grad.abs().sum() == 0
    assert torch.equal(bridge.gain.detach(), torch.zeros(()))

    with torch.no_grad():
        bridge.gain.fill_(0.25)
    bridge.zero_grad(set_to_none=True)
    output = bridge(logits, hidden, visual, roles, 2, 2)
    output.sum().backward()
    for parameter in (bridge.q_proj.weight, bridge.k_proj.weight, bridge.role_embeddings.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_parameter_count_and_permuted_control_are_fixed() -> None:
    bridge = CoordinateAddressReadout(tuple(range(1000)))
    assert sum(parameter.numel() for parameter in bridge.parameters()) == 262401
    assert bridge.q_proj.bias is None and bridge.k_proj.bias is None
    assert not torch.equal(
        grid_cell_addresses(2, 2),
        grid_cell_addresses(2, 2).index_select(0, address_permutation(2, 2)),
    )
    with pytest.raises(ValueError, match="at least two"):
        address_permutation(1, 1)


def test_wrong_address_axis_order_changes_enabled_readout() -> None:
    torch.manual_seed(19)
    correct = CoordinateAddressReadout(tuple(range(1000)))
    wrong = CoordinateAddressReadout(tuple(range(1000)), permuted=True)
    wrong.load_state_dict(correct.state_dict())
    with torch.no_grad():
        correct.gain.fill_(0.5)
        wrong.gain.fill_(0.5)
    logits = torch.randn(2, 1000)
    hidden = torch.randn(2, 2048)
    visual = torch.randn(6, 2048)
    roles = torch.tensor([0, 1])
    expected = correct(logits, hidden, visual, roles, 2, 3)
    permuted = wrong(logits, hidden, visual, roles, 2, 3)
    assert not torch.equal(expected, permuted)
    bad_geometry = grid_cell_addresses(2, 3).clone()
    bad_geometry[:, 0] = 1.0 - bad_geometry[:, 0]
    geometry_output = correct(
        logits,
        hidden,
        visual,
        roles,
        2,
        3,
        addresses=bad_geometry,
    )
    assert not torch.equal(expected, geometry_output)
