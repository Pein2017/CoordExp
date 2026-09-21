from __future__ import annotations


import pytest
import torch


from probes.logit_lens import causal as probe


class _Layer(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * 1.0


def test_residual_patch_changes_only_declared_position() -> None:
    layer = _Layer()
    value = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    replacement = torch.full((1, 6), -7.0)
    with probe.ResidualPatch(
        layer,
        positions=[2],
        replacement=replacement,
        expected_before=value[0, 2, :].unsqueeze(0),
    ) as patch:
        result = layer(value)
    assert torch.equal(result[0, 2], replacement[0])
    assert torch.equal(result[0, :2], value[0, :2])
    assert torch.equal(result[0, 3:], value[0, 3:])
    assert patch.receipt()["non_target_residual_exact"] is True


def test_residual_patch_rejects_shifted_site_alignment() -> None:
    layer = _Layer()
    value = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    patch = probe.ResidualPatch(
        layer,
        positions=[2],
        replacement=torch.zeros(1, 6),
        # Deliberately bind the expected state from the prior token position.
        expected_before=value[0, 1, :].unsqueeze(0),
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        with patch:
            layer(value)


def test_residual_patch_rejects_corrupted_receiver_state() -> None:
    layer = _Layer()
    value = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    corrupted = value[0, 2, :].clone()
    corrupted[0] += 1.0
    patch = probe.ResidualPatch(
        layer,
        positions=[2],
        replacement=torch.zeros(1, 6),
        expected_before=corrupted.unsqueeze(0),
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        with patch:
            layer(value)


def test_random_delta_is_deterministic_and_norm_matched() -> None:
    recipient = torch.linspace(-1, 1, 32)
    donor = recipient + torch.linspace(0.2, 1.3, 32)
    first, first_receipt = probe._random_replacement(recipient, donor, seed=104729)
    second, second_receipt = probe._random_replacement(recipient, donor, seed=104729)
    third, _ = probe._random_replacement(recipient, donor, seed=104759)
    assert torch.equal(first, second)
    assert first_receipt["unit_direction_sha256"] == second_receipt["unit_direction_sha256"]
    assert not torch.equal(first, third)
    assert torch.allclose((first - recipient).norm(), (donor - recipient).norm(), atol=1e-5, rtol=1e-5)
    assert first_receipt["norm_match_atol"] == 1e-5
    assert first_receipt["norm_match_rtol"] == 1e-5
    assert first_receipt["norm_match_passed"] is True


def test_middle_coordinate_selector_freezes_four_sites_or_explicit_exclusion() -> None:
    def row(offset: int) -> list[int]:
        return [
            probe.parent.OBJECT_REF_START,
            100 + offset,
            probe.parent.OBJECT_REF_END,
            probe.parent.BOX_START,
            probe.parent.COORD_START + offset,
            probe.parent.COORD_START + offset + 1,
            probe.parent.COORD_START + offset + 2,
            probe.parent.COORD_START + offset + 3,
            probe.parent.BOX_END,
        ]

    sites, exclusion = probe.middle_coordinate_sites(
        [0] * 5,
        {"token_ids": [*row(0), *row(10), *row(20), probe.parent.IM_END]},
    )
    assert exclusion is None
    assert len(sites) == 4
    assert [site["actual_next_token_id"] for site in sites] == [
        probe.parent.COORD_START + 10 + index for index in range(4)
    ]

    empty, exclusion = probe.middle_coordinate_sites([0] * 5, {"token_ids": [probe.parent.IM_END]})
    assert empty == []
    assert exclusion == "no_completed_middle_row"


def test_reduce_trace_preserves_population_exclusions_and_does_not_clip_R() -> None:
    rows = [
        {
            "direction": "overfit_to_source",
            "block_1based": 16,
            "scope": "current",
            "control_seed": None,
            "eligible_for_R": True,
            "patch_minus_receiver": 5.0,
            "donor_minus_receiver": 2.0,
            "donorward": True,
        },
        {
            "direction": "overfit_to_source",
            "block_1based": 16,
            "scope": "current",
            "control_seed": None,
            "eligible_for_R": False,
            "patch_minus_receiver": 1000.0,
            "donor_minus_receiver": 1000.0,
            "donorward": None,
        },
    ]
    group = probe.reduce_trace(rows)["groups"][0]
    assert group["population_site_count"] == 2
    assert group["eligible_site_count"] == 1
    assert group["excluded_site_count"] == 1
    assert group["raw_numerator_sum"] == 5.0
    assert group["raw_denominator_sum"] == 2.0
    assert group["R_unclipped"] == 2.5


def test_endpoint_eligibility_retains_noncoordinate_and_equal_reasons() -> None:
    start = probe.parent.COORD_START
    assert probe._eligible_endpoint(start + 1, start + 2) == (True, None)
    assert probe._eligible_endpoint(start + 1, start + 1) == (False, "equal_endpoint_tokens")
    assert probe._eligible_endpoint(17, start + 1) == (False, "overfit_endpoint_noncoordinate")
    assert probe._eligible_endpoint(start + 1, 17) == (False, "source_endpoint_noncoordinate")
