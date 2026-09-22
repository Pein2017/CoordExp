from __future__ import annotations


import pytest
import torch


from probes.logit_lens import radius as probe


def test_radius_and_direction_corners_are_distinct_and_exact() -> None:
    recipient = torch.tensor([3.0, 4.0, 0.0])
    donor = torch.tensor([0.0, 0.0, 10.0])
    corners, geometry = probe.radius_direction_corners(recipient, donor)

    assert torch.allclose(corners["radius_only"], torch.tensor([6.0, 8.0, 0.0]))
    assert torch.allclose(corners["direction_only"], torch.tensor([0.0, 0.0, 5.0]))
    assert torch.allclose(corners["full_current"], donor)
    assert not torch.equal(corners["radius_only"], corners["direction_only"])
    assert geometry["recipient_radius"] == [5.0]
    assert geometry["donor_radius"] == [10.0]

    recipient_unit = recipient / recipient.norm()
    donor_unit = donor / donor.norm()
    assert probe.validate_corner(
        corners["radius_only"],
        expected_radius=donor.norm(),
        expected_unit=recipient_unit,
        scope="radius_only",
    )["unit_direction_passed"]
    assert probe.validate_corner(
        corners["direction_only"],
        expected_radius=recipient.norm(),
        expected_unit=donor_unit,
        scope="direction_only",
    )["radius_passed"]


def test_radius_direction_corners_fail_closed_on_zero_norm() -> None:
    with pytest.raises(RuntimeError, match="zero recipient radius"):
        probe.radius_direction_corners(torch.zeros(4), torch.ones(4))
    with pytest.raises(RuntimeError, match="zero donor radius"):
        probe.radius_direction_corners(torch.ones(4), torch.zeros(4))


def test_final_rms_normalization_removes_positive_radius_but_preserves_direction() -> None:
    norm = torch.nn.RMSNorm(4, eps=0.0)
    head = torch.nn.Linear(4, 7, bias=False)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
        head.weight.copy_(torch.arange(28, dtype=torch.float32).reshape(7, 4) / 10.0)

    recipient = torch.tensor([3.0, 4.0, 1.0, 2.0])
    donor = torch.tensor([-2.0, 1.0, 5.0, 3.0])
    corners, _ = probe.radius_direction_corners(recipient, donor)
    recipient_logits = probe.project_norm_head(norm, head, recipient)
    donor_logits = probe.project_norm_head(norm, head, donor)
    radius_logits = probe.project_norm_head(norm, head, corners["radius_only"])
    direction_logits = probe.project_norm_head(norm, head, corners["direction_only"])

    assert torch.allclose(radius_logits, recipient_logits, atol=probe.ATOL, rtol=probe.RTOL)
    assert torch.allclose(direction_logits, donor_logits, atol=probe.ATOL, rtol=probe.RTOL)
