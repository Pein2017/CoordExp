from __future__ import annotations

import torch

from probes.coordinate_representation.coordinate_address_readout.bridge import CoordinateAddressReadout
from probes.coordinate_representation.coordinate_address_readout.training import batch_backward


def _caches() -> list[dict[str, torch.Tensor | tuple[int, int]]]:
    torch.manual_seed(1729)
    result = []
    for index, count in enumerate(range(1, 9)):
        coordinate_logits = torch.randn(count, 1000)
        result.append(
            {
                "coordinate_logits": coordinate_logits,
                "full_lse": torch.logsumexp(coordinate_logits, dim=-1) + 2.0,
                "hidden": torch.randn(count, 2048),
                "visual": torch.randn(2, 2048),
                "roles": torch.tensor([(index + row) % 4 for row in range(count)]),
                "targets": torch.tensor([(index * 13 + row) % 1000 for row in range(count)]),
                "grid": (1, 2),
            }
        )
    return result


def test_batch_backward_uses_pooled_token_denominator_and_reference_gradient() -> None:
    sidecar = CoordinateAddressReadout(tuple(range(1000)))
    caches = _caches()
    report = batch_backward(sidecar, caches, check=True)
    assert report["denominator"] == sum(range(1, 9))
    assert report["check"]["loss_abs_diff"] <= 2e-4
    assert report["check"]["max_abs_gradient_diff"] <= 2e-4
    assert report["check"]["unequal_image_counts"] is True
    assert report["check"]["wrong_image_mean_rejected"] is True
    assert report["gradient_norms"]["gain"] > 0
    assert report["gradient_norms"]["q_proj.weight"] == 0
    assert report["gradient_norms"]["k_proj.weight"] == 0
    assert report["gradient_norms"]["role_embeddings.weight"] == 0


def test_second_batch_after_gain_move_has_full_sidecar_gradient_signal() -> None:
    sidecar = CoordinateAddressReadout(tuple(range(1000)))
    with torch.no_grad():
        sidecar.gain.fill_(0.1)
    report = batch_backward(sidecar, _caches(), check=True)
    for name in ("gain", "q_proj.weight", "k_proj.weight", "role_embeddings.weight"):
        assert report["gradient_norms"][name] > 0


def test_batch_contract_rejects_non_eight_image_batches() -> None:
    sidecar = CoordinateAddressReadout(tuple(range(1000)))
    try:
        batch_backward(sidecar, _caches()[:7])
    except ValueError as error:
        assert "exactly eight" in str(error)
    else:
        raise AssertionError("batch helper accepted a non-eight-image batch")
