from __future__ import annotations

from pathlib import Path

import pytest
import torch

from probes.training_set_completion import source256_ranking_training as ranking


def test_pair_rank_uses_reference_delta_and_max_recorded_length() -> None:
    preferred = torch.tensor(-10.0, dtype=torch.float32, requires_grad=True)
    rejected = torch.tensor(-10.0, dtype=torch.float32, requires_grad=True)
    loss, card = ranking.pair_terms(
        preferred,
        rejected,
        reference_preferred=-10.0,
        reference_rejected=-10.0,
        denominator=3084,
    )
    torch.testing.assert_close(loss, torch.tensor(torch.log(torch.tensor(2.0))))
    assert card["recorded_length_denominator"] == 3084
    assert card["delta_from_reference"] == pytest.approx(0.0)
    loss.backward()
    torch.testing.assert_close(preferred.grad, torch.tensor(-1 / (2 * 3084), dtype=torch.float32))
    torch.testing.assert_close(rejected.grad, torch.tensor(1 / (2 * 3084), dtype=torch.float32))


def test_ranking_global_weight_is_half_without_world_size_division() -> None:
    preferred = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    rejected = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    rank_loss, _ = ranking.pair_terms(
        preferred,
        rejected,
        reference_preferred=0.0,
        reference_rejected=0.0,
        denominator=3084,
    )
    objective = ranking.objective_from_terms(
        [torch.zeros(())],
        [torch.zeros(())],
        [torch.zeros(())],
        [torch.zeros(())],
        [rank_loss],
        ranking_enabled=True,
        branch_denominator=1,
    )
    torch.testing.assert_close(objective, torch.tensor(0.5 * torch.log(torch.tensor(2.0))))
    objective.backward()
    torch.testing.assert_close(preferred.grad, torch.tensor(-1 / (4 * 3084), dtype=torch.float32))
    torch.testing.assert_close(rejected.grad, torch.tensor(1 / (4 * 3084), dtype=torch.float32))


def test_objective_preserves_geometry_scale_and_p_skips_ranking() -> None:
    canonical_ce = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    preferred_ce = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    canonical_geometry = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    preferred_geometry = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
    objective = ranking.objective_from_terms(
        [canonical_ce],
        [canonical_geometry],
        [preferred_ce],
        [preferred_geometry],
        branch_denominator=1,
    )
    expected = 0.5 * (2.0 + 0.01 * 3.0 + 4.0 + 0.01 * 5.0)
    torch.testing.assert_close(objective, torch.tensor(expected, dtype=torch.float64))
    with pytest.raises(ValueError, match="P objective cannot carry ranking"):
        ranking.objective_from_terms(
            [canonical_ce],
            [canonical_geometry],
            [preferred_ce],
            [preferred_geometry],
            [torch.tensor(1.0)],
            branch_denominator=1,
        )


def test_rejected_route_path_has_no_geometry_term() -> None:
    logits = torch.zeros(3, 8, dtype=torch.float32, requires_grad=True)
    route = {
        "continuation_token_ids": [1, 2, 3],
        "ce_weights": [1, 1, 1],
        "trusted_boxes": [{"x1_position": 0, "y1_position": 1, "x2_position": 1, "y2_position": 2}],
    }
    hinge = {
        "coordinate_token_ids": list(range(8)),
        "coordinate_bin_values": list(range(8)),
        "margin": 1 / 999,
    }
    original = ranking.training.raw_axis_validity_hinge
    ranking.training.raw_axis_validity_hinge = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rejected geometry was evaluated"))
    try:
        terms = ranking.route_terms(logits, route, hinge, include_geometry=False)
    finally:
        ranking.training.raw_axis_validity_hinge = original
    assert terms["geometry_included"] is False
    torch.testing.assert_close(terms["geometry"], torch.zeros(()))
    terms["logp_sum"].backward()
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())


def test_partition_preserves_global_presentation_indices() -> None:
    update = {
        "common_image_ids": list(range(32)),
        "pair_image_ids": list(range(100, 132)),
    }
    common, pairs = ranking._partition_update(update, rank=2)
    assert [index for index, _ in common] == list(range(16, 24))
    assert [index for index, _ in pairs] == list(range(48, 56))


def test_manifest_and_prepared_data_bind_the_frozen_call_budgets() -> None:
    preparation = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-09-16-source256-output-ranking-repair/preparation/R-main.json"
    )
    if not preparation.exists():
        pytest.skip("ranking preparation is not present")
    manifest = ranking.validate_manifest(ranking.read(preparation))
    data = ranking.load_data(manifest)
    assert ranking.reference_entries(data)[0]["role"] == "preferred"
    assert manifest["runtime"]["max_model_calls"] == 768
    assert manifest["runtime"]["max_model_forwards"] == 1536
    assert data["counts"]["canonical"] == 512
    assert data["counts"]["pair"] == 512
