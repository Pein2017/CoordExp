from __future__ import annotations

import json

import pytest
import torch

from src.analysis.post_x1_instance_basin_tomography.posterior import (
    axis_len_for_slot,
    classify_slot_posterior,
    strict_r95_radius,
)


def _logits(
    *,
    vocab_size: int = 1300,
    fill: float = -10.0,
    values: dict[int, float] | None = None,
) -> torch.Tensor:
    logits = torch.full((vocab_size,), fill)
    for token_id, value in (values or {}).items():
        logits[int(token_id)] = float(value)
    return logits


def test_strict_r95_uses_axis_fraction_floor_and_cap() -> None:
    assert strict_r95_radius(axis_len=20, fraction=0.04, cap=8) == 0
    assert strict_r95_radius(axis_len=50, fraction=0.04, cap=8) == 2
    assert strict_r95_radius(axis_len=200, fraction=0.04, cap=8) == 8
    assert strict_r95_radius(axis_len=400, fraction=0.04, cap=8) == 8


def test_axis_len_for_slot_uses_matching_bbox_axis() -> None:
    bbox = [10, 20, 210, 120]

    assert axis_len_for_slot("x1", bbox) == 200
    assert axis_len_for_slot("x2", bbox) == 200
    assert axis_len_for_slot("y1", bbox) == 100
    assert axis_len_for_slot("y2", bbox) == 100


def test_classify_slot_posterior_uses_full_vocab_coord_mass_not_conditional_coord_softmax() -> None:
    coord_token_ids = list(range(100, 1100))
    full_vocab_logits = _logits(values={42: 8.0, 200: 5.0, 400: 3.0})
    expected_mass = float(torch.softmax(full_vocab_logits, dim=-1)[coord_token_ids].sum().item())

    row = classify_slot_posterior(
        full_vocab_logits=full_vocab_logits,
        coord_token_ids=coord_token_ids,
        slot="y1",
        target_value=100,
        target_axis_len=80,
        competitors=[{"gt_idx": 2, "desc": "person", "value": 300, "axis_len": 90}],
        low_margin_threshold=0.05,
        coord_mass_low_threshold=0.2,
    )

    assert row["coord_vocab_mass"] == pytest.approx(expected_mass)
    assert row["coord_mass_low_flag"] is True
    assert row["slot_taxonomy"] == "invalid_low_coord_mass"
    assert row["winner_bucket"] == "invalid_low_coord_mass"
    assert row["noncoord_top_token_id"] == 42
    assert row["noncoord_top_prob"] > row["top_coord_candidates"][0]["full_vocab_prob"]
    json.dumps(row, allow_nan=False)


def test_classify_slot_posterior_prefers_target_over_same_desc_competitor() -> None:
    coord_token_ids = list(range(100, 1100))
    full_vocab_logits = _logits(values={42: 1.0, 200: 5.0, 400: 3.0})

    row = classify_slot_posterior(
        full_vocab_logits=full_vocab_logits,
        coord_token_ids=coord_token_ids,
        slot="y1",
        target_value=100,
        target_axis_len=80,
        competitors=[{"gt_idx": 2, "desc": "person", "value": 300, "axis_len": 90}],
        low_margin_threshold=0.05,
    )

    assert row["slot_taxonomy"] == "target"
    assert row["winner_bucket"] == "target"
    assert row["winner_instance_id"] == "target"
    assert row["target_r95_hit"] is True
    assert row["best_competitor_r95_hit"] is False
    assert row["boundary_extreme_flag"] is False
    assert row["target_center_logit"] == pytest.approx(5.0)
    assert row["best_competitor_center_logit"] == pytest.approx(3.0)
    assert row["top_coord_candidates"][0]["coord_value"] == 100
    assert row["top_coord_candidates"][0]["token_id"] == 200
    assert row["top_coord_candidates"][0]["logit"] == pytest.approx(5.0)
    json.dumps(row, allow_nan=False)


def test_classify_slot_posterior_can_label_same_desc_competitor() -> None:
    coord_token_ids = list(range(100, 1100))
    full_vocab_logits = _logits(values={200: 3.0, 400: 5.0})

    row = classify_slot_posterior(
        full_vocab_logits=full_vocab_logits,
        coord_token_ids=coord_token_ids,
        slot="x2",
        target_value=100,
        target_axis_len=80,
        competitors=[{"gt_idx": 2, "desc": "person", "value": 300, "axis_len": 90}],
        low_margin_threshold=0.05,
    )

    assert row["slot_taxonomy"] == "competitor_same_desc"
    assert row["winner_bucket"] == "competitor_same_desc"
    assert row["winner_instance_id"] == 2


def test_classify_slot_posterior_can_label_other_desc_object() -> None:
    coord_token_ids = list(range(100, 1100))
    full_vocab_logits = _logits(values={200: 2.0, 700: 5.0})

    row = classify_slot_posterior(
        full_vocab_logits=full_vocab_logits,
        coord_token_ids=coord_token_ids,
        slot="x1",
        target_value=100,
        target_axis_len=80,
        competitors=[],
        other_desc_objects=[{"gt_idx": 9, "desc": "chair", "value": 600, "axis_len": 80}],
        low_margin_threshold=0.05,
    )

    assert row["slot_taxonomy"] == "other_desc_object"
    assert row["winner_bucket"] == "other_desc_object"
    assert row["winner_instance_id"] == 9


def test_classify_slot_posterior_marks_background_when_peak_matches_no_object_basin() -> None:
    coord_token_ids = list(range(100, 1100))
    full_vocab_logits = _logits(values={200: 2.0, 555: 5.0})

    row = classify_slot_posterior(
        full_vocab_logits=full_vocab_logits,
        coord_token_ids=coord_token_ids,
        slot="y2",
        target_value=100,
        target_axis_len=80,
        competitors=[],
        other_desc_objects=[],
        low_margin_threshold=0.05,
    )

    assert row["slot_taxonomy"] == "background"
    assert row["winner_bucket"] == "background"
    assert row["background_or_outlier_flag"] is True


def test_boundary_extreme_is_orthogonal_to_identity_when_target_is_on_boundary() -> None:
    coord_token_ids = list(range(100, 1100))
    full_vocab_logits = _logits(values={1099: 5.0, 500: 3.0})

    row = classify_slot_posterior(
        full_vocab_logits=full_vocab_logits,
        coord_token_ids=coord_token_ids,
        slot="x2",
        target_value=999,
        target_axis_len=120,
        competitors=[],
        low_margin_threshold=0.05,
    )

    assert row["slot_taxonomy"] == "target"
    assert row["winner_bucket"] == "target"
    assert row["winner_instance_id"] == "target"
    assert row["boundary_extreme_flag"] is True


def test_boundary_extreme_is_a_taxonomy_when_it_matches_no_instance() -> None:
    coord_token_ids = list(range(100, 1100))
    full_vocab_logits = _logits(values={1099: 5.0, 750: 3.0})

    row = classify_slot_posterior(
        full_vocab_logits=full_vocab_logits,
        coord_token_ids=coord_token_ids,
        slot="x2",
        target_value=650,
        target_axis_len=100,
        competitors=[],
        low_margin_threshold=0.05,
    )

    assert row["slot_taxonomy"] == "boundary_extreme"
    assert row["winner_bucket"] == "boundary_extreme"
    assert row["boundary_extreme_flag"] is True


def test_classify_slot_posterior_marks_tied_state_before_forcing_identity() -> None:
    coord_token_ids = list(range(100, 1100))
    full_vocab_logits = _logits(values={200: 5.0, 400: 4.99})

    row = classify_slot_posterior(
        full_vocab_logits=full_vocab_logits,
        coord_token_ids=coord_token_ids,
        slot="x1",
        target_value=100,
        target_axis_len=100,
        competitors=[{"gt_idx": 2, "desc": "person", "value": 300, "axis_len": 100}],
        low_margin_threshold=0.05,
    )

    assert row["low_margin_flag"] is True
    assert row["slot_taxonomy"] == "tied"
    assert row["winner_bucket"] == "tied"
