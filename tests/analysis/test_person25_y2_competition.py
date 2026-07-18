from __future__ import annotations

import math

from scripts.research.score_person25_y2_competition import aggregate_coordinate_competition


def test_coordinate_competition_reports_band_mass_and_target_probability():
    logits = [0.0] * 12
    coordinate_token_ids = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 999: 5}
    logits[1] = 4.0
    logits[5] = 2.0
    result = aggregate_coordinate_competition(
        logits,
        coordinate_token_ids,
        target_coordinate=999,
        broad_band=(1, 2),
        observed_band=(1, 1),
        top_k=2,
    )
    assert result["coordinate_argmax"]["raw"]["coordinate"] == 1
    assert result["target"]["token_id"] == 5
    assert result["target"]["raw_probability"] < result["top_coordinate_tokens_by_raw_logit"][0]["raw_probability"]
    assert result["bands"]["1_2"]["raw"]["coordinate_low_inclusive"] == 1
    assert result["bands"]["1_2"]["raw"]["probability_mass_over_full_vocabulary"] > 0.0
    assert result["bands"]["1_1"]["raw"]["probability_mass_over_full_vocabulary"] > 0.0


def test_temperature_sharpens_the_coordinate_argmax_and_margin():
    logits = [0.0] * 4
    coordinate_token_ids = {0: 0, 1: 1, 999: 2, 2: 3}
    logits[0] = 3.0
    logits[1] = 2.8
    logits[2] = 2.0
    result = aggregate_coordinate_competition(
        logits,
        coordinate_token_ids,
        broad_band=(0, 1),
        observed_band=(0, 0),
    )
    assert result["coordinate_argmax"]["temperature_0_4"]["coordinate"] == 0
    assert result["bands"]["0_1"]["temperature_0_4"]["probability_mass_given_coordinate_token"] > 0.9
    assert result["bands"]["0_1"]["temperature_0_4"]["probability_mass_over_full_vocabulary"] > result["bands"]["0_1"]["raw"]["probability_mass_over_full_vocabulary"]
    assert math.isfinite(result["bands"]["0_1"]["raw_logsumexp_margin_band_vs_coord999"])


def test_invalid_coordinate_token_map_is_rejected():
    try:
        aggregate_coordinate_competition([0.0, 1.0], {0: 0, 999: 4})
    except ValueError as exc:
        assert "outside" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected coordinate token range validation")
