import math

import pytest
import torch

from scripts.research import run_fixed_prefix_nonboundary_box_grammar_image19432 as probe


def _logits(center: int, spread: float = 8.0) -> torch.Tensor:
    bins = torch.arange(1000, dtype=torch.float32)
    return -((bins - float(center)) ** 2) / (2.0 * spread**2)


def _metrics(widths=(100, 100, 100, 100)):
    return {
        name: probe.x2_distribution_metrics(_logits(x1 + width), x1=x1)
        for (name, x1, _), width in zip(probe.CUES, widths)
    }


def test_frozen_constants_are_explicit_and_uncensored():
    assert [item[1] for item in probe.CUES] == [537, 608, 643, 700]
    assert probe.FIXED_Y1 == 123
    assert 999 - probe.CUES[-1][1] == 299
    assert "physical_owner" in probe.PROHIBITED_CLAIMS
    assert "background" in probe.PROHIBITED_CLAIMS


def test_fixture_identity_and_synthetic_edge_distance():
    receipt = probe.load_and_validate_fixture()
    assert receipt["sha256"] == probe.FIXTURE_SHA256
    assert receipt["target"]["row"]["pre_x1_prompt_token_ids_sha256"] == probe.EXPECTED_PROMPT_SHA256
    edges = receipt["chair_left_edges"]
    assert min(abs(608 - edge) for edge in edges) >= 35
    assert min(abs(700 - edge) for edge in edges) >= 43


def test_support_uses_best_real_not_worst_real():
    scores = {
        "real_target_left_edge": 0.0,
        "synthetic_inter_edge_nonboundary": -math.log(10.0),
        "real_adjacent_left_edge": -0.5,
        "synthetic_right_dense_field_nonboundary": -math.log(10.0) - 0.01,
    }
    decision = probe.support_admission(scores)
    assert decision["arms"]["synthetic_inter_edge_nonboundary"]["passed"]
    assert not decision["arms"]["synthetic_right_dense_field_nonboundary"]["passed"]
    assert not decision["passed"]


def test_width_pmf_is_normalized_and_zero_padded():
    result = probe.x2_distribution_metrics(_logits(800), x1=700)
    assert result["valid_right_mass"] > 0.99
    assert result["width_pmf_sum"] == pytest.approx(1.0, abs=1e-6)
    assert len(result["width_pmf_float32"]) == 999
    assert all(value == 0.0 for value in result["width_pmf_float32"][299:])
    assert result["expected_right_given_valid"] == pytest.approx(800.0, abs=0.1)
    assert result["expected_width_given_valid"] == pytest.approx(100.0, abs=0.1)


def test_jensen_shannon_and_panel_statistics_detect_translation():
    metrics = _metrics()
    stats = probe.panel_statistics(metrics)
    assert stats["real_adjacent_minus_target_expected_right"] == pytest.approx(106.0, abs=0.2)
    assert stats["kendall_tau_x1_expected_right"] == pytest.approx(1.0)
    assert stats["ordinary_least_squares_slope"] == pytest.approx(1.0, abs=0.01)
    assert stats["expected_width_range"] < 0.2
    assert stats["median_translated_width_jensen_shannon"] < stats["median_absolute_x2_jensen_shannon"]


def test_classification_supports_only_bounded_grammar_handle():
    metrics = _metrics()
    stats = probe.panel_statistics(metrics)
    support = probe.support_admission({name: -1.0 for name, _, _ in probe.CUES})
    decision = probe.classify_panel(execution_trust_passed=True, support=support, metrics=metrics, statistics_payload=stats)
    assert decision["classification"] == "simple_translation_grammar_compatible"
    assert decision["admitted_positive_mechanism_handle"]
    assert "physical_owner" in decision["prohibited_claims"]


def test_classification_stops_on_support_before_outcome():
    metrics = _metrics()
    stats = probe.panel_statistics(metrics)
    scores = {name: -1.0 for name, _, _ in probe.CUES}
    scores["synthetic_right_dense_field_nonboundary"] = -10.0
    support = probe.support_admission(scores)
    decision = probe.classify_panel(execution_trust_passed=True, support=support, metrics=metrics, statistics_payload=stats)
    assert decision["classification"] == "unidentified_control_support_failure"
    assert not decision["admitted_positive_mechanism_handle"]


def test_classification_rejects_simple_grammar_when_widths_drift():
    metrics = _metrics((80, 120, 180, 250))
    stats = probe.panel_statistics(metrics)
    support = probe.support_admission({name: -1.0 for name, _, _ in probe.CUES})
    decision = probe.classify_panel(execution_trust_passed=True, support=support, metrics=metrics, statistics_payload=stats)
    assert decision["classification"] == "simple_translation_grammar_insufficient"


def test_greedy_parser_requires_two_coordinates_and_box_end():
    valid = probe._parse_greedy_suffix([
        probe.complete.coordinate_token_id(700),
        probe.complete.coordinate_token_id(350),
        probe.BOX_END_TOKEN_ID,
    ])
    assert valid["parser_valid"]
    assert valid["released_x2_y2"] == [700, 350]
    assert not probe._parse_greedy_suffix([probe.complete.coordinate_token_id(700)])["parser_valid"]
