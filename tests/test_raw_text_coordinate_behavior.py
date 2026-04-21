from src.analysis.raw_text_coordinate_behavior import (
    lexical_control_features,
    summarize_choice_margin,
)


def test_lexical_control_features_uses_real_token_edit_distance() -> None:
    features = lexical_control_features(
        candidate_value=820,
        center_value=819,
        gt_value=819,
        candidate_tokens=("8", "2", "0"),
        center_tokens=("8", "1", "9"),
    )

    assert features["numeric_distance_to_center"] == 1
    assert features["token_edit_distance"] == 2
    assert features["shared_prefix_length"] == 1


def test_summarize_choice_margin_prefers_higher_score() -> None:
    summary = summarize_choice_margin(
        choice_scores={
            "eos": {"logprob_sum": -4.0},
            "next_object": {"logprob_sum": -1.5},
        }
    )

    assert summary["winner"] == "next_object"
    assert summary["margin"] == 2.5
