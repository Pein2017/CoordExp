from src.analysis.raw_text_coordinate_behavior import (
    lexical_control_features,
    summarize_confirmatory_records,
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


def test_summarize_confirmatory_records_separates_serializer_surfaces_and_vision_lift() -> None:
    summary = summarize_confirmatory_records(
        records=[
            {
                "model_alias": "base_only",
                "serializer_surface": "model_native",
                "slot": "x1",
                "candidate_value": 100,
                "gt_value": 100,
                "distance_to_gt": 0,
                "image_condition": "correct",
                "logprob_sum": -1.0,
            },
            {
                "model_alias": "base_only",
                "serializer_surface": "model_native",
                "slot": "x1",
                "candidate_value": 100,
                "gt_value": 100,
                "distance_to_gt": 0,
                "image_condition": "swapped",
                "logprob_sum": -3.5,
            },
            {
                "model_alias": "base_only",
                "serializer_surface": "pretty_inline",
                "slot": "x1",
                "candidate_value": 104,
                "gt_value": 100,
                "distance_to_gt": 4,
                "image_condition": "correct",
                "logprob_sum": -2.0,
            },
        ]
    )

    model_native = next(
        row for row in summary if row["serializer_surface"] == "model_native"
    )
    pretty_inline = next(
        row for row in summary if row["serializer_surface"] == "pretty_inline"
    )

    assert model_native["mass_at_4"] == 1.0
    assert model_native["vision_lift"] == 2.5
    assert pretty_inline["mass_at_4"] == 1.0
