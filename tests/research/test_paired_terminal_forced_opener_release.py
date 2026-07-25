from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "summarize_paired_terminal_forced_opener_release.py"
)
_SPEC = importlib.util.spec_from_file_location("paired_terminal_release", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)


def _release(*, matched: list[str], valid: bool = True, unmatched: int = 0) -> dict:
    return {
        "strict_matched_owner_ids": matched,
        "parse_evidence": {
            "metric_bearing": valid,
            "valid_prediction_count": int(valid),
        },
        "unmatched_or_ambiguous_prediction_indices": list(range(unmatched)),
        "row_stop": {"stop_reason": "complete_row"},
        "raw_generated_token_ids": [151646, 1, 151649] if valid else [151645],
        "entity_matches": [
            {
                "matched_entity_id": owner_id,
                "description": "person",
                "predicted_bbox": [0, 0, 10, 10],
                "predicted_bbox_norm1000": [0, 0, 10, 10],
                "candidates": [
                    {
                        "entity_id": owner_id,
                        "iou": 0.8,
                        "center_distance_norm": 0.01,
                    }
                ],
            }
            for owner_id in matched
        ],
    }


def test_release_classification_separates_remaining_and_covered() -> None:
    value = summary._release_owner_sets(
        _release(matched=["20", "10"]), remaining={"20"}, covered={"10"}
    )

    assert value["verified_uncovered_owner_ids"] == ["20"]
    assert value["covered_repeat_owner_ids"] == ["10"]
    assert value["outcome"] == "verified_uncovered_owner"


def test_case_primary_is_forced_owner_not_found_by_native() -> None:
    raw = {
        "boundary_id": "b1",
        "image_id": "1",
        "prefix_depth": 2,
        "object_count_band": "medium_4_to_7",
        "annotation_object_count": 5,
        "remaining_owner_ids": ["1:20", "1:30"],
        "covered_owner_ids": ["1:10"],
        "releases": {
            "native": _release(matched=["20"]),
            "forced_opener": _release(matched=["30"]),
        },
    }
    statistics = {
        "checkpoint_margins": {
            "source": -1.0,
            "transition-step36": -0.2,
            "pairwise-lr3e6-step90": 0.5,
            "owner-conditioned-lr1e5-step90": 2.0,
        }
    }

    value = summary._classify_case(raw, statistics=statistics)

    assert value["causal_gained_owner_ids"] == ["30"]
    assert value["causal_lost_owner_ids"] == ["20"]
    assert value["causal_success"] is True
    assert value["owner_net_change"] == 0


def test_forced_same_owner_as_native_is_not_causal_success() -> None:
    raw = {
        "boundary_id": "b2",
        "image_id": "2",
        "prefix_depth": 1,
        "object_count_band": "sparse_1_to_3",
        "annotation_object_count": 2,
        "remaining_owner_ids": ["2:20"],
        "covered_owner_ids": ["2:10"],
        "releases": {
            "native": _release(matched=["20"]),
            "forced_opener": _release(matched=["20"]),
        },
    }
    statistics = {
        "checkpoint_margins": {
            "source": 0.1,
            "transition-step36": 0.5,
            "pairwise-lr3e6-step90": 1.0,
            "owner-conditioned-lr1e5-step90": 2.0,
        }
    }

    value = summary._classify_case(raw, statistics=statistics)

    assert value["causal_gained_owner_ids"] == []
    assert value["retained_uncovered_owner_ids"] == ["20"]
    assert value["causal_success"] is False


def test_aggregate_keeps_success_and_owner_net_separate() -> None:
    cases = [
        {
            "causal_success": True,
            "causal_gained_owner_ids": ["30"],
            "causal_lost_owner_ids": ["20"],
            "retained_uncovered_owner_ids": [],
            "owner_net_change": 0,
            "raw_row_token_ids_equal": False,
            "causal_gain_match_evidence": [
                {"iou": 0.8, "center_distance_norm": 0.01}
            ],
            "native": {"verified_uncovered_owner_ids": ["20"], "outcome": "verified_uncovered_owner"},
            "forced_opener": {"verified_uncovered_owner_ids": ["30"], "outcome": "verified_uncovered_owner"},
        },
        {
            "causal_success": False,
            "causal_gained_owner_ids": [],
            "causal_lost_owner_ids": [],
            "retained_uncovered_owner_ids": [],
            "owner_net_change": 0,
            "raw_row_token_ids_equal": True,
            "causal_gain_match_evidence": [],
            "native": {"verified_uncovered_owner_ids": [], "outcome": "immediate_terminal"},
            "forced_opener": {"verified_uncovered_owner_ids": [], "outcome": "covered_owner_repeat"},
        },
    ]

    value = summary._aggregate(cases)

    assert value["primary_causal_success"]["success_count"] == 1
    assert value["causal_gained_owner_count"] == 1
    assert value["causal_lost_owner_count"] == 1
    assert value["net_owner_change"] == 0
    assert value["causal_gain_match_quality"]["iou"]["mean"] == 0.8


def test_nested_checkpoint_band_uses_first_positive_checkpoint() -> None:
    assert (
        summary._diagnostic_positive_band(
            {
                "source": -1.0,
                "transition-step36": -0.1,
                "pairwise-lr3e6-step90": 0.2,
                "owner-conditioned-lr1e5-step90": 2.0,
            }
        )
        == "pairwise_positive_beyond_transition"
    )
