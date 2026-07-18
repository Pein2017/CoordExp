from scripts.research.analyze_image2299_near_complete_relabel_successor_transition import (
    coord_bins_to_pixel_box,
    intersection_over_union,
    match_prediction,
    summarize_records,
)


def test_coordinate_bins_use_the_coordexp_one_thousand_scale() -> None:
    assert coord_bins_to_pixel_box([615, 86, 724, 388], width=1216, height=736) == [
        748,
        63,
        880,
        286,
    ]


def test_match_requires_both_intersection_over_union_and_margin() -> None:
    objects = [
        {
            "description": "person",
            "category_rank": 2,
            "annotation_id": -2,
            "pixel_box": [100, 100, 200, 300],
        },
        {
            "description": "person",
            "category_rank": 3,
            "annotation_id": -23,
            "pixel_box": [300, 100, 400, 300],
        },
    ]
    matched = match_prediction(
        description="person", pixel_box=[101, 101, 199, 299], relabel_objects=objects
    )
    assert matched["matched"] is True
    assert matched["matched_category_rank"] == 2

    ambiguous_objects = [
        {**objects[0], "pixel_box": [100, 100, 200, 300]},
        {**objects[1], "pixel_box": [102, 100, 202, 300]},
    ]
    ambiguous = match_prediction(
        description="person",
        pixel_box=[101, 100, 201, 300],
        relabel_objects=ambiguous_objects,
    )
    assert ambiguous["matched"] is False
    assert ambiguous["reason"] == "top_match_margin_below_threshold"
    assert intersection_over_union([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def _record(owner: str, rank: int, successor: int, seed: int, variant: str) -> dict:
    return {
        "owner_id": owner,
        "emitted_person_rank": rank,
        "variant_label": variant,
        "sampling_seed": seed,
        "outcome": "matched_person",
        "matched_category_rank": successor,
        "immediate_self_repeat": successor == rank,
        "earlier_parent_repeat": successor in {0, 1},
        "backward_uncovered_person": successor < rank and successor not in {0, 1},
        "forward_uncovered_person": successor > rank,
    }


def test_summary_tracks_owner_specific_transition_counts() -> None:
    records = []
    owners = {"0003": 3, "0004": 2, "0006": 4, "0012": 14}
    successors = {"0003": 2, "0004": 6, "0006": 3, "0012": 10}
    for variant in ("variant-01", "variant-02", "variant-03"):
        for owner, rank in owners.items():
            records.append(_record(owner, rank, successors[owner], 7, variant))
    summary = summarize_records(records)
    assert summary["totals"]["call_count"] == 12
    assert summary["totals"]["immediate_self_repeat_count"] == 0
    assert summary["by_owner"]["0006"]["person_successor_rank_counts"] == {"3": 3}
    assert summary["paired_exact_variant_agreement"]["0004"]["agreeing_seed_count"] == 1
    assert summary["canonical_owner_differentiation"]["distinct_successor_count_histogram"] == {
        "4": 1
    }
