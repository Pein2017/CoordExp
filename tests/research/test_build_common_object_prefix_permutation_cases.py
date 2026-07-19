from __future__ import annotations

import copy

import pytest

from scripts.research.build_common_object_prefix_permutation_cases import (
    CASE_SCHEMA_VERSION,
    build_case_for_rows,
    intersection_over_union,
    permutation_inversion_count,
    render_row_text,
)


def _artifact(*, order: list[int], image_id: str = "2299") -> dict:
    gt = [
        {
            "bbox": [index * 10, 0, index * 10 + 8, 8],
            "description": "person",
            "object_id": f"ann-{index}",
        }
        for index in range(12)
    ]
    return {
        "example_id": f"coco2017_val_{int(image_id):012d}",
        "image_width": 120,
        "image_height": 100,
        "image_path": f"/tmp/{int(image_id):012d}.jpg",
        "gt": gt,
        "pred": [
            {
                "description": "person",
                "bbox": list(gt[index]["bbox"]),
                "generated_order": position,
            }
            for position, index in enumerate(order)
        ],
    }


def test_builds_common_v2_cases_with_shared_suffix_and_exact_gt_rows() -> None:
    sorted_row = _artifact(order=list(range(12)))
    random_row = _artifact(order=[4, 2, 0, 3, 1, 5, 6, 7, 8, 9, 10, 11])
    payload = build_case_for_rows(
        sorted_row,
        random_row,
        image_id="2299",
        prefix_depths=(6, 10),
        shuffle_seed=17,
        rollout_horizon_rows=3,
    )

    assert payload["schema_version"] == CASE_SCHEMA_VERSION
    assert payload["metadata"]["cohort_status"].startswith("artifact-local provisional")
    assert [entity["entity_id"] for entity in payload["entities"]] == [f"gt_{i:04d}" for i in range(12)]
    assert payload["metadata"]["entity_ledger_scope"] == "full_artifact_local_gt"
    assert payload["metadata"]["full_artifact_local_gt_entity_count"] == 12
    assert payload["metadata"]["selected_prefix_is_a_subset_of_full_entity_ledger"] is True
    assert payload["metadata"]["prefix_likelihood_scoring"]["status"] == "unavailable_in_this_pilot"
    assert payload["metadata"]["permutation_distance_metric"]["metric_full_name"].startswith("Kendall tau")
    assert payload["entities"][0]["row_text"] == (
        "<|object_ref_start|>person<|object_ref_end|><|box_start|>"
        "<|coord_0|><|coord_0|><|coord_67|><|coord_80|><|box_end|>"
    )
    for case in payload["cases"]:
        assert len(case["selected_common_entity_ids"]) == int(case["case_id"].rsplit("_", 1)[-1])
        canonical = case["arms"]["current_sorted_rollout_order"]["entity_ids"]
        for name, arm in case["arms"].items():
            assert set(arm["entity_ids"]) == set(canonical)
            if name == "reverse_earlier_rows_with_fixed_final_one":
                assert arm["entity_ids"][-1:] == canonical[-1:]
                assert arm["entity_ids"][-2:] != canonical[-2:]
            else:
                assert arm["entity_ids"][-2:] == canonical[-2:]
            assert arm["permutation_inversion_count_relative_to_canonical_order"] == permutation_inversion_count(
                arm["entity_ids"], canonical
            )
            if name != "current_sorted_rollout_order":
                assert any(
                    comparison["arm_names"] == ["current_sorted_rollout_order", name]
                    and comparison["rollout_horizon_rows"] == 3
                    and comparison["shared_suffix_length"] == (1 if name == "reverse_earlier_rows_with_fixed_final_one" else 2)
                    for comparison in case["comparisons"]
                )
        assert case["entity_ledger_scope"] == "full_artifact_local_gt"
        assert case["entity_ledger_entity_count"] == len(payload["entities"])
        assert case["omitted_comparisons"] == []
        assert case["arms"]["reverse_earlier_rows_with_fixed_final_one"]["entity_ids"] != case["arms"]["reverse_earlier_rows_with_fixed_final_two"]["entity_ids"]
        assert case["arms"]["adjacent_swap_before_fixed_final_two"]["entity_ids"][-4:] == [
            canonical[-3], canonical[-4], canonical[-2], canonical[-1]
        ]


def test_matching_is_global_greedy_with_deterministic_tie_break() -> None:
    row = _artifact(order=list(range(12)))
    # Two predictions compete for one GT.  The exact duplicate has higher IoU
    # and must win, leaving the other prediction available for its own GT.
    row["pred"] = [{"description": "person", "bbox": [0, 0, 8, 8]}]
    row["pred"] += [{"description": "person", "bbox": [0, 0, 7, 7]}]
    row["pred"] += [
        {"description": "person", "bbox": [index * 10, 0, index * 10 + 8, 8]}
        for index in range(1, 6)
    ]
    other = copy.deepcopy(row)
    other["pred"] = [other["pred"][2], other["pred"][0], other["pred"][1], *other["pred"][3:]]
    payload = build_case_for_rows(row, other, image_id="2299", prefix_depths=(6,))
    assert payload["metadata"]["common_matched_gt_indices"] == [0, 1, 2, 3, 4, 5]


def test_rejects_different_artifact_local_gt_arrays() -> None:
    sorted_row = _artifact(order=list(range(12)))
    random_row = _artifact(order=list(range(12)))
    random_row["gt"][0]["bbox"][0] += 1
    with pytest.raises(ValueError, match="GT arrays must be exactly equal"):
        build_case_for_rows(sorted_row, random_row, image_id="2299", prefix_depths=(6,))


def test_skips_sorted_only_entity_when_selecting_common_prefix() -> None:
    sorted_row = _artifact(order=list(range(12)))
    random_row = _artifact(order=[1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    random_row["pred"] = [prediction for prediction in random_row["pred"] if prediction["bbox"][0] != 50]
    payload = build_case_for_rows(sorted_row, random_row, image_id="2299", prefix_depths=(10,))
    selected = payload["cases"][0]["selected_common_gt_indices"]
    assert selected == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    assert 5 not in selected


def test_supports_canonical_width_height_desc_points_artifact_shape() -> None:
    sorted_row = _artifact(order=list(range(12)))
    random_row = _artifact(order=[4, 2, 0, 3, 1, 5, 6, 7, 8, 9, 10, 11])
    for row in (sorted_row, random_row):
        row["width"] = row.pop("image_width")
        row["height"] = row.pop("image_height")
        row["image"] = row.pop("image_path")
        row["gt"] = [
            {"desc": item.pop("description"), "points": item.pop("bbox"), "object_id": item["object_id"]}
            for item in row["gt"]
        ]
        row["pred"] = [
            {"desc": item.pop("description"), "points": item.pop("bbox")}
            for item in row["pred"]
        ]
    payload = build_case_for_rows(sorted_row, random_row, image_id="2299", prefix_depths=(6,))
    assert payload["metadata"]["image_width"] == 120
    assert payload["entities"][0]["description"] == "person"


def test_render_and_iou_helpers_are_deterministic() -> None:
    row_text, bins = render_row_text("person", [12, 10, 60, 50], image_width=120, image_height=100)
    assert bins == [100, 100, 500, 500]
    assert row_text.endswith("<|coord_500|><|box_end|>")
    assert intersection_over_union([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_permutation_inversion_count_is_deterministic_and_relative_to_canonical() -> None:
    canonical = ["a", "b", "c", "d"]
    assert permutation_inversion_count(canonical, canonical) == 0
    assert permutation_inversion_count(["b", "a", "c", "d"], canonical) == 1
    assert permutation_inversion_count(["d", "c", "b", "a"], canonical) == 6
    assert permutation_inversion_count(["c", "a", "d", "b"], canonical) == 3


def test_full_ledger_retains_entities_not_in_selected_common_prefix() -> None:
    sorted_row = _artifact(order=list(range(12)))
    secondary = _artifact(order=[1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    payload = build_case_for_rows(sorted_row, secondary, image_id="2299", prefix_depths=(6,))
    ledger_ids = {entity["entity_id"] for entity in payload["entities"]}
    selected_ids = set(payload["cases"][0]["selected_common_entity_ids"])
    assert len(ledger_ids) == 12
    assert len(selected_ids) == 6
    assert selected_ids < ledger_ids
    assert payload["metadata"]["full_artifact_local_gt_entity_count"] == 12


def test_new_permutation_arms_have_fixed_one_and_adjacent_swap_semantics() -> None:
    sorted_row = _artifact(order=list(range(12)))
    secondary = _artifact(order=[4, 2, 0, 3, 1, 5, 6, 7, 8, 9, 10, 11])
    payload = build_case_for_rows(sorted_row, secondary, image_id="2299", prefix_depths=(6,))
    case = payload["cases"][0]
    canonical = case["arms"]["current_sorted_rollout_order"]["entity_ids"]
    fixed_one = case["arms"]["reverse_earlier_rows_with_fixed_final_one"]["entity_ids"]
    adjacent = case["arms"]["adjacent_swap_before_fixed_final_two"]["entity_ids"]
    assert fixed_one[:-1] == list(reversed(canonical[:-1]))
    assert fixed_one[-1:] == canonical[-1:]
    assert fixed_one != case["arms"]["reverse_earlier_rows_with_fixed_final_two"]["entity_ids"]
    assert adjacent[:-4] == canonical[:-4]
    assert adjacent[-4:] == [canonical[-3], canonical[-4], canonical[-2], canonical[-1]]
    assert len({tuple(arm["entity_ids"]) for arm in case["arms"].values()}) == len(case["arms"])


def test_identical_predefined_permutation_is_omitted_deterministically() -> None:
    sorted_row = _artifact(order=list(range(12)))
    # For the six-row prefix, this historical relative order is exactly the
    # prescribed adjacent swap immediately before the fixed final two rows.
    secondary = _artifact(order=[0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11])
    payload = build_case_for_rows(sorted_row, secondary, image_id="2299", prefix_depths=(6,))
    case = payload["cases"][0]
    assert "adjacent_swap_before_fixed_final_two" not in case["arms"]
    assert not any(
        comparison["arm_names"] == ["current_sorted_rollout_order", "adjacent_swap_before_fixed_final_two"]
        for comparison in case["comparisons"]
    )
    assert case["omitted_comparisons"] == [{
        "arm_name": "adjacent_swap_before_fixed_final_two",
        "duplicate_of": "historical_random_relative_order_with_fixed_final_two",
        "reason": "identical_row_order",
    }]
