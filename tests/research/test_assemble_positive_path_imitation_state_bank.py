from __future__ import annotations

from scripts.research.assemble_positive_path_imitation_state_bank import (
    deterministic_batch_fit_exclusion,
    exact_row_site_types,
    exact_row_slices,
    geometry_eligibility_receipt,
    image_balanced_weights,
    row_gradient_eligibility,
    select_best_route,
)


def _row_receipt(
    index: int,
    owner: str | None,
    *,
    status: str = "verified_owner",
    category: str = "person",
    iou: float = 0.9,
) -> dict[str, object]:
    return {
        "generated_row_index": index,
        "owner_id": owner,
        "entity_status": status,
        "category": category,
        "intersection_over_iou": iou,
        "intersection_over_union": iou,
        "bbox": [100, 100, 200, 200],
        "owner_bbox": [100, 100, 200, 200],
        "raw": {
            "coord_bins": [100, 100, 200, 200],
            "raw_span_text": "exact",
        },
    }


def _route_image() -> dict[str, object]:
    def assignment(rows: list[dict[str, object]], *, duplicate: int = 0, malformed: int = 0, unresolved: int = 0) -> dict[str, object]:
        return {
            "row_assignment_receipts": rows,
            "row_counts": {
                "duplicate": duplicate,
                "malformed": malformed,
                "unresolved": unresolved,
            },
        }

    return {
        "image_id": "42",
        "greedy_trajectory_id": "greedy",
        "sampled_trajectory_ids": ["seed-9", "seed-3", "seed-1"],
        "trajectory_evidence": {
            "greedy": {"seed": 0, "stop_reason": "im_end", "parser": {"parse_status": "accepted"}},
            "seed-9": {"seed": 9, "stop_reason": "im_end", "parser": {"parse_status": "accepted"}},
            "seed-3": {"seed": 3, "stop_reason": "im_end", "parser": {"parse_status": "accepted"}},
            "seed-1": {"seed": 1, "stop_reason": "im_end", "parser": {"parse_status": "malformed"}},
        },
        "budgets": [
            {
                "budget": 16,
                "owner_sets": {
                    "greedy": ["42:a"],
                    "seed-9": ["42:a", "42:b", "42:c"],
                    "seed-3": ["42:a", "42:b", "42:c"],
                    "seed-1": ["42:a", "42:b", "42:c", "42:d"],
                },
                "trajectory_assignments": {
                    "greedy": assignment([_row_receipt(0, "42:a")], duplicate=1),
                    # Same owner gain, but a worse duplicate delta.  It must
                    # lose before unresolved/seed tie-breaks are considered.
                    "seed-9": assignment([_row_receipt(0, "42:a"), _row_receipt(1, "42:b"), _row_receipt(2, "42:c")], duplicate=1, unresolved=0),
                    "seed-3": assignment([_row_receipt(0, "42:a"), _row_receipt(1, "42:b"), _row_receipt(2, "42:c")], duplicate=0, unresolved=4),
                    "seed-1": assignment([_row_receipt(0, "42:a"), _row_receipt(1, "42:b")], malformed=1),
                },
            }
        ],
    }


def test_route_selection_requires_strict_gain_and_uses_declared_lexicographic_order() -> None:
    receipt = select_best_route(_route_image())
    assert receipt["selected_route_id"] == "seed-3"
    assert receipt["selected_seed"] == 3
    assert receipt["selected_route"]["added_owner_ids"] == ["42:b", "42:c"]
    assert receipt["candidate_route_count"] == 2
    assert receipt["candidate_routes"]["seed-1"]["admissible"] is False


def test_geometry_rule_retains_unique_owner_schema_but_masks_geometry() -> None:
    owners = [
        {"owner_id": "42:a", "category": "person"},
        {"owner_id": "42:b", "category": "person"},
        {"owner_id": "42:c", "category": "chair"},
    ]
    unique_low = _row_receipt(0, "42:c", category="chair", iou=0.2)
    decision = geometry_eligibility_receipt(unique_low, owners)
    assert decision["gradient_eligible"] is True
    assert decision["geometry_trusted"] is False
    assert row_gradient_eligibility(unique_low, owners) is True

    multi_low = _row_receipt(1, "42:a", category="person", iou=0.74)
    assert row_gradient_eligibility(multi_low, owners) is False
    assert geometry_eligibility_receipt(multi_low, owners)["reason"] == "multi_instance_iou_below_0.75"

    multi_trusted = _row_receipt(2, "42:b", category="person", iou=0.75)
    decision = geometry_eligibility_receipt(multi_trusted, owners)
    assert decision["gradient_eligible"] is True
    assert decision["geometry_trusted"] is True


def test_exact_integer_row_slicing_and_site_typing_never_retokenizes() -> None:
    first = [151646, 987654, 151647, 151648, 151670, 151671, 151672, 151673, 151649]
    second = [151646, 123, 456, 151647, 151648, 151700, 151800, 151900, 152000, 151649]
    prefix, candidate = exact_row_slices(first + second, 1)
    assert prefix == first
    assert candidate == second
    assert [site["intended_token_type"] for site in exact_row_site_types(candidate)] == [
        "schema",
        "desc_text",
        "desc_text",
        "schema",
        "schema",
        "coordinate",
        "coordinate",
        "coordinate",
        "coordinate",
        "schema",
    ]


def test_image_balanced_weights_have_equal_image_totals_and_mean_one() -> None:
    weights = image_balanced_weights({"a": 2, "b": 1, "c": 3})
    assert weights == {"a": 1.0, "b": 2.0, "c": 2 / 3}
    assert sum(weights[image] * count for image, count in {"a": 2, "b": 1, "c": 3}.items()) == 6.0
    assert sum(weights[image] * count for image, count in {"a": 2, "b": 1, "c": 3}.items()) / 6.0 == 1.0


def test_batch_fit_excludes_fewest_images_then_loses_fewest_added_owners() -> None:
    result = deterministic_batch_fit_exclusion(
        [
            {"image_id": "16531", "event_count": 6, "added_owner_count": 2, "unresolved_rows": 3},
            {"image_id": "35514", "event_count": 6, "added_owner_count": 2, "unresolved_rows": 6},
            {"image_id": "424960", "event_count": 6, "added_owner_count": 6, "unresolved_rows": 7},
            {"image_id": "999", "event_count": 4, "added_owner_count": 1, "unresolved_rows": 0},
        ],
        16,
    )
    assert result["excluded_image_ids"] == ["35514"]
    assert result["total_event_count_before"] == 22
    assert result["total_event_count_after"] == 16


def test_old_duplicate_ambiguous_unresolved_rows_are_context_only() -> None:
    owners = [{"owner_id": "42:a", "category": "person"}]
    for status in ("duplicate", "ambiguous_matched_review", "unresolved_pending_crop_review", ""):
        row = _row_receipt(0, "42:a", status=status)
        assert row_gradient_eligibility(row, owners) is False
