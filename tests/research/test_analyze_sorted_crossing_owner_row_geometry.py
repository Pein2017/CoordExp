"""Focused tests for the crossing owner-row geometric relation analyzer.

The analyzer is fail-closed, so most of what matters here is what it *refuses*:
a crossed arm/geometry wiring, a drifted input digest, an invalid box, a
coordinate-token identity that is not the sealed one, a broken owner join, a
pooled cross-image raw delta, or a non-identical republish.

Two fixture families are used.  A miniature sealed input tree exercises the
joins and the refusals with a narrowed :class:`DenominatorContract`; the real
frozen crossing artifacts, when present, exercise the frozen 26 / 26 / 12
denominators and byte determinism end to end.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import analyze_sorted_crossing_owner_row_geometry as geometry  # noqa: E402

START = geometry.COORD_TOKEN_START

REAL_CROSSING_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-03-sorted-crossing-boundary-owner-release-realization"
)
REAL_PLAN_DIR = REAL_CROSSING_ROOT / "20260804T013856Z" / "plan"
REAL_PRIMARY_ROWS = (
    REAL_CROSSING_ROOT / "20260804T020853Z" / "primary-analysis-fragility-v2" / "owner-rows.jsonl"
)
REAL_SECONDARY_ROWS = (
    REAL_CROSSING_ROOT
    / "20260804T020853Z"
    / "secondary-analysis-v3"
    / "secondary-owner-rows.jsonl"
)
REAL_MERGED_DIR = REAL_CROSSING_ROOT / "20260804T020853Z" / "secondary-merged-v2"

real_artifacts = pytest.mark.skipif(
    not (
        REAL_PLAN_DIR.is_dir()
        and REAL_PRIMARY_ROWS.is_file()
        and REAL_SECONDARY_ROWS.is_file()
        and REAL_MERGED_DIR.is_dir()
    ),
    reason="the frozen crossing-boundary artifacts are unavailable on this host",
)


# ---------------------------------------------------------------------------
# Norm-1000 geometry
# ---------------------------------------------------------------------------


def tokens(x1: int, y1: int, x2: int, y2: int) -> list[int]:
    return [START + x1, START + y1, START + x2, START + y2]


def test_decode_box_binds_the_sealed_coordinate_token_origin() -> None:
    box = geometry.decode_box(tokens(10, 20, 30, 40), label="box")
    assert box.as_list() == [10, 20, 30, 40]
    # The origin is the sealed constant, never inferred from the observed minimum.
    assert geometry.COORD_TOKEN_START == 151670
    assert geometry.COORD_BIN_COUNT == 1000


@pytest.mark.parametrize(
    "bad_tokens",
    [
        [START - 1, START + 1, START + 2, START + 3],
        [START, START, START + geometry.COORD_BIN_COUNT, START + 3],
        [START, START + 1, START + 2],
        [START, START + 1, START + 2, START + 3, START + 4],
    ],
)
def test_decode_box_rejects_tokens_outside_the_sealed_range(bad_tokens: list[int]) -> None:
    with pytest.raises(geometry.GeometryContractError):
        geometry.decode_box(bad_tokens, label="box")


@pytest.mark.parametrize(
    "box_bins",
    [(10, 10, 10, 20), (10, 10, 20, 10), (30, 10, 20, 20)],
)
def test_decode_box_requires_strict_positive_extents(box_bins: tuple[int, ...]) -> None:
    with pytest.raises(geometry.GeometryContractError, match="strict positive extents"):
        geometry.decode_box(tokens(*box_bins), label="box")


def test_box_geometry_uses_continuous_xyxy_without_a_plus_one_convention() -> None:
    c_box = geometry.Box(0, 0, 100, 100)
    r_box = geometry.Box(50, 0, 150, 100)
    result = geometry.box_geometry(c_box, r_box, label="pair")
    # Continuous geometry: areas are 100*100, the intersection is 50*100.
    assert result["c_area_bins2"] == pytest.approx(10000.0)
    assert result["intersection_area_bins2"] == pytest.approx(5000.0)
    assert result["union_area_bins2"] == pytest.approx(15000.0)
    assert result["iou"] == pytest.approx(5000.0 / 15000.0)
    assert result["intersection_over_c_area"] == pytest.approx(0.5)
    assert result["intersection_over_r_area"] == pytest.approx(0.5)


def test_box_geometry_signed_offsets_and_area_ratios() -> None:
    c_box = geometry.Box(100, 100, 300, 200)
    r_box = geometry.Box(200, 300, 500, 500)
    result = geometry.box_geometry(c_box, r_box, label="pair")
    # centres: C=(200,150), R=(350,400); extents: C=(200,100), R=(300,200)
    assert result["dx"] == pytest.approx(0.150)
    assert result["dy"] == pytest.approx(0.250)
    assert result["dw"] == pytest.approx(0.100)
    assert result["dh"] == pytest.approx(0.100)
    assert result["area_ratio_r_over_c"] == pytest.approx(60000.0 / 20000.0)
    assert result["abs_log_area_ratio"] == pytest.approx(abs(math.log(3.0)))
    assert result["center_distance_bins"] == pytest.approx(math.hypot(150.0, 250.0))
    assert result["center_distance_normalized"] == pytest.approx(
        math.hypot(150.0, 250.0) / (1000.0 * math.sqrt(2.0))
    )


def test_box_geometry_center_containment_includes_the_boundary() -> None:
    # R's centre lands exactly on C's right edge.
    c_box = geometry.Box(0, 0, 100, 100)
    r_box = geometry.Box(100, 40, 100 + 2 * 100, 60)
    assert r_box.center[0] == pytest.approx(200.0)
    boundary = geometry.Box(0, 0, 200, 100)
    result = geometry.box_geometry(boundary, r_box, label="pair")
    assert result["r_center_inside_c"] is True
    assert result["any_overlap_or_center_containment"] is True
    del c_box


def test_geometric_labels_are_exhaustive_and_mutually_exclusive() -> None:
    cases = {
        # exact IoU 0.5 sits on the frozen high-overlap cutoff
        geometry.LABEL_HIGH_OVERLAP: (geometry.Box(0, 0, 100, 100), geometry.Box(0, 0, 100, 200)),
        geometry.LABEL_PARTIAL: (geometry.Box(0, 0, 100, 100), geometry.Box(90, 90, 400, 400)),
        geometry.LABEL_SEPARATED: (geometry.Box(0, 0, 100, 100), geometry.Box(300, 300, 400, 400)),
    }
    for expected, (c_box, r_box) in cases.items():
        result = geometry.box_geometry(c_box, r_box, label="pair")
        assert result["geometric_label"] == expected
        assert result["any_overlap_or_center_containment"] is not result["clearly_separated"]
        if expected == geometry.LABEL_HIGH_OVERLAP:
            assert result["iou"] == pytest.approx(0.5)
            assert result["high_overlap"] is True
            assert result["any_overlap_or_center_containment"] is True


def test_zero_iou_with_center_containment_is_not_clearly_separated() -> None:
    # Degenerate touching boxes: no area overlap, but a centre lands on the shared edge.
    c_box = geometry.Box(0, 0, 100, 100)
    r_box = geometry.Box(100, 0, 300, 100)
    result = geometry.box_geometry(c_box, r_box, label="pair")
    assert result["iou"] == pytest.approx(0.0)
    assert result["clearly_separated"] is True
    contained = geometry.box_geometry(c_box, geometry.Box(50, 100, 150, 300), label="pair")
    assert contained["iou"] == pytest.approx(0.0)
    assert contained["clearly_separated"] is True


# ---------------------------------------------------------------------------
# Sorted-key rank gap
# ---------------------------------------------------------------------------


def _population(entries: Sequence[tuple[str, tuple[int, int, int, int]]]) -> dict[str, Any]:
    return {
        "image_id": "i",
        "normalized_description": "person",
        "members": sorted(
            (
                {
                    "gt_owner_id": owner_id,
                    "box": geometry.Box(*box),
                    "sealed_pixel_sort_key": [box[1], box[0]],
                }
                for owner_id, box in entries
            ),
            key=lambda item: (item["box"].y1, item["box"].x1, item["gt_owner_id"]),
        ),
        "pixel_sort_key_order_agrees": True,
    }


def test_sorted_key_rank_gap_inserts_the_row_key_and_reports_a_signed_gap() -> None:
    population = _population(
        [("gt:i:0", (10, 10, 20, 20)), ("gt:i:1", (10, 300, 20, 400)), ("gt:i:2", (10, 500, 20, 600))]
    )
    ranks = geometry.sorted_key_rank_gap(
        population,
        c_owner_id="gt:i:0",
        r_box=geometry.Box(10, 400, 20, 450),
        r_tie_key="row:p_plus_c_then_e:gt:i:0",
    )
    assert ranks["population_size"] == 3
    assert ranks["c_rank_in_population"] == 0
    assert ranks["c_rank_after_insertion"] == 0
    assert ranks["r_rank_after_insertion"] == 2
    assert ranks["sorted_key_rank_gap"] == 2
    assert ranks["rank_key_tie_with_physical_owner"] is False


def test_sorted_key_rank_gap_places_the_inserted_row_after_equal_key_owners() -> None:
    # "zzz:i:1" sorts *after* "row:..." lexicographically, so a string tie-break alone
    # would put the row first; the explicit insertion rule must still place it last.
    population = _population([("gt:i:0", (10, 10, 20, 20)), ("zzz:i:1", (300, 10, 400, 20))])
    ranks = geometry.sorted_key_rank_gap(
        population,
        c_owner_id="gt:i:0",
        r_box=geometry.Box(300, 10, 400, 20),
        r_tie_key="row:p_plus_c_then_e:gt:i:0",
    )
    assert ranks["rank_key_tie_with_physical_owner"] is True
    assert ranks["inserted_row_sorts_after_equal_key_owners"] is True
    assert ranks["r_rank_after_insertion"] == 2
    assert ranks["sorted_key_rank_gap"] == 2


def test_sorted_key_rank_gap_reports_pixel_order_as_provenance_only() -> None:
    population = _population([("gt:i:0", (10, 10, 20, 20)), ("gt:i:1", (10, 300, 20, 400))])
    population["pixel_sort_key_order_agrees"] = False
    ranks = geometry.sorted_key_rank_gap(
        population,
        c_owner_id="gt:i:0",
        r_box=geometry.Box(10, 400, 20, 450),
        r_tie_key="row:p_plus_c_then_e:gt:i:0",
    )
    # The pixel key is recorded, never mixed into the norm-1000 rank.
    assert ranks["pixel_sort_key_order_agrees"] is False
    assert ranks["pixel_sort_key_role"] == "provenance_only_never_mixed_into_this_rank"
    assert ranks["sorted_key_rank_gap"] == 2


def test_sorted_key_rank_gap_requires_c_in_its_own_population() -> None:
    population = _population([("gt:i:1", (10, 10, 20, 20))])
    with pytest.raises(geometry.GeometryContractError, match="missing from its own"):
        geometry.sorted_key_rank_gap(
            population,
            c_owner_id="gt:i:0",
            r_box=geometry.Box(10, 10, 20, 20),
            r_tie_key="row:p_plus_c_then_e:gt:i:0",
        )


# ---------------------------------------------------------------------------
# Spearman association
# ---------------------------------------------------------------------------


def test_spearman_rho_is_tie_corrected_and_undefined_without_variance() -> None:
    assert geometry.spearman_rho([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)
    assert geometry.spearman_rho([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    # Average ranks: a tied pair must not become an arbitrary ordering.
    assert geometry.spearman_rho([1.0, 1.0, 2.0, 3.0], [1.0, 1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert geometry.spearman_rho([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert geometry.spearman_rho([1.0, 2.0], [1.0, 2.0]) is None


# ---------------------------------------------------------------------------
# The frozen three-way decision
# ---------------------------------------------------------------------------


def decision_row(
    *,
    owner_id: str,
    image_id: str,
    material: bool,
    description_equal: bool,
    label: str,
    arm: str = geometry.PRIMARY_ARM,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "gt_owner_id": owner_id,
        "image_id": image_id,
        "material_negative": material,
        "description_equal": description_equal,
        "geometry": {
            "any_overlap_or_center_containment": label != geometry.LABEL_SEPARATED,
            "clearly_separated": label == geometry.LABEL_SEPARATED,
        },
    }


def overlap_row(owner_id: str, image_id: str, *, material: bool = True) -> dict[str, Any]:
    return decision_row(
        owner_id=owner_id,
        image_id=image_id,
        material=material,
        description_equal=True,
        label=geometry.LABEL_HIGH_OVERLAP,
    )


def separated_row(
    owner_id: str, image_id: str, *, description_equal: bool = False, material: bool = True
) -> dict[str, Any]:
    return decision_row(
        owner_id=owner_id,
        image_id=image_id,
        material=material,
        description_equal=description_equal,
        label=geometry.LABEL_SEPARATED,
    )


def test_decision_exhaustive_order_puts_separated_competition_first() -> None:
    assert geometry.DECISION_EXHAUSTIVE_ORDER == (
        geometry.OUTCOME_SEPARATED_COMPETITION,
        geometry.OUTCOME_IDENTITY_SLIPPAGE,
        geometry.OUTCOME_INCONCLUSIVE,
    )
    assert geometry.MIN_QUALIFYING_IMAGE_COUNT == 2


def test_decision_identity_slippage_branch() -> None:
    rows = [
        overlap_row("gt:a:0", "a"),
        overlap_row("gt:a:1", "a"),
        overlap_row("gt:b:0", "b"),
        separated_row("gt:b:1", "b", material=False),
    ]
    block = geometry._arm_decision_block(rows, arm=geometry.PRIMARY_ARM)
    assert block["material_negative_count"] == 3
    assert block["same_description_and_any_overlap_fraction"] == pytest.approx(1.0)
    assert block["same_description_and_any_overlap_image_ids"] == ["a", "b"]
    assert block["clearly_separated_count"] == 0
    assert block["outcome"] == geometry.OUTCOME_IDENTITY_SLIPPAGE


def test_decision_separated_competition_branch() -> None:
    rows = [
        separated_row("gt:a:0", "a"),
        separated_row("gt:a:1", "a"),
        separated_row("gt:b:0", "b"),
    ]
    block = geometry._arm_decision_block(rows, arm=geometry.PRIMARY_ARM)
    assert block["clearly_separated_count"] == 3
    assert block["clearly_separated_image_ids"] == ["a", "b"]
    assert block["clearly_separated_different_description_count"] == 3
    assert block["outcome"] == geometry.OUTCOME_SEPARATED_COMPETITION


def test_separated_competition_wins_when_both_routes_would_otherwise_hold() -> None:
    # Nine same-description overlapping rows over two images would satisfy the 75%
    # identity threshold, but three same-description separated rows are also present.
    rows = [overlap_row(f"gt:a:{index}", "a" if index % 2 else "b") for index in range(9)]
    rows.extend(
        [
            separated_row("gt:c:0", "c", description_equal=True),
            separated_row("gt:c:1", "c", description_equal=True),
            separated_row("gt:d:0", "d", description_equal=True),
        ]
    )
    block = geometry._arm_decision_block(rows, arm=geometry.PRIMARY_ARM)
    assert block["same_description_and_any_overlap_fraction"] == pytest.approx(0.75)
    assert block["predicates"]["zero_material_negative_clearly_separated_different_description"]
    assert block["predicates"]["identity_route_blocked_by_separated_precedence"] is True
    assert block["predicates"]["identity_slippage_duplicate_suppression_eligible"] is False
    assert block["outcome"] == geometry.OUTCOME_SEPARATED_COMPETITION


def test_single_image_separated_rows_route_inconclusive() -> None:
    rows = [separated_row(f"gt:a:{index}", "a") for index in range(4)]
    block = geometry._arm_decision_block(rows, arm=geometry.PRIMARY_ARM)
    assert block["clearly_separated_count"] == 4
    assert block["clearly_separated_image_ids"] == ["a"]
    assert block["predicates"]["clearly_separated_count_at_least_minimum"] is True
    assert block["predicates"]["separated_qualifying_rows_span_two_images"] is False
    assert block["outcome"] == geometry.OUTCOME_INCONCLUSIVE


def test_single_image_identity_rows_route_inconclusive() -> None:
    rows = [overlap_row(f"gt:a:{index}", "a") for index in range(5)]
    block = geometry._arm_decision_block(rows, arm=geometry.PRIMARY_ARM)
    assert block["same_description_and_any_overlap_fraction"] == pytest.approx(1.0)
    assert block["same_description_and_any_overlap_image_ids"] == ["a"]
    assert block["predicates"]["identity_qualifying_rows_span_two_images"] is False
    assert block["outcome"] == geometry.OUTCOME_INCONCLUSIVE


@pytest.mark.parametrize(
    "rows",
    [
        # too few material-negative rows for either branch
        [overlap_row("gt:a:0", "a"), overlap_row("gt:b:0", "b", material=False)],
        # enough material rows, but the same-description overlap fraction misses 0.75
        [
            overlap_row("gt:a:0", "a"),
            decision_row(owner_id="gt:b:0", image_id="b", material=True,
                         description_equal=False, label=geometry.LABEL_PARTIAL),
            decision_row(owner_id="gt:c:0", image_id="c", material=True,
                         description_equal=False, label=geometry.LABEL_PARTIAL),
        ],
    ],
)
def test_decision_inconclusive_branch(rows: list[dict[str, Any]]) -> None:
    block = geometry._arm_decision_block(rows, arm=geometry.PRIMARY_ARM)
    assert block["outcome"] == geometry.OUTCOME_INCONCLUSIVE


def test_decision_identity_slippage_is_blocked_by_one_separated_different_description_row() -> None:
    rows = [overlap_row(f"gt:a:{index}", "a" if index % 2 else "b") for index in range(9)]
    rows.append(separated_row("gt:c:0", "c"))
    block = geometry._arm_decision_block(rows, arm=geometry.PRIMARY_ARM)
    assert block["same_description_and_any_overlap_fraction"] == pytest.approx(0.9)
    assert block["predicates"][
        "zero_material_negative_clearly_separated_different_description"
    ] is False
    # One separated row is below the separated-route minimum, so neither route opens.
    assert block["predicates"]["clearly_separated_count_at_least_minimum"] is False
    assert block["outcome"] == geometry.OUTCOME_INCONCLUSIVE


def test_material_negative_cutoff_is_inclusive_at_minus_one_nat() -> None:
    assert geometry.MATERIAL_NEGATIVE_MAX_NATS == -1.0
    assert (-1.0 <= geometry.MATERIAL_NEGATIVE_MAX_NATS) is True
    assert (-0.999999 <= geometry.MATERIAL_NEGATIVE_MAX_NATS) is False


def test_primary_decision_never_counts_one_owner_twice() -> None:
    rows = [overlap_row("gt:a:0", "a"), overlap_row("gt:a:0", "a")]
    with pytest.raises(geometry.GeometryContractError, match="more than once"):
        geometry.build_decision(rows, [])


# ---------------------------------------------------------------------------
# The cross-image pooling guard
# ---------------------------------------------------------------------------


def test_pooling_guard_rejects_a_cross_image_raw_delta_summary() -> None:
    with pytest.raises(geometry.GeometryContractError, match="pools raw deltas"):
        geometry.assert_no_cross_image_raw_delta_pooling(
            {"summary": {"median_relative_coordinate_delta": -1.2}}, label="payload"
        )
    with pytest.raises(geometry.GeometryContractError, match="pools raw deltas"):
        geometry.assert_no_cross_image_raw_delta_pooling(
            {"blocks": [{"coordinate_delta_quantiles": [0.1]}]}, label="payload"
        )


def test_pooling_guard_allows_per_image_and_within_image_scopes() -> None:
    geometry.assert_no_cross_image_raw_delta_pooling(
        {
            "per_image": {
                "1584": {"within_image_median_relative_coordinate_delta": -0.5},
            },
            "associations": {"iou_versus_relative_coordinate_delta": {"spearman_rho": -0.5}},
        },
        label="payload",
    )


# ---------------------------------------------------------------------------
# Miniature sealed input tree
# ---------------------------------------------------------------------------

MINI_CONTRACT = geometry.DenominatorContract(
    crossing_owner_count=2,
    greedy_pair_count=1,
    benign_control_count=2,
    secondary_owner_row_count=4,
    secondary_merged_row_count=6,
)


def _row_tokens(coord_tokens: Sequence[int]) -> list[int]:
    """A sealed nine-token row: four description-path tokens, four coords, one end."""

    return [151646, 18147, 151647, 151648, *coord_tokens, 151649]


def _segments(delta: float) -> dict[str, Any]:
    return {
        "coordinates": {
            "baseline_sum": -4.0,
            "modified_sum": -4.0 + delta,
            "delta": delta,
            "delta_token_mean": delta / 4.0,
            "token_count": 4,
            "sign": -1 if delta < 0 else 1,
            "finite": True,
        },
        "complete_row": {
            "baseline_sum": -9.0,
            "modified_sum": -9.0 + delta,
            "delta": delta,
            "delta_token_mean": delta / 9.0,
            "token_count": 9,
            "sign": -1 if delta < 0 else 1,
            "finite": True,
        },
    }


def _owner_entry(
    owner_id: str,
    image_id: str,
    box: tuple[int, int, int, int],
    *,
    description: str,
    category_id: int = 1,
    native_true_positive: bool = True,
) -> dict[str, Any]:
    return {
        "gt_owner_id": owner_id,
        "image_id": image_id,
        "official_coco_category_id": category_id,
        "normalized_description": description,
        "owner_sort_key": [box[1], box[0]],
        "native_true_positive": native_true_positive,
        "calibration_role": (
            "native_true_positive_calibration" if native_true_positive else "native_false_negative"
        ),
        "greedy_eligibility_status": "eligible",
        "excluded_from_census": False,
        "disposition_eligibility": {"native_false_negative": not native_true_positive},
        "candidate_bank": {
            "logical_roles": [
                {"role": "exact_gt_anchor", "coord_token_ids": tokens(*box)},
                {"role": "translate_left", "coord_token_ids": tokens(*box)},
            ]
        },
    }


def _sealed_row(
    coord_box: tuple[int, int, int, int],
    *,
    description: str,
    row_index: int,
    strict_match: str | None,
    image_id: str,
) -> dict[str, Any]:
    coord = tokens(*coord_box)
    return {
        "coord_token_ids": coord,
        "full_row_token_ids": _row_tokens(coord),
        "full_row_token_count": 9,
        "normalized_description": description,
        "row_index": row_index,
        "pred_row_id": f"pred:sorted:greedy:0:{image_id}:{row_index}",
        "strict_match_status": "matched" if strict_match else "unmatched",
        "strict_match_gt_owner_id": strict_match,
    }


def default_spec() -> dict[str, Any]:
    """Two images, one crossing owner each, one greedy displacement pair."""

    owners = [
        _owner_entry("gt:img1:0", "img1", (100, 100, 200, 200), description="person"),
        _owner_entry(
            "gt:img1:1",
            "img1",
            (150, 150, 260, 260),
            description="person",
            native_true_positive=False,
        ),
        _owner_entry("gt:img2:0", "img2", (400, 400, 500, 500), description="chair"),
        _owner_entry("gt:img2:1", "img2", (700, 700, 800, 800), description="chair"),
    ]
    images = [
        {
            "image_id": image_id,
            "image_width": 1024,
            "image_height": 1024,
            "file_name": f"images/val2017/{image_id}.jpg",
            "executed_media_sha256": "0" * 64,
            "coordinate_token_ids": {
                "start": geometry.COORD_TOKEN_START,
                "bin_count": geometry.COORD_BIN_COUNT,
                "end_inclusive": geometry.COORD_TOKEN_START + geometry.COORD_BIN_COUNT - 1,
            },
        }
        for image_id in ("img1", "img2")
    ]
    cohort = [
        {
            "gt_owner_id": "gt:img1:0",
            "image_id": "img1",
            "official_coco_category_id": 1,
            "normalized_description": "person",
            "same_description_as_e": True,
            "f_row_present": True,
            "inserted_clean_row_c": {"coord_token_ids": tokens(100, 100, 200, 200)},
            "e_row": _sealed_row(
                (150, 150, 260, 260),
                description="person",
                row_index=4,
                strict_match="gt:img1:1",
                image_id="img1",
            ),
            "f_row": _sealed_row(
                (600, 600, 700, 700),
                description="person",
                row_index=5,
                strict_match=None,
                image_id="img1",
            ),
        },
        {
            "gt_owner_id": "gt:img2:0",
            "image_id": "img2",
            "official_coco_category_id": 1,
            "normalized_description": "chair",
            "same_description_as_e": False,
            "f_row_present": True,
            "inserted_clean_row_c": {"coord_token_ids": tokens(400, 400, 500, 500)},
            "e_row": _sealed_row(
                (10, 10, 60, 60),
                description="bottle",
                row_index=2,
                strict_match=None,
                image_id="img2",
            ),
            "f_row": _sealed_row(
                (20, 20, 90, 90),
                description="bottle",
                row_index=3,
                strict_match=None,
                image_id="img2",
            ),
        },
    ]
    primary = [
        {
            "cohort": geometry.CROSSING_COHORT,
            "gt_owner_id": "gt:img1:0",
            "image_id": "img1",
            "stratum": "matched_e",
            "primary_branch": "displaced",
            "primary_branch_reasons": ["greedy_displaced"],
            "sensitivity_branch_l": "displaced",
            "description_observability": "same_description_release_observable",
            "interpretable": True,
            "quarantined": False,
            "tie_or_nonunique": False,
            "displacement": {
                "greedy_displaced": True,
                "greedy_displaced_owner_id": "gt:img1:1",
                "likelihood_displaced": False,
                "likelihood_displaced_owner_id": None,
                "decoding_contradicted": False,
            },
        },
        {
            "cohort": geometry.CROSSING_COHORT,
            "gt_owner_id": "gt:img2:0",
            "image_id": "img2",
            "stratum": "unmatched_e",
            "primary_branch": "release_lost",
            "primary_branch_reasons": [],
            "sensitivity_branch_l": "release_lost",
            "description_observability": "different_description_release_observable",
            "interpretable": True,
            "quarantined": False,
            "tie_or_nonunique": False,
            "displacement": {
                "greedy_displaced": False,
                "greedy_displaced_owner_id": None,
                "likelihood_displaced": False,
                "likelihood_displaced_owner_id": None,
                "decoding_contradicted": False,
            },
        },
    ]
    deltas = {
        ("gt:img1:0", geometry.PRIMARY_ARM): -2.5,
        ("gt:img1:0", geometry.SENSITIVITY_ARM): 0.2,
        ("gt:img2:0", geometry.PRIMARY_ARM): -0.1,
        ("gt:img2:0", geometry.SENSITIVITY_ARM): -3.0,
    }
    secondary = []
    merged = []
    for row in cohort:
        owner_id = row["gt_owner_id"]
        readouts: dict[str, Any] = {}
        for arm, row_key, _label, _role in geometry.ARMS:
            request_id = f"req:{owner_id}:{arm}"
            delta = deltas[(owner_id, arm)]
            readouts[arm] = {
                "variant": arm,
                "request_id": request_id,
                "scored_target_kind": "exact_native_row",
                "scored_token_count": 9,
                "inserted_clean_row_c_token_count": 10,
                "segments": _segments(delta),
            }
            merged.append(
                {
                    "cohort": geometry.CROSSING_COHORT,
                    "gt_owner_id": owner_id,
                    "image_id": row["image_id"],
                    "variant": arm,
                    "request_id": request_id,
                    "scored_token_ids": list(row[row_key]["full_row_token_ids"]),
                    "segments": {
                        "coordinates": {"token_ids": list(row[row_key]["coord_token_ids"])}
                    },
                }
            )
        secondary.append(
            {
                "cohort": geometry.CROSSING_COHORT,
                "gt_owner_id": owner_id,
                "image_id": row["image_id"],
                "readout_variants": [arm for arm, *_ in geometry.ARMS],
                "readouts": readouts,
            }
        )
    for image_id in ("img1", "img2"):
        owner_id = f"gt:{image_id}:control"
        request_id = f"req:{owner_id}:benign"
        secondary.append(
            {
                "cohort": geometry.BENIGN_COHORT,
                "gt_owner_id": owner_id,
                "image_id": image_id,
                "readout_variants": [geometry.BENIGN_VARIANT],
                "readouts": {
                    geometry.BENIGN_VARIANT: {
                        "variant": geometry.BENIGN_VARIANT,
                        "request_id": request_id,
                        "scored_target_kind": "native_row",
                        "scored_token_count": 9,
                        "inserted_clean_row_c_token_count": 9,
                        "segments": _segments(-0.05),
                    }
                },
            }
        )
        merged.append(
            {
                "cohort": geometry.BENIGN_COHORT,
                "gt_owner_id": owner_id,
                "image_id": image_id,
                "variant": geometry.BENIGN_VARIANT,
                "request_id": request_id,
                "scored_token_ids": _row_tokens(tokens(1, 1, 2, 2)),
                "segments": {"coordinates": {"token_ids": tokens(1, 1, 2, 2)}},
            }
        )
    return {
        "owners": owners,
        "images": images,
        "cohort": cohort,
        "primary": primary,
        "secondary": secondary,
        "merged": merged,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _write_sealed_json(path: Path, payload: dict[str, Any], *, digest_key: str) -> None:
    payload = dict(payload)
    payload.pop(digest_key, None)
    payload[digest_key] = geometry.sha256_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _digest_entry(path: Path, row_count: int | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": path.name,
        "byte_size": path.stat().st_size,
        "sha256": geometry.sha256_file(path),
    }
    if row_count is not None:
        entry["row_count"] = row_count
    return entry


def build_tree(root: Path, spec: Mapping[str, Any]) -> dict[str, Path]:
    """Materialize a miniature but structurally complete sealed input tree."""

    census_root = root / "census-run"
    census_plan = census_root / "plan"
    owner_registry = census_plan / geometry.CENSUS_OWNER_REGISTRY_NAME
    image_registry = census_plan / geometry.CENSUS_IMAGE_REGISTRY_NAME
    _write_jsonl(owner_registry, spec["owners"])
    _write_jsonl(image_registry, spec["images"])
    _write_sealed_json(
        census_plan / geometry.CENSUS_RECEIPT_NAME,
        {"unit_id": geometry.CENSUS_UNIT_ID, "census_shape": {"owner_count": len(spec["owners"])}},
        digest_key="receipt_content_sha256",
    )
    census_receipt = json.loads(
        (census_plan / geometry.CENSUS_RECEIPT_NAME).read_text(encoding="utf-8")
    )

    plan_dir = root / "plan"
    cohort_path = plan_dir / geometry.COHORT_REGISTRY_NAME
    _write_jsonl(cohort_path, spec["cohort"])
    _write_sealed_json(
        plan_dir / geometry.PLAN_MANIFEST_NAME,
        {
            "unit_id": geometry.SOURCE_UNIT_ID,
            "builder_source": {"path": "builder.py", "sha256": "1" * 64},
            "output_file_digests": {
                geometry.COHORT_REGISTRY_NAME: _digest_entry(cohort_path, len(spec["cohort"]))
            },
            "lineage": {
                "census_unit_id": geometry.CENSUS_UNIT_ID,
                "census_run_root": str(census_root),
                "census_plan_receipt_content_sha256": census_receipt["receipt_content_sha256"],
                "census_input_files": {
                    f"plan/{geometry.CENSUS_OWNER_REGISTRY_NAME}": _digest_entry(owner_registry),
                    f"plan/{geometry.CENSUS_IMAGE_REGISTRY_NAME}": _digest_entry(image_registry),
                },
            },
        },
        digest_key="manifest_content_sha256",
    )

    primary_dir = root / "primary-analysis"
    primary_rows = primary_dir / geometry.PRIMARY_OWNER_ROWS_NAME
    _write_jsonl(primary_rows, spec["primary"])
    _write_sealed_json(
        primary_dir / geometry.PRIMARY_RECEIPT_NAME,
        {
            "unit_id": geometry.SOURCE_UNIT_ID,
            "analyzer_source_sha256": "2" * 64,
            "output_file_digests": {
                geometry.PRIMARY_OWNER_ROWS_NAME: _digest_entry(primary_rows, len(spec["primary"]))
            },
        },
        digest_key="receipt_content_sha256",
    )

    secondary_dir = root / "secondary-analysis"
    secondary_rows = secondary_dir / geometry.SECONDARY_OWNER_ROWS_NAME
    _write_jsonl(secondary_rows, spec["secondary"])
    _write_sealed_json(
        secondary_dir / geometry.SECONDARY_RECEIPT_NAME,
        {
            "unit_id": geometry.SOURCE_UNIT_ID,
            "analyzer_source_sha256": "3" * 64,
            "output_file_digests": {
                geometry.SECONDARY_OWNER_ROWS_NAME: _digest_entry(
                    secondary_rows, len(spec["secondary"])
                )
            },
        },
        digest_key="receipt_content_sha256",
    )

    merged_dir = root / "secondary-merged"
    merged_rows = merged_dir / geometry.SECONDARY_MERGED_ROWS_NAME
    _write_jsonl(merged_rows, spec["merged"])
    _write_sealed_json(
        merged_dir / geometry.SECONDARY_MERGE_RECEIPT_NAME,
        {
            "unit_id": geometry.SOURCE_UNIT_ID,
            "merger_source_sha256": "4" * 64,
            "output_file_digests": {
                geometry.SECONDARY_MERGED_ROWS_NAME: _digest_entry(
                    merged_rows, len(spec["merged"])
                )
            },
        },
        digest_key="receipt_content_sha256",
    )

    return {
        "plan_dir": plan_dir,
        "primary_owner_rows": primary_rows,
        "secondary_owner_rows": secondary_rows,
        "secondary_merged_dir": merged_dir,
        "census_plan_dir": census_plan,
    }


def run_mini(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    paths = build_tree(root, spec)
    return geometry.run_analysis(
        plan_dir=paths["plan_dir"],
        primary_owner_rows=paths["primary_owner_rows"],
        secondary_owner_rows=paths["secondary_owner_rows"],
        secondary_merged_dir=paths["secondary_merged_dir"],
        census_plan_dir=paths["census_plan_dir"],
        contract=MINI_CONTRACT,
    )


# ---------------------------------------------------------------------------
# Miniature-tree behaviour
# ---------------------------------------------------------------------------


def test_mini_tree_produces_one_row_per_owner_per_arm(tmp_path: Path) -> None:
    result = run_mini(tmp_path, default_spec())
    rows = result["owner_rows"]
    assert len(rows) == 4
    assert sorted((row["arm"], row["gt_owner_id"]) for row in rows) == [
        (geometry.PRIMARY_ARM, "gt:img1:0"),
        (geometry.PRIMARY_ARM, "gt:img2:0"),
        (geometry.SENSITIVITY_ARM, "gt:img1:0"),
        (geometry.SENSITIVITY_ARM, "gt:img2:0"),
    ]
    assert len(result["pair_rows"]) == 1
    pair = result["pair_rows"][0]
    assert pair["target_gt_owner_id"] == "gt:img1:0"
    assert pair["displacer_gt_owner_id"] == "gt:img1:1"
    assert pair["displacer_native_disposition"]["native_false_negative"] is True
    assert pair["sorted_key_ranks"]["sorted_key_rank_gap"] == 1


def test_each_arm_measures_its_own_downstream_row(tmp_path: Path) -> None:
    result = run_mini(tmp_path, default_spec())
    by_key = {(row["arm"], row["gt_owner_id"]): row for row in result["owner_rows"]}
    e_row = by_key[(geometry.PRIMARY_ARM, "gt:img1:0")]
    f_row = by_key[(geometry.SENSITIVITY_ARM, "gt:img1:0")]
    assert e_row["downstream_row_label"] == "E"
    assert e_row["geometry"]["r_box_norm1000_xyxy"] == [150, 150, 260, 260]
    assert e_row["enters_primary_decision"] is True
    assert f_row["downstream_row_label"] == "F"
    assert f_row["geometry"]["r_box_norm1000_xyxy"] == [600, 600, 700, 700]
    assert f_row["enters_primary_decision"] is False
    # The two arms must not share geometry.
    assert e_row["geometry"]["iou"] != f_row["geometry"]["iou"]


def test_relative_delta_subtracts_the_same_image_benign_reference(tmp_path: Path) -> None:
    result = run_mini(tmp_path, default_spec())
    row = next(
        row
        for row in result["owner_rows"]
        if row["arm"] == geometry.PRIMARY_ARM and row["gt_owner_id"] == "gt:img1:0"
    )
    assert row["likelihood"]["relative"]["coordinates"]["crossing_delta"] == pytest.approx(-2.5)
    assert row["likelihood"]["relative"]["coordinates"][
        "same_image_benign_delta"
    ] == pytest.approx(-0.05)
    assert row["relative_coordinate_delta"] == pytest.approx(-2.45)
    assert row["material_negative"] is True
    assert row["change_axis"] == "material_negative"


def test_benign_and_crossing_deltas_come_from_different_scored_rows(tmp_path: Path) -> None:
    result = run_mini(tmp_path, default_spec())
    row = next(
        row
        for row in result["owner_rows"]
        if row["arm"] == geometry.PRIMARY_ARM and row["gt_owner_id"] == "gt:img1:0"
    )
    reference = row["likelihood"]["benign_reference"]
    assert reference["scored_row_identity"] != row["likelihood"]["scored_row_identity"]
    assert reference["request_id"] != row["likelihood"]["request_id"]
    assert (
        reference["scored_coordinate_token_ids_sha256"]
        != row["likelihood"]["scored_coordinate_token_ids_sha256"]
    )
    assert reference["scored_row_is_distinct_from_this_arm"] is True
    # Each delta is a sum over its own row's four coordinate tokens.
    assert row["likelihood"]["segments"]["coordinates"]["token_count"] == 4


def test_benign_reference_sharing_a_request_with_an_arm_fails_closed(tmp_path: Path) -> None:
    spec = default_spec()
    shared = "req:gt:img1:0:" + geometry.PRIMARY_ARM
    for row in spec["secondary"]:
        if row["cohort"] == geometry.BENIGN_COHORT and row["image_id"] == "img1":
            row["readouts"][geometry.BENIGN_VARIANT]["request_id"] = shared
    for row in spec["merged"]:
        if row["cohort"] == geometry.BENIGN_COHORT and row["image_id"] == "img1":
            row["request_id"] = shared
    with pytest.raises(geometry.GeometryContractError, match="same request"):
        run_mini(tmp_path, spec)


def test_benign_control_must_score_a_four_token_coordinate_span(tmp_path: Path) -> None:
    spec = default_spec()
    for row in spec["merged"]:
        if row["cohort"] == geometry.BENIGN_COHORT and row["image_id"] == "img1":
            row["segments"]["coordinates"]["token_ids"] = tokens(1, 1, 2, 2)[:3]
    with pytest.raises(geometry.GeometryContractError, match="coordinate tokens, expected"):
        run_mini(tmp_path, spec)


def test_merge_rows_are_read_from_the_sealed_compatibility_file_name(tmp_path: Path) -> None:
    assert geometry.SECONDARY_MERGED_ROWS_NAME == "secondary-compatibility-rows.jsonl"
    paths = build_tree(tmp_path, default_spec())
    (paths["secondary_merged_dir"] / geometry.SECONDARY_MERGED_ROWS_NAME).rename(
        paths["secondary_merged_dir"] / "secondary-merged-rows.jsonl"
    )
    with pytest.raises(geometry.GeometryContractError, match="secondary merge rows are missing"):
        geometry.run_analysis(
            plan_dir=paths["plan_dir"],
            primary_owner_rows=paths["primary_owner_rows"],
            secondary_owner_rows=paths["secondary_owner_rows"],
            secondary_merged_dir=paths["secondary_merged_dir"],
            census_plan_dir=paths["census_plan_dir"],
            contract=MINI_CONTRACT,
        )


def test_summary_strata_cover_every_row_and_publish_the_full_grid(tmp_path: Path) -> None:
    result = run_mini(tmp_path, default_spec())
    strata = result["summary"]["strata"]
    for arm in (geometry.PRIMARY_ARM, geometry.SENSITIVITY_ARM):
        cells = strata[arm]["cells"]
        assert len(cells) == (
            len(geometry.DESCRIPTION_AXIS)
            * len(geometry.E_STRATUM_AXIS)
            * len(geometry.GEOMETRIC_LABELS)
            * len(geometry.CHANGE_AXIS)
        )
        assert sum(cell["count"] for cell in cells) == strata[arm]["row_count"] == 2


def test_row_mapping_verifies_the_secondary_owner_join(tmp_path: Path) -> None:
    mapping = run_mini(tmp_path, default_spec())["summary"]["row_mapping"]
    assert mapping["input_merge_row_count"] == 6
    assert mapping["secondary_owner_row_count"] == 4
    assert mapping["crossing_owner_count"] == 2
    assert mapping["crossing_readouts_per_owner"] == 2
    assert mapping["benign_control_count"] == 2
    assert mapping["owner_geometry_row_count"] == 4


def test_missing_sensitivity_variant_breaks_the_secondary_mapping(tmp_path: Path) -> None:
    spec = default_spec()
    spec["merged"] = [
        row
        for row in spec["merged"]
        if not (row["gt_owner_id"] == "gt:img1:0" and row["variant"] == geometry.SENSITIVITY_ARM)
    ]
    contract = geometry.DenominatorContract(
        crossing_owner_count=2,
        greedy_pair_count=1,
        benign_control_count=2,
        secondary_owner_row_count=4,
        secondary_merged_row_count=5,
    )
    paths = build_tree(tmp_path, spec)
    with pytest.raises(geometry.GeometryContractError, match="expected both"):
        geometry.run_analysis(
            plan_dir=paths["plan_dir"],
            primary_owner_rows=paths["primary_owner_rows"],
            secondary_owner_rows=paths["secondary_owner_rows"],
            secondary_merged_dir=paths["secondary_merged_dir"],
            census_plan_dir=paths["census_plan_dir"],
            contract=contract,
        )


def test_crossed_arm_wiring_fails_closed(tmp_path: Path) -> None:
    spec = default_spec()
    primary_row = next(
        row
        for row in spec["merged"]
        if row["gt_owner_id"] == "gt:img1:0" and row["variant"] == geometry.PRIMARY_ARM
    )
    sensitivity_row = next(
        row
        for row in spec["merged"]
        if row["gt_owner_id"] == "gt:img1:0" and row["variant"] == geometry.SENSITIVITY_ARM
    )
    primary_row["segments"], sensitivity_row["segments"] = (
        sensitivity_row["segments"],
        primary_row["segments"],
    )
    with pytest.raises(geometry.GeometryContractError, match="wrong geometry"):
        run_mini(tmp_path, spec)


def test_sealed_same_description_flag_must_match_the_exact_descriptions(tmp_path: Path) -> None:
    spec = default_spec()
    spec["cohort"][0]["same_description_as_e"] = False
    with pytest.raises(geometry.GeometryContractError, match="same_description_as_e"):
        run_mini(tmp_path, spec)


def test_sealed_stratum_must_match_the_sealed_strict_match_status(tmp_path: Path) -> None:
    spec = default_spec()
    spec["primary"][0]["stratum"] = "unmatched_e"
    with pytest.raises(geometry.GeometryContractError, match="disagrees with the sealed E"):
        run_mini(tmp_path, spec)


def test_matched_downstream_row_must_exist_in_the_owner_ledger(tmp_path: Path) -> None:
    spec = default_spec()
    spec["cohort"][0]["e_row"]["strict_match_gt_owner_id"] = "gt:img1:404"
    with pytest.raises(geometry.GeometryContractError, match="missing from the canonical"):
        run_mini(tmp_path, spec)


def test_inserted_c_must_equal_the_sealed_exact_gt_anchor(tmp_path: Path) -> None:
    spec = default_spec()
    spec["cohort"][0]["inserted_clean_row_c"]["coord_token_ids"] = tokens(101, 100, 200, 200)
    with pytest.raises(geometry.GeometryContractError, match="exact-GT-anchor"):
        run_mini(tmp_path, spec)


def test_duplicate_owner_ids_fail_closed(tmp_path: Path) -> None:
    spec = default_spec()
    spec["owners"].append(copy.deepcopy(spec["owners"][0]))
    with pytest.raises(geometry.GeometryContractError, match="duplicate gt_owner_id"):
        run_mini(tmp_path, spec)


def test_invalid_downstream_box_fails_closed(tmp_path: Path) -> None:
    spec = default_spec()
    spec["cohort"][0]["e_row"]["coord_token_ids"] = tokens(200, 150, 150, 260)
    spec["merged"][0]["segments"]["coordinates"]["token_ids"] = tokens(200, 150, 150, 260)
    with pytest.raises(geometry.GeometryContractError, match="strict positive extents"):
        run_mini(tmp_path, spec)


def test_nonfinite_segment_fails_closed(tmp_path: Path) -> None:
    spec = default_spec()
    readout = spec["secondary"][0]["readouts"][geometry.PRIMARY_ARM]
    readout["segments"]["coordinates"]["finite"] = False
    with pytest.raises(geometry.GeometryContractError, match="not finite"):
        run_mini(tmp_path, spec)


def test_pixel_order_disagreement_is_recorded_but_never_mixed_into_the_rank(
    tmp_path: Path,
) -> None:
    spec = default_spec()
    baseline = run_mini(tmp_path / "baseline", copy.deepcopy(spec))
    # gt:img1:1 sits below gt:img1:0 in norm space; claim the opposite in pixel space.
    spec["owners"][1]["owner_sort_key"] = [0, 0]
    drifted = run_mini(tmp_path / "drifted", spec)

    def ranks(result: Mapping[str, Any]) -> dict[str, Any]:
        return next(
            row["sorted_key_ranks"]
            for row in result["owner_rows"]
            if row["arm"] == geometry.PRIMARY_ARM and row["gt_owner_id"] == "gt:img1:0"
        )

    assert ranks(baseline)["pixel_sort_key_order_agrees"] is True
    assert ranks(drifted)["pixel_sort_key_order_agrees"] is False
    # The rank itself is unchanged: it is derived only from norm-1000 tokens.
    assert ranks(drifted)["sorted_key_rank_gap"] == ranks(baseline)["sorted_key_rank_gap"]
    assert drifted["summary"]["sorted_key_populations"][
        "populations_where_pixel_order_disagrees"
    ] == ["img1|person"]


def test_same_category_population_is_the_sealed_normalized_description(tmp_path: Path) -> None:
    spec = default_spec()
    # A same-category-id owner with a different sealed description must not enter the rank.
    spec["owners"].append(
        _owner_entry("gt:img1:2", "img1", (120, 120, 180, 180), description="bicycle")
    )
    result = run_mini(tmp_path, spec)
    ranks = next(
        row["sorted_key_ranks"]
        for row in result["owner_rows"]
        if row["arm"] == geometry.PRIMARY_ARM and row["gt_owner_id"] == "gt:img1:0"
    )
    assert ranks["population_normalized_description"] == "person"
    assert ranks["population_size"] == 2


def test_unsealed_coordinate_token_identity_fails_closed(tmp_path: Path) -> None:
    spec = default_spec()
    spec["images"][0]["coordinate_token_ids"]["start"] = 151000
    with pytest.raises(geometry.GeometryContractError, match="not the frozen"):
        run_mini(tmp_path, spec)


def test_greedy_displacer_missing_from_the_ledger_fails_closed(tmp_path: Path) -> None:
    spec = default_spec()
    spec["primary"][0]["displacement"]["greedy_displaced_owner_id"] = "gt:img1:404"
    with pytest.raises(geometry.GeometryContractError, match="missing from the canonical"):
        run_mini(tmp_path, spec)


def test_greedy_pair_count_must_match_the_contract(tmp_path: Path) -> None:
    spec = default_spec()
    spec["primary"][1]["displacement"]["greedy_displaced"] = True
    spec["primary"][1]["displacement"]["greedy_displaced_owner_id"] = "gt:img2:1"
    with pytest.raises(geometry.GeometryContractError, match="greedy displacement pairs hold"):
        run_mini(tmp_path, spec)


# ---------------------------------------------------------------------------
# Digest binding and publication
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target",
    [
        "plan/cohort-registry.jsonl",
        "primary-analysis/owner-rows.jsonl",
        "secondary-analysis/secondary-owner-rows.jsonl",
        "secondary-merged/secondary-compatibility-rows.jsonl",
        "census-run/plan/owner-registry.jsonl",
        "census-run/plan/image-registry.jsonl",
    ],
)
def test_input_digest_drift_fails_before_any_output(tmp_path: Path, target: str) -> None:
    paths = build_tree(tmp_path, default_spec())
    tampered = tmp_path / target
    tampered.write_text(tampered.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(geometry.GeometryContractError, match="digest mismatch"):
        geometry.run_analysis(
            plan_dir=paths["plan_dir"],
            primary_owner_rows=paths["primary_owner_rows"],
            secondary_owner_rows=paths["secondary_owner_rows"],
            secondary_merged_dir=paths["secondary_merged_dir"],
            census_plan_dir=paths["census_plan_dir"],
            contract=MINI_CONTRACT,
        )


def test_unsealed_receipt_fails_closed(tmp_path: Path) -> None:
    paths = build_tree(tmp_path, default_spec())
    manifest_path = paths["plan_dir"] / geometry.PLAN_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unit_id"] = geometry.SOURCE_UNIT_ID + "-tampered"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(geometry.GeometryContractError, match="not self-sealed"):
        geometry.run_analysis(
            plan_dir=paths["plan_dir"],
            primary_owner_rows=paths["primary_owner_rows"],
            secondary_owner_rows=paths["secondary_owner_rows"],
            secondary_merged_dir=paths["secondary_merged_dir"],
            census_plan_dir=paths["census_plan_dir"],
            contract=MINI_CONTRACT,
        )


def test_census_lineage_must_match_the_plan_seal(tmp_path: Path) -> None:
    spec = default_spec()
    paths = build_tree(tmp_path, spec)
    other_root = tmp_path / "other"
    other_paths = build_tree(other_root, {**spec, "owners": spec["owners"][:3]})
    with pytest.raises(geometry.GeometryContractError, match="does not match the crossing plan"):
        geometry.run_analysis(
            plan_dir=paths["plan_dir"],
            primary_owner_rows=paths["primary_owner_rows"],
            secondary_owner_rows=paths["secondary_owner_rows"],
            secondary_merged_dir=paths["secondary_merged_dir"],
            census_plan_dir=other_paths["census_plan_dir"],
            contract=MINI_CONTRACT,
        )


def test_publication_is_atomic_create_or_identical(tmp_path: Path) -> None:
    result = run_mini(tmp_path, default_spec())
    files = geometry.build_output_files(result)
    output_dir = tmp_path / "analysis"

    first = geometry.publish_analysis(output_dir, files)
    assert first["published"] is True
    assert first["publish_mode"] == "atomic_staging_directory_rename"
    assert sorted(entry.name for entry in output_dir.iterdir()) == sorted(files)
    assert not list(tmp_path.glob("analysis.staging-*"))

    second = geometry.publish_analysis(output_dir, geometry.build_output_files(result))
    assert second["published"] is False
    assert second["publish_mode"] == "no_op_identical_rerun"

    drifted = dict(files)
    drifted[geometry.REPORT_MD_NAME] = files[geometry.REPORT_MD_NAME] + b"drift\n"
    with pytest.raises(geometry.GeometryContractError, match="not a byte-identical rerun"):
        geometry.publish_analysis(output_dir, drifted)
    # The existing directory is left untouched.
    assert (output_dir / geometry.REPORT_MD_NAME).read_bytes() == files[geometry.REPORT_MD_NAME]


def test_identical_rerun_reproduces_identical_bytes(tmp_path: Path) -> None:
    paths = build_tree(tmp_path, default_spec())

    def once() -> dict[str, bytes]:
        return geometry.build_output_files(
            geometry.run_analysis(
                plan_dir=paths["plan_dir"],
                primary_owner_rows=paths["primary_owner_rows"],
                secondary_owner_rows=paths["secondary_owner_rows"],
                secondary_merged_dir=paths["secondary_merged_dir"],
                census_plan_dir=paths["census_plan_dir"],
                contract=MINI_CONTRACT,
            )
        )

    assert once() == once()


def test_analysis_content_does_not_depend_on_where_the_tree_lives(tmp_path: Path) -> None:
    spec = default_spec()
    first = geometry.build_output_files(run_mini(tmp_path / "a", copy.deepcopy(spec)))
    second = geometry.build_output_files(run_mini(tmp_path / "b", copy.deepcopy(spec)))
    for name in (geometry.OWNER_ROWS_NAME, geometry.PAIR_ROWS_NAME, geometry.VISUAL_PLAN_NAME):
        assert first[name] == second[name], name

    # The summary seals absolute input paths, which legitimately differ per tree.
    def strip_paths(payload: bytes) -> dict[str, Any]:
        summary = json.loads(payload.decode("utf-8"))
        summary.pop("binding", None)
        return summary

    assert strip_paths(first[geometry.SUMMARY_NAME]) == strip_paths(
        second[geometry.SUMMARY_NAME]
    )


def test_receipt_is_self_sealed_and_names_every_output(tmp_path: Path) -> None:
    files = geometry.build_output_files(run_mini(tmp_path, default_spec()))
    receipt = json.loads(files[geometry.RECEIPT_NAME].decode("utf-8"))
    geometry.assert_self_sealed(
        receipt, digest_key="receipt_content_sha256", label="analysis receipt"
    )
    digests = receipt["output_file_digests"]
    for name in (
        geometry.OWNER_ROWS_NAME,
        geometry.PAIR_ROWS_NAME,
        geometry.SUMMARY_NAME,
        geometry.REPORT_MD_NAME,
        geometry.VISUAL_PLAN_NAME,
    ):
        assert digests[name]["sha256"] == geometry.sha256_bytes(files[name])
    assert receipt["policy"]["compute_scope"] == "cpu_only_no_model_no_gpu_no_rescoring"
    assert receipt["policy"]["cutoffs"]["material_negative_max_nats"] == -1.0


def test_visual_plan_names_a_panel_for_every_material_negative_case(tmp_path: Path) -> None:
    result = run_mini(tmp_path, default_spec())
    plan = result["visual_plan"]
    material = [row for row in result["owner_rows"] if row["material_negative"]]
    crop_ids = {panel["panel_id"] for panel in plan["crop_panels"]}
    for row in material:
        assert f"crop:{row['arm']}:{row['gt_owner_id']}" in crop_ids
    assert plan["material_negative_case_count"] == len(material)
    assert plan["greedy_pair_panel_count"] == len(result["pair_rows"])
    assert len(plan["expected_output_files"]) == len(set(plan["expected_output_files"]))
    for panel in plan["crop_panels"]:
        window = panel["crop_window_norm1000_xyxy"]
        assert 0 <= window["x1"] < window["x2"] <= geometry.CANVAS_BINS
        assert 0 <= window["y1"] < window["y2"] <= geometry.CANVAS_BINS
        assert len({box["color"] for box in panel["boxes"]}) == 2


def test_summary_carries_no_cross_image_pooled_raw_delta(tmp_path: Path) -> None:
    summary = run_mini(tmp_path, default_spec())["summary"]
    geometry.assert_no_cross_image_raw_delta_pooling(summary, label="summary")
    assert summary["cross_image_pooling"] == geometry.NO_CROSS_IMAGE_POOLING
    for block in summary["per_image"].values():
        for arm_block in block["arms"].values():
            assert "within_image_median_relative_coordinate_delta" in arm_block


def test_reserved_panel_image_is_carried_forward_but_never_counted(tmp_path: Path) -> None:
    summary = run_mini(tmp_path, default_spec())["summary"]
    reserved = summary["reserved_next_panel"]
    assert reserved["image_id"] == "2299"
    assert reserved["enters_this_denominator"] is False
    assert reserved["physical_owner_annotation_count"] == 46
    assert summary["denominators"]["image_count"] == 2


# ---------------------------------------------------------------------------
# The real frozen artifacts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_result() -> dict[str, Any]:
    return geometry.run_analysis(
        plan_dir=REAL_PLAN_DIR,
        primary_owner_rows=REAL_PRIMARY_ROWS,
        secondary_owner_rows=REAL_SECONDARY_ROWS,
        secondary_merged_dir=REAL_MERGED_DIR,
    )


@real_artifacts
def test_real_denominators_are_the_frozen_twenty_six_and_twelve(real_result: Mapping[str, Any]) -> None:
    denominators = real_result["summary"]["denominators"]
    assert denominators["crossing_owner_count"] == 26
    assert denominators["primary_ce_row_count"] == 26
    assert denominators["sensitivity_cf_row_count"] == 26
    assert denominators["greedy_displacement_pair_count"] == 12
    assert denominators["benign_control_count"] == 12
    assert denominators["image_count"] == 12
    assert "2299" not in real_result["summary"]["per_image"]


@real_artifacts
def test_real_secondary_mapping_is_sixty_four_to_thirty_eight(real_result: Mapping[str, Any]) -> None:
    mapping = real_result["summary"]["row_mapping"]
    assert mapping["input_merge_row_count"] == 64
    assert mapping["secondary_owner_row_count"] == 38
    assert mapping["crossing_owner_count"] == 26
    assert mapping["crossing_readouts_per_owner"] == 2
    assert mapping["benign_control_count"] == 12
    assert 26 * 2 + 12 == mapping["input_merge_row_count"]
    assert 26 + 12 == mapping["secondary_owner_row_count"]


@real_artifacts
def test_real_rows_bind_each_arm_to_its_own_sealed_row(real_result: Mapping[str, Any]) -> None:
    cohort = {
        json.loads(line)["gt_owner_id"]: json.loads(line)
        for line in (REAL_PLAN_DIR / geometry.COHORT_REGISTRY_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    for row in real_result["owner_rows"]:
        row_key = "e_row" if row["arm"] == geometry.PRIMARY_ARM else "f_row"
        sealed = cohort[row["gt_owner_id"]][row_key]
        expected = [token - geometry.COORD_TOKEN_START for token in sealed["coord_token_ids"]]
        assert row["geometry"]["r_box_norm1000_xyxy"] == expected
        assert row["r_strict_match_status"] == sealed["strict_match_status"]


@real_artifacts
def test_real_primary_decision_uses_one_vote_per_owner(real_result: Mapping[str, Any]) -> None:
    decision = real_result["summary"]["decision"]
    primary = decision["primary_arm"]
    assert primary["arm"] == geometry.PRIMARY_ARM
    assert primary["row_count"] == primary["distinct_owner_count"] == 26
    assert decision["outcome"] == primary["outcome"]
    assert decision["outcome"] in geometry.DECISION_EXHAUSTIVE_ORDER
    assert decision["sensitivity_arm"]["arm"] == geometry.SENSITIVITY_ARM
    assert decision["pseudo_replication_guard"]["one_owner_one_vote"] is True


@real_artifacts
def test_real_material_negative_rows_are_reproducible_from_the_published_rows(
    real_result: Mapping[str, Any],
) -> None:
    recomputed = [
        row["gt_owner_id"]
        for row in real_result["owner_rows"]
        if row["arm"] == geometry.PRIMARY_ARM
        and row["likelihood"]["relative"]["coordinates"]["crossing_delta"]
        - row["likelihood"]["relative"]["coordinates"]["same_image_benign_delta"]
        <= geometry.MATERIAL_NEGATIVE_MAX_NATS
    ]
    published = real_result["summary"]["decision"]["primary_arm"][
        "material_negative_gt_owner_ids"
    ]
    assert sorted(recomputed) == sorted(published)


@real_artifacts
def test_real_run_is_byte_deterministic(real_result: Mapping[str, Any]) -> None:
    first = geometry.build_output_files(real_result)
    second = geometry.build_output_files(
        geometry.run_analysis(
            plan_dir=REAL_PLAN_DIR,
            primary_owner_rows=REAL_PRIMARY_ROWS,
            secondary_owner_rows=REAL_SECONDARY_ROWS,
            secondary_merged_dir=REAL_MERGED_DIR,
        )
    )
    assert first == second


@real_artifacts
def test_real_summary_never_pools_raw_deltas_across_images(
    real_result: Mapping[str, Any],
) -> None:
    summary = real_result["summary"]
    geometry.assert_no_cross_image_raw_delta_pooling(summary, label="real summary")
    assert set(summary["per_image"]) == {
        row["image_id"] for row in real_result["owner_rows"]
    }
