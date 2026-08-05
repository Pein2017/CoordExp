"""Focused tests for the sorted owner accessibility census visual atlas.

Contract item 14 makes JSONL artifacts and receipts the authority, so these
tests concentrate on provenance: every figure must be traceable to artifact
rows, must not recompute a score, must use within-context normalized
confidence, and must never render the ``x1`` distribution as a 2-D heatmap.

A small synthetic plan is committed to ``tmp_path`` so the tests are fast and
independent of the frozen inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_owner_accessibility_census_plan as planner  # noqa: E402
from scripts.research import (  # noqa: E402
    visualize_sorted_owner_accessibility_visual_atlas as atlas,
)

IMAGE_ID = "7511"
CONTEXT_ID = f"{IMAGE_ID}:boundary-000"
GROUP_ID = f"{CONTEXT_ID}|person"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(b"".join(planner.canonical_json_bytes(row) + b"\n" for row in rows))


@pytest.fixture()
def plan_dir(tmp_path: Path) -> Path:
    """A minimal but structurally faithful sealed plan."""

    directory = tmp_path / "plan"
    directory.mkdir()

    images = [
        {
            "image_id": IMAGE_ID,
            "split": "discovery",
            "image_width": 400,
            "image_height": 300,
            "file_name": "images/val2017/missing.jpg",
            "prompt_token_ids": [1, 2, 3],
        }
    ]
    owners = []
    for index, box in enumerate([[10, 10, 110, 210], [200, 20, 300, 220]]):
        owners.append(
            {
                "gt_owner_id": f"gt:{IMAGE_ID}:{index}",
                "image_id": IMAGE_ID,
                "split": "discovery",
                "normalized_description": "person",
                "bbox_pixel_xyxy": box,
                "owner_sort_key": [box[1], box[0]],
                "greedy_eligible": True,
                "native_true_positive": index == 0,
                "calibration_role": (
                    "native_true_positive_calibration" if index == 0 else "native_false_negative"
                ),
                "candidate_bank": {
                    "logical_role_count": 17,
                    "distinct_physical_candidate_count": 17,
                    "other_owner_strict_count": 0,
                    "bank_coverage_status": "full",
                    "disposition_eligible": True,
                    "strict_assignment_coverage": {"uniquely_assigned_candidate_count": 11},
                },
            }
        )
    candidates = []
    for index, box in enumerate([[10, 10, 110, 210], [20, 20, 120, 220], [200, 20, 300, 220]]):
        candidates.append(
            {
                "candidate_id": f"cand:{index:04d}",
                "image_id": IMAGE_ID,
                "normalized_description": "person",
                "decoded_bbox_pixel_xyxy": box,
                "representative_role": "exact_gt_anchor" if index == 0 else "translate_right",
                "candidate_class": "core",
                "candidate_provenance": "exact" if index == 0 else "neighborhood",
                "generator_gt_owner_ids": [f"gt:{IMAGE_ID}:{min(index, 1)}"],
                "cross_owner_generated": False,
                "strict_assignment_status": "matched",
                "strict_assignment_gt_owner_id": f"gt:{IMAGE_ID}:{min(index, 1)}",
            }
        )
    contexts = [
        {
            "context_id": CONTEXT_ID,
            "image_id": IMAGE_ID,
            "split": "discovery",
            "context_role": "root",
            "boundary_index": 0,
            "loop_marking": {
                "loop_tail": False,
                "prior_identical_row_count": None,
                "consecutive_identical_row_run_length": None,
                "repeated_raw_span_sha256": None,
                "flag_is_not_a_mechanism_label": True,
            },
        }
    ]
    categories = [
        {
            "category_query_id": f"{IMAGE_ID}:person",
            "image_id": IMAGE_ID,
            "normalized_description": "person",
            "status": "admitted",
        }
    ]
    query_groups = [
        {
            "query_group_id": GROUP_ID,
            "image_id": IMAGE_ID,
            "context_id": CONTEXT_ID,
            "normalized_description": "person",
            "status": "admitted",
            "candidate_ids": [row["candidate_id"] for row in candidates],
        }
    ]

    _write_jsonl(directory / "image-registry.jsonl", images)
    _write_jsonl(directory / "owner-registry.jsonl", owners)
    _write_jsonl(directory / "category-registry.jsonl", categories)
    _write_jsonl(directory / "context-registry.jsonl", contexts)
    _write_jsonl(directory / "candidate-bank.jsonl", candidates)
    _write_jsonl(directory / "query-group-registry.jsonl", query_groups)

    receipt = {"unit_id": atlas.UNIT_ID, "schema_version": planner.PLAN_SCHEMA_VERSION}
    receipt["receipt_content_sha256"] = planner.sha256_json(receipt)
    (directory / "receipt.json").write_bytes(planner.canonical_json_bytes(receipt) + b"\n")
    return directory


def _score_rows() -> list[dict]:
    """Localization score rows in the scorer's real schema.

    Note there is **no** scalar ``gt_owner_id``: attribution is by membership
    in ``generator_gt_owner_ids``.  ``cand:0002`` is deliberately generated by
    both owners, so cross-owner attribution is exercised.
    """

    generators = [
        [f"gt:{IMAGE_ID}:0"],
        [f"gt:{IMAGE_ID}:1"],
        [f"gt:{IMAGE_ID}:0", f"gt:{IMAGE_ID}:1"],
    ]
    rows = []
    for index, (logprob, owner_ids) in enumerate(zip([-2.0, -3.5, -9.0], generators, strict=True)):
        rows.append(
            {
                "schema_version": "sorted-owner-accessibility-census-score.v1",
                "row_kind": "census_localization_score",
                "request_id": f"{GROUP_ID}|cand:{index:04d}",
                "query_group_id": GROUP_ID,
                "image_id": IMAGE_ID,
                "context_id": CONTEXT_ID,
                "normalized_description": "person",
                "candidate_id": f"cand:{index:04d}",
                "generator_gt_owner_ids": owner_ids,
                "generator_owner_count": len(owner_ids),
                "cross_owner_generated": len(owner_ids) > 1,
                "generator_provenance_role": "provenance_only_never_rank_or_assignment",
                "complete_box_logprob_sum": logprob,
                "rank_key": {
                    "image_id": IMAGE_ID,
                    "context_id": CONTEXT_ID,
                    "normalized_description": "person",
                },
                "competition": {
                    "population": "collapsed_unique_physical_candidates_only",
                    "population_size": 3,
                    "rank": index + 1,
                    "margin_to_group_best": logprob - (-2.0),
                    "group_best_candidate_id": "cand:0000",
                },
                "is_sidecar": False,
            }
        )
    return rows


def _proposal_rows() -> list[dict]:
    """Proposal-surface rows in the scorer's real schema (no free next row)."""

    return [
        {
            "schema_version": "sorted-owner-accessibility-census-proposal.v1",
            "row_kind": "census_proposal_surface",
            "context_id": CONTEXT_ID,
            "image_id": IMAGE_ID,
            "split": "discovery",
            "boundary_gate": {
                "continue_probability": 0.9,
                "stop_probability": 0.05,
                # The scorer emits a logprob margin, not a logit margin.
                "continue_vs_stop_logprob_margin": 2.9,
                "semantics": "gate_only_never_description_accessibility",
            },
            "category_routing_event": [
                {
                    "normalized_description": "person",
                    "within_context_rank": 1,
                    "raw_sequence_logprob_sum": -0.4,
                    "aggregation": "raw_sequence_sum_no_token_mean",
                }
            ],
        }
    ]


def _free_decode_rows() -> list[dict]:
    """Both sidecar kinds, exactly as the scorer publishes them in one file."""

    return [
        {
            "schema_version": "sorted-owner-accessibility-census-free-decode.v1",
            "row_kind": "census_free_greedy_box_sidecar",
            "sidecar_id": f"free-box:{GROUP_ID}",
            "image_id": IMAGE_ID,
            "context_id": CONTEXT_ID,
            "normalized_description": "person",
            "coord_token_ids": [151670, 151671, 151672, 151673],
            "is_sidecar": True,
            "enters_core_ranks": False,
        },
        {
            "schema_version": "sorted-owner-accessibility-census-free-decode.v1",
            "row_kind": "census_free_next_row_sidecar",
            "sidecar_id": f"free-row:{CONTEXT_ID}",
            "image_id": IMAGE_ID,
            "context_id": CONTEXT_ID,
            "decode_mode": "greedy_explicit_position_cache",
            "uses_model_generate": False,
            "max_tokens": 32,
            "token_ids": [151646, 8987, 151647],
            "token_count": 3,
            "per_token_logprobs": [-0.1, -0.2, -0.3],
            "stop_reason": "token_151649",
            "truncated_at_cap": False,
            "reached_im_end": False,
            "reached_box_end": True,
            "is_sidecar": True,
            "enters_core_ranks": False,
            "is_behavior_not_probability": True,
        },
    ]


@pytest.fixture()
def shard_root(tmp_path: Path) -> Path:
    root = tmp_path / "shards"
    shard = root / IMAGE_ID
    shard.mkdir(parents=True)
    _write_jsonl(shard / atlas.SHARD_SCORES_NAME, _score_rows())
    _write_jsonl(shard / atlas.SHARD_PROPOSAL_NAME, _proposal_rows())
    _write_jsonl(shard / atlas.SHARD_FREE_DECODE_NAME, _free_decode_rows())
    return root


@pytest.fixture()
def artifacts(plan_dir: Path, shard_root: Path) -> atlas.Artifacts:
    return atlas.load_artifacts(plan_dir, shard_root=shard_root)


# ---------------------------------------------------------------------------
# Provenance and authority
# ---------------------------------------------------------------------------


def test_load_rejects_a_plan_whose_receipt_does_not_reconstruct(plan_dir: Path) -> None:
    receipt = json.loads((plan_dir / "receipt.json").read_text(encoding="utf-8"))
    receipt["unit_id"] = atlas.UNIT_ID
    receipt["receipt_content_sha256"] = "0" * 64
    (plan_dir / "receipt.json").write_bytes(planner.canonical_json_bytes(receipt) + b"\n")
    with pytest.raises(atlas.VisualContractError, match="reconstruct"):
        atlas.load_artifacts(plan_dir)


def test_load_rejects_a_foreign_unit_plan(plan_dir: Path) -> None:
    receipt = {"unit_id": "some-other-unit"}
    receipt["receipt_content_sha256"] = planner.sha256_json(receipt)
    (plan_dir / "receipt.json").write_bytes(planner.canonical_json_bytes(receipt) + b"\n")
    with pytest.raises(atlas.VisualContractError, match="unit_id"):
        atlas.load_artifacts(plan_dir)


def test_quarantined_shards_contribute_no_evidence(plan_dir: Path, shard_root: Path) -> None:
    (shard_root / IMAGE_ID / "shard-quarantine.json").write_text("{}", encoding="utf-8")
    loaded = atlas.load_artifacts(plan_dir, shard_root=shard_root)
    assert loaded.scores == []
    assert loaded.proposals == []


def test_every_figure_binds_artifact_files_and_row_ids(artifacts) -> None:
    specs = atlas.build_atlas_specs(artifacts)
    assert specs
    for spec in specs:
        provenance = spec["provenance"]
        assert provenance["artifact_files"], spec["figure_id"]
        assert "row_ids" in provenance


def test_manifest_declares_no_model_and_no_recomputation(artifacts) -> None:
    specs = atlas.build_atlas_specs(artifacts)
    manifest = atlas.build_manifest(artifacts, specs, rendered={})
    assert manifest["authority"] == "jsonl_artifacts_and_receipts"
    assert manifest["reads_any_model"] is False
    assert manifest["recomputes_any_score"] is False
    assert manifest["manifest_content_sha256"]
    assert manifest["figure_count"] == len(specs)


def test_manifest_rejects_a_figure_without_provenance(artifacts) -> None:
    spec = atlas.build_owner_map_spec(artifacts, IMAGE_ID)
    spec["provenance"]["artifact_files"] = []
    with pytest.raises(atlas.VisualContractError, match="provenance"):
        atlas.build_manifest(artifacts, [spec], rendered={})


def test_atlas_provides_all_five_required_products(artifacts) -> None:
    specs = atlas.build_atlas_specs(artifacts)
    assert {str(spec["product"]) for spec in specs} == set(atlas.PRODUCTS)


# ---------------------------------------------------------------------------
# Presentation rules
# ---------------------------------------------------------------------------


def test_within_context_confidence_is_a_normalized_softmax() -> None:
    confidence = atlas.within_context_confidence({"a": -1.0, "b": -2.0, "c": -10.0})
    assert pytest.approx(sum(confidence.values()), abs=1e-9) == 1.0
    assert confidence["a"] > confidence["b"] > confidence["c"]


def test_within_context_confidence_is_shift_invariant() -> None:
    """Adding a constant to one group's logprobs must not change the figure."""

    base = atlas.within_context_confidence({"a": -1.0, "b": -2.0})
    shifted = atlas.within_context_confidence({"a": -101.0, "b": -102.0})
    for key in base:
        assert pytest.approx(base[key], abs=1e-9) == shifted[key]


def test_short_ids_are_deterministic_and_short() -> None:
    mapping = atlas.assign_short_ids(["gt:7511:2", "gt:7511:17"], prefix="O")
    assert mapping == {"gt:7511:2": "O0", "gt:7511:17": "O1"}
    assert all(len(value) <= 4 for value in mapping.values())


def test_owner_map_labels_boxes_with_short_ids_and_a_side_legend(artifacts) -> None:
    spec = atlas.build_owner_map_spec(artifacts, IMAGE_ID)
    assert spec["label_policy"] == "short_ids_on_boxes_full_ids_in_side_legend"
    assert len(spec["legend"]) == len(spec["owners"])
    for entry, legend in zip(spec["owners"], spec["legend"], strict=True):
        assert entry["short_id"] == legend["short_id"]
        assert legend["full_id"].startswith("gt:")


def test_owner_map_is_sorted_by_owner_sort_key(artifacts) -> None:
    spec = atlas.build_owner_map_spec(artifacts, IMAGE_ID)
    keys = [entry["owner_sort_key"] for entry in spec["owners"]]
    assert keys == sorted(keys)


def test_localization_landscape_uses_within_context_confidence(artifacts) -> None:
    spec = atlas.build_localization_landscape_spec(artifacts, GROUP_ID)
    assert spec["confidence_normalization"] == (
        "within_context_softmax_over_one_query_group"
    )
    assert spec["raw_logprobs_drawn"] is False
    assert spec["cross_image_comparison"] is False
    total = sum(entry["within_context_confidence"] for entry in spec["candidates"])
    assert pytest.approx(total, abs=1e-9) == 1.0


def test_localization_landscape_never_renders_the_x1_distribution(artifacts) -> None:
    spec = atlas.build_localization_landscape_spec(artifacts, GROUP_ID)
    assert spec["x1_distribution_rendered"] is False
    assert spec["x1_distribution_policy"] == "diagnostic_only_never_a_2d_heatmap"
    assert "x1_logprobs" not in json.dumps(spec)


def test_landscape_crop_is_fixed_padding_and_keeps_the_neighbourhood(artifacts) -> None:
    spec = atlas.build_localization_landscape_spec(artifacts, GROUP_ID)
    crop = spec["crop"]
    assert crop["padding_pixels"] == atlas.CROP_PADDING_PIXELS
    assert crop["padding_policy"] == "fixed_never_adaptive"
    assert crop["includes_competition_neighbourhood"] is True
    assert crop["presentation_only"] is True
    assert crop["model_input"] == "original_full_image_never_this_crop"
    # The window must cover every drawn candidate box.
    window = crop["window_pixel_xyxy"]
    for entry in spec["candidates"]:
        box = entry["bbox_pixel_xyxy"]
        assert window[0] <= box[0] and window[1] <= box[1]
        assert window[2] >= box[2] and window[3] >= box[3]


def test_fixed_padding_crop_is_clamped_to_the_canvas() -> None:
    crop = atlas.fixed_padding_crop([[10, 10, 20, 20]], width=100, height=80, padding=50)
    assert crop["window_pixel_xyxy"] == [0, 0, 70, 70]


def test_crop_window_does_not_shrink_when_boxes_are_close() -> None:
    """Fixed padding, not adaptive: crowding must not change the padding."""

    tight = atlas.fixed_padding_crop([[100, 100, 110, 110]], width=1000, height=1000)
    loose = atlas.fixed_padding_crop([[100, 100, 400, 400]], width=1000, height=1000)
    assert tight["padding_pixels"] == loose["padding_pixels"] == atlas.CROP_PADDING_PIXELS


# ---------------------------------------------------------------------------
# Estimand semantics in figures
# ---------------------------------------------------------------------------


def test_proposal_map_is_per_context_and_never_per_owner(artifacts) -> None:
    spec = atlas.build_proposal_map_spec(artifacts, CONTEXT_ID)
    assert spec["emits_per_owner_proposal_probability"] is False
    assert spec["includes_coordinate_scores"] is False
    assert "gt_owner_id" not in json.dumps(spec["category_routing"])
    assert spec["boundary_gate"]["semantics"] == (
        "gate_only_never_description_accessibility"
    )


def test_boundary_gate_uses_the_logprob_margin_key(artifacts) -> None:
    """The scorer emits a logprob margin; a logit margin key never existed."""

    gate = atlas.build_proposal_map_spec(artifacts, CONTEXT_ID)["boundary_gate"]
    assert gate["continue_vs_stop_logprob_margin"] == 2.9
    assert "continue_vs_stop_logit_margin" not in gate
    assert gate["margin_units"] == "natural_log_probability_difference"
    assert gate["margin_semantics"] == "logprob(object_ref_start) - logprob(im_end)"


def test_proposal_map_joins_the_free_next_row_sidecar(artifacts) -> None:
    spec = atlas.build_proposal_map_spec(artifacts, CONTEXT_ID)
    assert spec["free_next_row_available"] is True
    free_row = spec["free_next_row"]
    assert free_row["row_kind"] == atlas.FREE_NEXT_ROW_KIND
    assert free_row["sidecar_id"] == f"free-row:{CONTEXT_ID}"
    assert free_row["token_ids"] == [151646, 8987, 151647]
    assert free_row["reached_box_end"] is True
    assert free_row["is_behavior_not_probability"] is True
    # The old phantom key must not reappear.
    assert "free_decoded_next_row" not in spec
    # Provenance must cite the sidecar file it was joined from.
    assert any(
        path.endswith(atlas.SHARD_FREE_DECODE_NAME)
        for path in spec["provenance"]["artifact_files"]
    )
    assert f"free-row:{CONTEXT_ID}" in spec["provenance"]["row_ids"]


def test_free_next_row_is_omitted_not_fabricated_when_absent(
    plan_dir: Path, tmp_path: Path
) -> None:
    root = tmp_path / "shards_no_free"
    shard = root / IMAGE_ID
    shard.mkdir(parents=True)
    _write_jsonl(shard / atlas.SHARD_SCORES_NAME, _score_rows())
    _write_jsonl(shard / atlas.SHARD_PROPOSAL_NAME, _proposal_rows())
    loaded = atlas.load_artifacts(plan_dir, shard_root=root)
    spec = atlas.build_proposal_map_spec(loaded, CONTEXT_ID)
    assert spec["free_next_row"] is None
    assert spec["free_next_row_available"] is False
    assert not any(
        path.endswith(atlas.SHARD_FREE_DECODE_NAME)
        for path in spec["provenance"]["artifact_files"]
    )


def test_free_greedy_box_sidecars_are_not_mistaken_for_next_rows(artifacts) -> None:
    """Both kinds share one file; only the next-row kind is a proposal input."""

    assert len(artifacts.free_next_rows) == 1
    assert all(
        str(row["row_kind"]) == atlas.FREE_NEXT_ROW_KIND
        for row in artifacts.free_next_rows
    )


def test_free_next_row_join_is_keyed_by_image_and_context(artifacts) -> None:
    assert artifacts.free_next_row(IMAGE_ID, CONTEXT_ID) is not None
    assert artifacts.free_next_row(IMAGE_ID, "7511:boundary-999") is None
    assert artifacts.free_next_row("9999", CONTEXT_ID) is None


def test_proposal_map_reports_raw_sequence_sums_not_token_means(artifacts) -> None:
    spec = atlas.build_proposal_map_spec(artifacts, CONTEXT_ID)
    for entry in spec["category_routing"]:
        assert entry["aggregation"] == "raw_sequence_sum_no_token_mean"


def test_proposal_map_requires_captured_evidence(artifacts) -> None:
    with pytest.raises(atlas.VisualContractError, match="no captured proposal-surface"):
        atlas.build_proposal_map_spec(artifacts, "7511:boundary-999")


def test_localization_landscape_requires_captured_scores(artifacts) -> None:
    with pytest.raises(atlas.VisualContractError, match="no captured score rows"):
        atlas.build_localization_landscape_spec(artifacts, "7511:boundary-000|chair")


def test_owner_card_attributes_scores_by_generator_membership(artifacts) -> None:
    """Score rows carry a generator list, never a scalar gt_owner_id."""

    owner0 = atlas.build_owner_card_spec(artifacts, f"gt:{IMAGE_ID}:0")
    owner1 = atlas.build_owner_card_spec(artifacts, f"gt:{IMAGE_ID}:1")
    # cand:0000 (owner 0) + cand:0002 (shared) -> 2 rows for owner 0.
    assert owner0["captured_score_row_count"] == 2
    # cand:0001 (owner 1) + cand:0002 (shared) -> 2 rows for owner 1.
    assert owner1["captured_score_row_count"] == 2
    for spec in (owner0, owner1):
        attribution = spec["score_attribution"]
        assert attribution["rule"] == "exact_membership_in_generator_gt_owner_ids"
        assert attribution["representative_generator_shortcut"] is False


def test_cross_owner_candidate_is_visible_on_both_owner_cards(artifacts) -> None:
    for owner_id in (f"gt:{IMAGE_ID}:0", f"gt:{IMAGE_ID}:1"):
        attribution = atlas.build_owner_card_spec(artifacts, owner_id)["score_attribution"]
        assert attribution["cross_owner_shared_score_row_count"] == 1
        assert attribution["cross_owner_shared_candidate_ids"] == ["cand:0002"]


def test_score_attribution_helpers_reject_a_scalar_owner_field() -> None:
    """A row carrying only the phantom scalar attributes to nobody."""

    phantom = {"gt_owner_id": "gt:7511:0", "candidate_id": "cand:0000"}
    assert atlas._generator_owner_ids(phantom) == []
    assert atlas._score_row_generated_by(phantom, "gt:7511:0") is False

    real = {"generator_gt_owner_ids": ["gt:7511:0", "gt:7511:1"]}
    assert atlas._score_row_generated_by(real, "gt:7511:0") is True
    assert atlas._score_row_generated_by(real, "gt:7511:1") is True
    assert atlas._score_row_generated_by(real, "gt:7511:2") is False


def test_owner_card_reports_both_bank_views(artifacts) -> None:
    spec = atlas.build_owner_card_spec(artifacts, f"gt:{IMAGE_ID}:0")
    bank = spec["bank"]
    assert bank["distinct_physical_candidate_count"] == 17
    assert bank["uniquely_assigned_candidate_count"] == 11
    assert bank["other_owner_strict_count"] == 0
    assert bank["disposition_eligible"] is True


def _bound(name: str, *, usable: bool, peak: float, concentration: float) -> dict:
    projected = {
        "context_id": CONTEXT_ID,
        "loop_tail": False,
        "peak_lift": peak,
        "local_concentration": concentration,
        "rank_within_group": 1,
        "owner_rank_within_group": 1,
        "owner_population_size": 2,
        "margin_to_best_owner_in_group": 0.0,
        "margin_semantics": "within_same_context_and_category_owner_competition",
        "exact_anchor_score": -2.0,
    }
    return {
        "bound": name,
        "usable_support": usable,
        "support_calibrated": True,
        "support_criterion_id": "local_peak_tp_calibrated_q10.v1",
        "never_frontier_tested": False,
        "loop_tail_only_support": False,
        "tested_context_count": 4,
        "non_loop_tested_context_count": 4,
        "primary_best_non_loop": projected,
        "primary_first_non_loop_minimal_abs_frontier": projected,
        "diagnostic_best_all": {**projected, "loop_tail": True},
        "diagnostic_best_all_role": "diagnostic_only_includes_loop_tail_contexts",
    }


@pytest.fixture()
def merged_dir(tmp_path: Path) -> Path:
    merged = tmp_path / "merged"
    merged.mkdir()
    _write_jsonl(
        merged / atlas.OWNER_SUMMARY_NAME,
        [
            {
                "gt_owner_id": f"gt:{IMAGE_ID}:0",
                "image_id": IMAGE_ID,
                "split": "discovery",
                "normalized_description": "person",
                "lower_bound_l": _bound("L", usable=True, peak=0.9, concentration=0.7),
                "upper_bound_u": _bound("U", usable=True, peak=1.1, concentration=0.8),
                "loop_tail_only_support": False,
                "never_frontier_tested": False,
                "frontier_tested": True,
                "has_non_loop_primary_context": True,
                "tested_context_count": 4,
                "non_loop_tested_context_count": 4,
                "exact_anchor_score_reported": True,
                "ambiguity_bound_disposition_flip": False,
                "disposition": "resolved_support",
                "disposition_blockers": [],
                "routing_summary": {
                    "population": "non_loop_contexts",
                    "ever_rank1": True,
                    "best_rank": 1,
                    "best_margin_to_best_owner": 0.0,
                    "role": "routing_and_competition_surface_never_a_support_input",
                },
                "threshold_category_support": {"flag": "pooled_underrepresented"},
                "bank_adequacy": {"status": "full"},
                "bank_report": {"distinct_physical_candidate_count": 17},
            }
        ],
    )
    return merged


def test_owner_card_copies_merged_summaries_without_recomputing(
    plan_dir: Path, shard_root: Path, merged_dir: Path
) -> None:
    loaded = atlas.load_artifacts(plan_dir, shard_root=shard_root, merged_dir=merged_dir)
    spec = atlas.build_owner_card_spec(loaded, f"gt:{IMAGE_ID}:0")
    record = spec["owner_record"]
    assert spec["recomputed_any_statistic"] is False
    assert spec["applied_any_threshold"] is False
    assert spec["support_source"] == "merge_sealed_discovery_calibration"
    assert record["tested_context_count"] == 4
    assert record["non_loop_tested_context_count"] == 4
    assert record["disposition"] == "resolved_support"
    assert record["threshold_category_support"]["flag"] == "pooled_underrepresented"


def test_owner_card_presents_peak_lift_and_local_concentration(
    plan_dir: Path, shard_root: Path, merged_dir: Path
) -> None:
    loaded = atlas.load_artifacts(plan_dir, shard_root=shard_root, merged_dir=merged_dir)
    support = atlas.build_owner_card_spec(loaded, f"gt:{IMAGE_ID}:0")["calibrated_support"]
    assert support["statistics"] == ["peak_lift", "local_concentration"]
    for bound_key, peak in (("lower_bound_l", 0.9), ("upper_bound_u", 1.1)):
        primary = support[bound_key]["primary_best_non_loop"]
        assert primary["peak_lift"] == peak
        assert "local_concentration" in primary
    assert support["disposition"] == "resolved_support"
    assert support["ambiguity_bound_disposition_flip"] is False


def test_owner_card_never_applies_a_threshold(
    plan_dir: Path, shard_root: Path, merged_dir: Path
) -> None:
    loaded = atlas.load_artifacts(plan_dir, shard_root=shard_root, merged_dir=merged_dir)
    support = atlas.build_owner_card_spec(loaded, f"gt:{IMAGE_ID}:0")["calibrated_support"]
    assert support["thresholds_applied_here"] is False
    assert support["thresholds_owner"] == "merge_sealed_discovery_calibration_receipt"
    assert support["rank_is_support_criterion"] is False
    # The calibrated status is copied, not derived.
    assert support["upper_bound_u"]["usable_support"] is True
    assert support["upper_bound_u"]["support_criterion_id"] == (
        "local_peak_tp_calibrated_q10.v1"
    )


def test_owner_card_keeps_routing_rank_separate_from_support(
    plan_dir: Path, shard_root: Path, merged_dir: Path
) -> None:
    loaded = atlas.load_artifacts(plan_dir, shard_root=shard_root, merged_dir=merged_dir)
    spec = atlas.build_owner_card_spec(loaded, f"gt:{IMAGE_ID}:0")
    routing = spec["routing_surface"]
    assert routing["role"] == "routing_and_competition_surface_never_a_support_input"
    assert routing["best_rank"] == 1
    # Rank and margin live in the routing block, never in calibrated support.
    support = json.dumps(spec["calibrated_support"])
    assert "ever_rank1" not in support
    primary = spec["calibrated_support"]["upper_bound_u"]["primary_best_non_loop"]
    assert set(primary["routing"]) == set(atlas.ROUTING_FIELDS)


def test_owner_card_marks_the_all_context_best_as_diagnostic(
    plan_dir: Path, shard_root: Path, merged_dir: Path
) -> None:
    loaded = atlas.load_artifacts(plan_dir, shard_root=shard_root, merged_dir=merged_dir)
    support = atlas.build_owner_card_spec(loaded, f"gt:{IMAGE_ID}:0")["calibrated_support"]
    bound = support["upper_bound_u"]
    assert bound["diagnostic_best_all_role"] == (
        "diagnostic_only_includes_loop_tail_contexts"
    )
    assert bound["diagnostic_best_all"]["loop_tail"] is True


def test_owner_card_without_merged_records_still_declares_no_recomputation(
    artifacts,
) -> None:
    spec = atlas.build_owner_card_spec(artifacts, f"gt:{IMAGE_ID}:0")
    assert "owner_record" not in spec
    assert spec["recomputed_any_statistic"] is False
    assert spec["applied_any_threshold"] is False


def test_feature_overview_reports_support_statistic_distributions(
    plan_dir: Path, shard_root: Path, merged_dir: Path
) -> None:
    loaded = atlas.load_artifacts(plan_dir, shard_root=shard_root, merged_dir=merged_dir)
    spec = atlas.build_feature_overview_spec(loaded)
    assert spec["support_statistics"] == ["peak_lift", "local_concentration"]
    assert spec["thresholds_applied_here"] is False
    assert spec["recomputed_any_support"] is False
    discovery = spec["per_split"]["discovery"]
    distributions = discovery["support_statistic_distributions"]
    assert distributions["peak_lift"]["count"] == 1
    assert distributions["peak_lift"]["median"] == 1.1
    assert distributions["local_concentration"]["count"] == 1
    assert discovery["disposition_counts"] == {"resolved_support": 1}
    assert discovery["underrepresented_categories"] == ["person"]


def test_manifest_declares_support_thresholds_are_not_owned_here(
    plan_dir: Path, shard_root: Path, merged_dir: Path
) -> None:
    loaded = atlas.load_artifacts(plan_dir, shard_root=shard_root, merged_dir=merged_dir)
    specs = atlas.build_atlas_specs(loaded)
    manifest = atlas.build_manifest(loaded, specs, rendered={})
    assert manifest["recomputes_any_support_or_threshold"] is False
    assert manifest["support_statistics"] == ["peak_lift", "local_concentration"]
    assert manifest["support_thresholds_owner"] == (
        "merge_sealed_discovery_calibration_receipt"
    )
    assert manifest["rank_presented_as"] == (
        "routing_and_competition_surface_never_support"
    )


def test_feature_overview_keeps_the_splits_separate(artifacts) -> None:
    spec = atlas.build_feature_overview_spec(artifacts)
    assert spec["splits_pooled"] is False
    assert spec["loop_tail_pooled_with_ordinary_contexts"] is False
    assert spec["continuous_features_first"] is True
    assert set(spec["per_split"]) == {"discovery", "confirmation"}
    discovery = spec["per_split"]["discovery"]
    assert discovery["owner_count"] == 2
    assert discovery["non_loop_context_count"] == 1


def test_feature_overview_counts_loop_tails_separately(artifacts) -> None:
    spec = atlas.build_feature_overview_spec(artifacts)
    for split in ("discovery", "confirmation"):
        entry = spec["per_split"][split]
        assert "loop_tail_context_count" in entry
        assert "non_loop_context_count" in entry


# ---------------------------------------------------------------------------
# Rendering and CLI
# ---------------------------------------------------------------------------


def test_render_writes_a_png_per_figure(artifacts, tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    output = tmp_path / "visual"
    spec = atlas.build_owner_map_spec(artifacts, IMAGE_ID)
    path = Path(atlas.render_spec(spec, artifacts, output))
    assert path.is_file()
    assert path.suffix == ".png"


def test_cli_specs_only_writes_specs_and_manifest(
    plan_dir: Path, shard_root: Path, tmp_path: Path
) -> None:
    output = tmp_path / "visual"
    exit_code = atlas.main(
        [
            "--plan-dir",
            str(plan_dir),
            "--shard-root",
            str(shard_root),
            "--output-dir",
            str(output),
            "--specs-only",
        ]
    )
    assert exit_code == 0
    manifest = json.loads((output / atlas.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == atlas.MANIFEST_SCHEMA_VERSION
    assert set(manifest["products"]) == set(atlas.PRODUCTS)
    specs = (output / "visual-specs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(specs) == manifest["figure_count"]
    for line in specs:
        json.loads(line)


# ---------------------------------------------------------------------------
# Round trip against the real scorer schema (anti-phantom regression)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scorer():
    return pytest.importorskip(
        "scripts.research.score_sorted_owner_accessibility_census_shard"
    )


def test_atlas_reads_shard_files_written_by_the_real_scorer(
    scorer, plan_dir: Path, tmp_path: Path
) -> None:
    """Round trip through the scorer's own publisher, not hand-written files.

    ``shard_output_files`` owns the shard file names and byte format, so a
    rename or reformat on that side surfaces here instead of silently leaving
    the atlas reading nothing.
    """

    result = scorer.ShardResult(
        receipt={"unit_id": atlas.UNIT_ID, "image_id": IMAGE_ID},
        scores=_score_rows(),
        x1=[],
        proposals=_proposal_rows(),
        free_decodes=_free_decode_rows(),
    )
    shard = tmp_path / "shards" / IMAGE_ID
    shard.mkdir(parents=True)
    for name, content in scorer.shard_output_files(result).items():
        (shard / name).write_bytes(content)

    loaded = atlas.load_artifacts(plan_dir, shard_root=tmp_path / "shards")
    assert len(loaded.scores) == 3
    assert len(loaded.proposals) == 1
    assert len(loaded.free_next_rows) == 1

    proposal = atlas.build_proposal_map_spec(loaded, CONTEXT_ID)
    assert proposal["boundary_gate"]["continue_vs_stop_logprob_margin"] == 2.9
    assert proposal["free_next_row_available"] is True

    card = atlas.build_owner_card_spec(loaded, f"gt:{IMAGE_ID}:0")
    assert card["captured_score_row_count"] == 2

    specs = atlas.build_atlas_specs(loaded)
    assert {str(spec["product"]) for spec in specs} == set(atlas.PRODUCTS)


def test_scorer_file_names_match_the_atlas_constants(scorer) -> None:
    assert atlas.SHARD_SCORES_NAME == scorer.SCORES_NAME
    assert atlas.SHARD_PROPOSAL_NAME == scorer.PROPOSAL_NAME
    assert atlas.SHARD_FREE_DECODE_NAME == scorer.FREE_DECODE_NAME
    assert atlas.SHARD_QUARANTINE_NAME == scorer.QUARANTINE_NAME


def test_phantom_field_names_are_absent_from_the_scorer(scorer) -> None:
    """The three drifted names must not exist on the producing side."""

    source = Path(scorer.__file__).read_text(encoding="utf-8")
    assert "continue_vs_stop_logit_margin" not in source
    assert "free_decoded_next_row" not in source
    # The real names must be present.
    assert "continue_vs_stop_logprob_margin" in source
    assert atlas.FREE_NEXT_ROW_KIND in source
    assert "generator_gt_owner_ids" in source


def test_atlas_does_not_reference_phantom_field_names() -> None:
    source = Path(atlas.__file__).read_text(encoding="utf-8")
    assert "continue_vs_stop_logit_margin" not in source
    assert "free_decoded_next_row" not in source


def test_owner_card_never_filters_scores_by_a_scalar_owner_field() -> None:
    """Owner-registry rows do carry a scalar ``gt_owner_id``; score rows do not.

    The guard is therefore scoped to the owner-card builder, which is where a
    scalar filter over ``artifacts.scores`` would silently attribute nothing.
    """

    import inspect

    source = inspect.getsource(atlas.build_owner_card_spec)
    assert 'row["gt_owner_id"]' not in source
    assert "_score_row_generated_by" in source


def test_cli_reports_a_missing_plan_cleanly(tmp_path: Path) -> None:
    assert atlas.main(
        ["--plan-dir", str(tmp_path / "absent"), "--output-dir", str(tmp_path / "out")]
    ) == 2
