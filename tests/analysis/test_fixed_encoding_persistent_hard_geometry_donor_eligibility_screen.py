from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from scripts.research import run_fixed_encoding_persistent_hard_geometry_donor_eligibility_screen as screen
from scripts.research import run_fixed_encoding_downstream_geometry_state_portability as geometry


COHORT_PATH = Path(screen.DEFAULT_COHORT)


def _cohort() -> dict:
    return json.loads(COHORT_PATH.read_text(encoding="utf-8"))


def _valid_suffix() -> list[int]:
    return [screen.COORDINATE_TOKEN_START + 100, screen.COORDINATE_TOKEN_START + 101, screen.COORDINATE_TOKEN_START + 200, screen.COORDINATE_TOKEN_START + 201, screen.BOX_END]


def _attribution(*, donor_iou: float = 0.8, competing_iou: float = 0.1, paired_iou: float = 0.0, closer: bool = True) -> dict:
    return {
        "donor_iou": donor_iou,
        "strongest_competing_iou": competing_iou,
        "paired_iou": paired_iou,
        "strictly_closer_to_donor": closer,
        "donor_iou_improves_over_support": True,
        "donor_l1_improves_over_support": True,
    }


def test_frozen_cohort_and_rectangles_are_valid() -> None:
    cohort = _cohort()
    screen.validate_cohort(cohort)
    case = cohort["cases"][0]
    height = case["image_grid_thw"][1] // case["merge_size"]
    width = case["image_grid_thw"][2] // case["merge_size"]
    target = screen.rectangle_indices(case["target_support_rectangle"], merged_height=height, merged_width=width)
    paired = screen.rectangle_indices(case["paired_support_rectangle"], merged_height=height, merged_width=width)
    unrelated = screen.rectangle_indices(case["unrelated_location_support_rectangle"], merged_height=height, merged_width=width)
    assert len(target) == len(paired) == len(unrelated)
    assert not set(target) & set(paired)
    assert not set(unrelated) & (set(target) | set(paired))
    with pytest.raises(ValueError):
        screen.rectangle_indices([0, 0, 1, 2], merged_height=height, merged_width=width)


def test_generic_row_prefix_accepts_multi_token_description() -> None:
    row = [screen.OBJECT_REF_START, 101, 102, 103, screen.OBJECT_REF_END, screen.BOX_START, *_valid_suffix()[:4], screen.BOX_END]
    assert screen.row_geometry_prefix(row) == row[:6]
    with pytest.raises(ValueError):
        screen.row_geometry_prefix(row[:-1])


def test_dynamic_mask_includes_current_next_token_query() -> None:
    ids = torch.tensor([[10, 99, 99, 99, screen.BOX_START]])
    mask, receipt = geometry.build_dynamic_mask_for_ids(ids=ids, image_token_id=99, selected_indices=[1], prefix_length=4, device=torch.device("cpu"))
    assert receipt["passed"]
    assert receipt["query_range"]["indices"] == [3, 4]
    assert bool(mask[0, 0, 4, 1]) is False
    extended = torch.cat([ids, torch.tensor([[screen.COORDINATE_TOKEN_START]])], dim=1)
    _, extended_receipt = geometry.build_dynamic_mask_for_ids(ids=extended, image_token_id=99, selected_indices=[1], prefix_length=4, device=torch.device("cpu"))
    assert extended_receipt["query_range"]["indices"] == [3, 4, 5]


def test_exact_geometry_suffix_parser_rejects_extra_or_bad_tokens() -> None:
    parsed = screen.parse_geometry_suffix(_valid_suffix())
    assert parsed["valid"] is True
    assert parsed["coordinate_bins"] == [100, 101, 200, 201]
    assert screen.parse_geometry_suffix(_valid_suffix() + [screen.BOX_END])["reason"] == "trailing_token_after_geometry_suffix"
    assert screen.parse_geometry_suffix(_valid_suffix()[:4] + [screen.OBJECT_REF_END])["reason"] == "box_end_not_final"
    assert screen.parse_geometry_suffix(_valid_suffix()[:3] + [screen.COORDINATE_TOKEN_START - 1, screen.BOX_END])["reason"] == "coordinate_token_out_of_range"


def test_donor_eligibility_requires_competing_and_unrelated_specificity() -> None:
    accepted = screen.assess_donor_eligibility(valid_path=True, natural_closure=True, coordinate_release=0.2, attribution=_attribution(), unrelated_donor_iou=0.1)
    assert accepted["passed"] is True
    assert not screen.assess_donor_eligibility(valid_path=True, natural_closure=True, coordinate_release=0.2, attribution=_attribution(), unrelated_donor_iou=None)["passed"]
    assert not screen.assess_donor_eligibility(valid_path=True, natural_closure=True, coordinate_release=0.2, attribution=_attribution(competing_iou=0.7), unrelated_donor_iou=0.1)["passed"]
    assert not screen.assess_donor_eligibility(valid_path=True, natural_closure=True, coordinate_release=0.2, attribution=_attribution(), unrelated_donor_iou=0.7)["passed"]
    assert not screen.assess_donor_eligibility(valid_path=True, natural_closure=True, coordinate_release=0.2, attribution=_attribution(closer=False), unrelated_donor_iou=0.1)["passed"]


def test_unrelated_specificity_is_evaluated_against_the_current_donor() -> None:
    target = _attribution()
    paired = _attribution()
    target["donor_annotation_id"] = "target"
    paired["donor_annotation_id"] = "paired"
    assert screen.assess_donor_eligibility(valid_path=True, natural_closure=True, coordinate_release=0.2, attribution=target, unrelated_donor_iou=0.1)["passed"]
    assert not screen.assess_donor_eligibility(valid_path=True, natural_closure=True, coordinate_release=0.2, attribution=paired, unrelated_donor_iou=0.7)["passed"]


def test_geometry_attribution_reports_all_objects_and_support_envelope() -> None:
    parsed = screen.parse_geometry_suffix(_valid_suffix())
    donor = [100 / 999.0, 101 / 999.0, 200 / 999.0, 201 / 999.0]
    paired = [0.7, 0.7, 0.8, 0.8]
    attributed = screen.geometry_attribution(parsed, donor_box=donor, paired_box=paired, accepted_objects=[{"annotation_id": "a", "normalized_box": donor}, {"annotation_id": "b", "normalized_box": paired}], support_box=donor)
    assert attributed["donor_iou"] == pytest.approx(1.0)
    assert attributed["support_envelope_iou"] == pytest.approx(1.0)
    assert set(attributed["all_object_ious"]) == {"a", "b"}


def test_exact_support_envelope_is_not_an_instance_owned_donor() -> None:
    parsed = screen.parse_geometry_suffix(_valid_suffix())
    donor = [100 / 999.0, 101 / 999.0, 200 / 999.0, 201 / 999.0]
    exact_support = screen.geometry_attribution(
        parsed,
        donor_box=donor,
        paired_box=[0.7, 0.7, 0.8, 0.8],
        donor_annotation_id="a",
        accepted_objects=[{"annotation_id": "a", "normalized_box": donor}],
        support_box=donor,
    )
    assert exact_support["donor_iou_improves_over_support"] is False
    assert not screen.assess_donor_eligibility(
        valid_path=True,
        natural_closure=True,
        coordinate_release=0.2,
        attribution=exact_support,
        unrelated_donor_iou=0.0,
    )["passed"]


def test_every_frozen_support_envelope_fails_but_exact_donor_geometry_can_pass() -> None:
    for case in _cohort()["cases"]:
        height = case["image_grid_thw"][1] // case["merge_size"]
        width = case["image_grid_thw"][2] // case["merge_size"]
        support = screen.support_envelope_box(case["target_support_rectangle"], merged_height=height, merged_width=width)
        donor_pixels = case["target_canvas_box_xyxy"]
        donor = [donor_pixels[0] / case["image_width"], donor_pixels[1] / case["image_height"], donor_pixels[2] / case["image_width"], donor_pixels[3] / case["image_height"]]
        paired_pixels = case["paired_canvas_box_xyxy"]
        paired = [paired_pixels[0] / case["image_width"], paired_pixels[1] / case["image_height"], paired_pixels[2] / case["image_width"], paired_pixels[3] / case["image_height"]]
        support_path = {"valid": True, "normalized_box": support}
        donor_path = {"valid": True, "normalized_box": donor}
        objects = [{"annotation_id": "target", "normalized_box": donor}, {"annotation_id": "paired", "normalized_box": paired}]
        support_attr = screen.geometry_attribution(support_path, donor_box=donor, paired_box=paired, donor_annotation_id="target", accepted_objects=objects, support_box=support)
        donor_attr = screen.geometry_attribution(donor_path, donor_box=donor, paired_box=paired, donor_annotation_id="target", accepted_objects=objects, support_box=support)
        assert not support_attr["donor_iou_improves_over_support"]
        assert not screen.assess_donor_eligibility(valid_path=True, natural_closure=True, coordinate_release=0.2, attribution=support_attr, unrelated_donor_iou=0.0)["passed"]
        assert screen.assess_donor_eligibility(valid_path=True, natural_closure=True, coordinate_release=0.2, attribution=donor_attr, unrelated_donor_iou=0.0)["passed"]


def test_panel_decision_separates_clean_and_stress() -> None:
    cohort = _cohort()
    results = []
    for case in cohort["cases"]:
        arm = {"eligibility": {"passed": str(case["image_id"]) == "7818"}}
        results.append({"image_id": case["image_id"], "trust_gate": {"passed": True}, "arms": {"target": arm}})
    decision = screen.classify_panel(results, cohort=cohort)
    assert decision["classification"] == "clean_geometry_donor_found_requires_portability_successor"
    assert decision["eligible_clean_image_ids"] == ["7818"]
    results[0]["arms"]["target"]["eligibility"]["passed"] = False
    decision = screen.classify_panel(results, cohort=cohort)
    assert decision["classification"] == "no_resolution_qualified_geometry_donor_in_curated_panel"
    stress_only = copy.deepcopy(results)
    stress_only[-1]["arms"]["target"]["eligibility"]["passed"] = True
    decision = screen.classify_panel(stress_only, cohort=cohort)
    assert decision["classification"] == "no_resolution_qualified_geometry_donor_in_curated_panel"


def test_panel_missing_trust_evidence_is_invalid() -> None:
    cohort = _cohort()
    results = [{"image_id": case["image_id"], "arms": {"target": {"eligibility": {"passed": False}}}} for case in cohort["cases"]]
    assert screen.classify_panel(results, cohort=cohort)["classification"] == "invalid_execution_trust_gate"


def _receipt(image_id: str, *, config: str = "c", source: str = "s", ledger: str = "l", cohort: str = "h") -> dict:
    return {"unit_id": screen.UNIT_ID, "image_id": image_id, "config_sha256": config, "source_jsonl_sha256": source, "audit_ledger_sha256": ledger, "cohort_sha256": cohort, "trust_gate": {"passed": True}, "arms": {}}


def test_merge_rejects_missing_duplicate_and_hash_mismatch() -> None:
    cohort = _cohort()
    receipts = [_receipt(image_id, config="c", source="s", ledger="l", cohort="h") for image_id in screen.EXPECTED_IMAGE_IDS]
    merged = screen.merge_case_receipts(receipts, cohort=cohort, config_sha256="c", source_jsonl_sha256="s", audit_ledger_sha256="l", cohort_sha256="h")
    assert [item["image_id"] for item in merged["results"]] == list(screen.EXPECTED_IMAGE_IDS)
    with pytest.raises(ValueError, match="duplicate"):
        screen.merge_case_receipts(receipts + [receipts[0]], cohort=cohort, config_sha256="c", source_jsonl_sha256="s", audit_ledger_sha256="l", cohort_sha256="h")
    with pytest.raises(ValueError, match="exactly"):
        screen.merge_case_receipts(receipts[:-1], cohort=cohort, config_sha256="c", source_jsonl_sha256="s", audit_ledger_sha256="l", cohort_sha256="h")
    bad = list(receipts)
    bad[0] = _receipt("7818", config="wrong")
    with pytest.raises(ValueError, match="config_sha256"):
        screen.merge_case_receipts(bad, cohort=cohort, config_sha256="c", source_jsonl_sha256="s", audit_ledger_sha256="l", cohort_sha256="h")
    missing_trust = list(receipts)
    missing_trust[0] = {key: value for key, value in missing_trust[0].items() if key != "trust_gate"}
    with pytest.raises(ValueError, match="trust gate"):
        screen.merge_case_receipts(missing_trust, cohort=cohort, config_sha256="c", source_jsonl_sha256="s", audit_ledger_sha256="l", cohort_sha256="h")


def test_merge_accepts_split_wrapper_receipts_and_flattens_cases() -> None:
    cohort = _cohort()
    wrappers = []
    for image_id in screen.EXPECTED_IMAGE_IDS:
        wrappers.append({"unit_id": screen.UNIT_ID, "config_sha256": "c", "source_jsonl_sha256": "s", "audit_ledger_sha256": "l", "cohort_sha256": "h", "results": [{"image_id": image_id, "arms": {}, "trust_gate": {"passed": True}}]})
    merged = screen.merge_case_receipts(wrappers, cohort=cohort, config_sha256="c", source_jsonl_sha256="s", audit_ledger_sha256="l", cohort_sha256="h")
    assert [item["image_id"] for item in merged["results"]] == list(screen.EXPECTED_IMAGE_IDS)
