"""Focused tests for the CPU-only sorted root-position diagnostic."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import analyze_sorted_root_position_bias as position  # noqa: E402


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(position.canonical_json_bytes(row) + b"\n" for row in rows)


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    result["receipt_content_sha256"] = position.sha256_json(result)
    return result


def _root_feature(image_id: str, owner_id: str, index: int) -> dict[str, Any]:
    def block(offset: float) -> dict[str, float]:
        return {
            "peak_lift": 1.0 + index + offset,
            "local_concentration": 2.0 + index + offset,
            # These raw fields are present upstream and must not survive output.
            "value": -10.0 - index,
            "bank_median": -20.0 - index,
        }

    return {
        "schema_version": position.LEGACY_CONTEXT_SCHEMA_VERSION,
        "row_kind": "census_owner_context",
        "image_id": image_id,
        "gt_owner_id": owner_id,
        "context_id": f"{image_id}:boundary-000",
        "owner_context_id": f"{owner_id}@root",
        "context_role": "root",
        "boundary_index": 0,
        "localization": {
            "generator_local_max_excluding_other_owner_strict": {
                "ambiguity_excluded_l": block(0.0),
                "ambiguity_included_u": block(0.25),
            }
        },
    }


def _summary(image_id: str, owner_id: str, index: int) -> dict[str, Any]:
    support_l = index % 2 == 0
    support_u = support_l or index == 1
    native_tp = index % 2 == 0
    greedy_eligible = index != 3
    return {
        "schema_version": position.LEGACY_SUMMARY_SCHEMA_VERSION,
        "row_kind": "census_owner_summary",
        "image_id": image_id,
        "gt_owner_id": owner_id,
        "split": "confirmation",
        "native_true_positive": native_tp,
        "greedy_eligible": greedy_eligible,
        "greedy_eligibility_status": (
            "eligible" if greedy_eligible else "globally_ambiguous_neutral"
        ),
        "disposition": (
            "native_true_positive_calibration_control"
            if native_tp
            else "resolved_tested_localization_support"
        ),
        "ambiguity_bound_disposition_flip": support_u and not support_l,
        "lower_bound_l": {
            "non_loop_context_support": {f"{image_id}:boundary-000": support_l}
        },
        "upper_bound_u": {
            "non_loop_context_support": {f"{image_id}:boundary-000": support_u}
        },
    }


def _owner(image_id: str, index: int) -> dict[str, Any]:
    boxes = (
        [2, 2, 20, 20],
        [38, 2, 60, 20],
        [2, 38, 20, 60],
        [38, 38, 60, 60],
    )
    bbox = boxes[index]
    native_tp = index % 2 == 0
    greedy_eligible = index != 3
    return {
        "schema_version": position.LEGACY_PLAN_SCHEMA_VERSION,
        "row_kind": "census_owner",
        "image_id": image_id,
        "gt_owner_id": f"gt:{image_id}:{index}",
        "bbox_pixel_xyxy": bbox,
        "owner_sort_key": [bbox[1], bbox[0]],
        "normalized_description": "person" if index < 3 else "tie",
        "native_true_positive": native_tp,
        "greedy_eligible": greedy_eligible,
        "greedy_eligibility_status": (
            "eligible" if greedy_eligible else "globally_ambiguous_neutral"
        ),
        "excluded_from_census": False,
        "disposition_eligibility": {"native_false_negative": not native_tp},
    }


def _write_run(root: Path, *, image_id: str) -> Path:
    presentation = root / "phases" / "presentation"
    plan = root / "plan"
    presentation.mkdir(parents=True)
    plan.mkdir(parents=True)
    owners = [_owner(image_id, index) for index in range(4)]
    summaries = [
        _summary(image_id, row["gt_owner_id"], index)
        for index, row in enumerate(owners)
    ]
    features: list[dict[str, Any]] = []
    for index, owner in enumerate(owners):
        features.append(_root_feature(image_id, owner["gt_owner_id"], index))
        features.append(
            {
                "schema_version": position.LEGACY_CONTEXT_SCHEMA_VERSION,
                "row_kind": "census_owner_context",
                "image_id": image_id,
                "gt_owner_id": owner["gt_owner_id"],
                "context_id": f"{image_id}:boundary-001",
                "context_role": "row_boundary",
                "boundary_index": 1,
                "raw_logprob": 999.0,
            }
        )
    image_rows = [
        {
            "schema_version": position.LEGACY_PLAN_SCHEMA_VERSION,
            "row_kind": "census_image",
            "image_id": image_id,
            "image_width": 64,
            "image_height": 64,
            "prompt_token_ids": [100, position.IMAGE_PAD_TOKEN_ID] * 0
            + [position.IMAGE_PAD_TOKEN_ID] * 4
            + [200, 201],
        }
    ]
    context_rows = [
        {
            "schema_version": position.LEGACY_PLAN_SCHEMA_VERSION,
            "row_kind": "census_context",
            "image_id": image_id,
            "context_id": f"{image_id}:boundary-000",
            "context_role": "root",
            "boundary_index": 0,
        }
    ]
    category_rows = [
        {
            "schema_version": position.LEGACY_PLAN_SCHEMA_VERSION,
            "row_kind": "census_category",
            "image_id": image_id,
            "normalized_description": "person",
        }
    ]
    files = {
        presentation / "owner-context-features.jsonl": _jsonl_bytes(features),
        presentation / "owner-summaries.jsonl": _jsonl_bytes(summaries),
        plan / "owner-registry.jsonl": _jsonl_bytes(owners),
        plan / "image-registry.jsonl": _jsonl_bytes(image_rows),
        plan / "context-registry.jsonl": _jsonl_bytes(context_rows),
        plan / "category-registry.jsonl": _jsonl_bytes(category_rows),
    }
    for path, content in files.items():
        path.write_bytes(content)
    plan_digests = {
        path.name: position.sha256_bytes(content)
        for path, content in files.items()
        if path.parent == plan
    }
    merge_digests = {
        path.name: position.sha256_bytes(content)
        for path, content in files.items()
        if path.parent == presentation
    }
    (plan / "receipt.json").write_bytes(
        position.canonical_json_bytes(
            _seal(
                {
                    "unit_id": position.CENSUS_UNIT_ID,
                    "schema_version": position.LEGACY_PLAN_SCHEMA_VERSION,
                    "output_file_digests": plan_digests,
                }
            )
        )
        + b"\n"
    )
    (presentation / "merge-receipt.json").write_bytes(
        position.canonical_json_bytes(
            _seal(
                {
                    "unit_id": position.CENSUS_UNIT_ID,
                    "schema_version": position.LEGACY_MERGE_RECEIPT_SCHEMA_VERSION,
                    "output_file_digests": merge_digests,
                }
            )
        )
        + b"\n"
    )
    return presentation / "owner-context-features.jsonl"


def _write_prospective_s1(root: Path) -> tuple[Path, Path]:
    analysis_dir = root / "s1-analysis"
    plan_dir = root / "s1-plan"
    analysis_dir.mkdir(parents=True)
    plan_dir.mkdir(parents=True)
    image_id = position.PROSPECTIVE_IMAGE_ID
    owners: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    for index in range(position.PROSPECTIVE_EXPECTED_OWNER_COUNT):
        x1 = (index % 8) * 32 + 2
        y1 = (index // 8) * 32 + 2
        category = "person" if index < 38 else "tie"
        native_tp = index % 3 == 0
        owner_id = f"gt:{image_id}:{index}"
        owners.append(
            {
                "schema_version": position.LEGACY_PLAN_SCHEMA_VERSION,
                "row_kind": "census_owner",
                "image_id": image_id,
                "gt_owner_id": owner_id,
                "bbox_pixel_xyxy": [x1, y1, x1 + 20, y1 + 20],
                "owner_sort_key": [y1, x1],
                "normalized_description": category,
                "native_true_positive": native_tp,
                "greedy_eligible": True,
                "greedy_eligibility_status": "eligible",
                "excluded_from_census": False,
                "disposition_eligibility": {"native_false_negative": not native_tp},
            }
        )
        summary = _summary(image_id, owner_id, index)
        summary.update(
            {
                "schema_version": position.PROSPECTIVE_SUMMARY_SCHEMA_VERSION,
                "row_kind": "image2299_owner_accessibility_summary",
                "native_true_positive": native_tp,
                "native_false_negative": not native_tp,
                "disposition": (
                    "native_true_positive_transfer_control"
                    if native_tp
                    else "resolved_tested_localization_support"
                ),
                "disposition_role": "validity_bearing",
            }
        )
        summaries.append(summary)
        root_feature = _root_feature(image_id, owner_id, index)
        root_feature.update(
            {
                "schema_version": position.PROSPECTIVE_CONTEXT_SCHEMA_VERSION,
                "row_kind": "image2299_owner_context_support",
            }
        )
        features.append(root_feature)
        features.append(
            {
                "schema_version": position.PROSPECTIVE_CONTEXT_SCHEMA_VERSION,
                "row_kind": "image2299_owner_context_support",
                "image_id": image_id,
                "gt_owner_id": owner_id,
                "context_id": f"{image_id}:boundary-001",
                "context_role": "row_boundary",
                "boundary_index": 1,
                "raw_logprob": 123.0,
            }
        )
    image_rows = [
        {
            "schema_version": position.LEGACY_PLAN_SCHEMA_VERSION,
            "row_kind": "census_image",
            "image_id": image_id,
            "image_width": 256,
            "image_height": 256,
            "prompt_token_ids": [100] + [position.IMAGE_PAD_TOKEN_ID] * 64 + [200, 201],
        }
    ]
    context_rows = [
        {
            "schema_version": position.LEGACY_PLAN_SCHEMA_VERSION,
            "row_kind": "census_context",
            "image_id": image_id,
            "context_id": f"{image_id}:boundary-000",
            "context_role": "root",
            "boundary_index": 0,
        },
        {
            "schema_version": position.LEGACY_PLAN_SCHEMA_VERSION,
            "row_kind": "census_context",
            "image_id": image_id,
            "context_id": f"{image_id}:boundary-001",
            "context_role": "row_boundary",
            "boundary_index": 1,
        },
    ]
    category_rows = [
        {
            "schema_version": position.LEGACY_PLAN_SCHEMA_VERSION,
            "row_kind": "census_category",
            "image_id": image_id,
            "normalized_description": category,
        }
        for category in ("person", "tie")
    ]
    plan_files = {
        plan_dir / "owner-registry.jsonl": _jsonl_bytes(owners),
        plan_dir / "image-registry.jsonl": _jsonl_bytes(image_rows),
        plan_dir / "context-registry.jsonl": _jsonl_bytes(context_rows),
        plan_dir / "category-registry.jsonl": _jsonl_bytes(category_rows),
    }
    for path, content in plan_files.items():
        path.write_bytes(content)
    plan_native_tp_count = sum(row["native_true_positive"] for row in owners)
    plan_receipt = _seal(
        {
            "schema_version": position.LEGACY_PLAN_SCHEMA_VERSION,
            "unit_id": position.CENSUS_UNIT_ID,
            "extension_unit_id": position.UNIT_ID,
            "census_shape": {
                "image_ids": [image_id],
                "owner_count": position.PROSPECTIVE_EXPECTED_OWNER_COUNT,
                "owner_category_counts": position.PROSPECTIVE_EXPECTED_CATEGORY_COUNTS,
                "native_true_positive_owner_count": plan_native_tp_count,
                "native_false_negative_owner_count": 46 - plan_native_tp_count,
            },
            "denominator_contract": {
                "prospective_image2299_owner_count": 46,
                "legacy_12_owner_count_unchanged": 346,
                "legacy_12_eligible_native_fn_denominator_unchanged": 202,
                "pooled_13_image_denominator_created": False,
                "report_slices_separately": True,
            },
            "output_file_digests": {
                path.name: position.sha256_bytes(content)
                for path, content in plan_files.items()
            },
        }
    )
    (plan_dir / "receipt.json").write_bytes(
        position.canonical_json_bytes(plan_receipt) + b"\n"
    )

    native_tp_count = plan_native_tp_count
    denominators = {
        "image2299_owner_count": 46,
        "image2299_native_tp_count": native_tp_count,
        "image2299_native_fn_count": 46 - native_tp_count,
        "legacy_12_owner_count_unchanged": 346,
        "legacy_12_eligible_native_fn_denominator_unchanged": 202,
        "pooled_13_image_denominator_created": False,
    }
    analysis = {
        "schema_version": position.PROSPECTIVE_ANALYSIS_SCHEMA_VERSION,
        "unit_id": position.UNIT_ID,
        "image_id": image_id,
        "slice_role": "prospective_single_image_separate_from_legacy_12",
        "denominators": denominators,
        "calibration_transfer": {
            "status": "calibration_nontransferring",
            "passes": False,
            "validity_bearing": True,
            "support_rate": 0.5,
            "floor": 0.8,
            "supported_due_boundary_count": 8,
            "transfer_denominator_native_tp_count": native_tp_count,
        },
        "proposal_localization_contract": {
            "proposal_surface_kept_separate": True,
            "proposal_surface_used_as_support_input": False,
            "localization_estimand": "category_field_support_at_owner_geometry",
            "is_per_owner_proposal_probability": False,
        },
    }
    analysis_files = {
        analysis_dir / "analysis.json": position.canonical_json_bytes(analysis) + b"\n",
        analysis_dir / "owner-summaries.jsonl": _jsonl_bytes(summaries),
        analysis_dir / "owner-context-features.jsonl": _jsonl_bytes(features),
        analysis_dir / "context-registry.jsonl": _jsonl_bytes(context_rows),
    }
    for path, content in analysis_files.items():
        path.write_bytes(content)
    analysis_receipt = _seal(
        {
            "schema_version": position.PROSPECTIVE_RECEIPT_SCHEMA_VERSION,
            "unit_id": position.UNIT_ID,
            "plan": {
                "path": str(plan_dir),
                "receipt_content_sha256": plan_receipt["receipt_content_sha256"],
            },
            "denominator_contract": denominators,
            "output_file_digests": {
                path.name: position.sha256_bytes(content)
                for path, content in analysis_files.items()
            },
        }
    )
    (analysis_dir / "receipt.json").write_bytes(
        position.canonical_json_bytes(analysis_receipt) + b"\n"
    )
    return analysis_dir, plan_dir


@pytest.fixture()
def two_slice_result(tmp_path: Path) -> dict[str, Any]:
    legacy_root = tmp_path / "legacy"
    prospective_root = tmp_path / "prospective"
    _write_run(legacy_root, image_id="1")
    prospective_analysis_dir, prospective_plan_dir = _write_prospective_s1(
        prospective_root
    )
    return position.run_analysis(
        legacy_run_root=legacy_root,
        prospective_analysis_dir=prospective_analysis_dir,
        prospective_plan_dir=prospective_plan_dir,
    )


def test_root_only_filtering_excludes_every_post_root_row(
    two_slice_result: dict[str, Any],
) -> None:
    legacy = two_slice_result["summary"]["slices"][position.LEGACY_SLICE]
    assert legacy["diagnostics"]["root_feature_row_count"] == 4
    assert legacy["diagnostics"]["post_root_feature_rows_excluded"] == 4
    assert all(
        row["root_context"]["context_role"] == "root"
        and row["root_context"]["boundary_index"] == 0
        for row in two_slice_result["owner_rows"]
    )


def test_position_rank_confounding_guard_keeps_due_index_provenance_only(
    two_slice_result: dict[str, Any],
) -> None:
    guard = two_slice_result["summary"]["confounding_guard"]
    assert guard["sorted_due_index_enters_any_analysis"] is False
    assert guard["post_root_rows_excluded"] is True
    assert guard["post_root_t_enters_native_fn_analysis"] is False
    assert "sorted_due_index" not in position.ANALYSIS_FEATURES
    assert all(
        row["sorted_due_index"]["role"]
        == "provenance_only_excluded_from_every_association_and_adjustment"
        for row in two_slice_result["owner_rows"]
    )


def test_quadrants_report_native_fn_on_matching_universe_and_exclude_outside(
    two_slice_result: dict[str, Any],
) -> None:
    legacy = two_slice_result["summary"]["slices"][position.LEGACY_SLICE]
    quadrants = legacy["strata"]["quadrant"]
    assert quadrants["top_right"]["native_matching_universe_owner_count"] == 1
    assert quadrants["top_right"]["native_false_negative_count"] == 1
    assert quadrants["top_right"]["native_false_negative_fraction"] == 1.0
    assert quadrants["top_right"]["outside_native_matching_universe_count"] == 0
    assert quadrants["bottom_right"]["native_matching_universe_owner_count"] == 0
    assert quadrants["bottom_right"]["native_false_negative_count"] == 0
    assert quadrants["bottom_right"]["native_false_negative_fraction"] is None
    assert quadrants["bottom_right"]["outside_native_matching_universe_count"] == 1
    outside = next(
        row
        for row in two_slice_result["owner_rows"]
        if row["slice_id"] == position.LEGACY_SLICE and row["gt_owner_id"] == "gt:1:3"
    )
    assert outside["native_false_negative"] is True
    assert outside["in_native_matching_universe"] is False
    assert outside["native_false_negative_in_matching_universe"] is None


def test_native_fn_associations_are_separate_from_root_support_and_root_only(
    two_slice_result: dict[str, Any],
) -> None:
    summary = two_slice_result["summary"]
    policy = summary["native_fn_screen_policy"]
    assert policy["independent_of_frozen_root_support_association"] is True
    assert policy["post_root_t_used"] is False
    assert policy["sorted_due_index_used"] is False
    for slice_id in summary["slice_order"]:
        block = summary["slices"][slice_id]
        assert set(block["within_image_native_fn_associations"]) == set(
            position.ANALYSIS_FEATURES
        )
        for association in block["within_image_native_fn_associations"].values():
            assert (
                association["target"]
                == "native_false_negative_indicator_within_native_matching_universe"
            )
            assert association["independent_of_root_support_association"] is True
            assert "no post-root context" in association["stratification"]


def test_no_raw_score_is_pooled_or_emitted(two_slice_result: dict[str, Any]) -> None:
    position.assert_no_pooled_raw_scores(two_slice_result)
    serialized = json.dumps(two_slice_result, sort_keys=True)
    assert "raw_logprob" not in serialized
    assert "bank_median" not in serialized
    with pytest.raises(position.RootPositionContractError, match="forbidden raw-score"):
        position.assert_no_pooled_raw_scores({"raw_logprob": -3.0})


def test_prospective_slice_is_stable_and_never_pooled(
    two_slice_result: dict[str, Any],
) -> None:
    summary = two_slice_result["summary"]
    assert summary["slice_order"] == [position.PROSPECTIVE_SLICE, position.LEGACY_SLICE]
    assert summary["slices_are_never_pooled"] is True
    assert set(summary["slices"]) == {position.PROSPECTIVE_SLICE, position.LEGACY_SLICE}
    assert summary["slices"][position.PROSPECTIVE_SLICE]["image_ids"] == ["2299"]
    assert (
        summary["slices"][position.PROSPECTIVE_SLICE]["overall"]["owner_count"]
        == position.PROSPECTIVE_EXPECTED_OWNER_COUNT
    )
    prospective_binding = summary["slices"][position.PROSPECTIVE_SLICE]["diagnostics"][
        "binding"
    ]
    assert prospective_binding["input_layout"] == "prospective_s1"
    assert (
        prospective_binding["analysis_schema_version"]
        == position.PROSPECTIVE_ANALYSIS_SCHEMA_VERSION
    )
    assert len(prospective_binding["analysis_receipt_content_sha256"]) == 64
    root_scope = summary["slices"][position.PROSPECTIVE_SLICE]["interpretation_scope"][
        "frozen_root_support_numbers_and_associations"
    ]
    assert root_scope["status"] == "descriptive_only_unmet_calibration_transfer_gate"
    assert root_scope["calibration_transfer_gate_met"] is False
    assert root_scope["calibration_transfer"]["status"] == "calibration_nontransferring"
    native_fn_scope = summary["slices"][position.PROSPECTIVE_SLICE][
        "interpretation_scope"
    ]["native_fn_spatial_numbers_and_associations"]
    assert native_fn_scope["status"] == "calibration_independent_unaffected"
    assert native_fn_scope["depends_on_frozen_support_calibration"] is False
    rendered = position.render_markdown(summary)
    assert "descriptive_only_unmet_calibration_transfer_gate" in rendered
    assert "calibration_independent_unaffected" in rendered
    assert (
        "root-support counts, quadrant columns, and root-support associations"
        in rendered
    )
    assert summary["slices"][position.LEGACY_SLICE]["image_ids"] == ["1"]
    assert [row["slice_id"] for row in two_slice_result["owner_rows"][:4]] == [
        position.PROSPECTIVE_SLICE
    ] * 4


def test_explicit_prospective_directories_are_a_paired_cli_contract(
    tmp_path: Path,
) -> None:
    legacy_root = tmp_path / "legacy"
    _write_run(legacy_root, image_id="1")
    with pytest.raises(
        position.RootPositionContractError, match="must be provided together"
    ):
        position.run_analysis(
            legacy_run_root=legacy_root,
            prospective_analysis_dir=tmp_path / "analysis-only",
        )
    parsed = position.build_parser().parse_args(
        [
            "--prospective-analysis-dir",
            "analysis",
            "--prospective-plan-dir",
            "plan",
            "--output-dir",
            "out",
        ]
    )
    assert parsed.prospective_analysis_dir == Path("analysis")
    assert parsed.prospective_plan_dir == Path("plan")


def test_output_receipt_self_seals_and_publication_is_create_or_identical(
    tmp_path: Path, two_slice_result: dict[str, Any]
) -> None:
    files = position.build_output_files(two_slice_result)
    receipt = json.loads(files[position.RECEIPT_NAME])
    declared = receipt.pop("receipt_content_sha256")
    assert declared == position.sha256_json(receipt)
    for name, entry in receipt["output_file_digests"].items():
        assert entry["sha256"] == position.sha256_bytes(files[name])
    output_dir = tmp_path / "published"
    first = position.publish_analysis(output_dir, files)
    second = position.publish_analysis(output_dir, files)
    assert first["publish_mode"] == "atomic_staging_directory_rename"
    assert second["publish_mode"] == "no_op_identical_rerun"
    edited = dict(files)
    edited[position.REPORT_NAME] += b"drift\n"
    with pytest.raises(position.RootPositionContractError, match="non-identical"):
        position.publish_analysis(output_dir, edited)
