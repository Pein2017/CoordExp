"""CPU fixtures for the prospective image-2299 visual atlas."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts.research import analyze_sorted_image2299_owner_accessibility as s1
from scripts.research import analyze_sorted_image2299_supported_fn_reachability as s2
from scripts.research import build_sorted_image2299_native_ledger as s0
from scripts.research import build_sorted_owner_accessibility_census_plan as legacy_plan
from scripts.research import visualize_sorted_image2299_mechanism_atlas as subject


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(subject.canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(
        b"".join(subject.canonical_json_bytes(row) + b"\n" for row in rows)
    )


def _seal(receipt: dict[str, object]) -> None:
    receipt["receipt_content_sha256"] = subject.sha256_json(receipt)


def _class_for_index(index: int) -> tuple[str, str | None]:
    if index < 10:
        return s1.DISPOSITION_TP, None
    if index < 22:
        return s1.DISPOSITION_RESOLVED, s1.DISPOSITION_RESOLVED
    if index < 34:
        return s1.DISPOSITION_PERSISTENT, s1.DISPOSITION_PERSISTENT
    if index < 40:
        return s1.DISPOSITION_FLIP, s1.DISPOSITION_FLIP
    return s1.DISPOSITION_UNRESOLVED, s1.DISPOSITION_UNRESOLVED


def _fixture(
    tmp_path: Path, *, with_s2: bool, failed_transfer: bool = False
) -> dict[str, Path]:
    from PIL import Image

    width, height = 800, 600
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (width, height), (84, 92, 106)).save(image_path, quality=95)

    s0_root = tmp_path / "s0"
    plan_dir = tmp_path / "plan"
    s1_dir = tmp_path / "s1"
    for root in (s0_root, plan_dir, s1_dir):
        root.mkdir()

    s0_owners: list[dict[str, object]] = []
    plan_owners: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for index in range(subject.EXPECTED_OWNER_COUNT):
        column = index % 8
        row = index // 8
        box = [column * 95 + 10, row * 95 + 8, column * 95 + 78, row * 95 + 85]
        owner_id = f"gt:2299:{index}"
        category = "person" if index < 38 else "tie"
        native_tp = index < (19 if failed_transfer else 10)
        if native_tp:
            disposition, descriptive = s1.DISPOSITION_TP, None
        elif failed_transfer:
            disposition = s1.DISPOSITION_WITHHELD
            descriptive = (
                s1.DISPOSITION_RESOLVED if index < 35 else s1.DISPOSITION_PERSISTENT
            )
        else:
            disposition, descriptive = _class_for_index(index)
        s0_owners.append(
            {
                "schema_version": s0.OWNER_SCHEMA_VERSION,
                "gt_owner_id": owner_id,
                "image_id": subject.IMAGE_ID,
                "original_annotation_index": index,
                "normalized_description": category,
                "bbox_xyxy": box,
                "native_greedy_match_status": "matched" if native_tp else "false_negative",
            }
        )
        plan_owners.append(
            {
                "schema_version": legacy_plan.PLAN_SCHEMA_VERSION,
                "gt_owner_id": owner_id,
                "image_id": subject.IMAGE_ID,
                "original_annotation_index": index,
                "normalized_description": category,
                "bbox_pixel_xyxy": box,
                "native_true_positive": native_tp,
            }
        )
        summaries.append(
            {
                "schema_version": s1.OWNER_SCHEMA_VERSION,
                "row_kind": "image2299_owner_accessibility_summary",
                "gt_owner_id": owner_id,
                "image_id": subject.IMAGE_ID,
                "normalized_description": category,
                "bbox_pixel_xyxy": box,
                "native_true_positive": native_tp,
                "native_false_negative": not native_tp,
                "disposition": disposition,
                "frozen_disposition_descriptive": descriptive,
                "ambiguity_bound_disposition_flip": descriptive == s1.DISPOSITION_FLIP,
                "fn_disposition_interpretable": bool(not native_tp and not failed_transfer),
                "disposition_role": (
                    "withheld_calibration_nontransfer"
                    if failed_transfer and not native_tp
                    else "validity_bearing"
                ),
            }
        )

    s0_owner_path = s0_root / "owner-ledger.jsonl"
    _write_jsonl(s0_owner_path, s0_owners)
    s0_receipt: dict[str, object] = {
        "schema_version": s0.RECEIPT_SCHEMA_VERSION,
        "unit_id": subject.UNIT_ID,
        "status": "admitted",
        "outputs": {
            s0_owner_path.name: {"sha256": subject.sha256_file(s0_owner_path)}
        },
    }
    _seal(s0_receipt)
    _write_json(s0_root / "receipt.json", s0_receipt)

    image_registry = plan_dir / "image-registry.jsonl"
    owner_registry = plan_dir / "owner-registry.jsonl"
    _write_jsonl(
        image_registry,
        [
            {
                "schema_version": legacy_plan.PLAN_SCHEMA_VERSION,
                "row_kind": "census_image",
                "image_id": subject.IMAGE_ID,
                "image_width": width,
                "image_height": height,
                "file_name": "image.jpg",
            }
        ],
    )
    _write_jsonl(owner_registry, plan_owners)
    panel_path = tmp_path / "panel.jsonl"
    _write_jsonl(
        panel_path,
        [
            {
                "image_id": subject.IMAGE_ID,
                "width": width,
                "height": height,
                "images": [image_path.name],
                "objects": [],
            }
        ],
    )
    plan_receipt: dict[str, object] = {
        "schema_version": legacy_plan.PLAN_SCHEMA_VERSION,
        "unit_id": legacy_plan.UNIT_ID,
        "extension_unit_id": subject.UNIT_ID,
        "source_paths": {"panel": str(panel_path)},
        "source_content_digests": {
            "image2299_bytes_sha256": subject.sha256_file(image_path)
        },
        "output_file_digests": {
            image_registry.name: subject.sha256_file(image_registry),
            owner_registry.name: subject.sha256_file(owner_registry),
        },
    }
    _seal(plan_receipt)
    _write_json(plan_dir / "receipt.json", plan_receipt)

    analysis_path = s1_dir / "analysis.json"
    summary_path = s1_dir / "owner-summaries.jsonl"
    feature_path = s1_dir / "owner-context-features.jsonl"
    context_path = s1_dir / "context-registry.jsonl"
    _write_json(
        analysis_path,
        {
            "schema_version": s1.ANALYSIS_SCHEMA_VERSION,
            "unit_id": subject.UNIT_ID,
            "image_id": subject.IMAGE_ID,
            "denominators": {
                "image2299_owner_count": subject.EXPECTED_OWNER_COUNT,
                "image2299_native_tp_count": 19 if failed_transfer else 10,
                "image2299_native_fn_count": 27 if failed_transfer else 36,
            },
            "calibration_transfer": (
                {
                    "status": "calibration_nontransferring",
                    "passes": False,
                    "validity_bearing": True,
                    "native_tp_owner_count": 19,
                    "supported_due_boundary_count": 14,
                    "support_rate": 14 / 19,
                    "floor": 0.8,
                    "on_failure": "withhold_all_native_fn_dispositions",
                }
                if failed_transfer
                else {
                    "status": "calibration_transfer_passed",
                    "passes": True,
                    "validity_bearing": True,
                }
            ),
        },
    )
    _write_jsonl(summary_path, summaries)
    _write_jsonl(feature_path, [])
    _write_jsonl(context_path, [])
    s1_paths = [analysis_path, summary_path, feature_path, context_path]
    s1_receipt: dict[str, object] = {
        "schema_version": s1.RECEIPT_SCHEMA_VERSION,
        "unit_id": subject.UNIT_ID,
        "output_file_digests": {
            path.name: subject.sha256_file(path) for path in s1_paths
        },
    }
    _seal(s1_receipt)
    _write_json(s1_dir / "receipt.json", s1_receipt)

    result = {
        "s0": s0_root,
        "plan": plan_dir,
        "s1": s1_dir,
        "image": image_path,
    }
    if with_s2:
        s2_dir = tmp_path / "s2"
        s2_dir.mkdir()
        report_path = s2_dir / s2.REPORT_JSON_NAME
        report_md_path = s2_dir / s2.REPORT_MD_NAME
        records_path = s2_dir / s2.OWNER_RECORDS_NAME
        _write_json(
            report_path,
            {
                "schema_version": s2.REPORT_SCHEMA_VERSION,
                "unit_id": subject.UNIT_ID,
                "image_id": subject.IMAGE_ID,
                "resolved_false_negative_cohort": {
                    "denominator": 0 if failed_transfer else 12
                },
            },
        )
        report_md_path.write_text("# fixture\n", encoding="utf-8")
        records: list[dict[str, object]] = []
        for index in (() if failed_transfer else range(10, 22)):
            records.append(
                {
                    "schema_version": s2.OWNER_RECORD_SCHEMA_VERSION,
                    "cohort": "image2299_resolved_native_false_negative",
                    "gt_owner_id": f"gt:2299:{index}",
                    "image_id": subject.IMAGE_ID,
                    "normalized_description": "person",
                    "upper_bound_u": {
                        "support_context_count": 2,
                        "any_gate_open": index % 2 == 0,
                        "any_category_rank_top3": True,
                        "any_owner_rank_one": index % 3 == 0,
                        "any_favorable_top3_before_or_at_frontier": index % 2 == 0,
                    },
                    "exact_crossing": {
                        "exact_crossing_exists": True,
                        "u_favorable_supported": index % 2 == 0,
                    },
                    "source_context_ids": {
                        "upper_bound_u": [f"2299:boundary-{index:03d}"],
                        "lower_bound_l": [],
                    },
                }
            )
        _write_jsonl(records_path, records)
        output_paths = [report_path, report_md_path, records_path]
        s2_receipt: dict[str, object] = {
            "schema_version": s2.RECEIPT_SCHEMA_VERSION,
            "unit_id": subject.UNIT_ID,
            "output_file_sha256": {
                path.name: subject.sha256_file(path) for path in output_paths
            },
        }
        _seal(s2_receipt)
        _write_json(s2_dir / s2.RECEIPT_NAME, s2_receipt)
        result["s2"] = s2_dir
    return result


def test_spec_preserves_all_owner_ids_indexes_and_categorical_classes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, with_s2=True)
    artifacts = subject.load_artifacts(
        fixture["s0"], fixture["plan"], fixture["s1"], s2_analysis_dir=fixture["s2"]
    )
    spec = subject.build_atlas_spec(artifacts)

    assert spec["owner_count"] == 46
    assert [row["owner_index"] for row in spec["owners"]] == list(range(46))
    assert [row["gt_owner_id"] for row in spec["owners"]] == [
        f"gt:2299:{index}" for index in range(46)
    ]
    assert spec["native_outcome_counts"] == {"FN": 36, "TP": 10}
    assert spec["primary_class_counts"] == {
        subject.CLASS_AMBIGUITY_FLIP: 6,
        subject.CLASS_NATIVE_TP: 10,
        subject.CLASS_PERSISTENT: 12,
        subject.CLASS_RESOLVED: 12,
        subject.CLASS_UNRESOLVED: 6,
    }
    assert spec["selection_contract"] == {
        "owner_denominator": 46,
        "selection": "all_owners_in_original_annotation_index_order",
        "score_based_case_selection": False,
        "score_fields_read": False,
        "rank_or_margin_used_for_selection": False,
    }
    assert spec["owners"][10]["s2_reachability"]["support_context_count"] == 2
    assert spec["owners"][22]["s2_reachability"] is None


def test_failed_transfer_real_analysis_regression_uses_withheld_primary_class(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, with_s2=True, failed_transfer=True)
    artifacts = subject.load_artifacts(
        fixture["s0"], fixture["plan"], fixture["s1"], s2_analysis_dir=fixture["s2"]
    )
    spec = subject.build_atlas_spec(artifacts)

    assert spec["native_outcome_counts"] == {"FN": 27, "TP": 19}
    assert spec["primary_class_counts"] == {
        subject.CLASS_NATIVE_TP: 19,
        subject.CLASS_WITHHELD: 27,
    }
    assert spec["secondary_sensitivity_counts"] == {
        subject.CLASS_PERSISTENT: 11,
        subject.CLASS_RESOLVED: 16,
    }
    assert spec["s2_status"] == "not_run_calibration_nontransfer"
    for row in spec["owners"]:
        if row["native_outcome"] == "TP":
            assert row["s2_status"] == "not_applicable_native_true_positive"
            continue
        assert row["primary_visual_class"] == subject.CLASS_WITHHELD
        assert row["secondary_sensitivity"]["role"] == "descriptive_only_nontransferring"
        assert row["s2_status"] == "not_run_calibration_nontransfer"
        assert row["s2_reachability"] is None


def test_cli_renders_every_owner_and_self_seals_manifest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, with_s2=False)
    output = tmp_path / "visual"
    assert subject.main(
        [
            "--s0-root",
            str(fixture["s0"]),
            "--plan-dir",
            str(fixture["plan"]),
            "--s1-analysis-dir",
            str(fixture["s1"]),
            "--output-dir",
            str(output),
        ]
    ) == 0

    manifest = json.loads((output / subject.MANIFEST_NAME).read_text(encoding="utf-8"))
    seal = manifest.pop("manifest_content_sha256")
    assert subject.sha256_json(manifest) == seal
    assert manifest["owner_count"] == 46
    assert manifest["score_based_case_selection"] is False
    assert manifest["s2_attached"] is False
    assert set(subject.OWNER_MAP_NAMES.values()) <= set(manifest["output_files"])
    expected_pages = math.ceil(46 / subject.OWNERS_PER_CROP_PANEL)
    assert len(list(output.glob("owner-crops-page-*.png"))) == expected_pages
    assert len(manifest["owners"]) == 46
    for name, descriptor in manifest["output_files"].items():
        assert subject.sha256_file(output / name) == descriptor["sha256"]


def test_s1_tamper_fails_before_rendering(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, with_s2=False)
    summaries = fixture["s1"] / "owner-summaries.jsonl"
    summaries.write_bytes(summaries.read_bytes() + b'{}\n')
    with pytest.raises(subject.VisualContractError, match="digest mismatch"):
        subject.load_artifacts(fixture["s0"], fixture["plan"], fixture["s1"])


def test_optional_s2_must_cover_exact_resolved_cohort(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, with_s2=True)
    records_path = fixture["s2"] / s2.OWNER_RECORDS_NAME
    rows = subject._read_jsonl(records_path, "fixture records")[:-1]
    _write_jsonl(records_path, rows)
    receipt_path = fixture["s2"] / s2.RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["output_file_sha256"][s2.OWNER_RECORDS_NAME] = subject.sha256_file(records_path)
    receipt.pop("receipt_content_sha256")
    _seal(receipt)
    _write_json(receipt_path, receipt)
    with pytest.raises(subject.VisualContractError, match="exactly equal"):
        subject.load_artifacts(
            fixture["s0"],
            fixture["plan"],
            fixture["s1"],
            s2_analysis_dir=fixture["s2"],
        )
