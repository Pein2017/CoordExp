from __future__ import annotations

from pathlib import Path

import pytest

from scripts.research import audit_sorted_image2299_legacy_transfer_context as subject


IMAGE_RESULTS = {
    "1584": ("confirmation", 8, 9),
    "2685": ("discovery", 7, 10),
    "4134": ("confirmation", 12, 12),
    "5001": ("discovery", 12, 13),
    "6040": ("discovery", 7, 9),
    "7511": ("discovery", 6, 6),
    "10707": ("discovery", 10, 12),
    "13348": ("confirmation", 5, 5),
    "13923": ("confirmation", 11, 11),
    "14038": ("discovery", 13, 20),
    "14439": ("confirmation", 17, 19),
    "16228": ("confirmation", 14, 15),
}


def _control_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for image_id, (split, supported, total) in IMAGE_RESULTS.items():
        for index in range(total):
            category = "tie" if image_id == "4134" and index == 0 else "person"
            rows.append(
                {
                    "schema_version": subject.ROW_SCHEMA_VERSION,
                    "split": split,
                    "image_id": image_id,
                    "gt_owner_id": f"gt:{image_id}:{index}",
                    "normalized_description": category,
                    "supported_under_both_bounds": index < supported,
                }
            )
    assert len(rows) == subject.EXPECTED_POOLED_TP_COUNT
    return rows


def _s1_analysis() -> dict[str, object]:
    return {
        "schema_version": "sorted-image2299-owner-accessibility-analysis.v1",
        "unit_id": subject.CURRENT_UNIT_ID,
        "image_id": subject.IMAGE_ID,
        "calibration_transfer": {
            "supported_due_boundary_count": 14,
            "transfer_denominator_native_tp_count": 19,
            "support_rate": 14 / 19,
            "floor": 0.8,
            "passes": False,
            "status": "calibration_nontransferring",
        },
    }


def _inputs(tmp_path: Path) -> subject.Inputs:
    return subject.Inputs(
        legacy_root=tmp_path / "legacy",
        s1_analysis_dir=tmp_path / "s1-analysis",
        input_file_sha256={"fixture": "0" * 64},
        calibration={},
        owners=[],
        native_sidecars=[],
        contexts_by_split={},
        summaries_by_split={},
        image2299_analysis=_s1_analysis(),
    )


def test_reports_legacy_context_and_strict_claim_boundary() -> None:
    report = subject.build_report(_control_rows(), _s1_analysis())

    assert report["image2299_transfer_assertion"]["supported"] == 14
    assert report["image2299_transfer_assertion"]["total"] == 19
    assert report["legacy_phase_rates"] == {
        "discovery": {"supported": 55, "total": 70, "rate": 55 / 70},
        "confirmation": {"supported": 67, "total": 71, "rate": 67 / 71},
        "pooled": {"supported": 122, "total": 141, "rate": 122 / 141},
    }
    context = report["legacy_per_image_context"]
    assert context["rate_range"] == {"minimum": 13 / 20, "maximum": 1.0}
    assert context["count_below_fixed_reference_floor"] == 3
    composition = report["calibration_control_category_composition"]
    assert composition["discovery"]["tie"]["count"] == 0
    assert composition["pooled"]["tie"] == {
        "count": 1,
        "fraction_of_controls": 1 / 141,
        "supported": 1,
    }
    assert report["claim_boundary"] == {
        "descriptive_instrument_context_only": True,
        "retroactive_legacy_regate": False,
        "new_threshold_created": False,
        "false_negative_mechanism_conclusion": False,
        "false_negative_prevalence_conclusion": False,
        "statement": (
            "Descriptive instrument-context only: this is not a retroactive re-gate, "
            "creates no new threshold, and supports no FN mechanism or prevalence conclusion."
        ),
    }


def test_current_14_of_19_is_an_assertion() -> None:
    analysis = _s1_analysis()
    analysis["calibration_transfer"]["supported_due_boundary_count"] = 15

    with pytest.raises(subject.AuditContractError, match="14/19 assertion drifted"):
        subject.build_report(_control_rows(), analysis)


def test_publication_is_atomic_create_or_identical(tmp_path: Path) -> None:
    rows = _control_rows()
    inputs = _inputs(tmp_path)
    report = subject.build_report(rows, inputs.image2299_analysis)
    analyzer = tmp_path / "analyzer.py"
    analyzer.write_text("# fixture\n", encoding="utf-8")
    files = subject.materialize_files(report, rows, inputs, analyzer)
    output_dir = tmp_path / "audit"

    assert subject.publish_create_or_identical(output_dir, files) == "created"
    assert subject.publish_create_or_identical(output_dir, files) == "identical_existing_output"
    assert {path.name for path in output_dir.iterdir()} == subject.OUTPUT_NAMES

    (output_dir / subject.REPORT_MD_NAME).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(subject.AuditContractError, match="not byte-identical"):
        subject.publish_create_or_identical(output_dir, files)
