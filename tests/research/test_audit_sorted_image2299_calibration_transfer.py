from __future__ import annotations

from pathlib import Path

import pytest

from scripts.research import audit_sorted_image2299_calibration_transfer as subject
from scripts.research import build_sorted_owner_accessibility_census_plan as planner


def _support_block(*, clears: bool, rank: int, population: int) -> dict[str, object]:
    return {
        "peak_lift": 3.0 if clears else 1.0,
        "local_concentration": 3.0,
        "rank": rank,
        "unique_population_size": population,
        "clears_frozen_support_rule": clears,
    }


def _fixture_inputs(tmp_path: Path) -> subject.LoadedInputs:
    owners: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    contexts: dict[str, dict[str, object]] = {}
    due_rows: list[dict[str, object]] = []
    tp_ids = {*(f"gt:2299:{index}" for index in range(16)), "gt:2299:38", "gt:2299:39", "gt:2299:40"}
    supported_ids = {
        *(f"gt:2299:{index}" for index in range(13)),
        "gt:2299:38",
    }
    for index in range(46):
        owner_id = f"gt:2299:{index}"
        category = "person" if index < 38 else "tie"
        x1 = 10 + index * 20
        box = [x1, 10, x1 + 10, 20] if category == "person" else [x1, 500, x1 + 10, 510]
        owners.append(
            {
                "gt_owner_id": owner_id,
                "normalized_description": category,
                "bbox_pixel_xyxy": box,
            }
        )
        native_tp = owner_id in tp_ids
        summary = {
            "schema_version": subject.SOURCE_OWNER_SCHEMA_VERSION,
            "gt_owner_id": owner_id,
            "normalized_description": category,
            "native_true_positive": native_tp,
            "native_false_negative": not native_tp,
            "disposition": (
                "native_true_positive_transfer_control"
                if native_tp
                else subject.DISPOSITION_WITHHELD
            ),
            "disposition_role": subject.DISPOSITION_WITHHELD,
            "fn_disposition_interpretable": False,
            "frozen_disposition_descriptive": (
                "resolved_tested_localization_support"
                if index % 2
                else "persistent_no_tested_localization_support"
            ),
        }
        summaries.append(summary)
        if not native_tp:
            continue
        context_id = f"2299:boundary-{index:03d}"
        supported = owner_id in supported_ids
        due_rows.append(
            {
                "gt_owner_id": owner_id,
                "due_context_id": context_id,
                "eligible": True,
                "supported_under_both_bounds": supported,
            }
        )
        contexts[f"{owner_id}@{context_id}"] = {
            "schema_version": subject.SOURCE_CONTEXT_SCHEMA_VERSION,
            "owner_context_id": f"{owner_id}@{context_id}",
            "gt_owner_id": owner_id,
            "context_id": context_id,
            "localization": {
                "generator_local_max_excluding_other_owner_strict": {
                    "ambiguity_excluded_l": _support_block(
                        clears=supported, rank=1 if supported else 2, population=38
                    ),
                    "ambiguity_included_u": _support_block(
                        clears=supported, rank=1 if supported else 2, population=38
                    ),
                }
            },
            "owner_competition_l": {"rank": 1 if supported else 2, "population_size": 38},
            "owner_competition_u": {"rank": 1 if supported else 2, "population_size": 38},
            "proposal_surface": {
                "category_routing_event": {
                    "within_context_rank": 1,
                    "within_context_population": 2,
                    "raw_sequence_logprob_sum": -0.1,
                },
                "boundary_gate": {
                    "continue_logprob": -0.1,
                    "stop_logprob": -2.0,
                    "continue_vs_stop_logprob_margin": 1.9,
                },
            },
        }

    canvas = planner.Canvas(1000, 1000)
    box_by_category = {
        "person": owners[0]["bbox_pixel_xyxy"],
        "tie": owners[38]["bbox_pixel_xyxy"],
    }
    sidecars: list[dict[str, object]] = []
    for category in ("person", "tie"):
        bins = canvas.pixel_to_bins(box_by_category[category])
        assert bins is not None
        for index in range(25):
            sidecars.append(
                {
                    "schema_version": subject.FREE_BOX_SCHEMA_VERSION,
                    "row_kind": subject.BOX_ROW_KIND,
                    "sidecar_id": f"free-box:{category}:{index}",
                    "normalized_description": category,
                    "channel": "query_suffix",
                    "is_sidecar": True,
                    "is_behavior_not_probability": True,
                    "enters_core_ranks": False,
                    "generation_phase_after_decision_scoring": True,
                    "well_formed_box": True,
                    "coord_bins": list(bins),
                }
            )

    analysis = {
        "schema_version": subject.SOURCE_ANALYSIS_SCHEMA_VERSION,
        "unit_id": subject.UNIT_ID,
        "image_id": subject.IMAGE_ID,
        "denominators": {
            "image2299_owner_count": 46,
            "image2299_native_tp_count": 19,
            "image2299_native_fn_count": 27,
            "legacy_12_owner_count_unchanged": 346,
            "legacy_12_eligible_native_fn_denominator_unchanged": 202,
            "pooled_13_image_denominator_created": False,
        },
        "calibration": {
            "content_sha256": subject.FROZEN_CALIBRATION_CONTENT_SHA256,
            "theta_peak_lift": 2.0,
            "theta_local_concentration": 2.0,
            "epsilon": 0.1,
            "thresholds_retuned": False,
            "phenotype_fitted": False,
        },
        "calibration_transfer": {
            "supported_due_boundary_count": 14,
            "transfer_denominator_native_tp_count": 19,
            "support_rate": 14 / 19,
            "floor": 0.8,
            "passes": False,
            "status": "calibration_nontransferring",
            "on_failure": "withhold_all_native_fn_dispositions",
            "owner_rows": due_rows,
        },
        "fn_disposition_counts": {subject.DISPOSITION_WITHHELD: 27},
    }
    sources = subject.SourceDirs(
        s0_native=tmp_path / "s0-native",
        s1_plan=tmp_path / "s1-plan",
        s1_shard=tmp_path / "s1-shard",
        s1_analysis=tmp_path / "s1-analysis",
    )
    return subject.LoadedInputs(
        sources=sources,
        input_file_sha256={},
        analysis=analysis,
        owner_summaries=summaries,
        owner_contexts=contexts,
        owners=owners,
        image={"image_id": "2299", "image_width": 1000, "image_height": 1000},
        free_box_sidecars=sidecars,
    )


def test_seals_nontransfer_sensitivity_and_descriptive_sidecar_boundary(
    tmp_path: Path,
) -> None:
    inputs = _fixture_inputs(tmp_path)

    result = subject.build_audit(inputs)
    report = result["report"]

    assert report["primary_conclusion"] == {
        "calibration_transfer": {
            "supported": 14,
            "total": 19,
            "rate": 14 / 19,
            "frozen_floor": 0.8,
            "passes": False,
            "status": "calibration_nontransferring",
        },
        "native_fn_dispositions_withheld": 27,
        "s2_gate_open": False,
        "s3_gate_open": False,
        "action": "stop_after_s1_transfer_audit",
    }
    assert report["post_hoc_tp_category_sensitivity"]["person"]["supported"] == 13
    assert report["post_hoc_tp_category_sensitivity"]["tie"]["supported"] == 1
    assert len(report["failed_tp_exact_due_boundary_details"]) == 5
    assert all(
        row["ambiguity_excluded_l"]["failed_frozen_criteria"] == ["peak_lift"]
        for row in report["failed_tp_exact_due_boundary_details"]
    )
    assert len(result["fn_rows"]) == 27
    assert all(row["disposition_role"] == subject.FN_ROLE for row in result["fn_rows"])
    assert all(row["valid_visual_recall_rate"] is False for row in result["fn_rows"])
    reachability = report["free_query_suffix_sidecar_reachability"]
    assert reachability["sidecar_count"] == 50
    assert reachability["unique_strict_owner_hit_count"] == 2
    assert reachability["owner_level_breakdown"]["native_tp"]["owners_with_any_hit"] == 2
    assert reachability["owner_level_breakdown"]["native_fn"]["owners_with_any_hit"] == 0
    assert report["claim_boundary"]["sidecar_misses_imply_absent_visual_support"] is False


def test_publication_is_atomic_create_or_identical(tmp_path: Path) -> None:
    inputs = _fixture_inputs(tmp_path)
    result = subject.build_audit(inputs)
    analyzer = tmp_path / "analyzer.py"
    analyzer.write_text("# fixture\n", encoding="utf-8")
    files = subject.materialize_files(result, inputs, analyzer)
    output_dir = tmp_path / "audit"

    assert subject.publish_create_or_identical(output_dir, files) == "created"
    assert subject.publish_create_or_identical(output_dir, files) == "identical_existing_output"
    assert set(path.name for path in output_dir.iterdir()) == subject.OUTPUT_NAMES

    (output_dir / subject.REPORT_MD_NAME).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(subject.AuditContractError, match="not byte-identical"):
        subject.publish_create_or_identical(output_dir, files)
