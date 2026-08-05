from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys

import pytest

from scripts.research import analyze_sorted_image2299_owner_accessibility as subject
from scripts.research import build_sorted_image2299_owner_accessibility_plan as planner2299
from scripts.research import build_sorted_owner_accessibility_census_plan as legacy
from scripts.research import score_sorted_owner_accessibility_census_shard as scorer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_build_sorted_image2299_owner_accessibility_plan import _fixture_sources  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(legacy.canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(
        b"".join(legacy.canonical_json_bytes(row) + b"\n" for row in rows)
    )


def _build_plan(
    tmp_path: Path,
    *,
    prediction_count: int = 2,
    unavailable_first_due_boundary: bool = False,
) -> tuple[Path, scorer.PlanBundle]:
    plan = planner2299.build_plan(
        _fixture_sources(tmp_path, prediction_count=prediction_count)
    )
    if unavailable_first_due_boundary:
        root = next(row for row in plan.contexts if int(row["boundary_index"]) == 0)
        root["loop_marking"]["loop_tail"] = True
        plan.receipt.pop("receipt_content_sha256")
        plan.receipt["output_file_digests"] = {
            name: hashlib.sha256(content).hexdigest()
            for name, content in sorted(plan.files().items())
        }
        plan.receipt["receipt_content_sha256"] = legacy.sha256_json(plan.receipt)
    path = tmp_path / "plan"
    planner2299.commit_plan(plan, path)
    return path, scorer.load_plan(path)


def _candidate_value(candidate: dict[str, object]) -> float:
    exact = any(
        generator["logical_transform_role"] == "exact_gt_anchor"
        for generator in candidate["generators"]
    )
    jitter = int(str(candidate["candidate_id"]).split(":", 1)[1][:6], 16) / 0xFFFFFF
    return (10.0 if exact else 0.0) + jitter


def _build_complete_shard(tmp_path: Path, plan: scorer.PlanBundle) -> Path:
    shard = tmp_path / "shard"
    shard.mkdir()
    score_rows: list[dict[str, object]] = []
    for group_id, group in sorted(plan.query_groups.items()):
        if group["status"] != "admitted":
            continue
        candidates = [plan.candidates[str(value)] for value in group["candidate_ids"]]
        ordered = sorted(
            candidates,
            key=lambda row: (-_candidate_value(dict(row)), str(row["candidate_id"])),
        )
        ranks = {str(row["candidate_id"]): index + 1 for index, row in enumerate(ordered)}
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            score_rows.append(
                {
                    "schema_version": scorer.SCORE_SCHEMA_VERSION,
                    "row_contract": scorer.P0_ROW_CONTRACT,
                    "unit_id": legacy.UNIT_ID,
                    "plan_schema_version": legacy.PLAN_SCHEMA_VERSION,
                    "plan_receipt_content_sha256": plan.receipt_content_sha256,
                    "capture_rules_sha256": plan.capture_rules_sha256,
                    "channel": legacy.CHANNEL_QUERY_SUFFIX,
                    "image_id": "2299",
                    "context_id": group["context_id"],
                    "query_group_id": group_id,
                    "normalized_description": group["normalized_description"],
                    "candidate_id": candidate_id,
                    "complete_box_logprob_sum": _candidate_value(dict(candidate)),
                    "competition": {
                        "population_size": len(candidates),
                        "rank": ranks[candidate_id],
                        "population": "collapsed_unique_physical_candidates_only",
                    },
                    "request_id": f"{group_id}|{candidate_id}",
                }
            )
    _write_jsonl(shard / subject.SCORES_NAME, score_rows)

    proposal_rows: list[dict[str, object]] = []
    categories = sorted(
        {
            str(row["normalized_description"])
            for row in plan.categories.values()
            if row["status"] == "admitted"
        }
    )
    for context in sorted(plan.contexts.values(), key=lambda row: int(row["boundary_index"])):
        proposal_rows.append(
            {
                "schema_version": scorer.PROPOSAL_SCHEMA_VERSION,
                "row_contract": scorer.P0_ROW_CONTRACT,
                "unit_id": legacy.UNIT_ID,
                "plan_receipt_content_sha256": plan.receipt_content_sha256,
                "capture_rules_sha256": plan.capture_rules_sha256,
                "image_id": "2299",
                "context_id": context["context_id"],
                "emits_per_owner_proposal_probability": False,
                "includes_coordinate_scores": False,
                "boundary_gate": {
                    "continue_logprob": -0.1,
                    "stop_logprob": -2.0,
                    "continue_vs_stop_logprob_margin": 1.9,
                },
                "category_routing_event": [
                    {
                        "normalized_description": description,
                        "raw_sequence_logprob_sum": -float(index + 1),
                        "within_context_rank": index + 1,
                        "within_context_population": len(categories),
                        "row_prefix_block_raw_sequence_logprob_sum": -float(index + 1),
                    }
                    for index, description in enumerate(categories)
                ],
            }
        )
    _write_jsonl(shard / subject.PROPOSAL_NAME, proposal_rows)
    _write_jsonl(shard / scorer.X1_NAME, [])
    _write_jsonl(shard / scorer.FREE_DECODE_NAME, [])
    receipt = {
        "schema_version": scorer.SHARD_RECEIPT_SCHEMA_VERSION,
        "unit_id": legacy.UNIT_ID,
        "image_id": "2299",
        "status": "captured",
        "capture_completeness": "complete_shard",
        "subset_capture": {
            "is_subset": False,
            "usable_as_complete_shard_evidence": True,
        },
        "plan": {
            "receipt_content_sha256": plan.receipt_content_sha256,
            "capture_rules_sha256": plan.capture_rules_sha256,
        },
        "checks": {
            "all_finite": True,
            "canonical_suffix_verified": True,
            "coordinate_domain_ok": True,
            "cross_owner_tuple_collapse_verified": True,
            "every_row_bound_to_an_admission_receipt": True,
        },
        "backend_identity": {
            "backend": "hf",
            "is_real_model": True,
            "executed_media_sha256": plan.images["2299"]["executed_media_sha256"],
        },
        "counts": {
            "localization_score_rows": len(score_rows),
            "proposal_surface_rows": len(proposal_rows),
        },
    }
    _write_json(shard / subject.SHARD_RECEIPT_NAME, receipt)
    return shard


def test_applies_frozen_calibration_and_reports_underpowered_transfer(tmp_path: Path) -> None:
    plan_dir, plan = _build_plan(tmp_path)
    shard_dir = _build_complete_shard(tmp_path, plan)

    product = subject.analyze(plan_dir, shard_dir)
    output = tmp_path / "analysis"
    subject.commit_analysis(product, output)

    assert product.analysis["denominators"]["image2299_owner_count"] == 46
    assert product.analysis["denominators"]["legacy_12_owner_count_unchanged"] == 346
    assert product.analysis["denominators"]["pooled_13_image_denominator_created"] is False
    assert product.analysis["calibration"]["content_sha256"] == subject.CALIBRATION_CONTENT_SHA256
    assert product.analysis["calibration"]["thresholds_retuned"] is False
    assert product.analysis["calibration"]["phenotype_fitted"] is False
    transfer = product.analysis["calibration_transfer"]
    assert transfer["native_tp_owner_count"] == 2
    assert transfer["status"] == "calibration_transfer_underpowered"
    assert transfer["passes"] is None
    assert transfer["validity_bearing"] is False
    assert len(product.owner_summaries) == 46
    assert all(
        row["disposition_role"]
        in {
            "descriptive_only_calibration_transfer_underpowered",
            "validity_bearing",
        }
        for row in product.owner_summaries
    )
    assert all(
        row["proposal_surface"]["separate_from_localization"] is True
        for row in product.owner_contexts
    )
    assert set(path.name for path in output.iterdir()) == {
        "analysis.json",
        "owner-summaries.jsonl",
        "owner-context-features.jsonl",
        "context-registry.jsonl",
        "receipt.json",
    }
    assert subject.commit_analysis(product, output)["receipt.json"]


def test_rejects_calibration_drift(tmp_path: Path) -> None:
    receipt = json.loads(subject.CALIBRATION_PATH.read_text())
    receipt["theta_peak_lift"] = float(receipt["theta_peak_lift"]) + 0.1
    drifted = tmp_path / "support-calibration.json"
    _write_json(drifted, receipt)
    with pytest.raises(subject.AnalysisContractError, match="calibration file digest mismatch"):
        subject.load_frozen_calibration(drifted)


def test_transfer_floor_is_validity_bearing_with_ten_controls(tmp_path: Path) -> None:
    plan_dir, plan = _build_plan(tmp_path, prediction_count=10)
    shard_dir = _build_complete_shard(tmp_path, plan)
    product = subject.analyze(plan_dir, shard_dir)
    transfer = product.analysis["calibration_transfer"]

    assert transfer["native_tp_owner_count"] == 10
    assert transfer["validity_bearing"] is True
    assert transfer["passes"] is True
    assert transfer["support_rate"] >= 0.80
    assert all(
        row["fn_disposition_interpretable"] is True
        for row in product.owner_summaries
        if row["native_false_negative"]
    )


def test_cross_checkpoint_policy_keeps_thresholds_descriptive(tmp_path: Path) -> None:
    plan_dir, plan = _build_plan(tmp_path, prediction_count=10)
    shard_dir = _build_complete_shard(tmp_path, plan)
    product = subject.analyze(
        plan_dir,
        shard_dir,
        force_descriptive_only=True,
    )

    transfer = product.analysis["calibration_transfer"]
    assert transfer["passes"] is True
    assert transfer["supports_binary_fn_classification"] is False
    assert product.analysis["classification_policy"] == (
        "cross_checkpoint_sensitivity_only"
    )
    false_negatives = [
        row for row in product.owner_summaries if row["native_false_negative"]
    ]
    assert false_negatives
    assert all(
        row["disposition"] == subject.DISPOSITION_WITHHELD
        and row["fn_disposition_interpretable"] is False
        and row["disposition_role"] == "cross_checkpoint_sensitivity_only"
        for row in false_negatives
    )


def test_cli_reports_all_native_tp_denominator_and_separate_eligible_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan_dir, plan = _build_plan(
        tmp_path,
        prediction_count=10,
        unavailable_first_due_boundary=True,
    )
    shard_dir = _build_complete_shard(tmp_path, plan)
    product = subject.analyze(plan_dir, shard_dir)
    transfer = product.analysis["calibration_transfer"]

    assert transfer["transfer_denominator_native_tp_count"] == 10
    assert transfer["eligible_due_boundary_count"] == 9
    assert transfer["supported_due_boundary_count"] == 9
    assert transfer["support_rate"] == pytest.approx(0.9)

    output = tmp_path / "analysis-cli"
    assert (
        subject.main(
            [
                "--plan-dir",
                str(plan_dir),
                "--scored-shard-dir",
                str(shard_dir),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out
    assert (
        "frozen-support transfer: 9/10 native-TP = 0.900; "
        "eligible_due_boundaries=9; passes=True"
    ) in stdout


def test_rejects_denominator_drift_before_reading_scores(tmp_path: Path) -> None:
    plan_dir, _ = _build_plan(tmp_path)
    receipt_path = plan_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["denominator_contract"]["legacy_12_eligible_native_fn_denominator_unchanged"] = 203
    receipt["receipt_content_sha256"] = legacy.sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    _write_json(receipt_path, receipt)
    with pytest.raises(subject.AnalysisContractError, match="denominator contract drifted"):
        subject.analyze(plan_dir, tmp_path / "missing-shard")


def test_threshold_retuning_has_no_api_or_cli_surface() -> None:
    assert "theta_peak_lift" not in inspect.signature(subject.analyze).parameters
    assert "theta_local_concentration" not in inspect.signature(subject.analyze).parameters
    with pytest.raises(SystemExit):
        subject._parse_args(
            [
                "--plan-dir", "plan",
                "--scored-shard-dir", "shard",
                "--output-dir", "out",
                "--theta-peak-lift", "0",
            ]
        )


@pytest.mark.parametrize("contamination", ["missing_file", "foreign_subdir"])
def test_commit_analysis_requires_exact_existing_artifact_set(
    tmp_path: Path, contamination: str
) -> None:
    plan_dir, plan = _build_plan(tmp_path)
    shard_dir = _build_complete_shard(tmp_path, plan)
    product = subject.analyze(plan_dir, shard_dir)
    output = tmp_path / "analysis"
    subject.commit_analysis(product, output)

    if contamination == "missing_file":
        (output / "receipt.json").unlink()
    else:
        (output / "foreign").mkdir()

    with pytest.raises(subject.AnalysisContractError, match="artifact set is not exact"):
        subject.commit_analysis(product, output)

    if contamination == "missing_file":
        assert not (output / "receipt.json").exists()
    else:
        assert (output / "foreign").is_dir()
