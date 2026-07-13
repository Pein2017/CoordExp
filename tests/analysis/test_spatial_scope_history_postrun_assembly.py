from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from src.analysis.spatial_scope_history.calibration import PrimaryScheduleArtifact
from src.analysis.spatial_scope_history.cohort_ledger import (
    AttemptLedger,
    canonical_json_text,
    sha256_file,
    sha256_payload,
)
from src.analysis.spatial_scope_history.metrics import RowMetricRecord, exact_reference_match
from src.analysis.spatial_scope_history import postrun_assembler as postrun_assembler_module
from src.analysis.spatial_scope_history.postrun_assembler import (
    POSTRUN_SOURCE_RELATIVE_PATHS,
    _build_metric_records,
    _derive_postrun_source_identity,
    assemble_supported_postrun_metrics,
    write_supported_postrun_assembly,
)
from src.analysis.spatial_scope_history.postrun_loader import load_postrun_evidence
from src.common.errors import ArtifactContractError, DataContractError
from test_spatial_scope_history_metrics import (
    _metric_admission_fixture,
    _prediction,
    _reference,
)


def _write_input_artifacts(tmp_path: Path):
    _, schedule, cohort, _, attempt_ledger, _ = _metric_admission_fixture(tmp_path)
    schedule_path = tmp_path / "schedule.json"
    cohort_path = tmp_path / "cohort.jsonl"
    attempt_path = tmp_path / "attempts.jsonl"
    cohort_path.write_bytes(cohort.to_jsonl_bytes())
    cohort_sha256 = sha256_file(cohort_path)
    artifact = PrimaryScheduleArtifact(
        schedule=schedule,
        cohort_artifact_name="cohort-manifest.jsonl",
        cohort_artifact_sha256=cohort_sha256,
        readiness_ledger_seal_sha256=sha256_file(
            tmp_path / "readiness-v2" / "ledger-seal.json"
        ),
        calibration_selection_receipt_sha256=sha256_payload(
            {"fixture": "calibration-selection-receipt"}
        ),
        calibration_selection_fingerprint=sha256_payload(
            {"fixture": "calibration-selection"}
        ),
        source_runtime_identity_receipt_sha256=sha256_payload(
            {"fixture": "source-runtime-identity-receipt"}
        ),
        source_runtime_identity_fingerprint=sha256_payload(
            {"fixture": "source-runtime-identity"}
        ),
        source_hashes=(("cohort-manifest.jsonl", cohort_sha256),),
    )
    schedule_path.write_text(
        canonical_json_text(artifact.to_artifact_dict()) + "\n",
        encoding="utf-8",
    )
    attempt_path.write_bytes(attempt_ledger.to_jsonl_bytes())
    return schedule_path, cohort_path, attempt_path, tmp_path / "readiness-v2"


def _committed_postrun_source_repo(tmp_path: Path) -> tuple[Path, str]:
    repository_root = tmp_path / "source-repository"
    live_root = Path(__file__).resolve().parents[2]
    for relative_path in POSTRUN_SOURCE_RELATIVE_PATHS:
        destination = repository_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((live_root / relative_path).read_bytes())
    subprocess.run(("git", "init", "-q", str(repository_root)), check=True)
    subprocess.run(
        ("git", "-C", str(repository_root), "config", "user.email", "test@example.com"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(repository_root), "config", "user.name", "Test User"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(repository_root), "add", "--", *POSTRUN_SOURCE_RELATIVE_PATHS),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(repository_root), "commit", "-q", "-m", "fixture"),
        check=True,
    )
    head = subprocess.run(
        ("git", "-C", str(repository_root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repository_root, head


def test_full_five_arm_failure_universe_assembles_zero_prediction_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_input_artifacts(tmp_path)
    evidence = load_postrun_evidence(
        schedule_path=paths[0],
        cohort_path=paths[1],
        attempt_ledger_path=paths[2],
        readiness_root=paths[3],
    )

    assert len(evidence.calls) == len(evidence.schedule.requests) == 260
    assert {call.attempt.attempt_status for call in evidence.calls} == {
        "failed",
        "skipped",
    }
    assert all(not call.normalized_predictions for call in evidence.calls)

    assembly = assemble_supported_postrun_metrics(evidence)

    assert len(assembly.arm_merges) == 5
    assert len(assembly.image_metric_primitives) == 80
    assert len(assembly.aggregate_reports) == 20
    assert len(assembly.bootstrap_reports) == 276
    assert {report.threshold for report in assembly.aggregate_reports} == {
        0.50,
        0.75,
    }
    assert {
        report.point_estimate.threshold for report in assembly.bootstrap_reports
    } == {0.50, 0.75}
    assert all(
        f"intersection_over_union_{report.threshold:.2f}" in report.scope
        for report in assembly.aggregate_reports
    )
    assert all(
        f"intersection_over_union_{report.point_estimate.threshold:.2f}"
        in report.point_estimate.scope
        for report in assembly.bootstrap_reports
    )
    assert all(
        not result.post_merge.predictions
        for arm in assembly.arm_merges
        for result in arm.image_results
    )

    source_repository, source_commit = _committed_postrun_source_repo(tmp_path)
    monkeypatch.setattr(
        postrun_assembler_module,
        "_postrun_repository_root",
        lambda: source_repository,
    )
    with pytest.raises(DataContractError, match="source_commit_mismatch"):
        write_supported_postrun_assembly(
            assembly,
            output_root=tmp_path / "false-commit",
            source_commit="a" * 40,
        )
    output_root = tmp_path / "assembled"
    receipt = write_supported_postrun_assembly(
        assembly,
        output_root=output_root,
        source_commit=source_commit,
    )
    assert receipt["status"] == "complete"
    assert receipt["primary_schedule_artifact_path"] == str(paths[0])
    assert receipt["primary_schedule_artifact_file_sha256"] == sha256_file(paths[0])
    assert receipt["analysis_source_identity"]["git_head_commit"] == source_commit
    assert set(receipt["analysis_source_identity"]["source_file_sha256s"]) == set(
        POSTRUN_SOURCE_RELATIVE_PATHS
    )
    assert set(receipt["output_artifact_sha256s"]) == {
        "aggregate-metric-reports.json",
        "arm-merge-results.json",
        "bootstrap-metric-reports.json",
        "image-metric-primitives.json",
    }
    for filename, expected_sha256 in receipt["output_artifact_sha256s"].items():
        assert sha256_file(output_root / filename) == expected_sha256
    with pytest.raises(DataContractError, match="postrun_output_exists"):
        write_supported_postrun_assembly(
            assembly,
            output_root=output_root,
            source_commit=source_commit,
        )
    dirty_path = (
        source_repository / "src/analysis/spatial_scope_history/metrics.py"
    )
    dirty_path.write_bytes(dirty_path.read_bytes() + b"\n")
    with pytest.raises(DataContractError, match="postrun_source_dirty"):
        _derive_postrun_source_identity(source_commit)


def test_loader_rejects_raw_research_schedule(tmp_path: Path) -> None:
    paths = _write_input_artifacts(tmp_path)
    loaded = load_postrun_evidence(
        schedule_path=paths[0],
        cohort_path=paths[1],
        attempt_ledger_path=paths[2],
        readiness_root=paths[3],
    )
    raw_schedule_path = tmp_path / "raw-research-schedule.json"
    raw_schedule_path.write_text(
        canonical_json_text(loaded.schedule.to_artifact_dict()) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactContractError, match="primary schedule artifact"):
        load_postrun_evidence(
            schedule_path=raw_schedule_path,
            cohort_path=paths[1],
            attempt_ledger_path=paths[2],
            readiness_root=paths[3],
        )


def test_frozen_intersection_over_union_thresholds_separate_boundary_match() -> None:
    prediction = _prediction("boundary-prediction", (16.0, 16.0, 46.0, 64.0))
    reference = _reference("boundary-reference", (16.0, 16.0, 64.0, 64.0))

    assert exact_reference_match((prediction,), (reference,), threshold=0.50).matches
    assert not exact_reference_match(
        (prediction,), (reference,), threshold=0.75
    ).matches


def test_loader_rejects_tampered_terminal_bundle(tmp_path: Path) -> None:
    paths = _write_input_artifacts(tmp_path)
    loaded = load_postrun_evidence(
        schedule_path=paths[0],
        cohort_path=paths[1],
        attempt_ledger_path=paths[2],
        readiness_root=paths[3],
    )
    terminal_path = Path(
        loaded.attempt_ledger.records[0].output_artifact_path or ""
    )
    terminal_path.write_bytes(terminal_path.read_bytes() + b"\n")

    with pytest.raises(DataContractError):
        load_postrun_evidence(
            schedule_path=paths[0],
            cohort_path=paths[1],
            attempt_ledger_path=paths[2],
            readiness_root=paths[3],
        )


def test_loader_rejects_swapped_terminal_artifact_bindings(tmp_path: Path) -> None:
    paths = _write_input_artifacts(tmp_path)
    loaded = load_postrun_evidence(
        schedule_path=paths[0],
        cohort_path=paths[1],
        attempt_ledger_path=paths[2],
        readiness_root=paths[3],
    )
    first, second, *remaining = loaded.attempt_ledger.records
    swapped = AttemptLedger(
        run_id=loaded.attempt_ledger.run_id,
        schedule_sha256=loaded.attempt_ledger.schedule_sha256,
        execution_identity=loaded.attempt_ledger.execution_identity,
        records=(
            replace(
                first,
                output_artifact_path=second.output_artifact_path,
                output_artifact_sha256=second.output_artifact_sha256,
            ),
            replace(
                second,
                output_artifact_path=first.output_artifact_path,
                output_artifact_sha256=first.output_artifact_sha256,
            ),
            *remaining,
        ),
    )
    paths[2].write_bytes(swapped.to_jsonl_bytes())

    with pytest.raises(DataContractError):
        load_postrun_evidence(
            schedule_path=paths[0],
            cohort_path=paths[1],
            attempt_ledger_path=paths[2],
            readiness_root=paths[3],
        )


def test_metric_record_assembly_preserves_raw_owner_and_row_match_lineage() -> None:
    owning = _prediction("prediction-owning", (16.0, 16.0, 64.0, 64.0))
    non_owning = _prediction(
        "prediction-non-owning",
        (72.0, 72.0, 96.0, 96.0),
        row_index=1,
    )
    object.__setattr__(owning, "ownership_status", "owned")
    object.__setattr__(non_owning, "ownership_status", "non_owning")
    rows = (
        RowMetricRecord(
            canonical_call_id="call-00",
            generated_row_index=0,
            prediction_id="prediction-owning",
            parse_status="parsed",
            validity_status="valid",
            ownership_status="owned",
            matched_reference_ids=(),
        ),
        RowMetricRecord(
            canonical_call_id="call-00",
            generated_row_index=1,
            prediction_id="prediction-non-owning",
            parse_status="parsed",
            validity_status="valid",
            ownership_status="non_owning",
            matched_reference_ids=(),
        ),
    )
    diagnostics = {
        "attempted_row_count": 2,
        "controller_cap_count": 0,
        "error_count": 0,
        "generated_token_count": 2,
        "image_token_count": 16,
        "invalid_row_count": 0,
        "malformed_row_count": 0,
        "natural_closure_count": 1,
        "non_owning_prediction_count": 1,
        "peak_device_memory_bytes": 0,
        "prompt_token_count": 4,
        "token_cap_count": 0,
        "valid_prediction_ids": ["prediction-non-owning", "prediction-owning"],
        "wall_time_seconds": 0.1,
    }
    call = SimpleNamespace(
        request=SimpleNamespace(
            request_id="call-00",
            cell_index=0,
            sampling_seed=1,
            image_sha256="1" * 64,
            image_frozen_order=0,
        ),
        attempt=SimpleNamespace(
            attempt_status="completed",
            output_artifact_sha256="2" * 64,
            output_artifact_path="/tmp/terminal.json",
            failure_code=None,
        ),
        normalized_predictions=(owning, non_owning),
        row_diagnostics=rows,
        call_diagnostics=diagnostics,
    )
    references = (
        _reference("reference-owning", (16.0, 16.0, 64.0, 64.0)),
        _reference("reference-non-owning", (72.0, 72.0, 96.0, 96.0)),
    )

    call_records, row_records = _build_metric_records(
        calls=(call,),
        references=references,
        threshold=0.50,
    )

    assert call_records[0].raw_any_call_matched_reference_ids == (
        "reference-non-owning",
        "reference-owning",
    )
    assert call_records[0].raw_owning_call_matched_reference_ids == (
        "reference-owning",
    )
    assert row_records[0].matched_reference_ids == ("reference-owning",)
    assert row_records[1].matched_reference_ids == ("reference-non-owning",)
