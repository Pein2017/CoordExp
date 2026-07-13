"""Minimal post-run merge and metric assembly for the frozen five-arm panel."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from src.analysis.spatial_scope_history.cohort_ledger import (
    canonical_json_text,
    sha256_file,
    sha256_payload,
)
from src.analysis.spatial_scope_history.merge import (
    ArmMergeResult,
    ExpectedImageFrame,
    NormalizedCallPrediction,
    merge_arm_predictions,
)
from src.analysis.spatial_scope_history.metrics import (
    AggregateMetricReport,
    BootstrapMetricReport,
    CallMetricRecord,
    ImageMetricPrimitive,
    ReferenceObject,
    ReferenceLedgerScope,
    RowMetricRecord,
    aggregate_metric_report,
    bootstrap_named_metric_report,
    bootstrap_paired_arm_metric_report,
    build_image_metric_primitive,
    exact_reference_match,
    _is_mask_harm_core_interior,
)
from src.analysis.spatial_scope_history.postrun_loader import (
    LoadedPostrunEvidence,
    LoadedTerminalCall,
)
from src.analysis.spatial_scope_history.schedule import PRIMARY_ARM_CODES
from src.common.errors import DataContractError


POSTRUN_ASSEMBLY_RECEIPT_SCHEMA_VERSION = (
    "spatial_scope_history.postrun_assembly_receipt.v1"
)
POSTRUN_ASSEMBLY_SCHEMA_VERSION = "spatial_scope_history.postrun_assembly.v1"
PRIMARY_THRESHOLDS = (0.50, 0.75)
REFERENCE_SCOPES: tuple[ReferenceLedgerScope, ...] = (
    "official_annotation",
    "audit_augmented",
)
ABSOLUTE_BOOTSTRAP_METRICS = (
    "post_merge_local_rescue_rate",
    "raw_any_call_union_rescue_rate",
    "overall_retention",
    "mask_harm_retention",
    "manual_precision",
    "manual_precision_uncertainty_ignored_as_unmatched",
    "post_merge_strict_duplicate_rate",
    "invalid_row_rate",
    "invalid_call_rate",
    "natural_closure_rate",
    "prediction_count_inflation",
)
OWNING_SEED_RAW_RESCUE_BOOTSTRAP_METRIC = "owning_seed_raw_rescue_difference"
PAIRED_ARM_DIRECTIONS = (
    ("MASK_RESET", "FULL_BAG_K"),
    ("MASK_RESET", "MASK_CUMULATIVE"),
    ("MASK_CUMULATIVE", "MASK_RESET"),
    ("TILE_RESET", "MASK_RESET"),
)
PAIRED_ARM_BOOTSTRAP_METRICS = (
    "post_merge_local_rescue_rate_difference",
    "manual_precision_difference",
    "post_merge_strict_duplicate_rate_difference",
    "invalid_row_rate_difference",
    "invalid_call_rate_difference",
    "natural_closure_rate_difference",
)
POSTRUN_SOURCE_RELATIVE_PATHS = (
    "scripts/research/assemble_spatial_scope_history_metrics.py",
    "src/analysis/spatial_scope_history/execution_evidence.py",
    "src/analysis/spatial_scope_history/merge.py",
    "src/analysis/spatial_scope_history/metrics.py",
    "src/analysis/spatial_scope_history/parse_score_evidence.py",
    "src/analysis/spatial_scope_history/postrun_assembler.py",
    "src/analysis/spatial_scope_history/postrun_loader.py",
    "src/analysis/spatial_scope_history/spatial.py",
)


@dataclass(frozen=True)
class SupportedPostrunAssembly:
    """In-memory products from existing merge and metric formulas only."""

    evidence: LoadedPostrunEvidence
    arm_merges: tuple[ArmMergeResult, ...]
    image_metric_primitives: tuple[ImageMetricPrimitive, ...]
    aggregate_reports: tuple[AggregateMetricReport, ...]
    bootstrap_reports: tuple[BootstrapMetricReport, ...]


def assemble_supported_postrun_metrics(
    evidence: LoadedPostrunEvidence,
) -> SupportedPostrunAssembly:
    """Invoke only the canonical five-arm merge, metric, and bootstrap formulas."""

    root_seed = evidence.schedule.identity.root_seed
    expected_images = tuple(
        ExpectedImageFrame(
            image_id=str(record.image_id),
            source_width=record.source_width,
            source_height=record.source_height,
        )
        for record in evidence.cohort.records
    )
    predictions_by_arm: dict[str, list[NormalizedCallPrediction]] = defaultdict(list)
    for call in evidence.calls:
        predictions_by_arm[call.request.arm.arm_code].extend(
            call.normalized_predictions
        )
    arm_merges = tuple(
        merge_arm_predictions(
            arm_identifier=arm_code,
            predictions=predictions_by_arm[arm_code],
            expected_images=expected_images,
            allowed_arm_identifiers=tuple(PRIMARY_ARM_CODES),
        )
        for arm_code in PRIMARY_ARM_CODES
    )
    merge_by_arm_image = {
        (arm.arm.arm_code, result.image_id): result
        for arm in arm_merges
        for result in arm.image_results
    }
    calls_by_image_arm: dict[tuple[str, str], list[LoadedTerminalCall]] = defaultdict(
        list
    )
    for call in evidence.calls:
        calls_by_image_arm[
            (str(call.request.image_id), call.request.arm.arm_code)
        ].append(call)

    primitives: list[ImageMetricPrimitive] = []
    for ledger_scope in REFERENCE_SCOPES:
        for threshold in PRIMARY_THRESHOLDS:
            for record in evidence.cohort.records:
                image_id = str(record.image_id)
                references = evidence.readiness.reference_objects(
                    image_id=image_id,
                    ledger_scope=ledger_scope,
                )
                references = _spatially_enriched_references(
                    references,
                    evidence=evidence,
                    image_id=image_id,
                )
                for arm_code in PRIMARY_ARM_CODES:
                    calls = tuple(calls_by_image_arm[(image_id, arm_code)])
                    call_records, row_records = _build_metric_records(
                        calls=calls,
                        references=references,
                        threshold=threshold,
                    )
                    primitives.append(
                        build_image_metric_primitive(
                            admission=evidence.admission,
                            merge_result=merge_by_arm_image[(arm_code, image_id)],
                            references=references,
                            ledger_scope=ledger_scope,
                            threshold=threshold,
                            configured_call_budget=len(calls),
                            density_tags=record.density_tags,
                            call_records=call_records,
                            row_records=row_records,
                        )
                    )

    primitive_index = {
        (
            primitive.ledger_scope,
            primitive.threshold,
            primitive.arm_identifier,
            primitive.image_id,
        ): primitive
        for primitive in primitives
    }
    aggregate_reports: list[AggregateMetricReport] = []
    bootstrap_reports: list[BootstrapMetricReport] = []
    for ledger_scope in REFERENCE_SCOPES:
        for threshold in PRIMARY_THRESHOLDS:
            threshold_scope = f"intersection_over_union_{threshold:.2f}"
            baseline = _ordered_primitives(
                primitive_index,
                evidence=evidence,
                ledger_scope=ledger_scope,
                threshold=threshold,
                arm_code="FULL_SINGLE",
            )
            arm_primitives = {
                arm_code: _ordered_primitives(
                    primitive_index,
                    evidence=evidence,
                    ledger_scope=ledger_scope,
                    threshold=threshold,
                    arm_code=arm_code,
                )
                for arm_code in PRIMARY_ARM_CODES
            }
            for arm_code, candidate in arm_primitives.items():
                aggregate_reports.append(
                    aggregate_metric_report(
                        candidate,
                        baseline,
                        scope=f"{ledger_scope}:{threshold_scope}",
                        paired_full_bag=(
                            arm_primitives["FULL_BAG_K"]
                            if arm_code == "MASK_RESET"
                            else None
                        ),
                    )
                )
                if arm_code == "FULL_SINGLE":
                    continue
                for metric_name in ABSOLUTE_BOOTSTRAP_METRICS:
                    bootstrap_reports.append(
                        bootstrap_named_metric_report(
                            candidate,
                            baseline,
                            metric_name=metric_name,
                            scope=f"{ledger_scope}:{threshold_scope}:{arm_code}",
                            root_seed=root_seed,
                        )
                    )
            bootstrap_reports.append(
                bootstrap_named_metric_report(
                    arm_primitives["MASK_RESET"],
                    baseline,
                    metric_name=OWNING_SEED_RAW_RESCUE_BOOTSTRAP_METRIC,
                    scope=(
                        f"{ledger_scope}:{threshold_scope}:"
                        "MASK_RESET_vs_FULL_BAG_K_owning_seed"
                    ),
                    paired_full_bag=arm_primitives["FULL_BAG_K"],
                    root_seed=root_seed,
                )
            )
            for candidate_arm, comparator_arm in PAIRED_ARM_DIRECTIONS:
                for metric_name in PAIRED_ARM_BOOTSTRAP_METRICS:
                    bootstrap_reports.append(
                        bootstrap_paired_arm_metric_report(
                            arm_primitives[candidate_arm],
                            arm_primitives[comparator_arm],
                            baseline,
                            metric_name=metric_name,
                            scope=(
                                f"{ledger_scope}:{threshold_scope}:"
                                f"{candidate_arm}_vs_{comparator_arm}"
                            ),
                            root_seed=root_seed,
                        )
                    )
    return SupportedPostrunAssembly(
        evidence=evidence,
        arm_merges=arm_merges,
        image_metric_primitives=tuple(primitives),
        aggregate_reports=tuple(aggregate_reports),
        bootstrap_reports=tuple(bootstrap_reports),
    )


def write_supported_postrun_assembly(
    assembly: SupportedPostrunAssembly,
    *,
    output_root: str | Path,
    source_commit: str,
) -> Mapping[str, Any]:
    """Write one immutable result root and a receipt over every emitted artifact."""

    root = Path(output_root)
    if not root.is_absolute():
        _fail(
            "post-run output root must be an explicit absolute path",
            "analysis.postrun_output_absolute",
        )
    if root.exists():
        _fail(
            "post-run output root is immutable and must not already exist",
            "analysis.postrun_output_exists",
            output_root=str(root),
        )
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source_commit):
        _fail(
            "source commit must be a lowercase Git object identifier",
            "analysis.postrun_source_commit",
        )
    if sha256_file(Path(assembly.evidence.schedule_path)) != (
        assembly.evidence.primary_schedule_artifact_file_sha256
    ):
        _fail(
            "primary schedule wrapper changed after post-run evidence loading",
            "analysis.postrun_primary_schedule_changed",
        )
    analysis_source_identity = _derive_postrun_source_identity(source_commit)
    payloads = _artifact_payloads(assembly)
    root.mkdir(parents=True, exist_ok=False)
    output_hashes: dict[str, str] = {}
    for name, payload in payloads.items():
        path = root / name
        _write_once(path, payload)
        output_hashes[name] = sha256_file(path)
    evidence = assembly.evidence
    terminal_entries = [
        {
            "bundle_sha256": call.bundle.bundle_sha256,
            "output_artifact_path": call.bundle_path,
            "request_id": call.request.request_id,
        }
        for call in evidence.calls
    ]
    receipt_identity = {
        "attempt_ledger_path": evidence.attempt_ledger_path,
        "attempt_ledger_sha256": sha256_file(Path(evidence.attempt_ledger_path)),
        "code_sha256": evidence.schedule.identity.execution_identity.code_sha256,
        "cohort_fingerprint": evidence.cohort.fingerprint,
        "cohort_path": evidence.cohort_path,
        "cohort_sha256": sha256_file(Path(evidence.cohort_path)),
        "config_sha256": evidence.schedule.identity.execution_identity.config_sha256,
        "estimator_registry": _estimator_registry(),
        "metric_admission_sha256": evidence.admission.fingerprint,
        "output_artifact_sha256s": dict(sorted(output_hashes.items())),
        "readiness": evidence.readiness.to_json_dict(),
        "readiness_root": evidence.readiness_root,
        "primary_schedule_artifact_file_sha256": (
            evidence.primary_schedule_artifact_file_sha256
        ),
        "primary_schedule_artifact_fingerprint": (
            evidence.primary_schedule_artifact.artifact_sha256
        ),
        "primary_schedule_artifact_path": evidence.schedule_path,
        "schedule_fingerprint": evidence.schedule.fingerprint,
        "analysis_source_identity": analysis_source_identity,
        "schema_version": POSTRUN_ASSEMBLY_RECEIPT_SCHEMA_VERSION,
        "source_commit": source_commit,
        "status": "complete",
        "terminal_bundle_set_sha256": sha256_payload(terminal_entries),
        "terminal_bundles": terminal_entries,
    }
    receipt = {
        **receipt_identity,
        "receipt_sha256": sha256_payload(receipt_identity),
    }
    _write_once(
        root / "postrun-assembly-receipt.json",
        (canonical_json_text(receipt) + "\n").encode("utf-8"),
    )
    return receipt


def _build_metric_records(
    *,
    calls: Sequence[LoadedTerminalCall],
    references: Sequence[Any],
    threshold: float,
) -> tuple[tuple[CallMetricRecord, ...], tuple[RowMetricRecord, ...]]:
    call_records: list[CallMetricRecord] = []
    row_records: list[RowMetricRecord] = []
    for call in calls:
        raw = call.normalized_predictions
        raw_owning = tuple(
            prediction
            for prediction in raw
            if prediction.ownership_status != "non_owning"
        )
        raw_match = exact_reference_match(raw, references, threshold=threshold)
        owning_match = exact_reference_match(
            raw_owning, references, threshold=threshold
        )
        matches_by_prediction = {
            match.prediction_id: (match.reference_id,)
            for match in raw_match.matches
        }
        scope_rows = tuple(
            replace(
                row,
                matched_reference_ids=(
                    ()
                    if row.prediction_id is None
                    else matches_by_prediction.get(row.prediction_id, ())
                ),
            )
            for row in call.row_diagnostics
        )
        diagnostics = _validated_call_diagnostics(call)
        call_records.append(
            CallMetricRecord(
                canonical_call_id=call.request.request_id,
                canonical_cell_index=call.request.cell_index,
                sampling_seed=call.request.sampling_seed,
                raw_any_call_matched_reference_ids=raw_match.matched_reference_ids,
                raw_owning_call_matched_reference_ids=(
                    owning_match.matched_reference_ids
                ),
                valid_prediction_ids=tuple(diagnostics["valid_prediction_ids"]),
                attempted_row_count=diagnostics["attempted_row_count"],
                malformed_row_count=diagnostics["malformed_row_count"],
                invalid_row_count=diagnostics["invalid_row_count"],
                non_owning_prediction_count=diagnostics[
                    "non_owning_prediction_count"
                ],
                natural_closure_count=diagnostics["natural_closure_count"],
                controller_cap_count=diagnostics["controller_cap_count"],
                token_cap_count=diagnostics["token_cap_count"],
                error_count=diagnostics["error_count"],
                prompt_token_count=diagnostics["prompt_token_count"],
                image_token_count=diagnostics["image_token_count"],
                generated_token_count=diagnostics["generated_token_count"],
                wall_time_seconds=diagnostics["wall_time_seconds"],
                peak_device_memory_bytes=diagnostics[
                    "peak_device_memory_bytes"
                ],
                attempt_status=call.attempt.attempt_status,
                source_image_sha256=call.request.image_sha256,
                image_frozen_order=call.request.image_frozen_order,
                terminal_attempt_output_artifact_sha256=(
                    call.attempt.output_artifact_sha256
                ),
                terminal_attempt_output_artifact_path=(
                    call.attempt.output_artifact_path
                ),
                terminal_attempt_failure_code=call.attempt.failure_code,
            )
        )
        row_records.extend(scope_rows)
    return tuple(call_records), tuple(row_records)


def _spatially_enriched_references(
    references: Sequence[ReferenceObject],
    *,
    evidence: LoadedPostrunEvidence,
    image_id: str,
) -> tuple[ReferenceObject, ...]:
    """Attach the frozen metric module's derived spatial reference fields."""

    grid = evidence.admission.image(image_id).spatial_grid
    enriched: list[ReferenceObject] = []
    for reference in references:
        if reference.state != "accepted":
            enriched.append(
                replace(
                    reference,
                    owner_cell_index=None,
                    core_interior_for_mask_harm=False,
                )
            )
            continue
        assert reference.source_canvas_bbox_xyxy is not None
        ownership = grid.ownership(reference.source_canvas_bbox_xyxy)
        if not ownership.is_valid or ownership.owner_cell_index is None:
            _fail(
                "accepted reference has no owner in the frozen spatial grid",
                "analysis.postrun_reference_owner",
                image_id=image_id,
                reference_id=reference.reference_id,
            )
        owner_cell_index = ownership.owner_cell_index
        enriched.append(
            replace(
                reference,
                owner_cell_index=owner_cell_index,
                core_interior_for_mask_harm=_is_mask_harm_core_interior(
                    reference.source_canvas_bbox_xyxy,
                    grid=grid,
                    owner_cell_index=owner_cell_index,
                ),
            )
        )
    return tuple(enriched)


def _validated_call_diagnostics(call: LoadedTerminalCall) -> dict[str, Any]:
    fields = {
        "attempted_row_count",
        "controller_cap_count",
        "error_count",
        "generated_token_count",
        "image_token_count",
        "invalid_row_count",
        "malformed_row_count",
        "natural_closure_count",
        "non_owning_prediction_count",
        "peak_device_memory_bytes",
        "prompt_token_count",
        "token_cap_count",
        "valid_prediction_ids",
        "wall_time_seconds",
    }
    if set(call.call_diagnostics) != fields:
        _fail(
            "terminal call diagnostics keys are not exact",
            "analysis.postrun_call_diagnostic_keys",
            request_id=call.request.request_id,
            missing=sorted(fields - set(call.call_diagnostics)),
            extra=sorted(set(call.call_diagnostics) - fields),
        )
    return dict(call.call_diagnostics)


def _ordered_primitives(
    index: Mapping[tuple[str, float, str, str], ImageMetricPrimitive],
    *,
    evidence: LoadedPostrunEvidence,
    ledger_scope: ReferenceLedgerScope,
    threshold: float,
    arm_code: str,
) -> tuple[ImageMetricPrimitive, ...]:
    return tuple(
        index[(ledger_scope, threshold, arm_code, str(record.image_id))]
        for record in evidence.cohort.records
    )


def _artifact_payloads(
    assembly: SupportedPostrunAssembly,
) -> Mapping[str, bytes]:
    payload_rows = {
        "arm-merge-results.json": (
            "arm_merge_results",
            [item.to_json_dict() for item in assembly.arm_merges],
        ),
        "image-metric-primitives.json": (
            "image_metric_primitives",
            [item.to_json_dict() for item in assembly.image_metric_primitives],
        ),
        "aggregate-metric-reports.json": (
            "aggregate_metric_reports",
            [item.to_json_dict() for item in assembly.aggregate_reports],
        ),
        "bootstrap-metric-reports.json": (
            "bootstrap_metric_reports",
            [item.to_json_dict() for item in assembly.bootstrap_reports],
        ),
    }
    return {
        filename: (
            canonical_json_text(
                {
                    field_name: rows,
                    "schema_version": POSTRUN_ASSEMBLY_SCHEMA_VERSION,
                }
            )
            + "\n"
        ).encode("utf-8")
        for filename, (field_name, rows) in payload_rows.items()
    }


def _estimator_registry() -> list[dict[str, str]]:
    meanings = {
        "post_merge_local_rescue_rate": (
            "Fraction of Full-Image Single Rollout missed references matched after "
            "the candidate arm's frozen merge."
        ),
        "raw_any_call_union_rescue_rate": (
            "Fraction of Full-Image Single Rollout missed references matched by any "
            "raw candidate call."
        ),
        "overall_retention": (
            "Fraction of Full-Image Single Rollout detections retained after candidate merge."
        ),
        "mask_harm_retention": (
            "Retention restricted to baseline-detected core-interior references."
        ),
        "manual_precision": (
            "Matched accepted references divided by matched plus unmatched valid predictions."
        ),
        "manual_precision_uncertainty_ignored_as_unmatched": (
            "Manual precision sensitivity counting uncertainty-ignored predictions as unmatched."
        ),
        "post_merge_strict_duplicate_rate": (
            "Strict duplicate excess divided by final valid predictions after merge."
        ),
        "invalid_row_rate": "Malformed plus invalid rows divided by attempted rows.",
        "invalid_call_rate": "Invalid terminal calls divided by attempted calls.",
        "natural_closure_rate": "Naturally closed calls divided by attempted calls.",
        "prediction_count_inflation": (
            "Candidate final valid prediction count divided by Full-Image Single Rollout count."
        ),
        "owning_seed_raw_rescue_difference": (
            "Masked owning-call raw rescue minus seed-matched full-image raw-call rescue."
        ),
        "post_merge_local_rescue_rate_difference": (
            "Paired candidate-minus-comparator post-merge Local Rescue Rate difference."
        ),
        "manual_precision_difference": (
            "Paired candidate-minus-comparator Manual Precision difference."
        ),
        "post_merge_strict_duplicate_rate_difference": (
            "Paired candidate-minus-comparator post-merge strict duplicate-rate difference."
        ),
        "invalid_row_rate_difference": (
            "Paired candidate-minus-comparator invalid-row-rate difference."
        ),
        "invalid_call_rate_difference": (
            "Paired candidate-minus-comparator invalid-call-rate difference."
        ),
        "natural_closure_rate_difference": (
            "Paired candidate-minus-comparator natural-closure-rate difference."
        ),
    }
    return [
        {
            "estimator_name": name,
            "operational_meaning": meanings[name],
        }
        for name in (
            *ABSOLUTE_BOOTSTRAP_METRICS,
            OWNING_SEED_RAW_RESCUE_BOOTSTRAP_METRIC,
            *PAIRED_ARM_BOOTSTRAP_METRICS,
        )
    ]


def _write_once(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _postrun_repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _derive_postrun_source_identity(source_commit: str) -> dict[str, Any]:
    """Bind this experiment's post-run analysis surface to committed Git bytes."""

    repository_root = _postrun_repository_root()
    head = _git_output(repository_root, "rev-parse", "--verify", "HEAD").strip()
    if source_commit != head:
        _fail(
            "declared analysis source commit differs from live Git HEAD",
            "analysis.postrun_source_commit_mismatch",
            declared_source_commit=source_commit,
            live_git_head=head,
        )
    source_hashes: dict[str, str] = {}
    for relative_path in POSTRUN_SOURCE_RELATIVE_PATHS:
        path = repository_root / relative_path
        if not path.is_file():
            _fail(
                "post-run source file is absent",
                "analysis.postrun_source_missing",
                relative_path=relative_path,
            )
        committed_bytes = _git_bytes(
            repository_root,
            "show",
            f"{head}:{relative_path}",
        )
        current_bytes = path.read_bytes()
        if current_bytes != committed_bytes:
            _fail(
                "post-run source file differs from live Git HEAD",
                "analysis.postrun_source_dirty",
                relative_path=relative_path,
            )
        source_hashes[relative_path] = hashlib.sha256(current_bytes).hexdigest()
    identity = {
        "git_head_commit": head,
        "source_file_sha256s": source_hashes,
    }
    return {
        **identity,
        "identity_sha256": sha256_payload(identity),
    }


def _git_bytes(repository_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(repository_root), *arguments),
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        _fail(
            "Git could not verify the post-run analysis source",
            "analysis.postrun_source_git",
            git_arguments=list(arguments),
            stderr=completed.stderr.decode("utf-8", errors="replace"),
        )
    return completed.stdout


def _git_output(repository_root: Path, *arguments: str) -> str:
    return _git_bytes(repository_root, *arguments).decode("ascii")


def _fail(message: str, code: str, **context: Any) -> None:
    raise DataContractError(message, code=code, context=context)
