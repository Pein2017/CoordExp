#!/usr/bin/env python3
"""Versioned Wave 7 exact-resume comparison and pre-child admission gates."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.coordexp_swift import (  # noqa: E402
    wave7_exact_resume_compare as v1,
)
from src.artifacts.checkpoint_payload import (  # noqa: E402
    admit_inference_checkpoint_payload_identity,
)
from src.qwen.parity import canonical_json_bytes  # noqa: E402


FINAL_RECEIPT_SCHEMA = "coordexp-swift-wave7-exact-resume-comparison-v2"
PRE_CHILD_RECEIPT_SCHEMA = "coordexp-swift-wave7-pre-child-gate-v1"
EVENT_SCHEMA = "coordexp-swift-checkpoint-publication-event"
EVENT_SCHEMA_VERSION = 2
PROGRESS_SCHEMA = "coordexp-swift-checkpoint-committed-progress"
PROGRESS_SCHEMA_VERSION = 1
V1_SOURCE_SHA256 = "dfbb4d63c22d5c0db78c296514af1fe8fe24c28299175c90547b1a2917064034"
V2_EVENT_FIELDS = v1.CHECKPOINT_PUBLICATION_EVENT_FIELDS | frozenset(
    {"schema", "schema_version", "inference_payload_identity", "committed_progress"}
)
COMMITTED_PROGRESS_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "completed_steps",
        "consumed_packs",
        "optimizer_update_status",
        "finite_status",
    }
)
ACCURACY_STATS_FIELDS = frozenset({"top1_correct", "top5_correct", "atom_count"})
ACCURACY_FIELDS = ("acc_top1", "acc_top5")


@dataclass(frozen=True)
class FinalRequest:
    uninterrupted_run_dir: Path
    interrupted_parent_run_dir: Path
    resume_child_run_dir: Path
    interruption_marker: Path
    termination_receipt: Path
    output: Path
    expected_interrupt_source_sha256: str
    expected_source_sha256: str
    expected_provenance_sha256: str


@dataclass(frozen=True)
class PreChildRequest:
    uninterrupted_run_dir: Path
    interrupted_parent_run_dir: Path
    interruption_marker: Path
    termination_receipt: Path
    output: Path
    expected_interrupt_source_sha256: str
    expected_source_sha256: str
    expected_provenance_sha256: str


def _common_arguments(parser: argparse.ArgumentParser, *, child: bool) -> None:
    parser.add_argument("--uninterrupted-run-dir", type=Path, required=True)
    parser.add_argument("--interrupted-parent-run-dir", type=Path, required=True)
    if child:
        parser.add_argument("--resume-child-run-dir", type=Path, required=True)
    parser.add_argument("--interruption-marker", type=Path, required=True)
    parser.add_argument("--termination-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-interrupt-source-sha256", type=v1._digest_arg, required=True
    )
    parser.add_argument("--expected-source-sha256", type=v1._digest_arg, required=True)
    parser.add_argument(
        "--expected-provenance-sha256", type=v1._digest_arg, required=True
    )


def parse_request(
    argv: Sequence[str] | None = None,
) -> FinalRequest | PreChildRequest | tuple[Path, str]:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare = subparsers.add_parser("compare")
    _common_arguments(compare, child=True)
    pre_child = subparsers.add_parser("pre-child")
    _common_arguments(pre_child, child=False)
    verify = subparsers.add_parser("verify-pre-child")
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--expected-payload-sha256", type=v1._digest_arg, required=True)
    args = parser.parse_args(argv)
    if args.command == "verify-pre-child":
        return args.receipt.expanduser().resolve(), args.expected_payload_sha256
    reference = args.uninterrupted_run_dir.expanduser().resolve()
    parent = args.interrupted_parent_run_dir.expanduser().resolve()
    marker = args.interruption_marker.expanduser().resolve()
    receipt = args.termination_receipt.expanduser().resolve()
    output = args.output.expanduser().resolve(strict=False)
    common = {
        "uninterrupted_run_dir": reference,
        "interrupted_parent_run_dir": parent,
        "interruption_marker": marker,
        "termination_receipt": receipt,
        "output": output,
        "expected_interrupt_source_sha256": args.expected_interrupt_source_sha256,
        "expected_source_sha256": args.expected_source_sha256,
        "expected_provenance_sha256": args.expected_provenance_sha256,
    }
    if args.command == "pre-child":
        return PreChildRequest(**common)
    child = args.resume_child_run_dir.expanduser().resolve()
    return FinalRequest(resume_child_run_dir=child, **common)


def _request_paths(request: FinalRequest | PreChildRequest) -> dict[str, Path]:
    result = {
        "uninterrupted_run_dir": request.uninterrupted_run_dir,
        "interrupted_parent_run_dir": request.interrupted_parent_run_dir,
        "interruption_marker": request.interruption_marker,
        "termination_receipt": request.termination_receipt,
        "output": request.output,
    }
    if isinstance(request, FinalRequest):
        result["resume_child_run_dir"] = request.resume_child_run_dir
    return result


def _validate_input_topology(request: FinalRequest | PreChildRequest) -> None:
    paths = _request_paths(request)
    items = tuple(paths.items())
    overlaps = [
        {
            "left": left_name,
            "left_path": str(left),
            "right": right_name,
            "right_path": str(right),
        }
        for index, (left_name, left) in enumerate(items)
        for right_name, right in items[index + 1 :]
        if v1._paths_overlap(left, right)
    ]
    if overlaps:
        raise v1.Wave7CompareError(
            "run trees and gate artifacts must be pairwise disjoint",
            code="wave7_compare_v2.path_topology",
            context={"overlaps": overlaps},
        )


def _legacy_request(
    request: FinalRequest | PreChildRequest,
) -> v1.CompareRequest:
    child = (
        request.resume_child_run_dir
        if isinstance(request, FinalRequest)
        else request.interrupted_parent_run_dir.parent / ".unused-resume-child"
    )
    return v1.CompareRequest(
        uninterrupted_run_dir=request.uninterrupted_run_dir,
        interrupted_parent_run_dir=request.interrupted_parent_run_dir,
        resume_child_run_dir=child,
        interruption_marker=request.interruption_marker,
        termination_receipt=request.termination_receipt,
        output=request.output,
        expected_interrupt_source_sha256=request.expected_interrupt_source_sha256,
        expected_source_sha256=V1_SOURCE_SHA256,
        expected_provenance_sha256=request.expected_provenance_sha256,
    )


def _validate_v2_source_inventory(
    bindings: Mapping[str, Any],
    *,
    request: FinalRequest | PreChildRequest,
    mismatches: list[dict[str, Any]],
) -> None:
    expected = {
        str(Path(__file__).resolve()): request.expected_source_sha256,
        str(Path(v1.__file__).resolve()): V1_SOURCE_SHA256,
    }
    rows = bindings.get("source_hashes")
    observed = (
        {row.get("path"): row.get("sha256") for row in rows if isinstance(row, Mapping)}
        if isinstance(rows, list)
        else {}
    )
    if any(observed.get(path) != digest for path, digest in expected.items()):
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.source_inventory",
            scope="interruption",
            path="source_hashes.v1_and_v2",
            expected=expected,
            observed=observed,
        )


def _event_version(event: Mapping[str, Any]) -> int:
    if set(event) == v1.CHECKPOINT_PUBLICATION_EVENT_FIELDS:
        return 1
    if set(event) == V2_EVENT_FIELDS:
        return 2
    raise v1.Wave7CompareError(
        "checkpoint publication event has an unsupported field set",
        code="wave7_compare_v2.checkpoint_publication_schema",
        context={
            "missing_v2": sorted(V2_EVENT_FIELDS - set(event)),
            "unexpected_v2": sorted(set(event) - V2_EVENT_FIELDS),
        },
    )


def _checkpoint_dir_for_event(run: v1.RunArtifacts, event: Mapping[str, Any]) -> Path:
    relative = event.get("checkpoint_path")
    if not isinstance(relative, str):
        raise v1.Wave7CompareError(
            "checkpoint event path is malformed",
            code="wave7_compare_v2.checkpoint_publication_schema",
        )
    path = (run.root / relative).resolve()
    if not path.is_relative_to(run.root) or path.parent != run.root / "checkpoints":
        raise v1.Wave7CompareError(
            "checkpoint event escapes the canonical checkpoint root",
            code="wave7_compare_v2.checkpoint_publication_schema",
        )
    return path


def _validate_committed_progress(
    event: Mapping[str, Any],
    *,
    run: v1.RunArtifacts,
    is_latest: bool,
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    progress = event.get("committed_progress")
    valid = isinstance(progress, Mapping) and set(progress) == COMMITTED_PROGRESS_FIELDS
    if valid:
        progress = dict(progress)
        completed = progress["completed_steps"]
        consumed = progress["consumed_packs"]
        valid = (
            progress["schema"] == PROGRESS_SCHEMA
            and progress["schema_version"] == PROGRESS_SCHEMA_VERSION
            and isinstance(completed, int)
            and not isinstance(completed, bool)
            and completed == event.get("step")
            and isinstance(consumed, int)
            and not isinstance(consumed, bool)
            and consumed >= 0
            and (
                progress["optimizer_update_status"] is None
                or isinstance(progress["optimizer_update_status"], str)
            )
            and (
                progress["finite_status"] is None
                or isinstance(progress["finite_status"], str)
            )
        )
    if not valid:
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.committed_progress",
            scope=run.role,
            path=f"checkpoint_event.step-{event.get('step')}.committed_progress",
            expected="strict committed progress v1",
            observed=progress,
        )
        return {}
    assert isinstance(progress, dict)
    if is_latest:
        observed = {
            "completed_steps": run.run["completed_steps"],
            "consumed_packs": run.run["consumed_packs"],
            "optimizer_update_status": run.run["final_optimizer_update_status"],
            "finite_status": run.run["final_finite_status"],
        }
        expected = {
            key: progress[key]
            for key in (
                "completed_steps",
                "consumed_packs",
                "optimizer_update_status",
                "finite_status",
            )
        }
        if observed != expected:
            v1._add_mismatch(
                mismatches,
                code="wave7_compare_v2.committed_progress",
                scope=run.role,
                path="run.top_level_progress",
                expected=expected,
                observed=observed,
            )
    return progress


def _validate_v2_event(
    event: Mapping[str, Any],
    *,
    run: v1.RunArtifacts,
    is_latest: bool,
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    if (
        event.get("schema") != EVENT_SCHEMA
        or event.get("schema_version") != EVENT_SCHEMA_VERSION
        or event.get("status") != "completed"
    ):
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.checkpoint_publication_schema",
            scope=run.role,
            path=f"checkpoint_event.step-{event.get('step')}.schema",
            expected={
                "schema": EVENT_SCHEMA,
                "schema_version": 2,
                "status": "completed",
            },
            observed={
                key: event.get(key) for key in ("schema", "schema_version", "status")
            },
        )
    checkpoint_dir = _checkpoint_dir_for_event(run, event)
    identity: dict[str, Any] | None = None
    try:
        identity = admit_inference_checkpoint_payload_identity(
            checkpoint_dir, event.get("inference_payload_identity")
        )
    except BaseException as exc:
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.inference_payload_identity",
            scope=run.role,
            path=f"checkpoint_event.step-{event.get('step')}.inference_payload_identity",
            expected="live payload exactly admitted by its committed manifest",
            observed={"error_code": getattr(exc, "code", type(exc).__name__)},
        )
    progress = _validate_committed_progress(
        event, run=run, is_latest=is_latest, mismatches=mismatches
    )
    return {"inference_payload_identity": identity, "committed_progress": progress}


def _validate_run_events(
    run: v1.RunArtifacts,
    *,
    allow_legacy: bool,
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    measurement = run.run.get("measurement")
    events = (
        measurement.get("checkpoint_publication_events")
        if isinstance(measurement, Mapping)
        else None
    )
    if not isinstance(events, list) or not events:
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.checkpoint_publication_schema",
            scope=run.role,
            path="run.measurement.checkpoint_publication_events",
            expected="nonempty event list",
            observed=events,
        )
        return {"event_versions": [], "events": []}
    versions: list[int] = []
    evidence: list[dict[str, Any]] = []
    for index, raw in enumerate(events):
        if not isinstance(raw, Mapping):
            raise v1.Wave7CompareError(
                "checkpoint event must be an object",
                code="wave7_compare_v2.checkpoint_publication_schema",
            )
        version = _event_version(raw)
        versions.append(version)
        if version == 1:
            v1._add_mismatch(
                mismatches,
                code="wave7_compare_v2.legacy_event_not_r5_evidence",
                scope=run.role,
                path=f"checkpoint_event[{index}]",
                expected="event schema v2",
                observed=(
                    "legacy event accepted for bounded r4 diagnostics only"
                    if allow_legacy
                    else "legacy event schema v1"
                ),
            )
            evidence.append({"legacy_event": True})
        else:
            evidence.append(
                _validate_v2_event(
                    raw,
                    run=run,
                    is_latest=index == len(events) - 1,
                    mismatches=mismatches,
                )
            )
    if len(set(versions)) != 1:
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.mixed_event_versions",
            scope=run.role,
            path="checkpoint_event_versions",
            expected="one event version per run",
            observed=versions,
        )
    if run.run["checkpoint_event_count"] != len(events) and versions[-1] == 2:
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.committed_progress",
            scope=run.role,
            path="run.checkpoint_event_count",
            expected=len(events),
            observed=run.run["checkpoint_event_count"],
        )
    return {"event_versions": versions, "events": evidence}


def _validate_interrupt_publication_binding(
    marker: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    parent: v1.RunArtifacts,
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    marker_event = marker.get("checkpoint_publication_event")
    marker_run = marker.get("checkpoint_publication_run")
    receipt_publication = receipt.get("checkpoint_publication")
    if not isinstance(marker_event, Mapping) or not isinstance(marker_run, Mapping):
        raise v1.Wave7CompareError(
            "interruption marker publication binding is missing",
            code="wave7_compare_v2.interruption_schema",
        )
    if not isinstance(receipt_publication, Mapping):
        raise v1.Wave7CompareError(
            "termination publication binding is missing",
            code="wave7_compare_v2.interruption_schema",
        )
    v1._require_fields(
        marker_run,
        v1.CHECKPOINT_PUBLICATION_RUN_FIELDS,
        "interruption checkpoint publication run",
    )
    v1._require_fields(
        receipt_publication,
        v1.CHECKPOINT_PUBLICATION_RECEIPT_FIELDS,
        "termination checkpoint publication binding",
    )
    receipt_event = receipt_publication.get("event")
    if not isinstance(receipt_event, Mapping):
        raise v1.Wave7CompareError(
            "termination publication event is missing",
            code="wave7_compare_v2.interruption_schema",
        )
    marker_version = _event_version(marker_event)
    receipt_version = _event_version(receipt_event)
    measurement = parent.run.get("measurement")
    parent_events = (
        measurement.get("checkpoint_publication_events")
        if isinstance(measurement, Mapping)
        else None
    )
    matching = (
        [
            dict(event)
            for event in parent_events
            if isinstance(event, Mapping)
            and event.get("step") == 3
            and not isinstance(event.get("step"), bool)
        ]
        if isinstance(parent_events, list)
        else []
    )
    expected_event = matching[0] if len(matching) == 1 else None
    run_path = parent.root / "run.json"
    expected_run = {
        "file_sha256": v1._sha256_file(run_path),
        "path": str(run_path),
        "size": run_path.stat().st_size,
    }
    expected_receipt = {
        "event": expected_event,
        "run_file_sha256": expected_run["file_sha256"],
        "run_path": expected_run["path"],
        "run_size": expected_run["size"],
    }
    if (
        marker_version != receipt_version
        or dict(marker_event) != expected_event
        or dict(marker_run) != expected_run
        or dict(receipt_publication) != expected_receipt
    ):
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.interruption_binding",
            scope="interruption",
            path="checkpoint_publication",
            expected={"event": expected_event, "run": expected_run},
            observed={
                "marker_event": marker_event,
                "marker_run": marker_run,
                "receipt": receipt_publication,
            },
        )
    return {**expected_receipt, "event_version": marker_version}


def _load_interrupt_evidence(
    request: FinalRequest | PreChildRequest,
    parent: v1.RunArtifacts,
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    marker = v1._strict_json_file(
        request.interruption_marker, owner="interruption marker"
    )
    receipt = v1._strict_json_file(
        request.termination_receipt, owner="termination receipt"
    )
    v1._require_fields(marker, v1.MARKER_FIELDS, "interruption marker")
    v1._require_fields(receipt, v1.TERMINATION_RECEIPT_FIELDS, "termination receipt")
    bindings = v1._validate_interrupt_bindings(
        marker,
        receipt,
        request=_legacy_request(request),
        parent=parent,
        mismatches=mismatches,
    )
    _validate_v2_source_inventory(bindings, request=request, mismatches=mismatches)
    if marker.get("schema") != v1.AUDITED_INTERRUPT_MARKER_SCHEMA:
        raise v1.Wave7CompareError(
            "interruption marker schema is unsupported",
            code="wave7_compare_v2.interruption_schema",
        )
    if receipt.get("schema") != v1.AUDITED_INTERRUPT_RECEIPT_SCHEMA:
        raise v1.Wave7CompareError(
            "termination receipt schema is unsupported",
            code="wave7_compare_v2.interruption_schema",
        )
    publication = _validate_interrupt_publication_binding(
        marker, receipt, parent=parent, mismatches=mismatches
    )
    postconditions = receipt.get("postconditions")
    termination = receipt.get("termination")
    marker_observation = receipt.get("marker")
    if (
        not isinstance(postconditions, Mapping)
        or not isinstance(termination, Mapping)
        or not isinstance(marker_observation, Mapping)
    ):
        raise v1.Wave7CompareError(
            "termination receipt is incomplete",
            code="wave7_compare_v2.interruption_schema",
        )
    v1._require_fields(
        postconditions, v1.POSTCONDITION_FIELDS, "termination postconditions"
    )
    v1._require_fields(termination, v1.TERMINATION_FIELDS, "termination details")
    v1._require_fields(
        marker_observation,
        v1.MARKER_OBSERVATION_FIELDS,
        "termination marker observation",
    )
    marker_sha256 = v1._sha256_file(request.interruption_marker)
    expected_marker_observation = {
        "expected_file_sha256": marker_sha256,
        "file_sha256": marker_sha256,
        "final_file_sha256": marker_sha256,
        "path": str(request.interruption_marker),
        "published": True,
        "strict_payload_equal": True,
        "unchanged": True,
    }
    gate = (
        receipt.get("status") == "passed"
        and receipt.get("errors") == []
        and marker_observation == expected_marker_observation
        and postconditions.get("late_write_detected") is False
        and postconditions.get("max_logged_train_step") == 3
        and postconditions.get("step_5_absent") is True
        and postconditions.get("final_json_absent") is True
        and postconditions.get("manifest_unchanged") is True
        and postconditions.get("checkpoint_publication_event_unchanged") is True
        and postconditions.get("checkpoint_publication_run_final_file_sha256")
        == publication["run_file_sha256"]
        and postconditions.get("run_not_completed") is True
        and postconditions.get("nvidia_compute_apps") == []
        and postconditions.get("nvidia_compute_apps_added") == []
        and termination.get("cleanup_errors") == []
        and termination.get("launcher_exited") is True
        and termination.get("remaining_pids") == []
        and termination.get("remaining_pgids") == []
    )
    if not gate:
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.interruption_gate",
            scope="interruption",
            path="termination_receipt",
            expected="passed bounded termination with immutable step-3",
            observed={
                "status": receipt.get("status"),
                "postconditions": postconditions,
            },
        )
    if marker.get("checkpoint") != receipt.get("checkpoint"):
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.interruption_binding",
            scope="interruption",
            path="checkpoint",
            expected=marker.get("checkpoint"),
            observed=receipt.get("checkpoint"),
        )
    current_tree = v1.interrupt._snapshot_run_tree(parent.root)
    expected_tree = postconditions["stability_after"]
    if not isinstance(expected_tree, Mapping) or (
        current_tree["entry_count"] != expected_tree.get("entry_count")
        or current_tree["fingerprint"] != expected_tree.get("fingerprint")
    ):
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.parent_tree_changed",
            scope="interrupted_parent",
            path="run_tree",
            expected=expected_tree,
            observed=v1._tree_identity(current_tree),
        )
    return {
        "marker_file_sha256": marker_sha256,
        "termination_receipt_file_sha256": v1._sha256_file(request.termination_receipt),
        "checkpoint": receipt.get("checkpoint"),
        "checkpoint_publication": publication,
        "identity_bindings": bindings,
        "parent_tree": v1._tree_identity(current_tree),
        "event_version": publication["event_version"],
    }


def _legacy_effective_parent(parent: v1.RunArtifacts) -> v1.RunArtifacts:
    train_rows = [row for row in parent.logs if row.get("split") == "train"]
    completed_steps = max((int(row["step"]) for row in train_rows), default=0)
    consumed_packs = 0
    for row in train_rows:
        count = row.get("micro_step_count")
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            consumed_packs += count
        else:
            consumed_packs = completed_steps
            break
    patched = dict(parent.run)
    patched.update(
        completed_steps=completed_steps,
        consumed_packs=consumed_packs,
        checkpoint_event_count=1,
    )
    return v1.RunArtifacts(
        role=parent.role,
        root=parent.root,
        run=patched,
        config=parent.config,
        logs=parent.logs,
        hashes=parent.hashes,
    )


def _effective_runs(
    runs: Mapping[str, v1.RunArtifacts], *, event_version: int
) -> dict[str, v1.RunArtifacts]:
    effective = dict(runs)
    if event_version == 1:
        effective["interrupted_parent"] = _legacy_effective_parent(
            runs["interrupted_parent"]
        )
    return effective


def _compare_run_contracts(
    runs: Mapping[str, v1.RunArtifacts],
    request: FinalRequest,
    *,
    event_version: int,
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    effective = _effective_runs(runs, event_version=event_version)
    result = v1._compare_run_contracts(effective, _legacy_request(request), mismatches)
    result["interrupted_parent_progress"] = {
        "authority": (
            "event_v2_atomic_committed_progress"
            if event_version == 2
            else "legacy_bound_marker_event_log_manifest_termination_derivation"
        ),
        "observed_top_level": {
            key: runs["interrupted_parent"].run[key]
            for key in ("completed_steps", "consumed_packs", "checkpoint_event_count")
        },
        "effective": {
            key: effective["interrupted_parent"].run[key]
            for key in ("completed_steps", "consumed_packs", "checkpoint_event_count")
        },
    }
    return result


def _checked_accuracy_stats(
    row: Mapping[str, Any],
    *,
    role: str,
    key: tuple[str, int],
    require_authoritative: bool,
    mismatches: list[dict[str, Any]],
) -> tuple[int, int, int] | None:
    raw = row.get("accuracy_stats")
    path = f"logging.{key[0]}.step-{key[1]}.accuracy_stats"
    if not isinstance(raw, Mapping) or set(raw) != ACCURACY_STATS_FIELDS:
        if require_authoritative:
            v1._add_mismatch(
                mismatches,
                code="wave7_compare_v2.accuracy_stats",
                scope=role,
                path=path,
                expected=sorted(ACCURACY_STATS_FIELDS),
                observed=raw,
            )
        return None
    values = tuple(
        raw[field] for field in ("top1_correct", "top5_correct", "atom_count")
    )
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.accuracy_stats",
            scope=role,
            path=path,
            expected="three strict integers",
            observed=raw,
        )
        return None
    top1, top5, atoms = values
    if atoms <= 0 or not (0 <= top1 <= atoms and 0 <= top5 <= atoms):
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.accuracy_stats",
            scope=role,
            path=path,
            expected="0 <= correct <= atom_count and atom_count > 0",
            observed=raw,
        )
        return None
    ratios = {"acc_top1": top1 / atoms, "acc_top5": top5 / atoms}
    for field, expected in ratios.items():
        observed = row.get(field)
        if (
            not isinstance(observed, (int, float))
            or isinstance(observed, bool)
            or not math.isfinite(float(observed))
            or not math.isclose(float(observed), expected, rel_tol=1e-6, abs_tol=1e-8)
        ):
            v1._add_mismatch(
                mismatches,
                code="wave7_compare_v2.accuracy_ratio",
                scope=role,
                path=f"logging.{key[0]}.step-{key[1]}.{field}",
                expected=expected,
                observed=observed,
            )
    return top1, top5, atoms


def _is_observational_timing(field: str) -> bool:
    return (
        field in v1.TIMING_FIELDS
        or field == "eval_duration_seconds"
        or field.endswith("_duration_seconds")
        or field.endswith("_wall_seconds")
        or field.startswith("time/")
    )


def _semantic_log_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in row.items()
        if not _is_observational_timing(key)
        and key != "per_rank_measurement"
        and not key.startswith("resource/")
        and key != "accuracy_stats"
        and key not in ACCURACY_FIELDS
    }
    non_finite = result.get("non_finite_fields")
    if isinstance(non_finite, list):
        result["non_finite_fields"] = sorted(
            field
            for field in non_finite
            if not _is_observational_timing(field) and not field.startswith("resource/")
        )
    return result


def _timing_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "split": row["split"],
        "step": row["step"],
        "values": {
            key: value
            for key, value in row.items()
            if _is_observational_timing(key)
            or key == "per_rank_measurement"
            or key.startswith("resource/")
        },
    }


def _compare_log_pairs(
    runs: Mapping[str, v1.RunArtifacts],
    pairs: Sequence[tuple[str, str, tuple[str, int]]],
    *,
    require_authoritative: bool,
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    indexed = {
        role: v1._index_logs(artifacts.logs, role=role)
        for role, artifacts in runs.items()
    }
    numeric_summary: dict[str, Any] = {}
    accuracy_rows: list[dict[str, Any]] = []
    for left_role, right_role, key in pairs:
        left = indexed[left_role].get(key)
        right = indexed[right_role].get(key)
        if left is None or right is None:
            v1._add_mismatch(
                mismatches,
                code="wave7_compare_v2.logging_inventory",
                scope=f"{left_role}_vs_{right_role}",
                path=f"{key[0]}.step-{key[1]}",
                expected="row present on both sides",
                observed={left_role: left is not None, right_role: right is not None},
            )
            continue
        left_stats = _checked_accuracy_stats(
            left,
            role=left_role,
            key=key,
            require_authoritative=require_authoritative,
            mismatches=mismatches,
        )
        right_stats = _checked_accuracy_stats(
            right,
            role=right_role,
            key=key,
            require_authoritative=require_authoritative,
            mismatches=mismatches,
        )
        if (
            left_stats is not None
            and right_stats is not None
            and left_stats != right_stats
        ):
            v1._add_mismatch(
                mismatches,
                code="wave7_compare_v2.accuracy_sufficient_statistics",
                scope=f"{left_role}_vs_{right_role}",
                path=f"{key[0]}.step-{key[1]}.accuracy_stats",
                expected=left_stats,
                observed=right_stats,
            )
        accuracy_rows.append(
            {
                "key": {"split": key[0], "step": key[1]},
                "left_role": left_role,
                "left": None if left_stats is None else list(left_stats),
                "right_role": right_role,
                "right": None if right_stats is None else list(right_stats),
                "authoritative": left_stats is not None and right_stats is not None,
            }
        )
        v1._compare_log_value(
            _semantic_log_projection(left),
            _semantic_log_projection(right),
            path=f"{key[0]}.step-{key[1]}",
            scope=f"{left_role}_vs_{right_role}",
            mismatches=mismatches,
            summary=numeric_summary,
        )
    if not require_authoritative:
        missing = [row for row in accuracy_rows if not row["authoritative"]]
        if missing:
            v1._add_mismatch(
                mismatches,
                code="wave7_compare_v2.legacy_accuracy_not_r5_evidence",
                scope="legacy_r4",
                path="logging.accuracy_stats",
                expected="authoritative persisted integer sufficient statistics",
                observed={"missing_pair_count": len(missing)},
            )
    return {
        "accuracy_sufficient_statistics": accuracy_rows,
        "excluded_from_semantic_equality": [
            "accuracy floats (validated against authoritative integer stats in r5)",
            "all declared *_duration_seconds fields",
            "input_build_seconds",
            "input_wait_seconds",
            "per_rank_measurement",
            "resource/*",
            "step_duration_seconds",
        ],
        "excluded_observations": {
            role: [_timing_observation(row) for row in artifacts.logs]
            for role, artifacts in runs.items()
        },
        "numeric_summary": numeric_summary,
    }


def _selection_eligibility(
    run: v1.RunArtifacts, alias: Mapping[str, Any] | None
) -> dict[str, Any]:
    if alias is None:
        return {
            "authoritative_value": None,
            "eligible": False,
            "metric_matches_alias": False,
            "reason": "best_alias_absent",
        }
    step = int(alias["step"])
    indexed = v1._index_logs(run.logs, role=run.role)
    train = indexed.get(("train", step))
    evaluation = indexed.get(("eval", step))
    selector = str(alias["selector"])
    raw_stats = None if evaluation is None else evaluation.get("accuracy_stats")
    stats_valid = (
        isinstance(raw_stats, Mapping)
        and set(raw_stats) == ACCURACY_STATS_FIELDS
        and all(
            isinstance(raw_stats.get(field), int)
            and not isinstance(raw_stats.get(field), bool)
            for field in ACCURACY_STATS_FIELDS
        )
    )
    authoritative_value: float | None = None
    if stats_valid:
        assert isinstance(raw_stats, Mapping)
        top1 = raw_stats["top1_correct"]
        top5 = raw_stats["top5_correct"]
        atoms = raw_stats["atom_count"]
        stats_valid = atoms > 0 and 0 <= top1 <= atoms and 0 <= top5 <= atoms
        if stats_valid and selector in ACCURACY_FIELDS:
            numerator = top1 if selector == "acc_top1" else top5
            authoritative_value = numerator / atoms
    selected_metric = None if evaluation is None else evaluation.get(selector)
    selected_metric_matches_stats = (
        authoritative_value is not None
        and isinstance(selected_metric, (int, float))
        and not isinstance(selected_metric, bool)
        and math.isfinite(float(selected_metric))
        and float(selected_metric) == authoritative_value
    )
    alias_value = alias["value"]
    alias_value_matches_stats = (
        authoritative_value is not None
        and isinstance(alias_value, (int, float))
        and not isinstance(alias_value, bool)
        and math.isfinite(float(alias_value))
        and float(alias_value) == authoritative_value
    )
    metric_matches = selected_metric_matches_stats and alias_value_matches_stats
    checkpoint_present = (run.root / str(alias["checkpoint_path"])).is_dir()
    eligible = (
        train is not None
        and train.get("optimizer_update_status") == "applied"
        and train.get("finite_status") == "finite"
        and evaluation is not None
        and selector in ACCURACY_FIELDS
        and stats_valid
        and metric_matches
        and checkpoint_present
    )
    return {
        "alias_value_matches_stats": alias_value_matches_stats,
        "authoritative_accuracy_stats": (None if not stats_valid else dict(raw_stats)),
        "authoritative_value": authoritative_value,
        "checkpoint_present": checkpoint_present,
        "eligible": eligible,
        "eval_present": evaluation is not None,
        "finite_status": None if train is None else train.get("finite_status"),
        "metric_matches_alias": metric_matches,
        "optimizer_update_status": (
            None if train is None else train.get("optimizer_update_status")
        ),
        "selected_metric_matches_stats": selected_metric_matches_stats,
        "selector_supported": selector in ACCURACY_FIELDS,
    }


def _compare_checkpoint_selection(
    runs: Mapping[str, v1.RunArtifacts], mismatches: list[dict[str, Any]]
) -> dict[str, Any]:
    aliases = {role: v1._load_best_alias(run) for role, run in runs.items()}
    eligibility = {
        role: _selection_eligibility(runs[role], alias)
        for role, alias in aliases.items()
    }
    before = len(mismatches)
    for role in ("uninterrupted", "interrupted_parent"):
        if aliases[role] is None or eligibility[role].get("eligible") is not True:
            v1._add_mismatch(
                mismatches,
                code="wave7_compare_v2.checkpoint_selection",
                scope=role,
                path="checkpoints/best.json.eligibility",
                expected="present and eligible",
                observed={"alias": aliases[role], "eligibility": eligibility[role]},
            )
    if (
        aliases["resume_child"] is not None
        and eligibility["resume_child"].get("eligible") is not True
    ):
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.checkpoint_selection",
            scope="resume_child",
            path="checkpoints/best.json.eligibility",
            expected="absent or eligible",
            observed={
                "alias": aliases["resume_child"],
                "eligibility": eligibility["resume_child"],
            },
        )
    candidates = [
        (role, aliases[role])
        for role in ("interrupted_parent", "resume_child")
        if aliases[role] is not None and eligibility[role].get("eligible") is True
    ]
    combined_role: str | None = None
    combined: Mapping[str, Any] | None = None
    if candidates:
        combined_role, combined = max(
            candidates,
            key=lambda item: float(eligibility[item[0]]["authoritative_value"]),
        )
    reference = aliases["uninterrupted"]
    selection_fields = ("checkpoint_path", "selector", "step", "value")

    def selection_identity(
        role: str, alias: Mapping[str, Any] | None
    ) -> dict[str, Any] | None:
        if alias is None:
            return None
        return {
            **{key: alias[key] for key in selection_fields},
            "eligibility": eligibility[role].get("eligible"),
        }

    reference_identity = selection_identity("uninterrupted", reference)
    combined_identity = (
        None if combined_role is None else selection_identity(combined_role, combined)
    )
    identity_matches = (
        reference_identity is not None
        and combined_identity is not None
        and reference_identity == combined_identity
    )
    if not identity_matches:
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.checkpoint_selection",
            scope="checkpoint_selection",
            path="reference_vs_combined_parent_child.identity",
            expected=reference_identity,
            observed=combined_identity,
        )
    return {
        "combined_owner": combined_role,
        "combined_parent_child": combined,
        "eligibility": eligibility,
        "identity_fields": [*selection_fields, "eligibility"],
        "metric_values_exact": {
            "reference": None if reference is None else reference["value"],
            "combined": None if combined is None else combined["value"],
        },
        "reference": reference,
        "status": "passed" if len(mismatches) == before else "failed",
    }


def _event_at_step(run: v1.RunArtifacts, step: int) -> Mapping[str, Any] | None:
    measurement = run.run.get("measurement")
    events = (
        measurement.get("checkpoint_publication_events")
        if isinstance(measurement, Mapping)
        else None
    )
    matches = (
        [
            event
            for event in events
            if isinstance(event, Mapping)
            and event.get("step") == step
            and not isinstance(event.get("step"), bool)
        ]
        if isinstance(events, list)
        else []
    )
    return matches[0] if len(matches) == 1 else None


def _compare_event_progress_pairs(
    runs: Mapping[str, v1.RunArtifacts],
    pairs: Sequence[tuple[str, str, int]],
    mismatches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for left_role, right_role, step in pairs:
        left_event = _event_at_step(runs[left_role], step)
        right_event = _event_at_step(runs[right_role], step)
        left = (
            left_event.get("committed_progress")
            if isinstance(left_event, Mapping)
            else None
        )
        right = (
            right_event.get("committed_progress")
            if isinstance(right_event, Mapping)
            else None
        )
        authoritative = (
            isinstance(left, Mapping)
            and set(left) == COMMITTED_PROGRESS_FIELDS
            and isinstance(right, Mapping)
            and set(right) == COMMITTED_PROGRESS_FIELDS
        )
        matched = authoritative and dict(left) == dict(right)
        if authoritative and not matched:
            v1._add_mismatch(
                mismatches,
                code="wave7_compare_v2.committed_progress_pair",
                scope=f"{left_role}_vs_{right_role}",
                path=f"checkpoint_event.step-{step}.committed_progress",
                expected=left,
                observed=right,
            )
        results.append(
            {
                "left_role": left_role,
                "right_role": right_role,
                "step": step,
                "authoritative": authoritative,
                "matched": matched,
                "left": None if left is None else dict(left),
                "right": None if right is None else dict(right),
            }
        )
    return results


def _sanitize_publication_runs(
    runs: Mapping[str, v1.RunArtifacts], *, event_version: int
) -> dict[str, v1.RunArtifacts]:
    effective = _effective_runs(runs, event_version=event_version)
    result: dict[str, v1.RunArtifacts] = {}
    for role, artifacts in effective.items():
        run = dict(artifacts.run)
        measurement = dict(run["measurement"])
        measurement["checkpoint_publication_events"] = [
            {key: event[key] for key in v1.CHECKPOINT_PUBLICATION_EVENT_FIELDS}
            for event in measurement["checkpoint_publication_events"]
            if isinstance(event, Mapping)
            and v1.CHECKPOINT_PUBLICATION_EVENT_FIELDS <= set(event)
        ]
        run["measurement"] = measurement
        result[role] = v1.RunArtifacts(
            role=artifacts.role,
            root=artifacts.root,
            run=run,
            config=artifacts.config,
            logs=artifacts.logs,
            hashes=artifacts.hashes,
        )
    return result


def _compare_prechild_run_contract(
    runs: Mapping[str, v1.RunArtifacts],
    request: PreChildRequest,
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    reference = runs["uninterrupted"]
    parent = runs["interrupted_parent"]
    commits: dict[str, str] = {}
    execution: dict[str, str] = {}
    for role, artifacts in runs.items():
        repository = artifacts.run.get("provenance", {}).get("repository", {})
        commits[role] = v1._available_digest(
            repository.get("commit"), owner=f"{role} commit", length=40
        )
        execution[role] = v1._available_digest(
            repository.get("execution_relevant_digest"),
            owner=f"{role} execution digest",
        )
    if len(set(commits.values())) != 1:
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.commit_mismatch",
            scope="runs",
            path="provenance.repository.commit",
            expected=commits["uninterrupted"],
            observed=commits,
        )
    if set(execution.values()) != {request.expected_provenance_sha256}:
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.provenance_mismatch",
            scope="runs",
            path="provenance.repository.execution_relevant_digest",
            expected=request.expected_provenance_sha256,
            observed=execution,
        )
    if reference.run["status"] != "completed" or reference.run["completed_steps"] != 5:
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.run_status",
            scope="uninterrupted",
            path="run.status_and_completed_steps",
            expected={"status": "completed", "completed_steps": 5},
            observed={
                "status": reference.run["status"],
                "completed_steps": reference.run["completed_steps"],
            },
        )
    if parent.run["status"] == "completed" or parent.run["completed_steps"] != 3:
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.run_status",
            scope="interrupted_parent",
            path="run.status_and_completed_steps",
            expected={"status": "not_completed", "completed_steps": 3},
            observed={
                "status": parent.run["status"],
                "completed_steps": parent.run["completed_steps"],
            },
        )
    for field, expected in (
        ("policy_identities", reference.run["policy_identities"]),
        ("provenance.dependencies", reference.run["provenance"]["dependencies"]),
    ):
        observed = (
            parent.run["policy_identities"]
            if field == "policy_identities"
            else parent.run["provenance"]["dependencies"]
        )
        if observed != expected:
            v1._add_mismatch(
                mismatches,
                code="wave7_compare_v2.run_identity",
                scope="interrupted_parent",
                path=field,
                expected=expected,
                observed=observed,
            )
    projections = {
        role: dict(v1.build_resume_compatibility_projection(artifacts.config))
        for role, artifacts in runs.items()
    }
    projection_hashes = {
        role: v1._sha256_bytes(canonical_json_bytes(projection))
        for role, projection in projections.items()
    }
    if len(set(projection_hashes.values())) != 1:
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.resume_compatibility_mismatch",
            scope="configs",
            path="resume_compatibility",
            expected=projection_hashes["uninterrupted"],
            observed=projection_hashes,
        )
    return {
        "commit": commits,
        "execution_relevant_digest": execution,
        "resume_compatibility_sha256": projection_hashes,
        "run_ids": {role: artifacts.run["run_id"] for role, artifacts in runs.items()},
    }


def _publish_receipt(
    *,
    output: Path,
    payload: dict[str, Any],
    parent: Path,
    initial_parent_tree: Mapping[str, Any],
    mismatches: list[dict[str, Any]],
) -> int:
    prepublication = v1._tree_identity(v1.interrupt._snapshot_run_tree(parent))
    payload["parent_tree_toctou"]["prepublication"] = prepublication
    if prepublication != dict(initial_parent_tree):
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.parent_tree_changed",
            scope="interrupted_parent",
            path="run_tree_during_comparison",
            expected=initial_parent_tree,
            observed=prepublication,
        )
    if payload["schema"] == PRE_CHILD_RECEIPT_SCHEMA:
        payload["child_launch_authorized"] = not mismatches
    v1._sign_payload(payload, mismatches)
    if mismatches:
        v1._publish(output, payload)
        return 1
    stage = v1._stage_receipt_payload(output, payload)
    final_tree = v1._tree_identity(v1.interrupt._snapshot_run_tree(parent))
    payload["parent_tree_toctou"]["final_precommit"] = final_tree
    if final_tree != prepublication:
        v1._discard_staged_receipt(stage)
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.parent_tree_changed",
            scope="interrupted_parent",
            path="run_tree_at_publication_boundary",
            expected=prepublication,
            observed=final_tree,
        )
        if payload["schema"] == PRE_CHILD_RECEIPT_SCHEMA:
            payload["child_launch_authorized"] = False
        v1._sign_payload(payload, mismatches)
        v1._publish(output, payload)
        return 1
    v1._discard_staged_receipt(stage)
    v1._sign_payload(payload, mismatches)
    stage = v1._stage_receipt_payload(output, payload)
    postlink = v1._commit_staged_receipt(
        stage,
        output,
        payload,
        interrupted_parent_run_dir=parent,
        expected_parent_identity=final_tree,
    )
    if postlink != final_tree:
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.parent_tree_changed",
            scope="interrupted_parent",
            path="run_tree_after_final_link",
            expected=final_tree,
            observed=postlink,
        )
        if payload["schema"] == PRE_CHILD_RECEIPT_SCHEMA:
            payload["child_launch_authorized"] = False
        v1._sign_payload(payload, mismatches)
        v1._publish(output, payload)
        return 1
    return 0


def _base_payload(
    *,
    schema: str,
    source_sha256: str,
    request: FinalRequest | PreChildRequest,
    initial_parent_tree: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": schema,
        "status": "failed",
        "receipt_payload_sha256": None,
        "source": {
            "path": str(Path(__file__).resolve()),
            "expected_sha256": request.expected_source_sha256,
            "observed_sha256": source_sha256,
            "legacy_helper": {
                "path": str(Path(v1.__file__).resolve()),
                "sha256": V1_SOURCE_SHA256,
            },
        },
        "mismatches": [],
        "runs": {},
        "run_contract": {},
        "interruption": {},
        "event_validation": {},
        "log_comparison": {},
        "storage": {},
        "parent_tree_toctou": {
            "initial": dict(initial_parent_tree),
            "prepublication": None,
            "final_precommit": None,
        },
    }
    if schema == PRE_CHILD_RECEIPT_SCHEMA:
        payload["child_launch_authorized"] = False
        payload["state_comparison"] = {}
    else:
        payload["state_comparisons"] = {}
    return payload


def execute_pre_child(request: PreChildRequest) -> int:
    try:
        _validate_input_topology(request)
        v1._validate_interrupt_source_pin(_legacy_request(request))
        v1.assert_absent_artifact_target(request.output)
        initial = v1._tree_identity(
            v1.interrupt._snapshot_run_tree(request.interrupted_parent_run_dir)
        )
    except BaseException as exc:
        print(
            f"{getattr(exc, 'code', 'wave7_compare_v2.preflight')}: {exc}",
            file=sys.stderr,
        )
        return 2
    source_sha256 = v1._sha256_file(Path(__file__).resolve())
    payload = _base_payload(
        schema=PRE_CHILD_RECEIPT_SCHEMA,
        source_sha256=source_sha256,
        request=request,
        initial_parent_tree=initial,
    )
    mismatches: list[dict[str, Any]] = payload["mismatches"]
    if source_sha256 != request.expected_source_sha256:
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.source_mismatch",
            scope="source",
            path="comparator_v2.sha256",
            expected=request.expected_source_sha256,
            observed=source_sha256,
        )
    else:
        try:
            runs = {
                "uninterrupted": v1._load_run(
                    request.uninterrupted_run_dir, role="uninterrupted"
                ),
                "interrupted_parent": v1._load_run(
                    request.interrupted_parent_run_dir, role="interrupted_parent"
                ),
            }
            payload["runs"] = {
                role: {
                    "path": str(run.root),
                    "run_id": run.run["run_id"],
                    "segment_id": run.run["continuation"]["segment_id"],
                    "hashes": dict(run.hashes),
                }
                for role, run in runs.items()
            }
            payload["interruption"] = _load_interrupt_evidence(
                request, runs["interrupted_parent"], mismatches
            )
            payload["event_validation"] = {
                role: _validate_run_events(
                    run, allow_legacy=False, mismatches=mismatches
                )
                for role, run in runs.items()
            }
            payload["event_progress_comparisons"] = _compare_event_progress_pairs(
                runs,
                (("uninterrupted", "interrupted_parent", 3),),
                mismatches,
            )
            payload["run_contract"] = _compare_prechild_run_contract(
                runs, request, mismatches
            )
            reference_dir = runs["uninterrupted"].root / "checkpoints/step-3"
            parent_dir = runs["interrupted_parent"].root / "checkpoints/step-3"
            payload["storage"] = {
                "reference_step_3": {
                    "path": str(reference_dir),
                    "inference": v1._validate_inference_payload(reference_dir),
                    "sizes": v1._checkpoint_inventory(reference_dir),
                },
                "parent_step_3": {
                    "path": str(parent_dir),
                    "inference": v1._validate_inference_payload(parent_dir),
                    "sizes": v1._checkpoint_inventory(parent_dir),
                },
            }
            payload["state_comparison"] = v1._compare_checkpoint_pair(
                reference_dir,
                parent_dir,
                step=3,
                scope="pre_child.reference_vs_parent_step_3",
                mismatches=mismatches,
            )
            payload["log_comparison"] = _compare_log_pairs(
                runs,
                (
                    ("uninterrupted", "interrupted_parent", ("train", 3)),
                    ("uninterrupted", "interrupted_parent", ("eval", 3)),
                ),
                require_authoritative=True,
                mismatches=mismatches,
            )
        except BaseException as exc:
            mismatches.append(v1._error_mismatch(exc, scope="pre_child_comparison"))
    try:
        return _publish_receipt(
            output=request.output,
            payload=payload,
            parent=request.interrupted_parent_run_dir,
            initial_parent_tree=initial,
            mismatches=mismatches,
        )
    except BaseException as exc:
        print(
            f"{getattr(exc, 'code', 'wave7_compare_v2.publication')}: {exc}",
            file=sys.stderr,
        )
        return 2


def execute_final(request: FinalRequest) -> int:
    try:
        _validate_input_topology(request)
        v1._validate_interrupt_source_pin(_legacy_request(request))
        v1.assert_absent_artifact_target(request.output)
        initial = v1._tree_identity(
            v1.interrupt._snapshot_run_tree(request.interrupted_parent_run_dir)
        )
    except BaseException as exc:
        print(
            f"{getattr(exc, 'code', 'wave7_compare_v2.preflight')}: {exc}",
            file=sys.stderr,
        )
        return 2
    source_sha256 = v1._sha256_file(Path(__file__).resolve())
    payload = _base_payload(
        schema=FINAL_RECEIPT_SCHEMA,
        source_sha256=source_sha256,
        request=request,
        initial_parent_tree=initial,
    )
    payload.update(
        aliases={},
        checkpoint_selection={},
        lineage={},
        publication_measurements={},
        claim_boundaries={
            "legacy_r4": (
                "diagnostic interpretation only; cannot satisfy r5 event, inference "
                "payload identity, or authoritative integer accuracy gates"
            ),
            "bitwise_cross_launch_reproducibility": "not_claimed",
            "workload_scope": (
                "one uninterrupted and one step3 interrupted continuation to step5"
            ),
        },
    )
    mismatches: list[dict[str, Any]] = payload["mismatches"]
    if source_sha256 != request.expected_source_sha256:
        v1._add_mismatch(
            mismatches,
            code="wave7_compare_v2.source_mismatch",
            scope="source",
            path="comparator_v2.sha256",
            expected=request.expected_source_sha256,
            observed=source_sha256,
        )
    else:
        try:
            runs = {
                "uninterrupted": v1._load_run(
                    request.uninterrupted_run_dir, role="uninterrupted"
                ),
                "interrupted_parent": v1._load_run(
                    request.interrupted_parent_run_dir, role="interrupted_parent"
                ),
                "resume_child": v1._load_run(
                    request.resume_child_run_dir, role="resume_child"
                ),
            }
            payload["runs"] = {
                role: {
                    "path": str(run.root),
                    "run_id": run.run["run_id"],
                    "segment_id": run.run["continuation"]["segment_id"],
                    "hashes": dict(run.hashes),
                }
                for role, run in runs.items()
            }
            payload["interruption"] = _load_interrupt_evidence(
                request, runs["interrupted_parent"], mismatches
            )
            event_version = int(payload["interruption"]["event_version"])
            legacy = event_version == 1
            payload["event_validation"] = {
                role: _validate_run_events(
                    run, allow_legacy=legacy, mismatches=mismatches
                )
                for role, run in runs.items()
            }
            payload["event_progress_comparisons"] = _compare_event_progress_pairs(
                runs,
                (
                    ("uninterrupted", "interrupted_parent", 3),
                    ("uninterrupted", "resume_child", 5),
                ),
                mismatches,
            )
            payload["run_contract"] = _compare_run_contracts(
                runs,
                request,
                event_version=event_version,
                mismatches=mismatches,
            )
            checkpoint_surfaces, manifests = v1._load_checkpoint_surfaces(
                runs, mismatches
            )
            payload["storage"] = checkpoint_surfaces
            payload["lineage"] = v1._validate_lineage(
                runs,
                manifests,
                checkpoint_surfaces,
                payload["interruption"],
                mismatches,
            )
            payload["state_comparisons"] = {
                "step_3": v1._compare_checkpoint_pair(
                    runs["uninterrupted"].root / "checkpoints/step-3",
                    runs["interrupted_parent"].root / "checkpoints/step-3",
                    step=3,
                    scope="reference_vs_parent_step_3",
                    mismatches=mismatches,
                ),
                "step_5": v1._compare_checkpoint_pair(
                    runs["uninterrupted"].root / "checkpoints/step-5",
                    runs["resume_child"].root / "checkpoints/step-5",
                    step=5,
                    scope="reference_vs_child_step_5",
                    mismatches=mismatches,
                ),
            }
            log_pairs = tuple(
                (
                    "uninterrupted",
                    "interrupted_parent" if step <= 3 else "resume_child",
                    ("train", step),
                )
                for step in range(1, 6)
            ) + (("uninterrupted", "interrupted_parent", ("eval", 3)),)
            payload["log_comparison"] = _compare_log_pairs(
                runs,
                log_pairs,
                require_authoritative=not legacy,
                mismatches=mismatches,
            )
            child_index = v1._index_logs(runs["resume_child"].logs, role="resume_child")
            if ("eval", 3) in child_index:
                v1._add_mismatch(
                    mismatches,
                    code="wave7_compare_v2.eval_replay",
                    scope="resume_child",
                    path="logging.eval.step-3",
                    expected="absent",
                    observed="present",
                )
            payload["aliases"] = v1._validate_aliases(runs, mismatches)
            payload["checkpoint_selection"] = _compare_checkpoint_selection(
                runs, mismatches
            )
            payload["publication_measurements"] = v1._publication_measurements(
                _sanitize_publication_runs(runs, event_version=event_version),
                checkpoint_surfaces,
                mismatches,
            )
        except BaseException as exc:
            mismatches.append(v1._error_mismatch(exc, scope="final_comparison"))
    try:
        return _publish_receipt(
            output=request.output,
            payload=payload,
            parent=request.interrupted_parent_run_dir,
            initial_parent_tree=initial,
            mismatches=mismatches,
        )
    except BaseException as exc:
        print(
            f"{getattr(exc, 'code', 'wave7_compare_v2.publication')}: {exc}",
            file=sys.stderr,
        )
        return 2


def verify_pre_child(receipt_path: Path, expected_payload_sha256: str) -> int:
    try:
        receipt = v1._strict_json_file(receipt_path, owner="pre-child receipt")
        observed_digest = receipt.get("receipt_payload_sha256")
        unsigned = dict(receipt)
        unsigned.pop("receipt_payload_sha256", None)
        recomputed = v1._sha256_bytes(canonical_json_bytes(unsigned))
        if (
            receipt.get("schema") != PRE_CHILD_RECEIPT_SCHEMA
            or receipt.get("status") != "passed"
            or receipt.get("child_launch_authorized") is not True
            or observed_digest != expected_payload_sha256
            or recomputed != expected_payload_sha256
            or receipt.get("mismatches") != []
        ):
            raise v1.Wave7CompareError(
                "pre-child receipt is not an exact passed launch authorization",
                code="wave7_compare_v2.pre_child_verification",
            )
    except BaseException as exc:
        print(
            f"{getattr(exc, 'code', 'wave7_compare_v2.pre_child_verification')}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    request = parse_request(argv)
    if isinstance(request, tuple):
        return verify_pre_child(*request)
    if isinstance(request, PreChildRequest):
        return execute_pre_child(request)
    return execute_final(request)


if __name__ == "__main__":
    raise SystemExit(main())
