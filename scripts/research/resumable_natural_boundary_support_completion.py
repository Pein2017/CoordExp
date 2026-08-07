#!/usr/bin/env python3
"""CPU mechanics for resumable natural-boundary support-completion shards.

This module deliberately owns no support rule, model invocation, or scientific
interpretation.  The active runner supplies a completed opaque observation for
each context; this adapter durably records it, schedules it, and can later
regroup it into the runner's unchanged legacy receipt shape.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import threading
from typing import Any

from src.artifacts.evidence_journal import (
    JOURNAL_SCHEMA_VERSION,
    ExecutionEvidenceJournal,
    JournalRecordView,
)
from src.artifacts.json_values import canonical_json_bytes, json_sha256, publish_json_exclusive, validate_json_value
from src.common.errors import ArtifactContractError


SCHEDULE_SCHEMA_VERSION = "natural_boundary_support_completion_lpt_schedule.v1"
MECHANICS_SCHEMA_VERSION = "natural_boundary_support_completion_mechanics.v1"
ADAPTER_SCHEMA_VERSION = "natural_boundary_support_completion_adapter.v1"
EXPECTED_CONSUMER_SHA256 = "9ade7ad861e7c822458f65fbdb39ddda702648d62339533a3faf5e0dd37a52b3"
EXPECTED_MERGER_SHA256 = "9eb7534641be4c87768698719ed84c2f84119959d9fa324b0fc5a72f7cd5edf1"
SCHEDULE_ALGORITHM = "lpt.declared_scalar_equivalent_forward_count.v1"


class SupportShardAdapterError(ValueError):
    """Raised for an immutable-plan, schedule, or receipt contract failure."""


class WorkerSIGTERM(BaseException):
    """Internal control flow raised by the optional worker SIGTERM handler."""


@dataclass(frozen=True)
class LogicalContext:
    context_id: str
    shard_index: int
    plan_position: int
    scalar_equivalent_forward_count: int
    source: Mapping[str, Any]


@dataclass(frozen=True)
class LogicalPlan:
    raw: Mapping[str, Any]
    file_sha256: str
    content_sha256: str
    contexts: tuple[LogicalContext, ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise SupportShardAdapterError(f"source is not a regular non-symlink file: {candidate}")
    source = candidate.resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise SupportShardAdapterError(f"source is not a regular non-symlink file: {candidate}")
    return _sha256_bytes(source.read_bytes())


def load_logical_plan(path: str | Path) -> LogicalPlan:
    """Read the sealed logical plan without modifying its legacy partition."""

    source = Path(path).expanduser().resolve(strict=True)
    raw_bytes = source.read_bytes()
    # The sealed research plan is an immutable, byte-bound input but is not a
    # journal artifact and therefore is not required to use our output newline
    # convention.  Its own canonical content fingerprint remains authoritative.
    try:
        value = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise SupportShardAdapterError("logical plan is not valid JSON") from exc
    if not isinstance(value, Mapping) or not isinstance(value.get("contexts"), list):
        raise SupportShardAdapterError("logical plan lacks a contexts array")
    content_sha = value.get("plan_content_sha256")
    if not isinstance(content_sha, str) or len(content_sha) != 64:
        raise SupportShardAdapterError("logical plan lacks plan_content_sha256")
    body = dict(value)
    body.pop("plan_content_sha256", None)
    if json_sha256(body) != content_sha:
        raise SupportShardAdapterError("logical plan content identity is invalid")
    contexts: list[LogicalContext] = []
    seen: set[str] = set()
    for row in value["contexts"]:
        if not isinstance(row, Mapping):
            raise SupportShardAdapterError("logical plan context is not a mapping")
        context_id = row.get("context_id")
        cost = row.get("scalar_equivalent_forward_count")
        shard = row.get("shard_index")
        position = row.get("context_plan_position")
        if (
            not isinstance(context_id, str)
            or not context_id
            or context_id in seen
            or isinstance(cost, bool)
            or not isinstance(cost, int)
            or cost < 0
            or isinstance(shard, bool)
            or not isinstance(shard, int)
            or shard < 0
            or isinstance(position, bool)
            or not isinstance(position, int)
            or position < 0
        ):
            raise SupportShardAdapterError("logical plan context identity/cost is invalid")
        validate_json_value(dict(row))
        seen.add(context_id)
        contexts.append(LogicalContext(context_id, shard, position, cost, dict(row)))
    if not contexts:
        raise SupportShardAdapterError("logical plan has no contexts")
    return LogicalPlan(dict(value), _sha256_bytes(raw_bytes), content_sha, tuple(contexts))


def project_logical_contexts(
    plan: LogicalPlan, *, context_ids: Sequence[str]
) -> LogicalPlan:
    """Select a bounded mechanics subset while retaining the sealed parent identity.

    This exists only for the real signal smoke.  It does not rewrite ``raw`` or
    claim to be a replacement scientific plan; the selected context mappings
    remain byte-derived projections of that plan.
    """

    requested = tuple(context_ids)
    if not requested or any(not isinstance(item, str) or not item for item in requested):
        raise SupportShardAdapterError("bounded context projection requires context identifiers")
    if len(requested) != len(set(requested)):
        raise SupportShardAdapterError("bounded context projection duplicates a context")
    by_id = {item.context_id: item for item in plan.contexts}
    if any(item not in by_id for item in requested):
        raise SupportShardAdapterError("bounded context projection contains a foreign context")
    return LogicalPlan(
        raw=plan.raw,
        file_sha256=plan.file_sha256,
        content_sha256=plan.content_sha256,
        contexts=tuple(by_id[item] for item in requested),
    )


def plan_physical_slots(plan: LogicalPlan, *, slot_count: int = 8) -> dict[str, Any]:
    """Derive the stable declared-cost LPT schedule, independent of shard IDs."""

    if isinstance(slot_count, bool) or not isinstance(slot_count, int) or slot_count <= 0:
        raise SupportShardAdapterError("slot_count must be a positive integer")
    slots: list[list[LogicalContext]] = [[] for _ in range(slot_count)]
    costs = [0 for _ in range(slot_count)]
    for context in sorted(plan.contexts, key=lambda item: (-item.scalar_equivalent_forward_count, item.context_id)):
        index = min(range(slot_count), key=lambda item: (costs[item], len(slots[item]), item))
        slots[index].append(context)
        costs[index] += context.scalar_equivalent_forward_count
    schedule = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "algorithm": SCHEDULE_ALGORITHM,
        "logical_plan_file_sha256": plan.file_sha256,
        "logical_plan_content_sha256": plan.content_sha256,
        "slot_count": slot_count,
        "slots": [
            {
                "slot_index": index,
                "context_ids": [item.context_id for item in values],
                "context_count": len(values),
                "scalar_equivalent_forward_count": costs[index],
            }
            for index, values in enumerate(slots)
        ],
    }
    schedule["content_sha256"] = json_sha256(schedule)
    return schedule


def validate_schedule(schedule: Mapping[str, Any], plan: LogicalPlan) -> dict[str, Any]:
    """Reject drift instead of regenerating a schedule during continuation."""

    if not isinstance(schedule, Mapping):
        raise SupportShardAdapterError("schedule is not a mapping")
    supplied = dict(schedule)
    digest = supplied.pop("content_sha256", None)
    if digest != json_sha256(supplied):
        raise SupportShardAdapterError("schedule content identity is invalid")
    expected = plan_physical_slots(plan, slot_count=supplied.get("slot_count"))
    if dict(schedule) != expected:
        raise SupportShardAdapterError("schedule differs from the exact logical-plan LPT schedule")
    return expected


def write_once_json(path: str | Path, value: Mapping[str, Any]) -> str:
    """Publish canonical bytes once, allowing only identical re-observation."""

    validate_json_value(dict(value))
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise SupportShardAdapterError(f"output must not be a symlink: {candidate}")
    destination = candidate.resolve()
    payload = canonical_json_bytes(value)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise SupportShardAdapterError(f"output is not a regular non-symlink file: {destination}")
        if destination.read_bytes() != payload:
            raise SupportShardAdapterError(f"refusing to overwrite immutable artifact: {destination}")
    else:
        try:
            publish_json_exclusive(destination, dict(value))
        except ArtifactContractError as exc:
            raise SupportShardAdapterError(f"could not publish immutable artifact: {destination}") from exc
    return _sha256_bytes(payload)


def source_binding_receipt(
    *, consumer_path: str | Path, merger_path: str | Path, plan_path: str | Path
) -> dict[str, Any]:
    """Bind the active untracked consumer by bytes, never by an invented commit."""

    consumer = Path(consumer_path).expanduser().resolve(strict=True)
    merger = Path(merger_path).expanduser().resolve(strict=True)
    plan = load_logical_plan(plan_path)
    consumer_digest = file_sha256(consumer)
    merger_digest = file_sha256(merger)
    if consumer_digest != EXPECTED_CONSUMER_SHA256 or merger_digest != EXPECTED_MERGER_SHA256:
        raise SupportShardAdapterError("active consumer/merger bytes differ from the approved source binding")
    return {
        "schema_version": MECHANICS_SCHEMA_VERSION,
        "kind": "source_binding",
        "consumer": {"path": str(consumer), "sha256": consumer_digest, "git_identity": None},
        "merger": {"path": str(merger), "sha256": merger_digest, "git_identity": None},
        "logical_plan": {"path": str(Path(plan_path).resolve()), "file_sha256": plan.file_sha256, "content_sha256": plan.content_sha256},
        "claim_boundary": "mechanics_only_no_active_unit_mutation_or_scientific_interpretation",
    }


def verify_active_nonmutation(
    *, source_bindings_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Re-hash source-binding inputs after infra work without touching them."""

    bindings_path = Path(source_bindings_path).expanduser()
    if bindings_path.is_symlink() or not bindings_path.is_file():
        raise SupportShardAdapterError("source-binding receipt is not a regular non-symlink file")
    try:
        bindings = json.loads(bindings_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise SupportShardAdapterError("source-binding receipt is unreadable") from exc
    if not isinstance(bindings, Mapping) or bindings.get("schema_version") != "natural_boundary_support_source_bindings.v1":
        raise SupportShardAdapterError("source-binding receipt schema is incompatible")
    sources = bindings.get("sources")
    if not isinstance(sources, Mapping):
        raise SupportShardAdapterError("source-binding receipt lacks source inventory")
    selected = {
        "consumer": sources.get("consumer"),
        "merger": sources.get("merger"),
        "sealed_plan": sources.get("sealed_plan"),
        "census": sources.get("census"),
    }
    observed: dict[str, Any] = {}
    for name, value in selected.items():
        if not isinstance(value, Mapping) or not isinstance(value.get("path"), str) or not isinstance(value.get("sha256"), str):
            raise SupportShardAdapterError(f"source-binding receipt lacks {name} identity")
        digest = file_sha256(value["path"])
        if digest != value["sha256"]:
            raise SupportShardAdapterError(f"{name} bytes differ from the source-binding receipt")
        observed[name] = {"path": value["path"], "sha256": digest}
    receipt = {
        "schema_version": MECHANICS_SCHEMA_VERSION,
        "kind": "active_and_sealed_nonmutation_recheck",
        "source_bindings_sha256": file_sha256(bindings_path),
        "observed": observed,
        "result": "unchanged_at_recheck",
        "claim_boundary": "byte_recheck_only_no_active_unit_edit_merge_cherry_pick_or_scientific_reinterpretation",
    }
    receipt["content_sha256"] = json_sha256(receipt)
    write_once_json(output_path, receipt)
    return receipt


def validate_sealed_consumer_plan(
    *,
    plan_path: str | Path,
    expected_plan_file_sha256: str,
    census_path: str | Path,
    runner_validate_execution_plan: Callable[..., Any],
) -> LogicalPlan:
    """Require the digest-bound consumer's own validator before any execution.

    The adapter accepts the validator as a callable so it never imports an
    untracked active consumer before the adoption glue has checked its digest.
    """

    plan = load_logical_plan(plan_path)
    if plan.file_sha256 != expected_plan_file_sha256:
        raise SupportShardAdapterError("sealed plan raw-byte identity differs from adoption binding")
    try:
        validated, source_info = runner_validate_execution_plan(
            Path(plan_path), expected_plan_sha256=expected_plan_file_sha256, census_path=census_path
        )
    except Exception as exc:  # caller-owned validator error is an admission failure
        raise SupportShardAdapterError("consumer validation rejected the sealed plan before execution") from exc
    if not isinstance(validated, Mapping) or not isinstance(source_info, Mapping):
        raise SupportShardAdapterError("consumer validator returned an invalid plan projection")
    if validated.get("plan_content_sha256") != plan.content_sha256:
        raise SupportShardAdapterError("consumer validator plan content identity differs from logical projection")
    return plan


def _slot_contexts(schedule: Mapping[str, Any], plan: LogicalPlan, slot_index: int) -> tuple[LogicalContext, ...]:
    validated = validate_schedule(schedule, plan)
    slots = validated["slots"]
    if isinstance(slot_index, bool) or not isinstance(slot_index, int) or not 0 <= slot_index < len(slots):
        raise SupportShardAdapterError("slot_index is outside the schedule")
    context_by_id = {item.context_id: item for item in plan.contexts}
    return tuple(context_by_id[item] for item in slots[slot_index]["context_ids"])


def _slot_identity(
    *, execution_identity: Mapping[str, Any], schedule: Mapping[str, Any], slot_index: int
) -> dict[str, Any]:
    validate_json_value(dict(execution_identity))
    return {
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "execution_identity": dict(execution_identity),
        "schedule_sha256": schedule["content_sha256"],
        "physical_slot_index": slot_index,
    }


def open_slot_journal(
    *, root: str | Path, execution_id: str, execution_identity: Mapping[str, Any],
    plan: LogicalPlan, schedule: Mapping[str, Any], slot_index: int, continuation: bool,
) -> tuple[ExecutionEvidenceJournal, tuple[LogicalContext, ...]]:
    """Open one exact-identity journal for one physical slot, before model work."""

    contexts = _slot_contexts(schedule, plan, slot_index)
    identity = _slot_identity(execution_identity=execution_identity, schedule=schedule, slot_index=slot_index)
    expected = [item.context_id for item in contexts]
    context = {
        "logical_plan_file_sha256": plan.file_sha256,
        "logical_plan_content_sha256": plan.content_sha256,
        "schedule_sha256": schedule["content_sha256"],
        "slot_index": slot_index,
    }
    root_path = Path(root).expanduser().resolve()
    if continuation:
        journal = ExecutionEvidenceJournal.open(
            root=root_path, execution_id=execution_id, execution_identity=identity,
            expected_work_item_ids=expected, context=context,
        )
    else:
        journal = ExecutionEvidenceJournal.create(
            root=root_path, execution_id=execution_id, execution_identity=identity,
            expected_work_item_ids=expected, context=context,
        )
    return journal, contexts


def execute_slot(
    *, root: str | Path, execution_id: str, execution_identity: Mapping[str, Any],
    plan: LogicalPlan, schedule: Mapping[str, Any], slot_index: int,
    observe_context: Callable[[Mapping[str, Any]], Mapping[str, Any]], continuation: bool = False,
    stop_after_records: int | None = None,
    cleanup: Callable[[], None] | None = None,
    install_sigterm_handler: bool = False,
) -> tuple[str, tuple[str, ...]]:
    """Perform one explicit attempt; no missing item is retried automatically."""

    journal, contexts = open_slot_journal(
        root=root, execution_id=execution_id, execution_identity=execution_identity,
        plan=plan, schedule=schedule, slot_index=slot_index, continuation=continuation,
    )
    attempt: str | None = None
    previous_sigterm_handler: Any = None
    if install_sigterm_handler:
        if not hasattr(signal, "SIGTERM"):
            raise SupportShardAdapterError("SIGTERM is unavailable on this platform")
        if not hasattr(signal, "signal"):
            raise SupportShardAdapterError("signal handler installation is unavailable")
        if threading.current_thread() is not threading.main_thread():
            raise SupportShardAdapterError("SIGTERM handler requires the main thread")

        def _interrupt_for_sigterm(_signum: int, _frame: Any) -> None:
            raise WorkerSIGTERM("SIGTERM received during context execution")

        previous_sigterm_handler = signal.signal(signal.SIGTERM, _interrupt_for_sigterm)
    try:
        attempt = journal.start_attempt()
        completed = set(journal.completed_work_item_ids)
        accepted: list[str] = []
        for context in contexts:
            if context.context_id in completed:
                continue
            if stop_after_records is not None and len(accepted) >= stop_after_records:
                break
            observation = observe_context(context.source)
            if not isinstance(observation, Mapping) or observation.get("context_id") != context.context_id:
                raise SupportShardAdapterError("observer payload does not bind the scheduled context")
            validate_json_value(dict(observation))
            journal.append_record(work_item_id=context.context_id, payload=dict(observation), attempt_id=attempt)
            accepted.append(context.context_id)
        if set(journal.completed_work_item_ids) == {context.context_id for context in contexts}:
            journal.finalize()
        return attempt, tuple(accepted)
    except BaseException as exc:
        # Journal attempt failure is best effort only; a parent launcher owns
        # the authoritative OS return code/signal diagnosis.
        if attempt is not None:
            try:
                journal.record_attempt_failure(
                    attempt_id=attempt,
                    failure_code=(
                        "worker.sigterm"
                        if isinstance(exc, WorkerSIGTERM)
                        else "worker.observer_exception"
                    ),
                    failure_message=str(exc),
                    exception_type=type(exc).__name__,
                )
            except BaseException:
                pass
        raise
    finally:
        try:
            if callable(cleanup):
                cleanup()
        finally:
            if previous_sigterm_handler is not None:
                signal.signal(signal.SIGTERM, previous_sigterm_handler)
            journal.close()


def materialize_legacy_shard_receipts(
    *, plan: LogicalPlan, schedule: Mapping[str, Any], slot_roots: Sequence[str | Path],
    execution_identity: Mapping[str, Any],
    census_binding: Mapping[str, Any],
    receipt_set_validator: Callable[[Mapping[int, Mapping[str, Any]]], Mapping[str, Any]],
    output_root: str | Path | None = None,
) -> dict[int, Mapping[str, Any]]:
    """Validate, regroup, merger-gate, then publish exact legacy receipts."""

    validated_schedule = validate_schedule(schedule, plan)
    if len(slot_roots) != validated_schedule["slot_count"]:
        raise SupportShardAdapterError("every physical slot journal is required for materialization")
    records: dict[str, JournalRecordView] = {}
    for slot_index, root in enumerate(slot_roots):
        diagnostics = ExecutionEvidenceJournal.inspect_diagnostics(Path(root))
        if diagnostics.snapshot.terminal is None:
            raise SupportShardAdapterError("cannot materialize legacy receipts from a nonterminal slot journal")
        expected_identity = _slot_identity(
            execution_identity=execution_identity,
            schedule=validated_schedule,
            slot_index=slot_index,
        )
        if diagnostics.snapshot.execution_identity_fingerprint != json_sha256(expected_identity):
            raise SupportShardAdapterError("slot journal execution identity is foreign")
        expected_ids = tuple(validated_schedule["slots"][slot_index]["context_ids"])
        expected_context = {
            "logical_plan_file_sha256": plan.file_sha256,
            "logical_plan_content_sha256": plan.content_sha256,
            "schedule_sha256": validated_schedule["content_sha256"],
            "slot_index": slot_index,
        }
        expected_plan_core = {
            "journal_schema_version": JOURNAL_SCHEMA_VERSION,
            "execution_id": diagnostics.snapshot.execution_id,
            "execution_identity": expected_identity,
            "execution_identity_fingerprint": json_sha256(expected_identity),
            "expected_work_item_ids": list(expected_ids),
            "context": expected_context,
            "context_fingerprint": json_sha256(expected_context),
        }
        if (
            diagnostics.snapshot.expected_work_item_ids != expected_ids
            or diagnostics.snapshot.completed_work_item_ids != expected_ids
            or diagnostics.snapshot.context_fingerprint != json_sha256(expected_context)
            or diagnostics.snapshot.plan_fingerprint != json_sha256(expected_plan_core)
        ):
            raise SupportShardAdapterError(
                "slot journal plan, context, or scheduled work-item denominator is foreign"
            )
        for record in diagnostics.records:
            if record.work_item_id in records:
                raise SupportShardAdapterError("duplicate context record across physical slots")
            records[record.work_item_id] = record
    expected = {item.context_id for item in plan.contexts}
    if set(records) != expected:
        raise SupportShardAdapterError("cannot publish legacy receipts from incomplete or foreign journals")
    by_shard: dict[int, list[LogicalContext]] = {}
    for context in plan.contexts:
        by_shard.setdefault(context.shard_index, []).append(context)
    result: dict[int, Mapping[str, Any]] = {}
    for shard_index, contexts in sorted(by_shard.items()):
        ordered = sorted(contexts, key=lambda item: item.plan_position)
        observations: list[Mapping[str, Any]] = []
        for context in ordered:
            payload = records[context.context_id].payload
            if not isinstance(payload, Mapping) or payload.get("context_id") != context.context_id:
                raise SupportShardAdapterError("journal payload cannot be materialized for its logical context")
            # Mechanical acceptance and legacy-receipt eligibility are separate.
            if payload.get("status") != "measured" or not isinstance(payload.get("support_features"), Mapping):
                raise SupportShardAdapterError("journal contains an observation ineligible for a completed legacy receipt")
            observations.append(_thaw_json(payload))
        receipt = build_legacy_receipt(
            plan,
            shard_index=shard_index,
            observations=observations,
            census_binding=census_binding,
        )
        validate_json_value(dict(receipt))
        result[shard_index] = dict(receipt)
    try:
        merger_projection = receipt_set_validator(result)
    except Exception as exc:
        raise SupportShardAdapterError(
            "unchanged merger validator rejected the materialized receipt set"
        ) from exc
    if not isinstance(merger_projection, Mapping):
        raise SupportShardAdapterError(
            "unchanged merger validator returned an invalid projection"
        )
    if output_root is not None:
        for shard_index, receipt in sorted(result.items()):
            write_once_json(Path(output_root) / f"shard-{shard_index}.receipt.json", receipt)
    return result


def materialize_bounded_mechanics_terminal(
    *,
    plan: LogicalPlan,
    schedule: Mapping[str, Any],
    slot_roots: Sequence[str | Path],
    execution_identity: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Publish a mechanics-only terminal projection for a bounded live smoke."""

    validated_schedule = validate_schedule(schedule, plan)
    if len(slot_roots) != validated_schedule["slot_count"]:
        raise SupportShardAdapterError("bounded terminal requires every physical slot journal")
    expected_ids = {item.context_id for item in plan.contexts}
    observed: dict[str, JournalRecordView] = {}
    terminals: list[str] = []
    plan_fingerprints: list[str] = []
    for slot_index, root in enumerate(slot_roots):
        diagnostics = ExecutionEvidenceJournal.inspect_diagnostics(Path(root))
        if diagnostics.snapshot.terminal is None:
            raise SupportShardAdapterError("bounded terminal requires terminal slot journals")
        expected_identity = _slot_identity(
            execution_identity=execution_identity,
            schedule=validated_schedule,
            slot_index=slot_index,
        )
        if diagnostics.snapshot.execution_identity_fingerprint != json_sha256(expected_identity):
            raise SupportShardAdapterError("bounded terminal slot identity is foreign")
        terminals.append(str(diagnostics.snapshot.terminal["content_sha256"]))
        plan_fingerprints.append(diagnostics.snapshot.plan_fingerprint)
        for record in diagnostics.records:
            if record.work_item_id in observed:
                raise SupportShardAdapterError("bounded terminal duplicates a context record")
            observed[record.work_item_id] = record
    if set(observed) != expected_ids:
        raise SupportShardAdapterError("bounded terminal context denominator is incomplete or foreign")
    ordered = [observed[item.context_id] for item in plan.contexts]
    receipt: dict[str, Any] = {
        "schema_version": MECHANICS_SCHEMA_VERSION,
        "kind": "bounded_terminal_materialization",
        "logical_plan_file_sha256": plan.file_sha256,
        "logical_plan_content_sha256": plan.content_sha256,
        "schedule_sha256": validated_schedule["content_sha256"],
        "context_count": len(ordered),
        "context_ids": [item.work_item_id for item in ordered],
        "record_digests": [item.record_digest for item in ordered],
        "payload_fingerprints": [item.payload_fingerprint for item in ordered],
        "slot_terminal_digests": terminals,
        "slot_plan_fingerprints": plan_fingerprints,
        "claim_boundary": "bounded_live_mechanics_only_no_scientific_receipt_or_outcome_interpretation",
    }
    receipt["content_sha256"] = json_sha256(receipt)
    write_once_json(output_path, receipt)
    return receipt


def build_legacy_receipt(
    plan: LogicalPlan,
    *,
    shard_index: int,
    observations: Sequence[Mapping[str, Any]],
    census_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct only the unchanged legacy shard receipt projection.

    This intentionally excludes device, physical-slot, attempt, and retry
    history.  It is a strict compatibility adapter, not a second executor.
    """

    if not isinstance(census_binding, Mapping):
        raise SupportShardAdapterError("legacy receipt requires a census binding")
    contexts = sorted(
        (item for item in plan.contexts if item.shard_index == shard_index),
        key=lambda item: item.plan_position,
    )
    if not contexts or len(observations) != len(contexts):
        raise SupportShardAdapterError("legacy receipt observation denominator is incomplete")
    normalized: list[dict[str, Any]] = []
    for context, observation in zip(contexts, observations, strict=True):
        if not isinstance(observation, Mapping) or observation.get("context_id") != context.context_id:
            raise SupportShardAdapterError("legacy observation order or context identity drifted")
        candidate_ids = context.source.get("candidate_ids")
        scores = observation.get("candidate_scores")
        if (
            observation.get("status") != "measured"
            or not isinstance(candidate_ids, list)
            or not isinstance(scores, Mapping)
            or set(map(str, scores)) != set(map(str, candidate_ids))
            or observation.get("candidate_score_count") != context.scalar_equivalent_forward_count
            or observation.get("candidate_scores_sha256") != json_sha256(dict(sorted(scores.items())))
            or not isinstance(observation.get("support_features"), Mapping)
        ):
            raise SupportShardAdapterError("observation is ineligible for the completed legacy receipt")
        normalized.append(_thaw_json(observation))
    calibration = plan.raw.get("calibration_reuse")
    lineage = plan.raw.get("support_lineage")
    work = plan.raw.get("work")
    if not isinstance(calibration, Mapping) or not isinstance(lineage, Mapping) or not isinstance(work, Mapping):
        raise SupportShardAdapterError("logical plan lacks legacy receipt lineage")
    batch_size = work.get("candidate_batch_size")
    total_cost = int(work.get("scalar_equivalent_forward_count", -1))
    per_shard = work.get("per_shard")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0 or not isinstance(per_shard, list):
        raise SupportShardAdapterError("logical plan legacy work accounting is invalid")
    import math

    batch_estimates = {
        "candidate_batch_size": batch_size,
        "pooled_ceiling": math.ceil(total_cost / batch_size),
        "sum_shard_local_ceilings": sum(math.ceil(int(item["scalar_equivalent_forward_count"]) / batch_size) for item in per_shard if isinstance(item, Mapping)),
        "batching_admitted": False,
        "status": "estimate_only_exact_history_api_scalar_only",
    }
    aliases = {
        "census_file_sha256": "file_sha256",
        "census_self_sha256": "self_sha256",
        "census_s_owner_ids_sha256": "s_owner_ids_sha256",
    }
    if any(not isinstance(census_binding.get(value), str) for value in aliases.values()):
        raise SupportShardAdapterError("census binding is missing a required identity")
    expected_cost = sum(item.scalar_equivalent_forward_count for item in contexts)
    receipt: dict[str, Any] = {
        "schema_version": "natural_boundary_owner_support_completion_execution.v1.receipt.v1",
        "unit_id": plan.raw.get("unit_id"),
        "status": "completed",
        "plan_content_sha256": plan.content_sha256,
        "shard_index": shard_index,
        "num_shards": len({item.shard_index for item in plan.contexts}),
        "batching": {"candidate_batch_size": 1, "batching_admitted": False, "status": "not_admitted_exact_history_api_scalar_only"},
        "batch_estimates": batch_estimates,
        "census_binding": dict(census_binding),
        "support_calibration_sha256": calibration.get("calibration_sha256"),
        "support_rule": _thaw_json(lineage.get("support_rule")),
        "assigned_context_count": len(contexts),
        "assigned_context_ids_sha256": json_sha256([item.context_id for item in contexts]),
        "expected_scalar_forward_count": expected_cost,
        "realized_scalar_forward_count": expected_cost,
        "complete_assigned_observations": True,
        "observations": normalized,
        "failure_log": [],
        "failure_count": 0,
        "failure_log_content_sha256": _sha256_bytes(b""),
        "native_tp_calibration_scored": False,
        "native_tp_other_not_scored": plan.raw.get("scope", {}).get("native_tp_other_not_scored"),
        "support_completion_denominator_only": True,
        "legacy_frozen_candidate_registry_read": False,
        "no_future_or_intervention_leakage": True,
    }
    for receipt_key, binding_key in aliases.items():
        receipt[receipt_key] = census_binding[binding_key]
    receipt["receipt_content_sha256"] = json_sha256(receipt)
    return receipt


def launch_slot_worker(
    *,
    command: Sequence[str],
    journal_root: str | Path,
    mechanics_receipt_path: str | Path,
    physical_slot_index: int,
    logical_plan_file_sha256: str,
    schedule_sha256: str,
    expected_context_count: int,
    terminate_after_first_durable_record: bool = False,
    poll_seconds: float = 0.05,
    timeout_seconds: float = 300.0,
    termination_grace_seconds: float = 10.0,
) -> dict[str, Any]:
    """Observe one child process and publish a separate mechanics-only receipt.

    The caller starts a continuation separately.  This function deliberately
    never retries, opens a journal writer, or gives scientific meaning to an
    exit.  It can request an external ``SIGTERM`` only after a validated record
    is visible, which is useful for the bounded GPU gate without embedding a
    device or output root in infrastructure.
    """

    if not command or any(not isinstance(item, str) or not item for item in command):
        raise SupportShardAdapterError("launcher command must be a nonempty string sequence")
    if isinstance(physical_slot_index, bool) or not isinstance(physical_slot_index, int) or physical_slot_index < 0:
        raise SupportShardAdapterError("physical_slot_index is invalid")
    if poll_seconds <= 0:
        raise SupportShardAdapterError("poll_seconds must be positive")
    if timeout_seconds <= 0 or termination_grace_seconds <= 0:
        raise SupportShardAdapterError("launcher timeout/grace must be positive")
    if isinstance(expected_context_count, bool) or not isinstance(expected_context_count, int) or expected_context_count < 0:
        raise SupportShardAdapterError("expected_context_count is invalid")
    root = Path(journal_root).expanduser().resolve()
    root_existed = root.exists()
    before_attempts: set[str] = set()
    if root_existed:
        before = ExecutionEvidenceJournal.inspect_diagnostics(root)
        before_attempts = {item.attempt_id for item in before.attempts}
    child = subprocess.Popen(list(command), start_new_session=True)
    signal_sent: int | None = None
    deadline = time.monotonic() + timeout_seconds
    terminate_deadline: float | None = None
    while child.poll() is None:
        if terminate_after_first_durable_record and signal_sent is None:
            try:
                current = ExecutionEvidenceJournal.inspect_diagnostics(root)
            except (ArtifactContractError, FileNotFoundError, OSError):
                current = None
            if current is not None and current.last_durable_record is not None:
                os.killpg(child.pid, signal.SIGTERM)
                signal_sent = signal.SIGTERM
                terminate_deadline = time.monotonic() + termination_grace_seconds
        if signal_sent is None and time.monotonic() >= deadline:
            os.killpg(child.pid, signal.SIGTERM)
            signal_sent = signal.SIGTERM
            terminate_deadline = time.monotonic() + termination_grace_seconds
        elif terminate_deadline is not None and time.monotonic() >= terminate_deadline:
            os.killpg(child.pid, signal.SIGKILL)
            signal_sent = signal.SIGKILL
            terminate_deadline = None
        time.sleep(poll_seconds)
    return_code = child.wait()
    diagnostics = None
    try:
        diagnostics = ExecutionEvidenceJournal.inspect_diagnostics(root)
    except (ArtifactContractError, FileNotFoundError, OSError) as exc:
        journal_absent = not root.exists()
        journal_inspection = {
            "status": "absent" if journal_absent else "invalid",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    else:
        journal_inspection = {"status": "validated", "error_type": None, "message": None}

    new_attempts = (
        []
        if diagnostics is None
        else [item for item in diagnostics.attempts if item.attempt_id not in before_attempts]
    )
    if len(new_attempts) > 1:
        raise SupportShardAdapterError("worker published more than one new attempt; launcher cannot attribute exit")
    last = None if diagnostics is None else diagnostics.last_durable_record
    accepted: int | None = None if diagnostics is None else len(diagnostics.records)
    if diagnostics is None and journal_inspection["status"] == "absent":
        accepted = 0
    missing: int | None = None if accepted is None else expected_context_count - accepted
    if missing is not None and missing < 0:
        raise SupportShardAdapterError("journal accepted more records than the bound physical slot")
    receipt = {
        "schema_version": MECHANICS_SCHEMA_VERSION,
        "kind": "worker_exit",
        "physical_slot_index": physical_slot_index,
        "command": list(command),
        "child_pid": child.pid,
        "attempt_id": new_attempts[0].attempt_id if new_attempts else None,
        "return_code": return_code,
        "terminating_signal": -return_code if return_code < 0 else None,
        "external_signal_sent": signal_sent,
        "logical_plan_file_sha256": logical_plan_file_sha256,
        "schedule_sha256": schedule_sha256,
        "accepted_context_count": accepted,
        "missing_context_count": missing,
        "last_durable_record": (
            None if last is None else {
                "sequence": last.sequence,
                "work_item_id": last.work_item_id,
                "record_digest": last.record_digest,
            }
        ),
        "journal_terminal": None if diagnostics is None else diagnostics.snapshot.terminal is not None,
        "temporary_paths": (
            []
            if diagnostics is None and journal_inspection["status"] == "absent"
            else None
            if diagnostics is None
            else [str(item) for item in diagnostics.snapshot.temporary_paths]
        ),
        "journal_inspection": journal_inspection,
        "claim_boundary": "mechanics_only_no_exit_interpretation_or_automatic_retry",
    }
    receipt["content_sha256"] = json_sha256(receipt)
    write_once_json(mechanics_receipt_path, receipt)
    return receipt


def _thaw_json(value: Any) -> Any:
    """Copy journal's defensive read projection into a canonical caller value."""

    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
