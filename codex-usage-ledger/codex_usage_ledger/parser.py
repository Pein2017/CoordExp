from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .model import USAGE_FIELDS, TokenEvent, TurnContext, Usage, UsageReceipt


_MARKERS = (
    '"type":"session_meta"',
    '"type": "session_meta"',
    '"type":"turn_context"',
    '"type": "turn_context"',
    '"type":"event_msg"',
    '"type": "event_msg"',
    '"type":"response_item"',
    '"type": "response_item"',
    '"type":"token_usage_record"',
    '"type": "token_usage_record"',
)


def _short_label(
    agent_path: str | None, agent_role: str | None, thread_id: str | None
) -> str:
    if agent_path:
        return agent_path.rstrip("/").rsplit("/", 1)[-1]
    if agent_role:
        return agent_role
    return thread_id or "unknown-thread"


def _timestamp(obj: dict[str, Any]) -> str | None:
    value = obj.get("timestamp")
    return value if isinstance(value, str) else None


def _parse_timestamp(value: str | None) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalise_boundary(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _in_window(
    timestamp: str | None,
    since: datetime | None,
    until: datetime | None,
) -> tuple[bool, bool]:
    """Return (included, invalid_timestamp) for a half-open receipt window."""

    if since is None and until is None:
        return True, False
    parsed = _parse_timestamp(timestamp)
    if parsed is None:
        return False, True
    if since is not None and parsed < since:
        return False, False
    if until is not None and parsed >= until:
        return False, False
    return True, False


@dataclass
class SessionRecord:
    file: str
    file_size_bytes: int
    thread_id: str | None = None
    session_id: str | None = None
    parent_thread_id: str | None = None
    forked_from_id: str | None = None
    thread_source: str | None = None
    agent_role: str | None = None
    agent_nickname: str | None = None
    agent_path: str | None = None
    model_provider: str | None = None
    cli_version: str | None = None
    cwd: str | None = None
    history_mode: str | None = None
    multi_agent_version: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    status: str = "unknown"
    first_total_usage: Usage | None = None
    latest_total_usage: Usage | None = None
    latest_last_usage: Usage | None = None
    token_count_events: int = 0
    parse_errors: int = 0
    usage_start_index: int | None = None
    usage_baseline: Usage | None = None
    session_meta_seen: bool = False
    task_start_markers: dict[str, tuple[int, Usage]] = field(
        default_factory=dict, repr=False
    )
    task_started_events: list[dict[str, Any]] = field(default_factory=list)
    task_completed_events: list[dict[str, Any]] = field(default_factory=list)
    sub_agent_activity_events: list[dict[str, Any]] = field(default_factory=list)
    agent_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    turn_contexts: list[TurnContext] = field(default_factory=list)
    token_events: list[TokenEvent] = field(default_factory=list)
    modern_usage_records_seen: int = 0
    modern_usage_records_invalid: int = 0
    usage_receipts: list[UsageReceipt] = field(default_factory=list, repr=False)
    accounted_usage_receipts: list[UsageReceipt] | None = field(
        default=None, repr=False
    )
    accounted_token_events: list[TokenEvent] | None = field(default=None, repr=False)
    accounted_token_baseline: Usage | None = field(default=None, repr=False)
    usage_reconciliation_warnings: list[str] = field(default_factory=list, repr=False)

    @property
    def is_subagent(self) -> bool:
        return self.thread_source == "subagent" or self.parent_thread_id is not None

    @property
    def task_label(self) -> str:
        return _short_label(self.agent_path, self.agent_role, self.thread_id)

    @property
    def models(self) -> list[str]:
        return sorted({ctx.model for ctx in self.turn_contexts if ctx.model})

    @property
    def efforts(self) -> list[str]:
        return sorted({ctx.effort for ctx in self.turn_contexts if ctx.effort})

    @property
    def model(self) -> str | None:
        values = self.models
        return values[0] if len(values) == 1 else None

    @property
    def effort(self) -> str | None:
        values = self.efforts
        return values[0] if len(values) == 1 else None

    @property
    def usage_format(self) -> str:
        if self.modern_usage_records_seen and self.token_count_events:
            return "mixed"
        if self.modern_usage_records_seen:
            return "modern"
        if self.token_count_events:
            return "legacy"
        return "none"

    def _context_for_receipt(self, receipt: UsageReceipt) -> TurnContext | None:
        """Resolve a response receipt to the nearest persisted route context."""

        candidates = [
            context
            for context in self.turn_contexts
            if receipt.turn_id and context.turn_id == receipt.turn_id
        ]
        if candidates:
            return candidates[-1]

        receipt_time = _parse_timestamp(receipt.timestamp)
        if receipt_time is not None:
            prior = [
                context
                for context in self.turn_contexts
                if (context_time := _parse_timestamp(context.timestamp)) is not None
                and context_time <= receipt_time
            ]
            if prior:
                return prior[-1]
        return None

    def route_usage(self) -> list[dict[str, Any]]:
        """Attribute response or cumulative usage to model/effort contexts."""

        grouped: dict[tuple[str | None, str | None], Usage] = {}
        turn_ids: dict[tuple[str | None, str | None], set[str]] = {}

        if self.modern_usage_records_seen:
            receipts = (
                self.accounted_usage_receipts
                if self.accounted_usage_receipts is not None
                else self.usage_receipts
            )
            for receipt in receipts:
                context = self._context_for_receipt(receipt)
                key = (
                    context.model if context and context.model else self.model,
                    context.effort if context and context.effort else self.effort,
                )
                grouped[key] = grouped.get(key, Usage()).add(receipt.usage)
                if receipt.turn_id:
                    turn_ids.setdefault(key, set()).add(receipt.turn_id)
        else:
            start_index = self.usage_start_index or 0
            if self.accounted_token_events is not None:
                events = self.accounted_token_events
                previous = (
                    self.accounted_token_baseline or self.usage_baseline or Usage()
                )
            else:
                events = self.token_events[start_index:]
                previous = self.usage_baseline or Usage()
            for event in events:
                delta = event.total.delta_from(previous)
                previous = event.total
                if delta.is_zero:
                    continue
                context = (
                    self.turn_contexts[event.context_index]
                    if 0 <= event.context_index < len(self.turn_contexts)
                    else None
                )
                key = (
                    context.model if context else None,
                    context.effort if context else None,
                )
                grouped[key] = grouped.get(key, Usage()).add(delta)
                if context and context.turn_id:
                    turn_ids.setdefault(key, set()).add(context.turn_id)

        if (
            not grouped
            and self.latest_total_usage is not None
            and self.accounted_token_events is None
            and not self.modern_usage_records_seen
        ):
            scoped = self.latest_total_usage.subtract(self.usage_baseline or Usage())
            if not scoped.is_zero:
                grouped[(self.model, self.effort)] = scoped

        return [
            {
                "model": model,
                "effort": effort,
                "turn_ids": sorted(turn_ids.get((model, effort), set())),
                "usage": usage.to_dict(),
            }
            for (model, effort), usage in sorted(
                grouped.items(), key=lambda item: (item[0][0] or "", item[0][1] or "")
            )
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "file_size_bytes": self.file_size_bytes,
            "thread_id": self.thread_id,
            "session_id": self.session_id,
            "parent_thread_id": self.parent_thread_id,
            "forked_from_id": self.forked_from_id,
            "thread_source": self.thread_source,
            "is_subagent": self.is_subagent,
            "task_label": self.task_label,
            "agent_role": self.agent_role,
            "agent_nickname": self.agent_nickname,
            "agent_path": self.agent_path,
            "model_provider": self.model_provider,
            "model": self.model,
            "models": self.models,
            "effort": self.effort,
            "efforts": self.efforts,
            "cli_version": self.cli_version,
            "cwd": self.cwd,
            "history_mode": self.history_mode,
            "multi_agent_version": self.multi_agent_version,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "first_total_usage": (
                self.first_total_usage.to_dict() if self.first_total_usage else None
            ),
            "latest_total_usage": (
                self.latest_total_usage.to_dict() if self.latest_total_usage else None
            ),
            "latest_last_usage": (
                self.latest_last_usage.to_dict() if self.latest_last_usage else None
            ),
            "usage_format": self.usage_format,
            "modern_usage_records_seen": self.modern_usage_records_seen,
            "modern_usage_records_invalid": self.modern_usage_records_invalid,
            "usage_receipt_count": len(self.usage_receipts),
            "accounted_usage_receipt_count": (
                len(self.accounted_usage_receipts)
                if self.accounted_usage_receipts is not None
                else None
            ),
            "usage_reconciliation_warnings": self.usage_reconciliation_warnings,
            "token_count_events": self.token_count_events,
            "parse_errors": self.parse_errors,
            "task_started_events": self.task_started_events,
            "task_completed_events": self.task_completed_events,
            "sub_agent_activity_events": self.sub_agent_activity_events,
            "agent_tool_calls": self.agent_tool_calls,
            "turn_contexts": [ctx.to_dict() for ctx in self.turn_contexts],
            "usage_by_route": self.route_usage(),
            "usage_baseline": (
                self.usage_baseline.to_dict() if self.usage_baseline else None
            ),
            "measured_usage": self.scoped_usage().to_dict(),
            "usage_scope": (
                "response_receipts_deduplicated_by_thread_response"
                if self.modern_usage_records_seen
                else "task_delta_from_pre_task_baseline"
                if self.usage_start_index is not None
                else "thread_cumulative; subtract an external baseline for resumed/forked work"
            ),
        }

    def scoped_usage(self) -> Usage:
        total = Usage()
        for route in self.route_usage():
            total = total.add(Usage.from_mapping(route.get("usage")))
        return total


def _new_record(path: Path) -> SessionRecord:
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return SessionRecord(file=str(path), file_size_bytes=size)


def _record_session_meta(record: SessionRecord, payload: dict[str, Any]) -> None:
    # Forked subagent rollouts can embed the parent thread's session_meta and
    # token history after the child's own metadata. Keep the first metadata
    # envelope, which identifies the rollout file being measured.
    if record.session_meta_seen:
        return
    record.session_meta_seen = True
    record.thread_id = (
        payload.get("id") if isinstance(payload.get("id"), str) else record.thread_id
    )
    record.session_id = (
        payload.get("session_id")
        if isinstance(payload.get("session_id"), str)
        else record.session_id
    )
    for attr, key in (
        ("parent_thread_id", "parent_thread_id"),
        ("forked_from_id", "forked_from_id"),
        ("thread_source", "thread_source"),
        ("agent_role", "agent_role"),
        ("agent_nickname", "agent_nickname"),
        ("agent_path", "agent_path"),
        ("model_provider", "model_provider"),
        ("cli_version", "cli_version"),
        ("cwd", "cwd"),
        ("history_mode", "history_mode"),
        ("multi_agent_version", "multi_agent_version"),
        ("started_at", "timestamp"),
    ):
        value = payload.get(key)
        if isinstance(value, str):
            setattr(record, attr, value)


def _record_turn_context(record: SessionRecord, obj: dict[str, Any]) -> None:
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        return
    collaboration_mode = payload.get("collaboration_mode")
    settings = (
        collaboration_mode.get("settings")
        if isinstance(collaboration_mode, dict)
        else {}
    )
    if not isinstance(settings, dict):
        settings = {}
    turn_id = (
        payload.get("turn_id") if isinstance(payload.get("turn_id"), str) else None
    )
    record.turn_contexts.append(
        TurnContext(
            timestamp=_timestamp(obj),
            turn_id=turn_id,
            model=payload.get("model")
            if isinstance(payload.get("model"), str)
            else None,
            effort=(
                payload.get("effort")
                if isinstance(payload.get("effort"), str)
                else settings.get("reasoning_effort")
                if isinstance(settings.get("reasoning_effort"), str)
                else None
            ),
            multi_agent_version=(
                payload.get("multi_agent_version")
                if isinstance(payload.get("multi_agent_version"), str)
                else None
            ),
        )
    )
    # In a forked rollout, the parent task_started event can precede the
    # child's own task_started event. Select the first task boundary that has a
    # matching turn_context instead of blindly using the first task_started.
    marker = record.task_start_markers.get(turn_id) if turn_id else None
    if marker is not None and record.usage_start_index is None:
        record.usage_start_index, record.usage_baseline = marker


def _modern_usage(value: Any) -> Usage | None:
    if not isinstance(value, dict):
        return None
    for field_name in USAGE_FIELDS:
        if field_name not in value:
            return None
        field_value = value.get(field_name, 0)
        if not isinstance(field_value, int) or isinstance(field_value, bool):
            return None
        if field_value < 0:
            return None
    usage = Usage.from_mapping(value)
    if usage.total_tokens != usage.input_tokens + usage.output_tokens:
        return None
    if usage.cached_input_tokens > usage.input_tokens:
        return None
    if usage.reasoning_output_tokens > usage.output_tokens:
        return None
    return usage


def _record_usage_receipt(record: SessionRecord, obj: dict[str, Any]) -> None:
    record.modern_usage_records_seen += 1
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        record.modern_usage_records_invalid += 1
        return
    usage = _modern_usage(payload.get("usage"))
    if usage is None:
        record.modern_usage_records_invalid += 1
        return

    def _string(name: str) -> str | None:
        value = payload.get(name)
        return value if isinstance(value, str) and value else None

    receipt = UsageReceipt(
        timestamp=_timestamp(obj),
        thread_id=_string("thread_id"),
        turn_id=_string("turn_id"),
        session_id=_string("session_id"),
        root_turn_id=_string("root_turn_id"),
        response_id=_string("response_id"),
        usage=usage,
    )
    record.usage_receipts.append(receipt)
    record.ended_at = receipt.timestamp or record.ended_at


def _record_event(record: SessionRecord, obj: dict[str, Any]) -> None:
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        return
    event_type = payload.get("type")
    if event_type == "sub_agent_activity":
        record.sub_agent_activity_events.append(
            {
                "timestamp": _timestamp(obj),
                "event_id": payload.get("event_id"),
                "agent_thread_id": payload.get("agent_thread_id"),
                "agent_path": payload.get("agent_path"),
                "kind": payload.get("kind"),
                "occurred_at_ms": payload.get("occurred_at_ms"),
            }
        )
        return
    if event_type == "token_count":
        info = payload.get("info")
        if not isinstance(info, dict):
            return
        total = Usage.from_mapping(info.get("total_token_usage"))
        last = Usage.from_mapping(info.get("last_token_usage"))
        if record.first_total_usage is None:
            record.first_total_usage = total
        record.latest_total_usage = total
        record.latest_last_usage = last
        record.token_events.append(
            TokenEvent(
                timestamp=_timestamp(obj),
                context_index=len(record.turn_contexts) - 1,
                total=total,
                last=last,
            )
        )
        record.token_count_events += 1
        record.ended_at = _timestamp(obj) or record.ended_at
        return
    if event_type == "task_started":
        turn_id = payload.get("turn_id")
        if isinstance(turn_id, str):
            record.task_start_markers.setdefault(
                turn_id,
                (len(record.token_events), record.latest_total_usage or Usage()),
            )
        event = {
            "timestamp": _timestamp(obj),
            "turn_id": payload.get("turn_id"),
            "started_at": payload.get("started_at"),
            "model_context_window": payload.get("model_context_window"),
        }
        record.task_started_events.append(event)
        if record.started_at is None:
            record.started_at = _timestamp(obj)
        return
    if event_type == "task_complete":
        event = {
            "timestamp": _timestamp(obj),
            "turn_id": payload.get("turn_id"),
            "started_at": payload.get("started_at"),
            "completed_at": payload.get("completed_at"),
            "duration_ms": payload.get("duration_ms"),
            "time_to_first_token_ms": payload.get("time_to_first_token_ms"),
        }
        record.task_completed_events.append(event)
        record.ended_at = _timestamp(obj) or record.ended_at


def _record_response_item(record: SessionRecord, obj: dict[str, Any]) -> None:
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        return
    item_type = payload.get("type")
    if item_type not in {"function_call", "custom_tool_call"}:
        return
    name = payload.get("name")
    if name not in {
        "spawn_agent",
        "followup_task",
        "send_message",
        "interrupt_agent",
    }:
        return
    call_id = payload.get("call_id") or payload.get("id")
    record.agent_tool_calls.append(
        {
            "timestamp": _timestamp(obj),
            "call_id": call_id,
            "name": name,
        }
    )


def parse_rollout(path: str | Path) -> SessionRecord:
    """Parse only compact metadata and usage fields from one rollout JSONL file."""

    rollout_path = Path(path)
    record = _new_record(rollout_path)
    try:
        with rollout_path.open("r", encoding="utf-8", errors="replace") as stream:
            for raw_line in stream:
                if not any(marker in raw_line for marker in _MARKERS):
                    continue
                try:
                    obj = json.loads(raw_line)
                except json.JSONDecodeError:
                    record.parse_errors += 1
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("type") == "session_meta":
                    payload = obj.get("payload")
                    if isinstance(payload, dict):
                        _record_session_meta(record, payload)
                elif obj.get("type") == "turn_context":
                    _record_turn_context(record, obj)
                elif obj.get("type") == "event_msg":
                    _record_event(record, obj)
                elif obj.get("type") == "response_item":
                    _record_response_item(record, obj)
                elif obj.get("type") == "token_usage_record":
                    _record_usage_receipt(record, obj)
    except OSError:
        record.parse_errors += 1

    if record.started_at is None:
        record.started_at = record.ended_at
    if record.usage_start_index is None and record.task_start_markers:
        # A malformed or older rollout may omit turn_context. Preserve a useful
        # fallback while making the ambiguity visible through the scope field.
        record.usage_start_index, record.usage_baseline = next(
            iter(record.task_start_markers.values())
        )
    if record.modern_usage_records_seen:
        # A fork may embed inherited response receipts. Keep them available for
        # reconciliation diagnostics, but never charge them to this file's owner.
        record.accounted_usage_receipts = [
            receipt
            for receipt in record.usage_receipts
            if receipt.key is not None
            and record.thread_id is not None
            and receipt.thread_id == record.thread_id
        ]
    if record.task_completed_events:
        record.status = "completed"
    elif record.task_started_events:
        record.status = "incomplete_or_active"
    elif record.latest_total_usage is not None or record.modern_usage_records_seen:
        record.status = "usage_recorded"
    else:
        record.status = "metadata_only"
    return record


def _receipt_usage_matches(left: UsageReceipt, right: UsageReceipt) -> bool:
    return left.usage == right.usage


def _receipt_metadata_matches(left: UsageReceipt, right: UsageReceipt) -> bool:
    return (
        left.timestamp == right.timestamp
        and left.thread_id == right.thread_id
        and left.turn_id == right.turn_id
        and left.session_id == right.session_id
        and left.root_turn_id == right.root_turn_id
        and left.response_id == right.response_id
    )


def _add_warning(record: SessionRecord, warning: str) -> None:
    if warning not in record.usage_reconciliation_warnings:
        record.usage_reconciliation_warnings.append(warning)


def _legacy_window(
    record: SessionRecord,
    since: datetime | None,
    until: datetime | None,
) -> tuple[list[TokenEvent], Usage, int, int]:
    """Select legacy events and carry the last pre-window cumulative baseline."""

    start_index = record.usage_start_index or 0
    selected: list[TokenEvent] = []
    baseline = record.usage_baseline or Usage()
    invalid_timestamps = 0
    out_of_window = 0
    for index, event in enumerate(record.token_events):
        if index < start_index:
            continue
        included, invalid = _in_window(event.timestamp, since, until)
        if invalid:
            invalid_timestamps += 1
            continue
        if included:
            selected.append(event)
        else:
            out_of_window += 1
            event_time = _parse_timestamp(event.timestamp)
            if since is not None and event_time is not None and event_time < since:
                baseline = event.total
    return selected, baseline, invalid_timestamps, out_of_window


def reconcile_usage(
    records: Iterable[SessionRecord],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, Any]:
    """Deduplicate modern receipts and window legacy events before reporting.

    Modern response receipts are authoritative whenever a file contains them;
    legacy cumulative token events remain the fallback for older files.
    """

    since = _normalise_boundary(since)
    until = _normalise_boundary(until)
    if since is not None and until is not None and since >= until:
        raise ValueError("receipt window since must be earlier than until")

    materialized = list(records)
    groups: dict[tuple[str, str], list[tuple[SessionRecord, UsageReceipt]]] = {}
    stats: dict[str, Any] = {
        "format_counts": {"modern": 0, "legacy": 0, "mixed": 0, "none": 0},
        "modern_files_seen": 0,
        "modern_records_seen": 0,
        "modern_receipts_seen": 0,
        "valid_receipts": 0,
        "accounted_receipts": 0,
        "modern_records_invalid": 0,
        "duplicate_receipts": 0,
        "metadata_conflicts": 0,
        "conflicting_duplicates": 0,
        "foreign_receipts": 0,
        "missing_identity_receipts": 0,
        "unresolved_owner_receipts": 0,
        "out_of_window_receipts": 0,
        "invalid_receipt_timestamps": 0,
        "legacy_token_count_events": 0,
        "legacy_events_accounted": 0,
        "legacy_events_out_of_window": 0,
        "invalid_legacy_timestamps": 0,
    }

    for record in materialized:
        record.accounted_usage_receipts = (
            [] if record.modern_usage_records_seen else None
        )
        record.accounted_token_events = None
        record.accounted_token_baseline = None
        record.usage_reconciliation_warnings = []
        format_name = record.usage_format
        stats["format_counts"][format_name] += 1
        if record.modern_usage_records_seen:
            stats["modern_files_seen"] += 1
        stats["modern_records_seen"] += record.modern_usage_records_seen
        stats["modern_receipts_seen"] += record.modern_usage_records_seen
        stats["valid_receipts"] += len(record.usage_receipts)
        stats["modern_records_invalid"] += record.modern_usage_records_invalid
        stats["legacy_token_count_events"] += record.token_count_events

        if record.modern_usage_records_seen:
            for receipt in record.usage_receipts:
                included, invalid_timestamp = _in_window(
                    receipt.timestamp, since, until
                )
                if invalid_timestamp:
                    stats["invalid_receipt_timestamps"] += 1
                    _add_warning(record, "invalid_receipt_timestamp")
                key = receipt.key
                if key is None:
                    stats["missing_identity_receipts"] += 1
                    _add_warning(record, "receipt_missing_thread_or_response_id")
                    continue
                if record.thread_id is None:
                    stats["unresolved_owner_receipts"] += 1
                    _add_warning(record, "receipt_owner_unresolved")
                    continue
                if receipt.thread_id != record.thread_id:
                    stats["foreign_receipts"] += 1
                    _add_warning(record, "foreign_receipt_excluded")
                    continue
                if invalid_timestamp:
                    continue
                if not included:
                    stats["out_of_window_receipts"] += 1
                    continue
                groups.setdefault(key, []).append((record, receipt))
        elif since is not None or until is not None:
            (
                record.accounted_token_events,
                record.accounted_token_baseline,
                invalid_timestamps,
                out_of_window,
            ) = _legacy_window(record, since, until)
            stats["legacy_events_accounted"] += len(record.accounted_token_events)
            stats["invalid_legacy_timestamps"] += invalid_timestamps
            stats["legacy_events_out_of_window"] += out_of_window
        else:
            stats["legacy_events_accounted"] += record.token_count_events

    for key, occurrences in groups.items():
        first_record, first_receipt = occurrences[0]
        if all(
            _receipt_usage_matches(first_receipt, receipt)
            for _, receipt in occurrences[1:]
        ):
            stats["duplicate_receipts"] += len(occurrences) - 1
            if any(
                not _receipt_metadata_matches(first_receipt, receipt)
                for _, receipt in occurrences[1:]
            ):
                stats["metadata_conflicts"] += 1
                for duplicate_record, _ in occurrences:
                    _add_warning(
                        duplicate_record, "duplicate_receipt_metadata_conflict"
                    )
            if first_record.accounted_usage_receipts is not None:
                first_record.accounted_usage_receipts.append(first_receipt)
            for duplicate_record, _ in occurrences[1:]:
                _add_warning(duplicate_record, "duplicate_receipt_suppressed")
            stats["accounted_receipts"] += 1
            continue

        stats["conflicting_duplicates"] += 1
        for conflicting_record, _ in occurrences:
            _add_warning(conflicting_record, "conflicting_duplicate_receipt_excluded")

    return stats
