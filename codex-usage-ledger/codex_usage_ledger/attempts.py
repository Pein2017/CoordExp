from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from math import ceil
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from .model import USAGE_FIELDS


DISPOSITIONS = {"accepted", "rework", "escalated", "failed", "unknown"}
POLICIES = {"strict", "completed", "followup_aware"}


def load_outcomes(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load optional human/lead dispositions keyed by attempt or thread ID."""

    if path is None:
        return {}
    outcomes: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid outcomes JSONL at line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"outcomes line {line_number} must be an object")
            key = value.get("attempt_id") or value.get("thread_id")
            disposition = value.get("disposition")
            if not isinstance(key, str) or not key:
                raise ValueError(
                    f"outcomes line {line_number} needs attempt_id or thread_id"
                )
            if disposition not in DISPOSITIONS - {"unknown"}:
                raise ValueError(
                    f"outcomes line {line_number} has invalid disposition: {disposition!r}"
                )
            if key in outcomes:
                raise ValueError(f"duplicate outcome identifier: {key}")
            outcomes[key] = dict(value)
    return outcomes


def _event_sort_key(event: dict[str, Any]) -> tuple[int, str]:
    occurred = event.get("occurred_at_ms")
    if isinstance(occurred, int):
        return occurred, event.get("timestamp") or ""
    return 0, event.get("timestamp") or ""


def _activity_index(
    records: Iterable[Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    by_thread: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    tool_names: dict[str, str] = {}
    for record in records:
        for event in getattr(record, "sub_agent_activity_events", []):
            thread_id = event.get("agent_thread_id")
            event_id = event.get("event_id")
            if isinstance(thread_id, str) and isinstance(event_id, str):
                by_thread[thread_id].setdefault(event_id, event)
        for call in getattr(record, "agent_tool_calls", []):
            call_id = call.get("call_id")
            name = call.get("name")
            if isinstance(call_id, str) and isinstance(name, str):
                tool_names.setdefault(call_id, name)
    return (
        {
            thread_id: sorted(events.values(), key=_event_sort_key)
            for thread_id, events in by_thread.items()
        },
        tool_names,
    )


def _proxy_disposition(
    status: str, followup_count: int, policy: str
) -> tuple[str, str]:
    if policy == "strict":
        return "unknown", "no_explicit_disposition"
    if status != "completed":
        return "failed", "rollout_terminal_status"
    if policy == "followup_aware" and followup_count:
        return "rework", "completed_with_observed_followup"
    return "accepted", "completed_without_observed_rework"


def annotate_attempts(
    items: list[dict[str, Any]],
    records: Iterable[Any],
    outcomes: dict[str, dict[str, Any]] | None = None,
    policy: str = "strict",
) -> list[dict[str, Any]]:
    """Attach one parent-invocation attempt identity and disposition to each item.

    A real acceptance label is optional. Without one, ``strict`` leaves the
    disposition unknown; the other policies are explicitly named proxies and
    must not be interpreted as human acceptance.
    """

    if policy not in POLICIES:
        raise ValueError(f"unknown disposition policy: {policy}")
    activity_by_thread, tool_names = _activity_index(records)
    labels = outcomes or {}
    annotated: list[dict[str, Any]] = []
    for item in items:
        thread_id = item.get("thread_id")
        thread_key = thread_id if isinstance(thread_id, str) else None
        events = activity_by_thread.get(thread_key or "", [])
        started = next(
            (event for event in events if event.get("kind") == "started"), None
        )
        started_event_id = started.get("event_id") if started else None
        attempt_id: str = (
            started_event_id
            if isinstance(started_event_id, str)
            else f"thread:{thread_key or 'unknown'}"
        )
        interactions = [event for event in events if event.get("kind") == "interacted"]
        completions = [event for event in events if event.get("kind") == "completed"]
        followups = []
        for event in interactions:
            event_id = event.get("event_id")
            if (
                isinstance(event_id, str)
                and tool_names.get(event_id) == "followup_task"
            ):
                followups.append(event)
        interruptions = [
            event for event in events if event.get("kind") == "interrupted"
        ]
        label = labels.get(attempt_id) or (
            labels.get(thread_key) if thread_key else None
        )
        if label is not None:
            disposition = label.get("disposition")
            disposition_source = "explicit_outcome"
            note = label.get("note")
        else:
            disposition, disposition_source = _proxy_disposition(
                item.get("status") or "unknown", len(followups), policy
            )
            note = None
        ambiguity_reasons = []
        if label is None and policy != "strict" and followups:
            ambiguity_reasons.append("followup_semantics_unverified")
        if len(completions) > 1:
            ambiguity_reasons.append("multiple_completion_events")
        if len(interactions) > len(followups):
            ambiguity_reasons.append("non_followup_interaction")
        activity_kinds = {
            kind for event in events if isinstance((kind := event.get("kind")), str)
        }
        attempt = {
            "attempt_id": attempt_id,
            "child_thread_id": thread_id,
            "parent_thread_id": item.get("parent_thread_id"),
            "spawn_event_id": started.get("event_id") if started else None,
            "agent_path": item.get("agent_path"),
            "agent_role": item.get("agent_role"),
            "followup_count": len(followups),
            "interaction_count": len(interactions),
            "completion_event_count": len(completions),
            "interrupted_count": len(interruptions),
            "activity_kinds": sorted(activity_kinds),
            "proxy_ambiguity_reasons": ambiguity_reasons,
            "disposition": disposition,
            "disposition_source": disposition_source,
            "note": note if isinstance(note, str) else None,
        }
        enriched = dict(item)
        enriched["attempt"] = attempt
        annotated.append(enriched)
    return annotated


def _priced_cost(item: dict[str, Any]) -> float | None:
    pricing = item.get("pricing") or {}
    amount = pricing.get("amount")
    if pricing.get("status") != "ok" or not isinstance(amount, (int, float)):
        return None
    return float(amount)


def _aggregate(members: list[dict[str, Any]]) -> dict[str, Any]:
    costs = [cost for item in members if (cost := _priced_cost(item)) is not None]
    accepted = [
        item for item in members if item["attempt"]["disposition"] == "accepted"
    ]
    accepted_costs = [
        cost for item in accepted if (cost := _priced_cost(item)) is not None
    ]
    total_cost = sum(costs) if costs else None
    accepted_total = sum(accepted_costs) if accepted_costs else None
    return {
        "attempts": len(members),
        "priced_attempts": len(costs),
        "accepted_attempts": len(accepted),
        "priced_accepted_attempts": len(accepted_costs),
        "total_estimated_cost": total_cost,
        "accepted_estimated_cost": accepted_total,
        "cost_per_accepted_task": (
            sum(accepted_costs) / len(accepted_costs) if accepted_costs else None
        ),
        "mean_estimated_cost": mean(costs) if costs else None,
        "median_estimated_cost": median(costs) if costs else None,
        "unpriced_attempts": len(members) - len(costs),
        "disposition_counts": {
            disposition: sum(
                item["attempt"]["disposition"] == disposition for item in members
            )
            for disposition in sorted(DISPOSITIONS)
        },
    }


def _distribution(values: Iterable[int | float]) -> dict[str, int | float | None]:
    materialized = sorted(values)
    if not materialized:
        return {
            "observations": 0,
            "total": None,
            "mean": None,
            "median": None,
            "p90": None,
        }
    return {
        "observations": len(materialized),
        "total": sum(materialized),
        "mean": mean(materialized),
        "median": median(materialized),
        "p90": materialized[ceil(len(materialized) * 0.9) - 1],
    }


def _numeric(value: Any) -> int | float | None:
    return (
        value
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _wall_seconds(item: dict[str, Any]) -> float | None:
    started = item.get("started_at")
    ended = item.get("ended_at")
    if not isinstance(started, str) or not isinstance(ended, str):
        return None
    try:
        duration = (
            datetime.fromisoformat(ended) - datetime.fromisoformat(started)
        ).total_seconds()
    except (TypeError, ValueError):
        return None
    return duration if duration >= 0 else None


def _usage_distributions(
    members: list[dict[str, Any]], key: str, fields: Iterable[str]
) -> dict[str, dict[str, int | float | None]]:
    output = {}
    for field in fields:
        values = []
        for item in members:
            usage = item.get(key)
            value = _numeric(usage.get(field)) if isinstance(usage, dict) else None
            if value is not None:
                values.append(value)
        output[field] = _distribution(values)
    return output


def _billable_usage(item: dict[str, Any]) -> dict[str, int] | None:
    pricing = item.get("pricing")
    if not isinstance(pricing, dict) or pricing.get("status") != "ok":
        return None
    totals = {
        field: 0
        for field in (
            "uncached_input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
        )
    }
    for segment in pricing.get("segments", []):
        segment_pricing = segment.get("pricing") if isinstance(segment, dict) else None
        billable = (
            segment_pricing.get("billable_tokens")
            if isinstance(segment_pricing, dict)
            else None
        )
        if not isinstance(billable, dict):
            return None
        for field in totals:
            value = billable.get(field)
            if not isinstance(value, int) or isinstance(value, bool):
                return None
            totals[field] += value
    return totals


def _decision_metrics(members: list[dict[str, Any]]) -> dict[str, Any]:
    wall_values = [
        value for item in members if (value := _wall_seconds(item)) is not None
    ]
    billable = [
        usage for item in members if (usage := _billable_usage(item)) is not None
    ]
    return {
        "rollout_wall_seconds": _distribution(wall_values),
        "measured_tokens": _usage_distributions(
            members, "measured_usage", USAGE_FIELDS
        ),
        "billable_tokens": _usage_distributions(
            [{"billable": usage} for usage in billable],
            "billable",
            (
                "uncached_input_tokens",
                "cached_input_tokens",
                "cache_write_input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
            ),
        ),
        "estimated_cost": _distribution(
            cost for item in members if (cost := _priced_cost(item)) is not None
        ),
    }


def summarize_attempts(items: Iterable[dict[str, Any]], policy: str) -> dict[str, Any]:
    materialized = list(items)
    aggregate = _aggregate(materialized)
    aggregate.update(
        {
            "policy": policy,
            "accepted_definition": (
                "explicit lead outcome"
                if policy == "strict"
                else "completed rollout proxy"
                if policy == "completed"
                else "completed rollout with no observed followup_task proxy"
            ),
        }
    )
    return aggregate


def summarize_attempt_routes(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        model = (
            item.get("model") or ",".join(item.get("models") or []) or "unknown-model"
        )
        effort = (
            item.get("effort")
            or ",".join(item.get("efforts") or [])
            or "unknown-effort"
        )
        role = item.get("agent_role") or item.get("task_label") or "unknown-role"
        groups[(model, effort, role)].append(item)
    summaries = []
    for (model, effort, role), members in sorted(groups.items()):
        summaries.append(
            {
                "model": model,
                "effort": effort,
                "agent_role": role,
                **_aggregate(members),
            }
        )
    return summaries


def summarize_route_pairs(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate the model × effort pair used by routing decisions."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        model = (
            item.get("model") or ",".join(item.get("models") or []) or "unknown-model"
        )
        effort = (
            item.get("effort")
            or ",".join(item.get("efforts") or [])
            or "unknown-effort"
        )
        groups[(model, effort)].append(item)
    return [
        {
            "model": model,
            "effort": effort,
            **_aggregate(members),
            **_decision_metrics(members),
        }
        for (model, effort), members in sorted(groups.items())
    ]
