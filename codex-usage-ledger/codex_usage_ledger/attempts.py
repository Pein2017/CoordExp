from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


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
                raise ValueError(f"invalid outcomes JSONL at line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"outcomes line {line_number} must be an object")
            key = value.get("attempt_id") or value.get("thread_id")
            disposition = value.get("disposition")
            if not isinstance(key, str) or not key:
                raise ValueError(f"outcomes line {line_number} needs attempt_id or thread_id")
            if disposition not in DISPOSITIONS - {"unknown"}:
                raise ValueError(
                    f"outcomes line {line_number} has invalid disposition: {disposition!r}"
                )
            outcomes[key] = dict(value)
    return outcomes


def _event_sort_key(event: dict[str, Any]) -> tuple[int, str]:
    occurred = event.get("occurred_at_ms")
    if isinstance(occurred, int):
        return occurred, event.get("timestamp") or ""
    return 0, event.get("timestamp") or ""


def _activity_index(records: Iterable[Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
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
        started = next((event for event in events if event.get("kind") == "started"), None)
        attempt_id = (
            started.get("event_id")
            if started and isinstance(started.get("event_id"), str)
            else f"thread:{thread_key or 'unknown'}"
        )
        followups = [
            event
            for event in events
            if event.get("kind") == "interacted"
            and tool_names.get(event.get("event_id")) == "followup_task"
        ]
        interruptions = [event for event in events if event.get("kind") == "interrupted"]
        label = labels.get(attempt_id) or (labels.get(thread_key) if thread_key else None)
        if label is not None:
            disposition = label.get("disposition")
            disposition_source = "explicit_outcome"
            note = label.get("note")
        else:
            disposition, disposition_source = _proxy_disposition(
                item.get("status") or "unknown", len(followups), policy
            )
            note = None
        attempt = {
            "attempt_id": attempt_id,
            "child_thread_id": thread_id,
            "parent_thread_id": item.get("parent_thread_id"),
            "spawn_event_id": started.get("event_id") if started else None,
            "agent_path": item.get("agent_path"),
            "agent_role": item.get("agent_role"),
            "followup_count": len(followups),
            "interrupted_count": len(interruptions),
            "activity_kinds": sorted({event.get("kind") for event in events if event.get("kind")}),
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
    accepted = [item for item in members if item["attempt"]["disposition"] == "accepted"]
    accepted_costs = [
        cost
        for item in accepted
        if (cost := _priced_cost(item)) is not None
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
            accepted_total / len(accepted_costs) if accepted_costs else None
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


def summarize_attempts(
    items: Iterable[dict[str, Any]], policy: str
) -> dict[str, Any]:
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
        model = item.get("model") or ",".join(item.get("models") or []) or "unknown-model"
        effort = item.get("effort") or ",".join(item.get("efforts") or []) or "unknown-effort"
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
        model = item.get("model") or ",".join(item.get("models") or []) or "unknown-model"
        effort = item.get("effort") or ",".join(item.get("efforts") or []) or "unknown-effort"
        groups[(model, effort)].append(item)
    return [
        {"model": model, "effort": effort, **_aggregate(members)}
        for (model, effort), members in sorted(groups.items())
    ]
