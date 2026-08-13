from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Any, Iterable

from .parser import SessionRecord
from .model import Usage
from .pricing import PriceRate, estimate_cost, lookup_rate


def enrich_record(
    record: SessionRecord, rates: dict[tuple[str, str], PriceRate]
) -> dict[str, Any]:
    output = record.to_dict()
    route_usage = record.route_usage()
    if not route_usage:
        output["pricing"] = {"status": "missing_usage", "amount": None}
        return output

    segments: list[dict[str, Any]] = []
    total_amount = 0.0
    all_priced = True
    for route in route_usage:
        model = route.get("model")
        effort = route.get("effort")
        usage = Usage.from_mapping(route.get("usage"))
        rate = lookup_rate(rates, record.model_provider, model)
        segment: dict[str, Any] = {
            "model": model,
            "effort": effort,
            "turn_ids": route.get("turn_ids", []),
            "usage": usage.to_dict(),
        }
        if rate is None:
            segment["pricing"] = {
                "status": "ambiguous_model" if model is None else "missing_rate",
                "amount": None,
            }
            all_priced = False
        else:
            amount, details = estimate_cost(usage, rate)
            segment["pricing"] = {
                **details,
                "amount": float(amount) if amount is not None else None,
            }
            if amount is None:
                all_priced = False
            else:
                total_amount += float(amount)
        segments.append(segment)

    currencies = {
        segment["pricing"].get("currency")
        for segment in segments
        if segment["pricing"].get("currency")
    }
    output["pricing"] = {
        "status": "ok" if all_priced else "partial_or_missing",
        "amount": total_amount if all_priced else None,
        "currency": next(iter(currencies)) if len(currencies) == 1 else None,
        "segments": segments,
    }
    return output


def _group_key(item: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        item.get("model_provider") or "unknown-provider",
        item.get("model") or ",".join(item.get("models") or []) or "unknown-model",
        item.get("effort") or ",".join(item.get("efforts") or []) or "unknown-effort",
        item.get("agent_role") or "unknown-role",
        item.get("task_label") or "unknown-task",
    )


def summarize(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for item in items:
        groups[_group_key(item)].append(item)

    summaries: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        token_values = [
            item["measured_usage"]["total_tokens"]
            for item in members
            if item.get("measured_usage")
        ]
        cost_values = [
            item["pricing"]["amount"]
            for item in members
            if item.get("pricing", {}).get("status") == "ok"
        ]
        completed = sum(item.get("status") == "completed" for item in members)
        provider, model, effort, role, task = key
        summaries.append(
            {
                "provider": provider,
                "model": model,
                "effort": effort,
                "agent_role": role,
                "task_label": task,
                "sessions": len(members),
                "completed_sessions": completed,
                "usage_sessions": len(token_values),
                "priced_sessions": len(cost_values),
                "total_tokens": sum(token_values),
                "mean_tokens": mean(token_values) if token_values else None,
                "median_tokens": median(token_values) if token_values else None,
                "total_estimated_cost": sum(cost_values) if cost_values else None,
                "mean_estimated_cost": mean(cost_values) if cost_values else None,
                "median_estimated_cost": median(cost_values) if cost_values else None,
            }
        )
    return summaries


def summarize_totals(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return aggregate measurements without hiding partial pricing."""

    materialized = list(items)
    fully_priced = [
        item for item in materialized if item.get("pricing", {}).get("status") == "ok"
    ]
    route_segments = [
        segment
        for item in materialized
        for segment in item.get("pricing", {}).get("segments", [])
    ]
    priced_segments = [
        segment
        for segment in route_segments
        if segment.get("pricing", {}).get("status") == "ok"
    ]
    unpriced_segments = [
        segment
        for segment in route_segments
        if segment.get("pricing", {}).get("status") != "ok"
    ]
    currencies = {
        item.get("pricing", {}).get("currency")
        for item in fully_priced
        if item.get("pricing", {}).get("currency")
    }
    return {
        "sessions": len(materialized),
        "completed_sessions": sum(
            item.get("status") == "completed" for item in materialized
        ),
        "measured_tokens": sum(
            (item.get("measured_usage") or {}).get("total_tokens", 0)
            for item in materialized
        ),
        "fully_priced_sessions": len(fully_priced),
        "fully_priced_cost": (
            sum(item["pricing"]["amount"] for item in fully_priced)
            if fully_priced
            else None
        ),
        "known_route_segments": len(priced_segments),
        "known_route_cost": (
            sum(segment["pricing"]["amount"] for segment in priced_segments)
            if priced_segments
            else None
        ),
        "unpriced_segments": len(unpriced_segments),
        "unpriced_tokens": sum(
            (segment.get("usage") or {}).get("total_tokens", 0)
            for segment in unpriced_segments
        ),
        "currency": next(iter(currencies)) if len(currencies) == 1 else None,
    }
