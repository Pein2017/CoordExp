#!/usr/bin/env python3
"""Fixed-capacity adaptive state for the agent-routing skill."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 4
CONTROLLER_VERSION = "4"
ROLE_ORDER = (
    "read_only_scout",
    "bounded_builder",
    "semantic_builder",
    "lifecycle_builder",
    "semantic_reviewer",
    "lifecycle_reviewer",
    "major_decision",
)
SLOTS_PER_ROLE = 3
QUALITY_HALF_LIFE_DAYS = 30.0
SURFACE_HALF_LIFE_DAYS = 7.0
CORRECTED_CREDIT = 0.60
LCB_PENALTY = 0.50
QUALITY_MARGIN = 0.05
MIN_COMPARABLE_EVENTS = 5
MIN_DISTINCT_ROOTS = 3
EVIDENCE_RING_SIZE = 8
RECEIPT_RING_SIZE = 64
PENDING_CAPACITY = 32
MAX_TEXT = 256
COMPLETED_FILTER_BITS = 262_144
COMPLETED_FILTER_HASHES = 4
COMPLETED_FILTER_HEX_LENGTH = COMPLETED_FILTER_BITS // 4
COMPLETED_FILTER_MAX_INSERTIONS = 10_000
COMPARISON_RUN_EPOCH_MAX = 2_147_483_647
KNOWN_TEMPLATE_SENTINELS = {
    "root-id",
    "episode-id",
    "root-id:episode-id:plan-v4",
    "root-id:episode-id:v4",
    "session-or-ledger-evidence-handle",
}

OBSERVED_OUTCOMES = (
    "accepted_first_pass",
    "accepted_after_correction",
    "rejected_or_escalated",
    "false_accept",
    "indeterminate",
)
SURFACE_STATUSES = (
    "ok",
    "auth_client_failure",
    "runtime_failure",
    "target_invalid",
    "receipt_invalid",
)
EVIDENCE_GRADES = ("ordinary", "comparable")
TARGET_OUTCOMES = ("accepted", "rejected", "not_reached", "not_applicable")
REVIEW_TARGET_VERDICTS = ("pass", "hold", "invalidated", "not_applicable")
REVIEW_ROLES = frozenset(("semantic_reviewer", "lifecycle_reviewer"))

STATS_KEYS = (
    "observed_n",
    "observed_first_pass",
    "observed_corrected",
    "observed_rejected",
    "observed_false_accept",
    "observed_elapsed_seconds_sum",
    "observed_lead_seconds_sum",
    "observed_cost_usd_sum",
    "observed_cost_n",
    "comparable_n",
    "comparable_first_pass",
    "comparable_corrected",
    "comparable_rejected",
    "comparable_false_accept",
    "comparable_elapsed_seconds_sum",
    "comparable_lead_seconds_sum",
    "comparable_cost_usd_sum",
    "comparable_cost_n",
    "surface_attempt_n",
    "surface_auth_client_failure",
    "surface_runtime_failure",
    "surface_target_invalid",
    "surface_receipt_invalid",
    "surface_failure_seconds_sum",
)
SURFACE_STATS_KEYS = tuple(key for key in STATS_KEYS if key.startswith("surface_"))
QUALITY_STATS_KEYS = tuple(key for key in STATS_KEYS if key not in SURFACE_STATS_KEYS)

TOP_KEYS = {
    "schema_version",
    "controller_version",
    "policy_epoch",
    "updated_at",
    "roles",
    "pending",
    "receipt_ring",
    "completed_episode_filter",
    "accepted_route_filter",
    "corrected_accept_filter",
}
COMPLETED_FILTER_KEYS = {"bits", "hashes", "insertions", "version"}
ROLE_KEYS = {
    "baseline_slot",
    "active_comparison_id",
    "active_comparison_contract_hash",
    "active_route_cohort_hash",
    "active_comparison_signature",
    "active_comparison_run_epoch",
    "slots",
}
SLOT_KEYS = {
    "slot",
    "active",
    "quarantined",
    "quarantine_reason",
    "route_change_reason",
    "route_changed_at",
    "route_epoch",
    "route_generation",
    "route",
    "resolved_model_id",
    "comparison_id",
    "route_cohort_hash",
    "comparison_run_epoch",
    "comparison_contract_hash",
    "comparison_event_at",
    "comparison_generation",
    "stats",
    "last_quality_decay_at",
    "last_surface_decay_at",
    "evidence_ring",
}
ROUTE_KEYS = {"surface", "model", "effort"}
EVIDENCE_KEYS = {
    "receipt_id",
    "root_id",
    "episode_id",
    "observed_at",
    "evidence_grade",
    "delivery_outcome",
    "surface_status",
    "comparison_id",
    "route_cohort_hash",
    "comparison_run_epoch",
    "task_shape_hash",
    "brief_hash",
    "verifier_hash",
    "target_hash",
    "route_epoch",
    "route_generation",
    "resolved_model_id",
    "eligible_routes",
    "correction_budget",
    "terminal_budget_seconds",
    "comparison_generation",
}
RECEIPT_KEYS = {
    "receipt_version",
    "plan_id",
    "receipt_id",
    "root_id",
    "episode_id",
    "role",
    "eligible_routes",
    "task_shape_hash",
    "brief_hash",
    "verifier_hash",
    "target_hash",
    "controller_version",
    "policy_epoch",
    "route_epoch",
    "route_generation",
    "route",
    "resolved_model_id",
    "evidence_grade",
    "comparison_id",
    "route_cohort_hash",
    "comparison_run_epoch",
    "delivery_outcome",
    "target_outcome",
    "review_target_verdict",
    "surface_status",
    "elapsed_seconds",
    "lead_seconds",
    "cost_usd",
    "correction_count",
    "correction_budget",
    "terminal_budget_seconds",
    "acceptance_authority",
    "observed_at",
    "evidence_id",
    "supersedes_receipt_id",
}
PLAN_KEYS = {
    "plan_version",
    "plan_id",
    "root_id",
    "episode_id",
    "role",
    "eligible_routes",
    "task_shape_hash",
    "brief_hash",
    "verifier_hash",
    "target_hash",
    "controller_version",
    "policy_epoch",
    "route_epoch",
    "route_generation",
    "route",
    "resolved_model_id",
    "evidence_grade",
    "comparison_id",
    "route_cohort_hash",
    "comparison_run_epoch",
    "correction_budget",
    "terminal_budget_seconds",
    "planned_at",
    "terminal_due_at",
    "issuer_authority",
}


class StateError(ValueError):
    pass


class UncertainCommitError(StateError):
    """The replacement is visible, but crash-durability could not be confirmed."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise StateError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise StateError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise StateError(f"{label} keys mismatch; missing={missing}, extra={extra}")


def require_text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise StateError(f"{field} must be a non-empty string")
    if len(value) > MAX_TEXT:
        raise StateError(f"{field} exceeds {MAX_TEXT} characters")
    if "\0" in value:
        raise StateError(f"{field} must not contain a NUL identity delimiter")
    return value


def require_concrete_text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    text = require_text(value, field, allow_empty=allow_empty)
    if text and (text.startswith("REPLACE_") or text in KNOWN_TEMPLATE_SENTINELS):
        raise StateError(f"{field} must replace the template sentinel")
    return text


def require_sha256(value: Any, field: str) -> str:
    text = require_text(value, field)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise StateError(f"{field} must be a lowercase SHA-256 hex digest")
    if text == "0" * 64:
        raise StateError(f"{field} must not use the all-zero template sentinel")
    return text


def require_resolved_model_id(value: Any, field: str) -> str:
    text = require_text(value, field)
    if text.startswith("REPLACE_") or text == "exact-live-resolved-model-id":
        raise StateError(f"{field} must be a verified concrete model identity")
    return text


def require_route_component(
    value: Any, field: str, *, allow_empty: bool = False
) -> str:
    text = require_text(value, field, allow_empty=allow_empty)
    if ":" in text:
        raise StateError(f"{field} must not contain the route-key ':' delimiter")
    return text


def require_number(value: Any, field: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StateError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise StateError(f"{field} must be finite and non-negative")
    return result


def comparison_contract_hash(value: dict[str, Any]) -> str:
    payload = {
        field: value[field]
        for field in ("task_shape_hash", "brief_hash", "verifier_hash", "target_hash")
    }
    payload["eligible_routes"] = sorted(value["eligible_routes"])
    payload["correction_budget"] = value["correction_budget"]
    payload["terminal_budget_seconds"] = value["terminal_budget_seconds"]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def route_cohort_hash(
    state: dict[str, Any], role: str, eligible_routes: list[str] | set[str]
) -> str:
    """Fingerprint the fixed route generations that form one matched cohort."""
    if role not in ROLE_ORDER:
        raise StateError("comparison role is not a fixed controller role")
    if not eligible_routes:
        raise StateError("comparison epoch requires at least one eligible route")
    cohort = []
    for key in sorted(eligible_routes):
        route = parse_route_key(key)
        slot = find_slot(state, role, route)
        cohort.append(
            {
                "route": key,
                "route_epoch": slot["route_epoch"],
                "route_generation": slot["route_generation"],
            }
        )
    payload = {
        "controller_version": state["controller_version"],
        "policy_epoch": state["policy_epoch"],
        "role": role,
        "cohort": cohort,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def comparison_signature_from_parts(
    comparison_id: str, contract_hash: str, cohort_hash: str
) -> str:
    payload = {
        "comparison_id": comparison_id,
        "comparison_contract_hash": contract_hash,
        "route_cohort_hash": cohort_hash,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def comparison_signature(value: dict[str, Any]) -> str:
    return comparison_signature_from_parts(
        value["comparison_id"],
        comparison_contract_hash(value),
        value["route_cohort_hash"],
    )


def empty_completed_filter() -> dict[str, Any]:
    return {
        "version": 1,
        "hashes": COMPLETED_FILTER_HASHES,
        "insertions": 0,
        "bits": "0" * COMPLETED_FILTER_HEX_LENGTH,
    }


def validate_completed_filter(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise StateError(f"{label} must be an object")
    require_exact_keys(value, COMPLETED_FILTER_KEYS, label)
    if value["version"] != 1:
        raise StateError(f"unsupported {label} version")
    if value["hashes"] != COMPLETED_FILTER_HASHES:
        raise StateError(f"{label} hash count mismatch")
    if (
        not isinstance(value["insertions"], int)
        or isinstance(value["insertions"], bool)
        or value["insertions"] < 0
    ):
        raise StateError(f"{label} insertions must be a non-negative integer")
    if value["insertions"] > COMPLETED_FILTER_MAX_INSERTIONS:
        raise StateError(f"{label} exceeds its reviewed fixed-capacity ceiling")
    bits = value["bits"]
    if (
        not isinstance(bits, str)
        or len(bits) != COMPLETED_FILTER_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in bits)
    ):
        raise StateError(f"{label} bits have the wrong fixed shape")


def completed_episode_indexes(root_id: str, episode_id: str) -> tuple[int, ...]:
    payload = (root_id + "\0" + episode_id).encode("utf-8")
    digest = hashlib.sha256(b"agent-routing-completed-v1\0" + payload).digest()
    return tuple(
        int.from_bytes(digest[offset : offset + 4], "big") % COMPLETED_FILTER_BITS
        for offset in range(0, COMPLETED_FILTER_HASHES * 4, 4)
    )


def completed_episode_seen(
    state: dict[str, Any], root_id: str, episode_id: str
) -> bool:
    bits = bytes.fromhex(state["completed_episode_filter"]["bits"])
    return all(
        bits[index // 8] & (1 << (index % 8))
        for index in completed_episode_indexes(root_id, episode_id)
    )


def accepted_route_indexes(value: dict[str, Any]) -> tuple[int, ...]:
    components = (
        value["root_id"],
        value["episode_id"],
        value["role"],
        route_key(value["route"]),
        value["route_epoch"],
        str(value["route_generation"]),
        value["resolved_model_id"],
    )
    payload = "\0".join(components).encode("utf-8")
    digest = hashlib.sha256(b"agent-routing-accepted-route-v2\0" + payload).digest()
    return tuple(
        int.from_bytes(digest[offset : offset + 4], "big") % COMPLETED_FILTER_BITS
        for offset in range(0, COMPLETED_FILTER_HASHES * 4, 4)
    )


def accepted_route_seen(state: dict[str, Any], value: dict[str, Any]) -> bool:
    bits = bytes.fromhex(state["accepted_route_filter"]["bits"])
    return all(
        bits[index // 8] & (1 << (index % 8)) for index in accepted_route_indexes(value)
    )


def corrected_accept_seen(state: dict[str, Any], value: dict[str, Any]) -> bool:
    bits = bytes.fromhex(state["corrected_accept_filter"]["bits"])
    return all(
        bits[index // 8] & (1 << (index % 8)) for index in accepted_route_indexes(value)
    )


def mark_filter(filter_state: dict[str, Any], indexes: tuple[int, ...]) -> None:
    if filter_state["insertions"] >= COMPLETED_FILTER_MAX_INSERTIONS:
        raise StateError(
            "fixed replay filter capacity is exhausted; adaptation must abstain"
        )
    bits = bytearray.fromhex(filter_state["bits"])
    for index in indexes:
        bits[index // 8] |= 1 << (index % 8)
    filter_state["bits"] = bits.hex()
    filter_state["insertions"] += 1


def mark_terminal_episode(state: dict[str, Any], receipt: dict[str, Any]) -> None:
    mark_filter(
        state["completed_episode_filter"],
        completed_episode_indexes(receipt["root_id"], receipt["episode_id"]),
    )
    if receipt["delivery_outcome"] in (
        "accepted_first_pass",
        "accepted_after_correction",
    ):
        mark_filter(state["accepted_route_filter"], accepted_route_indexes(receipt))


def mark_corrected_accept(state: dict[str, Any], receipt: dict[str, Any]) -> None:
    mark_filter(state["corrected_accept_filter"], accepted_route_indexes(receipt))


def route_key(route: dict[str, str]) -> str:
    return f"{route['surface']}:{route['model']}:{route['effort']}"


def parse_route_key(value: str) -> dict[str, str]:
    parts = value.split(":")
    if len(parts) != 3 or any(not part for part in parts):
        raise StateError("route must be surface:model:effort")
    return {"surface": parts[0], "model": parts[1], "effort": parts[2]}


def zero_stats() -> dict[str, float]:
    return {key: 0.0 for key in STATS_KEYS}


def seed_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "references"
        / "routing-state-seed.json"
    )


def default_state_path() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    base = Path(codex_home) if codex_home else Path(__file__).resolve().parents[3]
    return base / "state" / "agent-routing.json"


def parse_json_text(text: str, source: str) -> dict[str, Any]:
    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StateError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise StateError(f"invalid JSON numeric constant: {value}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise StateError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise StateError(f"{source} must contain a JSON object")
    return value


def load_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise StateError(f"missing file: {path}") from exc
    return parse_json_text(text, str(path))


def load_stdin_json() -> dict[str, Any]:
    return parse_json_text(sys.stdin.read(), "stdin")


def build_state(seed: dict[str, Any], now: datetime) -> dict[str, Any]:
    require_exact_keys(seed, {"seed_version", "policy_epoch", "roles"}, "seed")
    if seed["seed_version"] != 1:
        raise StateError("unsupported seed_version")
    require_text(seed["policy_epoch"], "seed.policy_epoch")
    if not isinstance(seed["roles"], dict) or set(seed["roles"]) != set(ROLE_ORDER):
        raise StateError("seed roles must match the fixed role set")

    roles: dict[str, Any] = {}
    for role_name in ROLE_ORDER:
        role_seed = seed["roles"][role_name]
        require_exact_keys(
            role_seed, {"baseline_slot", "routes"}, f"seed.roles.{role_name}"
        )
        baseline = role_seed["baseline_slot"]
        routes = role_seed["routes"]
        if (
            not isinstance(baseline, int)
            or isinstance(baseline, bool)
            or not 0 <= baseline < SLOTS_PER_ROLE
        ):
            raise StateError(f"invalid baseline slot for {role_name}")
        if not isinstance(routes, list) or len(routes) != SLOTS_PER_ROLE:
            raise StateError(f"{role_name} must define exactly {SLOTS_PER_ROLE} routes")
        slots = []
        for index, route_seed in enumerate(routes):
            require_exact_keys(
                route_seed,
                {"active", "route_epoch", "surface", "model", "effort"},
                f"seed route {role_name}[{index}]",
            )
            if not isinstance(route_seed["active"], bool):
                raise StateError("seed route active must be boolean")
            require_text(
                route_seed["route_epoch"],
                "seed route route_epoch",
                allow_empty=not route_seed["active"],
            )
            for key in ("surface", "model", "effort"):
                require_route_component(
                    route_seed[key],
                    f"seed route {key}",
                    allow_empty=not route_seed["active"],
                )
            if route_seed["active"] and not all(
                route_seed[key] for key in ("route_epoch", "surface", "model", "effort")
            ):
                raise StateError("active seed routes require complete identity")
            slots.append(
                {
                    "slot": index,
                    "active": route_seed["active"],
                    "quarantined": False,
                    "quarantine_reason": "",
                    "route_change_reason": "seed",
                    "route_changed_at": isoformat(now),
                    "route_epoch": route_seed["route_epoch"],
                    "route_generation": 1,
                    "route": {
                        "surface": route_seed["surface"],
                        "model": route_seed["model"],
                        "effort": route_seed["effort"],
                    },
                    "resolved_model_id": "",
                    "comparison_id": "",
                    "route_cohort_hash": "",
                    "comparison_run_epoch": 0,
                    "comparison_contract_hash": "",
                    "comparison_event_at": "",
                    "comparison_generation": 0,
                    "stats": zero_stats(),
                    "last_quality_decay_at": isoformat(now),
                    "last_surface_decay_at": isoformat(now),
                    "evidence_ring": [],
                }
            )
        if not slots[baseline]["active"]:
            raise StateError(f"baseline slot for {role_name} must be active")
        roles[role_name] = {
            "baseline_slot": baseline,
            "active_comparison_id": "",
            "active_comparison_contract_hash": "",
            "active_route_cohort_hash": "",
            "active_comparison_signature": "",
            "active_comparison_run_epoch": 0,
            "slots": slots,
        }

    state = {
        "schema_version": SCHEMA_VERSION,
        "controller_version": CONTROLLER_VERSION,
        "policy_epoch": seed["policy_epoch"],
        "updated_at": isoformat(now),
        "roles": roles,
        "pending": [],
        "receipt_ring": [],
        "completed_episode_filter": empty_completed_filter(),
        "accepted_route_filter": empty_completed_filter(),
        "corrected_accept_filter": empty_completed_filter(),
    }
    validate_state(state)
    return state


def validate_state(state: dict[str, Any]) -> None:
    require_exact_keys(state, TOP_KEYS, "state")
    if state["schema_version"] != SCHEMA_VERSION:
        raise StateError("unsupported schema_version")
    if state["controller_version"] != CONTROLLER_VERSION:
        raise StateError("controller_version mismatch")
    require_text(state["policy_epoch"], "policy_epoch")
    policy_seed = load_json(seed_path())
    require_exact_keys(
        policy_seed,
        {"seed_version", "policy_epoch", "roles"},
        "policy seed",
    )
    if policy_seed["seed_version"] != 1:
        raise StateError("unsupported policy seed_version")
    if state["policy_epoch"] != policy_seed["policy_epoch"]:
        raise StateError("state policy epoch does not match the bundled policy seed")
    if not isinstance(policy_seed["roles"], dict) or set(policy_seed["roles"]) != set(
        ROLE_ORDER
    ):
        raise StateError("policy seed roles must match the fixed role set")
    parse_time(require_text(state["updated_at"], "updated_at"), "updated_at")
    if not isinstance(state["roles"], dict) or set(state["roles"]) != set(ROLE_ORDER):
        raise StateError("state roles must match the fixed role set")
    if (
        not isinstance(state["pending"], list)
        or len(state["pending"]) > PENDING_CAPACITY
    ):
        raise StateError("pending table exceeds fixed capacity")
    if (
        not isinstance(state["receipt_ring"], list)
        or len(state["receipt_ring"]) > RECEIPT_RING_SIZE
    ):
        raise StateError("receipt_ring exceeds fixed capacity")
    if len(set(state["receipt_ring"])) != len(state["receipt_ring"]):
        raise StateError("receipt_ring contains duplicates")
    for index, receipt_id in enumerate(state["receipt_ring"]):
        require_text(receipt_id, f"receipt_ring[{index}]")
    validate_completed_filter(
        state["completed_episode_filter"], "completed_episode_filter"
    )
    validate_completed_filter(state["accepted_route_filter"], "accepted_route_filter")
    validate_completed_filter(
        state["corrected_accept_filter"], "corrected_accept_filter"
    )

    for role_name in ROLE_ORDER:
        role = state["roles"][role_name]
        if not isinstance(role, dict):
            raise StateError(f"role {role_name} must be an object")
        require_exact_keys(role, ROLE_KEYS, f"role {role_name}")
        baseline = role["baseline_slot"]
        if (
            not isinstance(baseline, int)
            or isinstance(baseline, bool)
            or not 0 <= baseline < SLOTS_PER_ROLE
        ):
            raise StateError(f"invalid baseline slot for {role_name}")
        policy_role = policy_seed["roles"][role_name]
        require_exact_keys(
            policy_role,
            {"baseline_slot", "routes"},
            f"policy seed role {role_name}",
        )
        if baseline != policy_role["baseline_slot"]:
            raise StateError(f"policy baseline slot mismatch for role {role_name}")
        if (
            not isinstance(policy_role["routes"], list)
            or len(policy_role["routes"]) != SLOTS_PER_ROLE
        ):
            raise StateError(
                f"policy seed role {role_name} must define exactly "
                f"{SLOTS_PER_ROLE} routes"
            )
        require_text(
            role["active_comparison_id"],
            "active_comparison_id",
            allow_empty=True,
        )
        require_text(
            role["active_comparison_contract_hash"],
            "active_comparison_contract_hash",
            allow_empty=True,
        )
        require_text(
            role["active_route_cohort_hash"],
            "active_route_cohort_hash",
            allow_empty=True,
        )
        require_text(
            role["active_comparison_signature"],
            "active_comparison_signature",
            allow_empty=True,
        )
        if role["active_comparison_contract_hash"]:
            require_sha256(
                role["active_comparison_contract_hash"],
                "active_comparison_contract_hash",
            )
        if role["active_route_cohort_hash"]:
            require_sha256(
                role["active_route_cohort_hash"],
                "active_route_cohort_hash",
            )
        if role["active_comparison_signature"]:
            require_sha256(
                role["active_comparison_signature"],
                "active_comparison_signature",
            )
        if (
            not isinstance(role["active_comparison_run_epoch"], int)
            or isinstance(role["active_comparison_run_epoch"], bool)
            or not 0 <= role["active_comparison_run_epoch"] <= COMPARISON_RUN_EPOCH_MAX
        ):
            raise StateError("active_comparison_run_epoch is outside fixed bounds")
        active_comparison_fields = (
            role["active_comparison_id"],
            role["active_comparison_contract_hash"],
            role["active_route_cohort_hash"],
            role["active_comparison_signature"],
            role["active_comparison_run_epoch"],
        )
        if len({bool(value) for value in active_comparison_fields}) != 1:
            raise StateError(
                "active comparison identity, contract, cohort, signature, and run epoch "
                "must be set together"
            )
        if role["active_comparison_signature"] != (
            comparison_signature_from_parts(
                role["active_comparison_id"],
                role["active_comparison_contract_hash"],
                role["active_route_cohort_hash"],
            )
            if role["active_comparison_id"]
            else ""
        ):
            raise StateError("active comparison signature does not match its fields")
        slots = role["slots"]
        if not isinstance(slots, list) or len(slots) != SLOTS_PER_ROLE:
            raise StateError(f"{role_name} must contain exactly {SLOTS_PER_ROLE} slots")
        for index, slot in enumerate(slots):
            if not isinstance(slot, dict):
                raise StateError(f"slot {role_name}[{index}] must be an object")
            require_exact_keys(slot, SLOT_KEYS, f"slot {role_name}[{index}]")
            if slot["slot"] != index:
                raise StateError(f"slot index mismatch for {role_name}[{index}]")
            if not isinstance(slot["active"], bool) or not isinstance(
                slot["quarantined"], bool
            ):
                raise StateError("active and quarantined must be booleans")
            require_text(
                slot["quarantine_reason"], "quarantine_reason", allow_empty=True
            )
            require_text(slot["route_change_reason"], "route_change_reason")
            parse_time(
                require_text(slot["route_changed_at"], "route_changed_at"),
                "route_changed_at",
            )
            require_text(
                slot["route_epoch"], "route_epoch", allow_empty=not slot["active"]
            )
            if (
                not isinstance(slot["route_generation"], int)
                or isinstance(slot["route_generation"], bool)
                or slot["route_generation"] <= 0
            ):
                raise StateError("route_generation must be a positive integer")
            if not isinstance(slot["route"], dict):
                raise StateError("route must be an object")
            require_exact_keys(slot["route"], ROUTE_KEYS, "route")
            for field in ROUTE_KEYS:
                require_route_component(
                    slot["route"][field],
                    f"route.{field}",
                    allow_empty=not slot["active"],
                )
            if slot["active"] and not all(slot["route"].values()):
                raise StateError("active routes require complete identity")
            if index == baseline:
                policy_route = policy_role["routes"][baseline]
                require_exact_keys(
                    policy_route,
                    {"active", "route_epoch", "surface", "model", "effort"},
                    f"policy baseline route {role_name}",
                )
                expected_route = {field: policy_route[field] for field in ROUTE_KEYS}
                if slot["route"] != expected_route:
                    raise StateError(
                        f"policy baseline route mismatch for role {role_name}"
                    )
            require_text(
                slot["resolved_model_id"], "resolved_model_id", allow_empty=True
            )
            if slot["resolved_model_id"]:
                require_resolved_model_id(
                    slot["resolved_model_id"], "resolved_model_id"
                )
            require_text(slot["comparison_id"], "comparison_id", allow_empty=True)
            require_text(
                slot["route_cohort_hash"], "route_cohort_hash", allow_empty=True
            )
            if (
                not isinstance(slot["comparison_run_epoch"], int)
                or isinstance(slot["comparison_run_epoch"], bool)
                or not 0 <= slot["comparison_run_epoch"] <= COMPARISON_RUN_EPOCH_MAX
            ):
                raise StateError("comparison_run_epoch is outside fixed bounds")
            if slot["comparison_run_epoch"] > role["active_comparison_run_epoch"]:
                raise StateError("slot comparison run cannot be newer than its role")
            require_text(
                slot["comparison_contract_hash"],
                "comparison_contract_hash",
                allow_empty=True,
            )
            require_text(
                slot["comparison_event_at"], "comparison_event_at", allow_empty=True
            )
            if slot["comparison_contract_hash"]:
                require_sha256(
                    slot["comparison_contract_hash"], "comparison_contract_hash"
                )
            if slot["route_cohort_hash"]:
                require_sha256(slot["route_cohort_hash"], "route_cohort_hash")
            if not (
                bool(slot["comparison_id"])
                == bool(slot["route_cohort_hash"])
                == bool(slot["comparison_run_epoch"])
                == bool(slot["comparison_contract_hash"])
                == bool(slot["comparison_event_at"])
            ):
                raise StateError(
                    "comparison id, cohort, run epoch, contract, and start time must be set together"
                )
            if (
                slot["comparison_run_epoch"] == role["active_comparison_run_epoch"]
                and slot["comparison_run_epoch"] > 0
                and comparison_signature_from_parts(
                    slot["comparison_id"],
                    slot["comparison_contract_hash"],
                    slot["route_cohort_hash"],
                )
                != role["active_comparison_signature"]
            ):
                raise StateError(
                    "current slot projection does not match the active comparison run"
                )
            if slot["comparison_event_at"]:
                parse_time(slot["comparison_event_at"], "comparison_event_at")
            if (
                not isinstance(slot["comparison_generation"], int)
                or isinstance(slot["comparison_generation"], bool)
                or slot["comparison_generation"] < 0
            ):
                raise StateError("comparison_generation must be a non-negative integer")
            if slot["comparison_id"] and slot["comparison_generation"] == 0:
                raise StateError(
                    "active comparison projection requires a positive generation"
                )
            if not isinstance(slot["stats"], dict) or set(slot["stats"]) != set(
                STATS_KEYS
            ):
                raise StateError("stats keys do not match fixed schema")
            for key in STATS_KEYS:
                require_number(slot["stats"][key], f"stats.{key}")
            if (
                sum(
                    slot["stats"][key]
                    for key in (
                        "observed_first_pass",
                        "observed_corrected",
                        "observed_rejected",
                        "observed_false_accept",
                    )
                )
                > slot["stats"]["observed_n"] + 1e-8
            ):
                raise StateError("observed outcome mass exceeds observed_n")
            if (
                sum(
                    slot["stats"][key]
                    for key in (
                        "comparable_first_pass",
                        "comparable_corrected",
                        "comparable_rejected",
                        "comparable_false_accept",
                    )
                )
                > slot["stats"]["comparable_n"] + 1e-8
            ):
                raise StateError("comparable outcome mass exceeds comparable_n")
            if slot["stats"]["comparable_n"] > 1e-8 and not slot["comparison_id"]:
                raise StateError(
                    "comparable statistics require one active comparison_id"
                )
            if (
                sum(
                    slot["stats"][key]
                    for key in (
                        "surface_auth_client_failure",
                        "surface_runtime_failure",
                        "surface_target_invalid",
                        "surface_receipt_invalid",
                    )
                )
                > slot["stats"]["surface_attempt_n"] + 1e-8
            ):
                raise StateError("surface failure mass exceeds surface_attempt_n")
            parse_time(
                require_text(slot["last_quality_decay_at"], "last_quality_decay_at"),
                "last_quality_decay_at",
            )
            parse_time(
                require_text(slot["last_surface_decay_at"], "last_surface_decay_at"),
                "last_surface_decay_at",
            )
            if (
                not isinstance(slot["evidence_ring"], list)
                or len(slot["evidence_ring"]) > EVIDENCE_RING_SIZE
            ):
                raise StateError("evidence_ring exceeds fixed capacity")
            evidence_receipt_ids: set[str] = set()
            for evidence in slot["evidence_ring"]:
                if not isinstance(evidence, dict):
                    raise StateError("evidence_ring entries must be objects")
                require_exact_keys(evidence, EVIDENCE_KEYS, "evidence")
                for field in (
                    "receipt_id",
                    "root_id",
                    "episode_id",
                    "observed_at",
                    "evidence_grade",
                    "delivery_outcome",
                    "surface_status",
                    "comparison_id",
                    "route_cohort_hash",
                    "route_epoch",
                    "resolved_model_id",
                ):
                    require_text(
                        evidence[field],
                        f"evidence.{field}",
                        allow_empty=field in ("comparison_id", "route_cohort_hash"),
                    )
                if evidence["receipt_id"] in evidence_receipt_ids:
                    raise StateError("evidence_ring contains duplicate receipt_id")
                evidence_receipt_ids.add(evidence["receipt_id"])
                parse_time(evidence["observed_at"], "evidence.observed_at")
                require_resolved_model_id(
                    evidence["resolved_model_id"], "evidence.resolved_model_id"
                )
                for field in (
                    "task_shape_hash",
                    "brief_hash",
                    "verifier_hash",
                    "target_hash",
                ):
                    require_sha256(evidence[field], f"evidence.{field}")
                if evidence["evidence_grade"] not in EVIDENCE_GRADES:
                    raise StateError("invalid evidence grade in ring")
                if evidence["delivery_outcome"] not in OBSERVED_OUTCOMES:
                    raise StateError("invalid outcome in ring")
                if evidence["surface_status"] not in SURFACE_STATUSES:
                    raise StateError("invalid surface status in ring")
                if evidence["evidence_grade"] == "comparable":
                    if (
                        not evidence["comparison_id"]
                        or not evidence["route_cohort_hash"]
                        or evidence["comparison_run_epoch"] <= 0
                    ):
                        raise StateError(
                            "comparable evidence requires id, cohort, and run epoch"
                        )
                elif (
                    evidence["comparison_id"]
                    or evidence["route_cohort_hash"]
                    or evidence["comparison_run_epoch"] != 0
                ):
                    raise StateError(
                        "ordinary evidence cannot carry comparison metadata"
                    )
                if evidence["route_cohort_hash"]:
                    require_sha256(
                        evidence["route_cohort_hash"], "evidence.route_cohort_hash"
                    )
                if (
                    not isinstance(evidence["comparison_run_epoch"], int)
                    or isinstance(evidence["comparison_run_epoch"], bool)
                    or not 0
                    <= evidence["comparison_run_epoch"]
                    <= COMPARISON_RUN_EPOCH_MAX
                ):
                    raise StateError(
                        "evidence comparison_run_epoch is outside fixed bounds"
                    )
                if (
                    evidence["comparison_run_epoch"]
                    > role["active_comparison_run_epoch"]
                ):
                    raise StateError(
                        "evidence comparison run cannot be newer than its role"
                    )
                if (
                    not isinstance(evidence["route_generation"], int)
                    or isinstance(evidence["route_generation"], bool)
                    or evidence["route_generation"] <= 0
                ):
                    raise StateError("evidence route_generation must be positive")
                if (
                    not isinstance(evidence["eligible_routes"], list)
                    or not evidence["eligible_routes"]
                    or len(evidence["eligible_routes"]) > SLOTS_PER_ROLE
                    or len(set(evidence["eligible_routes"]))
                    != len(evidence["eligible_routes"])
                ):
                    raise StateError(
                        "evidence eligible_routes have the wrong fixed shape"
                    )
                for route in evidence["eligible_routes"]:
                    require_text(route, "evidence.eligible_routes[]")
                    parse_route_key(route)
                if (
                    not isinstance(evidence["correction_budget"], int)
                    or isinstance(evidence["correction_budget"], bool)
                    or not 0 <= evidence["correction_budget"] <= 1
                ):
                    raise StateError("evidence correction_budget must be zero or one")
                if (
                    not isinstance(evidence["terminal_budget_seconds"], int)
                    or isinstance(evidence["terminal_budget_seconds"], bool)
                    or evidence["terminal_budget_seconds"] <= 0
                ):
                    raise StateError(
                        "evidence terminal_budget_seconds must be positive"
                    )
                if (
                    not isinstance(evidence["comparison_generation"], int)
                    or isinstance(evidence["comparison_generation"], bool)
                    or evidence["comparison_generation"] < 0
                ):
                    raise StateError(
                        "evidence comparison_generation must be non-negative"
                    )
                if (
                    evidence["comparison_run_epoch"]
                    == role["active_comparison_run_epoch"]
                    and evidence["comparison_run_epoch"] > 0
                    and comparison_signature_from_parts(
                        evidence["comparison_id"],
                        comparison_contract_hash(evidence),
                        evidence["route_cohort_hash"],
                    )
                    != role["active_comparison_signature"]
                ):
                    raise StateError(
                        "current evidence does not match the active comparison run"
                    )
        active_route_keys = [
            route_key(slot["route"]) for slot in slots if slot["active"]
        ]
        if len(active_route_keys) != len(set(active_route_keys)):
            raise StateError(f"role {role_name} contains duplicate active route keys")
        if not slots[baseline]["active"]:
            raise StateError(f"baseline slot for {role_name} must stay active")

    pending_ids: set[str] = set()
    pending_episodes: set[tuple[str, str]] = set()
    for index, plan in enumerate(state["pending"]):
        if not isinstance(plan, dict):
            raise StateError(f"pending[{index}] must be an object")
        validate_plan(plan, state, for_reserve=False)
        if plan["plan_id"] in pending_ids:
            raise StateError("pending table contains duplicate plan_id")
        episode_key = (plan["root_id"], plan["episode_id"])
        if episode_key in pending_episodes:
            raise StateError("pending table contains duplicate root/episode")
        pending_ids.add(plan["plan_id"])
        pending_episodes.add(episode_key)


def learned_parameter_count() -> int:
    return len(ROLE_ORDER) * SLOTS_PER_ROLE * len(STATS_KEYS)


def state_digest(state: dict[str, Any]) -> str:
    payload = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def state_lock(path: Path, *, exclusive: bool) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_write(path: Path, state: dict[str, Any]) -> None:
    validate_state(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            raise UncertainCommitError(
                "state replacement is visible but directory durability check failed; "
                f"do not retry blindly; validate state digest {state_digest(state)}"
            ) from exc
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def decay_factor(last: datetime, now: datetime, half_life_days: float) -> float:
    seconds = (now - last).total_seconds()
    if seconds < -1e-6:
        raise StateError("out-of-order receipt; replay receipts in event-time order")
    return 0.5 ** (max(0.0, seconds) / (half_life_days * 86400.0))


def decayed_stats(slot: dict[str, Any], now: datetime) -> dict[str, float]:
    quality_last = parse_time(slot["last_quality_decay_at"], "last_quality_decay_at")
    surface_last = parse_time(slot["last_surface_decay_at"], "last_surface_decay_at")
    quality_factor = decay_factor(quality_last, now, QUALITY_HALF_LIFE_DAYS)
    surface_factor = decay_factor(surface_last, now, SURFACE_HALF_LIFE_DAYS)
    return {
        key: float(slot["stats"][key])
        * (quality_factor if key in QUALITY_STATS_KEYS else surface_factor)
        for key in STATS_KEYS
    }


def apply_quality_decay(slot: dict[str, Any], now: datetime) -> float:
    last = parse_time(slot["last_quality_decay_at"], "last_quality_decay_at")
    if now < last:
        return 0.5 ** (
            (now - last).total_seconds() / (QUALITY_HALF_LIFE_DAYS * 86400.0) * -1.0
        )
    factor = decay_factor(last, now, QUALITY_HALF_LIFE_DAYS)
    for key in QUALITY_STATS_KEYS:
        slot["stats"][key] = float(slot["stats"][key]) * factor
    slot["last_quality_decay_at"] = isoformat(now)
    return 1.0


def apply_surface_decay(slot: dict[str, Any], now: datetime) -> float:
    last = parse_time(slot["last_surface_decay_at"], "last_surface_decay_at")
    if now < last:
        return 0.5 ** (
            (now - last).total_seconds() / (SURFACE_HALF_LIFE_DAYS * 86400.0) * -1.0
        )
    factor = decay_factor(last, now, SURFACE_HALF_LIFE_DAYS)
    for key in SURFACE_STATS_KEYS:
        slot["stats"][key] = float(slot["stats"][key]) * factor
    slot["last_surface_decay_at"] = isoformat(now)
    return 1.0


def validate_plan(
    plan: dict[str, Any], state: dict[str, Any], *, for_reserve: bool
) -> None:
    require_exact_keys(plan, PLAN_KEYS, "plan")
    if plan["plan_version"] != 4:
        raise StateError("unsupported plan_version")
    for field in (
        "plan_id",
        "root_id",
        "episode_id",
        "role",
        "task_shape_hash",
        "brief_hash",
        "verifier_hash",
        "target_hash",
        "controller_version",
        "policy_epoch",
        "route_epoch",
        "evidence_grade",
        "planned_at",
        "terminal_due_at",
        "issuer_authority",
    ):
        require_text(plan[field], f"plan.{field}")
    for field in ("plan_id", "root_id", "episode_id"):
        require_concrete_text(plan[field], f"plan.{field}")
    require_resolved_model_id(plan["resolved_model_id"], "plan.resolved_model_id")
    require_text(plan["comparison_id"], "plan.comparison_id", allow_empty=True)
    require_text(plan["route_cohort_hash"], "plan.route_cohort_hash", allow_empty=True)
    if (
        not isinstance(plan["comparison_run_epoch"], int)
        or isinstance(plan["comparison_run_epoch"], bool)
        or not 0 <= plan["comparison_run_epoch"] <= COMPARISON_RUN_EPOCH_MAX
    ):
        raise StateError("plan comparison_run_epoch is outside fixed bounds")
    for field in ("task_shape_hash", "brief_hash", "verifier_hash", "target_hash"):
        require_sha256(plan[field], f"plan.{field}")
    if plan["role"] not in ROLE_ORDER:
        raise StateError("plan role is not a fixed controller role")
    if plan["controller_version"] != state["controller_version"]:
        raise StateError("plan controller_version mismatch")
    if plan["policy_epoch"] != state["policy_epoch"]:
        raise StateError("plan policy_epoch mismatch")
    if plan["issuer_authority"] != "lead":
        raise StateError("only the lead can reserve a route episode")
    if plan["evidence_grade"] not in EVIDENCE_GRADES:
        raise StateError("invalid plan evidence_grade")
    if plan["evidence_grade"] == "comparable":
        if not plan["comparison_id"] or not plan["route_cohort_hash"]:
            raise StateError(
                "comparable plans require comparison_id and route_cohort_hash"
            )
        if for_reserve and plan["comparison_run_epoch"] != 0:
            raise StateError(
                "reserve assigns comparison_run_epoch under the state lock"
            )
        if not for_reserve and plan["comparison_run_epoch"] <= 0:
            raise StateError("pending comparable plans require an assigned run epoch")
    elif (
        plan["comparison_id"]
        or plan["route_cohort_hash"]
        or plan["comparison_run_epoch"] != 0
    ):
        raise StateError("ordinary plans cannot carry comparison metadata")
    if plan["route_cohort_hash"]:
        require_sha256(plan["route_cohort_hash"], "plan.route_cohort_hash")
    if not isinstance(plan["route"], dict):
        raise StateError("plan.route must be an object")
    require_exact_keys(plan["route"], ROUTE_KEYS, "plan.route")
    for field in ROUTE_KEYS:
        require_route_component(plan["route"][field], f"plan.route.{field}")
    if not isinstance(plan["eligible_routes"], list) or not plan["eligible_routes"]:
        raise StateError("plan eligible_routes must be a non-empty list")
    if len(plan["eligible_routes"]) > SLOTS_PER_ROLE:
        raise StateError("plan eligible_routes exceeds the fixed per-role capacity")
    if len(set(plan["eligible_routes"])) != len(plan["eligible_routes"]):
        raise StateError("plan eligible_routes must be unique")
    for index, value in enumerate(plan["eligible_routes"]):
        require_text(value, f"plan.eligible_routes[{index}]")
        parse_route_key(value)
    if route_key(plan["route"]) not in plan["eligible_routes"]:
        raise StateError("planned route was not in the lead-approved eligible set")
    if not isinstance(plan["correction_budget"], int) or isinstance(
        plan["correction_budget"], bool
    ):
        raise StateError("correction_budget must be an integer")
    if not 0 <= plan["correction_budget"] <= 1:
        raise StateError("correction_budget must be zero or one")
    if (
        not isinstance(plan["terminal_budget_seconds"], int)
        or isinstance(plan["terminal_budget_seconds"], bool)
        or plan["terminal_budget_seconds"] <= 0
    ):
        raise StateError("terminal_budget_seconds must be a positive integer")
    planned_at = parse_time(plan["planned_at"], "plan.planned_at")
    terminal_due_at = parse_time(plan["terminal_due_at"], "plan.terminal_due_at")
    if terminal_due_at <= planned_at:
        raise StateError("terminal_due_at must be after planned_at")
    if (terminal_due_at - planned_at).total_seconds() != plan[
        "terminal_budget_seconds"
    ]:
        raise StateError("terminal_due_at must match terminal_budget_seconds")
    slot = find_slot(state, plan["role"], plan["route"])
    if plan["route_epoch"] != slot["route_epoch"]:
        raise StateError("plan route_epoch mismatch")
    if (
        not isinstance(plan["route_generation"], int)
        or isinstance(plan["route_generation"], bool)
        or plan["route_generation"] != slot["route_generation"]
    ):
        raise StateError("plan route_generation mismatch")
    if (
        slot["resolved_model_id"]
        and plan["resolved_model_id"] != slot["resolved_model_id"]
    ):
        raise StateError(
            "resolved_model_id drift requires an explicit route-slot reset"
        )
    if plan["evidence_grade"] == "comparable" and plan[
        "route_cohort_hash"
    ] != route_cohort_hash(state, plan["role"], plan["eligible_routes"]):
        raise StateError(
            "route_cohort_hash does not match the current eligible route cohort"
        )
    if (
        plan["evidence_grade"] == "comparable"
        and slot["comparison_id"] == plan["comparison_id"]
        and slot["comparison_contract_hash"]
        and slot["comparison_contract_hash"] != comparison_contract_hash(plan)
    ):
        raise StateError("comparison_id was reused with a changed comparison contract")
    if plan["evidence_grade"] == "comparable" and not for_reserve:
        role_state = state["roles"][plan["role"]]
        if (
            role_state["active_comparison_signature"] != comparison_signature(plan)
            or role_state["active_comparison_run_epoch"] != plan["comparison_run_epoch"]
        ):
            raise StateError(
                "pending comparable plan is not in the active comparison run"
            )
    if for_reserve and slot["quarantined"]:
        raise StateError("cannot reserve a quarantined route")


def overdue_pending(state: dict[str, Any], now: datetime) -> list[str]:
    return [
        plan["plan_id"]
        for plan in state["pending"]
        if parse_time(plan["terminal_due_at"], "terminal_due_at") <= now
    ]


def replay_capacity_exhausted(state: dict[str, Any]) -> bool:
    return (
        state["completed_episode_filter"]["insertions"] + len(state["pending"])
        >= COMPLETED_FILTER_MAX_INSERTIONS
    )


def reserve_plan(
    state: dict[str, Any], plan: dict[str, Any], *, now: datetime | None = None
) -> None:
    validate_state(state)
    validate_plan(plan, state, for_reserve=True)
    planned_at = parse_time(plan["planned_at"], "plan.planned_at")
    action_time = now or planned_at
    if planned_at > action_time:
        raise StateError("planned_at cannot be in the future at reservation time")
    if parse_time(plan["terminal_due_at"], "terminal_due_at") <= action_time:
        raise StateError("route plan is already past terminal_due_at")
    overdue = overdue_pending(state, action_time)
    if overdue:
        raise StateError(f"overdue terminal receipts block new reservations: {overdue}")
    if len(state["pending"]) >= PENDING_CAPACITY:
        raise StateError("pending table is full; adaptive routing must abstain")
    if replay_capacity_exhausted(state):
        raise StateError(
            "fixed replay filter capacity is exhausted; adaptation must abstain"
        )
    if any(existing["plan_id"] == plan["plan_id"] for existing in state["pending"]):
        raise StateError("duplicate plan_id")
    if any(
        (existing["root_id"], existing["episode_id"])
        == (plan["root_id"], plan["episode_id"])
        for existing in state["pending"]
    ):
        raise StateError("root/episode already has a pending route plan")
    if completed_episode_seen(state, plan["root_id"], plan["episode_id"]):
        raise StateError(
            "root/episode is already terminal or rejected by the fixed replay filter"
        )
    if plan["evidence_grade"] == "comparable" and any(
        existing["role"] == plan["role"] and existing["evidence_grade"] == "comparable"
        for existing in state["pending"]
    ):
        raise StateError("only one comparable episode may be in flight per role")
    if plan["evidence_grade"] == "comparable":
        role_state = state["roles"][plan["role"]]
        incoming_contract = comparison_contract_hash(plan)
        if (
            role_state["active_comparison_id"] == plan["comparison_id"]
            and role_state["active_comparison_contract_hash"]
            and role_state["active_comparison_contract_hash"] != incoming_contract
        ):
            raise StateError(
                "comparison_id was reused with a changed role-wide comparison contract"
            )
        signature = comparison_signature(plan)
        if role_state["active_comparison_signature"] != signature:
            if role_state["active_comparison_run_epoch"] >= COMPARISON_RUN_EPOCH_MAX:
                raise StateError(
                    "comparison run epoch capacity is exhausted; adaptation must abstain"
                )
            role_state["active_comparison_id"] = plan["comparison_id"]
            role_state["active_comparison_contract_hash"] = incoming_contract
            role_state["active_route_cohort_hash"] = plan["route_cohort_hash"]
            role_state["active_comparison_signature"] = signature
            role_state["active_comparison_run_epoch"] += 1
        plan["comparison_run_epoch"] = role_state["active_comparison_run_epoch"]
    slot = find_slot(state, plan["role"], plan["route"])
    if not slot["resolved_model_id"]:
        slot["resolved_model_id"] = plan["resolved_model_id"]
        slot["route_change_reason"] = "first-resolved-identity-bind"
        slot["route_changed_at"] = isoformat(action_time)
    state["pending"].append(copy.deepcopy(plan))
    state["pending"].sort(
        key=lambda item: (item["planned_at"], item["root_id"], item["episode_id"])
    )
    state["updated_at"] = isoformat(
        max(parse_time(state["updated_at"], "updated_at"), action_time)
    )
    validate_state(state)


def validate_receipt(receipt: dict[str, Any], state: dict[str, Any]) -> None:
    require_exact_keys(receipt, RECEIPT_KEYS, "receipt")
    if receipt["receipt_version"] != 4:
        raise StateError("unsupported receipt_version")
    for field in (
        "receipt_id",
        "root_id",
        "episode_id",
        "role",
        "task_shape_hash",
        "brief_hash",
        "verifier_hash",
        "target_hash",
        "controller_version",
        "policy_epoch",
        "route_epoch",
        "evidence_grade",
        "delivery_outcome",
        "target_outcome",
        "review_target_verdict",
        "surface_status",
        "acceptance_authority",
        "observed_at",
        "evidence_id",
    ):
        require_text(receipt[field], f"receipt.{field}")
    for field in ("receipt_id", "root_id", "episode_id", "evidence_id"):
        require_concrete_text(receipt[field], f"receipt.{field}")
    require_resolved_model_id(receipt["resolved_model_id"], "receipt.resolved_model_id")
    require_text(receipt["plan_id"], "receipt.plan_id", allow_empty=True)
    require_concrete_text(receipt["plan_id"], "receipt.plan_id", allow_empty=True)
    for field in ("task_shape_hash", "brief_hash", "verifier_hash", "target_hash"):
        require_sha256(receipt[field], f"receipt.{field}")
    require_text(receipt["comparison_id"], "receipt.comparison_id", allow_empty=True)
    require_text(
        receipt["route_cohort_hash"], "receipt.route_cohort_hash", allow_empty=True
    )
    if (
        not isinstance(receipt["comparison_run_epoch"], int)
        or isinstance(receipt["comparison_run_epoch"], bool)
        or not 0 <= receipt["comparison_run_epoch"] <= COMPARISON_RUN_EPOCH_MAX
    ):
        raise StateError("receipt comparison_run_epoch is outside fixed bounds")
    require_text(
        receipt["supersedes_receipt_id"],
        "receipt.supersedes_receipt_id",
        allow_empty=True,
    )
    require_concrete_text(
        receipt["supersedes_receipt_id"],
        "receipt.supersedes_receipt_id",
        allow_empty=True,
    )
    if receipt["role"] not in ROLE_ORDER:
        raise StateError("receipt role is not a fixed controller role")
    if receipt["controller_version"] != state["controller_version"]:
        raise StateError("receipt controller_version mismatch")
    if receipt["policy_epoch"] != state["policy_epoch"]:
        raise StateError("receipt policy_epoch mismatch")
    if receipt["acceptance_authority"] != "lead":
        raise StateError("only a lead disposition can update routing state")
    if receipt["evidence_grade"] not in EVIDENCE_GRADES:
        raise StateError("invalid evidence_grade")
    if receipt["evidence_grade"] == "comparable":
        if (
            not receipt["comparison_id"]
            or not receipt["route_cohort_hash"]
            or receipt["comparison_run_epoch"] <= 0
        ):
            raise StateError(
                "comparable evidence requires id, cohort, and assigned run epoch"
            )
    elif (
        receipt["comparison_id"]
        or receipt["route_cohort_hash"]
        or receipt["comparison_run_epoch"] != 0
    ):
        raise StateError("ordinary evidence cannot carry comparison metadata")
    if receipt["route_cohort_hash"]:
        require_sha256(receipt["route_cohort_hash"], "receipt.route_cohort_hash")
    if receipt["delivery_outcome"] not in OBSERVED_OUTCOMES:
        raise StateError("invalid delivery_outcome")
    if receipt["target_outcome"] not in TARGET_OUTCOMES:
        raise StateError("invalid target_outcome")
    if receipt["review_target_verdict"] not in REVIEW_TARGET_VERDICTS:
        raise StateError("invalid review_target_verdict")
    if receipt["surface_status"] not in SURFACE_STATUSES:
        raise StateError("invalid surface_status")
    validate_receipt_outcome_disposition(receipt)
    if (
        receipt["surface_status"] != "ok"
        and receipt["delivery_outcome"] != "indeterminate"
    ):
        raise StateError("surface failures cannot update model capability")
    if (
        receipt["delivery_outcome"] == "false_accept"
        and not receipt["supersedes_receipt_id"]
    ):
        raise StateError("false_accept requires supersedes_receipt_id")
    if receipt["delivery_outcome"] == "false_accept" and receipt["plan_id"]:
        raise StateError("false_accept must be a planless post-hoc correction")
    if receipt["delivery_outcome"] != "false_accept" and not receipt["plan_id"]:
        raise StateError("terminal receipts require a reserved plan_id")
    if (
        receipt["delivery_outcome"] != "false_accept"
        and receipt["supersedes_receipt_id"]
    ):
        raise StateError("only false_accept may supersede an accepted receipt")
    if not isinstance(receipt["route"], dict):
        raise StateError("receipt.route must be an object")
    require_exact_keys(receipt["route"], ROUTE_KEYS, "receipt.route")
    for field in ROUTE_KEYS:
        require_route_component(receipt["route"][field], f"receipt.route.{field}")
    if (
        not isinstance(receipt["eligible_routes"], list)
        or not receipt["eligible_routes"]
    ):
        raise StateError("eligible_routes must be a non-empty list")
    if len(receipt["eligible_routes"]) > SLOTS_PER_ROLE:
        raise StateError("eligible_routes exceeds the fixed per-role capacity")
    if len(set(receipt["eligible_routes"])) != len(receipt["eligible_routes"]):
        raise StateError("eligible_routes must be unique")
    for index, value in enumerate(receipt["eligible_routes"]):
        require_text(value, f"eligible_routes[{index}]")
        parse_route_key(value)
    if route_key(receipt["route"]) not in receipt["eligible_routes"]:
        raise StateError("resolved route was not in the predeclared eligible set")
    for field in ("elapsed_seconds", "lead_seconds"):
        require_number(receipt[field], f"receipt.{field}")
    require_number(receipt["cost_usd"], "receipt.cost_usd", nullable=True)
    if (
        not isinstance(receipt["correction_count"], int)
        or isinstance(receipt["correction_count"], bool)
        or receipt["correction_count"] < 0
    ):
        raise StateError("correction_count must be a non-negative integer")
    if (
        not isinstance(receipt["route_generation"], int)
        or isinstance(receipt["route_generation"], bool)
        or receipt["route_generation"] <= 0
    ):
        raise StateError("receipt route_generation must be positive")
    if (
        not isinstance(receipt["correction_budget"], int)
        or isinstance(receipt["correction_budget"], bool)
        or not 0 <= receipt["correction_budget"] <= 1
    ):
        raise StateError("receipt correction_budget must be zero or one")
    if (
        not isinstance(receipt["terminal_budget_seconds"], int)
        or isinstance(receipt["terminal_budget_seconds"], bool)
        or receipt["terminal_budget_seconds"] <= 0
    ):
        raise StateError("receipt terminal_budget_seconds must be positive")
    if (
        receipt["delivery_outcome"] == "accepted_first_pass"
        and receipt["correction_count"] != 0
    ):
        raise StateError("first-pass acceptance cannot have corrections")
    if (
        receipt["delivery_outcome"] == "accepted_after_correction"
        and receipt["correction_count"] != 1
    ):
        raise StateError("corrected acceptance requires exactly one correction")
    parse_time(receipt["observed_at"], "receipt.observed_at")


def validate_receipt_outcome_disposition(receipt: dict[str, Any]) -> None:
    role = receipt["role"]
    delivery = receipt["delivery_outcome"]
    target = receipt["target_outcome"]
    review = receipt["review_target_verdict"]
    surface = receipt["surface_status"]

    if surface != "ok":
        valid = (
            delivery == "indeterminate"
            and target in ("not_reached", "not_applicable")
            and review == "not_applicable"
        )
    elif delivery == "indeterminate":
        valid = target in ("not_reached", "not_applicable") and review == (
            "not_applicable"
        )
    elif role in REVIEW_ROLES:
        if delivery in ("accepted_first_pass", "accepted_after_correction"):
            valid = (target, review) in (
                ("accepted", "pass"),
                ("rejected", "hold"),
                ("not_reached", "invalidated"),
            )
        elif delivery == "rejected_or_escalated":
            valid = (target, review) == ("not_reached", "not_applicable")
        else:
            valid = (target, review) == ("rejected", "invalidated")
    elif delivery in ("accepted_first_pass", "accepted_after_correction"):
        valid = (target, review) == ("accepted", "not_applicable")
    elif delivery in ("rejected_or_escalated", "false_accept"):
        valid = (target, review) == ("rejected", "not_applicable")
    else:
        valid = False

    if not valid:
        raise StateError(
            "receipt outcome disposition is inconsistent with its role and "
            "surface status"
        )


def match_receipt_to_plan(receipt: dict[str, Any], plan: dict[str, Any]) -> None:
    scalar_pairs = (
        ("root_id", "root_id"),
        ("episode_id", "episode_id"),
        ("role", "role"),
        ("task_shape_hash", "task_shape_hash"),
        ("brief_hash", "brief_hash"),
        ("verifier_hash", "verifier_hash"),
        ("target_hash", "target_hash"),
        ("controller_version", "controller_version"),
        ("policy_epoch", "policy_epoch"),
        ("route_epoch", "route_epoch"),
        ("route_generation", "route_generation"),
        ("resolved_model_id", "resolved_model_id"),
        ("evidence_grade", "evidence_grade"),
        ("comparison_id", "comparison_id"),
        ("route_cohort_hash", "route_cohort_hash"),
        ("comparison_run_epoch", "comparison_run_epoch"),
        ("correction_budget", "correction_budget"),
        ("terminal_budget_seconds", "terminal_budget_seconds"),
    )
    mismatches = [
        receipt_key
        for receipt_key, plan_key in scalar_pairs
        if receipt[receipt_key] != plan[plan_key]
    ]
    if receipt["eligible_routes"] != plan["eligible_routes"]:
        mismatches.append("eligible_routes")
    if receipt["route"] != plan["route"]:
        mismatches.append("route")
    if mismatches:
        raise StateError(
            f"receipt does not match reserved plan fields: {sorted(mismatches)}"
        )
    if receipt["correction_count"] > plan["correction_budget"]:
        raise StateError("receipt exceeds the reserved correction budget")
    if parse_time(receipt["observed_at"], "receipt.observed_at") < parse_time(
        plan["planned_at"], "plan.planned_at"
    ):
        raise StateError("terminal receipt predates its reserved route plan")


def find_slot(
    state: dict[str, Any], role: str, route: dict[str, str]
) -> dict[str, Any]:
    key = route_key(route)
    matches = [
        slot
        for slot in state["roles"][role]["slots"]
        if slot["active"] and route_key(slot["route"]) == key
    ]
    if len(matches) != 1:
        raise StateError(
            f"route {key} does not identify exactly one active slot in {role}"
        )
    return matches[0]


def apply_receipt(
    state: dict[str, Any], receipt: dict[str, Any], *, now: datetime | None = None
) -> None:
    validate_state(state)
    validate_receipt(receipt, state)
    if receipt["receipt_id"] in state["receipt_ring"]:
        raise StateError("duplicate receipt_id in fixed dedupe window")
    slot = find_slot(state, receipt["role"], receipt["route"])
    if any(
        item["receipt_id"] == receipt["receipt_id"] for item in slot["evidence_ring"]
    ):
        raise StateError("duplicate receipt_id in target route evidence window")
    if receipt["route_epoch"] != slot["route_epoch"]:
        raise StateError("route_epoch mismatch; replace the slot after identity drift")
    if receipt["route_generation"] != slot["route_generation"]:
        raise StateError(
            "route_generation mismatch; receipt targets a stale route slot"
        )
    late_unlinked_false_accept = False
    repeated_false_accept = False
    matching_plans = [
        plan for plan in state["pending"] if plan["plan_id"] == receipt["plan_id"]
    ]
    if matching_plans:
        if len(matching_plans) != 1:
            raise StateError("plan_id does not identify exactly one pending episode")
        plan = matching_plans[0]
        match_receipt_to_plan(receipt, plan)
    elif receipt["delivery_outcome"] == "false_accept":
        prior = next(
            (
                item
                for item in slot["evidence_ring"]
                if item["receipt_id"] == receipt["supersedes_receipt_id"]
            ),
            None,
        )
        identity_fields = (
            "root_id",
            "episode_id",
            "task_shape_hash",
            "brief_hash",
            "verifier_hash",
            "target_hash",
            "route_epoch",
            "route_generation",
            "resolved_model_id",
            "evidence_grade",
            "comparison_id",
            "route_cohort_hash",
            "comparison_run_epoch",
            "eligible_routes",
            "correction_budget",
            "terminal_budget_seconds",
        )
        if prior is not None and any(
            prior[field] != receipt[field] for field in identity_fields
        ):
            prior = None
        if prior is not None and prior["delivery_outcome"] not in (
            "accepted_first_pass",
            "accepted_after_correction",
        ):
            raise StateError(
                "false_accept must supersede a previously accepted receipt"
            )
        if prior is not None and parse_time(
            receipt["observed_at"], "receipt.observed_at"
        ) < parse_time(prior["observed_at"], "superseded receipt observed_at"):
            raise StateError("false_accept cannot predate its accepted receipt")
        late_unlinked_false_accept = prior is None
        if late_unlinked_false_accept and not accepted_route_seen(state, receipt):
            raise StateError(
                "late false_accept does not match an accepted route generation"
            )
        repeated_false_accept = state["corrected_accept_filter"][
            "insertions"
        ] >= COMPLETED_FILTER_MAX_INSERTIONS or corrected_accept_seen(state, receipt)
        plan = None
    else:
        raise StateError("missing reserved route plan for terminal receipt")
    if (
        not slot["resolved_model_id"]
        or receipt["resolved_model_id"] != slot["resolved_model_id"]
    ):
        raise StateError(
            "receipt resolved_model_id does not match the bound route slot"
        )
    observed_at = parse_time(receipt["observed_at"], "receipt.observed_at")
    action_time = now or observed_at
    if observed_at > action_time:
        raise StateError("receipt observed_at cannot be in the future at update time")
    outcome = receipt["delivery_outcome"]
    incoming_contract = None
    evidence_comparison_generation = 0
    role_state = state["roles"][receipt["role"]]
    active_comparison_receipt = (
        receipt["evidence_grade"] == "comparable"
        and role_state["active_comparison_signature"] == comparison_signature(receipt)
        and role_state["active_comparison_run_epoch"] == receipt["comparison_run_epoch"]
    )
    if receipt["evidence_grade"] == "comparable" and receipt["surface_status"] == "ok":
        incoming_contract = comparison_contract_hash(receipt)
        if (
            active_comparison_receipt
            and slot["comparison_id"] == receipt["comparison_id"]
            and slot["comparison_contract_hash"]
            and slot["comparison_contract_hash"] != incoming_contract
        ):
            raise StateError(
                "comparison_id was reused with a changed comparison contract"
            )
    stats = slot["stats"]
    surface_status = receipt["surface_status"]
    if not repeated_false_accept:
        surface_weight = apply_surface_decay(slot, observed_at)
        stats["surface_attempt_n"] += surface_weight
        surface_key = {
            "auth_client_failure": "surface_auth_client_failure",
            "runtime_failure": "surface_runtime_failure",
            "target_invalid": "surface_target_invalid",
            "receipt_invalid": "surface_receipt_invalid",
        }.get(surface_status)
        if surface_key:
            stats[surface_key] += surface_weight
            stats["surface_failure_seconds_sum"] += surface_weight * float(
                receipt["elapsed_seconds"]
            )

    if (
        surface_status == "ok"
        and outcome != "indeterminate"
        and not late_unlinked_false_accept
        and not repeated_false_accept
    ):
        quality_weight = apply_quality_decay(slot, observed_at)
        stats = slot["stats"]
        comparable_active = False
        if active_comparison_receipt:
            incoming_identity = (
                receipt["comparison_id"],
                receipt["route_cohort_hash"],
                receipt["comparison_run_epoch"],
                str(incoming_contract),
            )
            current_identity = (
                slot["comparison_id"],
                slot["route_cohort_hash"],
                slot["comparison_run_epoch"],
                slot["comparison_contract_hash"],
            )
            current_event = (
                parse_time(slot["comparison_event_at"], "comparison_event_at")
                if slot["comparison_id"]
                else None
            )
            if (
                current_event is None
                or receipt["comparison_run_epoch"] > slot["comparison_run_epoch"]
            ):
                for key in STATS_KEYS:
                    if key.startswith("comparable_"):
                        stats[key] = 0.0
                slot["comparison_id"] = receipt["comparison_id"]
                slot["route_cohort_hash"] = receipt["route_cohort_hash"]
                slot["comparison_run_epoch"] = receipt["comparison_run_epoch"]
                slot["comparison_contract_hash"] = str(incoming_contract)
                slot["comparison_event_at"] = receipt["observed_at"]
                slot["comparison_generation"] += 1
                comparable_active = True
            elif incoming_identity == current_identity:
                comparable_active = True
                if observed_at > current_event:
                    slot["comparison_event_at"] = receipt["observed_at"]
            if comparable_active:
                evidence_comparison_generation = slot["comparison_generation"]
        outcome_suffix = {
            "accepted_first_pass": "first_pass",
            "accepted_after_correction": "corrected",
            "rejected_or_escalated": "rejected",
            "false_accept": "false_accept",
        }[outcome]
        stats["observed_n"] += quality_weight
        stats[f"observed_{outcome_suffix}"] += quality_weight
        stats["observed_elapsed_seconds_sum"] += quality_weight * float(
            receipt["elapsed_seconds"]
        )
        stats["observed_lead_seconds_sum"] += quality_weight * float(
            receipt["lead_seconds"]
        )
        if receipt["cost_usd"] is not None:
            stats["observed_cost_usd_sum"] += quality_weight * float(
                receipt["cost_usd"]
            )
            stats["observed_cost_n"] += quality_weight
        if comparable_active:
            stats["comparable_n"] += quality_weight
            stats[f"comparable_{outcome_suffix}"] += quality_weight
            stats["comparable_elapsed_seconds_sum"] += quality_weight * float(
                receipt["elapsed_seconds"]
            )
            stats["comparable_lead_seconds_sum"] += quality_weight * float(
                receipt["lead_seconds"]
            )
            if receipt["cost_usd"] is not None:
                stats["comparable_cost_usd_sum"] += quality_weight * float(
                    receipt["cost_usd"]
                )
                stats["comparable_cost_n"] += quality_weight
        if outcome == "false_accept":
            slot["quarantined"] = True
            slot["quarantine_reason"] = f"false_accept:{receipt['evidence_id']}"[
                :MAX_TEXT
            ]
    if late_unlinked_false_accept or repeated_false_accept:
        slot["quarantined"] = True
        reason = (
            "repeated_false_accept"
            if repeated_false_accept
            else "late_unlinked_false_accept"
        )
        slot["quarantine_reason"] = f"{reason}:{receipt['evidence_id']}"[:MAX_TEXT]

    if not repeated_false_accept:
        slot["evidence_ring"].append(
            {
                "receipt_id": receipt["receipt_id"],
                "root_id": receipt["root_id"],
                "episode_id": receipt["episode_id"],
                "observed_at": receipt["observed_at"],
                "evidence_grade": receipt["evidence_grade"],
                "delivery_outcome": outcome,
                "surface_status": surface_status,
                "comparison_id": receipt["comparison_id"],
                "route_cohort_hash": receipt["route_cohort_hash"],
                "comparison_run_epoch": receipt["comparison_run_epoch"],
                "task_shape_hash": receipt["task_shape_hash"],
                "brief_hash": receipt["brief_hash"],
                "verifier_hash": receipt["verifier_hash"],
                "target_hash": receipt["target_hash"],
                "route_epoch": receipt["route_epoch"],
                "route_generation": receipt["route_generation"],
                "resolved_model_id": receipt["resolved_model_id"],
                "eligible_routes": copy.deepcopy(receipt["eligible_routes"]),
                "correction_budget": receipt["correction_budget"],
                "terminal_budget_seconds": receipt["terminal_budget_seconds"],
                "comparison_generation": evidence_comparison_generation,
            }
        )
        slot["evidence_ring"].sort(
            key=lambda item: (
                parse_time(item["observed_at"], "observed_at"),
                item["receipt_id"],
            )
        )
        slot["evidence_ring"] = slot["evidence_ring"][-EVIDENCE_RING_SIZE:]
    state["receipt_ring"].append(receipt["receipt_id"])
    state["receipt_ring"] = state["receipt_ring"][-RECEIPT_RING_SIZE:]
    if outcome == "false_accept" and not repeated_false_accept:
        mark_corrected_accept(state, receipt)
    if plan is not None:
        mark_terminal_episode(state, receipt)
        state["pending"] = [
            item for item in state["pending"] if item["plan_id"] != plan["plan_id"]
        ]
    state["updated_at"] = isoformat(
        max(parse_time(state["updated_at"], "updated_at"), observed_at, action_time)
    )
    validate_state(state)


def slot_metrics(slot: dict[str, Any], now: datetime) -> dict[str, Any]:
    stats = decayed_stats(slot, now)
    n = stats["comparable_n"]
    quality = None
    quality_lcb = None
    expected_seconds = None
    expected_cost = None
    if n > 0:
        quality = (
            stats["comparable_first_pass"]
            + CORRECTED_CREDIT * stats["comparable_corrected"]
        ) / n
        quality_lcb = max(0.0, quality - LCB_PENALTY / math.sqrt(n))
        expected_seconds = (
            stats["comparable_elapsed_seconds_sum"]
            + stats["comparable_lead_seconds_sum"]
        ) / n
        if stats["comparable_cost_n"] > 0:
            expected_cost = (
                stats["comparable_cost_usd_sum"] / stats["comparable_cost_n"]
            )
    recent_cutoff = now.timestamp() - QUALITY_HALF_LIFE_DAYS * 86400.0
    recent_comparable = [
        item
        for item in slot["evidence_ring"]
        if item["evidence_grade"] == "comparable"
        and item["surface_status"] == "ok"
        and item["delivery_outcome"] != "indeterminate"
        and item["comparison_id"] == slot["comparison_id"]
        and item["route_cohort_hash"] == slot["route_cohort_hash"]
        and item["comparison_run_epoch"] == slot["comparison_run_epoch"]
        and item["comparison_generation"] == slot["comparison_generation"]
        and comparison_contract_hash(item) == slot["comparison_contract_hash"]
        and parse_time(item["observed_at"], "observed_at").timestamp() >= recent_cutoff
    ]
    recent_roots = {item["root_id"] for item in recent_comparable}
    supported = (
        len(recent_comparable) >= MIN_COMPARABLE_EVENTS
        and len(recent_roots) >= MIN_DISTINCT_ROOTS
    )
    comparison_ids = [slot["comparison_id"]] if slot["comparison_id"] else []
    surface_attempts = stats["surface_attempt_n"]
    surface_failures = sum(
        stats[key]
        for key in (
            "surface_auth_client_failure",
            "surface_runtime_failure",
            "surface_target_invalid",
            "surface_receipt_invalid",
        )
    )
    surface_failure_rate = (
        None if surface_attempts <= 0 else surface_failures / surface_attempts
    )
    expected_surface_delay = (
        None
        if surface_attempts <= 0
        else stats["surface_failure_seconds_sum"] / surface_attempts
    )
    return {
        "slot": slot["slot"],
        "route": route_key(slot["route"]),
        "route_epoch": slot["route_epoch"],
        "route_generation": slot["route_generation"],
        "resolved_model_id": slot["resolved_model_id"],
        "comparison_id": slot["comparison_id"],
        "route_cohort_hash": slot["route_cohort_hash"],
        "comparison_run_epoch": slot["comparison_run_epoch"],
        "comparison_contract_hash": slot["comparison_contract_hash"],
        "comparison_event_at": slot["comparison_event_at"],
        "comparison_generation": slot["comparison_generation"],
        "quarantined": slot["quarantined"],
        "comparable_n": round(n, 6),
        "recent_distinct_roots": len(recent_roots),
        "comparison_ids": comparison_ids,
        "supported": supported,
        "quality": None if quality is None else round(quality, 6),
        "quality_lcb": None if quality_lcb is None else round(quality_lcb, 6),
        "expected_acceptance_seconds": None
        if expected_seconds is None
        else round(expected_seconds, 3),
        "expected_cost_usd": None if expected_cost is None else round(expected_cost, 6),
        "surface_failure_rate": None
        if surface_failure_rate is None
        else round(surface_failure_rate, 6),
        "expected_surface_delay_seconds": None
        if expected_surface_delay is None
        else round(expected_surface_delay, 3),
    }


def select_route(
    state: dict[str, Any],
    role: str,
    eligible: set[str],
    available: set[str],
    now: datetime,
) -> dict[str, Any]:
    validate_state(state)
    if role not in ROLE_ORDER:
        raise StateError("unknown fixed role")
    if replay_capacity_exhausted(state):
        return {
            "decision": "abstain",
            "role": role,
            "route": None,
            "reason_codes": ["replay_filter_capacity_exhausted"],
            "state_digest": state_digest(state),
            "learned_parameter_count": learned_parameter_count(),
            "candidates": [],
        }
    overdue = overdue_pending(state, now)
    if overdue:
        return {
            "decision": "abstain",
            "role": role,
            "route": None,
            "reason_codes": ["overdue_terminal_receipt"],
            "overdue_plan_ids": overdue,
            "state_digest": state_digest(state),
            "learned_parameter_count": learned_parameter_count(),
            "candidates": [],
        }
    if len(state["pending"]) >= PENDING_CAPACITY:
        return {
            "decision": "abstain",
            "role": role,
            "route": None,
            "reason_codes": ["pending_capacity_exhausted"],
            "state_digest": state_digest(state),
            "learned_parameter_count": learned_parameter_count(),
            "candidates": [],
        }
    role_state = state["roles"][role]
    candidates = [
        slot
        for slot in role_state["slots"]
        if slot["active"]
        and not slot["quarantined"]
        and route_key(slot["route"]) in eligible
        and route_key(slot["route"]) in available
    ]
    if not candidates:
        return {
            "decision": "abstain",
            "role": role,
            "route": None,
            "reason_codes": ["no_policy_eligible_live_route"],
            "state_digest": state_digest(state),
            "learned_parameter_count": learned_parameter_count(),
            "candidates": [],
        }
    candidates.sort(key=lambda slot: slot["slot"])
    candidate_route_keys = [route_key(slot["route"]) for slot in candidates]
    current_cohort_hash = route_cohort_hash(state, role, candidate_route_keys)
    baseline_candidate = next(
        (slot for slot in candidates if slot["slot"] == role_state["baseline_slot"]),
        None,
    )
    metrics = [slot_metrics(slot, now) for slot in candidates]
    if role == "major_decision" and baseline_candidate is None:
        return {
            "decision": "abstain",
            "role": role,
            "route": None,
            "fallback_route": route_key(
                role_state["slots"][role_state["baseline_slot"]]["route"]
            ),
            "comparable_eligible_routes": sorted(candidate_route_keys),
            "route_cohort_hash": current_cohort_hash,
            "active_comparison_run_epoch": role_state["active_comparison_run_epoch"],
            "reason_codes": ["major_decision_static_route_unavailable"],
            "state_digest": state_digest(state),
            "learned_parameter_count": learned_parameter_count(),
            "candidates": metrics,
        }
    fallback = baseline_candidate or candidates[0]
    supported = [
        metric
        for metric in metrics
        if metric["supported"]
        and metric["comparison_run_epoch"] == role_state["active_comparison_run_epoch"]
        and metric["route_cohort_hash"] == current_cohort_hash
    ]
    selected = fallback
    reason_codes = ["static_fallback_insufficient_comparable_support"]
    comparison_groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for metric in supported:
        for comparison_id in metric["comparison_ids"]:
            group_key = (
                comparison_id,
                metric["comparison_contract_hash"],
                metric["route_cohort_hash"],
                metric["comparison_run_epoch"],
            )
            comparison_groups.setdefault(group_key, []).append(metric)
    pairable_groups = [group for group in comparison_groups.values() if len(group) >= 2]
    if role == "major_decision":
        reason_codes = ["static_policy_major_decision_no_auto_promotion"]
    elif pairable_groups:
        pairable_groups.sort(
            key=lambda group: (-len(group), sorted(item["route"] for item in group))
        )
        supported = pairable_groups[0]
        best_lcb = max(float(metric["quality_lcb"]) for metric in supported)
        quality_peers = [
            metric
            for metric in supported
            if float(metric["quality_lcb"]) >= best_lcb - QUALITY_MARGIN
        ]
        quality_peers.sort(
            key=lambda metric: (
                float("inf")
                if metric["expected_acceptance_seconds"] is None
                else metric["expected_acceptance_seconds"],
                float("inf")
                if metric["expected_cost_usd"] is None
                else metric["expected_cost_usd"],
                metric["slot"],
            )
        )
        winning_slot = quality_peers[0]["slot"]
        selected = next(slot for slot in candidates if slot["slot"] == winning_slot)
        reason_codes = [
            "comparable_quality_supported",
            "time_to_final_acceptance_tiebreak",
            "cost_last",
        ]
    return {
        "decision": "route",
        "role": role,
        "route": route_key(selected["route"]),
        "slot": selected["slot"],
        "route_epoch": selected["route_epoch"],
        "route_generation": selected["route_generation"],
        "fallback_route": route_key(fallback["route"]),
        "comparable_eligible_routes": sorted(candidate_route_keys),
        "route_cohort_hash": current_cohort_hash,
        "active_comparison_run_epoch": role_state["active_comparison_run_epoch"],
        "reason_codes": reason_codes,
        "state_digest": state_digest(state),
        "learned_parameter_count": learned_parameter_count(),
        "candidates": metrics,
    }


def replace_slot(
    state: dict[str, Any],
    role: str,
    slot_index: int,
    route: dict[str, str],
    route_epoch: str,
    resolved_model_id: str,
    active: bool,
    reason: str,
    now: datetime,
) -> None:
    validate_state(state)
    if role not in ROLE_ORDER or not 0 <= slot_index < SLOTS_PER_ROLE:
        raise StateError("invalid role or slot")
    require_text(route_epoch, "route_epoch", allow_empty=not active)
    require_text(resolved_model_id, "resolved_model_id", allow_empty=not active)
    require_text(reason, "reason")
    for field in ROUTE_KEYS:
        require_route_component(route[field], f"route.{field}", allow_empty=not active)
    if active and not all(route.values()):
        raise StateError("active replacement requires a complete route")
    role_state = state["roles"][role]
    current_slot = role_state["slots"][slot_index]
    current_route = current_slot["route"]
    current_route_key = route_key(current_route)
    if any(
        plan["role"] == role and plan["route"] == current_route
        for plan in state["pending"]
    ):
        raise StateError("cannot replace a route generation with a pending episode")
    if any(
        plan["role"] == role
        and plan["evidence_grade"] == "comparable"
        and current_route_key in plan["eligible_routes"]
        for plan in state["pending"]
    ):
        raise StateError(
            "cannot replace a route generation in a pending comparison cohort"
        )
    if slot_index == role_state["baseline_slot"] and not active:
        raise StateError("cannot deactivate a baseline slot")
    if (
        slot_index == role_state["baseline_slot"]
        and active
        and route_key(route) != route_key(current_route)
    ):
        raise StateError(
            "baseline route key is policy-owned and cannot be replaced by runtime state"
        )
    if now < parse_time(state["updated_at"], "updated_at"):
        raise StateError("out-of-order route replacement")
    if active:
        duplicate = [
            slot
            for slot in role_state["slots"]
            if slot["slot"] != slot_index
            and slot["active"]
            and route_key(slot["route"]) == route_key(route)
        ]
        if duplicate:
            raise StateError("replacement would duplicate an active route in the role")
    role_state["slots"][slot_index] = {
        "slot": slot_index,
        "active": active,
        "quarantined": False,
        "quarantine_reason": "",
        "route_change_reason": reason,
        "route_changed_at": isoformat(now),
        "route_epoch": route_epoch,
        "route_generation": current_slot["route_generation"] + 1,
        "route": route,
        "resolved_model_id": resolved_model_id,
        "comparison_id": "",
        "route_cohort_hash": "",
        "comparison_run_epoch": 0,
        "comparison_contract_hash": "",
        "comparison_event_at": "",
        "comparison_generation": 0,
        "stats": zero_stats(),
        "last_quality_decay_at": isoformat(now),
        "last_surface_decay_at": isoformat(now),
        "evidence_ring": [],
    }
    state["updated_at"] = isoformat(now)
    validate_state(state)


def cmd_init(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.state)
    with state_lock(path, exclusive=True):
        if path.exists():
            raise StateError(f"state already exists: {path}")
        action_time = utc_now()
        now = parse_time(args.at, "--at") if args.at else action_time
        if now > action_time:
            raise StateError("initial state time cannot be in the future")
        state = build_state(load_json(seed_path()), now)
        atomic_write(path, state)
    return {
        "status": "initialized",
        "state": str(path),
        "state_digest": state_digest(state),
        "learned_parameter_count": learned_parameter_count(),
    }


def cmd_validate(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.state)
    with state_lock(path, exclusive=False):
        state = load_json(path)
        validate_state(state)
    return {
        "status": "valid",
        "state": str(path),
        "state_digest": state_digest(state),
        "roles": len(ROLE_ORDER),
        "slots": len(ROLE_ORDER) * SLOTS_PER_ROLE,
        "stats_per_slot": len(STATS_KEYS),
        "learned_parameter_count": learned_parameter_count(),
        "pending_count": len(state["pending"]),
        "pending_capacity": PENDING_CAPACITY,
        "completed_episode_insertions": state["completed_episode_filter"]["insertions"],
        "completed_episode_filter_bits": COMPLETED_FILTER_BITS,
        "accepted_route_insertions": state["accepted_route_filter"]["insertions"],
        "accepted_route_filter_bits": COMPLETED_FILTER_BITS,
        "corrected_accept_insertions": state["corrected_accept_filter"]["insertions"],
        "corrected_accept_filter_bits": COMPLETED_FILTER_BITS,
    }


def cmd_show(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.state)
    with state_lock(path, exclusive=False):
        state = load_json(path)
        now = parse_time(args.at, "--at") if args.at else utc_now()
        validate_state(state)
        roles = {
            role: [
                slot_metrics(slot, now)
                for slot in state["roles"][role]["slots"]
                if slot["active"]
            ]
            for role in ROLE_ORDER
        }
    return {
        "status": "ok",
        "state": str(path),
        "policy_epoch": state["policy_epoch"],
        "state_digest": state_digest(state),
        "learned_parameter_count": learned_parameter_count(),
        "pending": copy.deepcopy(state["pending"]),
        "roles": roles,
    }


def cmd_select(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.state)
    eligible = set(args.eligible)
    available = set(args.available)
    for value in eligible | available:
        parse_route_key(value)
    with state_lock(path, exclusive=False):
        state = load_json(path)
        now = parse_time(args.at, "--at") if args.at else utc_now()
        return select_route(state, args.role, eligible, available, now)


def cmd_reserve(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.state)
    plan = load_json(Path(args.plan)) if args.plan != "-" else load_stdin_json()
    if not isinstance(plan, dict):
        raise StateError("plan must be a JSON object")
    with state_lock(path, exclusive=True):
        state = load_json(path)
        reserve_plan(state, plan, now=utc_now())
        atomic_write(path, state)
    return {
        "status": "reserved",
        "plan_id": plan["plan_id"],
        "route_cohort_hash": plan["route_cohort_hash"],
        "comparison_run_epoch": plan["comparison_run_epoch"],
        "pending_count": len(state["pending"]),
        "pending_capacity": PENDING_CAPACITY,
        "state_digest": state_digest(state),
    }


def cmd_update(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.state)
    receipt = (
        load_json(Path(args.receipt)) if args.receipt != "-" else load_stdin_json()
    )
    if not isinstance(receipt, dict):
        raise StateError("receipt must be a JSON object")
    with state_lock(path, exclusive=True):
        state = load_json(path)
        apply_receipt(state, receipt, now=utc_now())
        atomic_write(path, state)
    slot = find_slot(state, receipt["role"], receipt["route"])
    return {
        "status": "updated",
        "receipt_id": receipt["receipt_id"],
        "route": route_key(receipt["route"]),
        "quarantined": slot["quarantined"],
        "state_digest": state_digest(state),
        "learned_parameter_count": learned_parameter_count(),
    }


def cmd_replace(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm_reset:
        raise StateError("replace requires --confirm-reset")
    path = Path(args.state)
    route = (
        parse_route_key(args.route)
        if args.active
        else {"surface": "", "model": "", "effort": ""}
    )
    with state_lock(path, exclusive=True):
        state = load_json(path)
        action_time = utc_now()
        now = parse_time(args.at, "--at") if args.at else action_time
        if now > action_time:
            raise StateError("route replacement time cannot be in the future")
        replace_slot(
            state,
            args.role,
            args.slot,
            route,
            args.route_epoch,
            args.resolved_model_id,
            args.active,
            args.reason,
            now,
        )
        atomic_write(path, state)
    return {
        "status": "replaced",
        "role": args.role,
        "slot": args.slot,
        "active": args.active,
        "route": route_key(route) if args.active else None,
        "state_digest": state_digest(state),
        "learned_parameter_count": learned_parameter_count(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=str(default_state_path()))
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="initialize fixed-capacity state")
    init_parser.add_argument("--at")
    init_parser.set_defaults(func=cmd_init)

    validate_parser = subparsers.add_parser(
        "validate", help="validate state shape and invariants"
    )
    validate_parser.set_defaults(func=cmd_validate)

    show_parser = subparsers.add_parser("show", help="show decayed route summaries")
    show_parser.add_argument("--at")
    show_parser.set_defaults(func=cmd_show)

    select_parser = subparsers.add_parser(
        "select", help="select only within lead-approved routes"
    )
    select_parser.add_argument("--role", required=True, choices=ROLE_ORDER)
    select_parser.add_argument("--eligible", action="append", required=True)
    select_parser.add_argument("--available", action="append", required=True)
    select_parser.add_argument("--at")
    select_parser.set_defaults(func=cmd_select)

    reserve_parser = subparsers.add_parser(
        "reserve", help="reserve one lead-issued route plan before spawn"
    )
    reserve_parser.add_argument(
        "--plan", required=True, help="JSON path or - for stdin"
    )
    reserve_parser.set_defaults(func=cmd_reserve)

    update_parser = subparsers.add_parser(
        "update", help="resolve one reserved episode with a lead receipt"
    )
    update_parser.add_argument(
        "--receipt", required=True, help="JSON path or - for stdin"
    )
    update_parser.set_defaults(func=cmd_update)

    replace_parser = subparsers.add_parser(
        "replace", help="reset one fixed route slot after identity drift"
    )
    replace_parser.add_argument("--role", required=True, choices=ROLE_ORDER)
    replace_parser.add_argument("--slot", required=True, type=int)
    replace_parser.add_argument("--route", default="::")
    replace_parser.add_argument("--route-epoch", default="")
    replace_parser.add_argument("--resolved-model-id", default="")
    replace_parser.add_argument(
        "--active", action=argparse.BooleanOptionalAction, default=True
    )
    replace_parser.add_argument("--reason", required=True)
    replace_parser.add_argument("--confirm-reset", action="store_true")
    replace_parser.add_argument("--at")
    replace_parser.set_defaults(func=cmd_replace)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = args.func(args)
    except (StateError, OSError, json.JSONDecodeError) as exc:
        print(
            json.dumps({"status": "error", "error": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
