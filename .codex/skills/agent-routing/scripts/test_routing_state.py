#!/usr/bin/env python3
"""Deterministic tests for the fixed-capacity routing controller."""

from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import routing_state as rs


START = datetime(2026, 8, 9, tzinfo=timezone.utc)
TERRA = {"surface": "codex", "model": "gpt-5.6-terra", "effort": "high"}
SONNET = {"surface": "claude-code", "model": "sonnet", "effort": "high"}
TERRA_MEDIUM = {"surface": "codex", "model": "gpt-5.6-terra", "effort": "medium"}
SOL_MAX = {"surface": "codex", "model": "gpt-5.6-sol", "effort": "max"}
FABLE_XHIGH = {"surface": "claude-code", "model": "fable", "effort": "xhigh"}


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


INITIAL_STATE = rs.build_state(rs.load_json(rs.seed_path()), START)
INITIAL_ROUTE_COHORT_HASH = rs.route_cohort_hash(
    INITIAL_STATE,
    "bounded_builder",
    [rs.route_key(TERRA), rs.route_key(SONNET)],
)


def make_receipt(
    index: int,
    *,
    route: dict[str, str] = TERRA,
    root_id: str | None = None,
    at: datetime | None = None,
    evidence_grade: str = "ordinary",
    delivery_outcome: str = "accepted_first_pass",
    surface_status: str = "ok",
    elapsed_seconds: float = 100.0,
    lead_seconds: float = 10.0,
    cost_usd: float | None = 1.0,
    correction_count: int | None = None,
    correction_budget: int = 1,
    terminal_budget_seconds: int = 3600,
    supersedes_receipt_id: str = "",
    comparison_id: str = "comparison-a",
) -> dict[str, object]:
    observed_at = at or START + timedelta(minutes=index + 1)
    if correction_count is None:
        correction_count = 1 if delivery_outcome == "accepted_after_correction" else 0
    if surface_status != "ok":
        delivery_outcome = "indeterminate"
    if surface_status != "ok" or delivery_outcome == "indeterminate":
        target_outcome = "not_reached"
    elif delivery_outcome in ("rejected_or_escalated", "false_accept"):
        target_outcome = "rejected"
    else:
        target_outcome = "accepted"
    return {
        "receipt_version": 4,
        "plan_id": f"plan-{index}",
        "receipt_id": f"receipt-{index}",
        "root_id": root_id or f"root-{index}",
        "episode_id": f"episode-{index}",
        "role": "bounded_builder",
        "eligible_routes": [rs.route_key(TERRA), rs.route_key(SONNET)],
        "task_shape_hash": digest("task-shape"),
        "brief_hash": digest("brief"),
        "verifier_hash": digest("verifier"),
        "target_hash": digest("target"),
        "controller_version": rs.CONTROLLER_VERSION,
        "policy_epoch": "2026-08-09-v1",
        "route_epoch": "2026-08-09",
        "route_generation": 1,
        "route": copy.deepcopy(route),
        "resolved_model_id": route["model"] + "-resolved",
        "evidence_grade": evidence_grade,
        "comparison_id": comparison_id if evidence_grade == "comparable" else "",
        "route_cohort_hash": INITIAL_ROUTE_COHORT_HASH
        if evidence_grade == "comparable"
        else "",
        "comparison_run_epoch": 0,
        "delivery_outcome": delivery_outcome,
        "target_outcome": target_outcome,
        "review_target_verdict": "not_applicable",
        "surface_status": surface_status,
        "elapsed_seconds": elapsed_seconds,
        "lead_seconds": lead_seconds,
        "cost_usd": cost_usd,
        "correction_count": correction_count,
        "correction_budget": correction_budget,
        "terminal_budget_seconds": terminal_budget_seconds,
        "acceptance_authority": "lead",
        "observed_at": rs.isoformat(observed_at),
        "evidence_id": f"session:{index}",
        "supersedes_receipt_id": supersedes_receipt_id,
    }


def plan_from_receipt(receipt: dict[str, object]) -> dict[str, object]:
    observed_at = rs.parse_time(str(receipt["observed_at"]), "observed_at")
    planned_at = observed_at - timedelta(minutes=1)
    return {
        "plan_version": 4,
        "plan_id": receipt["plan_id"],
        "root_id": receipt["root_id"],
        "episode_id": receipt["episode_id"],
        "role": receipt["role"],
        "eligible_routes": copy.deepcopy(receipt["eligible_routes"]),
        "task_shape_hash": receipt["task_shape_hash"],
        "brief_hash": receipt["brief_hash"],
        "verifier_hash": receipt["verifier_hash"],
        "target_hash": receipt["target_hash"],
        "controller_version": receipt["controller_version"],
        "policy_epoch": receipt["policy_epoch"],
        "route_epoch": receipt["route_epoch"],
        "route_generation": receipt["route_generation"],
        "route": copy.deepcopy(receipt["route"]),
        "resolved_model_id": receipt["resolved_model_id"],
        "evidence_grade": receipt["evidence_grade"],
        "comparison_id": receipt["comparison_id"],
        "route_cohort_hash": receipt["route_cohort_hash"],
        "comparison_run_epoch": receipt["comparison_run_epoch"],
        "correction_budget": receipt["correction_budget"],
        "terminal_budget_seconds": receipt["terminal_budget_seconds"],
        "planned_at": rs.isoformat(planned_at),
        "terminal_due_at": rs.isoformat(
            planned_at + timedelta(seconds=int(receipt["terminal_budget_seconds"]))
        ),
        "issuer_authority": "lead",
    }


def apply_with_plan(state: dict[str, object], receipt: dict[str, object]) -> None:
    plan = plan_from_receipt(receipt)
    rs.reserve_plan(state, plan)
    receipt["comparison_run_epoch"] = plan["comparison_run_epoch"]
    rs.apply_receipt(state, receipt)


def retarget_receipt(
    state: dict[str, object],
    receipt: dict[str, object],
    *,
    role: str,
    route: dict[str, str],
    eligible_routes: list[str],
) -> dict[str, object]:
    slot = rs.find_slot(state, role, route)
    receipt["role"] = role
    receipt["route"] = copy.deepcopy(route)
    receipt["eligible_routes"] = copy.deepcopy(eligible_routes)
    receipt["route_epoch"] = slot["route_epoch"]
    receipt["route_generation"] = slot["route_generation"]
    receipt["resolved_model_id"] = (
        slot["resolved_model_id"] or route["model"] + "-resolved"
    )
    if receipt["evidence_grade"] == "comparable":
        receipt["route_cohort_hash"] = rs.route_cohort_hash(
            state, role, eligible_routes
        )
        receipt["comparison_run_epoch"] = 0
    return receipt


class RoutingStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = rs.build_state(rs.load_json(rs.seed_path()), START)

    def slot(self, route: dict[str, str] = TERRA) -> dict[str, object]:
        return rs.find_slot(self.state, "bounded_builder", route)

    def test_fixed_shape_and_strict_schema(self) -> None:
        rs.validate_state(self.state)
        self.assertEqual(rs.learned_parameter_count(), 7 * 3 * 24)
        self.assertEqual(len(self.state["roles"]), 7)
        self.assertTrue(
            all(len(role["slots"]) == 3 for role in self.state["roles"].values())
        )
        self.assertEqual(
            len(self.state["completed_episode_filter"]["bits"]),
            rs.COMPLETED_FILTER_HEX_LENGTH,
        )
        self.assertEqual(
            len(self.state["accepted_route_filter"]["bits"]),
            rs.COMPLETED_FILTER_HEX_LENGTH,
        )
        self.assertEqual(
            len(self.state["corrected_accept_filter"]["bits"]),
            rs.COMPLETED_FILTER_HEX_LENGTH,
        )
        broken = copy.deepcopy(self.state)
        broken["unexpected"] = True
        with self.assertRaisesRegex(rs.StateError, "keys mismatch"):
            rs.validate_state(broken)

    def test_cold_start_uses_static_fallback(self) -> None:
        result = rs.select_route(
            self.state,
            "bounded_builder",
            {rs.route_key(TERRA), rs.route_key(SONNET)},
            {rs.route_key(TERRA), rs.route_key(SONNET)},
            START,
        )
        self.assertEqual(result["route"], rs.route_key(TERRA))
        self.assertIn(
            "static_fallback_insufficient_comparable_support", result["reason_codes"]
        )

    def test_ordinary_receipt_cannot_promote(self) -> None:
        for index in range(8):
            receipt = make_receipt(
                index,
                route=SONNET,
                at=START + timedelta(minutes=index + 1),
                elapsed_seconds=1.0,
            )
            apply_with_plan(self.state, receipt)
        result = rs.select_route(
            self.state,
            "bounded_builder",
            {rs.route_key(TERRA), rs.route_key(SONNET)},
            {rs.route_key(TERRA), rs.route_key(SONNET)},
            START + timedelta(hours=1),
        )
        self.assertEqual(result["route"], rs.route_key(TERRA))
        self.assertEqual(self.slot(SONNET)["stats"]["comparable_n"], 0.0)

    def test_comparable_multi_root_evidence_can_change_tiebreak(self) -> None:
        for index in range(6):
            apply_with_plan(
                self.state,
                make_receipt(
                    index,
                    route=TERRA,
                    root_id=f"terra-root-{index % 3}",
                    at=START + timedelta(minutes=index + 1),
                    evidence_grade="comparable",
                    delivery_outcome="accepted_after_correction",
                    elapsed_seconds=1000.0,
                    lead_seconds=100.0,
                ),
            )
        for offset in range(6):
            index = 10 + offset
            apply_with_plan(
                self.state,
                make_receipt(
                    index,
                    route=SONNET,
                    root_id=f"sonnet-root-{offset % 3}",
                    at=START + timedelta(minutes=index + 1),
                    evidence_grade="comparable",
                    delivery_outcome="accepted_first_pass",
                    elapsed_seconds=100.0,
                    lead_seconds=10.0,
                ),
            )
        result = rs.select_route(
            self.state,
            "bounded_builder",
            {rs.route_key(TERRA), rs.route_key(SONNET)},
            {rs.route_key(TERRA), rs.route_key(SONNET)},
            START + timedelta(hours=1),
        )
        self.assertEqual(result["route"], rs.route_key(SONNET))
        self.assertIn("comparable_quality_supported", result["reason_codes"])

    def test_ordinary_surface_receipts_cannot_promote_a_challenger(self) -> None:
        for route, offset, elapsed in ((TERRA, 0, 100.0), (SONNET, 10, 200.0)):
            for local_index in range(6):
                index = offset + local_index
                apply_with_plan(
                    self.state,
                    make_receipt(
                        index,
                        route=route,
                        root_id=f"{route['model']}-root-{local_index % 3}",
                        at=START + timedelta(minutes=index + 1),
                        evidence_grade="comparable",
                        elapsed_seconds=elapsed,
                    ),
                )
        for index in (30, 31):
            apply_with_plan(
                self.state,
                make_receipt(
                    index,
                    route=TERRA,
                    at=START + timedelta(minutes=index + 1),
                    surface_status="runtime_failure",
                    elapsed_seconds=10_000.0,
                ),
            )
        result = rs.select_route(
            self.state,
            "bounded_builder",
            {rs.route_key(TERRA), rs.route_key(SONNET)},
            {rs.route_key(TERRA), rs.route_key(SONNET)},
            START + timedelta(hours=1),
        )
        self.assertEqual(result["route"], rs.route_key(TERRA))
        self.assertIn("comparable_quality_supported", result["reason_codes"])
        terra_metrics = next(
            item
            for item in result["candidates"]
            if item["route"] == rs.route_key(TERRA)
        )
        self.assertGreater(terra_metrics["expected_surface_delay_seconds"], 1_000.0)
        self.assertEqual(terra_metrics["expected_acceptance_seconds"], 110.0)

    def test_major_decision_never_auto_promotes(self) -> None:
        eligible = [rs.route_key(SOL_MAX), rs.route_key(FABLE_XHIGH)]
        for route, offset, outcome, elapsed in (
            (SOL_MAX, 100, "accepted_after_correction", 1_000.0),
            (FABLE_XHIGH, 110, "accepted_first_pass", 100.0),
        ):
            for local_index in range(6):
                index = offset + local_index
                receipt = make_receipt(
                    index,
                    root_id=f"{route['model']}-{route['effort']}-root-{local_index % 3}",
                    at=START + timedelta(minutes=index + 1),
                    evidence_grade="comparable",
                    delivery_outcome=outcome,
                    elapsed_seconds=elapsed,
                )
                apply_with_plan(
                    self.state,
                    retarget_receipt(
                        self.state,
                        receipt,
                        role="major_decision",
                        route=route,
                        eligible_routes=eligible,
                    ),
                )
        result = rs.select_route(
            self.state,
            "major_decision",
            set(eligible),
            set(eligible),
            START + timedelta(hours=3),
        )
        self.assertEqual(result["route"], rs.route_key(SOL_MAX))
        self.assertEqual(
            result["reason_codes"],
            ["static_policy_major_decision_no_auto_promotion"],
        )

    def test_major_decision_abstains_when_static_route_is_unavailable(self) -> None:
        result = rs.select_route(
            self.state,
            "major_decision",
            {rs.route_key(FABLE_XHIGH)},
            {rs.route_key(FABLE_XHIGH)},
            START,
        )
        self.assertEqual(result["decision"], "abstain")
        self.assertIsNone(result["route"])
        self.assertEqual(
            result["reason_codes"],
            ["major_decision_static_route_unavailable"],
        )

    def test_surface_failure_leaves_capability_bytes_unchanged(self) -> None:
        apply_with_plan(self.state, make_receipt(1, at=START + timedelta(minutes=1)))
        slot = self.slot()
        before_stats = {key: slot["stats"][key] for key in rs.QUALITY_STATS_KEYS}
        before_time = slot["last_quality_decay_at"]
        apply_with_plan(
            self.state,
            make_receipt(
                2,
                at=START + timedelta(minutes=2),
                surface_status="auth_client_failure",
                elapsed_seconds=12.0,
            ),
        )
        slot = self.slot()
        self.assertEqual(
            before_stats, {key: slot["stats"][key] for key in rs.QUALITY_STATS_KEYS}
        )
        self.assertEqual(before_time, slot["last_quality_decay_at"])
        self.assertEqual(slot["stats"]["surface_auth_client_failure"], 1.0)

    def test_reviewer_hold_is_a_delivery_success(self) -> None:
        receipt = make_receipt(1)
        receipt["role"] = "lifecycle_reviewer"
        receipt["route"] = {
            "surface": "claude-code",
            "model": "opus",
            "effort": "xhigh",
        }
        receipt["eligible_routes"] = ["claude-code:opus:xhigh"]
        receipt["resolved_model_id"] = "opus-resolved"
        receipt["target_outcome"] = "rejected"
        receipt["review_target_verdict"] = "hold"
        apply_with_plan(self.state, receipt)
        slot = rs.find_slot(self.state, "lifecycle_reviewer", receipt["route"])
        self.assertAlmostEqual(slot["stats"]["observed_first_pass"], 1.0, places=4)
        self.assertEqual(slot["stats"]["observed_rejected"], 0.0)

    def test_receipt_role_outcome_matrix_rejects_inconsistent_dispositions(
        self,
    ) -> None:
        inconsistent = (
            ("bounded_builder", "accepted_first_pass", "not_reached", "not_applicable"),
            ("semantic_builder", "accepted_first_pass", "accepted", "hold"),
            ("lifecycle_reviewer", "accepted_first_pass", "accepted", "not_applicable"),
            ("semantic_reviewer", "rejected_or_escalated", "accepted", "pass"),
        )
        for role, delivery, target, review in inconsistent:
            with self.subTest(
                role=role,
                delivery=delivery,
                target=target,
                review=review,
            ):
                receipt = make_receipt(1, delivery_outcome=delivery)
                receipt["role"] = role
                receipt["target_outcome"] = target
                receipt["review_target_verdict"] = review
                with self.assertRaisesRegex(rs.StateError, "outcome disposition"):
                    rs.validate_receipt(receipt, self.state)

    def test_receipt_role_outcome_matrix_accepts_reviewer_findings(self) -> None:
        valid_findings = (
            ("accepted", "pass"),
            ("rejected", "hold"),
            ("not_reached", "invalidated"),
        )
        for target, review in valid_findings:
            with self.subTest(target=target, review=review):
                receipt = make_receipt(1)
                receipt["role"] = "semantic_reviewer"
                receipt["target_outcome"] = target
                receipt["review_target_verdict"] = review
                rs.validate_receipt(receipt, self.state)

    def test_receipt_role_outcome_matrix_is_exhaustive(self) -> None:
        accepted = {"accepted_first_pass", "accepted_after_correction"}
        for role, delivery, target, review, surface in product(
            rs.ROLE_ORDER,
            rs.OBSERVED_OUTCOMES,
            rs.TARGET_OUTCOMES,
            rs.REVIEW_TARGET_VERDICTS,
            rs.SURFACE_STATUSES,
        ):
            if surface != "ok":
                expected = (
                    delivery == "indeterminate"
                    and target in {"not_reached", "not_applicable"}
                    and review == "not_applicable"
                )
            elif delivery == "indeterminate":
                expected = (
                    target in {"not_reached", "not_applicable"}
                    and review == "not_applicable"
                )
            elif role in rs.REVIEW_ROLES:
                expected = (
                    (
                        delivery in accepted
                        and (target, review)
                        in {
                            ("accepted", "pass"),
                            ("rejected", "hold"),
                            ("not_reached", "invalidated"),
                        }
                    )
                    or (
                        delivery == "rejected_or_escalated"
                        and (target, review) == ("not_reached", "not_applicable")
                    )
                    or (
                        delivery == "false_accept"
                        and (target, review) == ("rejected", "invalidated")
                    )
                )
            elif delivery in accepted:
                expected = (target, review) == ("accepted", "not_applicable")
            elif delivery in {"rejected_or_escalated", "false_accept"}:
                expected = (target, review) == ("rejected", "not_applicable")
            else:
                expected = False

            disposition = {
                "role": role,
                "delivery_outcome": delivery,
                "target_outcome": target,
                "review_target_verdict": review,
                "surface_status": surface,
            }
            if expected:
                rs.validate_receipt_outcome_disposition(disposition)
            else:
                with self.assertRaisesRegex(rs.StateError, "outcome disposition"):
                    rs.validate_receipt_outcome_disposition(disposition)

    def test_false_accept_quarantines_without_rewriting_history(self) -> None:
        first = make_receipt(1, at=START + timedelta(minutes=1))
        apply_with_plan(self.state, first)
        later = copy.deepcopy(first)
        later.update(
            {
                "plan_id": "",
                "receipt_id": "receipt-false-accept",
                "delivery_outcome": "false_accept",
                "target_outcome": "rejected",
                "observed_at": rs.isoformat(START + timedelta(minutes=2)),
                "evidence_id": "session:false-accept",
                "supersedes_receipt_id": first["receipt_id"],
            }
        )
        rs.apply_receipt(self.state, later)
        slot = self.slot()
        self.assertTrue(slot["quarantined"])
        self.assertAlmostEqual(slot["stats"]["observed_first_pass"], 1.0, places=4)
        self.assertEqual(slot["stats"]["observed_false_accept"], 1.0)

    def test_false_accept_correction_is_numeric_at_most_once(self) -> None:
        accepted = make_receipt(1, at=START + timedelta(minutes=1))
        apply_with_plan(self.state, accepted)
        first = copy.deepcopy(accepted)
        first.update(
            {
                "plan_id": "",
                "receipt_id": "false-accept-first",
                "delivery_outcome": "false_accept",
                "target_outcome": "rejected",
                "observed_at": rs.isoformat(START + timedelta(minutes=2)),
                "evidence_id": "session:false-accept-first",
                "supersedes_receipt_id": accepted["receipt_id"],
            }
        )
        rs.apply_receipt(self.state, first)
        stats_after_first = copy.deepcopy(self.slot()["stats"])
        evidence_after_first = copy.deepcopy(self.slot()["evidence_ring"])
        repeated = copy.deepcopy(first)
        repeated.update(
            {
                "receipt_id": "false-accept-repeated",
                "observed_at": rs.isoformat(START + timedelta(minutes=3)),
                "evidence_id": "session:false-accept-repeated",
            }
        )
        rs.apply_receipt(self.state, repeated)
        self.assertEqual(self.slot()["stats"], stats_after_first)
        self.assertEqual(self.slot()["evidence_ring"], evidence_after_first)
        self.assertTrue(self.slot()["quarantined"])
        self.assertTrue(
            self.slot()["quarantine_reason"].startswith("repeated_false_accept:")
        )
        self.assertEqual(self.state["corrected_accept_filter"]["insertions"], 1)

    def test_false_accept_cannot_cross_episode_or_route(self) -> None:
        first = make_receipt(1, at=START + timedelta(minutes=1))
        apply_with_plan(self.state, first)
        later = copy.deepcopy(first)
        later.update(
            {
                "plan_id": "",
                "receipt_id": "receipt-wrong-episode",
                "episode_id": "different-episode",
                "delivery_outcome": "false_accept",
                "target_outcome": "rejected",
                "observed_at": rs.isoformat(START + timedelta(minutes=2)),
                "evidence_id": "session:wrong-episode",
                "supersedes_receipt_id": first["receipt_id"],
            }
        )
        with self.assertRaisesRegex(rs.StateError, "accepted route generation"):
            rs.apply_receipt(self.state, later)
        self.assertFalse(self.slot()["quarantined"])

    def test_false_accept_cannot_use_a_pending_plan_as_supersession_proof(self) -> None:
        receipt = make_receipt(1, at=START + timedelta(minutes=2))
        rs.reserve_plan(self.state, plan_from_receipt(receipt))
        receipt.update(
            {
                "delivery_outcome": "false_accept",
                "target_outcome": "rejected",
                "supersedes_receipt_id": "does-not-exist",
            }
        )
        with self.assertRaisesRegex(rs.StateError, "planless post-hoc"):
            rs.apply_receipt(self.state, receipt)
        self.assertEqual(len(self.state["pending"]), 1)

    def test_false_accept_must_supersede_an_accepted_receipt(self) -> None:
        rejected = make_receipt(
            1,
            at=START + timedelta(minutes=1),
            delivery_outcome="rejected_or_escalated",
        )
        rejected["target_outcome"] = "rejected"
        apply_with_plan(self.state, rejected)
        later = copy.deepcopy(rejected)
        later.update(
            {
                "plan_id": "",
                "receipt_id": "receipt-false-accept-after-rejection",
                "delivery_outcome": "false_accept",
                "observed_at": rs.isoformat(START + timedelta(minutes=2)),
                "evidence_id": "session:false-accept-after-rejection",
                "supersedes_receipt_id": rejected["receipt_id"],
            }
        )
        with self.assertRaisesRegex(rs.StateError, "previously accepted"):
            rs.apply_receipt(self.state, later)
        self.assertFalse(self.slot()["quarantined"])

    def test_false_accept_cannot_predate_its_accepted_receipt(self) -> None:
        accepted = make_receipt(1, at=START + timedelta(minutes=10))
        apply_with_plan(self.state, accepted)
        before = copy.deepcopy(self.slot())
        correction = copy.deepcopy(accepted)
        correction.update(
            {
                "plan_id": "",
                "receipt_id": "receipt-predating-false-accept",
                "delivery_outcome": "false_accept",
                "target_outcome": "rejected",
                "observed_at": rs.isoformat(START + timedelta(minutes=5)),
                "evidence_id": "session:predating-false-accept",
                "supersedes_receipt_id": accepted["receipt_id"],
            }
        )
        with self.assertRaisesRegex(rs.StateError, "cannot predate"):
            rs.apply_receipt(self.state, correction)
        self.assertEqual(self.slot(), before)

    def test_new_comparison_id_resets_only_comparable_projection(self) -> None:
        for index in range(5):
            apply_with_plan(
                self.state,
                make_receipt(
                    index,
                    root_id=f"root-{index % 3}",
                    at=START + timedelta(minutes=index + 1),
                    evidence_grade="comparable",
                    comparison_id="comparison-a",
                ),
            )
        slot = self.slot()
        observed_before = slot["stats"]["observed_n"]
        apply_with_plan(
            self.state,
            make_receipt(
                10,
                root_id="new-root",
                at=START + timedelta(minutes=11),
                evidence_grade="comparable",
                comparison_id="comparison-b",
            ),
        )
        slot = self.slot()
        self.assertEqual(slot["comparison_id"], "comparison-b")
        self.assertAlmostEqual(slot["stats"]["comparable_n"], 1.0, places=4)
        self.assertGreater(slot["stats"]["observed_n"], observed_before)
        metrics = rs.slot_metrics(slot, START + timedelta(minutes=12))
        self.assertFalse(metrics["supported"])

    def test_same_comparison_id_rejects_contract_drift(self) -> None:
        first = make_receipt(
            1,
            at=START + timedelta(minutes=1),
            evidence_grade="comparable",
        )
        apply_with_plan(self.state, first)
        changed = make_receipt(
            2,
            at=START + timedelta(minutes=2),
            evidence_grade="comparable",
        )
        changed["task_shape_hash"] = digest("different-task-shape")
        before = self.slot()["stats"]["comparable_n"]
        with self.assertRaisesRegex(rs.StateError, "changed comparison contract"):
            rs.reserve_plan(self.state, plan_from_receipt(changed))
        self.assertEqual(self.slot()["stats"]["comparable_n"], before)

    def test_same_comparison_id_rejects_cross_slot_contract_drift(self) -> None:
        first = make_receipt(
            1,
            route=TERRA,
            at=START + timedelta(minutes=1),
            evidence_grade="comparable",
        )
        apply_with_plan(self.state, first)
        changed = make_receipt(
            2,
            route=SONNET,
            at=START + timedelta(minutes=2),
            evidence_grade="comparable",
        )
        changed["task_shape_hash"] = digest("cross-slot-different-task-shape")
        with self.assertRaisesRegex(rs.StateError, "role-wide comparison contract"):
            rs.reserve_plan(self.state, plan_from_receipt(changed))
        self.assertEqual(self.slot(SONNET)["stats"]["comparable_n"], 0.0)

    def test_same_comparison_id_rejects_correction_budget_drift(self) -> None:
        apply_with_plan(
            self.state,
            make_receipt(
                1,
                at=START + timedelta(minutes=1),
                evidence_grade="comparable",
                correction_budget=0,
            ),
        )
        changed = make_receipt(
            2,
            at=START + timedelta(minutes=2),
            evidence_grade="comparable",
            correction_budget=1,
        )
        with self.assertRaisesRegex(rs.StateError, "changed comparison contract"):
            rs.reserve_plan(self.state, plan_from_receipt(changed))

    def test_comparison_id_a_b_a_cannot_pair_with_a_stale_peer(self) -> None:
        for route, offset, outcome, elapsed in (
            (TERRA, 0, "accepted_after_correction", 1_000.0),
            (SONNET, 10, "accepted_first_pass", 100.0),
        ):
            for local_index in range(6):
                index = offset + local_index
                apply_with_plan(
                    self.state,
                    make_receipt(
                        index,
                        route=route,
                        root_id=f"old-{route['model']}-root-{local_index % 3}",
                        at=START + timedelta(minutes=index + 1),
                        evidence_grade="comparable",
                        delivery_outcome=outcome,
                        elapsed_seconds=elapsed,
                    ),
                )
        old_peer_run = self.slot(SONNET)["comparison_run_epoch"]
        apply_with_plan(
            self.state,
            make_receipt(
                100,
                route=TERRA,
                root_id="comparison-b-root",
                at=START + timedelta(minutes=30),
                evidence_grade="comparable",
                comparison_id="comparison-b",
                delivery_outcome="accepted_after_correction",
                elapsed_seconds=1_000.0,
            ),
        )
        eligible = [rs.route_key(TERRA), rs.route_key(SONNET)]
        for local_index in range(6):
            index = 200 + local_index
            apply_with_plan(
                self.state,
                make_receipt(
                    index,
                    route=TERRA,
                    root_id=f"fresh-terra-root-{local_index % 3}",
                    at=START + timedelta(minutes=40 + local_index),
                    evidence_grade="comparable",
                    comparison_id="comparison-a",
                    delivery_outcome="accepted_after_correction",
                    elapsed_seconds=1_000.0,
                ),
            )
        result = rs.select_route(
            self.state,
            "bounded_builder",
            set(eligible),
            set(eligible),
            START + timedelta(hours=1),
        )
        self.assertEqual(result["route"], rs.route_key(TERRA))
        self.assertEqual(
            result["reason_codes"],
            ["static_fallback_insufficient_comparable_support"],
        )
        metrics = {item["route"]: item for item in result["candidates"]}
        self.assertEqual(
            metrics[rs.route_key(TERRA)]["route_cohort_hash"],
            metrics[rs.route_key(SONNET)]["route_cohort_hash"],
        )
        self.assertEqual(metrics[rs.route_key(TERRA)]["comparison_generation"], 3)
        self.assertEqual(metrics[rs.route_key(SONNET)]["comparison_generation"], 1)
        self.assertEqual(old_peer_run, 1)
        self.assertEqual(metrics[rs.route_key(TERRA)]["comparison_run_epoch"], 3)
        self.assertEqual(metrics[rs.route_key(SONNET)]["comparison_run_epoch"], 1)

    def test_route_cohort_drift_invalidates_old_supported_group(self) -> None:
        eligible = [
            rs.route_key(TERRA),
            rs.route_key(SONNET),
            rs.route_key(TERRA_MEDIUM),
        ]
        for route, offset, outcome, elapsed in (
            (TERRA, 0, "accepted_after_correction", 1_000.0),
            (SONNET, 10, "accepted_first_pass", 100.0),
        ):
            for local_index in range(6):
                index = offset + local_index
                receipt = make_receipt(
                    index,
                    root_id=f"cohort-{route['model']}-{route['effort']}-{local_index % 3}",
                    at=START + timedelta(minutes=index + 1),
                    evidence_grade="comparable",
                    delivery_outcome=outcome,
                    elapsed_seconds=elapsed,
                )
                apply_with_plan(
                    self.state,
                    retarget_receipt(
                        self.state,
                        receipt,
                        role="bounded_builder",
                        route=route,
                        eligible_routes=eligible,
                    ),
                )
        old_cohort = self.slot(TERRA)["route_cohort_hash"]
        rs.replace_slot(
            self.state,
            "bounded_builder",
            2,
            TERRA_MEDIUM,
            "terra-medium-build-2",
            "terra-medium-resolved-build-2",
            True,
            "verified peer reset",
            START + timedelta(hours=1),
        )
        result = rs.select_route(
            self.state,
            "bounded_builder",
            set(eligible),
            set(eligible),
            START + timedelta(hours=2),
        )
        self.assertNotEqual(result["route_cohort_hash"], old_cohort)
        self.assertEqual(result["route"], rs.route_key(TERRA))
        self.assertEqual(
            result["reason_codes"],
            ["static_fallback_insufficient_comparable_support"],
        )

    def test_indeterminate_comparable_receipts_do_not_create_support(self) -> None:
        for route, offset in ((TERRA, 0), (SONNET, 10)):
            for local_index in range(5):
                index = offset + local_index
                outcome = "accepted_first_pass" if local_index == 0 else "indeterminate"
                apply_with_plan(
                    self.state,
                    make_receipt(
                        index,
                        route=route,
                        root_id=f"{rs.route_key(route)}-root-{local_index % 3}",
                        at=START + timedelta(minutes=index + 1),
                        evidence_grade="comparable",
                        delivery_outcome=outcome,
                    ),
                )
        result = rs.select_route(
            self.state,
            "bounded_builder",
            {rs.route_key(TERRA), rs.route_key(SONNET)},
            {rs.route_key(TERRA), rs.route_key(SONNET)},
            START + timedelta(hours=1),
        )
        self.assertEqual(result["route"], rs.route_key(TERRA))
        self.assertTrue(
            all(not candidate["supported"] for candidate in result["candidates"])
        )

    def test_duplicate_fails_and_late_receipt_resolves_with_decay(self) -> None:
        receipt = make_receipt(1, at=START + timedelta(minutes=2))
        apply_with_plan(self.state, receipt)
        with self.assertRaisesRegex(rs.StateError, "duplicate"):
            rs.apply_receipt(self.state, receipt)
        old = make_receipt(2, at=START + timedelta(minutes=1))
        rs.reserve_plan(self.state, plan_from_receipt(old))
        rs.apply_receipt(self.state, old)
        self.assertFalse(
            any(plan["plan_id"] == old["plan_id"] for plan in self.state["pending"])
        )
        self.assertGreater(self.slot()["stats"]["observed_n"], 1.9)
        self.assertLess(self.slot()["stats"]["observed_n"], 2.0)

    def test_target_route_rejects_receipt_id_reuse_after_global_eviction(self) -> None:
        accepted = make_receipt(1, at=START + timedelta(minutes=1))
        accepted["receipt_id"] = "shared-receipt-id"
        apply_with_plan(self.state, accepted)
        self.state["receipt_ring"] = [
            f"other-route-receipt-{index}" for index in range(rs.RECEIPT_RING_SIZE)
        ]
        rs.validate_state(self.state)
        reused = make_receipt(
            2,
            root_id="new-root-with-reused-receipt-id",
            at=START + timedelta(minutes=2),
        )
        reused["receipt_id"] = accepted["receipt_id"]
        rs.reserve_plan(self.state, plan_from_receipt(reused))
        with self.assertRaisesRegex(rs.StateError, "target route evidence window"):
            rs.apply_receipt(self.state, reused)
        evidence = self.slot()["evidence_ring"]
        self.assertEqual(
            [
                item["root_id"]
                for item in evidence
                if item["receipt_id"] == reused["receipt_id"]
            ],
            [accepted["root_id"]],
        )

    def test_late_false_accept_survives_bounded_receipt_id_reuse(self) -> None:
        accepted = make_receipt(1, at=START + timedelta(minutes=1))
        accepted["receipt_id"] = "eventually-reused-receipt-id"
        apply_with_plan(self.state, accepted)
        for offset in range(rs.RECEIPT_RING_SIZE):
            index = 100 + offset
            apply_with_plan(
                self.state,
                make_receipt(
                    index,
                    at=START + timedelta(minutes=2 + offset),
                ),
            )
        self.assertNotIn(accepted["receipt_id"], self.state["receipt_ring"])
        self.assertNotIn(
            accepted["receipt_id"],
            {item["receipt_id"] for item in self.slot()["evidence_ring"]},
        )
        reused = make_receipt(
            1_000,
            root_id="new-episode-reusing-old-receipt-id",
            at=START + timedelta(minutes=67),
        )
        reused["receipt_id"] = accepted["receipt_id"]
        apply_with_plan(self.state, reused)
        correction = copy.deepcopy(accepted)
        correction.update(
            {
                "plan_id": "",
                "receipt_id": "late-false-accept-for-old-episode",
                "delivery_outcome": "false_accept",
                "target_outcome": "rejected",
                "observed_at": rs.isoformat(START + timedelta(minutes=68)),
                "evidence_id": "session:late-false-accept-after-id-reuse",
                "supersedes_receipt_id": accepted["receipt_id"],
            }
        )
        rs.apply_receipt(self.state, correction)
        self.assertTrue(self.slot()["quarantined"])
        self.assertTrue(
            self.slot()["quarantine_reason"].startswith("late_unlinked_false_accept:")
        )

    def test_out_of_order_receipts_have_order_independent_summaries(self) -> None:
        first = make_receipt(1, at=START + timedelta(minutes=1), elapsed_seconds=80.0)
        second = make_receipt(2, at=START + timedelta(minutes=2), elapsed_seconds=120.0)
        forward = rs.build_state(rs.load_json(rs.seed_path()), START)
        reverse = rs.build_state(rs.load_json(rs.seed_path()), START)
        for state in (forward, reverse):
            rs.reserve_plan(state, plan_from_receipt(first))
            rs.reserve_plan(state, plan_from_receipt(second))
        rs.apply_receipt(forward, first)
        rs.apply_receipt(forward, second)
        rs.apply_receipt(reverse, second)
        rs.apply_receipt(reverse, first)
        forward_stats = rs.find_slot(forward, "bounded_builder", TERRA)["stats"]
        reverse_stats = rs.find_slot(reverse, "bounded_builder", TERRA)["stats"]
        for key in rs.STATS_KEYS:
            self.assertAlmostEqual(forward_stats[key], reverse_stats[key], places=10)

    def test_new_comparison_run_is_order_independent_across_old_event_time(
        self,
    ) -> None:
        old = make_receipt(
            1,
            at=START + timedelta(minutes=10),
            evidence_grade="comparable",
            comparison_id="comparison-a",
        )
        apply_with_plan(self.state, old)
        forward = copy.deepcopy(self.state)
        reverse = copy.deepcopy(self.state)

        def new_run_receipts() -> tuple[dict[str, object], dict[str, object]]:
            early = make_receipt(
                101,
                root_id="new-run-early-root",
                at=START + timedelta(minutes=9),
                evidence_grade="comparable",
                comparison_id="comparison-b",
                elapsed_seconds=80.0,
            )
            late = make_receipt(
                102,
                root_id="new-run-late-root",
                at=START + timedelta(minutes=11),
                evidence_grade="comparable",
                comparison_id="comparison-b",
                elapsed_seconds=120.0,
            )
            return early, late

        forward_early, forward_late = new_run_receipts()
        reverse_early, reverse_late = new_run_receipts()
        apply_with_plan(forward, forward_early)
        apply_with_plan(forward, forward_late)
        apply_with_plan(reverse, reverse_late)
        apply_with_plan(reverse, reverse_early)
        forward_slot = rs.find_slot(forward, "bounded_builder", TERRA)
        reverse_slot = rs.find_slot(reverse, "bounded_builder", TERRA)
        self.assertEqual(forward_slot["comparison_id"], "comparison-b")
        self.assertEqual(reverse_slot["comparison_id"], "comparison-b")
        for key in rs.STATS_KEYS:
            self.assertAlmostEqual(
                forward_slot["stats"][key], reverse_slot["stats"][key], places=10
            )

    def test_late_old_receipt_cannot_evict_newer_evidence(self) -> None:
        old = make_receipt(100, at=START + timedelta(minutes=1))
        newer = [
            make_receipt(index, at=START + timedelta(minutes=index + 2))
            for index in range(rs.EVIDENCE_RING_SIZE)
        ]
        rs.reserve_plan(self.state, plan_from_receipt(old))
        for receipt in newer:
            rs.reserve_plan(self.state, plan_from_receipt(receipt))
            rs.apply_receipt(self.state, receipt)
        rs.apply_receipt(self.state, old)
        retained = [item["receipt_id"] for item in self.slot()["evidence_ring"]]
        self.assertNotIn(old["receipt_id"], retained)
        self.assertEqual(set(retained), {receipt["receipt_id"] for receipt in newer})

    def test_future_receipt_is_rejected_without_clock_poisoning(self) -> None:
        future = make_receipt(1, at=START + timedelta(days=365))
        rs.reserve_plan(self.state, plan_from_receipt(future))
        before = copy.deepcopy(self.slot())
        with self.assertRaisesRegex(rs.StateError, "cannot be in the future"):
            rs.apply_receipt(self.state, future, now=START + timedelta(days=1))
        self.assertEqual(self.slot(), before)

    def test_future_route_replacement_is_rejected_without_clock_poisoning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            rs.atomic_write(path, self.state)
            before = rs.load_json(path)
            args = SimpleNamespace(
                state=str(path),
                confirm_reset=True,
                route="codex:future-model:high",
                active=True,
                role="bounded_builder",
                slot=2,
                route_epoch="future-build",
                resolved_model_id="future-model-resolved",
                reason="future timestamp probe",
                at=rs.isoformat(START + timedelta(days=365)),
            )
            with patch.object(rs, "utc_now", lambda: START + timedelta(days=1)):
                with self.assertRaisesRegex(rs.StateError, "cannot be in the future"):
                    rs.cmd_replace(args)
            self.assertEqual(rs.load_json(path), before)

    def test_implicit_read_time_is_sampled_after_lock_acquisition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            rs.atomic_write(path, rs.build_state(rs.load_json(rs.seed_path()), START))
            future = rs.build_state(
                rs.load_json(rs.seed_path()),
                START + timedelta(hours=1),
            )
            clock = {"now": START}

            @contextmanager
            def delayed_lock(_path: Path, *, exclusive: bool):
                self.assertFalse(exclusive)
                rs.atomic_write(path, future)
                clock["now"] = START + timedelta(hours=1, seconds=1)
                yield

            args = SimpleNamespace(
                state=str(path),
                role="bounded_builder",
                eligible=[rs.route_key(TERRA)],
                available=[rs.route_key(TERRA)],
                at=None,
            )
            with (
                patch.object(rs, "state_lock", delayed_lock),
                patch.object(rs, "utc_now", lambda: clock["now"]),
            ):
                result = rs.cmd_select(args)
            self.assertEqual(result["route"], rs.route_key(TERRA))
            clock["now"] = START
            rs.atomic_write(path, rs.build_state(rs.load_json(rs.seed_path()), START))
            with (
                patch.object(rs, "state_lock", delayed_lock),
                patch.object(rs, "utc_now", lambda: clock["now"]),
            ):
                shown = rs.cmd_show(SimpleNamespace(state=str(path), at=None))
            self.assertEqual(shown["status"], "ok")

    def test_completed_episode_cannot_reenter_with_new_receipt_id(self) -> None:
        receipt = make_receipt(1, at=START + timedelta(minutes=1))
        apply_with_plan(self.state, receipt)
        replay = copy.deepcopy(receipt)
        replay["receipt_id"] = "receipt-1-retry"
        replay["observed_at"] = rs.isoformat(START + timedelta(minutes=2))
        replay["evidence_id"] = "session:retry"
        with self.assertRaisesRegex(rs.StateError, "already terminal"):
            rs.reserve_plan(self.state, plan_from_receipt(replay))

    def test_resolved_model_drift_requires_slot_reset(self) -> None:
        apply_with_plan(self.state, make_receipt(1, at=START + timedelta(minutes=1)))
        changed = make_receipt(2, at=START + timedelta(minutes=2))
        changed["resolved_model_id"] = "different-resolved-build"
        with self.assertRaisesRegex(rs.StateError, "resolved_model_id drift"):
            rs.reserve_plan(self.state, plan_from_receipt(changed))

    def test_late_false_accept_quarantines_without_numeric_update(self) -> None:
        first = make_receipt(0, at=START + timedelta(minutes=1))
        apply_with_plan(self.state, first)
        for index in range(1, rs.EVIDENCE_RING_SIZE + 1):
            apply_with_plan(
                self.state,
                make_receipt(index, at=START + timedelta(minutes=index + 1)),
            )
        self.assertNotIn(
            first["receipt_id"],
            [item["receipt_id"] for item in self.slot()["evidence_ring"]],
        )
        false_accepts_before = self.slot()["stats"]["observed_false_accept"]
        later = copy.deepcopy(first)
        later.update(
            {
                "plan_id": "",
                "receipt_id": "receipt-late-false-accept",
                "delivery_outcome": "false_accept",
                "target_outcome": "rejected",
                "observed_at": rs.isoformat(START + timedelta(hours=1)),
                "evidence_id": "session:late-false-accept",
                "supersedes_receipt_id": first["receipt_id"],
            }
        )
        rs.apply_receipt(self.state, later)
        self.assertTrue(self.slot()["quarantined"])
        self.assertEqual(
            self.slot()["stats"]["observed_false_accept"], false_accepts_before
        )

    def test_aged_rejection_cannot_masquerade_as_a_late_false_accept(self) -> None:
        rejected = make_receipt(
            0,
            at=START + timedelta(minutes=1),
            delivery_outcome="rejected_or_escalated",
        )
        rejected["target_outcome"] = "rejected"
        apply_with_plan(self.state, rejected)
        for index in range(1, rs.EVIDENCE_RING_SIZE + 1):
            apply_with_plan(
                self.state,
                make_receipt(index, at=START + timedelta(minutes=index + 1)),
            )
        self.assertNotIn(
            rejected["receipt_id"],
            [item["receipt_id"] for item in self.slot()["evidence_ring"]],
        )
        later = copy.deepcopy(rejected)
        later.update(
            {
                "plan_id": "",
                "receipt_id": "receipt-late-false-accept-after-rejection",
                "delivery_outcome": "false_accept",
                "observed_at": rs.isoformat(START + timedelta(hours=1)),
                "evidence_id": "session:late-false-accept-after-rejection",
                "supersedes_receipt_id": rejected["receipt_id"],
            }
        )
        with self.assertRaisesRegex(rs.StateError, "accepted route generation"):
            rs.apply_receipt(self.state, later)
        self.assertFalse(self.slot()["quarantined"])

    def test_late_false_accept_cannot_quarantine_a_different_route(self) -> None:
        first = make_receipt(1, at=START + timedelta(minutes=1), route=TERRA)
        apply_with_plan(self.state, first)
        apply_with_plan(
            self.state,
            make_receipt(2, at=START + timedelta(minutes=2), route=SONNET),
        )
        later = copy.deepcopy(first)
        later.update(
            {
                "plan_id": "",
                "receipt_id": "receipt-wrong-route-late-false-accept",
                "route": copy.deepcopy(SONNET),
                "resolved_model_id": "sonnet-resolved",
                "delivery_outcome": "false_accept",
                "target_outcome": "rejected",
                "observed_at": rs.isoformat(START + timedelta(minutes=3)),
                "evidence_id": "session:wrong-route-late-false-accept",
                "supersedes_receipt_id": first["receipt_id"],
            }
        )
        with self.assertRaisesRegex(rs.StateError, "accepted route generation"):
            rs.apply_receipt(self.state, later)
        self.assertFalse(self.slot(TERRA)["quarantined"])
        self.assertFalse(self.slot(SONNET)["quarantined"])

    def test_reservation_is_required_and_overdue_blocks_selection(self) -> None:
        receipt = make_receipt(1, at=START + timedelta(minutes=1))
        with self.assertRaisesRegex(rs.StateError, "missing reserved"):
            rs.apply_receipt(self.state, receipt)
        plan = plan_from_receipt(receipt)
        plan["terminal_due_at"] = rs.isoformat(START + timedelta(minutes=2))
        plan["terminal_budget_seconds"] = 120
        rs.reserve_plan(self.state, plan)
        result = rs.select_route(
            self.state,
            "bounded_builder",
            {rs.route_key(TERRA)},
            {rs.route_key(TERRA)},
            START + timedelta(minutes=2),
        )
        self.assertEqual(result["decision"], "abstain")
        self.assertIn("overdue_terminal_receipt", result["reason_codes"])

    def test_full_pending_table_blocks_selection(self) -> None:
        for index in range(rs.PENDING_CAPACITY):
            receipt = make_receipt(index, at=START + timedelta(minutes=index + 1))
            plan = plan_from_receipt(receipt)
            plan["terminal_due_at"] = rs.isoformat(START + timedelta(days=2))
            plan["terminal_budget_seconds"] = int(
                (
                    rs.parse_time(plan["terminal_due_at"], "terminal_due_at")
                    - rs.parse_time(plan["planned_at"], "planned_at")
                ).total_seconds()
            )
            rs.reserve_plan(self.state, plan)
        result = rs.select_route(
            self.state,
            "bounded_builder",
            {rs.route_key(TERRA)},
            {rs.route_key(TERRA)},
            START + timedelta(hours=1),
        )
        self.assertEqual(result["decision"], "abstain")
        self.assertIn("pending_capacity_exhausted", result["reason_codes"])

    def test_replay_filter_ceiling_fails_closed_before_saturation(self) -> None:
        self.state["completed_episode_filter"]["insertions"] = (
            rs.COMPLETED_FILTER_MAX_INSERTIONS
        )
        result = rs.select_route(
            self.state,
            "bounded_builder",
            {rs.route_key(TERRA)},
            {rs.route_key(TERRA)},
            START + timedelta(minutes=1),
        )
        self.assertEqual(result["decision"], "abstain")
        self.assertEqual(result["reason_codes"], ["replay_filter_capacity_exhausted"])
        receipt = make_receipt(1, at=START + timedelta(minutes=1))
        with self.assertRaisesRegex(rs.StateError, "filter capacity is exhausted"):
            rs.reserve_plan(self.state, plan_from_receipt(receipt))

    def test_decay_halves_mass_and_preserves_mean(self) -> None:
        receipt = make_receipt(
            1,
            at=START,
            evidence_grade="comparable",
            elapsed_seconds=100.0,
            lead_seconds=20.0,
        )
        apply_with_plan(self.state, receipt)
        slot = self.slot()
        decayed = rs.decayed_stats(slot, START + timedelta(days=30))
        self.assertAlmostEqual(decayed["comparable_n"], 0.5, places=8)
        self.assertAlmostEqual(
            decayed["comparable_elapsed_seconds_sum"] / decayed["comparable_n"], 100.0
        )

    def test_rings_and_parameter_shape_stay_bounded(self) -> None:
        for index in range(100):
            apply_with_plan(
                self.state,
                make_receipt(index, at=START + timedelta(minutes=index + 1)),
            )
        rs.validate_state(self.state)
        self.assertEqual(len(self.state["receipt_ring"]), rs.RECEIPT_RING_SIZE)
        self.assertEqual(len(self.slot()["evidence_ring"]), rs.EVIDENCE_RING_SIZE)
        self.assertEqual(rs.learned_parameter_count(), 504)

    def test_replace_resets_one_slot_without_growing_topology(self) -> None:
        apply_with_plan(self.state, make_receipt(1, at=START + timedelta(minutes=1)))
        replacement = {"surface": "codex", "model": "future-model", "effort": "high"}
        rs.replace_slot(
            self.state,
            "bounded_builder",
            2,
            replacement,
            "future-epoch",
            "future-model-exact-build",
            True,
            "verified identity change",
            START + timedelta(minutes=2),
        )
        self.assertEqual(len(self.state["roles"]["bounded_builder"]["slots"]), 3)
        replaced = self.state["roles"]["bounded_builder"]["slots"][2]
        self.assertEqual(replaced["route"], replacement)
        self.assertEqual(replaced["resolved_model_id"], "future-model-exact-build")
        self.assertTrue(all(value == 0.0 for value in replaced["stats"].values()))

    def test_replace_cannot_mutate_a_pending_comparison_cohort(self) -> None:
        receipt = make_receipt(
            1,
            at=START + timedelta(minutes=1),
            evidence_grade="comparable",
        )
        rs.reserve_plan(self.state, plan_from_receipt(receipt))
        before = copy.deepcopy(self.state)
        with self.assertRaisesRegex(rs.StateError, "pending comparison cohort"):
            rs.replace_slot(
                self.state,
                "bounded_builder",
                1,
                SONNET,
                "sonnet-next-build",
                "sonnet-next-resolved-build",
                True,
                "pending cohort mutation probe",
                START + timedelta(minutes=2),
            )
        self.assertEqual(self.state, before)

    def test_runtime_replacement_cannot_change_policy_baseline_route(self) -> None:
        with self.assertRaisesRegex(rs.StateError, "policy-owned"):
            rs.replace_slot(
                self.state,
                "bounded_builder",
                0,
                SONNET,
                "future-epoch",
                "sonnet-exact-build",
                True,
                "attempted baseline change",
                START + timedelta(minutes=1),
            )

    def test_atomic_file_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            rs.atomic_write(path, self.state)
            loaded = rs.load_json(path)
            self.assertEqual(loaded, self.state)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_directory_fsync_failure_reports_an_uncertain_visible_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            with patch.object(
                rs.os, "fsync", side_effect=[None, OSError("dir fsync failed")]
            ):
                with self.assertRaisesRegex(
                    rs.UncertainCommitError,
                    "replacement is visible.*do not retry blindly",
                ):
                    rs.atomic_write(path, self.state)
            self.assertEqual(rs.load_json(path), self.state)

    def test_directory_open_failure_reports_an_uncertain_visible_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            real_open = rs.os.open

            def fail_directory_open(
                target: str | bytes | Path, flags: int, *args: object, **kwargs: object
            ) -> int:
                if Path(target) == path.parent and flags == rs.os.O_RDONLY:
                    raise OSError("dir open failed")
                return real_open(target, flags, *args, **kwargs)

            with patch.object(rs.os, "open", side_effect=fail_directory_open):
                with self.assertRaisesRegex(
                    rs.UncertainCommitError,
                    "replacement is visible.*do not retry blindly",
                ):
                    rs.atomic_write(path, self.state)
            self.assertEqual(rs.load_json(path), self.state)

    def test_receipt_hashes_are_strict(self) -> None:
        receipt = make_receipt(1)
        receipt["brief_hash"] = "not-a-digest"
        with self.assertRaisesRegex(rs.StateError, "SHA-256"):
            rs.apply_receipt(self.state, receipt)

    def test_boolean_correction_count_is_rejected(self) -> None:
        receipt = make_receipt(1)
        receipt["correction_count"] = True
        with self.assertRaisesRegex(rs.StateError, "correction_count"):
            rs.validate_receipt(receipt, self.state)

    def test_ordinary_receipt_cannot_smuggle_partial_comparison_metadata(self) -> None:
        receipt = make_receipt(1)
        receipt["comparison_id"] = "partial-comparison"
        with self.assertRaisesRegex(rs.StateError, "ordinary evidence"):
            rs.validate_receipt(receipt, self.state)

    def test_comparison_run_epoch_overflow_fails_closed(self) -> None:
        role = self.state["roles"]["bounded_builder"]
        role["active_comparison_id"] = "prior-comparison"
        role["active_comparison_contract_hash"] = digest("prior-contract")
        role["active_route_cohort_hash"] = digest("prior-cohort")
        role["active_comparison_signature"] = rs.comparison_signature_from_parts(
            role["active_comparison_id"],
            role["active_comparison_contract_hash"],
            role["active_route_cohort_hash"],
        )
        role["active_comparison_run_epoch"] = rs.COMPARISON_RUN_EPOCH_MAX
        receipt = make_receipt(
            1,
            evidence_grade="comparable",
            comparison_id="new-comparison",
        )
        with self.assertRaisesRegex(rs.StateError, "run epoch capacity is exhausted"):
            rs.reserve_plan(self.state, plan_from_receipt(receipt))

    def test_validator_rejects_duplicate_active_routes(self) -> None:
        broken = copy.deepcopy(self.state)
        broken["roles"]["bounded_builder"]["slots"][1]["route"] = copy.deepcopy(TERRA)
        with self.assertRaisesRegex(rs.StateError, "duplicate active route"):
            rs.validate_state(broken)

    def test_validator_rejects_boolean_baseline_slot(self) -> None:
        broken = copy.deepcopy(self.state)
        broken["roles"]["bounded_builder"]["baseline_slot"] = True
        with self.assertRaisesRegex(rs.StateError, "invalid baseline slot"):
            rs.validate_state(broken)

    def test_validator_binds_policy_owned_baselines_to_seed(self) -> None:
        broken_epoch = copy.deepcopy(self.state)
        broken_epoch["policy_epoch"] = "attacker-policy"
        with self.assertRaisesRegex(rs.StateError, "policy epoch"):
            rs.validate_state(broken_epoch)

        broken_slot = copy.deepcopy(self.state)
        broken_slot["roles"]["bounded_builder"]["baseline_slot"] = 1
        with self.assertRaisesRegex(rs.StateError, "policy baseline slot"):
            rs.validate_state(broken_slot)

        broken_route = copy.deepcopy(self.state)
        broken_route["roles"]["bounded_builder"]["slots"][0]["route"] = {
            "surface": "codex",
            "model": "arbitrary-unreviewed-model",
            "effort": "high",
        }
        with self.assertRaisesRegex(rs.StateError, "policy baseline route"):
            rs.validate_state(broken_route)

    def test_validator_rejects_duplicate_receipt_ids_inside_slot_evidence(self) -> None:
        apply_with_plan(self.state, make_receipt(1))
        self.slot()["evidence_ring"].append(
            copy.deepcopy(self.slot()["evidence_ring"][0])
        )
        with self.assertRaisesRegex(rs.StateError, "duplicate receipt_id"):
            rs.validate_state(self.state)

    def test_validator_rejects_route_key_delimiters_inside_components(self) -> None:
        broken = copy.deepcopy(self.state)
        broken["roles"]["bounded_builder"]["slots"][0]["route"]["model"] = (
            "invalid:model"
        )
        with self.assertRaisesRegex(rs.StateError, "route-key ':' delimiter"):
            rs.validate_state(broken)

    def test_all_zero_hash_and_model_placeholder_fail_closed(self) -> None:
        receipt = make_receipt(1)
        receipt["brief_hash"] = "0" * 64
        with self.assertRaisesRegex(rs.StateError, "all-zero"):
            rs.reserve_plan(self.state, plan_from_receipt(receipt))
        receipt = make_receipt(2)
        receipt["resolved_model_id"] = "REPLACE_WITH_EXACT_RESOLVED_MODEL_ID"
        with self.assertRaisesRegex(rs.StateError, "verified concrete"):
            rs.reserve_plan(self.state, plan_from_receipt(receipt))

    def test_identity_and_evidence_template_sentinels_fail_closed(self) -> None:
        receipt = make_receipt(1)
        receipt["plan_id"] = "root-id:episode-id:plan-v4"
        receipt["root_id"] = "root-id"
        receipt["episode_id"] = "episode-id"
        with self.assertRaisesRegex(rs.StateError, "template sentinel"):
            rs.reserve_plan(self.state, plan_from_receipt(receipt))
        receipt = make_receipt(2)
        receipt["evidence_id"] = "session-or-ledger-evidence-handle"
        with self.assertRaisesRegex(rs.StateError, "template sentinel"):
            rs.validate_receipt(receipt, self.state)

    def test_nul_identity_delimiters_fail_closed(self) -> None:
        receipt = make_receipt(1)
        receipt["root_id"] = "ambiguous\0root"
        with self.assertRaisesRegex(rs.StateError, "NUL identity delimiter"):
            rs.reserve_plan(self.state, plan_from_receipt(receipt))

    def test_already_overdue_plan_cannot_be_reserved(self) -> None:
        receipt = make_receipt(1, at=START + timedelta(minutes=1))
        plan = plan_from_receipt(receipt)
        with self.assertRaisesRegex(rs.StateError, "already past"):
            rs.reserve_plan(self.state, plan, now=START + timedelta(hours=2))

    def test_json_parser_rejects_duplicate_keys_and_nonfinite_numbers(self) -> None:
        with self.assertRaisesRegex(rs.StateError, "duplicate JSON key"):
            rs.parse_json_text('{"a": 1, "a": 2}', "test")
        with self.assertRaisesRegex(rs.StateError, "numeric constant"):
            rs.parse_json_text('{"a": NaN}', "test")


if __name__ == "__main__":
    unittest.main(verbosity=2)
