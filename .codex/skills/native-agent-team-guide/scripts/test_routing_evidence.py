#!/usr/bin/env python3
"""Black-box tests for the routing-evidence CLI.

The tests intentionally construct every input in a temporary directory.  They
exercise the public CLI contract and do not import or inspect its
implementation.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(os.environ.get("ROUTING_EVIDENCE_SCRIPT", str(Path(__file__).with_name("routing_evidence.py"))))


def attempt(
    attempt_id,
    model="luna",
    effort="medium",
    fork="none",
    outcome="accepted",
    worker_usd=1.0,
    lead_usd=0.0,
    runtime_usd=0.0,
    cost_source="receipt",
    failure_kind="none",
):
    return {
        "attempt_id": attempt_id,
        "model": model,
        "effort": effort,
        "fork": fork,
        "outcome": outcome,
        "worker_usd": worker_usd,
        "lead_usd": lead_usd,
        "runtime_usd": runtime_usd,
        "cost_source": cost_source,
        "failure_kind": failure_kind,
    }


def row(
    task_id,
    *,
    revision=1,
    outcome="accepted",
    origin="production",
    task_class="detection",
    comparison_key="control",
    brief_style="direct",
    topology="flat",
    risk="low",
    verifier="deterministic",
    attempts=None,
    recorded_at="2026-09-13T00:00:00+00:00",
):
    if attempts is None:
        attempts = [attempt(f"{task_id}-a")]
    return {
        "schema_version": 1,
        "task_id": task_id,
        "revision": revision,
        "recorded_at": recorded_at,
        "origin": origin,
        "task_class": task_class,
        "comparison_key": comparison_key,
        "brief_style": brief_style,
        "topology": topology,
        "risk": risk,
        "verifier": verifier,
        "outcome": outcome,
        "evidence": [f"evidence/{task_id}.json"],
        "attempts": attempts,
    }


class RoutingEvidenceCliTests(unittest.TestCase):
    maxDiff = None

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, args)],
            text=True,
            capture_output=True,
            check=False,
        )

    def write_jsonl(self, path, rows, *, raw_lines=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        if raw_lines is None:
            raw_lines = [json.dumps(item, sort_keys=True) for item in rows]
        path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    def summarize(self, inputs, *, origin=None, task_class=None):
        output_dir = self.tmp / f"outputs-{len(list(self.tmp.glob('outputs-*')))}"
        args = ["summarize", *inputs, "--output-dir", output_dir]
        if origin is not None:
            args.extend(["--origin", origin])
        if task_class is not None:
            args.extend(["--task-class", task_class])
        result = self.run_cli(*args)
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
        self.assertTrue((output_dir / "summary.json").is_file())
        self.assertTrue((output_dir / "summary.md").is_file())
        self.assertTrue((output_dir / "summary.md").read_text(encoding="utf-8").strip())
        return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    @staticmethod
    def groups(summary):
        groups = summary["groups"]
        if isinstance(groups, list):
            return groups
        if isinstance(groups, dict):
            result = []
            for key, value in groups.items():
                if isinstance(value, dict):
                    record = dict(value)
                    record.setdefault("_group_key", key)
                    result.append(record)
            return result
        raise AssertionError(f"groups must be a list or object, got {type(groups)!r}")

    @staticmethod
    def total_metric(summary, name):
        return sum(group.get(name, 0) for group in RoutingEvidenceCliTests.groups(summary))

    def setUp(self):
        self._tmp_context = tempfile.TemporaryDirectory(prefix="routing-evidence-")
        self.tmp = Path(self._tmp_context.name)

    def tearDown(self):
        self._tmp_context.cleanup()

    def test_outcomes_costs_and_full_repair_route_are_summarized(self):
        rows = [
            row("accepted", attempts=[attempt("accepted-a", worker_usd=1.5)]),
            row(
                "failed",
                outcome="failed",
                attempts=[attempt("failed-a", outcome="failed", worker_usd=0.5, failure_kind="implementation")],
            ),
            row(
                "pending",
                outcome="pending",
                attempts=[attempt("pending-a", outcome="escalated", worker_usd=None, cost_source="not_recorded", failure_kind="unknown")],
            ),
            row(
                "invalidated",
                outcome="invalidated",
                attempts=[attempt("invalidated-a", outcome="escalated", worker_usd=0.25, failure_kind="verifier")],
            ),
            row(
                "repair-chain",
                attempts=[
                    attempt("repair-1", model="luna", effort="low", fork="all", outcome="rework", worker_usd=1.0, failure_kind="reasoning"),
                    attempt("repair-2", model="sol", effort="high", fork="none", outcome="accepted", worker_usd=2.0),
                ],
            ),
        ]
        source = self.tmp / "outcomes.jsonl"
        self.write_jsonl(source, rows)
        validation = self.run_cli("validate", source)
        self.assertEqual(validation.returncode, 0, msg=validation.stderr or validation.stdout)

        summary = self.summarize([source])
        self.assertEqual(summary["task_count"], 5)
        self.assertEqual(self.total_metric(summary, "task_count"), 5)
        self.assertEqual(self.total_metric(summary, "accepted_count"), 2)
        self.assertEqual(self.total_metric(summary, "first_pass_count"), 1)
        self.assertEqual(self.total_metric(summary, "failed_count"), 1)
        self.assertEqual(self.total_metric(summary, "pending_count"), 1)
        self.assertEqual(self.total_metric(summary, "invalidated_count"), 1)
        self.assertAlmostEqual(self.total_metric(summary, "known_cost_usd"), 5.25)
        self.assertTrue(
            any(group["cost_per_accepted_usd"] is None for group in self.groups(summary))
        )

        # The repair group must retain the complete ordered route, including
        # effort and fork for each attempt.  This catches crediting the task
        # only to its final model.
        route_groups = [
            group
            for group in self.groups(summary)
            if "luna" in json.dumps(group) and "sol" in json.dumps(group)
        ]
        self.assertTrue(route_groups, msg=json.dumps(summary, indent=2, sort_keys=True))
        route_text = json.dumps(route_groups[0], sort_keys=True)
        self.assertLess(route_text.index("luna"), route_text.index("sol"))
        for route_part in ("low", "all", "high", "none"):
            self.assertIn(route_part, route_text)

    def test_known_cost_is_required_for_cost_per_accepted_and_failed_cost_is_included(self):
        known_source = self.tmp / "known.jsonl"
        self.write_jsonl(
            known_source,
            [
                row("known-accepted", attempts=[attempt("known-accepted-a", worker_usd=2.0)]),
                row(
                    "known-failed",
                    outcome="failed",
                    attempts=[attempt("known-failed-a", outcome="failed", worker_usd=3.0, failure_kind="brief")],
                ),
            ],
        )
        summary = self.summarize([known_source])
        self.assertEqual(summary["task_count"], 2)
        self.assertAlmostEqual(self.total_metric(summary, "known_cost_usd"), 5.0)
        self.assertEqual(self.total_metric(summary, "complete_cost_tasks"), 2)
        for group in self.groups(summary):
            self.assertAlmostEqual(group["cost_per_accepted_usd"], 5.0)

        unknown_source = self.tmp / "unknown.jsonl"
        self.write_jsonl(
            unknown_source,
            [
                row("unknown-accepted", attempts=[attempt("unknown-accepted-a", worker_usd=None, cost_source="pending")]),
                row(
                    "unknown-failed",
                    outcome="failed",
                    attempts=[attempt("unknown-failed-a", outcome="failed", worker_usd=3.0, failure_kind="environment")],
                ),
            ],
        )
        summary = self.summarize([unknown_source])
        self.assertAlmostEqual(self.total_metric(summary, "known_cost_usd"), 3.0)
        self.assertEqual(self.total_metric(summary, "complete_cost_tasks"), 1)
        for group in self.groups(summary):
            self.assertIsNone(group["cost_per_accepted_usd"])

    def test_identical_duplicates_dedup_and_highest_revision_supersedes_old(self):
        rev_failed = row(
            "revision-task",
            revision=1,
            outcome="failed",
            attempts=[attempt("revision-a", outcome="failed", worker_usd=1.0, failure_kind="implementation")],
        )
        rev_accepted = row(
            "revision-task",
            revision=2,
            attempts=[attempt("revision-a", worker_usd=2.0)],
        )
        invalidated_v1 = row("later-invalidation", revision=1, attempts=[attempt("invalidate-a", worker_usd=4.0)])
        invalidated_v2 = row(
            "later-invalidation",
            revision=2,
            outcome="invalidated",
            attempts=[attempt("invalidate-a", outcome="escalated", worker_usd=0.2, failure_kind="verifier")],
        )
        source = self.tmp / "revisions.jsonl"
        self.write_jsonl(source, [rev_failed, rev_accepted, rev_accepted, invalidated_v1, invalidated_v2])
        validation = self.run_cli("validate", source)
        self.assertEqual(validation.returncode, 0, msg=validation.stderr or validation.stdout)
        summary = self.summarize([source])
        self.assertEqual(summary["task_count"], 2)
        self.assertEqual(self.total_metric(summary, "accepted_count"), 1)
        self.assertEqual(self.total_metric(summary, "invalidated_count"), 1)
        self.assertAlmostEqual(self.total_metric(summary, "known_cost_usd"), 2.2)

    def test_directory_discovery_origin_filter_and_explicit_duplicate_input(self):
        production = self.tmp / "production" / "delegation-outcomes.jsonl"
        benchmark = self.tmp / "nested" / "benchmark" / "delegation-outcomes.jsonl"
        ignored = self.tmp / "nested" / "benchmark" / "other.jsonl"
        self.write_jsonl(production, [row("prod-task", task_class="detection")])
        self.write_jsonl(benchmark, [row("bench-task", origin="benchmark", task_class="calibration")])
        self.write_jsonl(ignored, [row("ignored-task")])

        default_summary = self.summarize([self.tmp])
        self.assertEqual(default_summary["task_count"], 1)
        self.assertEqual(self.total_metric(default_summary, "accepted_count"), 1)

        all_summary = self.summarize([self.tmp], origin="all")
        self.assertEqual(all_summary["task_count"], 2)
        self.assertEqual(self.total_metric(all_summary, "accepted_count"), 2)

        filtered = self.summarize([self.tmp], origin="all", task_class="calibration")
        self.assertEqual(filtered["task_count"], 1)
        self.assertEqual(self.total_metric(filtered, "accepted_count"), 1)

        # Passing a directory and one of its discovered files must not bill it
        # twice after path resolution and deduplication.
        deduped = self.summarize([self.tmp, production])
        self.assertEqual(deduped["task_count"], 1)

    def test_conflicts_nonfinite_costs_and_cross_task_attempt_ids_are_rejected(self):
        conflict = self.tmp / "conflict.jsonl"
        first = row("same-task", comparison_key="a", attempts=[attempt("same-task-a")])
        second = row("same-task", comparison_key="b", attempts=[attempt("same-task-a")])
        self.write_jsonl(conflict, [first, second])
        result = self.run_cli("validate", conflict)
        self.assertNotEqual(result.returncode, 0)

        cross_task = self.tmp / "cross-task.jsonl"
        self.write_jsonl(
            cross_task,
            [row("task-a", attempts=[attempt("shared-attempt")]), row("task-b", attempts=[attempt("shared-attempt")])],
        )
        result = self.run_cli("validate", cross_task)
        self.assertNotEqual(result.returncode, 0)

        for label, literal in (("nan", "NaN"), ("positive-inf", "Infinity"), ("negative-inf", "-Infinity"), ("boolean", "true")):
            path = self.tmp / f"{label}.jsonl"
            invalid = row("invalid-cost", attempts=[attempt("invalid-cost-a")])
            encoded = json.dumps(invalid).replace("1.0", literal, 1)
            self.write_jsonl(path, [], raw_lines=[encoded])
            result = self.run_cli("validate", path)
            self.assertNotEqual(result.returncode, 0, msg=f"accepted invalid cost {literal}")


if __name__ == "__main__":
    unittest.main()
