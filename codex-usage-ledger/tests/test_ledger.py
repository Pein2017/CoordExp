from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codex_usage_ledger.parser import parse_rollout
from codex_usage_ledger.pricing import load_rates
from codex_usage_ledger.report import enrich_record, summarize, summarize_totals
from codex_usage_ledger.cli import filter_records, main
from codex_usage_ledger.parser import SessionRecord
from codex_usage_ledger.attempts import (
    annotate_attempts,
    load_outcomes,
    summarize_attempts,
    summarize_route_pairs,
)


def _line(timestamp: str, kind: str, payload: dict) -> str:
    return json.dumps({"timestamp": timestamp, "type": kind, "payload": payload}) + "\n"


class LedgerTests(unittest.TestCase):
    def test_scope_filters_exact_thread_session_and_root_subtree(self) -> None:
        root = SessionRecord(
            file="root",
            file_size_bytes=1,
            thread_id="root-a",
            session_id="session-a",
            thread_source="user",
        )
        child = SessionRecord(
            file="child",
            file_size_bytes=1,
            thread_id="child-a",
            session_id="session-a",
            parent_thread_id="root-a",
            thread_source="subagent",
        )
        grandchild = SessionRecord(
            file="grandchild",
            file_size_bytes=1,
            thread_id="grandchild-a",
            session_id="session-a",
            parent_thread_id="child-a",
            thread_source="subagent",
        )
        other = SessionRecord(
            file="other",
            file_size_bytes=1,
            thread_id="other",
            session_id="session-b",
            thread_source="user",
        )
        records = [root, child, grandchild, other]

        exact, exact_scope = filter_records(records, False, thread_id="child-a")
        self.assertEqual([record.thread_id for record in exact], ["child-a"])
        self.assertEqual(exact_scope["scope_filter"], "thread")

        session, session_scope = filter_records(records, False, session_id="session-a")
        self.assertEqual(
            [record.thread_id for record in session], ["child-a", "grandchild-a"]
        )
        self.assertEqual(session_scope["scope_filter"], "session")

        subtree, subtree_scope = filter_records(records, True, root_thread_id="root-a")
        self.assertEqual(
            [record.thread_id for record in subtree],
            ["root-a", "child-a", "grandchild-a"],
        )
        self.assertEqual(subtree_scope["scope_records"], 3)

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            filter_records(
                records,
                False,
                thread_id="child-a",
                root_thread_id="root-a",
            )

        with self.assertRaisesRegex(ValueError, "did not match"):
            filter_records(records, False, thread_id="missing")

    def test_forked_rollout_ignores_parent_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollout-2026-08-06T00-00-00-fork.jsonl"
            parent_usage = {
                "input_tokens": 1_000_000,
                "cached_input_tokens": 900_000,
                "cache_write_input_tokens": 0,
                "output_tokens": 10_000,
                "reasoning_output_tokens": 4_000,
                "total_tokens": 1_010_000,
            }
            child_usage = {
                "input_tokens": 1_000_300,
                "cached_input_tokens": 900_200,
                "cache_write_input_tokens": 0,
                "output_tokens": 10_030,
                "reasoning_output_tokens": 4_008,
                "total_tokens": 1_010_330,
            }
            path.write_text(
                "".join(
                    [
                        _line(
                            "2026-08-06T00:00:00Z",
                            "session_meta",
                            {
                                "id": "child-1",
                                "session_id": "root-1",
                                "parent_thread_id": "root-1",
                                "thread_source": "subagent",
                                "agent_path": "/root/forked-worker",
                                "model_provider": "openai",
                            },
                        ),
                        _line(
                            "2026-08-06T00:00:00Z",
                            "event_msg",
                            {
                                "type": "token_count",
                                "info": {
                                    "total_token_usage": parent_usage,
                                    "last_token_usage": parent_usage,
                                },
                            },
                        ),
                        _line(
                            "2026-08-06T00:00:00Z",
                            "session_meta",
                            {
                                "id": "root-1",
                                "session_id": "root-1",
                                "thread_source": "user",
                            },
                        ),
                        _line(
                            "2026-08-06T00:00:01Z",
                            "event_msg",
                            {"type": "task_started", "turn_id": "turn-1"},
                        ),
                        _line(
                            "2026-08-06T00:00:01Z",
                            "turn_context",
                            {
                                "turn_id": "turn-1",
                                "model": "gpt-5.6-luna",
                                "effort": "medium",
                            },
                        ),
                        _line(
                            "2026-08-06T00:00:02Z",
                            "event_msg",
                            {
                                "type": "token_count",
                                "info": {
                                    "total_token_usage": child_usage,
                                    "last_token_usage": child_usage,
                                },
                            },
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            record = parse_rollout(path)
            self.assertEqual(record.thread_id, "child-1")
            self.assertEqual(record.thread_source, "subagent")
            self.assertEqual(record.scoped_usage().total_tokens, 330)
            self.assertEqual(record.route_usage()[0]["usage"]["total_tokens"], 330)

    def test_parse_subagent_usage_and_route_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollout-2026-08-06T00-00-00-thread.jsonl"
            usage_one = {
                "input_tokens": 100,
                "cached_input_tokens": 20,
                "cache_write_input_tokens": 0,
                "output_tokens": 10,
                "reasoning_output_tokens": 4,
                "total_tokens": 110,
            }
            usage_two = {
                "input_tokens": 300,
                "cached_input_tokens": 80,
                "cache_write_input_tokens": 0,
                "output_tokens": 30,
                "reasoning_output_tokens": 8,
                "total_tokens": 330,
            }
            path.write_text(
                "".join(
                    [
                        _line(
                            "2026-08-06T00:00:00Z",
                            "session_meta",
                            {
                                "id": "child-1",
                                "session_id": "root-1",
                                "parent_thread_id": "root-1",
                                "thread_source": "subagent",
                                "agent_role": "implementation_worker",
                                "agent_path": "/root/worker",
                                "model_provider": "openai",
                                "cli_version": "0.146.0",
                                "timestamp": "2026-08-06T00:00:00Z",
                                "multi_agent_version": "v2",
                            },
                        ),
                        _line(
                            "2026-08-06T00:00:01Z",
                            "turn_context",
                            {
                                "turn_id": "turn-1",
                                "model": "gpt-5.6-luna",
                                "effort": "medium",
                                "multi_agent_version": "v2",
                            },
                        ),
                        _line(
                            "2026-08-06T00:00:01Z",
                            "event_msg",
                            {
                                "type": "task_started",
                                "turn_id": "turn-1",
                                "started_at": 1,
                            },
                        ),
                        _line(
                            "2026-08-06T00:00:02Z",
                            "event_msg",
                            {
                                "type": "token_count",
                                "info": {
                                    "total_token_usage": usage_one,
                                    "last_token_usage": usage_one,
                                },
                            },
                        ),
                        _line(
                            "2026-08-06T00:00:03Z",
                            "event_msg",
                            {
                                "type": "token_count",
                                "info": {
                                    "total_token_usage": usage_two,
                                    "last_token_usage": {
                                        **usage_two,
                                        "input_tokens": 200,
                                        "cached_input_tokens": 60,
                                        "output_tokens": 20,
                                        "reasoning_output_tokens": 4,
                                        "total_tokens": 220,
                                    },
                                },
                            },
                        ),
                        _line(
                            "2026-08-06T00:00:04Z",
                            "event_msg",
                            {
                                "type": "task_complete",
                                "turn_id": "turn-1",
                                "started_at": 1,
                                "completed_at": 4,
                                "duration_ms": 3000,
                            },
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            record = parse_rollout(path)
            self.assertTrue(record.is_subagent)
            self.assertEqual(record.task_label, "worker")
            self.assertEqual(record.model, "gpt-5.6-luna")
            self.assertEqual(record.effort, "medium")
            assert record.latest_total_usage is not None
            assert record.latest_last_usage is not None
            self.assertEqual(record.latest_total_usage.total_tokens, 330)
            self.assertEqual(record.latest_last_usage.total_tokens, 220)
            self.assertEqual(record.status, "completed")

    def test_cost_and_group_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            price_path = Path(tmp) / "prices.toml"
            price_path.write_text(
                """
[[rates]]
provider = "openai"
model = "gpt-5.6-luna"
currency = "USD"
input_per_million = 1.0
cached_input_per_million = 0.5
output_per_million = 2.0
reasoning_is_in_output = true
""",
                encoding="utf-8",
            )
            # Build a minimal record through the public parser shape without relying on a fixture file.
            from codex_usage_ledger.model import Usage, TurnContext

            record = SessionRecord(
                file="fixture",
                file_size_bytes=1,
                thread_id="child-1",
                parent_thread_id="root-1",
                thread_source="subagent",
                agent_role="implementation_worker",
                agent_path="/root/worker",
                model_provider="openai",
                latest_total_usage=Usage(
                    input_tokens=300,
                    cached_input_tokens=80,
                    output_tokens=30,
                    reasoning_output_tokens=8,
                    total_tokens=330,
                ),
                turn_contexts=[
                    TurnContext("now", "turn-1", "gpt-5.6-luna", "medium", "v2")
                ],
            )
            item = enrich_record(record, load_rates(price_path))
            self.assertEqual(item["pricing"]["status"], "ok")
            self.assertAlmostEqual(item["pricing"]["amount"], 0.00032)
            groups = summarize([item])
            self.assertEqual(groups[0]["sessions"], 1)
            self.assertAlmostEqual(groups[0]["total_estimated_cost"], 0.00032)
            totals = summarize_totals([item])
            self.assertEqual(totals["fully_priced_sessions"], 1)
            self.assertAlmostEqual(totals["known_route_cost"], 0.00032)
            self.assertEqual(totals["unpriced_tokens"], 0)

    def test_multi_model_usage_is_segmented(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollout-2026-08-06T00-00-00-multi.jsonl"
            meta = {
                "id": "child-2",
                "session_id": "root-2",
                "parent_thread_id": "root-2",
                "thread_source": "subagent",
                "agent_path": "/root/multi",
                "model_provider": "openai",
            }
            first = {
                "input_tokens": 100,
                "cached_input_tokens": 0,
                "output_tokens": 10,
                "reasoning_output_tokens": 0,
                "total_tokens": 110,
            }
            second = {
                "input_tokens": 250,
                "cached_input_tokens": 0,
                "output_tokens": 30,
                "reasoning_output_tokens": 0,
                "total_tokens": 280,
            }
            path.write_text(
                "".join(
                    [
                        _line("2026-08-06T00:00:00Z", "session_meta", meta),
                        _line(
                            "2026-08-06T00:00:01Z",
                            "turn_context",
                            {"turn_id": "turn-a", "model": "luna", "effort": "medium"},
                        ),
                        _line(
                            "2026-08-06T00:00:02Z",
                            "event_msg",
                            {
                                "type": "token_count",
                                "info": {
                                    "total_token_usage": first,
                                    "last_token_usage": first,
                                },
                            },
                        ),
                        _line(
                            "2026-08-06T00:00:03Z",
                            "turn_context",
                            {"turn_id": "turn-b", "model": "sol", "effort": "high"},
                        ),
                        _line(
                            "2026-08-06T00:00:04Z",
                            "event_msg",
                            {
                                "type": "token_count",
                                "info": {
                                    "total_token_usage": second,
                                    "last_token_usage": second,
                                },
                            },
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            record = parse_rollout(path)
            routes = {item["model"]: item["usage"] for item in record.route_usage()}
            self.assertEqual(record.model, None)
            self.assertEqual(routes["luna"]["total_tokens"], 110)
            self.assertEqual(routes["sol"]["total_tokens"], 170)

            price_path = Path(tmp) / "multi-prices.toml"
            price_path.write_text(
                """
[[rates]]
provider = "openai"
model = "luna"
input_per_million = 1.0
output_per_million = 2.0

[[rates]]
provider = "openai"
model = "sol"
input_per_million = 1.0
output_per_million = 2.0
""",
                encoding="utf-8",
            )
            priced = enrich_record(record, load_rates(price_path))
            self.assertEqual(priced["pricing"]["status"], "ok")
            self.assertAlmostEqual(priced["pricing"]["amount"], 0.00031)

    def test_attempt_identity_and_followup_aware_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollout-2026-08-06T00-00-00-attempt.jsonl"
            usage = {
                "input_tokens": 100,
                "cached_input_tokens": 0,
                "output_tokens": 10,
                "reasoning_output_tokens": 0,
                "total_tokens": 110,
            }
            path.write_text(
                "".join(
                    [
                        _line(
                            "2026-08-06T00:00:00Z",
                            "session_meta",
                            {
                                "id": "child-attempt",
                                "parent_thread_id": "parent-1",
                                "thread_source": "subagent",
                                "agent_path": "/root/worker",
                                "agent_role": "implementation_worker",
                                "model_provider": "openai",
                            },
                        ),
                        _line(
                            "2026-08-06T00:00:01Z",
                            "response_item",
                            {
                                "type": "function_call",
                                "call_id": "call-start",
                                "name": "spawn_agent",
                            },
                        ),
                        _line(
                            "2026-08-06T00:00:01Z",
                            "event_msg",
                            {
                                "type": "sub_agent_activity",
                                "event_id": "call-start",
                                "agent_thread_id": "child-attempt",
                                "agent_path": "/root/worker",
                                "kind": "started",
                                "occurred_at_ms": 1,
                            },
                        ),
                        _line(
                            "2026-08-06T00:00:02Z",
                            "event_msg",
                            {"type": "task_started", "turn_id": "turn-1"},
                        ),
                        _line(
                            "2026-08-06T00:00:02Z",
                            "turn_context",
                            {
                                "turn_id": "turn-1",
                                "model": "gpt-5.6-luna",
                                "effort": "medium",
                            },
                        ),
                        _line(
                            "2026-08-06T00:00:03Z",
                            "event_msg",
                            {
                                "type": "token_count",
                                "info": {
                                    "total_token_usage": usage,
                                    "last_token_usage": usage,
                                },
                            },
                        ),
                        _line(
                            "2026-08-06T00:00:04Z",
                            "event_msg",
                            {"type": "task_complete", "turn_id": "turn-1"},
                        ),
                        _line(
                            "2026-08-06T00:00:05Z",
                            "response_item",
                            {
                                "type": "function_call",
                                "call_id": "call-follow",
                                "name": "followup_task",
                            },
                        ),
                        _line(
                            "2026-08-06T00:00:05Z",
                            "event_msg",
                            {
                                "type": "sub_agent_activity",
                                "event_id": "call-follow",
                                "agent_thread_id": "child-attempt",
                                "agent_path": "/root/worker",
                                "kind": "interacted",
                                "occurred_at_ms": 5,
                            },
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            record = parse_rollout(path)
            self.assertEqual(
                record.sub_agent_activity_events[0]["event_id"], "call-start"
            )
            self.assertEqual(record.agent_tool_calls[1]["name"], "followup_task")
            price_path = Path(tmp) / "prices.toml"
            price_path.write_text(
                """
[[rates]]
provider = "openai"
model = "gpt-5.6-luna"
input_per_million = 1.0
output_per_million = 2.0
""",
                encoding="utf-8",
            )
            item = enrich_record(record, load_rates(price_path))
            followup_proxy = annotate_attempts(
                [item], [record], policy="followup_aware"
            )[0]
            self.assertEqual(followup_proxy["attempt"]["attempt_id"], "call-start")
            self.assertEqual(followup_proxy["attempt"]["disposition"], "rework")
            self.assertEqual(
                summarize_attempts([followup_proxy], "followup_aware")[
                    "cost_per_accepted_task"
                ],
                None,
            )
            completed_proxy = annotate_attempts([item], [record], policy="completed")[0]
            self.assertEqual(completed_proxy["attempt"]["disposition"], "accepted")
            self.assertAlmostEqual(
                summarize_attempts([completed_proxy], "completed")[
                    "cost_per_accepted_task"
                ],
                0.00012,
            )

    def test_explicit_outcome_overrides_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outcomes.jsonl"
            path.write_text(
                '{"attempt_id":"call-start","disposition":"accepted","note":"verified"}\n',
                encoding="utf-8",
            )
            outcomes = load_outcomes(path)
            self.assertEqual(outcomes["call-start"]["disposition"], "accepted")

    def test_route_pair_decision_distributions(self) -> None:
        def item(
            started_at: str,
            ended_at: str,
            total_tokens: int,
            cached_tokens: int,
            cost: float,
        ) -> dict:
            return {
                "model": "gpt-5.6-sol",
                "effort": "medium",
                "started_at": started_at,
                "ended_at": ended_at,
                "measured_usage": {
                    "input_tokens": total_tokens - 10,
                    "cached_input_tokens": cached_tokens,
                    "cache_write_input_tokens": 0,
                    "output_tokens": 10,
                    "reasoning_output_tokens": 4,
                    "total_tokens": total_tokens,
                },
                "pricing": {
                    "status": "ok",
                    "amount": cost,
                    "segments": [
                        {
                            "pricing": {
                                "status": "ok",
                                "amount": cost,
                                "billable_tokens": {
                                    "uncached_input_tokens": total_tokens
                                    - cached_tokens
                                    - 10,
                                    "cached_input_tokens": cached_tokens,
                                    "cache_write_input_tokens": 0,
                                    "output_tokens": 10,
                                    "reasoning_output_tokens": 0,
                                },
                            }
                        }
                    ],
                },
                "attempt": {"disposition": "accepted"},
            }

        route = summarize_route_pairs(
            [
                item(
                    "2026-08-06T00:00:00Z",
                    "2026-08-06T00:00:10Z",
                    100,
                    40,
                    1.0,
                ),
                item(
                    "2026-08-06T00:01:00Z",
                    "2026-08-06T00:01:30Z",
                    300,
                    100,
                    3.0,
                ),
                {
                    "model": "gpt-5.6-sol",
                    "effort": "medium",
                    "pricing": {"status": "partial_or_missing", "amount": None},
                    "attempt": {"disposition": "unknown"},
                },
            ]
        )[0]

        self.assertEqual(
            route["rollout_wall_seconds"],
            {
                "observations": 2,
                "total": 40.0,
                "mean": 20.0,
                "median": 20.0,
                "p90": 30.0,
            },
        )
        self.assertEqual(route["measured_tokens"]["total_tokens"]["total"], 400)
        self.assertEqual(route["billable_tokens"]["cached_input_tokens"]["median"], 70)
        self.assertEqual(route["estimated_cost"]["p90"], 3.0)
        self.assertEqual(route["estimated_cost"]["observations"], 2)
        self.assertEqual(route["unpriced_attempts"], 1)

    def test_compact_summary_price_receipt_and_outcomes_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sessions = root / "sessions"
            sessions.mkdir()
            rollout = sessions / "rollout-2026-08-06T00-00-00-child.jsonl"
            usage = {
                "input_tokens": 100,
                "cached_input_tokens": 40,
                "cache_write_input_tokens": 0,
                "output_tokens": 10,
                "reasoning_output_tokens": 4,
                "total_tokens": 110,
            }
            rollout.write_text(
                "".join(
                    [
                        _line(
                            "2026-08-06T00:00:00Z",
                            "session_meta",
                            {
                                "id": "child-summary",
                                "parent_thread_id": "parent-summary",
                                "thread_source": "subagent",
                                "agent_path": "/root/summary-worker",
                                "agent_role": "reviewer",
                                "model_provider": "openai",
                            },
                        ),
                        _line(
                            "2026-08-06T00:00:01Z",
                            "event_msg",
                            {"type": "task_started", "turn_id": "turn-1"},
                        ),
                        _line(
                            "2026-08-06T00:00:01Z",
                            "turn_context",
                            {
                                "turn_id": "turn-1",
                                "model": "gpt-5.6-sol",
                                "effort": "medium",
                            },
                        ),
                        _line(
                            "2026-08-06T00:00:02Z",
                            "event_msg",
                            {
                                "type": "token_count",
                                "info": {
                                    "total_token_usage": usage,
                                    "last_token_usage": usage,
                                },
                            },
                        ),
                        _line(
                            "2026-08-06T00:00:03Z",
                            "event_msg",
                            {"type": "task_complete", "turn_id": "turn-1"},
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            prices = root / "prices.toml"
            prices.write_text(
                """
[metadata]
effective_date = "2026-08-06"
source = "test price sheet"

[[rates]]
provider = "openai"
model = "gpt-5.6-sol"
input_per_million = 1.0
cached_input_per_million = 0.5
output_per_million = 2.0
source = "test price sheet"
""",
                encoding="utf-8",
            )
            summary_path = root / "summary.json"
            output_path = root / "items.jsonl"
            template_path = root / "outcomes-template.jsonl"
            common = [
                "--sessions",
                str(sessions),
                "--prices",
                str(prices),
                "--summary-out",
                str(summary_path),
                "--output",
                str(output_path),
                "--outcomes-template-out",
                str(template_path),
            ]

            self.assertEqual(main(common), 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertNotIn("groups", summary)
            self.assertNotIn("attempt_routes", summary)
            self.assertEqual(summary["filters"]["summary_mode"], "compact")
            self.assertEqual(summary["pricing_snapshot"]["status"], "loaded")
            self.assertEqual(
                summary["pricing_snapshot"]["effective_date"], "2026-08-06"
            )
            self.assertEqual(len(summary["pricing_snapshot"]["sha256"]), 64)
            template = json.loads(template_path.read_text(encoding="utf-8"))
            self.assertEqual(template["disposition"], "REPLACE_ME")
            self.assertEqual(template["model"], "gpt-5.6-sol")
            with self.assertRaisesRegex(ValueError, "invalid disposition.*REPLACE_ME"):
                load_outcomes(template_path)

            self.assertEqual(main([*common, "--full-summary"]), 0)
            full = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(full["filters"]["summary_mode"], "full")
            self.assertIn("groups", full)
            self.assertIn("attempt_routes", full)

            self.assertEqual(
                main(
                    [
                        "--sessions",
                        str(sessions),
                        "--summary-out",
                        str(summary_path),
                        "--output",
                        str(output_path),
                    ]
                ),
                0,
            )
            unpriced = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(unpriced["pricing_snapshot"]["status"], "unconfigured")

    def test_duplicate_inputs_and_invalid_price_flags_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            duplicate_prices = root / "duplicate-prices.toml"
            duplicate_prices.write_text(
                """
[[rates]]
provider = "openai"
model = "same"
cache_write_is_in_input = false

[[rates]]
provider = "openai"
model = "same"
cache_write_is_in_input = false
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "duplicate price rate.*openai:same"
            ):
                load_rates(duplicate_prices)

            invalid_bool = root / "invalid-bool.toml"
            invalid_bool.write_text(
                """
[[rates]]
provider = "openai"
model = "same"
cache_write_is_in_input = "false"
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cache_write_is_in_input.*boolean"):
                load_rates(invalid_bool)

            duplicate_outcomes = root / "duplicate-outcomes.jsonl"
            duplicate_outcomes.write_text(
                '{"attempt_id":"same","disposition":"accepted"}\n'
                '{"attempt_id":"same","disposition":"failed"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate outcome.*same"):
                load_outcomes(duplicate_outcomes)

    def test_proxy_ambiguity_is_structural(self) -> None:
        record = SessionRecord(
            file="fixture",
            file_size_bytes=1,
            thread_id="child-ambiguous",
            parent_thread_id="parent-ambiguous",
            thread_source="subagent",
            sub_agent_activity_events=[
                {
                    "event_id": "start",
                    "agent_thread_id": "child-ambiguous",
                    "kind": "started",
                    "occurred_at_ms": 1,
                },
                {
                    "event_id": "complete-1",
                    "agent_thread_id": "child-ambiguous",
                    "kind": "completed",
                    "occurred_at_ms": 2,
                },
                {
                    "event_id": "message",
                    "agent_thread_id": "child-ambiguous",
                    "kind": "interacted",
                    "occurred_at_ms": 3,
                },
                {
                    "event_id": "follow",
                    "agent_thread_id": "child-ambiguous",
                    "kind": "interacted",
                    "occurred_at_ms": 4,
                },
                {
                    "event_id": "complete-2",
                    "agent_thread_id": "child-ambiguous",
                    "kind": "completed",
                    "occurred_at_ms": 5,
                },
            ],
            agent_tool_calls=[
                {"call_id": "message", "name": "send_message"},
                {"call_id": "follow", "name": "followup_task"},
            ],
        )
        item = {
            "thread_id": "child-ambiguous",
            "parent_thread_id": "parent-ambiguous",
            "status": "completed",
            "pricing": {"status": "missing_usage", "amount": None},
        }

        attempt = annotate_attempts([item], [record], policy="followup_aware")[0][
            "attempt"
        ]
        self.assertEqual(attempt["interaction_count"], 2)
        self.assertEqual(attempt["completion_event_count"], 2)
        self.assertEqual(
            attempt["proxy_ambiguity_reasons"],
            [
                "followup_semantics_unverified",
                "multiple_completion_events",
                "non_followup_interaction",
            ],
        )


if __name__ == "__main__":
    unittest.main()
