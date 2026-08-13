from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codex_usage_ledger.parser import parse_rollout
from codex_usage_ledger.pricing import load_rates
from codex_usage_ledger.report import enrich_record, summarize, summarize_totals
from codex_usage_ledger.cli import filter_records
from codex_usage_ledger.parser import SessionRecord
from codex_usage_ledger.attempts import (
    annotate_attempts,
    load_outcomes,
    summarize_attempts,
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

        subtree, subtree_scope = filter_records(
            records, True, root_thread_id="root-a"
        )
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
            self.assertEqual(record.sub_agent_activity_events[0]["event_id"], "call-start")
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
            completed_proxy = annotate_attempts(
                [item], [record], policy="completed"
            )[0]
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


if __name__ == "__main__":
    unittest.main()
