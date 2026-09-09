"""Codex response-only rollout compatibility through the collector's main path."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import collect_sessions


class CodexResponseCollectionTests(unittest.TestCase):
    def collect(self, events=(), messages=True):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            home = root / "codex"
            session = home / "sessions" / "rollout-example.jsonl"
            session.parent.mkdir(parents=True)
            records = [{"type": "session_meta", "payload": {
                "id": "example", "cwd": str(repo),
            }}]
            # Sanitized response_item shapes from Aug 22 rollout
            # 01a02323-1107-7a02-8f28-31cb2d281f5b, messages at lines 4/18
            # and custom tool call at line 19. Content and metadata are sanitized.
            for role, text in [
                ("user", "<environment_context>Injected context</environment_context>"),
                ("user", "Inspect the example." if messages else ""),
                ("assistant", "The example passes." if messages else ""),
                ("assistant", "Checked the result." if messages else ""),
            ]:
                records.append({"type": "response_item", "payload": {
                    "type": "message", "role": role, "content": [{
                        "type": "input_text" if role == "user" else "output_text",
                        "text": text,
                    }],
                }})
            records.append({"type": "response_item", "payload": {
                "type": "custom_tool_call", "name": "exec",
                "input": 'text("example")',
            }})
            records.extend({"type": "event_msg", "payload": {"type": event}}
                           for event in events)
            session.write_text("\n".join(json.dumps(record) for record in records))
            out = root / "report"
            argv = ["collect_sessions", "--harness", "codex", "--codex-home",
                    str(home), "--repo", str(repo), "--out", str(out)]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                collect_sessions.main()
            return json.loads((out / "inventory.json").read_text())

    def test_response_only_session_reaches_sampling(self):
        inventory = self.collect()
        self.assertEqual(inventory["stats"]["sessions_in_scope"], 1)
        self.assertEqual(inventory["stats"]["sessions_considered"], 1)
        self.assertEqual(inventory["stats"]["sessions_sampled"], 1)
        stats = inventory["sessions"][0]["stats"]
        self.assertEqual((stats["user_turns"], stats["assistant_turns"]), (1, 2))

    def test_mixed_format_preserves_event_counts_without_double_counting(self):
        inventory = self.collect(events=("user_message", "agent_message"))
        self.assertEqual(inventory["stats"]["sessions_sampled"], 1)
        stats = inventory["sessions"][0]["stats"]
        self.assertEqual((stats["user_turns"], stats["assistant_turns"]), (1, 1))

    def test_each_role_falls_back_only_when_its_events_are_absent(self):
        inventory = self.collect(events=("user_message", "user_message"))
        self.assertEqual(inventory["stats"]["sessions_sampled"], 1)
        stats = inventory["sessions"][0]["stats"]
        self.assertEqual((stats["user_turns"], stats["assistant_turns"]), (2, 2))

    def test_injected_and_empty_messages_do_not_make_session_scoreable(self):
        inventory = self.collect(messages=False)
        self.assertEqual(inventory["stats"]["sessions_in_scope"], 1)
        self.assertEqual(inventory["stats"]["sessions_considered"], 0)
        self.assertEqual(inventory["stats"]["sessions_sampled"], 0)


if __name__ == "__main__":
    unittest.main()
