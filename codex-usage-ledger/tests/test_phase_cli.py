from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codex_usage_ledger.cli import main


def _line(timestamp: str, kind: str, payload: dict) -> str:
    return json.dumps({"timestamp": timestamp, "type": kind, "payload": payload}) + "\n"


def _usage(total: int, response_id: str, timestamp: str) -> str:
    return _line(
        timestamp,
        "token_usage_record",
        {
            "thread_id": "child-thread",
            "session_id": "child-thread",
            "turn_id": "turn-1",
            "root_turn_id": "turn-1",
            "response_id": response_id,
            "usage": {
                "input_tokens": total,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_output_tokens": 0,
                "total_tokens": total,
            },
        },
    )


def _rollout(root: Path) -> Path:
    path = root / "rollout-2026-09-01T00-00-00-child.jsonl"
    path.write_text(
        "".join(
            [
                _line(
                    "2026-09-01T00:00:00Z",
                    "session_meta",
                    {"id": "child-thread", "thread_source": "subagent"},
                ),
                _usage(100, "response-1", "2026-09-01T00:00:01Z"),
                _usage(50, "response-2", "2026-09-01T00:01:01Z"),
            ]
        ),
        encoding="utf-8",
    )
    return path


class PhaseCliTests(unittest.TestCase):
    def _summary(self, root: Path, name: str, argv: list[str]) -> dict:
        summary = root / f"{name}.json"
        self.assertEqual(main([*argv, "--summary-out", str(summary)]), 0)
        return json.loads(summary.read_text(encoding="utf-8"))

    def test_explicit_rollouts_window_receipts_without_double_billing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout = _rollout(root)
            common = ["--rollout", str(rollout), "--format", "json"]
            with patch.dict("codex_usage_ledger.cli.os.environ", {}, clear=True):
                full = self._summary(root, "full", common)
            early = self._summary(
                root,
                "early",
                [
                    *common,
                    "--receipt-since",
                    "2026-09-01T00:00:00+00:00",
                    "--receipt-until",
                    "2026-09-01T00:01:00+00:00",
                ],
            )
            late = self._summary(
                root,
                "late",
                [
                    *common,
                    "--receipt-since",
                    "2026-09-01T00:01:00+00:00",
                    "--receipt-until",
                    "2026-09-01T00:02:00+00:00",
                ],
            )

            self.assertEqual(full["totals"]["measured_tokens"], 150)
            self.assertEqual(early["totals"]["measured_tokens"], 100)
            self.assertEqual(late["totals"]["measured_tokens"], 50)
            self.assertEqual(
                early["totals"]["measured_tokens"] + late["totals"]["measured_tokens"],
                full["totals"]["measured_tokens"],
            )
            self.assertEqual(early["filters"]["rollout_files"], [str(rollout.resolve())])
            self.assertIsNone(full["sessions_root"])

    def test_explicit_rollouts_bypass_discovery_and_deduplicate_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout = _rollout(root)
            summary = root / "summary.json"
            with patch(
                "codex_usage_ledger.cli.discover_files",
                side_effect=AssertionError("explicit files must not discover"),
            ):
                self.assertEqual(
                    main(
                        [
                            "--rollout",
                            str(rollout),
                            "--rollout",
                            str(rollout.resolve()),
                            "--summary-out",
                            str(summary),
                        ]
                    ),
                    0,
                )
            report = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(report["scan"]["files_seen"], 1)
            self.assertEqual(report["filters"]["rollout_files"], [str(rollout.resolve())])

    def test_invalid_receipt_boundaries_and_mixed_windows_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rollout = _rollout(Path(tmp))
            for argv in (
                ["--rollout", str(rollout), "--receipt-since", "not-a-time"],
                ["--rollout", str(rollout), "--receipt-until", "2026-09-01T00:00:00"],
            ):
                with self.subTest(argv=argv), self.assertRaises(SystemExit) as raised:
                    main(argv)
                self.assertEqual(raised.exception.code, 2)
            for argv in (
                [
                    "--rollout",
                    str(rollout),
                    "--since",
                    "2026-09-01",
                    "--receipt-since",
                    "2026-09-01T00:00:00+00:00",
                ],
                ["--rollout", str(rollout), "--sessions", str(rollout.parent)],
                ["--rollout", str(rollout), "--max-files", "1"],
            ):
                with self.subTest(argv=argv):
                    self.assertEqual(main(argv), 2)

    def test_required_outcomes_fail_before_outputs_and_explicit_labels_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout = _rollout(root)
            output = root / "report.jsonl"
            summary = root / "summary.json"
            common = [
                "--rollout",
                str(rollout),
                "--require-outcomes",
                "--output",
                str(output),
                "--summary-out",
                str(summary),
            ]
            self.assertEqual(main(common), 2)
            self.assertFalse(output.exists())
            self.assertFalse(summary.exists())

            outcomes = root / "outcomes.jsonl"
            outcomes.write_text(
                '{"thread_id":"other-thread","disposition":"accepted"}\n',
                encoding="utf-8",
            )
            self.assertEqual(main([*common, "--outcomes", str(outcomes)]), 2)
            self.assertFalse(output.exists())
            self.assertFalse(summary.exists())

            outcomes.write_text(
                '{"thread_id":"child-thread","disposition":"accepted"}\n',
                encoding="utf-8",
            )
            self.assertEqual(main([*common, "--outcomes", str(outcomes)]), 0)
            self.assertTrue(output.exists())
            self.assertTrue(summary.exists())

    def test_explicit_pages_allow_calendar_dates_and_discovery_stays_compact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout = _rollout(root)
            explicit = self._summary(root, "calendar", ["--rollout", str(rollout), "--since", "2026-09-01", "--until", "2026-09-01"])
            self.assertEqual(explicit["totals"]["measured_tokens"], 150)
            discovered = self._summary(root, "discovered", ["--sessions", str(root)])
            self.assertIsNone(discovered["filters"]["rollout_files"])
            self.assertEqual(discovered["scan"]["files_seen"], 1)

    def test_required_outcomes_refreshes_existing_report_and_records_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout = _rollout(root)
            outcomes = root / "labels.jsonl"
            args = ["--rollout", str(rollout), "--outcomes", str(outcomes), "--require-outcomes"]
            for label in ("accepted", "failed"):
                outcomes.write_text(json.dumps({"thread_id": "child-thread", "disposition": label}) + "\n")
                report = self._summary(root, "same-output", args)
                self.assertTrue(report["filters"]["require_outcomes"])
                self.assertEqual(report["attempts"]["disposition_counts"][label], 1)

    def test_invalid_sources_windows_and_outcome_modes_write_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout = _rollout(root)
            labels = root / "labels.jsonl"
            labels.write_text('{"thread_id":"child-thread","disposition":"accepted"}\n')
            for args in (
                ["--rollout", str(root / "absent.jsonl")],
                ["--rollout", str(rollout), "--receipt-since", "2026-09-01T00:01:01Z", "--receipt-until", "2026-09-01T00:01:01Z"],
                ["--rollout", str(rollout), "--outcomes", str(labels), "--require-outcomes", "--disposition-policy", "completed"],
            ):
                summary = root / "must-not-exist.json"
                self.assertEqual(main([*args, "--summary-out", str(summary)]), 2)
                self.assertFalse(summary.exists())
