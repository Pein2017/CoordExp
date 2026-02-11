from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pycocotools")

from src.eval.detection import EvalCounters, load_jsonl


def test_load_jsonl_strict_reports_path_and_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("not json\n" + json.dumps({"ok": 1}) + "\n", encoding="utf-8")

    counters = EvalCounters()
    with pytest.raises(ValueError) as excinfo:
        load_jsonl(path, counters, strict=True, max_snippet_len=64)

    msg = str(excinfo.value)
    assert str(path) in msg
    assert f"{path}:1" in msg
    assert "not json" in msg


def test_load_jsonl_counts_invalid_json_and_continues(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text("not json\n" + json.dumps({"ok": 1}) + "\n", encoding="utf-8")

    counters = EvalCounters()
    records = load_jsonl(path, counters, strict=False)
    assert counters.invalid_json == 1
    assert records == [{"ok": 1}]


def test_load_jsonl_rejects_non_object_records_in_strict_mode(tmp_path: Path) -> None:
    path = tmp_path / "list.jsonl"
    path.write_text(json.dumps([1, 2, 3]) + "\n", encoding="utf-8")

    counters = EvalCounters()
    with pytest.raises(ValueError, match="Non-object JSONL record"):
        load_jsonl(path, counters, strict=True)
