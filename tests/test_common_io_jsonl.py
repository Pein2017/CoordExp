from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.common.io import load_jsonl_with_diagnostics


def test_load_jsonl_with_diagnostics_non_strict_skips_invalid_and_counts(
    tmp_path: Path, caplog
) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "not json\n" + json.dumps([1, 2, 3]) + "\n" + json.dumps({"ok": 1}) + "\n",
        encoding="utf-8",
    )

    caplog.set_level(logging.WARNING)

    records, invalid_count = load_jsonl_with_diagnostics(
        path,
        strict=False,
        warn_limit=1,
        max_snippet_len=64,
    )

    assert records == [{"ok": 1}]
    assert invalid_count == 2

    msgs = [r.getMessage() for r in caplog.records]
    assert any(f"{path}:1" in m and "Malformed JSONL at" in m for m in msgs)
    assert any("suppressing further warnings" in m for m in msgs)


def test_load_jsonl_with_diagnostics_strict_reports_path_and_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("not json\n" + json.dumps({"ok": 1}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_jsonl_with_diagnostics(path, strict=True, max_snippet_len=64)

    msg = str(excinfo.value)
    assert str(path) in msg
    assert f"{path}:1" in msg
    assert "not json" in msg


def test_load_jsonl_with_diagnostics_strict_rejects_non_object_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "list.jsonl"
    path.write_text(json.dumps({"ok": 1}) + "\n" + json.dumps([1, 2, 3]) + "\n")

    with pytest.raises(ValueError, match="Non-object JSONL record"):
        load_jsonl_with_diagnostics(path, strict=True)
