from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.analysis.prefix_state_transition_tomography.jsonl import read_jsonl, write_jsonl


def test_jsonl_roundtrip_valid_object_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"

    count = write_jsonl(path, [{"a": 1}, {"b": [2, 3]}])

    assert count == 2
    assert read_jsonl(path) == [{"a": 1}, {"b": [2, 3]}]


def test_write_jsonl_rejects_non_finite_floats(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"

    with pytest.raises(ValueError, match="row 1"):
        write_jsonl(path, [{"metric": math.nan}])


def test_read_jsonl_rejects_non_object_rows(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\n[1, 2]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="bad.jsonl:2"):
        read_jsonl(path)


def test_read_jsonl_rejects_non_finite_constants(tmp_path: Path) -> None:
    path = tmp_path / "nan.jsonl"
    path.write_text('{"metric": NaN}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="nan.jsonl:1"):
        read_jsonl(path)

