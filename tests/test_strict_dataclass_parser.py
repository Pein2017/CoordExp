from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.config.strict_dataclass import parse_dataclass_strict


@dataclass(frozen=True)
class _Inner:
    x: int = 0


@dataclass(frozen=True)
class _Outer:
    inner: _Inner
    items: list[_Inner]


def test_strict_parser_accepts_declared_keys():
    cfg = parse_dataclass_strict(
        _Outer,
        {"inner": {"x": 1}, "items": [{"x": 2}, {"x": 3}]},
        path="cfg",
    )
    assert isinstance(cfg, _Outer)
    assert cfg.inner.x == 1
    assert [item.x for item in cfg.items] == [2, 3]


def test_strict_parser_rejects_unknown_top_level_keys_with_dotted_paths():
    with pytest.raises(ValueError, match=r"cfg\.unknown"):
        parse_dataclass_strict(
            _Outer,
            {"inner": {"x": 1}, "items": [], "unknown": 123},
            path="cfg",
        )


def test_strict_parser_rejects_unknown_nested_keys_with_dotted_paths():
    with pytest.raises(ValueError, match=r"cfg\.inner\.unknown"):
        parse_dataclass_strict(
            _Outer,
            {"inner": {"x": 1, "unknown": 2}, "items": []},
            path="cfg",
        )


def test_strict_parser_includes_list_indices_in_unknown_key_paths():
    with pytest.raises(ValueError, match=r"cfg\.items\[0\]\.unknown"):
        parse_dataclass_strict(
            _Outer,
            {"inner": {"x": 1}, "items": [{"x": 2, "unknown": 3}]},
            path="cfg",
        )


def test_strict_parser_emits_clear_type_errors_for_shape_mismatches():
    with pytest.raises(TypeError, match=r"cfg\.items must be a list"):
        parse_dataclass_strict(
            _Outer,
            {"inner": {"x": 1}, "items": {"x": 2}},
            path="cfg",
        )

    with pytest.raises(TypeError, match=r"cfg\.inner must be a mapping"):
        parse_dataclass_strict(
            _Outer,
            {"inner": ["not", "a", "mapping"], "items": []},
            path="cfg",
        )
