from __future__ import annotations

import importlib.util
from pathlib import Path


REDUCER = Path(__file__).parents[1] / "reduce.py"
SPEC = importlib.util.spec_from_file_location("native_escape_reducer", REDUCER)
assert SPEC and SPEC.loader
reducer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reducer)


def row(order: int, bbox: list[int]) -> dict:
    return {"generated_order": order, "bbox": bbox}


def test_repeat_is_class_blind_strict_and_counted_once_per_later_row() -> None:
    record = {
        "parsed": {"pred": [
            row(0, [0, 0, 100, 100]),
            row(1, [200, 200, 300, 300]),
            row(2, [0, 0, 100, 100]),
            row(3, [0, 0, 100, 100]),
        ]},
        "parser_partition": {"prefix_row_count": 2},
    }
    assert reducer.strict_repeat_orders(record) == [2, 3]


def test_exact_point_ninety_five_is_not_strict_repeat() -> None:
    left = [0, 0, 100, 100]
    right = [0, 0, 95, 100]
    assert reducer.iou(left, right) == 0.95
    record = {
        "parsed": {"pred": [row(0, left), row(1, right)]},
        "parser_partition": {"prefix_row_count": 1},
    }
    assert reducer.strict_repeat_orders(record) == []


def test_invalid_burden_keeps_geometry_other_complete_and_incomplete_separate() -> None:
    record = {"parser_partition": {"free_rows": [
        {"generated_order": 1, "parser_disposition": "dropped",
         "reason": "geometry_invalid", "raw_span_text": "bad<|box_end|>"},
        {"generated_order": 2, "parser_disposition": "dropped",
         "reason": "schema_invalid", "raw_span_text": "bad<|box_end|>"},
        {"generated_order": 3, "parser_disposition": "dropped",
         "reason": "malformed_object_span", "raw_span_text": "<|object_ref_start|>"},
    ]}}
    assert reducer.burden(record) == {
        "geometry_invalid_complete_count": 1,
        "geometry_invalid_complete_orders": [1],
        "other_complete_malformed_count": 1,
        "other_complete_malformed_orders": [2],
        "incomplete_fragment_count": 1,
        "incomplete_fragment_orders": [3],
    }
