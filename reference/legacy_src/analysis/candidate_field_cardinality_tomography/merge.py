from __future__ import annotations

from typing import Mapping, Sequence


def assert_no_duplicate_active_rows(rows: Sequence[Mapping[str, object]], *, key: str = "row_id") -> None:
    seen: set[object] = set()
    for row in rows:
        value = row.get(key)
        if value in seen:
            raise ValueError(f"duplicate active row: {value}")
        seen.add(value)
