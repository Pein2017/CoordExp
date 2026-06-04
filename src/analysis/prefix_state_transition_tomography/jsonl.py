from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            try:
                line = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
            except ValueError as exc:
                raise ValueError(f"failed to write strict JSON row {count + 1} to {path}: {exc}") from exc
            f.write(line + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant {value!r}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=reject_constant)
            except ValueError as exc:
                raise ValueError(f"failed to read JSON object from {path}:{lineno}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object row in {path}:{lineno}")
            rows.append(row)
    return rows
