from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant {value!r}")

    rows: list[dict[str, Any]] = []
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, parse_constant=reject_constant)
            except ValueError as exc:
                raise ValueError(
                    f"failed to read strict JSON object from {jsonl_path}:{lineno}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object row in {jsonl_path}:{lineno}")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            try:
                line = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
            except ValueError as exc:
                raise ValueError(
                    f"failed to write strict JSON row {count + 1} to {out}: {exc}"
                ) from exc
            handle.write(line + "\n")
            count += 1
    return count


__all__ = ["read_jsonl", "write_jsonl"]
