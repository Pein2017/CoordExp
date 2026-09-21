"""JSONL loading for V1 raw examples."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.common.errors import DataContractError
from src.config.models import DatasetSplitConfig
from src.data.examples import RawExample, raw_example_from_jsonl_row


def iter_raw_examples(
    dataset: DatasetSplitConfig | str | Path,
    *,
    sample_limit: int | None = None,
) -> Iterator[RawExample]:
    jsonl_path, limit = _dataset_path_and_limit(dataset, sample_limit=sample_limit)
    if not jsonl_path.exists():
        raise DataContractError(
            "raw example JSONL does not exist",
            code="data.jsonl_missing",
            context={"path": str(jsonl_path)},
        )
    if not jsonl_path.is_file():
        raise DataContractError(
            "raw example JSONL path is not a file",
            code="data.jsonl_not_file",
            context={"path": str(jsonl_path)},
        )

    seen_example_ids: set[str] = set()
    accepted = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for row_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                raise DataContractError(
                    "blank JSONL rows are not allowed",
                    code="data.blank_row",
                    context={"path": str(jsonl_path), "row_number": row_number},
                )
            try:
                payload: Any = json.loads(line, parse_constant=_reject_json_constant)
            except DataContractError:
                raise
            except json.JSONDecodeError as exc:
                raise DataContractError(
                    "raw example row is not valid JSON",
                    code="data.json_decode",
                    context={"path": str(jsonl_path), "row_number": row_number},
                    cause=exc,
                ) from exc
            example = raw_example_from_jsonl_row(
                payload,
                jsonl_path=jsonl_path,
                row_number=row_number,
                raw_line=line,
            )
            if example.example_id in seen_example_ids:
                raise DataContractError(
                    "example_id must be unique within one JSONL source",
                    code="data.duplicate_example_id",
                    context={
                        "path": str(jsonl_path),
                        "row_number": row_number,
                        "example_id": example.example_id,
                    },
                )
            seen_example_ids.add(example.example_id)
            yield example
            accepted += 1
            if limit is not None and accepted >= limit:
                return


def load_raw_examples(
    dataset: DatasetSplitConfig | str | Path,
    *,
    sample_limit: int | None = None,
) -> tuple[RawExample, ...]:
    return tuple(iter_raw_examples(dataset, sample_limit=sample_limit))


def _dataset_path_and_limit(
    dataset: DatasetSplitConfig | str | Path,
    *,
    sample_limit: int | None,
) -> tuple[Path, int | None]:
    if isinstance(dataset, DatasetSplitConfig):
        path = Path(dataset.path).expanduser().resolve()
        limit = dataset.sample_limit if sample_limit is None else sample_limit
    else:
        path = Path(dataset).expanduser().resolve()
        limit = sample_limit
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise DataContractError(
            "sample_limit must be positive when provided",
            code="data.sample_limit",
            context={"sample_limit": limit},
        )
    return path, limit


def _reject_json_constant(value: str) -> None:
    raise DataContractError(
        "raw example JSONL must not contain NaN or Infinity",
        code="data.json_constant",
        context={"constant": value},
    )


__all__ = ["iter_raw_examples", "load_raw_examples"]
