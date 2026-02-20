"""Shared IO helpers (behavior-preserving)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple


logger = logging.getLogger(__name__)

def load_jsonl(jsonl_path: str, *, resolve_relative: bool = False) -> List[Dict[str, Any]]:
    """Load records from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file
        resolve_relative: If True, resolves non-absolute image paths against the
            JSONL's parent directory and stores absolute paths in-memory.

    Returns:
        List of dictionaries, one per line
    """
    base_dir = Path(jsonl_path).resolve().parent
    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if resolve_relative:
                images = record.get("images")
                if isinstance(images, list):
                    resolved = []
                    for img in images:
                        img_path = Path(str(img))
                        resolved_path = (
                            img_path if img_path.is_absolute() else (base_dir / img_path).resolve()
                        )
                        resolved.append(str(resolved_path))
                    record["images"] = resolved
            records.append(record)
    return records


def load_jsonl_with_diagnostics(
    path: Path,
    *,
    strict: bool = False,
    max_snippet_len: int = 200,
    warn_limit: int = 5,
) -> Tuple[List[Dict[str, Any]], int]:
    """Load JSONL objects with bounded diagnostics.

    This helper is intended for evaluation-grade ingestion paths that need
    consistent diagnostics (path + 1-based line number + clipped snippet).

    Returns:
      (records, invalid_count)

    Notes:
    - In non-strict mode, invalid lines are skipped.
    - Warnings are emitted for the first `warn_limit` invalid lines, then
      a suppression warning is emitted once.
    """

    records: List[Dict[str, Any]] = []
    invalid_seen = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_seen += 1
                snippet = (
                    line
                    if len(line) <= max_snippet_len
                    else (line[:max_snippet_len] + "...")
                )
                msg = f"Malformed JSONL at {path}:{line_no}: {snippet}"
                if strict:
                    raise ValueError(msg) from exc
                if invalid_seen <= warn_limit:
                    logger.warning("%s", msg)
                elif invalid_seen == warn_limit + 1:
                    logger.warning(
                        "Malformed JSONL: suppressing further warnings (showing first %s only)",
                        int(warn_limit),
                    )
                continue

            if not isinstance(parsed, dict):
                invalid_seen += 1
                snippet = (
                    line
                    if len(line) <= max_snippet_len
                    else (line[:max_snippet_len] + "...")
                )
                msg = f"Non-object JSONL record at {path}:{line_no}: {snippet}"
                if strict:
                    raise ValueError(msg)
                if invalid_seen <= warn_limit:
                    logger.warning("%s", msg)
                elif invalid_seen == warn_limit + 1:
                    logger.warning(
                        "Malformed JSONL: suppressing further warnings (showing first %s only)",
                        int(warn_limit),
                    )
                continue

            records.append(parsed)

    return records, invalid_seen


__all__ = ["load_jsonl", "load_jsonl_with_diagnostics"]
