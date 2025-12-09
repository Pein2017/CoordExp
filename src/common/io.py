"""Shared IO helpers (behavior-preserving)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


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


__all__ = ["load_jsonl"]
