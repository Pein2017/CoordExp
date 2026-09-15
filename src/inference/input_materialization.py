"""Narrow relocation of bound single-image research input rows.

This module does not define a general case schema.  It only preserves the
already-bound scientific payload while relocating its one image reference for a
target JSONL origin.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.config.fingerprint import sha256_file


def materialize_bound_single_image_case(
    case: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Relocate one hash-bound image reference without changing other payload."""

    path = Path(case["image_path"])
    if not (path.is_absolute() and path.is_file()):
        raise ValueError("candidate bound image missing")
    if sha256_file(path) != case["image_plan"]["image_content_sha256"]:
        raise ValueError("candidate bound image bytes changed")
    if len(case["input_record"]["images"]) != 1:
        raise ValueError("candidate requires one frozen image")

    materialized = copy.deepcopy(case)
    input_jsonl = Path(config["data"]["input_jsonl"])
    materialized["input_record"]["images"] = [
        os.path.relpath(path, input_jsonl.parent)
    ]
    return materialized
