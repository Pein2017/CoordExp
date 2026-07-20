#!/usr/bin/env python3
"""Convert the frozen image-2299 v1 case document to the matched-screen v2 form.

This is intentionally an experiment-local conversion, not a general case
builder.  The source ledger and its four ``A, B, C`` tuples are already frozen
in the 2026-07-19 image-2299 case document.  The conversion only changes the
case schema so the v2 runner can execute the same-covered-set order comparison
and the two symmetric leave-one-out controls at a one-row horizon.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from scripts.research.run_same_covered_set_prefix_order_probe import (
    CASE_SCHEMA_VERSION,
    CASE_SCHEMA_VERSION_V2,
    validate_case_spec,
)


SOURCE_CASE_PATH = Path(
    "research/investigations/qwen3-vl-dense-enumeration/experiments/"
    "2026-07-19-same-covered-set-prefix-order-equivalence/cases-image2299.json"
)
TARGET_EXPERIMENT_ROOT = Path(
    "research/investigations/qwen3-vl-dense-enumeration/experiments/"
    "2026-07-20-matched-random-sorted-prefix-order-screen"
)
DEFAULT_OUTPUT_PATH = TARGET_EXPERIMENT_ROOT / "cases-image2299-v2.json"
EXPECTED_IMAGE_ID = "2299"
EXPECTED_ENTITY_COUNT = 38
EXPECTED_CASE_COUNT = 4

ARM_A_THEN_B_THEN_C = "a_then_b_then_c"
ARM_B_THEN_A_THEN_C = "b_then_a_then_c"
ARM_B_THEN_C_A_OMITTED = "b_then_c_coverage_control"
ARM_A_THEN_C_B_OMITTED = "a_then_c_coverage_control"


def _comparison(
    comparison_id: str,
    arm_names: tuple[str, str],
    *,
    require_same_covered_set: bool,
) -> dict[str, Any]:
    """Build one fixed one-row comparison declaration."""

    return {
        "comparison_id": comparison_id,
        "arm_names": list(arm_names),
        "shared_suffix_length": 1,
        "rollout_horizon_rows": 1,
        "require_same_covered_set": require_same_covered_set,
    }


def _v2_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Expand one validated v1 ``A, B, C`` tuple into the four v2 arms."""

    case_id = str(case["case_id"])
    a_entity_id = str(case["a_entity_id"])
    b_entity_id = str(case["b_entity_id"])
    c_entity_id = str(case["c_entity_id"])
    return {
        "case_id": case_id,
        "arms": {
            ARM_A_THEN_B_THEN_C: {"entity_ids": [a_entity_id, b_entity_id, c_entity_id]},
            ARM_B_THEN_A_THEN_C: {"entity_ids": [b_entity_id, a_entity_id, c_entity_id]},
            ARM_B_THEN_C_A_OMITTED: {"entity_ids": [b_entity_id, c_entity_id]},
            ARM_A_THEN_C_B_OMITTED: {"entity_ids": [a_entity_id, c_entity_id]},
        },
        "comparisons": [
            _comparison(
                "same_covered_set_order_swap",
                (ARM_A_THEN_B_THEN_C, ARM_B_THEN_A_THEN_C),
                require_same_covered_set=True,
            ),
            _comparison(
                "a_omitted_removal_control",
                (ARM_B_THEN_A_THEN_C, ARM_B_THEN_C_A_OMITTED),
                require_same_covered_set=False,
            ),
            _comparison(
                "b_omitted_removal_control",
                (ARM_A_THEN_B_THEN_C, ARM_A_THEN_C_B_OMITTED),
                require_same_covered_set=False,
            ),
        ],
    }


def convert_case_spec(source: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact image-2299 v2 case document without mutating ``source``.

    ``validate_case_spec`` is called on both the v1 input and the converted v2
    output.  The returned document keeps the source entity dictionaries as
    supplied (including their frozen row text) and replaces only the schema
    version and case declarations.
    """

    source_schema = str(source.get("schema_version", ""))
    if source_schema != CASE_SCHEMA_VERSION:
        raise ValueError(
            f"image-2299 converter requires {CASE_SCHEMA_VERSION!r}; "
            f"got {source_schema!r}"
        )
    source_checked = validate_case_spec(source)
    if str(source_checked["image_id"]) != EXPECTED_IMAGE_ID:
        raise ValueError(
            f"image-2299 converter requires image_id {EXPECTED_IMAGE_ID!r}; "
            f"got {source_checked['image_id']!r}"
        )
    if len(source_checked["entities"]) != EXPECTED_ENTITY_COUNT:
        raise ValueError(
            f"image-2299 converter requires {EXPECTED_ENTITY_COUNT} entities; "
            f"got {len(source_checked['entities'])}"
        )
    if len(source_checked["cases"]) != EXPECTED_CASE_COUNT:
        raise ValueError(
            f"image-2299 converter requires {EXPECTED_CASE_COUNT} cases; "
            f"got {len(source_checked['cases'])}"
        )

    converted = copy.deepcopy(dict(source))
    converted["schema_version"] = CASE_SCHEMA_VERSION_V2
    converted["cases"] = [_v2_case(case) for case in source_checked["cases"]]
    # Validate the output contract before handing it to a caller or writing it.
    validate_case_spec(converted)
    return converted


def convert_file(input_path: Path = SOURCE_CASE_PATH, output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Read the frozen v1 file, write its validated v2 conversion, and return the path."""

    source = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(source, Mapping):
        raise ValueError(f"{input_path} must contain one JSON object")
    converted = convert_case_spec(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(converted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE_CASE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    print(convert_file(args.input, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
