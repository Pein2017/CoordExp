from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.research.convert_image2299_case_v1_to_v2 import (
    ARM_A_THEN_B_THEN_C,
    ARM_A_THEN_C_B_OMITTED,
    ARM_B_THEN_A_THEN_C,
    ARM_B_THEN_C_A_OMITTED,
    SOURCE_CASE_PATH,
    convert_case_spec,
    convert_file,
)
from scripts.research.run_same_covered_set_prefix_order_probe import (
    CASE_SCHEMA_VERSION_V2,
    validate_case_spec,
)


def _source() -> dict:
    return json.loads(SOURCE_CASE_PATH.read_text(encoding="utf-8"))


def test_conversion_preserves_image2299_ledger_and_expands_all_four_tuples() -> None:
    source = _source()
    source_before = copy.deepcopy(source)

    converted = convert_case_spec(source)
    checked = validate_case_spec(converted)

    assert source == source_before
    assert checked["schema_version"] == CASE_SCHEMA_VERSION_V2
    assert checked["image_id"] == "2299"
    assert converted["entities"] == source["entities"]
    assert len(converted["entities"]) == 38
    assert len(converted["cases"]) == 4

    for source_case, converted_case in zip(source["cases"], converted["cases"], strict=True):
        a = source_case["a_entity_id"]
        b = source_case["b_entity_id"]
        c = source_case["c_entity_id"]
        assert converted_case["case_id"] == source_case["case_id"]
        assert converted_case["arms"] == {
            ARM_A_THEN_B_THEN_C: {"entity_ids": [a, b, c]},
            ARM_B_THEN_A_THEN_C: {"entity_ids": [b, a, c]},
            ARM_B_THEN_C_A_OMITTED: {"entity_ids": [b, c]},
            ARM_A_THEN_C_B_OMITTED: {"entity_ids": [a, c]},
        }
        comparisons = converted_case["comparisons"]
        assert [comparison["comparison_id"] for comparison in comparisons] == [
            "same_covered_set_order_swap",
            "a_omitted_removal_control",
            "b_omitted_removal_control",
        ]
        assert all(
            comparison["shared_suffix_length"] == 1
            and comparison["rollout_horizon_rows"] == 1
            for comparison in comparisons
        )
        assert comparisons[0]["arm_names"] == [ARM_A_THEN_B_THEN_C, ARM_B_THEN_A_THEN_C]
        assert comparisons[0]["require_same_covered_set"] is True
        assert comparisons[1]["arm_names"] == [ARM_B_THEN_A_THEN_C, ARM_B_THEN_C_A_OMITTED]
        assert comparisons[1]["require_same_covered_set"] is False
        assert comparisons[2]["arm_names"] == [ARM_A_THEN_B_THEN_C, ARM_A_THEN_C_B_OMITTED]
        assert comparisons[2]["require_same_covered_set"] is False


def test_conversion_is_scoped_to_the_frozen_source_shape() -> None:
    source = _source()

    wrong_schema = copy.deepcopy(source)
    wrong_schema["schema_version"] = CASE_SCHEMA_VERSION_V2
    with pytest.raises(ValueError, match="requires .*case.v1"):
        convert_case_spec(wrong_schema)

    wrong_image = copy.deepcopy(source)
    wrong_image["image_id"] = "2300"
    with pytest.raises(ValueError, match="image_id"):
        convert_case_spec(wrong_image)

    missing_entity = copy.deepcopy(source)
    missing_entity["entities"] = missing_entity["entities"][:-1]
    with pytest.raises(ValueError, match="38 entities"):
        convert_case_spec(missing_entity)

    missing_case = copy.deepcopy(source)
    missing_case["cases"] = missing_case["cases"][:-1]
    with pytest.raises(ValueError, match="4 cases"):
        convert_case_spec(missing_case)


def test_convert_file_writes_a_validator_accepted_document(tmp_path: Path) -> None:
    output_path = tmp_path / "cases-image2299-v2.json"
    returned = convert_file(SOURCE_CASE_PATH, output_path)

    assert returned == output_path
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert validate_case_spec(written)["schema_version"] == CASE_SCHEMA_VERSION_V2
