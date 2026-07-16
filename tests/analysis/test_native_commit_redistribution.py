from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.run_native_commit_redistribution import (
    BOX_END,
    BOX_START,
    CONDITION_NAMES,
    NO_APPENDED_ROW,
    OBJECT_REF_END,
    OBJECT_REF_START,
    OTHER_DESCRIPTION_WITH_TARGET_GEOMETRY,
    TARGET_DESCRIPTION_WITH_OTHER_GEOMETRY,
    _row_factors,
    build_condition_rows,
    canonical_cell_seeds,
    compose_row_tokens,
    extract_source_rows,
)


def _row(description: list[int], geometry: list[int]) -> list[int]:
    return [OBJECT_REF_START, *description, OBJECT_REF_END, BOX_START, *geometry, BOX_END]


def _source_bundle() -> dict[str, object]:
    target = _row([65, 9605], [151832, 151815, 151908, 151858])
    other = _row([65, 62118], [152419, 152295, 152463, 152460])
    generated = [*target, *([999] * 50), *other, 151645]
    return {
        "request_id": "frozen-source",
        "execution_evidence": {"image_id": "7574"},
        "decode_result": {
            "prompt_token_ids": [1, 2, 3],
            "generated_token_ids": generated,
        },
    }


def test_row_factors_accepts_model_native_ten_token_row() -> None:
    tokens = _row([65, 9605], [151832, 151815, 151908, 151858])
    factors = _row_factors(tokens, row_label="target")
    assert factors["token_count"] == 10
    assert factors["description_token_ids"] == [65, 9605]
    assert factors["coordinate_token_ids"] == [151832, 151815, 151908, 151858]


@pytest.mark.parametrize(
    "tokens",
    [
        [OBJECT_REF_START, 65, OBJECT_REF_END, BOX_START, 151832, 151815, 151908, 151858, BOX_END],
        _row([65], [151832, 151815, 151908, 151858]),
        _row([65, 9605], [151669, 151815, 151908, 151858]),
    ],
)
def test_row_factors_rejects_non_contract_rows(tokens: list[int]) -> None:
    with pytest.raises(SystemExit):
        _row_factors(tokens, row_label="bad")


def test_crossed_rows_keep_description_and_geometry_factors_separate() -> None:
    source_rows = extract_source_rows(_source_bundle())
    conditions = build_condition_rows(source_rows)
    crossed_target = conditions[TARGET_DESCRIPTION_WITH_OTHER_GEOMETRY]["appended_row_token_ids"]
    crossed_other = conditions[OTHER_DESCRIPTION_WITH_TARGET_GEOMETRY]["appended_row_token_ids"]
    assert crossed_target[:3] == [OBJECT_REF_START, 65, 9605]
    assert crossed_target[5:9] == [152419, 152295, 152463, 152460]
    assert crossed_other[:3] == [OBJECT_REF_START, 65, 62118]
    assert crossed_other[5:9] == [151832, 151815, 151908, 151858]
    assert len(crossed_target) == len(crossed_other) == 10


def test_condition_set_has_one_no_row_and_four_full_name_rows() -> None:
    conditions = build_condition_rows(extract_source_rows(_source_bundle()))
    assert tuple(conditions) == CONDITION_NAMES
    assert conditions[NO_APPENDED_ROW]["appended_row_token_ids"] is None
    assert all(
        len(conditions[name]["appended_row_token_ids"]) == 10
        for name in CONDITION_NAMES[1:]
    )


def test_extract_source_rows_freezes_exact_spans_and_hashes() -> None:
    rows = extract_source_rows(_source_bundle())
    assert rows["target_row"]["source_token_span"] == [0, 10]
    assert rows["other_row"]["source_token_span"] == [60, 70]
    assert rows["target_row"]["annotation_id"] == "1535235"
    assert rows["other_row"]["annotation_id"] == "90913"
    assert len(rows["prompt_token_ids_sha256"]) == 64


def test_canonical_cell_seeds_reads_first_eight_full_bag_cells(tmp_path: Path) -> None:
    for index in range(8):
        path = tmp_path / f"cell-{index}" / "terminal-output-bundle.json"
        path.parent.mkdir()
        path.write_text(
            json.dumps(
                {
                    "execution_evidence": {
                        "image_id": "7574",
                        "arm": {"arm_code": "FULL_BAG_K"},
                    },
                    "scheduled_request": {
                        "cell_index": index,
                        "sampling_seed": index + 100,
                    },
                }
            ),
            encoding="utf-8",
        )
    assert canonical_cell_seeds(tmp_path) == {index: index + 100 for index in range(8)}


def test_compose_row_rejects_wrong_factor_lengths() -> None:
    with pytest.raises(SystemExit):
        compose_row_tokens([65], [151832, 151815, 151908, 151858], row_label="wrong")
