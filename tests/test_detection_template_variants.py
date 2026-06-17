import pytest

from src.detection.template_contracts import (
    BOX_END_TOKEN,
    COMPACT_TEMPLATE_IDS,
    OBJECT_REF_END_TOKEN,
    required_trainable_token_count,
    required_trainable_token_row_ids,
    render_compact_contract_row,
    resolve_detection_template_contract,
)


def test_semantic_template_ids_are_exact() -> None:
    assert COMPACT_TEMPLATE_IDS == (
        "compact",
        "compact_box_closed",
        "compact_object_box_closed",
        "compact_object_box_closed_lines",
    )


@pytest.mark.parametrize(
    ("template_id", "expected"),
    [
        (
            "compact",
            "<|object_ref_start|>cat<|box_start|><|coord_1|><|coord_2|><|coord_3|><|coord_4|>",
        ),
        (
            "compact_box_closed",
            "<|object_ref_start|>cat<|box_start|><|coord_1|><|coord_2|><|coord_3|><|coord_4|><|box_end|>",
        ),
        (
            "compact_object_box_closed",
            "<|object_ref_start|>cat<|object_ref_end|><|box_start|><|coord_1|><|coord_2|><|coord_3|><|coord_4|><|box_end|>",
        ),
        (
            "compact_object_box_closed_lines",
            "<|object_ref_start|>cat<|object_ref_end|><|box_start|><|coord_1|><|coord_2|><|coord_3|><|coord_4|><|box_end|>\n",
        ),
    ],
)
def test_contract_row_rendering(template_id: str, expected: str) -> None:
    contract = resolve_detection_template_contract(template_id)
    row = render_compact_contract_row(
        contract,
        desc="cat",
        bbox_tokens=("<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"),
    )
    assert row == expected


@pytest.mark.parametrize(
    ("template_id", "count"),
    [
        ("compact", 1002),
        ("compact_box_closed", 1003),
        ("compact_object_box_closed", 1004),
        ("compact_object_box_closed_lines", 1004),
    ],
)
def test_required_trainable_token_count(template_id: str, count: int) -> None:
    assert required_trainable_token_count(template_id) == count


@pytest.mark.parametrize(
    ("template_id", "expected_structural_ids"),
    [
        ("compact", (151646, 151648)),
        ("compact_box_closed", (151646, 151648, 151649)),
        ("compact_object_box_closed", (151646, 151647, 151648, 151649)),
        ("compact_object_box_closed_lines", (151646, 151647, 151648, 151649)),
    ],
)
def test_required_trainable_token_row_ids(
    template_id: str,
    expected_structural_ids: tuple[int, ...],
) -> None:
    row_ids = required_trainable_token_row_ids(template_id)
    assert row_ids[: len(expected_structural_ids)] == expected_structural_ids
    assert row_ids[len(expected_structural_ids) :] == tuple(range(151670, 152670))


def test_closed_token_constants_are_available() -> None:
    assert OBJECT_REF_END_TOKEN == "<|object_ref_end|>"
    assert BOX_END_TOKEN == "<|box_end|>"


def test_compact_full_is_rejected() -> None:
    with pytest.raises(ValueError, match="compact_full"):
        resolve_detection_template_contract("compact_full")
