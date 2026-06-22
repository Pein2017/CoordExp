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

CAT_BBOX_TOKENS = ("<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>")
EXPECTED_COMPACT_ROW = (
    "<|object_ref_start|>cat<|box_start|>"
    "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
)
EXPECTED_BOX_CLOSED_ROW = (
    "<|object_ref_start|>cat<|box_start|>"
    "<|coord_1|><|coord_2|><|coord_3|><|coord_4|><|box_end|>"
)
EXPECTED_OBJECT_CLOSED_ROW = (
    "<|object_ref_start|>cat<|object_ref_end|><|box_start|>"
    "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
)
EXPECTED_OBJECT_BOX_CLOSED_DESC_FIRST_ROW = (
    "<|object_ref_start|>cat<|object_ref_end|><|box_start|>"
    "<|coord_1|><|coord_2|><|coord_3|><|coord_4|><|box_end|>"
)
EXPECTED_OBJECT_BOX_CLOSED_GEOMETRY_FIRST_ROW = (
    "<|box_start|><|coord_1|><|coord_2|><|coord_3|><|coord_4|><|box_end|>"
    "<|object_ref_start|>cat<|object_ref_end|>"
)
EXPECTED_OBJECT_BOX_CLOSED_LINES_ROW = (
    f"{EXPECTED_OBJECT_BOX_CLOSED_DESC_FIRST_ROW}\n"
)


def test_semantic_template_ids_are_exact() -> None:
    assert COMPACT_TEMPLATE_IDS == (
        "compact",
        "compact_box_closed",
        "compact_object_closed",
        "compact_object_box_closed",
        "compact_object_box_closed_lines",
    )


@pytest.mark.parametrize(
    ("template_id", "expected"),
    [
        (
            "compact",
            EXPECTED_COMPACT_ROW,
        ),
        (
            "compact_box_closed",
            EXPECTED_BOX_CLOSED_ROW,
        ),
        (
            "compact_object_closed",
            EXPECTED_OBJECT_CLOSED_ROW,
        ),
        (
            "compact_object_box_closed",
            EXPECTED_OBJECT_BOX_CLOSED_DESC_FIRST_ROW,
        ),
        (
            "compact_object_box_closed_lines",
            EXPECTED_OBJECT_BOX_CLOSED_LINES_ROW,
        ),
    ],
)
def test_contract_row_rendering(template_id: str, expected: str) -> None:
    contract = resolve_detection_template_contract(template_id)
    row = render_compact_contract_row(
        contract,
        desc="cat",
        bbox_tokens=CAT_BBOX_TOKENS,
    )
    assert row == expected


@pytest.mark.parametrize(
    ("object_field_order", "expected"),
    [
        (
            "desc_first",
            EXPECTED_OBJECT_BOX_CLOSED_DESC_FIRST_ROW,
        ),
        (
            "geometry_first",
            EXPECTED_OBJECT_BOX_CLOSED_GEOMETRY_FIRST_ROW,
        ),
    ],
)
def test_compact_object_box_closed_contract_row_rendering_respects_field_order(
    object_field_order: str,
    expected: str,
) -> None:
    contract = resolve_detection_template_contract("compact_object_box_closed")
    row = render_compact_contract_row(
        contract,
        desc="cat",
        bbox_tokens=CAT_BBOX_TOKENS,
        object_field_order=object_field_order,
    )
    assert row == expected


@pytest.mark.parametrize(
    ("template_id", "count"),
    [
        ("compact", 1002),
        ("compact_box_closed", 1003),
        ("compact_object_closed", 1003),
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
        ("compact_object_closed", (151646, 151647, 151648)),
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
