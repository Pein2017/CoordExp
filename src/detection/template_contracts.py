"""Semantic detection template contracts.

This module owns the compact-template metadata that must stay shared across
training, inference, token-row adaptation, and artifact provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from src.tokens.qwen_native import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    EXPECTED_BOX_END_ID,
    EXPECTED_BOX_START_ID,
    EXPECTED_COORD_END_ID,
    EXPECTED_COORD_START_ID,
    EXPECTED_OBJECT_REF_END_ID,
    EXPECTED_OBJECT_REF_START_ID,
    OBJECT_REF_END_TOKEN,
    OBJECT_REF_START_TOKEN,
)

STAGE1_JSON_PRETTY_TEMPLATE_ID = "stage1_json_pretty"

COMPACT_TEMPLATE_IDS = (
    "compact",
    "compact_box_closed",
    "compact_object_box_closed",
    "compact_object_box_closed_lines",
)

SUPPORTED_DETECTION_TEMPLATE_IDS = (
    STAGE1_JSON_PRETTY_TEMPLATE_ID,
    *COMPACT_TEMPLATE_IDS,
)

DetectionTemplateId = Literal[
    "stage1_json_pretty",
    "compact",
    "compact_box_closed",
    "compact_object_box_closed",
    "compact_object_box_closed_lines",
]


@dataclass(frozen=True)
class DetectionTemplateContract:
    template_id: DetectionTemplateId
    is_compact: bool
    include_object_ref_end: bool
    include_box_end: bool
    row_separator: str
    canonical_final_separator: str
    required_structural_tokens: tuple[str, ...]
    required_structural_token_ids: tuple[int, ...]
    prompt_pattern: str


def resolve_detection_template_contract(
    template_id: str,
) -> DetectionTemplateContract:
    """Resolve a semantic detection template id to its shared contract."""

    value = str(template_id)
    if value == STAGE1_JSON_PRETTY_TEMPLATE_ID:
        return DetectionTemplateContract(
            template_id=STAGE1_JSON_PRETTY_TEMPLATE_ID,
            is_compact=False,
            include_object_ref_end=False,
            include_box_end=False,
            row_separator="",
            canonical_final_separator="",
            required_structural_tokens=(),
            required_structural_token_ids=(),
            prompt_pattern="stage1_json_pretty",
        )
    if value == "compact":
        return _compact_contract(
            template_id="compact",
            include_object_ref_end=False,
            include_box_end=False,
            row_separator="",
            canonical_final_separator="",
        )
    if value == "compact_box_closed":
        return _compact_contract(
            template_id="compact_box_closed",
            include_object_ref_end=False,
            include_box_end=True,
            row_separator="",
            canonical_final_separator="",
        )
    if value == "compact_object_box_closed":
        return _compact_contract(
            template_id="compact_object_box_closed",
            include_object_ref_end=True,
            include_box_end=True,
            row_separator="",
            canonical_final_separator="",
        )
    if value == "compact_object_box_closed_lines":
        return _compact_contract(
            template_id="compact_object_box_closed_lines",
            include_object_ref_end=True,
            include_box_end=True,
            row_separator="\n",
            canonical_final_separator="\n",
        )

    allowed = ", ".join(SUPPORTED_DETECTION_TEMPLATE_IDS)
    raise ValueError(
        f"Unsupported detection_template.id={template_id!r}; "
        f"supported semantic ids: {allowed}"
    )


def is_compact_template_id(template_id: str) -> bool:
    return resolve_detection_template_contract(template_id).is_compact


def render_compact_contract_row(
    contract: DetectionTemplateContract,
    *,
    desc: str,
    bbox_tokens: Sequence[str],
) -> str:
    """Render one compact row from an already-resolved template contract."""

    if not contract.is_compact:
        raise ValueError(
            f"detection_template.id={contract.template_id!r} is not compact"
        )
    if len(tuple(bbox_tokens)) != 4:
        raise ValueError("compact detection rows require exactly four bbox tokens")

    parts = [OBJECT_REF_START_TOKEN, str(desc)]
    if contract.include_object_ref_end:
        parts.append(OBJECT_REF_END_TOKEN)
    parts.append(BOX_START_TOKEN)
    parts.extend(str(token) for token in bbox_tokens)
    if contract.include_box_end:
        parts.append(BOX_END_TOKEN)
    parts.append(contract.canonical_final_separator)
    return "".join(parts)


def required_trainable_token_count(template_id: str) -> int:
    return len(required_trainable_token_row_ids(template_id))


def required_trainable_token_row_ids(template_id: str) -> tuple[int, ...]:
    contract = resolve_detection_template_contract(template_id)
    if not contract.is_compact:
        return ()
    return (
        *contract.required_structural_token_ids,
        *range(EXPECTED_COORD_START_ID, EXPECTED_COORD_END_ID + 1),
    )


def _compact_contract(
    *,
    template_id: Literal[
        "compact",
        "compact_box_closed",
        "compact_object_box_closed",
        "compact_object_box_closed_lines",
    ],
    include_object_ref_end: bool,
    include_box_end: bool,
    row_separator: str,
    canonical_final_separator: str,
) -> DetectionTemplateContract:
    structural_tokens = [OBJECT_REF_START_TOKEN]
    structural_token_ids = [EXPECTED_OBJECT_REF_START_ID]
    if include_object_ref_end:
        structural_tokens.append(OBJECT_REF_END_TOKEN)
        structural_token_ids.append(EXPECTED_OBJECT_REF_END_ID)
    structural_tokens.append(BOX_START_TOKEN)
    structural_token_ids.append(EXPECTED_BOX_START_ID)
    if include_box_end:
        structural_tokens.append(BOX_END_TOKEN)
        structural_token_ids.append(EXPECTED_BOX_END_ID)

    pattern = render_compact_contract_row(
        DetectionTemplateContract(
            template_id=template_id,
            is_compact=True,
            include_object_ref_end=include_object_ref_end,
            include_box_end=include_box_end,
            row_separator=row_separator,
            canonical_final_separator=canonical_final_separator,
            required_structural_tokens=tuple(structural_tokens),
            required_structural_token_ids=tuple(structural_token_ids),
            prompt_pattern="",
        ),
        desc="{desc}",
        bbox_tokens=(
            "<|coord_x1|>",
            "<|coord_y1|>",
            "<|coord_x2|>",
            "<|coord_y2|>",
        ),
    )

    return DetectionTemplateContract(
        template_id=template_id,
        is_compact=True,
        include_object_ref_end=include_object_ref_end,
        include_box_end=include_box_end,
        row_separator=row_separator,
        canonical_final_separator=canonical_final_separator,
        required_structural_tokens=tuple(structural_tokens),
        required_structural_token_ids=tuple(structural_token_ids),
        prompt_pattern=pattern,
    )


__all__ = [
    "BOX_END_TOKEN",
    "COMPACT_TEMPLATE_IDS",
    "DetectionTemplateContract",
    "DetectionTemplateId",
    "OBJECT_REF_END_TOKEN",
    "STAGE1_JSON_PRETTY_TEMPLATE_ID",
    "SUPPORTED_DETECTION_TEMPLATE_IDS",
    "is_compact_template_id",
    "render_compact_contract_row",
    "required_trainable_token_count",
    "required_trainable_token_row_ids",
    "resolve_detection_template_contract",
]
