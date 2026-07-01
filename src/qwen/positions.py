"""Packed Qwen MRoPE position input construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import inspect
from importlib import metadata
from typing import Any

import torch

from src.common.errors import QwenForwardContractError
from src.packing.planner import PackedSequence, PackedSegment


QWEN_POSITION_ROW_COUNT = 4
QWEN_POSITION_ROW_MEANING = ("text", "temporal", "height", "width")


@dataclass(frozen=True)
class QwenPositionBoundaryValidation:
    row_meaning: tuple[str, str, str, str]
    checks: dict[str, bool]
    source_sha256: str
    transformers_version: str

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "row_meaning": list(self.row_meaning),
            "checks": dict(self.checks),
            "source_sha256": self.source_sha256,
            "transformers_version": self.transformers_version,
        }


@dataclass(frozen=True)
class QwenPositionSegmentSummary:
    pack_index: int
    segment_index: int
    example_index: int
    example_id: str
    start: int
    end: int
    length: int
    image_start: int
    image_end: int
    image_token_id: int
    image_grid_thw: tuple[int, int, int]
    merge_size: int
    text_start: int
    text_end: int
    mrope_row_max: tuple[int, int, int]

    def to_artifact_dict(self) -> dict[str, int | str | list[int]]:
        return {
            "pack_index": self.pack_index,
            "segment_index": self.segment_index,
            "example_index": self.example_index,
            "example_id": self.example_id,
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "image_start": self.image_start,
            "image_end": self.image_end,
            "image_token_id": self.image_token_id,
            "image_grid_thw": list(self.image_grid_thw),
            "merge_size": self.merge_size,
            "text_start": self.text_start,
            "text_end": self.text_end,
            "mrope_row_max": list(self.mrope_row_max),
        }


@dataclass(frozen=True)
class QwenPositionInputs:
    pack_index: int
    position_ids: torch.Tensor
    row_meaning: tuple[str, str, str, str]
    segment_boundaries: tuple[int, ...]
    reset_points: tuple[int, ...]
    max_segment_length: int
    boundary_validation: QwenPositionBoundaryValidation
    segments: tuple[QwenPositionSegmentSummary, ...]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "pack_index": self.pack_index,
            "position_ids_shape": [int(item) for item in self.position_ids.shape],
            "position_ids_dtype": str(self.position_ids.dtype),
            "row_meaning": list(self.row_meaning),
            "segment_boundaries": list(self.segment_boundaries),
            "reset_points": list(self.reset_points),
            "max_segment_length": self.max_segment_length,
            "boundary_validation": self.boundary_validation.to_artifact_dict(),
            "segments": [segment.to_artifact_dict() for segment in self.segments],
        }


def build_qwen_position_inputs(
    pack: PackedSequence,
    encoded_examples: Sequence[Any],
    *,
    expected_cu_seq_lens: Sequence[int] | None = None,
    image_token_id: int | None = None,
    device: torch.device | str | None = None,
) -> QwenPositionInputs:
    """Build explicit 4-row Qwen position ids for one packed physical row.

    Rows are `[text, temporal, height, width]`. Row 0 is segment-local
    `arange(length)`. Rows 1-3 mimic installed Qwen3-VL `get_rope_index`, but
    are computed independently for each packed segment before concatenation.
    """

    if not isinstance(pack, PackedSequence):
        raise QwenForwardContractError(
            "Qwen position construction requires a PackedSequence",
            code="qwen.position_pack_type",
            context={"value_type": type(pack).__name__},
        )
    boundary_validation = validate_qwen_4row_position_boundary()
    _validate_pack_segments(pack)
    examples_by_id = _examples_by_id(encoded_examples)
    torch_device = torch.device("cpu") if device is None else torch.device(device)
    position_ids = torch.empty(
        (QWEN_POSITION_ROW_COUNT, 1, pack.length),
        dtype=torch.long,
        device=torch_device,
    )
    summaries: list[QwenPositionSegmentSummary] = []

    for segment in pack.segments:
        example = examples_by_id.get(segment.example_id)
        if example is None:
            raise QwenForwardContractError(
                "packed segment references an encoded example that was not provided",
                code="qwen.position_example_missing",
                context={
                    "pack_index": pack.pack_index,
                    "segment_index": segment.segment_index,
                    "example_id": segment.example_id,
                },
            )
        input_ids = _validate_segment_matches_example(pack, segment, example)
        image_start, image_end = _image_span(example, segment)
        resolved_image_token_id = _validate_image_token_span(
            input_ids,
            image_start=image_start,
            image_end=image_end,
            segment=segment,
            image_token_id=image_token_id,
        )
        image_grid_thw = _image_grid_thw(example, segment)
        merge_size = _merge_size(example, segment)
        mrope_positions = _build_segment_mrope_positions(
            segment_length=segment.length,
            image_start=image_start,
            image_end=image_end,
            image_grid_thw=image_grid_thw,
            merge_size=merge_size,
            device=torch_device,
        )
        text_positions = torch.arange(segment.length, dtype=torch.long, device=torch_device)
        segment_positions = torch.cat((text_positions.view(1, -1), mrope_positions), dim=0)
        position_ids[:, 0, segment.start:segment.end] = segment_positions
        summaries.append(
            QwenPositionSegmentSummary(
                pack_index=segment.pack_index,
                segment_index=segment.segment_index,
                example_index=segment.example_index,
                example_id=segment.example_id,
                start=segment.start,
                end=segment.end,
                length=segment.length,
                image_start=image_start,
                image_end=image_end,
                image_token_id=resolved_image_token_id,
                image_grid_thw=image_grid_thw,
                merge_size=merge_size,
                text_start=int(text_positions[0].item()),
                text_end=int(text_positions[-1].item()),
                mrope_row_max=tuple(
                    int(value.item()) for value in mrope_positions.max(dim=1).values
                ),
            )
        )

    segment_boundaries = _segment_boundaries(pack)
    reset_points = tuple(segment.start for segment in pack.segments)
    _validate_position_ids(
        position_ids,
        pack=pack,
        segment_boundaries=segment_boundaries,
        expected_cu_seq_lens=expected_cu_seq_lens,
    )
    return QwenPositionInputs(
        pack_index=pack.pack_index,
        position_ids=position_ids,
        row_meaning=QWEN_POSITION_ROW_MEANING,
        segment_boundaries=segment_boundaries,
        reset_points=reset_points,
        max_segment_length=max(segment.length for segment in pack.segments),
        boundary_validation=boundary_validation,
        segments=tuple(summaries),
    )


def validate_qwen_4row_position_boundary(
    *,
    forward_source: str | None = None,
) -> QwenPositionBoundaryValidation:
    """Validate the installed Qwen text model still consumes `[text,t,h,w]` rows."""

    if forward_source is None:
        forward_source = _installed_qwen_text_forward_source()
    compact_source = "".join(forward_source.split())
    checks = {
        "has_4row_branch": (
            "position_ids.ndim==3andposition_ids.shape[0]==4" in compact_source
        ),
        "splits_text_row": "text_position_ids=position_ids[0]" in compact_source,
        "uses_remaining_rows_for_rotary": "position_ids=position_ids[1:]" in compact_source,
        "routes_text_row_to_attention": "position_ids=text_position_ids" in compact_source,
        "routes_mrope_rows_to_rotary": (
            "self.rotary_emb(hidden_states,position_ids)" in compact_source
        ),
    }
    missing_checks = [name for name, passed in checks.items() if not passed]
    if missing_checks:
        raise QwenForwardContractError(
            "installed Qwen3-VL text forward no longer matches the expected "
            "4-row position boundary",
            code="qwen.position_boundary_semantics",
            context={
                "row_meaning": list(QWEN_POSITION_ROW_MEANING),
                "missing_checks": missing_checks,
                "transformers_version": _package_version("transformers"),
                "source_sha256": _sha256_text(forward_source),
            },
        )
    return QwenPositionBoundaryValidation(
        row_meaning=QWEN_POSITION_ROW_MEANING,
        checks=checks,
        source_sha256=_sha256_text(forward_source),
        transformers_version=_package_version("transformers"),
    )


def _build_segment_mrope_positions(
    *,
    segment_length: int,
    image_start: int,
    image_end: int,
    image_grid_thw: tuple[int, int, int],
    merge_size: int,
    device: torch.device,
) -> torch.Tensor:
    t, h, w = image_grid_thw
    llm_grid_t = t
    llm_grid_h = h // merge_size
    llm_grid_w = w // merge_size
    image_token_count = llm_grid_t * llm_grid_h * llm_grid_w
    observed_image_tokens = image_end - image_start
    if observed_image_tokens != image_token_count:
        raise QwenForwardContractError(
            "image placeholder span length must match Qwen merged visual-token count",
            code="qwen.position_image_token_count",
            context={
                "image_start": image_start,
                "image_end": image_end,
                "observed_image_tokens": observed_image_tokens,
                "expected_image_tokens": image_token_count,
                "image_grid_thw": list(image_grid_thw),
                "merge_size": merge_size,
            },
        )

    chunks: list[torch.Tensor] = []
    st_idx = _next_mrope_start(chunks)
    if image_start:
        chunks.append(
            torch.arange(image_start, dtype=torch.long, device=device)
            .view(1, -1)
            .expand(3, -1)
            + st_idx
        )

    t_index = (
        torch.arange(llm_grid_t, dtype=torch.long, device=device)
        .view(-1, 1)
        .expand(-1, llm_grid_h * llm_grid_w)
        .flatten()
    )
    h_index = (
        torch.arange(llm_grid_h, dtype=torch.long, device=device)
        .view(1, -1, 1)
        .expand(llm_grid_t, -1, llm_grid_w)
        .flatten()
    )
    w_index = (
        torch.arange(llm_grid_w, dtype=torch.long, device=device)
        .view(1, 1, -1)
        .expand(llm_grid_t, llm_grid_h, -1)
        .flatten()
    )
    chunks.append(torch.stack((t_index, h_index, w_index), dim=0) + image_start + st_idx)

    trailing_text_len = segment_length - image_end
    if trailing_text_len:
        st_idx = _next_mrope_start(chunks)
        chunks.append(
            torch.arange(trailing_text_len, dtype=torch.long, device=device)
            .view(1, -1)
            .expand(3, -1)
            + st_idx
        )

    mrope_positions = torch.cat(chunks, dim=1)
    if tuple(mrope_positions.shape) != (3, segment_length):
        raise QwenForwardContractError(
            "segment MRoPE positions must have shape [3, segment_length]",
            code="qwen.position_mrope_shape",
            context={
                "observed_shape": [int(item) for item in mrope_positions.shape],
                "segment_length": segment_length,
            },
        )
    return mrope_positions


def _next_mrope_start(chunks: list[torch.Tensor]) -> int:
    if not chunks:
        return 0
    return int(chunks[-1].max().item()) + 1


def _examples_by_id(encoded_examples: Sequence[Any]) -> dict[str, Any]:
    examples: dict[str, Any] = {}
    for example in encoded_examples:
        example_id = getattr(example, "example_id", None)
        if not isinstance(example_id, str) or not example_id:
            raise QwenForwardContractError(
                "encoded example must expose a non-empty example_id",
                code="qwen.position_example_id",
                context={"value_type": type(example_id).__name__},
            )
        if example_id in examples:
            raise QwenForwardContractError(
                "encoded example ids must be unique for Qwen position construction",
                code="qwen.position_duplicate_example_id",
                context={"example_id": example_id},
            )
        examples[example_id] = example
    return examples


def _validate_pack_segments(pack: PackedSequence) -> None:
    if pack.length <= 0 or not pack.segments:
        raise QwenForwardContractError(
            "Qwen position construction requires a non-empty packed sequence",
            code="qwen.position_empty_pack",
            context={"pack_index": pack.pack_index, "length": pack.length},
        )
    expected_start = 0
    for segment in pack.segments:
        if (
            segment.pack_index != pack.pack_index
            or segment.start != expected_start
            or segment.end <= segment.start
        ):
            raise QwenForwardContractError(
                "packed segment table must be contiguous and match the pack",
                code="qwen.position_segment_table",
                context={
                    "pack_index": pack.pack_index,
                    "segment_index": segment.segment_index,
                    "segment_pack_index": segment.pack_index,
                    "start": segment.start,
                    "end": segment.end,
                    "expected_start": expected_start,
                },
            )
        expected_start = segment.end
    if expected_start != pack.length:
        raise QwenForwardContractError(
            "packed segment table must end at pack length",
            code="qwen.position_segment_table",
            context={
                "pack_index": pack.pack_index,
                "last_segment_end": expected_start,
                "pack_length": pack.length,
            },
        )


def _installed_qwen_text_forward_source() -> str:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    try:
        return inspect.getsource(Qwen3VLTextModel.forward)
    except (OSError, TypeError) as exc:
        raise QwenForwardContractError(
            "could not inspect installed Qwen3-VL text forward source",
            code="qwen.position_boundary_source",
            context={"transformers_version": _package_version("transformers")},
            cause=exc,
        ) from exc


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _validate_segment_matches_example(
    pack: PackedSequence,
    segment: PackedSegment,
    example: Any,
) -> tuple[int, ...]:
    input_ids = _input_ids(example, segment)
    if len(input_ids) != segment.length:
        raise QwenForwardContractError(
            "encoded example length must match packed segment length",
            code="qwen.position_segment_length",
            context={
                "example_id": segment.example_id,
                "segment_length": segment.length,
                "encoded_length": len(input_ids),
            },
        )
    packed_slice = tuple(int(token_id) for token_id in pack.input_ids[segment.start:segment.end])
    if packed_slice != input_ids:
        raise QwenForwardContractError(
            "packed segment input ids must match the encoded example",
            code="qwen.position_segment_input_ids",
            context={
                "example_id": segment.example_id,
                "segment_index": segment.segment_index,
                "segment_start": segment.start,
                "segment_end": segment.end,
            },
        )
    return input_ids


def _input_ids(example: Any, segment: PackedSegment) -> tuple[int, ...]:
    value = getattr(example, "input_ids", None)
    if not isinstance(value, tuple):
        raise QwenForwardContractError(
            "encoded example input_ids must be a tuple for Qwen positions",
            code="qwen.position_input_ids",
            context={"example_id": segment.example_id, "value_type": type(value).__name__},
        )
    return tuple(int(token_id) for token_id in value)


def _image_span(example: Any, segment: PackedSegment) -> tuple[int, int]:
    try:
        image_start = int(getattr(example, "image_pad_physical_start"))
        image_end = int(getattr(example, "image_pad_physical_end"))
    except (TypeError, ValueError) as exc:
        raise QwenForwardContractError(
            "encoded example must expose image placeholder physical span",
            code="qwen.position_image_span",
            context={"example_id": segment.example_id},
            cause=exc,
        ) from exc
    if image_start < 0 or image_end <= image_start or image_end > segment.length:
        raise QwenForwardContractError(
            "image placeholder span must stay inside the packed segment",
            code="qwen.position_image_span",
            context={
                "example_id": segment.example_id,
                "segment_length": segment.length,
                "image_start": image_start,
                "image_end": image_end,
            },
        )
    return image_start, image_end


def _validate_image_token_span(
    input_ids: tuple[int, ...],
    *,
    image_start: int,
    image_end: int,
    segment: PackedSegment,
    image_token_id: int | None,
) -> int:
    span_ids = input_ids[image_start:image_end]
    if not span_ids:
        raise QwenForwardContractError(
            "image placeholder span must contain at least one token",
            code="qwen.position_image_token_span",
            context={
                "example_id": segment.example_id,
                "image_start": image_start,
                "image_end": image_end,
            },
        )
    if image_token_id is None:
        resolved_image_token_id = span_ids[0]
    else:
        resolved_image_token_id = int(image_token_id)
    if any(token_id != resolved_image_token_id for token_id in span_ids):
        raise QwenForwardContractError(
            "image placeholder span must be a contiguous run of one image token id",
            code="qwen.position_image_token_span",
            context={
                "example_id": segment.example_id,
                "image_start": image_start,
                "image_end": image_end,
                "image_token_id": resolved_image_token_id,
            },
        )
    if image_start > 0 and input_ids[image_start - 1] == resolved_image_token_id:
        raise QwenForwardContractError(
            "image placeholder span starts after the actual image token run",
            code="qwen.position_image_token_span",
            context={
                "example_id": segment.example_id,
                "image_start": image_start,
                "image_token_id": resolved_image_token_id,
            },
        )
    if image_end < len(input_ids) and input_ids[image_end] == resolved_image_token_id:
        raise QwenForwardContractError(
            "image placeholder span ends before the actual image token run",
            code="qwen.position_image_token_span",
            context={
                "example_id": segment.example_id,
                "image_end": image_end,
                "image_token_id": resolved_image_token_id,
            },
        )
    return resolved_image_token_id


def _image_grid_thw(example: Any, segment: PackedSegment) -> tuple[int, int, int]:
    image_encoding = getattr(example, "image_encoding", None)
    value = getattr(image_encoding, "image_grid_thw", None)
    if value is None:
        raise QwenForwardContractError(
            "encoded example image encoding must expose image_grid_thw",
            code="qwen.position_image_grid",
            context={"example_id": segment.example_id},
        )
    try:
        image_grid_thw = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise QwenForwardContractError(
            "image_grid_thw must contain integer temporal, height, and width values",
            code="qwen.position_image_grid",
            context={"example_id": segment.example_id, "image_grid_thw": value},
            cause=exc,
        ) from exc
    if len(image_grid_thw) != 3 or any(item <= 0 for item in image_grid_thw):
        raise QwenForwardContractError(
            "image_grid_thw must contain positive temporal, height, and width values",
            code="qwen.position_image_grid",
            context={"example_id": segment.example_id, "image_grid_thw": list(image_grid_thw)},
        )
    return image_grid_thw


def _merge_size(example: Any, segment: PackedSegment) -> int:
    image_encoding = getattr(example, "image_encoding", None)
    plan = getattr(image_encoding, "plan", None)
    value = getattr(plan, "merge_size", None)
    if value is None:
        value = getattr(image_encoding, "merge_size", None)
    if value is None:
        raise QwenForwardContractError(
            "encoded example image encoding must expose processor-derived merge_size",
            code="qwen.position_merge_size",
            context={"example_id": segment.example_id},
        )
    try:
        merge_size = int(value)
    except (TypeError, ValueError) as exc:
        raise QwenForwardContractError(
            "processor-derived merge_size must be an integer",
            code="qwen.position_merge_size",
            context={"example_id": segment.example_id, "merge_size": value},
            cause=exc,
        ) from exc
    if merge_size <= 0:
        raise QwenForwardContractError(
            "processor-derived merge_size must be positive",
            code="qwen.position_merge_size",
            context={"example_id": segment.example_id, "merge_size": merge_size},
        )
    t, h, w = _image_grid_thw(example, segment)
    if h % merge_size != 0 or w % merge_size != 0:
        raise QwenForwardContractError(
            "image_grid_thw spatial dimensions must divide by merge_size",
            code="qwen.position_image_grid_merge",
            context={
                "example_id": segment.example_id,
                "image_grid_thw": [t, h, w],
                "merge_size": merge_size,
            },
        )
    return merge_size


def _segment_boundaries(pack: PackedSequence) -> tuple[int, ...]:
    return tuple([0, *[segment.end for segment in pack.segments]])


def _validate_position_ids(
    position_ids: torch.Tensor,
    *,
    pack: PackedSequence,
    segment_boundaries: tuple[int, ...],
    expected_cu_seq_lens: Sequence[int] | None,
) -> None:
    observed_shape = tuple(int(item) for item in position_ids.shape)
    expected_shape = (QWEN_POSITION_ROW_COUNT, 1, pack.length)
    if observed_shape != expected_shape:
        raise QwenForwardContractError(
            "Qwen packed position_ids must have shape [4, 1, pack_length]",
            code="qwen.position_shape",
            context={
                "observed_shape": list(observed_shape),
                "expected_shape": list(expected_shape),
            },
        )
    if position_ids.dtype != torch.long:
        raise QwenForwardContractError(
            "Qwen packed position_ids must use torch.long dtype",
            code="qwen.position_dtype",
            context={"dtype": str(position_ids.dtype)},
        )
    for segment in pack.segments:
        text_slice = position_ids[0, 0, segment.start:segment.end]
        expected_text = torch.arange(segment.length, dtype=torch.long, device=position_ids.device)
        if not torch.equal(text_slice, expected_text):
            raise QwenForwardContractError(
                "Qwen text-position row must reset at every packed segment boundary",
                code="qwen.position_text_reset",
                context={
                    "pack_index": pack.pack_index,
                    "segment_index": segment.segment_index,
                    "segment_start": segment.start,
                    "segment_end": segment.end,
                },
            )
    if expected_cu_seq_lens is not None:
        try:
            cu_seq_lens = tuple(int(item) for item in expected_cu_seq_lens)
        except (TypeError, ValueError) as exc:
            raise QwenForwardContractError(
                "FA2 cumulative sequence lengths must contain integer boundaries",
                code="qwen.position_cu_seq_lens",
                context={"expected_cu_seq_lens": _safe_context_value(expected_cu_seq_lens)},
                cause=exc,
            ) from exc
        if cu_seq_lens != segment_boundaries:
            raise QwenForwardContractError(
                "Qwen position reset boundaries must match FA2 cumulative sequence lengths",
                code="qwen.position_boundary_mismatch",
                context={
                    "pack_index": pack.pack_index,
                    "segment_boundaries": list(segment_boundaries),
                    "expected_cu_seq_lens": list(cu_seq_lens),
                },
            )


def _safe_context_value(value: Any) -> Any:
    try:
        return list(value)
    except TypeError:
        return {"value_type": type(value).__name__, "value": repr(value)}


__all__ = [
    "QWEN_POSITION_ROW_COUNT",
    "QWEN_POSITION_ROW_MEANING",
    "QwenPositionInputs",
    "QwenPositionBoundaryValidation",
    "QwenPositionSegmentSummary",
    "build_qwen_position_inputs",
    "validate_qwen_4row_position_boundary",
]
