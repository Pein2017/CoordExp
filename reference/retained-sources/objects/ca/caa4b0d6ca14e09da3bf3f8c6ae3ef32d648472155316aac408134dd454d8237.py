"""Qwen-native multimodal preparation and exact replay, without a model lifecycle.

The HF adapter shares these mechanics and retains its historical error codes.
Research callers request identity checks explicitly; ordinary preparation does
not hash media or require an inference configuration.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image

from src.common.errors import RuntimeContractError
from src.qwen.images import apply_logical_image_transform, rgb_image_sha256

_QWEN_ROPE_OWNER_MAX_MODEL_DEPTH = 8
_STALE_HISTORY_FIELDS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "position_ids",
        "token_type_ids",
        "cache_position",
        "rope_deltas",
        "past_key_values",
        "inputs_embeds",
    }
)


@dataclass(frozen=True)
class NativeRequest:
    request_id: str
    chat_text: str
    image: str | Path | Image.Image
    expected_token_ids: tuple[int, ...] | None = None
    expected_image_grid: tuple[int, int, int] | None = None
    expected_image_size: tuple[int, int] | None = None
    image_sha256: str | None = None
    logical_transform: str = "identity"


@dataclass(frozen=True)
class NativeBatch:
    inputs: Mapping[str, Any]
    request_ids: tuple[str, ...]
    media_sha256: tuple[str, ...] | None = None

    @property
    def prompt_token_ids(self) -> tuple[tuple[int, ...], ...]:
        return unpadded_token_rows(
            self.inputs["input_ids"], self.inputs.get("attention_mask")
        )

    @property
    def image_grids(self) -> tuple[tuple[int, int, int] | None, ...]:
        return observed_image_grids(
            self.inputs.get("image_grid_thw"), batch_size=len(self.request_ids)
        )


def _open_image(request: NativeRequest) -> Image.Image:
    if isinstance(request.image, Image.Image):
        if request.image_sha256 is not None:
            raise ValueError("byte identity requires an image path")
        image = request.image.copy()
    else:
        path = Path(request.image)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise RuntimeContractError(
                "could not reopen request image",
                code="hf_backend.image_read",
                cause=exc,
            ) from exc
        if request.image_sha256 is not None:
            observed = hashlib.sha256(data).hexdigest()
            if observed != request.image_sha256:
                raise RuntimeContractError(
                    "request image bytes changed before native projection",
                    code="hf_backend.image_sha256_mismatch",
                    context={
                        "request_id": request.request_id,
                        "expected_sha256": request.image_sha256,
                        "observed_sha256": observed,
                    },
                )
        try:
            image = Image.open(BytesIO(data))
        except OSError as exc:
            raise RuntimeContractError(
                "could not decode request image",
                code="hf_backend.image_decode",
                cause=exc,
            ) from exc
    try:
        if (
            request.expected_image_size is not None
            and image.size != request.expected_image_size
        ):
            raise RuntimeContractError(
                "request image dimensions changed before native projection",
                code="hf_backend.image_dimensions_mismatch",
                context={
                    "request_id": request.request_id,
                    "expected": list(request.expected_image_size),
                    "observed": list(image.size),
                },
            )
        rgb = image.convert("RGB")
        try:
            result = apply_logical_image_transform(
                rgb,
                request.logical_transform,
                example_id=request.request_id,
                image_path=Path(request.image)
                if not isinstance(request.image, Image.Image)
                else "<in-memory>",
            )
        except BaseException:
            rgb.close()
            raise
        if result is not rgb:
            rgb.close()
        return result
    except OSError as exc:
        raise RuntimeContractError(
            "could not decode request image", code="hf_backend.image_decode", cause=exc
        ) from exc
    finally:
        image.close()


def prepare_native_inputs(
    processor: Any,
    requests: Sequence[NativeRequest],
    *,
    device: torch.device | str = "cpu",
    record_media_identity: bool = False,
) -> NativeBatch:
    requests = tuple(requests)
    if not requests or len({r.request_id for r in requests}) != len(requests):
        raise ValueError(
            "native preparation requires nonempty uniquely identified requests"
        )
    configure_left_padding(
        processor=processor, tokenizer=getattr(processor, "tokenizer", None)
    )
    images: list[Image.Image] = []
    try:
        for request in requests:
            images.append(_open_image(request))
        media = (
            tuple(rgb_image_sha256(image) for image in images)
            if record_media_identity
            else None
        )
        encoded = processor(
            text=[r.chat_text for r in requests],
            images=images,
            padding=True,
            return_tensors="pt",
            do_resize=False,
        )
    finally:
        for image in images:
            image.close()
    try:
        native = dict(encoded)
    except (TypeError, ValueError) as exc:
        raise RuntimeContractError(
            "processor output is not mapping-like",
            code="hf_backend.processor_output",
            cause=exc,
        ) from exc
    ids = _require_rank_two_tensor(native.get("input_ids"), field="input_ids")
    if ids.shape[0] != len(requests):
        raise RuntimeContractError(
            "processor batch shape differs", code="hf_backend.processor_batch_shape"
        )
    rows = unpadded_token_rows(ids, native.get("attention_mask"))
    grids = observed_image_grids(native.get("image_grid_thw"), batch_size=len(requests))
    for request, row, grid in zip(requests, rows, grids, strict=True):
        if request.expected_token_ids is not None and row != request.expected_token_ids:
            raise RuntimeContractError(
                "executed prompt ids differ from expected expansion",
                code="hf_backend.prompt_token_mismatch",
                context={"request_id": request.request_id},
            )
        if (
            request.expected_image_grid is not None
            and grid != request.expected_image_grid
        ):
            raise RuntimeContractError(
                "processor image grid differs from expected plan",
                code="hf_backend.image_grid_mismatch",
                context={"request_id": request.request_id},
            )
    return NativeBatch(
        move_to_device(native, device=torch.device(device)),
        tuple(r.request_id for r in requests),
        media,
    )


def padded_histories(
    token_rows: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = tuple(_checked_token_ids(row) for row in token_rows)
    if not rows or any(not row for row in rows):
        raise ValueError("exact histories must contain at least one token per row")
    _checked_token_ids((pad_token_id,))
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        ids[index, -len(row) :] = torch.tensor(row, dtype=torch.long, device=device)
        mask[index, -len(row) :] = 1
    return ids, mask


def _checked_token_ids(values: Sequence[int]) -> tuple[int, ...]:
    ids = tuple(values)
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids):
        raise ValueError("literal token IDs must be nonnegative integers")
    return ids


def exact_history_inputs(
    model: Any,
    native_inputs: Mapping[str, Any],
    token_rows: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    logits_to_keep: int = 0,
) -> dict[str, Any]:
    if (
        isinstance(logits_to_keep, bool)
        or not isinstance(logits_to_keep, int)
        or logits_to_keep < 0
    ):
        raise ValueError("logits_to_keep must be a nonnegative integer")
    for row in token_rows:
        if not _checked_token_ids(row):
            raise ValueError("exact histories must contain at least one token per row")
    original_ids = _require_rank_two_tensor(
        native_inputs.get("input_ids"), field="input_ids"
    )
    if original_ids.shape[0] != len(token_rows):
        raise ValueError("exact history rows must match the prepared image/text batch")
    device = model_device(model)
    ids, mask = padded_histories(token_rows, pad_token_id=pad_token_id, device=device)
    kwargs = move_to_device(
        {k: v for k, v in native_inputs.items() if k not in _STALE_HISTORY_FIELDS},
        device=device,
    )
    grid = kwargs.get("image_grid_thw")
    if not isinstance(grid, torch.Tensor):
        raise RuntimeContractError(
            "exact replay requires image_grid_thw",
            code="hf_backend.exact_history_image_grid",
        )
    positions = derive_position_ids(
        model=model,
        input_ids=ids,
        attention_mask=mask,
        image_grid_thw=grid,
        video_grid_thw=kwargs.get("video_grid_thw"),
    )
    kwargs.update(
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions,
        use_cache=False,
        return_dict=True,
        logits_to_keep=logits_to_keep,
    )
    return kwargs


def select_compact_replay_logits(
    logits: torch.Tensor,
    continuation_lengths: Sequence[int],
) -> list[torch.Tensor]:
    """Select each left-padded continuation's causal rows from compact logits."""

    lengths = tuple(continuation_lengths)
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
        or logits.shape[0] != len(lengths)
    ):
        raise ValueError("compact replay logits do not match the continuation batch")
    if not lengths or any(
        isinstance(count, bool) or not isinstance(count, int) or count <= 0
        for count in lengths
    ):
        raise ValueError("continuation lengths must be positive integers")
    width = max(lengths) + 1
    if logits.shape[1] != width:
        raise ValueError("compact replay logits have the wrong trailing history width")
    rows: list[torch.Tensor] = []
    for index, count in enumerate(lengths):
        start = width - count - 1
        selected = logits[index, start : width - 1]
        if selected.shape[0] != count:
            raise ValueError("compact replay logits do not cover an exact continuation")
        rows.append(selected.float())
    return rows


@dataclass(frozen=True)
class ExactReplay:
    inputs: Mapping[str, Any]
    target_ids: torch.Tensor
    prompt_length: int

    def aligned_logits(self, logits: torch.Tensor) -> torch.Tensor:
        count = self.target_ids.numel()
        compact = self.inputs["logits_to_keep"] != 0
        expected = count + 1 if compact else self.prompt_length + count
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 3
            or logits.shape[0] != 1
            or logits.shape[1] != expected
        ):
            raise RuntimeContractError(
                "replay logits do not cover the exact action",
                code="hf_backend.teacher_forced_alignment",
            )
        start = 0 if compact else self.prompt_length - 1
        return logits[0, start : start + count]


def prepare_replay(
    model: Any,
    native_inputs: Mapping[str, Any],
    *,
    prompt_token_ids: Sequence[int],
    continuation_token_ids: Sequence[int],
    compact_logits: bool = True,
) -> ExactReplay:
    prompt = _checked_token_ids(prompt_token_ids)
    continuation = _checked_token_ids(continuation_token_ids)
    if not prompt or not continuation:
        raise ValueError("exact replay requires nonempty prompt and continuation")
    kwargs = exact_history_inputs(
        model,
        native_inputs,
        [(*prompt, *continuation)],
        pad_token_id=0,
        logits_to_keep=len(continuation) + 1 if compact_logits else 0,
    )
    return ExactReplay(
        kwargs,
        torch.tensor(continuation, dtype=torch.long, device=kwargs["input_ids"].device),
        len(prompt),
    )


def resolve_rope_index(model: Any) -> Callable[..., tuple[Any, Any]]:
    """Bind the real Qwen ``get_rope_index`` beneath wrapper ``.model`` levels.

    Adapter wrappers (for example PEFT) and the transformers Qwen3-VL layout
    interpose a session-dependent number of ``.model`` levels above the module
    that owns exact multimodal MRoPE derivation; the owner is located, never
    reimplemented.
    """

    candidate = model
    searched: list[str] = []
    seen_ids: set[int] = set()
    while (
        candidate is not None
        and id(candidate) not in seen_ids
        and len(searched) < _QWEN_ROPE_OWNER_MAX_MODEL_DEPTH
    ):
        seen_ids.add(id(candidate))
        searched.append(type(candidate).__name__)
        get_rope_index_value = getattr(candidate, "get_rope_index", None)
        if callable(get_rope_index_value):
            return cast(Callable[..., tuple[Any, Any]], get_rope_index_value)
        candidate = getattr(candidate, "model", None)
    raise RuntimeContractError(
        "HF model does not expose Qwen get_rope_index",
        code="hf_backend.position_ids_unavailable",
        context={"searched_model_chain": searched},
    )


def derive_position_ids(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: torch.Tensor,
    video_grid_thw: torch.Tensor | None,
) -> torch.Tensor:
    get_rope_index = resolve_rope_index(model)
    try:
        with torch.no_grad():
            position_ids, _rope_deltas = get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask,
            )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeContractError(
            "HF model failed to derive Qwen position IDs",
            code="hf_backend.position_ids_invalid",
            cause=exc,
        ) from exc
    if (
        not isinstance(position_ids, torch.Tensor)
        or position_ids.ndim != 3
        or position_ids.shape[1:] != input_ids.shape
    ):
        raise RuntimeContractError(
            "Qwen get_rope_index returned invalid position IDs",
            code="hf_backend.position_ids_invalid",
            context={
                "value_type": type(position_ids).__name__,
                "shape": (
                    tuple(position_ids.shape)
                    if isinstance(position_ids, torch.Tensor)
                    else None
                ),
            },
        )
    # A caller may have prepared inputs under inference_mode. Return ordinary
    # integer tensors that can safely be saved by a later differentiable forward.
    with torch.inference_mode(False):
        return position_ids.detach().clone()


def configure_left_padding(*, processor: Any, tokenizer: Any) -> None:
    """Keep heterogeneous decoder-only batches aligned with the legacy HF path."""

    candidates = (tokenizer, getattr(processor, "tokenizer", None))
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        try:
            candidate.padding_side = "left"
        except (AttributeError, TypeError) as exc:
            raise RuntimeContractError(
                "HF tokenizer does not permit decoder-only left padding",
                code="hf_backend.padding_side_unsupported",
                context={"tokenizer_class": type(candidate).__name__},
                cause=exc,
            ) from exc
        if getattr(candidate, "padding_side", None) != "left":
            raise RuntimeContractError(
                "HF tokenizer did not retain decoder-only left padding",
                code="hf_backend.padding_side_unsupported",
                context={"tokenizer_class": type(candidate).__name__},
            )


def unpadded_token_rows(
    input_ids: torch.Tensor,
    attention_mask: Any,
) -> tuple[tuple[int, ...], ...]:
    if attention_mask is None:
        return tuple(tuple(int(value) for value in row.tolist()) for row in input_ids)
    mask = _require_rank_two_tensor(attention_mask, field="attention_mask")
    if mask.shape != input_ids.shape:
        raise RuntimeContractError(
            "HF processor attention mask shape does not match input_ids",
            code="hf_backend.processor_attention_shape",
            context={
                "input_shape": tuple(input_ids.shape),
                "mask_shape": tuple(mask.shape),
            },
        )
    return tuple(
        tuple(
            int(value)
            for value, keep in zip(row.tolist(), row_mask.tolist(), strict=True)
            if keep
        )
        for row, row_mask in zip(input_ids, mask, strict=True)
    )


def observed_image_grids(
    value: Any,
    *,
    batch_size: int,
) -> tuple[tuple[int, int, int] | None, ...]:
    if value is None:
        return tuple(None for _ in range(batch_size))
    tensor = _require_rank_two_tensor(value, field="image_grid_thw")
    if tensor.shape != (batch_size, 3):
        raise RuntimeContractError(
            "HF processor image_grid_thw shape does not match requests",
            code="hf_backend.image_grid_shape",
            context={"shape": tuple(tensor.shape), "batch_size": batch_size},
        )
    return tuple(tuple(int(item) for item in row.tolist()) for row in tensor)


def model_device(model: Any) -> torch.device:
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        first = next(iter(parameters()), None)
        if first is not None:
            return torch.device(first.device)
    return torch.device("cpu")


def move_to_device(value: Any, *, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        if torch.is_inference(value):
            with torch.inference_mode(False):
                return value.to(device).clone()
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device=device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device=device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device=device) for item in value]
    return value


def _require_rank_two_tensor(value: Any, *, field: str) -> torch.Tensor:
    if value is None:
        raise RuntimeContractError(
            f"HF {field} is missing",
            code="hf_backend.tensor_missing",
            context={"field": field},
        )
    try:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeContractError(
            f"HF {field} is not tensor-like",
            code="hf_backend.tensor_type",
            context={"field": field, "value_type": type(value).__name__},
            cause=exc,
        ) from exc
    if tensor.ndim != 2:
        raise RuntimeContractError(
            f"HF {field} must be rank two",
            code="hf_backend.tensor_shape",
            context={"field": field, "shape": tuple(tensor.shape)},
        )
    return tensor
