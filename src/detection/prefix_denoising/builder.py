from __future__ import annotations

import copy
import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, MutableMapping

from src.common.detection_chat import build_detection_chat_messages
from src.config.schema import PrefixDenoisingConfig
from src.datasets.geometry import BBoxNoiseConfig, construct_valid_norm1000_bbox_noise
from src.detection.data import (
    CoordinateTokenBox,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
    parse_raw_detection_row,
)
from src.detection.dataset import _encode_swift_template_no_resize
from src.detection.scene import (
    detection_scene_from_raw_row,
    normalized_detection_sample_from_scene,
)
from src.detection.template import RenderedDetectionSequence, get_detection_template

from .types import (
    CoordSlot,
    HybridPrefixDenoisingSample,
    PrefixDenoisingPackingEstimate,
    PrefixDenoisingKLSite,
    PrefixDenoisingSegment,
)

_COORD_SLOTS: tuple[CoordSlot, ...] = ("x1", "y1", "x2", "y2")
_CORE_ENCODED_KEYS = {"input_ids", "labels", "attention_mask", "length"}
PREFIX_DENOISING_FAST_ESTIMATOR_VERSION = "qwen3_vl_chat_template_alignment_v2"


def prefix_denoising_fast_estimator_fingerprint(swift_template: Any) -> dict[str, Any]:
    """Fingerprint fast-estimator inputs that affect static packing eligibility."""

    tokenizer = getattr(swift_template, "tokenizer", None)
    chat_template = getattr(tokenizer, "chat_template", None)
    patch_size, merge_size = _resolve_qwen_vl_patch_merge(swift_template)
    return {
        "schema_version": PREFIX_DENOISING_FAST_ESTIMATOR_VERSION,
        "tokenizer_chat_template_sha256": (
            hashlib.sha256(str(chat_template).encode("utf-8")).hexdigest()
            if chat_template is not None
            else None
        ),
        "tokenizer_chat_template_present": chat_template is not None,
        "qwen_vl_patch_size": int(patch_size),
        "qwen_vl_merge_size": int(merge_size),
    }


def build_hybrid_prefix_denoising_sample(
    row: Mapping[str, Any],
    *,
    base_sample_id: str,
    image_root: str | Path,
    swift_template: Any,
    user_prompt: str,
    system_prompt: str | None,
    prefix_denoising: PrefixDenoisingConfig,
    epoch: int,
    rng: random.Random,
    max_length: int,
) -> HybridPrefixDenoisingSample:
    """Build one clean/noisy full-sequence hybrid sample.

    This is intentionally limited to V1 materialization: paired full branches and
    optional selected-site KL metadata. Packing and loss resolution consume the
    returned sidecar later.
    """

    hybrid_sample_id = f"{base_sample_id}:prefix_denoising:e{int(epoch)}"
    if not prefix_denoising.enabled:
        return _skip_sample(
            base_sample_id=base_sample_id,
            hybrid_sample_id=hybrid_sample_id,
            reason="prefix_denoising_disabled",
        )
    if _raw_object_count(row) == 0:
        return _skip_sample(
            base_sample_id=base_sample_id,
            hybrid_sample_id=hybrid_sample_id,
            reason="zero_object_hybrid_sample",
        )

    image_reference = _resolve_single_image(row, image_root=image_root)
    raw = parse_raw_detection_row(row)

    scene = detection_scene_from_raw_row(
        raw,
        object_ordering=ObjectOrderingPlan.sorted(
            seed_source="prefix_denoising_hybrid_builder"
        ),
        image_reference=image_reference,
    )
    clean_sample = normalized_detection_sample_from_scene(scene)
    if not clean_sample.objects:
        return _skip_sample(
            base_sample_id=base_sample_id,
            hybrid_sample_id=hybrid_sample_id,
            reason="zero_object_hybrid_sample",
        )

    noised = _build_noisy_objects(
        clean_sample.objects,
        prefix_denoising=prefix_denoising,
        rng=rng,
    )
    if not noised["ok"]:
        return _skip_sample(
            base_sample_id=base_sample_id,
            hybrid_sample_id=hybrid_sample_id,
            reason=str(noised["skip_reason"]),
            metadata={"object_index": noised.get("object_index")},
        )

    noisy_sample = replace(
        clean_sample,
        objects=tuple(noised["objects"]),
    )
    template = get_detection_template("compact_full")
    clean_rendered = template.render_assistant(clean_sample)
    noisy_rendered = template.render_assistant(noisy_sample)
    clean_encoded = _encode_branch(
        swift_template=swift_template,
        images=scene.images,
        assistant_text=clean_rendered.text,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
    )
    noisy_encoded = _encode_branch(
        swift_template=swift_template,
        images=scene.images,
        assistant_text=noisy_rendered.text,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
    )

    clean_input_ids = _as_int_tuple(clean_encoded.get("input_ids"), "clean.input_ids")
    noisy_input_ids = _as_int_tuple(noisy_encoded.get("input_ids"), "noisy.input_ids")
    clean_labels = _masked_labels(clean_encoded.get("labels"), "clean.labels")
    clean_attention = _attention_tuple(clean_encoded, len(clean_input_ids))
    noisy_attention = _attention_tuple(noisy_encoded, len(noisy_input_ids))
    if len(clean_input_ids) != len(clean_labels):
        return _skip_sample(
            base_sample_id=base_sample_id,
            hybrid_sample_id=hybrid_sample_id,
            reason="clean_input_label_length_mismatch",
        )
    if len(clean_input_ids) != len(noisy_input_ids):
        return _skip_sample(
            base_sample_id=base_sample_id,
            hybrid_sample_id=hybrid_sample_id,
            reason="clean_noisy_length_mismatch",
            metadata={
                "clean_length": len(clean_input_ids),
                "noisy_length": len(noisy_input_ids),
            },
        )
    if len(clean_labels) != len(noisy_input_ids):
        return _skip_sample(
            base_sample_id=base_sample_id,
            hybrid_sample_id=hybrid_sample_id,
            reason="clean_label_noisy_input_length_mismatch",
        )
    total_length = len(clean_input_ids) + len(noisy_input_ids)
    if int(max_length) > 0 and total_length > int(max_length):
        return _skip_sample(
            base_sample_id=base_sample_id,
            hybrid_sample_id=hybrid_sample_id,
            reason="overlength_hybrid_sample",
            metadata={"total_length": total_length, "max_length": int(max_length)},
        )

    supervised_positions = tuple(
        index for index, label in enumerate(clean_labels) if index > 0 and label != -100
    )
    clean_segment = PrefixDenoisingSegment(
        segment_id=f"{hybrid_sample_id}:clean_full",
        branch_id="clean_full",
        input_ids=clean_input_ids,
        labels=clean_labels,
        attention_mask=clean_attention,
        supervised_positions=supervised_positions,
        ce_denominator=len(supervised_positions),
        metadata={
            "messages": tuple(clean_encoded["messages"]),
            "rendered_text": clean_rendered.text,
            "encoded_extras": _encoded_extras(clean_encoded),
        },
    )
    noisy_segment = PrefixDenoisingSegment(
        segment_id=f"{hybrid_sample_id}:noisy_full",
        branch_id="noisy_full",
        input_ids=noisy_input_ids,
        labels=clean_labels,
        attention_mask=noisy_attention,
        supervised_positions=supervised_positions,
        ce_denominator=len(supervised_positions),
        metadata={
            "messages": tuple(noisy_encoded["messages"]),
            "rendered_text": noisy_rendered.text,
            "encoded_extras": _encoded_extras(noisy_encoded),
        },
    )

    coord_positions = _locate_clean_coord_positions(
        clean_labels=clean_labels,
        clean_input_ids=clean_input_ids,
        clean_sample=clean_sample,
        rendered=clean_rendered,
        tokenizer=getattr(swift_template, "tokenizer", None),
    )
    if coord_positions is None:
        return _skip_sample(
            base_sample_id=base_sample_id,
            hybrid_sample_id=hybrid_sample_id,
            reason="coord_label_position_alignment_failed",
        )

    alignment_failure = _validate_clean_noisy_alignment(
        clean_sample=clean_sample,
        noisy_sample=noisy_sample,
        clean_input_ids=clean_input_ids,
        noisy_input_ids=noisy_input_ids,
        clean_labels=clean_labels,
        coord_positions=coord_positions,
        tokenizer=getattr(swift_template, "tokenizer", None),
    )
    if alignment_failure is not None:
        return _skip_sample(
            base_sample_id=base_sample_id,
            hybrid_sample_id=hybrid_sample_id,
            reason=alignment_failure,
        )

    kl_sites = _build_kl_sites(
        clean_segment=clean_segment,
        noisy_segment=noisy_segment,
        clean_sample=clean_sample,
        clean_input_ids=clean_input_ids,
        noisy_input_ids=noisy_input_ids,
        coord_positions=coord_positions,
        prefix_denoising=prefix_denoising,
        rng=rng,
    )
    return HybridPrefixDenoisingSample(
        ok=True,
        hybrid_sample_id=hybrid_sample_id,
        base_sample_id=base_sample_id,
        clean_full=clean_segment,
        noisy_full=noisy_segment,
        kl_sites=kl_sites,
        metadata={
            "epoch": int(epoch),
            "object_count": len(clean_sample.objects),
            "image": image_reference,
            "noising": tuple(noised["provenance"]),
        },
    )


def estimate_hybrid_prefix_denoising_packing(
    row: Mapping[str, Any],
    *,
    image_root: str | Path,
    swift_template: Any,
    user_prompt: str,
    system_prompt: str | None,
    prefix_denoising: PrefixDenoisingConfig,
    rng: random.Random,
    max_length: int,
) -> PrefixDenoisingPackingEstimate:
    """Estimate V1 hybrid sample eligibility and packed token length.

    This mirrors the full builder's row parsing, sorted order, constructive
    noising, compact-full rendering, and max-length skip policy, but it avoids
    Swift multimodal image encoding. Qwen-VL visual token count is deterministic
    from the stored image dimensions when `do_resize=false`, so this is suitable
    for static-packing precompute.
    """

    if not prefix_denoising.enabled:
        return PrefixDenoisingPackingEstimate(
            ok=False,
            skip_reason="prefix_denoising_disabled",
        )
    if _raw_object_count(row) == 0:
        return PrefixDenoisingPackingEstimate(
            ok=False,
            skip_reason="zero_object_hybrid_sample",
        )

    image_reference = _resolve_single_image(row, image_root=image_root)
    raw = parse_raw_detection_row(row)
    scene = detection_scene_from_raw_row(
        raw,
        object_ordering=ObjectOrderingPlan.sorted(
            seed_source="prefix_denoising_hybrid_builder"
        ),
        image_reference=image_reference,
    )
    clean_sample = normalized_detection_sample_from_scene(scene)
    if not clean_sample.objects:
        return PrefixDenoisingPackingEstimate(
            ok=False,
            skip_reason="zero_object_hybrid_sample",
        )

    noised = _build_noisy_objects(
        clean_sample.objects,
        prefix_denoising=prefix_denoising,
        rng=rng,
    )
    if not noised["ok"]:
        return PrefixDenoisingPackingEstimate(
            ok=False,
            skip_reason=str(noised["skip_reason"]),
            metadata={"object_index": noised.get("object_index")},
        )

    noisy_sample = replace(
        clean_sample,
        objects=tuple(noised["objects"]),
    )
    template = get_detection_template("compact_full")
    clean_rendered = template.render_assistant(clean_sample)
    noisy_rendered = template.render_assistant(noisy_sample)
    clean_token_ids = _estimate_branch_token_ids(
        swift_template=swift_template,
        images=scene.images,
        assistant_text=clean_rendered.text,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
    )
    noisy_token_ids = _estimate_branch_token_ids(
        swift_template=swift_template,
        images=scene.images,
        assistant_text=noisy_rendered.text,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
    )
    clean_length = _estimate_branch_length_from_token_ids(
        row=row,
        swift_template=swift_template,
        token_ids=clean_token_ids,
    )
    noisy_length = _estimate_branch_length_from_token_ids(
        row=row,
        swift_template=swift_template,
        token_ids=noisy_token_ids,
    )
    if clean_length != noisy_length or len(clean_token_ids) != len(noisy_token_ids):
        return PrefixDenoisingPackingEstimate(
            ok=False,
            skip_reason="clean_noisy_length_mismatch",
            metadata={
                "clean_length": int(clean_length),
                "noisy_length": int(noisy_length),
                "clean_chat_template_length": len(clean_token_ids),
                "noisy_chat_template_length": len(noisy_token_ids),
            },
        )
    tokenizer = getattr(swift_template, "tokenizer", None)
    assistant_span = _locate_rendered_text_span(
        token_ids=clean_token_ids,
        rendered_text=clean_rendered.text,
        tokenizer=tokenizer,
    )
    coord_positions = _locate_clean_coord_positions_in_range(
        clean_input_ids=clean_token_ids,
        clean_sample=clean_sample,
        tokenizer=tokenizer,
        start=assistant_span[0] if assistant_span is not None else 0,
        end=assistant_span[1] if assistant_span is not None else len(clean_token_ids),
    )
    if coord_positions is None:
        return PrefixDenoisingPackingEstimate(
            ok=False,
            skip_reason="coord_label_position_alignment_failed",
        )
    alignment_failure = _validate_clean_noisy_alignment(
        clean_sample=clean_sample,
        noisy_sample=noisy_sample,
        clean_input_ids=clean_token_ids,
        noisy_input_ids=noisy_token_ids,
        clean_labels=clean_token_ids,
        coord_positions=coord_positions,
        tokenizer=tokenizer,
    )
    if alignment_failure is not None:
        return PrefixDenoisingPackingEstimate(
            ok=False,
            skip_reason=alignment_failure,
        )

    total_length = int(clean_length + noisy_length)
    if int(max_length) > 0 and total_length > int(max_length):
        return PrefixDenoisingPackingEstimate(
            ok=False,
            skip_reason="overlength_hybrid_sample",
            metadata={"total_length": total_length, "max_length": int(max_length)},
        )

    return PrefixDenoisingPackingEstimate(
        ok=True,
        total_length=total_length,
        metadata={
            "clean_length": int(clean_length),
            "noisy_length": int(noisy_length),
            "object_count": len(clean_sample.objects),
        },
    )


def _raw_object_count(row: Mapping[str, Any]) -> int:
    objects = row.get("objects")
    if isinstance(objects, Sequence) and not isinstance(objects, (str, bytes, bytearray)):
        return len(objects)
    return -1


def _resolve_single_image(row: Mapping[str, Any], *, image_root: str | Path) -> str:
    images = row.get("images")
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes, bytearray)):
        raise ValueError("row.images must be a sequence")
    if len(images) != 1:
        raise ValueError(f"prefix-denoising expects one image, got {len(images)}")
    image_path = Path(str(images[0]))
    candidate = image_path if image_path.is_absolute() else Path(image_root) / image_path
    resolved = candidate.expanduser().resolve(strict=True)
    root = Path(image_root).expanduser().resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"image path resolves outside image_root: {resolved} (image_root={root})"
        ) from exc
    return str(resolved)


def _build_noisy_objects(
    objects: Sequence[NormalizedDetectionObject],
    *,
    prefix_denoising: PrefixDenoisingConfig,
    rng: random.Random,
) -> dict[str, Any]:
    noise_cfg = BBoxNoiseConfig(
        center_shift_frac=prefix_denoising.noise.center_shift_frac,
        uniform_scale_range=prefix_denoising.noise.uniform_scale_range,
    )
    noisy_objects: list[NormalizedDetectionObject] = []
    provenance: list[Mapping[str, object]] = []
    for index, obj in enumerate(objects):
        try:
            result = construct_valid_norm1000_bbox_noise(
                _bbox_tuple(obj.bbox_2d),
                config=noise_cfg,
                rng=rng,
                object_id=obj.object_instance_id,
            )
        except ValueError as exc:
            return {
                "ok": False,
                "skip_reason": "degenerate_gt_bbox",
                "object_index": index,
                "error": str(exc),
            }
        if not result.ok or result.noisy_bbox is None:
            return {
                "ok": False,
                "skip_reason": result.skip_reason or "noise_infeasible_valid_bbox",
                "object_index": index,
            }
        noisy_objects.append(
            replace(obj, bbox_2d=CoordinateTokenBox(*result.noisy_bbox))
        )
        provenance.append(
            {
                "object_index": index,
                "object_instance_id": obj.object_instance_id,
                "clean_bins": result.clean_bins,
                "noisy_bins": result.noisy_bins,
                "changed": result.changed,
                "provenance": result.provenance,
            }
        )
    return {
        "ok": True,
        "objects": tuple(noisy_objects),
        "provenance": tuple(provenance),
    }


def _estimate_branch_token_ids(
    *,
    swift_template: Any,
    images: Sequence[str],
    assistant_text: str,
    user_prompt: str,
    system_prompt: str | None,
) -> list[int]:
    tokenizer = getattr(swift_template, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("swift_template must expose tokenizer")
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise ValueError(
            "prefix-denoising fast length estimate requires "
            "tokenizer.apply_chat_template for Qwen3-VL chat-template parity"
        )
    messages = build_detection_chat_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=images,
        assistant_text=assistant_text,
    )
    token_ids = apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if not isinstance(token_ids, Sequence) or isinstance(
        token_ids, (str, bytes, bytearray)
    ):
        raise TypeError("tokenizer.apply_chat_template must return token ids")
    return [int(token_id) for token_id in token_ids]


def _estimate_branch_length_from_token_ids(
    *,
    row: Mapping[str, Any],
    swift_template: Any,
    token_ids: Sequence[int],
) -> int:
    tokenizer = getattr(swift_template, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("swift_template must expose tokenizer")
    image_token_id = _image_token_id(tokenizer)
    ids = [int(token_id) for token_id in token_ids]
    image_placeholders = int(ids.count(image_token_id))
    return int(
        len(ids)
        - image_placeholders
        + _estimate_qwen_vl_image_tokens(row=row, swift_template=swift_template)
    )


def _image_token_id(tokenizer: Any) -> int:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        raise ValueError("tokenizer must expose convert_tokens_to_ids")
    token_id = convert("<|image_pad|>")
    if token_id is None:
        raise ValueError("tokenizer did not resolve <|image_pad|>")
    return int(token_id)


def _coord_tokens_single_token(tokenizer: Any) -> bool:
    if tokenizer is None:
        return False
    cached = getattr(tokenizer, "_coordexp_coord_tokens_single_token", None)
    if cached is not None:
        return bool(cached)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        setattr(tokenizer, "_coordexp_coord_tokens_single_token", False)
        return False
    try:
        zero = encode("<|coord_0|>", add_special_tokens=False)
        high = encode("<|coord_999|>", add_special_tokens=False)
    except TypeError:
        zero = encode("<|coord_0|>")
        high = encode("<|coord_999|>")
    if hasattr(zero, "tolist"):
        zero = zero.tolist()
    if hasattr(high, "tolist"):
        high = high.tolist()
    ok = (
        isinstance(zero, Sequence)
        and not isinstance(zero, (str, bytes, bytearray))
        and isinstance(high, Sequence)
        and not isinstance(high, (str, bytes, bytearray))
        and len(zero) == 1
        and len(high) == 1
    )
    setattr(tokenizer, "_coordexp_coord_tokens_single_token", bool(ok))
    return bool(ok)


def _estimate_qwen_vl_image_tokens(
    *,
    row: Mapping[str, Any],
    swift_template: Any,
) -> int:
    try:
        width = int(row["width"])
        height = int(row["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("prefix-denoising fast length estimate requires width/height") from exc
    if width <= 0 or height <= 0:
        raise ValueError("prefix-denoising fast length estimate requires positive width/height")
    patch_size, merge_size = _resolve_qwen_vl_patch_merge(swift_template)
    stride = int(patch_size * merge_size)
    images = row.get("images") or ()
    image_count = max(1, len(images) if isinstance(images, Sequence) else 1)
    return int(math.ceil(height / stride) * math.ceil(width / stride) * image_count)


def _resolve_qwen_vl_patch_merge(swift_template: Any) -> tuple[int, int]:
    candidates = (
        getattr(swift_template, "image_processor", None),
        getattr(getattr(swift_template, "processor", None), "image_processor", None),
        getattr(swift_template, "processor", None),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        patch_size = getattr(candidate, "patch_size", None)
        merge_size = getattr(candidate, "merge_size", None)
        if merge_size is None:
            merge_size = getattr(candidate, "spatial_merge_size", None)
        if patch_size is not None and merge_size is not None:
            return int(patch_size), int(merge_size)
    return 16, 2


def _bbox_tuple(bbox: Any) -> tuple[int, int, int, int]:
    values_attr = getattr(bbox, "values", None)
    if values_attr is not None:
        values = tuple(int(value) for value in values_attr)
        if len(values) != 4:
            raise ValueError("bbox.values must contain four coordinates")
        return values  # type: ignore[return-value]
    if all(hasattr(bbox, field_name) for field_name in _COORD_SLOTS):
        return (
            int(getattr(bbox, "x1")),
            int(getattr(bbox, "y1")),
            int(getattr(bbox, "x2")),
            int(getattr(bbox, "y2")),
        )
    values = tuple(int(value) for value in bbox)
    if len(values) != 4:
        raise ValueError("bbox must contain four coordinates")
    return values  # type: ignore[return-value]


def _encode_branch(
    *,
    swift_template: Any,
    images: Sequence[str],
    assistant_text: str,
    user_prompt: str,
    system_prompt: str | None,
) -> MutableMapping[str, Any]:
    messages = build_detection_chat_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=images,
        assistant_text=assistant_text,
    )
    encoded = _encode_swift_template_no_resize(
        swift_template,
        {"messages": copy.deepcopy(messages)},
    )
    if not isinstance(encoded, MutableMapping):
        raise TypeError("swift_template.encode must return a mutable mapping")
    encoded["messages"] = tuple(messages)
    return encoded


def _as_int_tuple(value: Any, path: str) -> tuple[int, ...]:
    if value is None:
        raise ValueError(f"{path} is missing")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{path} must be a sequence")
    return tuple(int(item) for item in value)


def _masked_labels(value: Any, path: str) -> tuple[int, ...]:
    labels = list(_as_int_tuple(value, path))
    if labels:
        labels[0] = -100
    return tuple(labels)


def _attention_tuple(encoded: Mapping[str, Any], length: int) -> tuple[int, ...]:
    if "attention_mask" not in encoded:
        return tuple(1 for _ in range(length))
    mask = _as_int_tuple(encoded.get("attention_mask"), "encoded.attention_mask")
    if len(mask) != int(length):
        raise ValueError(
            "encoded attention_mask length must match input_ids length: "
            f"attention_mask={len(mask)}, length={int(length)}"
        )
    return mask


def _encoded_extras(encoded: Mapping[str, Any]) -> Mapping[str, object]:
    return {
        str(key): value
        for key, value in encoded.items()
        if str(key) not in _CORE_ENCODED_KEYS and str(key) != "messages"
    }


def _locate_clean_coord_positions(
    *,
    clean_labels: Sequence[int],
    clean_input_ids: Sequence[int],
    clean_sample: NormalizedDetectionSample,
    rendered: RenderedDetectionSequence,
    tokenizer: Any,
) -> tuple[tuple[int, int, int, int], ...] | None:
    del rendered
    supervised_positions = tuple(
        index
        for index, label in enumerate(clean_labels)
        if index > 0 and label != -100
    )
    cursor = 0
    object_positions: list[tuple[int, int, int, int]] = []
    for obj in clean_sample.objects:
        positions: list[int] = []
        for clean_bin in _bbox_tuple(obj.bbox_2d):
            try:
                token_id = _coord_token_id(tokenizer, int(clean_bin))
            except ValueError:
                return None
            found = None
            for index in supervised_positions:
                if index < cursor:
                    continue
                if (
                    int(clean_labels[index]) == token_id
                    and int(clean_input_ids[index]) == token_id
                ):
                    found = index
                    break
            if found is None:
                return None
            positions.append(found)
            cursor = found + 1
        object_positions.append(tuple(positions))  # type: ignore[arg-type]
    return tuple(object_positions)


def _locate_rendered_text_span(
    *,
    token_ids: Sequence[int],
    rendered_text: str,
    tokenizer: Any,
) -> tuple[int, int] | None:
    rendered_ids = _encode_text_token_ids(tokenizer, rendered_text)
    if not rendered_ids:
        return None
    max_start = len(token_ids) - len(rendered_ids)
    if max_start < 0:
        return None
    for start in range(max_start + 1):
        end = start + len(rendered_ids)
        if list(token_ids[start:end]) == rendered_ids:
            return start, end
    return None


def _encode_text_token_ids(tokenizer: Any, text: str) -> list[int]:
    if tokenizer is None:
        return []
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return []
    try:
        ids = encode(str(text), add_special_tokens=False)
    except TypeError:
        ids = encode(str(text))
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes, bytearray)):
        return []
    return [int(token_id) for token_id in ids]


def _locate_clean_coord_positions_in_range(
    *,
    clean_input_ids: Sequence[int],
    clean_sample: NormalizedDetectionSample,
    tokenizer: Any,
    start: int,
    end: int,
) -> tuple[tuple[int, int, int, int], ...] | None:
    lower = max(int(start), 0)
    upper = min(int(end), len(clean_input_ids))
    cursor = lower
    object_positions: list[tuple[int, int, int, int]] = []
    for obj in clean_sample.objects:
        positions: list[int] = []
        for clean_bin in _bbox_tuple(obj.bbox_2d):
            try:
                token_id = _coord_token_id(tokenizer, int(clean_bin))
            except ValueError:
                return None
            found = None
            for index in range(cursor, upper):
                if int(clean_input_ids[index]) == token_id:
                    found = index
                    break
            if found is None:
                return None
            positions.append(found)
            cursor = found + 1
        object_positions.append(tuple(positions))  # type: ignore[arg-type]
    return tuple(object_positions)


def _coord_token_id(tokenizer: Any, coord_bin: int) -> int:
    token = f"<|coord_{int(coord_bin)}|>"
    if tokenizer is None:
        raise ValueError("swift_template.tokenizer cannot resolve coord token ids")
    encode = getattr(tokenizer, "encode", None)
    encode_id: int | None = None
    if callable(encode):
        try:
            ids = encode(token, add_special_tokens=False)
        except TypeError:
            ids = encode(token)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes, bytearray)):
            raise ValueError(f"coord token {token!r} did not encode to a sequence")
        if len(ids) != 1:
            raise ValueError(f"coord token {token!r} must encode to exactly one id")
        encode_id = int(ids[0])
        _reject_unknown_coord_token_id(tokenizer, encode_id, token=token)

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    convert_id: int | None = None
    if callable(convert):
        raw_convert_id = convert(token)
        if raw_convert_id is None:
            raise ValueError(f"coord token {token!r} did not convert to an id")
        convert_id = int(raw_convert_id)
        _reject_unknown_coord_token_id(tokenizer, convert_id, token=token)

    if encode_id is not None and convert_id is not None and encode_id != convert_id:
        raise ValueError(
            f"coord token {token!r} encode id {encode_id} does not match "
            f"convert id {convert_id}"
        )
    if encode_id is not None:
        return encode_id
    if convert_id is not None:
        return convert_id
    raise ValueError("swift_template.tokenizer cannot resolve coord token ids")


def _reject_unknown_coord_token_id(tokenizer: Any, token_id: int, *, token: str) -> None:
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if unk_token_id is not None and int(token_id) == int(unk_token_id):
        raise ValueError(f"coord token {token!r} resolved to unk_token_id")


def _validate_clean_noisy_alignment(
    *,
    clean_sample: NormalizedDetectionSample,
    noisy_sample: NormalizedDetectionSample,
    clean_input_ids: Sequence[int],
    noisy_input_ids: Sequence[int],
    clean_labels: Sequence[int],
    coord_positions: Sequence[Sequence[int]],
    tokenizer: Any,
) -> str | None:
    expected_coord_positions = {
        int(position)
        for object_positions in coord_positions
        for position in object_positions
    }
    diff_positions = {
        index
        for index, (clean_id, noisy_id) in enumerate(
            zip(clean_input_ids, noisy_input_ids, strict=True)
        )
        if int(clean_id) != int(noisy_id)
    }
    if diff_positions - expected_coord_positions:
        return "clean_noisy_noncoord_alignment_failed"
    if diff_positions != expected_coord_positions:
        return "clean_noisy_coord_alignment_failed"

    for object_index, (clean_obj, noisy_obj) in enumerate(
        zip(clean_sample.objects, noisy_sample.objects, strict=True)
    ):
        for slot_idx, position in enumerate(coord_positions[object_index]):
            try:
                clean_token_id = _coord_token_id(
                    tokenizer, int(_bbox_tuple(clean_obj.bbox_2d)[slot_idx])
                )
                noisy_token_id = _coord_token_id(
                    tokenizer, int(_bbox_tuple(noisy_obj.bbox_2d)[slot_idx])
                )
            except ValueError:
                return "coord_label_position_alignment_failed"
            position_i = int(position)
            if int(clean_labels[position_i]) != clean_token_id:
                return "clean_noisy_coord_alignment_failed"
            if int(clean_input_ids[position_i]) != clean_token_id:
                return "clean_noisy_coord_alignment_failed"
            if int(noisy_input_ids[position_i]) != noisy_token_id:
                return "clean_noisy_coord_alignment_failed"
            if clean_token_id == noisy_token_id:
                return "clean_noisy_coord_alignment_failed"
    return None


def _build_kl_sites(
    *,
    clean_segment: PrefixDenoisingSegment,
    noisy_segment: PrefixDenoisingSegment,
    clean_sample: NormalizedDetectionSample,
    clean_input_ids: Sequence[int],
    noisy_input_ids: Sequence[int],
    coord_positions: Sequence[Sequence[int]],
    prefix_denoising: PrefixDenoisingConfig,
    rng: random.Random,
) -> tuple[PrefixDenoisingKLSite, ...]:
    kl_cfg = prefix_denoising.current_object_kl
    if float(kl_cfg.weight) <= 0.0:
        return ()
    object_indices = list(range(len(clean_sample.objects)))
    rng.shuffle(object_indices)
    selected = tuple(
        object_indices[
            : min(int(kl_cfg.num_objects_per_image), len(object_indices))
        ]
    )
    radius = int(kl_cfg.window_radius)
    sites: list[PrefixDenoisingKLSite] = []
    for object_index in selected:
        obj = clean_sample.objects[object_index]
        positions = coord_positions[object_index]
        for slot_idx, slot in enumerate(_COORD_SLOTS):
            clean_gt_bin = int(_bbox_tuple(obj.bbox_2d)[slot_idx])
            label_position = int(positions[slot_idx])
            support_bins = tuple(
                range(
                    max(0, clean_gt_bin - radius),
                    min(999, clean_gt_bin + radius) + 1,
                )
            )
            sites.append(
                PrefixDenoisingKLSite(
                    clean_segment_id=clean_segment.segment_id,
                    noisy_segment_id=noisy_segment.segment_id,
                    object_index=int(object_index),
                    history_object_count=int(object_index),
                    coord_slot=slot,
                    clean_label_position=label_position,
                    noisy_label_position=label_position,
                    clean_gt_bin=clean_gt_bin,
                    support_bins=support_bins,
                    identical_prefix=tuple(clean_input_ids[:label_position])
                    == tuple(noisy_input_ids[:label_position]),
                )
            )
    return tuple(sites)


def _skip_sample(
    *,
    base_sample_id: str,
    hybrid_sample_id: str,
    reason: str,
    metadata: Mapping[str, object] | None = None,
) -> HybridPrefixDenoisingSample:
    return HybridPrefixDenoisingSample(
        ok=False,
        hybrid_sample_id=hybrid_sample_id,
        base_sample_id=base_sample_id,
        clean_full=None,
        noisy_full=None,
        skip_reason=reason,
        metadata={} if metadata is None else dict(metadata),
    )


__all__ = [
    "build_hybrid_prefix_denoising_sample",
    "estimate_hybrid_prefix_denoising_packing",
]
