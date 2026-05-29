from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from .runtime import PromptBundle, PromptParityResult


COORDJSON_FORMAT = "coordjson"


@dataclass(frozen=True)
class RolloutPromptPolicyFacts:
    """Resolved prompt policy consumed by rollout prompt normalization."""

    rollout_backend: Literal["hf", "vllm"]
    detection_sequence_format: str
    training_prompt_variant: str
    object_ordering: str
    object_field_order: str
    template_system: Optional[str]


@dataclass(frozen=True)
class DetectionPromptPolicy:
    name: str
    version: str
    system_prompt: str
    user_prompt: str
    image_count: int = 1
    do_resize: bool = False

    @classmethod
    def default_for_detection(cls) -> "DetectionPromptPolicy":
        return cls(
            name="coordexp_detection_default",
            version="1",
            system_prompt="You are a precise visual object detection assistant.",
            user_prompt=(
                "Detect all visible objects in the image and return bounding boxes "
                "with concise descriptions."
            ),
        )


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def prompt_policy_fingerprint(policy: DetectionPromptPolicy) -> str:
    digest = hashlib.sha256(_canonical_json(asdict(policy)).encode("utf-8")).hexdigest()
    return f"prompt_policy:v1:{digest}"


def build_offline_detection_chat_messages(
    *,
    system_prompt: str,
    user_prompt: str,
    image: Any,
    image_content_type: str = "image",
) -> list[dict[str, Any]]:
    """Build canonical one-image offline detection chat messages."""

    from src.common.detection_chat import build_detection_chat_messages

    system_text = str(system_prompt or "").strip()
    user_text = str(user_prompt or "").strip()
    if not system_text or not user_text:
        raise ValueError(
            "offline detection chat messages require nonempty system and user prompts"
        )
    return build_detection_chat_messages(
        system_prompt=system_text,
        user_prompt=user_text,
        images=[image],
        image_content_type=image_content_type,
    )


def _normalize_image_field(value: Any, *, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return str(value)
    raise ValueError(f"sample.{field_name} must be an image path string or Path")


def _image_list_fields(sample: dict[str, Any]) -> list[str]:
    images_value = sample.get("images")
    if images_value is None:
        return []
    if (
        isinstance(images_value, (str, bytes, Path, Mapping))
        or not isinstance(images_value, Sequence)
    ):
        raise ValueError("sample.images must be a sequence of image path strings")
    images: list[str] = []
    for index, item in enumerate(images_value):
        if not isinstance(item, (str, Path)):
            raise ValueError(
                f"sample.images[{index}] must be an image path string or Path"
            )
        images.append(str(item))
    return images


def _resolve_image_path(
    image_field: str,
    *,
    field_name: str,
    root_image_dir: Optional[Path],
    jsonl_dir: Optional[Path],
) -> Path:
    resolved = _resolve_image_path_strict(
        image_field,
        jsonl_dir=jsonl_dir,
        root_image_dir=root_image_dir,
    )
    if resolved is None:
        raise ValueError(f"detection prompt {field_name} path does not exist: {image_field}")
    if not resolved.is_file():
        raise ValueError(f"detection prompt {field_name} path is not a file: {resolved}")
    return resolved.resolve()


def _resolve_image_path_strict(
    image_field: str,
    *,
    jsonl_dir: Optional[Path],
    root_image_dir: Optional[Path],
) -> Optional[Path]:
    try:
        from src.common.paths import resolve_image_path_strict
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
    else:
        return resolve_image_path_strict(
            image_field,
            jsonl_dir=jsonl_dir,
            root_image_dir=root_image_dir,
        )

    # Lightweight fallback for minimal inference-runtime imports when the
    # broader src.common package transitively requires unavailable torch.
    path = Path(str(image_field))
    if path.is_absolute() and path.exists():
        return path
    if root_image_dir is not None:
        candidate = root_image_dir / path
        if candidate.exists():
            return candidate
    if jsonl_dir is not None:
        candidate = jsonl_dir / path
        if candidate.exists():
            return candidate
    return None


def _resolve_one_image_path(
    sample: dict[str, Any],
    *,
    root_image_dir: Optional[Path],
    jsonl_dir: Optional[Path],
) -> Path:
    image = _normalize_image_field(sample.get("image"), field_name="image")
    images = _image_list_fields(sample)
    if len(images) > 1:
        raise ValueError(
            f"detection prompt requires exactly one image; found {len(images)}"
        )

    candidates = images or ([image] if image is not None else [])
    if len(candidates) != 1:
        raise ValueError(
            f"detection prompt requires exactly one image; found {len(candidates)}"
        )

    if image is not None and images:
        image_path = _resolve_image_path(
            image,
            field_name="sample.image",
            root_image_dir=root_image_dir,
            jsonl_dir=jsonl_dir,
        )
        images_path = _resolve_image_path(
            images[0],
            field_name="sample.images[0]",
            root_image_dir=root_image_dir,
            jsonl_dir=jsonl_dir,
        )
        if image_path != images_path:
            raise ValueError("sample image and images[0] refer to different files")
        return image_path

    return _resolve_image_path(
        candidates[0],
        field_name="sample.images[0]" if images else "sample.image",
        root_image_dir=root_image_dir,
        jsonl_dir=jsonl_dir,
    )


def _validate_policy(policy: DetectionPromptPolicy) -> None:
    if policy.image_count != 1:
        raise ValueError(
            "unsupported detection prompt policy: exactly one image is required"
        )
    if policy.do_resize is not False:
        raise ValueError(
            "unsupported detection prompt policy: do_resize must be False"
        )


def _read_png_dimensions(payload: bytes) -> Optional[tuple[int, int]]:
    if len(payload) < 24 or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    if payload[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", payload[16:24])
    return int(width), int(height)


def _read_jpeg_dimensions(payload: bytes) -> Optional[tuple[int, int]]:
    if len(payload) < 4 or not payload.startswith(b"\xff\xd8"):
        return None
    index = 2
    while index + 9 < len(payload):
        if payload[index] != 0xFF:
            index += 1
            continue
        marker = payload[index + 1]
        index += 2
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(payload):
            return None
        segment_length = int.from_bytes(payload[index : index + 2], "big")
        if segment_length < 2 or index + segment_length > len(payload):
            return None
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            height = int.from_bytes(payload[index + 3 : index + 5], "big")
            width = int.from_bytes(payload[index + 5 : index + 7], "big")
            return int(width), int(height)
        index += segment_length
    return None


def _read_image_dimensions(image_path: Path) -> tuple[int, int]:
    payload = image_path.read_bytes()
    dimensions = _read_png_dimensions(payload) or _read_jpeg_dimensions(payload)
    if dimensions is None:
        raise ValueError(f"unsupported or invalid image file: {image_path}")
    return dimensions


def _declared_dimension(sample: dict[str, Any], key: str) -> Optional[int]:
    value = sample.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sample {key} must be an integer image dimension") from exc


def _message_image_info(
    messages: Optional[Sequence[Mapping[str, Any]]],
) -> tuple[int, str | None]:
    image_count = 0
    image_path: str | None = None
    if messages is None:
        return image_count, image_path
    for message in messages:
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            continue
        for part in content:
            if not isinstance(part, Mapping) or part.get("type") != "image":
                continue
            image_count += 1
            raw_path = part.get("image")
            if isinstance(raw_path, (str, Path)):
                image_path = str(raw_path)
    return image_count, image_path


def _visual_metadata(sample: dict[str, Any], *, image_path: Path) -> dict[str, Any]:
    original_width, original_height = _read_image_dimensions(image_path)
    declared_width = _declared_dimension(sample, "width")
    declared_height = _declared_dimension(sample, "height")
    if declared_width is not None and declared_width != original_width:
        raise ValueError(
            "sample width does not match actual image width: "
            f"declared={declared_width} actual={original_width}"
        )
    if declared_height is not None and declared_height != original_height:
        raise ValueError(
            "sample height does not match actual image height: "
            f"declared={declared_height} actual={original_height}"
        )
    image_path_str = str(image_path)
    metadata: dict[str, Any] = {
        "image_count": 1,
        "do_resize": False,
        "image": image_path_str,
        "image_path": image_path_str,
        "image_placement": "messages[1].content[0]",
        "width": original_width,
        "height": original_height,
        "original_width": original_width,
        "original_height": original_height,
        "post_preprocessing_width": original_width,
        "post_preprocessing_height": original_height,
    }
    if declared_width is not None:
        metadata["declared_width"] = declared_width
    if declared_height is not None:
        metadata["declared_height"] = declared_height
    return metadata


def rollout_visual_metadata_from_sample(
    sample: Mapping[str, Any],
    *,
    messages: Optional[Sequence[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Build lightweight visual metadata for online rollout prompt parity."""

    images_raw = sample.get("images")
    image_raw = sample.get("image")
    images: list[str] = []
    if isinstance(images_raw, (str, Path)):
        images = [str(images_raw)]
    elif isinstance(images_raw, Sequence) and not isinstance(
        images_raw, (bytes, bytearray, str, Mapping)
    ):
        images = [str(item) for item in images_raw if isinstance(item, (str, Path))]
    elif isinstance(image_raw, (str, Path)):
        images = [str(image_raw)]

    message_image_count, message_image_path = _message_image_info(messages)
    if len(images) > 1:
        raise ValueError(
            "rollout prompt visual input requires exactly one image; "
            f"found {len(images)}"
        )

    if message_image_count > 1:
        raise ValueError(
            "rollout prompt visual input requires exactly one message image; "
            f"found {message_image_count}"
        )
    if len(images) == 0:
        if message_image_count == 1 and message_image_path is not None:
            images = [message_image_path]
        else:
            raise ValueError(
                "rollout prompt visual input requires exactly one image; "
                f"found {len(images)}"
            )
    if (
        message_image_count == 1
        and message_image_path is not None
        and message_image_path != images[0]
    ):
        raise ValueError(
            "rollout prompt message image does not match sample image side-channel"
        )

    width = _declared_dimension(dict(sample), "width")
    height = _declared_dimension(dict(sample), "height")
    return {
        "image_count": 1,
        "image_path": images[0],
        "image_placement": (
            "message_content" if message_image_count == 1 else "sample.images"
        ),
        "do_resize": False,
        "original_width": width,
        "original_height": height,
        "post_preprocessing_width": width,
        "post_preprocessing_height": height,
    }


def strip_trailing_assistant_turns_for_rollout(messages: Any) -> list[Any]:
    """Build a prompt-only message list for rollout generation.

    Many training datasets include a teacher-forced assistant answer inside
    ``sample["messages"]``. For rollouts, generation must start from a prompt
    that ends with the last user turn, while preserving earlier assistant turns
    that may be intentional conversational context.
    """

    if not isinstance(messages, list):
        return list(messages) if messages is not None else []

    last_user_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if isinstance(message, Mapping) and message.get("role") == "user":
            last_user_idx = int(i)
            break

    if last_user_idx is None:
        trimmed: list[Any] = []
        for message in messages:
            if isinstance(message, Mapping) and message.get("role") == "assistant":
                continue
            trimmed.append(message)
        return trimmed if trimmed else list(messages)

    return list(messages[: last_user_idx + 1])


def ensure_system_prompt_message(messages: Any, system_prompt: str) -> list[Any]:
    """Prepend an OpenAI-style system prompt message when absent."""

    if not system_prompt:
        return list(messages) if isinstance(messages, list) else []

    if not isinstance(messages, list):
        messages_list: list[Any] = []
    else:
        messages_list = list(messages)

    for message in messages_list:
        if isinstance(message, Mapping) and message.get("role") == "system":
            return messages_list

    return [{"role": "system", "content": str(system_prompt)}, *messages_list]


def force_last_user_prompt_text(messages: Any, user_prompt: str) -> list[Any]:
    """Replace the last user-turn text while preserving multimodal image items."""

    if not isinstance(messages, list):
        return [] if messages is None else [messages]
    if not isinstance(user_prompt, str) or not user_prompt:
        return list(messages)

    out = deepcopy(list(messages))

    last_user_idx: int | None = None
    for i in range(len(out) - 1, -1, -1):
        message = out[i]
        if isinstance(message, Mapping) and str(message.get("role", "")).lower() == "user":
            last_user_idx = int(i)
            break
    if last_user_idx is None:
        return out

    message_any = out[last_user_idx]
    if not isinstance(message_any, MutableMapping):
        return out

    content = message_any.get("content")
    if isinstance(content, str):
        message_any["content"] = str(user_prompt)
        return out

    if isinstance(content, list):
        replaced = False
        new_content: list[Any] = []
        for item in content:
            if isinstance(item, Mapping):
                is_text_item = str(item.get("type", "")).lower() == "text" or (
                    "text" in item
                )
                if is_text_item:
                    if replaced:
                        continue
                    new_item = dict(item)
                    new_item["type"] = "text"
                    new_item["text"] = str(user_prompt)
                    new_content.append(new_item)
                    replaced = True
                    continue
            new_content.append(item)
        if not replaced:
            new_content.append({"type": "text", "text": str(user_prompt)})
        message_any["content"] = new_content
        return out

    message_any["content"] = str(user_prompt)
    return out


def prepare_rollout_prompt_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    rollout_backend: Literal["hf", "vllm"],
    prompt_variant_override: Optional[str] = None,
    detection_sequence_format: str = COORDJSON_FORMAT,
    training_prompt_variant: Optional[str],
    object_ordering: str,
    object_field_order: str,
    template_system: Optional[str] = None,
) -> list[Mapping[str, Any]]:
    """Normalize Stage-2 rollout prompt samples for shared decode backends."""

    backend = str(rollout_backend).strip().lower()
    if backend not in {"hf", "vllm"}:
        raise ValueError("rollout_backend must be one of {'hf', 'vllm'}")

    user_prompt_override: str | None = None
    system_prompt_override: str | None = None
    should_rebuild_prompt = bool(
        prompt_variant_override is not None
        or str(detection_sequence_format) != COORDJSON_FORMAT
    )
    if should_rebuild_prompt:
        from src.config.prompts import (
            build_dense_system_prompt,
            build_dense_user_prompt,
            resolve_dense_prompt_variant_key,
        )

        variant_key = (
            resolve_dense_prompt_variant_key(prompt_variant_override)
            if prompt_variant_override is not None
            else resolve_dense_prompt_variant_key(training_prompt_variant)
        )
        user_prompt_override = build_dense_user_prompt(
            ordering=object_ordering,
            coord_mode="coord_tokens",
            prompt_variant=variant_key,
            object_field_order=object_field_order,
            detection_sequence_format=str(detection_sequence_format),
        )
        system_prompt_override = build_dense_system_prompt(
            ordering=object_ordering,
            coord_mode="coord_tokens",
            prompt_variant=variant_key,
            object_field_order=object_field_order,
            detection_sequence_format=str(detection_sequence_format),
        )

    system_prompt: str | None = None
    if backend == "vllm":
        if isinstance(system_prompt_override, str) and system_prompt_override.strip():
            system_prompt = str(system_prompt_override)
        elif isinstance(template_system, str) and template_system.strip():
            system_prompt = str(template_system)
        else:
            try:
                from src.config.prompts import (
                    build_dense_system_prompt,
                    resolve_dense_prompt_variant_key,
                )

                system_prompt = build_dense_system_prompt(
                    ordering=object_ordering,
                    coord_mode="coord_tokens",
                    prompt_variant=resolve_dense_prompt_variant_key(
                        training_prompt_variant
                    ),
                    object_field_order=object_field_order,
                    detection_sequence_format=str(detection_sequence_format),
                )
            except (TypeError, ValueError):
                system_prompt = None

    samples_for_rollout: list[Mapping[str, Any]] = []
    for sample in samples:
        messages = sample.get("messages")
        if not isinstance(messages, list):
            samples_for_rollout.append(sample)
            continue

        modified = False

        trimmed = strip_trailing_assistant_turns_for_rollout(messages)
        if len(trimmed) != len(messages):
            modified = True
            messages_out: list[Any] = trimmed
        else:
            messages_out = messages

        if user_prompt_override is not None:
            messages_prompt = force_last_user_prompt_text(
                messages_out,
                str(user_prompt_override),
            )
            if messages_prompt != messages_out:
                modified = True
                messages_out = messages_prompt

        if backend == "vllm" and system_prompt is not None:
            messages_sys = ensure_system_prompt_message(messages_out, system_prompt)
            if len(messages_sys) != len(messages_out):
                modified = True
                messages_out = messages_sys

        images_out = None
        images_from_message_only = False
        prompt_visual_metadata = None
        images_raw = sample.get("images", None)
        if images_raw is None:
            image = sample.get("image", None)
            if isinstance(image, (str, Path)) and str(image):
                images_raw = [str(image)]

        message_image_count, message_image_path = _message_image_info(
            messages_out if isinstance(messages_out, list) else None
        )
        if images_raw is not None:
            if isinstance(images_raw, (str, Path)):
                images_out = [str(images_raw)]
            elif isinstance(images_raw, list):
                images_out = list(images_raw)
            elif isinstance(images_raw, tuple):
                images_out = list(images_raw)

            if (
                images_out is not None
                and len(images_out) == 0
                and message_image_count == 1
                and message_image_path is not None
            ):
                images_out = [message_image_path]
                images_from_message_only = True

            if images_out is not None and len(images_out) != 1:
                raise ValueError(
                    "rollout prompt visual input requires exactly one image; "
                    f"found {len(images_out)}"
                )
        elif message_image_count == 1 and message_image_path is not None:
            images_out = [message_image_path]
            images_from_message_only = True

        if images_out is not None:
            if (
                backend == "vllm"
                and not images_from_message_only
                and (
                    not isinstance(sample.get("images"), list)
                    or sample.get("images") != images_out
                )
            ):
                modified = True

            sample_for_visual = dict(sample)
            if not images_from_message_only:
                sample_for_visual["images"] = list(images_out)
            prompt_visual_metadata = rollout_visual_metadata_from_sample(
                sample_for_visual,
                messages=messages_out if isinstance(messages_out, list) else None,
            )
            if sample.get("_coordexp_prompt_visual_metadata") != prompt_visual_metadata:
                modified = True

        if modified:
            sample_out = dict(sample)
            sample_out["messages"] = messages_out
            if images_out is not None and not images_from_message_only:
                sample_out["images"] = images_out
            if prompt_visual_metadata is not None:
                sample_out["_coordexp_prompt_visual_metadata"] = prompt_visual_metadata
            samples_for_rollout.append(sample_out)
        else:
            samples_for_rollout.append(sample)

    return samples_for_rollout


def prepare_rollout_prompt_samples_from_owner(
    owner: Any,
    samples: Sequence[Mapping[str, Any]],
    *,
    prompt_variant_override: Optional[str] = None,
    rollout_backend: Optional[Literal["hf", "vllm"]] = None,
) -> list[Mapping[str, Any]]:
    """Resolve owner policy values, then normalize rollout prompt samples.

    `src.infer` owns prompt/input normalization; the owner is only a duck-typed
    source for already-existing Stage-2 config policy values.
    """

    from src.infer.runtime import effective_rollout_backend_from_owner

    facts = RolloutPromptPolicyFacts(
        rollout_backend=(
            rollout_backend
            if rollout_backend is not None
            else effective_rollout_backend_from_owner(owner, context="train")
        ),
        detection_sequence_format=str(owner._detection_sequence_format()),
        training_prompt_variant=str(owner._training_prompt_variant()),
        object_ordering=str(owner._object_ordering()),
        object_field_order=str(owner._object_field_order()),
        template_system=getattr(getattr(owner, "template", None), "system", None),
    )
    return prepare_rollout_prompt_samples_from_facts(
        samples,
        facts=facts,
        prompt_variant_override=prompt_variant_override,
    )


def prepare_rollout_prompt_samples_from_facts(
    samples: Sequence[Mapping[str, Any]],
    *,
    facts: RolloutPromptPolicyFacts,
    prompt_variant_override: Optional[str] = None,
) -> list[Mapping[str, Any]]:
    """Normalize rollout prompt samples from resolved prompt facts."""

    return prepare_rollout_prompt_samples(
        samples,
        rollout_backend=facts.rollout_backend,
        prompt_variant_override=prompt_variant_override,
        detection_sequence_format=facts.detection_sequence_format,
        training_prompt_variant=facts.training_prompt_variant,
        object_ordering=facts.object_ordering,
        object_field_order=facts.object_field_order,
        template_system=facts.template_system,
    )


def _coerce_template_token_ids(encoded: Any) -> Optional[list[int]]:
    if encoded is None:
        return None
    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids")
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, tuple):
        encoded = list(encoded)
    if (
        isinstance(encoded, list)
        and len(encoded) == 1
        and isinstance(encoded[0], list)
    ):
        encoded = encoded[0]
    if not isinstance(encoded, list):
        return None
    return [int(token_id) for token_id in encoded]


def _apply_chat_template(
    renderer: Any,
    messages: list[dict[str, Any]],
    *,
    tokenize: bool,
) -> Any:
    return renderer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=True,
    )


def _render_prompt_with_template(
    *,
    messages: list[dict[str, Any]],
    fallback_text: str,
    tokenizer: Optional[Any],
    processor: Optional[Any],
) -> tuple[str, Optional[list[int]], str, str]:
    renderer = processor if hasattr(processor, "apply_chat_template") else tokenizer
    if not hasattr(renderer, "apply_chat_template"):
        return fallback_text, None, "unavailable", "unverifiable"

    prompt_text = _apply_chat_template(renderer, messages, tokenize=False)
    if not isinstance(prompt_text, str):
        prompt_text = fallback_text

    token_ids = _coerce_template_token_ids(
        _apply_chat_template(renderer, messages, tokenize=True)
    )
    if token_ids is None:
        return prompt_text, None, "unavailable", "unverifiable"
    return prompt_text, token_ids, "chat_template", "unverified"


_PARITY_VISUAL_KEYS = (
    "image_count",
    "image_path",
    "image_placement",
    "do_resize",
    "original_width",
    "original_height",
    "post_preprocessing_width",
    "post_preprocessing_height",
)


def compare_prompt_bundle_parity(
    reference: PromptBundle,
    candidate: PromptBundle,
) -> PromptParityResult:
    if reference.prompt_token_ids is None or candidate.prompt_token_ids is None:
        return PromptParityResult(
            prompt_token_parity="unverifiable",
            verified=False,
            reason="prompt token IDs are missing from at least one bundle",
        )
    if reference.prompt_token_ids != candidate.prompt_token_ids:
        return PromptParityResult(
            prompt_token_parity="unverified",
            verified=False,
            reason="prompt token IDs differ",
        )

    for key in _PARITY_VISUAL_KEYS:
        if reference.visual_metadata.get(key) != candidate.visual_metadata.get(key):
            return PromptParityResult(
                prompt_token_parity="unverified",
                verified=False,
                reason=f"visual metadata differs for {key}",
            )

    return PromptParityResult(
        prompt_token_parity="verified",
        verified=True,
        reason="prompt token IDs and visual metadata match",
    )


def require_verified_prompt_token_parity(
    *,
    local_prompt_token_ids: Sequence[int],
    backend_prompt_token_ids: Sequence[int],
    expected_prompt_len: Optional[int],
    context: str,
    require_backend_prompt_ids: bool = True,
) -> PromptParityResult:
    """Require byte-for-byte prompt-token parity before trainable decode use."""

    local_ids = [int(token_id) for token_id in local_prompt_token_ids]
    backend_ids = [int(token_id) for token_id in backend_prompt_token_ids]
    if require_backend_prompt_ids and not backend_ids:
        raise ValueError(f"{context}: backend prompt token IDs are missing")
    if not backend_ids:
        return PromptParityResult(
            prompt_token_parity="unverifiable",
            verified=False,
            reason="backend prompt token IDs are missing",
        )
    teacher_prefix = local_ids[: len(backend_ids)]
    if teacher_prefix != backend_ids:
        mismatch_at = next(
            (
                index
                for index, (local_id, backend_id) in enumerate(
                    zip(teacher_prefix, backend_ids)
                )
                if int(local_id) != int(backend_id)
            ),
            None,
        )
        window_start = max(0, int(mismatch_at or 0) - 3)
        window_end = min(int(len(backend_ids)), int(mismatch_at or 0) + 4)
        raise ValueError(
            f"{context}: prompt token IDs differ; "
            f"mismatch_at={mismatch_at} "
            f"local_ids={teacher_prefix[window_start:window_end]} "
            f"backend_ids={backend_ids[window_start:window_end]}"
        )
    if expected_prompt_len is not None and int(expected_prompt_len) != len(backend_ids):
        raise ValueError(
            f"{context}: prompt_len mismatch; "
            f"expected_prompt_len={int(expected_prompt_len)} "
            f"backend_prompt_len={len(backend_ids)}"
        )
    return PromptParityResult(
        prompt_token_parity="verified",
        verified=True,
        reason="prompt token IDs match",
    )


def require_verified_prompt_visual_parity(
    *,
    expected_visual_metadata: Mapping[str, Any],
    observed_visual_metadata: Mapping[str, Any] | None,
    context: str,
) -> PromptParityResult:
    """Require exact visual-input metadata parity before trainable rollout use."""

    if not isinstance(observed_visual_metadata, Mapping):
        raise ValueError(f"{context}: prompt visual metadata is missing")
    for key in _PARITY_VISUAL_KEYS:
        if expected_visual_metadata.get(key) != observed_visual_metadata.get(key):
            raise ValueError(
                f"{context}: prompt visual metadata differs for {key}; "
                f"expected={expected_visual_metadata.get(key)!r} "
                f"observed={observed_visual_metadata.get(key)!r}"
            )
    return PromptParityResult(
        prompt_token_parity="verified",
        verified=True,
        reason="prompt visual metadata matches",
    )


def build_prompt_bundle(
    sample: dict[str, Any],
    policy: DetectionPromptPolicy,
    tokenizer: Optional[Any] = None,
    processor: Optional[Any] = None,
    root_image_dir: Optional[Path | str] = None,
    jsonl_dir: Optional[Path | str] = None,
) -> PromptBundle:
    _validate_policy(policy)
    image_path = _resolve_one_image_path(
        sample,
        root_image_dir=Path(root_image_dir) if root_image_dir is not None else None,
        jsonl_dir=Path(jsonl_dir) if jsonl_dir is not None else None,
    )
    fingerprint = prompt_policy_fingerprint(policy)
    fallback_text = "\n".join(
        (
            policy.system_prompt,
            policy.user_prompt,
            "Image: <image>",
        )
    )
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": policy.system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": policy.user_prompt},
            ],
        },
    ]
    prompt_text, prompt_token_ids, token_ids_source, token_parity = (
        _render_prompt_with_template(
            messages=messages,
            fallback_text=fallback_text,
            tokenizer=tokenizer,
            processor=processor,
        )
    )
    return PromptBundle(
        messages=messages,
        prompt_text=prompt_text,
        prompt_policy_fingerprint=fingerprint,
        visual_metadata=_visual_metadata(sample, image_path=image_path),
        prompt_token_ids=prompt_token_ids,
        prompt_token_ids_source=token_ids_source,
        prompt_token_parity=token_parity,
    )
