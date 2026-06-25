"""Same-forward Qwen3-VL capture for coverage-ledger supervision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from src.trainers.teacher_forcing.forwards import (
    assert_unsliced_logits,
    prepare_forward_inputs,
)


@dataclass(frozen=True, slots=True)
class CoverageLedgerForwardCaptureResult:
    """Qwen forward products needed by runner loss and future ledger loss."""

    logits: torch.Tensor
    final_hidden_states: torch.Tensor
    image_embeds: torch.Tensor
    outputs: Any


class CoverageLedgerForwardCapture:
    """Capture post-merger Qwen image embeddings from the same model forward."""

    def capture(
        self,
        *,
        model: Any,
        inputs: Mapping[str, Any],
        ignored_keys: Sequence[str],
        packing_enabled: bool,
        where: str = "CoverageLedgerForwardCapture",
    ) -> CoverageLedgerForwardCaptureResult:
        """Run the lower Qwen model once and return full logits plus captures."""

        if "logits_to_keep" in inputs:
            raise ValueError(
                f"{where}: logits_to_keep is rejected for coverage-ledger capture; "
                "full logits are required"
            )

        active_model = self._unwrap_active_trainable_qwen_model(model)
        core_model, inputs_for_model, model_type = prepare_forward_inputs(
            model=active_model,
            inputs=inputs,
            ignored_keys=ignored_keys,
            packing_enabled=packing_enabled,
            where=where,
        )
        _validate_v0_media_inputs(
            inputs_for_model,
            packing_enabled=packing_enabled,
            where=where,
        )
        lower_model = self._require_qwen_conditional_generation(
            core_model,
            model_type=model_type,
            where=where,
        )

        captured_image_embeds: list[torch.Tensor] = []
        lower_model_vars = vars(lower_model)
        had_instance_get_image_features = "get_image_features" in lower_model_vars
        instance_get_image_features = lower_model_vars.get("get_image_features")
        original_get_image_features = lower_model.get_image_features

        def wrapped_get_image_features(*args: Any, **kwargs: Any) -> Any:
            result = original_get_image_features(*args, **kwargs)
            captured_image_embeds.append(
                _normalize_image_embeds(result, where=where)
            )
            return result

        try:
            lower_model.get_image_features = wrapped_get_image_features
            outputs = lower_model(**inputs_for_model)
        finally:
            if had_instance_get_image_features:
                lower_model.get_image_features = instance_get_image_features
            elif "get_image_features" in vars(lower_model):
                delattr(lower_model, "get_image_features")

        if len(captured_image_embeds) != 1:
            raise ValueError(
                f"{where}: expected exactly one image embedding capture from "
                f"get_image_features; got {len(captured_image_embeds)}"
            )

        image_embeds = captured_image_embeds[0]
        self._validate_image_embeds(
            image_embeds=image_embeds,
            inputs_for_model=inputs_for_model,
            core_model=core_model,
            lower_model=lower_model,
            packing_enabled=packing_enabled,
            where=where,
        )

        final_hidden_states = _extract_final_hidden_states(outputs, where=where)
        logits = core_model.lm_head(final_hidden_states)
        input_ids = inputs_for_model.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            assert_unsliced_logits(logits=logits, input_ids=input_ids, where=where)

        return CoverageLedgerForwardCaptureResult(
            logits=logits,
            final_hidden_states=final_hidden_states,
            image_embeds=image_embeds,
            outputs=outputs,
        )

    def _unwrap_active_trainable_qwen_model(self, model: Any) -> Any:
        """Return the live Qwen conditional module from common train wrappers."""

        root = getattr(model, "module", model)
        qwen_model = _find_qwen_conditional_generation(root)
        if qwen_model is not None:
            return qwen_model
        return root

    def _require_qwen_conditional_generation(
        self,
        core_model: Any,
        *,
        model_type: str,
        where: str,
    ) -> Any:
        if model_type != "qwen3_vl":
            raise TypeError(
                f"{where}: coverage-ledger capture requires a Qwen3-VL "
                f"conditional-generation model; got model_type={model_type!r}"
            )

        lower_model = getattr(core_model, "model", None)
        lm_head = getattr(core_model, "lm_head", None)
        get_image_features = getattr(lower_model, "get_image_features", None)
        if (
            lower_model is None
            or not callable(lower_model)
            or not callable(lm_head)
            or not callable(get_image_features)
        ):
            raise TypeError(
                f"{where}: coverage-ledger capture requires a Qwen3-VL "
                "conditional-generation object exposing .model, "
                ".model.get_image_features, and .lm_head"
            )
        return lower_model

    def _validate_image_embeds(
        self,
        *,
        image_embeds: torch.Tensor,
        inputs_for_model: Mapping[str, Any],
        core_model: Any,
        lower_model: Any,
        packing_enabled: bool,
        where: str,
    ) -> None:
        if (
            image_embeds.ndim == 0
            or image_embeds.numel() == 0
            or image_embeds.shape[0] == 0
        ):
            raise ValueError(f"{where}: captured image_embeds must be non-empty")

        input_ids = inputs_for_model.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError(f"{where}: input_ids are required to validate image embeds")

        image_token_id = _resolve_image_token_id(core_model, lower_model, where=where)
        image_token_count = int((input_ids == image_token_id).sum().item())
        grid_token_count = _expected_image_tokens_from_grid(
            inputs_for_model.get("image_grid_thw"),
            lower_model=lower_model,
            packing_enabled=packing_enabled,
            where=where,
        )
        embed_count = int(image_embeds.shape[0])

        if image_token_count <= 0:
            raise ValueError(f"{where}: image-token slots are missing from input_ids")
        if embed_count != image_token_count or embed_count != grid_token_count:
            raise ValueError(
                f"{where}: image embed count mismatch: image_embeds={embed_count}, "
                f"image_token_slots={image_token_count}, image_grid_tokens={grid_token_count}"
            )


def _normalize_image_embeds(result: Any, *, where: str) -> torch.Tensor:
    image_embeds = result[0] if isinstance(result, tuple) and result else result
    if isinstance(image_embeds, torch.Tensor):
        return image_embeds
    if isinstance(image_embeds, Sequence) and not isinstance(image_embeds, (str, bytes)):
        chunks = tuple(image_embeds)
        if not chunks:
            raise ValueError(f"{where}: captured image_embeds must be non-empty")
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, torch.Tensor):
                raise TypeError(
                    f"{where}: image_embeds[{index}] must be a torch.Tensor; "
                    f"got {type(chunk).__name__}"
                )
        return torch.cat(chunks, dim=0)
    raise TypeError(
        f"{where}: get_image_features must return post-merger image embeddings"
    )


def _find_qwen_conditional_generation(root: Any) -> Any | None:
    seen: set[int] = set()
    stack: list[Any] = [root]
    while stack:
        current = stack.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if _is_qwen_conditional_generation(current):
            return current

        get_base_model = getattr(current, "get_base_model", None)
        if callable(get_base_model):
            try:
                base_model = get_base_model()
            except TypeError:
                base_model = None
            if base_model is not None and id(base_model) not in seen:
                stack.append(base_model)

        for attr_name in ("base_model", "model"):
            child = getattr(current, attr_name, None)
            if child is not None and id(child) not in seen:
                stack.append(child)

    return None


def _is_qwen_conditional_generation(candidate: Any) -> bool:
    if _is_peft_like_wrapper(candidate):
        return False
    config = getattr(candidate, "config", None)
    if getattr(config, "model_type", None) != "qwen3_vl":
        return False
    lower_model = getattr(candidate, "model", None)
    return (
        lower_model is not None
        and callable(getattr(candidate, "lm_head", None))
        and callable(getattr(lower_model, "get_image_features", None))
    )


def _is_peft_like_wrapper(candidate: Any) -> bool:
    module_name = type(candidate).__module__
    if module_name == "peft" or module_name.startswith("peft."):
        return True
    return False


def _extract_final_hidden_states(outputs: Any, *, where: str) -> torch.Tensor:
    hidden_states = getattr(outputs, "last_hidden_state", None)
    if hidden_states is None:
        try:
            hidden_states = outputs[0]
        except (IndexError, KeyError, TypeError) as exc:
            raise TypeError(
                f"{where}: lower Qwen forward outputs must expose final hidden states"
            ) from exc
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError(
            f"{where}: final hidden states must be a torch.Tensor; "
            f"got {type(hidden_states).__name__}"
        )
    return hidden_states


def _resolve_image_token_id(core_model: Any, lower_model: Any, *, where: str) -> int:
    for owner in (
        getattr(core_model, "config", None),
        getattr(lower_model, "config", None),
    ):
        value = getattr(owner, "image_token_id", None)
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    raise ValueError(f"{where}: Qwen image_token_id is required")


def _expected_image_tokens_from_grid(
    image_grid_thw: Any,
    *,
    lower_model: Any,
    packing_enabled: bool,
    where: str,
) -> int:
    if not isinstance(image_grid_thw, torch.Tensor):
        raise ValueError(f"{where}: image_grid_thw is required")
    _validate_image_grid_thw_v0(
        image_grid_thw,
        packing_enabled=packing_enabled,
        where=where,
    )

    merge_size = _resolve_spatial_merge_size(lower_model, where=where)
    merge_area = merge_size * merge_size
    per_image_patches = image_grid_thw.prod(dim=-1)
    if bool((per_image_patches % merge_area != 0).any().item()):
        raise ValueError(
            f"{where}: image_grid_thw is not divisible by spatial_merge_size^2"
        )
    return int((per_image_patches // merge_area).sum().item())


def _validate_v0_media_inputs(
    inputs_for_model: Mapping[str, Any],
    *,
    packing_enabled: bool,
    where: str,
) -> None:
    if inputs_for_model.get("pixel_values_videos") is not None:
        raise ValueError(f"{where}: pixel_values_videos is unsupported in coverage-ledger v0")
    if inputs_for_model.get("video_grid_thw") is not None:
        raise ValueError(f"{where}: video_grid_thw is unsupported in coverage-ledger v0")
    image_grid_thw = inputs_for_model.get("image_grid_thw")
    if not isinstance(image_grid_thw, torch.Tensor):
        raise ValueError(f"{where}: image_grid_thw is required")
    _validate_image_grid_thw_v0(
        image_grid_thw,
        packing_enabled=packing_enabled,
        where=where,
    )


def _validate_image_grid_thw_v0(
    image_grid_thw: torch.Tensor,
    *,
    packing_enabled: bool,
    where: str,
) -> None:
    if image_grid_thw.ndim != 2 or int(image_grid_thw.shape[1]) != 3:
        raise ValueError(f"{where}: image_grid_thw must have shape (N, 3)")
    row_count = int(image_grid_thw.shape[0])
    if row_count <= 0:
        raise ValueError(f"{where}: image_grid_thw must contain at least one image row")
    if not packing_enabled and row_count != 1:
        raise ValueError(
            f"{where}: image_grid_thw must have shape (1, 3) when packing is disabled"
        )
    for row_index in range(row_count):
        grid_t = int(image_grid_thw[row_index, 0].item())
        if grid_t != 1:
            raise ValueError(
                f"{where}: image_grid_thw row {row_index} T must be 1 "
                "for coverage-ledger v0"
            )


def _resolve_spatial_merge_size(lower_model: Any, *, where: str) -> int:
    visual = getattr(lower_model, "visual", None)
    value = getattr(visual, "spatial_merge_size", None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{where}: Qwen visual.spatial_merge_size is required")
    return int(value)


__all__ = [
    "CoverageLedgerForwardCapture",
    "CoverageLedgerForwardCaptureResult",
]
