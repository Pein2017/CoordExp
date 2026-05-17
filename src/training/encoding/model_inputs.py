"""Strict model-input bundle contracts for unified training."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast

BackendKeyDisposition: TypeAlias = Literal[
    "forwarded",
    "bridge_consumed",
    "runner_owned_loss_stripped",
    "sidecar_only",
]

FORWARDED_MODEL_INPUT_KEYS: frozenset[str] = frozenset(
    {
        "attention_mask",
        "cache_position",
        "cross_attention_mask",
        "cu_seq_lens",
        "cu_seq_lens_k",
        "cu_seq_lens_q",
        "image_grid_thw",
        "input_ids",
        "max_length_k",
        "max_length_q",
        "output_router_logits",
        "past_key_values",
        "pixel_values",
        "pixel_values_videos",
        "position_ids",
        "second_per_grid_ts",
        "token_type_ids",
        "use_cache",
        "video_grid_thw",
    }
)

BRIDGE_CONSUMED_MODEL_INPUT_KEYS: frozenset[str] = frozenset(
    {
        "logits_to_keep",
        "text_position_ids",
    }
)

RUNNER_OWNED_LOSS_STRIPPED_KEYS: frozenset[str] = frozenset(
    {
        "compute_loss_func",
        "dataset_labels",
        "dataset_segments",
        "instability_meta_json",
        "loss_scale",
        "pack_num_samples",
        "proxy_coord_token_weights",
        "proxy_desc_token_weights",
        "recursive_detection_targets",
        "sft_structural_close_token_weights",
        "token_types",
    }
)

SIDECAR_ONLY_KEYS: frozenset[str] = frozenset(
    {
        "assignment_result",
        "detection_metadata",
        "diagnostics",
        "duplicate_filter_result",
        "encoded_detection_view",
        "rendered_assistant_text",
        "sample_id",
        "stage2_ownership",
        "supervision_context",
        "supervision_payload",
        "supervision_plan",
        "supervision_spans",
        "training_sidecars",
    }
)

_LABELS_KEY = "labels"

_STATIC_KEY_DISPOSITIONS: Mapping[str, BackendKeyDisposition] = MappingProxyType(
    {
        **{key: "forwarded" for key in FORWARDED_MODEL_INPUT_KEYS},
        **{key: "bridge_consumed" for key in BRIDGE_CONSUMED_MODEL_INPUT_KEYS},
        **{
            key: "runner_owned_loss_stripped"
            for key in RUNNER_OWNED_LOSS_STRIPPED_KEYS
        },
        **{key: "sidecar_only" for key in SIDECAR_ONLY_KEYS},
    }
)


def classify_backend_key(
    key: str,
    *,
    runner_owns_loss: bool = True,
) -> BackendKeyDisposition:
    """Return the backend disposition for a known batch key.

    :param key: Candidate model-input or batch-contract key.
    :param runner_owns_loss: Whether the unified runner owns loss computation.
    :returns: Key disposition for the Qwen3-VL/ms-swift boundary.
    :raises TypeError: If the key is not a plain string.
    :raises ValueError: If the key is unknown.
    """

    if type(key) is not str:
        raise TypeError("backend key must be a plain string")
    if type(runner_owns_loss) is not bool:
        raise TypeError("runner_owns_loss must be a plain bool")

    if key == _LABELS_KEY:
        if runner_owns_loss:
            return "runner_owned_loss_stripped"
        return "forwarded"

    disposition = _STATIC_KEY_DISPOSITIONS.get(key)
    if disposition is None:
        raise ValueError(f"unknown backend key: {key!r}")

    return disposition


def _copy_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a shallow immutable copy of a string-key mapping."""

    copied: dict[str, Any] = {}
    for key, value in mapping.items():
        if type(key) is not str:
            raise TypeError("model input keys must be plain strings")
        copied[key] = value

    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class ModelInputBundle:
    """Validated model-input boundary with explicit non-forwarded auxiliaries.

    The bundle accepts arbitrary values because real callers may supply tensors
    from torch or backend-specific containers. It validates only key ownership,
    never tensor shape or model behavior.

    :param payload: Immutable copy of backend payload values.
    :param runner_owns_loss: Whether labels/loss-side inputs are runner-owned.
    """

    payload: Mapping[str, Any]
    runner_owns_loss: bool = True

    def __post_init__(self) -> None:
        """Validate the payload key registry and freeze the top-level mapping."""

        if type(self.runner_owns_loss) is not bool:
            raise TypeError("runner_owns_loss must be a plain bool")

        payload = _copy_mapping(self.payload)
        for key in payload:
            disposition = classify_backend_key(
                key,
                runner_owns_loss=self.runner_owns_loss,
            )
            if disposition == "sidecar_only":
                raise ValueError(
                    f"sidecar-only key cannot be placed in model inputs: {key!r}"
                )

        object.__setattr__(self, "payload", payload)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        runner_owns_loss: bool = True,
    ) -> "ModelInputBundle":
        """Build a validated bundle from a raw backend mapping."""

        return cls(payload=payload, runner_owns_loss=runner_owns_loss)

    def classification_for(self, key: str) -> BackendKeyDisposition:
        """Return the effective disposition for a bundle key."""

        return classify_backend_key(key, runner_owns_loss=self.runner_owns_loss)

    def forwarded_inputs(self) -> dict[str, Any]:
        """Return keys that may be passed to ``model(**inputs)``."""

        return {
            key: value
            for key, value in self.payload.items()
            if self.classification_for(key) == "forwarded"
        }

    def bridge_auxiliaries(self) -> dict[str, Any]:
        """Return keys consumed by bridge logic before model forwarding."""

        return {
            key: value
            for key, value in self.payload.items()
            if self.classification_for(key) == "bridge_consumed"
        }

    def runner_loss_inputs(self) -> dict[str, Any]:
        """Return keys owned by the runner-side loss path."""

        return {
            key: value
            for key, value in self.payload.items()
            if self.classification_for(key) == "runner_owned_loss_stripped"
        }

    def stripped_for_model_forward(self) -> dict[str, Any]:
        """Return a fresh model-forward mapping with auxiliaries removed."""

        return self.forwarded_inputs()

    def copy_into(self, destination: MutableMapping[str, Any]) -> None:
        """Copy all validated payload keys into a caller-owned mapping."""

        destination.update(dict(self.payload))


def backend_key_registry(
    *,
    runner_owns_loss: bool = True,
) -> Mapping[str, BackendKeyDisposition]:
    """Return an immutable view of the effective backend key registry."""

    if type(runner_owns_loss) is not bool:
        raise TypeError("runner_owns_loss must be a plain bool")

    registry = dict(_STATIC_KEY_DISPOSITIONS)
    registry[_LABELS_KEY] = classify_backend_key(
        _LABELS_KEY,
        runner_owns_loss=runner_owns_loss,
    )

    return cast(Mapping[str, BackendKeyDisposition], MappingProxyType(registry))


__all__ = [
    "BRIDGE_CONSUMED_MODEL_INPUT_KEYS",
    "BackendKeyDisposition",
    "FORWARDED_MODEL_INPUT_KEYS",
    "ModelInputBundle",
    "RUNNER_OWNED_LOSS_STRIPPED_KEYS",
    "SIDECAR_ONLY_KEYS",
    "backend_key_registry",
    "classify_backend_key",
]
