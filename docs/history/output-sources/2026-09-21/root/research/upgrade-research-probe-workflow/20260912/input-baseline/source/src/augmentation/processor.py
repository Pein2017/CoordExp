"""Static stochastic train-example augmentation processors."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.augmentation.geometry import (
    COORD_AFFINE_MATRICES,
    GEOMETRY_FLIP_POLICY_VERSION,
    transform_bbox,
)
from src.common.errors import DataContractError
from src.config.models import GeometryFlipsAugmentationConfig
from src.data import RawExample, RawObject

TRANSFORM_IDS = ("identity", "hflip", "vflip", "hvflip")


@dataclass(frozen=True)
class AugmentationMaterializationResult:
    examples: tuple[RawExample, ...]
    receipt: dict[str, Any]


@dataclass(frozen=True)
class NoopAugmentationProcessor:
    runtime_seed: int

    def materialize(
        self,
        raw_examples: Sequence[RawExample],
        *,
        split: str,
        object_ordering: str,
    ) -> AugmentationMaterializationResult:
        examples = tuple(raw_examples)
        return AugmentationMaterializationResult(
            examples=examples,
            receipt={
                "split": split,
                "mode": "disabled",
                "policy": "geometry_flips",
                "enabled": False,
                "seed": self.runtime_seed,
                "input_example_count": len(examples),
                "output_example_count": len(examples),
                "presentation_count": len(examples),
                "object_ordering": object_ordering,
            },
        )


@dataclass(frozen=True)
class GeometryFlipAugmentationProcessor:
    config: GeometryFlipsAugmentationConfig
    runtime_seed: int

    def materialize(
        self,
        raw_examples: Sequence[RawExample],
        *,
        split: str,
        object_ordering: str,
    ) -> AugmentationMaterializationResult:
        examples = tuple(raw_examples)
        if not self.config.enabled:
            return NoopAugmentationProcessor(self.runtime_seed).materialize(
                examples,
                split=split,
                object_ordering=object_ordering,
            )
        transformed: list[RawExample] = []
        transform_counts = {transform_id: 0 for transform_id in TRANSFORM_IDS}
        random_order_receipts: list[dict[str, Any]] = []
        for source_index, raw_example in enumerate(examples):
            transform_id, seed_source = self._sample_transform(raw_example, split=split)
            transform_counts[transform_id] += 1
            presentation = _build_presentation(
                raw_example,
                source_index=source_index,
                split=split,
                transform_id=transform_id,
                transform_seed_source=seed_source,
                object_ordering=object_ordering,
                object_order_seed=self.runtime_seed
                if object_ordering == "random"
                else None,
            )
            if object_ordering == "random":
                random_order_receipts.append(
                    {
                        "source_example_id": raw_example.example_id,
                        "presentation_id": presentation.example_id,
                        "object_order_seed": self.runtime_seed,
                        "object_order_seed_source": (
                            f"{self.runtime_seed}:{presentation.example_id}"
                        ),
                    }
                )
            transformed.append(presentation)
        return AugmentationMaterializationResult(
            examples=tuple(transformed),
            receipt={
                "split": split,
                "mode": "static_stochastic_view",
                "policy": "geometry_flips",
                "policy_version": GEOMETRY_FLIP_POLICY_VERSION,
                "enabled": True,
                "seed": self.runtime_seed,
                "horizontal_prob": self.config.horizontal_prob,
                "vertical_prob": self.config.vertical_prob,
                "input_example_count": len(examples),
                "output_example_count": len(transformed),
                "presentation_count": len(transformed),
                "transform_counts": transform_counts,
                "object_ordering": object_ordering,
                "random_object_order_presentations": random_order_receipts,
            },
        )

    def _sample_transform(
        self,
        raw_example: RawExample,
        *,
        split: str,
    ) -> tuple[str, str]:
        seed_source = (
            f"{self.runtime_seed}:{split}:{raw_example.example_id}:"
            f"{raw_example.source.row_number}:geometry_flips:"
            f"{GEOMETRY_FLIP_POLICY_VERSION}"
        )
        seed_bytes = hashlib.sha256(seed_source.encode("utf-8")).digest()[:8]
        rng = random.Random(int.from_bytes(seed_bytes, "big"))
        horizontal = rng.random() < self.config.horizontal_prob
        vertical = rng.random() < self.config.vertical_prob
        if horizontal and vertical:
            return "hvflip", seed_source
        if horizontal:
            return "hflip", seed_source
        if vertical:
            return "vflip", seed_source
        return "identity", seed_source


def _build_presentation(
    raw_example: RawExample,
    *,
    source_index: int,
    split: str,
    transform_id: str,
    transform_seed_source: str,
    object_ordering: str,
    object_order_seed: int | None,
) -> RawExample:
    if object_ordering not in {"source_order", "geo_sorted", "random"}:
        raise DataContractError(
            "unsupported object ordering for augmentation",
            code="augmentation.object_ordering",
            context={"object_ordering": object_ordering},
        )
    presentation_id = (
        raw_example.example_id
        if transform_id == "identity"
        else f"{raw_example.example_id}::aug:{transform_id}"
    )
    transformed_objects = tuple(
        _transform_object(
            obj,
            original_object_index=index,
            transform_id=transform_id,
        )
        for index, obj in enumerate(raw_example.objects)
    )
    if object_ordering == "geo_sorted":
        transformed_objects = tuple(
            sorted(
                transformed_objects,
                key=lambda obj: (obj.bbox[1], obj.bbox[0]),
            )
        )
    metadata = dict(raw_example.metadata)
    metadata["augmentation"] = {
        "mode": "static_stochastic_view",
        "policy": "geometry_flips",
        "policy_version": GEOMETRY_FLIP_POLICY_VERSION,
        "split": split,
        "source_example_id": raw_example.example_id,
        "source_example_index": source_index,
        "presentation_id": presentation_id,
        "transform_id": transform_id,
        "coord_matrix": _matrix_artifact(transform_id),
        "horizontal": transform_id in {"hflip", "hvflip"},
        "vertical": transform_id in {"vflip", "hvflip"},
        "transform_seed_source": transform_seed_source,
    }
    if object_order_seed is not None:
        metadata["augmentation"]["object_order_seed"] = object_order_seed
        metadata["augmentation"]["object_order_seed_source"] = (
            f"{object_order_seed}:{presentation_id}"
        )
    return RawExample(
        example_id=presentation_id,
        image=raw_example.image,
        objects=transformed_objects,
        metadata=metadata,
        source=raw_example.source,
    )


def _transform_object(
    obj: RawObject,
    *,
    original_object_index: int,
    transform_id: str,
) -> RawObject:
    transformed_bbox = transform_bbox(obj.bbox, transform_id)
    metadata = dict(obj.metadata)
    metadata["augmentation"] = {
        "original_object_index": original_object_index,
        "original_bbox": list(obj.bbox),
        "transformed_bbox": list(transformed_bbox),
        "transform_id": transform_id,
    }
    return RawObject(
        object_id=obj.object_id,
        description=obj.description,
        bbox=transformed_bbox,
        metadata=metadata,
    )


def _matrix_artifact(transform_id: str) -> list[list[int]]:
    try:
        matrix = COORD_AFFINE_MATRICES[transform_id]
    except KeyError as exc:
        raise DataContractError(
            "unsupported geometry transform id",
            code="augmentation.transform_id",
            context={"transform_id": transform_id},
            cause=exc,
        ) from exc
    return [list(row) for row in matrix]


__all__ = [
    "AugmentationMaterializationResult",
    "GeometryFlipAugmentationProcessor",
    "NoopAugmentationProcessor",
]
