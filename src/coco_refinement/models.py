"""Native task, object, and Draft contracts for the standalone editor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from src.common.errors import DataContractError
from src.data.examples import JsonFrozen, freeze_json, thaw_json
from src.data.geometry import validate_bbox_bins
from src.label_studio_coco_refinement.categories import COCO80_REGISTRY

if TYPE_CHECKING:
    from src.label_studio_coco_refinement.runtime import AuthoritativeDraftSnapshot


Split = Literal["train", "val"]


@dataclass(frozen=True, order=True)
class NativeTaskIdentity:
    """Compact immutable identity for one exact-source row."""

    split: Split
    image_id: int
    source_row_index: int

    def __post_init__(self) -> None:
        _split(self.split)
        _integer(self.image_id, field="image_id", minimum=1)
        _integer(self.source_row_index, field="source_row_index", minimum=0)

    @property
    def task_key(self) -> str:
        return f"{self.split}:{self.image_id}"


@dataclass(frozen=True)
class NativeObject:
    """One canonical native norm1000 object; presentation is deliberately absent."""

    region_key: str
    bbox_2d: tuple[int, int, int, int]
    category_name: str
    category_id: int
    coco_ann_id: int | None = None
    metadata: Mapping[str, JsonFrozen] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.region_key, str) or not self.region_key:
            raise DataContractError(
                "region_key must be a non-empty string",
                code="coco_refinement.region_key",
            )
        object.__setattr__(
            self,
            "bbox_2d",
            validate_bbox_bins(self.bbox_2d, field="bbox_2d"),
        )
        COCO80_REGISTRY.validate(self.category_name, self.category_id)
        if self.coco_ann_id is not None:
            _integer(self.coco_ann_id, field="coco_ann_id", nonzero=True)
        if self.metadata is not None:
            if not isinstance(self.metadata, Mapping):
                raise DataContractError(
                    "object metadata must be a JSON object",
                    code="coco_refinement.metadata_shape",
                )
            object.__setattr__(self, "metadata", freeze_json(self.metadata))

    def to_json_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "region_key": self.region_key,
            "bbox_2d": list(self.bbox_2d),
            "category_name": self.category_name,
            "category_id": self.category_id,
        }
        if self.coco_ann_id is not None:
            value["coco_ann_id"] = self.coco_ann_id
        if self.metadata:
            value["metadata"] = thaw_json(self.metadata)
        return value

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "region_key": self.region_key,
            "bbox_2d": list(self.bbox_2d),
            "category_name": self.category_name,
            "category_id": self.category_id,
            "coco_ann_id": self.coco_ann_id,
        }


@dataclass(frozen=True)
class DraftSnapshotBinding:
    """Store-facing identity and generation fields outside native object semantics."""

    split: Split
    project_id: str
    image_id: int
    task_id: str
    annotation_id: str
    draft_id: str
    revision: int
    updated_at: str
    base_row_hash: str
    observed_generation: int

    def __post_init__(self) -> None:
        _split(self.split)
        _integer(self.image_id, field="image_id", minimum=1)
        _integer(self.revision, field="revision", minimum=0)
        _integer(self.observed_generation, field="observed_generation", minimum=0)
        for field in (
            "project_id",
            "task_id",
            "annotation_id",
            "draft_id",
            "updated_at",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise DataContractError(
                    f"{field} must be a non-empty string",
                    code="coco_refinement.snapshot_binding",
                    context={"field": field},
                )
        if (
            not isinstance(self.base_row_hash, str)
            or len(self.base_row_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.base_row_hash)
        ):
            raise DataContractError(
                "base_row_hash must be a lowercase SHA-256 digest",
                code="coco_refinement.base_row_hash",
            )


@dataclass(frozen=True)
class CanonicalDraft:
    """Canonical native payload plus the two deliberately distinct hashes."""

    split: Split
    objects: Sequence[NativeObject]
    semantic_hash: str
    result_hash: str
    inference_receipts: Sequence[str] = ()

    def __post_init__(self) -> None:
        _split(self.split)
        objects = tuple(self.objects)
        if any(not isinstance(item, NativeObject) for item in objects):
            raise DataContractError(
                "canonical Draft objects must be NativeObject values",
                code="coco_refinement.object_type",
            )
        object.__setattr__(self, "objects", objects)
        receipts = tuple(self.inference_receipts)
        if any(not isinstance(value, str) or not value for value in receipts):
            raise DataContractError(
                "inference receipt IDs must be non-empty strings",
                code="coco_refinement.receipt_id",
            )
        object.__setattr__(self, "inference_receipts", receipts)

    def to_json_regions(self) -> list[dict[str, Any]]:
        return [item.to_json_dict() for item in self.objects]

    def to_authoritative_snapshot(
        self, binding: DraftSnapshotBinding
    ) -> AuthoritativeDraftSnapshot:
        if binding.split != self.split:
            raise DataContractError(
                "Draft and snapshot binding splits differ",
                code="coco_refinement.snapshot_split",
                context={"draft": self.split, "binding": binding.split},
            )
        from src.label_studio_coco_refinement.runtime import (
            AuthoritativeDraftSnapshot,
        )

        return AuthoritativeDraftSnapshot(
            split=binding.split,
            project_id=binding.project_id,
            image_id=binding.image_id,
            task_id=binding.task_id,
            annotation_id=binding.annotation_id,
            draft_id=binding.draft_id,
            annotation_revision=str(binding.revision),
            draft_updated_at=binding.updated_at,
            semantic_hash=self.semantic_hash,
            result_hash=self.result_hash,
            base_row_hash=binding.base_row_hash,
            observed_generation=binding.observed_generation,
            regions=self.to_json_regions(),
            inference_receipts=self.inference_receipts,
        )


def _split(value: object) -> Split:
    if value not in ("train", "val"):
        raise DataContractError(
            "split must be train or val",
            code="coco_refinement.split",
            context={"split": value},
        )
    return value  # type: ignore[return-value]


def _integer(
    value: object,
    *,
    field: str,
    minimum: int | None = None,
    nonzero: bool = False,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataContractError(
            f"{field} must be an integer",
            code="coco_refinement.integer",
            context={"field": field, "value": value},
        )
    if minimum is not None and value < minimum:
        raise DataContractError(
            f"{field} is below its minimum",
            code="coco_refinement.integer_range",
            context={"field": field, "value": value, "minimum": minimum},
        )
    if nonzero and value == 0:
        raise DataContractError(
            f"{field} must be nonzero",
            code="coco_refinement.integer_range",
            context={"field": field, "value": value},
        )
    return value


__all__ = [
    "CanonicalDraft",
    "DraftSnapshotBinding",
    "NativeObject",
    "NativeTaskIdentity",
    "Split",
]
