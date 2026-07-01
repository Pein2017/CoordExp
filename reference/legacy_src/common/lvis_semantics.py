from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from src.common.semantic_desc import normalize_desc

LVIS_DATASET_POLICY = "lvis_federated"


def _normalize_frequency(value: Any) -> str:
    freq = str(value or "unknown").strip().lower()
    if freq in {"rare", "common", "frequent"}:
        return freq
    return "unknown"


@dataclass(frozen=True)
class LvisCategory:
    category_id: int
    name: str
    frequency: str
    norm_name: str


@dataclass(frozen=True)
class LvisImagePolicy:
    image_id: Any
    gt_objects: tuple[LvisCategory, ...]
    positive_categories: tuple[LvisCategory, ...]
    neg_categories: tuple[LvisCategory, ...]
    not_exhaustive_categories: tuple[LvisCategory, ...]
    positive_by_norm_name: Dict[str, LvisCategory]
    neg_by_norm_name: Dict[str, LvisCategory]
    not_exhaustive_by_norm_name: Dict[str, LvisCategory]

    def category_status_for_desc(
        self, desc: str
    ) -> tuple[Optional[str], Optional[LvisCategory]]:
        norm_desc = normalize_desc(str(desc or ""))
        if not norm_desc:
            return None, None
        positive = self.positive_by_norm_name.get(norm_desc)
        if positive is not None:
            return "verified_positive", positive
        negative = self.neg_by_norm_name.get(norm_desc)
        if negative is not None:
            return "verified_negative", negative
        not_exhaustive = self.not_exhaustive_by_norm_name.get(norm_desc)
        if not_exhaustive is not None:
            return "not_exhaustive", not_exhaustive
        return None, None

    def aligned_gt_category_ids(self) -> list[int]:
        return [int(item.category_id) for item in self.gt_objects]

    def aligned_gt_frequencies(self) -> list[str]:
        return [str(item.frequency) for item in self.gt_objects]


def _parse_category_entry(entry: Any) -> Optional[LvisCategory]:
    if not isinstance(entry, Mapping):
        return None
    raw_id = entry.get("id", entry.get("category_id"))
    raw_name = entry.get("name")
    try:
        category_id = int(raw_id)
    except (TypeError, ValueError):
        return None
    name = str(raw_name or "").strip()
    if not name:
        return None
    norm_name = normalize_desc(name)
    if not norm_name:
        return None
    return LvisCategory(
        category_id=int(category_id),
        name=name,
        frequency=_normalize_frequency(entry.get("frequency")),
        norm_name=norm_name,
    )


def is_lvis_federated_metadata(metadata: Any) -> bool:
    if not isinstance(metadata, Mapping):
        return False
    return str(metadata.get("dataset_policy") or "").strip().lower() == LVIS_DATASET_POLICY


def extract_lvis_image_policy(metadata: Any) -> Optional[LvisImagePolicy]:
    if not is_lvis_federated_metadata(metadata):
        return None
    lvis_meta = metadata.get("lvis")
    if not isinstance(lvis_meta, Mapping):
        return None

    def _parse_many(values: Any) -> tuple[LvisCategory, ...]:
        out = []
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            return tuple()
        for item in values:
            parsed = _parse_category_entry(item)
            if parsed is not None:
                out.append(parsed)
        return tuple(out)

    gt_objects = _parse_many(lvis_meta.get("gt_objects"))
    positive_categories = _parse_many(lvis_meta.get("positive_categories"))
    neg_categories = _parse_many(lvis_meta.get("neg_categories"))
    not_exhaustive_categories = _parse_many(lvis_meta.get("not_exhaustive_categories"))

    return LvisImagePolicy(
        image_id=metadata.get("image_id"),
        gt_objects=gt_objects,
        positive_categories=positive_categories,
        neg_categories=neg_categories,
        not_exhaustive_categories=not_exhaustive_categories,
        positive_by_norm_name={item.norm_name: item for item in positive_categories},
        neg_by_norm_name={item.norm_name: item for item in neg_categories},
        not_exhaustive_by_norm_name={
            item.norm_name: item for item in not_exhaustive_categories
        },
    )


def build_lvis_category_catalog(
    policies: Sequence[LvisImagePolicy],
) -> Dict[int, LvisCategory]:
    catalog: Dict[int, LvisCategory] = {}
    for policy in policies:
        for item in (
            list(policy.gt_objects)
            + list(policy.positive_categories)
            + list(policy.neg_categories)
            + list(policy.not_exhaustive_categories)
        ):
            catalog.setdefault(int(item.category_id), item)
    return catalog


def build_lvis_name_catalog(
    policies: Sequence[LvisImagePolicy],
) -> Dict[str, LvisCategory]:
    out: Dict[str, LvisCategory] = {}
    for item in build_lvis_category_catalog(policies).values():
        out.setdefault(str(item.norm_name), item)
    return out


__all__ = [
    "LVIS_DATASET_POLICY",
    "LvisCategory",
    "LvisImagePolicy",
    "build_lvis_category_catalog",
    "build_lvis_name_catalog",
    "extract_lvis_image_policy",
    "is_lvis_federated_metadata",
]
