"""Pure canonical-search and ``visual_policy_v1`` reference behavior.

This module deliberately returns presentation data separately from annotation
objects.  A browser implementation can consume the JSON forms without gaining
any route for aliases, free text, or visual metadata to enter dataset payloads.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from src.label_studio_coco_refinement.categories import COCO80_REGISTRY


VISUAL_POLICY_ID = "visual_policy_v1"
NEIGHBOR_EXPANSION_BINS = 12

# Fixed, dark-on-white, color-vision-deficiency-conscious presentation colors.
# Class text and numeric exhaustion badges remain visible, so color is never
# the sole carrier of category or instance identity.
ACCESSIBLE_HIGH_CONTRAST_PALETTE: tuple[str, ...] = (
    "#005A9C",
    "#A64073",
    "#007A5E",
    "#B24C00",
    "#5B4BB7",
    "#8A5A00",
    "#006B73",
    "#A32D2D",
)

EDITOR_POLICY_GOLDEN_VECTORS_SHA256 = (
    "6d02cec4cbfd4febb306dcf458121a6e3afc8433238fadf0958ff71c2a4d82d4"
)

# Fixed independently of the implementation under test.  The expected summaries
# and canonical-output hashes are intentionally literal so a policy mutation
# cannot rewrite its own golden answer.
_EDITOR_POLICY_GOLDEN_VECTORS_JSON = r"""
{
  "schema_version": "coordexp-editor-policy-golden-v1",
  "search": [
    {
      "expected": [{"canonical_name": "traffic light", "category_id": 10, "match_kind": "exact", "rank": 0, "spelling_distance": null}],
      "input": {"limit": 4, "query": "  TRAFFIC   light "},
      "name": "normalization_exact"
    },
    {
      "expected": [
        {"canonical_name": "car", "category_id": 3, "match_kind": "exact", "rank": 0, "spelling_distance": null},
        {"canonical_name": "carrot", "category_id": 57, "match_kind": "prefix", "rank": 1, "spelling_distance": null},
        {"canonical_name": "cat", "category_id": 17, "match_kind": "spelling", "rank": 2, "spelling_distance": 1}
      ],
      "input": {"limit": 5, "query": "car"},
      "name": "exact_prefix_then_spelling"
    },
    {
      "expected": [
        {"canonical_name": "snowboard", "category_id": 36, "match_kind": "substring", "rank": 0, "spelling_distance": null},
        {"canonical_name": "skateboard", "category_id": 41, "match_kind": "substring", "rank": 1, "spelling_distance": null},
        {"canonical_name": "surfboard", "category_id": 42, "match_kind": "substring", "rank": 2, "spelling_distance": null},
        {"canonical_name": "keyboard", "category_id": 76, "match_kind": "substring", "rank": 3, "spelling_distance": null}
      ],
      "input": {"limit": 4, "query": "board"},
      "name": "substring_registry_tie"
    },
    {
      "expected": [
        {"canonical_name": "traffic light", "category_id": 10, "match_kind": "spelling", "rank": 0, "spelling_distance": 2},
        {"canonical_name": "train", "category_id": 7, "match_kind": "spelling", "rank": 1, "spelling_distance": 2}
      ],
      "input": {"limit": 2, "query": "trafik"},
      "name": "spelling_distance_prefix_tie"
    },
    {
      "expected": [{"canonical_name": "traffic light", "category_id": 10, "match_kind": "spelling", "rank": 0, "spelling_distance": 1}],
      "input": {"limit": 3, "query": "trafic light"},
      "name": "phrase_typo"
    },
    {
      "expected": [],
      "input": {"limit": 3, "query": "\u4ea4\u901a\u706f"},
      "name": "noncanonical_chinese"
    }
  ],
  "visual_policy": [
    {
      "expected": {
        "duplicate_pairs": [],
        "neighbor_pairs": [["left", "right"]],
        "output_sha256": "ad26bb95d57f62e554ca35ee1c9fc083621b7a926093e34ff07facdc17368340",
        "palette": ["#005A9C", "#A64073", "#007A5E", "#B24C00", "#5B4BB7", "#8A5A00", "#006B73", "#A32D2D"],
        "presentations": [
          {"color": "#005A9C", "numeric_badge": null, "palette_index": 0, "stable_region_key": "left"},
          {"color": "#A64073", "numeric_badge": null, "palette_index": 1, "stable_region_key": "right"}
        ]
      },
      "input": [
        {"bbox_2d": [0, 0, 10, 10], "canonical_name": "person", "is_uncommitted_inference": true, "stable_region_key": "left"},
        {"bbox_2d": [34, 0, 50, 10], "canonical_name": "person", "is_uncommitted_inference": true, "stable_region_key": "right"}
      ],
      "name": "neighbor_gap_24_bins"
    },
    {
      "expected": {
        "duplicate_pairs": [],
        "neighbor_pairs": [],
        "output_sha256": "bf78f4ef8474855d1339499ce71eb99e1342298086fc9422dd4a566f6da53e19",
        "palette": ["#005A9C", "#A64073", "#007A5E", "#B24C00", "#5B4BB7", "#8A5A00", "#006B73", "#A32D2D"],
        "presentations": [
          {"color": "#005A9C", "numeric_badge": null, "palette_index": 0, "stable_region_key": "left"},
          {"color": "#005A9C", "numeric_badge": null, "palette_index": 0, "stable_region_key": "right"}
        ]
      },
      "input": [
        {"bbox_2d": [0, 0, 10, 10], "canonical_name": "person", "is_uncommitted_inference": true, "stable_region_key": "left"},
        {"bbox_2d": [35, 0, 50, 10], "canonical_name": "person", "is_uncommitted_inference": true, "stable_region_key": "right"}
      ],
      "name": "neighbor_gap_25_bins"
    },
    {
      "expected": {
        "duplicate_pairs": [],
        "neighbor_pair_count": 45,
        "neighbor_pairs_sha256": "c53eb1fcee370e3fdec63a648a942e0de73417eb2068d0937c7db481b0feacb9",
        "output_sha256": "e639c3952dd61a7b9a734fef80319ee68faf2734e1efa3b61cf8a77aee9f59ad",
        "palette": ["#005A9C", "#A64073", "#007A5E", "#B24C00", "#5B4BB7", "#8A5A00", "#006B73", "#A32D2D"],
        "presentations": [
          {"color": "#005A9C", "numeric_badge": null, "palette_index": 0, "stable_region_key": "region-00"},
          {"color": "#A64073", "numeric_badge": null, "palette_index": 1, "stable_region_key": "region-01"},
          {"color": "#007A5E", "numeric_badge": null, "palette_index": 2, "stable_region_key": "region-02"},
          {"color": "#B24C00", "numeric_badge": null, "palette_index": 3, "stable_region_key": "region-03"},
          {"color": "#5B4BB7", "numeric_badge": null, "palette_index": 4, "stable_region_key": "region-04"},
          {"color": "#8A5A00", "numeric_badge": null, "palette_index": 5, "stable_region_key": "region-05"},
          {"color": "#006B73", "numeric_badge": null, "palette_index": 6, "stable_region_key": "region-06"},
          {"color": "#A32D2D", "numeric_badge": null, "palette_index": 7, "stable_region_key": "region-07"},
          {"color": "#005A9C", "numeric_badge": 9, "palette_index": 0, "stable_region_key": "region-08"},
          {"color": "#A64073", "numeric_badge": 10, "palette_index": 1, "stable_region_key": "region-09"}
        ]
      },
      "input": [
        {"bbox_2d": [100, 100, 300, 300], "canonical_name": "person", "is_uncommitted_inference": true, "stable_region_key": "region-00"},
        {"bbox_2d": [100, 100, 300, 300], "canonical_name": "bicycle", "is_uncommitted_inference": true, "stable_region_key": "region-01"},
        {"bbox_2d": [100, 100, 300, 300], "canonical_name": "car", "is_uncommitted_inference": true, "stable_region_key": "region-02"},
        {"bbox_2d": [100, 100, 300, 300], "canonical_name": "motorcycle", "is_uncommitted_inference": true, "stable_region_key": "region-03"},
        {"bbox_2d": [100, 100, 300, 300], "canonical_name": "airplane", "is_uncommitted_inference": true, "stable_region_key": "region-04"},
        {"bbox_2d": [100, 100, 300, 300], "canonical_name": "bus", "is_uncommitted_inference": true, "stable_region_key": "region-05"},
        {"bbox_2d": [100, 100, 300, 300], "canonical_name": "train", "is_uncommitted_inference": true, "stable_region_key": "region-06"},
        {"bbox_2d": [100, 100, 300, 300], "canonical_name": "truck", "is_uncommitted_inference": true, "stable_region_key": "region-07"},
        {"bbox_2d": [100, 100, 300, 300], "canonical_name": "boat", "is_uncommitted_inference": true, "stable_region_key": "region-08"},
        {"bbox_2d": [100, 100, 300, 300], "canonical_name": "traffic light", "is_uncommitted_inference": true, "stable_region_key": "region-09"}
      ],
      "name": "palette_clique_exhaustion"
    },
    {
      "expected": {
        "duplicate_pairs": [["base", "half"]],
        "neighbor_pairs": [],
        "output_sha256": "9095cb5b3a278aace0a4b49387bee327b5d03a0024f298e9acc1b8720d0e312c",
        "palette": ["#005A9C", "#A64073", "#007A5E", "#B24C00", "#5B4BB7", "#8A5A00", "#006B73", "#A32D2D"],
        "presentations": [{"color": "#005A9C", "numeric_badge": null, "palette_index": 0, "stable_region_key": "half"}]
      },
      "input": [
        {"bbox_2d": [0, 0, 100, 100], "canonical_name": "person", "is_uncommitted_inference": false, "stable_region_key": "base"},
        {"bbox_2d": [0, 0, 100, 50], "canonical_name": "person", "is_uncommitted_inference": true, "stable_region_key": "half"}
      ],
      "name": "duplicate_exact_iou_half"
    }
  ]
}
"""


class MatchKind(str, Enum):
    EXACT = "exact"
    PREFIX = "prefix"
    SUBSTRING = "substring"
    SPELLING = "spelling"


@dataclass(frozen=True)
class ClassSearchResult:
    """One keyboard-selectable canonical result."""

    rank: int
    category_id: int
    canonical_name: str
    match_kind: MatchKind
    spelling_distance: int | None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "category_id": self.category_id,
            "canonical_name": self.canonical_name,
            "match_kind": self.match_kind.value,
            "spelling_distance": self.spelling_distance,
        }


def normalize_search_text(value: str) -> str:
    """Normalize case and repeated whitespace without inventing aliases."""

    if not isinstance(value, str):
        raise TypeError("search text must be a string")
    return " ".join(value.casefold().split())


def search_coco80_classes(
    query: str,
    *,
    limit: int | None = None,
) -> tuple[ClassSearchResult, ...]:
    """Rank only official COCO-80 English names for keyboard selection.

    Exact, prefix, and substring matches precede bounded spelling matches.
    Equal spelling distances prefer a longer shared prefix, then retain the
    frozen official registry order.  For a short partial typo, distance is also
    measured against same-length word windows (for example ``trafik`` against
    the ``traffic`` part of ``traffic light``).
    """

    normalized_query = normalize_search_text(query)
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise ValueError("limit must be a positive integer or None")

    scored: list[tuple[int, int, int, int, str, int]] = []
    for registry_index, category in enumerate(COCO80_REGISTRY.categories):
        normalized_name = normalize_search_text(category.name)
        match_kind: MatchKind
        spelling_distance = 0
        if normalized_name == normalized_query:
            match_kind = MatchKind.EXACT
        elif normalized_name.startswith(normalized_query):
            match_kind = MatchKind.PREFIX
        elif normalized_query in normalized_name:
            match_kind = MatchKind.SUBSTRING
        else:
            spelling_distance, spelling_prefix = _candidate_spelling_score(
                normalized_query,
                normalized_name,
            )
            if spelling_distance > _spelling_threshold(normalized_query):
                continue
            match_kind = MatchKind.SPELLING
        priority = {
            MatchKind.EXACT: 0,
            MatchKind.PREFIX: 1,
            MatchKind.SUBSTRING: 2,
            MatchKind.SPELLING: 3,
        }[match_kind]
        scored.append(
            (
                priority,
                spelling_distance if match_kind is MatchKind.SPELLING else 0,
                -spelling_prefix if match_kind is MatchKind.SPELLING else 0,
                registry_index,
                category.name,
                category.id,
            )
        )

    scored.sort(key=lambda item: item[:4])
    if limit is not None:
        scored = scored[:limit]
    return tuple(
        ClassSearchResult(
            rank=rank,
            category_id=category_id,
            canonical_name=canonical_name,
            match_kind=MatchKind(
                ("exact", "prefix", "substring", "spelling")[priority]
            ),
            spelling_distance=(distance if priority == 3 else None),
        )
        for rank, (
            priority,
            distance,
            _,
            _,
            canonical_name,
            category_id,
        ) in enumerate(scored)
    )


def canonical_class_payload(canonical_name: str) -> dict[str, Any]:
    """Return the only class fields permitted in a semantic object payload."""

    category = COCO80_REGISTRY.by_name(canonical_name)
    return {
        "desc": category.name,
        "category_name": category.name,
        "category_id": category.id,
    }


@dataclass(frozen=True)
class EditorRegion:
    """Minimal immutable input for visual policy derivation."""

    stable_region_key: str
    canonical_name: str
    bbox_2d: tuple[int, int, int, int]
    is_uncommitted_inference: bool

    def __post_init__(self) -> None:
        if not isinstance(self.stable_region_key, str) or not self.stable_region_key:
            raise ValueError("stable_region_key must be non-empty text")
        COCO80_REGISTRY.by_name(self.canonical_name)
        bbox = tuple(self.bbox_2d)
        if len(bbox) != 4 or any(
            isinstance(edge, bool) or not isinstance(edge, int) for edge in bbox
        ):
            raise ValueError("bbox_2d must contain four integer norm1000 edges")
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999):
            raise ValueError("bbox_2d must be strict xyxy on the 0..999 lattice")
        if not isinstance(self.is_uncommitted_inference, bool):
            raise ValueError("is_uncommitted_inference must be boolean")
        object.__setattr__(self, "bbox_2d", bbox)


@dataclass(frozen=True)
class RegionPresentation:
    stable_region_key: str
    color: str
    palette_index: int
    numeric_badge: int | None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stable_region_key": self.stable_region_key,
            "color": self.color,
            "palette_index": self.palette_index,
            "numeric_badge": self.numeric_badge,
        }


@dataclass(frozen=True)
class VisualPolicyResult:
    """Presentation-only output; no semantic annotation fields are present."""

    presentations: tuple[RegionPresentation, ...]
    neighbor_pairs: tuple[tuple[str, str], ...]
    duplicate_pairs: tuple[tuple[str, str], ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "policy_id": VISUAL_POLICY_ID,
            "neighbor_expansion_bins": NEIGHBOR_EXPANSION_BINS,
            "palette": list(ACCESSIBLE_HIGH_CONTRAST_PALETTE),
            "presentations": [item.to_json_dict() for item in self.presentations],
            "neighbor_pairs": [list(pair) for pair in self.neighbor_pairs],
            "duplicate_pairs": [list(pair) for pair in self.duplicate_pairs],
        }


def derive_visual_policy(regions: Iterable[EditorRegion]) -> VisualPolicyResult:
    """Derive deterministic colors and advisory duplicate cues.

    Colors and color-neighbor pairs cover only uncommitted inference-origin
    regions.  Duplicate cues cover every supplied active region so an inferred
    box can be compared with a pre-existing human/source box.
    """

    all_regions = tuple(regions)
    if any(not isinstance(region, EditorRegion) for region in all_regions):
        raise TypeError("regions must contain only EditorRegion values")
    by_key = {region.stable_region_key: region for region in all_regions}
    if len(by_key) != len(all_regions):
        raise ValueError("stable_region_key values must be unique")

    inference_regions = sorted(
        (region for region in all_regions if region.is_uncommitted_inference),
        key=lambda region: region.stable_region_key,
    )
    neighbor_pairs = tuple(
        (left.stable_region_key, right.stable_region_key)
        for index, left in enumerate(inference_regions)
        for right in inference_regions[index + 1 :]
        if _expanded_rectangles_intersect(left.bbox_2d, right.bbox_2d)
    )
    neighbors: dict[str, set[str]] = {
        region.stable_region_key: set() for region in inference_regions
    }
    for left_key, right_key in neighbor_pairs:
        neighbors[left_key].add(right_key)
        neighbors[right_key].add(left_key)

    assigned: dict[str, int] = {}
    presentations: list[RegionPresentation] = []
    for stable_index, region in enumerate(inference_regions, start=1):
        colored_neighbor_indices = [
            assigned[key]
            for key in neighbors[region.stable_region_key]
            if key in assigned
        ]
        used = set(colored_neighbor_indices)
        available = [
            index
            for index in range(len(ACCESSIBLE_HIGH_CONTRAST_PALETTE))
            if index not in used
        ]
        if available:
            palette_index = available[0]
            numeric_badge = None
        else:
            palette_index = min(
                range(len(ACCESSIBLE_HIGH_CONTRAST_PALETTE)),
                key=lambda index: (colored_neighbor_indices.count(index), index),
            )
            numeric_badge = stable_index
        assigned[region.stable_region_key] = palette_index
        presentations.append(
            RegionPresentation(
                stable_region_key=region.stable_region_key,
                color=ACCESSIBLE_HIGH_CONTRAST_PALETTE[palette_index],
                palette_index=palette_index,
                numeric_badge=numeric_badge,
            )
        )

    sorted_regions = sorted(all_regions, key=lambda region: region.stable_region_key)
    duplicate_pairs = tuple(
        (left.stable_region_key, right.stable_region_key)
        for index, left in enumerate(sorted_regions)
        for right in sorted_regions[index + 1 :]
        if left.canonical_name == right.canonical_name
        and _iou_at_least_half(left.bbox_2d, right.bbox_2d)
    )
    return VisualPolicyResult(
        presentations=tuple(presentations),
        neighbor_pairs=neighbor_pairs,
        duplicate_pairs=duplicate_pairs,
    )


def editor_policy_golden_vectors() -> dict[str, Any]:
    """Return a fresh copy of the fixed cross-language golden fixture."""

    return json.loads(_EDITOR_POLICY_GOLDEN_VECTORS_JSON)


def _candidate_spelling_score(query: str, candidate: str) -> tuple[int, int]:
    if not query:
        return len(candidate), 0
    query_words = query.split()
    candidate_words = candidate.split()
    window_size = len(query_words)
    comparisons = [candidate]
    if window_size <= len(candidate_words):
        comparisons.extend(
            " ".join(candidate_words[index : index + window_size])
            for index in range(len(candidate_words) - window_size + 1)
        )
    distance, negative_prefix = min(
        (
            _levenshtein_distance(query, value),
            -_common_prefix_length(query, value),
        )
        for value in comparisons
    )
    return distance, -negative_prefix


def _spelling_threshold(query: str) -> int:
    compact_length = len(query.replace(" ", ""))
    if compact_length == 0:
        return 0
    return min(3, max(1, math.ceil(compact_length / 3)))


def _levenshtein_distance(left: str, right: str) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for right_index, right_char in enumerate(right, start=1):
        current = [right_index]
        for left_index, left_char in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[left_index] + 1,
                    previous[left_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _common_prefix_length(left: str, right: str) -> int:
    return next(
        (index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
        min(len(left), len(right)),
    )


def _expanded_rectangles_intersect(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    left_expanded = _expand_bbox(left)
    right_expanded = _expand_bbox(right)
    return not (
        left_expanded[2] < right_expanded[0]
        or right_expanded[2] < left_expanded[0]
        or left_expanded[3] < right_expanded[1]
        or right_expanded[3] < left_expanded[1]
    )


def _expand_bbox(
    bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, x1 - NEIGHBOR_EXPANSION_BINS),
        max(0, y1 - NEIGHBOR_EXPANSION_BINS),
        min(999, x2 + NEIGHBOR_EXPANSION_BINS),
        min(999, y2 + NEIGHBOR_EXPANSION_BINS),
    )


def _iou_at_least_half(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    intersection_width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    if intersection == 0:
        return False
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return 2 * intersection >= union


__all__ = [
    "ACCESSIBLE_HIGH_CONTRAST_PALETTE",
    "EDITOR_POLICY_GOLDEN_VECTORS_SHA256",
    "NEIGHBOR_EXPANSION_BINS",
    "VISUAL_POLICY_ID",
    "ClassSearchResult",
    "EditorRegion",
    "MatchKind",
    "RegionPresentation",
    "VisualPolicyResult",
    "canonical_class_payload",
    "derive_visual_policy",
    "editor_policy_golden_vectors",
    "normalize_search_text",
    "search_coco80_classes",
]
