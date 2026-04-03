from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .semantic_desc import normalize_desc

_DUPLICATE_SPREAD_EXEMPT_MULTIPLIER = 2.5


@dataclass(frozen=True)
class DuplicateControlConfig:
    iou_threshold: float
    center_radius_scale: float


@dataclass(frozen=True)
class DuplicateControlObject:
    index: int
    desc: str
    desc_norm: str
    bbox_norm1000: Tuple[int, int, int, int]
    center_norm1000: Tuple[float, float]
    area_bins: float
    area_ratio: float
    border_saturated: bool
    source: str = "anchor"


@dataclass(frozen=True)
class DuplicateControlCluster:
    cluster_id: int
    member_indices: Tuple[int, ...]
    survivor_index: int
    suppressed_indices: Tuple[int, ...]
    is_exempt: bool
    exemption_reasons: Tuple[str, ...]
    max_center_distance: float


@dataclass(frozen=True)
class DuplicateControlDecision:
    object_index: int
    cluster_id: Optional[int]
    action: str
    survivor_index: int
    support_count: int
    support_rate: float
    border_saturated: bool
    is_exempt: bool
    exemption_reasons: Tuple[str, ...]


@dataclass(frozen=True)
class DuplicateControlResult:
    config: DuplicateControlConfig
    objects: Tuple[DuplicateControlObject, ...]
    clusters: Tuple[DuplicateControlCluster, ...]
    decisions: Tuple[DuplicateControlDecision, ...]
    kept_indices: Tuple[int, ...]
    suppressed_indices: Tuple[int, ...]
    exempt_indices: Tuple[int, ...]
    survivor_indices: Tuple[int, ...]
    support_counts: Tuple[int, ...]
    support_rates: Tuple[float, ...]
    raw_metrics: Dict[str, float]
    counter_metrics: Dict[str, float]


def _validate_threshold(name: str, value: float, *, low: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a float/int") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if parsed < float(low):
        raise ValueError(f"{name} must be >= {low}")
    return float(parsed)


def validate_duplicate_control_config(
    *,
    iou_threshold: float,
    center_radius_scale: float,
) -> DuplicateControlConfig:
    iou = _validate_threshold("iou_threshold", iou_threshold)
    if iou > 1.0:
        raise ValueError("iou_threshold must be <= 1.0")
    radius = _validate_threshold("center_radius_scale", center_radius_scale, low=0.0)
    return DuplicateControlConfig(
        iou_threshold=float(iou),
        center_radius_scale=float(radius),
    )


def _bbox_from_points(points: Sequence[int]) -> Optional[Tuple[int, int, int, int]]:
    if len(points) < 4 or len(points) % 2 != 0:
        return None
    xs = [int(points[idx]) for idx in range(0, len(points), 2)]
    ys = [int(points[idx]) for idx in range(1, len(points), 2)]
    if not xs or not ys:
        return None
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return int(x1), int(y1), int(x2), int(y2)


def _pixel_to_norm1000(
    bbox_xyxy: Sequence[float],
    *,
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    if int(width) <= 0 or int(height) <= 0 or len(bbox_xyxy) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    bins = [
        int(round((x1 / float(width)) * 1000.0)),
        int(round((y1 / float(height)) * 1000.0)),
        int(round((x2 / float(width)) * 1000.0)),
        int(round((y2 / float(height)) * 1000.0)),
    ]
    bins = [min(999, max(0, int(value))) for value in bins]
    if bins[2] <= bins[0] or bins[3] <= bins[1]:
        return None
    return int(bins[0]), int(bins[1]), int(bins[2]), int(bins[3])


def duplicate_control_object_from_bbox(
    *,
    index: int,
    desc: str,
    bbox_norm1000: Sequence[int],
    source: str,
) -> DuplicateControlObject:
    if len(bbox_norm1000) != 4:
        raise ValueError("bbox_norm1000 must contain exactly four values")
    x1, y1, x2, y2 = [int(value) for value in bbox_norm1000]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_norm1000 must be non-degenerate")
    width_bins = max(0, int(x2) - int(x1))
    height_bins = max(0, int(y2) - int(y1))
    area_bins = float(width_bins * height_bins)
    center_x = float(x1 + x2) / 2.0
    center_y = float(y1 + y2) / 2.0
    return DuplicateControlObject(
        index=int(index),
        desc=str(desc or ""),
        desc_norm=normalize_desc(str(desc or "")),
        bbox_norm1000=(int(x1), int(y1), int(x2), int(y2)),
        center_norm1000=(float(center_x), float(center_y)),
        area_bins=float(area_bins),
        area_ratio=float(area_bins / 1_000_000.0),
        border_saturated=any(int(value) in {0, 999} for value in (x1, y1, x2, y2)),
        source=str(source),
    )


def duplicate_control_object_from_mapping(
    obj: Mapping[str, Any],
    *,
    index: int,
    width: Optional[int],
    height: Optional[int],
    source: str,
) -> Optional[DuplicateControlObject]:
    bbox_norm1000_raw = obj.get("bbox_norm1000")
    bbox_norm1000: Optional[Tuple[int, int, int, int]] = None
    if isinstance(bbox_norm1000_raw, Sequence) and len(bbox_norm1000_raw) == 4:
        bbox_norm1000 = tuple(int(value) for value in bbox_norm1000_raw)
    elif isinstance(obj.get("points_norm1000"), Sequence):
        points_norm1000 = obj.get("points_norm1000")
        if isinstance(points_norm1000, Sequence):
            bbox_norm1000 = _bbox_from_points([int(value) for value in points_norm1000])
    elif isinstance(obj.get("bbox"), Sequence) and width is not None and height is not None:
        bbox_norm1000 = _pixel_to_norm1000(
            obj.get("bbox"),
            width=int(width),
            height=int(height),
        )
    elif isinstance(obj.get("points"), Sequence) and width is not None and height is not None:
        bbox_px = _bbox_from_points([int(value) for value in obj.get("points")])
        if bbox_px is not None:
            bbox_norm1000 = _pixel_to_norm1000(
                bbox_px,
                width=int(width),
                height=int(height),
            )
    if bbox_norm1000 is None:
        return None
    try:
        return duplicate_control_object_from_bbox(
            index=int(index),
            desc=str(obj.get("desc") or ""),
            bbox_norm1000=bbox_norm1000,
            source=str(source),
        )
    except ValueError:
        return None


def _bbox_iou_xyxy(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = [int(value) for value in a]
    bx1, by1, bx2, by2 = [int(value) for value in b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = float(inter_w * inter_h)
    if inter <= 0.0:
        return 0.0
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def pair_is_duplicate_like(
    a: DuplicateControlObject,
    b: DuplicateControlObject,
    *,
    config: DuplicateControlConfig,
) -> bool:
    if str(a.desc_norm or "") != str(b.desc_norm or ""):
        return False
    iou = _bbox_iou_xyxy(a.bbox_norm1000, b.bbox_norm1000)
    if float(iou) >= float(config.iou_threshold):
        return True
    dist = math.hypot(
        float(a.center_norm1000[0]) - float(b.center_norm1000[0]),
        float(a.center_norm1000[1]) - float(b.center_norm1000[1]),
    )
    ref = math.sqrt(max(1.0, min(float(a.area_bins), float(b.area_bins))))
    return bool(dist <= float(config.center_radius_scale) * ref)


def build_duplicate_clusters(
    objects: Sequence[DuplicateControlObject],
    *,
    config: DuplicateControlConfig,
) -> List[Tuple[int, ...]]:
    adjacency: Dict[int, set[int]] = {int(obj.index): set() for obj in objects}
    for left_pos, left in enumerate(objects):
        for right in objects[left_pos + 1 :]:
            if not pair_is_duplicate_like(left, right, config=config):
                continue
            adjacency[int(left.index)].add(int(right.index))
            adjacency[int(right.index)].add(int(left.index))

    components: List[Tuple[int, ...]] = []
    visited: set[int] = set()
    for obj in objects:
        node = int(obj.index)
        if node in visited or not adjacency[node]:
            continue
        stack = [node]
        visited.add(node)
        component: List[int] = []
        while stack:
            current = int(stack.pop())
            component.append(current)
            for neighbor in sorted(adjacency.get(current, set())):
                if int(neighbor) in visited:
                    continue
                visited.add(int(neighbor))
                stack.append(int(neighbor))
        components.append(tuple(sorted(component)))
    return components


def compute_duplicate_metrics(
    objects: Sequence[DuplicateControlObject],
    *,
    config: DuplicateControlConfig,
) -> Dict[str, float]:
    norm_descs = [str(obj.desc_norm or "") for obj in objects]
    max_desc_count = 0
    if norm_descs:
        counts = Counter(norm_descs)
        max_desc_count = int(max(counts.values()))

    entropy = 0.0
    total = float(len(norm_descs))
    if total > 0.0:
        counts = Counter(norm_descs)
        for count in counts.values():
            p = float(count) / total
            entropy -= p * math.log(max(p, 1e-12))

    saturated = sum(1 for obj in objects if bool(obj.border_saturated))
    near_same_desc = 0
    near_any_desc = 0
    for left_pos, left in enumerate(objects):
        for right in objects[left_pos + 1 :]:
            iou = _bbox_iou_xyxy(left.bbox_norm1000, right.bbox_norm1000)
            if iou < 0.90:
                continue
            near_any_desc += 1
            if left.desc_norm == right.desc_norm:
                near_same_desc += 1

    clusters = build_duplicate_clusters(objects, config=config)
    max_cluster_size = 0
    if objects:
        max_cluster_size = int(
            max((len(cluster) for cluster in clusters), default=1)
        )

    return {
        "dup/raw/max_desc_count": float(max_desc_count),
        "dup/raw/saturation_rate": (
            float(saturated) / float(len(objects)) if objects else 0.0
        ),
        "dup/raw/duplicate_like_max_cluster_size": float(max_cluster_size),
        "dup/raw/desc_entropy": float(entropy),
        "dup/raw/near_iou90_pairs_same_desc_count": float(near_same_desc),
        "dup/raw/near_iou90_pairs_any_desc_count": float(near_any_desc),
    }


def _max_weight_pair_sum(
    candidates: Sequence[Tuple[int, int, float]],
    *,
    tol: float = 1e-9,
) -> float:
    best_by_anchor: Dict[int, float] = {}
    best_by_explorer: Dict[int, float] = {}
    for anchor_i, explorer_i, score in candidates:
        best_by_anchor[int(anchor_i)] = max(
            float(best_by_anchor.get(int(anchor_i), 0.0)),
            float(score),
        )
        best_by_explorer[int(explorer_i)] = max(
            float(best_by_explorer.get(int(explorer_i), 0.0)),
            float(score),
        )
    upper = min(sum(best_by_anchor.values()), sum(best_by_explorer.values()))
    if upper <= float(tol):
        return 0.0

    ordered = sorted(
        candidates,
        key=lambda item: (-float(item[2]), int(item[0]), int(item[1])),
    )
    best = 0.0

    def _search(
        start: int,
        used_anchor: set[int],
        used_explorer: set[int],
        total: float,
    ) -> None:
        nonlocal best
        if float(total) > float(best):
            best = float(total)
        if start >= len(ordered):
            return
        remaining = float(total)
        available_anchor: set[int] = set()
        available_explorer: set[int] = set()
        for anchor_i, explorer_i, score in ordered[start:]:
            if int(anchor_i) in used_anchor or int(explorer_i) in used_explorer:
                continue
            if int(anchor_i) not in available_anchor:
                remaining += float(score)
                available_anchor.add(int(anchor_i))
            available_explorer.add(int(explorer_i))
        if remaining <= float(best) + float(tol):
            return
        for position in range(int(start), len(ordered)):
            anchor_i, explorer_i, score = ordered[position]
            if int(anchor_i) in used_anchor or int(explorer_i) in used_explorer:
                continue
            used_anchor.add(int(anchor_i))
            used_explorer.add(int(explorer_i))
            _search(
                position + 1,
                used_anchor,
                used_explorer,
                float(total) + float(score),
            )
            used_anchor.remove(int(anchor_i))
            used_explorer.remove(int(explorer_i))

    _search(0, set(), set(), 0.0)
    return float(best)


def _lexicographic_max_weight_pairs(
    *,
    candidates: Sequence[Tuple[int, int, float]],
    tol: float = 1e-9,
) -> List[Tuple[int, int]]:
    best_total = _max_weight_pair_sum(candidates=candidates, tol=float(tol))
    if best_total <= float(tol):
        return []

    ordered = sorted(
        candidates,
        key=lambda item: (int(item[0]), int(item[1])),
    )
    for anchor_i, explorer_i, score in ordered:
        residual = [
            (int(next_anchor_i), int(next_explorer_i), float(next_score))
            for next_anchor_i, next_explorer_i, next_score in ordered
            if int(next_anchor_i) != int(anchor_i)
            and int(next_explorer_i) != int(explorer_i)
            and (int(next_anchor_i), int(next_explorer_i))
            > (int(anchor_i), int(explorer_i))
        ]
        with_pair_total = float(score) + _max_weight_pair_sum(
            residual,
            tol=float(tol),
        )
        if abs(float(with_pair_total) - float(best_total)) <= float(tol):
            return [(int(anchor_i), int(explorer_i))] + _lexicographic_max_weight_pairs(
                candidates=residual,
                tol=float(tol),
            )
    return []


def _associate_explorer_support(
    *,
    anchor_objects: Sequence[DuplicateControlObject],
    explorer_objects_by_view: Sequence[Sequence[DuplicateControlObject]],
    support_iou_threshold: float,
) -> List[int]:
    threshold = _validate_threshold(
        "support_iou_threshold",
        support_iou_threshold,
        low=0.0,
    )
    if threshold > 1.0:
        raise ValueError("support_iou_threshold must be <= 1.0")
    support_counts = [0 for _ in range(len(anchor_objects))]
    if not anchor_objects:
        return support_counts

    anchor_pos_by_index = {
        int(obj.index): int(pos) for pos, obj in enumerate(anchor_objects)
    }
    for explorer_objects in explorer_objects_by_view:
        candidates: List[Tuple[int, int, float]] = []
        for anchor in anchor_objects:
            for explorer in explorer_objects:
                if str(anchor.desc_norm or "") != str(explorer.desc_norm or ""):
                    continue
                iou = _bbox_iou_xyxy(anchor.bbox_norm1000, explorer.bbox_norm1000)
                if float(iou) < float(threshold):
                    continue
                candidates.append((int(anchor.index), int(explorer.index), float(iou)))
        for anchor_i, _explorer_i in _lexicographic_max_weight_pairs(
            candidates=candidates
        ):
            anchor_pos = anchor_pos_by_index.get(int(anchor_i))
            if anchor_pos is None:
                continue
            support_counts[int(anchor_pos)] += 1
    return support_counts


def _cluster_max_center_distance(
    member_objects: Sequence[DuplicateControlObject],
) -> float:
    max_distance = 0.0
    for left_pos, left in enumerate(member_objects):
        for right in member_objects[left_pos + 1 :]:
            max_distance = max(
                float(max_distance),
                float(
                    math.hypot(
                        float(left.center_norm1000[0]) - float(right.center_norm1000[0]),
                        float(left.center_norm1000[1]) - float(right.center_norm1000[1]),
                    )
                ),
            )
    return float(max_distance)


def apply_duplicate_policy(
    *,
    anchor_objects: Sequence[DuplicateControlObject],
    explorer_objects_by_view: Sequence[Sequence[DuplicateControlObject]],
    config: DuplicateControlConfig,
    support_iou_threshold: float,
) -> DuplicateControlResult:
    ordered_anchor_objects = sorted(anchor_objects, key=lambda item: int(item.index))
    raw_metrics = compute_duplicate_metrics(ordered_anchor_objects, config=config)
    support_counts = _associate_explorer_support(
        anchor_objects=ordered_anchor_objects,
        explorer_objects_by_view=explorer_objects_by_view,
        support_iou_threshold=float(support_iou_threshold),
    )
    valid_explorer_count = int(len(explorer_objects_by_view))
    support_rates = [
        (
            float(int(count)) / float(valid_explorer_count)
            if valid_explorer_count > 0
            else 0.0
        )
        for count in support_counts
    ]
    support_count_by_index = {
        int(obj.index): int(support_counts[pos])
        for pos, obj in enumerate(ordered_anchor_objects)
    }
    support_rate_by_index = {
        int(obj.index): float(support_rates[pos])
        for pos, obj in enumerate(ordered_anchor_objects)
    }

    decisions_by_index: Dict[int, DuplicateControlDecision] = {}
    clusters_out: List[DuplicateControlCluster] = []
    kept_indices: set[int] = set()
    suppressed_indices: set[int] = set()
    exempt_indices: set[int] = set()
    survivor_indices: set[int] = set()

    raw_clusters = build_duplicate_clusters(ordered_anchor_objects, config=config)
    object_by_index = {
        int(obj.index): obj for obj in ordered_anchor_objects
    }
    raw_cluster_member_indices = {
        int(member_index)
        for cluster in raw_clusters
        for member_index in cluster
    }

    for cluster_id, member_indices in enumerate(raw_clusters):
        member_objects = [
            object_by_index[int(member_index)] for member_index in member_indices
        ]
        max_center_distance = _cluster_max_center_distance(member_objects)
        area_ref = math.sqrt(
            max(1.0, max(float(member.area_bins) for member in member_objects))
        )
        spread_radius = (
            float(_DUPLICATE_SPREAD_EXEMPT_MULTIPLIER)
            * float(config.center_radius_scale)
            * float(area_ref)
        )
        spatially_spread = bool(
            float(config.center_radius_scale) > 0.0
            and max_center_distance > float(spread_radius)
        )
        explorer_supported_members = [
            int(member_index)
            for member_index in member_indices
            if int(support_count_by_index.get(int(member_index), 0)) > 0
        ]
        exemption_reasons: List[str] = []
        if len(explorer_supported_members) >= 2:
            exemption_reasons.append("explorer_supported")
        if spatially_spread:
            exemption_reasons.append("spatially_spread")

        survivor_index = min(
            member_indices,
            key=lambda member_index: (
                -int(support_count_by_index.get(int(member_index), 0)),
                1 if bool(object_by_index[int(member_index)].border_saturated) else 0,
                int(member_index),
            ),
        )
        is_exempt = bool(exemption_reasons)
        suppressed_members = tuple(
            int(member_index)
            for member_index in member_indices
            if not is_exempt and int(member_index) != int(survivor_index)
        )

        cluster = DuplicateControlCluster(
            cluster_id=int(cluster_id),
            member_indices=tuple(int(member_index) for member_index in member_indices),
            survivor_index=int(survivor_index),
            suppressed_indices=tuple(int(member_index) for member_index in suppressed_members),
            is_exempt=bool(is_exempt),
            exemption_reasons=tuple(str(reason) for reason in exemption_reasons),
            max_center_distance=float(max_center_distance),
        )
        clusters_out.append(cluster)

        for member_index in member_indices:
            member_object = object_by_index[int(member_index)]
            if is_exempt:
                action = "keep"
                exempt_indices.add(int(member_index))
                kept_indices.add(int(member_index))
            elif int(member_index) == int(survivor_index):
                action = "keep"
                kept_indices.add(int(member_index))
                survivor_indices.add(int(member_index))
            else:
                action = "suppress"
                suppressed_indices.add(int(member_index))
            decisions_by_index[int(member_index)] = DuplicateControlDecision(
                object_index=int(member_index),
                cluster_id=int(cluster_id),
                action=str(action),
                survivor_index=int(survivor_index),
                support_count=int(support_count_by_index.get(int(member_index), 0)),
                support_rate=float(support_rate_by_index.get(int(member_index), 0.0)),
                border_saturated=bool(member_object.border_saturated),
                is_exempt=bool(is_exempt),
                exemption_reasons=tuple(str(reason) for reason in exemption_reasons),
            )

    for obj in ordered_anchor_objects:
        if int(obj.index) in raw_cluster_member_indices:
            continue
        kept_indices.add(int(obj.index))
        decisions_by_index[int(obj.index)] = DuplicateControlDecision(
            object_index=int(obj.index),
            cluster_id=None,
            action="keep",
            survivor_index=int(obj.index),
            support_count=int(support_count_by_index.get(int(obj.index), 0)),
            support_rate=float(support_rate_by_index.get(int(obj.index), 0.0)),
            border_saturated=bool(obj.border_saturated),
            is_exempt=False,
            exemption_reasons=(),
        )

    decisions = tuple(
        decisions_by_index[int(obj.index)] for obj in ordered_anchor_objects
    )
    counter_metrics = {
        "stage2_ab/channel_b/dup/N_clusters_total": float(len(clusters_out)),
        "stage2_ab/channel_b/dup/N_clusters_exempt": float(
            sum(1 for cluster in clusters_out if bool(cluster.is_exempt))
        ),
        "stage2_ab/channel_b/dup/N_clusters_suppressed": float(
            sum(1 for cluster in clusters_out if not bool(cluster.is_exempt))
        ),
        "stage2_ab/channel_b/dup/N_objects_suppressed": float(len(suppressed_indices)),
    }
    return DuplicateControlResult(
        config=config,
        objects=tuple(ordered_anchor_objects),
        clusters=tuple(clusters_out),
        decisions=decisions,
        kept_indices=tuple(sorted(int(index) for index in kept_indices)),
        suppressed_indices=tuple(sorted(int(index) for index in suppressed_indices)),
        exempt_indices=tuple(sorted(int(index) for index in exempt_indices)),
        survivor_indices=tuple(sorted(int(index) for index in survivor_indices)),
        support_counts=tuple(int(count) for count in support_counts),
        support_rates=tuple(float(rate) for rate in support_rates),
        raw_metrics=dict(raw_metrics),
        counter_metrics=counter_metrics,
    )


__all__ = [
    "DuplicateControlCluster",
    "DuplicateControlConfig",
    "DuplicateControlDecision",
    "DuplicateControlObject",
    "DuplicateControlResult",
    "apply_duplicate_policy",
    "build_duplicate_clusters",
    "compute_duplicate_metrics",
    "duplicate_control_object_from_bbox",
    "duplicate_control_object_from_mapping",
    "pair_is_duplicate_like",
    "validate_duplicate_control_config",
]
