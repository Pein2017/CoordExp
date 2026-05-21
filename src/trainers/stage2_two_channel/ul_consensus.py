from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence


Decision = Literal["promoted", "rejected", "quarantined"]
Box = tuple[float, float, float, float]


def _immutable_member_mapping(mapping: Mapping[str, Sequence["ULMember"]]) -> Mapping[str, tuple["ULMember", ...]]:
    return MappingProxyType(
        {
            str(key): tuple(sorted(value, key=_member_sort_key))
            for key, value in sorted(mapping.items(), key=lambda item: str(item[0]))
        }
    )


def _immutable_mapping_tuple(items: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    return tuple(MappingProxyType(dict(item)) for item in items)


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite real number")
    return converted


def _normalize_box(box: Sequence[float]) -> Box:
    try:
        values = tuple(_finite_float(value, name="bbox_norm1000") for value in box)
    except TypeError as exc:
        raise ValueError("bbox_norm1000 must contain exactly x1/y1/x2/y2") from exc
    if len(values) != 4:
        raise ValueError("bbox_norm1000 must contain exactly x1/y1/x2/y2")
    x1, y1, x2, y2 = values
    if any(value < 0.0 or value > 999.0 for value in values):
        raise ValueError("bbox_norm1000 coordinates must be in [0, 999]")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_norm1000 must be nondegenerate xyxy")
    return values  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ULGeometryConfig:
    iou_min: float
    center_distance_scale_max: float
    area_ratio_max: float
    aspect_ratio_max: float
    consumed_overlap_iou_min: float
    gray_iou_min: float | None = None
    duplicate_burst_iou_min: float | None = None

    def __post_init__(self) -> None:
        if self.gray_iou_min is None:
            object.__setattr__(self, "gray_iou_min", self.iou_min)
        if self.duplicate_burst_iou_min is None:
            object.__setattr__(self, "duplicate_burst_iou_min", self.iou_min)
        for name in (
            "iou_min",
            "center_distance_scale_max",
            "area_ratio_max",
            "aspect_ratio_max",
            "consumed_overlap_iou_min",
            "gray_iou_min",
            "duplicate_burst_iou_min",
        ):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name=name))
        if not 0.0 <= self.iou_min <= 1.0:
            raise ValueError("iou_min must be in [0, 1]")
        if not 0.0 <= self.gray_iou_min <= self.iou_min:
            raise ValueError("gray_iou_min must be in [0, iou_min]")
        if not 0.0 <= self.duplicate_burst_iou_min <= 1.0:
            raise ValueError("duplicate_burst_iou_min must be in [0, 1]")
        if not 0.0 <= self.consumed_overlap_iou_min <= 1.0:
            raise ValueError("consumed_overlap_iou_min must be in [0, 1]")
        if self.center_distance_scale_max < 0.0:
            raise ValueError("center_distance_scale_max must be nonnegative")
        if self.area_ratio_max < 1.0:
            raise ValueError("area_ratio_max must be at least 1")
        if self.aspect_ratio_max < 1.0:
            raise ValueError("aspect_ratio_max must be at least 1")


@dataclass(frozen=True, slots=True)
class ULMember:
    rollout_id: str
    local_index: int
    desc_id: str
    desc_text: str
    bbox_norm1000: Box

    def __post_init__(self) -> None:
        object.__setattr__(self, "rollout_id", str(self.rollout_id))
        if isinstance(self.local_index, bool) or not isinstance(self.local_index, int):
            raise ValueError("local_index must be an integer")
        if self.local_index < 0:
            raise ValueError("local_index must be nonnegative")
        object.__setattr__(self, "desc_id", str(self.desc_id))
        object.__setattr__(self, "desc_text", str(self.desc_text))
        object.__setattr__(self, "bbox_norm1000", _normalize_box(self.bbox_norm1000))


@dataclass(frozen=True, slots=True)
class ULRolloutEvidence:
    rollout_id: str
    is_valid: bool
    skip_reason: str | None
    unmatched_members: tuple[ULMember, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rollout_id", str(self.rollout_id))
        object.__setattr__(self, "is_valid", bool(self.is_valid))
        if self.skip_reason is not None:
            object.__setattr__(self, "skip_reason", str(self.skip_reason))
        members = tuple(
            member
            if member.rollout_id == self.rollout_id
            else ULMember(
                rollout_id=self.rollout_id,
                local_index=member.local_index,
                desc_id=member.desc_id,
                desc_text=member.desc_text,
                bbox_norm1000=member.bbox_norm1000,
            )
            for member in self.unmatched_members
        )
        object.__setattr__(self, "unmatched_members", tuple(sorted(members, key=_member_sort_key)))


@dataclass(frozen=True, slots=True)
class ULConsensusCluster:
    desc_id: str
    desc_text: str
    support_rollout_ids: tuple[str, ...]
    support_ratio: float
    decision: Decision
    reason: str
    members_by_rollout: Mapping[str, tuple[ULMember, ...]]
    pairwise_geometry: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    consumed_overlap: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "desc_id", str(self.desc_id))
        object.__setattr__(self, "desc_text", str(self.desc_text))
        object.__setattr__(self, "support_rollout_ids", tuple(sorted(str(item) for item in self.support_rollout_ids)))
        object.__setattr__(self, "support_ratio", _finite_float(self.support_ratio, name="support_ratio"))
        if self.decision not in {"promoted", "rejected", "quarantined"}:
            raise ValueError("decision must be promoted, rejected, or quarantined")
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "members_by_rollout", _immutable_member_mapping(self.members_by_rollout))
        object.__setattr__(self, "pairwise_geometry", _immutable_mapping_tuple(self.pairwise_geometry))
        object.__setattr__(self, "consumed_overlap", _immutable_mapping_tuple(self.consumed_overlap))


@dataclass(frozen=True, slots=True)
class ULConsensusResult:
    k_valid: int
    min_ul_valid_rollouts: int
    consensus_ratio: float
    geometry: ULGeometryConfig
    skip_reasons: Mapping[str, int]
    promoted_clusters: tuple[ULConsensusCluster, ...]
    rejected_clusters: tuple[ULConsensusCluster, ...]
    quarantined_clusters: tuple[ULConsensusCluster, ...]
    duplicate_like_suppressed_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_valid", int(self.k_valid))
        object.__setattr__(
            self,
            "min_ul_valid_rollouts",
            _validate_min_ul_valid_rollouts(self.min_ul_valid_rollouts),
        )
        object.__setattr__(self, "consensus_ratio", _finite_float(self.consensus_ratio, name="consensus_ratio"))
        object.__setattr__(
            self,
            "skip_reasons",
            MappingProxyType({str(key): int(value) for key, value in sorted(self.skip_reasons.items())}),
        )
        object.__setattr__(self, "promoted_clusters", _sort_clusters(self.promoted_clusters))
        object.__setattr__(self, "rejected_clusters", _sort_clusters(self.rejected_clusters))
        object.__setattr__(self, "quarantined_clusters", _sort_clusters(self.quarantined_clusters))
        object.__setattr__(self, "duplicate_like_suppressed_count", int(self.duplicate_like_suppressed_count))


def mine_ul_consensus(
    rollouts: Sequence[ULRolloutEvidence],
    *,
    min_ul_valid_rollouts: int,
    consensus_ratio: float,
    geometry: ULGeometryConfig,
    consumed_members: Sequence[ULMember] = (),
) -> ULConsensusResult:
    if consensus_ratio != 1.0:
        raise ValueError("mine_ul_consensus currently supports only consensus_ratio == 1.0")

    min_ul_valid_rollouts = _validate_min_ul_valid_rollouts(min_ul_valid_rollouts)
    rollouts = tuple(sorted(rollouts, key=_rollout_sort_key))
    valid_rollouts = tuple(rollout for rollout in rollouts if rollout.is_valid)
    k_valid = len(valid_rollouts)
    skip_reasons = _count_skip_reasons(rollouts)

    raw_clusters_by_desc: dict[str, list[_WorkingCluster]] = {}
    duplicate_like_suppressed_count = 0
    for rollout in valid_rollouts:
        for member in rollout.unmatched_members:
            clusters = raw_clusters_by_desc.setdefault(member.desc_id, [])
            duplicate_cluster = _find_same_rollout_duplicate_cluster(clusters, member, geometry)
            if duplicate_cluster is not None:
                duplicate_like_suppressed_count += 1
                duplicate_cluster.keep_earliest(member)
                continue

            compatible_cluster = _find_compatible_cluster(clusters, member, geometry)
            if compatible_cluster is None:
                clusters.append(_WorkingCluster(desc_id=member.desc_id, desc_text=member.desc_text, members=[member]))
            else:
                compatible_cluster.members.append(member)

    promoted_clusters: list[ULConsensusCluster] = []
    rejected_clusters: list[ULConsensusCluster] = []
    quarantined_clusters: list[ULConsensusCluster] = []

    for desc_id, working_clusters in sorted(raw_clusters_by_desc.items()):
        desc_support_ids = frozenset(member.rollout_id for cluster in working_clusters for member in cluster.members)
        full_desc_support_but_split = len(desc_support_ids) == k_valid
        for working_cluster in sorted(working_clusters, key=_working_cluster_sort_key):
            if k_valid == 0:
                rejected_clusters.append(_to_cluster(working_cluster, k_valid, "rejected", "insufficient_support", geometry))
                continue

            support_count = len(working_cluster.members_by_rollout())
            has_full_support = support_count == k_valid
            pairwise_pass, pairwise_records = _pairwise_geometry(working_cluster.members, geometry)
            pairwise_gray_pass = all(bool(record["gray_pass"]) for record in pairwise_records)
            if k_valid < min_ul_valid_rollouts:
                rejected_clusters.append(_to_cluster(working_cluster, k_valid, "rejected", "insufficient_support", geometry))
            elif not has_full_support:
                reason = "geometry_mismatch" if full_desc_support_but_split else "insufficient_support"
                rejected_clusters.append(_to_cluster(working_cluster, k_valid, "rejected", reason, geometry))
            elif not pairwise_pass:
                if pairwise_gray_pass:
                    quarantined_clusters.append(
                        _to_cluster(
                            working_cluster,
                            k_valid,
                            "quarantined",
                            "geometry_gray_zone",
                            geometry,
                        )
                    )
                else:
                    rejected_clusters.append(_to_cluster(working_cluster, k_valid, "rejected", "geometry_mismatch", geometry))
            else:
                consumed_overlap = _consumed_overlap_records(working_cluster.members, consumed_members, geometry)
                if consumed_overlap:
                    quarantined_clusters.append(
                        _to_cluster(
                            working_cluster,
                            k_valid,
                            "quarantined",
                            "consumed_target_overlap",
                            geometry,
                            consumed_overlap=consumed_overlap,
                        )
                    )
                else:
                    promoted_clusters.append(_to_cluster(working_cluster, k_valid, "promoted", "consensus", geometry))

    return ULConsensusResult(
        k_valid=k_valid,
        min_ul_valid_rollouts=min_ul_valid_rollouts,
        consensus_ratio=consensus_ratio,
        geometry=geometry,
        skip_reasons=skip_reasons,
        promoted_clusters=tuple(promoted_clusters),
        rejected_clusters=tuple(rejected_clusters),
        quarantined_clusters=tuple(quarantined_clusters),
        duplicate_like_suppressed_count=duplicate_like_suppressed_count,
    )


def ul_cluster_artifact_rows(
    result: ULConsensusResult,
    *,
    image_id: str,
    sample_id: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cluster in (*result.promoted_clusters, *result.rejected_clusters, *result.quarantined_clusters):
        rows.append(
            {
                "image_id": image_id,
                "sample_id": str(sample_id) if sample_id is not None else image_id,
                "decision": cluster.decision,
                "reason": cluster.reason,
                "desc_id": cluster.desc_id,
                "desc_text": cluster.desc_text,
                "k_valid": result.k_valid,
                "min_ul_valid_rollouts": result.min_ul_valid_rollouts,
                "consensus_ratio": result.consensus_ratio,
                "geometry_thresholds": _geometry_thresholds(result.geometry),
                "support_rollout_ids": list(cluster.support_rollout_ids),
                "support_ratio": cluster.support_ratio,
                "member_boxes": [
                    {
                        "rollout_id": member.rollout_id,
                        "local_index": member.local_index,
                        "bbox_norm1000": list(member.bbox_norm1000),
                    }
                    for _, members in sorted(cluster.members_by_rollout.items())
                    for member in members
                ],
                "pairwise_geometry": [_jsonable_mapping(item) for item in cluster.pairwise_geometry],
                "consumed_overlap": [_jsonable_mapping(item) for item in cluster.consumed_overlap],
            }
        )
    return rows


def _validate_min_ul_valid_rollouts(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("min_ul_valid_rollouts must be a positive integer")
    if value < 1:
        raise ValueError("min_ul_valid_rollouts must be a positive integer")
    return value


def _geometry_thresholds(geometry: ULGeometryConfig) -> dict[str, float]:
    return {
        "iou_min": geometry.iou_min,
        "center_distance_scale_max": geometry.center_distance_scale_max,
        "area_ratio_max": geometry.area_ratio_max,
        "aspect_ratio_max": geometry.aspect_ratio_max,
        "consumed_overlap_iou_min": geometry.consumed_overlap_iou_min,
        "gray_iou_min": float(geometry.gray_iou_min),
        "duplicate_burst_iou_min": float(geometry.duplicate_burst_iou_min),
    }


@dataclass
class _WorkingCluster:
    desc_id: str
    desc_text: str
    members: list[ULMember]

    def members_by_rollout(self) -> dict[str, tuple[ULMember, ...]]:
        by_rollout: dict[str, list[ULMember]] = {}
        for member in self.members:
            by_rollout.setdefault(member.rollout_id, []).append(member)
        return {rollout_id: tuple(sorted(members, key=_member_sort_key)) for rollout_id, members in by_rollout.items()}

    def keep_earliest(self, candidate: ULMember) -> None:
        for index, member in enumerate(self.members):
            if member.rollout_id != candidate.rollout_id:
                continue
            if _member_sort_key(candidate) < _member_sort_key(member):
                self.members[index] = candidate
            return


def _member_sort_key(member: ULMember) -> tuple[str, int, str, str, Box]:
    return (member.rollout_id, member.local_index, member.desc_id, member.desc_text, member.bbox_norm1000)


def _rollout_sort_key(rollout: ULRolloutEvidence) -> tuple[str, tuple[tuple[str, int, str, str, Box], ...]]:
    return (rollout.rollout_id, tuple(_member_sort_key(member) for member in rollout.unmatched_members))


def _working_cluster_sort_key(cluster: _WorkingCluster) -> tuple[str, str, tuple[str, ...]]:
    rollout_ids = tuple(sorted(member.rollout_id for member in cluster.members))
    return (cluster.desc_id, cluster.desc_text, rollout_ids)


def _cluster_sort_key(cluster: ULConsensusCluster) -> tuple[str, str, tuple[str, ...], str, str]:
    return (cluster.desc_id, cluster.desc_text, cluster.support_rollout_ids, cluster.decision, cluster.reason)


def _sort_clusters(clusters: Sequence[ULConsensusCluster]) -> tuple[ULConsensusCluster, ...]:
    return tuple(sorted(tuple(clusters), key=_cluster_sort_key))


def _count_skip_reasons(rollouts: Sequence[ULRolloutEvidence]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rollout in rollouts:
        if rollout.is_valid:
            continue
        reason = rollout.skip_reason or "invalid"
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _find_same_rollout_duplicate_cluster(
    clusters: Sequence[_WorkingCluster],
    member: ULMember,
    geometry: ULGeometryConfig,
) -> _WorkingCluster | None:
    candidates: list[tuple[tuple[float, float, float, float, tuple[str, str, tuple[str, ...]]], _WorkingCluster]] = []
    for cluster in clusters:
        for existing in cluster.members:
            if existing.rollout_id != member.rollout_id:
                continue
            record = _duplicate_geometry_record(existing, member, geometry)
            if record["pass"]:
                candidates.append((_geometry_score((record,), cluster), cluster))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _find_compatible_cluster(
    clusters: Sequence[_WorkingCluster],
    member: ULMember,
    geometry: ULGeometryConfig,
) -> _WorkingCluster | None:
    candidates: list[tuple[tuple[float, float, float, float, tuple[str, str, tuple[str, ...]]], _WorkingCluster]] = []
    for cluster in clusters:
        if any(existing.rollout_id == member.rollout_id for existing in cluster.members):
            continue
        records = tuple(_geometry_record(existing, member, geometry) for existing in cluster.members)
        if records and all(record["gray_pass"] for record in records):
            candidates.append((_geometry_score(records, cluster), cluster))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _geometry_score(
    records: Sequence[Mapping[str, Any]],
    cluster: "_WorkingCluster",
) -> tuple[float, float, float, float, tuple[str, str, tuple[str, ...]]]:
    return (
        max(float(record["center_distance_scale"]) for record in records),
        max(float(record["area_ratio"]) for record in records),
        max(float(record["aspect_ratio_ratio"]) for record in records),
        -min(float(record["iou"]) for record in records),
        _working_cluster_sort_key(cluster),
    )


def _to_cluster(
    working_cluster: _WorkingCluster,
    k_valid: int,
    decision: Decision,
    reason: str,
    geometry: ULGeometryConfig,
    *,
    consumed_overlap: Sequence[Mapping[str, Any]] = (),
) -> ULConsensusCluster:
    members_by_rollout = working_cluster.members_by_rollout()
    support_rollout_ids = tuple(sorted(members_by_rollout))
    support_ratio = float(len(support_rollout_ids) / k_valid) if k_valid else 0.0
    _, pairwise_geometry = _pairwise_geometry(working_cluster.members, geometry)
    return ULConsensusCluster(
        desc_id=working_cluster.desc_id,
        desc_text=working_cluster.desc_text,
        support_rollout_ids=support_rollout_ids,
        support_ratio=support_ratio,
        decision=decision,
        reason=reason,
        members_by_rollout=members_by_rollout,
        pairwise_geometry=pairwise_geometry,
        consumed_overlap=tuple(consumed_overlap),
    )


def _pairwise_geometry(
    members: Sequence[ULMember],
    geometry: ULGeometryConfig,
) -> tuple[bool, tuple[Mapping[str, Any], ...]]:
    records: list[Mapping[str, Any]] = []
    passes = True
    sorted_members = sorted(members, key=_member_sort_key)
    for left_index, left in enumerate(sorted_members):
        for right in sorted_members[left_index + 1 :]:
            record = _geometry_record(left, right, geometry)
            records.append(record)
            passes = passes and bool(record["pass"])
    return passes, tuple(records)


def _geometry_record(left: ULMember, right: ULMember, geometry: ULGeometryConfig) -> Mapping[str, Any]:
    iou = _bbox_iou(left.bbox_norm1000, right.bbox_norm1000)
    center_distance_scale = _center_distance_scale(left.bbox_norm1000, right.bbox_norm1000)
    area_ratio = _area_ratio(left.bbox_norm1000, right.bbox_norm1000)
    aspect_ratio_ratio = _aspect_ratio_ratio(left.bbox_norm1000, right.bbox_norm1000)
    passed = (
        iou >= geometry.iou_min
        and center_distance_scale <= geometry.center_distance_scale_max
        and area_ratio <= geometry.area_ratio_max
        and aspect_ratio_ratio <= geometry.aspect_ratio_max
    )
    gray_passed = (
        iou >= float(geometry.gray_iou_min)
        and center_distance_scale <= geometry.center_distance_scale_max
        and area_ratio <= geometry.area_ratio_max
        and aspect_ratio_ratio <= geometry.aspect_ratio_max
    )
    return MappingProxyType(
        {
            "left_rollout_id": left.rollout_id,
            "left_local_index": left.local_index,
            "right_rollout_id": right.rollout_id,
            "right_local_index": right.local_index,
            "iou": iou,
            "center_distance_scale": center_distance_scale,
            "area_ratio": area_ratio,
            "aspect_ratio_ratio": aspect_ratio_ratio,
            "pass": passed,
            "gray_pass": gray_passed,
        }
    )


def _duplicate_geometry_record(left: ULMember, right: ULMember, geometry: ULGeometryConfig) -> Mapping[str, Any]:
    iou = _bbox_iou(left.bbox_norm1000, right.bbox_norm1000)
    center_distance_scale = _center_distance_scale(left.bbox_norm1000, right.bbox_norm1000)
    area_ratio = _area_ratio(left.bbox_norm1000, right.bbox_norm1000)
    aspect_ratio_ratio = _aspect_ratio_ratio(left.bbox_norm1000, right.bbox_norm1000)
    return MappingProxyType(
        {
            "left_rollout_id": left.rollout_id,
            "left_local_index": left.local_index,
            "right_rollout_id": right.rollout_id,
            "right_local_index": right.local_index,
            "iou": iou,
            "center_distance_scale": center_distance_scale,
            "area_ratio": area_ratio,
            "aspect_ratio_ratio": aspect_ratio_ratio,
            "pass": iou >= float(geometry.duplicate_burst_iou_min),
            "gray_pass": iou >= float(geometry.duplicate_burst_iou_min),
        }
    )


def _consumed_overlap_records(
    members: Sequence[ULMember],
    consumed_members: Sequence[ULMember],
    geometry: ULGeometryConfig,
) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    for member in sorted(members, key=_member_sort_key):
        for consumed in sorted(consumed_members, key=_member_sort_key):
            if member.desc_id != consumed.desc_id:
                continue
            iou = _bbox_iou(member.bbox_norm1000, consumed.bbox_norm1000)
            if iou >= geometry.consumed_overlap_iou_min:
                records.append(
                    MappingProxyType(
                        {
                            "rollout_id": member.rollout_id,
                            "local_index": member.local_index,
                            "consumed_rollout_id": consumed.rollout_id,
                            "consumed_local_index": consumed.local_index,
                            "iou": iou,
                        }
                    )
                )
    return tuple(records)


def _bbox_iou(left: Box, right: Box) -> float:
    left_area = _area(left)
    right_area = _area(right)
    if left_area <= 0.0 or right_area <= 0.0:
        return 0.0

    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _center_distance_scale(left: Box, right: Box) -> float:
    """Center distance normalized by the pair median box diagonal."""
    left_cx = (left[0] + left[2]) / 2.0
    left_cy = (left[1] + left[3]) / 2.0
    right_cx = (right[0] + right[2]) / 2.0
    right_cy = (right[1] + right[3]) / 2.0
    distance = math.hypot(left_cx - right_cx, left_cy - right_cy)
    scale = max((_box_diagonal(left) + _box_diagonal(right)) / 2.0, 1.0)
    return distance / scale


def _box_diagonal(box: Box) -> float:
    return math.hypot(max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1]))


def _area_ratio(left: Box, right: Box) -> float:
    left_area = _area(left)
    right_area = _area(right)
    if left_area <= 0.0 or right_area <= 0.0:
        return math.inf
    return max(left_area, right_area) / min(left_area, right_area)


def _aspect_ratio_ratio(left: Box, right: Box) -> float:
    left_aspect = _aspect_ratio(left)
    right_aspect = _aspect_ratio(right)
    if left_aspect <= 0.0 or right_aspect <= 0.0:
        return math.inf
    return max(left_aspect, right_aspect) / min(left_aspect, right_aspect)


def _aspect_ratio(box: Box) -> float:
    height = max(0.0, box[3] - box[1])
    width = max(0.0, box[2] - box[0])
    if width <= 0.0 or height <= 0.0:
        return 0.0
    return width / height


def _jsonable_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable_value(value) for key, value in sorted(mapping.items(), key=lambda item: str(item[0]))}


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _jsonable_mapping(value)
    if isinstance(value, tuple):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    return value


__all__ = [
    "ULGeometryConfig",
    "ULMember",
    "ULRolloutEvidence",
    "ULConsensusCluster",
    "ULConsensusResult",
    "mine_ul_consensus",
    "ul_cluster_artifact_rows",
]
