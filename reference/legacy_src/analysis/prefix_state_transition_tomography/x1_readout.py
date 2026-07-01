from __future__ import annotations

from typing import Any, Mapping, Sequence


BOUNDARY_X1_VALUES = {0, 999}


def partition_x1_peaks(
    peaks: Sequence[Mapping[str, Any]],
    emitted_same_desc: Sequence[Mapping[str, Any]],
    residual_same_desc: Sequence[Mapping[str, Any]],
    *,
    radius: int = 24,
) -> list[dict[str, Any]]:
    emitted = [_object_x1(obj) for obj in emitted_same_desc]
    residual = [_object_x1(obj) for obj in residual_same_desc]
    assigned: list[dict[str, Any]] = []
    for peak in peaks:
        x1 = _peak_x1(peak)
        emitted_matches = [obj for obj in emitted if abs(x1 - int(obj["x1"])) <= radius]
        residual_matches = [obj for obj in residual if abs(x1 - int(obj["x1"])) <= radius]
        if residual_matches and emitted_matches:
            partition = "ambiguous_x1_collision"
        elif residual_matches:
            partition = "residual_same_desc_x1_peak"
        elif emitted_matches:
            partition = "emitted_same_desc_x1_peak"
        elif x1 in BOUNDARY_X1_VALUES:
            partition = "boundary_artifact_x1_peak"
        else:
            partition = "unmatched_x1_peak"
        assigned.append(
            {
                **dict(peak),
                "x1_bin": x1,
                "partition": partition,
                "matched_residual_gt_indices": [obj["gt_idx"] for obj in residual_matches],
                "matched_emitted_gt_indices": [obj["gt_idx"] for obj in emitted_matches],
            }
        )
    return assigned


def summarize_x1_partitions(
    partitioned_peaks: Sequence[Mapping[str, Any]],
    residual_same_desc: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    residual_gt_indices = {int(obj.get("gt_idx", idx)) for idx, obj in enumerate(residual_same_desc)}
    covered_residual = {
        int(gt_idx)
        for peak in partitioned_peaks
        if peak.get("partition") == "residual_same_desc_x1_peak"
        for gt_idx in peak.get("matched_residual_gt_indices", [])
    }
    peak_count = len(partitioned_peaks)
    counts = {
        "residual_same_desc_x1_peak_count": 0,
        "emitted_same_desc_x1_peak_count": 0,
        "ambiguous_x1_collision_count": 0,
        "unmatched_x1_peak_count": 0,
        "boundary_artifact_x1_peak_count": 0,
    }
    for peak in partitioned_peaks:
        key = f"{peak.get('partition')}_count"
        if key in counts:
            counts[key] += 1
    return {
        **counts,
        "partitioned_peak_count": peak_count,
        "residual_gt_count": len(residual_gt_indices),
        "residual_gt_covered_count": len(covered_residual),
        "forced_x1_residual_coverage": (
            0.0 if not residual_gt_indices else len(covered_residual) / len(residual_gt_indices)
        ),
        "emitted_attraction_rate": (
            0.0 if not peak_count else counts["emitted_same_desc_x1_peak_count"] / peak_count
        ),
        "unmatched_x1_peak_rate": 0.0 if not peak_count else counts["unmatched_x1_peak_count"] / peak_count,
        "boundary_artifact_x1_peak_rate": (
            0.0 if not peak_count else counts["boundary_artifact_x1_peak_count"] / peak_count
        ),
    }


def probe_forced_desc_pre_x1(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise NotImplementedError("GPU forced-desc pre-x1 runtime is implemented in paired_probe")


def _peak_x1(peak: Mapping[str, Any]) -> int:
    for key in ("x1_bin", "center", "x1", "bin"):
        if key in peak:
            return int(peak[key])
    raise ValueError("peak is missing x1_bin")


def _object_x1(obj: Mapping[str, Any]) -> dict[str, int]:
    bbox = obj.get("bbox_xyxy")
    if bbox is not None:
        x1 = int(bbox[0])
    else:
        x1 = int(obj["x1"])
    return {"gt_idx": int(obj.get("gt_idx", -1)), "x1": x1}


__all__ = [
    "BOUNDARY_X1_VALUES",
    "partition_x1_peaks",
    "probe_forced_desc_pre_x1",
    "summarize_x1_partitions",
]

