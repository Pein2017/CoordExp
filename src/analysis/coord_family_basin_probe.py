from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.analysis.coord_family_probe_registry import (
    canonical_xyxy_norm1000,
    get_family_probe_spec,
    native_slot_names,
)
from src.analysis.raw_text_coord_continuity_report import compute_basin_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str


@dataclass(frozen=True)
class BasinProbeRow:
    family_alias: str
    probe_id: str
    slot: str
    center_value: int
    target_value: int
    candidate_value: int
    score_mean: float
    abs_distance_to_target: int
    native_target_values: tuple[int, int, int, int] | None = None
    native_center_values: tuple[int, int, int, int] | None = None
    image_width: int | None = None
    image_height: int | None = None


@dataclass(frozen=True)
class BasinProbeConfig:
    run: RunConfig
    probe_rows: tuple[BasinProbeRow, ...]


def _artifact_root_for_repo(repo_root: Path) -> Path:
    repo_root = repo_root.resolve()
    parts = list(repo_root.parts)
    if ".worktrees" in parts:
        marker = parts.index(".worktrees")
        return Path(*parts[:marker])
    return repo_root


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return payload


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _require_nonempty_str(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    return value.strip()


def _require_int(parent: dict[str, Any], key: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer.")
    return int(value)


def _require_float(parent: dict[str, Any], key: str) -> float:
    value = parent.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric.")
    return float(value)


def _optional_nonempty_str(parent: dict[str, Any], key: str) -> str | None:
    value = parent.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string when provided.")
    return value.strip()


def _optional_int_quad(parent: dict[str, Any], key: str) -> tuple[int, int, int, int] | None:
    value = parent.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4 or any(not isinstance(v, int) for v in value):
        raise ValueError(f"{key} must be a list of four integers when provided.")
    return tuple(int(v) for v in value)


def _optional_positive_int(parent: dict[str, Any], key: str) -> int | None:
    value = parent.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer when provided.")
    return int(value)


def load_basin_probe_config(config_path: Path) -> BasinProbeConfig:
    payload = _load_yaml(config_path)
    run_raw = _require_mapping(payload, "run")
    run = RunConfig(
        name=_require_nonempty_str(run_raw, "name"),
        output_dir=_require_nonempty_str(run_raw, "output_dir"),
    )
    rows_raw = payload.get("probe_rows")
    if not isinstance(rows_raw, list) or not rows_raw:
        raise ValueError("probe_rows must be a non-empty list.")
    probe_rows: list[BasinProbeRow] = []
    for index, item in enumerate(rows_raw):
        if not isinstance(item, dict):
            raise ValueError(f"probe_rows[{index}] must be a mapping.")
        probe_rows.append(
            BasinProbeRow(
                family_alias=_require_nonempty_str(item, "family_alias"),
                probe_id=_optional_nonempty_str(item, "probe_id")
                or (
                    f"{_require_nonempty_str(item, 'family_alias')}"
                    f":{_require_nonempty_str(item, 'slot')}"
                    f":{_require_int(item, 'center_value')}"
                    f":{_require_int(item, 'target_value')}"
                ),
                slot=_require_nonempty_str(item, "slot"),
                center_value=_require_int(item, "center_value"),
                target_value=_require_int(item, "target_value"),
                candidate_value=_require_int(item, "candidate_value"),
                score_mean=_require_float(item, "score_mean"),
                abs_distance_to_target=_require_int(item, "abs_distance_to_target"),
                native_target_values=_optional_int_quad(item, "native_target_values"),
                native_center_values=_optional_int_quad(item, "native_center_values"),
                image_width=_optional_positive_int(item, "image_width"),
                image_height=_optional_positive_int(item, "image_height"),
            )
        )
    return BasinProbeConfig(run=run, probe_rows=tuple(probe_rows))


def summarize_basin_rows(rows: list[BasinProbeRow]) -> list[dict[str, Any]]:
    if not rows:
        return []
    grouped: dict[tuple[str, str, str, int, int], list[BasinProbeRow]] = defaultdict(list)
    for row in rows:
        spec = get_family_probe_spec(row.family_alias)
        if row.slot not in spec.native_slots:
            raise ValueError(
                f"slot {row.slot!r} is not valid for family {row.family_alias!r}: {spec.native_slots}"
            )
        grouped[
            (row.family_alias, row.probe_id, row.slot, row.center_value, row.target_value)
        ].append(row)

    summaries: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group_rows = grouped[key]
        head = group_rows[0]
        spec = get_family_probe_spec(head.family_alias)
        metric_rows = [
            {
                "candidate_value": row.candidate_value,
                "score": row.score_mean,
                "center_value": row.center_value,
                "target_value": row.target_value,
            }
            for row in group_rows
        ]
        target_metrics = compute_basin_metrics(metric_rows, center_key="target_value")
        center_metrics = compute_basin_metrics(metric_rows, center_key="center_value")
        summaries.append(
            {
                "family_alias": head.family_alias,
                "probe_id": head.probe_id,
                "slot": head.slot,
                "center_value": head.center_value,
                "target_value": head.target_value,
                "candidate_count": len(group_rows),
                "native_slots": list(spec.native_slots),
                "bbox_format": spec.bbox_format,
                "pred_coord_mode": spec.pred_coord_mode,
                "eval_compatibility_path": spec.eval_compatibility_path,
                "canonical_projection_kind": spec.canonical_projection_kind,
                "canonical_compare_group": f"canonical_xyxy_{spec.pred_coord_mode}",
                **target_metrics,
                "center_mass_at_4": center_metrics["mass_at_4"],
            }
        )
    return summaries


def _softmax(scores: list[float]) -> list[float]:
    peak = max(scores)
    exps = [math.exp(score - peak) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


def _compute_distance_metrics(
    rows: list[dict[str, float]],
    *,
    distance_key: str,
) -> dict[str, float]:
    if not rows:
        raise ValueError("_compute_distance_metrics requires at least one row")
    scores = [float(row["score"]) for row in rows]
    distances = [float(row[distance_key]) for row in rows]
    weights = _softmax(scores)

    def neighborhood_mass(radius: float) -> float:
        return float(
            sum(weight for distance, weight in zip(distances, weights) if distance <= radius)
        )

    local_expected_abs_error = float(
        sum(distance * weight for distance, weight in zip(distances, weights))
    )
    peak = max(scores)
    floor = min(scores)
    half_height = peak - (peak - floor) / 2.0
    half_height_width = float(
        max(distance for distance, score in zip(distances, scores) if score >= half_height)
    )
    return {
        "mass_at_1": neighborhood_mass(1.0),
        "mass_at_2": neighborhood_mass(2.0),
        "mass_at_4": neighborhood_mass(4.0),
        "mass_at_8": neighborhood_mass(8.0),
        "mass_at_16": neighborhood_mass(16.0),
        "local_expected_abs_error": local_expected_abs_error,
        "half_height_width": half_height_width,
    }


def _replace_slot_value(
    values: tuple[int, int, int, int],
    *,
    slot_index: int,
    candidate_value: int,
) -> tuple[int, int, int, int]:
    updated = list(values)
    updated[slot_index] = candidate_value
    return tuple(updated)


def _bbox_linf_distance(
    candidate_bbox: list[float],
    reference_bbox: list[float],
) -> float:
    return float(max(abs(candidate - reference) for candidate, reference in zip(candidate_bbox, reference_bbox)))


def _canonical_skip(
    *,
    family_alias: str,
    probe_id: str,
    reason: str,
) -> dict[str, str]:
    return {
        "family_alias": family_alias,
        "probe_id": probe_id,
        "reason": reason,
    }


def _mean_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(metric_dicts[0])
    return {
        key: sum(metric[key] for metric in metric_dicts) / len(metric_dicts)
        for key in keys
    }


def summarize_canonical_basin_rows(rows: list[BasinProbeRow]) -> dict[str, list[dict[str, Any]]]:
    if not rows:
        return {
            "probe_metrics": [],
            "family_rollup": [],
            "skipped": [],
        }

    grouped: dict[tuple[str, str], list[BasinProbeRow]] = defaultdict(list)
    for row in rows:
        valid_slots = native_slot_names(row.family_alias)
        if row.slot not in valid_slots:
            raise ValueError(
                f"slot {row.slot!r} is not valid for family {row.family_alias!r}: {valid_slots}"
            )
        grouped[(row.family_alias, row.probe_id)].append(row)

    probe_metrics: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for family_alias, probe_id in sorted(grouped):
        group_rows = grouped[(family_alias, probe_id)]
        spec = get_family_probe_spec(family_alias)
        if not spec.requires_canonical_projection:
            skipped.append(
                _canonical_skip(
                    family_alias=family_alias,
                    probe_id=probe_id,
                    reason="canonical_projection_not_required",
                )
            )
            continue
        if any(
            row.native_target_values is None or row.native_center_values is None
            for row in group_rows
        ):
            skipped.append(
                _canonical_skip(
                    family_alias=family_alias,
                    probe_id=probe_id,
                    reason="missing_native_bbox_context",
                )
            )
            continue

        if spec.pred_coord_mode == "pixel" and any(
            row.image_width is None or row.image_height is None for row in group_rows
        ):
            skipped.append(
                _canonical_skip(
                    family_alias=family_alias,
                    probe_id=probe_id,
                    reason="missing_image_size_for_pixel_projection",
                )
            )
            continue

        target_contexts = {row.native_target_values for row in group_rows}
        center_contexts = {row.native_center_values for row in group_rows}
        image_sizes = {(row.image_width, row.image_height) for row in group_rows}
        if len(target_contexts) != 1 or len(center_contexts) != 1:
            skipped.append(
                _canonical_skip(
                    family_alias=family_alias,
                    probe_id=probe_id,
                    reason="inconsistent_native_bbox_context",
                )
            )
            continue
        if spec.pred_coord_mode == "pixel" and len(image_sizes) != 1:
            skipped.append(
                _canonical_skip(
                    family_alias=family_alias,
                    probe_id=probe_id,
                    reason="inconsistent_image_size_context",
                )
            )
            continue

        target_values = next(iter(target_contexts))
        center_values = next(iter(center_contexts))
        image_width, image_height = next(iter(image_sizes))
        try:
            target_bbox_xyxy = canonical_xyxy_norm1000(
                family_alias,
                target_values,
                image_width=image_width,
                image_height=image_height,
            )
            center_bbox_xyxy = canonical_xyxy_norm1000(
                family_alias,
                center_values,
                image_width=image_width,
                image_height=image_height,
            )
        except NotImplementedError:
            skipped.append(
                _canonical_skip(
                    family_alias=family_alias,
                    probe_id=probe_id,
                    reason=f"unsupported_projection:{spec.canonical_projection_kind}",
                )
            )
            continue

        target_distance_rows: list[dict[str, float]] = []
        center_distance_rows: list[dict[str, float]] = []
        for row in group_rows:
            slot_index = spec.native_slots.index(row.slot)
            candidate_target_values = _replace_slot_value(
                target_values,
                slot_index=slot_index,
                candidate_value=row.candidate_value,
            )
            candidate_center_values = _replace_slot_value(
                center_values,
                slot_index=slot_index,
                candidate_value=row.candidate_value,
            )
            candidate_target_bbox = canonical_xyxy_norm1000(
                family_alias,
                candidate_target_values,
                image_width=image_width,
                image_height=image_height,
            )
            candidate_center_bbox = canonical_xyxy_norm1000(
                family_alias,
                candidate_center_values,
                image_width=image_width,
                image_height=image_height,
            )
            target_distance_rows.append(
                {
                    "score": row.score_mean,
                    "distance": _bbox_linf_distance(candidate_target_bbox, target_bbox_xyxy),
                }
            )
            center_distance_rows.append(
                {
                    "score": row.score_mean,
                    "distance": _bbox_linf_distance(candidate_center_bbox, center_bbox_xyxy),
                }
            )

        probe_metrics.append(
            {
                "family_alias": family_alias,
                "probe_id": probe_id,
                "candidate_count": len(group_rows),
                "slot_count": len({row.slot for row in group_rows}),
                "native_slots": list(spec.native_slots),
                "bbox_format": spec.bbox_format,
                "pred_coord_mode": spec.pred_coord_mode,
                "canonical_projection_kind": spec.canonical_projection_kind,
                "canonical_compare_group": "canonical_xyxy_norm1000",
                "target_bbox_xyxy": target_bbox_xyxy,
                "center_bbox_xyxy": center_bbox_xyxy,
                "target_bbox_metrics": _compute_distance_metrics(
                    target_distance_rows,
                    distance_key="distance",
                ),
                "center_bbox_metrics": _compute_distance_metrics(
                    center_distance_rows,
                    distance_key="distance",
                ),
            }
        )

    family_rollup: list[dict[str, Any]] = []
    family_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metric in probe_metrics:
        family_grouped[str(metric["family_alias"])].append(metric)
    for family_alias in sorted(family_grouped):
        metrics = family_grouped[family_alias]
        family_rollup.append(
            {
                "family_alias": family_alias,
                "probe_count": len(metrics),
                "pred_coord_mode": metrics[0]["pred_coord_mode"],
                "canonical_compare_group": "canonical_xyxy_norm1000",
                "mean_target_bbox_metrics": _mean_metric_dicts(
                    [metric["target_bbox_metrics"] for metric in metrics]
                ),
                "mean_center_bbox_metrics": _mean_metric_dicts(
                    [metric["center_bbox_metrics"] for metric in metrics]
                ),
            }
        )

    return {
        "probe_metrics": probe_metrics,
        "family_rollup": family_rollup,
        "skipped": skipped,
    }


def _run_dir(run: RunConfig, repo_root: Path) -> Path:
    output_dir = Path(run.output_dir)
    if not output_dir.is_absolute():
        output_dir = _artifact_root_for_repo(repo_root) / output_dir
    return output_dir / run.name


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_basin_probe(
    config_path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = (repo_root or REPO_ROOT).resolve()
    config = load_basin_probe_config(config_path)
    run_dir = _run_dir(config.run, repo_root)
    family_native_slot_metrics = summarize_basin_rows(list(config.probe_rows))
    canonical_comparison_view = summarize_canonical_basin_rows(list(config.probe_rows))
    rows_payload = [
        {
            "family_alias": row.family_alias,
            "probe_id": row.probe_id,
            "slot": row.slot,
            "center_value": row.center_value,
            "target_value": row.target_value,
            "candidate_value": row.candidate_value,
            "score_mean": row.score_mean,
            "abs_distance_to_target": row.abs_distance_to_target,
            "native_target_values": list(row.native_target_values) if row.native_target_values else None,
            "native_center_values": list(row.native_center_values) if row.native_center_values else None,
            "image_width": row.image_width,
            "image_height": row.image_height,
        }
        for row in config.probe_rows
    ]

    summary_path = run_dir / "summary.json"
    rows_path = run_dir / "basin_rows.jsonl"
    _write_json(
        summary_path,
        {
            "run_name": config.run.name,
            "family_native_slot_metrics": family_native_slot_metrics,
            "slot_metrics": family_native_slot_metrics,
            "canonical_comparison_view": canonical_comparison_view,
        },
    )
    _write_jsonl(rows_path, rows_payload)
    return {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "summary_json": str(summary_path),
        "basin_rows_jsonl": str(rows_path),
        "slot_metric_count": len(family_native_slot_metrics),
        "family_native_metric_count": len(family_native_slot_metrics),
        "canonical_metric_count": len(canonical_comparison_view["probe_metrics"]),
        "canonical_skip_count": len(canonical_comparison_view["skipped"]),
    }


__all__ = [
    "BasinProbeConfig",
    "BasinProbeRow",
    "RunConfig",
    "load_basin_probe_config",
    "run_basin_probe",
    "summarize_canonical_basin_rows",
    "summarize_basin_rows",
]
