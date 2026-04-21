from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.common.geometry.coord_utils import coerce_point_list, ints_to_pixels_norm1000
from src.common.geometry.object_geometry import extract_single_geometry
from src.common.prediction_parsing import load_prediction_dict

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str


@dataclass(frozen=True)
class InvalidGeometryFamilySpec:
    family_alias: str
    gt_vs_pred_jsonl: str
    raw_coord_mode: str = "pixel"


@dataclass(frozen=True)
class InvalidGeometryAuditConfig:
    run: RunConfig
    families: tuple[InvalidGeometryFamilySpec, ...]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


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


def _artifact_root_for_repo(repo_root: Path) -> Path:
    repo_root = repo_root.resolve()
    parts = list(repo_root.parts)
    if ".worktrees" in parts:
        marker = parts.index(".worktrees")
        return Path(*parts[:marker])
    return repo_root


def _resolve_input_path(path_str: str, *, config_dir: Path, artifact_root: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    config_relative = config_dir / path
    if config_relative.exists():
        return config_relative
    return artifact_root / path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{index + 1} must contain a JSON object per line.")
        rows.append(payload)
    return rows


def load_invalid_geometry_audit_config(config_path: Path) -> InvalidGeometryAuditConfig:
    payload = _load_yaml(config_path)
    run_raw = _require_mapping(payload, "run")
    families_raw = payload.get("families")
    if not isinstance(families_raw, list) or not families_raw:
        raise ValueError("families must be a non-empty list.")
    families: list[InvalidGeometryFamilySpec] = []
    for index, item in enumerate(families_raw):
        if not isinstance(item, dict):
            raise ValueError(f"families[{index}] must be a mapping.")
        families.append(
            InvalidGeometryFamilySpec(
                family_alias=_require_nonempty_str(item, "family_alias"),
                gt_vs_pred_jsonl=_require_nonempty_str(item, "gt_vs_pred_jsonl"),
                raw_coord_mode=str(item.get("raw_coord_mode") or "pixel").strip().lower(),
            )
        )
    return InvalidGeometryAuditConfig(
        run=RunConfig(
            name=_require_nonempty_str(run_raw, "name"),
            output_dir=_require_nonempty_str(run_raw, "output_dir"),
        ),
        families=tuple(families),
    )


def _normalize_raw_output(raw_output_json: Any) -> dict[str, Any] | None:
    if isinstance(raw_output_json, dict):
        return dict(raw_output_json)
    if isinstance(raw_output_json, str):
        return load_prediction_dict(raw_output_json)
    return None


def _to_raw_pixel_points(
    points_raw: list[Any],
    *,
    width: int,
    height: int,
    raw_coord_mode: str,
) -> list[float] | None:
    points, had_tokens = coerce_point_list(points_raw)
    if points is None:
        return None
    if raw_coord_mode == "norm1000" or had_tokens:
        ints = [int(round(float(v))) for v in points]
        return ints_to_pixels_norm1000(ints, width, height)
    return [float(v) for v in points]


def _classify_bbox_failure(
    points_px: list[float],
    *,
    width: int,
    height: int,
) -> tuple[list[str], bool]:
    if len(points_px) != 4:
        return (["bbox_points"], True)
    x1, y1, x2, y2 = [float(v) for v in points_px]
    reasons: list[str] = []
    if x1 < 0.0 or x2 < 0.0:
        reasons.append("negative_x")
    if y1 < 0.0 or y2 < 0.0:
        reasons.append("negative_y")
    if x1 > float(width - 1) or x2 > float(width - 1):
        reasons.append("overflow_right")
    if y1 > float(height - 1) or y2 > float(height - 1):
        reasons.append("overflow_bottom")
    if x2 <= x1:
        reasons.append("non_positive_width")
    if y2 <= y1:
        reasons.append("non_positive_height")
    near_ceiling = max(x1, y1, x2, y2) >= 990.0 and (
        "overflow_right" in reasons or "overflow_bottom" in reasons
    )
    return (sorted(set(reasons)), bool(near_ceiling))


def _bucket_counts(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(field) or "")
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        out.append({field: key, "count": int(len(group))})
    return out


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Coord Family Invalid Geometry Audit",
        "",
        f"- Run: `{summary['run_name']}`",
        "",
    ]
    for family_alias, payload in sorted(summary.get("families", {}).items()):
        lines.extend(
            [
                f"## {family_alias}",
                "",
                f"- images_with_invalid_geometry: `{payload.get('images_with_invalid_geometry')}`",
                f"- invalid_geometry_error_entries: `{payload.get('invalid_geometry_error_entries')}`",
                f"- invalid_raw_object_count: `{payload.get('invalid_raw_object_count')}`",
                f"- alignment_rate: `{payload.get('alignment_rate'):.4f}`",
                "",
                "### Failure Families",
                "",
            ]
        )
        for row in payload.get("failure_family_counts", [])[:10]:
            lines.append(f"- `{row['failure_family']}`: `{row['count']}`")
        lines.extend(["", "### Invalid Descs", ""])
        for row in payload.get("invalid_desc_counts", [])[:10]:
            lines.append(f"- `{row['desc']}`: `{row['count']}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_invalid_geometry_audit(
    config_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    config = load_invalid_geometry_audit_config(config_path)
    artifact_root = _artifact_root_for_repo(repo_root)
    output_root = artifact_root / config.run.output_dir
    run_dir = output_root / config.run.name
    run_dir.mkdir(parents=True, exist_ok=True)

    family_payloads: dict[str, Any] = {}
    all_invalid_rows: list[dict[str, Any]] = []
    for spec in config.families:
        gt_vs_pred_path = _resolve_input_path(
            spec.gt_vs_pred_jsonl,
            config_dir=config_path.parent,
            artifact_root=artifact_root,
        )
        rows = _read_jsonl(gt_vs_pred_path)
        invalid_rows: list[dict[str, Any]] = []
        images_with_invalid_geometry = 0
        invalid_geometry_error_entries = 0
        aligned_rows = 0
        compared_rows = 0
        for row in rows:
            width = int(row.get("width") or 0)
            height = int(row.get("height") or 0)
            if width <= 0 or height <= 0:
                continue
            error_entries = row.get("error_entries") or []
            error_count = sum(
                1
                for entry in error_entries
                if isinstance(entry, Mapping) and str(entry.get("code") or "") == "invalid_geometry"
            )
            if error_count <= 0:
                continue
            images_with_invalid_geometry += 1
            invalid_geometry_error_entries += int(error_count)
            raw_payload = _normalize_raw_output(row.get("raw_output_json"))
            objects = raw_payload.get("objects") if isinstance(raw_payload, Mapping) else None
            invalid_objects: list[dict[str, Any]] = []
            if isinstance(objects, list):
                for obj_idx, obj in enumerate(objects):
                    if not isinstance(obj, Mapping):
                        continue
                    try:
                        geom_type, points_raw = extract_single_geometry(
                            obj,
                            allow_type_and_points=True,
                            allow_nested_points=False,
                            path="raw_output_json.objects[*]",
                        )
                    except ValueError:
                        continue
                    if str(geom_type) != "bbox_2d":
                        continue
                    points_px = _to_raw_pixel_points(
                        points_raw,
                        width=width,
                        height=height,
                        raw_coord_mode=spec.raw_coord_mode,
                    )
                    if points_px is None:
                        continue
                    reasons, near_ceiling = _classify_bbox_failure(points_px, width=width, height=height)
                    if not reasons:
                        continue
                    invalid_objects.append(
                        {
                            "raw_object_index": int(obj_idx),
                            "desc": str(obj.get("desc") or "").strip(),
                            "raw_points": [float(v) for v in points_px],
                            "failure_family": "+".join(reasons),
                            "near_norm1000_ceiling": bool(near_ceiling),
                        }
                    )
            compared_rows += 1
            if len(invalid_objects) == int(error_count):
                aligned_rows += 1
            for invalid_obj in invalid_objects:
                enriched = {
                    "family_alias": spec.family_alias,
                    "image": row.get("image"),
                    "image_id": row.get("image_id"),
                    "width": width,
                    "height": height,
                    "error_count": int(error_count),
                    **invalid_obj,
                }
                invalid_rows.append(enriched)
                all_invalid_rows.append(enriched)

        family_payloads[spec.family_alias] = {
            "images_with_invalid_geometry": int(images_with_invalid_geometry),
            "invalid_geometry_error_entries": int(invalid_geometry_error_entries),
            "invalid_raw_object_count": int(len(invalid_rows)),
            "alignment_rate": float(aligned_rows / compared_rows) if compared_rows else 0.0,
            "failure_family_counts": _bucket_counts(invalid_rows, "failure_family"),
            "invalid_desc_counts": _bucket_counts(invalid_rows, "desc"),
            "near_norm1000_ceiling_count": int(
                sum(1 for row in invalid_rows if bool(row.get("near_norm1000_ceiling")))
            ),
        }

    summary = {
        "run_name": config.run.name,
        "families": family_payloads,
        "artifacts": {
            "invalid_rows_jsonl": str(run_dir / "invalid_rows.jsonl"),
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_jsonl(run_dir / "invalid_rows.jsonl", all_invalid_rows)
    (run_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    return {
        "run_dir": str(run_dir),
        "summary_json": str(run_dir / "summary.json"),
        "invalid_rows_jsonl": str(run_dir / "invalid_rows.jsonl"),
        "family_count": len(family_payloads),
        "invalid_row_count": len(all_invalid_rows),
    }
