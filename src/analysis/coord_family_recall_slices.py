from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.common.geometry.coord_utils import bbox_from_points, coerce_point_list, denorm_and_clamp

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str


@dataclass(frozen=True)
class FamilySliceSpec:
    family_alias: str
    oracle_fn_objects_jsonl: str
    subset_jsonl: str


@dataclass(frozen=True)
class RecallSliceConfig:
    run: RunConfig
    families: tuple[FamilySliceSpec, ...]


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


def load_recall_slice_config(config_path: Path) -> RecallSliceConfig:
    payload = _load_yaml(config_path)
    run_raw = _require_mapping(payload, "run")
    families_raw = payload.get("families")
    if not isinstance(families_raw, list) or not families_raw:
        raise ValueError("families must be a non-empty list.")
    families: list[FamilySliceSpec] = []
    for index, item in enumerate(families_raw):
        if not isinstance(item, dict):
            raise ValueError(f"families[{index}] must be a mapping.")
        families.append(
            FamilySliceSpec(
                family_alias=_require_nonempty_str(item, "family_alias"),
                oracle_fn_objects_jsonl=_require_nonempty_str(item, "oracle_fn_objects_jsonl"),
                subset_jsonl=_require_nonempty_str(item, "subset_jsonl"),
            )
        )
    return RecallSliceConfig(
        run=RunConfig(
            name=_require_nonempty_str(run_raw, "name"),
            output_dir=_require_nonempty_str(run_raw, "output_dir"),
        ),
        families=tuple(families),
    )


def _parse_bbox_points_px(
    points_raw: list[Any],
    *,
    width: int,
    height: int,
) -> list[int]:
    points, had_tokens = coerce_point_list(points_raw)
    if points is None:
        raise ValueError("failed to parse bbox points")
    coord_mode = "norm1000" if had_tokens else "pixel"
    return denorm_and_clamp(points, width, height, coord_mode=coord_mode)


def _size_bucket(area_ratio: float) -> str:
    if area_ratio < 0.001:
        return "tiny"
    if area_ratio < 0.01:
        return "small"
    if area_ratio < 0.05:
        return "medium"
    return "large"


def _crowd_bucket(image_gt_count: int) -> str:
    if image_gt_count <= 3:
        return "isolated"
    if image_gt_count <= 7:
        return "moderate"
    return "crowded"


def _same_desc_bucket(same_desc_count: int) -> str:
    if same_desc_count <= 1:
        return "singleton"
    if same_desc_count == 2:
        return "pair"
    return "triple_plus"


def _row_from_subset_record(
    subset_row: dict[str, Any],
    *,
    record_idx: int,
    gt_idx: int,
) -> dict[str, Any]:
    objects = subset_row.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"subset row {record_idx} missing objects list")
    if gt_idx < 0 or gt_idx >= len(objects):
        raise ValueError(f"subset row {record_idx} missing gt_idx={gt_idx}")
    obj = objects[int(gt_idx)]
    if not isinstance(obj, dict):
        raise ValueError(f"subset row {record_idx} object {gt_idx} must be a mapping")
    width = int(subset_row.get("width") or 0)
    height = int(subset_row.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"subset row {record_idx} missing width/height")
    bbox_points = _parse_bbox_points_px(list(obj.get("bbox_2d") or []), width=width, height=height)
    x1, y1, x2, y2 = bbox_from_points(bbox_points)
    area_ratio = max(0.0, float((x2 - x1) * (y2 - y1))) / float(width * height)
    desc = str(obj.get("desc") or obj.get("category_name") or "").strip()
    same_desc_count = sum(
        1
        for item in objects
        if isinstance(item, dict) and str(item.get("desc") or item.get("category_name") or "").strip() == desc
    )
    return {
        "record_idx": int(record_idx),
        "gt_idx": int(gt_idx),
        "image_id": subset_row.get("image_id"),
        "file_name": subset_row.get("file_name"),
        "width": width,
        "height": height,
        "gt_desc": desc,
        "category_name": str(obj.get("category_name") or desc),
        "category_id": obj.get("category_id"),
        "bbox_px": [int(round(v)) for v in [x1, y1, x2, y2]],
        "area_ratio": float(area_ratio),
        "size_bucket": _size_bucket(area_ratio),
        "image_gt_count": int(len(objects)),
        "crowd_bucket": _crowd_bucket(len(objects)),
        "same_desc_count": int(same_desc_count),
        "same_desc_bucket": _same_desc_bucket(int(same_desc_count)),
        "repeated_desc": bool(int(same_desc_count) >= 2),
    }


def _aggregate_bucket(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(field) or "")
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        total = len(group)
        recoverable = sum(1 for row in group if bool(row.get("ever_recovered_loc")))
        systematic = sum(1 for row in group if bool(row.get("systematic_loc")))
        out.append(
            {
                field: key,
                "fn_count": int(total),
                "recoverable_fn_count": int(recoverable),
                "systematic_fn_count": int(systematic),
                "recoverable_fraction": float(recoverable / total) if total else 0.0,
                "mean_recover_fraction_loc": float(
                    sum(float(row.get("recover_fraction_loc") or 0.0) for row in group) / total
                )
                if total
                else 0.0,
            }
        )
    return out


def _render_report(summary: dict[str, Any]) -> str:
    families = summary.get("families", {})
    lines = [
        "# Coord Family Recall Slices",
        "",
        f"- Run: `{summary['run_name']}`",
        f"- Family count: `{len(families)}`",
        "",
    ]
    for family_alias, payload in sorted(families.items()):
        if not isinstance(payload, dict):
            continue
        overall = payload.get("overall", {})
        lines.extend(
            [
                f"## {family_alias}",
                "",
                f"- baseline_fn_count: `{overall.get('baseline_fn_count_loc')}`",
                f"- recoverable_fn_count: `{overall.get('recoverable_fn_count_loc')}`",
                f"- systematic_fn_count: `{overall.get('systematic_fn_count_loc')}`",
                f"- recoverable_fraction_of_baseline_fn: `{overall.get('recoverable_fraction_of_baseline_fn_loc'):.4f}`",
                "",
                "### Size Buckets",
                "",
            ]
        )
        for row in payload.get("by_size_bucket", [])[:6]:
            lines.append(
                f"- `{row['size_bucket']}`: fn `{row['fn_count']}`, recoverable `{row['recoverable_fn_count']}`, systematic `{row['systematic_fn_count']}`, recoverable_fraction `{row['recoverable_fraction']:.4f}`"
            )
        lines.extend(["", "### Crowd Buckets", ""])
        for row in payload.get("by_crowd_bucket", [])[:6]:
            lines.append(
                f"- `{row['crowd_bucket']}`: fn `{row['fn_count']}`, recoverable `{row['recoverable_fn_count']}`, systematic `{row['systematic_fn_count']}`, recoverable_fraction `{row['recoverable_fraction']:.4f}`"
            )
        lines.extend(["", "### Same-Desc Buckets", ""])
        for row in payload.get("by_same_desc_bucket", [])[:6]:
            lines.append(
                f"- `{row['same_desc_bucket']}`: fn `{row['fn_count']}`, recoverable `{row['recoverable_fn_count']}`, systematic `{row['systematic_fn_count']}`, recoverable_fraction `{row['recoverable_fraction']:.4f}`"
            )
        lines.extend(["", "### Top Categories", ""])
        for row in payload.get("top_categories", [])[:8]:
            lines.append(
                f"- `{row['category_name']}`: fn `{row['fn_count']}`, recoverable `{row['recoverable_fn_count']}`, systematic `{row['systematic_fn_count']}`, recoverable_fraction `{row['recoverable_fraction']:.4f}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_recall_slice_analysis(
    config_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    config = load_recall_slice_config(config_path)
    artifact_root = _artifact_root_for_repo(repo_root)
    output_root = artifact_root / config.run.output_dir
    run_dir = output_root / config.run.name
    run_dir.mkdir(parents=True, exist_ok=True)

    family_payloads: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for spec in config.families:
        oracle_path = _resolve_input_path(
            spec.oracle_fn_objects_jsonl,
            config_dir=config_path.parent,
            artifact_root=artifact_root,
        )
        subset_path = _resolve_input_path(
            spec.subset_jsonl,
            config_dir=config_path.parent,
            artifact_root=artifact_root,
        )
        subset_rows = _read_jsonl(subset_path)
        fn_rows = _read_jsonl(oracle_path)
        family_rows: list[dict[str, Any]] = []
        for row in fn_rows:
            if not bool(row.get("baseline_fn_loc")):
                continue
            record_idx = int(row.get("record_idx"))
            gt_idx = int(row.get("gt_idx"))
            subset_row = subset_rows[record_idx]
            enriched = _row_from_subset_record(subset_row, record_idx=record_idx, gt_idx=gt_idx)
            enriched.update(
                {
                    "family_alias": spec.family_alias,
                    "baseline_fn_loc": bool(row.get("baseline_fn_loc")),
                    "ever_recovered_loc": bool(row.get("ever_recovered_loc")),
                    "systematic_loc": bool(row.get("systematic_loc")),
                    "recover_count_loc": int(row.get("recover_count_loc") or 0),
                    "recover_fraction_loc": float(row.get("recover_fraction_loc") or 0.0),
                }
            )
            family_rows.append(enriched)
            all_rows.append(enriched)

        total = len(family_rows)
        recoverable = sum(1 for row in family_rows if bool(row.get("ever_recovered_loc")))
        systematic = sum(1 for row in family_rows if bool(row.get("systematic_loc")))
        family_payloads[spec.family_alias] = {
            "overall": {
                "baseline_fn_count_loc": int(total),
                "recoverable_fn_count_loc": int(recoverable),
                "systematic_fn_count_loc": int(systematic),
                "recoverable_fraction_of_baseline_fn_loc": float(recoverable / total) if total else 0.0,
            },
            "by_size_bucket": _aggregate_bucket(family_rows, "size_bucket"),
            "by_crowd_bucket": _aggregate_bucket(family_rows, "crowd_bucket"),
            "by_same_desc_bucket": _aggregate_bucket(family_rows, "same_desc_bucket"),
            "top_categories": _aggregate_bucket(family_rows, "category_name")[:20],
        }

    summary = {
        "run_name": config.run.name,
        "families": family_payloads,
        "artifacts": {
            "slice_rows_jsonl": str(run_dir / "slice_rows.jsonl"),
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_jsonl(run_dir / "slice_rows.jsonl", all_rows)
    (run_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    return {
        "run_dir": str(run_dir),
        "summary_json": str(run_dir / "summary.json"),
        "slice_rows_jsonl": str(run_dir / "slice_rows.jsonl"),
        "family_count": len(family_payloads),
        "row_count": len(all_rows),
    }
