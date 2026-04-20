from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str


@dataclass(frozen=True)
class ComparisonReportInputs:
    basin_summary_json: str
    recall_summary_json: str
    vision_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ComparisonReportConfig:
    run: RunConfig
    inputs: ComparisonReportInputs


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
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


def _require_row_list(parent: dict[str, Any], key: str) -> tuple[dict[str, Any], ...]:
    rows = parent.get(key, [])
    if not isinstance(rows, list):
        raise ValueError(f"{key} must be a list.")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{key}[{index}] must be a mapping.")
        normalized.append(dict(row))
    return tuple(normalized)


def load_comparison_report_config(config_path: Path) -> ComparisonReportConfig:
    payload = _load_yaml(config_path)
    run_raw = _require_mapping(payload, "run")
    inputs_raw = _require_mapping(payload, "inputs")
    return ComparisonReportConfig(
        run=RunConfig(
            name=_require_nonempty_str(run_raw, "name"),
            output_dir=_require_nonempty_str(run_raw, "output_dir"),
        ),
        inputs=ComparisonReportInputs(
            basin_summary_json=_require_nonempty_str(inputs_raw, "basin_summary_json"),
            recall_summary_json=_require_nonempty_str(inputs_raw, "recall_summary_json"),
            vision_rows=_require_row_list(inputs_raw, "vision_rows"),
        ),
    )


def _artifact_root_for_repo(repo_root: Path) -> Path:
    repo_root = repo_root.resolve()
    parts = list(repo_root.parts)
    if ".worktrees" in parts:
        marker = parts.index(".worktrees")
        return Path(*parts[:marker])
    return repo_root


def _resolve(path_str: str, *, base_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _resolve_input_path(
    path_str: str,
    *,
    config_dir: Path,
    artifact_root: Path,
) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    config_relative = config_dir / path
    if config_relative.exists():
        return config_relative
    return artifact_root / path


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _float_value(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        return float(value)
    return None


def derive_family_verdicts(
    *,
    basin_rows: list[dict[str, Any]],
    recall_rows: list[dict[str, Any]],
    vision_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_family: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "mass_at_4": [],
            "wrong_anchor_advantage_at_4": [],
            "suppressed_fn_rate": [],
            "competitive_fn_rate": [],
            "weak_visual_fn_rate": [],
            "vision_lift": [],
        }
    )

    for row in basin_rows:
        alias = str(row["family_alias"])
        target_metrics = row.get("mean_target_bbox_metrics")
        center_metrics = row.get("mean_center_bbox_metrics")
        if isinstance(target_metrics, dict):
            mass_at_4 = _float_value(target_metrics, "mass_at_4")
        else:
            mass_at_4 = _float_value(row, "mass_at_4")
        if mass_at_4 is not None:
            by_family[alias]["mass_at_4"].append(mass_at_4)
        if isinstance(center_metrics, dict) and isinstance(target_metrics, dict):
            center_mass_at_4 = _float_value(center_metrics, "mass_at_4")
            target_mass_at_4 = _float_value(target_metrics, "mass_at_4")
            wrong_anchor = (
                center_mass_at_4 - target_mass_at_4
                if center_mass_at_4 is not None and target_mass_at_4 is not None
                else None
            )
        else:
            wrong_anchor = _float_value(
                row,
                "wrong_anchor_advantage_at_4",
                "pred_minus_gt_mass_at_4",
            )
        if wrong_anchor is not None:
            by_family[alias]["wrong_anchor_advantage_at_4"].append(wrong_anchor)

    for row in recall_rows:
        alias = str(row["family_alias"])
        for key in ("suppressed_fn_rate", "competitive_fn_rate", "weak_visual_fn_rate"):
            value = _float_value(row, key)
            if value is not None:
                by_family[alias][key].append(value)

    for row in vision_rows:
        alias = str(row["family_alias"])
        vision_lift = _float_value(row, "vision_lift", "avg_gt_score_lift")
        if vision_lift is not None:
            by_family[alias]["vision_lift"].append(vision_lift)

    verdicts: dict[str, dict[str, Any]] = {}
    for alias in sorted(by_family):
        metrics = by_family[alias]
        mean_mass_at_4 = _mean(metrics["mass_at_4"])
        mean_suppressed_fn_rate = _mean(metrics["suppressed_fn_rate"])
        mean_competitive_fn_rate = _mean(metrics["competitive_fn_rate"])
        mean_weak_visual_fn_rate = _mean(metrics["weak_visual_fn_rate"])
        mean_vision_lift = _mean(metrics["vision_lift"])
        max_wrong_anchor_advantage = (
            max(metrics["wrong_anchor_advantage_at_4"])
            if metrics["wrong_anchor_advantage_at_4"]
            else None
        )

        if (
            max_wrong_anchor_advantage is not None
            and max_wrong_anchor_advantage >= 0.15
        ) or (
            mean_suppressed_fn_rate is not None and mean_suppressed_fn_rate >= 0.5
        ):
            family_health = "risky"
        elif (
            mean_mass_at_4 is not None
            and mean_mass_at_4 >= 0.75
            and (mean_vision_lift is None or mean_vision_lift >= 2.0)
            and (max_wrong_anchor_advantage is None or max_wrong_anchor_advantage < 0.10)
            and (mean_suppressed_fn_rate is None or mean_suppressed_fn_rate < 0.25)
        ):
            family_health = "strong"
        else:
            family_health = "mixed"

        verdicts[alias] = {
            "family_health": family_health,
            "mean_mass_at_4": mean_mass_at_4,
            "max_wrong_anchor_advantage_at_4": max_wrong_anchor_advantage,
            "mean_suppressed_fn_rate": mean_suppressed_fn_rate,
            "mean_competitive_fn_rate": mean_competitive_fn_rate,
            "mean_weak_visual_fn_rate": mean_weak_visual_fn_rate,
            "mean_vision_lift": mean_vision_lift,
            "num_basin_rows": len(metrics["mass_at_4"]),
            "num_recall_rows": len(metrics["suppressed_fn_rate"]),
            "num_vision_rows": len(metrics["vision_lift"]),
        }
    return verdicts


def _render_report_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Coordinate Family Comparison Report",
        "",
        "This is a minimal comparative scaffold for later family-level synthesis.",
        "",
        "## Families",
        "",
    ]
    verdicts = summary["verdicts"]
    if not verdicts:
        lines.append("- No family verdicts were derived from the provided inputs.")
    else:
        for family_alias, verdict in verdicts.items():
            lines.append(
                f"- `{family_alias}`: `{verdict['family_health']}` "
                f"(mass@4={verdict['mean_mass_at_4']}, "
                f"wrong-anchor={verdict['max_wrong_anchor_advantage_at_4']}, "
                f"vision_lift={verdict['mean_vision_lift']})"
            )
    lines.extend(
        [
            "",
            "## Sources",
            "",
            f"- Basin summary: `{summary['source_artifacts']['basin_summary_json']}`",
            f"- Recall summary: `{summary['source_artifacts']['recall_summary_json']}`",
            f"- Vision rows: `{len(summary['vision_rows'])}` inline rows",
        ]
    )
    return "\n".join(lines) + "\n"


def build_comparison_report(
    config_path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    resolved_config_path = config_path.resolve()
    config = load_comparison_report_config(resolved_config_path)
    config_dir = resolved_config_path.parent
    artifact_root = _artifact_root_for_repo((repo_root or REPO_ROOT).resolve())
    run_dir = _resolve(config.run.output_dir, base_dir=artifact_root) / config.run.name
    run_dir.mkdir(parents=True, exist_ok=True)

    basin_summary_path = _resolve_input_path(
        config.inputs.basin_summary_json,
        config_dir=config_dir,
        artifact_root=artifact_root,
    )
    recall_summary_path = _resolve_input_path(
        config.inputs.recall_summary_json,
        config_dir=config_dir,
        artifact_root=artifact_root,
    )
    basin_payload = _read_json(basin_summary_path)
    canonical_view = basin_payload.get("canonical_comparison_view", {})
    basin_rows = list(canonical_view.get("family_rollup", []))
    recall_rows = list(_read_json(recall_summary_path).get("family_metrics", []))
    vision_rows = [dict(row) for row in config.inputs.vision_rows]

    verdicts = derive_family_verdicts(
        basin_rows=basin_rows,
        recall_rows=recall_rows,
        vision_rows=vision_rows,
    )
    summary = {
        "run_name": config.run.name,
        "source_artifacts": {
            "basin_summary_json": str(basin_summary_path),
            "recall_summary_json": str(recall_summary_path),
        },
        "basin_rows": basin_rows,
        "recall_rows": recall_rows,
        "vision_rows": vision_rows,
        "verdicts": verdicts,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_jsonl(run_dir / "basin_rows.jsonl", basin_rows)
    _write_jsonl(run_dir / "recall_rows.jsonl", recall_rows)
    _write_jsonl(run_dir / "vision_rows.jsonl", vision_rows)
    (run_dir / "report.md").write_text(_render_report_md(summary), encoding="utf-8")
    return {
        "config_path": str(resolved_config_path),
        "run_dir": str(run_dir),
        "summary_json": str(run_dir / "summary.json"),
        "report_md": str(run_dir / "report.md"),
        "family_count": len(verdicts),
    }


__all__ = [
    "build_comparison_report",
    "derive_family_verdicts",
    "load_comparison_report_config",
]
