from __future__ import annotations

import json
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
class ReportInputs:
    basin_summary_json: str
    recall_summary_json: str
    vision_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ComparisonReportConfig:
    run: RunConfig
    inputs: ReportInputs


def derive_family_verdicts(
    *,
    basin_rows: list[dict[str, Any]],
    recall_rows: list[dict[str, Any]],
    vision_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_alias: dict[str, dict[str, Any]] = {}
    for row in basin_rows:
        alias = str(row["family_alias"])
        by_alias.setdefault(alias, {})["basin"] = row
    for row in recall_rows:
        alias = str(row["family_alias"])
        by_alias.setdefault(alias, {})["recall"] = row
    for row in vision_rows:
        alias = str(row["family_alias"])
        by_alias.setdefault(alias, {})["vision"] = row

    verdicts: dict[str, dict[str, Any]] = {}
    for alias, pieces in sorted(by_alias.items()):
        basin = pieces.get("basin", {})
        recall = pieces.get("recall", {})
        vision = pieces.get("vision", {})
        bad_basin = float(basin.get("wrong_anchor_advantage_at_4", 0.0) or 0.0)
        competitive = float(recall.get("competitive_fn_rate", 0.0) or 0.0)
        mass_at_4 = float(basin.get("mass_at_4", 0.0) or 0.0)
        vision_lift = float(vision.get("vision_lift", 0.0) or 0.0)
        if bad_basin >= 0.15 or competitive >= 0.30:
            family_health = "risky"
        elif mass_at_4 >= 0.70 and competitive < 0.20 and vision_lift > 0.0:
            family_health = "promising"
        else:
            family_health = "mixed"
        verdicts[alias] = {
            "family_health": family_health,
            "basin_strength": basin.get("mass_at_4"),
            "competitive_fn_rate": recall.get("competitive_fn_rate"),
            "suppressed_fn_rate": recall.get("suppressed_fn_rate"),
            "vision_lift": vision.get("vision_lift"),
        }
    return verdicts


def _artifact_root_for_repo(repo_root: Path) -> Path:
    repo_root = repo_root.resolve()
    parts = list(repo_root.parts)
    if ".worktrees" in parts:
        marker = parts.index(".worktrees")
        return Path(*parts[:marker])
    return repo_root


def _resolve_input_path(path_str: str, *, repo_root: Path, config_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    config_relative = config_dir / path
    if config_relative.exists():
        return config_relative
    return _artifact_root_for_repo(repo_root) / path


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


def load_comparison_report_config(config_path: Path) -> ComparisonReportConfig:
    payload = _load_yaml(config_path)
    run_raw = _require_mapping(payload, "run")
    run = RunConfig(
        name=_require_nonempty_str(run_raw, "name"),
        output_dir=_require_nonempty_str(run_raw, "output_dir"),
    )
    inputs_raw = _require_mapping(payload, "inputs")
    vision_rows_raw = inputs_raw.get("vision_rows", [])
    if not isinstance(vision_rows_raw, list):
        raise ValueError("inputs.vision_rows must be a list.")
    return ComparisonReportConfig(
        run=run,
        inputs=ReportInputs(
            basin_summary_json=_require_nonempty_str(inputs_raw, "basin_summary_json"),
            recall_summary_json=_require_nonempty_str(inputs_raw, "recall_summary_json"),
            vision_rows=tuple(vision_rows_raw),
        ),
    )


def _run_dir(run: RunConfig, repo_root: Path) -> Path:
    output_dir = Path(run.output_dir)
    if not output_dir.is_absolute():
        output_dir = _artifact_root_for_repo(repo_root) / output_dir
    return output_dir / run.name


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _render_report(verdicts: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Coord Family Comparison Report",
        "",
        "| Family | Health | Basin Strength | Competitive FN Rate | Vision Lift |",
        "| --- | --- | --- | --- | --- |",
    ]
    for alias, row in sorted(verdicts.items()):
        lines.append(
            f"| {alias} | {row['family_health']} | {row.get('basin_strength')} | "
            f"{row.get('competitive_fn_rate')} | {row.get('vision_lift')} |"
        )
    return "\n".join(lines) + "\n"


def build_comparison_report(
    config_path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = (repo_root or REPO_ROOT).resolve()
    config = load_comparison_report_config(config_path)
    config_dir = config_path.resolve().parent
    basin_path = _resolve_input_path(
        config.inputs.basin_summary_json,
        repo_root=repo_root,
        config_dir=config_dir,
    )
    recall_path = _resolve_input_path(
        config.inputs.recall_summary_json,
        repo_root=repo_root,
        config_dir=config_dir,
    )
    basin_payload = json.loads(basin_path.read_text(encoding="utf-8"))
    recall_payload = json.loads(recall_path.read_text(encoding="utf-8"))
    basin_rows = list(basin_payload.get("slot_metrics", []))
    recall_rows = list(recall_payload.get("family_metrics", []))
    vision_rows = list(config.inputs.vision_rows)
    verdicts = derive_family_verdicts(
        basin_rows=basin_rows,
        recall_rows=recall_rows,
        vision_rows=vision_rows,
    )

    run_dir = _run_dir(config.run, repo_root)
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    _write_json(
        summary_path,
        {
            "run_name": config.run.name,
            "verdicts": verdicts,
        },
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(verdicts), encoding="utf-8")
    return {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "summary_json": str(summary_path),
        "report_md": str(report_path),
        "family_count": len(verdicts),
    }


__all__ = [
    "ComparisonReportConfig",
    "ReportInputs",
    "RunConfig",
    "build_comparison_report",
    "derive_family_verdicts",
    "load_comparison_report_config",
]
