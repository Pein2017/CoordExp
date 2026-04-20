from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.analysis.coord_family_probe_registry import native_slot_names
from src.analysis.raw_text_coord_continuity_report import compute_basin_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str


@dataclass(frozen=True)
class BasinProbeRow:
    family_alias: str
    slot: str
    center_value: int
    target_value: int
    candidate_value: int
    score_mean: float
    abs_distance_to_target: int


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
                slot=_require_nonempty_str(item, "slot"),
                center_value=_require_int(item, "center_value"),
                target_value=_require_int(item, "target_value"),
                candidate_value=_require_int(item, "candidate_value"),
                score_mean=_require_float(item, "score_mean"),
                abs_distance_to_target=_require_int(item, "abs_distance_to_target"),
            )
        )
    return BasinProbeConfig(run=run, probe_rows=tuple(probe_rows))


def summarize_basin_rows(rows: list[BasinProbeRow]) -> list[dict[str, Any]]:
    if not rows:
        return []
    grouped: dict[tuple[str, str, int, int], list[BasinProbeRow]] = defaultdict(list)
    for row in rows:
        valid_slots = native_slot_names(row.family_alias)
        if row.slot not in valid_slots:
            raise ValueError(
                f"slot {row.slot!r} is not valid for family {row.family_alias!r}: {valid_slots}"
            )
        grouped[
            (row.family_alias, row.slot, row.center_value, row.target_value)
        ].append(row)

    summaries: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group_rows = grouped[key]
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
        head = group_rows[0]
        summaries.append(
            {
                "family_alias": head.family_alias,
                "slot": head.slot,
                "center_value": head.center_value,
                "target_value": head.target_value,
                "candidate_count": len(group_rows),
                **target_metrics,
                "center_mass_at_4": center_metrics["mass_at_4"],
            }
        )
    return summaries


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
    slot_metrics = summarize_basin_rows(list(config.probe_rows))
    rows_payload = [
        {
            "family_alias": row.family_alias,
            "slot": row.slot,
            "center_value": row.center_value,
            "target_value": row.target_value,
            "candidate_value": row.candidate_value,
            "score_mean": row.score_mean,
            "abs_distance_to_target": row.abs_distance_to_target,
        }
        for row in config.probe_rows
    ]

    summary_path = run_dir / "summary.json"
    rows_path = run_dir / "basin_rows.jsonl"
    _write_json(
        summary_path,
        {
            "run_name": config.run.name,
            "slot_metrics": slot_metrics,
        },
    )
    _write_jsonl(rows_path, rows_payload)
    return {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "summary_json": str(summary_path),
        "basin_rows_jsonl": str(rows_path),
        "slot_metric_count": len(slot_metrics),
    }


__all__ = [
    "BasinProbeConfig",
    "BasinProbeRow",
    "RunConfig",
    "load_basin_probe_config",
    "run_basin_probe",
    "summarize_basin_rows",
]
