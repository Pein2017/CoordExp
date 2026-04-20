from __future__ import annotations

import json
from collections import Counter, defaultdict
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
class FnProbeRow:
    family_alias: str
    teacher_forced_support: float
    proposal_support: float
    oracle_k_recovered: bool
    competitor_margin: float


@dataclass(frozen=True)
class RecallProbeConfig:
    run: RunConfig
    fn_rows: tuple[FnProbeRow, ...]


def classify_fn_mechanism(
    *,
    teacher_forced_support: float,
    proposal_support: float,
    oracle_k_recovered: bool,
    competitor_margin: float,
) -> str:
    if competitor_margin > 0.20:
        return "competitive_fn"
    if (
        teacher_forced_support >= 0.60
        and proposal_support >= 0.60
        and oracle_k_recovered
    ):
        return "suppressed_fn"
    return "weak_visual_fn"


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


def _require_float(parent: dict[str, Any], key: str) -> float:
    value = parent.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric.")
    return float(value)


def _require_bool(parent: dict[str, Any], key: str) -> bool:
    value = parent.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean.")
    return value


def load_recall_probe_config(config_path: Path) -> RecallProbeConfig:
    payload = _load_yaml(config_path)
    run_raw = _require_mapping(payload, "run")
    run = RunConfig(
        name=_require_nonempty_str(run_raw, "name"),
        output_dir=_require_nonempty_str(run_raw, "output_dir"),
    )
    rows_raw = payload.get("fn_rows")
    if not isinstance(rows_raw, list) or not rows_raw:
        raise ValueError("fn_rows must be a non-empty list.")
    fn_rows: list[FnProbeRow] = []
    for index, item in enumerate(rows_raw):
        if not isinstance(item, dict):
            raise ValueError(f"fn_rows[{index}] must be a mapping.")
        fn_rows.append(
            FnProbeRow(
                family_alias=_require_nonempty_str(item, "family_alias"),
                teacher_forced_support=_require_float(item, "teacher_forced_support"),
                proposal_support=_require_float(item, "proposal_support"),
                oracle_k_recovered=_require_bool(item, "oracle_k_recovered"),
                competitor_margin=_require_float(item, "competitor_margin"),
            )
        )
    return RecallProbeConfig(run=run, fn_rows=tuple(fn_rows))


def summarize_fn_rows(rows: list[FnProbeRow]) -> list[dict[str, Any]]:
    if not rows:
        return []
    grouped: dict[str, list[FnProbeRow]] = defaultdict(list)
    for row in rows:
        grouped[row.family_alias].append(row)

    summaries: list[dict[str, Any]] = []
    for family_alias in sorted(grouped):
        family_rows = grouped[family_alias]
        labels = [
            classify_fn_mechanism(
                teacher_forced_support=row.teacher_forced_support,
                proposal_support=row.proposal_support,
                oracle_k_recovered=row.oracle_k_recovered,
                competitor_margin=row.competitor_margin,
            )
            for row in family_rows
        ]
        counts = Counter(labels)
        total = len(family_rows)
        summaries.append(
            {
                "family_alias": family_alias,
                "fn_count": total,
                "suppressed_fn_rate": counts["suppressed_fn"] / total,
                "competitive_fn_rate": counts["competitive_fn"] / total,
                "weak_visual_fn_rate": counts["weak_visual_fn"] / total,
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


def run_recall_probe(
    config_path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = (repo_root or REPO_ROOT).resolve()
    config = load_recall_probe_config(config_path)
    run_dir = _run_dir(config.run, repo_root)
    family_metrics = summarize_fn_rows(list(config.fn_rows))
    rows_payload = [
        {
            "family_alias": row.family_alias,
            "teacher_forced_support": row.teacher_forced_support,
            "proposal_support": row.proposal_support,
            "oracle_k_recovered": row.oracle_k_recovered,
            "competitor_margin": row.competitor_margin,
            "fn_mechanism": classify_fn_mechanism(
                teacher_forced_support=row.teacher_forced_support,
                proposal_support=row.proposal_support,
                oracle_k_recovered=row.oracle_k_recovered,
                competitor_margin=row.competitor_margin,
            ),
        }
        for row in config.fn_rows
    ]

    summary_path = run_dir / "summary.json"
    rows_path = run_dir / "fn_rows.jsonl"
    _write_json(
        summary_path,
        {
            "run_name": config.run.name,
            "family_metrics": family_metrics,
        },
    )
    _write_jsonl(rows_path, rows_payload)
    return {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "summary_json": str(summary_path),
        "fn_rows_jsonl": str(rows_path),
        "family_metric_count": len(family_metrics),
    }


__all__ = [
    "FnProbeRow",
    "RecallProbeConfig",
    "RunConfig",
    "classify_fn_mechanism",
    "load_recall_probe_config",
    "run_recall_probe",
    "summarize_fn_rows",
]
