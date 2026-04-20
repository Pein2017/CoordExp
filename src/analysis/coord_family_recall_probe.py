from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MECHANISM_LABELS = ("suppressed_fn", "competitive_fn", "weak_visual_fn")


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


def label_fn_rows(rows: list[FnProbeRow]) -> list[dict[str, Any]]:
    labeled_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        labeled_rows.append(
            {
                "fn_row_index": index,
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
        )
    return labeled_rows


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


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _mechanism_counts(labeled_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["fn_mechanism"]) for row in labeled_rows)
    return {mechanism: counts[mechanism] for mechanism in MECHANISM_LABELS}


def _mechanism_rates(
    *,
    counts: dict[str, int],
    total: int,
) -> dict[str, float]:
    if total <= 0:
        return {mechanism: 0.0 for mechanism in MECHANISM_LABELS}
    return {
        mechanism: counts[mechanism] / total
        for mechanism in MECHANISM_LABELS
    }


def summarize_family_mechanism_rows(
    labeled_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not labeled_rows:
        return []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labeled_rows:
        grouped[str(row["family_alias"])].append(row)

    summaries: list[dict[str, Any]] = []
    for family_alias in sorted(grouped):
        family_rows = grouped[family_alias]
        family_total = len(family_rows)
        for mechanism in sorted({str(row["fn_mechanism"]) for row in family_rows}):
            mechanism_rows = [
                row
                for row in family_rows
                if str(row["fn_mechanism"]) == mechanism
            ]
            summaries.append(
                {
                    "family_alias": family_alias,
                    "fn_mechanism": mechanism,
                    "fn_count": len(mechanism_rows),
                    "fn_rate": len(mechanism_rows) / family_total,
                    "teacher_forced_support_mean": _mean(
                        [float(row["teacher_forced_support"]) for row in mechanism_rows]
                    ),
                    "proposal_support_mean": _mean(
                        [float(row["proposal_support"]) for row in mechanism_rows]
                    ),
                    "oracle_k_recovery_rate": _mean(
                        [
                            1.0 if bool(row["oracle_k_recovered"]) else 0.0
                            for row in mechanism_rows
                        ]
                    ),
                    "competitor_margin_mean": _mean(
                        [float(row["competitor_margin"]) for row in mechanism_rows]
                    ),
                }
            )
    return summaries


def summarize_fn_rows(rows: list[FnProbeRow]) -> list[dict[str, Any]]:
    labeled_rows = label_fn_rows(rows)
    if not labeled_rows:
        return []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labeled_rows:
        grouped[str(row["family_alias"])].append(row)

    summaries: list[dict[str, Any]] = []
    for family_alias in sorted(grouped):
        family_rows = grouped[family_alias]
        counts = _mechanism_counts(family_rows)
        total = len(family_rows)
        rates = _mechanism_rates(counts=counts, total=total)
        summaries.append(
            {
                "family_alias": family_alias,
                "fn_count": total,
                "suppressed_fn_count": counts["suppressed_fn"],
                "competitive_fn_count": counts["competitive_fn"],
                "weak_visual_fn_count": counts["weak_visual_fn"],
                "suppressed_fn_rate": rates["suppressed_fn"],
                "competitive_fn_rate": rates["competitive_fn"],
                "weak_visual_fn_rate": rates["weak_visual_fn"],
                "mechanism_counts": counts,
                "mechanism_rates": rates,
                "teacher_forced_support_mean": _mean(
                    [float(row["teacher_forced_support"]) for row in family_rows]
                ),
                "proposal_support_mean": _mean(
                    [float(row["proposal_support"]) for row in family_rows]
                ),
                "oracle_k_recovery_rate": _mean(
                    [1.0 if bool(row["oracle_k_recovered"]) else 0.0 for row in family_rows]
                ),
                "competitor_margin_mean": _mean(
                    [float(row["competitor_margin"]) for row in family_rows]
                ),
            }
        )
    return summaries


def summarize_mechanism_totals(
    labeled_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = _mechanism_counts(labeled_rows)
    total = len(labeled_rows)
    rates = _mechanism_rates(counts=counts, total=total)
    return {
        "fn_count": total,
        "family_count": len({str(row["family_alias"]) for row in labeled_rows}),
        "suppressed_fn_count": counts["suppressed_fn"],
        "competitive_fn_count": counts["competitive_fn"],
        "weak_visual_fn_count": counts["weak_visual_fn"],
        "suppressed_fn_rate": rates["suppressed_fn"],
        "competitive_fn_rate": rates["competitive_fn"],
        "weak_visual_fn_rate": rates["weak_visual_fn"],
        "mechanism_counts": counts,
        "mechanism_rates": rates,
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


def run_recall_probe(
    config_path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = (repo_root or REPO_ROOT).resolve()
    config = load_recall_probe_config(config_path)
    run_dir = _run_dir(config.run, repo_root)
    rows_payload = label_fn_rows(list(config.fn_rows))
    family_metrics = summarize_fn_rows(list(config.fn_rows))
    family_mechanism_rows = summarize_family_mechanism_rows(rows_payload)
    mechanism_summary = summarize_mechanism_totals(rows_payload)

    summary_path = run_dir / "summary.json"
    rows_path = run_dir / "fn_rows.jsonl"
    _write_json(
        summary_path,
        {
            "run_name": config.run.name,
            "artifacts": {"fn_rows_jsonl": str(rows_path)},
            "fn_row_count": len(rows_payload),
            "family_metrics": family_metrics,
            "family_mechanism_rows": family_mechanism_rows,
            "mechanism_summary": mechanism_summary,
        },
    )
    _write_jsonl(rows_path, rows_payload)
    return {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "summary_json": str(summary_path),
        "fn_rows_jsonl": str(rows_path),
        "family_metric_count": len(family_metrics),
        "labeled_fn_row_count": len(rows_payload),
    }


__all__ = [
    "FnProbeRow",
    "RecallProbeConfig",
    "RunConfig",
    "classify_fn_mechanism",
    "label_fn_rows",
    "load_recall_probe_config",
    "run_recall_probe",
    "summarize_family_mechanism_rows",
    "summarize_fn_rows",
    "summarize_mechanism_totals",
]
