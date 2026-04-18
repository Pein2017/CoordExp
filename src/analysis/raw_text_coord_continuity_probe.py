from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import yaml

from src.analysis.raw_text_coord_continuity_report import write_report_bundle

_VALID_STAGES = ("audit", "pilot", "canonical", "bad_basin", "dense_scene", "report")
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str
    stages: tuple[str, ...]


@dataclass(frozen=True)
class ModelConfig:
    alias: str
    path: str
    prompt_surface: str


@dataclass(frozen=True)
class CohortConfig:
    jsonl_path: str
    sample_count: int
    seed: int


@dataclass(frozen=True)
class StudyModels:
    base: ModelConfig
    pure_ce: ModelConfig


@dataclass(frozen=True)
class StudyCohorts:
    val_headline: CohortConfig
    train_supplemental: CohortConfig


@dataclass(frozen=True)
class StudyConfig:
    run: RunConfig
    models: StudyModels
    cohorts: StudyCohorts


def _load_yaml(config_path: Path) -> dict[str, object]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("study config root must be a mapping")
    return raw


def _require_mapping(parent: dict[str, object], key: str) -> dict[str, object]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_input_path(path_str: str, *, config_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    config_relative = config_dir / path
    if config_relative.exists():
        return config_relative
    return REPO_ROOT / path


def _resolve_output_dir(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"jsonl row in {path} must be a mapping")
        rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in rows),
        encoding="utf-8",
    )


def run_phase0_audit(scorer: object) -> dict[str, object]:
    numbers = [0, 1, 9, 10, 99, 100, 199, 200, 210, 999]
    tokenizer = getattr(scorer, "tokenizer")
    rows = []
    for value in numbers:
        tokens = list(tokenizer.tokenize(str(value)))
        rows.append(
            {
                "value": value,
                "tokens": tokens,
                "token_count": len(tokens),
            }
        )
    return {"numbers": rows}


def build_random_cohort(
    rows: list[dict[str, object]],
    *,
    sample_count: int,
    seed: int,
) -> list[dict[str, object]]:
    cohort = list(rows)
    random.Random(seed).shuffle(cohort)
    return cohort[:sample_count]


def build_study_hard_cases(
    rows: list[dict[str, object]],
    *,
    max_cases: int,
) -> list[dict[str, object]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row.get("same_desc_duplicate_pair_count") or 0),
            int(row.get("max_desc_count") or 0),
            int(row.get("pred_count") or 0),
        ),
        reverse=True,
    )
    return ordered[:max_cases]


def _materialize_random_cohort(
    cohort_name: str,
    cohort_cfg: CohortConfig,
    *,
    config_dir: Path,
    run_dir: Path,
) -> dict[str, object]:
    source_path = _resolve_input_path(cohort_cfg.jsonl_path, config_dir=config_dir)
    source_rows = _read_jsonl(source_path)
    selected_rows = build_random_cohort(
        source_rows,
        sample_count=cohort_cfg.sample_count,
        seed=cohort_cfg.seed,
    )
    manifest_path = run_dir / "cohorts" / f"{cohort_name}.jsonl"
    _write_jsonl(manifest_path, selected_rows)
    return {
        "jsonl_path": cohort_cfg.jsonl_path,
        "resolved_jsonl_path": str(source_path),
        "sample_count": cohort_cfg.sample_count,
        "seed": cohort_cfg.seed,
        "manifest_path": str(manifest_path),
        "num_rows": len(selected_rows),
    }


def load_study_config(config_path: Path) -> StudyConfig:
    raw = _load_yaml(config_path)
    run_raw = _require_mapping(raw, "run")
    models_raw = _require_mapping(raw, "models")
    cohorts_raw = _require_mapping(raw, "cohorts")
    stages = tuple(str(value) for value in run_raw.get("stages") or ())
    invalid_stages = tuple(stage for stage in stages if stage not in _VALID_STAGES)
    if invalid_stages:
        raise ValueError(f"unsupported stage(s): {', '.join(invalid_stages)}")
    return StudyConfig(
        run=RunConfig(
            name=str(run_raw["name"]),
            output_dir=str(run_raw["output_dir"]),
            stages=stages,
        ),
        models=StudyModels(
            base=ModelConfig(
                alias=str(_require_mapping(models_raw, "base")["alias"]),
                path=str(_require_mapping(models_raw, "base")["path"]),
                prompt_surface=str(_require_mapping(models_raw, "base")["prompt_surface"]),
            ),
            pure_ce=ModelConfig(
                alias=str(_require_mapping(models_raw, "pure_ce")["alias"]),
                path=str(_require_mapping(models_raw, "pure_ce")["path"]),
                prompt_surface=str(_require_mapping(models_raw, "pure_ce")["prompt_surface"]),
            ),
        ),
        cohorts=StudyCohorts(
            val_headline=CohortConfig(
                jsonl_path=str(_require_mapping(cohorts_raw, "val_headline")["jsonl_path"]),
                sample_count=int(_require_mapping(cohorts_raw, "val_headline")["sample_count"]),
                seed=int(_require_mapping(cohorts_raw, "val_headline")["seed"]),
            ),
            train_supplemental=CohortConfig(
                jsonl_path=str(_require_mapping(cohorts_raw, "train_supplemental")["jsonl_path"]),
                sample_count=int(_require_mapping(cohorts_raw, "train_supplemental")["sample_count"]),
                seed=int(_require_mapping(cohorts_raw, "train_supplemental")["seed"]),
            ),
        ),
    )


def run_study(config_path: Path) -> dict[str, object]:
    resolved_config_path = config_path.resolve()
    cfg = load_study_config(resolved_config_path)
    run_dir = _resolve_output_dir(cfg.run.output_dir) / cfg.run.name
    run_dir.mkdir(parents=True, exist_ok=True)
    val_cohort = _materialize_random_cohort(
        "val_headline",
        cfg.cohorts.val_headline,
        config_dir=resolved_config_path.parent,
        run_dir=run_dir,
    )
    train_cohort = _materialize_random_cohort(
        "train_supplemental",
        cfg.cohorts.train_supplemental,
        config_dir=resolved_config_path.parent,
        run_dir=run_dir,
    )
    summary = {
        "run_name": cfg.run.name,
        "stages": list(cfg.run.stages),
        "models": {
            "base": {
                "alias": cfg.models.base.alias,
                "path": cfg.models.base.path,
                "prompt_surface": cfg.models.base.prompt_surface,
            },
            "pure_ce": {
                "alias": cfg.models.pure_ce.alias,
                "path": cfg.models.pure_ce.path,
                "prompt_surface": cfg.models.pure_ce.prompt_surface,
            },
        },
        "cohorts": {
            "val_headline": val_cohort,
            "train_supplemental": train_cohort,
        },
    }
    if "report" in cfg.run.stages:
        write_report_bundle(
            out_dir=run_dir,
            summary=summary,
            report_md="# Raw-Text Coordinate Continuity Probe\n",
            per_coord_rows=[],
            hard_cases=[],
        )
    else:
        _write_json(run_dir / "summary.json", summary)
    return {"run_dir": str(run_dir), "summary": summary}
