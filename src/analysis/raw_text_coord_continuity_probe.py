from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import yaml

_VALID_STAGES = ("audit", "pilot", "canonical", "bad_basin", "dense_scene", "report")


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
    cfg = load_study_config(config_path)
    run_dir = Path(cfg.run.output_dir) / cfg.run.name
    run_dir.mkdir(parents=True, exist_ok=True)
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
            "val_headline": {
                "jsonl_path": cfg.cohorts.val_headline.jsonl_path,
                "sample_count": cfg.cohorts.val_headline.sample_count,
                "seed": cfg.cohorts.val_headline.seed,
            },
            "train_supplemental": {
                "jsonl_path": cfg.cohorts.train_supplemental.jsonl_path,
                "sample_count": cfg.cohorts.train_supplemental.sample_count,
                "seed": cfg.cohorts.train_supplemental.seed,
            },
        },
    }
    _write_json(run_dir / "summary.json", summary)
    return {"run_dir": str(run_dir), "summary": summary}
