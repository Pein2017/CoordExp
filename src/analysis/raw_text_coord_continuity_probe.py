from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import yaml

from src.infer.checkpoints import resolve_inference_checkpoint
from src.utils.assistant_json import CANONICAL_JSON_SEPARATORS, dumps_canonical_json
from src.analysis.raw_text_coord_continuity_report import write_report_bundle

_VALID_STAGES = ("audit", "pilot", "canonical", "bad_basin", "dense_scene", "report")
REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_JSON_SURFACE = "pretty_inline"


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
    coord_mode: str = "norm1000_text"
    prompt_variant: str = "coco_80"
    object_field_order: str = "desc_first"
    json_surface: str = _CANONICAL_JSON_SURFACE


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


def _resolve_audit_model_info(model_path: Path) -> dict[str, object]:
    resolved = resolve_inference_checkpoint(model_checkpoint=str(model_path))
    processor_source = str(resolved.resolved_base_model_checkpoint)
    processor_source_path = Path(processor_source)
    return {
        "requested_model_path": str(model_path),
        "checkpoint_mode": resolved.checkpoint_mode,
        "resolved_base_model_checkpoint": resolved.resolved_base_model_checkpoint,
        "resolved_adapter_checkpoint": resolved.resolved_adapter_checkpoint,
        "processor_source": processor_source,
        "processor_source_is_local": processor_source_path.exists(),
        "has_coord_offset_adapter": bool(
            resolved.adapter_info is not None
            and resolved.adapter_info.coord_offset_spec is not None
        ),
    }


def _load_tokenizer_for_audit(model_path: Path) -> object:
    from transformers import AutoProcessor

    model_info = _resolve_audit_model_info(model_path)
    processor_source = str(model_info["processor_source"])
    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=True,
        local_files_only=bool(model_info["processor_source_is_local"]),
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(
            f"processor at {processor_source} did not expose a tokenizer"
        )
    return tokenizer


def _tokenize_text_for_audit(tokenizer: object, text: str) -> dict[str, object]:
    if hasattr(tokenizer, "encode"):
        input_ids = list(tokenizer.encode(text, add_special_tokens=False))
    else:
        tokens = list(getattr(tokenizer, "tokenize")(text))
        return {
            "text": text,
            "token_ids": list(range(len(tokens))),
            "tokens": tokens,
            "token_count": len(tokens),
        }
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        tokens = list(tokenizer.convert_ids_to_tokens(input_ids))
    else:
        tokens = list(getattr(tokenizer, "tokenize")(text))
    return {
        "text": text,
        "token_ids": input_ids,
        "tokens": tokens,
        "token_count": len(input_ids),
    }


def _first_diff_index(left: list[int], right: list[int]) -> int | None:
    for idx, (left_id, right_id) in enumerate(zip(left, right)):
        if int(left_id) != int(right_id):
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _build_surface_form_audit(tokenizer: object) -> dict[str, object]:
    sample_payload = {
        "objects": [
            {"desc": "book", "bbox_2d": [199, 200, 210, 250]},
            {"desc": "book", "bbox_2d": [231, 200, 260, 280]},
        ]
    }
    pretty_inline = dumps_canonical_json(sample_payload)
    compact = json.dumps(sample_payload, ensure_ascii=False, separators=(",", ":"))
    pretty_multiline = json.dumps(sample_payload, ensure_ascii=False, indent=2)
    variants = [
        ("pretty_inline", pretty_inline),
        ("compact", compact),
        ("pretty_multiline", pretty_multiline),
    ]
    tokenized = [
        {
            "label": label,
            **_tokenize_text_for_audit(tokenizer, text),
        }
        for label, text in variants
    ]
    canonical_token_ids = list(tokenized[0]["token_ids"])
    for row in tokenized[1:]:
        row["first_diff_vs_pretty_inline"] = _first_diff_index(
            canonical_token_ids,
            list(row["token_ids"]),
        )
    tokenized[0]["first_diff_vs_pretty_inline"] = None
    return {
        "canonical_label": _CANONICAL_JSON_SURFACE,
        "canonical_separators": list(CANONICAL_JSON_SEPARATORS),
        "sample_payload": sample_payload,
        "variants": tokenized,
    }


def run_phase0_audit(scorer: object) -> dict[str, object]:
    numbers = [0, 1, 9, 10, 99, 100, 199, 200, 210, 999]
    tokenizer = getattr(scorer, "tokenizer")
    rows = []
    for value in numbers:
        tokenized = _tokenize_text_for_audit(tokenizer, str(value))
        rows.append(
            {
                "value": value,
                "tokens": tokenized["tokens"],
                "token_ids": tokenized["token_ids"],
                "token_count": tokenized["token_count"],
            }
        )
    return {
        "numbers": rows,
        "surface_forms": _build_surface_form_audit(tokenizer),
    }


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


def _run_tokenization_audit(
    *,
    cfg: StudyConfig,
    config_dir: Path,
    run_dir: Path,
) -> dict[str, object]:
    audit_dir = run_dir / "audit"
    models = {
        "base": cfg.models.base,
        "pure_ce": cfg.models.pure_ce,
    }
    artifacts: dict[str, object] = {}
    for model_key, model_cfg in models.items():
        resolved_model_path = _resolve_input_path(model_cfg.path, config_dir=config_dir)
        model_info = _resolve_audit_model_info(resolved_model_path)
        tokenizer = _load_tokenizer_for_audit(resolved_model_path)
        audit = run_phase0_audit(type("AuditSurface", (), {"tokenizer": tokenizer})())
        audit["model_alias"] = model_cfg.alias
        audit["model_path"] = model_cfg.path
        audit["resolved_model_path"] = str(resolved_model_path)
        audit["serialization_surface"] = _CANONICAL_JSON_SURFACE
        audit["model_resolution"] = model_info
        out_path = audit_dir / f"{model_key}_tokenization.json"
        _write_json(out_path, audit)
        artifacts[model_key] = str(out_path)
    summary = {
        "artifacts": artifacts,
        "serialization_surface": _CANONICAL_JSON_SURFACE,
    }
    _write_json(audit_dir / "summary.json", summary)
    return summary


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
                coord_mode=str(
                    _require_mapping(models_raw, "base").get("coord_mode", "norm1000_text")
                ),
                prompt_variant=str(
                    _require_mapping(models_raw, "base").get("prompt_variant", "coco_80")
                ),
                object_field_order=str(
                    _require_mapping(models_raw, "base").get(
                        "object_field_order",
                        "desc_first",
                    )
                ),
                json_surface=str(
                    _require_mapping(models_raw, "base").get(
                        "json_surface",
                        _CANONICAL_JSON_SURFACE,
                    )
                ),
            ),
            pure_ce=ModelConfig(
                alias=str(_require_mapping(models_raw, "pure_ce")["alias"]),
                path=str(_require_mapping(models_raw, "pure_ce")["path"]),
                prompt_surface=str(_require_mapping(models_raw, "pure_ce")["prompt_surface"]),
                coord_mode=str(
                    _require_mapping(models_raw, "pure_ce").get(
                        "coord_mode",
                        "norm1000_text",
                    )
                ),
                prompt_variant=str(
                    _require_mapping(models_raw, "pure_ce").get(
                        "prompt_variant",
                        "coco_80",
                    )
                ),
                object_field_order=str(
                    _require_mapping(models_raw, "pure_ce").get(
                        "object_field_order",
                        "desc_first",
                    )
                ),
                json_surface=str(
                    _require_mapping(models_raw, "pure_ce").get(
                        "json_surface",
                        _CANONICAL_JSON_SURFACE,
                    )
                ),
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
                "coord_mode": cfg.models.base.coord_mode,
                "prompt_variant": cfg.models.base.prompt_variant,
                "object_field_order": cfg.models.base.object_field_order,
                "json_surface": cfg.models.base.json_surface,
            },
            "pure_ce": {
                "alias": cfg.models.pure_ce.alias,
                "path": cfg.models.pure_ce.path,
                "prompt_surface": cfg.models.pure_ce.prompt_surface,
                "coord_mode": cfg.models.pure_ce.coord_mode,
                "prompt_variant": cfg.models.pure_ce.prompt_variant,
                "object_field_order": cfg.models.pure_ce.object_field_order,
                "json_surface": cfg.models.pure_ce.json_surface,
            },
        },
        "cohorts": {
            "val_headline": val_cohort,
            "train_supplemental": train_cohort,
        },
    }
    if "audit" in cfg.run.stages:
        summary["audit"] = _run_tokenization_audit(
            cfg=cfg,
            config_dir=resolved_config_path.parent,
            run_dir=run_dir,
        )
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
