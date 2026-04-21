from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from src.analysis.raw_text_coordinate_case_bank import (
    build_case_bank_rows,
    freeze_review_shortlist,
)
from src.analysis.raw_text_coordinate_behavior import summarize_confirmatory_records
from src.analysis.raw_text_coordinate_exploratory import (
    build_prefix_intervention_matrix,
    label_fn_bucket,
)
from src.analysis.raw_text_coordinate_mechanism_report import write_report_bundle
from src.analysis.raw_text_coordinate_review_queue import build_review_queue_rows


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str
    stages: tuple[str, ...]


@dataclass(frozen=True)
class ModelObjectConfig:
    alias: str
    base_path: str
    adapter_path: str | None
    prompt_variant: str
    object_field_order: str
    serializer_surfaces: tuple[str, ...]


@dataclass(frozen=True)
class DatasetConfig:
    train_jsonl: str
    val_jsonl: str


@dataclass(frozen=True)
class ExecutionConfig:
    gpu_ids: tuple[int, ...]
    reuse_existing: bool


@dataclass(frozen=True)
class ReviewConfig:
    fp_budget: int
    fn_budget: int


@dataclass(frozen=True)
class StudyModels:
    base_only: ModelObjectConfig
    base_plus_adapter: ModelObjectConfig


@dataclass(frozen=True)
class StudyConfig:
    run: RunConfig
    models: StudyModels
    dataset: DatasetConfig
    execution: ExecutionConfig
    review: ReviewConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("study config must be a mapping")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_study_config(config_path: Path) -> StudyConfig:
    raw = _load_yaml(config_path)
    run_raw = raw["run"]
    models_raw = raw["models"]
    dataset_raw = raw["dataset"]
    execution_raw = raw["execution"]
    review_raw = raw["review"]
    return StudyConfig(
        run=RunConfig(
            name=str(run_raw["name"]),
            output_dir=str(run_raw["output_dir"]),
            stages=tuple(str(value) for value in run_raw["stages"]),
        ),
        models=StudyModels(
            base_only=ModelObjectConfig(
                alias=str(models_raw["base_only"]["alias"]),
                base_path=str(models_raw["base_only"]["base_path"]),
                adapter_path=models_raw["base_only"]["adapter_path"],
                prompt_variant=str(models_raw["base_only"]["prompt_variant"]),
                object_field_order=str(models_raw["base_only"]["object_field_order"]),
                serializer_surfaces=tuple(models_raw["base_only"]["serializer_surfaces"]),
            ),
            base_plus_adapter=ModelObjectConfig(
                alias=str(models_raw["base_plus_adapter"]["alias"]),
                base_path=str(models_raw["base_plus_adapter"]["base_path"]),
                adapter_path=models_raw["base_plus_adapter"]["adapter_path"],
                prompt_variant=str(models_raw["base_plus_adapter"]["prompt_variant"]),
                object_field_order=str(models_raw["base_plus_adapter"]["object_field_order"]),
                serializer_surfaces=tuple(models_raw["base_plus_adapter"]["serializer_surfaces"]),
            ),
        ),
        dataset=DatasetConfig(**dataset_raw),
        execution=ExecutionConfig(
            gpu_ids=tuple(int(value) for value in execution_raw["gpu_ids"]),
            reuse_existing=bool(execution_raw["reuse_existing"]),
        ),
        review=ReviewConfig(
            fp_budget=int(review_raw["fp_budget"]),
            fn_budget=int(review_raw["fn_budget"]),
        ),
    )


def plan_stage_cells(
    *,
    stage: str,
    gpu_ids: Sequence[int],
    model_aliases: Sequence[str],
    branch_names: Sequence[str],
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    if stage == "exploratory":
        for gpu_id, (model_alias, branch_name) in zip(
            gpu_ids,
            (
                (model_alias, branch_name)
                for model_alias in model_aliases
                for branch_name in branch_names
            ),
            strict=False,
        ):
            cells.append(
                {
                    "stage": stage,
                    "gpu_id": int(gpu_id),
                    "model_alias": str(model_alias),
                    "branch_name": str(branch_name),
                }
            )
    return cells


def run_case_bank_stage(
    *,
    duplicate_rows: list[dict[str, object]],
    fn_rows: list[dict[str, object]],
    fp_budget: int,
    fn_budget: int,
) -> dict[str, object]:
    case_rows = build_case_bank_rows(
        duplicate_rows=duplicate_rows,
        fn_rows=fn_rows,
    )
    shortlist = freeze_review_shortlist(
        case_rows,
        fp_budget=fp_budget,
        fn_budget=fn_budget,
    )
    return {
        "case_row_count": len(case_rows),
        "shortlist_count": len(shortlist),
    }


def run_confirmatory_stage(
    *,
    records: list[dict[str, object]],
) -> dict[str, object]:
    summary_rows = summarize_confirmatory_records(records=records)
    return {
        "summary_rows": summary_rows,
        "serializer_surfaces": sorted(
            {row["serializer_surface"] for row in summary_rows}
        ),
    }


def run_exploratory_stage(*, cases: list[dict[str, object]]) -> dict[str, object]:
    intervention_count = 0
    fn_bucket_counts: dict[str, int] = {}
    for case in cases:
        if case["review_bucket"] == "FP":
            intervention_count += len(
                build_prefix_intervention_matrix(
                    source_object=case["source_object"],
                    gt_next=case["gt_next"],
                )
            )
        else:
            bucket = label_fn_bucket(
                recovered_by_sampling=bool(case["recovered_by_sampling"]),
                recovered_by_clean_prefix=bool(case["recovered_by_clean_prefix"]),
                recovered_by_stop_probe=bool(case["recovered_by_stop_probe"]),
                has_teacher_forced_support=bool(case["has_teacher_forced_support"]),
                ambiguity_flag=bool(case["ambiguity_flag"]),
            )
            fn_bucket_counts[bucket] = fn_bucket_counts.get(bucket, 0) + 1
    return {
        "intervention_count": intervention_count,
        "fn_bucket_counts": fn_bucket_counts,
    }


def run_review_and_report_stage(
    *,
    output_dir: Path,
    shortlist: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    review_rows = build_review_queue_rows(shortlist=shortlist)
    write_report_bundle(
        output_dir=output_dir,
        summary=summary,
        review_rows=review_rows,
    )

def _run_study_impl(
    config_path: Path,
    *,
    stage_override: str | None = None,
    model_alias: str | None = None,
    branch_name: str | None = None,
) -> dict[str, object]:
    cfg = load_study_config(config_path)
    run_dir = Path(cfg.run.output_dir) / cfg.run.name
    run_dir.mkdir(parents=True, exist_ok=True)
    requested_stages = (
        [stage_override] if stage_override is not None else list(cfg.run.stages)
    )

    stage_manifest = {
        "run_name": cfg.run.name,
        "stages": requested_stages,
        "models": [
            cfg.models.base_only.alias,
            cfg.models.base_plus_adapter.alias,
        ],
        "requested_model_alias": model_alias,
        "requested_branch_name": branch_name,
    }
    summary = {
        "run_name": cfg.run.name,
        "stage_count": len(requested_stages),
        "requested_stage": stage_override,
        "requested_model_alias": model_alias,
        "requested_branch_name": branch_name,
        "review_budgets": {
            "fp": cfg.review.fp_budget,
            "fn": cfg.review.fn_budget,
        },
    }
    case_bank_rows: list[dict[str, object]] = []

    _write_json(run_dir / "stage_manifest.json", stage_manifest)
    _write_json(run_dir / "summary.json", summary)
    _write_jsonl(run_dir / "case_bank.jsonl", case_bank_rows)
    return {
        "run_dir": str(run_dir),
        "stage_manifest_path": str(run_dir / "stage_manifest.json"),
        "summary_path": str(run_dir / "summary.json"),
        "case_bank_path": str(run_dir / "case_bank.jsonl"),
        "requested_stage": stage_override,
    }


def run_study(
    config_path: Path,
    *,
    stage_override: str | None = None,
    model_alias: str | None = None,
    branch_name: str | None = None,
) -> dict[str, object]:
    return _run_study_impl(
        config_path,
        stage_override=stage_override,
        model_alias=model_alias,
        branch_name=branch_name,
    )
