from __future__ import annotations

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
