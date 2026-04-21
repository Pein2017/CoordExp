from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any, Sequence

import yaml

from src.analysis.raw_text_coordinate_case_bank import (
    build_case_bank_rows,
    freeze_review_shortlist,
)
from src.analysis.raw_text_coordinate_behavior import summarize_confirmatory_records
from src.analysis.raw_text_coord_continuity_probe import (
    _load_tokenizer_for_audit,
    _resolve_audit_model_info,
    run_phase0_audit,
    select_self_prefix_duplicate_anchor,
)
from src.analysis.duplication_collapse_analysis import mine_duplicate_like_rows
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


_GT_VS_PRED_CANDIDATE_PATHS = {
    "base_only": (
        "output/infer/coco1024_val200_lvis_proxy_base_coordexp_rawtext_probe/gt_vs_pred.jsonl",
    ),
    "base_plus_adapter": (
        "output/infer/coco1024_val200_lvis_proxy_rawtext_xyxy_v1_ckpt552_fanout8/gt_vs_pred.jsonl",
        "output/infer/coco1024_val200_lvis_proxy_rawtext_xyxy_v1_ckpt552/gt_vs_pred.jsonl",
    ),
}


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


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _normalize_desc(value: object) -> str:
    return str(value or "").strip().lower()


def _extract_bbox_xyxy(obj: dict[str, object]) -> list[float] | None:
    raw_bbox = obj.get("bbox_2d")
    if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
        return [float(value) for value in raw_bbox]
    raw_points = obj.get("points")
    if isinstance(raw_points, list) and len(raw_points) == 4:
        return [float(value) for value in raw_points]
    return None


def _bbox_iou(left: list[float], right: list[float]) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - inter_area
    return inter_area / union if union > 0.0 else 0.0


def _mine_labeled_fn_rows_from_gt_vs_pred(
    *,
    gt_vs_pred_path: Path,
    max_cases: int,
) -> list[dict[str, object]]:
    rows = _read_jsonl(gt_vs_pred_path)
    mined: list[dict[str, object]] = []
    for line_idx, row in enumerate(rows):
        preds = list(row.get("pred") or [])
        gts = list(row.get("gt") or [])
        used_pred_indexes: set[int] = set()
        for gt_idx, gt_obj in enumerate(gts):
            gt_bbox = _extract_bbox_xyxy(gt_obj)
            gt_desc = _normalize_desc(gt_obj.get("desc"))
            if gt_bbox is None or not gt_desc:
                continue
            best_pred_idx = None
            best_iou = 0.0
            for pred_idx, pred_obj in enumerate(preds):
                if pred_idx in used_pred_indexes:
                    continue
                pred_bbox = _extract_bbox_xyxy(pred_obj)
                pred_desc = _normalize_desc(pred_obj.get("desc"))
                if pred_bbox is None or pred_desc != gt_desc:
                    continue
                iou = _bbox_iou(gt_bbox, pred_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx
            if best_pred_idx is not None and best_iou >= 0.5:
                used_pred_indexes.add(best_pred_idx)
                continue
            mined.append(
                {
                    "image_id": row.get("image_id"),
                    "line_idx": line_idx,
                    "record_idx": line_idx,
                    "gt_idx": gt_idx,
                    "selection_rank": len(mined) + 1,
                }
            )
            if len(mined) >= max_cases:
                return mined
    return mined


def _shared_repo_root(anchor: Path) -> Path:
    try:
        resolved_anchor = anchor.resolve()
        cwd = resolved_anchor if resolved_anchor.is_dir() else resolved_anchor.parent
        completed = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        common_dir = Path(completed.stdout.strip())
        return common_dir.parent if common_dir.name == ".git" else common_dir
    except (OSError, subprocess.CalledProcessError, ValueError):
        resolved_anchor = anchor.resolve()
        return resolved_anchor if resolved_anchor.is_dir() else resolved_anchor.parent


def _resolve_repo_path(raw_path: str, *, anchor: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    local_candidate = anchor / path
    if local_candidate.exists():
        return local_candidate
    return _shared_repo_root(anchor) / path


def _effective_model_path(model_cfg: ModelObjectConfig) -> Path:
    if model_cfg.adapter_path:
        return Path(model_cfg.adapter_path)
    return Path(model_cfg.base_path)


def _resolve_best_gt_vs_pred_source(
    *,
    model_alias: str,
    anchor: Path,
) -> dict[str, object] | None:
    candidates = _GT_VS_PRED_CANDIDATE_PATHS.get(model_alias, ())
    ranked_candidates: list[dict[str, object]] = []
    for raw_path in candidates:
        resolved_path = _resolve_repo_path(raw_path, anchor=anchor)
        if not resolved_path.exists():
            continue
        ranked_candidates.append(
            {
                "path": resolved_path,
                "row_count": _count_jsonl_rows(resolved_path),
            }
        )
    if not ranked_candidates:
        return None
    best_candidate = max(
        ranked_candidates,
        key=lambda candidate: (
            int(candidate["row_count"]),
            -candidates.index(
                next(
                    raw_path
                    for raw_path in candidates
                    if _resolve_repo_path(raw_path, anchor=anchor) == candidate["path"]
                )
            ),
        ),
    )
    return {
        "path": best_candidate["path"],
        "row_count": int(best_candidate["row_count"]),
        "candidate_count": len(ranked_candidates),
    }


def _run_contract_audit_stage(cfg: StudyConfig) -> dict[str, object]:
    audit_rows: list[dict[str, object]] = []
    for model_cfg in (cfg.models.base_only, cfg.models.base_plus_adapter):
        model_path = _effective_model_path(model_cfg)
        model_info = _resolve_audit_model_info(model_path)
        if model_path.exists():
            tokenizer = _load_tokenizer_for_audit(model_path)
            scorer_stub = type("_ScorerStub", (), {"tokenizer": tokenizer})()
            phase0 = run_phase0_audit(scorer_stub)
        else:
            phase0 = {
                "numbers": [],
                "surface_forms": {},
                "skipped": True,
                "skip_reason": f"missing model path: {model_path}",
            }
        audit_rows.append(
            {
                "alias": model_cfg.alias,
                "model_info": model_info,
                "phase0_audit": phase0,
            }
        )
    return {
        "models": audit_rows,
    }


def _serialize_case_rows(rows: Sequence[object]) -> list[dict[str, object]]:
    return [
        {
            "case_uid": row.case_uid,
            "model_alias": row.model_alias,
            "image_id": row.image_id,
            "line_idx": row.line_idx,
            "record_idx": row.record_idx,
            "bucket": row.bucket,
            "review_bucket": row.review_bucket,
            "source_object_index": row.source_object_index,
            "onset_object_index": row.onset_object_index,
            "gt_idx": row.gt_idx,
            "selection_rank": row.selection_rank,
            "serializer_surface": row.serializer_surface,
        }
        for row in rows
    ]


def _run_case_bank_stage(
    cfg: StudyConfig,
    *,
    anchor: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, object]]]:
    duplicate_rows: list[dict[str, object]] = []
    fn_rows: list[dict[str, object]] = []
    source_artifacts: dict[str, dict[str, object]] = {}
    for model_cfg in (cfg.models.base_only, cfg.models.base_plus_adapter):
        source_info = _resolve_best_gt_vs_pred_source(
            model_alias=model_cfg.alias,
            anchor=anchor,
        )
        if source_info is None:
            continue
        gt_vs_pred_path = Path(source_info["path"])
        source_artifacts[model_cfg.alias] = {
            "path": str(gt_vs_pred_path),
            "row_count": int(source_info["row_count"]),
        }
        mined_rows = mine_duplicate_like_rows(
            gt_vs_pred_path=gt_vs_pred_path,
            max_cases=max(cfg.review.fp_budget, 8),
            min_pred_objects=2,
            min_duplicate_pairs=1,
            duplicate_iou_threshold=0.7,
        )
        source_rows = _read_jsonl(gt_vs_pred_path)
        mined_fn_rows = _mine_labeled_fn_rows_from_gt_vs_pred(
            gt_vs_pred_path=gt_vs_pred_path,
            max_cases=max(cfg.review.fn_budget, 4),
        )
        for selection_rank, mined_row in enumerate(mined_rows, start=1):
            line_idx = int(mined_row["line_idx"])
            if line_idx >= len(source_rows):
                continue
            anchor_row = select_self_prefix_duplicate_anchor(source_rows[line_idx])
            if anchor_row is None:
                continue
            for serializer_surface in model_cfg.serializer_surfaces:
                duplicate_rows.append(
                    {
                        "model_alias": model_cfg.alias,
                        "image_id": mined_row.get("image_id"),
                        "line_idx": line_idx,
                        "record_idx": line_idx,
                        "source_object_index": int(anchor_row["source_object_index"]),
                        "onset_object_index": int(anchor_row["object_index"]),
                        "selection_rank": selection_rank,
                        "serializer_surface": serializer_surface,
                    }
                )
        for mined_fn_row in mined_fn_rows:
            for serializer_surface in model_cfg.serializer_surfaces:
                fn_rows.append(
                    {
                        "model_alias": model_cfg.alias,
                        "image_id": mined_fn_row.get("image_id"),
                        "line_idx": int(mined_fn_row["line_idx"]),
                        "record_idx": int(mined_fn_row["record_idx"]),
                        "gt_idx": int(mined_fn_row["gt_idx"]),
                        "selection_rank": int(mined_fn_row["selection_rank"]),
                        "serializer_surface": serializer_surface,
                    }
                )
    case_rows = build_case_bank_rows(
        duplicate_rows=duplicate_rows,
        fn_rows=fn_rows,
    )
    shortlist = freeze_review_shortlist(
        case_rows,
        fp_budget=cfg.review.fp_budget,
        fn_budget=cfg.review.fn_budget,
    )
    return (
        _serialize_case_rows(case_rows),
        _serialize_case_rows(shortlist),
        source_artifacts,
    )


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
    repo_root = _shared_repo_root(config_path)
    output_root = _resolve_repo_path(cfg.run.output_dir, anchor=repo_root)
    run_dir = output_root / cfg.run.name
    run_dir.mkdir(parents=True, exist_ok=True)
    requested_stages = (
        [stage_override] if stage_override is not None else list(cfg.run.stages)
    )
    base_only_cfg = ModelObjectConfig(
        alias=cfg.models.base_only.alias,
        base_path=str(_resolve_repo_path(cfg.models.base_only.base_path, anchor=repo_root)),
        adapter_path=(
            str(_resolve_repo_path(cfg.models.base_only.adapter_path, anchor=repo_root))
            if cfg.models.base_only.adapter_path
            else None
        ),
        prompt_variant=cfg.models.base_only.prompt_variant,
        object_field_order=cfg.models.base_only.object_field_order,
        serializer_surfaces=cfg.models.base_only.serializer_surfaces,
    )
    base_plus_adapter_cfg = ModelObjectConfig(
        alias=cfg.models.base_plus_adapter.alias,
        base_path=str(
            _resolve_repo_path(cfg.models.base_plus_adapter.base_path, anchor=repo_root)
        ),
        adapter_path=(
            str(
                _resolve_repo_path(
                    cfg.models.base_plus_adapter.adapter_path,
                    anchor=repo_root,
                )
            )
            if cfg.models.base_plus_adapter.adapter_path
            else None
        ),
        prompt_variant=cfg.models.base_plus_adapter.prompt_variant,
        object_field_order=cfg.models.base_plus_adapter.object_field_order,
        serializer_surfaces=cfg.models.base_plus_adapter.serializer_surfaces,
    )
    resolved_cfg = StudyConfig(
        run=cfg.run,
        models=StudyModels(
            base_only=base_only_cfg,
            base_plus_adapter=base_plus_adapter_cfg,
        ),
        dataset=cfg.dataset,
        execution=cfg.execution,
        review=cfg.review,
    )

    stage_manifest = {
        "run_name": cfg.run.name,
        "stages": requested_stages,
        "models": [
            resolved_cfg.models.base_only.alias,
            resolved_cfg.models.base_plus_adapter.alias,
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
    shortlist_rows: list[dict[str, object]] = []
    source_artifacts: dict[str, dict[str, object]] = {}
    contract_audit = (
        _run_contract_audit_stage(resolved_cfg) if "contract" in requested_stages else {}
    )
    if "case_bank" in requested_stages:
        case_bank_rows, shortlist_rows, source_artifacts = _run_case_bank_stage(
            resolved_cfg,
            anchor=repo_root,
        )
        summary["case_bank_counts"] = {
            f"{bucket}:{model_alias}": count
            for (bucket, model_alias), count in Counter(
                (row["review_bucket"], row["model_alias"]) for row in case_bank_rows
            ).items()
        }
        summary["shortlist_counts"] = {
            f"{bucket}:{model_alias}": count
            for (bucket, model_alias), count in Counter(
                (row["review_bucket"], row["model_alias"]) for row in shortlist_rows
            ).items()
        }
        summary["source_artifacts"] = source_artifacts

    _write_json(run_dir / "stage_manifest.json", stage_manifest)
    _write_json(run_dir / "summary.json", summary)
    _write_jsonl(run_dir / "case_bank.jsonl", case_bank_rows)
    if contract_audit:
        _write_json(run_dir / "contract_audit.json", contract_audit)
    if shortlist_rows:
        _write_jsonl(run_dir / "shortlist.jsonl", shortlist_rows)
    return {
        "run_dir": str(run_dir),
        "stage_manifest_path": str(run_dir / "stage_manifest.json"),
        "summary_path": str(run_dir / "summary.json"),
        "case_bank_path": str(run_dir / "case_bank.jsonl"),
        "contract_audit_path": str(run_dir / "contract_audit.json"),
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
