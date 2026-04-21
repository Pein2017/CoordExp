from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.raw_text_coord_continuity_probe import (
    _object_to_norm1000,
    render_pretty_inline_assistant_text,
    select_self_prefix_duplicate_anchor,
)
from src.analysis.raw_text_coordinate_continuation_scoring import (
    score_candidate_continuations_batch,
)
from src.analysis.raw_text_coordinate_exploratory import (
    render_model_native_prefix_assistant_text,
    select_duplicate_burst_cases,
)
from src.analysis.raw_text_coordinate_preburst_probe import (
    build_preburst_variants,
    summarize_preburst_margin_rows,
)
from src.analysis.unmatched_proposal_verifier import TeacherForcedScorer
from src.common.paths import resolve_image_path_best_effort, resolve_image_path_strict


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str


@dataclass(frozen=True)
class InputConfig:
    case_bank_jsonl: str
    max_cases_per_model: int
    case_serializer_surface: str
    scoring_surface: str
    candidate_labels: tuple[str, ...]


@dataclass(frozen=True)
class ModelConfig:
    alias: str
    path: str
    prompt_variant: str
    object_field_order: str


@dataclass(frozen=True)
class StudyModels:
    base_only: ModelConfig
    base_plus_adapter: ModelConfig


@dataclass(frozen=True)
class ScoringConfig:
    device: str
    attn_implementation: str


@dataclass(frozen=True)
class Config:
    run: RunConfig
    input: InputConfig
    models: StudyModels
    scoring: ScoringConfig


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("preburst margin config root must be a mapping")
    return payload


def _shared_repo_root(anchor: Path) -> Path:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=anchor.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        common_dir = Path(completed.stdout.strip())
        return common_dir.parent if common_dir.name == ".git" else common_dir
    except (OSError, subprocess.CalledProcessError, ValueError):
        return anchor.parent


def _resolve_path(raw_path: str, *, anchor: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidate = anchor.parent / path
    if candidate.exists():
        return candidate
    return _shared_repo_root(anchor) / path


def _parse_model(raw: dict[str, object], key: str) -> ModelConfig:
    model_raw = raw[key]
    if not isinstance(model_raw, dict):
        raise TypeError(f"models.{key} must be a mapping")
    return ModelConfig(
        alias=str(model_raw["alias"]),
        path=str(model_raw["path"]),
        prompt_variant=str(model_raw.get("prompt_variant", "coco_80")),
        object_field_order=str(model_raw.get("object_field_order", "desc_first")),
    )


def load_config(config_path: Path) -> Config:
    raw = _load_yaml(config_path)
    run_raw = raw["run"]
    input_raw = raw["input"]
    models_raw = raw["models"]
    scoring_raw = raw["scoring"]
    if not isinstance(run_raw, dict) or not isinstance(input_raw, dict):
        raise TypeError("run/input must be mappings")
    if not isinstance(models_raw, dict) or not isinstance(scoring_raw, dict):
        raise TypeError("models/scoring must be mappings")
    return Config(
        run=RunConfig(
            name=str(run_raw["name"]),
            output_dir=str(run_raw["output_dir"]),
        ),
        input=InputConfig(
            case_bank_jsonl=str(input_raw["case_bank_jsonl"]),
            max_cases_per_model=int(input_raw.get("max_cases_per_model", 8)),
            case_serializer_surface=str(
                input_raw.get(
                    "case_serializer_surface",
                    input_raw.get("serializer_surface", "model_native"),
                )
            ),
            scoring_surface=str(
                input_raw.get(
                    "scoring_surface",
                    input_raw.get(
                        "case_serializer_surface",
                        input_raw.get("serializer_surface", "model_native"),
                    ),
                )
            ),
            candidate_labels=tuple(
                str(label)
                for label in input_raw.get(
                    "candidate_labels",
                    ("gt_next", "exact_duplicate", "predicted_object"),
                )
            ),
        ),
        models=StudyModels(
            base_only=_parse_model(models_raw, "base_only"),
            base_plus_adapter=_parse_model(models_raw, "base_plus_adapter"),
        ),
        scoring=ScoringConfig(
            device=str(scoring_raw.get("device", "cuda:0")),
            attn_implementation=str(scoring_raw.get("attn_implementation", "auto")),
        ),
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise TypeError(f"expected object rows in {path}")
            rows.append(payload)
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _resolve_case_image(source_path: Path, source_record: dict[str, object]) -> Image.Image:
    repo_root = _shared_repo_root(source_path)
    root_image_dir = repo_root / "public_data/coco/rescale_32_1024_bbox_max60"
    raw_image = str(source_record.get("image") or "")
    resolved = resolve_image_path_strict(
        raw_image,
        jsonl_dir=source_path.parent,
        root_image_dir=root_image_dir,
    )
    if resolved is None:
        resolved = resolve_image_path_best_effort(
            raw_image,
            jsonl_dir=source_path.parent,
            root_image_dir=root_image_dir,
        )
    return Image.open(resolved).convert("RGB")


def _resolve_pred_token_trace_path(source_path: Path) -> Path:
    trace_path = source_path.parent / "pred_token_trace.jsonl"
    if not trace_path.exists():
        raise FileNotFoundError(f"missing pred token trace next to {source_path}")
    return trace_path


def _render_assistant_text(
    *,
    scoring_surface: str,
    objects: list[dict[str, object]],
    object_field_order: str,
    native_assistant_text: object | None,
) -> str:
    if scoring_surface == "pretty_inline":
        return render_pretty_inline_assistant_text(
            {"objects": objects},
            object_field_order=object_field_order,
        )
    if scoring_surface == "model_native":
        return render_model_native_prefix_assistant_text(
            objects=objects,
            native_assistant_text=native_assistant_text,
        )
    raise ValueError(f"unsupported scoring surface: {scoring_surface}")


def _candidate_object_map(
    *,
    gt_next: dict[str, object],
    duplicate_object: dict[str, object],
    predicted_object: dict[str, object],
) -> dict[str, dict[str, object]]:
    return {
        "gt_next": dict(gt_next),
        "exact_duplicate": dict(duplicate_object),
        "predicted_object": dict(predicted_object),
    }


def _score_preburst_cases(
    *,
    cfg: Config,
    candidate_cases: list[dict[str, object]],
    config_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    source_cache: dict[Path, list[dict[str, object]]] = {}
    trace_cache: dict[Path, list[dict[str, object]]] = {}
    candidate_score_rows: list[dict[str, object]] = []
    margin_rows: list[dict[str, object]] = []
    selected_cases: list[dict[str, object]] = []
    selected_case_ids: set[tuple[str, str]] = set()
    selected_counts: dict[str, int] = {}
    model_cfgs = (cfg.models.base_only, cfg.models.base_plus_adapter)
    for model_cfg in model_cfgs:
        scorer = TeacherForcedScorer(
            checkpoint_path=_resolve_path(model_cfg.path, anchor=config_path),
            device=cfg.scoring.device,
            attn_implementation=cfg.scoring.attn_implementation,
            coord_mode="norm1000_text",
        )
        try:
            for case_row in candidate_cases:
                if str(case_row.get("model_alias") or "") != model_cfg.alias:
                    continue
                if selected_counts.get(model_cfg.alias, 0) >= cfg.input.max_cases_per_model:
                    continue
                source_path = Path(str(case_row["source_gt_vs_pred_jsonl"]))
                if source_path not in source_cache:
                    source_cache[source_path] = _read_jsonl(source_path)
                source_record = source_cache[source_path][int(case_row["line_idx"])]
                anchor = select_self_prefix_duplicate_anchor(source_record)
                if anchor is None or anchor.get("gt_next") is None:
                    continue
                width = int(source_record.get("width") or 0)
                height = int(source_record.get("height") or 0)
                prefix_objects: list[dict[str, object]] = []
                for pred_obj in list(source_record.get("pred") or [])[: int(anchor["object_index"])]:
                    norm_obj = _object_to_norm1000(dict(pred_obj), width=width, height=height)
                    if norm_obj is not None:
                        prefix_objects.append(norm_obj)
                predicted_object = _object_to_norm1000(
                    dict(anchor["pred_object"]),
                    width=width,
                    height=height,
                )
                gt_next = _object_to_norm1000(
                    dict(anchor["gt_next"]),
                    width=width,
                    height=height,
                )
                if predicted_object is None or gt_next is None:
                    continue
                native_assistant_text: object | None = None
                if cfg.input.scoring_surface == "model_native":
                    trace_path = _resolve_pred_token_trace_path(source_path)
                    if trace_path not in trace_cache:
                        trace_cache[trace_path] = _read_jsonl(trace_path)
                    native_assistant_text = trace_cache[trace_path][int(case_row["line_idx"])].get(
                        "generated_token_text"
                    )
                image = _resolve_case_image(source_path, source_record)
                try:
                    variants = build_preburst_variants(
                        prefix_objects=prefix_objects,
                        source_object_index=int(anchor["source_object_index"]),
                        gt_next=gt_next,
                    )
                    for variant in variants:
                        prefix_variant = [
                            {
                                "desc": str(obj.get("desc") or ""),
                                "bbox_2d": [int(value) for value in list(obj["bbox_2d"])],
                            }
                            for obj in list(variant["prefix_objects"])
                        ]
                        baseline_assistant_text = _render_assistant_text(
                            scoring_surface=cfg.input.scoring_surface,
                            objects=prefix_variant,
                            object_field_order=model_cfg.object_field_order,
                            native_assistant_text=native_assistant_text,
                        )
                        candidate_object_map = _candidate_object_map(
                            gt_next=gt_next,
                            duplicate_object=dict(variant["duplicate_object"]),
                            predicted_object=predicted_object,
                        )
                        batch_rows = [
                            {
                                "candidate_label": label,
                                "candidate_assistant_text": _render_assistant_text(
                                    scoring_surface=cfg.input.scoring_surface,
                                    objects=prefix_variant + [candidate_object_map[label]],
                                    object_field_order=model_cfg.object_field_order,
                                    native_assistant_text=native_assistant_text,
                                ),
                            }
                            for label in cfg.input.candidate_labels
                            if label in candidate_object_map
                        ]
                        scored_rows = score_candidate_continuations_batch(
                            scorer=scorer,
                            image=image,
                            baseline_assistant_text=baseline_assistant_text,
                            candidate_rows=batch_rows,
                            prompt_variant=model_cfg.prompt_variant,
                            object_field_order=model_cfg.object_field_order,
                        )
                        scored_by_label: dict[str, dict[str, object]] = {}
                        for scored in scored_rows:
                            candidate_label = str(scored["candidate_label"])
                            scored_by_label[candidate_label] = dict(scored)
                            candidate_score_rows.append(
                                {
                                    "case_id": case_row["case_uid"],
                                    "case_uid": case_row["case_uid"],
                                    "model_alias": model_cfg.alias,
                                    "image_id": case_row["image_id"],
                                    "line_idx": case_row["line_idx"],
                                    "source_gt_vs_pred_jsonl": str(source_path),
                                    "case_serializer_surface": str(
                                        case_row.get("serializer_surface") or ""
                                    ),
                                    "scoring_surface": cfg.input.scoring_surface,
                                    "object_index": int(anchor["object_index"]),
                                    "source_object_index": int(anchor["source_object_index"]),
                                    "variant_label": str(variant["variant_label"]),
                                    "candidate_label": candidate_label,
                                    "token_count": int(scored["count"]),
                                    "sum_logprob": float(scored["sum_logprob"]),
                                    "mean_logprob": float(scored["mean_logprob"]),
                                    "candidate_token_span": len(
                                        list(scored["assistant_relative_positions"])
                                    ),
                                    "prefix_object_count": len(prefix_variant),
                                    "duplicate_bbox": list(
                                        candidate_object_map["exact_duplicate"]["bbox_2d"]
                                    ),
                                    "gt_next_bbox": list(gt_next["bbox_2d"]),
                                    "predicted_bbox": list(predicted_object["bbox_2d"]),
                                }
                            )
                        gt_scored = scored_by_label.get("gt_next")
                        dup_scored = scored_by_label.get("exact_duplicate")
                        pred_scored = scored_by_label.get("predicted_object")
                        if gt_scored is None or dup_scored is None:
                            continue
                        margin_rows.append(
                            {
                                "case_id": case_row["case_uid"],
                                "case_uid": case_row["case_uid"],
                                "model_alias": model_cfg.alias,
                                "image_id": case_row["image_id"],
                                "line_idx": case_row["line_idx"],
                                "source_gt_vs_pred_jsonl": str(source_path),
                                "case_serializer_surface": str(
                                    case_row.get("serializer_surface") or ""
                                ),
                                "scoring_surface": cfg.input.scoring_surface,
                                "object_index": int(anchor["object_index"]),
                                "source_object_index": int(anchor["source_object_index"]),
                                "variant_label": str(variant["variant_label"]),
                                "prefix_object_count": len(prefix_variant),
                                "gt_next_sum_logprob": float(gt_scored["sum_logprob"]),
                                "gt_next_mean_logprob": float(gt_scored["mean_logprob"]),
                                "exact_duplicate_sum_logprob": float(
                                    dup_scored["sum_logprob"]
                                ),
                                "exact_duplicate_mean_logprob": float(
                                    dup_scored["mean_logprob"]
                                ),
                                "margin_sum_logprob": float(gt_scored["sum_logprob"])
                                - float(dup_scored["sum_logprob"]),
                                "margin_mean_logprob": float(gt_scored["mean_logprob"])
                                - float(dup_scored["mean_logprob"]),
                                "gt_next_token_count": int(gt_scored["count"]),
                                "exact_duplicate_token_count": int(dup_scored["count"]),
                                "token_count_gap_gt_minus_duplicate": int(gt_scored["count"])
                                - int(dup_scored["count"]),
                                "predicted_object_sum_logprob": (
                                    float(pred_scored["sum_logprob"])
                                    if pred_scored is not None
                                    else None
                                ),
                                "predicted_object_mean_logprob": (
                                    float(pred_scored["mean_logprob"])
                                    if pred_scored is not None
                                    else None
                                ),
                            }
                        )
                        case_key = (model_cfg.alias, str(case_row["case_uid"]))
                        if case_key not in selected_case_ids:
                            selected_case_ids.add(case_key)
                            selected_counts[model_cfg.alias] = (
                                selected_counts.get(model_cfg.alias, 0) + 1
                            )
                            selected_cases.append(dict(case_row))
                finally:
                    image.close()
        finally:
            del scorer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return (
        selected_cases,
        candidate_score_rows,
        margin_rows,
        summarize_preburst_margin_rows(margin_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    case_bank_path = _resolve_path(cfg.input.case_bank_jsonl, anchor=config_path)
    output_dir = _resolve_path(cfg.run.output_dir, anchor=config_path) / cfg.run.name
    case_bank_rows = _read_jsonl(case_bank_path)
    candidate_cases = select_duplicate_burst_cases(
        rows=case_bank_rows,
        max_cases_per_model=max(1, int(cfg.input.max_cases_per_model) * 4),
        serializer_surface=cfg.input.case_serializer_surface,
    )
    selected_cases, candidate_score_rows, margin_rows, summary = _score_preburst_cases(
        cfg=cfg,
        candidate_cases=candidate_cases,
        config_path=config_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "selected_cases.jsonl", selected_cases)
    _write_jsonl(output_dir / "per_candidate_scores.jsonl", candidate_score_rows)
    _write_jsonl(output_dir / "per_case_margins.jsonl", margin_rows)
    _write_json(
        output_dir / "summary.json",
        {
            "candidate_case_count": len(candidate_cases),
            "selected_case_count": len(selected_cases),
            "candidate_labels": list(cfg.input.candidate_labels),
            "case_serializer_surface": cfg.input.case_serializer_surface,
            "scoring_surface": cfg.input.scoring_surface,
            "selected_cases_path": str(output_dir / "selected_cases.jsonl"),
            "per_candidate_scores_path": str(output_dir / "per_candidate_scores.jsonl"),
            "per_case_margins_path": str(output_dir / "per_case_margins.jsonl"),
            **summary,
        },
    )
    print(
        json.dumps(
            {
                "run_dir": str(output_dir),
                "candidate_case_count": len(candidate_cases),
                "selected_case_count": len(selected_cases),
                "margin_rows": len(margin_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
