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
    _SLOT_TO_INDEX,
    _object_to_norm1000,
    build_candidate_values_around,
    render_pretty_inline_assistant_text,
    select_self_prefix_duplicate_anchor,
    summarize_bad_basin_coordinate_records,
)
from src.analysis.raw_text_coord_continuity_scoring import (
    score_candidate_coordinate_sequence,
    score_candidate_coordinate_sequences_batch,
)
from src.analysis.raw_text_coordinate_exploratory import (
    render_model_native_prefix_assistant_text,
    select_duplicate_burst_cases,
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
    slots: tuple[str, ...]


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
    candidate_radius: int


@dataclass(frozen=True)
class Config:
    run: RunConfig
    input: InputConfig
    models: StudyModels
    scoring: ScoringConfig


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("duplicate burst config root must be a mapping")
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
            slots=tuple(str(slot) for slot in input_raw.get("slots", ("x1", "y1"))),
        ),
        models=StudyModels(
            base_only=_parse_model(models_raw, "base_only"),
            base_plus_adapter=_parse_model(models_raw, "base_plus_adapter"),
        ),
        scoring=ScoringConfig(
            device=str(scoring_raw.get("device", "cuda:0")),
            attn_implementation=str(scoring_raw.get("attn_implementation", "auto")),
            candidate_radius=int(scoring_raw.get("candidate_radius", 8)),
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


def _score_duplicate_burst_cases(
    *,
    cfg: Config,
    selected_cases: list[dict[str, object]],
    config_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    source_cache: dict[Path, list[dict[str, object]]] = {}
    trace_cache: dict[Path, list[dict[str, object]]] = {}
    all_rows: list[dict[str, object]] = []
    model_cfgs = (cfg.models.base_only, cfg.models.base_plus_adapter)
    for model_cfg in model_cfgs:
        scorer = TeacherForcedScorer(
            checkpoint_path=_resolve_path(model_cfg.path, anchor=config_path),
            device=cfg.scoring.device,
            attn_implementation=cfg.scoring.attn_implementation,
            coord_mode="norm1000_text",
        )
        try:
            for case_row in selected_cases:
                if str(case_row.get("model_alias") or "") != model_cfg.alias:
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
                pred_object = _object_to_norm1000(
                    dict(anchor["pred_object"]),
                    width=width,
                    height=height,
                )
                gt_next = _object_to_norm1000(
                    dict(anchor["gt_next"]),
                    width=width,
                    height=height,
                )
                source_object = _object_to_norm1000(
                    dict(anchor["source_object"]),
                    width=width,
                    height=height,
                )
                if pred_object is None or gt_next is None or source_object is None:
                    continue
                assistant_objects = prefix_objects + [pred_object]
                if cfg.input.scoring_surface == "pretty_inline":
                    assistant_text = render_pretty_inline_assistant_text(
                        {"objects": assistant_objects},
                        object_field_order=model_cfg.object_field_order,
                    )
                elif cfg.input.scoring_surface == "model_native":
                    trace_path = _resolve_pred_token_trace_path(source_path)
                    if trace_path not in trace_cache:
                        trace_cache[trace_path] = _read_jsonl(trace_path)
                    trace_row = trace_cache[trace_path][int(case_row["line_idx"])]
                    assistant_text = render_model_native_prefix_assistant_text(
                        objects=assistant_objects,
                        native_assistant_text=trace_row.get("generated_token_text"),
                    )
                else:
                    raise ValueError(
                        f"unsupported scoring surface: {cfg.input.scoring_surface}"
                    )
                object_index = len(prefix_objects)
                pred_bbox = list(pred_object["bbox_2d"])
                gt_bbox = list(gt_next["bbox_2d"])
                source_bbox = list(source_object["bbox_2d"])
                image = _resolve_case_image(source_path, source_record)
                try:
                    for slot in cfg.input.slots:
                        slot_idx = int(_SLOT_TO_INDEX[slot])
                        pred_value = int(pred_bbox[slot_idx])
                        gt_value = int(gt_bbox[slot_idx])
                        candidate_values = sorted(
                            set(
                                build_candidate_values_around(
                                    pred_value,
                                    radius=cfg.scoring.candidate_radius,
                                )
                                + build_candidate_values_around(
                                    gt_value,
                                    radius=cfg.scoring.candidate_radius,
                                )
                            )
                        )
                        try:
                            scored_rows = score_candidate_coordinate_sequences_batch(
                                scorer=scorer,
                                image=image,
                                assistant_text=assistant_text,
                                slot=slot,
                                original_bbox=pred_bbox,
                                candidate_values=candidate_values,
                                prompt_variant=model_cfg.prompt_variant,
                                object_field_order=model_cfg.object_field_order,
                                object_index=object_index,
                            )
                            for scored in scored_rows:
                                candidate_value = int(scored["candidate_value"])
                                all_rows.append(
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
                                        "object_index": object_index,
                                        "source_object_index": int(anchor["source_object_index"]),
                                        "slot": slot,
                                        "pred_value": pred_value,
                                        "gt_value": gt_value,
                                        "previous_value": int(source_bbox[slot_idx]),
                                        "candidate_value": candidate_value,
                                        "numeric_distance_to_pred": abs(candidate_value - pred_value),
                                        "numeric_distance_to_gt": abs(candidate_value - gt_value),
                                        "score": float(scored["sum_logprob"]),
                                        "sum_logprob": float(scored["sum_logprob"]),
                                        "mean_logprob": float(scored["mean_logprob"]),
                                        "token_count": int(scored["count"]),
                                        "candidate_token_span": len(
                                            list(scored["absolute_positions"])
                                        ),
                                        "scoring_status": "ok",
                                        "failure_reason": None,
                                    }
                                )
                            continue
                        except (RuntimeError, ValueError):
                            pass
                        for candidate_value in candidate_values:
                            try:
                                scored = score_candidate_coordinate_sequence(
                                    scorer=scorer,
                                    image=image,
                                    assistant_text=assistant_text,
                                    slot=slot,
                                    original_bbox=pred_bbox,
                                    candidate_value=int(candidate_value),
                                    prompt_variant=model_cfg.prompt_variant,
                                    object_field_order=model_cfg.object_field_order,
                                    object_index=object_index,
                                )
                                all_rows.append(
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
                                        "object_index": object_index,
                                        "source_object_index": int(anchor["source_object_index"]),
                                        "slot": slot,
                                        "pred_value": pred_value,
                                        "gt_value": gt_value,
                                        "previous_value": int(source_bbox[slot_idx]),
                                        "candidate_value": int(candidate_value),
                                        "numeric_distance_to_pred": abs(int(candidate_value) - pred_value),
                                        "numeric_distance_to_gt": abs(int(candidate_value) - gt_value),
                                        "score": float(scored["sum_logprob"]),
                                        "sum_logprob": float(scored["sum_logprob"]),
                                        "mean_logprob": float(scored["mean_logprob"]),
                                        "token_count": int(scored["count"]),
                                        "candidate_token_span": len(
                                            list(scored["absolute_positions"])
                                        ),
                                        "scoring_status": "ok",
                                        "failure_reason": None,
                                    }
                                )
                            except (RuntimeError, ValueError) as exc:
                                all_rows.append(
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
                                        "object_index": object_index,
                                        "source_object_index": int(anchor["source_object_index"]),
                                        "slot": slot,
                                        "pred_value": pred_value,
                                        "gt_value": gt_value,
                                        "candidate_value": int(candidate_value),
                                        "scoring_status": "failed",
                                        "failure_reason": str(exc),
                                    }
                                )
                finally:
                    image.close()
        finally:
            del scorer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return all_rows, summarize_bad_basin_coordinate_records(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    case_bank_path = _resolve_path(cfg.input.case_bank_jsonl, anchor=config_path)
    output_dir = _resolve_path(cfg.run.output_dir, anchor=config_path) / cfg.run.name
    case_bank_rows = _read_jsonl(case_bank_path)
    selected_cases = select_duplicate_burst_cases(
        rows=case_bank_rows,
        max_cases_per_model=cfg.input.max_cases_per_model,
        serializer_surface=cfg.input.case_serializer_surface,
    )
    per_coord_rows, summary = _score_duplicate_burst_cases(
        cfg=cfg,
        selected_cases=selected_cases,
        config_path=config_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "selected_cases.jsonl", selected_cases)
    _write_jsonl(output_dir / "per_coord_scores.jsonl", per_coord_rows)
    _write_json(
        output_dir / "summary.json",
        {
            "selected_case_count": len(selected_cases),
            "case_serializer_surface": cfg.input.case_serializer_surface,
            "scoring_surface": cfg.input.scoring_surface,
            "selected_cases_path": str(output_dir / "selected_cases.jsonl"),
            "per_coord_scores_path": str(output_dir / "per_coord_scores.jsonl"),
            **summary,
        },
    )
    print(
        json.dumps(
            {
                "run_dir": str(output_dir),
                "selected_case_count": len(selected_cases),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
