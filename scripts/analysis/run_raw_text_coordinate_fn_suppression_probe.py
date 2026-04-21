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
)
from src.analysis.raw_text_coordinate_continuation_scoring import (
    score_candidate_continuations_batch,
)
from src.analysis.raw_text_coordinate_exploratory import (
    render_model_native_prefix_assistant_text,
)
from src.analysis.raw_text_coordinate_fn_probe import (
    rank_fn_suppression_candidates,
    summarize_fn_suppression_margin_rows,
)
from src.analysis.unmatched_proposal_verifier import TeacherForcedScorer
from src.common.paths import resolve_image_path_best_effort, resolve_image_path_strict


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str


@dataclass(frozen=True)
class InputConfig:
    fn_objects_jsonl: str
    gt_proxy_scores_jsonl: str
    proposal_proxy_scores_jsonl: str
    baseline_gt_vs_pred_jsonl: str
    baseline_pred_token_trace_jsonl: str
    max_cases: int
    min_recover_fraction_full: float
    scoring_surface: str


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
        raise TypeError("fn suppression config root must be a mapping")
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
            fn_objects_jsonl=str(input_raw["fn_objects_jsonl"]),
            gt_proxy_scores_jsonl=str(input_raw["gt_proxy_scores_jsonl"]),
            proposal_proxy_scores_jsonl=str(input_raw["proposal_proxy_scores_jsonl"]),
            baseline_gt_vs_pred_jsonl=str(input_raw["baseline_gt_vs_pred_jsonl"]),
            baseline_pred_token_trace_jsonl=str(input_raw["baseline_pred_token_trace_jsonl"]),
            max_cases=int(input_raw.get("max_cases", 5)),
            min_recover_fraction_full=float(input_raw.get("min_recover_fraction_full", 0.5)),
            scoring_surface=str(input_raw.get("scoring_surface", "model_native")),
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


def _best_gt_support_rows(gt_rows: list[dict[str, object]]) -> dict[tuple[int, int], dict[str, object]]:
    gt_by_key: dict[tuple[int, int], dict[str, object]] = {}
    for row in gt_rows:
        if str(row.get("scoring_status") or "ok") != "ok":
            continue
        image_idx = row.get("source_image_idx", row.get("image_idx"))
        gt_idx = row.get("source_gt_idx", row.get("gt_idx"))
        if not isinstance(image_idx, int) or not isinstance(gt_idx, int):
            continue
        key = (int(image_idx), int(gt_idx))
        current = gt_by_key.get(key)
        current_score = float(current.get("combined_linear") or 0.0) if current else float("-inf")
        row_score = float(row.get("combined_linear") or 0.0)
        if current is None or row_score > current_score:
            gt_by_key[key] = row
    return gt_by_key


def _proposal_support_indices(
    proposal_rows: list[dict[str, object]],
) -> tuple[dict[tuple[int, int, str], float], dict[tuple[int, str], list[float]]]:
    support_by_target: dict[tuple[int, int, str], float] = {}
    competitor_scores: dict[tuple[int, str], list[float]] = {}
    for row in proposal_rows:
        if str(row.get("scoring_status") or "ok") != "ok":
            continue
        if not bool(row.get("collection_valid", True)):
            continue
        if int(row.get("is_unmatched") or 0) != 1:
            continue
        image_idx = row.get("image_idx")
        nearest_gt_idx = row.get("nearest_gt_idx")
        desc = str(row.get("desc") or "").strip()
        if not isinstance(image_idx, int) or not isinstance(nearest_gt_idx, int) or not desc:
            continue
        score = float(row.get("combined_linear") or 0.0)
        target_key = (int(image_idx), int(nearest_gt_idx), desc)
        support_by_target[target_key] = max(support_by_target.get(target_key, float("-inf")), score)
        competitor_scores.setdefault((int(image_idx), desc), []).append(score)
    return support_by_target, competitor_scores


def _build_selected_fn_cases(
    *,
    cfg: Config,
    config_path: Path,
) -> list[dict[str, object]]:
    fn_rows = _read_jsonl(_resolve_path(cfg.input.fn_objects_jsonl, anchor=config_path))
    gt_rows = _read_jsonl(_resolve_path(cfg.input.gt_proxy_scores_jsonl, anchor=config_path))
    proposal_rows = _read_jsonl(_resolve_path(cfg.input.proposal_proxy_scores_jsonl, anchor=config_path))
    gt_by_key = _best_gt_support_rows(gt_rows)
    proposal_by_target, competitor_by_image_desc = _proposal_support_indices(proposal_rows)
    ranked_rows: list[dict[str, object]] = []
    for row in fn_rows:
        if not bool(row.get("ever_recovered_full")):
            continue
        recover_fraction_full = float(row.get("recover_fraction_full") or 0.0)
        if recover_fraction_full < cfg.input.min_recover_fraction_full:
            continue
        image_idx = row.get("record_idx", row.get("image_id"))
        gt_idx = row.get("gt_idx")
        gt_desc = str(row.get("gt_desc") or "").strip()
        if not isinstance(image_idx, int) or not isinstance(gt_idx, int) or not gt_desc:
            continue
        key = (int(image_idx), int(gt_idx))
        gt_support_row = gt_by_key.get(key)
        teacher_forced_support = (
            float(gt_support_row.get("combined_linear") or 0.0)
            if gt_support_row is not None
            else 0.0
        )
        proposal_support = proposal_by_target.get((int(image_idx), int(gt_idx), gt_desc), 0.0)
        support_anchor = max(teacher_forced_support, proposal_support)
        competitor_margin = max(
            [
                score
                for score in competitor_by_image_desc.get((int(image_idx), gt_desc), [])
                if score > proposal_support or proposal_support == 0.0
            ]
            or [0.0]
        ) - support_anchor
        ranked_rows.append(
            {
                **dict(row),
                "case_id": f"fn:{image_idx}:{gt_idx}:{gt_desc}",
                "teacher_forced_support": teacher_forced_support,
                "proposal_support": float(proposal_support),
                "competitor_margin": float(competitor_margin),
            }
        )
    ranked = rank_fn_suppression_candidates(rows=ranked_rows, max_cases=max(len(ranked_rows), 1))
    selected: list[dict[str, object]] = []
    seen_images: set[int] = set()
    for row in ranked:
        image_idx = int(row["record_idx"])
        if image_idx in seen_images:
            continue
        selected.append(dict(row))
        seen_images.add(image_idx)
        if len(selected) >= cfg.input.max_cases:
            return selected
    for row in ranked:
        if len(selected) >= cfg.input.max_cases:
            break
        if any(str(existing["case_id"]) == str(row["case_id"]) for existing in selected):
            continue
        selected.append(dict(row))
    return selected


def _render_full_assistant_text(
    *,
    scoring_surface: str,
    objects: list[dict[str, object]],
    object_field_order: str,
    native_assistant_text: object,
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


def _render_append_ready_prefix(
    *,
    scoring_surface: str,
    objects: list[dict[str, object]],
    object_field_order: str,
    native_assistant_text: object,
) -> str:
    full_text = _render_full_assistant_text(
        scoring_surface=scoring_surface,
        objects=objects,
        object_field_order=object_field_order,
        native_assistant_text=native_assistant_text,
    )
    stripped = full_text.strip()
    if scoring_surface == "pretty_inline":
        if not stripped.endswith("]}"):
            raise ValueError("pretty_inline assistant text must end with ]}")
        prefix = stripped[:-2]
        return prefix + (", " if objects else "")
    if stripped.startswith("```"):
        fence_prefix = "```json\n"
        fence_suffix = "\n```"
        if not stripped.startswith(fence_prefix) or not stripped.endswith(fence_suffix):
            raise ValueError("unexpected fenced native assistant text")
        inner = stripped[len(fence_prefix) : -len(fence_suffix)]
        tail = "\n  ]\n}"
        if not inner.endswith(tail):
            raise ValueError("unexpected native assistant JSON tail")
        prefix_inner = inner[: -len(tail)]
        if objects:
            prefix_inner += ",\n"
        return fence_prefix + prefix_inner
    if not stripped.endswith("]}"):
        raise ValueError("native assistant text must end with ]}")
    prefix = stripped[:-2]
    return prefix + (", " if objects else "")


def _score_fn_cases(
    *,
    cfg: Config,
    selected_cases: list[dict[str, object]],
    config_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    baseline_gt_rows = _read_jsonl(_resolve_path(cfg.input.baseline_gt_vs_pred_jsonl, anchor=config_path))
    baseline_trace_rows = _read_jsonl(
        _resolve_path(cfg.input.baseline_pred_token_trace_jsonl, anchor=config_path)
    )
    case_score_rows: list[dict[str, object]] = []
    margin_rows: list[dict[str, object]] = []
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
                record_idx = int(case_row["record_idx"])
                gt_idx = int(case_row["gt_idx"])
                source_record = baseline_gt_rows[record_idx]
                trace_row = baseline_trace_rows[record_idx]
                width = int(source_record.get("width") or 0)
                height = int(source_record.get("height") or 0)
                prefix_objects: list[dict[str, object]] = []
                for pred_obj in list(source_record.get("pred") or []):
                    norm_obj = _object_to_norm1000(dict(pred_obj), width=width, height=height)
                    if norm_obj is not None:
                        prefix_objects.append(norm_obj)
                gt_objects = list(source_record.get("gt") or [])
                if gt_idx >= len(gt_objects):
                    continue
                gt_object = _object_to_norm1000(dict(gt_objects[gt_idx]), width=width, height=height)
                if gt_object is None:
                    continue
                native_assistant_text = trace_row.get("generated_token_text")
                prefix_assistant_text = _render_append_ready_prefix(
                    scoring_surface=cfg.input.scoring_surface,
                    objects=prefix_objects,
                    object_field_order=model_cfg.object_field_order,
                    native_assistant_text=native_assistant_text,
                )
                candidate_rows = [
                    {
                        "candidate_label": "eos_now",
                        "candidate_assistant_text": _render_full_assistant_text(
                            scoring_surface=cfg.input.scoring_surface,
                            objects=prefix_objects,
                            object_field_order=model_cfg.object_field_order,
                            native_assistant_text=native_assistant_text,
                        ),
                    },
                    {
                        "candidate_label": "continue_with_gt",
                        "candidate_assistant_text": _render_full_assistant_text(
                            scoring_surface=cfg.input.scoring_surface,
                            objects=prefix_objects + [gt_object],
                            object_field_order=model_cfg.object_field_order,
                            native_assistant_text=native_assistant_text,
                        ),
                    },
                ]
                image = _resolve_case_image(
                    _resolve_path(cfg.input.baseline_gt_vs_pred_jsonl, anchor=config_path),
                    source_record,
                )
                try:
                    scored_rows = score_candidate_continuations_batch(
                        scorer=scorer,
                        image=image,
                        baseline_assistant_text=prefix_assistant_text,
                        candidate_rows=candidate_rows,
                        prompt_variant=model_cfg.prompt_variant,
                        object_field_order=model_cfg.object_field_order,
                    )
                finally:
                    image.close()
                scored_by_label: dict[str, dict[str, object]] = {}
                for scored in scored_rows:
                    candidate_label = str(scored["candidate_label"])
                    scored_by_label[candidate_label] = dict(scored)
                    case_score_rows.append(
                        {
                            "case_id": case_row["case_id"],
                            "model_alias": model_cfg.alias,
                            "record_idx": record_idx,
                            "gt_idx": gt_idx,
                            "image_id": case_row["image_id"],
                            "gt_desc": str(case_row["gt_desc"]),
                            "recover_fraction_full": float(case_row["recover_fraction_full"]),
                            "teacher_forced_support": float(case_row["teacher_forced_support"]),
                            "proposal_support": float(case_row["proposal_support"]),
                            "competitor_margin": float(case_row["competitor_margin"]),
                            "candidate_label": candidate_label,
                            "scoring_surface": cfg.input.scoring_surface,
                            "sum_logprob": float(scored["sum_logprob"]),
                            "mean_logprob": float(scored["mean_logprob"]),
                            "token_count": int(scored["count"]),
                            "candidate_token_span": len(
                                list(scored["assistant_relative_positions"])
                            ),
                        }
                    )
                eos_now = scored_by_label.get("eos_now")
                continue_with_gt = scored_by_label.get("continue_with_gt")
                if eos_now is None or continue_with_gt is None:
                    continue
                continue_minus_eos_sum = float(continue_with_gt["sum_logprob"]) - float(
                    eos_now["sum_logprob"]
                )
                continue_minus_eos_mean = float(continue_with_gt["mean_logprob"]) - float(
                    eos_now["mean_logprob"]
                )
                margin_rows.append(
                    {
                        "case_id": case_row["case_id"],
                        "model_alias": model_cfg.alias,
                        "record_idx": record_idx,
                        "gt_idx": gt_idx,
                        "image_id": case_row["image_id"],
                        "gt_desc": str(case_row["gt_desc"]),
                        "recover_fraction_full": float(case_row["recover_fraction_full"]),
                        "teacher_forced_support": float(case_row["teacher_forced_support"]),
                        "proposal_support": float(case_row["proposal_support"]),
                        "competitor_margin": float(case_row["competitor_margin"]),
                        "scoring_surface": cfg.input.scoring_surface,
                        "eos_now_sum_logprob": float(eos_now["sum_logprob"]),
                        "eos_now_mean_logprob": float(eos_now["mean_logprob"]),
                        "eos_now_token_count": int(eos_now["count"]),
                        "continue_with_gt_sum_logprob": float(continue_with_gt["sum_logprob"]),
                        "continue_with_gt_mean_logprob": float(continue_with_gt["mean_logprob"]),
                        "continue_with_gt_token_count": int(continue_with_gt["count"]),
                        "continue_minus_eos_sum_logprob": continue_minus_eos_sum,
                        "continue_minus_eos_mean_logprob": continue_minus_eos_mean,
                        "stop_pressure_signature": bool(
                            continue_minus_eos_sum < 0.0 and continue_minus_eos_mean > 0.0
                        ),
                    }
                )
        finally:
            del scorer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return case_score_rows, margin_rows, summarize_fn_suppression_margin_rows(margin_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    output_dir = _resolve_path(cfg.run.output_dir, anchor=config_path) / cfg.run.name
    selected_cases = _build_selected_fn_cases(cfg=cfg, config_path=config_path)
    case_score_rows, margin_rows, summary = _score_fn_cases(
        cfg=cfg,
        selected_cases=selected_cases,
        config_path=config_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "selected_cases.jsonl", selected_cases)
    _write_jsonl(output_dir / "per_candidate_scores.jsonl", case_score_rows)
    _write_jsonl(output_dir / "per_case_margins.jsonl", margin_rows)
    _write_json(
        output_dir / "summary.json",
        {
            "selected_case_count": len(selected_cases),
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
                "selected_case_count": len(selected_cases),
                "margin_rows": len(margin_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
