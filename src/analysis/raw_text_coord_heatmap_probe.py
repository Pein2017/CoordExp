from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from src.analysis.duplication_followup import build_prefix_perturbation_variants
from src.analysis.raw_text_coord_continuity_report import build_xy_heatmap_grid
from src.analysis.raw_text_coord_continuity_scoring import (
    score_candidate_coordinate_xy_grid_batch,
)
from src.analysis.raw_text_coord_perturbation_probe import (
    PerturbationModelConfig,
    PerturbationModels,
    PerturbationScoringConfig,
    _build_case_context,
    _canonicalize_variant_prefix_objects,
    _parse_model_config,
    _read_jsonl,
)
from src.analysis.raw_text_coord_continuity_probe import (
    _require_mapping,
    _resolve_input_path,
    _resolve_output_dir,
    _write_json,
    _write_jsonl,
    render_pretty_inline_assistant_text,
)


@dataclass(frozen=True)
class HeatmapRunConfig:
    name: str
    output_dir: str


@dataclass(frozen=True)
class HeatmapSelectionConfig:
    selected_cases_jsonl: str
    max_cases: int = 4


@dataclass(frozen=True)
class HeatmapGridConfig:
    grid_radius: int = 16
    batch_size: int = 64
    centers: tuple[str, ...] = ("pred", "gt")
    variants: tuple[str, ...] = (
        "baseline",
        "replace_source_with_gt_next",
        "source_x1y1_from_gt_next",
    )


@dataclass(frozen=True)
class HeatmapConfig:
    run: HeatmapRunConfig
    selection: HeatmapSelectionConfig
    models: PerturbationModels
    scoring: PerturbationScoringConfig
    heatmap: HeatmapGridConfig


def _load_yaml(config_path: Path) -> dict[str, object]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("heatmap config root must be a mapping")
    return raw


def load_heatmap_config(config_path: Path) -> HeatmapConfig:
    raw = _load_yaml(config_path)
    run_raw = _require_mapping(raw, "run")
    selection_raw = _require_mapping(raw, "selection")
    models_raw = _require_mapping(raw, "models")
    scoring_raw = raw.get("scoring") or {}
    heatmap_raw = raw.get("heatmap") or {}
    if not isinstance(scoring_raw, dict):
        raise ValueError("scoring must be a mapping")
    if not isinstance(heatmap_raw, dict):
        raise ValueError("heatmap must be a mapping")
    return HeatmapConfig(
        run=HeatmapRunConfig(
            name=str(run_raw["name"]),
            output_dir=str(run_raw["output_dir"]),
        ),
        selection=HeatmapSelectionConfig(
            selected_cases_jsonl=str(selection_raw["selected_cases_jsonl"]),
            max_cases=int(selection_raw.get("max_cases", 4)),
        ),
        models=PerturbationModels(
            base=_parse_model_config(models_raw, "base"),
            pure_ce=_parse_model_config(models_raw, "pure_ce"),
        ),
        scoring=PerturbationScoringConfig(
            device=str(scoring_raw.get("device", "cuda:0")),
            attn_implementation=str(scoring_raw.get("attn_implementation", "auto")),
            candidate_radius=int(scoring_raw.get("candidate_radius", 8)),
        ),
        heatmap=HeatmapGridConfig(
            grid_radius=int(heatmap_raw.get("grid_radius", 16)),
            batch_size=int(heatmap_raw.get("batch_size", 64)),
            centers=tuple(
                str(value) for value in heatmap_raw.get("centers", ("pred", "gt"))
            ),
            variants=tuple(
                str(value)
                for value in heatmap_raw.get(
                    "variants",
                    ("baseline", "replace_source_with_gt_next", "source_x1y1_from_gt_next"),
                )
            ),
        ),
    )


def select_heatmap_cases(
    rows: Sequence[dict[str, object]],
    *,
    max_cases: int,
) -> list[dict[str, object]]:
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            int(row.get("selection_rank") if row.get("selection_rank") is not None else 10**9),
            -float(row.get("selection_wrong_anchor_advantage_at_4") or 0.0),
            str(row.get("case_id") or ""),
        ),
    )
    return ordered[: max(0, int(max_cases))]


def summarize_heatmap_rows(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    grids = {
        (
            str(row.get("model_alias") or ""),
            str(row.get("case_id") or ""),
            str(row.get("variant") or ""),
            str(row.get("center_kind") or ""),
        )
        for row in rows
    }
    case_ids = sorted({str(row.get("case_id") or "") for row in rows if row.get("case_id")})
    return {
        "num_heatmap_rows": len(rows),
        "num_grids": len(grids),
        "case_ids": case_ids,
    }


def _render_heatmap_png(
    *,
    grid: dict[str, object],
    output_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    z_matrix = np.array(grid["z_matrix"], dtype=float)
    x_values = list(grid["x_values"])
    y_values = list(grid["y_values"])
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    image = ax.imshow(z_matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("candidate_x1")
    ax.set_ylabel("candidate_y1")
    if x_values:
        tick_idx = list(range(0, len(x_values), max(1, len(x_values) // 4)))
        if tick_idx[-1] != len(x_values) - 1:
            tick_idx.append(len(x_values) - 1)
        ax.set_xticks(tick_idx, [str(x_values[idx]) for idx in tick_idx])
    if y_values:
        tick_idx = list(range(0, len(y_values), max(1, len(y_values) // 4)))
        if tick_idx[-1] != len(y_values) - 1:
            tick_idx.append(len(y_values) - 1)
        ax.set_yticks(tick_idx, [str(y_values[idx]) for idx in tick_idx])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="score")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _candidate_xy_pairs(center_x: int, center_y: int, *, radius: int) -> list[tuple[int, int]]:
    x_values = range(max(0, center_x - radius), min(999, center_x + radius) + 1)
    y_values = range(max(0, center_y - radius), min(999, center_y + radius) + 1)
    return [(int(x_value), int(y_value)) for y_value in y_values for x_value in x_values]


def _variant_prefix_lookup(
    *,
    prefix_objects: Sequence[dict[str, object]],
    source_object_index: int,
    gt_next: dict[str, object],
) -> dict[str, list[dict[str, object]]]:
    variants: dict[str, list[dict[str, object]]] = {"baseline": [dict(obj) for obj in prefix_objects]}
    for variant, variant_prefix in build_prefix_perturbation_variants(
        prefix_objects=prefix_objects,
        source_index_in_prefix=source_object_index,
        gt_next=gt_next,
    ):
        variants[str(variant)] = _canonicalize_variant_prefix_objects(variant_prefix)
    return variants


def _run_selected_heatmap_cases(
    *,
    cfg: HeatmapConfig,
    config_dir: Path,
    run_dir: Path,
    selected_cases: Sequence[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    from src.analysis.unmatched_proposal_verifier import TeacherForcedScorer
    import torch

    selected_cases_jsonl_path = _resolve_input_path(
        cfg.selection.selected_cases_jsonl,
        config_dir=config_dir,
    )
    per_cell_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    models: Sequence[PerturbationModelConfig] = (cfg.models.base, cfg.models.pure_ce)
    source_cache: dict[Path, list[dict[str, object]]] = {}
    for model_cfg in models:
        scorer = TeacherForcedScorer(
            checkpoint_path=_resolve_input_path(model_cfg.path, config_dir=config_dir),
            device=cfg.scoring.device,
            attn_implementation=cfg.scoring.attn_implementation,
            coord_mode=model_cfg.coord_mode,
        )
        try:
            for selected_case in selected_cases:
                image = None
                try:
                    case_context = _build_case_context(
                        selected_case,
                        cohort_jsonl_path=selected_cases_jsonl_path,
                        source_cache=source_cache,
                    )
                    image = case_context["image"]
                    prefix_objects = list(case_context["prefix_objects"])
                    pred_object = dict(case_context["pred_object"])
                    gt_next = dict(case_context["gt_next"])
                    pred_bbox = [int(value) for value in pred_object["bbox_2d"]]
                    gt_bbox = [int(value) for value in gt_next["bbox_2d"]]
                    center_lookup = {
                        "pred": (int(pred_bbox[0]), int(pred_bbox[1])),
                        "gt": (int(gt_bbox[0]), int(gt_bbox[1])),
                    }
                    variant_lookup = _variant_prefix_lookup(
                        prefix_objects=prefix_objects,
                        source_object_index=int(selected_case["source_object_index"]),
                        gt_next=gt_next,
                    )
                    for variant in cfg.heatmap.variants:
                        variant_prefix = variant_lookup.get(variant)
                        if variant_prefix is None:
                            continue
                        assistant_text = render_pretty_inline_assistant_text(
                            {"objects": list(variant_prefix) + [pred_object]},
                            object_field_order=model_cfg.object_field_order,
                        )
                        object_index = len(variant_prefix)
                        for center_kind in cfg.heatmap.centers:
                            if center_kind not in center_lookup:
                                continue
                            center_x, center_y = center_lookup[center_kind]
                            candidate_pairs = _candidate_xy_pairs(
                                center_x,
                                center_y,
                                radius=cfg.heatmap.grid_radius,
                            )
                            scored_rows = score_candidate_coordinate_xy_grid_batch(
                                scorer=scorer,
                                image=image,
                                assistant_text=assistant_text,
                                original_bbox=pred_bbox,
                                candidate_xy_pairs=candidate_pairs,
                                prompt_variant=model_cfg.prompt_variant,
                                object_field_order=model_cfg.object_field_order,
                                object_index=object_index,
                                batch_size=cfg.heatmap.batch_size,
                            )
                            enriched_rows: list[dict[str, object]] = []
                            for scored in scored_rows:
                                enriched_rows.append(
                                    {
                                        "model_alias": model_cfg.alias,
                                        "case_id": str(selected_case["case_id"]),
                                        "image_id": int(selected_case["image_id"]),
                                        "line_idx": int(selected_case["line_idx"]),
                                        "object_index": int(selected_case["object_index"]),
                                        "source_object_index": int(selected_case["source_object_index"]),
                                        "variant": str(variant),
                                        "center_kind": str(center_kind),
                                        "center_x1": int(center_x),
                                        "center_y1": int(center_y),
                                        "pred_x1": int(pred_bbox[0]),
                                        "pred_y1": int(pred_bbox[1]),
                                        "gt_x1": int(gt_bbox[0]),
                                        "gt_y1": int(gt_bbox[1]),
                                        "candidate_x1": int(scored["candidate_x1"]),
                                        "candidate_y1": int(scored["candidate_y1"]),
                                        "score": float(scored["sum_logprob"]),
                                        "sum_logprob": float(scored["sum_logprob"]),
                                        "mean_logprob": float(scored["mean_logprob"]),
                                        "token_count": int(scored["count"]),
                                        "candidate_token_span": len(list(scored["absolute_positions"])),
                                    }
                                )
                            per_cell_rows.extend(enriched_rows)
                            grid_json = build_xy_heatmap_grid(enriched_rows)
                            figure_path = (
                                run_dir
                                / "figures"
                                / f"{selected_case['case_id']}__{model_cfg.alias}__{variant}__{center_kind}.png"
                            )
                            _render_heatmap_png(
                                grid=grid_json,
                                output_path=figure_path,
                                title=(
                                    f"{selected_case['case_id']} | {model_cfg.alias} | "
                                    f"{variant} | center={center_kind}"
                                ),
                            )
                            grid_rows.append(
                                {
                                    "model_alias": model_cfg.alias,
                                    "case_id": str(selected_case["case_id"]),
                                    "variant": str(variant),
                                    "center_kind": str(center_kind),
                                    "figure_path": str(figure_path),
                                    "grid_json": grid_json,
                                }
                            )
                finally:
                    if image is not None:
                        image.close()
        finally:
            del scorer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {"per_cell_rows": per_cell_rows, "grid_rows": grid_rows}


def run_heatmap_probe(config_path: Path) -> dict[str, object]:
    resolved_config_path = config_path.resolve()
    cfg = load_heatmap_config(resolved_config_path)
    run_dir = _resolve_output_dir(cfg.run.output_dir) / cfg.run.name
    run_dir.mkdir(parents=True, exist_ok=True)

    selected_cases_jsonl_path = _resolve_input_path(
        cfg.selection.selected_cases_jsonl,
        config_dir=resolved_config_path.parent,
    )
    selected_case_rows = _read_jsonl(selected_cases_jsonl_path)
    selected_cases = select_heatmap_cases(
        selected_case_rows,
        max_cases=cfg.selection.max_cases,
    )
    _write_jsonl(run_dir / "selected_cases.jsonl", selected_cases)

    execution = _run_selected_heatmap_cases(
        cfg=cfg,
        config_dir=resolved_config_path.parent,
        run_dir=run_dir,
        selected_cases=selected_cases,
    )
    per_cell_rows = list(execution["per_cell_rows"])
    grid_rows = list(execution["grid_rows"])
    _write_jsonl(run_dir / "per_cell_scores.jsonl", per_cell_rows)
    _write_jsonl(run_dir / "heatmaps.jsonl", grid_rows)

    summary = {
        "run_name": cfg.run.name,
        "selection": {
            "num_input_cases": len(selected_case_rows),
            "num_selected_cases": len(selected_cases),
            "selected_cases_jsonl": str(selected_cases_jsonl_path),
        },
        "results": {
            **summarize_heatmap_rows(per_cell_rows),
            "num_heatmap_artifacts": len(grid_rows),
        },
    }
    _write_json(run_dir / "summary.json", summary)
    return {"run_dir": str(run_dir), "summary": summary}
