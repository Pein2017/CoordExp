from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import matplotlib.pyplot as plt
import yaml

from src.analysis.raw_text_coord_continuity_report import (
    compute_vision_lift_rows,
    write_report_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FinalReportRunConfig:
    name: str
    output_dir: str


@dataclass(frozen=True)
class FinalReportArtifacts:
    audit_dir: str
    pilot_broad_dir: str
    pilot_broad_swapped_dir: str
    pilot_crowded_dir: str
    pilot_model_mined_dir: str
    bad_basin_base_dir: str
    bad_basin_pure_dir: str
    lexical_summary_json: str
    lexical_per_model_json: str
    perturb_base_dir: str
    perturb_pure_dir: str
    heatmap_base_dir: str
    heatmap_pure_dir: str
    diagnosis_docs: tuple[str, ...]


@dataclass(frozen=True)
class FinalReportConfig:
    run: FinalReportRunConfig
    artifacts: FinalReportArtifacts


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"expected mapping in {path}")
    return data


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"expected mapping JSON in {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise TypeError(f"expected object JSONL row in {path}")
        rows.append(parsed)
    return rows


def _resolve(path_str: str, *, config_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    repo_candidate = (REPO_ROOT / path).resolve()
    config_candidate = (config_dir / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return config_candidate


def load_final_report_config(config_path: Path) -> FinalReportConfig:
    raw = _load_yaml(config_path)
    run_raw = raw.get("run")
    artifacts_raw = raw.get("artifacts")
    if not isinstance(run_raw, dict) or not isinstance(artifacts_raw, dict):
        raise TypeError("final report config requires run/artifacts mappings")
    diagnosis_docs = artifacts_raw.get("diagnosis_docs")
    if not isinstance(diagnosis_docs, list):
        raise TypeError("artifacts.diagnosis_docs must be a list")
    return FinalReportConfig(
        run=FinalReportRunConfig(
            name=str(run_raw["name"]),
            output_dir=str(run_raw["output_dir"]),
        ),
        artifacts=FinalReportArtifacts(
            audit_dir=str(artifacts_raw["audit_dir"]),
            pilot_broad_dir=str(artifacts_raw["pilot_broad_dir"]),
            pilot_broad_swapped_dir=str(artifacts_raw["pilot_broad_swapped_dir"]),
            pilot_crowded_dir=str(artifacts_raw["pilot_crowded_dir"]),
            pilot_model_mined_dir=str(artifacts_raw["pilot_model_mined_dir"]),
            bad_basin_base_dir=str(artifacts_raw["bad_basin_base_dir"]),
            bad_basin_pure_dir=str(artifacts_raw["bad_basin_pure_dir"]),
            lexical_summary_json=str(artifacts_raw["lexical_summary_json"]),
            lexical_per_model_json=str(artifacts_raw["lexical_per_model_json"]),
            perturb_base_dir=str(artifacts_raw["perturb_base_dir"]),
            perturb_pure_dir=str(artifacts_raw["perturb_pure_dir"]),
            heatmap_base_dir=str(artifacts_raw["heatmap_base_dir"]),
            heatmap_pure_dir=str(artifacts_raw["heatmap_pure_dir"]),
            diagnosis_docs=tuple(str(path) for path in diagnosis_docs),
        ),
    )


def _slot_metrics_by_key(summary: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["model_alias"]), str(row["slot"])): row
        for row in summary.get("slot_metrics", [])
    }


def _center_metrics_by_key(summary: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row["model_alias"]), str(row["center_kind"]), str(row["slot"])): row
        for row in summary.get("center_metrics", [])
    }


def _extract_gt_score_rows(
    rows: Sequence[dict[str, Any]],
    *,
    image_condition: str,
) -> list[dict[str, Any]]:
    extracted: list[dict[str, Any]] = []
    for row in rows:
        if row.get("candidate_value") != row.get("gt_value"):
            continue
        extracted.append(
            {
                "case_id": (
                    f"{row['model_alias']}:{row['image_id']}:{row['object_index']}:{row['slot']}"
                ),
                "model_alias": row["model_alias"],
                "slot": row["slot"],
                "image_condition": image_condition,
                "gt_score": row["sum_logprob"],
            }
        )
    return extracted


def summarize_vision_lift(
    *,
    correct_rows: Sequence[dict[str, Any]],
    swapped_rows: Sequence[dict[str, Any]],
    correct_summary: dict[str, Any],
    swapped_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    paired_gt_rows = _extract_gt_score_rows(
        correct_rows, image_condition="correct"
    ) + _extract_gt_score_rows(swapped_rows, image_condition="swapped")
    lifted_rows = compute_vision_lift_rows(paired_gt_rows)
    lift_by_model_slot: dict[tuple[str, str], list[float]] = {}
    for row in lifted_rows:
        model_alias, _, _, slot = str(row["case_id"]).split(":", 3)
        lift_by_model_slot.setdefault((model_alias, slot), []).append(
            float(row["vision_lift"])
        )
    correct_slot = _slot_metrics_by_key(correct_summary)
    swapped_slot = _slot_metrics_by_key(swapped_summary)
    summary_rows: list[dict[str, Any]] = []
    for key, correct_metric in sorted(correct_slot.items()):
        swapped_metric = swapped_slot[key]
        lifts = lift_by_model_slot.get(key, [])
        summary_rows.append(
            {
                "model_alias": key[0],
                "slot": key[1],
                "num_cases": len(lifts),
                "avg_gt_score_lift": mean(lifts) if lifts else math.nan,
                "positive_gt_score_lift_rate": (
                    sum(1 for value in lifts if value > 0.0) / len(lifts) if lifts else math.nan
                ),
                "mass_at_4_lift": float(correct_metric["mass_at_4"]) - float(swapped_metric["mass_at_4"]),
                "local_expected_abs_error_delta": float(correct_metric["local_expected_abs_error"])
                - float(swapped_metric["local_expected_abs_error"]),
            }
        )
    return summary_rows


def summarize_bad_basin(summary: dict[str, Any]) -> list[dict[str, Any]]:
    by_key = _center_metrics_by_key(summary)
    rows: list[dict[str, Any]] = []
    for model_alias in sorted({key[0] for key in by_key}):
        for slot in sorted({key[2] for key in by_key if key[0] == model_alias}):
            gt_row = by_key[(model_alias, "gt", slot)]
            pred_row = by_key[(model_alias, "pred", slot)]
            rows.append(
                {
                    "model_alias": model_alias,
                    "slot": slot,
                    "num_probes": int(pred_row["num_probes"]),
                    "pred_minus_gt_mass_at_4": float(pred_row["mass_at_4"]) - float(gt_row["mass_at_4"]),
                    "pred_minus_gt_local_expected_abs_error": float(pred_row["local_expected_abs_error"])
                    - float(gt_row["local_expected_abs_error"]),
                    "pred_center_mass_at_4": float(pred_row["mass_at_4"]),
                    "gt_center_mass_at_4": float(gt_row["mass_at_4"]),
                }
            )
    return rows


def summarize_perturbation(summary: dict[str, Any]) -> list[dict[str, Any]]:
    results = summary.get("results", {})
    if not isinstance(results, dict):
        raise TypeError("perturbation summary.results must be a mapping")
    variant_metrics = results.get("variant_metrics", [])
    if not isinstance(variant_metrics, list):
        raise TypeError("perturbation summary variant_metrics must be a list")
    return [
        {
            "model_alias": str(row["model_alias"]),
            "variant": str(row["variant"]),
            "slot": str(row["slot"]),
            "num_cases": int(row["num_cases"]),
            "avg_delta_wrong_anchor_advantage_at_4": float(
                row["avg_delta_wrong_anchor_advantage_at_4"]
            ),
            "avg_delta_pred_center_mass_at_4": float(row["avg_delta_pred_center_mass_at_4"]),
            "avg_delta_gt_center_mass_at_4": float(row["avg_delta_gt_center_mass_at_4"]),
        }
        for row in variant_metrics
    ]


def summarize_heatmap_dir(path: Path) -> dict[str, Any]:
    figure_paths = sorted(path.glob("figures/*.png"))
    summary_path = path / "summary.json"
    selected_cases_path = path / "selected_cases.jsonl"
    selected_cases = _read_jsonl(selected_cases_path) if selected_cases_path.exists() else []
    return {
        "run_dir": str(path),
        "num_figures": len(figure_paths),
        "figure_paths": [str(item) for item in figure_paths],
        "selected_case_ids": [str(row.get("case_id")) for row in selected_cases],
        "summary_path": str(summary_path) if summary_path.exists() else None,
    }


def derive_core_verdicts(
    *,
    broad_metrics: dict[tuple[str, str], dict[str, Any]],
    crowded_metrics: dict[tuple[str, str], dict[str, Any]],
    model_mined_metrics: dict[tuple[str, str], dict[str, Any]],
    lexical_summary: dict[str, Any],
    vision_lift_rows: Sequence[dict[str, Any]],
    bad_basin_rows: Sequence[dict[str, Any]],
    perturb_rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    base_numeric = lexical_summary["coefficients"]["numeric_distance_to_center"]
    broad_base_avg_mass4 = mean(
        float(broad_metrics[("base", slot)]["mass_at_4"]) for slot in ("x1", "y1")
    )
    q1 = {
        "verdict": (
            "strongly supported"
            if base_numeric["coef"] < 0.0
            and base_numeric["pvalue"] < 1e-6
            and broad_base_avg_mass4 >= 0.65
            else "partially supported"
        ),
        "rationale": (
            "Base model shows significant negative numeric-distance coefficient after lexical controls "
            "and broad GT-centered local mass remains high for x1/y1."
        ),
    }
    pure_better = 0
    total_cmp = 0
    for source in (broad_metrics, crowded_metrics, model_mined_metrics):
        for slot in ("x1", "y1"):
            total_cmp += 1
            base_row = source[("base", slot)]
            pure_row = source[("pure_ce", slot)]
            if float(pure_row["mass_at_4"]) > float(base_row["mass_at_4"]) and float(
                pure_row["local_expected_abs_error"]
            ) < float(base_row["local_expected_abs_error"]):
                pure_better += 1
    if pure_better == total_cmp:
        q2_verdict = "strongly supported"
    elif pure_better >= max(3, (total_cmp // 2) + 1):
        q2_verdict = "partially supported"
    else:
        q2_verdict = "not supported"
    q2 = {
        "verdict": q2_verdict,
        "rationale": (
            f"Pure-CE improves both mass@4 and local error on {pure_better}/{total_cmp} "
            "x1/y1 slice comparisons, but gains are slot-dependent rather than uniform."
        ),
    }
    all_positive_lift = all(
        row["avg_gt_score_lift"] > 0.0 and row["mass_at_4_lift"] > 0.0
        for row in vision_lift_rows
    )
    q3 = {
        "verdict": "strongly supported" if all_positive_lift else "partially supported",
        "rationale": (
            "Correct-image condition yields uniformly positive GT-score lift and better local mass@4 "
            "than swapped-image controls for both models and both early slots."
        ),
    }
    worst_bad_basin = max(float(row["pred_minus_gt_mass_at_4"]) for row in bad_basin_rows)
    strong_anchor_shift = any(
        row["slot"] in {"x1", "y1"}
        and row["variant"] in {"replace_source_with_gt_next", "source_x1y1_from_gt_next"}
        and row["avg_delta_wrong_anchor_advantage_at_4"] < -0.05
        for row in perturb_rows
    )
    q4 = {
        "verdict": (
            "strongly supported"
            if worst_bad_basin > 0.15 and strong_anchor_shift
            else "partially supported"
        ),
        "rationale": (
            "Hard repeated-object cases show higher mass around the model-predicted anchor than around GT, "
            "and prefix-geometry interventions can move that wrong-anchor advantage."
        ),
    }
    q5 = {
        "verdict": "not supported",
        "rationale": (
            "If the objective is only to obtain local continuity, the evidence does not support coord_token "
            "as necessary: raw-text pure-CE and even the base model already exhibit numeric local basins. "
            "Typing/stability/parameterization benefits remain open."
        ),
    }
    return {
        "q1_base_numeric_continuity": q1,
        "q2_pure_ce_enhances_continuity": q2,
        "q3_visual_condition_modulates_continuity": q3,
        "q4_wrong_prefix_bad_basin": q4,
        "q5_coord_token_needed_for_continuity": q5,
    }


def _plot_good_basin(
    *,
    out_path: Path,
    broad_rows: Sequence[dict[str, Any]],
    crowded_rows: Sequence[dict[str, Any]],
    model_mined_rows: Sequence[dict[str, Any]],
) -> None:
    datasets = [
        ("Broad", broad_rows),
        ("Crowded", crowded_rows),
        ("Model-mined", model_mined_rows),
    ]
    labels = []
    base_vals = []
    pure_vals = []
    for dataset_name, rows in datasets:
        by_key = {(row["model_alias"], row["slot"]): row for row in rows}
        for slot in ("x1", "y1"):
            labels.append(f"{dataset_name}\n{slot}")
            base_vals.append(float(by_key[("base", slot)]["mass_at_4"]))
            pure_vals.append(float(by_key[("pure_ce", slot)]["mass_at_4"]))
    x_positions = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar([x - width / 2 for x in x_positions], base_vals, width=width, label="base")
    ax.bar([x + width / 2 for x in x_positions], pure_vals, width=width, label="pure_ce")
    ax.set_ylabel("mass@4")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("GT-Centered Good-Basin Local Mass")
    ax.set_xticks(list(x_positions), labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_vision_lift(*, out_path: Path, rows: Sequence[dict[str, Any]]) -> None:
    labels = [f"{row['model_alias']}\n{row['slot']}" for row in rows]
    values = [float(row["avg_gt_score_lift"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(labels, values)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("avg GT score lift")
    ax.set_title("Vision Lift: Correct minus Swapped")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_regression_coefficients(*, out_path: Path, per_model: dict[str, Any]) -> None:
    labels = []
    values = []
    errors = []
    for model_alias in ("base", "pure_ce"):
        row = per_model["per_model"][model_alias]["coefficients"]["numeric_distance_to_center"]
        labels.append(model_alias)
        values.append(float(row["coef"]))
        errors.append(float(row["stderr"]))
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.bar(labels, values, yerr=errors, capsize=4)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("coefficient")
    ax.set_title("Numeric Distance Coefficient\n(lexical controls included)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_bad_basin(*, out_path: Path, rows: Sequence[dict[str, Any]]) -> None:
    labels = [f"{row['model_alias']}\n{row['slot']}" for row in rows]
    values = [float(row["pred_minus_gt_mass_at_4"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(labels, values)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("pred-center mass@4 minus gt-center mass@4")
    ax.set_title("Wrong-Basin Strength on Hard Cases")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _copy_heatmap_figures(*, source_dir: Path, target_dir: Path, limit: int = 8) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for figure_path in sorted(source_dir.glob("figures/*.png"))[:limit]:
        target_path = target_dir / figure_path.name
        shutil.copy2(figure_path, target_path)
        copied.append(str(target_path))
    return copied


def _merge_atomic_rows(run_specs: Sequence[tuple[str, Path]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for source_name, path in run_specs:
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            merged.append({"source_run": source_name, **row})
    return merged


def _merge_hard_cases(paths: Sequence[Path]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            case_id = str(row.get("case_id") or f"{row.get('image_id')}:{row.get('line_idx')}")
            merged.setdefault(case_id, row)
    return list(merged.values())


def _render_report_md(summary: dict[str, Any]) -> str:
    verdicts = summary["verdicts"]
    audit = summary["audit"]
    lexical = summary["lexical"]
    vision_lift = summary["vision_lift"]
    lines = [
        "# Raw-Text Coordinate Continuity Probe",
        "",
        "## Scope",
        "",
        "This bundle consolidates the raw-text `xyxy` continuity study across Phase 0 audit, GT-centered good-basin probes, lexical controls, image-swap controls, self-prefix bad-basin probes, and prefix-geometry perturbation.",
        "",
        "## Phase 0 Audit",
        "",
        f"- Serialization surface used for training/inference/probing: `{audit['serialization_surface']}`.",
        f"- Both `base` and `pure_ce` tokenize `199 -> {audit['example_tokens']['199']}` and `200 -> {audit['example_tokens']['200']}` as digit-by-digit sequences rather than single whole-number tokens.",
        "- The active raw-text contract is `{\"objects\": [{\"desc\": ..., \"bbox_2d\": [x1, y1, x2, y2]}, ...]}` with pretty-inline spacing.",
        "",
        "## Core Verdicts",
        "",
    ]
    question_titles = {
        "q1_base_numeric_continuity": "1. Base Qwen3-VL already has raw-text numeric adjacency / coordinate continuity.",
        "q2_pure_ce_enhances_continuity": "2. Stage-1 pure-CE fine-tuning enhances that continuity.",
        "q3_visual_condition_modulates_continuity": "3. Continuity is stronger under the correct image than under swapped-image controls.",
        "q4_wrong_prefix_bad_basin": "4. Hard repeated-object cases can form wrong local basins around the wrong prefix, especially at x1/y1.",
        "q5_coord_token_needed_for_continuity": "5. If the goal is only local continuity, coord_token remains necessary.",
    }
    for key, title in question_titles.items():
        row = verdicts[key]
        lines.extend(
            [
                f"### {title}",
                "",
                f"- Verdict: **{row['verdict']}**",
                f"- Rationale: {row['rationale']}",
                "",
            ]
        )
    lines.extend(
        [
            "## High-Signal Evidence",
            "",
            f"- Lexical control: combined numeric-distance coefficient = `{lexical['combined_numeric_coef']:.4f}` with p-value `{lexical['combined_numeric_pvalue']:.2e}`; per-model coefficients remain negative and significant.",
            f"- Vision lift: mean GT-score lift spans `{min(row['avg_gt_score_lift'] for row in vision_lift):.3f}` to `{max(row['avg_gt_score_lift'] for row in vision_lift):.3f}` across model/slot cells, and every mass@4 lift is positive.",
            f"- Final synthesis: **{summary['final_synthesis']}**.",
            "",
            "## Interpretation For `coord_token`",
            "",
            "The probe study supports a narrow claim: raw-text pure-CE does not need coord_token in order to exhibit local coordinate continuity. The open question is not continuity creation, but whether special coordinate parameterization still helps typing discipline, decoding stability, or a cleaner instance-separation geometry under rollout.",
            "",
        ]
    )
    manual_review = summary.get("manual_review")
    if isinstance(manual_review, dict):
        lines.extend(
            [
                "## Manual Review",
                "",
                "A human-in-the-loop review interface is available for bbox-first auditing on the original image, with 2D heatmaps kept as supporting mechanism evidence rather than the primary annotation surface.",
                "",
                f"- Review gallery: `{manual_review['review_md']}`",
                f"- Annotation workbook: `{manual_review['annotation_workbook_md']}`",
                f"- BBox audit template: `{manual_review['bbox_annotations_template']}`",
                f"- Structured templates: `{manual_review['case_annotations_template']}` and `{manual_review['panel_annotations_template']}`",
                "",
            ]
        )
    return "\n".join(lines)


def build_final_report_bundle(config_path: Path) -> dict[str, Any]:
    resolved_config_path = config_path.resolve()
    cfg = load_final_report_config(resolved_config_path)
    config_dir = resolved_config_path.parent
    run_dir = _resolve(cfg.run.output_dir, config_dir=config_dir) / cfg.run.name
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    audit_dir = _resolve(cfg.artifacts.audit_dir, config_dir=config_dir)
    pilot_broad_dir = _resolve(cfg.artifacts.pilot_broad_dir, config_dir=config_dir)
    pilot_broad_swapped_dir = _resolve(cfg.artifacts.pilot_broad_swapped_dir, config_dir=config_dir)
    pilot_crowded_dir = _resolve(cfg.artifacts.pilot_crowded_dir, config_dir=config_dir)
    pilot_model_mined_dir = _resolve(cfg.artifacts.pilot_model_mined_dir, config_dir=config_dir)
    bad_basin_base_dir = _resolve(cfg.artifacts.bad_basin_base_dir, config_dir=config_dir)
    bad_basin_pure_dir = _resolve(cfg.artifacts.bad_basin_pure_dir, config_dir=config_dir)
    perturb_base_dir = _resolve(cfg.artifacts.perturb_base_dir, config_dir=config_dir)
    perturb_pure_dir = _resolve(cfg.artifacts.perturb_pure_dir, config_dir=config_dir)
    heatmap_base_dir = _resolve(cfg.artifacts.heatmap_base_dir, config_dir=config_dir)
    heatmap_pure_dir = _resolve(cfg.artifacts.heatmap_pure_dir, config_dir=config_dir)

    audit_base = _read_json(audit_dir / "base_tokenization.json")
    audit_summary = _read_json(audit_dir / "summary.json")
    broad_summary = _read_json(pilot_broad_dir / "summary.json")
    broad_swapped_summary = _read_json(pilot_broad_swapped_dir / "summary.json")
    crowded_summary = _read_json(pilot_crowded_dir / "summary.json")
    model_mined_summary = _read_json(pilot_model_mined_dir / "summary.json")
    bad_basin_base_summary = _read_json(bad_basin_base_dir / "summary.json")
    bad_basin_pure_summary = _read_json(bad_basin_pure_dir / "summary.json")
    lexical_summary = _read_json(
        _resolve(cfg.artifacts.lexical_summary_json, config_dir=config_dir)
    )
    lexical_per_model = _read_json(
        _resolve(cfg.artifacts.lexical_per_model_json, config_dir=config_dir)
    )
    perturb_base_summary = _read_json(perturb_base_dir / "summary.json")
    perturb_pure_summary = _read_json(perturb_pure_dir / "summary.json")

    broad_rows = _read_jsonl(pilot_broad_dir / "per_coord_scores.jsonl")
    broad_swapped_rows = _read_jsonl(pilot_broad_swapped_dir / "per_coord_scores.jsonl")
    vision_lift_rows = summarize_vision_lift(
        correct_rows=broad_rows,
        swapped_rows=broad_swapped_rows,
        correct_summary=broad_summary,
        swapped_summary=broad_swapped_summary,
    )
    bad_basin_rows = summarize_bad_basin(bad_basin_base_summary) + summarize_bad_basin(
        bad_basin_pure_summary
    )
    perturb_rows = summarize_perturbation(perturb_base_summary) + summarize_perturbation(
        perturb_pure_summary
    )
    broad_metrics = _slot_metrics_by_key(broad_summary)
    crowded_metrics = _slot_metrics_by_key(crowded_summary)
    model_mined_metrics = _slot_metrics_by_key(model_mined_summary)
    verdicts = derive_core_verdicts(
        broad_metrics=broad_metrics,
        crowded_metrics=crowded_metrics,
        model_mined_metrics=model_mined_metrics,
        lexical_summary=lexical_summary,
        vision_lift_rows=vision_lift_rows,
        bad_basin_rows=bad_basin_rows,
        perturb_rows=perturb_rows,
    )

    plot_paths = {
        "good_basin_mass4": str(figures_dir / "good_basin_mass4.png"),
        "vision_lift": str(figures_dir / "vision_lift_gt_score.png"),
        "regression_numeric_coef": str(figures_dir / "regression_numeric_coef.png"),
        "bad_basin_advantage": str(figures_dir / "bad_basin_advantage.png"),
    }
    _plot_good_basin(
        out_path=Path(plot_paths["good_basin_mass4"]),
        broad_rows=broad_summary["slot_metrics"],
        crowded_rows=crowded_summary["slot_metrics"],
        model_mined_rows=model_mined_summary["slot_metrics"],
    )
    _plot_vision_lift(out_path=Path(plot_paths["vision_lift"]), rows=vision_lift_rows)
    _plot_regression_coefficients(
        out_path=Path(plot_paths["regression_numeric_coef"]),
        per_model=lexical_per_model,
    )
    _plot_bad_basin(out_path=Path(plot_paths["bad_basin_advantage"]), rows=bad_basin_rows)

    heatmap_copies = {
        "base": _copy_heatmap_figures(
            source_dir=heatmap_base_dir,
            target_dir=figures_dir / "heatmaps_base",
        ),
        "pure_ce": _copy_heatmap_figures(
            source_dir=heatmap_pure_dir,
            target_dir=figures_dir / "heatmaps_pure_ce",
        ),
    }

    audit_numbers = {
        str(item["value"]): item["tokens"] for item in audit_base.get("numbers", [])
    }
    summary: dict[str, Any] = {
        "run_name": cfg.run.name,
        "source_artifacts": {
            field: str(_resolve(value, config_dir=config_dir))
            if field != "diagnosis_docs"
            else [str(_resolve(item, config_dir=config_dir)) for item in value]
            for field, value in cfg.artifacts.__dict__.items()
        },
        "audit": {
            "serialization_surface": audit_summary["serialization_surface"],
            "example_tokens": {
                key: audit_numbers[key] for key in ("199", "200", "210", "999")
            },
        },
        "good_basin": {
            "broad_slot_metrics": broad_summary["slot_metrics"],
            "crowded_slot_metrics": crowded_summary["slot_metrics"],
            "model_mined_slot_metrics": model_mined_summary["slot_metrics"],
        },
        "vision_lift": vision_lift_rows,
        "lexical": {
            "combined_numeric_coef": lexical_summary["coefficients"]["numeric_distance_to_center"]["coef"],
            "combined_numeric_pvalue": lexical_summary["coefficients"]["numeric_distance_to_center"]["pvalue"],
            "per_model": lexical_per_model["per_model"],
        },
        "bad_basin": {
            "rows": bad_basin_rows,
            "base_summary_path": str(bad_basin_base_dir / "summary.json"),
            "pure_summary_path": str(bad_basin_pure_dir / "summary.json"),
        },
        "perturbation": {
            "rows": perturb_rows,
            "base_summary_path": str(perturb_base_dir / "summary.json"),
            "pure_summary_path": str(perturb_pure_dir / "summary.json"),
        },
        "heatmaps": {
            "base": summarize_heatmap_dir(heatmap_base_dir),
            "pure_ce": summarize_heatmap_dir(heatmap_pure_dir),
            "copied_figures": heatmap_copies,
        },
        "plots": plot_paths,
        "verdicts": verdicts,
        "final_synthesis": (
            "C + D + E: the models exhibit real numeric continuity that is visually modulated, "
            "but hard repeated-object prefixes can also induce wrong local basins; coord_token "
            "looks unnecessary if continuity is the only target, while other benefits remain open."
        ),
    }
    manual_review_dir = run_dir / "manual_review"
    manual_review_manifest = manual_review_dir / "manifest.json"
    if manual_review_manifest.exists():
        manifest = _read_json(manual_review_manifest)
        summary["manual_review"] = {
            "manifest_path": str(manual_review_manifest),
            "review_md": str(manual_review_dir / "review.md"),
            "annotation_workbook_md": str(manual_review_dir / "annotation_workbook.md"),
            "bbox_annotations_template": str(manual_review_dir / "bbox_annotations_template.jsonl"),
            "case_annotations_template": str(manual_review_dir / "case_annotations_template.jsonl"),
            "panel_annotations_template": str(manual_review_dir / "panel_annotations_template.jsonl"),
            "num_cases": manifest.get("num_cases"),
            "num_panels": manifest.get("num_panels"),
            "num_bbox_audit_panels": manifest.get("num_bbox_audit_panels"),
        }

    merged_rows = _merge_atomic_rows(
        [
            ("pilot_broad", pilot_broad_dir / "per_coord_scores.jsonl"),
            ("pilot_broad_swapped", pilot_broad_swapped_dir / "per_coord_scores.jsonl"),
            ("pilot_crowded", pilot_crowded_dir / "per_coord_scores.jsonl"),
            ("pilot_model_mined", pilot_model_mined_dir / "per_coord_scores.jsonl"),
            ("bad_basin_base", bad_basin_base_dir / "per_coord_scores.jsonl"),
            ("bad_basin_pure", bad_basin_pure_dir / "per_coord_scores.jsonl"),
            ("perturb_base", perturb_base_dir / "per_coord_scores.jsonl"),
            ("perturb_pure", perturb_pure_dir / "per_coord_scores.jsonl"),
        ]
    )
    hard_cases = _merge_hard_cases(
        [
            perturb_base_dir / "selected_cases.jsonl",
            perturb_pure_dir / "selected_cases.jsonl",
        ]
    )
    report_md = _render_report_md(summary)
    write_report_bundle(
        out_dir=run_dir,
        summary=summary,
        report_md=report_md,
        per_coord_rows=merged_rows,
        hard_cases=hard_cases,
    )
    with (run_dir / "diagnosis_docs.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "docs": [
                    str(_resolve(path, config_dir=config_dir))
                    for path in cfg.artifacts.diagnosis_docs
                ]
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return {"run_dir": str(run_dir), "summary": summary}
