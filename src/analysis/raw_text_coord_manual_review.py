from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont
import yaml

from src.analysis.duplication_collapse_analysis import _choose_gt_next_candidate
from src.analysis.raw_text_coord_perturbation_probe import _controls_stub
from src.common.geometry import bbox_from_points, coerce_point_list, denorm_and_clamp
from src.common.geometry.object_geometry import extract_single_geometry
from src.vis.gt_vs_pred import canonicalize_gt_vs_pred_record, render_gt_vs_pred_review


@dataclass(frozen=True)
class ManualReviewRunConfig:
    final_bundle_dir: str
    review_subdir: str


@dataclass(frozen=True)
class ManualReviewSourceConfig:
    label: str
    path: str


@dataclass(frozen=True)
class ManualReviewConfig:
    run: ManualReviewRunConfig
    sources: tuple[ManualReviewSourceConfig, ...]


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"expected mapping in {path}")
    return data


def _resolve(path_str: str, *, config_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    repo_candidate = (Path(__file__).resolve().parents[2] / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (config_dir / path).resolve()


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
        row = json.loads(stripped)
        if not isinstance(row, dict):
            raise TypeError(f"expected object row in {path}")
        rows.append(row)
    return rows


def load_manual_review_config(config_path: Path) -> ManualReviewConfig:
    raw = _load_yaml(config_path)
    run_raw = raw.get("run")
    sources_raw = raw.get("sources")
    if not isinstance(run_raw, dict) or not isinstance(sources_raw, list):
        raise TypeError("manual review config requires run mapping and sources list")
    sources: list[ManualReviewSourceConfig] = []
    for item in sources_raw:
        if not isinstance(item, dict):
            raise TypeError("manual review sources entries must be mappings")
        sources.append(
            ManualReviewSourceConfig(label=str(item["label"]), path=str(item["path"]))
        )
    return ManualReviewConfig(
        run=ManualReviewRunConfig(
            final_bundle_dir=str(run_raw["final_bundle_dir"]),
            review_subdir=str(run_raw.get("review_subdir", "manual_review")),
        ),
        sources=tuple(sources),
    )


def _sanitize_case_id(case_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", case_id)


def _panel_sort_key(panel: dict[str, Any]) -> tuple[int, int, int]:
    model_rank = {"base": 0, "pure_ce": 1}.get(str(panel["model_alias"]), 99)
    variant_rank = {
        "baseline": 0,
        "replace_source_with_gt_next": 1,
        "source_x1y1_from_gt_next": 2,
        "interp_source_to_gt_next_0p5": 3,
        "drop_source": 4,
    }.get(str(panel["variant"]), 99)
    center_rank = {"pred": 0, "gt": 1}.get(str(panel["center_kind"]), 99)
    return (model_rank, variant_rank, center_rank)


def _panel_label(panel: dict[str, Any]) -> str:
    return (
        f"{panel['model_alias']} | {panel['variant']} | center={panel['center_kind']}"
    )


def _load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def build_contact_sheet(
    *,
    case_id: str,
    panels: Sequence[dict[str, Any]],
    output_path: Path,
    title: str,
    columns: int = 3,
) -> None:
    sorted_panels = sorted(panels, key=_panel_sort_key)
    font = _load_font()
    title_font = _load_font()
    tile_w = 360
    tile_h = 280
    label_h = 54
    padding = 16
    rows = math.ceil(len(sorted_panels) / columns)
    canvas_w = columns * tile_w + (columns + 1) * padding
    canvas_h = 64 + rows * (tile_h + label_h) + (rows + 1) * padding
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, 18), f"{title} | {case_id}", fill=(0, 0, 0), font=title_font)
    for index, panel in enumerate(sorted_panels):
        row = index // columns
        col = index % columns
        x0 = padding + col * (tile_w + padding)
        y0 = 64 + padding + row * (tile_h + label_h)
        image = Image.open(str(panel["figure_path"])).convert("RGB")
        try:
            image.thumbnail((tile_w, tile_h))
            paste_x = x0 + (tile_w - image.width) // 2
            paste_y = y0 + (tile_h - image.height) // 2
            canvas.paste(image, (paste_x, paste_y))
        finally:
            image.close()
        label = _panel_label(panel)
        draw.text((x0, y0 + tile_h + 8), label, fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _copy_selected_case_fields(row: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "case_id",
        "image_id",
        "line_idx",
        "slot",
        "selection_reason",
        "wrong_anchor_advantage_at_4",
        "pred_value",
        "gt_value",
        "source_object_index",
        "object_index",
        "model_alias",
        "top_desc",
        "pred_count",
        "max_desc_count",
        "same_desc_duplicate_pair_count",
    ]
    copied = {key: row.get(key) for key in keep if key in row}
    images = row.get("images")
    if isinstance(images, list) and images:
        copied["image_path"] = images[0]
    return copied


def _build_case_annotation_template(case_row: dict[str, Any], source_label: str) -> dict[str, Any]:
    return {
        "case_id": case_row["case_id"],
        "source_label": source_label,
        "slot": case_row.get("slot"),
        "pred_value": case_row.get("pred_value"),
        "gt_value": case_row.get("gt_value"),
        "wrong_anchor_advantage_at_4": case_row.get("wrong_anchor_advantage_at_4"),
        "human_summary": "",
        "is_good_basin_visible": "",
        "is_bad_basin_visible": "",
        "does_geometry_perturbation_shift_basin": "",
        "strongest_panels": [],
        "notes": "",
    }


def _build_panel_annotation_templates(
    panels: Sequence[dict[str, Any]],
    *,
    source_label: str,
) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    for panel in sorted(panels, key=_panel_sort_key):
        templates.append(
            {
                "panel_id": (
                    f"{source_label}::{panel['case_id']}::{panel['model_alias']}::"
                    f"{panel['variant']}::{panel['center_kind']}"
                ),
                "case_id": panel["case_id"],
                "source_label": source_label,
                "model_alias": panel["model_alias"],
                "variant": panel["variant"],
                "center_kind": panel["center_kind"],
                "figure_path": panel["figure_path"],
                "strength_rank": "",
                "human_label": "",
                "notes": "",
            }
        )
    return templates


def _normalize_desc(value: object) -> str:
    return str(value or "").strip().casefold()


def _object_desc(obj: dict[str, Any]) -> str:
    return str(obj.get("desc") or obj.get("category_name") or "").strip()


def _resolve_case_image_path(
    *,
    case_row: dict[str, Any],
    source_payload: dict[str, Any],
    source_jsonl_path: Path,
    selected_cases_jsonl_path: Path,
) -> str:
    for value, base_dir in (
        (case_row.get("images"), selected_cases_jsonl_path.parent),
        (case_row.get("image"), selected_cases_jsonl_path.parent),
        (source_payload.get("images"), source_jsonl_path.parent),
        (source_payload.get("image"), source_jsonl_path.parent),
    ):
        if isinstance(value, list) and value:
            path = Path(str(value[0]))
        elif isinstance(value, str) and value.strip():
            path = Path(value)
        else:
            continue
        if path.is_absolute():
            return str(path)
        return str((base_dir / path).resolve())
    raise FileNotFoundError("case_image_not_found")


def _load_bbox_audit_context(
    *,
    case_row: dict[str, Any],
    selected_cases_jsonl_path: Path,
    source_cache: dict[Path, list[dict[str, Any]]],
) -> dict[str, Any]:
    source_jsonl_path = Path(str(case_row["source_gt_vs_pred_jsonl"]))
    if source_jsonl_path not in source_cache:
        source_cache[source_jsonl_path] = _read_jsonl(source_jsonl_path)
    source_payload = source_cache[source_jsonl_path][int(case_row["line_idx"])]
    preds = list(source_payload.get("pred") or [])
    gt_objects = list(source_payload.get("gt") or [])
    object_index = int(case_row["object_index"])
    source_object_index = int(case_row["source_object_index"])
    pred_object = dict(preds[object_index])
    source_object = dict(preds[source_object_index])
    gt_next = _choose_gt_next_candidate(
        prefix_objects=[dict(obj) for obj in preds[:object_index]],
        gt_objects=gt_objects,
        source_object=source_object,
        cfg=_controls_stub(),
    )
    if gt_next is None:
        raise ValueError("unable_to_choose_gt_next_for_bbox_audit")
    desc_key = _normalize_desc(_object_desc(pred_object) or case_row.get("top_desc"))
    same_desc_gt = [
        dict(obj)
        for obj in gt_objects
        if _normalize_desc(_object_desc(dict(obj))) == desc_key
    ]
    if not same_desc_gt:
        same_desc_gt = [dict(obj) for obj in gt_objects]
    image_path = _resolve_case_image_path(
        case_row=case_row,
        source_payload=source_payload,
        source_jsonl_path=source_jsonl_path,
        selected_cases_jsonl_path=selected_cases_jsonl_path,
    )
    return {
        "image_path": image_path,
        "width": int(source_payload.get("width") or case_row.get("width") or 0),
        "height": int(source_payload.get("height") or case_row.get("height") or 0),
        "record_coord_mode": str(source_payload.get("coord_mode") or "pixel"),
        "same_desc_gt": same_desc_gt,
        "pred_object": pred_object,
        "source_object": source_object,
        "gt_next": dict(gt_next),
        "top_desc": _object_desc(pred_object) or str(case_row.get("top_desc") or ""),
    }


def _canonical_object(
    obj: dict[str, Any],
    *,
    index: int,
    width: int,
    height: int,
    record_coord_mode: str,
) -> dict[str, Any]:
    geom_type = "bbox_2d"
    raw_points: Sequence[Any] | None = None
    coord_mode_hint = str(
        obj.get("_coord_mode") or obj.get("coord_mode") or record_coord_mode or ""
    ).strip()

    if obj.get("bbox") is not None:
        raw_points = obj.get("bbox")
        geom_type = "bbox_2d"
    elif obj.get("points_norm1000") is not None:
        raw_points = obj.get("points_norm1000")
        geom_type = str(obj.get("geom_type") or obj.get("type") or "bbox_2d").strip()
        coord_mode_hint = "norm1000"
    else:
        geom_type, raw_points = extract_single_geometry(
            obj,
            allow_type_and_points=True,
            allow_nested_points=True,
            path=f"object[{index}]",
        )

    points_numeric, had_tokens = coerce_point_list(raw_points or [])
    if points_numeric is None:
        raise ValueError(f"failed to coerce geometry coordinates for object[{index}]")
    coord_mode = "norm1000" if (had_tokens or coord_mode_hint == "norm1000") else "pixel"
    points_px = denorm_and_clamp(
        points_numeric,
        width,
        height,
        coord_mode=coord_mode,
    )
    if geom_type == "bbox_2d":
        if len(points_px) != 4:
            raise ValueError(f"bbox_2d for object[{index}] must contain 4 values")
        x1, y1, x2, y2 = [int(value) for value in points_px]
    else:
        x1f, y1f, x2f, y2f = bbox_from_points(points_px)
        x1, y1, x2, y2 = (
            int(round(x1f)),
            int(round(y1f)),
            int(round(x2f)),
            int(round(y2f)),
        )
    x_lo, x_hi = (x1, x2) if x1 <= x2 else (x2, x1)
    y_lo, y_hi = (y1, y2) if y1 <= y2 else (y2, y1)
    return {
        "index": int(index),
        "desc": _object_desc(obj),
        "bbox_2d": [x_lo, y_lo, x_hi, y_hi],
    }


def _render_bbox_audit_case(
    *,
    case_row: dict[str, Any],
    source_label: str,
    review_dir: Path,
    selected_cases_jsonl_path: Path,
    source_cache: dict[Path, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    context = _load_bbox_audit_context(
        case_row=case_row,
        selected_cases_jsonl_path=selected_cases_jsonl_path,
        source_cache=source_cache,
    )
    case_id = str(case_row["case_id"])
    case_dir = review_dir / "bbox_audit" / f"{_sanitize_case_id(case_id)}__{source_label}"
    render_dir = case_dir / "rendered"
    render_dir.mkdir(parents=True, exist_ok=True)
    gt_objects = [
        _canonical_object(
            obj,
            index=index,
            width=context["width"],
            height=context["height"],
            record_coord_mode=context["record_coord_mode"],
        )
        for index, obj in enumerate(context["same_desc_gt"])
    ]
    role_objects = [
        ("baseline_pred", context["pred_object"]),
        ("source_anchor", context["source_object"]),
        ("chosen_gt_next", context["gt_next"]),
    ]
    canonical_records: list[dict[str, Any]] = []
    canonical_role_objects: list[tuple[str, dict[str, Any]]] = []
    for record_idx, (role, candidate) in enumerate(role_objects):
        canonical_candidate = _canonical_object(
            candidate,
            index=0,
            width=context["width"],
            height=context["height"],
            record_coord_mode=context["record_coord_mode"],
        )
        canonical_role_objects.append((role, canonical_candidate))
        raw_record = {
            "record_idx": record_idx,
            "source_kind": "raw_text_coord_bbox_audit",
            "image": context["image_path"],
            "width": context["width"],
            "height": context["height"],
            "coord_mode": "pixel",
            "gt": gt_objects,
            "pred": [canonical_candidate],
            "provenance": {
                "source_jsonl_dir": str(Path(context["image_path"]).parent),
                "case_id": case_id,
                "audit_role": role,
                "source_label": source_label,
            },
        }
        canonical_records.append(
            canonicalize_gt_vs_pred_record(
                raw_record,
                fallback_record_idx=record_idx,
                source_kind="raw_text_coord_bbox_audit",
                materialize_matching=True,
            )
        )
    records_path = case_dir / "records.jsonl"
    _write_jsonl(records_path, canonical_records)
    render_gt_vs_pred_review(records_path, out_dir=render_dir, limit=0)
    outputs: list[dict[str, Any]] = []
    for index, (role, canonical_candidate) in enumerate(canonical_role_objects):
        default_path = render_dir / f"vis_{index:04d}.png"
        final_path = render_dir / f"{role}.png"
        if default_path.exists():
            default_path.replace(final_path)
        outputs.append(
            {
                "role": role,
                "figure_path": str(final_path),
                "bbox_2d": canonical_candidate["bbox_2d"],
                "desc": canonical_candidate["desc"] or context["top_desc"],
            }
        )
    return outputs


def _build_bbox_annotation_templates(
    bbox_panels: Sequence[dict[str, Any]],
    *,
    case_id: str,
    source_label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for panel in bbox_panels:
        rows.append(
            {
                "bbox_audit_id": f"{source_label}::{case_id}::{panel['role']}",
                "case_id": case_id,
                "source_label": source_label,
                "role": panel["role"],
                "bbox_2d": panel["bbox_2d"],
                "desc": panel["desc"],
                "figure_path": panel["figure_path"],
                "bbox_judgment": "",
                "is_correct_instance": "",
                "is_gt_missing": "",
                "notes": "",
            }
        )
    return rows


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _render_review_md(
    *,
    cases: Sequence[dict[str, Any]],
    review_dir: Path,
    final_summary: dict[str, Any],
) -> str:
    verdicts = final_summary["verdicts"]
    lines = [
        "# Manual Review Interface",
        "",
        "This workbook is for the remaining human-in-the-loop interpretation layer.",
        "",
        "## What To Do",
        "",
        "1. Start with the bbox-audit figures on the original image.",
        "2. For each candidate box, decide only four things: `correct`, `wrong-instance`, `GT-missing`, or `unclear`.",
        "3. Treat the heatmap contact sheets as mechanism evidence only after the bbox judgment is done.",
        "4. Fill `annotation_workbook.md` in plain language.",
        "5. Optionally fill the structured JSONL templates if you want machine-readable summaries.",
        "",
        "## What These Figures Are",
        "",
        "- `bbox_audit/*.png`: original image plus GT-vs-candidate overlay. This is the main human task.",
        "- `contact_sheets/*.png`: score-landscape summaries over candidate coordinates. These are not raw attention maps.",
        "- If we add attention later, it should be read only as an auxiliary 'where the model looked' overlay, not the primary judgment surface.",
        "",
        "## Current Machine Verdicts",
        "",
    ]
    for key, row in verdicts.items():
        lines.append(f"- `{key}`: **{row['verdict']}**")
    lines.extend(["", "## Cases", ""])
    for case in cases:
        contact_sheet_path = case["contact_sheet_path"]
        lines.extend(
            [
                f"### {case['case_id']} ({case['source_label']})",
                "",
                f"- Slot: `{case['slot']}`",
                f"- Pred vs GT: `{case['pred_value']}` vs `{case['gt_value']}`",
                f"- Wrong-anchor advantage@4: `{case['wrong_anchor_advantage_at_4']}`",
                f"- Selection reason: {case['selection_reason']}",
            ]
        )
        image_path = case.get("image_path")
        if image_path:
            lines.append(f"- Source image: [image]({image_path})")
        bbox_panels = case.get("bbox_panels") or []
        if bbox_panels:
            lines.extend(["", "BBox audit panels:"])
            for panel in bbox_panels:
                lines.append(
                    f"- `{panel['role']}` bbox `{panel['bbox_2d']}`: [figure]({panel['figure_path']})"
                )
        lines.extend(
            [
                "",
                f"![{case['case_id']} contact sheet]({contact_sheet_path})",
                "",
                "Panel links:",
            ]
        )
        for panel in case["panels"]:
            lines.append(
                f"- `{panel['model_alias']} | {panel['variant']} | {panel['center_kind']}`: "
                f"[panel]({panel['figure_path']})"
            )
        lines.extend(
            [
                "",
                "Suggested human prompts:",
                "- For `baseline_pred`: is the bbox actually correct on the image?",
                "- If `baseline_pred` looks valid but unmatched, is this a GT-miss candidate?",
                "- Does `baseline_pred` land on the wrong same-desc instance?",
                "- Compare `baseline_pred` vs `source_anchor`: do they look like the same local instance family?",
                "- Which panel most clearly shows a GT-centered good basin?",
                "- Which panel most clearly shows a wrong local basin around the predicted anchor?",
                "- Which perturbation most clearly moves the basin, and in which direction?",
                "- Does the base-vs-pure comparison change your interpretation of the mechanism?",
                "",
            ]
        )
    return "\n".join(lines)


def _render_annotation_workbook(cases: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Annotation Workbook",
        "",
        "Fill the sections below with your manual interpretation. Start from bbox correctness, then use the heatmaps only as supporting mechanism evidence.",
        "",
    ]
    for case in cases:
        lines.extend(
            [
                f"## {case['case_id']} ({case['source_label']})",
                "",
                f"- Slot: `{case['slot']}`",
                f"- Pred vs GT: `{case['pred_value']}` vs `{case['gt_value']}`",
                f"- Wrong-anchor advantage@4: `{case['wrong_anchor_advantage_at_4']}`",
                "",
                "### Interpretation",
                "",
                "- baseline_pred judgment (correct / wrong-instance / GT-missing / unclear):",
                "- source_anchor judgment:",
                "- chosen_gt_next judgment:",
                "- Overall reading:",
                "- Strongest panel(s):",
                "- Evidence of good basin:",
                "- Evidence of bad basin:",
                "- Effect of geometry perturbation:",
                "- Base vs pure_ce comparison:",
                "- Open questions / doubts:",
                "",
            ]
        )
    return "\n".join(lines)


def build_manual_review_bundle(config_path: Path) -> dict[str, Any]:
    resolved_config_path = config_path.resolve()
    cfg = load_manual_review_config(resolved_config_path)
    config_dir = resolved_config_path.parent
    final_bundle_dir = _resolve(cfg.run.final_bundle_dir, config_dir=config_dir)
    review_dir = final_bundle_dir / cfg.run.review_subdir
    review_dir.mkdir(parents=True, exist_ok=True)
    final_summary = _read_json(final_bundle_dir / "summary.json")
    contact_sheets_dir = review_dir / "contact_sheets"
    cases_for_review: list[dict[str, Any]] = []
    case_annotation_rows: list[dict[str, Any]] = []
    panel_annotation_rows: list[dict[str, Any]] = []
    bbox_annotation_rows: list[dict[str, Any]] = []
    source_cache: dict[Path, list[dict[str, Any]]] = {}

    for source in cfg.sources:
        source_dir = _resolve(source.path, config_dir=config_dir)
        selected_cases_jsonl_path = source_dir / "selected_cases.jsonl"
        selected_cases = _read_jsonl(selected_cases_jsonl_path)
        heatmap_rows = _read_jsonl(source_dir / "heatmaps.jsonl")
        panels_by_case: dict[str, list[dict[str, Any]]] = {}
        for row in heatmap_rows:
            panels_by_case.setdefault(str(row["case_id"]), []).append(row)
        for selected_case in selected_cases:
            case_id = str(selected_case["case_id"])
            case_panels = panels_by_case.get(case_id, [])
            if not case_panels:
                continue
            contact_sheet_path = (
                contact_sheets_dir / f"{_sanitize_case_id(case_id)}__{source.label}.png"
            )
            build_contact_sheet(
                case_id=case_id,
                panels=case_panels,
                output_path=contact_sheet_path,
                title=source.label,
            )
            case_row = {
                **_copy_selected_case_fields(selected_case),
                "source_label": source.label,
                "contact_sheet_path": str(contact_sheet_path),
                "panels": sorted(case_panels, key=_panel_sort_key),
            }
            if "source_gt_vs_pred_jsonl" in selected_case:
                bbox_panels = _render_bbox_audit_case(
                    case_row=selected_case,
                    source_label=source.label,
                    review_dir=review_dir,
                    selected_cases_jsonl_path=selected_cases_jsonl_path,
                    source_cache=source_cache,
                )
                case_row["bbox_panels"] = bbox_panels
                bbox_annotation_rows.extend(
                    _build_bbox_annotation_templates(
                        bbox_panels,
                        case_id=case_id,
                        source_label=source.label,
                    )
                )
            cases_for_review.append(case_row)
            case_annotation_rows.append(
                _build_case_annotation_template(selected_case, source.label)
            )
            panel_annotation_rows.extend(
                _build_panel_annotation_templates(case_panels, source_label=source.label)
            )

    manifest = {
        "final_bundle_dir": str(final_bundle_dir),
        "review_dir": str(review_dir),
        "num_cases": len(cases_for_review),
        "num_panels": len(panel_annotation_rows),
        "num_bbox_audit_panels": len(bbox_annotation_rows),
        "cases": [
            {
                "case_id": case["case_id"],
                "source_label": case["source_label"],
                "contact_sheet_path": case["contact_sheet_path"],
                "num_panels": len(case["panels"]),
                "num_bbox_audit_panels": len(case.get("bbox_panels") or []),
            }
            for case in cases_for_review
        ],
    }
    _write_json(review_dir / "manifest.json", manifest)
    _write_jsonl(review_dir / "case_annotations_template.jsonl", case_annotation_rows)
    _write_jsonl(review_dir / "panel_annotations_template.jsonl", panel_annotation_rows)
    _write_jsonl(review_dir / "bbox_annotations_template.jsonl", bbox_annotation_rows)
    (review_dir / "review.md").write_text(
        _render_review_md(cases=cases_for_review, review_dir=review_dir, final_summary=final_summary),
        encoding="utf-8",
    )
    (review_dir / "annotation_workbook.md").write_text(
        _render_annotation_workbook(cases_for_review),
        encoding="utf-8",
    )
    return {
        "review_dir": str(review_dir),
        "manifest": manifest,
    }
