"""Visual review bundle for raw-text FN suppression cases."""

from __future__ import annotations

import html
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.common.paths import resolve_image_path_best_effort, resolve_image_path_strict
from src.vis.gt_vs_pred import canonicalize_gt_vs_pred_record, render_gt_vs_pred_review


def choose_best_oracle_run(
    *,
    case_row: Mapping[str, object],
) -> dict[str, object]:
    oracle_runs = [
        dict(run)
        for run in list(case_row.get("oracle_runs") or [])
        if isinstance(run, dict)
    ]
    if not oracle_runs:
        raise ValueError("fn_review_missing_oracle_runs")

    def _sort_key(run: Mapping[str, object]) -> tuple[int, int, float, str]:
        return (
            1 if bool(run.get("full_hit")) else 0,
            1 if bool(run.get("loc_hit")) else 0,
            float(run.get("iou") or 0.0),
            str(run.get("label") or ""),
        )

    return max(oracle_runs, key=_sort_key)


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


def _sanitize_case_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value).strip("-")


def _materialize_record_render(
    *,
    source_record: dict[str, object],
    record_idx: int,
    source_jsonl_path: Path,
    gallery_root: Path,
    case_panel_dir: Path,
    panel_name: str,
) -> tuple[str, str]:
    canonical = canonicalize_gt_vs_pred_record(
        source_record,
        fallback_record_idx=record_idx,
        source_kind="fn_review",
    )
    repo_root = _shared_repo_root(source_jsonl_path)
    root_image_dir = repo_root / "public_data/coco/rescale_32_1024_bbox_max60"
    resolved_image = resolve_image_path_strict(
        str(canonical.get("image") or ""),
        jsonl_dir=source_jsonl_path.parent,
        root_image_dir=root_image_dir,
    )
    if resolved_image is None:
        resolved_image = resolve_image_path_best_effort(
            str(canonical.get("image") or ""),
            jsonl_dir=source_jsonl_path.parent,
            root_image_dir=root_image_dir,
        )
    canonical["image"] = str(resolved_image)
    provenance = dict(canonical.get("provenance") or {})
    provenance.setdefault("source_jsonl_dir", str(source_jsonl_path.parent))
    canonical["provenance"] = provenance

    panel_dir = case_panel_dir / panel_name
    panel_dir.mkdir(parents=True, exist_ok=True)
    record_jsonl_path = panel_dir / "record.jsonl"
    record_jsonl_path.write_text(json.dumps(canonical, ensure_ascii=False) + "\n", encoding="utf-8")
    render_dir = panel_dir / "rendered"
    render_gt_vs_pred_review(record_jsonl_path, out_dir=render_dir, limit=1)
    review_image_path = render_dir / "vis_0000.png"
    return (
        str(review_image_path.relative_to(gallery_root)),
        str(record_jsonl_path.relative_to(gallery_root)),
    )


def _mechanism_hint(margins_by_model: Mapping[str, Mapping[str, object]]) -> str:
    base_stop = bool(
        (margins_by_model.get("base_only") or {}).get("stop_pressure_signature", False)
    )
    adapter_stop = bool(
        (margins_by_model.get("base_plus_adapter") or {}).get("stop_pressure_signature", False)
    )
    if base_stop and adapter_stop:
        return "both_models_stop_pressure"
    if base_stop or adapter_stop:
        return "mixed_stop_pressure"
    return "weak_stop_pressure_signal"


def materialize_fn_review_gallery(
    *,
    selected_cases: Sequence[dict[str, object]],
    margin_rows: Sequence[dict[str, object]],
    baseline_gt_vs_pred_path: Path,
    output_dir: Path,
    title: str = "Raw-Text FN Review",
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows = _read_jsonl(baseline_gt_vs_pred_path)
    margin_map: dict[str, dict[str, dict[str, object]]] = {}
    for row in margin_rows:
        case_id = str(row["case_id"])
        model_alias = str(row["model_alias"])
        margin_map.setdefault(case_id, {})[model_alias] = dict(row)

    oracle_cache: dict[Path, list[dict[str, object]]] = {}
    gallery_rows: list[dict[str, object]] = []
    for case_row in selected_cases:
        record_idx = int(case_row["record_idx"])
        case_id = str(case_row["case_id"])
        oracle_run = choose_best_oracle_run(case_row=case_row)
        oracle_path = _resolve_path(str(oracle_run["pred_jsonl"]), anchor=baseline_gt_vs_pred_path)
        if oracle_path not in oracle_cache:
            oracle_cache[oracle_path] = _read_jsonl(oracle_path)

        case_dir = output_dir / "cases" / _sanitize_case_id(case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        baseline_review_image, baseline_record_jsonl = _materialize_record_render(
            source_record=dict(baseline_rows[record_idx]),
            record_idx=record_idx,
            source_jsonl_path=baseline_gt_vs_pred_path,
            gallery_root=output_dir,
            case_panel_dir=case_dir / "baseline",
            panel_name="bundle",
        )
        oracle_review_image, oracle_record_jsonl = _materialize_record_render(
            source_record=dict(oracle_cache[oracle_path][record_idx]),
            record_idx=record_idx,
            source_jsonl_path=oracle_path,
            gallery_root=output_dir,
            case_panel_dir=case_dir / "oracle",
            panel_name="bundle",
        )

        case_margins = margin_map.get(case_id, {})
        base_margin = dict(case_margins.get("base_only") or {})
        adapter_margin = dict(case_margins.get("base_plus_adapter") or {})
        gallery_rows.append(
            {
                "case_id": case_id,
                "image_id": int(case_row["image_id"]),
                "record_idx": record_idx,
                "gt_idx": int(case_row["gt_idx"]),
                "gt_desc": str(case_row["gt_desc"]),
                "gt_bbox": list(case_row["gt_bbox"]),
                "recover_fraction_full": float(case_row.get("recover_fraction_full") or 0.0),
                "teacher_forced_support": float(case_row.get("teacher_forced_support") or 0.0),
                "proposal_support": float(case_row.get("proposal_support") or 0.0),
                "competitor_margin": float(case_row.get("competitor_margin") or 0.0),
                "oracle_label": str(oracle_run.get("label") or ""),
                "oracle_iou": float(oracle_run.get("iou") or 0.0),
                "baseline_review_image": baseline_review_image,
                "oracle_review_image": oracle_review_image,
                "baseline_record_jsonl": baseline_record_jsonl,
                "oracle_record_jsonl": oracle_record_jsonl,
                "base_only_continue_minus_eos_sum_logprob": base_margin.get(
                    "continue_minus_eos_sum_logprob"
                ),
                "base_only_continue_minus_eos_mean_logprob": base_margin.get(
                    "continue_minus_eos_mean_logprob"
                ),
                "base_only_stop_pressure_signature": bool(
                    base_margin.get("stop_pressure_signature", False)
                ),
                "base_plus_adapter_continue_minus_eos_sum_logprob": adapter_margin.get(
                    "continue_minus_eos_sum_logprob"
                ),
                "base_plus_adapter_continue_minus_eos_mean_logprob": adapter_margin.get(
                    "continue_minus_eos_mean_logprob"
                ),
                "base_plus_adapter_stop_pressure_signature": bool(
                    adapter_margin.get("stop_pressure_signature", False)
                ),
                "mechanism_hint": _mechanism_hint(case_margins),
            }
        )

    write_fn_review_gallery_html(
        rows=gallery_rows,
        output_path=output_dir / "review.html",
        title=title,
    )
    _write_json(output_dir / "manifest.json", {"title": title, "rows": gallery_rows})
    return gallery_rows


def write_fn_review_gallery_html(
    *,
    rows: Sequence[dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    def _metric_text(row: Mapping[str, object], prefix: str) -> str:
        stop = bool(row.get(f"{prefix}_stop_pressure_signature"))
        sum_delta = row.get(f"{prefix}_continue_minus_eos_sum_logprob")
        mean_delta = row.get(f"{prefix}_continue_minus_eos_mean_logprob")
        return (
            f"{prefix}: stop={'yes' if stop else 'no'} | "
            f"sum={sum_delta:.3f} | mean={mean_delta:.3f}"
            if isinstance(sum_delta, (int, float)) and isinstance(mean_delta, (int, float))
            else f"{prefix}: unavailable"
        )

    def _render_cards() -> str:
        cards: list[str] = []
        for row in rows:
            cards.append(
                """
<article class="card">
  <div class="meta">
    <div class="badge">{mechanism_hint}</div>
    <div class="case-id">{case_id}</div>
    <div class="detail">desc={gt_desc} | image_id={image_id} | gt_idx={gt_idx} | oracle={oracle_label} | iou={oracle_iou:.3f}</div>
    <div class="subtle">recover_fraction_full={recover_fraction_full:.2f} | teacher_forced_support={teacher_forced_support:.3f} | competitor_margin={competitor_margin:.3f}</div>
    <div class="metric">{base_metric}</div>
    <div class="metric">{adapter_metric}</div>
  </div>
  <div class="panels">
    <a class="panel" href="{baseline_review_image}">
      <div class="panel-label">Baseline Miss</div>
      <img src="{baseline_review_image}" alt="{case_id} baseline" loading="lazy" />
    </a>
    <a class="panel" href="{oracle_review_image}">
      <div class="panel-label">Recovered Oracle</div>
      <img src="{oracle_review_image}" alt="{case_id} oracle" loading="lazy" />
    </a>
  </div>
</article>
                """.strip().format(
                    mechanism_hint=html.escape(str(row["mechanism_hint"])),
                    case_id=html.escape(str(row["case_id"])),
                    gt_desc=html.escape(str(row["gt_desc"])),
                    image_id=int(row["image_id"]),
                    gt_idx=int(row["gt_idx"]),
                    oracle_label=html.escape(str(row["oracle_label"])),
                    oracle_iou=float(row["oracle_iou"]),
                    recover_fraction_full=float(row["recover_fraction_full"]),
                    teacher_forced_support=float(row["teacher_forced_support"]),
                    competitor_margin=float(row["competitor_margin"]),
                    base_metric=html.escape(_metric_text(row, "base_only")),
                    adapter_metric=html.escape(_metric_text(row, "base_plus_adapter")),
                    baseline_review_image=html.escape(str(row["baseline_review_image"])),
                    oracle_review_image=html.escape(str(row["oracle_review_image"])),
                )
            )
        return "\n".join(cards) if cards else '<p class="empty">No FN review cases.</p>'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(title)}</title>
    <style>
      body {{
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        margin: 0;
        padding: 24px;
        background: #f5f4ee;
        color: #171717;
      }}
      h1 {{
        margin: 0 0 12px;
      }}
      .intro {{
        max-width: 1080px;
        margin-bottom: 28px;
        line-height: 1.5;
      }}
      .grid {{
        display: grid;
        grid-template-columns: 1fr;
        gap: 20px;
      }}
      .card {{
        background: white;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        overflow: hidden;
      }}
      .meta {{
        padding: 16px 18px 12px;
      }}
      .badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        background: #dff2ff;
        color: #005a8d;
      }}
      .case-id {{
        margin-top: 10px;
        font-family: Menlo, Consolas, monospace;
        font-size: 12px;
        color: #4d4d4d;
      }}
      .detail {{
        margin-top: 8px;
        font-size: 14px;
        font-weight: 600;
      }}
      .subtle, .metric {{
        margin-top: 6px;
        color: #555;
        font-size: 13px;
      }}
      .panels {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 14px;
        padding: 0 18px 18px;
      }}
      .panel {{
        display: block;
        text-decoration: none;
        color: inherit;
      }}
      .panel-label {{
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #666;
        margin-bottom: 8px;
      }}
      img {{
        display: block;
        width: 100%;
        height: auto;
        border-radius: 12px;
        border: 1px solid #e5e2d8;
      }}
      .empty {{
        color: #666;
      }}
    </style>
  </head>
  <body>
    <h1>{html.escape(title)}</h1>
    <p class="intro">
      Each case shows the baseline miss and the best recovered oracle hit side by side.
      Use this page to judge whether the recovered object really corresponds to the labeled COCO GT target
      and whether the stop-pressure signal looks believable for that scene.
    </p>
    <section class="grid">
      {_render_cards()}
    </section>
  </body>
</html>
""",
        encoding="utf-8",
    )
