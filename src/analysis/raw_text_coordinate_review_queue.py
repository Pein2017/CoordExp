from __future__ import annotations

import html
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from src.common.paths import resolve_image_path_best_effort, resolve_image_path_strict
from src.vis.gt_vs_pred import canonicalize_gt_vs_pred_record, render_gt_vs_pred_review


def build_review_queue_rows(
    *,
    shortlist: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in shortlist:
        rows.append(
            {
                "case_uid": row["case_uid"],
                "bucket": row["review_bucket"],
                "priority": row["selection_rank"],
                "status": "unreviewed",
                "model_focus": row["model_alias"],
                "source_gt_vs_pred_jsonl": row["source_gt_vs_pred_jsonl"],
                "line_idx": row["line_idx"],
                "object_index": row.get("object_index"),
                "source_object_index": row.get("source_object_index"),
                "gt_idx": row.get("gt_idx"),
                "bbox_judgment": "",
                "mechanism_label": "",
                "best_evidence": "",
                "confidence": "",
                "notes": "",
                "asset_links": "",
            }
        )
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise TypeError(f"expected object row in {path}")
            rows.append(payload)
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _sanitize_case_uid(case_uid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", case_uid)


def _case_detail(
    *,
    row: dict[str, object],
    source_record: dict[str, Any],
) -> str:
    if row["review_bucket"] == "FP":
        pred_rows = list(source_record.get("pred") or [])
        object_index = row.get("object_index")
        desc = ""
        if isinstance(object_index, int) and 0 <= int(object_index) < len(pred_rows):
            desc = str(pred_rows[int(object_index)].get("desc") or "").strip()
        return (
            f"duplicate burst | desc={desc or 'unknown'} | "
            f"source={row.get('source_object_index')} -> onset={row.get('object_index')}"
        )
    gt_rows = list(source_record.get("gt") or [])
    gt_idx = row.get("gt_idx")
    desc = ""
    if isinstance(gt_idx, int) and 0 <= int(gt_idx) < len(gt_rows):
        desc = str(gt_rows[int(gt_idx)].get("desc") or "").strip()
    return f"labeled FN | desc={desc or 'unknown'} | gt_idx={gt_idx}"


def materialize_review_gallery(
    *,
    shortlist: list[dict[str, object]],
    output_dir: Path,
    title: str = "Raw-Text Coordinate Mechanism Review",
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_cache: dict[Path, list[dict[str, Any]]] = {}
    gallery_rows: list[dict[str, object]] = []
    for row in shortlist:
        source_path = Path(str(row["source_gt_vs_pred_jsonl"]))
        if source_path not in source_cache:
            source_cache[source_path] = _read_jsonl(source_path)
        line_idx = int(row["line_idx"])
        source_record = dict(source_cache[source_path][line_idx])
        canonical = canonicalize_gt_vs_pred_record(
            source_record,
            fallback_record_idx=line_idx,
            source_kind="mechanism_review",
        )
        repo_root = _shared_repo_root(source_path)
        root_image_dir = repo_root / "public_data/coco/rescale_32_1024_bbox_max60"
        resolved_image = resolve_image_path_strict(
            str(canonical.get("image") or ""),
            jsonl_dir=source_path.parent,
            root_image_dir=root_image_dir,
        )
        if resolved_image is None:
            resolved_image = resolve_image_path_best_effort(
                str(canonical.get("image") or ""),
                jsonl_dir=source_path.parent,
                root_image_dir=root_image_dir,
            )
        canonical["image"] = str(resolved_image)
        provenance = dict(canonical.get("provenance") or {})
        provenance.setdefault("source_jsonl_dir", str(source_path.parent))
        canonical["provenance"] = provenance

        case_dir = output_dir / "cases" / _sanitize_case_uid(str(row["case_uid"]))
        case_dir.mkdir(parents=True, exist_ok=True)
        case_jsonl_path = case_dir / "record.jsonl"
        case_jsonl_path.write_text(
            json.dumps(canonical, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        render_dir = case_dir / "rendered"
        render_gt_vs_pred_review(
            case_jsonl_path,
            out_dir=render_dir,
            limit=1,
        )
        review_image_path = render_dir / "vis_0000.png"
        gallery_rows.append(
            {
                **row,
                "detail": _case_detail(row=row, source_record=source_record),
                "review_image": str(review_image_path.relative_to(output_dir)),
                "record_jsonl": str(case_jsonl_path.relative_to(output_dir)),
            }
        )

    write_review_gallery_html(
        rows=gallery_rows,
        output_path=output_dir / "review.html",
        title=title,
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "title": title,
            "rows": gallery_rows,
        },
    )
    return gallery_rows


def write_review_gallery_html(
    *,
    rows: list[dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    def _bucket_rows(bucket: str) -> list[dict[str, object]]:
        return [row for row in rows if str(row["review_bucket"]) == bucket]

    def _render_cards(bucket_rows: list[dict[str, object]]) -> str:
        cards: list[str] = []
        for row in bucket_rows:
            cards.append(
                """
<article class="card">
  <div class="meta">
    <div class="badge {bucket_class}">{bucket}</div>
    <div class="case-id">{case_uid}</div>
    <div class="detail">{detail}</div>
    <div class="subtle">model={model_alias} | image_id={image_id} | line_idx={line_idx}</div>
  </div>
  <a class="image-link" href="{review_image}">
    <img src="{review_image}" alt="{case_uid}" loading="lazy" />
  </a>
</article>
                """.strip().format(
                    bucket_class="fp" if row["review_bucket"] == "FP" else "fn",
                    bucket=html.escape(str(row["review_bucket"])),
                    case_uid=html.escape(str(row["case_uid"])),
                    detail=html.escape(str(row["detail"])),
                    model_alias=html.escape(str(row["model_alias"])),
                    image_id=html.escape(str(row["image_id"])),
                    line_idx=html.escape(str(row["line_idx"])),
                    review_image=html.escape(str(row["review_image"])),
                )
            )
        return "\n".join(cards) if cards else '<p class="empty">No cases in this bucket.</p>'

    fp_rows = _bucket_rows("FP")
    fn_rows = _bucket_rows("FN")
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
      h1, h2 {{
        margin: 0 0 12px;
      }}
      .intro {{
        max-width: 960px;
        margin-bottom: 28px;
        line-height: 1.5;
      }}
      .section {{
        margin-top: 32px;
      }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 18px;
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
      }}
      .badge.fp {{
        background: #ffe7d6;
        color: #9f3a00;
      }}
      .badge.fn {{
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
      .subtle {{
        margin-top: 6px;
        color: #666;
        font-size: 13px;
      }}
      .image-link {{
        display: block;
        padding: 0 18px 18px;
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
      Review the rendered GT-vs-Pred cards directly in this page. Each card opens the full-size image.
      Reply with the case ids you want to label, or the cases that look like true FP duplicate bursts versus real-but-unlabeled objects.
    </p>
    <section class="section">
      <h2>FP Cases ({len(fp_rows)})</h2>
      <div class="grid">
        {_render_cards(fp_rows)}
      </div>
    </section>
    <section class="section">
      <h2>FN Cases ({len(fn_rows)})</h2>
      <div class="grid">
        {_render_cards(fn_rows)}
      </div>
    </section>
  </body>
</html>
""",
        encoding="utf-8",
    )
