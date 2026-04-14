from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

CXCY_LOGW_LOGH_CONSTANT_PRED_SCORE_SOURCE = "cxcy_logw_logh_constant"
CXCY_LOGW_LOGH_CONSTANT_PRED_SCORE_VERSION = 1
CXCY_LOGW_LOGH_CONSTANT_SCORE = 1.0


def with_stem_suffix(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def resolve_guarded_prediction_artifact_path(
    *,
    out_dir: Path,
    scored_input: bool,
) -> Path:
    name = "gt_vs_pred_scored_guarded.jsonl" if scored_input else "gt_vs_pred_guarded.jsonl"
    return out_dir / name


def resolve_duplicate_guard_report_path(*, out_dir: Path) -> Path:
    return out_dir / "duplicate_guard_report.json"


def resolve_matches_artifact_path(
    *,
    out_dir: Path,
    iou_thr_key: str | None,
    name_suffix: str = "",
) -> Path:
    if iou_thr_key:
        return out_dir / f"matches@{iou_thr_key}{name_suffix}.jsonl"
    return out_dir / f"matches{name_suffix}.jsonl"


def write_jsonl_records(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def with_constant_scores(
    *,
    records: Sequence[Mapping[str, Any]],
    pred_score_source: str,
    pred_score_version: int,
    constant_score: float,
) -> List[Dict[str, Any]]:
    scored_rows: List[Dict[str, Any]] = []
    score = float(constant_score)
    for row in records:
        scored_row = dict(row)
        scored_row["pred_score_source"] = str(pred_score_source)
        scored_row["pred_score_version"] = int(pred_score_version)
        preds_raw = row.get("pred")
        preds_out: List[Dict[str, Any]] = []
        if isinstance(preds_raw, list):
            for pred in preds_raw:
                if not isinstance(pred, Mapping):
                    continue
                pred_out = dict(pred)
                pred_out["score"] = score
                preds_out.append(pred_out)
        scored_row["pred"] = preds_out
        scored_rows.append(scored_row)
    return scored_rows


def write_outputs(
    out_dir: Path,
    *,
    coco_gt: Dict[str, Any] | None,
    coco_preds: List[Dict[str, Any]] | None,
    summary: Dict[str, Any],
    per_image: List[Dict[str, Any]],
    name_suffix: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if coco_gt is not None:
        with_stem_suffix(out_dir / "coco_gt.json", name_suffix).write_text(
            json.dumps(coco_gt, ensure_ascii=False), encoding="utf-8"
        )
    if coco_preds is not None:
        with_stem_suffix(out_dir / "coco_preds.json", name_suffix).write_text(
            json.dumps(coco_preds, ensure_ascii=False), encoding="utf-8"
        )
    metrics_payload = {
        "metrics": summary.get("metrics", {}),
        "counters": summary.get("counters", {}),
    }
    with_stem_suffix(out_dir / "metrics.json", name_suffix).write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if coco_gt is not None and coco_preds is not None:
        with_stem_suffix(out_dir / "per_class.csv", name_suffix).write_text(
            "category,AP\n"
            + "\n".join(f"{k},{v}" for k, v in summary.get("per_class", {}).items()),
            encoding="utf-8",
        )
    with_stem_suffix(out_dir / "per_image.json", name_suffix).write_text(
        json.dumps(per_image, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_per_image_report(
    gt_samples: List[Any],
    pred_samples: List[Tuple[int, List[Dict[str, Any]]]],
    invalid_preds: Dict[int, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    report: List[Dict[str, Any]] = []
    pred_lookup = {img_id: preds for img_id, preds in pred_samples}
    for sample in gt_samples:
        preds = pred_lookup.get(sample.image_id, [])
        report.append(
            {
                "image_id": sample.image_id,
                "file_name": sample.file_name,
                "gt_count": len(sample.objects),
                "pred_count": len(preds),
                "invalid_gt": sample.invalid,
                "invalid_pred": invalid_preds.get(sample.image_id, []),
            }
        )
    return report
