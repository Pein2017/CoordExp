from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def write_outputs(
    out_dir: Path,
    *,
    coco_gt: Dict[str, Any] | None,
    coco_preds: List[Dict[str, Any]] | None,
    summary: Dict[str, Any],
    per_image: List[Dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if coco_gt is not None:
        (out_dir / "coco_gt.json").write_text(
            json.dumps(coco_gt, ensure_ascii=False), encoding="utf-8"
        )
    if coco_preds is not None:
        (out_dir / "coco_preds.json").write_text(
            json.dumps(coco_preds, ensure_ascii=False), encoding="utf-8"
        )
    metrics_payload = {
        "metrics": summary.get("metrics", {}),
        "counters": summary.get("counters", {}),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if coco_gt is not None and coco_preds is not None:
        (out_dir / "per_class.csv").write_text(
            "category,AP\n"
            + "\n".join(f"{k},{v}" for k, v in summary.get("per_class", {}).items()),
            encoding="utf-8",
        )
    (out_dir / "per_image.json").write_text(
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
