from __future__ import annotations


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
                "bbox_judgment": "",
                "mechanism_label": "",
                "best_evidence": "",
                "confidence": "",
                "notes": "",
                "asset_links": "",
            }
        )
    return rows
