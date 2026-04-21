from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CaseRow:
    case_uid: str
    model_alias: str
    image_id: int
    line_idx: int
    record_idx: int
    source_gt_vs_pred_jsonl: str
    bucket: str
    review_bucket: str
    source_object_index: int | None
    onset_object_index: int | None
    gt_idx: int | None
    selection_rank: int
    serializer_surface: str


def _case_uid(row: dict[str, object], bucket: str) -> str:
    base_uid = (
        f"{row['model_alias']}:{row['image_id']}:{row['line_idx']}:"
        f"{row['record_idx']}:{bucket}"
    )
    if bucket == "first_burst_onset":
        return (
            f"{base_uid}:src{row['source_object_index']}:"
            f"on{row['onset_object_index']}"
        )
    if bucket == "labeled_fn":
        return f"{base_uid}:gt{row['gt_idx']}"
    return base_uid


def _serializer_surface_priority(surface: str) -> int:
    if surface == "model_native":
        return 0
    if surface == "pretty_inline":
        return 1
    return 2


def _representative_case_rows(rows: list[CaseRow]) -> list[CaseRow]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row.selection_rank,
            row.model_alias,
            row.image_id,
            row.line_idx,
            _serializer_surface_priority(row.serializer_surface),
            row.case_uid,
        ),
    )
    representatives: list[CaseRow] = []
    seen_case_uids: set[str] = set()
    for row in ordered:
        if row.case_uid in seen_case_uids:
            continue
        representatives.append(row)
        seen_case_uids.add(row.case_uid)
    return representatives


def build_case_bank_rows(
    *,
    duplicate_rows: Iterable[dict[str, object]],
    fn_rows: Iterable[dict[str, object]],
) -> list[CaseRow]:
    rows: list[CaseRow] = []
    for row in duplicate_rows:
        rows.append(
            CaseRow(
                case_uid=_case_uid(row, "first_burst_onset"),
                model_alias=str(row["model_alias"]),
                image_id=int(row["image_id"]),
                line_idx=int(row["line_idx"]),
                record_idx=int(row["record_idx"]),
                source_gt_vs_pred_jsonl=str(row["source_gt_vs_pred_jsonl"]),
                bucket="first_burst_onset",
                review_bucket="FP",
                source_object_index=int(row["source_object_index"]),
                onset_object_index=int(row["onset_object_index"]),
                gt_idx=None,
                selection_rank=int(row["selection_rank"]),
                serializer_surface=str(row["serializer_surface"]),
            )
        )
    for row in fn_rows:
        rows.append(
            CaseRow(
                case_uid=_case_uid(row, "labeled_fn"),
                model_alias=str(row["model_alias"]),
                image_id=int(row["image_id"]),
                line_idx=int(row["line_idx"]),
                record_idx=int(row["record_idx"]),
                source_gt_vs_pred_jsonl=str(row["source_gt_vs_pred_jsonl"]),
                bucket="labeled_fn",
                review_bucket="FN",
                source_object_index=None,
                onset_object_index=None,
                gt_idx=int(row["gt_idx"]),
                selection_rank=int(row["selection_rank"]),
                serializer_surface=str(row["serializer_surface"]),
            )
        )
    review_bucket_order = {"FP": 0, "FN": 1}
    return sorted(
        rows,
        key=lambda row: (
            review_bucket_order[row.review_bucket],
            row.selection_rank,
            row.model_alias,
            row.case_uid,
            _serializer_surface_priority(row.serializer_surface),
        ),
    )


def freeze_review_shortlist(
    rows: list[CaseRow],
    *,
    fp_budget: int,
    fn_budget: int,
) -> list[CaseRow]:
    representative_rows = _representative_case_rows(rows)
    fp_rows = [row for row in representative_rows if row.review_bucket == "FP"][
        :fp_budget
    ]
    fn_rows = [row for row in representative_rows if row.review_bucket == "FN"][
        :fn_budget
    ]
    return fp_rows + fn_rows
