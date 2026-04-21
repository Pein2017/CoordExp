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
    bucket: str
    review_bucket: str
    source_object_index: int | None
    onset_object_index: int | None
    gt_idx: int | None
    selection_rank: int
    serializer_surface: str


def _case_uid(row: dict[str, object], bucket: str) -> str:
    return (
        f"{row['model_alias']}:{row['image_id']}:{row['line_idx']}:"
        f"{row['record_idx']}:{bucket}"
    )


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
            row.case_uid,
        ),
    )


def freeze_review_shortlist(
    rows: list[CaseRow],
    *,
    fp_budget: int,
    fn_budget: int,
) -> list[CaseRow]:
    fp_rows = [row for row in rows if row.review_bucket == "FP"][:fp_budget]
    fn_rows = [row for row in rows if row.review_bucket == "FN"][:fn_budget]
    return fp_rows + fn_rows
