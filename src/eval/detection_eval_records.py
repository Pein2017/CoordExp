from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class DetectionEvalRecord:
    """Schema-preserving raw gt_vs_pred row wrapper."""

    row: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.row, Mapping):
            raise TypeError("DetectionEvalRecord row must be a mapping")
        object.__setattr__(self, "row", copy.deepcopy(dict(self.row)))

    @classmethod
    def from_json_record(cls, record: Mapping[str, Any]) -> "DetectionEvalRecord":
        return cls(record)

    def to_json_record(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.row))


@dataclass(frozen=True)
class ScoredDetectionEvalRecord:
    """Schema-preserving scored gt_vs_pred row wrapper plus score provenance."""

    row: Mapping[str, Any]
    score_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.row, Mapping):
            raise TypeError("ScoredDetectionEvalRecord row must be a mapping")
        if not isinstance(self.score_provenance, Mapping):
            raise TypeError("score_provenance must be a mapping")
        object.__setattr__(self, "row", copy.deepcopy(dict(self.row)))
        object.__setattr__(
            self,
            "score_provenance",
            copy.deepcopy(dict(self.score_provenance)),
        )

    @classmethod
    def from_json_record(
        cls,
        record: Mapping[str, Any],
        *,
        score_provenance: Mapping[str, Any] | None = None,
    ) -> "ScoredDetectionEvalRecord":
        return cls(record, score_provenance=dict(score_provenance or {}))

    @classmethod
    def from_detection_eval_record(
        cls,
        record: DetectionEvalRecord,
        *,
        pred_score_source: str,
        pred_score_version: int,
        constant_score: float,
        score_provenance: Mapping[str, Any] | None = None,
    ) -> "ScoredDetectionEvalRecord":
        row = record.to_json_record()
        row["pred_score_source"] = str(pred_score_source)
        row["pred_score_version"] = int(pred_score_version)
        score = float(constant_score)
        preds_out: list[dict[str, Any]] = []
        preds_raw = row.get("pred")
        if isinstance(preds_raw, list):
            for pred in preds_raw:
                if not isinstance(pred, Mapping):
                    continue
                pred_out = copy.deepcopy(dict(pred))
                pred_out["score"] = score
                preds_out.append(pred_out)
        row["pred"] = preds_out
        return cls.from_json_record(
            row,
            score_provenance=dict(score_provenance or {}),
        )

    def to_json_record(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.row))
