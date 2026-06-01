from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DetectionEvalRecord(Mapping[str, Any]):
    """Schema-preserving raw gt_vs_pred row wrapper."""

    row: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.row, Mapping):
            raise TypeError("DetectionEvalRecord row must be a mapping")
        object.__setattr__(self, "row", copy.deepcopy(dict(self.row)))

    def __getitem__(self, key: str) -> Any:
        return self.row[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.row)

    def __len__(self) -> int:
        return len(self.row)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DetectionEvalRecord):
            return self.row == other.row
        if isinstance(other, Mapping):
            return dict(self.row) == dict(other)
        return NotImplemented

    @classmethod
    def from_json_record(cls, record: Mapping[str, Any]) -> "DetectionEvalRecord":
        return cls(record)

    def to_json_record(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.row))


@dataclass(frozen=True)
class ScoredDetectionEvalRecord(Mapping[str, Any]):
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
        source = self.row.get("pred_score_source")
        if not isinstance(source, str) or not source:
            raise ValueError(
                "ScoredDetectionEvalRecord requires non-empty pred_score_source"
            )
        version = self.row.get("pred_score_version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError(
                "ScoredDetectionEvalRecord requires integer pred_score_version"
            )
        preds = self.row.get("pred")
        if isinstance(preds, list):
            for pred_idx, pred in enumerate(preds):
                if not isinstance(pred, Mapping):
                    raise ValueError(
                        "ScoredDetectionEvalRecord pred entries must be mappings"
                    )
                if "score" not in pred:
                    raise ValueError(
                        "ScoredDetectionEvalRecord pred entries must include "
                        f"score (pred_idx={pred_idx})"
                    )
        for key, expected in (
            ("pred_score_source", source),
            ("pred_score_version", version),
        ):
            if key in self.score_provenance and self.score_provenance[key] != expected:
                raise ValueError(
                    f"score_provenance {key} does not match scored row "
                    f"({self.score_provenance[key]!r} != {expected!r})"
                )

    def __getitem__(self, key: str) -> Any:
        return self.row[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.row)

    def __len__(self) -> int:
        return len(self.row)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ScoredDetectionEvalRecord):
            return (
                self.row == other.row
                and self.score_provenance == other.score_provenance
            )
        if isinstance(other, Mapping):
            return dict(self.row) == dict(other)
        return NotImplemented

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
