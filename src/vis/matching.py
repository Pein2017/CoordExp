"""Pixel-space matching and duplicate hints for detection visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.vis.normalization import VisualObject, VisualRow


@dataclass(frozen=True)
class MatchPair:
    pred_index: int
    gt_index: int
    iou: float

    def to_manifest(self) -> dict[str, Any]:
        return {
            "pred_index": self.pred_index,
            "gt_index": self.gt_index,
            "iou": self.iou,
        }


@dataclass(frozen=True)
class DuplicateCandidate:
    pred_a_index: int
    pred_b_index: int
    description: str
    iou: float
    pred_a_status: str
    pred_b_status: str

    def to_manifest(self) -> dict[str, Any]:
        return {
            "pred_a_index": self.pred_a_index,
            "pred_b_index": self.pred_b_index,
            "description": self.description,
            "iou": self.iou,
            "pred_a_status": self.pred_a_status,
            "pred_b_status": self.pred_b_status,
        }


@dataclass(frozen=True)
class MatchResult:
    matches: tuple[MatchPair, ...]
    missing_gt_indices: tuple[int, ...]
    fp_pred_indices: tuple[int, ...]
    duplicate_candidates: tuple[DuplicateCandidate, ...]

    @property
    def matched_pred_indices(self) -> set[int]:
        return {match.pred_index for match in self.matches}

    @property
    def matched_gt_indices(self) -> set[int]:
        return {match.gt_index for match in self.matches}

    def stats(self) -> dict[str, float | int]:
        tp = len(self.matches)
        fn = len(self.missing_gt_indices)
        fp = len(self.fp_pred_indices)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def to_manifest(self) -> dict[str, Any]:
        return {
            **self.stats(),
            "matched_pairs": [match.to_manifest() for match in self.matches],
            "missing_gt_indices": list(self.missing_gt_indices),
            "fp_pred_indices": list(self.fp_pred_indices),
            "duplicate_candidates": [
                candidate.to_manifest() for candidate in self.duplicate_candidates
            ],
        }


def match_row(
    row: VisualRow,
    *,
    match_iou_threshold: float = 0.50,
    duplicate_iou_threshold: float = 0.30,
) -> MatchResult:
    candidates: list[tuple[float, int, int]] = []
    for gt in row.gt:
        for pred in row.pred:
            value = iou_xyxy(gt.bbox_pixel_xyxy, pred.bbox_pixel_xyxy)
            if (
                gt.normalized_description == pred.normalized_description
                and value >= match_iou_threshold
            ):
                candidates.append((value, gt.index, pred.index))
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[MatchPair] = []
    for value, gt_index, pred_index in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append(MatchPair(pred_index=pred_index, gt_index=gt_index, iou=value))
    duplicates = _duplicate_candidates(
        row.pred,
        matched_pred_indices=used_pred,
        duplicate_iou_threshold=duplicate_iou_threshold,
    )
    return MatchResult(
        matches=tuple(matches),
        missing_gt_indices=tuple(index for index in range(len(row.gt)) if index not in used_gt),
        fp_pred_indices=tuple(index for index in range(len(row.pred)) if index not in used_pred),
        duplicate_candidates=tuple(duplicates),
    )


def iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _duplicate_candidates(
    pred: tuple[VisualObject, ...],
    *,
    matched_pred_indices: set[int],
    duplicate_iou_threshold: float,
) -> list[DuplicateCandidate]:
    candidates: list[DuplicateCandidate] = []
    for left_index, left in enumerate(pred):
        for right in pred[left_index + 1 :]:
            if left.normalized_description != right.normalized_description:
                continue
            value = iou_xyxy(left.bbox_pixel_xyxy, right.bbox_pixel_xyxy)
            if value < duplicate_iou_threshold:
                continue
            candidates.append(
                DuplicateCandidate(
                    pred_a_index=left.index,
                    pred_b_index=right.index,
                    description=left.description,
                    iou=value,
                    pred_a_status="matched" if left.index in matched_pred_indices else "fp",
                    pred_b_status="matched" if right.index in matched_pred_indices else "fp",
                )
            )
    return candidates
