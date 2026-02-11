"""Public contracts for rollout-matching.

This module is intentionally import-light (no trainer/framework imports) so it can
be used by Stage-2 AB and by tests/static tooling without pulling in trainer
implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple


GeomType = Literal["bbox_2d", "poly"]


@dataclass
class ParsedPredObject:
    key: str
    index: int
    desc: str
    geom_type: GeomType
    # Indices into rollout response_token_ids (assistant-local).
    coord_token_indices: List[int]
    # [char_start, char_end) span of the object value dict inside response text.
    value_span: Tuple[int, int]


@dataclass
class RolloutParseResult:
    # Stripped stop tokens, full rollout (assistant-local).
    response_token_ids: List[int]
    response_text: str

    # Suffix-trimmed prefix (assistant-local, append-ready).
    prefix_token_ids: List[int]
    prefix_text: str

    max_object_index_in_prefix: Optional[int]
    valid_objects: List[ParsedPredObject]
    dropped_invalid: int
    dropped_invalid_by_reason: Dict[str, int] = field(default_factory=dict)
    dropped_ambiguous: int = 0
    truncated: bool = False


@dataclass
class GTObject:
    index: int
    geom_type: GeomType
    # bbox: [x1,y1,x2,y2]; poly: flat [x,y,...]
    points_norm1000: List[int]
    desc: str


@dataclass
class MatchResult:
    # (pred_idx, gt_idx)
    matched_pairs: List[Tuple[int, int]]
    fn_gt_indices: List[int]
    fp_pred_indices: List[int]
    gating_rejections: int

    # Sum/count over matched pairs (maskIoU in norm1000 space). Used for lightweight monitoring.
    matched_maskiou_sum: float
    matched_maskiou_count: int


__all__ = [
    "GeomType",
    "ParsedPredObject",
    "RolloutParseResult",
    "GTObject",
    "MatchResult",
]
