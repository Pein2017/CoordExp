from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

PrefixDenoisingBranchId = Literal["clean_full", "noisy_full"]
CoordSlot = Literal["x1", "y1", "x2", "y2"]


@dataclass(frozen=True)
class PrefixDenoisingSegment:
    segment_id: str
    branch_id: PrefixDenoisingBranchId
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    supervised_positions: tuple[int, ...]
    ce_denominator: int
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PrefixDenoisingKLSite:
    clean_segment_id: str
    noisy_segment_id: str
    object_index: int
    history_object_count: int
    coord_slot: CoordSlot
    clean_label_position: int
    noisy_label_position: int
    clean_gt_bin: int
    support_bins: tuple[int, ...]
    identical_prefix: bool = False


@dataclass(frozen=True)
class ResolvedPrefixDenoisingKLSite:
    clean_batch_index: int
    noisy_batch_index: int
    clean_label_position: int
    noisy_label_position: int
    clean_gt_bin: int
    support_bins: tuple[int, ...]
    coord_slot: CoordSlot
    object_index: int
    identical_prefix: bool = False


@dataclass(frozen=True)
class HybridPrefixDenoisingSample:
    ok: bool
    hybrid_sample_id: str
    base_sample_id: str
    clean_full: PrefixDenoisingSegment | None
    noisy_full: PrefixDenoisingSegment | None
    kl_sites: tuple[PrefixDenoisingKLSite, ...] = ()
    skip_reason: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def total_length(self) -> int:
        if self.clean_full is None or self.noisy_full is None:
            return 0
        return len(self.clean_full.input_ids) + len(self.noisy_full.input_ids)


@dataclass(frozen=True)
class PrefixDenoisingPackingEstimate:
    ok: bool
    total_length: int = 0
    skip_reason: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
