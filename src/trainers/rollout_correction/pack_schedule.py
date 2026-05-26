from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Stage2PackSlot:
    slot_index: int
    local_pack_index: int | None
    sync_gradients: bool

    @property
    def is_empty(self) -> bool:
        return self.local_pack_index is None


@dataclass(frozen=True)
class Stage2PackSchedule:
    local_pack_count: int
    rank_pack_counts: tuple[int, ...]
    global_slot_count: int
    empty_slot_count: int
    slots: tuple[Stage2PackSlot, ...]

    @classmethod
    def from_rank_pack_counts(
        cls,
        *,
        local_pack_count: int,
        rank_pack_counts: Sequence[int],
        sync_every_slot: bool | None = None,
    ) -> "Stage2PackSchedule":
        local_count = int(local_pack_count)
        if local_count < 0:
            raise ValueError("local_pack_count must be non-negative")

        counts = tuple(int(value) for value in rank_pack_counts)
        if not counts:
            counts = (local_count,)
        if any(value < 0 for value in counts):
            raise ValueError("rank_pack_counts must be non-negative")

        global_count = int(max(counts))
        if local_count > global_count:
            raise ValueError(
                "local_pack_count cannot exceed global_slot_count: "
                f"local_pack_count={local_count} global_slot_count={global_count}"
            )

        if sync_every_slot is None:
            # Uneven local pack counts require shadow slots on some ranks. In
            # practice, combining no_sync real slots with no_sync shadow slots
            # can leave DDP reducer state fragile on the final synchronized
            # backward. Synchronize every slot in that shape; the shadow slots
            # contribute a graph-connected zero loss, so the summed gradient is
            # equivalent to a single delayed all-reduce but much easier for DDP
            # to keep aligned.
            sync_every_slot = bool(global_count > 0 and len(set(counts)) > 1)

        empty_count = int(global_count - local_count)
        slots: list[Stage2PackSlot] = []
        for slot_index in range(global_count):
            local_index = int(slot_index) - int(empty_count)
            slots.append(
                Stage2PackSlot(
                    slot_index=int(slot_index),
                    local_pack_index=(
                        int(local_index) if int(local_index) >= 0 else None
                    ),
                    sync_gradients=bool(
                        sync_every_slot or int(slot_index) == int(global_count - 1)
                    ),
                )
            )

        return cls(
            local_pack_count=int(local_count),
            rank_pack_counts=counts,
            global_slot_count=int(global_count),
            empty_slot_count=int(empty_count),
            slots=tuple(slots),
        )

    @property
    def has_global_packs(self) -> bool:
        return int(self.global_slot_count) > 0

    @property
    def has_local_packs(self) -> bool:
        return int(self.local_pack_count) > 0

    def trace_payload(self) -> dict[str, int]:
        sync_slot_count = int(sum(1 for slot in self.slots if bool(slot.sync_gradients)))
        return {
            "local_pack_count": int(self.local_pack_count),
            "global_slot_count": int(self.global_slot_count),
            "empty_slot_count": int(self.empty_slot_count),
            "sync_slot_count": int(sync_slot_count),
            "pack_counts_min": int(min(self.rank_pack_counts))
            if self.rank_pack_counts
            else int(self.local_pack_count),
            "pack_counts_max": int(max(self.rank_pack_counts))
            if self.rank_pack_counts
            else int(self.local_pack_count),
        }


__all__ = [
    "Stage2PackSchedule",
    "Stage2PackSlot",
]
