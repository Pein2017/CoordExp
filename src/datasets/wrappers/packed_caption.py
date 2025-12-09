"""Packed dataset wrapper for vision-language training.

This wrapper groups already-encoded samples (prompt + image -> text) into
packed sequences using ms-swift's bin-packing heuristic. It operates
rank-locally and leaves evaluation unchanged by default.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import IterableDataset, get_worker_info

from ...utils.logger import get_logger

logger = get_logger(__name__)


class PackedCaptionDataset(IterableDataset):
    """Iterable wrapper that packs encoded samples into a single sequence.

    Each yielded item is a `List[Dict]` that the ms-swift template collator
    (padding_free path) will flatten into one training sample.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        template: Any,
        packing_length: int,
        buffer_size: int = 512,
        min_fill_ratio: float = 0.65,
        drop_last: bool = True,
        allow_single_long: bool = True,
    ) -> None:
        super().__init__()
        if packing_length <= 0:
            raise ValueError("packing_length must be positive")
        self.dataset = dataset
        self.template = template
        self.packing_length = int(packing_length)
        self.buffer_size = max(int(buffer_size), 1)
        self.min_fill_ratio = float(min_fill_ratio)
        self.drop_last = bool(drop_last)
        self.allow_single_long = bool(allow_single_long)
        # Cached base length for telemetry only (do not expose __len__)
        try:
            self.length_hint = len(dataset)
        except Exception:
            self.length_hint = None

        # Enable packing flags on template for padding_free collator
        try:
            self.template.packing = True
            self.template.padding_free = True
        except Exception:
            logger.warning("Unable to set template packing flags; packing may misbehave")

        self._epoch: int | None = None

    # ---- Dataset plumbing -------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        if hasattr(self.dataset, "set_epoch"):
            try:
                self.dataset.set_epoch(epoch)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Failed to forward set_epoch to base dataset: {exc}")

    # Intentionally no __len__: treat as pure IterableDataset to avoid
    # ms-swift attaching BatchSamplerShard (PyTorch forbids batch_sampler with
    # IterableDataset). Keep a length_hint derived from the base dataset when
    # needed for telemetry.

    # ---- Core iterator ----------------------------------------------------
    def __iter__(self) -> Iterable[List[dict]]:
        # Ensure base dataset sees current epoch each iterator build
        if self._epoch is not None and hasattr(self.dataset, "set_epoch"):
            try:
                self.dataset.set_epoch(self._epoch)
            except Exception:
                pass

        iterator = self._iter_base_dataset()
        buffer: list[Tuple[dict, int]] = []
        emit_count = 0
        fill_sum = 0.0
        single_long = 0
        skipped_long = 0

        try:
            for sample in iterator:
                length = self._extract_length(sample)
                if length is None:
                    logger.warning("Sample missing length; skipping sample")
                    continue

                if length >= self.packing_length:
                    if self.allow_single_long:
                        single_long += 1
                        emit_count, fill_sum = self._log_fill_stats(
                            emit_count, fill_sum, length
                        )
                        yield [sample]
                    else:
                        skipped_long += 1
                        logger.warning(
                            "Dropping sample with length %s >= packing_length=%s",
                            length,
                            self.packing_length,
                        )
                    continue

                buffer.append((sample, length))
                if len(buffer) >= self.buffer_size:
                    emitted, buffer = self._pack_buffer(buffer, finished=False)
                    for pack in emitted:
                        emit_count, fill_sum = self._log_fill_stats(
                            emit_count, fill_sum, sum(self._extract_length(x) or 0 for x in pack)
                        )
                        yield pack

            if buffer:
                emitted, leftover = self._pack_buffer(buffer, finished=True)
                for pack in emitted:
                    emit_count, fill_sum = self._log_fill_stats(
                        emit_count, fill_sum, sum(self._extract_length(x) or 0 for x in pack)
                    )
                    yield pack
                if leftover and not self.drop_last:
                    total = sum(length for _, length in leftover)
                    if total > 0:
                        emit_count, fill_sum = self._log_fill_stats(
                            emit_count, fill_sum, total
                        )
                        yield [item for item, _ in leftover]
        finally:
            self._emit_epoch_stats(emit_count, fill_sum, single_long, skipped_long)

    # ---- Helpers ----------------------------------------------------------
    def _iter_base_dataset(self) -> Iterable[dict]:
        # Support map-style datasets by slicing indices per worker to avoid duplication
        worker = get_worker_info()
        if worker is None:
            if hasattr(self.dataset, "__iter__") and not hasattr(self.dataset, "__getitem__"):
                yield from self.dataset
                return
            for idx in range(len(self.dataset)):
                yield self.dataset[idx]
        else:
            base_len = len(self.dataset)
            for idx in range(worker.id, base_len, worker.num_workers):
                yield self.dataset[idx]

    def _extract_length(self, sample: dict) -> int | None:
        length = sample.get("length")
        if isinstance(length, int):
            return length
        input_ids = sample.get("input_ids")
        if input_ids is None:
            return None
        try:
            return len(input_ids)
        except Exception:
            return None

    def _pack_buffer(
        self, items: Sequence[Tuple[dict, int]], *, finished: bool
    ) -> Tuple[List[List[dict]], List[Tuple[dict, int]]]:
        try:
            import binpacking
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "binpacking is required for packing; install with `pip install binpacking`"
            ) from exc

        sequences = binpacking.to_constant_volume(
            [(idx, length) for idx, (_, length) in enumerate(items)],
            self.packing_length,
            weight_pos=1,
        )

        carry: List[Tuple[dict, int]] = []
        if sequences and not finished:
            # Carry the last group forward to mimic ms-swift behavior
            carry_group = sequences.pop()
            carry = [items[idx] for idx, _ in carry_group]

        emitted: List[List[dict]] = []
        for group in sequences:
            pack = [items[idx][0] for idx, _ in group]
            total_len = sum(items[idx][1] for idx, _ in group)
            if not finished and total_len < self.packing_length * self.min_fill_ratio:
                carry.extend(items[idx] for idx, _ in group)
                continue
            if finished and self.drop_last and total_len < self.packing_length * self.min_fill_ratio:
                # Drop underfilled final group when drop_last requested
                continue
            emitted.append(pack)

        return emitted, carry

    def _log_fill_stats(
        self, emit_count: int, fill_sum: float, total_length: int
    ) -> tuple[int, float]:
        emit_count += 1
        fill_sum += float(total_length) / float(self.packing_length)
        return emit_count, fill_sum

    def _emit_epoch_stats(
        self,
        emit_count: int,
        fill_sum: float,
        single_long: int,
        skipped_long: int,
    ) -> None:
        if emit_count == 0:
            logger.info(
                "Packing stats: no packs emitted; single_long=%s skipped_long=%s",
                single_long,
                skipped_long,
            )
            return
        avg_fill = fill_sum / emit_count if emit_count else 0.0
        logger.info(
            "Packing stats: packs=%s avg_fill=%.3f single_long=%s skipped_long=%s",
            emit_count,
            avg_fill,
            single_long,
            skipped_long,
        )


def build_packed_dataset(
    dataset: Any,
    template: Any,
    *,
    packing_length: int,
    buffer_size: int = 512,
    min_fill_ratio: float = 0.65,
    drop_last: bool = True,
    allow_single_long: bool = True,
) -> PackedCaptionDataset:
    """Helper to construct the wrapper."""
    return PackedCaptionDataset(
        dataset=dataset,
        template=template,
        packing_length=packing_length,
        buffer_size=buffer_size,
        min_fill_ratio=min_fill_ratio,
        drop_last=drop_last,
        allow_single_long=allow_single_long,
    )
