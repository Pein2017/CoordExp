"""Packed dataset wrappers for vision-language training.

Dynamic packing keeps existing iterable behavior.
Static packing precomputes a deterministic, countable plan of pack indices.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from ...utils.logger import get_logger

logger = get_logger(__name__)

_LENGTH_CACHE_VERSION = 1
_PLAN_CACHE_VERSION = 1
_DEFAULT_LENGTH_CACHE_PERSIST_EVERY = 4096
_DEFAULT_LENGTH_CACHE_MAX_FLUSHES = 64


def _extract_sample_length(sample: Mapping[str, Any]) -> int | None:
    length = sample.get("length")
    if isinstance(length, int):
        return length
    input_ids = sample.get("input_ids")
    if input_ids is None:
        return None
    try:
        return len(input_ids)
    except TypeError:
        return None


def _json_canonical_dumps(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except TypeError as exc:
        raise ValueError(
            "Static packing fingerprint contains non-JSON-serializable values."
        ) from exc


def _fingerprint_digest(fingerprint: Mapping[str, Any]) -> str:
    text = _json_canonical_dumps(fingerprint)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_plan_checksum(plan: Sequence[Sequence[int]]) -> str:
    canonical = [[int(i) for i in pack] for pack in plan]
    payload = json.dumps(canonical, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        cache_text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Static packing cache file not found: {path}") from exc
    try:
        payload = json.loads(cache_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse static packing cache file: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Static packing cache must be a JSON object: {path}")
    return payload


def _resolve_rank_world(world_size_hint: int | None) -> tuple[int, int]:
    has_hint = world_size_hint is not None and int(world_size_hint) > 0
    world_size = max(int(world_size_hint or 1), 1)
    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            rank = int(torch.distributed.get_rank())
            world_size = int(torch.distributed.get_world_size())
            return rank, max(world_size, 1)
        except (RuntimeError, TypeError, ValueError):
            pass

    rank_env = os.environ.get("RANK")
    if isinstance(rank_env, str) and rank_env.strip().lstrip("-").isdigit():
        rank = int(rank_env)

    world_env = os.environ.get("WORLD_SIZE")
    if (not has_hint) and isinstance(world_env, str) and world_env.strip().isdigit():
        world_size = max(int(world_env), 1)

    return rank, max(world_size, 1)


def _wait_for_file(path: Path, *, timeout_s: float) -> None:
    timeout = float(timeout_s)
    if not math.isfinite(timeout):
        raise ValueError(f"wait_timeout_s must be finite, got {timeout_s!r}")

    deadline = None
    if timeout > 0:
        deadline = time.monotonic() + timeout

    while not path.exists():
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(
                "Timed out waiting for static packing cache artifact: "
                f"{path}. Increase training.packing_wait_timeout_s for large datasets."
            )
        time.sleep(0.2)


def _is_fusion_dataset(dataset: Any) -> bool:
    class_name = type(dataset).__name__.lower()
    module_name = type(dataset).__module__.lower()
    if "fusioncaptiondataset" in class_name:
        return True
    return "fusion" in module_name and "dataset" in module_name


def _select_probe_indices(num_samples: int, probe_size: int) -> list[int]:
    if num_samples <= 0:
        return []
    probe_size = max(1, int(probe_size))
    if num_samples <= probe_size:
        return list(range(num_samples))

    stride = max(num_samples // probe_size, 1)
    indices = list(range(0, num_samples, stride))[:probe_size]
    if (num_samples - 1) not in indices:
        indices[-1] = num_samples - 1
    return sorted(set(indices))


def _probe_order_invariant_lengths(
    dataset: Any,
    *,
    probe_size: int,
) -> None:
    num_samples = len(dataset)
    probe_indices = _select_probe_indices(num_samples, probe_size)
    if len(probe_indices) <= 1:
        return

    def _collect(order: Sequence[int]) -> dict[int, int]:
        lengths: dict[int, int] = {}
        for index in order:
            sample = dataset[int(index)]
            length = _extract_sample_length(sample)
            if length is None:
                raise ValueError(
                    "Static packing length probe failed: sample has no deterministic token length. "
                    f"Index={index}. Ensure dataset encoding emits `length` or `input_ids`."
                )
            lengths[int(index)] = int(length)
        return lengths

    lengths_forward = _collect(probe_indices)
    lengths_reverse = _collect(list(reversed(probe_indices)))
    changed = [
        i for i in probe_indices if lengths_forward.get(i) != lengths_reverse.get(i)
    ]
    if changed:
        preview = changed[:8]
        raise ValueError(
            "Static packing requires order-invariant per-index token lengths, but the probe "
            f"detected mismatches for indices {preview}. This usually indicates an order-sensitive "
            "dataset schedule (e.g., fusion/mixing) or non-deterministic preprocessing. "
            "Use training.packing_mode=dynamic for this run."
        )


def _validate_cache_fingerprint(
    *,
    path: Path,
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> None:
    if dict(observed) == dict(expected):
        return
    raise ValueError(
        "Static packing cache fingerprint mismatch at "
        f"{path}. expected_sha256={_fingerprint_digest(expected)} "
        f"observed_sha256={_fingerprint_digest(observed)}. "
        "Use a fresh training.output_dir/static_packing directory for this run."
    )


def _load_length_cache(
    *,
    path: Path,
    fingerprint: Mapping[str, Any],
    num_samples: int,
) -> list[int | None]:
    if not path.exists():
        return [None] * num_samples

    payload = _read_json(path)
    version = payload.get("version")
    if int(version or -1) != _LENGTH_CACHE_VERSION:
        raise ValueError(
            f"Unsupported static length cache version in {path}: {version!r}"
        )

    observed_fingerprint = payload.get("fingerprint")
    if not isinstance(observed_fingerprint, Mapping):
        raise TypeError(f"Static length cache fingerprint must be a mapping: {path}")
    _validate_cache_fingerprint(
        path=path,
        expected=fingerprint,
        observed=observed_fingerprint,
    )

    observed_num_samples = payload.get("num_samples")
    if int(observed_num_samples or -1) != int(num_samples):
        raise ValueError(
            "Static length cache dataset size mismatch in "
            f"{path}: expected={num_samples} observed={observed_num_samples!r}"
        )

    raw_lengths = payload.get("lengths")
    if not isinstance(raw_lengths, list):
        raise TypeError(f"Static length cache field `lengths` must be a list: {path}")
    if len(raw_lengths) != num_samples:
        raise ValueError(
            "Static length cache list length mismatch in "
            f"{path}: expected={num_samples} observed={len(raw_lengths)}"
        )

    normalized: list[int | None] = []
    for value in raw_lengths:
        if value is None:
            normalized.append(None)
            continue
        if isinstance(value, bool):
            raise TypeError(
                f"Static length cache contains non-integer length value in {path}: {value!r}"
            )
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Static length cache contains non-integer length value in {path}: {value!r}"
            ) from exc
        if parsed < 0:
            raise ValueError(
                f"Static length cache contains negative length value in {path}: {parsed}"
            )
        normalized.append(parsed)

    return normalized


def _persist_length_cache(
    *,
    path: Path,
    fingerprint: Mapping[str, Any],
    lengths: Sequence[int | None],
    num_samples: int,
) -> None:
    payload: dict[str, Any] = {
        "version": _LENGTH_CACHE_VERSION,
        "fingerprint": dict(fingerprint),
        "num_samples": int(num_samples),
        "lengths": [None if v is None else int(v) for v in lengths],
    }
    _write_json_atomic(path, payload)


def _compute_missing_lengths(
    *,
    dataset: Any,
    lengths: list[int | None],
    cache_path: Path,
    fingerprint: Mapping[str, Any],
    persist_every: int | None = None,
) -> list[int]:
    missing = [idx for idx, value in enumerate(lengths) if value is None]
    if not missing:
        return [int(v) for v in lengths if v is not None]

    if persist_every is None:
        persist_interval = max(
            _DEFAULT_LENGTH_CACHE_PERSIST_EVERY,
            int(
                math.ceil(
                    len(missing) / max(_DEFAULT_LENGTH_CACHE_MAX_FLUSHES, 1)
                )
            ),
        )
    else:
        persist_interval = max(int(persist_every), 1)

    logger.info(
        "Static packing length precompute: missing=%s persist_every=%s",
        len(missing),
        persist_interval,
    )

    missing_count = len(missing)
    for offset, index in enumerate(missing, start=1):
        sample = dataset[int(index)]
        length = _extract_sample_length(sample)
        if length is None:
            raise ValueError(
                "Static packing length precompute failed: sample has no token length. "
                f"Index={index}. Ensure dataset encoding emits `length` or `input_ids`."
            )
        lengths[int(index)] = int(length)
        if (offset % persist_interval == 0) and (offset < missing_count):
            _persist_length_cache(
                path=cache_path,
                fingerprint=fingerprint,
                lengths=lengths,
                num_samples=len(lengths),
            )

    _persist_length_cache(
        path=cache_path,
        fingerprint=fingerprint,
        lengths=lengths,
        num_samples=len(lengths),
    )

    return [int(v) for v in lengths if v is not None]


def _build_raw_pack_plan(
    *,
    lengths: Sequence[int],
    packing_length: int,
    min_fill_ratio: float,
    drop_last: bool,
    allow_single_long: bool,
) -> tuple[list[list[int]], int, int, float]:
    try:
        import binpacking
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "binpacking is required for packing; install with `pip install binpacking`"
        ) from exc

    short_items: list[tuple[int, int]] = []
    singleton_long_packs: list[list[int]] = []
    single_long = 0
    skipped_long = 0

    for index, length in enumerate(lengths):
        token_length = int(length)
        if token_length >= packing_length:
            if allow_single_long:
                singleton_long_packs.append([int(index)])
                single_long += 1
            else:
                skipped_long += 1
            continue
        short_items.append((int(index), token_length))

    packed_groups = binpacking.to_constant_volume(
        short_items,
        packing_length,
        weight_pos=1,
    )

    packs: list[list[int]] = []
    fill_sum = 0.0
    fill_floor = packing_length * float(min_fill_ratio)

    for group in packed_groups:
        pack_indices = sorted(int(item[0]) for item in group)
        if not pack_indices:
            continue
        total_len = sum(int(lengths[i]) for i in pack_indices)
        if len(pack_indices) >= 2 and total_len > packing_length:
            raise RuntimeError(
                "Static packing produced an invalid multi-sample pack with total length "
                f"{total_len} > packing_length={packing_length}."
            )
        if total_len < fill_floor and drop_last:
            continue
        packs.append(pack_indices)
        fill_sum += float(total_len) / float(packing_length)

    for pack in singleton_long_packs:
        packs.append(pack)
        total_len = int(lengths[pack[0]])
        fill_sum += float(total_len) / float(packing_length)

    packs.sort(key=lambda seq: (min(seq), tuple(seq)))
    avg_fill = float(fill_sum / len(packs)) if packs else 0.0
    return packs, single_long, skipped_long, avg_fill


def _align_pack_plan_for_ddp(
    *,
    raw_plan: Sequence[Sequence[int]],
    world_size: int,
    dataloader_drop_last: bool,
) -> tuple[list[list[int]], int, list[int]]:
    plan = [[int(i) for i in pack] for pack in raw_plan]
    if world_size <= 1:
        return plan, 0, []

    num_packs = len(plan)
    if num_packs <= 0:
        raise ValueError(
            "Static packing produced no packs; cannot align for DDP. "
            "Check packing_length/min_fill_ratio/sample limits."
        )

    remainder = num_packs % int(world_size)
    if dataloader_drop_last:
        kept = num_packs - remainder
        return plan[:kept], 0, []

    pad_needed = (int(world_size) - remainder) % int(world_size)
    if pad_needed <= 0:
        return plan, 0, []

    repeats = list(range(pad_needed))
    aligned = list(plan) + [list(plan[i]) for i in repeats]
    return aligned, int(pad_needed), repeats


def _read_plan_cache(
    *,
    path: Path,
    fingerprint: Mapping[str, Any],
    world_size: int,
    dataloader_drop_last: bool,
) -> dict[str, Any]:
    payload = _read_json(path)

    version = payload.get("version")
    if int(version or -1) != _PLAN_CACHE_VERSION:
        raise ValueError(f"Unsupported static plan cache version in {path}: {version!r}")

    observed_fingerprint = payload.get("fingerprint")
    if not isinstance(observed_fingerprint, Mapping):
        raise TypeError(f"Static plan cache fingerprint must be a mapping: {path}")
    _validate_cache_fingerprint(
        path=path,
        expected=fingerprint,
        observed=observed_fingerprint,
    )

    if int(payload.get("world_size") or -1) != int(world_size):
        raise ValueError(
            f"Static plan cache world_size mismatch in {path}: "
            f"expected={world_size} observed={payload.get('world_size')!r}"
        )
    if bool(payload.get("dataloader_drop_last", False)) != bool(dataloader_drop_last):
        raise ValueError(
            f"Static plan cache dataloader_drop_last mismatch in {path}."
        )

    raw_plan = payload.get("raw_plan")
    aligned_plan = payload.get("aligned_plan")
    if not isinstance(raw_plan, list) or not isinstance(aligned_plan, list):
        raise TypeError(
            f"Static plan cache fields `raw_plan` and `aligned_plan` must be lists: {path}"
        )

    return payload


def _persist_plan_cache(
    *,
    path: Path,
    fingerprint: Mapping[str, Any],
    raw_plan: Sequence[Sequence[int]],
    aligned_plan: Sequence[Sequence[int]],
    world_size: int,
    dataloader_drop_last: bool,
    pad_needed: int,
    repeated_pack_indices: Sequence[int],
    single_long: int,
    skipped_long: int,
    avg_fill: float,
) -> None:
    payload: dict[str, Any] = {
        "version": _PLAN_CACHE_VERSION,
        "fingerprint": dict(fingerprint),
        "world_size": int(world_size),
        "dataloader_drop_last": bool(dataloader_drop_last),
        "raw_plan": [[int(i) for i in pack] for pack in raw_plan],
        "aligned_plan": [[int(i) for i in pack] for pack in aligned_plan],
        "raw_plan_checksum": _stable_plan_checksum(raw_plan),
        "aligned_plan_checksum": _stable_plan_checksum(aligned_plan),
        "pad_needed": int(pad_needed),
        "repeated_pack_indices": [int(i) for i in repeated_pack_indices],
        "single_long": int(single_long),
        "skipped_long": int(skipped_long),
        "avg_fill": float(avg_fill),
    }
    _write_json_atomic(path, payload)


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
        try:
            self.length_hint = len(dataset)
        except TypeError:
            self.length_hint = None

        try:
            self.template.packing = True
            self.template.padding_free = True
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                "PackedCaptionDataset requires template to expose 'packing' and 'padding_free' flags."
            ) from exc

        self._epoch: int | None = None

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        set_epoch_fn = getattr(self.dataset, "set_epoch", None)
        if callable(set_epoch_fn):
            set_epoch_fn(epoch)

    def __iter__(self) -> Iterable[List[dict]]:
        if self._epoch is not None:
            set_epoch_fn = getattr(self.dataset, "set_epoch", None)
            if callable(set_epoch_fn):
                set_epoch_fn(self._epoch)

        iterator = self._iter_base_dataset()
        buffer: list[Tuple[dict, int]] = []
        emit_count = 0
        fill_sum = 0.0
        single_long = 0
        skipped_long = 0

        try:
            for sample in iterator:
                length = _extract_sample_length(sample)
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
                            emit_count,
                            fill_sum,
                            sum(_extract_sample_length(x) or 0 for x in pack),
                        )
                        yield pack

            if buffer:
                emitted, leftover = self._pack_buffer(buffer, finished=True)
                for pack in emitted:
                    emit_count, fill_sum = self._log_fill_stats(
                        emit_count,
                        fill_sum,
                        sum(_extract_sample_length(x) or 0 for x in pack),
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

    def _iter_base_dataset(self) -> Iterable[dict]:
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
        sequences = [sorted(group, key=lambda x: int(x[0])) for group in sequences]

        carry: List[Tuple[dict, int]] = []
        if sequences and not finished:
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


class StaticPackedCaptionDataset(Dataset):
    """Map-style packed dataset with deterministic pack plan and stable length."""

    def __init__(
        self,
        dataset: Any,
        *,
        template: Any,
        packing_length: int,
        raw_plan: Sequence[Sequence[int]],
        aligned_plan: Sequence[Sequence[int]],
        lengths: Sequence[int],
        world_size: int,
        dataloader_drop_last: bool,
        pad_needed: int,
        repeated_pack_indices: Sequence[int],
        raw_plan_checksum: str,
        aligned_plan_checksum: str,
        avg_fill: float,
        single_long: int,
        skipped_long: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.template = template
        self.packing_length = int(packing_length)
        self.raw_plan = [[int(i) for i in pack] for pack in raw_plan]
        self.pack_plan = [[int(i) for i in pack] for pack in aligned_plan]
        self.lengths = [int(v) for v in lengths]
        self.world_size = int(world_size)
        self.dataloader_drop_last = bool(dataloader_drop_last)
        self.pad_needed = int(pad_needed)
        self.repeated_pack_indices = [int(i) for i in repeated_pack_indices]
        self.raw_plan_checksum = str(raw_plan_checksum)
        self.aligned_plan_checksum = str(aligned_plan_checksum)
        self.avg_fill = float(avg_fill)
        self.single_long = int(single_long)
        self.skipped_long = int(skipped_long)

        try:
            self.template.packing = True
            self.template.padding_free = True
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                "StaticPackedCaptionDataset requires template to expose 'packing' and 'padding_free' flags."
            ) from exc

        if len(self.pack_plan) <= 0:
            raise ValueError("StaticPackedCaptionDataset requires at least one packed sequence")

    def __len__(self) -> int:
        return len(self.pack_plan)

    def __getitem__(self, index: int) -> list[dict]:
        pack_indices = self.pack_plan[int(index)]
        return [self.dataset[int(raw_index)] for raw_index in pack_indices]

    def set_epoch(self, epoch: int) -> None:
        _ = int(epoch)


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
    """Construct the dynamic iterable packing wrapper."""
    return PackedCaptionDataset(
        dataset=dataset,
        template=template,
        packing_length=packing_length,
        buffer_size=buffer_size,
        min_fill_ratio=min_fill_ratio,
        drop_last=drop_last,
        allow_single_long=allow_single_long,
    )


def build_static_packed_dataset(
    dataset: Any,
    template: Any,
    *,
    packing_length: int,
    min_fill_ratio: float = 0.65,
    packing_drop_last: bool = True,
    dataloader_drop_last: bool = False,
    allow_single_long: bool = True,
    cache_dir: str | Path,
    fingerprint: Mapping[str, Any],
    world_size: int = 1,
    train_dataloader_shuffle: bool | None = None,
    order_probe_size: int = 16,
    wait_timeout_s: float = 7200.0,
    length_cache_persist_every: int | None = None,
) -> StaticPackedCaptionDataset:
    """Construct deterministic static packed dataset with run-scoped cache.

    Rank-0 computes/updates caches; other ranks wait and then load identical artifacts.
    """
    if packing_length <= 0:
        raise ValueError("packing_length must be positive")
    if not (0 < float(min_fill_ratio) <= 1):
        raise ValueError("packing_min_fill_ratio must be in (0,1]")

    wait_timeout_s = float(wait_timeout_s)
    if not math.isfinite(wait_timeout_s):
        raise ValueError(f"wait_timeout_s must be finite, got {wait_timeout_s!r}")

    if length_cache_persist_every is not None:
        length_cache_persist_every = int(length_cache_persist_every)
        if length_cache_persist_every <= 0:
            raise ValueError(
                "length_cache_persist_every must be > 0 when provided"
            )

    try:
        num_samples = int(len(dataset))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Static packing requires a sized map-style dataset with a stable __len__."
        ) from exc

    if num_samples <= 0:
        raise ValueError("Static packing requires a non-empty base dataset")

    if _is_fusion_dataset(dataset):
        raise ValueError(
            "Static packing does not support fusion/mixing datasets because their sample schedule "
            "is order-sensitive. Use training.packing_mode=dynamic for fusion runs."
        )

    rank, discovered_world_size = _resolve_rank_world(world_size_hint=world_size)
    world_size = max(int(world_size or discovered_world_size), 1)

    cache_root = Path(cache_dir)
    lengths_cache_path = cache_root / "lengths.json"
    plan_cache_path = cache_root / (
        f"plan_ws{world_size}_drop{int(bool(dataloader_drop_last))}.json"
    )

    canonical_fingerprint: dict[str, Any] = dict(fingerprint)
    canonical_fingerprint["dataset_class"] = type(dataset).__name__
    canonical_fingerprint["dataset_module"] = type(dataset).__module__
    canonical_fingerprint["dataset_len"] = int(num_samples)
    canonical_fingerprint["packing_length"] = int(packing_length)
    canonical_fingerprint["packing_min_fill_ratio"] = float(min_fill_ratio)
    canonical_fingerprint["packing_allow_single_long"] = bool(allow_single_long)
    canonical_fingerprint["packing_drop_last"] = bool(packing_drop_last)
    canonical_fingerprint["dataloader_drop_last"] = bool(dataloader_drop_last)
    canonical_fingerprint["train_dataloader_shuffle"] = (
        None
        if train_dataloader_shuffle is None
        else bool(train_dataloader_shuffle)
    )
    canonical_fingerprint = json.loads(_json_canonical_dumps(canonical_fingerprint))

    if rank == 0:
        cache_root.mkdir(parents=True, exist_ok=True)
        _probe_order_invariant_lengths(dataset, probe_size=order_probe_size)
        lengths_state = _load_length_cache(
            path=lengths_cache_path,
            fingerprint=canonical_fingerprint,
            num_samples=num_samples,
        )
        lengths = _compute_missing_lengths(
            dataset=dataset,
            lengths=lengths_state,
            cache_path=lengths_cache_path,
            fingerprint=canonical_fingerprint,
            persist_every=length_cache_persist_every,
        )
        raw_plan, single_long, skipped_long, avg_fill = _build_raw_pack_plan(
            lengths=lengths,
            packing_length=packing_length,
            min_fill_ratio=min_fill_ratio,
            drop_last=packing_drop_last,
            allow_single_long=allow_single_long,
        )
        aligned_plan, pad_needed, repeated_pack_indices = _align_pack_plan_for_ddp(
            raw_plan=raw_plan,
            world_size=world_size,
            dataloader_drop_last=bool(dataloader_drop_last),
        )
        _persist_plan_cache(
            path=plan_cache_path,
            fingerprint=canonical_fingerprint,
            raw_plan=raw_plan,
            aligned_plan=aligned_plan,
            world_size=world_size,
            dataloader_drop_last=bool(dataloader_drop_last),
            pad_needed=pad_needed,
            repeated_pack_indices=repeated_pack_indices,
            single_long=single_long,
            skipped_long=skipped_long,
            avg_fill=avg_fill,
        )
    else:
        if wait_timeout_s <= 0:
            logger.info(
                "Static packing cache wait: rank=%s waiting without timeout for %s and %s",
                rank,
                lengths_cache_path,
                plan_cache_path,
            )
        _wait_for_file(lengths_cache_path, timeout_s=wait_timeout_s)
        _wait_for_file(plan_cache_path, timeout_s=wait_timeout_s)

    lengths_state = _load_length_cache(
        path=lengths_cache_path,
        fingerprint=canonical_fingerprint,
        num_samples=num_samples,
    )
    lengths = [int(v) for v in lengths_state if v is not None]
    if len(lengths) != num_samples:
        raise ValueError(
            "Static length cache contains unresolved entries after rank synchronization. "
            f"cache={lengths_cache_path}"
        )

    plan_payload = _read_plan_cache(
        path=plan_cache_path,
        fingerprint=canonical_fingerprint,
        world_size=world_size,
        dataloader_drop_last=bool(dataloader_drop_last),
    )
    raw_plan = plan_payload.get("raw_plan")
    aligned_plan = plan_payload.get("aligned_plan")
    if not isinstance(raw_plan, list) or not isinstance(aligned_plan, list):
        raise TypeError(
            f"Static plan cache contains invalid plan payload in {plan_cache_path}"
        )

    raw_plan_checksum = str(plan_payload.get("raw_plan_checksum") or "")
    aligned_plan_checksum = str(plan_payload.get("aligned_plan_checksum") or "")
    if not raw_plan_checksum:
        raw_plan_checksum = _stable_plan_checksum(raw_plan)
    if not aligned_plan_checksum:
        aligned_plan_checksum = _stable_plan_checksum(aligned_plan)

    pad_needed = int(plan_payload.get("pad_needed") or 0)
    repeated_pack_indices = plan_payload.get("repeated_pack_indices") or []
    if not isinstance(repeated_pack_indices, list):
        raise TypeError(
            f"Static plan cache field repeated_pack_indices must be a list: {plan_cache_path}"
        )

    avg_fill = float(plan_payload.get("avg_fill") or 0.0)
    single_long = int(plan_payload.get("single_long") or 0)
    skipped_long = int(plan_payload.get("skipped_long") or 0)

    static_dataset = StaticPackedCaptionDataset(
        dataset=dataset,
        template=template,
        packing_length=packing_length,
        raw_plan=raw_plan,
        aligned_plan=aligned_plan,
        lengths=lengths,
        world_size=world_size,
        dataloader_drop_last=bool(dataloader_drop_last),
        pad_needed=pad_needed,
        repeated_pack_indices=[int(i) for i in repeated_pack_indices],
        raw_plan_checksum=raw_plan_checksum,
        aligned_plan_checksum=aligned_plan_checksum,
        avg_fill=avg_fill,
        single_long=single_long,
        skipped_long=skipped_long,
    )

    logger.info(
        "Static packing cache ready: rank=%s world_size=%s num_samples=%s lengths_cache=%s plan_cache=%s",
        rank,
        world_size,
        num_samples,
        lengths_cache_path,
        plan_cache_path,
    )

    return static_dataset
