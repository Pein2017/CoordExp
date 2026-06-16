from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.common.io import load_jsonl_with_diagnostics
from src.config.schema import PrefixDenoisingConfig
from src.detection.dataset import resolve_detection_jsonl_image_root

from .builder import (
    build_hybrid_prefix_denoising_sample,
    estimate_hybrid_prefix_denoising_packing,
    prefix_denoising_fast_estimator_fingerprint,
)
from .types import (
    HybridPrefixDenoisingSample,
    PrefixDenoisingPackingEstimate,
    PrefixDenoisingSegment,
)

_CORE_SEGMENT_KEYS = {"input_ids", "labels", "attention_mask"}
logger = logging.getLogger(__name__)


class PrefixDenoisingTrainingDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        swift_template: Any,
        image_root: str | Path,
        user_prompt: str,
        system_prompt: str | None,
        prefix_denoising: PrefixDenoisingConfig,
        max_length: int,
        dataset_name: str,
        seed: int,
        eligibility_cache_dir: str | Path | None = None,
        eligibility_cache_fingerprint: Mapping[str, Any] | None = None,
        eligibility_cache_wait_timeout_s: float = 7200.0,
        eligibility_precompute_workers: int = 1,
    ) -> None:
        self.rows = tuple(copy.deepcopy(dict(row)) for row in rows)
        if not self.rows:
            raise ValueError("PrefixDenoisingTrainingDataset requires at least one row")
        self.swift_template = swift_template
        self.template = swift_template
        self.tokenizer = getattr(swift_template, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError("swift_template must expose tokenizer")
        self.image_root = Path(image_root).expanduser().resolve(strict=False)
        self.user_prompt = str(user_prompt)
        self.system_prompt = system_prompt
        self.prefix_denoising = prefix_denoising
        self.max_length = int(max_length)
        self.dataset_name = str(dataset_name)
        self.seed = int(seed)
        self._epoch = 0
        eligibility = load_or_build_prefix_denoising_eligibility_index(
            self.rows,
            swift_template=self.swift_template,
            image_root=self.image_root,
            user_prompt=self.user_prompt,
            system_prompt=self.system_prompt,
            prefix_denoising=self.prefix_denoising,
            max_length=self.max_length,
            dataset_name=self.dataset_name,
            seed=self.seed,
            cache_dir=eligibility_cache_dir,
            fingerprint=(
                eligibility_cache_fingerprint
                or _default_eligibility_fingerprint(
                    rows=self.rows,
                    image_root=self.image_root,
                    user_prompt=self.user_prompt,
                    system_prompt=self.system_prompt,
                    prefix_denoising=self.prefix_denoising,
                    swift_template=self.swift_template,
                    max_length=self.max_length,
                    dataset_name=self.dataset_name,
                    seed=self.seed,
                )
            ),
            wait_timeout_s=eligibility_cache_wait_timeout_s,
            precompute_workers=eligibility_precompute_workers,
        )
        self._eligible_indices = tuple(eligibility["eligible_indices"])
        self._static_lengths = dict(eligibility["static_lengths"])
        self.skip_counters = Counter(eligibility["skip_counters"])
        if self.skip_counters:
            logger.warning(
                "prefix-denoising skipped rows during dataset eligibility build: "
                "dataset=%s eligible=%d total=%d skip_counters=%s",
                self.dataset_name,
                len(self._eligible_indices),
                len(self.rows),
                dict(sorted(self.skip_counters.items())),
            )
        if not self._eligible_indices:
            raise ValueError(
                "PrefixDenoisingTrainingDataset has no eligible rows; "
                f"skip_counters={dict(self.skip_counters)}"
            )

    @classmethod
    def from_jsonl(
        cls,
        jsonl_path: str | Path,
        *,
        swift_template: Any,
        image_root: str | Path | None,
        user_prompt: str,
        system_prompt: str | None,
        prefix_denoising: PrefixDenoisingConfig,
        max_length: int,
        seed: int,
        sample_limit: int | None = None,
        dataset_name: str | None = None,
        eligibility_cache_dir: str | Path | None = None,
        eligibility_cache_fingerprint: Mapping[str, Any] | None = None,
        eligibility_cache_wait_timeout_s: float = 7200.0,
        eligibility_precompute_workers: int = 1,
    ) -> "PrefixDenoisingTrainingDataset":
        path = Path(jsonl_path)
        resolved_image_root = resolve_detection_jsonl_image_root(
            path,
            image_root=image_root,
        )
        rows, _invalid_count = load_jsonl_with_diagnostics(path, strict=True)
        if sample_limit is not None:
            if sample_limit <= 0:
                raise ValueError("sample_limit must be positive when provided")
            rows = rows[: int(sample_limit)]
        return cls(
            rows,
            swift_template=swift_template,
            image_root=resolved_image_root,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            prefix_denoising=prefix_denoising,
            max_length=max_length,
            dataset_name=dataset_name or path.stem,
            seed=seed,
            eligibility_cache_dir=eligibility_cache_dir,
            eligibility_cache_fingerprint=eligibility_cache_fingerprint,
            eligibility_cache_wait_timeout_s=eligibility_cache_wait_timeout_s,
            eligibility_precompute_workers=eligibility_precompute_workers,
        )

    def __len__(self) -> int:
        return len(self._eligible_indices)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _static_packing_length(self, index: int) -> int | None:
        base_idx = self._base_index(index)
        return self._static_lengths.get(base_idx)

    def _static_packing_precompute_info(self) -> dict[str, Any]:
        return {
            "thread_safe": True,
            "prefix_denoising_dataset_summary": self.prefix_denoising_dataset_summary(),
        }

    def prefix_denoising_dataset_summary(self) -> dict[str, Any]:
        skip_counters = {
            str(key): int(value)
            for key, value in sorted(
                self.skip_counters.items(), key=lambda item: str(item[0])
            )
        }
        eligible_rows = len(self._eligible_indices)
        source_rows = len(self.rows)
        return {
            "dataset_name": self.dataset_name,
            "source_rows": int(source_rows),
            "eligible_rows": int(eligible_rows),
            "skipped_rows": int(max(source_rows - eligible_rows, 0)),
            "skip_counters": skip_counters,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_idx = self._base_index(index)
        sample = self._build_sample(base_idx, epoch=self._epoch)
        if not sample.ok:
            raise ValueError(
                "prefix-denoising eligible row became ineligible at getitem: "
                f"base_idx={base_idx}, reason={sample.skip_reason}"
            )
        return materialize_hybrid_model_ready_item(
            sample=sample,
            dataset_name=self.dataset_name,
            base_idx=base_idx,
        )

    def _base_index(self, index: int) -> int:
        item_idx = int(index)
        if item_idx < 0 or item_idx >= len(self._eligible_indices):
            raise IndexError(
                f"PrefixDenoisingTrainingDataset index {index!r} is out of range "
                f"for dataset of size {len(self)}"
            )
        return int(self._eligible_indices[item_idx])

    def _build_sample(self, base_idx: int, *, epoch: int) -> HybridPrefixDenoisingSample:
        return build_hybrid_prefix_denoising_sample(
            self.rows[int(base_idx)],
            base_sample_id=_make_base_sample_id(self.dataset_name, int(base_idx)),
            image_root=self.image_root,
            swift_template=self.swift_template,
            user_prompt=self.user_prompt,
            system_prompt=self.system_prompt,
            prefix_denoising=self.prefix_denoising,
            epoch=int(epoch),
            rng=random.Random(_mix_seed(self.seed, int(epoch), int(base_idx))),
            max_length=self.max_length,
        )


def build_prefix_denoising_eligibility_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    swift_template: Any,
    image_root: str | Path,
    user_prompt: str,
    system_prompt: str | None,
    prefix_denoising: PrefixDenoisingConfig,
    max_length: int,
    dataset_name: str,
    seed: int,
    precompute_workers: int = 1,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    eligible_indices: list[int] = []
    static_lengths: dict[int, int] = {}
    skip_counters: Counter[str] = Counter()
    total = len(rows)

    def _build_one(base_idx: int) -> tuple[int, PrefixDenoisingPackingEstimate]:
        return (
            int(base_idx),
            estimate_hybrid_prefix_denoising_packing(
                rows[base_idx],
                image_root=image_root,
                swift_template=swift_template,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                prefix_denoising=prefix_denoising,
                rng=random.Random(_mix_seed(seed, 0, base_idx)),
                max_length=max_length,
            ),
        )

    def _record(base_idx: int, estimate: PrefixDenoisingPackingEstimate) -> None:
        if not estimate.ok:
            skip_counters[str(estimate.skip_reason or "unknown_skip_reason")] += 1
            return
        eligible_indices.append(base_idx)
        static_lengths[base_idx] = int(estimate.total_length)

    completed = 0
    requested_workers = max(int(precompute_workers or 1), 1)
    if requested_workers > 1:
        logger.info(
            "prefix-denoising eligibility build: rows=%s requested_workers=%s effective_workers=1 reason=fast_estimator_tokenizer_gil",
            total,
            requested_workers,
        )
    else:
        logger.info("prefix-denoising eligibility build: rows=%s workers=1", total)
    for base_idx in range(total):
        base_idx, sample = _build_one(base_idx)
        _record(base_idx, sample)
        completed += 1
        if progress_callback is not None:
            progress_callback(
                completed=completed,
                total=total,
                eligible=len(eligible_indices),
                skip_counters=skip_counters,
            )

    eligible_indices.sort()
    static_lengths = {
        int(index): int(static_lengths[index]) for index in eligible_indices
    }
    return {
        "eligible_indices": tuple(eligible_indices),
        "static_lengths": static_lengths,
        "skip_counters": dict(skip_counters),
    }


def load_or_build_prefix_denoising_eligibility_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    swift_template: Any,
    image_root: str | Path,
    user_prompt: str,
    system_prompt: str | None,
    prefix_denoising: PrefixDenoisingConfig,
    max_length: int,
    dataset_name: str,
    seed: int,
    cache_dir: str | Path | None = None,
    fingerprint: Mapping[str, Any] | None = None,
    wait_timeout_s: float = 7200.0,
    precompute_workers: int = 1,
) -> dict[str, Any]:
    if cache_dir is None:
        return build_prefix_denoising_eligibility_index(
            rows,
            swift_template=swift_template,
            image_root=image_root,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            prefix_denoising=prefix_denoising,
            max_length=max_length,
            dataset_name=dataset_name,
            seed=seed,
            precompute_workers=precompute_workers,
        )

    fingerprint_payload = dict(
        fingerprint
            or _default_eligibility_fingerprint(
                rows=rows,
                image_root=image_root,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                prefix_denoising=prefix_denoising,
                swift_template=swift_template,
                max_length=max_length,
                dataset_name=dataset_name,
                seed=seed,
        )
    )
    fingerprint_payload["source_rows"] = int(len(rows))
    fingerprint_payload = _json_ready(fingerprint_payload)
    digest = _fingerprint_digest(fingerprint_payload)
    cache_root = Path(cache_dir).expanduser().resolve(strict=False) / digest
    cache_path = cache_root / "eligibility.json"
    progress_path = cache_root / "progress.json"
    rank, world_size = _resolve_rank_world()

    logger.info(
        "prefix-denoising eligibility cache resolved: rank=%s world_size=%s cache_root=%s",
        rank,
        world_size,
        cache_root,
    )

    if cache_path.exists():
        return _load_eligibility_cache(
            cache_path,
            fingerprint=fingerprint_payload,
            source_rows=len(rows),
        )

    if rank == 0:
        cache_root.mkdir(parents=True, exist_ok=True)
        compatible = _load_compatible_eligibility_cache(
            Path(cache_dir).expanduser().resolve(strict=False),
            fingerprint=fingerprint_payload,
            source_rows=len(rows),
            exact_cache_path=cache_path,
        )
        if compatible is not None:
            _persist_eligibility_cache(
                cache_path,
                fingerprint=fingerprint_payload,
                source_rows=len(rows),
                eligibility=compatible,
            )
            logger.info(
                "prefix-denoising eligibility cache aliased: rank=0 eligible=%s total=%s cache=%s",
                len(compatible["eligible_indices"]),
                len(rows),
                cache_path,
            )
            return compatible

        start_time = time.monotonic()
        last_progress = 0.0
        progress_interval = max(100, min(5000, max(len(rows) // 20, 1)))

        def _progress(
            *,
            completed: int,
            total: int,
            eligible: int,
            skip_counters: Counter[str],
        ) -> None:
            nonlocal last_progress
            now = time.monotonic()
            is_checkpoint = completed >= total or completed % progress_interval == 0
            if not is_checkpoint and now - last_progress < 30.0:
                return
            last_progress = now
            elapsed = max(now - start_time, 1e-6)
            rate = float(completed) / elapsed
            payload = {
                "schema_version": "prefix_denoising_eligibility_progress_v1",
                "completed": int(completed),
                "total": int(total),
                "eligible": int(eligible),
                "skipped": int(max(completed - eligible, 0)),
                "skip_counters": {
                    str(key): int(value)
                    for key, value in sorted(skip_counters.items())
                },
                "elapsed_s": float(elapsed),
                "rows_per_s": float(rate),
                "updated_at": time.time(),
            }
            _write_json_atomic(progress_path, payload)
            logger.info(
                "prefix-denoising eligibility progress: %s/%s eligible=%s skipped=%s rows_per_s=%.2f cache_root=%s",
                completed,
                total,
                eligible,
                max(completed - eligible, 0),
                rate,
                cache_root,
            )

        logger.warning(
            "prefix-denoising eligibility build starting: rank=0 cache_root=%s rows=%s "
            "requested_workers=%s effective_workers=1. "
            "Static packing cache is created after this precompute finishes.",
            cache_root,
            len(rows),
            max(int(precompute_workers or 1), 1),
        )
        eligibility = build_prefix_denoising_eligibility_index(
            rows,
            swift_template=swift_template,
            image_root=image_root,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            prefix_denoising=prefix_denoising,
            max_length=max_length,
            dataset_name=dataset_name,
            seed=seed,
            precompute_workers=precompute_workers,
            progress_callback=_progress,
        )
        _persist_eligibility_cache(
            cache_path,
            fingerprint=fingerprint_payload,
            source_rows=len(rows),
            eligibility=eligibility,
        )
        logger.info(
            "prefix-denoising eligibility cache ready: rank=0 eligible=%s total=%s cache=%s",
            len(eligibility["eligible_indices"]),
            len(rows),
            cache_path,
        )
        return eligibility

    logger.info(
        "prefix-denoising eligibility cache wait: rank=%s waiting for %s",
        rank,
        cache_path,
    )
    _wait_for_file(cache_path, timeout_s=wait_timeout_s)
    return _load_eligibility_cache(
        cache_path,
        fingerprint=fingerprint_payload,
        source_rows=len(rows),
    )


def _persist_eligibility_cache(
    path: Path,
    *,
    fingerprint: Mapping[str, Any],
    source_rows: int,
    eligibility: Mapping[str, Any],
) -> None:
    static_lengths = eligibility.get("static_lengths") or {}
    payload = {
        "schema_version": "prefix_denoising_eligibility_cache_v1",
        "fingerprint": _json_ready(fingerprint),
        "source_rows": int(source_rows),
        "eligible_indices": [int(index) for index in eligibility["eligible_indices"]],
        "static_lengths": [
            [int(index), int(length)]
            for index, length in sorted(
                static_lengths.items(), key=lambda item: int(item[0])
            )
        ],
        "skip_counters": {
            str(key): int(value)
            for key, value in sorted((eligibility.get("skip_counters") or {}).items())
        },
        "created_at": time.time(),
    }
    _write_json_atomic(path, payload)


def _load_eligibility_cache(
    path: Path,
    *,
    fingerprint: Mapping[str, Any],
    source_rows: int,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != "prefix_denoising_eligibility_cache_v1":
        raise ValueError(f"invalid prefix-denoising eligibility cache schema: {path}")
    if payload.get("fingerprint") != _json_ready(fingerprint):
        raise ValueError(f"prefix-denoising eligibility cache fingerprint mismatch: {path}")
    if int(payload.get("source_rows") or -1) != int(source_rows):
        raise ValueError(f"prefix-denoising eligibility cache row-count mismatch: {path}")
    result = _eligibility_from_cache_payload(payload, path=path)
    logger.info(
        "prefix-denoising eligibility cache loaded: eligible=%s total=%s cache=%s",
        len(result["eligible_indices"]),
        int(source_rows),
        path,
    )
    return result


def _load_compatible_eligibility_cache(
    cache_dir: Path,
    *,
    fingerprint: Mapping[str, Any],
    source_rows: int,
    exact_cache_path: Path,
) -> dict[str, Any] | None:
    """Load a cache with the same admission/length semantics but a different run key."""

    reuse_fingerprint = _eligibility_reuse_fingerprint(fingerprint)
    for candidate in sorted(cache_dir.glob("*/eligibility.json")):
        if candidate == exact_cache_path:
            continue
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("schema_version") != "prefix_denoising_eligibility_cache_v1":
                continue
            if int(payload.get("source_rows") or -1) != int(source_rows):
                continue
            cached_fingerprint = payload.get("fingerprint")
            if not isinstance(cached_fingerprint, Mapping):
                continue
            if (
                _eligibility_reuse_fingerprint(cached_fingerprint)
                != reuse_fingerprint
            ):
                continue
            result = _eligibility_from_cache_payload(payload, path=candidate)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "prefix-denoising compatible eligibility cache ignored: cache=%s reason=%s",
                candidate,
                exc,
            )
            continue
        logger.info(
            "prefix-denoising compatible eligibility cache loaded: source_cache=%s target_cache=%s",
            candidate,
            exact_cache_path,
        )
        return result
    return None


def _eligibility_from_cache_payload(
    payload: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, Any]:
    eligible_indices = tuple(int(index) for index in payload.get("eligible_indices") or [])
    static_lengths = {
        int(index): int(length)
        for index, length in (payload.get("static_lengths") or [])
    }
    skip_counters = {
        str(key): int(value)
        for key, value in (payload.get("skip_counters") or {}).items()
    }
    for index in eligible_indices:
        if index not in static_lengths:
            raise ValueError(
                "prefix-denoising eligibility cache missing static length for "
                f"eligible index {index}: {path}"
            )
    return {
        "eligible_indices": eligible_indices,
        "static_lengths": static_lengths,
        "skip_counters": skip_counters,
    }


def _wait_for_file(path: Path, *, timeout_s: float) -> None:
    timeout = float(timeout_s)
    if not math.isfinite(timeout):
        raise ValueError(f"wait_timeout_s must be finite, got {timeout_s!r}")
    if timeout < 0:
        raise ValueError(f"wait_timeout_s must be >= 0, got {timeout_s!r}")
    start = time.monotonic()
    while not path.exists():
        if timeout > 0 and time.monotonic() - start > timeout:
            raise TimeoutError(f"timed out waiting for prefix-denoising cache: {path}")
        time.sleep(2.0)


def _resolve_rank_world() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()), int(torch.distributed.get_world_size())
    rank_raw = os.environ.get("RANK")
    world_raw = os.environ.get("WORLD_SIZE")
    try:
        rank = int(rank_raw) if rank_raw is not None else 0
    except ValueError:
        rank = 0
    try:
        world_size = int(world_raw) if world_raw is not None else 1
    except ValueError:
        world_size = 1
    return rank, max(world_size, 1)


def _default_eligibility_fingerprint(
    *,
    rows: Sequence[Mapping[str, Any]],
    image_root: str | Path,
    user_prompt: str,
    system_prompt: str | None,
    prefix_denoising: PrefixDenoisingConfig,
    swift_template: Any,
    max_length: int,
    dataset_name: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": "prefix_denoising_eligibility_fingerprint_v1",
        "dataset_name": str(dataset_name),
        "source_rows": int(len(rows)),
        "image_root": str(Path(image_root).expanduser().resolve(strict=False)),
        "user_prompt": str(user_prompt),
        "system_prompt": system_prompt,
        "seed": int(seed),
        "max_length": int(max_length),
        "prefix_denoising": _prefix_denoising_fingerprint(prefix_denoising),
        "fast_estimator": prefix_denoising_fast_estimator_fingerprint(swift_template),
    }


def _prefix_denoising_fingerprint(config: PrefixDenoisingConfig) -> dict[str, Any]:
    return {
        "enabled": bool(config.enabled),
        "noise": {
            "center_shift_frac": float(config.noise.center_shift_frac),
            "uniform_scale_range": [
                float(config.noise.uniform_scale_range[0]),
                float(config.noise.uniform_scale_range[1]),
            ],
        },
        "current_object_kl": {
            "weight": float(config.current_object_kl.weight),
            "window_radius": int(config.current_object_kl.window_radius),
            "num_objects_per_image": int(
                config.current_object_kl.num_objects_per_image
            ),
        },
    }


def _eligibility_reuse_fingerprint(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    """Return the cache-sharing key for admission and static-length semantics only."""

    payload = _json_ready(fingerprint)
    if not isinstance(payload, dict):
        return {"fingerprint": payload}
    payload = copy.deepcopy(payload)
    prefix_denoising = payload.get("prefix_denoising")
    if isinstance(prefix_denoising, dict):
        # KL site sampling/weighting changes loss metadata, not whether a row is
        # eligible or how long the clean/noisy full branches are.
        prefix_denoising.pop("current_object_kl", None)
    for key in (
        "artifact_subdir",
        "checkpoint_mode",
        "dataset_jsonl_mtime_ns",
        "output_dir",
        "run_name",
        "save_delay_steps",
        "save_last_epoch",
        "save_steps",
        "save_strategy",
        "save_total_limit",
    ):
        payload.pop(key, None)
    return payload


def _fingerprint_digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        _json_ready(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, ensure_ascii=True, sort_keys=True)
    tmp.replace(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def materialize_hybrid_model_ready_item(
    *,
    sample: HybridPrefixDenoisingSample,
    dataset_name: str,
    base_idx: int | None = None,
) -> dict[str, Any]:
    if not sample.ok or sample.clean_full is None or sample.noisy_full is None:
        raise ValueError(
            f"cannot materialize ineligible prefix-denoising sample: {sample.skip_reason}"
        )
    clean = sample.clean_full
    noisy = sample.noisy_full
    input_ids = tuple(clean.input_ids) + tuple(noisy.input_ids)
    labels = tuple(clean.labels) + tuple(noisy.labels)
    attention_mask = tuple(clean.attention_mask) + tuple(noisy.attention_mask)
    length = len(input_ids)
    if len(labels) != length or len(attention_mask) != length:
        raise ValueError(
            "prefix-denoising materialized length invariant failed: "
            f"input_ids={length}, labels={len(labels)}, "
            f"attention_mask={len(attention_mask)}"
        )
    clean_start = 0
    clean_end = len(clean.input_ids)
    noisy_start = clean_end
    noisy_end = noisy_start + len(noisy.input_ids)
    item: dict[str, Any] = {
        "input_ids": list(input_ids),
        "labels": list(labels),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0),
        "length": length,
        "dataset": dataset_name,
        "sample_id": sample.hybrid_sample_id,
        "prefix_denoising_segment_meta": (
            _segment_meta(clean, start=clean_start, end=clean_end),
            _segment_meta(noisy, start=noisy_start, end=noisy_end),
        ),
        "prefix_denoising_hybrid": sample,
    }
    if base_idx is not None:
        item["base_idx"] = int(base_idx)
    item.update(_combine_segment_extras(clean, noisy))
    return item


def _segment_meta(
    segment: PrefixDenoisingSegment, *, start: int, end: int
) -> dict[str, Any]:
    return {
        "hybrid_sample_id": segment.segment_id.rsplit(":", 1)[0],
        "segment_id": segment.segment_id,
        "branch_id": segment.branch_id,
        "local_token_start": int(start),
        "local_token_end": int(end),
        "local_supervised_positions": tuple(segment.supervised_positions),
        "ce_denominator": int(segment.ce_denominator),
    }


def _combine_segment_extras(
    clean: PrefixDenoisingSegment,
    noisy: PrefixDenoisingSegment,
) -> dict[str, Any]:
    clean_extras = dict(clean.metadata.get("encoded_extras", {}))  # type: ignore[arg-type]
    noisy_extras = dict(noisy.metadata.get("encoded_extras", {}))  # type: ignore[arg-type]
    combined: dict[str, Any] = {}
    for key in sorted(set(clean_extras) | set(noisy_extras)):
        if key in _CORE_SEGMENT_KEYS:
            continue
        if key not in clean_extras:
            combined[key] = noisy_extras[key]
            continue
        if key not in noisy_extras:
            combined[key] = clean_extras[key]
            continue
        combined[key] = _combine_encoded_value(clean_extras[key], noisy_extras[key])
    return combined


def _combine_encoded_value(clean_value: Any, noisy_value: Any) -> Any:
    if _is_torch_tensor(clean_value) and _is_torch_tensor(noisy_value):
        import torch

        try:
            return torch.cat((clean_value, noisy_value), dim=0)
        except RuntimeError:
            return (clean_value, noisy_value)
    if isinstance(clean_value, list) and isinstance(noisy_value, list):
        return list(clean_value) + list(noisy_value)
    if isinstance(clean_value, tuple) and isinstance(noisy_value, tuple):
        return tuple(clean_value) + tuple(noisy_value)
    if clean_value == noisy_value:
        return clean_value
    return (clean_value, noisy_value)


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and hasattr(value, "shape")


def _mix_seed(seed: int, epoch: int, base_idx: int) -> int:
    value = (
        (int(seed) & 0xFFFFFFFF)
        ^ ((int(epoch) + 1) * 0x9E3779B1)
        ^ ((int(base_idx) + 1) * 0xC2B2AE35)
    )
    return int(value & 0xFFFFFFFF)


def _make_base_sample_id(dataset_name: str, base_idx: int) -> str:
    return f"{dataset_name}:{int(base_idx)}"


__all__ = [
    "PrefixDenoisingTrainingDataset",
    "build_prefix_denoising_eligibility_index",
    "materialize_hybrid_model_ready_item",
]
