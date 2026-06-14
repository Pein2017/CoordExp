from __future__ import annotations

import copy
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.common.io import load_jsonl_with_diagnostics
from src.config.schema import PrefixDenoisingConfig
from src.detection.dataset import resolve_detection_jsonl_image_root

from .builder import build_hybrid_prefix_denoising_sample
from .types import HybridPrefixDenoisingSample, PrefixDenoisingSegment

_CORE_SEGMENT_KEYS = {"input_ids", "labels", "attention_mask"}


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
        eligibility = build_prefix_denoising_eligibility_index(
            self.rows,
            swift_template=self.swift_template,
            image_root=self.image_root,
            user_prompt=self.user_prompt,
            system_prompt=self.system_prompt,
            prefix_denoising=self.prefix_denoising,
            max_length=self.max_length,
            dataset_name=self.dataset_name,
            seed=self.seed,
        )
        self._eligible_indices = tuple(eligibility["eligible_indices"])
        self._static_lengths = dict(eligibility["static_lengths"])
        self.skip_counters = Counter(eligibility["skip_counters"])
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
        )

    def __len__(self) -> int:
        return len(self._eligible_indices)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _static_packing_length(self, index: int) -> int | None:
        base_idx = self._base_index(index)
        return self._static_lengths.get(base_idx)

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
) -> dict[str, Any]:
    eligible_indices: list[int] = []
    static_lengths: dict[int, int] = {}
    skip_counters: Counter[str] = Counter()
    for base_idx, row in enumerate(rows):
        sample = build_hybrid_prefix_denoising_sample(
            row,
            base_sample_id=_make_base_sample_id(dataset_name, base_idx),
            image_root=image_root,
            swift_template=swift_template,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            prefix_denoising=prefix_denoising,
            epoch=0,
            rng=random.Random(_mix_seed(seed, 0, base_idx)),
            max_length=max_length,
        )
        if not sample.ok:
            skip_counters[str(sample.skip_reason or "unknown_skip_reason")] += 1
            continue
        eligible_indices.append(base_idx)
        static_lengths[base_idx] = int(sample.total_length)
    return {
        "eligible_indices": tuple(eligible_indices),
        "static_lengths": static_lengths,
        "skip_counters": dict(skip_counters),
    }


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
    clean_start = 0
    clean_end = len(clean.input_ids)
    noisy_start = clean_end
    noisy_end = noisy_start + len(noisy.input_ids)
    item: dict[str, Any] = {
        "input_ids": list(input_ids),
        "labels": list(labels),
        "attention_mask": list(attention_mask),
        "length": len(input_ids),
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
