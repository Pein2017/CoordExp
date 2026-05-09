"""Length-bucketed sampling for latest detection datasets."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch.utils.data import Sampler
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    LengthGroupedSampler,
    get_length_grouped_indices,
)

from src.detection.dataset import DetectionTrainingDataset


@dataclass(frozen=True)
class LatestDetectionLengthBucketingConfig:
    """Runtime switch for row-atomic latest-detection length bucketing."""

    enabled: bool = False
    seed: int = 0
    cache_policy: str = "run_local_only"


@dataclass(frozen=True)
class LatestDetectionLengthBucketingProvenance:
    """Sampler provenance recorded in runtime artifacts."""

    enabled: bool
    mode: str
    length_source: str
    cache_policy: str
    sampler_class: str
    batch_size: int | None = None
    seed: int | None = None
    dataset_size: int | None = None
    min_length: int | None = None
    max_length: int | None = None
    mean_length: float | None = None
    world_size: int | None = None
    rank: int | None = None
    drop_last: bool | None = None
    disabled_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Dictionary form without null-valued optional fields."""

        return {
            key: value
            for key, value in self.__dict__.items()
            if value is not None
        }


@dataclass
class LatestDetectionLengthProvider:
    """Run-local encoded-length provider for latest detection rows."""

    dataset: DetectionTrainingDataset
    cache_policy: str = "run_local_only"
    _lengths: list[int] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the dataset contract needed for sidecar-safe bucketing."""

        if not isinstance(self.dataset, DetectionTrainingDataset):
            raise TypeError(
                "LatestDetectionLengthProvider requires DetectionTrainingDataset"
            )
        if self.cache_policy != "run_local_only":
            raise ValueError("latest detection length cache must be run_local_only")

    def length_for_row(
        self,
        base_idx: int,
        *,
        forced_rollin_k: int | None = None,
        epoch: int | None = None,
    ) -> int:
        """Encoded input length for one base row.

        ``forced_rollin_k`` is validated by the dataset and exists to make the
        prefix-rollin invariance contract explicit. The rendered full assistant
        sequence is row-local, so K changes labels and sidecars but not the
        encoded sequence length.
        """

        return int(
            self.dataset.encoded_length_for_row(
                base_idx,
                forced_rollin_k=forced_rollin_k,
                epoch=epoch,
            )
        )

    def all_lengths(self) -> list[int]:
        """Run-local encoded lengths for every row without calling ``__getitem__``."""

        if self._lengths is None:
            self._lengths = [
                self.length_for_row(base_idx=index) for index in range(len(self.dataset))
            ]
        return list(self._lengths)

    def provenance(
        self,
        *,
        sampler_class: str,
        batch_size: int,
        seed: int,
        world_size: int,
        rank: int,
        drop_last: bool,
    ) -> LatestDetectionLengthBucketingProvenance:
        """Provenance summary for runtime manifests."""

        lengths = self.all_lengths()
        mean_length = (
            float(sum(lengths)) / float(len(lengths)) if lengths else None
        )
        return LatestDetectionLengthBucketingProvenance(
            enabled=True,
            mode="row_atomic_length_bucketing",
            length_source="DetectionTrainingDataset.encoded_length_for_row",
            cache_policy=self.cache_policy,
            sampler_class=sampler_class,
            batch_size=int(batch_size),
            seed=int(seed),
            dataset_size=len(lengths),
            min_length=min(lengths) if lengths else None,
            max_length=max(lengths) if lengths else None,
            mean_length=mean_length,
            world_size=int(world_size),
            rank=int(rank),
            drop_last=bool(drop_last),
        )


class LatestDetectionLengthGroupedSampler(LengthGroupedSampler):
    """Deterministic epoch-aware length-grouped sampler with explicit lengths."""

    def __init__(
        self,
        *,
        batch_size: int,
        lengths: Sequence[int],
        seed: int,
    ) -> None:
        super().__init__(batch_size=int(batch_size), lengths=[int(v) for v in lengths])
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Epoch value used to reshuffle row order deterministically."""

        self.epoch = int(epoch)

    def __iter__(self):
        """Indices grouped by encoded length for the current epoch."""

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = get_length_grouped_indices(
            self.lengths,
            self.batch_size,
            generator=generator,
        )
        return iter(indices)


def build_latest_detection_length_grouped_sampler(
    *,
    batch_size: int,
    seed: int,
    drop_last: bool,
    dataset: DetectionTrainingDataset | None = None,
    lengths: Sequence[int] | None = None,
    world_size: int = 1,
    rank: int = 0,
) -> Sampler[int]:
    """Length-grouped sampler with explicit precomputed lengths.

    The helper accepts either a latest-detection dataset or an explicit length
    sequence so unit tests can cover DDP sharding without constructing full
    multimodal rows.
    """

    if lengths is None:
        if dataset is None:
            raise ValueError("dataset or lengths must be provided")
        lengths = LatestDetectionLengthProvider(dataset).all_lengths()
    resolved_lengths = [int(value) for value in lengths]
    if not resolved_lengths:
        raise ValueError("length grouped sampler requires at least one length")

    if int(world_size) > 1:
        return DistributedLengthGroupedSampler(
            batch_size=int(batch_size),
            lengths=resolved_lengths,
            num_replicas=int(world_size),
            rank=int(rank),
            seed=int(seed),
            drop_last=bool(drop_last),
        )

    return LatestDetectionLengthGroupedSampler(
        batch_size=int(batch_size),
        lengths=resolved_lengths,
        seed=int(seed),
    )


class LatestDetectionLengthGroupedTrainerMixin:
    """Trainer mixin that injects explicit latest-detection lengths into HF grouping."""

    def _distributed_sampler_context(self) -> tuple[int, int]:
        """World-size and rank values from the active trainer arguments."""

        world_size = int(getattr(self.args, "world_size", 1) or 1)
        rank = int(getattr(self.args, "process_index", 0) or 0)
        if world_size <= 1:
            return 1, 0
        return world_size, rank

    def _fallback_train_sampler(self, train_dataset):
        """Superclass sampler for trainer variants with older method signatures."""

        method = super()._get_train_sampler
        parameters = inspect.signature(method).parameters
        if "train_dataset" in parameters:
            return method(train_dataset=train_dataset)
        return method()

    def _fallback_eval_sampler(self, eval_dataset):
        """Superclass eval sampler for trainer variants with older method signatures."""

        method = super()._get_eval_sampler
        parameters = inspect.signature(method).parameters
        if "eval_dataset" in parameters:
            return method(eval_dataset=eval_dataset)
        return method(eval_dataset)

    def _get_train_sampler(self, train_dataset=None):
        """Build a row-atomic length-grouped sampler when explicitly enabled."""

        cfg = getattr(self, "latest_detection_length_bucketing", None)
        if not isinstance(cfg, LatestDetectionLengthBucketingConfig) or not cfg.enabled:
            return self._fallback_train_sampler(train_dataset)

        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if not isinstance(dataset, DetectionTrainingDataset):
            return self._fallback_train_sampler(train_dataset)

        batch_size = int(getattr(self.args, "train_batch_size", 1) or 1) * int(
            getattr(self.args, "gradient_accumulation_steps", 1) or 1
        )
        drop_last = bool(getattr(self.args, "dataloader_drop_last", False))
        world_size, rank = self._distributed_sampler_context()

        sampler = build_latest_detection_length_grouped_sampler(
            dataset=dataset,
            batch_size=batch_size,
            seed=int(cfg.seed),
            drop_last=drop_last,
            world_size=world_size,
            rank=rank,
        )
        provider = LatestDetectionLengthProvider(dataset, cache_policy=cfg.cache_policy)
        self.latest_detection_length_bucketing_runtime = provider.provenance(
            sampler_class=type(sampler).__name__,
            batch_size=batch_size,
            seed=int(cfg.seed),
            world_size=world_size,
            rank=rank,
            drop_last=drop_last,
        ).to_dict()
        return sampler

    def _get_eval_sampler(self, eval_dataset):
        """Build an explicit-length eval sampler without iterating dataset rows."""

        cfg = getattr(self, "latest_detection_length_bucketing", None)
        if not isinstance(cfg, LatestDetectionLengthBucketingConfig) or not cfg.enabled:
            return self._fallback_eval_sampler(eval_dataset)
        if not isinstance(eval_dataset, DetectionTrainingDataset):
            return self._fallback_eval_sampler(eval_dataset)

        batch_size = int(
            getattr(self.args, "eval_batch_size", None)
            or getattr(self.args, "per_device_eval_batch_size", 1)
            or 1
        )
        world_size, rank = self._distributed_sampler_context()
        return build_latest_detection_length_grouped_sampler(
            dataset=eval_dataset,
            batch_size=batch_size,
            seed=int(cfg.seed),
            drop_last=False,
            world_size=world_size,
            rank=rank,
        )


def disabled_length_bucketing_provenance(
    *, reason: str
) -> LatestDetectionLengthBucketingProvenance:
    """Disabled runtime-provenance object for explicit artifact truth."""

    return LatestDetectionLengthBucketingProvenance(
        enabled=False,
        mode="none",
        length_source="none",
        cache_policy="none",
        sampler_class="none",
        disabled_reason=str(reason),
    )
