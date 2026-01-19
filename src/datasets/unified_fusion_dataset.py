"""Unified fusion dataset for multi-JSONL dense-caption training.

This dataset reads multiple JSONL pools (from a FusionConfig) and builds a
per-epoch sampling schedule. CoordExp v1 semantics:
- `targets` and `sources` are both accepted in configs, but are treated the same
  at runtime (no target/source semantic split).
- Each dataset contributes `round(len(pool) * ratio)` samples per epoch.
- Evaluation uses any dataset entry with a non-null `val_jsonl`; missing/null
  `val_jsonl` skips evaluation for that dataset.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, MutableMapping, Optional, cast

from torch.utils.data import get_worker_info

from src.config.schema import CoordTokensConfig
from src.coord_tokens.validator import annotate_coord_tokens

from ..config.prompts import USER_PROMPT_SUMMARY
from .builders import JSONLinesBuilder
from .contracts import validate_conversation_record
from .dense_caption import LAST_SAMPLE_DEBUG, BaseCaptionDataset
from .fusion import FusionConfig
from .fusion_types import FusionDatasetSpec
from .preprocessors import AugmentationPreprocessor, ObjectCapPreprocessor
from .utils import load_jsonl, sort_objects_by_topleft


@dataclass(frozen=True)
class _PromptResolution:
    user: str
    system: Optional[str]
    source: Literal["default", "dataset"]


@dataclass
class _DatasetPolicy:
    spec: FusionDatasetSpec
    prompts: _PromptResolution
    max_objects_per_image: Optional[int]
    seed: Optional[int]


class FusionCaptionDataset(BaseCaptionDataset):
    """Fusion dataset that samples from multiple JSONL pools with one template."""

    def __init__(
        self,
        *,
        fusion_config: FusionConfig,
        base_template: Any,
        user_prompt: str,
        emit_norm: Literal["none"],
        json_format: Literal["standard"],
        augmenter: Optional[Any],
        bypass_prob: float,
        curriculum_state: Optional[MutableMapping[str, Any]],
        use_summary: bool,
        system_prompt_dense: Optional[str],
        system_prompt_summary: Optional[str],
        coord_tokens: Optional[CoordTokensConfig] = None,
        seed: int = 42,
        shuffle: bool = True,
        sample_limit: Optional[int] = None,
        split: Literal["train", "eval"] = "train",
        object_ordering: Literal["sorted", "random"] = "sorted",
    ) -> None:
        self._fusion_config = fusion_config
        self._split: Literal["train", "eval"] = split
        self._augmenter = augmenter
        self.bypass_prob = float(bypass_prob)
        self.curriculum_state = curriculum_state
        self.coord_tokens = coord_tokens or CoordTokensConfig()
        self._shuffle = bool(shuffle)
        self._sample_limit = sample_limit
        self._epoch = 0
        self._schedule: list[tuple[str, int]] = []
        self._epoch_counts: dict[str, int] = {}
        self._policies: dict[str, _DatasetPolicy] = {}
        self._record_pools: dict[str, list[dict[str, Any]]] = {}
        self._preprocessors_aug: dict[str, AugmentationPreprocessor] = {}
        self._preprocessors_cap: dict[str, ObjectCapPreprocessor] = {}
        self.epoch_plan: dict[str, dict[str, Any]] = {}
        self._hard_sample_plan: dict[str, Any] | None = None
        self.object_ordering: Literal["sorted", "random"] = object_ordering

        self._dataset_order = [spec.name for spec in fusion_config.datasets]
        default_system_prompt = system_prompt_dense

        # Load pools for the selected split.
        for spec in fusion_config.datasets:
            path: Optional[Path]
            if split == "eval":
                path = spec.val_jsonl
                if path is None:
                    # Explicitly skipped in eval.
                    self._record_pools[spec.name] = []
                    continue
            else:
                path = spec.train_jsonl

            records = self._load_records(
                path,
                limit=sample_limit if isinstance(sample_limit, int) and sample_limit > 0 else None,
            )
            if not records:
                if split == "train" and float(spec.ratio) > 0:
                    raise ValueError(
                        f"Fusion dataset '{spec.name}' is empty while ratio={spec.ratio}"
                    )
                if split == "eval":
                    raise ValueError(
                        f"Fusion eval dataset '{spec.name}' is empty (val_jsonl={path})"
                    )
            self._record_pools[spec.name] = [
                self._annotate_record(rec, spec) for rec in records
            ]

            prompts = self._resolve_prompts(
                spec,
                default_user_prompt=user_prompt,
                default_system_prompt=default_system_prompt,
                ordering=self.object_ordering,
            )
            self._policies[spec.name] = _DatasetPolicy(
                spec=spec,
                prompts=prompts,
                max_objects_per_image=spec.max_objects_per_image,
                seed=spec.seed,
            )

            if (
                split == "train"
                and self._augmenter is not None
                and spec.supports_augmentation
            ):
                curriculum_state = (
                    self.curriculum_state if spec.supports_curriculum else None
                )
                self._preprocessors_aug[spec.name] = AugmentationPreprocessor(
                    augmenter=self._augmenter,
                    bypass_prob=self.bypass_prob,
                    curriculum_state=curriculum_state,
                    coord_tokens_enabled=self.coord_tokens.enabled,
                )
            if spec.max_objects_per_image is not None:
                self._preprocessors_cap[spec.name] = ObjectCapPreprocessor(
                    spec.max_objects_per_image
                )

        # Initialize parent BaseCaptionDataset with a single template instance.
        super().__init__(
            base_records=[],
            template=base_template,
            user_prompt=user_prompt,
            emit_norm=emit_norm,
            json_format=json_format,
            augmenter=None,
            preprocessor=None,
            bypass_prob=bypass_prob,
            curriculum_state=curriculum_state,
            use_summary=use_summary,
            system_prompt_dense=system_prompt_dense,
            system_prompt_summary=system_prompt_summary,
            coord_tokens=self.coord_tokens,
            seed=seed,
            dataset_name="fusion",
            allow_empty=True,
            object_ordering=object_ordering,
        )

        self.set_epoch(0)

    @staticmethod
    def _annotate_record(
        record: MutableMapping[str, Any], spec: FusionDatasetSpec
    ) -> dict[str, Any]:
        """Annotate record with fusion metadata for debugging/analysis."""
        annotated = copy.deepcopy(dict(record))
        metadata = annotated.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["_fusion_source"] = spec.name
        metadata["_fusion_template"] = spec.template
        metadata["_fusion_domain"] = spec.domain
        annotated["metadata"] = metadata
        return annotated

    @staticmethod
    def _load_records(path: Path, *, limit: Optional[int]) -> list[MutableMapping[str, Any]]:
        records = load_jsonl(str(path), resolve_relative=True)
        records = [cast(MutableMapping[str, Any], rec) for rec in records]
        if limit is not None and limit > 0:
            records = records[:limit]
        validated: list[MutableMapping[str, Any]] = []
        for idx, record in enumerate(records):
            try:
                validated.append(
                    cast(
                        MutableMapping[str, Any],
                        copy.deepcopy(validate_conversation_record(record)),
                    )
                )
            except ValueError as exc:
                raise ValueError(f"Record {idx} in {path} is invalid: {exc}") from exc
        return validated

    @staticmethod
    def _mix_seed(base: int, *parts: int) -> int:
        seed_val = base & 0xFFFFFFFF
        for offset, part in enumerate(parts):
            seed_val ^= ((int(part) + offset + 1) * 0x9E3779B1) & 0xFFFFFFFF
        return seed_val & 0xFFFFFFFF

    def _resolve_prompts(
        self,
        spec: FusionDatasetSpec,
        *,
        default_user_prompt: str,
        default_system_prompt: Optional[str],
        ordering: Literal["sorted", "random"] = "sorted",
    ) -> _PromptResolution:
        # Dataset-level overrides are supported; otherwise fall back to the
        # runner's resolved prompts.
        user_prompt = spec.prompt_user or default_user_prompt
        system_prompt = spec.prompt_system or default_system_prompt

        source = "dataset" if (spec.prompt_user or spec.prompt_system) else "default"
        return _PromptResolution(user=user_prompt, system=system_prompt, source=source)

    def _build_train_schedule(self) -> None:
        schedule: list[tuple[str, int]] = []
        counts: dict[str, int] = {}

        for dataset_name in self._dataset_order:
            policy = self._policies.get(dataset_name)
            pool = self._record_pools.get(dataset_name, [])
            spec = policy.spec if policy is not None else None
            if spec is None:
                counts[dataset_name] = 0
                continue

            pool_len = len(pool)
            quota = round(pool_len * float(spec.ratio))
            if quota <= 0:
                counts[dataset_name] = 0
                continue
            if pool_len <= 0:
                raise ValueError(
                    f"Fusion dataset '{dataset_name}' is empty while ratio={spec.ratio}"
                )

            rng = random.Random(
                self._mix_seed(self.seed, self._epoch, policy.seed or 0, 0xA1)
            )
            indices = list(range(pool_len))
            if pool_len > 1:
                rng.shuffle(indices)
            if quota <= pool_len:
                sampled = indices[:quota]
            else:
                sampled = indices[:]
                if not spec.sample_without_replacement:
                    sampled.extend(
                        rng.randrange(pool_len) for _ in range(quota - pool_len)
                    )

            counts[dataset_name] = len(sampled)
            schedule.extend((dataset_name, idx) for idx in sampled)

        if self._shuffle and len(schedule) > 1:
            rng_shuffle = random.Random(self._mix_seed(self.seed, self._epoch, 0xC3))
            rng_shuffle.shuffle(schedule)

        self._schedule = schedule
        self._epoch_counts = counts
        self._update_epoch_plan(eval_mode=False)

    def _build_eval_schedule(self) -> None:
        schedule: list[tuple[str, int]] = []
        counts: dict[str, int] = {}

        for dataset_name in self._dataset_order:
            pool = self._record_pools.get(dataset_name, [])
            if not pool:
                continue
            counts[dataset_name] = len(pool)
            schedule.extend((dataset_name, idx) for idx in range(len(pool)))

        self._schedule = schedule
        self._epoch_counts = counts
        self._update_epoch_plan(eval_mode=True)

    def _update_epoch_plan(self, *, eval_mode: bool) -> None:
        plan: dict[str, dict[str, Any]] = {}
        for dataset_name in self._dataset_order:
            policy = self._policies.get(dataset_name)
            if policy is None:
                continue
            plan[dataset_name] = {
                "count": self._epoch_counts.get(dataset_name, 0),
                "eval_mode": eval_mode,
                "ratio": float(policy.spec.ratio),
                "prompt_source": policy.prompts.source,
                "max_objects_per_image": policy.max_objects_per_image,
            }
        self.epoch_plan = plan

    def set_hard_sample_plan(self, plan: Optional[MutableMapping[str, Any]]) -> None:
        # Hard-sample mining is deprecated in CoordExp; keep the hook for
        # compatibility but do not change fusion scheduling semantics.
        self._hard_sample_plan = dict(plan) if plan is not None else None

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._rng = random.Random(self._seed_for_epoch(self._epoch))
        if self._split == "eval":
            self._build_eval_schedule()
        else:
            self._build_train_schedule()

    def __len__(self) -> int:
        return len(self._schedule)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not self._schedule:
            raise IndexError("FusionCaptionDataset is empty")

        dataset_name, base_idx = self._schedule[index % len(self._schedule)]
        pool = self._record_pools.get(dataset_name, [])
        record = copy.deepcopy(pool[base_idx])
        policy = self._policies[dataset_name]

        worker = get_worker_info()
        seed_local = self._rng.randrange(0, 2**32 - 1)
        if worker is not None:
            seed_local ^= ((worker.id + 1) * 0xC2B2AE35) & 0xFFFFFFFF
        rng_local = random.Random(seed_local & 0xFFFFFFFF)

        was_augmented = False
        cap_applied = False
        objects_before = len(record.get("objects") or [])

        aug_pre = self._preprocessors_aug.get(dataset_name)
        if self._split == "train" and aug_pre is not None:
            if hasattr(aug_pre, "rng"):
                aug_pre.rng = rng_local
            processed = aug_pre(record)
            if processed is None:
                raise ValueError("Preprocessor removed the record; dataset does not duplicate samples")
            record = processed
            was_augmented = True

        cap_pre = self._preprocessors_cap.get(dataset_name)
        if cap_pre is not None and policy.max_objects_per_image is not None:
            if hasattr(cap_pre, "rng"):
                cap_pre.rng = rng_local
            capped = cap_pre(record)
            if capped is None:
                raise ValueError("Preprocessor removed the record; dataset does not duplicate samples")
            record = capped
            objects_after = len(record.get("objects") or [])
            cap_applied = objects_after < objects_before or (objects_before > policy.max_objects_per_image)
        else:
            objects_after = len(record.get("objects") or [])

        # Apply ordering or randomization for ablations.
        objects_list = record.get("objects") or []
        if isinstance(objects_list, list) and objects_list:
            if self.object_ordering == "sorted":
                record["objects"] = sort_objects_by_topleft(objects_list)
            elif self.object_ordering == "random":
                objs_copy = list(objects_list)
                rng_local.shuffle(objs_copy)
                record["objects"] = objs_copy

        if self.coord_tokens.enabled:
            annotate_coord_tokens(record)

        # Build conversation (dense-caption path).
        mode = self.mode
        prompts = policy.prompts
        system_prompt = prompts.system
        if mode == "summary" and self.system_prompt_summary is not None:
            system_prompt = self.system_prompt_summary

        builder = JSONLinesBuilder(
            user_prompt=USER_PROMPT_SUMMARY if mode == "summary" else prompts.user,
            emit_norm=self.emit_norm,
            mode=mode,
            json_format=self.json_format,
            coord_tokens_enabled=self.coord_tokens.enabled,
        )
        merged = builder.build_many([record])
        conversation_messages = copy.deepcopy(merged.get("messages", []) or [])
        if system_prompt:
            conversation_messages = [{"role": "system", "content": system_prompt}, *conversation_messages]

        original_system = getattr(self.template, "system", None)
        try:
            if system_prompt:
                try:
                    self.template.system = system_prompt
                except Exception:
                    pass
            encoded = self.template.encode(merged, return_length=True)
        finally:
            if system_prompt is not None and original_system is not None:
                try:
                    self.template.system = original_system
                except Exception:
                    pass

        encoded["messages"] = conversation_messages
        for key in ("assistant_payload", "objects", "metadata"):
            if key in merged:
                encoded[key] = copy.deepcopy(merged[key])

        try:
            info = {
                "dataset": dataset_name,
                "base_idx": base_idx,
                "objects": objects_after,
                "mode": mode,
                "prompt_source": prompts.source,
                "augmentation_enabled": was_augmented,
                "object_cap_applied": cap_applied,
                "object_cap_limit": policy.max_objects_per_image,
            }
            input_ids = encoded.get("input_ids")
            if input_ids is not None and hasattr(input_ids, "__len__"):
                try:
                    info["input_length"] = len(input_ids)
                except Exception:
                    pass
            sample_id = self._make_sample_id(dataset_name, base_idx)
            encoded["sample_id"] = sample_id
            encoded["dataset"] = dataset_name
            encoded["base_idx"] = base_idx
            self.last_sample_debug = info
            LAST_SAMPLE_DEBUG.update(info)
        except Exception:
            pass

        return encoded


# Backward compatibility alias
UnifiedFusionDataset = FusionCaptionDataset
