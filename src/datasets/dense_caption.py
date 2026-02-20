from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Literal, Mapping, MutableMapping, Optional, Sequence

from torch.utils.data import Dataset

from src.common.object_field_order import (
    ObjectFieldOrder,
    normalize_object_field_order,
)
from src.config.prompts import USER_PROMPT_SUMMARY
from src.config.schema import CoordTokensConfig
from src.coord_tokens.validator import annotate_coord_tokens

from .builders import JSONLinesBuilder
from .contracts import ConversationRecord, validate_conversation_record
from .preprocessors import AugmentationPreprocessor, SequentialPreprocessor
from .utils import (
    find_first_unsorted_object_pair_by_topleft,
    load_jsonl,
)

# Exposed for debugging (e.g., OOM tracing)
LAST_SAMPLE_DEBUG: Dict[str, Any] = {}


class BaseCaptionDataset(Dataset):
    """Base caption dataset without dynamic pairing.

    Each sample corresponds to a single base record. The dataset supports
    optional augmentation, summary/dense mode selection, and epoch-level shuffling
    consistent with the legacy dynamic pairing pipeline.
    """

    def __init__(
        self,
        base_records: Sequence[Any],
        template: Any,
        user_prompt: str,
        emit_norm: Literal["none"],
        json_format: Literal["standard"],
        augmenter: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
        use_summary: bool = False,
        system_prompt_dense: Optional[str] = None,
        system_prompt_summary: Optional[str] = None,
        bypass_prob: float = 0.0,
        seed: int = 2025,
        curriculum_state: Optional[MutableMapping[str, Any]] = None,
        dataset_name: Optional[str] = None,
        allow_empty: bool = False,
        coord_tokens: Optional[CoordTokensConfig] = None,
        object_ordering: Literal["sorted", "random"] = "sorted",
        object_field_order: ObjectFieldOrder = "desc_first",
    ):
        self.use_summary = bool(use_summary)
        self.system_prompt_dense = system_prompt_dense
        self.system_prompt_summary = system_prompt_summary
        self.user_prompt = user_prompt
        self.emit_norm: Literal["none"] = emit_norm
        self.json_format: Literal["standard"] = json_format
        self.bypass_prob = float(bypass_prob)
        self.seed = int(seed)
        self.template = template
        self.mode: Literal["dense", "summary"] = (
            "summary" if self.use_summary else "dense"
        )
        self.coord_tokens = coord_tokens or CoordTokensConfig()
        self.object_ordering: Literal["sorted", "random"] = object_ordering
        self.object_field_order = normalize_object_field_order(object_field_order)

        if self.use_summary:
            if self.system_prompt_summary is None:
                self.system_prompt_summary = getattr(self.template, "system", None)
            if self.system_prompt_summary is None:
                raise ValueError(
                    "system_prompt_summary is required when use_summary is true."
                )
            try:
                setattr(self.template, "system", self.system_prompt_summary)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Failed to set template.system for summary-mode dataset initialization."
                ) from exc
        else:
            if self.system_prompt_dense is None:
                self.system_prompt_dense = getattr(self.template, "system", None)
            if self.system_prompt_dense is not None:
                try:
                    setattr(self.template, "system", self.system_prompt_dense)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        "Failed to set template.system for dense-mode dataset initialization."
                    ) from exc

        validated_records: List[ConversationRecord] = []
        for idx, record in enumerate(base_records):
            try:
                validated = validate_conversation_record(record)
            except ValueError as exc:
                raise ValueError(f"Base record {idx} is invalid: {exc}") from exc
            validated_records.append(copy.deepcopy(validated))

        if not validated_records and not allow_empty:
            raise ValueError("BaseCaptionDataset requires at least one valid record")

        self.base_records = validated_records

        preprocessors = []
        if preprocessor is not None:
            preprocessors.append(preprocessor)
        if augmenter is not None:
            preprocessors.append(
                AugmentationPreprocessor(
                    augmenter=augmenter,
                    bypass_prob=self.bypass_prob,
                    curriculum_state=curriculum_state,
                    coord_tokens_enabled=self.coord_tokens.enabled,
                )
            )
        if preprocessors:
            self.preprocessor = (
                preprocessors[0]
                if len(preprocessors) == 1
                else SequentialPreprocessor(preprocessors)
            )
            if hasattr(self.preprocessor, "curriculum_state"):
                setattr(self.preprocessor, "curriculum_state", curriculum_state)
        else:
            self.preprocessor = None

        self._epoch = 0
        self._rng = random.Random(self._seed_for_epoch(self._epoch))
        self._index_perm = list(range(len(self.base_records)))
        self._hard_sample_plan: Dict[str, Any] | None = None
        self._rebuild_perm_for_epoch()
        self.dataset_name = dataset_name or "dataset"
        self.last_sample_debug: Dict[str, Any] = {}

    @staticmethod
    def _make_sample_id(dataset_name: str, base_idx: int) -> int:
        import zlib

        ns = zlib.crc32(dataset_name.encode("utf-8")) & 0xFFFF
        return (ns << 32) | (int(base_idx) & 0xFFFFFFFF)

    @staticmethod
    def from_jsonl(
        jsonl_path: str,
        template: Any,
        **kwargs,
    ) -> "BaseCaptionDataset":
        records = load_jsonl(jsonl_path, resolve_relative=True)
        # Optional sample limiting for quick smoke tests
        sample_limit = kwargs.pop("sample_limit", None)
        if isinstance(sample_limit, int) and sample_limit > 0:
            records = records[:sample_limit]
        elif isinstance(sample_limit, str) and sample_limit.isdigit():
            records = records[: int(sample_limit)]
        # Backward-compatibility: drop unused arg if present
        if "summary_ratio" in kwargs:
            raise TypeError(
                "summary_ratio is no longer supported; use use_summary instead."
            )
            kwargs.pop("use_detailed_caption", None)
            kwargs.pop("output_variant", None)  # Backward compat
        if kwargs.get("dataset_name") is None:
            from pathlib import Path

            path = Path(jsonl_path)
            parts = {p.lower() for p in path.parts}
            for candidate in ("lvis", "coco", "refcoco", "refcoco+", "refcocog", "vg"):
                if candidate in parts:
                    kwargs["dataset_name"] = candidate.replace("+", "plus")
                    break
            if kwargs.get("dataset_name") is None:
                kwargs["dataset_name"] = path.stem
        return BaseCaptionDataset(
            base_records=records,
            template=template,
            **kwargs,
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._rng = random.Random(self._seed_for_epoch(self._epoch))
        self._rebuild_perm_for_epoch()

    def _seed_for_epoch(self, epoch: int) -> int:
        base_seed = self.seed & 0xFFFFFFFF
        mixed = (base_seed ^ ((int(epoch) + 1) * 0x9E3779B1)) & 0xFFFFFFFF
        return mixed

    def _rebuild_perm_for_epoch(self) -> None:
        base_len = len(self.base_records)
        plan = self._hard_sample_plan or {}
        target_len = int(plan.get("target_epoch_size") or base_len)
        weights_map = plan.get("weights") if isinstance(plan, MutableMapping) else None

        if weights_map:
            indices = list(range(base_len))
            weights = [float(weights_map.get(i, 1.0)) for i in indices]
            self._index_perm = self._rng.choices(indices, weights=weights, k=target_len)
        else:
            perm = list(range(base_len))
            if len(perm) > 1:
                self._rng.shuffle(perm)
            if target_len == base_len:
                self._index_perm = perm
            elif target_len < base_len:
                self._index_perm = perm[:target_len]
            else:
                extra = self._rng.choices(perm, k=target_len - base_len)
                self._index_perm = perm + extra

    def set_hard_sample_plan(self, plan: Optional[Mapping[str, Any]]) -> None:
        self._hard_sample_plan = dict(plan) if plan is not None else None

    def __len__(self) -> int:
        return len(self.base_records)

    def _create_builder(self, mode: Literal["dense", "summary"]) -> JSONLinesBuilder:
        user_prompt = USER_PROMPT_SUMMARY if mode == "summary" else self.user_prompt
        return JSONLinesBuilder(
            user_prompt=user_prompt,
            emit_norm=self.emit_norm,
            mode=mode,
            json_format=self.json_format,
            coord_tokens_enabled=self.coord_tokens.enabled,
            object_field_order=self.object_field_order,
        )


    def _apply_object_ordering(
        self, record: Dict[str, Any], rng_local: random.Random
    ) -> None:
        objects_list = record.get("objects") or []
        if not isinstance(objects_list, list) or not objects_list:
            return

        if self.object_ordering == "sorted":
            unsorted = find_first_unsorted_object_pair_by_topleft(objects_list)
            if unsorted is not None:
                prev_idx, curr_idx, prev_anchor, curr_anchor = unsorted
                raise ValueError(
                    "Objects must already be top-left sorted when "
                    "custom.object_ordering='sorted'; "
                    f"found out-of-order pair at positions {prev_idx}->{curr_idx} "
                    f"with anchors {prev_anchor}->{curr_anchor}."
                )
            return

        if self.object_ordering == "random":
            objs_copy = list(objects_list)
            rng_local.shuffle(objs_copy)
            record["objects"] = objs_copy

    def _maybe_annotate_coord_tokens(self, record: Dict[str, Any]) -> None:
        if self.coord_tokens.enabled:
            annotate_coord_tokens(record)

    def _enforce_max_pixels(self, record: Mapping[str, Any], *, base_idx: int) -> None:
        """Fail fast if a record would exceed template.max_pixels.

        CoordExp forbids any runtime image resizing because it breaks grounding
        coordinates. Images must be pre-rescaled offline and JSONL width/height
        must match the rescaled images.
        """

        def _coerce_positive_int(value: Any, *, name: str) -> int:
            if value is None:
                raise ValueError(f"{name} is required")
            if isinstance(value, bool):
                raise ValueError(f"{name} must be an integer, got bool")
            if isinstance(value, int):
                out = value
            elif isinstance(value, float):
                if not value.is_integer():
                    raise ValueError(f"{name} must be an integer, got {value!r}")
                out = int(value)
            elif isinstance(value, str):
                s = value.strip()
                if not s:
                    raise ValueError(f"{name} must be a non-empty integer string")
                if s.isdigit():
                    out = int(s)
                else:
                    try:
                        f = float(s)
                    except ValueError as exc:
                        raise ValueError(f"{name} must be integer-like, got {value!r}") from exc
                    if not f.is_integer():
                        raise ValueError(f"{name} must be an integer, got {value!r}")
                    out = int(f)
            else:
                raise ValueError(
                    f"{name} must be int/float/str (integer-like), got {type(value).__name__}"
                )

            if out <= 0:
                raise ValueError(f"{name} must be positive, got {out!r}")
            return out

        image_hint = (
            record.get("image_id")
            or record.get("id")
            or record.get("image")
            or record.get("image_path")
            or record.get("path")
        )

        max_pixels_raw = getattr(self.template, "max_pixels", None)
        try:
            max_pixels = _coerce_positive_int(max_pixels_raw, name="template.max_pixels")
        except ValueError as exc:
            raise ValueError(
                "Invalid template.max_pixels for max_pixels enforcement; "
                f"dataset={self.dataset_name!r}, base_idx={base_idx}, image={image_hint!r}: {exc}"
            ) from exc

        try:
            width = _coerce_positive_int(record.get("width"), name="width")
            height = _coerce_positive_int(record.get("height"), name="height")
        except ValueError as exc:
            raise ValueError(
                "Invalid width/height for max_pixels enforcement; "
                f"dataset={self.dataset_name!r}, base_idx={base_idx}, image={image_hint!r}: {exc}"
            ) from exc

        pixels = width * height
        if pixels > max_pixels:
            raise ValueError(
                f"Image resolution {width}x{height}={pixels} exceeds template.max_pixels={max_pixels} "
                f"for dataset={self.dataset_name!r}, base_idx={base_idx}, image={image_hint!r}. "
                "Runtime image resizing is forbidden because it breaks grounding coordinates. "
                "Pre-rescale images offline (and update JSONL width/height + geometry) so that "
                "width*height <= template.max_pixels."
            )

    def _encode_record(
        self,
        *,
        record: Dict[str, Any],
        builder: JSONLinesBuilder,
        system_prompt: Optional[str],
    ) -> Dict[str, Any]:
        merged = builder.build_many([record])

        conversation_messages = copy.deepcopy(merged.get("messages", []) or [])
        if system_prompt is not None:
            conversation_messages = [
                {"role": "system", "content": system_prompt},
                *conversation_messages,
            ]

        had_system_attr = hasattr(self.template, "system")
        original_system = getattr(self.template, "system", None) if had_system_attr else None
        if system_prompt is not None:
            try:
                setattr(self.template, "system", system_prompt)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Failed to apply temporary template.system override before encoding."
                ) from exc

        try:
            encoded = self.template.encode(merged, return_length=True)
        finally:
            if system_prompt is not None:
                try:
                    if had_system_attr:
                        setattr(self.template, "system", original_system)
                    elif hasattr(self.template, "system"):
                        delattr(self.template, "system")
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        "Failed to restore template.system after encoding."
                    ) from exc

        if not isinstance(encoded, MutableMapping):
            raise TypeError(
                "Template encode output must be a mutable mapping for conversation payload "
                "and downstream metadata keys ('sample_id', 'dataset', 'base_idx')."
            )

        encoded["messages"] = conversation_messages
        for key in ("assistant_payload", "objects", "metadata"):
            if key in merged:
                encoded[key] = copy.deepcopy(merged[key])

        return encoded

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not self.base_records:
            raise IndexError("BaseCaptionDataset is empty")

        base_idx = self._index_perm[index % len(self._index_perm)]
        record = copy.deepcopy(self.base_records[base_idx])

        # Deterministic per-sample RNG:
        # Avoid order-sensitive randomness under multi-worker prefetching by deriving the seed
        # as a pure function of (epoch, dataset seed, base_idx, requested index).
        epoch_seed = self._seed_for_epoch(self._epoch)
        seed_local = (
            int(epoch_seed)
            ^ ((int(base_idx) + 1) * 0x9E3779B1)
            ^ ((int(index) + 1) * 0xC2B2AE35)
        ) & 0xFFFFFFFF
        rng_local = random.Random(int(seed_local))

        if self.preprocessor is not None:
            if hasattr(self.preprocessor, "rng"):
                self.preprocessor.rng = rng_local
            processed = self.preprocessor(record)
            if processed is None:
                raise ValueError(
                    "Preprocessor removed the record; dataset does not duplicate samples"
                )
            record = processed

        self._enforce_max_pixels(record, base_idx=base_idx)

        self._apply_object_ordering(record, rng_local)
        self._maybe_annotate_coord_tokens(record)

        mode = self.mode
        builder = self._create_builder(mode)

        system_prompt = None
        if mode == "summary" and self.system_prompt_summary:
            system_prompt = self.system_prompt_summary
        elif mode == "dense" and self.system_prompt_dense:
            system_prompt = self.system_prompt_dense

        encoded = self._encode_record(
            record=record,
            builder=builder,
            system_prompt=system_prompt,
        )

        if not isinstance(encoded, MutableMapping):
            raise TypeError(
                "Dataset encode output must be a mutable mapping to attach metadata keys "
                "('sample_id', 'dataset', 'base_idx')."
            )

        objects = record.get("objects") or []
        max_poly = 0
        for obj in objects:
            if "poly_points" in obj:
                max_poly = max(max_poly, int(obj.get("poly_points") or 0))
        info = {
            "dataset": self.dataset_name,
            "base_idx": base_idx,
            "objects": len(objects),
            "max_poly_points": max_poly,
            "width": record.get("width"),
            "height": record.get("height"),
            "mode": mode,
        }
        input_ids = encoded.get("input_ids")
        if input_ids is not None and hasattr(input_ids, "__len__"):
            info["input_ids_len"] = len(input_ids)

        sample_id = self._make_sample_id(self.dataset_name, base_idx)
        try:
            encoded["sample_id"] = sample_id
            encoded["dataset"] = self.dataset_name
            encoded["base_idx"] = base_idx
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to attach sample metadata keys ('sample_id', 'dataset', 'base_idx') "
                f"for dataset={self.dataset_name!r}, base_idx={base_idx}."
            ) from exc

        self.last_sample_debug = info
        LAST_SAMPLE_DEBUG.update(info)

        return encoded


# Backward compatibility alias
DenseCaptionDataset = BaseCaptionDataset
