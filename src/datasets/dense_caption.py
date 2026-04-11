from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Literal, Mapping, MutableMapping, Optional, Sequence

from torch.utils.data import Dataset
from swift.llm import MaxLengthError

from src.common.object_field_order import (
    ObjectFieldOrder,
    normalize_object_field_order,
)
from src.config.prompts import USER_PROMPT_SUMMARY
from src.config.schema import CoordTokensConfig
from src.coord_tokens.validator import annotate_coord_tokens

from .builders import JSONLinesBuilder
from .contracts import ConversationRecord, validate_conversation_record
from .encoded_sample_cache import setup_encoded_sample_cache_for_dataset
from .preprocessors import AugmentationPreprocessor, SequentialPreprocessor
from .utils import (
    find_first_unsorted_object_pair_by_topleft,
    load_jsonl_with_diagnostics,
)

# Exposed for debugging (e.g., OOM tracing)
LAST_SAMPLE_DEBUG: Dict[str, Any] = {}
logger = logging.getLogger(__name__)


def _dataset_timing_enabled() -> bool:
    raw = str(os.environ.get("COORDEXP_DATASET_TIMING", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _dataset_timing_min_ms() -> float:
    raw = str(os.environ.get("COORDEXP_DATASET_TIMING_MIN_MS", "")).strip()
    if not raw:
        return 250.0
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 250.0


def _append_timing_line(line: str) -> None:
    path = str(os.environ.get("COORDEXP_TIMING_LOG", "")).strip()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")
    except OSError:
        return


def _extract_sample_length(sample: Mapping[str, Any]) -> int | None:
    length = sample.get("length")
    if isinstance(length, int):
        return int(length)
    input_ids = sample.get("input_ids")
    try:
        return len(input_ids) if input_ids is not None else None
    except TypeError:
        return None


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
        offline_max_pixels: Optional[int] = None,
        object_ordering: Literal["sorted", "random"] = "sorted",
        object_field_order: ObjectFieldOrder = "desc_first",
        encoded_sample_cache: Optional[Mapping[str, Any]] = None,
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
        self._curriculum_state = curriculum_state
        self.offline_max_pixels = (
            None if offline_max_pixels is None else int(offline_max_pixels)
        )
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
            except (AttributeError, TypeError) as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Failed to set template.system for summary-mode dataset initialization."
                ) from exc
        else:
            if self.system_prompt_dense is None:
                self.system_prompt_dense = getattr(self.template, "system", None)
            if self.system_prompt_dense is not None:
                try:
                    setattr(self.template, "system", self.system_prompt_dense)
                except (AttributeError, TypeError) as exc:  # noqa: BLE001
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
        self._encoded_sample_cache = None
        self._encoded_sample_cache_info: Dict[str, Any] | None = None
        if encoded_sample_cache is not None:
            cache_store, cache_info = setup_encoded_sample_cache_for_dataset(
                self, encoded_sample_cache
            )
            self._encoded_sample_cache = cache_store
            self._encoded_sample_cache_info = (
                None if cache_info is None else dict(cache_info)
            )

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
        records, _invalid_count = load_jsonl_with_diagnostics(
            Path(str(jsonl_path)),
            strict=True,
            resolve_relative=True,
        )
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
            path = Path(str(jsonl_path))
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
        if plan is not None and self._encoded_sample_cache is not None:
            raise ValueError(
                "Encoded sample cache does not support hard-sample or epoch-varying "
                "sample plans. Disable training.encoded_sample_cache or set "
                "training.encoded_sample_cache.ineligible_policy=bypass for this run."
            )
        self._hard_sample_plan = dict(plan) if plan is not None else None

    def __len__(self) -> int:
        return len(self.base_records)

    def get_encoded_sample_cache_info(self) -> Optional[Dict[str, Any]]:
        if self._encoded_sample_cache_info is None:
            return None
        return copy.deepcopy(self._encoded_sample_cache_info)

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
        """Fail fast if a record would exceed the configured offline image budget.

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

        if self.offline_max_pixels is not None:
            max_pixels_raw = self.offline_max_pixels
            max_pixels_name = "custom.offline_max_pixels"
        else:
            max_pixels_raw = getattr(self.template, "max_pixels", None)
            max_pixels_name = "template.max_pixels"
        try:
            max_pixels = _coerce_positive_int(max_pixels_raw, name=max_pixels_name)
        except ValueError as exc:
            raise ValueError(
                f"Invalid {max_pixels_name} for max_pixels enforcement; "
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
                f"Image resolution {width}x{height}={pixels} exceeds {max_pixels_name}={max_pixels} "
                f"for dataset={self.dataset_name!r}, base_idx={base_idx}, image={image_hint!r}. "
                "Runtime image resizing is forbidden because it breaks grounding coordinates. "
                "Pre-rescale images offline (and update JSONL width/height + geometry) so that "
                f"width*height <= {max_pixels_name}."
            )

    def _encode_record(
        self,
        *,
        record: Dict[str, Any],
        builder: JSONLinesBuilder,
        system_prompt: Optional[str],
    ) -> Dict[str, Any]:
        rendered, conversation_messages = self._render_prepared_record(
            record=record,
            builder=builder,
            system_prompt=system_prompt,
        )
        encoded = self._encode_rendered_record(
            rendered=rendered,
            system_prompt=system_prompt,
        )
        return self._finalize_encoded_example(
            encoded=encoded,
            rendered=rendered,
            conversation_messages=conversation_messages,
        )

    def _render_prepared_record(
        self,
        *,
        record: Dict[str, Any],
        builder: JSONLinesBuilder,
        system_prompt: Optional[str],
    ) -> tuple[Dict[str, Any], list[dict[str, Any]]]:
        rendered = builder.build_many([record])
        conversation_messages = copy.deepcopy(rendered.get("messages", []) or [])
        if system_prompt is not None:
            conversation_messages = [
                {"role": "system", "content": system_prompt},
                *conversation_messages,
            ]
        return rendered, conversation_messages

    def _encode_rendered_record(
        self,
        *,
        rendered: Mapping[str, Any],
        system_prompt: Optional[str],
    ) -> MutableMapping[str, Any]:
        had_system_attr = hasattr(self.template, "system")
        original_system = getattr(self.template, "system", None) if had_system_attr else None
        if system_prompt is not None:
            try:
                setattr(self.template, "system", system_prompt)
            except (AttributeError, TypeError) as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Failed to apply temporary template.system override before encoding."
                ) from exc

        try:
            encoded = self.template.encode(dict(rendered), return_length=True)
        finally:
            if system_prompt is not None:
                try:
                    if had_system_attr:
                        setattr(self.template, "system", original_system)
                    elif hasattr(self.template, "system"):
                        delattr(self.template, "system")
                except (AttributeError, TypeError) as exc:  # noqa: BLE001
                    raise RuntimeError(
                        "Failed to restore template.system after encoding."
                    ) from exc

        if not isinstance(encoded, MutableMapping):
            raise TypeError(
                "Template encode output must be a mutable mapping for conversation payload "
                "and downstream metadata keys ('sample_id', 'dataset', 'base_idx')."
            )
        return encoded

    def _probe_full_sample_length(
        self,
        *,
        rendered: Mapping[str, Any],
        system_prompt: Optional[str],
    ) -> int | None:
        truncation_strategy = getattr(self.template, "truncation_strategy", None)
        max_length = getattr(self.template, "max_length", None)
        try:
            setattr(self.template, "truncation_strategy", "raise")
            encoded = self._encode_rendered_record(
                rendered=rendered,
                system_prompt=system_prompt,
            )
        except MaxLengthError:
            if isinstance(max_length, int) and max_length > 0:
                return int(max_length) + 1
            return None
        finally:
            try:
                setattr(self.template, "truncation_strategy", truncation_strategy)
            except (AttributeError, TypeError):
                pass
        return _extract_sample_length(encoded)

    def _finalize_encoded_example(
        self,
        *,
        encoded: MutableMapping[str, Any],
        rendered: Mapping[str, Any],
        conversation_messages: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        encoded["messages"] = [
            dict(message) for message in list(conversation_messages or [])
        ]
        for key in ("assistant_payload", "objects", "metadata"):
            if key in rendered:
                encoded[key] = copy.deepcopy(rendered[key])
        return dict(encoded)

    def _seed_for_fetch(self, *, base_idx: int, index: int) -> int:
        epoch_seed = self._seed_for_epoch(self._epoch)
        return int(
            (
                int(epoch_seed)
                ^ ((int(base_idx) + 1) * 0x9E3779B1)
                ^ ((int(index) + 1) * 0xC2B2AE35)
            )
            & 0xFFFFFFFF
        )

    def _prepare_record_for_fetch(
        self,
        *,
        base_idx: int,
        index: int,
    ) -> Dict[str, Any]:
        record = copy.deepcopy(self.base_records[int(base_idx)])
        rng_local = random.Random(self._seed_for_fetch(base_idx=int(base_idx), index=int(index)))

        if self.preprocessor is not None:
            if hasattr(self.preprocessor, "rng"):
                self.preprocessor.rng = rng_local
            processed = self.preprocessor(record)
            if processed is None:
                raise ValueError(
                    "Preprocessor removed the record; dataset does not duplicate samples"
                )
            record = processed

        self._enforce_max_pixels(record, base_idx=int(base_idx))
        self._apply_object_ordering(record, rng_local)
        self._maybe_annotate_coord_tokens(record)
        return record

    def _prepare_record_for_cache(self, *, base_idx: int) -> Dict[str, Any]:
        if self.preprocessor is not None:
            raise ValueError(
                "Encoded sample cache requires datasets without fetch-time preprocessors."
            )

        record = copy.deepcopy(self.base_records[int(base_idx)])
        self._enforce_max_pixels(record, base_idx=int(base_idx))

        rng_local = random.Random(self.seed ^ ((int(base_idx) + 1) * 0x9E3779B1))
        self._apply_object_ordering(record, rng_local)
        self._maybe_annotate_coord_tokens(record)
        return record

    def _static_packing_length(self, base_idx: int) -> int | None:
        record = self._prepare_record_for_cache(base_idx=int(base_idx))
        rendered, _ = self._render_prepared_record(
            record=record,
            builder=self._create_builder(self.mode),
            system_prompt=self._current_system_prompt(),
        )
        return self._probe_full_sample_length(
            rendered=rendered,
            system_prompt=self._current_system_prompt(),
        )

    def _static_packing_thread_safe(self) -> bool:
        return bool(
            self._encoded_sample_cache is not None
            and bool(getattr(self._encoded_sample_cache, "thread_safe", False))
        )

    def _static_packing_precompute_info(self) -> dict[str, Any]:
        return {
            "thread_safe": bool(self._static_packing_thread_safe()),
            "prepare_sidecar_eligible": bool(
                self.preprocessor is None and self.object_ordering != "random"
            ),
            "fingerprint_surfaces": {
                "dataset_name": self.dataset_name,
                "mode": self.mode,
                "object_ordering": self.object_ordering,
                "object_field_order": self.object_field_order,
                "coord_tokens_enabled": bool(self.coord_tokens.enabled),
                "offline_max_pixels": self.offline_max_pixels,
                "system_prompt": self._current_system_prompt(),
                "full_sample_packing_probe": "raise_overlength_v1",
                "epoch_sensitive": bool(self.object_ordering == "random"),
                "has_fetch_preprocessor": bool(self.preprocessor is not None),
            },
        }

    def _current_system_prompt(self) -> Optional[str]:
        if self.mode == "summary" and self.system_prompt_summary:
            return self.system_prompt_summary
        if self.mode == "dense" and self.system_prompt_dense:
            return self.system_prompt_dense
        return None

    def _attach_runtime_metadata(
        self, encoded: MutableMapping[str, Any], *, base_idx: int
    ) -> None:
        sample_id = self._make_sample_id(self.dataset_name, base_idx)
        try:
            encoded["sample_id"] = sample_id
            encoded["dataset"] = self.dataset_name
            encoded["base_idx"] = base_idx
        except (AttributeError, TypeError) as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to attach sample metadata keys ('sample_id', 'dataset', 'base_idx') "
                f"for dataset={self.dataset_name!r}, base_idx={base_idx}."
            ) from exc

    def _build_sample_debug_info(
        self,
        *,
        base_idx: int,
        record: Mapping[str, Any],
        encoded: Mapping[str, Any],
    ) -> Dict[str, Any]:
        objects = encoded.get("objects")
        if not isinstance(objects, list):
            objects = record.get("objects") or []

        max_poly = 0
        for obj in objects:
            if isinstance(obj, Mapping) and "poly_points" in obj:
                max_poly = max(max_poly, int(obj.get("poly_points") or 0))

        info = {
            "dataset": self.dataset_name,
            "base_idx": base_idx,
            "objects": len(objects),
            "max_poly_points": max_poly,
            "width": record.get("width"),
            "height": record.get("height"),
            "mode": self.mode,
        }
        input_ids = encoded.get("input_ids")
        if input_ids is not None and hasattr(input_ids, "__len__"):
            info["input_ids_len"] = len(input_ids)
        return info

    def _encode_base_record_for_cache(self, base_idx: int) -> Dict[str, Any]:
        record = self._prepare_record_for_cache(base_idx=int(base_idx))
        encoded = self._encode_record(
            record=record,
            builder=self._create_builder(self.mode),
            system_prompt=self._current_system_prompt(),
        )
        if not isinstance(encoded, MutableMapping):
            raise TypeError(
                "Encoded sample cache build requires template.encode(...) to return "
                "a mutable mapping."
            )
        encoded.pop("sample_id", None)
        encoded.pop("dataset", None)
        encoded.pop("base_idx", None)
        return dict(encoded)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not self.base_records:
            raise IndexError("BaseCaptionDataset is empty")

        base_idx = self._index_perm[index % len(self._index_perm)]
        record_for_debug: Mapping[str, Any] = self.base_records[base_idx]
        timing_enabled = _dataset_timing_enabled()
        t0 = time.perf_counter() if timing_enabled else 0.0
        t_prepare_ms = 0.0
        t_encode_ms = 0.0
        if timing_enabled:
            try:
                image_id = record_for_debug.get("image_id")
            except Exception:
                image_id = None
            _append_timing_line(
                "dataset_fetch_start "
                f"base_idx={int(base_idx)} index={int(index)} image_id={image_id}"
            )

        if self._encoded_sample_cache is not None:
            encoded = copy.deepcopy(self._encoded_sample_cache.load_sample(base_idx))
        else:
            t_prepare_start = time.perf_counter() if timing_enabled else 0.0
            record = self._prepare_record_for_fetch(
                base_idx=int(base_idx),
                index=int(index),
            )
            if timing_enabled:
                t_prepare_ms = (time.perf_counter() - t_prepare_start) * 1000.0
                t_encode_start = time.perf_counter()
            encoded = self._encode_record(
                record=record,
                builder=self._create_builder(self.mode),
                system_prompt=self._current_system_prompt(),
            )
            if timing_enabled:
                t_encode_ms = (time.perf_counter() - t_encode_start) * 1000.0
            record_for_debug = record

        if not isinstance(encoded, MutableMapping):
            raise TypeError(
                "Dataset encode output must be a mutable mapping to attach metadata keys "
                "('sample_id', 'dataset', 'base_idx')."
            )

        info = self._build_sample_debug_info(
            base_idx=base_idx,
            record=record_for_debug,
            encoded=encoded,
        )
        self._attach_runtime_metadata(encoded, base_idx=base_idx)

        self.last_sample_debug = info
        LAST_SAMPLE_DEBUG.update(info)
        if timing_enabled:
            total_ms = (time.perf_counter() - t0) * 1000.0
            if total_ms >= _dataset_timing_min_ms():
                sample_len = _extract_sample_length(encoded)
                try:
                    object_count = len(record_for_debug.get("objects") or [])
                except Exception:
                    object_count = None
                try:
                    images = list(record_for_debug.get("images") or [])
                    image_ref = images[0] if images else None
                except Exception:
                    image_ref = None
                logger.info(
                    "dataset_timing: total_ms=%.1f prepare_ms=%.1f encode_ms=%.1f "
                    "base_idx=%s index=%s image_id=%s objects=%s length=%s image=%s",
                    total_ms,
                    t_prepare_ms,
                    t_encode_ms,
                    int(base_idx),
                    int(index),
                    record_for_debug.get("image_id"),
                    object_count,
                    sample_len,
                    image_ref,
                )
                _append_timing_line(
                    "dataset_timing "
                    f"total_ms={total_ms:.1f} prepare_ms={t_prepare_ms:.1f} "
                    f"encode_ms={t_encode_ms:.1f} base_idx={int(base_idx)} "
                    f"index={int(index)} image_id={record_for_debug.get('image_id')} "
                    f"objects={object_count} length={sample_len} image={image_ref}"
                )

        return encoded


# Backward compatibility alias
DenseCaptionDataset = BaseCaptionDataset
