"""Latest-schema detection dataset with recursive CE sidecar alignment checks."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, MutableMapping, Sequence

from torch.utils.data import Dataset

from src.common.detection_chat import build_detection_chat_messages
from src.common.io import load_jsonl_with_diagnostics
from src.detection.data import (
    ObjectOrderingPlan,
    parse_raw_detection_row,
    normalize_detection_row,
)
from src.detection.objective import (
    DetectionTrainingMode,
    LossNormalizationStrategy,
    RecursiveDetectionTargets,
    StateWeightingStrategy,
    build_compact_prefix_rollin_example,
    compute_eos_trust_weight,
    prepare_detection_training_example,
)
from src.detection.template import TemplateId, get_detection_template

DetectionObjectOrdering = Literal["sorted", "random_permutation"]

REGISTERED_DETECTION_SIDECAR_KEYS: tuple[str, ...] = (
    "recursive_detection_targets",
    "detection_metadata",
    "assistant_payload",
    "sample_id",
    "dataset",
    "base_idx",
)

DETECTION_DROPPED_BEFORE_MODEL_KEYS: tuple[str, ...] = (
    "messages",
    "metadata",
)

DETECTION_MODEL_INPUT_KEYS: frozenset[str] = frozenset(
    {
        "input_ids",
        "attention_mask",
        "labels",
        "position_ids",
        "text_position_ids",
        "token_type_ids",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "second_per_grid_ts",
        "cross_attention_mask",
        "cache_position",
        "past_key_values",
        "use_cache",
        "logits_to_keep",
        "cu_seq_lens_q",
        "cu_seq_lens_k",
        "max_length_q",
        "max_length_k",
    }
)

TRAINER_BATCH_EXTRA_KEYS: frozenset[str] = frozenset(
    {
        "compute_loss_func",
        "dataset_labels",
        "dataset_segments",
        "pack_num_samples",
        "token_types",
        "instability_meta_json",
        "proxy_desc_token_weights",
        "proxy_coord_token_weights",
        "sft_structural_close_token_weights",
    }
)


@dataclass(frozen=True)
class DetectionDatasetRuntimeConfig:
    image_root: str
    detection_template_id: TemplateId
    mode: DetectionTrainingMode
    object_ordering: DetectionObjectOrdering
    user_prompt: str
    system_prompt: str | None
    max_objects: int
    seed: int
    state_weighting: str
    normalization: str
    eos_trust_weight_config: Any | None = None
    type_gate_config: Any | None = None


class DetectionTrainingDataset(Dataset):
    """Map-style dataset for latest detection configs.

    The dataset derives recursive CE targets independently from the rendered
    assistant sequence, then fails fast unless those token IDs/labels exactly
    match the model-ready sample returned by the Swift template.
    """

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        swift_template: Any,
        config: DetectionDatasetRuntimeConfig,
        dataset_name: str = "detection",
    ) -> None:
        if not rows:
            raise ValueError("DetectionTrainingDataset requires at least one row")
        if config.max_objects <= 0:
            raise ValueError("max_objects must be positive")
        self.rows = tuple(copy.deepcopy(dict(row)) for row in rows)
        self.swift_template = swift_template
        self.template = swift_template
        self.tokenizer = getattr(swift_template, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError("swift_template must expose tokenizer for span alignment")
        self.config = config
        self._image_root = Path(config.image_root).expanduser().resolve(strict=False)
        self.dataset_name = str(dataset_name)
        self._epoch = 0

    @classmethod
    def from_jsonl(
        cls,
        jsonl_path: str | Path,
        *,
        swift_template: Any,
        image_root: str | Path,
        detection_template_id: TemplateId,
        mode: DetectionTrainingMode,
        object_ordering: DetectionObjectOrdering,
        user_prompt: str,
        system_prompt: str | None,
        max_objects: int,
        seed: int,
        state_weighting: str,
        normalization: str,
        eos_trust_weight_config: Any | None = None,
        type_gate_config: Any | None = None,
        sample_limit: int | None = None,
        dataset_name: str | None = None,
    ) -> "DetectionTrainingDataset":
        path = Path(jsonl_path)
        rows, _invalid_count = load_jsonl_with_diagnostics(path, strict=True)
        if sample_limit is not None:
            if sample_limit <= 0:
                raise ValueError("sample_limit must be positive when provided")
            rows = rows[: int(sample_limit)]
        return cls(
            rows,
            swift_template=swift_template,
            config=DetectionDatasetRuntimeConfig(
                image_root=str(image_root),
                detection_template_id=detection_template_id,
                mode=mode,
                object_ordering=object_ordering,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                max_objects=int(max_objects),
                seed=int(seed),
                state_weighting=str(state_weighting),
                normalization=str(normalization),
                eos_trust_weight_config=eos_trust_weight_config,
                type_gate_config=type_gate_config,
            ),
            dataset_name=dataset_name or path.stem,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _base_index(self, index: int) -> int:
        """Validated row index for map-style dataset access."""

        base_idx = int(index)
        if base_idx < 0 or base_idx >= len(self.rows):
            raise IndexError(
                f"DetectionTrainingDataset index {index!r} is out of range "
                f"for dataset of size {len(self.rows)}"
            )
        return base_idx

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def encoded_length_for_row(
        self,
        index: int,
        *,
        forced_rollin_k: int | None = None,
        epoch: int | None = None,
    ) -> int:
        """Return the exact encoded input length for one base row.

        The method follows the same message-rendering and Swift-template encode
        path as ``__getitem__`` but intentionally does not construct recursive
        CE sidecars. For ``prefix_rollin_et_rmp_ce``, ``forced_rollin_k`` is
        validated to make the K-invariance contract explicit; K changes labels
        and targets, not the full teacher-forced input sequence.
        """

        base_idx = self._base_index(index)
        raw = parse_raw_detection_row(self.rows[base_idx])
        if len(raw.objects) > self.config.max_objects:
            raise ValueError(
                f"row {base_idx} has {len(raw.objects)} objects, exceeding "
                f"data.max_objects={self.config.max_objects}"
            )

        ordering_plan = self._ordering_plan(base_idx=base_idx, epoch=epoch)
        normalized = normalize_detection_row(raw, object_ordering=ordering_plan)
        if forced_rollin_k is not None:
            if self.config.mode != "prefix_rollin_et_rmp_ce":
                raise ValueError(
                    "forced_rollin_k is only valid for prefix_rollin_et_rmp_ce"
                )
            k = int(forced_rollin_k)
            if k < 0 or k > len(normalized.objects):
                raise ValueError(
                    "forced_rollin_k must be in [0, object_count], "
                    f"got {forced_rollin_k!r} for object_count={len(normalized.objects)}"
                )

        detection_template = get_detection_template(self.config.detection_template_id)
        rendered_assistant = detection_template.render_assistant(normalized)
        messages = self._messages(raw.images, assistant_text=rendered_assistant.text)
        encoded = self._encode_messages(messages)
        length = encoded.get("length")
        if length is not None:
            return int(length)
        return len(_as_int_tuple(encoded.get("input_ids"), path="encoded.input_ids"))

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_idx = self._base_index(index)
        raw = parse_raw_detection_row(self.rows[base_idx])
        if len(raw.objects) > self.config.max_objects:
            raise ValueError(
                f"row {base_idx} has {len(raw.objects)} objects, exceeding "
                f"data.max_objects={self.config.max_objects}"
            )

        ordering_plan = self._ordering_plan(base_idx=base_idx)
        normalized = normalize_detection_row(raw, object_ordering=ordering_plan)
        detection_template = get_detection_template(self.config.detection_template_id)
        rendered_assistant = detection_template.render_assistant(normalized)
        messages = self._messages(raw.images, assistant_text=rendered_assistant.text)

        if self.config.mode == "prefix_rollin_et_rmp_ce":
            if self.config.detection_template_id != "compact_full":
                raise ValueError("prefix_rollin_et_rmp_ce requires compact_full template")
            if self.config.eos_trust_weight_config is None:
                raise ValueError(
                    "prefix_rollin_et_rmp_ce requires eos_trust_weight_config"
                )
            k_rng = random.Random(_mix_seed(self.config.seed, self._epoch, base_idx))
            k = k_rng.randint(0, len(normalized.objects))
            eos_trust_weight = compute_eos_trust_weight(
                len(normalized.objects),
                self.config.eos_trust_weight_config,
            )
            prepared = build_compact_prefix_rollin_example(
                objects=normalized.objects,
                rollin_order=normalized.objects,
                k=k,
                tokenizer=self.tokenizer,
                eos_trust_weight=eos_trust_weight,
                normalized_sample=normalized,
                type_gate_config=self.config.type_gate_config,
                messages=messages,
            )
        else:
            prepared = prepare_detection_training_example(
                normalized,
                template=detection_template,
                tokenizer=self.tokenizer,
                mode=self.config.mode,
                state_weighting=self._state_weighting_for_prepare(),
                normalization=self._normalization_for_prepare(),
                messages=messages,
            )

        encoded = self._encode_messages(messages)
        recursive_detection_targets = self._align_prepared_targets_to_encoded(
            encoded,
            prepared,
        )

        encoded["messages"] = copy.deepcopy(messages)
        encoded["assistant_payload"] = detection_template.parse_assistant(
            prepared.rendered_assistant.text
        )
        detection_metadata = {
            "dataset": self.dataset_name,
            "base_idx": base_idx,
            "template_id": prepared.template_id,
            "template_version": prepared.template_version,
            "mode": prepared.mode,
            "object_ordering": prepared.object_ordering.strategy,
            "object_ordering_seed": prepared.object_ordering.seed,
            "object_ordering_seed_source": prepared.object_ordering.seed_source,
            "realized_source_object_indices": list(
                prepared.realized_source_object_indices
            ),
            "object_count": len(normalized.objects),
        }
        if self.config.mode == "prefix_rollin_et_rmp_ce":
            detection_metadata.update(
                {
                    "rollin_k": int(prepared.rollin_state.k),
                    "rollin_emitted_object_instance_ids": list(
                        prepared.rollin_state.emitted
                    ),
                    "rollin_remaining_object_instance_ids": list(
                        prepared.rollin_state.remaining
                    ),
                    "rollin_prefix_token_count": len(
                        prepared.debug_spans["rollin_prefix"].token_positions
                    ),
                    "supervised_suffix_token_count": len(
                        prepared.debug_spans["supervised_suffix"].token_positions
                    ),
                    "semantic_eos_token_count": len(
                        prepared.debug_spans["semantic_eos"].token_positions
                    ),
                    "eos_trust_weight": float(prepared.eos_trust_weight),
                }
            )
        encoded["metadata"] = {
            "source": raw.metadata.source,
            "split": raw.metadata.split,
            "image_id": raw.image_id,
            "file_name": raw.file_name,
        }
        encoded["detection_metadata"] = detection_metadata
        encoded["sample_id"] = _make_sample_id(self.dataset_name, base_idx)
        encoded["dataset"] = self.dataset_name
        encoded["base_idx"] = base_idx
        if recursive_detection_targets is not None:
            encoded["recursive_detection_targets"] = recursive_detection_targets
        return dict(encoded)

    def _ordering_plan(
        self, *, base_idx: int, epoch: int | None = None
    ) -> ObjectOrderingPlan:
        resolved_epoch = self._epoch if epoch is None else int(epoch)
        if self.config.object_ordering == "sorted":
            return ObjectOrderingPlan.sorted(seed_source="latest_detection_dataset")
        seed = _mix_seed(self.config.seed, resolved_epoch, base_idx)
        return ObjectOrderingPlan.random_permutation(
            seed=seed,
            seed_source=f"latest_detection_dataset:seed={self.config.seed}:epoch={resolved_epoch}:base_idx={base_idx}",
        )

    def _messages(
        self,
        images: Sequence[str],
        *,
        assistant_text: str,
    ) -> tuple[dict[str, Any], ...]:
        messages = build_detection_chat_messages(
            system_prompt=self.config.system_prompt,
            user_prompt=self.config.user_prompt,
            images=[self._resolve_image(image) for image in images],
            assistant_text=assistant_text,
        )
        return tuple(messages)

    def _resolve_image(self, image: str) -> str:
        image_path = Path(str(image))
        candidate = image_path if image_path.is_absolute() else self._image_root / image_path
        try:
            resolved = candidate.expanduser().resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"image path does not exist under image_root: {candidate}"
            ) from exc
        try:
            resolved.relative_to(self._image_root)
        except ValueError as exc:
            raise ValueError(
                f"image path resolves outside image_root: {resolved} "
                f"(image_root={self._image_root})"
            ) from exc
        return str(resolved)

    def _encode_messages(
        self, messages: Sequence[Mapping[str, Any]]
    ) -> MutableMapping[str, Any]:
        encoded = self.swift_template.encode(
            {"messages": copy.deepcopy([dict(message) for message in messages])},
            return_length=True,
        )
        if not isinstance(encoded, MutableMapping):
            raise TypeError("swift_template.encode must return a mutable mapping")
        if "input_ids" not in encoded:
            raise ValueError("swift_template.encode output missing input_ids")
        if "labels" not in encoded:
            raise ValueError("swift_template.encode output missing labels")
        return encoded

    def _align_prepared_targets_to_encoded(
        self,
        encoded: MutableMapping[str, Any],
        prepared: Any,
    ) -> RecursiveDetectionTargets | None:
        encoded_input_ids = _as_int_tuple(encoded.get("input_ids"), path="encoded.input_ids")
        encoded_labels = _as_int_tuple(encoded.get("labels"), path="encoded.labels")
        if len(encoded_input_ids) != len(encoded_labels):
            raise ValueError("encoded input_ids and labels must have the same length")

        prepared_positions = tuple(
            index for index, label in enumerate(prepared.labels) if int(label) != -100
        )
        encoded_positions = tuple(
            index for index, label in enumerate(encoded_labels) if int(label) != -100
        )
        prefix_rollin_mode = getattr(prepared, "mode", None) == "prefix_rollin_et_rmp_ce"
        if prefix_rollin_mode:
            prepared_assistant_positions = [
                index
                for index, active in enumerate(prepared.assistant_mask)
                if bool(active)
            ]
            stop_span = getattr(prepared, "assistant_stop_token_span", None)
            if stop_span is not None:
                for position in stop_span.token_indices():
                    if position not in prepared_assistant_positions:
                        prepared_assistant_positions.append(int(position))
            prepared_alignment_positions = tuple(sorted(prepared_assistant_positions))
            if len(prepared_alignment_positions) != len(encoded_positions):
                raise ValueError(
                    "encoded labels do not align with prefix_rollin_et_rmp_ce "
                    "assistant payload and <|im_end|> stop span"
                )
            if prepared_alignment_positions:
                position_delta = int(encoded_positions[0]) - int(
                    prepared_alignment_positions[0]
                )
            else:
                position_delta = 0
            expected_encoded_positions = tuple(
                int(position) + position_delta
                for position in prepared_alignment_positions
            )
            if expected_encoded_positions != encoded_positions:
                raise ValueError(
                    "encoded assistant positions are not a constant shift of "
                    "prefix-rollin sidecar positions"
                )
            prepared_supervised_ids = tuple(
                int(prepared.input_ids[index]) for index in prepared_alignment_positions
            )
            encoded_supervised_ids = tuple(
                int(encoded_input_ids[index]) for index in encoded_positions
            )
            if prepared_supervised_ids != encoded_supervised_ids:
                raise ValueError(
                    "encoded assistant token ids do not match prefix-rollin sidecar "
                    "tokenization"
                )
            active_encoded_positions = {
                int(position) + position_delta for position in prepared_positions
            }
            encoded_labels = tuple(
                int(token_id) if index in active_encoded_positions else -100
                for index, token_id in enumerate(encoded_input_ids)
            )
            encoded["labels"] = list(encoded_labels)
        else:
            if len(prepared_positions) != len(encoded_positions):
                raise ValueError("encoded labels do not supervise the same target count")

            prepared_supervised_ids = tuple(
                int(prepared.input_ids[index]) for index in prepared_positions
            )
            encoded_supervised_ids = tuple(
                int(encoded_input_ids[index]) for index in encoded_positions
            )
            if prepared_supervised_ids != encoded_supervised_ids:
                raise ValueError(
                    "encoded supervised token ids do not match sidecar tokenization"
                )
            if not prepared_positions:
                return None

            position_delta = int(encoded_positions[0]) - int(prepared_positions[0])
            expected_encoded_positions = tuple(
                int(position) + position_delta for position in prepared_positions
            )
            if expected_encoded_positions != encoded_positions:
                raise ValueError(
                    "encoded supervised positions are not a constant shift of sidecar positions"
                )

        targets = prepared.recursive_detection_targets
        if targets is None:
            return None

        shifted_token_targets = tuple(
            replace(target, position=int(target.position) + position_delta)
            for target in targets.token_targets
        )
        shifted_loss_atoms = tuple(
            replace(
                atom,
                token_positions=tuple(
                    int(position) + position_delta for position in atom.token_positions
                ),
            )
            for atom in targets.loss_atoms
        )
        shifted = replace(
            targets,
            token_targets=shifted_token_targets,
            loss_atoms=shifted_loss_atoms,
            token_position_origin="DetectionTrainingDataset.encoded",
        )
        for target in shifted.token_targets:
            if encoded_input_ids[target.position] != int(target.teacher_token_id):
                raise ValueError(
                    "shifted recursive target does not match encoded input_ids"
                )
            if encoded_labels[target.position] != int(target.teacher_token_id):
                raise ValueError("shifted recursive target is not supervised by labels")
        return shifted

    def _state_weighting_for_prepare(self) -> StateWeightingStrategy:
        if self.config.mode in {"sorted_sft", "random_order_sft"}:
            return "uniform_permutation"
        return self.config.state_weighting  # type: ignore[return-value]

    def _normalization_for_prepare(self) -> LossNormalizationStrategy:
        if self.config.mode in {"sorted_sft", "random_order_sft"}:
            return "semantic_image_bucket_balanced"
        return self.config.normalization  # type: ignore[return-value]


def _as_int_tuple(value: Any, *, path: str) -> tuple[int, ...]:
    if value is None:
        raise ValueError(f"{path} is missing")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{path} must be a sequence")
    return tuple(int(item) for item in value)


def _mix_seed(seed: int, epoch: int, base_idx: int) -> int:
    value = (
        (int(seed) & 0xFFFFFFFF)
        ^ ((int(epoch) + 1) * 0x9E3779B1)
        ^ ((int(base_idx) + 1) * 0xC2B2AE35)
    )
    return int(value & 0xFFFFFFFF)


def _make_sample_id(dataset_name: str, base_idx: int) -> int:
    import zlib

    namespace = zlib.crc32(str(dataset_name).encode("utf-8")) & 0xFFFF
    return (namespace << 32) | (int(base_idx) & 0xFFFFFFFF)


def strip_non_model_detection_sidecars(
    batch: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Strip registered detection sidecars from a model-input batch.

    Latest detection samples carry sidecars for recursive CE and diagnostics.
    Those sidecars may need to survive dataset collation and Trainer column
    filtering, but they must not leak into ``model(**inputs)``.  This helper is
    the narrow model-input boundary for latest detection batches: every
    non-model detection sidecar must be registered here, and unknown leftovers
    fail fast instead of being silently forwarded.
    """

    registered = set(REGISTERED_DETECTION_SIDECAR_KEYS)
    dropped_before_model = set(DETECTION_DROPPED_BEFORE_MODEL_KEYS)
    trainer_extras = set(TRAINER_BATCH_EXTRA_KEYS)
    allowed = set(DETECTION_MODEL_INPUT_KEYS) | set(TRAINER_BATCH_EXTRA_KEYS)
    unknown = sorted(
        str(key)
        for key in batch
        if key not in registered | dropped_before_model | allowed
    )
    if unknown:
        raise ValueError(
            "Unregistered detection batch extras at model-input stripping boundary: "
            f"{unknown}. Register intentional sidecars in "
            "REGISTERED_DETECTION_SIDECAR_KEYS or add true model inputs to "
            "DETECTION_MODEL_INPUT_KEYS."
        )

    for key in REGISTERED_DETECTION_SIDECAR_KEYS:
        if key in batch:
            batch.pop(key)
    for key in DETECTION_DROPPED_BEFORE_MODEL_KEYS:
        if key in batch:
            batch.pop(key)
    for key in trainer_extras:
        if key in batch:
            batch.pop(key)
    return batch


__all__ = [
    "DetectionDatasetRuntimeConfig",
    "DetectionTrainingDataset",
    "REGISTERED_DETECTION_SIDECAR_KEYS",
    "DETECTION_DROPPED_BEFORE_MODEL_KEYS",
    "strip_non_model_detection_sidecars",
]
