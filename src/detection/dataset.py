"""Latest-schema detection dataset with recursive CE sidecar alignment checks."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, MutableMapping, Sequence

from torch.utils.data import Dataset

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
    prepare_detection_training_example,
)
from src.detection.template import TemplateId, get_detection_template

DetectionObjectOrdering = Literal["sorted", "random_permutation"]


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
            ),
            dataset_name=dataset_name or path.stem,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_idx = int(index) % len(self.rows)
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
            rendered_assistant.text
        )
        encoded["metadata"] = {
            "source": raw.metadata.source,
            "split": raw.metadata.split,
            "image_id": raw.image_id,
            "file_name": raw.file_name,
        }
        encoded["detection_metadata"] = {
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
        encoded["sample_id"] = _make_sample_id(self.dataset_name, base_idx)
        encoded["dataset"] = self.dataset_name
        encoded["base_idx"] = base_idx
        if recursive_detection_targets is not None:
            encoded["recursive_detection_targets"] = recursive_detection_targets
        return dict(encoded)

    def _ordering_plan(self, *, base_idx: int) -> ObjectOrderingPlan:
        if self.config.object_ordering == "sorted":
            return ObjectOrderingPlan.sorted(seed_source="latest_detection_dataset")
        seed = _mix_seed(self.config.seed, self._epoch, base_idx)
        return ObjectOrderingPlan.random_permutation(
            seed=seed,
            seed_source=f"latest_detection_dataset:seed={self.config.seed}:epoch={self._epoch}:base_idx={base_idx}",
        )

    def _messages(
        self,
        images: Sequence[str],
        *,
        assistant_text: str,
    ) -> tuple[dict[str, Any], ...]:
        user_content: list[dict[str, Any]] = [
            {"type": "image", "image": self._resolve_image(image)} for image in images
        ]
        user_content.append({"type": "text", "text": self.config.user_prompt})
        messages: list[dict[str, Any]] = []
        if self.config.system_prompt is not None:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": user_content})
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            }
        )
        return tuple(messages)

    def _resolve_image(self, image: str) -> str:
        image_path = Path(str(image))
        if image_path.is_absolute():
            return str(image_path)
        return str((Path(self.config.image_root) / image_path).resolve(strict=False))

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
        encoded: Mapping[str, Any],
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


__all__ = [
    "DetectionDatasetRuntimeConfig",
    "DetectionTrainingDataset",
]
