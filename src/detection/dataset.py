"""Current-schema detection dataset with recursive CE sidecar alignment checks."""

from __future__ import annotations

import copy
import inspect
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, MutableMapping, Sequence, cast

from public_data.view_contracts import (
    load_view_metadata,
    resolve_view_image_root,
    resolve_view_repo_root,
)
from torch.utils.data import Dataset

from src.common.detection_chat import build_detection_chat_messages
from src.common.io import load_jsonl_with_diagnostics
from src.detection.data import (
    ObjectOrderingPlan,
    parse_raw_detection_row,
)
from src.detection.scene import (
    DetectionScene,
    detection_scene_from_raw_row,
    normalized_detection_sample_from_scene,
)
from src.detection.objective import (
    DetectionTrainingMode,
    LossNormalizationStrategy,
    RecursiveDetectionTargets,
    StateWeightingStrategy,
    build_compact_prefix_rollin_example,
    prepare_detection_training_example,
)
from src.detection.teacher_forcing.target_builder import (
    TeacherForcingBuildResult,
    TeacherForcingBuilderProfile,
    build_teacher_forcing_target,
)
from src.detection.template import TemplateId, get_detection_template
from src.detection.tokenization import (
    DetectionSupervisionView,
    TokenSpan,
    tokenize_rendered_detection_conversation,
)
from src.training.teacher_forcing.constants import TEACHER_FORCING_TARGET_IR_KEY
from src.training.teacher_forcing.ir import TeacherForcingTargetIR

DetectionObjectOrdering = Literal["sorted", "random_permutation"]

REGISTERED_DETECTION_SIDECAR_KEYS: tuple[str, ...] = (
    "recursive_detection_targets",
    TEACHER_FORCING_TARGET_IR_KEY,
    "rendered_span_sources",
    "detection_supervision_view_metadata",
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
        "output_router_logits",
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


def resolve_detection_jsonl_image_root(
    jsonl_path: str | Path,
    *,
    image_root: str | Path | None,
) -> Path:
    """Resolve the image root for a latest compact detection JSONL.

    :param jsonl_path: Training or evaluation JSONL path.
    :param image_root: Optional legacy explicit image root override.
    :returns: Absolute image-store root path.
    """

    path = Path(jsonl_path)

    meta_path = path.parent / "meta.json"
    if meta_path.exists():
        metadata = load_view_metadata(meta_path)
        metadata_repo_root = (
            None
            if Path(metadata.image_store).is_absolute()
            else resolve_view_repo_root(metadata, path.parent)
        )
        metadata_image_root = resolve_view_image_root(
            metadata,
            path.parent,
            repo_root=metadata_repo_root,
        )
        explicit_image_root = _resolve_explicit_image_root(
            image_root,
            metadata_image_root=metadata_image_root,
            metadata_image_store=metadata.image_store,
            metadata_repo_root=metadata_repo_root,
        )
        if (
            explicit_image_root is not None
            and explicit_image_root != metadata_image_root
        ):
            raise ValueError(
                "explicit image_root does not match view metadata image_store: "
                f"image_root={explicit_image_root}, "
                f"meta.json={meta_path}, "
                f"resolved_image_store={metadata_image_root}"
            )

        return metadata_image_root

    explicit_image_root = _resolve_explicit_image_root(
        image_root,
        metadata_image_root=None,
        metadata_image_store=None,
        metadata_repo_root=None,
    )
    if explicit_image_root is not None:
        return explicit_image_root

    raise ValueError(
        "DetectionTrainingDataset requires image_root or view metadata: "
        f"expected meta.json next to JSONL at {meta_path}"
    )


def _resolve_explicit_image_root(
    image_root: str | Path | None,
    *,
    metadata_image_root: Path | None,
    metadata_image_store: str | None,
    metadata_repo_root: Path | None,
) -> Path | None:
    """Resolve an explicit image root without CWD dependence when metadata exists."""

    if image_root is None:
        return None

    explicit_path = Path(image_root).expanduser()
    if explicit_path.is_absolute():
        return explicit_path.resolve(strict=False)

    if (
        metadata_image_root is None
        or metadata_image_store is None
        or metadata_repo_root is None
    ):
        return explicit_path.resolve(strict=False)

    metadata_image_store_path = Path(metadata_image_store)
    if metadata_image_store_path.is_absolute():
        return explicit_path.resolve(strict=False)

    if explicit_path == metadata_image_store_path:
        return metadata_image_root

    return (metadata_repo_root / explicit_path).resolve(strict=False)


@dataclass(frozen=True)
class DetectionDatasetRuntimeConfig:
    image_root: str | None
    detection_template_id: TemplateId
    mode: DetectionTrainingMode
    object_ordering: DetectionObjectOrdering
    user_prompt: str
    system_prompt: str | None
    seed: int
    state_weighting: str
    normalization: str
    type_gate_config: Any | None = None
    teacher_forcing_profile: str | None = None
    teacher_forcing_rollin_base_seed: int | None = None


def _encode_swift_template_no_resize(
    swift_template: Any,
    payload: Mapping[str, Any],
) -> Any:
    encode = getattr(swift_template, "encode")
    if _callable_accepts_keyword(encode, "do_resize"):
        return encode(payload, return_length=True, do_resize=False)
    return encode(payload, return_length=True)


def _callable_accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == keyword:
            return True
    return False


class DetectionTrainingDataset(Dataset):
    """Map-style dataset for detection configs.

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
        if config.image_root is None:
            raise ValueError(
                "DetectionTrainingDataset requires resolved image_root; call "
                "from_jsonl with data.image_root or a sibling view meta.json"
            )
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
        image_root: str | Path | None,
        detection_template_id: TemplateId,
        mode: DetectionTrainingMode,
        object_ordering: DetectionObjectOrdering,
        user_prompt: str,
        system_prompt: str | None,
        seed: int,
        state_weighting: str,
        normalization: str,
        type_gate_config: Any | None = None,
        teacher_forcing_profile: str | None = None,
        teacher_forcing_rollin_base_seed: int | None = None,
        sample_limit: int | None = None,
        dataset_name: str | None = None,
    ) -> "DetectionTrainingDataset":
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
            config=DetectionDatasetRuntimeConfig(
                image_root=str(resolved_image_root),
                detection_template_id=detection_template_id,
                mode=mode,
                object_ordering=object_ordering,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                seed=int(seed),
                state_weighting=str(state_weighting),
                normalization=str(normalization),
                type_gate_config=type_gate_config,
                teacher_forcing_profile=teacher_forcing_profile,
                teacher_forcing_rollin_base_seed=teacher_forcing_rollin_base_seed,
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

    def scene_for_row(
        self,
        index: int,
        *,
        epoch: int | None = None,
    ) -> DetectionScene:
        """Return the canonical semantic scene for one raw JSONL row."""

        base_idx = self._base_index(index)
        return self._scene_for_base_index(base_idx, epoch=epoch)

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
        scene = self._scene_for_base_index(base_idx, epoch=epoch)
        normalized = normalized_detection_sample_from_scene(scene)
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
        messages = self._messages(scene.images, assistant_text=rendered_assistant.text)
        encoded = self._encode_messages(messages)
        length = encoded.get("length")
        if length is not None:
            return int(length)
        return len(_as_int_tuple(encoded.get("input_ids"), path="encoded.input_ids"))

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_idx = self._base_index(index)
        scene = self._scene_for_base_index(base_idx)
        normalized = normalized_detection_sample_from_scene(scene)
        detection_template = get_detection_template(self.config.detection_template_id)
        recursive_detection_targets = None
        teacher_forcing_target_ir = None
        metadata_object_ordering = normalized.object_ordering
        if self.config.teacher_forcing_profile is not None:
            if self.config.detection_template_id != "compact_full":
                raise ValueError("teacher_forcing target IR requires compact_full template")
            build_result = build_teacher_forcing_target(
                scene,
                tokenizer=self.tokenizer,
                profile=cast(
                    TeacherForcingBuilderProfile,
                    self.config.teacher_forcing_profile,
                ),
                epoch=self._epoch,
                stable_sample_id=str(_make_sample_id(self.dataset_name, base_idx)),
                base_seed=int(self.config.teacher_forcing_rollin_base_seed or 17),
                input_prefix_token_id=self._teacher_forcing_input_prefix_token_id(),
            )
            if not build_result.ok:
                raise ValueError(
                    "teacher_forcing target IR construction failed: "
                    f"{build_result.drop_reason}"
                )
            if build_result.target_ir is None:
                raise ValueError("teacher_forcing target IR construction failed")
            selected_indices = tuple(
                int(index)
                for index in build_result.target_ir.metadata[
                    "selected_normalized_object_indices"
                ]
            )
            rendered_scene = replace(
                scene,
                objects=tuple(scene.objects[index] for index in selected_indices),
                object_ordering=scene.object_ordering.with_realized(
                    tuple(
                        scene.objects[index].source_object_index
                        for index in selected_indices
                    )
                ),
            )
            rendered_assistant = detection_template.render_assistant(rendered_scene)
            metadata_object_ordering = rendered_scene.object_ordering
            if rendered_assistant.text != build_result.rendered_text:
                raise ValueError(
                    "teacher_forcing rendered assistant does not match target IR roll-in"
                )
            messages = self._messages(scene.images, assistant_text=build_result.rendered_text)
            supervision_view = tokenize_rendered_detection_conversation(
                rendered_assistant,
                tokenizer=self.tokenizer,
                messages=messages,
            )
            encoded = self._encode_messages(messages)
            teacher_forcing_target_ir = self._align_teacher_forcing_target_to_encoded(
                encoded,
                build_result,
                supervision_view,
            )
            encoded["detection_supervision_view_metadata"] = (
                _detection_supervision_view_metadata(supervision_view)
            )
            prepared = None
        elif self.config.mode == "prefix_rollin_et_rmp_ce":
            rendered_assistant = detection_template.render_assistant(normalized)
            messages = self._messages(scene.images, assistant_text=rendered_assistant.text)
            if self.config.detection_template_id != "compact_full":
                raise ValueError(
                    "prefix_rollin_et_rmp_ce requires compact_full template"
                )
            k_rng = random.Random(_mix_seed(self.config.seed, self._epoch, base_idx))
            k = k_rng.randint(0, len(normalized.objects))
            prepared = build_compact_prefix_rollin_example(
                objects=normalized.objects,
                rollin_order=normalized.objects,
                k=k,
                tokenizer=self.tokenizer,
                normalized_sample=normalized,
                type_gate_config=self.config.type_gate_config,
                messages=messages,
            )
            encoded = self._encode_messages(messages)
            recursive_detection_targets = self._align_prepared_targets_to_encoded(
                encoded,
                prepared,
            )
        else:
            rendered_assistant = detection_template.render_assistant(normalized)
            messages = self._messages(scene.images, assistant_text=rendered_assistant.text)
            prepared = prepare_detection_training_example(
                normalized,
                template=detection_template,
                tokenizer=self.tokenizer,
                mode=self.config.mode,
                state_weighting=self._state_weighting_for_prepare(),
                normalization=self._normalization_for_prepare(),
                type_gate_config=self.config.type_gate_config,
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
        detection_metadata = {
            "dataset": self.dataset_name,
            "base_idx": base_idx,
            "template_id": rendered_assistant.template_id,
            "template_version": rendered_assistant.template_version,
            "mode": self.config.mode,
            "object_ordering": metadata_object_ordering.strategy,
            "object_ordering_seed": metadata_object_ordering.seed,
            "object_ordering_seed_source": metadata_object_ordering.seed_source,
            "realized_source_object_indices": list(
                metadata_object_ordering.realized_source_object_indices
            ),
            "object_count": len(normalized.objects),
        }
        if prepared is not None and self.config.mode == "prefix_rollin_et_rmp_ce":
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
                }
            )
        encoded["metadata"] = {
            "source": scene.metadata.source,
            "split": scene.metadata.split,
            "image_id": scene.image_id,
            "file_name": scene.file_name,
        }
        encoded["detection_metadata"] = detection_metadata
        encoded["rendered_span_sources"] = _rendered_span_sources(
            rendered_assistant.render_span_events
        )
        encoded["sample_id"] = _make_sample_id(self.dataset_name, base_idx)
        encoded["dataset"] = self.dataset_name
        encoded["base_idx"] = base_idx
        if recursive_detection_targets is not None:
            encoded["recursive_detection_targets"] = recursive_detection_targets
        if teacher_forcing_target_ir is not None:
            encoded[TEACHER_FORCING_TARGET_IR_KEY] = teacher_forcing_target_ir
        return dict(encoded)

    def _scene_for_base_index(
        self,
        base_idx: int,
        *,
        epoch: int | None = None,
    ) -> DetectionScene:
        raw = parse_raw_detection_row(self.rows[base_idx])
        ordering_plan = self._ordering_plan(base_idx=base_idx, epoch=epoch)
        return detection_scene_from_raw_row(
            raw,
            object_ordering=ordering_plan,
            image_reference=self._resolve_scene_image_reference(raw.images),
        )

    def _resolve_scene_image_reference(self, images: Sequence[str]) -> str:
        if len(images) != 1:
            raise ValueError(
                "DetectionScene requires exactly one image reference; "
                f"got {len(images)}"
            )
        return self._resolve_image(images[0])

    def _ordering_plan(
        self, *, base_idx: int, epoch: int | None = None
    ) -> ObjectOrderingPlan:
        resolved_epoch = self._epoch if epoch is None else int(epoch)
        if self.config.object_ordering == "sorted":
            return ObjectOrderingPlan.sorted(seed_source="detection_training_dataset")
        seed = _mix_seed(self.config.seed, resolved_epoch, base_idx)
        return ObjectOrderingPlan.random_permutation(
            seed=seed,
            seed_source=f"detection_training_dataset:seed={self.config.seed}:epoch={resolved_epoch}:base_idx={base_idx}",
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
        candidate = (
            image_path if image_path.is_absolute() else self._image_root / image_path
        )
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
        encoded = _encode_swift_template_no_resize(
            self.swift_template,
            {"messages": copy.deepcopy([dict(message) for message in messages])},
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
        encoded_input_ids = _as_int_tuple(
            encoded.get("input_ids"), path="encoded.input_ids"
        )
        encoded_labels = _as_int_tuple(encoded.get("labels"), path="encoded.labels")
        if len(encoded_input_ids) != len(encoded_labels):
            raise ValueError("encoded input_ids and labels must have the same length")

        prepared_positions = tuple(
            index for index, label in enumerate(prepared.labels) if int(label) != -100
        )
        encoded_positions = tuple(
            index for index, label in enumerate(encoded_labels) if int(label) != -100
        )
        prefix_rollin_mode = (
            getattr(prepared, "mode", None) == "prefix_rollin_et_rmp_ce"
        )
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
                raise ValueError(
                    "encoded labels do not supervise the same target count"
                )

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

    def _align_teacher_forcing_target_to_encoded(
        self,
        encoded: MutableMapping[str, Any],
        build_result: TeacherForcingBuildResult,
        supervision_view: DetectionSupervisionView,
    ) -> TeacherForcingTargetIR:
        target_ir = build_result.target_ir
        if target_ir is None:
            raise ValueError("teacher_forcing build result missing target_ir")
        encoded_input_ids = _as_int_tuple(
            encoded.get("input_ids"), path="encoded.input_ids"
        )
        encoded_labels = _as_int_tuple(encoded.get("labels"), path="encoded.labels")
        if len(encoded_input_ids) != len(encoded_labels):
            raise ValueError("encoded input_ids and labels must have the same length")
        encoded_positions = tuple(
            index for index, label in enumerate(encoded_labels) if int(label) != -100
        )
        atom_positions = tuple(int(atom.target_position) for atom in target_ir.atoms)
        view_positions = supervision_view.supervised_label_positions
        if len(atom_positions) != len(view_positions):
            raise ValueError(
                "DetectionSupervisionView does not supervise the same target count "
                "as teacher_forcing_target_ir"
            )
        if not atom_positions:
            raise ValueError("teacher_forcing_target_ir requires at least one atom")
        view_position_delta = int(view_positions[0]) - int(atom_positions[0])
        expected_view_positions = tuple(
            int(position) + view_position_delta for position in atom_positions
        )
        if expected_view_positions != view_positions:
            raise ValueError(
                "DetectionSupervisionView supervised positions are not a constant "
                "shift of teacher_forcing_target_ir positions"
            )
        build_input_ids = tuple(int(token_id) for token_id in build_result.input_ids)
        _validate_teacher_forcing_target_ir_against_supervision_view(
            build_input_ids=build_input_ids,
            atom_positions=atom_positions,
            target_ir=target_ir,
            supervision_view=supervision_view,
        )

        encoded_position_set = set(encoded_positions)
        view_supervised_ids = tuple(
            int(supervision_view.input_ids[position]) for position in view_positions
        )
        candidate_deltas: list[int] = []
        first_view_position = int(view_positions[0])
        for encoded_position in encoded_positions:
            candidate_delta = int(encoded_position) - first_view_position
            candidate_positions = tuple(
                int(position) + candidate_delta for position in view_positions
            )
            if not all(
                0 <= int(position) < len(encoded_input_ids)
                and int(position) in encoded_position_set
                for position in candidate_positions
            ):
                continue
            candidate_ids = tuple(
                int(encoded_input_ids[position]) for position in candidate_positions
            )
            if candidate_ids == view_supervised_ids:
                candidate_deltas.append(candidate_delta)

        unique_deltas = tuple(dict.fromkeys(candidate_deltas))
        if not unique_deltas:
            raise ValueError(
                "encoded labels do not contain an aligned supervised subset for "
                "DetectionSupervisionView"
            )
        if len(unique_deltas) > 1:
            raise ValueError(
                "encoded supervised positions are ambiguous relative to "
                "DetectionSupervisionView supervised positions"
            )
        encoded_position_delta = int(unique_deltas[0])
        expected_encoded_positions = tuple(
            int(position) + encoded_position_delta for position in view_positions
        )
        expected_encoded_position_set = set(expected_encoded_positions)
        extra_encoded_positions = tuple(
            int(position)
            for position in encoded_positions
            if int(position) not in expected_encoded_position_set
        )
        if extra_encoded_positions and min(extra_encoded_positions) <= max(
            expected_encoded_positions
        ):
            raise ValueError(
                "encoded labels include interleaved supervised tokens outside "
                "DetectionSupervisionView supervised positions"
            )
        encoded_labels = tuple(
            int(token_id) if index in expected_encoded_position_set else -100
            for index, token_id in enumerate(encoded_input_ids)
        )
        encoded["labels"] = list(encoded_labels)
        shifted_atoms = []
        for atom, view_position, encoded_position in zip(
            target_ir.atoms,
            view_positions,
            expected_encoded_positions,
            strict=True,
        ):
            target_position = int(atom.target_position)
            shifted_target_position = int(encoded_position)
            shifted_logit_position = int(encoded_position) - 1
            if target_position + view_position_delta != int(view_position):
                raise ValueError(
                    "teacher_forcing target IR to DetectionSupervisionView "
                    "position mapping drifted"
                )
            if encoded_input_ids[shifted_target_position] != int(atom.selected_token_id):
                raise ValueError(
                    "Swift encoded input_ids do not match DetectionSupervisionView "
                    "teacher token at adapted position"
                )
            if encoded_labels[shifted_target_position] != int(atom.selected_token_id):
                raise ValueError(
                    "Swift encoded labels are not supervised according to "
                    "DetectionSupervisionView"
                )
            shifted_atoms.append(
                replace(
                    atom,
                    logit_position=shifted_logit_position,
                    target_position=shifted_target_position,
                )
            )
        return replace(
            target_ir,
            atoms=tuple(shifted_atoms),
            metadata={
                **dict(target_ir.metadata),
                "token_position_origin": (
                    "DetectionSupervisionView.adapter_to_swift_encoded"
                ),
                "supervision_view_token_position_origin": (
                    supervision_view.token_position_origin
                ),
                "supervision_view_position_delta": view_position_delta,
                "swift_encoded_position_delta": encoded_position_delta,
                "swift_encoded_extra_label_count": len(extra_encoded_positions),
            },
        )

    def _teacher_forcing_input_prefix_token_id(self) -> int | None:
        bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        if bos_token_id is not None:
            return int(bos_token_id)
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        unk = getattr(self.tokenizer, "unk_token_id", None)
        if callable(convert):
            token_id = convert("<|im_start|>")
            if token_id is not None and token_id != unk:
                return int(token_id)
        return None

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


def _rendered_span_sources(render_span_events: Sequence[Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for event in render_span_events:
        if getattr(event, "object_instance_id", None) is None:
            continue
        if getattr(event, "span_family", None) is None:
            continue
        sources.append(
            {
                "char_span": {
                    "start": int(event.char_span.start),
                    "end": int(event.char_span.end),
                    "label": str(event.char_span.label),
                },
                "event_kind": str(event.span_kind),
                "object_id": getattr(event, "object_id", None),
                "object_instance_id": str(event.object_instance_id),
                "supervision_key": getattr(event, "supervision_key", None),
                "span_family": getattr(event, "span_family", None),
                "field_name": getattr(event, "field_name", None),
                "source_role": getattr(event, "source_role", None),
                "relation_snapshot": _json_safe_sidecar_value(
                    getattr(event, "relation_snapshot", None)
                ),
                "coordinate_weight": getattr(event, "coordinate_weight", None),
                "regression_weight": getattr(event, "regression_weight", None),
                "hard_bbox_supervision": getattr(event, "hard_bbox_supervision", None),
            }
        )
    return sources


def _validate_teacher_forcing_target_ir_against_supervision_view(
    *,
    build_input_ids: Sequence[int],
    atom_positions: Sequence[int],
    target_ir: TeacherForcingTargetIR,
    supervision_view: DetectionSupervisionView,
) -> None:
    for atom, atom_position, view_position in zip(
        target_ir.atoms,
        atom_positions,
        supervision_view.supervised_label_positions,
        strict=True,
    ):
        selected_token_id = int(atom.selected_token_id)
        if int(build_input_ids[int(atom_position)]) != selected_token_id:
            raise ValueError(
                "teacher_forcing target IR selected_token_id does not match "
                "builder input_ids"
            )
        if int(supervision_view.input_ids[int(view_position)]) != selected_token_id:
            raise ValueError(
                "DetectionSupervisionView input_ids do not match target IR "
                "selected_token_id"
            )
        if int(supervision_view.labels[int(view_position)]) != selected_token_id:
            raise ValueError(
                "DetectionSupervisionView labels do not supervise target IR "
                "selected_token_id"
            )
        if atom.coord_role is not None and not supervision_view.coord_mask[int(view_position)]:
            raise ValueError(
                "teacher_forcing coordinate atom is not backed by "
                "DetectionSupervisionView coord_mask"
            )


def _detection_supervision_view_metadata(
    supervision_view: DetectionSupervisionView,
) -> dict[str, Any]:
    return {
        "authority": "DetectionSupervisionView",
        "adapter": "DetectionTrainingDataset.teacher_forcing_swift_adapter",
        "token_position_origin": supervision_view.token_position_origin,
        "rendered_sequence_type": type(supervision_view.rendered_assistant).__name__,
        "template_id": supervision_view.rendered_assistant.template_id,
        "template_version": supervision_view.rendered_assistant.template_version,
        "supervised_label_positions": list(supervision_view.supervised_label_positions),
        "next_token_prediction_positions": list(
            supervision_view.next_token_prediction_positions
        ),
        "assistant_token_span": _token_span_metadata(
            supervision_view.assistant_token_span
        ),
        "assistant_stop_token_span": _optional_token_span_metadata(
            supervision_view.assistant_stop_token_span
        ),
        "coord_token_positions": _mask_true_indices(supervision_view.coord_mask),
        "bbox_token_positions": _mask_true_indices(supervision_view.bbox_mask),
        "terminal_token_positions": _mask_true_indices(supervision_view.terminal_mask),
        "object_entry_token_spans": [
            _token_span_metadata(entry.entry_span)
            for entry in supervision_view.object_entries
        ],
        "coord_token_spans": [
            _token_span_metadata(span)
            for entry in supervision_view.object_entries
            for span in entry.coord_spans
        ],
    }


def _token_span_metadata(span: TokenSpan) -> dict[str, Any]:
    return {
        "start": int(span.start),
        "end": int(span.end),
        "label": span.label,
    }


def _optional_token_span_metadata(span: TokenSpan | None) -> dict[str, Any] | None:
    if span is None:
        return None
    return _token_span_metadata(span)


def _mask_true_indices(mask: Sequence[bool]) -> list[int]:
    return [index for index, value in enumerate(mask) if bool(value)]


def _json_safe_sidecar_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe_sidecar_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe_sidecar_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def strip_non_model_detection_sidecars(
    batch: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Strip registered detection sidecars from a model-input batch.

    Detection samples carry sidecars for recursive CE and diagnostics.
    Those sidecars may need to survive dataset collation and Trainer column
    filtering, but they must not leak into ``model(**inputs)``.  This helper is
    the narrow model-input boundary for detection batches: every
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
    "resolve_detection_jsonl_image_root",
    "strip_non_model_detection_sidecars",
]
