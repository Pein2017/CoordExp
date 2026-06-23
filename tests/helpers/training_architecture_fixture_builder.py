from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.training.bridge.coordinate_mapper import PredictionCoordinateMapper
from src.training.objectives.runner import ObjectiveRunner
from src.training.objectives.types import ObjectiveSpec
from src.training.span_adapters.compact_projector import CompactFullSpanProjector
from src.training.span_adapters.stage1_compact import (
    CompactCoordinateSoftTargetSpec,
    CompactCoordinateTokenWeightSpec,
    CompactTrieTargetSpec,
    Stage1CompactSpanAdapter,
)
from src.training.stage2.assignment import GreedyIoUAssignment
from src.training.stage2.duplicate_filter import DuplicateFilter
from src.training.stage2.planners import (
    Stage2RolloutCorrectionPlanner,
    Stage2GreedyIoUShadowPlanner,
    Stage2PlanningObject,
)
from src.training.supervision.batch import SupervisionBatch
from src.training.supervision.plans import SupervisionObject, SupervisionPlan
from src.training.templates.compact_full import create_compact_full_codec
from src.training.pipeline_registry import TrainingPipelineRegistry


class SnapshotTokenizer:
    """Deterministic tokenizer stub that exposes offset mappings."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        """Return a compact deterministic chat rendering."""

        assert tokenize is False
        assert add_generation_prompt is False

        return "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )

    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool,
        add_special_tokens: bool,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        """Tokenize text with stable ids and character offsets."""

        assert return_offsets_mapping is True
        assert add_special_tokens is False

        token_pattern = re.compile(
            r"<\|object_ref_start\|>|<\|box_start\|>|<\|coord_\d+\|>|"
            r"<\|im_end\|>|"
            r"[A-Za-z0-9_]+|"
            r"\s|"
            r".",
            re.DOTALL,
        )
        matches = list(token_pattern.finditer(text))
        token_texts = [match.group(0) for match in matches]
        vocab = {
            token: index + 1
            for index, token in enumerate(dict.fromkeys(token_texts))
        }

        return {
            "input_ids": [vocab[token] for token in token_texts],
            "offset_mapping": [(match.start(), match.end()) for match in matches],
        }


@dataclass(frozen=True, slots=True)
class Stage1GoldenThread:
    """Built Stage-1 golden-thread artifacts."""

    snapshot: dict[str, Any]
    supervision: SupervisionBatch
    objectives: tuple[ObjectiveSpec, ...]
    vocab_size: int


@dataclass(frozen=True, slots=True)
class Stage2GoldenThread:
    """Built Stage-2 rollout-planning artifacts."""

    snapshot: dict[str, Any]


def load_fixture(path: Path) -> dict[str, Any]:
    """Load a small JSON fixture mapping."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("fixture root must be a JSON object")

    return payload


def build_stage1_golden_thread(source: Mapping[str, Any]) -> Stage1GoldenThread:
    """Build the Stage-1 compact-full golden thread through real owners."""

    sample_id = _require_string(source, "sample_id")
    resolved_pipeline = TrainingPipelineRegistry().resolve(
        _require_mapping(source, "pipeline_config")
    )
    sample = _build_detection_sample(source)
    encoded = create_compact_full_codec().encode_sample(
        sample,
        tokenizer=SnapshotTokenizer(),
    )
    projection = CompactFullSpanProjector().project(encoded)
    plan = _build_stage1_plan(sample_id=sample_id, source=source)

    target_config = _require_mapping(source, "stage1_targets")
    trie_config = _require_mapping(target_config, "trie")
    coordinate_config = _require_mapping(target_config, "coordinate")
    token_ids_by_text = _token_ids_by_text(encoded)
    trie_position = _select_stage1_target_position(
        projection,
        trie_config,
        default_selector="first_description_token",
    )
    coordinate_position = _select_stage1_target_position(
        projection,
        coordinate_config,
        default_selector="first_coordinate_token",
    )
    trie_target_texts = _require_string_sequence(trie_config, "target_token_texts")
    coordinate_target_texts = _require_string_sequence(
        coordinate_config,
        "target_token_texts",
    )
    trie_target_ids = _target_ids_for_texts(
        token_ids_by_text,
        trie_target_texts,
        target_name="trie",
    )
    coordinate_target_ids = _target_ids_for_texts(
        token_ids_by_text,
        coordinate_target_texts,
        target_name="coordinate",
    )
    trie_target_weights = _require_float_sequence(
        trie_config,
        "target_token_weights",
    )
    coordinate_target_weights = _require_float_sequence(
        coordinate_config,
        "target_token_weights",
    )
    _require_matching_target_weights(
        target_token_texts=trie_target_texts,
        target_token_weights=trie_target_weights,
        target_name="trie",
    )
    _require_matching_target_weights(
        target_token_texts=coordinate_target_texts,
        target_token_weights=coordinate_target_weights,
        target_name="coordinate",
    )
    trie_target_id = int(encoded.labels[trie_position])
    coordinate_target_id = int(encoded.labels[coordinate_position])
    objectives = _objective_specs_from_resolved_pipeline(resolved_pipeline)
    supervision = Stage1CompactSpanAdapter().build_batch(
        sample_id=sample_id,
        projection=projection,
        trie_targets=(
            CompactTrieTargetSpec(
                label_position=trie_position,
                token_ids=trie_target_ids,
                token_weights=trie_target_weights,
            ),
        ),
        coordinate_targets=(
            CompactCoordinateSoftTargetSpec(
                label_position=coordinate_position,
                token_weights=tuple(
                    CompactCoordinateTokenWeightSpec(
                        token_id=token_id,
                        weight=weight,
                    )
                    for token_id, weight in zip(
                        coordinate_target_ids,
                        coordinate_target_weights,
                        strict=True,
                    )
                ),
            ),
        ),
        context_id=f"{sample_id}:context",
        provenance="stage1_compact_fixture_builder",
        batch_id=f"{sample_id}:batch",
        metadata={"fixture": "compact_full_stage1"},
    )
    mapper = PredictionCoordinateMapper.from_logits(
        torch.zeros(
            (len(encoded.input_ids), max(encoded.input_ids) + 8),
            dtype=torch.float32,
        ),
        supervision=supervision,
    )

    snapshot = {
        "sample_id": sample_id,
        "pipeline": {
            "pipeline_id": resolved_pipeline.pipeline.identity.pipeline_id,
            "implementation_id": resolved_pipeline.pipeline.identity.implementation_id,
            "lifecycle": resolved_pipeline.pipeline.identity.lifecycle.value,
            "enabled_objectives": [
                entry.objective_id
                for entry in resolved_pipeline.objectives.enabled_objectives
            ],
        },
        "supervision_plan": {
            "stage": plan.stage,
            "channel": plan.channel,
            "template_id": plan.template_id,
            "object_ids": [item.object_id for item in plan.objects],
            "descriptions": [item.description for item in plan.objects],
        },
        "encoding": {
            "template_id": encoded.template_id,
            "template_version": encoded.template_version,
            "rendered_assistant_text": encoded.rendered_assistant_text,
            "object_entries": [
                {
                    "object_instance_id": entry.object_instance_id,
                    "object_index": entry.object_index,
                    "source_object_index": entry.source_object_index,
                }
                for entry in encoded.object_entries
            ],
            "coordinate_slots": [
                {
                    "object_index": slot.object_index,
                    "slot_name": slot.slot_name,
                    "token_text": _span_text(encoded, slot.token_span),
                    "label_positions": list(
                        projection.objects[slot.object_index]
                        .coordinate_slots[slot.slot_index]
                        .label_positions
                    ),
                }
                for slot in encoded.coordinate_slots
            ],
        },
        "projection": {
            "schema_positions": list(projection.schema_positions),
            "description_positions": list(projection.description_positions),
            "coordinate_positions": list(projection.coordinate_positions),
            "stop_positions": list(projection.stop_positions),
            "object_count": len(projection.objects),
        },
        "selected_targets": {
            "trie": {
                "label_position": trie_position,
                "label_token_id": trie_target_id,
                "label_token_text": _token_text_at_position(encoded, trie_position),
                "target_token_ids": list(trie_target_ids),
                "target_token_texts": list(trie_target_texts),
                "target_token_weights": list(trie_target_weights),
            },
            "coordinate": {
                "label_position": coordinate_position,
                "label_token_id": coordinate_target_id,
                "label_token_text": _token_text_at_position(
                    encoded,
                    coordinate_position,
                ),
                "target_token_ids": list(coordinate_target_ids),
                "target_token_texts": list(coordinate_target_texts),
                "target_token_weights": list(coordinate_target_weights),
            },
        },
        "spans": [
            {
                "role": span.role,
                "label_positions": list(span.label_positions),
                "distribution": _serialize_distribution(span.distribution),
                "provenance": span.provenance,
            }
            for span in supervision.spans
        ],
        "mapper": [
            {
                "label_position": row.label_position,
                "row_index": row.row_index,
            }
            for row in mapper.resolved_rows
        ],
        "diagnostics": {
            "object_count": len(plan.objects),
            "coordinate_slot_count": len(encoded.coordinate_slots),
            "span_count": len(supervision.spans),
            "label_position_count": len(projection.label_positions),
        },
    }

    return Stage1GoldenThread(
        snapshot=snapshot,
        supervision=supervision,
        objectives=objectives,
        vocab_size=max(encoded.input_ids) + 8,
    )


def build_stage2_golden_thread(source: Mapping[str, Any]) -> Stage2GoldenThread:
    """Build the Stage-2 rollout-planning golden thread through real owners."""

    sample_id = _require_string(source, "sample_id")
    resolved_pipeline = TrainingPipelineRegistry().resolve(
        _require_mapping(source, "pipeline_config")
    )
    predicted_objects = tuple(
        _build_stage2_object(item, default_provenance="rollout_accepted")
        for item in _require_sequence(source, "predicted_objects")
    )
    ground_truth_objects = tuple(
        _build_stage2_object(item, default_provenance="ground_truth")
        for item in _require_sequence(source, "ground_truth_objects")
    )
    assignment_iou = float(source.get("assignment_iou_threshold", 0.5))
    duplicate_iou = float(source.get("duplicate_iou_threshold", 0.5))

    planner = Stage2GreedyIoUShadowPlanner(
        assignment_strategy=GreedyIoUAssignment(iou_threshold=assignment_iou),
        rollout_correction_planner=Stage2RolloutCorrectionPlanner(
            duplicate_filter=DuplicateFilter(iou_threshold=duplicate_iou),
        ),
    )
    planning_result = planner.plan_with_diagnostics(
        sample_id=sample_id,
        template_id="compact_full",
        predicted_objects=predicted_objects,
        ground_truth_objects=ground_truth_objects,
        context_id=f"{sample_id}:context",
    )
    plan = planning_result.plan
    assignment_result = planning_result.assignment_result

    snapshot = {
        "sample_id": sample_id,
        "pipeline": {
            "pipeline_id": resolved_pipeline.pipeline.identity.pipeline_id,
            "implementation_id": resolved_pipeline.pipeline.identity.implementation_id,
            "lifecycle": resolved_pipeline.pipeline.identity.lifecycle.value,
        },
        "duplicate_decisions": [
            {
                "object_id": decision.object_id,
                "input_index": decision.input_index,
                "action": decision.action,
                "reason": decision.reason,
                "survivor_id": decision.survivor_id,
            }
            for decision in planning_result.duplicate_decisions
        ],
        "assignment": {
            "pairs": [
                {
                    "prediction_index": pair.prediction_index,
                    "ground_truth_index": pair.ground_truth_index,
                    "prediction_id": pair.prediction_id,
                    "ground_truth_id": pair.ground_truth_id,
                    "iou": round(pair.iou, 6),
                    "reason": pair.reason,
                }
                for pair in assignment_result.pairs
            ],
            "unmatched_predictions": [
                {
                    "index": item.index,
                    "object_id": item.object_id,
                    "reason": item.reason,
                    "best_iou": round(item.best_iou, 6),
                }
                for item in assignment_result.unmatched_predictions
            ],
            "unmatched_ground_truth": [
                {
                    "index": item.index,
                    "object_id": item.object_id,
                    "reason": item.reason,
                    "best_iou": round(item.best_iou, 6),
                }
                for item in assignment_result.unmatched_ground_truth
            ],
            "metadata": dict(assignment_result.metadata),
        },
        "plan": {
            "stage": plan.stage,
            "channel": plan.channel,
            "provenance": plan.provenance,
            "object_ids": [item.object_id for item in plan.objects],
            "descriptions": [item.description for item in plan.objects],
            "source_roles": [
                str(item.metadata["source_role"]) for item in plan.objects
            ],
            "source_indices": [
                int(item.metadata["source_index"]) for item in plan.objects
            ],
            "metadata": dict(plan.metadata),
        },
        "diagnostics": {
            "predicted_count": len(predicted_objects),
            "ground_truth_count": len(ground_truth_objects),
            "final_object_count": len(plan.objects),
            "suppressed_duplicate_count": int(
                plan.metadata["duplicate_suppressed_count"]
            ),
            "false_negative_count": int(plan.metadata["false_negative_count"]),
            "post_duplicate_prediction_count": (
                planning_result.post_duplicate_prediction_count
            ),
        },
    }

    return Stage2GoldenThread(snapshot=snapshot)


def run_tiny_fake_backward_smoke(source: Mapping[str, Any]) -> dict[str, Any]:
    """Run a tiny differentiable objective smoke using fixture-built spans."""

    thread = build_stage1_golden_thread(source)
    time_steps = max(
        label_position
        for span in thread.supervision.spans
        for label_position in span.label_positions
    ) + 2
    logits = torch.nn.Parameter(torch.zeros((time_steps, thread.vocab_size)))
    result = ObjectiveRunner().run(
        logits=logits,
        supervision=thread.supervision,
        objectives=thread.objectives,
    )

    result.loss.backward()
    row_gradients = _row_gradient_sums(
        logits.grad,
        supervision=thread.supervision,
    )

    return {
        "loss_is_finite": bool(torch.isfinite(result.loss).detach().item()),
        "loss": float(result.loss.detach().item()),
        "gradient_nonzero": bool((logits.grad.abs().sum() > 0).detach().item()),
        "objective_ids": sorted(result.objectives.keys()),
        "objective_losses": {
            objective_id: float(objective_result.loss.detach().item())
            for objective_id, objective_result in result.objectives.items()
        },
        "objective_span_counts": {
            objective_id: int(objective_result.span_count)
            for objective_id, objective_result in result.objectives.items()
        },
        "row_gradient_sums": row_gradients,
        "metric_keys": sorted(event.key for event in result.metric_events),
    }


def _build_detection_sample(source: Mapping[str, Any]) -> NormalizedDetectionSample:
    """Return a normalized detection sample from the source fixture."""

    objects = tuple(
        _build_detection_object(item, normalized_index=index)
        for index, item in enumerate(_require_sequence(source, "objects"))
    )

    return NormalizedDetectionSample(
        images=(_require_string(source, "image"),),
        objects=objects,
        width=int(source["width"]),
        height=int(source["height"]),
        image_id=int(source["image_id"]),
        file_name=_require_string(source, "image"),
        metadata=DetectionMetadata(source="training_architecture_fixture", split="unit"),
        object_ordering=ObjectOrderingPlan.sorted().with_realized(
            tuple(int(item["source_object_index"]) for item in source["objects"])
        ),
    )


def _build_detection_object(
    item: Mapping[str, Any],
    *,
    normalized_index: int,
) -> NormalizedDetectionObject:
    """Return one normalized detection object from fixture data."""

    coords = tuple(int(value) for value in _require_sequence(item, "bbox"))
    if len(coords) != 4:
        raise ValueError("fixture bbox must contain four coordinates")

    return NormalizedDetectionObject(
        normalized_object_index=normalized_index,
        source_object_index=int(item["source_object_index"]),
        object_instance_id=_require_string(item, "object_id"),
        desc=_require_string(item, "description"),
        bbox_2d=CoordinateTokenBox(
            *(f"<|coord_{coord}|>" for coord in coords),
        ),
        category_id=int(item["category_id"]),
        category_name=_require_string(item, "category_name"),
        coco_ann_id=int(item["annotation_id"]),
    )


def _build_stage1_plan(
    *,
    sample_id: str,
    source: Mapping[str, Any],
) -> SupervisionPlan:
    """Return the semantic Stage-1 plan from source fixture objects."""

    return SupervisionPlan(
        sample_id=sample_id,
        stage="stage1",
        template_id="compact_full",
        objects=tuple(
            SupervisionObject(
                object_id=_require_string(item, "object_id"),
                description=_require_string(item, "description"),
                bbox=tuple(float(value) for value in _require_sequence(item, "bbox")),
                provenance="ground_truth",
                metadata={"source_object_index": int(item["source_object_index"])},
            )
            for item in _require_sequence(source, "objects")
        ),
        channel="primary",
        provenance="stage1_compact_fixture_builder",
        context_id=f"{sample_id}:context",
        metadata={"fixture": "compact_full_stage1"},
    )


def _build_stage2_object(
    item: Mapping[str, Any],
    *,
    default_provenance: str,
) -> Stage2PlanningObject:
    """Return a Stage-2 planning object from fixture data."""

    return Stage2PlanningObject(
        object_id=_require_string(item, "object_id"),
        description=_require_string(item, "description"),
        bbox=tuple(float(value) for value in _require_sequence(item, "bbox")),
        provenance=str(item.get("provenance", default_provenance)),
        confidence=(
            None
            if item.get("confidence") is None
            else float(item.get("confidence"))
        ),
        evidence_count=item.get("evidence_count", 0),
        explorer_support=item.get("explorer_support", 0),
        crowd_exempt=item.get("crowd_exempt", False),
        metadata={"fixture_role": str(item.get("fixture_role", default_provenance))},
    )


def _span_text(encoded: Any, token_span: Any) -> str:
    """Return text covered by a token span in an encoded view."""

    rendered = str(encoded.rendered_assistant_text or "")
    if token_span.char_span is not None:
        return rendered[token_span.char_span.start : token_span.char_span.end]

    start = encoded.offset_mapping[token_span.start][0]
    end = encoded.offset_mapping[token_span.end - 1][1]
    assistant_base = encoded.offset_mapping[encoded.assistant_token_span.start][0]

    return rendered[max(0, start - assistant_base) : max(0, end - assistant_base)]


def _token_text_at_position(encoded: Any, position: int) -> str:
    """Return the rendered assistant token text at one encoded position."""

    rendered = str(encoded.rendered_assistant_text or "")
    start = encoded.offset_mapping[position][0]
    end = encoded.offset_mapping[position][1]
    assistant_base = encoded.offset_mapping[encoded.assistant_token_span.start][0]

    return rendered[max(0, start - assistant_base) : max(0, end - assistant_base)]


def _token_ids_by_text(encoded: Any) -> Mapping[str, int]:
    """Return first assistant token id for each token text."""

    token_ids: dict[str, int] = {}
    for position in range(
        encoded.assistant_token_span.start,
        encoded.assistant_token_span.end,
    ):
        token_ids.setdefault(
            _token_text_at_position(encoded, position),
            int(encoded.input_ids[position]),
        )

    return token_ids


def _target_ids_for_texts(
    token_ids_by_text: Mapping[str, int],
    target_texts: Sequence[str],
    *,
    target_name: str,
) -> tuple[int, ...]:
    """Return token ids for explicit fixture target texts."""

    token_ids: list[int] = []
    for token_text in target_texts:
        try:
            token_ids.append(int(token_ids_by_text[token_text]))
        except KeyError as exc:
            raise ValueError(
                f"{target_name} target token text is absent from encoded sample: "
                f"{token_text!r}"
            ) from exc

    return tuple(token_ids)


def _select_stage1_target_position(
    projection: Any,
    target_config: Mapping[str, Any],
    *,
    default_selector: str,
) -> int:
    """Return the configured compact-full label position for a fixture target."""

    selector = str(target_config.get("label_selector", default_selector))
    if selector == "first_description_token":
        return int(projection.description_positions[0])
    if selector == "first_coordinate_token":
        return int(projection.coordinate_positions[0])
    if selector == "first_schema_token":
        return int(projection.schema_positions[0])

    raise ValueError(f"unsupported stage1 target label selector: {selector!r}")


def _objective_specs_from_resolved_pipeline(
    resolved_pipeline: Any,
) -> tuple[ObjectiveSpec, ...]:
    """Return enabled objective specs from a resolved fixture pipeline."""

    specs: list[ObjectiveSpec] = []
    for entry in resolved_pipeline.objectives.enabled_objectives:
        if entry.objective_id == "standard_ce":
            specs.append(
                ObjectiveSpec(
                    objective_id="token_ce",
                    weight=entry.weight,
                    config=entry.config,
                )
            )
            continue
        if entry.objective_id == "research_teacher_forcing":
            terms = entry.config.get("terms", {})
            if not isinstance(terms, Mapping):
                raise TypeError(
                    "research_teacher_forcing fixture objective config.terms must be a mapping"
                )
            for term_id, term_config in terms.items():
                if not isinstance(term_config, Mapping):
                    raise TypeError(
                        f"research_teacher_forcing term {term_id!r} must be a mapping"
                    )
                enabled = term_config.get("enabled", True)
                if enabled is False:
                    continue
                if type(enabled) is not bool:
                    raise TypeError(
                        f"research_teacher_forcing term {term_id!r}.enabled must be a boolean"
                    )
                term_weight = term_config.get("weight", 1.0)
                if not isinstance(term_weight, (int, float)) or isinstance(
                    term_weight, bool
                ):
                    raise TypeError(
                        f"research_teacher_forcing term {term_id!r}.weight must be numeric"
                    )
                specs.append(
                    ObjectiveSpec(
                        objective_id=str(term_id),
                        weight=entry.weight * float(term_weight),
                        config={
                            key: value
                            for key, value in term_config.items()
                            if key not in {"enabled", "weight"}
                        },
                    )
                )
            continue
        specs.append(
            ObjectiveSpec(
                objective_id=entry.objective_id,
                weight=entry.weight,
                config=entry.config,
            )
        )

    return tuple(specs)


def _row_gradient_sums(
    grad: torch.Tensor | None,
    *,
    supervision: SupervisionBatch,
) -> dict[str, float]:
    """Return absolute gradient mass by semantic span role."""

    if grad is None:
        raise AssertionError("tiny smoke expected logits.grad to be populated")

    row_sums: dict[str, float] = {}
    for span in supervision.spans:
        role = str(span.role)
        row_sums.setdefault(role, 0.0)
        for label_position in span.label_positions:
            row_sums[role] += float(grad[int(label_position) - 1].abs().sum().item())

    return row_sums


def _serialize_distribution(distribution: Any) -> dict[str, Any]:
    """Return the compact JSON view of a target distribution."""

    payload: dict[str, Any] = {"kind": str(distribution.kind)}
    if hasattr(distribution, "token_id"):
        payload["token_id"] = int(distribution.token_id)
    if hasattr(distribution, "token_ids"):
        payload["token_ids"] = [int(token_id) for token_id in distribution.token_ids]
    if hasattr(distribution, "token_weights"):
        token_weights = distribution.token_weights
        if payload["kind"] == "coordinate_soft_token":
            payload["token_weights"] = [
                {
                    "token_id": int(token_weight.token_id),
                    "weight": float(token_weight.weight),
                }
                for token_weight in token_weights
            ]
        elif token_weights is None:
            payload["token_weights"] = None
        else:
            payload["token_weights"] = [float(weight) for weight in token_weights]
    if hasattr(distribution, "target_family"):
        payload["target_family"] = str(distribution.target_family)
    if hasattr(distribution, "loss_mode"):
        payload["loss_mode"] = str(distribution.loss_mode)

    return payload


def _require_mapping(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a fixture mapping field."""

    value = source[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")

    return value


def _require_sequence(source: Mapping[str, Any], key: str) -> Sequence[Any]:
    """Return a fixture sequence field."""

    value = source[key]
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError(f"{key} must be a sequence")

    return value


def _require_string(source: Mapping[str, Any], key: str) -> str:
    """Return a fixture string field."""

    value = source[key]
    if type(value) is not str or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")

    return value


def _require_string_sequence(source: Mapping[str, Any], key: str) -> tuple[str, ...]:
    """Return a fixture sequence of strings."""

    values = _require_sequence(source, key)
    normalized: list[str] = []
    for value in values:
        if type(value) is not str or not value:
            raise ValueError(f"{key} must contain non-empty strings")
        normalized.append(value)

    return tuple(normalized)


def _require_float_sequence(source: Mapping[str, Any], key: str) -> tuple[float, ...]:
    """Return a fixture sequence of numeric weights."""

    values = _require_sequence(source, key)
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key} must contain numeric weights")
        normalized.append(float(value))

    return tuple(normalized)


def _require_matching_target_weights(
    *,
    target_token_texts: Sequence[str],
    target_token_weights: Sequence[float],
    target_name: str,
) -> None:
    """Require one fixture weight per explicit target token."""

    if len(target_token_texts) != len(target_token_weights):
        raise ValueError(
            f"{target_name} target_token_texts and target_token_weights "
            "must have the same length"
        )
