from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, dataclass
from types import MappingProxyType

import pytest

from src.training.supervision.context import SupervisionContext
from src.training.supervision.plans import SupervisionObject, SupervisionPlan


def test_supervision_plan_is_semantic_only() -> None:
    plan = SupervisionPlan(
        sample_id="fixture-stage1",
        stage="stage1",
        template_id="compact_full",
        objects=(
            SupervisionObject(
                object_id="obj-1",
                description="red cup",
                bbox=(10.0, 20.0, 30.0, 40.0),
                provenance="coco_gt",
            ),
        ),
        provenance="prepared_jsonl",
        metadata={"source_index": 7},
    )

    assert plan.sample_id == "fixture-stage1"
    assert plan.stage == "stage1"
    assert plan.template_id == "compact_full"
    assert plan.channel == "primary"
    assert plan.objects[0].description == "red cup"
    assert plan.objects[0].bbox == (10.0, 20.0, 30.0, 40.0)
    assert plan.metadata["source_index"] == 7

    for forbidden_field in (
        "rendered_assistant_text",
        "input_ids",
        "model_inputs",
        "raw_config",
        "tokenizer",
        "tensors",
    ):
        assert not hasattr(plan, forbidden_field)


def test_stage2_plan_records_per_example_channel_ownership_and_provenance() -> None:
    rollout_correction_plan = SupervisionPlan(
        sample_id="fixture-stage2",
        stage="stage2",
        template_id="compact_full",
        channel="rollout_correction",
        objects=(
            SupervisionObject(
                object_id="pred-4",
                description="unmatched dog candidate",
                bbox=(100.0, 120.0, 160.0, 190.0),
                provenance="rollout_unmatched",
                metadata={"iou": 0.21, "pseudo_positive": True},
            ),
        ),
        provenance="rollout_matching",
        context_id="ctx-stage2-smoke",
        metadata={"assignment": "greedy_iou"},
    )

    assert rollout_correction_plan.sample_id == "fixture-stage2"
    assert rollout_correction_plan.stage == "stage2"
    assert rollout_correction_plan.channel == "rollout_correction"
    assert rollout_correction_plan.provenance == "rollout_matching"
    assert rollout_correction_plan.context_id == "ctx-stage2-smoke"
    assert rollout_correction_plan.objects[0].object_id == "pred-4"
    assert rollout_correction_plan.objects[0].provenance == "rollout_unmatched"
    assert rollout_correction_plan.objects[0].metadata["pseudo_positive"] is True

    assert not hasattr(rollout_correction_plan, "rendered_assistant_text")
    assert not hasattr(rollout_correction_plan, "input_ids")
    assert not hasattr(rollout_correction_plan, "model_inputs")
    assert not hasattr(rollout_correction_plan, "raw_config")
    assert not hasattr(rollout_correction_plan, "tokenizer")
    assert not hasattr(rollout_correction_plan, "tensors")


def test_plan_and_context_are_frozen() -> None:
    context = SupervisionContext(
        context_id="ctx-stage1-smoke",
        dataset_id="coco1024",
        split="train",
        template_id="compact_full",
        stage="stage1",
        experiment_id="exp-42",
    )
    plan = SupervisionPlan(
        sample_id="fixture-stage1",
        stage="stage1",
        template_id="compact_full",
        objects=(),
    )

    with pytest.raises(FrozenInstanceError):
        context.split = "val"

    with pytest.raises(FrozenInstanceError):
        plan.sample_id = "other"

    with pytest.raises(TypeError):
        plan.metadata["new"] = "value"

    assert isinstance(plan.metadata, MappingProxyType)
    assert not hasattr(context, "__dict__")
    assert not hasattr(plan, "__dict__")


def test_context_and_plan_split_execution_context_from_per_example_plan() -> None:
    context = SupervisionContext(
        context_id="ctx-stage2-smoke",
        dataset_id="coco1024_lvis_proxy",
        split="train",
        template_id="compact_full",
        stage="stage2",
        channel="rollout_correction",
        experiment_id="exp-stage2",
        metadata={"surface": "rollout_correction"},
    )
    plan = SupervisionPlan(
        sample_id="sample-9",
        stage="stage2",
        template_id="compact_full",
        channel="rollout_correction",
        context_id=context.context_id,
        objects=(
            SupervisionObject(
                object_id="gt-9",
                description="person",
                bbox=(1.0, 2.0, 3.0, 4.0),
                provenance="gt_matched",
            ),
        ),
        provenance="rollout_correction_ground_truth",
    )

    assert context.dataset_id == "coco1024_lvis_proxy"
    assert context.split == "train"
    assert context.template_id == "compact_full"
    assert context.stage == "stage2"
    assert context.channel == "rollout_correction"
    assert context.experiment_id == "exp-stage2"

    assert plan.sample_id == "sample-9"
    assert plan.context_id == context.context_id
    assert plan.objects[0].description == "person"
    assert not hasattr(plan, "dataset_id")
    assert not hasattr(plan, "experiment_id")
    assert not hasattr(plan, "raw_config")
    assert not hasattr(context, "raw_config")


def test_supervision_contract_rejects_invalid_stage_and_channel() -> None:
    with pytest.raises(ValueError, match="unsupported supervision stage"):
        SupervisionContext(
            context_id="ctx-bad-stage",
            dataset_id="coco1024",
            split="train",
            template_id="compact_full",
            stage="stage3",
        )

    with pytest.raises(ValueError, match="unsupported supervision channel"):
        SupervisionContext(
            context_id="ctx-bad-channel",
            dataset_id="coco1024",
            split="train",
            template_id="compact_full",
            stage="stage1",
            channel="rollout",
        )

    with pytest.raises(ValueError, match="unsupported supervision stage"):
        SupervisionPlan(
            sample_id="sample-bad-stage",
            stage="stage3",
            template_id="compact_full",
        )

    with pytest.raises(ValueError, match="unsupported supervision channel"):
        SupervisionPlan(
            sample_id="sample-bad-channel",
            stage="stage2",
            template_id="compact_full",
            channel="rollout",
        )

    with pytest.raises(TypeError, match="supervision stage"):
        SupervisionPlan(
            sample_id="sample-raw-stage",
            stage={"raw_config": {"x": 1}},
            template_id="compact_full",
        )

    with pytest.raises(TypeError, match="supervision stage"):
        SupervisionPlan(
            sample_id="sample-raw-stage-string",
            stage=RawString("stage1"),
            template_id="compact_full",
        )

    with pytest.raises(TypeError, match="supervision channel"):
        SupervisionContext(
            context_id="ctx-raw-channel",
            dataset_id="coco1024",
            split="train",
            template_id="compact_full",
            stage="stage1",
            channel={"raw_config": {"x": 1}},
        )

    with pytest.raises(TypeError, match="supervision channel"):
        SupervisionContext(
            context_id="ctx-raw-channel-string",
            dataset_id="coco1024",
            split="train",
            template_id="compact_full",
            stage="stage1",
            channel=RawString("primary"),
        )

    with pytest.raises(ValueError, match="stage1 supervision channel"):
        SupervisionContext(
            context_id="ctx-stage1-rollout-correction",
            dataset_id="coco1024",
            split="train",
            template_id="compact_full",
            stage="stage1",
            channel="rollout_correction",
        )

    with pytest.raises(ValueError, match="stage1 supervision channel"):
        SupervisionPlan(
            sample_id="sample-stage1-rollout-correction",
            stage="stage1",
            template_id="compact_full",
            channel="rollout_correction",
        )


class RawString(str):
    """String subclass carrying raw payload attributes for guard tests."""

    input_ids = (1, 2, 3)


@pytest.mark.parametrize(
    "owner_factory",
    (
        lambda value: SupervisionContext(
            context_id=value,
            dataset_id="coco1024",
            split="train",
            template_id="compact_full",
            stage="stage1",
        ),
        lambda value: SupervisionObject(
            object_id="obj-raw-string",
            description=value,
        ),
        lambda value: SupervisionPlan(
            sample_id=value,
            stage="stage1",
            template_id="compact_full",
        ),
    ),
)
def test_supervision_contract_rejects_string_subclasses(owner_factory) -> None:
    with pytest.raises(TypeError, match="semantic string"):
        owner_factory(RawString("raw"))


@pytest.mark.parametrize(
    "field_name,payload",
    (
        ("context_id", {"raw_config": {"x": 1}}),
        ("dataset_id", ["coco1024"]),
        ("split", {"input_ids": [1, 2]}),
        ("template_id", {"raw_config": {"x": 1}}),
        ("experiment_id", {"model_inputs": {"input_ids": [1]}}),
    ),
)
def test_supervision_context_rejects_raw_payloads_in_string_fields(
    field_name: str,
    payload: object,
) -> None:
    kwargs = {
        "context_id": "ctx",
        "dataset_id": "coco1024",
        "split": "train",
        "template_id": "compact_full",
        "stage": "stage1",
        "experiment_id": "exp",
    }
    kwargs[field_name] = payload

    with pytest.raises(TypeError, match=field_name):
        SupervisionContext(**kwargs)


@pytest.mark.parametrize(
    "field_name,payload",
    (
        ("object_id", {"input_ids": [1, 2]}),
        ("description", {"raw_config": {"x": 1}}),
        ("provenance", ["gt"]),
    ),
)
def test_supervision_object_rejects_raw_payloads_in_string_fields(
    field_name: str,
    payload: object,
) -> None:
    kwargs = {
        "object_id": "obj",
        "description": "cup",
        "provenance": "gt",
    }
    kwargs[field_name] = payload

    with pytest.raises(TypeError, match=field_name):
        SupervisionObject(**kwargs)


@pytest.mark.parametrize(
    "field_name,payload",
    (
        ("sample_id", {"raw_config": {"x": 1}}),
        ("template_id", {"tokenizer": "bad"}),
        ("provenance", ["prepared_jsonl"]),
        ("context_id", {"model_inputs": {"input_ids": [1]}}),
    ),
)
def test_supervision_plan_rejects_raw_payloads_in_string_fields(
    field_name: str,
    payload: object,
) -> None:
    kwargs = {
        "sample_id": "sample",
        "stage": "stage1",
        "template_id": "compact_full",
        "provenance": "prepared_jsonl",
        "context_id": "ctx",
    }
    kwargs[field_name] = payload

    with pytest.raises(TypeError, match=field_name):
        SupervisionPlan(**kwargs)


def test_supervision_plan_rejects_non_semantic_object_payloads() -> None:
    with pytest.raises(TypeError, match="SupervisionObject"):
        SupervisionPlan(
            sample_id="sample-raw-payload",
            stage="stage1",
            template_id="compact_full",
            objects=(
                {
                    "description": "raw dict payload",
                    "input_ids": [1, 2],
                    "raw_config": {"x": 1},
                },
            ),
        )


def test_supervision_plan_rejects_subclass_object_payloads() -> None:
    @dataclass(frozen=True, slots=True)
    class RawObject(SupervisionObject):
        input_ids: tuple[int, ...] = (1, 2)

    with pytest.raises(TypeError, match="SupervisionObject"):
        SupervisionPlan(
            sample_id="sample-subclass-payload",
            stage="stage1",
            template_id="compact_full",
            objects=(
                RawObject(
                    object_id="raw-object",
                    description="subclass payload",
                ),
            ),
        )


def test_supervision_object_validates_bbox_shape_and_freezes_copy() -> None:
    mutable_bbox = [1, 2, 3, 4]
    supervision_object = SupervisionObject(
        object_id="obj-bbox",
        description="box",
        bbox=mutable_bbox,
    )

    mutable_bbox[0] = 999

    assert supervision_object.bbox == (1.0, 2.0, 3.0, 4.0)
    assert isinstance(supervision_object.bbox, tuple)
    assert not hasattr(supervision_object, "__dict__")

    with pytest.raises(ValueError, match="exactly four"):
        SupervisionObject(
            object_id="obj-short-bbox",
            description="box",
            bbox=(1.0, 2.0, 3.0),
        )

    with pytest.raises(TypeError, match="numeric"):
        SupervisionObject(
            object_id="obj-bad-bbox",
            description="box",
            bbox="1,2,3,4",
        )


@pytest.mark.parametrize(
    "bbox,error_type",
    (
        ({0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}, TypeError),
        ((1.0, 2.0, 3.0, math.nan), ValueError),
        ((1.0, 2.0, 3.0, math.inf), ValueError),
        ((1.0, 2.0, 3.0, True), TypeError),
        (("1", "2", "3", "4"), TypeError),
    ),
)
def test_supervision_object_rejects_non_coordinate_bbox_values(
    bbox: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        SupervisionObject(
            object_id="obj-invalid-bbox",
            description="box",
            bbox=bbox,
        )


def test_supervision_metadata_rejects_non_scalar_payloads_and_copies_source() -> None:
    metadata = {"source": "fixture", "count": 1}
    plan = SupervisionPlan(
        sample_id="sample-metadata",
        stage="stage1",
        template_id="compact_full",
        metadata=metadata,
    )

    metadata["source"] = "mutated"

    assert plan.metadata["source"] == "fixture"

    with pytest.raises(ValueError, match="metadata key"):
        SupervisionPlan(
            sample_id="sample-raw-config",
            stage="stage1",
            template_id="compact_full",
            metadata={"raw_config": {"objective": "token_ce"}},
        )


@pytest.mark.parametrize(
    "owner_factory",
    (
        lambda metadata: SupervisionContext(
            context_id="ctx-metadata",
            dataset_id="coco1024",
            split="train",
            template_id="compact_full",
            stage="stage1",
            metadata=metadata,
        ),
        lambda metadata: SupervisionObject(
            object_id="obj-metadata",
            description="cup",
            metadata=metadata,
        ),
        lambda metadata: SupervisionPlan(
            sample_id="sample-metadata-owner",
            stage="stage1",
            template_id="compact_full",
            metadata=metadata,
        ),
    ),
)
def test_supervision_metadata_rejects_raw_keys_and_non_mapping_inputs(
    owner_factory,
) -> None:
    with pytest.raises(ValueError, match="metadata key"):
        owner_factory({"raw_config": "serialized"})

    with pytest.raises(ValueError, match="metadata key"):
        owner_factory({"input_ids.debug": "1,2,3"})

    with pytest.raises(ValueError, match="metadata key"):
        owner_factory({"tensor_shape": "2x4"})

    with pytest.raises(ValueError, match="metadata key"):
        owner_factory({"tokenizer_name": "qwen"})

    for key in (
        "assistant_text",
        "model",
        "model_handle",
        "model_output",
        "raw",
        "raw_payload",
        "rendered",
        "rendered-assistant",
        "rendered.debug",
        "rendered_output",
        "rendered_prompt",
        "rendered_text",
        "token",
        "token_ids",
        "tokens",
    ):
        with pytest.raises(ValueError, match="metadata key"):
            owner_factory({key: "serialized"})

    with pytest.raises(TypeError, match="metadata must be a mapping"):
        owner_factory(["not", "a", "mapping"])

    with pytest.raises(TypeError, match="metadata keys"):
        owner_factory({RawString("note"): "payload"})

    with pytest.raises(TypeError, match="metadata values"):
        owner_factory({"note": RawString("payload")})
