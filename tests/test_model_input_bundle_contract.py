from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any

import pytest

from src.training.encoding.model_inputs import (
    ModelInputBundle,
    backend_key_registry,
    classify_backend_key,
)
from src.training.sidecars import (
    DatasetSidecars,
    DiagnosticSidecars,
    Stage2OwnershipSidecars,
    SupervisionSidecars,
    TrainingSidecars,
)


def test_model_input_bundle_forwards_only_registered_backend_inputs() -> None:
    payload = {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "pixel_values": object(),
        "text_position_ids": [0, 1, 2],
        "logits_to_keep": 4,
        "labels": [1, 2, 3],
        "pack_num_samples": [1],
        "compute_loss_func": object(),
        "loss_scale": [1.0],
    }

    bundle = ModelInputBundle.from_mapping(payload, runner_owns_loss=True)

    assert bundle.classification_for("input_ids") == "forwarded"
    assert bundle.classification_for("text_position_ids") == "bridge_consumed"
    assert bundle.classification_for("logits_to_keep") == "bridge_consumed"
    assert bundle.classification_for("labels") == "runner_owned_loss_stripped"
    assert bundle.classification_for("pack_num_samples") == "runner_owned_loss_stripped"
    assert bundle.classification_for("compute_loss_func") == "runner_owned_loss_stripped"
    assert bundle.classification_for("loss_scale") == "runner_owned_loss_stripped"

    assert bundle.forwarded_inputs() == {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "pixel_values": payload["pixel_values"],
    }
    assert bundle.bridge_auxiliaries() == {
        "text_position_ids": [0, 1, 2],
        "logits_to_keep": 4,
    }
    assert set(bundle.runner_loss_inputs()) == {
        "labels",
        "pack_num_samples",
        "compute_loss_func",
        "loss_scale",
    }


def test_labels_can_be_forwarded_when_runner_does_not_own_loss() -> None:
    bundle = ModelInputBundle.from_mapping(
        {"input_ids": [1], "labels": [1]},
        runner_owns_loss=False,
    )

    assert classify_backend_key("labels", runner_owns_loss=False) == "forwarded"
    assert bundle.forwarded_inputs() == {"input_ids": [1], "labels": [1]}
    assert bundle.runner_loss_inputs() == {}


@pytest.mark.parametrize("runner_owns_loss", [0, 1, "false"])
def test_runner_owns_loss_requires_plain_bool(runner_owns_loss: Any) -> None:
    with pytest.raises(TypeError, match="runner_owns_loss.*plain bool"):
        classify_backend_key("labels", runner_owns_loss=runner_owns_loss)

    with pytest.raises(TypeError, match="runner_owns_loss.*plain bool"):
        ModelInputBundle.from_mapping(
            {"input_ids": [1]},
            runner_owns_loss=runner_owns_loss,
        )

    with pytest.raises(TypeError, match="runner_owns_loss.*plain bool"):
        backend_key_registry(runner_owns_loss=runner_owns_loss)


@pytest.mark.parametrize(
    "key",
    (
        "supervision_spans",
        "assignment_result",
        "duplicate_filter_result",
        "training_sidecars",
    ),
)
def test_sidecar_only_keys_are_rejected_from_model_input_bundle(key: str) -> None:
    with pytest.raises(ValueError, match="sidecar-only"):
        ModelInputBundle.from_mapping({"input_ids": [1], key: object()})


def test_unknown_backend_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backend key"):
        ModelInputBundle.from_mapping({"input_ids": [1], "surprise_tensor": [2]})

    with pytest.raises(ValueError, match="unknown backend key"):
        classify_backend_key("surprise_tensor")


def test_model_input_bundle_copies_mapping_structure_and_returns_fresh_dicts() -> None:
    payload = {"input_ids": [1], "text_position_ids": [0]}
    bundle = ModelInputBundle.from_mapping(payload)
    payload["attention_mask"] = [1]

    assert isinstance(bundle.payload, MappingProxyType)
    assert set(bundle.payload) == {"input_ids", "text_position_ids"}

    forwarded = bundle.forwarded_inputs()
    forwarded["input_ids"] = [9]
    assert bundle.payload["input_ids"] == [1]

    bridge = bundle.bridge_auxiliaries()
    bridge.clear()
    assert bundle.payload["text_position_ids"] == [0]


def test_backend_registry_classifies_current_qwen3_vl_boundary() -> None:
    registry = backend_key_registry(runner_owns_loss=True)

    assert registry["input_ids"] == "forwarded"
    assert registry["attention_mask"] == "forwarded"
    assert registry["token_type_ids"] == "forwarded"
    assert registry["pixel_values"] == "forwarded"
    assert registry["pixel_values_videos"] == "forwarded"
    assert registry["image_grid_thw"] == "forwarded"
    assert registry["video_grid_thw"] == "forwarded"
    assert registry["second_per_grid_ts"] == "forwarded"
    assert registry["position_ids"] == "forwarded"
    assert registry["cross_attention_mask"] == "forwarded"
    assert registry["cache_position"] == "forwarded"
    assert registry["past_key_values"] == "forwarded"
    assert registry["use_cache"] == "forwarded"
    assert registry["cu_seq_lens"] == "forwarded"
    assert registry["cu_seq_lens_q"] == "forwarded"
    assert registry["cu_seq_lens_k"] == "forwarded"
    assert registry["max_length_q"] == "forwarded"
    assert registry["max_length_k"] == "forwarded"
    assert registry["output_router_logits"] == "forwarded"
    assert registry["text_position_ids"] == "bridge_consumed"
    assert registry["logits_to_keep"] == "bridge_consumed"
    assert registry["labels"] == "runner_owned_loss_stripped"
    assert registry["supervision_payload"] == "sidecar_only"
    assert registry["supervision_spans"] == "sidecar_only"
    assert registry["assignment_result"] == "sidecar_only"
    assert registry["duplicate_filter_result"] == "sidecar_only"

    with pytest.raises(TypeError):
        registry["new"] = "forwarded"


def test_training_sidecars_group_semantic_payloads_and_are_not_model_inputs() -> None:
    sidecars = TrainingSidecars(
        supervision=SupervisionSidecars(
            spans=("span-1",),
            payloads=("payload-1",),
            metadata={"source": "unit"},
        ),
        diagnostics=DiagnosticSidecars(
            rendered_assistant_text="assistant text",
            token_roles=("desc", "coord"),
        ),
        dataset=DatasetSidecars(
            sample_id="sample-1",
            dataset_id="coco1024",
            split="train",
            base_idx=7,
        ),
        stage2=Stage2OwnershipSidecars(
            assignment_result={"matched": 1},
            duplicate_filter_result={"kept": 1},
        ),
    )

    assert sidecars.supervision.spans == ("span-1",)
    assert sidecars.supervision.payloads == ("payload-1",)
    assert sidecars.diagnostics.rendered_assistant_text == "assistant text"
    assert sidecars.dataset.sample_id == "sample-1"
    assert sidecars.stage2.assignment_result == {"matched": 1}
    assert sidecars.stage2.duplicate_filter_result == {"kept": 1}

    for forbidden_field in (
        "forwarded_inputs",
        "model_forward_inputs",
        "model_inputs",
        "stripped_for_model_forward",
    ):
        assert not hasattr(sidecars, forbidden_field)

    with pytest.raises(FrozenInstanceError):
        sidecars.dataset.sample_id = "other"
    with pytest.raises(TypeError):
        sidecars.supervision.metadata["new"] = "value"
