"""Current processor preserves the historical inputs and declares modality types."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from src.common.errors import EncodingContractError, RuntimeContractError
from src.config.inference import load_research_infer_config
from src.data import load_raw_examples
from src.inference import inputs
from src.inference.runtime import assemble_frontend
from src.qwen.native import prepare_native_inputs


FIXTURE = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack")
PROFILE = FIXTURE / "input_profile.json"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@pytest.fixture(scope="module")
def context():
    config = load_research_infer_config(PROFILE).config
    with patch("src.qwen.runtime_loading._load_model_from_options", side_effect=AssertionError("weights loaded")):
        frontend = assemble_frontend(config, generation_config_fingerprint=_digest(config.generation.model_dump(mode="json")))
    assert frontend.qwen.model is None
    return config, frontend.qwen, load_raw_examples(FIXTURE / "examples.jsonl")


def test_real_inputs_preserve_old_receipt_and_add_current_modality_types(context):
    config, components, rows = context
    expected = json.loads((FIXTURE / "expected_probe_inputs.json").read_text())["rows"]
    with patch.object(inputs, "render_example", wraps=inputs.render_example) as render, patch.object(
        inputs, "plan_qwen_image", wraps=inputs.plan_qwen_image
    ) as plan:
        prepared = inputs.plan_examples(rows, config=config, components=components, target_max_length=12000)
    assert render.call_count == plan.call_count == 2
    for entry, golden in zip(prepared, expected, strict=True):
        assert entry.target.image_encoding.pixel_values is None
        assert entry.target.image_encoding.image_grid_thw_tensor is None
        batch = prepare_native_inputs(components.processor, [entry.request], record_media_identity=True)
        target = entry.target
        assert target.input_ids[:target.supervised_token_spans[0].physical_token_start] == batch.prompt_token_ids[0]
        actual = {
            "prompt_token_ids": list(batch.prompt_token_ids[0]),
            "full_target_token_ids": list(target.input_ids),
            "supervised_spans": [span.to_artifact_dict() for span in target.supervised_token_spans],
            "ignored_spans": [span.to_artifact_dict() for span in target.ignored_token_spans],
            "realized_object_order": entry.prompt.realized_object_order,
        }
        for key, value in actual.items():
            assert _digest(value) == golden[key + "_sha256"]
        assert list(batch.image_grids[0]) == golden["grid"]
        assert batch.media_sha256[0] == golden["executed_rgb_sha256"]
        assert entry.image.image_content_sha256 == golden["image_file_sha256"]
        assert target.supervised_token_spans[-1].token_ids == (golden["terminal_eos_id"],)
        assert target.ignored_token_spans[0].physical_token_start == target.supervised_token_spans[-1].physical_token_end
        assert set(batch.inputs) == set(golden["native_tensors"]) | {"mm_token_type_ids"}
        for key, tensor_receipt in golden["native_tensors"].items():
            tensor = batch.inputs[key]
            assert isinstance(tensor, torch.Tensor)
            raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            assert {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": hashlib.sha256(raw).hexdigest()} == tensor_receipt
        modality_types = torch.zeros_like(batch.inputs["input_ids"])
        modality_types[batch.inputs["input_ids"] == components.processor.image_token_id] = 1
        modality_types[batch.inputs["input_ids"] == components.processor.video_token_id] = 2
        assert batch.inputs["mm_token_type_ids"].dtype == modality_types.dtype
        assert torch.equal(batch.inputs["mm_token_type_ids"], modality_types)


def test_generation_only_is_lazy_and_has_no_target(context, monkeypatch):
    config, components, rows = context
    def forbidden(*args, **kwargs):
        raise AssertionError("generation planning encoded a target or materialized pixels")
    monkeypatch.setattr(inputs, "encode_rendered_example", forbidden)
    monkeypatch.setattr(type(components.processor.image_processor), "__call__", forbidden)
    prepared = inputs.plan_examples(rows, config=config, components=components, row_indices=[9, 3])
    assert [entry.prompt.row_index for entry in prepared] == [9, 3]
    assert all(entry.target is None for entry in prepared)


def test_prompt_and_order_changes_affect_their_own_token_sequences(context):
    config, components, rows = context
    baseline = inputs.plan_examples(rows[:1], config=config, components=components, target_max_length=12000)[0]
    changed_prompt = config.model_copy(update={"template": config.template.model_copy(update={
        "prompt": config.template.prompt.model_copy(update={"user": config.template.prompt.user + " Inspect carefully."}),
    })})
    prompt = inputs.plan_examples(rows[:1], config=changed_prompt, components=components, target_max_length=12000)[0]
    assert prompt.request.expected_token_ids != baseline.request.expected_token_ids
    changed_order = config.model_copy(update={"template": config.template.model_copy(update={
        "object_ordering": "random", "object_order_seed": 1,
    })})
    ordered = inputs.plan_examples(rows[:1], config=changed_order, components=components, target_max_length=12000)[0]
    assert ordered.prompt.realized_object_order != baseline.prompt.realized_object_order
    assert ordered.request.expected_token_ids == baseline.request.expected_token_ids
    assert ordered.target.input_ids != baseline.target.input_ids


def test_tampered_target_prefix_and_wrong_media_fail_at_real_boundaries(context, monkeypatch):
    config, components, rows = context
    encode = inputs.encode_rendered_example
    def corrupt(*args, **kwargs):
        value = encode(*args, **kwargs)
        return replace(value, input_ids=(value.input_ids[0] + 1, *value.input_ids[1:]))
    with monkeypatch.context() as mutation:
        mutation.setattr(inputs, "encode_rendered_example", corrupt)
        with pytest.raises(EncodingContractError, match="annotated target prefix"):
            inputs.plan_examples(rows[:1], config=config, components=components, target_max_length=12000)
    entry = inputs.plan_examples(rows[:1], config=config, components=components)[0]
    with pytest.raises(RuntimeContractError):
        prepare_native_inputs(components.processor, [replace(entry.request, image=rows[1].image.path)])


def test_reused_target_image_plan_rejects_another_row(context):
    config, components, rows = context
    entry = inputs.plan_examples(rows[:1], config=config, components=components, target_max_length=12000)[0]
    rendered = inputs.render_example(rows[1], inputs.template_config(config))
    with pytest.raises(EncodingContractError, match="reused image plan"):
        inputs.encode_rendered_example(rows[1], rendered, components=components,
            processor_config=inputs.processor_config(config), global_max_length=12000,
            materialize_image_pixels=False, _image_encoding=entry.target.image_encoding)


def test_actual_profile_requests_preserve_tokens_policies_and_lazy_planning(context, monkeypatch):
    from src.inference.inputs import build_single_step_decode_requests

    config, components, rows = context
    frontend = SimpleNamespace(qwen=components)
    golden = json.loads((FIXTURE / "expected_probe_inputs.json").read_text())["rows"]
    with patch.object(type(components.processor.image_processor), "__call__", side_effect=AssertionError("eager pixels")):
        requests = build_single_step_decode_requests(config, frontend, rows)
    assert [request.request_id for request in requests] == [row.example_id for row in rows]
    assert all(request.generation_policy.max_new_tokens == 1 for request in requests)
    for request, expected in zip(requests, golden, strict=True):
        assert _digest(list(request.expected_executed_prompt_token_ids)) == expected["prompt_token_ids_sha256"]


@pytest.mark.parametrize("length", [0, -1, True, 1.5])
def test_invalid_target_length_fails_before_preparation(context, length):
    config, components, rows = context
    with pytest.raises(ValueError, match="positive integer"):
        inputs.plan_examples(rows, config=config, components=components, target_max_length=length)
