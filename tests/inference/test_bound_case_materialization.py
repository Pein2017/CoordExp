"""Bound-image rebasing and strict one-step request construction."""
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.inference.bound_requests import materialize_bound_case
from src.inference.inputs import build_single_step_decode_requests


def test_rebase_uses_verified_absolute_image_without_mutating_case(tmp_path):
    image = tmp_path / "original.bin"
    image.write_bytes(b"fixed image bytes")
    case = {"image_path": str(image), "image_plan": {
        "image_content_sha256": hashlib.sha256(image.read_bytes()).hexdigest()},
        "input_record": {"images": ["old/location"], "nested": {"value": 2}}}
    original = copy.deepcopy(case)
    config = {"data": {"input_jsonl": str(tmp_path / "new/rows.jsonl")}}
    actual = materialize_bound_case(case, config)
    assert actual["input_record"]["images"] == ["../original.bin"]
    actual["input_record"]["nested"]["value"] = 3
    assert case == original
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        materialize_bound_case(case, config)
    case["image_path"] = "relative.bin"
    with pytest.raises(ValueError, match="image missing"):
        materialize_bound_case(case, config)


def test_single_step_request_keeps_planner_identity_and_budget(monkeypatch):
    from src.inference import inputs
    image = SimpleNamespace(image_path='/fixed/image', declared_width=10,
        declared_height=20, decoded_width=10, decoded_height=20,
        image_content_sha256='a' * 64, expected_image_grid_thw=(1,2,3),
        logical_transform_id='identity')
    prompt = SimpleNamespace(chat_text='chat', input_prompt_token_ids=(1,2),
        expected_executed_prompt_token_ids=(1,2,3))
    plan = SimpleNamespace(image=image, prompt=prompt,
        request=SimpleNamespace(request_id='row-id'))
    frontend = SimpleNamespace(qwen=object())
    config, examples = object(), [object()]

    def planner(rows, **kwargs):
        assert rows is examples
        assert kwargs == {'config': config, 'components': frontend.qwen}
        return [plan]

    monkeypatch.setattr(inputs, 'plan_examples', planner)
    request, = build_single_step_decode_requests(config, frontend, examples)
    assert request.request_id == 'row-id'
    assert request.expected_executed_prompt_token_ids == (1,2,3)
    assert request.image_sha256 == 'a' * 64
    assert request.generation_policy.max_new_tokens == 1
