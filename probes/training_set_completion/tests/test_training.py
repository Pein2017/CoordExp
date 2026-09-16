from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from probes.training_set_completion import training as t
from src.losses.raw_axis_validity_hinge import raw_axis_validity_hinge as shared_hinge


def test_masked_ce_normalizes_only_active_targets_and_leaves_masked_input_gradient_path():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.5, 1.0, -0.5], [0.0, 0.0, 1.0]], requires_grad=True)
    targets = torch.tensor([0, 1, 2])
    loss, card = t.masked_ce_loss(logits, targets, [1, 0, 1])
    expected = -torch.log_softmax(logits, -1)[[0, 2], [0, 2]].mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert card["active_tokens"] == 2
    assert torch.equal(logits.grad[1], torch.zeros_like(logits.grad[1]))


def test_route_objective_keeps_masked_token_in_replay_but_excludes_its_ce(monkeypatch):
    captured = {}

    class Replay:
        target_ids = torch.tensor([1, 2, 3])
        inputs = {}

        @staticmethod
        def aligned_logits(logits):
            return logits[0]

    def fake_replay(_model, _inputs, *, prompt_token_ids, continuation_token_ids):
        captured["prompt"] = prompt_token_ids
        captured["continuation"] = continuation_token_ids
        return Replay()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([[[1.0, 3.0, 0.0, -1.0], [0.0, 1.0, 3.0, -1.0], [0.0, 0.0, 1.0, 3.0]]]))

        def forward(self, **_):
            return SimpleNamespace(logits=self.logits)

    monkeypatch.setattr(t, "prepare_replay", fake_replay)
    route = {"prompt_token_ids": [8, 9], "continuation_token_ids": [1, 2, 3], "ce_weights": [1, 0, 1], "trusted_boxes": []}
    model = Model()
    loss, card = t.route_objective(model, {}, route, {"weight": 0.0, "margin": 1.0, "coordinate_token_ids": [0], "coordinate_bin_values": [0]})
    loss.backward()
    assert captured == {"prompt": [8, 9], "continuation": [1, 2, 3]}
    assert card["active_tokens"] == 2
    assert torch.equal(model.logits.grad[0, 1], torch.zeros(4))


def _route(*, bins=(1, 2, 4, 5), eos=False):
    continuation = [11, 12, 13, 14, 151645] if eos else [11, 12, 13, 14]
    weights = [1, 1, 1, 1, 1] if eos else [1, 1, 1, 1]
    plan = {"image_content_sha256": "image", "executed_media_sha256": "media", "observed_image_grid_thw": [1, 2, 3]}
    return {"route_id": "route", "image_id": 1, "example_id": "one", "case": {"image_path": "/tmp/image", "image_plan": plan}, "image_identity": {"image_path": "/tmp/image", **plan}, "prompt_token_ids": [7], "continuation_token_ids": continuation, "ce_weights": weights, "trusted_boxes": [{"x1_position": 0, "y1_position": 1, "x2_position": 2, "y2_position": 3, "expected_bins": list(bins)}], "provenance": {}}


def test_route_rejects_invalid_raw_bbox_axes_and_untrusted_eos():
    with pytest.raises(ValueError, match="invalid raw axes"):
        t.validate_route(_route(bins=(5, 2, 4, 6)), eos_token_id=151645)
    with pytest.raises(ValueError, match="premature/untrusted EOS"):
        t.validate_route(_route(eos=True), eos_token_id=151645)
    endpoint = _route(eos=True)
    endpoint["trusted_complete_support_endpoint"] = True
    t.validate_route(endpoint, eos_token_id=151645)


def test_raw_axis_hinge_is_differentiable_and_does_not_canonicalize_axes():
    logits = torch.zeros(4, 4, requires_grad=True)
    box = {"x1_position": 0, "y1_position": 1, "x2_position": 2, "y2_position": 3}
    loss = t.raw_axis_validity_hinge(logits, [box], coordinate_token_ids=[0, 1, 2, 3], coordinate_bin_values=[0, 1, 2, 3], margin=1 / 999)
    assert loss.item() == pytest.approx(1 / 999)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_probe_raw_axis_hinge_is_exactly_the_shared_formula():
    logits = torch.tensor(
        [
            [3.0, 1.0, -2.0, 0.0],
            [0.0, 2.0, 1.0, -1.0],
            [-2.0, 0.0, 1.0, 3.0],
            [-1.0, 1.0, 2.0, 0.0],
        ],
        requires_grad=True,
    )
    boxes = (
        {
            "x1_position": 0,
            "y1_position": 1,
            "x2_position": 2,
            "y2_position": 3,
        },
    )
    kwargs = {
        "coordinate_token_ids": (0, 1, 2, 3),
        "coordinate_bin_values": (0, 1, 2, 3),
        "margin": 1.0 / 999.0,
    }

    probe = t.raw_axis_validity_hinge(logits, boxes, **kwargs)
    shared = shared_hinge(logits, boxes, **kwargs)

    torch.testing.assert_close(probe, shared, rtol=0.0, atol=0.0)


def test_manifest_rejects_duplicate_image_routes_and_insufficient_forward_budget():
    route = _route()
    coordinate_ids = list(range(10_000, 11_000))
    coordinate_ids[1], coordinate_ids[2], coordinate_ids[4], coordinate_ids[5] = 11, 12, 13, 14
    value = {"schema": t.SCHEMA, "status": "candidate_ready", "sources": {"reviewed_routes": {"path": "/tmp/no", "sha256": "x", "size_bytes": 1}, "producer": {"path": "/tmp/no", "sha256": "x", "size_bytes": 1}}, "acquisition_manifest": {"path": "/tmp/no", "sha256": "x", "size_bytes": 1}, "source_adapter": {"root": "/tmp/no", "fingerprint": "x"}, "model_config": {"backend": {"type": "hf"}, "model": {"dtype": "fp32", "base_model": "/tmp/base"}}, "routes": [route, {**route, "route_id": "other"}], "optimizer": dict(t.DEFAULT_OPTIMIZER), "runtime": {"updates": 2, "checkpoint_steps": [1, 2], "wall_seconds": 100, "max_model_forwards": 1, "eos_token_id": 151645}, "validity_hinge": {"weight": 0.01, "margin": 1 / 999, "coordinate_token_ids": coordinate_ids, "coordinate_bin_values": list(range(1000)), "coordinate_token_spellings": [f"<|coord_{index}|>" for index in range(1000)], "coordinate_units": "normalized_0_1_from_raw_bins_0_999"}}
    value["content_sha256"] = t.digest(value)
    with pytest.raises(ValueError, match="one coherent route per image"):
        t.validate_manifest(value, verify_sources=False)
    value["routes"] = [route]
    value["content_sha256"] = t.digest({key: val for key, val in value.items() if key != "content_sha256"})
    with pytest.raises(ValueError, match="forward budget"):
        t.validate_manifest(value, verify_sources=False)


def test_manifest_rejects_shifted_coordinate_positions_against_literal_continuation():
    route = _route()
    route["trusted_boxes"][0]["x2_position"] = 3
    route["trusted_boxes"][0]["y2_position"] = 2
    coordinate_ids = list(range(10_000, 11_000))
    coordinate_ids[1], coordinate_ids[2], coordinate_ids[4], coordinate_ids[5] = 11, 12, 13, 14
    value = {"schema": t.SCHEMA, "status": "candidate_ready", "sources": {"reviewed_routes": {"path": "/tmp/no", "sha256": "x", "size_bytes": 1}, "producer": {"path": "/tmp/no", "sha256": "x", "size_bytes": 1}}, "acquisition_manifest": {"path": "/tmp/no", "sha256": "x", "size_bytes": 1}, "source_adapter": {"root": "/tmp/no", "fingerprint": "x"}, "model_config": {"backend": {"type": "hf"}, "model": {"dtype": "fp32", "base_model": "/tmp/base"}}, "routes": [route], "optimizer": dict(t.DEFAULT_OPTIMIZER), "runtime": {"updates": 1, "checkpoint_steps": [1], "wall_seconds": 100, "max_model_forwards": 1, "eos_token_id": 151645}, "validity_hinge": {"weight": 0.01, "margin": 1 / 999, "coordinate_token_ids": coordinate_ids, "coordinate_bin_values": list(range(1000)), "coordinate_token_spellings": [f"<|coord_{index}|>" for index in range(1000)], "coordinate_units": "normalized_0_1_from_raw_bins_0_999"}}
    value["content_sha256"] = t.digest(value)
    with pytest.raises(ValueError, match="positions do not match literal coordinate"):
        t.validate_manifest(value, verify_sources=False)


def test_real_bound_tokenizer_has_coord_zero_and_no_object_ref_alias():
    table = t.coordinate_token_table("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent")
    assert table["ids"][0] == 151670
    assert table["ids"][124] == 151794
    assert table["ids"][0] != 151646



def test_execution_producer_gate_accepts_current_bytes_and_rejects_other_binding(tmp_path: Path):
    current = t.binding(Path(t.__file__))
    assert t.validate_training_execution_producer({"sources": {"producer": current}}) == current

    other = tmp_path / "other.py"
    other.write_text("# different producer\n")
    with pytest.raises(ValueError, match="execution producer differs"):
        t.validate_training_execution_producer(
            {"sources": {"producer": t.binding(other)}}
        )


def test_single_gpu_run_rejects_historical_manifest_before_cuda_or_output(tmp_path: Path):
    from probes.training_set_completion import continue_training

    output = tmp_path / "attempt"
    with pytest.raises(ValueError, match="execution producer differs"):
        t.run(continue_training.OLD_MANIFEST, output=output, device="cuda:0")
    assert not output.exists()
