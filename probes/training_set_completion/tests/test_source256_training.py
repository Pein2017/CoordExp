from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from probes.training_set_completion import source256_training as source256
from probes.training_set_completion import source256_trial


EOS = 99


def _route(image_id: int, *, completion: bool = False) -> dict:
    prefix = [30, 31] if completion else []
    suffix = [40, 41, 42, 43, EOS]
    ids = prefix + suffix
    weights = [0] * len(prefix) + [1] * len(suffix)
    case = {
        "image_path": f"/images/{image_id}.jpg",
        "image_plan": {
            "image_content_sha256": "a" * 64,
            "executed_media_sha256": "b" * 64,
            "observed_image_grid_thw": [1, 2, 3],
        },
    }
    bank_owner_ids = [f"{image_id}:0", f"{image_id}:1"]
    prefix_owner_ids = [bank_owner_ids[0]] if completion else []
    suffix_owner_ids = [bank_owner_ids[1]] if completion else bank_owner_ids
    route = {
        "route_id": f"{image_id}-{'completion' if completion else 'canonical'}",
        "image_id": image_id,
        "example_id": str(image_id),
        "case": case,
        "image_identity": {
            "image_path": case["image_path"],
            **case["image_plan"],
        },
        "prompt_token_ids": [10, 11],
        "continuation_token_ids": ids,
        "ce_weights": weights,
        "labels": [token if weight else -100 for token, weight in zip(ids, weights)],
        "geometry_weights": [0] * len(ids),
        "geometry_target_bins": [-100] * len(ids),
        "trusted_boxes": [],
        "trusted_complete_support_endpoint": True,
        "provenance": {
            "route_kind": "fixed_source_prefix_completion" if completion else "canonical",
            "bank_owner_ids": bank_owner_ids,
            "prefix_owner_ids": prefix_owner_ids,
            "suffix_owner_ids": suffix_owner_ids,
            "prefix_token_ids": prefix,
            "suffix_token_ids": suffix,
            "prefix_token_ids_sha256": source256.training.digest(prefix),
            "suffix_token_ids_sha256": source256.training.digest(suffix),
            "source_greedy_generated_token_ids_sha256": "c" * 64,
            "mask_semantics": "prefix_masked_suffix_supervised",
            "geometry_semantics": "supervised_rows_only",
        },
    }
    return route


def _records(eligible: set[int]) -> list[dict]:
    return [
        {
            "image_id": image_id,
            "example_id": str(image_id),
            "canonical_route": _route(image_id),
            "completion_route": _route(image_id, completion=True)
            if image_id in eligible
            else None,
            "eligibility": {
                "fully_eligible": image_id in eligible,
                "fallback_reason": None if image_id in eligible else "no_trusted_prefix",
            },
        }
        for image_id in range(source256.IMAGE_COUNT)
    ]


def _schedule() -> dict:
    # The branch offset makes the within-step image sets different while every
    # image is still exposed exactly eight times in each branch.
    updates = []
    for step in range(source256.UPDATE_COUNT):
        start = (step * source256.BRANCH_IMAGE_COUNT) % source256.IMAGE_COUNT
        common = [(start + index) % source256.IMAGE_COUNT for index in range(32)]
        variable = [
            (start + 64 + index) % source256.IMAGE_COUNT for index in range(32)
        ]
        updates.append(
            {
                "step": step + 1,
                "common_image_ids": common,
                "variable_image_ids": variable,
            }
        )
    return {"updates": updates}


def test_balanced_schedule_and_gate_use_all_256_denominator() -> None:
    schedule = source256.validate_schedule(
        _schedule(), expected_image_ids=list(range(source256.IMAGE_COUNT))
    )
    assert schedule["branch_presentations"] == {"common": 2048, "variable": 2048}
    assert schedule["presentations_per_image"] == 16

    passed = source256.eligibility_gate(_records(set(range(64))), schedule)
    assert passed["status"] == "passed"
    assert passed["eligible_image_count"] == 64
    assert passed["completion_presentations"] == 512
    assert passed["completion_presentation_fraction"] == 0.125

    stopped = source256.eligibility_gate(_records(set(range(63))), schedule)
    assert stopped["status"] == "stopped_below_eligibility_gate"
    assert stopped["train_image_denominator"] == 256
    assert stopped["completion_presentations"] == 504


def test_route_resolution_uses_completion_only_for_B_variable_and_keeps_fallback() -> None:
    records = _records({64})
    update = _schedule()["updates"][0]
    arm_a = source256.resolve_update_presentations(records, update, arm="A")
    arm_b = source256.resolve_update_presentations(records, update, arm="B")
    assert Counter(row["branch"] for row in arm_b) == {"common": 32, "variable": 32}
    assert {row["route_kind"] for row in arm_a} == {"canonical"}
    assert sum(row["route_kind"] == "completion" for row in arm_b) == 1
    completion = next(row for row in arm_b if row["route_kind"] == "completion")
    assert completion["target_owner_ids"] == ["64:1"]
    assert completion["target_owner_exposure_count"] == 1
    assert completion["prefix_owner_count"] == 1
    assert completion["prefix_token_count"] == 2
    fallback = [
        row for row in arm_b if row["branch"] == "variable" and row["image_id"] != 64
    ]
    assert len(fallback) == 31
    assert all(row["route_kind"] == "canonical" for row in fallback)
    assert all(row["fallback_reason"] == "no_trusted_prefix" for row in fallback)


def test_four_rank_local_objectives_sum_to_exact_half_half_sample_mean() -> None:
    parameter = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    common = [parameter * float(index + 1) for index in range(32)]
    variable = [parameter.square() * float(index + 1) for index in range(32)]
    common_hinges = [parameter.square() for _ in common]
    variable_hinges = [2 * parameter.square() for _ in variable]
    local = parameter.new_zeros(())
    for rank in range(4):
        start, stop = rank * 8, (rank + 1) * 8
        local = local + source256.objective_from_presentation_terms(
            common[start:stop] + variable[start:stop],
            common_hinges[start:stop] + variable_hinges[start:stop],
            ["common"] * 8 + ["variable"] * 8,
        )
    expected = 0.5 * sum(common) / 32 + 0.5 * sum(variable) / 32
    expected = expected + 0.01 * (
        0.5 * sum(common_hinges) / 32 + 0.5 * sum(variable_hinges) / 32
    )
    torch.testing.assert_close(local, expected)
    observed_gradient = torch.autograd.grad(local, parameter, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, parameter)[0]
    torch.testing.assert_close(observed_gradient, expected_gradient)


def test_completion_prefix_is_masked_but_remains_in_differentiable_history() -> None:
    class CausalToy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float64))

        def get_rope_index(
            self,
            input_ids: torch.Tensor,
            image_grid_thw: torch.Tensor,
            video_grid_thw: torch.Tensor | None,
            *,
            attention_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del image_grid_thw, video_grid_thw
            positions = attention_mask.cumsum(dim=1).sub(1).clamp_min(0)
            return positions.unsqueeze(0).expand(3, -1, -1), torch.zeros(
                input_ids.shape[0], 1, dtype=torch.long
            )

        def forward(self, **inputs: torch.Tensor | int | bool) -> SimpleNamespace:
            ids = inputs["input_ids"]
            assert isinstance(ids, torch.Tensor)
            cumulative = ids.double().cumsum(dim=1)
            vocab = torch.arange(128, dtype=torch.float64).view(1, 1, -1)
            logits = self.scale * cumulative.unsqueeze(-1) * vocab
            keep = inputs["logits_to_keep"]
            assert isinstance(keep, int)
            return SimpleNamespace(logits=logits[:, -keep:])

    record = source256.validate_route_record(
        _records({0})[0], eos_token_id=EOS
    )
    route = record["completion_route"]
    assert route is not None and route["ce_weights"] == [0, 0, 1, 1, 1, 1, 1]
    model = CausalToy()
    logits, _ = source256._batched_aligned_logits(
        model,
        {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "image_grid_thw": torch.ones((1, 3), dtype=torch.long),
        },
        [route],
        pad_token_id=0,
    )
    ce, _, active, _ = source256._route_terms(
        logits[0],
        route,
        {
            "coordinate_token_ids": list(range(128)),
            "coordinate_bin_values": list(range(128)),
            "margin": 1 / 999,
        },
    )
    assert active == 5
    ce.backward()
    assert model.scale.grad is not None and model.scale.grad.abs() > 0


def test_completion_rejects_geometry_on_masked_prefix() -> None:
    record = _records({0})[0]
    record["completion_route"]["trusted_boxes"] = [
        {
            "x1_position": 0,
            "y1_position": 2,
            "x2_position": 3,
            "y2_position": 4,
            "expected_bins": [30, 40, 41, 42],
        }
    ]
    with pytest.raises(ValueError, match="trusted coordinates must be active"):
        source256.validate_route_record(record, eos_token_id=EOS)


def test_seed_preparation_builds_matched_qualification_manifests(tmp_path) -> None:
    preparation = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-09-16-source256-fixed-prefix-completion/preparation/"
        "source256-data-v3/preparation.json"
    )
    if not preparation.is_file():
        pytest.skip("Source256 seed preparation is unavailable")
    try:
        manifests = {
            arm: source256_trial.build_training_manifest(
                preparation_path=preparation,
                arm=arm,
                mode="qualification",
                output=tmp_path,
            )
            for arm in source256.ARMS
        }
    except ValueError as exc:
        if "source bytes changed" not in str(exc):
            raise
        pytest.skip("concurrent Source256 preparation producer has not republished")
    for manifest in manifests.values():
        source256.validate_training_manifest(manifest)
        assert source256.source_adapter_scalar_count(manifest["source_adapter"]) == 18_006_016
        assert manifest["runtime"]["updates"] == 2
        assert manifest["runtime"]["max_model_forwards"] == 128
        assert manifest["scheduler"]["total_updates"] == 64
    ignored = {"arm", "model_config", "content_sha256"}
    assert {
        key: value for key, value in manifests["A"].items() if key not in ignored
    } == {
        key: value for key, value in manifests["B"].items() if key not in ignored
    }


def test_v3_lean_cases_hydrate_for_the_actual_bound_request_builder(monkeypatch) -> None:
    from src.inference import bound_requests
    from src.config.inference import load_research_infer_config

    preparation = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-09-16-source256-fixed-prefix-completion/preparation/"
        "source256-data-v3/preparation.json"
    )
    if not preparation.is_file():
        pytest.skip("Source256 v3 preparation is unavailable")
    checked = source256.validate_preparation(json.loads(preparation.read_text()))
    records = source256.hydrate_bound_cases(checked)
    selected = [record["canonical_route"] for record in records[:4]]
    prompt_by_id = {
        route["example_id"]: route["prompt_token_ids"] for route in selected
    }

    def fake_prompt(raw, *_args, **_kwargs):
        return SimpleNamespace(
            chat_text=f"prompt:{raw.example_id}",
            expected_executed_prompt_token_ids=prompt_by_id[str(raw.example_id)],
            to_artifact_dict=lambda: {},
        )

    monkeypatch.setattr(bound_requests, "build_prompt_record", fake_prompt)
    config = load_research_infer_config(source256_trial.SOURCE_CONFIG).config_dict
    requests, _ = bound_requests.build_bound_native_requests(
        SimpleNamespace(processor=object()),
        config,
        [route["case"] for route in selected],
    )
    assert [request.request_id for request in requests] == [
        route["example_id"] for route in selected
    ]
    assert [list(request.expected_token_ids or ()) for request in requests] == [
        route["prompt_token_ids"] for route in selected
    ]
    assert all(
        route["case"]["image_plan"]["backend_prompt_token_count"]
        == len(route["prompt_token_ids"])
        for route in selected
    )
