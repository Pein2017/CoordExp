from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.config.models import OptimizerConfig
from src.losses.normalizers import SegmentBalancedDenominator
from src.optim.factory import build_optimizer_and_scheduler
from src.optim.parameter_groups import OptimizerGroupAssignment, OptimizerGroupPlan

from probes.coordinate_representation.coordinate_codebook_alignment.scale_train import ScaleTrainingError, _ScaleProbe, _validate_parameter_delta, expected_first_two_updates


def _packet() -> dict:
    return {
        "status": "CPU_packing_and_schedule_verified",
        "schedule": {
            "resolved_max_steps": 1968,
            "runtime_batch": {
                "world_size": 4,
                "resolved_grad_accum_steps": 2,
                "effective_batch_size": 8,
            },
        },
        "rank_checks": {
            "4": {
                "gradient_accumulation": 2,
                "microsteps_per_rank": 3936,
                "rank_major_exact": True,
            }
        },
        "global_pack_indices": [index % 492 for index in range(1968 * 8)],
        "packs": [
            {"index": index, "row_ids": [f"row-{index}"]}
            for index in range(492)
        ],
    }


def test_rank_schedule_mutation_fails_closed() -> None:
    packet = _packet()
    packet["rank_checks"]["4"]["gradient_accumulation"] = 1

    with pytest.raises(ScaleTrainingError, match="schedule mismatch"):
        expected_first_two_updates(packet, rank=0)


def test_capture_plan_retains_global_denominator_and_rank_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _packet()
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")

    expected_by_rank = {
        rank: expected_first_two_updates(packet, rank=rank)[:2]
        for rank in range(4)
    }

    def gather(output: list[object], _local: object) -> None:
        for rank in range(4):
            output[rank] = [
                {"rank": rank, **record} for record in expected_by_rank[rank]
            ]

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    probe = _ScaleProbe(tmp_path / "first-production.json", packet)
    denominator = SegmentBalancedDenominator(
        term_name="base_ce",
        denominator_scope="planned_step_global",
        eligible_segment_count=8,
        selected_atom_count=64,
        skipped_segment_count=0,
        context_count=8,
    )
    plan = SimpleNamespace(
        denominators={"base_ce": denominator},
        world_size=4,
        backend_gradient_scale=4.0,
        counts={"count/eligible_segments": 8},
    )
    micro_steps = [
        SimpleNamespace(metadata={"pack_id": 0, "example_ids": ["row-0"]}),
        SimpleNamespace(metadata={"pack_id": 4, "example_ids": ["row-4"]}),
    ]

    probe.capture_plan(micro_steps, plan)

    record = probe.plan_records[0]
    assert record["eligible_segment_count"] == 8
    assert record["global_denominator"]["denominator_scope"] == "planned_step_global"
    assert record["backend_gradient_scale"] == 4.0
    assert record["global_pack_binding_count"] == 8
    assert [item["pack_id"] for item in record["pack_bindings"]] == [0, 4]


def test_eight_rank_one_pack_global_plan_rejects_old_scaling(tmp_path, monkeypatch):
    from probes.coordinate_representation.coordinate_codebook_alignment import scale_train as caller

    monkeypatch.setattr(caller, "EXPECTED_WORLD_SIZE", 8)
    monkeypatch.setattr(caller, "EXPECTED_GRAD_ACCUM", 1)
    monkeypatch.setattr(caller, "EXPECTED_UPDATES", 984)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "7")
    packet = _packet()
    packet["schedule"]["resolved_max_steps"] = 984
    packet["schedule"]["runtime_batch"].update(world_size=8, resolved_grad_accum_steps=1)
    packet["rank_checks"]["8"] = {
        "gradient_accumulation": 1, "microsteps_per_rank": 984, "rank_major_exact": True,
    }
    packet["global_pack_indices"] = list(range(492)) * 16
    assert [x["pack_id"] for x in expected_first_two_updates(packet, rank=7)] == [7, 15]
    with monkeypatch.context() as changed:
        changed.setitem(packet["schedule"]["runtime_batch"], "resolved_grad_accum_steps", 2)
        with pytest.raises(ScaleTrainingError, match="schedule mismatch"):
            expected_first_two_updates(packet, rank=7)

    def gather(output, _local):
        for rank in range(8):
            output[rank] = [{"rank": rank, **expected_first_two_updates(packet, rank=rank)[0]}]

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    denominator = SegmentBalancedDenominator(
        term_name="base_ce", denominator_scope="planned_step_global",
        eligible_segment_count=8, selected_atom_count=64,
        skipped_segment_count=0, context_count=8,
    )
    plan = SimpleNamespace(denominators={"base_ce": denominator}, world_size=8,
                           backend_gradient_scale=8.0, counts={})
    micro_steps = [SimpleNamespace(metadata={"pack_id": 7, "example_ids": ["row-7"]})]
    probe = _ScaleProbe(tmp_path / "eight.json", packet)
    probe.capture_plan(micro_steps, plan)
    assert probe.plan_records[0]["global_pack_binding_count"] == 8
    assert probe.plan_records[0]["backend_gradient_scale"] == 8.0
    wrong = SimpleNamespace(**{**vars(plan), "backend_gradient_scale": 4.0})
    with pytest.raises(ScaleTrainingError, match="backend gradient scale"):
        _ScaleProbe(tmp_path / "wrong.json", packet).capture_plan(micro_steps, wrong)


def test_zero_initial_warmup_lr_is_allowed_but_positive_second_update_is_required() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer_config = OptimizerConfig.model_validate(
        {
            "name": "adamw_torch",
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-8,
            "kwargs": {},
            "groups": {
                "adapters": {
                    "language": None,
                    "vision": None,
                    "aligner": None,
                },
                "token_embeddings": {"lr": 1.0e-3, "weight_decay": 0.0},
                "coordinate_codebook": None,
            },
            "scheduler": {
                "name": "constant_with_warmup",
                "warmup_steps": 10,
                "kwargs": {},
            },
        }
    )
    group_plan = OptimizerGroupPlan(
        groups=(OptimizerGroupAssignment("token_embeddings", 1.0e-3, 0.0, ("p",)),),
        parameters_by_name={"p": parameter},
    )
    optimizer, scheduler = build_optimizer_and_scheduler(
        optimizer_config, group_plan, total_training_steps=20
    )

    parameter.grad = torch.ones_like(parameter)
    first_rates = [float(group["lr"]) for group in optimizer.param_groups]
    first_before = parameter.detach().clone()
    optimizer.step()
    first_maxima = {"p": float((parameter.detach() - first_before).abs().max())}
    scheduler.step()
    assert first_rates == [0.0]
    assert not any(first_maxima.values())  # the old unconditional gate rejected this
    _validate_parameter_delta(
        planned_step_id=1, learning_rates=first_rates, maxima=first_maxima
    )

    optimizer.zero_grad()
    parameter.grad = torch.ones_like(parameter)
    second_rates = [float(group["lr"]) for group in optimizer.param_groups]
    second_before = parameter.detach().clone()
    optimizer.step()
    second_maxima = {"p": float((parameter.detach() - second_before).abs().max())}
    assert second_rates[0] > 0.0
    assert any(second_maxima.values())
    _validate_parameter_delta(
        planned_step_id=2, learning_rates=second_rates, maxima=second_maxima
    )
    with pytest.raises(ScaleTrainingError, match="positive-LR"):
        _validate_parameter_delta(
            planned_step_id=2, learning_rates=second_rates, maxima={"p": 0.0}
        )


def test_actual_probe_before_optimizer_after_order(tmp_path, monkeypatch):
    from probes.coordinate_representation.coordinate_codebook_alignment import scale_train as caller

    model = torch.nn.Module()
    for path in ("language_model.p", "visual.p", "visual.merger.p",
                 "language_model.embed_tokens.shared_embed_delta",
                 "lm_head.shared_embed_delta", "coordinate_codebook.raw_gain"):
        owner = model
        parts = path.split(".")
        for part in parts[:-1]:
            if not hasattr(owner, part):
                setattr(owner, part, torch.nn.Module())
            owner = getattr(owner, part)
        setattr(owner, parts[-1], torch.nn.Parameter(torch.tensor([1.0])))
    model.get_input_embeddings = lambda: model.language_model.embed_tokens
    model.get_output_embeddings = lambda: model.lm_head
    names = tuple(dict(model.named_parameters()))
    config = OptimizerConfig.model_validate({
        "name": "adamw_torch", "betas": [0.9, 0.999], "epsilon": 1e-8,
        "groups": {"adapters": {"language": None, "vision": None, "aligner": None},
        "token_embeddings": {"lr": 1e-3, "weight_decay": 0}, "coordinate_codebook": None},
        "scheduler": {"name": "constant_with_warmup", "warmup_steps": 10}})
    optimizer, scheduler = build_optimizer_and_scheduler(config, OptimizerGroupPlan(
        groups=(OptimizerGroupAssignment("token_embeddings", 1e-3, 0, names),),
        parameters_by_name=dict(model.named_parameters())), total_training_steps=1968)
    runtime = SimpleNamespace(model=model, optimizer=optimizer)
    probe = _ScaleProbe(tmp_path / "actual-caller.json", _packet())
    probe.plan_records = [{}, {}]  # Loss-plan identity has its separate real-type test above.
    probe.frozen_before = caller._parameter_hashes(model)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = probe.before_update(planned_step_id=1, runtime=runtime)
    optimizer.step()
    probe.after_update(planned_step_id=1, runtime=runtime, before=before)
    scheduler.step()
    assert all(lr == 0 for lr in before["optimizer_learning_rates_before"])
    assert not any(probe.update_records[0]["parameter_max_abs_deltas"].values())
    assert all(float(state["step"]) == 1 for state in optimizer.state.values())

    validator = caller._validate_parameter_delta
    def premature_delta_check(**kwargs):
        kwargs["require_change"] = True
        return validator(**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(caller, "_validate_parameter_delta", premature_delta_check)
        with pytest.raises(ScaleTrainingError, match="positive-LR"):
            probe.before_update(planned_step_id=2, runtime=runtime)
    before = probe.before_update(planned_step_id=2, runtime=runtime)
    with pytest.raises(ScaleTrainingError, match="positive-LR"):
        probe.after_update(planned_step_id=2, runtime=runtime, before=before)
    optimizer.step()
    probe.after_update(planned_step_id=2, runtime=runtime, before=before)
    assert any(probe.update_records[1]["parameter_max_abs_deltas"].values())
    assert probe.receipt_written
