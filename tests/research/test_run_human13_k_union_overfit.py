from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from scripts.research import run_human13_k_union_overfit as runner
from src.artifacts.checkpoints import CheckpointWriter
from src.artifacts.run_writer import RunWriter
from src.config.models import RuntimeBatchResolution, RuntimeConfig
from src.losses.runner import LossBundle
from src.runtime import GateDecision
from src.runtime.train_runtime import TrainRuntime
from src.training.supervised_trainer import SupervisedMicroStep


def test_logical_roles_keep_coherent_and_atomic_image_units() -> None:
    segments = runner.build_logical_segments(
        (
            _segment("a1:1", 1, "a1_full_h", 8),
            _segment("a8:2", 2, "a8_full_h", 8),
            _segment("gt:3", 3, "full_gt", 8),
            _segment("a4:4", 4, "a4_union", 8),
            _segment("replay:1", 1, "source_replay", 8),
            _segment("dup:1:0", 1, "duplicate_event", 8),
        )
    )

    assert tuple(item.role for item in segments) == (
        "a1_full_h",
        "a8_full_h",
        "full_gt",
        "a4_union",
        "source_replay",
        "duplicate_event",
    )
    with pytest.raises(ValueError, match="one coherent segment per image"):
        runner.build_logical_segments(
            (
                _segment("a1:1:first", 1, "a1_full_h", 8),
                _segment("a1:1:split", 1, "a1_full_h", 7),
            )
        )
    with pytest.raises(ValueError, match="atomic candidate group"):
        runner.build_logical_segments(
            (
                _segment("a4:1:first", 1, "a4_union", 8),
                _segment("a4:1:split", 1, "a4_union", 7),
            )
        )


def test_stable_descending_length_first_fit_reuses_no_padding_planner() -> None:
    plan = runner.plan_panel_packs(
        (
            _segment("d", 4, "source_replay", 6),
            _segment("b", 2, "source_replay", 8),
            _segment("a", 1, "source_replay", 9),
            _segment("c", 3, "source_replay", 7),
        ),
        global_max_length=15,
    )

    assert tuple(
        tuple(segment.example_id for segment in pack.pack.segments)
        for pack in plan.packs
    ) == (("a", "d"), ("b", "c"))
    assert tuple(pack.pack.length for pack in plan.packs) == (15, 15)
    assert plan.packing_claims == ("zero_padding", "physical_launch_reduction")


def test_equal_lengths_use_stable_identity_tie_breaker() -> None:
    plan = runner.plan_panel_packs(
        (
            _segment("z", 3, "source_replay", 6),
            _segment("a", 1, "source_replay", 6),
        ),
        global_max_length=12,
    )

    assert tuple(item.example_id for item in plan.packs[0].pack.segments) == (
        "a",
        "z",
    )


def test_every_packed_segment_has_independent_causal_and_mrope_positions() -> None:
    plan = runner.plan_panel_packs(
        (
            _segment("long", 1, "source_replay", 8),
            _segment("short", 2, "duplicate_event", 7),
        ),
        global_max_length=20,
    )

    packed = plan.packs[0]
    assert packed.fa2_varlen_plan.segment_boundaries == (0, 8, 15)
    assert packed.fa2_varlen_plan.attention_mask is None
    assert packed.position_inputs.reset_points == (0, 8)
    assert packed.position_inputs.position_ids[0, 0, 8:15].tolist() == list(range(7))


def test_12000_preflight_rejects_before_position_or_forward_planning() -> None:
    with pytest.raises(ValueError, match="12,000-token"):
        runner.plan_panel_packs(
            (_segment("too-long", 1, "a4_union", 12_001),),
        )


def test_compact_counters_contain_only_named_measured_fields() -> None:
    plan = runner.plan_panel_packs(
        (
            _segment("a", 1, "source_replay", 8),
            _segment("b", 2, "duplicate_event", 7),
        ),
        global_max_length=10,
    )
    counters = plan.performance_counters(
        gpu_seconds=1.25,
        wall_time_seconds=2.5,
        peak_memory_bytes=4096,
    ).to_artifact_dict()

    assert counters == {
        "pack_count": 2,
        "logical_tokens": 15,
        "packed_tokens": 15,
        "padding_tokens": 0,
        "utilization": 0.75,
        "gpu_seconds": 1.25,
        "wall_time_seconds": 2.5,
        "peak_memory_bytes": 4096,
    }
    assert not any(
        "cache" in key or "reuse" in key or "flop" in key for key in counters
    )


def test_dry_run_has_zero_model_optimizer_checkpoint_and_gpu_actions() -> None:
    plan = runner.plan_panel_packs(
        (_segment("a", 1, "source_replay", 7),),
        global_max_length=10,
    )

    receipt = runner.dry_run_receipt(plan)

    assert receipt["mode"] == "dry_run"
    assert receipt["actions"] == {
        "model_loads": 0,
        "forwards": 0,
        "backwards": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "gpu_allocations": 0,
    }


def test_every_pack_uses_complete_panel_denominators() -> None:
    denominator_plan = runner.Human13PanelDenominators(
        family_counts=(("h", 3), ("replay", 2), ("duplicate", 2)),
        duplicate_event_counts=((1, 2), (2, 1)),
    )
    loss_runner = runner.Human13PanelLossRunner(
        denominators=denominator_plan,
        coefficients=(("h", 1.0), ("replay", 1.0), ("duplicate", 1.0)),
    )
    micro_steps = (
        _loss_micro_step(0, denominator_plan),
        _loss_micro_step(1, denominator_plan),
    )

    loss_plan = loss_runner.prepare_planned_step(micro_steps)
    first = loss_runner.compute_micro_step(
        _owner_ce_context(target_token_id=0, unit_id="owner-1", image_id=1),
        loss_plan,
        local_micro_step_index=0,
    )
    second = loss_runner.compute_micro_step(
        _owner_ce_context(target_token_id=1, unit_id="owner-2", image_id=2),
        loss_plan,
        local_micro_step_index=1,
    )

    assert first.term_by_name("h").denominator.eligible_segment_count == 3
    assert second.term_by_name("h").denominator.eligible_segment_count == 3
    expected = torch.log(torch.tensor(3.0)).item() / 3
    assert first.term_by_name("h").raw_loss.item() == pytest.approx(expected)
    assert second.term_by_name("h").raw_loss.item() == pytest.approx(expected)


def test_a4_union_dispatch_is_once_per_image_across_candidate_rows() -> None:
    denominators = runner.Human13PanelDenominators(family_counts=(("h", 1),))
    loss_runner = runner.Human13PanelLossRunner(
        denominators=denominators,
        coefficients=(("h", 1.0),),
    )
    plan = loss_runner.prepare_planned_step((_loss_micro_step(0, denominators),))
    context = runner.Human13PackLossContext(
        logits=torch.tensor(
            [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]],
            requires_grad=True,
        ),
        logits_position_ids=(0, 1, 2, 3),
        sites=(
            runner.Human13LossSite(
                family="h",
                objective="union_mass",
                unit_id="candidate-a",
                image_id=1,
                logits_positions=(0, 1),
                target_token_ids=(0, 0),
            ),
            runner.Human13LossSite(
                family="h",
                objective="union_mass",
                unit_id="candidate-b",
                image_id=1,
                logits_positions=(2, 3),
                target_token_ids=(1, 1),
            ),
        ),
    )

    bundle = loss_runner.compute_micro_step(context, plan, local_micro_step_index=0)

    assert bundle.term_by_name("h").raw_loss.item() == pytest.approx(
        torch.log(torch.tensor(2.0)).item()
    )
    assert bundle.term_by_name("h").diagnostics["atomic_image_count"] == 1


def test_duplicate_events_split_across_packs_keep_full_image_and_panel_denominators() -> (
    None
):
    denominators = runner.Human13PanelDenominators(
        family_counts=(("duplicate", 2),),
        duplicate_event_counts=((1, 2), (2, 1)),
    )
    loss_runner = runner.Human13PanelLossRunner(
        denominators=denominators,
        coefficients=(("duplicate", 1.0),),
    )
    plan = loss_runner.prepare_planned_step(
        (_loss_micro_step(0, denominators), _loss_micro_step(1, denominators))
    )
    first = loss_runner.compute_micro_step(
        _duplicate_context(image_id=1, unit_id="event-1"),
        plan,
        local_micro_step_index=0,
    )
    second = loss_runner.compute_micro_step(
        _duplicate_context(image_id=1, unit_id="event-2"),
        plan,
        local_micro_step_index=1,
    )

    expected_event_loss = torch.nn.functional.softplus(-torch.log(torch.tensor(2.0)))
    assert (
        first.term_by_name("duplicate").raw_loss
        + second.term_by_name("duplicate").raw_loss
    ).item() == pytest.approx((expected_event_loss / 2).item())


def test_denominator_mismatch_in_any_pack_fails_before_forward() -> None:
    expected = runner.Human13PanelDenominators(family_counts=(("h", 3),))
    wrong = runner.Human13PanelDenominators(family_counts=(("h", 2),))
    loss_runner = runner.Human13PanelLossRunner(
        denominators=expected,
        coefficients=(("h", 1.0),),
    )

    with pytest.raises(ValueError, match="complete panel denominator"):
        loss_runner.prepare_planned_step(
            (_loss_micro_step(0, expected), _loss_micro_step(1, wrong))
        )


def test_plan_adapts_to_existing_qwen_compact_forward_and_loss_context() -> None:
    plan = runner.plan_panel_packs(
        (_segment("owner", 1, "a1_full_h", 8),), global_max_length=10
    )
    denominators = runner.Human13PanelDenominators(family_counts=(("h", 1),))
    site = runner.Human13LossSite(
        family="h",
        objective="owner_ce",
        unit_id="owner-1",
        image_id=1,
        logits_positions=(4, 6),
        target_token_ids=(2, 1),
    )
    token_sequence = SimpleNamespace(pack_index=0)

    micro_steps = runner.build_supervised_micro_steps(
        plan,
        denominators=denominators,
        token_sequences={0: token_sequence},
        vocab_groups=SimpleNamespace(vocab_size=3),
        sites_by_pack={0: (site,)},
        expected_vocab_size=3,
    )

    assert len(micro_steps) == 1
    micro_step = micro_steps[0]
    assert micro_step.pack is plan.packs[0].pack
    assert micro_step.position_inputs is plan.packs[0].position_inputs
    assert micro_step.calibration_metadata.selected_causal_logits_positions == (4, 6)
    context = runner.human13_loss_context_factory(
        micro_step,
        SimpleNamespace(
            logits=torch.tensor([[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]]),
            logits_position_ids=(4, 6),
        ),
    )
    assert context.sites == (site,)
    assert context.logits_position_ids == (4, 6)


def test_training_entry_rejects_unsealed_and_partial_manifest(tmp_path: Path) -> None:
    unsealed = tmp_path / "manifest.json"
    unsealed.write_text("{}\n", encoding="utf-8")
    with pytest.raises((FileNotFoundError, ValueError)):
        runner.load_sealed_training_manifest(unsealed)

    with pytest.raises(ValueError, match="full-panel"):
        runner._require_full_panel_manifest(SimpleNamespace(full_panel=False))


def test_execution_rejects_sealed_manifest_binding_mismatch_before_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed, execution, micro_steps = _bound_a1_execution()
    mismatched = replace(
        execution,
        manifest_identity=replace(
            execution.manifest_identity, manifest_sha256="f" * 64
        ),
    )
    monkeypatch.setattr(runner, "load_sealed_training_manifest", lambda _: sealed)
    forward_calls = 0

    def forbidden_forward(*_: object) -> object:
        nonlocal forward_calls
        forward_calls += 1
        raise AssertionError("forward must not run")

    writer = RecordingWriter()
    with pytest.raises(ValueError, match="sealed manifest identity"):
        runner.run_panel_exposure(
            manifest_path=tmp_path / "sealed.json",
            execution_plan=mismatched,
            model=torch.nn.Linear(1, 1, bias=False),
            micro_steps=micro_steps,
            runtime=RecordingRuntime(
                torch.nn.Linear(1, 1, bias=False),
                torch.optim.AdamW(torch.nn.Linear(1, 1).parameters()),
            ),
            qwen_forward=forbidden_forward,
            run_writer=writer,
            checkpoint_writer=RecordingCheckpointWriter(),
            checkpoint_kwargs={"adapter_name": "human13"},
            updated_at="2026-08-12T00:00:00Z",
        )

    assert forward_calls == 0
    assert writer.finalized["status"] == "failed"


def test_a4_complete_candidate_group_cannot_split_across_packs() -> None:
    sealed, packed_plan, sites_by_pack = _a4_split_fixture()

    with pytest.raises(ValueError, match="A4 candidate group.*one segment.*one pack"):
        runner.build_execution_plan(
            sealed,
            arm_id="A4",
            packed_plan=packed_plan,
            sites_by_pack=sites_by_pack,
        )


def test_finalized_artifact_preserves_bounded_family_diagnostics() -> None:
    denominators = runner.Human13PanelDenominators(
        family_counts=(("h", 1), ("duplicate", 1)),
        duplicate_event_counts=((1, 1),),
    )
    loss_runner = runner.Human13PanelLossRunner(
        denominators=denominators,
        coefficients=(("h", 1.0), ("duplicate", 1.0)),
    )
    plan = loss_runner.prepare_planned_step((_loss_micro_step(0, denominators),))
    context = _a4_and_duplicate_context()
    bundle = loss_runner.compute_micro_step(context, plan, local_micro_step_index=0)

    finalized = loss_runner.finalize_planned_step((bundle.to_artifact_dict(),), plan)

    terms = {term["name"]: term for term in finalized["terms"]}
    assert terms["h"]["denominator"]["eligible_segment_count"] == 1
    assert terms["h"]["diagnostics"]["candidate_weights"] == pytest.approx([0.5, 0.5])
    assert terms["h"]["diagnostics"]["effective_owner_count"] == pytest.approx(2.0)
    assert terms["duplicate"]["diagnostics"] == {
        "raw_event_count": 1,
        "consumed_event_count": 1,
        "capped_event_count": 0,
    }


def test_nonfinite_multi_pack_refusal_finalizes_failed_without_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed, execution, micro_steps = _bound_a1_execution(two_packs=True)
    monkeypatch.setattr(runner, "load_sealed_training_manifest", lambda _: sealed)
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    runtime = NonfiniteRecordingRuntime(model, optimizer)
    writer = RecordingWriter()
    checkpoint = RecordingCheckpointWriter()

    with pytest.raises(RuntimeError, match="finite applied panel update"):
        runner.run_panel_exposure(
            manifest_path=tmp_path / "sealed.json",
            execution_plan=execution,
            model=model,
            micro_steps=micro_steps,
            runtime=runtime,
            qwen_forward=lambda _model, micro_step: SimpleNamespace(
                logits=torch.zeros(
                    (
                        1,
                        len(
                            micro_step.calibration_metadata.selected_causal_logits_positions
                        ),
                        3,
                    ),
                    requires_grad=True,
                ),
                logits_position_ids=micro_step.calibration_metadata.selected_causal_logits_positions,
            ),
            run_writer=writer,
            checkpoint_writer=checkpoint,
            checkpoint_kwargs={"adapter_name": "human13"},
            updated_at="2026-08-12T00:00:00Z",
        )

    assert runtime.optimizer_step_count == 0
    assert checkpoint.calls == []
    assert writer.finalized["status"] == "failed"
    assert writer.finalized["completed_steps"] == 0


def test_one_exposure_keeps_parameters_and_adamw_state_between_packs_then_steps_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0)
    runtime = RecordingRuntime(model, optimizer)
    snapshots: list[tuple[float, int]] = []

    def qwen_forward(_model: object, micro_step: SupervisedMicroStep) -> object:
        snapshots.append((float(model.weight.detach()), len(optimizer.state)))
        positions = micro_step.calibration_metadata.selected_causal_logits_positions
        value = model.weight.square() * float(micro_step.pack.pack_index + 1)
        logits = torch.cat((value, -value, value * 0.0), dim=1).unsqueeze(0)
        return SimpleNamespace(
            logits=logits.expand(1, len(positions), 3),
            logits_position_ids=positions,
        )

    writer = RecordingWriter()
    checkpoint_writer = RecordingCheckpointWriter()
    sealed, execution, micro_steps = _bound_a1_execution(two_packs=True)
    monkeypatch.setattr(
        runner,
        "load_sealed_training_manifest",
        lambda _: sealed,
    )

    result = runner.run_panel_exposure(
        manifest_path=tmp_path / "sealed.json",
        execution_plan=execution,
        model=model,
        micro_steps=micro_steps,
        runtime=runtime,
        qwen_forward=qwen_forward,
        run_writer=writer,
        checkpoint_writer=checkpoint_writer,
        checkpoint_kwargs={"adapter_name": "human13"},
        updated_at="2026-08-12T00:00:00Z",
    )

    assert snapshots == [(1.0, 0), (1.0, 0)]
    assert runtime.optimizer_step_count == 1
    assert result.completed_steps == 1
    assert result.consumed_micro_steps == 2
    assert len(checkpoint_writer.calls) == 1
    assert checkpoint_writer.calls[0]["step"] == 1
    assert writer.rows[0]["performance"] == {
        "pack_count": 2,
        "logical_tokens": 24,
        "packed_tokens": 24,
        "padding_tokens": 0,
        "utilization": 0.75,
        "gpu_seconds": None,
        "wall_time_seconds": None,
        "peak_memory_bytes": None,
    }
    assert writer.finalized["completed_steps"] == 1
    assert writer.finalized["consumed_packs"] == 2


def test_cpu_vertical_real_qwen_runtime_and_writers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed, execution, micro_steps = _bound_a1_execution(two_packs=True)
    monkeypatch.setattr(runner, "load_sealed_training_manifest", lambda _: sealed)
    model = CpuPackedModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    accelerator = CpuAccelerator()
    runtime = TrainRuntime(
        runtime_config=RuntimeConfig(seed=17),
        runtime_batch=RuntimeBatchResolution(
            world_size=1,
            effective_batch_size=2,
            resolved_grad_accum_steps=2,
        ),
        model=model,
        optimizer=optimizer,
        scheduler=None,
        expected_mixed_precision="no",
        max_grad_norm=1.0,
        accelerator=accelerator,
    )
    run_dir = tmp_path / "run"
    writer = RunWriter.initialize(
        run_dir=run_dir,
        run_id="human13-cpu",
        run_name="human13-cpu",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="2026-08-12T00:00:00Z",
        config_fingerprint="c" * 64,
        resolved_config={"arm": "A1"},
        world_size=1,
        resolved_max_steps=1,
    )

    result = runner.run_panel_exposure(
        manifest_path=tmp_path / "sealed.json",
        execution_plan=execution,
        model=model,
        micro_steps=micro_steps,
        runtime=runtime,
        run_writer=writer,
        checkpoint_writer=CheckpointWriter(run_dir),
        checkpoint_kwargs={"adapter_name": "default"},
        updated_at="2026-08-12T00:00:01Z",
    )

    state = writer.read_run()
    log_row = json.loads(writer.logging_path.read_text(encoding="utf-8"))
    assert result.completed_steps == runtime.optimizer_step_count == 1
    assert accelerator.backward_calls == 2
    assert model.selected_lengths == [2, 1]
    assert state["status"] == "completed"
    assert state["checkpoint_event_count"] == 1
    assert log_row["loss_bundle"]["terms"]
    assert (
        run_dir / "checkpoints" / "step-1" / "adapter" / "adapter_model.safetensors"
    ).is_file()


@dataclass(frozen=True)
class FakeEncodedExample:
    example_id: str
    input_ids: tuple[int, ...]
    image_pad_physical_start: int = 1
    image_pad_physical_end: int = 5
    image_grid_thw: tuple[int, int, int] = (1, 4, 4)
    merge_size: int = 2
    image_token_id: int = 151655

    @property
    def image_encoding(self) -> "FakeImageEncoding":
        return FakeImageEncoding(
            image_grid_thw=self.image_grid_thw,
            merged_visual_tokens=self.image_pad_physical_end
            - self.image_pad_physical_start,
            plan=FakeImagePlan(merge_size=self.merge_size),
        )


@dataclass(frozen=True)
class FakeImageEncoding:
    image_grid_thw: tuple[int, int, int]
    merged_visual_tokens: int
    plan: "FakeImagePlan"
    pixel_values: torch.Tensor = field(default_factory=lambda: torch.zeros((16, 2)))


@dataclass(frozen=True)
class FakeImagePlan:
    merge_size: int


def _segment(
    segment_id: str,
    image_id: int,
    role: runner.LogicalRole,
    length: int,
) -> runner.LogicalPanelSegment:
    token_ids = (10, 151655, 151655, 151655, 151655) + tuple(range(20, 20 + length - 5))
    encoded = FakeEncodedExample(segment_id, token_ids)
    return runner.LogicalPanelSegment(
        segment_id=segment_id,
        image_id=image_id,
        role=role,
        encoded_example=encoded,
    )


def _loss_micro_step(
    pack_index: int,
    denominators: runner.Human13PanelDenominators,
) -> SimpleNamespace:
    return SimpleNamespace(
        metadata={"human13_panel_denominators": denominators},
        pack=SimpleNamespace(pack_index=pack_index),
    )


def _owner_ce_context(
    *, target_token_id: int, unit_id: str, image_id: int
) -> runner.Human13PackLossContext:
    return runner.Human13PackLossContext(
        logits=torch.tensor([[[0.0, 0.0, 0.0]]], requires_grad=True),
        logits_position_ids=(0,),
        sites=(
            runner.Human13LossSite(
                family="h",
                objective="owner_ce",
                unit_id=unit_id,
                image_id=image_id,
                logits_positions=(0,),
                target_token_ids=(target_token_id,),
            ),
        ),
    )


def _duplicate_context(*, image_id: int, unit_id: str) -> runner.Human13PackLossContext:
    return runner.Human13PackLossContext(
        logits=torch.tensor([[[0.0, 0.0, 0.0]]], requires_grad=True),
        logits_position_ids=(0,),
        sites=(
            runner.Human13LossSite(
                family="duplicate",
                objective="duplicate_unlikelihood",
                unit_id=unit_id,
                image_id=image_id,
                logits_positions=(0,),
                target_token_ids=(0,),
            ),
        ),
    )


def _training_micro_step(pack_index: int) -> SupervisedMicroStep:
    return SupervisedMicroStep(
        pack=SimpleNamespace(pack_index=pack_index),
        encoded_examples=(),
        position_inputs=None,
        token_sequence=SimpleNamespace(),
        vocab_groups=SimpleNamespace(),
    )


def _sealed_manifest(*, arm_id: str = "A1") -> SimpleNamespace:
    selected = SimpleNamespace(
        owner_id="owner-1",
        row_id="row-1",
        token_ids=(2,),
        target_token_mask=(True,),
    )
    image = SimpleNamespace(
        image_id=1,
        owners=(SimpleNamespace(owner_id="owner-1"),),
        h_owner_ids=("owner-1",),
        g_owner_ids=(),
        duplicate_events=(SimpleNamespace(event_id="event-1"),),
        selected_rows=(selected,),
        candidate_row_ids=("row-1", "row-2") if arm_id == "A4" else ("row-1",),
        replay_row_ids=("replay-1",),
    )
    return SimpleNamespace(
        schema_version="human13_k_union_manifest.v1",
        full_panel=True,
        binding=SimpleNamespace(
            unit_id="2026-08-12-human13-k-union-to-greedy-overfit-screen",
            panel=SimpleNamespace(panel_sha256="a" * 64, owner_count=1),
        ),
        arms=(SimpleNamespace(arm_id=arm_id),),
        images=(image,),
        denominators=SimpleNamespace(
            target_image_count=1,
            target_owner_count=1,
            replay_owner_count=1,
            duplicate_image_count=1,
            duplicate_event_count=1,
        ),
        canonical_sha256="b" * 64,
    )


def _bound_a1_execution(
    *, two_packs: bool = False
) -> tuple[
    SimpleNamespace, runner.Human13ExecutionPlan, tuple[SupervisedMicroStep, ...]
]:
    sealed = _sealed_manifest()
    segments = (
        _segment("a1:1", 1, "a1_full_h", 8),
        _segment("replay:1", 1, "source_replay", 8),
        _segment("duplicate:1", 1, "duplicate_event", 8),
    )
    packed = runner.plan_panel_packs(
        segments, global_max_length=16 if two_packs else 24
    )
    sites_by_pack = {
        pack.pack.pack_index: tuple(
            runner.Human13LossSite(
                family=(
                    "h"
                    if logical.role == "a1_full_h"
                    else "replay"
                    if logical.role == "source_replay"
                    else "duplicate"
                ),
                objective=(
                    "duplicate_unlikelihood"
                    if logical.role == "duplicate_event"
                    else "owner_ce"
                ),
                unit_id=(
                    "owner-1"
                    if logical.role == "a1_full_h"
                    else "replay-1"
                    if logical.role == "source_replay"
                    else "event-1"
                ),
                image_id=1,
                segment_id=logical.segment_id,
                manifest_row_ids=(
                    "row-1"
                    if logical.role == "a1_full_h"
                    else "replay-1"
                    if logical.role == "source_replay"
                    else "event-1",
                ),
                logits_positions=(packed_segment.end - 2,),
                target_token_ids=(2,),
            )
            for logical, packed_segment in zip(
                pack.logical_segments, pack.pack.segments, strict=True
            )
        )
        for pack in packed.packs
    }
    execution = runner.build_execution_plan(
        sealed,
        arm_id="A1",
        packed_plan=packed,
        sites_by_pack=sites_by_pack,
    )
    micro_steps = runner.build_supervised_micro_steps(
        packed,
        denominators=execution.denominators,
        token_sequences={
            pack.pack.pack_index: SimpleNamespace(pack_index=pack.pack.pack_index)
            for pack in packed.packs
        },
        vocab_groups=SimpleNamespace(vocab_size=3),
        sites_by_pack=sites_by_pack,
        expected_vocab_size=3,
    )
    return sealed, execution, micro_steps


def _a4_split_fixture() -> tuple[
    SimpleNamespace,
    runner.PackedPanelPlan,
    dict[int, tuple[runner.Human13LossSite, ...]],
]:
    sealed = _sealed_manifest(arm_id="A4")
    packed = runner.plan_panel_packs(
        (
            _segment("a4:1:first", 1, "source_replay", 8),
            _segment("a4:1:second", 1, "source_replay", 8),
        ),
        global_max_length=8,
    )
    sites = {
        0: (
            runner.Human13LossSite(
                family="h",
                objective="union_mass",
                unit_id="candidate-a",
                image_id=1,
                segment_id="a4:1:first",
                manifest_row_ids=("row-1",),
                logits_positions=(6,),
                target_token_ids=(0,),
            ),
        ),
        1: (
            runner.Human13LossSite(
                family="h",
                objective="union_mass",
                unit_id="candidate-b",
                image_id=1,
                segment_id="a4:1:second",
                manifest_row_ids=("row-2",),
                logits_positions=(6,),
                target_token_ids=(1,),
            ),
        ),
    }
    return sealed, packed, sites


def _a4_and_duplicate_context() -> runner.Human13PackLossContext:
    return runner.Human13PackLossContext(
        logits=torch.zeros((1, 3, 3), requires_grad=True),
        logits_position_ids=(0, 1, 2),
        sites=(
            runner.Human13LossSite(
                family="h",
                objective="union_mass",
                unit_id="candidate-a",
                image_id=1,
                segment_id="a4:1",
                manifest_row_ids=("row-1",),
                logits_positions=(0,),
                target_token_ids=(0,),
            ),
            runner.Human13LossSite(
                family="h",
                objective="union_mass",
                unit_id="candidate-b",
                image_id=1,
                segment_id="a4:1",
                manifest_row_ids=("row-2",),
                logits_positions=(1,),
                target_token_ids=(1,),
            ),
            runner.Human13LossSite(
                family="duplicate",
                objective="duplicate_unlikelihood",
                unit_id="event-1",
                image_id=1,
                segment_id="dup:1",
                manifest_row_ids=("duplicate-1",),
                logits_positions=(2,),
                target_token_ids=(2,),
            ),
        ),
    )


class CpuPackedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.0]))
        self.config = SimpleNamespace(vocab_size=3)
        self.selected_lengths: list[int] = []

    def forward(self, **kwargs: object) -> object:
        selected = kwargs["logits_to_keep"]
        count = int(selected.numel()) if isinstance(selected, torch.Tensor) else 0
        self.selected_lengths.append(count)
        return SimpleNamespace(logits=self.weight.view(1, 1, 3).expand(1, count, 3))

    def save_pretrained(self, output_dir: str | Path, **_: object) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA", "use_dora": True}) + "\n",
            encoding="utf-8",
        )
        save_file(
            {
                "base_model.q_proj.lora_A.default.weight": self.weight[:2]
                .view(1, 2)
                .clone(),
                "base_model.q_proj.lora_B.default.weight": self.weight[:2]
                .view(2, 1)
                .clone(),
                "base_model.q_proj.lora_magnitude_vector.default.weight": self.weight.clone(),
            },
            str(path / "adapter_model.safetensors"),
        )


class CpuAccelerator:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.process_index = 0
        self.num_processes = 1
        self.is_main_process = True
        self.distributed_type = SimpleNamespace(name="NO")
        self.gradient_accumulation_steps = 1
        self.mixed_precision = "no"
        self.scaler = None
        self.backward_calls = 0

    def prepare(self, *objects: object) -> tuple[object, ...]:
        return objects

    def no_sync(self, _model: object) -> object:
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        self.backward_calls += 1
        loss.backward()

    def clip_grad_norm_(self, parameters: object, max_norm: float) -> None:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def wait_for_everyone(self) -> None:
        return None

    def unwrap_model(self, model: object) -> object:
        return model

    def broadcast_object_list(
        self, _values: list[object], *, from_process: int
    ) -> None:
        assert from_process == 0


class RecordingRuntime:
    def __init__(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.optimizer_step_count = 0
        self.scheduler_step_count = 0

    def move_micro_step(
        self, micro_step: SupervisedMicroStep, **_: object
    ) -> SupervisedMicroStep:
        return micro_step

    def pre_backward(self, bundle: LossBundle, *, planned_step_id: int) -> GateDecision:
        del bundle
        return _gate(planned_step_id, backward=True, optimizer=False)

    def accumulation_context(self, *, sync_gradients: bool) -> object:
        del sync_gradients
        return nullcontext()

    def backward(self, loss: torch.Tensor, **_: object) -> None:
        loss.backward()

    def post_backward(self, *, planned_step_id: int) -> GateDecision:
        return _gate(planned_step_id, backward=False, optimizer=True)

    def clip_gradients(self, **_: object) -> None:
        return None

    def optimizer_step(self, **_: object) -> None:
        self.optimizer.step()
        self.optimizer_step_count += 1

    def scheduler_step(self, **_: object) -> dict[str, int]:
        self.scheduler_step_count += 1
        return {"scheduler_step_count": self.scheduler_step_count}

    def zero_gradients(self, **_: object) -> None:
        self.optimizer.zero_grad(set_to_none=True)


class NonfiniteRecordingRuntime(RecordingRuntime):
    def pre_backward(self, bundle: LossBundle, *, planned_step_id: int) -> GateDecision:
        del bundle
        return GateDecision(
            stage="pre_backward_scalar",
            planned_step_id=planned_step_id,
            world_size=1,
            ranks=(0,),
            all_ranks_safe=False,
            should_call_backward=False,
            should_call_optimizer_step=False,
            should_clear_gradients=True,
            optimizer_update_status="skipped_non_finite",
            finite_status="nonfinite",
            reason_codes=("rank0:non_finite_scalar",),
            rank_diagnostics=(),
            diagnostics={},
        )


def _gate(planned_step_id: int, *, backward: bool, optimizer: bool) -> GateDecision:
    return GateDecision(
        stage="test",
        planned_step_id=planned_step_id,
        world_size=1,
        ranks=(0,),
        all_ranks_safe=True,
        should_call_backward=backward,
        should_call_optimizer_step=optimizer,
        should_clear_gradients=False,
        optimizer_update_status="ready_to_step" if optimizer else "accumulating",
        finite_status="finite",
        reason_codes=(),
        rank_diagnostics=(),
        diagnostics={"max_grad_norm": 1.0},
    )


class RecordingWriter:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self.finalized: dict[str, object] = {}

    def append_logging_row(self, row: dict[str, object]) -> None:
        self.rows.append(dict(row))

    def finalize(self, **kwargs: object) -> None:
        self.finalized = dict(kwargs)


class RecordingCheckpointWriter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def write_checkpoint(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(step=kwargs["step"])
