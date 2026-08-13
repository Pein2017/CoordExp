from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

from src.artifacts.json_values import validate_json_value

from scripts.research.build_human13_row_contrast_successor import (
    DuplicateContrastEvent,
    Human13RowContrastLedger,
    OwnerCandidateGroup,
    PositiveRow,
    SuccessorRow,
)
from scripts.research.human13_row_contrast_live import (
    SuccessorDenominators,
    SuccessorLossPlan,
    SuccessorLossSite,
    SuccessorPayload,
    SuccessorLossRunner,
    build_gradient_projection_handler,
    build_successor_payload,
    materialize_successor_segments,
    successor_loss_context_factory,
)


@dataclass(frozen=True)
class _ImageEncoding:
    image_grid_thw: tuple[int, int, int] = (1, 4, 4)
    merged_visual_tokens: int = 4
    plan: object = field(default_factory=lambda: SimpleNamespace(merge_size=2))


@dataclass(frozen=True)
class _Skeleton:
    example_id: str
    input_ids: tuple[int, ...]
    prompt_token_count: int
    image_pad_physical_start: int = 1
    image_pad_physical_end: int = 5
    image_token_id: int = 151655

    @property
    def image_encoding(self) -> _ImageEncoding:
        return _ImageEncoding()


def _row(row_id: str, owner_id: str, x1: int = 151700) -> SuccessorRow:
    return SuccessorRow(
        row_id=row_id,
        trajectory_id="t",
        owner_id=owner_id,
        category="book",
        token_ids=(151646, 42, 151647, 151648, x1, 151710, 151720, 151730, 151649),
        description_offsets=(1,),
        coordinate_offsets=(4, 5, 6, 7),
    )


def _ledger() -> Human13RowContrastLedger:
    g_owner = "gt:1:0"
    h_owner = "gt:1:1"
    g = _row("g-row", g_owner)
    h = _row("h-row", h_owner, 151701)
    duplicate = _row("dup-row", g_owner, 151702)
    event = DuplicateContrastEvent(
        event_id="event",
        source_kind="manifest",
        source_id="sealed_manifest",
        image_id=1,
        trajectory_id="source",
        retained_row_id="g-row",
        prefix_token_ids=g.token_ids,
        duplicate_row=duplicate,
        target_owner_ids=(g_owner, h_owner),
        covered_owner_ids=(g_owner,),
        uncovered_owner_ids=(h_owner,),
        candidate_groups=(OwnerCandidateGroup(h_owner, "book", (h,)),),
        contrast_branch="same_description",
    )
    return Human13RowContrastLedger(
        schema_version="human13_row_contrast_successor.v1",
        manifest_path="/manifest",
        manifest_sha256="a" * 64,
        panel_sha256="b" * 64,
        tokenizer_sha256="c" * 64,
        coordinate_token_start=151670,
        coordinate_token_end_exclusive=152670,
        positive_rows=(
            PositiveRow(g_owner, "G", g),
            PositiveRow(h_owner, "H", h),
        ),
        g_watch_rows=(PositiveRow(g_owner, "G", g),),
        events=(event,),
        prior_sources=(),
        exclusions=(),
    )


def test_materialization_emits_union_replay_contrast_rectangle_and_watch() -> None:
    skeleton = _Skeleton("image:1", (10, 151655, 151655, 151655, 151655, 20), 6)
    result = materialize_successor_segments(
        _ledger(), {1: skeleton}, global_max_length=100
    )
    kinds = {
        binding.objective
        for segment in result.segments
        for binding in segment.encoded_example.human13_successor_bindings
    }
    assert kinds == {
        "union_candidate",
        "replay_ce",
        "row_duplicate",
        "row_candidate",
        "rectangle",
        "watch_ce",
    }
    assert len({segment.segment_id for segment in result.segments}) == len(
        result.segments
    )
    assert any(
        segment.encoded_length > len(skeleton.input_ids) for segment in result.segments
    )


def test_payload_maps_exact_causal_positions_and_global_counts() -> None:
    skeleton = _Skeleton("image:1", (10, 151655, 151655, 151655, 151655, 20), 6)
    materialized = materialize_successor_segments(
        _ledger(), {1: skeleton}, global_max_length=100
    )
    payload = build_successor_payload(
        materialized,
        arm_id="R1",
        expected_vocab_size=153000,
        vocab_groups=SimpleNamespace(vocab_size=153000),
        global_max_length=100,
        duplicate_margin=0.0,
        rectangle_margin=0.0,
    )
    assert payload.arm_id == "R1"
    assert payload.denominators.union_images == 1
    assert payload.denominators.replay_owners == 1
    assert payload.denominators.contrast_images == 1
    assert payload.denominators.rectangle_rows == 2
    assert payload.denominators.watch_owners == 1
    assert all(
        position >= 0
        for sites in payload.sites_by_pack.values()
        for site in sites
        for position in site.logits_positions
    )
    assert all(
        step.calibration_metadata.selected_causal_logits_positions
        for step in payload.micro_steps
    )


def test_loss_runner_scores_then_replays_exact_cross_pack_objective() -> None:
    skeleton = _Skeleton("image:1", (10, 151655, 151655, 151655, 151655, 20), 6)
    payload = build_successor_payload(
        materialize_successor_segments(_ledger(), {1: skeleton}, global_max_length=32),
        arm_id="R1",
        expected_vocab_size=153000,
        vocab_groups=SimpleNamespace(vocab_size=153000),
        global_max_length=32,
        duplicate_margin=0.25,
        rectangle_margin=0.5,
    )
    model = torch.nn.Linear(1, 1)

    def forward(_model, micro_step):
        positions = micro_step.calibration_metadata.selected_causal_logits_positions
        logits = torch.zeros(1, len(positions), 153000)
        for site in micro_step.metadata["human13_successor_sites"]:
            for token in site.target_token_ids:
                logits[..., token] = 2.0
        return SimpleNamespace(logits=logits, logits_position_ids=positions)

    runner = SuccessorLossRunner(
        arm_id="R1",
        denominators=payload.denominators,
        model=model,
        score_forward=forward,
        duplicate_margin=0.25,
        rectangle_margin=0.5,
    )
    plan = runner.prepare_planned_step(payload.micro_steps, world_size=1)
    assert plan.score_forward_count == len(payload.micro_steps)
    bundles = []
    for index, micro_step in enumerate(payload.micro_steps):
        result = forward(model, micro_step)
        bundles.append(
            runner.compute_micro_step(
                successor_loss_context_factory(micro_step, result),
                plan,
                local_micro_step_index=index,
            )
        )
    artifact = runner.finalize_planned_step(
        tuple(bundle.to_artifact_dict() for bundle in bundles), plan
    )
    assert artifact["two_pass_exact"] is True
    assert artifact["row_contrast"]["event_count"] == 1
    assert artifact["union"]["candidate_count"] == 1
    validate_json_value(artifact)


def test_r2_payload_has_same_training_sites_and_declares_watch_projection() -> None:
    skeleton = _Skeleton("image:1", (10, 151655, 151655, 151655, 151655, 20), 6)
    materialized = materialize_successor_segments(
        _ledger(), {1: skeleton}, global_max_length=100
    )
    r1 = build_successor_payload(
        materialized,
        arm_id="R1",
        expected_vocab_size=153000,
        vocab_groups=SimpleNamespace(vocab_size=153000),
        global_max_length=100,
        duplicate_margin=0.0,
        rectangle_margin=0.0,
    )
    r2 = build_successor_payload(
        materialized,
        arm_id="R2",
        expected_vocab_size=153000,
        vocab_groups=SimpleNamespace(vocab_size=153000),
        global_max_length=100,
        duplicate_margin=0.0,
        rectangle_margin=0.0,
    )
    assert r1.sites_by_pack == r2.sites_by_pack
    assert r1.gradient_projection is False
    assert r2.gradient_projection is True


def test_r2_projection_handler_replays_watch_and_writes_safe_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = parameter

    model = Model()
    denominators = SuccessorDenominators(1, 1, 1, ((1, 1),), 0, (), 1, 1)
    site = SuccessorLossSite(
        objective="watch_ce",
        unit_id="gt:1:0",
        row_id="g-row",
        event_id=None,
        owner_id="gt:1:0",
        image_id=1,
        segment_id="watch",
        logits_positions=(0,),
        target_token_ids=(0,),
    )
    micro_step = SimpleNamespace(
        metadata={"human13_successor_sites": (site,)},
    )
    payload = SuccessorPayload(
        arm_id="R2",
        selected_segments=(),
        packed_plan=SimpleNamespace(),
        micro_steps=(micro_step,),
        sites_by_pack={0: (site,)},
        denominators=denominators,
        duplicate_margin=0.0,
        rectangle_margin=0.0,
        gradient_projection=True,
    )
    plan = SuccessorLossPlan(
        denominators=denominators,
        micro_step_count=1,
        union_weights=(),
        union_reference_nll=(),
        contrast_weights=(),
        contrast_reference_losses=(),
        score_forward_count=0,
        parameter_versions=(),
    )

    class Runtime:
        world_size = 1

        def __init__(self) -> None:
            self.zero_count = 0
            self.backward_count = 0

        def zero_gradients(self, *, planned_step_id: int) -> None:
            assert planned_step_id == 1
            model.zero_grad(set_to_none=True)
            self.zero_count += 1

        def backward(self, loss, *, planned_step_id: int, sync_gradients: bool):
            assert planned_step_id == 1
            assert sync_gradients is True
            loss.backward()
            self.backward_count += 1

    runtime = Runtime()

    def forward(_model, _micro_step):
        logits = torch.stack(
            (model.weight, -model.weight, model.weight * 0.0), dim=0
        ).reshape(1, 1, 3)
        return SimpleNamespace(logits=logits, logits_position_ids=(0,))

    # The R1 direction is adverse to the watch gradient and must be projected.
    parameter.grad = torch.tensor(1.0)
    receipt = build_gradient_projection_handler(payload, epsilon=1e-12, tolerance=1e-5)(
        forward, model, (micro_step,), plan, runtime, 1
    )

    assert runtime.zero_count == 1
    assert runtime.backward_count == 1
    assert receipt["applied"] is True
    assert receipt["watch_forward_count"] == 1
    assert receipt["written_post_dot"] >= -1e-5
