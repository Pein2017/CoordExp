from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from src.coordinate_targets import CoordinateLossTarget
from src.losses import LossRunner
from src.losses.vocab import TokenVocabularyGroups
from src.packing.planner import PackedSegment
from src.supervision import TokenAtom, TokenSequence

from probes.training_set_completion.coordinate_codebook_alignment.three_loss_checks import (
    ThreeLossQualificationError,
    ThreeLossQualificationProbe,
    ThreeLossRunnerHook,
)


COORDINATE_IDS = tuple(range(5, 1005))
VOCAB_SIZE = 1008
GROUPS = TokenVocabularyGroups(
    vocab_size=VOCAB_SIZE,
    desc_text=(1007,),
    schema=(0, 1),
    coordinate=COORDINATE_IDS,
    eos=(1006,),
    blocked=(2, 3, 4, 1005),
)


def test_three_loss_hook_matches_maintained_runner_with_unequal_roles() -> None:
    context = _context(requires_grad=True)
    runner = LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.2,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
        raw_axis_validity_hinge_weight=0.01,
    )
    probe = ThreeLossQualificationProbe()
    hook = ThreeLossRunnerHook(runner, probe)

    plan = hook.prepare_planned_step((context.token_sequence,))
    bundle = hook.compute_micro_step(
        context,
        plan,
        local_micro_step_index=0,
    )
    bundle.total_loss.backward()

    receipt = probe.validate()

    assert receipt["status"] == "candidate"
    assert receipt["plans"][0]["denominators"]["base_ce"]["eligible_segment_count"] == 3
    assert receipt["loss_checks"][0]["segment_target_counts"] == [4, 2, 2]
    assert receipt["loss_checks"][0]["role_target_counts"] == {
        "desc_text": 2,
        "schema": 1,
        "coordinate": 4,
        "eos": 1,
    }
    assert receipt["loss_checks"][0]["terms"]["raw_axis_validity_hinge"]["raw_loss"] == pytest.approx(0.0)
    assert receipt["loss_checks"][0]["total_gradient_max_delta"] == pytest.approx(0.0)


def test_global_rank_denominator_and_backend_scale_are_checked() -> None:
    context = _context(requires_grad=True)
    runner = LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.2,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
        raw_axis_validity_hinge_weight=0.01,
    )
    probe = ThreeLossQualificationProbe()
    hook = ThreeLossRunnerHook(runner, probe)

    def gather(payload):
        peer = {
            name: {
                **dict(value),
                "eligible_segment_count": 2,
                "selected_atom_count": 4,
                "skipped_segment_count": 1,
                "context_count": 1,
                **(
                    {
                        "complete_box_count": 0,
                        "incomplete_box_count": 0,
                        "zero_box_segment_count": 2,
                    }
                    if name == "raw_axis_validity_hinge"
                    else {}
                ),
            }
            for name, value in payload.items()
        }
        return (payload, peer)

    plan = hook.prepare_planned_step(
        (context.token_sequence,),
        denominator_gatherer=gather,
        world_size=2,
        rank=0,
    )
    bundle = hook.compute_micro_step(context, plan, local_micro_step_index=0)

    assert plan.denominator_scope == "planned_step_global"
    assert plan.backend_gradient_scale == 2.0
    assert bundle.term_by_name("base_ce").denominator.eligible_segment_count == 5
    assert probe.records[0]["terms"]["base_ce"]["backend_gradient_scale"] == 2.0


def test_known_geometry_violation_has_positive_hinge_and_gradient() -> None:
    context = _context(requires_grad=True)
    violated_logits = context.logits.detach().clone().requires_grad_()
    with torch.no_grad():
        for atom, predicted in zip(
            context.atoms[:4], (900, 900, 100, 100), strict=True
        ):
            violated_logits[0, atom.causal_logits_position].zero_()
            violated_logits[
                0, atom.causal_logits_position, COORDINATE_IDS[predicted]
            ] = 10.0
    violated = replace(context, logits=violated_logits)
    runner = LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.2,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
        raw_axis_validity_hinge_weight=0.01,
    )
    probe = ThreeLossQualificationProbe()
    hook = ThreeLossRunnerHook(runner, probe)
    plan = hook.prepare_planned_step((violated.token_sequence,))
    bundle = hook.compute_micro_step(violated, plan, local_micro_step_index=0)
    hinge = bundle.term_by_name("raw_axis_validity_hinge")
    gradient = torch.autograd.grad(hinge.weighted_loss, violated_logits, retain_graph=True)[0]

    assert hinge.raw_loss.item() > 0.0
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max().item() > 0.0
    assert probe.validate()["loss_checks"][0]["terms"]["raw_axis_validity_hinge"]["gradient_max_delta"] == pytest.approx(0.0)


def test_disabled_auxiliary_is_rejected() -> None:
    context = _context(requires_grad=True)
    runner = LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.0,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
        raw_axis_validity_hinge_weight=0.0,
    )
    probe = ThreeLossQualificationProbe()
    hook = ThreeLossRunnerHook(runner, probe)

    with pytest.raises(ThreeLossQualificationError, match="disabled or missing"):
        hook.prepare_planned_step((context.token_sequence,))


def test_misnormalized_term_is_rejected_by_loss_and_gradient_checks() -> None:
    context = _context(requires_grad=True)
    inner = LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.2,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
        raw_axis_validity_hinge_weight=0.01,
    )
    probe = ThreeLossQualificationProbe()

    class MisnormalizedRunner:
        def __getattr__(self, name):
            return getattr(inner, name)

        def prepare_planned_step(self, *args, **kwargs):
            return inner.prepare_planned_step(*args, **kwargs)

        def compute_micro_step(self, context, plan, *, local_micro_step_index):
            bundle = inner.compute_micro_step(
                context,
                plan,
                local_micro_step_index=local_micro_step_index,
            )
            terms = tuple(
                replace(
                    term,
                    raw_loss=term.raw_loss * 2.0,
                    weighted_loss=term.weighted_loss * 2.0,
                    segment_mean_numerator=term.segment_mean_numerator * 2.0,
                )
                if term.name == "token_type_gate"
                else term
                for term in bundle.terms
            )
            return replace(
                bundle,
                total_loss=sum((term.weighted_loss for term in terms), terms[0].weighted_loss.new_zeros(())),
                terms=terms,
            )

    hook = ThreeLossRunnerHook(MisnormalizedRunner(), probe)
    plan = hook.prepare_planned_step((context.token_sequence,))
    with pytest.raises(ThreeLossQualificationError, match="token_type_gate raw loss"):
        hook.compute_micro_step(context, plan, local_micro_step_index=0)


@pytest.mark.parametrize("kind", ["type", "geometry"])
def test_known_type_or_geometry_logit_mutation_is_rejected(kind: str) -> None:
    context = _context(requires_grad=True)
    inner = LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.2,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
        raw_axis_validity_hinge_weight=0.01,
    )
    probe = ThreeLossQualificationProbe()

    class MutatingRunner:
        def __getattr__(self, name):
            return getattr(inner, name)

        def prepare_planned_step(self, *args, **kwargs):
            return inner.prepare_planned_step(*args, **kwargs)

        def compute_micro_step(self, original, plan, *, local_micro_step_index):
            mutated = original.logits.detach().clone().requires_grad_()
            atom = original.atoms[0] if kind == "type" else original.atoms[1]
            row = atom.causal_logits_position
            with torch.no_grad():
                mutated[0, row].zero_()
                if kind == "type":
                    mutated[0, row, 1006] = 10.0
                else:
                    # Reverse the x/y ordering for a complete box, making the
                    # raw-axis hinge active while the hook's context stays intact.
                    coordinate_bins = (900, 900, 100, 100)
                    for item, predicted in zip(original.atoms[:4], coordinate_bins, strict=True):
                        mutated[0, item.causal_logits_position].zero_()
                        mutated[0, item.causal_logits_position, COORDINATE_IDS[predicted]] = 10.0
            altered = replace(original, logits=mutated)
            return inner.compute_micro_step(
                altered,
                plan,
                local_micro_step_index=local_micro_step_index,
            )

    hook = ThreeLossRunnerHook(MutatingRunner(), probe)
    plan = hook.prepare_planned_step((context.token_sequence,))
    with pytest.raises(ThreeLossQualificationError):
        hook.compute_micro_step(context, plan, local_micro_step_index=0)


def test_zero_lr_update_and_zero_hinge_are_recordable_without_nonzero_requirement() -> None:
    context = _context(requires_grad=True)
    runner = LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.2,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
        raw_axis_validity_hinge_weight=0.01,
    )
    probe = ThreeLossQualificationProbe()
    hook = ThreeLossRunnerHook(runner, probe)
    plan = hook.prepare_planned_step((context.token_sequence,))
    hook.compute_micro_step(context, plan, local_micro_step_index=0)
    update = probe.note_update(
        planned_step_id=1,
        learning_rates=(0.0, 1e-5),
        gradient_nonzero=False,
        parameter_delta_max=0.0,
    )

    assert update["zero_learning_rate"] is True
    assert probe.validate()["updates"][0]["gradient_nonzero"] is False


def _context(*, requires_grad: bool) -> Any:
    segments = (
        PackedSegment(0, 0, 0, "box", 0, 7),
        PackedSegment(0, 1, 1, "schema", 7, 11),
        PackedSegment(0, 2, 2, "text", 11, 16),
    )
    atoms = (
        *_box_atoms(),
        _atom(1, 8, 0, "schema"),
        _atom(1, 9, 1006, "eos"),
        _atom(2, 12, 1007, "desc_text"),
        _atom(2, 13, 1007, "desc_text"),
    )
    generator = torch.Generator().manual_seed(1729)
    logits = torch.randn((1, 16, VOCAB_SIZE), generator=generator)
    with torch.no_grad():
        for atom in atoms:
            logits[0, atom.causal_logits_position].zero_()
            logits[0, atom.causal_logits_position, atom.token_id] = 4.0
    if requires_grad:
        logits.requires_grad_()
    sequence = TokenSequence(
        pack_index=0,
        input_ids=tuple(0 for _ in range(16)),
        segments=segments,
        atoms=atoms,
        spans=(),
    )
    return __import__("src.losses.context", fromlist=["LossContext"]).LossContext(
        logits=logits,
        token_sequence=sequence,
        vocab_groups=GROUPS,
    )


def _box_atoms() -> tuple[TokenAtom, ...]:
    bbox = (100, 100, 900, 900)
    return tuple(
        _atom(
            0,
            1 + slot,
            COORDINATE_IDS[value],
            "coordinate",
            object_id="box-a",
            coordinate_target=CoordinateLossTarget(bbox=bbox, slot_index=slot),
            field=f"bbox[{slot}]",
        )
        for slot, value in enumerate(bbox)
    )


def _atom(
    segment_index: int,
    target_position: int,
    token_id: int,
    token_type: str,
    *,
    object_id: str | None = None,
    coordinate_target: CoordinateLossTarget | None = None,
    field: str | None = None,
) -> TokenAtom:
    return TokenAtom(
        pack_index=0,
        segment_index=segment_index,
        example_index=segment_index,
        example_id=("box", "schema", "text")[segment_index],
        target_position=target_position,
        token_id=token_id,
        token_type=token_type,
        text="x",
        logical_target_position=target_position,
        object_id=object_id,
        field=field,
        source="three-loss-test",
        coordinate_target=coordinate_target,
    )
