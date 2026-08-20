"""Wave-3 task 3.1: semantic value vs backward contribution separation.

These fixtures distinguish three per-term concepts that the pre-Wave-3
implementation collapsed into one tensor:

1. **raw semantic value** - the globally normalized planned-step
   `segment_balanced` term value *before* configured weighting, carrying no
   backend compensation whatsoever;
2. **weighted semantic value** - `raw x configured weight`;
3. **differentiable local backward contribution** - rank-local numerator over
   the *global* denominator, weighted, and multiplied **exactly once** by the
   world size to compensate the Accelerate/DDP mean-gradient reduction.

Only (3) may participate in backward, and (1)/(2) must never inherit the
backend factor (design decision 4; spec scenario "Backend compensation is not
telemetry"). The ratio assertions below fail against an implementation that
applies the compensation twice (ratio `world_size ** 2`) or zero times
(ratio `1.0`), and against any implementation where telemetry inherits it.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.losses import (
    BaseTokenCE,
    LossContext,
    LossRunner,
    PlannedStepLossSlice,
    TokenTypeGateLoss,
    TokenVocabularyGroups,
)
from src.losses.normalizers import segment_balanced_contribution
from src.packing.planner import PackedSegment
from src.supervision import TokenAtom, TokenSequence


_GATE_WEIGHT = 0.1
_WORLD_SIZE = 2


def test_world_size_one_semantic_values_equal_the_local_backward_contribution() -> None:
    """At world size one the three concepts coincide exactly.

    This is the anchor the distributed fixtures below are compared against:
    the separation must not perturb single-rank semantics at all.
    """

    context = _context_a()
    runner = _runner()
    plan = runner.prepare_planned_step((context.token_sequence,))
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)

    assert plan.backend_gradient_scale == 1.0
    base = bundle.term_by_name("base_ce")
    gate = bundle.term_by_name("token_type_gate")
    base_reference = _reference_raw(context, "base_ce", BaseTokenCE(), plan)
    gate_reference = _reference_raw(context, "token_type_gate", TokenTypeGateLoss(), plan)

    assert torch.equal(base.raw_loss, base_reference)
    assert torch.equal(gate.raw_loss, gate_reference)
    assert torch.equal(base.weighted_loss, base_reference * 1.0)
    assert torch.equal(gate.weighted_loss, gate_reference * _GATE_WEIGHT)
    # (3) == (2) exactly when there is nothing to compensate.
    assert torch.equal(base.backward_contribution, base.weighted_loss)
    assert torch.equal(gate.backward_contribution, gate.weighted_loss)
    assert base.backend_gradient_scale == 1.0
    assert torch.equal(bundle.backward_loss, bundle.total_loss)


def test_unequal_rank_segment_counts_keep_backend_scale_out_of_semantic_values() -> None:
    """Unequal rank-local eligible-segment counts, world size two.

    Rank 0 contributes 1 eligible segment; the peer contributes 3. The global
    `segment_balanced` denominator is therefore 4 for every term, and this
    rank's semantic contribution is `local_numerator / 4` with **no** world
    size factor. Only the backward contribution carries the factor.
    """

    context = _context_a()
    runner = _runner()
    plan = runner.prepare_planned_step(
        (context.token_sequence,),
        denominator_gatherer=_peer_gatherer(
            eligible_segment_count=3,
            selected_atom_count=3,
            context_count=3,
        ),
        world_size=_WORLD_SIZE,
        rank=0,
    )
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)

    assert plan.denominator_scope == "planned_step_global"
    assert plan.backend_gradient_scale == float(_WORLD_SIZE)
    for name in ("base_ce", "token_type_gate"):
        assert plan.denominators[name].eligible_segment_count == 4

    base = bundle.term_by_name("base_ce")
    gate = bundle.term_by_name("token_type_gate")
    base_reference = _reference_raw(context, "base_ce", BaseTokenCE(), plan)
    gate_reference = _reference_raw(context, "token_type_gate", TokenTypeGateLoss(), plan)

    # (1) raw semantic value: local numerator / GLOBAL denominator, unscaled.
    assert torch.equal(base.raw_loss, base_reference)
    assert torch.equal(gate.raw_loss, gate_reference)
    # (2) weighted semantic value: raw x configured weight, still unscaled.
    assert torch.equal(base.weighted_loss, base_reference * 1.0)
    assert torch.equal(gate.weighted_loss, gate_reference * _GATE_WEIGHT)
    # (3) differentiable local contribution: weighted x world size, once.
    assert torch.equal(
        base.backward_contribution, (base_reference * 1.0) * float(_WORLD_SIZE)
    )
    assert torch.equal(
        gate.backward_contribution,
        (gate_reference * _GATE_WEIGHT) * float(_WORLD_SIZE),
    )
    assert base.backend_gradient_scale == float(_WORLD_SIZE)


def test_backend_compensation_is_applied_exactly_once_never_twice_or_zero() -> None:
    """Ratio proof, independent of the term's numeric value.

    `backward / weighted` must be exactly the world size. A double-scaled
    implementation yields `world_size ** 2`; an unscaled one yields `1.0`.
    A telemetry-inheriting implementation moves the factor into
    `weighted / raw`, which must stay exactly the configured weight.
    """

    context = _context_a()
    runner = _runner()
    plan = runner.prepare_planned_step(
        (context.token_sequence,),
        denominator_gatherer=_peer_gatherer(
            eligible_segment_count=3,
            selected_atom_count=3,
            context_count=3,
        ),
        world_size=_WORLD_SIZE,
        rank=0,
    )
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)

    base = bundle.term_by_name("base_ce")
    raw = float(base.raw_loss.detach())
    weighted = float(base.weighted_loss.detach())
    backward = float(base.backward_contribution.detach())
    assert raw > 0.0

    assert weighted / raw == pytest.approx(1.0)  # configured base-CE weight
    assert backward / weighted == pytest.approx(float(_WORLD_SIZE))
    assert backward / weighted != pytest.approx(1.0)  # zero compensations
    assert backward / weighted != pytest.approx(float(_WORLD_SIZE) ** 2)  # twice

    gate = bundle.term_by_name("token_type_gate")
    gate_raw = float(gate.raw_loss.detach())
    assert float(gate.weighted_loss.detach()) / gate_raw == pytest.approx(_GATE_WEIGHT)
    assert float(gate.backward_contribution.detach()) / gate_raw == pytest.approx(
        _GATE_WEIGHT * float(_WORLD_SIZE)
    )


def test_bundle_total_loss_is_semantic_while_backward_loss_carries_the_scale() -> None:
    """`total_loss`/`metrics` are telemetry; `backward_loss` is the objective."""

    context = _context_a()
    runner = _runner()
    plan = runner.prepare_planned_step(
        (context.token_sequence,),
        denominator_gatherer=_peer_gatherer(
            eligible_segment_count=3,
            selected_atom_count=3,
            context_count=3,
        ),
        world_size=_WORLD_SIZE,
        rank=0,
    )
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)

    base = bundle.term_by_name("base_ce")
    gate = bundle.term_by_name("token_type_gate")
    semantic_total = float(base.weighted_loss.detach()) + float(
        gate.weighted_loss.detach()
    )

    assert float(bundle.total_loss.detach()) == pytest.approx(semantic_total)
    assert float(bundle.backward_loss.detach()) == pytest.approx(
        semantic_total * float(_WORLD_SIZE)
    )
    assert bundle.metrics["loss/total"] == pytest.approx(semantic_total)
    assert bundle.metrics["loss/base_ce/weighted"] == pytest.approx(
        float(base.weighted_loss.detach())
    )
    assert bundle.metrics["loss/token_type_gate/weighted"] == pytest.approx(
        float(gate.weighted_loss.detach())
    )

    artifact = bundle.to_artifact_dict()
    assert artifact["total_loss"] == pytest.approx(semantic_total)
    assert artifact["backward_loss"] == pytest.approx(
        semantic_total * float(_WORLD_SIZE)
    )
    base_artifact = next(
        term for term in artifact["terms"] if term["name"] == "base_ce"
    )
    assert base_artifact["backend_gradient_scale"] == float(_WORLD_SIZE)
    assert base_artifact["raw_loss"] == pytest.approx(float(base.raw_loss.detach()))
    assert base_artifact["backward_contribution"] == pytest.approx(
        float(base.raw_loss.detach()) * float(_WORLD_SIZE)
    )


def test_only_the_backward_contribution_participates_in_backward() -> None:
    """The compensated tensor is the one whose gradient is world-size scaled."""

    logits = _logits(
        (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0),
            (0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0),
        ),
        requires_grad=True,
    )
    context = _context(
        logits,
        (_segment(0, 0, 2),),
        (_atom(segment_index=0, target_position=1, token_id=7),),
    )
    runner = _runner()
    plan = runner.prepare_planned_step(
        (context.token_sequence,),
        denominator_gatherer=_peer_gatherer(
            eligible_segment_count=3,
            selected_atom_count=3,
            context_count=3,
        ),
        world_size=_WORLD_SIZE,
        rank=0,
    )
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)

    semantic_grad = torch.autograd.grad(
        bundle.total_loss, logits, retain_graph=True
    )[0]
    backward_grad = torch.autograd.grad(
        bundle.backward_loss, logits, retain_graph=True
    )[0]
    assert torch.allclose(backward_grad, semantic_grad * float(_WORLD_SIZE))
    assert not torch.allclose(backward_grad, semantic_grad)


def test_finalized_planned_step_raw_and_weighted_exclude_the_backend_scale() -> None:
    """Finalization aggregates unscaled rank-local semantic contributions."""

    contexts = (_context_a(), _context_b())
    runner = _runner()
    plan = runner.prepare_planned_step(
        tuple(context.token_sequence for context in contexts),
        denominator_gatherer=_peer_gatherer(
            eligible_segment_count=3,
            selected_atom_count=3,
            context_count=3,
        ),
        world_size=_WORLD_SIZE,
        rank=0,
    )
    micro_bundles = tuple(
        runner.compute_micro_step(context, plan, local_micro_step_index=index)
        for index, context in enumerate(contexts)
    )
    finalized = runner.finalize_planned_step(
        tuple(bundle.to_artifact_dict() for bundle in micro_bundles), plan
    )

    expected_raw = {
        name: sum(
            float(bundle.term_by_name(name).raw_loss.detach())
            for bundle in micro_bundles
        )
        for name in ("base_ce", "token_type_gate")
    }
    expected_weighted = {
        "base_ce": expected_raw["base_ce"] * 1.0,
        "token_type_gate": expected_raw["token_type_gate"] * _GATE_WEIGHT,
    }
    for term in finalized["terms"]:
        name = str(term["name"])
        assert term["raw_loss"] == pytest.approx(expected_raw[name])
        assert term["weighted_loss"] == pytest.approx(expected_weighted[name])
        assert term["backward_contribution"] == pytest.approx(
            expected_weighted[name] * float(_WORLD_SIZE)
        )
    semantic_total = sum(expected_weighted.values())
    assert finalized["total_loss"] == pytest.approx(semantic_total)
    assert finalized["metrics"]["loss/total"] == pytest.approx(semantic_total)
    assert finalized["metrics"]["loss/base_ce/weighted"] == pytest.approx(
        expected_weighted["base_ce"]
    )


def test_zero_weight_gate_backward_contribution_is_exact_zero_not_scaled_zero() -> None:
    """The detached diagnostic never acquires a compensated tensor."""

    context = _context_a()
    runner = LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.0,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
    )
    plan = runner.prepare_planned_step(
        (context.token_sequence,),
        denominator_gatherer=_peer_gatherer(
            eligible_segment_count=3,
            selected_atom_count=3,
            context_count=3,
        ),
        world_size=_WORLD_SIZE,
        rank=0,
    )
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)

    gate = bundle.term_by_name("token_type_gate")
    base = bundle.term_by_name("base_ce")
    assert torch.equal(gate.weighted_loss, gate.raw_loss.new_zeros(()))
    assert torch.equal(gate.backward_contribution, gate.raw_loss.new_zeros(()))
    assert not gate.backward_contribution.requires_grad
    assert gate.backward_contribution.grad_fn is None
    # The raw diagnostic still carries the honest unscaled semantic value.
    gate_reference = _reference_raw(context, "token_type_gate", TokenTypeGateLoss(), plan)
    assert torch.equal(gate.raw_loss, gate_reference.detach())
    # The objective is exactly base CE, compensated once.
    assert torch.equal(bundle.total_loss, base.weighted_loss)
    assert torch.equal(bundle.backward_loss, base.backward_contribution)


def _reference_raw(
    context: LossContext,
    term_name: str,
    term: Any,
    plan: Any,
) -> torch.Tensor:
    """Independent unscaled reference: local numerator / global denominator."""

    # Every atom used here belongs to a configured gate group, so the runner's
    # token-type filtering is a no-op and the unfiltered context is the exact
    # reference for both terms.
    return segment_balanced_contribution(
        PlannedStepLossSlice(
            term_name=term_name,
            context=context,
            per_atom_losses=term.per_atom_loss(context),
            local_micro_step_index=0,
        ),
        denominator=plan.denominators[term_name],
    )


def _peer_gatherer(
    *,
    eligible_segment_count: int,
    selected_atom_count: int,
    context_count: int,
) -> Any:
    def gatherer(local_payload: Any) -> tuple[Any, Any]:
        peer = {
            name: {
                **dict(payload),
                "eligible_segment_count": eligible_segment_count,
                "selected_atom_count": selected_atom_count,
                "skipped_segment_count": 0,
                "context_count": context_count,
            }
            for name, payload in local_payload.items()
        }
        return (local_payload, peer)

    return gatherer


def _runner() -> LossRunner:
    return LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=_GATE_WEIGHT,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
    )


def _context_a() -> LossContext:
    return _context(
        _logits(
            (
                (0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0),
            )
        ),
        (_segment(0, 0, 2),),
        (_atom(segment_index=0, target_position=1, token_id=7),),
    )


def _context_b() -> LossContext:
    return _context(
        _logits(
            (
                (0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0, 1.0, 6.0, 0.0, 0.0, 2.0),
            )
        ),
        (_segment(0, 0, 2),),
        (
            _atom(
                segment_index=0,
                target_position=1,
                token_id=3,
                token_type="coordinate",
            ),
        ),
        pack_index=1,
    )


def _context(
    logits: torch.Tensor,
    segments: tuple[PackedSegment, ...],
    atoms: tuple[TokenAtom, ...],
    *,
    pack_index: int = 0,
) -> LossContext:
    pack_length = int(logits.shape[1])
    return LossContext(
        logits=logits,
        token_sequence=TokenSequence(
            pack_index=pack_index,
            input_ids=tuple(0 for _ in range(pack_length)),
            segments=tuple(
                PackedSegment(
                    pack_index=pack_index,
                    segment_index=segment.segment_index,
                    example_index=segment.example_index,
                    example_id=segment.example_id,
                    start=segment.start,
                    end=segment.end,
                )
                for segment in segments
            ),
            atoms=tuple(
                TokenAtom(
                    pack_index=pack_index,
                    segment_index=atom.segment_index,
                    example_index=atom.example_index,
                    example_id=atom.example_id,
                    target_position=atom.target_position,
                    token_id=atom.token_id,
                    token_type=atom.token_type,
                    text=atom.text,
                    logical_target_position=atom.logical_target_position,
                    object_id=atom.object_id,
                    field=atom.field,
                    source=atom.source,
                    coordinate_target=atom.coordinate_target,
                )
                for atom in atoms
            ),
            spans=(),
        ),
        vocab_groups=_groups(),
    )


def _logits(
    rows: tuple[tuple[float, ...], ...],
    *,
    requires_grad: bool = False,
) -> torch.Tensor:
    logits = torch.tensor((rows,), dtype=torch.float32)
    if requires_grad:
        logits.requires_grad_()
    return logits


def _segment(segment_index: int, start: int, end: int) -> PackedSegment:
    return PackedSegment(
        pack_index=0,
        segment_index=segment_index,
        example_index=segment_index,
        example_id=f"ex-{segment_index}",
        start=start,
        end=end,
    )


def _atom(
    *,
    segment_index: int,
    target_position: int,
    token_id: int,
    token_type: str = "desc_text",
) -> TokenAtom:
    return TokenAtom(
        pack_index=0,
        segment_index=segment_index,
        example_index=segment_index,
        example_id=f"ex-{segment_index}",
        target_position=target_position,
        token_id=token_id,
        token_type=token_type,
        text="x",
        logical_target_position=target_position,
        object_id=None,
        field=None,
        source="unit",
        coordinate_target=None,
    )


def _groups() -> TokenVocabularyGroups:
    return TokenVocabularyGroups(
        vocab_size=8,
        desc_text=(7,),
        schema=(1, 2),
        coordinate=(3, 4),
        eos=(5,),
        blocked=(0, 6),
    )
