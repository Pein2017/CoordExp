"""Wave-2 tasks 2.3/2.4: end-to-end behaviour of the three zero policies.

Scope note for the executed autograd/sentinel probes in this module: they are
**correctness** evidence only. They prove that an omitted auxiliary term does
no term work and leaves no autograd graph, and that the zero-weight gate
ablation creates no autograd edge into the optimized objective. They are NOT
a throughput, wall-time, or peak-memory claim; no such claim may be derived
from them without a separate benchmark (change design, Non-Goals).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.common.errors import LossContractError
from src.config.models import (
    AuxiliaryLossesConfig,
    BaseCELossConfig,
    CoordGaussianRPSLossConfig,
    LossesConfig,
    ProtectedLossesConfig,
    TokenTypeGateLossConfig,
)
from src.coordinate_targets import CoordinateLossTarget
from src.losses import (
    BaseTokenCE,
    CoordGaussianRPSLoss,
    LossContext,
    LossRunner,
    PlannedStepLossSlice,
    TokenTypeGateLoss,
    TokenVocabularyGroups,
)
from src.losses.normalizers import (
    SegmentBalancedDenominator,
    segment_balanced_contribution,
)
from src.losses.runner import LossBundle, LossTermResult, _build_finite_status
from src.packing.planner import PackedSegment
from src.runtime.finite_gates import (
    RankScalarFiniteReport,
    reduce_scalar_finite_reports,
)
from src.supervision import TokenAtom, TokenSequence

CANONICAL_GROUPS = ("desc_text", "schema", "coordinate", "eos")


# --------------------------------------------------------------------------
# fixtures (self-contained; mirrors tests/losses/test_runner.py helpers)
# --------------------------------------------------------------------------


def _groups() -> TokenVocabularyGroups:
    return TokenVocabularyGroups(
        vocab_size=8,
        desc_text=(7,),
        schema=(1, 2),
        coordinate=(3, 4),
        eos=(5,),
        blocked=(0, 6),
    )


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
    coordinate_target: CoordinateLossTarget | None = None,
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
        object_id="obj-1" if coordinate_target is not None else None,
        field="bbox[0]" if coordinate_target is not None else None,
        source="unit",
        coordinate_target=coordinate_target,
    )


def _context(
    logits: torch.Tensor,
    segments: tuple[PackedSegment, ...],
    atoms: tuple[TokenAtom, ...],
) -> LossContext:
    return LossContext(
        logits=logits,
        token_sequence=TokenSequence(
            pack_index=0,
            input_ids=tuple(0 for _ in range(int(logits.shape[1]))),
            segments=segments,
            atoms=atoms,
            spans=(),
        ),
        vocab_groups=_groups(),
        logits_position_ids=None,
    )


def _projected_logits() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.tensor((((1.0, -0.5), (0.25, 2.0)),), dtype=torch.float32)
    weight = torch.linspace(-0.4, 0.6, steps=16).reshape(2, 8).requires_grad_()
    bias = torch.linspace(0.3, -0.2, steps=8).requires_grad_()
    return features @ weight + bias, weight, bias


def _text_context(logits: torch.Tensor) -> LossContext:
    return _context(
        logits,
        (_segment(0, 0, 2),),
        (_atom(segment_index=0, target_position=1, token_id=7),),
    )


def _coordinate_context(logits: torch.Tensor) -> LossContext:
    return _context(
        logits,
        (_segment(0, 0, 2),),
        (
            _atom(
                segment_index=0,
                target_position=1,
                token_id=3,
                token_type="coordinate",
                coordinate_target=CoordinateLossTarget(bbox=(2, 3, 8, 13), slot_index=0),
            ),
        ),
    )


def _losses_config(
    *,
    base_ce_weight: float = 1.0,
    gate_mode: str = "enabled",
    gate_weight: float = 0.1,
    coord_weight: float | None = None,
) -> LossesConfig:
    auxiliary = (
        None
        if coord_weight is None
        else AuxiliaryLossesConfig(
            coord_gaussian_rps=CoordGaussianRPSLossConfig(
                weight=coord_weight,
                gaussian_weight=0.5,
                rps_weight=0.2,
                temperature=1.0,
                gaussian_r95_axis_fraction=0.5,
                gaussian_r95_cap_bins=4,
                gaussian_r95_min_bins=1,
                gaussian_r95_fallback_bins=4,
            )
        )
    )
    return LossesConfig(
        normalizer="segment_balanced",
        protected=ProtectedLossesConfig(
            base_ce=BaseCELossConfig(weight=base_ce_weight),
            token_type_gate=TokenTypeGateLossConfig(
                mode=gate_mode,
                weight=gate_weight,
                groups=CANONICAL_GROUPS,
            ),
        ),
        auxiliary=auxiliary,
    )


def _runner(
    *,
    base_ce_weight: float = 1.0,
    token_type_gate_weight: float = 0.1,
    coord_gaussian_rps_weight: float = 0.0,
    coord_gaussian_rps: CoordGaussianRPSLoss | None = None,
) -> LossRunner:
    return LossRunner(
        base_ce_weight=base_ce_weight,
        token_type_gate_weight=token_type_gate_weight,
        token_type_gate_groups=CANONICAL_GROUPS,
        coord_gaussian_rps_weight=coord_gaussian_rps_weight,
        coord_gaussian_rps=coord_gaussian_rps,
    )


def _reference_raw(
    context: LossContext,
    *,
    term_name: str,
    term: Any,
    denominator: Any,
    backend_gradient_scale: float = 1.0,
) -> torch.Tensor:
    raw = segment_balanced_contribution(
        PlannedStepLossSlice(
            term_name=term_name,
            context=context,
            per_atom_losses=term.per_atom_loss(context),
            local_micro_step_index=0,
        ),
        denominator=denominator,
    )
    return raw * float(backend_gradient_scale)


def _graph_node_names(tensor: torch.Tensor) -> list[str]:
    names: list[str] = []
    seen: set[int] = set()
    stack = [tensor.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        names.append(type(node).__name__)
        for next_node, _ in getattr(node, "next_functions", ()):
            stack.append(next_node)
    return sorted(names)


# --------------------------------------------------------------------------
# base_ce zero policy: `forbid`
# --------------------------------------------------------------------------


@pytest.mark.parametrize("weight", (0.0, 0.5, 1.3, 2.0))
def test_base_ce_forbid_policy_fails_closed_at_composition(weight: float) -> None:
    """`forbid` is enforced at composition, not only by strict config.

    Strict config already rejects any base-CE weight other than 1.0; this is
    the runtime backstop for any path that reaches loss construction with a
    reweighted or zeroed protected base CE.
    """

    with pytest.raises(LossContractError) as excinfo:
        _runner(base_ce_weight=weight)
    assert excinfo.value.code == "loss.base_ce_weight_forbidden"
    assert excinfo.value.context["zero_policy"] == "forbid"


def test_base_ce_weight_one_composes() -> None:
    runner = _runner(base_ce_weight=1.0)
    assert [binding.name for binding in runner.active_bindings()] == [
        "base_ce",
        "token_type_gate",
    ]


# --------------------------------------------------------------------------
# token_type_gate zero policy: `detached_diagnostic` (entry-audit F-2)
# --------------------------------------------------------------------------


def _hand_built_bundle(
    *,
    gate_raw: float,
    gate_weighted: float,
) -> LossBundle:
    """A bundle shaped exactly like a zero-weight gate ablation micro-step.

    `weighted_loss` is the literal zero the ablation emits; `raw_loss` carries
    the real protected diagnostic. Keying the finite decision on `weighted`
    would silently report this bundle safe (entry-audit F-2).
    """

    denominator = SegmentBalancedDenominator(
        term_name="token_type_gate",
        denominator_scope="planned_step",
        eligible_segment_count=1,
        selected_atom_count=1,
        skipped_segment_count=0,
        context_count=1,
    )
    base = LossTermResult(
        name="base_ce",
        raw_loss=torch.tensor(1.5),
        weighted_loss=torch.tensor(1.5),
        weight=1.0,
        segment_mean_numerator=torch.tensor(1.5),
        denominator=SegmentBalancedDenominator(
            term_name="base_ce",
            denominator_scope="planned_step",
            eligible_segment_count=1,
            selected_atom_count=1,
            skipped_segment_count=0,
            context_count=1,
        ),
        reducer_name="segment_balanced",
        selected_count=1,
        skipped_count=0,
        math_dtype="float32",
        token_weighted_diagnostic=torch.tensor(1.5),
        diagnostics={},
    )
    gate = LossTermResult(
        name="token_type_gate",
        raw_loss=torch.tensor(gate_raw),
        weighted_loss=torch.tensor(gate_weighted),
        weight=0.0,
        segment_mean_numerator=torch.tensor(gate_raw),
        denominator=denominator,
        reducer_name="segment_balanced",
        selected_count=1,
        skipped_count=0,
        math_dtype="float32",
        token_weighted_diagnostic=torch.tensor(0.0),
        diagnostics={},
    )
    return LossBundle(
        total_loss=torch.tensor(1.5),
        terms=(base, gate),
        metrics={},
        counts={},
        diagnostics={},
        finite_status={},
        accuracy_stats={"top1_correct": 1, "top5_correct": 1, "atom_count": 1},
    )


def test_detached_diagnostic_finite_label_derives_from_raw_not_weighted_zero() -> None:
    """Entry-audit F-2 binding guard.

    A non-finite RAW protected gate diagnostic whose WEIGHTED value is the
    literal zero of the ablation MUST still be labelled non-finite, and MUST
    still make the all-rank pre-backward scalar gate refuse backward.
    """

    unsafe = _hand_built_bundle(gate_raw=float("nan"), gate_weighted=0.0)

    finite_status = _build_finite_status(
        total_loss=unsafe.total_loss, terms=unsafe.terms
    )
    assert finite_status["terms"]["base_ce"] == "finite"
    assert finite_status["terms"]["token_type_gate"] == "non_finite"

    report = RankScalarFiniteReport.from_loss_bundle(
        unsafe, planned_step_id=7, rank=0, world_size=1
    )
    assert report.term_finite["token_type_gate"] is False
    assert report.term_raw_losses["token_type_gate"] is None
    assert report.is_safe() is False

    decision = reduce_scalar_finite_reports((report,))
    assert decision.all_ranks_safe is False
    assert decision.should_call_backward is False
    assert decision.finite_status == "non_finite"
    assert decision.optimizer_update_status == "skipped_non_finite_scalar"


def test_detached_diagnostic_finite_raw_stays_safe() -> None:
    safe = _hand_built_bundle(gate_raw=3.25, gate_weighted=0.0)
    finite_status = _build_finite_status(total_loss=safe.total_loss, terms=safe.terms)
    assert finite_status["terms"]["token_type_gate"] == "finite"
    report = RankScalarFiniteReport.from_loss_bundle(
        safe, planned_step_id=7, rank=0, world_size=1
    )
    assert report.is_safe() is True
    assert reduce_scalar_finite_reports((report,)).should_call_backward is True


def test_gate_ablation_emits_exact_zero_weighted_and_keeps_raw_diagnostic() -> None:
    logits, _weight, _bias = _projected_logits()
    context = _text_context(logits)
    runner = _runner(token_type_gate_weight=0.0)
    plan = runner.prepare_planned_step((context.token_sequence,))
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)

    gate = bundle.term_by_name("token_type_gate")
    assert float(gate.weighted_loss) == 0.0
    assert gate.weighted_loss.requires_grad is False
    assert gate.weighted_loss.grad_fn is None
    assert gate.raw_loss.requires_grad is False
    assert gate.raw_loss.grad_fn is None
    assert float(gate.raw_loss) > 0.0
    assert gate.selected_count == 1
    assert bundle.finite_status["terms"]["token_type_gate"] == "finite"
    assert bundle.metrics["finite/token_type_gate"] == 1.0
    assert bundle.metrics["loss/token_type_gate/segment_count"] == 1.0


def test_gate_ablation_non_finite_raw_survives_the_literal_zero_weighted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entry-audit F-2 end to end through the real composition path."""

    def _nan_per_atom(self: TokenTypeGateLoss, context: LossContext) -> torch.Tensor:
        _logits_fp32, _target_ids, atoms = context.select_logits_fp32()
        return torch.full((len(atoms),), float("nan"), dtype=torch.float32)

    monkeypatch.setattr(TokenTypeGateLoss, "per_atom_loss", _nan_per_atom)

    logits, _weight, _bias = _projected_logits()
    context = _text_context(logits)
    runner = _runner(token_type_gate_weight=0.0)
    plan = runner.prepare_planned_step((context.token_sequence,))
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)

    gate = bundle.term_by_name("token_type_gate")
    assert float(gate.weighted_loss) == 0.0
    assert torch.isnan(gate.raw_loss).item() is True
    assert bool(torch.isfinite(bundle.total_loss).all().item()) is True
    assert bundle.finite_status["terms"]["token_type_gate"] == "non_finite"
    assert bundle.metrics["finite/token_type_gate"] == 0.0

    report = RankScalarFiniteReport.from_loss_bundle(
        bundle, planned_step_id=3, rank=0, world_size=1
    )
    assert report.is_safe() is False
    assert reduce_scalar_finite_reports((report,)).should_call_backward is False

    finalized = runner.finalize_planned_step((bundle.to_artifact_dict(),), plan)
    assert finalized["finite_status"]["terms"]["token_type_gate"] == "non_finite"
    assert finalized["metrics"]["finite/token_type_gate"] == 0.0


def test_gate_ablation_creates_no_autograd_edge_into_the_objective() -> None:
    """Executed correctness probe (task 2.4) - NOT an efficiency claim.

    Walks the optimized objective's autograd graph and proves no gate
    operation (`logsumexp`, the gate's characteristic op) is reachable from
    `total_loss`, and that the parameter gradients are bit-identical to an
    independently computed base-CE-only reference.
    """

    logits, weight, bias = _projected_logits()
    context = _text_context(logits)
    runner = _runner(token_type_gate_weight=0.0)
    plan = runner.prepare_planned_step((context.token_sequence,))
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)

    reference_weighted = (
        _reference_raw(
            context,
            term_name="base_ce",
            term=BaseTokenCE(),
            denominator=plan.denominators["base_ce"],
            backend_gradient_scale=plan.backend_gradient_scale,
        )
        * 1.0
    )
    # Mirrors the runner's objective accumulation (`sum(..., start=zeros)`)
    # so the comparison isolates term participation, not accumulation shape.
    reference_total = reference_weighted.new_zeros(()) + reference_weighted

    objective_nodes = _graph_node_names(bundle.total_loss)
    assert objective_nodes, "objective must retain a differentiable graph"
    assert not any("logsumexp" in name.lower() for name in objective_nodes)
    assert objective_nodes == _graph_node_names(reference_total)

    observed = torch.autograd.grad(bundle.total_loss, (weight, bias), retain_graph=True)
    expected = torch.autograd.grad(reference_total, (weight, bias))
    for observed_grad, expected_grad in zip(observed, expected, strict=True):
        assert torch.equal(observed_grad, expected_grad)


# --------------------------------------------------------------------------
# coord_gaussian_rps zero policy: `omit`
# --------------------------------------------------------------------------


def test_omitted_auxiliary_rejects_a_constructed_term_instance() -> None:
    """`omit` means "never instantiated": a term object at weight 0 is a bug."""

    with pytest.raises(LossContractError) as excinfo:
        _runner(
            coord_gaussian_rps_weight=0.0,
            coord_gaussian_rps=CoordGaussianRPSLoss(
                gaussian_weight=0.5,
                rps_weight=0.2,
                temperature=1.0,
                gaussian_r95_axis_fraction=0.5,
                gaussian_r95_cap_bins=4,
                gaussian_r95_min_bins=1,
                gaussian_r95_fallback_bins=4,
            ),
        )
    assert excinfo.value.code == "loss.auxiliary_omitted_term_instantiated"


def test_omitted_auxiliary_is_never_constructed_or_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executed correctness probe (task 2.4) - NOT an efficiency claim.

    Construction and call sentinels prove the omitted auxiliary does zero
    term work: the loss class is never instantiated by the composition path
    and its per-atom math is never invoked.
    """

    constructions = 0
    calls = 0

    class _SentinelCoordLoss(CoordGaussianRPSLoss):
        def __init__(self, **kwargs: Any) -> None:
            nonlocal constructions
            constructions += 1
            super().__init__(**kwargs)

        def per_atom_loss(self, context: LossContext) -> torch.Tensor:
            nonlocal calls
            calls += 1
            return super().per_atom_loss(context)

    monkeypatch.setattr("src.losses.runner.CoordGaussianRPSLoss", _SentinelCoordLoss)

    logits, _weight, _bias = _projected_logits()
    context = _coordinate_context(logits)
    runner = LossRunner.from_config(_losses_config(coord_weight=0.0))
    plan = runner.prepare_planned_step((context.token_sequence,))
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)
    runner.finalize_planned_step((bundle.to_artifact_dict(),), plan)

    assert constructions == 0
    assert calls == 0
    assert runner.coord_gaussian_rps is None


def test_omitted_auxiliary_builds_and_gathers_no_denominator() -> None:
    logits, _weight, _bias = _projected_logits()
    context = _coordinate_context(logits)
    runner = LossRunner.from_config(_losses_config(coord_weight=0.0))

    gathered_payloads: list[dict[str, Any]] = []

    def _gatherer(local_payload: Any) -> tuple[Any, ...]:
        gathered_payloads.append(dict(local_payload))
        return (dict(local_payload), dict(local_payload))

    plan = runner.prepare_planned_step(
        (context.token_sequence,),
        denominator_gatherer=_gatherer,
        world_size=2,
        rank=0,
    )
    assert set(plan.denominators) == {"base_ce", "token_type_gate"}
    assert gathered_payloads
    for payload in gathered_payloads:
        assert "coord_gaussian_rps" not in payload


def test_omitted_auxiliary_never_triggers_a_zero_eligible_failure() -> None:
    """Spec scenario: "Disabled auxiliary has no denominator"."""

    logits, _weight, _bias = _projected_logits()
    context = _text_context(logits)  # no coordinate atoms at all

    omitted = LossRunner.from_config(_losses_config(coord_weight=0.0))
    plan = omitted.prepare_planned_step((context.token_sequence,))
    assert "coord_gaussian_rps" not in plan.denominators

    enabled = LossRunner.from_config(_losses_config(coord_weight=1.0))
    with pytest.raises(LossContractError) as excinfo:
        enabled.prepare_planned_step((context.token_sequence,))
    assert excinfo.value.code == "loss.segment_balanced_zero_eligible"
    assert excinfo.value.context["term"] == "coord_gaussian_rps"


def test_omitted_auxiliary_emits_no_bundle_or_metric_fields() -> None:
    logits, _weight, _bias = _projected_logits()
    context = _coordinate_context(logits)
    runner = LossRunner.from_config(_losses_config(coord_weight=0.0))
    plan = runner.prepare_planned_step((context.token_sequence,))
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)
    finalized = runner.finalize_planned_step((bundle.to_artifact_dict(),), plan)

    assert [term.name for term in bundle.terms] == ["base_ce", "token_type_gate"]
    assert "coord_gaussian_rps" not in bundle.finite_status["terms"]
    assert [term["name"] for term in finalized["terms"]] == [
        "base_ce",
        "token_type_gate",
    ]
    assert "coord_gaussian_rps" not in finalized["finite_status"]["terms"]
    for scalars in (bundle.metrics, finalized["metrics"]):
        assert not [key for key in scalars if "coord_gaussian_rps" in key]
    assert not [
        key
        for key in finalized["diagnostics"]["term_denominators"]
        if "coord_gaussian_rps" in key
    ]
    assert finalized["diagnostics"]["term_order"] == ["base_ce", "token_type_gate"]

    report = RankScalarFiniteReport.from_loss_bundle(
        bundle, planned_step_id=1, rank=0, world_size=1
    )
    assert "coord_gaussian_rps" not in report.term_finite
    assert "coord_gaussian_rps" not in report.to_diagnostic_dict()["terms"]


def test_omitted_auxiliary_retains_no_autograd_graph_or_saved_tensors() -> None:
    """Executed correctness probe (task 2.4) - NOT an efficiency claim.

    Compares the saved-tensor signature and objective autograd graph of a
    weight-zero auxiliary against a config where the auxiliary block is
    absent entirely. Identical signatures prove the omitted term retains no
    graph; this says nothing about wall time or peak memory.
    """

    def _run(config: LossesConfig) -> tuple[list[tuple[Any, ...]], list[str]]:
        signatures: list[tuple[Any, ...]] = []

        def _pack(tensor: torch.Tensor) -> torch.Tensor:
            signatures.append(
                (tuple(tensor.shape), str(tensor.dtype), bool(tensor.requires_grad))
            )
            return tensor

        def _unpack(tensor: torch.Tensor) -> torch.Tensor:
            return tensor

        logits, _weight, _bias = _projected_logits()
        context = _coordinate_context(logits)
        runner = LossRunner.from_config(config)
        plan = runner.prepare_planned_step((context.token_sequence,))
        with torch.autograd.graph.saved_tensors_hooks(_pack, _unpack):
            bundle = runner.compute_micro_step(
                context, plan, local_micro_step_index=0
            )
        return sorted(signatures), _graph_node_names(bundle.total_loss)

    absent_signatures, absent_nodes = _run(_losses_config(coord_weight=None))
    omitted_signatures, omitted_nodes = _run(_losses_config(coord_weight=0.0))

    assert omitted_signatures == absent_signatures
    assert omitted_nodes == absent_nodes
    assert not any("gaussian" in name.lower() for name in omitted_nodes)


def test_enabled_auxiliary_still_participates_in_the_objective() -> None:
    logits, weight, bias = _projected_logits()
    context = _coordinate_context(logits)
    runner = LossRunner.from_config(_losses_config(coord_weight=1.0))
    assert [binding.name for binding in runner.active_bindings()] == [
        "base_ce",
        "token_type_gate",
        "coord_gaussian_rps",
    ]
    plan = runner.prepare_planned_step((context.token_sequence,))
    assert "coord_gaussian_rps" in plan.denominators
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)
    coord = bundle.term_by_name("coord_gaussian_rps")
    assert coord.weighted_loss.requires_grad is True
    assert float(coord.raw_loss.detach()) > 0.0
    gradients = torch.autograd.grad(bundle.total_loss, (weight, bias))
    assert all(bool(torch.isfinite(grad).all().item()) for grad in gradients)
