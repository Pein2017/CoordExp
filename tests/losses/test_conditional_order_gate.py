from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.common.errors import LossContractError
from src.coordinate_targets import CoordinateLossTarget, CoordinateTargetContractError
from src.losses.conditional_order_gate import ConditionalOrderGateLoss
from src.losses.context import LossContext
from src.losses.runner import LossRunner
from src.losses.vocab import TokenVocabularyGroups
from src.packing.planner import PackedSegment
from src.supervision import TokenAtom, TokenSequence


IDS = tuple(range(5, 1005))
VOCAB = TokenVocabularyGroups(
    vocab_size=1008, desc_text=(1007,), schema=(0, 1),
    coordinate=IDS, eos=(1006,), blocked=(2, 3, 4, 1005),
)


def _context(boxes=((600, 200, 601, 201),), *, extra_segment=False):
    atoms = []
    ids = []
    segments = []
    for segment_index, bbox in enumerate(boxes):
        ids.append(0)
        start = len(ids)
        ids.extend(IDS[value] for value in bbox)
        segments.append(PackedSegment(0, segment_index, segment_index, str(segment_index), start - 1, start + 4))
        for slot, value in enumerate(bbox):
            atoms.append(TokenAtom(
                pack_index=0, segment_index=segment_index, example_index=segment_index,
                example_id=str(segment_index), target_position=start + slot,
                token_id=IDS[value], token_type="coordinate", text=f"<|coord_{value}|>",
                logical_target_position=start + slot, object_id="box", field=f"bbox[{slot}]",
                source="test", coordinate_target=CoordinateLossTarget(bbox, slot),
            ))
    if extra_segment:
        start = len(ids)
        ids.extend((0, 1007))
        segments.append(PackedSegment(0, len(segments), len(segments), "desc", start, start + 2))
        atoms.append(TokenAtom(
            pack_index=0, segment_index=len(segments)-1, example_index=len(segments)-1,
            example_id="desc", target_position=start + 1, token_id=1007,
            token_type="desc_text", text="thing", logical_target_position=start+1,
        ))
    sequence = TokenSequence(0, tuple(ids), tuple(segments), tuple(atoms), ())
    logits = torch.zeros((1, len(ids), 1008), requires_grad=True)
    return LossContext(logits, sequence, VOCAB)


def test_all_illegal_bins_and_equality_receive_corrective_gradient():
    context = _context()
    with torch.no_grad():
        # Mean is legal even though the highest logit is an illegal 600.
        context.logits[0, 2, IDS[600]] = 3.0
        context.logits[0, 2, IDS[999]] = 2.9
        context.logits[0, 3, IDS[999]] = 30.0
    result = ConditionalOrderGateLoss().per_segment_loss(context)
    assert result.segment_losses.item() > 0
    assert result.participating_slot_count == 2
    result.segment_losses.sum().backward()
    grad = context.logits.grad[0, 2]
    assert torch.all(grad[list(IDS[:601])] > 0)
    assert torch.all(grad[list(IDS[601:])] < 0)
    assert grad[0].item() == 0
    raised = context.logits.detach().clone()
    raised[0, 2, IDS[600]] += 1
    assert ConditionalOrderGateLoss().per_segment_loss(
        replace(context, logits=raised)
    ).segment_losses.item() > result.segment_losses.item()


@pytest.mark.parametrize("a,valid_count", [(0, 999), (600, 399), (998, 1)])
def test_prefix_threshold_exact_and_sharp_legal_box(a, valid_count):
    context = _context(((a, a, a + 1, a + 1),))
    logits = context.logits.detach().clone()
    with torch.no_grad():
        logits[0, 2, IDS[a + 1]] = 30
        logits[0, 3, IDS[a + 1]] = 30
    result = ConditionalOrderGateLoss().per_segment_loss(replace(context, logits=logits))
    assert result.segment_losses.item() < 1e-4
    row = logits[0, 2, list(IDS)].float()
    expected = torch.logsumexp(row, 0) - torch.logsumexp(row[a + 1:], 0)
    assert row[a + 1:].numel() == valid_count
    assert result.segment_losses.item() == pytest.approx(expected.item(), abs=1e-6)


def test_real_runner_global_segment_mean_and_prefix_isolation():
    context = _context(((600, 200, 601, 201), (0, 0, 999, 999)), extra_segment=True)
    runner = LossRunner(
        base_ce_weight=1, token_type_gate_weight=.2,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
        conditional_order_gate_weight=.2, conditional_order_gate=ConditionalOrderGateLoss(),
    )
    plan = runner.prepare_planned_step((context.token_sequence,))
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)
    gate = bundle.term_by_name("conditional_order_gate")
    assert set(plan.denominators) == {"base_ce", "token_type_gate", "conditional_order_gate"}
    assert gate.denominator.eligible_segment_count == 3
    assert gate.raw_loss.item() == pytest.approx(
        ConditionalOrderGateLoss().per_segment_loss(context).segment_losses.sum().item()/3
    )
    assert bundle.total_loss.item() == pytest.approx(sum(term.weighted_loss.item() for term in bundle.terms))
    artifact = runner.finalize_planned_step((bundle.to_artifact_dict(),), plan)
    assert gate.diagnostics["term_diagnostics"]["participating_slot_count"] == 4
    # Another object and segment cannot silently provide x1 to the first box.
    ids = list(context.token_sequence.input_ids)
    ids[1] = IDS[599]
    changed = replace(context, token_sequence=replace(context.token_sequence, input_ids=tuple(ids)))
    with pytest.raises(LossContractError, match="prefix"):
        runner.compute_micro_step(changed, plan, local_micro_step_index=0)


def test_invalid_annotation_and_impossible_prefix_fail_closed():
    with pytest.raises(CoordinateTargetContractError):
        CoordinateLossTarget((999, 0, 999, 1), 0)
    with pytest.raises(CoordinateTargetContractError):
        CoordinateLossTarget((10, 10, 10, 11), 0)


def test_wrong_role_and_compact_shift_fail_at_actual_caller():
    context = _context()
    atoms = list(context.token_sequence.atoms)
    bbox = atoms[2].coordinate_target.bbox
    atoms[2] = replace(atoms[2], coordinate_target=CoordinateLossTarget(bbox, 3))
    atoms[3] = replace(atoms[3], coordinate_target=CoordinateLossTarget(bbox, 2))
    with pytest.raises(LossContractError):
        ConditionalOrderGateLoss().per_segment_loss(replace(
            context, token_sequence=replace(context.token_sequence, atoms=tuple(atoms))
        ))
    with pytest.raises(LossContractError):
        LossContext(
            context.logits[:, :3, :], context.token_sequence, VOCAB,
            logits_position_ids=(0, 1, 2),
        )


def test_new_term_uses_same_global_segment_denominator_as_other_losses():
    context = _context(extra_segment=True)
    runner = LossRunner(
        base_ce_weight=1, token_type_gate_weight=.2,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
        conditional_order_gate_weight=.2, conditional_order_gate=ConditionalOrderGateLoss(),
    )
    plan = runner.prepare_planned_step(
        (context.token_sequence,), world_size=2, rank=0,
        denominator_gatherer=lambda payload: (payload, payload),
    )
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)
    # Identical ranks: DDP divides gradients by two, so each rank scales its
    # contribution by two and uses the doubled global eligible count.
    local_plan = runner.prepare_planned_step((context.token_sequence,))
    local = runner.compute_micro_step(context, local_plan, local_micro_step_index=0)
    assert plan.denominators["conditional_order_gate"].eligible_segment_count == 4
    assert plan.backend_gradient_scale == 2
    assert bundle.term_by_name("conditional_order_gate").raw_loss.item() == pytest.approx(
        local.term_by_name("conditional_order_gate").raw_loss.item()/2
    )


def test_legal_mass_not_mean_or_future_target_and_no_earlier_slot_gradient():
    context = _context(((100, 100, 900, 900),))
    a = torch.full_like(context.logits, -100.)
    # Illegal greedy x2=50 despite a large mean: only illegal mass matters.
    a[0, 2, IDS[50]] = torch.log(torch.tensor(.51))
    a[0, 2, IDS[950]] = torch.log(torch.tensor(.49))
    a[0, 3, IDS[950]] = 0
    a.requires_grad_()
    term = ConditionalOrderGateLoss()
    loss = term.per_segment_loss(replace(context, logits=a)).segment_losses.sum()
    assert loss.item() == pytest.approx(-.5 * __import__('math').log(.49), abs=1e-6)
    # Moving all legal mass far from the annotated x2 must not change this loss.
    b = a.detach().clone()
    b[0, 2, IDS[950]], b[0, 2, IDS[101]] = a[0, 2, IDS[101]], a[0, 2, IDS[950]]
    torch.testing.assert_close(term.per_segment_loss(replace(context, logits=b)).segment_losses.sum(), loss)
    loss.backward()
    assert a.grad[0, :2].count_nonzero() == 0
    assert a.grad[0, 2, IDS[50]] > 0
    assert a.grad[0, 2, IDS[950]] < 0


def test_compact_bf16_and_ddp_gradients_match_dense():
    context = _context(((600,200,601,201),(0,0,999,999)), extra_segment=True)
    runner = LossRunner(1., 0., ('desc_text','schema','coordinate','eos'),
                       conditional_order_gate_weight=.01, conditional_order_gate=ConditionalOrderGateLoss())
    plan = runner.prepare_planned_step((context.token_sequence,),world_size=2,rank=0,
                                      denominator_gatherer=lambda p:(p,p))
    dense = runner.compute_micro_step(context,plan,local_micro_step_index=0).term_by_name('conditional_order_gate')
    ref = ConditionalOrderGateLoss().per_segment_loss(context).segment_losses.mean()*.01
    torch.testing.assert_close(dense.backward_contribution,ref)
    positions=tuple(atom.causal_logits_position for atom in context.atoms)
    compact=context.logits.detach()[:,list(positions)].clone().requires_grad_()
    cc=LossContext(compact,context.token_sequence,VOCAB,logits_position_ids=positions)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        result=runner.compute_micro_step(cc,plan,local_micro_step_index=0).term_by_name('conditional_order_gate')
    torch.testing.assert_close(result.backward_contribution,ref)
    g=torch.autograd.grad(ref,context.logits)[0]
    cg=torch.autograd.grad(result.backward_contribution,compact)[0]
    torch.testing.assert_close(cg,g[:,list(positions)])


def test_new_config_and_removed_expectation_key():
    from src.config.loader import load_train_config
    from src.config.models import AuxiliaryLossesConfig
    from pydantic import ValidationError
    cfg=load_train_config('configs/train/geo_sorted_xy/untied_illegal_mass.yaml').config
    assert cfg.losses.auxiliary.conditional_order_gate.weight==.01
    assert not cfg.model.special_token_embeddings.tie_word_embeddings
    assert cfg.resume.checkpoint_dir is None
    assert LossRunner.from_config(cfg.losses).conditional_order_gate is not None
    assert LossRunner.from_config(load_train_config('configs/train/geo_sorted_xy/untied.yaml').config.losses).conditional_order_gate is None
    for payload in ({'raw_axis_validity_hinge':{'weight':.01}},
                    {'conditional_order_gate':{'weight':.01,'margin':.01}},
                    {'conditional_order_gate':{'weight':-1}}):
        with pytest.raises(ValidationError):AuxiliaryLossesConfig.model_validate(payload)
