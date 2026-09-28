from dataclasses import replace

import pytest
import torch

from src.common.errors import LossContractError
from src.coordinate_targets import CoordinateLossTarget
from src.losses.context import LossContext
from src.losses.runner import LossRunner
from src.losses.start_coordinate import StartCoordinateLoss
from src.losses.vocab import TokenVocabularyGroups
from src.packing.planner import PackedSegment
from src.supervision import TokenAtom, TokenSequence


IDS = tuple(range(5, 1005))
VOCAB = TokenVocabularyGroups(vocab_size=1008, desc_text=(1007,), schema=(0, 1),
                             coordinate=IDS, eos=(1006,), blocked=(2, 3, 4, 1005))


def context(scenes):
    ids, atoms, segments = [], [], []
    for si, objects in enumerate(scenes):
        begin = len(ids)
        ids.append(0)
        for oi, (desc, box) in enumerate(objects):
            pos = len(ids)
            ids.append(1007)
            common = dict(pack_index=0, segment_index=si, example_index=si, example_id=str(si), object_id=str(oi), source="test")
            atoms.append(TokenAtom(**common, target_position=pos, logical_target_position=pos,
                                   token_id=1007, token_type="desc_text", text=desc, field="desc"))
            for role, value in enumerate(box):
                pos = len(ids)
                ids.append(IDS[value])
                atoms.append(TokenAtom(**common, target_position=pos, logical_target_position=pos,
                                       token_id=IDS[value], token_type="coordinate", text=f"<|coord_{value}|>",
                                       field=f"bbox[{role}]", coordinate_target=CoordinateLossTarget(box, role)))
        segments.append(PackedSegment(0, si, si, str(si), begin, len(ids)))
    return LossContext(torch.zeros(1, len(ids), 1008, requires_grad=True),
                       TokenSequence(0, tuple(ids), tuple(segments), tuple(atoms), ()), VOCAB)


def start_atom(c, obj=0, role=0, segment=0):
    return next(a for a in c.atoms if a.segment_index == segment and a.object_id == str(obj)
                and a.coordinate_target is not None and a.coordinate_target.slot_index == role)


def test_margin_changes_only_competing_onset_peaks_and_has_teeth():
    c = context([[('person', (120, 200, 320, 400)), ('person', (740, 100, 900, 300))]])
    a = start_atom(c)
    with torch.no_grad():
        c.logits[0, a.causal_logits_position, IDS[120]] = 1
        c.logits[0, a.causal_logits_position, IDS[740]] = 1.4
    term = StartCoordinateLoss('instance_margin', radius_cap=0)
    loss = term.per_atom_loss(c).mean()
    loss.backward()
    g = c.logits.grad[0, a.causal_logits_position]
    assert g[IDS[120]] < 0 and g[IDS[740]] > 0 and g.count_nonzero() == 2
    assert not c.logits.grad[0, a.target_position].any()  # y1: other x1 is incompatible.
    changed = c.logits.detach().clone()
    changed[0, a.causal_logits_position, IDS[740]] += 1
    assert term.per_atom_loss(replace(c, logits=changed)).mean() > loss
    for atom in c.atoms:
        if atom.coordinate_target and atom.coordinate_target.slot_index > 1:
            assert not c.logits.grad[0, atom.causal_logits_position].any()


def test_shared_x_uses_y_and_never_compares_other_images_or_classes():
    c = context([[('person', (100, 100, 300, 300)), ('person', (100, 700, 300, 900)),
                  ('dog', (800, 100, 900, 200))], [('person', (900, 300, 950, 400))]])
    term = StartCoordinateLoss('instance_margin', radius_cap=0)
    term.per_atom_loss(c).mean().backward()
    assert term.last_diagnostics['margin_eligible_slot_count'] == 2
    assert not c.logits.grad[0, start_atom(c).causal_logits_position].any()
    assert c.logits.grad[0, start_atom(c, role=1).causal_logits_position].any()
    assert not c.logits.grad[0, start_atom(c, segment=1).causal_logits_position].any()


def test_local_mass_matches_formula_and_does_not_force_uniform_positive_bins():
    c = context([[('person', (120, 200, 320, 400))]])
    a = start_atom(c)
    term = StartCoordinateLoss('local_mass')
    with torch.no_grad():
        c.logits[0, a.causal_logits_position, IDS[120]] = 2
    loss = term.per_atom_loss(c).mean()
    x = c.logits[0, a.causal_logits_position, list(IDS)]
    expected = ((x.logsumexp(0) - x[116:125].logsumexp(0)) + torch.log(torch.tensor(1000. / 9))) / 2
    assert loss.item() == pytest.approx(expected.detach().item(), abs=1e-6)
    loss.backward()
    g = c.logits.grad[0, a.causal_logits_position, list(IDS)]
    assert torch.all(g[116:125] < 0) and torch.all(g[400:700] > 0)


def test_runner_uses_image_mean_and_global_ddp_scale_with_empty_margin_support():
    c = context([[('person', (100, 100, 300, 300)), ('person', (700, 700, 900, 900))],
                 [('person', (500, 500, 600, 600))]])
    term = StartCoordinateLoss('instance_margin', radius_cap=0)
    runner = LossRunner(1, .1, ('desc_text', 'schema', 'coordinate', 'eos'),
                        start_coordinate_weight=.3, start_coordinate=term)
    plan = runner.prepare_planned_step((c.token_sequence,))
    bundle = runner.compute_micro_step(c, plan, local_micro_step_index=0)
    result = bundle.term_by_name('start_coordinate')
    # Image 0: two eligible x1 and two zero y1 slots; image 1: zero support.
    assert result.raw_loss.item() == pytest.approx(.05)
    assert result.weighted_loss.item() == pytest.approx(.015)
    bundle.backward_loss.backward()
    assert torch.isfinite(c.logits.grad).all()


def test_prefix_metadata_mismatch_fails_and_compact_logits_preserve_causal_rows():
    c = context([[('person', (100, 100, 300, 300)), ('person', (700, 700, 900, 900))]])
    term = StartCoordinateLoss('ce')
    positions = tuple(a.causal_logits_position for a in c.atoms)
    compact = LossContext(c.logits[:, list(positions)], c.token_sequence, VOCAB, positions)
    assert torch.equal(term.per_atom_loss(c), term.per_atom_loss(compact))
    ids = list(c.token_sequence.input_ids)
    ids[start_atom(c).target_position] = IDS[101]
    with pytest.raises(LossContractError):
        term.per_atom_loss(replace(c, token_sequence=replace(c.token_sequence, input_ids=tuple(ids))))


def test_analytic_calibration_matches_autograd(capsys):
    c = context([[('person', (100, 100, 300, 300)), ('person', (700, 700, 900, 900))]])
    StartCoordinateLoss('ce', calibrate=True).per_atom_loss(c)
    import json
    record = json.loads(capsys.readouterr().out.split('START_LOSS_CALIBRATION ')[1])
    for mode in ('ce', 'local_mass', 'instance_margin'):
        value = StartCoordinateLoss(mode).per_atom_loss(c).mean()
        g, = torch.autograd.grad(value, c.logits)
        assert g.square().sum().item() == pytest.approx(record['grad_sq'][mode], rel=1e-5)
