import math

import pytest
import torch

from probes.dora_owner_learning.selective_preservation import selective_loss
from probes.dora_owner_learning.selective_preservation_wide import (
    SUPPORT_COUNT, image_objective, prepare, support_loss, work_items,
)

EOS, VOCAB = 151645, 151646


def two_mass(length, p):
    logits = torch.full((length, VOCAB), -1000.)
    logits[:, 0], logits[:, 1] = math.log(p[0]), math.log(p[1])
    return logits


def test_support_full_vocabulary_mean_image_weight_and_gradient_includes_eos():
    ids = [0, 1, EOS]
    logits = two_mass(3, (.5, .5)).requires_grad_(True)
    ref = two_mass(3, (.8, .2)).requires_grad_(True)
    before = ref.detach().clone()
    loss, kl = support_loss(logits, torch.tensor(ids), ids, ref)
    expected_kl = .8 * math.log(.8 / .5) + .2 * math.log(.2 / .5)
    assert float(kl.detach()) == pytest.approx(expected_kl, abs=2e-7)
    assert float(loss.detach()) == pytest.approx((10 / 31) * expected_kl, abs=2e-7)
    loss.backward()
    expected_grad = torch.tensor([-.3, .3]) * (10 / (31 * 3))
    assert torch.allclose(logits.grad[0, :2], expected_grad, atol=1e-7)
    assert torch.allclose(logits.grad[-1, :2], expected_grad, atol=1e-7)
    assert ref.grad is None and torch.equal(ref.detach(), before)


def test_initial_identity_and_no_support_ce_labels():
    ids = [0, 1, EOS]
    logits = two_mass(3, (.8, .2)).requires_grad_(True)
    ref = torch.log_softmax(logits.detach(), -1)
    loss, kl = support_loss(logits, torch.tensor(ids), ids, ref)
    assert float(kl.detach()) == 0 and float(loss.detach()) == 0
    loss.backward()
    assert float(logits.grad.abs().max()) < 1e-7
    changed_ids = [1, 0, EOS]
    changed_loss, _ = support_loss(logits, torch.tensor(changed_ids), changed_ids, ref)
    assert torch.equal(changed_loss, loss)  # aligned IDs identify context, not positive CE labels.


def test_original_two_loss_weight_is_exactly_reused_not_reaveraged():
    ids = [0, 1, 2, EOS]
    entry = dict(action_index=1, state_ids=[0], A_id=1)
    logits = two_mass(4, (.5, .5))
    ref = two_mass(2, (.8, .2))
    item = dict(kind='entrance', case={'entrance': entry}, action_ids=ids, positions=[0, 3], suffix_start=3)
    got = image_objective(logits, torch.tensor(ids), item, ref)
    expected = selective_loss(logits, torch.tensor(ids), entry, ids, 3, [0, 3], ref)
    assert all(torch.equal(x, y) for x, y in zip(got, expected))
    assert float(got[0]) == pytest.approx(.5 * (math.log(2) + 10 * (.8 * math.log(1.6) + .2 * math.log(.4))), abs=2e-6)


def test_support_images_equal_weight_not_pooled_states():
    contributions, kls = [], []
    for length, q in [(2, (.5, .5)), (7, (.8, .2))]:
        ids = [0] * (length - 1) + [EOS]
        loss, kl = support_loss(two_mass(length, q), torch.tensor(ids), ids, two_mass(length, (.8, .2)))
        contributions.append(loss)
        kls.append(kl)
    assert torch.allclose(sum(contributions), (10 / SUPPORT_COUNT) * sum(kls))
    assert not torch.isclose(sum(contributions), (10 / SUPPORT_COUNT) * (kls[0] * 2 + kls[1] * 7) / 9 * 2)


def test_missing_eos_wrong_alignment_and_partial_support_mask_rejected():
    ids = [0, 1, EOS]
    logits, ref = two_mass(3, (.5, .5)), two_mass(3, (.8, .2))
    with pytest.raises(ValueError):
        support_loss(logits, torch.tensor([1, 0, EOS]), ids, ref)
    with pytest.raises(ValueError):
        support_loss(logits, torch.tensor([0, 1, 2]), [0, 1, 2], ref)
    with pytest.raises(ValueError):
        support_loss(logits, torch.tensor(ids), ids, ref[:-1])
    with pytest.raises(ValueError, match='full-state mask'):
        image_objective(logits, torch.tensor(ids), dict(kind='support', action_ids=ids, positions=[0, 1]), ref)


def test_all33_loss_coverage_and_output_collision(tmp_path):
    with pytest.raises(ValueError, match='all33'):
        work_items(dict(cases=[], trajectories={}, support_cases=[]))
    with pytest.raises(ValueError, match='occupied'):
        prepare(tmp_path)
