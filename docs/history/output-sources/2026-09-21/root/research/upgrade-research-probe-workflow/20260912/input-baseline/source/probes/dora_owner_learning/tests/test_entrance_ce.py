import copy

import pytest
import torch

from probes.dora_owner_learning.entrance_ce import (
    OPTIMIZER, file_hash, last_target_loss, prepare, publish, stop_reason, validate_completed_receipt,
)


def test_only_last_target_has_loss_and_gradient():
    logits = torch.tensor([[3., -2., 1.], [5., 0., -1.], [1., 2., 3.]], requires_grad=True)
    target = torch.tensor([0, 1, 2])
    entry = dict(state_ids=[0, 1], A_id=2)
    loss = last_target_loss(logits, target, entry)
    expected = -0.5 * torch.log_softmax(logits[-1], -1)[2]
    assert torch.equal(loss, expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:-1]) == 0
    assert torch.count_nonzero(logits.grad[-1]) == 3
    changed = logits.detach().clone()
    changed[:-1] += torch.tensor([50., -90., 12.])
    assert torch.equal(last_target_loss(changed, target, entry), loss.detach())
    with pytest.raises(ValueError):
        last_target_loss(logits, torch.tensor([0, 2, 1]), entry)
    with pytest.raises(ValueError):
        last_target_loss(logits[:-1], target, entry)


def test_two_images_accumulate_before_one_fresh_adamw_step():
    parameter = torch.nn.Parameter(torch.tensor([0.3, -0.4, 0.8]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-5, betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False)
    original = parameter.detach().clone()
    losses = []
    for target in (0, 2):
        logits = parameter.repeat(2, 1)
        loss = last_target_loss(logits, torch.tensor([1, target]), dict(state_ids=[1], A_id=target))
        losses.append(float(loss.detach()))
        loss.backward()
        assert torch.equal(parameter, original)
        assert not optimizer.state
    expected = torch.softmax(original, -1) - torch.tensor([.5, 0., .5])
    assert torch.allclose(parameter.grad, expected)
    assert sum(losses) == pytest.approx(float(-.5 * (torch.log_softmax(original, -1)[0] + torch.log_softmax(original, -1)[2])))
    optimizer.step()
    assert int(optimizer.state[parameter]['step']) == 1
    assert not torch.equal(parameter, original)


def test_joint_margin_stop_is_first_joint_crossing_or_finite_cap():
    scores = lambda a, b: {'a': {'A_vs_best_other_margin': a}, 'b': {'A_vs_best_other_margin': b}}
    assert stop_reason(1, scores(.2, .09)) is None
    assert stop_reason(2, scores(.1, .1)) == 'both_fixed_prefix_margins_at_least_0.1'
    assert stop_reason(32, scores(.1, -.1)) == '32_update_limit'
    with pytest.raises(ValueError):
        stop_reason(33, scores(.2, .2))
    with pytest.raises(ValueError):
        stop_reason(1, scores(float('nan'), .2))
    with pytest.raises(ValueError):
        stop_reason(1, {'a': {'A_vs_best_other_margin': .2}})


def completed_fixture(tmp_path):
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    for name in ['adapter_config.json', 'adapter_model.safetensors']:
        (adapter / name).write_bytes(b'fixture')
    identity = dict(root=str(adapter), files=[dict(relative_path=p.name, sha256=file_hash(p)) for p in sorted(adapter.iterdir())])
    return dict(status='completed', schema_version='native_entrance_ce.training.v1', updates=1,
                stop_reason='both_fixed_prefix_margins_at_least_0.1', adapter=identity, source_embedding=identity,
                cases=[dict(case_id=k, target_token_id=2, state_ids=[1], prefix_token_ids=[1], action_index=1) for k in ('a', 'b')],
                final_scores={k: dict(target_id=2, A_vs_best_other_margin=.2) for k in ('a', 'b')})


def test_completed_checkpoint_publication_requires_complete_bound_bytes(tmp_path):
    receipt = completed_fixture(tmp_path)
    validate_completed_receipt(receipt)
    publish(tmp_path / 'receipt.json', receipt)
    before = (tmp_path / 'receipt.json').read_bytes()
    with pytest.raises(Exception):
        publish(tmp_path / 'receipt.json', {'bad': True})
    assert (tmp_path / 'receipt.json').read_bytes() == before
    for key, value in [('status', 'running'), ('final_scores', {}), ('updates', 2), ('cases', receipt['cases'][:1])]:
        bad = copy.deepcopy(receipt)
        bad[key] = value
        if key == 'updates':
            # A stop step alone is valid; corrupt the associated target instead.
            bad['final_scores']['a']['target_id'] = 3
        with pytest.raises(ValueError):
            validate_completed_receipt(bad)
    (tmp_path / 'adapter' / 'adapter_model.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='missing/corrupt'):
        validate_completed_receipt(receipt)


def test_occupied_training_root(tmp_path):
    with pytest.raises(ValueError, match='occupied'):
        prepare(tmp_path)


def test_frozen_optimizer_settings_publish_and_execute(tmp_path):
    publish(tmp_path / 'optimizer.json', OPTIMIZER)
    p = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([p], **OPTIMIZER)
    p.sum().backward()
    optimizer.step()
    assert optimizer.param_groups[0]['lr'] == 1e-5
    assert optimizer.param_groups[0]['weight_decay'] == 0
