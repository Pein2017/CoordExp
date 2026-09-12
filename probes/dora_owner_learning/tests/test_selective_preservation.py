import copy
import importlib
import json
from pathlib import Path

import pytest
import torch

from probes.dora_owner_learning.selective_preservation import (
    LAMBDA, file_hash, prepare, preservation_positions, publish, selective_loss, validate_receipt,
)

EOS = 151645


@pytest.mark.parametrize("suffix", ["", "_wide", "_dense", "_strong", "_seven", "_stable"])
def test_fresh_profile_preparation_captures_shared_input_sources(tmp_path, suffix):
    profile = importlib.import_module(f"probes.dora_owner_learning.selective_preservation{suffix}")
    output = tmp_path / "prepared"
    profile.prepare(output)
    identity = json.loads((output / "code_identity.json").read_text())
    files = {Path(row["path"]).resolve(): row for row in identity["files"]}
    for source in (Path(profile.__file__).with_name("runtime.py"), Path("src/inference/inputs.py"),
                   Path("src/inference/prompt.py"), Path("src/inference/image_plan.py"),
                   Path("src/qwen/encoding.py"), Path("src/qwen/images.py")):
        record = files[source.resolve()]
        assert file_hash(source) == record["sha256"] == file_hash(Path(record["staged"]))


def test_fresh_source_snapshot_changes_with_shared_helper_bytes(tmp_path, monkeypatch):
    from probes.dora_owner_learning import selective_preservation as profile

    output = tmp_path / "prepared"
    source = Path("src/inference/inputs.py").resolve()
    changed_source = tmp_path / "changed-inputs.py"
    changed_source.write_bytes(source.read_bytes() + b"\n# changed producer fixture\n")
    copyfile = profile.shutil.copyfile

    def copy_changed_helper(src, dst, **kwargs):
        return copyfile(changed_source if Path(src).resolve() == source else src, dst, **kwargs)

    monkeypatch.setattr(profile.shutil, "copyfile", copy_changed_helper)
    profile.prepare(output)
    identity = json.loads((output / "code_identity.json").read_text())
    record = next(row for row in identity["files"] if Path(row["path"]) == source)
    assert record["sha256"] == file_hash(changed_source) != file_hash(source)


def fixture(length=4):
    ids = [1, 2] + [3] * (length - 3) + [EOS]
    entry = dict(state_ids=[1], action_index=1, A_id=2)
    return ids, entry, 3, preservation_positions(ids, entry, 3)


def test_mask_is_prefix_before_entry_and_free_suffix_including_eos():
    ids, entry, suffix, positions = fixture()
    assert positions == [0, 3]
    assert 1 not in positions and 2 not in positions and len(ids) - 1 in positions
    with pytest.raises(ValueError):
        preservation_positions([1, 3, 2, EOS], entry, suffix)
    with pytest.raises(ValueError):
        preservation_positions([1, EOS, 3, EOS], entry, suffix)
    with pytest.raises(ValueError):
        preservation_positions(ids, entry, 1)


def test_source_identity_zero_kl_and_reference_detached_gradient():
    ids, entry, suffix, positions = fixture()
    # Full toy vocabulary includes actual EOS; no selected-token KL proxy.
    source = torch.zeros(len(ids), EOS + 1)
    source[:, :4] = torch.tensor([1., 2., -1., .5])
    ref = torch.log_softmax(source[positions], -1).detach().requires_grad_(True)
    initial = source.detach().clone().requires_grad_(True)
    _, _, kl = selective_loss(initial, torch.tensor(ids), entry, ids, suffix, positions, ref)
    assert float(kl.detach()) == 0
    student = source.clone()
    student[:, 0] += .7
    student.requires_grad_(True)
    loss, ce, kl = selective_loss(student, torch.tensor(ids), entry, ids, suffix, positions, ref)
    expected = (ref.detach().exp() * (ref.detach() - torch.log_softmax(student[positions], -1))).sum(-1).mean()
    assert torch.equal(kl, expected)
    assert kl > 0
    assert torch.equal(loss, .5 * (ce + 10 * kl))
    loss.backward()
    assert ref.grad is None
    assert torch.count_nonzero(student.grad[0]) > 0  # shared prefix is not frozen
    assert torch.count_nonzero(student.grad[1]) > 0  # the only CE target
    assert torch.count_nonzero(student.grad[2]) == 0  # supplied remaining A row
    assert torch.count_nonzero(student.grad[-1]) > 0  # EOS preservation
    assert torch.isfinite(student.grad).all()


def test_wrong_mask_and_target_positions_fail_closed():
    ids, entry, suffix, positions = fixture()
    logits = torch.zeros(len(ids), EOS + 1)
    ref = torch.log_softmax(logits[positions], -1)
    with pytest.raises(ValueError, match='mask alignment'):
        selective_loss(logits, torch.tensor(ids), entry, ids, suffix, [0, 2, 3], ref)
    with pytest.raises(ValueError, match='mask alignment'):
        selective_loss(logits, torch.tensor([1, 3, 2, EOS]), entry, ids, suffix, positions, ref)
    with pytest.raises(ValueError, match='reference distribution shape'):
        selective_loss(logits, torch.tensor(ids), entry, ids, suffix, positions, ref[:1])


def test_equal_image_weighting_not_pooled_state_mean():
    contributions, ces, kls, sizes = [], [], [], []
    for length, perturbation in [(4, 2.), (8, .2)]:
        ids, entry, suffix, positions = fixture(length)
        source = torch.full((length, EOS + 1), -20.)
        source[:, :4] = 0
        student = source.clone()
        student[:, 0] += perturbation
        ref = torch.log_softmax(source[positions], -1)
        loss, ce, kl = selective_loss(student, torch.tensor(ids), entry, ids, suffix, positions, ref)
        contributions.append(loss)
        ces.append(ce)
        kls.append(kl)
        sizes.append(len(positions))
    actual = sum(contributions)
    expected = .5 * sum(ces) + LAMBDA * .5 * sum(kls)
    wrong = .5 * sum(ces) + LAMBDA * sum(k * n for k, n in zip(kls, sizes)) / sum(sizes)
    assert torch.allclose(actual, expected)
    assert not torch.isclose(actual, wrong)


def test_fixed_step_checkpoint_does_not_require_margin_crossing(tmp_path):
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    for name in ('adapter_model.safetensors', 'adapter_config.json'):
        (adapter / name).write_bytes(b'fixture')
    identity = dict(root=str(adapter), files=[dict(relative_path=p.name, sha256=file_hash(p)) for p in sorted(adapter.iterdir())])
    receipt = dict(schema_version='selective_preservation.training.v1', status='completed', updates=23,
                   stop_reason='fixed_steps', adapter=identity, source_embedding=identity,
                   cases=[dict(case_id=c, target_token_id=2) for c in ('a', 'b')],
                   final_scores={c: dict(target_id=2, A_vs_best_other_margin=-3.) for c in ('a', 'b')})
    validate_receipt(receipt)
    publish(tmp_path / 'receipt.json', receipt)
    with pytest.raises(Exception):
        publish(tmp_path / 'receipt.json', receipt)
    for changes in ({'updates': 22}, {'stop_reason': 'both_margins_cross'}, {'final_scores': {}}):
        with pytest.raises(ValueError):
            validate_receipt({**copy.deepcopy(receipt), **changes})
    (adapter / 'adapter_model.safetensors').write_bytes(b'bad')
    with pytest.raises(ValueError):
        validate_receipt(receipt)


def test_occupied_training_root(tmp_path):
    with pytest.raises(ValueError, match='occupied'):
        prepare(tmp_path)
