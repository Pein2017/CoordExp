import copy
import json
import random

import pytest
import torch

from probes.training_set_completion import continue_training as c
from probes.training_set_completion import training


@pytest.fixture(scope="module")
def predecessor_and_candidate():
    old = training.validate_manifest(json.loads(c.OLD_MANIFEST.read_text()))
    state = c._state(c.RESUME)
    candidate = c.build_manifest()
    return old, state, candidate


def _rehash(value):
    value["content_sha256"] = training.digest({key: item for key, item in value.items() if key != "content_sha256"})


def test_actual_step64_state_and_candidate_preserve_adamw_counters_and_rng(predecessor_and_candidate):
    old, state, candidate = predecessor_and_candidate
    before = c._rng_identity(state)
    summary = c.validate_predecessor(old, state)
    assert summary["checkpoint_step"] == 64
    assert summary["parameter_layout_count"] == 588
    assert summary["optimizer"]["state_count"] == 588
    assert summary["optimizer"]["step_values"] == [64]
    assert c._rng_identity(c._state(c.RESUME)) == before
    assert candidate["runtime"] == {**old["runtime"], "updates": 256, "checkpoint_steps": [128, 192, 256], "wall_seconds": 4500, "max_model_forwards": 2816}


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda value: value["routes"][0]["ce_weights"].__setitem__(0, 0), "continuation changed frozen field: routes"),
        (lambda value: value["optimizer"].__setitem__("lr", 2e-5), "continuation changed frozen field: optimizer"),
        (lambda value: value["runtime"].__setitem__("eos_token_id", 1), "runtime extension changed unlisted field"),
    ],
)
def test_extension_rejects_mutation_of_masks_optimizer_or_unlisted_runtime(predecessor_and_candidate, mutate, error):
    old, _, candidate = predecessor_and_candidate
    changed = copy.deepcopy(candidate)
    mutate(changed)
    _rehash(changed)
    with pytest.raises(ValueError, match=error):
        c._validate_extension(old, changed)


def test_extension_rejects_old_state_manifest_binding_change(predecessor_and_candidate):
    old, state, _ = predecessor_and_candidate
    changed = copy.deepcopy(state)
    changed["manifest"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="old state manifest binding"):
        c.validate_predecessor(old, changed)


def test_extension_rejects_source_change_and_wrong_resume_checkpoint(predecessor_and_candidate, tmp_path):
    old, _, candidate = predecessor_and_candidate
    changed = copy.deepcopy(candidate)
    changed["sources"]["producer"]["sha256"] = "0" * 64
    _rehash(changed)
    with pytest.raises(ValueError, match="source bindings changed"):
        c._validate_extension(old, changed)
    with pytest.raises(ValueError, match="only declared predecessor"):
        c.build_manifest(resume=tmp_path)


def test_wrapper_calls_original_restore_and_restores_real_optimizer_rng(predecessor_and_candidate, monkeypatch, tmp_path):
    old, state, candidate = predecessor_and_candidate
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(candidate))
    named = [(item["name"], torch.nn.Parameter(torch.zeros(item["shape"], dtype=torch.float32)))
             for item in state["parameter_layout"]]
    optimizer = torch.optim.AdamW([parameter for _, parameter in named],
                                 **{**old["optimizer"], "betas": tuple(old["optimizer"]["betas"])})
    original_restore = training._restore
    before_torch, before_python = torch.get_rng_state(), random.getstate()

    def actual_restore_boundary(path, *, output, device, resume):
        assert path == manifest_path and resume == c.RESUME
        restored_step = training._restore(resume, manifest_path=path, manifest=candidate,
                                          optimizer=optimizer, named=named)
        assert restored_step == 64
        restored = optimizer.state_dict()
        assert restored["param_groups"] == state["optimizer_state_dict"]["param_groups"]
        assert restored["state"].keys() == state["optimizer_state_dict"]["state"].keys()
        for index, values in restored["state"].items():
            for field, value in values.items():
                assert torch.equal(value, state["optimizer_state_dict"]["state"][index][field])
        assert torch.equal(torch.get_rng_state(), state["torch_rng_state"])
        assert random.getstate() == state["python_random_state"]
        raise RuntimeError("sentinel after verified original restore")

    monkeypatch.setattr(training, "run", actual_restore_boundary)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        torch.manual_seed(918273)
        random.seed(918273)
        with pytest.raises(RuntimeError, match="sentinel after verified original restore"):
            c.run(manifest_path, output=tmp_path / "unused", device="cpu")
        assert training._restore is original_restore
    finally:
        torch.set_rng_state(before_torch)
        random.setstate(before_python)
