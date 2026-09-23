from pathlib import Path
import multiprocessing

import pytest

from probes.training_set_completion.coordinate_codebook_alignment import paired2048_train, scale_train, three_loss_train


def _write_arm_receipt(arm: str, output: str) -> None:
    from contextlib import nullcontext

    def patched(_probe, path):
        scale_train._write_once(path, {"arm": arm})
        return nullcontext()

    three_loss_train.patched = patched
    scale_train.run = lambda config, out, plan: scale_train._patch_trainer(object())
    paired2048_train.run(arm, Path("config"), Path(output), Path("plan"))


def test_paired_entry_changes_only_horizon_and_restores_caller(monkeypatch):
    prior = (scale_train.EXPECTED_UPDATES, scale_train._patch_trainer)

    expected = 472
    observed = []
    monkeypatch.setattr(three_loss_train, "patched", lambda probe, path: observed.append(path.name))

    def fake_run(config, output, packing_plan):
        assert scale_train.EXPECTED_UPDATES == expected
        assert scale_train.EXPECTED_WORLD_SIZE == 4
        assert scale_train.EXPECTED_GRAD_ACCUM == 2
        assert scale_train._patch_trainer is not prior[1]
        scale_train._patch_trainer(object())
        return (config, output, packing_plan)

    monkeypatch.setattr(scale_train, "run", fake_run)
    paths = tuple(Path(value) for value in ("config", "output", "plan"))
    assert paired2048_train.run("early", *paths) == paths
    assert paired2048_train.run("late", *paths) == paths
    assert observed == ["early-three-loss-objective.json", "late-three-loss-objective.json"]
    expected = 2
    assert paired2048_train.run("early", *paths, smoke=True) == paths
    assert (scale_train.EXPECTED_UPDATES, scale_train._patch_trainer) == prior
    with pytest.raises(ValueError):
        paired2048_train.run("off", *paths)


def test_two_arm_objective_receipts_are_isolated_in_concurrent_callers(tmp_path):
    output = str(tmp_path / "first-entry.json")
    context = multiprocessing.get_context("spawn")
    children = [context.Process(target=_write_arm_receipt, args=(arm, output))
                for arm in ("early", "late")]
    for child in children:
        child.start()
    for child in children:
        child.join(timeout=30)
        assert child.exitcode == 0
    assert (tmp_path / "early-three-loss-objective.json").read_text().find('"early"') >= 0
    assert (tmp_path / "late-three-loss-objective.json").read_text().find('"late"') >= 0


def test_corrected_entry_passes_explicit_order_gate_contract(monkeypatch):
    observed = []
    monkeypatch.setattr(
        three_loss_train, "patched",
        lambda probe, path, **kwargs: observed.append((path.name, kwargs)),
    )

    def fake_run(config, output, packing_plan):
        scale_train._patch_trainer(object())

    monkeypatch.setattr(scale_train, "run", fake_run)
    paired2048_train.run("early", Path("config"), Path("output"), Path("plan"), order_gate=True)
    assert observed == [("early-three-loss-objective.json", {"expected_weights": {
        "base_ce": 1.0, "token_type_gate": 0.2, "conditional_order_gate": 0.2,
    }})]
