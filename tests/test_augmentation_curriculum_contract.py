import pytest

from src.datasets.preprocessors.augmentation import AugmentationPreprocessor


class _DummyOp:
    _aug_name = "dummy"

    def __init__(self) -> None:
        self.prob = 0.5


class _DummyAugmenter:
    def __init__(self) -> None:
        self.ops = [_DummyOp()]


def test_curriculum_step_invalid_fails_fast() -> None:
    state = {"step": "not-an-int", "ops": {}}
    pre = AugmentationPreprocessor(augmenter=_DummyAugmenter(), curriculum_state=state)

    with pytest.raises(ValueError, match=r"curriculum_state\.step"):
        pre._sync_curriculum()


def test_curriculum_bypass_prob_invalid_fails_fast() -> None:
    state = {"step": 1, "bypass_prob": "not-a-float", "ops": {}}
    pre = AugmentationPreprocessor(augmenter=_DummyAugmenter(), curriculum_state=state)

    with pytest.raises(ValueError, match=r"curriculum_state\.bypass_prob"):
        pre._sync_curriculum()


def test_curriculum_bypass_prob_out_of_range_fails_fast() -> None:
    state = {"step": 1, "bypass_prob": 1.2, "ops": {}}
    pre = AugmentationPreprocessor(augmenter=_DummyAugmenter(), curriculum_state=state)

    with pytest.raises(ValueError, match=r"curriculum_state\.bypass_prob"):
        pre._sync_curriculum()


def test_curriculum_ops_type_invalid_fails_fast() -> None:
    state = {"step": 1, "ops": "not-a-mapping"}
    pre = AugmentationPreprocessor(augmenter=_DummyAugmenter(), curriculum_state=state)

    with pytest.raises(TypeError, match=r"curriculum_state\.ops"):
        pre._sync_curriculum()


def test_curriculum_unknown_op_fails_fast() -> None:
    state = {"step": 1, "ops": {"unknown": {"prob": 0.2}}}
    pre = AugmentationPreprocessor(augmenter=_DummyAugmenter(), curriculum_state=state)

    with pytest.raises(KeyError, match=r"Unknown augmentation op override\(s\)"):
        pre._sync_curriculum()


def test_curriculum_unknown_param_fails_fast() -> None:
    state = {"step": 1, "ops": {"dummy": {"does_not_exist": 0.2}}}
    pre = AugmentationPreprocessor(augmenter=_DummyAugmenter(), curriculum_state=state)

    with pytest.raises(AttributeError, match=r"has no parameter"):
        pre._sync_curriculum()


def test_curriculum_override_applies_to_known_op() -> None:
    state = {"step": 1, "ops": {"dummy": {"prob": 0.2}}}
    augmenter = _DummyAugmenter()
    pre = AugmentationPreprocessor(augmenter=augmenter, curriculum_state=state)

    pre._sync_curriculum()

    assert augmenter.ops[0].prob == pytest.approx(0.2)
