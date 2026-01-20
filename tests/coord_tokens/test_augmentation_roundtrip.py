import random
import types
import sys

import pytest


# Minimal yaml shim to avoid pulling PyYAML in the test environment
class _YamlShim:
    @staticmethod
    def safe_load(_):
        return {}

    @staticmethod
    def safe_dump(data, *_, **__):
        return str(data)


sys.modules.setdefault("yaml", _YamlShim())

# Minimal torch shim for dataset imports
torch_mod = types.ModuleType("torch")
torch_utils = types.ModuleType("torch.utils")
torch_utils_data = types.ModuleType("torch.utils.data")

class _Dataset:
    pass


def _get_worker_info():
    return None


torch_utils_data.Dataset = _Dataset
torch_utils_data.get_worker_info = _get_worker_info
torch_utils.data = torch_utils_data
torch_mod.utils = torch_utils

sys.modules.setdefault("torch", torch_mod)
sys.modules.setdefault("torch.utils", torch_utils)
sys.modules.setdefault("torch.utils.data", torch_utils_data)

# Minimal swift shim to satisfy config.loader imports
swift_mod = types.ModuleType("swift")
swift_utils = types.ModuleType("swift.utils")
swift_utils.get_dist_setting = lambda: (0, 0, 1, 0)
swift_llm = types.ModuleType("swift.llm")
swift_llm_argument = types.ModuleType("swift.llm.argument")
swift_llm_argument.RLHFArguments = object
swift_llm_argument.TrainArguments = object

swift_mod.utils = swift_utils
swift_mod.llm = swift_llm
swift_llm.argument = swift_llm_argument

sys.modules.setdefault("swift", swift_mod)
sys.modules.setdefault("swift.utils", swift_utils)
sys.modules.setdefault("swift.llm", swift_llm)
sys.modules.setdefault("swift.llm.argument", swift_llm_argument)

from src.datasets.preprocessors.augmentation import AugmentationPreprocessor


def _install_dummy_compose(monkeypatch):
    """Install a lightweight Compose substitute and stub augmentation package imports."""

    class DummyCompose:
        def __init__(self, ops):
            self.ops = ops

        def apply(self, images, geoms, *, width, height, rng):
            return images, geoms

    aug_pkg = types.ModuleType("src.datasets.augmentation")
    aug_base = types.ModuleType("src.datasets.augmentation.base")
    aug_base.Compose = DummyCompose
    aug_ops = types.ModuleType("src.datasets.augmentation.ops")
    aug_registry = types.ModuleType("src.datasets.augmentation.registry")
    aug_registry.get = lambda name: None

    aug_pkg.base = aug_base
    aug_pkg.ops = aug_ops
    aug_pkg.registry = aug_registry

    monkeypatch.setitem(sys.modules, "src.datasets.augmentation", aug_pkg)
    monkeypatch.setitem(sys.modules, "src.datasets.augmentation.base", aug_base)
    monkeypatch.setitem(sys.modules, "src.datasets.augmentation.ops", aug_ops)
    monkeypatch.setitem(sys.modules, "src.datasets.augmentation.registry", aug_registry)

    return DummyCompose


def _sample_record():
    return {
        "images": ["dummy.png"],
        "objects": [
            {"bbox_2d": ["<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"]},
            {"poly": ["<|coord_0|>", "<|coord_0|>", "<|coord_999|>", "<|coord_0|>", "<|coord_999|>", "<|coord_999|>", "<|coord_0|>", "<|coord_999|>"]},
        ],
        "width": 1024,
        "height": 1024,
    }


@pytest.fixture(autouse=True)
def _clean_sys_modules(monkeypatch):
    # Ensure dummy module can be injected cleanly per test
    import sys

    to_remove = [k for k in sys.modules if k.startswith("src.datasets.augmentation.base")]
    for k in to_remove:
        sys.modules.pop(k, None)
    yield
    for k in to_remove:
        sys.modules.pop(k, None)


def test_coord_tokens_roundtrip_identity(monkeypatch):
    import sys

    DummyCompose = _install_dummy_compose(monkeypatch)

    # Stub apply_augmentations to bypass PIL IO and return identity geometry
    def fake_apply(images, geoms, pipeline, rng=None):
        return images, geoms

    monkeypatch.setitem(sys.modules, "src.datasets.augment", types.SimpleNamespace(apply_augmentations=fake_apply))

    pre = AugmentationPreprocessor(
        augmenter=DummyCompose([]),
        rng=random.Random(123),
        coord_tokens_enabled=True,
    )

    sample = _sample_record()
    out = pre.preprocess(sample)
    objs = out["objects"]

    assert objs[0]["bbox_2d"] == ["<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"]
    assert objs[1]["poly"] == sample["objects"][1]["poly"]

    assert objs[0]["_coord_token_ints"]["bbox_2d"] == [10, 20, 30, 40]
    assert objs[1]["_coord_tokens"]["poly"] == sample["objects"][1]["poly"]


def test_coord_tokens_roundtrip_after_affine(monkeypatch):
    import sys

    DummyCompose = _install_dummy_compose(monkeypatch)

    # Simulate a simple translation affine on geometry output (adds +1 to all coords)
    def fake_apply(images, geoms, pipeline, rng=None):
        shifted = []
        for g in geoms:
            key, vals = next(iter(g.items()))
            shifted_vals = [min(max(v + 1, 0), 999) for v in vals]
            shifted.append({key: shifted_vals})
        return images, shifted

    monkeypatch.setitem(sys.modules, "src.datasets.augment", types.SimpleNamespace(apply_augmentations=fake_apply))

    pre = AugmentationPreprocessor(
        augmenter=DummyCompose([]),
        rng=random.Random(123),
        coord_tokens_enabled=True,
    )

    sample = _sample_record()
    out = pre.preprocess(sample)
    objs = out["objects"]

    assert objs[0]["bbox_2d"] == ["<|coord_11|>", "<|coord_21|>", "<|coord_31|>", "<|coord_41|>"]
    assert objs[0]["_coord_token_norm"]["bbox_2d"] == [v / 1000.0 for v in [11, 21, 31, 41]]
