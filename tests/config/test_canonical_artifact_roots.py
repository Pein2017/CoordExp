from pathlib import Path

from src.config.inference import load_infer_config
from src.config.loader import load_train_config


EXTERNAL_INFRA_ROOT = Path("/data/CoordExp/outputs/infra_base")


def test_canonical_production_artifact_roots_stay_outside_the_worktree() -> None:
    train_root = Path(
        load_train_config("configs/train/production.yaml").config.run.artifact_root
    )
    infer_root = Path(
        load_infer_config("configs/infer/production.yaml").config.run.artifact_root
    )

    assert train_root.is_absolute()
    assert infer_root.is_absolute()
    assert train_root.is_relative_to(EXTERNAL_INFRA_ROOT)
    assert infer_root.is_relative_to(EXTERNAL_INFRA_ROOT)
