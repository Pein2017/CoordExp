from pathlib import Path

import pytest

from src.common.errors import ConfigContractError
from src.config.inference import load_infer_config, resolve_infer_run_directory
from src.config.loader import load_train_config
from src.config.paths import resolve_run_directory


@pytest.mark.parametrize("kind", ["train", "infer"])
def test_run_destinations_enforce_physical_output_ownership(tmp_path: Path, kind: str):
    if kind == "train":
        config = load_train_config("tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml").config
        resolver = resolve_run_directory
    else:
        config = load_infer_config("configs/coordexp_infras/infer/base.yaml").config
        resolver = resolve_infer_run_directory
    alias = tmp_path / "shared-alias"
    alias.symlink_to("/data/CoordExp/outputs", target_is_directory=True)
    for root, output in [
        (Path("/data/CoordExp/outputs"), "new-run"),
        (Path("/data/CoordExp"), "outputs/shared/new-run"),
        (alias, "new-run"),
    ]:
        run = config.run.model_copy(update={"artifact_root": str(root), "output_dir": output})
        with pytest.raises(ConfigContractError) as error:
            resolver(config.model_copy(update={"run": run}))
        assert error.value.code == "config.shared_output_destination"
    own_root = tmp_path / "worktree" / "outputs"
    run = config.run.model_copy(update={"artifact_root": str(own_root), "output_dir": "new-run"})
    assert resolver(config.model_copy(update={"run": run})).run_dir == own_root / "new-run"
