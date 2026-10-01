from pathlib import Path

from src.config.inference import load_infer_config, resolve_infer_run_directory
from src.config.loader import load_train_config
from src.config.paths import resolve_run_directory


def test_canonical_templates_resolve_outputs_in_the_owning_worktree(tmp_path) -> None:
    worktree = tmp_path / "owning-worktree"
    for path in sorted(Path("configs/train").rglob("*.yaml")) + sorted(
        Path("configs/smoke").rglob("*.yaml")
    ):
        config = load_train_config(path).config
        assert not Path(config.run.artifact_root).is_absolute(), path
        directory = resolve_run_directory(config, cwd=worktree)
        assert directory.run_dir.is_relative_to(worktree / "outputs"), path

    for path in sorted(Path("configs/infer").rglob("*.yaml")):
        config = load_infer_config(path).config
        # Inference paths are bound to the declaring config, even after extends.
        directory = resolve_infer_run_directory(config)
        assert directory.run_dir.is_relative_to(Path.cwd() / "outputs"), path
