from __future__ import annotations

from pathlib import Path

import pytest

from src.trainers.rollout_matching.preflight import resolve_stage2_launcher_preflight


def test_stage2_preflight_resolves_root_image_dir_relative_to_repo_root_not_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs/stage2_two_channel/prod/ab_mixed.yaml"

    # Simulate a user invoking the launcher from outside the repo.
    monkeypatch.chdir(tmp_path)

    preflight = resolve_stage2_launcher_preflight(str(config_path))

    expected = (repo_root / "public_data/lvis/rescale_32_768_bbox_max60").resolve()
    assert Path(preflight["root_image_dir_resolved"]).resolve() == expected
