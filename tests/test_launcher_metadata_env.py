from __future__ import annotations

import pytest

from src.sft import _collect_launcher_metadata_from_env


def test_collect_launcher_metadata_from_env_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COORDEXP_STAGE2_LAUNCHER", "scripts/train_stage2.sh")
    monkeypatch.setenv("COORDEXP_STAGE2_SERVER_DP", "2")

    meta = _collect_launcher_metadata_from_env()

    assert meta["COORDEXP_STAGE2_LAUNCHER"] == "scripts/train_stage2.sh"
    assert meta["COORDEXP_STAGE2_SERVER_DP"] == "2"
