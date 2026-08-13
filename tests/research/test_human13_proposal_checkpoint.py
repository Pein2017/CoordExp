from __future__ import annotations

from pathlib import Path

import pytest

from scripts.research.human13_proposal_checkpoint import private_proposal_checkpoint


def _write_payload(run_dir: Path) -> Path:
    checkpoint = run_dir / "checkpoints" / "step-1"
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "special_token_embeddings").mkdir()
    return checkpoint


def test_private_proposal_exists_only_inside_context(tmp_path: Path) -> None:
    parent = tmp_path / "private"
    with private_proposal_checkpoint(parent, writer=_write_payload) as checkpoint:
        assert checkpoint.parent.name == "checkpoints"
        assert checkpoint.is_dir()
        private_root = checkpoint.parent.parent
        assert private_root.parent == parent
        assert private_root.name.startswith(".transaction-proposal-")
    assert not private_root.exists()
    assert list(parent.iterdir()) == []


def test_private_proposal_cleans_after_gate_failure(tmp_path: Path) -> None:
    parent = tmp_path / "private"
    with pytest.raises(RuntimeError, match="reject"):
        with private_proposal_checkpoint(parent, writer=_write_payload):
            raise RuntimeError("reject")
    assert list(parent.iterdir()) == []


def test_private_proposal_rejects_wrong_writer_target(tmp_path: Path) -> None:
    parent = tmp_path / "private"

    def outside(_run_dir: Path) -> Path:
        checkpoint = tmp_path / "accepted" / "checkpoints" / "step-1"
        (checkpoint / "adapter").mkdir(parents=True)
        (checkpoint / "special_token_embeddings").mkdir()
        return checkpoint

    with pytest.raises(ValueError, match="transaction-private root"):
        with private_proposal_checkpoint(parent, writer=outside):
            pass
