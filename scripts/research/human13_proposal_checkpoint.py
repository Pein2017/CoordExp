"""Transaction-private checkpoint lifecycle for decision-surface evaluation."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import tempfile
import uuid


@contextmanager
def private_proposal_checkpoint(
    parent: str | Path,
    *,
    writer: Callable[[Path], str | Path],
) -> Iterator[Path]:
    """Materialize one private proposal and erase it after its behavior gate.

    ``writer`` receives a fresh transaction run directory and must return the
    exact ``checkpoints/step-1`` directory it created.  The directory is never
    an accepted/public checkpoint and is removed after both success and error.
    """

    parent_path = Path(parent).expanduser()
    if not parent_path.is_absolute() or parent_path.is_symlink():
        raise ValueError("proposal parent must be an absolute non-symlink path")
    parent_path.mkdir(parents=True, exist_ok=True)
    parent_path = parent_path.resolve(strict=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".transaction-proposal-", dir=parent_path)
    ).resolve(strict=True)
    try:
        checkpoint = Path(writer(temporary)).expanduser()
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise ValueError("proposal writer did not return a regular checkpoint")
        checkpoint = checkpoint.resolve(strict=True)
        expected = temporary / "checkpoints" / "step-1"
        if checkpoint != expected:
            raise ValueError("proposal writer escaped its transaction-private root")
        children = {item.name for item in checkpoint.iterdir()}
        if children != {"adapter", "special_token_embeddings"} or any(
            item.is_symlink() or not item.is_dir() for item in checkpoint.iterdir()
        ):
            raise ValueError("proposal checkpoint payload is incomplete or unsafe")
        yield checkpoint
    finally:
        if temporary.parent == parent_path and temporary.name.startswith(
            ".transaction-proposal-"
        ):
            shutil.rmtree(temporary, ignore_errors=False)


def promote_private_proposal_checkpoint(
    checkpoint: str | Path,
    *,
    accepted_run_dir: str | Path,
    accepted_step: int,
) -> Path:
    """Atomically copy one gated private payload into its accepted ordinal."""

    if (
        isinstance(accepted_step, bool)
        or not isinstance(accepted_step, int)
        or not 1 <= accepted_step <= 8
    ):
        raise ValueError("accepted_step must be an integer from one through eight")
    source = Path(checkpoint).expanduser()
    if source.is_symlink() or not source.is_dir():
        raise ValueError("private proposal checkpoint must be a regular directory")
    source = source.resolve(strict=True)
    if (
        source.name != "step-1"
        or source.parent.name != "checkpoints"
        or not source.parent.parent.name.startswith(".transaction-proposal-")
    ):
        raise ValueError("source is not a transaction-private proposal checkpoint")
    accepted_root = Path(accepted_run_dir).expanduser()
    if not accepted_root.is_absolute() or accepted_root.is_symlink():
        raise ValueError("accepted run directory must be an absolute non-symlink path")
    checkpoints = accepted_root / "checkpoints"
    target = checkpoints / f"step-{accepted_step}"
    if target.exists() or target.is_symlink():
        raise ValueError("accepted checkpoint already exists")
    checkpoints.mkdir(parents=True, exist_ok=True)
    staging = checkpoints / f".step-{accepted_step}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copytree(source, staging, symlinks=False)
        if any(item.is_symlink() for item in staging.rglob("*")):
            raise ValueError("private proposal payload contains a symlink")
        os.replace(staging, target)
        directory_fd = os.open(checkpoints, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=False)
    return target


__all__ = [
    "private_proposal_checkpoint",
    "promote_private_proposal_checkpoint",
]
