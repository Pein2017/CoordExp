"""Transaction-private checkpoint lifecycle for decision-surface evaluation."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile


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


__all__ = ["private_proposal_checkpoint"]
