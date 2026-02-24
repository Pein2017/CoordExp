"""Stage-2 two-channel trainer package.

This package hosts helper mixins and re-exports the canonical trainer
implementation from the sibling module file `src/trainers/stage2_two_channel.py`.

It also mirrors attribute assignments into the implementation module so test
monkeypatches targeting `src.trainers.stage2_two_channel.*` affect runtime code.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from .executors import Stage2ABChannelExecutorsMixin
from .scheduler import Stage2ABSchedulerMixin

_PARENT_PKG = __name__.rsplit(".", 1)[0]
_IMPL_MOD_NAME = f"{_PARENT_PKG}._stage2_two_channel_impl"
_IMPL_PATH = Path(__file__).resolve().parents[1] / "stage2_two_channel.py"

logger = logging.getLogger(__name__)


def _load_impl() -> ModuleType:
    mod = sys.modules.get(_IMPL_MOD_NAME)
    if isinstance(mod, ModuleType):
        try:
            return importlib.reload(mod)
        except ImportError:
            logger.warning(
                "Failed to reload stage2_two_channel implementation module %s; continuing with cached module.",
                _IMPL_MOD_NAME,
                exc_info=True,
            )
            return mod

    spec = importlib.util.spec_from_file_location(_IMPL_MOD_NAME, _IMPL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load stage2_two_channel impl from {_IMPL_PATH}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[_IMPL_MOD_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


_IMPL = _load_impl()


class _Stage2ModuleProxy(ModuleType):
    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if hasattr(_IMPL, name):
            try:
                setattr(_IMPL, name, value)
            except (AttributeError, TypeError):
                logger.warning(
                    "Failed to mirror attribute %s onto stage2_two_channel implementation module.",
                    name,
                    exc_info=True,
                )


_module_obj = sys.modules.get(__name__)
if isinstance(_module_obj, ModuleType) and not isinstance(_module_obj, _Stage2ModuleProxy):
    _module_obj.__class__ = _Stage2ModuleProxy

for _name in (
    "Stage2ABTrainingTrainer",
    "Stage2TwoChannelTrainer",
    "_PendingStage2Log",
    "_expectation_decode_coords",
    "_bbox_smoothl1_ciou_loss",
    "_build_teacher_forced_payload",
    "_extract_gt_bboxonly",
    "_matched_prefix_structure_positions",
    "_stage2_ab_tail_closure_positions",
    "hungarian_match_maskiou",
    "parse_rollout_for_matching",
):
    if hasattr(_IMPL, _name):
        globals()[_name] = getattr(_IMPL, _name)


def __getattr__(name: str) -> Any:
    return getattr(_IMPL, name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(dir(_IMPL)))


__all__ = [
    "Stage2ABChannelExecutorsMixin",
    "Stage2ABSchedulerMixin",
    "Stage2ABTrainingTrainer",
    "Stage2TwoChannelTrainer",
    "_PendingStage2Log",
]
