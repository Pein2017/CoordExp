"""Immutable model-free execution plan for one training entry.

Design decision 2 of ``decompose-coordexp-swift-training-orchestration``: the
plan is the frozen value the facade resolves before any live runtime exists.
Construction loads and freezes strict config, resolves the repository root and
the model-free launcher identity, copies the bounded measurement context, and
captures entry evidence.  It MUST NOT construct an Accelerator, load model
weights or a tokenizer, create a run directory, build or materialize a cache,
register a callback, or perform a collective.

The plan is a value passed to the next owners, not a service locator: it holds
no writer, model, runtime, cache, callback registry, or mutable lifecycle
state, and it never imports the facade or the training session.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import os
from pathlib import Path
import re
import time
from types import MappingProxyType
from typing import Any

from src.artifacts.resources import collect_resource_snapshot
from src.common.errors import RuntimeContractError
from src.config.loader import load_train_config
from src.config.models import ResolvedTrainConfig


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _resolve_model_free_launch_identity() -> tuple[int, int]:
    """Resolve launcher rank identity without constructing Accelerate state."""

    rank_present = "RANK" in os.environ
    world_size_present = "WORLD_SIZE" in os.environ
    if not rank_present and not world_size_present:
        return 0, 1
    if rank_present != world_size_present:
        raise RuntimeContractError(
            "launcher rank identity requires RANK and WORLD_SIZE together",
            code="runtime.preflight_launch_identity_invalid",
            context={
                "rank_present": rank_present,
                "world_size_present": world_size_present,
            },
        )
    raw_rank = os.environ["RANK"]
    raw_world_size = os.environ["WORLD_SIZE"]
    if (
        re.fullmatch(r"0|[1-9][0-9]*", raw_rank) is None
        or re.fullmatch(r"[1-9][0-9]*", raw_world_size) is None
    ):
        raise RuntimeContractError(
            "launcher rank identity must contain strict decimal RANK and WORLD_SIZE values",
            code="runtime.preflight_launch_identity_invalid",
            context={"rank_present": True, "world_size_present": True},
        )
    world_size = int(raw_world_size)
    rank = int(raw_rank)
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise RuntimeContractError(
            "launcher rank identity is outside the declared world",
            code="runtime.preflight_launch_identity_invalid",
            context={"rank": rank, "world_size": world_size},
        )
    return rank, world_size


@dataclass(frozen=True)
class TrainingExecutionPlan:
    """Frozen model-free decisions taken at the training entry point."""

    resolved_config: ResolvedTrainConfig
    repo_root: Path
    launch_rank: int
    launch_world_size: int
    measurement_context: Mapping[str, Any]
    entry_started_at: str
    entry_started_monotonic: float
    entry_resources: Mapping[str, Any]


def build_training_execution_plan(
    config_path: str | Path,
    *,
    measurement_context: Mapping[str, Any] | None,
) -> TrainingExecutionPlan:
    """Resolve every static model-free decision for one training entry."""

    entry_started_at = _utc_now()
    entry_started_monotonic = time.monotonic()
    entry_resources = collect_resource_snapshot()
    repo_root = Path.cwd().resolve()
    resolved_config = load_train_config(config_path)
    launch_rank, launch_world_size = _resolve_model_free_launch_identity()
    return TrainingExecutionPlan(
        resolved_config=resolved_config,
        repo_root=repo_root,
        launch_rank=launch_rank,
        launch_world_size=launch_world_size,
        measurement_context=MappingProxyType(dict(measurement_context or {})),
        entry_started_at=entry_started_at,
        entry_started_monotonic=entry_started_monotonic,
        entry_resources=MappingProxyType(dict(entry_resources)),
    )
