"""Data-parallel inference planning and CUDA binding helpers."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from src.common.errors import RuntimeContractError
from src.config.fingerprint import sha256_json


@dataclass(frozen=True)
class DecodeBatchBlock:
    batch_id: int
    row_indices: tuple[int, ...]
    row_ids: tuple[str, ...]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "row_indices": list(self.row_indices),
            "row_ids": list(self.row_ids),
        }


@dataclass(frozen=True)
class RankShardPlan:
    rank: int
    world_size: int
    parent_visible_device_token: str
    per_device_batch_size: int
    batch_ids: tuple[int, ...]
    row_indices: tuple[int, ...]
    row_ids: tuple[str, ...]
    shard_dir_name: str

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "parent_visible_device_token": self.parent_visible_device_token,
            "per_device_batch_size": self.per_device_batch_size,
            "batch_ids": list(self.batch_ids),
            "row_indices": list(self.row_indices),
            "row_ids": list(self.row_ids),
            "shard_dir_name": self.shard_dir_name,
        }


@dataclass(frozen=True)
class DataParallelPlan:
    visible_cuda_tokens: tuple[str, ...]
    active_ranks: int
    per_device_batch_size: int
    decode_batch_count: int
    decode_batches: tuple[DecodeBatchBlock, ...]
    ranks: tuple[RankShardPlan, ...]
    fingerprint: str

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "visible_cuda_tokens": list(self.visible_cuda_tokens),
            "active_ranks": self.active_ranks,
            "per_device_batch_size": self.per_device_batch_size,
            "decode_batch_count": self.decode_batch_count,
            "decode_batches": [
                batch.to_artifact_dict() for batch in self.decode_batches
            ],
            "ranks": [rank.to_artifact_dict() for rank in self.ranks],
            "fingerprint": self.fingerprint,
        }


def detect_cuda_device_count() -> int:
    return int(torch.cuda.device_count())


def resolve_visible_cuda_tokens(
    *,
    environ: Mapping[str, str] | None = None,
    cuda_device_count: Callable[[], int] | None = None,
) -> tuple[str, ...]:
    env = os.environ if environ is None else environ
    if "CUDA_VISIBLE_DEVICES" in env:
        value = str(env.get("CUDA_VISIBLE_DEVICES") or "").strip()
        if not value:
            return ()
        tokens = tuple(token.strip() for token in value.split(",") if token.strip())
        return tuple(token for token in tokens if token != "-1")
    count_fn = cuda_device_count or detect_cuda_device_count
    count = int(count_fn())
    if count <= 0:
        return ()
    return tuple(str(index) for index in range(count))


def require_visible_cuda_for_inference(
    *,
    debug_dry_run: bool,
    environ: Mapping[str, str] | None = None,
    cuda_device_count: Callable[[], int] | None = None,
) -> tuple[str, ...]:
    tokens = resolve_visible_cuda_tokens(
        environ=environ,
        cuda_device_count=cuda_device_count,
    )
    if debug_dry_run:
        return tokens
    if not tokens:
        _raise_cuda_unavailable(debug_dry_run=debug_dry_run, visible_cuda_tokens=tokens)
    count_fn = cuda_device_count or detect_cuda_device_count
    if int(count_fn()) <= 0:
        _raise_cuda_unavailable(debug_dry_run=debug_dry_run, visible_cuda_tokens=tokens)
    return tokens


def _raise_cuda_unavailable(
    *,
    debug_dry_run: bool,
    visible_cuda_tokens: Sequence[str],
) -> None:
    raise RuntimeContractError(
        "non-dry inference requires at least one visible CUDA device",
        code="inference.cuda_unavailable",
        context={
            "debug.dry_run": debug_dry_run,
            "visible_cuda_tokens": list(visible_cuda_tokens),
        },
    )


def plan_data_parallel_shards(
    *,
    row_ids: Sequence[str],
    per_device_batch_size: int,
    visible_cuda_tokens: Sequence[str],
) -> DataParallelPlan:
    row_ids_tuple = tuple(str(row_id) for row_id in row_ids)
    visible_tokens = tuple(str(token) for token in visible_cuda_tokens)
    if not row_ids_tuple:
        raise RuntimeContractError(
            "inference input JSONL must contain at least one row",
            code="inference.empty_input_jsonl",
        )
    if per_device_batch_size <= 0:
        raise RuntimeContractError(
            "generation.batch_size must be a positive per-device batch size",
            code="inference.invalid_per_device_batch_size",
            context={"generation.batch_size": per_device_batch_size},
        )
    if not visible_tokens:
        raise RuntimeContractError(
            "cannot plan inference shards without visible CUDA devices",
            code="inference.cuda_unavailable",
            context={"debug.dry_run": False},
        )

    decode_batches = _decode_batches(
        row_ids_tuple,
        per_device_batch_size=per_device_batch_size,
    )
    active_ranks = min(len(visible_tokens), len(decode_batches))
    rank_plans = _rank_plans(
        decode_batches,
        visible_cuda_tokens=visible_tokens,
        active_ranks=active_ranks,
        per_device_batch_size=per_device_batch_size,
    )
    fingerprint_payload = {
        "version": "coordexp-swift-infer-data-parallel-plan-v1",
        "visible_cuda_tokens": list(visible_tokens),
        "active_ranks": active_ranks,
        "per_device_batch_size": per_device_batch_size,
        "row_ids": list(row_ids_tuple),
        "decode_batches": [batch.to_artifact_dict() for batch in decode_batches],
        "ranks": [rank.to_artifact_dict() for rank in rank_plans],
    }
    return DataParallelPlan(
        visible_cuda_tokens=visible_tokens,
        active_ranks=active_ranks,
        per_device_batch_size=per_device_batch_size,
        decode_batch_count=len(decode_batches),
        decode_batches=decode_batches,
        ranks=rank_plans,
        fingerprint=sha256_json(fingerprint_payload),
    )


def sort_rows_by_index(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted((dict(row) for row in rows), key=lambda row: int(row["row_index"]))


def _decode_batches(
    row_ids: tuple[str, ...],
    *,
    per_device_batch_size: int,
) -> tuple[DecodeBatchBlock, ...]:
    batches: list[DecodeBatchBlock] = []
    for start in range(0, len(row_ids), per_device_batch_size):
        end = min(start + per_device_batch_size, len(row_ids))
        batch_id = len(batches)
        batches.append(
            DecodeBatchBlock(
                batch_id=batch_id,
                row_indices=tuple(range(start, end)),
                row_ids=row_ids[start:end],
            )
        )
    return tuple(batches)


def _rank_plans(
    decode_batches: tuple[DecodeBatchBlock, ...],
    *,
    visible_cuda_tokens: tuple[str, ...],
    active_ranks: int,
    per_device_batch_size: int,
) -> tuple[RankShardPlan, ...]:
    plans: list[RankShardPlan] = []
    for rank in range(active_ranks):
        assigned = tuple(
            batch for batch in decode_batches if batch.batch_id % active_ranks == rank
        )
        row_indices = tuple(
            index for batch in assigned for index in batch.row_indices
        )
        row_ids = tuple(row_id for batch in assigned for row_id in batch.row_ids)
        plans.append(
            RankShardPlan(
                rank=rank,
                world_size=active_ranks,
                parent_visible_device_token=visible_cuda_tokens[rank],
                per_device_batch_size=per_device_batch_size,
                batch_ids=tuple(batch.batch_id for batch in assigned),
                row_indices=row_indices,
                row_ids=row_ids,
                shard_dir_name=f"rank-{rank:03d}",
            )
        )
    return tuple(plans)
