"""Private data-parallel inference worker entrypoint and launch helpers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

from src.common.errors import RuntimeContractError
from src.config.inference import InferConfig, ResolvedInferConfig
from src.config.models import ConfigSource, PathOrigin
from src.inference import pipeline
from src.inference.data_parallel import (
    DataParallelPlan,
    DecodeBatchBlock,
    RankShardPlan,
)
from src.inference.execution_model import load_execution_model_receipt


WORKER_MODULE = "src.inference.worker"


def build_worker_environment(
    *,
    base_env: Mapping[str, str] | None,
    parent_visible_device_token: str,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env["CUDA_VISIBLE_DEVICES"] = str(parent_visible_device_token)
    return env


def validate_worker_cuda_environment(
    *,
    rank: int,
    world_size: int,
    parent_visible_device_token: str,
    environ: Mapping[str, str] | None = None,
    torch_module: Any = torch,
) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    worker_cuda_visible_devices = str(env.get("CUDA_VISIBLE_DEVICES", ""))
    _validate_worker_identity(
        rank=rank,
        world_size=world_size,
        parent_visible_device_token=parent_visible_device_token,
    )
    cuda = torch_module.cuda
    cuda_available = bool(cuda.is_available())
    cuda_device_count = int(cuda.device_count())
    cuda_current_device = int(cuda.current_device()) if cuda_device_count > 0 else None
    if (
        not cuda_available
        or cuda_device_count != 1
        or cuda_current_device != 0
        or worker_cuda_visible_devices != str(parent_visible_device_token)
    ):
        raise RuntimeContractError(
            "worker must see exactly one CUDA device bound as logical cuda:0",
            code="inference.worker_cuda_binding_invalid",
            context={
                "rank": rank,
                "world_size": world_size,
                "parent_visible_device_token": str(parent_visible_device_token),
                "worker_cuda_visible_devices": worker_cuda_visible_devices,
                "cuda_available": cuda_available,
                "cuda_device_count": cuda_device_count,
                "cuda_current_device": cuda_current_device,
            },
        )
    return {
        "rank": rank,
        "world_size": world_size,
        "parent_visible_device_token": str(parent_visible_device_token),
        "worker_cuda_visible_devices": worker_cuda_visible_devices,
        "cuda_device_count": cuda_device_count,
        "cuda_current_device": cuda_current_device,
        "logical_device": "cuda:0",
    }


def build_worker_runtime_metadata(
    *,
    rank: int,
    world_size: int,
    parent_visible_device_token: str,
    environ: Mapping[str, str] | None = None,
    torch_module: Any = torch,
    model: Any | None = None,
) -> dict[str, Any]:
    metadata = validate_worker_cuda_environment(
        rank=rank,
        world_size=world_size,
        parent_visible_device_token=parent_visible_device_token,
        environ=environ,
        torch_module=torch_module,
    )
    metadata["model_first_parameter_device"] = _model_first_parameter_device(model)
    return metadata


def _validate_worker_identity(
    *,
    rank: int,
    world_size: int,
    parent_visible_device_token: str,
) -> None:
    token = str(parent_visible_device_token)
    if world_size <= 0 or rank < 0 or rank >= world_size or token in {"", "-1"} or "," in token:
        raise RuntimeContractError(
            "worker rank/world/token identity is invalid",
            code="inference.worker_identity_invalid",
            context={
                "rank": rank,
                "world_size": world_size,
                "parent_visible_device_token": token,
            },
        )


def launch_worker_subprocess(
    *,
    rank: int,
    world_size: int,
    parent_visible_device_token: str,
    resolved_config_json: str | Path,
    shard_plan_json: str | Path,
    output_dir: str | Path,
    execution_model_json: str | Path | None = None,
    base_env: Mapping[str, str] | None = None,
) -> subprocess.Popen[Any]:
    env = build_worker_environment(
        base_env=base_env,
        parent_visible_device_token=parent_visible_device_token,
    )
    return subprocess.Popen(
        build_worker_command(
            rank=rank,
            world_size=world_size,
            parent_visible_device_token=parent_visible_device_token,
            resolved_config_json=resolved_config_json,
            shard_plan_json=shard_plan_json,
            output_dir=output_dir,
            execution_model_json=execution_model_json,
        ),
        env=env,
    )


def build_worker_command(
    *,
    rank: int,
    world_size: int,
    parent_visible_device_token: str,
    resolved_config_json: str | Path,
    shard_plan_json: str | Path,
    output_dir: str | Path,
    execution_model_json: str | Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        WORKER_MODULE,
        "--rank",
        str(rank),
        "--world-size",
        str(world_size),
        "--parent-visible-device-token",
        str(parent_visible_device_token),
        "--resolved-config-json",
        str(resolved_config_json),
        "--shard-plan-json",
        str(shard_plan_json),
        "--output-dir",
        str(output_dir),
    ]
    if execution_model_json is not None:
        command.extend(["--execution-model-json", str(execution_model_json)])
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m {WORKER_MODULE}")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--parent-visible-device-token", required=True)
    parser.add_argument("--resolved-config-json", required=True)
    parser.add_argument("--shard-plan-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-model-json")
    args = parser.parse_args(argv)
    resolved = load_resolved_infer_config_artifact(args.resolved_config_json)
    plan = load_data_parallel_plan_artifact(args.shard_plan_json)
    rank_plan = _rank_plan_for(plan=plan, rank=args.rank)
    if rank_plan.world_size != args.world_size:
        raise RuntimeContractError(
            "worker CLI world size disagrees with shard plan",
            code="inference.worker_plan_mismatch",
            context={
                "rank": args.rank,
                "cli_world_size": args.world_size,
                "plan_world_size": rank_plan.world_size,
            },
        )
    if str(rank_plan.parent_visible_device_token) != str(args.parent_visible_device_token):
        raise RuntimeContractError(
            "worker CLI device token disagrees with shard plan",
            code="inference.worker_plan_mismatch",
            context={
                "rank": args.rank,
                "cli_parent_visible_device_token": args.parent_visible_device_token,
                "plan_parent_visible_device_token": rank_plan.parent_visible_device_token,
            },
        )
    runtime_metadata = build_worker_runtime_metadata(
        rank=args.rank,
        world_size=args.world_size,
        parent_visible_device_token=args.parent_visible_device_token,
    )
    execution_model = (
        None
        if args.execution_model_json is None
        else load_execution_model_receipt(args.execution_model_json)
    )
    pipeline.run_shard(
        resolved=resolved,
        output_dir=Path(args.output_dir),
        row_indices=rank_plan.row_indices,
        worker_metadata={
            "shard_plan_fingerprint": plan.fingerprint,
            "rank": runtime_metadata["rank"],
            "world_size": runtime_metadata["world_size"],
            "parent_visible_device_token": runtime_metadata[
                "parent_visible_device_token"
            ],
            "worker_cuda_visible_devices": runtime_metadata[
                "worker_cuda_visible_devices"
            ],
            "worker_logical_device": runtime_metadata["logical_device"],
            "cuda_device_count": runtime_metadata["cuda_device_count"],
            "cuda_current_device": runtime_metadata["cuda_current_device"],
            "model_first_parameter_device": runtime_metadata.get(
                "model_first_parameter_device"
            ),
            "per_device_batch_size": rank_plan.per_device_batch_size,
            "batch_ids": list(rank_plan.batch_ids),
        },
        rank_plan=rank_plan,
        execution_model=execution_model,
    )
    return 0


def load_resolved_infer_config_artifact(path: str | Path) -> ResolvedInferConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config_dict = dict(payload["config"])
    resolution = dict(payload["resolution"])
    sources = tuple(
        ConfigSource(path=Path(source["path"]), sha256=str(source["sha256"]))
        for source in resolution.get("sources", [])
    )
    path_origins = {
        field: PathOrigin(
            field=field,
            declared_path=str(origin["declared_path"]),
            declaring_config_path=Path(origin["declaring_config_path"]),
            resolved_path=Path(origin["resolved_path"]),
        )
        for field, origin in dict(resolution.get("path_origins", {})).items()
    }
    return ResolvedInferConfig(
        config=InferConfig.model_validate(config_dict),
        config_dict=config_dict,
        fingerprint=str(resolution["fingerprint"]),
        schema_version=int(resolution["schema_version"]),
        loader_version=str(resolution["loader_version"]),
        entry_config_path=Path(resolution["entry_config_path"]),
        sources=sources,
        path_origins=path_origins,
    )


def load_data_parallel_plan_artifact(path: str | Path) -> DataParallelPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    decode_batches = tuple(
        DecodeBatchBlock(
            batch_id=int(batch["batch_id"]),
            row_indices=tuple(int(index) for index in batch["row_indices"]),
            row_ids=tuple(str(row_id) for row_id in batch["row_ids"]),
        )
        for batch in payload["decode_batches"]
    )
    ranks = tuple(
        RankShardPlan(
            rank=int(rank["rank"]),
            world_size=int(rank["world_size"]),
            parent_visible_device_token=str(rank["parent_visible_device_token"]),
            per_device_batch_size=int(rank["per_device_batch_size"]),
            batch_ids=tuple(int(batch_id) for batch_id in rank["batch_ids"]),
            row_indices=tuple(int(index) for index in rank["row_indices"]),
            row_ids=tuple(str(row_id) for row_id in rank["row_ids"]),
            shard_dir_name=str(rank["shard_dir_name"]),
        )
        for rank in payload["ranks"]
    )
    return DataParallelPlan(
        visible_cuda_tokens=tuple(str(token) for token in payload["visible_cuda_tokens"]),
        active_ranks=int(payload["active_ranks"]),
        per_device_batch_size=int(payload["per_device_batch_size"]),
        decode_batch_count=int(payload["decode_batch_count"]),
        decode_batches=decode_batches,
        ranks=ranks,
        fingerprint=str(payload["fingerprint"]),
    )


def _rank_plan_for(*, plan: DataParallelPlan, rank: int) -> RankShardPlan:
    for rank_plan in plan.ranks:
        if rank_plan.rank == rank:
            return rank_plan
    raise RuntimeContractError(
        "worker rank is not present in shard plan",
        code="inference.worker_plan_mismatch",
        context={
            "rank": rank,
            "planned_ranks": [rank_plan.rank for rank_plan in plan.ranks],
        },
    )


def _model_first_parameter_device(model: Any | None) -> str | None:
    if model is None:
        return None
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    first_param = next(iter(parameters()), None)
    device = getattr(first_param, "device", None)
    return None if device is None else str(device)


if __name__ == "__main__":
    raise SystemExit(main())
