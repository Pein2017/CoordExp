#!/usr/bin/env python3
"""Apply one receipt-bound eight-rank Source256 CE or K4-RLOO update."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence

import torch


from .prepare import (  # noqa: E402
    BASE_MODEL,
    EOS_TOKEN_ID,
    K,
    OPTIMIZER_SCHEMA,
    UPDATE_SCHEMA,
    WORLD_SIZE,
    file_sha256,
    json_sha256,
    require,
    validate_plan,
)
from src.config.inference import load_research_infer_config
from src.qwen.native import prepare_replay
from src.losses import aligned_token_logprobs
from .runtime import build_request, load_policy, materialize
from src.adapters.dora import (  # noqa: E402
    select_dora_parameters,
    inspect_dora_adapter_payload,
    save_dora_adapter_payload,
)
from src.runtime.distributed import gather_objects
from src.runtime.model_state import parameter_layout, tensor_state_sha256
from src.qwen.special_token_embeddings import (  # noqa: E402
    inspect_special_token_embedding_delta_payload,
)


LEARNING_RATE = 2.5e-6
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPSILON = 1e-8
MAX_GRAD_NORM = 1.0
EXPECTED_TRAINABLE_TENSORS = 588
EXPECTED_TRAINABLE_SCALARS = 18_006_016


class TrajectoryScorer(torch.nn.Module):
    """Return per-action-token scores from literal native multimodal replay."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, model_inputs, image_grid_thw, prompt_ids, action_ids):
        require(bool(action_ids), "complete action is empty")
        replay = prepare_replay(
            self.model, {**model_inputs, "image_grid_thw": image_grid_thw},
            prompt_token_ids=prompt_ids, continuation_token_ids=action_ids,
        )
        logits = replay.aligned_logits(self.model(**replay.inputs).logits)
        return aligned_token_logprobs(logits, replay.target_ids)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    ).stdout.strip()


def _dist_values(value: Any) -> list[Any]:
    return gather_objects(value)


def _all_true(value: bool, device: torch.device) -> bool:
    import torch.distributed as dist

    flag = torch.tensor(int(value), device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _parameter_layout(named: Sequence[tuple[str, torch.nn.Parameter]]) -> list[dict[str, Any]]:
    return parameter_layout(named)


def _tensor_state_hash(named: Sequence[tuple[str, torch.Tensor]]) -> str:
    return tensor_state_sha256(named)


def _objective_scale(arm: str, *, image_count: int, world_size: int = WORLD_SIZE) -> float:
    require(arm in {"ce", "rloo"} and image_count > 0 and world_size > 0, "invalid objective scale inputs")
    return world_size / image_count


def _action_loss(
    chosen_logprobs: torch.Tensor,
    *,
    arm: str,
    advantage: float | None,
    image_count: int,
    world_size: int = WORLD_SIZE,
) -> torch.Tensor:
    scale = _objective_scale(arm, image_count=image_count, world_size=world_size)
    if arm == "ce":
        return -chosen_logprobs.mean() * scale
    require(advantage is not None, "RLOO action is missing advantage")
    return -float(advantage) * chosen_logprobs.sum() * (scale / K)


def backward_action(ddp, *, model_inputs, grid, prompt_ids, action, arm,
                    image_count, sync_gradients, world_size=WORLD_SIZE):
    """Keep this profile's complete forward/backward pair in one DDP sync scope."""
    with nullcontext() if sync_gradients else ddp.no_sync():
        chosen = ddp(model_inputs, grid, prompt_ids, action["action_token_ids"])
        loss = _action_loss(chosen, arm=arm, advantage=action.get("advantage"),
                            image_count=image_count, world_size=world_size)
        finite = bool(torch.isfinite(chosen).all() and torch.isfinite(loss))
        loss.backward()
    return chosen, loss, finite


def _validate_optimizer_state_document(
    value: Any,
    *,
    arm: str,
    previous_round: int,
    layout: Sequence[Mapping[str, Any]],
    current_adapter_fingerprint: str,
    source_embedding_fingerprint: str,
) -> dict[str, Any]:
    require(isinstance(value, dict), "optimizer state must be a dictionary")
    expected_layout_hash = json_sha256(list(layout))
    require(
        value.get("schema_version") == OPTIMIZER_SCHEMA
        and value.get("arm") == arm
        and value.get("round") == previous_round
        and value.get("optimizer_step") == previous_round
        and value.get("parameter_layout") == list(layout)
        and value.get("parameter_layout_sha256") == expected_layout_hash
        and value.get("saved_adapter_fingerprint") == current_adapter_fingerprint
        and value.get("source_embedding_fingerprint") == source_embedding_fingerprint,
        "optimizer state identity, parameter layout, or round binding changed",
    )
    state_dict = value.get("optimizer_state_dict")
    require(isinstance(state_dict, dict), "optimizer state_dict is missing")
    groups = state_dict.get("param_groups")
    state = state_dict.get("state")
    require(
        isinstance(groups, list)
        and len(groups) == 1
        and groups[0].get("params") == list(range(len(layout)))
        and isinstance(state, dict)
        and set(state) == set(range(len(layout))),
        "optimizer state does not cover the exact ordered trainable surface",
    )
    require(
        groups[0].get("lr") == LEARNING_RATE
        and tuple(groups[0].get("betas", ())) == ADAMW_BETAS
        and groups[0].get("eps") == ADAMW_EPSILON
        and groups[0].get("weight_decay") == 0.0
        and groups[0].get("foreach") is False,
        "optimizer hyperparameters changed",
    )
    for index, item in state.items():
        shape = tuple(layout[index]["shape"])
        dtype_name = str(layout[index]["dtype"])
        require(
            isinstance(item, dict)
            and isinstance(item.get("step"), torch.Tensor)
            and float(item["step"].item()) == previous_round
            and isinstance(item.get("exp_avg"), torch.Tensor)
            and tuple(item["exp_avg"].shape) == shape
            and str(item["exp_avg"].dtype) == dtype_name
            and isinstance(item.get("exp_avg_sq"), torch.Tensor)
            and tuple(item["exp_avg_sq"].shape) == shape
            and str(item["exp_avg_sq"].dtype) == dtype_name,
            f"optimizer moment state is incomplete at parameter {index}",
        )
    return value


def _load_previous_optimizer(
    plan: Mapping[str, Any],
    *,
    named: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any] | None:
    round_index = int(plan["round"])
    if round_index == 1:
        require(plan.get("lineage") is None and not optimizer.state, "round 1 must start with fresh AdamW")
        return None
    lineage = plan.get("lineage")
    require(isinstance(lineage, Mapping), "round 2+ is missing lineage")
    state_ref = lineage.get("optimizer_state", {})
    state_path = Path(str(state_ref.get("path"))).expanduser().resolve(strict=True)
    require(file_sha256(state_path) == state_ref.get("sha256"), "previous optimizer state changed")
    document = torch.load(state_path, map_location="cpu", weights_only=True)
    layout = _parameter_layout(named)
    _validate_optimizer_state_document(
        document,
        arm=str(plan["arm"]),
        previous_round=round_index - 1,
        layout=layout,
        current_adapter_fingerprint=str(plan["model"]["current_adapter"]["fingerprint"]),
        source_embedding_fingerprint=str(plan["model"]["source_embedding"]["fingerprint"]),
    )
    require(lineage.get("parameter_layout_sha256") == json_sha256(layout), "plan lineage records a different parameter layout")
    optimizer.load_state_dict(document["optimizer_state_dict"])
    return document


def _optimizer_step(optimizer: torch.optim.Optimizer) -> int:
    steps = {
        int(float(state["step"].item()))
        for state in optimizer.state.values()
        if isinstance(state, Mapping) and isinstance(state.get("step"), torch.Tensor)
    }
    require(len(steps) == 1, "AdamW parameters do not share one step counter")
    return next(iter(steps))


def _save_adapter_only(model: Any, *, source_root: Path, output: Path) -> dict[str, Any]:
    return save_dora_adapter_payload(
        model,
        source_root=source_root,
        output=output,
        expected_base_model_path=BASE_MODEL,
        expected_tensor_count=EXPECTED_TRAINABLE_TENSORS,
    )


def _materialize_group(*, qwen, frontend, config, raw, group):
    request, image, _ = build_request(raw, config=config, qwen=frontend.qwen)
    batch = materialize(qwen, request)
    prompt_ids = list(batch.prompt_token_ids[0])
    require(prompt_ids == group["prompt_token_ids"]
            and json_sha256(prompt_ids) == group["prompt_token_ids_sha256"],
            f"live prompt differs from plan: {group['example_id']}")
    require(image.image_content_sha256 == group["image_content_sha256"]
            and Path(image.image_path).resolve() == Path(group["image_path"]).resolve(),
            f"live image differs from plan: {group['example_id']}")
    expected_grid = group.get("observed_image_grid_thw", group.get("expected_image_grid_thw"))
    require(batch.image_grids[0] is not None and list(batch.image_grids[0]) == expected_grid,
            "live image grid differs from plan")
    if "executed_media_sha256" in group:
        require(batch.media_sha256[0] == group["executed_media_sha256"],
                "live executed media differs from sampled bank")
    grid = batch.inputs.get("image_grid_thw")
    require(isinstance(grid, torch.Tensor), "live native inputs are missing image_grid_thw")
    return prompt_ids, dict(batch.inputs), grid


def _publish(
    *,
    output_root: Path,
    plan: Mapping[str, Any],
    plan_path: Path,
    model: torch.nn.Module,
    named: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer: torch.optim.Optimizer,
    backend_receipt: Mapping[str, Any],
    per_rank: Sequence[Mapping[str, Any]],
    adapter_before: str,
    adapter_after: str,
    gradient_hash: str,
    raw_grad_norm: float,
    clipped_grad_norm: float,
    start: float,
    device: torch.device,
) -> dict[str, Any]:
    temp = output_root.with_name(f".{output_root.name}.tmp-{os.getpid()}")
    require(not output_root.exists() and not temp.exists(), "output or atomic temporary root already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=False)
    try:
        adapter_temp = temp / "adapter"
        saved_temp = _save_adapter_only(
            model,
            source_root=Path(plan["model"]["current_adapter"]["root"]),
            output=adapter_temp,
        )
        layout = _parameter_layout(named)
        layout_hash = json_sha256(layout)
        round_index = int(plan["round"])
        require(_optimizer_step(optimizer) == round_index, "AdamW step counter differs from completed round")
        state_document = {
            "schema_version": OPTIMIZER_SCHEMA,
            "arm": plan["arm"],
            "round": round_index,
            "optimizer_step": round_index,
            "parameter_layout": layout,
            "parameter_layout_sha256": layout_hash,
            "prior_adapter_fingerprint": plan["model"]["current_adapter"]["fingerprint"],
            "saved_adapter_fingerprint": saved_temp["fingerprint"],
            "source_embedding_fingerprint": plan["model"]["source_embedding"]["fingerprint"],
            "plan_content_sha256": plan["content_sha256"],
            "optimizer_state_dict": optimizer.state_dict(),
        }
        state_path = temp / "optimizer.pt"
        torch.save(state_document, state_path)
        roundtrip = torch.load(state_path, map_location="cpu", weights_only=True)
        _validate_optimizer_state_document(
            roundtrip,
            arm=str(plan["arm"]),
            previous_round=round_index,
            layout=layout,
            current_adapter_fingerprint=saved_temp["fingerprint"],
            source_embedding_fingerprint=plan["model"]["source_embedding"]["fingerprint"],
        )
        saved = dict(saved_temp)
        saved["root"] = str(output_root / "adapter")
        receipt = {
            "schema_version": UPDATE_SCHEMA,
            "mechanical_status": "MECHANICALLY_VALID",
            "arm": plan["arm"],
            "round": round_index,
            "git_commit": _git_commit(),
            "plan": {"path": str(plan_path), "sha256": file_sha256(plan_path), "content_sha256": plan["content_sha256"]},
            "composition": {
                "base_model_path": plan["model"]["base_model_path"],
                "prior_adapter": plan["model"]["current_adapter"],
                "loaded_policy": backend_receipt,
                "native_execution": {"operation": "prepare_replay", "gradients": True, "model_mode": "eval"},
                "dtype": "fp32",
                "attention_implementation": "sdpa",
                "unmerged": True,
            },
            "source_embedding": plan["model"]["source_embedding"],
            "objective": plan["objective"],
            "population": {
                key: plan["population"][key]
                for key in ("mode", "image_count", "world_size", "images_per_rank", "k", "actions_per_image")
            },
            "distributed": {"backend": "nccl", "ddp_rank_averaging": True, "per_rank": list(per_rank)},
            "optimizer": {
                "name": "torch.optim.AdamW",
                "learning_rate": LEARNING_RATE,
                "betas": list(ADAMW_BETAS),
                "epsilon": ADAMW_EPSILON,
                "weight_decay": 0.0,
                "scheduler": "constant",
                "max_grad_norm": MAX_GRAD_NORM,
                "raw_gradient_norm": raw_grad_norm,
                "clipped_gradient_norm": clipped_grad_norm,
                "step_count": round_index,
                "persistent_from_previous_round": round_index > 1,
            },
            "trainable_surface": {
                "tensor_count": len(named),
                "scalar_count": sum(parameter.numel() for _, parameter in named),
                "parameter_layout_sha256": layout_hash,
                "gradient_sha256": gradient_hash,
                "adapter_state_sha256_before": adapter_before,
                "adapter_state_sha256_after": adapter_after,
            },
            "saved_adapter": saved,
            "optimizer_state": {"path": str(output_root / "optimizer.pt"), "sha256": file_sha256(state_path)},
            "runtime": {
                "forward_count": sum(int(item["forward_count"]) for item in per_rank),
                "backward_count": sum(int(item["backward_count"]) for item in per_rank),
                "action_token_count": sum(int(item["action_token_count"]) for item in per_rank),
                "peak_cuda_memory_allocated_bytes_rank0": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_memory_reserved_bytes_rank0": int(torch.cuda.max_memory_reserved(device)),
                "peak_cuda_memory_allocated_bytes_per_rank_max": max(int(item["peak_cuda_memory_allocated_bytes"]) for item in per_rank),
                "peak_cuda_memory_reserved_bytes_per_rank_max": max(int(item["peak_cuda_memory_reserved_bytes"]) for item in per_rank),
                "peak_host_rss_bytes_rank0": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                "peak_host_rss_bytes_per_rank_max": max(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024), *(int(item["peak_host_rss_bytes"]) for item in per_rank)),
                "elapsed_seconds_rank0": time.monotonic() - start,
            },
        }
        (temp / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        os.rename(temp, output_root)
        return receipt
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def run_update(args: argparse.Namespace) -> dict[str, Any] | None:
    plan_path = args.plan.expanduser().resolve(strict=True)
    plan = validate_plan(plan_path)
    output_root = args.output_root.expanduser().resolve()
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    require(torch.cuda.is_available(), "Source256 update requires CUDA")
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    require(world_size == WORLD_SIZE and 0 <= rank < WORLD_SIZE and 0 <= local_rank < torch.cuda.device_count(), "torchrun topology must be exactly eight CUDA ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=device)
    start = time.monotonic()
    try:
        plan_identity = (file_sha256(plan_path), plan["content_sha256"])
        require(len(set(_dist_values(plan_identity))) == 1, "ranks loaded different update plans")
        require(_all_true(not output_root.exists(), device), "output root exists on at least one rank")

        from src.config.fingerprint import sha256_json as config_json_sha256
        from src.data import load_raw_examples
        from src.inference.runtime import assemble_frontend

        resolved = load_research_infer_config(Path(plan["sources"]["infer_config"]["path"]))
        config = resolved.config
        require(
            resolved.fingerprint == plan["sources"]["infer_config"]["resolved_fingerprint"]
            and Path(config.data.input_jsonl).resolve() == Path(plan["sources"]["train_jsonl"]["path"]).resolve()
            and config.backend.type == "hf"
            and config.backend.hf.attn_implementation == "sdpa"
            and config.model.dtype == "fp32"
            and config.adapter is not None
            and Path(config.adapter.path).resolve() == Path(plan["model"]["current_adapter"]["root"]).resolve()
            and config.embedding_delta is not None
            and Path(config.embedding_delta.path).resolve() == Path(plan["model"]["source_embedding"]["root"]).resolve(),
            "live inference config differs from the plan",
        )
        frontend = assemble_frontend(config, generation_config_fingerprint=config_json_sha256(config.generation.model_dump(mode="json")))
        raw_examples = list(load_raw_examples(config.data.input_jsonl))
        raw_by_example = {str(raw.example_id): raw for raw in raw_examples}
        assignment = plan["population"]["assignments"][rank]
        groups_by_example = {group["example_id"]: group for group in plan["population"]["groups"]}
        groups = [groups_by_example[example_id] for example_id in assignment["example_ids"]]
        require(len(groups) == assignment["image_count"], "rank assignment changed")

        torch.cuda.reset_peak_memory_stats(device)
        qwen, backend_receipt = load_policy(config, device=device)
        require(qwen.token_identity.im_end_token_ids == (EOS_TOKEN_ID,), "live im_end token ID changed")
        model = qwen.model
        require(model is not None, "HF backend has no live model")
        model.eval()
        identity = backend_receipt["model_identity"]
        adapter_receipt = identity["adapter"]
        require(
            Path(identity["base"]["path"]).resolve() == BASE_MODEL.resolve()
            and Path(adapter_receipt["adapter_path"]).resolve() == Path(plan["model"]["current_adapter"]["root"]).resolve()
            and adapter_receipt["merged_adapters"] == []
            and adapter_receipt["adapter_state_evidence"].get("state_checked") is True
            and backend_receipt["effective_settings"]["observed_attn_implementation"] == "sdpa"
            and set(backend_receipt["effective_settings"]["observed_model_dtype"]["parameter_dtype_names"]) == {"torch.float32"},
            "live model composition differs from the planned FP32/SDPA unmerged adapter",
        )
        live_embedding = inspect_special_token_embedding_delta_payload(identity["embedding_delta"]["identity"]["delta_path"], BASE_MODEL)
        require(live_embedding["fingerprint"] == plan["model"]["source_embedding"]["fingerprint"], "live Source embedding changed")

        all_named = tuple(model.named_parameters())
        for _, parameter in all_named:
            parameter.requires_grad_(False)
        named = select_dora_parameters(model, towers=("language",), adapter_name="default")
        require(
            len(named) == EXPECTED_TRAINABLE_TENSORS
            and sum(parameter.numel() for _, parameter in named) == EXPECTED_TRAINABLE_SCALARS
            and all("language_model" in name for name, _ in named)
            and not any(
                fragment in name
                for name, _ in named
                for fragment in ("visual", "vision", "merger", "embed_tokens", "lm_head")
            ),
            "live language DoRA trainable surface changed",
        )
        for _, parameter in named:
            parameter.requires_grad_(True)
        adapter_before = _tensor_state_hash([(name, parameter) for name, parameter in named])
        require(len(set(_dist_values(adapter_before))) == 1, "adapter state differs across ranks")

        scorer = TrajectoryScorer(model)
        ddp = DDP(scorer, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, init_sync=False)
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in named],
            lr=LEARNING_RATE,
            betas=ADAMW_BETAS,
            eps=ADAMW_EPSILON,
            weight_decay=0.0,
            foreach=False,
        )
        previous = _load_previous_optimizer(plan, named=named, optimizer=optimizer)
        previous_identity = None if previous is None else (previous["parameter_layout_sha256"], previous["optimizer_step"])
        require(len(set(_dist_values(previous_identity))) == 1, "ranks loaded different optimizer continuation state")
        optimizer.zero_grad(set_to_none=True)

        expected_actions = int(assignment["action_count"])
        action_index = 0
        local_losses: list[float] = []
        local_logprobs: list[float] = []
        local_finite = True
        action_token_count = 0
        for group in groups:
            require(group["example_id"] in raw_by_example, "planned image is absent from live input")
            prompt_ids, model_inputs, grid = _materialize_group(
                qwen=qwen,
                frontend=frontend,
                config=config,
                raw=raw_by_example[group["example_id"]],
                group=group,
            )
            for action in group["actions"]:
                sync = action_index == expected_actions - 1
                chosen, loss, finite = backward_action(
                    ddp, model_inputs=model_inputs, grid=grid, prompt_ids=prompt_ids,
                    action=action, arm=plan["arm"], image_count=plan["population"]["image_count"],
                    sync_gradients=sync,
                )
                local_finite = local_finite and finite
                local_logprobs.append(float(chosen.detach().sum()))
                local_losses.append(float(loss.detach()))
                action_token_count += len(action["action_token_ids"])
                action_index += 1
            del prompt_ids, model_inputs, grid
        require(action_index == expected_actions and expected_actions > 0, "rank action execution count changed")
        require(_all_true(local_finite, device), "nonfinite action logprob or loss")
        require(all(parameter.grad is not None for _, parameter in named), "DoRA gradient is missing")
        require(_all_true(all(bool(torch.isfinite(parameter.grad).all()) for _, parameter in named), device), "nonfinite reduced gradient")
        gradient_hash = _tensor_state_hash([(name, parameter.grad) for name, parameter in named if parameter.grad is not None])
        require(len(set(_dist_values(gradient_hash))) == 1, "DDP gradients differ across ranks")
        raw_grad_norm = float(torch.nn.utils.clip_grad_norm_([parameter for _, parameter in named], MAX_GRAD_NORM, error_if_nonfinite=True))
        clipped_grad_norm = math.sqrt(sum(float(parameter.grad.detach().double().square().sum()) for _, parameter in named if parameter.grad is not None))
        optimizer.step()
        require(len(optimizer.state) == EXPECTED_TRAINABLE_TENSORS and _optimizer_step(optimizer) == plan["round"], "AdamW state did not advance exactly one round")
        adapter_after = _tensor_state_hash([(name, parameter) for name, parameter in named])
        require(len(set(_dist_values(adapter_after))) == 1, "post-update adapter differs across ranks")
        local = {
            "rank": rank,
            "image_ids": assignment["image_ids"],
            "image_count": len(groups),
            "forward_count": action_index,
            "backward_count": action_index,
            "action_token_count": action_token_count,
            "loss_sum_scaled_for_ddp": sum(local_losses),
            "action_logprob_sum": sum(local_logprobs),
            "synchronized_backward_count": 1,
            "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "peak_host_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        }
        per_rank = _dist_values(local)
        publication: dict[str, Any] = {"ok": True}
        if rank == 0:
            try:
                _publish(
                    output_root=output_root,
                    plan=plan,
                    plan_path=plan_path,
                    model=model,
                    named=named,
                    optimizer=optimizer,
                    backend_receipt=backend_receipt,
                    per_rank=per_rank,
                    adapter_before=adapter_before,
                    adapter_after=adapter_after,
                    gradient_hash=gradient_hash,
                    raw_grad_norm=raw_grad_norm,
                    clipped_grad_norm=clipped_grad_norm,
                    start=start,
                    device=device,
                )
            except Exception as exc:
                publication = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        publication = _dist_values(publication if rank == 0 else None)[0]
        require(publication["ok"], f"rank-zero atomic publication failed: {publication.get('error')}")
        dist.barrier()
        if rank == 0:
            result = json.loads((output_root / "receipt.json").read_text(encoding="utf-8"))
            print(json.dumps({"mechanical_status": result["mechanical_status"], "receipt": str(output_root / "receipt.json")}, sort_keys=True))
            return result
        return None
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        gc.collect()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    return result


def main() -> int:
    run_update(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
