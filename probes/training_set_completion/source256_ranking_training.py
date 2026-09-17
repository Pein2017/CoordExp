"""Small direct training backend for the Source256 output-ranking repair.

The data producer owns the 256-route preparation, fifteen observed pairs and
the fixed schedule.  This module owns only the model calls, loss accounting,
fresh AdamW step, checkpoint, and the 30-route frozen reference cache.  It
uses the shared completion replay implementation; its ranking objective and
reference-cache contract remain separate from the Source256 paired CE runner.
"""

from __future__ import annotations

from probes.training_set_completion import replay

import argparse
import copy
from datetime import timedelta
import json
import math
import os
import random
import signal
import socket
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from probes.training_set_completion import distributed
from probes.training_set_completion import source256_ranking_data as data_source
from probes.training_set_completion import source256_training as source256
from probes.training_set_completion import training
from src.losses import aligned_token_logprobs


SCHEMA = "training_set_completion.source256_ranking_training.v1"
REFERENCE_SCHEMA = f"{SCHEMA}.reference_cache.v1"
LIKELIHOOD_SCHEMA = f"{SCHEMA}.likelihood.v1"
MANIFEST_SCHEMA = "source256.output_ranking_repair.v1.manifest"
ROOT = data_source.ROOT
EOS = data_source.EOS
REQUIRED_WORLD_SIZE = 4
GLOBAL_BRANCH_IMAGES = 32
GLOBAL_PRESENTATIONS = 64
PAIR_COUNT = 15
UPDATE_COUNT = 16
MICROBATCH_SIZE = 2
GEOMETRY_WEIGHT = 0.01
RANKING_DENOMINATOR = "max_recorded_action_lengths"
REFERENCE_NAME = "frozen_starting_Bnormalized64"
ARMS = ("P", "R")

# The parent requested these names as the narrow data boundary.
load_data = data_source.load_data
read = data_source.read


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _binding(value: Any, *, name: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{name} binding")
    require(set(value) == {"path", "sha256", "size_bytes"}, f"{name} binding fields")
    require(
        isinstance(value["path"], str)
        and isinstance(value["sha256"], str)
        and len(value["sha256"]) == 64
        and type(value["size_bytes"]) is int
        and value["size_bytes"] >= 0,
        f"{name} binding values",
    )
    return dict(value)


def _verify_binding(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    checked = _binding(value, name=name)
    require(training.binding(checked["path"]) == checked, f"{name} bytes changed")
    return checked


def _model_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return model identity while excluding only the run output directory."""

    result = copy.deepcopy(dict(config))
    result.pop("run", None)
    return result


def _validate_schedule(data: Mapping[str, Any]) -> None:
    canonical = data["canonical_routes"]
    pairs = data["pairs"]
    schedule = data["schedule"]
    require(isinstance(schedule, list) and len(schedule) == UPDATE_COUNT, "sixteen ranking updates")
    common_counts: dict[str, int] = {str(key): 0 for key in canonical}
    pair_counts: dict[str, int] = {str(key): 0 for key in pairs}
    for step, update in enumerate(schedule, 1):
        require(
            isinstance(update, Mapping)
            and set(update) == {"step", "common_image_ids", "pair_image_ids"},
            f"ranking schedule step {step} fields",
        )
        require(update["step"] == step, "ranking schedule steps")
        common = update["common_image_ids"]
        pair = update["pair_image_ids"]
        require(
            isinstance(common, list)
            and isinstance(pair, list)
            and len(common) == len(pair) == GLOBAL_BRANCH_IMAGES,
            f"ranking schedule step {step} batch",
        )
        for values, known, counts, name in (
            (common, canonical, common_counts, "common"),
            (pair, pairs, pair_counts, "pair"),
        ):
            require(all(type(image_id) is int and str(image_id) in known for image_id in values), f"unknown {name} image")
            for rank in range(REQUIRED_WORLD_SIZE):
                chunk = values[rank * 8 : (rank + 1) * 8]
                # NativeRequest IDs are the bound row IDs.  The frozen cyclic
                # schedule keeps each rank's microbatches request-unique.
                require(len(set(chunk)) == 8, f"{name} rank chunk repeats a request")
            for image_id in values:
                counts[str(image_id)] += 1
    require(set(common_counts.values()) == {2}, "canonical schedule is not two passes")
    require(sum(pair_counts.values()) == 512 and set(pair_counts.values()) <= {34, 35}, "pair schedule count")
    require(sum(value == 35 for value in pair_counts.values()) == 2, "pair schedule 34/35 balance")


def _validate_data(data: Mapping[str, Any]) -> dict[str, Any]:
    require(data.get("schema") == data_source.SCHEMA + ".data", "ranking data schema")
    canonical = data.get("canonical_routes")
    pairs = data.get("pairs")
    require(isinstance(canonical, Mapping) and len(canonical) == 256, "256 canonical routes")
    require(isinstance(pairs, Mapping) and len(pairs) == PAIR_COUNT, "15 observed pairs")
    for key, route in canonical.items():
        require(str(key) == str(route.get("image_id")), "canonical image identity")
        require(route.get("ce_weights") == [1] * len(route["continuation_token_ids"]), "canonical CE mask")
        require(route["continuation_token_ids"][-1] == EOS, "canonical EOS")
    for key, pair in pairs.items():
        require(set(pair) == {"preferred", "rejected", "denominator", "preferred_tokens", "rejected_tokens"}, "pair fields")
        preferred, rejected = pair["preferred"], pair["rejected"]
        require(str(key) == str(preferred["image_id"]) == str(rejected["image_id"]), "pair image identity")
        require(preferred["prompt_token_ids"] == rejected["prompt_token_ids"], "pair prompt identity")
        require(preferred["image_identity"] == rejected["image_identity"], "pair media identity")
        require(preferred["ce_weights"] == [1] * len(preferred["continuation_token_ids"]), "preferred CE mask")
        require(rejected["ce_weights"] == [1] * len(rejected["continuation_token_ids"]), "rejected CE mask")
        require(preferred["continuation_token_ids"][-1] == EOS, "preferred EOS")
        require(pair["denominator"] == max(pair["preferred_tokens"], pair["rejected_tokens"]), "pair length denominator")
        require(pair["preferred_tokens"] == len(preferred["continuation_token_ids"]), "preferred length")
        require(pair["rejected_tokens"] == len(rejected["continuation_token_ids"]), "rejected length")
        require(0 < pair["rejected_tokens"] <= 3084, "rejected cap")
        require(not rejected["trusted_boxes"], "rejected geometry must be empty")
        require(rejected["continuation_token_ids"][-1] == EOS or pair["rejected_tokens"] == 3084, "rejected EOS/cap identity")
    _validate_schedule(data)
    return dict(data)


def validate_manifest(value: Mapping[str, Any], *, verify_sources: bool = True) -> dict[str, Any]:
    """Validate the frozen ranking manifest without importing the old validator."""

    required = {
        "schema",
        "arm",
        "mode",
        "data",
        "preparation",
        "source_adapter",
        "model_config",
        "optimizer",
        "scheduler",
        "validity_hinge",
        "runtime",
        "reference_cache_path",
        "ranking",
        "sources",
    }
    require(set(value) == required, "ranking manifest fields")
    require(value["schema"] == MANIFEST_SCHEMA, "ranking manifest schema")
    require(value["arm"] in ARMS and value["mode"] in ("qualification", "main"), "ranking arm/mode")
    data_binding = _binding(value["data"], name="ranking data")
    preparation = _binding(value["preparation"], name="shared preparation")
    require(preparation["path"] == str(data_source.PREPARATION), "shared preparation path")
    require(preparation["sha256"] == "9b94baeb0699e479413483cba5ee6fb4b4d98c062aebb0ba462ae61a871cf242", "shared preparation digest")
    sources = value["sources"]
    require(isinstance(sources, Mapping) and set(sources) == {"data_producer", "paired_embeddings"}, "ranking source bindings")
    if verify_sources:
        _verify_binding(data_binding, name="ranking data")
        _verify_binding(preparation, name="shared preparation")
        _verify_binding(sources["data_producer"], name="ranking data producer")
        _verify_binding(sources["paired_embeddings"], name="paired embeddings")
        require(Path(sources["data_producer"]["path"]).resolve() == Path(data_source.__file__).resolve(), "ranking producer path")
    adapter = value["source_adapter"]
    require(isinstance(adapter, Mapping) and isinstance(adapter.get("root"), str), "ranking source adapter")
    require(adapter.get("semantic_identity", {}).get("tensor_key_count") == 588, "ranking DoRA tensor count")
    source256.source_adapter_scalar_count(adapter)
    if verify_sources:
        for entry in adapter.get("files", []):
            require(training.binding(Path(adapter["root"]) / entry["relative_path"]) == {
                "path": str((Path(adapter["root"]) / entry["relative_path"]).resolve()),
                "sha256": entry["sha256"],
                "size_bytes": entry["size_bytes"],
            }, "ranking adapter bytes changed")
    config = value["model_config"]
    require(
        isinstance(config, Mapping)
        and config.get("backend", {}).get("type") == "hf"
        and config.get("backend", {}).get("hf", {}).get("attn_implementation") == "sdpa"
        and config.get("model", {}).get("dtype") == "fp32"
        and config.get("adapter", {}).get("path") == adapter.get("root"),
        "ranking FP32/SDPA model identity",
    )
    optimizer = value["optimizer"]
    require(optimizer == training.DEFAULT_OPTIMIZER, "ranking AdamW recipe")
    scheduler = value["scheduler"]
    require(
        scheduler == {"type": "cosine", "total_updates": 16, "min_lr_ratio": 0.0, "warmup_updates": 0},
        "ranking cosine recipe",
    )
    hinge = value["validity_hinge"]
    require(hinge.get("weight") == GEOMETRY_WEIGHT and hinge.get("margin") == 1 / 999, "ranking geometry recipe")
    runtime = value["runtime"]
    updates = 1 if value["mode"] == "qualification" else UPDATE_COUNT
    expected_calls = updates * (32 if value["arm"] == "P" else 48)
    expected_forwards = updates * (64 if value["arm"] == "P" else 96)
    require(
        runtime.get("seed") == 19
        and runtime.get("updates") == updates
        and runtime.get("world_size") == REQUIRED_WORLD_SIZE
        and runtime.get("microbatch_size") == MICROBATCH_SIZE
        and runtime.get("effective_image_batch") == 64
        and runtime.get("branch_image_count") == GLOBAL_BRANCH_IMAGES
        and runtime.get("gradient_clip_norm") == 1.0
        and runtime.get("activation_checkpointing") is True
        and runtime.get("fresh_optimizer") is True
        and runtime.get("eos_token_id") == EOS
        and runtime.get("max_model_calls") == expected_calls
        and runtime.get("max_model_forwards") == expected_forwards
        and runtime.get("checkpoint_steps") == [updates],
        "ranking runtime budget",
    )
    ranking = value["ranking"]
    require(
        ranking == {
            "lambda_value": 1.0,
            "branch_weight": 0.5,
            "denominator": RANKING_DENOMINATOR,
            "reference": REFERENCE_NAME,
            "rejected_geometry": False,
        },
        "ranking objective recipe",
    )
    require(isinstance(value["reference_cache_path"], str) and value["reference_cache_path"], "reference cache path")
    _validate_data(load_data(value))
    return dict(value)


def reference_entries(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return preferred then rejected routes so native request IDs stay unique."""

    result: list[dict[str, Any]] = []
    for role in ("preferred", "rejected"):
        for image_key in sorted(data["pairs"], key=int):
            pair = data["pairs"][image_key]
            route = pair[role]
            result.append(
                {
                    "key": f"{image_key}:{role}",
                    "image_id": int(image_key),
                    "role": role,
                    "route": route,
                    "length": len(route["continuation_token_ids"]),
                }
            )
    require(len(result) == 30 and len({item["route"]["route_id"] for item in result}) == 30, "30 unique reference routes")
    return result


def _route_logp(logits: torch.Tensor, route: Mapping[str, Any]) -> tuple[torch.Tensor, int, float]:
    targets = torch.tensor(route["continuation_token_ids"], dtype=torch.long, device=logits.device)
    require(logits.ndim == 2 and logits.shape[0] == targets.numel(), "aligned route logits")
    weights = route["ce_weights"]
    require(isinstance(weights, list) and weights == [1] * len(weights), "ranking routes require all-token CE")
    logp = aligned_token_logprobs(logits.float(), targets)
    summed = logp.sum()
    require(bool(torch.isfinite(summed)), "nonfinite route log probability")
    return summed, int(targets.numel()), float((-summed).detach())


def route_terms(
    logits: torch.Tensor,
    route: Mapping[str, Any],
    hinge: Mapping[str, Any],
    *,
    include_geometry: bool = True,
) -> dict[str, Any]:
    """Compute one current-model route term while retaining its graph."""

    targets = torch.tensor(route["continuation_token_ids"], dtype=torch.long, device=logits.device)
    ce, metrics = training.masked_ce_loss(logits, targets, route["ce_weights"])
    active = int(metrics["active_tokens"])
    require(active == len(route["continuation_token_ids"]), "ranking route active-token count")
    logp_sum = -ce * active
    if include_geometry:
        geometry = training.raw_axis_validity_hinge(
            logits,
            route["trusted_boxes"],
            coordinate_token_ids=hinge["coordinate_token_ids"],
            coordinate_bin_values=hinge["coordinate_bin_values"],
            margin=hinge["margin"],
        )
    else:
        geometry = logits.new_zeros(())
    require(bool(torch.isfinite(ce)) and bool(torch.isfinite(logp_sum)) and bool(torch.isfinite(geometry)), "nonfinite route term")
    return {
        "ce": ce,
        "geometry": geometry,
        "logp_sum": logp_sum,
        "active_tokens": active,
        "ce_numerator": float(metrics["masked_nll_sum"]),
        "ce_denominator": active,
        "active_token_mean_ce": float(ce.detach()),
        "geometry_included": bool(include_geometry),
        "raw_axis_validity_hinge": float(geometry.detach()),
    }


def pair_terms(
    preferred_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    *,
    reference_preferred: float,
    reference_rejected: float,
    denominator: int,
    lambda_value: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the reference-relative pair ranking loss and its accounting."""

    require(preferred_logp.ndim == rejected_logp.ndim == 0, "pair log probabilities must be scalars")
    require(type(denominator) is int and denominator > 0, "pair length denominator")
    require(math.isfinite(float(reference_preferred)) and math.isfinite(float(reference_rejected)), "finite reference log probabilities")
    require(type(lambda_value) in (int, float) and math.isfinite(float(lambda_value)) and lambda_value == 1.0, "ranking lambda")
    reference_delta = float(reference_preferred) - float(reference_rejected)
    delta = (preferred_logp - rejected_logp) - preferred_logp.new_tensor(reference_delta)
    loss = float(lambda_value) * F.softplus(-delta / float(denominator))
    require(bool(torch.isfinite(loss)), "nonfinite pair ranking loss")
    return loss, {
        "reference_preferred_logp_sum": float(reference_preferred),
        "reference_rejected_logp_sum": float(reference_rejected),
        "reference_delta": reference_delta,
        "current_preferred_logp_sum": float(preferred_logp.detach()),
        "current_rejected_logp_sum": float(rejected_logp.detach()),
        "current_delta": float((preferred_logp - rejected_logp).detach()),
        "delta_from_reference": float(delta.detach()),
        "recorded_length_denominator": denominator,
        "lambda_value": float(lambda_value),
        "softplus_rank": float(loss.detach()),
    }


def objective_from_terms(
    canonical_ce: Sequence[torch.Tensor],
    canonical_geometry: Sequence[torch.Tensor],
    preferred_ce: Sequence[torch.Tensor],
    preferred_geometry: Sequence[torch.Tensor],
    ranking: Sequence[torch.Tensor] = (),
    *,
    branch_weight: float = 0.5,
    geometry_weight: float = GEOMETRY_WEIGHT,
    ranking_enabled: bool = False,
    branch_denominator: int = GLOBAL_BRANCH_IMAGES,
) -> torch.Tensor:
    """Compose the globally normalized local contribution.

    Each rank supplies eight terms per branch, while every denominator remains
    the global 32.  Gradients are subsequently SUM-reduced by the existing
    distributed helper.  Ranking is ``+.5 * mean(rank)`` at the global level;
    it is not given another hidden pair-branch factor.
    """

    require(len(canonical_ce) == len(canonical_geometry), "canonical term cardinality")
    require(len(preferred_ce) == len(preferred_geometry), "preferred term cardinality")
    require(type(branch_denominator) is int and branch_denominator > 0, "objective denominator")
    require(branch_weight == 0.5 and geometry_weight == GEOMETRY_WEIGHT, "frozen objective weights")
    tensors = [*canonical_ce, *canonical_geometry, *preferred_ce, *preferred_geometry, *ranking]
    require(tensors, "empty objective terms")
    zero = tensors[0].new_zeros(())
    canonical_value = (
        sum(canonical_ce, zero) / branch_denominator
        + float(geometry_weight) * sum(canonical_geometry, zero) / branch_denominator
        if canonical_ce
        else zero
    )
    preferred_value = (
        sum(preferred_ce, zero) / branch_denominator
        + float(geometry_weight) * sum(preferred_geometry, zero) / branch_denominator
        if preferred_ce
        else zero
    )
    result = float(branch_weight) * (canonical_value + preferred_value)
    if ranking_enabled:
        require(ranking, "R objective requires ranking terms")
        result = result + float(branch_weight) * sum(ranking, zero) / branch_denominator
    else:
        require(not ranking, "P objective cannot carry ranking terms")
    require(result.ndim == 0 and bool(torch.isfinite(result)), "nonfinite composed objective")
    return result


def _reference_cache_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_adapter": manifest["source_adapter"],
        "model_config": _model_identity(manifest["model_config"]),
        "data": manifest["data"],
        "reference": REFERENCE_NAME,
    }


def validate_reference_cache(manifest: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, float]:
    path = Path(manifest["reference_cache_path"]).resolve(strict=True)
    cache = read(path)
    require(cache.get("schema") == REFERENCE_SCHEMA and cache.get("status") == "completed", "reference cache schema/status")
    require(cache.get("identity") == _reference_cache_identity(manifest), "reference cache anchor identity")
    manifest_binding = _binding(cache.get("manifest"), name="reference cache manifest")
    require(training.binding(manifest_binding["path"]) == manifest_binding, "reference cache manifest bytes changed")
    require(cache.get("data") == manifest["data"], "reference cache data binding")
    expected = reference_entries(data)
    observed = cache.get("routes")
    require(isinstance(observed, list) and len(observed) == len(expected), "reference cache route count")
    result: dict[str, float] = {}
    for wanted, got in zip(expected, observed, strict=True):
        require(
            got.get("key") == wanted["key"]
            and got.get("route_id") == wanted["route"]["route_id"]
            and got.get("image_id") == wanted["image_id"]
            and got.get("role") == wanted["role"]
            and got.get("length") == wanted["length"]
            and got.get("token_ids_sha256") == training.digest(wanted["route"]["continuation_token_ids"]),
            "reference cache route identity",
        )
        score = got.get("logp_sum")
        require(type(score) in (int, float) and math.isfinite(float(score)), "reference cache score")
        result[wanted["key"]] = float(score)
    require(cache.get("counts", {}).get("routes") == 30, "reference cache route count receipt")
    return result


def _rank_context() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world = int(os.environ.get("WORLD_SIZE", "-1"))
    require(world == REQUIRED_WORLD_SIZE and 0 <= rank < world and 0 <= local_rank < world, "launch with four torchrun ranks")
    require(torch.cuda.is_available(), "ranking runtime requires CUDA")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world, torch.device("cuda", local_rank)


def _init_dist() -> tuple[int, int, int, torch.device]:
    rank, local_rank, world, device = _rank_context()
    dist.init_process_group("nccl", timeout=timedelta(seconds=distributed.COLLECTIVE_TIMEOUT_SECONDS))
    return rank, local_rank, world, device


def _load_model(manifest: Mapping[str, Any], *, device: torch.device, train: bool) -> tuple[Any, Mapping[str, Any], torch.nn.Module, tuple[tuple[str, torch.nn.Parameter], ...], tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any] | None]:
    from probes.dora_owner_learning.runtime import bind_source256_language_dora, load_policy
    from src.config.inference import InferConfig
    from src.qwen.checkpointing import install_language_decoder_checkpointing

    random.seed(19)
    torch.manual_seed(19)
    torch.cuda.manual_seed(19)
    config = InferConfig.model_validate(manifest["model_config"])
    qwen, loaded = load_policy(config, device=device)
    model = qwen.model
    model.eval()
    if not train:
        empty: tuple[tuple[str, torch.nn.Parameter], ...] = ()
        return qwen, loaded, model, empty, empty, None
    named, frozen = bind_source256_language_dora(
        model,
        expected_tensor_count=manifest["source_adapter"]["semantic_identity"]["tensor_key_count"],
        expected_scalar_count=source256.SOURCE_ADAPTER_SCALAR_COUNT,
    )
    checkpointing = install_language_decoder_checkpointing(model, expected_layer_count=28)
    checkpointing.update(enabled=True, phase="train")
    return qwen, loaded, model, named, frozen, checkpointing


def _forward_routes(
    qwen: Any,
    model: torch.nn.Module,
    manifest: Mapping[str, Any],
    routes: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    pad_token_id: int,
    no_grad: bool,
) -> tuple[list[torch.Tensor], dict[str, int], dict[str, Any]]:
    require(len(routes) > 0 and len(routes) <= MICROBATCH_SIZE, "native microbatch route count")
    context = torch.no_grad() if no_grad else torch.enable_grad()
    with context:
        groups, preparation = replay.prepare_microbatches(
            qwen,
            manifest,
            routes,
            device=device,
            microbatch_size=MICROBATCH_SIZE,
        )
        require(len(groups) == 1 and len(groups[0]["routes"]) == len(routes), "one direct ranking microbatch")
        logits, padding = replay.batched_aligned_logits(
            model,
            groups[0]["inputs"],
            groups[0]["routes"],
            pad_token_id=pad_token_id,
        )
    return logits, padding, preparation


def _reference_records(data: Mapping[str, Any], values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected = {item["key"]: item for item in reference_entries(data)}
    result = []
    for value in values:
        key = value["key"]
        route = expected[key]["route"]
        result.append(
            {
                "key": key,
                "image_id": expected[key]["image_id"],
                "role": expected[key]["role"],
                "route_id": route["route_id"],
                "length": len(route["continuation_token_ids"]),
                "token_ids_sha256": training.digest(route["continuation_token_ids"]),
                "logp_sum": float(value["logp_sum"]),
            }
        )
    return sorted(result, key=lambda item: [0 if item["role"] == "preferred" else 1, item["image_id"]])


def build_reference_cache(manifest_path: Path, *, output: Path | None = None) -> dict[str, Any] | None:
    """Score the 30 observed routes once on the frozen starting adapter."""

    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = validate_manifest(read(manifest_path))
    data = _validate_data(load_data(manifest))
    cache_path = Path(output or manifest["reference_cache_path"]).resolve()
    require(cache_path == Path(manifest["reference_cache_path"]).resolve(), "reference output differs from manifest")
    rank, local_rank, world, device = _init_dist()
    started = time.monotonic()
    try:
        distributed.coordinated_call(
            lambda: (require(not cache_path.exists(), "reference cache already exists"), cache_path.parent.mkdir(parents=True, exist_ok=True)) if rank == 0 else None,
            phase="reference_output_setup",
        )
        dist.barrier()
        qwen, loaded, model, _, _, _ = distributed.coordinated_call(
            lambda: _load_model(manifest, device=device, train=False), phase="reference_model_setup"
        )
        pad_token_id = qwen.tokenizer.pad_token_id
        require(type(pad_token_id) is int and pad_token_id >= 0, "reference pad token")
        entries = reference_entries(data)
        local = [entries[index] for index in range(rank, len(entries), world)]
        local_values: list[dict[str, Any]] = []
        local_calls = 0
        local_forwards = 0
        local_padding = 0
        with torch.no_grad():
            for start in range(0, len(local), MICROBATCH_SIZE):
                selected = local[start : start + MICROBATCH_SIZE]
                logits, padding, preparation = _forward_routes(
                    qwen,
                    model,
                    manifest,
                    [item["route"] for item in selected],
                    device=device,
                    pad_token_id=pad_token_id,
                    no_grad=True,
                )
                for item, row in zip(selected, logits, strict=True):
                    score, length, _ = _route_logp(row, item["route"])
                    local_values.append({"key": item["key"], "logp_sum": float(score.detach()), "length": length})
                local_calls += 1
                local_forwards += len(selected)
                local_padding += int(padding["history_padding_tokens"]) + int(preparation["prompt_padding_tokens"])
        gathered = distributed.gather_objects(local_values)
        receipts = distributed.gather_objects({"rank": rank, "local_routes": len(local), "model_calls": local_calls, "logical_model_forwards": local_forwards, "padding_tokens": local_padding})
        if rank == 0:
            values = [row for rows in gathered for row in rows]
            require(len(values) == 30 and len({row["key"] for row in values}) == 30, "reference cache gathered routes")
            payload = {
                "schema": REFERENCE_SCHEMA,
                "status": "completed",
                "manifest": training.binding(manifest_path),
                "identity": _reference_cache_identity(manifest),
                "data": manifest["data"],
                "routes": _reference_records(data, values),
                "counts": {"routes": 30, "logical_model_forwards": 30, "model_calls": sum(item["model_calls"] for item in receipts)},
                "rank_receipts": receipts,
                "loaded_model": loaded,
                "elapsed_seconds": time.monotonic() - started,
            }
            require(payload["counts"]["model_calls"] == 16, "reference cache call count")
        else:
            payload = None
        distributed.coordinated_call(
            lambda: training.publish(cache_path, payload) if rank == 0 else None,
            phase="reference_publish",
        )
        dist.barrier()
        return payload if rank == 0 else None
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _partition_update(update: Mapping[str, Any], *, rank: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    common = update["common_image_ids"]
    pairs = update["pair_image_ids"]
    start = rank * 8
    return (
        [(start + index, int(image_id)) for index, image_id in enumerate(common[start : start + 8])],
        [(GLOBAL_BRANCH_IMAGES + start + index, int(image_id)) for index, image_id in enumerate(pairs[start : start + 8])],
    )


def _dependency_bindings(manifest: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    return {
        "ranking_training": training.binding(Path(__file__)),
        "ranking_data": manifest["sources"]["data_producer"],
        "source256_training": training.binding(Path(source256.__file__)),
        "training_helpers": training.binding(Path(training.__file__)),
        "artifact_primitives": training.binding(root / "probes/training_set_completion/artifacts.py"),
        "native_replay_helpers": training.binding(root / "src/qwen/native.py"),
        "batched_replay_helpers": training.binding(Path(replay.__file__)),
        "distributed_helpers": training.binding(Path(distributed.__file__)),
        "shared_geometry": training.binding(root / "src/losses/raw_axis_validity_hinge.py"),
    }


def _route_card(route: Mapping[str, Any], terms: Mapping[str, Any], *, branch: str, presentation_index: int) -> dict[str, Any]:
    return {
        "presentation_index": presentation_index,
        "branch": branch,
        "image_id": route["image_id"],
        "route_id": route["route_id"],
        "active_tokens": terms["active_tokens"],
        "ce_numerator": terms["ce_numerator"],
        "ce_denominator_active_tokens": terms["ce_denominator"],
        "active_token_mean_ce": terms["active_token_mean_ce"],
        "geometry_included": terms["geometry_included"],
        "geometry": terms["raw_axis_validity_hinge"],
        "eos_present": route["continuation_token_ids"][-1] == EOS,
    }


def _checkpoint_save(
    *,
    output: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    named: Sequence[tuple[str, torch.nn.Parameter]],
    step: int,
) -> dict[str, Any]:
    receipt = training._checkpoint(
        output,
        manifest_path=manifest_path,
        manifest=manifest,
        model=model,
        optimizer=optimizer,
        named=named,
        step=step,
    )
    scheduler_path = output / "checkpoints" / f"step-{step:05d}" / "scheduler.pt"
    torch.save(scheduler.state_dict(), scheduler_path)
    receipt["scheduler"] = training.binding(scheduler_path)
    return receipt


def run(manifest_path: Path, *, output: Path) -> dict[str, Any] | None:
    """Run one fresh P or R arm; the caller owns durable torchrun execution."""

    from src.qwen.checkpointing import language_decoder_checkpointing_receipt

    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = validate_manifest(read(manifest_path))
    data = _validate_data(load_data(manifest))
    reference_scores = validate_reference_cache(manifest, data)
    output = Path(output).resolve()
    rank, local_rank, world, device = _init_dist()
    started = time.monotonic()
    local_forwards = 0
    local_model_calls = 0
    checkpoints: list[dict[str, Any]] = []
    checkpoint_consensus: list[dict[str, Any]] = []
    old_alarm = signal.getsignal(signal.SIGALRM)

    def expired(*_: Any) -> None:
        raise TimeoutError("ranking training wall budget")

    try:
        distributed.coordinated_call(
            lambda: (require(not output.exists(), "ranking output already exists"), output.mkdir(parents=True)) if rank == 0 else None,
            phase="training_output_setup",
        )
        dist.barrier()
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(math.ceil(manifest["runtime"]["wall_seconds"]))
        qwen, loaded, model, named, frozen, checkpointing = distributed.coordinated_call(
            lambda: _load_model(manifest, device=device, train=True), phase="training_model_setup"
        )
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in named],
            **{**manifest["optimizer"], "betas": tuple(manifest["optimizer"]["betas"])},
        )
        require(not optimizer.state, "ranking training requires fresh AdamW")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=manifest["scheduler"]["total_updates"],
            eta_min=manifest["optimizer"]["lr"] * manifest["scheduler"]["min_lr_ratio"],
        )
        distributed.require_consensus(distributed.state_fingerprint(named, optimizer), label="ranking initial state")
        pad_token_id = qwen.tokenizer.pad_token_id
        require(type(pad_token_id) is int and pad_token_id >= 0, "training pad token")
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)
        for step in range(1, manifest["runtime"]["updates"] + 1):
            update_started = time.monotonic()
            update = data["schedule"][step - 1]
            common_rows, pair_rows = _partition_update(update, rank=rank)
            optimizer.zero_grad(set_to_none=True)
            local_calls_before = local_model_calls
            local_forwards_before = local_forwards
            local_metric = {
                "canonical_ce_mean_sum": 0.0,
                "canonical_ce_numerator_sum": 0.0,
                "canonical_active_tokens": 0,
                "canonical_geometry_sum": 0.0,
                "preferred_ce_mean_sum": 0.0,
                "preferred_ce_numerator_sum": 0.0,
                "preferred_active_tokens": 0,
                "preferred_geometry_sum": 0.0,
                "rejected_logp_sum": 0.0,
                "rejected_active_tokens": 0,
                "ranking_sum": 0.0,
            }
            local_cards: list[dict[str, Any]] = []
            history_padding = 0
            prompt_padding = 0
            phase = f"training_update_{step}_forward_backward"

            def add_route_metrics(prefix: str, terms: Mapping[str, Any]) -> None:
                local_metric[f"{prefix}_ce_mean_sum"] += float(terms["ce"].detach())
                local_metric[f"{prefix}_ce_numerator_sum"] += float(terms["ce_numerator"])
                local_metric[f"{prefix}_active_tokens"] += int(terms["active_tokens"])
                local_metric[f"{prefix}_geometry_sum"] += float(terms["geometry"].detach())

            def forward_backward() -> None:
                nonlocal local_forwards, local_model_calls, history_padding, prompt_padding
                for start in range(0, len(common_rows), MICROBATCH_SIZE):
                    selected = common_rows[start : start + MICROBATCH_SIZE]
                    routes = [data["canonical_routes"][str(image_id)] for _, image_id in selected]
                    logits_rows, padding, preparation = _forward_routes(
                        qwen, model, manifest, routes, device=device, pad_token_id=pad_token_id, no_grad=False
                    )
                    ce_terms: list[torch.Tensor] = []
                    geometry_terms: list[torch.Tensor] = []
                    for (presentation_index, _), route, logits in zip(selected, routes, logits_rows, strict=True):
                        terms = route_terms(logits, route, manifest["validity_hinge"], include_geometry=True)
                        ce_terms.append(terms["ce"])
                        geometry_terms.append(terms["geometry"])
                        add_route_metrics("canonical", terms)
                        local_cards.append(_route_card(route, terms, branch="canonical", presentation_index=presentation_index))
                    objective_from_terms(ce_terms, geometry_terms, (), (), (), branch_denominator=GLOBAL_BRANCH_IMAGES).backward()
                    local_forwards += len(routes)
                    local_model_calls += 1
                    history_padding += int(padding["history_padding_tokens"])
                    prompt_padding += int(preparation["prompt_padding_tokens"])

                for start in range(0, len(pair_rows), MICROBATCH_SIZE):
                    selected = pair_rows[start : start + MICROBATCH_SIZE]
                    pair_values = [data["pairs"][str(image_id)] for _, image_id in selected]
                    preferred_routes = [pair["preferred"] for pair in pair_values]
                    rejected_routes = [pair["rejected"] for pair in pair_values]
                    preferred_logits, preferred_padding, preferred_preparation = _forward_routes(
                        qwen, model, manifest, preferred_routes, device=device, pad_token_id=pad_token_id, no_grad=False
                    )
                    if manifest["arm"] == "R":
                        rejected_logits, rejected_padding, rejected_preparation = _forward_routes(
                            qwen, model, manifest, rejected_routes, device=device, pad_token_id=pad_token_id, no_grad=False
                        )
                    else:
                        rejected_logits = []
                        rejected_padding = {"history_padding_tokens": 0}
                        rejected_preparation = {"prompt_padding_tokens": 0}
                    rejected_rows: Sequence[torch.Tensor | None] = (
                        rejected_logits
                        if manifest["arm"] == "R"
                        else [None] * len(selected)
                    )
                    preferred_ce: list[torch.Tensor] = []
                    preferred_geometry: list[torch.Tensor] = []
                    ranking_losses: list[torch.Tensor] = []
                    for (presentation_index, image_id), pair, preferred_route, rejected_route, preferred_logits_row, rejected_logits_row in zip(
                        selected, pair_values, preferred_routes, rejected_routes, preferred_logits, rejected_rows, strict=True
                    ):
                        preferred = route_terms(preferred_logits_row, preferred_route, manifest["validity_hinge"], include_geometry=True)
                        # The rejected path deliberately computes only current
                        # log probability; it never calls the geometry helper.
                        preferred_ce.append(preferred["ce"])
                        preferred_geometry.append(preferred["geometry"])
                        ranking_loss = None
                        ranking_card = None
                        rejected_logp = None
                        rejected_length = 0
                        rejected_nll = 0.0
                        if manifest["arm"] == "R":
                            require(rejected_logits_row is not None, "rejected logits missing for R")
                            rejected_logp, rejected_length, rejected_nll = _route_logp(rejected_logits_row, rejected_route)
                            ranking_loss, ranking_card = pair_terms(
                                preferred["logp_sum"],
                                rejected_logp,
                                reference_preferred=reference_scores[f"{image_id}:preferred"],
                                reference_rejected=reference_scores[f"{image_id}:rejected"],
                                denominator=pair["denominator"],
                                lambda_value=manifest["ranking"]["lambda_value"],
                            )
                            ranking_losses.append(ranking_loss)
                        add_route_metrics("preferred", preferred)
                        if manifest["arm"] == "R":
                            assert rejected_logp is not None and ranking_loss is not None and ranking_card is not None
                            local_metric["rejected_logp_sum"] += float(rejected_logp.detach())
                            local_metric["rejected_active_tokens"] += rejected_length
                            local_metric["ranking_sum"] += float(ranking_loss.detach())
                        pair_card = {
                            "route_id": rejected_route["route_id"],
                            "active_tokens": rejected_length,
                            "logp_sum": float(rejected_logp.detach()) if rejected_logp is not None else None,
                            "ce_numerator": rejected_nll if manifest["arm"] == "R" else None,
                            "geometry_included": False,
                            "scored": manifest["arm"] == "R",
                            "eos_present": rejected_route["continuation_token_ids"][-1] == EOS,
                        }
                        local_cards.append(
                            {
                                "presentation_index": presentation_index,
                                "branch": "pair",
                                "image_id": image_id,
                                "preferred": _route_card(preferred_route, preferred, branch="preferred", presentation_index=presentation_index),
                                "rejected": pair_card,
                                "ranking": ranking_card,
                            }
                        )
                    objective_from_terms(
                        (),
                        (),
                        preferred_ce,
                        preferred_geometry,
                        ranking_losses,
                        ranking_enabled=manifest["arm"] == "R",
                        branch_denominator=GLOBAL_BRANCH_IMAGES,
                    ).backward()
                    local_forwards += len(preferred_routes) + (len(rejected_routes) if manifest["arm"] == "R" else 0)
                    local_model_calls += 1 + int(manifest["arm"] == "R")
                    history_padding += int(preferred_padding["history_padding_tokens"] + rejected_padding["history_padding_tokens"])
                    prompt_padding += int(preferred_preparation["prompt_padding_tokens"] + rejected_preparation["prompt_padding_tokens"])

                expected_local_calls = 8 if manifest["arm"] == "P" else 12
                expected_local_forwards = 16 if manifest["arm"] == "P" else 24
                require(local_model_calls - local_calls_before == expected_local_calls, "local ranking model-call count")
                require(local_forwards - local_forwards_before == expected_local_forwards, "local ranking forward count")
                require(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for _, parameter in named), "missing/nonfinite DoRA gradient")
                require(all(parameter.grad is None for _, parameter in frozen), "frozen parameter received ranking gradient")

            distributed.coordinated_call(forward_backward, phase=phase)
            distributed.sum_gradients_(named)
            raw_norm = float(torch.nn.utils.clip_grad_norm_([parameter for _, parameter in named], manifest["runtime"]["gradient_clip_norm"], error_if_nonfinite=True, foreach=False))
            applied_lr = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            metric_keys = tuple(local_metric)
            metric = torch.tensor([local_metric[key] for key in metric_keys], dtype=torch.float64, device=device)
            dist.all_reduce(metric, op=dist.ReduceOp.SUM)
            global_metric = {key: float(metric[index].item()) for index, key in enumerate(metric_keys)}
            gathered_cards = distributed.gather_objects(local_cards)
            cards = [card for rows in gathered_cards for card in rows]
            cards.sort(key=lambda card: card["presentation_index"])
            require(len(cards) == GLOBAL_PRESENTATIONS and [card["presentation_index"] for card in cards] == list(range(GLOBAL_PRESENTATIONS)), "ranking presentation cards")
            canonical_ce = global_metric["canonical_ce_mean_sum"] / GLOBAL_BRANCH_IMAGES
            preferred_ce = global_metric["preferred_ce_mean_sum"] / GLOBAL_BRANCH_IMAGES
            canonical_geometry = global_metric["canonical_geometry_sum"] / GLOBAL_BRANCH_IMAGES
            preferred_geometry = global_metric["preferred_geometry_sum"] / GLOBAL_BRANCH_IMAGES
            ranking_mean = global_metric["ranking_sum"] / GLOBAL_BRANCH_IMAGES
            p_total = 0.5 * (canonical_ce + GEOMETRY_WEIGHT * canonical_geometry + preferred_ce + GEOMETRY_WEIGHT * preferred_geometry)
            objective_total = p_total + (0.5 * ranking_mean if manifest["arm"] == "R" else 0.0)
            rank_timings = distributed.gather_objects({"rank": rank, "local_presentations": 16, "local_model_calls": local_model_calls - local_calls_before, "local_logical_model_forwards": local_forwards - local_forwards_before, "history_padding_tokens": history_padding, "prompt_padding_tokens": prompt_padding, "elapsed_seconds": time.monotonic() - update_started, "resources": distributed.resource_receipt(device)})
            update_receipt = {
                "schema": f"{SCHEMA}.update.v1",
                "status": "completed",
                "step": step,
                "arm": manifest["arm"],
                "objective_total": objective_total,
                "P_total": p_total,
                "ranking_mean": ranking_mean,
                "branch_metrics": {
                    "canonical": {"ce_mean": canonical_ce, "ce_numerator_sum": global_metric["canonical_ce_numerator_sum"], "active_token_denominator_sum": int(global_metric["canonical_active_tokens"]), "geometry_mean": canonical_geometry},
                    "preferred": {"ce_mean": preferred_ce, "ce_numerator_sum": global_metric["preferred_ce_numerator_sum"], "active_token_denominator_sum": int(global_metric["preferred_active_tokens"]), "geometry_mean": preferred_geometry},
                    "rejected": {"logp_sum": global_metric["rejected_logp_sum"], "recorded_token_count_sum": int(global_metric["rejected_active_tokens"]), "geometry_included": False},
                },
                "gradient_norm_before_clip": raw_norm,
                "applied_lr": applied_lr,
                "next_lr": float(optimizer.param_groups[0]["lr"]),
                "presentations": cards,
                "logical_model_forwards": step * (64 if manifest["arm"] == "P" else 96),
                "model_calls": step * (32 if manifest["arm"] == "P" else 48),
                "distributed": {
                    "world_size": world,
                    "rank_presentations": [16, 16, 16, 16],
                    "rank_timings": rank_timings,
                    "normalization": {
                        "canonical": "0.5 * SUM(per-example active-token mean CE + 0.01 geometry) / 32",
                        "preferred": "0.5 * SUM(per-example active-token mean CE + 0.01 geometry) / 32",
                        "ranking": "+0.5 * SUM(reference-relative softplus) / 32 for R",
                        "gradient_collective": "SUM",
                        "post_collective_divisor": 1,
                    },
                },
            }
            distributed.coordinated_call(
                lambda: training.publish(output / "updates" / f"step-{step:05d}.json", update_receipt) if rank == 0 else None,
                phase=f"training_update_{step}_receipt",
            )
            if step in manifest["runtime"]["checkpoint_steps"]:
                state = distributed.state_fingerprint(named, optimizer)
                states = distributed.require_consensus(state, label=f"ranking checkpoint {step}")
                require(state["optimizer_steps"] == [step], "ranking optimizer step counter")
                holder: dict[str, Any] = {}

                def save() -> None:
                    if rank == 0:
                        holder["receipt"] = _checkpoint_save(output=output, manifest_path=manifest_path, manifest=manifest, model=model, optimizer=optimizer, scheduler=scheduler, named=named, step=step)

                distributed.coordinated_call(save, phase=f"ranking_checkpoint_{step}_save")
                dist.barrier()
                consensus = {"step": step, "state": state, "rank_count": len(states)}
                distributed.coordinated_call(
                    lambda: training.publish(output / "checkpoints" / f"step-{step:05d}" / "consensus.json", consensus) if rank == 0 else None,
                    phase=f"ranking_checkpoint_{step}_consensus",
                )
                checkpoint_consensus.append(consensus)
                if rank == 0:
                    checkpoints.append(holder["receipt"])

        rank_receipt = {
            "schema": f"{SCHEMA}.rank.v1",
            "status": "completed",
            "rank": rank,
            "local_rank": local_rank,
            "host": socket.gethostname(),
            "device": str(device),
            "local_model_calls": local_model_calls,
            "local_logical_model_forwards": local_forwards,
            "updates": manifest["runtime"]["updates"],
            "resources": distributed.resource_receipt(device),
        }
        rank_receipts = distributed.gather_objects(rank_receipt)
        distributed.coordinated_call(lambda: training.publish(output / "ranks" / f"rank-{rank:03d}.json", rank_receipt), phase="ranking_rank_receipt")
        terminal = {
            "schema": f"{SCHEMA}.terminal.v1",
            "status": "completed",
            "arm": manifest["arm"],
            "mode": manifest["mode"],
            "manifest": training.binding(manifest_path),
            "data": manifest["data"],
            "reference_cache": training.binding(manifest["reference_cache_path"]),
            "loaded_model": loaded,
            "optimizer_mode": "fresh",
            "trainable_surface": training._layout(named),
            "updates": manifest["runtime"]["updates"],
            "logical_model_forwards": manifest["runtime"]["max_model_forwards"],
            "model_calls": manifest["runtime"]["max_model_calls"],
            "checkpoints": checkpoints,
            "activation_checkpointing": language_decoder_checkpointing_receipt(model, checkpointing),
            "elapsed_seconds": time.monotonic() - started,
            "distributed": {
                "backend": dist.get_backend(),
                "world_size": world,
                "rank_receipts": rank_receipts,
                "checkpoint_consensus": checkpoint_consensus,
                "normalization": {
                    "branch_weight": 0.5,
                    "branch_denominator": 32,
                    "ranking_weight": 0.5 if manifest["arm"] == "R" else 0.0,
                    "geometry_weight": GEOMETRY_WEIGHT,
                    "gradient_collective": "SUM",
                    "post_collective_divisor": 1,
                },
                "source_bindings": _dependency_bindings(manifest),
            },
        }
        distributed.coordinated_call(lambda: training.publish(output / "terminal.json", terminal) if rank == 0 else None, phase="ranking_terminal")
        dist.barrier()
        return terminal if rank == 0 else None
    except Exception as exc:
        if rank == 0 and output.exists() and not (output / "terminal.json").exists():
            try:
                training.publish(output / "terminal.json", {"schema": f"{SCHEMA}.terminal.v1", "status": "failed", "arm": manifest["arm"], "mode": manifest["mode"], "manifest": training.binding(manifest_path), "phase": "training", "error": f"{type(exc).__name__}: {exc}", "local_model_calls": local_model_calls, "local_logical_model_forwards": local_forwards})
            except Exception:
                pass
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        if dist.is_initialized():
            dist.destroy_process_group()


def _checkpoint_binding(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    root = checkpoint.parent if checkpoint.name == "adapter" else checkpoint
    adapter = root / "adapter" if (root / "adapter").is_dir() else checkpoint
    files = []
    for path in sorted(adapter.iterdir()):
        if path.is_file():
            files.append(training.binding(path))
    result = {"root": str(root), "adapter": {"path": str(adapter), "files": files}}
    state = root / "state.pt"
    if state.is_file():
        result["state"] = training.binding(state)
    return result


def likelihood(manifest_path: Path, *, checkpoint: Path, output: Path) -> dict[str, Any] | None:
    """Score the exact 30 prepared observed routes at a saved adapter."""

    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from src.config.inference import InferConfig

    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = validate_manifest(read(manifest_path))
    data = _validate_data(load_data(manifest))
    entries = reference_entries(data)
    checkpoint = Path(checkpoint).resolve(strict=True)
    adapter = checkpoint / "adapter" if (checkpoint / "adapter").is_dir() else checkpoint
    require(adapter.is_dir(), "likelihood checkpoint adapter directory")
    output = Path(output).resolve()
    rank, local_rank, world, device = _init_dist()
    started = time.monotonic()
    try:
        distributed.coordinated_call(
            lambda: (require(not output.exists(), "likelihood output already exists"), output.parent.mkdir(parents=True, exist_ok=True)) if rank == 0 else None,
            phase="likelihood_output_setup",
        )
        dist.barrier()
        config = checkpoint_config(InferConfig.model_validate(manifest["model_config"]), str(adapter))
        qwen, loaded = distributed.coordinated_call(lambda: load_policy(config, device=device), phase="likelihood_model_setup")
        model = qwen.model
        model.eval()
        pad_token_id = qwen.tokenizer.pad_token_id
        local = [entries[index] for index in range(rank, len(entries), world)]
        local_values: list[dict[str, Any]] = []
        local_calls = 0
        with torch.no_grad():
            for start in range(0, len(local), MICROBATCH_SIZE):
                selected = local[start : start + MICROBATCH_SIZE]
                logits_rows, _, _ = _forward_routes(qwen, model, manifest, [item["route"] for item in selected], device=device, pad_token_id=pad_token_id, no_grad=True)
                for item, logits in zip(selected, logits_rows, strict=True):
                    score, length, _ = _route_logp(logits, item["route"])
                    local_values.append({"key": item["key"], "logp_sum": float(score), "length": length})
                local_calls += 1
        gathered = distributed.gather_objects(local_values)
        call_receipts = distributed.gather_objects({"rank": rank, "model_calls": local_calls, "logical_model_forwards": len(local)})
        if rank == 0:
            values = [row for rows in gathered for row in rows]
            by_key = {row["key"]: row for row in values}
            require(set(by_key) == {item["key"] for item in entries}, "likelihood route set")
            routes = []
            for item in entries:
                row = by_key[item["key"]]
                routes.append({"key": item["key"], "image_id": item["image_id"], "role": item["role"], "route_id": item["route"]["route_id"], "length": row["length"], "token_ids_sha256": training.digest(item["route"]["continuation_token_ids"]), "logp_sum": row["logp_sum"], "mean_logp": row["logp_sum"] / row["length"]})
            payload = {
                "schema": LIKELIHOOD_SCHEMA,
                "status": "completed",
                "manifest": training.binding(manifest_path),
                "data": manifest["data"],
                "checkpoint": _checkpoint_binding(checkpoint),
                "routes": routes,
                "counts": {"routes": 30, "logical_model_forwards": 30, "model_calls": sum(item["model_calls"] for item in call_receipts)},
                "rank_receipts": call_receipts,
                "loaded_model": loaded,
                "elapsed_seconds": time.monotonic() - started,
            }
            require(payload["counts"]["model_calls"] == 16, "likelihood call count")
        else:
            payload = None
        distributed.coordinated_call(lambda: training.publish(output, payload) if rank == 0 else None, phase="likelihood_publish")
        dist.barrier()
        return payload if rank == 0 else None
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--likelihood", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    require(int(args.cache) + int(args.likelihood) <= 1, "choose one auxiliary mode")
    if args.cache:
        result = build_reference_cache(args.manifest, output=args.output)
    elif args.likelihood:
        require(args.checkpoint is not None, "likelihood requires --checkpoint")
        result = likelihood(args.manifest, checkpoint=args.checkpoint, output=args.output)
    else:
        result = run(args.manifest, output=args.output)
    if result is not None:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
