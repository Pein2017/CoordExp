"""Literal-row training for the two authorized parallel owner-learning lanes.

Scientific records, exposure schedules, coefficients and denominators are caller
owned. This is a small execution adapter, not a replacement for receipt-bound
old producers. Preparation/verification are CPU-only; launches require a grant.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch

from src.losses import aligned_token_logprobs
from src.qwen.native import prepare_replay
from src.runtime.distributed import gather_objects as _dist_values
from src.runtime.model_state import parameter_layout as _parameter_layout, tensor_state_sha256 as _tensor_state_hash
from probes.dora_owner_learning import repeat_recovery_train as old
from probes.dora_owner_learning import margin_preserved_train as margin_engine
from probes.dora_owner_learning.candidate_opportunity import file_hash, require
from probes.dora_owner_learning.route_access import CONFIG, checkpoint_config, publish

SCHEMA = "parallel_owner_training.inputs.v1"
COMPONENTS = ("positive", "conditional_kl", "normal_kl", "margin")
NORMAL_MASK = "bound_original_literal_mask_no_reinference"
MARGIN_POLICY = "current_full_vocabulary_other_max_floor_min_0.1_half_source_margin"
LIMITS = ("max_rank_seconds", "max_cuda_allocated_bytes", "max_cuda_reserved_bytes",
          "max_rss_bytes", "max_model_forwards_per_rank", "max_image_forwards_per_rank")
QUALIFICATION_TOLERANCES = {
    "objective_component_atol": 1e-5, "objective_component_rtol": 1e-4,
    "gradient_norm_atol": 1e-5, "gradient_norm_rtol": 1e-4,
    "update_norm_atol": 1e-7, "update_norm_rtol": 1e-3,
    "adapter_max_abs_difference": 2e-6, "adapter_delta_relative_l2": 5e-3,
    "adapter_delta_cosine_minimum": 0.99999,
    "positive_score_atol": 5e-4, "cold_score_atol": 1e-5,
}


def binding(path: str | Path) -> dict[str, str]:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path)}


def _read_bound(reference: Mapping[str, Any]) -> Any:
    _verify_bound_file(reference)
    return old.load_json(reference["path"])


def _verify_bound_file(reference: Mapping[str, Any]) -> None:
    require(Path(reference["path"]).is_absolute(), "binding path must be absolute")
    require(file_hash(reference["path"]) == reference["sha256"],
            f"bound source changed: {reference['path']}")


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), label)
    return float(value)


def validate_record(record: Mapping[str, Any], *, kind: str,
                    verify_image: bool = True) -> dict[str, Any]:
    require(kind in ("positive", "conditional"), "literal record kind")
    require(isinstance(record.get("record_id"), str) and record["record_id"], "record ID")
    prompt = old._checked_ids(record.get("prompt_token_ids"), field="record.prompt")
    old._checked_ids(record.get("prefix_token_ids"), field="record.prefix", nonempty=False)
    targets = old._checked_ids(record.get("target_token_ids"), field="record.target")
    require(old.digest_ids(prompt) == record.get("prompt_token_ids_sha256"), "record prompt hash")
    image = record.get("image", {})
    require(str(record.get("example_id")) == str(image.get("row_id"))
            and isinstance(image.get("row_index"), int), "record example/image identity")
    for key in ("image_path", "image_sha256", "observed_image_grid_thw", "executed_media_sha256"):
        require(key in image, f"record image lacks {key}")
    if verify_image:
        require(file_hash(image["image_path"]) == image["image_sha256"], "record image changed")
    if kind == "positive":
        require(targets[0] == 151646 and targets[-1] == old.BOX_END
                and targets.count(151646) == targets.count(old.BOX_END) == 1
                and old.EOS not in targets and old.PAD not in targets,
                "positive target must be exactly one complete non-EOS row")
        require("kl_positions" not in record, "positive targets cannot carry a partial loss mask")
    else:
        positions = old._checked_positions(record.get("kl_positions"), length=len(targets),
                                           field="conditional.kl_positions")
        require(positions and record.get("unknown_mask_policy") == "literal_positions_only",
                "conditional mask must be explicit and nonempty")
    return dict(record)


def validate_contract(packet: Mapping[str, Any], *, verify_images: bool = True) -> None:
    """Pure packet checks, including the masks and reductions consumed by training."""
    require(packet.get("schema") == SCHEMA, "training packet schema")
    require(packet.get("normal_mask_policy") == NORMAL_MASK, "normal mask policy")
    require(packet.get("margin_policy") == MARGIN_POLICY, "full-vocabulary margin policy")
    raw_sources = packet.get("materialization_raw_sources", [])
    require(isinstance(raw_sources, list)
            and len({row.get("path") for row in raw_sources}) == len(raw_sources)
            and all(set(row) == {"path", "sha256"} and Path(row["path"]).is_absolute()
                    for row in raw_sources), "materialization raw-source bindings")
    require(packet.get("normal_keys") and len(set(packet["normal_keys"])) == len(packet["normal_keys"]),
            "normal keys empty or duplicated")
    records: dict[str, Mapping[str, Any]] = {}
    for kind, field in (("positive", "positive_records"), ("conditional", "conditional_records")):
        require(isinstance(packet.get(field), list), f"{field} list")
        for record in packet[field]:
            validate_record(record, kind=kind, verify_image=verify_images)
            require(record["record_id"] not in records, "record IDs must be globally unique")
            records[record["record_id"]] = record
    require(not set(records).intersection(packet["normal_keys"]), "normal/literal record IDs collide")
    positives = {row["record_id"] for row in packet["positive_records"]}
    require(positives, "positive records empty")
    for field in ("weights", "denominators"):
        require(set(packet.get(field, {})) == set(COMPONENTS), f"explicit {field} keys")
        for key in COMPONENTS:
            _number(packet[field][key], f"{field}.{key}", positive=field == "denominators")
    require(packet["weights"]["positive"] > 0, "positive coefficient must be nonzero")
    require(bool(packet["conditional_records"]) or packet["weights"]["conditional_kl"] == 0,
            "conditional KL coefficient has no records")
    if packet["conditional_records"]:
        require(packet["denominators"]["conditional_kl"] == len(packet["conditional_records"]),
                "conditional KL must average the fixed record count")
    for key in ("normal_kl", "margin"):
        require(packet["denominators"][key] == len(packet["normal_keys"]),
                f"{key} denominator must be selected normal image count")
    require(isinstance(packet.get("arms"), dict) and packet["arms"], "explicit arms")
    used: set[str] = set()
    for name, arm in packet["arms"].items():
        require(isinstance(name, str) and name and isinstance(arm.get("steps"), list)
                and arm["steps"], "arm explicit finite steps")
        for step in arm["steps"]:
            require(isinstance(step, list) and step, "empty positive exposure step")
            for exposure in step:
                require(set(exposure) == {"record_id", "weight"}
                        and exposure["record_id"] in positives, "unknown exposure record or fields")
                _number(exposure["weight"], "exposure weight", positive=True)
                used.add(exposure["record_id"])
    require(used == positives, "positive record has no exposure in any arm")
    runtime = packet.get("runtime", {})
    require(set(runtime) == {"world_sizes", *LIMITS}, "explicit runtime fields")
    require(isinstance(runtime["world_sizes"], list) and runtime["world_sizes"]
            and set(runtime["world_sizes"]).issubset({2, 8}), "only qualified two/eight rank route")
    for key in LIMITS:
        _number(runtime[key], f"runtime.{key}", positive=True)
    optimizer = packet.get("optimizer", {})
    require(set(optimizer) == {"lr", "betas", "eps", "weight_decay", "foreach"},
            "explicit AdamW settings required")
    for key in ("lr", "eps", "weight_decay"):
        _number(optimizer[key], f"optimizer.{key}", positive=key != "weight_decay")
    require(isinstance(optimizer["betas"], list) and len(optimizer["betas"]) == 2
            and all(type(v) in (int, float) and 0 <= v < 1 for v in optimizer["betas"])
            and optimizer["foreach"] is False, "AdamW beta/foreach contract")
    _number(packet.get("clip_gradient_norm"), "explicit gradient clip", positive=True)


def prepare_packet(output_path: str | Path, *, lane: str,
                   anchor_input_path: str | Path, margin_input_path: str | Path,
                   normal_keys: Sequence[str], positive_records: Sequence[Mapping[str, Any]],
                   conditional_records: Sequence[Mapping[str, Any]], arms: Mapping[str, Any],
                   weights: Mapping[str, float], denominators: Mapping[str, float],
                   optimizer: Mapping[str, Any], clip_gradient_norm: float,
                   runtime: Mapping[str, Any],
                   materialization_raw_source_paths: Sequence[str | Path] = ()) -> dict[str, Any]:
    packet = {
        "schema": SCHEMA, "status": "prepared_no_model_execution", "lane": lane,
        "anchor_input": binding(anchor_input_path), "margin_input": binding(margin_input_path),
        "normal_keys": list(normal_keys), "positive_records": list(positive_records),
        "conditional_records": list(conditional_records), "arms": dict(arms),
        "weights": dict(weights), "denominators": dict(denominators),
        "optimizer": {**optimizer, "betas": list(optimizer["betas"])},
        "clip_gradient_norm": clip_gradient_norm, "runtime": dict(runtime),
        "materialization_raw_sources": [binding(path) for path in materialization_raw_source_paths],
        "normal_mask_policy": NORMAL_MASK, "margin_policy": MARGIN_POLICY,
        "code_identity": [binding(path) for path in
                          dict.fromkeys([Path(__file__), Path(margin_engine.__file__), *old._code_paths()])],
    }
    validate_contract(packet)
    # Validate external identities BEFORE publishing a packet; no partial accepted artifact.
    _load_dependencies(packet)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    publish(output, packet)
    return packet


def _index_raw_groups(groups: Sequence[Sequence[Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for rows in groups:
        for row in rows:
            require(row.example_id not in result, f"duplicate materialization raw ID: {row.example_id}")
            result[row.example_id] = row
    return result


def _load_materialization_raw(packet: Mapping[str, Any], *, config_input: str | Path) -> dict[str, Any]:
    from src.data import load_raw_examples

    references = packet.get("materialization_raw_sources", [])
    if not references:
        references = [binding(config_input)]
    for reference in references:
        _verify_bound_file(reference)
    paths = [Path(reference["path"]).resolve() for reference in references]
    require(Path(config_input).resolve() in paths, "configured raw source missing from materialization union")
    return _index_raw_groups([load_raw_examples(path) for path in paths])


def _load_dependencies(packet: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    _read_bound(packet["anchor_input"])
    anchor, manifest = old.validate_inputs(
        Path(packet["anchor_input"]["path"]), verify_sources=True, require_current_code=False,
    )
    margins = margin_engine._validated_margin_table(_read_bound(packet["margin_input"]), manifest=manifest)
    normal_by_key = {row["key"]: row for row in manifest["normals"]["cases"]}
    require(set(packet["normal_keys"]).issubset(normal_by_key), "normal key outside bound source bank")
    normals = [normal_by_key[key] for key in packet["normal_keys"]]
    raw = _load_materialization_raw(packet, config_input=anchor["config"]["data"]["input_jsonl"])
    literal_ids = {row["example_id"] for row in
                   [*packet["positive_records"], *packet["conditional_records"]]}
    require(literal_ids.issubset(raw), "literal record missing from bound materialization raw sources")
    active_images = {str(row["image"]["image_id"]) for row in
                     [*packet["positive_records"], *packet["conditional_records"]]}
    require(not active_images.intersection(str(row["image_id"]) for row in normals),
            "active images must be excluded from normal bank")
    return anchor, normals, {key: margins[key] for key in packet["normal_keys"]}


def load_packet(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    packet = old.load_json(path)
    validate_contract(packet)
    for source in packet["code_identity"]:
        require(file_hash(source["path"]) == source["sha256"], f"execution source changed: {source['path']}")
    anchor, normals, margins = _load_dependencies(packet)
    return packet, anchor, normals, margins


def local_scale(packet: Mapping[str, Any], component: str, *, world_size: int,
                exposure_weight: float = 1.0) -> float:
    """DDP averages ranks: only globally sharded terms need the world factor."""
    require(component in COMPONENTS and world_size > 0, "loss component/world")
    replicas = world_size if component in ("normal_kl", "margin") else 1
    return replicas * packet["weights"][component] * exposure_weight / packet["denominators"][component]


def item_loss(logits: torch.Tensor, targets: torch.Tensor, *, kind: str,
              scales: Mapping[str, float], positions: Sequence[int],
              reference_logp: torch.Tensor | None,
              margin: Mapping[str, Any] | None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Consumer-facing loss boundary shared by replay and gradient-parity tests."""
    require(logits.dtype == torch.float32 and logits.ndim == 2
            and targets.ndim == 1 and logits.shape[0] == targets.numel(), "FP32 aligned item")
    if kind == "positive":
        require(reference_logp is None and not positions, "positive full-row target has no KL mask")
        raw = -aligned_token_logprobs(logits, targets).sum()
        loss = raw * scales["positive"]
        stats = {"positive_nll_sum": float(raw.detach()),
                 "positive": float(loss.detach()), "route": old._target_score_stats(logits.detach(), targets)}
    else:
        require(kind in ("conditional", "normal") and reference_logp is not None and positions,
                "KL item reference/mask missing")
        component = "conditional_kl" if kind == "conditional" else "normal_kl"
        kl = old.reference_kl(logits, reference_logp, positions)
        loss = kl * scales[component]
        stats = {"raw_kl": float(kl.detach()), component: float(loss.detach())}
        if kind == "normal":
            require(margin is not None, "normal full-vocabulary margin table missing")
            penalty, detail = margin_engine.worst_margin_penalty(logits, targets, margin)
            margin_loss = penalty * scales["margin"]
            if scales["margin"] != 0:
                loss = loss + margin_loss
            stats.update(margin=float(margin_loss.detach()), raw_margin=float(penalty.detach()),
                         margin_detail=detail)
    require(loss.ndim == 0 and bool(torch.isfinite(loss)), "nonfinite scalar item loss")
    return loss, stats


class LiteralRowScorer(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: Mapping[str, Any], prompt_ids: Sequence[int], target_ids: Sequence[int],
                **loss_kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        replay = prepare_replay(self.model, inputs, prompt_token_ids=prompt_ids,
                                continuation_token_ids=target_ids)
        logits = replay.aligned_logits(self.model(**replay.inputs).logits)
        require(replay.target_ids.tolist() == list(target_ids), "literal replay target identity")
        return item_loss(logits, replay.target_ids, **loss_kwargs)


def _materialize(record: Mapping[str, Any], *, qwen: Any, frontend: Any,
                 config: Any, raw: Mapping[str, Any]) -> dict[str, Any]:
    case = {"image": record["image"], "example_id": record["example_id"],
            "prompt_token_ids": record["prompt_token_ids"],
            "prompt_token_ids_sha256": record["prompt_token_ids_sha256"]}
    entry = old._materialize_case(qwen=qwen, frontend=frontend, config=config,
                                  raw=raw[str(record["example_id"])], case=case, positive=False)
    return {**entry, "record": record, "prompt_ids": [*entry["prompt_ids"], *record["prefix_token_ids"]]}


def _normal_record(case: Mapping[str, Any]) -> dict[str, Any]:
    return {**case, "record_id": case["key"], "prefix_token_ids": [],
            "target_token_ids": case["action_ids"], "kl_positions": case["initial_layout"]["kl_positions"]}


def _load_model(anchor: Mapping[str, Any], *, adapter_path: str, device: torch.device,
                evidence_dir: Path, packet: Mapping[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_research_infer_config
    from src.inference.runtime import assemble_frontend
    from src.qwen.special_token_embeddings import inspect_special_token_embedding_delta_payload
    from probes.dora_owner_learning.runtime import load_policy

    base = load_research_infer_config(CONFIG).config
    require(base.model_dump(mode="json") == anchor["config"], "live base config identity")
    config = checkpoint_config(base, adapter_path)
    require(config.model.dtype == "fp32" and config.backend.hf.attn_implementation == "sdpa"
            and config.backend.hf.patch_embed_linearization == "enabled"
            and config.embedding_delta is not None
            and str(config.embedding_delta.path) == anchor["source_embedding"]["root"],
            "effective fp32/SDPA/source-embedding configuration")
    frontend = assemble_frontend(config, generation_config_fingerprint=sha256_json(
        config.generation.model_dump(mode="json")))
    qwen, identity = load_policy(config, device=device)
    qwen.model.eval()
    inspected = inspect_special_token_embedding_delta_payload(
        identity["model_identity"]["embedding_delta"]["identity"]["delta_path"],
        anchor["model_identity"]["base_model"])
    composition = old.loaded_composition_evidence(loaded_identity=identity,
        expected_base=anchor["model_identity"]["base_model"], expected_adapter=adapter_path,
        expected_embedding=anchor["source_embedding"], inspected_embedding=inspected)
    publish(evidence_dir / "loaded-model.json", identity)
    publish(evidence_dir / "loaded-composition-check.json", composition)
    require(composition["passed"], "loaded composition does not match explicit adapter")
    require(qwen.token_identity.im_end_token_ids == (old.EOS,) and qwen.tokenizer.pad_token_id == old.PAD,
            "native terminal token identity")
    raw = _load_materialization_raw(packet, config_input=config.data.input_jsonl)
    return qwen, frontend, config, raw


def _score_records(model: torch.nn.Module, entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    with torch.no_grad():
        for entry in entries:
            record = entry["record"]
            replay = prepare_replay(model, entry["inputs"], prompt_token_ids=entry["prompt_ids"],
                                    continuation_token_ids=record["target_token_ids"])
            logits = replay.aligned_logits(model(**replay.inputs).logits)
            result[record["record_id"]] = old._target_score_stats(logits, replay.target_ids)
    return result


def work_counts(packet: Mapping[str, Any], arm: str, world_size: int, rank: int) -> dict[str, int]:
    steps = packet["arms"][arm]["steps"]
    normal = len(packet["normal_keys"][rank::world_size])
    conditional = len(packet["conditional_records"])
    positive = len(packet["positive_records"])
    references = conditional + normal
    training = sum(len(step) + conditional + normal for step in steps)
    scores = 2 * positive if rank == 0 else 0
    return {"reference_forwards": references, "training_replays": training,
            "final_reference_forwards": normal, "positive_score_replays": scores,
            "model_forwards": references + training + normal + scores,
            "backwards": training, "synchronized_backwards": len(steps),
            "optimizer_steps": len(steps)}


def execute_rank(*, input_path: Path, arm: str, world_size: int, output_root: Path) -> None:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from src.adapters.dora import select_dora_parameters
    from src.qwen.checkpointing import (
        install_language_decoder_checkpointing,
        language_decoder_checkpointing_receipt as checkpointing_receipt,
    )
    from src.adapters.dora import save_dora_adapter_payload

    packet, anchor, normals, margins = load_packet(input_path)
    rank, local_rank, world = [int(os.environ.get(key, "-1")) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    require(world == world_size and world in packet["runtime"]["world_sizes"]
            and rank == local_rank and 0 <= rank < world and len(visible) == world
            and len(set(visible)) == world and all(v.isdigit() for v in visible),
            "explicit single-node physical GPU topology")
    require(arm in packet["arms"], "unknown arm")
    expected = work_counts(packet, arm, world, rank)
    limits = packet["runtime"]
    require(expected["model_forwards"] <= limits["max_model_forwards_per_rank"], "planned forward bound")
    run = output_root / "ranks" / f"rank{rank}"
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    counters = {key: 0 for key in [*expected, "model_loads", "image_forwards"]}
    resources: list[dict[str, Any]] = []
    status, error, phase = "failed", None, "preload"
    initialized = False
    device = torch.device("cuda", local_rank)

    def expired(*_: Any) -> None:
        raise TimeoutError("rank lifecycle time bound exceeded")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(math.ceil(limits["max_rank_seconds"]))
    with (run / "execution.log").open("x") as log, redirect_stdout(log), redirect_stderr(log):
      try:
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        dist.init_process_group("nccl", timeout=timedelta(seconds=600), device_id=device)
        initialized = True
        require(len(set(_dist_values(file_hash(input_path)))) == 1, "rank packet bytes differ")
        qwen, frontend, config, raw = _load_model(anchor, adapter_path=anchor["stable50_adapter"]["root"],
            device=device, evidence_dir=run, packet=packet)
        counters["model_loads"] = 1
        model = qwen.model

        def count_model(*_: Any) -> None:
            counters["model_forwards"] += 1
            require(counters["model_forwards"] <= limits["max_model_forwards_per_rank"], "model forward bound")

        def count_image(*_: Any) -> None:
            counters["image_forwards"] += 1
            require(counters["image_forwards"] <= limits["max_image_forwards_per_rank"], "image forward bound")

        model.register_forward_pre_hook(count_model)
        visuals = [module for name, module in model.named_modules() if name.endswith("visual")]
        require(len(visuals) == 1, "single visual module")
        visuals[0].register_forward_pre_hook(count_image)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        named = select_dora_parameters(model, towers=("language",), adapter_name="default")
        require(len(named) == old.EXPECTED_TRAINABLE_TENSORS
                and sum(p.numel() for _, p in named) == old.EXPECTED_TRAINABLE_SCALARS
                and all("language_model" in n and not any(x in n for x in
                    ("visual", "vision", "merger", "embed_tokens", "lm_head")) for n, _ in named),
                "exact unmerged language-only DoRA surface")
        for _, parameter in named:
            parameter.requires_grad_(True)
        selected = {id(p) for _, p in named}
        frozen = [(n, p) for n, p in model.named_parameters() if id(p) not in selected]
        versions = [(p, p._version) for _, p in frozen]
        frozen_hash = _tensor_state_hash(frozen)
        source_hash = _tensor_state_hash(named)
        require(len(set(_dist_values((frozen_hash, source_hash)))) == 1, "initial rank parameter identity")
        publish(run / "trainable-layout.json", _parameter_layout(named))
        positives = [_materialize(r, qwen=qwen, frontend=frontend, config=config, raw=raw)
                     for r in packet["positive_records"]]
        conditional = [_materialize(r, qwen=qwen, frontend=frontend, config=config, raw=raw)
                       for r in packet["conditional_records"]]
        normal = [_materialize(_normal_record(r), qwen=qwen, frontend=frontend, config=config, raw=raw)
                  for r in normals[rank::world]]
        positive_by_id = {e["record"]["record_id"]: e for e in positives}
        checkpointing = install_language_decoder_checkpointing(model, expected_layer_count=28)
        checkpointing["enabled"] = False
        phase = "reference"
        refs: dict[str, torch.Tensor] = {}
        cards = []
        for entry in [*conditional, *normal]:
            record = entry["record"]
            ref = old._reference_logp(model, entry["inputs"], entry["prompt_ids"],
                                     record["target_token_ids"], record["kl_positions"])
            refs[record["record_id"]] = ref
            counters["reference_forwards"] += 1
            cards.append({"record_id": record["record_id"], "shape": list(ref.shape),
                          "bytes": ref.numel() * ref.element_size(),
                          "sha256": _tensor_state_hash([("reference", ref)])})
        publish(run / "reference-cache.json", cards)
        resources.append(old._capture_lifecycle_resources(device=device, started=started, phase="post_reference_cache"))
        scorer = LiteralRowScorer(model)
        ddp = DDP(scorer, device_ids=[local_rank], output_device=local_rank,
                  broadcast_buffers=False, init_sync=False)
        checkpointing.update(enabled=True, phase="train")
        optimizer = torch.optim.AdamW([p for _, p in named], **packet["optimizer"])
        require(not optimizer.state, "fresh optimizer required")
        initial_scores = _score_records(model, positives) if rank == 0 else None
        if rank == 0:
            counters["positive_score_replays"] += len(positives)
            publish(output_root / "initial-positive-scores.json", initial_scores)
        scales = {key: local_scale(packet, key, world_size=world) for key in COMPONENTS}
        for step_index, schedule in enumerate(packet["arms"][arm]["steps"], 1):
            phase = f"train_update_{step_index}"
            items = [("positive", positive_by_id[e["record_id"]], e["weight"]) for e in schedule]
            items += [("conditional", e, 1.0) for e in conditional]
            items += [("normal", e, 1.0) for e in normal]
            optimizer.zero_grad(set_to_none=True)
            before = [p.detach().clone() for _, p in named]
            records = []
            for index, (kind, entry, exposure_weight) in enumerate(items):
                record = entry["record"]
                reference = refs[record["record_id"]].to(device) if kind != "positive" else None
                synchronized = index == len(items) - 1
                item_scales = {**scales, "positive": scales["positive"] * exposure_weight}
                with nullcontext() if synchronized else ddp.no_sync():
                    loss, stats = ddp(entry["inputs"], entry["prompt_ids"], record["target_token_ids"],
                        kind=kind, scales=item_scales, positions=record.get("kl_positions", []),
                        reference_logp=reference, margin=margins.get(record["record_id"]))
                    loss.backward()
                records.append({"record_id": record["record_id"], "kind": kind,
                    "exposure_weight": exposure_weight, "scaled_loss": float(loss.detach()), **stats})
                counters["training_replays"] += 1
                counters["backwards"] += 1
                counters["synchronized_backwards"] += int(synchronized)
                del reference, loss
            finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _, p in named)
            require(all(_dist_values(finite)), "missing or nonfinite selected gradients")
            require(all(p.grad is None and not p.requires_grad for _, p in frozen), "frozen surface gradient")
            gradient_hash = _tensor_state_hash([(n, p.grad) for n, p in named])
            require(len(set(_dist_values(gradient_hash))) == 1, "reduced rank gradients differ")
            raw_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in named],
                packet["clip_gradient_norm"], error_if_nonfinite=True, foreach=False))
            clipped_norm = math.sqrt(sum(float(p.grad.detach().double().square().sum()) for _, p in named))
            require(raw_norm > 0 and clipped_norm <= packet["clip_gradient_norm"] + 1e-6, "gradient clip")
            optimizer.step()
            counters["optimizer_steps"] += 1
            movement = math.sqrt(sum(float((p.detach() - b).double().square().sum())
                                     for (_, p), b in zip(named, before, strict=True)))
            del before
            require(movement > 0 and math.isfinite(movement), "no finite parameter movement")
            require(all(p._version == version for p, version in versions), "frozen parameter changed")
            adapter_hash = _tensor_state_hash(named)
            optimizer_hash = old._optimizer_hash(optimizer, named)
            require(len(set(_dist_values((adapter_hash, optimizer_hash)))) == 1, "rank adapter/optimizer mismatch")
            local = {"rank": rank, "step": step_index, "records": records,
                "raw_gradient_norm": raw_norm, "clipped_gradient_norm": clipped_norm,
                "gradient_sha256": gradient_hash, "adapter_sha256": adapter_hash,
                "optimizer_sha256": optimizer_hash, "movement_l2": movement,
                "loss_sum_before_ddp_mean": sum(r["scaled_loss"] for r in records),
                "counters": dict(counters)}
            publish(run / f"update-{step_index:02d}.json", local)
            rows = _dist_values(local)
            if rank == 0:
                publish(output_root / f"update-{step_index:02d}.json", {
                    "step": step_index, "arm": arm, "world_size": world, "ranks": rows,
                    "global_objective_value": sum(r["loss_sum_before_ddp_mean"] for r in rows) / world,
                    "objective_components": {k: sum(e.get(k, 0.0) for r in rows for e in r["records"]) / world
                                             for k in COMPONENTS}})
            dist.barrier()
        phase = "final_readback"
        final_reference = []
        state_before = (_tensor_state_hash(named), old._optimizer_hash(optimizer, named))
        with torch.no_grad():
            for entry in normal:
                record = entry["record"]
                ref = refs[record["record_id"]].to(device)
                loss, stats = scorer(entry["inputs"], entry["prompt_ids"], record["target_token_ids"],
                    kind="normal", scales=scales, positions=record["kl_positions"],
                    reference_logp=ref, margin=margins[record["record_id"]])
                counters["final_reference_forwards"] += 1
                final_reference.append({"record_id": record["record_id"], **stats})
                del loss, ref
        publish(run / "final-reference.json", {"records": final_reference})
        final_scores = _score_records(model, positives) if rank == 0 else None
        if rank == 0:
            counters["positive_score_replays"] += len(positives)
            publish(output_root / "final-live-positive-scores.json", final_scores)
        require(state_before == (_tensor_state_hash(named), old._optimizer_hash(optimizer, named)),
                "no-grad readback changed adapter/optimizer")
        require(all(_dist_values(_tensor_state_hash(frozen) == frozen_hash)), "frozen bytes changed")
        require(all(counters[k] == v for k, v in expected.items()), "measured work differs from explicit schedule")
        state = {"rank": rank, "counters": dict(counters), "adapter_sha256": state_before[0],
                 "optimizer_sha256": state_before[1], "frozen_sha256": frozen_hash,
                 "reference_cache_bytes": sum(c["bytes"] for c in cards),
                 "activation_checkpointing": checkpointing_receipt(model, checkpointing)}
        states = _dist_values(state)
        phase = "adapter_export"
        export = None
        if rank == 0:
            try:
                saved = save_dora_adapter_payload(
                    model,
                    source_root=Path(anchor["stable50_adapter"]["root"]),
                    output=output_root / "adapter",
                    expected_base_model_path=anchor["model_identity"]["base_model"],
                    expected_tensor_count=old.EXPECTED_TRAINABLE_TENSORS,
                )
                publish(output_root / "provisional.json", {"schema": "parallel_owner_training.receipt.v1",
                    "status": "unsealed_candidate", "arm": arm, "world_size": world,
                    "updates": len(packet["arms"][arm]["steps"]), "input": binding(input_path),
                    "saved_adapter": saved, "source_adapter": anchor["stable50_adapter"],
                    "source_embedding": anchor["source_embedding"], "model_identity": anchor["model_identity"],
                    "source_adapter_state_sha256": source_hash, "final_adapter_state_sha256": state_before[0],
                    "rank_states": states, "initial_positive_scores": initial_scores,
                    "final_live_positive_scores": final_scores,
                    "stop_reason": "completed_explicit_frozen_schedule"})
                export = {"ok": True}
            except BaseException as exc:
                export = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        export = _dist_values(export)[0]
        require(export["ok"], f"rank0 export failed: {export.get('error')}")
        dist.barrier()
        resources.append(old._capture_lifecycle_resources(device=device, started=started, phase="post_export"))
        lifecycle = old.combine_lifecycle_resource_observations(resources)
        require(not lifecycle["cuda_measurement_errors"] and all(
            lifecycle[k] <= limits[limit] for k, limit in (
                ("peak_cuda_allocated_bytes", "max_cuda_allocated_bytes"),
                ("peak_cuda_reserved_bytes", "max_cuda_reserved_bytes"),
                ("peak_rss_bytes", "max_rss_bytes"), ("elapsed_seconds", "max_rank_seconds"))),
                "rank lifecycle resource bound")
        status = "completed"
      except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
      finally:
        signal.alarm(0)
        resources.append(old._capture_lifecycle_resources(device=device, started=started, phase="terminal_finally"))
        publish(run / "terminal.json", {"rank": rank, "status": status, "error": error,
            "arm": arm, "world_size": world, "counters": counters, "phase": phase,
            "lifecycle_resources": old.combine_lifecycle_resource_observations(resources)})
        if initialized:
            dist.destroy_process_group()


def verify_receipt(output_root: Path) -> dict[str, Any]:
    receipt = old.load_json(output_root / "receipt.json")
    require(receipt["status"] == "technically_completed_cold_pending", "uncompleted training receipt")
    packet, _, _, _ = load_packet(receipt["input"]["path"])
    require(file_hash(receipt["input"]["path"]) == receipt["input"]["sha256"], "training input changed")
    old._verify_identity(receipt["saved_adapter"], label="saved adapter")
    for rank in range(receipt["world_size"]):
        terminal = _read_bound(receipt["terminals"][rank])
        require(terminal["status"] == "completed" and terminal["rank"] == rank, "failed/misassociated rank")
        expected = work_counts(packet, receipt["arm"], receipt["world_size"], rank)
        require(all(terminal["counters"][k] == v for k, v in expected.items()), "terminal schedule counter mismatch")
    for record in receipt["step_records"]:
        _read_bound(record)
    require(receipt["updates"] == len(packet["arms"][receipt["arm"]]["steps"])
            and len(receipt["step_records"]) == receipt["updates"], "receipt completed schedule")
    for reference in receipt["final_reference_records"]:
        _read_bound(reference)
    _read_bound(receipt["provisional"])
    require(_read_bound(receipt["launcher_exit"])["returncode"] == 0, "launcher exit failure")
    return receipt


def launch(*, input_path: Path, arm: str, world_size: int, output_root: Path) -> dict[str, Any]:
    packet, _, _, _ = load_packet(input_path)
    require(arm in packet["arms"] and world_size in packet["runtime"]["world_sizes"], "launch arm/world")
    output_root.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc-per-node={world_size}", "--module", "probes.parallel_owner_research.training", "rank",
        "--input", str(input_path.resolve()), "--arm", arm, "--world-size", str(world_size),
        "--output-root", str(output_root.resolve())]
    publish(output_root / "launch.json", {"command": command, "input": binding(input_path),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "work_counts": [work_counts(packet, arm, world_size, r) for r in range(world_size)]})
    with (output_root / "launcher.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    publish(output_root / "launcher-exit.json", {"returncode": result.returncode})
    require(result.returncode == 0, f"rank launch failed with exit {result.returncode}; preserve {output_root}")
    receipt = old.load_json(output_root / "provisional.json")
    terminals = [binding(output_root / "ranks" / f"rank{r}" / "terminal.json") for r in range(world_size)]
    require(all(_read_bound(t)["status"] == "completed" for t in terminals), "rank terminal gate")
    receipt.update(status="technically_completed_cold_pending", terminals=terminals,
        provisional=binding(output_root / "provisional.json"),
        launcher_exit=binding(output_root / "launcher-exit.json"),
        final_reference_records=[binding(output_root / "ranks" / f"rank{r}" / "final-reference.json")
                                 for r in range(world_size)],
        step_records=[binding(output_root / f"update-{s:02d}.json")
                      for s in range(1, len(packet["arms"][arm]["steps"]) + 1)])
    publish(output_root / "receipt.json", receipt)
    return verify_receipt(output_root)


def cold_check(*, input_path: Path, output_root: Path) -> dict[str, Any]:
    require(len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) == 1
            and os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit(), "cold check single visible physical GPU")
    receipt = verify_receipt(output_root)
    require(receipt["input"] == binding(input_path), "cold/training packet identity")
    packet, anchor, _, _ = load_packet(input_path)
    run = output_root / "cold"
    run.mkdir(exist_ok=False)
    started = time.monotonic()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    qwen, frontend, config, raw = _load_model(anchor, adapter_path=receipt["saved_adapter"]["root"],
        device=device, evidence_dir=run, packet=packet)
    entries = [_materialize(r, qwen=qwen, frontend=frontend, config=config, raw=raw)
               for r in packet["positive_records"]]
    observed = _score_records(qwen.model, entries)
    expected = receipt["final_live_positive_scores"]
    require(set(observed) == set(expected), "cold positive IDs")
    errors = {}
    for key, scores in observed.items():
        require(all(scores[k] == expected[key][k] for k in ("token_count", "argmax_target_tokens")),
                "cold discrete positive scores")
        errors[key] = {k: scores[k] - expected[key][k]
                       for k in ("sum_logprob", "mean_logprob", "mean_target_margin", "min_target_margin")}
    require(all(abs(v) <= 1e-5 for row in errors.values() for v in row.values()), "cold/live score tolerance")
    result = {"schema": "parallel_owner_training.cold_check.v1", "status": "passed",
              "training_receipt": binding(output_root / "receipt.json"),
              "saved_adapter": receipt["saved_adapter"], "positive_scores": observed,
              "live_score_deltas": errors, "model_loads": 1, "score_forwards": len(entries),
              "resources": old._capture_lifecycle_resources(device=device, started=started, phase="cold_complete")}
    publish(output_root / "cold-check.json", result)
    return result


def prepare_qualification(output_path: Path) -> dict[str, Any]:
    """The frozen old C2 contrast used ONLY for shared mechanical qualification."""
    anchor_path = old.PREPARATION / "inputs.json"
    _, manifest = old.validate_inputs(anchor_path, verify_sources=True)
    positives, conditional = [], []
    for case in manifest["positives"]:
        common = {"example_id": str(case["image"]["row_id"]), "image": case["image"],
                  "prompt_token_ids": case["prompt"]["token_ids"],
                  "prompt_token_ids_sha256": case["prompt"]["ids_sha256"]}
        positives.append({**common, "record_id": case["candidate_id"],
                          "prefix_token_ids": case["h"]["token_ids"],
                          "target_token_ids": case["c"]["token_ids"]})
        conditional.append({**common, "record_id": case["candidate_id"] + ":conditional",
            "prefix_token_ids": [*case["h"]["token_ids"], *case["c"]["token_ids"]],
            "target_token_ids": case["w"]["token_ids"],
            "kl_positions": list(range(len(case["w"]["token_ids"]))),
            "unknown_mask_policy": "literal_positions_only"})
    step = [{"record_id": row["record_id"], "weight": 1.0} for row in positives]
    return prepare_packet(output_path, lane="shared-mechanical-C2", anchor_input_path=anchor_path,
        margin_input_path=margin_engine.MARGIN_INPUT,
        normal_keys=[row["key"] for row in manifest["normals"]["cases"]],
        positive_records=positives, conditional_records=conditional,
        arms={"C2": {"steps": [step, step]}},
        weights={"positive": 1., "conditional_kl": 10., "normal_kl": 100., "margin": 10.},
        denominators={"positive": 3., "conditional_kl": 3., "normal_kl": 56., "margin": 56.},
        optimizer={"lr": 1e-5, "betas": [0.9, 0.999], "eps": 1e-8,
                   "weight_decay": 0., "foreach": False}, clip_gradient_norm=1.,
        runtime={"world_sizes": [2, 8], "max_rank_seconds": 1200,
            "max_cuda_allocated_bytes": 24 * 1024**3, "max_cuda_reserved_bytes": 24 * 1024**3,
            "max_rss_bytes": 32 * 1024**3, "max_model_forwards_per_rank": 160,
            "max_image_forwards_per_rank": 160})


def compare_qualification(*, output_root: Path, oracle_root: Path) -> dict[str, Any]:
    """Compare faithful objective and update; archived oracle has no raw gradients."""
    from safetensors import safe_open

    actual = verify_receipt(output_root)
    oracle = margin_engine.verify_receipt(oracle_root)
    require(actual["arm"] == "C2" and actual["updates"] == oracle["updates"] == 2
            and oracle["margin_weight"] == 10., "exact C2 margin10 oracle")
    require(actual["source_adapter"] == oracle["source_adapter"], "qualification anchors differ")
    cold = old.load_json(output_root / "cold-check.json")
    require(cold["status"] == "passed" and cold["training_receipt"] == binding(output_root / "receipt.json")
            and cold["saved_adapter"] == actual["saved_adapter"], "qualification cold identity")
    tolerance = QUALIFICATION_TOLERANCES
    checks, step_deltas = {}, []

    def close(key: str, a: float, b: float, *, atol: float, rtol: float = 0.) -> float:
        delta = a - b
        checks[key] = abs(delta) <= atol + rtol * abs(b)
        return delta

    for step in (1, 2):
        a = old.load_json(output_root / f"update-{step:02d}.json")
        b = old.load_json(oracle_root / f"update-{step:02d}.json")
        components = {key: close(f"step{step}.{key}", a["objective_components"][key],
            b["objective_components"][key], atol=tolerance["objective_component_atol"],
            rtol=tolerance["objective_component_rtol"]) for key in COMPONENTS}
        gradient = close(f"step{step}.gradient_norm", a["ranks"][0]["raw_gradient_norm"],
            b["ranks"][0]["raw_gradient_norm"], atol=tolerance["gradient_norm_atol"],
            rtol=tolerance["gradient_norm_rtol"])
        movement = close(f"step{step}.movement", a["ranks"][0]["movement_l2"],
            b["ranks"][0]["movement_l2"], atol=tolerance["update_norm_atol"],
            rtol=tolerance["update_norm_rtol"])
        step_deltas.append({"step": step, "components": components,
                            "gradient_norm_delta": gradient, "movement_l2_delta": movement})
    score_deltas = {}
    for key, scores in actual["final_live_positive_scores"].items():
        reference = oracle["final_live_positive_scores"][key]
        checks[f"score.{key}.discrete"] = all(scores[k] == reference[k]
                                              for k in ("token_count", "argmax_target_tokens"))
        score_deltas[key] = {k: close(f"score.{key}.{k}", scores[k], reference[k],
            atol=tolerance["positive_score_atol"]) for k in
            ("sum_logprob", "mean_logprob", "mean_target_margin", "min_target_margin")}
    roots = [actual["saved_adapter"]["root"], oracle["saved_adapter"]["root"], actual["source_adapter"]["root"]]
    maximum = difference_sq = expected_sq = actual_sq = dot = 0.
    tensors = scalars = 0
    with safe_open(str(Path(roots[0]) / "adapter_model.safetensors"), framework="pt") as a, \
            safe_open(str(Path(roots[1]) / "adapter_model.safetensors"), framework="pt") as b, \
            safe_open(str(Path(roots[2]) / "adapter_model.safetensors"), framework="pt") as source:
        require(a.keys() == b.keys() == source.keys(), "qualification saved tensor keys")
        for key in a.keys():
            av, bv, sv = (handle.get_tensor(key) for handle in (a, b, source))
            require(av.dtype == bv.dtype == sv.dtype == torch.float32 and av.shape == bv.shape == sv.shape,
                    "qualification saved tensor dtype/shape")
            delta_a, delta_b = (av - sv).double(), (bv - sv).double()
            difference = av.double() - bv.double()
            maximum = max(maximum, float(difference.abs().max()))
            difference_sq += float(difference.square().sum())
            expected_sq += float(delta_b.square().sum())
            actual_sq += float(delta_a.square().sum())
            dot += float((delta_a * delta_b).sum())
            tensors += 1
            scalars += av.numel()
    require(tensors == old.EXPECTED_TRAINABLE_TENSORS and scalars == old.EXPECTED_TRAINABLE_SCALARS,
            "qualification exact adapter tensor/scalar count")
    relative_l2 = math.sqrt(difference_sq / expected_sq)
    cosine = dot / math.sqrt(actual_sq * expected_sq)
    checks.update(adapter_max_abs=maximum <= tolerance["adapter_max_abs_difference"],
        adapter_delta_relative_l2=relative_l2 <= tolerance["adapter_delta_relative_l2"],
        adapter_delta_cosine=cosine >= tolerance["adapter_delta_cosine_minimum"])
    result = {"schema": "parallel_owner_training.qualification.v1",
        "status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "tolerances": tolerance, "candidate": binding(output_root / "receipt.json"),
        "oracle": binding(oracle_root / "receipt.json"), "cold_check": binding(output_root / "cold-check.json"),
        "step_deltas": step_deltas, "positive_score_deltas": score_deltas,
        "adapter_comparison": {"tensors": tensors, "scalars": scalars, "max_abs": maximum,
            "delta_relative_l2": relative_l2, "delta_cosine": cosine},
        "claim_boundary": "objective scaling and numerically compatible update/consumer; NOT raw-gradient vector parity"}
    publish(output_root / "qualification.json", result)
    require(result["status"] == "passed", "mechanical qualification failed; preserve raw comparison")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "launch", "rank", "cold-check"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--arm")
    parser.add_argument("--world-size", type=int, choices=(2, 8))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        packet, _, normals, _ = load_packet(args.input)
        print(json.dumps({"status": "CPU_valid", "lane": packet["lane"], "normal_count": len(normals),
                          "arms": {k: len(v["steps"]) for k, v in packet["arms"].items()}}))
    elif args.command == "cold-check":
        require(args.output_root is not None, "output root required")
        cold_check(input_path=args.input, output_root=args.output_root)
    else:
        require(args.arm is not None and args.world_size is not None and args.output_root is not None,
                "explicit arm/world/output required")
        fn = launch if args.command == "launch" else execute_rank
        fn(input_path=args.input, arm=args.arm, world_size=args.world_size, output_root=args.output_root)


if __name__ == "__main__":
    main()
