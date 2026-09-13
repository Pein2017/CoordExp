"""Serial training caller for the fixed-dose row-feedback pilot.

The research packet owns all choices that affect the comparison.  This module
only admits that packet, materializes its records once, calls the shared
causal runtime, and performs the declared AdamW updates.  Slot construction,
causal positions, and visible-logit alignment remain in :mod:`runtime`.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import time
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from probes.row_feedback import data, runtime, teacher


SCHEMA = "row_feedback.training_packet.v1"
PACKET_STATUSES = {"cost_only_pre_fit", "root_frozen_ready_for_fit"}
FIT_STATUS = "root_frozen_ready_for_fit"
SEED = 20260913
DOSES = (16, 32, 64)
ARMS = ("S", "F")
OPTIMIZER = {
    "lr": 1e-5,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0,
    "foreach": False,
}
LOSS = {
    "entry_c": 1,
    "post_completion_w": 1,
    "normal_kl": 100,
    "reduction": "token_mean_then_record_mean",
}
DIST_SCHEMA = "row_feedback.explicit_gradient_sum.v1"
DIST_ASSIGNMENT_SCHEMA = "row_feedback.distributed_assignment.v1"
DIST_ASSIGNMENT_POLICY = "lpt_physical_tokens_v1"
DIST_WORLD_SIZES = (2, 4)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_bound(reference: Mapping[str, Any], *, label: str) -> Any:
    _require(isinstance(reference, Mapping), f"{label} binding must be an object")
    path_value, digest_value = reference.get("path"), reference.get("sha256")
    _require(isinstance(path_value, str) and Path(path_value).is_absolute(),
             f"{label} binding path must be absolute")
    path = Path(path_value)
    _require(path.is_file(), f"missing bound {label}: {path}")
    _require(isinstance(digest_value, str) and file_hash(path) == digest_value,
             f"{label} source hash drift: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"bound {label} is not JSON: {path}") from exc


def _validate_code_bindings(packet: Mapping[str, Any]) -> None:
    bindings = packet.get("code_bindings")
    _require(isinstance(bindings, Mapping), "training packet code_bindings missing")
    expected = {"runtime", "training", "data", "teacher"}
    _require(set(bindings) == expected, "training packet code binding names")
    here = Path(__file__).resolve()
    for name in sorted(expected):
        reference = bindings[name]
        _require(isinstance(reference, Mapping), f"code binding {name} must be an object")
        path_value, digest_value = reference.get("path"), reference.get("sha256")
        _require(isinstance(path_value, str) and Path(path_value).is_absolute(),
                 f"code binding {name} path must be absolute")
        path = Path(path_value).resolve()
        _require(path.is_file() and isinstance(digest_value, str)
                 and file_hash(path) == digest_value,
                 f"code binding drift: {name}")
        if name == "training":
            _require(path == here, "training packet binds a different training caller")


def _validate_anchor(packet: Mapping[str, Any], bank: Mapping[str, Any]) -> None:
    anchor = packet.get("anchor_adapter")
    _require(isinstance(anchor, Mapping), "training packet anchor_adapter missing")
    root_value, fingerprint = anchor.get("root"), anchor.get("fingerprint")
    _require(isinstance(root_value, str) and Path(root_value).is_absolute()
             and Path(root_value).is_dir(), "training anchor adapter root")
    expected = bank["source_roles"]["fit_anchor"]["adapter"]
    _require(dict(anchor) == {"root": expected["root"], "fingerprint": expected["fingerprint"]},
             "training anchor does not match accepted bank fit anchor")
    # The explicit fingerprint check is separate from the path check so a
    # plausible replacement directory cannot pass packet admission.
    _require(fingerprint == "a7dcb56ea71ee8ab37a944778b7947c78ca22dfd321a0dcc59dde5227acecc80",
             "unexpected frozen N16 anchor fingerprint")


def _verify_loaded_anchor(
    packet: Mapping[str, Any], bank: Mapping[str, Any], identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Bridge the loader receipt to the canonical on-disk adapter descriptor."""

    from src.adapters.dora import inspect_dora_adapter_payload

    model_identity = identity.get("model_identity")
    _require(isinstance(model_identity, Mapping), "loaded model identity missing")
    loaded_adapter = model_identity.get("adapter")
    _require(isinstance(loaded_adapter, Mapping), "loaded adapter receipt missing")
    anchor = packet["anchor_adapter"]
    _require(loaded_adapter.get("adapter_path") == anchor["root"],
             "loaded adapter path differs from packet anchor")
    expected = bank["source_roles"]["fit_anchor"]["adapter"]
    base_model = expected["semantic_identity"]["base_model_name_or_path"]
    _require(loaded_adapter.get("base_model_path") == base_model,
             "loaded adapter base model differs from bank anchor")
    observed = inspect_dora_adapter_payload(anchor["root"], base_model)
    _require(observed == expected, "loaded adapter payload differs from bank anchor")
    return observed


def _validate_optimizer(value: Any) -> None:
    _require(value == OPTIMIZER, "optimizer contract changed")


def _validate_loss(value: Any) -> None:
    _require(value == LOSS, "loss contract changed")


def cyclic_pair_schedule(record_ids: Sequence[str], *, seed: int, updates: int) -> list[list[str]]:
    """Return the seeded permutation paired into a cyclic two-package schedule."""

    _require(not isinstance(seed, bool) and isinstance(seed, int), "schedule seed must be an integer")
    _require(not isinstance(updates, bool) and isinstance(updates, int) and updates > 0,
             "schedule updates must be positive")
    ids = [str(value) for value in record_ids]
    _require(len(ids) == 16 and len(set(ids)) == 16, "schedule requires sixteen unique packages")
    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    pairs = [shuffled[index:index + 2] for index in range(0, len(shuffled), 2)]
    return [list(pairs[index % len(pairs)]) for index in range(updates)]


def _validate_schedule(packet: Mapping[str, Any], bank: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    records = bank["records"]
    record_ids = [str(record["record_id"]) for record in records]
    schedule = packet.get("schedule")
    _require(isinstance(schedule, list) and schedule, "training packet schedule missing")
    _require(all(isinstance(pair, list) and len(pair) == 2 for pair in schedule),
             "schedule must contain two-package updates")
    _require(all(all(isinstance(record_id, str) and record_id in set(record_ids) for record_id in pair)
                 for pair in schedule), "schedule contains an unknown package")
    _require(all(pair[0] != pair[1] for pair in schedule), "schedule repeats a package within an update")
    status = packet["status"]
    dose = packet.get("dose")
    _require(source.get("schema") == "row_feedback.frozen_schedule_candidates.v1"
             and source.get("status") == "root_frozen_before_cost_and_fit"
             and source.get("seed") == SEED
             and source.get("scientific_fit_updates") == 0
             and source.get("bank") == packet["bank"],
             "schedule source binding changed")
    if status == FIT_STATUS:
        _require(type(dose) is int and dose in DOSES, "fit dose must be 16, 32, or 64")
        _require(len(schedule) == dose, "fit schedule length differs from dose")
        _require(schedule == source.get("schedule_prefixes", {}).get(str(dose))
                 and schedule == cyclic_pair_schedule(record_ids, seed=SEED, updates=dose),
                 "fit schedule is not the frozen seeded cyclic schedule")
    else:
        _require(dose in (1, "cost"), "cost packet dose must be one disposable update")
        _require(len(schedule) == 1, "cost packet must contain one explicit cost pair")
        _require(schedule[0] == source.get("cost_pair"), "cost pair differs from frozen schedule source")


def validate_packet(packet: Mapping[str, Any], *, validate_children: bool = True) -> None:
    """Fail closed on the immutable packet and all source identities it names."""

    _require(packet.get("schema") == SCHEMA, "unexpected training packet schema")
    _require(packet.get("status") in PACKET_STATUSES, "training packet status")
    _require(packet.get("seed") == SEED, "training packet seed changed")
    _validate_optimizer(packet.get("optimizer"))
    _validate_loss(packet.get("loss"))
    _validate_code_bindings(packet)
    for key in ("bank", "protection", "teacher_cache", "schedule_source"):
        _require(isinstance(packet.get(key), Mapping), f"training packet {key} binding missing")
    if not validate_children:
        return
    bank = _read_bound(packet["bank"], label="supervision bank")
    data.validate_bank(bank)
    protection = _read_bound(packet["protection"], label="protection records")
    data.validate_protection_records(protection)
    _require(packet["bank"] == protection["sources"]["supervision_bank"],
             "training packet bank/protection source binding changed")
    cache_path = Path(packet["teacher_cache"]["path"])
    _require(cache_path.name == "manifest.json", "teacher cache binding must name manifest.json")
    cache = _read_bound(packet["teacher_cache"], label="teacher cache manifest")
    teacher.validate_manifest(cache)
    _require(cache["source_protection_records"] == packet["protection"],
             "teacher cache/protection binding changed")
    _validate_anchor(packet, bank)
    _require(protection["fresh_n16_teacher"] == bank["source_roles"]["native_teacher"]["adapter"],
             "protection/native-teacher identity differs from bank")
    schedule_source = _read_bound(packet["schedule_source"], label="schedule source")
    _validate_schedule(packet, bank, schedule_source)


def load_packet(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read and validate packet, bank, protection, and teacher manifest once."""

    packet_path = Path(path).resolve()
    _require(packet_path.is_file(), f"missing training packet: {packet_path}")
    packet = json.loads(packet_path.read_text())
    validate_packet(packet)
    bank = _read_bound(packet["bank"], label="supervision bank")
    protection = _read_bound(packet["protection"], label="protection records")
    cache = _read_bound(packet["teacher_cache"], label="teacher cache manifest")
    packet["_packet_file"] = {"path": str(packet_path), "sha256": file_hash(packet_path)}
    return packet, bank, protection, cache


def token_mean_nll(logits: torch.Tensor, target_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
    """Mean visible-token NLL, retaining the logits graph for the caller."""

    if not isinstance(target_ids, torch.Tensor):
        target_ids = torch.tensor(list(target_ids), dtype=torch.long, device=logits.device)
    _require(logits.ndim == 2 and target_ids.ndim == 1 and logits.shape[0] == target_ids.numel()
             and target_ids.numel() > 0, "token-mean CE shape")
    return runtime.visible_nll_sum({"logits": logits, "target_ids": target_ids}) / target_ids.numel()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _distributed_work_items(
    package_ids: Sequence[str],
    bank_by_id: Mapping[str, Mapping[str, Any]],
    normal_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Describe the 58 indivisible replays using input structure only."""

    _require(len(package_ids) == 2 and package_ids[0] != package_ids[1],
             "one distributed update needs two packages")
    _require(len(normal_records) == 54, "distributed protection denominator must remain 54")
    items: list[dict[str, Any]] = []

    def add(
        kind: str,
        record_id: str,
        prompt: Sequence[int],
        history: Sequence[int],
        targets: Sequence[int],
        *,
        denominator: int,
        numerator: int,
        protected_positions: int = 0,
    ) -> None:
        prompt_ids = [int(value) for value in prompt]
        history_ids = [int(value) for value in history]
        target_ids = [int(value) for value in targets]
        structural_physical_tokens = (
            len(prompt_ids)
            + len(history_ids)
            + history_ids.count(runtime.BOX_END)
            + len(target_ids)
            + target_ids.count(runtime.BOX_END)
        )
        items.append({
            "item_id": f"{kind}:{record_id}",
            "serial_index": len(items),
            "kind": kind,
            "record_id": record_id,
            "global_numerator": numerator,
            "global_denominator": denominator,
            "loss_scale": numerator / denominator,
            "visible_target_tokens": len(target_ids),
            "protected_positions": protected_positions,
            "structural_physical_tokens": structural_physical_tokens,
        })

    for record_id_value in package_ids:
        record_id = str(record_id_value)
        record = bank_by_id[record_id]
        h = record["literal_rows"]["h"]["token_ids"]
        c = record["literal_rows"]["c"]["token_ids"]
        w = record["literal_rows"]["w"]["token_ids"]
        add("entry_c", record_id, record["prompt_token_ids"], h, c,
            denominator=2, numerator=1)
        add("post_completion_w", record_id, record["prompt_token_ids"],
            record["visible_history_token_ids"], w, denominator=2, numerator=1)
    for record in normal_records:
        key = str(record["key"])
        add("normal_kl", key, record["prompt_token_ids"], (), record["action_ids"],
            denominator=54, numerator=100, protected_positions=len(record["kl_positions"]))
    _require(len(items) == 58 and len({item["item_id"] for item in items}) == 58,
             "distributed update work coverage changed")
    return items


def build_distributed_assignment_manifest(
    *,
    schedule: Sequence[Sequence[str]],
    bank: Mapping[str, Any],
    protection: Mapping[str, Any],
    world_size: int,
) -> dict[str, Any]:
    """Compute the canonical structural LPT shards for packet binding."""

    _require(type(world_size) is int and world_size in DIST_WORLD_SIZES,
             "distributed world size must be 2 or 4")
    bank_by_id = {str(record["record_id"]): record for record in bank["records"]}
    normal_records = list(protection["records"])
    updates = []
    for update_index, package_ids in enumerate(schedule, start=1):
        items = _distributed_work_items(package_ids, bank_by_id, normal_records)
        shards: list[list[dict[str, Any]]] = [[] for _ in range(world_size)]
        loads = [0] * world_size
        for item in sorted(
            items,
            key=lambda value: (
                -int(value["structural_physical_tokens"]),
                str(value["kind"]),
                str(value["record_id"]),
            ),
        ):
            rank = min(range(world_size), key=lambda value: (loads[value], len(shards[value]), value))
            shards[rank].append(item)
            loads[rank] += int(item["structural_physical_tokens"])
        updates.append({
            "update_index": update_index,
            "package_ids": list(package_ids),
            "global_item_count": len(items),
            "global_structural_physical_tokens": sum(loads),
            "shards": [
                {
                    "rank": rank,
                    "structural_physical_tokens": loads[rank],
                    "items": sorted(shards[rank], key=lambda value: int(value["serial_index"])),
                }
                for rank in range(world_size)
            ],
        })
    return {
        "schema": DIST_ASSIGNMENT_SCHEMA,
        "policy": DIST_ASSIGNMENT_POLICY,
        "world_size": world_size,
        "global_denominators": {"entry_c": 2, "post_completion_w": 2, "normal_kl": 54},
        "updates": updates,
    }


def distributed_assignment_binding(
    *,
    schedule: Sequence[Sequence[str]],
    bank: Mapping[str, Any],
    protection: Mapping[str, Any],
    world_size: int,
) -> tuple[dict[str, Any], str]:
    """Return the canonical assignment manifest and packet-ready SHA256."""

    manifest = build_distributed_assignment_manifest(
        schedule=schedule, bank=bank, protection=protection, world_size=world_size,
    )
    return manifest, _canonical_hash(manifest)


def _validate_distributed_topology(
    packet: Mapping[str, Any],
    bank: Mapping[str, Any],
    protection: Mapping[str, Any],
    *,
    world_size: int,
) -> tuple[Mapping[str, Any], dict[str, Any], str]:
    execution = packet.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("distributed packet execution missing")
    topology = execution.get("distributed")
    if not isinstance(topology, Mapping):
        raise ValueError("distributed topology binding missing")
    expected = {
        "schema": DIST_SCHEMA,
        "world_size": world_size,
        "backend": "nccl",
        "gradient_reduction": "sum",
        "assignment_policy": DIST_ASSIGNMENT_POLICY,
        "assignment_sha256": topology.get("assignment_sha256"),
    }
    _require(dict(topology) == expected, "distributed topology contract changed")
    _require(world_size in DIST_WORLD_SIZES, "distributed world size must be 2 or 4")
    manifest, digest = distributed_assignment_binding(
        schedule=packet["schedule"], bank=bank, protection=protection, world_size=world_size,
    )
    _require(topology["assignment_sha256"] == digest,
             "distributed assignment manifest binding changed")
    return topology, manifest, digest


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _write_exclusive_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Persist one immutable scalar receipt without clobbering prior work."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _materialized_entry(qwen: Any, frontend: Any, config: Any, record: Mapping[str, Any]) -> dict[str, Any]:
    entry = runtime.materialize_record(qwen, frontend, config, record)
    _require(entry["prompt_ids"] == list(record["prompt_token_ids"]),
             f"materialized prompt mismatch: {record.get('record_id', record.get('key'))}")
    return entry


def _load_teacher_records(
    manifest: Mapping[str, Any], normal_records: Sequence[Mapping[str, Any]],
    *, manifest_validated: bool = False,
) -> dict[str, torch.Tensor]:
    """Validate the complete cache once, then load each bound tensor once.

    Calling ``teacher.load_teacher_record`` for every row would repeat its
    complete manifest and 54-file hash validation.  The manifest validation is
    equivalent here; the per-tensor checks below preserve the loader's
    contiguous CPU FP32, ordinal, normalization, and tensor-digest contract.
    """

    from safetensors.torch import load_file

    if not manifest_validated:
        teacher.validate_manifest(manifest)
    by_key = {str(record["key"]): record for record in manifest["records"]}
    result: dict[str, torch.Tensor] = {}
    for source in normal_records:
        key = str(source["key"])
        cached = by_key.get(key)
        _require(cached is not None, f"teacher key not found: {key}")
        _require(cached["prompt_token_ids_sha256"] == source["prompt_token_ids_sha256"]
                 and cached["action_ids_sha256"] == source["action_ids_sha256"]
                 and cached["visible_target_ordinals"] == source["kl_positions"],
                 f"teacher ordinal binding changed: {key}")
        tensor_ref = cached["log_probs"]
        loaded = load_file(tensor_ref["path"], device="cpu")
        _require(set(loaded) == {teacher.TENSOR_KEY}, f"teacher tensor keys changed: {key}")
        tensor = loaded[teacher.TENSOR_KEY].contiguous()
        _require(tensor.dtype == torch.float32 and tensor.ndim == 2
                 and tensor.shape == tuple(tensor_ref["shape"])
                 and tensor.shape[0] == len(source["kl_positions"])
                 and bool(torch.isfinite(tensor).all()), f"teacher tensor shape/dtype: {key}")
        _require(bool(torch.allclose(torch.logsumexp(tensor, dim=-1), torch.zeros(tensor.shape[0]),
                                     atol=teacher.NORMALIZATION_ATOL, rtol=0)),
                 f"teacher tensor normalization: {key}")
        _require(teacher._tensor_digest(tensor) == tensor_ref["tensor_sha256"],
                 f"teacher tensor digest changed: {key}")
        result[key] = tensor
    _require(len(result) == len(normal_records), "duplicate normal teacher key")
    return result


def _replay_loss(
    *, qwen: Any, entry: Mapping[str, Any], history: Sequence[int], targets: Sequence[int], arm: str,
    positions: Sequence[int] = (), reference: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any], str]:
    replay = runtime.replay_visible(
        qwen, entry["inputs"], prompt_ids=entry["prompt_ids"], history_ids=history,
        target_ids=targets, arm=arm, capture_feedback_sources=False,
    )
    if reference is None:
        loss = token_mean_nll(replay["logits"], replay["target_ids"])
        label = "ce"
    else:
        _require(positions and len(positions) == reference.shape[0], "normal teacher/position shape")
        loss = runtime.mapped_teacher_kl(
            replay["logits"][list(positions)], reference.to(replay["logits"].device), reduction="mean",
        )
        label = "kl"
    _require(bool(torch.isfinite(loss)), "nonfinite replay loss")
    return loss, replay, label


def _run_update(
    *, qwen: Any, optimizer: torch.optim.Optimizer, arm: str, package_ids: Sequence[str],
    bank_by_id: Mapping[str, Mapping[str, Any]], bank_entries: Mapping[str, Mapping[str, Any]],
    normal_records: Sequence[Mapping[str, Any]], normal_entries: Mapping[str, Mapping[str, Any]],
    teacher_by_key: Mapping[str, torch.Tensor], named: Sequence[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any]:
    _require(len(package_ids) == 2 and package_ids[0] != package_ids[1], "one update needs two packages")
    update_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    components = {"entry_c": 0.0, "post_completion_w": 0.0, "normal_kl": 0.0}
    counts = {"entry_c_tokens": 0, "post_completion_w_tokens": 0, "normal_protected_tokens": 0}
    model_forwards = image_forwards = slots = visible_tokens = 0
    replay_count = 0
    replay_receipts: list[dict[str, Any]] = []

    def backward_one(loss: torch.Tensor, replay: Mapping[str, Any], label: str, *, scale: float,
                     record_id: str, target_count: int) -> None:
        nonlocal model_forwards, image_forwards, slots, visible_tokens, replay_count
        scaled = loss * scale
        scaled.backward()
        components[label] += float(loss.detach())
        model_forwards += int(replay["model_forwards"])
        image_forwards += int(replay["image_forwards"])
        slots += int(replay["internal_slot_count"])
        visible_tokens += int(replay["visible_target_tokens"])
        replay_count += 1
        replay_receipts.append({
            "record_id": record_id,
            "kind": label,
            "loss": float(loss.detach()),
            "scale": scale,
            "visible_target_tokens": target_count,
            "internal_slot_count": replay["internal_slot_count"],
            "physical_token_count": replay["physical_token_count"],
            "model_forwards": replay["model_forwards"],
            "image_forwards": replay["image_forwards"],
            "timing": replay["timing"],
        })

    for record_id in package_ids:
        record = bank_by_id[record_id]
        entry = bank_entries[record_id]
        h = record["literal_rows"]["h"]["token_ids"]
        c = record["literal_rows"]["c"]["token_ids"]
        w = record["literal_rows"]["w"]["token_ids"]
        loss, replay, _ = _replay_loss(qwen=qwen, entry=entry, history=h, targets=c, arm=arm)
        backward_one(loss, replay, "entry_c", scale=0.5, record_id=record_id, target_count=len(c))
        counts["entry_c_tokens"] += len(c)
        del loss, replay
        loss, replay, _ = _replay_loss(
            qwen=qwen, entry=entry, history=record["visible_history_token_ids"], targets=w, arm=arm,
        )
        backward_one(loss, replay, "post_completion_w", scale=0.5, record_id=record_id, target_count=len(w))
        counts["post_completion_w_tokens"] += len(w)
        del loss, replay

    normal_scale = 100.0 / len(normal_records)
    for record in normal_records:
        key = str(record["key"])
        loss, replay, _ = _replay_loss(
            qwen=qwen, entry=normal_entries[key], history=(), targets=record["action_ids"], arm=arm,
            positions=record["kl_positions"], reference=teacher_by_key[key],
        )
        backward_one(loss, replay, "normal_kl", scale=normal_scale, record_id=key,
                     target_count=len(record["kl_positions"]))
        counts["normal_protected_tokens"] += len(record["kl_positions"])
        del loss, replay

    gradients = [parameter for _, parameter in named]
    _require(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                 for parameter in gradients),
             "selected DoRA gradient is missing or nonfinite")
    raw_norm = float(torch.nn.utils.clip_grad_norm_(
        gradients, 1.0, error_if_nonfinite=True, foreach=False,
    ))
    before = [parameter.detach().clone() for _, parameter in named]
    optimizer.step()
    movement = float(torch.sqrt(torch.stack([
        (parameter.detach() - prior).double().square().sum()
        for (_, parameter), prior in zip(named, before, strict=True)
    ]).sum()).item())
    _require(movement > 0 and torch.isfinite(torch.tensor(movement)), "optimizer made no finite update")
    component_record_means = {
        "entry_c": components["entry_c"] / 2,
        "post_completion_w": components["post_completion_w"] / 2,
        "normal_kl": components["normal_kl"] / len(normal_records),
    }
    return {
        "loss": component_record_means["entry_c"] + component_record_means["post_completion_w"]
                + 100 * component_record_means["normal_kl"],
        "components": components,
        "component_record_means": component_record_means,
        "gradient_norm_before_clip": raw_norm,
        "clip_norm": 1.0,
        "adapter_movement_l2": movement,
        "optimizer_steps": 1,
        "replays": replay_count,
        "model_forwards": model_forwards,
        "image_forwards": image_forwards,
        "internal_slots": slots,
        "visible_target_tokens": visible_tokens,
        "counts": counts,
        "replay_receipts": replay_receipts,
        "wall_seconds": time.perf_counter() - update_started,
    }


def _run_shard_backward(
    *,
    qwen: Any,
    arm: str,
    shard: Mapping[str, Any],
    bank_by_id: Mapping[str, Mapping[str, Any]],
    bank_entries: Mapping[str, Mapping[str, Any]],
    normal_by_key: Mapping[str, Mapping[str, Any]],
    normal_entries: Mapping[str, Mapping[str, Any]],
    teacher_by_key: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Backpropagate one rank's indivisible records with global scales."""

    started = time.perf_counter()
    components = {"entry_c": 0.0, "post_completion_w": 0.0, "normal_kl": 0.0}
    counts = {"entry_c_tokens": 0, "post_completion_w_tokens": 0,
              "normal_protected_tokens": 0}
    model_forwards = image_forwards = slots = visible_tokens = 0
    replay_receipts: list[dict[str, Any]] = []
    items = list(shard["items"])
    _require(bool(items), "distributed rank received an empty shard")

    for item in items:
        kind = str(item["kind"])
        record_id = str(item["record_id"])
        if kind == "entry_c":
            record = bank_by_id[record_id]
            history = record["literal_rows"]["h"]["token_ids"]
            targets = record["literal_rows"]["c"]["token_ids"]
            entry = bank_entries[record_id]
            positions: Sequence[int] = ()
            reference = None
            count_key = "entry_c_tokens"
            count_value = len(targets)
        elif kind == "post_completion_w":
            record = bank_by_id[record_id]
            history = record["visible_history_token_ids"]
            targets = record["literal_rows"]["w"]["token_ids"]
            entry = bank_entries[record_id]
            positions = ()
            reference = None
            count_key = "post_completion_w_tokens"
            count_value = len(targets)
        elif kind == "normal_kl":
            record = normal_by_key[record_id]
            history = ()
            targets = record["action_ids"]
            entry = normal_entries[record_id]
            positions = record["kl_positions"]
            reference = teacher_by_key[record_id]
            count_key = "normal_protected_tokens"
            count_value = len(positions)
        else:
            raise ValueError(f"unknown distributed work kind: {kind}")
        expected_scale = {"entry_c": 0.5, "post_completion_w": 0.5,
                          "normal_kl": 100.0 / 54}[kind]
        _require(float(item["loss_scale"]) == expected_scale,
                 f"distributed global loss scale changed: {kind}")
        loss, replay, _ = _replay_loss(
            qwen=qwen, entry=entry, history=history, targets=targets, arm=arm,
            positions=positions, reference=reference,
        )
        (loss * expected_scale).backward()
        _require(int(replay["physical_token_count"]) == int(item["structural_physical_tokens"]),
                 f"distributed structural cost mismatch: {item['item_id']}")
        components[kind] += float(loss.detach())
        counts[count_key] += count_value
        model_forwards += int(replay["model_forwards"])
        image_forwards += int(replay["image_forwards"])
        slots += int(replay["internal_slot_count"])
        visible_tokens += int(replay["visible_target_tokens"])
        replay_receipts.append({
            "item_id": item["item_id"],
            "serial_index": item["serial_index"],
            "record_id": record_id,
            "kind": kind,
            "loss": float(loss.detach()),
            "scale": expected_scale,
            "visible_target_tokens": item["visible_target_tokens"],
            "protected_positions": item["protected_positions"],
            "internal_slot_count": replay["internal_slot_count"],
            "physical_token_count": replay["physical_token_count"],
            "model_forwards": replay["model_forwards"],
            "image_forwards": replay["image_forwards"],
            "timing": replay["timing"],
        })
        del loss, replay
    return {
        "rank": int(shard["rank"]),
        "item_ids": [item["item_id"] for item in items],
        "item_count": len(items),
        "structural_physical_tokens": int(shard["structural_physical_tokens"]),
        "components": components,
        "counts": counts,
        "replays": len(items),
        "model_forwards": model_forwards,
        "image_forwards": image_forwards,
        "internal_slots": slots,
        "visible_target_tokens": visible_tokens,
        "replay_receipts": replay_receipts,
        "backward_wall_seconds": time.perf_counter() - started,
    }


def _aggregate_shard_summaries(
    summaries: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact global coverage and reconstruct the frozen serial loss."""

    _require(len(summaries) == len(assignment["shards"]), "distributed rank summary count")
    expected_items = [item for shard in assignment["shards"] for item in shard["items"]]
    expected_ids = {str(item["item_id"]) for item in expected_items}
    observed_ids = [str(item_id) for summary in summaries for item_id in summary["item_ids"]]
    _require(len(observed_ids) == 58 and len(set(observed_ids)) == 58
             and set(observed_ids) == expected_ids,
             "distributed update duplicated or omitted a replay record")
    kind_counts = {
        kind: sum(item["kind"] == kind for item in expected_items)
        for kind in ("entry_c", "post_completion_w", "normal_kl")
    }
    _require(kind_counts == {"entry_c": 2, "post_completion_w": 2, "normal_kl": 54},
             "distributed global loss denominators changed")
    components = {
        kind: sum(float(summary["components"][kind]) for summary in summaries)
        for kind in kind_counts
    }
    component_record_means = {
        "entry_c": components["entry_c"] / 2,
        "post_completion_w": components["post_completion_w"] / 2,
        "normal_kl": components["normal_kl"] / 54,
    }
    counts = {
        key: sum(int(summary["counts"][key]) for summary in summaries)
        for key in ("entry_c_tokens", "post_completion_w_tokens", "normal_protected_tokens")
    }
    expected_counts = {
        "entry_c_tokens": sum(int(item["visible_target_tokens"])
                              for item in expected_items if item["kind"] == "entry_c"),
        "post_completion_w_tokens": sum(int(item["visible_target_tokens"])
                                        for item in expected_items
                                        if item["kind"] == "post_completion_w"),
        "normal_protected_tokens": sum(int(item["protected_positions"])
                                       for item in expected_items if item["kind"] == "normal_kl"),
    }
    _require(counts == expected_counts, "distributed token/protection counts changed")
    _require(sum(int(summary["replays"]) for summary in summaries) == 58,
             "distributed replay count changed")
    replay_receipts = sorted(
        [row for summary in summaries for row in summary["replay_receipts"]],
        key=lambda row: int(row["serial_index"]),
    )
    return {
        "loss": component_record_means["entry_c"] + component_record_means["post_completion_w"]
                + 100 * component_record_means["normal_kl"],
        "components": components,
        "component_record_means": component_record_means,
        "counts": counts,
        "replays": sum(int(summary["replays"]) for summary in summaries),
        "model_forwards": sum(int(summary["model_forwards"]) for summary in summaries),
        "image_forwards": sum(int(summary["image_forwards"]) for summary in summaries),
        "internal_slots": sum(int(summary["internal_slots"]) for summary in summaries),
        "visible_target_tokens": sum(int(summary["visible_target_tokens"])
                                     for summary in summaries),
        "structural_physical_tokens": sum(int(summary["structural_physical_tokens"])
                                          for summary in summaries),
        "global_invariants": {
            "gradient_reduction": "sum",
            "loss_scales": {"entry_c": 0.5, "post_completion_w": 0.5,
                            "normal_kl": 100.0 / 54},
            "global_record_counts": kind_counts,
            "global_item_count": len(observed_ids),
            "coverage_sha256": _canonical_hash(sorted(observed_ids)),
        },
        "replay_receipts": replay_receipts,
        "rank_summaries": [
            {key: value for key, value in summary.items() if key != "replay_receipts"}
            for summary in summaries
        ],
    }


def _named_parameter_order_hash(named: Sequence[tuple[str, torch.nn.Parameter]]) -> str:
    return _canonical_hash([
        {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
        for name, parameter in named
    ])


def _sum_parameter_gradients(named: Sequence[tuple[str, torch.nn.Parameter]]) -> None:
    """SUM one same-order gradient buffer; never divide by world size."""

    gradients: list[torch.Tensor] = []
    for name, parameter in named:
        gradient = parameter.grad
        if gradient is None:
            raise ValueError(f"local distributed gradient missing: {name}")
        gradients.append(gradient)
    flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    _require(bool(torch.isfinite(flat).all()), "global distributed gradient is nonfinite")
    offset = 0
    with torch.no_grad():
        for gradient in gradients:
            count = gradient.numel()
            gradient.copy_(flat[offset:offset + count].view_as(gradient))
            offset += count
    _require(offset == flat.numel(), "distributed gradient buffer length changed")


def _apply_optimizer_step(
    optimizer: torch.optim.Optimizer,
    named: Sequence[tuple[str, torch.nn.Parameter]],
) -> dict[str, float | int]:
    gradients = [parameter for _, parameter in named]
    for name, parameter in named:
        gradient = parameter.grad
        _require(gradient is not None and bool(torch.isfinite(gradient).all()),
                 f"global selected DoRA gradient is missing or nonfinite: {name}")
    raw_norm = float(torch.nn.utils.clip_grad_norm_(
        gradients, 1.0, error_if_nonfinite=True, foreach=False,
    ))
    before = [parameter.detach().clone() for _, parameter in named]
    optimizer.step()
    movement = float(torch.sqrt(torch.stack([
        (parameter.detach() - prior).double().square().sum()
        for (_, parameter), prior in zip(named, before, strict=True)
    ]).sum()).item())
    _require(movement > 0 and bool(torch.isfinite(torch.tensor(movement))),
             "distributed optimizer made no finite update")
    return {
        "gradient_norm_before_clip": raw_norm,
        "clip_norm": 1.0,
        "adapter_movement_l2": movement,
        "optimizer_steps": 1,
    }


def _trainable_state_hash(named: Sequence[tuple[str, torch.nn.Parameter]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in named:
        value = parameter.detach()
        _require(bool(torch.isfinite(value).all()), f"nonfinite saved trainable tensor: {name}")
        cpu = value.to(device="cpu").contiguous()
        digest.update(name.encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode())
        digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _load_model_and_materialize(
    packet: Mapping[str, Any], bank: Mapping[str, Any], protection: Mapping[str, Any],
    cache: Mapping[str, Any],
    *, arm: str, device: torch.device,
) -> tuple[
    Any, Any, Any, dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]],
    dict[str, torch.Tensor], list[Mapping[str, Any]], Mapping[str, Any], Mapping[str, float],
]:
    _require(arm in ARMS, "unknown row-feedback arm")
    setup_started = time.perf_counter()
    model_started = time.perf_counter()
    qwen, frontend, config, identity = runtime.load_feedback_policy(
        adapter_path=packet["anchor_adapter"]["root"], device=device,
    )
    _verify_loaded_anchor(packet, bank, identity)
    model_seconds = time.perf_counter() - model_started
    surface_started = time.perf_counter()
    named, _frozen = runtime.bind_language_dora(qwen.model)
    for parameter in qwen.model.parameters():
        parameter.requires_grad_(False)
    for _, parameter in named:
        parameter.requires_grad_(True)
    surface_seconds = time.perf_counter() - surface_started
    # The shared runtime reads the exact train/dev union once and prepares all
    # bank inputs on CPU.  The update loop only reuses these immutable entries.
    materialization_started = time.perf_counter()
    all_bank_entries = runtime.materialize_bank_records(qwen, frontend, config, bank)
    selected_ids = sorted({record_id for pair in packet["schedule"] for record_id in pair})
    bank_entries = {record_id: all_bank_entries[record_id] for record_id in selected_ids}
    normal_records = list(protection["records"])
    normal_entries = {str(record["key"]): _materialized_entry(qwen, frontend, config, record)
                      for record in normal_records}
    materialization_seconds = time.perf_counter() - materialization_started
    teacher_started = time.perf_counter()
    teacher_by_key = _load_teacher_records(cache, normal_records, manifest_validated=True)
    teacher_seconds = time.perf_counter() - teacher_started
    _require(set(teacher_by_key) == set(normal_entries), "teacher/normal keys do not match")
    return qwen, frontend, config, bank_entries, normal_entries, teacher_by_key, normal_records, identity, {
        "model_load_seconds": model_seconds,
        "trainable_surface_bind_seconds": surface_seconds,
        "materialization_seconds": materialization_seconds,
        "teacher_cache_load_seconds": teacher_seconds,
        "setup_seconds": time.perf_counter() - setup_started,
    }


def _scalar_update_receipt(update: Mapping[str, Any]) -> dict[str, Any]:
    """Keep progress receipts eager and scalar, without replay graph objects."""

    return {key: value for key, value in update.items() if key != "replay_receipts"}


def run_training(*, packet_path: str | Path, output: str | Path, arm: str, mode: str) -> Mapping[str, Any]:
    """Run the declared serial fit or one disposable complete cost update."""

    _require(mode in ("cost", "fit"), "mode must be cost or fit")
    started = time.monotonic()
    admission_started = time.perf_counter()
    packet, bank, protection, cache = load_packet(packet_path)
    packet_admission_seconds = time.perf_counter() - admission_started
    if mode == "fit":
        _require(packet["status"] == FIT_STATUS, "fit requires root-frozen execution packet")
    output_path = Path(output).resolve()
    _require(not output_path.exists(), f"refuse to overwrite training output: {output_path}")
    output_path.mkdir(parents=True)
    random.seed(packet["seed"])
    torch.manual_seed(packet["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(packet["seed"])
    _require(torch.cuda.is_available(), "serial training requires an assigned CUDA device")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    _require(len(visible) == 1 and visible[0].isdigit(), "serial training requires one explicit physical GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    phase = "load"
    try:
        loaded = _load_model_and_materialize(
            packet, bank, protection, cache, arm=arm, device=device,
        )
        (qwen, _frontend, _config, bank_entries, normal_entries, teacher_by_key,
         normal_records, identity, setup_timing) = loaded
        named, frozen = runtime.bind_language_dora(qwen.model)
        frozen_versions = [(parameter, parameter._version) for _, parameter in frozen]
        optimizer = torch.optim.AdamW([parameter for _, parameter in named], **packet["optimizer"])
        updates = 1 if mode == "cost" else int(packet["dose"])
        schedules = packet["schedule"][:updates]
        bank_by_id = {str(record["record_id"]): record for record in bank["records"]}
        update_receipts = []
        update_dir = output_path / "updates"
        for update_index, pair in enumerate(schedules, 1):
            phase = f"update_{update_index}"
            update = _run_update(
                qwen=qwen, optimizer=optimizer, arm=arm, package_ids=pair,
                bank_by_id=bank_by_id, bank_entries=bank_entries,
                normal_records=normal_records, normal_entries=normal_entries,
                teacher_by_key=teacher_by_key, named=named,
            )
            _require(all(parameter._version == version for parameter, version in frozen_versions),
                     "optimizer mutated a frozen model parameter")
            scalar_update = _scalar_update_receipt(update)
            update_receipts.append(scalar_update)
            _write_exclusive_json(update_dir / f"update-{update_index:04d}.json", {
                "schema": "row_feedback.training_update_receipt.v1",
                "status": "complete",
                "packet": packet["_packet_file"],
                "mode": mode,
                "arm": arm,
                "update_index": update_index,
                "pair": pair,
                "update": scalar_update,
            })
            del update
        phase = "adapter_save"
        save_started = time.perf_counter()
        saved = runtime.save_feedback_adapter(
            qwen, source_adapter=packet["anchor_adapter"]["root"], output=output_path / "adapter",
        )
        save_seconds = time.perf_counter() - save_started
        receipt = {
            "schema": "row_feedback.training_receipt.v1",
            "status": "cost_only_disposable_complete_update_saved" if mode == "cost"
                      else "fit_complete_serial_arm_saved",
            "claim_boundary": "Serial training execution and loss accounting only; no endpoint quality claim.",
            "mode": mode,
            "arm": arm,
            "packet": packet["_packet_file"],
            "packet_status": packet["status"],
            "code_bindings": packet["code_bindings"],
            "anchor_adapter": packet["anchor_adapter"],
            "loaded_identity": identity,
            "bank": packet["bank"], "protection": packet["protection"], "teacher_cache": packet["teacher_cache"],
            "schedule_source": packet["schedule_source"],
            "seed": packet["seed"], "optimizer": packet["optimizer"], "loss": packet["loss"],
            "dose": 1 if mode == "cost" else packet["dose"],
            "schedule": schedules,
            "trainable_surface": {"tensor_count": len(named),
                                  "scalar_count": sum(parameter.numel() for _, parameter in named)},
            "materialization": {
                "bank": {record_id: entry["prepared_inputs_sha256"]
                         for record_id, entry in bank_entries.items()},
                "normal": {key: entry["prepared_inputs_sha256"]
                            for key, entry in normal_entries.items()},
                "policy": "all native inputs prepared once on CPU before optimizer updates",
            },
            "updates": update_receipts,
            "saved_adapter": saved,
            "phase_timing": {
                "packet_admission_seconds": packet_admission_seconds,
                **setup_timing,
                "update_wall_seconds": sum(row["wall_seconds"] for row in update_receipts),
                "adapter_save_seconds": save_seconds,
            },
            "resources": {
                "wall_seconds": time.monotonic() - started,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "peak_rss_bytes": _rss_bytes(),
                "physical_gpu": int(visible[0]),
                "model_forwards": sum(row["model_forwards"] for row in update_receipts),
                "image_forwards": sum(row["image_forwards"] for row in update_receipts),
                "internal_slots": sum(row["internal_slots"] for row in update_receipts),
                "visible_target_tokens": sum(row["visible_target_tokens"] for row in update_receipts),
                "replays": sum(row["replays"] for row in update_receipts),
                "materialized_bank_records": len(bank_entries),
                "admitted_bank_records": len(bank["records"]),
                "materialized_normal_records": len(normal_records),
                "inputs_cpu_materialized_once": True,
            },
        }
        _write_exclusive_json(output_path / "training-receipt.json", receipt)
        return receipt
    except BaseException as exc:
        failure = {"schema": "row_feedback.training_failure.v1", "status": "failed",
                   "mode": mode, "arm": arm, "phase": phase,
                   "error": f"{type(exc).__name__}: {exc}",
                   "packet": {"path": str(Path(packet_path).resolve()), "sha256": file_hash(packet_path)},
                   "wall_seconds": time.monotonic() - started}
        _write_exclusive_json(output_path / "failure.json", failure)
        raise


def _distributed_environment() -> tuple[int, int, int, list[int]]:
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except (KeyError, ValueError) as exc:
        raise ValueError("distributed training requires torchrun rank environment") from exc
    visible_values = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    _require(world_size in DIST_WORLD_SIZES and 0 <= rank < world_size
             and 0 <= local_rank < world_size, "distributed rank/world environment changed")
    _require(len(visible_values) == world_size and all(value.isdigit() for value in visible_values),
             "distributed training requires one explicit physical GPU per rank")
    return rank, world_size, local_rank, [int(value) for value in visible_values]


def _synchronize_phase_error(rank: int, phase: str, error: BaseException | None) -> None:
    local = None if error is None else {
        "rank": rank,
        "phase": phase,
        "error": f"{type(error).__name__}: {error}",
    }
    rows: list[Mapping[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(rows, local)
    failures = [row for row in rows if row is not None]
    if failures:
        raise RuntimeError(f"distributed phase failed: {json.dumps(failures, sort_keys=True)}")


def _all_gather_object(value: Any) -> list[Any]:
    rows: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(rows, value)
    return rows


def _run_distributed_update(
    *,
    rank: int,
    qwen: Any,
    optimizer: torch.optim.Optimizer,
    arm: str,
    assignment: Mapping[str, Any],
    bank_by_id: Mapping[str, Mapping[str, Any]],
    bank_entries: Mapping[str, Mapping[str, Any]],
    normal_by_key: Mapping[str, Mapping[str, Any]],
    normal_entries: Mapping[str, Mapping[str, Any]],
    teacher_by_key: Mapping[str, torch.Tensor],
    named: Sequence[tuple[str, torch.nn.Parameter]],
    parameter_order_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one globally equivalent update over explicit independent shards."""

    update_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    shard = assignment["shards"][rank]
    local: dict[str, Any] | None = None
    local_error: BaseException | None = None
    try:
        local = _run_shard_backward(
            qwen=qwen, arm=arm, shard=shard, bank_by_id=bank_by_id,
            bank_entries=bank_entries, normal_by_key=normal_by_key,
            normal_entries=normal_entries, teacher_by_key=teacher_by_key,
        )
    except BaseException as exc:
        local_error = exc
    _synchronize_phase_error(rank, f"update_{assignment['update_index']}_local_backward", local_error)
    if local is None:
        raise AssertionError("local shard result vanished after synchronized validation")
    summaries = _all_gather_object(local)
    global_update = _aggregate_shard_summaries(summaries, assignment)

    local_error = None
    try:
        for name, parameter in named:
            _require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
                     f"local distributed gradient missing or nonfinite: {name}")
    except BaseException as exc:
        local_error = exc
    _synchronize_phase_error(rank, f"update_{assignment['update_index']}_gradient_ready", local_error)
    _sum_parameter_gradients(named)
    step: dict[str, float | int] | None = None
    local_error = None
    try:
        step = _apply_optimizer_step(optimizer, named)
    except BaseException as exc:
        local_error = exc
    _synchronize_phase_error(rank, f"update_{assignment['update_index']}_optimizer_step", local_error)
    if step is None:
        raise AssertionError("optimizer step vanished after synchronized validation")
    step_rows = _all_gather_object({"rank": rank, **step})
    for key in ("gradient_norm_before_clip", "adapter_movement_l2"):
        values = [float(row[key]) for row in step_rows]
        _require(max(values) == min(values), f"distributed ranks differ after gradient SUM: {key}")
    _require(all(int(row["optimizer_steps"]) == 1 for row in step_rows),
             "distributed rank optimizer-step count changed")
    wall_rows = _all_gather_object(time.perf_counter() - update_started)
    global_update.update(step)
    global_update.update({
        "optimizer_steps_per_rank": [int(row["optimizer_steps"]) for row in step_rows],
        "scientific_optimizer_steps": 1,
        "parameter_order_sha256": parameter_order_sha256,
        "gradient_sum_buffer_scalars": sum(parameter.numel() for _, parameter in named),
        "wall_seconds": max(float(value) for value in wall_rows),
        "rank_wall_seconds": [float(value) for value in wall_rows],
    })
    local.update({
        "global_gradient_norm_before_clip": step["gradient_norm_before_clip"],
        "global_adapter_movement_l2": step["adapter_movement_l2"],
        "optimizer_steps": 1,
        "update_wall_seconds": float(wall_rows[rank]),
    })
    return global_update, local


def run_distributed_training(
    *, packet_path: str | Path, output: str | Path, arm: str, mode: str,
) -> Mapping[str, Any]:
    """Run explicit sharded backward, gradient SUM, and one global AdamW step."""

    _require(mode in ("cost", "fit"), "mode must be cost or fit")
    _require(torch.cuda.is_available(), "distributed training requires CUDA")
    rank, world_size, local_rank, physical_gpus = _distributed_environment()
    # NCCL object collectives allocate on the current device.  Select the
    # torchrun-local GPU before the first synchronized admission check so each
    # process cannot accidentally use cuda:0.
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    started = time.monotonic()
    output_path = Path(output).resolve()
    phase = "packet_admission"
    try:
        packet_result: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
        local_error: BaseException | None = None
        admission_started = time.perf_counter()
        try:
            packet_result = load_packet(packet_path)
        except BaseException as exc:
            local_error = exc
        _synchronize_phase_error(rank, phase, local_error)
        if packet_result is None:
            raise AssertionError("packet result vanished after synchronized validation")
        packet, bank, protection, cache = packet_result
        packet_admission_seconds = time.perf_counter() - admission_started
        phase = "topology_admission"
        topology_result: tuple[Mapping[str, Any], dict[str, Any], str] | None = None
        local_error = None
        try:
            if mode == "fit":
                _require(packet["status"] == FIT_STATUS,
                         "fit requires root-frozen execution packet")
            topology_result = _validate_distributed_topology(
                packet, bank, protection, world_size=world_size,
            )
        except BaseException as exc:
            local_error = exc
        _synchronize_phase_error(rank, phase, local_error)
        if topology_result is None:
            raise AssertionError("topology result vanished after synchronized validation")
        topology, assignment_manifest, assignment_sha256 = topology_result

        phase = "output_create"
        local_error = None
        if rank == 0:
            try:
                _require(not output_path.exists(),
                         f"refuse to overwrite training output: {output_path}")
                output_path.mkdir(parents=True)
            except BaseException as exc:
                local_error = exc
        _synchronize_phase_error(rank, phase, local_error)
        dist.barrier()

        random.seed(packet["seed"])
        torch.manual_seed(packet["seed"])
        torch.cuda.manual_seed_all(packet["seed"])
        torch.cuda.reset_peak_memory_stats(device)

        phase = "load"
        loaded = None
        local_error = None
        try:
            loaded = _load_model_and_materialize(
                packet, bank, protection, cache, arm=arm, device=device,
            )
        except BaseException as exc:
            local_error = exc
        _synchronize_phase_error(rank, phase, local_error)
        if loaded is None:
            raise AssertionError("loaded model vanished after synchronized validation")
        (qwen, _frontend, _config, bank_entries, normal_entries, teacher_by_key,
         normal_records, identity, setup_timing) = loaded
        phase = "trainable_surface"
        surface = None
        local_error = None
        try:
            named, frozen = runtime.bind_language_dora(qwen.model)
            _require(len(named) == 588, "distributed trainable tensor count changed")
            surface = (
                named,
                [(parameter, parameter._version) for _, parameter in frozen],
                _named_parameter_order_hash(named),
                _trainable_state_hash(named),
            )
        except BaseException as exc:
            local_error = exc
        _synchronize_phase_error(rank, phase, local_error)
        if surface is None:
            raise AssertionError("trainable surface vanished after synchronized validation")
        named, frozen_versions, parameter_order_sha256, initial_state_sha256 = surface
        surface_rows = _all_gather_object({
            "parameter_order_sha256": parameter_order_sha256,
            "initial_trainable_state_sha256": initial_state_sha256,
        })
        _require(len({_canonical_hash(row) for row in surface_rows}) == 1,
                 "distributed initial trainable surface differs across ranks")
        optimizer = torch.optim.AdamW([parameter for _, parameter in named], **packet["optimizer"])
        updates = 1 if mode == "cost" else int(packet["dose"])
        schedules = packet["schedule"][:updates]
        _require(len(assignment_manifest["updates"]) >= updates,
                 "distributed assignment is shorter than requested dose")
        bank_by_id = {str(record["record_id"]): record for record in bank["records"]}
        normal_by_key = {str(record["key"]): record for record in normal_records}
        global_updates: list[dict[str, Any]] = []
        local_updates: list[dict[str, Any]] = []
        for update_index, pair in enumerate(schedules, start=1):
            phase = f"update_{update_index}"
            assignment = assignment_manifest["updates"][update_index - 1]
            _require(assignment["package_ids"] == list(pair), "distributed update pair changed")
            global_update, local_update = _run_distributed_update(
                rank=rank, qwen=qwen, optimizer=optimizer, arm=arm, assignment=assignment,
                bank_by_id=bank_by_id, bank_entries=bank_entries,
                normal_by_key=normal_by_key, normal_entries=normal_entries,
                teacher_by_key=teacher_by_key, named=named,
                parameter_order_sha256=parameter_order_sha256,
            )
            local_error = None
            try:
                _require(all(parameter._version == version for parameter, version in frozen_versions),
                         "distributed optimizer mutated a frozen model parameter")
            except BaseException as exc:
                local_error = exc
            _synchronize_phase_error(rank, f"update_{update_index}_frozen_parameters", local_error)
            global_updates.append(_scalar_update_receipt(global_update))
            local_updates.append({key: value for key, value in local_update.items()
                                  if key != "replay_receipts"})
            local_update_path = (output_path / "ranks" / f"rank-{rank:04d}" / "updates"
                                 / f"update-{update_index:04d}.json")
            global_update_path = output_path / "updates" / f"update-{update_index:04d}.json"
            local_error = None
            try:
                _write_exclusive_json(local_update_path, {
                    "schema": "row_feedback.distributed_rank_update_receipt.v1",
                    "status": "complete", "rank": rank, "world_size": world_size,
                    "packet": packet["_packet_file"], "mode": mode, "arm": arm,
                    "update_index": update_index, "pair": pair,
                    "assignment_sha256": assignment_sha256,
                    "shard": assignment["shards"][rank], "update": local_update,
                })
            except BaseException as exc:
                local_error = exc
            _synchronize_phase_error(rank, f"update_{update_index}_rank_receipts", local_error)
            local_error = None
            if rank == 0:
                try:
                    _write_exclusive_json(global_update_path, {
                        "schema": "row_feedback.distributed_update_receipt.v1",
                        "status": "complete", "world_size": world_size,
                        "packet": packet["_packet_file"], "mode": mode, "arm": arm,
                        "update_index": update_index, "pair": pair,
                        "assignment_sha256": assignment_sha256,
                        "update": _scalar_update_receipt(global_update),
                    })
                except BaseException as exc:
                    local_error = exc
            _synchronize_phase_error(rank, f"update_{update_index}_global_receipt", local_error)
            del global_update, local_update

        phase = "trainable_state"
        state_sha256: str | None = None
        local_error = None
        try:
            state_sha256 = _trainable_state_hash(named)
        except BaseException as exc:
            local_error = exc
        _synchronize_phase_error(rank, phase, local_error)
        if state_sha256 is None:
            raise AssertionError("trainable state hash vanished after synchronized validation")
        state_rows = _all_gather_object(state_sha256)
        _require(len(set(state_rows)) == 1, "distributed trainable weights differ across ranks")

        phase = "adapter_save"
        save_started = time.perf_counter()
        saved: Mapping[str, Any] | None = None
        local_error = None
        if rank == 0:
            try:
                saved = runtime.save_feedback_adapter(
                    qwen, source_adapter=packet["anchor_adapter"]["root"],
                    output=output_path / "adapter",
                )
            except BaseException as exc:
                local_error = exc
        _synchronize_phase_error(rank, phase, local_error)
        saved_rows: list[Mapping[str, Any] | None] = [saved]
        dist.broadcast_object_list(saved_rows, src=0)
        saved = saved_rows[0]
        _require(saved is not None, "rank0 saved adapter receipt missing")
        save_seconds = time.perf_counter() - save_started

        materialization = {
            "bank": {record_id: entry["prepared_inputs_sha256"]
                     for record_id, entry in bank_entries.items()},
            "normal": {key: entry["prepared_inputs_sha256"]
                       for key, entry in normal_entries.items()},
        }
        materialization_hash = _canonical_hash(materialization)
        materialization_rows = _all_gather_object(materialization_hash)
        _require(len(set(materialization_rows)) == 1,
                 "distributed materialized input hashes differ across ranks")
        rank_receipt = {
            "schema": "row_feedback.distributed_rank_receipt.v1",
            "status": "complete", "rank": rank, "world_size": world_size,
            "physical_gpu": physical_gpus[local_rank], "mode": mode, "arm": arm,
            "packet": packet["_packet_file"], "assignment_sha256": assignment_sha256,
            "parameter_order_sha256": parameter_order_sha256,
            "initial_trainable_state_sha256": initial_state_sha256,
            "trainable_state_sha256": state_sha256,
            "all_trainable_tensors_finite": True,
            "frozen_parameter_versions_unchanged": True,
            "setup_timing": setup_timing,
            "adapter_save_seconds": save_seconds if rank == 0 else 0.0,
            "updates": local_updates,
            "resources": {
                "wall_seconds": time.monotonic() - started,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "peak_rss_bytes": _rss_bytes(),
                "model_forwards": sum(row["model_forwards"] for row in local_updates),
                "image_forwards": sum(row["image_forwards"] for row in local_updates),
                "internal_slots": sum(row["internal_slots"] for row in local_updates),
                "visible_target_tokens": sum(row["visible_target_tokens"] for row in local_updates),
                "replays": sum(row["replays"] for row in local_updates),
            },
        }
        rank_rows = _all_gather_object(rank_receipt)
        rank_receipt_path = output_path / "ranks" / f"rank-{rank:04d}" / "rank-receipt.json"
        local_error = None
        try:
            _write_exclusive_json(rank_receipt_path, rank_receipt)
        except BaseException as exc:
            local_error = exc
        _synchronize_phase_error(rank, "rank_receipts", local_error)
        dist.barrier()
        admission_rows = _all_gather_object(packet_admission_seconds)
        setup_rows = _all_gather_object(setup_timing)

        receipt_path = output_path / "training-receipt.json"
        phase = "global_receipt_prepare"
        receipt: dict[str, Any] | None = None
        local_error = None
        if rank == 0:
            try:
                rank_receipt_files = [
                    {
                        "rank": row["rank"],
                        "path": str(output_path / "ranks" / f"rank-{row['rank']:04d}"
                                    / "rank-receipt.json"),
                        "sha256": file_hash(output_path / "ranks" / f"rank-{row['rank']:04d}"
                                             / "rank-receipt.json"),
                        "resources": row["resources"],
                    }
                    for row in rank_rows
                ]
                receipt = {
                    "schema": "row_feedback.distributed_training_receipt.v1",
                    "status": ("cost_only_disposable_distributed_complete_update_saved"
                               if mode == "cost" else "fit_complete_distributed_arm_saved"),
                    "claim_boundary": (
                        "Explicit gradient-SUM training execution and loss accounting only; "
                        "no endpoint quality claim."
                    ),
                    "mode": mode, "arm": arm, "packet": packet["_packet_file"],
                    "packet_status": packet["status"], "code_bindings": packet["code_bindings"],
                    "anchor_adapter": packet["anchor_adapter"], "loaded_identity": identity,
                    "bank": packet["bank"], "protection": packet["protection"],
                    "teacher_cache": packet["teacher_cache"],
                    "schedule_source": packet["schedule_source"],
                    "seed": packet["seed"], "optimizer": packet["optimizer"], "loss": packet["loss"],
                    "dose": 1 if mode == "cost" else packet["dose"], "schedule": schedules,
                    "topology": dict(topology),
                    "assignment": {"sha256": assignment_sha256, "manifest": assignment_manifest},
                    "parameter_order_sha256": parameter_order_sha256,
                    "initial_trainable_state_sha256": initial_state_sha256,
                    "trainable_state_sha256": state_sha256,
                    "all_trainable_tensors_finite": True,
                    "frozen_parameter_versions_unchanged": True,
                    "trainable_surface": {"tensor_count": len(named),
                                          "scalar_count": sum(parameter.numel() for _, parameter in named)},
                    "materialization": {**materialization,
                                        "policy": "all native inputs prepared once per rank on CPU"},
                    "updates": global_updates,
                    "saved_adapter": saved,
                    "rank_receipts": rank_receipt_files,
                    "phase_timing": {
                        "packet_admission_seconds_per_rank": admission_rows,
                        "setup_per_rank": setup_rows,
                        "update_wall_seconds": sum(row["wall_seconds"] for row in global_updates),
                        "adapter_save_seconds": save_seconds,
                    },
                    "resources": {
                        "wall_seconds": time.monotonic() - started,
                        "physical_gpus": physical_gpus,
                        "max_peak_cuda_allocated_bytes": max(
                            row["resources"]["peak_cuda_allocated_bytes"] for row in rank_rows),
                        "max_peak_cuda_reserved_bytes": max(
                            row["resources"]["peak_cuda_reserved_bytes"] for row in rank_rows),
                        "max_peak_rss_bytes": max(row["resources"]["peak_rss_bytes"]
                                                  for row in rank_rows),
                        "sum_rank_gpu_process_wall_seconds": sum(
                            row["resources"]["wall_seconds"] for row in rank_rows),
                        "model_forwards": sum(row["model_forwards"] for row in global_updates),
                        "image_forwards": sum(row["image_forwards"] for row in global_updates),
                        "internal_slots": sum(row["internal_slots"] for row in global_updates),
                        "visible_target_tokens": sum(
                            row["visible_target_tokens"] for row in global_updates),
                        "replays": sum(row["replays"] for row in global_updates),
                        "materialized_bank_records_per_rank": len(bank_entries),
                        "admitted_bank_records": len(bank["records"]),
                        "materialized_normal_records_per_rank": len(normal_records),
                        "inputs_cpu_materialized_once_per_rank": True,
                    },
                }
            except BaseException as exc:
                local_error = exc
        _synchronize_phase_error(rank, phase, local_error)
        # Every rank has finished its updates, save, and rank receipt before
        # rank 0 publishes the canonical success receipt.
        dist.barrier()
        phase = "global_receipt_write"
        local_error = None
        if rank == 0:
            try:
                if receipt is None:
                    raise AssertionError("global receipt vanished after preparation")
                _write_exclusive_json(receipt_path, receipt)
            except BaseException as exc:
                local_error = exc
        _synchronize_phase_error(rank, phase, local_error)
        dist.barrier()
        return json.loads(receipt_path.read_text())
    except BaseException as exc:
        if output_path.is_dir():
            failure_path = output_path / "ranks" / f"rank-{rank:04d}" / "failure.json"
            if not failure_path.exists():
                try:
                    _write_exclusive_json(failure_path, {
                        "schema": "row_feedback.distributed_training_failure.v1",
                        "status": "failed", "rank": rank, "world_size": world_size,
                        "mode": mode, "arm": arm, "phase": phase,
                        "error": f"{type(exc).__name__}: {exc}",
                        "packet": {"path": str(Path(packet_path).resolve()),
                                   "sha256": file_hash(packet_path)},
                        "wall_seconds": time.monotonic() - started,
                    })
                except BaseException:
                    pass
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--mode", choices=("cost", "fit"), required=True)
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()
    if args.distributed:
        receipt = run_distributed_training(
            packet_path=args.packet, output=args.output, arm=args.arm, mode=args.mode,
        )
    else:
        receipt = run_training(packet_path=args.packet, output=args.output, arm=args.arm, mode=args.mode)
    if not args.distributed or int(os.environ["RANK"]) == 0:
        print(json.dumps({"schema": receipt["schema"], "status": receipt["status"],
                          "output": str(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
