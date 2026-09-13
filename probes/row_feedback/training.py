"""Serial training caller for the fixed-dose row-feedback pilot.

The research packet owns all choices that affect the comparison.  This module
only admits that packet, materializes its records once, calls the shared
causal runtime, and performs the declared AdamW updates.  Slot construction,
causal positions, and visible-logit alignment remain in :mod:`runtime`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import time
from typing import Any, Mapping, Sequence

import torch

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
                "materialized_bank_records": len(bank["records"]),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--mode", choices=("cost", "fit"), required=True)
    args = parser.parse_args()
    receipt = run_training(packet_path=args.packet, output=args.output, arm=args.arm, mode=args.mode)
    print(json.dumps({"schema": receipt["schema"], "status": receipt["status"],
                      "output": str(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
