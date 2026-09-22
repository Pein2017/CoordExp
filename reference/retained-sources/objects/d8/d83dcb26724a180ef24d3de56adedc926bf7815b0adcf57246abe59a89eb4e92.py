"""Paired credible-successor training contract and loss consumer.

The module owns the new A/B mathematics and schedule.  It deliberately reuses
the N16 producer's record materialization, reference-KL implementation,
full-vocabulary reference margin, model loader, optimizer settings, and adapter
receipts.  The throughput lane owns only ``batched_aligned_logits``.

No command in this module starts training without a separately bound root
grant.  In particular, gradient-scale calibration is a no-update operation and
its result must be sealed into the packet before a launch can be granted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from typing import Any

import torch

from probes.dora_owner_learning.candidate_opportunity import require
from probes.dora_owner_learning import margin_preserved_train as margin_engine
from probes.parallel_owner_research import training as old_training
from probes.owner_successor_scale.replay import batched_aligned_logits
from src.losses import aligned_token_logprobs


SCHEMA = "owner_successor_scale.training.inputs.v1"
GRANT_SCHEMA = "owner_successor_scale.training.root_grant.v1"
INTEGRATED_SMOKE_ACCEPTANCE_SCHEMA = "owner_successor_scale.training.integrated_smoke_acceptance.v1"
CALIBRATION_SCHEMA = "owner_successor_scale.training.calibration.v1"
PHYSICAL_BANK_SCHEMA = "owner_successor_scale.physical_training_bank.v1"
PHYSICAL_DECISIONS_SCHEMA = "owner_successor_scale.physical_training_decisions.v1"
REPLAY_ACCEPTANCE_SCHEMA = "owner_successor_scale.replay_acceptance.v1"
RAW_SOURCE_COMPOSITION_SCHEMA = "owner_successor_scale.raw_source_composition.v1"
UPDATES = 256
BLOCK_SIZE = 8
BLOCKS = UPDATES // BLOCK_SIZE
OLD_PACKAGE_COUNT = 16
REFERENCE_COUNT = 54
MIN_NEW_PACKAGES = 32
MIN_NEW_IMAGES = 16
MAX_NEW_PACKAGES = 128
MAX_NEW_TOKENS = 3084
REFRESH_INTERVAL = 8
FORK_GAMMA = 1.0
FORK_TARGET_RATIO = 0.25
COEFFICIENTS = {
    "old_ce": 1.0,
    "new_ce": 1.0,
    "old_witness_kl": 10.0,
    "new_witness_kl": 10.0,
    "normal_kl": 100.0,
    "normal_margin": 10.0,
}
OPTIMIZER = {
    "lr": 1e-5,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "foreach": False,
}
CLIP_GRADIENT_NORM = 1.0
RUNTIME_LIMIT_KEYS = {
    "max_rank_seconds", "max_cuda_allocated_bytes", "max_cuda_reserved_bytes",
    "max_rss_bytes", "max_model_forwards_per_rank", "max_image_forwards_per_rank",
}
STAGED_REPLAY_STAGES = ["calibration", "integrated_two_rank_one_update_save_cold"]
RAW_SOURCE_ALLOWED_DIFFERENCES = (
    "image.declared_path",
    "metadata.source.original_images[]",
    "source.row_number",
    "source.row_sha256",
    "source.source_path",
)
N16_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-12-native-owner-scale-and-state/scale/training/full-fixedP-N16-v2"
)
N16_WEIGHTS_SHA256 = "092b47a56b2b50475e97e0a2f1fcbd058ba6f53f131cc7750af992ff4922db35"
N16_ADAPTER_FINGERPRINT = "a7dcb56ea71ee8ab37a944778b7947c78ca22dfd321a0dcc59dde5227acecc80"
GENERATION = {
    "decode_mode": "greedy",
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "max_new_tokens_total": MAX_NEW_TOKENS,
    "refresh_interval_updates": REFRESH_INTERVAL,
    "history_policy": "same_frozen_h_plus",
    "negative_sampling": "none",
}


def _ids(value: Any, label: str, *, nonempty: bool = True) -> list[int]:
    require(
        isinstance(value, list)
        and (bool(value) or not nonempty)
        and all(isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in value),
        label,
    )
    return list(value)


def _digest_ids(values: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(values), separators=(",", ":")).encode()).hexdigest()


def first_divergence(repeat_ids: Sequence[int], credible_ids: Sequence[int]) -> dict[str, Any]:
    """Freeze the literal shared row prefix ``u`` and its first distinct tokens."""
    repeated, credible = list(repeat_ids), list(credible_ids)
    require(repeated and credible, "fork rows must be nonempty")
    index = next((i for i, pair in enumerate(zip(repeated, credible)) if pair[0] != pair[1]), None)
    require(index is not None, "repeat and credible rows have no first divergence")
    return {
        "prefix_token_ids": credible[:index],
        "divergence_index": index,
        "repeat_token_id": repeated[index],
        "credible_token_id": credible[index],
    }


def fork_hinge(
    fork_logits: torch.Tensor,
    *,
    repeat_token_id: int,
    credible_token_id: int,
    gamma: float = FORK_GAMMA,
) -> torch.Tensor:
    """Return ``[gamma + z(r_f) - z(c_f)]_+`` at one frozen literal fork."""
    require(fork_logits.dtype == torch.float32 and fork_logits.ndim == 1, "FP32 fork logits")
    require(
        isinstance(repeat_token_id, int)
        and isinstance(credible_token_id, int)
        and repeat_token_id != credible_token_id
        and 0 <= repeat_token_id < fork_logits.numel()
        and 0 <= credible_token_id < fork_logits.numel(),
        "literal distinct fork token IDs",
    )
    require(type(gamma) in (int, float) and math.isfinite(gamma) and gamma == FORK_GAMMA,
            "frozen one-logit fork gamma")
    return torch.relu(fork_logits[repeat_token_id] - fork_logits[credible_token_id] + gamma)


def normalized_fork_loss(
    event_logits: Sequence[torch.Tensor | None],
    event_specs: Sequence[Mapping[str, Any] | None],
    *,
    package_count: int,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Average over the fixed package bank; inactive refreshes remain literal zero.

    The denominator is never the number of observed repeats.  That prevents
    an outcome-dependent increase in dose when only a few packages repeat.
    """
    require(package_count >= MIN_NEW_PACKAGES and len(event_logits) == len(event_specs) == package_count,
            "fork bank cardinality")
    require(zero.ndim == 0 and zero.dtype == torch.float32, "fork differentiable zero")
    total = zero * 0.0
    active = 0
    raw = []
    for logits, spec in zip(event_logits, event_specs, strict=True):
        require((logits is None) == (spec is None), "fork event/logit gating mismatch")
        if spec is None:
            raw.append(0.0)
            continue
        value = fork_hinge(
            logits,
            repeat_token_id=spec["repeat_token_id"],
            credible_token_id=spec["credible_token_id"],
        )
        total = total + value
        raw.append(float(value.detach()))
        active += 1
    return total / package_count, {
        "active_events": active,
        "registered_packages": package_count,
        "denominator": package_count,
        "raw_hinges": raw,
    }


def calibrate_fork_lambda(
    *, common_gradient_norm: float, raw_fork_gradient_norm: float, active_events: int
) -> float:
    """One initial calibration: ``lambda*||g_fork||=.25*||g_common||``."""
    require(active_events > 0, "HOLD: calibration refresh observed no strict-repeat events")
    require(math.isfinite(common_gradient_norm) and common_gradient_norm > 0,
            "HOLD: common objective has zero/nonfinite gradient")
    require(math.isfinite(raw_fork_gradient_norm) and raw_fork_gradient_norm > 0,
            "HOLD: fork objective has zero/nonfinite gradient")
    value = FORK_TARGET_RATIO * common_gradient_norm / raw_fork_gradient_norm
    require(math.isfinite(value) and value > 0, "HOLD: invalid calibrated fork lambda")
    return value


def gradient_l2(loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter]) -> float:
    """Measure an objective gradient without changing ``.grad`` or parameters."""
    parameters = tuple(parameters)
    require(loss.ndim == 0 and parameters, "scalar calibration loss and selected parameters")
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=False)
    require(all(bool(torch.isfinite(gradient).all()) for gradient in gradients),
            "HOLD: nonfinite calibration gradient")
    return math.sqrt(sum(float(gradient.detach().double().square().sum()) for gradient in gradients))


def _rotated(values: Sequence[str], offset: int) -> list[str]:
    values = list(values)
    return values[offset % len(values):] + values[:offset % len(values)]


def balanced_schedule(old_ids: Sequence[str], new_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Create 32 deterministic eight-update blocks with one exposure/package/block."""
    old_ids, new_ids = list(old_ids), list(new_ids)
    require(len(old_ids) == OLD_PACKAGE_COUNT and len(set(old_ids)) == len(old_ids),
            "exact 16 distinct original packages")
    require(MIN_NEW_PACKAGES <= len(new_ids) <= MAX_NEW_PACKAGES
            and len(set(new_ids)) == len(new_ids), "32..128 distinct new packages")
    require(not set(old_ids).intersection(new_ids), "old/new package IDs collide")
    steps: list[dict[str, Any]] = []
    for block in range(BLOCKS):
        # Rotations change co-exposure while preserving exact per-block mass.
        old = _rotated(old_ids, 2 * block)
        new = _rotated(new_ids, block * math.ceil(len(new_ids) / BLOCK_SIZE))
        new_counts = [len(new_ids) // BLOCK_SIZE + int(i < len(new_ids) % BLOCK_SIZE)
                      for i in range(BLOCK_SIZE)]
        cursor = 0
        for index in range(BLOCK_SIZE):
            selected = new[cursor: cursor + new_counts[index]]
            cursor += new_counts[index]
            steps.append({
                "update": block * BLOCK_SIZE + index + 1,
                "block": block + 1,
                "refresh_before_update": index == 0,
                "old_positive_ids": old[2 * index:2 * index + 2],
                "new_positive_ids": selected,
                "new_positive_per_record_scale": BLOCK_SIZE / len(new_ids),
            })
        require(cursor == len(new_ids), "new package block partition")
    return steps


def validate_schedule(steps: Sequence[Mapping[str, Any]], old_ids: Sequence[str], new_ids: Sequence[str]) -> None:
    require(len(steps) == UPDATES, "exact 256-update schedule")
    old_ids, new_ids = list(old_ids), list(new_ids)
    old_counts: Counter[str] = Counter()
    new_counts: Counter[str] = Counter()
    for block in range(BLOCKS):
        rows = steps[block * BLOCK_SIZE:(block + 1) * BLOCK_SIZE]
        require([row["refresh_before_update"] for row in rows] == [True] + [False] * 7,
                "refresh only at frozen eight-update block boundary")
        block_old = [value for row in rows for value in row["old_positive_ids"]]
        block_new = [value for row in rows for value in row["new_positive_ids"]]
        require(Counter(block_old) == Counter(old_ids), "each old package once per block")
        require(Counter(block_new) == Counter(new_ids), "each new package once per block")
        require(all(len(row["old_positive_ids"]) == 2 for row in rows), "two old positives/update")
        require(max(map(len, (row["new_positive_ids"] for row in rows)))
                - min(map(len, (row["new_positive_ids"] for row in rows))) <= 1,
                "new-package step sizes differ by more than one")
        require(all(row["new_positive_per_record_scale"] == BLOCK_SIZE / len(new_ids) for row in rows),
                "new positive denominator changed by step")
        old_counts.update(block_old)
        new_counts.update(block_new)
    require(set(old_counts.values()) == {32} and set(new_counts.values()) == {32},
            "exact 32 exposures/package")


def _positive_nll(logits: torch.Tensor, target_ids: Sequence[int]) -> torch.Tensor:
    targets = torch.tensor(list(target_ids), dtype=torch.long, device=logits.device)
    require(logits.dtype == torch.float32 and logits.ndim == 2 and logits.shape[0] == targets.numel(),
            "FP32 aligned positive logits")
    return -aligned_token_logprobs(logits, targets).sum()


def reference_bank_items(
    current_logits: Sequence[torch.Tensor],
    entries: Sequence[Mapping[str, Any]],
    reference_logp: Mapping[str, torch.Tensor],
    *,
    margins: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Reuse the N16 KL/margin consumers on an ordered replay bank.

    ``reference_logp`` must have been cached from the frozen N16 model before
    either arm updates.  References are detached again here so an accidental
    live teacher graph cannot enter optimization.
    """
    require(len(current_logits) == len(entries) and bool(entries), "reference bank logits/entries")
    kl_items: list[torch.Tensor] = []
    margin_items: list[torch.Tensor] = []
    for logits, entry in zip(current_logits, entries, strict=True):
        record = entry["record"]
        record_id = record["record_id"]
        targets = torch.tensor(record["target_token_ids"], dtype=torch.long, device=logits.device)
        positions = record["kl_positions"]
        require(record_id in reference_logp, "reference item absent from frozen N16 teacher cache")
        kl_items.append(old_training.old.reference_kl(
            logits, reference_logp[record_id].detach().to(logits.device), positions
        ))
        if margins is not None:
            require(record_id in margins, "normal item absent from fixed margin bank")
            penalty, _ = margin_engine.worst_margin_penalty(logits, targets, margins[record_id])
            margin_items.append(penalty)
    require(margins is None or len(margin_items) == len(entries), "normal margin item count")
    return kl_items, margin_items


def independent_bank_objective(
    *,
    old_positive: Sequence[torch.Tensor],
    old_positive_targets: Sequence[Sequence[int]],
    new_positive: Sequence[torch.Tensor],
    new_positive_targets: Sequence[Sequence[int]],
    old_witness_kl: Sequence[torch.Tensor],
    new_witness_kl: Sequence[torch.Tensor],
    normal_kl: Sequence[torch.Tensor],
    normal_margin: Sequence[torch.Tensor],
    new_package_count: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine already-computed scalar items with the six frozen bank reductions."""
    require(len(old_positive) == len(old_positive_targets) == 2, "two old CE records/update")
    require(len(new_positive) == len(new_positive_targets) in {
        new_package_count // BLOCK_SIZE, math.ceil(new_package_count / BLOCK_SIZE)},
        "balanced new CE records/update")
    require(len(old_witness_kl) == OLD_PACKAGE_COUNT, "old witness bank denominator")
    require(len(new_witness_kl) == new_package_count, "new witness bank denominator")
    require(len(normal_kl) == len(normal_margin) == REFERENCE_COUNT, "54-reference bank denominator")
    old_ce = sum((_positive_nll(x, y) for x, y in zip(old_positive, old_positive_targets, strict=True)),
                 old_positive[0].new_zeros(())) / 2
    new_ce = sum((_positive_nll(x, y) for x, y in zip(new_positive, new_positive_targets, strict=True)),
                 new_positive[0].new_zeros(())) * (BLOCK_SIZE / new_package_count)
    components = {
        "old_ce": old_ce,
        "new_ce": new_ce,
        "old_witness_kl": sum(old_witness_kl, old_ce.new_zeros(())) / OLD_PACKAGE_COUNT,
        "new_witness_kl": sum(new_witness_kl, old_ce.new_zeros(())) / new_package_count,
        "normal_kl": sum(normal_kl, old_ce.new_zeros(())) / REFERENCE_COUNT,
        "normal_margin": sum(normal_margin, old_ce.new_zeros(())) / REFERENCE_COUNT,
    }
    total = sum((components[key] * COEFFICIENTS[key] for key in COEFFICIENTS), old_ce.new_zeros(()))
    return total, {key: float(value.detach()) for key, value in components.items()}


def _binding(path: str | Path) -> dict[str, str]:
    return old_training.binding(Path(path))


def _binding_shape(value: Any, label: str) -> None:
    require(isinstance(value, Mapping) and set(value) == {"path", "sha256"}
            and Path(value["path"]).is_absolute()
            and isinstance(value["sha256"], str) and len(value["sha256"]) == 64, label)


def _normalized_binding(value: Mapping[str, Any], label: str) -> dict[str, str]:
    """Project producer bindings onto the two fields consumed by training."""
    require(isinstance(value, Mapping), label)
    binding = {"path": value.get("path"), "sha256": value.get("sha256")}
    _binding_shape(binding, label)
    require(_binding(binding["path"]) == binding, f"{label} changed")
    return binding


def _raw_candidate_id(row: Mapping[str, Any]) -> str:
    """Derive only the typed ID needed to select bound raw rows."""
    if "example_id" in row:
        value = row["example_id"]
        require(isinstance(value, str) and value, "selected canonical raw example ID")
        return value
    from src.data.examples import _current_source_metadata, _source_example_id

    metadata = _current_source_metadata(row.get("metadata"))
    return _source_example_id(row, metadata=metadata)


def _selected_bound_raw(
    sources: Sequence[Mapping[str, Any]], wanted_ids: set[str]
) -> dict[str, Any]:
    """Load typed rows only for exact wanted IDs from immutable bound sources."""
    from src.data.examples import raw_example_from_jsonl_row

    selected: dict[str, Any] = {}
    for source in sources:
        normalized = _normalized_binding(source, "selected raw source")
        path = Path(normalized["path"])
        with path.open("r", encoding="utf-8") as handle:
            for row_number, raw_line in enumerate(handle, start=1):
                line = raw_line.rstrip("\n")
                require(bool(line.strip()), "blank selected raw-source row")
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid selected raw-source JSON: {path}:{row_number}") from error
                require(isinstance(payload, Mapping), "selected raw-source row mapping")
                candidate = _raw_candidate_id(payload)
                if candidate not in wanted_ids:
                    continue
                row = raw_example_from_jsonl_row(
                    payload, jsonl_path=path, row_number=row_number, raw_line=line,
                )
                require(row.example_id == candidate, "selected raw candidate/typed ID identity")
                require(candidate not in selected, f"duplicate selected augmentation raw ID: {candidate}")
                selected[candidate] = row
    return selected


def _raw_artifact_differences(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.extend(_raw_artifact_differences(left.get(key), right.get(key), child))
        return result
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        result = []
        for left_item, right_item in zip(left, right, strict=True):
            result.extend(_raw_artifact_differences(left_item, right_item, f"{prefix}[]"))
        return result
    return [] if left == right else [prefix]


def _raw_materialization_projection(row: Any) -> dict[str, Any]:
    """Native planning fields; source spelling is retained separately as provenance."""
    value = row.to_artifact_dict()
    return {
        "example_id": value["example_id"],
        "image": {key: value["image"][key] for key in ("path", "width", "height", "stat")},
        "objects": value["objects"],
        "metadata": {
            **value["metadata"],
            "source": {
                key: item for key, item in value["metadata"]["source"].items()
                if key != "original_images"
            },
        },
        "source_format": value["source"]["source_format"],
    }


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def compose_materialization_raw_lookup(
    *, base_sources: Sequence[Mapping[str, Any]], augmentation_sources: Sequence[Mapping[str, Any]],
    required_example_ids: Sequence[str], config_input: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep the configured raw owner and add only required rows from a bound superset."""
    base_bindings = [_normalized_binding(source, "base raw source") for source in base_sources]
    augmentation_bindings = [
        _normalized_binding(source, "augmentation raw source") for source in augmentation_sources
    ]
    require(base_bindings and augmentation_bindings, "base and augmentation raw sources required")
    all_paths = [source["path"] for source in [*base_bindings, *augmentation_bindings]]
    require(len(set(all_paths)) == len(all_paths), "raw composition source paths overlap")
    required = sorted(set(required_example_ids))
    require(required and len(required) == len(required_example_ids), "unique required raw example IDs")
    base = old_training._load_materialization_raw(
        {"materialization_raw_sources": base_bindings}, config_input=config_input,
    )
    selected = _selected_bound_raw(augmentation_bindings, set(base) | set(required))
    lookup = dict(base)
    collisions: list[dict[str, Any]] = []
    additions: list[dict[str, Any]] = []
    allowed = set(RAW_SOURCE_ALLOWED_DIFFERENCES)
    for example_id in sorted(selected):
        row = selected[example_id]
        if example_id in base:
            left = base[example_id]
            differences = sorted(set(_raw_artifact_differences(
                left.to_artifact_dict(), row.to_artifact_dict()
            )))
            require(set(differences).issubset(allowed)
                    and _raw_materialization_projection(left) == _raw_materialization_projection(row),
                    f"raw collision differs outside allowed source provenance: {example_id}")
            collisions.append({
                "example_id": example_id,
                "difference_paths": differences,
                "base_source": left.source.to_artifact_dict(),
                "augmentation_source": row.source.to_artifact_dict(),
                "materialization_projection_sha256": _json_digest(
                    _raw_materialization_projection(left)
                ),
            })
            continue
        if example_id in required:
            lookup[example_id] = row
            additions.append({
                "example_id": example_id,
                "source": row.source.to_artifact_dict(),
                "materialization_projection_sha256": _json_digest(
                    _raw_materialization_projection(row)
                ),
            })
    missing = sorted(set(required) - set(lookup))
    require(not missing, f"required raw examples absent from composed lookup: {missing[:3]}")
    required_projections = {
        example_id: _json_digest(_raw_materialization_projection(lookup[example_id]))
        for example_id in required
    }
    receipt = {
        "schema": RAW_SOURCE_COMPOSITION_SCHEMA,
        "status": "passed_exact_projection",
        "policy": "retain_configured_base_owner_add_required_only",
        "base_sources": base_bindings,
        "augmentation_sources": augmentation_bindings,
        "allowed_difference_paths": list(RAW_SOURCE_ALLOWED_DIFFERENCES),
        "counts": {
            "base_records": len(base),
            "augmentation_selected_records": len(selected),
            "allowed_equivalent_collisions": len(collisions),
            "required_additions": len(additions),
            "required_records": len(required),
            "final_lookup_records": len(lookup),
        },
        "required_example_ids_sha256": _json_digest(required),
        "required_materialization_projections_sha256": _json_digest(required_projections),
        "collision_records_sha256": _json_digest(collisions),
        "addition_records_sha256": _json_digest(additions),
        "base_rows_replaced": 0,
    }
    return lookup, receipt


def _required_raw_example_ids(
    old_packet: Mapping[str, Any], packages: Sequence[Mapping[str, Any]]
) -> list[str]:
    values = [
        *(str(record["example_id"]) for record in old_packet["positive_records"]),
        *(str(record["example_id"]) for record in old_packet["conditional_records"]),
        *(str(value) for value in old_packet["normal_keys"]),
        *(str(package["c_record"]["example_id"]) for package in packages),
        *(str(package["witness_record"]["example_id"]) for package in packages),
    ]
    return sorted(set(values))


def _jsonl_rows(binding: Mapping[str, Any], label: str) -> list[tuple[str, dict[str, Any]]]:
    normalized = _normalized_binding(binding, label)
    rows = []
    for line in Path(normalized["path"]).read_text().splitlines():
        if line.strip():
            rows.append((line, json.loads(line)))
    return rows


def _line_sha256(line: str) -> str:
    return hashlib.sha256(line.encode()).hexdigest()


def validate_replay_acceptance(value: Mapping[str, Any], *, verify_files: bool) -> None:
    """Accept replay capacity for calibration/smoke without pre-requiring that smoke."""
    require(value.get("schema") == REPLAY_ACCEPTANCE_SCHEMA
            and value.get("status") == "lead_accepted_staged", "staged replay acceptance status")
    require(value.get("authorized_stages") == STAGED_REPLAY_STAGES,
            "replay acceptance stage scope")
    require(value.get("api") == "batched_aligned_logits(model,entries)"
            and value.get("microbatch_size") == 2
            and value.get("activation_checkpointing") is False,
            "accepted MB2/checkpoint-off replay choice")
    evidence = value.get("technical_evidence", {})
    require(set(evidence) == {"final_warm_start_diagnostic", "capacity_probe"},
            "replay technical evidence fields")
    for name, binding in evidence.items():
        _binding_shape(binding, f"bound replay evidence {name}")
        if verify_files:
            require(_binding(binding["path"]) == binding, f"bound replay evidence {name} changed")
    if verify_files:
        diagnostic = json.loads(Path(evidence["final_warm_start_diagnostic"]["path"]).read_text())
        capacity = json.loads(Path(evidence["capacity_probe"]["path"]).read_text())
        require(diagnostic.get("schema") == "owner_successor_scale.final_warm_start_diagnostic.v1"
                and diagnostic.get("status") == "passed"
                and diagnostic.get("logical_optimizer_updates") == 2,
                "passed final warm-start replay evidence")
        selected = capacity.get("selected", {})
        require(capacity.get("schema") == "owner_successor_scale.capacity_probe.v1"
                and capacity.get("status") == "passed" and selected.get("status") == "passed"
                and selected.get("microbatch_size") == 2
                and selected.get("activation_checkpointing") is False,
                "passed MB2/checkpoint-off capacity evidence")
    runtime = value.get("runtime", {})
    require(set(runtime) == {"world_sizes", *RUNTIME_LIMIT_KEYS}
            and runtime["world_sizes"] == [2, 8], "staged replay runtime envelope fields")
    for field in RUNTIME_LIMIT_KEYS:
        require(type(runtime[field]) in (int, float) and math.isfinite(runtime[field])
                and runtime[field] > 0, f"positive staged runtime envelope {field}")
    require(runtime["max_cuda_allocated_bytes"] == 69_793_218_560
            and runtime["max_cuda_reserved_bytes"] == 85_899_345_920
            and runtime["max_rss_bytes"] == 17_179_869_184
            and runtime["max_model_forwards_per_rank"] == 1_600_000
            and runtime["max_image_forwards_per_rank"] == 1_600_000
            and runtime["max_rank_seconds"] == 86_400,
            "frozen staged replay resource envelope")


def _physical_record(
    *, job: Mapping[str, Any], row: Mapping[str, Any], image_source: Mapping[str, Any],
    review: Mapping[str, Any], decisions_binding: Mapping[str, str],
) -> dict[str, Any]:
    """Convert one root-admitted exact job/history into the old materializer schema."""
    frozen = image_source["frozen"]
    case, plan = frozen["case"], frozen["case"]["image_plan"]
    prompt = _ids(frozen["prompt_token_ids"], "supply prompt token IDs")
    h_plus = _ids(job["h_token_ids"], "supply exact h_plus token IDs", nonempty=False)
    credible = _ids(job["c_token_ids"], "supply credible c token IDs")
    local_w = row["local_w"]
    require(local_w.get("status") == "candidate_local_w", "supply row lacks local immediate w")
    witness = _ids(local_w["w_token_ids"], "supply immediate w token IDs")
    require(_digest_ids(credible) == review["c"]["token_ids_sha256"], "review/source c token identity")
    require(_digest_ids(witness) == review["w"]["token_ids_sha256"], "review/source w token identity")
    require(_digest_ids(witness) == local_w["w_token_ids_sha256"], "local w token identity")
    require(str(job["job_id"]) == str(row["job_id"]) == str(review["job_id"]),
            "job/row/review exact identity")
    require(str(job["example_id"]) == str(frozen["example_id"]) == str(review["example_id"]),
            "example identity")
    require(int(job["image_id"]) == int(row["image_id"]) == int(frozen["image_id"])
            == int(review["image_id"]), "image identity")
    require(case["row_id"] == frozen["example_id"] and plan["status"] == "ok",
            "frozen image materialization identity")
    image = {
        "row_id": case["row_id"],
        "row_index": case["row_index"],
        "image_id": int(frozen["image_id"]),
        "image_path": plan["image_path"],
        "image_sha256": plan["image_content_sha256"],
        "observed_image_grid_thw": plan["observed_image_grid_thw"],
        "executed_media_sha256": plan["executed_media_sha256"],
    }
    common = {
        "example_id": frozen["example_id"],
        "prompt_token_ids": prompt,
        "prompt_token_ids_sha256": old_training.old.digest_ids(prompt),
        "image": image,
    }
    c_record = {
        **common, "record_id": f"{job['job_id']}:c", "prefix_token_ids": h_plus,
        "target_token_ids": credible,
    }
    witness_record = {
        **common, "record_id": f"{job['job_id']}:w_kl",
        "prefix_token_ids": [*h_plus, *credible], "target_token_ids": witness,
        "kl_positions": list(range(len(witness))), "unknown_mask_policy": "literal_positions_only",
    }
    old_training.validate_record(c_record, kind="positive", verify_image=True)
    old_training.validate_record(witness_record, kind="conditional", verify_image=True)
    evidence = dict(decisions_binding)
    return {
        "package_id": str(job["job_id"]), "image_id": int(job["image_id"]),
        "h_plus_token_ids": h_plus, "credible_c_token_ids": credible,
        "c_trust": {"status": "physically_trusted", "supported_not_yet_covered": True,
                    "evidence": evidence},
        "w_trust": {"status": "physically_trusted", "immediate_successor": True,
                    "evidence": evidence},
        "c_record": c_record, "witness_record": witness_record,
        "case": case, "golden": frozen["golden"], "h_text": job["h_text"],
        "provenance": {
            "visual_group_id": review["visual_group_id"],
            "source_job_ordinal": review["source_job_ordinal"],
            "source_job_sha256": review["source_job_sha256"],
            "source_row_ordinal": review["source_row_ordinal"],
            "source_row_sha256": review["source_row_sha256"],
            "history_index": job["history_index"], "candidate_index": job["candidate_index"],
            "first_owner_axis": review["first_owner_axis"],
            "same_image_aliases": review["same_image_aliases"],
            "execution_aliases_same_visual_group": review["execution_aliases_same_visual_group"],
        },
    }


def materialize_physical_bank(
    *, pool_path: str | Path, confirmation_selection_path: str | Path,
    review_index_paths: Sequence[str | Path], root_decisions_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Build the exact real-source c/w bank, or fail before writing any output.

    Root decisions are labels, not nominations: every admitted visual group must
    separately affirm physical c trust, immediate-w trust, and its canonical
    exact execution history. HOLD/reject rows never become packages.
    """
    require(review_index_paths, "at least one sealed physical review index")
    pool_path, confirmation_selection_path = Path(pool_path), Path(confirmation_selection_path)
    decisions_path, destination = Path(root_decisions_path), Path(output).resolve()
    require(not destination.exists(), "no physical bank overwrite")
    pool = json.loads(pool_path.read_text())
    confirmation = json.loads(confirmation_selection_path.read_text())
    decisions = json.loads(decisions_path.read_text())
    pool_binding, confirmation_binding = _binding(pool_path), _binding(confirmation_selection_path)
    review_bindings = [_binding(path) for path in review_index_paths]
    require(pool.get("schema") == "owner_successor_scale.supply_pool.v1"
            and pool.get("status") == "frozen_before_new_rollouts", "frozen 4096 supply pool")
    require(len(pool.get("image_ids", [])) == 4096 and len(set(pool["image_ids"])) == 4096,
            "exact 4096-image supply pool")
    require(confirmation.get("schema") == "native_owner_successor_scale_throughput.confirmation_selection.v1"
            and confirmation.get("status") == "frozen_cpu_no_model_calls", "frozen confirmation selection")
    require(decisions.get("schema") == PHYSICAL_DECISIONS_SCHEMA
            and decisions.get("status") == "root_decisions_complete", "completed root physical decisions")
    require(decisions.get("pool") == pool_binding
            and decisions.get("confirmation_selection") == confirmation_binding
            and decisions.get("review_indexes") == review_bindings,
            "root decisions do not bind exact pool/confirmation/review indexes")
    decision_rows = decisions.get("decisions")
    require(isinstance(decision_rows, list), "root physical decisions list")
    require(len({row.get("visual_group_id") for row in decision_rows}) == len(decision_rows),
            "duplicate root visual-group decision")

    reviews: dict[str, Mapping[str, Any]] = {}
    groups: dict[str, Mapping[str, Any]] = {}
    shard_bindings: dict[str, Mapping[str, Any]] = {}
    for path in review_index_paths:
        index = json.loads(Path(path).read_text())
        require(index.get("schema") == "owner_successor_scale.physical_admission_review_index.v1"
                and index.get("status") in {"review_ready_pending_view_image", "review_complete"},
                "sealed physical review index")
        for name, binding in index["source_bindings"].items():
            if name == "packet":
                continue
            require(name not in shard_bindings or shard_bindings[name] == binding,
                    "conflicting review shard binding")
            shard_bindings[name] = binding
        for group in index["groups"]:
            key = group["visual_group_id"]
            require(key not in groups, "duplicate visual group across review indexes")
            groups[key] = group
        for review in index["rows"]:
            key = review["job_id"]
            require(key not in reviews, "duplicate execution job across review indexes")
            reviews[key] = review
    require({row["visual_group_id"] for row in decision_rows} == set(groups),
            "root decisions must resolve every reviewed visual group")

    decisions_binding = _binding(decisions_path)
    accepted: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for decision in decision_rows:
        require(decision.get("disposition") in {"admit", "hold", "reject"},
                "root decision disposition")
        if decision["disposition"] != "admit":
            continue
        group_id, job_id = decision.get("visual_group_id"), decision.get("canonical_job_id")
        require(group_id in groups and job_id in reviews, "admitted group/job absent from review index")
        group, review = groups[group_id], reviews[job_id]
        require(review["visual_group_id"] == group_id and job_id in group["job_ids"]
                and decision.get("alias_history") == {
                    "status": "canonical_exact_job_history", "canonical_job_id": job_id,
                }, "canonical exact alias/history ruling")
        require(decision.get("c_trust") == {
                    "status": "physically_trusted", "supported_not_yet_covered": True,
                }, "separate physical c trust ruling")
        require(decision.get("w_trust") == {
                    "status": "physically_trusted", "immediate_successor": True,
                }, "separate physical immediate-w trust ruling")
        accepted.append((decision, review))

    pool_order = {int(image_id): index for index, image_id in enumerate(pool["image_ids"])}
    confirmation_ids = {int(image_id) for image_id in confirmation["image_ids"]}
    pool_excluded = {int(image_id) for image_id in pool["excluded_ids"]}
    for _, review in accepted:
        image_id = int(review["image_id"])
        require(image_id in pool_order and image_id not in confirmation_ids
                and image_id not in pool_excluded, "training/confirmation or pool role overlap")
    accepted.sort(key=lambda pair: (
        pool_order[int(pair[1]["image_id"])], pair[1]["source_identity"]["history_index"],
        pair[1]["source_identity"]["candidate_index"], pair[1]["job_id"],
    ))
    by_image: dict[int, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for pair in accepted:
        by_image.setdefault(int(pair[1]["image_id"]), []).append(pair)
    chosen_images = list(by_image)[:64]
    selected = [by_image[image_id][0] for image_id in chosen_images]
    selected += [pair for image_id in chosen_images for pair in by_image[image_id][1:2]]
    selected = selected[:MAX_NEW_PACKAGES]
    require(len(selected) >= MIN_NEW_PACKAGES and len(chosen_images) >= MIN_NEW_IMAGES,
            "root-admitted physical bank below package/image floor")

    source_rows: dict[str, dict[str, list[tuple[str, dict[str, Any]]]]] = {}
    for shard, source in shard_bindings.items():
        source_rows[shard] = {
            name: _jsonl_rows(source[name], f"{shard} {name}")
            for name in ("jobs", "rows", "images")
        }
    packages = []
    for _, review in selected:
        shard = review["shard"]
        require(shard in source_rows, "review shard source binding")
        jobs, rows, images = (source_rows[shard][name] for name in ("jobs", "rows", "images"))
        job_ordinal, row_ordinal = review["source_job_ordinal"], review["source_row_ordinal"]
        require(0 <= job_ordinal < len(jobs) and 0 <= row_ordinal < len(rows),
                "review source ordinal")
        job_line, job = jobs[job_ordinal]
        row_line, row = rows[row_ordinal]
        require(_line_sha256(job_line) == review["source_job_sha256"]
                and _line_sha256(row_line) == review["source_row_sha256"],
                "review source line identity")
        image_matches = [value for _, value in images if int(value["image_id"]) == int(review["image_id"])]
        require(len(image_matches) == 1, "one exact supply image record")
        packages.append(_physical_record(job=job, row=row, image_source=image_matches[0],
                                         review=review, decisions_binding=decisions_binding))

    image_counts = Counter(int(package["image_id"]) for package in packages)
    require(max(image_counts.values()) <= 2, "at most two packages per image")
    bank = {
        "schema": PHYSICAL_BANK_SCHEMA, "status": "root_admitted",
        "selection_rule": "frozen_pool_order_distinct_image_first_then_one_second_package",
        "pool": pool_binding, "confirmation_selection": confirmation_binding,
        "review_indexes": review_bindings, "root_decisions": decisions_binding,
        "counts": {"packages": len(packages), "images": len(image_counts)},
        "materialization_raw_sources": [_normalized_binding(pool["source"], "pool raw source")],
        "packages": packages,
    }
    # Exercise the final consumer schema before publishing the immutable bank.
    probe_packet = {"new_packages": packages}
    require(MIN_NEW_PACKAGES <= len(probe_packet["new_packages"]) <= MAX_NEW_PACKAGES,
            "physical bank package cap")
    destination.parent.mkdir(parents=True, exist_ok=True)
    old_training.publish(destination, bank)
    return bank


def physical_source_preflight(
    *, pool_path: str | Path, confirmation_selection_path: str | Path,
    review_index_paths: Sequence[str | Path], output: str | Path,
) -> dict[str, Any]:
    """Validate real review/source transport without manufacturing physical labels."""
    require(review_index_paths, "at least one sealed physical review index")
    pool = json.loads(Path(pool_path).read_text())
    confirmation = json.loads(Path(confirmation_selection_path).read_text())
    require(pool.get("schema") == "owner_successor_scale.supply_pool.v1"
            and pool.get("status") == "frozen_before_new_rollouts"
            and len(pool.get("image_ids", [])) == 4096, "frozen 4096 supply pool")
    require(confirmation.get("schema") == "native_owner_successor_scale_throughput.confirmation_selection.v1"
            and confirmation.get("status") == "frozen_cpu_no_model_calls", "frozen confirmation selection")
    _normalized_binding(pool["source"], "pool raw source")
    pool_ids = {int(value) for value in pool["image_ids"]}
    require(not pool_ids.intersection(int(value) for value in confirmation["image_ids"]),
            "supply pool overlaps confirmation panel")
    total_rows, group_ids, job_ids, image_ids = 0, set(), set(), set()
    statuses = Counter()
    for review_path in review_index_paths:
        index = json.loads(Path(review_path).read_text())
        require(index.get("schema") == "owner_successor_scale.physical_admission_review_index.v1"
                and index.get("status") in {"review_ready_pending_view_image", "review_complete"},
                "sealed physical review index")
        for binding in index["source_bindings"].values():
            if "path" in binding:
                _normalized_binding(binding, "review packet binding")
            else:
                for name, value in binding.items():
                    _normalized_binding(value, f"review shard {name} binding")
        groups = {group["visual_group_id"]: group for group in index["groups"]}
        require(len(groups) == len(index["groups"]), "duplicate visual group in review index")
        require(not group_ids.intersection(groups), "duplicate visual group across review indexes")
        group_ids.update(groups)
        shard_rows = {}
        for shard, binding in index["source_bindings"].items():
            if shard == "packet":
                continue
            shard_rows[shard] = {
                name: _jsonl_rows(binding[name], f"{shard} {name}")
                for name in ("jobs", "rows", "images")
            }
        for review in index["rows"]:
            require(review["job_id"] not in job_ids and review["visual_group_id"] in groups,
                    "review job/group identity")
            job_ids.add(review["job_id"])
            image_ids.add(int(review["image_id"]))
            total_rows += 1
            statuses[review["c"]["physical_review"]["status"]] += 1
            statuses[review["w"]["physical_review"]["status"]] += 1
            source = shard_rows[review["shard"]]
            job_line, job = source["jobs"][review["source_job_ordinal"]]
            row_line, row = source["rows"][review["source_row_ordinal"]]
            require(_line_sha256(job_line) == review["source_job_sha256"]
                    and _line_sha256(row_line) == review["source_row_sha256"],
                    "review source line identity")
            require(job["job_id"] == row["job_id"] == review["job_id"]
                    and int(job["image_id"]) == int(row["image_id"]) == int(review["image_id"]),
                    "review exact job/image join")
            require(_digest_ids(job["c_token_ids"]) == review["c"]["token_ids_sha256"],
                    "review/source c token identity")
            local_w = row.get("local_w", {})
            require(local_w.get("status") == "candidate_local_w"
                    and _digest_ids(local_w["w_token_ids"]) == review["w"]["token_ids_sha256"],
                    "review/source immediate-w token identity")
            matches = [value for _, value in source["images"]
                       if int(value["image_id"]) == int(review["image_id"])]
            require(len(matches) == 1 and matches[0]["frozen"]["example_id"] == review["example_id"],
                    "review/source frozen image identity")
            require(int(review["image_id"]) in pool_ids, "review image outside frozen pool")
    receipt = {
        "schema": "owner_successor_scale.physical_source_preflight.v1",
        "status": "CPU_valid_pending_root_physical_decisions",
        "pool": _binding(pool_path), "confirmation_selection": _binding(confirmation_selection_path),
        "review_indexes": [_binding(path) for path in review_index_paths],
        "counts": {"candidate_rows": total_rows, "visual_groups": len(group_ids),
                   "images": len(image_ids), "physical_axis_statuses": dict(statuses)},
        "admission_created": False,
    }
    destination = Path(output).resolve()
    require(not destination.exists(), "no physical preflight overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    old_training.publish(destination, receipt)
    return receipt


def validate_packet(packet: Mapping[str, Any]) -> None:
    """Pure fail-closed validation; file bindings are checked by ``load_packet``."""
    require(packet.get("schema") == SCHEMA, "training packet schema")
    require(packet.get("status") in {"calibration_pending", "sealed_root_grant_pending"},
            "training packet status")
    require(packet.get("updates") == UPDATES and packet.get("block_size") == BLOCK_SIZE,
            "fixed update/block contract")
    require(packet.get("coefficients") == COEFFICIENTS, "frozen common component coefficients")
    require(packet.get("optimizer") == OPTIMIZER and packet.get("clip_gradient_norm") == CLIP_GRADIENT_NORM,
            "frozen AdamW/clip contract")
    require(packet.get("generation") == GENERATION, "full greedy 3084 refresh contract")
    require(packet.get("reference_teacher") == "frozen_N16_source_snapshot", "N16 teacher identity")
    for field in ("n16_receipt", "n16_training_input", "physical_package_bank"):
        _binding_shape(packet.get(field), f"bound {field}")
    _binding_shape(packet.get("replay_acceptance"), "bound replay_acceptance")
    throughput = packet.get("throughput", {})
    require(throughput.get("api") == "batched_aligned_logits(model,entries)"
            and throughput.get("microbatch_size") in (1, 2, 4, 8, 16)
            and isinstance(throughput.get("activation_checkpointing"), bool),
            "accepted replay execution choice")
    runtime = packet.get("runtime", {})
    require(set(runtime) == {"world_sizes", *RUNTIME_LIMIT_KEYS}
            and runtime["world_sizes"] == [2, 8], "staged runtime envelope fields")
    for field in RUNTIME_LIMIT_KEYS:
        require(type(runtime[field]) in (int, float) and math.isfinite(runtime[field])
                and runtime[field] > 0, f"positive runtime envelope {field}")
    require(packet.get("stage_permissions") == {
        "authorized": STAGED_REPLAY_STAGES,
        "full_256": "requires_root_integrated_smoke_acceptance",
    }, "staged calibration/smoke permissions")
    raw_sources = packet.get("materialization_raw_sources")
    require(isinstance(raw_sources, list) and raw_sources, "materialization raw-source bindings")
    for source in raw_sources:
        _binding_shape(source, "materialization raw-source binding")
    require(len({source["path"] for source in raw_sources}) == len(raw_sources),
            "duplicate materialization raw-source path")
    composition = packet.get("raw_source_composition", {})
    require(composition.get("schema") == RAW_SOURCE_COMPOSITION_SCHEMA
            and composition.get("status") == "passed_exact_projection"
            and composition.get("policy") == "retain_configured_base_owner_add_required_only"
            and composition.get("base_sources") == raw_sources
            and composition.get("allowed_difference_paths") == list(RAW_SOURCE_ALLOWED_DIFFERENCES)
            and composition.get("base_rows_replaced") == 0,
            "probe-local raw-source composition receipt")
    augmentation_sources = composition.get("augmentation_sources")
    require(isinstance(augmentation_sources, list) and augmentation_sources,
            "raw-source augmentation bindings")
    for source in augmentation_sources:
        _binding_shape(source, "raw-source augmentation binding")
    require(not ({source["path"] for source in raw_sources}
                 & {source["path"] for source in augmentation_sources}),
            "base/augmentation raw-source separation")
    counts = composition.get("counts", {})
    require(all(isinstance(counts.get(key), int) and counts[key] >= 0 for key in (
        "base_records", "augmentation_selected_records", "allowed_equivalent_collisions",
        "required_additions", "required_records", "final_lookup_records",
    )) and counts["required_records"] > 0,
            "raw-source composition counts")
    for key in (
        "required_example_ids_sha256", "required_materialization_projections_sha256",
        "collision_records_sha256", "addition_records_sha256",
    ):
        require(isinstance(composition.get(key), str) and len(composition[key]) == 64,
                f"raw-source composition {key}")
    require(packet.get("fork", {}).get("gamma") == FORK_GAMMA
            and packet["fork"].get("target_gradient_ratio") == FORK_TARGET_RATIO
            and packet["fork"].get("selection_gradient") == "detached"
            and packet["fork"].get("denominator") == "fixed_new_package_count"
            and packet["fork"].get("comparison") == "actual_repeat_token_vs_fixed_credible_token",
            "frozen repeat-fork contract")
    packages = packet.get("new_packages")
    require(isinstance(packages, list) and MIN_NEW_PACKAGES <= len(packages) <= MAX_NEW_PACKAGES,
            "new package admission floor/cap")
    package_ids, image_ids, record_ids = [], [], []
    for package in packages:
        require(isinstance(package.get("package_id"), str) and package["package_id"], "new package ID")
        package_ids.append(package["package_id"])
        image_ids.append(str(package.get("image_id")))
        h_plus = _ids(package.get("h_plus_token_ids"), "frozen h_plus token IDs", nonempty=False)
        credible = _ids(package.get("credible_c_token_ids"), "fixed credible complete c row")
        c_trust, w_trust = package.get("c_trust"), package.get("w_trust")
        require(isinstance(c_trust, Mapping) and c_trust.get("status") == "physically_trusted"
                and c_trust.get("supported_not_yet_covered") is True
                and isinstance(c_trust.get("evidence"), Mapping), "physical c trust evidence")
        require(isinstance(w_trust, Mapping) and w_trust.get("status") == "physically_trusted"
                and w_trust.get("immediate_successor") is True
                and isinstance(w_trust.get("evidence"), Mapping), "physical immediate-w trust evidence")
        _binding_shape(c_trust["evidence"], "bound physical c trust evidence")
        _binding_shape(w_trust["evidence"], "bound physical immediate-w trust evidence")
        require(len(h_plus) + len(credible) < MAX_NEW_TOKENS, "c does not fit total 3084 allowance")
        c_record, w_record = package.get("c_record"), package.get("witness_record")
        require(isinstance(c_record, Mapping) and isinstance(w_record, Mapping), "new c/w records")
        old_training.validate_record(c_record, kind="positive", verify_image=False)
        old_training.validate_record(w_record, kind="conditional", verify_image=False)
        require(c_record["prefix_token_ids"] == h_plus
                and c_record["target_token_ids"] == credible,
                "c record differs from frozen h_plus/credible row")
        require(w_record["prefix_token_ids"] == [*h_plus, *credible]
                and w_record["kl_positions"] == list(range(len(w_record["target_token_ids"]))),
                "w must be the full immediate successor after h_plus+c")
        witness_ids = w_record["target_token_ids"]
        require(witness_ids[0] == 151646 and witness_ids[-1] == old_training.old.BOX_END
                and witness_ids.count(151646) == witness_ids.count(old_training.old.BOX_END) == 1
                and old_training.old.EOS not in witness_ids and old_training.old.PAD not in witness_ids,
                "witness target must be one complete non-EOS row")
        require(c_record["example_id"] == w_record["example_id"]
                and str(package["image_id"]) == str(c_record["image"]["image_id"])
                and c_record["image"] == w_record["image"], "new c/w image identity")
        require(isinstance(package.get("case"), Mapping) and isinstance(package.get("golden"), Mapping)
                and isinstance(package.get("h_text"), str), "refresh parser evidence")
        record_ids.extend((c_record["record_id"], w_record["record_id"]))
    require(len(set(package_ids)) == len(package_ids), "duplicate new package ID")
    require(len(set(record_ids)) == len(record_ids), "duplicate new c/w record ID")
    require(len(set(image_ids)) >= MIN_NEW_IMAGES, "new package distinct-image floor")
    old_ids = packet.get("old_package_ids")
    require(isinstance(old_ids, list), "old package IDs")
    validate_schedule(packet.get("schedule", []), old_ids, package_ids)
    calibration = packet.get("calibration")
    if packet["status"] == "calibration_pending":
        require(calibration is None, "pending packet cannot preselect fork lambda")
    else:
        require(isinstance(calibration, Mapping)
                and calibration.get("schema") == CALIBRATION_SCHEMA
                and calibration.get("status") == "passed"
                and calibration.get("active_events", 0) > 0
                and type(calibration.get("fork_lambda")) in (int, float)
                and math.isfinite(calibration["fork_lambda"])
                and calibration["fork_lambda"] > 0,
                "sealed packet requires one passed no-update calibration")


def load_packet(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify source/bank bytes and the exact completed N16 anchor before any model load."""
    packet = json.loads(Path(path).read_text())
    validate_packet(packet)
    for field in ("n16_receipt", "n16_training_input", "physical_package_bank", "replay_acceptance"):
        require(_binding(packet[field]["path"]) == packet[field], f"bound {field} changed")
    require(Path(packet["n16_receipt"]["path"]).resolve() == N16_ROOT / "receipt.json",
            "unexpected N16 receipt path")
    receipt = json.loads(Path(packet["n16_receipt"]["path"]).read_text())
    require(receipt.get("updates") == UPDATES and receipt.get("world_size") == 8,
            "N16 source receipt completion")
    require(receipt.get("input") == packet["n16_training_input"], "N16 input binding mismatch")
    adapter = receipt.get("saved_adapter", {})
    require(Path(adapter.get("root", "")).resolve() == N16_ROOT / "adapter"
            and adapter.get("fingerprint") == N16_ADAPTER_FINGERPRINT,
            "N16 adapter identity")
    weights = N16_ROOT / "adapter" / "adapter_model.safetensors"
    require(old_training.file_hash(weights) == N16_WEIGHTS_SHA256, "N16 adapter weights changed")
    old_packet, _, normals, _ = old_training.load_packet(packet["n16_training_input"]["path"])
    require(len(old_packet["positive_records"]) == len(old_packet["conditional_records"]) == OLD_PACKAGE_COUNT
            and len(normals) == REFERENCE_COUNT,
            "N16 old-positive/witness/reference bank identity")
    bank = json.loads(Path(packet["physical_package_bank"]["path"]).read_text())
    require(bank.get("schema") == PHYSICAL_BANK_SCHEMA and bank.get("status") == "root_admitted",
            "physical package bank lacks root admission")
    require(bank.get("packages") == packet["new_packages"], "packet packages differ from admitted bank")
    for package in packet["new_packages"]:
        old_training.validate_record(package["c_record"], kind="positive", verify_image=True)
        old_training.validate_record(package["witness_record"], kind="conditional", verify_image=True)
        for trust in (package["c_trust"], package["w_trust"]):
            require(_binding(trust["evidence"]["path"]) == trust["evidence"],
                    "physical trust evidence changed")
    for source in packet["materialization_raw_sources"]:
        require(_binding(source["path"]) == source, "materialization raw source changed")
    from src.config.inference import load_research_infer_config

    config_input = load_research_infer_config(old_training.CONFIG).config.data.input_jsonl
    _, composition = compose_materialization_raw_lookup(
        base_sources=packet["materialization_raw_sources"],
        augmentation_sources=packet["raw_source_composition"]["augmentation_sources"],
        required_example_ids=_required_raw_example_ids(old_packet, packet["new_packages"]),
        config_input=config_input,
    )
    require(composition == packet["raw_source_composition"],
            "raw-source composition receipt changed")
    replay_acceptance = json.loads(Path(packet["replay_acceptance"]["path"]).read_text())
    validate_replay_acceptance(replay_acceptance, verify_files=True)
    require(replay_acceptance.get("api") == packet["throughput"]["api"]
            and replay_acceptance.get("microbatch_size") == packet["throughput"]["microbatch_size"]
            and replay_acceptance.get("activation_checkpointing")
            == packet["throughput"]["activation_checkpointing"]
            and replay_acceptance.get("runtime") == packet["runtime"],
            "batched replay is not staged-accepted for calibration/integrated smoke")
    return packet, receipt, bank


def prepare_packet(
    *,
    physical_package_bank: str | Path,
    replay_acceptance: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Bind admitted physical packages to the exact completed N16 source."""
    receipt_path = N16_ROOT / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    bank = json.loads(Path(physical_package_bank).read_text())
    acceptance = json.loads(Path(replay_acceptance).read_text())
    require(bank.get("schema") == PHYSICAL_BANK_SCHEMA and bank.get("status") == "root_admitted"
            and isinstance(bank.get("packages"), list),
            "root-admitted physical package bank required")
    validate_replay_acceptance(acceptance, verify_files=True)
    old_input = receipt["input"]
    old_packet = json.loads(Path(old_input["path"]).read_text())
    old_ids = [record["record_id"] for record in old_packet["positive_records"]]
    package_ids = [package["package_id"] for package in bank["packages"]]
    base_sources = [
        _normalized_binding(source, "N16 base raw source")
        for source in old_packet["materialization_raw_sources"]
    ]
    augmentation_sources = [
        _normalized_binding(source, "new-bank augmentation raw source")
        for source in bank.get("materialization_raw_sources", [])
        if source["path"] not in {base["path"] for base in base_sources}
    ]
    from src.config.inference import load_research_infer_config

    config_input = load_research_infer_config(old_training.CONFIG).config.data.input_jsonl
    _, composition = compose_materialization_raw_lookup(
        base_sources=base_sources, augmentation_sources=augmentation_sources,
        required_example_ids=_required_raw_example_ids(old_packet, bank["packages"]),
        config_input=config_input,
    )
    packet = {
        "schema": SCHEMA,
        "status": "calibration_pending",
        "updates": UPDATES,
        "block_size": BLOCK_SIZE,
        "coefficients": dict(COEFFICIENTS),
        "optimizer": {**OPTIMIZER, "betas": list(OPTIMIZER["betas"])},
        "clip_gradient_norm": CLIP_GRADIENT_NORM,
        "generation": dict(GENERATION),
        "reference_teacher": "frozen_N16_source_snapshot",
        "n16_receipt": _binding(receipt_path),
        "n16_training_input": old_input,
        "physical_package_bank": _binding(physical_package_bank),
        "replay_acceptance": _binding(replay_acceptance),
        "throughput": {
            "api": "batched_aligned_logits(model,entries)",
            "microbatch_size": acceptance["microbatch_size"],
            "activation_checkpointing": acceptance["activation_checkpointing"],
        },
        "runtime": acceptance["runtime"],
        "stage_permissions": {
            "authorized": list(acceptance["authorized_stages"]),
            "full_256": "requires_root_integrated_smoke_acceptance",
        },
        "fork": {
            "gamma": FORK_GAMMA,
            "target_gradient_ratio": FORK_TARGET_RATIO,
            "selection_gradient": "detached",
            "denominator": "fixed_new_package_count",
            "comparison": "actual_repeat_token_vs_fixed_credible_token",
        },
        "old_package_ids": old_ids,
        "new_packages": bank["packages"],
        "schedule": balanced_schedule(old_ids, package_ids),
        "materialization_raw_sources": base_sources,
        "raw_source_composition": composition,
        "calibration": None,
    }
    validate_packet(packet)
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    old_training.publish(destination, packet)
    load_packet(destination)
    return packet


def source_preflight(*, replay_diagnostic: str | Path, output: str | Path) -> dict[str, Any]:
    """Real CPU identity check without manufacturing a successor admission floor."""
    receipt_path = N16_ROOT / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get("updates") == UPDATES and receipt.get("world_size") == 8,
            "completed N16 source receipt")
    require(receipt.get("saved_adapter", {}).get("fingerprint") == N16_ADAPTER_FINGERPRINT,
            "N16 source adapter fingerprint")
    require(old_training.file_hash(N16_ROOT / "adapter" / "adapter_model.safetensors")
            == N16_WEIGHTS_SHA256, "N16 source weights")
    old_packet, _, normals, _ = old_training.load_packet(receipt["input"]["path"])
    require(len(old_packet["positive_records"]) == len(old_packet["conditional_records"])
            == OLD_PACKAGE_COUNT and len(normals) == REFERENCE_COUNT, "real N16 bank counts")
    diagnostic_path = Path(replay_diagnostic).resolve()
    diagnostic = json.loads(diagnostic_path.read_text())
    require(diagnostic.get("status") == "passed", "final replay diagnostic status")
    result = {
        "schema": "owner_successor_scale.training.source_preflight.v1",
        "status": "passed_source_identity_HOLD_capacity_admission_and_lead_acceptance",
        "n16_receipt": _binding(receipt_path),
        "n16_training_input": receipt["input"],
        "n16_adapter": receipt["saved_adapter"],
        "counts": {"old_positive": OLD_PACKAGE_COUNT, "old_witness": OLD_PACKAGE_COUNT,
                   "reference": REFERENCE_COUNT, "new_trusted_packages": None},
        "named_real_records_for_future_technical_slice": {
            "positive": [record["record_id"] for record in old_packet["positive_records"][:2]],
            "witness": [record["record_id"] for record in old_packet["conditional_records"][:2]],
        },
        "replay_diagnostic": _binding(diagnostic_path),
        "transport_acceptance": "not_inferred_from_diagnostic; capacity selection and lead acceptance pending",
        "parameter_updates": 0,
        "model_loads": 0,
    }
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    old_training.publish(destination, result)
    return result


class BatchedObjective(torch.nn.Module):
    """One bounded replay microbatch; callers backward and free it immediately."""

    COMPONENT_ORDER = (*COEFFICIENTS, "fork")

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, items: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        require(bool(items), "empty training replay microbatch")
        logits_rows = batched_aligned_logits(self.model, [item["entry"] for item in items])
        total = logits_rows[0].new_zeros(())
        components = {key: total.clone() for key in self.COMPONENT_ORDER}
        for logits, item in zip(logits_rows, items, strict=True):
            component = item["component"]
            record = item["entry"]["record"]
            targets = torch.tensor(record["target_token_ids"], dtype=torch.long, device=logits.device)
            if component in ("old_ce", "new_ce"):
                raw = _positive_nll(logits, record["target_token_ids"])
                value = raw * item["scale"]
                components[component] = components[component] + value
            elif component in ("old_witness_kl", "new_witness_kl"):
                raw = old_training.old.reference_kl(
                    logits, item["reference_logp"].detach().to(logits.device), record["kl_positions"]
                )
                value = raw * item["scale"]
                components[component] = components[component] + value
            elif component == "normal_kl":
                raw = old_training.old.reference_kl(
                    logits, item["reference_logp"].detach().to(logits.device), record["kl_positions"]
                )
                penalty, _ = margin_engine.worst_margin_penalty(logits, targets, item["margin"])
                value = raw * item["scale"] + penalty * item["margin_scale"]
                components["normal_kl"] = components["normal_kl"] + raw * item["scale"]
                components["normal_margin"] = components["normal_margin"] + penalty * item["margin_scale"]
            elif component == "fork":
                value = fork_hinge(
                    logits[0], repeat_token_id=item["repeat_token_id"],
                    credible_token_id=item["credible_token_id"],
                ) * item["scale"]
                components["fork"] = components["fork"] + value
            else:
                raise ValueError(f"unknown training component: {component}")
            total = total + value
        return total, torch.stack([components[key] for key in self.COMPONENT_ORDER])


def _length_batches(items: Sequence[Mapping[str, Any]], microbatch_size: int) -> list[list[Mapping[str, Any]]]:
    require(microbatch_size in (1, 2, 4, 8, 16), "replay microbatch size")
    ordered = sorted(enumerate(items), key=lambda pair: (
        len(pair[1]["entry"]["prompt_ids"]) + len(pair[1]["entry"]["record"]["target_token_ids"]), pair[0]
    ))
    values = [item for _, item in ordered]
    return [values[start:start + microbatch_size] for start in range(0, len(values), microbatch_size)]


def backward_microbatches(
    scorer: torch.nn.Module,
    items: Sequence[Mapping[str, Any]],
    *,
    microbatch_size: int,
    synchronize: bool,
) -> dict[str, float]:
    """Backward each bounded graph immediately; never retain a whole-bank graph."""
    batches = _length_batches(items, microbatch_size)
    require(bool(batches), "training update has no replay work")
    totals = torch.zeros(len(BatchedObjective.COMPONENT_ORDER), dtype=torch.float64)
    for index, batch in enumerate(batches):
        last = index == len(batches) - 1
        no_sync = getattr(scorer, "no_sync", None)
        context = nullcontext() if (synchronize and last) or no_sync is None else no_sync()
        with context:
            loss, components = scorer(batch)
            require(loss.ndim == 0 and bool(torch.isfinite(loss)), "finite scalar microbatch loss")
            loss.backward()
        totals += components.detach().double().cpu()
        del loss, components
    return {key: float(value) for key, value in zip(BatchedObjective.COMPONENT_ORDER, totals, strict=True)}


def evaluate_microbatches(
    scorer: torch.nn.Module, items: Sequence[Mapping[str, Any]], *, microbatch_size: int
) -> dict[str, float]:
    """Read back protection banks with no graphs and no parameter change."""
    totals = torch.zeros(len(BatchedObjective.COMPONENT_ORDER), dtype=torch.float64)
    with torch.no_grad():
        for batch in _length_batches(items, microbatch_size):
            _, components = scorer(batch)
            totals += components.double().cpu()
    return {key: float(value) for key, value in
            zip(BatchedObjective.COMPONENT_ORDER, totals, strict=True)}


def _select_trainable(model: torch.nn.Module) -> tuple[list[tuple[str, torch.nn.Parameter]], list[tuple[str, torch.nn.Parameter]]]:
    from src.adapters.dora import select_dora_parameters

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    named = list(select_dora_parameters(model, towers=("language",), adapter_name="default"))
    require(len(named) == old_training.old.EXPECTED_TRAINABLE_TENSORS
            and sum(parameter.numel() for _, parameter in named) == old_training.old.EXPECTED_TRAINABLE_SCALARS
            and all("language_model" in name and not any(part in name for part in (
                "visual", "vision", "merger", "embed_tokens", "lm_head"
            )) for name, _ in named), "exact original language-only DoRA trainable surface")
    for _, parameter in named:
        parameter.requires_grad_(True)
    selected = {id(parameter) for _, parameter in named}
    frozen = [(name, parameter) for name, parameter in model.named_parameters() if id(parameter) not in selected]
    return named, frozen


def _composed_runtime_raw(
    packet: Mapping[str, Any], old_packet: Mapping[str, Any], *, config_input: str | Path
) -> dict[str, Any]:
    raw, composition = compose_materialization_raw_lookup(
        base_sources=packet["materialization_raw_sources"],
        augmentation_sources=packet["raw_source_composition"]["augmentation_sources"],
        required_example_ids=_required_raw_example_ids(old_packet, packet["new_packages"]),
        config_input=config_input,
    )
    require(composition == packet["raw_source_composition"],
            "runtime raw-source composition differs from sealed CPU receipt")
    return raw


def _runtime_banks(packet: Mapping[str, Any], *, device: torch.device, evidence_dir: Path) -> dict[str, Any]:
    old_packet, anchor, normals, margins = old_training.load_packet(packet["n16_training_input"]["path"])
    loader_packet = {"materialization_raw_sources": packet["materialization_raw_sources"]}
    receipt = json.loads(Path(packet["n16_receipt"]["path"]).read_text())
    qwen, frontend, config, raw = old_training._load_model(
        anchor, adapter_path=receipt["saved_adapter"]["root"], device=device,
        evidence_dir=evidence_dir, packet=loader_packet,
    )
    raw = _composed_runtime_raw(packet, old_packet, config_input=config.data.input_jsonl)
    old_positive = [old_training._materialize(record, qwen=qwen, frontend=frontend, config=config, raw=raw)
                    for record in old_packet["positive_records"]]
    old_witness = [old_training._materialize(record, qwen=qwen, frontend=frontend, config=config, raw=raw)
                   for record in old_packet["conditional_records"]]
    new_positive = [old_training._materialize(package["c_record"], qwen=qwen, frontend=frontend,
                                              config=config, raw=raw)
                    for package in packet["new_packages"]]
    new_witness = [old_training._materialize(package["witness_record"], qwen=qwen, frontend=frontend,
                                             config=config, raw=raw)
                   for package in packet["new_packages"]]
    normal = [old_training._materialize(old_training._normal_record(record), qwen=qwen,
                                        frontend=frontend, config=config, raw=raw)
              for record in normals]
    return {
        "qwen": qwen,
        "old_positive": old_positive,
        "old_witness": old_witness,
        "new_positive": new_positive,
        "new_witness": new_witness,
        "normal": normal,
        "margins": margins,
    }


def _reference_cache(model: torch.nn.Module, entries: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    references: dict[str, torch.Tensor] = {}
    for entry in entries:
        record = entry["record"]
        references[record["record_id"]] = old_training.old._reference_logp(
            model, entry["inputs"], entry["prompt_ids"], record["target_token_ids"], record["kl_positions"]
        )
    return references


def _append_runtime_journal(path: Path, value: Mapping[str, Any]) -> None:
    """Durably append one bounded rank-local runtime evidence record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _first_literal_next_row(token_ids: Sequence[int]) -> tuple[list[int] | None, str]:
    """Select only a structurally complete row beginning at the first token.

    This deliberately does not resynchronize after malformed material and does
    not require later generated material to be a complete stored history.
    """
    from probes.parallel_owner_research.history import EOS, ROW_END, ROW_START, complete_rows

    values = _ids(list(token_ids), "literal generated token IDs", nonempty=False)
    if not values or values[0] == EOS:
        return None, "inactive_empty_or_eos_only"
    if values[0] != ROW_START:
        return None, "inactive_malformed_first_token"
    for index, token in enumerate(values[1:], start=1):
        if token == ROW_START:
            return None, "inactive_nested_first_row"
        if token == EOS:
            return None, "inactive_incomplete_first_row"
        if token == ROW_END:
            row = values[:index + 1]
            if len(row) < 8:
                return None, "inactive_incomplete_first_row"
            require(complete_rows(row) == [row], "selected first-row grammar identity")
            return row, "complete_first_row"
    return None, "inactive_incomplete_first_row"


def _mine_one(
    model: torch.nn.Module,
    qwen: Any,
    package: Mapping[str, Any],
    c_entry: Mapping[str, Any],
    *,
    journal_path: Path,
    phase: str,
    rank: int,
    before_update: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Full-greedy refresh at the same h+; row selection is entirely no-grad."""
    from probes.source_rweak_row_cross.run import native_record
    from src.data.geometry import iou_xyxy
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import NativeBatch

    h_plus = package["h_plus_token_ids"]
    require(qwen.tokenizer.decode(h_plus, skip_special_tokens=False) == package["h_text"],
            "frozen h_plus token/text identity")
    budget = MAX_NEW_TOKENS - len(h_plus)
    require(budget > 0, "refresh budget after frozen h_plus")
    batch = NativeBatch(inputs=c_entry["inputs"], request_ids=(package["package_id"],))
    policy = NativeGenerationPolicy(temperature=0.0, top_p=1.0, top_k=0,
                                    repetition_penalty=1.0, use_model_defaults=False)
    with torch.inference_mode():
        generated = generate_continuations(
            model, batch, extensions=(h_plus,), budgets=(budget,),
            eos_token_id=old_training.old.EOS, pad_token_id=qwen.tokenizer.pad_token_id,
            policy=policy, trace="none",
        )[0]
    token_ids = _ids(list(generated.token_ids), "literal generated token IDs", nonempty=False)
    journal_id = f"{phase}:rank{rank}:before-update-{before_update}:{package['package_id']}"
    _append_runtime_journal(journal_path, {
        "schema": "owner_successor_scale.refresh_journal.v1",
        "kind": "generated_refresh",
        "journal_id": journal_id,
        "phase": phase,
        "rank": rank,
        "before_update": before_update,
        "package_id": package["package_id"],
        "c_record_id": package["c_record"]["record_id"],
        "c_example_id": package["c_record"]["example_id"],
        "c_prompt_token_ids_sha256": package["c_record"]["prompt_token_ids_sha256"],
        "executed_media_sha256": package["c_record"]["image"]["executed_media_sha256"],
        "materialized_prompt_token_ids_sha256": _digest_ids(c_entry["prompt_ids"]),
        "h_plus_token_ids": h_plus,
        "h_plus_token_ids_sha256": _digest_ids(h_plus),
        "h_text_sha256": hashlib.sha256(package["h_text"].encode()).hexdigest(),
        "continuation_budget": budget,
        "total_allowance": MAX_NEW_TOKENS,
        "stop_reason": generated.stop_reason,
        "generated_token_ids": token_ids,
        "generated_token_ids_sha256": _digest_ids(token_ids),
        "generated_tokens": len(token_ids),
    })
    row_ids, first_row_outcome = _first_literal_next_row(token_ids)
    receipt: dict[str, Any] = {
        "package_id": package["package_id"], "stop_reason": generated.stop_reason,
        "generated_tokens": len(token_ids), "total_allowance": MAX_NEW_TOKENS,
        "first_complete_row": row_ids, "strict_repeat": False,
        "first_row_outcome": first_row_outcome,
    }

    def finish(event: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        _append_runtime_journal(journal_path, {
            "schema": "owner_successor_scale.refresh_journal.v1",
            "kind": "parsed_outcome",
            "journal_id": journal_id,
            "phase": phase,
            "rank": rank,
            "before_update": before_update,
            "package_id": package["package_id"],
            "event_active": event is not None,
            "receipt": receipt,
        })
        return event, receipt

    if row_ids is None:
        return finish()
    text = qwen.tokenizer.decode(row_ids, skip_special_tokens=False)
    parsed = native_record(text, package["case"], package["golden"], generated.stop_reason)
    if len(parsed["pred"]) != 1 or parsed["dropped_predictions"]:
        receipt["first_row_parse"] = "invalid_or_dropped"
        receipt["first_row_outcome"] = "inactive_invalid_or_dropped"
        receipt["first_row_valid_predictions"] = len(parsed["pred"])
        receipt["first_row_dropped_predictions"] = len(parsed["dropped_predictions"])
        return finish()
    history = native_record(package["h_text"], package["case"], package["golden"], "supplied_prefix")
    current_box = parsed["pred"][0]["bbox"]
    overlaps = [iou_xyxy(current_box, previous["bbox"]) for previous in history["pred"]]
    repeated_index = next((index for index, overlap in enumerate(overlaps) if overlap > 0.95), None)
    if repeated_index is None:
        receipt["max_history_iou"] = max(overlaps, default=0.0)
        receipt["first_row_outcome"] = "inactive_nonrepeat"
        return finish()
    divergence = first_divergence(row_ids, package["credible_c_token_ids"])
    synthetic = {
        **c_entry,
        "prompt_ids": [*c_entry["prompt_ids"], *divergence["prefix_token_ids"]],
        "record": {"record_id": package["package_id"] + ":fork",
                   "target_token_ids": [divergence["credible_token_id"]]},
    }
    receipt.update(strict_repeat=True, repeated_history_row_index=repeated_index,
                   repeated_history_iou=overlaps[repeated_index], divergence=divergence,
                   first_row_outcome="active_strict_repeat")
    return finish({"entry": synthetic, **divergence})


def _journal_completed_update(
    path: Path, *, rank: int, arm: str, update: int, components: Mapping[str, float],
    raw_gradient_norm: float, replays: int, active_local_forks: int,
    counters: Mapping[str, int],
) -> None:
    _append_runtime_journal(path, {
        "schema": "owner_successor_scale.update_journal.v1",
        "kind": "completed_optimizer_update",
        "rank": rank,
        "arm": arm,
        "update": update,
        "components": dict(components),
        "raw_gradient_norm": raw_gradient_norm,
        "replays": replays,
        "active_local_forks": active_local_forks,
        "counters": dict(counters),
    })


def _gradient_norm(named: Sequence[tuple[str, torch.nn.Parameter]]) -> float:
    require(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                for _, parameter in named), "missing/nonfinite selected gradient")
    return math.sqrt(sum(float(parameter.grad.detach().double().square().sum())
                         for _, parameter in named))


def _step_items(
    packet: Mapping[str, Any], banks: Mapping[str, Any], references: Mapping[str, torch.Tensor],
    step: Mapping[str, Any], *, rank: int, world: int,
    fork_events: Mapping[str, Mapping[str, Any]], fork_lambda: float,
) -> list[dict[str, Any]]:
    count = len(packet["new_packages"])
    old_positive = {entry["record"]["record_id"]: entry for entry in banks["old_positive"]}
    new_positive = {package["package_id"]: entry for package, entry in
                    zip(packet["new_packages"], banks["new_positive"], strict=True)}
    new_witness = {package["package_id"]: entry for package, entry in
                   zip(packet["new_packages"], banks["new_witness"], strict=True)}
    items: list[dict[str, Any]] = []

    def sharded(values: Sequence[Any]) -> Sequence[Any]:
        return values[rank::world]

    for record_id in sharded(step["old_positive_ids"]):
        items.append({"component": "old_ce", "entry": old_positive[record_id],
                      "scale": world * COEFFICIENTS["old_ce"] / 2})
    for package_id in sharded(step["new_positive_ids"]):
        items.append({"component": "new_ce", "entry": new_positive[package_id],
                      "scale": world * COEFFICIENTS["new_ce"] * BLOCK_SIZE / count})
    for entry in sharded(banks["old_witness"]):
        record_id = entry["record"]["record_id"]
        items.append({"component": "old_witness_kl", "entry": entry,
                      "reference_logp": references[record_id],
                      "scale": world * COEFFICIENTS["old_witness_kl"] / OLD_PACKAGE_COUNT})
    for package in sharded(packet["new_packages"]):
        entry = new_witness[package["package_id"]]
        record_id = entry["record"]["record_id"]
        items.append({"component": "new_witness_kl", "entry": entry,
                      "reference_logp": references[record_id],
                      "scale": world * COEFFICIENTS["new_witness_kl"] / count})
    for entry in sharded(banks["normal"]):
        record_id = entry["record"]["record_id"]
        items.append({"component": "normal_kl", "entry": entry,
                      "reference_logp": references[record_id], "margin": banks["margins"][record_id],
                      "scale": world * COEFFICIENTS["normal_kl"] / REFERENCE_COUNT,
                      "margin_scale": world * COEFFICIENTS["normal_margin"] / REFERENCE_COUNT})
    if fork_lambda:
        for package in sharded(packet["new_packages"]):
            event = fork_events.get(package["package_id"])
            if event is not None:
                items.append({"component": "fork", "entry": event["entry"],
                              "repeat_token_id": event["repeat_token_id"],
                              "credible_token_id": event["credible_token_id"],
                              "scale": world * fork_lambda / count})
    return items


def planned_work_upper_bound(
    packet: Mapping[str, Any], *, rank: int, world: int, updates: int,
) -> dict[str, int]:
    """Conservative top-model/image-forward ceiling before ranks are started."""
    require(world in (2, 8) and 0 <= rank < world and updates in (1, UPDATES), "planned work scope")
    microbatch = packet["throughput"]["microbatch_size"]
    new_count = len(packet["new_packages"])
    local_new = len(packet["new_packages"][rank::world])
    teacher = (len(range(rank, OLD_PACKAGE_COUNT, world)) + local_new
               + len(range(rank, REFERENCE_COUNT, world)))
    training = 0
    for step in packet["schedule"][:updates]:
        # Fork assumes every local package remains active: an upper bound.
        items = (len(step["old_positive_ids"][rank::world])
                 + len(step["new_positive_ids"][rank::world])
                 + len(range(rank, OLD_PACKAGE_COUNT, world)) + local_new
                 + len(range(rank, REFERENCE_COUNT, world)) + local_new)
        training += math.ceil(items / microbatch)
    refresh_rounds = sum(step["refresh_before_update"] for step in packet["schedule"][:updates])
    refresh = refresh_rounds * sum(
        MAX_NEW_TOKENS - len(packet["new_packages"][index]["h_plus_token_ids"])
        for index in range(rank, new_count, world)
    )
    final_items = (len(range(rank, OLD_PACKAGE_COUNT, world)) + local_new
                   + len(range(rank, REFERENCE_COUNT, world)))
    final_protection = math.ceil(final_items / microbatch)
    final_positive_scores = OLD_PACKAGE_COUNT + new_count if rank == 0 else 0
    model = teacher + training + refresh + final_protection + final_positive_scores
    return {
        "teacher_forwards": teacher,
        "training_batched_forwards": training,
        "refresh_model_forwards_upper_bound": refresh,
        "final_protection_batched_forwards": final_protection,
        "final_positive_score_forwards": final_positive_scores,
        "model_forwards_upper_bound": model,
        # Native replay/generation may call vision no more often than the top model.
        "image_forwards_upper_bound": model,
        "optimizer_steps": updates,
        "refresh_rounds": refresh_rounds,
    }


def _local_reference_cache(
    banks: Mapping[str, Any], *, rank: int, world: int
) -> dict[str, torch.Tensor]:
    values = [*banks["old_witness"], *banks["new_witness"], *banks["normal"]]
    # The static banks are independently sharded in _step_items using their own
    # index frame, so cache the matching union rather than slicing the concat.
    selected = [*banks["old_witness"][rank::world], *banks["new_witness"][rank::world],
                *banks["normal"][rank::world]]
    require(set(id(value) for value in selected).issubset(set(id(value) for value in values)),
            "reference shard identity")
    return _reference_cache(banks["qwen"].model, selected)


def _refresh_events(
    packet: Mapping[str, Any], banks: Mapping[str, Any], *, rank: int, world: int,
    journal_path: Path, phase: str, before_update: int,
) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    events: dict[str, Mapping[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for index in range(rank, len(packet["new_packages"]), world):
        package = packet["new_packages"][index]
        event, receipt = _mine_one(banks["qwen"].model, banks["qwen"], package,
                                   banks["new_positive"][index], journal_path=journal_path,
                                   phase=phase, rank=rank, before_update=before_update)
        receipts.append(receipt)
        if event is not None:
            events[package["package_id"]] = event
    return events, receipts


def calibrate(
    *, input_path: Path, output_root: Path, sealed_output: Path
) -> dict[str, Any]:
    """Run the sole no-update gradient calibration and seal lambda into a new packet."""
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit()
            and torch.cuda.device_count() == 1, "calibration requires one explicit physical GPU")
    packet, _, _ = load_packet(input_path)
    require(packet["status"] == "calibration_pending", "calibration requires pending packet")
    require(not output_root.exists() and not sealed_output.exists(), "no calibration overwrite/retry")
    output_root.mkdir(parents=True)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    banks = _runtime_banks(packet, device=device, evidence_dir=output_root)
    model = banks["qwen"].model
    named, frozen = _select_trainable(model)
    from probes.dora_owner_learning.geometric_dedup_train import install_language_decoder_checkpointing
    from probes.dora_owner_learning.train import _tensor_state_hash

    checkpoint = install_language_decoder_checkpointing(model)
    checkpoint["enabled"] = packet["throughput"]["activation_checkpointing"]
    checkpoint["phase"] = "calibration"
    initial_selected = _tensor_state_hash(named)
    initial_frozen = _tensor_state_hash(frozen)
    versions = [(parameter, parameter._version) for _, parameter in frozen]
    references = _local_reference_cache(banks, rank=0, world=1)
    scorer = BatchedObjective(model)
    first = packet["schedule"][0]
    common = _step_items(packet, banks, references, first, rank=0, world=1,
                         fork_events={}, fork_lambda=0.0)
    model.zero_grad(set_to_none=True)
    common_components = backward_microbatches(
        scorer, common, microbatch_size=packet["throughput"]["microbatch_size"], synchronize=False
    )
    common_norm = _gradient_norm(named)
    if common_norm == 0:
        result = {"schema": CALIBRATION_SCHEMA, "status": "HOLD_zero_common_gradient",
                  "active_events": None, "common_gradient_norm": 0.0, "parameter_updates": 0}
        require(_tensor_state_hash(named) == initial_selected and _tensor_state_hash(frozen) == initial_frozen
                and all(parameter._version == version for parameter, version in versions),
                "held calibration changed parameter bytes/version")
        old_training.publish(output_root / "calibration.json", result)
        return result
    model.zero_grad(set_to_none=True)
    events, refresh = _refresh_events(
        packet, banks, rank=0, world=1,
        journal_path=output_root / "refresh-journal.jsonl",
        phase="calibration", before_update=0,
    )
    if not events:
        result = {"schema": CALIBRATION_SCHEMA, "status": "HOLD_no_events",
                  "active_events": 0, "refresh": refresh, "common_gradient_norm": common_norm,
                  "parameter_updates": 0}
        require(_tensor_state_hash(named) == initial_selected and _tensor_state_hash(frozen) == initial_frozen
                and all(parameter._version == version for parameter, version in versions),
                "held calibration changed parameter bytes/version")
        old_training.publish(output_root / "calibration.json", result)
        return result
    fork_items = _step_items(packet, banks, references, first, rank=0, world=1,
                             fork_events=events, fork_lambda=1.0)
    fork_items = [item for item in fork_items if item["component"] == "fork"]
    fork_components = backward_microbatches(
        scorer, fork_items, microbatch_size=packet["throughput"]["microbatch_size"], synchronize=False
    )
    fork_norm = _gradient_norm(named)
    if fork_norm == 0:
        result = {"schema": CALIBRATION_SCHEMA, "status": "HOLD_zero_fork_gradient",
                  "active_events": len(events), "refresh": refresh,
                  "common_gradient_norm": common_norm, "raw_fork_gradient_norm": 0.0,
                  "parameter_updates": 0}
        require(_tensor_state_hash(named) == initial_selected and _tensor_state_hash(frozen) == initial_frozen
                and all(parameter._version == version for parameter, version in versions),
                "held calibration changed parameter bytes/version")
        old_training.publish(output_root / "calibration.json", result)
        return result
    fork_lambda = calibrate_fork_lambda(common_gradient_norm=common_norm,
                                        raw_fork_gradient_norm=fork_norm,
                                        active_events=len(events))
    require(_tensor_state_hash(named) == initial_selected and _tensor_state_hash(frozen) == initial_frozen
            and all(parameter._version == version for parameter, version in versions),
            "calibration changed parameter bytes/version")
    result = {
        "schema": CALIBRATION_SCHEMA, "status": "passed", "source_packet": _binding(input_path),
        "active_events": len(events), "registered_packages": len(packet["new_packages"]),
        "common_gradient_norm": common_norm, "raw_fork_gradient_norm": fork_norm,
        "target_ratio": FORK_TARGET_RATIO, "fork_lambda": fork_lambda,
        "common_components": common_components, "fork_components": fork_components,
        "refresh": refresh, "parameter_updates": 0,
        "selected_state_sha256": initial_selected, "frozen_state_sha256": initial_frozen,
    }
    old_training.publish(output_root / "calibration.json", result)
    sealed = dict(packet)
    sealed.update(status="sealed_root_grant_pending", calibration=result)
    old_training.publish(sealed_output, sealed)
    validate_packet(sealed)
    return result


def execute_rank(
    *, input_path: Path, grant_path: Path, arm: str, updates: int,
    world_size: int, output_root: Path,
) -> None:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from probes.dora_owner_learning.geometric_dedup_train import (
        checkpointing_receipt, install_language_decoder_checkpointing,
    )
    from probes.dora_owner_learning.train import _save_adapter_only, _tensor_state_hash

    verify_root_grant(input_path, grant_path, updates=updates, arm=arm,
                      world_size=world_size, output_root=output_root)
    packet, receipt, _ = load_packet(input_path)
    require(arm in ("A", "B"), "paired arm")
    rank, local_rank, world = [int(os.environ.get(key, "-1")) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    require(world == world_size and world in (2, 8) and rank == local_rank
            and 0 <= rank < world and len(visible) == world and len(set(visible)) == world,
            "explicit single-node two/eight-rank topology")
    work_plan = planned_work_upper_bound(packet, rank=rank, world=world, updates=updates)
    require(work_plan["model_forwards_upper_bound"]
            <= packet["runtime"]["max_model_forwards_per_rank"]
            and work_plan["image_forwards_upper_bound"]
            <= packet["runtime"]["max_image_forwards_per_rank"],
            "planned work exceeds accepted forward envelope")
    run = output_root / "ranks" / f"rank{rank}"
    run.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda", local_rank)
    status, error = "failed", None
    counters = {"model_loads": 0, "model_forwards": 0, "image_forwards": 0,
                "optimizer_steps": 0, "refreshes": 0}
    started = time.monotonic()
    handles: list[Any] = []

    def expired(*_: Any) -> None:
        raise TimeoutError("training rank exceeded accepted wall-time envelope")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(math.ceil(packet["runtime"]["max_rank_seconds"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=900), device_id=device)
    try:
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        banks = _runtime_banks(packet, device=device, evidence_dir=run)
        counters["model_loads"] = 1
        model = banks["qwen"].model

        def count_model(*_: Any) -> None:
            counters["model_forwards"] += 1
            require(counters["model_forwards"] <= packet["runtime"]["max_model_forwards_per_rank"],
                    "accepted model-forward envelope")

        def count_image(*_: Any) -> None:
            counters["image_forwards"] += 1
            require(counters["image_forwards"] <= packet["runtime"]["max_image_forwards_per_rank"],
                    "accepted image-forward envelope")

        handles.append(model.register_forward_pre_hook(count_model))
        visual = [module for name, module in model.named_modules() if name.endswith("visual")]
        require(len(visual) == 1, "single visual module")
        handles.append(visual[0].register_forward_pre_hook(count_image))
        named, frozen = _select_trainable(model)
        frozen_hash = _tensor_state_hash(frozen)
        source_hash = _tensor_state_hash(named)
        references = _local_reference_cache(banks, rank=rank, world=world)
        checkpoint = install_language_decoder_checkpointing(model)
        checkpoint["enabled"] = packet["throughput"]["activation_checkpointing"]
        checkpoint["phase"] = "train"
        scorer = DDP(BatchedObjective(model), device_ids=[local_rank], output_device=local_rank,
                     broadcast_buffers=False, init_sync=False)
        optimizer = torch.optim.AdamW([parameter for _, parameter in named], **packet["optimizer"])
        fork_lambda = 0.0 if arm == "A" else packet["calibration"]["fork_lambda"]
        refresh_events: dict[str, Mapping[str, Any]] = {}
        refresh_receipts: list[dict[str, Any]] = []
        update_receipts = []
        for step in packet["schedule"][:updates]:
            if step["refresh_before_update"]:
                refresh_events, observed = _refresh_events(
                    packet, banks, rank=rank, world=world,
                    journal_path=run / "refresh-journal.jsonl",
                    phase="train", before_update=step["update"],
                )
                refresh_receipts.extend({"before_update": step["update"], **row} for row in observed)
                counters["refreshes"] += len(observed)
            items = _step_items(packet, banks, references, step, rank=rank, world=world,
                                fork_events=refresh_events, fork_lambda=fork_lambda)
            optimizer.zero_grad(set_to_none=True)
            components = backward_microbatches(
                scorer, items, microbatch_size=packet["throughput"]["microbatch_size"], synchronize=True
            )
            raw_norm = float(torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in named], packet["clip_gradient_norm"],
                error_if_nonfinite=True, foreach=False,
            ))
            optimizer.step()
            counters["optimizer_steps"] += 1
            update_receipt = {"update": step["update"], "components": components,
                              "raw_gradient_norm": raw_norm, "replays": len(items),
                              "active_local_forks": len(refresh_events) if arm == "B" else 0}
            _journal_completed_update(
                run / "update-journal.jsonl", rank=rank, arm=arm,
                update=step["update"], components=components,
                raw_gradient_norm=raw_norm, replays=len(items),
                active_local_forks=update_receipt["active_local_forks"], counters=counters,
            )
            update_receipts.append(update_receipt)
        require(_tensor_state_hash(frozen) == frozen_hash, "frozen base/special delta changed")
        final_items = _step_items(packet, banks, references, packet["schedule"][0],
                                  rank=rank, world=world, fork_events={}, fork_lambda=0.0)
        final_items = [item for item in final_items if item["component"] in (
            "old_witness_kl", "new_witness_kl", "normal_kl"
        )]
        final_protection = evaluate_microbatches(
            scorer.module, final_items, microbatch_size=packet["throughput"]["microbatch_size"]
        )
        final_scores = None
        if rank == 0:
            final_scores = old_training._score_records(model, [*banks["old_positive"], *banks["new_positive"]])
            saved = _save_adapter_only(model, source_root=N16_ROOT / "adapter", output=output_root / "adapter")
            old_training.publish(output_root / "provisional.json", {
                "schema": "owner_successor_scale.training.receipt.v1", "status": "unsealed_candidate",
                "arm": arm, "updates": updates, "world_size": world, "input": _binding(input_path),
                "grant": _binding(grant_path), "source_adapter": receipt["saved_adapter"],
                "source_adapter_state_sha256": source_hash, "saved_adapter": saved,
                "final_live_positive_scores": final_scores,
            })
        old_training.publish(run / "updates.json", {"updates": update_receipts})
        old_training.publish(run / "refresh.json", {"refreshes": refresh_receipts})
        old_training.publish(run / "final-protection.json", {
            "components": final_protection, "records": len(final_items),
        })
        old_training.publish(run / "state.json", {
            "rank": rank, "selected_sha256": _tensor_state_hash(named), "frozen_sha256": frozen_hash,
            "activation_checkpointing": checkpointing_receipt(model, checkpoint),
        })
        dist.barrier()
        resources = {
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }
        require(resources["elapsed_seconds"] <= packet["runtime"]["max_rank_seconds"]
                and resources["peak_cuda_allocated_bytes"] <= packet["runtime"]["max_cuda_allocated_bytes"]
                and resources["peak_cuda_reserved_bytes"] <= packet["runtime"]["max_cuda_reserved_bytes"]
                and resources["peak_rss_bytes"] <= packet["runtime"]["max_rss_bytes"],
                "accepted rank resource envelope")
        status = "completed"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        signal.alarm(0)
        for handle in handles:
            handle.remove()
        resources = {
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }
        terminal = {"rank": rank, "status": status, "error": error,
                    "arm": arm, "updates": updates, "work_plan": work_plan,
                    "counters": counters, "resources": resources}
        for name in ("refresh-journal.jsonl", "update-journal.jsonl"):
            path = run / name
            if path.exists():
                terminal[name.removesuffix(".jsonl").replace("-", "_")] = _binding(path)
        old_training.publish(run / "terminal.json", terminal)
        dist.destroy_process_group()


def launch(
    *, input_path: Path, grant_path: Path, arm: str, updates: int,
    world_size: int, output_root: Path,
) -> dict[str, Any]:
    verify_root_grant(input_path, grant_path, updates=updates, arm=arm,
                      world_size=world_size, output_root=output_root)
    packet, _, _ = load_packet(input_path)
    work_plans = [planned_work_upper_bound(packet, rank=rank, world=world_size, updates=updates)
                  for rank in range(world_size)]
    require(all(plan["model_forwards_upper_bound"]
                <= packet["runtime"]["max_model_forwards_per_rank"]
                and plan["image_forwards_upper_bound"]
                <= packet["runtime"]["max_image_forwards_per_rank"] for plan in work_plans),
            "planned launch exceeds accepted forward envelope")
    require(not output_root.exists(), "no training output overwrite/retry")
    output_root.mkdir(parents=True)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc-per-node={world_size}", "--module", "probes.owner_successor_scale.training", "rank",
               "--input", str(input_path.resolve()), "--grant", str(grant_path.resolve()),
               "--arm", arm, "--updates", str(updates), "--world-size", str(world_size),
               "--output-root", str(output_root.resolve())]
    old_training.publish(output_root / "launch.json", {"command": command, "input": _binding(input_path),
                                                        "grant": _binding(grant_path), "arm": arm,
                                                        "updates": updates, "world_size": world_size,
                                                        "work_plans": work_plans})
    with (output_root / "launcher.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    old_training.publish(output_root / "launcher-exit.json", {"returncode": result.returncode})
    require(result.returncode == 0, f"training launcher failed with exit {result.returncode}")
    terminals = [json.loads((output_root / "ranks" / f"rank{rank}" / "terminal.json").read_text())
                 for rank in range(world_size)]
    require(all(row["status"] == "completed" for row in terminals), "rank terminal gate")
    update_payloads = [json.loads((output_root / "ranks" / f"rank{rank}" / "updates.json").read_text())
                       for rank in range(world_size)]
    require(all(len(payload["updates"]) == updates for payload in update_payloads),
            "rank update receipt count")
    global_updates = []
    for index in range(updates):
        rows = [payload["updates"][index] for payload in update_payloads]
        require({row["update"] for row in rows} == {index + 1}, "rank update identity")
        global_updates.append({
            "update": index + 1,
            "objective_components": {
                key: sum(row["components"][key] for row in rows) / world_size
                for key in BatchedObjective.COMPONENT_ORDER
            },
            "active_forks": sum(row["active_local_forks"] for row in rows),
            "replays": sum(row["replays"] for row in rows),
        })
    protection_payloads = [json.loads(
        (output_root / "ranks" / f"rank{rank}" / "final-protection.json").read_text()
    ) for rank in range(world_size)]
    global_protection = {
        key: sum(payload["components"][key] for payload in protection_payloads) / world_size
        for key in BatchedObjective.COMPONENT_ORDER
    }
    provisional = json.loads((output_root / "provisional.json").read_text())
    receipt = {**provisional, "status": "technically_completed_cold_pending",
               "terminals": [_binding(output_root / "ranks" / f"rank{rank}" / "terminal.json")
                             for rank in range(world_size)],
               "update_records": [_binding(output_root / "ranks" / f"rank{rank}" / "updates.json")
                                  for rank in range(world_size)],
               "refresh_records": [_binding(output_root / "ranks" / f"rank{rank}" / "refresh.json")
                                   for rank in range(world_size)],
               "final_protection_records": [
                   _binding(output_root / "ranks" / f"rank{rank}" / "final-protection.json")
                   for rank in range(world_size)
               ],
               "global_updates": global_updates,
               "global_final_protection": global_protection,
               "launcher_exit": _binding(output_root / "launcher-exit.json")}
    old_training.publish(output_root / "receipt.json", receipt)
    return receipt


def _cold_positive_entries(
    old_packet: Mapping[str, Any], packet: Mapping[str, Any], *, qwen: Any, frontend: Any,
    config: Any, raw: Mapping[str, Any],
) -> list[dict[str, Any]]:
    entries = [
        old_training._materialize(record, qwen=qwen, frontend=frontend, config=config, raw=raw)
        for record in old_packet["positive_records"]
    ]
    entries.extend(
        old_training._materialize(
            package["c_record"], qwen=qwen, frontend=frontend, config=config, raw=raw,
        )
        for package in packet["new_packages"]
    )
    return entries


def cold_check(*, input_path: Path, output_root: Path) -> dict[str, Any]:
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit() and torch.cuda.device_count() == 1,
            "cold check requires one explicit physical GPU")
    packet, _, _ = load_packet(input_path)
    receipt = json.loads((output_root / "receipt.json").read_text())
    require(receipt["status"] == "technically_completed_cold_pending"
            and receipt["input"] == _binding(input_path), "cold/training receipt identity")
    old_training.old._verify_identity(receipt["saved_adapter"], label="saved adapter")
    run = output_root / "cold"
    run.mkdir(exist_ok=False)
    old_packet, anchor, _, _ = old_training.load_packet(packet["n16_training_input"]["path"])
    qwen, frontend, config, raw = old_training._load_model(
        anchor, adapter_path=receipt["saved_adapter"]["root"], device=torch.device("cuda", 0),
        evidence_dir=run, packet={"materialization_raw_sources": packet["materialization_raw_sources"]},
    )
    raw = _composed_runtime_raw(packet, old_packet, config_input=config.data.input_jsonl)
    entries = _cold_positive_entries(
        old_packet, packet, qwen=qwen, frontend=frontend, config=config, raw=raw,
    )
    observed = old_training._score_records(qwen.model, entries)
    expected = receipt["final_live_positive_scores"]
    require(set(observed) == set(expected), "cold positive record IDs")
    deltas = {record_id: {key: observed[record_id][key] - expected[record_id][key] for key in (
        "sum_logprob", "mean_logprob", "mean_target_margin", "min_target_margin"
    )} for record_id in observed}
    require(all(observed[key][field] == expected[key][field] for key in observed
                for field in ("token_count", "argmax_target_tokens")), "cold discrete positive scores")
    require(all(abs(value) <= 1e-5 for row in deltas.values() for value in row.values()),
            "cold/live positive tolerance")
    result = {"schema": "owner_successor_scale.training.cold_check.v1", "status": "passed",
              "training_receipt": _binding(output_root / "receipt.json"), "positive_scores": observed,
              "live_score_deltas": deltas}
    old_training.publish(output_root / "cold-check.json", result)
    return result


def verify_root_grant(
    packet_path: str | Path, grant_path: str | Path, *, updates: int,
    arm: str | None = None, world_size: int | None = None, output_root: Path | None = None,
) -> dict[str, Any]:
    """Bind the exact sealed bytes and exact permitted update count."""
    packet = json.loads(Path(packet_path).read_text())
    validate_packet(packet)
    require(packet["status"] == "sealed_root_grant_pending", "root cannot grant unsealed calibration")
    grant = json.loads(Path(grant_path).read_text())
    require(grant.get("schema") == GRANT_SCHEMA and grant.get("status") == "granted",
            "root launch grant")
    require(grant.get("packet") == _binding(packet_path), "grant does not bind exact training packet")
    require(grant.get("authorized_updates") == updates and updates in (1, UPDATES),
            "grant update scope")
    if updates == 1:
        require("integrated_two_rank_one_update_save_cold"
                in packet["stage_permissions"]["authorized"], "integrated smoke stage not accepted")
        require(world_size in (None, 2), "integrated smoke must use two ranks")
    else:
        smoke_binding = grant.get("integrated_smoke_acceptance")
        _binding_shape(smoke_binding, "bound integrated smoke acceptance")
        require(_binding(smoke_binding["path"]) == smoke_binding,
                "bound integrated smoke acceptance changed")
        smoke = json.loads(Path(smoke_binding["path"]).read_text())
        require(smoke.get("schema") == INTEGRATED_SMOKE_ACCEPTANCE_SCHEMA
                and smoke.get("status") == "lead_accepted"
                and smoke.get("packet") == _binding(packet_path)
                and smoke.get("authorized_full_256") is True,
                "full256 lacks accepted integrated update/save/cold evidence")
        for field in ("training_receipt", "cold_check"):
            _binding_shape(smoke.get(field), f"bound integrated smoke {field}")
            require(_binding(smoke[field]["path"]) == smoke[field],
                    f"bound integrated smoke {field} changed")
        require(world_size in (None, 8), "full256 must use eight ranks")
    if arm is not None:
        require(grant.get("authorized_arm") == arm, "grant arm scope")
    if world_size is not None:
        require(grant.get("authorized_world_size") == world_size and world_size in (2, 8),
                "grant world-size scope")
    if output_root is not None:
        require(Path(grant.get("authorized_output_root", "")).resolve() == output_root.resolve(),
                "grant output-root scope")
    require(grant.get("root_admission_floor") == {"packages_at_least": 32, "images_at_least": 16},
            "root physical admission floor")
    return grant


def readiness_receipt() -> dict[str, Any]:
    return {
        "schema": "owner_successor_scale.training.implementation_readiness.v1",
        "status": "HOLD_root_physical_decisions_and_staged_replay_receipt",
        "implemented": [
            "balanced_256_update_schedule",
            "six_independently_normalized_common_banks",
            "actual_repeat_vs_fixed_credible_first_fork_hinge",
            "fixed_bank_zero_on_nonrepeat_gating",
            "one_no_update_gradient_norm_calibration",
            "root_bound_launch_gate",
            "exact_source_physical_bank_materializer",
            "separate_c_w_and_canonical_alias_history_admission",
            "pool_order_distinct_image_first_selection",
            "batched_replay_consumer",
            "incremental_microbatch_backward_without_bank_graph_retention",
            "prepare_calibrate_distributed_train_save_and_cold_readback_entrypoints",
            "full_greedy_fixed_h_plus_refresh_mining",
            "final_registered_protection_readback",
        ],
        "not_executed": ["model_load", "gradient_calibration", "optimizer_update", "adapter_save", "cold_readback"],
        "stage_gates": [
            "root complete physical decisions plus complete review index -> physical bank",
            "root staged replay receipt -> prepare",
            "no-update calibration -> sealed packet",
            "exact root grant -> sole two-rank one-update/save/cold",
            "lead-accepted integrated smoke receipt plus new root grant -> full256",
        ],
        "scientific_identity": {
            "source": "N16_snapshot",
            "arms": {"A": "common_only", "B": "common_plus_calibrated_fork"},
            "fork_gamma_logits": FORK_GAMMA,
            "fork_target_common_gradient_ratio": FORK_TARGET_RATIO,
            "natural_output_allowance": MAX_NEW_TOKENS,
        },
        "entrypoints": {
            "physical_source_preflight": "python -m probes.owner_successor_scale.training physical-source-preflight --pool POOL --confirmation-selection CONFIRMATION --review-index REVIEW_INDEX --output PREFLIGHT",
            "materialize_physical_bank": "python -m probes.owner_successor_scale.training materialize-physical-bank --pool POOL --confirmation-selection CONFIRMATION --review-index FINAL_REVIEW_INDEX --root-decisions ROOT_DECISIONS --output BANK",
            "prepare": "python -m probes.owner_successor_scale.training prepare --physical-package-bank BANK --replay-acceptance ACCEPTANCE --output INPUT",
            "calibrate_no_update": "CUDA_VISIBLE_DEVICES=GPU python -m probes.owner_successor_scale.training calibrate --input INPUT --output-root CALIBRATION_DIR --sealed-output SEALED_INPUT",
            "one_update_two_rank": "CUDA_VISIBLE_DEVICES=GPU0,GPU1 python -m probes.owner_successor_scale.training launch --input SEALED_INPUT --grant GRANT --arm A_OR_B --updates 1 --world-size 2 --output-root RUN",
            "cold": "CUDA_VISIBLE_DEVICES=GPU python -m probes.owner_successor_scale.training cold-check --input SEALED_INPUT --output-root RUN",
            "full_after_integrated_smoke_acceptance_and_separate_grant": "CUDA_VISIBLE_DEVICES=EIGHT_GPUS python -m probes.owner_successor_scale.training launch --input SEALED_INPUT --grant GRANT_WITH_INTEGRATED_SMOKE_ACCEPTANCE --arm A_OR_B --updates 256 --world-size 8 --output-root RUN",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "source-preflight", "physical-source-preflight", "materialize-physical-bank", "prepare", "verify", "readiness",
        "check-grant", "calibrate", "launch", "rank", "cold-check",
    ))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--grant", type=Path)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--physical-package-bank", type=Path)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--confirmation-selection", type=Path)
    parser.add_argument("--review-index", type=Path, action="append")
    parser.add_argument("--root-decisions", type=Path)
    parser.add_argument("--replay-acceptance", type=Path)
    parser.add_argument("--replay-diagnostic", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--sealed-output", type=Path)
    parser.add_argument("--arm", choices=("A", "B"))
    parser.add_argument("--world-size", type=int, choices=(2, 8))
    args = parser.parse_args()
    if args.command == "readiness":
        print(json.dumps(readiness_receipt(), sort_keys=True))
        return
    if args.command == "source-preflight":
        require(args.replay_diagnostic is not None and args.output is not None,
                "source preflight diagnostic/output required")
        source_preflight(replay_diagnostic=args.replay_diagnostic, output=args.output)
        return
    if args.command == "materialize-physical-bank":
        require(args.pool is not None and args.confirmation_selection is not None
                and args.review_index and args.root_decisions is not None and args.output is not None,
                "physical bank pool/confirmation/review/decisions/output required")
        materialize_physical_bank(
            pool_path=args.pool, confirmation_selection_path=args.confirmation_selection,
            review_index_paths=args.review_index, root_decisions_path=args.root_decisions,
            output=args.output,
        )
        return
    if args.command == "physical-source-preflight":
        require(args.pool is not None and args.confirmation_selection is not None
                and args.review_index and args.output is not None,
                "physical source preflight pool/confirmation/review/output required")
        physical_source_preflight(
            pool_path=args.pool, confirmation_selection_path=args.confirmation_selection,
            review_index_paths=args.review_index, output=args.output,
        )
        return
    if args.command == "prepare":
        require(args.physical_package_bank is not None and args.replay_acceptance is not None
                and args.output is not None, "prepare bank/acceptance/output required")
        prepare_packet(physical_package_bank=args.physical_package_bank,
                       replay_acceptance=args.replay_acceptance, output=args.output)
        return
    require(args.input is not None, "input packet required")
    if args.command == "calibrate":
        require(args.output_root is not None and args.sealed_output is not None,
                "calibration output-root/sealed-output required")
        calibrate(input_path=args.input, output_root=args.output_root, sealed_output=args.sealed_output)
        return
    if args.command in ("launch", "rank"):
        require(args.grant is not None and args.arm is not None and args.updates is not None
                and args.world_size is not None and args.output_root is not None,
                "training grant/arm/updates/world/output required")
        function = launch if args.command == "launch" else execute_rank
        function(input_path=args.input, grant_path=args.grant, arm=args.arm, updates=args.updates,
                 world_size=args.world_size, output_root=args.output_root)
        return
    if args.command == "cold-check":
        require(args.output_root is not None, "cold output-root required")
        cold_check(input_path=args.input, output_root=args.output_root)
        return
    packet, _, _ = load_packet(args.input)
    if args.command == "check-grant":
        require(args.grant is not None and args.updates is not None, "grant and updates required")
        verify_root_grant(args.input, args.grant, updates=args.updates, arm=args.arm,
                          world_size=args.world_size, output_root=args.output_root)
    print(json.dumps({"status": "CPU_valid", "packet_status": packet["status"],
                      "new_packages": len(packet["new_packages"]), "updates": packet["updates"]}))


if __name__ == "__main__":
    main()
