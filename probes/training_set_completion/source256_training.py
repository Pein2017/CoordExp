"""Irreducible Source256 paired-training consumer contracts.

The data producer owns literal routes and the balanced image schedule.  This
module selects the route consumed by each arm and returns local objective
contributions whose cross-rank SUM is the registered 50/50, sample-equal loss.
It deliberately does not construct or repair Source prefixes.
"""
from __future__ import annotations

from probes.training_set_completion import replay

from collections import Counter
import copy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import time
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.distributed as dist

from probes.training_set_completion import distributed
from probes.training_set_completion import training


SCHEMA = "training_set_completion.source256_training.v1"
MANIFEST_SCHEMA = "training_set_completion.source256_training_manifest.v1"
ARMS = ("A", "B")
BRANCHES = ("common", "variable")
IMAGE_COUNT = 256
UPDATE_COUNT = 64
BRANCH_IMAGE_COUNT = 32
PRESENTATIONS_PER_UPDATE = 64
PRESENTATIONS_PER_IMAGE_PER_BRANCH = 8
MIN_ELIGIBLE_IMAGES = 64
MIN_COMPLETION_PRESENTATIONS = 512
REQUIRED_WORLD_SIZE = 4
SOURCE_ADAPTER_FINGERPRINT = "b8ca2461c93bf32e886c9e42f7ab0e52495ef2d439a605207c86af7258408815"
SOURCE_ADAPTER_SCALAR_COUNT = 18_006_016


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _image_ids(value: Any, *, name: str) -> list[int]:
    require(
        isinstance(value, list)
        and len(value) == BRANCH_IMAGE_COUNT
        and all(type(image_id) is int and image_id >= 0 for image_id in value),
        f"{name} image IDs",
    )
    require(len(set(value)) == len(value), f"{name} duplicate image")
    return list(value)


def validate_schedule(
    value: Mapping[str, Any], *, expected_image_ids: Sequence[int]
) -> dict[str, Any]:
    """Validate 64 balanced updates without inventing a schedule in the runtime."""

    image_ids = list(expected_image_ids)
    require(
        len(image_ids) == IMAGE_COUNT
        and len(set(image_ids)) == IMAGE_COUNT
        and all(type(image_id) is int and image_id >= 0 for image_id in image_ids),
        "exact unique train256 image IDs",
    )
    updates = value.get("updates")
    require(isinstance(updates, list) and len(updates) == UPDATE_COUNT, "64 updates")
    expected = set(image_ids)
    counts = {branch: Counter() for branch in BRANCHES}
    checked: list[dict[str, Any]] = []
    for expected_step, update in enumerate(updates, start=1):
        require(
            isinstance(update, Mapping)
            and set(update) == {"step", "common_image_ids", "variable_image_ids"},
            f"schedule update {expected_step} fields",
        )
        require(update["step"] == expected_step, "contiguous one-based update steps")
        common = _image_ids(update["common_image_ids"], name=f"step {expected_step} common")
        variable = _image_ids(
            update["variable_image_ids"], name=f"step {expected_step} variable"
        )
        require(set(common) <= expected and set(variable) <= expected, "unknown schedule image")
        counts["common"].update(common)
        counts["variable"].update(variable)
        checked.append(
            {
                "step": expected_step,
                "common_image_ids": common,
                "variable_image_ids": variable,
            }
        )
    for branch in BRANCHES:
        require(
            counts[branch]
            == Counter({image_id: PRESENTATIONS_PER_IMAGE_PER_BRANCH for image_id in image_ids}),
            f"{branch} schedule is not eight presentations per image",
        )
    return {
        "updates": checked,
        "image_count": IMAGE_COUNT,
        "updates_count": UPDATE_COUNT,
        "presentations_per_update": PRESENTATIONS_PER_UPDATE,
        "presentations_per_image": 2 * PRESENTATIONS_PER_IMAGE_PER_BRANCH,
        "branch_presentations": {
            branch: sum(counts[branch].values()) for branch in BRANCHES
        },
    }


def _checked_route(
    route: Mapping[str, Any],
    *,
    eos_token_id: int,
    coordinate_token_ids: Sequence[int] | None,
) -> dict[str, Any]:
    return training.validate_route(
        route,
        eos_token_id=eos_token_id,
        coordinate_token_ids=coordinate_token_ids,
    )


def _validate_explicit_route_surface(
    route: Mapping[str, Any], *, expected_kind: str
) -> None:
    """Prove the explicit data masks are exactly the masks consumed by helpers."""

    continuation = route["continuation_token_ids"]
    weights = route["ce_weights"]
    labels = route.get("labels")
    geometry_weights = route.get("geometry_weights")
    geometry_targets = route.get("geometry_target_bins")
    require(
        isinstance(labels, list)
        and labels
        == [token if weight == 1 else -100 for token, weight in zip(continuation, weights, strict=True)],
        "explicit labels differ from consumed CE mask",
    )
    expected_geometry = [0] * len(continuation)
    expected_bins = [-100] * len(continuation)
    for box in route["trusted_boxes"]:
        for position, value in zip(
            (
                box["x1_position"],
                box["y1_position"],
                box["x2_position"],
                box["y2_position"],
            ),
            box["expected_bins"],
            strict=True,
        ):
            require(expected_geometry[position] == 0, "overlapping geometry targets")
            expected_geometry[position] = 1
            expected_bins[position] = value
    require(
        geometry_weights == expected_geometry,
        "explicit geometry weights differ from consumed trusted boxes",
    )
    require(
        geometry_targets == expected_bins,
        "explicit geometry bins differ from consumed trusted boxes",
    )
    provenance = route.get("provenance")
    required = {
        "route_kind",
        "bank_owner_ids",
        "prefix_owner_ids",
        "suffix_owner_ids",
        "prefix_token_ids",
        "suffix_token_ids",
        "prefix_token_ids_sha256",
        "suffix_token_ids_sha256",
        "source_greedy_generated_token_ids_sha256",
        "mask_semantics",
        "geometry_semantics",
    }
    require(isinstance(provenance, Mapping) and set(provenance) == required, "route provenance fields")
    require(provenance["route_kind"] == expected_kind, "route kind")
    for key in ("bank_owner_ids", "prefix_owner_ids", "suffix_owner_ids"):
        values = provenance[key]
        require(
            isinstance(values, list)
            and len(values) == len(set(values))
            and all(isinstance(item, str) and item for item in values),
            f"{key} owner IDs",
        )
    prefix = provenance["prefix_token_ids"]
    suffix = provenance["suffix_token_ids"]
    require(
        isinstance(prefix, list)
        and isinstance(suffix, list)
        and prefix + suffix == continuation,
        "prefix/suffix token partition",
    )
    require(
        provenance["prefix_token_ids_sha256"] == training.digest(prefix)
        and provenance["suffix_token_ids_sha256"] == training.digest(suffix),
        "prefix/suffix token digest",
    )
    require(
        isinstance(provenance["source_greedy_generated_token_ids_sha256"], str)
        and len(provenance["source_greedy_generated_token_ids_sha256"]) == 64,
        "Source greedy token digest",
    )
    require(
        all(
            isinstance(provenance[key], str) and provenance[key]
            for key in ("mask_semantics", "geometry_semantics")
        ),
        "route mask semantics",
    )
    bank = provenance["bank_owner_ids"]
    prefix_owners = provenance["prefix_owner_ids"]
    suffix_owners = provenance["suffix_owner_ids"]
    require(
        not (set(prefix_owners) & set(suffix_owners))
        and set(prefix_owners) | set(suffix_owners) == set(bank),
        "prefix/suffix owners must partition the bank",
    )
    require(
        weights == [0] * len(prefix) + [1] * len(suffix),
        "CE mask differs from prefix/suffix partition",
    )
    if expected_kind == "canonical":
        require(
            not prefix
            and not prefix_owners
            and suffix == continuation
            and suffix_owners == bank,
            "canonical route provenance",
        )
    else:
        require(prefix and prefix_owners and suffix and suffix_owners, "completion prefix/suffix provenance")


def validate_route_record(
    value: Mapping[str, Any],
    *,
    eos_token_id: int,
    coordinate_token_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Validate one canonical route and its optional Source-prefix completion."""

    require(
        isinstance(value, Mapping)
        and set(value)
        == {"image_id", "example_id", "canonical_route", "completion_route", "eligibility"},
        "Source256 route record fields",
    )
    image_id = value["image_id"]
    require(type(image_id) is int and image_id >= 0, "route-record image ID")
    example_id = value["example_id"]
    require(isinstance(example_id, str) and example_id, "route-record example ID")
    canonical = _checked_route(
        value["canonical_route"],
        eos_token_id=eos_token_id,
        coordinate_token_ids=coordinate_token_ids,
    )
    require(canonical["image_id"] == image_id, "canonical image ID")
    require(canonical["example_id"] == example_id, "canonical example ID")
    require(all(canonical["ce_weights"]), "canonical route must supervise every target")
    _validate_explicit_route_surface(canonical, expected_kind="canonical")
    eligibility = value["eligibility"]
    require(
        isinstance(eligibility, Mapping)
        and type(eligibility.get("fully_eligible")) is bool,
        "eligibility record",
    )
    completion_value = value["completion_route"]
    if not eligibility["fully_eligible"]:
        require(completion_value is None, "ineligible image exposes a completion route")
        require(bool(eligibility["fallback_reason"]), "ineligible image needs fallback reason")
        completion = None
    else:
        require(completion_value is not None, "eligible image lacks completion route")
        require(eligibility["fallback_reason"] is None, "eligible image has fallback reason")
        completion = _checked_route(
            completion_value,
            eos_token_id=eos_token_id,
            coordinate_token_ids=coordinate_token_ids,
        )
        require(completion["image_id"] == image_id, "completion image ID")
        require(completion["example_id"] == example_id, "completion example ID")
        for field in ("case", "image_identity", "prompt_token_ids"):
            require(completion[field] == canonical[field], f"completion changed {field}")
        weights = completion["ce_weights"]
        first_active = weights.index(1)
        require(first_active > 0, "completion requires a nonempty masked Source prefix")
        require(
            weights == [0] * first_active + [1] * (len(weights) - first_active),
            "completion mask must be one prefix then one supervised suffix",
        )
        require(
            completion["continuation_token_ids"][-1] == eos_token_id,
            "completion suffix requires standard EOS",
        )
        _validate_explicit_route_surface(
            completion, expected_kind="fixed_source_prefix_completion"
        )
    return {
        "image_id": image_id,
        "example_id": example_id,
        "canonical_route": canonical,
        "completion_route": completion,
        "eligibility": dict(eligibility),
    }


def validate_records(
    values: Sequence[Mapping[str, Any]],
    *,
    eos_token_id: int,
    coordinate_token_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    require(isinstance(values, list) and len(values) == IMAGE_COUNT, "exact train256 routes")
    checked = [
        validate_route_record(
            value,
            eos_token_id=eos_token_id,
            coordinate_token_ids=coordinate_token_ids,
        )
        for value in values
    ]
    require(
        len({record["image_id"] for record in checked}) == IMAGE_COUNT,
        "duplicate train256 image",
    )
    return checked


def validate_preparation(
    value: Mapping[str, Any], *, verify_sources: bool = True
) -> dict[str, Any]:
    """Admit the producer artifact and re-prove its runtime-consumed projection."""

    from probes.training_set_completion import source256_data

    prepared = source256_data.validate_preparation(
        value, verify_sources=verify_sources
    )
    require(prepared.get("status") == "candidate_ready", "preparation failed scarcity gate")
    contract = prepared.get("identity", {}).get("runtime_contract")
    require(isinstance(contract, Mapping), "runtime identity contract")
    eos_token_id = contract.get("eos_token_id")
    coordinate_token_ids = contract.get("coordinate_token_ids")
    require(eos_token_id == source256_data.EOS, "runtime EOS identity")
    require(
        coordinate_token_ids
        == list(range(source256_data.COORD_START, source256_data.COORD_START + 1000)),
        "runtime coordinate-token identity",
    )
    records = validate_records(
        prepared["routes"],
        eos_token_id=eos_token_id,
        coordinate_token_ids=coordinate_token_ids,
    )
    schedule = validate_schedule(
        prepared["schedule"],
        expected_image_ids=[record["image_id"] for record in records],
    )
    observed_gate = eligibility_gate(records, schedule)
    declared_gate = prepared["gate"]
    require(
        declared_gate["passed"]
        and declared_gate["disposition"] == "ready_for_runtime_qualification"
        and declared_gate["fully_eligible_count"]
        == observed_gate["eligible_image_count"]
        and declared_gate["effective_completion_presentations"]
        == observed_gate["completion_presentations"]
        and declared_gate["total_presentations"] == observed_gate["total_presentations"]
        and declared_gate["effective_completion_fraction"]
        == observed_gate["completion_presentation_fraction"],
        "runtime eligibility gate projection",
    )
    return {
        "preparation": dict(prepared),
        "records": records,
        "schedule": schedule,
        "gate": observed_gate,
        "runtime_contract": dict(contract),
    }


def hydrate_bound_cases(prepared: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Join lean v3 routes to their bound train rows for native materialization.

    The preparation intentionally stores only the immutable prompt/media identity
    needed by the learning contract.  ``build_bound_native_requests`` additionally
    needs the source JSON row and a few deterministic image-plan dimensions.  Join
    those fields from the preparation's already-verified ``train_jsonl`` binding;
    never replan or rewrite prompt/media authority here.
    """

    source = prepared.get("preparation", {}).get("sources", {}).get("train_jsonl")
    require(isinstance(source, Mapping), "bound train JSONL source")
    path = Path(str(source.get("path", ""))).resolve(strict=True)
    require(training.binding(path) == dict(source), "bound train JSONL changed")
    raw_rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    require(len(raw_rows) == IMAGE_COUNT, "train JSONL must contain train256")
    row_by_image: dict[int, tuple[int, Mapping[str, Any]]] = {}
    for row_index, raw in enumerate(raw_rows):
        require(isinstance(raw, Mapping), "train JSONL row")
        image_id = raw.get("image_id")
        require(type(image_id) is int and image_id not in row_by_image, "unique train image ID")
        row_by_image[image_id] = (row_index, raw)

    hydrated = copy.deepcopy(prepared["records"])
    for record in hydrated:
        image_id = int(record["image_id"])
        require(image_id in row_by_image, "route image missing from bound train JSONL")
        row_index, raw = row_by_image[image_id]
        require(
            str(record["example_id"]) == f"coco2017_train_{image_id:012d}",
            "route/train example identity",
        )
        images = raw.get("images")
        require(isinstance(images, list) and len(images) == 1, "one train image reference")
        resolved_image = (path.parent / str(images[0])).resolve(strict=True)
        canonical = record["canonical_route"]
        lean_case = canonical["case"]
        plan = lean_case["image_plan"]
        require(
            Path(str(lean_case["image_path"])).resolve(strict=True) == resolved_image,
            "route/train image path identity",
        )
        width, height = raw.get("width"), raw.get("height")
        require(
            type(width) is int and width > 0 and type(height) is int and height > 0,
            "train image dimensions",
        )
        grid = plan.get("observed_image_grid_thw")
        require(
            isinstance(grid, list)
            and len(grid) == 3
            and all(type(item) is int and item > 0 for item in grid),
            "route image grid",
        )
        raw_patch_rows = math.prod(grid)
        require(raw_patch_rows % 4 == 0, "Source256 merge-size-two image grid")
        full_case = {
            "row_id": str(record["example_id"]),
            "row_index": row_index,
            "input_record": raw,
            "image_path": str(resolved_image),
            "image_width": width,
            "image_height": height,
            "image_plan": {
                **plan,
                "backend_prompt_token_count": len(canonical["prompt_token_ids"]),
                "merged_visual_tokens": raw_patch_rows // 4,
                "logical_transform_id": "identity",
            },
        }
        for key in ("canonical_route", "completion_route"):
            route = record[key]
            if route is None:
                continue
            require(
                route["case"] == lean_case
                and route["prompt_token_ids"] == canonical["prompt_token_ids"]
                and route["image_identity"] == canonical["image_identity"],
                "canonical/completion native input identity",
            )
            route["case"] = full_case
    require({int(record["image_id"]) for record in hydrated} == set(row_by_image), "train256 join")
    return hydrated


def source_adapter_scalar_count(adapter: Mapping[str, Any]) -> int:
    """Return the receipt-declared scalar count from literal tensor shapes."""

    tensors = adapter.get("tensor_manifest", {}).get("tensors")
    expected = adapter.get("semantic_identity", {}).get("tensor_key_count")
    require(
        isinstance(tensors, list)
        and len(tensors) == expected == 588,
        "Source adapter tensor manifest count",
    )
    total = 0
    for item in tensors:
        shape = item.get("shape") if isinstance(item, Mapping) else None
        require(
            isinstance(shape, list)
            and bool(shape)
            and all(type(axis) is int and axis > 0 for axis in shape),
            "Source adapter tensor shape",
        )
        total += math.prod(shape)
    require(total == SOURCE_ADAPTER_SCALAR_COUNT, "Source adapter scalar count")
    return total


def validate_training_recipe(
    value: Mapping[str, Any], *, manifest_schema: str,
    producer_path: Path, verify_sources: bool = True,
) -> dict[str, Any]:
    """Validate one arm's lean manifest against its immutable preparation."""

    required = {
        "schema",
        "status",
        "arm",
        "mode",
        "sources",
        "preparation",
        "source_adapter",
        "model_config",
        "optimizer",
        "scheduler",
        "objective",
        "validity_hinge",
        "runtime",
        "content_sha256",
    }
    require(set(value) == required, "training manifest fields")
    require(value["schema"] == manifest_schema, "training manifest schema")
    require(
        value["content_sha256"]
        == training.digest({key: item for key, item in value.items() if key != "content_sha256"}),
        "training manifest content digest",
    )
    require(value["status"] == "candidate_ready", "training manifest status")
    require(value["arm"] in ARMS and value["mode"] in ("qualification", "main"), "arm/mode")
    sources = value["sources"]
    require(
        isinstance(sources, Mapping)
        and set(sources) == {"producer", "source_config"},
        "training source bindings",
    )
    if verify_sources:
        for name, source in sources.items():
            require(training.binding(source["path"]) == source, f"{name} source changed")
        require(
            Path(sources["producer"]["path"]).resolve() == producer_path.resolve(),
            "training producer path",
        )
        require(
            training.binding(value["preparation"]["path"]) == value["preparation"],
            "preparation bytes changed",
        )
    prepared = validate_preparation(
        json.loads(Path(value["preparation"]["path"]).read_text()),
        verify_sources=verify_sources,
    )
    contract = prepared["runtime_contract"]
    adapter = value["source_adapter"]
    require(
        isinstance(adapter, Mapping)
        and adapter.get("fingerprint") == SOURCE_ADAPTER_FINGERPRINT
        and Path(adapter.get("root", "")).resolve()
        == Path(contract["adapter_root"]).resolve(),
        "Source adapter identity",
    )
    source_adapter_scalar_count(adapter)
    if verify_sources:
        observed_adapter = training.inspect_dora_adapter_payload(
            adapter["root"], contract["base_model_root"]
        )
        require(observed_adapter == adapter, "Source adapter payload changed")
    config = value["model_config"]
    require(
        isinstance(config, Mapping)
        and config.get("backend", {}).get("type") == "hf"
        and config.get("backend", {}).get("hf", {}).get("attn_implementation")
        == contract["attention_implementation"]
        and config.get("model", {}).get("dtype") == contract["model_dtype"]
        and Path(config.get("model", {}).get("base_model", "")).resolve()
        == Path(contract["base_model_root"]).resolve()
        and Path(config.get("adapter", {}).get("path", "")).resolve()
        == Path(contract["adapter_root"]).resolve()
        and Path(config.get("embedding_delta", {}).get("path", "")).resolve()
        == Path(contract["embedding_root"]).resolve(),
        "model/source checkpoint config",
    )
    require(value["optimizer"] == training.DEFAULT_OPTIMIZER, "fresh AdamW recipe")
    require(
        value["scheduler"]
        == {"type": "cosine", "total_updates": 64, "warmup_updates": 0, "min_lr_ratio": 0.0},
        "cosine64 scheduler",
    )
    require(
        value["objective"]
        == {
            "ce_reduction": "sample_equal",
            "geometry_reduction": "sample_equal",
            "branch_weights": {"common": 0.5, "variable": 0.5},
        },
        "paired sample-equal objective",
    )
    hinge = value["validity_hinge"]
    require(
        isinstance(hinge, Mapping)
        and hinge.get("weight") == 0.01
        and hinge.get("margin") == 1 / 999
        and hinge.get("coordinate_token_ids") == contract["coordinate_token_ids"]
        and hinge.get("coordinate_bin_values") == list(range(1000)),
        "shared geometry recipe",
    )
    runtime = value["runtime"]
    require(
        isinstance(runtime, Mapping)
        and runtime.get("seed") == 19
        and runtime.get("world_size") == REQUIRED_WORLD_SIZE
        and runtime.get("effective_image_batch") == PRESENTATIONS_PER_UPDATE
        and runtime.get("branch_image_count") == BRANCH_IMAGE_COUNT
        and runtime.get("microbatch_size") == 2
        and runtime.get("activation_checkpointing") is True
        and runtime.get("fresh_optimizer") is True
        and runtime.get("gradient_clip_norm") == 1.0
        and runtime.get("eos_token_id") == contract["eos_token_id"],
        "Source256 runtime recipe",
    )
    expected = {
        "qualification": {
            "updates": 2,
            "checkpoint_steps": [2],
            "max_model_forwards": 128,
            "max_model_calls": 64,
        },
        "main": {
            "updates": 64,
            "checkpoint_steps": [16, 32, 64],
            "max_model_forwards": 4096,
            "max_model_calls": 2048,
        },
    }[value["mode"]]
    require(
        all(runtime.get(key) == expected_value for key, expected_value in expected.items()),
        "mode runtime bounds",
    )
    require(
        type(runtime.get("wall_seconds")) is int
        and 0 < runtime["wall_seconds"] <= (3600 if value["mode"] == "qualification" else 14_400),
        "runtime wall bound",
    )
    return dict(value)


def validate_training_manifest(
    value: Mapping[str, Any], *, verify_sources: bool = True
) -> dict[str, Any]:
    """Validate the original A/B recipe with its own producer identity."""
    return validate_training_recipe(
        value, manifest_schema=MANIFEST_SCHEMA, producer_path=Path(__file__),
        verify_sources=verify_sources,
    )


def resolve_update_presentations(
    records: Sequence[Mapping[str, Any]],
    update: Mapping[str, Any],
    *,
    arm: str,
) -> list[dict[str, Any]]:
    """Resolve one registered update; fallbacks remain in both denominators."""

    require(arm in ARMS, "Source256 arm")
    by_image = {int(record["image_id"]): record for record in records}
    require(len(by_image) == len(records), "duplicate route-record image")
    presentations: list[dict[str, Any]] = []
    for branch, key in (("common", "common_image_ids"), ("variable", "variable_image_ids")):
        ids = _image_ids(update.get(key), name=f"step {update.get('step')} {branch}")
        for slot, image_id in enumerate(ids):
            require(image_id in by_image, "schedule image absent route records")
            record = by_image[image_id]
            use_completion = (
                arm == "B"
                and branch == "variable"
                and bool(record["eligibility"]["fully_eligible"])
            )
            route = record["completion_route"] if use_completion else record["canonical_route"]
            require(route is not None, "selected route is absent")
            provenance = route["provenance"]
            presentations.append(
                {
                    "presentation_id": f"step-{int(update['step']):05d}:{branch}:{slot:02d}",
                    "step": int(update["step"]),
                    "branch": branch,
                    "slot": slot,
                    "image_id": image_id,
                    "route_kind": "completion" if use_completion else "canonical",
                    "fallback_reason": ""
                    if use_completion or branch == "common" or arm == "A"
                    else record["eligibility"]["fallback_reason"],
                    "target_owner_ids": list(provenance["suffix_owner_ids"]),
                    "target_owner_exposure_count": len(provenance["suffix_owner_ids"]),
                    "prefix_owner_count": len(provenance["prefix_owner_ids"]),
                    "prefix_token_count": len(provenance["prefix_token_ids"]),
                    "route": route,
                }
            )
    require(len(presentations) == PRESENTATIONS_PER_UPDATE, "64 update presentations")
    return presentations


def partition_update_presentations(
    presentations: Sequence[Mapping[str, Any]], *, rank: int, world_size: int
) -> list[dict[str, Any]]:
    """Give every four-GPU rank eight examples from each objective branch."""

    require(world_size == REQUIRED_WORLD_SIZE and 0 <= rank < world_size, "four-rank partition")
    require(len(presentations) == PRESENTATIONS_PER_UPDATE, "partition input size")
    selected: list[dict[str, Any]] = []
    for branch in BRANCHES:
        rows = [dict(row) for row in presentations if row.get("branch") == branch]
        require(len(rows) == BRANCH_IMAGE_COUNT, f"{branch} partition population")
        width = BRANCH_IMAGE_COUNT // world_size
        selected.extend(rows[rank * width : (rank + 1) * width])
    return selected


def objective_from_presentation_terms(
    ce_means: Sequence[torch.Tensor],
    raw_hinges: Sequence[torch.Tensor],
    branches: Sequence[str],
    *,
    geometry_weight: float = 0.01,
) -> torch.Tensor:
    """Return a local contribution whose cross-rank SUM is the paired objective."""

    require(len(ce_means) == len(raw_hinges) == len(branches) > 0, "presentation terms")
    require(all(branch in BRANCHES for branch in branches), "unknown objective branch")
    require(type(geometry_weight) in (int, float) and geometry_weight == 0.01, "geometry weight")
    zero = ce_means[0].new_zeros(())
    total = zero
    for branch in BRANCHES:
        branch_terms = [
            ce + geometry_weight * hinge
            for ce, hinge, observed in zip(ce_means, raw_hinges, branches, strict=True)
            if observed == branch
        ]
        total = total + 0.5 * sum(branch_terms, zero) / BRANCH_IMAGE_COUNT
    return total


def eligibility_gate(
    records: Sequence[Mapping[str, Any]], schedule: Mapping[str, Any]
) -> dict[str, Any]:
    """Compute the lead-owned pre-main-run gate without selecting a smaller cohort."""

    eligible = {
        int(record["image_id"])
        for record in records
        if bool(record["eligibility"]["fully_eligible"])
    }
    variable_presentations = [
        image_id
        for update in schedule["updates"]
        for image_id in update["variable_image_ids"]
    ]
    completion_presentations = sum(image_id in eligible for image_id in variable_presentations)
    total_presentations = UPDATE_COUNT * PRESENTATIONS_PER_UPDATE
    reasons = Counter(
        str(record["eligibility"]["fallback_reason"])
        for record in records
        if not bool(record["eligibility"]["fully_eligible"])
    )
    passed = (
        len(eligible) >= MIN_ELIGIBLE_IMAGES
        and completion_presentations >= MIN_COMPLETION_PRESENTATIONS
        and completion_presentations / total_presentations >= 0.125
    )
    return {
        "status": "passed" if passed else "stopped_below_eligibility_gate",
        "train_image_denominator": IMAGE_COUNT,
        "eligible_image_count": len(eligible),
        "minimum_eligible_images": MIN_ELIGIBLE_IMAGES,
        "total_presentations": total_presentations,
        "completion_presentations": completion_presentations,
        "minimum_completion_presentations": MIN_COMPLETION_PRESENTATIONS,
        "completion_presentation_fraction": completion_presentations / total_presentations,
        "fallback_reasons": dict(sorted(reasons.items())),
    }


def _dependency_bindings() -> dict[str, dict[str, Any]]:
    from probes.training_set_completion import source256_data

    root = Path(__file__).resolve().parents[2]
    return {
        "data_consumer": training.binding(Path(source256_data.__file__)),
        "training_helpers": training.binding(Path(training.__file__)),
        "artifact_primitives": training.binding(root / "probes/training_set_completion/artifacts.py"),
        "native_replay_helpers": training.binding(root / "src/qwen/native.py"),
        "batched_replay_helpers": training.binding(Path(replay.__file__)),
        "distributed_helpers": training.binding(Path(distributed.__file__)),
        "shared_geometry": training.binding(root / "src/losses/raw_axis_validity_hinge.py"),
    }


def _rank_receipt(
    *,
    receipt_schema: str,
    rank: int,
    local_rank: int,
    start_step: int,
    terminal_step: int,
    local_forwards: int,
    local_model_calls: int,
    started: float,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "schema": f"{receipt_schema}.rank.v1",
        "status": "completed",
        "rank": rank,
        "local_rank": local_rank,
        "host": socket.gethostname(),
        "device": str(device),
        "branches_per_update": {"common": 8, "variable": 8},
        "start_step": start_step,
        "terminal_step": terminal_step,
        "local_logical_forwards": local_forwards,
        "local_model_calls": local_model_calls,
        "elapsed_seconds": time.monotonic() - started,
        "resources": distributed.resource_receipt(device),
    }


def run_paired_training(
    manifest_path: Path, *, output: Path, producer_path: Path, receipt_schema: str,
    validate_manifest: Callable[..., dict[str, Any]],
    resolve_presentations: Callable[..., list[dict[str, Any]]],
    route_terms: Callable[..., tuple[torch.Tensor, torch.Tensor, int, dict[str, Any]]],
    dependency_bindings: Mapping[str, Mapping[str, Any]],
    enrich_update: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Execute the shared Source256 A/B and B-normalized four-rank recipe.

    This is not a generic trainer. Population, partition, geometry, SUM,
    scheduler and checkpoint semantics remain the fixed Source256 contract.
    Variant-owned validation, routes, CE terms and update evidence are explicit;
    no copied globals, module proxy or code-object transplantation is involved.
    """

    from probes.dora_owner_learning.runtime import (
        bind_source256_language_dora,
        load_policy,
    )
    from src.config.inference import InferConfig
    from src.qwen.checkpointing import (
        install_language_decoder_checkpointing,
        language_decoder_checkpointing_receipt as checkpointing_receipt,
    )

    manifest_path = manifest_path.resolve(strict=True)
    manifest = validate_manifest(json.loads(manifest_path.read_text()))
    prepared = validate_preparation(
        json.loads(Path(manifest["preparation"]["path"]).read_text())
    )
    hydrated_records = hydrate_bound_cases(prepared)
    dependencies = dict(dependency_bindings)
    require(manifest["sources"]["producer"] == training.binding(producer_path),
            "paired training execution producer differs from manifest")
    require(torch.cuda.is_available(), "Source256 distributed training requires CUDA")
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    require(
        world_size == REQUIRED_WORLD_SIZE
        and 0 <= rank < world_size
        and 0 <= local_rank < world_size,
        "launch Source256 with four torchrun ranks",
    )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        "nccl", timeout=timedelta(seconds=distributed.COLLECTIVE_TIMEOUT_SECONDS)
    )
    started = time.monotonic()
    phase = "output_setup"
    local_forwards = 0
    local_model_calls = 0
    checkpoints: list[dict[str, Any]] = []
    checkpoint_consensus: list[dict[str, Any]] = []
    old_alarm = signal.getsignal(signal.SIGALRM)

    def expired(*_: Any) -> None:
        raise TimeoutError("Source256 distributed training wall budget")

    try:
        distributed.coordinated_call(
            lambda: (
                require(not output.exists(), "attempt output already exists"),
                output.mkdir(parents=True),
            )
            if rank == 0
            else None,
            phase=phase,
        )
        dist.barrier()
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(math.ceil(manifest["runtime"]["wall_seconds"]))
        phase = "model_setup"

        def setup_model() -> tuple[
            Any,
            Mapping[str, Any],
            torch.nn.Module,
            tuple[tuple[str, torch.nn.Parameter], ...],
            tuple[tuple[str, torch.nn.Parameter], ...],
            Mapping[str, Any],
            torch.optim.Optimizer,
            torch.optim.lr_scheduler.CosineAnnealingLR,
        ]:
            random.seed(19)
            torch.manual_seed(19)
            torch.cuda.manual_seed(19)
            config = InferConfig.model_validate(manifest["model_config"])
            qwen, loaded = load_policy(config, device=device)
            model = qwen.model
            model.eval()
            scalar_count = source_adapter_scalar_count(manifest["source_adapter"])
            named, frozen = bind_source256_language_dora(
                model,
                expected_tensor_count=manifest["source_adapter"]["semantic_identity"][
                    "tensor_key_count"
                ],
                expected_scalar_count=scalar_count,
            )
            checkpointing = install_language_decoder_checkpointing(
                model, expected_layer_count=28
            )
            checkpointing.update(enabled=True, phase="train")
            optimizer = torch.optim.AdamW(
                [parameter for _, parameter in named],
                **{
                    **manifest["optimizer"],
                    "betas": tuple(manifest["optimizer"]["betas"]),
                },
            )
            require(not optimizer.state, "Source256 requires fresh AdamW")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=manifest["scheduler"]["total_updates"],
                eta_min=manifest["optimizer"]["lr"]
                * manifest["scheduler"]["min_lr_ratio"],
            )
            return qwen, loaded, model, named, frozen, checkpointing, optimizer, scheduler

        (
            qwen,
            loaded,
            model,
            named,
            frozen,
            checkpointing,
            optimizer,
            scheduler,
        ) = distributed.coordinated_call(setup_model, phase=phase)
        distributed.require_consensus(
            distributed.state_fingerprint(named, optimizer), label="initial state"
        )
        pad_token_id = qwen.tokenizer.pad_token_id
        require(type(pad_token_id) is int and pad_token_id >= 0, "runtime pad token")
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)

        for step in range(1, manifest["runtime"]["updates"] + 1):
            optimizer.zero_grad(set_to_none=True)
            update_started = time.monotonic()
            presentations = resolve_presentations(
                hydrated_records,
                prepared["schedule"]["updates"][step - 1],
                arm=manifest["arm"],
            )
            local = partition_update_presentations(
                presentations, rank=rank, world_size=world_size
            )
            local_cards: list[dict[str, Any]] = []
            local_metric = {branch: {"ce": 0.0, "hinge": 0.0, "active": 0} for branch in BRANCHES}
            local_step_calls = 0
            history_padding = 0
            prompt_padding = 0
            phase = f"update_{step}_forward_backward"

            def forward_backward() -> None:
                nonlocal local_forwards, local_model_calls, local_step_calls
                nonlocal history_padding, prompt_padding
                for start in range(0, len(local), manifest["runtime"]["microbatch_size"]):
                    selected = local[start : start + manifest["runtime"]["microbatch_size"]]
                    routes = [row["route"] for row in selected]
                    groups, preparation = replay.prepare_microbatches(
                        qwen,
                        manifest,
                        routes,
                        device=device,
                        microbatch_size=manifest["runtime"]["microbatch_size"],
                    )
                    require(len(groups) == 1, "one runtime microbatch per preparation")
                    group = groups[0]
                    logits_rows, padding = replay.batched_aligned_logits(
                        model,
                        group["inputs"],
                        group["routes"],
                        pad_token_id=pad_token_id,
                    )
                    ce_means: list[torch.Tensor] = []
                    hinges: list[torch.Tensor] = []
                    branches: list[str] = []
                    for logits, presentation in zip(logits_rows, selected, strict=True):
                        ce, hinge, active, card = route_terms(
                            logits, presentation["route"], manifest["validity_hinge"]
                        )
                        branch = presentation["branch"]
                        ce_means.append(ce)
                        hinges.append(hinge)
                        branches.append(branch)
                        local_metric[branch]["ce"] += float(ce.detach())
                        local_metric[branch]["hinge"] += float(hinge.detach())
                        local_metric[branch]["active"] += active
                        local_cards.append(
                            {
                                **{key: value for key, value in presentation.items() if key != "route"},
                                **card,
                                "eos_active": bool(
                                    presentation["route"]["ce_weights"][-1]
                                    and presentation["route"]["continuation_token_ids"][-1]
                                    == manifest["runtime"]["eos_token_id"]
                                ),
                            }
                        )
                    objective_from_presentation_terms(
                        ce_means,
                        hinges,
                        branches,
                        geometry_weight=manifest["validity_hinge"]["weight"],
                    ).backward()
                    local_forwards += len(selected)
                    local_model_calls += 1
                    local_step_calls += 1
                    history_padding += padding["history_padding_tokens"]
                    prompt_padding += preparation["prompt_padding_tokens"]
                require(
                    all(
                        parameter.grad is not None
                        and bool(torch.isfinite(parameter.grad).all())
                        for _, parameter in named
                    ),
                    "missing/nonfinite local DoRA gradients",
                )
                require(
                    all(parameter.grad is None for _, parameter in frozen),
                    "frozen parameter received gradient",
                )

            distributed.coordinated_call(forward_backward, phase=phase)
            phase = f"update_{step}_gradient_sum"
            distributed.sum_gradients_(named)
            raw_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [parameter for _, parameter in named],
                    manifest["runtime"]["gradient_clip_norm"],
                    error_if_nonfinite=True,
                    foreach=False,
                )
            )
            applied_lr = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            metric = torch.tensor(
                [
                    local_metric["common"]["ce"],
                    local_metric["common"]["hinge"],
                    local_metric["common"]["active"],
                    local_metric["variable"]["ce"],
                    local_metric["variable"]["hinge"],
                    local_metric["variable"]["active"],
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(metric, op=dist.ReduceOp.SUM)
            gathered_cards = distributed.gather_objects(local_cards)
            cards = [card for rows in gathered_cards for card in rows]
            order = {
                row["presentation_id"]: index for index, row in enumerate(presentations)
            }
            cards.sort(key=lambda row: order[row["presentation_id"]])
            require(
                [row["presentation_id"] for row in cards]
                == [row["presentation_id"] for row in presentations],
                "gathered presentation order",
            )
            require(len(cards) == 64 and all(row["eos_active"] for row in cards), "EOS/full batch")
            branch_metrics = {
                branch: {
                    "sample_equal_ce": float(metric[offset].item() / 32),
                    "sample_equal_raw_geometry": float(metric[offset + 1].item() / 32),
                    "active_tokens": int(metric[offset + 2].item()),
                }
                for branch, offset in (("common", 0), ("variable", 3))
            }
            objective_total = 0.5 * sum(
                values["sample_equal_ce"]
                + manifest["validity_hinge"]["weight"]
                * values["sample_equal_raw_geometry"]
                for values in branch_metrics.values()
            )
            rank_timings = distributed.gather_objects(
                {
                    "rank": rank,
                    "local_presentations": len(local),
                    "local_model_calls": local_step_calls,
                    "history_padding_tokens": history_padding,
                    "prompt_padding_tokens": prompt_padding,
                    "elapsed_seconds": time.monotonic() - update_started,
                    "resources": distributed.resource_receipt(device),
                }
            )
            update = {
                "schema": f"{receipt_schema}.update.v1",
                "step": step,
                "arm": manifest["arm"],
                "image_presentations": 64,
                "branch_metrics": branch_metrics,
                "objective_total": objective_total,
                "applied_lr": applied_lr,
                "next_lr": float(optimizer.param_groups[0]["lr"]),
                "gradient_norm_before_clip": raw_norm,
                "presentations": cards,
                "logical_model_forwards": step * 64,
                "model_calls": step * 32,
                "distributed": {
                    "world_size": world_size,
                    "rank_presentations": [16, 16, 16, 16],
                    "rank_timings": rank_timings,
                    "normalization": {
                        "common": "0.5 * SUM(sample means) / 32",
                        "variable": "0.5 * SUM(sample means) / 32",
                        "geometry_weight": 0.01,
                        "gradient_collective": "SUM",
                        "post_collective_divisor": 1,
                    },
                },
            }
            distributed.coordinated_call(
                lambda: training.publish(
                    output / "updates" / f"step-{step:05d}.json",
                    update if enrich_update is None else enrich_update(update),
                )
                if rank == 0
                else None,
                phase=f"update_{step}_receipt",
            )
            if step in manifest["runtime"]["checkpoint_steps"]:
                phase = f"checkpoint_{step}"
                state = distributed.state_fingerprint(named, optimizer)
                states = distributed.require_consensus(
                    state, label=f"checkpoint {step}"
                )
                require(state["optimizer_steps"] == [step], "optimizer counter")
                holder: dict[str, Any] = {}

                def save_checkpoint() -> None:
                    if rank == 0:
                        holder["receipt"] = training._checkpoint(
                            output,
                            manifest_path=manifest_path,
                            manifest=manifest,
                            model=model,
                            optimizer=optimizer,
                            named=named,
                            step=step,
                        )
                        scheduler_path = (
                            output / "checkpoints" / f"step-{step:05d}" / "scheduler.pt"
                        )
                        torch.save(scheduler.state_dict(), scheduler_path)
                        holder["receipt"]["scheduler"] = training.binding(scheduler_path)

                distributed.coordinated_call(
                    save_checkpoint, phase=f"checkpoint_{step}_save"
                )
                dist.barrier()
                consensus = {"step": step, "state": state, "rank_count": len(states)}
                distributed.coordinated_call(
                    lambda: training.publish(
                        output / "checkpoints" / f"step-{step:05d}" / "consensus.json",
                        consensus,
                    )
                    if rank == 0
                    else None,
                    phase=f"checkpoint_{step}_consensus",
                )
                checkpoint_consensus.append(consensus)
                if rank == 0:
                    checkpoints.append(holder["receipt"])

        receipt = _rank_receipt(
            receipt_schema=receipt_schema,
            rank=rank,
            local_rank=local_rank,
            start_step=0,
            terminal_step=manifest["runtime"]["updates"],
            local_forwards=local_forwards,
            local_model_calls=local_model_calls,
            started=started,
            device=device,
        )
        receipts = distributed.gather_objects(receipt)
        distributed.coordinated_call(
            lambda: training.publish(output / "ranks" / f"rank-{rank:03d}.json", receipt),
            phase="rank_receipt",
        )
        terminal = {
            "schema": f"{receipt_schema}.terminal.v1",
            "status": "completed",
            "arm": manifest["arm"],
            "mode": manifest["mode"],
            "manifest": training.binding(manifest_path),
            "preparation": manifest["preparation"],
            "loaded_model": loaded,
            "optimizer_mode": "fresh",
            "trainable_surface": training._layout(named),
            "updates": manifest["runtime"]["updates"],
            "logical_model_forwards": manifest["runtime"]["max_model_forwards"],
            "model_calls": manifest["runtime"]["max_model_calls"],
            "checkpoints": checkpoints,
            "activation_checkpointing": checkpointing_receipt(model, checkpointing),
            "elapsed_seconds": time.monotonic() - started,
            "distributed": {
                "backend": dist.get_backend(),
                "world_size": world_size,
                "rank_receipts": receipts,
                "checkpoint_consensus": checkpoint_consensus,
                "normalization": {
                    "branches": {"common": 0.5, "variable": 0.5},
                    "branch_denominator": 32,
                    "gradient_collective": "SUM",
                    "post_collective_divisor": 1,
                },
                "source_bindings": {
                    "training_backend": training.binding(producer_path),
                    **dependencies,
                },
            },
        }
        distributed.coordinated_call(
            lambda: training.publish(output / "terminal.json", terminal)
            if rank == 0
            else None,
            phase="terminal",
        )
        dist.barrier()
        return terminal if rank == 0 else None
    except Exception as exc:
        if rank == 0 and output.exists() and not (output / "terminal.json").exists():
            try:
                training.publish(
                    output / "terminal.json",
                    {
                        "schema": f"{receipt_schema}.terminal.v1",
                        "status": "failed",
                        "manifest": training.binding(manifest_path),
                        "phase": phase,
                        "local_logical_forwards": local_forwards,
                        "local_model_calls": local_model_calls,
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_seconds": time.monotonic() - started,
                        "distributed": {"rank": rank, "world_size": world_size},
                    },
                )
            except Exception:
                pass
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        if dist.is_initialized():
            dist.destroy_process_group()


def run(manifest_path: Path, *, output: Path) -> dict[str, Any] | None:
    """Original Source256 A/B entry; variant choices are bound at the call site."""
    return run_paired_training(
        manifest_path, output=output, producer_path=Path(__file__),
        receipt_schema=SCHEMA, validate_manifest=validate_training_manifest,
        resolve_presentations=resolve_update_presentations,
        route_terms=replay.route_terms, dependency_bindings=_dependency_bindings(),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.manifest, output=args.output)
    if result is not None:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
