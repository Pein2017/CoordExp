"""Source256 ``B-normalized`` training contracts.

This successor deliberately reuses the admitted Source256 route producer and
distributed runner.  Its only scientific change is the CE denominator for an
eligible completion route:

``sum(active suffix NLL, including EOS) / same-image canonical target length``.

Canonical routes and canonical fallbacks retain their active-token mean.  The
geometry hinge, its denominator and its weight are delegated to the unchanged
Source256 implementation.  The adapter around the runner exists so the old
producer remains byte-for-byte bound to its historical receipts.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import statistics
from types import FunctionType
from typing import Any, Mapping, Sequence

import torch

from probes.training_set_completion import source256_training as predecessor
from probes.training_set_completion import training


UNIT_ID = "2026-09-16-source256-completion-ce-normalization"
ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-16-source256-completion-ce-normalization"
)
SHARED_PREPARATION = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-16-source256-fixed-prefix-completion/preparation/"
    "source256-admitted-v1/preparation.json"
)
SHARED_PREPARATION_SHA256 = (
    "9b94baeb0699e479413483cba5ee6fb4b4d98c062aebb0ba462ae61a871cf242"
)
SOURCE_CONFIG = Path(__file__).resolve().parents[1] / "dora_owner_learning/configs/source256.yaml"
PREDECESSOR_PATH = Path(predecessor.__file__).resolve()
_PREDECESSOR_VALIDATE_MANIFEST = predecessor.validate_training_manifest
_PREDECESSOR_RESOLVE_PRESENTATIONS = predecessor.resolve_update_presentations
SCHEMA = "training_set_completion.source256_normalized_training.v1"
MANIFEST_SCHEMA = "training_set_completion.source256_normalized_training_manifest.v1"
PREPARATION_SCHEMA = "training_set_completion.source256_normalized_preparation.v1"
RELEASE_SCHEMA = "training_set_completion.source256_normalized_release.v1"
TRIAL_SCHEMA = "training_set_completion.source256_normalized_trial.v1"
CPU_INVARIANT_SCHEMA = "training_set_completion.source256_normalized_cpu_invariants.v1"
QUALIFICATION_SCHEMA = "training_set_completion.source256_normalized_training.v1.qualification"
B_NORMALIZED = "B-normalized"
ARMS = (B_NORMALIZED,)
QUALIFICATION_NORMALIZATION = {
    "completion_denominator": "same_image_canonical_full_target_token_count_including_eos",
    "binary_ce_masks": True,
    "no_geometry_change": True,
    "geometry_weight": 0.01,
}
BRANCHES = predecessor.BRANCHES
IMAGE_COUNT = predecessor.IMAGE_COUNT
UPDATE_COUNT = predecessor.UPDATE_COUNT
BRANCH_IMAGE_COUNT = predecessor.BRANCH_IMAGE_COUNT
PRESENTATIONS_PER_UPDATE = predecessor.PRESENTATIONS_PER_UPDATE
REQUIRED_WORLD_SIZE = predecessor.REQUIRED_WORLD_SIZE
SOURCE_ADAPTER_FINGERPRINT = predecessor.SOURCE_ADAPTER_FINGERPRINT
SOURCE_ADAPTER_SCALAR_COUNT = predecessor.SOURCE_ADAPTER_SCALAR_COUNT

# These are read-only aliases to the established producer/runtime helpers.
_batched_aligned_logits = predecessor._batched_aligned_logits
_prepare_microbatches = predecessor._prepare_microbatches
distributed = predecessor.distributed
validate_schedule = predecessor.validate_schedule
validate_route_record = predecessor.validate_route_record
validate_records = predecessor.validate_records
validate_preparation = predecessor.validate_preparation
hydrate_bound_cases = predecessor.hydrate_bound_cases
source_adapter_scalar_count = predecessor.source_adapter_scalar_count
eligibility_gate = predecessor.eligibility_gate
partition_update_presentations = predecessor.partition_update_presentations

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _route_kind(route: Mapping[str, Any]) -> str:
    provenance = route.get("provenance")
    require(isinstance(provenance, Mapping), "route provenance for CE normalization")
    kind = provenance.get("route_kind")
    require(kind in ("canonical", "fixed_source_prefix_completion"), "route kind")
    return str(kind)


def _binary_mask(route: Mapping[str, Any]) -> list[int]:
    weights = route.get("ce_weights")
    require(
        isinstance(weights, list)
        and all(type(weight) is int and weight in (0, 1) for weight in weights),
        "CE masks must contain binary 0/1 values",
    )
    return list(weights)


def canonical_full_target_token_count(
    route: Mapping[str, Any], *, explicit: int | None = None
) -> int:
    """Return the frozen same-image canonical target denominator.

    Completion routes do not guess this denominator from their suffix.  The
    route resolver attaches the count from the paired canonical route, and
    direct callers may pass it explicitly for CPU accounting tests.
    """

    kind = _route_kind(route)
    if kind == "canonical":
        count = len(route.get("continuation_token_ids", ()))
    else:
        count = explicit
        if count is None:
            count = route.get("_canonical_full_target_token_count")
        if count is None:
            count = route.get("canonical_full_target_token_count")
    require(type(count) is int and count > 0, "canonical full target token denominator")
    require(
        isinstance(route.get("continuation_token_ids"), list)
        and route["continuation_token_ids"],
        "route continuation for CE normalization",
    )
    return count


_CANONICAL_FULL_TARGET_COUNT = canonical_full_target_token_count


def _annotate_route(
    route: Mapping[str, Any], *, canonical_count: int
) -> dict[str, Any]:
    annotated = dict(route)
    annotated["_canonical_full_target_token_count"] = canonical_count
    return annotated


def resolve_update_presentations(
    records: Sequence[Mapping[str, Any]],
    update: Mapping[str, Any],
    *,
    arm: str,
) -> list[dict[str, Any]]:
    """Resolve the single B-normalized arm without changing route selection."""

    require(arm == B_NORMALIZED, "B-normalized arm")
    by_image = {int(record["image_id"]): record for record in records}
    require(len(by_image) == len(records), "duplicate route-record image")
    # The predecessor resolver owns common/variable schedule semantics.  It is
    # called with its historical B spelling solely to select the same routes.
    selected = _PREDECESSOR_RESOLVE_PRESENTATIONS(records, update, arm="B")
    checked: list[dict[str, Any]] = []
    for presentation in selected:
        image_id = int(presentation["image_id"])
        canonical = by_image[image_id]["canonical_route"]
        canonical_count = len(canonical["continuation_token_ids"])
        require(canonical_count > 0, "canonical route has no target tokens")
        row = dict(presentation)
        row["canonical_full_target_token_count"] = canonical_count
        row["route"] = _annotate_route(
            presentation["route"], canonical_count=canonical_count
        )
        checked.append(row)
    require(len(checked) == PRESENTATIONS_PER_UPDATE, "64 update presentations")
    return checked


def _route_terms(
    logits: torch.Tensor,
    route: Mapping[str, Any],
    hinge: Mapping[str, Any],
    *,
    canonical_full_target_token_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, Any]]:
    """Compute route CE and unchanged geometry with explicit accounting.

    ``training.masked_ce_loss`` remains the scoring owner.  Multiplying its
    graph-connected active-token mean by ``active/new_denominator`` preserves
    the exact NLL numerator and autograd through the complete input history.
    """

    weights = _binary_mask(route)
    targets = torch.tensor(
        route["continuation_token_ids"], dtype=torch.long, device=logits.device
    )
    old_ce, metrics = training.masked_ce_loss(logits, targets, weights)
    active = int(metrics["active_tokens"])
    require(active > 0, "CE has no active suffix tokens")
    explicit_denominator = canonical_full_target_token_count
    # Keep the call explicit so the denominator cannot accidentally be derived
    # from a batch, a generated length, or the active suffix.
    new_denominator = _CANONICAL_FULL_TARGET_COUNT(
        route, explicit=explicit_denominator
    )
    normalized_ce = old_ce * (float(active) / float(new_denominator))
    raw_hinge = training.raw_axis_validity_hinge(
        logits,
        route["trusted_boxes"],
        coordinate_token_ids=hinge["coordinate_token_ids"],
        coordinate_bin_values=hinge["coordinate_bin_values"],
        margin=hinge["margin"],
    )
    require(
        bool(torch.isfinite(normalized_ce)) and bool(torch.isfinite(raw_hinge)),
        "nonfinite normalized route terms",
    )
    scale = float(active) / float(new_denominator)
    card = {
        "route_id": route["route_id"],
        "active_tokens": active,
        "masked_nll_sum": metrics["masked_nll_sum"],
        "ce_numerator": metrics["masked_nll_sum"],
        "old_ce_denominator_active_tokens": active,
        "new_ce_denominator_canonical_full_target_tokens": new_denominator,
        "canonical_full_target_tokens": new_denominator,
        "active_token_mean_ce": float(old_ce.detach()),
        "normalized_canonical_target_mean_ce": float(normalized_ce.detach()),
        "ce_scale_active_over_canonical": scale,
        "ce_normalization": "completion_canonical_full_target_tokens",
        "geometry_scale": 1.0,
        "raw_axis_validity_hinge": float(raw_hinge.detach()),
    }
    return normalized_ce, raw_hinge, active, card


def objective_from_presentation_terms(
    ce_means: Sequence[torch.Tensor],
    raw_hinges: Sequence[torch.Tensor],
    branches: Sequence[str],
    *,
    geometry_weight: float = 0.01,
) -> torch.Tensor:
    """Delegate unchanged 50/50 branch and rank compensation."""

    return predecessor.objective_from_presentation_terms(
        ce_means, raw_hinges, branches, geometry_weight=geometry_weight
    )


def summarize_normalization(cards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize per-presentation numerator/denominators and CE scale values."""

    result: dict[str, Any] = {}
    for branch in BRANCHES:
        rows = [row for row in cards if row.get("branch") == branch]
        require(rows, f"normalization cards for {branch}")
        scales = [float(row["ce_scale_active_over_canonical"]) for row in rows]
        old_denominators = [int(row["old_ce_denominator_active_tokens"]) for row in rows]
        new_denominators = [
            int(row["new_ce_denominator_canonical_full_target_tokens"]) for row in rows
        ]
        numerators = [float(row["ce_numerator"]) for row in rows]
        sorted_scales = sorted(scales)
        result[branch] = {
            "presentations": len(rows),
            "ce_numerator_sum": sum(numerators),
            "old_active_token_denominator_sum": sum(old_denominators),
            "new_canonical_full_target_denominator_sum": sum(new_denominators),
            "scale_distribution": {
                "count": len(scales),
                "values": scales,
                "min": min(scales),
                "max": max(scales),
                "mean": statistics.fmean(scales),
                "median": statistics.median(sorted_scales),
            },
        }
    return result


def _cpu_sensitivity_route(kind: str) -> dict[str, Any]:
    canonical_suffix = [40, 41, 42, 43, 44, 45, 46, 47, 48, 99]
    if kind == "canonical":
        prefix: list[int] = []
        continuation = canonical_suffix
        weights = [1] * len(continuation)
        route_kind = "canonical"
        prefix_owners: list[str] = []
        suffix_owners = ["probe:owner-0", "probe:owner-1"]
    else:
        prefix = [30, 31]
        continuation = prefix + [40, 41, 42, 43, 99]
        weights = [0, 0, 1, 1, 1, 1, 1]
        route_kind = "fixed_source_prefix_completion"
        prefix_owners = ["probe:owner-0"]
        suffix_owners = ["probe:owner-1"]
    return {
        "route_id": f"cpu-sensitivity-{kind}",
        "continuation_token_ids": continuation,
        "ce_weights": weights,
        "trusted_boxes": [],
        "provenance": {
            "route_kind": route_kind,
            "bank_owner_ids": prefix_owners + suffix_owners,
            "prefix_owner_ids": prefix_owners,
            "suffix_owner_ids": suffix_owners,
            "prefix_token_ids": prefix,
            "suffix_token_ids": continuation[len(prefix) :],
            "source_greedy_generated_token_ids_sha256": "0" * 64,
        },
    }


def build_cpu_invariant_receipt(*, preparation_path: Path = SHARED_PREPARATION) -> dict[str, Any]:
    """Run deterministic CPU sensitivity checks and return their receipt."""

    preparation_path = preparation_path.resolve(strict=True)
    require(preparation_path == SHARED_PREPARATION.resolve(), "CPU invariant preparation path")
    shared = training.binding(preparation_path)
    require(shared["sha256"] == SHARED_PREPARATION_SHA256, "CPU invariant preparation digest")
    preparation_value = json.loads(preparation_path.read_text())
    accounting = preparation_accounting(validate_preparation(preparation_value)["preparation"])

    canonical = _cpu_sensitivity_route("canonical")
    completion = _cpu_sensitivity_route("completion")
    canonical_logits = torch.zeros((10, 128), dtype=torch.float64, requires_grad=True)
    completion_logits = torch.zeros((7, 128), dtype=torch.float64, requires_grad=True)
    old_canonical, old_metrics = training.masked_ce_loss(
        canonical_logits,
        torch.tensor(canonical["continuation_token_ids"]),
        canonical["ce_weights"],
    )
    new_canonical, canonical_hinge, canonical_active, canonical_card = _route_terms(
        canonical_logits, canonical, {"coordinate_token_ids": list(range(128)), "coordinate_bin_values": list(range(128)), "margin": 1 / 999}
    )
    new_completion, completion_hinge, completion_active, completion_card = _route_terms(
        completion_logits,
        completion,
        {"coordinate_token_ids": list(range(128)), "coordinate_bin_values": list(range(128)), "margin": 1 / 999},
        canonical_full_target_token_count=len(canonical["continuation_token_ids"]),
    )
    canonical_card = {"branch": "common", **canonical_card}
    completion_card = {"branch": "variable", **completion_card}
    summary = summarize_normalization([canonical_card, completion_card])
    canonical_delta = float((new_canonical - old_canonical).abs().detach())
    require(canonical_delta == 0.0, "CPU canonical CE changed")
    require(canonical_active == old_metrics["active_tokens"] == 10, "CPU canonical denominator")
    require(completion_active == 5, "CPU completion active suffix")
    require(completion_card["new_ce_denominator_canonical_full_target_tokens"] == 10, "CPU completion denominator")
    require(completion_card["ce_scale_active_over_canonical"] == 0.5, "CPU completion scale")
    require(completion["continuation_token_ids"][-1] == 99 and completion["ce_weights"][-1] == 1, "CPU completion EOS")

    context_signal = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    context_logits = context_signal * torch.arange(128, dtype=torch.float64).view(1, -1).expand(7, -1)
    context_ce, _, _, _ = _route_terms(
        context_logits,
        completion,
        {"coordinate_token_ids": list(range(128)), "coordinate_bin_values": list(range(128)), "margin": 1 / 999},
        canonical_full_target_token_count=10,
    )
    context_ce.backward()
    context_gradient = float(context_signal.grad.detach())
    require(context_gradient != 0.0 and bool(torch.isfinite(context_signal.grad)), "CPU context gradient")

    parameter = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    common = [parameter * float(index + 1) for index in range(32)]
    variable = [parameter.square() * float(index + 1) for index in range(32)]
    common_hinges = [parameter.square() for _ in common]
    variable_hinges = [2 * parameter.square() for _ in variable]
    local = parameter.new_zeros(())
    for rank in range(REQUIRED_WORLD_SIZE):
        start, stop = rank * 8, (rank + 1) * 8
        local = local + objective_from_presentation_terms(
            common[start:stop] + variable[start:stop],
            common_hinges[start:stop] + variable_hinges[start:stop],
            ["common"] * 8 + ["variable"] * 8,
        )
    expected = 0.5 * sum(common) / 32 + 0.5 * sum(variable) / 32
    expected = expected + 0.01 * (0.5 * sum(common_hinges) / 32 + 0.5 * sum(variable_hinges) / 32)
    local_gradient = float(torch.autograd.grad(local, parameter, retain_graph=True)[0])
    expected_gradient = float(torch.autograd.grad(expected, parameter)[0])
    rank_loss_delta = float((local - expected).abs().detach())
    rank_gradient_delta = abs(local_gradient - expected_gradient)
    require(rank_loss_delta <= 1e-12, "CPU rank compensation loss")
    require(rank_gradient_delta <= 1e-12, "CPU rank compensation gradient")

    return {
        "schema": CPU_INVARIANT_SCHEMA,
        "status": "passed_cpu_invariants",
        "unit_id": UNIT_ID,
        "preparation": shared,
        "source_bindings": _dependency_bindings(),
        "normalization": {
            "completion_formula": "ce_numerator / same_image_canonical_full_target_token_count_including_eos",
            "canonical_and_fallback_formula": "ce_numerator / active_target_tokens_including_eos",
            "binary_ce_masks": True,
            "geometry_scale": 1.0,
            "geometry_weight": 0.01,
            "branch_weights": {"common": 0.5, "variable": 0.5},
            "rank_gradient_collective": "SUM with unchanged post-collective divisor 1",
        },
        "preparation_accounting": accounting,
        "sensitivity": {
            "canonical_ce_absolute_delta": canonical_delta,
            "completion_old_active_tokens": completion_card["old_ce_denominator_active_tokens"],
            "completion_new_canonical_full_target_tokens": completion_card["new_ce_denominator_canonical_full_target_tokens"],
            "completion_scale_active_over_canonical": completion_card["ce_scale_active_over_canonical"],
            "completion_eos_active": True,
            "context_gradient": context_gradient,
            "geometry": {
                "canonical_raw_hinge": float(canonical_hinge.detach()),
                "completion_raw_hinge": float(completion_hinge.detach()),
                "canonical_geometry_scale": canonical_card["geometry_scale"],
                "completion_geometry_scale": completion_card["geometry_scale"],
            },
            "rank_compensation": {
                "loss_absolute_delta": rank_loss_delta,
                "gradient_absolute_delta": rank_gradient_delta,
                "world_size": REQUIRED_WORLD_SIZE,
                "rank_branch_population": {"common": 8, "variable": 8},
            },
        },
        "numerator_denominator_scale_summary": summary,
        "gpu_launch": "held_for_lead_release",
    }


def _enrich_update(value: Any) -> Any:
    if not isinstance(value, Mapping) or value.get("schema") != f"{SCHEMA}.update.v1":
        return value
    update = copy.deepcopy(dict(value))
    cards = update.get("presentations")
    require(isinstance(cards, list) and cards, "update presentation cards")
    summary = summarize_normalization(cards)
    update["ce_normalization"] = {
        "kind": "completion_only_canonical_full_target_denominator",
        "completion_formula": "masked_nll_sum / same_image_canonical_full_target_tokens",
        "canonical_and_fallback_formula": "masked_nll_sum / active_target_tokens",
        "geometry_scale": 1.0,
        "geometry_weight": 0.01,
        "branches": summary,
    }
    branch_metrics = update.get("branch_metrics")
    require(isinstance(branch_metrics, Mapping), "update branch metrics")
    for branch in BRANCHES:
        metrics = dict(branch_metrics[branch])
        metrics["ce_numerator_sum"] = summary[branch]["ce_numerator_sum"]
        metrics["old_active_token_denominator_sum"] = summary[branch][
            "old_active_token_denominator_sum"
        ]
        metrics["new_canonical_full_target_denominator_sum"] = summary[branch][
            "new_canonical_full_target_denominator_sum"
        ]
        metrics["ce_scale_distribution"] = summary[branch]["scale_distribution"]
        branch_metrics[branch] = metrics
    normalization = dict(update.get("distributed", {}).get("normalization", {}))
    normalization["ce_denominator"] = (
        "completion: same-image canonical full target token count including EOS; "
        "canonical/fallback: active target token count"
    )
    normalization["geometry"] = "unchanged raw hinge, weight 0.01, branch denominator 32"
    update.setdefault("distributed", {})["normalization"] = normalization
    return update


def _dependency_bindings() -> dict[str, dict[str, Any]]:
    from probes.training_set_completion import source256_data

    root = Path(__file__).resolve().parents[2]
    return {
        "normalized_training": training.binding(Path(__file__)),
        "data_consumer": training.binding(Path(source256_data.__file__)),
        "training_helpers": training.binding(Path(training.__file__)),
        "batched_replay_helpers": training.binding(PREDECESSOR_PATH),
        "distributed_helpers": training.binding(Path(distributed.__file__)),
        "shared_geometry": training.binding(root / "src/losses/raw_axis_validity_hinge.py"),
    }


def _shadow_for_predecessor_validation(value: Mapping[str, Any]) -> dict[str, Any]:
    shadow = copy.deepcopy(dict(value))
    shadow.pop("ce_normalization", None)
    shadow["arm"] = "B"
    shadow["objective"] = {
        "ce_reduction": "sample_equal",
        "geometry_reduction": "sample_equal",
        "branch_weights": {"common": 0.5, "variable": 0.5},
    }
    shadow["schema"] = MANIFEST_SCHEMA
    shadow["content_sha256"] = training.digest(
        {key: item for key, item in shadow.items() if key != "content_sha256"}
    )
    return shadow


def _validate_with_predecessor_contract(
    shadow: Mapping[str, Any], *, verify_sources: bool
) -> dict[str, Any]:
    """Run the old structural validator through private globals."""

    validator_globals = _PREDECESSOR_VALIDATE_MANIFEST.__globals__.copy()
    validator_globals.update(
        {
            "MANIFEST_SCHEMA": MANIFEST_SCHEMA,
            "__file__": str(Path(__file__).resolve()),
        }
    )
    validator = FunctionType(
        _PREDECESSOR_VALIDATE_MANIFEST.__code__,
        validator_globals,
        name="source256_normalized_predecessor_manifest_validator",
    )
    return validator(shadow, verify_sources=verify_sources)


def validate_training_manifest(
    value: Mapping[str, Any], *, verify_sources: bool = True
) -> dict[str, Any]:
    """Validate the successor contract and all unchanged Source256 invariants."""

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
        "ce_normalization",
        "validity_hinge",
        "runtime",
        "content_sha256",
    }
    require(set(value) == required, "normalized training manifest fields")
    require(value["schema"] == MANIFEST_SCHEMA, "normalized manifest schema")
    require(
        value["content_sha256"]
        == training.digest(
            {key: item for key, item in value.items() if key != "content_sha256"}
        ),
        "normalized training manifest content digest",
    )
    require(value["arm"] == B_NORMALIZED, "normalized manifest arm")
    require(
        value["objective"]
        == {
            "ce_reduction": "sample_equal",
            "geometry_reduction": "sample_equal",
            "branch_weights": {"common": 0.5, "variable": 0.5},
        },
        "normalized objective must preserve branch/sample reduction",
    )
    require(
        value["ce_normalization"]
        == {
            "completion_only": True,
            "completion_denominator": "same_image_canonical_full_target_tokens_including_eos",
            "canonical_and_fallback_denominator": "active_target_tokens_including_eos",
            "mask_values": [0, 1],
            "geometry_scale": 1.0,
        },
        "normalized CE contract",
    )
    shadow = _shadow_for_predecessor_validation(value)
    _validate_with_predecessor_contract(shadow, verify_sources=verify_sources)
    return dict(value)


def preparation_accounting(value: Mapping[str, Any]) -> dict[str, Any]:
    """Build deterministic CPU accounting from the shared final preparation."""

    routes = value.get("routes")
    require(isinstance(routes, list) and len(routes) == IMAGE_COUNT, "256 preparation routes")
    eligible = [
        row
        for row in routes
        if bool(row.get("eligibility", {}).get("fully_eligible"))
    ]
    ratios: list[float] = []
    for row in eligible:
        canonical = row["canonical_route"]
        completion = row.get("completion_route")
        require(completion is not None, "eligible route completion")
        active = sum(completion["ce_weights"])
        denominator = len(canonical["continuation_token_ids"])
        require(active > 0 and denominator > 0, "positive completion accounting lengths")
        ratios.append(active / denominator)
    require(len(eligible) == 143, "frozen 143 eligible images")
    return {
        "train_images": IMAGE_COUNT,
        "eligible_images": len(eligible),
        "completion_presentations": 1144,
        "total_presentations": 4096,
        "completion_fraction": 1144 / 4096,
        "ratio_active_suffix_over_canonical_full_target": {
            "count": len(ratios),
            "min": min(ratios),
            "max": max(ratios),
            "mean": statistics.fmean(ratios),
            "median": statistics.median(ratios),
        },
        "old_ce_amplification_over_new": {
            "median": 1 / statistics.median(ratios),
            "max": 1 / min(ratios),
        },
        "mask_values": [0, 1],
        "eos_included": True,
        "geometry_weight": 0.01,
        "geometry_denominator": "unchanged per-branch sample denominator 32",
    }


def build_preparation_receipt(
    preparation_path: Path, *, producer_path: Path | None = None
) -> dict[str, Any]:
    """Return a pointer receipt while retaining the exact shared preparation bytes."""

    preparation_path = preparation_path.resolve(strict=True)
    require(
        preparation_path == SHARED_PREPARATION.resolve(),
        "normalized trial must consume the shared final preparation path",
    )
    shared_binding = training.binding(preparation_path)
    require(shared_binding["sha256"] == SHARED_PREPARATION_SHA256, "shared preparation SHA256")
    value = json.loads(preparation_path.read_text())
    checked = validate_preparation(value)
    receipt: dict[str, Any] = {
        "schema": PREPARATION_SCHEMA,
        "status": "candidate_ready",
        "unit_id": UNIT_ID,
        "shared_preparation": shared_binding,
        "accounting": preparation_accounting(checked["preparation"]),
        "normalization": {
            "completion_formula": "sum(active suffix NLL including EOS) / same-image canonical full target tokens including EOS",
            "canonical_and_fallback": "unchanged active-token mean",
            "geometry": "unchanged contribution, denominator and weight 0.01",
        },
        "producer": training.binding(producer_path or Path(__file__)),
    }
    receipt["content_sha256"] = training.digest(
        {key: item for key, item in receipt.items() if key != "content_sha256"}
    )
    return receipt


def _model_config(*, output: Path) -> dict[str, Any]:
    from src.config.inference import load_research_infer_config

    resolved = load_research_infer_config(SOURCE_CONFIG)
    config = copy.deepcopy(resolved.config_dict)
    config["run"].update(
        name="source256-completion-ce-normalization-B",
        artifact_root=str(output.resolve()),
        output_dir=None,
        collision_policy="fail",
    )
    return config


def build_training_manifest(
    *,
    preparation_path: Path,
    mode: str,
    output: Path,
) -> dict[str, Any]:
    require(mode in ("qualification", "main"), "normalized training mode")
    checked = validate_preparation(json.loads(Path(preparation_path).read_text()))
    contract = checked["runtime_contract"]
    source_adapter = training.inspect_dora_adapter_payload(
        contract["adapter_root"], contract["base_model_root"]
    )
    mode_runtime = {
        "qualification": {
            "updates": 2,
            "checkpoint_steps": [2],
            "max_model_forwards": 128,
            "max_model_calls": 64,
            "wall_seconds": 3600,
        },
        "main": {
            "updates": 64,
            "checkpoint_steps": [16, 32, 64],
            "max_model_forwards": 4096,
            "max_model_calls": 2048,
            "wall_seconds": 14_400,
        },
    }[mode]
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "candidate_ready",
        "arm": B_NORMALIZED,
        "mode": mode,
        "sources": {
            "producer": training.binding(Path(__file__)),
            "source_config": training.binding(SOURCE_CONFIG),
        },
        "preparation": training.binding(preparation_path),
        "source_adapter": source_adapter,
        "model_config": _model_config(output=output),
        "optimizer": copy.deepcopy(training.DEFAULT_OPTIMIZER),
        "scheduler": {
            "type": "cosine",
            "total_updates": 64,
            "warmup_updates": 0,
            "min_lr_ratio": 0.0,
        },
        "objective": {
            "ce_reduction": "sample_equal",
            "geometry_reduction": "sample_equal",
            "branch_weights": {"common": 0.5, "variable": 0.5},
        },
        "ce_normalization": {
            "completion_only": True,
            "completion_denominator": "same_image_canonical_full_target_tokens_including_eos",
            "canonical_and_fallback_denominator": "active_target_tokens_including_eos",
            "mask_values": [0, 1],
            "geometry_scale": 1.0,
        },
        "validity_hinge": {
            "weight": 0.01,
            "margin": 1 / 999,
            "coordinate_token_ids": contract["coordinate_token_ids"],
            "coordinate_bin_values": list(range(1000)),
        },
        "runtime": {
            **mode_runtime,
            "seed": 19,
            "world_size": REQUIRED_WORLD_SIZE,
            "effective_image_batch": 64,
            "branch_image_count": 32,
            "microbatch_size": 2,
            "activation_checkpointing": True,
            "fresh_optimizer": True,
            "gradient_clip_norm": 1.0,
            "eos_token_id": contract["eos_token_id"],
        },
        "content_sha256": None,
    }
    manifest["content_sha256"] = training.digest(
        {key: item for key, item in manifest.items() if key != "content_sha256"}
    )
    validate_training_manifest(manifest)
    return manifest


def _validate_preparation_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "status",
        "unit_id",
        "shared_preparation",
        "accounting",
        "normalization",
        "producer",
        "content_sha256",
    }
    require(set(value) == required, "normalized preparation receipt fields")
    require(value["schema"] == PREPARATION_SCHEMA, "normalized preparation schema")
    require(value["status"] == "candidate_ready" and value["unit_id"] == UNIT_ID, "normalized preparation status")
    require(
        value["content_sha256"]
        == training.digest({key: item for key, item in value.items() if key != "content_sha256"}),
        "normalized preparation receipt digest",
    )
    shared = value["shared_preparation"]
    require(shared == training.binding(SHARED_PREPARATION), "shared preparation binding changed")
    require(shared["sha256"] == SHARED_PREPARATION_SHA256, "shared preparation digest")
    return dict(value)


def prepare(
    *, preparation_path: Path = SHARED_PREPARATION,
    output: Path,
    mode: str = "qualification",
) -> dict[str, Any]:
    """Prepare CPU-only receipt and one B-normalized qualification/main manifest."""

    output = output.resolve()
    require(not output.exists(), f"normalized trial output collision: {output}")
    receipt = build_preparation_receipt(preparation_path)
    prep_root = ROOT / "preparation" / "source256-admitted-v1"
    prep_copy = prep_root / "preparation.json"
    prep_receipt_path = prep_root / "normalization-receipt.json"
    # Keep an exact byte mirror for the new root; the manifest still binds the
    # explicitly requested shared path above.  Existing copies are accepted
    # only when byte-identical, so this never overwrites another experiment.
    prep_root.mkdir(parents=True, exist_ok=True)
    source_bytes = Path(preparation_path).read_bytes()
    if prep_copy.exists():
        require(prep_copy.read_bytes() == source_bytes, "normalized preparation mirror changed")
    else:
        prep_copy.write_bytes(source_bytes)
    if prep_receipt_path.exists():
        existing_receipt = _validate_preparation_receipt(
            json.loads(prep_receipt_path.read_text())
        )
        require(existing_receipt == receipt, "normalized preparation receipt collision")
    else:
        training.publish(prep_receipt_path, receipt)
    manifest = build_training_manifest(
        preparation_path=Path(preparation_path), mode=mode, output=output / B_NORMALIZED
    )
    manifest_path = output / B_NORMALIZED / "training-manifest.json"
    training.publish(manifest_path, manifest)
    value: dict[str, Any] = {
        "schema": TRIAL_SCHEMA,
        "status": "candidate_ready_for_GPU_qualification"
        if mode == "qualification"
        else "candidate_ready_for_paired64_after_qualification",
        "unit_id": UNIT_ID,
        "mode": mode,
        "preparation": training.binding(Path(preparation_path)),
        "preparation_mirror": training.binding(prep_copy),
        "preparation_receipt": training.binding(prep_receipt_path),
        "accounting": receipt["accounting"],
        "arm": B_NORMALIZED,
        "arms": {B_NORMALIZED: training.binding(manifest_path)},
        "manifest": training.binding(manifest_path),
        "qualification_terminal": {
            "status": "pending_lead_release",
            "path": str(output / B_NORMALIZED / "training" / "terminal.json"),
        },
        "topology": {
            "implementation": "four ranks per arm",
            "training_world_size": 4,
            "microbatch_size": 2,
            "matched_control": "predecessor B four-rank Source256 entry",
        },
        "launch": "held for lead release; preparation does not launch GPU work",
        "release_contract": {
            "schema": RELEASE_SCHEMA,
            "required_status": "lead_released",
            "required_mode": mode,
            "required_arm": B_NORMALIZED,
        },
        "producer": training.binding(Path(__file__)),
    }
    trial_path = output / "trial.json"
    output.mkdir(parents=True, exist_ok=True)
    training.publish(trial_path, value)
    ready = {
        "schema": f"{TRIAL_SCHEMA}.qualification_ready.v1",
        "status": "candidate_ready_for_lead_release",
        "unit_id": UNIT_ID,
        "mode": mode,
        "arm": B_NORMALIZED,
        "trial": training.binding(trial_path),
        "preparation": training.binding(Path(preparation_path)),
        "manifest": training.binding(manifest_path),
        "qualification_terminal": value["qualification_terminal"],
        "accounting": receipt["accounting"],
        "entry": {
            "module": "probes.training_set_completion.source256_normalized_training",
            "world_size": REQUIRED_WORLD_SIZE,
            "microbatch_size": 2,
            "requires_release_receipt": True,
            "launch_held": True,
        },
        "release_contract": value["release_contract"],
        "producer": training.binding(Path(__file__)),
    }
    training.publish(output / "qualification-ready.json", ready)
    return value


def validate_release(
    release_path: Path, *, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    release_path = release_path.resolve(strict=True)
    value = json.loads(release_path.read_text())
    require(isinstance(value, Mapping), "qualification release object")
    require(value.get("schema") == RELEASE_SCHEMA, "qualification release schema")
    require(value.get("status") == "lead_released", "lead qualification release required")
    require(value.get("unit_id") == UNIT_ID, "qualification release unit")
    require(value.get("mode") == manifest["mode"], "qualification release mode")
    arms = value.get("arms")
    require(isinstance(arms, list) and manifest["arm"] in arms, "qualification release arm")
    prep = value.get("preparation")
    if prep is not None:
        require(prep == manifest["preparation"], "qualification release preparation binding")
    return {
        "path": str(release_path),
        "binding": training.binding(release_path),
        "value": dict(value),
    }


def validate_qualification_receipt(
    value: Mapping[str, Any],
    *,
    manifest_path: Path | None = None,
    main_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Admit the post-run receipt consumed by the main execution controller."""

    require(value.get("schema") == QUALIFICATION_SCHEMA, "qualification receipt schema")
    require(value.get("status") == "accepted_actual_entry", "qualification not accepted")
    require(value.get("unit_id") == UNIT_ID, "qualification unit")
    require(value.get("arm") == B_NORMALIZED, "qualification arm")
    require(
        value.get("normalization") == QUALIFICATION_NORMALIZATION,
        "qualification normalization contract",
    )
    manifest = value.get("qualification_manifest")
    main_manifest = value.get("main_training_manifest")
    terminal = value.get("training_terminal")
    qualification_terminal = value.get("qualification_terminal")
    preparation = value.get("preparation")
    require(
        isinstance(manifest, Mapping)
        and isinstance(main_manifest, Mapping)
        and isinstance(terminal, Mapping)
        and isinstance(qualification_terminal, Mapping)
        and isinstance(preparation, Mapping),
        "qualification bindings",
    )
    require(terminal == qualification_terminal, "qualification terminal aliases")
    if manifest_path is not None:
        require(manifest == training.binding(manifest_path), "qualification manifest binding")
        require(
            preparation == json.loads(Path(manifest_path).read_text())["preparation"],
            "qualification preparation binding",
        )
    if main_manifest_path is not None:
        require(
            main_manifest == training.binding(main_manifest_path),
            "qualification main manifest binding",
        )
    return dict(value)


class _TrainingProxy:
    """Delegate the runner's helpers while enriching only normalized updates."""

    def __getattr__(self, name: str) -> Any:
        return getattr(training, name)

    def publish(self, path: str | Path, value: Any) -> None:
        training.publish(path, _enrich_update(value))


_TRAINING_PROXY = _TrainingProxy()


def _normalized_rank_receipt(**kwargs: Any) -> dict[str, Any]:
    receipt = predecessor._rank_receipt(**kwargs)
    receipt["schema"] = f"{SCHEMA}.rank.v1"
    return receipt


def _runner() -> FunctionType:
    """Create a private-global view of the old runner without mutating it.

    ``source256_training.run`` has no dependency-injection parameter.  A
    FunctionType view keeps its tested distributed/checkpoint control flow while
    resolving only the normalized globals in this module.  The predecessor
    module object and its bound bytes remain untouched in every process.
    """

    runner_globals = predecessor.run.__globals__.copy()
    runner_globals.update(
        {
            "SCHEMA": SCHEMA,
            "MANIFEST_SCHEMA": MANIFEST_SCHEMA,
            "__file__": str(Path(__file__).resolve()),
            "training": _TRAINING_PROXY,
            "_route_terms": _route_terms,
            "resolve_update_presentations": resolve_update_presentations,
            "validate_training_manifest": validate_training_manifest,
            "_dependency_bindings": _dependency_bindings,
            "_rank_receipt": _normalized_rank_receipt,
            "objective_from_presentation_terms": objective_from_presentation_terms,
            "validate_preparation": validate_preparation,
            "hydrate_bound_cases": hydrate_bound_cases,
            "source_adapter_scalar_count": source_adapter_scalar_count,
            "_batched_aligned_logits": _batched_aligned_logits,
            "_prepare_microbatches": _prepare_microbatches,
            "distributed": distributed,
            "partition_update_presentations": partition_update_presentations,
            "BRANCHES": BRANCHES,
            "PRESENTATIONS_PER_UPDATE": PRESENTATIONS_PER_UPDATE,
        }
    )
    return FunctionType(
        predecessor.run.__code__, runner_globals, name="source256_normalized_run"
    )


def _default_main_manifest_path(qualification_manifest_path: Path) -> Path:
    """Locate the paired main manifest in the standard successor runtime root."""

    return (
        qualification_manifest_path.resolve().parents[2]
        / "main-normalized-v1"
        / B_NORMALIZED
        / "training-manifest.json"
    )


def _validate_paired_main_manifest(
    qualification_manifest: Mapping[str, Any], main_manifest_path: Path
) -> tuple[Path, dict[str, Any]]:
    main_manifest_path = main_manifest_path.resolve(strict=True)
    main_manifest = validate_training_manifest(json.loads(main_manifest_path.read_text()))
    require(main_manifest["mode"] == "main", "paired main manifest mode")
    require(
        main_manifest["arm"] == B_NORMALIZED
        and main_manifest["preparation"] == qualification_manifest["preparation"],
        "paired main manifest identity",
    )
    return main_manifest_path, main_manifest


def run(
    manifest_path: Path,
    *,
    output: Path,
    release_receipt: Path | None = None,
    main_manifest_path: Path | None = None,
) -> dict[str, Any] | None:
    """Run only after an explicit lead release receipt, using four ranks."""

    require(release_receipt is not None, "lead qualification release receipt required before GPU entry")
    manifest_path = manifest_path.resolve(strict=True)
    manifest = validate_training_manifest(json.loads(manifest_path.read_text()))
    release = validate_release(release_receipt, manifest=manifest)
    paired_main: tuple[Path, dict[str, Any]] | None = None
    if manifest["mode"] == "qualification":
        paired_main = _validate_paired_main_manifest(
            manifest,
            main_manifest_path or _default_main_manifest_path(manifest_path),
        )
    result = _runner()(manifest_path, output=output)
    if result is not None and manifest["mode"] == "qualification":
        require(paired_main is not None, "paired main manifest qualification binding")
        paired_main_path, _paired_main_manifest = paired_main
        terminal_path = output.resolve() / "terminal.json"
        require(terminal_path.is_file(), "actual entry terminal receipt missing")
        qualification = {
            "schema": QUALIFICATION_SCHEMA,
            "status": "accepted_actual_entry"
            if result.get("status") == "completed"
            else "failed_actual_entry",
            "unit_id": UNIT_ID,
            "arm": manifest["arm"],
            "mode": manifest["mode"],
            "qualification_manifest": training.binding(manifest_path),
            "main_training_manifest": training.binding(paired_main_path),
            "training_terminal": training.binding(terminal_path),
            "qualification_terminal": training.binding(terminal_path),
            "preparation": manifest["preparation"],
            "lead_release": release["binding"],
            "normalization": QUALIFICATION_NORMALIZATION,
            "source_bindings": _dependency_bindings(),
        }
        qualification_path = output.resolve() / "qualification.json"
        training.publish(qualification_path, qualification)
        result["qualification"] = training.binding(qualification_path)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="CPU-only preparation and manifest")
    prepare_parser.add_argument("--preparation", type=Path, default=SHARED_PREPARATION)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--mode", choices=("qualification", "main"), default="qualification")
    verify_parser = sub.add_parser("verify", help="CPU-only manifest validation")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    run_parser = sub.add_parser("run", help="four-rank actual training entry")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--release-receipt", type=Path, required=True)
    run_parser.add_argument("--main-manifest", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(preparation_path=args.preparation, output=args.output, mode=args.mode), sort_keys=True))
    elif args.command == "verify":
        validate_training_manifest(json.loads(args.manifest.read_text()))
    else:
        result = run(
            args.manifest,
            output=args.output,
            release_receipt=args.release_receipt,
            main_manifest_path=args.main_manifest,
        )
        if result is not None:
            print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
