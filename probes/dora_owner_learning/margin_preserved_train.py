"""Eight-rank Stable50 positive-row learning with normal-margin preservation.

This task-local runner is a transparent derivative of the accepted positive32
runner.  It consumes that runner's immutable input packet plus a compact frozen
margin table.  Each normal replay produces both KL and worst-token margin from
the same logits.  ``prepare`` and ``verify`` are CPU-only; ``launch``/``rank``
and ``cold-check`` require an explicit later GPU grant.
"""
from __future__ import annotations

from src.artifacts.source_provenance import preserve_source

import argparse
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from datetime import timedelta
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch

from src.data.geometry import iou_xyxy
from src.losses import aligned_token_logprobs
from src.qwen.native import prepare_replay
from .candidate_opportunity import file_hash, require
from .composition import digest_json, loaded_composition_evidence
from .route_access import CONFIG, checkpoint_config, publish


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
RAW_ROOT = BASE / "2026-09-11-margin-preserved-positive-branch"
OLD_ROOT = BASE / "2026-09-11-positive-branch-vs-repeat-event"
BASE_INPUT = OLD_ROOT / "trainer-preparation" / "inputs.json"
OLD_A2_RECEIPT = OLD_ROOT / "smoke-A-retry1" / "receipt.json"
OLD_A32_RECEIPT = OLD_ROOT / "full-A" / "receipt.json"
MARGIN_INPUT = RAW_ROOT / "margin-preparation" / "inputs.json"
PREPARATION = RAW_ROOT / "trainer-preparation-v2"

SCHEMA = "margin_preserved_train.inputs.v1"
PROTOCOL_SCHEMA = "margin_preserved_train.protocol.v1"
RECEIPT_SCHEMA = "margin_preserved_train.receipt.v1"
RESOURCE_SCHEMA = "margin_preserved_train.resource_envelope.v1"
ARMS = ("C",)
WORLD_SIZE = 8
SMOKE_UPDATES = 2
FULL_UPDATES = 32
POSITIVE_COUNT = 3
NORMAL_COUNT = 56
NORMALS_PER_RANK = 7
EVENTS_PER_RANK = 3
EVENT_DENOMINATOR = 24
BOX_END = 151649
EOS = 151645
PAD = 151643
ACTION_BUDGET = 64
EXPECTED_TRAINABLE_TENSORS = 588
EXPECTED_TRAINABLE_SCALARS = 18_006_016
OPTIMIZER = dict(
    lr=1e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
    foreach=False,
)
CLIP_GRADIENT_NORM = 1.0
CONDITIONAL_KL_WEIGHT = 10.0
NORMAL_KL_WEIGHT = 100.0
EVENT_WEIGHT = 1.0
MARGIN_WEIGHT = 10.0
BASE_ENGINE_SHA256 = "bc6d1728dc5c156b6d617ecb9b99a71b1e81ee305e338c2f1476b3cdc9035308"
MARGIN_INPUT_SHA256 = "afeb7f46139eef57f28927201d5c7839ee706f78152491f9f260be981d775e36"
ELIGIBLE_MARGIN_COUNT = 6_030
INELIGIBLE_NEAR_TIE_COUNT = 17
SOURCE_MARGIN_MINIMUM = 0.001
SOURCE_MARGIN_FLOOR_MAXIMUM = 0.1
SMOKE_MAX_RANK_SECONDS = 600
SMOKE_MAX_CUDA_BYTES = 24 * 1024**3
SMOKE_MAX_RSS_BYTES = 24 * 1024**3
SMOKE_MAX_MODEL_FORWARDS_PER_RANK = 480
SMOKE_MAX_IMAGE_FORWARDS_PER_RANK = 64
FULL_MAX_RANK_SECONDS = 1500
FULL_MAX_CUDA_BYTES = 24 * 1024**3
FULL_MAX_RSS_BYTES = 24 * 1024**3
FULL_MAX_MODEL_FORWARDS_PER_RANK = 535
FULL_MAX_IMAGE_FORWARDS_PER_RANK = 535
SELECTED = ("351017-c01", "417044-c01", "477415-c02")


def combine_lifecycle_resource_observations(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Keep process-lifecycle peaks and the latest elapsed observation."""

    require(observations, "resource lifecycle requires at least one observation")
    cuda_allocated = [row.get("peak_cuda_allocated_bytes") for row in observations]
    cuda_reserved = [row.get("peak_cuda_reserved_bytes") for row in observations]
    require(
        all(type(row.get("peak_rss_bytes")) is int and row["peak_rss_bytes"] >= 0
            and isinstance(row.get("elapsed_seconds"), (int, float))
            and row["elapsed_seconds"] >= 0 for row in observations),
        "resource lifecycle observation",
    )
    return {
        "observation_count": len(observations),
        "phases": [str(row.get("phase")) for row in observations],
        "peak_cuda_allocated_bytes": (
            max(value for value in cuda_allocated if type(value) is int)
            if any(type(value) is int for value in cuda_allocated) else None
        ),
        "peak_cuda_reserved_bytes": (
            max(value for value in cuda_reserved if type(value) is int)
            if any(type(value) is int for value in cuda_reserved) else None
        ),
        "peak_rss_bytes": max(int(row["peak_rss_bytes"]) for row in observations),
        "elapsed_seconds": max(float(row["elapsed_seconds"]) for row in observations),
        "cuda_measurement_errors": [
            str(row["cuda_measurement_error"])
            for row in observations if row.get("cuda_measurement_error")
        ],
    }


def _capture_lifecycle_resources(
    *, device: torch.device | None, started: float, phase: str,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "phase": phase,
        "peak_cuda_allocated_bytes": None,
        "peak_cuda_reserved_bytes": None,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "elapsed_seconds": time.monotonic() - started,
    }
    if device is not None:
        try:
            observation["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            observation["peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        except RuntimeError as exc:
            observation["cuda_measurement_error"] = f"{type(exc).__name__}: {exc}"
    return observation


def load_json(path: str | Path) -> Any:
    """Read bound JSON; producer formatting is not part of its byte identity."""

    return json.loads(Path(path).read_text(encoding="utf-8"))








def digest_ids(values: Sequence[int]) -> str:
    # The producer intentionally hashes literal ID arrays without key sorting.
    return hashlib.sha256(
        json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _checked_ids(values: Any, *, field: str, nonempty: bool = True) -> list[int]:
    require(isinstance(values, list), f"{field} must be a literal token-id list")
    require(
        all(type(value) is int and value >= 0 for value in values),
        f"{field} contains an invalid token id",
    )
    require(not nonempty or values, f"{field} must be nonempty")
    return list(values)


def _checked_positions(values: Any, *, length: int, field: str) -> list[int]:
    require(isinstance(values, list), f"{field} must be a list")
    require(
        all(type(value) is int and 0 <= value < length for value in values),
        f"{field} contains an out-of-range position",
    )
    require(len(set(values)) == len(values), f"{field} contains duplicate positions")
    return list(values)


def validate_positive_case(case: Mapping[str, Any]) -> dict[str, Any]:
    candidate_id = str(case.get("candidate_id"))
    require(candidate_id in SELECTED, "unexpected positive candidate")
    prompt = case.get("prompt", {})
    h, c, w = (case.get(key, {}) for key in ("h", "c", "w"))
    prompt_ids = _checked_ids(prompt.get("token_ids"), field=f"{candidate_id}.prompt")
    h_ids = _checked_ids(h.get("token_ids"), field=f"{candidate_id}.h")
    c_ids = _checked_ids(c.get("token_ids"), field=f"{candidate_id}.c")
    w_ids = _checked_ids(w.get("token_ids"), field=f"{candidate_id}.w")
    for label, item, ids in (
        ("prompt", prompt, prompt_ids), ("h", h, h_ids),
        ("c", c, c_ids), ("w", w, w_ids),
    ):
        require(item.get("length", len(ids)) == len(ids), f"{candidate_id}.{label} length")
        hash_value = item.get("ids_sha256")
        require(hash_value is None or hash_value == digest_ids(ids), f"{candidate_id}.{label} hash")
    require(c_ids[0] == 151646 and c_ids[-1] == BOX_END and EOS not in c_ids,
            f"{candidate_id}: c is not one complete non-EOS row")
    require(w_ids[0] == 151646 and w_ids[-1] == BOX_END and EOS not in w_ids,
            f"{candidate_id}: w is not one complete non-EOS row")
    offsets = case.get("continuation_offsets", {})
    h_end, c_end, w_end = len(h_ids), len(h_ids) + len(c_ids), len(h_ids) + len(c_ids) + len(w_ids)
    require(offsets.get("h_span") == [0, h_end], f"{candidate_id}: h span shifted")
    require(offsets.get("c_target_span") == [h_end, c_end], f"{candidate_id}: c span shifted")
    require(offsets.get("w_only_kl_span") == [c_end, w_end], f"{candidate_id}: w span shifted")
    require(offsets.get("c_target_positions") == list(range(h_end, c_end)),
            f"{candidate_id}: c target is not complete/exact")
    require(offsets.get("w_only_kl_positions") == list(range(c_end, w_end)),
            f"{candidate_id}: w KL mask is not complete/exact")
    require(offsets.get("eos_in_c_target") is False and offsets.get("eos_in_w_only_kl") is False,
            f"{candidate_id}: EOS entered c/w")
    image = case.get("image", {})
    require(str(image.get("image_id")) in {item.split("-", 1)[0] for item in SELECTED},
            f"{candidate_id}: image identity")
    path = Path(str(image.get("image_path", "")))
    require(path.is_file() and file_hash(path) == image.get("image_sha256"),
            f"{candidate_id}: image bytes changed")
    return dict(case)


def validate_manifest(value: Mapping[str, Any], *, verify_sources: bool = True) -> dict[str, Any]:
    require(value.get("schema") == "positive_branch_vs_repeat_event.input_preparation.v1",
            "unexpected input-preparation schema")
    require(value.get("status") == "candidate_only" and value.get("immutable_candidate_manifest") is True,
            "input package is not immutable candidate evidence")
    positives = value.get("positives")
    require(isinstance(positives, list) and [row.get("candidate_id") for row in positives] == list(SELECTED),
            "exact ordered positive cohort changed")
    checked = [validate_positive_case(row) for row in positives]
    normals = value.get("normals", {}).get("cases")
    require(isinstance(normals, list) and len(normals) == NORMAL_COUNT, "normal56 coverage")
    require(len({str(row.get("image_id")) for row in normals}) == NORMAL_COUNT,
            "normal references repeat an image")
    positive_images = {str(row["image"]["image_id"]) for row in checked}
    require(not positive_images & {str(row["image_id"]) for row in normals},
            "positive/normal populations overlap")
    normal_states = 0
    normal_action_tokens = 0
    for row in normals:
        ids = _checked_ids(row.get("action_ids"), field=f"normal.{row.get('image_id')}.action")
        require(ids[-1] == EOS and row.get("stop_reason") == "im_end", "normal reference is not native EOS")
        require(row.get("action_ids_sha256") == digest_ids(ids), "normal action hash changed")
        layout = row.get("initial_layout", {})
        require(layout.get("action_token_count") == len(ids), "normal layout length changed")
        positions = _checked_positions(
            layout.get("kl_positions"), length=len(ids), field=f"normal.{row.get('image_id')}.kl_positions",
        )
        normal_states += len(positions)
        normal_action_tokens += len(ids)
        path = Path(str(row.get("image", {}).get("image_path", "")))
        require(path.is_file() and file_hash(path) == row.get("image", {}).get("image_sha256"),
                "normal image bytes changed")
    require(value.get("counts", {}).get("normal_kl_positions") == normal_states == 6047,
            "normal KL state count changed")
    counts = value.get("counts", {})
    require(
        counts.get("positive_count") == POSITIVE_COUNT
        and counts.get("normal_count") == NORMAL_COUNT
        and counts.get("positive_h_tokens") == sum(len(row["h"]["token_ids"]) for row in checked) == 161
        and counts.get("positive_c_tokens") == sum(len(row["c"]["token_ids"]) for row in checked) == 30
        and counts.get("positive_w_tokens") == sum(len(row["w"]["token_ids"]) for row in checked) == 29
        and counts.get("normal_action_tokens") == normal_action_tokens == 6056,
        "frozen token/mask totals changed",
    )
    summary = value.get("normals", {}).get("mask_summary", {})
    require(summary.get("geometry_invalid_reference_image_ids") == ["360573"] and
            summary.get("parser_nonclean_reference_count") == 1,
            "known normal360573 exclusion changed")
    exceptional, = [row for row in normals if str(row["image_id"]) == "360573"]
    exception_layout = exceptional["initial_layout"]
    require(
        exception_layout["invalid_geometry_rows"] == [8]
        and len(exception_layout["parser_drop_rows"]) == 1
        and exception_layout["parser_drop_rows"][0]["reason"] == "geometry_invalid"
        and exception_layout["parser_drop_rows"][0]["token_positions"] == list(range(75, 84))
        and exception_layout["kl_positions"] == [*range(75), 84],
        "normal360573 exact nine-token mask exception changed",
    )
    model = value.get("model_identity", {})
    effective = model.get("effective_adapter", {})
    anchor = Path(str(effective.get("path", "")))
    require(anchor.is_dir() and effective.get("role") == "Stable50_anchor_adapter",
            "effective Stable50 anchor is absent")
    for entry in effective.get("files", []):
        require(file_hash(anchor / entry["relative_path"]) == entry["sha256"],
                "Stable50 adapter bytes changed")
    require(model.get("declared_packet_adapter", {}).get("path") != str(anchor),
            "source adapter trap is no longer explicit")
    if verify_sources:
        for binding in value.get("source_bindings", {}).values():
            if isinstance(binding, Mapping) and "path" in binding and "sha256" in binding:
                require(file_hash(binding["path"]) == binding["sha256"],
                        f"bound source changed: {binding['path']}")
    return dict(value)


def truncate_first_action(
    generated_ids: Sequence[int], *, stop_reason: str, budget: int = ACTION_BUDGET,
) -> dict[str, Any]:
    ids = list(generated_ids)
    require(ids and all(type(token) is int and token >= 0 for token in ids), "generated suffix IDs")
    require(stop_reason in ("im_end", "length"), "unsupported native stop")
    require(len(ids) <= budget, "generated suffix exceeds budget")
    if stop_reason == "im_end":
        require(ids[-1] == EOS and EOS not in ids[:-1], "native EOS bookkeeping changed")
    else:
        require(len(ids) == budget and EOS not in ids, "length-censored suffix bookkeeping changed")
    box_index = ids.index(BOX_END) if BOX_END in ids else None
    eos_index = ids.index(EOS) if EOS in ids else None
    if box_index is not None and (eos_index is None or box_index < eos_index):
        end, terminal = box_index + 1, "box_end"
    elif eos_index is not None:
        end, terminal = eos_index + 1, "eos"
    else:
        end, terminal = len(ids), "censored"
    retained = ids[:end]
    return {
        "terminal_class": terminal,
        "retained_ids": retained,
        "retained_ids_sha256": digest_ids(retained),
        "retained_tokens": len(retained),
        "full_generated_ids": ids,
        "full_generated_ids_sha256": digest_ids(ids),
        "full_generated_tokens": len(ids),
        "discarded_tail_ids": ids[end:],
        "discarded_tail_tokens": len(ids) - end,
        "native_stop_reason": stop_reason,
        "native_eos_observed": eos_index is not None,
    }


def duplicate_indicator(
    candidate_bbox: Sequence[int | float], prior_bboxes: Sequence[Sequence[int | float]],
    *, threshold: float = 0.95,
) -> dict[str, Any]:
    overlaps = [float(iou_xyxy(tuple(candidate_bbox), tuple(previous))) for previous in prior_bboxes]
    matches = [index for index, overlap in enumerate(overlaps) if overlap > threshold]
    return {
        "D": int(bool(matches)),
        "max_iou_to_h": max(overlaps, default=0.0),
        "matching_prior_row_indices": matches,
        "matching_prior_row_count": len(matches),
        "counted_event_rows": int(bool(matches)),
    }


def classify_event(
    *, h_ids: Sequence[int], action: Mapping[str, Any], tokenizer: Any,
    image_width: int, image_height: int, row_id: str,
) -> dict[str, Any]:
    terminal = action["terminal_class"]
    if terminal != "box_end":
        return dict(D=0, outcome=terminal, geometry_valid=False, max_iou_to_h=0.0,
                    matching_prior_row_count=0, counted_event_rows=0)
    from .geometric_dedup import trajectory_layout

    prefix = trajectory_layout(
        list(h_ids), tokenizer, image_width=image_width, image_height=image_height,
        row_id=f"{row_id}:h",
    )
    retained = list(action["retained_ids"])
    combined = trajectory_layout(
        [*h_ids, *retained], tokenizer, image_width=image_width, image_height=image_height,
        row_id=f"{row_id}:h-plus-action",
    )
    h_length = len(h_ids)
    prior = [row for row in combined["row_spans"] if row["token_end"] <= h_length]
    require(
        [(row["token_start"], row["token_end"], row["bbox_pixel_xyxy"]) for row in prior] ==
        [(row["token_start"], row["token_end"], row["bbox_pixel_xyxy"]) for row in prefix["row_spans"]],
        "event parsing changed literal h rows",
    )
    new_rows = [row for row in combined["row_spans"] if row["token_start"] >= h_length]
    if len(new_rows) != 1 or new_rows[0]["token_start"] != h_length or \
            new_rows[0]["token_end"] != h_length + len(retained):
        invalid = any(
            row.get("token_start") == h_length and row.get("token_end") == h_length + len(retained)
            and row.get("reason") == "geometry_invalid"
            for row in combined["parser_drop_rows"]
        )
        return dict(D=0, outcome="geometry_invalid" if invalid else "malformed",
                    geometry_valid=False, max_iou_to_h=0.0,
                    matching_prior_row_count=0, counted_event_rows=0,
                    parser_drops=combined["parser_drops"])
    candidate = new_rows[0]
    result = duplicate_indicator(
        candidate["bbox_pixel_xyxy"], [row["bbox_pixel_xyxy"] for row in prior],
    )
    return {
        **result,
        "outcome": "duplicate" if result["D"] else "valid_nonduplicate",
        "geometry_valid": True,
        "candidate_bbox_pixel_xyxy": candidate["bbox_pixel_xyxy"],
        "candidate_description": candidate["description"],
        "prior_valid_rows": len(prior),
    }


def sample_seed(step: int, rank: int, case_index: int) -> int:
    require(1 <= step <= FULL_UPDATES and 0 <= rank < WORLD_SIZE and
            0 <= case_index < POSITIVE_COUNT, "sample seed coordinates")
    return 2_026_091_100_000 + step * 1_000 + rank * 10 + case_index


def event_loss(chosen_logprobs: torch.Tensor, event: int) -> torch.Tensor:
    require(chosen_logprobs.ndim == 1 and chosen_logprobs.numel() > 0,
            "event chosen logprobs")
    require(event in (0, 1), "event indicator")
    # Local /3 followed by DDP /8 is exactly the frozen /24 denominator.
    return chosen_logprobs.sum() * float(event) * (EVENT_WEIGHT / POSITIVE_COUNT)


def positive_loss(logits: torch.Tensor, targets: torch.Tensor, c_ids: Sequence[int]) -> torch.Tensor:
    require(logits.ndim == 2 and targets.ndim == 1 and logits.shape[0] == len(c_ids),
            "positive logits must cover only complete c")
    require(targets.tolist() == list(c_ids), "positive target differs from literal complete c")
    return -aligned_token_logprobs(logits, targets).sum() / POSITIVE_COUNT


def reference_kl(
    logits: torch.Tensor, reference_logp: torch.Tensor, positions: Sequence[int],
) -> torch.Tensor:
    require(logits.dtype == reference_logp.dtype == torch.float32, "KL must remain FP32")
    require(reference_logp.shape == (len(positions), logits.shape[1]), "KL reference shape")
    ref = reference_logp.detach()
    current = torch.log_softmax(logits[list(positions)], dim=-1)
    return (ref.exp() * (ref - current)).sum(-1).mean()


def replicated_positive_contribution(per_case_losses: Sequence[torch.Tensor]) -> torch.Tensor:
    require(len(per_case_losses) == POSITIVE_COUNT, "replicated positive count")
    return sum(per_case_losses)


def local_event_contribution(per_case_losses: Sequence[torch.Tensor]) -> torch.Tensor:
    require(len(per_case_losses) == EVENTS_PER_RANK, "local event count")
    return sum(per_case_losses)


def local_normal_contribution(per_reference_kl: Sequence[torch.Tensor]) -> torch.Tensor:
    require(len(per_reference_kl) == NORMALS_PER_RANK, "local normal count")
    return sum(value * (NORMAL_KL_WEIGHT * WORLD_SIZE / NORMAL_COUNT)
               for value in per_reference_kl)


def current_target_margins(
    logits: torch.Tensor, target_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return literal-target margins and the current full-vocabulary competitor IDs."""

    require(
        logits.ndim == 2 and target_ids.ndim == 1
        and logits.shape[0] == target_ids.numel() and logits.shape[1] >= 2,
        "margin logits/targets",
    )
    require(
        target_ids.dtype == torch.long and bool((target_ids >= 0).all())
        and bool((target_ids < logits.shape[1]).all()),
        "margin target IDs",
    )
    top = torch.topk(logits, k=2, dim=-1)
    target_is_top = top.indices[:, 0] == target_ids
    best_other_ids = torch.where(target_is_top, top.indices[:, 1], top.indices[:, 0])
    best_other_logits = logits.gather(1, best_other_ids[:, None]).squeeze(1)
    target_logits = logits.gather(1, target_ids[:, None]).squeeze(1)
    return target_logits - best_other_logits, best_other_ids


def validate_margin_case(
    value: Mapping[str, Any], *, action_ids: Sequence[int], kl_positions: Sequence[int],
    expected_key: str | None = None,
) -> dict[str, Any]:
    """Validate one compact, literal-position margin record without model execution."""

    key = str(value.get("key", ""))
    require(key and (expected_key is None or key == expected_key), "margin case key")
    positions = _checked_positions(
        value.get("eligible_positions"), length=len(action_ids), field=f"margin.{key}.positions",
    )
    target_ids = _checked_ids(
        value.get("target_ids"), field=f"margin.{key}.target_ids",
        nonempty=bool(positions),
    )
    source_margins = value.get("source_margins")
    floors = value.get("floors")
    require(
        len(positions) == len(target_ids)
        and isinstance(source_margins, list) and len(source_margins) == len(positions)
        and isinstance(floors, list) and len(floors) == len(positions),
        f"margin.{key} aligned arrays",
    )
    require(set(positions).issubset(set(kl_positions)), f"margin.{key} left original KL mask")
    require(
        target_ids == [int(action_ids[position]) for position in positions],
        f"margin.{key} literal target mismatch",
    )
    require(
        all(isinstance(margin, (int, float)) and math.isfinite(float(margin))
            and float(margin) > SOURCE_MARGIN_MINIMUM for margin in source_margins),
        f"margin.{key} source margin eligibility",
    )
    expected_floors = [min(SOURCE_MARGIN_FLOOR_MAXIMUM, 0.5 * float(margin))
                       for margin in source_margins]
    require(
        all(isinstance(floor, (int, float)) and math.isfinite(float(floor))
            and abs(float(floor) - expected) <= 1e-7
            for floor, expected in zip(floors, expected_floors, strict=True)),
        f"margin.{key} floor formula",
    )
    return {
        **dict(value),
        "key": key,
        "eligible_positions": positions,
        "target_ids": target_ids,
        "source_margins": [float(value) for value in source_margins],
        "floors": [float(value) for value in floors],
    }


def worst_margin_penalty(
    logits: torch.Tensor, action_targets: torch.Tensor, margin: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Worst one-sided floor violation for one image, using current competitors."""

    checked = validate_margin_case(
        margin,
        action_ids=action_targets.detach().cpu().tolist(),
        kl_positions=margin.get("original_kl_positions", margin["eligible_positions"]),
        expected_key=str(margin.get("key", "")),
    )
    positions = checked["eligible_positions"]
    if not positions:
        zero = logits.sum() * 0.0
        return zero, {
            "eligible_count": 0, "active_count": 0, "penalty": 0.0,
            "literal_argmax_flips": 0,
            "worst_action_position": None, "worst_target_id": None,
            "worst_best_other_id": None, "worst_source_margin": None,
            "worst_floor": None, "worst_current_margin": None,
        }
    selected = logits[positions]
    target_ids = torch.tensor(checked["target_ids"], device=logits.device, dtype=torch.long)
    floors = torch.tensor(checked["floors"], device=logits.device, dtype=logits.dtype)
    current, best_other_ids = current_target_margins(selected, target_ids)
    violations = torch.relu(floors - current)
    penalty, worst_index = violations.max(dim=0)
    index = int(worst_index.detach())
    return penalty, {
        "eligible_count": len(positions),
        "active_count": int((violations.detach() > 0).sum()),
        "literal_argmax_flips": int((selected.detach().argmax(-1) != target_ids).sum()),
        "penalty": float(penalty.detach()),
        "worst_action_position": positions[index],
        "worst_target_id": checked["target_ids"][index],
        "worst_best_other_id": int(best_other_ids[index].detach()),
        "worst_source_margin": checked["source_margins"][index],
        "worst_floor": checked["floors"][index],
        "worst_current_margin": float(current[index].detach()),
        "minimum_current_margin": float(current.detach().min()),
        "mean_current_margin": float(current.detach().mean()),
    }


def local_margin_contribution(penalty: torch.Tensor, *, weight: float) -> torch.Tensor:
    require(weight in (0.0, MARGIN_WEIGHT), "margin weight must be frozen zero or ten")
    return penalty * (weight * WORLD_SIZE / NORMAL_COUNT)


def summarize_final_reference(
    rows: Sequence[Mapping[str, Any]], *, margin_weight: float,
) -> dict[str, Any]:
    require(len(rows) == NORMAL_COUNT, "final reference normal56")
    require(
        sum(row["margin"]["eligible_count"] for row in rows) == ELIGIBLE_MARGIN_COUNT,
        "final reference eligible6030",
    )
    raw_kl = sum(float(row["raw_normal_kl"]) for row in rows) / NORMAL_COUNT
    raw_margin = sum(float(row["raw_margin_R_i"]) for row in rows) / NORMAL_COUNT
    return {
        "normal_count": NORMAL_COUNT,
        "eligible_positions": ELIGIBLE_MARGIN_COUNT,
        "raw_normal_kl_mean": raw_kl,
        "scaled_normal_kl": NORMAL_KL_WEIGHT * raw_kl,
        "raw_margin_R_mean": raw_margin,
        "scaled_margin": margin_weight * raw_margin,
        "active_images": sum(float(row["raw_margin_R_i"]) > 0.0 for row in rows),
        "active_floors": sum(int(row["margin"]["active_count"]) for row in rows),
        "eligible_literal_argmax_flips": sum(
            int(row["margin"]["literal_argmax_flips"]) for row in rows
        ),
        "active_worst_positions": [
            {"key": row["key"], **row["margin"]}
            for row in rows if float(row["raw_margin_R_i"]) > 0.0
        ],
    }


def _protocol() -> dict[str, Any]:
    return {
        "schema": PROTOCOL_SCHEMA,
        "status": "frozen",
        "arms": list(ARMS),
        "world_size": WORLD_SIZE,
        "selected": list(SELECTED),
        "objectives": {
            "C": (
                "mean3(-sum_logp(c))+10*mean3(KL_anchor(w))"
                "+100*mean56(KL_anchor(normal))+margin_weight*mean56(R_i)"
            ),
            "R_i": "max_eligible relu(min(0.1,0.5*m0)-m_current)",
            "m_current": "literal_target_logit-max_over_current_full_vocab_other_logits",
        },
        "margin": {
            "runtime_weights": [0.0, MARGIN_WEIGHT],
            "full_weight": MARGIN_WEIGHT,
            "source_margin_minimum_strict": SOURCE_MARGIN_MINIMUM,
            "floor": "min(0.1,0.5*m0)",
            "floor_maximum": SOURCE_MARGIN_FLOOR_MAXIMUM,
            "eligible_positions": ELIGIBLE_MARGIN_COUNT,
            "retained_kl_only_near_ties": INELIGIBLE_NEAR_TIE_COUNT,
            "same_normal_forward_as_kl": True,
        },
        "optimizer": {**OPTIMIZER, "betas": list(OPTIMIZER["betas"])},
        "clip_gradient_norm": CLIP_GRADIENT_NORM,
        "updates": {"smoke": SMOKE_UPDATES, "full": FULL_UPDATES},
        "sampling": None,
        "distributed": {
            "replicated_positive_count_per_rank": 3,
            "replicated_conditional_count_per_rank": 3,
            "normal_references_per_rank": 7,
            "margin_references_per_rank": 7,
            "backwards_per_rank": {"C": 13},
            "synchronized_backwards_per_step": 1,
            "normal_margin_local_scale": "margin_weight*8/56",
        },
        "full_bounds": {
            "model_loads": 8, "reference_forwards": 80,
            "model_forwards": 3566, "image_forwards": 3566,
            "backwards": 3328, "synchronized_backwards": 256,
            "post_update_normal_readback_forwards": 56,
            "sampling_calls": 0,
        },
    }


def _code_paths() -> list[Path]:
    return [
        Path(__file__).with_name("composition.py"),
        Path(__file__),
        Path(__file__).with_name("tests") / "test_margin_preserved_train.py",
        Path(__file__).with_name("repeat_recovery_train.py"),
        Path(__file__).with_name("runtime.py"),
        Path(__file__).with_name("train.py"),
        Path(__file__).with_name("route_access.py"),
        Path(__file__).with_name("geometric_dedup.py"),
        Path(__file__).with_name("geometric_dedup_train.py"),
        Path(__file__).with_name("selective_preservation_dense.py"),
        Path(__file__).parents[2] / "src" / "qwen" / "generation.py",
        Path(__file__).parents[2] / "src" / "qwen" / "native.py",
        Path(__file__).parents[2] / "src" / "losses" / "token_scores.py",
        Path(__file__).parents[2] / "src" / "adapters" / "dora.py",
        CONFIG,
    ]


def _validated_margin_table(
    value: Mapping[str, Any], *, manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    require(
        value.get("schema") == "margin_preserved_positive_branch.margin_inputs.v1"
        and value.get("status") == "prepared_no_model_execution"
        and value.get("candidate_only") is True,
        "margin table is not a prepared CPU artifact",
    )
    rows = value.get("cases")
    require(isinstance(rows, list) and len(rows) == NORMAL_COUNT, "margin normal56 coverage")
    normals = manifest["normals"]["cases"]
    normal_by_key = {str(row["key"]): row for row in normals}
    require(len(normal_by_key) == NORMAL_COUNT, "manifest normal keys")
    checked: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("key", ""))
        require(key in normal_by_key and key not in checked, "margin key coverage/uniqueness")
        normal = normal_by_key[key]
        require(
            str(row.get("image_id")) == str(normal["image_id"])
            and row.get("action_ids_sha256") == normal["action_ids_sha256"],
            f"margin.{key} source identity",
        )
        checked[key] = validate_margin_case(
            row,
            action_ids=normal["action_ids"],
            kl_positions=normal["initial_layout"]["kl_positions"],
            expected_key=key,
        )
        checked[key]["original_kl_positions"] = list(normal["initial_layout"]["kl_positions"])
    require(set(checked) == set(normal_by_key), "margin normal56 keys changed")
    eligible = sum(len(row["eligible_positions"]) for row in checked.values())
    require(eligible == ELIGIBLE_MARGIN_COUNT, "eligible margin count changed")
    require(
        sum(len(normal_by_key[key]["initial_layout"]["kl_positions"])
            - len(row["eligible_positions"]) for key, row in checked.items())
        == INELIGIBLE_NEAR_TIE_COUNT,
        "retained KL-only near-tie count changed",
    )
    counts = value.get("counts", {})
    require(
        counts.get("normal_count") == NORMAL_COUNT
        and counts.get("eligible_margin_positions") == ELIGIBLE_MARGIN_COUNT
        and counts.get("retained_kl_only_positions") == INELIGIBLE_NEAR_TIE_COUNT,
        "margin table counts",
    )
    return checked


def prepare(base_input: Path, margin_input: Path, output: Path) -> dict[str, Any]:
    """Bind accepted positive32 inputs to the separately frozen margin table."""

    require(not output.exists(), "occupied trainer-preparation root")
    from . import repeat_recovery_train as base_train

    require(file_hash(base_train.__file__) == BASE_ENGINE_SHA256,
            "accepted positive32 engine bytes changed")
    base_input = base_input.resolve(strict=True)
    base_packet, manifest = base_train.validate_inputs(base_input, verify_sources=True)
    margin_input = margin_input.resolve(strict=True)
    require(file_hash(margin_input) == MARGIN_INPUT_SHA256,
            "lead-accepted margin input bytes changed")
    margin_packet = load_json(margin_input)
    margin_cases = _validated_margin_table(margin_packet, manifest=manifest)
    output.mkdir(parents=True, exist_ok=False)
    publish(output / "protocol.json", _protocol())
    diff_text = "".join(difflib.unified_diff(
        Path(base_train.__file__).read_text(encoding="utf-8").splitlines(keepends=True),
        Path(__file__).read_text(encoding="utf-8").splitlines(keepends=True),
        fromfile=f"accepted/{Path(base_train.__file__).name}",
        tofile=f"candidate/{Path(__file__).name}",
    ))
    (output / "base-engine.diff").write_text(diff_text, encoding="utf-8")
    publish(output / "change-map.json", {
        "base_engine_sha256": BASE_ENGINE_SHA256,
        "candidate_engine_sha256": file_hash(__file__),
        "unified_diff": {"path": str(output / "base-engine.diff"),
                         "sha256": file_hash(output / "base-engine.diff"),
                         "line_count": len(diff_text.splitlines())},
        "semantic_changes": [
            "bind accepted positive32 packet plus frozen normal margin table",
            "compute KL and worst-token margin from each normal replay's one logits tensor",
            "record raw KL, raw margin, scaled components, current competitor, and active worst positions",
            "read all 56 normals once without grad after the final update and seal final burden metrics",
            "support only C with weight0 smoke oracle or frozen weight10 objective",
            "seal fixed counters, resource bounds, retained A2 parity, and margin summaries",
        ],
        "unchanged_route": [
            "Stable50/base/embedding/frontend", "positive and conditional masks",
            "normal KL masks", "optimizer and clip", "DDP item order and one sync per update",
            "activation checkpointing and adapter-only export",
        ],
    })
    staged = []
    for path in [*_code_paths(), Path(preserve_source.__code__.co_filename)]:
        path = path.resolve(strict=True)
        target = preserve_source(path, run_root=output, relative_name=Path('effective_code') / str(path.resolve()).lstrip('/'))
        staged.append(dict(path=str(path), staged=str(target), sha256=file_hash(target)))
    code_identity = {
        "files": staged,
        "base_engine_sha256": BASE_ENGINE_SHA256,
        "change_map": {"path": str(output / "change-map.json"),
                       "sha256": file_hash(output / "change-map.json")},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
    }
    publish(output / "code_identity.json", code_identity)
    packet = {
        "schema": SCHEMA,
        "status": "prepared_no_model_execution",
        "base_input": {"path": str(base_input), "sha256": file_hash(base_input)},
        "base_engine": {"path": str(Path(base_train.__file__).resolve()),
                        "sha256": BASE_ENGINE_SHA256},
        "margin_input": {"path": str(margin_input), "sha256": file_hash(margin_input)},
        "protocol": {"path": str(output / "protocol.json"),
                     "sha256": file_hash(output / "protocol.json")},
        "code_identity": {"path": str(output / "code_identity.json"),
                          "sha256": file_hash(output / "code_identity.json")},
        "change_map": {"path": str(output / "change-map.json"),
                       "sha256": file_hash(output / "change-map.json")},
        "parity_oracles": {
            "retained_A2": {"path": str(OLD_A2_RECEIPT), "sha256": file_hash(OLD_A2_RECEIPT),
                            "final_adapter_state_sha256": load_json(OLD_A2_RECEIPT)["final_adapter_state_sha256"]},
            "retained_A32": {"path": str(OLD_A32_RECEIPT), "sha256": file_hash(OLD_A32_RECEIPT),
                             "final_adapter_state_sha256": load_json(OLD_A32_RECEIPT)["final_adapter_state_sha256"]},
        },
        "manifest": base_packet["manifest"],
        "input_admission": base_packet["input_admission"],
        "stable50_inputs": base_packet["stable50_inputs"],
        "model_identity": base_packet["model_identity"],
        "stable50_adapter": base_packet["stable50_adapter"],
        "source_embedding": base_packet["source_embedding"],
        "config": base_packet["config"],
        "counts": {**base_packet["counts"], "eligible_margin_positions": sum(
            len(case["eligible_positions"]) for case in margin_cases.values()
        )},
        "smoke_limits": {
            "max_rank_seconds": SMOKE_MAX_RANK_SECONDS,
            "max_cuda_allocated_bytes": SMOKE_MAX_CUDA_BYTES,
            "max_cuda_reserved_bytes": SMOKE_MAX_CUDA_BYTES,
            "max_rss_bytes": SMOKE_MAX_RSS_BYTES,
            "max_model_forwards_per_rank": SMOKE_MAX_MODEL_FORWARDS_PER_RANK,
            "max_image_forwards_per_rank": SMOKE_MAX_IMAGE_FORWARDS_PER_RANK,
            "max_negative_samples_global": 0,
            "max_raw_sampled_tokens_global": 0,
        },
        "full_limits": "pending measured weight10 smoke resource envelope",
    }
    publish(output / "inputs.json", packet)
    module = "probes.dora_owner_learning.margin_preserved_train"
    commands = {
        "prepare": f"python -m {module} prepare --base-input {base_input} --margin-input {margin_input} --output {output}",
        "smoke_weight0": (
            f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m {module} launch "
            f"--input {output / 'inputs.json'} --arm C --mode smoke --margin-weight 0 "
            f"--output-root {RAW_ROOT / 'smoke-weight0'}"
        ),
        "smoke_weight10": (
            f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m {module} launch "
            f"--input {output / 'inputs.json'} --arm C --mode smoke --margin-weight 10 "
            f"--output-root {RAW_ROOT / 'smoke-weight10'}"
        ),
        "cold_weight10": (
            f"CUDA_VISIBLE_DEVICES=0 python -m {module} cold-check "
            f"--input {output / 'inputs.json'} --output-root {RAW_ROOT / 'smoke-weight10'}"
        ),
        "smoke_gate": (
            "Run weight0 once, verify exact final adapter state against retained A2, then run "
            "weight10 once; each requires a separate explicit root launch grant."
        ),
        "full_C": (
            f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m {module} launch "
            f"--input {output / 'inputs.json'} --arm C --mode full --margin-weight 10 "
            f"--resource-envelope {output / 'resource-envelope-C.json'} "
            f"--output-root {RAW_ROOT / 'full-C'}"
        ),
        "full_gate": "Requires a later explicit root grant after both paired smoke receipts.",
    }
    publish(output / "proposed-commands.json", commands)
    return packet


def _validate_code_identity(packet: Mapping[str, Any]) -> None:
    reference = packet["code_identity"]
    require(file_hash(reference["path"]) == reference["sha256"], "code identity record changed")
    identity = load_json(reference["path"])
    require(identity.get("base_engine_sha256") == BASE_ENGINE_SHA256,
            "code identity base engine")
    for entry in identity["files"]:
        require(file_hash(entry["path"]) == entry["sha256"] and
                file_hash(entry["staged"]) == entry["sha256"],
                f"effective code changed: {entry['path']}")


def validate_inputs(path: Path, *, verify_sources: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    packet = load_json(path)
    require(packet.get("schema") == SCHEMA and packet.get("status") == "prepared_no_model_execution",
            "trainer input packet status/schema")
    require(file_hash(packet["protocol"]["path"]) == packet["protocol"]["sha256"] and
            load_json(packet["protocol"]["path"]) == _protocol(),
            "frozen task-local protocol changed")
    require(file_hash(packet["change_map"]["path"]) == packet["change_map"]["sha256"],
            "base-engine change map changed")
    _validate_code_identity(packet)
    from . import repeat_recovery_train as base_train
    base_ref = packet["base_input"]
    require(
        file_hash(base_ref["path"]) == base_ref["sha256"]
        and file_hash(packet["base_engine"]["path"]) == BASE_ENGINE_SHA256
        and packet["base_engine"]["sha256"] == BASE_ENGINE_SHA256,
        "accepted base packet/engine changed",
    )
    base, manifest = base_train.validate_inputs(
        Path(base_ref["path"]), verify_sources=verify_sources,
    )
    require(
        all(packet[key] == base[key] for key in (
            "manifest", "input_admission", "stable50_inputs", "model_identity",
            "stable50_adapter", "source_embedding", "config",
        )),
        "base packet fields were rewritten",
    )
    margin_ref = packet["margin_input"]
    require(file_hash(margin_ref["path"]) == margin_ref["sha256"] == MARGIN_INPUT_SHA256,
            "lead-accepted margin table changed")
    margins = _validated_margin_table(load_json(margin_ref["path"]), manifest=manifest)
    require(packet["counts"]["eligible_margin_positions"] == ELIGIBLE_MARGIN_COUNT,
            "prepared margin count")
    for name, expected in (
        ("retained_A2", OLD_A2_RECEIPT), ("retained_A32", OLD_A32_RECEIPT),
    ):
        reference = packet["parity_oracles"][name]
        receipt = load_json(expected)
        require(
            reference["path"] == str(expected)
            and file_hash(expected) == reference["sha256"]
            and reference["final_adapter_state_sha256"] == receipt["final_adapter_state_sha256"],
            f"{name} parity oracle changed",
        )
    _verify_identity(packet["stable50_adapter"], label="Stable50 adapter")
    _verify_identity(packet["source_embedding"], label="source embedding")
    packet = {**packet, "_validated_margins": margins}
    return packet, manifest


def _target_score_stats(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    chosen = aligned_token_logprobs(logits, targets)
    top = torch.topk(logits, k=2, dim=-1)
    best_other = torch.where(top.indices[:, 0] == targets, top.values[:, 1], top.values[:, 0])
    target_logits = logits.gather(1, targets[:, None]).squeeze(1)
    margins = target_logits - best_other
    return {
        "token_count": int(targets.numel()),
        "sum_logprob": float(chosen.detach().sum()),
        "mean_logprob": float(chosen.detach().mean()),
        "mean_target_margin": float(margins.detach().mean()),
        "min_target_margin": float(margins.detach().min()),
        "argmax_target_tokens": int((logits.argmax(-1) == targets).sum()),
    }


class MarginPreservedScorer(torch.nn.Module):
    """One exact-replay item; normal KL and margin share one logits tensor."""

    def __init__(self, model: torch.nn.Module, *, margin_weight: float) -> None:
        super().__init__()
        require(margin_weight in (0.0, MARGIN_WEIGHT), "frozen margin weight")
        self.model = model
        self.margin_weight = float(margin_weight)

    def forward(
        self,
        inputs: Mapping[str, Any],
        prompt_ids: Sequence[int],
        action_ids: Sequence[int],
        *,
        kind: str,
        reference_logp: torch.Tensor | None = None,
        positions: Sequence[int] = (),
        margin: Mapping[str, Any] | None = None,
        event: int = 0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        replay = prepare_replay(
            self.model, inputs, prompt_token_ids=prompt_ids,
            continuation_token_ids=action_ids,
        )
        logits = replay.aligned_logits(self.model(**replay.inputs).logits)
        require(replay.target_ids.tolist() == list(action_ids), "exact replay target identity")
        if kind == "positive":
            loss = positive_loss(logits, replay.target_ids, action_ids)
            stats = {"positive_nll_sum": float(-aligned_token_logprobs(logits, replay.target_ids).detach().sum()),
                     "conditional_kl": 0.0, "normal_kl": 0.0,
                     "margin_penalty": 0.0, "margin_scaled_loss": 0.0,
                     "event_loss": 0.0,
                     "route": _target_score_stats(logits.detach(), replay.target_ids)}
        elif kind in ("conditional", "normal"):
            require(reference_logp is not None, f"{kind} reference is absent")
            kl = reference_kl(logits, reference_logp, positions)
            coefficient = (CONDITIONAL_KL_WEIGHT / POSITIVE_COUNT
                           if kind == "conditional"
                           else NORMAL_KL_WEIGHT * WORLD_SIZE / NORMAL_COUNT)
            loss = coefficient * kl
            stats = {"positive_nll_sum": 0.0,
                     "conditional_kl": float(kl.detach()) if kind == "conditional" else 0.0,
                     "normal_kl": float(kl.detach()) if kind == "normal" else 0.0,
                     "normal_kl_scaled_loss": float(loss.detach()) if kind == "normal" else 0.0,
                     "margin_penalty": 0.0, "margin_scaled_loss": 0.0,
                     "event_loss": 0.0}
            if kind == "normal":
                require(margin is not None, "normal margin record absent")
                penalty, margin_stats = worst_margin_penalty(logits, replay.target_ids, margin)
                margin_loss = local_margin_contribution(penalty, weight=self.margin_weight)
                # The weight-zero oracle follows the old A arithmetic and graph exactly.
                if self.margin_weight != 0.0:
                    loss = loss + margin_loss
                stats.update(
                    margin_penalty=float(penalty.detach()),
                    margin_scaled_loss=float(margin_loss.detach()),
                    margin=margin_stats,
                )
        else:
            require(kind == "event" and reference_logp is None, "unsupported scorer item")
            chosen = aligned_token_logprobs(logits, replay.target_ids)
            loss = event_loss(chosen, event)
            stats = {"positive_nll_sum": 0.0, "conditional_kl": 0.0,
                     "normal_kl": 0.0, "margin_penalty": 0.0,
                     "margin_scaled_loss": 0.0, "event_loss": float(loss.detach()),
                     "action_sum_logprob": float(chosen.detach().sum()),
                     "action_mean_logprob": float(chosen.detach().mean()),
                     "chosen_token_score": _target_score_stats(logits.detach(), replay.target_ids),
                     "event": int(event)}
        require(loss.ndim == 0 and bool(torch.isfinite(loss)), "nonfinite scalar item loss")
        return loss, stats


def _materialize_case(
    *, qwen: Any, frontend: Any, config: Any, raw: Any, case: Mapping[str, Any],
    positive: bool,
) -> dict[str, Any]:
    from .runtime import build_request, materialize

    if positive:
        image, prompt = case["image"], case["prompt"]
        example_id = str(image["row_id"])
        row_index = int(image["row_index"])
        expected_prompt = list(prompt["token_ids"])
        expected_prompt_hash = prompt["ids_sha256"]
    else:
        image, example_id = case["image"], str(case["example_id"])
        row_index = int(image["row_index"])
        expected_prompt = list(case["prompt_token_ids"])
        expected_prompt_hash = case["prompt_token_ids_sha256"]
    require(str(raw.example_id) == example_id, "raw/materialized example identity")
    request, planned, _ = build_request(
        raw, config=config, qwen=frontend.qwen, row_index=row_index,
    )
    batch = materialize(qwen, request)
    prompt_ids = list(batch.prompt_token_ids[0])
    require(prompt_ids == expected_prompt and digest_ids(prompt_ids) == expected_prompt_hash,
            "live prompt token identity")
    require(Path(planned.image_path).resolve() == Path(image["image_path"]).resolve() and
            planned.image_content_sha256 == image["image_sha256"],
            "live planned image identity")
    require(list(batch.image_grids[0]) == list(image["observed_image_grid_thw"]) and
            batch.media_sha256[0] == image["executed_media_sha256"],
            "live projected image identity")
    return {"case": case, "batch": batch, "inputs": dict(batch.inputs),
            "prompt_ids": prompt_ids, "positive": positive}


def _reference_logp(
    model: torch.nn.Module, inputs: Mapping[str, Any], prompt_ids: Sequence[int],
    action_ids: Sequence[int], positions: Sequence[int],
) -> torch.Tensor:
    with torch.no_grad():
        replay = prepare_replay(
            model, inputs, prompt_token_ids=prompt_ids,
            continuation_token_ids=action_ids,
        )
        logits = replay.aligned_logits(model(**replay.inputs).logits)
        require(replay.target_ids.tolist() == list(action_ids), "reference target identity")
        result = torch.log_softmax(logits[list(positions)], -1).detach().cpu()
    require(result.dtype == torch.float32 and result.shape[0] == len(positions) and
            not result.requires_grad and result.grad_fn is None,
            "detached FP32 reference cache")
    return result


def _resource_limits(
    packet: Mapping[str, Any], *, arm: str, mode: str,
    envelope_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    require(arm in ARMS and mode in ("smoke", "full"), "arm/mode")
    if mode == "smoke":
        require(envelope_path is None, "smoke cannot consume a post-smoke envelope")
        return dict(packet["smoke_limits"]), None
    require(envelope_path is not None and envelope_path.is_file(),
            "full run requires a measured resource envelope")
    envelope = load_json(envelope_path)
    require(envelope.get("schema") == RESOURCE_SCHEMA and
            envelope.get("status") == "lead_accepted" and envelope.get("arm") == arm and
            envelope.get("mode") == "full", "resource envelope schema/status/arm")
    smoke = envelope.get("source_smoke_receipt", {})
    require(file_hash(smoke.get("path", "")) == smoke.get("sha256"),
            "resource envelope smoke receipt changed")
    receipt = load_json(smoke["path"])
    require(receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("arm") == arm and
            receipt.get("mode") == "smoke" and receipt.get("margin_weight") == MARGIN_WEIGHT
            and receipt.get("status") == "completed",
            "resource envelope does not bind a completed same-arm smoke")
    cold = envelope.get("source_smoke_cold_check", {})
    require(file_hash(cold.get("path", "")) == cold.get("sha256"),
            "resource envelope smoke cold-check changed")
    cold_check = load_json(cold["path"])
    require(cold_check.get("status") == "passed" and cold_check.get("arm") == arm and
            cold_check.get("mode") == "smoke" and
            cold_check.get("training_receipt", {}).get("sha256") == smoke["sha256"],
            "full run requires bound completed same-arm cold reload")
    limits = envelope.get("limits", {})
    keys = (
        "max_rank_seconds", "max_cuda_allocated_bytes", "max_cuda_reserved_bytes",
        "max_rss_bytes", "max_model_forwards_per_rank", "max_image_forwards_per_rank",
    )
    require(all(type(limits.get(key)) in (int, float) and limits[key] > 0 for key in keys),
            "resource envelope limits")
    require(
        limits["max_rank_seconds"] <= FULL_MAX_RANK_SECONDS
        and limits["max_cuda_allocated_bytes"] <= FULL_MAX_CUDA_BYTES
        and limits["max_cuda_reserved_bytes"] <= FULL_MAX_CUDA_BYTES
        and limits["max_rss_bytes"] <= FULL_MAX_RSS_BYTES
        and limits["max_model_forwards_per_rank"] <= FULL_MAX_MODEL_FORWARDS_PER_RANK
        and limits["max_image_forwards_per_rank"] <= FULL_MAX_IMAGE_FORWARDS_PER_RANK,
        "full resource envelope exceeds frozen unit ceilings",
    )
    return dict(limits), {"path": str(envelope_path.resolve()), "sha256": file_hash(envelope_path)}


def _optimizer_hash(optimizer: torch.optim.Optimizer, named: Sequence[tuple[str, torch.Tensor]]) -> str:
    from .selective_preservation_dense import optimizer_hash
    return optimizer_hash(optimizer, named)


def _positive_gradient_projection(
    scorer: MarginPreservedScorer,
    positive_entries: Sequence[Mapping[str, Any]],
    named: Sequence[tuple[str, torch.nn.Parameter]],
) -> tuple[dict[str, Any], tuple[torch.Tensor, ...]]:
    losses = []
    for entry in positive_entries:
        case = entry["case"]
        loss, _ = scorer(
            entry["inputs"], [*entry["prompt_ids"], *case["h"]["token_ids"]],
            case["c"]["token_ids"], kind="positive",
        )
        losses.append(loss)
    positive = replicated_positive_contribution(losses)
    gradients = torch.autograd.grad(
        positive, [parameter for _, parameter in named], retain_graph=False,
        create_graph=False, allow_unused=False,
    )
    actual_sq = positive_sq = dot = 0.0
    for (_, parameter), gradient in zip(named, gradients, strict=True):
        require(parameter.grad is not None, "actual reduced gradient absent")
        actual = parameter.grad.detach().double()
        value = gradient.detach().double()
        actual_sq += float(actual.square().sum())
        positive_sq += float(value.square().sum())
        dot += float((actual * value).sum())
    require(actual_sq > 0 and positive_sq > 0, "projection requires nonzero gradients")
    return ({
        "assumption": "replicated_positive_inputs_loss_and_eval_mode_make_positive_gradient_rank_identical",
        "actual_raw_gradient_l2": math.sqrt(actual_sq),
        "replicated_positive_gradient_l2": math.sqrt(positive_sq),
        "actual_dot_positive_gradient": dot,
        "actual_cosine_positive_gradient": dot / math.sqrt(actual_sq * positive_sq),
        "gradient_descent_projection_on_positive_descent": dot / positive_sq,
    }, tuple(gradient.detach() for gradient in gradients))


def _score_positive_routes(
    model: torch.nn.Module, positive_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    with torch.no_grad():
        for entry in positive_entries:
            case = entry["case"]
            replay = prepare_replay(
                model, entry["inputs"],
                prompt_token_ids=[*entry["prompt_ids"], *case["h"]["token_ids"]],
                continuation_token_ids=case["c"]["token_ids"],
            )
            logits = replay.aligned_logits(model(**replay.inputs).logits)
            result[case["candidate_id"]] = _target_score_stats(logits, replay.target_ids)
    return result


def _sample_events(
    *, model: torch.nn.Module, qwen: Any,
    positive_entries: Sequence[Mapping[str, Any]], step: int, rank: int,
    counters: dict[str, int],
) -> list[dict[str, Any]]:
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations

    policy = NativeGenerationPolicy(
        temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
        use_model_defaults=False,
    )
    events = []
    for case_index, entry in enumerate(positive_entries):
        case = entry["case"]
        seed = sample_seed(step, rank, case_index)
        generated, = generate_continuations(
            model, entry["batch"], extensions=(case["h"]["token_ids"],),
            budgets=(ACTION_BUDGET,), eos_token_id=EOS,
            pad_token_id=qwen.tokenizer.pad_token_id, policy=policy,
            trace="none", seed=seed, allow_pad_tokens=True,
        )
        action = truncate_first_action(generated.token_ids, stop_reason=generated.stop_reason)
        event = classify_event(
            h_ids=case["h"]["token_ids"], action=action, tokenizer=qwen.tokenizer,
            image_width=int(case["image"]["image_width"]),
            image_height=int(case["image"]["image_height"]),
            row_id=f"{case['candidate_id']}:step{step}:rank{rank}",
        )
        counters["negative_samples"] += 1
        counters["raw_sampled_tokens"] += action["full_generated_tokens"]
        counters["retained_event_tokens"] += action["retained_tokens"]
        events.append({"candidate_id": case["candidate_id"], "seed": seed,
                       "action": action, "event": event})
    return events


def execute_rank(
    *, input_path: Path, arm: str, mode: str, output_root: Path,
    envelope_path: Path | None, margin_weight: float,
) -> None:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from src.adapters.dora import select_dora_parameters
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_research_infer_config
    from src.data import load_raw_examples
    from src.inference.runtime import assemble_frontend
    from .geometric_dedup_train import (
        checkpointing_receipt, install_language_decoder_checkpointing,
    )
    from .runtime import load_policy
    from .train import (
        _all_true, _dist_values, _parameter_layout, _save_adapter_only,
        _tensor_state_hash,
    )

    rank, local_rank, world = [int(os.environ.get(key, "-1"))
                               for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")]
    require(world == WORLD_SIZE and rank == local_rank and 0 <= rank < WORLD_SIZE and
            os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3,4,5,6,7",
            "exact single-node eight-rank topology")
    require(arm == "C" and margin_weight in (0.0, MARGIN_WEIGHT),
            "exact C arm and frozen smoke/full margin weight")
    require(mode == "smoke" or margin_weight == MARGIN_WEIGHT,
            "full C cannot disable margin")
    updates = SMOKE_UPDATES if mode == "smoke" else FULL_UPDATES
    packet, manifest = validate_inputs(input_path, verify_sources=True)
    limits, envelope_ref = _resource_limits(
        packet, arm=arm, mode=mode, envelope_path=envelope_path,
    )
    run = output_root / "ranks" / f"rank{rank}"
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    counters = {
        "model_loads": 0, "model_forwards": 0, "image_forwards": 0,
        "reference_forwards": 0, "training_replays": 0,
        "positive_projection_replays": 0, "positive_score_replays": 0,
        "generation_calls": 0, "negative_samples": 0,
        "raw_sampled_tokens": 0, "retained_event_tokens": 0,
        "backwards": 0, "synchronized_backwards": 0,
        "optimizer_steps": 0, "explicit_collectives": 0,
        "normal_kl_items": 0, "margin_items": 0,
        "active_margin_images": 0,
        "final_reference_forwards": 0,
    }
    status, error, final_state = "failed", None, {}
    dist_initialized = False
    device: torch.device | None = None
    resource_observations: list[dict[str, Any]] = []

    def expired(*_: Any) -> None:
        raise TimeoutError(f"{limits['max_rank_seconds']}-second rank ceiling")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(math.ceil(float(limits["max_rank_seconds"])))
    with (run / "execution.log").open("x", encoding="utf-8") as log, \
            redirect_stdout(log), redirect_stderr(log):
      try:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", timeout=timedelta(seconds=600), device_id=device)
        dist_initialized = True

        def gather(value: Any) -> list[Any]:
            counters["explicit_collectives"] += 1
            return _dist_values(value)

        def all_true(value: bool) -> bool:
            counters["explicit_collectives"] += 1
            return _all_true(value, device)

        require(len(set(gather((file_hash(input_path), file_hash(packet["manifest"]["path"]))))) == 1,
                "rank input identity")
        base_config = load_research_infer_config(CONFIG).config
        require(base_config.model_dump(mode="json") == packet["config"],
                "live Source-shaped config differs from prepared packet")
        anchor = packet["model_identity"]["effective_adapter"]
        require(base_config.adapter is not None and
                str(base_config.adapter.path) == packet["model_identity"]["declared_packet_adapter"]["path"] and
                str(base_config.adapter.path) != anchor["path"],
                "Stable50 override trap was not preserved")
        config = checkpoint_config(base_config, anchor["path"])
        require(str(config.adapter.path) == anchor["path"] and config.model.dtype == "fp32" and
                config.embedding_delta is not None and
                str(config.embedding_delta.path) == packet["source_embedding"]["root"] and
                config.backend.hf.attn_implementation == "sdpa" and
                config.backend.hf.patch_embed_linearization == "enabled",
                "effective Stable50 FP32/SDPA config")
        frontend = assemble_frontend(
            config,
            generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")),
        )
        raw = {str(row.example_id): row for row in load_raw_examples(config.data.input_jsonl)}
        qwen, loaded_identity = load_policy(config, device=device)
        counters["model_loads"] = 1
        require(qwen.token_identity.im_end_token_ids == (EOS,) and qwen.tokenizer.pad_token_id == PAD,
                "native terminal token identity")
        model = qwen.model
        model.eval()
        identity = loaded_identity["model_identity"]
        from src.qwen.special_token_embeddings import inspect_special_token_embedding_delta_payload
        live_embedding = inspect_special_token_embedding_delta_payload(
            identity["embedding_delta"]["identity"]["delta_path"],
            packet["model_identity"]["base_model"],
        )
        publish(run / "loaded-model.json", loaded_identity)
        composition_check = loaded_composition_evidence(
            loaded_identity=loaded_identity,
            expected_base=packet["model_identity"]["base_model"],
            expected_adapter=anchor["path"],
            expected_embedding=packet["source_embedding"],
            inspected_embedding=live_embedding,
        )
        publish(run / "loaded-composition-check.json", composition_check)
        require(composition_check["passed"], "loaded Stable50 composition")

        active_phase = "idle"
        def count_model(*_: Any) -> None:
            counters["model_forwards"] += 1
            require(counters["model_forwards"] <= limits["max_model_forwards_per_rank"],
                    "model-forward ceiling")
        model.register_forward_pre_hook(count_model)
        visuals = [module for name, module in model.named_modules() if name.endswith("visual")]
        require(len(visuals) == 1, "single visual module")
        def count_image(*_: Any) -> None:
            counters["image_forwards"] += 1
            require(counters["image_forwards"] <= limits["max_image_forwards_per_rank"],
                    "image-forward ceiling")
        visuals[0].register_forward_pre_hook(count_image)

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        named = select_dora_parameters(model, towers=("language",), adapter_name="default")
        require(len(named) == EXPECTED_TRAINABLE_TENSORS and
                sum(parameter.numel() for _, parameter in named) == EXPECTED_TRAINABLE_SCALARS and
                all("language_model" in name and not any(fragment in name for fragment in
                    ("visual", "vision", "merger", "embed_tokens", "lm_head"))
                    for name, _ in named),
                "exact language-only DoRA surface")
        for _, parameter in named:
            parameter.requires_grad_(True)
        selected = {id(parameter) for _, parameter in named}
        frozen = [(name, parameter) for name, parameter in model.named_parameters()
                  if id(parameter) not in selected]
        frozen_versions = [(parameter, parameter._version) for _, parameter in frozen]
        frozen_hash = _tensor_state_hash(frozen)
        source_adapter_hash = _tensor_state_hash(named)
        require(len(set(gather((source_adapter_hash, frozen_hash)))) == 1,
                "initial parameter bytes differ by rank")
        publish(run / "trainable-layout.json", _parameter_layout(named))

        positive_entries = []
        for case in manifest["positives"]:
            example_id = str(case["image"]["row_id"])
            require(example_id in raw, "positive raw example absent")
            positive_entries.append(_materialize_case(
                qwen=qwen, frontend=frontend, config=config, raw=raw[example_id],
                case=case, positive=True,
            ))
        normal_cases = manifest["normals"]["cases"][rank::WORLD_SIZE]
        require(len(normal_cases) == NORMALS_PER_RANK, "rank normal7 assignment")
        normal_entries = []
        for case in normal_cases:
            require(case["example_id"] in raw, "normal raw example absent")
            normal_entries.append(_materialize_case(
                qwen=qwen, frontend=frontend, config=config, raw=raw[case["example_id"]],
                case=case, positive=False,
            ))

        checkpointing = install_language_decoder_checkpointing(model)
        checkpointing["enabled"] = False
        conditional_refs: dict[str, torch.Tensor] = {}
        normal_refs: dict[str, torch.Tensor] = {}
        reference_cards = []
        active_phase = "reference"
        for entry in positive_entries:
            case = entry["case"]
            reference = _reference_logp(
                model, entry["inputs"],
                [*entry["prompt_ids"], *case["h"]["token_ids"], *case["c"]["token_ids"]],
                case["w"]["token_ids"], list(range(len(case["w"]["token_ids"]))),
            )
            counters["reference_forwards"] += 1
            conditional_refs[case["candidate_id"]] = reference
            reference_cards.append({"kind": "conditional", "key": case["candidate_id"],
                                    "shape": list(reference.shape),
                                    "bytes": reference.numel() * reference.element_size(),
                                    "sha256": _tensor_state_hash([("reference", reference)])})
        for entry in normal_entries:
            case = entry["case"]
            positions = case["initial_layout"]["kl_positions"]
            reference = _reference_logp(
                model, entry["inputs"], entry["prompt_ids"], case["action_ids"], positions,
            )
            counters["reference_forwards"] += 1
            normal_refs[case["key"]] = reference
            reference_cards.append({"kind": "normal", "key": case["key"],
                                    "shape": list(reference.shape),
                                    "bytes": reference.numel() * reference.element_size(),
                                    "sha256": _tensor_state_hash([("reference", reference)])})
        require(counters["reference_forwards"] == 10, "three conditional plus seven normal references")
        publish(run / "reference-cache.json", reference_cards)
        resource_observations.append(_capture_lifecycle_resources(
            device=device, started=started, phase="post_preparation_and_reference_cache",
        ))

        scorer = MarginPreservedScorer(model, margin_weight=margin_weight)
        ddp = DDP(scorer, device_ids=[local_rank], output_device=local_rank,
                  broadcast_buffers=False, init_sync=False)
        checkpointing["enabled"] = True
        checkpointing["phase"] = "train"
        optimizer = torch.optim.AdamW([parameter for _, parameter in named], **OPTIMIZER)
        require(not optimizer.state, "each arm must start with fresh AdamW")
        initial_scores = _score_positive_routes(model, positive_entries) if rank == 0 else None
        if rank == 0:
            counters["positive_score_replays"] += POSITIVE_COUNT
            publish(output_root / "initial-positive-scores.json", initial_scores)
        for step in range(1, updates + 1):
            require(time.monotonic() - started < float(limits["max_rank_seconds"]), "rank wall ceiling")
            active_phase = "generation"
            events: list[dict[str, Any]] = []
            active_phase = "train"
            items: list[dict[str, Any]] = []
            for entry in positive_entries:
                case = entry["case"]
                items.append({"kind": "positive", "entry": entry,
                              "prompt_ids": [*entry["prompt_ids"], *case["h"]["token_ids"]],
                              "action_ids": case["c"]["token_ids"], "reference": None,
                              "positions": [], "event": 0, "key": case["candidate_id"]})
            for entry in positive_entries:
                case = entry["case"]
                items.append({"kind": "conditional", "entry": entry,
                              "prompt_ids": [*entry["prompt_ids"], *case["h"]["token_ids"], *case["c"]["token_ids"]],
                              "action_ids": case["w"]["token_ids"],
                              "reference": conditional_refs[case["candidate_id"]],
                              "positions": list(range(len(case["w"]["token_ids"]))),
                              "event": 0, "key": case["candidate_id"]})
            for entry in normal_entries:
                case = entry["case"]
                items.append({"kind": "normal", "entry": entry,
                              "prompt_ids": entry["prompt_ids"], "action_ids": case["action_ids"],
                              "reference": normal_refs[case["key"]],
                              "positions": case["initial_layout"]["kl_positions"],
                              "margin": packet["_validated_margins"][case["key"]],
                              "event": 0, "key": case["key"]})
            expected_backwards = 13
            require(len(items) == expected_backwards, "C local backward count")
            optimizer.zero_grad(set_to_none=True)
            before = [parameter.detach().clone() for _, parameter in named]
            local_records = []
            for index, item in enumerate(items):
                synchronized = index == len(items) - 1
                reference = (None if item["reference"] is None
                             else item["reference"].to(device))
                with nullcontext() if synchronized else ddp.no_sync():
                    loss, stats = ddp(
                        item["entry"]["inputs"], item["prompt_ids"], item["action_ids"],
                        kind=item["kind"], reference_logp=reference,
                        positions=item["positions"], margin=item.get("margin"),
                        event=item["event"],
                    )
                    loss.backward()
                counters["training_replays"] += 1
                counters["backwards"] += 1
                counters["synchronized_backwards"] += int(synchronized)
                counters["normal_kl_items"] += int(item["kind"] == "normal")
                counters["margin_items"] += int(item["kind"] == "normal")
                counters["active_margin_images"] += int(
                    item["kind"] == "normal" and stats["margin_penalty"] > 0.0
                )
                record = {"index": index, "kind": item["kind"], "key": item["key"],
                          "synchronized": synchronized, "scaled_loss": float(loss.detach()), **stats}
                local_records.append(record)
                del reference, loss
            require(counters["backwards"] == step * expected_backwards and
                    counters["synchronized_backwards"] == step,
                    "single-sync backward choreography")
            finite = all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                         for _, parameter in named)
            require(all_true(finite), "selected gradients missing/nonfinite")
            require(all(parameter.grad is None and not parameter.requires_grad for _, parameter in frozen),
                    "frozen surface acquired gradients")
            gradient_hash = _tensor_state_hash([(name, parameter.grad) for name, parameter in named])
            require(len(set(gather(gradient_hash))) == 1, "reduced gradient differs by rank")
            positive_routes = [record["route"] for record in local_records if record["kind"] == "positive"]
            require(len(set(gather(digest_json(positive_routes)))) == 1,
                    "replicated positive route scores differ by rank")
            projection = None
            positive_gradients: tuple[torch.Tensor, ...] | None = None
            if rank == 0:
                projection, positive_gradients = _positive_gradient_projection(
                    scorer, positive_entries, named,
                )
                counters["positive_projection_replays"] += POSITIVE_COUNT
            raw_norm = float(torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in named], CLIP_GRADIENT_NORM,
                error_if_nonfinite=True, foreach=False,
            ))
            clipped_norm = math.sqrt(sum(float(parameter.grad.detach().double().square().sum())
                                         for _, parameter in named))
            require(math.isfinite(raw_norm) and raw_norm > 0 and clipped_norm <= 1.000001,
                    "global gradient clip")
            optimizer.step()
            counters["optimizer_steps"] = step
            require(all(parameter._version == version for parameter, version in frozen_versions),
                    "frozen parameter version changed")
            movement = math.sqrt(sum(float((parameter.detach() - old).double().square().sum())
                                     for (_, parameter), old in zip(named, before, strict=True)))
            require(movement > 0 and math.isfinite(movement), "zero/nonfinite update movement")
            adapter_hash = _tensor_state_hash(named)
            optimizer_hash = _optimizer_hash(optimizer, named)
            require(len(set(gather((adapter_hash, optimizer_hash)))) == 1,
                    "adapter/optimizer state differs by rank")
            if rank == 0:
                assert projection is not None and positive_gradients is not None
                descent_dot = descent_sq = delta_sq = 0.0
                for (_, parameter), old, positive_gradient in zip(named, before, positive_gradients, strict=True):
                    delta = (parameter.detach() - old).double()
                    descent = -positive_gradient.double()
                    descent_dot += float((delta * descent).sum())
                    descent_sq += float(descent.square().sum())
                    delta_sq += float(delta.square().sum())
                projection.update(
                    optimizer_delta_l2=math.sqrt(delta_sq),
                    optimizer_delta_dot_positive_descent=descent_dot,
                    optimizer_delta_cosine_positive_descent=(
                        descent_dot / math.sqrt(delta_sq * descent_sq)
                    ),
                )
            del before, positive_gradients
            local = {
                "rank": rank, "step": step, "records": local_records,
                "loss_sum_before_ddp_mean": sum(record["scaled_loss"] for record in local_records),
                "raw_gradient_norm": raw_norm, "clipped_gradient_norm": clipped_norm,
                "gradient_sha256": gradient_hash, "adapter_sha256": adapter_hash,
                "optimizer_sha256": optimizer_hash, "movement_l2": movement,
                "positive_gradient_projection": projection,
                "counters": dict(counters), "elapsed_seconds": time.monotonic() - started,
            }
            publish(run / f"update-{step:02d}.json", local)
            ranks = gather(local)
            if rank == 0:
                global_row = {
                    "step": step, "arm": arm, "mode": mode,
                    "margin_weight": margin_weight,
                    "ranks": ranks,
                    "global_objective_value": sum(row["loss_sum_before_ddp_mean"] for row in ranks) / WORLD_SIZE,
                    "objective_components": {
                        "positive": sum(record["scaled_loss"] for row in ranks for record in row["records"]
                                        if record["kind"] == "positive") / WORLD_SIZE,
                        "conditional_kl": sum(record["scaled_loss"] for row in ranks for record in row["records"]
                                              if record["kind"] == "conditional") / WORLD_SIZE,
                        "normal_kl": sum(record["normal_kl_scaled_loss"] for row in ranks
                                         for record in row["records"] if record["kind"] == "normal") / WORLD_SIZE,
                        "margin": sum(record["margin_scaled_loss"] for row in ranks
                                      for record in row["records"] if record["kind"] == "normal") / WORLD_SIZE,
                    },
                    "margin_summary": {
                        "eligible_positions": sum(record["margin"]["eligible_count"] for row in ranks
                                                  for record in row["records"] if record["kind"] == "normal"),
                        "active_images": sum(record["margin_penalty"] > 0.0 for row in ranks
                                             for record in row["records"] if record["kind"] == "normal"),
                        "active_positions": sum(record["margin"]["active_count"] for row in ranks
                                                for record in row["records"] if record["kind"] == "normal"),
                        "worst_positions": [
                            {"key": record["key"], **record["margin"]}
                            for row in ranks for record in row["records"]
                            if record["kind"] == "normal" and record["margin_penalty"] > 0.0
                        ],
                    },
                    "event_denominator": 0, "event_count": 0,
                    "event_classes": {}, "raw_sampled_tokens": 0,
                    "retained_event_tokens": 0,
                }
                require(global_row["margin_summary"]["eligible_positions"] == ELIGIBLE_MARGIN_COUNT,
                        "global eligible margin denominator")
                publish(output_root / f"update-{step:02d}.json", global_row)
            dist.barrier()
            del local_records, ranks, events, items

        active_phase = "final_normal_readback"
        final_adapter_before = _tensor_state_hash(named)
        final_optimizer_before = _optimizer_hash(optimizer, named)
        final_normal_records = []
        with torch.no_grad():
            for entry in normal_entries:
                case = entry["case"]
                reference = normal_refs[case["key"]].to(device)
                loss, stats = scorer(
                    entry["inputs"], entry["prompt_ids"], case["action_ids"],
                    kind="normal", reference_logp=reference,
                    positions=case["initial_layout"]["kl_positions"],
                    margin=packet["_validated_margins"][case["key"]],
                )
                counters["final_reference_forwards"] += 1
                final_normal_records.append({
                    "key": case["key"],
                    "raw_normal_kl": stats["normal_kl"],
                    "raw_margin_R_i": stats["margin_penalty"],
                    "scaled_normal_kl": stats["normal_kl_scaled_loss"],
                    "scaled_margin": stats["margin_scaled_loss"],
                    "margin": stats["margin"],
                })
                del reference, loss
        require(len(final_normal_records) == NORMALS_PER_RANK,
                "final normal7 readback count")
        require(
            _tensor_state_hash(named) == final_adapter_before
            and _optimizer_hash(optimizer, named) == final_optimizer_before,
            "final no-grad readback changed trainable or optimizer state",
        )
        final_reference = {
            "schema": "margin_preserved_train.final_reference.v1",
            "rank": rank, "arm": arm, "mode": mode,
            "margin_weight": margin_weight, "post_update": updates,
            "records": final_normal_records,
            "summary": {
                "normal_count": len(final_normal_records),
                "eligible_positions": sum(row["margin"]["eligible_count"]
                                          for row in final_normal_records),
                "raw_normal_kl_sum": sum(row["raw_normal_kl"] for row in final_normal_records),
                "raw_margin_R_i_sum": sum(row["raw_margin_R_i"] for row in final_normal_records),
                "active_images": sum(row["raw_margin_R_i"] > 0.0
                                     for row in final_normal_records),
                "active_floors": sum(row["margin"]["active_count"]
                                    for row in final_normal_records),
                "eligible_literal_argmax_flips": sum(
                    row["margin"]["literal_argmax_flips"] for row in final_normal_records
                ),
            },
            "state_unchanged": {
                "adapter_sha256": final_adapter_before,
                "optimizer_sha256": final_optimizer_before,
            },
        }
        publish(run / "final-reference.json", final_reference)

        active_phase = "final_positive_readback"
        final_scores = _score_positive_routes(model, positive_entries) if rank == 0 else None
        if rank == 0:
            counters["positive_score_replays"] += POSITIVE_COUNT
            publish(output_root / "final-live-positive-scores.json", final_scores)
        require(all_true(_tensor_state_hash(frozen) == frozen_hash), "terminal frozen bytes changed")
        resource_observations.append(_capture_lifecycle_resources(
            device=device, started=started, phase="post_training_pre_export",
        ))
        state = {
            "rank": rank, "counters": dict(counters),
            "adapter_sha256": _tensor_state_hash(named),
            "optimizer_sha256": _optimizer_hash(optimizer, named),
            "frozen_sha256": frozen_hash,
            "reference_cache_bytes": sum(card["bytes"] for card in reference_cards),
            "pre_export_lifecycle_resources": combine_lifecycle_resource_observations(
                resource_observations
            ),
            "activation_checkpointing": checkpointing_receipt(model, checkpointing),
        }
        states = gather(state)
        require(len({row["adapter_sha256"] for row in states}) == 1 and
                len({row["optimizer_sha256"] for row in states}) == 1,
                "terminal rank state differs")
        save_status = None
        if rank == 0:
            try:
                adapter = _save_adapter_only(
                    model, source_root=Path(anchor["path"]), output=output_root / "adapter",
                )
                provisional = {
                    "schema": RECEIPT_SCHEMA,
                    "status": "unsealed_candidate",
                    "arm": arm, "mode": mode, "updates": updates,
                    "margin_weight": margin_weight,
                    "input": {"path": str(input_path), "sha256": file_hash(input_path)},
                    "manifest": packet["manifest"], "input_admission": packet["input_admission"],
                    "protocol": packet["protocol"], "code_identity": packet["code_identity"],
                    "change_map": packet["change_map"],
                    "base_input": packet["base_input"],
                    "base_engine": packet["base_engine"],
                    "margin_input": packet["margin_input"],
                    "parity_oracles": packet["parity_oracles"],
                    "resource_envelope": envelope_ref,
                    "source_adapter": packet["stable50_adapter"], "saved_adapter": adapter,
                    "composition": {
                        "base_model_path": packet["model_identity"]["base_model"],
                        "source_embedding": packet["source_embedding"],
                        "source_adapter": packet["stable50_adapter"],
                        "unmerged": True,
                        "dtype": "fp32",
                        "attention_implementation": "sdpa",
                    },
                    "optimizer": {**OPTIMIZER, "betas": list(OPTIMIZER["betas"])},
                    "clip_gradient_norm": CLIP_GRADIENT_NORM,
                    "rank_states": states,
                    "initial_positive_scores": initial_scores,
                    "final_live_positive_scores": final_scores,
                    "step_records": [{"step": step, "path": str(output_root / f"update-{step:02d}.json"),
                                      "sha256": file_hash(output_root / f"update-{step:02d}.json")}
                                     for step in range(1, updates + 1)],
                    "final_reference_records": [
                        {"rank": item_rank,
                         "path": str(output_root / "ranks" / f"rank{item_rank}" / "final-reference.json"),
                         "sha256": file_hash(output_root / "ranks" / f"rank{item_rank}" / "final-reference.json")}
                        for item_rank in range(WORLD_SIZE)
                    ],
                    "source_adapter_state_sha256": source_adapter_hash,
                    "final_adapter_state_sha256": states[0]["adapter_sha256"],
                    "stop_reason": "smoke_fixed_2_plumbing_only" if mode == "smoke" else "fixed_32_updates",
                }
                publish(output_root / "provisional.json", provisional)
                save_status = {"ok": True, "adapter_fingerprint": adapter["fingerprint"]}
            except BaseException as exc:
                save_status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        save_status = gather(save_status)[0]
        require(save_status["ok"], f"rank0 checkpoint publication failed: {save_status.get('error')}")
        dist.barrier()
        resource_observations.append(_capture_lifecycle_resources(
            device=device, started=started, phase="post_export_and_terminal_sync",
        ))
        lifecycle = combine_lifecycle_resource_observations(resource_observations)
        require(
            lifecycle["peak_cuda_allocated_bytes"] is not None
            and lifecycle["peak_cuda_reserved_bytes"] is not None
            and not lifecycle["cuda_measurement_errors"]
            and lifecycle["peak_cuda_allocated_bytes"] <= limits["max_cuda_allocated_bytes"]
            and lifecycle["peak_cuda_reserved_bytes"] <= limits["max_cuda_reserved_bytes"]
            and lifecycle["peak_rss_bytes"] <= limits["max_rss_bytes"]
            and lifecycle["elapsed_seconds"] <= limits["max_rank_seconds"],
            "full rank lifecycle resource envelope exceeded",
        )
        final_state = {**state, "success_gate_lifecycle_resources": lifecycle}
        status = "completed"
      except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
      finally:
        signal.alarm(0)
        resource_observations.append(_capture_lifecycle_resources(
            device=device, started=started, phase="terminal_finally",
        ))
        publish(run / "terminal.json", {
            "rank": rank, "status": status, "error": error,
            "arm": arm, "mode": mode, "margin_weight": margin_weight,
            "updates": counters["optimizer_steps"],
            "counters": counters, "state": final_state,
            "resource_limits": limits,
            "lifecycle_resources": combine_lifecycle_resource_observations(
                resource_observations
            ),
            "last_phase": active_phase if "active_phase" in locals() else "preload",
            "elapsed_seconds": time.monotonic() - started,
        })
        if dist_initialized:
            dist.destroy_process_group()


def _verify_identity(identity: Mapping[str, Any], *, label: str) -> None:
    root = Path(str(identity.get("root", "")))
    files = identity.get("files")
    require(root.is_dir() and isinstance(files, list) and files, f"{label} identity")
    require(
        all(file_hash(root / row["relative_path"]) == row["sha256"] for row in files),
        f"{label} payload bytes changed",
    )


def finalize(output_root: Path) -> dict[str, Any]:
    """Seal one successful torchrun without claiming a scientific result."""

    receipt_path = output_root / "receipt.json"
    require(not receipt_path.exists(), "occupied sealed receipt")
    exit_path = output_root / "launcher-exit.json"
    launcher_exit = load_json(exit_path)
    require(launcher_exit.get("exit_code") == 0, "torchrun launcher failed")
    provisional_path = output_root / "provisional.json"
    provisional = load_json(provisional_path)
    require(
        provisional.get("schema") == RECEIPT_SCHEMA
        and provisional.get("status") == "unsealed_candidate"
        and provisional.get("arm") in ARMS
        and provisional.get("mode") in ("smoke", "full"),
        "provisional schema/status",
    )
    arm, mode = provisional["arm"], provisional["mode"]
    margin_weight = provisional.get("margin_weight")
    updates = SMOKE_UPDATES if mode == "smoke" else FULL_UPDATES
    expected_backwards = 13
    require(
        arm == "C" and margin_weight in (0.0, MARGIN_WEIGHT)
        and (mode == "smoke" or margin_weight == MARGIN_WEIGHT)
        and provisional.get("updates") == updates
        and provisional.get("stop_reason") == (
            "smoke_fixed_2_plumbing_only" if mode == "smoke" else "fixed_32_updates"
        ),
        "fixed update stop",
    )
    rank_root = output_root / "ranks"
    require(
        rank_root.is_dir()
        and {path.name for path in rank_root.iterdir() if path.is_dir()}
        == {f"rank{rank}" for rank in range(WORLD_SIZE)},
        "exact rank0..rank7 coverage",
    )
    terminals, terminal_refs = [], []
    for rank in range(WORLD_SIZE):
        path = rank_root / f"rank{rank}" / "terminal.json"
        terminal = load_json(path)
        counters = terminal.get("counters", {})
        require(
            terminal.get("rank") == rank
            and terminal.get("status") == "completed"
            and terminal.get("arm") == arm
            and terminal.get("mode") == mode
            and terminal.get("margin_weight") == margin_weight
            and terminal.get("updates") == updates
            and counters.get("model_loads") == 1
            and counters.get("reference_forwards") == 10
            and counters.get("training_replays") == updates * expected_backwards
            and counters.get("backwards") == updates * expected_backwards
            and counters.get("synchronized_backwards") == updates
            and counters.get("optimizer_steps") == updates
            and counters.get("normal_kl_items") == updates * NORMALS_PER_RANK
            and counters.get("margin_items") == updates * NORMALS_PER_RANK
            and counters.get("final_reference_forwards") == NORMALS_PER_RANK
            and counters.get("negative_samples") == 0
            and counters.get("generation_calls") == 0
            and counters.get("positive_projection_replays") == (updates * POSITIVE_COUNT if rank == 0 else 0)
            and counters.get("positive_score_replays") == (2 * POSITIVE_COUNT if rank == 0 else 0),
            f"rank{rank} terminal counters",
        )
        state = terminal.get("state", {})
        lifecycle = terminal.get("lifecycle_resources", {})
        resource_limits = terminal.get("resource_limits", {})
        require(
            state.get("rank") == rank
            and state.get("adapter_sha256")
            and state.get("optimizer_sha256")
            and state.get("frozen_sha256")
            and state.get("activation_checkpointing", {}).get("model_eval") is True,
            f"rank{rank} terminal state",
        )
        require(
            lifecycle.get("observation_count", 0) >= 3
            and lifecycle.get("phases", [])[-1:] == ["terminal_finally"]
            and not lifecycle.get("cuda_measurement_errors")
            and type(lifecycle.get("peak_cuda_allocated_bytes")) is int
            and type(lifecycle.get("peak_cuda_reserved_bytes")) is int
            and lifecycle["peak_cuda_allocated_bytes"] <= resource_limits["max_cuda_allocated_bytes"]
            and lifecycle["peak_cuda_reserved_bytes"] <= resource_limits["max_cuda_reserved_bytes"]
            and lifecycle["peak_rss_bytes"] <= resource_limits["max_rss_bytes"]
            and lifecycle["elapsed_seconds"] <= resource_limits["max_rank_seconds"],
            f"rank{rank} terminal lifecycle resources",
        )
        terminals.append(terminal)
        terminal_refs.append({"rank": rank, "path": str(path), "sha256": file_hash(path)})
    require(
        len({row["state"]["adapter_sha256"] for row in terminals}) == 1
        and len({row["state"]["optimizer_sha256"] for row in terminals}) == 1
        and len({row["state"]["frozen_sha256"] for row in terminals}) == 1,
        "terminal distributed state identity",
    )
    final_refs = provisional.get("final_reference_records")
    require(
        isinstance(final_refs, list) and len(final_refs) == WORLD_SIZE
        and [row.get("rank") for row in final_refs] == list(range(WORLD_SIZE)),
        "final reference rank coverage",
    )
    final_reference_rows = []
    for reference, terminal in zip(final_refs, terminals, strict=True):
        require(file_hash(reference["path"]) == reference["sha256"],
                "final reference record changed")
        value = load_json(reference["path"])
        rows = value.get("records")
        require(
            value.get("schema") == "margin_preserved_train.final_reference.v1"
            and value.get("rank") == reference["rank"]
            and value.get("arm") == arm and value.get("mode") == mode
            and value.get("margin_weight") == margin_weight
            and value.get("post_update") == updates
            and isinstance(rows, list) and len(rows) == NORMALS_PER_RANK
            and value.get("state_unchanged", {}).get("adapter_sha256")
                == terminal["state"]["adapter_sha256"]
            and value.get("state_unchanged", {}).get("optimizer_sha256")
                == terminal["state"]["optimizer_sha256"],
            "final reference post-update identity",
        )
        final_reference_rows.extend(rows)
    require(
        len(final_reference_rows) == NORMAL_COUNT
        and sum(row["margin"]["eligible_count"] for row in final_reference_rows)
            == ELIGIBLE_MARGIN_COUNT,
        "final reference normal56/eligible6030",
    )
    final_reference_summary = {
        "post_update": updates,
        **summarize_final_reference(final_reference_rows, margin_weight=margin_weight),
    }
    records = provisional.get("step_records")
    require(
        isinstance(records, list)
        and len(records) == updates
        and all(
            row.get("step") == step and file_hash(row.get("path", "")) == row.get("sha256")
            for step, row in enumerate(records, 1)
        ),
        "global update records",
    )
    margin_summaries = []
    for row in records:
        update = load_json(row["path"])
        require(
            update.get("arm") == arm
            and update.get("mode") == mode
            and update.get("margin_weight") == margin_weight
            and len(update.get("ranks", [])) == WORLD_SIZE
            and update.get("event_denominator") == 0
            and update.get("event_count") == 0
            and update.get("raw_sampled_tokens") == 0
            and update.get("retained_event_tokens") == 0,
            "global update topology/denominator",
        )
        summary = update.get("margin_summary", {})
        require(summary.get("eligible_positions") == ELIGIBLE_MARGIN_COUNT,
                "update margin denominator")
        components = update.get("objective_components", {})
        require(set(components) == {"positive", "conditional_kl", "normal_kl", "margin"},
                "separate objective components")
        margin_summaries.append(summary)
    require(margin_summaries[0]["active_images"] == 0
            and margin_summaries[0]["active_positions"] == 0,
            "Stable50 initial margin value must be zero")
    _verify_identity(provisional["saved_adapter"], label="saved adapter")
    _verify_identity(provisional["composition"]["source_embedding"], label="source embedding")
    _verify_identity(provisional["composition"]["source_adapter"], label="Stable50 source adapter")
    require(
        provisional["saved_adapter"]["root"] == str(output_root / "adapter")
        and provisional["composition"]["unmerged"] is True
        and provisional["composition"]["dtype"] == "fp32"
        and provisional["composition"]["attention_implementation"] == "sdpa",
        "exported composition",
    )
    resources = {
        "model_loads": sum(row["counters"]["model_loads"] for row in terminals),
        "model_forwards": sum(row["counters"]["model_forwards"] for row in terminals),
        "image_forwards": sum(row["counters"]["image_forwards"] for row in terminals),
        "reference_forwards": sum(row["counters"]["reference_forwards"] for row in terminals),
        "training_replays": sum(row["counters"]["training_replays"] for row in terminals),
        "backwards": sum(row["counters"]["backwards"] for row in terminals),
        "synchronized_backwards": sum(row["counters"]["synchronized_backwards"] for row in terminals),
        "negative_samples": sum(row["counters"]["negative_samples"] for row in terminals),
        "normal_kl_items": sum(row["counters"]["normal_kl_items"] for row in terminals),
        "margin_items": sum(row["counters"]["margin_items"] for row in terminals),
        "final_reference_forwards": sum(
            row["counters"]["final_reference_forwards"] for row in terminals
        ),
        "active_margin_image_items": sum(
            row["counters"]["active_margin_images"] for row in terminals
        ),
        "raw_sampled_tokens": sum(row["counters"]["raw_sampled_tokens"] for row in terminals),
        "reference_cache_bytes": sum(row["state"]["reference_cache_bytes"] for row in terminals),
        "max_rank_seconds": max(row["lifecycle_resources"]["elapsed_seconds"] for row in terminals),
        "peak_cuda_allocated_bytes_max_rank": max(
            row["lifecycle_resources"]["peak_cuda_allocated_bytes"] for row in terminals
        ),
        "peak_cuda_reserved_bytes_max_rank": max(
            row["lifecycle_resources"]["peak_cuda_reserved_bytes"] for row in terminals
        ),
        "peak_rss_bytes_max_rank": max(
            row["lifecycle_resources"]["peak_rss_bytes"] for row in terminals
        ),
    }
    expected_model_forwards = 356 if mode == "smoke" else 3566
    require(
        resources["model_loads"] == 8
        and resources["reference_forwards"] == 80
        and resources["model_forwards"] == expected_model_forwards
        and resources["image_forwards"] == expected_model_forwards
        and resources["training_replays"] == updates * 13 * WORLD_SIZE
        and resources["backwards"] == updates * 13 * WORLD_SIZE
        and resources["synchronized_backwards"] == updates * WORLD_SIZE
        and resources["normal_kl_items"] == updates * NORMAL_COUNT
        and resources["margin_items"] == updates * NORMAL_COUNT
        and resources["final_reference_forwards"] == NORMAL_COUNT
        and resources["negative_samples"] == 0
        and resources["raw_sampled_tokens"] == 0,
        "global fixed runtime counters",
    )
    receipt = {
        **provisional,
        "status": "completed",
        "scientific_status": "candidate" if mode == "full" else "plumbing_only",
        "margin_signal": {
            "initial_zero_value": True,
            "post_first_step_active": any(
                summary["active_images"] > 0 for summary in margin_summaries[1:]
            ),
            "initial": margin_summaries[0],
            "last_training_pre_update": margin_summaries[-1],
            "final_post_update": final_reference_summary,
        },
        "rank_terminals": terminal_refs,
        "launcher_exit": {"path": str(exit_path), "sha256": file_hash(exit_path)},
        "provisional_sha256": file_hash(provisional_path),
        "resources": resources,
    }
    publish(receipt_path, receipt)
    return receipt


def verify_receipt(output_root: Path) -> dict[str, Any]:
    receipt_path = output_root / "receipt.json"
    receipt = load_json(receipt_path)
    require(
        receipt.get("schema") == RECEIPT_SCHEMA
        and receipt.get("status") == "completed"
        and receipt.get("arm") in ARMS
        and receipt.get("mode") in ("smoke", "full"),
        "sealed receipt schema/status",
    )
    expected = SMOKE_UPDATES if receipt["mode"] == "smoke" else FULL_UPDATES
    require(
        receipt.get("updates") == expected
        and receipt.get("margin_weight") in (0.0, MARGIN_WEIGHT)
        and (receipt["mode"] == "smoke" or receipt["margin_weight"] == MARGIN_WEIGHT),
        "sealed fixed update count/margin weight",
    )
    for key in ("input", "manifest", "input_admission", "protocol", "code_identity",
                "change_map", "base_input", "base_engine", "margin_input", "launcher_exit"):
        reference = receipt[key]
        require(file_hash(reference["path"]) == reference["sha256"], f"sealed {key}")
    require(file_hash(output_root / "provisional.json") == receipt["provisional_sha256"],
            "sealed provisional")
    for reference in [*receipt["rank_terminals"], *receipt["step_records"],
                      *receipt["final_reference_records"]]:
        require(file_hash(reference["path"]) == reference["sha256"], "sealed execution record")
    _verify_identity(receipt["saved_adapter"], label="saved adapter")
    _verify_identity(receipt["composition"]["source_embedding"], label="source embedding")
    _verify_identity(receipt["composition"]["source_adapter"], label="Stable50 source adapter")
    from src.adapters.dora import inspect_dora_adapter_payload
    observed = inspect_dora_adapter_payload(
        receipt["saved_adapter"]["root"], receipt["composition"]["base_model_path"],
    )
    require(observed == receipt["saved_adapter"], "saved adapter semantic fingerprint")
    validate_inputs(Path(receipt["input"]["path"]), verify_sources=True)
    if receipt["mode"] == "smoke" and receipt["margin_weight"] == 0.0:
        oracle = receipt["parity_oracles"]["retained_A2"]
        require(
            receipt["final_adapter_state_sha256"] == oracle["final_adapter_state_sha256"],
            "weight0 smoke differs from retained A2 adapter state",
        )
    return receipt


def launch(
    *, input_path: Path, arm: str, mode: str, output_root: Path,
    envelope_path: Path | None, margin_weight: float,
) -> int:
    require(
        arm in ARMS and margin_weight in (0.0, MARGIN_WEIGHT)
        and (mode == "smoke" or margin_weight == MARGIN_WEIGHT)
        and mode in ("smoke", "full")
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3,4,5,6,7"
        and not output_root.exists(),
        "launch requires exact free eight-GPU arm root",
    )
    packet, _ = validate_inputs(input_path, verify_sources=True)
    _resource_limits(packet, arm=arm, mode=mode, envelope_path=envelope_path)
    output_root.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        "--nproc-per-node=8", "-m", "probes.dora_owner_learning.margin_preserved_train",
        "rank", "--input", str(input_path), "--arm", arm, "--mode", mode,
        "--margin-weight", str(margin_weight), "--output-root", str(output_root),
    ]
    if envelope_path is not None:
        command.extend(("--resource-envelope", str(envelope_path)))
    publish(output_root / "launcher-owner.json", {
        "pid": os.getpid(), "started": time.time(), "command": command,
        "input_sha256": file_hash(input_path), "arm": arm, "mode": mode,
        "margin_weight": margin_weight,
    })
    started = time.monotonic()
    with (output_root / "torchrun.log").open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    publish(output_root / "launcher-exit.json", {
        "exit_code": result.returncode, "elapsed_seconds": time.monotonic() - started,
        "finished": time.time(),
    })
    if result.returncode:
        return result.returncode
    receipt = finalize(output_root)
    print(json.dumps({"status": receipt["status"], "arm": arm, "mode": mode,
                      "margin_weight": margin_weight,
                      "resources": receipt["resources"]}))
    return 0


def cold_check(*, input_path: Path, output_root: Path) -> dict[str, Any]:
    """Cold-load the exported adapter once and replay the three positive rows."""

    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
        and not (output_root / "cold-check.json").exists(),
        "cold check requires one visible GPU and a fresh receipt adjunct",
    )
    receipt = verify_receipt(output_root)
    packet, manifest = validate_inputs(input_path, verify_sources=True)
    require(receipt["input"]["sha256"] == file_hash(input_path), "cold input/training input")
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_research_infer_config
    from src.data import load_raw_examples
    from src.inference.runtime import assemble_frontend
    from .runtime import load_policy

    started = time.monotonic()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    base_config = load_research_infer_config(CONFIG).config
    require(base_config.model_dump(mode="json") == packet["config"], "cold base config")
    config = checkpoint_config(base_config, receipt["saved_adapter"]["root"])
    require(config.embedding_delta is not None and
            str(config.embedding_delta.path) == packet["source_embedding"]["root"],
            "cold source embedding config")
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")),
    )
    raw = {str(row.example_id): row for row in load_raw_examples(config.data.input_jsonl)}
    qwen, identity = load_policy(config, device=device)
    model = qwen.model
    model.eval()
    from src.qwen.special_token_embeddings import inspect_special_token_embedding_delta_payload
    live_embedding = inspect_special_token_embedding_delta_payload(
        identity["model_identity"]["embedding_delta"]["identity"]["delta_path"],
        receipt["composition"]["base_model_path"],
    )
    composition_check = loaded_composition_evidence(
        loaded_identity=identity,
        expected_base=receipt["composition"]["base_model_path"],
        expected_adapter=receipt["saved_adapter"]["root"],
        expected_embedding=receipt["composition"]["source_embedding"],
        inspected_embedding=live_embedding,
    )
    require(composition_check["passed"], "cold exported composition")
    entries = []
    for case in manifest["positives"]:
        example_id = str(case["image"]["row_id"])
        entries.append(_materialize_case(
            qwen=qwen, frontend=frontend, config=config, raw=raw[example_id],
            case=case, positive=True,
        ))
    observed = _score_positive_routes(model, entries)
    expected = receipt["final_live_positive_scores"]
    errors: dict[str, dict[str, float]] = {}
    for candidate_id in SELECTED:
        require(
            observed[candidate_id]["token_count"] == expected[candidate_id]["token_count"]
            and observed[candidate_id]["argmax_target_tokens"]
            == expected[candidate_id]["argmax_target_tokens"],
            "cold discrete positive score mismatch",
        )
        errors[candidate_id] = {
            key: observed[candidate_id][key] - expected[candidate_id][key]
            for key in ("sum_logprob", "mean_logprob", "mean_target_margin", "min_target_margin")
        }
    require(
        all(abs(value) <= 1e-5 for row in errors.values() for value in row.values()),
        "cold/live positive scores differ",
    )
    result = {
        "schema": "margin_preserved_train.cold_check.v1",
        "status": "passed", "arm": receipt["arm"], "mode": receipt["mode"],
        "training_receipt": {"path": str(output_root / "receipt.json"),
                             "sha256": file_hash(output_root / "receipt.json")},
        "saved_adapter": receipt["saved_adapter"],
        "model_identity": identity,
        "positive_scores": observed, "live_score_deltas": errors,
        "model_loads": 1, "score_forwards": POSITIVE_COUNT,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "elapsed_seconds": time.monotonic() - started,
    }
    require(
        result["peak_cuda_allocated_bytes"] <= SMOKE_MAX_CUDA_BYTES
        and result["peak_cuda_reserved_bytes"] <= SMOKE_MAX_CUDA_BYTES
        and result["peak_rss_bytes"] <= SMOKE_MAX_RSS_BYTES
        and result["elapsed_seconds"] <= SMOKE_MAX_RANK_SECONDS,
        "cold-check resource ceiling",
    )
    publish(output_root / "cold-check.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "prepare", "rank", "launch", "finalize", "verify", "cold-check",
    ))
    parser.add_argument("--base-input", type=Path, default=BASE_INPUT)
    parser.add_argument("--margin-input", type=Path, default=MARGIN_INPUT)
    parser.add_argument("--input", type=Path, default=PREPARATION / "inputs.json")
    parser.add_argument("--output", type=Path, default=PREPARATION)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--mode", choices=("smoke", "full"))
    parser.add_argument("--margin-weight", type=float)
    parser.add_argument("--resource-envelope", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.base_input, args.margin_input, args.output), indent=2))
        return
    require(args.output_root is not None, "--output-root is required")
    if args.command in ("rank", "launch"):
        require(args.arm is not None and args.mode is not None and args.margin_weight is not None,
                "--arm/--mode/--margin-weight are required")
    if args.command == "rank":
        execute_rank(input_path=args.input, arm=args.arm, mode=args.mode,
                     output_root=args.output_root, envelope_path=args.resource_envelope,
                     margin_weight=args.margin_weight)
    elif args.command == "launch":
        raise SystemExit(launch(input_path=args.input, arm=args.arm, mode=args.mode,
                                output_root=args.output_root,
                                envelope_path=args.resource_envelope,
                                margin_weight=args.margin_weight))
    elif args.command == "finalize":
        print(json.dumps(finalize(args.output_root), indent=2))
    elif args.command == "verify":
        receipt = verify_receipt(args.output_root)
        print(json.dumps({"status": receipt["status"], "arm": receipt["arm"],
                          "mode": receipt["mode"], "resources": receipt["resources"]}, indent=2))
    else:
        print(json.dumps(cold_check(input_path=args.input, output_root=args.output_root), indent=2))


if __name__ == "__main__":
    main()
