#!/usr/bin/env python3
"""Score exact natural rows before and after one natural donor row.

This is a deliberately small, experiment-local scorer for the research unit
``2026-07-17-next-row-probability-transition-and-causal-source-trace``.  The
manifest supplies all token intervals and ownership labels.  The script never
rebuilds a row from text or coordinates: it slices the exact token IDs from a
recorded native source bundle and proves the resulting hashes before model
execution.

The output contains paired raw teacher-forced log-likelihood changes.  It does
not construct a common normalized probability matrix.  Terminal behavior is a
separate one-token boundary score and is never compared numerically with a
full-row score.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.run_fixed_encoding_object_centered_spatial_eligibility_crossover import (  # noqa: E402
    derive_explicit_position_ids,
)
from scripts.research.run_native_sibling_branch_replay import (  # noqa: E402
    _attention_implementation,
    _model_dtype_summary as _runtime_model_dtype_summary,
    SCHEMA_VERSION as _NATIVE_SIBLING_SCHEMA_VERSION,
    complete_row_spans,
)


SCHEMA_VERSION = "next_row_likelihood_change"
MANIFEST_SCHEMA_VERSION = "next_row_likelihood_change.manifest.v1"
RECEIPT_SCHEMA_VERSION = "next_row_likelihood_change.receipt.v1"
NATIVE_CALL_BUNDLE_SCHEMA_VERSION = f"{_NATIVE_SIBLING_SCHEMA_VERSION}.call_bundle.v1"

OBJECT_REF_START = 151646
OBJECT_REF_END = 151647
BOX_START = 151648
BOX_END = 151649
COORDINATE_TOKEN_START = 151670
COORDINATE_TOKEN_END_EXCLUSIVE = 152670

DEFAULT_CONFIG = Path(
    "/data/CoordExp/.worktrees/research-probes/configs/coordexp_swift/infer/"
    "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200.yaml"
)
DEFAULT_SOURCE_JSONL = Path(
    "/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/"
    "val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl"
)


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _decode_payload(bundle: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = bundle.get("decode_result", bundle)
    if not isinstance(payload, Mapping):
        raise ValueError("source bundle decode_result must be a JSON object")
    return payload


def _token_ids(bundle: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    payload = _decode_payload(bundle)
    prompt = payload.get("prompt_token_ids")
    generated = payload.get("generated_token_ids")
    if not isinstance(prompt, list) or not isinstance(generated, list):
        raise ValueError("source bundle must contain prompt_token_ids and generated_token_ids")
    return [int(value) for value in prompt], [int(value) for value in generated]


def _bundle_image_id(bundle: Mapping[str, Any]) -> str | None:
    payload = _decode_payload(bundle)
    if bundle.get("image_id") is not None:
        return str(bundle["image_id"])
    execution = bundle.get("execution_evidence")
    if isinstance(execution, Mapping) and execution.get("image_id") is not None:
        return str(execution["image_id"])
    if payload.get("image_id") is not None:
        return str(payload["image_id"])
    return None


def _bundle_execution_value(bundle: Mapping[str, Any], key: str) -> Any:
    donor = bundle.get("donor")
    if isinstance(donor, Mapping):
        direct_key = {
            "source_width": "source_image_width",
            "source_height": "source_image_height",
        }.get(key, key)
        if donor.get(direct_key) is not None:
            return donor.get(direct_key)
        lineage = donor.get("donor_lineage")
        if isinstance(lineage, Mapping) and lineage.get(key) is not None:
            return lineage.get(key)
    runtime = bundle.get("runtime")
    if isinstance(runtime, Mapping) and runtime.get(key) is not None:
        return runtime.get(key)
    execution = bundle.get("execution_evidence")
    if isinstance(execution, Mapping):
        return execution.get(key)
    return None


def _canonical_row_phases(row_tokens: Sequence[int]) -> dict[str, list[int]]:
    """Return row-relative phases, keeping closure separate from geometry."""

    row = [int(value) for value in row_tokens]
    if not row or row[0] != OBJECT_REF_START or row[-1] != BOX_END:
        raise ValueError("natural row has non-canonical object/box wrappers")
    try:
        object_end = row.index(OBJECT_REF_END, 1)
        box_start = row.index(BOX_START, object_end + 1)
    except ValueError as exc:
        raise ValueError("natural row lacks object-ref or box-start marker") from exc
    if object_end <= 1 or box_start != object_end + 1:
        raise ValueError("natural row has malformed description or box boundary")
    description = row[1:object_end]
    if any(
        value in {OBJECT_REF_START, OBJECT_REF_END, BOX_START, BOX_END}
        or COORDINATE_TOKEN_START <= value < COORDINATE_TOKEN_END_EXCLUSIVE
        for value in description
    ):
        raise ValueError("natural row description contains a wrapper or coordinate token")
    coord_start = box_start + 1
    coord_end = coord_start + 4
    if coord_end != len(row) - 1:
        raise ValueError("natural row must contain exactly four coordinate tokens")
    coords = row[coord_start:coord_end]
    if any(not COORDINATE_TOKEN_START <= value < COORDINATE_TOKEN_END_EXCLUSIVE for value in coords):
        raise ValueError("natural row contains a non-coordinate geometry token")
    return {
        "row_entry": [0],
        "description": list(range(1, object_end)),
        "geometry": [box_start, *range(coord_start, coord_end)],
        "x1": [coord_start],
        "y1": [coord_start + 1],
        "x2": [coord_start + 2],
        "y2": [coord_start + 3],
        "closure": [len(row) - 1],
        "full_row": list(range(len(row))),
    }


def score_token_logits(
    logits: torch.Tensor,
    *,
    boundary_length: int,
    row_tokens: Sequence[int],
    terminal_token_id: int | None = None,
) -> dict[str, Any]:
    """Score one exact row from logits using float32 log-softmax.

    The return value deliberately contains raw sums, means, and token counts,
    but no row-normalized probability distribution across candidates.
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape [sequence, vocabulary]")
    row = [int(value) for value in row_tokens]
    if not row:
        raise ValueError("row_tokens must not be empty")
    if int(logits.shape[0]) < int(boundary_length) + len(row) - 1:
        raise ValueError("logits do not cover all teacher-forced row positions")
    phases = _canonical_row_phases(row)
    log_probs = torch.log_softmax(logits.to(dtype=torch.float32), dim=-1)
    selected = torch.stack(
        [log_probs[int(boundary_length) + index - 1, token] for index, token in enumerate(row)]
    )
    result: dict[str, Any] = {
        "token_count": len(row),
        "token_ids_sha256": sha256_json(row),
        "token_log_probabilities": [float(value) for value in selected.detach().cpu().tolist()],
    }
    for phase, indices in phases.items():
        values = selected[torch.tensor(indices, dtype=torch.long, device=selected.device)]
        result[phase] = {
            "sum": float(values.sum().item()),
            "mean": float(values.mean().item()),
            "count": int(len(indices)),
        }
    if terminal_token_id is not None:
        boundary_log_probs = log_probs[int(boundary_length) - 1]
        row_entry = boundary_log_probs[int(row[0])]
        terminal = boundary_log_probs[int(terminal_token_id)]
        result["row_entry_vs_terminal"] = {
            "row_entry_log_probability": float(row_entry.item()),
            "terminal_log_probability": float(terminal.item()),
            "row_entry_minus_terminal": float((row_entry - terminal).item()),
        }
    return result


def terminal_boundary_score(
    logits: torch.Tensor, *, boundary_length: int, row_entry_token_id: int, terminal_token_id: int
) -> dict[str, Any]:
    """Return the one-token row-entry versus terminal margin at a boundary."""

    if logits.ndim != 2 or int(boundary_length) <= 0 or int(logits.shape[0]) < int(boundary_length):
        raise ValueError("boundary logits do not cover the requested final position")
    values = torch.log_softmax(logits.to(dtype=torch.float32), dim=-1)[int(boundary_length) - 1]
    row_entry = values[int(row_entry_token_id)]
    terminal = values[int(terminal_token_id)]
    return {
        "row_entry_token_id": int(row_entry_token_id),
        "terminal_token_id": int(terminal_token_id),
        "row_entry_log_probability": float(row_entry.item()),
        "terminal_log_probability": float(terminal.item()),
        "row_entry_minus_terminal": float((row_entry - terminal).item()),
    }


def paired_score_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Subtract the same frozen candidate score before and after a donor."""

    if before.get("token_ids_sha256") != after.get("token_ids_sha256"):
        raise ValueError("before and after scores refer to different candidate rows")
    if int(before.get("token_count", -1)) != int(after.get("token_count", -2)):
        raise ValueError("before and after candidate token counts differ")
    phases = ("row_entry", "description", "geometry", "x1", "y1", "x2", "y2", "closure", "full_row")
    output: dict[str, Any] = {
        "token_ids_sha256": before["token_ids_sha256"],
        "token_count": int(before["token_count"]),
    }
    for phase in phases:
        left = before.get(phase)
        right = after.get(phase)
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise ValueError(f"missing phase {phase} in before/after score")
        output[phase] = {
            "sum_delta": float(right["sum"]) - float(left["sum"]),
            "mean_delta": float(right["mean"]) - float(left["mean"]),
            "count": int(left["count"]),
        }
    return output


def emit_candidate_score(score: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Attach candidate provenance while keeping terminal state out of row scores."""

    if "row_entry_vs_terminal" in score:
        raise ValueError("candidate row score must not contain terminal boundary state")
    return {
        **dict(score),
        "candidate_id": str(candidate["candidate_id"]),
        "owner": str(candidate["owner"]),
        "category": str(candidate["category"]),
        "role": str(candidate["role"]),
        "source_bundle": str(candidate["source_bundle"]),
        "source_bundle_sha256": str(candidate["source_bundle_sha256"]),
        "source_token_span": list(candidate["source_token_span"]),
        "source_prompt_ids_sha256": str(candidate["source_prompt_ids_sha256"]),
        "source_prompt_matches_base_prompt": bool(candidate["source_prompt_matches_base_prompt"]),
        "source_prompt_matches_reconstructed_parent": bool(candidate["source_prompt_matches_reconstructed_parent"]),
        "token_ids": [int(value) for value in candidate["token_ids"]],
        # Keep the score phase names ``description`` and ``geometry`` intact.
        # Candidate metadata uses explicit names so it cannot overwrite those
        # phase mappings when the score is later paired before/after a donor.
        "description_text": candidate["description"],
        "geometry_xyxy": candidate["geometry"],
        "natural_support": candidate["natural_support"],
        "token_count": int(candidate["token_count"]),
    }


def object_specific_crossover(
    donor_score: Mapping[str, Any],
    candidate_score: Mapping[str, Any],
    donor_before: Mapping[str, Any],
    candidate_before: Mapping[str, Any],
    *,
    donor_category: str,
    candidate_category: str,
    donor_owner: str,
    candidate_owner: str,
) -> dict[str, Any]:
    """Remove a generic continuation shift using the declared difference-in-differences."""

    donor_delta = paired_score_delta(donor_before, donor_score)
    candidate_delta = paired_score_delta(candidate_before, candidate_score)
    category_matched = str(donor_category) == str(candidate_category)
    token_length_matched = int(donor_before["token_count"]) == int(candidate_before["token_count"])
    distinct_owner = str(donor_owner) != str(candidate_owner)
    ineligibility_reasons: list[str] = []
    if not distinct_owner:
        ineligibility_reasons.append("same_physical_owner")
    if not category_matched:
        ineligibility_reasons.append("category_mismatch")
    if not token_length_matched:
        ineligibility_reasons.append("token_length_mismatch")
    donor_score_id = donor_score.get("candidate_id")
    candidate_score_id = candidate_score.get("candidate_id")
    if donor_score_id is not None and donor_score_id == candidate_score_id:
        ineligibility_reasons.append("same_candidate_id")
    conclusion_eligible = not ineligibility_reasons
    output: dict[str, Any] = {
        "category_matched": category_matched,
        "token_length_matched": token_length_matched,
        "distinct_owner": distinct_owner,
        "ineligibility_reasons": ineligibility_reasons,
        "conclusion_eligible": conclusion_eligible,
        "comparison_scope": "conclusion_eligible" if conclusion_eligible else "descriptive_only",
    }
    for phase in ("description", "geometry", "x1", "y1", "x2", "y2", "closure", "full_row"):
        values = {
            "candidate_minus_donor_delta_sum": float(candidate_delta[phase]["sum_delta"])
            - float(donor_delta[phase]["sum_delta"]),
            "candidate_minus_donor_delta_mean": float(candidate_delta[phase]["mean_delta"])
            - float(donor_delta[phase]["mean_delta"]),
        }
        output[phase] = values if conclusion_eligible else {"descriptive_only": values}
    return output


def compute_object_specific_crossover_scores(
    candidates: Sequence[Mapping[str, Any]],
    *,
    donor_candidate_id: str,
    before_scores: Mapping[str, Mapping[str, Any]],
    after_scores: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute crossover only after every candidate has an after-score.

    The donor row may be any candidate variant, not necessarily the first one
    in manifest order.  Keeping this pass separate from model scoring prevents
    candidate order from becoming a hidden runtime dependency.
    """

    donor_id = str(donor_candidate_id)
    donor_score = after_scores.get(donor_id)
    donor_before = before_scores.get(donor_id)
    if donor_score is None or donor_before is None:
        raise ValueError(f"donor candidate {donor_id} lacks before/after score")
    donor_variant = next(
        (item for item in candidates if str(item.get("candidate_id")) == donor_id),
        None,
    )
    if donor_variant is None:
        raise ValueError(f"donor candidate {donor_id} is absent from candidate variants")
    output: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        candidate_score = after_scores.get(candidate_id)
        candidate_before = before_scores.get(candidate_id)
        if candidate_score is None or candidate_before is None:
            raise ValueError(f"candidate {candidate_id} lacks before/after score")
        output[candidate_id] = object_specific_crossover(
            donor_score=donor_score,
            candidate_score=candidate_score,
            donor_before=donor_before,
            candidate_before=candidate_before,
            donor_category=str(donor_variant["category"]),
            candidate_category=str(candidate["category"]),
            donor_owner=str(donor_variant["owner"]),
            candidate_owner=str(candidate["owner"]),
        )
    return output


def aggregate_frozen_owner_scores(scores: Sequence[Mapping[str, Any]], *, phase: str = "full_row") -> dict[str, Any]:
    """Expose a declared log-sum-exp over exactly frozen variants of one owner."""

    if not scores:
        raise ValueError("owner aggregation requires at least one frozen variant")
    owners = {str(item.get("owner")) for item in scores}
    if len(owners) != 1:
        raise ValueError("owner aggregation received multiple owners")
    values = [float(item[phase]["sum"]) for item in scores]
    return {
        "owner": next(iter(owners)),
        "aggregation": "logsumexp_over_frozen_variants",
        "variant_ids": [str(item.get("candidate_id")) for item in scores],
        "variant_token_ids_sha256": [str(item.get("token_ids_sha256")) for item in scores],
        "constituents": [
            {
                "candidate_id": str(item.get("candidate_id")),
                "owner": str(item.get("owner")),
                "token_count": int(item.get("token_count", 0)),
                "token_ids_sha256": str(item.get("token_ids_sha256")),
                "phase_sum": float(item[phase]["sum"]),
            }
            for item in scores
        ],
        "constituent_sums": values,
        "logsumexp_sum": float(torch.logsumexp(torch.tensor(values, dtype=torch.float32), dim=0).item()),
        "variant_count": len(values),
        "phase": phase,
    }


def paired_owner_aggregate_delta(
    before: Mapping[str, Any], after: Mapping[str, Any], *, phase: str = "full_row"
) -> dict[str, Any]:
    """Compare two declared frozen-variant owner aggregates, not probabilities."""

    if str(before.get("owner")) != str(after.get("owner")):
        raise ValueError("owner aggregate identities differ")
    if str(before.get("phase")) != phase or str(after.get("phase")) != phase:
        raise ValueError("owner aggregate phase mismatch")
    return {
        "owner": str(before["owner"]),
        "aggregation": "paired_logsumexp_over_same_frozen_variants",
        "phase": phase,
        "variant_ids_before": list(before.get("variant_ids", [])),
        "variant_ids_after": list(after.get("variant_ids", [])),
        "sum_delta": float(after["logsumexp_sum"]) - float(before["logsumexp_sum"]),
    }


def _resolve_manifest_path(raw: str | Path, manifest_path: Path) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else manifest_path.parent / path).resolve(strict=True)


def _require_text(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} requires non-empty {key}")
    return value


def validate_frozen_manifest(manifest: Mapping[str, Any], *, manifest_path: Path) -> dict[str, Any]:
    """Load source bundles and prove parent/candidate token lineage."""

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"expected {MANIFEST_SCHEMA_VERSION}")
    image_id = _require_text(manifest, "image_id", "manifest")
    parent = manifest.get("parent")
    if not isinstance(parent, Mapping):
        raise ValueError("manifest requires parent object")
    parent_path = _resolve_manifest_path(_require_text(parent, "source_bundle", "parent"), manifest_path)
    parent_bundle = _read_json(parent_path)
    base_prompt, generated = _token_ids(parent_bundle)
    prefix_count = parent.get("prefix_token_count")
    if not isinstance(prefix_count, int) or prefix_count < 0 or prefix_count > len(generated):
        raise ValueError("parent prefix_token_count is invalid")
    prefix = generated[:prefix_count]
    if prefix_count and (not complete_row_spans(prefix) or complete_row_spans(prefix)[-1][1] != prefix_count):
        raise ValueError("parent prefix must end at a complete native row")
    expected_prompt = [*base_prompt, *prefix]
    expected = {
        "prompt_token_ids": base_prompt,
        "prompt_token_ids_sha256": sha256_json(base_prompt),
        "prefix_token_ids": prefix,
        "prefix_token_ids_sha256": sha256_json(prefix),
        "prefix_token_count": prefix_count,
        "reconstructed_prompt_token_ids": expected_prompt,
        "reconstructed_prompt_token_ids_sha256": sha256_json(expected_prompt),
        "source_bundle": str(parent_path),
        "source_bundle_sha256": sha256_file(parent_path),
    }
    for key in ("source_bundle_sha256", "prefix_token_ids_sha256"):
        declared = parent.get(key)
        if not isinstance(declared, str) or declared != expected[key]:
            raise ValueError(f"parent {key} does not match source bundle")
    if parent.get("prompt_token_ids_sha256") not in {None, expected["prompt_token_ids_sha256"]}:
        raise ValueError("parent prompt_token_ids_sha256 does not match source bundle")
    if parent.get("reconstructed_prompt_token_ids_sha256") not in {None, expected["reconstructed_prompt_token_ids_sha256"]}:
        raise ValueError("parent reconstructed prompt hash does not match source bundle")
    if _bundle_image_id(parent_bundle) not in {None, image_id}:
        raise ValueError("parent source bundle image does not match manifest")
    parent_source_image_sha256 = _bundle_execution_value(parent_bundle, "source_image_sha256")

    variants = manifest.get("candidate_variants", manifest.get("candidates"))
    donors = manifest.get("donors")
    if not isinstance(variants, list) or not variants:
        raise ValueError("manifest requires non-empty candidate_variants list")
    if not isinstance(donors, list) or not donors:
        raise ValueError("manifest requires non-empty donors list")
    normalized_variants: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    for index, item in enumerate(variants):
        if not isinstance(item, Mapping):
            raise ValueError(f"candidate_variants[{index}] must be an object")
        candidate = dict(item)
        candidate_id = _require_text(candidate, "candidate_id", f"candidate_variants[{index}]")
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate_id {candidate_id}")
        candidate_ids.add(candidate_id)
        source = _resolve_manifest_path(_require_text(candidate, "source_bundle", candidate_id), manifest_path)
        bundle = _read_json(source)
        source_hash = sha256_file(source)
        declared_source_hash = _require_text(candidate, "source_bundle_sha256", candidate_id)
        if declared_source_hash != source_hash:
            raise ValueError(f"{candidate_id} source_bundle_sha256 mismatch")
        source_prompt, source_generated = _token_ids(bundle)
        if source_prompt != base_prompt and source_prompt != expected_prompt:
            raise ValueError(f"{candidate_id} source prompt is outside parent lineage")
        candidate_source_image_sha256 = _bundle_execution_value(bundle, "source_image_sha256")
        if (
            parent_source_image_sha256 is not None
            and candidate_source_image_sha256 is not None
            and candidate_source_image_sha256 != parent_source_image_sha256
        ):
            raise ValueError(f"{candidate_id} source image digest disagrees with parent lineage")
        if _bundle_image_id(bundle) not in {None, image_id}:
            raise ValueError(f"{candidate_id} source bundle image does not match manifest")
        span = candidate.get("source_token_span")
        if not isinstance(span, list) or len(span) != 2 or any(not isinstance(v, int) for v in span):
            raise ValueError(f"{candidate_id} requires integer source_token_span [start,end]")
        start, end = span
        if (start, end) not in complete_row_spans(source_generated):
            raise ValueError(f"{candidate_id} source span is not one complete native row")
        expected_start = prefix_count if source_prompt == base_prompt else 0
        if start != expected_start:
            raise ValueError(f"{candidate_id} row is not contiguous with the declared parent")
        row = [int(value) for value in source_generated[start:end]]
        token_hash = sha256_json(row)
        if _require_text(candidate, "token_ids_sha256", candidate_id) != token_hash:
            raise ValueError(f"{candidate_id} token_ids_sha256 mismatch")
        _canonical_row_phases(row)
        for field in ("owner", "category", "role"):
            _require_text(candidate, field, candidate_id)
        for field in ("description", "geometry", "natural_support"):
            if field not in candidate:
                raise ValueError(f"{candidate_id} requires {field} metadata")
        support = candidate["natural_support"]
        if not isinstance(support, Mapping):
            raise ValueError(f"{candidate_id} natural_support must be an object")
        for field in ("source", "count", "policy"):
            if field not in support:
                raise ValueError(f"{candidate_id} natural_support requires {field}")
        normalized_variants.append({
            **candidate,
            "candidate_id": candidate_id,
            "source_bundle": str(source),
            "source_bundle_sha256": source_hash,
            "source_prompt_kind": "base_prompt" if source_prompt == base_prompt else "reconstructed_parent_prompt",
            "source_prompt_ids_sha256": sha256_json(source_prompt),
            "source_prompt_matches_base_prompt": source_prompt == base_prompt,
            "source_prompt_matches_reconstructed_parent": source_prompt == expected_prompt,
            "source_token_span": [start, end],
            "token_ids": row,
            "token_ids_sha256": token_hash,
            "token_count": len(row),
        })
    normalized_donors: list[dict[str, Any]] = []
    donor_ids: set[str] = set()
    for index, item in enumerate(donors):
        if not isinstance(item, Mapping):
            raise ValueError(f"donors[{index}] must be an object")
        donor = dict(item)
        donor_id = _require_text(donor, "donor_id", f"donors[{index}]")
        if donor_id in donor_ids:
            raise ValueError(f"duplicate donor_id {donor_id}")
        donor_ids.add(donor_id)
        candidate_id = _require_text(donor, "candidate_id", donor_id)
        referenced = next((item for item in normalized_variants if str(item["candidate_id"]) == candidate_id), None)
        if referenced is None:
            raise ValueError(f"{donor_id} references unknown candidate_id {candidate_id}")
        for field in ("owner", "category", "role"):
            _require_text(donor, field, donor_id)
            if str(donor[field]) != str(referenced[field]):
                raise ValueError(f"{donor_id} {field} disagrees with candidate {candidate_id}")
        normalized_donors.append({**donor, "donor_id": donor_id, "candidate_id": candidate_id})
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "image_id": image_id,
        "parent": expected,
        "parent_bundle": parent_bundle,
        "candidate_variants": normalized_variants,
        "donors": normalized_donors,
    }


def validate_active_source_lineage(
    bundle: Mapping[str, Any],
    *,
    image_id: str,
    image_sha256: str,
    image_width: int,
    image_height: int,
    model_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    attention_implementation: str,
    model_config_dtype: str,
) -> dict[str, Any]:
    """Require the executed native sibling call-bundle lineage.

    This check intentionally targets the current native sibling replay schema.
    Historical bundles with a top-level ``execution_evidence`` object are not
    accepted for conclusion-owning scores because they do not prove that the
    same call, runtime, and source image were executed.
    """

    if bundle.get("schema_version") != NATIVE_CALL_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            "source bundle must use native sibling call-bundle schema"
        )
    persisted_image_id = bundle.get("image_id")
    if persisted_image_id is None or str(persisted_image_id) != str(image_id):
        raise ValueError("source bundle top-level image_id does not match active image")

    donor = bundle.get("donor")
    if not isinstance(donor, Mapping):
        raise ValueError("source bundle lacks donor lineage")
    donor_lineage = donor.get("donor_lineage")
    if not isinstance(donor_lineage, Mapping):
        raise ValueError("source bundle donor lacks donor_lineage")
    source_values = (
        ("source_image_sha256", donor.get("source_image_sha256"), image_sha256),
        ("source_image_width", donor.get("source_image_width"), int(image_width)),
        ("source_image_height", donor.get("source_image_height"), int(image_height)),
    )
    for key, actual, expected in source_values:
        if actual is None or (str(actual) if isinstance(expected, str) else int(actual)) != expected:
            raise ValueError(f"source bundle lacks or mismatches {key}")
    lineage_values = (
        ("image_id", donor_lineage.get("image_id"), str(image_id)),
        ("source_image_sha256", donor_lineage.get("source_image_sha256"), image_sha256),
        ("source_width", donor_lineage.get("source_width"), int(image_width)),
        ("source_height", donor_lineage.get("source_height"), int(image_height)),
    )
    for key, actual, expected in lineage_values:
        if actual is None or (str(actual) if isinstance(expected, str) else int(actual)) != expected:
            raise ValueError(f"source donor_lineage lacks or mismatches {key}")

    runtime = bundle.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("source bundle lacks runtime lineage")
    payload = _decode_payload(bundle)
    runtime_model = runtime.get("model_identity")
    runtime_tokenizer = runtime.get("tokenizer_identity")
    if not isinstance(runtime_model, Mapping) or not isinstance(runtime_tokenizer, Mapping):
        raise ValueError("source runtime lacks model_identity or tokenizer_identity")
    if dict(runtime_model) != dict(model_identity):
        raise ValueError("source model identity disagrees with active runtime")
    if dict(runtime_tokenizer) != dict(tokenizer_identity):
        raise ValueError("source tokenizer identity disagrees with active runtime")
    persisted_model = payload.get("model_identity")
    persisted_tokenizer = payload.get("tokenizer_identity")
    if not isinstance(persisted_model, Mapping) or not isinstance(persisted_tokenizer, Mapping):
        raise ValueError("source decode result lacks model_identity or tokenizer_identity")
    if dict(persisted_model) != dict(runtime_model):
        raise ValueError("source decode model identity disagrees with runtime")
    if dict(persisted_tokenizer) != dict(runtime_tokenizer):
        raise ValueError("source decode tokenizer identity disagrees with runtime")
    runtime_attention = runtime.get("attention_implementation")
    if runtime_attention is None or str(runtime_attention) != str(attention_implementation):
        raise ValueError("source runtime attention implementation disagrees with active runtime")
    runtime_config_dtype = runtime.get("model_config_dtype")
    if runtime_config_dtype is None or str(runtime_config_dtype) != str(model_config_dtype):
        raise ValueError("source runtime model_config_dtype disagrees with active runtime")
    receipt = payload.get("execution_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("source decode result lacks execution_receipt")
    if receipt.get("schema_version") != "decode_execution_receipt.v1":
        raise ValueError("source execution_receipt schema is not decode_execution_receipt.v1")
    receipt_attention = receipt.get("attention_implementation")
    if receipt_attention is None or str(receipt_attention) != str(runtime_attention):
        raise ValueError("source execution receipt attention disagrees with runtime")
    lineage_attention = donor_lineage.get("attention_implementation")
    if lineage_attention is None or str(lineage_attention) != str(attention_implementation):
        raise ValueError("source donor_lineage attention disagrees with active runtime")
    model_hash = sha256_json(dict(model_identity))
    tokenizer_hash = sha256_json(dict(tokenizer_identity))
    if donor_lineage.get("model_identity_sha256") != model_hash:
        raise ValueError("source donor_lineage model identity hash disagrees with active runtime")
    if donor_lineage.get("tokenizer_identity_sha256") != tokenizer_hash:
        raise ValueError("source donor_lineage tokenizer identity hash disagrees with active runtime")
    return {
        "image_id": str(image_id),
        "source_image_sha256": image_sha256,
        "source_width": int(image_width),
        "source_height": int(image_height),
        "model_identity_sha256": model_hash,
        "tokenizer_identity_sha256": tokenizer_hash,
        "attention_implementation": str(attention_implementation),
        "model_config_dtype": str(model_config_dtype),
        "execution_receipt_schema_version": str(receipt["schema_version"]),
        "prompt_token_ids_sha256": sha256_json(_token_ids(bundle)[0]),
    }


def _forward_logits(model: Any, model_inputs: Mapping[str, Any], input_ids: Sequence[int], image_grid_thw: torch.Tensor) -> torch.Tensor:
    device = next(model.parameters()).device
    ids = torch.tensor([list(map(int, input_ids))], dtype=torch.long, device=device)
    attention = torch.ones_like(ids, dtype=torch.long)
    positions = derive_explicit_position_ids(
        model,
        input_ids=ids,
        attention_mask=attention,
        image_grid_thw=image_grid_thw.to(device=device),
    )
    kwargs = {
        key: value.to(device=device) if isinstance(value, torch.Tensor) else value
        for key, value in model_inputs.items()
    }
    kwargs.update({
        "input_ids": ids,
        "attention_mask": attention,
        "position_ids": positions,
        "use_cache": False,
        "return_dict": True,
        "logits_to_keep": 0,
    })
    with torch.inference_mode():
        output = model(**kwargs)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("model forward did not return logits")
    return logits[0].detach().to(device="cpu", dtype=torch.float32).contiguous()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--runtime-dtype", choices=("config", "fp32"), default="config")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    manifest = validate_frozen_manifest(_read_json(manifest_path), manifest_path=manifest_path)
    config_path = args.infer_config.expanduser().resolve(strict=True)
    source_jsonl = args.source_jsonl.expanduser().resolve(strict=True)

    from src.config.fingerprint import sha256_json as config_sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.image_plan import materialize_image_plan_batch, verify_processor_model_vision_parity
    from src.inference.pipeline import _processor_config, _template_config, _tokenizer_identity
    from src.inference.prompt import build_prompt_record
    from src.inference.runtime import assemble_runtime

    import contextlib
    import os

    @contextlib.contextmanager
    def temporary_cwd(path: Path):
        previous = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(previous)

    with temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    runtime = assemble_runtime(resolved.config, source_gate_root=config_path.parents[3])
    qwen = runtime.qwen
    if args.runtime_dtype == "fp32":
        qwen.model.to(dtype=torch.float32)
    qwen.model.eval()
    verify_processor_model_vision_parity(processor_identity=qwen.processor_identity, model_config=qwen.model.config)

    raw_rows = load_raw_examples(source_jsonl)
    raw = next(
        (
            row for row in raw_rows
            if str(row.metadata.get("source", {}).get("image_id")) == str(manifest["image_id"])
        ),
        None,
    )
    if raw is None:
        raise ValueError(f"image {manifest['image_id']} is absent from source JSONL")
    source_image_path = Path(raw.image.path).expanduser().resolve(strict=True)
    source_image_sha256 = sha256_file(source_image_path)
    active_model_identity = dict(runtime.model_identity)
    active_tokenizer_identity = _tokenizer_identity(qwen)
    actual_attention = _attention_implementation(qwen.model, resolved.config.model.attn_implementation)
    parent_lineage = validate_active_source_lineage(
        manifest["parent_bundle"],
        image_id=str(manifest["image_id"]),
        image_sha256=source_image_sha256,
        image_width=int(raw.image.width),
        image_height=int(raw.image.height),
        model_identity=active_model_identity,
        tokenizer_identity=active_tokenizer_identity,
        attention_implementation=actual_attention,
        model_config_dtype=str(resolved.config.model.dtype),
    )
    candidate_lineage: dict[str, dict[str, Any]] = {}
    for candidate in manifest["candidate_variants"]:
        candidate_path = Path(candidate["source_bundle"]).expanduser().resolve(strict=True)
        candidate_lineage[str(candidate["candidate_id"])] = validate_active_source_lineage(
            _read_json(candidate_path),
            image_id=str(manifest["image_id"]),
            image_sha256=source_image_sha256,
            image_width=int(raw.image.width),
            image_height=int(raw.image.height),
            model_identity=active_model_identity,
            tokenizer_identity=active_tokenizer_identity,
            attention_implementation=actual_attention,
            model_config_dtype=str(resolved.config.model.dtype),
        )
    template = _template_config(resolved.config)
    prompt_record = build_prompt_record(raw, template, processor=qwen.processor, row_index=0)
    if list(prompt_record.prompt_token_ids) != list(manifest["parent"]["prompt_token_ids"]):
        raise ValueError("active processor prompt does not equal frozen parent prompt")
    image_plan = materialize_image_plan_batch(
        [raw],
        components=qwen,
        processor_config=_processor_config(resolved.config),
        materialize=True,
        row_indices=[0],
    )
    model_inputs = image_plan.model_inputs_by_row_id[prompt_record.row_id]
    image_grid_thw = model_inputs.get("image_grid_thw")
    if not isinstance(image_grid_thw, torch.Tensor):
        raise ValueError("materialized image plan lacks image_grid_thw")
    base_prompt = manifest["parent"]["prompt_token_ids"]
    prefix = manifest["parent"]["prefix_token_ids"]
    parent_prompt = [*base_prompt, *prefix]
    row_entry_id = int(OBJECT_REF_START)
    terminal_id = qwen.tokenizer.eos_token_id
    if terminal_id is None or int(terminal_id) < 0:
        raise ValueError("tokenizer does not expose a valid eos_token_id")
    terminal_id = int(terminal_id)
    boundary_tokens: dict[str, list[int]] = {"parent": parent_prompt}
    terminal_scores: dict[str, dict[str, Any]] = {}
    boundary_logits: dict[str, torch.Tensor] = {}
    for label, tokens in boundary_tokens.items():
        logits = _forward_logits(qwen.model, model_inputs, tokens, image_grid_thw)
        boundary_logits[label] = logits
        terminal_scores[label] = terminal_boundary_score(
            logits,
            boundary_length=len(tokens),
            row_entry_token_id=row_entry_id,
            terminal_token_id=terminal_id,
        )

    before_scores: dict[str, dict[str, Any]] = {}
    for candidate in manifest["candidate_variants"]:
        logits = _forward_logits(qwen.model, model_inputs, parent_prompt + candidate["token_ids"], image_grid_thw)
        score = score_token_logits(
            logits,
            boundary_length=len(parent_prompt),
            row_tokens=candidate["token_ids"],
        )
        before_scores[str(candidate["candidate_id"])] = emit_candidate_score(score, candidate)

    donor_results: list[dict[str, Any]] = []
    for donor in manifest["donors"]:
        donor_id = str(donor["donor_id"])
        donor_candidate_id = str(donor["candidate_id"])
        donor_variant = next(item for item in manifest["candidate_variants"] if str(item["candidate_id"]) == donor_candidate_id)
        donor_prompt = [*parent_prompt, *donor_variant["token_ids"]]
        donor_logits = _forward_logits(qwen.model, model_inputs, donor_prompt, image_grid_thw)
        donor_terminal = terminal_boundary_score(
            donor_logits,
            boundary_length=len(donor_prompt),
            row_entry_token_id=row_entry_id,
            terminal_token_id=terminal_id,
        )
        after_scores: dict[str, dict[str, Any]] = {}
        delta_scores: dict[str, dict[str, Any]] = {}
        for candidate in manifest["candidate_variants"]:
            candidate_id = str(candidate["candidate_id"])
            logits = _forward_logits(qwen.model, model_inputs, donor_prompt + candidate["token_ids"], image_grid_thw)
            score = score_token_logits(
                logits,
                boundary_length=len(donor_prompt),
                row_tokens=candidate["token_ids"],
            )
            after_scores[candidate_id] = emit_candidate_score(score, candidate)
            delta_scores[candidate_id] = paired_score_delta(before_scores[candidate_id], after_scores[candidate_id])
        crossover_scores = compute_object_specific_crossover_scores(
            manifest["candidate_variants"],
            donor_candidate_id=donor_candidate_id,
            before_scores=before_scores,
            after_scores=after_scores,
        )
        before_owner_scores: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        after_owner_scores: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in before_scores.values():
            before_owner_scores[str(item["owner"])].append(item)
        for item in after_scores.values():
            after_owner_scores[str(item["owner"])].append(item)
        before_owner_aggregates = {
            owner: aggregate_frozen_owner_scores(items)
            for owner, items in before_owner_scores.items()
        }
        after_owner_aggregates = {
            owner: aggregate_frozen_owner_scores(items)
            for owner, items in after_owner_scores.items()
        }
        owner_aggregate_deltas = {
            owner: paired_owner_aggregate_delta(before_owner_aggregates[owner], after_owner_aggregates[owner])
            for owner in sorted(set(before_owner_aggregates) & set(after_owner_aggregates))
        }
        donor_results.append({
            "donor_id": donor_id,
            "candidate_id": donor_candidate_id,
            "owner": donor["owner"],
            "category": donor["category"],
            "role": donor["role"],
            "after_boundary_token_count": len(donor_prompt),
            "terminal_boundary": donor_terminal,
            "after_scores": after_scores,
            "paired_deltas": delta_scores,
            "object_specific_crossover": crossover_scores,
            "owner_frozen_variant_scores_before": before_owner_aggregates,
            "owner_frozen_variant_scores_after": after_owner_aggregates,
            "paired_owner_aggregate_deltas": owner_aggregate_deltas,
        })

    owners: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in before_scores.values():
        owners[str(item["owner"])].append(item)
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": "2026-07-17-next-row-probability-transition-and-causal-source-trace",
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "image_id": manifest["image_id"],
        "parent": {
            **{key: manifest["parent"][key] for key in ("source_bundle", "source_bundle_sha256", "prefix_token_count", "prefix_token_ids_sha256", "reconstructed_prompt_token_ids_sha256")},
            "prompt_token_count": len(base_prompt),
            "prompt_token_ids_sha256": sha256_json(base_prompt),
            "reconstructed_prompt_token_count": len(parent_prompt),
        },
        "candidate_variants": list(before_scores.values()),
        "owner_frozen_variant_scores": [aggregate_frozen_owner_scores(items) for items in owners.values()],
        "parent_terminal_boundary": terminal_scores["parent"],
        "donors": donor_results,
        "runtime": {
            "physical_batch_size": 1,
            "score_accumulation_dtype": "torch.float32",
            "model_dtype": _runtime_model_dtype_summary(qwen.model),
            "runtime_dtype_mode": str(args.runtime_dtype),
            "config_path": str(config_path),
            "config_sha256": config_sha256_json(resolved.config.model_dump(mode="json")),
            "source_jsonl": str(source_jsonl),
            "source_jsonl_sha256": sha256_file(source_jsonl),
            "repetition_penalty_processing": False,
            "cache": False,
            "feature_gating_intervention": False,
            "model_identity": active_model_identity,
            "tokenizer_identity": active_tokenizer_identity,
            "attention_implementation": actual_attention,
            "row_entry_token_id": row_entry_id,
            "eos_token_id": terminal_id,
            "image_identity": {
                "path": str(source_image_path),
                "sha256": source_image_sha256,
                "width": int(raw.image.width),
                "height": int(raw.image.height),
            },
            "prompt_identity": {
                "base_prompt_token_ids_sha256": sha256_json(base_prompt),
                "reconstructed_parent_prompt_token_ids_sha256": sha256_json(parent_prompt),
                "wrapper_token_ids": {
                    "object_ref_start": OBJECT_REF_START,
                    "object_ref_end": OBJECT_REF_END,
                    "box_start": BOX_START,
                    "box_end": BOX_END,
                },
                "template": resolved.config.template.model_dump(mode="json"),
            },
        },
        "lineage_proof": {
            "parent": parent_lineage,
            "candidate_sources": candidate_lineage,
            "candidate_prompt_reconstruction_equal_parent_or_base": {
                str(item["candidate_id"]): str(item["source_prompt_kind"])
                for item in manifest["candidate_variants"]
            },
        },
        "forbidden_comparisons": [
            "candidate rows are not normalized into a common probability matrix",
            "terminal boundary margin is not compared with full-row scores",
        ],
    }
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "receipt.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable receipt: {output_path}")
    output_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
