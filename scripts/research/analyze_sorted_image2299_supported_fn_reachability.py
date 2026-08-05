#!/usr/bin/env python3
"""Analyze native-prefix reachability for resolved image-2299 false negatives.

This CPU-only S2 consumer reads explicit S1 frozen-threshold artifacts.  It
does not score a model, recompute support, fit a threshold, or launch the S3
crossing intervention.  The upper ambiguity bound is primary and the lower
bound is reported only as sensitivity.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import (  # noqa: E402
    analyze_sorted_supported_fn_native_prefix_reachability_prevalence as prevalence,
)
from scripts.research import analyze_sorted_image2299_owner_accessibility as accessibility  # noqa: E402
from scripts.research import build_sorted_owner_accessibility_census_plan as planner  # noqa: E402
from scripts.research import merge_sorted_owner_accessibility_census_shards as merge  # noqa: E402


UNIT_ID = "2026-08-04-sorted-image2299-prospective-mechanism-extension"
IMAGE_ID = "2299"

SOURCE_ANALYSIS_SCHEMA_VERSION = accessibility.ANALYSIS_SCHEMA_VERSION
SOURCE_OWNER_SCHEMA_VERSION = accessibility.OWNER_SCHEMA_VERSION
SOURCE_OWNER_CONTEXT_SCHEMA_VERSION = accessibility.CONTEXT_SCHEMA_VERSION
SOURCE_RECEIPT_SCHEMA_VERSION = accessibility.RECEIPT_SCHEMA_VERSION
REPORT_SCHEMA_VERSION = "sorted-image2299-supported-fn-reachability-report.v1"
OWNER_RECORD_SCHEMA_VERSION = "sorted-image2299-supported-fn-reachability-owner-record.v1"
RECEIPT_SCHEMA_VERSION = "sorted-image2299-supported-fn-reachability-receipt.v1"

FROZEN_CALIBRATION_CONTENT_SHA256 = (
    "9dd6d7646fc55db6155124dc4bbfa46642b32b006d39758bd2e24d1ca97058c5"
)

REPORT_JSON_NAME = "report.json"
REPORT_MD_NAME = "report.md"
OWNER_RECORDS_NAME = "owner-records.jsonl"
RECEIPT_NAME = "receipt.json"
OUTPUT_NAMES = frozenset({REPORT_JSON_NAME, REPORT_MD_NAME, OWNER_RECORDS_NAME, RECEIPT_NAME})

ROOT_OR_AHEAD_STATES = frozenset({"root_no_frontier", "ahead_of_frontier"})
PASSED_STATE = "passed_by_frontier"
S3_MINIMUM_COHORT = 8

sha256_bytes = merge.sha256_bytes
sha256_json = merge.sha256_json
canonical_json_bytes = merge.canonical_json_bytes


class AnalysisContractError(RuntimeError):
    """A conclusion-bearing S2 precondition was not established."""


def _fail(message: str) -> NoReturn:
    raise AnalysisContractError(message)


@dataclass(frozen=True)
class SourcePaths:
    analysis: Path
    owner_summaries: Path
    owner_context_features: Path
    context_registry: Path
    source_receipt: Path


@dataclass(frozen=True)
class Inputs:
    paths: SourcePaths
    file_sha256: dict[str, str]
    analysis: dict[str, Any]
    source_receipt: dict[str, Any]
    owner_summaries: list[dict[str, Any]]
    owner_context_by_id: dict[str, dict[str, Any]]
    contexts_by_id: dict[str, dict[str, Any]]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"{label} is missing at {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{label} is unreadable at {path}: {exc}")
    if not isinstance(value, Mapping):
        _fail(f"{label} is not a JSON object")
    return dict(value)


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        _fail(f"{label} is missing at {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"{label} line {line_number} is not valid JSON: {exc}")
        if not isinstance(value, Mapping):
            _fail(f"{label} line {line_number} is not a JSON object")
        rows.append(dict(value))
    return rows


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    return sha256_json({key: value for key, value in receipt.items() if key != "receipt_content_sha256"})


def _attach_best_owner_identities(rows_by_id: Mapping[str, dict[str, Any]]) -> None:
    """Join S1's sealed rank-one row back as the best-owner identity.

    S1 publishes every physical owner at every native context and preserves
    bound-specific ranks, but intentionally does not duplicate the rank-one
    owner's ID into every competition block.  Covered/uncovered classification
    therefore uses this exact relational join; ranks are never recomputed.
    """

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows_by_id.values():
        grouped.setdefault(
            (str(row.get("context_id")), str(row.get("normalized_description"))), []
        ).append(row)
    for group_key, group in grouped.items():
        for bound in ("u", "l"):
            field = f"owner_competition_{bound}"
            ranked = [row for row in group if (row.get(field) or {}).get("rank") is not None]
            rank_one = [row for row in ranked if (row.get(field) or {}).get("rank") == 1]
            populations = {(row.get(field) or {}).get("population_size") for row in group}
            if len(populations) != 1:
                _fail(f"S1 owner competition population drifts within group {group_key!r} bound {bound}")
            population = populations.pop()
            if population != len(ranked):
                _fail(f"S1 owner competition population count is inconsistent in group {group_key!r} bound {bound}")
            if ranked and len(rank_one) != 1:
                _fail(f"S1 owner competition lacks one unique rank-one owner in group {group_key!r} bound {bound}")
            best_owner_id = str(rank_one[0]["gt_owner_id"]) if rank_one else None
            for row in group:
                competition = row.get(field)
                if not isinstance(competition, dict):
                    _fail(f"S1 owner-context {row.get('owner_context_id')!r} lacks {field}")
                competition["best_gt_owner_id"] = best_owner_id


def load_inputs(paths: SourcePaths) -> Inputs:
    normalized = SourcePaths(**{name: Path(getattr(paths, name)).resolve() for name in paths.__dataclass_fields__})
    file_sha256 = {
        name: sha256_bytes(getattr(normalized, name).read_bytes())
        if getattr(normalized, name).is_file()
        else _fail(f"explicit input {name} is missing at {getattr(normalized, name)}")
        for name in normalized.__dataclass_fields__
    }
    analysis = _read_json(normalized.analysis, "S1 analysis")
    source_receipt = _read_json(normalized.source_receipt, "source receipt")
    # Authenticate every input's bytes before parsing any conclusion-bearing
    # JSONL row.  A tampered row must fail as a source-identity violation,
    # rather than reach a later structural interpretation first.
    _validate_source_documents(analysis, source_receipt, file_sha256)
    owner_summaries = _read_jsonl(normalized.owner_summaries, "owner summaries")
    owner_context_rows = _read_jsonl(normalized.owner_context_features, "owner-context features")
    contexts = _read_jsonl(normalized.context_registry, "context registry")

    owner_context_by_id: dict[str, dict[str, Any]] = {}
    for row in owner_context_rows:
        identity = str(row.get("owner_context_id"))
        expected = f"{row.get('gt_owner_id')}@{row.get('context_id')}"
        if identity != expected:
            _fail(f"owner-context identity {identity!r} does not equal {expected!r}")
        if identity in owner_context_by_id:
            _fail(f"duplicate owner-context identity {identity!r}")
        # S1 deliberately calls this independently-scored channel
        # ``proposal_surface``.  The frozen predecessor ladder consumes the
        # same payload under its older ``category_proposal_channel`` name.
        # Adapt only the key; no score, margin, route, or rank is recomputed.
        adapted = dict(row)
        adapted["category_proposal_channel"] = row.get("proposal_surface")
        owner_context_by_id[identity] = adapted

    contexts_by_id: dict[str, dict[str, Any]] = {}
    for row in contexts:
        context_id = str(row.get("context_id"))
        if context_id in contexts_by_id:
            _fail(f"duplicate context identity {context_id!r}")
        contexts_by_id[context_id] = row

    _attach_best_owner_identities(owner_context_by_id)

    return Inputs(
        paths=normalized,
        file_sha256=file_sha256,
        analysis=analysis,
        source_receipt=source_receipt,
        owner_summaries=owner_summaries,
        owner_context_by_id=owner_context_by_id,
        contexts_by_id=contexts_by_id,
    )


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    return value


def _validate_source_documents(
    analysis: Mapping[str, Any], receipt: Mapping[str, Any], file_sha256: Mapping[str, str]
) -> None:
    if analysis.get("schema_version") != SOURCE_ANALYSIS_SCHEMA_VERSION:
        _fail("support analysis schema_version is not the frozen image-2299 S1 schema")
    if analysis.get("unit_id") != UNIT_ID or str(analysis.get("image_id")) != IMAGE_ID:
        _fail("support analysis does not identify this image-2299 research unit")

    calibration = _require_mapping(analysis.get("calibration"), label="calibration")
    if calibration.get("thresholds_retuned") is not False:
        _fail("S1 analysis does not prove non-retuned legacy calibration")
    if calibration.get("phenotype_fitted") is not False:
        _fail("S1 analysis fitted or failed to exclude a phenotype rule")
    if calibration.get("content_sha256") != FROZEN_CALIBRATION_CONTENT_SHA256:
        _fail("support analysis carries the wrong frozen calibration content digest")

    if receipt.get("schema_version") != SOURCE_RECEIPT_SCHEMA_VERSION:
        _fail("source receipt schema_version is not the frozen image-2299 S1 receipt schema")
    if receipt.get("unit_id") != UNIT_ID:
        _fail("source receipt does not identify this image-2299 research unit")
    if _receipt_digest(receipt) != receipt.get("receipt_content_sha256"):
        _fail("source receipt does not reconstruct its own receipt_content_sha256")
    receipt_calibration = _require_mapping(
        receipt.get("calibration"), label="source receipt calibration"
    )
    if (
        receipt_calibration.get("content_sha256") != FROZEN_CALIBRATION_CONTENT_SHA256
        or receipt_calibration.get("thresholds_retuned") is not False
        or receipt_calibration.get("phenotype_fitted") is not False
    ):
        _fail("source receipt does not bind the frozen support-only calibration contract")
    declared = _require_mapping(receipt.get("output_file_digests"), label="output_file_digests")
    keys = {
        "analysis": "analysis.json",
        "owner_summaries": "owner-summaries.jsonl",
        "owner_context_features": "owner-context-features.jsonl",
        "context_registry": "context-registry.jsonl",
    }
    for input_key, receipt_key in keys.items():
        if declared.get(receipt_key) != file_sha256[input_key]:
            _fail(f"source receipt digest mismatch for {receipt_key}")


def _validate_source_seal(inputs: Inputs) -> None:
    _validate_source_documents(
        inputs.analysis, inputs.source_receipt, inputs.file_sha256
    )


def _validate_contexts(inputs: Inputs) -> None:
    for context_id, row in inputs.contexts_by_id.items():
        if row.get("schema_version") != planner.PLAN_SCHEMA_VERSION:
            _fail(f"context {context_id!r} has an unexpected schema_version")
        if str(row.get("image_id")) != IMAGE_ID or not context_id.startswith(f"{IMAGE_ID}:"):
            _fail(f"context {context_id!r} is outside image {IMAGE_ID}")
        role = str(row.get("context_role"))
        if role not in prevalence.NATIVE_CONTEXT_ROLES:
            _fail(f"context {context_id!r} role {role!r} is not native root/row/terminal")
        admission = _require_mapping(row.get("prefix_admission"), label=f"context {context_id} admission")
        if admission.get("forced_continue_rows_excluded") is not True:
            _fail(f"context {context_id!r} does not exclude forced-continue rows")
        if admission.get("source") != "native_greedy_complete_rows_only":
            _fail(f"context {context_id!r} is not sourced only from native greedy complete rows")


def _validate_owner_contexts(inputs: Inputs) -> None:
    for identity, row in inputs.owner_context_by_id.items():
        if row.get("schema_version") != SOURCE_OWNER_CONTEXT_SCHEMA_VERSION:
            _fail(f"owner-context {identity!r} has an unexpected schema_version")
        if str(row.get("image_id")) != IMAGE_ID:
            _fail(f"owner-context {identity!r} is outside image {IMAGE_ID}")
        context_id = str(row.get("context_id"))
        context = inputs.contexts_by_id.get(context_id)
        if context is None:
            _fail(f"owner-context {identity!r} references missing context {context_id!r}")
        if int(row.get("boundary_index", -1)) != int(context.get("boundary_index", -2)):
            _fail(f"owner-context {identity!r} boundary index disagrees with its context")
        if row.get("context_role") != context.get("context_role"):
            _fail(f"owner-context {identity!r} role disagrees with its context")
        _require_mapping(row.get("loop_marking"), label=f"owner-context {identity} loop_marking")
        frontier = _require_mapping(row.get("frontier_features"), label=f"owner-context {identity} frontier")
        if frontier.get("passed_state") not in {
            "root_no_frontier", "ahead_of_frontier", "at_frontier", "passed_by_frontier"
        }:
            _fail(f"owner-context {identity!r} has unknown frontier passed_state")
        if row.get("native_context_only") is not True:
            _fail(f"owner-context {identity!r} is not declared native-only")
        localization = _require_mapping(
            row.get("localization"), label=f"owner-context {identity} localization"
        )
        if localization.get("estimand") != "category_field_support_at_owner_geometry":
            _fail(f"owner-context {identity!r} has a foreign localization estimand")
        if localization.get("proposal_probability") is not False:
            _fail(f"owner-context {identity!r} treats localization as proposal probability")
        support = _require_mapping(
            localization.get("generator_local_max_excluding_other_owner_strict"),
            label=f"owner-context {identity} frozen support",
        )
        for support_bound in ("ambiguity_excluded_l", "ambiguity_included_u"):
            bound_block = _require_mapping(
                support.get(support_bound), label=f"owner-context {identity} {support_bound}"
            )
            if bound_block.get("clears_frozen_support_rule") not in (True, False):
                _fail(f"owner-context {identity!r} lacks a boolean frozen support disposition")
        proposal = _require_mapping(
            row.get("proposal_surface"), label=f"owner-context {identity} proposal_surface"
        )
        if proposal.get("separate_from_localization") is not True:
            _fail(f"owner-context {identity!r} combines proposal and localization surfaces")
        gate = _require_mapping(proposal.get("boundary_gate"), label=f"owner-context {identity} gate")
        margin = gate.get("continue_vs_stop_logprob_margin")
        if margin is not None and not isinstance(margin, (int, float)):
            _fail(f"owner-context {identity!r} gate margin must be numeric or null")
        routing = proposal.get("category_routing_event")
        if routing is not None:
            routing = _require_mapping(routing, label=f"owner-context {identity} routing")
            rank = routing.get("within_context_rank")
            if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
                _fail(f"owner-context {identity!r} category route rank must be a positive integer")
        for bound in ("u", "l"):
            competition = _require_mapping(
                row.get(f"owner_competition_{bound}"),
                label=f"owner-context {identity} owner_competition_{bound}",
            )
            rank = competition.get("rank")
            if rank is not None and (not isinstance(rank, int) or isinstance(rank, bool) or rank < 1):
                _fail(f"owner-context {identity!r} owner rank {bound} must be positive or null")


def validate_inputs(inputs: Inputs) -> dict[str, Any]:
    _validate_source_seal(inputs)
    _validate_contexts(inputs)
    _validate_owner_contexts(inputs)

    owner_ids: set[str] = set()
    resolved: list[dict[str, Any]] = []
    ambiguity_flip_count = 0
    for row in inputs.owner_summaries:
        if row.get("schema_version") != SOURCE_OWNER_SCHEMA_VERSION:
            _fail(f"owner summary {row.get('gt_owner_id')!r} has an unexpected schema_version")
        if str(row.get("image_id")) != IMAGE_ID:
            _fail(f"owner summary {row.get('gt_owner_id')!r} is outside image {IMAGE_ID}")
        owner_id = str(row.get("gt_owner_id"))
        if owner_id in owner_ids:
            _fail(f"duplicate owner summary {owner_id!r}")
        owner_ids.add(owner_id)
        native_tp = row.get("native_true_positive")
        if native_tp not in (True, False) or row.get("native_false_negative") is not (not native_tp):
            _fail(f"owner {owner_id!r} has an incoherent native TP/FN identity")
        if native_tp is False and row.get("disposition") == accessibility.DISPOSITION_RESOLVED:
            resolved.append(row)
            ambiguity_flip_count += int(bool(row.get("ambiguity_bound_disposition_flip")))
        support_by_block = {
            "upper_bound_u": "ambiguity_included_u",
            "lower_bound_l": "ambiguity_excluded_l",
        }
        for block_name, support_bound in support_by_block.items():
            block = _require_mapping(row.get(block_name), label=f"owner {owner_id} {block_name}")
            context_ids = block.get("usable_support_context_ids")
            if not isinstance(context_ids, list) or any(not isinstance(value, str) for value in context_ids):
                _fail(f"owner {owner_id!r} {block_name}.usable_support_context_ids must be strings")
            if len(context_ids) != len(set(context_ids)):
                _fail(f"owner {owner_id!r} {block_name} repeats a usable support context")
            for context_id in context_ids:
                feature = inputs.owner_context_by_id.get(f"{owner_id}@{context_id}")
                if feature is None:
                    _fail(f"owner {owner_id!r} {block_name} references a missing feature row")
                if bool(feature["loop_marking"]["loop_tail"]):
                    _fail(f"owner {owner_id!r} {block_name} lists a loop_tail support context")
            expected_ids = sorted(
                str(feature["context_id"])
                for feature in inputs.owner_context_by_id.values()
                if str(feature.get("gt_owner_id")) == owner_id
                and not bool(feature["loop_marking"]["loop_tail"])
                and feature["localization"][
                    "generator_local_max_excluding_other_owner_strict"
                ][support_bound]["clears_frozen_support_rule"]
                is True
            )
            if sorted(context_ids) != expected_ids:
                _fail(f"owner {owner_id!r} {block_name} support IDs disagree with S1 feature rows")
        expected_flip = bool(row["lower_bound_l"]["usable_support_context_ids"]) != bool(
            row["upper_bound_u"]["usable_support_context_ids"]
        )
        if bool(row.get("ambiguity_bound_disposition_flip")) != expected_flip:
            _fail(f"owner {owner_id!r} ambiguity-bound flip flag is incoherent")

    expected_pairs = {
        f"{owner_id}@{context_id}"
        for owner_id in owner_ids
        for context_id in inputs.contexts_by_id
    }
    if set(inputs.owner_context_by_id) != expected_pairs:
        _fail("S1 owner-context features are not the complete owner by native-context matrix")

    denominators = _require_mapping(inputs.analysis.get("denominators"), label="S1 denominators")
    native_fn_count = sum(row.get("native_false_negative") is True for row in inputs.owner_summaries)
    if denominators.get("image2299_owner_count") != len(inputs.owner_summaries):
        _fail("S1 image-2299 owner denominator does not match owner summaries")
    if denominators.get("image2299_native_fn_count") != native_fn_count:
        _fail("S1 native-FN denominator does not match owner summaries")

    transfer = _require_mapping(inputs.analysis.get("calibration_transfer"), label="calibration_transfer")
    status = transfer.get("status")
    transfer_contract = {
        "calibration_transfer_underpowered": (None, False),
        "calibration_transfer_passed": (True, True),
        "calibration_nontransferring": (False, True),
    }
    if status not in transfer_contract:
        _fail("S1 calibration transfer status is unknown")
    expected_passes, expected_validity = transfer_contract[str(status)]
    if transfer.get("passes") is not expected_passes or transfer.get("validity_bearing") is not expected_validity:
        _fail("S1 calibration transfer status/passes/validity fields are incoherent")

    expected_counts = {
        "owner_count": len(inputs.owner_summaries),
        "owner_context_row_count": len(inputs.owner_context_by_id),
        "context_count": len(inputs.contexts_by_id),
        "native_false_negative_count": native_fn_count,
        "resolved_native_false_negative_count": len(resolved),
        "resolved_native_false_negative_ambiguity_flip_count": ambiguity_flip_count,
    }

    return {
        **expected_counts,
        "resolved_fn_denominator_rule": (
            "native_true_positive is false and disposition is "
            f"{accessibility.DISPOSITION_RESOLVED!r}; no other owner enters the denominator"
        ),
        "image_id": IMAGE_ID,
        "native_contexts_only": True,
        "forced_continue_rows_excluded": True,
        "frozen_calibration_imported_not_recomputed": True,
        "calibration_transfer_status": status,
        "calibration_transfer_validity_bearing": transfer["validity_bearing"],
        "source_receipt_content_sha256": inputs.source_receipt["receipt_content_sha256"],
    }


def _owner_context_rows(
    owner_id: str, inputs: Inputs
) -> dict[int, Mapping[str, Any]]:
    rows: dict[int, Mapping[str, Any]] = {}
    for row in inputs.owner_context_by_id.values():
        if str(row.get("gt_owner_id")) != owner_id:
            continue
        boundary = int(row["boundary_index"])
        if boundary in rows:
            _fail(f"owner {owner_id!r} has duplicate owner-context boundary {boundary}")
        rows[boundary] = row
    if not rows:
        _fail(f"resolved owner {owner_id!r} has no owner-context feature rows")
    return rows


def derive_crossing_boundary(rows: Mapping[int, Mapping[str, Any]]) -> int | None:
    """Return P for the exact first P -> P+E frontier crossing, if it exists."""

    for boundary in sorted(rows):
        if str(rows[boundary]["frontier_features"]["passed_state"]) != PASSED_STATE:
            continue
        previous = boundary - 1
        if previous not in rows:
            return None
        previous_state = str(rows[previous]["frontier_features"]["passed_state"])
        if previous_state not in ROOT_OR_AHEAD_STATES:
            return None
        return previous
    return None


def _extract_channel(
    owner_context_row: Mapping[str, Any], context: Mapping[str, Any], *, bound: str
) -> dict[str, Any]:
    try:
        return prevalence.extract_context_channel(owner_context_row, context, bound=bound)
    except (KeyError, TypeError, ValueError, prevalence.AnalysisContractError) as exc:
        _fail(f"invalid frozen channel row {owner_context_row.get('owner_context_id')!r}: {exc}")


def _crossing_record(
    summary: Mapping[str, Any], inputs: Inputs
) -> dict[str, Any]:
    owner_id = str(summary["gt_owner_id"])
    rows = _owner_context_rows(owner_id, inputs)
    boundary = derive_crossing_boundary(rows)
    if boundary is None:
        return {
            "exact_crossing_exists": False,
            "boundary_index": None,
            "p_context_id": None,
            "p_plus_e_context_id": None,
            "u_favorable_supported": False,
            "l_favorable_supported": False,
            "u_channel": None,
            "l_channel": None,
        }

    p_row = rows[boundary]
    pe_row = rows.get(boundary + 1)
    if pe_row is None:
        _fail(f"owner {owner_id!r} crossing has no P+E owner-context row")
    p_context_id = str(p_row["context_id"])
    pe_context_id = str(pe_row["context_id"])
    p_context = inputs.contexts_by_id[p_context_id]
    channels = {bound: _extract_channel(p_row, p_context, bound=bound) for bound in ("u", "l")}
    favorable: dict[str, bool] = {}
    for bound, block_name in (("u", "upper_bound_u"), ("l", "lower_bound_l")):
        usable = p_context_id in set(summary[block_name]["usable_support_context_ids"])
        channel = channels[bound]
        favorable[bound] = bool(
            usable
            and channel["gate_open"]
            and channel["category_rank_top3"]
            and channel["owner_rank_one"]
            and channel["before_or_at_frontier"]
        )
        channel["usable_support"] = usable
        channel["favorable_top3_supported_before_or_at_frontier"] = favorable[bound]

    return {
        "exact_crossing_exists": True,
        "boundary_index": boundary,
        "p_context_id": p_context_id,
        "p_plus_e_context_id": pe_context_id,
        "u_favorable_supported": favorable["u"],
        "l_favorable_supported": favorable["l"],
        "u_channel": channels["u"],
        "l_channel": channels["l"],
    }


def analyze_owner(summary: Mapping[str, Any], inputs: Inputs) -> dict[str, Any]:
    try:
        base = prevalence.analyze_fn_owner(
            summary, inputs.owner_context_by_id, inputs.contexts_by_id
        )
    except (KeyError, TypeError, ValueError, prevalence.AnalysisContractError) as exc:
        _fail(f"resolved owner {summary.get('gt_owner_id')!r} violates reachability contract: {exc}")
    base = dict(base)
    base["schema_version"] = OWNER_RECORD_SCHEMA_VERSION
    base["cohort"] = "image2299_resolved_native_false_negative"
    base["source_owner_summary_sha256"] = sha256_json(summary)
    base["source_context_ids"] = {
        "upper_bound_u": list(summary["upper_bound_u"]["usable_support_context_ids"]),
        "lower_bound_l": list(summary["lower_bound_l"]["usable_support_context_ids"]),
    }
    base["exact_crossing"] = _crossing_record(summary, inputs)
    return base


def _gate_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    resolved_count = len(records)
    u_ids = sorted(
        str(row["gt_owner_id"])
        for row in records
        if row["exact_crossing"]["u_favorable_supported"]
    )
    l_ids = sorted(
        str(row["gt_owner_id"])
        for row in records
        if row["exact_crossing"]["l_favorable_supported"]
    )
    resolved_meets = resolved_count >= S3_MINIMUM_COHORT
    crossing_meets = len(u_ids) >= S3_MINIMUM_COHORT
    opens = resolved_meets and crossing_meets
    return {
        "minimum_each": S3_MINIMUM_COHORT,
        "resolved_cohort_count": resolved_count,
        "resolved_cohort_meets_minimum": resolved_meets,
        "u_primary_exact_crossing_favorable_count": len(u_ids),
        "u_primary_exact_crossing_favorable_owner_ids": u_ids,
        "u_primary_exact_crossing_favorable_wilson_95": prevalence.wilson_interval(
            len(u_ids), resolved_count
        ),
        "l_sensitivity_exact_crossing_favorable_count": len(l_ids),
        "l_sensitivity_exact_crossing_favorable_owner_ids": l_ids,
        "l_sensitivity_exact_crossing_favorable_wilson_95": prevalence.wilson_interval(
            len(l_ids), resolved_count
        ),
        "crossing_favorable_cohort_meets_minimum": crossing_meets,
        "s3_gate_open": opens,
        "action": (
            "eligible_for_separately_authorized_s3_capture_not_launched_by_this_analysis"
            if opens
            else "stop_s3_branch_for_insufficient_frozen_cohort"
        ),
        "s3_launched": False,
    }


def build_report(inputs: Inputs, validation: Mapping[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    headline = prevalence.aggregate_fn_headline(records)
    gate = _gate_summary(records)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "image_id": IMAGE_ID,
        "input_paths": {name: str(getattr(inputs.paths, name)) for name in inputs.paths.__dataclass_fields__},
        "input_file_sha256": dict(inputs.file_sha256),
        "validation": dict(validation),
        "resolved_false_negative_cohort": {
            "denominator": len(records),
            "denominator_rule": validation["resolved_fn_denominator_rule"],
            "upper_bound_u_primary": True,
            "lower_bound_l_sensitivity_only": True,
            "headline": headline,
        },
        "exact_crossing_gate": gate,
        "one_image_caveat": (
            "Image 2299 is one shared image encoding, not an iid population sample. "
            "Wilson 95% intervals are descriptive references only and do not establish "
            "population prevalence or independence among owners."
        ),
        "claims": {
            "scores_or_thresholds_recomputed": False,
            "forced_continue_contexts_used": False,
            "s3_launched": False,
            "population_prevalence_claim": False,
            "causal_decoder_claim": False,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    cohort = report["resolved_false_negative_cohort"]
    gate = report["exact_crossing_gate"]
    lines = [
        "# Image 2299 supported-FN native-prefix reachability",
        "",
        f"Resolved native-FN denominator: **{cohort['denominator']}**.",
        "",
        str(report["one_image_caveat"]),
        "",
        "## U-primary reachability ladder",
        "",
        "| metric | successes | total | proportion | Wilson 95% interval |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    metrics = cohort["headline"]["upper_bound_u"]["metrics"]
    for name in prevalence.LADDER_KEYS:
        metric = metrics[name]
        proportion = "n/a" if metric["proportion"] is None else f"{metric['proportion']:.3f}"
        interval = (
            "n/a"
            if metric["lower"] is None
            else f"[{metric['lower']:.3f}, {metric['upper']:.3f}]"
        )
        lines.append(
            f"| {name} | {metric['successes']} | {metric['total']} | {proportion} | {interval} |"
        )
    lines.extend(
        [
            "",
            "## Conditional S3 gate",
            "",
            f"- Resolved cohort: {gate['resolved_cohort_count']} "
            f"(>= {gate['minimum_each']}: {gate['resolved_cohort_meets_minimum']}).",
            f"- U-primary exact crossing-favorable cohort: "
            f"{gate['u_primary_exact_crossing_favorable_count']} "
            f"(>= {gate['minimum_each']}: {gate['crossing_favorable_cohort_meets_minimum']}).",
            f"- Gate open: {gate['s3_gate_open']}.",
            f"- Action: `{gate['action']}`.",
            "- This analyzer never launches S3.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(paths: SourcePaths) -> dict[str, Any]:
    inputs = load_inputs(paths)
    validation = validate_inputs(inputs)
    summaries = [
        row
        for row in inputs.owner_summaries
        if row.get("native_true_positive") is False
        and row.get("disposition") == accessibility.DISPOSITION_RESOLVED
    ]
    records = [analyze_owner(row, inputs) for row in sorted(summaries, key=lambda r: str(r["gt_owner_id"]))]
    return {"report": build_report(inputs, validation, records), "owner_records": records}


def _materialize_files(result: Mapping[str, Any]) -> dict[str, bytes]:
    report = result["report"]
    records = result["owner_records"]
    report_json = canonical_json_bytes(report) + b"\n"
    report_md = render_markdown(report).encode("utf-8")
    owner_records = b"".join(canonical_json_bytes(row) + b"\n" for row in records)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "image_id": IMAGE_ID,
        "input_paths": dict(report["input_paths"]),
        "input_file_sha256": dict(report["input_file_sha256"]),
        "source_receipt_content_sha256": report["validation"]["source_receipt_content_sha256"],
        "analyzer_source_sha256": sha256_bytes(Path(__file__).resolve().read_bytes()),
        "output_file_sha256": {
            REPORT_JSON_NAME: sha256_bytes(report_json),
            REPORT_MD_NAME: sha256_bytes(report_md),
            OWNER_RECORDS_NAME: sha256_bytes(owner_records),
        },
        "resolved_fn_denominator": report["resolved_false_negative_cohort"]["denominator"],
        "u_primary_exact_crossing_favorable_count": report["exact_crossing_gate"][
            "u_primary_exact_crossing_favorable_count"
        ],
        "s3_gate_open": report["exact_crossing_gate"]["s3_gate_open"],
        "s3_launched": False,
        "publication_policy": "atomic_create_or_identical",
    }
    receipt["receipt_content_sha256"] = _receipt_digest(receipt)
    return {
        REPORT_JSON_NAME: report_json,
        REPORT_MD_NAME: report_md,
        OWNER_RECORDS_NAME: owner_records,
        RECEIPT_NAME: canonical_json_bytes(receipt) + b"\n",
    }


def publish_create_or_identical(output_dir: Path, files: Mapping[str, bytes]) -> str:
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            _fail(f"output path exists but is not a directory: {output_dir}")
        names = {path.name for path in output_dir.iterdir() if path.is_file()}
        if names != OUTPUT_NAMES:
            _fail("output directory contains a foreign or partial file set")
        if any((output_dir / name).read_bytes() != content for name, content in files.items()):
            _fail("output directory exists with non-identical content")
        return "identical_existing_output"

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        for name, content in files.items():
            (temp_dir / name).write_bytes(content)
        os.replace(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return "created"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--owner-summaries", type=Path, required=True)
    parser.add_argument("--owner-context-features", type=Path, required=True)
    parser.add_argument("--context-registry", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = SourcePaths(
        analysis=args.analysis,
        owner_summaries=args.owner_summaries,
        owner_context_features=args.owner_context_features,
        context_registry=args.context_registry,
        source_receipt=args.source_receipt,
    )
    try:
        result = run_analysis(paths)
        status = publish_create_or_identical(args.output_dir, _materialize_files(result))
    except AnalysisContractError as exc:
        raise SystemExit(f"analysis contract violated: {exc}") from exc
    print(json.dumps({"output_dir": str(args.output_dir), "publication_status": status}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
