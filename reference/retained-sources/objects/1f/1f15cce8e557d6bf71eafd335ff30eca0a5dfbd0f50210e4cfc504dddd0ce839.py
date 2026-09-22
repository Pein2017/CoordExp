#!/usr/bin/env python3
"""CPU-only reanalysis for the sorted supported false-negative native-prefix
reachability prevalence unit
(``2026-08-03-sorted-supported-fn-native-prefix-reachability-prevalence``).

This module scores nothing and captures nothing new.  It re-reads the frozen
presentation output of the predecessor sorted owner accessibility phenotype
census, re-verifies every declared digest, and reports -- per owner, never per
context row -- whether any calibrated support context on the native greedy
trajectory jointly has an open continue-vs-stop gate, a favorable
category-route rank, a winning same-category owner competition, and a
before-or-at-frontier state.  See
``research/investigations/qwen3-vl-dense-enumeration/experiments/
2026-08-03-sorted-supported-fn-native-prefix-reachability-prevalence/unit.md``
for the frozen question and operational definitions this module implements.

What it reads
-------------
Six authoritative files under the immutable predecessor run root:

``plan/receipt.json``, ``plan/context-registry.jsonl``,
``plan/native-sidecar-registry.jsonl``, ``phases/presentation/merge-receipt.json``,
``phases/presentation/owner-summaries.jsonl``,
``phases/presentation/owner-context-features.jsonl``.

Every file's exact sha256 is sealed into the report and receipt.  Both
predecessor receipts are re-verified to reconstruct their own declared
digest, and both are cross-checked against each other
(``merge-receipt.json.plan.receipt_content_sha256`` must equal
``plan/receipt.json.receipt_content_sha256``) before a single owner row is
interpreted.

What it never does
-------------------
It never recomputes a localization score, a support calibration threshold, an
owner rank, a category rank, or a gate margin: every one of those quantities
is read verbatim from the sealed ``owner-context-features.jsonl`` /
``owner-summaries.jsonl`` rows.  ``owner_competition_u.rank`` is read as
*the* same-category owner rank; ``category_routing_event.within_context_rank``
is read as *the* category-route rank; these are never substituted for one
another or for the local candidate-bank rank.

Two owner cohorts
-----------------
``native_false_negative_supported``
    The ``114`` owners whose predecessor disposition is
    ``resolved_tested_localization_support``.  Each is examined at every
    context in its optimistic ``upper_bound_u.usable_support_context_ids``
    (primary), with the same channel ladder recomputed over
    ``lower_bound_l.usable_support_context_ids`` as a sensitivity view.

``native_true_positive_reference``
    The ``141`` owners with ``native_true_positive`` true, examined at their
    single native strict-match due boundary (the boundary immediately before
    the native row that matched them), located from
    ``native-sidecar-registry.jsonl`` alone.  This is a descriptive positive
    reference, never a causal control, and it uses one exact context rather
    than the false-negative arm's optimistic ``any usable context`` operator.

What it publishes
-----------------
``analysis/report.json``, ``analysis/report.md``, ``analysis/owner-records.jsonl``,
``analysis/receipt.json`` under a CLI-supplied output root.  It never writes
to the predecessor run root.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import merge_sorted_owner_accessibility_census_shards as merge  # noqa: E402
from scripts.research import build_sorted_owner_accessibility_census_plan as planner  # noqa: E402

UNIT_ID = "2026-08-03-sorted-supported-fn-native-prefix-reachability-prevalence"
PREDECESSOR_UNIT_ID = merge.UNIT_ID
REPORT_SCHEMA_VERSION = "sorted-supported-fn-native-prefix-reachability-prevalence-report.v1"
RECEIPT_SCHEMA_VERSION = "sorted-supported-fn-native-prefix-reachability-prevalence-receipt.v1"
OWNER_RECORD_SCHEMA_VERSION = "sorted-supported-fn-native-prefix-reachability-prevalence-owner-record.v1"

REPORT_JSON_NAME = "report.json"
REPORT_MD_NAME = "report.md"
OWNER_RECORDS_NAME = "owner-records.jsonl"
RECEIPT_NAME = "receipt.json"

#: Authoritative input files, named relative to the predecessor run root.
PLAN_RECEIPT_REL = "plan/receipt.json"
CONTEXT_REGISTRY_REL = "plan/context-registry.jsonl"
NATIVE_SIDECAR_REGISTRY_REL = "plan/native-sidecar-registry.jsonl"
MERGE_RECEIPT_REL = "phases/presentation/merge-receipt.json"
OWNER_SUMMARIES_REL = "phases/presentation/owner-summaries.jsonl"
OWNER_CONTEXT_FEATURES_REL = "phases/presentation/owner-context-features.jsonl"
AUTHORITATIVE_INPUT_FILES: tuple[str, ...] = (
    PLAN_RECEIPT_REL,
    CONTEXT_REGISTRY_REL,
    NATIVE_SIDECAR_REGISTRY_REL,
    MERGE_RECEIPT_REL,
    OWNER_SUMMARIES_REL,
    OWNER_CONTEXT_FEATURES_REL,
)

EXPECTED_FN_DENOMINATOR = 114
EXPECTED_TP_DENOMINATOR = 141

NATIVE_CONTEXT_ROLES: frozenset[str] = frozenset({"root", "row_boundary", "terminal"})
BEFORE_OR_AT_FRONTIER_STATES: frozenset[str] = frozenset(
    {"root_no_frontier", "ahead_of_frontier", "at_frontier"}
)

FN_COHORT = "native_false_negative_supported"
TP_COHORT = "native_true_positive_reference"

COMPETITOR_TARGET = "target"
COMPETITOR_COVERED_OTHER = "covered_other"
COMPETITOR_UNCOVERED_OTHER = "uncovered_other"
COMPETITOR_NO_POPULATION = "no_competition_population"

SEMANTICS_NOTES: tuple[str, ...] = (
    "Every localization score is obtained after the category description is "
    "teacher-forced; the gate, category route, and coordinate bank are separate "
    "readouts scored independently, so their conjunction is a favorable OBSERVED "
    "SURFACE, never the joint probability of a naturally generated row.",
    "Category-conditioned readouts (category rank, owner rank) are not natural "
    "proposal probabilities and are never renormalized as one.",
    "The favorable conjunction is descriptive: absence of the conjunction does "
    "not identify which decoder token caused a miss, and its presence does not "
    "certify that a naturally decoded rollout would have emitted the row.",
    "Raw complete-box logprob sums and gate/routing logprobs are never compared "
    "across images; every ranked or margin quantity here is scoped to one "
    "(image, context, category) population.",
    "The native true-positive due-boundary reference is a positive descriptive "
    "reference, not a matched causal control: the false-negative arm uses an "
    "optimistic 'any usable-support context' operator, while the reference uses "
    "one exact due context, so differences between the two arms are never a "
    "causal effect size.",
    "No binary phenotype threshold is fitted here; the conjunction ladder is "
    "transparent and every owner record retains its underlying continuous ranks "
    "and margins.",
)


class AnalysisContractError(RuntimeError):
    """A precondition for a conclusion-bearing reanalysis was not proven."""


def _fail(message: str) -> NoReturn:
    raise AnalysisContractError(message)


# ---------------------------------------------------------------------------
# IO helpers (reuse the merge module's canonicalization so digests agree)
# ---------------------------------------------------------------------------

sha256_bytes = merge.sha256_bytes
sha256_json = merge.sha256_json
canonical_json_bytes = merge.canonical_json_bytes


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"{label} is missing at {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{label} is unreadable at {path}: {exc}")
    if not isinstance(value, Mapping):
        _fail(f"{label} at {path} is not a JSON object")
    return dict(value)


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        _fail(f"{label} is missing at {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"{label} line {number} is not valid JSON: {exc}")
        if not isinstance(value, Mapping):
            _fail(f"{label} line {number} is not a JSON object")
        rows.append(dict(value))
    return rows


def _reconstruct_receipt_digest(receipt: Mapping[str, Any], *, label: str) -> None:
    reconstructed = sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    if reconstructed != receipt.get("receipt_content_sha256"):
        _fail(f"{label} does not reconstruct its own receipt_content_sha256; it was edited")


# ---------------------------------------------------------------------------
# Input loading and validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Inputs:
    run_root: Path
    file_sha256: dict[str, str]
    plan_receipt: dict[str, Any]
    merge_receipt: dict[str, Any]
    owner_summaries: list[dict[str, Any]]
    owner_context_by_id: dict[str, dict[str, Any]]
    contexts_by_id: dict[str, dict[str, Any]]
    native_sidecars: list[dict[str, Any]]


def load_inputs(run_root: Path) -> Inputs:
    """Read the six authoritative files and hash them, before any parsing."""

    run_root = Path(run_root)
    file_sha256: dict[str, str] = {}
    for relative_name in AUTHORITATIVE_INPUT_FILES:
        path = run_root / relative_name
        if not path.is_file():
            _fail(f"authoritative input file is missing: {relative_name} under {run_root}")
        file_sha256[relative_name] = sha256_bytes(path.read_bytes())

    plan_receipt = _read_json(run_root / PLAN_RECEIPT_REL, "plan/receipt.json")
    merge_receipt = _read_json(run_root / MERGE_RECEIPT_REL, "phases/presentation/merge-receipt.json")
    owner_summaries = _read_jsonl(run_root / OWNER_SUMMARIES_REL, "owner-summaries.jsonl")
    owner_context_rows = _read_jsonl(
        run_root / OWNER_CONTEXT_FEATURES_REL, "owner-context-features.jsonl"
    )
    contexts = _read_jsonl(run_root / CONTEXT_REGISTRY_REL, "context-registry.jsonl")
    native_sidecars = _read_jsonl(
        run_root / NATIVE_SIDECAR_REGISTRY_REL, "native-sidecar-registry.jsonl"
    )

    owner_context_by_id: dict[str, dict[str, Any]] = {}
    for row in owner_context_rows:
        owner_context_id = str(row.get("owner_context_id"))
        expected = f"{row.get('gt_owner_id')}@{row.get('context_id')}"
        if owner_context_id != expected:
            _fail(
                f"owner-context row owner_context_id {owner_context_id!r} does not match "
                f"gt_owner_id@context_id {expected!r}"
            )
        if owner_context_id in owner_context_by_id:
            _fail(f"owner-context-features.jsonl carries duplicate owner_context_id {owner_context_id!r}")
        owner_context_by_id[owner_context_id] = row

    contexts_by_id: dict[str, dict[str, Any]] = {}
    for row in contexts:
        context_id = str(row.get("context_id"))
        if context_id in contexts_by_id:
            _fail(f"context-registry.jsonl carries duplicate context_id {context_id!r}")
        contexts_by_id[context_id] = row

    return Inputs(
        run_root=run_root,
        file_sha256=file_sha256,
        plan_receipt=plan_receipt,
        merge_receipt=merge_receipt,
        owner_summaries=owner_summaries,
        owner_context_by_id=owner_context_by_id,
        contexts_by_id=contexts_by_id,
        native_sidecars=native_sidecars,
    )


def validate_inputs(inputs: Inputs) -> dict[str, Any]:
    """Every fail-closed precondition this reanalysis depends on.

    Returns a validation summary embedded in both the report and the receipt.
    """

    plan_receipt = inputs.plan_receipt
    merge_receipt = inputs.merge_receipt

    if plan_receipt.get("schema_version") != planner.PLAN_SCHEMA_VERSION:
        _fail("plan/receipt.json schema_version does not match the frozen census plan schema")
    if plan_receipt.get("unit_id") != PREDECESSOR_UNIT_ID:
        _fail("plan/receipt.json unit_id does not match the predecessor census unit")
    _reconstruct_receipt_digest(plan_receipt, label="plan/receipt.json")

    declared_plan_digests = plan_receipt.get("output_file_digests") or {}
    for relative_name, plan_key in (
        (CONTEXT_REGISTRY_REL, "context-registry.jsonl"),
        (NATIVE_SIDECAR_REGISTRY_REL, "native-sidecar-registry.jsonl"),
    ):
        if declared_plan_digests.get(plan_key) != inputs.file_sha256[relative_name]:
            _fail(
                f"{relative_name} bytes do not match the digest sealed in plan/receipt.json"
            )

    if merge_receipt.get("schema_version") != merge.MERGE_SCHEMA_VERSION:
        _fail("merge-receipt.json schema_version does not match the frozen census merge schema")
    if merge_receipt.get("unit_id") != PREDECESSOR_UNIT_ID:
        _fail("merge-receipt.json unit_id does not match the predecessor census unit")
    if merge_receipt.get("phase") != merge.PHASE_PRESENTATION:
        _fail("merge-receipt.json is not the presentation-phase receipt")
    _reconstruct_receipt_digest(merge_receipt, label="phases/presentation/merge-receipt.json")

    declared_merge_digests = merge_receipt.get("output_file_digests") or {}
    for relative_name, merge_key in (
        (OWNER_SUMMARIES_REL, "owner-summaries.jsonl"),
        (OWNER_CONTEXT_FEATURES_REL, "owner-context-features.jsonl"),
    ):
        if declared_merge_digests.get(merge_key) != inputs.file_sha256[relative_name]:
            _fail(
                f"{relative_name} bytes do not match the digest sealed in merge-receipt.json"
            )

    plan_block = merge_receipt.get("plan") or {}
    if plan_block.get("receipt_content_sha256") != plan_receipt.get("receipt_content_sha256"):
        _fail(
            "merge-receipt.json.plan.receipt_content_sha256 does not match "
            "plan/receipt.json.receipt_content_sha256; the presentation phase was not "
            "sealed against this exact plan"
        )

    if merge_receipt.get("usable_as_census_conclusion") is not True:
        _fail(
            "merge-receipt.json.usable_as_census_conclusion is not True; the predecessor "
            "census is not itself usable as a conclusion, so this reanalysis refuses to run"
        )

    counts = merge_receipt.get("counts") or {}
    if int(counts.get("owner_context_row_count", -1)) != len(inputs.owner_context_by_id):
        _fail(
            "merge-receipt.json.counts.owner_context_row_count does not match the number "
            "of rows read from owner-context-features.jsonl"
        )
    if int(counts.get("owner_summary_row_count", -1)) != len(inputs.owner_summaries):
        _fail(
            "merge-receipt.json.counts.owner_summary_row_count does not match the number "
            "of rows read from owner-summaries.jsonl"
        )

    for row in inputs.owner_summaries:
        if row.get("schema_version") != merge.OWNER_SUMMARY_SCHEMA_VERSION:
            _fail(
                f"owner-summaries.jsonl row for {row.get('gt_owner_id')!r} carries an "
                "unexpected schema_version"
            )
    for owner_context_id, row in inputs.owner_context_by_id.items():
        if row.get("schema_version") != merge.OWNER_CONTEXT_SCHEMA_VERSION:
            _fail(f"owner-context-features.jsonl row {owner_context_id!r} carries an unexpected schema_version")
    for context_id, row in inputs.contexts_by_id.items():
        if row.get("schema_version") != planner.PLAN_SCHEMA_VERSION:
            _fail(f"context-registry.jsonl row {context_id!r} carries an unexpected schema_version")
        if str(row.get("context_role")) not in NATIVE_CONTEXT_ROLES:
            _fail(
                f"context {context_id!r} has context_role {row.get('context_role')!r}, outside "
                f"the native root/row_boundary/terminal universe"
            )
        admission = row.get("prefix_admission") or {}
        if admission.get("forced_continue_rows_excluded") is not True:
            _fail(f"context {context_id!r} does not exclude forced-continue rows")
        if admission.get("source") != "native_greedy_complete_rows_only":
            _fail(f"context {context_id!r} prefix is not sourced from native greedy complete rows only")
    for row in inputs.native_sidecars:
        if row.get("schema_version") != planner.PLAN_SCHEMA_VERSION:
            _fail("native-sidecar-registry.jsonl carries a row with an unexpected schema_version")

    fn_owners = [
        row for row in inputs.owner_summaries if row.get("disposition") == merge.DISPOSITION_RESOLVED
    ]
    if len(fn_owners) != EXPECTED_FN_DENOMINATOR:
        _fail(
            f"expected exactly {EXPECTED_FN_DENOMINATOR} owners with disposition "
            f"{merge.DISPOSITION_RESOLVED!r}, found {len(fn_owners)}"
        )
    tp_owners = [row for row in inputs.owner_summaries if row.get("native_true_positive") is True]
    if len(tp_owners) != EXPECTED_TP_DENOMINATOR:
        _fail(
            f"expected exactly {EXPECTED_TP_DENOMINATOR} native_true_positive owners, "
            f"found {len(tp_owners)}"
        )

    return {
        "plan_receipt_content_sha256": plan_receipt["receipt_content_sha256"],
        "merge_receipt_content_sha256": merge_receipt["receipt_content_sha256"],
        "usable_as_census_conclusion": True,
        "owner_summary_row_count": len(inputs.owner_summaries),
        "owner_context_row_count": len(inputs.owner_context_by_id),
        "context_row_count": len(inputs.contexts_by_id),
        "native_sidecar_row_count": len(inputs.native_sidecars),
        "fn_denominator": len(fn_owners),
        "tp_denominator": len(tp_owners),
    }


def build_due_context_map(inputs: Inputs) -> dict[str, dict[str, Any]]:
    """Map each native true-positive owner to its exact due boundary context.

    This mapping is target-blind and score-blind: it comes only from
    ``native-sidecar-registry.jsonl``'s own ``strict_match_gt_owner_id`` and
    ``row_index``, joined to the due (pre-boundary) context
    ``<image>:boundary-{row_index:03d}``.  No owner-registry row, score, or
    frontier is consulted.
    """

    matched = [
        row
        for row in inputs.native_sidecars
        if row.get("strict_match_status") == "matched" and row.get("strict_match_gt_owner_id")
    ]
    by_owner: dict[str, list[dict[str, Any]]] = {}
    for row in matched:
        by_owner.setdefault(str(row["strict_match_gt_owner_id"]), []).append(row)

    due_map: dict[str, dict[str, Any]] = {}
    for owner_id, rows in by_owner.items():
        if len(rows) != 1:
            _fail(
                f"owner {owner_id!r} has {len(rows)} strict-matched native sidecar rows; "
                "expected exactly one unique native strict match"
            )
        row = rows[0]
        image_id = str(row["image_id"])
        row_index = int(row["row_index"])
        due_context_id = f"{image_id}:boundary-{row_index:03d}"
        if due_context_id not in inputs.contexts_by_id:
            _fail(
                f"due context {due_context_id!r} for owner {owner_id!r} is absent from "
                "context-registry.jsonl"
            )
        due_map[owner_id] = {
            "gt_owner_id": owner_id,
            "image_id": image_id,
            "pred_row_id": str(row["pred_row_id"]),
            "row_index": row_index,
            "due_context_id": due_context_id,
        }
    return due_map


# ---------------------------------------------------------------------------
# Per-context channel ladder
# ---------------------------------------------------------------------------


def _prefix_matched_owner_ids(context: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(
        str(row["strict_match_gt_owner_id"])
        for row in context.get("prefix_rows") or ()
        if row.get("strict_match_status") == "matched" and row.get("strict_match_gt_owner_id")
    )


#: Which owner-context field carries the same-context, same-category owner
#: competition for each ambiguity bound.  The false-negative sensitivity view
#: must rank owners under the L bound's own competition, never reuse the U
#: bound's rank or best-owner identity.
_OWNER_COMPETITION_FIELD_BY_BOUND: Mapping[str, str] = {
    "u": "owner_competition_u",
    "l": "owner_competition_l",
}


def extract_context_channel(
    owner_context_row: Mapping[str, Any], context: Mapping[str, Any], *, bound: str
) -> dict[str, Any]:
    """The transparent conjunction-ladder channels for one owner at one context.

    Every field is read verbatim from the sealed owner-context row; nothing is
    recomputed.  ``bound`` selects which ambiguity bound's same-context,
    same-category owner competition (``owner_competition_u`` or
    ``owner_competition_l``) supplies ``owner_rank_within_group`` and
    ``best_gt_owner_id``; this is never the local candidate-bank rank or the
    category-route rank, and the L view must never silently reuse the U
    bound's rank or competitor identity.
    """

    if bound not in _OWNER_COMPETITION_FIELD_BY_BOUND:
        _fail(f"extract_context_channel called with unknown bound {bound!r}")
    competition_field = _OWNER_COMPETITION_FIELD_BY_BOUND[bound]

    gt_owner_id = str(owner_context_row["gt_owner_id"])
    frontier = owner_context_row["frontier_features"]
    proposal = owner_context_row["category_proposal_channel"]
    gate = proposal.get("boundary_gate") or {}
    routing = proposal.get("category_routing_event") or {}
    competition = owner_context_row[competition_field]

    gate_margin = gate.get("continue_vs_stop_logprob_margin")
    gate_open = bool(gate_margin is not None and float(gate_margin) > 0.0)

    category_rank = routing.get("within_context_rank")
    category_rank_one = category_rank is not None and int(category_rank) == 1
    category_rank_top3 = category_rank is not None and int(category_rank) <= 3

    owner_rank = competition.get("rank")
    owner_rank_one = owner_rank is not None and int(owner_rank) == 1

    passed_state = str(frontier["passed_state"])
    before_or_at_frontier = passed_state in BEFORE_OR_AT_FRONTIER_STATES

    best_gt_owner_id = competition.get("best_gt_owner_id")
    if best_gt_owner_id is None:
        competitor_status = COMPETITOR_NO_POPULATION
    elif str(best_gt_owner_id) == gt_owner_id:
        competitor_status = COMPETITOR_TARGET
    elif str(best_gt_owner_id) in _prefix_matched_owner_ids(context):
        competitor_status = COMPETITOR_COVERED_OTHER
    else:
        competitor_status = COMPETITOR_UNCOVERED_OTHER

    favorable_top3 = gate_open and category_rank_top3 and owner_rank_one
    favorable_rank1 = gate_open and category_rank_one and owner_rank_one

    return {
        "bound": bound,
        "context_id": str(owner_context_row["context_id"]),
        "boundary_index": int(owner_context_row["boundary_index"]),
        "context_role": str(owner_context_row["context_role"]),
        "loop_tail": bool(owner_context_row["loop_marking"]["loop_tail"]),
        "gate_continue_vs_stop_logprob_margin": gate_margin,
        "gate_open": gate_open,
        "category_routing_within_context_rank": category_rank,
        "category_rank_one": category_rank_one,
        "category_rank_top3": category_rank_top3,
        "owner_rank_within_group": owner_rank,
        "owner_rank_one": owner_rank_one,
        "owner_competition_margin_to_best_owner": competition.get("margin_to_best_owner"),
        "owner_competition_population_size": competition.get("population_size"),
        "best_gt_owner_id": best_gt_owner_id,
        "passed_state": passed_state,
        "before_or_at_frontier": before_or_at_frontier,
        "competitor_status": competitor_status,
        "favorable_top3_any_frontier": favorable_top3,
        "favorable_top3_before_or_at_frontier": favorable_top3 and before_or_at_frontier,
        "favorable_rank1_any_frontier": favorable_rank1,
        "favorable_rank1_before_or_at_frontier": favorable_rank1 and before_or_at_frontier,
    }


# ---------------------------------------------------------------------------
# Owner-level records
# ---------------------------------------------------------------------------


def _bound_rollup(channels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    role_counts = Counter(str(c["context_role"]) for c in channels)
    competitor_counts = Counter(str(c["competitor_status"]) for c in channels)
    return {
        "support_context_count": len(channels),
        "context_role_counts": dict(role_counts),
        "any_root": role_counts.get("root", 0) > 0,
        "any_row_boundary": role_counts.get("row_boundary", 0) > 0,
        "any_terminal": role_counts.get("terminal", 0) > 0,
        "any_before_or_at_frontier": any(c["before_or_at_frontier"] for c in channels),
        "any_after_frontier": any(not c["before_or_at_frontier"] for c in channels),
        "any_gate_open": any(c["gate_open"] for c in channels),
        "any_category_rank_one": any(c["category_rank_one"] for c in channels),
        "any_category_rank_top3": any(c["category_rank_top3"] for c in channels),
        "any_owner_rank_one": any(c["owner_rank_one"] for c in channels),
        "any_favorable_top3_any_frontier": any(c["favorable_top3_any_frontier"] for c in channels),
        "any_favorable_top3_before_or_at_frontier": any(
            c["favorable_top3_before_or_at_frontier"] for c in channels
        ),
        "any_favorable_rank1_any_frontier": any(c["favorable_rank1_any_frontier"] for c in channels),
        "any_favorable_rank1_before_or_at_frontier": any(
            c["favorable_rank1_before_or_at_frontier"] for c in channels
        ),
        "competitor_status_counts": dict(competitor_counts),
        "any_target": competitor_counts.get(COMPETITOR_TARGET, 0) > 0,
        "any_covered_other": competitor_counts.get(COMPETITOR_COVERED_OTHER, 0) > 0,
        "any_uncovered_other": competitor_counts.get(COMPETITOR_UNCOVERED_OTHER, 0) > 0,
        "contexts": list(channels),
    }


#: The nonexclusive obstruction/asynchrony tags this analysis publishes.  Each
#: is a transparent boolean read off the already-published ``any_*`` U-bound
#: rollup fields; none is a fitted phenotype, none is mutually exclusive with
#: any other, and an owner may carry several (or none) at once.
OBSTRUCTION_FLAG_NAMES: tuple[str, ...] = (
    "never_gate_open",
    "never_category_rank_top3",
    "never_category_rank_one",
    "never_owner_rank_one",
    "support_only_after_frontier",
    "top3_and_owner_rank1_occur_but_never_at_the_same_context",
    "rank1_and_owner_rank1_occur_but_never_at_the_same_context",
    "favorable_top3_only_after_frontier",
    "favorable_rank1_only_after_frontier",
)


def _obstruction_flags(bound: Mapping[str, Any]) -> dict[str, Any]:
    """Which channel(s) block the conjunction for one owner, U-bound.

    Every flag is nonexclusive and descriptive: overlap is expected (an owner
    can be tagged "never gate open" and "never category rank top three" at
    once), and none of these labels a mechanism or decoder cause.
    """

    flags = {
        "never_gate_open": not bound["any_gate_open"],
        "never_category_rank_top3": not bound["any_category_rank_top3"],
        "never_category_rank_one": not bound["any_category_rank_one"],
        "never_owner_rank_one": not bound["any_owner_rank_one"],
        "support_only_after_frontier": (
            bool(bound["any_after_frontier"]) and not bound["any_before_or_at_frontier"]
        ),
        "top3_and_owner_rank1_occur_but_never_at_the_same_context": (
            bool(bound["any_category_rank_top3"])
            and bool(bound["any_owner_rank_one"])
            and not bound["any_favorable_top3_any_frontier"]
        ),
        "rank1_and_owner_rank1_occur_but_never_at_the_same_context": (
            bool(bound["any_category_rank_one"])
            and bool(bound["any_owner_rank_one"])
            and not bound["any_favorable_rank1_any_frontier"]
        ),
        "favorable_top3_only_after_frontier": (
            bool(bound["any_favorable_top3_any_frontier"])
            and not bound["any_favorable_top3_before_or_at_frontier"]
        ),
        "favorable_rank1_only_after_frontier": (
            bool(bound["any_favorable_rank1_any_frontier"])
            and not bound["any_favorable_rank1_before_or_at_frontier"]
        ),
    }
    flags["role"] = "nonexclusive_descriptive_obstruction_tags_never_a_fitted_or_causal_phenotype"
    return flags


def analyze_fn_owner(
    owner_summary: Mapping[str, Any],
    owner_context_by_id: Mapping[str, Mapping[str, Any]],
    contexts_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    gt_owner_id = str(owner_summary["gt_owner_id"])
    bounds: dict[str, dict[str, Any]] = {}
    for bound_name, bound_key in (("u", "upper_bound_u"), ("l", "lower_bound_l")):
        context_ids = list(owner_summary[bound_key]["usable_support_context_ids"])
        channels: list[dict[str, Any]] = []
        for context_id in context_ids:
            owner_context_id = f"{gt_owner_id}@{context_id}"
            owner_context_row = owner_context_by_id.get(owner_context_id)
            if owner_context_row is None:
                _fail(
                    f"owner {gt_owner_id!r} usable support context {context_id!r} ({bound_name} "
                    "bound) has no matching owner-context-features.jsonl row"
                )
            if bool(owner_context_row["loop_marking"]["loop_tail"]):
                _fail(
                    f"owner {gt_owner_id!r} usable support context {context_id!r} ({bound_name} "
                    "bound) is loop_tail; usable non-loop support must exclude loop tails"
                )
            context = contexts_by_id.get(context_id)
            if context is None:
                _fail(f"usable support context {context_id!r} is absent from context-registry.jsonl")
            channels.append(extract_context_channel(owner_context_row, context, bound=bound_name))
        bounds[bound_name] = _bound_rollup(channels)

    never_owner_rank_one = not bounds["u"]["any_owner_rank_one"]
    only_covered_other = (
        never_owner_rank_one
        and bounds["u"]["any_covered_other"]
        and not bounds["u"]["any_uncovered_other"]
    )
    only_uncovered_other = (
        never_owner_rank_one
        and bounds["u"]["any_uncovered_other"]
        and not bounds["u"]["any_covered_other"]
    )

    return {
        "schema_version": OWNER_RECORD_SCHEMA_VERSION,
        "cohort": FN_COHORT,
        "gt_owner_id": gt_owner_id,
        "image_id": str(owner_summary["image_id"]),
        "normalized_description": str(owner_summary["normalized_description"]),
        "disposition": str(owner_summary["disposition"]),
        "upper_bound_u": bounds["u"],
        "lower_bound_l": bounds["l"],
        "ambiguity_bound_disposition_flip": bool(owner_summary.get("ambiguity_bound_disposition_flip")),
        "never_owner_rank_one": never_owner_rank_one,
        "only_covered_other_among_never_rank_one": only_covered_other,
        "only_uncovered_other_among_never_rank_one": only_uncovered_other,
        "obstruction_flags": _obstruction_flags(bounds["u"]),
    }


def analyze_native_tp_owner(
    owner_summary: Mapping[str, Any],
    due_info: Mapping[str, Any],
    owner_context_by_id: Mapping[str, Mapping[str, Any]],
    contexts_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    gt_owner_id = str(owner_summary["gt_owner_id"])
    due_context_id = str(due_info["due_context_id"])
    owner_context_id = f"{gt_owner_id}@{due_context_id}"
    owner_context_row = owner_context_by_id.get(owner_context_id)
    if owner_context_row is None:
        _fail(
            f"native true-positive owner {gt_owner_id!r} due context {due_context_id!r} has no "
            "matching owner-context-features.jsonl row"
        )
    context = contexts_by_id.get(due_context_id)
    if context is None:
        _fail(f"due context {due_context_id!r} is absent from context-registry.jsonl")

    # The native true-positive reference uses the exact due boundary, never a
    # bound-sensitivity view; "owner rank one" for this reference is
    # owner_competition_u.rank, matching the frozen operational definition.
    channel = extract_context_channel(owner_context_row, context, bound="u")
    non_loop_support = (owner_summary.get("upper_bound_u") or {}).get("non_loop_context_support") or {}
    due_support = non_loop_support.get(due_context_id)
    due_support_status = (
        "supported"
        if due_support is True
        else "not_supported"
        if due_support is False
        else "excluded_loop_tail_or_unavailable"
    )
    due_support_bool = bool(due_support is True)

    return {
        "schema_version": OWNER_RECORD_SCHEMA_VERSION,
        "cohort": TP_COHORT,
        "gt_owner_id": gt_owner_id,
        "image_id": str(owner_summary["image_id"]),
        "normalized_description": str(owner_summary["normalized_description"]),
        "due_context_id": due_context_id,
        "due_pred_row_id": str(due_info["pred_row_id"]),
        "due_row_index": int(due_info["row_index"]),
        "due_context_support_status": due_support_status,
        "due_context_support": due_support_bool,
        "channel": channel,
        "favorable_top3_any_frontier": due_support_bool and channel["favorable_top3_any_frontier"],
        "favorable_top3_before_or_at_frontier": (
            due_support_bool and channel["favorable_top3_before_or_at_frontier"]
        ),
        "favorable_rank1_any_frontier": due_support_bool and channel["favorable_rank1_any_frontier"],
        "favorable_rank1_before_or_at_frontier": (
            due_support_bool and channel["favorable_rank1_before_or_at_frontier"]
        ),
    }


# ---------------------------------------------------------------------------
# Wilson intervals and headline aggregation
# ---------------------------------------------------------------------------

_Z_95 = 1.959963984540054


def wilson_interval(successes: int, total: int, *, z: float = _Z_95) -> dict[str, float | int | None]:
    if total <= 0:
        return {"successes": successes, "total": total, "proportion": None, "lower": None, "upper": None}
    phat = successes / total
    denom = 1.0 + (z * z) / total
    center = phat + (z * z) / (2.0 * total)
    margin = z * math.sqrt((phat * (1.0 - phat)) / total + (z * z) / (4.0 * total * total))
    return {
        "successes": successes,
        "total": total,
        "proportion": phat,
        "lower": max(0.0, (center - margin) / denom),
        "upper": min(1.0, (center + margin) / denom),
    }


def _rate(records: Sequence[Mapping[str, Any]], predicate) -> dict[str, Any]:
    total = len(records)
    successes = sum(1 for record in records if predicate(record))
    return wilson_interval(successes, total)


#: The full primary conjunction ladder, broken out per image and per category
#: in addition to the overall Wilson-interval owner-level rate.  Every entry
#: here is an "any usable-support context satisfies this" owner-level binary;
#: none of them is a context-row count.
LADDER_KEYS: tuple[str, ...] = (
    "any_root",
    "any_row_boundary",
    "any_terminal",
    "any_before_or_at_frontier",
    "any_after_frontier",
    "any_gate_open",
    "any_category_rank_one",
    "any_category_rank_top3",
    "any_owner_rank_one",
    "any_favorable_top3_any_frontier",
    "any_favorable_top3_before_or_at_frontier",
    "any_favorable_rank1_any_frontier",
    "any_favorable_rank1_before_or_at_frontier",
    "any_target",
    "any_covered_other",
    "any_uncovered_other",
)

#: Owner-context-support-row-weighted histograms (FN arm): each unit is one
#: usable-support (owner, context) row, so an owner with many support
#: contexts contributes many rows to the histogram total even though it is a
#: single owner in every "any_*" owner-level count above.  This is a
#: deliberately different unit from the owner-level metrics and must never be
#: read as an owner count.
FN_HISTOGRAM_UNIT = "owner_context_support_rows"
FN_HISTOGRAM_WEIGHTING = "context_row_weighted_not_owner_deduplicated"


def _labelled_histogram(counter: Mapping[int, int], *, unit: str, weighting: str) -> dict[str, Any]:
    return {
        "unit": unit,
        "weighting": weighting,
        "total": sum(counter.values()),
        "counts": dict(sorted(counter.items())),
    }


def aggregate_fn_headline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)

    def bound_metrics(bound_key: str) -> dict[str, Any]:
        def get(record: Mapping[str, Any]) -> Mapping[str, Any]:
            return record[bound_key]

        metrics = {key: _rate(records, lambda r, key=key: get(r)[key]) for key in LADDER_KEYS}
        support_counts = Counter(int(get(r)["support_context_count"]) for r in records)
        category_rank_hist = Counter()
        owner_rank_hist = Counter()
        for record in records:
            for channel in get(record)["contexts"]:
                if channel["category_routing_within_context_rank"] is not None:
                    category_rank_hist[int(channel["category_routing_within_context_rank"])] += 1
                if channel["owner_rank_within_group"] is not None:
                    owner_rank_hist[int(channel["owner_rank_within_group"])] += 1

        def _strata_counts(grouped: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
            return {
                stratum_key: {
                    "owner_count": len(stratum_records),
                    **{
                        key: sum(1 for r in stratum_records if get(r)[key]) for key in LADDER_KEYS
                    },
                }
                for stratum_key, stratum_records in grouped.items()
            }

        per_image = _strata_counts(_group_by(records, lambda r: r["image_id"]))
        per_category = _strata_counts(_group_by(records, lambda r: r["normalized_description"]))
        return {
            "metrics": metrics,
            "support_context_count_histogram": dict(sorted(support_counts.items())),
            "category_routing_rank_histogram": _labelled_histogram(
                category_rank_hist, unit=FN_HISTOGRAM_UNIT, weighting=FN_HISTOGRAM_WEIGHTING
            ),
            "owner_rank_histogram": _labelled_histogram(
                owner_rank_hist, unit=FN_HISTOGRAM_UNIT, weighting=FN_HISTOGRAM_WEIGHTING
            ),
            "per_image": per_image,
            "per_category": per_category,
        }

    def _wilson_strata_view(
        subset: Sequence[Mapping[str, Any]], predicates: Mapping[str, Any]
    ) -> dict[str, Any]:
        def strata(grouped: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
            return {
                stratum_key: {
                    "owner_count": len(stratum_records),
                    **{
                        name: sum(1 for r in stratum_records if predicate(r))
                        for name, predicate in predicates.items()
                    },
                }
                for stratum_key, stratum_records in grouped.items()
            }

        return {
            "owner_count": len(subset),
            "metrics": {name: _rate(subset, predicate) for name, predicate in predicates.items()},
            "per_image": strata(_group_by(subset, lambda r: r["image_id"])),
            "per_category": strata(_group_by(subset, lambda r: r["normalized_description"])),
        }

    never_rank_one = [r for r in records if r["never_owner_rank_one"]]

    # Owner-level target/covered/uncovered competitor prevalence over every
    # supported false-negative owner (all 114), independent of whether that
    # owner ever attains owner rank one.
    all_owners_predicates = {
        "any_target": lambda r: r["upper_bound_u"]["any_target"],
        "any_covered_other": lambda r: r["upper_bound_u"]["any_covered_other"],
        "any_uncovered_other": lambda r: r["upper_bound_u"]["any_uncovered_other"],
    }
    competitor_all_supported_fn_owners = _wilson_strata_view(records, all_owners_predicates)
    competitor_all_supported_fn_owners["role"] = (
        "owner_level_target_covered_uncovered_prevalence_over_every_supported_false_negative_owner"
    )

    never_rank_one_predicates = {
        "any_covered_other": lambda r: r["upper_bound_u"]["any_covered_other"],
        "any_uncovered_other": lambda r: r["upper_bound_u"]["any_uncovered_other"],
        "only_covered_other": lambda r: r["only_covered_other_among_never_rank_one"],
        "only_uncovered_other": lambda r: r["only_uncovered_other_among_never_rank_one"],
    }
    never_rank_one_view = _wilson_strata_view(never_rank_one, never_rank_one_predicates)

    competitor_summary = {
        "never_owner_rank_one_count": len(never_rank_one),
        # New, explicitly named block: owner-level prevalence over all 114
        # supported false-negative owners.
        "all_supported_fn_owners": competitor_all_supported_fn_owners,
        # Original block, unchanged in shape and meaning: owner-level
        # prevalence over only the owners that never attain owner rank one.
        # Plain counts are kept for continuity with earlier reports.
        "any_covered_other": never_rank_one_view["metrics"]["any_covered_other"]["successes"],
        "any_uncovered_other": never_rank_one_view["metrics"]["any_uncovered_other"]["successes"],
        "only_covered_other": never_rank_one_view["metrics"]["only_covered_other"]["successes"],
        "only_uncovered_other": never_rank_one_view["metrics"]["only_uncovered_other"]["successes"],
        "metrics": never_rank_one_view["metrics"],
        "per_image": never_rank_one_view["per_image"],
        "per_category": never_rank_one_view["per_category"],
        "only_covered_other_per_image": dict(
            Counter(
                r["image_id"] for r in never_rank_one if r["only_covered_other_among_never_rank_one"]
            )
        ),
    }

    obstruction_summary: dict[str, Any] = {}
    for flag in OBSTRUCTION_FLAG_NAMES:
        owner_ids = sorted(r["gt_owner_id"] for r in records if r["obstruction_flags"][flag])
        obstruction_summary[flag] = {"count": len(owner_ids), "owner_ids": owner_ids}
    obstruction_summary["role"] = (
        "nonexclusive_descriptive_obstruction_tags_over_the_u_bound_overlap_is_expected_"
        "never_a_causal_or_exclusive_label"
    )

    return {
        "denominator": total,
        "upper_bound_u": bound_metrics("upper_bound_u"),
        "lower_bound_l": bound_metrics("lower_bound_l"),
        "bound_role": "upper_bound_u_is_primary_lower_bound_l_is_sensitivity_only",
        "competitor_status_among_never_owner_rank_one": competitor_summary,
        "obstruction_summary": obstruction_summary,
    }


def _group_by(records: Sequence[Mapping[str, Any]], key) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(key(record)), []).append(record)
    return grouped


#: The native true-positive reference's field ladder, at the exact due
#: boundary.  This mirrors the false-negative arm's atomic channels
#: (``LADDER_KEYS``) so the two cohorts are reported on the same ladder, per
#: the frozen requirement to "report the exact same field ladder for native
#: true positives at their due boundary".  There is one context per owner
#: here (not an "any usable-support context" operator), so there is no
#: root/row-boundary/terminal *coverage* axis to report for this cohort.
_TP_LADDER_PREDICATES: dict[str, Any] = {
    "due_context_support": lambda r: r["due_context_support"],
    "any_gate_open": lambda r: r["channel"]["gate_open"],
    "category_rank_one": lambda r: r["channel"]["category_rank_one"],
    "category_rank_top3": lambda r: r["channel"]["category_rank_top3"],
    "owner_rank_one": lambda r: r["channel"]["owner_rank_one"],
    "before_or_at_frontier": lambda r: r["channel"]["before_or_at_frontier"],
    "favorable_top3_any_frontier": lambda r: r["favorable_top3_any_frontier"],
    "favorable_top3_before_or_at_frontier": lambda r: r["favorable_top3_before_or_at_frontier"],
    "favorable_rank1_any_frontier": lambda r: r["favorable_rank1_any_frontier"],
    "favorable_rank1_before_or_at_frontier": lambda r: r["favorable_rank1_before_or_at_frontier"],
    # Due-context role coverage: each TP owner has exactly one due context, so
    # these are owner-level role predicates, not an "any usable-support
    # context" operator.
    "due_context_role_root": lambda r: r["channel"]["context_role"] == "root",
    "due_context_role_row_boundary": lambda r: r["channel"]["context_role"] == "row_boundary",
    "due_context_role_terminal": lambda r: r["channel"]["context_role"] == "terminal",
}
TP_LADDER_KEYS: tuple[str, ...] = tuple(_TP_LADDER_PREDICATES)

#: TP rank histograms are owner-weighted: exactly one due-boundary row per
#: native true-positive owner, never a context-row-weighted count like the
#: FN arm's histograms.
TP_HISTOGRAM_UNIT = "native_true_positive_owners"
TP_HISTOGRAM_WEIGHTING = "one_row_per_owner_at_the_exact_due_boundary"


def aggregate_native_tp_headline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    channel_of = lambda r: r["channel"]  # noqa: E731

    metrics = {
        key: _rate(records, predicate) for key, predicate in _TP_LADDER_PREDICATES.items()
    }
    # Kept as a named overall-only joint metric alongside the atomic ladder
    # above; it does not replace either atomic channel.
    metrics["gate_open_and_category_rank_one"] = _rate(
        records, lambda r: channel_of(r)["gate_open"] and channel_of(r)["category_rank_one"]
    )

    def _strata_counts(grouped: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
        return {
            stratum_key: {
                "owner_count": len(stratum_records),
                **{
                    key: sum(1 for r in stratum_records if predicate(r))
                    for key, predicate in _TP_LADDER_PREDICATES.items()
                },
            }
            for stratum_key, stratum_records in grouped.items()
        }

    per_image = _strata_counts(_group_by(records, lambda r: r["image_id"]))
    per_category = _strata_counts(_group_by(records, lambda r: r["normalized_description"]))

    category_rank_hist = Counter(
        int(channel_of(r)["category_routing_within_context_rank"])
        for r in records
        if channel_of(r)["category_routing_within_context_rank"] is not None
    )
    owner_rank_hist = Counter(
        int(channel_of(r)["owner_rank_within_group"])
        for r in records
        if channel_of(r)["owner_rank_within_group"] is not None
    )

    competitor_counts = Counter(channel_of(r)["competitor_status"] for r in records)
    return {
        "denominator": total,
        "metrics": metrics,
        "per_image": per_image,
        "per_category": per_category,
        "category_routing_rank_histogram": _labelled_histogram(
            category_rank_hist, unit=TP_HISTOGRAM_UNIT, weighting=TP_HISTOGRAM_WEIGHTING
        ),
        "owner_rank_histogram": _labelled_histogram(
            owner_rank_hist, unit=TP_HISTOGRAM_UNIT, weighting=TP_HISTOGRAM_WEIGHTING
        ),
        "competitor_status_counts": dict(competitor_counts),
        "role": "descriptive_positive_reference_never_a_matched_causal_control",
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(
    *,
    inputs: Inputs,
    validation: Mapping[str, Any],
    fn_records: Sequence[Mapping[str, Any]],
    tp_records: Sequence[Mapping[str, Any]],
    fn_headline: Mapping[str, Any],
    tp_headline: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "predecessor_unit_id": PREDECESSOR_UNIT_ID,
        "predecessor_run_root": str(inputs.run_root),
        "input_file_sha256": dict(inputs.file_sha256),
        "validation": dict(validation),
        "operational_definitions": {
            "supported_false_negative": (
                f"owner disposition == {merge.DISPOSITION_RESOLVED!r}; denominator exactly "
                f"{EXPECTED_FN_DENOMINATOR}"
            ),
            "usable_support_context": "context_id in upper_bound_u.usable_support_context_ids",
            "gate_open": "boundary_gate.continue_vs_stop_logprob_margin > 0",
            "category_rank_one_or_top_three": (
                "category_routing_event.within_context_rank == 1, or <= 3"
            ),
            "owner_rank_one": "owner_competition_u.rank == 1",
            "before_or_at_frontier": (
                "frontier_features.passed_state in "
                "{root_no_frontier, ahead_of_frontier, at_frontier}"
            ),
            "covered_competitor": (
                "best owner at a support context is present among that context's "
                "prefix_rows with strict_match_status == matched"
            ),
        },
        "semantics": list(SEMANTICS_NOTES),
        "false_negative_cohort": {
            "cohort": FN_COHORT,
            "denominator": len(fn_records),
            "headline": fn_headline,
        },
        "native_true_positive_reference": {
            "cohort": TP_COHORT,
            "denominator": len(tp_records),
            "headline": tp_headline,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# {UNIT_ID}")
    lines.append("")
    lines.append(f"Predecessor run root: `{report['predecessor_run_root']}`")
    lines.append("")
    lines.append("## Validation")
    lines.append("")
    for key, value in report["validation"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Semantics")
    lines.append("")
    for note in report["semantics"]:
        lines.append(f"- {note}")
    lines.append("")

    fn = report["false_negative_cohort"]
    lines.append(f"## False-negative cohort ({fn['cohort']}, n={fn['denominator']})")
    lines.append("")
    u_metrics = fn["headline"]["upper_bound_u"]["metrics"]
    lines.append("| U-bound metric | successes | total | proportion | 95% CI |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, interval in u_metrics.items():
        proportion = interval["proportion"]
        lower = interval["lower"]
        upper = interval["upper"]
        proportion_str = "n/a" if proportion is None else f"{proportion:.3f}"
        ci_str = "n/a" if lower is None else f"[{lower:.3f}, {upper:.3f}]"
        lines.append(
            f"| {name} | {interval['successes']} | {interval['total']} | {proportion_str} | {ci_str} |"
        )
    lines.append("")
    competitor = fn["headline"]["competitor_status_among_never_owner_rank_one"]
    lines.append("### Competitor status among owners that never attain owner rank one")
    lines.append("")
    for key, value in competitor.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")

    obstruction = fn["headline"]["obstruction_summary"]
    lines.append("### Obstruction/asynchrony tags (nonexclusive, U-bound; overlap is expected)")
    lines.append("")
    for flag in OBSTRUCTION_FLAG_NAMES:
        lines.append(f"- `{flag}`: {obstruction[flag]['count']}")
    lines.append("")

    tp = report["native_true_positive_reference"]
    lines.append(f"## Native true-positive due-boundary reference (n={tp['denominator']})")
    lines.append("")
    lines.append("| metric | successes | total | proportion | 95% CI |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, interval in tp["headline"]["metrics"].items():
        proportion = interval["proportion"]
        lower = interval["lower"]
        upper = interval["upper"]
        proportion_str = "n/a" if proportion is None else f"{proportion:.3f}"
        ci_str = "n/a" if lower is None else f"[{lower:.3f}, {upper:.3f}]"
        lines.append(
            f"| {name} | {interval['successes']} | {interval['total']} | {proportion_str} | {ci_str} |"
        )
    lines.append("")
    lines.append(f"Competitor status counts: {tp['headline']['competitor_status_counts']}")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_analysis(run_root: Path) -> dict[str, Any]:
    """Load, validate, and analyze; returns report/records/receipt payloads."""

    inputs = load_inputs(run_root)
    validation = validate_inputs(inputs)
    due_context_map = build_due_context_map(inputs)

    fn_summaries = [
        row for row in inputs.owner_summaries if row.get("disposition") == merge.DISPOSITION_RESOLVED
    ]
    tp_summaries = [row for row in inputs.owner_summaries if row.get("native_true_positive") is True]

    tp_owner_ids = {str(row["gt_owner_id"]) for row in tp_summaries}
    if set(due_context_map) != tp_owner_ids:
        missing = sorted(tp_owner_ids - set(due_context_map))
        extra = sorted(set(due_context_map) - tp_owner_ids)
        _fail(
            "native true-positive due-context map does not exactly cover the "
            f"{EXPECTED_TP_DENOMINATOR} native_true_positive owners "
            f"(missing={missing}, extra={extra})"
        )

    fn_records = [
        analyze_fn_owner(row, inputs.owner_context_by_id, inputs.contexts_by_id) for row in fn_summaries
    ]
    tp_records = [
        analyze_native_tp_owner(
            row, due_context_map[str(row["gt_owner_id"])], inputs.owner_context_by_id, inputs.contexts_by_id
        )
        for row in tp_summaries
    ]

    fn_headline = aggregate_fn_headline(fn_records)
    tp_headline = aggregate_native_tp_headline(tp_records)

    report = build_report(
        inputs=inputs,
        validation=validation,
        fn_records=fn_records,
        tp_records=tp_records,
        fn_headline=fn_headline,
        tp_headline=tp_headline,
    )
    owner_records = [*fn_records, *tp_records]
    return {"report": report, "owner_records": owner_records}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Immutable predecessor sorted owner accessibility phenotype census run root",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output root")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = Path(args.output_root) / "analysis"
    if output_root.exists() and not args.force:
        existing = [p.name for p in output_root.iterdir()]
        if existing:
            raise SystemExit(f"refusing to overwrite non-empty {output_root}; pass --force")

    try:
        result = run_analysis(Path(args.run_root))
    except AnalysisContractError as exc:
        raise SystemExit(f"analysis contract violated: {exc}") from exc

    report = result["report"]
    owner_records = result["owner_records"]

    output_root.mkdir(parents=True, exist_ok=True)
    report_json_text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    (output_root / REPORT_JSON_NAME).write_text(report_json_text, encoding="utf-8")
    report_md_text = render_markdown(report)
    (output_root / REPORT_MD_NAME).write_text(report_md_text, encoding="utf-8")
    owner_records_text = "".join(
        canonical_json_bytes(row).decode("utf-8") + "\n" for row in owner_records
    )
    (output_root / OWNER_RECORDS_NAME).write_text(owner_records_text, encoding="utf-8")

    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "predecessor_unit_id": PREDECESSOR_UNIT_ID,
        "predecessor_run_root": str(args.run_root),
        "input_file_sha256": dict(report["input_file_sha256"]),
        "validation": dict(report["validation"]),
        "analyzer_source_sha256": sha256_bytes(Path(__file__).resolve().read_bytes()),
        "report_json_sha256": sha256_bytes(report_json_text.encode("utf-8")),
        "report_md_sha256": sha256_bytes(report_md_text.encode("utf-8")),
        "owner_records_jsonl_sha256": sha256_bytes(owner_records_text.encode("utf-8")),
        "fn_denominator": report["false_negative_cohort"]["denominator"],
        "tp_denominator": report["native_true_positive_reference"]["denominator"],
    }
    receipt["receipt_content_sha256"] = sha256_json(receipt)
    (output_root / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
