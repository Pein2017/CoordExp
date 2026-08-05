#!/usr/bin/env python3
"""CPU-only merge for the sorted owner accessibility phenotype census
(``2026-08-03-sorted-owner-accessibility-phenotype-census``).

This module scores nothing.  It admits the twelve sealed per-image shards
against the frozen plan, publishes one deterministic census event table, and
derives the owner-context and owner-level views the phenotype analysis reads.

What it consumes
----------------
``--plan-dir``
    The planner's sealed directory (``receipt.json``, ``capture-rules.json``
    and the eight registries).  Every declared file is re-digested before a
    single row is read.
``--shard-root``
    One subdirectory per image, named ``<image_id>`` or ``shard-<image_id>``.
    A shard is *captured* when it holds ``shard-receipt.json``, *quarantined*
    when it holds ``shard-quarantine.json``, and *missing* otherwise.

What it publishes
-----------------
``census-events.jsonl``
    One row per admitted unique coordinate event, carrying the competition
    rank, the margin to the group best, and the renormalized within-group
    posterior.  Ranks are computed strictly within ``(image, context,
    category)`` over collapsed unique physical candidates; native and
    free-decode sidecars never enter that population.
``owner-context-features.jsonl``
    The score-independent continuous owner-context reconstruction (signed
    frontier ordinal/pixel distances, same-description owners ahead, passed
    state, frontier overlap/IoU/center/extent) joined to the five mandated
    localization quantities and to the *separate* category proposal channel.
``owner-summaries.jsonl``
    Per-owner diagnostic and primary summaries plus the evidence-state
    disposition.
``merge-receipt.json``
    Lineage, digests, quarantine ledger, and every count this merge asserts.

Why the merge re-derives geometry from the plan
-----------------------------------------------
Score rows are treated as *values only*: ``complete_box_logprob_sum`` plus the
identity digests needed to prove where the value came from.  Owner identity,
generator provenance, alias membership and assignment all come back from the
sealed candidate bank, so a scorer that carries a stale or enriched geometry
block can never move an owner's evidence.

The five mandated per-owner-context localization quantities
-----------------------------------------------------------
The primary owner neighbourhood is the ``generator_local_landscape``: the
alias-collapsed physical candidates the owner's seventeen fixed
score-independent roles reached.  Strict assignment is a *separate*,
owner-identifiable lower bound and is never bank membership.  Each
owner-context therefore reports:

``generator_local_max``
    Over the whole generator-local neighbourhood, unfiltered.
``exclusion_filtered_primary_max``
    Over the neighbourhood minus every candidate strictly assigned to another
    owner.  Those excluded candidates move to the collision diagnostic; they
    never count as target support.
``strict_assigned_max``
    Over candidates strictly assigned to this owner (the parallel strict table).
``exact_anchor_score``
    The owner's mandatory exact anchor, reported on its own.
``ambiguous_upper_max``
    Over ambiguity-neutral candidates in the neighbourhood; these count toward
    the target only through the U bound.

The L and U ambiguity bounds are the same exclusion-filtered maximum with
ambiguity-neutral candidates excluded (L) and included (U).  Any disposition
that flips between them closes the owner ``unresolved``.

Bank adequacy is *bound*, never chosen here.  The lead froze the bands before
any score existed and the planner sealed them into
``capture-rules.json → owner_support.bank_adequacy_rule``: an admitted,
self-localizing exact anchor plus a distinct alias-collapsed candidate count at
or above ``adequate_at_least`` is disposition-eligible, ``full_at_least`` is
``full``, the band between is ``adequate_reduced`` (eligible, flagged), and
anything below is ``undercovered_unresolved_only``.  This module reads those
numbers and refuses to run if they are absent; it never substitutes its own.

Two-phase discovery and confirmation
------------------------------------
This merge emits **continuous features and evidence states only**.  It derives
no phenotype and chooses no cut.  :data:`OWNER_FEATURE_CATALOG` is the published
vocabulary: each entry names a quantity, defines it, and states its direction of
meaning.

A phenotype rule is an *explicit, authored* artifact.  The intended sequence is:

1. ``--phase describe-discovery`` prints the discovery-half distributions,
   including the native true-positive calibration stratum.  It is a diagnostic
   suggestion only: it emits no threshold, no cut, and no phenotype, and its
   output cannot be fed to confirmation.
2. The lead writes the rule by hand -- which features, which directions, which
   thresholds, on which bound and view, with provenance naming what was
   inspected and an explicit disclaimer of any causal claim.
3. ``--phase seal-rule`` validates that rule against the published catalog and
   content-addresses it.  It derives nothing.
4. ``--phase confirmation`` binds that exact digest and applies the rule
   mechanically to the held-out half.  There is no override seam.

No causal label is ever emitted.  ``loop_tail`` is the planner's literal
bookkeeping flag, dispositions are evidence states, and every published label
carries ``is_causal_label: false``.  Cross-context deltas are named as deltas;
the only quantity called a competition margin is the one taken from a single
``(context, category)`` rank population.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_owner_accessibility_census_plan as planner  # noqa: E402

UNIT_ID = planner.UNIT_ID
MERGE_SCHEMA_VERSION = "sorted-owner-accessibility-census-merge.v1"
EVENT_SCHEMA_VERSION = "sorted-owner-accessibility-census-event.v1"
OWNER_CONTEXT_SCHEMA_VERSION = "sorted-owner-accessibility-census-owner-context.v1"
OWNER_SUMMARY_SCHEMA_VERSION = "sorted-owner-accessibility-census-owner-summary.v1"
SIDECAR_DIAGNOSTIC_SCHEMA_VERSION = "sorted-owner-accessibility-census-sidecar-diagnostic.v1"
DISCOVERY_RULE_SCHEMA_VERSION = "sorted-owner-accessibility-census-discovery-rule.v1"
DISCOVERY_DISTRIBUTION_SCHEMA_VERSION = (
    "sorted-owner-accessibility-census-discovery-distributions.v1"
)
CONFIRMATION_SCHEMA_VERSION = "sorted-owner-accessibility-census-confirmation.v1"

#: Shard artifact names, as written by ``score_sorted_owner_accessibility_census_shard``.
SHARD_RECEIPT_NAME = "shard-receipt.json"
SHARD_QUARANTINE_NAME = "shard-quarantine.json"
SHARD_SCORES_NAME = "census-scores.jsonl"
SHARD_PROPOSAL_NAME = "proposal-surface.jsonl"
SHARD_FREE_DECODE_NAME = "free-decode-sidecars.jsonl"
SHARD_X1_NAME = "x1-distributions.jsonl"
SHARD_EVIDENCE_NAMES: tuple[str, ...] = (
    SHARD_SCORES_NAME,
    SHARD_PROPOSAL_NAME,
    SHARD_FREE_DECODE_NAME,
    SHARD_X1_NAME,
)

#: Expected shard-side schema identities.  Held locally so a scorer schema bump
#: fails this merge closed instead of silently joining a foreign row shape.
EXPECTED_SHARD_RECEIPT_SCHEMA = "sorted-owner-accessibility-census-shard-receipt.v1"
EXPECTED_SCORE_SCHEMA = "sorted-owner-accessibility-census-score.v1"
EXPECTED_PROPOSAL_SCHEMA = "sorted-owner-accessibility-census-proposal.v1"
EXPECTED_FREE_DECODE_SCHEMA = "sorted-owner-accessibility-census-free-decode.v1"
EXPECTED_ADMISSION_SCHEMA = "sorted-owner-accessibility-census-admission.v1"
EXPECTED_QUARANTINE_SCHEMA = "sorted-owner-accessibility-census-shard-quarantine.v1"

EXPECTED_IMAGE_COUNT = 12

#: Output artifact names.
EVENTS_NAME = "census-events.jsonl"
OWNER_CONTEXT_NAME = "owner-context-features.jsonl"
OWNER_SUMMARY_NAME = "owner-summaries.jsonl"
SIDECAR_DIAGNOSTIC_NAME = "sidecar-diagnostics.jsonl"
MERGE_RECEIPT_NAME = "merge-receipt.json"
DISCOVERY_RULE_NAME = "discovery-rule.json"
DISCOVERY_DISTRIBUTION_NAME = "discovery-distributions.json"
CAPTURE_MANIFEST_NAME = "capture-manifest.json"
CALIBRATION_NAME = "support-calibration.json"
CONFIRMATION_REPORT_NAME = "confirmation-report.json"

#: The continuous quantities this merge publishes per owner.  This module owns
#: the *vocabulary and its definitions*; it does not own, and never derives,
#: which of them a phenotype rule cuts on or where.  A sealed rule may only name
#: features from this catalog, so a rule can never cut on an undefined or
#: unpublished quantity.
OWNER_FEATURE_CATALOG: Mapping[str, str] = {
    "primary_exclusion_filtered_max": (
        "complete-box logprob sum of the owner's exclusion-filtered generator-local "
        "maximum in the selected view; higher is stronger localization support"
    ),
    "margin_to_best_owner_in_group": (
        "owner's exclusion-filtered maximum minus the best owner's, ranked inside the "
        "same (context, category) population; zero means this owner is the group best; "
        "this is the only true competition margin"
    ),
    "owner_rank_within_group": (
        "1-based rank of this owner among owners of the same (context, category)"
    ),
    "exact_anchor_score": "complete-box logprob sum of the owner's mandatory exact anchor",
    "strict_assigned_max": (
        "maximum over candidates strictly assigned to this owner; the parallel "
        "owner-identifiable lower bound, never bank membership"
    ),
    "signed_frontier_ordinal_distance": (
        "owner sort ordinal minus the frontier's insertion ordinal; positive means the "
        "owner lies ahead of the frontier, negative means the sweep has passed it"
    ),
    "abs_signed_frontier_ordinal_distance": "absolute value of the signed ordinal distance",
    "signed_frontier_sort_axis_pixel_distance": (
        "signed pixel distance along the geometry sort axis (y, then x on ties)"
    ),
    "frontier_intersection_over_union": "IoU between the owner box and the frontier row's box",
    "same_description_owners_ahead_of_frontier": (
        "count of same-description owners not yet passed by the frontier"
    ),
    "boundary_gate_continue_vs_stop_logprob_margin": (
        "proposal channel: continue minus stop logprob at the natural boundary; a gate, "
        "never description accessibility"
    ),
    "category_routing_within_context_rank": (
        "proposal channel: rank of this description's routing event within the context"
    ),
    "distinct_physical_candidate_count": (
        "alias-collapsed distinct physical candidates the owner's fixed roles reached"
    ),
    "uniquely_assigned_candidate_count": (
        "distinct candidates strictly assigned to this owner"
    ),
    "non_loop_tested_context_count": "number of non-loop-tail contexts in which the owner was tested",
    "cross_context_delta_primary_best_vs_minimal_frontier": (
        "primary maximum minus the minimal-frontier context's maximum, taken ACROSS two "
        "contexts; a stability delta, explicitly NOT a competition margin"
    ),
}

#: Names carried through verbatim from an owner-context projection.
OWNER_CONTEXT_FEATURE_NAMES: tuple[str, ...] = (
    "margin_to_best_owner_in_group",
    "owner_rank_within_group",
    "exact_anchor_score",
    "strict_assigned_max",
    "signed_frontier_ordinal_distance",
    "abs_signed_frontier_ordinal_distance",
    "signed_frontier_sort_axis_pixel_distance",
    "frontier_intersection_over_union",
    "same_description_owners_ahead_of_frontier",
    "boundary_gate_continue_vs_stop_logprob_margin",
    "category_routing_within_context_rank",
)

#: The published owner views a rule may evaluate against.
OWNER_FEATURE_VIEWS: tuple[str, ...] = (
    "primary_best_non_loop",
    "primary_first_non_loop_minimal_abs_frontier",
    "diagnostic_best_all",
)

#: Evidence states a rule may *stratify* on (for example to hold the native
#: true-positive calibration stratum separate).  These are never cut points.
EVIDENCE_STATE_NAMES: frozenset[str] = frozenset(
    {
        "native_true_positive",
        "greedy_eligible",
        "frontier_tested",
        "loop_tail_only_support",
        "bank_adequacy_status",
        "disposition",
    }
)

RULE_DIRECTIONS: tuple[str, ...] = ("at_least", "at_most")

#: Local-peak support criterion.  Attaining rank one in a same-category owner
#: competition was rejected as the primary support test: it can measure routing
#: or competition rather than the existence of a likelihood peak at the owner's
#: own geometry.  Rank and margin remain published as continuous
#: routing/competition features, but support is decided by two local-peak
#: quantities that never consult an owner ranking:
#:
#: ``peak_lift``
#:     ``log p(best owner candidate) - log(1/N)`` over the unique
#:     ``(context, category)`` population.
#: ``local_concentration``
#:     the owner's best exclusion-filtered score minus the median of that
#:     owner's own exclusion-filtered bank scores.
SUPPORT_CRITERION_ID = "local_peak_lift_and_local_concentration_under_both_ambiguity_bounds"
SUPPORT_FEATURE_NAMES: tuple[str, ...] = ("peak_lift", "local_concentration")

#: Every threshold, quantile and epsilon this module applies is *bound* from
#: ``capture-rules.json -> owner_support`` by :func:`load_support_contract`.  No
#: numeric decision value is declared here: a merge that carried its own copy
#: could silently substitute one if the sealed rules ever moved, and the whole
#: point of sealing them before capture is that they cannot.
#:
#: Only vocabulary lives here -- the name of the "adequately represented"
#: category flag, whose counterpart flag string is itself bound from the rules.
CATEGORY_SUPPORT_POOLED = "pooled"

#: Scorer-declared capture completeness.  Only ``complete_shard`` output may
#: enter a conclusion-bearing phase.
COMPLETE_SHARD = "complete_shard"
SUBSET_SMOKE = "subset_smoke"


@dataclass(frozen=True)
class SupportContract:
    """Support constants, bound from the sealed capture rules.

    The merge never chooses these.  ``owner_support`` in ``capture-rules.json``
    froze the statistic names, the primary and sensitivity quantiles, both
    epsilon bands and the category-contribution floor before any score existed;
    this loader binds them and fails closed if they are absent.
    """

    statistics: tuple[str, ...]
    primary_quantile: float
    sensitivity_quantiles: tuple[float, ...]
    support_epsilon: float
    cross_context_delta_epsilon: float
    category_contribution_min: int
    underrepresented_flag: str


def load_support_contract(plan: PlanBundle) -> SupportContract:
    support = plan.capture_rules.get("owner_support") or {}
    definition = support.get("support_definition")
    calibration = support.get("support_calibration")
    epsilons = support.get("epsilons")
    for name, block in (
        ("support_definition", definition),
        ("support_calibration", calibration),
        ("epsilons", epsilons),
    ):
        if not isinstance(block, Mapping):
            _fail(
                f"capture-rules.json does not seal owner_support.{name}; the merge will "
                "not substitute support constants of its own"
            )
    if definition.get("rank_is_support_criterion") is not False:
        _fail("sealed support definition does not forbid a rank criterion")
    statistics = tuple(str(value) for value in definition["statistics"])
    if set(statistics) != set(SUPPORT_FEATURE_NAMES):
        _fail(
            f"sealed support statistics {list(statistics)} do not match the statistics this "
            f"merge computes {list(SUPPORT_FEATURE_NAMES)}"
        )
    return SupportContract(
        statistics=statistics,
        primary_quantile=float(calibration["primary_quantile"]),
        sensitivity_quantiles=tuple(
            float(value) for value in calibration["sensitivity_quantiles"]
        ),
        support_epsilon=float(epsilons["support_epsilon"]),
        cross_context_delta_epsilon=float(epsilons["cross_context_delta_epsilon"]),
        category_contribution_min=int(calibration["category_contribution_min"]),
        underrepresented_flag=str(calibration["underrepresented_flag"]),
    )


@dataclass(frozen=True)
class SupportCalibration:
    """TP-calibrated support thresholds, derived on the discovery half only.

    Each threshold is the quantile of the *minimum across the two ambiguity
    bounds* of that statistic, because the support test requires both bounds to
    clear.  Note this fixes each statistic's marginal quantile only: the support
    test is a conjunction of two such cuts, so the joint failure rate among the
    calibration population is at least the nominal quantile and generally above
    it.  The observed joint rate is reported rather than assumed.
    """

    theta_peak_lift: float
    theta_local_concentration: float
    epsilon: float
    quantile: float
    observation_count: int
    per_category_counts: Mapping[str, int]
    sensitivity: Mapping[str, Any]
    exclusions: tuple[Mapping[str, Any], ...]
    consumed_shard_digests: tuple[str, ...]
    capture_manifest_sha256: str
    category_contribution_min: int
    underrepresented_flag: str
    cross_context_delta_epsilon: float
    statistics: tuple[str, ...]

    def clears(self, features: Mapping[str, Any] | None) -> bool | None:
        if features is None:
            return None
        lift = features.get("peak_lift")
        concentration = features.get("local_concentration")
        if lift is None or concentration is None:
            return None
        return bool(
            float(lift) >= self.theta_peak_lift + self.epsilon
            and float(concentration) >= self.theta_local_concentration + self.epsilon
        )

    def category_support(self, normalized_description: str) -> str:
        """Flag only.  An underrepresented category never changes a threshold."""

        count = int(self.per_category_counts.get(str(normalized_description), 0))
        return (
            CATEGORY_SUPPORT_POOLED
            if count >= self.category_contribution_min
            else self.underrepresented_flag
        )

    def describe(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "criterion_id": SUPPORT_CRITERION_ID,
            "statistics": list(self.statistics),
            "rule": (
                "peak_lift >= theta_peak_lift + epsilon AND "
                "local_concentration >= theta_local_concentration + epsilon, "
                "required under both ambiguity bounds"
            ),
            "rank_is_not_a_support_input": True,
            "theta_peak_lift": self.theta_peak_lift,
            "theta_local_concentration": self.theta_local_concentration,
            "epsilon": self.epsilon,
            "epsilon_source": "sealed_capture_rules_owner_support_epsilons",
            "epsilon_is_adaptive": False,
            "observed_parity_role": "compliance_check_against_bound_never_resizes_it",
            "cross_context_delta_epsilon": self.cross_context_delta_epsilon,
            "quantile": self.quantile,
            "quantile_is_primary_and_fixed": True,
            "calibration_stratum": "pooled_discovery_native_true_positives",
            "observation_count": self.observation_count,
            "per_category_tp_counts": dict(sorted(self.per_category_counts.items())),
            "category_stratification_role": "report_only_sensitivity_never_a_threshold",
            "category_contribution_min": self.category_contribution_min,
            "underrepresented_flag": self.underrepresented_flag,
            "mid_run_category_refinement": "forbidden",
            "sensitivity": dict(self.sensitivity),
            "calibration_exclusions": [dict(row) for row in self.exclusions],
            "due_context_rule": (
                "boundary index of the owner's unique native strict-match row, taken from "
                "the native sidecar registry; never inferred from any score or frontier"
            ),
            "loop_tail_contexts_excluded": True,
            "consumed_shard_digests": list(self.consumed_shard_digests),
            "capture_manifest_sha256": self.capture_manifest_sha256,
            # The analyzer that derived these thresholds.  Provenance only --
            # it moves no threshold, formula or quantile -- but confirmation
            # binds it, so thresholds produced by foreign analysis code cannot
            # be applied to the held-out half under a different implementation.
            "merge_source_sha256": MERGE_SOURCE_SHA256,
            "merge_source_role": "derived_these_thresholds_provenance_only",
            "discovery_image_ids": list(planner.DISCOVERY_IMAGE_IDS),
            "confirmation_evidence_consumed": False,
        }
        payload["calibration_sha256"] = sha256_json(payload)
        return payload


def calibration_from_receipt(receipt: Mapping[str, Any]) -> SupportCalibration:
    """Rebuild a calibration from its sealed receipt, proving the digest first."""

    if receipt.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        _fail("calibration receipt has an unexpected schema_version")
    if receipt.get("unit_id") != UNIT_ID:
        _fail("calibration receipt belongs to another unit")
    reconstructed = sha256_json(
        {key: value for key, value in receipt.items() if key != "calibration_sha256"}
    )
    if reconstructed != receipt.get("calibration_sha256"):
        _fail("calibration receipt does not reconstruct its own digest; it was edited")
    sealed_source = receipt.get("merge_source_sha256")
    if not sealed_source:
        _fail(
            "calibration receipt does not record the analyzer source that derived it; "
            "refusing to apply thresholds of unknown provenance"
        )
    if str(sealed_source) != MERGE_SOURCE_SHA256:
        _fail(
            "calibration receipt was derived by a different analyzer source "
            f"({sealed_source}); this merge is {MERGE_SOURCE_SHA256}. Re-derive the "
            "calibration with the current analyzer rather than applying foreign "
            "thresholds to the held-out half"
        )
    return SupportCalibration(
        theta_peak_lift=float(receipt["theta_peak_lift"]),
        theta_local_concentration=float(receipt["theta_local_concentration"]),
        epsilon=float(receipt["epsilon"]),
        quantile=float(receipt["quantile"]),
        observation_count=int(receipt["observation_count"]),
        per_category_counts=dict(receipt["per_category_tp_counts"]),
        sensitivity=dict(receipt["sensitivity"]),
        exclusions=tuple(receipt["calibration_exclusions"]),
        consumed_shard_digests=tuple(receipt["consumed_shard_digests"]),
        capture_manifest_sha256=str(receipt["capture_manifest_sha256"]),
        category_contribution_min=int(receipt["category_contribution_min"]),
        underrepresented_flag=str(receipt["underrepresented_flag"]),
        cross_context_delta_epsilon=float(receipt["cross_context_delta_epsilon"]),
        statistics=tuple(str(value) for value in receipt["statistics"]),
    )


#: Evidence-state dispositions.  These are states of the *evidence*, not
#: mechanisms; nothing here asserts why a model did anything.
DISPOSITION_RESOLVED = "resolved_tested_localization_support"
DISPOSITION_PERSISTENT_NEGATIVE = "persistent_no_tested_localization_support"
DISPOSITION_UNRESOLVED = "unresolved_insufficient_tested_localization_support"
DISPOSITION_UNRESOLVED_FLIP = "unresolved_ambiguity_bound_disposition_flip"
DISPOSITION_UNCALIBRATED = "unresolved_support_thresholds_not_calibrated"
DISPOSITION_NOT_ELIGIBLE = "unresolved_owner_outside_the_native_matching_universe"
#: Native true positives are the calibration/positive-control population.  The
#: frozen q10 threshold guarantees a share of them fail the support test by
#: construction, so labelling them with the false-negative disposition would
#: count the control population as false negatives.
DISPOSITION_TP_CONTROL = "native_true_positive_calibration_control"


def _routing_summary(contexts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Owner routing/competition surface, kept apart from support.

    Rank is informative about which owner the model routes a category to; it is
    not evidence that a likelihood peak exists at any owner's geometry.  It is
    therefore summarized here, separately, and never consulted by the support
    test.
    """

    ranked = [
        row
        for row in contexts
        if row["owner_competition_u"]["rank"] is not None
        and not row["loop_marking"]["loop_tail"]
    ]
    if not ranked:
        return {
            "population": "non_loop_contexts",
            "ever_rank1": False,
            "best_rank": None,
            "best_margin_to_best_owner": None,
            "best_rank_context_id": None,
            "role": "routing_and_competition_surface_never_a_support_input",
        }
    best = min(
        ranked,
        key=lambda row: (
            int(row["owner_competition_u"]["rank"]),
            -float(row["owner_competition_u"]["margin_to_best_owner"]),
        ),
    )
    return {
        "population": "non_loop_contexts",
        "ever_rank1": any(int(row["owner_competition_u"]["rank"]) == 1 for row in ranked),
        "best_rank": int(best["owner_competition_u"]["rank"]),
        "best_margin_to_best_owner": max(
            float(row["owner_competition_u"]["margin_to_best_owner"]) for row in ranked
        ),
        "best_rank_context_id": str(best["context_id"]),
        "role": "routing_and_competition_surface_never_a_support_input",
    }


#: Blind-analysis phases.  The split is mechanical: a phase may only open the
#: shard files its allowlist names, so the discovery analysis that derives the
#: calibration cannot materialize a confirmation feature at all.
PHASE_CAPTURE_MANIFEST = "capture-manifest"
PHASE_SMOKE_ADMIT = "smoke-admit"
PHASE_DISCOVERY = "discovery"
PHASE_CONFIRMATION = "confirmation"
PHASE_PRESENTATION = "presentation"
PHASE_FULL = "full"
MERGE_PHASES: tuple[str, ...] = (
    PHASE_SMOKE_ADMIT,
    PHASE_DISCOVERY,
    PHASE_CONFIRMATION,
    PHASE_PRESENTATION,
    PHASE_FULL,
)

#: The representative real-HF launch-gate image.  A discovery image by
#: construction, so the gate never spends held-out evidence.
SMOKE_IMAGE_ID = planner.REPRESENTATIVE_SMOKE_IMAGE_ID

CAPTURE_MANIFEST_SCHEMA_VERSION = "sorted-owner-accessibility-census-capture-manifest.v1"

#: Content address of this module.  The sealed artifacts already bind the
#: planner and scorer sources, but the merge is the code that interprets scores,
#: applies the calibration and closes dispositions, so a run is not reproducible
#: from its receipts unless the analyzer itself is content-addressed too.
#:
#: Provenance only: it is never read by the support test, a rank, a posterior or
#: a candidate population, and changing it changes no research outcome.
MERGE_SOURCE_SHA256 = planner.sha256_file(Path(__file__).resolve())
CALIBRATION_SCHEMA_VERSION = "sorted-owner-accessibility-census-support-calibration.v1"


class MergeContractError(RuntimeError):
    """A precondition for a conclusion-bearing census merge was not proven."""


class GlobalStopError(MergeContractError):
    """More images are unusable than the frozen stop policy tolerates."""


def _fail(message: str) -> NoReturn:
    raise MergeContractError(message)


# ---------------------------------------------------------------------------
# Digest and IO helpers (deliberate local copies; see the planner's rationale)
# ---------------------------------------------------------------------------


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
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


def _require(row: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in row or row[key] is None:
        _fail(f"{label} is missing required field {key!r}")
    return row[key]


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        _fail(f"{label} is not a number")
    if not math.isfinite(number):
        _fail(f"{label} is not finite")
    return number


def _quantile(values: Sequence[float], level: float) -> float:
    """Deterministic linear-interpolated quantile; no numpy dependency."""

    if not values:
        _fail("cannot take a quantile of an empty population")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = float(level) * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[int(position)]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _log_softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    peak = max(values)
    total = math.log(math.fsum(math.exp(value - peak) for value in values))
    return [value - peak - total for value in values]


# ---------------------------------------------------------------------------
# Plan loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanBundle:
    """The sealed plan, re-verified file by file before any row is read."""

    plan_dir: Path
    receipt: dict[str, Any]
    capture_rules: dict[str, Any]
    images: dict[str, dict[str, Any]]
    owners: dict[str, dict[str, Any]]
    categories: dict[str, dict[str, Any]]
    contexts: dict[str, dict[str, Any]]
    candidates: dict[str, dict[str, Any]]
    query_groups: dict[str, dict[str, Any]]
    native_sidecars: list[dict[str, Any]]
    shards: dict[str, dict[str, Any]]

    def owners_in_image(self, image_id: str) -> list[dict[str, Any]]:
        return sorted(
            (row for row in self.owners.values() if str(row["image_id"]) == image_id),
            key=lambda row: (
                list(row["owner_sort_key"]),
                str(row["gt_owner_id"]),
            ),
        )

    def contexts_in_image(self, image_id: str) -> list[dict[str, Any]]:
        return sorted(
            (row for row in self.contexts.values() if str(row["image_id"]) == image_id),
            key=lambda row: int(row["boundary_index"]),
        )


def load_plan(plan_dir: Path) -> PlanBundle:
    """Re-verify the sealed plan: receipt digest, every declared file, capture rules.

    The schema-version gate is what makes a pre-P0 plan mechanically
    unjoinable: this merge accepts one plan schema and refuses every other.
    """

    plan_dir = Path(plan_dir)
    receipt = _read_json(plan_dir / "receipt.json", "plan receipt")
    if receipt.get("schema_version") != planner.PLAN_SCHEMA_VERSION:
        _fail(
            "plan receipt schema_version does not match this unit's plan schema; "
            "refusing to merge against a foreign or pre-P0 plan"
        )
    if receipt.get("unit_id") != UNIT_ID:
        _fail("plan receipt unit_id does not match this unit")
    reconstructed = sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    if reconstructed != receipt.get("receipt_content_sha256"):
        _fail("plan receipt does not reconstruct its own digest")

    declared = receipt.get("output_file_digests") or {}
    for name in planner.PLAN_FILE_NAMES:
        path = plan_dir / name
        if not path.is_file():
            _fail(f"plan is missing declared file {name!r}")
        if sha256_bytes(path.read_bytes()) != declared.get(name):
            _fail(f"plan file {name!r} does not match its sealed digest")

    capture_rules = _read_json(plan_dir / planner.CAPTURE_RULES_NAME, "capture rules")
    if capture_rules.get("schema_version") != planner.CAPTURE_RULES_SCHEMA_VERSION:
        _fail("capture-rules.json schema_version does not match this unit")
    sealed = capture_rules.get("capture_rules_sha256")
    if sealed != sha256_json(
        {key: value for key, value in capture_rules.items() if key != "capture_rules_sha256"}
    ):
        _fail("capture-rules.json does not reconstruct its own digest")
    if sealed != receipt.get("capture_rules_sha256"):
        _fail("capture-rules.json digest does not match the plan receipt")

    def index(name: str, key: str) -> dict[str, dict[str, Any]]:
        rows = _read_jsonl(plan_dir / name, name)
        indexed: dict[str, dict[str, Any]] = {}
        for row in rows:
            row_key = str(_require(row, key, name))
            if row_key in indexed:
                _fail(f"{name} carries duplicate {key} {row_key!r}")
            indexed[row_key] = row
        return indexed

    bundle = PlanBundle(
        plan_dir=plan_dir,
        receipt=receipt,
        capture_rules=capture_rules,
        images=index("image-registry.jsonl", "image_id"),
        owners=index("owner-registry.jsonl", "gt_owner_id"),
        categories=index("category-registry.jsonl", "category_query_id"),
        contexts=index("context-registry.jsonl", "context_id"),
        candidates=index("candidate-bank.jsonl", "candidate_id"),
        query_groups=index("query-group-registry.jsonl", "query_group_id"),
        native_sidecars=_read_jsonl(
            plan_dir / "native-sidecar-registry.jsonl", "native-sidecar-registry.jsonl"
        ),
        shards=index("shard-manifest.jsonl", "image_id"),
    )
    if len(bundle.images) != EXPECTED_IMAGE_COUNT:
        _fail(
            f"plan declares {len(bundle.images)} images, expected {EXPECTED_IMAGE_COUNT}; "
            "this merge is the generalized twelve-image merge"
        )
    if set(bundle.images) != set(planner.SPLIT_BY_IMAGE_ID):
        _fail("plan image set does not match the frozen discovery/confirmation split")
    return bundle


# ---------------------------------------------------------------------------
# Shard discovery, quarantine, and the global stop policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardArtifacts:
    """One image shard as found on disk.  Quarantined shards carry no evidence."""

    image_id: str
    split: str
    status: str  # captured | quarantined | missing
    directory: Path | None
    receipt: dict[str, Any] | None = None
    quarantine: dict[str, Any] | None = None
    scores: list[dict[str, Any]] = field(default_factory=list)
    proposals: list[dict[str, Any]] = field(default_factory=list)
    free_decodes: list[dict[str, Any]] = field(default_factory=list)
    file_digests: dict[str, str] = field(default_factory=dict)
    detail: str = ""
    evidence_loaded: bool = True


def _shard_directory(shard_root: Path, image_id: str) -> Path | None:
    for name in (image_id, f"shard-{image_id}"):
        candidate = shard_root / name
        if candidate.is_dir():
            return candidate
    return None


def discover_shards(
    plan: PlanBundle,
    shard_root: Path,
    *,
    allowlist: Sequence[str] | None = None,
    load_evidence: bool = True,
) -> list[ShardArtifacts]:
    """Locate the allowlisted image shards and classify them.

    ``load_evidence=False`` is the metadata-only mode the capture-manifest phase
    uses: shard files are hashed as bytes and the receipt and quarantine
    documents are parsed for lineage, but no score, proposal or free-decode row
    is ever parsed.  Hashing bytes is not interpretation; parsing a score row
    is, and doing it before the calibration is sealed would materialize
    confirmation evidence ahead of the blind boundary.

    ``allowlist`` is a *mechanical* inaccessibility boundary, not a filter: an
    image outside it is never opened at all, so a discovery-phase analysis
    cannot materialize a single confirmation feature even accidentally.
    Filtering rows after reading them would not satisfy that, because the values
    would already exist in the process.

    A quarantined shard is likewise never opened for scores: the frozen stop
    policy forbids evidence from a failed shard, so this does not read it and
    then discard it, it does not read it at all.  A shard that is entirely
    absent is unusable for the same reason and is accounted under its own
    status.
    """

    shard_root = Path(shard_root)
    permitted = None if allowlist is None else {str(value) for value in allowlist}
    if permitted is not None:
        unknown = sorted(permitted - set(plan.images), key=str)
        if unknown:
            _fail(f"shard allowlist names images the plan does not declare: {unknown}")

    found: list[ShardArtifacts] = []
    for image_id in sorted(plan.images, key=int):
        if permitted is not None and image_id not in permitted:
            continue
        split = str(planner.SPLIT_BY_IMAGE_ID[image_id])
        directory = _shard_directory(shard_root, image_id)
        if directory is None:
            found.append(
                ShardArtifacts(
                    image_id=image_id,
                    split=split,
                    status="missing",
                    directory=None,
                    detail=f"no shard directory for image {image_id} under {shard_root}",
                )
            )
            continue

        quarantine_path = directory / SHARD_QUARANTINE_NAME
        if quarantine_path.is_file():
            payload = _read_json(quarantine_path, f"shard {image_id} quarantine")
            if payload.get("schema_version") != EXPECTED_QUARANTINE_SCHEMA:
                _fail(f"shard {image_id} quarantine has an unexpected schema_version")
            if str(payload.get("image_id")) != image_id:
                _fail(f"shard {image_id} quarantine names a different image")
            found.append(
                ShardArtifacts(
                    image_id=image_id,
                    split=split,
                    status="quarantined",
                    directory=directory,
                    quarantine=payload,
                    detail=str(payload.get("reason", "")),
                )
            )
            continue

        receipt_path = directory / SHARD_RECEIPT_NAME
        if not receipt_path.is_file():
            found.append(
                ShardArtifacts(
                    image_id=image_id,
                    split=split,
                    status="missing",
                    directory=directory,
                    detail=f"shard directory has neither {SHARD_RECEIPT_NAME} nor "
                    f"{SHARD_QUARANTINE_NAME}",
                )
            )
            continue

        receipt = _read_json(receipt_path, f"shard {image_id} receipt")
        digests = {SHARD_RECEIPT_NAME: sha256_bytes(receipt_path.read_bytes())}
        # A shard that published a receipt but not its whole evidence set is
        # incomplete, not captured.  Admitting it would let a partial capture
        # slip past the frozen quarantine/missing ledger before the split
        # analysis ever runs.
        missing_files = [name for name in SHARD_EVIDENCE_NAMES if not (directory / name).is_file()]
        if missing_files:
            found.append(
                ShardArtifacts(
                    image_id=image_id,
                    split=split,
                    status="incomplete",
                    directory=directory,
                    file_digests=digests,
                    detail=f"shard published a receipt but not {missing_files}",
                    evidence_loaded=False,
                )
            )
            continue
        for name in SHARD_EVIDENCE_NAMES:
            digests[name] = sha256_bytes((directory / name).read_bytes())

        scores: list[dict[str, Any]] = []
        proposals: list[dict[str, Any]] = []
        free_decodes: list[dict[str, Any]] = []
        if load_evidence:
            scores = _read_jsonl(directory / SHARD_SCORES_NAME, f"shard {image_id} scores")
            proposals = _read_jsonl(
                directory / SHARD_PROPOSAL_NAME, f"shard {image_id} proposals"
            )
            free_decodes = _read_jsonl(
                directory / SHARD_FREE_DECODE_NAME, f"shard {image_id} free decodes"
            )
        found.append(
            ShardArtifacts(
                image_id=image_id,
                split=split,
                status="captured",
                directory=directory,
                receipt=receipt,
                scores=scores,
                proposals=proposals,
                free_decodes=free_decodes,
                file_digests=digests,
                evidence_loaded=load_evidence,
            )
        )
    return found


def enforce_stop_policy(
    plan: PlanBundle, shards: Sequence[ShardArtifacts]
) -> dict[str, Any]:
    """Apply the frozen global stop policy over unusable images.

    ``global_stop_quarantined_image_count_above`` is read from the sealed
    capture rules rather than hardcoded, so the stop threshold this merge
    enforces is provably the one the capture was launched under.
    """

    stop_policy = plan.capture_rules.get("stop_policy") or {}
    if "global_stop_quarantined_image_count_above" not in stop_policy:
        _fail("capture rules do not declare a global stop threshold")
    threshold = int(stop_policy["global_stop_quarantined_image_count_above"])

    quarantined = sorted(row.image_id for row in shards if row.status == "quarantined")
    missing = sorted(row.image_id for row in shards if row.status == "missing")
    incomplete = sorted(row.image_id for row in shards if row.status == "incomplete")
    unusable = sorted(set(quarantined) | set(missing) | set(incomplete), key=int)
    ledger = {
        "global_stop_quarantined_image_count_above": threshold,
        "quarantined_image_ids": quarantined,
        "quarantined_image_count": len(quarantined),
        "missing_image_ids": missing,
        "missing_image_count": len(missing),
        "incomplete_image_ids": incomplete,
        "incomplete_image_count": len(incomplete),
        "unusable_image_ids": unusable,
        "unusable_image_count": len(unusable),
        "captured_image_ids": sorted(
            (row.image_id for row in shards if row.status == "captured"), key=int
        ),
        "evidence_from_failed_shards": "forbidden_never_read",
        "global_stop_triggered": len(unusable) > threshold,
    }
    if ledger["global_stop_triggered"]:
        raise GlobalStopError(
            f"{len(unusable)} images are unusable ({unusable}), above the frozen global stop "
            f"threshold of {threshold}; the census may not be merged into a conclusion"
        )
    return ledger


def build_capture_manifest(plan: PlanBundle, shard_root: Path) -> dict[str, Any]:
    """Phase A: content-address all twelve shards and freeze the quarantine ledger.

    This runs *before* any analysis and deliberately performs no score-derived
    interpretation: it hashes bytes and validates receipt lineage, nothing more.
    That is what makes the later blind split mechanical rather than procedural --
    the digests that partition discovery from confirmation are fixed here, so a
    discovery calibration can be proven to have consumed none of them.

    The frozen global stop policy is enforced here too, over all twelve images,
    so a later phase reading only six of them can never appear to satisfy a
    coverage requirement that the run as a whole failed.
    """

    shards = discover_shards(plan, shard_root, load_evidence=False)
    quarantine_ledger = enforce_stop_policy(plan, shards)

    per_image: list[dict[str, Any]] = []
    digests_by_split: dict[str, list[str]] = {"discovery": [], "confirmation": []}
    lineages: list[dict[str, Any]] = []
    for shard in shards:
        lineage: dict[str, Any] | None = None
        if shard.status == "captured":
            # Receipt lineage only: plan binding, capture rules, runtime
            # identity.  No score row is interpreted at this phase.
            lineage = validate_shard_lineage(plan, shard)
            lineages.append(lineage)
        entry = {
            "image_id": shard.image_id,
            "split": shard.split,
            "status": shard.status,
            "detail": shard.detail,
            "file_digests": dict(sorted(shard.file_digests.items())),
            "quarantine_disposition": (
                dict(shard.quarantine) if shard.quarantine is not None else None
            ),
            "lineage": lineage,
        }
        per_image.append(entry)
        digests_by_split[shard.split].extend(sorted(shard.file_digests.values()))

    overlap = sorted(set(digests_by_split["discovery"]) & set(digests_by_split["confirmation"]))
    if overlap:
        _fail(
            "a shard file digest appears in both the discovery and confirmation split; "
            "the blind boundary cannot be established"
        )

    # Uniformity is asserted here, across *both* splits at once.  Each later
    # phase only ever sees its own half, so a cross-split model, tokenizer or
    # code drift would otherwise survive undetected while confirmation still
    # bound this manifest's digest.
    runtime = assert_uniform_runtime(lineages)

    manifest: dict[str, Any] = {
        "schema_version": CAPTURE_MANIFEST_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "phase": PHASE_CAPTURE_MANIFEST,
        "shard_root": str(shard_root),
        "plan_receipt_content_sha256": plan.receipt["receipt_content_sha256"],
        "capture_rules_sha256": plan.capture_rules["capture_rules_sha256"],
        "images": per_image,
        "quarantine_ledger": quarantine_ledger,
        "quarantine_dispositions_frozen": True,
        "runtime_identity": runtime,
        "runtime_identity_sha256": sha256_json(runtime),
        "runtime_uniform_across_both_splits": True,
        "discovery_shard_digests": sorted(set(digests_by_split["discovery"])),
        "confirmation_shard_digests": sorted(set(digests_by_split["confirmation"])),
        "score_derived_interpretation_performed": False,
        "reads_score_rows": False,
        "code": {
            "merge_source_sha256": MERGE_SOURCE_SHA256,
            "role": "sealed_the_metadata_and_hash_policy_provenance_only",
        },
    }
    manifest["capture_manifest_sha256"] = sha256_json(manifest)
    return manifest


def _allowlist_for_phase(phase: str) -> tuple[str, ...] | None:
    if phase == PHASE_DISCOVERY:
        return tuple(planner.DISCOVERY_IMAGE_IDS)
    if phase == PHASE_CONFIRMATION:
        return tuple(planner.CONFIRMATION_IMAGE_IDS)
    if phase == PHASE_SMOKE_ADMIT:
        return (SMOKE_IMAGE_ID,)
    return None


def assert_manifest_binding(
    plan: PlanBundle,
    manifest: Mapping[str, Any],
    shards: Sequence[ShardArtifacts],
    *,
    phase: str,
) -> dict[str, Any]:
    """Bind a phase to the exact bytes the capture manifest sealed.

    Split-disjointness alone is not binding: it proves a phase did not consume
    a digest *listed for the other split*, which says nothing about whether the
    bytes this phase is reading are still the bytes that were sealed.  A shard
    rewritten after manifest time would sail through disjointness and be
    analyzed under a stale digest, and the confirmation receipt would still
    bind a manifest digest that no longer describes the evidence.

    So every current shard's file digests must equal the sealed per-image map
    exactly -- no missing file, no extra file, no changed bytes -- and its
    split and status must match the frozen disposition.
    """

    if manifest.get("schema_version") != CAPTURE_MANIFEST_SCHEMA_VERSION:
        _fail("capture manifest has an unexpected schema_version")
    if manifest.get("unit_id") != UNIT_ID:
        _fail("capture manifest belongs to another unit")
    sealed_digest = manifest.get("capture_manifest_sha256")
    if sealed_digest != sha256_json(
        {key: value for key, value in manifest.items() if key != "capture_manifest_sha256"}
    ):
        _fail("capture manifest does not reconstruct its own digest; it was edited")
    if manifest.get("plan_receipt_content_sha256") != plan.receipt["receipt_content_sha256"]:
        _fail("capture manifest was sealed against a different plan receipt")
    if manifest.get("capture_rules_sha256") != plan.capture_rules["capture_rules_sha256"]:
        _fail("capture manifest was sealed against different capture rules")

    sealed_by_image: dict[str, Mapping[str, Any]] = {}
    for entry in manifest.get("images") or ():
        image_id = str(entry["image_id"])
        if image_id in sealed_by_image:
            _fail(f"capture manifest lists image {image_id!r} twice")
        sealed_by_image[image_id] = entry
    if set(sealed_by_image) != set(plan.images):
        _fail(
            "capture manifest image set does not match the plan; the manifest must cover "
            "every planned shard"
        )

    bound: list[str] = []
    for shard in shards:
        sealed = sealed_by_image.get(shard.image_id)
        if sealed is None:
            _fail(
                f"shard {shard.image_id} is being consumed but is absent from the capture "
                "manifest"
            )
        if str(sealed["split"]) != shard.split:
            _fail(f"shard {shard.image_id} split differs from the sealed manifest")
        if str(sealed["status"]) != shard.status:
            _fail(
                f"shard {shard.image_id} status drifted from the frozen manifest "
                f"disposition ({sealed['status']!r} -> {shard.status!r})"
            )
        sealed_digests = {str(k): str(v) for k, v in (sealed.get("file_digests") or {}).items()}
        current = {str(k): str(v) for k, v in shard.file_digests.items()}
        if current != sealed_digests:
            missing = sorted(set(sealed_digests) - set(current))
            extra = sorted(set(current) - set(sealed_digests))
            changed = sorted(
                name
                for name in set(current) & set(sealed_digests)
                if current[name] != sealed_digests[name]
            )
            _fail(
                f"shard {shard.image_id} bytes changed after the capture manifest was "
                f"sealed (missing={missing}, extra={extra}, changed={changed}); the "
                "analysis would run under a stale manifest digest"
            )
        bound.append(shard.image_id)

    return {
        "capture_manifest_sha256": str(sealed_digest),
        "bound_image_ids": sorted(bound, key=int),
        "byte_identity_verified": True,
        "status_and_split_verified": True,
        "phase": phase,
    }


def assert_digest_disjointness(
    manifest: Mapping[str, Any], shards: Sequence[ShardArtifacts], *, phase: str
) -> None:
    """Prove a phase consumed no digest belonging to the other split.

    The allowlist already prevents the files from being opened; this is the
    receipted half of the same guarantee, so the claim survives in the artifact
    rather than only in the control flow.
    """

    consumed = {digest for shard in shards for digest in shard.file_digests.values()}
    if phase == PHASE_DISCOVERY:
        forbidden = set(manifest["confirmation_shard_digests"])
        other = "confirmation"
    elif phase in {PHASE_CONFIRMATION}:
        forbidden = set(manifest["discovery_shard_digests"])
        other = "discovery"
    else:
        return
    leaked = sorted(consumed & forbidden)
    if leaked:
        _fail(
            f"the {phase} phase consumed {len(leaked)} shard file digest(s) belonging to "
            f"the {other} split; the blind-analysis boundary is broken"
        )


# ---------------------------------------------------------------------------
# Shard admission: lineage, runtime identity, and per-channel admission receipts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdmissionIndex:
    """Every admission receipt one shard proved, keyed by receipt ID."""

    image_id: str
    by_receipt_id: dict[str, dict[str, Any]]

    def covering(self, receipt_id: str) -> dict[str, Any] | None:
        return self.by_receipt_id.get(receipt_id)


def reconstruct_admission_receipt_id(
    plan: PlanBundle, entry: Mapping[str, Any], *, label: str
) -> str:
    """Rebuild a receipt's ID from its own declared identity, per channel.

    There is no generic ``prefix_sha256``: each channel is keyed on a different
    quantity, and collapsing them onto one field would let a receipt proved on
    one execution shape be relabelled onto another.

    ``query_suffix``
        the exact ``query_prefix_sha256``.
    ``proposal_boundary_gate``
        the exact ``observed_prefix_sha256`` (the gate forces nothing, so its
        prefix *is* the observed prefix).
    ``proposal_category_route``
        the observed prefix combined with the exact routing-path digest, which
        is additionally re-derived from the named category's tokens so a receipt
        cannot assert a routing path it did not execute.
    """

    channel = str(_require(entry, "channel", label))
    if channel not in planner.ADMISSION_CHANNELS:
        _fail(f"{label} names unknown admission channel {channel!r}")
    context_id = str(_require(entry, "context_id", label))

    if channel == planner.CHANNEL_QUERY_SUFFIX:
        return planner.admission_receipt_id(
            context_id=context_id,
            channel=channel,
            prefix_sha256=str(_require(entry, "query_prefix_sha256", label)),
        )

    observed_sha = str(_require(entry, "observed_prefix_sha256", label))
    if channel == planner.CHANNEL_PROPOSAL_BOUNDARY_GATE:
        return planner.admission_receipt_id(
            context_id=context_id, channel=channel, prefix_sha256=observed_sha
        )

    category_query_id = str(_require(entry, "category_query_id", label))
    category = plan.categories.get(category_query_id)
    if category is None:
        _fail(f"{label} names unknown category {category_query_id!r}")
    category_tokens = [int(value) for value in category["category_token_ids"]]
    declared_digest = str(_require(entry, "routing_path_digest", label))
    if declared_digest != planner.proposal_route_digest(category_tokens):
        _fail(
            f"{label} declares a routing-path digest that does not match the planned "
            f"routing path of category {category_query_id!r}"
        )
    return planner.proposal_route_admission_receipt_id(
        context_id=context_id,
        observed_prefix_sha256=observed_sha,
        category_token_ids=category_tokens,
    )


def build_admission_index(plan: PlanBundle, shard: ShardArtifacts) -> AdmissionIndex:
    """Index one shard's admission receipts and prove each one's own identity.

    Reads the scorer's authoritative location, ``shard-receipt.json ->
    admission.receipts``, and re-derives every receipt ID channel-specifically.
    A receipt that merely asserts an ID could otherwise let one context's,
    category's, or channel's admission be relabelled to cover another.
    """

    receipt = shard.receipt or {}
    admission = receipt.get("admission")
    if not isinstance(admission, Mapping):
        _fail(
            f"shard {shard.image_id} receipt has no admission block; this merge reads the "
            "scorer's authoritative admission.receipts and nothing else"
        )
    entries = admission.get("receipts")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        _fail(f"shard {shard.image_id} admission.receipts is not a list")

    indexed: dict[str, dict[str, Any]] = {}
    for position, entry in enumerate(entries):
        label = f"shard {shard.image_id} admission receipt {position}"
        if not isinstance(entry, Mapping):
            _fail(f"{label} is not a JSON object")
        if entry.get("schema_version") != EXPECTED_ADMISSION_SCHEMA:
            _fail(f"{label} has an unexpected schema_version")
        context_id = str(_require(entry, "context_id", label))
        if str(plan.contexts[context_id]["image_id"]) != shard.image_id:
            _fail(f"{label} admits a context belonging to another image")
        expected = reconstruct_admission_receipt_id(plan, entry, label=label)
        receipt_id = str(_require(entry, "admission_receipt_id", label))
        if receipt_id != expected:
            _fail(
                f"{label} for context {context_id!r} does not reconstruct its own "
                f"admission_receipt_id on channel {entry.get('channel')!r}"
            )
        if entry.get("inherited_from_another_prefix") or entry.get(
            "inherited_from_another_category"
        ):
            _fail(
                f"{label} is marked inherited; admission is never transferable across "
                "prefixes or categories"
            )
        if receipt_id in indexed:
            _fail(f"{label} duplicates admission receipt {receipt_id!r}")
        indexed[receipt_id] = dict(entry)
    return AdmissionIndex(image_id=shard.image_id, by_receipt_id=indexed)


def validate_shard_lineage(
    plan: PlanBundle, shard: ShardArtifacts, *, require_complete_shard: bool = True
) -> dict[str, Any]:
    """Bind one captured shard to the sealed plan, capture rules, and runtime.

    ``require_complete_shard`` is on for every conclusion-bearing path.  The
    scorer marks a partial run ``capture_completeness == "subset_smoke"``; such
    a shard dropped into a full ``shard_root`` would otherwise be hashed and
    counted as captured.  Only the explicitly non-conclusion smoke-admit path
    relaxes this.
    """

    receipt = shard.receipt or {}
    label = f"shard {shard.image_id} receipt"
    if receipt.get("schema_version") != EXPECTED_SHARD_RECEIPT_SCHEMA:
        _fail(f"{label} has an unexpected schema_version")
    if receipt.get("unit_id") != UNIT_ID:
        _fail(f"{label} unit_id does not match this unit")
    if str(_require(receipt, "image_id", label)) != shard.image_id:
        _fail(f"{label} names a different image than its directory")
    if str(receipt.get("split")) != shard.split:
        _fail(f"{label} split does not match the frozen split assignment")

    plan_block = receipt.get("plan")
    if not isinstance(plan_block, Mapping):
        _fail(f"{label} has no plan lineage block")
    if plan_block.get("receipt_content_sha256") != plan.receipt["receipt_content_sha256"]:
        _fail(f"{label} was captured against a different plan receipt")
    if plan_block.get("capture_rules_sha256") != plan.capture_rules["capture_rules_sha256"]:
        _fail(f"{label} was captured against different capture rules")

    identity = receipt.get("backend_identity")
    if not isinstance(identity, Mapping):
        _fail(f"{label} has no backend_identity block")
    # A deterministic stub exercises the identical code path but is not
    # evidence.  It is refused here by its own self-declaration rather than by
    # a backend-name special case, so a stub can never be merged into a census.
    if identity.get("usable_as_evidence") is False or identity.get("is_real_model") is False:
        _fail(
            f"{label} was captured on a backend that declares itself unusable as evidence "
            "(is_real_model/usable_as_evidence false); it may not enter a census merge"
        )
    model_identity = identity.get("model_identity")
    tokenizer_identity = identity.get("tokenizer_identity")
    if not model_identity:
        _fail(f"{label} backend_identity does not carry a model identity")
    if not tokenizer_identity:
        _fail(f"{label} backend_identity does not carry a tokenizer identity")

    completeness = str(receipt.get("capture_completeness", ""))
    subset = receipt.get("subset_capture") or {}
    complete_evidence = bool(subset.get("usable_as_complete_shard_evidence", False))
    if require_complete_shard and (
        completeness != COMPLETE_SHARD or not complete_evidence
    ):
        _fail(
            f"{label} declares capture_completeness={completeness!r} "
            f"(usable_as_complete_shard_evidence={complete_evidence}); only a complete "
            "shard may enter a conclusion-bearing merge"
        )
    if completeness not in {COMPLETE_SHARD, SUBSET_SMOKE}:
        _fail(f"{label} declares an unknown capture_completeness {completeness!r}")

    code = receipt.get("code")
    if not isinstance(code, Mapping):
        _fail(f"{label} has no code lineage block")

    declared_counts = receipt.get("counts") or {}
    if shard.evidence_loaded:
        # Only checkable once the evidence is genuinely loaded; the
        # metadata-only manifest phase deliberately has no row counts to compare.
        declared_rows = declared_counts.get("localization_score_rows")
        if declared_rows is not None and int(declared_rows) != len(shard.scores):
            _fail(
                f"{label} declares {declared_rows} localization score rows but "
                f"{len(shard.scores)} were published"
            )
        declared_proposals = declared_counts.get("proposal_surface_rows")
        if declared_proposals is not None and int(declared_proposals) != len(shard.proposals):
            _fail(
                f"{label} declares {declared_proposals} proposal rows but "
                f"{len(shard.proposals)} were published"
            )

    declared_digests = receipt.get("output_file_digests")
    if isinstance(declared_digests, Mapping):
        for name, expected in declared_digests.items():
            observed = shard.file_digests.get(str(name))
            if observed is None:
                _fail(
                    f"{label} declares an output digest for {name!r} but no such file was "
                    "published; a declared-but-absent output is an incomplete shard"
                )
            if observed != expected:
                _fail(f"{label} declares a different digest for {name!r} than was published")

    return {
        "image_id": shard.image_id,
        "split": shard.split,
        "shard_dir": str(shard.directory),
        "model_identity_sha256": sha256_json(model_identity),
        "tokenizer_identity_sha256": sha256_json(tokenizer_identity),
        "executed_source_sha256": code.get("executed_source_sha256"),
        "planner_source_sha256": code.get("planner_source_sha256"),
        "published_file_digests": dict(sorted(shard.file_digests.items())),
        "capture_completeness": completeness,
        "usable_as_complete_shard_evidence": complete_evidence,
        "evidence_loaded": shard.evidence_loaded,
        "score_row_count": len(shard.scores) if shard.evidence_loaded else None,
        "proposal_row_count": len(shard.proposals) if shard.evidence_loaded else None,
        "free_decode_row_count": len(shard.free_decodes) if shard.evidence_loaded else None,
    }


def assert_uniform_runtime(lineages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every captured shard must have run on one model, tokenizer, and code state."""

    if not lineages:
        _fail("no captured shard survived admission; there is nothing to merge")
    fields = (
        "model_identity_sha256",
        "tokenizer_identity_sha256",
        "executed_source_sha256",
        "planner_source_sha256",
    )
    uniform: dict[str, Any] = {}
    for name in fields:
        observed = {str(row.get(name)) for row in lineages}
        if len(observed) != 1:
            _fail(
                f"captured shards disagree on {name}: {sorted(observed)}; "
                "a census merge may not span two runtime or code identities"
            )
        uniform[name] = next(iter(observed))
    return uniform


# ---------------------------------------------------------------------------
# Event admission
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreEvent:
    """One admitted unique coordinate score event."""

    image_id: str
    split: str
    context_id: str
    normalized_description: str
    query_group_id: str
    candidate_id: str
    coord_token_ids: tuple[int, ...]
    complete_box_logprob_sum: float
    observed_prefix_sha256: str
    query_prefix_sha256: str
    query_suffix_token_ids_sha256: str
    admission_receipt_id: str


def admit_score_rows(
    plan: PlanBundle, shard: ShardArtifacts, admissions: AdmissionIndex
) -> list[ScoreEvent]:
    """Admit one shard's localization rows against the plan and its admissions.

    Rejects, in this order: foreign schema, a pre-P0 row lacking the query
    suffix digest, a prefix digest that does not match the plan's query group,
    a row whose covering admission receipt is absent or not admitted, a row
    whose covering receipt belongs to the wrong channel, and a duplicate
    ``(context, category, coordinate tuple)`` event.
    """

    seen_events: dict[tuple[str, str, tuple[int, ...]], str] = {}
    events: list[ScoreEvent] = []
    for index, row in enumerate(shard.scores):
        label = f"shard {shard.image_id} score row {index}"
        if row.get("schema_version") != EXPECTED_SCORE_SCHEMA:
            _fail(f"{label} has an unexpected schema_version")
        if row.get("row_kind") != "census_localization_score":
            _fail(f"{label} is not a localization score row")
        if row.get("is_sidecar"):
            _fail(f"{label} is marked a sidecar; sidecars never enter the event table")

        # Pre-P0 quarantine: a row without the canonical query-suffix digest
        # cannot be proven to have been read at the forced box opener.
        suffix_sha = row.get("query_suffix_token_ids_sha256")
        if not suffix_sha:
            _fail(
                f"{label} carries no query_suffix_token_ids_sha256; pre-P0 rows are "
                "quarantined and mechanically unjoinable to this schema"
            )

        group_id = str(_require(row, "query_group_id", label))
        group = plan.query_groups.get(group_id)
        if group is None:
            _fail(f"{label} names unknown query group {group_id!r}")
        if group.get("status") != "admitted":
            _fail(f"{label} scores query group {group_id!r}, which the plan did not admit")
        if str(group["image_id"]) != shard.image_id:
            _fail(f"{label} scores a query group belonging to another image")

        if str(suffix_sha) != str(group["query_suffix_token_ids_sha256"]):
            _fail(f"{label} query suffix digest does not match its plan query group")
        observed_sha = str(_require(row, "observed_prefix_sha256", label))
        query_sha = str(_require(row, "query_prefix_sha256", label))
        if observed_sha != str(group["observed_prefix_sha256"]):
            _fail(f"{label} observed prefix digest does not match its plan query group")
        if query_sha != str(group["query_prefix_sha256"]):
            _fail(f"{label} query prefix digest does not match its plan query group")

        context_id = str(group["context_id"])
        description = str(group["normalized_description"])
        rank_key = row.get("rank_key")
        if not isinstance(rank_key, Mapping):
            _fail(f"{label} carries no rank_key")
        expected_rank_key = {
            "image_id": shard.image_id,
            "context_id": context_id,
            "normalized_description": description,
        }
        if {str(key): str(value) for key, value in rank_key.items()} != {
            key: str(value) for key, value in expected_rank_key.items()
        }:
            _fail(f"{label} rank_key is not the plan's (image, context, category) key")

        # Exact admission coverage for this row's channel and prefix.
        expected_receipt_id = planner.admission_receipt_id(
            context_id=context_id,
            channel=planner.CHANNEL_QUERY_SUFFIX,
            prefix_sha256=query_sha,
        )
        declared_receipt_id = str(_require(row, "admission_receipt_id", label))
        if declared_receipt_id != expected_receipt_id:
            _fail(
                f"{label} references admission receipt {declared_receipt_id!r} but its "
                f"channel/prefix require {expected_receipt_id!r}"
            )
        covering = admissions.covering(expected_receipt_id)
        if covering is None:
            _fail(
                f"{label} has no covering admission receipt for channel "
                f"{planner.CHANNEL_QUERY_SUFFIX!r} on its exact query prefix"
            )
        if str(covering.get("channel")) != planner.CHANNEL_QUERY_SUFFIX:
            _fail(f"{label} covering admission receipt belongs to another channel")
        if not covering.get("admitted"):
            _fail(f"{label} covering admission receipt did not admit its context")

        candidate_id = str(_require(row, "candidate_id", label))
        candidate = plan.candidates.get(candidate_id)
        if candidate is None:
            _fail(f"{label} names unknown physical candidate {candidate_id!r}")
        if candidate_id not in set(group["candidate_ids"]):
            _fail(f"{label} scores a candidate outside its query group's bank")
        coord_tokens = tuple(int(value) for value in _require(row, "coord_token_ids", label))
        if coord_tokens != tuple(int(value) for value in candidate["coord_token_ids"]):
            _fail(f"{label} coordinate tokens do not match the sealed candidate bank")

        event_key = (context_id, description, coord_tokens)
        if event_key in seen_events:
            _fail(
                f"{label} duplicates the (context, category, coordinate tuple) event already "
                f"scored by {seen_events[event_key]!r}; a collapsed tuple carries one mass"
            )
        seen_events[event_key] = candidate_id

        events.append(
            ScoreEvent(
                image_id=shard.image_id,
                split=shard.split,
                context_id=context_id,
                normalized_description=description,
                query_group_id=group_id,
                candidate_id=candidate_id,
                coord_token_ids=coord_tokens,
                complete_box_logprob_sum=_finite(
                    _require(row, "complete_box_logprob_sum", label),
                    f"{label} complete_box_logprob_sum",
                ),
                observed_prefix_sha256=observed_sha,
                query_prefix_sha256=query_sha,
                query_suffix_token_ids_sha256=str(suffix_sha),
                admission_receipt_id=expected_receipt_id,
            )
        )
    return events


def _require_covering(
    admissions: AdmissionIndex, receipt_id: str, *, channel: str, label: str
) -> Mapping[str, Any]:
    covering = admissions.covering(receipt_id)
    if covering is None:
        _fail(
            f"{label} has no covering admission receipt for channel {channel!r}; "
            "an admission proved on another channel can never cover it"
        )
    if str(covering.get("channel")) != channel:
        _fail(f"{label} covering admission receipt belongs to another channel")
    if not covering.get("admitted"):
        _fail(f"{label} covering admission receipt did not admit its context")
    return covering


def admit_proposal_rows(
    plan: PlanBundle, shard: ShardArtifacts, admissions: AdmissionIndex
) -> dict[str, dict[str, Any]]:
    """Admit the proposal surface on both of its distinct channels.

    The boundary gate reads the observed prefix with nothing forced, so it is
    admitted once per context.  Each category routing event traverses its own
    token identity and length, so it is admitted per ``(context, category)`` on
    the observed prefix *plus* the routing-path digest.  A query-suffix
    admission covers neither, and a boundary-gate admission does not cover a
    routing event.
    """

    surfaces: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(shard.proposals):
        label = f"shard {shard.image_id} proposal row {index}"
        if row.get("schema_version") != EXPECTED_PROPOSAL_SCHEMA:
            _fail(f"{label} has an unexpected schema_version")
        context_id = str(_require(row, "context_id", label))
        context = plan.contexts.get(context_id)
        if context is None:
            _fail(f"{label} names unknown context {context_id!r}")
        if str(context["image_id"]) != shard.image_id:
            _fail(f"{label} names a context belonging to another image")
        observed_sha = str(_require(row, "observed_prefix_sha256", label))

        gate_receipt_id = planner.admission_receipt_id(
            context_id=context_id,
            channel=planner.CHANNEL_PROPOSAL_BOUNDARY_GATE,
            prefix_sha256=observed_sha,
        )
        declared = str(_require(row, "boundary_gate_admission_receipt_id", label))
        if declared != gate_receipt_id:
            _fail(
                f"{label} references boundary-gate admission {declared!r} but its "
                f"channel/prefix require {gate_receipt_id!r}"
            )
        _require_covering(
            admissions,
            gate_receipt_id,
            channel=planner.CHANNEL_PROPOSAL_BOUNDARY_GATE,
            label=f"{label} boundary gate",
        )

        routing = row.get("category_routing_event") or ()
        seen_routes: set[str] = set()
        for position, event in enumerate(routing):
            event_label = f"{label} routing event {position}"
            if not isinstance(event, Mapping):
                _fail(f"{event_label} is not a JSON object")
            description = str(_require(event, "normalized_description", event_label))
            category = _category_for(plan, shard.image_id, description, event_label)
            route_receipt_id = planner.proposal_route_admission_receipt_id(
                context_id=context_id,
                observed_prefix_sha256=observed_sha,
                category_token_ids=category["category_token_ids"],
            )
            if str(event.get("channel")) != planner.CHANNEL_PROPOSAL_CATEGORY_ROUTE:
                _fail(
                    f"{event_label} does not declare the "
                    f"{planner.CHANNEL_PROPOSAL_CATEGORY_ROUTE!r} channel"
                )
            declared_digest = str(_require(event, "routing_path_digest", event_label))
            if declared_digest != planner.proposal_route_digest(
                [int(value) for value in category["category_token_ids"]]
            ):
                _fail(
                    f"{event_label} declares a routing-path digest that does not match its "
                    "planned category routing path"
                )
            declared_route = str(_require(event, "admission_receipt_id", event_label))
            if declared_route != route_receipt_id:
                _fail(
                    f"{event_label} references route admission {declared_route!r} but its "
                    f"channel/prefix require {route_receipt_id!r}"
                )
            _require_covering(
                admissions,
                route_receipt_id,
                channel=planner.CHANNEL_PROPOSAL_CATEGORY_ROUTE,
                label=event_label,
            )
            if route_receipt_id in seen_routes:
                _fail(f"{event_label} duplicates a routing event for this category")
            seen_routes.add(route_receipt_id)

        if context_id in surfaces:
            _fail(f"{label} duplicates the proposal surface for context {context_id!r}")
        surfaces[context_id] = dict(row)
    return surfaces


def _category_for(
    plan: PlanBundle, image_id: str, description: str, label: str
) -> Mapping[str, Any]:
    for row in plan.categories.values():
        if (
            str(row["image_id"]) == image_id
            and str(row["normalized_description"]) == description
        ):
            return row
    _fail(f"{label} names category {description!r}, which image {image_id} does not declare")



#: The free-decode file carries two row kinds: one free greedy *box* per
#: ``(context, category)``, and one free next-*row* decode per context.  Only
#: the box sidecars carry a category and a coordinate tuple.
FREE_BOX_ROW_KIND = "census_free_greedy_box_sidecar"
FREE_NEXT_ROW_KIND = "census_free_next_row_sidecar"


def free_box_sidecars(shard: ShardArtifacts) -> list[Mapping[str, Any]]:
    """The free greedy box rows of one shard, validated and row-kind filtered."""

    rows: list[Mapping[str, Any]] = []
    for row in shard.free_decodes:
        if row.get("schema_version") != EXPECTED_FREE_DECODE_SCHEMA:
            _fail(f"shard {shard.image_id} free-decode row has an unexpected schema_version")
        kind = str(row.get("row_kind"))
        if kind == FREE_BOX_ROW_KIND:
            rows.append(row)
        elif kind != FREE_NEXT_ROW_KIND:
            _fail(f"shard {shard.image_id} free-decode row has unknown row_kind {kind!r}")
    return rows


def _sidecar_coord_tokens(row: Mapping[str, Any]) -> tuple[int, ...] | None:
    """A free box's coordinate tuple, or ``None`` when it decoded malformed."""

    tokens = row.get("coord_token_ids")
    if tokens is None:
        bins = row.get("coord_bins")
        if bins is None:
            return None
        tokens = planner.coord_token_ids([int(value) for value in bins])
    tokens = tuple(int(value) for value in tokens)
    if len(tokens) != 4:
        return None
    if any(
        token < planner.COORD_TOKEN_START or token > planner.COORD_TOKEN_END
        for token in tokens
    ):
        return None
    return tokens


def index_sidecars(
    plan: PlanBundle, shards: Sequence[ShardArtifacts]
) -> dict[tuple[str, str, tuple[int, ...]], dict[str, Any]]:
    """Sidecar provenance keyed by ``(image, category, coordinate tuple)``.

    Native emitted boxes and free greedy decodes join an event's *provenance*
    and never add a second rank mass, so they are indexed here and never
    appended to any population.
    """

    index: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    for row in plan.native_sidecars:
        key = (
            str(row["image_id"]),
            str(row["normalized_description"]),
            tuple(int(value) for value in row["coord_token_ids"]),
        )
        entry = index.setdefault(key, {"native_sidecar_ids": [], "free_decode_sidecar_ids": []})
        entry["native_sidecar_ids"].append(str(row["sidecar_id"]))
    for shard in shards:
        for row in free_box_sidecars(shard):
            tokens = _sidecar_coord_tokens(row)
            if tokens is None:
                continue
            key = (str(row["image_id"]), str(row["normalized_description"]), tokens)
            entry = index.setdefault(
                key, {"native_sidecar_ids": [], "free_decode_sidecar_ids": []}
            )
            entry["free_decode_sidecar_ids"].append(str(row["sidecar_id"]))
    for entry in index.values():
        entry["native_sidecar_ids"].sort()
        entry["free_decode_sidecar_ids"].sort()
    return index



def build_sidecar_diagnostics(
    plan: PlanBundle,
    shards: Sequence[ShardArtifacts],
    owner_contexts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Continuous sidecar surface for finite-bank undercoverage diagnosis.

    A free greedy box that lands *outside* the fixed bank is the sharpest
    available evidence that the seventeen-role bank under-covers an owner's
    real likelihood landscape.  Joining only exact-tuple provenance onto core
    events would make exactly those boxes disappear, so they are published here
    instead, with their assignment against the same-description ground truth and
    their signed gap against each relevant owner's fixed-bank best at the same
    context.

    Nothing here enters a rank, a posterior or a support test.  No threshold or
    undercoverage label is applied: the quantities are continuous and the
    interpretation is left to the analysis.
    """

    best_by_owner_context: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in owner_contexts:
        best_by_owner_context[(str(row["gt_owner_id"]), str(row["context_id"]))] = row

    bank_tuples: dict[tuple[str, str], set[tuple[int, ...]]] = {}
    for candidate in plan.candidates.values():
        bank_tuples.setdefault(
            (str(candidate["image_id"]), str(candidate["normalized_description"])), set()
        ).add(tuple(int(value) for value in candidate["coord_token_ids"]))

    rows: list[dict[str, Any]] = []
    for shard in shards:
        image = plan.images[shard.image_id]
        canvas = planner.Canvas(int(image["image_width"]), int(image["image_height"]))
        image_owners = plan.owners_in_image(shard.image_id)

        for row in free_box_sidecars(shard):
            description = str(row["normalized_description"])
            context_id = str(row["context_id"])
            tokens = _sidecar_coord_tokens(row)
            well_formed = tokens is not None and row.get("well_formed_box") is not False
            entry: dict[str, Any] = {
                "schema_version": SIDECAR_DIAGNOSTIC_SCHEMA_VERSION,
                "row_kind": "census_sidecar_diagnostic",
                "sidecar_kind": "free_greedy_box",
                "sidecar_id": str(row["sidecar_id"]),
                "image_id": shard.image_id,
                "split": shard.split,
                "context_id": context_id,
                "normalized_description": description,
                "query_group_id": str(row.get("query_group_id", "")),
                "coord_token_ids": list(tokens) if tokens else None,
                "complete_box_logprob_sum": row.get("complete_box_logprob_sum"),
                "well_formed_box": bool(well_formed),
                "malformed_reason": row.get("malformed_reason"),
                "enters_core_ranks": False,
                "enters_support_test": False,
                "role": "continuous_undercoverage_diagnostic_only",
            }
            if not well_formed or tokens is None:
                entry["joins_fixed_bank_candidate_id"] = None
                entry["inside_fixed_bank"] = None
                entry["owner_gaps"] = []
                rows.append(entry)
                continue

            joined = next(
                (
                    str(candidate["candidate_id"])
                    for candidate in plan.candidates.values()
                    if str(candidate["image_id"]) == shard.image_id
                    and str(candidate["normalized_description"]) == description
                    and tuple(int(v) for v in candidate["coord_token_ids"]) == tokens
                ),
                None,
            )
            entry["joins_fixed_bank_candidate_id"] = joined
            entry["inside_fixed_bank"] = joined is not None
            entry["fixed_bank_tuple_count"] = len(
                bank_tuples.get((shard.image_id, description), ())
            )

            decoded = canvas.bins_to_pixel(
                [token - planner.COORD_TOKEN_START for token in tokens]
            )
            entry["decoded_bbox_pixel_xyxy"] = list(decoded)
            entry.update(
                planner.strict_assignment(
                    decoded, image_owners, normalized_description=description
                )
            )

            score = row.get("complete_box_logprob_sum")
            gaps: list[dict[str, Any]] = []
            for owner in image_owners:
                if str(owner["normalized_description"]) != description:
                    continue
                owner_id = str(owner["gt_owner_id"])
                context_row = best_by_owner_context.get((owner_id, context_id))
                bounds = (
                    context_row["localization"][
                        "generator_local_max_excluding_other_owner_strict"
                    ]
                    if context_row is not None
                    else None
                )
                owner_box = [float(v) for v in owner["bbox_pixel_xyxy"]]
                overlap = planner.iou(decoded, owner_box)
                gap_entry: dict[str, Any] = {
                    "gt_owner_id": owner_id,
                    "fixed_bank_best_l": (
                        bounds["ambiguity_excluded_l"]["value"] if bounds else None
                    ),
                    "fixed_bank_best_u": (
                        bounds["ambiguity_included_u"]["value"] if bounds else None
                    ),
                    "geometry_provenance": (
                        "merge_computed_free_box_versus_owner_no_plan_entry_exists"
                    ),
                    "intersection_over_union_with_owner": overlap,
                    "center_offset_pixels": [
                        ((decoded[0] + decoded[2]) / 2.0)
                        - ((owner_box[0] + owner_box[2]) / 2.0),
                        ((decoded[1] + decoded[3]) / 2.0)
                        - ((owner_box[1] + owner_box[3]) / 2.0),
                    ],
                    "extent_ratio": [
                        (decoded[2] - decoded[0]) / (owner_box[2] - owner_box[0])
                        if owner_box[2] > owner_box[0]
                        else None,
                        (decoded[3] - decoded[1]) / (owner_box[3] - owner_box[1])
                        if owner_box[3] > owner_box[1]
                        else None,
                    ],
                }
                for suffix in ("l", "u"):
                    best = gap_entry[f"fixed_bank_best_{suffix}"]
                    # Positive: the free box outscores everything the owner's
                    # fixed bank contains at this context.
                    gap_entry[f"signed_gap_vs_fixed_bank_best_{suffix}"] = (
                        float(score) - float(best)
                        if score is not None and best is not None
                        else None
                    )
                gaps.append(gap_entry)
            entry["owner_gaps"] = gaps
            rows.append(entry)

    # Native emitted boxes: join/outside-bank provenance only.
    for sidecar in plan.native_sidecars:
        image_id = str(sidecar["image_id"])
        if image_id not in {shard.image_id for shard in shards}:
            continue
        description = str(sidecar["normalized_description"])
        tokens = tuple(int(value) for value in sidecar.get("coord_token_ids", ()))
        rows.append(
            {
                "schema_version": SIDECAR_DIAGNOSTIC_SCHEMA_VERSION,
                "row_kind": "census_sidecar_diagnostic",
                "sidecar_kind": "native_emitted_box",
                "sidecar_id": str(sidecar["sidecar_id"]),
                "image_id": image_id,
                "split": str(sidecar.get("split", planner.SPLIT_BY_IMAGE_ID[image_id])),
                "normalized_description": description,
                "coord_token_ids": list(tokens),
                "joins_fixed_bank_candidate_id": sidecar.get("joins_physical_candidate_id"),
                "inside_fixed_bank": tokens in bank_tuples.get((image_id, description), set()),
                "strict_match_status": sidecar.get("strict_match_status"),
                "strict_match_gt_owner_id": sidecar.get("strict_match_gt_owner_id"),
                "enters_core_ranks": False,
                "enters_support_test": False,
                "role": "continuous_undercoverage_diagnostic_only",
            }
        )

    rows.sort(key=lambda row: (str(row["image_id"]), str(row["sidecar_id"])))
    return rows


# ---------------------------------------------------------------------------
# Competition: ranks, margins, and posteriors within (image, context, category)
# ---------------------------------------------------------------------------


def build_event_table(
    plan: PlanBundle,
    events: Sequence[ScoreEvent],
    sidecars: Mapping[tuple[str, str, tuple[int, ...]], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Publish the ranked event table.

    The rank population is exactly the collapsed unique physical candidates of
    one ``(image, context, category)``.  A candidate two owners both generated
    appears once and carries one mass; sidecars are excluded from the
    population entirely and appear only as provenance.
    """

    grouped: dict[tuple[str, str, str], list[ScoreEvent]] = {}
    for event in events:
        grouped.setdefault(
            (event.image_id, event.context_id, event.normalized_description), []
        ).append(event)

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        image_id, context_id, description = key
        population = sorted(
            grouped[key],
            key=lambda item: (-item.complete_box_logprob_sum, item.candidate_id),
        )
        scores = [item.complete_box_logprob_sum for item in population]
        posteriors = _log_softmax(scores)
        best = scores[0]
        runner_up = scores[1] if len(scores) > 1 else None
        context = plan.contexts[context_id]

        for position, (event, log_posterior) in enumerate(zip(population, posteriors, strict=True)):
            candidate = plan.candidates[event.candidate_id]
            provenance = sidecars.get(
                (image_id, description, event.coord_token_ids),
                {"native_sidecar_ids": [], "free_decode_sidecar_ids": []},
            )
            rows.append(
                {
                    "schema_version": EVENT_SCHEMA_VERSION,
                    "row_kind": "census_merged_event",
                    "event_id": f"{event.query_group_id}|{event.candidate_id}",
                    "image_id": image_id,
                    "split": event.split,
                    "context_id": context_id,
                    "boundary_index": int(context["boundary_index"]),
                    "context_role": str(context["context_role"]),
                    "loop_tail": bool(context["loop_marking"]["loop_tail"]),
                    "normalized_description": description,
                    "query_group_id": event.query_group_id,
                    "candidate_id": event.candidate_id,
                    "coord_token_ids": list(event.coord_token_ids),
                    "complete_box_logprob_sum": event.complete_box_logprob_sum,
                    "rank_key": {
                        "image_id": image_id,
                        "context_id": context_id,
                        "normalized_description": description,
                    },
                    "competition": {
                        "population": "collapsed_unique_physical_candidates",
                        "population_size": len(population),
                        "rank": position + 1,
                        "margin_to_group_best": event.complete_box_logprob_sum - best,
                        "margin_to_runner_up": (
                            best - runner_up if position == 0 and runner_up is not None else None
                        ),
                        "within_group_log_posterior": log_posterior,
                        "within_group_posterior": math.exp(log_posterior),
                        "posterior_semantics": (
                            "renormalized_competition_posterior_over_collapsed_unique_candidates"
                        ),
                        "is_model_probability": False,
                        "sidecars_excluded_from_population": True,
                    },
                    "candidate_provenance": {
                        "generator_gt_owner_ids": list(candidate["generator_gt_owner_ids"]),
                        "generator_owner_count": int(candidate["generator_owner_count"]),
                        "cross_owner_generated": bool(candidate["cross_owner_generated"]),
                        "representative_role": str(candidate["representative_role"]),
                        "candidate_class": str(candidate["candidate_class"]),
                        "candidate_provenance": str(candidate["candidate_provenance"]),
                        "role": "provenance_only_never_rank_or_assignment",
                    },
                    "strict_assignment": {
                        "scope": str(candidate["strict_assignment_scope"]),
                        "status": str(candidate["strict_assignment_status"]),
                        "gt_owner_id": candidate["strict_assignment_gt_owner_id"],
                        "ambiguity_owner_ids": list(candidate["ambiguity_owner_ids"]),
                    },
                    "sidecar_provenance": {
                        **provenance,
                        "join_semantics": "joins_provenance_never_adds_rank_mass",
                    },
                    "identity": {
                        "observed_prefix_sha256": event.observed_prefix_sha256,
                        "query_prefix_sha256": event.query_prefix_sha256,
                        "query_suffix_token_ids_sha256": event.query_suffix_token_ids_sha256,
                        "admission_receipt_id": event.admission_receipt_id,
                    },
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Owner-context continuous feature reconstruction
# ---------------------------------------------------------------------------


def _frontier_features(
    context: Mapping[str, Any],
    owner: Mapping[str, Any],
    image_owners: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Signed, continuous frontier geometry for one owner at one context.

    Positive ordinal/pixel distance means the owner lies *ahead* of the
    frontier in the geometry sort order; negative means the sweep has already
    passed it.  The root context has no frontier and every distance is null,
    never zero, so "no frontier" can never be read as "at the frontier".
    """

    description = str(owner["normalized_description"])
    same_description = [
        row for row in image_owners if str(row["normalized_description"]) == description
    ]
    frontier = context.get("frontier")
    if frontier is None:
        return {
            "frontier_present": False,
            "frontier_pred_row_id": None,
            "signed_frontier_ordinal_distance": None,
            "abs_signed_frontier_ordinal_distance": None,
            "signed_frontier_pixel_distance_y": None,
            "signed_frontier_pixel_distance_x": None,
            "signed_frontier_sort_axis_pixel_distance": None,
            "same_description_owners_ahead_of_frontier": len(same_description),
            "same_description_owners_between_frontier_and_owner": None,
            "passed_state": "root_no_frontier",
            "frontier_overlap": {
                "intersection_over_union": None,
                "overlaps_frontier": None,
                "center_offset_pixels": None,
                "extent_ratio": None,
            },
        }

    frontier_key = [float(value) for value in frontier["sort_key"]]
    owner_key = [float(value) for value in owner["owner_sort_key"]]
    all_keys = [[float(value) for value in row["owner_sort_key"]] for row in image_owners]

    # Signed ordinal distance is measured against the frontier's own position in
    # the geometry sort order, not against an insertion index.  An insertion
    # index would give the next-unreached owner a distance of zero and so
    # conflate "immediately ahead" with "at the frontier"; here zero means the
    # owner sorts exactly at the frontier, +n means n owners ahead of it, and
    # -n means n owners behind it.
    if owner_key == frontier_key:
        signed_ordinal = 0
    elif owner_key > frontier_key:
        signed_ordinal = 1 + sum(1 for key in all_keys if frontier_key < key < owner_key)
    else:
        signed_ordinal = -(1 + sum(1 for key in all_keys if owner_key < key < frontier_key))

    delta_y = owner_key[0] - frontier_key[0]
    delta_x = owner_key[1] - frontier_key[1]
    sort_axis = delta_y if delta_y != 0.0 else delta_x

    if signed_ordinal > 0:
        passed_state = "ahead_of_frontier"
    elif signed_ordinal == 0:
        passed_state = "at_frontier"
    else:
        passed_state = "passed_by_frontier"

    low_key, high_key = sorted((owner_key, frontier_key))

    owner_box = [float(value) for value in owner["bbox_pixel_xyxy"]]
    frontier_box = [float(value) for value in frontier["bbox_pixel_xyxy"]]
    overlap = planner.iou(owner_box, frontier_box)
    owner_center = ((owner_box[0] + owner_box[2]) / 2.0, (owner_box[1] + owner_box[3]) / 2.0)
    frontier_center = (
        (frontier_box[0] + frontier_box[2]) / 2.0,
        (frontier_box[1] + frontier_box[3]) / 2.0,
    )
    frontier_width = frontier_box[2] - frontier_box[0]
    frontier_height = frontier_box[3] - frontier_box[1]

    return {
        "frontier_present": True,
        "frontier_pred_row_id": str(frontier["pred_row_id"]),
        "signed_frontier_ordinal_distance": signed_ordinal,
        "abs_signed_frontier_ordinal_distance": abs(signed_ordinal),
        "signed_frontier_pixel_distance_y": delta_y,
        "signed_frontier_pixel_distance_x": delta_x,
        "signed_frontier_sort_axis_pixel_distance": sort_axis,
        "same_description_owners_ahead_of_frontier": sum(
            1
            for row in same_description
            if [float(value) for value in row["owner_sort_key"]] > frontier_key
        ),
        # Direction-agnostic: counts the same-description owners lying strictly
        # between the frontier and this owner, whichever side the owner is on.
        "same_description_owners_between_frontier_and_owner": sum(
            1
            for row in same_description
            if low_key < [float(value) for value in row["owner_sort_key"]] < high_key
        ),
        "passed_state": passed_state,
        "frontier_overlap": {
            "intersection_over_union": overlap,
            "overlaps_frontier": overlap > 0.0,
            "center_offset_pixels": [
                owner_center[0] - frontier_center[0],
                owner_center[1] - frontier_center[1],
            ],
            "extent_ratio": [
                (owner_box[2] - owner_box[0]) / frontier_width if frontier_width > 0 else None,
                (owner_box[3] - owner_box[1]) / frontier_height if frontier_height > 0 else None,
            ],
        },
    }


def _proposal_channel(
    surface: Mapping[str, Any] | None, description: str
) -> dict[str, Any]:
    """The category proposal channel, kept strictly separate from localization."""

    block: dict[str, Any] = {
        "channels": [
            planner.CHANNEL_PROPOSAL_BOUNDARY_GATE,
            planner.CHANNEL_PROPOSAL_CATEGORY_ROUTE,
        ],
        "separate_from_localization": True,
        "combined_with_localization": False,
        "surface_present": surface is not None,
        "boundary_gate": None,
        "category_routing_event": None,
    }
    if surface is None:
        return block
    gate = surface.get("boundary_gate") or {}
    block["boundary_gate"] = {
        "continue_logprob": gate.get("continue_logprob"),
        "stop_logprob": gate.get("stop_logprob"),
        "continue_vs_stop_logprob_margin": gate.get("continue_vs_stop_logprob_margin"),
        "semantics": "gate_only_never_description_accessibility",
    }
    for routing in surface.get("category_routing_event") or ():
        if str(routing.get("normalized_description")) == description:
            block["category_routing_event"] = {
                "raw_sequence_logprob_sum": routing.get("raw_sequence_logprob_sum"),
                "within_context_rank": routing.get("within_context_rank"),
                "within_context_population": routing.get("within_context_population"),
                "row_prefix_block_raw_sequence_logprob_sum": routing.get(
                    "row_prefix_block_raw_sequence_logprob_sum"
                ),
                "aggregation": "raw_sequence_sum_no_token_mean",
            }
            break
    return block


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _best(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Maximum ``complete_box_logprob_sum`` over an event subset, with its identity."""

    if not rows:
        return {
            "value": None,
            "candidate_id": None,
            "rank": None,
            "count": 0,
            "log_posterior": None,
            "unique_population_size": None,
        }
    top = max(
        rows,
        key=lambda row: (row["complete_box_logprob_sum"], -int(row["competition"]["rank"])),
    )
    return {
        "value": float(top["complete_box_logprob_sum"]),
        "candidate_id": str(top["candidate_id"]),
        "rank": int(top["competition"]["rank"]),
        "count": len(rows),
        "log_posterior": float(top["competition"]["within_group_log_posterior"]),
        "unique_population_size": int(top["competition"]["population_size"]),
    }


def _support_features(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Local-peak evidence for one owner subset in one ``(context, category)`` group.

    ``peak_lift`` is how much posterior mass the owner's best candidate holds
    above what a flat distribution over the group's ``N`` unique candidates would
    give it: ``log p_best - log(1/N)``.  It is a *local peak* statement, not a
    ranking: an owner can carry a large lift while another owner's candidate
    ranks above it.

    ``local_concentration`` is the owner's best score minus the median of that
    owner's own exclusion-filtered bank scores, so it measures whether the
    likelihood is concentrated at the owner's geometry rather than spread flat
    across its whole fixed probe landscape.  Both are continuous and neither
    consults an owner ranking.
    """

    best = _best(rows)
    if best["value"] is None:
        return {
            **best,
            "peak_lift": None,
            "local_concentration": None,
            "bank_median": None,
            "bank_size": 0,
        }
    population = int(best["unique_population_size"])
    median = _median([float(row["complete_box_logprob_sum"]) for row in rows])
    return {
        **best,
        "peak_lift": float(best["log_posterior"]) + math.log(float(population)),
        "peak_lift_definition": "log_posterior_of_owner_best_minus_log_uniform_over_unique_population",
        "local_concentration": float(best["value"]) - float(median),
        "local_concentration_definition": (
            "owner_best_exclusion_filtered_score_minus_median_of_own_exclusion_filtered_bank"
        ),
        "bank_median": median,
        "bank_size": len(rows),
    }



def candidate_owner_geometry(
    candidate: Mapping[str, Any], gt_owner_id: str
) -> dict[str, Any]:
    """The sealed per-generator geometry of one candidate against one owner.

    The plan registry is the single source of this geometry and the merge does
    not re-derive it.  The distinction is not cosmetic: after cross-owner alias
    collapse one physical candidate can carry several generators with
    *different* GT boxes, so the same coordinate tuple has a different IoU,
    centre offset and extent ratio for each of them.  A merge-local formula
    keyed on "the candidate and the current owner" happens to agree today, but
    it is a second implementation of a published contract, and the
    candidate-level ``representative_generator_geometry`` summary names only
    the first generator -- reading it for a second owner would silently
    attribute one owner's displacement to another.

    Fails closed when this owner has no geometry provenance on the candidate,
    or when the candidate carries conflicting geometry for the same owner.
    """

    owner_id = str(gt_owner_id)
    candidate_id = str(candidate.get("candidate_id"))
    matches = [
        entry
        for entry in candidate.get("generators", ())
        if str(entry.get("generator_gt_owner_id")) == owner_id
    ]
    if not matches:
        _fail(
            f"candidate {candidate_id!r} carries no generator geometry for owner "
            f"{owner_id!r}; a candidate in an owner's bank must name that owner as a "
            "generator, and the merge will not re-derive the geometry itself"
        )
    geometries = [entry.get("geometry") for entry in matches]
    if any(geometry is None for geometry in geometries):
        _fail(
            f"candidate {candidate_id!r} has a generator entry for owner {owner_id!r} "
            "without sealed geometry"
        )
    # One owner may reach the same tuple through two roles; the geometry is a
    # function of the decoded box and the owner box, so those must agree.
    distinct = {sha256_json(geometry) for geometry in geometries}
    if len(distinct) != 1:
        _fail(
            f"candidate {candidate_id!r} carries {len(distinct)} conflicting generator "
            f"geometries for owner {owner_id!r}; the sealed plan is not self-consistent"
        )

    entry = matches[0]
    sealed = dict(entry["geometry"])
    center = [float(value) for value in sealed["center_offset_pixels"]]
    extent = list(sealed["extent_ratio"])
    generator_extent = [float(value) for value in sealed["generator_extent_pixels"]]

    return {
        # Provenance first: these numbers are published by the planner, not
        # computed here.
        "geometry_provenance": "sealed_plan_per_generator_geometry",
        "authority": "candidate.generators[].geometry",
        "recomputed_in_merge": False,
        "generator_gt_owner_id": owner_id,
        "logical_transform_role": str(entry.get("logical_transform_role", "")),
        "generator_entry_count_for_owner": len(matches),
        "measured_on": sealed.get("measured_on"),
        # Passthrough of the sealed per-generator fields.
        "owner_bbox_pixel_xyxy": sealed["generator_bbox_pixel_xyxy"],
        "intersection_over_union_with_owner": sealed[
            "intersection_over_union_with_generator"
        ],
        "signed_center_offset_pixels": center,
        "width_extent_ratio": extent[0],
        "height_extent_ratio": extent[1],
        "area_ratio": sealed["area_ratio"],
        "candidate_extent_pixels": sealed["candidate_extent_pixels"],
        "generator_extent_pixels": generator_extent,
        # Normalized view, derived from the sealed offset and the sealed
        # generator extent only -- never from the candidate and owner boxes.
        "signed_center_offset_owner_relative": [
            center[0] / generator_extent[0] if generator_extent[0] > 0 else None,
            center[1] / generator_extent[1] if generator_extent[1] > 0 else None,
        ],
        # Candidate-level passthrough; identical for every generator because it
        # is the decoded tuple, not a per-owner quantity.
        "decoded_bbox_pixel_xyxy": list(candidate.get("decoded_bbox_pixel_xyxy", ())),
        "role": "continuous_geometry_no_threshold_no_phenotype_label",
        "enters_score_rank_support_or_population": False,
    }


def _attach_geometry(
    block: dict[str, Any], plan: PlanBundle, owner: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach this owner's sealed generator geometry to one owner-local best."""

    candidate_id = block.get("candidate_id")
    if not candidate_id:
        block["geometry"] = None
        return block
    candidate = plan.candidates.get(str(candidate_id))
    if candidate is None:
        _fail(f"owner-local block names unknown candidate {candidate_id!r}")
    block["geometry"] = candidate_owner_geometry(
        candidate, str(owner["gt_owner_id"])
    )
    return block


def build_owner_context_features(
    plan: PlanBundle,
    events: Sequence[Mapping[str, Any]],
    proposal_surfaces: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """One row per ``(owner, context)`` that this census actually tested.

    The primary neighbourhood is generator-local; the only exclusion is a
    candidate strictly assigned to *another* owner, which moves to the collision
    diagnostic.  Ambiguity-neutral candidates count toward the target only
    through the U bound, so the L and U maxima are published side by side and a
    disposition that flips between them is never resolved.
    """

    by_group: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in events:
        by_group.setdefault(
            (
                str(row["image_id"]),
                str(row["context_id"]),
                str(row["normalized_description"]),
            ),
            [],
        ).append(row)

    rows: list[dict[str, Any]] = []

    for key in sorted(by_group):
        image_id, context_id, description = key
        group_events = by_group[key]
        events_by_candidate = {str(row["candidate_id"]): row for row in group_events}
        context = plan.contexts[context_id]
        image_owners = plan.owners_in_image(image_id)
        surface = proposal_surfaces.get(context_id)

        # Group-level competition over exclusion-filtered generator-local maxima
        # is attached after every owner's maxima are known.
        owner_rows: list[dict[str, Any]] = []
        for owner in image_owners:
            if str(owner["normalized_description"]) != description:
                continue
            owner_id = str(owner["gt_owner_id"])
            bank = owner["candidate_bank"]
            reached_ids = [str(value) for value in bank["physical_candidate_ids"]]
            neighbourhood = [
                events_by_candidate[candidate_id]
                for candidate_id in reached_ids
                if candidate_id in events_by_candidate
            ]
            if not neighbourhood:
                continue

            assigned: list[Mapping[str, Any]] = []
            ambiguous: list[Mapping[str, Any]] = []
            other_owner_strict: list[Mapping[str, Any]] = []
            unassigned: list[Mapping[str, Any]] = []
            for row in neighbourhood:
                assignment = row["strict_assignment"]
                status = str(assignment["status"])
                if status == "matched":
                    if str(assignment["gt_owner_id"]) == owner_id:
                        assigned.append(row)
                    else:
                        other_owner_strict.append(row)
                elif status == "ambiguous_neutral":
                    ambiguous.append(row)
                else:
                    unassigned.append(row)

            # The sealed owner-support partition.  ``strict_assigned_self``
            # counts under both bounds; ``ambiguous_upper`` and
            # ``unmatched_generator_local`` count under U only;
            # ``other_owner_strict`` is excluded from both and moves to the
            # collision diagnostic.  Leaving unmatched probes in L would let a
            # perturbation that landed on nothing manufacture a conservative
            # support claim.
            ambiguity_excluded = list(assigned)
            exclusion_filtered = [*assigned, *ambiguous, *unassigned]

            exact_candidate_id = next(
                (
                    str(record.get("physical_candidate_id"))
                    for record in bank["logical_roles"]
                    if str(record["role"]) == "exact_gt_anchor"
                    and record.get("physical_candidate_id")
                ),
                None,
            )
            exact_event = (
                events_by_candidate.get(exact_candidate_id) if exact_candidate_id else None
            )

            owner_rows.append(
                {
                    "schema_version": OWNER_CONTEXT_SCHEMA_VERSION,
                    "row_kind": "census_owner_context",
                    "owner_context_id": f"{owner_id}@{context_id}",
                    "gt_owner_id": owner_id,
                    "image_id": image_id,
                    "split": str(owner["split"]),
                    "context_id": context_id,
                    "boundary_index": int(context["boundary_index"]),
                    "context_role": str(context["context_role"]),
                    "normalized_description": description,
                    "loop_marking": {
                        "loop_tail": bool(context["loop_marking"]["loop_tail"]),
                        "prior_identical_row_count": context["loop_marking"][
                            "prior_identical_row_count"
                        ],
                        "consecutive_identical_row_run_length": context["loop_marking"][
                            "consecutive_identical_row_run_length"
                        ],
                        "flag_is_not_a_mechanism_label": True,
                    },
                    "frontier_features": _frontier_features(context, owner, image_owners),
                    "localization": {
                        "primary_neighbourhood": "generator_local_landscape",
                        "generator_local_max": _attach_geometry(
                            _best(neighbourhood), plan, owner
                        ),
                        "exclusion_filtered_primary_max": _attach_geometry(
                            _best(exclusion_filtered), plan, owner
                        ),
                        "exclusion_filtered_primary_bound": "ambiguity_included_u",
                        "strict_assigned_max": _attach_geometry(
                            _best(assigned), plan, owner
                        ),
                        "exact_anchor_score": (
                            {
                                "value": float(exact_event["complete_box_logprob_sum"]),
                                "candidate_id": str(exact_event["candidate_id"]),
                                "rank": int(exact_event["competition"]["rank"]),
                                "reported": True,
                                "geometry": candidate_owner_geometry(
                                    plan.candidates[str(exact_event["candidate_id"])],
                                    owner_id,
                                ),
                            }
                            if exact_event is not None
                            else {
                                "value": None,
                                "candidate_id": exact_candidate_id,
                                "rank": None,
                                "reported": False,
                            }
                        ),
                        "ambiguous_upper_max": _attach_geometry(
                            _best(ambiguous), plan, owner
                        ),
                        "generator_local_max_excluding_other_owner_strict": {
                            "ambiguity_excluded_l": _attach_geometry(
                                _support_features(ambiguity_excluded), plan, owner
                            ),
                            "ambiguity_included_u": _attach_geometry(
                                _support_features(exclusion_filtered), plan, owner
                            ),
                        },
                        "support_partition": {
                            "strict_assigned_self": "counts_under_both_bounds",
                            "ambiguous_upper": "counts_under_u_only",
                            "unmatched_generator_local": "counts_under_u_only",
                            "other_owner_strict": "excluded_from_both_bounds",
                        },
                        "counts": {
                            "generator_local_event_count": len(neighbourhood),
                            "exclusion_filtered_event_count": len(exclusion_filtered),
                            "ambiguity_excluded_event_count": len(ambiguity_excluded),
                            "strict_assigned_event_count": len(assigned),
                            "ambiguous_event_count": len(ambiguous),
                            "unassigned_event_count": len(unassigned),
                        },
                    },
                    "collision_diagnostic": {
                        "other_owner_strict_event_count": len(other_owner_strict),
                        "other_owner_strict_candidate_ids": sorted(
                            str(row["candidate_id"]) for row in other_owner_strict
                        ),
                        "other_owner_strict_gt_owner_ids": sorted(
                            {
                                str(row["strict_assignment"]["gt_owner_id"])
                                for row in other_owner_strict
                            }
                        ),
                        "role": "excluded_from_target_support_never_evidence_against_target",
                    },
                    "strict_table": {
                        "role": "parallel_owner_identifiable_lower_bound_never_bank_membership",
                        "strict_assigned_max": _best(assigned),
                    },
                    "category_proposal_channel": _proposal_channel(surface, description),
                    "group_population_size": len(group_events),
                }
            )

        _attach_owner_context_competition(owner_rows)
        rows.extend(owner_rows)

    rows.sort(
        key=lambda row: (
            str(row["image_id"]),
            str(row["gt_owner_id"]),
            int(row["boundary_index"]),
        )
    )
    return rows


def _attach_owner_context_competition(owner_rows: list[dict[str, Any]]) -> None:
    """Rank owners in one group by their exclusion-filtered generator-local maximum.

    The strict table is ranked in parallel and never substitutes for the
    primary: the two answer different questions and must not be conflated.
    """

    def _rank(rows: list[dict[str, Any]], path: str, target: str) -> None:
        scored = [row for row in rows if row["localization"][path]["value"] is not None]
        ordered = sorted(
            scored,
            key=lambda row: (-row["localization"][path]["value"], str(row["gt_owner_id"])),
        )
        best = ordered[0]["localization"][path]["value"] if ordered else None
        for position, row in enumerate(ordered):
            value = row["localization"][path]["value"]
            row[target] = {
                "population": "owners_of_this_image_context_category",
                "population_size": len(ordered),
                "rank": position + 1,
                "margin_to_best_owner": value - best,
                "best_gt_owner_id": str(ordered[0]["gt_owner_id"]),
            }
        for row in rows:
            row.setdefault(
                target,
                {
                    "population": "owners_of_this_image_context_category",
                    "population_size": len(ordered),
                    "rank": None,
                    "margin_to_best_owner": None,
                    "best_gt_owner_id": (
                        str(ordered[0]["gt_owner_id"]) if ordered else None
                    ),
                },
            )

    def _rank_bound(rows: list[dict[str, Any]], path: str, target: str) -> None:
        scored = [
            row
            for row in rows
            if row["localization"]["generator_local_max_excluding_other_owner_strict"][path][
                "value"
            ]
            is not None
        ]
        ordered = sorted(
            scored,
            key=lambda row: (
                -row["localization"]["generator_local_max_excluding_other_owner_strict"][path][
                    "value"
                ],
                str(row["gt_owner_id"]),
            ),
        )
        best = (
            ordered[0]["localization"]["generator_local_max_excluding_other_owner_strict"][path][
                "value"
            ]
            if ordered
            else None
        )
        for position, row in enumerate(ordered):
            value = row["localization"]["generator_local_max_excluding_other_owner_strict"][path][
                "value"
            ]
            row[target] = {
                "population": "owners_of_this_image_context_category",
                "population_size": len(ordered),
                "rank": position + 1,
                "margin_to_best_owner": value - best,
                "best_gt_owner_id": str(ordered[0]["gt_owner_id"]),
            }
        for row in rows:
            row.setdefault(
                target,
                {
                    "population": "owners_of_this_image_context_category",
                    "population_size": len(ordered),
                    "rank": None,
                    "margin_to_best_owner": None,
                    "best_gt_owner_id": (str(ordered[0]["gt_owner_id"]) if ordered else None),
                },
            )

    _rank(owner_rows, "exclusion_filtered_primary_max", "owner_competition")
    _rank(owner_rows, "strict_assigned_max", "strict_table_competition")
    _rank_bound(owner_rows, "ambiguity_excluded_l", "owner_competition_l")
    _rank_bound(owner_rows, "ambiguity_included_u", "owner_competition_u")


# ---------------------------------------------------------------------------
# Owner summaries and dispositions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BankAdequacyRule:
    """The lead's pre-score adequacy bands, bound from the sealed capture rules."""

    full_at_least: int
    adequate_at_least: int
    requires_exact_anchor_uniquely_self_assigned: bool
    eligible_statuses: frozenset[str]
    statuses: tuple[str, ...]


def load_bank_adequacy_rule(plan: PlanBundle) -> BankAdequacyRule:
    """Bind the adequacy thresholds from ``capture-rules.json``; never invent them.

    The bands were frozen before any score existed.  This merge reads them and
    refuses to run if they are absent, rather than substituting a number of its
    own -- a merge that could choose its own adequacy cut could choose which
    owners are allowed to close negative.
    """

    support = (plan.capture_rules.get("owner_support") or {}).get("bank_adequacy_rule")
    if not isinstance(support, Mapping):
        _fail(
            "capture-rules.json does not seal owner_support.bank_adequacy_rule; the merge "
            "will not substitute a bank adequacy threshold of its own"
        )
    for key in ("full_at_least", "adequate_at_least", "disposition_eligible_statuses"):
        if key not in support:
            _fail(f"sealed bank_adequacy_rule is missing {key!r}")
    full_at_least = int(support["full_at_least"])
    adequate_at_least = int(support["adequate_at_least"])
    if adequate_at_least > full_at_least:
        _fail("sealed bank_adequacy_rule has adequate_at_least above full_at_least")
    return BankAdequacyRule(
        full_at_least=full_at_least,
        adequate_at_least=adequate_at_least,
        requires_exact_anchor_uniquely_self_assigned=bool(
            support.get("requires_exact_anchor_uniquely_self_assigned", True)
        ),
        eligible_statuses=frozenset(
            str(value) for value in support["disposition_eligible_statuses"]
        ),
        statuses=tuple(str(value) for value in support.get("statuses", ())),
    )


def _bank_adequacy(owner: Mapping[str, Any], *, rule: BankAdequacyRule) -> dict[str, Any]:
    """Adequacy over the alias-collapsed distinct tuple count, exact anchor mandatory.

    ``full`` and ``adequate_reduced`` are both disposition-eligible: a reduced
    bank is flagged, not floored.  Only a bank below the sealed adequate band,
    or one whose exact anchor did not admit and self-localize, forces the
    unresolved-only floor.
    """

    bank = owner["candidate_bank"]
    distinct = int(bank["distinct_physical_candidates_reached"])
    exact_anchor_admitted = bool(bank["exact_anchor_admitted"])
    exact_anchor_self = bool(bank["exact_anchor_uniquely_self_assigned"])
    exact_anchor_ok = exact_anchor_admitted and (
        exact_anchor_self or not rule.requires_exact_anchor_uniquely_self_assigned
    )

    if not exact_anchor_ok or distinct < rule.adequate_at_least:
        status = "undercovered_unresolved_only"
    elif distinct >= rule.full_at_least:
        status = "full"
    else:
        status = "adequate_reduced"

    plan_status = str(bank["generator_local_bank_adequacy"]["status"])
    if plan_status != status:
        _fail(
            f"owner {owner['gt_owner_id']!r} bank adequacy recomputes to {status!r} from the "
            f"sealed thresholds but the plan sealed {plan_status!r}"
        )

    return {
        "measured_on": "distinct_physical_candidate_count_after_alias_collapse",
        "distinct_physical_candidate_count": distinct,
        "uniquely_assigned_candidate_count": int(
            bank["strict_assignment_coverage"]["uniquely_assigned_candidate_count"]
        ),
        "exact_anchor_admitted": exact_anchor_admitted,
        "exact_anchor_uniquely_self_assigned": exact_anchor_self,
        "exact_anchor_mandatory": True,
        "status": status,
        "adequate": status in rule.eligible_statuses,
        "flagged_reduced": status == "adequate_reduced",
        "full_at_least": rule.full_at_least,
        "adequate_at_least": rule.adequate_at_least,
        "threshold_source": "sealed_capture_rules_owner_support_bank_adequacy_rule",
        "strict_assignment_is_bank_membership": False,
    }


def _summarize_bound(
    contexts: Sequence[Mapping[str, Any]],
    *,
    bound: str,
    calibration: SupportCalibration | None,
) -> dict[str, Any]:
    """Best-context summaries for one ambiguity bound.

    ``diagnostic_best_all`` spans every tested context including loop tails and
    is explicitly diagnostic.  The two primary views exclude loop tails: the
    best non-loop context, and the frozen minimal-frontier tie context.
    """

    path = "ambiguity_excluded_l" if bound == "l" else "ambiguity_included_u"

    def _value(row: Mapping[str, Any]) -> float | None:
        return row["localization"]["generator_local_max_excluding_other_owner_strict"][path][
            "value"
        ]

    def _rank(row: Mapping[str, Any]) -> int | None:
        return row["localization"]["generator_local_max_excluding_other_owner_strict"][path][
            "rank"
        ]

    competition_key = "owner_competition_l" if bound == "l" else "owner_competition_u"

    def _project(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """Project one context into the continuous quantities a rule may cut on.

        ``margin_to_best_owner_in_group`` is the *true* competition margin: it
        comes from the owner ranking inside this exact ``(context, category)``
        group, never from a difference taken across two contexts.
        """

        if row is None:
            return None
        frontier = row["frontier_features"]
        competition = row[competition_key]
        proposal = row["category_proposal_channel"]
        gate = proposal.get("boundary_gate") or {}
        routing = proposal.get("category_routing_event") or {}
        bound_block = row["localization"][
            "generator_local_max_excluding_other_owner_strict"
        ][path]
        return {
            "context_id": str(row["context_id"]),
            "boundary_index": int(row["boundary_index"]),
            "loop_tail": bool(row["loop_marking"]["loop_tail"]),
            "value": _value(row),
            # Local-peak support evidence.  These, not the rank, decide support.
            "peak_lift": bound_block["peak_lift"],
            "local_concentration": bound_block["local_concentration"],
            "unique_population_size": bound_block["unique_population_size"],
            "bank_median": bound_block["bank_median"],
            "bank_size": bound_block["bank_size"],
            "geometry": bound_block.get("geometry"),
            # Routing/competition surface: published, never a support input.
            "rank_within_group": _rank(row),
            "attains_group_best": _rank(row) == 1,
            "rank_role": "routing_and_competition_surface_never_a_support_input",
            "owner_rank_within_group": competition["rank"],
            "owner_population_size": competition["population_size"],
            "margin_to_best_owner_in_group": competition["margin_to_best_owner"],
            "margin_semantics": "within_same_context_and_category_owner_competition",
            "exact_anchor_score": row["localization"]["exact_anchor_score"]["value"],
            "strict_assigned_max": row["localization"]["strict_assigned_max"]["value"],
            "signed_frontier_ordinal_distance": frontier["signed_frontier_ordinal_distance"],
            "abs_signed_frontier_ordinal_distance": frontier[
                "abs_signed_frontier_ordinal_distance"
            ],
            "signed_frontier_sort_axis_pixel_distance": frontier[
                "signed_frontier_sort_axis_pixel_distance"
            ],
            "frontier_intersection_over_union": frontier["frontier_overlap"][
                "intersection_over_union"
            ],
            "same_description_owners_ahead_of_frontier": frontier[
                "same_description_owners_ahead_of_frontier"
            ],
            "passed_state": str(frontier["passed_state"]),
            "boundary_gate_continue_vs_stop_logprob_margin": gate.get(
                "continue_vs_stop_logprob_margin"
            ),
            "category_routing_within_context_rank": routing.get("within_context_rank"),
        }

    scored = [row for row in contexts if _value(row) is not None]
    non_loop = [row for row in scored if not row["loop_marking"]["loop_tail"]]
    frontier_tested = [
        row for row in non_loop if row["frontier_features"]["frontier_present"]
    ]

    best_all = max(scored, key=lambda row: (_value(row), -int(row["boundary_index"]))) if scored else None
    best_non_loop = (
        max(non_loop, key=lambda row: (_value(row), -int(row["boundary_index"])))
        if non_loop
        else None
    )

    # Frozen tie rule: among contexts at the minimal absolute signed frontier
    # distance, take the *first non-loop* boundary in ascending boundary order.
    minimal_frontier = None
    if frontier_tested:
        minimum = min(
            int(row["frontier_features"]["abs_signed_frontier_ordinal_distance"])
            for row in frontier_tested
        )
        minimal_frontier = min(
            (
                row
                for row in frontier_tested
                if int(row["frontier_features"]["abs_signed_frontier_ordinal_distance"])
                == minimum
            ),
            key=lambda row: int(row["boundary_index"]),
        )

    def _supported(row: Mapping[str, Any]) -> bool | None:
        if calibration is None:
            return None
        return calibration.clears(
            row["localization"]["generator_local_max_excluding_other_owner_strict"][path]
        )

    support_contexts = [row for row in non_loop if _supported(row)]
    loop_only_support = bool(
        not support_contexts and any(_supported(row) for row in scored)
    )

    return {
        "bound": bound,
        "diagnostic_best_all": _project(best_all),
        "diagnostic_best_all_role": "diagnostic_only_includes_loop_tail_contexts",
        "primary_best_non_loop": _project(best_non_loop),
        "primary_first_non_loop_minimal_abs_frontier": _project(minimal_frontier),
        "minimal_frontier_tie_rule": planner.MINIMAL_FRONTIER_TIE_RULE,
        "tested_context_count": len(scored),
        "non_loop_tested_context_count": len(non_loop),
        "frontier_tested_non_loop_context_count": len(frontier_tested),
        "never_frontier_tested": not frontier_tested,
        "loop_tail_only_support": loop_only_support,
        "usable_support": None if calibration is None else bool(support_contexts),
        "usable_support_context_ids": sorted(str(row["context_id"]) for row in support_contexts),
        "non_loop_context_support": (
            None
            if calibration is None
            else {str(row["context_id"]): bool(_supported(row)) for row in non_loop}
        ),
        "support_criterion_id": SUPPORT_CRITERION_ID,
        "support_calibrated": calibration is not None,
    }


def build_owner_summaries(
    plan: PlanBundle,
    owner_contexts: Sequence[Mapping[str, Any]],
    *,
    captured_image_ids: Sequence[str],
    adequacy_rule: BankAdequacyRule,
    calibration: SupportCalibration | None = None,
) -> list[dict[str, Any]]:
    """Per-owner evidence state, with every protection the contract requires.

    Support is the calibrated local-peak test, never a rank.  A
    ``persistent_no_tested_localization_support`` close is admissible only when
    the owner was frontier-tested, has non-loop primary contexts, has an
    adequate distinct bank, has its exact anchor score reported, and has *no
    usable support in any non-loop context under the optimistic U bound*.
    Anything else closes ``unresolved``.  Without a calibration nothing closes
    at all.  None of these labels is causal.
    """

    captured = set(captured_image_ids)
    by_owner: dict[str, list[Mapping[str, Any]]] = {}
    for row in owner_contexts:
        by_owner.setdefault(str(row["gt_owner_id"]), []).append(row)

    rows: list[dict[str, Any]] = []
    for owner_id in sorted(plan.owners):
        owner = plan.owners[owner_id]
        image_id = str(owner["image_id"])
        if image_id not in captured:
            continue
        contexts = sorted(by_owner.get(owner_id, []), key=lambda row: int(row["boundary_index"]))
        adequacy = _bank_adequacy(owner, rule=adequacy_rule)
        lower = _summarize_bound(contexts, bound="l", calibration=calibration)
        upper = _summarize_bound(contexts, bound="u", calibration=calibration)

        exact_anchor_reported = any(
            row["localization"]["exact_anchor_score"]["reported"] for row in contexts
        )
        frontier_tested = not upper["never_frontier_tested"]
        has_non_loop_primary = upper["non_loop_tested_context_count"] > 0

        blockers: list[str] = []
        if not contexts:
            blockers.append("no_tested_context")
        if not frontier_tested:
            blockers.append("never_frontier_tested")
        if not has_non_loop_primary:
            blockers.append("no_non_loop_primary_context")
        if upper["loop_tail_only_support"] or lower["loop_tail_only_support"]:
            blockers.append("loop_tail_only_support")
        if not adequacy["adequate"]:
            blockers.append("bank_undercoverage")
        if not exact_anchor_reported:
            blockers.append("exact_anchor_score_not_reported")
        # The plan deliberately carries all 346 owners, but the native matching
        # universe excludes the globally ambiguity-neutral ones.  They keep
        # their continuous features and stay visible in the census; they can
        # never carry a negative disposition, because nothing in the native
        # rollout was ever eligible to match them.
        greedy_eligible = bool(owner["greedy_eligible"])
        if not greedy_eligible:
            blockers.append("not_greedy_eligible")
        # ``persistent_no_tested_localization_support`` is a disposition about
        # native *false negatives*.  A native true positive is a positive
        # control: its support outcome is calibration evidence, never an FN
        # finding, and it never enters the FN prevalence denominator.
        native_true_positive = bool(owner["native_true_positive"])
        if native_true_positive:
            blockers.append("native_true_positive_is_a_calibration_control")

        routing = _routing_summary(contexts)

        if calibration is None:
            # Nothing closes without a sealed calibration: the support test is
            # threshold-bearing and its thresholds live in the discovery
            # calibration receipt.
            bound_flip = False
            disposition = DISPOSITION_UNCALIBRATED
        else:
            bound_flip = bool(lower["usable_support"] != upper["usable_support"])
            if not greedy_eligible:
                disposition = DISPOSITION_NOT_ELIGIBLE
            elif native_true_positive:
                disposition = DISPOSITION_TP_CONTROL
            elif bound_flip:
                disposition = DISPOSITION_UNRESOLVED_FLIP
            elif lower["usable_support"] and upper["usable_support"]:
                disposition = DISPOSITION_RESOLVED
            elif blockers:
                disposition = DISPOSITION_UNRESOLVED
            else:
                # No usable support in any non-loop context under the optimistic
                # U bound, and every gate cleared.
                disposition = DISPOSITION_PERSISTENT_NEGATIVE

        rows.append(
            {
                "schema_version": OWNER_SUMMARY_SCHEMA_VERSION,
                "row_kind": "census_owner_summary",
                "gt_owner_id": owner_id,
                "image_id": image_id,
                "split": str(owner["split"]),
                "normalized_description": str(owner["normalized_description"]),
                "greedy_eligible": bool(owner["greedy_eligible"]),
                "native_true_positive": bool(owner["native_true_positive"]),
                "calibration_role": str(owner["calibration_role"]),
                "bank_adequacy": adequacy,
                "bank_report": {
                    "distinct_physical_candidate_count": adequacy[
                        "distinct_physical_candidate_count"
                    ],
                    "uniquely_assigned_candidate_count": adequacy[
                        "uniquely_assigned_candidate_count"
                    ],
                    "other_owner_strict_count": sum(
                        int(row["collision_diagnostic"]["other_owner_strict_event_count"])
                        for row in contexts
                    ),
                    "other_owner_strict_distinct_candidate_count": len(
                        {
                            candidate_id
                            for row in contexts
                            for candidate_id in row["collision_diagnostic"][
                                "other_owner_strict_candidate_ids"
                            ]
                        }
                    ),
                },
                "lower_bound_l": lower,
                "upper_bound_u": upper,
                "exact_anchor_score_reported": exact_anchor_reported,
                "frontier_tested": frontier_tested,
                "has_non_loop_primary_context": has_non_loop_primary,
                "never_frontier_tested": upper["never_frontier_tested"],
                "loop_tail_only_support": upper["loop_tail_only_support"]
                or lower["loop_tail_only_support"],
                "tested_context_count": upper["tested_context_count"],
                "non_loop_tested_context_count": upper["non_loop_tested_context_count"],
                "ambiguity_bound_disposition_flip": bound_flip,
                "routing_summary": routing,
                "disposition_eligible": greedy_eligible and not native_true_positive,
                "in_false_negative_prevalence_denominator": (
                    greedy_eligible and not native_true_positive
                ),
                "cohort": (
                    "native_true_positive_control"
                    if native_true_positive
                    else "native_false_negative"
                    if greedy_eligible
                    else "outside_native_matching_universe"
                ),
                "eligibility_semantics": {
                    "retained_in_census": True,
                    "continuous_features_published": True,
                    "floor_when_not_greedy_eligible": (
                        "unresolved_only_never_persistent_negative_never_a_denominator"
                    ),
                    "native_true_positive_role": (
                        "calibration_positive_control_never_a_false_negative_finding"
                    ),
                },
                "threshold_category_support": (
                    calibration.category_support(str(owner["normalized_description"]))
                    if calibration is not None
                    else None
                ),
                "disposition": disposition,
                "disposition_blockers": sorted(set(blockers)),
                "support_criterion": (
                    calibration.describe()
                    if calibration is not None
                    else {
                        "criterion_id": SUPPORT_CRITERION_ID,
                        "calibrated": False,
                        "note": "no sealed discovery calibration was supplied",
                    }
                ),
                "persistent_negative_preconditions": {
                    "support_calibrated": calibration is not None,
                    "greedy_eligible": greedy_eligible,
                    "is_native_false_negative": not native_true_positive,
                    "frontier_tested": frontier_tested,
                    "non_loop_primary_contexts": has_non_loop_primary,
                    "adequate_distinct_bank": adequacy["adequate"],
                    "exact_anchor_score_reported": exact_anchor_reported,
                    "no_usable_support_in_any_non_loop_context_under_u": (
                        upper["usable_support"] is False
                    ),
                    "optimistic_bound": "u",
                },
                "disposition_semantics": {
                    "is_causal_label": False,
                    "describes": "state_of_the_tested_evidence_never_a_mechanism",
                    "undercovered_owner_floor": (
                        "unresolved_only_never_persistent_no_tested_localization_support"
                    ),
                    "support_is_local_peak_evidence_not_a_rank": True,
                },
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Merge assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergedCensus:
    plan: PlanBundle
    events: list[dict[str, Any]]
    owner_contexts: list[dict[str, Any]]
    owner_summaries: list[dict[str, Any]]
    receipt: dict[str, Any]
    sidecar_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def files(self) -> dict[str, bytes]:
        return {
            EVENTS_NAME: _jsonl_bytes(self.events),
            OWNER_CONTEXT_NAME: _jsonl_bytes(self.owner_contexts),
            OWNER_SUMMARY_NAME: _jsonl_bytes(self.owner_summaries),
            SIDECAR_DIAGNOSTIC_NAME: _jsonl_bytes(self.sidecar_diagnostics),
        }

    def summaries_for_split(self, split: str) -> list[dict[str, Any]]:
        return [row for row in self.owner_summaries if str(row["split"]) == split]


def assert_required_fields(plan: PlanBundle, merged_rows: Sequence[Mapping[str, Any]]) -> None:
    """Prove the published owner-context rows carry every sealed required field.

    The capture rules name the minimum per-owner-context quantities before
    capture; this check keeps the merge from silently dropping one of them.
    """

    support = plan.capture_rules.get("owner_support") or {}
    required = [str(name) for name in support.get("required_per_owner_context_fields", ())]
    if not required or not merged_rows:
        return
    published = set(merged_rows[0]["localization"])
    missing = sorted(name for name in required if name not in published)
    if missing:
        _fail(f"owner-context rows omit sealed required localization fields {missing}")


def merge_census(
    plan: PlanBundle,
    shard_root: Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    allowlist: Sequence[str] | None = None,
    phase: str = PHASE_FULL,
    calibration: SupportCalibration | None = None,
    usable_as_census_conclusion: bool = True,
) -> MergedCensus:
    """Admit the allowlisted shards and publish the merged census views.

    ``allowlist`` is the blind-analysis boundary.  In the discovery phase it is
    exactly :data:`planner.DISCOVERY_IMAGE_IDS`, and the confirmation shard
    files are never opened, so no confirmation feature can be materialized in
    the process that derives the calibration.  When a ``manifest`` is supplied
    the consumed shard digests are additionally proven disjoint from the
    complementary split's digest set, which is the mechanical part of the
    guarantee; the allowlist alone is the operational part.
    """

    if phase not in MERGE_PHASES:
        _fail(f"unknown merge phase {phase!r}")
    if phase == PHASE_SMOKE_ADMIT:
        if list(allowlist or ()) != [SMOKE_IMAGE_ID]:
            _fail(
                f"the smoke-admit phase admits exactly image {SMOKE_IMAGE_ID!r} and nothing else"
            )
        if usable_as_census_conclusion:
            _fail("the smoke-admit phase may never be marked usable as a census conclusion")
    adequacy_rule = load_bank_adequacy_rule(plan)
    support_contract = load_support_contract(plan)
    shards = discover_shards(plan, shard_root, allowlist=allowlist)
    manifest_binding: dict[str, Any] | None = None
    if phase == PHASE_SMOKE_ADMIT:
        # The launch gate exists precisely to run before the other eleven
        # shards are captured, so the all-twelve stop policy cannot apply to it.
        # It buys this by never being conclusion-bearing.
        if manifest is not None:
            _fail("the smoke-admit phase takes no capture manifest")
        quarantine_ledger = {
            "scope": "single_image_launch_gate",
            "global_stop_evaluated": False,
            "reason": "smoke-admit runs before the full capture and concludes nothing",
        }
    elif manifest is not None:
        quarantine_ledger = dict(manifest["quarantine_ledger"])
        # Binding first: prove the bytes are still the sealed bytes, then prove
        # the split boundary was respected.
        manifest_binding = assert_manifest_binding(plan, manifest, shards, phase=phase)
        assert_digest_disjointness(manifest, shards, phase=phase)
    else:
        quarantine_ledger = enforce_stop_policy(plan, shards)
    captured = [row for row in shards if row.status == "captured"]

    require_complete = phase != PHASE_SMOKE_ADMIT
    lineages = [
        validate_shard_lineage(plan, shard, require_complete_shard=require_complete)
        for shard in captured
    ]
    runtime = assert_uniform_runtime(lineages)

    events: list[ScoreEvent] = []
    proposal_surfaces: dict[str, dict[str, Any]] = {}
    admitted_receipt_ids: set[str] = set()
    for shard in captured:
        admissions = build_admission_index(plan, shard)
        admitted_receipt_ids.update(admissions.by_receipt_id)
        events.extend(admit_score_rows(plan, shard, admissions))
        proposal_surfaces.update(admit_proposal_rows(plan, shard, admissions))

    # Duplicate events are rejected shard-locally above; a repeat across shards
    # would mean two images claimed one context, which is also a contract break.
    global_keys: dict[tuple[str, str, tuple[int, ...]], str] = {}
    for event in events:
        key = (event.context_id, event.normalized_description, event.coord_token_ids)
        if key in global_keys:
            _fail(
                "two shards published the same (context, category, coordinate tuple) event "
                f"for context {event.context_id!r}"
            )
        global_keys[key] = event.candidate_id

    sidecars = index_sidecars(plan, captured)
    event_rows = build_event_table(plan, events, sidecars)
    owner_contexts = build_owner_context_features(plan, event_rows, proposal_surfaces)
    assert_required_fields(plan, owner_contexts)
    sidecar_diagnostics = build_sidecar_diagnostics(plan, captured, owner_contexts)
    captured_image_ids = [shard.image_id for shard in captured]
    if phase == PHASE_SMOKE_ADMIT and calibration is not None:
        _fail("the smoke-admit phase is never conclusion-bearing and takes no calibration")
    owner_summaries = build_owner_summaries(
        plan,
        owner_contexts,
        captured_image_ids=captured_image_ids,
        adequacy_rule=adequacy_rule,
        calibration=calibration,
    )

    disposition_counts: dict[str, int] = {}
    for row in owner_summaries:
        disposition_counts[row["disposition"]] = disposition_counts.get(row["disposition"], 0) + 1

    receipt: dict[str, Any] = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "merge_strategy": "twelve_image_owner_accessibility_census_shard_merge",
        "phase": phase,
        "usable_as_census_conclusion": bool(usable_as_census_conclusion),
        "image_allowlist": (None if allowlist is None else sorted(set(allowlist), key=int)),
        "capture_manifest_sha256": (
            None if manifest is None else str(manifest["capture_manifest_sha256"])
        ),
        "capture_manifest_binding": manifest_binding,
        "plan": {
            "plan_dir": str(plan.plan_dir),
            "receipt_content_sha256": plan.receipt["receipt_content_sha256"],
            "capture_rules_sha256": plan.capture_rules["capture_rules_sha256"],
            "plan_schema_version": planner.PLAN_SCHEMA_VERSION,
            "capture_rules_schema_version": planner.CAPTURE_RULES_SCHEMA_VERSION,
        },
        "runtime_identity": runtime,
        "code": {
            "merge_source_sha256": MERGE_SOURCE_SHA256,
            "planner_source_sha256": runtime.get("planner_source_sha256"),
            "scorer_source_sha256": runtime.get("executed_source_sha256"),
            "role": (
                "interpreted_scores_and_applied_calibration_provenance_only"
            ),
            "is_support_input": False,
        },
        "shard_lineage": lineages,
        "quarantine_ledger": quarantine_ledger,
        "admission": {
            "admission_key": ["context_id", "channel", "exact_prefix_sha256"],
            "channels": list(planner.ADMISSION_CHANNELS),
            "admission_receipt_count": len(admitted_receipt_ids),
            "every_score_row_covered": True,
            "query_suffix_admission_covers_proposal_channel": False,
            "reuse_admission_across_contexts": False,
        },
        "quarantined_score_semantics": {
            "pre_p0_rows": "rejected_missing_query_suffix_digest",
            "reads_quarantined_shard_evidence": False,
        },
        "sidecar_contract": {
            "surface": SIDECAR_DIAGNOSTIC_NAME,
            "role": "continuous_finite_bank_undercoverage_diagnostic",
            "enters_core_ranks": False,
            "enters_support_test": False,
            "undercoverage_threshold_applied": False,
        },
        "rank_contract": {
            "rank_keys": ["image_id", "context_id", "normalized_description"],
            "population": "collapsed_unique_physical_candidates_only",
            "sidecars_excluded": True,
            "duplicate_event_policy": "fail_closed",
            "posterior": "renormalized_within_group_never_a_model_probability",
        },
        "owner_evidence_contract": {
            "primary_neighbourhood": "generator_local_landscape",
            "adequacy_gate": "distinct_physical_candidate_count_after_alias_collapse",
            "exact_anchor_mandatory": True,
            "strict_assignment_role": (
                "separate_owner_identifiable_lower_bound_never_bank_membership"
            ),
            "other_owner_strict_candidates": (
                "excluded_from_target_support_moved_to_collision_diagnostic"
            ),
            "ambiguous_candidates": "count_toward_target_only_through_the_u_bound",
            "l_u_disposition_flip": "unresolved",
            "undercovered_owner_floor": (
                "unresolved_only_never_persistent_no_tested_localization_support"
            ),
            "bank_adequacy_rule": {
                "full_at_least": adequacy_rule.full_at_least,
                "adequate_at_least": adequacy_rule.adequate_at_least,
                "requires_exact_anchor_uniquely_self_assigned": (
                    adequacy_rule.requires_exact_anchor_uniquely_self_assigned
                ),
                "disposition_eligible_statuses": sorted(adequacy_rule.eligible_statuses),
                "threshold_source": (
                    "sealed_capture_rules_owner_support_bank_adequacy_rule"
                ),
                "adequate_reduced_is_eligible_and_flagged": (
                    "adequate_reduced" in adequacy_rule.eligible_statuses
                ),
            },
        },
        "split": {
            "discovery_image_ids": list(planner.DISCOVERY_IMAGE_IDS),
            "confirmation_image_ids": list(planner.CONFIRMATION_IMAGE_IDS),
            "tuning_policy": "confirmation_rules_are_frozen_on_discovery_only",
        },
        "counts": {
            "captured_image_count": len(captured),
            "event_count": len(event_rows),
            "owner_context_row_count": len(owner_contexts),
            "owner_summary_row_count": len(owner_summaries),
            "sidecar_diagnostic_row_count": len(sidecar_diagnostics),
            "out_of_bank_free_box_count": sum(
                1
                for row in sidecar_diagnostics
                if row["sidecar_kind"] == "free_greedy_box"
                and row.get("inside_fixed_bank") is False
            ),
            "proposal_surface_count": len(proposal_surfaces),
            "discovery_owner_summary_count": sum(
                1 for row in owner_summaries if row["split"] == "discovery"
            ),
            "confirmation_owner_summary_count": sum(
                1 for row in owner_summaries if row["split"] == "confirmation"
            ),
            "disposition_counts": dict(sorted(disposition_counts.items())),
        },
        "causal_labels": {
            "emitted": False,
            "note": "every published label is an evidence state, never a mechanism",
        },
        "support_criterion": (
            calibration.describe()
            if calibration is not None
            else {
                "criterion_id": SUPPORT_CRITERION_ID,
                "calibrated": False,
                "note": (
                    "continuous features published; no support test applied and no "
                    "disposition closed without a sealed discovery calibration"
                ),
            }
        ),
        "support_semantics": {
            "criterion_id": SUPPORT_CRITERION_ID,
            "inputs": list(support_contract.statistics),
            "rank_is_not_a_support_input": True,
            "rank_and_margin_role": (
                "published as routing/competition features, never a support input"
            ),
            "epsilon": support_contract.support_epsilon,
            "cross_context_delta_epsilon": support_contract.cross_context_delta_epsilon,
            "primary_quantile": support_contract.primary_quantile,
            "constants_source": "sealed_capture_rules_owner_support",
            "observed_drift_role": "compliance_signal_only_never_widens_epsilon",
        },
        "dispositions_closed": calibration is not None,
        "output_file_digests": {},
    }

    merged = MergedCensus(
        plan=plan,
        events=event_rows,
        owner_contexts=owner_contexts,
        owner_summaries=owner_summaries,
        receipt=receipt,
        sidecar_diagnostics=sidecar_diagnostics,
    )
    receipt["output_file_digests"] = {
        name: sha256_bytes(content) for name, content in sorted(merged.files().items())
    }
    receipt["receipt_content_sha256"] = sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    return merged


def commit_merge(merged: MergedCensus, output_dir: Path) -> dict[str, str]:
    """Create-or-identical commit: an existing merge may only be reused byte-for-byte."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = dict(merged.files())
    files[MERGE_RECEIPT_NAME] = canonical_json_bytes(merged.receipt) + b"\n"
    written: dict[str, str] = {}
    for name, content in sorted(files.items()):
        path = output_dir / name
        if path.exists() and path.read_bytes() != content:
            _fail(f"refusing to overwrite existing merge file {name!r} with different bytes")
        path.write_bytes(content)
        written[name] = sha256_bytes(content)
    return written


# ---------------------------------------------------------------------------
# Two-phase discovery / confirmation API
# ---------------------------------------------------------------------------


def due_context_index(plan: PlanBundle) -> dict[str, dict[str, Any]]:
    """Map each native true-positive owner to its deterministic due context.

    The owner's ``native_strict_match_pred_row_ids`` names the emitted row that
    matched it; the native sidecar registry maps that ``pred_row_id`` to its
    ``row_index``; the due (pre-boundary) context is the boundary *before* that
    row, ``<image>:boundary-{row_index:03d}``.

    This is target-blind and score-blind by construction: the mapping is a
    property of the frozen native rollout, never of any score or frontier
    geometry.  An owner without exactly one strict match, or whose due context
    is not in the plan, is recorded as an explicit calibration exclusion rather
    than silently resolved some other way.
    """

    row_index_by_pred: dict[str, int] = {}
    for row in plan.native_sidecars:
        pred_row_id = row.get("pred_row_id")
        if pred_row_id is None or "row_index" not in row:
            continue
        pred_row_id = str(pred_row_id)
        if pred_row_id in row_index_by_pred:
            _fail(f"native sidecar registry maps {pred_row_id!r} to two row indices")
        row_index_by_pred[pred_row_id] = int(row["row_index"])

    mapping: dict[str, dict[str, Any]] = {}
    for owner_id, owner in plan.owners.items():
        if not owner.get("native_true_positive"):
            continue
        matches = [str(value) for value in owner.get("native_strict_match_pred_row_ids", ())]
        if len(matches) != 1:
            mapping[owner_id] = {
                "gt_owner_id": owner_id,
                "excluded": True,
                "reason": (
                    "no_unique_native_strict_match"
                    if len(matches) != 1
                    else "unmapped"
                ),
                "match_count": len(matches),
            }
            continue
        pred_row_id = matches[0]
        if pred_row_id not in row_index_by_pred:
            mapping[owner_id] = {
                "gt_owner_id": owner_id,
                "excluded": True,
                "reason": "pred_row_id_absent_from_native_sidecar_registry",
                "pred_row_id": pred_row_id,
            }
            continue
        row_index = row_index_by_pred[pred_row_id]
        context_id = f"{owner['image_id']}:boundary-{row_index:03d}"
        if context_id not in plan.contexts:
            mapping[owner_id] = {
                "gt_owner_id": owner_id,
                "excluded": True,
                "reason": "due_context_absent_from_plan",
                "due_context_id": context_id,
            }
            continue
        mapping[owner_id] = {
            "gt_owner_id": owner_id,
            "excluded": False,
            "pred_row_id": pred_row_id,
            "row_index": row_index,
            "due_context_id": context_id,
        }
    return mapping


def calibrate_support(
    merged: MergedCensus, manifest: Mapping[str, Any]
) -> SupportCalibration:
    """Phase B: derive the TP-calibrated support thresholds on discovery only.

    Observations are the discovery-half native true positives evaluated at their
    deterministic due context, excluding loop tails.  For each observation the
    calibrated statistic is the *minimum across the two ambiguity bounds*,
    because the support test requires both bounds to clear.  This fixes each
    statistic's marginal quantile; the support test is a conjunction, so the
    joint calibration failure rate is at least the nominal quantile and is
    measured and reported rather than assumed.

    The primary calibration is pooled at ``q=0.10``.  Other quantiles and the
    per-category stratification are recorded as report-only sensitivity and
    never move a threshold.
    """

    if merged.receipt.get("phase") != PHASE_DISCOVERY:
        _fail(
            "support calibration must be derived from a discovery-phase merge, so the "
            "confirmation shard files were never opened"
        )
    consumed = sorted(
        {
            digest
            for lineage in merged.receipt["shard_lineage"]
            for digest in lineage["published_file_digests"].values()
        }
    )
    forbidden = set(manifest["confirmation_shard_digests"])
    if set(consumed) & forbidden:
        _fail("calibration consumed a confirmation shard digest; refusing to seal it")

    due = due_context_index(merged.plan)
    contexts_by_owner: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in merged.owner_contexts:
        contexts_by_owner.setdefault(str(row["gt_owner_id"]), {})[str(row["context_id"])] = row

    observations: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for summary in merged.summaries_for_split("discovery"):
        owner_id = str(summary["gt_owner_id"])
        if not summary["native_true_positive"]:
            continue
        entry = due.get(owner_id)
        if entry is None or entry["excluded"]:
            exclusions.append(entry or {"gt_owner_id": owner_id, "reason": "no_due_mapping"})
            continue
        row = contexts_by_owner.get(owner_id, {}).get(str(entry["due_context_id"]))
        if row is None:
            exclusions.append(
                {**entry, "excluded": True, "reason": "owner_not_tested_at_its_due_context"}
            )
            continue
        if row["loop_marking"]["loop_tail"]:
            exclusions.append({**entry, "excluded": True, "reason": "due_context_is_a_loop_tail"})
            continue
        bounds = row["localization"]["generator_local_max_excluding_other_owner_strict"]
        lifts = [bounds[key]["peak_lift"] for key in ("ambiguity_excluded_l", "ambiguity_included_u")]
        concentrations = [
            bounds[key]["local_concentration"]
            for key in ("ambiguity_excluded_l", "ambiguity_included_u")
        ]
        if any(value is None for value in (*lifts, *concentrations)):
            exclusions.append(
                {**entry, "excluded": True, "reason": "no_observation_under_one_of_the_bounds"}
            )
            continue
        observations.append(
            {
                "gt_owner_id": owner_id,
                "image_id": str(summary["image_id"]),
                "normalized_description": str(summary["normalized_description"]),
                "due_context_id": str(entry["due_context_id"]),
                "peak_lift_min_over_bounds": min(float(value) for value in lifts),
                "local_concentration_min_over_bounds": min(
                    float(value) for value in concentrations
                ),
            }
        )

    if not observations:
        _fail(
            "no discovery native true positive produced a due-context observation under "
            "both ambiguity bounds; the support thresholds cannot be calibrated"
        )

    lifts = [row["peak_lift_min_over_bounds"] for row in observations]
    concentrations = [row["local_concentration_min_over_bounds"] for row in observations]

    per_category: dict[str, int] = {}
    for row in observations:
        key = str(row["normalized_description"])
        per_category[key] = per_category.get(key, 0) + 1

    contract = load_support_contract(merged.plan)
    sensitivity = {
        "role": "report_only_never_moves_a_threshold",
        "quantiles": {
            str(level): {
                "theta_peak_lift": _quantile(lifts, level),
                "theta_local_concentration": _quantile(concentrations, level),
            }
            for level in contract.sensitivity_quantiles
        },
        "per_category": {
            description: {
                "observation_count": count,
                "meets_category_contribution_min": (
                    count >= contract.category_contribution_min
                ),
                "theta_peak_lift": _quantile(
                    [
                        row["peak_lift_min_over_bounds"]
                        for row in observations
                        if str(row["normalized_description"]) == description
                    ],
                    contract.primary_quantile,
                ),
                "theta_local_concentration": _quantile(
                    [
                        row["local_concentration_min_over_bounds"]
                        for row in observations
                        if str(row["normalized_description"]) == description
                    ],
                    contract.primary_quantile,
                ),
            }
            for description, count in sorted(per_category.items())
        },
    }

    return SupportCalibration(
        theta_peak_lift=_quantile(lifts, contract.primary_quantile),
        theta_local_concentration=_quantile(concentrations, contract.primary_quantile),
        epsilon=contract.support_epsilon,
        quantile=contract.primary_quantile,
        observation_count=len(observations),
        per_category_counts=per_category,
        sensitivity=sensitivity,
        exclusions=tuple(exclusions),
        consumed_shard_digests=tuple(consumed),
        capture_manifest_sha256=str(manifest["capture_manifest_sha256"]),
        category_contribution_min=contract.category_contribution_min,
        underrepresented_flag=contract.underrepresented_flag,
        cross_context_delta_epsilon=contract.cross_context_delta_epsilon,
        statistics=contract.statistics,
    )


def owner_feature_vector(
    summary: Mapping[str, Any], *, bound: str, view: str = "primary_best_non_loop"
) -> dict[str, Any] | None:
    """The continuous quantities this merge publishes for one owner.

    This is a *vocabulary*, not a decision.  The merge names and measures the
    features; which of them a phenotype rule cuts on, in which direction, and at
    which threshold is decided afterwards by inspecting the discovery
    distributions, and is supplied back as an explicit sealed rule.
    """

    if bound not in {"l", "u"}:
        _fail(f"unknown ambiguity bound {bound!r}")
    if view not in OWNER_FEATURE_VIEWS:
        _fail(f"unknown owner feature view {view!r}")
    block = summary["lower_bound_l" if bound == "l" else "upper_bound_u"]
    projected = block[view]
    if projected is None or projected["value"] is None:
        return None
    features: dict[str, Any] = {
        name: projected[name] for name in OWNER_CONTEXT_FEATURE_NAMES if name in projected
    }
    features["primary_exclusion_filtered_max"] = projected["value"]
    features["distinct_physical_candidate_count"] = summary["bank_adequacy"][
        "distinct_physical_candidate_count"
    ]
    features["uniquely_assigned_candidate_count"] = summary["bank_adequacy"][
        "uniquely_assigned_candidate_count"
    ]
    features["non_loop_tested_context_count"] = summary["non_loop_tested_context_count"]

    # Cross-context deltas are named for what they are.  They are never a
    # competition margin: a competition margin only exists inside one
    # (context, category) rank population.
    minimal = block["primary_first_non_loop_minimal_abs_frontier"]
    features["cross_context_delta_primary_best_vs_minimal_frontier"] = (
        float(projected["value"]) - float(minimal["value"])
        if minimal is not None and minimal["value"] is not None
        else None
    )
    return features


def _evidence_states(summary: Mapping[str, Any]) -> dict[str, Any]:
    """The boolean evidence states a rule may stratify on (never cut on)."""

    return {
        "native_true_positive": bool(summary["native_true_positive"]),
        "greedy_eligible": bool(summary["greedy_eligible"]),
        "frontier_tested": bool(summary["frontier_tested"]),
        "loop_tail_only_support": bool(summary["loop_tail_only_support"]),
        "bank_adequacy_status": str(summary["bank_adequacy"]["status"]),
        "disposition": str(summary["disposition"]),
    }


def describe_discovery_distributions(
    merged: MergedCensus,
    *,
    bound: str = "u",
    features: Sequence[str] | None = None,
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> dict[str, Any]:
    """Diagnostic distribution summary of the discovery half.  Never a rule.

    This exists so the lead can *look* at the discovery distributions, including
    the native true-positive calibration stratum, before writing a rule.  It
    emits no threshold, no cut, and no phenotype, and its output cannot be
    passed to :func:`apply_confirmation`.
    """

    wanted = list(features or OWNER_FEATURE_CATALOG)
    unknown = sorted(set(wanted) - set(OWNER_FEATURE_CATALOG))
    if unknown:
        _fail(f"unknown owner features {unknown}")

    strata: dict[str, list[Mapping[str, Any]]] = {"all": [], "native_true_positive": [], "native_false_negative": []}
    for summary in merged.summaries_for_split("discovery"):
        vector = owner_feature_vector(summary, bound=bound)
        if vector is None:
            continue
        strata["all"].append(vector)
        key = (
            "native_true_positive"
            if summary["native_true_positive"]
            else "native_false_negative"
        )
        strata[key].append(vector)

    described: dict[str, Any] = {}
    for stratum, vectors in strata.items():
        per_feature: dict[str, Any] = {}
        for name in wanted:
            values = [
                float(vector[name])
                for vector in vectors
                if vector.get(name) is not None
            ]
            per_feature[name] = {
                "count": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "quantiles": (
                    {str(level): _quantile(values, level) for level in quantiles}
                    if values
                    else None
                ),
            }
        described[stratum] = {"owner_count": len(vectors), "features": per_feature}

    return {
        "schema_version": DISCOVERY_DISTRIBUTION_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "split": "discovery",
        "bound": bound,
        "strata": described,
        "calibration_stratum": "native_true_positive_owners_are_the_calibration_reference",
        "is_a_rule": False,
        "decision_bearing": False,
        "role": "diagnostic_suggestion_only_never_a_phenotype_cut",
        "merge_receipt_content_sha256": merged.receipt["receipt_content_sha256"],
    }


def seal_discovery_rule(spec: Mapping[str, Any], merged: MergedCensus) -> dict[str, Any]:
    """Validate and content-address an *explicit* discovery rule.

    The rule is authored after inspecting the discovery distributions; this
    function derives no threshold of its own.  It checks that every cut names a
    feature this merge actually publishes, a direction, and a finite threshold;
    that the rule declares its provenance and disclaims a causal reading; that
    it was sealed against the frozen discovery half; and then seals it.
    """

    conditions = spec.get("conditions")
    if not isinstance(conditions, Sequence) or isinstance(conditions, (str, bytes)):
        _fail("a discovery rule must carry an explicit list of conditions")
    if not conditions:
        _fail("a discovery rule must carry at least one condition")

    normalized: list[dict[str, Any]] = []
    for index, condition in enumerate(conditions):
        label = f"discovery rule condition {index}"
        if not isinstance(condition, Mapping):
            _fail(f"{label} is not a JSON object")
        feature = str(_require(condition, "feature", label))
        if feature not in OWNER_FEATURE_CATALOG:
            _fail(
                f"{label} cuts on {feature!r}, which this merge does not publish; "
                f"admissible features are {sorted(OWNER_FEATURE_CATALOG)}"
            )
        direction = str(_require(condition, "direction", label))
        if direction not in RULE_DIRECTIONS:
            _fail(f"{label} has direction {direction!r}; expected one of {list(RULE_DIRECTIONS)}")
        threshold = _finite(_require(condition, "threshold", label), f"{label} threshold")
        normalized.append(
            {
                "feature": feature,
                "direction": direction,
                "threshold": threshold,
                "feature_definition": OWNER_FEATURE_CATALOG[feature],
                "rationale": str(condition.get("rationale", "")),
            }
        )

    bound = str(spec.get("bound", "u"))
    if bound not in {"l", "u"}:
        _fail(f"a discovery rule must name ambiguity bound 'l' or 'u', not {bound!r}")
    view = str(spec.get("view", "primary_best_non_loop"))
    if view not in OWNER_FEATURE_VIEWS:
        _fail(f"a discovery rule must name a published owner view, not {view!r}")

    stratum = spec.get("stratum") or {}
    if not isinstance(stratum, Mapping):
        _fail("a discovery rule stratum must be a JSON object")
    for key in stratum:
        if str(key) not in EVIDENCE_STATE_NAMES:
            _fail(
                f"a discovery rule may only stratify on published evidence states; "
                f"{key!r} is not one of {sorted(EVIDENCE_STATE_NAMES)}"
            )

    provenance = spec.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance.get("constructed_after_inspecting"):
        _fail(
            "a discovery rule must record its provenance, naming what discovery evidence "
            "was inspected before the cuts were chosen"
        )
    if provenance.get("causal_claim_asserted") is not False:
        _fail(
            "a discovery rule must explicitly disclaim a causal claim "
            "(provenance.causal_claim_asserted must be false)"
        )

    label_hold = str(_require(spec, "phenotype_when_all_conditions_hold", "discovery rule"))
    label_otherwise = str(_require(spec, "phenotype_otherwise", "discovery rule"))

    summaries = merged.summaries_for_split("discovery")
    used_image_ids = sorted({str(row["image_id"]) for row in summaries}, key=int)
    frozen = list(planner.DISCOVERY_IMAGE_IDS)
    leaked = sorted(set(used_image_ids) - set(frozen), key=int)
    if leaked:
        _fail(f"discovery rule was sealed against non-discovery images {leaked}")

    rule: dict[str, Any] = {
        "schema_version": DISCOVERY_RULE_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "phase": "discovery",
        "rule_id": str(spec.get("rule_id", "")),
        "authored": "explicit_post_discovery_never_auto_derived",
        "bound": bound,
        "view": view,
        "stratum": {str(key): value for key, value in stratum.items()},
        "conditions": normalized,
        "phenotype_when_all_conditions_hold": label_hold,
        "phenotype_otherwise": label_otherwise,
        "phenotype_labels_are_causal": False,
        "provenance": dict(provenance),
        "discovery_image_ids_frozen": frozen,
        "discovery_image_ids_used": used_image_ids,
        "confirmation_image_ids_frozen": list(planner.CONFIRMATION_IMAGE_IDS),
        "confirmation_evidence_consumed": False,
        "discovery_owner_count": len(summaries),
        "merge_receipt_content_sha256": merged.receipt["receipt_content_sha256"],
        "retuning_after_sealing": "forbidden",
    }
    rule["discovery_rule_sha256"] = sha256_json(
        {key: value for key, value in rule.items() if key != "discovery_rule_sha256"}
    )
    return rule


def _in_stratum(rule: Mapping[str, Any], summary: Mapping[str, Any]) -> bool:
    states = _evidence_states(summary)
    return all(states.get(str(key)) == value for key, value in (rule.get("stratum") or {}).items())


def _apply_rule(rule: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    """Mechanically evaluate one sealed rule against one owner summary."""

    base = {
        "gt_owner_id": str(summary["gt_owner_id"]),
        "image_id": str(summary["image_id"]),
        "split": str(summary["split"]),
        "evidence_states": _evidence_states(summary),
    }
    if not _in_stratum(rule, summary):
        return {**base, "features": None, "phenotype": "outside_rule_stratum", "conditions": []}

    features = owner_feature_vector(
        summary, bound=str(rule["bound"]), view=str(rule["view"])
    )
    if features is None:
        return {
            **base,
            "features": None,
            "phenotype": "not_evaluable_no_published_observation_for_this_view",
            "conditions": [],
        }

    outcomes: list[dict[str, Any]] = []
    for condition in rule["conditions"]:
        value = features.get(str(condition["feature"]))
        if value is None:
            outcomes.append({**condition, "value": None, "holds": None})
            continue
        holds = (
            float(value) >= float(condition["threshold"])
            if condition["direction"] == "at_least"
            else float(value) <= float(condition["threshold"])
        )
        outcomes.append({**condition, "value": float(value), "holds": bool(holds)})

    if any(row["holds"] is None for row in outcomes):
        phenotype = "not_evaluable_missing_feature_value"
    elif all(row["holds"] for row in outcomes):
        phenotype = str(rule["phenotype_when_all_conditions_hold"])
    else:
        phenotype = str(rule["phenotype_otherwise"])
    return {**base, "features": features, "phenotype": phenotype, "conditions": outcomes}


def apply_confirmation(
    merged: MergedCensus,
    rule: Mapping[str, Any],
    *,
    expect_discovery_rule_sha256: str,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply a sealed discovery rule to the held-out confirmation half.

    Refuses to run unless the caller names the exact expected rule digest and
    the rule reconstructs it.  There is no retuning seam: passing
    ``overrides`` is a hard error rather than a supported knob, because a
    confirmation that may adjust its own cut is not a confirmation.
    """

    if overrides:
        _fail(
            "confirmation may not retune the sealed discovery rule; overrides are refused "
            "by contract"
        )
    if rule.get("schema_version") != DISCOVERY_RULE_SCHEMA_VERSION:
        _fail("discovery rule has an unexpected schema_version")
    if rule.get("unit_id") != UNIT_ID:
        _fail("discovery rule belongs to another unit")
    if rule.get("phase") != "discovery":
        _fail("confirmation requires a rule sealed in the discovery phase")
    reconstructed = sha256_json(
        {key: value for key, value in rule.items() if key != "discovery_rule_sha256"}
    )
    if reconstructed != rule.get("discovery_rule_sha256"):
        _fail("discovery rule does not reconstruct its own digest; it was edited after sealing")
    if not expect_discovery_rule_sha256:
        _fail("confirmation requires the expected discovery rule digest to be named explicitly")
    if expect_discovery_rule_sha256 != rule["discovery_rule_sha256"]:
        _fail(
            "confirmation was asked to bind discovery rule "
            f"{expect_discovery_rule_sha256!r} but was given "
            f"{rule['discovery_rule_sha256']!r}"
        )
    unknown = sorted(
        {
            str(condition.get("feature"))
            for condition in rule.get("conditions") or ()
            if str(condition.get("feature")) not in OWNER_FEATURE_CATALOG
        }
    )
    if unknown:
        _fail(f"discovery rule cuts on features this merge does not publish: {unknown}")
    if list(rule.get("discovery_image_ids_frozen") or ()) != list(planner.DISCOVERY_IMAGE_IDS):
        _fail("discovery rule was sealed against a different frozen discovery image list")

    summaries = merged.summaries_for_split("confirmation")
    leaked = sorted(
        {str(row["image_id"]) for row in summaries} & set(planner.DISCOVERY_IMAGE_IDS), key=int
    )
    if leaked:
        _fail(f"confirmation evaluation saw discovery images {leaked}")

    applied = [_apply_rule(rule, summary) for summary in summaries]
    counts: dict[str, int] = {}
    for row in applied:
        counts[row["phenotype"]] = counts.get(row["phenotype"], 0) + 1

    report: dict[str, Any] = {
        "schema_version": CONFIRMATION_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "phase": "confirmation",
        "bound_discovery_rule_sha256": rule["discovery_rule_sha256"],
        "rule_retuned": False,
        "conditions_applied": [dict(condition) for condition in rule["conditions"]],
        "stratum_applied": dict(rule.get("stratum") or {}),
        "bound": str(rule["bound"]),
        "view": str(rule["view"]),
        "confirmation_image_ids_frozen": list(planner.CONFIRMATION_IMAGE_IDS),
        "confirmation_image_ids_used": sorted(
            {str(row["image_id"]) for row in summaries}, key=int
        ),
        "discovery_images_seen": [],
        "owner_count": len(summaries),
        "phenotype_counts": dict(sorted(counts.items())),
        "phenotype_labels_are_causal": False,
        "assignments": applied,
        "merge_receipt_content_sha256": merged.receipt["receipt_content_sha256"],
    }
    report["confirmation_report_sha256"] = sha256_json(
        {key: value for key, value in report.items() if key != "confirmation_report_sha256"}
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def summarize(merged: MergedCensus) -> str:
    """Human-readable summary, rendered against the phase's own ledger schema.

    The smoke-admit ledger is deliberately *not* the full-census ledger: the
    launch gate runs before the other eleven shards exist, so there is no
    census-wide quarantined/missing/unusable aggregate to report.  Rendering it
    through the full-census template would either crash or, worse, print
    invented zeros that read as "nothing was quarantined" for a capture that
    never happened.
    """

    counts = merged.receipt["counts"]
    ledger = merged.receipt["quarantine_ledger"]
    phase = str(merged.receipt.get("phase", PHASE_FULL))
    lines = [
        f"unit: {UNIT_ID}",
        f"schema: {MERGE_SCHEMA_VERSION}",
        f"phase: {phase}",
        "",
    ]

    if phase == PHASE_SMOKE_ADMIT:
        lines += [
            "single-image launch gate -- NOT a census conclusion",
            f"  usable as conclusion     {merged.receipt['usable_as_census_conclusion']}",
            f"  images                   {merged.receipt['image_allowlist']}",
            f"  ledger scope             {ledger.get('scope')}",
            f"  global stop evaluated    {ledger.get('global_stop_evaluated')}",
            f"  reason                   {ledger.get('reason')}",
        ]
    else:
        lines += [
            f"captured images            {counts['captured_image_count']}",
            f"  quarantined              {ledger['quarantined_image_count']}"
            f" {ledger['quarantined_image_ids']}",
            f"  missing                  {ledger['missing_image_count']}"
            f" {ledger['missing_image_ids']}",
            f"  incomplete               {ledger['incomplete_image_count']}"
            f" {ledger['incomplete_image_ids']}",
        ]

    lines += [
        f"events                     {counts['event_count']}",
        f"owner-context rows         {counts['owner_context_row_count']}",
        f"owner summaries            {counts['owner_summary_row_count']}"
        f" (discovery {counts['discovery_owner_summary_count']}"
        f" / confirmation {counts['confirmation_owner_summary_count']})",
        f"sidecar diagnostics        {counts['sidecar_diagnostic_row_count']}",
        f"dispositions (closed: {merged.receipt['dispositions_closed']})",
    ]
    for name, value in counts["disposition_counts"].items():
        lines.append(f"  {name:<52} {value}")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--phase",
        choices=(
            PHASE_CAPTURE_MANIFEST,
            PHASE_SMOKE_ADMIT,
            PHASE_DISCOVERY,
            PHASE_CONFIRMATION,
            PHASE_PRESENTATION,
        ),
        default=PHASE_CAPTURE_MANIFEST,
        help=(
            "capture-manifest: content-address all twelve shards and freeze the "
            "quarantine ledger, reading no score row; smoke-admit: the single-image "
            "launch gate, never a census conclusion; discovery: merge the discovery "
            "allowlist and seal the support calibration; confirmation: bind that exact "
            "calibration and apply it to the held-out half; presentation: the combined "
            "view, only once a confirmation report exists"
        ),
    )
    parser.add_argument(
        "--capture-manifest",
        type=Path,
        default=None,
        help="sealed capture-manifest.json (required by every post-manifest phase)",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="sealed support-calibration.json to bind (confirmation phase)",
    )
    parser.add_argument(
        "--expect-calibration-sha256",
        default=None,
        help="the exact support calibration digest confirmation must bind",
    )
    parser.add_argument(
        "--rule-spec",
        type=Path,
        default=None,
        help="explicit authored discovery rule specification to seal",
    )
    parser.add_argument(
        "--discovery-rule",
        type=Path,
        default=None,
        help="sealed discovery-rule.json to bind (confirmation phase)",
    )
    parser.add_argument(
        "--expect-discovery-rule-sha256",
        default=None,
        help="the exact discovery rule digest confirmation must bind",
    )
    return parser.parse_args(argv)


def _require_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.capture_manifest is None:
        _fail(f"the {args.phase} phase requires --capture-manifest")
    manifest = _read_json(args.capture_manifest, "capture manifest")
    if manifest.get("schema_version") != CAPTURE_MANIFEST_SCHEMA_VERSION:
        _fail("capture manifest has an unexpected schema_version")
    recomputed = sha256_json(
        {key: value for key, value in manifest.items() if key != "capture_manifest_sha256"}
    )
    if recomputed != manifest.get("capture_manifest_sha256"):
        _fail("capture manifest does not reconstruct its own digest")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        plan = load_plan(args.plan_dir)
        extra: dict[str, bytes] = {}
        merged: MergedCensus | None = None

        if args.phase == PHASE_CAPTURE_MANIFEST:
            manifest = build_capture_manifest(plan, args.shard_root)
            extra[CAPTURE_MANIFEST_NAME] = canonical_json_bytes(manifest) + b"\n"
            print(
                f"capture manifest sealed: {manifest['capture_manifest_sha256']} "
                f"({manifest['quarantine_ledger']['unusable_image_count']} unusable image(s); "
                "no score row was read)"
            )
        elif args.phase == PHASE_SMOKE_ADMIT:
            if args.capture_manifest is not None:
                _fail(
                    "smoke-admit runs before the full capture exists and therefore takes "
                    "no --capture-manifest"
                )
            merged = merge_census(
                plan,
                args.shard_root,
                allowlist=_allowlist_for_phase(PHASE_SMOKE_ADMIT),
                phase=PHASE_SMOKE_ADMIT,
                usable_as_census_conclusion=False,
            )
            print(
                f"smoke admit for image {SMOKE_IMAGE_ID}: launch gate only, "
                "usable_as_census_conclusion=false"
            )
        elif args.phase == PHASE_DISCOVERY:
            manifest = _require_manifest(args)
            merged = merge_census(
                plan,
                args.shard_root,
                manifest=manifest,
                allowlist=_allowlist_for_phase(PHASE_DISCOVERY),
                phase=PHASE_DISCOVERY,
            )
            calibration = calibrate_support(merged, manifest)
            receipt = calibration.describe()
            # Re-apply the sealed calibration to the discovery half, so the
            # committed discovery artifacts carry calibrated support states
            # rather than the uncalibrated pass the calibration was derived
            # from.  Still the discovery allowlist only: no confirmation shard
            # file is opened by this second pass either.
            merged = merge_census(
                plan,
                args.shard_root,
                manifest=manifest,
                allowlist=_allowlist_for_phase(PHASE_DISCOVERY),
                phase=PHASE_DISCOVERY,
                calibration=calibration,
            )
            extra[CALIBRATION_NAME] = canonical_json_bytes(receipt) + b"\n"
            extra[DISCOVERY_DISTRIBUTION_NAME] = (
                canonical_json_bytes(describe_discovery_distributions(merged)) + b"\n"
            )
            if args.rule_spec is not None:
                rule = seal_discovery_rule(
                    _read_json(args.rule_spec, "discovery rule specification"), merged
                )
                extra[DISCOVERY_RULE_NAME] = canonical_json_bytes(rule) + b"\n"
                print(f"discovery rule sealed: {rule['discovery_rule_sha256']}")
            print(f"support calibration sealed: {receipt['calibration_sha256']}")
        elif args.phase == PHASE_CONFIRMATION:
            manifest = _require_manifest(args)
            if args.calibration is None:
                _fail("confirmation requires --calibration")
            if not args.expect_calibration_sha256:
                _fail("confirmation requires --expect-calibration-sha256")
            receipt = _read_json(args.calibration, "support calibration")
            calibration = calibration_from_receipt(receipt)
            if args.expect_calibration_sha256 != receipt["calibration_sha256"]:
                _fail(
                    "confirmation was asked to bind calibration "
                    f"{args.expect_calibration_sha256!r} but was given "
                    f"{receipt['calibration_sha256']!r}"
                )
            if str(receipt["capture_manifest_sha256"]) != str(
                manifest["capture_manifest_sha256"]
            ):
                _fail("the sealed calibration was derived against a different capture manifest")
            merged = merge_census(
                plan,
                args.shard_root,
                manifest=manifest,
                allowlist=_allowlist_for_phase(PHASE_CONFIRMATION),
                phase=PHASE_CONFIRMATION,
                calibration=calibration,
            )
            if args.discovery_rule is not None:
                if not args.expect_discovery_rule_sha256:
                    _fail("binding a discovery rule requires --expect-discovery-rule-sha256")
                report = apply_confirmation(
                    merged,
                    _read_json(args.discovery_rule, "discovery rule"),
                    expect_discovery_rule_sha256=args.expect_discovery_rule_sha256,
                )
                extra[CONFIRMATION_REPORT_NAME] = canonical_json_bytes(report) + b"\n"
                print(f"confirmation bound rule: {report['bound_discovery_rule_sha256']}")
            print(f"confirmation bound calibration: {receipt['calibration_sha256']}")
        else:  # presentation
            manifest = _require_manifest(args)
            if args.output_dir is None or not (
                Path(args.output_dir) / CONFIRMATION_REPORT_NAME
            ).is_file():
                _fail(
                    "the presentation phase may only run once a confirmation report exists "
                    "in the output directory"
                )
            receipt = _read_json(args.calibration, "support calibration")
            merged = merge_census(
                plan,
                args.shard_root,
                manifest=manifest,
                phase=PHASE_PRESENTATION,
                calibration=calibration_from_receipt(receipt),
            )

        if args.output_dir is not None:
            if merged is not None:
                commit_merge(merged, args.output_dir)
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            for name, content in sorted(extra.items()):
                path = Path(args.output_dir) / name
                if path.exists() and path.read_bytes() != content:
                    _fail(f"refusing to overwrite existing {name!r} with different bytes")
                path.write_bytes(content)
        if merged is not None:
            print(summarize(merged))
    except GlobalStopError as exc:
        print(f"GLOBAL STOP: {exc}", file=sys.stderr)
        return 3
    except MergeContractError as exc:
        print(f"merge contract error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
