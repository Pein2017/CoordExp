#!/usr/bin/env python3
"""Primary branch analysis for the sorted crossing-boundary owner
release/realization unit
(``2026-08-03-sorted-crossing-boundary-owner-release-realization``).

Frozen unit::

    research/investigations/qwen3-vl-dense-enumeration/experiments/
    2026-08-03-sorted-crossing-boundary-owner-release-realization/{unit.md,tasks.md}

What this module does
---------------------
It reads **only** the merged primary owner records published by
``merge_sorted_crossing_boundary_owner_release`` and assigns the frozen four
branches -- ``displaced``, ``release_lost``, ``realization_fail``, ``ambiguous``
-- using the scorer's own pure helpers
(:func:`classify_displacement` / :func:`classify_primary_branch`), so branch
semantics cannot drift between capture parity and analysis.

Frozen reporting surface:

* the **26** U-bound crossing owners are the only primary denominator; the
  matched-E ``12`` / unmatched-E ``14`` strata and the exact same-context U&L
  ``24`` sensitivity are reported beside it and never replace it;
* **U owns** support, rank, competitor and branch.  **L** is recomputed as a
  side-by-side sensitivity branch that never changes the U branch;
* the two control cohorts (14 disjoint timing, 12 native-TP replay) are reported
  separately and are never in any primary denominator;
* ``decoding_contradicted`` cells -- ``likelihood_displaced`` true while the
  coordinate-only greedy box still strict-matches the target -- are reported
  separately and drive the frozen split rule;
* the interpretability gate is ``>= 20/26`` interpretable owners including
  ``>= 6`` matched-E and ``>= 7`` unmatched-E; and
* the two-thirds routing rule is evaluated **over the deterministic
  interpretable owners**, never over all 26.

Conclusion-fragility slices (``report.v2``)
-------------------------------------------
The routing decision is a single leading share, so ``report.json`` and
``report.md`` also carry the explicit slices a reader needs to judge how fragile
that share is.  None of them changes branch assignment, any denominator or the
routed decision; they are read-outs over the same owner rows:

* the displaced cohort's likelihood/greedy sub-tag contingency, cell by cell;
* the greedy-only "mirror" cells and the routing sensitivity to excluding them;
* the likelihood-only cells whose ``P+E`` coordinate-only greedy box is
  unmatched, which therefore carry no realized-target support;
* forced-``D_C`` coordinate-only greedy target realization at ``P``;
* for every greedy-displaced owner, the displacing matched owner beside crossing
  ``E``'s own strict-match owner -- an identity comparison only, never a causal
  claim that ``E`` displaced the target; and
* the U/L support-disposition disagreement cells at both boundaries, which are
  reported as support-level cells and never collapsed into branch disagreement.

Crossing ``E``'s physical owner is not a capture readout: it is the frozen plan
cohort registry's own ``e_row.strict_match_gt_owner_id``.  It is read back from
the plan directory the merge receipt already seals, digest-verified against that
seal, and cross-checked against each owner's sealed stratum.  When the sealed
plan directory is not resolvable the comparison degrades to an explicit
``crossing_e_owner_not_resolvable`` status rather than guessing an identity.

What this module never does
---------------------------
* It never reads or infers the deferred secondary ``P+C`` / ``P+E+C``
  compatibility evidence.  Every merged record must still carry the deferred
  sentinel, and a record that carries secondary fields fails closed.
* It never fits a threshold and never changes cohort membership from a score.
  Cohort and stratum come only from the sealed ``cohort``/``stratum`` fields.
* It never classifies at ``P``.  Only ``P+E`` is decision-bearing; the ``P``
  readout is used for the paired access tags and, for same-description owners,
  is construction-determined and therefore never displacement evidence.
* Unknown, calibration-unavailable, tie, nonunique and missing readouts route to
  ``ambiguous`` rather than to a guessed branch.

Outputs (one explicit analysis directory)::

    owner-rows.jsonl   one machine-readable row per merged owner
    report.json        denominators, branch shares, sensitivities, decision
    report.md          the same content, rendered
    receipt.json       input/output digests and the sealed decision
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import merge_sorted_crossing_boundary_owner_release as merge  # noqa: E402
from scripts.research import score_sorted_crossing_boundary_owner_release as scorer  # noqa: E402

# ---------------------------------------------------------------------------
# 0. Frozen identities
# ---------------------------------------------------------------------------

UNIT_ID = scorer.UNIT_ID
#: ``v2`` adds the ``conclusion_fragility`` block.  Every ``v1`` field keeps its
#: name and meaning, so the bump is additive and is stated rather than implied.
REPORT_SCHEMA_VERSION = "sorted-crossing-boundary-owner-release-report.v2"
RECEIPT_SCHEMA_VERSION = "sorted-crossing-boundary-owner-release-receipt.v1"
#: ``v2`` adds ``paired_transitions`` to the control rows, which the timing
#: controls need to carry their descriptive suppression readout.  Primary owner
#: rows are unchanged, so the bump is additive and is stated rather than implied.
OWNER_ROW_SCHEMA_VERSION = "sorted-crossing-boundary-owner-release-owner-row.v2"

OWNER_ROWS_NAME = "owner-rows.jsonl"
REPORT_JSON_NAME = "report.json"
REPORT_MD_NAME = "report.md"
RECEIPT_NAME = "receipt.json"

#: Only ``P+E`` is decision-bearing.
DECISION_VARIANT = "at_p_plus_e"
PAIRED_VARIANT = "at_p"
DECISION_BOUNDARY_LABEL = "P_plus_E"

PRIMARY_COHORT = merge.PRIMARY_COHORT
TIMING_CONTROL_COHORT = merge.TIMING_CONTROL_COHORT
TP_REPLAY_CONTROL_COHORT = merge.TP_REPLAY_CONTROL_COHORT

BOUND_U = scorer.SUPPORT_BOUND_U
BOUND_L = scorer.SUPPORT_BOUND_L

#: unit.md "Separately tag the paired change from ``P`` to ``P+E`` as target
#: access opened, retained, or suppressed for each ladder."  The three tags are
#: the unit's own; the two non-tags below mark the cases where the unit forbids
#: a paired reading rather than inventing one.
ACCESS_OPENED = "opened"
ACCESS_RETAINED = "retained"
ACCESS_SUPPRESSED = "suppressed"
ACCESS_NOT_OBSERVABLE = "not_observable"
ACCESS_CONSTRUCTION_DETERMINED_AT_P = "construction_determined_at_p"
ACCESS_TAGS: tuple[str, ...] = (
    ACCESS_OPENED,
    ACCESS_RETAINED,
    ACCESS_SUPPRESSED,
    ACCESS_NOT_OBSERVABLE,
    ACCESS_CONSTRUCTION_DETERMINED_AT_P,
)

#: The ladder each branch blames, used only by the pre/post coherence readout.
MECHANISM_LADDER_BY_BRANCH: Mapping[str, str | None] = {
    "displaced": "coordinate",
    "release_lost": "natural_release",
    "realization_fail": "coordinate",
    "ambiguous": None,
}

#: Transparent per-component ``P`` -> ``P+E`` changes, reported beside the coarse
#: ladder tag so the underdetermined opened/retained/suppressed operationalization
#: can never erase the stronger raw paired evidence.
CHANGE_IMPROVED = "improved"
CHANGE_WORSENED = "worsened"
CHANGE_EQUAL = "equal"
CHANGE_NOT_DETERMINABLE = "not_determinable"
#: Each component and the field inside its block that names the change, so the
#: report can tabulate every component without re-deriving any of them.
TRANSITION_LABEL_KEY: Mapping[str, str] = {
    "target_rank": "change",
    "target_minus_competitor_margin": "sign_transition",
    "best_competitor_owner_id": "transition",
    "release_target_minus_native_margin": "sign_transition",
    "support_disposition_u": "transition",
    "support_disposition_l": "transition",
    "greedy_status": "transition",
}
TRANSITION_COMPONENTS: tuple[str, ...] = tuple(TRANSITION_LABEL_KEY)

COHERENCE_COHERENT_CHANGE = "coherent_change_present"
COHERENCE_NO_PAIRED_CHANGE = "no_paired_change_at_the_crossing"
COHERENCE_CONTRADICTED = "contradicted"
COHERENCE_NOT_DETERMINABLE = "not_determinable"

DECISION_ROUTE = "route_one_mechanism_matched_successor"
DECISION_CLOSE = "close_favorable_surface_route_as_weak_local_descriptor"
DECISION_GATE_FAILED = "no_successor_interpretability_gate_failed"

#: The timing controls carry the same ``P`` -> ``P+row`` pair shape as the
#: primary cohort, at a boundary that is deliberately not the crossing.
CONTROL_PAIRED_VARIANTS: tuple[str, str] = (
    "at_control_boundary",
    "at_control_boundary_plus_row",
)

#: The four descriptive suppression readouts the timing controls are checked for.
CONTROL_SUPPRESSION_RANK_WORSENED = "target_rank_worsened"
CONTROL_SUPPRESSION_U_SUPPORT_LOST = "u_support_lost"
CONTROL_SUPPRESSION_MARGIN_POSITIVE_TO_NEGATIVE = (
    "target_minus_competitor_margin_positive_to_negative"
)
CONTROL_SUPPRESSION_GREEDY_TARGET_LOST = "greedy_target_match_lost"

#: The frozen plan file that owns crossing ``E``'s physical-owner identity.  It
#: is reached only through the path and digest the merge receipt already seals.
PLAN_COHORT_REGISTRY_NAME = "cohort-registry.jsonl"
PLAN_CONTROL_REGISTRY_NAME = "control-registry.jsonl"
CROSSING_E_SOURCE_SEALED_PLAN = "sealed_plan_cohort_registry"
CROSSING_E_SOURCE_NOT_RESOLVABLE = "crossing_e_owner_not_resolvable"

#: The upstream census plan file that owns every *other* owner's description,
#: native strict-match rows and final native disposition.  It is reached only
#: through the crossing plan's own sealed lineage contract.
CENSUS_OWNER_REGISTRY_NAME = "plan/owner-registry.jsonl"
CENSUS_OWNER_SOURCE_SEALED_LINEAGE = "sealed_census_owner_registry_via_plan_lineage"
CENSUS_OWNER_SOURCE_NOT_RESOLVABLE = "census_owner_facts_not_resolvable"

#: A displacer's final native disposition, taken from the census registry only.
NATIVE_DISPOSITION_TRUE_POSITIVE = "native_true_positive"
NATIVE_DISPOSITION_FALSE_NEGATIVE = "native_false_negative"
NATIVE_DISPOSITION_UNCLASSIFIED = "native_disposition_unclassified"
NATIVE_DISPOSITION_NOT_RESOLVABLE = "native_disposition_not_resolvable"
#: The registry's own words for an ``E`` row that strict-matches no physical
#: owner; such a crossing can never have a physical ``E`` owner to compare with.
E_STRICT_MATCH_UNMATCHED = "unmatched"

#: The three dispositions of "is the displacing owner crossing ``E``'s owner?".
DISPLACER_EQUALS_E = "displacer_is_crossing_e_owner"
DISPLACER_NOT_E = "displacer_is_not_crossing_e_owner"
DISPLACER_E_NOT_DETERMINABLE = "crossing_e_owner_not_determinable"

#: Written verbatim into ``report.json`` so no reader has to reconstruct the
#: operational choices this analyzer made where ``unit.md`` leaves them open.
OPERATIONAL_DEFINITIONS: Mapping[str, str] = {
    "primary_denominator": (
        "the 26 U-bound crossing owners; matched-E 12 / unmatched-E 14 and the exact "
        "same-context U&L 24 are sensitivities and never replace it"
    ),
    "interpretable": (
        "replay admitted, not quarantined, no missing required primary score field, no "
        "exact tie or nonunique owner match, and the truth table assigns exactly one of "
        "displaced / release_lost / realization_fail without optional sampling"
    ),
    "routing_denominator": (
        "the deterministic INTERPRETABLE owners, not all 26; the two-thirds rule is "
        "leading_branch_count / interpretable_count and is evaluated only after the "
        ">=20/26 gate (incl. >=6 matched-E, >=7 unmatched-E) passes"
    ),
    "support_bound": (
        "U owns the primary support disposition and therefore the primary branch; L is "
        "recomputed as a side-by-side sensitivity branch that never changes the U branch"
    ),
    "release_ladder_target_access": (
        "at one boundary: release is observable and its target-minus-native margin at the "
        "first observable divergence is strictly positive; this is the same signed axis "
        "branch 2 (release_lost) uses, so the paired tag and the branch cannot drift apart"
    ),
    "coordinate_ladder_target_access": (
        "at one boundary: the U-calibrated forced-D_C target-local support disposition is "
        "'supported'; this is the same axis branches 2 and 3 use"
    ),
    "paired_access_tag": (
        "opened = access absent at P and present at P+E; suppressed = present at P and "
        "absent at P+E; retained = unchanged (the raw access_at_p / access_at_p_plus_e "
        "booleans are carried so 'retained present' and 'retained absent' are never "
        "confused); not_observable = either boundary's predicate is undeterminable; "
        "construction_determined_at_p = the coordinate ladder of a same-description owner, "
        "whose P readout unit.md records for replay only and never as diagnostic evidence"
    ),
    "paired_component_transitions": (
        "raw, non-interpretive P -> P+E changes carried beside the coarse ladder tag: "
        "target_rank (improved = strictly smaller rank at P+E), the sign transition and "
        "numeric delta of the target-minus-competitor margin, whether the best competing "
        "owner changed, the sign transition and numeric delta of the release "
        "target-minus-native margin, the U and L support-disposition transitions, and the "
        "greedy owner-match transition; every field is the literal capture readout at each "
        "boundary and none of them is thresholded here"
    ),
    "pre_post_coherence": (
        "an explicit operational definition of unit.md's otherwise underdetermined 'pre/post "
        "changes cohere with the branch ladder' clause, restricted to the interpretable "
        "owners in the leading branch and to that branch's own mechanism ladder "
        "(displaced -> coordinate, release_lost -> natural_release, realization_fail -> "
        "coordinate): 'contradicted' if any determinable mechanism-ladder tag is 'opened' "
        "(the crossing row improved the very ladder the branch blames); "
        "'no_paired_change_at_the_crossing' if every determinable tag is 'retained'; "
        "'coherent_change_present' if at least one is 'suppressed' and none is 'opened'; "
        "'not_determinable' if no such owner has a determinable tag. Only 'contradicted' "
        "closes the route automatically; the other non-coherent statuses are surfaced with "
        "requires_adjudication=true rather than silently deciding the unit"
    ),
    "secondary_evidence": (
        "never read: every merged record must still carry the deferred secondary sentinel, "
        "and no P+C / P+E+C field is loaded, parsed or inferred here"
    ),
    "cohort_membership": (
        "taken only from the sealed cohort/stratum fields; no threshold is fitted and no "
        "cohort is changed from a score"
    ),
    "conclusion_fragility": (
        "read-only slices of the same owner rows that expose how much the single routed "
        "share depends on one kind of evidence: they never reassign a branch, never change "
        "a denominator and never change the decision, and every cell is named by owner ID "
        "so a reader can re-derive each count"
    ),
    "two_thirds_threshold": (
        "the smallest leading-branch count that reaches the frozen two-thirds rule over a "
        "given interpretable total: ceil(2 * total / 3), which is exactly the scorer's own "
        "integer test count * 3 >= total * 2 and never a float ratio"
    ),
    "displaced_sub_tag_contingency": (
        "the four disjoint cells of the displaced cohort under the two displacement "
        "sub-tags: likelihood_displaced and greedy_displaced, likelihood only, greedy only, "
        "and neither; the sub-tags are the scorer's own and are not recomputed here"
    ),
    "greedy_only_mirror_exclusion": (
        "the displaced cells carried only by the coordinate-only greedy mirror, and what "
        "the leading share becomes if they are removed from the interpretable routing "
        "denominator; reported as a sensitivity and never applied to the routed decision"
    ),
    "likelihood_only_realized_target_support": (
        "displaced cells that are likelihood_displaced but not greedy_displaced, tabulated "
        "by their P+E coordinate-only greedy status: an 'unmatched' cell contributes "
        "likelihood evidence and no realized-target support, because no decoded box at P+E "
        "strict-matches any physical owner"
    ),
    "forced_dc_greedy_target_realization_at_p": (
        "the literal P-boundary coordinate-only greedy status of the forced-D_C readout for "
        "all 26 primary owners; P is never decision-bearing here, and the same-description "
        "owners whose P coordinate readout unit.md calls construction-determined are "
        "counted separately rather than dropped or merged"
    ),
    "greedy_displacer_identity": (
        "for each greedy_displaced owner, the owner its P+E coordinate-only greedy box "
        "strict-matches, compared for identity only against crossing E's own strict-match "
        "owner from the sealed plan cohort registry; an unmatched E row has no physical "
        "owner and therefore can never equal the displacer.  This is an identity readout: "
        "it is never evidence that E caused, enabled or produced the displacement"
    ),
    "crossing_e_owner_identity_source": (
        "e_row.strict_match_gt_owner_id from the frozen plan cohort registry, reached only "
        "through the plan directory the merge receipt seals, re-hashed against that sealed "
        "digest before it is read, and cross-checked against each owner's sealed stratum; a "
        "missing plan path or a digest mismatch fails closed, and a sealed registry that "
        "does not carry the field degrades to crossing_e_owner_not_resolvable"
    ),
    "support_bound_disagreement_cells": (
        "every (owner, boundary) cell whose U and L support dispositions differ, at P and "
        "at P+E alike; this is a support-level readout and is reported beside, never "
        "collapsed into, the L-bound branch disagreement, because a bound can move the "
        "support disposition without moving the branch"
    ),
    "covered_before_p": (
        "the displacing owner already has a native strict-match sorted row whose row index "
        "is strictly less than the crossing owner's sealed P boundary index.  The frozen "
        "convention is that boundary b contains native rows < b, so 'covered before P' is "
        "exactly 'already emitted as a native strict match inside P's own prefix'.  Row "
        "indices come from the census owner registry's native_strict_match_pred_row_ids and "
        "the boundary index from the sealed plan registry; neither is derived from a score"
    ),
    "displacer_characterization": (
        "what the owners that displace the crossing targets are, joined from the sealed "
        "census owner registry: whether they share the target's normalized description, "
        "whether they were already covered before P, and their final native disposition.  "
        "Reported for both displacement sub-tags.  It is descriptive only: nothing here "
        "shows that a displacer caused, or was caused by, the target's omission"
    ),
    "displacer_same_description_is_construction_determined": (
        "the coordinate competitor family is same-category by construction, so a displacer "
        "sharing the target's normalized description is a property of how the family was "
        "built, not a discovered same-description displacement mechanism.  It is reported "
        "as an audit property and must never be read as evidence for such a mechanism.  The "
        "nontrivial readouts in this block are the covered-before-P share and the final "
        "native disposition split, neither of which the family construction fixes"
    ),
    "timing_control_competitor_coverage": (
        "the same covered-before-P audit, applied to the timing controls' own best "
        "competing owner at the control boundary pair and reported beside the primary "
        "cohort.  Descriptive and confounded in exactly the way the timing controls are: it "
        "supports only 'within-category scheduling instability also occurs away from a "
        "crossing', never a crossing-E-specific effect and never a causal contrast"
    ),
    "timing_control_suppression": (
        "the same four raw P -> P+row changes, read over the 14 disjoint timing controls "
        "at a deliberately non-crossing boundary: target rank worsened, U support lost, "
        "the target-minus-competitor margin sign turning from positive to negative, and a "
        "coordinate-only greedy target match lost.  Timing, description identity and route "
        "tier are entangled in this cohort, so it is a confound exhibit and not a matched "
        "counterfactual: it shows only that these suppressions also occur away from a "
        "crossing, never how much of the crossing cohort's suppression E explains"
    ),
}

NOT_CLAIMED: tuple[str, ...] = (
    "no natural probability for the complete target row",
    "no population prevalence beyond the frozen twelve images",
    "no causal effect of emitting E or inserting the GT row for C",
    "no distinction of trajectory-level starvation",
    "no final-set, natural-stop, or long-horizon preservation result",
    "no training or architecture promotion",
)


class AnalysisContractError(RuntimeError):
    """A precondition for a conclusion-bearing branch analysis was not proven."""


def _fail(message: str) -> NoReturn:
    raise AnalysisContractError(message)


canonical_json_bytes = scorer.canonical_json_bytes
sha256_bytes = scorer.sha256_bytes
sha256_json = scorer.sha256_json


# ---------------------------------------------------------------------------
# 1. Merged input loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergedInputs:
    merged_dir: Path
    merge_receipt: dict[str, Any]
    owner_records: list[dict[str, Any]]
    file_sha256: dict[str, str]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not Path(path).is_file():
        _fail(f"{label} is missing at {path}")
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{label} is unreadable at {path}: {exc}")
    if not isinstance(value, Mapping):
        _fail(f"{label} at {path} is not a JSON object")
    return dict(value)


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        _fail(f"{label} is missing at {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
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


def load_merged_inputs(merged_dir: Path) -> MergedInputs:
    """Load the merged family and re-prove every digest the merge sealed."""

    merged_dir = Path(merged_dir)
    receipt = _read_json(merged_dir / merge.MERGE_RECEIPT_NAME, "merge receipt")
    if str(receipt.get("schema_version")) != merge.MERGE_SCHEMA_VERSION:
        _fail(
            f"merge receipt schema {receipt.get('schema_version')!r} is not "
            f"{merge.MERGE_SCHEMA_VERSION!r}"
        )
    if str(receipt.get("unit_id")) != UNIT_ID:
        _fail("merge receipt belongs to another unit")
    merge.assert_self_sealed(
        receipt, digest_key="receipt_content_sha256", label="merge receipt"
    )
    policy = receipt.get("policy") or {}
    if bool(policy.get("branch_assignment_performed")):
        _fail("the merge claims to have assigned branches; branch assignment belongs here")
    if bool(policy.get("secondary_compatibility_merged")):
        _fail(
            "the merged family carries secondary compatibility evidence; primary branches "
            "must be sealed before any secondary field exists"
        )

    declared = receipt.get("output_file_digests")
    if not isinstance(declared, Mapping):
        _fail("merge receipt declares no output_file_digests")
    file_sha256: dict[str, str] = {}
    for name, entry in sorted(declared.items()):
        path = merged_dir / str(name)
        if not path.is_file():
            _fail(f"merged file {name!r} is missing at {path}")
        payload = path.read_bytes()
        observed = sha256_bytes(payload)
        if observed != str((entry or {}).get("sha256")):
            _fail(
                f"merged file {name!r} does not match the digest the merge receipt sealed; "
                "the merged evidence was modified"
            )
        file_sha256[str(name)] = observed
    file_sha256[merge.MERGE_RECEIPT_NAME] = sha256_bytes(
        (merged_dir / merge.MERGE_RECEIPT_NAME).read_bytes()
    )

    owner_records = _read_jsonl(
        merged_dir / merge.MERGED_OWNER_RECORDS_NAME, "merged owner records"
    )
    expected_rows = int(
        (declared.get(merge.MERGED_OWNER_RECORDS_NAME) or {}).get("row_count", -1)
    )
    if len(owner_records) != expected_rows:
        _fail(
            f"merged owner records carry {len(owner_records)} rows, not the "
            f"{expected_rows} the merge receipt sealed"
        )
    return MergedInputs(
        merged_dir=merged_dir,
        merge_receipt=receipt,
        owner_records=owner_records,
        file_sha256=file_sha256,
    )


def load_sealed_plan_crossing_facts(merge_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Read crossing ``E``'s strict-match owner and ``P``'s boundary index from the plan.

    Both are *plan* facts, not capture readouts, so they are reached only through
    the plan directory and per-file digest the merge receipt already seals: the
    file is re-hashed against that seal before a single row is parsed.  A missing
    plan path or a digest mismatch fails closed.  A sealed registry that simply
    does not carry the fields -- an older or otherwise differently-shaped plan
    revision -- degrades to an explicit ``crossing_e_owner_not_resolvable`` status
    rather than a guessed identity.

    Nothing here reads the deferred secondary evidence, and no ``E`` owner is ever
    inferred from a stratum, a description or a score.
    """

    plan = merge_receipt.get("plan")
    if not isinstance(plan, Mapping):
        _fail("merge receipt declares no plan block; crossing E identity is unreachable")
    plan_dir = Path(str(plan.get("plan_dir")))
    cohort_rows, cohort_sha256 = _read_sealed_plan_file(plan, PLAN_COHORT_REGISTRY_NAME)
    control_rows, control_sha256 = _read_sealed_plan_file(plan, PLAN_CONTROL_REGISTRY_NAME)

    def unresolved(reason: str) -> dict[str, Any]:
        return {
            "source": CROSSING_E_SOURCE_NOT_RESOLVABLE,
            "reason": reason,
            "plan_dir": str(plan_dir),
            "cohort_registry_sha256": cohort_sha256,
            "control_registry_sha256": control_sha256,
            "by_owner": {},
            "control_by_owner": {},
        }

    resolved: dict[str, dict[str, Any]] = {}
    for row in cohort_rows:
        owner_id = row.get("gt_owner_id")
        e_row = row.get("e_row")
        crossing = row.get("crossing")
        if (
            not isinstance(owner_id, str)
            or not isinstance(e_row, Mapping)
            or not isinstance(crossing, Mapping)
        ):
            return unresolved(
                "the sealed plan cohort registry carries no per-owner e_row/crossing "
                "block; crossing E's owner and P's boundary index are not resolvable "
                "from this plan revision"
            )
        status = str(e_row.get("strict_match_status"))
        matched_owner = e_row.get("strict_match_gt_owner_id")
        if status == E_STRICT_MATCH_UNMATCHED and matched_owner is not None:
            _fail(
                f"the plan cohort registry declares owner {owner_id!r} an unmatched crossing "
                f"E while naming {matched_owner!r} as its strict match"
            )
        boundary_index = crossing.get("p_boundary_index")
        resolved[owner_id] = {
            "crossing_e_owner_id": None if matched_owner is None else str(matched_owner),
            "crossing_e_strict_match_status": status,
            "stratum": e_row.get("stratum"),
            "p_boundary_index": None if boundary_index is None else int(boundary_index),
        }

    # The control registry names no ``p_boundary_index`` of its own: under the
    # frozen convention that boundary ``b`` contains native rows ``< b``, the row
    # the control boundary is followed by *is* row index ``b``, exactly as ``E``
    # is row index ``b`` for a crossing owner.
    # Only the timing controls have a boundary pair at all; the TP replay controls
    # are a single due-boundary calibration cohort and carry no ``next_row``.
    control_resolved: dict[str, dict[str, Any]] = {}
    for row in control_rows:
        owner_id = row.get("gt_owner_id")
        if not isinstance(owner_id, str):
            return unresolved(
                "the sealed plan control registry carries rows without a gt_owner_id; the "
                "control boundary index is not resolvable from this plan revision"
            )
        if str(row.get("cohort")) != TIMING_CONTROL_COHORT:
            continue
        next_row = row.get("next_row")
        if not isinstance(next_row, Mapping):
            return unresolved(
                "the sealed plan control registry carries no next_row block for a timing "
                "control; the control boundary index is not resolvable from this plan "
                "revision"
            )
        row_index = next_row.get("row_index")
        control_resolved[owner_id] = {
            "cohort": row.get("cohort"),
            "control_boundary_index": None if row_index is None else int(row_index),
        }

    return {
        "source": CROSSING_E_SOURCE_SEALED_PLAN,
        "reason": None,
        "plan_dir": str(plan_dir),
        "cohort_registry_sha256": cohort_sha256,
        "control_registry_sha256": control_sha256,
        "by_owner": resolved,
        "control_by_owner": control_resolved,
    }


def _read_sealed_plan_file(
    plan: Mapping[str, Any], name: str
) -> tuple[list[dict[str, Any]], str]:
    """One frozen plan file, re-hashed against the digest the merge receipt sealed."""

    declared = plan.get("plan_file_sha256")
    if not isinstance(declared, Mapping) or name not in declared:
        _fail(f"merge receipt seals no {name!r} digest; the plan file would be unverifiable")
    path = Path(str(plan.get("plan_dir"))) / name
    if not path.is_file():
        _fail(
            f"the sealed plan file {name!r} is missing at {path}; plan facts are only ever "
            "read from the plan path the merge receipt sealed"
        )
    observed = sha256_bytes(path.read_bytes())
    if observed != str(declared[name]):
        _fail(
            f"the plan file at {path} does not match the digest the merge receipt sealed; "
            "the frozen plan was modified"
        )
    return _read_jsonl(path, f"plan file {name!r}"), observed


def load_census_owner_facts(merge_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Read the census owner registry the crossing plan's own lineage seals.

    The displacer characterization needs facts about *other* owners -- their
    normalized description, their native strict-match sorted rows and their final
    native disposition -- which live in the upstream census plan.  That file is
    reached only through ``plan.lineage.census_run_root`` plus the sealed
    ``plan.lineage.census_input_files['plan/owner-registry.jsonl']`` entry, and is
    re-hashed against that entry before it is read.  A declared-but-missing path,
    a byte-size mismatch or a digest mismatch fails closed; a lineage block that
    does not declare the contract at all degrades to an explicit not-resolvable
    status so the characterization is omitted rather than inferred.
    """

    plan = merge_receipt.get("plan")
    lineage = plan.get("lineage") if isinstance(plan, Mapping) else None

    def unresolved(reason: str) -> dict[str, Any]:
        return {
            "source": CENSUS_OWNER_SOURCE_NOT_RESOLVABLE,
            "reason": reason,
            "census_run_root": None,
            "owner_registry_path": None,
            "owner_registry_sha256": None,
            "by_owner": {},
        }

    if not isinstance(lineage, Mapping):
        return unresolved("the merge receipt's plan block declares no lineage")
    declared_files = lineage.get("census_input_files")
    run_root = lineage.get("census_run_root")
    if (
        not isinstance(declared_files, Mapping)
        or CENSUS_OWNER_REGISTRY_NAME not in declared_files
        or not isinstance(run_root, str)
    ):
        return unresolved(
            "the plan lineage declares no census_run_root plus "
            f"census_input_files[{CENSUS_OWNER_REGISTRY_NAME!r}] contract; the displacer "
            "characterization is omitted rather than inferred"
        )
    entry = declared_files[CENSUS_OWNER_REGISTRY_NAME] or {}
    path = Path(run_root) / CENSUS_OWNER_REGISTRY_NAME
    if not path.is_file():
        _fail(
            f"the census owner registry the plan lineage seals is missing at {path}; the "
            "displacer characterization is only ever read from the sealed lineage path"
        )
    payload = path.read_bytes()
    declared_size = entry.get("byte_size")
    if declared_size is not None and len(payload) != int(declared_size):
        _fail(
            f"the census owner registry at {path} is {len(payload)} bytes, not the "
            f"{int(declared_size)} the plan lineage sealed"
        )
    observed = sha256_bytes(payload)
    if observed != str(entry.get("sha256")):
        _fail(
            f"the census owner registry at {path} does not match the digest the plan "
            "lineage sealed; the upstream census plan was modified"
        )

    resolved: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path, "census owner registry"):
        owner_id = row.get("gt_owner_id")
        pred_row_ids = row.get("native_strict_match_pred_row_ids")
        eligibility = row.get("disposition_eligibility")
        if (
            not isinstance(owner_id, str)
            or not isinstance(pred_row_ids, Sequence)
            or isinstance(pred_row_ids, str)
            or not isinstance(eligibility, Mapping)
        ):
            return unresolved(
                "the sealed census owner registry carries no per-owner "
                "native_strict_match_pred_row_ids / disposition_eligibility fields; the "
                "displacer characterization is omitted rather than inferred"
            )
        resolved[owner_id] = {
            "normalized_description": row.get("normalized_description"),
            "native_strict_match_pred_row_ids": [str(value) for value in pred_row_ids],
            "native_strict_match_row_indices": sorted(
                _native_row_index(value) for value in pred_row_ids
            ),
            "native_true_positive": bool(row.get("native_true_positive")),
            "native_false_negative": bool(eligibility.get("native_false_negative")),
        }
    return {
        "source": CENSUS_OWNER_SOURCE_SEALED_LINEAGE,
        "reason": None,
        "census_run_root": str(run_root),
        "owner_registry_path": str(path),
        "owner_registry_sha256": observed,
        "by_owner": resolved,
    }


def _native_row_index(pred_row_id: Any) -> int:
    """The sorted native row index a census strict-match predicted-row ID names."""

    tail = str(pred_row_id).rsplit(":", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        _fail(
            f"the census owner registry names strict-match row {pred_row_id!r}, whose "
            "trailing segment is not a sorted row index"
        )


def assert_no_secondary_evidence(record: Mapping[str, Any]) -> None:
    """The deferred ``P+C`` / ``P+E+C`` readouts are never read or inferred here."""

    value = record.get("secondary_compatibility")
    if value != merge.SECONDARY_COMPATIBILITY_SENTINEL:
        _fail(
            f"owner record {record.get('gt_owner_id')!r} carries secondary compatibility "
            f"payload {value!r}; unit.md seals every primary branch before any secondary "
            "field is read"
        )
    if str(record.get("branch_assignment")) != merge.BRANCH_ASSIGNMENT_SENTINEL:
        _fail(
            f"owner record {record.get('gt_owner_id')!r} already carries a branch; branch "
            "assignment is this pass and is never inherited"
        )


# ---------------------------------------------------------------------------
# 2. Ladder readback (pure; nothing here re-derives a score)
# ---------------------------------------------------------------------------


def _ladder(record: Mapping[str, Any], variant: str) -> Mapping[str, Any] | None:
    ladders = record.get("ladders")
    if not isinstance(ladders, Mapping):
        return None
    ladder = ladders.get(variant)
    return ladder if isinstance(ladder, Mapping) else None


def _coordinate(ladder: Mapping[str, Any]) -> Mapping[str, Any]:
    coordinate = ladder.get("coordinate")
    if not isinstance(coordinate, Mapping):
        _fail("a ladder carries no coordinate readout")
    return coordinate


def support_disposition(ladder: Mapping[str, Any], *, bound: str) -> str:
    """The calibrated support disposition under one bound, U primary, L sensitivity."""

    coordinate = _coordinate(ladder)
    by_bound = coordinate.get("support_by_bound")
    if not isinstance(by_bound, Mapping) or bound not in by_bound:
        _fail(f"a ladder carries no {bound!r}-bound support readout")
    disposition = str((by_bound[bound] or {}).get("support_disposition"))
    if disposition not in scorer.SUPPORT_DISPOSITIONS:
        _fail(f"unknown {bound!r}-bound support disposition {disposition!r}")
    if bound == BOUND_U:
        if str(coordinate.get("primary_support_bound")) != BOUND_U:
            _fail("a ladder does not declare U as its primary support bound")
        if str(coordinate.get("support_disposition")) != disposition:
            _fail(
                "a ladder's primary support disposition disagrees with its own U-bound "
                "readout"
            )
    return disposition


def _owner_rank(ladder: Mapping[str, Any], *, owner_id: str) -> scorer.OwnerRankResult | None:
    coordinate = _coordinate(ladder)
    if coordinate.get("target_rank") is None:
        return None
    disposition = coordinate.get("family_rank_disposition")
    if disposition is None or str(disposition) not in scorer.RANK_DISPOSITIONS:
        _fail(f"owner {owner_id!r} carries unknown rank disposition {disposition!r}")
    margin = coordinate.get("target_minus_competitor_margin")
    return scorer.OwnerRankResult(
        target_owner_id=str(owner_id),
        target_rank=int(coordinate["target_rank"]),
        best_competitor_owner_id=(
            None
            if coordinate.get("best_competitor_owner_id") is None
            else str(coordinate["best_competitor_owner_id"])
        ),
        target_minus_competitor_margin=None if margin is None else float(margin),
        family_rank_disposition=str(disposition),
    )


def _greedy(ladder: Mapping[str, Any]) -> Mapping[str, Any]:
    greedy = _coordinate(ladder).get("greedy")
    if not isinstance(greedy, Mapping):
        _fail("a ladder carries no coordinate-only greedy readout")
    return greedy


def release_target_access(ladder: Mapping[str, Any] | None) -> bool | None:
    """Release-ladder target access at one boundary; ``None`` when undeterminable."""

    if ladder is None:
        return None
    release = ladder.get("release")
    if not isinstance(release, Mapping):
        return None
    if not bool(release.get("observable")):
        return None
    margin = release.get("target_minus_native_margin")
    if margin is None:
        return None
    return float(margin) > 0.0


def coordinate_target_access(ladder: Mapping[str, Any] | None, *, bound: str) -> bool | None:
    """Coordinate-ladder target access at one boundary; ``None`` when undeterminable."""

    if ladder is None:
        return None
    disposition = support_disposition(ladder, bound=bound)
    if disposition in (
        scorer.SUPPORT_AMBIGUOUS_TIE,
        scorer.SUPPORT_CALIBRATION_UNAVAILABLE,
    ):
        return None
    return disposition == scorer.SUPPORT_SUPPORTED


def paired_access_tag(at_p: bool | None, at_p_plus_e: bool | None) -> str:
    """unit.md's paired ``P`` -> ``P+E`` tag: opened, retained, or suppressed."""

    if at_p is None or at_p_plus_e is None:
        return ACCESS_NOT_OBSERVABLE
    if not at_p and at_p_plus_e:
        return ACCESS_OPENED
    if at_p and not at_p_plus_e:
        return ACCESS_SUPPRESSED
    return ACCESS_RETAINED


def _margin_sign(value: Any) -> str:
    """A margin's literal sign, kept as a label so ``None`` is never read as zero."""

    if value is None:
        return "null"
    numeric = float(value)
    if numeric > 0.0:
        return "positive"
    if numeric < 0.0:
        return "negative"
    return "zero"


def _label(value: Any) -> str:
    return "null" if value is None else str(value)


def _transition(at_p: Any, at_p_plus_e: Any) -> str:
    return f"{_label(at_p)}->{_label(at_p_plus_e)}"


def _numeric_delta(at_p: Any, at_p_plus_e: Any) -> float | None:
    if at_p is None or at_p_plus_e is None:
        return None
    return float(at_p_plus_e) - float(at_p)


def _rank_change(at_p: Any, at_p_plus_e: Any) -> str:
    """A smaller rank at ``P+E`` is an improvement; ``None`` never guesses."""

    if at_p is None or at_p_plus_e is None:
        return CHANGE_NOT_DETERMINABLE
    if int(at_p_plus_e) < int(at_p):
        return CHANGE_IMPROVED
    if int(at_p_plus_e) > int(at_p):
        return CHANGE_WORSENED
    return CHANGE_EQUAL


def build_paired_transitions(
    at_p: Mapping[str, Any],
    at_p_plus_e: Mapping[str, Any],
    *,
    coordinate_is_construction_determined: bool,
) -> dict[str, Any]:
    """Every raw ``P`` -> ``P+E`` component change, with no interpretation applied."""

    return {
        "coordinate_transitions_are_construction_determined_at_p": bool(
            coordinate_is_construction_determined
        ),
        "target_rank": {
            "at_p": at_p.get("target_rank"),
            "at_p_plus_e": at_p_plus_e.get("target_rank"),
            "change": _rank_change(at_p.get("target_rank"), at_p_plus_e.get("target_rank")),
            "delta": _numeric_delta(
                at_p.get("target_rank"), at_p_plus_e.get("target_rank")
            ),
        },
        "target_minus_competitor_margin": {
            "at_p": at_p.get("target_minus_competitor_margin"),
            "at_p_plus_e": at_p_plus_e.get("target_minus_competitor_margin"),
            "sign_at_p": _margin_sign(at_p.get("target_minus_competitor_margin")),
            "sign_at_p_plus_e": _margin_sign(
                at_p_plus_e.get("target_minus_competitor_margin")
            ),
            "sign_transition": _transition(
                _margin_sign(at_p.get("target_minus_competitor_margin")),
                _margin_sign(at_p_plus_e.get("target_minus_competitor_margin")),
            ),
            "delta": _numeric_delta(
                at_p.get("target_minus_competitor_margin"),
                at_p_plus_e.get("target_minus_competitor_margin"),
            ),
        },
        "best_competitor_owner_id": {
            "at_p": at_p.get("best_competitor_owner_id"),
            "at_p_plus_e": at_p_plus_e.get("best_competitor_owner_id"),
            "transition": _transition(
                at_p.get("best_competitor_owner_id"),
                at_p_plus_e.get("best_competitor_owner_id"),
            ),
            "changed": at_p.get("best_competitor_owner_id")
            != at_p_plus_e.get("best_competitor_owner_id"),
        },
        "release_target_minus_native_margin": {
            "at_p": at_p.get("release_target_minus_native_margin"),
            "at_p_plus_e": at_p_plus_e.get("release_target_minus_native_margin"),
            "sign_at_p": _margin_sign(at_p.get("release_target_minus_native_margin")),
            "sign_at_p_plus_e": _margin_sign(
                at_p_plus_e.get("release_target_minus_native_margin")
            ),
            "sign_transition": _transition(
                _margin_sign(at_p.get("release_target_minus_native_margin")),
                _margin_sign(at_p_plus_e.get("release_target_minus_native_margin")),
            ),
            "delta": _numeric_delta(
                at_p.get("release_target_minus_native_margin"),
                at_p_plus_e.get("release_target_minus_native_margin"),
            ),
            "argmax_follows_target_transition": _transition(
                at_p.get("release_argmax_follows_target"),
                at_p_plus_e.get("release_argmax_follows_target"),
            ),
        },
        "support_disposition_u": {
            "at_p": at_p.get("support_disposition_u"),
            "at_p_plus_e": at_p_plus_e.get("support_disposition_u"),
            "transition": _transition(
                at_p.get("support_disposition_u"), at_p_plus_e.get("support_disposition_u")
            ),
        },
        "support_disposition_l": {
            "at_p": at_p.get("support_disposition_l"),
            "at_p_plus_e": at_p_plus_e.get("support_disposition_l"),
            "transition": _transition(
                at_p.get("support_disposition_l"), at_p_plus_e.get("support_disposition_l")
            ),
        },
        "greedy_status": {
            "at_p": at_p.get("greedy_status"),
            "at_p_plus_e": at_p_plus_e.get("greedy_status"),
            "transition": _transition(
                at_p.get("greedy_status"), at_p_plus_e.get("greedy_status")
            ),
            "owner_match_transition": _transition(
                at_p.get("greedy_owner_match"), at_p_plus_e.get("greedy_owner_match")
            ),
            "nonunique_match_transition": _transition(
                at_p.get("greedy_nonunique_match"),
                at_p_plus_e.get("greedy_nonunique_match"),
            ),
        },
    }


def _boundary_fields(
    ladder: Mapping[str, Any], *, owner_id: str, construction_determined: bool = False
) -> dict[str, Any]:
    """Every raw readout one boundary carries, preserved for matrix and plots."""

    release = ladder.get("release")
    release = release if isinstance(release, Mapping) else None
    rank = _owner_rank(ladder, owner_id=owner_id)
    greedy = _greedy(ladder)
    return {
        "context_id": ladder.get("context_id"),
        "boundary_label": ladder.get("boundary_label"),
        "native_action_kind": ladder.get("native_action_kind"),
        "native_argmax_replay_admitted": ladder.get("native_argmax_replay_admitted"),
        "release_observable": None if release is None else bool(release.get("observable")),
        "release_target_minus_native_margin": (
            None if release is None else release.get("target_minus_native_margin")
        ),
        "release_argmax_follows_target": (
            None if release is None else release.get("argmax_follows_target")
        ),
        "release_first_divergence_index": (
            None if release is None else release.get("first_divergence_index")
        ),
        "release_gate_margin_is_versus_stop": (
            None if release is None else release.get("gate_margin_is_versus_stop")
        ),
        "target_rank": None if rank is None else rank.target_rank,
        "best_competitor_owner_id": None if rank is None else rank.best_competitor_owner_id,
        "target_minus_competitor_margin": (
            None if rank is None else rank.target_minus_competitor_margin
        ),
        "family_rank_disposition": None if rank is None else rank.family_rank_disposition,
        "support_disposition_u": support_disposition(ladder, bound=BOUND_U),
        "support_disposition_l": support_disposition(ladder, bound=BOUND_L),
        "greedy_status": greedy.get("greedy_status"),
        "greedy_owner_match": greedy.get("greedy_owner_match"),
        "greedy_nonunique_match": bool(greedy.get("greedy_nonunique_match")),
        "coordinate_readout_is_construction_determined": bool(construction_determined),
    }


# ---------------------------------------------------------------------------
# 3. One owner: branch, sensitivity, paired tags
# ---------------------------------------------------------------------------


def _branch_inputs(
    record: Mapping[str, Any], ladder: Mapping[str, Any]
) -> dict[str, Any]:
    owner_id = str(record["gt_owner_id"])
    rank = _owner_rank(ladder, owner_id=owner_id)
    release = ladder.get("release")
    greedy = _greedy(ladder)
    greedy_status = greedy.get("greedy_status")
    missing_fields = (
        rank is None or not isinstance(release, Mapping) or greedy_status is None
    )
    if not missing_fields and str(greedy_status) not in scorer.GREEDY_STATUSES:
        _fail(f"owner {owner_id!r} carries unknown greedy status {greedy_status!r}")
    return {
        "owner_id": owner_id,
        "rank": rank,
        "release": release if isinstance(release, Mapping) else None,
        "greedy": greedy,
        "greedy_status": None if greedy_status is None else str(greedy_status),
        "missing_fields": bool(missing_fields),
    }


def classify_owner(record: Mapping[str, Any]) -> dict[str, Any]:
    """Assign the frozen branches for one merged primary owner record.

    ``U`` decides the primary branch; ``L`` is recomputed beside it as a
    sensitivity.  Only ``P+E`` is classified: the ``P`` leg supplies the paired
    access tags and, for a same-description owner, is construction-determined.
    """

    assert_no_secondary_evidence(record)
    owner_id = str(record["gt_owner_id"])
    if str(record.get("cohort")) != PRIMARY_COHORT:
        _fail(f"owner {owner_id!r} is not in the primary cohort and is never classified")

    decision_ladder = _ladder(record, DECISION_VARIANT)
    paired_ladder = _ladder(record, PAIRED_VARIANT)
    if decision_ladder is None or paired_ladder is None:
        _fail(f"primary owner {owner_id!r} is missing one of its two frozen ladders")
    if str(decision_ladder.get("boundary_label")) != DECISION_BOUNDARY_LABEL:
        _fail(
            f"owner {owner_id!r} declares boundary {decision_ladder.get('boundary_label')!r} "
            f"for {DECISION_VARIANT!r}; only P+E is decision-bearing"
        )

    same_description = bool(record.get("same_description_as_e"))
    quarantined = bool(record.get("quarantined"))
    replay_admitted = bool(record.get("replay_admitted"))

    inputs = _branch_inputs(record, decision_ladder)
    rank = inputs["rank"]
    release = inputs["release"]
    greedy = inputs["greedy"]
    greedy_status = inputs["greedy_status"]
    missing_fields = inputs["missing_fields"]

    u_support = support_disposition(decision_ladder, bound=BOUND_U)
    l_support = support_disposition(decision_ladder, bound=BOUND_L)

    tie_or_nonunique = bool(
        rank is not None
        and (
            rank.family_rank_disposition == scorer.RANK_TIE
            or greedy_status == scorer.GREEDY_AMBIGUOUS
            or bool(greedy.get("greedy_nonunique_match"))
        )
    )

    displacement: scorer.DisplacementResult | None = None
    branch_u: str | None = None
    reasons_u: tuple[str, ...] = ()
    branch_l: str | None = None
    if missing_fields:
        branch_u = "ambiguous"
        reasons_u = ("ambiguous: a required primary score field is missing",)
        branch_l = "ambiguous"
    else:
        displacement = scorer.classify_displacement(
            target_owner_id=owner_id,
            owner_rank=rank,
            greedy_status=greedy_status,
            greedy_owner_match=greedy.get("greedy_owner_match"),
        )
        result_u = scorer.classify_primary_branch(
            displacement=displacement,
            release_observable=bool(release.get("observable")),
            release_margin=release.get("target_minus_native_margin"),
            forced_dc_support_disposition=u_support,
            greedy_status=greedy_status,
            tie_or_nonunique=tie_or_nonunique,
            missing_fields=False,
            boundary_label=DECISION_BOUNDARY_LABEL,
        )
        branch_u = result_u.branch
        reasons_u = result_u.reasons
        # L never changes the U branch: it re-enters exactly one input.
        branch_l = scorer.classify_primary_branch(
            displacement=displacement,
            release_observable=bool(release.get("observable")),
            release_margin=release.get("target_minus_native_margin"),
            forced_dc_support_disposition=l_support,
            greedy_status=greedy_status,
            tie_or_nonunique=tie_or_nonunique,
            missing_fields=False,
            boundary_label=DECISION_BOUNDARY_LABEL,
        ).branch

    if quarantined or not replay_admitted:
        # unit.md: a case is interpretable only if replay is admitted.  An
        # unadmitted replay is never given a branch to be counted from.
        branch_u = None
        branch_l = None
        reasons_u = (
            "not classified: native argmax replay was not admitted for this owner",
        )

    interpretable = bool(
        replay_admitted
        and not quarantined
        and not missing_fields
        and not tie_or_nonunique
        and branch_u in ("displaced", "release_lost", "realization_fail")
    )
    not_interpretable_reasons: list[str] = []
    if quarantined or not replay_admitted:
        not_interpretable_reasons.append("replay_not_admitted")
    if missing_fields:
        not_interpretable_reasons.append("missing_required_primary_score_field")
    if tie_or_nonunique:
        not_interpretable_reasons.append("exact_tie_or_nonunique_owner_match")
    if branch_u == "ambiguous":
        not_interpretable_reasons.append("branch_ambiguous")

    release_tag = paired_access_tag(
        release_target_access(paired_ladder), release_target_access(decision_ladder)
    )
    if same_description:
        # unit.md: the same-description P+D_C readout is construction-determined
        # because D_C is already the exact prefix of native row E.  It is
        # recorded for replay and never enters a paired diagnostic tag.
        coordinate_tag = ACCESS_CONSTRUCTION_DETERMINED_AT_P
        coordinate_access_p: bool | None = None
    else:
        coordinate_access_p = coordinate_target_access(paired_ladder, bound=BOUND_U)
        coordinate_tag = paired_access_tag(
            coordinate_access_p, coordinate_target_access(decision_ladder, bound=BOUND_U)
        )

    at_p_plus_e = _boundary_fields(decision_ladder, owner_id=owner_id)
    at_p = _boundary_fields(
        paired_ladder, owner_id=owner_id, construction_determined=same_description
    )

    return {
        "schema_version": OWNER_ROW_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "row_kind": "crossing_boundary_primary_owner_row",
        "gt_owner_id": owner_id,
        "image_id": str(record.get("image_id")),
        "cohort": PRIMARY_COHORT,
        "stratum": str(record.get("stratum")),
        "normalized_description": record.get("normalized_description"),
        "same_description_as_e": same_description,
        "description_observability": record.get("description_observability"),
        "quarantined": quarantined,
        "replay_admitted": replay_admitted,
        "at_p_plus_e": at_p_plus_e,
        "at_p": at_p,
        "paired_transitions": build_paired_transitions(
            at_p,
            at_p_plus_e,
            coordinate_is_construction_determined=same_description,
        ),
        "displacement": (
            None
            if displacement is None
            else {
                "likelihood_displaced": displacement.likelihood_displaced,
                "likelihood_displaced_owner_id": displacement.likelihood_displaced_owner_id,
                "greedy_displaced": displacement.greedy_displaced,
                "greedy_displaced_owner_id": displacement.greedy_displaced_owner_id,
                "decoding_contradicted": displacement.decoding_contradicted,
            }
        ),
        "primary_branch": branch_u,
        "primary_branch_reasons": list(reasons_u),
        "sensitivity_branch_l": branch_l,
        "l_sensitivity_agrees_with_u": (
            None if branch_u is None or branch_l is None else branch_u == branch_l
        ),
        "tie_or_nonunique": tie_or_nonunique,
        "missing_required_fields": missing_fields,
        "interpretable": interpretable,
        "not_interpretable_reasons": not_interpretable_reasons,
        "paired_access": {
            "natural_release": {
                "tag": release_tag,
                "access_at_p": release_target_access(paired_ladder),
                "access_at_p_plus_e": release_target_access(decision_ladder),
            },
            "coordinate": {
                "tag": coordinate_tag,
                "access_at_p": coordinate_access_p,
                "access_at_p_plus_e": coordinate_target_access(
                    decision_ladder, bound=BOUND_U
                ),
            },
        },
    }


def summarize_control(record: Mapping[str, Any]) -> dict[str, Any]:
    """A control owner is described, never branched and never in a denominator."""

    assert_no_secondary_evidence(record)
    ladders = record.get("ladders")
    variants = sorted(ladders) if isinstance(ladders, Mapping) else []
    owner_id = str(record.get("gt_owner_id"))
    paired_transitions: dict[str, Any] | None = None
    if list(variants) == list(CONTROL_PAIRED_VARIANTS):
        # The timing controls carry the same two-boundary shape as the primary
        # cohort, so the identical raw transition readout is available for them.
        # It stays purely descriptive: a control is never branched or counted.
        paired_transitions = build_paired_transitions(
            _boundary_fields(ladders[CONTROL_PAIRED_VARIANTS[0]], owner_id=owner_id),
            _boundary_fields(ladders[CONTROL_PAIRED_VARIANTS[1]], owner_id=owner_id),
            coordinate_is_construction_determined=False,
        )
    return {
        "schema_version": OWNER_ROW_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "row_kind": "crossing_boundary_control_owner_row",
        "gt_owner_id": owner_id,
        "image_id": str(record.get("image_id")),
        "cohort": str(record.get("cohort")),
        "in_primary_denominator": False,
        "quarantined": bool(record.get("quarantined")),
        "replay_admitted": bool(record.get("replay_admitted")),
        "ladder_variants": variants,
        "primary_branch": None,
        "primary_branch_reasons": [
            "controls are descriptive; unit.md never assigns them a primary branch"
        ],
        "support_disposition_u_by_variant": {
            variant: support_disposition(ladders[variant], bound=BOUND_U)
            for variant in variants
        },
        "support_disposition_l_by_variant": {
            variant: support_disposition(ladders[variant], bound=BOUND_L)
            for variant in variants
        },
        "paired_transitions": paired_transitions,
    }


# ---------------------------------------------------------------------------
# 4. Cohort aggregation, gate, routing, coherence
# ---------------------------------------------------------------------------


def _counts(values: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = "null" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def component_transition_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Tabulate every raw component transition over a cohort of owner rows."""

    return {
        component: _counts(
            [row["paired_transitions"][component][key] for row in rows]
        )
        for component, key in sorted(TRANSITION_LABEL_KEY.items())
    }


def evaluate_pre_post_coherence(
    rows: Sequence[Mapping[str, Any]], *, leading_branch: str | None
) -> dict[str, Any]:
    """The explicit reading of unit.md's pre/post coherence clause.

    See ``OPERATIONAL_DEFINITIONS['pre_post_coherence']``: the clause is
    underdetermined in ``unit.md``, so the smallest checkable form is stated here
    rather than silently assumed, and only an outright contradiction closes the
    route automatically.
    """

    ladder = MECHANISM_LADDER_BY_BRANCH.get(str(leading_branch)) if leading_branch else None
    cohort = [
        row
        for row in rows
        if row["interpretable"] and row["primary_branch"] == leading_branch
    ]
    tags = (
        []
        if ladder is None
        else [str(row["paired_access"][ladder]["tag"]) for row in cohort]
    )
    determinable = [
        tag for tag in tags if tag in (ACCESS_OPENED, ACCESS_RETAINED, ACCESS_SUPPRESSED)
    ]
    if ladder is None or not determinable:
        status = COHERENCE_NOT_DETERMINABLE
    elif ACCESS_OPENED in determinable:
        status = COHERENCE_CONTRADICTED
    elif ACCESS_SUPPRESSED in determinable:
        status = COHERENCE_COHERENT_CHANGE
    else:
        status = COHERENCE_NO_PAIRED_CHANGE
    return {
        "leading_branch": leading_branch,
        "mechanism_ladder": ladder,
        "cohort_size": len(cohort),
        "tag_counts": _counts(tags),
        "determinable_tag_count": len(determinable),
        # The coarse tag is a summary; these are the raw component changes the
        # same cohort actually shows, so a coherence reading is never made on the
        # tag alone.
        "component_transition_counts": component_transition_counts(cohort),
        "status": status,
        "closes_route": status == COHERENCE_CONTRADICTED,
        "requires_adjudication": status
        in (COHERENCE_NO_PAIRED_CHANGE, COHERENCE_NOT_DETERMINABLE),
        "definition": OPERATIONAL_DEFINITIONS["pre_post_coherence"],
    }


# ---------------------------------------------------------------------------
# 4b. Conclusion-fragility slices
#
# Every function below is a read-out over the owner rows already classified
# above.  None of them assigns a branch, changes a denominator, or changes the
# decision; they exist so a reader can see how much of the single routed share
# rests on one kind of evidence.
# ---------------------------------------------------------------------------


def two_thirds_threshold(total: int) -> int:
    """The smallest count that reaches the frozen two-thirds rule over ``total``.

    ``count >= ceil(2 * total / 3)`` is exactly the scorer's own integer test
    ``count * 3 >= total * 2``; it is expressed as a threshold here only so the
    report can state how many owners of margin the route actually has.
    """

    numerator = int(total) * scorer.ROUTING_MAJORITY_NUMERATOR
    return -(-numerator // scorer.ROUTING_MAJORITY_DENOMINATOR)


def _ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(str(row["gt_owner_id"]) for row in rows)


def _cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"count": len(rows), "owner_ids": _ids(rows)}


def displaced_sub_tag_contingency(
    rows: Sequence[Mapping[str, Any]], *, branch: str = "displaced"
) -> dict[str, Any]:
    """The displaced cohort split into the four disjoint displacement sub-tag cells."""

    cohort = [row for row in rows if row["primary_branch"] == branch]
    cells: dict[str, list[Mapping[str, Any]]] = {
        "likelihood_and_greedy_displaced": [],
        "likelihood_displaced_only": [],
        "greedy_displaced_only": [],
        "neither_sub_tag": [],
    }
    for row in cohort:
        displacement = row["displacement"] or {}
        likelihood = bool(displacement.get("likelihood_displaced"))
        greedy = bool(displacement.get("greedy_displaced"))
        if likelihood and greedy:
            cells["likelihood_and_greedy_displaced"].append(row)
        elif likelihood:
            cells["likelihood_displaced_only"].append(row)
        elif greedy:
            cells["greedy_displaced_only"].append(row)
        else:
            cells["neither_sub_tag"].append(row)
    return {
        "definition": OPERATIONAL_DEFINITIONS["displaced_sub_tag_contingency"],
        "branch": branch,
        "denominator": len(cohort),
        "owner_ids": _ids(cohort),
        "cells": {name: _cell(members) for name, members in cells.items()},
    }


def greedy_only_mirror_exclusion(
    rows: Sequence[Mapping[str, Any]], *, leading_branch: str | None
) -> dict[str, Any]:
    """What the leading share becomes without the greedy-only mirror cells.

    The excluded cells are the leading branch's own deterministic interpretable
    owners that carry no likelihood displacement at all, so their whole branch
    claim rests on the coordinate-only greedy mirror.
    """

    interpretable = [row for row in rows if row["interpretable"]]
    excluded = [
        row
        for row in interpretable
        if row["primary_branch"] == leading_branch
        and bool((row["displacement"] or {}).get("greedy_displaced"))
        and not bool((row["displacement"] or {}).get("likelihood_displaced"))
    ]
    excluded_ids = {str(row["gt_owner_id"]) for row in excluded}
    retained = [
        row for row in interpretable if str(row["gt_owner_id"]) not in excluded_ids
    ]
    retained_counts = _counts([row["primary_branch"] for row in retained])
    retained_leading_count = (
        0 if leading_branch is None else retained_counts.get(str(leading_branch), 0)
    )
    threshold = two_thirds_threshold(len(retained))
    return {
        "definition": OPERATIONAL_DEFINITIONS["greedy_only_mirror_exclusion"],
        "leading_branch": leading_branch,
        "excluded_owner_ids": sorted(excluded_ids),
        "excluded_count": len(excluded_ids),
        "retained_interpretable_count": len(retained),
        "retained_branch_counts": retained_counts,
        "retained_leading_branch_count": retained_leading_count,
        "retained_two_thirds_threshold": threshold,
        "retained_reaches_two_thirds": bool(
            retained and retained_leading_count >= threshold
        ),
        "note": (
            "a sensitivity only: the routed decision is never recomputed from this "
            "exclusion"
        ),
    }


def likelihood_only_realized_target_support(
    rows: Sequence[Mapping[str, Any]], *, branch: str = "displaced"
) -> dict[str, Any]:
    """Likelihood-only displaced cells, tabulated by their ``P+E`` greedy status."""

    cohort = [
        row
        for row in rows
        if row["primary_branch"] == branch
        and bool((row["displacement"] or {}).get("likelihood_displaced"))
        and not bool((row["displacement"] or {}).get("greedy_displaced"))
    ]
    unmatched = [
        row
        for row in cohort
        if str(row["at_p_plus_e"]["greedy_status"]) == scorer.GREEDY_UNMATCHED
    ]
    return {
        "definition": OPERATIONAL_DEFINITIONS["likelihood_only_realized_target_support"],
        "branch": branch,
        "denominator": len(cohort),
        "owner_ids": _ids(cohort),
        "greedy_status_counts_at_p_plus_e": _counts(
            [row["at_p_plus_e"]["greedy_status"] for row in cohort]
        ),
        "unmatched_at_p_plus_e": _cell(unmatched),
        "note": (
            "an unmatched P+E coordinate-only greedy box strict-matches no physical owner, "
            "so these cells contribute likelihood displacement evidence and no "
            "realized-target support"
        ),
    }


def forced_dc_greedy_target_realization_at_p(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The literal forced-``D_C`` coordinate-only greedy status at ``P``, over all owners."""

    target_match = [
        row
        for row in rows
        if str(row["at_p"]["greedy_status"]) == scorer.GREEDY_TARGET_MATCH
    ]
    construction_determined = [
        row for row in target_match if bool(row["same_description_as_e"])
    ]
    return {
        "definition": OPERATIONAL_DEFINITIONS["forced_dc_greedy_target_realization_at_p"],
        "boundary_label": "P",
        "denominator": len(rows),
        "greedy_status_counts": _counts([row["at_p"]["greedy_status"] for row in rows]),
        "target_match": _cell(target_match),
        "target_match_construction_determined_at_p": _cell(construction_determined),
        "note": (
            "P is never decision-bearing in this unit; this is the realization baseline the "
            "P+E readout is read against"
        ),
    }


def greedy_displacer_identity(
    rows: Sequence[Mapping[str, Any]], crossing_e: Mapping[str, Any]
) -> dict[str, Any]:
    """Each greedy-displaced owner's displacer beside crossing ``E``'s own owner."""

    by_owner = crossing_e.get("by_owner") or {}
    resolvable = str(crossing_e.get("source")) == CROSSING_E_SOURCE_SEALED_PLAN
    cohort = [
        row
        for row in rows
        if bool((row["displacement"] or {}).get("greedy_displaced"))
    ]
    pairs: list[dict[str, Any]] = []
    for row in sorted(cohort, key=lambda value: str(value["gt_owner_id"])):
        owner_id = str(row["gt_owner_id"])
        entry = by_owner.get(owner_id) if resolvable else None
        displacing = (row["displacement"] or {}).get("greedy_displaced_owner_id")
        displacing = None if displacing is None else str(displacing)
        if entry is None:
            disposition = DISPLACER_E_NOT_DETERMINABLE
            crossing_e_owner_id = None
            strict_match_status = None
        else:
            crossing_e_owner_id = entry.get("crossing_e_owner_id")
            strict_match_status = entry.get("crossing_e_strict_match_status")
            if crossing_e_owner_id is None:
                # An unmatched E row has no physical owner at all, so no
                # displacer can be it.  This is the registry's own strict-match
                # status, never an inference from the stratum.
                disposition = DISPLACER_NOT_E
            elif displacing is not None and displacing == str(crossing_e_owner_id):
                disposition = DISPLACER_EQUALS_E
            else:
                disposition = DISPLACER_NOT_E
        pairs.append(
            {
                "gt_owner_id": owner_id,
                "stratum": row["stratum"],
                "primary_branch": row["primary_branch"],
                "interpretable": bool(row["interpretable"]),
                "displacing_owner_id": displacing,
                "crossing_e_owner_id": crossing_e_owner_id,
                "crossing_e_strict_match_status": strict_match_status,
                "disposition": disposition,
            }
        )
    dispositions = [pair["disposition"] for pair in pairs]
    return {
        "definition": OPERATIONAL_DEFINITIONS["greedy_displacer_identity"],
        "crossing_e_owner_identity_source": {
            "definition": OPERATIONAL_DEFINITIONS["crossing_e_owner_identity_source"],
            "source": crossing_e.get("source"),
            "reason": crossing_e.get("reason"),
            "plan_dir": crossing_e.get("plan_dir"),
            "cohort_registry_sha256": crossing_e.get("cohort_registry_sha256"),
        },
        "denominator": len(pairs),
        "owner_ids": [pair["gt_owner_id"] for pair in pairs],
        "pairs": pairs,
        "disposition_counts": _counts(dispositions),
        "displacer_equals_crossing_e_count": dispositions.count(DISPLACER_EQUALS_E),
        "displacer_equals_crossing_e_owner_ids": [
            pair["gt_owner_id"]
            for pair in pairs
            if pair["disposition"] == DISPLACER_EQUALS_E
        ],
        "not_determinable_count": dispositions.count(DISPLACER_E_NOT_DETERMINABLE),
        "note": (
            "identity only; this is never evidence that emitting E caused, enabled or "
            "produced the displacement"
        ),
    }


def support_bound_disagreement_cells(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Every (owner, boundary) cell whose U and L support dispositions differ."""

    cells: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: str(value["gt_owner_id"])):
        for context_role in (PAIRED_VARIANT, DECISION_VARIANT):
            boundary = row[context_role]
            support_u = boundary["support_disposition_u"]
            support_l = boundary["support_disposition_l"]
            if support_u == support_l:
                continue
            cells.append(
                {
                    "gt_owner_id": str(row["gt_owner_id"]),
                    "context_role": context_role,
                    "boundary_label": boundary["boundary_label"],
                    "context_id": boundary["context_id"],
                    "support_disposition_u": support_u,
                    "support_disposition_l": support_l,
                    "primary_branch": row["primary_branch"],
                    "interpretable": bool(row["interpretable"]),
                    "l_sensitivity_agrees_with_u": row["l_sensitivity_agrees_with_u"],
                }
            )
    branch_disagreement_ids = sorted(
        str(row["gt_owner_id"])
        for row in rows
        if row["l_sensitivity_agrees_with_u"] is False
    )
    return {
        "definition": OPERATIONAL_DEFINITIONS["support_bound_disagreement_cells"],
        "total_count": len(cells),
        "cells": cells,
        "count_by_context_role": _counts([cell["context_role"] for cell in cells]),
        "owner_ids_by_context_role": {
            context_role: sorted(
                {
                    cell["gt_owner_id"]
                    for cell in cells
                    if cell["context_role"] == context_role
                }
            )
            for context_role in (PAIRED_VARIANT, DECISION_VARIANT)
        },
        "distinct_owner_ids": sorted({cell["gt_owner_id"] for cell in cells}),
        "branch_disagreement_owner_ids": branch_disagreement_ids,
        "note": (
            "support-level cells; they are reported beside the L-bound branch "
            "disagreement and are never collapsed into it"
        ),
    }


def _native_disposition(entry: Mapping[str, Any] | None) -> str:
    if entry is None:
        return NATIVE_DISPOSITION_NOT_RESOLVABLE
    true_positive = bool(entry.get("native_true_positive"))
    false_negative = bool(entry.get("native_false_negative"))
    if true_positive and not false_negative:
        return NATIVE_DISPOSITION_TRUE_POSITIVE
    if false_negative and not true_positive:
        return NATIVE_DISPOSITION_FALSE_NEGATIVE
    return NATIVE_DISPOSITION_UNCLASSIFIED


def _coverage_audit(
    *,
    displacing_owner_id: str | None,
    boundary_index: int | None,
    census_by_owner: Mapping[str, Any],
    resolvable: bool,
) -> dict[str, Any]:
    """One owner's covered-before-boundary and native-disposition readout."""

    entry = (
        census_by_owner.get(str(displacing_owner_id))
        if resolvable and displacing_owner_id is not None
        else None
    )
    if entry is None or boundary_index is None:
        return {
            "native_strict_match_row_indices": None,
            "native_strict_match_row_indices_before_boundary": None,
            "covered_before_boundary": None,
            "normalized_description": None if entry is None else entry.get(
                "normalized_description"
            ),
            "native_disposition": _native_disposition(entry),
        }
    indices = [int(value) for value in entry["native_strict_match_row_indices"]]
    before = [value for value in indices if value < int(boundary_index)]
    return {
        "native_strict_match_row_indices": indices,
        "native_strict_match_row_indices_before_boundary": before,
        "covered_before_boundary": bool(before),
        "normalized_description": entry.get("normalized_description"),
        "native_disposition": _native_disposition(entry),
    }


def _characterize_sub_tag(
    rows: Sequence[Mapping[str, Any]],
    *,
    sub_tag: str,
    owner_id_key: str,
    plan_facts: Mapping[str, Any],
    census_facts: Mapping[str, Any],
) -> dict[str, Any]:
    """One displacement sub-tag's displacers, characterized cell by cell."""

    plan_resolvable = str(plan_facts.get("source")) == CROSSING_E_SOURCE_SEALED_PLAN
    census_resolvable = (
        str(census_facts.get("source")) == CENSUS_OWNER_SOURCE_SEALED_LINEAGE
    )
    plan_by_owner = plan_facts.get("by_owner") or {}
    census_by_owner = census_facts.get("by_owner") or {}

    pairs: list[dict[str, Any]] = []
    for row in sorted(
        (row for row in rows if bool((row["displacement"] or {}).get(sub_tag))),
        key=lambda value: str(value["gt_owner_id"]),
    ):
        owner_id = str(row["gt_owner_id"])
        displacing = (row["displacement"] or {}).get(owner_id_key)
        displacing = None if displacing is None else str(displacing)
        plan_entry = plan_by_owner.get(owner_id) if plan_resolvable else None
        boundary_index = None if plan_entry is None else plan_entry.get("p_boundary_index")
        audit = _coverage_audit(
            displacing_owner_id=displacing,
            boundary_index=boundary_index,
            census_by_owner=census_by_owner,
            resolvable=census_resolvable,
        )
        target_description = row["normalized_description"]
        displacer_description = audit["normalized_description"]
        pairs.append(
            {
                "gt_owner_id": owner_id,
                "displacing_owner_id": displacing,
                "p_boundary_index": boundary_index,
                "target_normalized_description": target_description,
                "displacer_normalized_description": displacer_description,
                "same_normalized_description": (
                    None
                    if displacer_description is None
                    else displacer_description == target_description
                ),
                "native_strict_match_row_indices": audit[
                    "native_strict_match_row_indices"
                ],
                "native_strict_match_row_indices_before_p": audit[
                    "native_strict_match_row_indices_before_boundary"
                ],
                "covered_before_p": audit["covered_before_boundary"],
                "displacer_native_disposition": audit["native_disposition"],
            }
        )

    def cell(predicate) -> dict[str, Any]:
        selected = [pair for pair in pairs if predicate(pair)]
        return {
            "count": len(selected),
            "gt_owner_ids": [pair["gt_owner_id"] for pair in selected],
            "displacing_owner_ids": [pair["displacing_owner_id"] for pair in selected],
        }

    return {
        "sub_tag": sub_tag,
        "denominator": len(pairs),
        "pairs": pairs,
        "same_normalized_description": {
            **cell(lambda pair: pair["same_normalized_description"] is True),
            "claim_guard": OPERATIONAL_DEFINITIONS[
                "displacer_same_description_is_construction_determined"
            ],
        },
        "covered_before_p": cell(lambda pair: pair["covered_before_p"] is True),
        "uncovered_before_p": cell(lambda pair: pair["covered_before_p"] is False),
        "coverage_not_determinable": cell(lambda pair: pair["covered_before_p"] is None),
        "native_disposition_counts": _counts(
            [pair["displacer_native_disposition"] for pair in pairs]
        ),
        "displacer_native_true_positive": cell(
            lambda pair: pair["displacer_native_disposition"]
            == NATIVE_DISPOSITION_TRUE_POSITIVE
        ),
        "displacer_native_false_negative": cell(
            lambda pair: pair["displacer_native_disposition"]
            == NATIVE_DISPOSITION_FALSE_NEGATIVE
        ),
    }


def displacer_characterization(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan_facts: Mapping[str, Any],
    census_facts: Mapping[str, Any],
) -> dict[str, Any]:
    """What the displacing owners are, for both displacement sub-tags."""

    return {
        "definition": OPERATIONAL_DEFINITIONS["displacer_characterization"],
        "covered_before_p_definition": OPERATIONAL_DEFINITIONS["covered_before_p"],
        "census_owner_registry_source": {
            "source": census_facts.get("source"),
            "reason": census_facts.get("reason"),
            "census_run_root": census_facts.get("census_run_root"),
            "owner_registry_path": census_facts.get("owner_registry_path"),
            "owner_registry_sha256": census_facts.get("owner_registry_sha256"),
        },
        "by_sub_tag": {
            sub_tag: _characterize_sub_tag(
                rows,
                sub_tag=sub_tag,
                owner_id_key=owner_id_key,
                plan_facts=plan_facts,
                census_facts=census_facts,
            )
            for sub_tag, owner_id_key in (
                ("greedy_displaced", "greedy_displaced_owner_id"),
                ("likelihood_displaced", "likelihood_displaced_owner_id"),
            )
        },
        "note": (
            "descriptive: a displacer's coverage state and native disposition are read-outs "
            "about that owner, never evidence that it caused the target's omission"
        ),
    }


def timing_control_competitor_coverage(
    control_rows: Sequence[Mapping[str, Any]],
    *,
    plan_facts: Mapping[str, Any],
    census_facts: Mapping[str, Any],
) -> dict[str, Any]:
    """The covered-before-boundary audit for the timing controls' own competitors.

    Restricted to the controls whose target-minus-competitor margin sign turns
    from positive to negative across the control pair -- the same suppression the
    primary cohort is read for -- so the two can be compared side by side without
    the comparison ever being treated as a matched counterfactual.
    """

    plan_resolvable = str(plan_facts.get("source")) == CROSSING_E_SOURCE_SEALED_PLAN
    census_resolvable = (
        str(census_facts.get("source")) == CENSUS_OWNER_SOURCE_SEALED_LINEAGE
    )
    control_by_owner = plan_facts.get("control_by_owner") or {}
    census_by_owner = census_facts.get("by_owner") or {}

    cells: list[dict[str, Any]] = []
    for row in sorted(
        (
            row
            for row in control_rows
            if str(row["cohort"]) == TIMING_CONTROL_COHORT
            and row.get("paired_transitions") is not None
            and str(
                row["paired_transitions"]["target_minus_competitor_margin"][
                    "sign_transition"
                ]
            )
            == "positive->negative"
        ),
        key=lambda value: str(value["gt_owner_id"]),
    ):
        owner_id = str(row["gt_owner_id"])
        competitor = row["paired_transitions"]["best_competitor_owner_id"]["at_p_plus_e"]
        competitor = None if competitor is None else str(competitor)
        plan_entry = control_by_owner.get(owner_id) if plan_resolvable else None
        boundary_index = (
            None if plan_entry is None else plan_entry.get("control_boundary_index")
        )
        audit = _coverage_audit(
            displacing_owner_id=competitor,
            boundary_index=boundary_index,
            census_by_owner=census_by_owner,
            resolvable=census_resolvable,
        )
        cells.append(
            {
                "gt_owner_id": owner_id,
                "best_competitor_owner_id": competitor,
                "control_boundary_index": boundary_index,
                "native_strict_match_row_indices": audit[
                    "native_strict_match_row_indices"
                ],
                "native_strict_match_row_indices_before_control_boundary": audit[
                    "native_strict_match_row_indices_before_boundary"
                ],
                "covered_before_control_boundary": audit["covered_before_boundary"],
                "competitor_native_disposition": audit["native_disposition"],
            }
        )

    def cell(predicate) -> dict[str, Any]:
        selected = [value for value in cells if predicate(value)]
        return {
            "count": len(selected),
            "gt_owner_ids": [value["gt_owner_id"] for value in selected],
            "best_competitor_owner_ids": [
                value["best_competitor_owner_id"] for value in selected
            ],
        }

    return {
        "definition": OPERATIONAL_DEFINITIONS["timing_control_competitor_coverage"],
        "cohort": TIMING_CONTROL_COHORT,
        "in_primary_denominator": False,
        "in_routing_denominator": False,
        "selection": "target_minus_competitor_margin sign transition positive->negative",
        "denominator": len(cells),
        "cells": cells,
        "covered_before_control_boundary": cell(
            lambda value: value["covered_before_control_boundary"] is True
        ),
        "uncovered_before_control_boundary": cell(
            lambda value: value["covered_before_control_boundary"] is False
        ),
        "coverage_not_determinable": cell(
            lambda value: value["covered_before_control_boundary"] is None
        ),
        "competitor_native_disposition_counts": _counts(
            [value["competitor_native_disposition"] for value in cells]
        ),
        "note": (
            "descriptive and confounded: this supports generic within-category scheduling "
            "instability, not crossing-E specificity, and no causal contrast with the "
            "primary cohort is claimed or supported"
        ),
    }


def timing_control_suppression(
    control_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The same four raw suppressions, read over the disjoint timing controls.

    This is the confound exhibit, not a counterfactual: the timing controls are
    matched on neither description identity nor route tier, so they show only
    that these suppressions also occur at a boundary that is not a crossing.
    """

    cohort = [
        row
        for row in control_rows
        if str(row["cohort"]) == TIMING_CONTROL_COHORT
        and row.get("paired_transitions") is not None
    ]
    skipped = [
        str(row["gt_owner_id"])
        for row in control_rows
        if str(row["cohort"]) == TIMING_CONTROL_COHORT
        and row.get("paired_transitions") is None
    ]
    readouts: dict[str, list[str]] = {
        CONTROL_SUPPRESSION_RANK_WORSENED: [],
        CONTROL_SUPPRESSION_U_SUPPORT_LOST: [],
        CONTROL_SUPPRESSION_MARGIN_POSITIVE_TO_NEGATIVE: [],
        CONTROL_SUPPRESSION_GREEDY_TARGET_LOST: [],
    }
    for row in sorted(cohort, key=lambda value: str(value["gt_owner_id"])):
        owner_id = str(row["gt_owner_id"])
        transitions = row["paired_transitions"]
        if str(transitions["target_rank"]["change"]) == CHANGE_WORSENED:
            readouts[CONTROL_SUPPRESSION_RANK_WORSENED].append(owner_id)
        support = transitions["support_disposition_u"]
        if (
            str(support["at_p"]) == scorer.SUPPORT_SUPPORTED
            and str(support["at_p_plus_e"]) != scorer.SUPPORT_SUPPORTED
        ):
            readouts[CONTROL_SUPPRESSION_U_SUPPORT_LOST].append(owner_id)
        margin = transitions["target_minus_competitor_margin"]
        if str(margin["sign_transition"]) == "positive->negative":
            readouts[CONTROL_SUPPRESSION_MARGIN_POSITIVE_TO_NEGATIVE].append(owner_id)
        greedy = transitions["greedy_status"]
        if (
            str(greedy["at_p"]) == scorer.GREEDY_TARGET_MATCH
            and str(greedy["at_p_plus_e"]) != scorer.GREEDY_TARGET_MATCH
        ):
            readouts[CONTROL_SUPPRESSION_GREEDY_TARGET_LOST].append(owner_id)
    return {
        "definition": OPERATIONAL_DEFINITIONS["timing_control_suppression"],
        "cohort": TIMING_CONTROL_COHORT,
        "in_primary_denominator": False,
        "in_routing_denominator": False,
        "denominator": len(cohort),
        "owner_ids": _ids(cohort),
        "owners_without_a_paired_readout": sorted(skipped),
        "readouts": {
            name: {"count": len(owner_ids), "owner_ids": owner_ids}
            for name, owner_ids in sorted(readouts.items())
        },
        "component_transition_counts": component_transition_counts(cohort),
        "note": (
            "descriptive and confounded: suppression is not unique to crossing owners, and "
            "no causal contrast between this cohort and the crossing cohort is claimed or "
            "supported"
        ),
    }


def assert_crossing_e_matches_strata(
    rows: Sequence[Mapping[str, Any]], crossing_e: Mapping[str, Any]
) -> None:
    """A resolved crossing-``E`` identity must belong to exactly this cohort.

    The registry is reached through the merge receipt's own sealed plan path, so
    it is already the right plan; this proves it is also the right *cohort* by
    requiring every primary owner to be present with the stratum the merge sealed.
    """

    if str(crossing_e.get("source")) != CROSSING_E_SOURCE_SEALED_PLAN:
        return
    by_owner = crossing_e.get("by_owner") or {}
    for row in rows:
        owner_id = str(row["gt_owner_id"])
        entry = by_owner.get(owner_id)
        if entry is None:
            _fail(
                f"the sealed plan cohort registry carries no row for primary owner "
                f"{owner_id!r}; crossing E identity would be guessed"
            )
        registry_stratum = entry.get("stratum")
        if registry_stratum is not None and str(registry_stratum) != str(row["stratum"]):
            _fail(
                f"the sealed plan cohort registry puts owner {owner_id!r} in stratum "
                f"{registry_stratum!r} while the merged evidence seals {row['stratum']!r}"
            )
        matched_owner = entry.get("crossing_e_owner_id")
        if str(row["stratum"]) == merge.UNMATCHED_E_STRATUM and matched_owner is not None:
            _fail(
                f"owner {owner_id!r} is sealed unmatched-E while the plan cohort registry "
                f"names {matched_owner!r} as crossing E's owner"
            )


def build_conclusion_fragility(
    rows: Sequence[Mapping[str, Any]],
    *,
    control_rows: Sequence[Mapping[str, Any]],
    routing_evaluated: Mapping[str, Any] | None,
    leading_branch: str | None,
    plan_facts: Mapping[str, Any],
    census_facts: Mapping[str, Any],
) -> dict[str, Any]:
    """Every fragility slice the routed share has to be read against."""

    interpretable_count = sum(1 for row in rows if row["interpretable"])
    leading_count = (
        0 if routing_evaluated is None else int(routing_evaluated["leading_branch_count"])
    )
    threshold = two_thirds_threshold(interpretable_count)
    return {
        "definition": OPERATIONAL_DEFINITIONS["conclusion_fragility"],
        "changes_branch_assignment_or_decision": False,
        "route_margin": {
            "definition": OPERATIONAL_DEFINITIONS["two_thirds_threshold"],
            "leading_branch": leading_branch,
            "leading_branch_count": leading_count,
            "routing_denominator_count": interpretable_count,
            "two_thirds_threshold": threshold,
            "owners_of_margin": leading_count - threshold,
        },
        "displaced_sub_tag_contingency": displaced_sub_tag_contingency(rows),
        "greedy_only_mirror_exclusion": greedy_only_mirror_exclusion(
            rows, leading_branch=leading_branch
        ),
        "likelihood_only_realized_target_support": (
            likelihood_only_realized_target_support(rows)
        ),
        "forced_dc_greedy_target_realization_at_p": (
            forced_dc_greedy_target_realization_at_p(rows)
        ),
        "greedy_displacer_identity": greedy_displacer_identity(rows, plan_facts),
        "displacer_characterization": displacer_characterization(
            rows, plan_facts=plan_facts, census_facts=census_facts
        ),
        "support_bound_disagreement_cells": support_bound_disagreement_cells(rows),
        "timing_control_suppression": timing_control_suppression(control_rows),
        "timing_control_competitor_coverage": timing_control_competitor_coverage(
            control_rows, plan_facts=plan_facts, census_facts=census_facts
        ),
    }


def build_report(
    *,
    merged_dir: Path,
    merge_receipt: Mapping[str, Any],
    file_sha256: Mapping[str, str],
    primary_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    plan_facts: Mapping[str, Any],
    census_facts: Mapping[str, Any],
) -> dict[str, Any]:
    """Every frozen denominator, share, sensitivity and the stop-rule decision."""

    matched = [row for row in primary_rows if row["stratum"] == merge.MATCHED_E_STRATUM]
    unmatched = [
        row for row in primary_rows if row["stratum"] == merge.UNMATCHED_E_STRATUM
    ]
    denominators = scorer.validate_primary_cohort_counts(
        u_count=len(primary_rows),
        l_count=int(merge_receipt["cohort_denominators"]["l_count"]),
        same_context_ul_count=int(
            merge_receipt["cohort_denominators"]["same_context_ul_count"]
        ),
        matched_e_count=len(matched),
        unmatched_e_count=len(unmatched),
    )

    records = [
        scorer.OwnerRecord(
            owner_id=row["gt_owner_id"],
            stratum=row["stratum"],
            interpretable=bool(row["interpretable"]),
            branch=row["primary_branch"],
        )
        for row in primary_rows
    ]
    gate = scorer.evaluate_interpretability_gate(records)
    interpretable_rows = [row for row in primary_rows if row["interpretable"]]

    decoding_contradicted = sorted(
        row["gt_owner_id"]
        for row in primary_rows
        if (row["displacement"] or {}).get("decoding_contradicted")
    )
    decoding_contradicted_interpretable = sorted(
        row["gt_owner_id"]
        for row in interpretable_rows
        if (row["displacement"] or {}).get("decoding_contradicted")
    )

    routing: dict[str, Any] | None = None
    if gate["passed"]:
        routing = scorer.evaluate_branch_routing(records, decoding_contradicted)
        decision = DECISION_ROUTE if routing["status"] == "routed" else DECISION_CLOSE
        leading_branch = routing["leading_branch"]
    else:
        decision = DECISION_GATE_FAILED
        leading_branch = None

    coherence = evaluate_pre_post_coherence(primary_rows, leading_branch=leading_branch)
    if decision == DECISION_ROUTE and coherence["closes_route"]:
        decision = DECISION_CLOSE

    assert_crossing_e_matches_strata(primary_rows, plan_facts)
    fragility = build_conclusion_fragility(
        primary_rows,
        control_rows=control_rows,
        routing_evaluated=routing,
        leading_branch=leading_branch,
        plan_facts=plan_facts,
        census_facts=census_facts,
    )

    quarantined = sorted(row["gt_owner_id"] for row in primary_rows if row["quarantined"])
    if len(quarantined) > merge.MAX_PRIMARY_QUARANTINES:
        _fail(
            f"{len(quarantined)} primary owners are quarantined; more than "
            f"{merge.MAX_PRIMARY_QUARANTINES} stops this unit before interpretation"
        )

    controls_by_cohort: dict[str, list[Mapping[str, Any]]] = {}
    for row in control_rows:
        controls_by_cohort.setdefault(str(row["cohort"]), []).append(row)

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "merged_dir": str(merged_dir),
        "merge_receipt_content_sha256": str(merge_receipt["receipt_content_sha256"]),
        "runtime_identity_sha256": str(merge_receipt["runtime_identity_sha256"]),
        "input_file_sha256": dict(sorted(file_sha256.items())),
        "operational_definitions": dict(sorted(OPERATIONAL_DEFINITIONS.items())),
        "not_claimed": list(NOT_CLAIMED),
        "primary_cohort": {
            "cohort": PRIMARY_COHORT,
            "denominator": len(primary_rows),
            "frozen_denominators": dict(denominators),
            "matched_e_count": len(matched),
            "unmatched_e_count": len(unmatched),
            "exact_same_context_u_and_l_count": denominators["same_context_ul_count"],
            "l_bound_crossing_count": denominators["l_count"],
            "same_description_count": sum(
                1 for row in primary_rows if row["same_description_as_e"]
            ),
            "different_description_count": sum(
                1 for row in primary_rows if not row["same_description_as_e"]
            ),
            "quarantined_owner_ids": quarantined,
            "maximum_primary_quarantines": merge.MAX_PRIMARY_QUARANTINES,
        },
        "branches": {
            "branch_counts_all_primary_owners": _counts(
                [row["primary_branch"] for row in primary_rows]
            ),
            "branch_counts_interpretable": _counts(
                [row["primary_branch"] for row in interpretable_rows]
            ),
            "branch_counts_by_stratum": {
                merge.MATCHED_E_STRATUM: _counts(
                    [row["primary_branch"] for row in matched if row["interpretable"]]
                ),
                merge.UNMATCHED_E_STRATUM: _counts(
                    [row["primary_branch"] for row in unmatched if row["interpretable"]]
                ),
            },
            "branch_counts_by_description": {
                "same_description": _counts(
                    [
                        row["primary_branch"]
                        for row in interpretable_rows
                        if row["same_description_as_e"]
                    ]
                ),
                "different_description": _counts(
                    [
                        row["primary_branch"]
                        for row in interpretable_rows
                        if not row["same_description_as_e"]
                    ]
                ),
            },
            "displacement_sub_tags": {
                "likelihood_displaced": sum(
                    1
                    for row in primary_rows
                    if (row["displacement"] or {}).get("likelihood_displaced")
                ),
                "greedy_displaced": sum(
                    1
                    for row in primary_rows
                    if (row["displacement"] or {}).get("greedy_displaced")
                ),
            },
        },
        "interpretability_gate": dict(gate),
        "routing": {
            "primary_cohort_denominator": len(primary_rows),
            "routing_denominator": "deterministic_interpretable_owners",
            "routing_denominator_count": len(interpretable_rows),
            "majority_rule": (
                f"{scorer.ROUTING_MAJORITY_NUMERATOR}/"
                f"{scorer.ROUTING_MAJORITY_DENOMINATOR} of the deterministic interpretable "
                "owners"
            ),
            "evaluated": routing,
        },
        "decoding_contradicted_sensitivity": {
            "definition": (
                "likelihood_displaced is true while the coordinate-only greedy box still "
                "strict-matches the target"
            ),
            "owner_ids": decoding_contradicted,
            "count": len(decoding_contradicted),
            "interpretable_owner_ids": decoding_contradicted_interpretable,
            "interpretable_count": len(decoding_contradicted_interpretable),
            "split_rule": (
                "if excluding these cells would change whether a branch reaches two thirds, "
                "the result is split and no successor is routed"
            ),
        },
        "l_bound_sensitivity": {
            "bound": BOUND_L,
            "branch_counts_interpretable": _counts(
                [row["sensitivity_branch_l"] for row in interpretable_rows]
            ),
            "disagreeing_owner_ids": sorted(
                row["gt_owner_id"]
                for row in primary_rows
                if row["l_sensitivity_agrees_with_u"] is False
            ),
            "note": "L never changes the U branch; it is reported beside it",
        },
        "paired_access_tags": {
            "natural_release": _counts(
                [row["paired_access"]["natural_release"]["tag"] for row in primary_rows]
            ),
            "coordinate": _counts(
                [row["paired_access"]["coordinate"]["tag"] for row in primary_rows]
            ),
            "note": (
                "descriptive local prefix interaction; never a counterfactual claim that the "
                "owner would otherwise have been emitted"
            ),
        },
        "paired_component_transitions": {
            "definition": OPERATIONAL_DEFINITIONS["paired_component_transitions"],
            "all_primary_owners": component_transition_counts(primary_rows),
            "interpretable_owners": component_transition_counts(interpretable_rows),
            "by_branch": {
                branch: component_transition_counts(
                    [row for row in interpretable_rows if row["primary_branch"] == branch]
                )
                for branch in sorted(
                    {
                        str(row["primary_branch"])
                        for row in interpretable_rows
                        if row["primary_branch"] is not None
                    }
                )
            },
        },
        "pre_post_coherence": coherence,
        "conclusion_fragility": fragility,
        "controls": {
            "in_primary_denominator": False,
            "note": (
                "timing controls are descriptive because timing, description identity and "
                "route tier are entangled; TP replay controls are the coordinate replay "
                "calibration reference"
            ),
            "cohorts": {
                cohort: {
                    "owner_count": len(rows),
                    "owner_ids": sorted(str(row["gt_owner_id"]) for row in rows),
                    "quarantined_owner_count": sum(
                        1 for row in rows if row["quarantined"]
                    ),
                }
                for cohort, rows in sorted(controls_by_cohort.items())
            },
        },
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# 5. Rendering
# ---------------------------------------------------------------------------


def _render_ids(owner_ids: Sequence[Any]) -> str:
    return ", ".join(f"`{value}`" for value in owner_ids) if owner_ids else "none"


def _render_conclusion_fragility(fragility: Mapping[str, Any]) -> list[str]:
    """The fragility slices, plus the short reading a decision-maker needs."""

    margin = fragility["route_margin"]
    contingency = fragility["displaced_sub_tag_contingency"]
    cells = contingency["cells"]
    mirror = fragility["greedy_only_mirror_exclusion"]
    likelihood_only = fragility["likelihood_only_realized_target_support"]
    realization = fragility["forced_dc_greedy_target_realization_at_p"]
    identity = fragility["greedy_displacer_identity"]
    characterization = fragility["displacer_characterization"]
    disagreement = fragility["support_bound_disagreement_cells"]
    control = fragility["timing_control_suppression"]
    control_coverage = fragility["timing_control_competitor_coverage"]

    lines: list[str] = ["## Conclusion fragility", ""]
    lines.append(
        f"- route margin: leading `{margin['leading_branch']}` "
        f"{margin['leading_branch_count']}/{margin['routing_denominator_count']} against a "
        f"two-thirds threshold of {margin['two_thirds_threshold']}, i.e. "
        f"{margin['owners_of_margin']} owner(s) of margin"
    )
    lines.append("")

    lines.append(
        f"### Displaced sub-tag contingency (n={contingency['denominator']})"
    )
    lines.append("")
    lines.append("| cell | count | owner IDs |")
    lines.append("| --- | --- | --- |")
    for name in (
        "likelihood_and_greedy_displaced",
        "likelihood_displaced_only",
        "greedy_displaced_only",
        "neither_sub_tag",
    ):
        cell = cells[name]
        lines.append(f"| {name} | {cell['count']} | {_render_ids(cell['owner_ids'])} |")
    lines.append("")

    lines.append(
        f"- greedy-only mirror cells: {mirror['excluded_count']} "
        f"({_render_ids(mirror['excluded_owner_ids'])})"
    )
    lines.append(
        f"- excluding them leaves {mirror['retained_leading_branch_count']}/"
        f"{mirror['retained_interpretable_count']} `{mirror['leading_branch']}` against a "
        f"threshold of {mirror['retained_two_thirds_threshold']}; reaches two thirds: "
        f"{mirror['retained_reaches_two_thirds']}"
    )
    lines.append(
        f"- likelihood-only cells unmatched at P+E: "
        f"{likelihood_only['unmatched_at_p_plus_e']['count']}/"
        f"{likelihood_only['denominator']} "
        f"({_render_ids(likelihood_only['unmatched_at_p_plus_e']['owner_ids'])}); these "
        "contribute likelihood displacement and no realized-target support"
    )
    lines.append(
        f"- forced-D_C coordinate-only greedy target match at P: "
        f"{realization['target_match']['count']}/{realization['denominator']} "
        f"({_render_ids(realization['target_match']['owner_ids'])}); statuses "
        f"{realization['greedy_status_counts']}"
    )
    lines.append("")

    lines.append(f"### Greedy displacer identity (n={identity['denominator']})")
    lines.append("")
    lines.append(
        f"- crossing E owner source: `{identity['crossing_e_owner_identity_source']['source']}`"
    )
    lines.append(
        f"- displacer equals crossing E's owner: "
        f"{identity['displacer_equals_crossing_e_count']}/{identity['denominator']} "
        f"({_render_ids(identity['displacer_equals_crossing_e_owner_ids'])}); not "
        f"determinable: {identity['not_determinable_count']}"
    )
    lines.append("")
    lines.append("| crossing owner | displacing owner | crossing E owner | disposition |")
    lines.append("| --- | --- | --- | --- |")
    for pair in identity["pairs"]:
        crossing_e_owner = (
            f"`{pair['crossing_e_owner_id']}`"
            if pair["crossing_e_owner_id"] is not None
            else f"none ({pair['crossing_e_strict_match_status']})"
        )
        lines.append(
            f"| `{pair['gt_owner_id']}` | `{pair['displacing_owner_id']}` | "
            f"{crossing_e_owner} | {pair['disposition']} |"
        )
    lines.append("")

    lines.append("### Displacer characterization")
    lines.append("")
    lines.append(
        f"- census owner registry source: "
        f"`{characterization['census_owner_registry_source']['source']}`"
    )
    for sub_tag, block in characterization["by_sub_tag"].items():
        lines.append(
            f"- `{sub_tag}` (n={block['denominator']}): covered before P "
            f"{block['covered_before_p']['count']}, uncovered before P "
            f"{block['uncovered_before_p']['count']} "
            f"({_render_ids(block['uncovered_before_p']['displacing_owner_ids'])}); "
            f"displacer native disposition {block['native_disposition_counts']}"
        )
        lines.append(
            f"  - same normalized description as the target: "
            f"{block['same_normalized_description']['count']}/{block['denominator']} -- "
            "construction-determined by the same-category competitor family, an audit "
            "property only and never evidence of a same-description mechanism"
        )
    lines.append("")

    lines.append(
        f"### U/L support-disposition disagreement cells "
        f"(n={disagreement['total_count']})"
    )
    lines.append("")
    lines.append("| owner | context role | boundary | U | L | branch |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for cell in disagreement["cells"]:
        lines.append(
            f"| `{cell['gt_owner_id']}` | {cell['context_role']} | "
            f"{cell['boundary_label']} | {cell['support_disposition_u']} | "
            f"{cell['support_disposition_l']} | {cell['primary_branch']} |"
        )
    lines.append("")
    lines.append(
        f"- by context role: {disagreement['count_by_context_role']}; L-bound branch "
        f"disagreement: {_render_ids(disagreement['branch_disagreement_owner_ids'])}"
    )
    lines.append(
        "- these are support-level cells, reported beside and never collapsed into the "
        "branch disagreement"
    )
    lines.append("")

    lines.append(
        f"### Timing-control suppression (descriptive, n={control['denominator']}, "
        "never in any denominator)"
    )
    lines.append("")
    lines.append("| readout | count | owner IDs |")
    lines.append("| --- | --- | --- |")
    for name, readout in control["readouts"].items():
        lines.append(
            f"| {name} | {readout['count']} | {_render_ids(readout['owner_ids'])} |"
        )
    lines.append("")
    lines.append(
        f"- of the {control_coverage['denominator']} controls whose target-minus-competitor "
        f"margin turns positive->negative, "
        f"{control_coverage['uncovered_before_control_boundary']['count']} displace toward "
        "an owner uncovered at the control boundary "
        f"({_render_ids(control_coverage['uncovered_before_control_boundary']['best_competitor_owner_ids'])}); "
        f"competitor native disposition {control_coverage['competitor_native_disposition_counts']}"
    )
    lines.append(
        "- descriptive and confounded: suppression is not unique to crossing owners, this "
        "supports generic within-category scheduling instability rather than crossing-E "
        "specificity, and no causal contrast is claimed"
    )
    lines.append("")

    lines.append("### Reading")
    lines.append("")
    lines.append(
        f"- the route is threshold-crossing: `{margin['leading_branch']}` reaches "
        f"{margin['leading_branch_count']}/{margin['routing_denominator_count']} against a "
        f"threshold of {margin['two_thirds_threshold']}, so {margin['owners_of_margin']} "
        "owner(s) separate routing from split"
    )
    lines.append(
        f"- it is directionally stable to excluding the {mirror['excluded_count']} "
        f"greedy-only mirror cells: {mirror['retained_leading_branch_count']}/"
        f"{mirror['retained_interpretable_count']} still reaches two thirds "
        f"({mirror['retained_reaches_two_thirds']})"
    )
    lines.append(
        "- the displaced evidence is heterogeneous, not one mechanism: "
        f"{cells['likelihood_and_greedy_displaced']['count']} cells carry both sub-tags, "
        f"{cells['likelihood_displaced_only']['count']} carry likelihood only and "
        f"{cells['greedy_displaced_only']['count']} carry the greedy mirror only"
    )
    lines.append(
        "- there is no evidence that crossing E itself is the displacer: "
        f"{identity['displacer_equals_crossing_e_count']}/{identity['denominator']} greedy "
        "displacers equal crossing E's own owner, and this is an identity readout with no "
        "causal claim either way"
    )
    lines.append("")
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    primary = report["primary_cohort"]
    routing = report["routing"]
    gate = report["interpretability_gate"]

    lines.append(f"# {UNIT_ID}")
    lines.append("")
    lines.append(f"Merged evidence: `{report['merged_dir']}`")
    lines.append("")
    lines.append(f"**Decision: `{report['decision']}`**")
    lines.append("")

    lines.append("## Primary cohort")
    lines.append("")
    lines.append(f"- U-bound crossing owners (primary denominator): {primary['denominator']}")
    lines.append(f"- matched-E: {primary['matched_e_count']}")
    lines.append(f"- unmatched-E: {primary['unmatched_e_count']}")
    lines.append(
        f"- exact same-context U&L sensitivity: {primary['exact_same_context_u_and_l_count']}"
    )
    lines.append(f"- L-bound crossing sensitivity: {primary['l_bound_crossing_count']}")
    lines.append(
        f"- same description as E: {primary['same_description_count']}; "
        f"different description: {primary['different_description_count']}"
    )
    lines.append(
        f"- quarantined primary owners: {len(primary['quarantined_owner_ids'])} "
        f"(maximum {primary['maximum_primary_quarantines']})"
    )
    lines.append("")

    lines.append("## Branches")
    lines.append("")
    lines.append("| branch | all 26 primary owners | deterministic interpretable |")
    lines.append("| --- | --- | --- |")
    all_counts = report["branches"]["branch_counts_all_primary_owners"]
    interp_counts = report["branches"]["branch_counts_interpretable"]
    for branch in sorted(set(all_counts) | set(interp_counts)):
        lines.append(
            f"| {branch} | {all_counts.get(branch, 0)} | {interp_counts.get(branch, 0)} |"
        )
    lines.append("")
    for stratum, counts in report["branches"]["branch_counts_by_stratum"].items():
        lines.append(f"- interpretable `{stratum}`: {counts}")
    for label, counts in report["branches"]["branch_counts_by_description"].items():
        lines.append(f"- interpretable `{label}`: {counts}")
    lines.append(f"- displacement sub-tags: {report['branches']['displacement_sub_tags']}")
    lines.append("")

    lines.append("## Interpretability gate")
    lines.append("")
    lines.append(
        f"- interpretable: {gate['interpretable_count']} / {primary['denominator']} "
        f"(minimum {gate['minimum_interpretable']})"
    )
    lines.append(
        f"- matched-E interpretable: {gate['matched_e_interpretable_count']} "
        f"(minimum {gate['minimum_matched_e']})"
    )
    lines.append(
        f"- unmatched-E interpretable: {gate['unmatched_e_interpretable_count']} "
        f"(minimum {gate['minimum_unmatched_e']})"
    )
    lines.append(f"- passed: {gate['passed']}")
    lines.append("")

    lines.append("## Routing")
    lines.append("")
    lines.append(f"- primary cohort denominator: {routing['primary_cohort_denominator']}")
    lines.append(
        f"- routing denominator ({routing['routing_denominator']}): "
        f"{routing['routing_denominator_count']}"
    )
    lines.append(f"- majority rule: {routing['majority_rule']}")
    evaluated = routing["evaluated"]
    if evaluated is None:
        lines.append("- routing was not evaluated: the interpretability gate did not pass")
    else:
        lines.append(f"- status: {evaluated['status']}")
        lines.append(f"- leading branch: {evaluated['leading_branch']} "
                     f"({evaluated['leading_branch_count']}/{evaluated['interpretable_count']})")
        lines.append(f"- routed branch: {evaluated['routed_branch']}")
        sensitivity = evaluated["decoding_contradicted_sensitivity"]
        if sensitivity is not None:
            lines.append(
                f"- decoding-contradicted exclusion: retained "
                f"{sensitivity['retained_interpretable_count']} owners, leading "
                f"{sensitivity['retained_leading_branch']}, reaches two thirds "
                f"{sensitivity['retained_reaches_two_thirds']}"
            )
    lines.append("")

    contradicted = report["decoding_contradicted_sensitivity"]
    lines.append("## Decoding-contradicted likelihood displacement")
    lines.append("")
    lines.append(f"- count: {contradicted['count']} ({contradicted['owner_ids']})")
    lines.append(f"- interpretable count: {contradicted['interpretable_count']}")
    lines.append("")

    lines.append("## L-bound sensitivity")
    lines.append("")
    lines.append(
        f"- interpretable branch counts under L: "
        f"{report['l_bound_sensitivity']['branch_counts_interpretable']}"
    )
    lines.append(
        f"- owners whose L branch differs from U: "
        f"{report['l_bound_sensitivity']['disagreeing_owner_ids']}"
    )
    lines.append("")

    lines.append("## Paired P -> P+E target-access tags")
    lines.append("")
    lines.append(f"- natural release: {report['paired_access_tags']['natural_release']}")
    lines.append(f"- coordinate: {report['paired_access_tags']['coordinate']}")
    coherence = report["pre_post_coherence"]
    lines.append(
        f"- pre/post coherence: `{coherence['status']}` on the "
        f"`{coherence['mechanism_ladder']}` ladder of the `{coherence['leading_branch']}` "
        f"cohort (n={coherence['cohort_size']}, tags={coherence['tag_counts']})"
    )
    lines.append(
        f"- closes route: {coherence['closes_route']}; requires adjudication: "
        f"{coherence['requires_adjudication']}"
    )
    lines.append("")

    lines.append("## Raw P -> P+E component transitions")
    lines.append("")
    lines.append("| component | all 26 primary owners | deterministic interpretable |")
    lines.append("| --- | --- | --- |")
    transitions = report["paired_component_transitions"]
    for component in sorted(transitions["all_primary_owners"]):
        lines.append(
            f"| {component} | {transitions['all_primary_owners'][component]} | "
            f"{transitions['interpretable_owners'][component]} |"
        )
    lines.append("")
    for branch, counts in transitions["by_branch"].items():
        lines.append(f"- interpretable `{branch}` cohort: {counts}")
    lines.append("")

    lines.extend(_render_conclusion_fragility(report["conclusion_fragility"]))

    lines.append("## Controls (never in the primary denominator)")
    lines.append("")
    for cohort, summary in report["controls"]["cohorts"].items():
        lines.append(
            f"- `{cohort}`: {summary['owner_count']} owners, "
            f"{summary['quarantined_owner_count']} quarantined"
        )
    lines.append("")

    lines.append("## Not claimed")
    lines.append("")
    for note in report["not_claimed"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 6. Orchestration and CLI
# ---------------------------------------------------------------------------


def run_analysis(merged_dir: Path) -> dict[str, Any]:
    """Load merged primary records, classify, and build every output payload."""

    inputs = load_merged_inputs(merged_dir)

    primary_records = [
        record
        for record in inputs.owner_records
        if str(record.get("cohort")) == PRIMARY_COHORT
    ]
    control_records = [
        record
        for record in inputs.owner_records
        if str(record.get("cohort")) != PRIMARY_COHORT
    ]
    for record in control_records:
        if str(record.get("cohort")) not in (
            TIMING_CONTROL_COHORT,
            TP_REPLAY_CONTROL_COHORT,
        ):
            _fail(
                f"merged owner record {record.get('gt_owner_id')!r} declares unknown cohort "
                f"{record.get('cohort')!r}"
            )

    primary_rows = sorted(
        (classify_owner(record) for record in primary_records),
        key=lambda row: (row["image_id"], row["gt_owner_id"]),
    )
    control_rows = sorted(
        (summarize_control(record) for record in control_records),
        key=lambda row: (row["cohort"], row["image_id"], row["gt_owner_id"]),
    )
    owner_ids = [row["gt_owner_id"] for row in primary_rows]
    if len(set(owner_ids)) != len(owner_ids):
        _fail("the merged primary cohort carries a duplicated owner")

    report = build_report(
        merged_dir=inputs.merged_dir,
        merge_receipt=inputs.merge_receipt,
        file_sha256=inputs.file_sha256,
        primary_rows=primary_rows,
        control_rows=control_rows,
        plan_facts=load_sealed_plan_crossing_facts(inputs.merge_receipt),
        census_facts=load_census_owner_facts(inputs.merge_receipt),
    )
    return {
        "report": report,
        "owner_rows": [*primary_rows, *control_rows],
        "input_file_sha256": dict(inputs.file_sha256),
    }


def build_output_files(result: Mapping[str, Any]) -> dict[str, bytes]:
    """The deterministic, self-sealed analysis byte content."""

    report = result["report"]
    owner_rows = result["owner_rows"]
    fragility = report["conclusion_fragility"]

    report_json = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    report_md = render_markdown(report).encode("utf-8")
    owner_rows_bytes = b"".join(
        canonical_json_bytes(row) + b"\n" for row in owner_rows
    )

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "analyzer_source_sha256": scorer.sha256_file(Path(__file__).resolve()),
        "scorer_source_sha256": merge.FROZEN_SCORER_SOURCE_SHA256,
        "merger_source_sha256": scorer.sha256_file(Path(merge.__file__).resolve()),
        "merged_dir": report["merged_dir"],
        "merge_receipt_content_sha256": report["merge_receipt_content_sha256"],
        "runtime_identity_sha256": report["runtime_identity_sha256"],
        "input_file_sha256": dict(sorted(result["input_file_sha256"].items())),
        "artifact_determinism": "no_wall_clock_or_host_specific_field_is_sealed",
        "primary_denominator": report["primary_cohort"]["denominator"],
        "routing_denominator_count": report["routing"]["routing_denominator_count"],
        "interpretability_gate_passed": report["interpretability_gate"]["passed"],
        "decision": report["decision"],
        "policy": {
            "secondary_compatibility_read": False,
            "branch_source": "scorer pure helpers classify_displacement/classify_primary_branch",
            "threshold_fitting": False,
            "score_dependent_cohort_change": False,
            "conclusion_fragility_changed_the_decision": False,
            # The two lineage-bound reference reads, sealed here so a reader can
            # tell whether the identity and coverage slices were resolvable.
            "crossing_e_owner_identity_source": fragility["greedy_displacer_identity"][
                "crossing_e_owner_identity_source"
            ]["source"],
            "census_owner_registry_source": fragility["displacer_characterization"][
                "census_owner_registry_source"
            ]["source"],
            "census_owner_registry_sha256": fragility["displacer_characterization"][
                "census_owner_registry_source"
            ]["owner_registry_sha256"],
        },
        "output_file_digests": {
            OWNER_ROWS_NAME: {
                "path": OWNER_ROWS_NAME,
                "byte_size": len(owner_rows_bytes),
                "row_count": len(owner_rows),
                "sha256": sha256_bytes(owner_rows_bytes),
            },
            REPORT_JSON_NAME: {
                "path": REPORT_JSON_NAME,
                "byte_size": len(report_json),
                "sha256": sha256_bytes(report_json),
            },
            REPORT_MD_NAME: {
                "path": REPORT_MD_NAME,
                "byte_size": len(report_md),
                "sha256": sha256_bytes(report_md),
            },
        },
    }
    receipt["receipt_content_sha256"] = sha256_json(receipt)

    return {
        OWNER_ROWS_NAME: owner_rows_bytes,
        REPORT_JSON_NAME: report_json,
        REPORT_MD_NAME: report_md,
        RECEIPT_NAME: canonical_json_bytes(receipt) + b"\n",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merged-dir",
        type=Path,
        required=True,
        help="Merged primary evidence directory published by the merge module",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_analysis(Path(args.merged_dir))
        files = build_output_files(result)
        published = merge.publish_merge(Path(args.output_dir), files)
    except (AnalysisContractError, merge.MergeContractError, scorer.CrossingBoundaryContractError) as exc:
        raise SystemExit(f"analysis contract violated: {exc}") from exc
    print(
        json.dumps(
            {
                "analysis": published,
                "decision": result["report"]["decision"],
                "primary_denominator": result["report"]["primary_cohort"]["denominator"],
                "routing_denominator_count": result["report"]["routing"][
                    "routing_denominator_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
