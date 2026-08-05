#!/usr/bin/env python3
"""CPU-only planner for the sorted owner accessibility phenotype census
(``2026-08-03-sorted-owner-accessibility-phenotype-census``).

This planner materializes the *entire* prospective, score-independent plan for
a twelve-image, 346-owner observational census on the geometry-sorted
step-``4887`` checkpoint, and reads no model output of any kind.  It emits
immutable JSONL registries plus a self-reconstructing receipt:

``image-registry.jsonl``
    Per-image prompt tokens, canvas dimensions, wrapper/coordinate token
    identity, native stop reason, and split assignment.
``owner-registry.jsonl``
    Exactly 346 census owners with sort keys, greedy eligibility, native
    true-positive calibration marking, and bank-integrity summaries.
``category-registry.jsonl``
    Every normalized description present in each image, with its category
    token IDs and the canonical coordinate-query suffix.
``context-registry.jsonl``
    412 contexts: root plus every complete native greedy row boundary plus
    terminal, per image, with continuous loop counts and the frontier.
``candidate-bank.jsonl``
    Physical (alias-collapsed) candidates derived from exactly seventeen fixed
    logical transform roles per owner.
``query-group-registry.jsonl``
    One row per ``(context, category)``: the prefill unit, with the full-prefix
    digest.
``native-sidecar-registry.jsonl``
    The 400 native emitted boxes, excluded from every core rank.
``shard-manifest.jsonl``
    Twelve per-image shards with ``estimated_work_units`` and a largest-first
    dispatch order.

Why this module does not import the all-person planner
------------------------------------------------------
``build_sorted_all_person_route_landscape.py`` is a valid precedent for the
size-aware transform geometry, but it is hardcoded to image ``7511``'s canvas
(module-level ``IMAGE_WIDTH``/``IMAGE_HEIGHT``) and to a 41-owner
``person``-only universe, and its plan-v2 receipt carries the pre-P0
``full_prefix`` semantics this unit explicitly quarantines.  This module
therefore re-derives the geometry per image and keeps its own digest helpers.
The three tiny JSON-digest helpers below are deliberate local copies, not an
import, so this unit's artifacts never inherit another unit's schema identity.

The canonical coordinate-query suffix (P0)
------------------------------------------
For a context ``c`` and normalized description ``d`` the literal reforward
prefix is exactly::

    observed_self_prefix(c) + [OBJECT_REF_START, tokens(d), OBJECT_REF_END, BOX_START]

and nothing else.  ``score_complete_box_candidate``'s contract is that its
prefill logits already *are* the ``x1`` distribution, which holds only when the
prefix ends at the forced box opener.  A prefix ending at a prior ``box_end``
silently asks "what comes after a finished row" instead of "what coordinate
opens this forced box".  :func:`assert_canonical_query_suffix` fails closed on
any deviation, at plan time and again at score time.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Protocol

UNIT_ID = "2026-08-03-sorted-owner-accessibility-phenotype-census"
PLAN_SCHEMA_VERSION = "sorted-owner-accessibility-census-plan.v1"
CAPTURE_RULES_SCHEMA_VERSION = "sorted-owner-accessibility-census-capture-rules.v1"

ROOT = Path("/data/CoordExp")
OUTPUTS = ROOT / "outputs/research/qwen3-vl-dense-enumeration"
PANEL_PATH = (
    OUTPUTS
    / "2026-07-21-best-sampled-trajectory-positive-row-imitation-screen"
    / "evaluation-inputs/human-refined-12.coord.jsonl"
)
TASK0_ROOT = OUTPUTS / "2026-08-01-sorted-owner-basin-task0-v2-global-ambiguity-neutral-final"
OWNER_LEDGER_PATH = TASK0_ROOT / "owner-ledger.jsonl"
PREDICTION_LEDGER_PATH = TASK0_ROOT / "prediction-row-ledger.jsonl"
GREEDY_PATH = (
    OUTPUTS / "2026-07-29-three-checkpoint-human-refined12-max3084/sorted/greedy/greedy.json"
)
BASE_MODEL_PATH = (
    "/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
)

#: Registered digests of the immutable inputs (unit.md "Frozen source
#: boundary").  ``greedy.json`` is bound at plan time rather than pinned here
#: because it is a large multi-rollout artifact; its digest is sealed into the
#: receipt.
REGISTERED_DIGESTS: Mapping[str, str] = {
    "panel": "cfe4f693133287f9e6c561fc094710c642f049aec3d50f44dea98b764ba2aa85",
    "owner_ledger": "80357539069ac04c12522a211bb70a0bc6841b54a32627e993aeaa75574bd8a0",
    "prediction_ledger": "e47e0104a0e98cda68be6383424e8066a0f05c34b2e24ceac7dd19490be8cb70",
}

OBJECT_REF_START = 151646
OBJECT_REF_END = 151647
BOX_START = 151648
BOX_END = 151649
IM_END = 151645
COORD_TOKEN_START = 151670
COORD_TOKEN_END = 152669
COORD_BIN_COUNT = 1000

WRAPPER_TOKEN_IDS: Mapping[str, int] = {
    "object_ref_start": OBJECT_REF_START,
    "object_ref_end": OBJECT_REF_END,
    "box_start": BOX_START,
    "box_end": BOX_END,
    "im_end": IM_END,
}

EXPECTED_OWNER_COUNT = 346
EXPECTED_GREEDY_ELIGIBLE_COUNT = 343
EXPECTED_NATIVE_ROW_COUNT = 400
EXPECTED_CONTEXT_COUNT = 412

#: Contract item 12.  Frozen; never re-derived, never re-balanced.
DISCOVERY_IMAGE_IDS: tuple[str, ...] = ("10707", "14038", "2685", "5001", "6040", "7511")
CONFIRMATION_IMAGE_IDS: tuple[str, ...] = ("13348", "13923", "14439", "1584", "16228", "4134")
SPLIT_BY_IMAGE_ID: Mapping[str, str] = {
    **{image_id: "discovery" for image_id in DISCOVERY_IMAGE_IDS},
    **{image_id: "confirmation" for image_id in CONFIRMATION_IMAGE_IDS},
}

IOU_THRESHOLD = 0.5
MATCHER_EPSILON = 1e-12

#: Contract item 7.  Seventeen fixed logical transform roles per owner, in this
#: exact order.  The order is the alias-collapse tiebreak: the first role
#: producing a coordinate-token tuple becomes that tuple's representative.
#:
#: ``core`` reproduces the all-person precedent's non-degenerate outcome (its
#: exact anchor plus the first four translations and first four extents of its
#: fixed families).  ``extension`` is exactly the eight additional transforms
#: named by the brief.  There is no reserve list and no substitution: a role
#: that cannot be realized is recorded as not admitted, never replaced.
CORE_ROLES: tuple[str, ...] = (
    "exact_gt_anchor",
    "translate_left",
    "translate_right",
    "translate_up",
    "translate_down",
    "width_expand",
    "width_shrink",
    "height_expand",
    "height_shrink",
)
EXTENSION_ROLES: tuple[str, ...] = (
    "translate_up_left",
    "translate_up_right",
    "translate_down_left",
    "translate_down_right",
    "isotropic_expand",
    "isotropic_shrink",
    "top_anchored_height_shrink",
    "bottom_anchored_height_shrink",
)
LOGICAL_ROLES: tuple[str, ...] = CORE_ROLES + EXTENSION_ROLES
LOGICAL_ROLE_COUNT = 17
CORE_ROLE_COUNT = 9
EXTENSION_ROLE_COUNT = 8
ROLE_CLASS: Mapping[str, str] = {
    **{role: "core" for role in CORE_ROLES},
    **{role: "extension" for role in EXTENSION_ROLES},
}
ROLE_ORDINAL: Mapping[str, int] = {role: index for index, role in enumerate(LOGICAL_ROLES)}

#: Frozen literal loop flag (unit.md "Loop marking").  A bookkeeping
#: convenience for aggregation weighting only -- never a mechanism label.
LOOP_TAIL_MIN_CONSECUTIVE_RUN = 3

#: Frozen bank-coverage bands, set before any score is read and used only to
#: qualify an owner's evidence -- never to drop the owner or grow the bank.
#:
#: Coverage is measured on ``distinct_physical_candidates_reached``: how much
#: of the intended seventeen-role local landscape actually exists as distinct
#: scoreable geometry for this owner.  It is deliberately **not** measured on
#: ``uniquely_assigned_candidate_count``.  The bank is a local landscape probe,
#: not a set of alternative detections: a role that translates by ``w/4`` or
#: isotropically expands by ``dx`` per side is *designed* to move off the
#: owner, and an isotropic expansion has IoU ``1/2.25 ~ 0.44`` with its own
#: generator, so it correctly fails the ``0.5`` strict matcher.  Measured on
#: the frozen panel the dominant loss is ``unmatched`` (2119 roles) rather than
#: same-category competition (``other`` 163, ``ambiguous`` 0), and owners that
#: are the only instance of their category in their image show the same
#: distribution -- confirming the loss is perturbation distance, not
#: competition.  Keying coverage on unique assignment would therefore mark
#: ~95% of owners undercovered and neuter the census.
#:
#: ``uniquely_assigned_candidate_count`` is still recorded per owner, as the
#: contract requires, and remains available as a diagnostic.
#:
#: Frozen before any score:
#:
#: * ``full``                        -- all 17 distinct physical candidates;
#: * ``adequate_reduced``            -- 12 to 16 (>= 70.6% of the landscape),
#:                                      still fully disposition-eligible;
#: * ``undercovered_unresolved_only`` -- fewer than 12, or the exact anchor is
#:                                      missing or not uniquely self-assigned.
#:
#: Only the last status floors an owner to ``unresolved``.  An owner must never
#: be floored merely because a few transforms produced token-identical boxes.
BANK_COVERAGE_FULL_REACHED_AT_LEAST = LOGICAL_ROLE_COUNT
BANK_COVERAGE_ADEQUATE_REACHED_AT_LEAST = 12
BANK_COVERAGE_STATUSES: tuple[str, ...] = (
    "full",
    "adequate_reduced",
    "undercovered_unresolved_only",
)
DISPOSITION_ELIGIBLE_BANK_STATUSES: frozenset[str] = frozenset({"full", "adequate_reduced"})

#: Frozen support-calibration contract.  Support is a property of the *shape*
#: of an owner's local landscape, never of its position in a ranking: rank and
#: margin are a routing/competition surface and are never a support criterion.
#:
#: Thresholds are fixed quantiles of the pooled discovery **native true
#: positive** population read at each TP owner's deterministic due boundary --
#: owners the native greedy route actually emitted, whose localization is not
#: in question.  There is no search and no outcome-dependent tuning.
SUPPORT_STATISTICS: tuple[str, ...] = ("peak_lift", "local_concentration")
SUPPORT_PRIMARY_QUANTILE = 0.10
SUPPORT_SENSITIVITY_QUANTILES: tuple[float, ...] = (0.05, 0.25)
#: Fixed tolerances, never adaptive.  Observed scalar-reference parity is a
#: compliance check *against* these bounds; it never resizes them.
SUPPORT_EPSILON = 0.002
CROSS_CONTEXT_DELTA_EPSILON = 0.004
#: A category with fewer than this many calibration true positives falls back
#: to the pooled threshold and is flagged; it never receives a changed one.
CATEGORY_CONTRIBUTION_MIN = 20
POOLED_UNDERREPRESENTED_FLAG = "pooled_underrepresented"

SUPPORT_PEAK_LIFT_FORMULA = (
    "best_logprob - logsumexp(all unique candidate logprobs in the same query group) "
    "+ log(unique_population_size)"
)
SUPPORT_LOCAL_CONCENTRATION_FORMULA = (
    "best exclusion-filtered generator-local score - median(that owner's own "
    "exclusion-filtered bank scores)"
)


def compute_peak_lift(
    owner_best_logprob: float, unique_group_logprobs: Sequence[float]
) -> float:
    """Lift of an owner's best candidate over the uniform posterior.

    Exactly the owner's best log posterior within the *unique* ``(context,
    category)`` candidate population, minus ``log(1 / N_unique)``::

        peak_lift = best_logprob - logsumexp(all unique logprobs) + log(N_unique)

    The population is every unique candidate in the query group, not just this
    owner's -- the statistic asks how sharply the category's coordinate mass
    concentrates on this owner's geometry relative to a flat distribution over
    the whole group.  It is **not** the owner's best minus an owner-local
    reference level; that is :func:`compute_local_concentration`.

    Scale-free by construction: adding a constant to every logprob in the group
    leaves ``peak_lift`` unchanged.
    """

    values = [float(value) for value in unique_group_logprobs]
    if not values:
        raise PlanContractError("peak_lift requires a non-empty unique candidate population")
    peak = max(values)
    log_sum_exp = peak + math.log(math.fsum(math.exp(value - peak) for value in values))
    return float(owner_best_logprob) - log_sum_exp + math.log(len(values))


def compute_local_concentration(
    owner_best_logprob: float, owner_bank_logprobs: Sequence[float]
) -> float:
    """How far an owner's best candidate stands above its own bank's median.

    Exactly::

        local_concentration = best exclusion-filtered generator-local score
                              - median(that owner's own exclusion-filtered bank scores)

    ``median`` is the ordinary median (the mean of the two central values for an
    even-sized bank).  The reference population is the owner's own
    exclusion-filtered generator-local bank, so a crowded image cannot depress
    this statistic through competition it never had to win.

    It is **not** a normalized-mass share.
    """

    values = sorted(float(value) for value in owner_bank_logprobs)
    if not values:
        raise PlanContractError(
            "local_concentration requires a non-empty exclusion-filtered owner bank"
        )
    middle = len(values) // 2
    median = (
        values[middle]
        if len(values) % 2
        else (values[middle - 1] + values[middle]) / 2.0
    )
    return float(owner_best_logprob) - median


def clears_support(
    *,
    peak_lift: float,
    local_concentration: float,
    peak_lift_threshold: float,
    local_concentration_threshold: float,
    epsilon: float = SUPPORT_EPSILON,
) -> bool:
    """Usable support requires **both** statistics to clear their threshold.

    A statistic clears when it is at least ``threshold + epsilon``; the epsilon
    is added so a value sitting inside numerical tolerance of the threshold does
    not count as support.
    """

    return (
        float(peak_lift) >= float(peak_lift_threshold) + float(epsilon)
        and float(local_concentration)
        >= float(local_concentration_threshold) + float(epsilon)
    )


def owner_disposition_eligibility(
    *,
    greedy_eligible: bool,
    greedy_eligibility_status: str,
    bank_coverage_status: str,
    native_true_positive: bool,
) -> dict[str, Any]:
    """Whether an owner may close as a persistent negative, and why not.

    Three independent floors, all of which must lift:

    * **bank adequacy** -- an undercovered bank cannot support a negative
      conclusion about geometry it never probed;
    * **greedy eligibility** -- a ``globally_ambiguous_neutral`` owner has no
      unique identity under the canonical matcher, so "this owner was not
      localized" is not a statement the census can make about it; and
    * **native false negative** -- the persistent-negative label is a statement
      about owners the native route *missed*.  Native true positives are the
      calibration and positive-control population: the ``q10`` threshold is
      derived *from* them, so by construction roughly a tenth of them sit below
      ``q10 + epsilon``.  Labelling those as persistent negatives would be
      circular, and counting them in the prevalence denominator would mix the
      calibration population into the quantity being estimated.

    Ineligible owners are still censused: they keep every continuous row. They
    simply cannot close negative, and they never enter a false-negative
    prevalence or confirmation denominator.
    """

    bank_eligible = bank_coverage_status in DISPOSITION_ELIGIBLE_BANK_STATUSES
    native_false_negative = not bool(native_true_positive)
    eligible = bool(greedy_eligible) and bank_eligible and native_false_negative
    blockers: list[str] = []
    if not greedy_eligible:
        blockers.append(f"not_greedy_eligible:{greedy_eligibility_status}")
    if not bank_eligible:
        blockers.append(f"bank_{bank_coverage_status}")
    if not native_false_negative:
        blockers.append("native_true_positive_calibration_control")
    return {
        "greedy_eligible": bool(greedy_eligible),
        "greedy_eligibility_status": str(greedy_eligibility_status),
        "bank_coverage_status": str(bank_coverage_status),
        "bank_disposition_eligible": bank_eligible,
        "native_true_positive": bool(native_true_positive),
        "native_false_negative": native_false_negative,
        "calibration_role": (
            "native_true_positive_calibration"
            if native_true_positive
            else "native_false_negative"
        ),
        "persistent_negative_eligible": eligible,
        "enters_false_negative_confirmation_denominator": eligible,
        "enters_false_negative_prevalence_denominator": eligible,
        "retained_in_continuous_census_rows": True,
        "excluded_from_census": False,
        "disposition_floor": (
            "no_floor"
            if eligible
            else "unresolved_only_never_persistent_no_tested_localization_support"
        ),
        "disposition_floor_blockers": blockers,
    }


def is_persistent_negative(context_support_flags: Sequence[bool]) -> bool:
    """No optimistic-``U`` context in which **both** statistics cleared.

    The quantifier matters.  A persistent negative is *not* "each statistic is
    individually below its threshold in every context" -- an owner could clear
    ``peak_lift`` in one context and ``local_concentration`` in another and
    still never have both at once.  The correct test is that no tested context
    exists where the conjunction holds under the optimistic ``U`` bound.
    """

    return not any(bool(flag) for flag in context_support_flags)


#: Frozen minimal-frontier tie rule: among contexts attaining the minimal
#: absolute signed frontier distance for an owner, take the *first non-loop*
#: boundary in ascending boundary order.  Frozen here, before capture, so no
#: tie is ever resolved by looking at scores.
MINIMAL_FRONTIER_TIE_RULE = "first_non_loop_boundary_at_minimal_absolute_signed_distance"

#: Deterministic, target-blind scalar reference policy (contract item 10).  The
#: probe subset is chosen from request-ID digests only -- never from owner
#: identity, candidate class, or any score.
SCALAR_REFERENCE_REPEAT_COUNT = 8
SCALAR_REFERENCE_SUBSET_MODULUS = 64
SCALAR_VS_BATCH_MAX_ABS_DIFF = 1e-3

#: Representative real smoke image.  Deliberately a *discovery* image: running
#: a representative smoke on a confirmation image would consume held-out
#: evidence before the confirmation rule is frozen.  ``6040`` is the smallest
#: discovery shard (15 owners, 4 categories, 11 contexts).
REPRESENTATIVE_SMOKE_IMAGE_ID = "6040"

#: Repetition penalty ``1.0`` is the only stratum this unit scores.  An
#: ``rp1.10`` robustness view is explicitly *deferred*, not blocking: no
#: canonical ``rp1.10`` native rollout artifact is frozen for this panel, so
#: there is no admissible context registry to score it against.
NATIVE_REPETITION_PENALTY_STRATUM = 1.0
DEFERRED_ROBUSTNESS_STRATA: Mapping[str, str] = {
    "rp1.10": "deferred_no_frozen_canonical_native_artifact",
}

assert len(LOGICAL_ROLES) == LOGICAL_ROLE_COUNT
assert len(set(LOGICAL_ROLES)) == LOGICAL_ROLE_COUNT
assert len(CORE_ROLES) == CORE_ROLE_COUNT
assert len(EXTENSION_ROLES) == EXTENSION_ROLE_COUNT


#: Capture granularity (runtime-seam audit).  These are *knobs with frozen
#: defaults*, sealed into ``capture-rules.json`` so the launch granularity can
#: be finalized without redesigning the harness.
#:
#: ``session_scope`` is how long one loaded model/session lives.  The preferred
#: default is one long-lived session per image shard, which avoids 412 model
#: loads.
#:
#: ``admission_scope`` is the unit at which the cache/parity admission probe is
#: *independently re-run*.  It is deliberately ``context``: the legacy
#: ``run()`` performs strict admission once for the first group in a process
#: and that result is **not** transferable to later contexts.  Admission is
#: never inherited across contexts.
#:
#: ``cache_scope`` is the unit at which KV-cache state is rebuilt from scratch.
#: It is ``query_group`` -- fresh cache per ``(context, category)`` -- because
#: the query suffix is category-dependent.
DEFAULT_SESSION_SCOPE = "image_shard"
#: Lead's final runtime choice for the first capture: accuracy over efficiency.
#: Admission identity is the **exact** ``(context, category, query prefix
#: digest)``.  Suffix-shape sharing was considered and rejected: the runtime's
#: existing admission identity already includes the exact prefix, and no proof
#: exists that admission is invariant under token identity at equal shape.
DEFAULT_ADMISSION_SCOPE = "context_category_exact_query_prefix"
DEFAULT_CACHE_SCOPE = "query_group"
SESSION_SCOPES: tuple[str, ...] = ("image_shard", "context", "query_group")
ADMISSION_SCOPES: tuple[str, ...] = ("context_category_exact_query_prefix",)
CACHE_SCOPES: tuple[str, ...] = ("query_group",)

#: Admission channels.  Three distinct execution shapes, each separately
#: exact-prefix admitted; no admission ever covers another channel.
#:
#: ``query_suffix``
#:     The coordinate query: observed prefix plus the canonical query suffix.
#: ``proposal_boundary_gate``
#:     The natural boundary with *nothing forced*.  Genuinely context-level:
#:     one read of the observed prefix, shared by every category.
#: ``proposal_category_route``
#:     The category routing path, conditional on a forced
#:     ``object_ref_start``.  Per ``(context, category)``: two categories in one
#:     context traverse different token identities and lengths, so a single
#:     context-level observed-prefix receipt does **not** cover them.
CHANNEL_QUERY_SUFFIX = "query_suffix"
CHANNEL_PROPOSAL_BOUNDARY_GATE = "proposal_boundary_gate"
CHANNEL_PROPOSAL_CATEGORY_ROUTE = "proposal_category_route"
ADMISSION_CHANNELS: tuple[str, ...] = (
    CHANNEL_QUERY_SUFFIX,
    CHANNEL_PROPOSAL_BOUNDARY_GATE,
    CHANNEL_PROPOSAL_CATEGORY_ROUTE,
)


def proposal_route_token_ids(category_token_ids: Sequence[int]) -> list[int]:
    """The category routing path, conditional on a forced ``object_ref_start``.

    Exactly ``[object_ref_start, *category tokens, object_ref_end]``.  Note it
    deliberately stops *before* ``box_start``: the routing event is the
    category decision, and no coordinate score may enter a proposal quantity.
    """

    tokens = [int(value) for value in category_token_ids]
    if not tokens:
        raise PlanContractError("a proposal route requires at least one category token")
    return [OBJECT_REF_START, *tokens, OBJECT_REF_END]


def proposal_route_digest(category_token_ids: Sequence[int]) -> str:
    return sha256_json(proposal_route_token_ids(category_token_ids))


def proposal_route_admission_receipt_id(
    *, context_id: str, observed_prefix_sha256: str, category_token_ids: Sequence[int]
) -> str:
    """Per-``(context, category)`` proposal-route admission identity.

    Keyed on the observed prefix *and* the exact routing-path digest, so two
    categories in one context can never share a proposal-route admission even
    when their suffix lengths coincide.
    """

    combined = sha256_json(
        [str(observed_prefix_sha256), proposal_route_digest(category_token_ids)]
    )
    return admission_receipt_id(
        context_id=context_id,
        channel=CHANNEL_PROPOSAL_CATEGORY_ROUTE,
        prefix_sha256=combined,
    )


def suffix_shape_class(category_token_ids: Sequence[int]) -> str:
    """Execution-shape class of a category's canonical query suffix.

    Retained as a *diagnostic* label so shape coverage stays auditable.  It is
    deliberately **not** the admission key for the first capture: admission is
    keyed on the exact query prefix digest, so two categories of equal suffix
    length never share an admission receipt.
    """

    return f"suffix_len_{len(build_query_suffix(category_token_ids))}"


def admission_receipt_id(*, context_id: str, channel: str, prefix_sha256: str) -> str:
    """Stable ID of the admission receipt that must cover a scored row.

    Keyed on the exact prefix digest for the channel: the query prefix for
    ``query_suffix`` rows, the observed prefix for ``proposal`` rows.  Every
    score row references its covering ``admission_receipt_id`` and the merge
    fails any row whose covering receipt is absent, so a row can never inherit
    another context's, category's, or channel's admission.
    """

    if channel not in ADMISSION_CHANNELS:
        raise PlanContractError(f"unknown admission channel {channel!r}")
    return f"admission:{context_id}|{channel}|{prefix_sha256}"


class PlanContractError(ValueError):
    """Raised when a frozen input or a planner invariant is not satisfied."""


def build_capture_rules(
    *,
    session_scope: str = DEFAULT_SESSION_SCOPE,
    admission_scope: str = DEFAULT_ADMISSION_SCOPE,
    cache_scope: str = DEFAULT_CACHE_SCOPE,
) -> dict[str, Any]:
    """The pre-capture rule object every shard and the merge must bind.

    Sealed *before* any GPU work, digested, and re-verified by the scorer and
    the merge.  It fixes the query suffix, the loop rule, the minimal-frontier
    tie rule, alias collapse, admission, and the stop policy, so none of those
    can be quietly re-chosen after seeing scores.

    Phenotype *thresholds* are deliberately absent: they are derived later, on
    the discovery half only, and sealed into their own frozen discovery-rule
    digest that the confirmation analysis must bind without retuning.
    """

    if session_scope not in SESSION_SCOPES:
        raise PlanContractError(f"unknown session scope {session_scope!r}")
    if admission_scope not in ADMISSION_SCOPES:
        raise PlanContractError(f"unknown admission scope {admission_scope!r}")
    if cache_scope not in CACHE_SCOPES:
        raise PlanContractError(f"unknown cache scope {cache_scope!r}")

    rules: dict[str, Any] = {
        "schema_version": CAPTURE_RULES_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "query_suffix": {
            "shape": ["object_ref_start", "category_token_ids", "object_ref_end", "box_start"],
            "wrapper_token_ids": dict(WRAPPER_TOKEN_IDS),
            "x1_distribution_read_point": "immediately_after_box_start",
            "on_suffix_or_token_alignment_mismatch": "fail_closed",
            "pre_p0_scores": "quarantined_never_read_mechanically_unjoinable",
        },
        "loop_rule": {
            "continuous_fields": [
                "prior_identical_row_count",
                "consecutive_identical_row_run_length",
                "repeated_raw_span_sha256",
            ],
            "flag_rule": f"consecutive_identical_row_run_length >= {LOOP_TAIL_MIN_CONSECUTIVE_RUN}",
            "flag_is_not_a_mechanism_label": True,
        },
        "minimal_frontier_tie_rule": MINIMAL_FRONTIER_TIE_RULE,
        # Owner support semantics, frozen before capture.  The primary owner
        # neighbourhood is the generator-local landscape; strict assignment is
        # a separate, narrower, owner-identifiable lower bound.
        "owner_support": {
            "primary_neighbourhood": "generator_local_landscape",
            "adequacy_gate": "distinct_physical_candidate_count_after_alias_collapse",
            "exact_anchor": "mandatory",
            "bank_adequacy_rule": {
                "statuses": list(BANK_COVERAGE_STATUSES),
                "full_at_least": BANK_COVERAGE_FULL_REACHED_AT_LEAST,
                "adequate_at_least": BANK_COVERAGE_ADEQUATE_REACHED_AT_LEAST,
                "requires_exact_anchor_uniquely_self_assigned": True,
                "disposition_eligible_statuses": sorted(DISPOSITION_ELIGIBLE_BANK_STATUSES),
                "threshold_status": "frozen_before_any_score",
                "token_alias_never_floors_an_owner": True,
            },
            "strict_assignment_role": (
                "separate_owner_identifiable_lower_bound_not_bank_membership"
            ),
            "partitions": list(OWNER_CANDIDATE_PARTITIONS),
            "other_owner_strict_candidate": "excluded_from_target_support_moved_to_collision_diagnostic",
            "ambiguous_candidate": "counts_toward_upper_bound_only",
            # A perturbation designed to move off its own owner is still part
            # of that owner's landscape (upper bound) but is not
            # owner-identifiable (never the lower bound).
            "unmatched_generator_local": "upper_bound_only",
            "partition_bounds": {
                "strict_assigned_self": ["lower", "upper"],
                "ambiguous_upper": ["upper"],
                "unmatched_generator_local": ["upper"],
                "other_owner_strict": [],
                "not_generated_by_owner": [],
            },
            "lower_upper_disposition_flip": "unresolved",
            # Support is defined solely by the calibrated local-peak statistics
            # below.  Rank/margin describe who won a query group and are never
            # read as evidence that an owner has localization support.
            "support_definition": {
                "statistics": list(SUPPORT_STATISTICS),
                "peak_lift": {
                    "formula": SUPPORT_PEAK_LIFT_FORMULA,
                    "equivalently": (
                        "best owner-candidate log posterior within the unique "
                        "(context, category) population minus log(1 / N_unique)"
                    ),
                    "population": "all_unique_candidates_in_the_query_group",
                    "not": "best_minus_an_owner_local_reference",
                    "shift_invariant": True,
                },
                "local_concentration": {
                    "formula": SUPPORT_LOCAL_CONCENTRATION_FORMULA,
                    "reference_population": "owner_own_exclusion_filtered_bank_scores",
                    "reference_statistic": "median_mean_of_two_central_values_when_even",
                    "not": "normalized_mass_share",
                },
                "rank_is_support_criterion": False,
                "rank_one_as_support": "forbidden",
                "computed_on": "generator_local_max_excluding_other_owner_strict",
                "evaluated_under_both_ambiguity_bounds": True,
                "usable_support_rule": (
                    "peak_lift >= threshold + epsilon AND "
                    "local_concentration >= threshold + epsilon"
                ),
                "usable_support_combinator": "conjunction_both_statistics",
            },
            "support_calibration": {
                "population": "discovery_native_true_positive_owners",
                "read_at": "deterministic_due_boundary",
                "stratification": "pooled",
                "primary_quantile": SUPPORT_PRIMARY_QUANTILE,
                "sensitivity_quantiles": list(SUPPORT_SENSITIVITY_QUANTILES),
                "sensitivity_role": "diagnostic_only_never_primary_never_a_disposition",
                "category_stratified_role": "diagnostic_only",
                "category_contribution_min": CATEGORY_CONTRIBUTION_MIN,
                "underrepresented_flag": POOLED_UNDERREPRESENTED_FLAG,
                "underrepresented_policy": "fall_back_to_pooled_never_change_the_threshold",
                "degenerate_quantile_fallback": (
                    "pooled when the stratum q10 lies within support_epsilon of its minimum"
                ),
                "threshold_search_or_tuning": "forbidden",
            },
            "epsilons": {
                "support_epsilon": SUPPORT_EPSILON,
                "cross_context_delta_epsilon": CROSS_CONTEXT_DELTA_EPSILON,
                "adaptive": False,
                "observed_parity_role": "compliance_check_against_bound_never_resizes_it",
            },
            # Analysis order is enforced by digests, not by capture order, so
            # all twelve raw shards may be captured before any analysis runs.
            "confirmation_blinding": {
                "capture_order": "all_twelve_shards_may_be_front_loaded",
                "stages": [
                    "capture_manifest_seals_every_shard_digest_first",
                    "calibration_consumes_discovery_digests_only",
                    "calibration_receipt_sealed_with_digest",
                    "confirmation_binds_exact_calibration_receipt_digest",
                ],
                "calibration_reads_confirmation_rows": False,
                "confirmation_retune": "forbidden",
                "rederivation_on_all_twelve": "forbidden",
                "on_calibration_receipt_digest_mismatch": "fail_closed",
            },
            "required_per_owner_context_fields": [
                "generator_local_max",
                "generator_local_max_excluding_other_owner_strict",
                "strict_assigned_max",
                "exact_anchor_score",
                "ambiguous_upper_max",
            ],
            "required_per_owner_report_fields": [
                "distinct_physical_candidate_count",
                "uniquely_assigned_candidate_count",
                "other_owner_strict_count",
            ],
            "competition_ranks": "routing_surface_never_support",
            "competition_rank_population": "exclusion_filtered_generator_local_maxima",
            "strict_table": "parallel_only_never_the_primary_rank",
            "persistent_negative_requires": [
                # A globally-ambiguity-neutral owner has no decidable identity
                # under the canonical matcher, so "this owner was not
                # localized" is not a statement the census can make about it.
                "greedy_eligible",
                # The label is about owners the native route missed.  True
                # positives are the calibration population that *defines* the
                # q10 threshold, so ~10% of them sit below it by construction.
                "native_false_negative",
                "frontier_tested",
                "non_loop_primary_contexts",
                "adequate_distinct_bank",
                "exact_anchor_score_reported",
                "no_optimistic_u_context_where_both_statistics_clear",
            ],
            # Contract item 1 keeps all 346 owners in the census.  The three
            # `globally_ambiguous_neutral` owners (gt:14038:41/42/43) are
            # retained in every continuous row, but a negative conclusion about
            # them would be an artefact of their undecidable identity rather
            # than a fact about the model.
            "non_eligible_owner_policy": {
                "greedy_eligibility_source": "owner_ledger_decision_eligibility_greedy_natural",
                "retained_in_continuous_census_rows": True,
                "excluded_from_census": False,
                "may_close_as_persistent_negative": False,
                "enters_false_negative_confirmation_denominator": False,
                "disposition_floor": (
                    "unresolved_only_never_persistent_no_tested_localization_support"
                ),
                "rationale": (
                    "a globally ambiguity-neutral owner has no unique identity under the "
                    "canonical matcher, so absence of support cannot be attributed to it"
                ),
            },
            # Native true positives are calibration and positive controls, not
            # candidates for a false-negative label.
            "native_true_positive_policy": {
                "role": "calibration_and_positive_control",
                "defines_the_q10_threshold": True,
                "may_fall_below_q10_plus_epsilon": True,
                "may_close_as_persistent_negative": False,
                "enters_false_negative_prevalence_denominator": False,
                "enters_false_negative_confirmation_denominator": False,
                "retained_in_continuous_census_rows": True,
                "rationale": (
                    "the q10 threshold is derived from this population, so roughly a tenth "
                    "of it sits below q10 + epsilon by construction; labelling those as "
                    "persistent negatives would be circular"
                ),
            },
            # The quantifier is over contexts, and the conjunction is inside
            # it.  Requiring each statistic to be individually below threshold
            # in every context is a *different, wrong* test: an owner could
            # clear peak_lift in one context and local_concentration in
            # another and still never clear both at once.
            "persistent_negative_rule": (
                "not any(context: peak_lift_U >= t_peak + eps AND "
                "local_concentration_U >= t_conc + eps)"
            ),
            "persistent_negative_forbids": [
                "any_rank_or_margin_criterion",
                "requiring_each_statistic_individually_below_threshold_at_every_context",
            ],
        },
        "candidate_collapse": {
            "logical_roles": list(LOGICAL_ROLES),
            "logical_role_count": LOGICAL_ROLE_COUNT,
            "physical_identity": "digest(image_id, normalized_description, coord_token_ids)",
            "collapse_scope": "image_and_normalized_description",
            "cross_owner_collapse": True,
            "generator_provenance_role": "provenance_only_never_rank_or_assignment",
            "assignment_scope": "same_normalized_description_only",
            "assignment_computed": "once_per_physical_candidate",
            "substitution_policy": "none_never_score_selected",
            "mid_run_growth": "forbidden",
            "rank_keys": ["image_id", "context_id", "normalized_description"],
            "rank_population": "collapsed_unique_physical_candidates_only",
            "duplicate_request_per_query_group": "fail_closed",
            "sidecar_identical_tuple": "joins_provenance_never_adds_rank_mass",
            "undercovered_owner_disposition_floor": (
                "unresolved_only_never_persistent_no_tested_localization_support"
            ),
        },
        "admission": {
            "session_scope": session_scope,
            "admission_scope": admission_scope,
            "cache_scope": cache_scope,
            "admission_key": ["context_id", "channel", "exact_prefix_sha256"],
            "channels": list(ADMISSION_CHANNELS),
            "channel_scopes": {
                CHANNEL_QUERY_SUFFIX: "context_and_category_exact_query_prefix",
                CHANNEL_PROPOSAL_BOUNDARY_GATE: "context_only_observed_prefix",
                CHANNEL_PROPOSAL_CATEGORY_ROUTE: (
                    "context_and_category_observed_prefix_plus_routing_path_digest"
                ),
            },
            "proposal_route_path": ["object_ref_start", "category_token_ids", "object_ref_end"],
            "proposal_route_excludes_box_start": True,
            "proposal_is_per_category_never_per_owner": True,
            "suffix_shape_sharing": "rejected_no_token_identity_invariance_proof",
            "reuse_admission_across_contexts": False,
            "reuse_admission_across_categories": False,
            "inherit_first_group_admission": False,
            "every_score_row_references_covering_admission_receipt_id": True,
            "merge_fails_rows_lacking_admission_coverage": True,
            "bulk_scoring_path": "admitted_kv_cache",
            "batched_full_reforward_role": "parity_or_fallback_diagnostic_only",
            "scalar_reference": {
                "selection": "request_id_digest_modulus",
                "modulus": SCALAR_REFERENCE_SUBSET_MODULUS,
                "repeat_count": SCALAR_REFERENCE_REPEAT_COUNT,
                "target_blind": True,
                "max_abs_diff": SCALAR_VS_BATCH_MAX_ABS_DIFF,
            },
        },
        # Hard Qwen3-VL runtime invariants.  The model carries a shared mutable
        # ``rope_deltas``, so an implicit-position call silently corrupts later
        # positions for every other group on the same model object.
        "runtime_invariants": {
            "explicit_position_ids_required_on_every_model_call": True,
            "model_generate_forbidden_on_scoring_model": True,
            "raw_model_call_without_explicit_positions_forbidden": True,
            "concurrent_group_threads_on_one_model": "forbidden",
            "groups_executed": "sequentially_within_one_image_session",
            "fresh_dynamic_cache_per_group": True,
            "assert_cache_length_equals_prefill_length": True,
            "delete_group_backend_before_next_group": True,
        },
        "stop_policy": {
            "shard_local_failure": "quarantine_that_image_only",
            "global_stop_quarantined_image_count_above": 2,
            "global_serial_calibration_gate": False,
            "ambiguous_close_decision": "may_request_scalar_overlay",
            "evidence_from_failed_shards": "forbidden",
        },
        "strata": {
            "scored": [NATIVE_REPETITION_PENALTY_STRATUM],
            "deferred_robustness": dict(DEFERRED_ROBUSTNESS_STRATA),
        },
        "representative_smoke": {
            "image_id": REPRESENTATIVE_SMOKE_IMAGE_ID,
            "split": SPLIT_BY_IMAGE_ID[REPRESENTATIVE_SMOKE_IMAGE_ID],
            "rationale": "smallest discovery shard; confirmation images stay unspent",
        },
        "diagnostics": {
            # One-pass GPU capture, then CPU-only analysis: the full 1000-bin
            # x1 distribution is captured in the GPU product even though it is
            # analysis-optional, so no later question forces a recapture.
            "full_x1_distribution": "captured_in_gpu_product",
            "full_x1_distribution_role": "diagnostic_only_never_a_2d_heatmap_never_a_rank",
        },
        "phenotype_thresholds": "not_frozen_here_derived_on_discovery_only",
    }
    rules["capture_rules_sha256"] = sha256_json(rules)
    return rules


# ---------------------------------------------------------------------------
# Small digest helpers (deliberate local copies; see the module docstring)
# ---------------------------------------------------------------------------


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanContractError(f"{label} is unreadable at {path}") from exc
    if not isinstance(payload, Mapping):
        raise PlanContractError(f"{label} is not a JSON object")
    return payload


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PlanContractError(f"{label} is unreadable at {path}") from exc
    for index, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PlanContractError(f"{label} line {index} is not valid JSON") from exc
        if not isinstance(row, Mapping):
            raise PlanContractError(f"{label} line {index} is not a JSON object")
        rows.append(dict(row))
    return rows


def _assert_digest(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise PlanContractError(
            f"{label} digest mismatch: expected {expected}, observed {actual}; "
            "the frozen source boundary has changed"
        )
    return actual


# ---------------------------------------------------------------------------
# Canonical coordinate-query suffix (contract item 5)
# ---------------------------------------------------------------------------


def build_query_suffix(category_token_ids: Sequence[int]) -> list[int]:
    """The one canonical suffix shape this unit ever scores against."""

    tokens = [int(value) for value in category_token_ids]
    if not tokens:
        raise PlanContractError("a category query suffix requires at least one category token")
    return [OBJECT_REF_START, *tokens, OBJECT_REF_END, BOX_START]


def assert_canonical_query_suffix(
    full_prefix_token_ids: Sequence[int],
    *,
    category_token_ids: Sequence[int],
    label: str,
) -> None:
    """Fail closed unless the literal prefix ends at the canonical suffix.

    Checks both the terminal ``box_start`` position and the exact suffix
    content, so a prefix that merely happens to end in ``box_start`` (for
    example one carrying a different category, or an extra token between
    ``object_ref_end`` and ``box_start``) is still rejected.
    """

    expected = build_query_suffix(category_token_ids)
    tokens = [int(value) for value in full_prefix_token_ids]
    if not tokens or tokens[-1] != BOX_START:
        raise PlanContractError(
            f"{label}: full prefix does not end at box_start; the reforward prefill "
            "would not yield the x1 distribution (pre-P0 semantics)"
        )
    if tokens[-len(expected) :] != expected:
        raise PlanContractError(
            f"{label}: full prefix does not end with the canonical query suffix "
            "[object_ref_start, category tokens, object_ref_end, box_start]"
        )


# ---------------------------------------------------------------------------
# Per-image geometry: production pixel <-> coordinate-token conversion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Canvas:
    """One image's pixel canvas; every conversion is image-local."""

    width: int
    height: int

    def round_to_bin(self, pixel: float, extent: int) -> int:
        return max(0, min(COORD_BIN_COUNT - 1, int(round(pixel * 1000.0 / float(extent)))))

    def pixel_to_bins(self, box: Sequence[float]) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = box
        bins = (
            self.round_to_bin(x1, self.width),
            self.round_to_bin(y1, self.height),
            self.round_to_bin(x2, self.width),
            self.round_to_bin(y2, self.height),
        )
        if bins[0] >= bins[2] or bins[1] >= bins[3]:
            return None
        return bins

    def bins_to_pixel(self, bins: Sequence[int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bins
        return (
            int(round(x1 * self.width / 1000.0)),
            int(round(y1 * self.height / 1000.0)),
            int(round(x2 * self.width / 1000.0)),
            int(round(y2 * self.height / 1000.0)),
        )

    def valid_pixel_box(self, box: Sequence[float]) -> bool:
        x1, y1, x2, y2 = box
        return 0 <= x1 < x2 <= self.width and 0 <= y1 < y2 <= self.height


def coord_token_ids(bins: Sequence[int]) -> list[int]:
    return [COORD_TOKEN_START + int(value) for value in bins]


def iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _assign_against(
    pixel_box: Sequence[float], owners: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Canonical matcher for one prediction against a given owner population.

    A single candidate has a strict owner only when its best eligible edge is
    unique; equal best edges are ambiguity-neutral.
    """

    eligible: list[tuple[str, float]] = []
    receipts: list[dict[str, Any]] = []
    for owner in owners:
        owner_id = str(owner["gt_owner_id"])
        overlap = iou(pixel_box, owner["bbox_pixel_xyxy"])
        receipts.append({"gt_owner_id": owner_id, "intersection_over_union": overlap})
        if overlap + MATCHER_EPSILON >= IOU_THRESHOLD:
            eligible.append((owner_id, overlap))
    receipts.sort(key=lambda item: (-item["intersection_over_union"], item["gt_owner_id"]))
    top_receipts = receipts[:4]
    if not eligible:
        return {
            "status": "unmatched",
            "gt_owner_id": None,
            "ambiguity_owner_ids": [],
            "top_owner_iou_receipts": top_receipts,
        }
    maximum = max(value for _, value in eligible)
    optimal = sorted(
        owner_id for owner_id, value in eligible if abs(value - maximum) <= MATCHER_EPSILON
    )
    if len(optimal) != 1:
        return {
            "status": "ambiguous_neutral",
            "gt_owner_id": None,
            "ambiguity_owner_ids": optimal,
            "top_owner_iou_receipts": top_receipts,
        }
    return {
        "status": "matched",
        "gt_owner_id": optimal[0],
        "ambiguity_owner_ids": [],
        "top_owner_iou_receipts": top_receipts,
    }


def strict_assignment(
    pixel_box: Sequence[float],
    owners: Sequence[Mapping[str, Any]],
    *,
    normalized_description: str,
) -> dict[str, Any]:
    """Category-local strict assignment, with a separately named diagnostic.

    A census candidate is generated and scored under a *forced category query*,
    so its competition is same-description by construction.  Matching it
    against every owner in the image would let a different-category box that
    happens to overlap manufacture spurious ``ambiguous_neutral`` or
    ``unmatched`` semantics for a category it was never competing in.

    The primary fields (``strict_assignment_*``) are therefore **category-local**:
    they consider only owners whose ``normalized_description`` equals the
    generating candidate's.  The all-category view is retained under the
    distinct name ``any_category_assignment_*`` as a diagnostic, so the two are
    never silently interchanged.
    """

    same_description = [
        owner
        for owner in owners
        if str(owner["normalized_description"]) == str(normalized_description)
    ]
    primary = _assign_against(pixel_box, same_description)
    diagnostic = _assign_against(pixel_box, owners)
    return {
        "strict_assignment_scope": "same_normalized_description_only",
        "strict_assignment_normalized_description": str(normalized_description),
        "strict_assignment_population_size": len(same_description),
        "strict_assignment_status": primary["status"],
        "strict_assignment_gt_owner_id": primary["gt_owner_id"],
        "ambiguity_owner_ids": primary["ambiguity_owner_ids"],
        "top_owner_iou_receipts": primary["top_owner_iou_receipts"],
        "any_category_assignment_status": diagnostic["status"],
        "any_category_assignment_gt_owner_id": diagnostic["gt_owner_id"],
        "any_category_ambiguity_owner_ids": diagnostic["ambiguity_owner_ids"],
        "any_category_assignment_role": "diagnostic_only_never_primary_rank_or_ambiguity",
    }


# ---------------------------------------------------------------------------
# The seventeen fixed logical transform roles
# ---------------------------------------------------------------------------


def apply_role(
    box: Sequence[int], *, role: str, dx: int, dy: int, canvas: Canvas
) -> tuple[int, int, int, int]:
    """Realize one fixed logical role against an owner box.

    Translations preserve width/height and slide within the canvas; extents
    change per-side by ``dx``/``dy`` and are clipped.  Anchored height shrinks
    hold one edge fixed, which is exactly what distinguishes them from the
    symmetric ``height_shrink`` core role.
    """

    x1, y1, x2, y2 = (int(value) for value in box)
    width, height = x2 - x1, y2 - y1

    if role == "exact_gt_anchor":
        return x1, y1, x2, y2

    if role.startswith("translate_"):
        direction = role[len("translate_") :]
        x_offset = (-dx if "left" in direction else dx if "right" in direction else 0)
        y_offset = (-dy if "up" in direction else dy if "down" in direction else 0)
        new_x1 = min(max(0, x1 + x_offset), max(0, canvas.width - width))
        new_y1 = min(max(0, y1 + y_offset), max(0, canvas.height - height))
        return new_x1, new_y1, new_x1 + width, new_y1 + height

    if role == "top_anchored_height_shrink":
        return x1, y1, x2, y2 - dy
    if role == "bottom_anchored_height_shrink":
        return x1, y1 + dy, x2, y2

    x_low, x_high, y_low, y_high = x1, x2, y1, y2
    if role in {"width_expand", "isotropic_expand"}:
        x_low, x_high = x_low - dx, x_high + dx
    if role in {"width_shrink", "isotropic_shrink"}:
        x_low, x_high = x_low + dx, x_high - dx
    if role in {"height_expand", "isotropic_expand"}:
        y_low, y_high = y_low - dy, y_high + dy
    if role in {"height_shrink", "isotropic_shrink"}:
        y_low, y_high = y_low + dy, y_high - dy
    return (
        max(0, min(canvas.width, x_low)),
        max(0, min(canvas.height, y_low)),
        max(0, min(canvas.width, x_high)),
        max(0, min(canvas.height, y_high)),
    )


def candidate_generator_geometry(
    candidate_pixel: Sequence[float], owner_pixel: Sequence[float]
) -> dict[str, Any]:
    """Deterministic geometry of one candidate box against its *generating* owner.

    Measured on the **decoded** candidate box -- the box that survives the
    pixel-to-bin-to-pixel round trip, i.e. what the model is actually asked
    about -- so the recorded displacement is the realized one rather than the
    requested one.

    This is per ``(candidate, generating owner)``, not per candidate.  After
    cross-owner alias collapse a single physical candidate can have several
    generators with *different* GT boxes, so the same coordinate tuple has a
    different IoU and a different centre offset for each of them.  Attaching
    geometry to the candidate rather than to the generator would silently pick
    one generator's numbers and attribute them to all of them.
    """

    cx1, cy1, cx2, cy2 = (float(v) for v in candidate_pixel)
    ox1, oy1, ox2, oy2 = (float(v) for v in owner_pixel)
    candidate_width, candidate_height = cx2 - cx1, cy2 - cy1
    owner_width, owner_height = ox2 - ox1, oy2 - oy1
    candidate_area = candidate_width * candidate_height
    generator_area = owner_width * owner_height
    return {
        "generator_bbox_pixel_xyxy": [ox1, oy1, ox2, oy2],
        "intersection_over_union_with_generator": iou(candidate_pixel, owner_pixel),
        "center_offset_pixels": [
            ((cx1 + cx2) / 2.0) - ((ox1 + ox2) / 2.0),
            ((cy1 + cy2) / 2.0) - ((oy1 + oy2) / 2.0),
        ],
        "extent_ratio": [
            candidate_width / owner_width if owner_width > 0 else None,
            candidate_height / owner_height if owner_height > 0 else None,
        ],
        "candidate_extent_pixels": [candidate_width, candidate_height],
        "generator_extent_pixels": [owner_width, owner_height],
        "candidate_area_pixels": candidate_area,
        "generator_area_pixels": generator_area,
        "area_ratio": candidate_area / generator_area if generator_area > 0 else None,
        "measured_on": "decoded_candidate_box_versus_generator_gt_box",
    }


def realize_owner_roles(owner: Mapping[str, Any], *, canvas: Canvas) -> list[dict[str, Any]]:
    """Realize all seventeen fixed logical roles for one owner.

    Pure geometry: no collapse, no assignment, no score.  Every role survives
    in the output, admitted or not; a role that cannot be realized is recorded
    with an ``invalid_reason`` and is never substituted.
    """

    owner_box = [int(v) for v in owner["bbox_pixel_xyxy"]]
    x1, y1, x2, y2 = owner_box
    dx = max(1, int(round((x2 - x1) / 4.0)))
    dy = max(1, int(round((y2 - y1) / 4.0)))

    records: list[dict[str, Any]] = []
    for role in LOGICAL_ROLES:
        proposed = apply_role(owner_box, role=role, dx=dx, dy=dy, canvas=canvas)
        record: dict[str, Any] = {
            "role": role,
            "role_ordinal": ROLE_ORDINAL[role],
            "candidate_class": ROLE_CLASS[role],
            "source_bbox_pixel_xyxy": list(proposed),
            "size_aware_offsets_pixels": {"dx": dx, "dy": dy},
        }
        if not canvas.valid_pixel_box(proposed):
            record.update(
                admitted=False,
                invalid_reason="invalid_or_degenerate_pixel_box",
                coord_token_ids=None,
            )
        else:
            bins = canvas.pixel_to_bins(proposed)
            if bins is None:
                record.update(
                    admitted=False,
                    invalid_reason="collapsed_after_pixel_to_bin_conversion",
                    coord_token_ids=None,
                )
            else:
                decoded = canvas.bins_to_pixel(bins)
                record.update(
                    admitted=True,
                    invalid_reason=None,
                    coord_bins=list(bins),
                    coord_token_ids=coord_token_ids(bins),
                    decoded_bbox_pixel_xyxy=list(decoded),
                    # Geometry against this role's own generating owner.
                    geometry=candidate_generator_geometry(decoded, owner_box),
                )
        if not record["admitted"]:
            record.setdefault("decoded_bbox_pixel_xyxy", None)
            record.setdefault("geometry", None)
        records.append(record)
    if len(records) != LOGICAL_ROLE_COUNT:
        raise PlanContractError(
            f"owner {owner['gt_owner_id']} did not produce exactly seventeen logical roles"
        )
    return records


#: Partition of a physical candidate relative to one owner.  The merge derives
#: every per-owner-context maximum from these labels, so the semantics live
#: here rather than being re-derived downstream.
#:
#: ``generator_local``      -- this owner generated the candidate and it is not
#:                             strictly assigned to a *different* owner; it is
#:                             the owner's primary landscape.
#: ``strict_assigned_self`` -- additionally strictly assigned to this owner.
#: ``ambiguous_upper``      -- ambiguity-neutral; counts toward the owner's
#:                             upper (``U``) bound only.
#: ``other_owner_strict``   -- strictly assigned to a different owner;
#:                             **excluded** from this owner's support and moved
#:                             to the collision diagnostic.
OWNER_CANDIDATE_PARTITIONS: tuple[str, ...] = (
    "strict_assigned_self",
    "ambiguous_upper",
    "unmatched_generator_local",
    "other_owner_strict",
    "not_generated_by_owner",
)


def classify_candidate_for_owner(
    candidate: Mapping[str, Any], owner_id: str
) -> dict[str, Any]:
    """Partition one physical candidate relative to one owner.

    The primary owner neighbourhood is the *generator-local landscape*: the
    candidates this owner generated.  Strict assignment is a separate,
    narrower, owner-identifiable lower bound and is never bank membership.

    A candidate strictly assigned to another owner is excluded from this
    owner's support entirely and belongs to the collision diagnostic; an
    ambiguity-neutral candidate counts toward the upper bound only.
    """

    generated = any(
        str(row["generator_gt_owner_id"]) == str(owner_id)
        for row in candidate.get("generators", ())
    )
    status = str(candidate.get("strict_assignment_status"))
    assigned = candidate.get("strict_assignment_gt_owner_id")

    if status == "matched" and str(assigned) != str(owner_id):
        partition = "other_owner_strict"
    elif not generated:
        partition = "not_generated_by_owner"
    elif status == "matched":
        partition = "strict_assigned_self"
    elif status == "ambiguous_neutral":
        partition = "ambiguous_upper"
    else:
        partition = "unmatched_generator_local"

    return {
        "partition": partition,
        "generator_local": generated and partition != "other_owner_strict",
        # Lower bound: only candidates uniquely assigned to this owner.
        "counts_toward_lower_bound": partition == "strict_assigned_self",
        # Upper bound: the generator-local landscape plus ambiguity-neutral.
        "counts_toward_upper_bound": partition
        in {"strict_assigned_self", "ambiguous_upper", "unmatched_generator_local"},
        "excluded_from_target_support": partition == "other_owner_strict",
        "collision_diagnostic": partition == "other_owner_strict",
    }


def physical_candidate_id(
    *, image_id: str, normalized_description: str, tokens: Sequence[int]
) -> str:
    """Content-addressed physical candidate identity.

    Identity is ``digest(image, category, coord tokens)`` and deliberately
    excludes the generating owner: two owners of the same category that realize
    the same coordinate tuple are **one** physical candidate, scored once.
    Generator provenance is retained separately and never enters identity,
    rank, or assignment.
    """

    digest = sha256_json(
        {
            "image_id": str(image_id),
            "normalized_description": str(normalized_description),
            "coord_token_ids": [int(v) for v in tokens],
        }
    )
    return f"cand:{digest[:24]}"


def build_candidate_bank(
    owners_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
    panel: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Collapse token-identical candidates once within ``(image, category)``.

    Returns ``(physical_candidates, owner_bank_accounting)``.

    Collapse is *cross-owner*: within one ``(image, normalized_description)``
    every distinct coordinate-token tuple becomes exactly one physical
    candidate, so a tuple two owners both generate contributes one request and
    one rank mass rather than two.  Geometry assignment is computed once,
    category-locally, from the decoded box; the generator list is provenance
    only.
    """

    physical_by_key: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    role_records_by_owner: dict[str, list[dict[str, Any]]] = {}

    for image_id in sorted(owners_by_image, key=int):
        canvas = Canvas(panel[image_id]["width"], panel[image_id]["height"])
        image_owners = owners_by_image[image_id]
        for owner in image_owners:
            owner_id = str(owner["gt_owner_id"])
            description = str(owner["normalized_description"])
            records = realize_owner_roles(owner, canvas=canvas)
            role_records_by_owner[owner_id] = records
            for record in records:
                if not record["admitted"]:
                    continue
                tokens = tuple(int(v) for v in record["coord_token_ids"])
                key = (image_id, description, tokens)
                existing = physical_by_key.get(key)
                if existing is None:
                    decoded = canvas.bins_to_pixel(record["coord_bins"])
                    candidate_id = physical_candidate_id(
                        image_id=image_id, normalized_description=description, tokens=tokens
                    )
                    existing = {
                        "schema_version": PLAN_SCHEMA_VERSION,
                        "row_kind": "physical_candidate",
                        "candidate_id": candidate_id,
                        "image_id": image_id,
                        "normalized_description": description,
                        "coord_bins": list(record["coord_bins"]),
                        "coord_token_ids": list(tokens),
                        "coord_token_ids_sha256": sha256_json(list(tokens)),
                        "decoded_bbox_pixel_xyxy": list(decoded),
                        "identity_rule": "digest_image_category_coord_tokens",
                        "collapse_scope": "image_and_normalized_description",
                        "selection_policy": "fixed_seventeen_role_order_without_scores",
                        "generators": [],
                        # Assignment is computed once, category-locally, from
                        # the decoded box.  It never consults the generator.
                        **strict_assignment(
                            decoded, image_owners, normalized_description=description
                        ),
                    }
                    physical_by_key[key] = existing
                existing["generators"].append(
                    {
                        "generator_gt_owner_id": owner_id,
                        "logical_transform_role": record["role"],
                        "role_ordinal": record["role_ordinal"],
                        "candidate_class": record["candidate_class"],
                        # Per-generator: after cross-owner collapse the same
                        # coordinate tuple has different geometry against each
                        # generating owner's own GT box.
                        "geometry": record["geometry"],
                    }
                )
                record["physical_candidate_id"] = existing["candidate_id"]

    physical: list[dict[str, Any]] = []
    for key in sorted(physical_by_key, key=lambda item: (item[0], item[1], item[2])):
        candidate = physical_by_key[key]
        candidate["generators"].sort(
            key=lambda row: (str(row["generator_gt_owner_id"]), int(row["role_ordinal"]))
        )
        generator_owner_ids = sorted({row["generator_gt_owner_id"] for row in candidate["generators"]})
        first = candidate["generators"][0]
        candidate["generator_gt_owner_ids"] = generator_owner_ids
        candidate["generator_owner_count"] = len(generator_owner_ids)
        candidate["cross_owner_generated"] = len(generator_owner_ids) > 1
        candidate["representative_role"] = first["logical_transform_role"]
        candidate["representative_role_ordinal"] = first["role_ordinal"]
        # A single, explicitly-named geometry view for consumers that want one
        # row per candidate.  It names its generator, so a cross-owner
        # collapsed candidate can never have one generator's numbers silently
        # read as if they applied to all of them; ``generators`` remains the
        # authoritative per-generator record.
        candidate["representative_generator_geometry"] = {
            "generator_gt_owner_id": first["generator_gt_owner_id"],
            "logical_transform_role": first["logical_transform_role"],
            **(first["geometry"] or {}),
        }
        candidate["geometry_scope"] = (
            "per_generator_authoritative_in_generators_representative_view_is_a_summary"
        )
        candidate["candidate_class"] = (
            "core"
            if any(row["candidate_class"] == "core" for row in candidate["generators"])
            else "extension"
        )
        candidate["candidate_provenance"] = (
            "exact"
            if any(
                row["logical_transform_role"] == "exact_gt_anchor"
                for row in candidate["generators"]
            )
            else ("neighborhood" if candidate["candidate_class"] == "core" else "extension")
        )
        candidate["generator_provenance_role"] = "provenance_only_never_rank_or_assignment"
        candidate["sidecar_provenance"] = []
        physical.append(candidate)

    physical_by_id = {str(row["candidate_id"]): row for row in physical}
    accounting: dict[str, dict[str, Any]] = {}
    for owner_id, records in role_records_by_owner.items():
        reached_ids = sorted(
            {str(row["physical_candidate_id"]) for row in records if row.get("physical_candidate_id")}
        )
        uniquely_assigned = 0
        lost_other = 0
        lost_ambiguous = 0
        lost_unmatched = 0
        for candidate_id in reached_ids:
            candidate = physical_by_id[candidate_id]
            status = candidate["strict_assignment_status"]
            if status == "matched":
                if candidate["strict_assignment_gt_owner_id"] == owner_id:
                    uniquely_assigned += 1
                else:
                    lost_other += 1
            elif status == "ambiguous_neutral":
                lost_ambiguous += 1
            else:
                lost_unmatched += 1

        exact_record = next(row for row in records if row["role"] == "exact_gt_anchor")
        exact_candidate_id = exact_record.get("physical_candidate_id")
        exact_self_assigned = bool(
            exact_candidate_id
            and physical_by_id[exact_candidate_id]["strict_assignment_gt_owner_id"] == owner_id
        )
        reached = len(reached_ids)

        # View 1 -- generator-local bank adequacy: how much of the intended
        # seventeen-role landscape exists as distinct scoreable geometry.  This
        # is what qualifies an owner's evidence.
        if not exact_self_assigned or reached < BANK_COVERAGE_ADEQUATE_REACHED_AT_LEAST:
            adequacy = "undercovered_unresolved_only"
        elif reached >= BANK_COVERAGE_FULL_REACHED_AT_LEAST:
            adequacy = "full"
        else:
            adequacy = "adequate_reduced"
        disposition_eligible = adequacy in DISPOSITION_ELIGIBLE_BANK_STATUSES

        admitted_roles = [row for row in records if row["admitted"]]
        accounting[owner_id] = {
            "logical_role_count": LOGICAL_ROLE_COUNT,
            "core_role_count": CORE_ROLE_COUNT,
            "extension_role_count": EXTENSION_ROLE_COUNT,
            "admitted_role_count": len(admitted_roles),
            "not_admitted_role_count": LOGICAL_ROLE_COUNT - len(admitted_roles),
            # Fable's canonical adequacy name, plus the descriptive alias.
            "distinct_physical_candidate_count": reached,
            "distinct_physical_candidates_reached": reached,
            "physical_candidate_ids": reached_ids,
            "generator_local_landscape_candidate_ids": [
                candidate_id
                for candidate_id in reached_ids
                if not classify_candidate_for_owner(physical_by_id[candidate_id], owner_id)[
                    "excluded_from_target_support"
                ]
            ],
            "other_owner_strict_count": lost_other,
            "cross_owner_shared_candidate_count": sum(
                1
                for candidate_id in reached_ids
                if physical_by_id[candidate_id]["cross_owner_generated"]
            ),
            "exact_anchor_admitted": exact_record["admitted"],
            "exact_anchor_uniquely_self_assigned": exact_self_assigned,
            # The two views are kept explicitly separate and both are measured.
            # They answer different questions and must never be conflated.
            "generator_local_bank_adequacy": {
                "status": adequacy,
                "measured_on": "distinct_physical_candidate_count",
                "distinct_physical_candidate_count": reached,
                "exact_anchor_uniquely_self_assigned": exact_self_assigned,
                "full_at_least": BANK_COVERAGE_FULL_REACHED_AT_LEAST,
                "adequate_at_least": BANK_COVERAGE_ADEQUATE_REACHED_AT_LEAST,
                "disposition_eligible": disposition_eligible,
                "threshold_status": "frozen_before_any_score",
                "rationale": (
                    "the seventeen score-independent probes remain valid localization "
                    "landscape samples even when a translated or expanded box drops "
                    "below IoU 0.5; an owner is never floored merely because a few "
                    "transforms produced token-identical boxes"
                ),
            },
            "strict_assignment_coverage": {
                "uniquely_assigned_candidate_count": uniquely_assigned,
                "roles_lost_to_other_owner_assignment": lost_other,
                "roles_lost_to_ambiguous_assignment": lost_ambiguous,
                "roles_lost_to_unmatched_assignment": lost_unmatched,
                "role": "separate_lower_bound_view_never_the_adequacy_gate",
                "note": (
                    "perturbations are designed to move off the owner; a low count here "
                    "is expected and is not evidence of an inadequate bank"
                ),
            },
            "bank_coverage_status": adequacy,
            "disposition_eligible": disposition_eligible,
            "undercovered": not disposition_eligible,
            # Contract protection: only a genuinely undercovered bank floors an
            # owner to "unresolved".  ``adequate_reduced`` stays visibly flagged
            # but keeps full disposition eligibility.
            "disposition_floor": (
                "no_floor"
                if disposition_eligible
                else "unresolved_only_never_persistent_no_tested_localization_support"
            ),
            "logical_roles": records,
        }
    return physical, accounting


# ---------------------------------------------------------------------------
# Frozen input loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourcePaths:
    panel: Path = PANEL_PATH
    owner_ledger: Path = OWNER_LEDGER_PATH
    prediction_ledger: Path = PREDICTION_LEDGER_PATH
    greedy: Path = GREEDY_PATH


def load_panel(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path, "frozen panel")
    panel: dict[str, dict[str, Any]] = {}
    for row in rows:
        image_id = str(row["image_id"])
        panel[image_id] = {
            "image_id": image_id,
            "width": int(row["width"]),
            "height": int(row["height"]),
            "file_name": str(row.get("file_name", "")),
            "images": list(row.get("images", [])),
        }
    if len(panel) != 12:
        raise PlanContractError(f"frozen panel has {len(panel)} images, expected 12")
    return panel


def load_owners(path: Path, panel: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Load the canonical 346-owner census; never reconstruct it from predictions."""

    rows = _read_jsonl(path, "canonical owner ledger")
    if len(rows) != EXPECTED_OWNER_COUNT:
        raise PlanContractError(
            f"canonical owner ledger has {len(rows)} rows, expected {EXPECTED_OWNER_COUNT}"
        )
    owners: list[dict[str, Any]] = []
    eligible_count = 0
    for row in rows:
        image_id = str(row["image_id"])
        if image_id not in panel:
            raise PlanContractError(f"owner ledger references image {image_id} absent from the panel")
        eligibility = row.get("decision_eligibility", {}).get("greedy_natural", {})
        greedy_eligible = bool(eligibility.get("eligible", False))
        eligible_count += int(greedy_eligible)
        box = [int(round(float(v))) for v in row["bbox_xyxy"]]
        owners.append(
            {
                "gt_owner_id": str(row["gt_owner_id"]),
                "image_id": image_id,
                "normalized_description": str(row["normalized_description"]),
                "description": str(row["description"]),
                "official_coco_category_id": row.get("official_coco_category_id"),
                "original_annotation_index": row.get("original_annotation_index"),
                "bbox_pixel_xyxy": box,
                "owner_sort_key": [box[1], box[0]],
                "greedy_eligible": greedy_eligible,
                "greedy_eligibility_status": str(eligibility.get("status", "unknown")),
                "split": SPLIT_BY_IMAGE_ID[image_id],
            }
        )
    if eligible_count != EXPECTED_GREEDY_ELIGIBLE_COUNT:
        raise PlanContractError(
            f"owner ledger has {eligible_count} greedy-eligible owners, "
            f"expected {EXPECTED_GREEDY_ELIGIBLE_COUNT}"
        )
    observed_images = {owner["image_id"] for owner in owners}
    expected_images = set(SPLIT_BY_IMAGE_ID)
    if observed_images != expected_images:
        raise PlanContractError(
            "owner ledger image set does not match the frozen discovery/confirmation split"
        )
    owners.sort(key=lambda owner: (owner["image_id"], owner["gt_owner_id"]))
    return owners


@dataclass(frozen=True)
class NativeRollout:
    image_id: str
    prompt_token_ids: list[int]
    prompt_token_ids_sha256: str
    generated_token_ids: list[int]
    generated_token_ids_sha256: str
    executed_media_sha256: str
    row_token_spans: list[list[int]]
    predictions: list[Mapping[str, Any]]
    stop_reason: str
    seed: int
    decode_mode: str


def split_generated_rows(tokens: Sequence[int]) -> list[list[int]]:
    """Split native generated tokens into complete closed one-box rows.

    Enforces the closed grammar (``object_ref_start``, description tokens,
    ``object_ref_end``, ``box_start``, four coordinate tokens, ``box_end``) so
    a malformed or truncated row can never silently enter a prefix.
    """

    starts = [index for index, token in enumerate(tokens) if token == OBJECT_REF_START]
    if not starts or starts[0] != 0:
        raise PlanContractError("native generated tokens do not begin with an object row")
    ends = [*starts[1:], len(tokens)]
    rows = [list(tokens[start:end]) for start, end in zip(starts, ends, strict=True)]
    for index, row in enumerate(rows):
        if len(row) < 8 or row[0] != OBJECT_REF_START:
            raise PlanContractError(f"native row {index} has an invalid object start")
        try:
            object_end = row.index(OBJECT_REF_END)
        except ValueError as exc:
            raise PlanContractError(f"native row {index} lacks object_ref_end") from exc
        if object_end <= 1 or row[object_end + 1 : object_end + 2] != [BOX_START]:
            raise PlanContractError(f"native row {index} has an invalid object/box boundary")
        coordinate = row[object_end + 2 : object_end + 6]
        if len(coordinate) != 4 or any(
            token < COORD_TOKEN_START or token > COORD_TOKEN_END for token in coordinate
        ):
            raise PlanContractError(f"native row {index} lacks four coordinate tokens")
        if row[object_end + 6 :] != [BOX_END]:
            raise PlanContractError(f"native row {index} violates the closed one-box grammar")
    return rows


def row_description_token_ids(row: Sequence[int]) -> list[int]:
    return list(row[1 : row.index(OBJECT_REF_END)])


def row_coord_token_ids(row: Sequence[int]) -> list[int]:
    object_end = row.index(OBJECT_REF_END)
    return list(row[object_end + 2 : object_end + 6])


def load_native_rollouts(path: Path) -> dict[str, NativeRollout]:
    payload = _read_json(path, "native sorted greedy rollout")
    rollouts = payload.get("rollouts")
    if not isinstance(rollouts, list) or len(rollouts) != 12:
        raise PlanContractError("native greedy rollout does not contain exactly twelve rollouts")
    loaded: dict[str, NativeRollout] = {}
    total_rows = 0
    for rollout in rollouts:
        if str(rollout.get("decode_mode")) != "greedy":
            raise PlanContractError("a native rollout is not a greedy decode")
        image_id = str(rollout["image_id"])
        tokens = [int(value) for value in rollout["generated_token_ids"]]
        spans = split_generated_rows(tokens)
        # ``predictions`` is the parser envelope; the per-row list is nested
        # one level deeper.
        envelope = rollout["predictions"]
        predictions = list(envelope["predictions"])
        if envelope.get("parse_status") != "accepted" or int(
            envelope.get("dropped_prediction_count", 0)
        ):
            raise PlanContractError(f"image {image_id} native rollout has dropped or rejected rows")
        if len(predictions) != len(spans):
            raise PlanContractError(
                f"image {image_id} native row count disagrees with its parsed predictions"
            )
        total_rows += len(spans)
        loaded[image_id] = NativeRollout(
            image_id=image_id,
            prompt_token_ids=[int(v) for v in rollout["prompt_token_ids"]],
            prompt_token_ids_sha256=str(rollout["prompt_token_ids_sha256"]),
            generated_token_ids=tokens,
            generated_token_ids_sha256=str(rollout["generated_token_ids_sha256"]),
            executed_media_sha256=str(rollout["executed_media_sha256"]),
            row_token_spans=spans,
            predictions=predictions,
            stop_reason=str(rollout["stop_reason"]),
            seed=int(rollout["seed"]),
            decode_mode="greedy",
        )
    if total_rows != EXPECTED_NATIVE_ROW_COUNT:
        raise PlanContractError(
            f"native rollouts contain {total_rows} complete rows, expected {EXPECTED_NATIVE_ROW_COUNT}"
        )
    if set(loaded) != set(SPLIT_BY_IMAGE_ID):
        raise PlanContractError("native rollout image set does not match the frozen split")
    return loaded


def load_native_row_ledger(path: Path) -> dict[tuple[str, int], Mapping[str, Any]]:
    """Index the canonical greedy prediction rows by ``(image_id, row_index)``."""

    rows = _read_jsonl(path, "canonical prediction-row ledger")
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("decode_mode")) != "greedy":
            continue
        key = (str(row["image_id"]), int(row["original_row_index"]))
        if key in indexed:
            raise PlanContractError(f"prediction-row ledger has a duplicate greedy row {key}")
        indexed[key] = row
    if len(indexed) != EXPECTED_NATIVE_ROW_COUNT:
        raise PlanContractError(
            f"prediction-row ledger has {len(indexed)} greedy rows, "
            f"expected {EXPECTED_NATIVE_ROW_COUNT}"
        )
    return indexed


# ---------------------------------------------------------------------------
# Category token resolution
# ---------------------------------------------------------------------------


class CategoryTokenResolver(Protocol):
    """Resolves a normalized description to its exact category token IDs."""

    def __call__(self, description: str) -> list[int] | None: ...


def build_observed_category_tokens(
    rollouts: Mapping[str, NativeRollout],
) -> dict[str, list[int]]:
    """Derive category token spans from the exact native rows, never hardcoded.

    A description whose observed spans are not uniform across every native row
    is a hard error rather than a silently-picked span.
    """

    observed: dict[str, set[tuple[int, ...]]] = {}
    for rollout in rollouts.values():
        for span, prediction in zip(rollout.row_token_spans, rollout.predictions, strict=True):
            description = str(prediction["description"])
            observed.setdefault(description, set()).add(tuple(row_description_token_ids(span)))
    resolved: dict[str, list[int]] = {}
    for description, spans in observed.items():
        if len(spans) != 1:
            raise PlanContractError(
                f"description {description!r} has non-uniform observed category token spans"
            )
        resolved[description] = list(next(iter(spans)))
    return resolved


def build_tokenizer_resolver(tokenizer_path: str, observed: Mapping[str, Sequence[int]]):
    """Build a tokenizer fallback, validated against every observed span.

    The fallback is only trustworthy if re-encoding each natively observed
    description reproduces its observed span exactly; any mismatch fails
    closed rather than silently admitting a differently-tokenized category.
    """

    from transformers import AutoTokenizer  # imported lazily: CPU-only, optional

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    mismatches = [
        description
        for description, span in sorted(observed.items())
        if tokenizer.encode(description, add_special_tokens=False) != list(span)
    ]
    if mismatches:
        raise PlanContractError(
            "tokenizer fallback disagrees with observed native category spans for "
            f"{mismatches}; the fallback cannot be trusted for unobserved categories"
        )

    def resolve(description: str) -> list[int] | None:
        encoded = [int(value) for value in tokenizer.encode(description, add_special_tokens=False)]
        return encoded or None

    return resolve


# ---------------------------------------------------------------------------
# Registry construction
# ---------------------------------------------------------------------------


def build_image_registry(
    panel: Mapping[str, Mapping[str, Any]], rollouts: Mapping[str, NativeRollout]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image_id in sorted(panel, key=int):
        rollout = rollouts[image_id]
        rows.append(
            {
                "schema_version": PLAN_SCHEMA_VERSION,
                "row_kind": "census_image",
                "image_id": image_id,
                "split": SPLIT_BY_IMAGE_ID[image_id],
                "image_width": panel[image_id]["width"],
                "image_height": panel[image_id]["height"],
                "file_name": panel[image_id]["file_name"],
                "prompt_token_ids": rollout.prompt_token_ids,
                "prompt_token_ids_sha256": rollout.prompt_token_ids_sha256,
                "generated_token_ids_sha256": rollout.generated_token_ids_sha256,
                "executed_media_sha256": rollout.executed_media_sha256,
                "native_complete_row_count": len(rollout.row_token_spans),
                "native_stop_reason": rollout.stop_reason,
                "native_seed": rollout.seed,
                "wrapper_token_ids": dict(WRAPPER_TOKEN_IDS),
                "coordinate_token_ids": {
                    "start": COORD_TOKEN_START,
                    "end_inclusive": COORD_TOKEN_END,
                    "bin_count": COORD_BIN_COUNT,
                },
            }
        )
    return rows


def terminal_kind_for(stop_reason: str) -> str:
    if stop_reason == "im_end":
        return "natural_stop"
    if stop_reason in {"length", "max_new_tokens", "length_truncated"}:
        return "length_truncated"
    return "other"


def build_context_registry(
    rollouts: Mapping[str, NativeRollout],
    row_ledger: Mapping[tuple[str, int], Mapping[str, Any]],
    owners_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
    panel: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Root plus every complete native row boundary plus terminal, per image.

    Only natively generated complete rows enter a prefix; no forced-continue
    row is ever admitted.  Loop marking is continuous and score-independent.
    """

    contexts: list[dict[str, Any]] = []
    for image_id in sorted(rollouts, key=int):
        rollout = rollouts[image_id]
        canvas = Canvas(panel[image_id]["width"], panel[image_id]["height"])
        image_owners = owners_by_image[image_id]
        spans = rollout.row_token_spans
        total = len(spans)

        span_tuples = [tuple(span) for span in spans]
        prior_identical = [
            sum(1 for j in range(index) if span_tuples[j] == span_tuples[index])
            for index in range(total)
        ]
        consecutive_run: list[int] = []
        for index in range(total):
            if index and span_tuples[index] == span_tuples[index - 1]:
                consecutive_run.append(consecutive_run[-1] + 1)
            else:
                consecutive_run.append(1)

        for boundary_index in range(total + 1):
            row_indices = list(range(boundary_index))
            prefix_rows = []
            for row_index in row_indices:
                ledger_row = row_ledger[(image_id, row_index)]
                prediction = rollout.predictions[row_index]
                prefix_rows.append(
                    {
                        "row_index": row_index,
                        "pred_row_id": str(ledger_row["pred_row_id"]),
                        "description": str(prediction["description"]),
                        "raw_span_sha256": str(ledger_row["raw_span_sha256"]),
                        "strict_match_status": str(ledger_row["strict_match_status"]),
                        "strict_match_gt_owner_id": ledger_row.get("strict_match_gt_owner_id"),
                    }
                )
            generated_prefix = [token for index in row_indices for token in spans[index]]
            observed_self_prefix = [*rollout.prompt_token_ids, *generated_prefix]

            if boundary_index == 0:
                context_role = "root"
            elif boundary_index == total:
                context_role = "terminal"
            else:
                context_role = "row_boundary"

            last_index = boundary_index - 1
            if last_index >= 0:
                last_ledger = row_ledger[(image_id, last_index)]
                last_box = [float(v) for v in last_ledger["bbox_xyxy"]]
                frontier = {
                    "row_index": last_index,
                    "pred_row_id": str(last_ledger["pred_row_id"]),
                    "description": str(rollout.predictions[last_index]["description"]),
                    "bbox_pixel_xyxy": last_box,
                    "sort_key": [last_box[1], last_box[0]],
                    "strict_match_status": str(last_ledger["strict_match_status"]),
                    "strict_match_gt_owner_id": last_ledger.get("strict_match_gt_owner_id"),
                }
                loop_counts = {
                    "prior_identical_row_count": prior_identical[last_index],
                    "consecutive_identical_row_run_length": consecutive_run[last_index],
                    "repeated_raw_span_sha256": (
                        str(last_ledger["raw_span_sha256"])
                        if prior_identical[last_index] > 0
                        else None
                    ),
                }
            else:
                frontier = None
                loop_counts = {
                    "prior_identical_row_count": None,
                    "consecutive_identical_row_run_length": None,
                    "repeated_raw_span_sha256": None,
                }

            run_length = loop_counts["consecutive_identical_row_run_length"]
            loop_tail = bool(run_length is not None and run_length >= LOOP_TAIL_MIN_CONSECUTIVE_RUN)

            contexts.append(
                {
                    "schema_version": PLAN_SCHEMA_VERSION,
                    "row_kind": "census_context",
                    "context_id": f"{image_id}:boundary-{boundary_index:03d}",
                    "image_id": image_id,
                    "split": SPLIT_BY_IMAGE_ID[image_id],
                    "boundary_index": boundary_index,
                    "total_complete_row_count": total,
                    "context_role": context_role,
                    "terminal_kind": (
                        terminal_kind_for(rollout.stop_reason)
                        if context_role == "terminal"
                        else None
                    ),
                    "prefix_row_indices": row_indices,
                    "prefix_row_ids": [row["pred_row_id"] for row in prefix_rows],
                    "prefix_rows": prefix_rows,
                    "prompt_token_ids_sha256": rollout.prompt_token_ids_sha256,
                    "generated_prefix_token_ids": generated_prefix,
                    "generated_prefix_token_ids_sha256": sha256_json(generated_prefix),
                    "observed_self_prefix_token_ids_sha256": sha256_json(observed_self_prefix),
                    "observed_self_prefix_token_count": len(observed_self_prefix),
                    "prefix_admission": {
                        "source": "native_greedy_complete_rows_only",
                        "forced_continue_rows_excluded": True,
                        "retokenized": False,
                    },
                    "loop_marking": {
                        **loop_counts,
                        "loop_tail": loop_tail,
                        "loop_tail_rule": (
                            f"consecutive_identical_row_run_length >= {LOOP_TAIL_MIN_CONSECUTIVE_RUN}"
                        ),
                        "flag_is_not_a_mechanism_label": True,
                    },
                    "frontier": frontier,
                    "image_owner_count": len(image_owners),
                    "canvas": {"width": canvas.width, "height": canvas.height},
                }
            )
    if len(contexts) != EXPECTED_CONTEXT_COUNT:
        raise PlanContractError(
            f"context registry has {len(contexts)} rows, expected {EXPECTED_CONTEXT_COUNT}"
        )
    return contexts


def build_category_registry(
    owners_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
    observed_tokens: Mapping[str, Sequence[int]],
    fallback: CategoryTokenResolver | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image_id in sorted(owners_by_image, key=int):
        owners = owners_by_image[image_id]
        descriptions = sorted({owner["normalized_description"] for owner in owners})
        for description in descriptions:
            members = [
                owner["gt_owner_id"]
                for owner in owners
                if owner["normalized_description"] == description
            ]
            tokens: list[int] | None = None
            token_source = "unresolved"
            if description in observed_tokens:
                tokens = list(observed_tokens[description])
                token_source = "observed_native_row"
            elif fallback is not None:
                resolved = fallback(description)
                if resolved:
                    tokens = list(resolved)
                    token_source = "tokenizer_encode"
            row: dict[str, Any] = {
                "schema_version": PLAN_SCHEMA_VERSION,
                "row_kind": "census_category",
                "category_query_id": f"{image_id}:{description}",
                "image_id": image_id,
                "split": SPLIT_BY_IMAGE_ID[image_id],
                "normalized_description": description,
                "owner_count_in_image": len(members),
                "owner_ids": sorted(members),
                "token_source": token_source,
                # Contract item 2: this is the estimand this category query
                # measures.  It is never a per-owner proposal probability.
                "estimand_name": "category_field_support_at_owner_geometry",
            }
            if tokens is None:
                row.update(
                    status="blocked_unresolved_category_tokens",
                    category_token_ids=None,
                    category_token_ids_sha256=None,
                    query_suffix_token_ids=None,
                    query_suffix_token_ids_sha256=None,
                )
            else:
                suffix = build_query_suffix(tokens)
                row.update(
                    status="admitted",
                    category_token_ids=tokens,
                    category_token_ids_sha256=sha256_json(tokens),
                    query_suffix_token_ids=suffix,
                    query_suffix_token_ids_sha256=sha256_json(suffix),
                )
            rows.append(row)
    return rows


def build_query_group_registry(
    contexts: Sequence[Mapping[str, Any]],
    categories: Sequence[Mapping[str, Any]],
    rollouts: Mapping[str, NativeRollout],
    bank_by_category: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """One singleton scoring work item per ``(context, category)``.

    The query suffix is category-dependent, so the cache/prefill unit is the
    pair, not the context alone; vision encoding is still shared per image.

    Each row is a *singleton group*: exactly one
    ``(image_id, context_id, observed_prefix_sha256, query_prefix_sha256)``
    tuple.  Strict cache admission is only valid for the group it was measured
    on, so a scoring unit that mixed groups would silently reuse one group's
    admission for another.  The scorer validates this invariant fail-fast.

    ``observed_prefix_sha256`` (prompt plus native rows) and
    ``query_prefix_sha256`` (that prefix plus the canonical query suffix) are
    kept as *distinct* digests so a score row can never be joined to a
    differently-suffixed capture.
    """

    categories_by_image: dict[str, list[Mapping[str, Any]]] = {}
    for category in categories:
        categories_by_image.setdefault(str(category["image_id"]), []).append(category)

    rows: list[dict[str, Any]] = []
    for context in contexts:
        image_id = str(context["image_id"])
        rollout = rollouts[image_id]
        observed_self_prefix = [
            *rollout.prompt_token_ids,
            *context["generated_prefix_token_ids"],
        ]
        for category in sorted(
            categories_by_image[image_id], key=lambda row: str(row["normalized_description"])
        ):
            description = str(category["normalized_description"])
            group_id = f"{context['context_id']}|{description}"
            if category["status"] != "admitted":
                rows.append(
                    {
                        "schema_version": PLAN_SCHEMA_VERSION,
                        "row_kind": "census_query_group",
                        "query_group_id": group_id,
                        "image_id": image_id,
                        "split": SPLIT_BY_IMAGE_ID[image_id],
                        "context_id": context["context_id"],
                        "category_query_id": category["category_query_id"],
                        "normalized_description": description,
                        "status": "blocked_unresolved_category_tokens",
                        "candidate_ids": [],
                        "candidate_count": 0,
                        "observed_prefix_sha256": sha256_json(observed_self_prefix),
                        "query_prefix_sha256": None,
                        "query_suffix_token_ids_sha256": None,
                        "query_prefix_token_count": None,
                    }
                )
                continue
            suffix = list(category["query_suffix_token_ids"])
            full_prefix = [*observed_self_prefix, *suffix]
            assert_canonical_query_suffix(
                full_prefix,
                category_token_ids=category["category_token_ids"],
                label=f"query group {group_id}",
            )
            observed_sha = sha256_json(observed_self_prefix)
            query_sha = sha256_json(full_prefix)

            # Exactly one scoring request per unique coordinate tuple in this
            # (image, category) bank.  The bank is already collapsed, so a
            # duplicate here is a contract violation, not something to dedupe
            # quietly.
            bank = list(bank_by_category.get((image_id, description), ()))
            candidate_ids = [str(candidate["candidate_id"]) for candidate in bank]
            token_tuples = [tuple(candidate["coord_token_ids"]) for candidate in bank]
            if len(set(candidate_ids)) != len(candidate_ids):
                raise PlanContractError(
                    f"query group {group_id}: duplicate physical candidate IDs"
                )
            if len(set(token_tuples)) != len(token_tuples):
                raise PlanContractError(
                    f"query group {group_id}: duplicate coordinate tuples survived collapse"
                )
            rows.append(
                {
                    "schema_version": PLAN_SCHEMA_VERSION,
                    "row_kind": "census_query_group",
                    "query_group_id": group_id,
                    "image_id": image_id,
                    "split": SPLIT_BY_IMAGE_ID[image_id],
                    "context_id": context["context_id"],
                    "category_query_id": category["category_query_id"],
                    "normalized_description": description,
                    "status": "admitted",
                    "query_suffix_token_ids": suffix,
                    "query_suffix_token_ids_sha256": sha256_json(suffix),
                    "suffix_shape_class": suffix_shape_class(category["category_token_ids"]),
                    "observed_prefix_sha256": observed_sha,
                    "query_prefix_sha256": query_sha,
                    "query_prefix_token_count": len(full_prefix),
                    "observed_prefix_token_count": len(observed_self_prefix),
                    "singleton_group_key": {
                        "image_id": image_id,
                        "context_id": context["context_id"],
                        "normalized_description": description,
                        "observed_prefix_sha256": observed_sha,
                        "query_prefix_sha256": query_sha,
                    },
                    # Admission identity is the exact query prefix for this
                    # channel; the proposal channel is admitted separately on
                    # the observed prefix.
                    "admission_receipt_id": admission_receipt_id(
                        context_id=str(context["context_id"]),
                        channel=CHANNEL_QUERY_SUFFIX,
                        prefix_sha256=query_sha,
                    ),
                    # Context-level: the gate reads the observed prefix with
                    # nothing forced, so every category in this context shares
                    # this one receipt.
                    "proposal_boundary_gate_admission_receipt_id": admission_receipt_id(
                        context_id=str(context["context_id"]),
                        channel=CHANNEL_PROPOSAL_BOUNDARY_GATE,
                        prefix_sha256=observed_sha,
                    ),
                    # Per (context, category): distinct routing token identity
                    # and length, so never shared across categories.
                    "proposal_route_admission_receipt_id": (
                        proposal_route_admission_receipt_id(
                            context_id=str(context["context_id"]),
                            observed_prefix_sha256=observed_sha,
                            category_token_ids=category["category_token_ids"],
                        )
                    ),
                    "proposal_route_token_ids": proposal_route_token_ids(
                        category["category_token_ids"]
                    ),
                    "proposal_route_digest": proposal_route_digest(
                        category["category_token_ids"]
                    ),
                    "proposal_is_per_category_never_per_owner": True,
                    "rank_key": {
                        "image_id": image_id,
                        "context_id": context["context_id"],
                        "normalized_description": description,
                    },
                    "candidate_ids": sorted(candidate_ids),
                    "candidate_count": len(candidate_ids),
                    "unique_coordinate_tuple_count": len(set(token_tuples)),
                    "request_policy": "exactly_one_request_per_unique_coordinate_tuple",
                }
            )
    return rows


def build_native_sidecar_registry(
    rollouts: Mapping[str, NativeRollout],
    row_ledger: Mapping[tuple[str, int], Mapping[str, Any]],
    bank_token_index: Mapping[tuple[str, str], Mapping[tuple[int, ...], str]],
) -> list[dict[str, Any]]:
    """The 400 native emitted boxes, explicitly excluded from every core rank.

    A sidecar whose coordinate tuple is identical to a bank candidate's *joins
    that candidate's provenance* -- it never adds a second rank mass.
    """

    rows: list[dict[str, Any]] = []
    for image_id in sorted(rollouts, key=int):
        rollout = rollouts[image_id]
        for row_index, span in enumerate(rollout.row_token_spans):
            ledger_row = row_ledger[(image_id, row_index)]
            tokens = tuple(row_coord_token_ids(span))
            description = str(ledger_row["normalized_description"])
            joined = bank_token_index.get((image_id, description), {}).get(tokens)
            rows.append(
                {
                    "schema_version": PLAN_SCHEMA_VERSION,
                    "row_kind": "native_sidecar",
                    "sidecar_id": f"native:{image_id}:{row_index}",
                    "image_id": image_id,
                    "split": SPLIT_BY_IMAGE_ID[image_id],
                    "row_index": row_index,
                    "pred_row_id": str(ledger_row["pred_row_id"]),
                    "normalized_description": str(ledger_row["normalized_description"]),
                    "coord_token_ids": list(tokens),
                    "coord_token_ids_sha256": sha256_json(list(tokens)),
                    "bbox_pixel_xyxy": [float(v) for v in ledger_row["bbox_xyxy"]],
                    "strict_match_status": str(ledger_row["strict_match_status"]),
                    "strict_match_gt_owner_id": ledger_row.get("strict_match_gt_owner_id"),
                    "raw_span_sha256": str(ledger_row["raw_span_sha256"]),
                    "joins_physical_candidate_id": joined,
                    "join_semantics": (
                        "joins_provenance_never_adds_rank_mass"
                        if joined
                        else "no_identical_bank_tuple"
                    ),
                    "excluded_from_core_ranks": True,
                }
            )
    return rows


def build_shard_manifest(
    query_groups: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, Any]],
    categories: Sequence[Mapping[str, Any]],
    owners_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Per-image shards with ``estimated_work_units`` for largest-first dispatch.

    Front-loading GPU capture means every shard is independently launchable;
    ordering only decides who starts first on a busy queue.
    """

    by_image: dict[str, dict[str, Any]] = {}
    for image_id in sorted(owners_by_image, key=int):
        by_image[image_id] = {
            "image_id": image_id,
            "split": SPLIT_BY_IMAGE_ID[image_id],
            "owner_count": len(owners_by_image[image_id]),
            "context_count": 0,
            "loop_tail_context_count": 0,
            "category_count": 0,
            "blocked_category_count": 0,
            "query_group_count": 0,
            "blocked_query_group_count": 0,
            "physical_candidate_score_rows": 0,
        }
    for context in contexts:
        entry = by_image[str(context["image_id"])]
        entry["context_count"] += 1
        entry["loop_tail_context_count"] += int(bool(context["loop_marking"]["loop_tail"]))
    for category in categories:
        entry = by_image[str(category["image_id"])]
        entry["category_count"] += 1
        entry["blocked_category_count"] += int(category["status"] != "admitted")
    for group in query_groups:
        entry = by_image[str(group["image_id"])]
        entry["query_group_count"] += 1
        if group["status"] != "admitted":
            entry["blocked_query_group_count"] += 1
            continue
        entry["physical_candidate_score_rows"] += int(group["candidate_count"])

    rows: list[dict[str, Any]] = []
    for entry in by_image.values():
        admitted_groups = entry["query_group_count"] - entry["blocked_query_group_count"]
        # Work units: one prefill per admitted query group, one proposal
        # prefill per context, one scored row per physical candidate, and one
        # free decode per admitted query group.
        estimated = (
            entry["physical_candidate_score_rows"]
            + admitted_groups * 2
            + entry["context_count"]
        )
        rows.append(
            {
                "schema_version": PLAN_SCHEMA_VERSION,
                "row_kind": "census_shard",
                "shard_id": f"shard-{entry['image_id']}",
                **entry,
                "admitted_query_group_count": admitted_groups,
                "proposal_prefill_count": entry["context_count"],
                "free_decode_count": admitted_groups,
                "estimated_work_units": estimated,
            }
        )
    rows.sort(key=lambda row: (-int(row["estimated_work_units"]), str(row["image_id"])))
    for position, row in enumerate(rows):
        row["dispatch_order"] = position
        row["dispatch_policy"] = "largest_estimated_work_units_first"
    return rows


# ---------------------------------------------------------------------------
# Plan assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CensusPlan:
    images: list[dict[str, Any]]
    owners: list[dict[str, Any]]
    categories: list[dict[str, Any]]
    contexts: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    query_groups: list[dict[str, Any]]
    native_sidecars: list[dict[str, Any]]
    shards: list[dict[str, Any]]
    capture_rules: dict[str, Any]
    receipt: dict[str, Any]

    def files(self) -> dict[str, bytes]:
        return {
            "image-registry.jsonl": _jsonl_bytes(self.images),
            "owner-registry.jsonl": _jsonl_bytes(self.owners),
            "category-registry.jsonl": _jsonl_bytes(self.categories),
            "context-registry.jsonl": _jsonl_bytes(self.contexts),
            "candidate-bank.jsonl": _jsonl_bytes(self.candidates),
            "query-group-registry.jsonl": _jsonl_bytes(self.query_groups),
            "native-sidecar-registry.jsonl": _jsonl_bytes(self.native_sidecars),
            "shard-manifest.jsonl": _jsonl_bytes(self.shards),
            CAPTURE_RULES_NAME: canonical_json_bytes(self.capture_rules) + b"\n",
        }


CAPTURE_RULES_NAME = "capture-rules.json"
PLAN_JSONL_NAMES: tuple[str, ...] = (
    "image-registry.jsonl",
    "owner-registry.jsonl",
    "category-registry.jsonl",
    "context-registry.jsonl",
    "candidate-bank.jsonl",
    "query-group-registry.jsonl",
    "native-sidecar-registry.jsonl",
    "shard-manifest.jsonl",
)
#: Every declared plan output, digested in the receipt.  Consumers should
#: verify all of these before reading any row.
PLAN_FILE_NAMES: tuple[str, ...] = (*PLAN_JSONL_NAMES, CAPTURE_RULES_NAME)


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def build_plan(
    sources: SourcePaths = SourcePaths(),
    *,
    tokenizer_path: str | None = None,
    capture_rules: Mapping[str, Any] | None = None,
) -> CensusPlan:
    source_digests = {
        "panel": _assert_digest(sources.panel, REGISTERED_DIGESTS["panel"], "frozen panel"),
        "owner_ledger": _assert_digest(
            sources.owner_ledger, REGISTERED_DIGESTS["owner_ledger"], "canonical owner ledger"
        ),
        "prediction_ledger": _assert_digest(
            sources.prediction_ledger,
            REGISTERED_DIGESTS["prediction_ledger"],
            "canonical prediction-row ledger",
        ),
        "greedy_rollout": sha256_file(sources.greedy),
    }

    panel = load_panel(sources.panel)
    owners = load_owners(sources.owner_ledger, panel)
    rollouts = load_native_rollouts(sources.greedy)
    row_ledger = load_native_row_ledger(sources.prediction_ledger)

    owners_by_image: dict[str, list[dict[str, Any]]] = {}
    for owner in owners:
        owners_by_image.setdefault(owner["image_id"], []).append(owner)

    # Mark native true positives.  Contract item 1: true positives are
    # calibration, so they stay in the census and are never excluded.
    matched_owner_ids: dict[str, list[str]] = {}
    for (image_id, row_index), ledger_row in row_ledger.items():
        owner_id = ledger_row.get("strict_match_gt_owner_id")
        if ledger_row.get("strict_match_status") == "matched" and owner_id:
            matched_owner_ids.setdefault(str(owner_id), []).append(
                str(ledger_row["pred_row_id"])
            )

    observed_tokens = build_observed_category_tokens(rollouts)
    fallback = (
        build_tokenizer_resolver(tokenizer_path, observed_tokens) if tokenizer_path else None
    )

    images = build_image_registry(panel, rollouts)
    categories = build_category_registry(owners_by_image, observed_tokens, fallback)
    contexts = build_context_registry(rollouts, row_ledger, owners_by_image, panel)

    candidates, bank_accounting = build_candidate_bank(owners_by_image, panel)
    bank_by_category: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for candidate in candidates:
        bank_by_category.setdefault(
            (str(candidate["image_id"]), str(candidate["normalized_description"])), []
        ).append(candidate)

    owner_rows: list[dict[str, Any]] = []
    for owner in owners:
        native_rows = sorted(matched_owner_ids.get(owner["gt_owner_id"], []))
        owner_rows.append(
            {
                "schema_version": PLAN_SCHEMA_VERSION,
                "row_kind": "census_owner",
                **owner,
                "native_strict_match_pred_row_ids": native_rows,
                "native_true_positive": bool(native_rows),
                "calibration_role": (
                    "native_true_positive_calibration" if native_rows else "native_false_negative"
                ),
                "excluded_from_census": False,
                "candidate_bank": bank_accounting[owner["gt_owner_id"]],
                "disposition_eligibility": owner_disposition_eligibility(
                    greedy_eligible=bool(owner["greedy_eligible"]),
                    greedy_eligibility_status=str(owner["greedy_eligibility_status"]),
                    bank_coverage_status=str(
                        bank_accounting[owner["gt_owner_id"]]["bank_coverage_status"]
                    ),
                    native_true_positive=bool(native_rows),
                ),
            }
        )

    bank_token_index: dict[tuple[str, str], dict[tuple[int, ...], str]] = {}
    for candidate in candidates:
        bank_token_index.setdefault(
            (str(candidate["image_id"]), str(candidate["normalized_description"])), {}
        )[tuple(candidate["coord_token_ids"])] = str(candidate["candidate_id"])

    query_groups = build_query_group_registry(
        contexts, categories, rollouts, bank_by_category
    )
    native_sidecars = build_native_sidecar_registry(rollouts, row_ledger, bank_token_index)
    shards = build_shard_manifest(query_groups, contexts, categories, owners_by_image)

    logical_rows = sum(
        LOGICAL_ROLE_COUNT
        * sum(
            1
            for owner in owners_by_image[str(group["image_id"])]
            if owner["normalized_description"] == group["normalized_description"]
        )
        for group in query_groups
    )
    physical_rows = sum(int(group["candidate_count"]) for group in query_groups)
    rules = dict(capture_rules) if capture_rules is not None else build_capture_rules()

    receipt: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "plan_strategy": "twelve_image_owner_accessibility_census",
        "source_paths": {
            "panel": str(sources.panel),
            "owner_ledger": str(sources.owner_ledger),
            "prediction_ledger": str(sources.prediction_ledger),
            "greedy_rollout": str(sources.greedy),
        },
        "source_digests": source_digests,
        "score_input_policy": {
            # Contract item 5: the planner and scorer consume no score
            # artifact at all, so every pre-P0 score is structurally
            # unreachable rather than merely unused.
            "reads_any_score_artifact": False,
            "pre_p0_scores": "quarantined_never_read",
            "candidate_selection_uses_scores": False,
        },
        "census_shape": {
            "owner_count": len(owner_rows),
            "greedy_eligible_owner_count": sum(1 for row in owner_rows if row["greedy_eligible"]),
            "native_true_positive_owner_count": sum(
                1 for row in owner_rows if row["native_true_positive"]
            ),
            "native_false_negative_owner_count": sum(
                1 for row in owner_rows if not row["native_true_positive"]
            ),
            # The conclusion domain: owners that may close as persistent
            # negatives and that form the false-negative denominator.  It is
            # *not* the census retention count (346) and *not* the bank-adequacy
            # count: it additionally excludes native true positives (the
            # calibration population that defines q10) and the three
            # globally-ambiguity-neutral owners.
            "persistent_negative_eligible_owner_count": sum(
                1
                for row in owner_rows
                if row["disposition_eligibility"]["persistent_negative_eligible"]
            ),
            "persistent_negative_eligible_semantics": (
                "greedy_eligible AND native_false_negative AND adequate bank; "
                "equals the false-negative prevalence and confirmation denominator"
            ),
            "bank_disposition_eligible_owner_count": sum(
                1 for row in owner_rows if row["candidate_bank"]["disposition_eligible"]
            ),
            "image_count": len(images),
            "context_count": len(contexts),
            "loop_tail_context_count": sum(
                1 for row in contexts if row["loop_marking"]["loop_tail"]
            ),
            "native_row_count": len(native_sidecars),
            "category_count": len(categories),
            "blocked_category_count": sum(1 for row in categories if row["status"] != "admitted"),
            "query_group_count": len(query_groups),
            "admitted_query_group_count": sum(
                1 for row in query_groups if row["status"] == "admitted"
            ),
            "physical_candidate_count": len(candidates),
            "logical_candidate_score_rows": logical_rows,
            "physical_candidate_score_rows": physical_rows,
        },
        "split": {
            "discovery_image_ids": list(DISCOVERY_IMAGE_IDS),
            "confirmation_image_ids": list(CONFIRMATION_IMAGE_IDS),
            "discovery_image_count": len(DISCOVERY_IMAGE_IDS),
            "confirmation_image_count": len(CONFIRMATION_IMAGE_IDS),
            "tuning_policy": "confirmation_rules_are_frozen_on_discovery_only",
        },
        "candidate_bank_contract": {
            "logical_roles": list(LOGICAL_ROLES),
            "core_roles": list(CORE_ROLES),
            "extension_roles": list(EXTENSION_ROLES),
            "logical_role_count": LOGICAL_ROLE_COUNT,
            "physical_identity": "digest(image_id, normalized_description, coord_token_ids)",
            "collapse_scope": "image_and_normalized_description",
            "cross_owner_collapse": True,
            "assignment_scope": "same_normalized_description_only",
            "substitution_policy": "none_never_score_selected",
            "mid_run_growth": "forbidden",
            "cross_owner_shared_candidate_count": sum(
                1 for row in candidates if row["cross_owner_generated"]
            ),
        },
        "capture_rules_sha256": rules["capture_rules_sha256"],
        "query_suffix_contract": {
            "shape": ["object_ref_start", "category_token_ids", "object_ref_end", "box_start"],
            "wrapper_token_ids": dict(WRAPPER_TOKEN_IDS),
            "x1_distribution_read_point": "immediately_after_box_start",
            "on_mismatch": "fail_closed",
        },
        "loop_marking_contract": {
            "continuous_fields": [
                "prior_identical_row_count",
                "consecutive_identical_row_run_length",
                "repeated_raw_span_sha256",
            ],
            "flag_rule": f"consecutive_identical_row_run_length >= {LOOP_TAIL_MIN_CONSECUTIVE_RUN}",
            "flag_is_not_a_mechanism_label": True,
        },
        "estimand": {
            "name": "category_field_support_at_owner_geometry",
            "is_per_owner_proposal_probability": False,
            "proposal_and_localization_separate": True,
        },
        "shard_dispatch": [
            {
                "image_id": row["image_id"],
                "dispatch_order": row["dispatch_order"],
                "estimated_work_units": row["estimated_work_units"],
            }
            for row in shards
        ],
        "output_file_digests": {},
    }

    plan = CensusPlan(
        images=images,
        owners=owner_rows,
        categories=categories,
        contexts=contexts,
        candidates=candidates,
        query_groups=query_groups,
        native_sidecars=native_sidecars,
        shards=shards,
        capture_rules=rules,
        receipt=receipt,
    )
    files = plan.files()
    receipt["output_file_digests"] = {
        name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())
    }
    receipt["receipt_content_sha256"] = sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    return plan


def commit_plan(plan: CensusPlan, output_dir: Path) -> dict[str, str]:
    """Create-or-identical commit: an existing plan may only be reused byte-for-byte."""

    output_dir.mkdir(parents=True, exist_ok=True)
    files = dict(plan.files())
    files["receipt.json"] = canonical_json_bytes(plan.receipt) + b"\n"
    written: dict[str, str] = {}
    for name, content in sorted(files.items()):
        path = output_dir / name
        if path.exists():
            existing = path.read_bytes()
            if existing != content:
                raise PlanContractError(
                    f"refusing to overwrite existing plan file {name!r} with different bytes"
                )
        else:
            path.write_bytes(content)
        written[name] = hashlib.sha256(content).hexdigest()
    return written


def summarize(plan: CensusPlan) -> str:
    shape = plan.receipt["census_shape"]
    lines = [
        f"unit: {UNIT_ID}",
        f"schema: {PLAN_SCHEMA_VERSION}",
        "",
        f"owners censused (retained)  {shape['owner_count']}",
        f"  greedy-eligible           {shape['greedy_eligible_owner_count']}",
        f"  native true positives     {shape['native_true_positive_owner_count']}"
        f"  (calibration; cannot close negative)",
        f"  native false negatives    {shape['native_false_negative_owner_count']}",
        "",
        "CONCLUSION DOMAIN (persistent-negative eligible = FN denominator)",
        f"  persistent-negative eligible owners: "
        f"{shape['persistent_negative_eligible_owner_count']}"
        f"  = greedy-eligible AND native-FN AND adequate bank",
        f"images                      {shape['image_count']}"
        f" (discovery 6 / confirmation 6)",
        f"contexts                    {shape['context_count']}"
        f" (loop_tail {shape['loop_tail_context_count']})",
        f"native rows                 {shape['native_row_count']}",
        f"categories                  {shape['category_count']}"
        f" (blocked {shape['blocked_category_count']})",
        f"query groups                {shape['query_group_count']}"
        f" (admitted {shape['admitted_query_group_count']})",
        f"physical candidates         {shape['physical_candidate_count']}",
        f"logical candidate rows      {shape['logical_candidate_score_rows']}",
        f"physical candidate rows     {shape['physical_candidate_score_rows']}",
        "",
        "shard dispatch (largest estimated_work_units first):",
    ]
    for row in plan.shards:
        lines.append(
            f"  {row['dispatch_order']:>2}. image {row['image_id']:>6} "
            f"[{row['split']:<12}] owners={row['owner_count']:>3} "
            f"contexts={row['context_count']:>4} groups={row['admitted_query_group_count']:>5} "
            f"rows={row['physical_candidate_score_rows']:>6} "
            f"work_units={row['estimated_work_units']:>6}"
        )
    blocked = [row for row in plan.categories if row["status"] != "admitted"]
    if blocked:
        lines.append("")
        lines.append("blocked categories (need --tokenizer-path):")
        for row in blocked:
            lines.append(f"  image {row['image_id']:>6}  {row['normalized_description']}")
    cross_owner = sum(1 for row in plan.candidates if row["cross_owner_generated"])
    lines.append("")
    lines.append(
        f"cross-owner collapsed candidates: {cross_owner} of {len(plan.candidates)}"
    )
    by_status: dict[str, int] = {}
    for row in plan.owners:
        status = row["candidate_bank"]["bank_coverage_status"]
        by_status[status] = by_status.get(status, 0) + 1
    lines.append("bank adequacy (frozen before any score):")
    for status in BANK_COVERAGE_STATUSES:
        lines.append(f"  {status:<30} {by_status.get(status, 0):>4}")
    risky = [row for row in plan.owners if row["candidate_bank"]["undercovered"]]
    reduced = [
        row
        for row in plan.owners
        if row["candidate_bank"]["bank_coverage_status"] == "adequate_reduced"
    ]
    lines.append(
        f"owners passing the BANK-ADEQUACY floor only: "
        f"{sum(1 for row in plan.owners if row['candidate_bank']['disposition_eligible'])}"
        f" of {len(plan.owners)}  (bank adequacy alone; NOT the conclusion domain)"
    )
    lines.append(
        f"owners floored to unresolved by bank undercoverage: {len(risky)}"
    )
    lines.append(f"owners flagged adequate_reduced (still eligible): {len(reduced)}")
    for row in (risky + reduced)[:10]:
        bank = row["candidate_bank"]
        lines.append(
            f"  {row['gt_owner_id']:<16} reached={bank['distinct_physical_candidates_reached']:>2} "
            f"unique_assigned="
            f"{bank['strict_assignment_coverage']['uniquely_assigned_candidate_count']:>2} "
            f"status={bank['bank_coverage_status']}"
        )
    if len(risky) + len(reduced) > 10:
        lines.append(f"  ... and {len(risky) + len(reduced) - 10} more")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and summarize the plan without writing any artifact",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help=(
            "production tokenizer used only to resolve categories never emitted "
            "natively; validated against every observed span before use"
        ),
    )
    parser.add_argument("--panel", type=Path, default=PANEL_PATH)
    parser.add_argument("--owner-ledger", type=Path, default=OWNER_LEDGER_PATH)
    parser.add_argument("--prediction-ledger", type=Path, default=PREDICTION_LEDGER_PATH)
    parser.add_argument("--greedy", type=Path, default=GREEDY_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.dry_run and args.output_dir is None:
        raise SystemExit("--output-dir is required unless --dry-run is given")
    sources = SourcePaths(
        panel=args.panel,
        owner_ledger=args.owner_ledger,
        prediction_ledger=args.prediction_ledger,
        greedy=args.greedy,
    )
    try:
        plan = build_plan(sources, tokenizer_path=args.tokenizer_path)
    except PlanContractError as exc:
        print(f"plan contract error: {exc}", file=sys.stderr)
        return 2
    print(summarize(plan))
    if args.dry_run:
        print("\ndry run: no artifact written")
        return 0
    commit_plan(plan, args.output_dir)
    print(f"\nplan written to {args.output_dir}")
    print(f"receipt_content_sha256 {plan.receipt['receipt_content_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
