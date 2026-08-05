#!/usr/bin/env python3
"""Frozen-route analysis for the sorted crossing matched-length neutral-row
insertion control
(``2026-08-04-sorted-crossing-matched-length-neutral-row-insertion-control``).

Frozen unit::

    research/investigations/qwen3-vl-dense-enumeration/experiments/
    2026-08-04-sorted-crossing-matched-length-neutral-row-insertion-control/unit.md

The one question
----------------
At the frozen sorted crossing boundaries where inserting the exact clean row for
skipped owner ``C`` materially lowered the likelihood of the exact downstream row
``E`` coordinate tokens, does inserting the deterministic neutral control row
``N`` reproduce that materiality?

The estimand, per owner ``i``::

    neutral_coordinate_delta(i)
      = logprob(E coordinates | P+N) - logprob(E coordinates | P)

    relative_neutral_coordinate_delta(i)
      = neutral_coordinate_delta(i) - same_run_benign_coordinate_delta(image(i))

``N`` is material iff ``relative_neutral_coordinate_delta <= -1.0`` nat.  The
``-1.0`` cutoff is *imported* from the geometry unit that froze it, never
re-declared here.  ``same_run_benign_coordinate_delta`` is always the same-run
benign replay of that image; the sealed historical value is a gate reference
only and never enters an estimand.

Ordered gates
-------------
``unit.md`` "Validity gates" are evaluated in one fixed order.  Gates 1, 2 and 5
are owned by the merge and are re-read from its self-sealed receipt; gates 3, 4
and 6 are computed here:

3. every executed ``P+C -> E`` replay reproduces the frozen sign and materiality
   and agrees with the sealed raw coordinate delta within ``0.05`` nat; more than
   two failures stops the unit, and the sentinels ``gt:13923:14`` and
   ``gt:4134:27`` must pass regardless of the total;
4. all twelve benign pairs reproduce their sealed sign and materiality, with each
   selected-token sum within ``1e-3`` and preserving the compared argmax and each
   coordinate delta within ``0.05`` nat; any mismatch is visible and blocks
   image-referenced interpretation for the affected images;
6. if ``N`` is material for at least four of the frozen twelve specificity
   owners, the route is ``neutral_row_not_neutral_inconclusive``.

Route arithmetic is absolute over the frozen nine voting owners and their four
clearly separated members.  A quarantined or benign-blocked owner is neither
material nor nonmaterial and therefore conservatively satisfies no route; the
receipt makes that arithmetic explicit.

Separation this module preserves
--------------------------------
Exact downstream-string likelihood is not owner emission, strict matching, or
final-set coverage.  Strict-matcher-unmatched ``E`` rows stay unknown-neutral in
every stratum, and a drop in an imperfect exact ``E`` coordinate string is never
reported as a lost owner or a reduced coverage.

Outputs::

    neutral-row-owner-rows.jsonl
    neutral-row-summary.json
    neutral-row-report.md
    neutral-row-analysis-receipt.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import analyze_sorted_crossing_owner_row_geometry as geometry  # noqa: E402
from scripts.research import merge_sorted_crossing_neutral_row_control as merger  # noqa: E402
from scripts.research import prepare_sorted_crossing_neutral_row_control as plan_builder  # noqa: E402
from scripts.research import (  # noqa: E402
    score_sorted_crossing_boundary_owner_release as crossing_scorer,
)
from scripts.research import score_sorted_crossing_neutral_row_control as scorer  # noqa: E402

UNIT_ID = plan_builder.UNIT_ID
OWNER_ROW_SCHEMA_VERSION = "sorted-crossing-neutral-row-control-owner-row.v1"
SUMMARY_SCHEMA_VERSION = "sorted-crossing-neutral-row-control-summary.v1"
RECEIPT_SCHEMA_VERSION = "sorted-crossing-neutral-row-control-analysis-receipt.v1"

OWNER_ROWS_NAME = "neutral-row-owner-rows.jsonl"
SUMMARY_NAME = "neutral-row-summary.json"
REPORT_MD_NAME = "neutral-row-report.md"
RECEIPT_NAME = "neutral-row-analysis-receipt.json"

#: The frozen image-referenced materiality cutoff, imported from the unit that
#: froze it.  It is never re-declared, re-fitted or widened here.
MATERIALITY_MAX_NATS = geometry.MATERIAL_NEGATIVE_MAX_NATS

#: unit.md gate tolerances and stop rules.
REPLAY_MAX_SELECTED_LOGIT_ABS_DIFF = plan_builder.REPLAY_MAX_SELECTED_LOGIT_ABS_DIFF
REPLAY_MAX_COORDINATE_DELTA_ABS_DIFF = plan_builder.REPLAY_MAX_COORDINATE_DELTA_ABS_DIFF
MAX_CLEAN_REPLAY_FAILURES = plan_builder.MAX_CLEAN_REPLAY_FAILURES
SENTINEL_OWNER_IDS = plan_builder.SENTINEL_OWNER_IDS
SPECIFICITY_MATERIAL_MAX = plan_builder.SPECIFICITY_MATERIAL_MAX

GATE_INPUT_AND_TOKEN_IDENTITY = merger.GATE_INPUT_AND_TOKEN_IDENTITY
GATE_RUNTIME_REPLAY = merger.GATE_RUNTIME_REPLAY
GATE_POSITIVE_CONTROLS = "same_run_positive_controls"
GATE_BENIGN_REPLAY = "benign_reference_replay"
GATE_CACHE_PARITY = merger.GATE_CACHE_PARITY
GATE_SPECIFICITY = "neutral_row_specificity"
#: unit.md "Validity gates": this exact order, always.
GATE_ORDER: tuple[str, ...] = (
    GATE_INPUT_AND_TOKEN_IDENTITY,
    GATE_RUNTIME_REPLAY,
    GATE_POSITIVE_CONTROLS,
    GATE_BENIGN_REPLAY,
    GATE_CACHE_PARITY,
    GATE_SPECIFICITY,
)

ROUTE_C_CONTENT_SPECIFIC = plan_builder.ROUTE_C_CONTENT_SPECIFIC
ROUTE_GENERIC_SUFFICIENT = plan_builder.ROUTE_GENERIC_SUFFICIENT
ROUTE_NEUTRAL_NOT_NEUTRAL = plan_builder.ROUTE_NEUTRAL_NOT_NEUTRAL
ROUTE_INCONCLUSIVE = plan_builder.ROUTE_INCONCLUSIVE

ROUTE_READINGS: Mapping[str, str] = {
    ROUTE_C_CONTENT_SPECIFIC: (
        "Mere row insertion, even using a duplicate and sort-inconsistent real row, does not "
        "reproduce the tail. A C-content-specific local interaction survives at these frozen "
        "boundaries. This does not identify whether the relevant content is owner identity, "
        "description, novelty, sorted-route consistency, or another joined property."
    ),
    ROUTE_GENERIC_SUFFICIENT: (
        "Generic boundary sensitivity is sufficient; C-specific competition is not required at "
        "these frozen boundaries. Owner-specific competition was never refuted: duplicate and "
        "sorted-order regression can inflate N materiality."
    ),
    ROUTE_NEUTRAL_NOT_NEUTRAL: (
        "The neutral row is not neutral enough to interpret: it is material on at least four of "
        "the frozen twelve specificity owners, so material N is not read as generic-insertion "
        "evidence for the voting owners."
    ),
    ROUTE_INCONCLUSIVE: (
        "No frozen route is satisfied. This line stops inconclusive; there is no adaptive "
        "threshold, second neutral-row choice, or automatic GPU successor."
    ),
}

#: unit.md "Non-voting sensitivities": descriptive recomputations only.
SENSITIVITY_EXCLUDE_LENGTH_DELTA_TWO = "exclude_row_length_delta_two"
SENSITIVITY_EXCLUDE_POSTHOC_UNCERTAIN = "exclude_posthoc_support_extent_uncertain"
SENSITIVITY_CUTOFF_PREFIX = "materiality_cutoff"
SENSITIVITY_CUTOFFS_NATS = plan_builder.SENSITIVITY_CUTOFFS_NATS

STRATUM_AXES: tuple[str, ...] = (
    "e_strict_match",
    "description_relation",
    "geometric_separation",
    "visual_adjudication",
)

CLAIM_BOUNDARY = plan_builder.CLAIM_BOUNDARY
NOT_CLAIMED: tuple[str, ...] = (
    "no causal-mechanism claim beyond a local prefix-sensitivity contrast",
    "no final-set coverage, eventual-recovery or natural-stop claim",
    "no free-rollout, emission or owner-matching claim",
    "no population-prevalence claim beyond the frozen twelve-image boundary cohort",
    "no training, architecture, inference-policy or production claim",
)
LIKELIHOOD_VERSUS_COVERAGE = (
    "exact teacher-forced downstream-string likelihood is a different evidence level from owner "
    "emission, strict matching and final-set coverage; a drop in an imperfect exact E coordinate "
    "string is not evidence that the model hallucinated, lost a real owner, or reduced coverage"
)
UNKNOWN_NEUTRAL_NOTE = (
    "strict-matcher-unmatched E rows remain unknown-neutral in every stratum and are never read "
    "as a displacement, a hallucination or a coverage loss"
)


class NeutralRowAnalysisContractError(RuntimeError):
    """A precondition of the frozen neutral-row analysis was not proven."""


def _fail(message: str) -> NoReturn:
    raise NeutralRowAnalysisContractError(message)


canonical_json_bytes = geometry.canonical_json_bytes
sha256_bytes = geometry.sha256_bytes
sha256_json = geometry.sha256_json
sha256_file = geometry.sha256_file


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return geometry.read_json(Path(path), label)
    except geometry.GeometryContractError as exc:
        _fail(str(exc))


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        return geometry.read_jsonl(Path(path), label)
    except geometry.GeometryContractError as exc:
        _fail(str(exc))


def _assert_self_sealed(payload: Mapping[str, Any], *, digest_key: str, label: str) -> str:
    try:
        return geometry.assert_self_sealed(payload, digest_key=digest_key, label=label)
    except geometry.GeometryContractError as exc:
        _fail(str(exc))


def _assert_file_digest(path: Path, *, expected: str, label: str) -> str:
    try:
        return geometry.assert_file_digest(Path(path), expected=expected, label=label)
    except geometry.GeometryContractError as exc:
        _fail(str(exc))


def _finite(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        _fail(f"{label} is not a number ({value!r})")
    if not math.isfinite(number):
        _fail(f"{label} is not finite ({value!r})")
    return number


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def is_material(relative_delta: float, *, cutoff: float = MATERIALITY_MAX_NATS) -> bool:
    """The frozen image-referenced materiality rule, at one explicit cutoff."""

    return relative_delta <= cutoff


# ---------------------------------------------------------------------------
# 1. Sealed inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergedEvidence:
    """The immutable merged capture, bound to the sealed plan it was produced for."""

    merged_dir: Path
    receipt: dict[str, Any]
    receipt_content_sha256: str
    rows: list[dict[str, Any]]
    plan: merger.FrozenPlan
    file_sha256: dict[str, str]


def load_merged_evidence(merged_dir: Path, *, plan_dir: Path | None = None) -> MergedEvidence:
    """Re-prove the merge receipt, its rows and its plan binding before use."""

    merged_dir = Path(merged_dir)
    receipt = _read_json(merged_dir / merger.MERGE_RECEIPT_NAME, "neutral-row merge receipt")
    if str(receipt.get("schema_version")) != merger.MERGE_SCHEMA_VERSION:
        _fail(
            f"merge receipt schema {receipt.get('schema_version')!r} is not "
            f"{merger.MERGE_SCHEMA_VERSION!r}"
        )
    if str(receipt.get("unit_id")) != UNIT_ID:
        _fail("merge receipt belongs to another unit")
    receipt_seal = _assert_self_sealed(
        receipt, digest_key="receipt_content_sha256", label="merge receipt"
    )
    digests = receipt.get("output_file_digests")
    if not isinstance(digests, Mapping):
        _fail("merge receipt carries no output_file_digests")

    file_sha256: dict[str, str] = {
        str(merged_dir / merger.MERGE_RECEIPT_NAME): sha256_file(
            merged_dir / merger.MERGE_RECEIPT_NAME
        )
    }
    for name in (merger.MERGED_ROWS_NAME, merger.MERGED_PARITY_NAME):
        entry = digests.get(name)
        if not isinstance(entry, Mapping):
            _fail(f"merge receipt declares no digest for {name}")
        file_sha256[str(merged_dir / name)] = _assert_file_digest(
            merged_dir / name, expected=str(entry.get("sha256")), label=f"merged {name}"
        )
    rows = _read_jsonl(merged_dir / merger.MERGED_ROWS_NAME, "merged rows")

    sealed_plan = receipt.get("plan")
    if not isinstance(sealed_plan, Mapping):
        _fail("merge receipt seals no plan block")
    resolved_plan_dir = Path(
        plan_dir if plan_dir is not None else str(sealed_plan.get("plan_dir", ""))
    )
    try:
        plan = merger.load_frozen_plan(resolved_plan_dir)
    except crossing_scorer.CrossingBoundaryContractError as exc:
        _fail(f"sealed plan: {exc}")
    if plan.manifest_content_sha256 != str(sealed_plan.get("manifest_content_sha256")):
        _fail(
            "the plan on disk is not the one the merge was produced against; refusing to route "
            "over a cohort the evidence never saw"
        )
    file_sha256[str(plan.plan_dir / plan_builder.MANIFEST_NAME)] = sha256_file(
        plan.plan_dir / plan_builder.MANIFEST_NAME
    )

    gates = receipt.get("gates")
    if not isinstance(gates, Mapping):
        _fail("merge receipt seals no gate block")
    for gate in merger.MERGE_OWNED_GATES:
        block = gates.get(gate)
        if not isinstance(block, Mapping) or block.get("passed") is not True:
            _fail(f"the merge receipt does not declare gate {gate!r} passed")
    if len(rows) != scorer.TOTAL_REQUEST_COUNT:
        _fail(
            f"the merged evidence holds {len(rows)} rows, not the frozen "
            f"{scorer.TOTAL_REQUEST_COUNT}"
        )
    return MergedEvidence(
        merged_dir=merged_dir,
        receipt=receipt,
        receipt_content_sha256=receipt_seal,
        rows=rows,
        plan=plan,
        file_sha256=dict(sorted(file_sha256.items())),
    )


def _coordinate_delta(row: Mapping[str, Any], *, label: str) -> float:
    deltas = row.get("deltas")
    if not isinstance(deltas, Mapping):
        _fail(f"{label} carries no deltas block")
    block = deltas.get(scorer.SEGMENT_COORDINATES)
    if not isinstance(block, Mapping):
        _fail(f"{label} carries no coordinate delta")
    return _finite(block.get("delta"), label=f"{label} coordinate delta")


def _baseline_coordinate_sum(row: Mapping[str, Any], *, label: str) -> float:
    roots = row.get("roots")
    if not isinstance(roots, Mapping):
        _fail(f"{label} carries no roots block")
    baseline = roots.get(scorer.ROOT_BASELINE)
    if not isinstance(baseline, Mapping):
        _fail(f"{label} carries no baseline root")
    return _finite(
        baseline["segment_sums"][scorer.SEGMENT_COORDINATES]["sum"],
        label=f"{label} baseline coordinate sum",
    )


def _baseline_argmax_digest(row: Mapping[str, Any], *, label: str) -> str:
    roots = row["roots"][scorer.ROOT_BASELINE]
    return sha256_json(
        crossing_scorer._token_ids(  # noqa: SLF001
            roots.get("argmax_token_ids"), label=f"{label} baseline argmax"
        )
    )


def index_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["gt_owner_id"]), str(row["arm"]))
        if key in index:
            _fail(f"the merged evidence carries duplicate readout {key!r}")
        index[key] = dict(row)
    return index


# ---------------------------------------------------------------------------
# 2. Gate 4: the same-run benign reference
# ---------------------------------------------------------------------------


def evaluate_benign_gate(
    evidence: MergedEvidence, by_key: Mapping[tuple[str, str], Mapping[str, Any]]
) -> dict[str, Any]:
    """``unit.md`` gate 4, and the same-run reference every estimand subtracts.

    A mismatch does not stop the unit: it is visible and blocks image-referenced
    interpretation for the affected images, whose owners then satisfy no route.
    """

    per_image: dict[str, dict[str, Any]] = {}
    for gt_owner_id, control in sorted(evidence.plan.benign_by_owner.items()):
        image_id = str(control["image_id"])
        row = by_key.get((gt_owner_id, scorer.ARM_BENIGN))
        if row is None:
            _fail(f"benign control {gt_owner_id!r} has no merged readout")
        label = f"benign control {gt_owner_id!r}"
        sealed = dict(control["sealed_benign_reference"])
        observed_delta = _coordinate_delta(row, label=label)
        observed_sum = _baseline_coordinate_sum(row, label=label)
        sealed_delta = _finite(sealed["coordinate_delta"], label=f"{label} sealed delta")
        sealed_sum = _finite(
            sealed["baseline_coordinate_sum"], label=f"{label} sealed baseline sum"
        )
        sum_abs_diff = abs(observed_sum - sealed_sum)
        delta_abs_diff = abs(observed_delta - sealed_delta)
        argmax_preserved = _baseline_argmax_digest(row, label=label) == str(
            sealed["baseline_argmax_token_ids_sha256"]
        )
        sign_reproduced = _sign(observed_delta) == int(sealed["coordinate_delta_sign"])
        material_reproduced = is_material(observed_delta) is bool(sealed["material_negative"])
        passed = (
            sum_abs_diff <= REPLAY_MAX_SELECTED_LOGIT_ABS_DIFF
            and argmax_preserved
            and delta_abs_diff <= REPLAY_MAX_COORDINATE_DELTA_ABS_DIFF
            and sign_reproduced
            and material_reproduced
        )
        if image_id in per_image:
            _fail(f"image {image_id!r} carries more than one benign reference")
        per_image[image_id] = {
            "gt_owner_id": gt_owner_id,
            "image_id": image_id,
            "same_run_benign_coordinate_delta": observed_delta,
            "sealed_benign_coordinate_delta": sealed_delta,
            "coordinate_delta_abs_diff": delta_abs_diff,
            "coordinate_delta_tolerance": REPLAY_MAX_COORDINATE_DELTA_ABS_DIFF,
            "observed_baseline_coordinate_sum": observed_sum,
            "sealed_baseline_coordinate_sum": sealed_sum,
            "selected_sum_abs_diff": sum_abs_diff,
            "selected_sum_tolerance": REPLAY_MAX_SELECTED_LOGIT_ABS_DIFF,
            "compared_argmax_preserved": argmax_preserved,
            "sign_reproduced": sign_reproduced,
            "materiality_reproduced": material_reproduced,
            "passed": passed,
            "blocks_image_referenced_interpretation": not passed,
        }
    blocked = sorted(image_id for image_id, entry in per_image.items() if not entry["passed"])
    return {
        "gate": GATE_BENIGN_REPLAY,
        "passed": not blocked,
        "checked_image_count": len(per_image),
        "blocked_image_ids": blocked,
        "estimand_source": "same_run_benign_replay_only",
        "sealed_value_role": "gate_reference_only_never_the_estimand",
        "per_image": per_image,
    }


# ---------------------------------------------------------------------------
# 3. Owner rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OwnerReadout:
    """One executed crossing owner's ``N`` and ``C`` readouts, image-referenced."""

    gt_owner_id: str
    image_id: str
    cohort_role: str
    votes: bool
    clearly_separated: bool
    row: dict[str, Any]

    @property
    def interpretable(self) -> bool:
        return bool(self.row["interpretable"])

    @property
    def neutral_material(self) -> bool:
        return bool(self.row["neutral"]["material"])


def build_owner_rows(
    evidence: MergedEvidence,
    by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    benign: Mapping[str, Any],
    replay_gate: Mapping[str, Any],
) -> list[OwnerReadout]:
    """One owner-level row per executed crossing owner, never pooled across images."""

    quarantined = set(replay_gate.get("quarantined_owner_ids") or ())
    blocked_images = set(benign["blocked_image_ids"])
    readouts: list[OwnerReadout] = []
    for gt_owner_id in evidence.plan.executed_owner_ids:
        selection = evidence.plan.selection_by_owner[gt_owner_id]
        image_id = str(selection["image_id"])
        reference = benign["per_image"].get(image_id)
        if reference is None:
            _fail(f"owner {gt_owner_id!r} has no same-run benign reference on image {image_id!r}")
        benign_delta = _finite(
            reference["same_run_benign_coordinate_delta"], label=f"{image_id} benign delta"
        )

        arms: dict[str, dict[str, Any]] = {}
        for arm in scorer.PAIRED_CROSSING_ARMS:
            row = by_key.get((gt_owner_id, arm))
            if row is None:
                _fail(f"owner {gt_owner_id!r} has no merged {arm!r} readout")
            label = f"owner {gt_owner_id!r} {arm!r}"
            raw = _coordinate_delta(row, label=label)
            arms[arm] = {
                "request_id": str(row["request_id"]),
                "scored_token_ids_sha256": str(row["scored_token_ids_sha256"]),
                "coordinate_delta": raw,
                "coordinate_delta_sign": _sign(raw),
                "relative_coordinate_delta": raw - benign_delta,
                "complete_row_delta": _finite(
                    row["deltas"]["complete_row"]["delta"], label=f"{label} complete row"
                ),
                "description_delta": _finite(
                    row["deltas"]["description"]["delta"], label=f"{label} description"
                ),
                "baseline_coordinate_sum": _baseline_coordinate_sum(row, label=label),
            }
        digests = {arms[arm]["scored_token_ids_sha256"] for arm in scorer.PAIRED_CROSSING_ARMS}
        if len(digests) != 1:
            _fail(
                f"owner {gt_owner_id!r} scores different E tokens across arms; the contrast is "
                "meaningless"
            )

        sealed = dict(selection["sealed_clean_reference"])
        neutral = arms[scorer.ARM_NEUTRAL]
        clean = arms[scorer.ARM_CLEAN_REPLAY]
        interpretable = gt_owner_id not in quarantined and image_id not in blocked_images
        e_row = selection["scored_e_row"]
        neutral_row = selection["neutral_row_n"]
        row_payload = {
            "schema_version": OWNER_ROW_SCHEMA_VERSION,
            "row_kind": "neutral_row_control_owner_row",
            "unit_id": UNIT_ID,
            "gt_owner_id": gt_owner_id,
            "image_id": image_id,
            "cohort_role": str(selection["cohort_role"]),
            "votes": bool(selection["votes"]),
            "clearly_separated": bool(selection["clearly_separated"]),
            "sentinel_owner": bool(selection["sentinel_owner"]),
            "posthoc_support_extent_uncertain": bool(
                selection["posthoc_support_extent_uncertain"]
            ),
            "interpretable": interpretable,
            "quarantined": gt_owner_id in quarantined,
            "benign_blocked": image_id in blocked_images,
            "same_run_benign_reference": {
                "gt_owner_id": str(reference["gt_owner_id"]),
                "image_id": image_id,
                "same_run_benign_coordinate_delta": benign_delta,
                "role": "the only reference either relative estimand subtracts",
            },
            "neutral": {
                **neutral,
                # Materiality is a property of the relative delta alone.
                # Interpretability is a separate, later question: gate 3 runs
                # before gate 4 in unit.md's fixed order, so a benign-blocked or
                # quarantined owner still has a well-defined materiality and is
                # only withheld from the route arithmetic.
                "material": is_material(neutral["relative_coordinate_delta"]),
                "counts_toward_route": interpretable,
                "materiality_cutoff_nats": MATERIALITY_MAX_NATS,
                "neutral_row_gt_owner_id": str(neutral_row["gt_owner_id"]),
                "row_length_delta_tokens": int(neutral_row["row_length_delta_tokens"]),
                "rows_back_distance": int(neutral_row["rows_back_distance"]),
                "min_center_distance_normalized": _finite(
                    neutral_row["min_center_distance_normalized"],
                    label=f"{gt_owner_id} min center distance",
                ),
                "confounds": [
                    "duplicate_of_an_already_emitted_row",
                    "sorted_route_regression",
                ],
            },
            "clean": {
                **clean,
                "material": is_material(clean["relative_coordinate_delta"]),
                "counts_toward_route": interpretable,
                "sealed_coordinate_delta": _finite(
                    sealed["coordinate_delta"], label=f"{gt_owner_id} sealed C delta"
                ),
                "sealed_coordinate_delta_sign": int(sealed["coordinate_delta_sign"]),
                "sealed_material_negative": bool(sealed["material_negative"]),
                "sealed_relative_coordinate_delta": _finite(
                    sealed["relative_coordinate_delta"], label=f"{gt_owner_id} sealed relative"
                ),
                "sealed_role": "gate_reference_only_never_the_estimand",
            },
            "strata": {
                "e_strict_match": str(e_row["strict_match_status"]),
                "e_stratum": (
                    "matched_e"
                    if str(e_row["strict_match_status"]) == "matched"
                    else "unmatched_e"
                ),
                "description_relation": (
                    "same_description"
                    if str(e_row["normalized_description"])
                    == str(selection["normalized_description"])
                    else "different_description"
                ),
                "geometric_separation": (
                    "clearly_separated"
                    if bool(selection["clearly_separated"])
                    else "not_clearly_separated"
                ),
                "visual_adjudication": (
                    "plausible_support_extent_uncertain"
                    if bool(selection["posthoc_support_extent_uncertain"])
                    else "not_flagged"
                ),
                "unknown_neutral_note": UNKNOWN_NEUTRAL_NOTE,
            },
            "likelihood_versus_coverage": LIKELIHOOD_VERSUS_COVERAGE,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        readouts.append(
            OwnerReadout(
                gt_owner_id=gt_owner_id,
                image_id=image_id,
                cohort_role=str(selection["cohort_role"]),
                votes=bool(selection["votes"]),
                clearly_separated=bool(selection["clearly_separated"]),
                row=row_payload,
            )
        )
    return readouts


# ---------------------------------------------------------------------------
# 4. Gate 3: the same-run positive controls
# ---------------------------------------------------------------------------


def evaluate_positive_control_gate(readouts: Sequence[OwnerReadout]) -> dict[str, Any]:
    """``unit.md`` gate 3, over every executed ``P+C -> E`` replay."""

    entries: list[dict[str, Any]] = []
    for readout in readouts:
        clean = readout.row["clean"]
        raw_abs_diff = abs(
            float(clean["coordinate_delta"]) - float(clean["sealed_coordinate_delta"])
        )
        sign_reproduced = int(clean["coordinate_delta_sign"]) == int(
            clean["sealed_coordinate_delta_sign"]
        )
        materiality_reproduced = bool(clean["material"]) is bool(
            clean["sealed_material_negative"]
        )
        passed = (
            raw_abs_diff <= REPLAY_MAX_COORDINATE_DELTA_ABS_DIFF
            and sign_reproduced
            and materiality_reproduced
        )
        entries.append(
            {
                "gt_owner_id": readout.gt_owner_id,
                "image_id": readout.image_id,
                "cohort_role": readout.cohort_role,
                "sentinel_owner": bool(readout.row["sentinel_owner"]),
                "interpretable": readout.interpretable,
                "coordinate_delta_abs_diff": raw_abs_diff,
                "coordinate_delta_tolerance": REPLAY_MAX_COORDINATE_DELTA_ABS_DIFF,
                "sign_reproduced": sign_reproduced,
                "materiality_reproduced": materiality_reproduced,
                "passed": passed,
            }
        )
    failures = [entry for entry in entries if not entry["passed"]]
    failed_sentinels = sorted(
        str(entry["gt_owner_id"]) for entry in failures if entry["sentinel_owner"]
    )
    if failed_sentinels:
        _fail(
            f"mandatory sentinel owner(s) {failed_sentinels!r} failed the same-run positive "
            "control; unit.md requires both sentinels to pass regardless of the total count"
        )
    if len(failures) > MAX_CLEAN_REPLAY_FAILURES:
        _fail(
            f"{len(failures)} clean-row replays failed the same-run positive control, more than "
            f"the frozen {MAX_CLEAN_REPLAY_FAILURES}; unit.md stops the unit"
        )
    return {
        "gate": GATE_POSITIVE_CONTROLS,
        "passed": True,
        "checked_owner_count": len(entries),
        "failure_count": len(failures),
        "max_failures": MAX_CLEAN_REPLAY_FAILURES,
        "failed_owner_ids": sorted(str(entry["gt_owner_id"]) for entry in failures),
        "mandatory_sentinel_owner_ids": list(SENTINEL_OWNER_IDS),
        "sentinels_passed": True,
        "owner_rows": entries,
    }


# ---------------------------------------------------------------------------
# 5. Gate 6 and the frozen routes
# ---------------------------------------------------------------------------


def _quarantine_arithmetic(readouts: Sequence[OwnerReadout], *, cohort_role: str) -> dict[str, Any]:
    """The explicit material / nonmaterial / neither split of one stratum."""

    members = [item for item in readouts if item.cohort_role == cohort_role]
    material = sorted(
        item.gt_owner_id for item in members if item.interpretable and item.neutral_material
    )
    nonmaterial = sorted(
        item.gt_owner_id
        for item in members
        if item.interpretable and not item.neutral_material
    )
    neither = sorted(item.gt_owner_id for item in members if not item.interpretable)
    if len(material) + len(nonmaterial) + len(neither) != len(members):
        _fail("the material / nonmaterial / neither split does not partition the stratum")
    return {
        "cohort_role": cohort_role,
        "denominator": len(members),
        "material_count": len(material),
        "material_owner_ids": material,
        "nonmaterial_count": len(nonmaterial),
        "nonmaterial_owner_ids": nonmaterial,
        "neither_count": len(neither),
        "neither_owner_ids": neither,
        "neither_semantics": (
            "a quarantined or benign-blocked owner is neither material nor nonmaterial and "
            "therefore conservatively satisfies no route"
        ),
    }


def evaluate_specificity_gate(
    readouts: Sequence[OwnerReadout], *, cutoff: float = MATERIALITY_MAX_NATS
) -> dict[str, Any]:
    """``unit.md`` gate 6, on the absolute integer four over the frozen twelve."""

    members = [item for item in readouts if item.cohort_role == plan_builder.COHORT_SPECIFICITY]
    material = sorted(
        item.gt_owner_id
        for item in members
        if item.interpretable
        and is_material(
            float(item.row["neutral"]["relative_coordinate_delta"]), cutoff=cutoff
        )
    )
    passed = len(material) < SPECIFICITY_MATERIAL_MAX
    return {
        "gate": GATE_SPECIFICITY,
        "passed": passed,
        "denominator": len(members),
        "frozen_denominator": plan_builder.SPECIFICITY_OWNER_COUNT,
        "material_count": len(material),
        "material_owner_ids": material,
        "threshold": SPECIFICITY_MATERIAL_MAX,
        "threshold_is_absolute": True,
        "materiality_cutoff_nats": cutoff,
        "failure_route": ROUTE_NEUTRAL_NOT_NEUTRAL,
        "semantics": (
            "the threshold is the absolute integer four even when fewer than twelve owners are "
            "interpretable; a quarantined owner never counts as material"
        ),
    }


def _voting_side(
    readouts: Sequence[OwnerReadout], *, material: bool, cutoff: float
) -> dict[str, Any]:
    members = [item for item in readouts if item.votes and item.interpretable]
    qualifying = [
        item
        for item in members
        if is_material(float(item.row["neutral"]["relative_coordinate_delta"]), cutoff=cutoff)
        is material
    ]
    separated = [item for item in qualifying if item.clearly_separated]
    return {
        "owner_count": len(qualifying),
        "owner_ids": sorted(item.gt_owner_id for item in qualifying),
        "image_count": len({item.image_id for item in qualifying}),
        "image_ids": sorted({item.image_id for item in qualifying}),
        "clearly_separated_count": len(separated),
        "clearly_separated_owner_ids": sorted(item.gt_owner_id for item in separated),
        "clearly_separated_image_count": len({item.image_id for item in separated}),
        "clearly_separated_image_ids": sorted({item.image_id for item in separated}),
    }


def decide_route(
    readouts: Sequence[OwnerReadout],
    specificity: Mapping[str, Any],
    *,
    cutoff: float = MATERIALITY_MAX_NATS,
) -> dict[str, Any]:
    """Exactly one frozen route, in order, on absolute integer thresholds."""

    nonmaterial = _voting_side(readouts, material=False, cutoff=cutoff)
    material = _voting_side(readouts, material=True, cutoff=cutoff)
    c_ok = (
        nonmaterial["owner_count"] >= plan_builder.ROUTE_C_MIN_NONMATERIAL
        and nonmaterial["image_count"] >= plan_builder.ROUTE_C_MIN_IMAGES
        and nonmaterial["clearly_separated_count"] >= plan_builder.ROUTE_C_MIN_SEPARATED
        and nonmaterial["clearly_separated_image_count"]
        >= plan_builder.ROUTE_C_MIN_SEPARATED_IMAGES
    )
    generic_ok = (
        material["owner_count"] >= plan_builder.ROUTE_GENERIC_MIN_MATERIAL
        and material["image_count"] >= plan_builder.ROUTE_GENERIC_MIN_IMAGES
        and material["clearly_separated_count"] >= plan_builder.ROUTE_GENERIC_MIN_SEPARATED
        and material["clearly_separated_image_count"]
        >= plan_builder.ROUTE_GENERIC_MIN_SEPARATED_IMAGES
    )
    if not bool(specificity["passed"]):
        route = ROUTE_NEUTRAL_NOT_NEUTRAL
    elif c_ok:
        route = ROUTE_C_CONTENT_SPECIFIC
    elif generic_ok:
        route = ROUTE_GENERIC_SUFFICIENT
    else:
        route = ROUTE_INCONCLUSIVE
    return {
        "route": route,
        "permitted_reading": ROUTE_READINGS[route],
        "materiality_cutoff_nats": cutoff,
        "specificity_gate_passed": bool(specificity["passed"]),
        "voting_denominator": plan_builder.VOTING_OWNER_COUNT,
        "clearly_separated_denominator": len(plan_builder.FROZEN_CLEARLY_SEPARATED_OWNER_IDS),
        "denominators_are_absolute": True,
        "nonmaterial_side": nonmaterial,
        "material_side": material,
        "route_tests": {
            ROUTE_C_CONTENT_SPECIFIC: {
                "satisfied": c_ok,
                "requires": {
                    "min_nonmaterial_voting_owners": plan_builder.ROUTE_C_MIN_NONMATERIAL,
                    "min_images": plan_builder.ROUTE_C_MIN_IMAGES,
                    "min_clearly_separated": plan_builder.ROUTE_C_MIN_SEPARATED,
                    "min_clearly_separated_images": plan_builder.ROUTE_C_MIN_SEPARATED_IMAGES,
                },
            },
            ROUTE_GENERIC_SUFFICIENT: {
                "satisfied": generic_ok and bool(specificity["passed"]),
                "requires": {
                    "min_material_voting_owners": plan_builder.ROUTE_GENERIC_MIN_MATERIAL,
                    "min_images": plan_builder.ROUTE_GENERIC_MIN_IMAGES,
                    "min_clearly_separated": plan_builder.ROUTE_GENERIC_MIN_SEPARATED,
                    "min_clearly_separated_images": (
                        plan_builder.ROUTE_GENERIC_MIN_SEPARATED_IMAGES
                    ),
                    "requires_specificity_gate_pass": True,
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# 6. Non-voting sensitivities
# ---------------------------------------------------------------------------


def build_sensitivities(
    readouts: Sequence[OwnerReadout], primary_route: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """``unit.md`` "Non-voting sensitivities": recomputed, never a replacement."""

    results: list[dict[str, Any]] = []
    baseline_route = str(primary_route["route"])

    def _recompute(
        subset: Sequence[OwnerReadout], *, cutoff: float, name: str, description: str
    ) -> dict[str, Any]:
        specificity = evaluate_specificity_gate(subset, cutoff=cutoff)
        route = decide_route(subset, specificity, cutoff=cutoff)
        return {
            "name": name,
            "description": description,
            "role": "non_voting_descriptive_recomputation_never_replaces_the_frozen_route",
            "materiality_cutoff_nats": cutoff,
            "excluded_owner_ids": sorted(
                {item.gt_owner_id for item in readouts} - {item.gt_owner_id for item in subset}
            ),
            "specificity_gate": specificity,
            "route": route["route"],
            "route_changed": str(route["route"]) != baseline_route,
            "nonmaterial_side": route["nonmaterial_side"],
            "material_side": route["material_side"],
        }

    length_subset = [
        item
        for item in readouts
        if abs(int(item.row["neutral"]["row_length_delta_tokens"]))
        != plan_builder.MAX_ROW_LENGTH_DELTA_TOKENS
    ]
    results.append(
        _recompute(
            length_subset,
            cutoff=MATERIALITY_MAX_NATS,
            name=SENSITIVITY_EXCLUDE_LENGTH_DELTA_TWO,
            description=(
                "excludes owners whose selected neutral row differs from C by exactly two tokens"
            ),
        )
    )
    posthoc_subset = [
        item for item in readouts if not bool(item.row["posthoc_support_extent_uncertain"])
    ]
    results.append(
        _recompute(
            posthoc_subset,
            cutoff=MATERIALITY_MAX_NATS,
            name=SENSITIVITY_EXCLUDE_POSTHOC_UNCERTAIN,
            description=(
                "excludes the two posthoc plausible_support_extent_uncertain rows "
                f"{list(plan_builder.POSTHOC_SUPPORT_UNCERTAIN_OWNER_IDS)!r}"
            ),
        )
    )
    for cutoff in SENSITIVITY_CUTOFFS_NATS:
        results.append(
            _recompute(
                readouts,
                cutoff=float(cutoff),
                name=f"{SENSITIVITY_CUTOFF_PREFIX}_{cutoff}",
                description=f"recomputes the frozen route at a {cutoff} nat materiality cutoff",
            )
        )
    for entry in results:
        entry["cutoff_fragile"] = bool(
            entry["route_changed"]
            and str(entry["name"]).startswith(SENSITIVITY_CUTOFF_PREFIX)
        )
    return results


def build_strata(readouts: Sequence[OwnerReadout]) -> dict[str, Any]:
    """Descriptive per-axis counts.  Never a route, never pooled across images."""

    strata: dict[str, Any] = {}
    for axis in STRATUM_AXES:
        buckets: dict[str, dict[str, Any]] = {}
        for item in readouts:
            key = str(item.row["strata"][axis])
            bucket = buckets.setdefault(
                key,
                {
                    "owner_count": 0,
                    "interpretable_count": 0,
                    "neutral_material_count": 0,
                    "neutral_nonmaterial_count": 0,
                    "owner_ids": [],
                },
            )
            bucket["owner_count"] += 1
            bucket["owner_ids"].append(item.gt_owner_id)
            if not item.interpretable:
                continue
            bucket["interpretable_count"] += 1
            if item.neutral_material:
                bucket["neutral_material_count"] += 1
            else:
                bucket["neutral_nonmaterial_count"] += 1
        for bucket in buckets.values():
            bucket["owner_ids"] = sorted(bucket["owner_ids"])
        strata[axis] = dict(sorted(buckets.items()))
    strata["unknown_neutral_note"] = UNKNOWN_NEUTRAL_NOTE
    strata["role"] = "descriptive_only_never_a_route"
    return strata


# ---------------------------------------------------------------------------
# 7. Assembly, report and publication
# ---------------------------------------------------------------------------


def run_analysis(merged_dir: Path, *, plan_dir: Path | None = None) -> dict[str, Any]:
    """Bind, gate in the fixed order, route, and return the complete summary."""

    evidence = load_merged_evidence(merged_dir, plan_dir=plan_dir)
    by_key = index_rows(evidence.rows)
    merge_gates = evidence.receipt["gates"]
    replay_gate = merge_gates[GATE_RUNTIME_REPLAY]

    benign = evaluate_benign_gate(evidence, by_key)
    readouts = build_owner_rows(evidence, by_key, benign, replay_gate)
    positive = evaluate_positive_control_gate(readouts)
    specificity = evaluate_specificity_gate(readouts)
    route = decide_route(readouts, specificity)

    gates = {
        "order": list(GATE_ORDER),
        GATE_INPUT_AND_TOKEN_IDENTITY: {
            **dict(merge_gates[GATE_INPUT_AND_TOKEN_IDENTITY]),
            "owner": "merge",
        },
        GATE_RUNTIME_REPLAY: {**dict(replay_gate), "owner": "merge"},
        GATE_POSITIVE_CONTROLS: {**positive, "owner": "analysis"},
        GATE_BENIGN_REPLAY: {**benign, "owner": "analysis"},
        GATE_CACHE_PARITY: {**dict(merge_gates[GATE_CACHE_PARITY]), "owner": "merge"},
        GATE_SPECIFICITY: {**specificity, "owner": "analysis"},
    }
    quarantine_arithmetic = {
        role: _quarantine_arithmetic(readouts, cohort_role=role)
        for role in (plan_builder.COHORT_VOTING, plan_builder.COHORT_SPECIFICITY)
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "materiality": {
            "cutoff_nats": MATERIALITY_MAX_NATS,
            "cutoff_source": (
                "analyze_sorted_crossing_owner_row_geometry.MATERIAL_NEGATIVE_MAX_NATS"
            ),
            "imported_not_redeclared": True,
            "estimand": (
                "relative_delta(i) = coordinate_delta(i) - same_run_benign_coordinate_delta("
                "image(i))"
            ),
            "primary_segment": scorer.SEGMENT_COORDINATES,
            "benign_reference_source": "same_run_replay_only",
            "sealed_value_role": "gate_reference_only_never_the_estimand",
        },
        "cohort": evidence.plan.manifest.get("cohort"),
        "gates": gates,
        "quarantine_arithmetic": quarantine_arithmetic,
        "decision": route,
        "non_voting_sensitivities": build_sensitivities(readouts, route),
        "strata": build_strata(readouts),
        "evidence_separation": {
            "likelihood_versus_coverage": LIKELIHOOD_VERSUS_COVERAGE,
            "exact_row_likelihood_is_the_only_readout": True,
            "owner_coverage_measured_here": False,
            "owner_emission_measured_here": False,
            "strict_matching_measured_here": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "not_claimed": list(NOT_CLAIMED),
    }
    owner_rows = [readout.row for readout in readouts]
    geometry.assert_emitted_payload(summary, label="neutral-row summary")
    geometry.assert_emitted_payload(owner_rows, label="neutral-row owner rows")
    geometry.assert_no_cross_image_raw_delta_pooling(
        summary, label="neutral-row summary"
    )
    return {"evidence": evidence, "summary": summary, "owner_rows": owner_rows}


def _fmt(value: Any, *, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(summary: Mapping[str, Any], owner_rows: Sequence[Mapping[str, Any]]) -> str:
    decision = summary["decision"]
    gates = summary["gates"]
    lines: list[str] = [
        "# Sorted Crossing Matched-Length Neutral-Row Insertion Control",
        "",
        f"Route: **{decision['route']}**",
        "",
        decision["permitted_reading"],
        "",
        "## Gates, in the frozen order",
        "",
        "| # | Gate | Owner | Passed |",
        "| --- | --- | --- | --- |",
    ]
    for index, gate in enumerate(gates["order"], start=1):
        block = gates[gate]
        lines.append(
            f"| {index} | `{gate}` | {block.get('owner')} | {bool(block.get('passed'))} |"
        )
    replay = gates[GATE_RUNTIME_REPLAY]
    benign = gates[GATE_BENIGN_REPLAY]
    specificity = gates[GATE_SPECIFICITY]
    lines += [
        "",
        f"Quarantined owners: {replay.get('quarantined_owner_count')} "
        f"(cap {replay.get('max_quarantined_owners')}); "
        f"benign-blocked images: {benign.get('blocked_image_ids')}.",
        "",
        f"Specificity: {specificity['material_count']} of "
        f"{specificity['frozen_denominator']} specificity owners are material at the absolute "
        f"threshold {specificity['threshold']}.",
        "",
        "## Route arithmetic",
        "",
        "| Side | Owners | Images | Clearly separated | Separated images |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, key in (("N nonmaterial", "nonmaterial_side"), ("N material", "material_side")):
        side = decision[key]
        lines.append(
            f"| {label} | {side['owner_count']}/{decision['voting_denominator']} | "
            f"{side['image_count']} | "
            f"{side['clearly_separated_count']}/{decision['clearly_separated_denominator']} | "
            f"{side['clearly_separated_image_count']} |"
        )
    for role, block in summary["quarantine_arithmetic"].items():
        lines += [
            "",
            f"`{role}`: {block['material_count']} material + {block['nonmaterial_count']} "
            f"nonmaterial + {block['neither_count']} neither = {block['denominator']}. "
            f"{block['neither_semantics']}.",
        ]
    lines += [
        "",
        "## Owner-level readouts",
        "",
        "| Owner | Image | Role | N relative | N material | C relative | C material | Interp. |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in owner_rows:
        lines.append(
            f"| `{row['gt_owner_id']}` | {row['image_id']} | {row['cohort_role']} | "
            f"{_fmt(row['neutral']['relative_coordinate_delta'])} | "
            f"{row['neutral']['material']} | "
            f"{_fmt(row['clean']['relative_coordinate_delta'])} | "
            f"{row['clean']['material']} | {row['interpretable']} |"
        )
    lines += ["", "## Non-voting sensitivities", "", "| Name | Route | Changed | Fragile |", "| --- | --- | --- | --- |"]
    for entry in summary["non_voting_sensitivities"]:
        lines.append(
            f"| `{entry['name']}` | {entry['route']} | {entry['route_changed']} | "
            f"{entry['cutoff_fragile']} |"
        )
    lines += [
        "",
        "## Evidence boundary",
        "",
        summary["evidence_separation"]["likelihood_versus_coverage"] + ".",
        "",
        UNKNOWN_NEUTRAL_NOTE + ".",
        "",
        "Not claimed:",
        "",
        *[f"- {item}" for item in summary["not_claimed"]],
        "",
    ]
    return "\n".join(lines)


def build_output_files(result: Mapping[str, Any]) -> dict[str, bytes]:
    evidence: MergedEvidence = result["evidence"]
    summary = result["summary"]
    owner_rows = result["owner_rows"]
    files: dict[str, bytes] = {
        OWNER_ROWS_NAME: b"".join(canonical_json_bytes(row) + b"\n" for row in owner_rows),
        SUMMARY_NAME: canonical_json_bytes(summary) + b"\n",
        REPORT_MD_NAME: render_markdown(summary, owner_rows).encode("utf-8"),
    }
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "analyzer_source_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "source_identity": {
            name: sha256_file(Path(module.__file__))
            for name, module in (
                ("prepare_sorted_crossing_neutral_row_control", plan_builder),
                ("score_sorted_crossing_neutral_row_control", scorer),
                ("merge_sorted_crossing_neutral_row_control", merger),
                ("analyze_sorted_crossing_owner_row_geometry", geometry),
            )
        },
        "binding": {
            "merged_dir": str(evidence.merged_dir),
            "merge_receipt_content_sha256": evidence.receipt_content_sha256,
            "plan_dir": str(evidence.plan.plan_dir),
            "plan_manifest_content_sha256": evidence.plan.manifest_content_sha256,
            "runtime_identity_sha256": evidence.receipt.get("runtime_identity_sha256"),
        },
        "input_file_sha256": dict(evidence.file_sha256),
        "materiality_cutoff_nats": MATERIALITY_MAX_NATS,
        "materiality_cutoff_source": (
            "analyze_sorted_crossing_owner_row_geometry.MATERIAL_NEGATIVE_MAX_NATS"
        ),
        "gate_order": list(GATE_ORDER),
        "decision_route": str(summary["decision"]["route"]),
        "quarantine_arithmetic": summary["quarantine_arithmetic"],
        "policy": {
            "adaptive_threshold": False,
            "second_neutral_row_choice": False,
            "automatic_gpu_successor": False,
            "sensitivities_are_non_voting": True,
            "sealed_reference_role": "gate_reference_only_never_the_estimand",
            "same_run_benign_replay_owns_both_relative_estimands": True,
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "not_claimed": list(NOT_CLAIMED),
        "artifact_determinism": "no_wall_clock_or_host_specific_field_is_sealed",
        "output_file_digests": {
            name: {"path": name, "byte_size": len(payload), "sha256": sha256_bytes(payload)}
            for name, payload in sorted(files.items())
        },
    }
    receipt["receipt_content_sha256"] = sha256_json(receipt)
    files[RECEIPT_NAME] = canonical_json_bytes(receipt) + b"\n"
    return files


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merged-dir", required=True, type=Path, help="immutable merged evidence directory"
    )
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=None,
        help="sealed CPU plan directory; defaults to the merge receipt's own binding",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="analysis output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_analysis(Path(args.merged_dir), plan_dir=args.plan_dir)
        files = build_output_files(result)
        crossing_scorer._publish(Path(args.output_dir), files)  # noqa: SLF001
    except (
        NeutralRowAnalysisContractError,
        geometry.GeometryContractError,
        crossing_scorer.CrossingBoundaryContractError,
    ) as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        return 1
    decision = result["summary"]["decision"]
    counts = Counter(str(row["cohort_role"]) for row in result["owner_rows"])
    print(
        f"neutral-row route: {decision['route']} "
        f"nonmaterial={decision['nonmaterial_side']['owner_count']}/"
        f"{decision['voting_denominator']} "
        f"material={decision['material_side']['owner_count']}/"
        f"{decision['voting_denominator']} "
        f"cohort={dict(sorted(counts.items()))}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
