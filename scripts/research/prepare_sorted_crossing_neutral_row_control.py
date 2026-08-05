#!/usr/bin/env python3
"""CPU-only fail-closed plan builder for the sorted crossing matched-length
neutral-row insertion control
(``2026-08-04-sorted-crossing-matched-length-neutral-row-insertion-control``).

Frozen unit::

    research/investigations/qwen3-vl-dense-enumeration/experiments/
    2026-08-04-sorted-crossing-matched-length-neutral-row-insertion-control/unit.md

What this module is
-------------------
``unit.md`` asks one bounded question at the frozen sorted crossing boundaries:
does inserting a real, already-covered, different-description, geometrically
separated, length-matched control row -- the **neutral row** ``N`` -- reproduce
the downstream exact-coordinate likelihood tail that inserting the clean
skipped-owner row ``C`` produced?

This planner re-verifies every sealed input, deterministically selects one ``N``
per crossing owner from CPU-visible census/crossing/geometry data alone, and
emits the sealed request plan a later paired-score pass executes unchanged.  It
loads no model, no tokenizer, and no new logit, and it never inspects a
per-owner likelihood delta beyond the frozen material/nonmaterial cohort label.

The deterministic neutral-row rule
----------------------------------
A candidate ``N`` must satisfy all five sealed unit.md predicates, plus the
structural requirement that it can be assembled without retokenization.
:data:`PREDICATE_ORDER` is the exact evaluation order, in which assemblability
precedes the exact row-length measurement it makes possible.  Candidates are
ordered by smallest absolute
row-token-length difference from ``C``, then latest strict-matched native row
index before ``P``, then largest minimum normalized center distance to ``C`` and
``E``, then ascending physical-owner id; the first candidate is selected.  There
is no minimum center-distance floor, no alternate neutral row, no sampling and
no manual reselection.

The CPU plan owns the frozen split and fails closed unless it reproduces exactly
nine voting owners, twelve feasible specificity owners and five ledgered
infeasible owners, and exactly 21 + 21 + 12 requests.  Image ``2299`` is
prohibited everywhere.

Exact token ownership
---------------------
Nothing is retokenized.  ``N`` is the sealed category ``query_suffix_token_ids``
of its own ``(image, normalized description)`` plus its sealed ``exact_gt_anchor``
four coordinate token ids plus the exact ``<|box_end|>`` token.  ``C`` and the
scored ``E`` row carry their sealed predecessor token identities verbatim, so the
scored ``E`` token ids and digest are identical across the ``N`` and ``C`` arms by
construction and are proven so before a request is emitted.

Historical sealed likelihood values travel in the plan as **gate references
only**.  The same-run benign replay -- never a sealed historical number -- owns
both the ``N`` and the ``C`` relative estimands.

What it publishes
-----------------
Under a CLI-supplied output root, atomically and create-or-identical::

    plan/manifest.json
    plan/selection-registry.jsonl
    plan/benign-registry.jsonl
    plan/request-plan.jsonl

Every input and output receives path, byte size and SHA-256 lineage in the
manifest, and the manifest self-seals.  Nothing is ever written to a predecessor
run root.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import analyze_sorted_crossing_owner_row_geometry as geometry  # noqa: E402
from scripts.research import (  # noqa: E402
    prepare_sorted_crossing_boundary_owner_release_realization as crossing_plan,
)
from scripts.research import (  # noqa: E402
    score_sorted_crossing_boundary_owner_release as crossing_scorer,
)

UNIT_ID = "2026-08-04-sorted-crossing-matched-length-neutral-row-insertion-control"
GEOMETRY_UNIT_ID = geometry.UNIT_ID
CROSSING_UNIT_ID = geometry.SOURCE_UNIT_ID
CENSUS_UNIT_ID = geometry.CENSUS_UNIT_ID

MANIFEST_SCHEMA_VERSION = "sorted-crossing-neutral-row-control-plan.v1"
SELECTION_SCHEMA_VERSION = "sorted-crossing-neutral-row-control-selection.v1"
BENIGN_SCHEMA_VERSION = "sorted-crossing-neutral-row-control-benign.v1"
REQUEST_SCHEMA_VERSION = "sorted-crossing-neutral-row-control-request.v1"

PLAN_DIR_NAME = "plan"
MANIFEST_NAME = "manifest.json"
SELECTION_REGISTRY_NAME = "selection-registry.jsonl"
BENIGN_REGISTRY_NAME = "benign-registry.jsonl"
REQUEST_PLAN_NAME = "request-plan.jsonl"

#: Every conclusion-bearing input, source and output identity carries these
#: three fields; a digest alone is not a complete seal.
SEAL_FIELDS: tuple[str, ...] = ("path", "byte_size", "sha256")

# --- Immutable inputs (defaults only; every path is overridable on the CLI) ---

DEFAULT_GEOMETRY_ANALYSIS_DIR = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-04-sorted-crossing-owner-row-geometric-relation-stratification/"
    "20260804T060015Z/geometry-analysis"
)
DEFAULT_CROSSING_PLAN_DIR = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-03-sorted-crossing-boundary-owner-release-realization/20260804T013856Z/plan"
)
DEFAULT_SECONDARY_MERGED_DIR = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-03-sorted-crossing-boundary-owner-release-realization/20260804T020853Z/"
    "secondary-merged-v2"
)

GEOMETRY_RECEIPT_NAME = geometry.RECEIPT_NAME
GEOMETRY_OWNER_ROWS_NAME = geometry.OWNER_ROWS_NAME
CROSSING_MANIFEST_NAME = crossing_plan.MANIFEST_NAME
CROSSING_COHORT_REGISTRY_NAME = crossing_plan.COHORT_REGISTRY_NAME
CROSSING_CONTROL_REGISTRY_NAME = crossing_plan.CONTROL_REGISTRY_NAME
SECONDARY_MERGE_RECEIPT_NAME = geometry.SECONDARY_MERGE_RECEIPT_NAME
SECONDARY_MERGED_ROWS_NAME = geometry.SECONDARY_MERGED_ROWS_NAME
CENSUS_RECEIPT_NAME = geometry.CENSUS_RECEIPT_NAME

#: Census plan registries this planner reads.  Each is proven against the census
#: ``plan/receipt.json`` and against the crossing plan's own sealed lineage.
CENSUS_OWNER_REGISTRY_NAME = geometry.CENSUS_OWNER_REGISTRY_NAME
CENSUS_IMAGE_REGISTRY_NAME = geometry.CENSUS_IMAGE_REGISTRY_NAME
CENSUS_CATEGORY_REGISTRY_NAME = "category-registry.jsonl"
CENSUS_CANDIDATE_BANK_NAME = "candidate-bank.jsonl"
CENSUS_SIDECAR_REGISTRY_NAME = "native-sidecar-registry.jsonl"
CENSUS_CONTEXT_REGISTRY_NAME = "context-registry.jsonl"
CENSUS_PLAN_FILES: tuple[str, ...] = (
    CENSUS_OWNER_REGISTRY_NAME,
    CENSUS_IMAGE_REGISTRY_NAME,
    CENSUS_CATEGORY_REGISTRY_NAME,
    CENSUS_CANDIDATE_BANK_NAME,
    CENSUS_SIDECAR_REGISTRY_NAME,
    CENSUS_CONTEXT_REGISTRY_NAME,
)

# --- Frozen cohort (unit.md "Cohort and strata") ------------------------------

#: The completed human-refined twelve-image panel is the only image set here.
IMAGE_COUNT = 12
PROHIBITED_IMAGE_IDS: frozenset[str] = frozenset({"2299"})
CROSSING_OWNER_COUNT = geometry.FROZEN_DENOMINATORS.crossing_owner_count
BENIGN_CONTROL_COUNT = geometry.FROZEN_DENOMINATORS.benign_control_count

VOTING_OWNER_COUNT = 9
SPECIFICITY_OWNER_COUNT = 12
INFEASIBLE_OWNER_COUNT = 5

#: unit.md names these five ledgered structurally infeasible ``C``-nonmaterial
#: owners and these four clearly separated voting owners explicitly.  They are
#: re-derived here and a drift fails the build rather than being reported.
FROZEN_INFEASIBLE_OWNER_IDS: tuple[str, ...] = (
    "gt:13923:1",
    "gt:14439:3",
    "gt:16228:3",
    "gt:5001:10",
    "gt:7511:1",
)
FROZEN_CLEARLY_SEPARATED_OWNER_IDS: tuple[str, ...] = (
    "gt:13348:7",
    "gt:16228:11",
    "gt:4134:27",
    "gt:4134:29",
)
#: unit.md "Same-run positive controls": these two must pass the ``C`` replay
#: gate regardless of the total mismatch count.
SENTINEL_OWNER_IDS: tuple[str, ...] = ("gt:13923:14", "gt:4134:27")
#: unit.md "Non-voting sensitivities": the two posthoc
#: ``plausible_support_extent_uncertain`` rows.
POSTHOC_SUPPORT_UNCERTAIN_OWNER_IDS: tuple[str, ...] = ("gt:16228:11", "gt:4134:29")

COHORT_VOTING = "material_skipped_owner_voting"
COHORT_SPECIFICITY = "c_nonmaterial_specificity"
COHORT_INFEASIBLE = "c_nonmaterial_structurally_infeasible"
COHORT_BENIGN = "same_run_benign_reference"
COHORT_ROLES: tuple[str, ...] = (
    COHORT_VOTING,
    COHORT_SPECIFICITY,
    COHORT_INFEASIBLE,
)

# --- Frozen selection rule (unit.md "Deterministic neutral-row selection") -----

#: unit.md predicate 5.
MAX_ROW_LENGTH_DELTA_TOKENS = 2

PREDICATE_SAME_IMAGE_CENSUS_OWNER = "same_image_census_owner_not_excluded"
PREDICATE_NOT_C_AND_NOT_E_OWNER = "neither_c_nor_the_strict_matched_owner_of_e"
PREDICATE_COVERED_BEFORE_P = "strict_matched_a_native_row_at_index_below_p"
PREDICATE_DIFFERENT_DESCRIPTION = "description_differs_from_both_c_and_e"
PREDICATE_GEOMETRICALLY_SEPARATED = "iou_zero_and_no_center_containment_against_c_and_e"
#: Not a scientific predicate but the structural prerequisite of the one that
#: follows it: the sealed query suffix is what defines the row length, so a
#: candidate with no sealed admitted category suffix or no sealed exact-GT
#: anchor is refused *before* its exact row length can be measured, rather than
#: having that length guessed from a reconstruction.
PREDICATE_ASSEMBLABLE = "assemblable_from_sealed_tokens_without_retokenization"
#: unit.md predicate 5, measured on the assembled sealed tokens.
PREDICATE_MATCHED_ROW_LENGTH = "clean_row_token_length_within_two_tokens_of_c"

#: The exact evaluation order of :func:`evaluate_candidate`, declared once and
#: sealed into every selection row.
PREDICATE_ORDER: tuple[str, ...] = (
    PREDICATE_SAME_IMAGE_CENSUS_OWNER,
    PREDICATE_NOT_C_AND_NOT_E_OWNER,
    PREDICATE_COVERED_BEFORE_P,
    PREDICATE_DIFFERENT_DESCRIPTION,
    PREDICATE_GEOMETRICALLY_SEPARATED,
    PREDICATE_ASSEMBLABLE,
    PREDICATE_MATCHED_ROW_LENGTH,
)
#: The entries of :data:`PREDICATE_ORDER` that are structural prerequisites of a
#: later measurement rather than unit.md scientific predicates.
STRUCTURAL_PREREQUISITE_PREDICATES: tuple[str, ...] = (PREDICATE_ASSEMBLABLE,)

ORDERING_RULE: tuple[str, ...] = (
    "smallest_absolute_row_token_length_difference_from_c",
    "latest_strict_matched_native_row_index_before_p",
    "largest_minimum_normalized_center_distance_to_c_and_e",
    "ascending_physical_owner_id",
)

# --- Frozen requests (unit.md "Requests and score semantics") ------------------

ARM_NEUTRAL = "p_plus_n_then_e"
ARM_CLEAN_REPLAY = "p_plus_c_then_e"
ARM_BENIGN = "benign_substitution_then_following_native_action"
ARMS: tuple[str, ...] = (ARM_NEUTRAL, ARM_CLEAN_REPLAY, ARM_BENIGN)

REQUEST_NEUTRAL_INSERTION = "neutral_row_insertion_then_downstream_row"
REQUEST_CLEAN_REPLAY = "clean_skipped_owner_replay_then_downstream_row"
REQUEST_BENIGN_REPLAY = "benign_substitution_replay_then_following_native_action"
REQUEST_FAMILY_BY_ARM: Mapping[str, str] = {
    ARM_NEUTRAL: REQUEST_NEUTRAL_INSERTION,
    ARM_CLEAN_REPLAY: REQUEST_CLEAN_REPLAY,
    ARM_BENIGN: REQUEST_BENIGN_REPLAY,
}
APPENDED_ROLE_BY_ARM: Mapping[str, str] = {
    ARM_NEUTRAL: "inserted_deterministic_neutral_control_row_n",
    ARM_CLEAN_REPLAY: "inserted_exact_clean_gt_row_c",
    ARM_BENIGN: "inserted_exact_clean_gt_twin_of_a_native_true_positive_row",
}

EXECUTED_OWNER_COUNT = VOTING_OWNER_COUNT + SPECIFICITY_OWNER_COUNT
EXPECTED_REQUEST_COUNT_BY_ARM: Mapping[str, int] = {
    ARM_NEUTRAL: EXECUTED_OWNER_COUNT,
    ARM_CLEAN_REPLAY: EXECUTED_OWNER_COUNT,
    ARM_BENIGN: BENIGN_CONTROL_COUNT,
}
TOTAL_REQUEST_COUNT = sum(EXPECTED_REQUEST_COUNT_BY_ARM.values())

# --- Frozen gates and routes (unit.md "Validity gates" / "Frozen routes") ------

#: The image-referenced materiality cutoff, imported rather than re-declared.
MATERIALITY_MAX_NATS = geometry.MATERIAL_NEGATIVE_MAX_NATS
#: unit.md gate 2 / 4 / 5: selected-logit replay and parity tolerance.
REPLAY_MAX_SELECTED_LOGIT_ABS_DIFF = crossing_scorer.CACHE_PARITY_MAX_SELECTED_LOGIT_ABS_DIFF
#: unit.md gate 3 / 4: raw coordinate-delta agreement tolerance.
REPLAY_MAX_COORDINATE_DELTA_ABS_DIFF = 0.05
#: unit.md gate 2: more than two quarantined owners stops the unit.
MAX_QUARANTINED_OWNERS = crossing_scorer.MAX_PRIMARY_QUARANTINES
#: unit.md gate 3: more than two ``C`` replay failures stops the unit.
MAX_CLEAN_REPLAY_FAILURES = 2
#: unit.md gate 6: absolute over the frozen twelve specificity owners.
SPECIFICITY_MATERIAL_MAX = 4

ROUTE_C_CONTENT_SPECIFIC = "c_content_specific_interference_survives"
ROUTE_GENERIC_SUFFICIENT = "generic_boundary_sensitivity_sufficient_weak_close"
ROUTE_NEUTRAL_NOT_NEUTRAL = "neutral_row_not_neutral_inconclusive"
ROUTE_INCONCLUSIVE = "inconclusive"
ROUTE_ORDER: tuple[str, ...] = (
    ROUTE_C_CONTENT_SPECIFIC,
    ROUTE_GENERIC_SUFFICIENT,
    ROUTE_INCONCLUSIVE,
)

#: Absolute integers over the frozen nine voting owners and their four clearly
#: separated members.  A quarantined voting owner satisfies neither side.
ROUTE_C_MIN_NONMATERIAL = 5
ROUTE_C_MIN_IMAGES = 3
ROUTE_C_MIN_SEPARATED = 2
ROUTE_C_MIN_SEPARATED_IMAGES = 2
ROUTE_GENERIC_MIN_MATERIAL = 7
ROUTE_GENERIC_MIN_IMAGES = 3
ROUTE_GENERIC_MIN_SEPARATED = 3
ROUTE_GENERIC_MIN_SEPARATED_IMAGES = 2

#: unit.md "Non-voting sensitivities": descriptive recomputations that never
#: replace the frozen route.
SENSITIVITY_CUTOFFS_NATS: tuple[float, ...] = (-0.75, -1.25)

#: The two descriptive axes of a crossing owner, imported from the frozen
#: geometry analysis rather than re-declared: ``("matched_e", "unmatched_e")``
#: and ``("same_description", "different_description")``.
E_STRATUM_AXIS: tuple[str, ...] = geometry.E_STRATUM_AXIS
DESCRIPTION_AXIS: tuple[str, ...] = geometry.DESCRIPTION_AXIS
E_STRATUM_MATCHED, E_STRATUM_UNMATCHED = E_STRATUM_AXIS
DESCRIPTION_SAME, DESCRIPTION_DIFFERENT = DESCRIPTION_AXIS
#: unit.md "Representative smoke": the four strata the one real smoke image must
#: exercise inside a single image session before any full capture runs.
REQUIRED_SMOKE_STRATA: tuple[str, ...] = (*E_STRATUM_AXIS, *DESCRIPTION_AXIS)

CLAIM_BOUNDARY = (
    "one exact added row at the frozen twelve-image sorted crossing boundaries, read "
    "as exact teacher-forced downstream token likelihood; never owner emission, "
    "final-set coverage, eventual recovery, free rollout, training, or architecture"
)

SEMANTICS_NOTES: tuple[str, ...] = (
    "This planner is score-blind for selection: candidate admission and ordering "
    "read only sealed owner, category, candidate-bank, native-sidecar, crossing and "
    "geometry registries, never a new logit, a per-owner likelihood delta beyond the "
    "frozen material/nonmaterial cohort label, or a manual preference.",
    "The neutral row N is neutral only with respect to the declared description and "
    "geometry relations: because it was already emitted before the boundary, "
    "inserting it is both a duplicate and a regression in the learned sorted route, "
    "so the interpretation is asymmetric and nonmaterial N is a stacked-deck test.",
    "The same-run benign replay owns both the N and the C relative estimands; every "
    "sealed historical likelihood value in this plan is a gate reference only and "
    "never enters an estimand.",
    "N is assembled without retokenization from the sealed category query suffix, the "
    "owner's sealed exact_gt_anchor coordinate tokens and the exact <|box_end|> "
    "token; C and E carry their sealed predecessor token identities verbatim.",
    "The scored E token ids and digest are identical across the N and C arms by "
    "construction and are proven identical before any request is emitted.",
    "Route thresholds and the specificity threshold are absolute integers over the "
    "frozen denominators; a quarantined owner is neither material nor nonmaterial and "
    "therefore conservatively satisfies no route.",
    "There is no minimum center-distance floor, alternate neutral row, uncovered-row "
    "arm, position-gap arm, sampling or manual reselection in this unit.",
)


class NeutralRowPlanContractError(RuntimeError):
    """A precondition for the sealed neutral-row CPU plan was not proven."""


def _fail(message: str) -> NoReturn:
    raise NeutralRowPlanContractError(message)


canonical_json_bytes = geometry.canonical_json_bytes
sha256_bytes = geometry.sha256_bytes
sha256_json = geometry.sha256_json
sha256_file = geometry.sha256_file
context_id_for = crossing_plan.context_id_for


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return geometry.read_json(Path(path), label)
    except geometry.GeometryContractError as exc:  # pragma: no cover - re-raise shape
        _fail(str(exc))


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        return geometry.read_jsonl(Path(path), label)
    except geometry.GeometryContractError as exc:  # pragma: no cover - re-raise shape
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


def _decode_box(coord_token_ids: Any, *, label: str) -> geometry.Box:
    try:
        return geometry.decode_box(coord_token_ids, label=label)
    except geometry.GeometryContractError as exc:
        _fail(str(exc))


def _box_geometry(
    reference: geometry.Box, other: geometry.Box, *, label: str
) -> dict[str, Any]:
    try:
        return geometry.box_geometry(reference, other, label=label)
    except geometry.GeometryContractError as exc:
        _fail(str(exc))


def _token_ids(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(f"{label} is not a token-id sequence")
    tokens: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            _fail(f"{label} carries a non-integer token id {item!r}")
        tokens.append(int(item))
    return tokens


def _declared_digest(digests: Mapping[str, Any], name: str, *, label: str) -> str:
    entry = digests.get(name)
    if isinstance(entry, Mapping):
        entry = entry.get("sha256")
    if not isinstance(entry, str) or len(entry) != 64:
        _fail(f"{label} declares no sha256 digest for {name!r}")
    return entry


def _index_unique(
    rows: Sequence[Mapping[str, Any]], key: str, *, label: str
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            _fail(f"{label} row carries no {key}")
        if value in index:
            _fail(f"{label} carries a duplicate {key}: {value}")
        index[value] = dict(row)
    return index


def _assert_no_prohibited_image(image_ids: Sequence[str], *, label: str) -> None:
    present = sorted(set(image_ids) & PROHIBITED_IMAGE_IDS)
    if present:
        _fail(
            f"{label} carries prohibited image(s) {present!r}; image 2299 and the "
            "prospective thirteen-image panel are excluded from every denominator of "
            "this unit"
        )


# ---------------------------------------------------------------------------
# 1. Sealed inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SealedInputs:
    """Every sealed predecessor surface this planner is allowed to read."""

    geometry_dir: Path
    geometry_receipt: dict[str, Any]
    geometry_receipt_content_sha256: str
    geometry_rows: tuple[dict[str, Any], ...]
    crossing_plan_dir: Path
    crossing_manifest: dict[str, Any]
    crossing_manifest_content_sha256: str
    cohort_by_owner: dict[str, dict[str, Any]]
    benign_controls: tuple[dict[str, Any], ...]
    secondary_merged_dir: Path
    secondary_merge_receipt_content_sha256: str
    merged_by_key: dict[tuple[str, str], dict[str, Any]]
    census_plan_dir: Path
    census_receipt_content_sha256: str
    owners_by_id: dict[str, dict[str, Any]]
    images_by_id: dict[str, dict[str, Any]]
    categories_by_query_id: dict[str, dict[str, Any]]
    exact_anchor_by_owner: dict[str, dict[str, Any]]
    matched_rows_by_owner: dict[str, list[dict[str, Any]]]
    contexts_by_id: dict[str, dict[str, Any]]
    file_sha256: dict[str, str]


def load_sealed_inputs(
    *,
    geometry_analysis_dir: Path,
    crossing_plan_dir: Path,
    secondary_merged_dir: Path,
    census_plan_dir: Path | None = None,
) -> SealedInputs:
    """Bind every declared digest before a single selection field exists.

    The chain is: the geometry receipt self-seals and names the crossing plan
    manifest, the secondary merge receipt and the census plan receipt it was
    decided over; the crossing plan manifest self-seals and owns the cohort and
    control registry digests plus the full census lineage; the census plan
    receipt self-seals and owns every census registry digest.  Every hop is
    proven before a row is interpreted.
    """

    geometry_dir = Path(geometry_analysis_dir)
    file_sha256: dict[str, str] = {}

    geometry_receipt = _read_json(
        geometry_dir / GEOMETRY_RECEIPT_NAME, "geometry analysis receipt"
    )
    geometry_seal = _assert_self_sealed(
        geometry_receipt,
        digest_key="receipt_content_sha256",
        label="geometry analysis receipt",
    )
    if str(geometry_receipt.get("unit_id")) != GEOMETRY_UNIT_ID:
        _fail(
            f"geometry analysis receipt belongs to unit {geometry_receipt.get('unit_id')!r}, "
            f"not {GEOMETRY_UNIT_ID!r}"
        )
    geometry_outputs = geometry_receipt.get("output_file_digests")
    if not isinstance(geometry_outputs, Mapping):
        _fail("geometry analysis receipt carries no output_file_digests")
    geometry_rows_path = geometry_dir / GEOMETRY_OWNER_ROWS_NAME
    file_sha256[str(geometry_rows_path)] = _assert_file_digest(
        geometry_rows_path,
        expected=_declared_digest(
            geometry_outputs, GEOMETRY_OWNER_ROWS_NAME, label="geometry analysis receipt"
        ),
        label="geometry owner rows",
    )
    file_sha256[str(geometry_dir / GEOMETRY_RECEIPT_NAME)] = sha256_file(
        geometry_dir / GEOMETRY_RECEIPT_NAME
    )
    geometry_rows = _read_jsonl(geometry_rows_path, "geometry owner rows")

    binding = geometry_receipt.get("binding")
    if not isinstance(binding, Mapping):
        _fail("geometry analysis receipt carries no binding block")
    geometry_inputs = geometry_receipt.get("input_file_sha256")
    if not isinstance(geometry_inputs, Mapping):
        _fail("geometry analysis receipt carries no input_file_sha256")

    # --- crossing plan -----------------------------------------------------
    crossing_plan_dir = Path(crossing_plan_dir)
    manifest_path = crossing_plan_dir / CROSSING_MANIFEST_NAME
    crossing_manifest = _read_json(manifest_path, "crossing plan manifest")
    crossing_seal = _assert_self_sealed(
        crossing_manifest,
        digest_key="manifest_content_sha256",
        label="crossing plan manifest",
    )
    if str(crossing_manifest.get("unit_id")) != CROSSING_UNIT_ID:
        _fail(
            f"crossing plan manifest belongs to unit {crossing_manifest.get('unit_id')!r}, "
            f"not {CROSSING_UNIT_ID!r}"
        )
    if crossing_seal != str(binding.get("plan_manifest_content_sha256")):
        _fail(
            "the crossing plan manifest is not the one the sealed geometry analysis was "
            "decided over; the two predecessor artifacts disagree"
        )
    for name in (CROSSING_MANIFEST_NAME, CROSSING_COHORT_REGISTRY_NAME):
        declared = geometry_inputs.get(str(crossing_plan_dir / name))
        if not isinstance(declared, str):
            _fail(
                f"the geometry analysis receipt seals no digest for {crossing_plan_dir / name}; "
                "the crossing plan cannot be bound to the geometry cohort"
            )
        file_sha256[str(crossing_plan_dir / name)] = _assert_file_digest(
            crossing_plan_dir / name, expected=declared, label=f"crossing plan {name}"
        )
    crossing_outputs = crossing_manifest.get("output_file_digests")
    if not isinstance(crossing_outputs, Mapping):
        _fail("crossing plan manifest carries no output_file_digests")
    for name in (CROSSING_COHORT_REGISTRY_NAME, CROSSING_CONTROL_REGISTRY_NAME):
        file_sha256[str(crossing_plan_dir / name)] = _assert_file_digest(
            crossing_plan_dir / name,
            expected=_declared_digest(
                crossing_outputs, name, label="crossing plan manifest"
            ),
            label=f"crossing plan {name}",
        )
    cohort_rows = _read_jsonl(
        crossing_plan_dir / CROSSING_COHORT_REGISTRY_NAME, "crossing cohort registry"
    )
    control_rows = _read_jsonl(
        crossing_plan_dir / CROSSING_CONTROL_REGISTRY_NAME, "crossing control registry"
    )
    if len(cohort_rows) != CROSSING_OWNER_COUNT:
        _fail(
            f"the crossing cohort registry holds {len(cohort_rows)} rows, not the frozen "
            f"{CROSSING_OWNER_COUNT}"
        )
    benign_controls = tuple(
        row
        for row in control_rows
        if str(row.get("cohort")) == crossing_plan.TP_REPLAY_CONTROL_COHORT
    )
    if len(benign_controls) != BENIGN_CONTROL_COUNT:
        _fail(
            f"the crossing control registry holds {len(benign_controls)} benign-reference "
            f"controls, not the frozen {BENIGN_CONTROL_COUNT}"
        )
    cohort_by_owner = _index_unique(cohort_rows, "gt_owner_id", label="crossing cohort registry")

    # --- sealed secondary capture -----------------------------------------
    secondary_merged_dir = Path(secondary_merged_dir)
    merge_receipt = _read_json(
        secondary_merged_dir / SECONDARY_MERGE_RECEIPT_NAME, "secondary merge receipt"
    )
    merge_seal = _assert_self_sealed(
        merge_receipt, digest_key="receipt_content_sha256", label="secondary merge receipt"
    )
    if merge_seal != str(binding.get("secondary_merge_receipt_content_sha256")):
        _fail(
            "the secondary merge receipt is not the one the sealed geometry analysis was "
            "decided over; the sealed reference values would describe another capture"
        )
    merge_outputs = merge_receipt.get("output_file_digests")
    if not isinstance(merge_outputs, Mapping):
        _fail("secondary merge receipt carries no output_file_digests")
    merged_rows_path = secondary_merged_dir / SECONDARY_MERGED_ROWS_NAME
    file_sha256[str(merged_rows_path)] = _assert_file_digest(
        merged_rows_path,
        expected=_declared_digest(
            merge_outputs, SECONDARY_MERGED_ROWS_NAME, label="secondary merge receipt"
        ),
        label="secondary merged rows",
    )
    merged_rows = _read_jsonl(merged_rows_path, "secondary merged rows")
    merged_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in merged_rows:
        key = (str(row.get("gt_owner_id")), str(row.get("variant")))
        if key in merged_by_key:
            _fail(f"the sealed secondary capture carries duplicate readout {key!r}")
        merged_by_key[key] = dict(row)

    # --- census plan registries -------------------------------------------
    census_run_root = Path(str(binding.get("census_run_root", "")))
    census_dir = (
        Path(census_plan_dir) if census_plan_dir is not None else census_run_root / "plan"
    )
    census_receipt = _read_json(census_dir / CENSUS_RECEIPT_NAME, "census plan receipt")
    census_seal = _assert_self_sealed(
        census_receipt, digest_key="receipt_content_sha256", label="census plan receipt"
    )
    if census_seal != str(binding.get("census_plan_receipt_content_sha256")):
        _fail(
            "the census plan receipt does not match the geometry analysis binding; the "
            "registries this selection reads are not the sealed ones"
        )
    lineage = crossing_manifest.get("lineage")
    if not isinstance(lineage, Mapping):
        _fail("crossing plan manifest carries no lineage")
    if str(lineage.get("census_plan_receipt_content_sha256")) != census_seal:
        _fail(
            "the crossing plan and the geometry analysis disagree about the census plan "
            "receipt; the two predecessor chains do not share one census"
        )
    census_inputs = lineage.get("census_input_files")
    if not isinstance(census_inputs, Mapping):
        _fail("crossing plan lineage carries no census_input_files")
    census_outputs = census_receipt.get("output_file_digests")
    if not isinstance(census_outputs, Mapping):
        _fail("census plan receipt carries no output_file_digests")

    census_files: dict[str, list[dict[str, Any]]] = {}
    for name in CENSUS_PLAN_FILES:
        sealed = _declared_digest(census_outputs, name, label="census plan receipt")
        lineage_entry = census_inputs.get(f"plan/{name}")
        if not isinstance(lineage_entry, Mapping) or str(lineage_entry.get("sha256")) != sealed:
            _fail(
                f"census plan/{name} is not sealed identically by the census receipt and the "
                "crossing plan lineage; the registry this selection reads is unproven"
            )
        path = census_dir / name
        file_sha256[str(path)] = _assert_file_digest(
            path, expected=sealed, label=f"census {name}"
        )
        census_files[name] = _read_jsonl(path, f"census {name}")

    owners_by_id = _index_unique(
        census_files[CENSUS_OWNER_REGISTRY_NAME], "gt_owner_id", label="census owner registry"
    )
    images_by_id = _index_unique(
        census_files[CENSUS_IMAGE_REGISTRY_NAME], "image_id", label="census image registry"
    )
    categories_by_query_id = _index_unique(
        census_files[CENSUS_CATEGORY_REGISTRY_NAME],
        "category_query_id",
        label="census category registry",
    )
    contexts_by_id = _index_unique(
        census_files[CENSUS_CONTEXT_REGISTRY_NAME],
        "context_id",
        label="census context registry",
    )

    exact_anchor_by_owner: dict[str, dict[str, Any]] = {}
    for row in census_files[CENSUS_CANDIDATE_BANK_NAME]:
        for generator in row.get("generators") or ():
            if not isinstance(generator, Mapping):
                continue
            if generator.get("logical_transform_role") != "exact_gt_anchor":
                continue
            owner_id = str(generator.get("generator_gt_owner_id"))
            if owner_id in exact_anchor_by_owner:
                _fail(
                    f"owner {owner_id!r} has more than one exact_gt_anchor candidate in the "
                    "sealed candidate bank"
                )
            exact_anchor_by_owner[owner_id] = dict(row)

    matched_rows_by_owner: dict[str, list[dict[str, Any]]] = {}
    for row in census_files[CENSUS_SIDECAR_REGISTRY_NAME]:
        if str(row.get("strict_match_status")) != crossing_plan.SIDECAR_MATCHED:
            continue
        owner_id = row.get("strict_match_gt_owner_id")
        if not isinstance(owner_id, str) or not owner_id:
            _fail(
                "a strict-matched native sidecar row names no physical owner; the coverage "
                "predicate cannot be evaluated"
            )
        matched_rows_by_owner.setdefault(owner_id, []).append(dict(row))

    _assert_no_prohibited_image(
        [str(row.get("image_id")) for row in cohort_rows], label="the crossing cohort registry"
    )
    _assert_no_prohibited_image(sorted(images_by_id), label="the census image registry")

    return SealedInputs(
        geometry_dir=geometry_dir,
        geometry_receipt=geometry_receipt,
        geometry_receipt_content_sha256=geometry_seal,
        geometry_rows=tuple(geometry_rows),
        crossing_plan_dir=crossing_plan_dir,
        crossing_manifest=crossing_manifest,
        crossing_manifest_content_sha256=crossing_seal,
        cohort_by_owner=cohort_by_owner,
        benign_controls=benign_controls,
        secondary_merged_dir=secondary_merged_dir,
        secondary_merge_receipt_content_sha256=merge_seal,
        merged_by_key=merged_by_key,
        census_plan_dir=census_dir,
        census_receipt_content_sha256=census_seal,
        owners_by_id=owners_by_id,
        images_by_id=images_by_id,
        categories_by_query_id=categories_by_query_id,
        exact_anchor_by_owner=exact_anchor_by_owner,
        matched_rows_by_owner=matched_rows_by_owner,
        contexts_by_id=contexts_by_id,
        file_sha256=dict(sorted(file_sha256.items())),
    )


# ---------------------------------------------------------------------------
# 2. The frozen material / nonmaterial cohort label
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortLabel:
    """The only historical likelihood fact selection may see: material or not."""

    gt_owner_id: str
    image_id: str
    material_negative: bool
    clearly_separated: bool


def read_cohort_labels(inputs: SealedInputs) -> dict[str, CohortLabel]:
    """Read the frozen ``p_plus_c_then_e`` material/nonmaterial split.

    Only the boolean label and the clearly-separated stratum are taken.  The
    per-owner relative delta is deliberately *not* returned here: it is a gate
    reference, sealed separately, and may never reach candidate admission.
    """

    labels: dict[str, CohortLabel] = {}
    for row in inputs.geometry_rows:
        if str(row.get("arm")) != ARM_CLEAN_REPLAY:
            continue
        if str(row.get("unit_id")) != GEOMETRY_UNIT_ID:
            _fail("the sealed geometry rows carry a row from another unit")
        owner_id = str(row.get("gt_owner_id"))
        if owner_id in labels:
            _fail(f"the sealed geometry rows carry duplicate primary-arm owner {owner_id!r}")
        if row.get("enters_primary_decision") is not True:
            _fail(
                f"geometry owner {owner_id!r} does not enter the primary decision; the frozen "
                "cohort label is not readable"
            )
        material = row.get("material_negative")
        if not isinstance(material, bool):
            _fail(f"geometry owner {owner_id!r} carries no boolean material_negative label")
        geometry_block = row.get("geometry")
        if not isinstance(geometry_block, Mapping):
            _fail(f"geometry owner {owner_id!r} carries no geometry block")
        separated = geometry_block.get("clearly_separated")
        if not isinstance(separated, bool):
            _fail(f"geometry owner {owner_id!r} carries no boolean clearly_separated label")
        labels[owner_id] = CohortLabel(
            gt_owner_id=owner_id,
            image_id=str(row.get("image_id")),
            material_negative=material,
            clearly_separated=separated,
        )
    if len(labels) != CROSSING_OWNER_COUNT:
        _fail(
            f"the sealed geometry analysis carries {len(labels)} primary-arm owners, not the "
            f"frozen {CROSSING_OWNER_COUNT}"
        )
    return labels


# ---------------------------------------------------------------------------
# 3. Deterministic neutral-row selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetBinding:
    """One crossing owner's sealed ``C``, ``P`` and ``E`` identities."""

    gt_owner_id: str
    image_id: str
    normalized_description: str
    boundary_index_b: int
    p_context_id: str
    c_token_ids: tuple[int, ...]
    c_token_count: int
    c_coord_token_ids: tuple[int, ...]
    c_box: geometry.Box
    e_row: Mapping[str, Any]
    e_description: str
    e_strict_match_gt_owner_id: str | None
    e_box: geometry.Box


def bind_target(inputs: SealedInputs, gt_owner_id: str) -> TargetBinding:
    """Re-verify one crossing owner's sealed ``C`` / ``P`` / ``E`` binding."""

    row = inputs.cohort_by_owner.get(gt_owner_id)
    if row is None:
        _fail(f"owner {gt_owner_id!r} is absent from the sealed crossing cohort registry")
    if str(row.get("cohort")) != crossing_plan.PRIMARY_COHORT:
        _fail(f"owner {gt_owner_id!r} is not a primary crossing-cohort row")
    image_id = str(row.get("image_id"))
    crossing = row.get("crossing")
    if not isinstance(crossing, Mapping):
        _fail(f"owner {gt_owner_id!r} carries no crossing block")
    boundary_index = int(crossing["boundary_index_b"])
    p_context_id = str(crossing["p_context_id"])
    if p_context_id != context_id_for(image_id, boundary_index):
        _fail(
            f"owner {gt_owner_id!r} declares P context {p_context_id!r}, which is not boundary "
            f"{boundary_index} of image {image_id!r}"
        )
    if p_context_id not in inputs.contexts_by_id:
        _fail(f"context {p_context_id!r} is absent from the sealed census context registry")

    e_row = row.get("e_row")
    if not isinstance(e_row, Mapping):
        _fail(f"owner {gt_owner_id!r} carries no sealed E row")
    if int(e_row["row_index"]) != boundary_index:
        _fail(
            f"owner {gt_owner_id!r} seals E at native row {e_row['row_index']!r} but crosses at "
            f"boundary {boundary_index}; E must be the native row the boundary emits"
        )
    if str(e_row.get("pre_row_context_id")) != p_context_id:
        _fail(
            f"owner {gt_owner_id!r} seals an E row whose pre-row context "
            f"{e_row.get('pre_row_context_id')!r} is not the crossing boundary {p_context_id!r}"
        )
    e_tokens = _token_ids(e_row.get("full_row_token_ids"), label=f"owner {gt_owner_id!r} E row")
    if sha256_json(e_tokens) != str(e_row.get("full_row_token_ids_sha256")):
        _fail(f"owner {gt_owner_id!r} sealed E row does not reconstruct its own token digest")

    inserted = row.get("inserted_clean_row_c")
    if not isinstance(inserted, Mapping):
        _fail(f"owner {gt_owner_id!r} carries no sealed inserted clean row C")
    c_tokens = _token_ids(inserted.get("token_ids"), label=f"owner {gt_owner_id!r} C row")
    if sha256_json(c_tokens) != str(inserted.get("token_ids_sha256")):
        _fail(f"owner {gt_owner_id!r} sealed C row does not reconstruct its own token digest")
    if int(inserted.get("token_count", -1)) != len(c_tokens):
        _fail(f"owner {gt_owner_id!r} sealed C row token count disagrees with its token ids")
    c_coords = _token_ids(
        inserted.get("coord_token_ids"), label=f"owner {gt_owner_id!r} C coordinates"
    )

    strict_owner = e_row.get("strict_match_gt_owner_id")
    if strict_owner is not None and not isinstance(strict_owner, str):
        _fail(f"owner {gt_owner_id!r} sealed E row carries a non-string strict-match owner")
    return TargetBinding(
        gt_owner_id=gt_owner_id,
        image_id=image_id,
        normalized_description=str(row["normalized_description"]),
        boundary_index_b=boundary_index,
        p_context_id=p_context_id,
        c_token_ids=tuple(c_tokens),
        c_token_count=len(c_tokens),
        c_coord_token_ids=tuple(c_coords),
        c_box=_decode_box(c_coords, label=f"owner {gt_owner_id!r} C row"),
        e_row=e_row,
        e_description=str(e_row["normalized_description"]),
        e_strict_match_gt_owner_id=strict_owner,
        e_box=_decode_box(
            _token_ids(e_row.get("coord_token_ids"), label=f"owner {gt_owner_id!r} E coordinates"),
            label=f"owner {gt_owner_id!r} E row",
        ),
    )


def _neutral_row_tokens(
    inputs: SealedInputs, *, image_id: str, normalized_description: str, coord_token_ids: Sequence[int]
) -> dict[str, Any] | None:
    """Assemble ``N`` from sealed tokens, or ``None`` when it is unassemblable."""

    category = inputs.categories_by_query_id.get(f"{image_id}:{normalized_description}")
    if category is None or str(category.get("status")) != "admitted":
        return None
    suffix = _token_ids(
        category.get("query_suffix_token_ids"),
        label=f"category {image_id}:{normalized_description} query suffix",
    )
    if sha256_json(suffix) != str(category.get("query_suffix_token_ids_sha256")):
        _fail(
            f"category {image_id}:{normalized_description} query suffix does not reconstruct its "
            "own sealed digest"
        )
    if not suffix or suffix[-1] != crossing_scorer.BOX_START:
        _fail(
            f"category {image_id}:{normalized_description} query suffix does not close with "
            "<|box_start|>; N cannot be assembled without retokenization"
        )
    token_ids = [*suffix, *[int(value) for value in coord_token_ids], crossing_scorer.BOX_END]
    return {
        "category_query_id": str(category["category_query_id"]),
        "query_suffix_token_ids": suffix,
        "query_suffix_token_ids_sha256": str(category["query_suffix_token_ids_sha256"]),
        "token_ids": token_ids,
        "token_ids_sha256": sha256_json(token_ids),
        "token_count": len(token_ids),
        "token_source": (
            "sealed_category_query_suffix_plus_sealed_exact_gt_anchor_coordinates_plus_box_end"
        ),
        "retokenized": False,
    }


@dataclass(frozen=True)
class Candidate:
    """One admitted neutral-row candidate and its total ordering key."""

    gt_owner_id: str
    normalized_description: str
    row_length_delta_tokens: int
    matched_native_row_index: int
    rows_back_distance: int
    center_distance_to_c: float
    center_distance_to_e: float
    min_center_distance: float
    relation_to_c: dict[str, Any]
    relation_to_e: dict[str, Any]
    assembled: dict[str, Any]
    coord_token_ids: tuple[int, ...]
    box_norm1000_xyxy: tuple[int, int, int, int]

    @property
    def order_key(self) -> tuple[int, int, float, str]:
        return (
            abs(self.row_length_delta_tokens),
            -self.matched_native_row_index,
            -self.min_center_distance,
            self.gt_owner_id,
        )


def evaluate_candidate(
    inputs: SealedInputs, target: TargetBinding, candidate_owner_id: str
) -> tuple[Candidate | None, str | None]:
    """Apply :data:`PREDICATE_ORDER` in that exact order to one same-image owner.

    Returns ``(candidate, None)`` when every predicate holds, otherwise
    ``(None, first_failed_predicate)``.  Assemblability is evaluated where it is
    declared -- immediately before the row-length predicate it makes measurable
    -- so a refusal reason is always the one that actually stopped the
    candidate.  No score, rank, margin or historical per-owner delta is read.
    """

    owner = inputs.owners_by_id.get(candidate_owner_id)
    if owner is None:
        return None, PREDICATE_SAME_IMAGE_CENSUS_OWNER
    if str(owner.get("image_id")) != target.image_id:
        return None, PREDICATE_SAME_IMAGE_CENSUS_OWNER
    if bool(owner.get("excluded_from_census")):
        return None, PREDICATE_SAME_IMAGE_CENSUS_OWNER

    if candidate_owner_id == target.gt_owner_id:
        return None, PREDICATE_NOT_C_AND_NOT_E_OWNER
    if (
        target.e_strict_match_gt_owner_id is not None
        and candidate_owner_id == target.e_strict_match_gt_owner_id
    ):
        return None, PREDICATE_NOT_C_AND_NOT_E_OWNER

    covered = [
        row
        for row in inputs.matched_rows_by_owner.get(candidate_owner_id, ())
        if str(row.get("image_id")) == target.image_id
        and int(row["row_index"]) < target.boundary_index_b
    ]
    if not covered:
        return None, PREDICATE_COVERED_BEFORE_P
    matched_row_index = max(int(row["row_index"]) for row in covered)

    description = str(owner.get("normalized_description"))
    if description == target.normalized_description or description == target.e_description:
        return None, PREDICATE_DIFFERENT_DESCRIPTION

    anchor = inputs.exact_anchor_by_owner.get(candidate_owner_id)
    if anchor is None:
        return None, PREDICATE_GEOMETRICALLY_SEPARATED
    coord_tokens = _token_ids(
        anchor.get("coord_token_ids"), label=f"owner {candidate_owner_id!r} exact anchor"
    )
    if sha256_json(coord_tokens) != str(anchor.get("coord_token_ids_sha256")):
        _fail(
            f"owner {candidate_owner_id!r} exact anchor coordinates do not reconstruct their own "
            "sealed digest"
        )
    n_box = _decode_box(coord_tokens, label=f"owner {candidate_owner_id!r} exact anchor")
    relation_to_c = _box_geometry(
        target.c_box, n_box, label=f"{candidate_owner_id!r} versus C {target.gt_owner_id!r}"
    )
    relation_to_e = _box_geometry(
        target.e_box, n_box, label=f"{candidate_owner_id!r} versus E of {target.gt_owner_id!r}"
    )
    if not (relation_to_c["clearly_separated"] and relation_to_e["clearly_separated"]):
        return None, PREDICATE_GEOMETRICALLY_SEPARATED

    assembled = _neutral_row_tokens(
        inputs,
        image_id=target.image_id,
        normalized_description=description,
        coord_token_ids=coord_tokens,
    )
    if assembled is None:
        return None, PREDICATE_ASSEMBLABLE
    row_length_delta = int(assembled["token_count"]) - target.c_token_count
    if abs(row_length_delta) > MAX_ROW_LENGTH_DELTA_TOKENS:
        return None, PREDICATE_MATCHED_ROW_LENGTH

    distance_to_c = float(relation_to_c["center_distance_normalized"])
    distance_to_e = float(relation_to_e["center_distance_normalized"])
    return (
        Candidate(
            gt_owner_id=candidate_owner_id,
            normalized_description=description,
            row_length_delta_tokens=row_length_delta,
            matched_native_row_index=matched_row_index,
            rows_back_distance=target.boundary_index_b - matched_row_index,
            center_distance_to_c=distance_to_c,
            center_distance_to_e=distance_to_e,
            min_center_distance=min(distance_to_c, distance_to_e),
            relation_to_c=relation_to_c,
            relation_to_e=relation_to_e,
            assembled=assembled,
            coord_token_ids=tuple(coord_tokens),
            box_norm1000_xyxy=(n_box.x1, n_box.y1, n_box.x2, n_box.y2),
        ),
        None,
    )


def select_neutral_row(
    inputs: SealedInputs, target: TargetBinding
) -> tuple[Candidate | None, list[dict[str, Any]]]:
    """The deterministic first candidate under the frozen total order.

    Every same-image census owner is evaluated and ledgered, admitted or not, so
    an infeasible target is visibly infeasible rather than silently dropped.
    """

    ledger: list[dict[str, Any]] = []
    admitted: list[Candidate] = []
    same_image = sorted(
        owner_id
        for owner_id, owner in inputs.owners_by_id.items()
        if str(owner.get("image_id")) == target.image_id
    )
    for owner_id in same_image:
        candidate, failed = evaluate_candidate(inputs, target, owner_id)
        if candidate is None:
            ledger.append(
                {
                    "gt_owner_id": owner_id,
                    "admitted": False,
                    "first_failed_predicate": failed,
                }
            )
            continue
        admitted.append(candidate)
        ledger.append(
            {
                "gt_owner_id": owner_id,
                "admitted": True,
                "first_failed_predicate": None,
                "row_length_delta_tokens": candidate.row_length_delta_tokens,
                "matched_native_row_index": candidate.matched_native_row_index,
                "min_center_distance_normalized": candidate.min_center_distance,
            }
        )
    if not admitted:
        return None, ledger
    admitted.sort(key=lambda item: item.order_key)
    return admitted[0], ledger


# ---------------------------------------------------------------------------
# 4. Sealed gate references
# ---------------------------------------------------------------------------


def _coordinate_reference(row: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """The sealed coordinate readout of one predecessor paired row."""

    deltas = row.get("deltas")
    roots = row.get("roots")
    if not isinstance(deltas, Mapping) or not isinstance(roots, Mapping):
        _fail(f"{label} carries no sealed deltas/roots block")
    coordinates = deltas.get("coordinates")
    if not isinstance(coordinates, Mapping):
        _fail(f"{label} carries no sealed coordinate delta")
    baseline = roots.get("baseline_unmodified_native_root")
    modified = roots.get("modified_inserted_clean_row_c_root")
    if not isinstance(baseline, Mapping) or not isinstance(modified, Mapping):
        _fail(f"{label} carries no sealed paired roots")
    baseline_sums = baseline.get("segment_sums")
    modified_sums = modified.get("segment_sums")
    if not isinstance(baseline_sums, Mapping) or not isinstance(modified_sums, Mapping):
        _fail(f"{label} carries no sealed segment sums")
    return {
        "coordinate_delta": float(coordinates["delta"]),
        "coordinate_delta_sign": int(coordinates["sign"]),
        "baseline_coordinate_sum": float(baseline_sums["coordinates"]["sum"]),
        "modified_coordinate_sum": float(modified_sums["coordinates"]["sum"]),
        # The compared argmax surface gate 2 replays against.  Sealed as a digest
        # of the literal argmax token ids so a replay cannot agree numerically
        # while decoding a different row.
        "baseline_argmax_token_ids_sha256": sha256_json(
            _token_ids(baseline.get("argmax_token_ids"), label=f"{label} baseline argmax")
        ),
        "modified_argmax_token_ids_sha256": sha256_json(
            _token_ids(modified.get("argmax_token_ids"), label=f"{label} modified argmax")
        ),
        "baseline_argmax_reproduces_description_path": bool(
            baseline.get("argmax_reproduces_description_path")
        ),
        "baseline_argmax_reproduces_complete_row": bool(
            baseline.get("argmax_reproduces_complete_row")
        ),
        "scored_token_ids_sha256": str(row["scored_token_ids_sha256"]),
        "scored_token_count": int(row["scored_token_count"]),
        "request_id": str(row["request_id"]),
    }


def bind_clean_reference(inputs: SealedInputs, gt_owner_id: str, label: CohortLabel) -> dict[str, Any]:
    """The sealed ``P+C -> E`` gate reference for one crossing owner."""

    row = inputs.merged_by_key.get((gt_owner_id, ARM_CLEAN_REPLAY))
    if row is None:
        _fail(
            f"owner {gt_owner_id!r} has no sealed {ARM_CLEAN_REPLAY!r} readout; its replay gate "
            "reference cannot be bound"
        )
    reference = _coordinate_reference(row, label=f"sealed C readout of {gt_owner_id!r}")
    geometry_row = next(
        (
            item
            for item in inputs.geometry_rows
            if str(item.get("gt_owner_id")) == gt_owner_id
            and str(item.get("arm")) == ARM_CLEAN_REPLAY
        ),
        None,
    )
    if geometry_row is None:  # pragma: no cover - guarded by read_cohort_labels
        _fail(f"owner {gt_owner_id!r} has no sealed geometry row")
    relative = float(geometry_row["relative_coordinate_delta"])
    if (relative <= MATERIALITY_MAX_NATS) is not label.material_negative:
        _fail(
            f"owner {gt_owner_id!r} sealed relative coordinate delta {relative!r} disagrees with "
            f"its sealed material_negative label at the frozen {MATERIALITY_MAX_NATS} nat cutoff"
        )
    return {
        **reference,
        "relative_coordinate_delta": relative,
        "material_negative": label.material_negative,
        "clearly_separated": label.clearly_separated,
        "role": "gate_reference_only_never_the_estimand",
        "materiality_cutoff_nats": MATERIALITY_MAX_NATS,
        "replay_max_coordinate_delta_abs_diff": REPLAY_MAX_COORDINATE_DELTA_ABS_DIFF,
        "replay_max_selected_logit_abs_diff": REPLAY_MAX_SELECTED_LOGIT_ABS_DIFF,
    }


def bind_benign_reference(inputs: SealedInputs, gt_owner_id: str) -> dict[str, Any]:
    """The sealed benign-substitution gate reference for one image."""

    row = inputs.merged_by_key.get((gt_owner_id, ARM_BENIGN))
    if row is None:
        _fail(
            f"benign control {gt_owner_id!r} has no sealed {ARM_BENIGN!r} readout; its replay "
            "gate reference cannot be bound"
        )
    reference = _coordinate_reference(row, label=f"sealed benign readout of {gt_owner_id!r}")
    return {
        **reference,
        "material_negative": reference["coordinate_delta"] <= MATERIALITY_MAX_NATS,
        "role": "gate_reference_only_the_same_run_replay_owns_the_estimand",
        "materiality_cutoff_nats": MATERIALITY_MAX_NATS,
        "replay_max_coordinate_delta_abs_diff": REPLAY_MAX_COORDINATE_DELTA_ABS_DIFF,
        "replay_max_selected_logit_abs_diff": REPLAY_MAX_SELECTED_LOGIT_ABS_DIFF,
    }


# ---------------------------------------------------------------------------
# 5. Registries and requests
# ---------------------------------------------------------------------------


def _context_identity(inputs: SealedInputs, context_id: str) -> dict[str, Any]:
    context = inputs.contexts_by_id.get(context_id)
    if context is None:
        _fail(f"context {context_id!r} is absent from the sealed census context registry")
    tokens = _token_ids(
        context.get("generated_prefix_token_ids"), label=f"context {context_id!r}"
    )
    if sha256_json(tokens) != str(context.get("generated_prefix_token_ids_sha256")):
        _fail(f"context {context_id!r} prefix tokens do not reconstruct their sealed digest")
    return {
        "context_id": context_id,
        "image_id": str(context["image_id"]),
        "boundary_index": int(context["boundary_index"]),
        "context_role": str(context.get("context_role")),
        "base_prefix_token_count": len(tokens),
        "base_prefix_token_ids_sha256": str(context["generated_prefix_token_ids_sha256"]),
    }


def build_selection_rows(
    inputs: SealedInputs, labels: Mapping[str, CohortLabel]
) -> list[dict[str, Any]]:
    """One sealed registry row per crossing owner, feasible or ledgered."""

    rows: list[dict[str, Any]] = []
    for gt_owner_id in sorted(labels):
        label = labels[gt_owner_id]
        target = bind_target(inputs, gt_owner_id)
        if target.image_id != label.image_id:
            _fail(
                f"owner {gt_owner_id!r} is registered on image {target.image_id!r} by the "
                f"crossing plan and {label.image_id!r} by the geometry analysis"
            )
        candidate, ledger = select_neutral_row(inputs, target)
        feasible = candidate is not None
        if label.material_negative:
            cohort_role = COHORT_VOTING
        elif feasible:
            cohort_role = COHORT_SPECIFICITY
        else:
            cohort_role = COHORT_INFEASIBLE
        if label.material_negative and not feasible:
            _fail(
                f"voting owner {gt_owner_id!r} has no admissible neutral row after the norm-1000 "
                "recomputation; unit.md stops preparation for review rather than relaxing a "
                "predicate"
            )
        eliminated = Counter(
            str(entry["first_failed_predicate"])
            for entry in ledger
            if not entry["admitted"]
        )
        rows.append(
            {
                "schema_version": SELECTION_SCHEMA_VERSION,
                "row_kind": "neutral_row_selection",
                "unit_id": UNIT_ID,
                "gt_owner_id": gt_owner_id,
                "image_id": target.image_id,
                "normalized_description": target.normalized_description,
                "cohort_role": cohort_role,
                "votes": cohort_role == COHORT_VOTING,
                "executed": feasible,
                "material_negative": label.material_negative,
                "clearly_separated": label.clearly_separated,
                "sentinel_owner": gt_owner_id in SENTINEL_OWNER_IDS,
                "posthoc_support_extent_uncertain": (
                    gt_owner_id in POSTHOC_SUPPORT_UNCERTAIN_OWNER_IDS
                ),
                "crossing": {
                    "boundary_index_b": target.boundary_index_b,
                    "boundary_convention": (
                        "boundary b contains native rows < b; E is native row index b and is "
                        "scored from the same boundary every arm appends to"
                    ),
                    "p_context_id": target.p_context_id,
                    **{
                        key: value
                        for key, value in _context_identity(inputs, target.p_context_id).items()
                        if key not in {"context_id", "image_id"}
                    },
                },
                "inserted_clean_row_c": {
                    "token_ids": list(target.c_token_ids),
                    "token_ids_sha256": sha256_json(list(target.c_token_ids)),
                    "token_count": target.c_token_count,
                    "coord_token_ids": list(target.c_coord_token_ids),
                    "box_norm1000_xyxy": target.c_box.as_list(),
                    "token_source": "sealed_predecessor_inserted_clean_row_c_verbatim",
                    "retokenized": False,
                },
                "scored_e_row": {
                    "row_index": int(target.e_row["row_index"]),
                    "pred_row_id": str(target.e_row["pred_row_id"]),
                    "normalized_description": target.e_description,
                    "strict_match_status": str(target.e_row["strict_match_status"]),
                    "strict_match_gt_owner_id": target.e_strict_match_gt_owner_id,
                    "stratum": str(target.e_row.get("stratum")),
                    "token_ids": list(
                        _token_ids(target.e_row["full_row_token_ids"], label="E row")
                    ),
                    "token_ids_sha256": str(target.e_row["full_row_token_ids_sha256"]),
                    "token_count": int(target.e_row["full_row_token_count"]),
                    "coord_token_ids": list(
                        _token_ids(target.e_row["coord_token_ids"], label="E coordinates")
                    ),
                    "box_norm1000_xyxy": target.e_box.as_list(),
                    "pre_row_context_id": str(target.e_row["pre_row_context_id"]),
                    "post_row_context_id": str(target.e_row["post_row_context_id"]),
                    "unknown_neutral_when_unmatched": True,
                },
                "neutral_row_n": (
                    None
                    if candidate is None
                    else {
                        "gt_owner_id": candidate.gt_owner_id,
                        "normalized_description": candidate.normalized_description,
                        "category_query_id": candidate.assembled["category_query_id"],
                        "coord_token_ids": list(candidate.coord_token_ids),
                        "box_norm1000_xyxy": list(candidate.box_norm1000_xyxy),
                        "token_ids": list(candidate.assembled["token_ids"]),
                        "token_ids_sha256": candidate.assembled["token_ids_sha256"],
                        "token_count": candidate.assembled["token_count"],
                        "token_source": candidate.assembled["token_source"],
                        "retokenized": False,
                        "row_length_delta_tokens": candidate.row_length_delta_tokens,
                        "matched_native_row_index": candidate.matched_native_row_index,
                        "rows_back_distance": candidate.rows_back_distance,
                        "sorted_route_regression": {
                            "is_regression": True,
                            "rows_back": candidate.rows_back_distance,
                            "note": (
                                "N was already emitted at a native row before P, so re-inserting "
                                "it duplicates a row and moves the learned sorted route backwards"
                            ),
                        },
                        "center_distance_to_c_normalized": candidate.center_distance_to_c,
                        "center_distance_to_e_normalized": candidate.center_distance_to_e,
                        "min_center_distance_normalized": candidate.min_center_distance,
                        "geometry_relation_to_c": candidate.relation_to_c,
                        "geometry_relation_to_e": candidate.relation_to_e,
                        "order_key": list(candidate.order_key),
                    }
                ),
                "selection": {
                    "feasible": feasible,
                    "candidate_count": sum(1 for entry in ledger if entry["admitted"]),
                    "only_one_candidate": (
                        sum(1 for entry in ledger if entry["admitted"]) == 1
                    ),
                    "evaluated_owner_count": len(ledger),
                    "predicate_order": list(PREDICATE_ORDER),
                    "ordering_rule": list(ORDERING_RULE),
                    "eliminated_by_predicate": dict(sorted(eliminated.items())),
                    "candidate_ledger": ledger,
                    "uses_scores": False,
                    "uses_manual_preference": False,
                    "minimum_center_distance_floor": None,
                },
                "sealed_clean_reference": bind_clean_reference(inputs, gt_owner_id, label),
                "request_ids": [],
            }
        )
    return rows


def selection_row_strata(row: Mapping[str, Any]) -> dict[str, str]:
    """The two sealed descriptive strata of one published selection row.

    ``unit.md`` "Non-voting sensitivities" and the representative-smoke gate read
    the same two axes: whether the scored ``E`` row strict-matched a physical
    owner, and whether ``E`` realized ``C``'s own description.  The ``E``
    stratum is the value the crossing plan already sealed, re-proven here
    against that row's own strict-match status, so a drifted pair fails closed
    instead of silently relabelling a smoke or a sensitivity.
    """

    gt_owner_id = str(row.get("gt_owner_id"))
    e_row = row.get("scored_e_row")
    if not isinstance(e_row, Mapping):
        _fail(f"selection row {gt_owner_id!r} carries no scored_e_row")
    strict_status = str(e_row.get("strict_match_status"))
    expected = (
        E_STRATUM_MATCHED
        if strict_status == crossing_plan.SIDECAR_MATCHED
        else E_STRATUM_UNMATCHED
    )
    sealed = str(e_row.get("stratum"))
    if sealed != expected:
        _fail(
            f"selection row {gt_owner_id!r} seals E stratum {sealed!r} against strict-match "
            f"status {strict_status!r}; the sealed pair disagrees"
        )
    return {
        "e_stratum": expected,
        "description_relation": (
            DESCRIPTION_SAME
            if str(e_row.get("normalized_description"))
            == str(row.get("normalized_description"))
            else DESCRIPTION_DIFFERENT
        ),
    }


def build_benign_rows(inputs: SealedInputs) -> list[dict[str, Any]]:
    """The twelve frozen benign-substitution controls, replayed in this unit."""

    rows: list[dict[str, Any]] = []
    for control in sorted(inputs.benign_controls, key=lambda row: str(row["gt_owner_id"])):
        gt_owner_id = str(control["gt_owner_id"])
        image_id = str(control["image_id"])
        due_context_id = str(control["due_context_id"])
        following = control.get("following_native_action")
        if not isinstance(following, Mapping):
            _fail(f"benign control {gt_owner_id!r} seals no following native action")
        if str(following.get("kind")) != crossing_plan.NATIVE_ACTION_ROW:
            _fail(
                f"benign control {gt_owner_id!r} follows a {following.get('kind')!r} action; only "
                "a complete native row is a benign coordinate reference"
            )
        inserted = control.get("inserted_clean_row_c")
        if not isinstance(inserted, Mapping):
            _fail(f"benign control {gt_owner_id!r} seals no inserted clean GT twin")
        twin_tokens = _token_ids(
            inserted.get("token_ids"), label=f"benign control {gt_owner_id!r} twin"
        )
        if sha256_json(twin_tokens) != str(inserted.get("token_ids_sha256")):
            _fail(f"benign control {gt_owner_id!r} twin does not reconstruct its sealed digest")
        scored_tokens = _token_ids(
            following.get("token_ids"), label=f"benign control {gt_owner_id!r} scored row"
        )
        if sha256_json(scored_tokens) != str(following.get("token_ids_sha256")):
            _fail(
                f"benign control {gt_owner_id!r} scored row does not reconstruct its sealed digest"
            )
        rows.append(
            {
                "schema_version": BENIGN_SCHEMA_VERSION,
                "row_kind": "benign_reference_control",
                "unit_id": UNIT_ID,
                "cohort_role": COHORT_BENIGN,
                "gt_owner_id": gt_owner_id,
                "image_id": image_id,
                "normalized_description": str(control["normalized_description"]),
                "due_context_id": due_context_id,
                "replaced_native_row_index": int(control["row_index"]),
                "following_native_action": {
                    "context_id": str(following["context_id"]),
                    "native_row_index": int(following["row_index"]),
                    "token_ids": scored_tokens,
                    "token_ids_sha256": str(following["token_ids_sha256"]),
                    "token_count": len(scored_tokens),
                },
                "inserted_clean_row_c": {
                    "token_ids": twin_tokens,
                    "token_ids_sha256": str(inserted["token_ids_sha256"]),
                    "token_count": len(twin_tokens),
                    "token_source": "sealed_predecessor_inserted_clean_row_c_verbatim",
                    "retokenized": False,
                },
                "sealed_benign_reference": bind_benign_reference(inputs, gt_owner_id),
                "estimand_role": (
                    "the same-run replay of this pair is the image-referenced benign delta for "
                    "both the N and the C relative estimands"
                ),
                "request_ids": [],
            }
        )
    images = [str(row["image_id"]) for row in rows]
    if len(set(images)) != len(images):
        _fail("the benign-reference registry carries more than one control per image")
    if len(images) != BENIGN_CONTROL_COUNT:
        _fail(
            f"the benign-reference registry holds {len(images)} controls, not the frozen "
            f"{BENIGN_CONTROL_COUNT}"
        )
    return rows


def _make_request(
    *,
    arm: str,
    cohort_role: str,
    gt_owner_id: str,
    image_id: str,
    context: Mapping[str, Any],
    baseline_context_id: str,
    appended_token_ids: Sequence[int],
    scored_target: Mapping[str, Any],
    sealed_reference: Mapping[str, Any],
) -> dict[str, Any]:
    appended = [int(value) for value in appended_token_ids]
    family = REQUEST_FAMILY_BY_ARM[arm]
    identity = {
        "unit_id": UNIT_ID,
        "request_family": family,
        "arm": arm,
        "cohort_role": cohort_role,
        "gt_owner_id": gt_owner_id,
        "context_id": context["context_id"],
        "baseline_context_id": baseline_context_id,
        "appended_token_ids": appended,
        "base_prefix_token_ids_sha256": context["base_prefix_token_ids_sha256"],
        "scored_target": dict(scored_target),
    }
    digest = sha256_json(identity)
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "row_kind": "neutral_row_control_request",
        "unit_id": UNIT_ID,
        "request_id": f"req:{digest[:32]}",
        "request_key": f"{family}|{cohort_role}|{gt_owner_id}|{context['context_id']}|{arm}",
        "request_family": family,
        "arm": arm,
        "cohort_role": cohort_role,
        "gt_owner_id": gt_owner_id,
        "image_id": image_id,
        "context_id": str(context["context_id"]),
        "context_role": str(context["context_role"]),
        "boundary_index": int(context["boundary_index"]),
        "paired_roots": {
            "baseline_context_id": baseline_context_id,
            "modified_context_id": str(context["context_id"]),
            "orientation": "modified_minus_baseline",
        },
        "prefix": {
            "base_context_id": str(context["context_id"]),
            "base_prefix_token_count": int(context["base_prefix_token_count"]),
            "base_prefix_token_ids_sha256": str(context["base_prefix_token_ids_sha256"]),
            "appended_token_ids": appended,
            "appended_token_ids_sha256": sha256_json(appended),
            "appended_token_count": len(appended),
            "appended_role": APPENDED_ROLE_BY_ARM[arm],
            "retokenized": False,
        },
        "scored_target": dict(scored_target),
        "sealed_reference": dict(sealed_reference),
        "score_blind_plan": True,
        "inspects_new_model_logits": False,
        "identity_digest": digest,
    }


def build_requests(
    inputs: SealedInputs,
    selection_rows: Sequence[Mapping[str, Any]],
    benign_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Exactly 21 ``P`` versus ``P+N``, 21 ``P`` versus ``P+C`` and 12 benign pairs."""

    requests: list[dict[str, Any]] = []
    for row in selection_rows:
        if not bool(row["executed"]):
            continue
        gt_owner_id = str(row["gt_owner_id"])
        image_id = str(row["image_id"])
        cohort_role = str(row["cohort_role"])
        context = _context_identity(inputs, str(row["crossing"]["p_context_id"]))
        e_row = row["scored_e_row"]
        scored_target = {
            "kind": "exact_native_row",
            "token_ids": list(e_row["token_ids"]),
            "token_ids_sha256": str(e_row["token_ids_sha256"]),
            "native_row_index": int(e_row["row_index"]),
            "baseline_context_id": str(e_row["pre_row_context_id"]),
            "successor_context_id": str(e_row["post_row_context_id"]),
            "compare_against": "the same exact E row scored at the unmodified native P",
            "report": ["description_delta", "coordinate_delta", "complete_row_delta"],
            "primary_segment": "coordinates",
        }
        for arm, appended in (
            (ARM_NEUTRAL, row["neutral_row_n"]["token_ids"]),
            (ARM_CLEAN_REPLAY, row["inserted_clean_row_c"]["token_ids"]),
        ):
            requests.append(
                _make_request(
                    arm=arm,
                    cohort_role=cohort_role,
                    gt_owner_id=gt_owner_id,
                    image_id=image_id,
                    context=context,
                    baseline_context_id=str(e_row["pre_row_context_id"]),
                    appended_token_ids=appended,
                    scored_target=scored_target,
                    sealed_reference=row["sealed_clean_reference"],
                )
            )

    for row in benign_rows:
        gt_owner_id = str(row["gt_owner_id"])
        image_id = str(row["image_id"])
        context = _context_identity(inputs, str(row["due_context_id"]))
        following = row["following_native_action"]
        requests.append(
            _make_request(
                arm=ARM_BENIGN,
                cohort_role=COHORT_BENIGN,
                gt_owner_id=gt_owner_id,
                image_id=image_id,
                context=context,
                baseline_context_id=str(following["context_id"]),
                appended_token_ids=row["inserted_clean_row_c"]["token_ids"],
                scored_target={
                    "kind": "exact_native_row",
                    "token_ids": list(following["token_ids"]),
                    "token_ids_sha256": str(following["token_ids_sha256"]),
                    "native_row_index": int(following["native_row_index"]),
                    "baseline_context_id": str(following["context_id"]),
                    "successor_context_id": context_id_for(
                        image_id, int(following["native_row_index"]) + 1
                    ),
                    "compare_against": (
                        "the same exact following native row scored at the unmodified native "
                        "successor context"
                    ),
                    "report": ["description_delta", "coordinate_delta", "complete_row_delta"],
                    "primary_segment": "coordinates",
                },
                sealed_reference=row["sealed_benign_reference"],
            )
        )

    seen: dict[str, str] = {}
    for request in requests:
        request_id = str(request["request_id"])
        if request_id in seen:
            _fail(
                f"duplicate request identity {request_id!r} for {request['request_key']!r} and "
                f"{seen[request_id]!r}"
            )
        seen[request_id] = str(request["request_key"])
    return requests


def assert_scored_token_identity_across_arms(
    requests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The scored ``E`` tokens must be identical between the ``N`` and ``C`` arms.

    ``unit.md`` gate 1: a modified-minus-baseline contrast over two different
    target rows is meaningless, and the whole unit compares ``N`` against ``C``
    on one sealed row.
    """

    by_owner: dict[str, dict[str, Mapping[str, Any]]] = {}
    for request in requests:
        arm = str(request["arm"])
        if arm == ARM_BENIGN:
            continue
        by_owner.setdefault(str(request["gt_owner_id"]), {})[arm] = request["scored_target"]
    checked: list[str] = []
    for gt_owner_id, arms in sorted(by_owner.items()):
        neutral = arms.get(ARM_NEUTRAL)
        clean = arms.get(ARM_CLEAN_REPLAY)
        if neutral is None or clean is None:
            _fail(
                f"owner {gt_owner_id!r} does not carry both a neutral-row and a clean-replay "
                "request; the two arms would not be paired on one scored row"
            )
        if list(neutral["token_ids"]) != list(clean["token_ids"]):
            _fail(
                f"owner {gt_owner_id!r} scores different E tokens in the N and C arms; the "
                "contrast would be meaningless"
            )
        if str(neutral["token_ids_sha256"]) != str(clean["token_ids_sha256"]):
            _fail(f"owner {gt_owner_id!r} N and C arms declare different scored-token digests")
        if sha256_json(list(neutral["token_ids"])) != str(neutral["token_ids_sha256"]):
            _fail(f"owner {gt_owner_id!r} scored E tokens do not reconstruct their own digest")
        checked.append(gt_owner_id)
    return {
        "checked_owner_count": len(checked),
        "checked_owner_ids": checked,
        "identical_scored_e_tokens_across_arms": True,
    }


# ---------------------------------------------------------------------------
# 6. Denominators
# ---------------------------------------------------------------------------


def derive_cohort_counts(
    selection_rows: Sequence[Mapping[str, Any]],
    benign_rows: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Re-derive and enforce every frozen denominator before the plan is sealed."""

    by_role = Counter(str(row["cohort_role"]) for row in selection_rows)
    voting = sorted(str(row["gt_owner_id"]) for row in selection_rows if row["votes"])
    specificity = sorted(
        str(row["gt_owner_id"])
        for row in selection_rows
        if str(row["cohort_role"]) == COHORT_SPECIFICITY
    )
    infeasible = sorted(
        str(row["gt_owner_id"])
        for row in selection_rows
        if str(row["cohort_role"]) == COHORT_INFEASIBLE
    )
    separated = sorted(
        str(row["gt_owner_id"])
        for row in selection_rows
        if row["votes"] and bool(row["clearly_separated"])
    )
    for actual, expected, label in (
        (len(voting), VOTING_OWNER_COUNT, "the voting material skipped-owner set"),
        (len(specificity), SPECIFICITY_OWNER_COUNT, "the feasible specificity stratum"),
        (len(infeasible), INFEASIBLE_OWNER_COUNT, "the structural infeasibility ledger"),
        (len(selection_rows), CROSSING_OWNER_COUNT, "the crossing selection registry"),
    ):
        if actual != expected:
            _fail(
                f"{label} holds {actual} owners, not the frozen {expected}; the CPU plan owns "
                "this split and fails closed instead of relaxing a predicate"
            )
    if tuple(infeasible) != tuple(sorted(FROZEN_INFEASIBLE_OWNER_IDS)):
        _fail(
            f"the structural infeasibility ledger is {infeasible!r}, not the frozen "
            f"{sorted(FROZEN_INFEASIBLE_OWNER_IDS)!r}"
        )
    if tuple(separated) != tuple(sorted(FROZEN_CLEARLY_SEPARATED_OWNER_IDS)):
        _fail(
            f"the clearly separated voting subset is {separated!r}, not the frozen "
            f"{sorted(FROZEN_CLEARLY_SEPARATED_OWNER_IDS)!r}"
        )
    missing_sentinels = sorted(set(SENTINEL_OWNER_IDS) - set(voting))
    if missing_sentinels:
        _fail(
            f"sentinel owner(s) {missing_sentinels!r} are not in the voting set; the mandatory "
            "positive-control sentinels would never be checked"
        )

    executed = [row for row in selection_rows if bool(row["executed"])]
    if len(executed) != EXECUTED_OWNER_COUNT:
        _fail(
            f"{len(executed)} owners are executed, not the frozen {EXECUTED_OWNER_COUNT} "
            f"({VOTING_OWNER_COUNT} voting + {SPECIFICITY_OWNER_COUNT} specificity)"
        )
    by_arm = Counter(str(request["arm"]) for request in requests)
    for arm in ARMS:
        expected = EXPECTED_REQUEST_COUNT_BY_ARM[arm]
        if by_arm.get(arm, 0) != expected:
            _fail(
                f"the request plan holds {by_arm.get(arm, 0)} {arm!r} requests, not the frozen "
                f"{expected}"
            )
    if len(requests) != TOTAL_REQUEST_COUNT:
        _fail(
            f"the request plan holds {len(requests)} requests, not the frozen "
            f"{TOTAL_REQUEST_COUNT}"
        )

    images = sorted({str(row["image_id"]) for row in selection_rows})
    _assert_no_prohibited_image(images, label="the neutral-row selection registry")
    if len(images) != IMAGE_COUNT:
        _fail(f"the selection registry spans {len(images)} images, not the frozen {IMAGE_COUNT}")
    benign_images = sorted({str(row["image_id"]) for row in benign_rows})
    if benign_images != images:
        _fail(
            "the benign-reference registry does not cover exactly the selection registry's "
            "images; an image-referenced estimand would have no same-run reference"
        )
    executed_images = [str(row["image_id"]) for row in executed]
    uncovered = sorted(set(executed_images) - set(benign_images))
    if uncovered:
        _fail(f"executed image(s) {uncovered!r} have no same-run benign reference")

    return {
        "crossing_owner_count": len(selection_rows),
        "voting_owner_count": len(voting),
        "voting_owner_ids": voting,
        "clearly_separated_voting_owner_ids": separated,
        "specificity_owner_count": len(specificity),
        "specificity_owner_ids": specificity,
        "infeasible_owner_count": len(infeasible),
        "infeasible_owner_ids": infeasible,
        "executed_owner_count": len(executed),
        "benign_control_count": len(benign_rows),
        "image_count": len(images),
        "image_ids": images,
        "per_image_executed_owner_counts": dict(sorted(Counter(executed_images).items())),
        "cohort_role_counts": dict(sorted(by_role.items())),
        "request_count_by_arm": dict(sorted(by_arm.items())),
        "request_count_total": len(requests),
        "expected": {
            "crossing_owner_count": CROSSING_OWNER_COUNT,
            "voting_owner_count": VOTING_OWNER_COUNT,
            "specificity_owner_count": SPECIFICITY_OWNER_COUNT,
            "infeasible_owner_count": INFEASIBLE_OWNER_COUNT,
            "executed_owner_count": EXECUTED_OWNER_COUNT,
            "benign_control_count": BENIGN_CONTROL_COUNT,
            "image_count": IMAGE_COUNT,
            "request_count_by_arm": dict(EXPECTED_REQUEST_COUNT_BY_ARM),
            "request_count_total": TOTAL_REQUEST_COUNT,
        },
    }


# ---------------------------------------------------------------------------
# 7. Emission
# ---------------------------------------------------------------------------


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _builder_source_seal() -> dict[str, Any]:
    path = Path(__file__).resolve()
    payload = path.read_bytes()
    try:
        relative_path = str(path.relative_to(REPO_ROOT))
    except ValueError:  # pragma: no cover - the module always lives under the repo
        relative_path = path.name
    return {"path": relative_path, "byte_size": len(payload), "sha256": sha256_bytes(payload)}


def _seal_files(file_sha256: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Re-read every input and seal ``{path, byte_size, sha256}`` from one observation."""

    seals: dict[str, dict[str, Any]] = {}
    for path_text, declared in sorted(file_sha256.items()):
        path = Path(path_text)
        if not path.is_file():
            _fail(f"input file {path} disappeared before it could be sealed")
        payload = path.read_bytes()
        digest = sha256_bytes(payload)
        if digest != declared:
            _fail(
                f"input file {path} changed on disk between validation and sealing; the lineage "
                "would not describe the bytes that were read"
            )
        seals[path_text] = {"path": path_text, "byte_size": len(payload), "sha256": digest}
    return seals


def build_plan(
    *,
    geometry_analysis_dir: Path,
    crossing_plan_dir: Path,
    secondary_merged_dir: Path,
    output_root: Path,
    census_plan_dir: Path | None = None,
) -> dict[str, Any]:
    """Derive, validate and seal the complete CPU plan.  Returns the manifest."""

    inputs = load_sealed_inputs(
        geometry_analysis_dir=geometry_analysis_dir,
        crossing_plan_dir=crossing_plan_dir,
        secondary_merged_dir=secondary_merged_dir,
        census_plan_dir=census_plan_dir,
    )
    labels = read_cohort_labels(inputs)
    selection_rows = build_selection_rows(inputs, labels)
    benign_rows = build_benign_rows(inputs)
    requests = build_requests(inputs, selection_rows, benign_rows)
    token_identity = assert_scored_token_identity_across_arms(requests)
    counts = derive_cohort_counts(selection_rows, benign_rows, requests)

    requests_by_owner: dict[str, list[str]] = {}
    for request in requests:
        requests_by_owner.setdefault(str(request["gt_owner_id"]), []).append(
            str(request["request_id"])
        )
    for row in list(selection_rows) + list(benign_rows):
        row["request_ids"] = sorted(requests_by_owner.get(str(row["gt_owner_id"]), ()))
        if bool(row.get("executed", True)) and not row["request_ids"]:
            _fail(f"owner {row['gt_owner_id']!r} is executed but carries no request")

    row_sets: dict[str, Sequence[Mapping[str, Any]]] = {
        SELECTION_REGISTRY_NAME: selection_rows,
        BENIGN_REGISTRY_NAME: benign_rows,
        REQUEST_PLAN_NAME: requests,
    }
    files = {name: _jsonl_bytes(rows) for name, rows in row_sets.items()}
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "builder_source": _builder_source_seal(),
        "lineage": {
            "seal_fields": list(SEAL_FIELDS),
            "geometry_analysis_dir": str(inputs.geometry_dir),
            "geometry_unit_id": GEOMETRY_UNIT_ID,
            "geometry_receipt_content_sha256": inputs.geometry_receipt_content_sha256,
            "crossing_plan_dir": str(inputs.crossing_plan_dir),
            "crossing_unit_id": CROSSING_UNIT_ID,
            "crossing_plan_manifest_content_sha256": inputs.crossing_manifest_content_sha256,
            "secondary_merged_dir": str(inputs.secondary_merged_dir),
            "secondary_merge_receipt_content_sha256": (
                inputs.secondary_merge_receipt_content_sha256
            ),
            "census_plan_dir": str(inputs.census_plan_dir),
            "census_unit_id": CENSUS_UNIT_ID,
            "census_plan_receipt_content_sha256": inputs.census_receipt_content_sha256,
            "input_files": _seal_files(inputs.file_sha256),
        },
        "cohort": counts,
        "selection_rule": {
            "predicate_order": list(PREDICATE_ORDER),
            "predicate_order_is_the_evaluation_order": True,
            "structural_prerequisite_predicates": list(STRUCTURAL_PREREQUISITE_PREDICATES),
            "ordering_rule": list(ORDERING_RULE),
            "max_row_length_delta_tokens": MAX_ROW_LENGTH_DELTA_TOKENS,
            "minimum_center_distance_floor": None,
            "reads_new_model_output": False,
            "reads_historical_per_owner_deltas": False,
            "reads_only_the_frozen_material_cohort_label": True,
            "alternate_neutral_row": None,
            "uncovered_row_arm": None,
            "position_gap_arm": None,
            "sampling": None,
            "manual_reselection": None,
        },
        "materiality": {
            "cutoff_nats": MATERIALITY_MAX_NATS,
            "cutoff_source": (
                "analyze_sorted_crossing_owner_row_geometry.MATERIAL_NEGATIVE_MAX_NATS"
            ),
            "estimand": (
                "relative_delta(i) = logprob(E coordinates | P+X) - logprob(E coordinates | P) "
                "- same_run_benign_coordinate_delta(image(i))"
            ),
            "primary_segment": "coordinates",
            "benign_reference_source": "same_run_replay_only",
            "sealed_values_role": "gate_reference_only_never_the_estimand",
        },
        "gates": {
            "order": [
                "input_and_token_identity",
                "runtime_replay",
                "same_run_positive_controls",
                "benign_reference_replay",
                "cached_versus_uncached_parity",
                "neutral_row_specificity",
            ],
            "replay_max_selected_logit_abs_diff": REPLAY_MAX_SELECTED_LOGIT_ABS_DIFF,
            "replay_max_coordinate_delta_abs_diff": REPLAY_MAX_COORDINATE_DELTA_ABS_DIFF,
            "max_quarantined_owners": MAX_QUARANTINED_OWNERS,
            "max_clean_replay_failures": MAX_CLEAN_REPLAY_FAILURES,
            "mandatory_sentinel_owner_ids": list(SENTINEL_OWNER_IDS),
            "specificity_material_max": SPECIFICITY_MATERIAL_MAX,
            "specificity_threshold_is_absolute": True,
            "quarantined_owner_satisfies_no_side": True,
            "representative_smoke_required_strata": list(REQUIRED_SMOKE_STRATA),
        },
        "routes": {
            "order": list(ROUTE_ORDER),
            "specificity_failure_route": ROUTE_NEUTRAL_NOT_NEUTRAL,
            ROUTE_C_CONTENT_SPECIFIC: {
                "min_nonmaterial_voting_owners": ROUTE_C_MIN_NONMATERIAL,
                "min_images": ROUTE_C_MIN_IMAGES,
                "min_clearly_separated": ROUTE_C_MIN_SEPARATED,
                "min_clearly_separated_images": ROUTE_C_MIN_SEPARATED_IMAGES,
            },
            ROUTE_GENERIC_SUFFICIENT: {
                "min_material_voting_owners": ROUTE_GENERIC_MIN_MATERIAL,
                "min_images": ROUTE_GENERIC_MIN_IMAGES,
                "min_clearly_separated": ROUTE_GENERIC_MIN_SEPARATED,
                "min_clearly_separated_images": ROUTE_GENERIC_MIN_SEPARATED_IMAGES,
                "requires_specificity_gate_pass": True,
            },
            "denominators_are_absolute_over_the_frozen_nine": True,
        },
        "non_voting_sensitivities": {
            "exclude_row_length_delta_two": True,
            "exclude_posthoc_support_uncertain_owner_ids": list(
                POSTHOC_SUPPORT_UNCERTAIN_OWNER_IDS
            ),
            "materiality_cutoffs_nats": list(SENSITIVITY_CUTOFFS_NATS),
            "strata": [*E_STRATUM_AXIS, *DESCRIPTION_AXIS, "visual_adjudication"],
            "role": "descriptive_recomputation_never_replaces_the_frozen_route",
        },
        "request_counts": {
            "total": len(requests),
            "by_arm": dict(sorted(Counter(str(r["arm"]) for r in requests).items())),
            "by_family": dict(
                sorted(Counter(str(r["request_family"]) for r in requests).items())
            ),
            "by_cohort_role": dict(
                sorted(Counter(str(r["cohort_role"]) for r in requests).items())
            ),
        },
        "token_identity_contract": {
            **token_identity,
            "object_ref_start": crossing_scorer.OBJECT_REF_START,
            "object_ref_end": crossing_scorer.OBJECT_REF_END,
            "box_start": crossing_scorer.BOX_START,
            "box_end": crossing_scorer.BOX_END,
            "coordinate_token_id_start": crossing_scorer.COORDINATE_TOKEN_ID_START,
            "coordinate_token_id_end_exclusive": (
                crossing_scorer.COORDINATE_TOKEN_ID_END_EXCLUSIVE
            ),
            "coordinate_token_count": crossing_scorer.COORDINATE_TOKEN_COUNT,
            "retokenization": "forbidden_every_token_id_is_literal",
            "neutral_row_shape": "sealed_query_suffix_then_four_coordinate_tokens_then_box_end",
        },
        "score_input_policy": {
            "loads_model_or_tokenizer": False,
            "inspects_new_model_logits": False,
            "selection_uses_scores": False,
            "reads": (
                "sealed_geometry_crossing_and_census_registries_plus_the_frozen_material_label"
            ),
        },
        "prohibited": {
            "image_ids": sorted(PROHIBITED_IMAGE_IDS),
            "free_decode": True,
            "greedy_coordinate_decode": True,
            "sampling": True,
            "rollout": True,
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "output_file_digests": {
            name: {
                "path": name,
                "byte_size": len(files[name]),
                "sha256": sha256_bytes(files[name]),
                "row_count": len(row_sets[name]),
            }
            for name in sorted(files)
        },
        "semantics_notes": list(SEMANTICS_NOTES),
    }
    manifest["manifest_content_sha256"] = sha256_json(manifest)

    plan_dir = Path(output_root) / PLAN_DIR_NAME
    crossing_scorer._publish(  # noqa: SLF001
        plan_dir,
        {**files, MANIFEST_NAME: canonical_json_bytes(manifest) + b"\n"},
    )
    return manifest


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry-analysis-dir",
        default=str(DEFAULT_GEOMETRY_ANALYSIS_DIR),
        help="immutable geometric-relation stratification analysis directory",
    )
    parser.add_argument(
        "--crossing-plan-dir",
        default=str(DEFAULT_CROSSING_PLAN_DIR),
        help="immutable crossing-boundary CPU plan directory",
    )
    parser.add_argument(
        "--secondary-merged-dir",
        default=str(DEFAULT_SECONDARY_MERGED_DIR),
        help="immutable merged secondary compatibility capture directory",
    )
    parser.add_argument(
        "--census-plan-dir",
        default=None,
        help="census plan directory; defaults to the geometry receipt's sealed census run root",
    )
    parser.add_argument("--output-root", required=True, help="local run directory to write")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_plan(
            geometry_analysis_dir=Path(args.geometry_analysis_dir),
            crossing_plan_dir=Path(args.crossing_plan_dir),
            secondary_merged_dir=Path(args.secondary_merged_dir),
            census_plan_dir=(
                None if args.census_plan_dir is None else Path(args.census_plan_dir)
            ),
            output_root=Path(args.output_root),
        )
    except (NeutralRowPlanContractError, geometry.GeometryContractError) as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        return 1
    cohort = manifest["cohort"]
    print(
        "sealed neutral-row CPU plan: "
        f"voting={cohort['voting_owner_count']} "
        f"specificity={cohort['specificity_owner_count']} "
        f"infeasible={cohort['infeasible_owner_count']} "
        f"benign={cohort['benign_control_count']} "
        f"requests={cohort['request_count_total']} "
        f"{cohort['request_count_by_arm']} "
        f"manifest={manifest['manifest_content_sha256'][:12]}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
