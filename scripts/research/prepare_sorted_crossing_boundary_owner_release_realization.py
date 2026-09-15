#!/usr/bin/env python3
"""CPU-only fail-closed plan builder for the sorted crossing-boundary owner
release/realization unit
(``2026-08-03-sorted-crossing-boundary-owner-release-realization``).

This module scores nothing, loads no model or tokenizer, and never inspects a
new model logit.  It re-reads two immutable predecessor runs -- the sealed
native-prefix reachability *prevalence* analysis and, through that analysis'
own receipt, the sorted owner accessibility phenotype *census* it was computed
from -- re-verifies every declared digest, and emits the sealed CPU cohort
registry, control registry, and request plan that a later GPU pass will
execute unchanged.  See
``docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/
2026-08-03-sorted-crossing-boundary-owner-release-realization/unit.md`` for the
frozen question, the exact ``P``/``E``/``P+E`` state pair, and the branch
truth table this planner prepares (but never evaluates).

The crossing boundary
---------------------
The predecessor boundary convention is authoritative: boundary index ``b``
contains native rows ``< b``.  For a target owner ``C``:

* ``P`` is the last root/ahead boundary ``b`` before the next complete native
  row moves ``C`` to ``passed_by_frontier``;
* ``E`` is native sidecar row index ``b``; and
* ``P+E`` is boundary ``b + 1``.

``F``, when present, is native sidecar row index ``b + 1`` and its post-``E``
boundary is ``b + 2``.

Three cohort denominators are re-derived here and must reproduce the sealed
census exactly or the build fails before a single request is constructed: the
U-bound favorable-top-three crossing cohort (``26``), the L-bound sensitivity
cohort (``25``), the owners favorable under both bounds at the same crossing
context (``24``), and the two preregistered native-row strata of the primary
cohort (``12`` strict-matched ``E`` rows, ``14`` unmatched).

Target-local support calibration
--------------------------------
Support is **not** a rank.  The frozen predecessor census decides target-local
support with two local-peak statistics and one sealed threshold pair, and this
planner seals that exact receipt rather than re-deriving or re-declaring it:

* the calibration artifact ``phases/discovery-sealed/support-calibration.json``
  is read, proven to reconstruct its own ``calibration_sha256``, proven to have
  been derived by the merge source this planner imports, and proven to be the
  *same* receipt the presentation phase applied to the owner summaries this
  planner consumes;
* the statistic formulas, the partition-to-bound map, the primary quantile, the
  epsilon and the conjunction rule are bound from the sealed
  ``plan/capture-rules.json -> owner_support`` block and the frozen census
  classifier, never authored here; and
* the emitted plan carries, per owner, the exact U and L exclusion-filtered
  member candidate IDs of the seventeen-role target-local family plus the
  unique ``(context, category)`` population, so the later scorer can reproduce
  ``peak_lift`` and ``local_concentration`` without consulting a rank, a
  margin, or any score this planner is forbidden to see.

Any drift in the calibration source, schema, thresholds, epsilon, quantile,
partition semantics, or population fails the build closed.

Exact token ownership
---------------------
No conclusion-bearing string is retokenized.  ``P`` and ``P+E`` carry the
literal ``generated_prefix_token_ids`` of the sealed context registry.  The
full ``E`` row token identity is owned *only* by the literal suffix
``tokens(P+E) - tokens(P)``; the native sidecar registry holds no full-row
token sequence and is used only to validate ``E``'s coordinate-token
subsequence, coordinate digest, and declared row metadata (row id/index,
description, strict-match fields, raw-span digest).  ``D_C`` is the sealed
category ``query_suffix_token_ids`` through ``<|box_start|>``; the inserted
clean row ``C`` is ``D_C`` plus the owner's sealed exact-GT-anchor coordinate
token IDs plus the exact ``<|box_end|>`` token.  Coordinate-only greedy has a
fixed grammar: exactly four coordinate-token IDs followed by ``<|box_end|>``.

What it publishes
-----------------
Under a CLI-supplied output root::

    plan/manifest.json
    plan/cohort-registry.jsonl
    plan/control-registry.jsonl
    plan/request-plan.jsonl

Every input and output file receives path, byte size, and SHA-256 lineage in
the manifest, and the manifest self-seals.  Nothing is ever written to either
predecessor run root.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import (  # noqa: E402
    analyze_sorted_supported_fn_native_prefix_reachability_prevalence as prevalence,
)
from scripts.research import build_sorted_owner_accessibility_census_plan as planner  # noqa: E402
from scripts.research import merge_sorted_owner_accessibility_census_shards as merge  # noqa: E402

UNIT_ID = "2026-08-03-sorted-crossing-boundary-owner-release-realization"
PREVALENCE_UNIT_ID = prevalence.UNIT_ID
CENSUS_UNIT_ID = merge.UNIT_ID

#: ``.v2`` replaced the scalar ``builder_source_sha256`` and the flat
#: ``*_input_file_sha256`` digest maps with complete
#: ``{path, byte_size, sha256}`` seals, so every conclusion-bearing input and
#: source identity carries path, byte size, and digest rather than a digest
#: alone.  No cohort, control, or request semantics changed with that bump.
#:
#: ``.v3`` seals the predecessor target-local support calibration receipt and
#: publishes the minimal explicit support-calibration contract the scorer needs
#: to reproduce U- and L-bound target-local support.  The cohort, control, and
#: request schemas bump alongside it because ``candidate_families`` gained the
#: per-bound partition membership those reproductions consume.  No cohort
#: denominator, control selection, literal candidate, or request family
#: changed: the sealed counts are still 26/25/24, 12/14, 14 + 12, and 500.
MANIFEST_SCHEMA_VERSION = "sorted-crossing-boundary-owner-release-realization-plan.v3"
COHORT_SCHEMA_VERSION = "sorted-crossing-boundary-owner-release-realization-cohort.v2"
CONTROL_SCHEMA_VERSION = "sorted-crossing-boundary-owner-release-realization-control.v2"
REQUEST_SCHEMA_VERSION = "sorted-crossing-boundary-owner-release-realization-request.v2"

#: Every conclusion-bearing input, source, and output identity carries these
#: three fields; a digest alone is not a complete seal.
SEAL_FIELDS: tuple[str, ...] = ("path", "byte_size", "sha256")

PLAN_DIR_NAME = "plan"
MANIFEST_NAME = "manifest.json"
COHORT_REGISTRY_NAME = "cohort-registry.jsonl"
CONTROL_REGISTRY_NAME = "control-registry.jsonl"
REQUEST_PLAN_NAME = "request-plan.jsonl"

#: The immutable prevalence run this unit is routed from.  It is only a
#: default: every path is explicitly overridable on the command line, and the
#: tests bind temporary fixtures instead.
DEFAULT_PREVALENCE_RUN_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-03-sorted-supported-fn-native-prefix-reachability-prevalence/20260803T231934Z"
)

PREVALENCE_RECEIPT_REL = "analysis/receipt.json"
PREVALENCE_REPORT_JSON_REL = "analysis/report.json"
PREVALENCE_REPORT_MD_REL = "analysis/report.md"
PREVALENCE_OWNER_RECORDS_REL = "analysis/owner-records.jsonl"
PREVALENCE_INPUT_FILES: tuple[str, ...] = (
    PREVALENCE_RECEIPT_REL,
    PREVALENCE_REPORT_JSON_REL,
    PREVALENCE_REPORT_MD_REL,
    PREVALENCE_OWNER_RECORDS_REL,
)

#: Census plan files this planner needs beyond the six the prevalence analysis
#: already binds.  Each is verified against ``plan/receipt.json``'s own sealed
#: ``output_file_digests``.
CATEGORY_REGISTRY_REL = "plan/category-registry.jsonl"
IMAGE_REGISTRY_REL = "plan/image-registry.jsonl"
OWNER_REGISTRY_REL = "plan/owner-registry.jsonl"
CANDIDATE_BANK_REL = "plan/candidate-bank.jsonl"
QUERY_GROUP_REGISTRY_REL = "plan/query-group-registry.jsonl"
CAPTURE_RULES_REL = "plan/capture-rules.json"
EXTRA_CENSUS_PLAN_FILES: tuple[str, ...] = (
    CATEGORY_REGISTRY_REL,
    IMAGE_REGISTRY_REL,
    OWNER_REGISTRY_REL,
    CANDIDATE_BANK_REL,
    QUERY_GROUP_REGISTRY_REL,
    CAPTURE_RULES_REL,
)

#: The sealed support-calibration receipt.  It lives under ``phases/`` rather
#: than ``plan/``, so it is not covered by ``plan/receipt.json``; its integrity
#: chain instead runs prevalence receipt -> presentation ``merge-receipt.json``
#: bytes -> that receipt's ``support_criterion`` block -> this artifact's own
#: ``calibration_sha256``.  Every link is proven before a threshold is used.
SUPPORT_CALIBRATION_REL = "phases/discovery-sealed/support-calibration.json"
EXTRA_CENSUS_PHASE_FILES: tuple[str, ...] = (SUPPORT_CALIBRATION_REL,)

#: Frozen denominators.  Every one of these is re-derived from the sealed
#: census and a mismatch fails the build before request construction.
EXPECTED_U_CROSSING_COUNT = 26
EXPECTED_L_CROSSING_COUNT = 25
EXPECTED_EXACT_UL_CROSSING_COUNT = 24
EXPECTED_MATCHED_E_COUNT = 12
EXPECTED_UNMATCHED_E_COUNT = 14
EXPECTED_TIMING_CONTROL_COUNT = 14
EXPECTED_TP_REPLAY_CONTROL_COUNT = 12

#: ``P`` must be a root/ahead boundary.  ``at_frontier`` is deliberately
#: excluded: an owner sorting exactly at the frontier has not been moved past
#: by the next native row, so it is not a crossing.
ROOT_OR_AHEAD_STATES: frozenset[str] = frozenset({"root_no_frontier", "ahead_of_frontier"})
PASSED_STATE = "passed_by_frontier"
BEFORE_OR_AT_FRONTIER_STATES = prevalence.BEFORE_OR_AT_FRONTIER_STATES

PRIMARY_COHORT = "u_bound_crossing_primary"
L_SENSITIVITY_COHORT = "l_bound_crossing_sensitivity"
TIMING_CONTROL_COHORT = "disjoint_timing_control"
TP_REPLAY_CONTROL_COHORT = "native_tp_replay_control"

STRATUM_MATCHED_E = "matched_e"
STRATUM_UNMATCHED_E = "unmatched_e"

#: The sealed census emits three sidecar match statuses.  Only ``matched`` is a
#: strict physical-owner match; ``ambiguous_neutral`` names an owner but is not
#: a strict match, so it stays in the preserved unmatched-E stratum and never
#: becomes a displacement claim.
SIDECAR_MATCHED = "matched"
SIDECAR_UNMATCHED = "unmatched"
SIDECAR_AMBIGUOUS_NEUTRAL = "ambiguous_neutral"
SIDECAR_MATCH_STATUSES: frozenset[str] = frozenset(
    {SIDECAR_MATCHED, SIDECAR_UNMATCHED, SIDECAR_AMBIGUOUS_NEUTRAL}
)

OBSERVABILITY_SAME_DESCRIPTION = "same_description_coordinate_only"
OBSERVABILITY_DIFFERENT_DESCRIPTION = "different_description_release_observable"

NATIVE_ACTION_ROW = "native_row"
NATIVE_ACTION_STOP = "stop"

FAMILY_TARGET_LOCAL = "target_local"
FAMILY_SAME_CATEGORY_OTHER_OWNER = "same_category_other_owner"

#: The published shape of the support-calibration object this plan carries.  It
#: is a *binding* of the sealed predecessor receipt, not a second declaration:
#: every number in it is copied from the sealed census artifacts after the
#: digest chain is proven, and nothing here may be edited independently.
SUPPORT_CONTRACT_ID = "sorted-crossing-boundary-target-local-support-calibration.v1"

#: The census bound names and the owner-context path each one reads.  ``U``
#: (ambiguity-included) owns primary branch support; ``L`` (ambiguity-excluded)
#: is a sensitivity that never changes a primary branch.
SUPPORT_BOUND_PATHS: Mapping[str, str] = {
    "u": "ambiguity_included_u",
    "l": "ambiguity_excluded_l",
}
SUPPORT_PRIMARY_BOUND = "u"

#: The census partition names, in the frozen order the planner publishes them.
SUPPORT_PARTITIONS: tuple[str, ...] = planner.OWNER_CANDIDATE_PARTITIONS

REQUEST_NATURAL_RELEASE = "natural_release_target_description_path"
REQUEST_NATIVE_NEXT_ACTION = "native_next_action_replay"
REQUEST_COORDINATE_TARGET_LOCAL = "coordinate_target_local_candidates"
REQUEST_COORDINATE_COMPETITOR = "coordinate_same_category_competitor_candidates"
REQUEST_COORDINATE_GREEDY = "coordinate_greedy_row"
REQUEST_DOWNSTREAM_COMPATIBILITY = "downstream_compatibility_inserted_clean_row"
REQUEST_FAMILIES: tuple[str, ...] = (
    REQUEST_NATURAL_RELEASE,
    REQUEST_NATIVE_NEXT_ACTION,
    REQUEST_COORDINATE_TARGET_LOCAL,
    REQUEST_COORDINATE_COMPETITOR,
    REQUEST_COORDINATE_GREEDY,
    REQUEST_DOWNSTREAM_COMPATIBILITY,
)

#: The exhaustive primary branch order.  This planner declares the schema so
#: the later scorer/analyzer can bind to stable names; it never evaluates a
#: branch, because branch assignment needs model logits this pass must not see.
BRANCH_SCHEMA_ID = "sorted-crossing-boundary-primary-branch.v1"
BRANCH_ORDER: tuple[str, ...] = (
    "displaced",
    "release_lost",
    "realization_fail",
    "ambiguous",
)
BRANCH_SUB_TAGS: Mapping[str, tuple[str, ...]] = {
    "displaced": ("likelihood_displaced", "greedy_displaced"),
}

SEMANTICS_NOTES: tuple[str, ...] = (
    "This planner is score-blind: it reads only the sealed predecessor plan "
    "registries and the sealed predecessor owner/context feature rows, and it "
    "never loads a model, a tokenizer, or a new logit.",
    "Full E and F row token identity is owned only by the literal adjacent "
    "context prefix suffix tokens(next boundary) - tokens(this boundary); the "
    "native sidecar registry carries no full-row token sequence and is used "
    "only to validate the coordinate subsequence, coordinate digest, and "
    "declared row metadata.",
    "The U bound owns the primary cohort, support, rank, competitor, and "
    "branch fields; the L bound is a sensitivity that never replaces the "
    "primary denominator.",
    "Same-description owners have no observable description divergence at the "
    "crossing boundary: their P+D_C coordinate readout is construction-"
    "determined, because D_C is already the exact prefix of native row E.",
    "The disjoint timing controls are descriptive only: timing, description "
    "identity, and route tier are entangled, so they are never a matched "
    "causal control.",
    "The exact-GT-anchor inserted row C is an oracle intervention; its "
    "downstream compatibility does not imply the model could naturally "
    "generate it.",
    "Target-local support is the sealed predecessor local-peak conjunction on "
    "peak_lift and local_concentration, evaluated under both ambiguity bounds. "
    "Owner rank, group-best attainment, and competition margin are published "
    "routing surfaces and never create, raise, or lower support.",
    "This plan binds the sealed calibration receipt and its thresholds; it "
    "never re-derives a quantile, re-fits a threshold, widens an epsilon, or "
    "substitutes a crossing-unit score for the frozen census calibration.",
)


class PlanContractError(RuntimeError):
    """A precondition for the sealed CPU plan was not proven."""


def _fail(message: str) -> NoReturn:
    raise PlanContractError(message)


@contextmanager
def _predecessor_contract(label: str) -> Iterator[None]:
    """Re-raise a predecessor contract failure as this planner's own failure.

    The prevalence analyzer and the census merge both own fail-closed
    preconditions this planner depends on -- input validation on one side, the
    support-calibration provenance chain on the other.  Surfacing both as
    ``PlanContractError`` keeps one exception type at the plan boundary instead
    of leaking three, and keeps the CLI's fail-closed exit path intact.
    """

    try:
        yield
    except (prevalence.AnalysisContractError, merge.MergeContractError) as exc:
        raise PlanContractError(f"{label}: {exc}") from exc


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


def context_id_for(image_id: str, boundary_index: int) -> str:
    """The census context-id convention: ``<image>:boundary-{index:03d}``."""

    return f"{image_id}:boundary-{boundary_index:03d}"


def _token_ids(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(f"{label} is not a token-id sequence")
    tokens: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            _fail(f"{label} carries a non-integer token id {item!r}")
        tokens.append(int(item))
    return tokens


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanInputs:
    """Every sealed predecessor surface this planner is allowed to read."""

    prevalence_run_root: Path
    census_run_root: Path
    prevalence_file_sha256: dict[str, str]
    census_file_sha256: dict[str, str]
    prevalence_receipt: dict[str, Any]
    census: prevalence.Inputs
    capture_rules: dict[str, Any]
    support_calibration_receipt: dict[str, Any]
    images_by_id: dict[str, dict[str, Any]]
    owners_by_id: dict[str, dict[str, Any]]
    categories_by_query_id: dict[str, dict[str, Any]]
    query_groups_by_id: dict[str, dict[str, Any]]
    candidates_by_image_description: dict[tuple[str, str], list[dict[str, Any]]]
    exact_anchor_by_owner: dict[str, dict[str, Any]]
    sidecars_by_image_row: dict[tuple[str, int], dict[str, Any]]
    summaries_by_owner: dict[str, dict[str, Any]]
    wrapper_token_ids: dict[str, int]
    coordinate_token_ids: dict[str, int]


def load_plan_inputs(
    prevalence_run_root: Path, census_run_root: Path | None = None
) -> PlanInputs:
    """Read and hash both predecessor runs before a single row is interpreted."""

    prevalence_run_root = Path(prevalence_run_root)
    prevalence_file_sha256: dict[str, str] = {}
    for relative_name in PREVALENCE_INPUT_FILES:
        path = prevalence_run_root / relative_name
        if not path.is_file():
            _fail(
                f"prevalence input file is missing: {relative_name} under {prevalence_run_root}"
            )
        prevalence_file_sha256[relative_name] = sha256_bytes(path.read_bytes())

    prevalence_receipt = _read_json(
        prevalence_run_root / PREVALENCE_RECEIPT_REL, "prevalence analysis/receipt.json"
    )
    declared_root = prevalence_receipt.get("predecessor_run_root")
    if census_run_root is None:
        if not isinstance(declared_root, str) or not declared_root:
            _fail(
                "prevalence analysis/receipt.json carries no predecessor_run_root and no "
                "--census-run-root was supplied"
            )
        census_run_root = Path(declared_root)
    census_run_root = Path(census_run_root)

    with _predecessor_contract("sealed census inputs"):
        census = prevalence.load_inputs(census_run_root)

    census_file_sha256 = dict(census.file_sha256)
    for relative_name in EXTRA_CENSUS_PLAN_FILES + EXTRA_CENSUS_PHASE_FILES:
        path = census_run_root / relative_name
        if not path.is_file():
            _fail(f"census plan file is missing: {relative_name} under {census_run_root}")
        census_file_sha256[relative_name] = sha256_bytes(path.read_bytes())

    capture_rules = _read_json(census_run_root / CAPTURE_RULES_REL, "plan/capture-rules.json")
    support_calibration_receipt = _read_json(
        census_run_root / SUPPORT_CALIBRATION_REL, SUPPORT_CALIBRATION_REL
    )
    image_rows = _read_jsonl(census_run_root / IMAGE_REGISTRY_REL, "image-registry.jsonl")
    owner_rows = _read_jsonl(census_run_root / OWNER_REGISTRY_REL, "owner-registry.jsonl")
    category_rows = _read_jsonl(census_run_root / CATEGORY_REGISTRY_REL, "category-registry.jsonl")
    query_group_rows = _read_jsonl(
        census_run_root / QUERY_GROUP_REGISTRY_REL, "query-group-registry.jsonl"
    )
    candidate_rows = _read_jsonl(census_run_root / CANDIDATE_BANK_REL, "candidate-bank.jsonl")

    images_by_id: dict[str, dict[str, Any]] = {}
    for row in image_rows:
        image_id = str(row.get("image_id"))
        if image_id in images_by_id:
            _fail(f"image-registry.jsonl carries duplicate image_id {image_id!r}")
        images_by_id[image_id] = row

    owners_by_id: dict[str, dict[str, Any]] = {}
    for row in owner_rows:
        owner_id = str(row.get("gt_owner_id"))
        if owner_id in owners_by_id:
            _fail(f"owner-registry.jsonl carries duplicate gt_owner_id {owner_id!r}")
        owners_by_id[owner_id] = row

    categories_by_query_id: dict[str, dict[str, Any]] = {}
    for row in category_rows:
        query_id = str(row.get("category_query_id"))
        if query_id in categories_by_query_id:
            _fail(f"category-registry.jsonl carries duplicate category_query_id {query_id!r}")
        categories_by_query_id[query_id] = row

    query_groups_by_id: dict[str, dict[str, Any]] = {}
    for row in query_group_rows:
        group_id = str(row.get("query_group_id"))
        if group_id in query_groups_by_id:
            _fail(f"query-group-registry.jsonl carries duplicate query_group_id {group_id!r}")
        query_groups_by_id[group_id] = row

    candidates_by_image_description: dict[tuple[str, str], list[dict[str, Any]]] = {}
    exact_anchor_by_owner: dict[str, dict[str, Any]] = {}
    seen_candidate_ids: set[str] = set()
    for row in candidate_rows:
        candidate_id = str(row.get("candidate_id"))
        if candidate_id in seen_candidate_ids:
            _fail(f"candidate-bank.jsonl carries duplicate candidate_id {candidate_id!r}")
        seen_candidate_ids.add(candidate_id)
        key = (str(row.get("image_id")), str(row.get("normalized_description")))
        candidates_by_image_description.setdefault(key, []).append(row)
        for generator in row.get("generators") or ():
            if generator.get("logical_transform_role") != "exact_gt_anchor":
                continue
            owner_id = str(generator.get("generator_gt_owner_id"))
            if owner_id in exact_anchor_by_owner:
                _fail(
                    f"owner {owner_id!r} has more than one exact_gt_anchor candidate in "
                    "candidate-bank.jsonl"
                )
            exact_anchor_by_owner[owner_id] = row
    for key, rows in candidates_by_image_description.items():
        rows.sort(key=lambda row: str(row["candidate_id"]))
        del key

    sidecars_by_image_row: dict[tuple[str, int], dict[str, Any]] = {}
    for row in census.native_sidecars:
        key = (str(row.get("image_id")), int(row.get("row_index")))
        if key in sidecars_by_image_row:
            _fail(f"native-sidecar-registry.jsonl carries duplicate (image, row_index) {key!r}")
        sidecars_by_image_row[key] = row

    summaries_by_owner: dict[str, dict[str, Any]] = {}
    for row in census.owner_summaries:
        owner_id = str(row.get("gt_owner_id"))
        if owner_id in summaries_by_owner:
            _fail(f"owner-summaries.jsonl carries duplicate gt_owner_id {owner_id!r}")
        summaries_by_owner[owner_id] = row

    wrapper_token_ids, coordinate_token_ids = _bind_token_registry(images_by_id)

    return PlanInputs(
        prevalence_run_root=prevalence_run_root,
        census_run_root=census_run_root,
        prevalence_file_sha256=prevalence_file_sha256,
        census_file_sha256=census_file_sha256,
        prevalence_receipt=prevalence_receipt,
        census=census,
        capture_rules=capture_rules,
        support_calibration_receipt=support_calibration_receipt,
        images_by_id=images_by_id,
        owners_by_id=owners_by_id,
        categories_by_query_id=categories_by_query_id,
        query_groups_by_id=query_groups_by_id,
        candidates_by_image_description=candidates_by_image_description,
        exact_anchor_by_owner=exact_anchor_by_owner,
        sidecars_by_image_row=sidecars_by_image_row,
        summaries_by_owner=summaries_by_owner,
        wrapper_token_ids=wrapper_token_ids,
        coordinate_token_ids=coordinate_token_ids,
    )


def _bind_token_registry(
    images_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    """One wrapper/coordinate token registry shared by every executed image."""

    if not images_by_id:
        _fail("image-registry.jsonl is empty; no token registry can be bound")
    wrappers: dict[str, int] | None = None
    coordinates: dict[str, int] | None = None
    for image_id, row in sorted(images_by_id.items()):
        image_wrappers = row.get("wrapper_token_ids")
        image_coordinates = row.get("coordinate_token_ids")
        if not isinstance(image_wrappers, Mapping) or not isinstance(image_coordinates, Mapping):
            _fail(f"image-registry.jsonl row {image_id!r} carries no token registry")
        image_wrappers = {str(key): int(value) for key, value in image_wrappers.items()}
        image_coordinates = {str(key): int(value) for key, value in image_coordinates.items()}
        if wrappers is None:
            wrappers, coordinates = image_wrappers, image_coordinates
            continue
        if image_wrappers != wrappers or image_coordinates != coordinates:
            _fail(
                f"image {image_id!r} declares a token registry different from the other "
                "images; the plan refuses to mix token identities"
            )
    assert wrappers is not None and coordinates is not None  # for type-checkers
    for name in ("object_ref_start", "object_ref_end", "box_start", "box_end", "im_end"):
        if name not in wrappers:
            _fail(f"image-registry.jsonl token registry is missing wrapper {name!r}")
    for name in ("start", "end_inclusive", "bin_count"):
        if name not in coordinates:
            _fail(f"image-registry.jsonl coordinate token registry is missing {name!r}")
    if coordinates["end_inclusive"] - coordinates["start"] + 1 != coordinates["bin_count"]:
        _fail("image-registry.jsonl coordinate token range does not match its declared bin count")
    return wrappers, coordinates


# ---------------------------------------------------------------------------
# Fail-closed validation of both predecessor runs
# ---------------------------------------------------------------------------


def validate_plan_inputs(inputs: PlanInputs) -> dict[str, Any]:
    """Every precondition the sealed plan depends on, proven before use."""

    receipt = inputs.prevalence_receipt
    if receipt.get("schema_version") != prevalence.RECEIPT_SCHEMA_VERSION:
        _fail("prevalence analysis/receipt.json schema_version is not the frozen prevalence schema")
    if receipt.get("unit_id") != PREVALENCE_UNIT_ID:
        _fail("prevalence analysis/receipt.json unit_id does not match the prevalence unit")
    reconstructed = sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    if reconstructed != receipt.get("receipt_content_sha256"):
        _fail(
            "prevalence analysis/receipt.json does not reconstruct its own "
            "receipt_content_sha256; it was edited"
        )
    for relative_name, receipt_key in (
        (PREVALENCE_REPORT_JSON_REL, "report_json_sha256"),
        (PREVALENCE_REPORT_MD_REL, "report_md_sha256"),
        (PREVALENCE_OWNER_RECORDS_REL, "owner_records_jsonl_sha256"),
    ):
        if receipt.get(receipt_key) != inputs.prevalence_file_sha256[relative_name]:
            _fail(
                f"{relative_name} bytes do not match the digest sealed in the prevalence receipt"
            )
    prevalence_validation = receipt.get("validation") or {}
    if prevalence_validation.get("usable_as_census_conclusion") is not True:
        _fail(
            "the prevalence receipt does not declare its census input usable as a "
            "conclusion; this planner refuses to run"
        )

    with _predecessor_contract("sealed census validation"):
        census_validation = prevalence.validate_inputs(inputs.census)

    declared_inputs = receipt.get("input_file_sha256") or {}
    for relative_name in prevalence.AUTHORITATIVE_INPUT_FILES:
        declared = declared_inputs.get(relative_name)
        if declared is None:
            _fail(
                f"the prevalence receipt does not seal {relative_name}; the census run cannot "
                "be bound to the prevalence result"
            )
        if declared != inputs.census_file_sha256[relative_name]:
            _fail(
                f"{relative_name} in the census run does not match the digest sealed in the "
                "prevalence receipt; the two predecessor runs disagree"
            )

    declared_plan_digests = inputs.census.plan_receipt.get("output_file_digests") or {}
    for relative_name in EXTRA_CENSUS_PLAN_FILES:
        plan_key = relative_name.split("/", 1)[1]
        if declared_plan_digests.get(plan_key) != inputs.census_file_sha256[relative_name]:
            _fail(f"{relative_name} bytes do not match the digest sealed in plan/receipt.json")

    for label, rows in (
        ("image-registry.jsonl", inputs.images_by_id),
        ("owner-registry.jsonl", inputs.owners_by_id),
        ("category-registry.jsonl", inputs.categories_by_query_id),
        ("query-group-registry.jsonl", inputs.query_groups_by_id),
    ):
        for key, row in rows.items():
            if row.get("schema_version") != planner.PLAN_SCHEMA_VERSION:
                _fail(f"{label} row {key!r} carries an unexpected schema_version")

    score_policy = inputs.census.plan_receipt.get("score_input_policy") or {}
    if score_policy.get("candidate_selection_uses_scores") is not False:
        _fail("the census plan does not declare score-blind candidate selection")

    query_contract = inputs.census.plan_receipt.get("query_suffix_contract") or {}
    declared_wrappers = {
        str(key): int(value) for key, value in (query_contract.get("wrapper_token_ids") or {}).items()
    }
    if declared_wrappers != inputs.wrapper_token_ids:
        _fail(
            "plan/receipt.json query_suffix_contract.wrapper_token_ids disagrees with the "
            "image-registry token registry"
        )
    if list(query_contract.get("shape") or ()) != [
        "object_ref_start",
        "category_token_ids",
        "object_ref_end",
        "box_start",
    ]:
        _fail("plan/receipt.json query_suffix_contract.shape is not the frozen D_C shape")

    return {
        "census": census_validation,
        "prevalence_receipt_content_sha256": receipt.get("receipt_content_sha256"),
        "prevalence_unit_id": PREVALENCE_UNIT_ID,
        "census_unit_id": CENSUS_UNIT_ID,
        "image_count": len(inputs.images_by_id),
        "owner_registry_row_count": len(inputs.owners_by_id),
        "category_registry_row_count": len(inputs.categories_by_query_id),
        "query_group_row_count": len(inputs.query_groups_by_id),
        "candidate_bank_row_count": sum(
            len(rows) for rows in inputs.candidates_by_image_description.values()
        ),
        "exact_gt_anchor_owner_count": len(inputs.exact_anchor_by_owner),
    }


# ---------------------------------------------------------------------------
# Target-local support calibration
# ---------------------------------------------------------------------------


class _CaptureRulesView:
    """Minimal ``PlanBundle`` view so the merge's own loader can be reused.

    ``merge.load_support_contract`` binds the statistic names, the primary and
    sensitivity quantiles, both epsilons, the category floor, and the
    "rank is not a criterion" guarantee straight out of the sealed capture
    rules, and fails closed if any of them is absent.  Reusing it is the whole
    point: a second local reader would be a second implementation of a frozen
    contract and could drift away from it silently.
    """

    __slots__ = ("capture_rules",)

    def __init__(self, capture_rules: Mapping[str, Any]) -> None:
        self.capture_rules = capture_rules


def _derived_partition_bounds() -> dict[str, list[str]]:
    """The partition-to-bound map, derived from the frozen census classifier.

    Nothing here decides anything: it re-runs
    ``planner.classify_candidate_for_owner`` on one synthetic candidate per
    partition so the map this plan publishes is provably the census classifier's
    map, and a classifier change fails the build instead of silently reshaping
    an owner's U or L bank.
    """

    owner_id = "gt:probe:0"
    other_id = "gt:probe:1"

    def _candidate(generator_id: str, status: str, assigned: str | None) -> dict[str, Any]:
        return {
            "generators": [{"generator_gt_owner_id": generator_id}],
            "strict_assignment_status": status,
            "strict_assignment_gt_owner_id": assigned,
        }

    probes = {
        "strict_assigned_self": _candidate(owner_id, "matched", owner_id),
        "ambiguous_upper": _candidate(owner_id, "ambiguous_neutral", None),
        "unmatched_generator_local": _candidate(owner_id, "unmatched", None),
        "other_owner_strict": _candidate(owner_id, "matched", other_id),
        "not_generated_by_owner": _candidate(other_id, "unmatched", None),
    }
    if set(probes) != set(SUPPORT_PARTITIONS):
        _fail(
            "the census partition vocabulary changed; this plan will not publish a "
            "partition-to-bound map it cannot derive from the frozen classifier"
        )
    derived: dict[str, list[str]] = {}
    for partition, candidate in probes.items():
        classified = planner.classify_candidate_for_owner(candidate, owner_id)
        if str(classified["partition"]) != partition:
            _fail(
                f"the census classifier no longer assigns partition {partition!r} to its own "
                "probe candidate; the sealed support partition semantics drifted"
            )
        bounds: list[str] = []
        if classified["counts_toward_lower_bound"]:
            bounds.append("lower")
        if classified["counts_toward_upper_bound"]:
            bounds.append("upper")
        derived[partition] = bounds
    return derived


def bind_support_calibration(inputs: PlanInputs) -> dict[str, Any]:
    """Seal the exact predecessor support calibration and publish its contract.

    Four independent things are proven before any threshold is copied:

    1. the calibration artifact reconstructs its own ``calibration_sha256`` and
       names the merge source this planner imports (``merge.calibration_from_
       receipt`` owns both checks and refuses foreign-provenance thresholds);
    2. the presentation phase -- the phase whose owner summaries this planner
       reads -- applied *that* receipt, byte-for-byte, and declares the same
       criterion, epsilon and quantile in its ``support_semantics``;
    3. the sealed capture rules, which froze the constants before any score
       existed, agree with the receipt on every number; and
    4. the partition-to-bound map is the frozen classifier's, not an authored
       copy.

    Returns the minimal explicit contract the scorer needs, with the source
    sealed by path, byte size and digest.
    """

    receipt = dict(inputs.support_calibration_receipt)
    with _predecessor_contract("sealed census support calibration"):
        calibration = merge.calibration_from_receipt(receipt)

    if str(receipt.get("criterion_id")) != merge.SUPPORT_CRITERION_ID:
        _fail(
            f"the sealed calibration criterion_id {receipt.get('criterion_id')!r} is not the "
            f"frozen {merge.SUPPORT_CRITERION_ID!r}"
        )
    if tuple(calibration.statistics) != merge.SUPPORT_FEATURE_NAMES:
        _fail(
            f"the sealed calibration statistics {list(calibration.statistics)} are not the frozen "
            f"{list(merge.SUPPORT_FEATURE_NAMES)}"
        )
    for key in ("rank_is_not_a_support_input", "quantile_is_primary_and_fixed"):
        if receipt.get(key) is not True:
            _fail(f"the sealed calibration does not declare {key} true")
    if receipt.get("epsilon_is_adaptive") is not False:
        _fail("the sealed calibration does not declare a fixed, non-adaptive epsilon")

    merge_receipt = inputs.census.merge_receipt
    applied = merge_receipt.get("support_criterion")
    if not isinstance(applied, Mapping):
        _fail(
            "the presentation merge receipt carries no support_criterion block, so the "
            "calibration actually applied to the owner summaries cannot be proven"
        )
    if dict(applied) != receipt:
        _fail(
            "the sealed calibration artifact is not the receipt the presentation phase applied "
            "to the owner summaries this plan consumes; the two disagree"
        )
    semantics = merge_receipt.get("support_semantics")
    if not isinstance(semantics, Mapping):
        _fail("the presentation merge receipt carries no support_semantics block")
    for receipt_key, semantics_key in (
        ("criterion_id", "criterion_id"),
        ("epsilon", "epsilon"),
        ("cross_context_delta_epsilon", "cross_context_delta_epsilon"),
        ("quantile", "primary_quantile"),
    ):
        if semantics.get(semantics_key) != receipt.get(receipt_key):
            _fail(
                f"presentation support_semantics.{semantics_key} disagrees with the sealed "
                f"calibration {receipt_key}"
            )
    if semantics.get("rank_is_not_a_support_input") is not True:
        _fail("presentation support_semantics does not forbid a rank support input")
    if list(semantics.get("inputs") or ()) != list(calibration.statistics):
        _fail("presentation support_semantics inputs disagree with the calibration statistics")

    with _predecessor_contract("sealed capture-rules owner_support"):
        contract = merge.load_support_contract(_CaptureRulesView(inputs.capture_rules))
    for label, sealed, declared in (
        ("primary_quantile", contract.primary_quantile, calibration.quantile),
        ("support_epsilon", contract.support_epsilon, calibration.epsilon),
        (
            "cross_context_delta_epsilon",
            contract.cross_context_delta_epsilon,
            calibration.cross_context_delta_epsilon,
        ),
        (
            "category_contribution_min",
            contract.category_contribution_min,
            calibration.category_contribution_min,
        ),
        (
            "underrepresented_flag",
            contract.underrepresented_flag,
            calibration.underrepresented_flag,
        ),
    ):
        if sealed != declared:
            _fail(
                f"capture-rules owner_support {label} ({sealed!r}) disagrees with the sealed "
                f"calibration receipt ({declared!r})"
            )
    if tuple(contract.statistics) != tuple(calibration.statistics):
        _fail("capture-rules owner_support statistics disagree with the calibration receipt")

    owner_support = inputs.capture_rules.get("owner_support") or {}
    definition = owner_support.get("support_definition") or {}
    if definition.get("evaluated_under_both_ambiguity_bounds") is not True:
        _fail("the sealed support definition is not evaluated under both ambiguity bounds")
    if definition.get("rank_one_as_support") != "forbidden":
        _fail("the sealed support definition does not forbid rank-one as support")
    if str(definition.get("usable_support_combinator")) != "conjunction_both_statistics":
        _fail("the sealed support definition is not the two-statistic conjunction")
    computed_on = str(definition.get("computed_on"))
    if computed_on != "generator_local_max_excluding_other_owner_strict":
        _fail(
            f"the sealed support definition is computed on {computed_on!r}, not the "
            "exclusion-filtered generator-local landscape this plan partitions"
        )

    sealed_partition_bounds = owner_support.get("partition_bounds")
    if not isinstance(sealed_partition_bounds, Mapping):
        _fail("capture-rules owner_support seals no partition_bounds map")
    derived_partition_bounds = _derived_partition_bounds()
    normalized_sealed = {
        str(key): [str(value) for value in (values or ())]
        for key, values in sealed_partition_bounds.items()
    }
    if normalized_sealed != derived_partition_bounds:
        _fail(
            "the sealed partition_bounds map disagrees with the frozen census classifier; the "
            "U/L membership of a target-local candidate would not be reproducible"
        )

    source_path = Path(inputs.census_run_root) / SUPPORT_CALIBRATION_REL
    source_bytes = source_path.read_bytes()
    source_digest = sha256_bytes(source_bytes)
    if source_digest != inputs.census_file_sha256[SUPPORT_CALIBRATION_REL]:
        _fail(
            f"{SUPPORT_CALIBRATION_REL} changed on disk between loading and sealing; the "
            "calibration lineage would not describe the bytes that were read"
        )

    return {
        "support_contract_id": SUPPORT_CONTRACT_ID,
        "criterion_id": merge.SUPPORT_CRITERION_ID,
        "source": {
            "path": SUPPORT_CALIBRATION_REL,
            "byte_size": len(source_bytes),
            "sha256": source_digest,
            "run_root": str(inputs.census_run_root),
            "calibration_sha256": str(receipt["calibration_sha256"]),
            "calibration_schema_version": str(receipt["schema_version"]),
            "calibration_unit_id": str(receipt["unit_id"]),
            "merge_source_sha256": str(receipt["merge_source_sha256"]),
            "merge_source_role": str(receipt.get("merge_source_role")),
            "capture_manifest_sha256": str(calibration.capture_manifest_sha256),
            "applied_by_phase": str(merge_receipt.get("phase")),
            "applied_receipt_content_sha256": merge_receipt.get("receipt_content_sha256"),
            "reproduced_from": "the sealed census receipt; no threshold is authored here",
        },
        "thresholds": {
            "theta_peak_lift": calibration.theta_peak_lift,
            "theta_local_concentration": calibration.theta_local_concentration,
            "epsilon": calibration.epsilon,
            "quantile": calibration.quantile,
            "quantile_is_primary_and_fixed": True,
            "epsilon_is_adaptive": False,
            "cross_context_delta_epsilon": calibration.cross_context_delta_epsilon,
            "calibration_stratum": str(receipt["calibration_stratum"]),
            "observation_count": calibration.observation_count,
        },
        "rule": {
            "statistics": list(calibration.statistics),
            "text": str(receipt["rule"]),
            "combinator": str(definition["usable_support_combinator"]),
            "usable_support_rule": str(definition["usable_support_rule"]),
            "computed_on": computed_on,
            "evaluated_under_both_ambiguity_bounds": True,
            "peak_lift": dict(definition["peak_lift"]),
            "local_concentration": dict(definition["local_concentration"]),
        },
        "bounds": {
            "paths": dict(SUPPORT_BOUND_PATHS),
            "primary_bound": SUPPORT_PRIMARY_BOUND,
            "primary_bound_role": "owns_primary_support_rank_competitor_and_branch_fields",
            "sensitivity_bound": "l",
            "sensitivity_bound_role": "reported_beside_u_never_changes_a_u_branch",
            "partition_bounds": derived_partition_bounds,
            "partition_bounds_source": "frozen_census_classify_candidate_for_owner",
        },
        "populations": {
            "peak_lift_population": (
                "collapsed_unique_physical_candidates_of_one_image_context_"
                "normalized_description_query_group"
            ),
            "peak_lift_population_includes": [
                FAMILY_TARGET_LOCAL,
                FAMILY_SAME_CATEGORY_OTHER_OWNER,
            ],
            "local_concentration_population": (
                "the target owner's own exclusion-filtered generator-local bank under the same "
                "bound, never the whole query group"
            ),
            "median_convention": str(
                definition["local_concentration"]["reference_statistic"]
            ),
            "other_owner_strict_role": str(
                owner_support.get("other_owner_strict_candidate")
            ),
        },
        "rank_and_margin": {
            "rank_is_a_support_input": False,
            "margin_is_a_support_input": False,
            "role": str(semantics.get("rank_and_margin_role")),
            "rank_one_as_support": "forbidden",
        },
        "category_stratification": {
            "role": "report_only_sensitivity_never_a_threshold",
            "category_contribution_min": calibration.category_contribution_min,
            "underrepresented_flag": calibration.underrepresented_flag,
        },
        "reproduction_inputs": [
            "candidate_families.target_local.bounds.<bound>.candidate_ids",
            "candidate_families.unique_population.candidate_ids",
            "support_calibration.thresholds",
            "support_calibration.rule",
        ],
        "derived_here": False,
        "uses_crossing_unit_scores": False,
    }


def _support_reference(support: Mapping[str, Any]) -> dict[str, Any]:
    """The compact per-row pointer back to the sealed calibration contract."""

    return {
        "support_contract_id": support["support_contract_id"],
        "criterion_id": support["criterion_id"],
        "calibration_sha256": support["source"]["calibration_sha256"],
        "calibration_source_sha256": support["source"]["sha256"],
        "theta_peak_lift": support["thresholds"]["theta_peak_lift"],
        "theta_local_concentration": support["thresholds"]["theta_local_concentration"],
        "epsilon": support["thresholds"]["epsilon"],
        "rule": support["rule"]["text"],
        "statistics": list(support["rule"]["statistics"]),
        "primary_bound": support["bounds"]["primary_bound"],
        "rank_is_a_support_input": False,
        "margin_is_a_support_input": False,
    }


# ---------------------------------------------------------------------------
# Crossing-boundary derivation
# ---------------------------------------------------------------------------


def owner_context_rows(inputs: PlanInputs, gt_owner_id: str, image_id: str) -> dict[int, dict[str, Any]]:
    """Every sealed owner-context row for one owner, keyed by boundary index.

    Rows are image-isolated by construction: a row whose ``image_id`` is not
    the owner's own image fails the build rather than silently joining across
    images.
    """

    rows: dict[int, dict[str, Any]] = {}
    for owner_context_id, row in inputs.census.owner_context_by_id.items():
        if str(row.get("gt_owner_id")) != gt_owner_id:
            continue
        if str(row.get("image_id")) != image_id:
            _fail(
                f"owner-context row {owner_context_id!r} belongs to image "
                f"{row.get('image_id')!r} but owner {gt_owner_id!r} belongs to {image_id!r}"
            )
        context_id = str(row.get("context_id"))
        if not context_id.startswith(f"{image_id}:"):
            _fail(
                f"owner-context row {owner_context_id!r} references context {context_id!r} "
                f"outside image {image_id!r}"
            )
        boundary_index = int(row["boundary_index"])
        if boundary_index in rows:
            _fail(
                f"owner {gt_owner_id!r} carries two owner-context rows at boundary "
                f"{boundary_index}"
            )
        rows[boundary_index] = row
    if not rows:
        _fail(f"owner {gt_owner_id!r} has no sealed owner-context rows")
    return rows


def derive_crossing_boundary(rows: Mapping[int, Mapping[str, Any]]) -> int | None:
    """The crossing boundary ``b`` for one owner, or ``None`` when it never crosses.

    The first boundary at which the owner is ``passed_by_frontier`` ends the
    search: ``b`` is the immediately preceding boundary and must itself be a
    root/ahead boundary.  A first-passed boundary with no root/ahead
    predecessor is not a crossing and is never repaired by scanning further.
    """

    for boundary_index in sorted(rows):
        state = str(rows[boundary_index]["frontier_features"]["passed_state"])
        if state != PASSED_STATE:
            continue
        previous = boundary_index - 1
        if previous not in rows:
            return None
        if str(rows[previous]["frontier_features"]["passed_state"]) not in ROOT_OR_AHEAD_STATES:
            return None
        return previous
    return None


def _favorable_top3(channel: Mapping[str, Any]) -> bool:
    return bool(channel["gate_open"] and channel["category_rank_top3"] and channel["owner_rank_one"])


def _usable_support_context_ids(summary: Mapping[str, Any], bound: str) -> frozenset[str]:
    block_name = "upper_bound_u" if bound == "u" else "lower_bound_l"
    block = summary.get(block_name) or {}
    ids = block.get("usable_support_context_ids")
    if ids is None:
        _fail(
            f"owner-summaries.jsonl row for {summary.get('gt_owner_id')!r} carries no "
            f"{block_name}.usable_support_context_ids"
        )
    return frozenset(str(value) for value in ids)


def _channel_at(
    inputs: PlanInputs, gt_owner_id: str, context_id: str, *, bound: str
) -> dict[str, Any]:
    owner_context_id = f"{gt_owner_id}@{context_id}"
    owner_context_row = inputs.census.owner_context_by_id.get(owner_context_id)
    if owner_context_row is None:
        _fail(f"owner-context row {owner_context_id!r} is absent from the sealed census")
    context = inputs.census.contexts_by_id.get(context_id)
    if context is None:
        _fail(f"context {context_id!r} is absent from context-registry.jsonl")
    with _predecessor_contract(f"context channel {owner_context_id!r}"):
        return prevalence.extract_context_channel(owner_context_row, context, bound=bound)


def favorable_supported(
    inputs: PlanInputs,
    summary: Mapping[str, Any],
    context_id: str,
    *,
    bound: str,
) -> tuple[bool, dict[str, Any]]:
    """Is this owner favorable *and* calibrated-supported at this context?

    ``favorable`` is the frozen conjunction gate-open AND category route in the
    top three AND same-category owner rank one, read verbatim from the sealed
    bound-specific competition; ``supported`` is membership in the same bound's
    sealed ``usable_support_context_ids``.  Neither is recomputed here.
    """

    channel = _channel_at(inputs, str(summary["gt_owner_id"]), context_id, bound=bound)
    supported = context_id in _usable_support_context_ids(summary, bound)
    favorable = _favorable_top3(channel) and bool(channel["before_or_at_frontier"]) and supported
    channel = dict(channel)
    channel["usable_support"] = supported
    channel["favorable_top3_supported_before_or_at_frontier"] = favorable
    return favorable, channel


@dataclass(frozen=True)
class CrossingCase:
    gt_owner_id: str
    image_id: str
    boundary_index: int
    p_context_id: str
    pe_context_id: str
    u_channel: dict[str, Any]
    l_channel: dict[str, Any]
    u_favorable: bool
    l_favorable: bool


def derive_crossing_cases(inputs: PlanInputs) -> list[CrossingCase]:
    """Every supported false-negative owner that crosses, with both bounds read."""

    cases: list[CrossingCase] = []
    fn_owners = [
        row
        for row in inputs.census.owner_summaries
        if row.get("disposition") == merge.DISPOSITION_RESOLVED
    ]
    for summary in sorted(fn_owners, key=lambda row: str(row["gt_owner_id"])):
        gt_owner_id = str(summary["gt_owner_id"])
        image_id = str(summary["image_id"])
        rows = owner_context_rows(inputs, gt_owner_id, image_id)
        boundary_index = derive_crossing_boundary(rows)
        if boundary_index is None:
            continue
        p_context_id = context_id_for(image_id, boundary_index)
        pe_context_id = context_id_for(image_id, boundary_index + 1)
        if pe_context_id not in inputs.census.contexts_by_id:
            _fail(
                f"owner {gt_owner_id!r} crosses at boundary {boundary_index} but the "
                f"post-crossing context {pe_context_id!r} is absent from the context registry"
            )
        u_favorable, u_channel = favorable_supported(inputs, summary, p_context_id, bound="u")
        l_favorable, l_channel = favorable_supported(inputs, summary, p_context_id, bound="l")
        cases.append(
            CrossingCase(
                gt_owner_id=gt_owner_id,
                image_id=image_id,
                boundary_index=boundary_index,
                p_context_id=p_context_id,
                pe_context_id=pe_context_id,
                u_channel=u_channel,
                l_channel=l_channel,
                u_favorable=u_favorable,
                l_favorable=l_favorable,
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Literal token identity for native rows
# ---------------------------------------------------------------------------


def _context_prefix_tokens(inputs: PlanInputs, context_id: str) -> list[int]:
    context = inputs.census.contexts_by_id.get(context_id)
    if context is None:
        _fail(f"context {context_id!r} is absent from context-registry.jsonl")
    tokens = _token_ids(
        context.get("generated_prefix_token_ids"),
        label=f"context {context_id!r} generated_prefix_token_ids",
    )
    declared = context.get("generated_prefix_token_ids_sha256")
    if sha256_json(tokens) != declared:
        _fail(
            f"context {context_id!r} generated_prefix_token_ids do not reconstruct their own "
            "declared digest"
        )
    return tokens


def bind_native_row(inputs: PlanInputs, image_id: str, row_index: int, *, label: str) -> dict[str, Any]:
    """The literal token identity of one complete native row.

    The row's token IDs are owned only by ``tokens(boundary row_index + 1) -
    tokens(boundary row_index)``.  The sidecar is a validator, never a source:
    it supplies the declared coordinate tokens, coordinate digest, row id/index,
    description, strict-match fields, and raw-span digest that the literal
    suffix must satisfy.
    """

    pre_context_id = context_id_for(image_id, row_index)
    post_context_id = context_id_for(image_id, row_index + 1)
    if post_context_id not in inputs.census.contexts_by_id:
        _fail(
            f"{label}: native row {row_index} of image {image_id!r} has no post-row boundary "
            f"context {post_context_id!r}"
        )
    pre_tokens = _context_prefix_tokens(inputs, pre_context_id)
    post_tokens = _context_prefix_tokens(inputs, post_context_id)
    if post_tokens[: len(pre_tokens)] != pre_tokens:
        _fail(
            f"{label}: context {pre_context_id!r} is not a literal token prefix of "
            f"{post_context_id!r}; the adjacent-context row attribution is unsafe"
        )
    suffix = post_tokens[len(pre_tokens) :]
    if not suffix:
        _fail(f"{label}: native row {row_index} of image {image_id!r} has an empty token suffix")

    sidecar = inputs.sidecars_by_image_row.get((image_id, row_index))
    if sidecar is None:
        _fail(f"{label}: native sidecar row {row_index} of image {image_id!r} is missing")
    if str(sidecar.get("image_id")) != image_id:
        _fail(f"{label}: native sidecar row {row_index} belongs to another image")

    wrappers = inputs.wrapper_token_ids
    if suffix[0] != wrappers["object_ref_start"]:
        _fail(f"{label}: native row {row_index} does not open with <|object_ref_start|>")
    if suffix[-1] != wrappers["box_end"]:
        _fail(f"{label}: native row {row_index} does not close with <|box_end|>")

    coord_tokens = _token_ids(
        sidecar.get("coord_token_ids"), label=f"{label}: sidecar coord_token_ids"
    )
    if len(coord_tokens) != 4:
        _fail(f"{label}: native row {row_index} sidecar does not declare exactly four coordinate tokens")
    if sha256_json(coord_tokens) != sidecar.get("coord_token_ids_sha256"):
        _fail(f"{label}: native row {row_index} sidecar coordinate digest does not reconstruct")
    _validate_coordinate_tokens(inputs, coord_tokens, label=f"{label}: native row {row_index}")

    occurrences = [
        index for index in range(len(suffix) - 3) if suffix[index : index + 4] == coord_tokens
    ]
    if len(occurrences) != 1:
        _fail(
            f"{label}: sidecar coordinate tokens occur {len(occurrences)} times in the literal "
            f"token suffix of native row {row_index}; exactly one occurrence is required"
        )
    coord_offset = occurrences[0]
    if coord_offset < 3 or coord_offset + 5 != len(suffix):
        _fail(f"{label}: native row {row_index} does not have the frozen single-row token shape")
    if suffix[coord_offset - 1] != wrappers["box_start"]:
        _fail(f"{label}: native row {row_index} coordinate block is not preceded by <|box_start|>")
    if suffix[coord_offset - 2] != wrappers["object_ref_end"]:
        _fail(f"{label}: native row {row_index} description block does not close with <|object_ref_end|>")

    description_token_ids = suffix[1 : coord_offset - 2]
    if not description_token_ids:
        _fail(f"{label}: native row {row_index} carries an empty description token block")

    post_context = inputs.census.contexts_by_id[post_context_id]
    declared_rows = [
        row for row in post_context.get("prefix_rows") or () if int(row["row_index"]) == row_index
    ]
    if len(declared_rows) != 1:
        _fail(
            f"{label}: context {post_context_id!r} declares {len(declared_rows)} prefix rows at "
            f"row index {row_index}; exactly one is required"
        )
    declared_row = declared_rows[0]
    for field, sidecar_field in (
        ("pred_row_id", "pred_row_id"),
        ("raw_span_sha256", "raw_span_sha256"),
        ("description", "normalized_description"),
        ("strict_match_status", "strict_match_status"),
        ("strict_match_gt_owner_id", "strict_match_gt_owner_id"),
    ):
        if declared_row.get(field) != sidecar.get(sidecar_field):
            _fail(
                f"{label}: native row {row_index} sidecar {sidecar_field!r} disagrees with the "
                f"context-registry prefix row {field!r}"
            )

    strict_match_status = str(sidecar.get("strict_match_status"))
    strict_match_owner = sidecar.get("strict_match_gt_owner_id")
    if strict_match_status not in SIDECAR_MATCH_STATUSES:
        _fail(
            f"{label}: native row {row_index} carries unknown strict_match_status "
            f"{strict_match_status!r}"
        )
    if strict_match_status == SIDECAR_MATCHED and not strict_match_owner:
        _fail(f"{label}: native row {row_index} is strict-matched but names no owner")
    if strict_match_status == SIDECAR_UNMATCHED and strict_match_owner:
        _fail(f"{label}: native row {row_index} is unmatched but names an owner")

    return {
        "row_index": row_index,
        "image_id": image_id,
        "sidecar_id": str(sidecar.get("sidecar_id")),
        "pred_row_id": str(sidecar.get("pred_row_id")),
        "normalized_description": str(sidecar.get("normalized_description")),
        "strict_match_status": strict_match_status,
        "strict_match_gt_owner_id": strict_match_owner,
        "raw_span_sha256": sidecar.get("raw_span_sha256"),
        "bbox_pixel_xyxy": sidecar.get("bbox_pixel_xyxy"),
        "coord_token_ids": coord_tokens,
        "coord_token_ids_sha256": sidecar.get("coord_token_ids_sha256"),
        "coord_token_offset_in_row": coord_offset,
        "description_token_ids": description_token_ids,
        "full_row_token_ids": suffix,
        "full_row_token_ids_sha256": sha256_json(suffix),
        "full_row_token_count": len(suffix),
        "full_row_token_source": (
            "literal_adjacent_context_prefix_suffix_tokens_post_boundary_minus_pre_boundary"
        ),
        "pre_row_context_id": pre_context_id,
        "post_row_context_id": post_context_id,
        "sidecar_role": (
            "validates_coordinate_subsequence_digest_and_declared_row_metadata_only_never_owns_row_tokens"
        ),
    }


def _validate_coordinate_tokens(
    inputs: PlanInputs, tokens: Sequence[int], *, label: str
) -> None:
    start = inputs.coordinate_token_ids["start"]
    end = inputs.coordinate_token_ids["end_inclusive"]
    for token in tokens:
        if not start <= token <= end:
            _fail(f"{label}: token {token} is outside the frozen coordinate-token range")


def native_next_action(inputs: PlanInputs, image_id: str, boundary_index: int) -> dict[str, Any]:
    """The exact native action taken at one boundary: a complete row, or STOP."""

    context_id = context_id_for(image_id, boundary_index)
    context = inputs.census.contexts_by_id.get(context_id)
    if context is None:
        _fail(f"context {context_id!r} is absent from context-registry.jsonl")
    if str(context.get("context_role")) == "terminal":
        return {
            "kind": NATIVE_ACTION_STOP,
            "context_id": context_id,
            "terminal_kind": context.get("terminal_kind"),
            "token_ids": [inputs.wrapper_token_ids["im_end"]],
            "token_ids_sha256": sha256_json([inputs.wrapper_token_ids["im_end"]]),
            "row_index": None,
        }
    row = bind_native_row(
        inputs, image_id, boundary_index, label=f"native next action at {context_id!r}"
    )
    return {
        "kind": NATIVE_ACTION_ROW,
        "context_id": context_id,
        "terminal_kind": None,
        "token_ids": row["full_row_token_ids"],
        "token_ids_sha256": row["full_row_token_ids_sha256"],
        "row_index": row["row_index"],
    }


# ---------------------------------------------------------------------------
# Target description path (D_C), inserted clean row (C), candidate families
# ---------------------------------------------------------------------------


def bind_target_description_path(
    inputs: PlanInputs, gt_owner_id: str, image_id: str, description: str, context_ids: Sequence[str]
) -> dict[str, Any]:
    """``D_C`` = the sealed category query suffix through ``<|box_start|>``.

    The tokens come from the category registry and must be reproduced exactly
    by the predecessor query-group request identity at every context this plan
    will use, otherwise the request is not the predecessor's request.
    """

    category_query_id = f"{image_id}:{description}"
    category = inputs.categories_by_query_id.get(category_query_id)
    if category is None:
        _fail(
            f"owner {gt_owner_id!r} has no category-registry row {category_query_id!r}; its "
            "description path cannot be bound without retokenization"
        )
    if str(category.get("status")) != "admitted":
        _fail(f"category {category_query_id!r} is not admitted in the sealed census")
    suffix_tokens = _token_ids(
        category.get("query_suffix_token_ids"), label=f"category {category_query_id!r} query suffix"
    )
    category_tokens = _token_ids(
        category.get("category_token_ids"), label=f"category {category_query_id!r} category tokens"
    )
    wrappers = inputs.wrapper_token_ids
    expected = (
        [wrappers["object_ref_start"]]
        + category_tokens
        + [wrappers["object_ref_end"], wrappers["box_start"]]
    )
    if suffix_tokens != expected:
        _fail(
            f"category {category_query_id!r} query suffix is not the frozen "
            "<|object_ref_start|> description <|object_ref_end|> <|box_start|> shape"
        )
    if sha256_json(suffix_tokens) != category.get("query_suffix_token_ids_sha256"):
        _fail(f"category {category_query_id!r} query suffix digest does not reconstruct")

    query_group_ids: list[str] = []
    for context_id in context_ids:
        query_group_id = f"{context_id}|{description}"
        group = inputs.query_groups_by_id.get(query_group_id)
        if group is None:
            _fail(
                f"query group {query_group_id!r} is absent from the sealed census; the "
                "predecessor request identity cannot be reused at this context"
            )
        if str(group.get("image_id")) != image_id:
            _fail(f"query group {query_group_id!r} belongs to another image")
        if _token_ids(
            group.get("query_suffix_token_ids"), label=f"query group {query_group_id!r}"
        ) != suffix_tokens or group.get("query_suffix_token_ids_sha256") != category.get(
            "query_suffix_token_ids_sha256"
        ):
            _fail(
                f"query group {query_group_id!r} carries a query suffix different from the "
                "sealed category registry"
            )
        query_group_ids.append(query_group_id)

    return {
        "category_query_id": category_query_id,
        "normalized_description": description,
        "category_token_ids": category_tokens,
        "category_token_ids_sha256": category.get("category_token_ids_sha256"),
        "query_suffix_token_ids": suffix_tokens,
        "query_suffix_token_ids_sha256": category.get("query_suffix_token_ids_sha256"),
        "token_source": "sealed_category_registry_query_suffix_never_retokenized",
        "shape": ["object_ref_start", "category_token_ids", "object_ref_end", "box_start"],
        "predecessor_query_group_ids": query_group_ids,
    }


def bind_inserted_clean_row(
    inputs: PlanInputs, gt_owner_id: str, description_path: Mapping[str, Any]
) -> dict[str, Any]:
    """``C`` = ``D_C`` + the owner's exact-GT-anchor coordinate tokens + ``<|box_end|>``."""

    candidate = inputs.exact_anchor_by_owner.get(gt_owner_id)
    if candidate is None:
        _fail(
            f"owner {gt_owner_id!r} has no exact_gt_anchor candidate in the sealed bank; the "
            "clean inserted row cannot be built without reconstructing coordinates"
        )
    coord_tokens = _token_ids(
        candidate.get("coord_token_ids"), label=f"exact anchor of owner {gt_owner_id!r}"
    )
    if len(coord_tokens) != 4:
        _fail(f"owner {gt_owner_id!r} exact anchor does not carry exactly four coordinate tokens")
    if sha256_json(coord_tokens) != candidate.get("coord_token_ids_sha256"):
        _fail(f"owner {gt_owner_id!r} exact anchor coordinate digest does not reconstruct")
    _validate_coordinate_tokens(inputs, coord_tokens, label=f"owner {gt_owner_id!r} exact anchor")
    if str(candidate.get("normalized_description")) != description_path["normalized_description"]:
        _fail(f"owner {gt_owner_id!r} exact anchor belongs to another category")

    token_ids = (
        list(description_path["query_suffix_token_ids"])
        + coord_tokens
        + [inputs.wrapper_token_ids["box_end"]]
    )
    return {
        "exact_gt_anchor_candidate_id": str(candidate.get("candidate_id")),
        "coord_token_ids": coord_tokens,
        "coord_token_ids_sha256": candidate.get("coord_token_ids_sha256"),
        "decoded_bbox_pixel_xyxy": candidate.get("decoded_bbox_pixel_xyxy"),
        "token_ids": token_ids,
        "token_ids_sha256": sha256_json(token_ids),
        "token_count": len(token_ids),
        "token_source": (
            "sealed_query_suffix_plus_sealed_exact_gt_anchor_coordinate_tokens_plus_box_end"
        ),
        "role": "oracle_intervention_never_a_natural_generation_claim",
    }


def _bind_unique_population(
    inputs: PlanInputs,
    *,
    gt_owner_id: str,
    description: str,
    context_ids: Sequence[str],
    population: Sequence[str],
) -> dict[str, Any]:
    """Prove the ``peak_lift`` population at every context this family is scored at.

    ``peak_lift`` is a statement about the collapsed unique physical candidates
    of one ``(image, context, category)`` query group, so the plan may not
    assert a population it has not checked against the sealed query-group
    registry at each context the scorer will actually use.
    """

    expected = set(population)
    query_group_ids: list[str] = []
    for context_id in context_ids:
        query_group_id = f"{context_id}|{description}"
        group = inputs.query_groups_by_id.get(query_group_id)
        if group is None:
            _fail(
                f"owner {gt_owner_id!r} has no sealed query group {query_group_id!r}; the unique "
                "peak_lift population at that context cannot be proven"
            )
        if str(group.get("status")) != "admitted":
            _fail(
                f"query group {query_group_id!r} is not admitted; a non-admitted group cannot "
                "own a support population"
            )
        declared = {str(value) for value in group.get("candidate_ids") or ()}
        if declared != expected:
            _fail(
                f"query group {query_group_id!r} population ({len(declared)} candidates) is not "
                f"the target-local plus same-category family this plan publishes "
                f"({len(expected)} candidates)"
            )
        declared_size = group.get("unique_coordinate_tuple_count")
        if declared_size is not None and int(declared_size) != len(declared):
            _fail(
                f"query group {query_group_id!r} declares {declared_size} unique coordinate "
                f"tuples but carries {len(declared)} collapsed candidates"
            )
        query_group_ids.append(query_group_id)
    ordered = sorted(expected)
    return {
        "candidate_ids": ordered,
        "candidate_ids_sha256": sha256_json(ordered),
        "unique_population_size": len(ordered),
        "scope": "image_context_normalized_description",
        "source": "sealed_census_query_group_registry",
        "verified_query_group_ids": query_group_ids,
        "role": "peak_lift_denominator_only_never_the_local_concentration_bank",
    }


def bind_candidate_families(
    inputs: PlanInputs,
    gt_owner_id: str,
    image_id: str,
    description: str,
    exact_anchor_id: str,
    *,
    context_ids: Sequence[str],
    support: Mapping[str, Any],
) -> dict[str, Any]:
    """The frozen target-local and same-category physical-owner families.

    The target-local family is the owner's seventeen fixed score-independent
    roles after alias collapse.  Each member carries its sealed strict
    assignment and the census partition it falls in, and the family publishes
    the exact U (ambiguity-included) and L (ambiguity-excluded) exclusion-
    filtered member lists, so the later scorer reproduces both bounds from
    sealed identities rather than from a rank, a margin, or its own matcher.
    """

    rows = inputs.candidates_by_image_description.get((image_id, description))
    if not rows:
        _fail(
            f"owner {gt_owner_id!r} has no candidate bank for image {image_id!r} / description "
            f"{description!r}"
        )
    target_local: list[str] = []
    competitor: list[str] = []
    competitor_owner_ids: set[str] = set()
    members: list[dict[str, Any]] = []
    by_bound: dict[str, list[str]] = {bound: [] for bound in SUPPORT_BOUND_PATHS}
    excluded: list[str] = []
    for row in rows:
        generators = [str(value) for value in row.get("generator_gt_owner_ids") or ()]
        candidate_id = str(row["candidate_id"])
        if gt_owner_id in generators:
            target_local.append(candidate_id)
            classified = planner.classify_candidate_for_owner(row, gt_owner_id)
            partition = str(classified["partition"])
            if partition not in SUPPORT_PARTITIONS:
                _fail(
                    f"candidate {candidate_id!r} falls in unknown support partition "
                    f"{partition!r} for owner {gt_owner_id!r}"
                )
            if classified["counts_toward_upper_bound"]:
                by_bound["u"].append(candidate_id)
            if classified["counts_toward_lower_bound"]:
                by_bound["l"].append(candidate_id)
            if classified["excluded_from_target_support"]:
                excluded.append(candidate_id)
            members.append(
                {
                    "candidate_id": candidate_id,
                    "coord_token_ids_sha256": row.get("coord_token_ids_sha256"),
                    "strict_assignment_status": row.get("strict_assignment_status"),
                    "strict_assignment_gt_owner_id": row.get("strict_assignment_gt_owner_id"),
                    "strict_assignment_scope": row.get("strict_assignment_scope"),
                    "ambiguity_owner_ids": [
                        str(value) for value in row.get("ambiguity_owner_ids") or ()
                    ],
                    "logical_transform_role": row.get("representative_role"),
                    "partition": partition,
                    "counts_toward_upper_bound_u": bool(classified["counts_toward_upper_bound"]),
                    "counts_toward_lower_bound_l": bool(classified["counts_toward_lower_bound"]),
                    "excluded_from_target_support": bool(
                        classified["excluded_from_target_support"]
                    ),
                    "partition_source": "frozen_census_classify_candidate_for_owner",
                }
            )
            continue
        if not generators:
            continue
        competitor.append(candidate_id)
        competitor_owner_ids.update(generators)
    if exact_anchor_id not in target_local:
        _fail(
            f"owner {gt_owner_id!r} exact anchor candidate is not inside its own target-local "
            "candidate family"
        )

    owner = _describe_owner(inputs, gt_owner_id, image_id)
    bank = owner.get("candidate_bank") or {}
    sealed_ids = bank.get("physical_candidate_ids")
    if sealed_ids is not None and sorted(str(value) for value in sealed_ids) != sorted(
        target_local
    ):
        _fail(
            f"owner {gt_owner_id!r} target-local family disagrees with the sealed "
            "candidate_bank.physical_candidate_ids the census scored as its neighbourhood"
        )
    if not by_bound["u"]:
        _fail(
            f"owner {gt_owner_id!r} has an empty U-bound exclusion-filtered target-local bank; "
            "target-local support could not be evaluated at any context"
        )
    if not by_bound["l"]:
        _fail(
            f"owner {gt_owner_id!r} has an empty L-bound exclusion-filtered target-local bank; "
            "the L sensitivity could not be evaluated at any context"
        )

    unique_population = _bind_unique_population(
        inputs,
        gt_owner_id=gt_owner_id,
        description=description,
        context_ids=context_ids,
        population=[*target_local, *competitor],
    )
    reference = _support_reference(support)
    return {
        FAMILY_TARGET_LOCAL: {
            "candidate_ids": target_local,
            "candidate_count": len(target_local),
            "candidate_ids_sha256": sha256_json(target_local),
            "exact_gt_anchor_candidate_id": exact_anchor_id,
            "logical_role_count": bank.get("logical_role_count"),
            "distinct_physical_candidate_count": bank.get("distinct_physical_candidate_count"),
            "bank_coverage_status": bank.get("bank_coverage_status"),
            "members": members,
            "bounds": {
                bound: {
                    "owner_context_path": SUPPORT_BOUND_PATHS[bound],
                    "candidate_ids": by_bound[bound],
                    "candidate_count": len(by_bound[bound]),
                    "candidate_ids_sha256": sha256_json(by_bound[bound]),
                    "role": (
                        "local_concentration_bank_and_peak_lift_best_subset_under_this_bound"
                    ),
                }
                for bound in sorted(SUPPORT_BOUND_PATHS)
            },
            "excluded_other_owner_strict_candidate_ids": excluded,
            "support_calibration": reference,
            "support_membership_source": "sealed_strict_assignment_never_a_rank_or_margin",
        },
        FAMILY_SAME_CATEGORY_OTHER_OWNER: {
            "candidate_ids": competitor,
            "candidate_count": len(competitor),
            "candidate_ids_sha256": sha256_json(competitor),
            "generator_gt_owner_ids": sorted(competitor_owner_ids),
            "support_role": (
                "enters_the_unique_peak_lift_population_only_never_the_target_local_"
                "concentration_bank"
            ),
            "support_calibration": reference,
        },
        "unique_population": unique_population,
        "collapse_scope": "image_and_normalized_description",
        "selection_uses_scores": False,
    }


# ---------------------------------------------------------------------------
# Cohort registry
# ---------------------------------------------------------------------------


def _describe_owner(inputs: PlanInputs, gt_owner_id: str, image_id: str) -> dict[str, Any]:
    owner = inputs.owners_by_id.get(gt_owner_id)
    if owner is None:
        _fail(f"owner {gt_owner_id!r} is absent from owner-registry.jsonl")
    if str(owner.get("image_id")) != image_id:
        _fail(f"owner {gt_owner_id!r} owner-registry image disagrees with its summary image")
    return owner


def build_cohort_rows(
    inputs: PlanInputs, cases: Sequence[CrossingCase], support: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """One sealed registry row per U-bound crossing owner."""

    rows: list[dict[str, Any]] = []
    for case in cases:
        owner = _describe_owner(inputs, case.gt_owner_id, case.image_id)
        description = str(owner["normalized_description"])
        boundary = case.boundary_index

        e_row = bind_native_row(
            inputs, case.image_id, boundary, label=f"owner {case.gt_owner_id!r} E row"
        )
        f_row: dict[str, Any] | None = None
        if context_id_for(case.image_id, boundary + 2) in inputs.census.contexts_by_id:
            f_row = bind_native_row(
                inputs, case.image_id, boundary + 1, label=f"owner {case.gt_owner_id!r} F row"
            )

        p_context = inputs.census.contexts_by_id[case.p_context_id]
        pe_context = inputs.census.contexts_by_id[case.pe_context_id]
        description_path = bind_target_description_path(
            inputs,
            case.gt_owner_id,
            case.image_id,
            description,
            (case.p_context_id, case.pe_context_id),
        )
        inserted_row = bind_inserted_clean_row(inputs, case.gt_owner_id, description_path)
        families = bind_candidate_families(
            inputs,
            case.gt_owner_id,
            case.image_id,
            description,
            inserted_row["exact_gt_anchor_candidate_id"],
            context_ids=(case.p_context_id, case.pe_context_id),
            support=support,
        )

        same_description = e_row["normalized_description"] == description
        stratum = (
            STRATUM_MATCHED_E
            if e_row["strict_match_status"] == SIDECAR_MATCHED
            else STRATUM_UNMATCHED_E
        )
        if e_row["strict_match_gt_owner_id"] == case.gt_owner_id:
            _fail(
                f"owner {case.gt_owner_id!r} is strict-matched by its own crossing row E; it is "
                "not a false negative at this boundary"
            )

        rows.append(
            {
                "schema_version": COHORT_SCHEMA_VERSION,
                "row_kind": "crossing_cohort_owner",
                "unit_id": UNIT_ID,
                "cohort": PRIMARY_COHORT,
                "gt_owner_id": case.gt_owner_id,
                "image_id": case.image_id,
                "normalized_description": description,
                "official_coco_category_id": owner.get("official_coco_category_id"),
                "owner_bbox_pixel_xyxy": owner.get("bbox_pixel_xyxy"),
                "owner_sort_key": owner.get("owner_sort_key"),
                "crossing": {
                    "boundary_index_b": boundary,
                    "boundary_convention": (
                        "boundary b contains native rows < b; E is sidecar row index b; "
                        "P+E is boundary b+1"
                    ),
                    "p_context_id": case.p_context_id,
                    "p_context_role": str(p_context.get("context_role")),
                    "p_boundary_index": int(p_context["boundary_index"]),
                    "p_prefix_token_count": len(
                        _token_ids(
                            p_context.get("generated_prefix_token_ids"),
                            label=f"context {case.p_context_id!r}",
                        )
                    ),
                    "p_prefix_token_ids_sha256": p_context.get("generated_prefix_token_ids_sha256"),
                    "p_passed_state": case.u_channel["passed_state"],
                    "pe_context_id": case.pe_context_id,
                    "pe_context_role": str(pe_context.get("context_role")),
                    "pe_boundary_index": int(pe_context["boundary_index"]),
                    "pe_prefix_token_count": len(
                        _token_ids(
                            pe_context.get("generated_prefix_token_ids"),
                            label=f"context {case.pe_context_id!r}",
                        )
                    ),
                    "pe_prefix_token_ids_sha256": pe_context.get(
                        "generated_prefix_token_ids_sha256"
                    ),
                    "pe_terminal_kind": pe_context.get("terminal_kind"),
                },
                "e_row": {**e_row, "stratum": stratum},
                "f_row": (
                    None
                    if f_row is None
                    else {
                        **f_row,
                        "post_e_boundary_context_id": context_id_for(case.image_id, boundary + 2),
                    }
                ),
                "f_row_present": f_row is not None,
                "native_next_action_at_p": native_next_action(inputs, case.image_id, boundary),
                "native_next_action_at_pe": native_next_action(
                    inputs, case.image_id, boundary + 1
                ),
                "bounds": {
                    "u_favorable_top3_supported_at_p": case.u_favorable,
                    "l_favorable_top3_supported_at_p": case.l_favorable,
                    "exact_same_context_u_and_l": case.u_favorable and case.l_favorable,
                    "u_channel": case.u_channel,
                    "l_channel": case.l_channel,
                    "primary_bound": "u",
                    "l_role": "sensitivity_never_replaces_the_primary_denominator",
                },
                "description_observability": (
                    OBSERVABILITY_SAME_DESCRIPTION
                    if same_description
                    else OBSERVABILITY_DIFFERENT_DESCRIPTION
                ),
                "same_description_as_e": same_description,
                "same_description_note": (
                    "P+D_C is construction-determined because D_C is already the exact prefix "
                    "of native row E; record it for replay only, never as displacement evidence"
                    if same_description
                    else None
                ),
                "target_description_path": description_path,
                "inserted_clean_row_c": inserted_row,
                "candidate_families": families,
                "branch_schema_id": BRANCH_SCHEMA_ID,
                "branch_order": list(BRANCH_ORDER),
                "request_ids": [],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def build_timing_controls(
    inputs: PlanInputs,
    cases: Sequence[CrossingCase],
    primary_owner_ids: frozenset[str],
    support: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Disjoint descriptive timing controls, deterministically selected.

    Every primary owner is excluded first.  A qualifying boundary is a
    U-favorable, calibrated-supported, before-or-at-frontier boundary that is
    *not* the owner's own crossing boundary and whose immediate next sidecar row
    strict-matches a physical owner.  The latest such boundary wins; a residual
    tie is broken by context ID.
    """

    crossing_by_owner = {case.gt_owner_id: case.boundary_index for case in cases}
    fn_owners = [
        row
        for row in inputs.census.owner_summaries
        if row.get("disposition") == merge.DISPOSITION_RESOLVED
    ]
    controls: list[dict[str, Any]] = []
    for summary in sorted(fn_owners, key=lambda row: str(row["gt_owner_id"])):
        gt_owner_id = str(summary["gt_owner_id"])
        if gt_owner_id in primary_owner_ids:
            continue
        image_id = str(summary["image_id"])
        rows = owner_context_rows(inputs, gt_owner_id, image_id)
        crossing_boundary = crossing_by_owner.get(gt_owner_id)
        qualifying: list[tuple[int, str]] = []
        for boundary_index in sorted(rows):
            if crossing_boundary is not None and boundary_index == crossing_boundary:
                continue
            context_id = context_id_for(image_id, boundary_index)
            favorable, _channel = favorable_supported(inputs, summary, context_id, bound="u")
            if not favorable:
                continue
            sidecar = inputs.sidecars_by_image_row.get((image_id, boundary_index))
            if sidecar is None or str(sidecar.get("strict_match_status")) != SIDECAR_MATCHED:
                continue
            if context_id_for(image_id, boundary_index + 1) not in inputs.census.contexts_by_id:
                continue
            qualifying.append((boundary_index, context_id))
        if not qualifying:
            continue
        boundary_index, context_id = max(qualifying, key=lambda item: (item[0], item[1]))
        owner = _describe_owner(inputs, gt_owner_id, image_id)
        description = str(owner["normalized_description"])
        next_context_id = context_id_for(image_id, boundary_index + 1)
        control_row = bind_native_row(
            inputs, image_id, boundary_index, label=f"timing control {gt_owner_id!r}"
        )
        _favorable, channel = favorable_supported(inputs, summary, context_id, bound="u")
        description_path = bind_target_description_path(
            inputs, gt_owner_id, image_id, description, (context_id, next_context_id)
        )
        inserted_row = bind_inserted_clean_row(inputs, gt_owner_id, description_path)
        families = bind_candidate_families(
            inputs,
            gt_owner_id,
            image_id,
            description,
            inserted_row["exact_gt_anchor_candidate_id"],
            context_ids=(context_id, next_context_id),
            support=support,
        )
        controls.append(
            {
                "schema_version": CONTROL_SCHEMA_VERSION,
                "row_kind": "crossing_control_owner",
                "unit_id": UNIT_ID,
                "cohort": TIMING_CONTROL_COHORT,
                "gt_owner_id": gt_owner_id,
                "image_id": image_id,
                "normalized_description": description,
                "disjoint_from_primary_cohort": True,
                "own_crossing_boundary_index": crossing_boundary,
                "selection_rule": (
                    "latest U-favorable calibrated-supported before-or-at-frontier noncrossing "
                    "boundary whose immediate next sidecar row strict-matches a physical owner; "
                    "residual ties broken by context id"
                ),
                "qualifying_boundary_indices": [item[0] for item in qualifying],
                "selected_boundary_index": boundary_index,
                "context_id": context_id,
                "next_context_id": next_context_id,
                "u_channel": channel,
                "next_row": control_row,
                "description_observability": (
                    OBSERVABILITY_SAME_DESCRIPTION
                    if control_row["normalized_description"] == description
                    else OBSERVABILITY_DIFFERENT_DESCRIPTION
                ),
                "target_description_path": description_path,
                "inserted_clean_row_c": inserted_row,
                "candidate_families": families,
                "role": (
                    "descriptive_only_timing_description_identity_and_route_tier_are_entangled"
                ),
                "request_ids": [],
            }
        )
    return controls


def _category_owner_count(inputs: PlanInputs, image_id: str, description: str) -> int:
    category = inputs.categories_by_query_id.get(f"{image_id}:{description}")
    if category is None:
        _fail(f"category {image_id}:{description} is absent from category-registry.jsonl")
    owner_ids = category.get("owner_ids")
    if owner_ids is None:
        _fail(f"category {image_id}:{description} declares no owner_ids")
    return len(owner_ids)


def build_tp_replay_controls(
    inputs: PlanInputs, primary_owner_ids: frozenset[str], support: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Exactly one due-boundary native true positive per image.

    Preference order is due-supported non-singleton, then due-supported, then
    any due owner; residual ties break by native row index and then owner ID.
    """

    with _predecessor_contract("native true-positive due-boundary map"):
        due_map = prevalence.build_due_context_map(inputs.census)
    by_image: dict[str, list[tuple[tuple[int, int, str], dict[str, Any]]]] = {}
    for summary in inputs.census.owner_summaries:
        if summary.get("native_true_positive") is not True:
            continue
        gt_owner_id = str(summary["gt_owner_id"])
        if gt_owner_id in primary_owner_ids:
            _fail(
                f"owner {gt_owner_id!r} is both a native true positive and a primary crossing "
                "owner; the cohorts are not disjoint"
            )
        due = due_map.get(gt_owner_id)
        if due is None:
            _fail(f"native true positive {gt_owner_id!r} has no sealed due boundary")
        image_id = str(due["image_id"])
        if str(summary["image_id"]) != image_id:
            _fail(f"native true positive {gt_owner_id!r} due boundary belongs to another image")
        owner = _describe_owner(inputs, gt_owner_id, image_id)
        description = str(owner["normalized_description"])
        due_context_id = str(due["due_context_id"])
        supported = due_context_id in _usable_support_context_ids(summary, "u")
        non_singleton = _category_owner_count(inputs, image_id, description) > 1
        if supported and non_singleton:
            tier = 0
        elif supported:
            tier = 1
        else:
            tier = 2
        entry = {
            "gt_owner_id": gt_owner_id,
            "image_id": image_id,
            "normalized_description": description,
            "due_context_id": due_context_id,
            "row_index": int(due["row_index"]),
            "pred_row_id": str(due["pred_row_id"]),
            "due_supported": supported,
            "non_singleton_category": non_singleton,
            "preference_tier": tier,
        }
        by_image.setdefault(image_id, []).append(
            ((tier, int(due["row_index"]), gt_owner_id), entry)
        )

    controls: list[dict[str, Any]] = []
    for image_id in sorted(by_image):
        _key, entry = min(by_image[image_id], key=lambda item: item[0])
        gt_owner_id = entry["gt_owner_id"]
        description = entry["normalized_description"]
        row_index = entry["row_index"]
        replaced_row = bind_native_row(
            inputs, image_id, row_index, label=f"tp replay control {gt_owner_id!r}"
        )
        if replaced_row["strict_match_gt_owner_id"] != gt_owner_id:
            _fail(
                f"tp replay control {gt_owner_id!r} due row does not strict-match the control owner"
            )
        following_action = native_next_action(inputs, image_id, row_index + 1)
        description_path = bind_target_description_path(
            inputs, gt_owner_id, image_id, description, (entry["due_context_id"],)
        )
        inserted_row = bind_inserted_clean_row(inputs, gt_owner_id, description_path)
        families = bind_candidate_families(
            inputs,
            gt_owner_id,
            image_id,
            description,
            inserted_row["exact_gt_anchor_candidate_id"],
            context_ids=(entry["due_context_id"],),
            support=support,
        )
        controls.append(
            {
                "schema_version": CONTROL_SCHEMA_VERSION,
                "row_kind": "crossing_control_owner",
                "unit_id": UNIT_ID,
                "cohort": TP_REPLAY_CONTROL_COHORT,
                **entry,
                "disjoint_from_primary_cohort": True,
                "selection_rule": (
                    "one exact due-boundary native true positive per image, preferring a "
                    "due-supported non-singleton owner, then breaking ties by native row index "
                    "and owner id"
                ),
                "image_candidate_count": len(by_image[image_id]),
                "replaced_native_row": replaced_row,
                "following_native_action": following_action,
                "target_description_path": description_path,
                "inserted_clean_row_c": inserted_row,
                "candidate_families": families,
                "role": (
                    "coordinate_replay_calibration_and_benign_substitution_reference_only"
                ),
                "request_ids": [],
            }
        )
    return controls


# ---------------------------------------------------------------------------
# Request plan
# ---------------------------------------------------------------------------


def _request_id(payload: Mapping[str, Any]) -> str:
    return f"req:{sha256_json(payload)[:32]}"


def _make_request(
    *,
    family: str,
    cohort: str,
    gt_owner_id: str,
    image_id: str,
    context_id: str,
    context_role: str,
    boundary_index: int,
    appended_token_ids: Sequence[int],
    appended_role: str,
    base_prefix_token_count: int,
    base_prefix_token_ids_sha256: Any,
    scored_target: Mapping[str, Any],
    readout_tier: str,
    branch_inputs: Sequence[str],
    decode: Mapping[str, Any] | None = None,
    candidate_family: Mapping[str, Any] | None = None,
    predecessor_query_group_id: str | None = None,
    variant: str = "primary",
    optional: bool = False,
) -> dict[str, Any]:
    appended = list(appended_token_ids)
    identity = {
        "unit_id": UNIT_ID,
        "request_family": family,
        "cohort": cohort,
        "gt_owner_id": gt_owner_id,
        "context_id": context_id,
        "variant": variant,
        "appended_token_ids": appended,
        "base_prefix_token_ids_sha256": base_prefix_token_ids_sha256,
        "scored_target": dict(scored_target),
        "decode": None if decode is None else dict(decode),
        "candidate_family": None if candidate_family is None else dict(candidate_family),
    }
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "row_kind": "crossing_boundary_request",
        "unit_id": UNIT_ID,
        "request_id": _request_id(identity),
        "request_key": f"{family}|{cohort}|{gt_owner_id}|{context_id}|{variant}",
        "request_family": family,
        "cohort": cohort,
        "variant": variant,
        "optional": optional,
        "readout_tier": readout_tier,
        "gt_owner_id": gt_owner_id,
        "image_id": image_id,
        "context_id": context_id,
        "context_role": context_role,
        "boundary_index": boundary_index,
        "prefix": {
            "base_context_id": context_id,
            "base_prefix_token_count": base_prefix_token_count,
            "base_prefix_token_ids_sha256": base_prefix_token_ids_sha256,
            "appended_token_ids": appended,
            "appended_token_ids_sha256": sha256_json(appended),
            "appended_token_count": len(appended),
            "appended_role": appended_role,
            "retokenized": False,
        },
        "scored_target": dict(scored_target),
        "decode": None if decode is None else dict(decode),
        "candidate_family": None if candidate_family is None else dict(candidate_family),
        "predecessor_query_group_id": predecessor_query_group_id,
        "branch_inputs": list(branch_inputs),
        "branch_schema_id": BRANCH_SCHEMA_ID,
        "score_blind_plan": True,
        "inspects_new_model_logits": False,
        "identity_digest": sha256_json(identity),
    }


def _greedy_decode_spec(inputs: PlanInputs) -> dict[str, Any]:
    return {
        "mode": "deterministic_greedy_no_generate_explicit_position_ids",
        "grammar": {
            "coordinate_token_id_start": inputs.coordinate_token_ids["start"],
            "coordinate_token_id_end_inclusive": inputs.coordinate_token_ids["end_inclusive"],
            "coordinate_token_count": 4,
            "terminal_token_id": inputs.wrapper_token_ids["box_end"],
            "max_new_tokens": 5,
        },
        "on_grammar_violation": "malformed_never_silently_repaired",
        "sampling": "disabled_unless_the_optional_sampling_receipt_contract_is_implemented",
    }


def _context_identity(inputs: PlanInputs, context_id: str) -> tuple[str, int, int, Any]:
    context = inputs.census.contexts_by_id.get(context_id)
    if context is None:
        _fail(f"context {context_id!r} is absent from context-registry.jsonl")
    tokens = _token_ids(
        context.get("generated_prefix_token_ids"), label=f"context {context_id!r}"
    )
    return (
        str(context.get("context_role")),
        int(context["boundary_index"]),
        len(tokens),
        context.get("generated_prefix_token_ids_sha256"),
    )


def _ladder_requests(
    inputs: PlanInputs,
    *,
    cohort: str,
    gt_owner_id: str,
    image_id: str,
    context_id: str,
    variant: str,
    description_path: Mapping[str, Any],
    native_action: Mapping[str, Any],
    families: Mapping[str, Any],
    include_release: bool,
) -> list[dict[str, Any]]:
    """The two measurement ladders at one context, as concrete requests."""

    context_role, boundary_index, token_count, token_digest = _context_identity(inputs, context_id)
    query_group_id = f"{context_id}|{description_path['normalized_description']}"
    requests: list[dict[str, Any]] = []
    common = {
        "cohort": cohort,
        "gt_owner_id": gt_owner_id,
        "image_id": image_id,
        "context_id": context_id,
        "context_role": context_role,
        "boundary_index": boundary_index,
        "base_prefix_token_count": token_count,
        "base_prefix_token_ids_sha256": token_digest,
        "variant": variant,
    }
    if include_release:
        requests.append(
            _make_request(
                family=REQUEST_NATURAL_RELEASE,
                appended_token_ids=(),
                appended_role="none_scored_at_the_bare_context",
                scored_target={
                    "kind": "target_description_path",
                    "token_ids": list(description_path["query_suffix_token_ids"]),
                    "token_ids_sha256": description_path["query_suffix_token_ids_sha256"],
                    "teacher_forced": True,
                    "record": [
                        "gate_margin_when_native_alternative_is_stop",
                        "first_divergent_token_target_versus_native_margin",
                        "complete_description_path_sum_and_token_mean",
                        "argmax_follows_target_through_every_observable_description_token",
                    ],
                },
                readout_tier="primary",
                branch_inputs=["release_lost"],
                predecessor_query_group_id=query_group_id,
                **common,
            )
        )
        requests.append(
            _make_request(
                family=REQUEST_NATIVE_NEXT_ACTION,
                appended_token_ids=(),
                appended_role="none_scored_at_the_bare_context",
                scored_target={
                    "kind": native_action["kind"],
                    "token_ids": list(native_action["token_ids"]),
                    "token_ids_sha256": native_action["token_ids_sha256"],
                    "native_row_index": native_action["row_index"],
                    "terminal_kind": native_action["terminal_kind"],
                    "teacher_forced": True,
                    "role": "replay_alignment_and_native_alternative_margin",
                },
                readout_tier="primary",
                branch_inputs=["replay_admission", "release_lost"],
                predecessor_query_group_id=query_group_id,
                **common,
            )
        )
    population = families["unique_population"]
    if query_group_id not in population["verified_query_group_ids"]:
        _fail(
            f"query group {query_group_id!r} was never verified as the unique support population "
            f"for owner {gt_owner_id!r}; the request would assert an unproven peak_lift denominator"
        )
    for family_name, request_family, branch_inputs in (
        (FAMILY_TARGET_LOCAL, REQUEST_COORDINATE_TARGET_LOCAL, ["realization_fail", "displaced"]),
        (
            FAMILY_SAME_CATEGORY_OTHER_OWNER,
            REQUEST_COORDINATE_COMPETITOR,
            ["displaced", "realization_fail"],
        ),
    ):
        family = families[family_name]
        requests.append(
            _make_request(
                family=request_family,
                appended_token_ids=description_path["query_suffix_token_ids"],
                appended_role="forced_target_description_path_d_c",
                scored_target={
                    "kind": "coordinate_candidate_family",
                    "family": family_name,
                    "candidate_ids_sha256": family["candidate_ids_sha256"],
                    "candidate_count": family["candidate_count"],
                    "shape": "four_coordinate_tokens_then_box_end",
                    "terminal_token_id": inputs.wrapper_token_ids["box_end"],
                    "support_evaluation": {
                        "support_calibration": family["support_calibration"],
                        "unique_population": {
                            "query_group_id": query_group_id,
                            "candidate_ids_sha256": population["candidate_ids_sha256"],
                            "unique_population_size": population["unique_population_size"],
                            "scope": population["scope"],
                        },
                        "target_local_bounds": (
                            {
                                bound: {
                                    "owner_context_path": block["owner_context_path"],
                                    "candidate_ids_sha256": block["candidate_ids_sha256"],
                                    "candidate_count": block["candidate_count"],
                                }
                                for bound, block in family["bounds"].items()
                            }
                            if family_name == FAMILY_TARGET_LOCAL
                            else None
                        ),
                        "family_role": (
                            "target_local_support_bank_and_peak_lift_numerator"
                            if family_name == FAMILY_TARGET_LOCAL
                            else "peak_lift_population_member_only_never_target_support"
                        ),
                        "rank_is_a_support_input": False,
                        "margin_is_a_support_input": False,
                    },
                },
                readout_tier="primary",
                branch_inputs=branch_inputs,
                candidate_family=family,
                predecessor_query_group_id=query_group_id,
                **common,
            )
        )
    requests.append(
        _make_request(
            family=REQUEST_COORDINATE_GREEDY,
            appended_token_ids=description_path["query_suffix_token_ids"],
            appended_role="forced_target_description_path_d_c",
            scored_target={
                "kind": "greedy_coordinate_row",
                "shape": "four_coordinate_tokens_then_box_end",
                "owner_match": "predecessor_one_row_strict_physical_owner_matcher",
            },
            decode=_greedy_decode_spec(inputs),
            readout_tier="primary",
            branch_inputs=["displaced", "realization_fail"],
            predecessor_query_group_id=query_group_id,
            **common,
        )
    )
    return requests


def build_requests(
    inputs: PlanInputs,
    cohort_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Every request this unit will execute, with stable identities."""

    requests: list[dict[str, Any]] = []
    for row in cohort_rows:
        gt_owner_id = str(row["gt_owner_id"])
        image_id = str(row["image_id"])
        description_path = row["target_description_path"]
        families = row["candidate_families"]
        for variant, context_id, native_action in (
            ("at_p", row["crossing"]["p_context_id"], row["native_next_action_at_p"]),
            ("at_p_plus_e", row["crossing"]["pe_context_id"], row["native_next_action_at_pe"]),
        ):
            requests.extend(
                _ladder_requests(
                    inputs,
                    cohort=PRIMARY_COHORT,
                    gt_owner_id=gt_owner_id,
                    image_id=image_id,
                    context_id=context_id,
                    variant=variant,
                    description_path=description_path,
                    native_action=native_action,
                    families=families,
                    include_release=True,
                )
            )
        requests.append(
            _compatibility_request(
                inputs,
                cohort=PRIMARY_COHORT,
                gt_owner_id=gt_owner_id,
                image_id=image_id,
                context_id=row["crossing"]["p_context_id"],
                inserted_row=row["inserted_clean_row_c"],
                scored_row=row["e_row"],
                variant="p_plus_c_then_e",
                optional=False,
            )
        )
        if row["f_row_present"]:
            requests.append(
                _compatibility_request(
                    inputs,
                    cohort=PRIMARY_COHORT,
                    gt_owner_id=gt_owner_id,
                    image_id=image_id,
                    context_id=row["crossing"]["pe_context_id"],
                    inserted_row=row["inserted_clean_row_c"],
                    scored_row=row["f_row"],
                    variant="p_plus_e_plus_c_then_f",
                    optional=True,
                )
            )

    for row in control_rows:
        gt_owner_id = str(row["gt_owner_id"])
        image_id = str(row["image_id"])
        if row["cohort"] == TIMING_CONTROL_COHORT:
            for variant, context_id in (
                ("at_control_boundary", row["context_id"]),
                ("at_control_boundary_plus_row", row["next_context_id"]),
            ):
                boundary_index = int(context_id.rsplit("-", 1)[1])
                requests.extend(
                    _ladder_requests(
                        inputs,
                        cohort=TIMING_CONTROL_COHORT,
                        gt_owner_id=gt_owner_id,
                        image_id=image_id,
                        context_id=context_id,
                        variant=variant,
                        description_path=row["target_description_path"],
                        native_action=native_next_action(inputs, image_id, boundary_index),
                        families=row["candidate_families"],
                        include_release=True,
                    )
                )
            continue
        requests.extend(
            _ladder_requests(
                inputs,
                cohort=TP_REPLAY_CONTROL_COHORT,
                gt_owner_id=gt_owner_id,
                image_id=image_id,
                context_id=str(row["due_context_id"]),
                variant="at_due_boundary",
                description_path=row["target_description_path"],
                native_action=native_next_action(inputs, image_id, int(row["row_index"])),
                families=row["candidate_families"],
                include_release=False,
            )
        )
        requests.append(
            _compatibility_request(
                inputs,
                cohort=TP_REPLAY_CONTROL_COHORT,
                gt_owner_id=gt_owner_id,
                image_id=image_id,
                context_id=str(row["due_context_id"]),
                inserted_row=row["inserted_clean_row_c"],
                scored_row=row["following_native_action"],
                variant="benign_substitution_then_following_native_action",
                optional=False,
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


def _compatibility_request(
    inputs: PlanInputs,
    *,
    cohort: str,
    gt_owner_id: str,
    image_id: str,
    context_id: str,
    inserted_row: Mapping[str, Any],
    scored_row: Mapping[str, Any],
    variant: str,
    optional: bool,
) -> dict[str, Any]:
    context_role, boundary_index, token_count, token_digest = _context_identity(inputs, context_id)
    if "full_row_token_ids" in scored_row:
        scored_target = {
            "kind": "exact_native_row",
            "token_ids": list(scored_row["full_row_token_ids"]),
            "token_ids_sha256": scored_row["full_row_token_ids_sha256"],
            "native_row_index": scored_row["row_index"],
            "compare_against": "the same exact row scored at the unmodified native context",
            "report": ["description_delta", "coordinate_delta", "complete_row_delta"],
        }
    else:
        scored_target = {
            "kind": scored_row["kind"],
            "token_ids": list(scored_row["token_ids"]),
            "token_ids_sha256": scored_row["token_ids_sha256"],
            "native_row_index": scored_row["row_index"],
            "compare_against": "the same exact action scored at the unmodified native context",
            "report": ["description_delta", "coordinate_delta", "complete_row_delta"],
        }
    return _make_request(
        family=REQUEST_DOWNSTREAM_COMPATIBILITY,
        cohort=cohort,
        gt_owner_id=gt_owner_id,
        image_id=image_id,
        context_id=context_id,
        context_role=context_role,
        boundary_index=boundary_index,
        appended_token_ids=inserted_row["token_ids"],
        appended_role="inserted_exact_clean_gt_row_c",
        base_prefix_token_count=token_count,
        base_prefix_token_ids_sha256=token_digest,
        scored_target=scored_target,
        readout_tier="secondary_sealed_after_primary_branches",
        branch_inputs=[],
        variant=variant,
        optional=optional,
    )


# ---------------------------------------------------------------------------
# Cohort denominators
# ---------------------------------------------------------------------------


def _check_count(actual: int, expected: int, *, label: str) -> None:
    if actual != expected:
        _fail(
            f"{label} is {actual}, not the frozen {expected}; the sealed census no longer "
            "reproduces this unit's denominator and the plan fails closed"
        )


def derive_cohort_counts(
    cases: Sequence[CrossingCase], cohort_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Re-derive and enforce every frozen denominator before request construction."""

    u_cases = [case for case in cases if case.u_favorable]
    l_cases = [case for case in cases if case.l_favorable]
    ul_cases = [case for case in cases if case.u_favorable and case.l_favorable]
    _check_count(len(u_cases), EXPECTED_U_CROSSING_COUNT, label="the U-bound crossing cohort")
    _check_count(len(l_cases), EXPECTED_L_CROSSING_COUNT, label="the L-bound crossing cohort")
    _check_count(
        len(ul_cases),
        EXPECTED_EXACT_UL_CROSSING_COUNT,
        label="the exact same-context U and L crossing cohort",
    )

    strata = Counter(str(row["e_row"]["stratum"]) for row in cohort_rows)
    _check_count(
        strata.get(STRATUM_MATCHED_E, 0), EXPECTED_MATCHED_E_COUNT, label="the matched-E stratum"
    )
    _check_count(
        strata.get(STRATUM_UNMATCHED_E, 0),
        EXPECTED_UNMATCHED_E_COUNT,
        label="the unmatched-E stratum",
    )
    if len(cohort_rows) != len(u_cases):
        _fail("the cohort registry does not have exactly one row per U-bound crossing owner")

    observability = Counter(str(row["description_observability"]) for row in cohort_rows)
    return {
        "all_resolved_owner_crossing_count": len(cases),
        "u_bound_crossing_count": len(u_cases),
        "l_bound_crossing_count": len(l_cases),
        "exact_same_context_u_and_l_count": len(ul_cases),
        "matched_e_count": strata.get(STRATUM_MATCHED_E, 0),
        "unmatched_e_count": strata.get(STRATUM_UNMATCHED_E, 0),
        "same_description_count": observability.get(OBSERVABILITY_SAME_DESCRIPTION, 0),
        "different_description_count": observability.get(OBSERVABILITY_DIFFERENT_DESCRIPTION, 0),
        "f_row_present_count": sum(1 for row in cohort_rows if row["f_row_present"]),
        "terminal_p_plus_e_count": sum(
            1 for row in cohort_rows if row["crossing"]["pe_context_role"] == "terminal"
        ),
        "per_image_owner_counts": dict(
            sorted(Counter(str(row["image_id"]) for row in cohort_rows).items())
        ),
        "expected": {
            "u_bound_crossing_count": EXPECTED_U_CROSSING_COUNT,
            "l_bound_crossing_count": EXPECTED_L_CROSSING_COUNT,
            "exact_same_context_u_and_l_count": EXPECTED_EXACT_UL_CROSSING_COUNT,
            "matched_e_count": EXPECTED_MATCHED_E_COUNT,
            "unmatched_e_count": EXPECTED_UNMATCHED_E_COUNT,
            "timing_control_count": EXPECTED_TIMING_CONTROL_COUNT,
            "tp_replay_control_count": EXPECTED_TP_REPLAY_CONTROL_COUNT,
        },
    }


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(payload)
    return {"path": path.name, "byte_size": len(payload), "sha256": sha256_bytes(payload), "row_count": len(rows)}


def seal_input_files(
    root: Path, sha256_by_relative_name: Mapping[str, str], *, label: str
) -> dict[str, dict[str, Any]]:
    """A complete ``{path, byte_size, sha256}`` seal for every read input file.

    The bytes are re-read here rather than stat-ed, so the sealed byte size and
    digest always describe the same observation, and a file that changed
    between validation and sealing fails closed instead of being sealed under a
    digest it no longer has.
    """

    seals: dict[str, dict[str, Any]] = {}
    for relative_name, declared_sha256 in sorted(sha256_by_relative_name.items()):
        path = Path(root) / relative_name
        if not path.is_file():
            _fail(f"{label} input file {relative_name} disappeared before it could be sealed")
        payload = path.read_bytes()
        digest = sha256_bytes(payload)
        if digest != declared_sha256:
            _fail(
                f"{label} input file {relative_name} changed on disk between validation and "
                "sealing; the lineage would not describe the bytes that were read"
            )
        seals[relative_name] = {
            "path": relative_name,
            "byte_size": len(payload),
            "sha256": digest,
        }
    return seals


def _builder_source_seal() -> dict[str, Any]:
    """The builder's own source identity, sealed like every other input."""

    path = Path(__file__).resolve()
    payload = path.read_bytes()
    try:
        relative_path = str(path.relative_to(REPO_ROOT))
    except ValueError:  # pragma: no cover - the module always lives under the repo
        relative_path = path.name
    return {"path": relative_path, "byte_size": len(payload), "sha256": sha256_bytes(payload)}


def build_plan(
    *,
    prevalence_run_root: Path,
    output_root: Path,
    census_run_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Derive, validate, and seal the complete CPU plan.  Returns the manifest."""

    output_root = Path(output_root)
    plan_dir = output_root / PLAN_DIR_NAME
    if plan_dir.exists() and any(plan_dir.iterdir()) and not overwrite:
        _fail(
            f"output plan directory {plan_dir} already exists and is not empty; pass "
            "--overwrite to replace a previous build"
        )

    inputs = load_plan_inputs(prevalence_run_root, census_run_root)
    validation = validate_plan_inputs(inputs)
    support = bind_support_calibration(inputs)

    cases = derive_crossing_cases(inputs)
    primary_cases = [case for case in cases if case.u_favorable]
    cohort_rows = build_cohort_rows(inputs, primary_cases, support)
    counts = derive_cohort_counts(cases, cohort_rows)

    primary_owner_ids = frozenset(str(row["gt_owner_id"]) for row in cohort_rows)
    timing_controls = build_timing_controls(inputs, cases, primary_owner_ids, support)
    _check_count(
        len(timing_controls),
        EXPECTED_TIMING_CONTROL_COUNT,
        label="the disjoint timing-control registry",
    )
    tp_controls = build_tp_replay_controls(inputs, primary_owner_ids, support)
    _check_count(
        len(tp_controls),
        EXPECTED_TP_REPLAY_CONTROL_COUNT,
        label="the native true-positive replay-control registry",
    )
    control_rows = timing_controls + tp_controls
    control_owner_ids = frozenset(str(row["gt_owner_id"]) for row in control_rows)
    overlap = sorted(primary_owner_ids & control_owner_ids)
    if overlap:
        _fail(
            f"control owners {overlap!r} also appear in the primary cohort; the control "
            "registry is not disjoint and is never patched by allowing overlap"
        )
    if len(control_owner_ids) != len(control_rows):
        _fail("the control registry carries more than one row per control owner")

    requests = build_requests(inputs, cohort_rows, control_rows)
    requests_by_owner: dict[str, list[str]] = {}
    for request in requests:
        requests_by_owner.setdefault(str(request["gt_owner_id"]), []).append(
            str(request["request_id"])
        )
    for row in list(cohort_rows) + list(control_rows):
        row["request_ids"] = sorted(requests_by_owner.get(str(row["gt_owner_id"]), ()))
        if not row["request_ids"]:
            _fail(f"owner {row['gt_owner_id']!r} is registered but has no request")

    plan_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        COHORT_REGISTRY_NAME: _write_jsonl(plan_dir / COHORT_REGISTRY_NAME, cohort_rows),
        CONTROL_REGISTRY_NAME: _write_jsonl(plan_dir / CONTROL_REGISTRY_NAME, control_rows),
        REQUEST_PLAN_NAME: _write_jsonl(plan_dir / REQUEST_PLAN_NAME, requests),
    }

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "builder_source": _builder_source_seal(),
        "lineage": {
            "seal_fields": list(SEAL_FIELDS),
            "prevalence_run_root": str(inputs.prevalence_run_root),
            "prevalence_unit_id": PREVALENCE_UNIT_ID,
            "prevalence_receipt_content_sha256": inputs.prevalence_receipt.get(
                "receipt_content_sha256"
            ),
            "prevalence_input_files": seal_input_files(
                inputs.prevalence_run_root,
                inputs.prevalence_file_sha256,
                label="prevalence run",
            ),
            "census_run_root": str(inputs.census_run_root),
            "census_unit_id": CENSUS_UNIT_ID,
            "census_plan_receipt_content_sha256": inputs.census.plan_receipt.get(
                "receipt_content_sha256"
            ),
            "census_merge_receipt_content_sha256": inputs.census.merge_receipt.get(
                "receipt_content_sha256"
            ),
            "census_input_files": seal_input_files(
                inputs.census_run_root, inputs.census_file_sha256, label="census run"
            ),
        },
        "validation": validation,
        "support_calibration": support,
        "cohort_counts": counts,
        "control_counts": {
            "timing_control_count": len(timing_controls),
            "tp_replay_control_count": len(tp_controls),
            "timing_control_owner_ids": sorted(
                str(row["gt_owner_id"]) for row in timing_controls
            ),
            "tp_replay_control_owner_ids": sorted(str(row["gt_owner_id"]) for row in tp_controls),
            "disjoint_from_primary_cohort": True,
        },
        "request_counts": {
            "total": len(requests),
            "by_family": dict(sorted(Counter(str(r["request_family"]) for r in requests).items())),
            "by_cohort": dict(sorted(Counter(str(r["cohort"]) for r in requests).items())),
            "optional": sum(1 for r in requests if r["optional"]),
        },
        "token_identity_contract": {
            "wrapper_token_ids": dict(sorted(inputs.wrapper_token_ids.items())),
            "coordinate_token_ids": dict(sorted(inputs.coordinate_token_ids.items())),
            "retokenization": "forbidden_every_token_id_is_literal",
            "full_row_token_ownership": (
                "literal_adjacent_context_prefix_suffix_only_never_the_sidecar"
            ),
            "sidecar_role": (
                "validates_coordinate_subsequence_digest_and_declared_row_metadata_only"
            ),
            "coordinate_grammar": "exactly_four_coordinate_tokens_then_box_end",
        },
        "branch_schema": {
            "branch_schema_id": BRANCH_SCHEMA_ID,
            "exhaustive_order": list(BRANCH_ORDER),
            "sub_tags": {key: list(value) for key, value in BRANCH_SUB_TAGS.items()},
            "evaluated_here": False,
            "evaluation_owner": "the later GPU scoring and analysis pass",
        },
        "request_families": list(REQUEST_FAMILIES),
        "score_input_policy": {
            "loads_model_or_tokenizer": False,
            "inspects_new_model_logits": False,
            "reads": "sealed_predecessor_plan_registries_and_sealed_owner_context_features_only",
            "cohort_selection_uses_new_scores": False,
        },
        "output_file_digests": {name: value for name, value in sorted(outputs.items())},
        "semantics_notes": list(SEMANTICS_NOTES),
    }
    manifest["manifest_content_sha256"] = sha256_json(manifest)
    manifest_bytes = canonical_json_bytes(manifest) + b"\n"
    (plan_dir / MANIFEST_NAME).write_bytes(manifest_bytes)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prevalence-run-root",
        default=str(DEFAULT_PREVALENCE_RUN_ROOT),
        help="immutable predecessor prevalence run root (default: the sealed 20260803T231934Z run)",
    )
    parser.add_argument(
        "--census-run-root",
        default=None,
        help=(
            "immutable census run root; defaults to the predecessor_run_root sealed in the "
            "prevalence receipt"
        ),
    )
    parser.add_argument("--output-root", required=True, help="local run directory to write")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a previous build in the output plan directory",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = build_plan(
            prevalence_run_root=Path(args.prevalence_run_root),
            census_run_root=None if args.census_run_root is None else Path(args.census_run_root),
            output_root=Path(args.output_root),
            overwrite=bool(args.overwrite),
        )
    except PlanContractError as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        return 1
    counts = manifest["cohort_counts"]
    print(
        "sealed CPU plan: "
        f"U={counts['u_bound_crossing_count']} L={counts['l_bound_crossing_count']} "
        f"U&L={counts['exact_same_context_u_and_l_count']} "
        f"matched-E={counts['matched_e_count']} unmatched-E={counts['unmatched_e_count']} "
        f"timing-controls={manifest['control_counts']['timing_control_count']} "
        f"tp-controls={manifest['control_counts']['tp_replay_control_count']} "
        f"requests={manifest['request_counts']['total']} "
        f"calibration={manifest['support_calibration']['source']['calibration_sha256'][:12]}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
