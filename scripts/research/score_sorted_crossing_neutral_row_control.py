#!/usr/bin/env python3
"""Same-run paired capture for the sorted crossing matched-length neutral-row
insertion control
(``2026-08-04-sorted-crossing-matched-length-neutral-row-insertion-control``).

Frozen unit::

    research/investigations/qwen3-vl-dense-enumeration/experiments/
    2026-08-04-sorted-crossing-matched-length-neutral-row-insertion-control/unit.md

What this module is
-------------------
``unit.md`` "Requests and score semantics" executes 21 owners and three sealed
arms in **one** runtime, so the ``N`` contrast, the ``C`` positive control and
the benign reference are comparable without crossing a run boundary:

``p_plus_n_then_e`` (21)
    Append the deterministic neutral control row ``N`` to the native ``P``, then
    teacher-force the exact sealed downstream row ``E``.
``p_plus_c_then_e`` (21)
    Append the exact clean skipped-owner row ``C`` to the same ``P`` and force
    the *same* sealed ``E`` tokens.  This is the same-run positive control.
``benign_substitution_then_following_native_action`` (12)
    Append the exact clean GT twin of a natively emitted true positive to that
    row's due-boundary predecessor and force the exact following native row.

Both roots of every pair force byte-identical target tokens, so every reported
number is a within-owner, within-target paired quantity.  Because the two
crossing arms share one sealed ``E`` row, the scored token ids and digest are
identical between them by construction and are re-proven per shard.

What this module is deliberately **not**
----------------------------------------
* It never decodes freely, decodes coordinates greedily, samples, calls
  ``model.generate()`` or touches image ``2299``.
* It never computes a relative delta, a materiality, a specificity gate or a
  route: those belong to the analysis pass, after merge-time validation.
* It never re-implements a Qwen runtime: teacher-forcing, root payloads,
  segment sums, paired deltas, parity and quarantine are imported from the
  predecessor crossing scorers.

Outputs (one explicit shard directory, published atomically)::

    neutral-row-control-rows.jsonl      one row per executed request
    neutral-row-control-parity.json     backend + cached/uncached parity seam
    neutral-row-control-receipt.json    identities, digests, counters, policy

A ``--mode smoke`` shard publishes ``neutral-row-control-admission.json`` beside
its parity and receipt and no evidence rows.  It runs only on an image whose
sealed selection rows cover all four strata, and it seals the representative
owner and request ids of each one; a capture inherits nothing less.  A shard
stopped by the quarantine
rule publishes ``neutral-row-control-quarantine.json`` and its receipt alone: on
mismatch there is no partial evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import prepare_sorted_crossing_neutral_row_control as plan_builder  # noqa: E402
from scripts.research import (  # noqa: E402
    score_sorted_crossing_boundary_owner_release as crossing_scorer,
)
from scripts.research import (  # noqa: E402
    score_sorted_crossing_boundary_owner_release_secondary as paired,
)

# ---------------------------------------------------------------------------
# 0. Frozen schema / arm / artifact constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "sorted_crossing_neutral_row_control_rows.v1"
RECEIPT_SCHEMA_VERSION = "sorted_crossing_neutral_row_control_receipt.v1"
PARITY_SCHEMA_VERSION = "sorted_crossing_neutral_row_control_parity.v1"
QUARANTINE_SCHEMA_VERSION = "sorted_crossing_neutral_row_control_quarantine.v1"
ADMISSION_SCHEMA_VERSION = "sorted_crossing_neutral_row_control_admission.v1"

UNIT_ID = plan_builder.UNIT_ID

ROWS_NAME = "neutral-row-control-rows.jsonl"
PARITY_NAME = "neutral-row-control-parity.json"
RECEIPT_NAME = "neutral-row-control-receipt.json"
QUARANTINE_NAME = "neutral-row-control-quarantine.json"
ADMISSION_NAME = "neutral-row-control-admission.json"

#: Files that are *evidence*.  A quarantined shard leaves none of them.
EVIDENCE_OUTPUT_NAMES: tuple[str, ...] = (ROWS_NAME, PARITY_NAME)

MODE_SMOKE = crossing_scorer.MODE_SMOKE
MODE_CAPTURE = crossing_scorer.MODE_CAPTURE

ARM_NEUTRAL = plan_builder.ARM_NEUTRAL
ARM_CLEAN_REPLAY = plan_builder.ARM_CLEAN_REPLAY
ARM_BENIGN = plan_builder.ARM_BENIGN
ARMS: tuple[str, ...] = plan_builder.ARMS
EXPECTED_REQUEST_COUNT_BY_ARM = plan_builder.EXPECTED_REQUEST_COUNT_BY_ARM
TOTAL_REQUEST_COUNT = plan_builder.TOTAL_REQUEST_COUNT

#: The two crossing arms that must score one identical sealed ``E`` row.
PAIRED_CROSSING_ARMS: tuple[str, ...] = (ARM_NEUTRAL, ARM_CLEAN_REPLAY)

#: ``unit.md`` "Representative smoke": the four strata one real smoke image must
#: exercise, taken from the plan vocabulary so the two cannot drift.
REQUIRED_SMOKE_STRATA: tuple[str, ...] = plan_builder.REQUIRED_SMOKE_STRATA

#: The sealed cohort role each arm may belong to.
ARM_COHORT_ROLES: Mapping[str, frozenset[str]] = {
    ARM_NEUTRAL: frozenset({plan_builder.COHORT_VOTING, plan_builder.COHORT_SPECIFICITY}),
    ARM_CLEAN_REPLAY: frozenset(
        {plan_builder.COHORT_VOTING, plan_builder.COHORT_SPECIFICITY}
    ),
    ARM_BENIGN: frozenset({plan_builder.COHORT_BENIGN}),
}

#: Paired roots, segments, token identity and backends, taken from the
#: predecessor paired scorer so the two cannot drift.
ROOT_BASELINE = paired.ROOT_BASELINE
ROOT_MODIFIED = paired.ROOT_MODIFIED
PAIRED_ROOTS = paired.PAIRED_ROOTS
SEGMENTS = paired.SEGMENTS
SEGMENT_COORDINATES = paired.SEGMENT_COORDINATES

OBJECT_REF_START = crossing_scorer.OBJECT_REF_START
BOX_START = crossing_scorer.BOX_START
BOX_END = crossing_scorer.BOX_END
COORDINATE_TOKEN_COUNT = crossing_scorer.COORDINATE_TOKEN_COUNT

KV_CACHE_BACKEND = crossing_scorer.KV_CACHE_BACKEND
UNCACHED_BACKEND = crossing_scorer.UNCACHED_BACKEND
CACHE_ADMITTED = crossing_scorer.CACHE_ADMITTED
UNCACHED_FALLBACK = crossing_scorer.UNCACHED_FALLBACK
CACHE_PARITY_MAX_SELECTED_LOGIT_ABS_DIFF = (
    crossing_scorer.CACHE_PARITY_MAX_SELECTED_LOGIT_ABS_DIFF
)
LIKELIHOOD_CHANNEL = crossing_scorer.LIKELIHOOD_CHANNEL
NATIVE_REPETITION_PENALTY_STRATUM = crossing_scorer.NATIVE_REPETITION_PENALTY_STRATUM

CLAIM_BOUNDARY = plan_builder.CLAIM_BOUNDARY

#: Modules whose source determines this pass's semantics.
SOURCE_IDENTITY_MODULES: tuple[str, ...] = (
    "scripts.research.score_sorted_crossing_neutral_row_control",
    "scripts.research.prepare_sorted_crossing_neutral_row_control",
    "scripts.research.score_sorted_crossing_boundary_owner_release_secondary",
    *crossing_scorer.SOURCE_IDENTITY_MODULES,
)

#: This path teacher-forces one literal span per root over two distinct roots,
#: so there is no shared batched root.  ``--batch-size`` is accepted for CLI
#: parity, reported, and never silently pretended to apply.
BATCH_POLICY_NOT_APPLICABLE = paired.BATCH_POLICY_NOT_APPLICABLE
DEFAULT_BATCH_SIZE = crossing_scorer.DEFAULT_BATCH_SIZE


class NeutralRowScoreContractError(crossing_scorer.CrossingBoundaryContractError):
    """A precondition of the neutral-row control capture was not proven."""


def _fail(message: str) -> NoReturn:
    raise NeutralRowScoreContractError(message)


canonical_json_bytes = crossing_scorer.canonical_json_bytes
sha256_json = crossing_scorer.sha256_json
sha256_bytes = crossing_scorer.sha256_bytes
sha256_file = crossing_scorer.sha256_file


def _token_ids(value: Any, *, label: str) -> list[int]:
    return crossing_scorer._token_ids(value, label=label)  # noqa: SLF001


def _read_json(path: Path, label: str) -> dict[str, Any]:
    return crossing_scorer._read_json(Path(path), label)  # noqa: SLF001


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    return crossing_scorer._read_jsonl(Path(path), label)  # noqa: SLF001


def _assert_self_sealed(payload: Mapping[str, Any], *, seal_key: str, label: str) -> str:
    declared = payload.get(seal_key)
    unsealed = {key: value for key, value in payload.items() if key != seal_key}
    reconstructed = sha256_json(unsealed)
    if reconstructed != declared:
        _fail(f"{label} does not self-seal; it has been edited after it was written")
    return str(reconstructed)


# ---------------------------------------------------------------------------
# 1. The sealed neutral-row plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SealedNeutralPlan:
    """This unit's sealed CPU plan, bound to the predecessor census surfaces.

    ``crossing`` is the predecessor sealed crossing plan, loaded unchanged.  It
    is used purely as the proven census/context surface -- literal boundary
    prefixes, executed prefixes and the one-image session guard -- so this pass
    never re-implements or widens the predecessor's frozen request vocabulary.
    """

    plan_dir: Path
    manifest: dict[str, Any]
    selection_rows: list[dict[str, Any]]
    benign_rows: list[dict[str, Any]]
    request_rows: list[dict[str, Any]]
    plan_file_sha256: dict[str, str]
    crossing: crossing_scorer.SealedPlan

    @property
    def manifest_content_sha256(self) -> str:
        return str(self.manifest["manifest_content_sha256"])

    @property
    def registry_by_owner(self) -> dict[str, dict[str, Any]]:
        registry: dict[str, dict[str, Any]] = {}
        for row in (*self.selection_rows, *self.benign_rows):
            registry[str(row["gt_owner_id"])] = row
        return registry

    def context(self, context_id: str) -> Mapping[str, Any]:
        context = self.crossing.contexts_by_id.get(str(context_id))
        if context is None:
            _fail(f"context {context_id!r} is absent from the sealed census context registry")
        return context

    def context_prefix_token_ids(self, context_id: str) -> list[int]:
        return self.crossing.context_prefix_token_ids(str(context_id))

    def executed_prefix_token_ids(self, context_id: str) -> list[int]:
        return self.crossing.executed_prefix_token_ids(str(context_id))


def load_sealed_plan(
    plan_dir: Path,
    *,
    crossing_plan_dir: Path | None = None,
    prevalence_run_root: Path | None = None,
    census_run_root: Path | None = None,
) -> SealedNeutralPlan:
    """Re-prove every digest of this unit's plan and of the predecessor surfaces."""

    plan_dir = Path(plan_dir)
    manifest = _read_json(plan_dir / plan_builder.MANIFEST_NAME, "neutral-row plan manifest")
    if str(manifest.get("schema_version")) != plan_builder.MANIFEST_SCHEMA_VERSION:
        _fail(
            f"neutral-row plan manifest schema {manifest.get('schema_version')!r} is not "
            f"{plan_builder.MANIFEST_SCHEMA_VERSION!r}"
        )
    if str(manifest.get("unit_id")) != UNIT_ID:
        _fail(f"neutral-row plan manifest belongs to unit {manifest.get('unit_id')!r}")
    _assert_self_sealed(
        manifest, seal_key="manifest_content_sha256", label="neutral-row plan manifest"
    )

    digests = manifest.get("output_file_digests")
    if not isinstance(digests, Mapping):
        _fail("neutral-row plan manifest carries no output_file_digests")
    plan_file_sha256: dict[str, str] = {}
    for name in (
        plan_builder.SELECTION_REGISTRY_NAME,
        plan_builder.BENIGN_REGISTRY_NAME,
        plan_builder.REQUEST_PLAN_NAME,
    ):
        entry = digests.get(name)
        if not isinstance(entry, Mapping):
            _fail(f"neutral-row plan manifest declares no digest for {name}")
        observed = sha256_file(plan_dir / name)
        if observed != str(entry.get("sha256")):
            _fail(
                f"sealed plan file {name} hashes to {observed}, not the manifest's "
                f"{entry.get('sha256')}"
            )
        if int(entry.get("byte_size", -1)) != (plan_dir / name).stat().st_size:
            _fail(f"sealed plan file {name} byte size disagrees with the manifest")
        plan_file_sha256[name] = observed

    selection_rows = _read_jsonl(
        plan_dir / plan_builder.SELECTION_REGISTRY_NAME, "selection registry"
    )
    benign_rows = _read_jsonl(plan_dir / plan_builder.BENIGN_REGISTRY_NAME, "benign registry")
    request_rows = _read_jsonl(plan_dir / plan_builder.REQUEST_PLAN_NAME, "request plan")
    for label, rows, expected_schema in (
        ("selection registry", selection_rows, plan_builder.SELECTION_SCHEMA_VERSION),
        ("benign registry", benign_rows, plan_builder.BENIGN_SCHEMA_VERSION),
        ("request plan", request_rows, plan_builder.REQUEST_SCHEMA_VERSION),
    ):
        if not rows:
            _fail(f"{label} is empty")
        for row in rows:
            if str(row.get("schema_version")) != expected_schema:
                _fail(f"{label} carries a row with schema {row.get('schema_version')!r}")
            if str(row.get("unit_id")) != UNIT_ID:
                _fail(f"{label} carries a row from another unit")
        crossing_scorer.assert_no_retokenizable_text(rows, label=label)

    lineage = manifest.get("lineage")
    if not isinstance(lineage, Mapping):
        _fail("neutral-row plan manifest carries no lineage")
    resolved_crossing = Path(
        crossing_plan_dir
        if crossing_plan_dir is not None
        else str(lineage.get("crossing_plan_dir", ""))
    )
    crossing = crossing_scorer.load_sealed_plan(
        resolved_crossing,
        prevalence_run_root=prevalence_run_root,
        census_run_root=census_run_root,
    )
    if str(crossing.manifest.get("manifest_content_sha256")) != str(
        lineage.get("crossing_plan_manifest_content_sha256")
    ):
        _fail(
            "the crossing plan on disk is not the one this neutral-row plan was sealed over; "
            "refusing to score against a drifted predecessor cohort"
        )
    return SealedNeutralPlan(
        plan_dir=plan_dir,
        manifest=manifest,
        selection_rows=selection_rows,
        benign_rows=benign_rows,
        request_rows=request_rows,
        plan_file_sha256=plan_file_sha256,
        crossing=crossing,
    )


# ---------------------------------------------------------------------------
# 2. The sealed request contract
# ---------------------------------------------------------------------------


def assert_request(row: Mapping[str, Any]) -> str:
    """Unit, arm, family, cohort role and appended role, or fail."""

    request_id = str(row.get("request_id"))
    if str(row.get("unit_id")) != UNIT_ID:
        _fail(f"request {request_id!r} belongs to another unit")
    arm = str(row.get("arm"))
    if arm not in ARMS:
        _fail(f"request {request_id!r} declares unknown arm {arm!r}")
    if str(row.get("request_family")) != plan_builder.REQUEST_FAMILY_BY_ARM[arm]:
        _fail(
            f"request {request_id!r} at arm {arm!r} declares family "
            f"{row.get('request_family')!r}, not the sealed "
            f"{plan_builder.REQUEST_FAMILY_BY_ARM[arm]!r}"
        )
    cohort_role = str(row.get("cohort_role"))
    if cohort_role not in ARM_COHORT_ROLES[arm]:
        _fail(
            f"request {request_id!r} at arm {arm!r} declares cohort role {cohort_role!r}, which "
            f"is not one of the sealed {sorted(ARM_COHORT_ROLES[arm])!r}"
        )
    prefix = row.get("prefix")
    if not isinstance(prefix, Mapping):
        _fail(f"request {request_id!r} carries no prefix binding")
    if str(prefix.get("appended_role")) != plan_builder.APPENDED_ROLE_BY_ARM[arm]:
        _fail(
            f"request {request_id!r} appends role {prefix.get('appended_role')!r}, not the "
            f"sealed {plan_builder.APPENDED_ROLE_BY_ARM[arm]!r}"
        )
    if bool(prefix.get("retokenized")):
        _fail(f"request {request_id!r} declares a retokenized prefix")
    if bool(row.get("inspects_new_model_logits")):
        _fail(f"request {request_id!r} declares that the plan inspected a new model logit")
    return arm


def counts_by_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {arm: 0 for arm in ARMS}
    for row in rows:
        counts[str(row["arm"])] += 1
    return counts


def validate_plan_counts(plan: Any) -> dict[str, Any]:
    """Re-derive the frozen 9/12/5 cohort and 21/21/12 request census.

    Takes any object exposing ``request_rows`` and ``manifest`` so the merge
    pass, which binds the plan directory alone and never loads the census
    surfaces, re-derives the same denominators from the same code.
    """

    rows = plan.request_rows
    seen: set[str] = set()
    for row in rows:
        assert_request(row)
        request_id = str(row["request_id"])
        if request_id in seen:
            _fail(f"the sealed plan carries a duplicate request {request_id!r}")
        seen.add(request_id)
    observed = counts_by_arm(rows)
    drifted = sorted(
        arm for arm in ARMS if observed[arm] != EXPECTED_REQUEST_COUNT_BY_ARM[arm]
    )
    if drifted or len(rows) != TOTAL_REQUEST_COUNT:
        _fail(
            f"the sealed plan holds {len(rows)} requests {observed!r}, not the frozen "
            f"{TOTAL_REQUEST_COUNT} {dict(EXPECTED_REQUEST_COUNT_BY_ARM)!r} "
            f"(drifted arm(s): {drifted!r})"
        )

    cohort = plan.manifest.get("cohort")
    if not isinstance(cohort, Mapping):
        _fail("the sealed plan manifest carries no cohort block")
    for key, expected in (
        ("voting_owner_count", plan_builder.VOTING_OWNER_COUNT),
        ("specificity_owner_count", plan_builder.SPECIFICITY_OWNER_COUNT),
        ("infeasible_owner_count", plan_builder.INFEASIBLE_OWNER_COUNT),
        ("benign_control_count", plan_builder.BENIGN_CONTROL_COUNT),
        ("executed_owner_count", plan_builder.EXECUTED_OWNER_COUNT),
    ):
        if int(cohort.get(key, -1)) != expected:
            _fail(
                f"the sealed plan declares {key}={cohort.get(key)!r}, not the frozen {expected}"
            )
    prohibited = sorted(
        set(str(value) for value in cohort.get("image_ids") or ())
        & plan_builder.PROHIBITED_IMAGE_IDS
    )
    if prohibited:
        _fail(f"the sealed plan spans prohibited image(s) {prohibited!r}")
    return {
        "expected_total": TOTAL_REQUEST_COUNT,
        "observed_total": len(rows),
        "expected_by_arm": dict(EXPECTED_REQUEST_COUNT_BY_ARM),
        "observed_by_arm": observed,
        "cohort": {
            key: cohort.get(key)
            for key in (
                "voting_owner_count",
                "specificity_owner_count",
                "infeasible_owner_count",
                "executed_owner_count",
                "benign_control_count",
                "image_count",
            )
        },
    }


def requests_for_image(
    rows: Sequence[Mapping[str, Any]], *, image_id: str
) -> list[dict[str, Any]]:
    """Every sealed request of one image, in stable request-id order.

    Selection is by sealed image id alone: no arm, cohort role, sealed reference
    value or stratum participates, so the executed set cannot be narrowed by
    what the predecessor concluded.
    """

    selected = [dict(row) for row in rows if str(row["image_id"]) == str(image_id)]
    if not selected:
        _fail(
            f"image {image_id!r} has no sealed request; the sealed plan covers image(s) "
            f"{sorted({str(row['image_id']) for row in rows})!r}"
        )
    return sorted(selected, key=lambda row: str(row["request_id"]))


def assert_smoke_strata_coverage(
    plan: SealedNeutralPlan, *, rows: Sequence[Mapping[str, Any]], image_id: str
) -> dict[str, Any]:
    """``unit.md`` "Representative smoke": one image must exercise all four strata.

    Every stratum is read from the sealed selection row of a crossing owner the
    smoke actually executes -- never from a runtime score, a cohort role or a
    sealed reference value -- so an image that cannot prove matched-``E``,
    unmatched-``E``, same-description and different-description coverage is
    refused before a single request is executed, and on the CLI path before a
    model is even opened.  The first owner of each stratum under ascending owner
    id is sealed as that stratum's representative.
    """

    registry = plan.registry_by_owner
    owners: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row["arm"]) not in PAIRED_CROSSING_ARMS:
            continue
        gt_owner_id = str(row["gt_owner_id"])
        selection = registry.get(gt_owner_id)
        if selection is None:
            _fail(
                f"request {str(row['request_id'])!r} names owner {gt_owner_id!r}, which the "
                "sealed selection registry does not carry"
            )
        entry = owners.get(gt_owner_id)
        if entry is None:
            entry = {
                "gt_owner_id": gt_owner_id,
                **plan_builder.selection_row_strata(selection),
                "request_ids": [],
            }
            owners[gt_owner_id] = entry
        entry["request_ids"].append(str(row["request_id"]))

    covered: dict[str, list[dict[str, Any]]] = {name: [] for name in REQUIRED_SMOKE_STRATA}
    for _gt_owner_id, entry in sorted(owners.items()):
        entry["request_ids"] = sorted(entry["request_ids"])
        for stratum in (entry["e_stratum"], entry["description_relation"]):
            covered[stratum].append(entry)
    missing = [name for name in REQUIRED_SMOKE_STRATA if not covered[name]]
    if missing:
        _fail(
            f"smoke image {image_id!r} carries no {missing!r} crossing owner; unit.md requires "
            f"one real smoke to exercise {list(REQUIRED_SMOKE_STRATA)!r}, and the sealed plan "
            f"offers only {sorted({name for name, entries in covered.items() if entries})!r} "
            "here"
        )
    return {
        "required": list(REQUIRED_SMOKE_STRATA),
        "source": "sealed_selection_registry_rows_of_the_executed_crossing_owners",
        "representatives": {
            name: dict(entries[0]) for name, entries in sorted(covered.items())
        },
        "owner_strata": [dict(entry) for _owner, entry in sorted(owners.items())],
    }


def assert_scored_token_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``unit.md`` gate 1: one sealed ``E`` row across the ``N`` and ``C`` arms.

    Applied to whatever subset is in hand -- one shard's rows or a whole
    capture -- so an owner whose two arms disagree is caught before its delta is
    ever read.
    """

    by_owner: dict[str, dict[str, str]] = {}
    for row in rows:
        arm = str(row["arm"])
        if arm not in PAIRED_CROSSING_ARMS:
            continue
        target = row.get("scored_target") or row
        digest = str(target.get("token_ids_sha256") or row.get("scored_token_ids_sha256"))
        by_owner.setdefault(str(row["gt_owner_id"]), {})[arm] = digest
    checked: list[str] = []
    for gt_owner_id, arms in sorted(by_owner.items()):
        if len(arms) < len(PAIRED_CROSSING_ARMS):
            continue
        digests = sorted(set(arms.values()))
        if len(digests) != 1:
            _fail(
                f"owner {gt_owner_id!r} scores different E token digests across the N and C "
                f"arms ({digests!r}); the contrast would be meaningless"
            )
        checked.append(gt_owner_id)
    return {
        "checked_owner_count": len(checked),
        "checked_owner_ids": checked,
        "identical_scored_e_tokens_across_arms": True,
    }


# ---------------------------------------------------------------------------
# 3. Paired-root resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedRootBinding:
    """The two sealed contexts one request scores against."""

    arm: str
    gt_owner_id: str
    image_id: str
    modified_context_id: str
    baseline_context_id: str
    successor_context_id: str
    native_row_index: int
    registry_source: str


def _context_at_boundary(
    plan: SealedNeutralPlan, *, image_id: str, boundary_index: int, label: str
) -> str:
    matches = sorted(
        str(context_id)
        for context_id, context in plan.crossing.contexts_by_id.items()
        if str(context.get("image_id")) == str(image_id)
        and context.get("boundary_index") is not None
        and int(context["boundary_index"]) == int(boundary_index)
    )
    if len(matches) != 1:
        _fail(
            f"{label}: image {image_id!r} has {len(matches)} sealed contexts at boundary "
            f"{boundary_index}, expected exactly one"
        )
    return matches[0]


def _sealed_baseline_context(
    registry_row: Mapping[str, Any], *, arm: str, gt_owner_id: str
) -> tuple[str, str]:
    """The baseline context the *registry* seals for this arm, plus its source."""

    if arm in PAIRED_CROSSING_ARMS:
        scored = registry_row.get("scored_e_row")
        if not isinstance(scored, Mapping):
            _fail(f"owner {gt_owner_id!r} seals no scored E row")
        return str(scored["pre_row_context_id"]), "selection_registry.scored_e_row"
    following = registry_row.get("following_native_action")
    if not isinstance(following, Mapping):
        _fail(
            f"benign control {gt_owner_id!r} seals no following native action; the benign "
            "substitution has no unmodified native successor context to pair against"
        )
    return str(following["context_id"]), "benign_registry.following_native_action"


def resolve_paired_roots(
    plan: SealedNeutralPlan,
    request: Mapping[str, Any],
    registry_row: Mapping[str, Any],
) -> PairedRootBinding:
    """Bind one request to its modified root and its unmodified native baseline.

    The baseline is derived three ways and all three must agree: from the sealed
    registry row, from the request's own ``scored_target.baseline_context_id``,
    and from the boundary convention (the context whose boundary index is the
    scored row's native index).  The exact-row identity is then re-proven as the
    literal token suffix between two adjacent sealed boundaries, so no full-row
    token sequence is ever taken on a registry's word.
    """

    arm = str(request["arm"])
    gt_owner_id = str(request["gt_owner_id"])
    image_id = str(request["image_id"])
    modified_context_id = str(request["context_id"])
    target = request.get("scored_target")
    if not isinstance(target, Mapping):
        _fail(f"request {request.get('request_id')!r} carries no scored target")
    native_row_index = int(target["native_row_index"])

    if str(registry_row.get("gt_owner_id")) != gt_owner_id:
        _fail(f"request {request['request_id']!r} joined the wrong registry owner row")
    if str(registry_row.get("image_id")) != image_id:
        _fail(
            f"owner {gt_owner_id!r} is registered on image {registry_row.get('image_id')!r} but "
            f"the request declares image {image_id!r}"
        )
    if str(registry_row.get("cohort_role")) != str(request.get("cohort_role")):
        _fail(
            f"owner {gt_owner_id!r} is registered in cohort role "
            f"{registry_row.get('cohort_role')!r} but its request declares "
            f"{request.get('cohort_role')!r}"
        )

    sealed_baseline, registry_source = _sealed_baseline_context(
        registry_row, arm=arm, gt_owner_id=gt_owner_id
    )
    if str(target.get("baseline_context_id")) != sealed_baseline:
        _fail(
            f"request {request['request_id']!r} declares baseline context "
            f"{target.get('baseline_context_id')!r} but its registry seals {sealed_baseline!r}"
        )
    derived_baseline = _context_at_boundary(
        plan,
        image_id=image_id,
        boundary_index=native_row_index,
        label=f"baseline root of {request['request_id']!r}",
    )
    if sealed_baseline != derived_baseline:
        _fail(
            f"request {request['request_id']!r} baseline context disagrees between the sealed "
            f"registry ({sealed_baseline!r}) and the boundary convention ({derived_baseline!r})"
        )
    if str(plan.context(sealed_baseline)["image_id"]) != image_id:
        _fail(f"baseline context {sealed_baseline!r} belongs to another image")

    modified_boundary = int(plan.context(modified_context_id)["boundary_index"])
    if arm == ARM_BENIGN:
        # The clean twin replaces the native TP row, so the modified root is that
        # row's predecessor boundary and the baseline is the native successor.
        if modified_boundary + 1 != native_row_index:
            _fail(
                f"benign substitution {request['request_id']!r} appends at boundary "
                f"{modified_boundary} but scores native row {native_row_index}; the clean twin "
                "must replace exactly the row between them"
            )
    elif modified_context_id != sealed_baseline:
        _fail(
            f"request {request['request_id']!r} at arm {arm!r} must append to the same native "
            f"boundary it scores against ({sealed_baseline!r}), not {modified_context_id!r}"
        )

    successor_context_id = _context_at_boundary(
        plan,
        image_id=image_id,
        boundary_index=native_row_index + 1,
        label=f"successor of the baseline root of {request['request_id']!r}",
    )
    if str(target.get("successor_context_id")) != successor_context_id:
        _fail(
            f"request {request['request_id']!r} declares successor context "
            f"{target.get('successor_context_id')!r}, not the sealed {successor_context_id!r}"
        )
    baseline_tokens = plan.context_prefix_token_ids(sealed_baseline)
    successor_tokens = plan.context_prefix_token_ids(successor_context_id)
    if successor_tokens[: len(baseline_tokens)] != baseline_tokens:
        _fail(
            f"sealed boundary {successor_context_id!r} does not extend {sealed_baseline!r}; the "
            "native row identity cannot be read off their suffix"
        )
    suffix = successor_tokens[len(baseline_tokens) :]
    if suffix != _token_ids(target["token_ids"], label="scored target tokens"):
        _fail(
            f"request {request['request_id']!r} scores a row that is not the literal token "
            f"suffix between sealed boundaries {sealed_baseline!r} and "
            f"{successor_context_id!r}"
        )
    return PairedRootBinding(
        arm=arm,
        gt_owner_id=gt_owner_id,
        image_id=image_id,
        modified_context_id=modified_context_id,
        baseline_context_id=sealed_baseline,
        successor_context_id=successor_context_id,
        native_row_index=native_row_index,
        registry_source=registry_source,
    )


def assert_appended_row_matches_registry(
    request: Mapping[str, Any], registry_row: Mapping[str, Any]
) -> str:
    """The appended row is the owner's own sealed ``N``, ``C`` or clean twin."""

    arm = str(request["arm"])
    if arm == ARM_NEUTRAL:
        sealed = registry_row.get("neutral_row_n")
        label = "deterministic neutral control row N"
    else:
        sealed = registry_row.get("inserted_clean_row_c")
        label = "clean GT row C"
    if not isinstance(sealed, Mapping):
        _fail(
            f"owner {request['gt_owner_id']!r} seals no {label}; the appended prefix cannot be "
            "attributed"
        )
    declared = str(request["prefix"]["appended_token_ids_sha256"])
    if declared != str(sealed.get("token_ids_sha256")):
        _fail(
            f"request {request['request_id']!r} appends a row hashing to {declared}, not the "
            f"owner's sealed {label} {sealed.get('token_ids_sha256')}"
        )
    appended = _token_ids(
        request["prefix"]["appended_token_ids"], label=f"request {request['request_id']!r}"
    )
    if sha256_json(appended) != declared:
        _fail(f"request {request['request_id']!r} appended tokens do not reconstruct their digest")
    if appended != _token_ids(sealed["token_ids"], label=f"sealed {label}"):
        _fail(f"request {request['request_id']!r} appends tokens the registry does not seal")
    if appended[0] != OBJECT_REF_START or appended[-1] != BOX_END:
        _fail(
            f"request {request['request_id']!r} appends a row that is not a complete "
            "<|object_ref_start|> ... <|box_end|> row"
        )
    if len(appended) < COORDINATE_TOKEN_COUNT + 3 or BOX_START not in appended:
        _fail(f"request {request['request_id']!r} appends a row with no coordinate block")
    return declared


# ---------------------------------------------------------------------------
# 4. Capture
# ---------------------------------------------------------------------------


@dataclass
class CaptureState:
    """Mutable bookkeeping shared by every request of one shard."""

    context_group_ids: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    quarantine: paired.SecondaryQuarantineLedger = field(
        default_factory=paired.SecondaryQuarantineLedger
    )
    root_backend: dict[str, str] = field(
        default_factory=lambda: {root: KV_CACHE_BACKEND for root in PAIRED_ROOTS}
    )


def _capture_request(
    plan: SealedNeutralPlan,
    backend: Any,
    state: CaptureState,
    *,
    shard_id: str,
    session_image_id: str,
    request: Mapping[str, Any],
    registry_row: Mapping[str, Any],
    enforce_replay: bool,
) -> dict[str, Any]:
    """Score one sealed request on both of its paired roots."""

    arm = assert_request(request)
    request_id = str(request["request_id"])
    gt_owner_id = str(request["gt_owner_id"])
    image_id = str(request["image_id"])
    if image_id != str(session_image_id):
        _fail(
            f"request {request_id!r} belongs to image {image_id!r} but the open session holds "
            f"image {session_image_id!r}"
        )
    roots = resolve_paired_roots(plan, request, registry_row)
    appended_digest = assert_appended_row_matches_registry(request, registry_row)
    for context_id in (roots.modified_context_id, roots.baseline_context_id):
        plan.crossing.assert_context_belongs_to_session_image(
            context_id,
            session_image_id=session_image_id,
            label=f"neutral-row request {request_id!r}",
        )

    target_tokens = _token_ids(
        request["scored_target"]["token_ids"], label=f"request {request_id!r} target"
    )
    segments = paired.row_segments(target_tokens)
    appended = _token_ids(
        request["prefix"]["appended_token_ids"], label=f"request {request_id!r} appended"
    )
    root_prefix = {
        ROOT_BASELINE: plan.executed_prefix_token_ids(roots.baseline_context_id),
        ROOT_MODIFIED: plan.executed_prefix_token_ids(roots.modified_context_id),
    }
    root_appended: dict[str, list[int]] = {ROOT_BASELINE: [], ROOT_MODIFIED: appended}
    root_context = {
        ROOT_BASELINE: roots.baseline_context_id,
        ROOT_MODIFIED: roots.modified_context_id,
    }

    payloads: dict[str, dict[str, Any]] = {}
    forced: dict[str, list[int]] = {}
    for root in PAIRED_ROOTS:
        scoring_backend = state.root_backend[root]
        group_id = paired._root_group_id(  # noqa: SLF001
            gt_owner_id=gt_owner_id,
            context_id=root_context[root],
            variant=arm,
            root=root,
            appended_digest=sha256_json(root_appended[root]),
            scoring_backend=scoring_backend,
        )
        state.context_group_ids.append(group_id)
        scored = paired._score_one_root(  # noqa: SLF001
            backend,
            root_token_ids=root_prefix[root] + root_appended[root],
            target_token_ids=target_tokens,
            uncached=scoring_backend == UNCACHED_BACKEND,
            label=f"{root} of {request_id}",
        )
        forced[root] = [token.token_id for token in scored]
        payloads[root] = paired._root_payload(  # noqa: SLF001
            root=root,
            context_id=root_context[root],
            executed_prefix_token_ids=root_prefix[root],
            appended_token_ids=root_appended[root],
            context_group_id=group_id,
            scoring_backend=scoring_backend,
            scored=scored,
            segments=segments,
            target_token_ids=target_tokens,
        )

    target_digest = paired.assert_paired_token_identity(
        forced[ROOT_BASELINE],
        forced[ROOT_MODIFIED],
        declared_sha256=str(request["scored_target"]["token_ids_sha256"]),
        label=f"request {request_id!r}",
    )
    deltas = paired.paired_deltas(
        payloads[ROOT_BASELINE]["segment_sums"], payloads[ROOT_MODIFIED]["segment_sums"]
    )

    replay_admitted = bool(payloads[ROOT_BASELINE]["argmax_reproduces_description_path"])
    if enforce_replay:
        state.quarantine = paired.apply_secondary_quarantine(
            admitted=replay_admitted,
            request_id=request_id,
            gt_owner_id=gt_owner_id,
            reason="baseline_native_argmax_replay_mismatch",
            detail=(
                "deterministic argmax at the unmodified native root did not reproduce the "
                "scored row's description path"
            ),
            ledger=state.quarantine,
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "row_kind": "neutral_row_control_row",
        "unit_id": UNIT_ID,
        "shard_id": str(shard_id),
        "request_id": request_id,
        "request_key": str(request["request_key"]),
        "request_family": str(request["request_family"]),
        "arm": arm,
        "cohort_role": str(request["cohort_role"]),
        "gt_owner_id": gt_owner_id,
        "image_id": image_id,
        "session_image_id": str(session_image_id),
        "plan_identity_digest": str(request["identity_digest"]),
        "plan_manifest_content_sha256": plan.manifest_content_sha256,
        "native_row_index": roots.native_row_index,
        "scored_target_kind": str(request["scored_target"]["kind"]),
        "baseline_context_id": roots.baseline_context_id,
        "modified_context_id": roots.modified_context_id,
        "successor_context_id": roots.successor_context_id,
        "baseline_context_source": roots.registry_source,
        "appended_row_token_ids_sha256": appended_digest,
        "appended_row_token_count": len(appended),
        "appended_role": str(request["prefix"]["appended_role"]),
        "scored_token_ids": target_tokens,
        "scored_token_ids_sha256": target_digest,
        "scored_token_count": len(target_tokens),
        "segments": {
            segment: {
                "token_index_start": segments.span(segment)[0],
                "token_index_stop": segments.span(segment)[1],
                "token_ids": target_tokens[
                    segments.span(segment)[0] : segments.span(segment)[1]
                ],
            }
            for segment in SEGMENTS
        },
        "roots": payloads,
        "deltas": deltas,
        "sealed_reference": dict(request["sealed_reference"]),
        "baseline_replay_admitted": replay_admitted,
        "replay_admission_enforced": bool(enforce_replay),
        "likelihood_channel": LIKELIHOOD_CHANNEL,
        "repetition_penalty_stratum": float(NATIVE_REPETITION_PENALTY_STRATUM),
        "uses_model_generate": False,
        "sampling": "disabled_deterministic_teacher_forcing_only",
        "retokenized": False,
        "delta_orientation": "modified_minus_baseline",
        "primary_segment": SEGMENT_COORDINATES,
        "relative_estimand_owner": (
            "the analysis pass, from the same-run benign replay of this image only"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


# ---------------------------------------------------------------------------
# 5. Shard assembly and publication
# ---------------------------------------------------------------------------


@dataclass
class ShardResult:
    receipt: dict[str, Any]
    rows: list[dict[str, Any]] = field(default_factory=list)
    parity: dict[str, Any] = field(default_factory=dict)
    quarantine: dict[str, Any] | None = None
    admission: dict[str, Any] | None = None


def shard_output_files(result: ShardResult) -> dict[str, bytes]:
    """The indivisible published byte content of one shard."""

    if result.quarantine is not None:
        return {
            QUARANTINE_NAME: canonical_json_bytes(result.quarantine) + b"\n",
            RECEIPT_NAME: canonical_json_bytes(result.receipt) + b"\n",
        }
    if result.admission is not None:
        return {
            ADMISSION_NAME: canonical_json_bytes(result.admission) + b"\n",
            PARITY_NAME: canonical_json_bytes(result.parity) + b"\n",
            RECEIPT_NAME: canonical_json_bytes(result.receipt) + b"\n",
        }
    return {
        ROWS_NAME: b"".join(canonical_json_bytes(row) + b"\n" for row in result.rows),
        PARITY_NAME: canonical_json_bytes(result.parity) + b"\n",
        RECEIPT_NAME: canonical_json_bytes(result.receipt) + b"\n",
    }


def source_identity() -> dict[str, str]:
    """Content digests of every module this capture's semantics depend on."""

    import importlib

    digests: dict[str, str] = {}
    for name in SOURCE_IDENTITY_MODULES:
        module = importlib.import_module(name)
        source = getattr(module, "__file__", None)
        if source is None:  # pragma: no cover - every module here has a file
            _fail(f"module {name!r} exposes no source file to hash")
        digests[name] = sha256_file(Path(source))
    return dict(sorted(digests.items()))


def _base_receipt(
    plan: SealedNeutralPlan,
    *,
    shard_id: str,
    mode: str,
    runtime_identity: Mapping[str, Any],
    plan_counts: Mapping[str, Any],
    image_counts: Mapping[str, Any],
    executed: Mapping[str, Any],
    quarantine: paired.SecondaryQuarantineLedger,
    enforce_replay: bool,
    backend_kind: str,
    batch_size: int,
) -> dict[str, Any]:
    """The identity/denominator/policy spine both run modes seal."""

    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "mode": str(mode),
        "shard_id": str(shard_id),
        "scorer_source_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "source_identity": source_identity(),
        "plan": {
            "plan_dir": str(plan.plan_dir),
            "manifest_schema_version": plan.manifest.get("schema_version"),
            "manifest_content_sha256": plan.manifest_content_sha256,
            "builder_source_sha256": str(
                (plan.manifest.get("builder_source") or {}).get("sha256")
            ),
            "plan_file_sha256": dict(sorted(plan.plan_file_sha256.items())),
            "lineage": plan.manifest.get("lineage"),
            "cohort": plan.manifest.get("cohort"),
            "gates": plan.manifest.get("gates"),
            "routes": plan.manifest.get("routes"),
        },
        "crossing_plan": {
            "plan_dir": str(plan.crossing.plan_dir),
            "manifest_content_sha256": plan.crossing.manifest.get("manifest_content_sha256"),
        },
        "runtime_identity": dict(runtime_identity),
        "runtime_identity_sha256": crossing_scorer.runtime_identity_digest(runtime_identity),
        "request_counts": {"plan": dict(plan_counts), "image": dict(image_counts)},
        "executed": dict(executed),
        "quarantine": {
            "schema_version": QUARANTINE_SCHEMA_VERSION,
            "count": quarantine.count,
            "entries": [
                {
                    "request_id": entry.request_id,
                    "gt_owner_id": entry.gt_owner_id,
                    "reason": entry.reason,
                    "detail": entry.detail,
                }
                for entry in quarantine.entries
            ],
            "policy": "any quarantined request withholds the whole shard's evidence",
        },
        "policy": {
            "uses_model_generate": False,
            "free_decode": False,
            "greedy_coordinate_decode": False,
            "sampling": "not_implemented_deterministic_teacher_forcing_only",
            "retokenizes": False,
            "likelihood_channel": LIKELIHOOD_CHANNEL,
            "repetition_penalty_stratum": float(NATIVE_REPETITION_PENALTY_STRATUM),
            "executes_every_sealed_request_of_the_image": True,
            "request_selection_uses_sealed_reference_values": False,
            "one_image_session_per_shard": True,
            "fresh_cache_per_logical_root": True,
            "paired_roots_force_identical_target_tokens": True,
            "scored_e_tokens_identical_across_n_and_c_arms": True,
            "delta_orientation": "modified_minus_baseline",
            "relative_estimand_computed_here": False,
            "materiality_decided_here": False,
            "route_decided_here": False,
            "sealed_reference_role": "gate_reference_only_never_the_estimand",
            "replay_admission_enforced": bool(enforce_replay),
            "backend_kind": str(backend_kind),
            "requested_batch_size": int(batch_size),
            "effective_batch_size": 1,
            "batching_applicable": False,
            "batching_note": BATCH_POLICY_NOT_APPLICABLE,
            "claim_boundary": CLAIM_BOUNDARY,
        },
        # No wall-clock field is sealed: the published artifact set must be
        # byte-identical across re-runs so an idempotent re-capture publishes as
        # a no-op instead of colliding with itself.
        "artifact_determinism": "no_wall_clock_or_host_specific_field_is_sealed",
    }


def _seal_output_digests(receipt: dict[str, Any], files: Mapping[str, bytes]) -> None:
    receipt["output_file_digests"] = {
        name: {"path": name, "byte_size": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(files.items())
        if name != RECEIPT_NAME
    }
    receipt["receipt_content_sha256"] = sha256_json(receipt)


def _quarantine_payload(
    *, shard_id: str, session_image_id: str, ledger: paired.SecondaryQuarantineLedger
) -> dict[str, Any]:
    return {
        "schema_version": QUARANTINE_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "shard_id": str(shard_id),
        "session_image_id": str(session_image_id),
        "reason": "one_or_more_requests_failed_their_admission_control",
        "entries": [
            {
                "request_id": entry.request_id,
                "gt_owner_id": entry.gt_owner_id,
                "reason": entry.reason,
                "detail": entry.detail,
            }
            for entry in ledger.entries
        ],
        "evidence_withheld": list(EVIDENCE_OUTPUT_NAMES),
        "next_step": (
            "repair the runtime or plan alignment rather than publishing partial neutral-row "
            "evidence"
        ),
    }


def _prepare_shard(
    plan: SealedNeutralPlan, *, image_id: str
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Validate the whole sealed census, then narrow to one image."""

    plan_counts = validate_plan_counts(plan)
    image_rows = requests_for_image(plan.request_rows, image_id=image_id)
    expected = {
        "image_id": str(image_id),
        "expected_by_arm": counts_by_arm(image_rows),
        "expected_total": len(image_rows),
    }
    return image_rows, plan_counts, expected


def reconcile_image_counts(
    expected: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Fold what a shard produced back onto what the sealed plan required."""

    observed = counts_by_arm(rows)
    if observed != expected["expected_by_arm"] or len(rows) != int(expected["expected_total"]):
        _fail(
            f"image {expected['image_id']!r} produced {len(rows)} rows {observed!r}, not the "
            f"sealed {expected['expected_total']} {expected['expected_by_arm']!r}"
        )
    return {**dict(expected), "observed_by_arm": observed, "observed_total": len(rows)}


def _capture_rows(
    plan: SealedNeutralPlan,
    backend: Any,
    state: CaptureState,
    *,
    shard_id: str,
    session_image_id: str,
    rows: Sequence[Mapping[str, Any]],
    enforce_replay: bool,
) -> list[dict[str, Any]]:
    registry = plan.registry_by_owner
    captured: list[dict[str, Any]] = []
    for request in rows:
        owner_id = str(request["gt_owner_id"])
        registry_row = registry.get(owner_id)
        if registry_row is None:
            _fail(
                f"owner {owner_id!r} is in neither the selection nor the benign registry; a "
                "request cannot be attributed"
            )
        captured.append(
            _capture_request(
                plan,
                backend,
                state,
                shard_id=shard_id,
                session_image_id=session_image_id,
                request=request,
                registry_row=registry_row,
                enforce_replay=enforce_replay,
            )
        )
    assert_scored_token_identity(captured)
    return captured


def _parity_payload(
    *,
    shard_id: str,
    session_image_id: str,
    backend_kind: str,
    root_backend: Mapping[str, str],
    parity: paired.SecondaryParityResult | None,
    inherited_admission: Mapping[str, Any] | None,
    compared_request_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": PARITY_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "shard_id": str(shard_id),
        "session_image_id": str(session_image_id),
        "backend_kind": str(backend_kind),
        "backend_is_evidence_bearing": backend_kind == "hf",
        "root_backend": dict(sorted(root_backend.items())),
        "max_selected_logit_abs_diff_threshold": CACHE_PARITY_MAX_SELECTED_LOGIT_ABS_DIFF,
        "compared_request_ids": [str(value) for value in compared_request_ids],
        "cached_versus_uncached": (
            None
            if parity is None
            else {
                "status": parity.status,
                "max_selected_logit_abs_diff": parity.max_selected_logit_abs_diff,
                "mismatched_fields": list(parity.mismatched_fields),
                "delta_sign_parity": parity.delta_sign_parity,
                "per_root": [
                    {
                        "root": result.root,
                        "aligned": result.aligned,
                        "compared_token_count": result.compared_token_count,
                        "max_selected_logit_abs_diff": result.max_selected_logit_abs_diff,
                        "argmax_parity": result.argmax_parity,
                        "selected_is_argmax_parity": result.selected_is_argmax_parity,
                    }
                    for result in parity.per_root
                ],
            }
        ),
        "inherited_admission": (
            None if inherited_admission is None else dict(inherited_admission)
        ),
        "batching": {
            "applicable": False,
            "effective_batch_size": 1,
            "reason": BATCH_POLICY_NOT_APPLICABLE,
        },
        "compared_fields": [
            "selected_logprob",
            "argmax_token_id",
            "selected_is_argmax",
            "segment_delta_sign",
        ],
    }


def run_smoke_shard(
    *,
    plan: SealedNeutralPlan,
    backend: Any,
    shard_id: str,
    image_id: str,
    runtime_identity: Mapping[str, Any],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ShardResult:
    """Prove cached execution on one image and seal an admission receipt.

    ``unit.md`` "Representative smoke": the smoke image must carry every sealed
    arm *and* all four sealed strata, so one session really exercises a
    matched-``E`` and an unmatched-``E`` owner, a same-description and a
    different-description row, and the ``C``/``N`` scored-token identity check.
    Both are proven from the sealed plan before a request is executed.  Every
    request of the image is then scored twice -- once entirely uncached, once
    entirely on the KV cache -- and the two executions must agree on every
    selected logit within ``1e-3`` and on every argmax, rank flag and segment
    delta sign exactly.
    """

    rows, plan_counts, expected_counts = _prepare_shard(plan, image_id=image_id)
    missing = sorted(arm for arm, count in counts_by_arm(rows).items() if count == 0)
    if missing:
        _fail(
            f"image {image_id!r} carries no sealed request for arm(s) {missing!r}; the smoke "
            "matrix must run inside one image session"
        )
    smoke_strata = assert_smoke_strata_coverage(plan, rows=rows, image_id=image_id)
    enforce_replay = paired.replay_enforcement_for(backend.identity)
    backend_kind = str(backend.identity.get("backend"))

    sides: dict[str, list[dict[str, Any]]] = {}
    state = CaptureState()
    for label, root_backend in (
        ("uncached", {root: UNCACHED_BACKEND for root in PAIRED_ROOTS}),
        ("cached", {root: KV_CACHE_BACKEND for root in PAIRED_ROOTS}),
    ):
        side_state = CaptureState(root_backend=dict(root_backend))
        sides[label] = _capture_rows(
            plan,
            backend,
            side_state,
            shard_id=shard_id,
            session_image_id=image_id,
            rows=rows,
            enforce_replay=enforce_replay,
        )
        state.context_group_ids.extend(side_state.context_group_ids)
        state.quarantine = paired.SecondaryQuarantineLedger(
            entries=(*state.quarantine.entries, *side_state.quarantine.entries)
        )
    crossing_scorer.assert_fresh_context_per_owner(state.context_group_ids)

    image_counts = reconcile_image_counts(expected_counts, sides["cached"])
    reconcile_image_counts(expected_counts, sides["uncached"])
    parity = paired.evaluate_secondary_parity(
        cached_rows=sides["cached"], uncached_rows=sides["uncached"]
    )
    cache_admitted = parity.status == CACHE_ADMITTED
    root_backend = {
        root: (KV_CACHE_BACKEND if cache_admitted else UNCACHED_BACKEND)
        for root in PAIRED_ROOTS
    }
    parity_payload = _parity_payload(
        shard_id=shard_id,
        session_image_id=image_id,
        backend_kind=backend_kind,
        root_backend=root_backend,
        parity=parity,
        inherited_admission=None,
        compared_request_ids=[str(row["request_id"]) for row in sides["cached"]],
    )

    admission = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "smoke_shard_id": str(shard_id),
        "smoke_image_id": str(image_id),
        "plan_manifest_content_sha256": plan.manifest_content_sha256,
        "runtime_identity_sha256": crossing_scorer.runtime_identity_digest(runtime_identity),
        "admission_identity_fields": list(crossing_scorer.ADMISSION_IDENTITY_FIELDS),
        **crossing_scorer.admission_identity_payload(runtime_identity),
        "smoke_arms": dict(counts_by_arm(rows)),
        "smoke_strata": smoke_strata,
        "cache_admitted": cache_admitted,
        "root_backend": dict(sorted(root_backend.items())),
        "max_selected_logit_abs_diff": parity.max_selected_logit_abs_diff,
        "mismatched_fields": list(parity.mismatched_fields),
        "replay_admission_enforced": bool(enforce_replay),
        "scope": (
            "one image session carried every sealed arm; no context of another image was "
            "forwarded through it"
        ),
    }
    admission["admission_content_sha256"] = sha256_json(admission)

    receipt = _base_receipt(
        plan,
        shard_id=shard_id,
        mode=MODE_SMOKE,
        runtime_identity=runtime_identity,
        plan_counts=plan_counts,
        image_counts=image_counts,
        executed={
            "session_image_id": str(image_id),
            "row_count": 0,
            "executed_unpublished_row_count": len(sides["cached"]) + len(sides["uncached"]),
            "logical_context_group_count": len(state.context_group_ids),
            "logical_context_groups_sha256": sha256_json(sorted(state.context_group_ids)),
            "request_ids_sha256": sha256_json(sorted(str(row["request_id"]) for row in rows)),
        },
        quarantine=state.quarantine,
        enforce_replay=enforce_replay,
        backend_kind=backend_kind,
        batch_size=batch_size,
    )
    receipt["admission"] = admission

    if state.quarantine.count:
        result = ShardResult(
            receipt=receipt,
            quarantine=_quarantine_payload(
                shard_id=shard_id, session_image_id=image_id, ledger=state.quarantine
            ),
        )
        _seal_output_digests(receipt, shard_output_files(result))
        return result
    result = ShardResult(receipt=receipt, parity=parity_payload, admission=admission)
    _seal_output_digests(receipt, shard_output_files(result))
    return result


def _admitted_strata(admission: Mapping[str, Any]) -> list[str]:
    """The strata one admission receipt actually proved, in frozen order."""

    representatives = (admission.get("smoke_strata") or {}).get("representatives") or {}
    return [name for name in REQUIRED_SMOKE_STRATA if name in representatives]


def validate_admission_receipt(
    admission: Mapping[str, Any],
    *,
    plan: SealedNeutralPlan,
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """An admission is inherited only under the exact plan and runtime it proved."""

    if str(admission.get("schema_version")) != ADMISSION_SCHEMA_VERSION:
        _fail(
            f"admission schema {admission.get('schema_version')!r} is not "
            f"{ADMISSION_SCHEMA_VERSION!r}"
        )
    if str(admission.get("unit_id")) != UNIT_ID:
        _fail("admission receipt belongs to another unit")
    _assert_self_sealed(
        admission, seal_key="admission_content_sha256", label="admission receipt"
    )
    if str(admission.get("plan_manifest_content_sha256")) != plan.manifest_content_sha256:
        _fail(
            "the admission was sealed against a different plan manifest; a smoke cannot admit "
            "a capture of another plan"
        )
    declared_digest = crossing_scorer.runtime_identity_digest(admission)
    if declared_digest != str(admission.get("runtime_identity_sha256")):
        _fail(
            "the admission's runtime_identity_sha256 does not reconstruct from its own declared "
            "identity fields; the receipt is internally inconsistent"
        )
    if declared_digest != crossing_scorer.runtime_identity_digest(runtime_identity):
        differing = sorted(
            name
            for name in crossing_scorer.ADMISSION_IDENTITY_FIELDS
            if sha256_json(admission.get(name)) != sha256_json(runtime_identity.get(name))
        )
        _fail(
            "the admission was sealed under a different runtime identity (differing field(s): "
            f"{differing or ['<unreported>']!r}); cached execution is never inherited across a "
            "changed model, tokenizer, numerical runtime or scorer"
        )
    proven = _admitted_strata(admission)
    if proven != list(REQUIRED_SMOKE_STRATA):
        _fail(
            f"the admission proves strata {proven!r}, not the frozen "
            f"{list(REQUIRED_SMOKE_STRATA)!r}; unit.md admits a capture only behind one real "
            "smoke that exercised every stratum"
        )
    root_backend = admission.get("root_backend") or {}
    for root in PAIRED_ROOTS:
        if root not in root_backend:
            _fail(f"admission declares no backend for root {root!r}")
    if not bool(admission.get("cache_admitted")) and any(
        str(value) != UNCACHED_BACKEND for value in root_backend.values()
    ):
        _fail("admission is internally inconsistent about its root backends")
    return dict(admission)


def run_capture_shard(
    *,
    plan: SealedNeutralPlan,
    backend: Any,
    shard_id: str,
    session_image_id: str,
    admission: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ShardResult:
    """Execute every sealed request of one image under an admission."""

    rows, plan_counts, expected_counts = _prepare_shard(plan, image_id=session_image_id)
    enforce_replay = paired.replay_enforcement_for(backend.identity)
    backend_kind = str(backend.identity.get("backend"))
    state = CaptureState(
        root_backend={root: str(admission["root_backend"][root]) for root in PAIRED_ROOTS}
    )
    state.rows = _capture_rows(
        plan,
        backend,
        state,
        shard_id=shard_id,
        session_image_id=session_image_id,
        rows=rows,
        enforce_replay=enforce_replay,
    )
    crossing_scorer.assert_fresh_context_per_owner(state.context_group_ids)
    image_counts = reconcile_image_counts(expected_counts, state.rows)

    parity_payload = _parity_payload(
        shard_id=shard_id,
        session_image_id=session_image_id,
        backend_kind=backend_kind,
        root_backend=state.root_backend,
        parity=None,
        inherited_admission={
            "smoke_shard_id": admission.get("smoke_shard_id"),
            "smoke_image_id": admission.get("smoke_image_id"),
            "admission_content_sha256": admission.get("admission_content_sha256"),
            "cache_admitted": admission.get("cache_admitted"),
            "smoke_strata_proven": _admitted_strata(admission),
        },
        compared_request_ids=[],
    )

    receipt = _base_receipt(
        plan,
        shard_id=shard_id,
        mode=MODE_CAPTURE,
        runtime_identity=runtime_identity,
        plan_counts=plan_counts,
        image_counts=image_counts,
        executed={
            "session_image_id": str(session_image_id),
            "row_count": len(state.rows),
            "gt_owner_ids": sorted({str(row["gt_owner_id"]) for row in state.rows}),
            "arm_counts": dict(sorted(counts_by_arm(state.rows).items())),
            "logical_context_group_count": len(state.context_group_ids),
            "logical_context_groups_sha256": sha256_json(sorted(state.context_group_ids)),
            "request_ids_sha256": sha256_json(
                sorted(str(row["request_id"]) for row in state.rows)
            ),
        },
        quarantine=state.quarantine,
        enforce_replay=enforce_replay,
        backend_kind=backend_kind,
        batch_size=batch_size,
    )
    receipt["admission"] = {
        "smoke_shard_id": admission.get("smoke_shard_id"),
        "smoke_image_id": admission.get("smoke_image_id"),
        "admission_content_sha256": admission.get("admission_content_sha256"),
        "smoke_strata_proven": _admitted_strata(admission),
        "root_backend": dict(sorted(state.root_backend.items())),
    }

    if state.quarantine.count:
        result = ShardResult(
            receipt=receipt,
            quarantine=_quarantine_payload(
                shard_id=shard_id,
                session_image_id=session_image_id,
                ledger=state.quarantine,
            ),
        )
        _seal_output_digests(receipt, shard_output_files(result))
        return result
    result = ShardResult(receipt=receipt, rows=state.rows, parity=parity_payload)
    _seal_output_digests(receipt, shard_output_files(result))
    return result


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir", required=True, type=Path, help="sealed neutral-row CPU plan directory"
    )
    parser.add_argument(
        "--crossing-plan-dir",
        type=Path,
        default=None,
        help="sealed crossing plan directory; defaults to the neutral-row plan's own lineage",
    )
    parser.add_argument("--prevalence-run-root", type=Path, default=None)
    parser.add_argument("--census-run-root", type=Path, default=None)
    parser.add_argument("--infer-config", type=Path, default=None)
    parser.add_argument(
        "--runtime-identity",
        type=Path,
        default=None,
        help="frozen model/tokenizer identity JSON; required for --backend hf",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="explicit output shard")
    parser.add_argument("--shard-id", required=True, help="explicit shard identity")
    parser.add_argument(
        "--mode",
        choices=(MODE_SMOKE, MODE_CAPTURE),
        default=MODE_CAPTURE,
        help=(
            "smoke: prove cached-versus-uncached parity on one image whose sealed rows cover "
            "all four strata and seal an admission receipt. capture: score one image's sealed "
            "requests under it."
        ),
    )
    parser.add_argument(
        "--image-id",
        required=True,
        help="the one image this run opens a session for; nothing else may be scored on it",
    )
    parser.add_argument(
        "--admission-receipt",
        type=Path,
        default=None,
        help=f"the smoke run's {ADMISSION_NAME}; required for --mode capture",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "accepted for parity with the predecessor scorers and reported in the receipt; this "
            "path teacher-forces one lane per root and never batches"
        ),
    )
    parser.add_argument("--backend", choices=("hf", "fake"), default="hf")
    parser.add_argument(
        "--validate-plan-only",
        action="store_true",
        help="re-prove every plan digest and denominator, then exit without opening a session",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Load, validate, execute and publish exactly one shard."""

    plan = load_sealed_plan(
        Path(args.plan_dir),
        crossing_plan_dir=args.crossing_plan_dir,
        prevalence_run_root=args.prevalence_run_root,
        census_run_root=args.census_run_root,
    )
    plan_counts = validate_plan_counts(plan)
    if args.validate_plan_only:
        return {"validated": True, "request_counts": plan_counts, "published": False}

    if args.backend == "hf" and args.runtime_identity is None:
        _fail("--runtime-identity is required for --backend hf")
    if int(args.batch_size) < 1:
        _fail("--batch-size must be at least one")

    image_id = str(args.image_id)
    admission: Mapping[str, Any] | None = None
    if args.mode == MODE_CAPTURE:
        if args.admission_receipt is None:
            _fail(
                f"--mode {MODE_CAPTURE} requires --admission-receipt pointing at the smoke run's "
                f"{ADMISSION_NAME}; cached execution is never assumed"
            )
        admission = _read_json(Path(args.admission_receipt), "admission receipt")
    # Both fail closed, with the covered images or the missing strata named,
    # before a model is opened.
    image_requests = requests_for_image(plan.request_rows, image_id=image_id)
    if args.mode == MODE_SMOKE:
        assert_smoke_strata_coverage(plan, rows=image_requests, image_id=image_id)

    if args.backend == "hf":
        crossing_scorer._basin().pin_fp32_parity_flags()  # noqa: SLF001

    with crossing_scorer._open_backend(  # noqa: SLF001
        args, plan.crossing, image_id
    ) as backend:
        numerics = crossing_scorer.build_runtime_numerics(
            backend, infer_config=args.infer_config
        )
        runtime_identity = {
            **crossing_scorer.validate_runtime_identity(
                backend.identity, args.runtime_identity, numerics=numerics
            ),
            "numerics": numerics,
            "source_identity": source_identity(),
            "session_image_id": image_id,
        }
        if args.mode == MODE_SMOKE:
            result = run_smoke_shard(
                plan=plan,
                backend=backend,
                shard_id=str(args.shard_id),
                image_id=str(args.image_id),
                runtime_identity=runtime_identity,
                batch_size=int(args.batch_size),
            )
        else:
            assert admission is not None  # for type-checkers; guarded above
            validated = validate_admission_receipt(
                admission, plan=plan, runtime_identity=runtime_identity
            )
            result = run_capture_shard(
                plan=plan,
                backend=backend,
                shard_id=str(args.shard_id),
                session_image_id=str(args.image_id),
                admission=validated,
                runtime_identity=runtime_identity,
                batch_size=int(args.batch_size),
            )

    files = shard_output_files(result)
    published = crossing_scorer._publish(Path(args.output_dir), files)  # noqa: SLF001
    return {
        "validated": True,
        "request_counts": plan_counts,
        "mode": str(args.mode),
        "quarantined": result.quarantine is not None,
        "row_count": len(result.rows),
        **published,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except crossing_scorer.CrossingBoundaryContractError as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        return 1
    print(
        "neutral-row shard: "
        f"mode={summary.get('mode')} rows={summary.get('row_count')} "
        f"quarantined={summary.get('quarantined')} "
        f"published={summary.get('published')} dir={summary.get('output_dir')}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
