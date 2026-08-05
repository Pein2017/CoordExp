#!/usr/bin/env python3
"""Secondary downstream-compatibility capture for the sorted crossing-boundary
owner release/realization unit
(``2026-08-03-sorted-crossing-boundary-owner-release-realization``).

Frozen unit::

    research/investigations/qwen3-vl-dense-enumeration/experiments/
    2026-08-03-sorted-crossing-boundary-owner-release-realization/unit.md

What this module is
-------------------
``unit.md`` -- "Secondary downstream compatibility" -- defers a *separate*
teacher-forced readout until every primary branch is sealed.  The sealed CPU
plan already materialises those 64 requests under
``readout_tier == "secondary_sealed_after_primary_branches"``; the primary
scorer deliberately refuses to execute them and only counts them.  This module
is the pass that executes them, after -- and only after -- a complete sealed
primary analysis directory proves the branch gate is closed.

The three sealed variants, in the plan's own vocabulary:

``p_plus_c_then_e`` (26, non-optional)
    Append the exact clean GT row ``C`` to the native ``P``, then teacher-force
    the exact native full-row ``E``.  Paired against the *identical* ``E`` token
    sequence teacher-forced at the unmodified native ``P``.
``p_plus_e_plus_c_then_f`` (26, plan-optional, all materialised in this plan)
    Append ``C`` to the native ``P+E``, then teacher-force the exact native
    ``F``.  Paired against the identical ``F`` tokens at the unmodified ``P+E``.
``benign_substitution_then_following_native_action`` (12, non-optional)
    Append the exact clean GT twin of a natively emitted true-positive row to
    that row's due-boundary predecessor, then force the exact following native
    action.  Paired against the same following native action at the *unmodified
    native successor context* the plan seals.

Both roots of a pair force byte-identical target token ids, so every reported
number is a within-owner, within-target paired quantity.  Reported per request:
raw selected-token log probabilities per token on both roots, exact segment
sums, and modified-minus-baseline deltas for the description path (through
``<|box_start|>``), the four coordinate tokens, and the complete row.

What this module is deliberately **not**
----------------------------------------
* It never assigns, reads, or re-derives a primary branch.  The sealed primary
  analysis is verified as a *gate* -- counts, digests, routing decision, absence
  of secondary fields -- and its branch labels are never used to choose which
  requests run.  Every sealed secondary request of the selected image executes.
* It is not a retention, recovery, or rollout result.  ``unit.md``: these
  readouts "measure local compatibility of exact row sequences; they are not
  final-set retention, eventual recovery, or free-rollout results."  That
  boundary travels in every row and every receipt (:data:`CLAIM_BOUNDARY`).
* It never retokenizes, never samples, never calls ``model.generate()``, and
  never re-implements a Qwen runtime: the teacher-forcing, session, publish,
  admission and identity primitives are imported from the primary scorer.

Outputs (one explicit shard directory, published atomically)::

    secondary-compatibility-rows.jsonl     one row per executed request
    secondary-compatibility-parity.json    backend + cached/uncached parity seam
    secondary-compatibility-receipt.json   identities, digests, counters, policy

A ``--mode smoke`` shard publishes ``secondary-compatibility-admission.json``
beside its parity and receipt and no evidence rows.  A shard stopped by the
quarantine rule publishes ``secondary-compatibility-quarantine.json`` and its
receipt alone: on mismatch there is no partial evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import (  # noqa: E402
    analyze_sorted_crossing_boundary_owner_release as analyzer,
)
from scripts.research import (  # noqa: E402
    prepare_sorted_crossing_boundary_owner_release_realization as plan_builder,
)
from scripts.research import (  # noqa: E402
    score_sorted_crossing_boundary_owner_release as primary,
)

# ---------------------------------------------------------------------------
# 0. Frozen schema / tier / variant constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "sorted_crossing_boundary_secondary_compatibility_rows.v1"
RECEIPT_SCHEMA_VERSION = "sorted_crossing_boundary_secondary_compatibility_receipt.v1"
PARITY_SCHEMA_VERSION = "sorted_crossing_boundary_secondary_compatibility_parity.v1"
QUARANTINE_SCHEMA_VERSION = (
    "sorted_crossing_boundary_secondary_compatibility_quarantine.v1"
)
ADMISSION_SCHEMA_VERSION = (
    "sorted_crossing_boundary_secondary_compatibility_admission.v1"
)

UNIT_ID = primary.UNIT_ID

ROWS_NAME = "secondary-compatibility-rows.jsonl"
PARITY_NAME = "secondary-compatibility-parity.json"
RECEIPT_NAME = "secondary-compatibility-receipt.json"
QUARANTINE_NAME = "secondary-compatibility-quarantine.json"
ADMISSION_NAME = "secondary-compatibility-admission.json"

#: Files that are *secondary evidence*.  A quarantined shard leaves none of them.
EVIDENCE_OUTPUT_NAMES: tuple[str, ...] = (ROWS_NAME, PARITY_NAME)

MODE_SMOKE = primary.MODE_SMOKE
MODE_CAPTURE = primary.MODE_CAPTURE

#: The only readout tier this module executes.  The primary pass executes only
#: ``"primary"`` and defers exactly this tier, so the two never overlap.
SECONDARY_READOUT_TIER = "secondary_sealed_after_primary_branches"
REQUEST_FAMILY = plan_builder.REQUEST_DOWNSTREAM_COMPATIBILITY
APPENDED_ROLE = "inserted_exact_clean_gt_row_c"

VARIANT_P_PLUS_C_THEN_E = "p_plus_c_then_e"
VARIANT_P_PLUS_E_PLUS_C_THEN_F = "p_plus_e_plus_c_then_f"
VARIANT_BENIGN_SUBSTITUTION = "benign_substitution_then_following_native_action"
SECONDARY_VARIANTS: tuple[str, ...] = (
    VARIANT_P_PLUS_C_THEN_E,
    VARIANT_P_PLUS_E_PLUS_C_THEN_F,
    VARIANT_BENIGN_SUBSTITUTION,
)

#: The sealed cohort each variant may belong to.  A benign-substitution request
#: attributed to the primary cohort (or the reverse) would silently move a
#: control readout into the primary case series.
VARIANT_COHORT: Mapping[str, str] = {
    VARIANT_P_PLUS_C_THEN_E: plan_builder.PRIMARY_COHORT,
    VARIANT_P_PLUS_E_PLUS_C_THEN_F: plan_builder.PRIMARY_COHORT,
    VARIANT_BENIGN_SUBSTITUTION: plan_builder.TP_REPLAY_CONTROL_COHORT,
}

#: ``unit.md``: only the late catch-up ``F`` extension is optional.
VARIANT_IS_PLAN_OPTIONAL: Mapping[str, bool] = {
    VARIANT_P_PLUS_C_THEN_E: False,
    VARIANT_P_PLUS_E_PLUS_C_THEN_F: True,
    VARIANT_BENIGN_SUBSTITUTION: False,
}

#: The frozen secondary request census of this plan.  ``unit.md`` puts the
#: primary compatibility readout on all 26 crossing owners, the optional ``F``
#: extension on every owner whose ``F`` exists (all 26 here), and the benign
#: substitution reference on the twelve TP replay controls.
EXPECTED_REQUEST_COUNT_BY_VARIANT: Mapping[str, int] = {
    VARIANT_P_PLUS_C_THEN_E: primary.PRIMARY_OWNER_COUNT_U,
    VARIANT_P_PLUS_E_PLUS_C_THEN_F: primary.PRIMARY_OWNER_COUNT_U,
    VARIANT_BENIGN_SUBSTITUTION: primary.TP_CALIBRATION_OWNER_COUNT,
}
EXPECTED_SECONDARY_REQUEST_COUNT = sum(EXPECTED_REQUEST_COUNT_BY_VARIANT.values())

#: The two paired roots.  They are separate logical context groups with their
#: own prefill, never one cache cropped back and reused.
ROOT_BASELINE = "baseline_unmodified_native_root"
ROOT_MODIFIED = "modified_inserted_clean_row_c_root"
PAIRED_ROOTS: tuple[str, ...] = (ROOT_BASELINE, ROOT_MODIFIED)

#: The three reported segments of one exact native row.
SEGMENT_DESCRIPTION = "description"
SEGMENT_COORDINATES = "coordinates"
SEGMENT_COMPLETE_ROW = "complete_row"
SEGMENTS: tuple[str, ...] = (
    SEGMENT_DESCRIPTION,
    SEGMENT_COORDINATES,
    SEGMENT_COMPLETE_ROW,
)

#: Token identity, taken from the primary scorer so the two cannot drift.
OBJECT_REF_START = primary.OBJECT_REF_START
OBJECT_REF_END = primary.OBJECT_REF_END
BOX_START = primary.BOX_START
BOX_END = primary.BOX_END
COORDINATE_TOKEN_ID_START = primary.COORDINATE_TOKEN_ID_START
COORDINATE_TOKEN_ID_END_EXCLUSIVE = primary.COORDINATE_TOKEN_ID_END_EXCLUSIVE
COORDINATE_TOKEN_COUNT = primary.COORDINATE_TOKEN_COUNT

KV_CACHE_BACKEND = primary.KV_CACHE_BACKEND
UNCACHED_BACKEND = primary.UNCACHED_BACKEND
CACHE_ADMITTED = primary.CACHE_ADMITTED
UNCACHED_FALLBACK = primary.UNCACHED_FALLBACK
CACHE_PARITY_MAX_SELECTED_LOGIT_ABS_DIFF = (
    primary.CACHE_PARITY_MAX_SELECTED_LOGIT_ABS_DIFF
)

LIKELIHOOD_CHANNEL = primary.LIKELIHOOD_CHANNEL
NATIVE_REPETITION_PENALTY_STRATUM = primary.NATIVE_REPETITION_PENALTY_STRATUM

#: ``unit.md`` "Secondary downstream compatibility" and "Not claimed".
CLAIM_BOUNDARY = (
    "local compatibility of exact row sequences at one boundary; never final-set "
    "retention, eventual recovery, natural stop, or a free-rollout result"
)

#: The sealed primary analysis directory this pass gates on.
ANALYSIS_OWNER_ROWS_NAME = analyzer.OWNER_ROWS_NAME
ANALYSIS_REPORT_JSON_NAME = analyzer.REPORT_JSON_NAME
ANALYSIS_REPORT_MD_NAME = analyzer.REPORT_MD_NAME
ANALYSIS_RECEIPT_NAME = analyzer.RECEIPT_NAME
ANALYSIS_REQUIRED_FILES: tuple[str, ...] = (
    ANALYSIS_OWNER_ROWS_NAME,
    ANALYSIS_REPORT_JSON_NAME,
    ANALYSIS_REPORT_MD_NAME,
    ANALYSIS_RECEIPT_NAME,
)
#: Digested outputs the analysis receipt must reconcile byte-for-byte.
ANALYSIS_DIGESTED_OUTPUTS: tuple[str, ...] = (
    ANALYSIS_OWNER_ROWS_NAME,
    ANALYSIS_REPORT_JSON_NAME,
    ANALYSIS_REPORT_MD_NAME,
)
#: Source/lineage digests the sealed analysis must carry for the gate to close.
ANALYSIS_REQUIRED_DIGEST_FIELDS: tuple[str, ...] = (
    "analyzer_source_sha256",
    "scorer_source_sha256",
    "merger_source_sha256",
    "merge_receipt_content_sha256",
    "runtime_identity_sha256",
)
ANALYSIS_DECISIONS: frozenset[str] = frozenset(
    {analyzer.DECISION_ROUTE, analyzer.DECISION_CLOSE, analyzer.DECISION_GATE_FAILED}
)

#: The one analysis artifact revision this gate admits.  Pinned to the live
#: analyzer's own constants: a secondary readout is only interpretable beside the
#: branch registry that gates it, so an analysis emitted by any other analyzer
#: revision -- earlier or later -- fails closed instead of being reconciled.
ANALYSIS_RECEIPT_SCHEMA_VERSION = analyzer.RECEIPT_SCHEMA_VERSION
ANALYSIS_REPORT_SCHEMA_VERSION = analyzer.REPORT_SCHEMA_VERSION
ANALYSIS_OWNER_ROW_SCHEMA_VERSION = analyzer.OWNER_ROW_SCHEMA_VERSION
ANALYSIS_MERGE_RECEIPT_NAME = "merge-receipt.json"
PRIMARY_BRANCHES = frozenset(primary.BRANCH_ORDER)

#: Any key naming a secondary readout is forbidden inside a *primary* analysis
#: artifact: the gate exists to prove the branch pass never saw one.
FORBIDDEN_SECONDARY_KEY_FRAGMENTS: tuple[str, ...] = ("secondary", "compatibility")

#: Modules whose source determines this pass's semantics.
SOURCE_IDENTITY_MODULES: tuple[str, ...] = (
    "scripts.research.score_sorted_crossing_boundary_owner_release_secondary",
    *primary.SOURCE_IDENTITY_MODULES,
    "scripts.research.analyze_sorted_crossing_boundary_owner_release",
)

#: This path teacher-forces one literal span per root; there is no candidate
#: family to spread over equal-shape lanes, and the two roots differ, so they
#: cannot share one copied prefill.  ``--batch-size`` is accepted for CLI parity
#: with the primary scorer and reported, never silently pretended to apply.
BATCH_POLICY_NOT_APPLICABLE = (
    "single_lane_teacher_forcing_over_two_distinct_roots_no_shared_batched_root"
)
DEFAULT_BATCH_SIZE = primary.DEFAULT_BATCH_SIZE


class SecondaryCompatibilityContractError(primary.CrossingBoundaryContractError):
    """A precondition of the secondary compatibility capture was not proven."""


def _fail(message: str) -> NoReturn:
    raise SecondaryCompatibilityContractError(message)


# ---------------------------------------------------------------------------
# 1. Digest / IO helpers (delegated so the canonical form cannot drift)
# ---------------------------------------------------------------------------

canonical_json_bytes = primary.canonical_json_bytes
sha256_json = primary.sha256_json
sha256_bytes = primary.sha256_bytes
sha256_file = primary.sha256_file


def _token_ids(value: Any, *, label: str) -> list[int]:
    return primary._token_ids(value, label=label)  # noqa: SLF001


def _read_json(path: Path, label: str) -> dict[str, Any]:
    return primary._read_json(Path(path), label)  # noqa: SLF001


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    return primary._read_jsonl(Path(path), label)  # noqa: SLF001


def _hex64(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        _fail(f"{label} is not a sha256 hex digest ({value!r})")
    return text


def current_analyzer_source_sha256() -> str:
    """The digest of the analyzer this checkout would run.

    The sealed analysis must have been produced by exactly this source: a branch
    registry emitted by a different analyzer revision is a different branch
    contract, and a secondary readout is only interpretable beside the registry
    that gates it.
    """

    return sha256_file(Path(analyzer.__file__).resolve())


def assert_no_secondary_fields(payload: Any, *, label: str) -> None:
    """Refuse any secondary/compatibility key, at any depth, inside a gate input.

    ``unit.md`` seals every primary branch *before* a secondary field is read.
    The sealed analysis therefore has to be free of one; finding a payload here
    would mean the branch that gates this pass was decided while looking at the
    readout this pass produces.
    """

    if isinstance(payload, Mapping):
        present = sorted(
            str(key)
            for key in payload
            if any(
                fragment in str(key).lower()
                for fragment in FORBIDDEN_SECONDARY_KEY_FRAGMENTS
            )
        )
        if present:
            _fail(
                f"{label} carries secondary readout key(s) {present!r}; the sealed primary "
                "analysis must have been decided before any secondary field existed"
            )
        for value in payload.values():
            assert_no_secondary_fields(value, label=label)
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for value in payload:
            assert_no_secondary_fields(value, label=label)


# ---------------------------------------------------------------------------
# 2. The sealed primary-branch gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SealedPrimaryAnalysis:
    """A complete, self-sealed primary analysis, bound to this exact plan."""

    analysis_dir: Path
    receipt: dict[str, Any]
    report: dict[str, Any]
    owner_rows: list[dict[str, Any]]
    file_sha256: dict[str, str]
    branch_counts: dict[str, int]
    branch_row_count: int
    binding: dict[str, Any]

    @property
    def binding_sha256(self) -> str:
        return sha256_json(self.binding)

    @property
    def decision(self) -> str:
        return str(self.receipt["decision"])


def _assert_self_sealed(
    payload: Mapping[str, Any], *, seal_key: str, label: str
) -> None:
    declared = payload.get(seal_key)
    unsealed = {key: value for key, value in payload.items() if key != seal_key}
    if sha256_json(unsealed) != declared:
        _fail(f"{label} does not self-seal; it has been edited after it was written")


def _assert_analysis_directory_shape(analysis_dir: Path) -> dict[str, str]:
    if not analysis_dir.is_dir():
        _fail(f"primary analysis directory {analysis_dir} does not exist")
    observed = sorted(child.name for child in analysis_dir.iterdir())
    missing = sorted(set(ANALYSIS_REQUIRED_FILES) - set(observed))
    unknown = sorted(set(observed) - set(ANALYSIS_REQUIRED_FILES))
    if missing:
        _fail(
            f"primary analysis directory {analysis_dir} is incomplete (missing {missing!r}); "
            "an incomplete analysis never seals the branch gate"
        )
    if unknown:
        _fail(
            f"primary analysis directory {analysis_dir} carries unknown artifact(s) "
            f"{unknown!r}; an unrecognised file in a sealed directory fails closed"
        )
    return {name: sha256_file(analysis_dir / name) for name in ANALYSIS_REQUIRED_FILES}


def _assert_analysis_branch_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Exactly the frozen 26 primary branch rows, each carrying one branch."""

    for row in rows:
        if str(row.get("unit_id")) != UNIT_ID:
            _fail("the sealed analysis owner rows carry a row from another unit")
        if str(row.get("schema_version")) != ANALYSIS_OWNER_ROW_SCHEMA_VERSION:
            _fail(
                f"sealed analysis owner row schema {row.get('schema_version')!r} is not "
                f"{ANALYSIS_OWNER_ROW_SCHEMA_VERSION!r}"
            )
        assert_no_secondary_fields(row, label="sealed analysis owner row")
    branch_rows = [
        row for row in rows if str(row.get("cohort")) == plan_builder.PRIMARY_COHORT
    ]
    if len(branch_rows) != primary.PRIMARY_OWNER_COUNT_U:
        _fail(
            f"the sealed analysis carries {len(branch_rows)} primary branch rows, not the "
            f"frozen {primary.PRIMARY_OWNER_COUNT_U}; the branch gate is not closed over the "
            "whole cohort"
        )
    owner_ids = [str(row.get("gt_owner_id")) for row in branch_rows]
    if len(set(owner_ids)) != len(owner_ids):
        _fail("the sealed analysis carries a duplicated primary owner")
    counts: dict[str, int] = {}
    for row in branch_rows:
        branch = str(row.get("primary_branch"))
        if branch not in PRIMARY_BRANCHES:
            _fail(
                f"sealed analysis owner {row.get('gt_owner_id')!r} carries unknown primary "
                f"branch {branch!r}"
            )
        counts[branch] = counts.get(branch, 0) + 1
    return dict(sorted(counts.items()))


def _bind_analysis_to_plan(
    receipt: Mapping[str, Any], *, plan: primary.SealedPlan, analysis_dir: Path
) -> dict[str, Any]:
    """Re-prove that the sealed analysis was decided over *this* sealed plan.

    The analysis directory names its merged inputs but not the CPU plan; the
    merge receipt it seals is what carries the plan manifest digest.  Without
    this hop a secondary shard could bind an analysis of a different plan cut
    and pair its rows against a cohort that analysis never saw.
    """

    merged_dir = Path(str(receipt.get("merged_dir")))
    merge_path = merged_dir / ANALYSIS_MERGE_RECEIPT_NAME
    if not merge_path.is_file():
        _fail(
            f"the sealed analysis at {analysis_dir} names merged directory {merged_dir}, whose "
            f"{ANALYSIS_MERGE_RECEIPT_NAME} is not readable; the analysis cannot be bound to a "
            "sealed plan"
        )
    declared_inputs = receipt.get("input_file_sha256")
    if not isinstance(declared_inputs, Mapping) or not declared_inputs:
        _fail("the sealed analysis receipt declares no input_file_sha256")
    for name, digest in declared_inputs.items():
        _hex64(digest, label=f"analysis input digest for {name!r}")
    if ANALYSIS_MERGE_RECEIPT_NAME not in declared_inputs:
        _fail(
            f"the sealed analysis receipt seals no digest for {ANALYSIS_MERGE_RECEIPT_NAME}"
        )
    observed_merge_file = sha256_file(merge_path)
    if observed_merge_file != str(declared_inputs[ANALYSIS_MERGE_RECEIPT_NAME]):
        _fail(
            f"{merge_path} hashes to {observed_merge_file}, not the digest the sealed analysis "
            "consumed; the merged evidence drifted after the branch pass"
        )
    merge_receipt = _read_json(merge_path, "primary merge receipt")
    if str(merge_receipt.get("unit_id")) != UNIT_ID:
        _fail("the primary merge receipt belongs to another unit")
    if str(merge_receipt.get("receipt_content_sha256")) != str(
        receipt.get("merge_receipt_content_sha256")
    ):
        _fail(
            "the primary merge receipt's own content digest is not the one the sealed analysis "
            "recorded; the two artifacts do not describe one merge"
        )
    merge_plan = merge_receipt.get("plan")
    if not isinstance(merge_plan, Mapping):
        _fail("the primary merge receipt seals no plan block")
    if str(merge_plan.get("manifest_content_sha256")) != str(
        plan.manifest.get("manifest_content_sha256")
    ):
        _fail(
            "the sealed primary analysis was decided over a different CPU plan manifest than "
            "the one this secondary capture executes; refusing to pair secondary rows against "
            "a cohort that analysis never saw"
        )
    merge_policy = merge_receipt.get("policy") or {}
    if bool(merge_policy.get("secondary_compatibility_merged")):
        _fail(
            "the primary merge receipt declares secondary compatibility rows were merged; the "
            "branch pass must be free of them"
        )
    if str(merge_receipt.get("runtime_identity_sha256")) != str(
        receipt.get("runtime_identity_sha256")
    ):
        _fail(
            "the sealed analysis and its merge receipt disagree about the runtime identity the "
            "primary evidence was captured under"
        )
    return {
        "merged_dir": str(merged_dir),
        "merge_receipt_path": str(merge_path),
        "merge_receipt_file_sha256": observed_merge_file,
        "merge_receipt_content_sha256": str(receipt.get("merge_receipt_content_sha256")),
        "plan_manifest_content_sha256": str(merge_plan.get("manifest_content_sha256")),
        "merge_scorer_source_sha256": merge_receipt.get("scorer_source_sha256"),
        "merge_merger_source_sha256": merge_receipt.get("merger_source_sha256"),
    }


def load_sealed_primary_analysis(
    analysis_dir: Path, *, plan: primary.SealedPlan
) -> SealedPrimaryAnalysis:
    """Prove the primary branch gate is closed, before any model is loaded.

    ``unit.md``: "Seal all primary branch assignments before reading any
    secondary field."  Everything this function checks is a precondition of
    *executing* the secondary pass, so it runs on the CPU path and fails closed
    rather than being reported alongside the evidence it would have gated.
    """

    analysis_dir = Path(analysis_dir)
    file_sha256 = _assert_analysis_directory_shape(analysis_dir)

    receipt = _read_json(analysis_dir / ANALYSIS_RECEIPT_NAME, "primary analysis receipt")
    if str(receipt.get("schema_version")) != ANALYSIS_RECEIPT_SCHEMA_VERSION:
        _fail(
            f"primary analysis receipt schema {receipt.get('schema_version')!r} is not "
            f"{ANALYSIS_RECEIPT_SCHEMA_VERSION!r}"
        )
    if str(receipt.get("unit_id")) != UNIT_ID:
        _fail("primary analysis receipt belongs to another unit")
    _assert_self_sealed(
        receipt, seal_key="receipt_content_sha256", label="primary analysis receipt"
    )

    declared_outputs = receipt.get("output_file_digests")
    if not isinstance(declared_outputs, Mapping):
        _fail("primary analysis receipt carries no output_file_digests")
    for name in ANALYSIS_DIGESTED_OUTPUTS:
        entry = declared_outputs.get(name)
        if not isinstance(entry, Mapping):
            _fail(f"primary analysis receipt declares no digest for {name}")
        if str(entry.get("sha256")) != file_sha256[name]:
            _fail(
                f"primary analysis file {name} hashes to {file_sha256[name]}, not the receipt's "
                f"{entry.get('sha256')}"
            )
        observed_size = (analysis_dir / name).stat().st_size
        if int(entry.get("byte_size", -1)) != observed_size:
            _fail(
                f"primary analysis file {name} is {observed_size} bytes, not the receipt's "
                f"{entry.get('byte_size')}"
            )

    report = _read_json(analysis_dir / ANALYSIS_REPORT_JSON_NAME, "primary analysis report")
    if str(report.get("schema_version")) != ANALYSIS_REPORT_SCHEMA_VERSION:
        _fail(
            f"primary analysis report schema {report.get('schema_version')!r} is not "
            f"{ANALYSIS_REPORT_SCHEMA_VERSION!r}"
        )
    if str(report.get("unit_id")) != UNIT_ID:
        _fail("primary analysis report belongs to another unit")

    owner_rows = _read_jsonl(
        analysis_dir / ANALYSIS_OWNER_ROWS_NAME, "primary analysis owner rows"
    )
    if len(owner_rows) != int(declared_outputs[ANALYSIS_OWNER_ROWS_NAME]["row_count"]):
        _fail(
            f"primary analysis owner rows hold {len(owner_rows)} rows, not the receipt's "
            f"{declared_outputs[ANALYSIS_OWNER_ROWS_NAME]['row_count']}"
        )
    branch_counts = _assert_analysis_branch_rows(owner_rows)

    decision = str(receipt.get("decision"))
    if decision not in ANALYSIS_DECISIONS:
        _fail(
            f"the sealed analysis declares unknown routing decision {decision!r}; the branch "
            "gate is not interpretable"
        )
    if decision != str(report.get("decision")):
        _fail("the sealed analysis receipt and report disagree about the routing decision")
    if int(receipt.get("primary_denominator", -1)) != primary.PRIMARY_OWNER_COUNT_U:
        _fail(
            f"the sealed analysis declares primary denominator "
            f"{receipt.get('primary_denominator')!r}, not the frozen "
            f"{primary.PRIMARY_OWNER_COUNT_U}"
        )
    routing = report.get("routing") or {}
    if int(receipt.get("routing_denominator_count", -1)) != int(
        routing.get("routing_denominator_count", -2)
    ):
        _fail(
            "the sealed analysis receipt and report disagree about the routing denominator"
        )
    gate = report.get("interpretability_gate") or {}
    if bool(receipt.get("interpretability_gate_passed")) != bool(gate.get("passed")):
        _fail(
            "the sealed analysis receipt and report disagree about the interpretability gate"
        )
    policy = receipt.get("policy") or {}
    if policy.get("secondary_compatibility_read") is not False:
        _fail(
            "the sealed analysis does not declare secondary_compatibility_read=false; its "
            "branches may have been decided while reading the readout this pass produces"
        )
    for field_name in ANALYSIS_REQUIRED_DIGEST_FIELDS:
        _hex64(receipt.get(field_name), label=f"sealed analysis {field_name}")
    declared_analyzer_sha256 = str(receipt.get("analyzer_source_sha256"))
    executed_analyzer_sha256 = current_analyzer_source_sha256()
    if declared_analyzer_sha256 != executed_analyzer_sha256:
        _fail(
            f"the sealed analysis was produced by analyzer source {declared_analyzer_sha256}, "
            f"but the analyzer in this checkout hashes to {executed_analyzer_sha256}; the "
            "branch registry this pass gates on is not the one the current analyzer emits"
        )

    merge_binding = _bind_analysis_to_plan(receipt, plan=plan, analysis_dir=analysis_dir)

    binding = {
        "unit_id": UNIT_ID,
        "analysis_dir": str(analysis_dir),
        "analysis_file_sha256": dict(sorted(file_sha256.items())),
        "receipt_content_sha256": str(receipt.get("receipt_content_sha256")),
        "analyzer_source_sha256": declared_analyzer_sha256,
        "primary_scorer_source_sha256": str(receipt.get("scorer_source_sha256")),
        "merger_source_sha256": str(receipt.get("merger_source_sha256")),
        "runtime_identity_sha256": str(receipt.get("runtime_identity_sha256")),
        "input_file_sha256": dict(sorted(receipt["input_file_sha256"].items())),
        "decision": decision,
        "primary_denominator": int(receipt["primary_denominator"]),
        "routing_denominator_count": int(receipt["routing_denominator_count"]),
        "interpretability_gate_passed": bool(receipt["interpretability_gate_passed"]),
        "branch_row_count": primary.PRIMARY_OWNER_COUNT_U,
        "branch_counts": branch_counts,
        "secondary_fields_present": False,
        "branch_labels_used_to_select_requests": False,
        **merge_binding,
    }
    return SealedPrimaryAnalysis(
        analysis_dir=analysis_dir,
        receipt=receipt,
        report=report,
        owner_rows=owner_rows,
        file_sha256=file_sha256,
        branch_counts=branch_counts,
        branch_row_count=primary.PRIMARY_OWNER_COUNT_U,
        binding=binding,
    )


# ---------------------------------------------------------------------------
# 3. Sealed secondary request contract
# ---------------------------------------------------------------------------


def assert_secondary_request(row: Mapping[str, Any]) -> str:
    """Tier, family, variant, cohort, optionality and appended role, or fail."""

    request_id = str(row.get("request_id"))
    if str(row.get("unit_id")) != UNIT_ID:
        _fail(f"request {request_id!r} belongs to another unit")
    if str(row.get("readout_tier")) != SECONDARY_READOUT_TIER:
        _fail(
            f"request {request_id!r} is tier {row.get('readout_tier')!r}, not "
            f"{SECONDARY_READOUT_TIER!r}; this pass executes only the deferred secondary tier"
        )
    if str(row.get("request_family")) != REQUEST_FAMILY:
        _fail(
            f"request {request_id!r} is family {row.get('request_family')!r}, not "
            f"{REQUEST_FAMILY!r}"
        )
    variant = str(row.get("variant"))
    if variant not in SECONDARY_VARIANTS:
        _fail(f"request {request_id!r} declares unknown secondary variant {variant!r}")
    expected_cohort = VARIANT_COHORT[variant]
    if str(row.get("cohort")) != expected_cohort:
        _fail(
            f"request {request_id!r} at variant {variant!r} declares cohort "
            f"{row.get('cohort')!r}, not the sealed {expected_cohort!r}"
        )
    if bool(row.get("optional")) is not VARIANT_IS_PLAN_OPTIONAL[variant]:
        _fail(
            f"request {request_id!r} at variant {variant!r} declares optional="
            f"{row.get('optional')!r}, not the sealed "
            f"{VARIANT_IS_PLAN_OPTIONAL[variant]!r}"
        )
    if list(row.get("branch_inputs") or []) != []:
        _fail(
            f"request {request_id!r} declares branch inputs; a secondary readout never feeds a "
            "primary branch"
        )
    prefix = row.get("prefix")
    if not isinstance(prefix, Mapping):
        _fail(f"request {request_id!r} carries no prefix binding")
    if str(prefix.get("appended_role")) != APPENDED_ROLE:
        _fail(
            f"request {request_id!r} appends role {prefix.get('appended_role')!r}, not the "
            f"sealed {APPENDED_ROLE!r}"
        )
    if bool(prefix.get("retokenized")):
        _fail(f"request {request_id!r} declares a retokenized prefix")
    return variant


def secondary_request_rows(plan: primary.SealedPlan) -> list[dict[str, Any]]:
    """Every sealed secondary request of the plan, validated and deduplicated."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in plan.request_rows:
        if str(row.get("readout_tier")) != SECONDARY_READOUT_TIER:
            continue
        assert_secondary_request(row)
        request_id = str(row["request_id"])
        if request_id in seen:
            _fail(f"the sealed plan carries a duplicate secondary request {request_id!r}")
        seen.add(request_id)
        rows.append(dict(row))
    return rows


def counts_by_variant(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {variant: 0 for variant in SECONDARY_VARIANTS}
    for row in rows:
        counts[str(row["variant"])] += 1
    return counts


def validate_secondary_plan_counts(
    rows: Sequence[Mapping[str, Any]], *, manifest: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Re-derive the frozen 26/26/12 secondary census before anything executes."""

    observed = counts_by_variant(rows)
    drifted = sorted(
        variant
        for variant in SECONDARY_VARIANTS
        if observed[variant] != EXPECTED_REQUEST_COUNT_BY_VARIANT[variant]
    )
    if drifted or len(rows) != EXPECTED_SECONDARY_REQUEST_COUNT:
        _fail(
            f"the sealed plan holds {len(rows)} secondary requests {observed!r}, not the frozen "
            f"{EXPECTED_SECONDARY_REQUEST_COUNT} {dict(EXPECTED_REQUEST_COUNT_BY_VARIANT)!r} "
            f"(drifted variant(s): {drifted!r})"
        )
    if manifest is not None:
        sealed_f_rows = int(
            (manifest.get("cohort_counts") or {}).get("f_row_present_count", -1)
        )
        if sealed_f_rows != observed[VARIANT_P_PLUS_E_PLUS_C_THEN_F]:
            _fail(
                f"the plan manifest seals {sealed_f_rows!r} owners with an F row but "
                f"{observed[VARIANT_P_PLUS_E_PLUS_C_THEN_F]} optional F requests were "
                "materialized; an optional readout is never materialized without its row"
            )
    return {
        "expected_total": EXPECTED_SECONDARY_REQUEST_COUNT,
        "observed_total": len(rows),
        "expected_by_variant": dict(EXPECTED_REQUEST_COUNT_BY_VARIANT),
        "observed_by_variant": observed,
    }


def secondary_requests_for_image(
    rows: Sequence[Mapping[str, Any]], *, image_id: str
) -> list[dict[str, Any]]:
    """Every sealed secondary request of one image, in stable request-id order.

    Selection is by sealed image id alone.  No branch label, score, or stratum
    participates, so the executed set cannot be narrowed by what the primary
    pass concluded.
    """

    selected = [dict(row) for row in rows if str(row["image_id"]) == str(image_id)]
    if not selected:
        _fail(
            f"image {image_id!r} has no sealed secondary request; the sealed plan covers "
            f"image(s) {sorted({str(row['image_id']) for row in rows})!r}"
        )
    return sorted(selected, key=lambda row: str(row["request_id"]))


# ---------------------------------------------------------------------------
# 4. Exact row grammar and segment identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowSegments:
    """Literal half-open token spans of one exact native row.

    The boundaries are the row's own wrapper tokens, never a re-derived offset:
    ``description`` is the primary scorer's ``_description_path`` (through
    ``<|box_start|>``), ``coordinates`` is the frozen four-token coordinate
    slot, and ``complete_row`` is everything through ``<|box_end|>``.
    """

    description: tuple[int, int]
    coordinates: tuple[int, int]
    complete_row: tuple[int, int]
    box_start_index: int

    def span(self, segment: str) -> tuple[int, int]:
        if segment not in SEGMENTS:
            _fail(f"unknown row segment {segment!r}")
        return getattr(self, segment)


def row_segments(token_ids: Sequence[int]) -> RowSegments:
    """Validate the frozen row grammar and return its literal segment spans."""

    tokens = [int(value) for value in token_ids]
    if not tokens:
        _fail("an exact native row cannot be empty")
    if tokens[0] != OBJECT_REF_START:
        _fail(
            f"an exact native row must open with <|object_ref_start|>={OBJECT_REF_START}, not "
            f"{tokens[0]}"
        )
    if BOX_START not in tokens:
        _fail(
            "an exact native row must carry <|box_start|>; a terminal or partial action is not "
            "a compatibility target"
        )
    box_start_index = tokens.index(BOX_START)
    if box_start_index < 2 or tokens[box_start_index - 1] != OBJECT_REF_END:
        _fail("an exact native row must close its description with <|object_ref_end|>")
    expected_length = box_start_index + 1 + COORDINATE_TOKEN_COUNT + 1
    if len(tokens) != expected_length:
        _fail(
            f"an exact native row must hold exactly {COORDINATE_TOKEN_COUNT} coordinate tokens "
            f"then <|box_end|> ({expected_length} tokens), not {len(tokens)}"
        )
    coordinates = tokens[box_start_index + 1 : box_start_index + 1 + COORDINATE_TOKEN_COUNT]
    outside = [
        token
        for token in coordinates
        if not (
            COORDINATE_TOKEN_ID_START <= token < COORDINATE_TOKEN_ID_END_EXCLUSIVE
        )
    ]
    if outside:
        _fail(f"coordinate slot carries out-of-domain token(s) {outside!r}")
    if tokens[-1] != BOX_END:
        _fail(f"an exact native row must terminate with <|box_end|>={BOX_END}")
    description = primary._description_path(tokens)  # noqa: SLF001
    if description != tokens[: box_start_index + 1]:
        _fail("the description path does not reconstruct from the row's own wrapper tokens")
    return RowSegments(
        description=(0, box_start_index + 1),
        coordinates=(box_start_index + 1, box_start_index + 1 + COORDINATE_TOKEN_COUNT),
        complete_row=(0, len(tokens)),
        box_start_index=box_start_index,
    )


def segment_sums(
    segments: RowSegments, selected_logprobs: Sequence[float], *, label: str
) -> dict[str, dict[str, float | int]]:
    """Exact per-segment selected-logprob sums, token counts and token means."""

    values = [float(value) for value in selected_logprobs]
    if len(values) != segments.complete_row[1]:
        _fail(
            f"{label}: {len(values)} selected logprobs do not cover the row's "
            f"{segments.complete_row[1]} tokens"
        )
    nonfinite = [index for index, value in enumerate(values) if not math.isfinite(value)]
    if nonfinite:
        _fail(f"{label}: non-finite selected logprob at token index/indices {nonfinite!r}")
    sums: dict[str, dict[str, float | int]] = {}
    for segment in SEGMENTS:
        start, stop = segments.span(segment)
        window = values[start:stop]
        total = math.fsum(window)
        sums[segment] = {
            "sum": total,
            "token_count": len(window),
            "token_mean": total / len(window),
            "token_index_start": start,
            "token_index_stop": stop,
        }
    return sums


def paired_deltas(
    baseline: Mapping[str, Mapping[str, Any]], modified: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Modified-minus-baseline per segment, on identical token spans."""

    deltas: dict[str, dict[str, Any]] = {}
    for segment in SEGMENTS:
        left = baseline.get(segment)
        right = modified.get(segment)
        if left is None or right is None:
            _fail(f"segment {segment!r} is missing from one side of the pair")
        if int(left["token_count"]) != int(right["token_count"]) or int(
            left["token_index_start"]
        ) != int(right["token_index_start"]):
            _fail(
                f"segment {segment!r} does not cover the same token span on both roots; the "
                "pair is not comparable"
            )
        delta = float(right["sum"]) - float(left["sum"])
        if not math.isfinite(delta):
            _fail(f"segment {segment!r} produced a non-finite modified-minus-baseline delta")
        deltas[segment] = {
            "baseline_sum": float(left["sum"]),
            "modified_sum": float(right["sum"]),
            "delta": delta,
            "delta_token_mean": delta / int(left["token_count"]),
            "token_count": int(left["token_count"]),
            "sign": primary._margin_sign(delta),  # noqa: SLF001
        }
    return deltas


def assert_paired_token_identity(
    baseline_token_ids: Sequence[int],
    modified_token_ids: Sequence[int],
    *,
    declared_sha256: str,
    label: str,
) -> str:
    """Both roots must force byte-identical target tokens, or the pair is void."""

    left = [int(value) for value in baseline_token_ids]
    right = [int(value) for value in modified_token_ids]
    if left != right:
        _fail(
            f"{label}: the baseline and modified roots forced different target tokens; a "
            "modified-minus-baseline delta over two different rows is meaningless"
        )
    digest = sha256_json(left)
    if digest != str(declared_sha256):
        _fail(
            f"{label}: the forced target tokens hash to {digest}, not the sealed "
            f"{declared_sha256}"
        )
    return digest


# ---------------------------------------------------------------------------
# 5. Paired-root resolution against the sealed registries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedRootBinding:
    """The two sealed contexts one secondary request scores against."""

    variant: str
    gt_owner_id: str
    image_id: str
    modified_context_id: str
    baseline_context_id: str
    successor_context_id: str
    native_row_index: int
    registry_source: str


def _boundary_index_of(context: Mapping[str, Any], *, context_id: str) -> int:
    value = context.get("boundary_index")
    if value is None:
        _fail(f"sealed context {context_id!r} declares no boundary index")
    return int(value)


def _context_at_boundary(
    plan: primary.SealedPlan, *, image_id: str, boundary_index: int, label: str
) -> str:
    matches = sorted(
        str(context_id)
        for context_id, context in plan.contexts_by_id.items()
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
    registry_row: Mapping[str, Any], *, variant: str, gt_owner_id: str
) -> tuple[str, str]:
    """The baseline context the *plan* seals for this variant, plus its source."""

    if variant == VARIANT_P_PLUS_C_THEN_E:
        return str(registry_row["e_row"]["pre_row_context_id"]), "cohort_registry.e_row"
    if variant == VARIANT_P_PLUS_E_PLUS_C_THEN_F:
        if not bool(registry_row.get("f_row_present")):
            _fail(
                f"owner {gt_owner_id!r} has a materialized optional F request but the sealed "
                "cohort registry declares no F row"
            )
        return str(registry_row["f_row"]["pre_row_context_id"]), "cohort_registry.f_row"
    following = registry_row.get("following_native_action")
    if not isinstance(following, Mapping):
        _fail(
            f"TP replay control {gt_owner_id!r} seals no following native action; the benign "
            "substitution has no unmodified native successor context to pair against"
        )
    return str(following["context_id"]), "control_registry.following_native_action"


def resolve_paired_roots(
    plan: primary.SealedPlan,
    request: Mapping[str, Any],
    registry_row: Mapping[str, Any],
) -> PairedRootBinding:
    """Bind one request to its modified root and its unmodified native baseline.

    The baseline is derived twice and both derivations must agree: once from the
    registry row the plan sealed (``pre_row_context_id`` / the TP control's
    ``following_native_action.context_id``) and once from the boundary
    convention (the context whose boundary index is the scored row's native
    index).  The exact-row identity is then re-proven the way ``unit.md``
    requires -- the literal suffix between two adjacent sealed boundaries -- so
    no full-row token sequence is ever taken on a sidecar's word.
    """

    variant = str(request["variant"])
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

    sealed_baseline, registry_source = _sealed_baseline_context(
        registry_row, variant=variant, gt_owner_id=gt_owner_id
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
            f"registry ({sealed_baseline!r}) and the boundary convention "
            f"({derived_baseline!r})"
        )
    if plan.image_of_context(sealed_baseline) != image_id:
        _fail(f"baseline context {sealed_baseline!r} belongs to another image")

    modified_context = plan.contexts_by_id.get(modified_context_id)
    if modified_context is None:
        _fail(f"context {modified_context_id!r} is absent from the sealed context registry")
    modified_boundary = _boundary_index_of(
        modified_context, context_id=modified_context_id
    )
    if variant == VARIANT_BENIGN_SUBSTITUTION:
        # The clean twin replaces the native TP row, so the modified root is the
        # row's *predecessor* boundary and the baseline is the native successor.
        if modified_boundary + 1 != native_row_index:
            _fail(
                f"benign substitution {request['request_id']!r} appends at boundary "
                f"{modified_boundary} but scores native row {native_row_index}; the clean twin "
                "must replace exactly the row between them"
            )
    elif modified_context_id != sealed_baseline:
        _fail(
            f"request {request['request_id']!r} at variant {variant!r} must append to the same "
            f"native boundary it scores against ({sealed_baseline!r}), not "
            f"{modified_context_id!r}"
        )

    successor_context_id = _context_at_boundary(
        plan,
        image_id=image_id,
        boundary_index=native_row_index + 1,
        label=f"successor of the baseline root of {request['request_id']!r}",
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
        variant=variant,
        gt_owner_id=gt_owner_id,
        image_id=image_id,
        modified_context_id=modified_context_id,
        baseline_context_id=sealed_baseline,
        successor_context_id=successor_context_id,
        native_row_index=native_row_index,
        registry_source=registry_source,
    )


def assert_inserted_row_matches_registry(
    request: Mapping[str, Any], registry_row: Mapping[str, Any]
) -> str:
    """The appended ``C`` is the owner's own sealed exact-GT-anchor row."""

    sealed = registry_row.get("inserted_clean_row_c")
    if not isinstance(sealed, Mapping):
        _fail(
            f"owner {request['gt_owner_id']!r} seals no inserted clean GT row; the appended "
            "prefix cannot be attributed"
        )
    declared = str(request["prefix"]["appended_token_ids_sha256"])
    if declared != str(sealed.get("token_ids_sha256")):
        _fail(
            f"request {request['request_id']!r} appends a row hashing to {declared}, not the "
            f"owner's sealed clean GT row {sealed.get('token_ids_sha256')}"
        )
    return declared


# ---------------------------------------------------------------------------
# 6. Cached-versus-uncached parity across the paired roots
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RootParityResult:
    """One root's numeric and discrete agreement between two executions."""

    root: str
    max_selected_logit_abs_diff: float
    argmax_parity: bool
    selected_is_argmax_parity: bool
    compared_token_count: int
    aligned: bool


@dataclass(frozen=True)
class SecondaryParityResult:
    """Whether cached execution is admissible evidence for this readout."""

    status: str
    max_selected_logit_abs_diff: float
    mismatched_fields: tuple[str, ...]
    per_root: tuple[RootParityResult, ...] = ()
    delta_sign_parity: bool = True


def parity_streams_by_root(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, primary.SurfaceParityStreams]:
    """Concatenate one execution's scored tokens per root, request-id ordered.

    Ordering by request id (not execution order) is what makes two executions
    comparable position by position; the id is derived from the sealed request
    identity, so equal ids mean equal literal token spans.
    """

    ordered = sorted(rows, key=lambda row: str(row["request_id"]))
    return {
        root: primary.SurfaceParityStreams(
            request_ids=tuple(str(row["request_id"]) for row in ordered),
            selected_logprobs=tuple(
                float(value)
                for row in ordered
                for value in row["roots"][root]["selected_logprobs"]
            ),
            argmax_token_ids=tuple(
                int(value)
                for row in ordered
                for value in row["roots"][root]["argmax_token_ids"]
            ),
        )
        for root in PAIRED_ROOTS
    }


def delta_signs_by_request(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """The decision-bearing discrete surface of this readout: per-segment signs."""

    return {
        str(row["request_id"]): {
            segment: int(row["deltas"][segment]["sign"]) for segment in SEGMENTS
        }
        for row in rows
    }


def selected_is_argmax_by_root(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[bool, ...]]:
    """Per-token "the forced token is rank one", the rank surface ``top_k=1`` owns."""

    ordered = sorted(rows, key=lambda row: str(row["request_id"]))
    return {
        root: tuple(
            int(token) == int(argmax)
            for row in ordered
            for token, argmax in zip(
                row["scored_token_ids"], row["roots"][root]["argmax_token_ids"], strict=True
            )
        )
        for root in PAIRED_ROOTS
    }


def evaluate_secondary_parity(
    *,
    cached_rows: Sequence[Mapping[str, Any]],
    uncached_rows: Sequence[Mapping[str, Any]],
) -> SecondaryParityResult:
    """Tolerance ``1e-3`` *and* exact argmax/rank/sign agreement, or fall back.

    Modelled on the primary scorer's parity seam: the reported maximum is the
    largest selected-logit difference over every compared token of every root,
    never one summary scalar, and every compared discrete field must be exactly
    preserved.
    """

    mismatched: list[str] = []
    cached_streams = parity_streams_by_root(cached_rows)
    uncached_streams = parity_streams_by_root(uncached_rows)
    per_root: list[RootParityResult] = []
    cached_rank = selected_is_argmax_by_root(cached_rows)
    uncached_rank = selected_is_argmax_by_root(uncached_rows)

    for root in PAIRED_ROOTS:
        left = cached_streams.get(root)
        right = uncached_streams.get(root)
        aligned = (
            left is not None
            and right is not None
            and left.request_ids == right.request_ids
            and len(left.selected_logprobs) == len(right.selected_logprobs)
            and len(left.argmax_token_ids) == len(right.argmax_token_ids)
        )
        if not aligned:
            mismatched.append(f"stream_alignment:{root}")
            per_root.append(
                RootParityResult(
                    root=root,
                    max_selected_logit_abs_diff=math.inf,
                    argmax_parity=False,
                    selected_is_argmax_parity=False,
                    compared_token_count=0,
                    aligned=False,
                )
            )
            continue
        diffs = [
            abs(float(a) - float(b))
            for a, b in zip(left.selected_logprobs, right.selected_logprobs, strict=True)
        ]
        if not diffs:
            # Aligned but empty: nothing was compared, so nothing is proven.
            mismatched.append(f"stream_empty:{root}")
        root_max = max(diffs) if diffs else 0.0
        argmax_parity = left.argmax_token_ids == right.argmax_token_ids
        rank_parity = cached_rank.get(root, ()) == uncached_rank.get(root, ())
        if not math.isfinite(root_max) or root_max > (
            CACHE_PARITY_MAX_SELECTED_LOGIT_ABS_DIFF
        ):
            mismatched.append(f"selected_logit:{root}")
        if not argmax_parity:
            mismatched.append(f"argmax:{root}")
        if not rank_parity:
            mismatched.append(f"selected_is_argmax:{root}")
        per_root.append(
            RootParityResult(
                root=root,
                max_selected_logit_abs_diff=root_max,
                argmax_parity=argmax_parity,
                selected_is_argmax_parity=rank_parity,
                compared_token_count=len(diffs),
                aligned=True,
            )
        )

    sign_parity = delta_signs_by_request(cached_rows) == delta_signs_by_request(
        uncached_rows
    )
    if not sign_parity:
        mismatched.append("delta_sign")
    overall = max(
        [0.0, *(result.max_selected_logit_abs_diff for result in per_root)]
    )
    return SecondaryParityResult(
        status=CACHE_ADMITTED if not mismatched else UNCACHED_FALLBACK,
        max_selected_logit_abs_diff=overall,
        mismatched_fields=tuple(dict.fromkeys(mismatched)),
        per_root=tuple(per_root),
        delta_sign_parity=sign_parity,
    )


# ---------------------------------------------------------------------------
# 7. Replay admission and quarantine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SecondaryQuarantineEntry:
    request_id: str
    gt_owner_id: str
    reason: str
    detail: str


@dataclass(frozen=True)
class SecondaryQuarantineLedger:
    """Append-only: an entry is never edited away once a request has failed."""

    entries: tuple[SecondaryQuarantineEntry, ...] = ()

    @property
    def count(self) -> int:
        return len(self.entries)


def replay_enforcement_for(backend_identity: Mapping[str, Any]) -> bool:
    """Whether a baseline replay mismatch withholds this shard's evidence.

    The backend declares whether it is evidence-bearing
    (``usable_as_evidence``); the deterministic ``FakeCensusBackend`` declares
    it is not, and its argmax is a hash of the token sequence rather than a
    model's, so enforcing native replay against it would quarantine every
    contract exercise while proving nothing.  Enforcement is therefore keyed to
    that declaration and sealed in the receipt, never silently skipped.
    """

    return bool(backend_identity.get("usable_as_evidence"))


def apply_secondary_quarantine(
    *,
    admitted: bool,
    request_id: str,
    gt_owner_id: str,
    reason: str,
    detail: str,
    ledger: SecondaryQuarantineLedger,
) -> SecondaryQuarantineLedger:
    """Quarantine on mismatch; otherwise return the ledger unchanged."""

    if admitted:
        return ledger
    return SecondaryQuarantineLedger(
        entries=(
            *ledger.entries,
            SecondaryQuarantineEntry(
                request_id=str(request_id),
                gt_owner_id=str(gt_owner_id),
                reason=str(reason),
                detail=str(detail),
            ),
        )
    )


# ---------------------------------------------------------------------------
# 8. Runtime capture
# ---------------------------------------------------------------------------


@dataclass
class SecondaryCaptureState:
    """Mutable bookkeeping shared by every request of one shard."""

    context_group_ids: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    quarantine: SecondaryQuarantineLedger = field(
        default_factory=SecondaryQuarantineLedger
    )
    root_backend: dict[str, str] = field(
        default_factory=lambda: {root: KV_CACHE_BACKEND for root in PAIRED_ROOTS}
    )


def _root_group_id(
    *, gt_owner_id: str, context_id: str, variant: str, root: str, appended_digest: str,
    scoring_backend: str,
) -> str:
    return primary._context_group_id(  # noqa: SLF001
        gt_owner_id=gt_owner_id,
        context_id=context_id,
        variant=f"{variant}|{root}",
        appended_digest=appended_digest,
        scoring_backend=scoring_backend,
    )


def _score_one_root(
    backend: Any,
    *,
    root_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
    uncached: bool,
    label: str,
) -> list[primary.ScoredToken]:
    """Teacher-force the target span from one freshly prefilled root."""

    prefill = backend.prefill(list(root_token_ids))
    try:
        prefill.assert_rooted(label=label)
        scored = primary._teacher_forced(  # noqa: SLF001
            backend,
            prefill,
            root_token_ids=list(root_token_ids),
            token_ids=list(target_token_ids),
            uncached=uncached,
        )
        prefill.assert_rooted(label=f"{label} (post)")
    finally:
        prefill.close()
    return scored


def _root_payload(
    *,
    root: str,
    context_id: str,
    executed_prefix_token_ids: Sequence[int],
    appended_token_ids: Sequence[int],
    context_group_id: str,
    scoring_backend: str,
    scored: Sequence[primary.ScoredToken],
    segments: RowSegments,
    target_token_ids: Sequence[int],
) -> dict[str, Any]:
    identity = primary.build_request_identity(
        context_id=context_id,
        prefix_token_ids=list(executed_prefix_token_ids),
        appended_token_ids=list(appended_token_ids),
    )
    selected = [token.selected_logprob for token in scored]
    output = primary.build_output_identity(
        request_identity_sha256=identity["request_identity_sha256"],
        selected_logits=selected,
        token_ids=[token.token_id for token in scored],
    )
    sums = segment_sums(segments, selected, label=f"{root} readout")
    argmax_ids = [token.argmax_token_id for token in scored]
    expected = [int(value) for value in target_token_ids]
    description_stop = segments.description[1]
    return {
        "root": root,
        "context_id": str(context_id),
        "root_token_count": len(executed_prefix_token_ids) + len(appended_token_ids),
        "executed_prefix_token_count": len(executed_prefix_token_ids),
        "executed_prefix_token_ids_sha256": sha256_json(
            [int(value) for value in executed_prefix_token_ids]
        ),
        "appended_token_count": len(appended_token_ids),
        "appended_token_ids_sha256": sha256_json(
            [int(value) for value in appended_token_ids]
        ),
        "context_group_id": str(context_group_id),
        "scoring_backend": str(scoring_backend),
        "request_identity_sha256": identity["request_identity_sha256"],
        "output_identity_sha256": output["output_identity_sha256"],
        "selected_logprobs": selected,
        # Diagnostics: the readout channel is the selected-token log probability;
        # argmax ids and their log probabilities travel beside it and never
        # replace it.
        "argmax_token_ids": argmax_ids,
        "argmax_logprobs": [token.argmax_logprob for token in scored],
        "selected_is_argmax": [
            int(token) == int(argmax)
            for token, argmax in zip(expected, argmax_ids, strict=True)
        ],
        "argmax_reproduces_description_path": primary.replay_argmax_through_prefix(
            expected_token_ids=expected,
            argmax_token_ids=argmax_ids,
            up_to_index=description_stop,
        ),
        "argmax_reproduces_complete_row": expected == argmax_ids,
        "segment_sums": sums,
    }


def _capture_request(
    plan: primary.SealedPlan,
    backend: Any,
    state: SecondaryCaptureState,
    *,
    shard_id: str,
    session_image_id: str,
    request: Mapping[str, Any],
    registry_row: Mapping[str, Any],
    binding: Mapping[str, Any],
    analysis: SealedPrimaryAnalysis,
    enforce_replay: bool,
) -> dict[str, Any]:
    """Score one sealed secondary request on both of its paired roots."""

    variant = assert_secondary_request(request)
    request_id = str(request["request_id"])
    gt_owner_id = str(request["gt_owner_id"])
    image_id = str(request["image_id"])
    if image_id != str(session_image_id):
        _fail(
            f"request {request_id!r} belongs to image {image_id!r} but the open session holds "
            f"image {session_image_id!r}"
        )
    roots = resolve_paired_roots(plan, request, registry_row)
    inserted_digest = assert_inserted_row_matches_registry(request, registry_row)
    for context_id in (roots.modified_context_id, roots.baseline_context_id):
        plan.assert_context_belongs_to_session_image(
            context_id,
            session_image_id=session_image_id,
            label=f"secondary request {request_id!r}",
        )

    target_tokens = _token_ids(
        request["scored_target"]["token_ids"], label=f"request {request_id!r} target"
    )
    segments = row_segments(target_tokens)
    appended = list(binding["appended_token_ids"])
    if sha256_json(appended) != inserted_digest:
        _fail(f"request {request_id!r} appended tokens do not reconstruct their sealed digest")

    baseline_prefix = plan.executed_prefix_token_ids(roots.baseline_context_id)
    modified_prefix = plan.executed_prefix_token_ids(roots.modified_context_id)
    if binding["executed_prefix_token_ids"] != modified_prefix:
        _fail(
            f"request {request_id!r} modified root does not reconstruct the sealed executed "
            "prefix"
        )
    root_prefix: dict[str, list[int]] = {
        ROOT_BASELINE: list(baseline_prefix),
        ROOT_MODIFIED: list(modified_prefix),
    }
    root_appended: dict[str, list[int]] = {ROOT_BASELINE: [], ROOT_MODIFIED: appended}
    root_tokens = {
        root: root_prefix[root] + root_appended[root] for root in PAIRED_ROOTS
    }
    root_context = {
        ROOT_BASELINE: roots.baseline_context_id,
        ROOT_MODIFIED: roots.modified_context_id,
    }

    payloads: dict[str, dict[str, Any]] = {}
    forced: dict[str, list[int]] = {}
    for root in PAIRED_ROOTS:
        scoring_backend = state.root_backend[root]
        group_id = _root_group_id(
            gt_owner_id=gt_owner_id,
            context_id=root_context[root],
            variant=variant,
            root=root,
            appended_digest=sha256_json(root_appended[root]),
            scoring_backend=scoring_backend,
        )
        state.context_group_ids.append(group_id)
        scored = _score_one_root(
            backend,
            root_token_ids=root_tokens[root],
            target_token_ids=target_tokens,
            uncached=scoring_backend == UNCACHED_BACKEND,
            label=f"{root} of {request_id}",
        )
        forced[root] = [token.token_id for token in scored]
        payloads[root] = _root_payload(
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

    target_digest = assert_paired_token_identity(
        forced[ROOT_BASELINE],
        forced[ROOT_MODIFIED],
        declared_sha256=str(request["scored_target"]["token_ids_sha256"]),
        label=f"request {request_id!r}",
    )
    deltas = paired_deltas(
        payloads[ROOT_BASELINE]["segment_sums"], payloads[ROOT_MODIFIED]["segment_sums"]
    )

    replay_admitted = bool(payloads[ROOT_BASELINE]["argmax_reproduces_description_path"])
    if enforce_replay:
        state.quarantine = apply_secondary_quarantine(
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
        "row_kind": "secondary_compatibility_row",
        "unit_id": UNIT_ID,
        "shard_id": str(shard_id),
        "request_id": request_id,
        "request_key": str(request["request_key"]),
        "request_family": REQUEST_FAMILY,
        "readout_tier": SECONDARY_READOUT_TIER,
        "cohort": str(request["cohort"]),
        "variant": variant,
        "plan_optional": bool(request["optional"]),
        "gt_owner_id": gt_owner_id,
        "image_id": image_id,
        "session_image_id": str(session_image_id),
        "plan_identity_digest": str(binding["identity_digest"]),
        "primary_analysis_binding_sha256": analysis.binding_sha256,
        "native_row_index": roots.native_row_index,
        "scored_target_kind": str(request["scored_target"]["kind"]),
        "baseline_context_id": roots.baseline_context_id,
        "modified_context_id": roots.modified_context_id,
        "successor_context_id": roots.successor_context_id,
        "baseline_context_source": roots.registry_source,
        "inserted_clean_row_c_token_ids_sha256": inserted_digest,
        "inserted_clean_row_c_token_count": len(appended),
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
        "baseline_replay_admitted": replay_admitted,
        "replay_admission_enforced": bool(enforce_replay),
        "likelihood_channel": LIKELIHOOD_CHANNEL,
        "repetition_penalty_stratum": float(NATIVE_REPETITION_PENALTY_STRATUM),
        "uses_model_generate": False,
        "sampling": "disabled_secondary_deterministic_teacher_forcing_only",
        "retokenized": False,
        "delta_orientation": "modified_minus_baseline",
        "claim_boundary": CLAIM_BOUNDARY,
    }


# ---------------------------------------------------------------------------
# 9. Shard assembly and publication
# ---------------------------------------------------------------------------


@dataclass
class SecondaryShardResult:
    receipt: dict[str, Any]
    rows: list[dict[str, Any]] = field(default_factory=list)
    parity: dict[str, Any] = field(default_factory=dict)
    quarantine: dict[str, Any] | None = None
    admission: dict[str, Any] | None = None


def shard_output_files(result: SecondaryShardResult) -> dict[str, bytes]:
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
        if source is None:
            _fail(f"module {name!r} exposes no source file to hash")
        digests[name] = sha256_file(Path(source))
    return dict(sorted(digests.items()))


def _base_receipt(
    plan: primary.SealedPlan,
    analysis: SealedPrimaryAnalysis,
    *,
    shard_id: str,
    mode: str,
    runtime_identity: Mapping[str, Any],
    plan_counts: Mapping[str, Any],
    image_counts: Mapping[str, Any],
    executed: Mapping[str, Any],
    quarantine: SecondaryQuarantineLedger,
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
            "manifest_content_sha256": plan.manifest.get("manifest_content_sha256"),
            "builder_source_sha256": primary.builder_source_sha256(plan.manifest),
            "plan_file_sha256": dict(sorted(plan.plan_file_sha256.items())),
            "lineage": plan.manifest.get("lineage"),
            "cohort_counts": plan.manifest.get("cohort_counts"),
            "control_counts": plan.manifest.get("control_counts"),
        },
        "primary_analysis": dict(analysis.binding),
        "primary_analysis_binding_sha256": analysis.binding_sha256,
        "runtime_identity": dict(runtime_identity),
        "runtime_identity_sha256": primary.runtime_identity_digest(runtime_identity),
        "secondary_request_counts": {
            "plan": dict(plan_counts),
            "image": dict(image_counts),
        },
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
            "sampling": "not_implemented_secondary_deterministic_teacher_forcing_only",
            "retokenizes": False,
            "likelihood_channel": LIKELIHOOD_CHANNEL,
            "repetition_penalty_stratum": float(NATIVE_REPETITION_PENALTY_STRATUM),
            "primary_branches_sealed_before_this_pass": True,
            "primary_branch_assignment_performed": False,
            "branch_labels_used_to_select_requests": False,
            "executes_every_sealed_secondary_request_of_the_image": True,
            "one_image_session_per_shard": True,
            "fresh_cache_per_logical_root": True,
            "paired_roots_force_identical_target_tokens": True,
            "delta_orientation": "modified_minus_baseline",
            "replay_admission_enforced": bool(enforce_replay),
            "backend_kind": str(backend_kind),
            "requested_batch_size": int(batch_size),
            "effective_batch_size": 1,
            "batching_applicable": False,
            "batching_note": BATCH_POLICY_NOT_APPLICABLE,
            "claim_boundary": CLAIM_BOUNDARY,
            "final_set_retention_or_free_rollout_claimed": False,
        },
        # No wall-clock field is sealed: the published artifact set must be
        # byte-identical across re-runs so an idempotent re-capture publishes as
        # a no-op instead of colliding with itself.
        "artifact_determinism": "no_wall_clock_or_host_specific_field_is_sealed",
    }


def _seal_output_digests(receipt: dict[str, Any], files: Mapping[str, bytes]) -> None:
    receipt["output_file_digests"] = {
        name: {
            "path": name,
            "byte_size": len(payload),
            "sha256": sha256_bytes(payload),
        }
        for name, payload in sorted(files.items())
        if name != RECEIPT_NAME
    }
    receipt["receipt_content_sha256"] = sha256_json(receipt)


def _quarantine_payload(
    *, shard_id: str, session_image_id: str, ledger: SecondaryQuarantineLedger
) -> dict[str, Any]:
    return {
        "schema_version": QUARANTINE_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "shard_id": str(shard_id),
        "session_image_id": str(session_image_id),
        "reason": "one_or_more_secondary_requests_failed_their_admission_control",
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
            "repair the runtime or plan alignment rather than publishing partial secondary "
            "evidence"
        ),
    }


def _registry_by_owner(plan: primary.SealedPlan) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for row in (*plan.cohort_rows, *plan.control_rows):
        registry[str(row["gt_owner_id"])] = dict(row)
    return registry


def _prepare_shard(
    plan: primary.SealedPlan, *, image_id: str
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the whole sealed secondary census, then narrow to one image."""

    all_rows = secondary_request_rows(plan)
    plan_counts = validate_secondary_plan_counts(all_rows, manifest=plan.manifest)
    image_rows = secondary_requests_for_image(all_rows, image_id=image_id)
    # Expected is the sealed plan's own per-image census, read score-blind
    # before anything executes; the shard fills ``observed_*`` from what it
    # actually produced and the two are reconciled before publication.
    expected = {
        "image_id": str(image_id),
        "expected_by_variant": counts_by_variant(image_rows),
        "expected_total": len(image_rows),
    }
    bindings = {
        str(row["request_id"]): primary.validate_request_row(plan, row)
        for row in image_rows
    }
    return image_rows, plan_counts, expected, bindings


def reconcile_image_counts(
    expected: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Fold what a shard produced back onto what the sealed plan required."""

    observed = counts_by_variant(rows)
    if observed != expected["expected_by_variant"] or len(rows) != int(
        expected["expected_total"]
    ):
        _fail(
            f"image {expected['image_id']!r} produced {len(rows)} secondary rows {observed!r}, "
            f"not the sealed {expected['expected_total']} "
            f"{expected['expected_by_variant']!r}"
        )
    return {
        **dict(expected),
        "observed_by_variant": observed,
        "observed_total": len(rows),
    }


def _capture_rows(
    plan: primary.SealedPlan,
    backend: Any,
    analysis: SealedPrimaryAnalysis,
    state: SecondaryCaptureState,
    *,
    shard_id: str,
    session_image_id: str,
    rows: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Mapping[str, Any]],
    enforce_replay: bool,
) -> list[dict[str, Any]]:
    registry = _registry_by_owner(plan)
    captured: list[dict[str, Any]] = []
    for request in rows:
        owner_id = str(request["gt_owner_id"])
        registry_row = registry.get(owner_id)
        if registry_row is None:
            _fail(
                f"owner {owner_id!r} is in neither the cohort nor the control registry; a "
                "secondary request cannot be attributed"
            )
        if str(registry_row.get("cohort")) != str(request.get("cohort")):
            _fail(
                f"owner {owner_id!r} is registered in cohort {registry_row.get('cohort')!r} but "
                f"its secondary request declares {request.get('cohort')!r}"
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
                binding=bindings[str(request["request_id"])],
                analysis=analysis,
                enforce_replay=enforce_replay,
            )
        )
    return captured


def _parity_payload(
    *,
    shard_id: str,
    session_image_id: str,
    backend_kind: str,
    root_backend: Mapping[str, str],
    parity: SecondaryParityResult | None,
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
        "max_selected_logit_abs_diff_threshold": (
            CACHE_PARITY_MAX_SELECTED_LOGIT_ABS_DIFF
        ),
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
                        "max_selected_logit_abs_diff": (
                            result.max_selected_logit_abs_diff
                        ),
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
    plan: primary.SealedPlan,
    analysis: SealedPrimaryAnalysis,
    backend: Any,
    shard_id: str,
    image_id: str,
    runtime_identity: Mapping[str, Any],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> SecondaryShardResult:
    """Prove cached execution on one image and seal an admission receipt.

    Every sealed secondary request of the image is scored twice -- once entirely
    uncached, once entirely on the KV cache -- and the two executions must agree
    on every selected logit within ``1e-3`` and on every argmax, rank flag and
    segment delta sign exactly.  All three sealed variants are present on every
    image of this plan, so one image session carries the whole role matrix.
    """

    rows, plan_counts, expected_counts, bindings = _prepare_shard(plan, image_id=image_id)
    missing = sorted(
        variant
        for variant, count in counts_by_variant(rows).items()
        if count == 0
    )
    if missing:
        _fail(
            f"image {image_id!r} carries no sealed secondary request for variant(s) "
            f"{missing!r}; the smoke matrix must run inside one image session"
        )
    enforce_replay = replay_enforcement_for(backend.identity)
    backend_kind = str(backend.identity.get("backend"))

    sides: dict[str, list[dict[str, Any]]] = {}
    state = SecondaryCaptureState()
    for label, root_backend in (
        ("uncached", {root: UNCACHED_BACKEND for root in PAIRED_ROOTS}),
        ("cached", {root: KV_CACHE_BACKEND for root in PAIRED_ROOTS}),
    ):
        side_state = SecondaryCaptureState(root_backend=dict(root_backend))
        sides[label] = _capture_rows(
            plan,
            backend,
            analysis,
            side_state,
            shard_id=shard_id,
            session_image_id=image_id,
            rows=rows,
            bindings=bindings,
            enforce_replay=enforce_replay,
        )
        state.context_group_ids.extend(side_state.context_group_ids)
        state.quarantine = SecondaryQuarantineLedger(
            entries=(*state.quarantine.entries, *side_state.quarantine.entries)
        )
    # The cached and uncached sides are two separate executions on their own
    # fresh state, never one cache read twice; this proves it.
    primary.assert_fresh_context_per_owner(state.context_group_ids)

    image_counts = reconcile_image_counts(expected_counts, sides["cached"])
    reconcile_image_counts(expected_counts, sides["uncached"])
    parity = evaluate_secondary_parity(
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
        "plan_manifest_content_sha256": plan.manifest.get("manifest_content_sha256"),
        "primary_analysis_binding_sha256": analysis.binding_sha256,
        "runtime_identity_sha256": primary.runtime_identity_digest(runtime_identity),
        "admission_identity_fields": list(primary.ADMISSION_IDENTITY_FIELDS),
        **primary.admission_identity_payload(runtime_identity),
        "smoke_variants": dict(counts_by_variant(rows)),
        "cache_admitted": cache_admitted,
        "root_backend": dict(sorted(root_backend.items())),
        "max_selected_logit_abs_diff": parity.max_selected_logit_abs_diff,
        "mismatched_fields": list(parity.mismatched_fields),
        "replay_admission_enforced": bool(enforce_replay),
        "scope": (
            "one image session carried every sealed secondary variant; no context of another "
            "image was forwarded through it"
        ),
    }
    admission["admission_content_sha256"] = sha256_json(admission)

    receipt = _base_receipt(
        plan,
        analysis,
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
            "request_ids_sha256": sha256_json(sorted(bindings)),
        },
        quarantine=state.quarantine,
        enforce_replay=enforce_replay,
        backend_kind=backend_kind,
        batch_size=batch_size,
    )
    receipt["admission"] = admission

    if state.quarantine.count:
        quarantine = _quarantine_payload(
            shard_id=shard_id, session_image_id=image_id, ledger=state.quarantine
        )
        result = SecondaryShardResult(receipt=receipt, quarantine=quarantine)
        _seal_output_digests(receipt, shard_output_files(result))
        return result
    result = SecondaryShardResult(
        receipt=receipt, parity=parity_payload, admission=admission
    )
    _seal_output_digests(receipt, shard_output_files(result))
    return result


def validate_secondary_admission_receipt(
    admission: Mapping[str, Any],
    *,
    plan: primary.SealedPlan,
    analysis: SealedPrimaryAnalysis,
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """An admission is inherited only under the exact plan, gate and runtime it proved."""

    if str(admission.get("schema_version")) != ADMISSION_SCHEMA_VERSION:
        _fail(
            f"secondary admission schema {admission.get('schema_version')!r} is not "
            f"{ADMISSION_SCHEMA_VERSION!r}"
        )
    if str(admission.get("unit_id")) != UNIT_ID:
        _fail("secondary admission receipt belongs to another unit")
    _assert_self_sealed(
        admission,
        seal_key="admission_content_sha256",
        label="secondary admission receipt",
    )
    if admission.get("plan_manifest_content_sha256") != plan.manifest.get(
        "manifest_content_sha256"
    ):
        _fail(
            "the secondary admission was sealed against a different plan manifest; a smoke "
            "cannot admit a capture of another plan"
        )
    if str(admission.get("primary_analysis_binding_sha256")) != analysis.binding_sha256:
        _fail(
            "the secondary admission was sealed against a different primary analysis; the "
            "branch gate this capture inherits is not the one the smoke proved"
        )
    declared_digest = primary.runtime_identity_digest(admission)
    if declared_digest != str(admission.get("runtime_identity_sha256")):
        _fail(
            "the secondary admission's runtime_identity_sha256 does not reconstruct from its "
            "own declared identity fields; the receipt is internally inconsistent"
        )
    if declared_digest != primary.runtime_identity_digest(runtime_identity):
        differing = sorted(
            name
            for name in primary.ADMISSION_IDENTITY_FIELDS
            if sha256_json(admission.get(name)) != sha256_json(runtime_identity.get(name))
        )
        _fail(
            "the secondary admission was sealed under a different runtime identity "
            f"(differing field(s): {differing or ['<unreported>']!r}); cached execution is "
            "never inherited across a changed model, tokenizer, numerical runtime or scorer"
        )
    root_backend = admission.get("root_backend") or {}
    for root in PAIRED_ROOTS:
        if root not in root_backend:
            _fail(f"secondary admission declares no backend for root {root!r}")
    if not bool(admission.get("cache_admitted")) and any(
        str(value) != UNCACHED_BACKEND for value in root_backend.values()
    ):
        _fail("secondary admission is internally inconsistent about its root backends")
    return dict(admission)


def run_capture_shard(
    *,
    plan: primary.SealedPlan,
    analysis: SealedPrimaryAnalysis,
    backend: Any,
    shard_id: str,
    session_image_id: str,
    admission: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> SecondaryShardResult:
    """Execute every sealed secondary request of one image under an admission."""

    rows, plan_counts, expected_counts, bindings = _prepare_shard(
        plan, image_id=session_image_id
    )
    enforce_replay = replay_enforcement_for(backend.identity)
    backend_kind = str(backend.identity.get("backend"))
    state = SecondaryCaptureState(
        root_backend={
            root: str(admission["root_backend"][root]) for root in PAIRED_ROOTS
        }
    )
    state.rows = _capture_rows(
        plan,
        backend,
        analysis,
        state,
        shard_id=shard_id,
        session_image_id=session_image_id,
        rows=rows,
        bindings=bindings,
        enforce_replay=enforce_replay,
    )
    primary.assert_fresh_context_per_owner(state.context_group_ids)
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
            "primary_analysis_binding_sha256": admission.get(
                "primary_analysis_binding_sha256"
            ),
        },
        compared_request_ids=[],
    )

    receipt = _base_receipt(
        plan,
        analysis,
        shard_id=shard_id,
        mode=MODE_CAPTURE,
        runtime_identity=runtime_identity,
        plan_counts=plan_counts,
        image_counts=image_counts,
        executed={
            "session_image_id": str(session_image_id),
            "row_count": len(state.rows),
            "gt_owner_ids": sorted({str(row["gt_owner_id"]) for row in state.rows}),
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
        "root_backend": dict(sorted(state.root_backend.items())),
    }

    if state.quarantine.count:
        quarantine = _quarantine_payload(
            shard_id=shard_id,
            session_image_id=session_image_id,
            ledger=state.quarantine,
        )
        result = SecondaryShardResult(receipt=receipt, quarantine=quarantine)
        _seal_output_digests(receipt, shard_output_files(result))
        return result
    result = SecondaryShardResult(
        receipt=receipt, rows=state.rows, parity=parity_payload
    )
    _seal_output_digests(receipt, shard_output_files(result))
    return result


# ---------------------------------------------------------------------------
# 10. CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True, type=Path, help="sealed CPU plan directory")
    parser.add_argument(
        "--primary-analysis-dir",
        required=True,
        type=Path,
        help="sealed primary analysis directory whose branches gate this pass",
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
            "smoke: prove cached-versus-uncached parity on one image and seal an admission "
            "receipt. capture: score one image's sealed secondary requests under it."
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
            "accepted for parity with the primary scorer and reported in the receipt; this "
            "path teacher-forces one lane per root and never batches"
        ),
    )
    parser.add_argument("--backend", choices=("hf", "fake"), default="hf")
    parser.add_argument(
        "--validate-plan-only",
        action="store_true",
        help=(
            "re-prove every plan digest and the sealed primary branch gate, then exit without "
            "loading a model"
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.time()
    plan = primary.load_sealed_plan(
        args.plan_dir,
        prevalence_run_root=args.prevalence_run_root,
        census_run_root=args.census_run_root,
    )
    # The branch gate is a *precondition*, so it is proven on the CPU path
    # before any model, session or GPU state exists.
    analysis = load_sealed_primary_analysis(args.primary_analysis_dir, plan=plan)
    all_rows = secondary_request_rows(plan)
    plan_counts = validate_secondary_plan_counts(all_rows, manifest=plan.manifest)

    if args.validate_plan_only:
        return {
            "status": "plan_and_branch_gate_validated",
            "plan_dir": str(plan.plan_dir),
            "manifest_content_sha256": plan.manifest.get("manifest_content_sha256"),
            "primary_analysis_dir": str(analysis.analysis_dir),
            "primary_analysis_binding_sha256": analysis.binding_sha256,
            "primary_analysis_decision": analysis.decision,
            "primary_branch_counts": dict(analysis.branch_counts),
            "secondary_request_counts": dict(plan_counts),
            "loads_model": False,
        }

    if args.backend == "hf" and args.runtime_identity is None:
        _fail("--runtime-identity is required for --backend hf")
    if int(args.batch_size) < 1:
        _fail("--batch-size must be at least one")

    image_id = str(args.image_id)
    if args.mode == MODE_CAPTURE and args.admission_receipt is None:
        _fail(
            f"--mode {MODE_CAPTURE} requires --admission-receipt pointing at the smoke run's "
            f"{ADMISSION_NAME}; cached execution is never assumed"
        )
    # Fails closed with the covered images named before a model is opened.
    secondary_requests_for_image(all_rows, image_id=image_id)

    if args.backend == "hf":
        primary._basin().pin_fp32_parity_flags()  # noqa: SLF001

    with primary._open_backend(args, plan, image_id) as backend:  # noqa: SLF001
        numerics = primary.build_runtime_numerics(backend, infer_config=args.infer_config)
        runtime_identity = {
            **primary.validate_runtime_identity(
                backend.identity, args.runtime_identity, numerics=numerics
            ),
            "numerics": numerics,
            "source_identity": source_identity(),
            "session_image_id": image_id,
        }
        if args.mode == MODE_SMOKE:
            result = run_smoke_shard(
                plan=plan,
                analysis=analysis,
                backend=backend,
                shard_id=str(args.shard_id),
                image_id=image_id,
                runtime_identity=runtime_identity,
                batch_size=int(args.batch_size),
            )
        else:
            admission = validate_secondary_admission_receipt(
                _read_json(Path(args.admission_receipt), "secondary admission receipt"),
                plan=plan,
                analysis=analysis,
                runtime_identity=runtime_identity,
            )
            result = run_capture_shard(
                plan=plan,
                analysis=analysis,
                backend=backend,
                shard_id=str(args.shard_id),
                session_image_id=image_id,
                admission=admission,
                runtime_identity=runtime_identity,
                batch_size=int(args.batch_size),
            )

    publish = primary._publish(Path(args.output_dir), shard_output_files(result))  # noqa: SLF001
    status = (
        "quarantined"
        if result.quarantine is not None
        else ("admitted" if result.admission is not None else "captured")
    )
    return {
        "status": status,
        "mode": str(args.mode),
        "shard_id": str(args.shard_id),
        "image_id": image_id,
        "row_count": len(result.rows),
        "primary_analysis_binding_sha256": analysis.binding_sha256,
        "receipt_content_sha256": result.receipt.get("receipt_content_sha256"),
        "admission_content_sha256": (
            None if result.admission is None else result.admission["admission_content_sha256"]
        ),
        # Reported, never sealed: a wall-clock duration would make otherwise
        # identical re-captures publish differing bytes.
        "elapsed_seconds": round(time.time() - started_at, 3),
        "publish": publish,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except (
        primary.CrossingBoundaryContractError,
        plan_builder.PlanContractError,
    ) as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
