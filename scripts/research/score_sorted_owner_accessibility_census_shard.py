#!/usr/bin/env python3
"""Shard scorer/executor for the sorted owner accessibility phenotype census
(``2026-08-03-sorted-owner-accessibility-phenotype-census``).

Consumes only the immutable CPU plan sealed by
``build_sorted_owner_accessibility_census_plan.py``.  It reads **no score
artifact of any kind**, so every pre-P0 score is structurally unreachable
rather than merely unused (contract item 5).

Execution contract
------------------
*Scoring unit.*  One *singleton query group*: exactly one
``(image_id, context_id, normalized_description, observed_prefix_sha256,
query_prefix_sha256)`` tuple.  A fresh KV cache is built per group and released
before the next, so no group ever inherits another group's cache state.

*Admission.*  Admission identity is the **exact prefix digest** for a channel,
never a shared shape class:

* the ``query_suffix`` (box) channel is admitted on the group's own exact query
  prefix, using the production :func:`run_cache_parity_gate` at all four
  coordinate depths against an independent uncached reforward;
* the ``proposal_boundary_gate`` channel reads the *observed prefix with
  nothing forced* -- a different execution shape -- and is admitted on that
  exact observed prefix, at root depth only;
* the ``proposal_category_route`` channel is admitted per
  ``(context, category)`` on the exact executed routing path, because every
  category's forced path has its own token content *and* its own length.  A
  first-category probe never authorizes another category's path.

``suffix_shape_class`` is still recorded and coverage-checked, so shape
coverage stays auditable even though it is not the admission key.

The root context's observed prefix *is* the executed prompt, so it prefills
through :func:`prefill_prompt_only` -- the same native inputs, explicit
positions, fresh ``DynamicCache`` and ``HFCacheBackend`` as every other
context, minus a generated history that does not exist.

*Runtime invariants (Qwen3-VL).*  Every model call passes explicit non-``None``
``position_ids``; the model carries a shared mutable ``rope_deltas`` that a
positionless call would corrupt.  This module therefore never calls
``model.generate()`` and never runs concurrent groups on one model.  Free
decoding is a custom explicit-position greedy loop over the same admitted cache
seam.

*Phase order.*  All decision-bearing likelihood scoring (box + proposal)
completes and every cache backend is released **before** the terminal
behavior phase that produces free-decode sidecars.  No likelihood score is
taken after that phase opens; :class:`PhaseGuard` enforces it.

Backends
--------
``--backend hf`` wires the real production stack: ``open_backend_session`` over
the resolved infer config, ``HFBackendSession._materialize_native_inputs`` for
the true ``pixel_values``/``image_grid_thw``, then
``score_sorted_owner_basin_landscape.prefill_context`` /
``HFCacheBackend`` / ``BranchCursor`` for prefill, branching and continuation.
``--backend fake`` drives the *identical* code path -- including the real
production parity gate -- with a deterministic in-process stub, so every
contract assertion is unit-testable without a GPU.

``--validate-contract-only`` runs the full CPU contract check over every planned
work item in a shard and loads no model at all.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import contextlib
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Protocol
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_owner_accessibility_census_plan as planner  # noqa: E402

UNIT_ID = planner.UNIT_ID
PLAN_SCHEMA_VERSION = planner.PLAN_SCHEMA_VERSION

#: Every row this module emits declares this contract token.  A row that does
#: not carry it -- in particular any pre-P0 score row, which predates exact
#: prefix-bound admission -- is refused by :func:`assert_row_bindings` and can
#: never be joined into this unit's analysis.
P0_ROW_CONTRACT = "p0-exact-prefix-admission-bound"

SCORE_SCHEMA_VERSION = "sorted-owner-accessibility-census-score.v1"
X1_SCHEMA_VERSION = "sorted-owner-accessibility-census-x1.v1"
PROPOSAL_SCHEMA_VERSION = "sorted-owner-accessibility-census-proposal.v1"
FREE_DECODE_SCHEMA_VERSION = "sorted-owner-accessibility-census-free-decode.v1"
ADMISSION_SCHEMA_VERSION = "sorted-owner-accessibility-census-admission.v1"
SHARD_RECEIPT_SCHEMA_VERSION = "sorted-owner-accessibility-census-shard-receipt.v1"
QUARANTINE_SCHEMA_VERSION = "sorted-owner-accessibility-census-shard-quarantine.v1"
CONTRACT_CHECK_SCHEMA_VERSION = "sorted-owner-accessibility-census-contract-check.v1"

SCORES_NAME = "census-scores.jsonl"
X1_NAME = "x1-distributions.jsonl"
PROPOSAL_NAME = "proposal-surface.jsonl"
FREE_DECODE_NAME = "free-decode-sidecars.jsonl"
RECEIPT_NAME = "shard-receipt.json"
QUARANTINE_NAME = "shard-quarantine.json"

#: Files this module writes as *primary evidence*.  A quarantined shard must
#: leave none of them behind.
PRIMARY_OUTPUT_NAMES: tuple[str, ...] = (SCORES_NAME, X1_NAME, PROPOSAL_NAME, FREE_DECODE_NAME)

OBJECT_REF_START = planner.OBJECT_REF_START
OBJECT_REF_END = planner.OBJECT_REF_END
BOX_START = planner.BOX_START
BOX_END = planner.BOX_END
IM_END = planner.IM_END
COORD_TOKEN_START = planner.COORD_TOKEN_START
COORD_TOKEN_END = planner.COORD_TOKEN_END
COORD_BIN_COUNT = planner.COORD_BIN_COUNT

CHANNEL_QUERY_SUFFIX = planner.CHANNEL_QUERY_SUFFIX
CHANNEL_PROPOSAL_BOUNDARY_GATE = planner.CHANNEL_PROPOSAL_BOUNDARY_GATE
CHANNEL_PROPOSAL_CATEGORY_ROUTE = planner.CHANNEL_PROPOSAL_CATEGORY_ROUTE

COORD_SLOTS: tuple[str, ...] = ("x1", "y1", "x2", "y2")

DEFAULT_TOP_K = 5

#: Number of independent candidates scored per batched forward inside one exact
#: query group.  ``1`` is the frozen default and reproduces the single-branch
#: bytes exactly.  Values above one are a throughput knob only: they never
#: change which candidates are scored, their order, or the schema, and they are
#: always receipted.
DEFAULT_CANDIDATE_BATCH_SIZE = 1

#: Frozen free-decode caps, sealed into the shard receipt.  Both loops are
#: greedy, explicit-position, and terminal: they are behavior records, never
#: probabilities, and never enter a rank.
FREE_BOX_MAX_TOKENS = 5
FREE_ROW_MAX_TOKENS = 48

#: Explicitly requested relaxed cache-admission tolerance on the *selected*
#: coordinate log probability.  This preserves the numerical policy already
#: accepted for this model/attention stack: strict full-vocabulary parity is
#: still computed and recorded, and coordinate-domain argmax parity is still
#: required, but a strict-only failure inside this bound does not force the
#: uncached fallback.
RELAXED_SELECTED_LOGPROB_MAX_ABS_DIFF = 1e-3

#: Owner-bank accounting keys the shard receipt requires from the plan.  Named
#: explicitly so a plan that drops one fails closed instead of silently
#: emitting an unqualified owner.
REQUIRED_OWNER_BANK_KEYS: tuple[str, ...] = (
    "logical_role_count",
    "distinct_physical_candidates_reached",
    "physical_candidate_ids",
    "generator_local_bank_adequacy",
    "strict_assignment_coverage",
    "bank_coverage_status",
    "disposition_eligible",
    "undercovered",
    "disposition_floor",
)

#: The strict-assignment view is a separate lower bound, never the adequacy
#: gate; its counts are still required so an owner's evidence is qualified.
REQUIRED_OWNER_STRICT_ASSIGNMENT_KEYS: tuple[str, ...] = (
    "uniquely_assigned_candidate_count",
    "roles_lost_to_other_owner_assignment",
    "roles_lost_to_ambiguous_assignment",
    "roles_lost_to_unmatched_assignment",
)

#: An undercovered owner may never be given a persistent-negative disposition.
UNDERCOVERED_DISPOSITION_FLOOR = (
    "unresolved_only_never_persistent_no_tested_localization_support"
)

#: Binding fields every emitted row must carry.  These make a row joinable to
#: exactly one plan, one capture-rule object, one channel and one admission
#: receipt -- and make a pre-P0 row mechanically unjoinable.
REQUIRED_ROW_BINDING_FIELDS: tuple[str, ...] = (
    "row_contract",
    "unit_id",
    "plan_schema_version",
    "plan_receipt_content_sha256",
    "capture_rules_sha256",
    "channel",
    "admission_receipt_id",
    "observed_prefix_sha256",
    "query_prefix_sha256",
    "query_suffix_token_ids_sha256",
)

#: The census scores exactly one repetition-penalty stratum.  The inherited
#: human-refined12 config is rp1.10, which would stamp a foreign stratum onto
#: the session's runtime identity even though this unit reads raw logits, so
#: the default here is the rp1.0 override and :func:`assert_repetition_penalty_stratum`
#: fails closed on anything else.
DEFAULT_INFER_CONFIG = (
    REPO_ROOT
    / "configs/coordexp_infras/infer"
    / "qwen3_vl_2b_desc_first_geo_sorted_step4887_human_refined12_hf_fp32_rp1p0.yaml"
)

sha256_json = planner.sha256_json
canonical_json_bytes = planner.canonical_json_bytes

#: This module's own source bytes, hashed once at *import* time.  A live
#: end-of-job re-read would capture edits that landed while a long shard was
#: still running and misattribute the run's code identity.
EXECUTED_SOURCE_SHA256 = planner.sha256_file(Path(__file__).resolve())

#: Digest of the empty suffix: the proposal channel's ``query_suffix`` binding.
#: The field is never omitted, so a proposal row and a box row can never be
#: confused for one another by an absent key.
EMPTY_SUFFIX_SHA256 = sha256_json([])


class ShardContractError(RuntimeError):
    """Raised when a shard precondition or invariant fails; quarantines the image."""


def _basin():
    """Lazy handle on the production scoring seams (imports torch)."""

    from scripts.research import score_sorted_owner_basin_landscape as module

    return module


# ---------------------------------------------------------------------------
# Plan loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanBundle:
    plan_dir: Path
    receipt: Mapping[str, Any]
    capture_rules: Mapping[str, Any]
    images: dict[str, Mapping[str, Any]]
    owners: dict[str, Mapping[str, Any]]
    categories: dict[str, Mapping[str, Any]]
    contexts: dict[str, Mapping[str, Any]]
    candidates: dict[str, Mapping[str, Any]]
    query_groups: dict[str, Mapping[str, Any]]
    native_sidecars: list[Mapping[str, Any]]
    shards: dict[str, Mapping[str, Any]]

    @property
    def receipt_content_sha256(self) -> str:
        return str(self.receipt["receipt_content_sha256"])

    @property
    def capture_rules_sha256(self) -> str:
        return str(self.capture_rules["capture_rules_sha256"])

    def image_query_groups(self, image_id: str) -> list[Mapping[str, Any]]:
        return sorted(
            (row for row in self.query_groups.values() if str(row["image_id"]) == image_id),
            key=lambda row: str(row["query_group_id"]),
        )

    def image_categories(self, image_id: str) -> list[Mapping[str, Any]]:
        return sorted(
            (
                row
                for row in self.categories.values()
                if str(row["image_id"]) == image_id and row.get("status") == "admitted"
            ),
            key=lambda row: str(row["normalized_description"]),
        )

    def image_owners(self, image_id: str) -> list[Mapping[str, Any]]:
        return sorted(
            (row for row in self.owners.values() if str(row["image_id"]) == image_id),
            key=lambda row: str(row["gt_owner_id"]),
        )

    def image_native_sidecars(self, image_id: str) -> list[Mapping[str, Any]]:
        return [row for row in self.native_sidecars if str(row["image_id"]) == image_id]


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ShardContractError(f"{label} is missing at {path}")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ShardContractError(f"{label} line {index} is not valid JSON") from exc
    return rows


def load_plan(plan_dir: Path) -> PlanBundle:
    """Load and re-verify the sealed plan, including every output digest.

    This is also the pre-P0 gate: the scorer accepts a *plan directory* and
    nothing else, and the plan's own schema version must match this unit's, so
    a foreign or pre-P0 plan cannot be executed at all.
    """

    plan_dir = Path(plan_dir)
    receipt_path = plan_dir / "receipt.json"
    if not receipt_path.is_file():
        raise ShardContractError(f"plan receipt is missing at {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ShardContractError(
            "plan receipt schema_version does not match this unit's plan schema; "
            "refusing to score against a foreign or pre-P0 plan"
        )
    if receipt.get("unit_id") != UNIT_ID:
        raise ShardContractError("plan receipt unit_id does not match this unit")
    reconstructed = sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    if reconstructed != receipt.get("receipt_content_sha256"):
        raise ShardContractError("plan receipt does not reconstruct its own digest")

    digests = receipt.get("output_file_digests") or {}
    for name in planner.PLAN_FILE_NAMES:
        path = plan_dir / name
        if not path.is_file():
            raise ShardContractError(f"plan is missing declared file {name!r}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digests.get(name):
            raise ShardContractError(f"plan file {name!r} does not match its sealed digest")

    capture_rules_path = plan_dir / planner.CAPTURE_RULES_NAME
    if not capture_rules_path.is_file():
        raise ShardContractError("plan is missing capture-rules.json")
    capture_rules = json.loads(capture_rules_path.read_text(encoding="utf-8"))
    sealed = capture_rules.get("capture_rules_sha256")
    recomputed = sha256_json(
        {key: value for key, value in capture_rules.items() if key != "capture_rules_sha256"}
    )
    if sealed != recomputed:
        raise ShardContractError("capture-rules.json does not reconstruct its own digest")
    if capture_rules.get("schema_version") != planner.CAPTURE_RULES_SCHEMA_VERSION:
        raise ShardContractError("capture-rules.json schema_version does not match this unit")

    def index(name: str, key: str) -> dict[str, Mapping[str, Any]]:
        rows = _read_jsonl(plan_dir / name, name)
        indexed = {str(row[key]): row for row in rows}
        if len(indexed) != len(rows):
            raise ShardContractError(f"plan file {name!r} has duplicate {key} values")
        return indexed

    return PlanBundle(
        plan_dir=plan_dir,
        receipt=receipt,
        capture_rules=capture_rules,
        images=index("image-registry.jsonl", "image_id"),
        owners=index("owner-registry.jsonl", "gt_owner_id"),
        categories=index("category-registry.jsonl", "category_query_id"),
        contexts=index("context-registry.jsonl", "context_id"),
        candidates=index("candidate-bank.jsonl", "candidate_id"),
        query_groups=index("query-group-registry.jsonl", "query_group_id"),
        native_sidecars=_read_jsonl(plan_dir / "native-sidecar-registry.jsonl", "sidecars"),
        shards=index("shard-manifest.jsonl", "image_id"),
    )


# ---------------------------------------------------------------------------
# Work items: one singleton group, with cross-owner tuple collapse enforced
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkItem:
    """One singleton scoring group, fully resolved against the plan."""

    query_group_id: str
    image_id: str
    context_id: str
    normalized_description: str
    category_query_id: str
    observed_prefix_token_ids: list[int]
    query_suffix_token_ids: list[int]
    query_prefix_token_ids: list[int]
    observed_prefix_sha256: str
    query_prefix_sha256: str
    query_suffix_token_ids_sha256: str
    suffix_shape_class: str
    category_token_ids: list[int]
    candidates: list[Mapping[str, Any]]
    admission_receipt_id: str
    proposal_boundary_gate_admission_receipt_id: str
    proposal_route_admission_receipt_id: str

    @property
    def singleton_group_key(self) -> dict[str, str]:
        return {
            "image_id": self.image_id,
            "context_id": self.context_id,
            "normalized_description": self.normalized_description,
            "observed_prefix_sha256": self.observed_prefix_sha256,
            "query_prefix_sha256": self.query_prefix_sha256,
        }

    @property
    def probe_candidate(self) -> Mapping[str, Any]:
        """Deterministic, score-blind admission probe for this group.

        The lexicographically smallest physical candidate ID.  Candidate IDs
        are content digests of ``(image, category, coord tokens)``, so the
        choice is fixed by the plan and cannot follow any score.
        """

        return min(self.candidates, key=lambda row: str(row["candidate_id"]))


def collapse_group_candidates(
    candidates: Sequence[Mapping[str, Any]], *, label: str
) -> list[Mapping[str, Any]]:
    """Exactly one executed request per unique coordinate tuple; duplicates fail.

    Cross-owner identical tuples are one physical candidate by plan
    construction.  A duplicate surviving into a query group would double-count
    rank mass, so it is a contract violation rather than something to dedupe
    quietly.
    """

    by_id: dict[str, Mapping[str, Any]] = {}
    by_tokens: dict[tuple[int, ...], str] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in by_id:
            raise ShardContractError(
                f"{label}: physical candidate {candidate_id!r} appears more than once; "
                "the query group is not collapsed"
            )
        tokens = tuple(int(v) for v in candidate["coord_token_ids"])
        previous = by_tokens.get(tokens)
        if previous is not None:
            raise ShardContractError(
                f"{label}: candidates {previous!r} and {candidate_id!r} share coordinate "
                "tuple; cross-owner identical tuples must collapse to one physical candidate"
            )
        by_tokens[tokens] = candidate_id
        by_id[candidate_id] = candidate
    return [by_id[key] for key in sorted(by_id)]


def resolve_work_item(plan: PlanBundle, query_group_id: str) -> WorkItem:
    group = plan.query_groups.get(query_group_id)
    if group is None:
        raise ShardContractError(f"unknown query group {query_group_id!r}")
    if group.get("status") != "admitted":
        raise ShardContractError(
            f"query group {query_group_id!r} is not admitted ({group.get('status')})"
        )
    image_id = str(group["image_id"])
    context = plan.contexts[str(group["context_id"])]
    image = plan.images[image_id]
    category = plan.categories[str(group["category_query_id"])]

    observed_prefix = [
        *(int(v) for v in image["prompt_token_ids"]),
        *(int(v) for v in context["generated_prefix_token_ids"]),
    ]
    suffix = [int(v) for v in group["query_suffix_token_ids"]]
    query_prefix = [*observed_prefix, *suffix]

    observed_sha = sha256_json(observed_prefix)
    query_sha = sha256_json(query_prefix)
    if observed_sha != group["observed_prefix_sha256"]:
        raise ShardContractError(
            f"query group {query_group_id!r} observed prefix digest does not reconstruct"
        )
    if query_sha != group["query_prefix_sha256"]:
        raise ShardContractError(
            f"query group {query_group_id!r} query prefix digest does not reconstruct"
        )
    if sha256_json(suffix) != group["query_suffix_token_ids_sha256"]:
        raise ShardContractError(
            f"query group {query_group_id!r} query suffix digest does not reconstruct"
        )

    category_tokens = [int(v) for v in category["category_token_ids"]]
    # Contract item 5, enforced again at score time and not only at plan time:
    # the variable-length canonical suffix must be the literal tail and the
    # prefix must terminate at box_start.
    try:
        planner.assert_canonical_query_suffix(
            query_prefix,
            category_token_ids=category_tokens,
            label=f"work item {query_group_id}",
        )
    except planner.PlanContractError as exc:
        raise ShardContractError(str(exc)) from exc
    if suffix != planner.build_query_suffix(category_tokens):
        raise ShardContractError(
            f"query group {query_group_id!r} suffix is not this category's canonical suffix"
        )

    shape_class = planner.suffix_shape_class(category_tokens)
    declared_shape = group.get("suffix_shape_class")
    if declared_shape is not None and str(declared_shape) != shape_class:
        raise ShardContractError(
            f"query group {query_group_id!r} declares suffix shape {declared_shape!r} "
            f"but its canonical suffix is {shape_class!r}"
        )

    missing = [cid for cid in group["candidate_ids"] if str(cid) not in plan.candidates]
    if missing:
        raise ShardContractError(
            f"query group {query_group_id!r} references unknown candidates {sorted(missing)!r}"
        )
    raw_candidates = [plan.candidates[str(cid)] for cid in group["candidate_ids"]]
    for candidate in raw_candidates:
        if str(candidate["normalized_description"]) != str(group["normalized_description"]):
            raise ShardContractError(
                f"query group {query_group_id!r} contains a candidate of another category"
            )
        if str(candidate["image_id"]) != image_id:
            raise ShardContractError(
                f"query group {query_group_id!r} contains a candidate of another image"
            )
    candidates = collapse_group_candidates(
        raw_candidates, label=f"query group {query_group_id}"
    )
    if not candidates:
        raise ShardContractError(f"query group {query_group_id!r} has no candidate to score")

    expected_admission = planner.admission_receipt_id(
        context_id=str(group["context_id"]),
        channel=CHANNEL_QUERY_SUFFIX,
        prefix_sha256=query_sha,
    )
    declared_admission = group.get("admission_receipt_id")
    if declared_admission is not None and str(declared_admission) != expected_admission:
        raise ShardContractError(
            f"query group {query_group_id!r} declares an admission receipt that is not "
            "keyed on its own exact query prefix"
        )
    boundary_admission = planner.admission_receipt_id(
        context_id=str(group["context_id"]),
        channel=CHANNEL_PROPOSAL_BOUNDARY_GATE,
        prefix_sha256=observed_sha,
    )
    route_admission = planner.proposal_route_admission_receipt_id(
        context_id=str(group["context_id"]),
        observed_prefix_sha256=observed_sha,
        category_token_ids=category_tokens,
    )
    for field_name, expected in (
        ("proposal_boundary_gate_admission_receipt_id", boundary_admission),
        ("proposal_route_admission_receipt_id", route_admission),
    ):
        declared = group.get(field_name)
        if declared is not None and str(declared) != expected:
            raise ShardContractError(
                f"query group {query_group_id!r} declares a {field_name} that is not keyed "
                "on its own exact proposal identity"
            )

    return WorkItem(
        query_group_id=query_group_id,
        image_id=image_id,
        context_id=str(group["context_id"]),
        normalized_description=str(group["normalized_description"]),
        category_query_id=str(group["category_query_id"]),
        observed_prefix_token_ids=observed_prefix,
        query_suffix_token_ids=suffix,
        query_prefix_token_ids=query_prefix,
        observed_prefix_sha256=observed_sha,
        query_prefix_sha256=query_sha,
        query_suffix_token_ids_sha256=str(group["query_suffix_token_ids_sha256"]),
        suffix_shape_class=shape_class,
        category_token_ids=category_tokens,
        candidates=candidates,
        admission_receipt_id=expected_admission,
        proposal_boundary_gate_admission_receipt_id=boundary_admission,
        proposal_route_admission_receipt_id=route_admission,
    )


def validate_singleton_group(items: Sequence[WorkItem]) -> dict[str, str]:
    """Fail fast unless the scoring unit is exactly one group.

    Admission is measured on one exact prefix and is not transferable.  A
    scoring unit holding two groups would silently score the second under the
    first's admission.
    """

    if len(items) != 1:
        raise ShardContractError(
            f"a scoring unit must contain exactly one singleton group, received {len(items)}; "
            "exact-prefix admission is not transferable across groups"
        )
    return items[0].singleton_group_key


# ---------------------------------------------------------------------------
# Row bindings
# ---------------------------------------------------------------------------


def row_bindings(
    plan: PlanBundle,
    *,
    channel: str,
    admission_receipt_id: str,
    observed_prefix_sha256: str,
    query_prefix_sha256: str,
    query_suffix_token_ids_sha256: str,
) -> dict[str, Any]:
    if channel not in planner.ADMISSION_CHANNELS:
        raise ShardContractError(f"unknown admission channel {channel!r}")
    return {
        "row_contract": P0_ROW_CONTRACT,
        "unit_id": UNIT_ID,
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "plan_receipt_content_sha256": plan.receipt_content_sha256,
        "capture_rules_sha256": plan.capture_rules_sha256,
        "channel": channel,
        "admission_receipt_id": admission_receipt_id,
        "observed_prefix_sha256": observed_prefix_sha256,
        "query_prefix_sha256": query_prefix_sha256,
        "query_suffix_token_ids_sha256": query_suffix_token_ids_sha256,
    }


def assert_row_bindings(row: Mapping[str, Any], *, label: str) -> None:
    """Refuse any row that is not bound to this unit's P0 capture contract.

    A pre-P0 row carries no ``row_contract``, no ``channel``, and no
    ``admission_receipt_id``; it is rejected here rather than silently pooled.
    """

    missing = [
        key
        for key in REQUIRED_ROW_BINDING_FIELDS
        if row.get(key) in (None, "")
    ]
    if missing:
        raise ShardContractError(
            f"{label}: row is missing required P0 bindings {missing!r}; "
            "pre-P0 rows are mechanically unjoinable to this unit"
        )
    if row["row_contract"] != P0_ROW_CONTRACT:
        raise ShardContractError(
            f"{label}: row declares contract {row['row_contract']!r}, expected {P0_ROW_CONTRACT!r}"
        )
    if row["unit_id"] != UNIT_ID:
        raise ShardContractError(f"{label}: row belongs to another unit")
    if row["channel"] not in planner.ADMISSION_CHANNELS:
        raise ShardContractError(f"{label}: row declares unknown channel {row['channel']!r}")


# ---------------------------------------------------------------------------
# Numeric readout
# ---------------------------------------------------------------------------


@dataclass
class Readout:
    selected_logprob: float
    selected_token_id: int
    argmax_token_id: int
    argmax_logprob: float
    top_token_ids: list[int]
    top_logprobs: list[float]
    coord_distribution: list[float] | None


def readout(
    logits: Any,
    *,
    selected_token_id: int | None = None,
    top_k: int = DEFAULT_TOP_K,
    with_coord_distribution: bool = False,
) -> Readout:
    """Log-softmax over the *full* vocabulary, then read one token.

    Alternatives are full-vocabulary, not coordinate-restricted, so a
    non-coordinate token winning a coordinate slot is visible rather than
    hidden by a restricted domain.  ``selected_token_id=None`` reads the
    greedy argmax, which is how the free-decode loops select.
    """

    import torch

    flat = (
        logits.detach() if isinstance(logits, torch.Tensor) else torch.as_tensor(logits)
    )
    flat = flat.to(dtype=torch.float32).reshape(-1)
    log_probs = torch.log_softmax(flat, dim=-1)
    argmax_index = int(torch.argmax(log_probs))
    chosen = argmax_index if selected_token_id is None else int(selected_token_id)
    if chosen < 0 or chosen >= int(log_probs.numel()):
        raise ShardContractError(
            f"token id {chosen} is outside the executed vocabulary of {int(log_probs.numel())}"
        )
    top_values, top_indices = torch.topk(log_probs, k=min(int(top_k), int(log_probs.numel())))
    coord: list[float] | None = None
    if with_coord_distribution:
        if int(log_probs.numel()) <= COORD_TOKEN_END:
            raise ShardContractError(
                "executed vocabulary does not span the coordinate token domain"
            )
        coord = [float(v) for v in log_probs[COORD_TOKEN_START : COORD_TOKEN_END + 1].tolist()]
    return Readout(
        selected_logprob=float(log_probs[chosen]),
        selected_token_id=chosen,
        argmax_token_id=argmax_index,
        argmax_logprob=float(log_probs[argmax_index]),
        top_token_ids=[int(v) for v in top_indices.tolist()],
        top_logprobs=[float(v) for v in top_values.tolist()],
        coord_distribution=coord,
    )


def _finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(v)) for v in values)


# ---------------------------------------------------------------------------
# Backend seam
# ---------------------------------------------------------------------------


class CacheBackendLike(Protocol):
    """The production cache-branch surface (``HFCacheBackend``)."""

    @property
    def cache_length(self) -> int: ...

    def crop(self, length: int) -> None: ...

    def step(self, token_ids: Sequence[int]) -> Any: ...


@dataclass
class GroupPrefill:
    """One fresh KV cache rooted at exactly one literal prefix.

    ``cache_scope`` is the query group: this object is built per group, its
    root length is asserted against the prefill length, and it is released
    before the next group begins.
    """

    owner: "CensusBackend"
    cache_backend: Any
    root_logits: Any
    prefill_length: int
    root_token_ids: list[int]
    closed: bool = False

    def assert_rooted(self, *, label: str) -> None:
        if self.closed:
            raise ShardContractError(f"{label}: prefill has already been released")
        observed = int(self.cache_backend.cache_length)
        if observed != int(self.prefill_length):
            raise ShardContractError(
                f"{label}: cache length {observed} does not match the prefill length "
                f"{int(self.prefill_length)}; a branch leaked into the group root"
            )

    def branch(self):
        if self.closed:
            raise ShardContractError("cannot branch a released prefill")
        return _basin().BranchCursor(self.cache_backend)

    def batched_branch(self, width: int):
        """``width`` independent branches, all rooted at this exact prefill.

        Every branch starts from a byte-identical copy of the admitted root, so
        batching is a throughput device only: it cannot move probability mass
        between candidates.  The group root cache is never mutated.
        """

        if self.closed:
            raise ShardContractError("cannot branch a released prefill")
        if int(width) < 1:
            raise ShardContractError("a batched branch needs at least one candidate")
        return self.owner.batched_branch(self, int(width))

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.owner.release(self)
        self.cache_backend = None
        self.root_logits = None


class CensusBackend(Protocol):
    def prefill(self, token_ids: Sequence[int]) -> GroupPrefill: ...

    def full_reforward(self, token_ids: Sequence[int]) -> Any:
        """Uncached literal reforward of the *complete* prefix (prompt included)."""

    def release(self, prefill: GroupPrefill) -> None: ...

    def batched_branch(self, prefill: GroupPrefill, width: int):
        """Context manager yielding a ``width``-wide branch off one root."""

    @property
    def live_prefill_count(self) -> int: ...

    @property
    def expected_layer_count(self) -> int | None: ...

    @property
    def identity(self) -> Mapping[str, Any]: ...


class _PrefillTracker:
    """Shared bookkeeping: at most one live cache per process at a time."""

    def __init__(self) -> None:
        self._live: list[GroupPrefill] = []

    def _track(self, prefill: GroupPrefill) -> GroupPrefill:
        if self._live:
            raise ShardContractError(
                "a second KV cache was requested while one is still live; groups are "
                "scored sequentially with a fresh cache each"
            )
        self._live.append(prefill)
        return prefill

    def release(self, prefill: GroupPrefill) -> None:
        self._live = [row for row in self._live if row is not prefill]

    @property
    def live_prefill_count(self) -> int:
        return len(self._live)


# ---------------------------------------------------------------------------
# Phase ordering
# ---------------------------------------------------------------------------


class PhaseGuard:
    """Decision-bearing likelihood scoring strictly precedes free decoding.

    Free decoding is a generation phase.  Even though this module's greedy
    loops pass explicit positions, the ordering is kept hard so no future
    generation path (which may touch the model's shared ``rope_deltas``) can
    interleave with a decision-bearing score.
    """

    def __init__(self) -> None:
        self._generation_open = False

    @property
    def generation_open(self) -> bool:
        return self._generation_open

    def require_decision_phase(self, what: str) -> None:
        if self._generation_open:
            raise ShardContractError(
                f"{what} was requested after the terminal generation phase opened; "
                "no likelihood score may follow free decoding on this model instance"
            )

    def open_generation_phase(self, *, backend: CensusBackend) -> None:
        if self._generation_open:
            raise ShardContractError("the generation phase is already open")
        if backend.live_prefill_count:
            raise ShardContractError(
                "the generation phase cannot open while a decision-phase cache is still live"
            )
        self._generation_open = True

    def require_generation_phase(self, what: str) -> None:
        if not self._generation_open:
            raise ShardContractError(f"{what} may only run in the terminal generation phase")


# ---------------------------------------------------------------------------
# Admission: exact-prefix, per channel
# ---------------------------------------------------------------------------


def run_box_channel_admission(
    backend: CensusBackend,
    item: WorkItem,
    *,
    guard: PhaseGuard,
    relaxed_tolerance: float = RELAXED_SELECTED_LOGPROB_MAX_ABS_DIFF,
) -> dict[str, Any]:
    """Admit *this exact query prefix* with the production 4-depth parity gate.

    Reuses :func:`score_sorted_owner_basin_landscape.run_cache_parity_gate`
    verbatim, so the cache-branch path is checked against an independent
    uncached reforward at the root and after each of ``x1``/``y1``/``x2``,
    under raw and both repetition-penalty views, plus branch order invariance
    and cache-length restoration.  Admission is never inherited: a different
    category in the same context has a different exact prefix and therefore its
    own receipt.
    """

    guard.require_decision_phase("box-channel admission")
    basin = _basin()
    probe = item.probe_candidate
    coord_tokens = [int(v) for v in probe["coord_token_ids"]]
    if len(coord_tokens) != 4:
        raise ShardContractError(
            f"admission probe {probe['candidate_id']!r} does not carry four coordinate tokens"
        )

    prefill = backend.prefill(item.query_prefix_token_ids)
    try:
        prefill.assert_rooted(label=f"box admission {item.query_group_id}")
        gate = basin.run_cache_parity_gate(
            backend=prefill.cache_backend,
            prefill_logits=prefill.root_logits,
            prefix_token_ids=item.query_prefix_token_ids,
            x1_token_id=coord_tokens[0],
            y1_token_id=coord_tokens[1],
            x2_token_id=coord_tokens[2],
            y2_token_id=coord_tokens[3],
            coordinate_token_id_start=COORD_TOKEN_START,
            coordinate_token_id_end_exclusive=COORD_TOKEN_END + 1,
            full_reforward=backend.full_reforward,
            expected_layer_count=backend.expected_layer_count,
            relaxed_selected_logprob_max_abs_diff=float(relaxed_tolerance),
        )
        prefill.assert_rooted(label=f"box admission {item.query_group_id} (post-gate)")
    finally:
        prefill.close()

    selection = basin.select_scoring_backend_from_parity(gate)
    admitted = selection["selected_backend"] == basin.KV_CACHE_SCORING_BACKEND
    return {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "admission_receipt_id": item.admission_receipt_id,
        "channel": CHANNEL_QUERY_SUFFIX,
        "context_id": item.context_id,
        "query_group_id": item.query_group_id,
        "normalized_description": item.normalized_description,
        "admission_key": "exact_query_prefix_sha256",
        "observed_prefix_sha256": item.observed_prefix_sha256,
        "query_prefix_sha256": item.query_prefix_sha256,
        "query_suffix_token_ids_sha256": item.query_suffix_token_ids_sha256,
        "suffix_shape_class": item.suffix_shape_class,
        "inherited_from_another_prefix": False,
        "probe_candidate_id": str(probe["candidate_id"]),
        "probe_selection_rule": "lexicographically_smallest_physical_candidate_id",
        "probe_is_score_blind": True,
        "depth_count": 4,
        "parity_gate": gate,
        "scoring_backend_selection": selection,
        "relaxed_selected_logprob_max_abs_diff": float(relaxed_tolerance),
        "admitted": bool(admitted),
        "bulk_scoring_path": "admitted_kv_cache",
    }


#: The literal forced path one category routing event executes, taken from the
#: planner so the scorer and the plan cannot drift apart.
proposal_routing_path_token_ids = planner.proposal_route_token_ids


def _observed_prefix_token_ids(plan: PlanBundle, context_id: str) -> list[int]:
    context = plan.contexts[context_id]
    image = plan.images[str(context["image_id"])]
    return [
        *(int(v) for v in image["prompt_token_ids"]),
        *(int(v) for v in context["generated_prefix_token_ids"]),
    ]


def _compare_cache_versus_reforward(
    backend: CensusBackend,
    *,
    root_token_ids: Sequence[int],
    path_token_ids: Sequence[int],
    selected_token_ids: Sequence[int],
    label: str,
    relaxed_tolerance: float,
) -> list[dict[str, Any]]:
    """Depth-by-depth cache-branch versus independent uncached reforward.

    Structurally the same check the production coordinate gate performs, but
    over an arbitrary literal path, because the proposal channel steps wrapper
    and category tokens rather than coordinate tokens and therefore cannot be
    covered by a coordinate-domain gate.
    """

    import torch

    basin = _basin()
    path = [int(v) for v in path_token_ids]
    selected = [int(v) for v in selected_token_ids]
    if len(selected) != len(path) + 1:
        raise ShardContractError(f"{label}: one selected token is required per depth")

    depths: list[dict[str, Any]] = []
    prefill = backend.prefill(root_token_ids)
    try:
        prefill.assert_rooted(label=label)
        cache_views = [prefill.root_logits]
        if path:
            with prefill.branch() as branch:
                for token_id in path:
                    cache_views.append(branch.step([token_id])[-1])
            prefill.assert_rooted(label=f"{label} (post-branch)")

        running = [int(v) for v in root_token_ids]
        for index, cache_logits in enumerate(cache_views):
            reference = backend.full_reforward(running)
            cache_vector = (
                torch.as_tensor(cache_logits).detach().to(dtype=torch.float32).reshape(-1)
            )
            reference_vector = (
                torch.as_tensor(reference).detach().to(dtype=torch.float32).reshape(-1)
            )
            if cache_vector.shape != reference_vector.shape:
                raise ShardContractError(
                    f"{label}: cache and reforward logits have different shapes"
                )
            selected_token = selected[index]
            cache_view = readout(cache_vector, selected_token_id=selected_token, top_k=1)
            reference_view = readout(reference_vector, selected_token_id=selected_token, top_k=1)
            selected_abs_diff = abs(
                cache_view.selected_logprob - reference_view.selected_logprob
            )
            depths.append(
                {
                    "depth_index": index,
                    "executed_context_token_count": len(running),
                    "selected_token_id": selected_token,
                    "full_vocabulary_max_abs_diff": float(
                        (cache_vector - reference_vector).abs().max().item()
                    ),
                    "strict_within_tolerance": bool(
                        torch.allclose(
                            cache_vector,
                            reference_vector,
                            atol=basin.CACHE_PARITY_ATOL,
                            rtol=basin.CACHE_PARITY_RTOL,
                        )
                    ),
                    "argmax_parity": cache_view.argmax_token_id == reference_view.argmax_token_id,
                    "selected_logprob_cache": cache_view.selected_logprob,
                    "selected_logprob_reforward": reference_view.selected_logprob,
                    "selected_logprob_abs_diff": selected_abs_diff,
                    "relaxed_within_tolerance": selected_abs_diff <= float(relaxed_tolerance),
                    "finite": math.isfinite(cache_view.selected_logprob)
                    and math.isfinite(reference_view.selected_logprob),
                }
            )
            if index < len(path):
                running.append(path[index])
    finally:
        prefill.close()
    return depths


def _proposal_admission_verdict(depths: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    basin = _basin()
    strict_passed = all(row["strict_within_tolerance"] for row in depths)
    argmax_parity = all(row["argmax_parity"] for row in depths)
    relaxed_passed = all(row["relaxed_within_tolerance"] for row in depths)
    finite = all(row["finite"] for row in depths)
    admitted = bool(finite and argmax_parity and (strict_passed or relaxed_passed))
    return {
        "depth_count": len(depths),
        "depths": list(depths),
        "strict_full_vocabulary_parity_passed": strict_passed,
        "all_argmax_parity": argmax_parity,
        "relaxed_selected_logprob_within_tolerance": relaxed_passed,
        "finite": finite,
        "effective_mode": (
            basin.STRICT_CACHE_ADMISSION_MODE
            if strict_passed
            else basin.RELAXED_CACHE_ADMISSION_MODE
            if admitted
            else basin.UNCACHED_CACHE_ADMISSION_MODE
        ),
        "admitted": admitted,
    }


#: Per-candidate agreement required between the batched and single-branch
#: paths on the live model.  Batching changes GEMM shapes, so exact bitwise
#: equality is not guaranteed; this is the bound a capture must meet for the
#: two paths to be interchangeable evidence.
BATCHED_PATH_MAX_ABS_DIFF = 1e-4


def run_batched_path_parity(
    backend: CensusBackend,
    item: WorkItem,
    *,
    candidate_batch_size: int,
    max_abs_diff: float = BATCHED_PATH_MAX_ABS_DIFF,
    top_k: int = 1,
) -> dict[str, Any]:
    """Prove on the live model that batched lanes equal the single branch.

    Scores the group's first ``candidate_batch_size`` candidates -- distinct
    coordinate tuples, in the plan's own order, chosen without reference to any
    score -- once through each path at the *real* lane width, then compares
    them **candidate by candidate**.  Contamination between lanes would move a
    candidate's mass onto its neighbour's tokens and show up here as a
    per-candidate mismatch, so this is a contamination test as well as a
    numerical one.

    Failure raises: there is no silent fallback to the other path.
    """

    width = max(2, int(candidate_batch_size))
    probes = list(item.candidates[:width])
    if len(probes) < 2:
        probes = list(item.candidates)
    request_ids = [f"{item.query_group_id}|{row['candidate_id']}" for row in probes]
    bindings: dict[str, Any] = {}

    prefill = backend.prefill(item.query_prefix_token_ids)
    try:
        single_rows = _score_candidates_single(
            prefill, item, probes, request_ids, bindings=bindings, top_k=top_k
        )
        batched_rows = _score_candidates_batched(
            prefill,
            item,
            probes,
            request_ids,
            bindings=bindings,
            top_k=top_k,
            candidate_batch_size=len(probes),
        )
    finally:
        prefill.close()

    per_candidate: list[dict[str, Any]] = []
    for single, batched in zip(single_rows, batched_rows, strict=True):
        if single["candidate_id"] != batched["candidate_id"]:
            raise ShardContractError(
                "batched scoring returned candidates in a different order than the "
                "single-branch path"
            )
        slot_diffs = [
            abs(one["selected_logprob"] - other["selected_logprob"])
            for one, other in zip(single["per_coordinate"], batched["per_coordinate"], strict=True)
        ]
        argmax_parity = all(
            one["argmax_token_id"] == other["argmax_token_id"]
            for one, other in zip(single["per_coordinate"], batched["per_coordinate"], strict=True)
        )
        sum_diff = abs(
            float(single["complete_box_logprob_sum"]) - float(batched["complete_box_logprob_sum"])
        )
        per_candidate.append(
            {
                "candidate_id": single["candidate_id"],
                "coord_token_ids": list(single["coord_token_ids"]),
                "single_complete_box_logprob_sum": float(single["complete_box_logprob_sum"]),
                "batched_complete_box_logprob_sum": float(batched["complete_box_logprob_sum"]),
                "complete_box_logprob_sum_abs_diff": sum_diff,
                "max_selected_logprob_abs_diff": max(slot_diffs, default=0.0),
                "all_argmax_parity": argmax_parity,
            }
        )

    observed = max(
        (
            max(row["complete_box_logprob_sum_abs_diff"], row["max_selected_logprob_abs_diff"])
            for row in per_candidate
        ),
        default=0.0,
    )
    argmax_parity = all(row["all_argmax_parity"] for row in per_candidate)
    distinct_tuples = {tuple(row["coord_token_ids"]) for row in per_candidate}
    return {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "scope": "batched_candidate_path",
        "query_group_id": item.query_group_id,
        "lane_width": len(probes),
        "candidate_batch_size": int(candidate_batch_size),
        "probe_candidate_ids": [row["candidate_id"] for row in per_candidate],
        "probe_selection_rule": "first_candidates_in_sealed_plan_order",
        "probe_is_score_blind": True,
        "distinct_coordinate_tuple_count": len(distinct_tuples),
        "lanes_carry_distinct_tokens": len(distinct_tuples) == len(per_candidate),
        "per_candidate": per_candidate,
        "max_abs_diff_observed": observed,
        "max_abs_diff_bound": float(max_abs_diff),
        "all_argmax_parity": argmax_parity,
        "bitwise_identical": observed == 0.0,
        "admitted": bool(argmax_parity and observed <= float(max_abs_diff)),
        "on_failure": "fail_closed_no_silent_path_fallback",
    }


def run_proposal_boundary_admission(
    backend: CensusBackend,
    plan: PlanBundle,
    *,
    context_id: str,
    guard: PhaseGuard,
    relaxed_tolerance: float = RELAXED_SELECTED_LOGPROB_MAX_ABS_DIFF,
) -> dict[str, Any]:
    """Admit the *observed prefix with no query suffix* at the boundary itself.

    The boundary gate is a root-level readout with nothing forced, so its
    admission is the root-depth cache-versus-reforward reference on the exact
    observed prefix.  It authorizes the gate only: every forced category
    routing path is admitted separately by
    :func:`run_proposal_path_admission`.
    """

    guard.require_decision_phase("proposal boundary admission")
    observed_prefix = _observed_prefix_token_ids(plan, context_id)
    observed_sha = sha256_json(observed_prefix)
    depths = _compare_cache_versus_reforward(
        backend,
        root_token_ids=observed_prefix,
        path_token_ids=[],
        selected_token_ids=[OBJECT_REF_START],
        label=f"proposal boundary admission {context_id}",
        relaxed_tolerance=relaxed_tolerance,
    )
    verdict = _proposal_admission_verdict(depths)
    return {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "admission_receipt_id": planner.admission_receipt_id(
            context_id=context_id,
            channel=CHANNEL_PROPOSAL_BOUNDARY_GATE,
            prefix_sha256=observed_sha,
        ),
        "channel": CHANNEL_PROPOSAL_BOUNDARY_GATE,
        "proposal_scope": "boundary_gate",
        "context_id": context_id,
        "category_query_id": None,
        "admission_key": "exact_observed_prefix_sha256",
        "observed_prefix_sha256": observed_sha,
        "query_prefix_sha256": observed_sha,
        "query_suffix_token_ids_sha256": EMPTY_SUFFIX_SHA256,
        "suffix_shape_class": "no_query_suffix",
        "inherited_from_another_prefix": False,
        "execution_shape": "observed_prefix_root_readout_nothing_forced",
        "covered_by_a_query_suffix_admission": False,
        "authorizes_category_routing_paths": False,
        "relaxed_selected_logprob_max_abs_diff": float(relaxed_tolerance),
        "bulk_scoring_path": "admitted_kv_cache",
        **verdict,
    }


def run_proposal_path_admission(
    backend: CensusBackend,
    plan: PlanBundle,
    *,
    context_id: str,
    category: Mapping[str, Any],
    guard: PhaseGuard,
    relaxed_tolerance: float = RELAXED_SELECTED_LOGPROB_MAX_ABS_DIFF,
) -> dict[str, Any]:
    """Admit exactly one ``(context, category routing path)``.

    Every forced path has its own token content *and* its own length, so a
    first-category probe can never authorize another category's path.  The
    admission is keyed on the exact executed path digest and checks parity at
    every depth of that path, including the terminal ``box_start`` readout the
    ``row_prefix_block`` quantity consumes.
    """

    guard.require_decision_phase("proposal path admission")
    observed_prefix = _observed_prefix_token_ids(plan, context_id)
    observed_sha = sha256_json(observed_prefix)
    category_tokens = [int(v) for v in category["category_token_ids"]]
    path = proposal_routing_path_token_ids(category_tokens)
    route_digest = planner.proposal_route_digest(category_tokens)
    executed_sha = sha256_json([*observed_prefix, *path])
    depths = _compare_cache_versus_reforward(
        backend,
        root_token_ids=observed_prefix,
        path_token_ids=path,
        # Root predicts object_ref_start; each forced token predicts the next;
        # the final depth predicts box_start.
        selected_token_ids=[*path, BOX_START],
        label=f"proposal path admission {context_id}|{category['normalized_description']}",
        relaxed_tolerance=relaxed_tolerance,
    )
    verdict = _proposal_admission_verdict(depths)
    return {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "admission_receipt_id": planner.proposal_route_admission_receipt_id(
            context_id=context_id,
            observed_prefix_sha256=observed_sha,
            category_token_ids=category_tokens,
        ),
        "channel": CHANNEL_PROPOSAL_CATEGORY_ROUTE,
        "proposal_scope": "category_routing_path",
        "context_id": context_id,
        "category_query_id": str(category["category_query_id"]),
        "normalized_description": str(category["normalized_description"]),
        "admission_key": "exact_executed_routing_path_sha256",
        "observed_prefix_sha256": observed_sha,
        "query_prefix_sha256": observed_sha,
        "query_suffix_token_ids_sha256": EMPTY_SUFFIX_SHA256,
        "routing_path_token_ids": path,
        "routing_path_digest": route_digest,
        "executed_routing_prefix_sha256": executed_sha,
        "routing_path_token_count": len(path),
        "routing_path_shape_class": f"routing_path_len_{len(path)}",
        "suffix_shape_class": planner.suffix_shape_class(category_tokens),
        "inherited_from_another_prefix": False,
        "inherited_from_another_category": False,
        "execution_shape": "observed_prefix_then_wrapper_and_category_tokens",
        "covered_by_a_query_suffix_admission": False,
        "relaxed_selected_logprob_max_abs_diff": float(relaxed_tolerance),
        "bulk_scoring_path": "admitted_kv_cache",
        **verdict,
    }


def run_image_epsilon_receipt(
    backend: CensusBackend,
    *,
    probe_token_ids: Sequence[int],
    probe_label: str,
    repeat_count: int,
    guard: PhaseGuard,
) -> dict[str, Any]:
    """One target-blind repeat-noise receipt per image.

    Repeats an *uncached* reforward of one fixed shared prefix and reads a
    fixed probe token.  It reads no candidate, so the measured epsilon cannot
    depend on any target, geometry, owner, or score.  Per-group parity is the
    separate 4-depth gate; this receipt bounds run-to-run numerical noise once
    for the whole image.
    """

    guard.require_decision_phase("image epsilon receipt")
    if int(repeat_count) < 2:
        raise ShardContractError("a repeat-noise receipt needs at least two repeats")
    logprobs: list[float] = []
    argmaxes: list[int] = []
    for _ in range(int(repeat_count)):
        view = readout(
            backend.full_reforward(list(probe_token_ids)),
            selected_token_id=COORD_TOKEN_START,
            top_k=1,
        )
        logprobs.append(view.selected_logprob)
        argmaxes.append(view.argmax_token_id)
    median = sorted(logprobs)[len(logprobs) // 2]
    epsilon = max(abs(value - median) for value in logprobs)
    return {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "scope": "image",
        "target_blind": True,
        "reads_any_candidate": False,
        "probe_label": probe_label,
        "probe_prefix_sha256": sha256_json([int(v) for v in probe_token_ids]),
        "probe_token_id": COORD_TOKEN_START,
        "backend": "full_reforward_uncached",
        "repeat_count": int(repeat_count),
        "repeat_logprobs": logprobs,
        "repeat_median": median,
        "repeat_max_abs_deviation": epsilon,
        "repeat_argmax_token_ids": argmaxes,
        "argmax_stable": len(set(argmaxes)) == 1,
        "finite": _finite(logprobs),
    }


def assert_suffix_shape_coverage(
    admissions: Mapping[str, Mapping[str, Any]], items: Sequence[WorkItem]
) -> dict[str, Any]:
    """Every scored ``(context, suffix shape class)`` must be admission-covered.

    Exact-prefix admission is strictly finer than shape-class admission, so
    this is a *coverage audit*, not the admission key: it makes an execution
    shape that was never admitted visible instead of silently absent.
    """

    required: dict[tuple[str, str], list[str]] = {}
    for item in items:
        required.setdefault((item.context_id, item.suffix_shape_class), []).append(
            item.query_group_id
        )
    covered: set[tuple[str, str]] = set()
    for receipt in admissions.values():
        if receipt.get("channel") != CHANNEL_QUERY_SUFFIX:
            continue
        if not receipt.get("admitted"):
            continue
        covered.add((str(receipt["context_id"]), str(receipt["suffix_shape_class"])))
    missing = sorted(key for key in required if key not in covered)
    if missing:
        raise ShardContractError(
            "box-channel admission does not cover every executed query-suffix shape class: "
            f"{[list(key) for key in missing]!r}"
        )
    return {
        "required_context_shape_class_count": len(required),
        "covered_context_shape_class_count": len(covered & set(required)),
        "distinct_suffix_shape_classes": sorted({key[1] for key in required}),
        "admission_key": "exact_prefix_sha256",
        "shape_class_role": "coverage_audit_only_never_the_admission_key",
    }


# ---------------------------------------------------------------------------
# Deterministic, target-blind scalar reference selection
# ---------------------------------------------------------------------------


def _request_digest(request_id: str) -> int:
    return int.from_bytes(hashlib.sha256(request_id.encode("utf-8")).digest()[:8], "big")


def scalar_reference_selected(request_id: str, *, modulus: int) -> bool:
    """Target-blind spot-check membership from the request-ID digest only.

    Never a function of owner identity, candidate class, geometry, or score.
    """

    return _request_digest(request_id) % int(modulus) == 0


def select_scalar_reference_request_ids(
    request_ids: Sequence[str], *, modulus: int
) -> tuple[set[str], str | None]:
    """Digest-selected spot checks, with at least one guaranteed per group.

    The modulus can select nothing in a small group.  Rather than leave a group
    unchecked, the request with the smallest digest is added; the choice is
    still a pure function of request IDs and therefore still target-blind.
    """

    selected = {rid for rid in request_ids if scalar_reference_selected(rid, modulus=modulus)}
    forced: str | None = None
    if not selected and request_ids:
        forced = min(request_ids, key=lambda rid: (_request_digest(rid), rid))
        selected.add(forced)
    return selected, forced


# ---------------------------------------------------------------------------
# Box channel: localization scoring for one singleton query group
# ---------------------------------------------------------------------------


def _coordinate_view(
    logits: Any, *, slot: str, token_id: int, top_k: int
) -> dict[str, Any]:
    view = readout(logits, selected_token_id=int(token_id), top_k=top_k)
    return {
        "slot": slot,
        "token_id": int(token_id),
        "coord_bin": int(token_id) - COORD_TOKEN_START,
        "selected_logprob": view.selected_logprob,
        "argmax_token_id": view.argmax_token_id,
        "argmax_logprob": view.argmax_logprob,
        "selected_is_argmax": view.argmax_token_id == int(token_id),
        "top_alternatives": {
            "token_ids": view.top_token_ids,
            "logprobs": view.top_logprobs,
        },
    }


def _assert_coordinate_tokens(candidate: Mapping[str, Any]) -> list[int]:
    coord_tokens = [int(v) for v in candidate["coord_token_ids"]]
    if len(coord_tokens) != 4 or any(
        token < COORD_TOKEN_START or token > COORD_TOKEN_END for token in coord_tokens
    ):
        raise ShardContractError(
            f"candidate {candidate['candidate_id']!r} has non-coordinate tokens"
        )
    return coord_tokens


def _build_score_row(
    item: WorkItem,
    candidate: Mapping[str, Any],
    *,
    bindings: Mapping[str, Any],
    request_id: str,
    coord_tokens: Sequence[int],
    per_coordinate: Sequence[Mapping[str, Any]],
    box_end_view: Readout,
) -> dict[str, Any]:
    """One localization score row.  Identical for the single and batched paths."""

    coordinate_logprobs = [row["selected_logprob"] for row in per_coordinate]
    sequence_sum = math.fsum(coordinate_logprobs)
    if not _finite([*coordinate_logprobs, box_end_view.selected_logprob, sequence_sum]):
        raise ShardContractError(
            f"candidate {candidate['candidate_id']!r} produced a non-finite logprob"
        )
    return {
        **bindings,
        "schema_version": SCORE_SCHEMA_VERSION,
        "row_kind": "census_localization_score",
        "request_id": request_id,
        "rank_key": {
            "image_id": item.image_id,
            "context_id": item.context_id,
            "normalized_description": item.normalized_description,
        },
        "query_group_id": item.query_group_id,
        "image_id": item.image_id,
        "context_id": item.context_id,
        "normalized_description": item.normalized_description,
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_identity_rule": candidate.get("identity_rule"),
        "candidate_collapse_scope": candidate.get("collapse_scope"),
        "candidate_class": candidate.get("candidate_class"),
        "candidate_provenance": candidate.get("candidate_provenance"),
        # Generator provenance is carried, never consulted: it enters
        # no rank and no assignment.
        "generators": list(candidate.get("generators") or []),
        "generator_gt_owner_ids": list(candidate.get("generator_gt_owner_ids") or []),
        "generator_owner_count": candidate.get("generator_owner_count"),
        "cross_owner_generated": candidate.get("cross_owner_generated"),
        "generator_provenance_role": "provenance_only_never_rank_or_assignment",
        "coord_token_ids": [int(v) for v in coord_tokens],
        "coord_token_ids_sha256": candidate["coord_token_ids_sha256"],
        "complete_box_logprob_sum": sequence_sum,
        "per_coordinate": [dict(row) for row in per_coordinate],
        "box_end": {
            "token_id": BOX_END,
            "selected_logprob": box_end_view.selected_logprob,
            "argmax_token_id": box_end_view.argmax_token_id,
            "selected_is_argmax": box_end_view.argmax_token_id == BOX_END,
        },
        "scalar_reference": None,
        # Assignment is the plan's category-local geometry decision,
        # computed once and never re-derived from the generator.
        "assignment": {
            "strict_assignment_scope": candidate.get("strict_assignment_scope"),
            "strict_assignment_status": candidate.get("strict_assignment_status"),
            "strict_assignment_gt_owner_id": candidate.get("strict_assignment_gt_owner_id"),
            "ambiguity_owner_ids": candidate.get("ambiguity_owner_ids"),
            "any_category_assignment_status": candidate.get("any_category_assignment_status"),
            "any_category_assignment_role": candidate.get("any_category_assignment_role"),
        },
        "geometry": {
            "coord_bins": candidate.get("coord_bins"),
            "decoded_bbox_pixel_xyxy": candidate.get("decoded_bbox_pixel_xyxy"),
        },
        "repetition_penalty_stratum": planner.NATIVE_REPETITION_PENALTY_STRATUM,
        "is_sidecar": False,
        "enters_core_ranks": True,
    }


def _score_candidates_single(
    prefill: GroupPrefill,
    item: WorkItem,
    candidates: Sequence[Mapping[str, Any]],
    request_ids: Sequence[str],
    *,
    bindings: Mapping[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    """One branch per candidate: the frozen default path."""

    rows: list[dict[str, Any]] = []
    for candidate, request_id in zip(candidates, request_ids, strict=True):
        coord_tokens = _assert_coordinate_tokens(candidate)
        per_coordinate: list[dict[str, Any]] = []
        with prefill.branch() as branch:
            logits = prefill.root_logits
            for slot, token_id in zip(COORD_SLOTS, coord_tokens, strict=True):
                per_coordinate.append(
                    _coordinate_view(logits, slot=slot, token_id=token_id, top_k=top_k)
                )
                logits = branch.step([token_id])[-1]
            box_end_view = readout(logits, selected_token_id=BOX_END, top_k=top_k)
        prefill.assert_rooted(
            label=f"query group {item.query_group_id} after {candidate['candidate_id']}"
        )
        rows.append(
            _build_score_row(
                item,
                candidate,
                bindings=bindings,
                request_id=request_id,
                coord_tokens=coord_tokens,
                per_coordinate=per_coordinate,
                box_end_view=box_end_view,
            )
        )
    return rows


def _score_candidates_batched(
    prefill: GroupPrefill,
    item: WorkItem,
    candidates: Sequence[Mapping[str, Any]],
    request_ids: Sequence[str],
    *,
    bindings: Mapping[str, Any],
    top_k: int,
    candidate_batch_size: int,
) -> list[dict[str, Any]]:
    """Independent candidates of one exact query group, scored in lanes.

    Only candidates of *this* group and *this* admitted prefix are ever batched
    together, every lane starts at the identical root, coordinate decoding is
    still exactly four teacher-forced steps, and rows are appended in the
    original candidate order.  A short tail batch is scored the same way.
    """

    rows: list[dict[str, Any]] = []
    for start in range(0, len(candidates), int(candidate_batch_size)):
        chunk = list(candidates[start : start + int(candidate_batch_size)])
        chunk_requests = list(request_ids[start : start + int(candidate_batch_size)])
        coord_matrix = [_assert_coordinate_tokens(candidate) for candidate in chunk]
        per_lane: list[list[dict[str, Any]]] = [[] for _ in chunk]

        with prefill.batched_branch(len(chunk)) as lanes:
            # Slot x1 is read from the shared root for every lane: it is the
            # same tensor the single-branch path reads.
            logits_per_lane: list[Any] = [prefill.root_logits] * len(chunk)
            for slot_index, slot in enumerate(COORD_SLOTS):
                for lane_index in range(len(chunk)):
                    per_lane[lane_index].append(
                        _coordinate_view(
                            logits_per_lane[lane_index],
                            slot=slot,
                            token_id=coord_matrix[lane_index][slot_index],
                            top_k=top_k,
                        )
                    )
                stepped = lanes.step(
                    [[coord_matrix[lane_index][slot_index]] for lane_index in range(len(chunk))]
                )
                logits_per_lane = [stepped[lane_index][-1] for lane_index in range(len(chunk))]
            box_end_views = [
                readout(logits_per_lane[lane_index], selected_token_id=BOX_END, top_k=top_k)
                for lane_index in range(len(chunk))
            ]
        prefill.assert_rooted(
            label=f"query group {item.query_group_id} after batch at offset {start}"
        )

        for lane_index, candidate in enumerate(chunk):
            rows.append(
                _build_score_row(
                    item,
                    candidate,
                    bindings=bindings,
                    request_id=chunk_requests[lane_index],
                    coord_tokens=coord_matrix[lane_index],
                    per_coordinate=per_lane[lane_index],
                    box_end_view=box_end_views[lane_index],
                )
            )
    return rows


def score_query_group(
    backend: CensusBackend,
    plan: PlanBundle,
    item: WorkItem,
    *,
    guard: PhaseGuard,
    admission: Mapping[str, Any],
    top_k: int = DEFAULT_TOP_K,
    scalar_modulus: int,
    capture_full_x1: bool = True,
    candidate_batch_size: int = DEFAULT_CANDIDATE_BATCH_SIZE,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Score every collapsed unique physical candidate of one query group.

    One fresh prefill roots the group.  With ``candidate_batch_size == 1``
    (the frozen default) each candidate is scored on its own
    :class:`BranchCursor` off that root and the cache is cropped back
    afterwards.  Above one, independent candidates *of this same exact query
    group* are scored in parallel lanes off byte-identical copies of the same
    admitted root.  Either way candidates never contaminate each other,
    coordinate decoding is four teacher-forced steps, and the emitted rows and
    their order are identical.
    """

    guard.require_decision_phase(f"box scoring for {item.query_group_id}")
    if not admission.get("admitted"):
        raise ShardContractError(
            f"query group {item.query_group_id!r} has no passing box-channel admission"
        )
    if str(admission.get("admission_receipt_id")) != item.admission_receipt_id:
        raise ShardContractError(
            f"query group {item.query_group_id!r} was handed an admission receipt for "
            "another exact prefix"
        )

    bindings = row_bindings(
        plan,
        channel=CHANNEL_QUERY_SUFFIX,
        admission_receipt_id=item.admission_receipt_id,
        observed_prefix_sha256=item.observed_prefix_sha256,
        query_prefix_sha256=item.query_prefix_sha256,
        query_suffix_token_ids_sha256=item.query_suffix_token_ids_sha256,
    )
    request_ids = [
        f"{item.query_group_id}|{candidate['candidate_id']}" for candidate in item.candidates
    ]
    scalar_selected, forced_request_id = select_scalar_reference_request_ids(
        request_ids, modulus=int(scalar_modulus)
    )

    checks: dict[str, Any] = {
        "finite": True,
        "token_alignment": True,
        "coordinate_domain": True,
        "scalar_reference_count": 0,
        "scalar_max_abs_diff": 0.0,
        "scalar_within_bound": True,
        "scalar_forced_request_id": forced_request_id,
        "cache_root_restored_after_every_candidate": True,
        "candidate_batch_size": int(candidate_batch_size),
        "candidate_batching_enabled": int(candidate_batch_size) > 1,
    }
    rows: list[dict[str, Any]] = []
    scalar_probes: list[tuple[dict[str, Any], float]] = []

    if int(candidate_batch_size) < 1:
        raise ShardContractError("candidate batch size must be at least one")

    prefill = backend.prefill(item.query_prefix_token_ids)
    try:
        prefill.assert_rooted(label=f"query group {item.query_group_id}")
        # The x1 distribution is read *only here*: immediately after the
        # canonical suffix's box_start (contract item 5/8).
        root = readout(
            prefill.root_logits,
            selected_token_id=COORD_TOKEN_START,
            top_k=top_k,
            with_coord_distribution=capture_full_x1,
        )
        x1_distribution = root.coord_distribution
        if x1_distribution is not None and len(x1_distribution) != COORD_BIN_COUNT:
            raise ShardContractError("x1 coordinate distribution does not span exactly 1000 bins")

        if int(candidate_batch_size) <= 1:
            rows = _score_candidates_single(
                prefill,
                item,
                item.candidates,
                request_ids,
                bindings=bindings,
                top_k=top_k,
            )
        else:
            rows = _score_candidates_batched(
                prefill,
                item,
                item.candidates,
                request_ids,
                bindings=bindings,
                top_k=top_k,
                candidate_batch_size=int(candidate_batch_size),
            )
        for row in rows:
            if row["request_id"] in scalar_selected:
                scalar_probes.append((row, float(row["complete_box_logprob_sum"])))
    finally:
        prefill.close()

    # Scalar spot checks are uncached literal reforwards.  They run after the
    # group's cache is released so the two paths cannot share state.
    for row, sequence_sum in scalar_probes:
        scalar_sum = _scalar_reference_sequence_sum(
            backend, item, [int(v) for v in row["coord_token_ids"]]
        )
        difference = abs(scalar_sum - sequence_sum)
        checks["scalar_reference_count"] += 1
        checks["scalar_max_abs_diff"] = max(float(checks["scalar_max_abs_diff"]), difference)
        row["scalar_reference"] = {
            "complete_box_logprob_sum": scalar_sum,
            "abs_diff_vs_cached": difference,
            "backend": "full_reforward_uncached",
            "selection": (
                "forced_minimum_digest_guarantee"
                if row["request_id"] == forced_request_id
                else "request_id_digest_modulus"
            ),
            "target_blind": True,
        }

    if checks["scalar_reference_count"] < 1:
        raise ShardContractError(
            f"query group {item.query_group_id!r} produced no scalar spot check; "
            "every group must carry at least one uncached reference"
        )
    checks["scalar_within_bound"] = (
        float(checks["scalar_max_abs_diff"]) <= planner.SCALAR_VS_BATCH_MAX_ABS_DIFF
    )
    if not checks["scalar_within_bound"]:
        raise ShardContractError(
            f"query group {item.query_group_id!r} scalar-versus-cached difference "
            f"{checks['scalar_max_abs_diff']} exceeds {planner.SCALAR_VS_BATCH_MAX_ABS_DIFF}"
        )

    competition = attach_competition_ranks(rows)
    diagnostic = {
        **bindings,
        "schema_version": X1_SCHEMA_VERSION,
        "row_kind": "census_x1_distribution",
        "query_group_id": item.query_group_id,
        "image_id": item.image_id,
        "context_id": item.context_id,
        "normalized_description": item.normalized_description,
        "read_point": "immediately_after_box_start",
        "bin_count": COORD_BIN_COUNT,
        # Contract item 8: diagnostic only.  Captured in the one-pass GPU
        # product so later analysis never forces a recapture, but never a
        # 2-D heatmap and never a rank input.
        "role": "diagnostic_only_never_a_2d_heatmap_never_a_rank",
        "x1_logprobs": x1_distribution,
        "checks": checks,
    }
    return rows, diagnostic, competition


def _scalar_reference_sequence_sum(
    backend: CensusBackend, item: WorkItem, coord_tokens: Sequence[int]
) -> float:
    """Uncached literal reforward of the same four coordinates.

    A parity diagnostic, never the bulk path: it re-forwards the whole literal
    prefix once per coordinate and reuses no cache.
    """

    total = 0.0
    prefix = list(item.query_prefix_token_ids)
    for token_id in coord_tokens:
        view = readout(backend.full_reforward(prefix), selected_token_id=int(token_id), top_k=1)
        total += view.selected_logprob
        prefix.append(int(token_id))
    return total


def attach_competition_ranks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Context+category-local ranks over unique coordinate sequences only.

    The population is the collapsed unique physical candidates of exactly one
    ``(image, context, normalized_description)``.  Sidecars never enter, and no
    generator identity participates: a tuple two owners both generate is one
    rank mass.
    """

    if not rows:
        return {"population_size": 0}
    tuples = {tuple(int(v) for v in row["coord_token_ids"]) for row in rows}
    if len(tuples) != len(rows):
        raise ShardContractError(
            "competition population contains duplicate coordinate sequences; "
            "ranks must be computed over unique tuples only"
        )
    ordered = sorted(
        rows, key=lambda row: (-float(row["complete_box_logprob_sum"]), str(row["candidate_id"]))
    )
    best = float(ordered[0]["complete_box_logprob_sum"])
    runner_up = (
        float(ordered[1]["complete_box_logprob_sum"]) if len(ordered) > 1 else None
    )
    for position, row in enumerate(ordered):
        value = float(row["complete_box_logprob_sum"])
        row["competition"] = {
            "population": "collapsed_unique_physical_candidates_only",
            "population_scope": "image_context_normalized_description",
            "population_size": len(ordered),
            "sidecars_excluded": True,
            "generator_identity_excluded": True,
            "rank": position + 1,
            "margin_to_group_best": value - best,
            "margin_to_runner_up": (
                value - runner_up if position == 0 and runner_up is not None else None
            ),
            "group_best_candidate_id": str(ordered[0]["candidate_id"]),
        }
    return {
        "population": "collapsed_unique_physical_candidates_only",
        "population_size": len(ordered),
        "unique_coordinate_tuple_count": len(tuples),
        "group_best_candidate_id": str(ordered[0]["candidate_id"]),
        "group_best_logprob_sum": best,
    }


# ---------------------------------------------------------------------------
# Proposal channel: three separate named quantities (contract item 9)
# ---------------------------------------------------------------------------


def score_proposal_surface(
    backend: CensusBackend,
    plan: PlanBundle,
    *,
    context_id: str,
    guard: PhaseGuard,
    boundary_admission: Mapping[str, Any],
    route_admissions: Mapping[str, Mapping[str, Any]],
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Per-context proposal surface; never per owner.

    Keeps three named quantities strictly separate:

    (a) ``boundary_gate`` -- at the natural boundary, with *nothing forced*,
        the continue (``object_ref_start``) versus stop (``im_end``) mass.
        This is a gate; it is never description accessibility.
    (b) ``category_routing_event`` -- conditional on a forced
        ``object_ref_start``, the raw sequence sum over
        ``[category tokens, object_ref_end]`` for every description present in
        the image, plus its within-context rank.  Raw sum, never a token mean.
    (c) ``row_prefix_block`` -- the raw sequence sum over the full
        ``[object_ref_start, category tokens, object_ref_end, box_start]``
        block.

    No coordinate score enters any of them, and no per-owner proposal
    probability is ever emitted.

    The boundary gate and each forced category routing path are *separate*
    admission domains: the gate reads the observed prefix with nothing forced,
    while every routing path executes its own token content and its own length.
    One category's admission therefore never authorizes another's, and each
    routing entry carries the receipt that actually covers it.
    """

    guard.require_decision_phase(f"proposal scoring for {context_id}")
    context = plan.contexts[context_id]
    image_id = str(context["image_id"])
    observed_prefix = _observed_prefix_token_ids(plan, context_id)
    observed_sha = sha256_json(observed_prefix)
    if not boundary_admission.get("admitted"):
        raise ShardContractError(
            f"context {context_id!r} has no passing proposal boundary-gate admission"
        )
    if boundary_admission.get("channel") != CHANNEL_PROPOSAL_BOUNDARY_GATE:
        raise ShardContractError(
            f"context {context_id!r} boundary gate was handed a {boundary_admission.get('channel')!r} receipt"
        )
    if str(boundary_admission.get("observed_prefix_sha256")) != observed_sha:
        raise ShardContractError(
            f"context {context_id!r} was handed a boundary receipt for another observed prefix"
        )
    categories = plan.image_categories(image_id)

    # Every scored routing path must be covered by *its own* receipt.
    for category in categories:
        category_query_id = str(category["category_query_id"])
        receipt = route_admissions.get(category_query_id)
        if receipt is None:
            raise ShardContractError(
                f"context {context_id!r} has no proposal-route admission for category "
                f"{category_query_id!r}; a first-category admission may not authorize it"
            )
        expected = planner.proposal_route_admission_receipt_id(
            context_id=context_id,
            observed_prefix_sha256=observed_sha,
            category_token_ids=category["category_token_ids"],
        )
        if str(receipt.get("admission_receipt_id")) != expected:
            raise ShardContractError(
                f"context {context_id!r} category {category_query_id!r} was handed a "
                "routing admission for another path"
            )
        if receipt.get("channel") != CHANNEL_PROPOSAL_CATEGORY_ROUTE:
            raise ShardContractError(
                f"context {context_id!r} category {category_query_id!r} routing receipt is "
                f"on channel {receipt.get('channel')!r}"
            )
        if not receipt.get("admitted"):
            raise ShardContractError(
                f"context {context_id!r} category {category_query_id!r} routing path is not admitted"
            )

    bindings = row_bindings(
        plan,
        channel=CHANNEL_PROPOSAL_BOUNDARY_GATE,
        admission_receipt_id=str(boundary_admission["admission_receipt_id"]),
        observed_prefix_sha256=observed_sha,
        query_prefix_sha256=observed_sha,
        query_suffix_token_ids_sha256=EMPTY_SUFFIX_SHA256,
    )

    prefill = backend.prefill(observed_prefix)
    try:
        prefill.assert_rooted(label=f"proposal surface {context_id}")
        boundary = readout(prefill.root_logits, selected_token_id=OBJECT_REF_START, top_k=top_k)
        stop_view = readout(prefill.root_logits, selected_token_id=IM_END, top_k=1)
        continue_logprob = boundary.selected_logprob
        stop_logprob = stop_view.selected_logprob
        gate = {
            "continue_token_id": OBJECT_REF_START,
            "stop_token_id": IM_END,
            "continue_logprob": continue_logprob,
            "stop_logprob": stop_logprob,
            "continue_probability": math.exp(continue_logprob),
            "stop_probability": math.exp(stop_logprob),
            "continue_vs_stop_logprob_margin": continue_logprob - stop_logprob,
            "argmax_token_id": boundary.argmax_token_id,
            "top_alternatives": {
                "token_ids": boundary.top_token_ids,
                "logprobs": boundary.top_logprobs,
            },
            "nothing_forced": True,
            "semantics": "gate_only_never_description_accessibility",
        }

        routing: list[dict[str, Any]] = []
        for category in categories:
            category_tokens = [int(v) for v in category["category_token_ids"]]
            tokens = [*category_tokens, OBJECT_REF_END]
            per_token: list[dict[str, Any]] = []
            with prefill.branch() as branch:
                # (b)/(c) are conditional on a forced object_ref_start.
                logits = branch.step([OBJECT_REF_START])[-1]
                for token_id in tokens:
                    view = readout(logits, selected_token_id=token_id, top_k=1)
                    per_token.append(
                        {"token_id": token_id, "selected_logprob": view.selected_logprob}
                    )
                    logits = branch.step([token_id])[-1]
                box_open = readout(logits, selected_token_id=BOX_START, top_k=1)
            prefill.assert_rooted(
                label=f"proposal surface {context_id} after {category['category_query_id']}"
            )
            event_sum = math.fsum(row["selected_logprob"] for row in per_token)
            routing.append(
                {
                    "normalized_description": str(category["normalized_description"]),
                    "category_query_id": str(category["category_query_id"]),
                    "category_token_ids": category_tokens,
                    "category_token_count": len(category_tokens),
                    "suffix_shape_class": planner.suffix_shape_class(category_tokens),
                    # This entry's own covering receipt: never the boundary
                    # gate's and never another category's.
                    "channel": CHANNEL_PROPOSAL_CATEGORY_ROUTE,
                    "admission_receipt_id": str(
                        route_admissions[str(category["category_query_id"])][
                            "admission_receipt_id"
                        ]
                    ),
                    "routing_path_token_ids": proposal_routing_path_token_ids(category_tokens),
                    "routing_path_digest": planner.proposal_route_digest(category_tokens),
                    "scored_token_count": len(tokens),
                    "raw_sequence_logprob_sum": event_sum,
                    "per_token": per_token,
                    "aggregation": "raw_sequence_sum_no_token_mean",
                    "box_start_logprob": box_open.selected_logprob,
                    "row_prefix_block_raw_sequence_logprob_sum": (
                        continue_logprob + event_sum + box_open.selected_logprob
                    ),
                }
            )
    finally:
        prefill.close()

    order = sorted(
        routing,
        key=lambda row: (-float(row["raw_sequence_logprob_sum"]), row["normalized_description"]),
    )
    ranks = {row["normalized_description"]: index + 1 for index, row in enumerate(order)}
    for row in routing:
        row["within_context_rank"] = ranks[row["normalized_description"]]
        row["within_context_population"] = len(routing)

    return {
        **bindings,
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "row_kind": "census_proposal_surface",
        "context_id": context_id,
        "image_id": image_id,
        "split": str(context["split"]),
        "context_role": str(context["context_role"]),
        "loop_marking": context["loop_marking"],
        "boundary_gate": gate,
        "boundary_gate_admission_receipt_id": str(
            boundary_admission["admission_receipt_id"]
        ),
        "category_routing_event": routing,
        "category_routing_admission_receipt_ids": {
            str(row["category_query_id"]): str(row["admission_receipt_id"]) for row in routing
        },
        "boundary_gate_admission_authorizes_routing_paths": False,
        "emits_per_owner_proposal_probability": False,
        "includes_coordinate_scores": False,
    }


# ---------------------------------------------------------------------------
# Terminal generation phase: free-decode sidecars
# ---------------------------------------------------------------------------


def _greedy_decode(
    prefill: GroupPrefill,
    *,
    max_tokens: int,
    stop_token_ids: Sequence[int],
) -> tuple[list[int], list[float], str]:
    """Explicit-position greedy loop over the admitted cache seam.

    Never ``model.generate()``: every continuation step goes through
    ``HFCacheBackend.step``, which passes explicit ``position_ids`` derived
    from the prefill-time ``rope_deltas`` and therefore never mutates the
    model's shared attribute.
    """

    stop = {int(v) for v in stop_token_ids}
    tokens: list[int] = []
    logprobs: list[float] = []
    stop_reason = "max_tokens"
    with prefill.branch() as branch:
        logits = prefill.root_logits
        for _ in range(int(max_tokens)):
            view = readout(logits, selected_token_id=None, top_k=1)
            tokens.append(view.selected_token_id)
            logprobs.append(view.selected_logprob)
            if view.selected_token_id in stop:
                stop_reason = f"token_{view.selected_token_id}"
                break
            logits = branch.step([view.selected_token_id])[-1]
    return tokens, logprobs, stop_reason


def free_greedy_box_sidecar(
    backend: CensusBackend,
    plan: PlanBundle,
    item: WorkItem,
    *,
    guard: PhaseGuard,
    admission: Mapping[str, Any],
    bank_tokens: Mapping[tuple[int, ...], str],
    max_tokens: int = FREE_BOX_MAX_TOKENS,
) -> dict[str, Any]:
    """One free greedy box per ``(context, category)``; excluded from core ranks.

    Behavior, not a probability.  A free box whose coordinate tuple is
    identical to a bank candidate's *joins that candidate's provenance*; it
    never adds a second rank mass.
    """

    guard.require_generation_phase(f"free box sidecar for {item.query_group_id}")
    prefill = backend.prefill(item.query_prefix_token_ids)
    try:
        prefill.assert_rooted(label=f"free box {item.query_group_id}")
        tokens, logprobs, stop_reason = _greedy_decode(
            prefill, max_tokens=max_tokens, stop_token_ids=(BOX_END, IM_END)
        )
        prefill.assert_rooted(label=f"free box {item.query_group_id} (post-decode)")
    finally:
        prefill.close()

    coord_tokens = [
        token for token in tokens if COORD_TOKEN_START <= token <= COORD_TOKEN_END
    ]
    well_formed = (
        len(tokens) == 5
        and all(COORD_TOKEN_START <= token <= COORD_TOKEN_END for token in tokens[:4])
        and tokens[4] == BOX_END
    )
    joined = (
        bank_tokens.get(tuple(tokens[:4])) if well_formed else None
    )
    return {
        **row_bindings(
            plan,
            channel=CHANNEL_QUERY_SUFFIX,
            admission_receipt_id=item.admission_receipt_id,
            observed_prefix_sha256=item.observed_prefix_sha256,
            query_prefix_sha256=item.query_prefix_sha256,
            query_suffix_token_ids_sha256=item.query_suffix_token_ids_sha256,
        ),
        "schema_version": FREE_DECODE_SCHEMA_VERSION,
        "row_kind": "census_free_greedy_box_sidecar",
        "sidecar_id": f"free-box:{item.query_group_id}",
        "query_group_id": item.query_group_id,
        "image_id": item.image_id,
        "context_id": item.context_id,
        "normalized_description": item.normalized_description,
        "decode_mode": "greedy_explicit_position_cache",
        "uses_model_generate": False,
        "max_tokens": int(max_tokens),
        "token_ids": tokens,
        "token_count": len(tokens),
        "per_token_logprobs": logprobs,
        "stop_reason": stop_reason,
        "truncated_at_cap": stop_reason == "max_tokens",
        "coordinate_token_count": len(coord_tokens),
        "coord_bins": (
            [token - COORD_TOKEN_START for token in tokens[:4]] if well_formed else None
        ),
        "well_formed_box": well_formed,
        "malformed_reason": (
            None
            if well_formed
            else (
                "truncated_before_box_end"
                if stop_reason == "max_tokens"
                else "non_coordinate_token_in_box"
            )
        ),
        "complete_box_logprob_sum": math.fsum(logprobs[:4]) if well_formed else None,
        "joins_physical_candidate_id": joined,
        "join_semantics": (
            "joins_provenance_never_adds_rank_mass" if joined else "no_identical_bank_tuple"
        ),
        "is_sidecar": True,
        "enters_core_ranks": False,
        "is_behavior_not_probability": True,
        "generation_phase_after_decision_scoring": True,
    }


def free_next_row_sidecar(
    backend: CensusBackend,
    plan: PlanBundle,
    *,
    context_id: str,
    guard: PhaseGuard,
    admission: Mapping[str, Any],
    max_tokens: int = FREE_ROW_MAX_TOKENS,
) -> dict[str, Any]:
    """One free greedy next row per context, from the observed boundary.

    Runs until ``im_end`` or the first complete ``box_end``, capped by a frozen
    maximum row-token budget; truncation is recorded rather than hidden.
    """

    guard.require_generation_phase(f"free next row for {context_id}")
    context = plan.contexts[context_id]
    image_id = str(context["image_id"])
    observed_prefix = _observed_prefix_token_ids(plan, context_id)
    observed_sha = sha256_json(observed_prefix)

    prefill = backend.prefill(observed_prefix)
    try:
        prefill.assert_rooted(label=f"free next row {context_id}")
        tokens, logprobs, stop_reason = _greedy_decode(
            prefill, max_tokens=max_tokens, stop_token_ids=(IM_END, BOX_END)
        )
        prefill.assert_rooted(label=f"free next row {context_id} (post-decode)")
    finally:
        prefill.close()

    return {
        **row_bindings(
            plan,
            channel=CHANNEL_PROPOSAL_BOUNDARY_GATE,
            admission_receipt_id=str(admission["admission_receipt_id"]),
            observed_prefix_sha256=observed_sha,
            query_prefix_sha256=observed_sha,
            query_suffix_token_ids_sha256=EMPTY_SUFFIX_SHA256,
        ),
        "schema_version": FREE_DECODE_SCHEMA_VERSION,
        "row_kind": "census_free_next_row_sidecar",
        "sidecar_id": f"free-row:{context_id}",
        "image_id": image_id,
        "context_id": context_id,
        "decode_mode": "greedy_explicit_position_cache",
        "uses_model_generate": False,
        "max_tokens": int(max_tokens),
        "token_ids": tokens,
        "token_count": len(tokens),
        "per_token_logprobs": logprobs,
        "stop_reason": stop_reason,
        "truncated_at_cap": stop_reason == "max_tokens",
        "reached_im_end": stop_reason == f"token_{IM_END}",
        "reached_box_end": stop_reason == f"token_{BOX_END}",
        "is_sidecar": True,
        "enters_core_ranks": False,
        "is_behavior_not_probability": True,
        "generation_phase_after_decision_scoring": True,
    }


# ---------------------------------------------------------------------------
# Per-owner bank accounting
# ---------------------------------------------------------------------------


def build_owner_bank_accounting(
    plan: PlanBundle, image_id: str, *, scored_candidate_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """Propagate the frozen per-owner bank accounting into the shard receipt.

    The floor is re-asserted here: an owner whose bank is undercovered cannot
    later be given a persistent-negative disposition, only ``unresolved``.
    """

    scored = set(scored_candidate_ids)
    rows: list[dict[str, Any]] = []
    for owner in plan.image_owners(image_id):
        bank = owner.get("candidate_bank")
        if not isinstance(bank, Mapping):
            raise ShardContractError(
                f"owner {owner['gt_owner_id']!r} carries no candidate-bank accounting"
            )
        missing = [key for key in REQUIRED_OWNER_BANK_KEYS if key not in bank]
        if missing:
            raise ShardContractError(
                f"owner {owner['gt_owner_id']!r} bank accounting is missing {missing!r}"
            )
        if int(bank["logical_role_count"]) != planner.LOGICAL_ROLE_COUNT:
            raise ShardContractError(
                f"owner {owner['gt_owner_id']!r} does not account for exactly "
                f"{planner.LOGICAL_ROLE_COUNT} logical roles"
            )
        strict = bank["strict_assignment_coverage"]
        if not isinstance(strict, Mapping):
            raise ShardContractError(
                f"owner {owner['gt_owner_id']!r} strict-assignment coverage is not a mapping"
            )
        strict_missing = [
            key for key in REQUIRED_OWNER_STRICT_ASSIGNMENT_KEYS if key not in strict
        ]
        if strict_missing:
            raise ShardContractError(
                f"owner {owner['gt_owner_id']!r} strict-assignment coverage is missing "
                f"{strict_missing!r}"
            )
        undercovered = bool(bank["undercovered"])
        if undercovered == bool(bank["disposition_eligible"]):
            raise ShardContractError(
                f"owner {owner['gt_owner_id']!r} disposition eligibility contradicts its "
                "undercoverage flag"
            )
        if undercovered and str(bank["disposition_floor"]) != UNDERCOVERED_DISPOSITION_FLOOR:
            raise ShardContractError(
                f"owner {owner['gt_owner_id']!r} is undercovered but does not carry the "
                "unresolved-only disposition floor"
            )
        reached = [str(v) for v in bank["physical_candidate_ids"]]
        rows.append(
            {
                "gt_owner_id": str(owner["gt_owner_id"]),
                "image_id": image_id,
                "normalized_description": str(owner["normalized_description"]),
                "logical_role_count": int(bank["logical_role_count"]),
                "distinct_physical_candidates_reached": int(
                    bank["distinct_physical_candidates_reached"]
                ),
                # Adequacy is the gate; strict assignment is a separate lower
                # bound and must never be read as the gate.
                "generator_local_bank_adequacy": dict(bank["generator_local_bank_adequacy"]),
                "uniquely_assigned_candidate_count": int(
                    strict["uniquely_assigned_candidate_count"]
                ),
                "roles_lost_to_other_owner_assignment": int(
                    strict["roles_lost_to_other_owner_assignment"]
                ),
                "roles_lost_to_ambiguous_assignment": int(
                    strict["roles_lost_to_ambiguous_assignment"]
                ),
                "roles_lost_to_unmatched_assignment": int(
                    strict["roles_lost_to_unmatched_assignment"]
                ),
                "strict_assignment_coverage_role": (
                    "separate_lower_bound_view_never_the_adequacy_gate"
                ),
                "bank_coverage_status": str(bank["bank_coverage_status"]),
                "disposition_eligible": bool(bank["disposition_eligible"]),
                "undercovered": undercovered,
                "disposition_floor": str(bank["disposition_floor"]),
                "physical_candidates_scored_in_shard": sum(
                    1 for candidate_id in reached if candidate_id in scored
                ),
                "physical_candidates_reached_but_unscored": sorted(
                    candidate_id for candidate_id in reached if candidate_id not in scored
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# CPU contract validation (no model load)
# ---------------------------------------------------------------------------


def validate_shard_contract(plan: PlanBundle, image_id: str) -> dict[str, Any]:
    """Full CPU contract check over every planned work item in one shard."""

    if image_id not in plan.shards:
        raise ShardContractError(f"image {image_id!r} is not a planned shard")
    groups = plan.image_query_groups(image_id)
    admitted = [row for row in groups if row["status"] == "admitted"]
    blocked = [row for row in groups if row["status"] != "admitted"]

    candidate_ids: set[str] = set()
    contexts_seen: set[str] = set()
    shape_classes: set[str] = set()
    context_shape_pairs: set[tuple[str, str]] = set()
    admission_ids: set[str] = set()
    boundary_admission_ids: set[str] = set()
    route_admission_ids: set[str] = set()
    for group in admitted:
        item = resolve_work_item(plan, str(group["query_group_id"]))
        validate_singleton_group([item])
        contexts_seen.add(item.context_id)
        shape_classes.add(item.suffix_shape_class)
        context_shape_pairs.add((item.context_id, item.suffix_shape_class))
        admission_ids.add(item.admission_receipt_id)
        boundary_admission_ids.add(item.proposal_boundary_gate_admission_receipt_id)
        route_admission_ids.add(item.proposal_route_admission_receipt_id)
        candidate_ids.update(str(row["candidate_id"]) for row in item.candidates)

    if len(admission_ids) != len(admitted):
        raise ShardContractError(
            "two admitted query groups share a box-channel admission receipt; "
            "admission must be keyed on the exact query prefix"
        )
    build_owner_bank_accounting(plan, image_id, scored_candidate_ids=sorted(candidate_ids))

    shard = plan.shards[image_id]
    context_ids = {
        str(row["context_id"]) for row in plan.contexts.values() if str(row["image_id"]) == image_id
    }
    return {
        "schema_version": CONTRACT_CHECK_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "image_id": image_id,
        "split": str(shard["split"]),
        "plan_receipt_content_sha256": plan.receipt_content_sha256,
        "capture_rules_sha256": plan.capture_rules_sha256,
        "planned_query_group_count": len(groups),
        "admitted_query_group_count": len(admitted),
        "blocked_query_group_count": len(blocked),
        "context_count": len(context_ids),
        "context_count_with_admitted_group": len(contexts_seen),
        "distinct_candidate_count": len(candidate_ids),
        "distinct_suffix_shape_classes": sorted(shape_classes),
        "context_shape_class_pair_count": len(context_shape_pairs),
        "box_channel_admission_receipt_count": len(admission_ids),
        "proposal_boundary_gate_admission_receipt_count": len(boundary_admission_ids),
        "proposal_route_admission_receipt_count": len(route_admission_ids),
        "estimated_work_units": int(shard["estimated_work_units"]),
        "singleton_group_invariant": "enforced",
        "canonical_query_suffix": "verified_for_every_admitted_group",
        "cross_owner_tuple_collapse": "verified_for_every_admitted_group",
        "reads_any_score_artifact": False,
        "loads_any_model": False,
        "status": "contract_ok",
    }


# ---------------------------------------------------------------------------
# Deterministic fake backend (no GPU, same code path)
# ---------------------------------------------------------------------------


class FakeCacheBackend:
    """Deterministic stand-in for ``HFCacheBackend``, tensor-for-tensor.

    Logits are a pure function of the literal token sequence, so the cached and
    uncached paths agree exactly and the production parity gate is meaningful
    rather than vacuous.
    """

    def __init__(self, owner: "FakeCensusBackend", token_ids: Sequence[int]) -> None:
        self._owner = owner
        self._tokens = [int(v) for v in token_ids]

    @property
    def cache_length(self) -> int:
        return len(self._tokens)

    @property
    def layer_count(self) -> int:
        return self._owner.layer_count

    def crop(self, length: int) -> None:
        if int(length) > len(self._tokens):
            raise ShardContractError("cannot crop a cache beyond its length")
        del self._tokens[int(length) :]

    def step(self, token_ids: Sequence[int]) -> Any:
        import torch

        rows = []
        for token_id in token_ids:
            self._tokens.append(int(token_id))
            rows.append(self._owner.logits_for(self._tokens))
        return torch.stack(rows, dim=0)


class _FakeBatchedBranch:
    """Deterministic batched branch: each lane keeps its own literal history.

    Logits stay a pure function of one lane's token sequence, so a lane that
    ever saw another lane's token would produce visibly different numbers.
    """

    def __init__(self, owner: "FakeCensusBackend", root_token_ids: Sequence[int], width: int) -> None:
        self._owner = owner
        self._lanes = [[int(v) for v in root_token_ids] for _ in range(int(width))]

    @property
    def width(self) -> int:
        return len(self._lanes)

    def step(self, token_ids_per_lane: Sequence[Sequence[int]]) -> Any:
        import torch

        if len(token_ids_per_lane) != self.width:
            raise ShardContractError(
                f"batched step received {len(token_ids_per_lane)} lanes, expected {self.width}"
            )
        rows = []
        for lane, token_ids in zip(self._lanes, token_ids_per_lane, strict=True):
            per_step = []
            for token_id in token_ids:
                lane.append(int(token_id))
                per_step.append(self._owner.logits_for(lane))
            rows.append(torch.stack(per_step, dim=0))
        return torch.stack(rows, dim=0)


class FakeCensusBackend(_PrefillTracker):
    """Torch-backed deterministic backend exercising the identical code path."""

    def __init__(
        self,
        *,
        vocab_size: int = COORD_TOKEN_END + 11,
        seed: str = "fake",
        layer_count: int | None = None,
    ) -> None:
        super().__init__()
        self._vocab_size = int(vocab_size)
        self._seed = seed
        self._layer_count = (
            int(layer_count)
            if layer_count is not None
            else int(_basin().EXPECTED_LIVE_QWEN_DECODER_LAYER_COUNT)
        )
        self.prefill_calls: list[list[int]] = []
        self.reforward_calls: list[list[int]] = []
        self.batched_branch_widths: list[int] = []

    @property
    def layer_count(self) -> int:
        return self._layer_count

    @property
    def expected_layer_count(self) -> int | None:
        return self._layer_count

    def logits_for(self, token_ids: Sequence[int]) -> Any:
        import torch

        digest = hashlib.sha256(
            (self._seed + ":" + ",".join(str(int(v)) for v in token_ids)).encode("utf-8")
        ).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big"))
        return torch.randn(self._vocab_size, generator=generator, dtype=torch.float32)

    def prefill(self, token_ids: Sequence[int]) -> GroupPrefill:
        tokens = [int(v) for v in token_ids]
        self.prefill_calls.append(list(tokens))
        return self._track(
            GroupPrefill(
                owner=self,
                cache_backend=FakeCacheBackend(self, tokens),
                root_logits=self.logits_for(tokens),
                prefill_length=len(tokens),
                root_token_ids=list(tokens),
            )
        )

    def full_reforward(self, token_ids: Sequence[int]) -> Any:
        tokens = [int(v) for v in token_ids]
        self.reforward_calls.append(list(tokens))
        return self.logits_for(tokens)

    @contextlib.contextmanager
    def batched_branch(self, prefill: GroupPrefill, width: int):
        self.batched_branch_widths.append(int(width))
        yield _FakeBatchedBranch(self, prefill.root_token_ids, int(width))

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "backend": "fake",
            "seed": self._seed,
            "vocab_size": self._vocab_size,
            "layer_count": self._layer_count,
            "is_real_model": False,
            "usable_as_evidence": False,
        }


# ---------------------------------------------------------------------------
# Real HF backend
# ---------------------------------------------------------------------------


class BatchedCacheBranch:
    """``width`` independent continuation lanes over one copied, expanded root.

    Built from a *copy* of the group's admitted root cache -- via the
    documented ``to_legacy_cache``/``from_legacy_cache`` round trip and the
    supported ``DynamicCache.batch_repeat_interleave`` -- so every lane starts
    byte-identical to the single-branch root and the group root itself is never
    mutated.  Positions come from the production
    :func:`continuation_position_ids` formula, expanded across lanes: all lanes
    sit at the same depth with the same prompt, so their positions are
    identical by construction.

    Batching is a throughput device only.  Causal attention keeps lanes
    independent, so no lane can see another lane's coordinate tokens; the
    contamination test asserts this rather than assuming it.  An out-of-memory
    error is never caught here: it must fail the shard loudly rather than
    silently degrade to a different execution path.
    """

    def __init__(self, *, model: Any, cache: Any, rope_deltas: Any, width: int) -> None:
        self._model = model
        self.cache = cache
        self._rope_deltas = rope_deltas
        self._width = int(width)

    @property
    def width(self) -> int:
        return self._width

    @property
    def cache_length(self) -> int:
        return int(self.cache.get_seq_length())

    def step(self, token_ids_per_lane: Sequence[Sequence[int]]) -> Any:
        import torch

        basin = _basin()
        lanes = [[int(v) for v in row] for row in token_ids_per_lane]
        if len(lanes) != self._width:
            raise ShardContractError(
                f"batched step received {len(lanes)} lanes, expected {self._width}"
            )
        widths = {len(row) for row in lanes}
        if len(widths) != 1:
            raise ShardContractError(
                "every lane of a batched step must advance the same number of tokens"
            )
        new_token_count = widths.pop()
        device = next(self._model.parameters()).device
        ids = torch.tensor(lanes, dtype=torch.long, device=device)
        context_length_before = self.cache_length
        # Same formula as the single-lane path, replicated across lanes.
        position_ids = (
            basin.continuation_position_ids(
                context_length_before_step=context_length_before,
                rope_deltas=self._rope_deltas,
                new_token_count=new_token_count,
            )
            .expand(3, self._width, new_token_count)
            .contiguous()
            .to(device=device)
        )
        attention_mask = torch.ones(
            (self._width, context_length_before + new_token_count),
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode():
            outputs = self._model(
                input_ids=ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self.cache,
                use_cache=True,
                return_dict=True,
                logits_to_keep=new_token_count,
            )
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise RuntimeError("batched forward did not return rank-3 logits")
        if int(logits.shape[0]) != self._width:
            raise ShardContractError(
                f"batched forward returned {int(logits.shape[0])} lanes, expected {self._width}"
            )
        self.cache = outputs.past_key_values
        return logits.detach().to(device="cpu", dtype=torch.float32)

    def release(self) -> None:
        self.cache = None


def _cache_layer_shapes(cache: Any) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Per-layer ``(keys, values)`` shapes, so root drift is detectable."""

    return [
        (tuple(int(v) for v in layer.keys.shape), tuple(int(v) for v in layer.values.shape))
        for layer in getattr(cache, "layers", [])
    ]


def matmul_precision_state() -> dict[str, Any]:
    """Record the float32 matmul precision that produced these numbers.

    TF32 changes float32 matmul results, so a capture that does not record it
    cannot be compared against one that used a different setting.
    """

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is required for any capture
        return {"torch_available": False}
    backends = torch.backends
    return {
        "torch_available": True,
        "torch_version": str(torch.__version__),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cuda_matmul_allow_tf32": bool(getattr(backends.cuda.matmul, "allow_tf32", False)),
        "cudnn_allow_tf32": bool(getattr(backends.cudnn, "allow_tf32", False)),
    }


def expand_cache_for_lanes(cache: Any, width: int) -> Any:
    """An independent, ``width``-wide copy of one batch-1 cache.

    Uses only documented cache APIs and clones every tensor, so the source
    cache is provably untouched and the lanes cannot alias one another.
    """

    from transformers.cache_utils import DynamicCache

    legacy = cache.to_legacy_cache()
    copied = DynamicCache.from_legacy_cache(
        tuple((keys.clone(), values.clone()) for keys, values in legacy)
    )
    if int(width) > 1:
        copied.batch_repeat_interleave(int(width))
    # A stride-0 / shared-storage expansion would make every lane alias the
    # same key-value memory, so one lane's appended coordinate token would be
    # visible to the others.  Refuse anything that is not real, per-lane
    # storage rather than trusting the vendor implementation.
    for layer_index, layer in enumerate(copied.layers):
        for name, tensor in (("keys", layer.keys), ("values", layer.values)):
            if int(tensor.shape[0]) != int(width):
                raise ShardContractError(
                    f"expanded cache layer {layer_index} {name} has batch "
                    f"{int(tensor.shape[0])}, expected {int(width)}"
                )
            if 0 in tuple(tensor.stride()):
                raise ShardContractError(
                    f"expanded cache layer {layer_index} {name} is a stride-0 view; "
                    "lanes would share key-value storage and contaminate each other"
                )
            if tensor.untyped_storage().nbytes() < tensor.numel() * tensor.element_size():
                raise ShardContractError(
                    f"expanded cache layer {layer_index} {name} does not own storage for "
                    "every lane"
                )
    return copied


def prefill_prompt_only(model: Any, *, native_prompt_inputs: Mapping[str, Any]) -> Any:
    """Prefill the executed prompt alone, on exactly the production seams.

    ``prefill_context`` requires at least one generated-history token, but the
    census's *root* context is the prompt itself: its observed prefix has no
    native rows.  This mirrors ``prefill_context`` token-for-token minus the
    history concatenation -- the same materialized native inputs, the same
    ``get_rope_index``-derived explicit ``position_ids``, a fresh
    ``DynamicCache``, and the same ``HFCacheBackend``/``PrefillResult`` -- so
    the root context is not a second, differently-behaved code path.
    """

    import torch
    from transformers.cache_utils import DynamicCache

    basin = _basin()
    prompt_input_ids = native_prompt_inputs.get("input_ids")
    if (
        not isinstance(prompt_input_ids, torch.Tensor)
        or prompt_input_ids.ndim != 2
        or int(prompt_input_ids.shape[0]) != 1
    ):
        raise ShardContractError(
            "native_prompt_inputs.input_ids must be a materialized [1, prompt_len] tensor"
        )
    ids = prompt_input_ids
    attention_mask = torch.ones_like(ids)
    image_grid_thw = basin._require_native_image_grid_thw(  # noqa: SLF001
        native_prompt_inputs, context="prefill_prompt_only"
    )
    position_ids, rope_deltas = basin.derive_prefill_position_state(
        model, input_ids=ids, attention_mask=attention_mask, image_grid_thw=image_grid_thw
    )
    cache = DynamicCache()
    vision_kwargs = basin._native_vision_kwargs(native_prompt_inputs)  # noqa: SLF001
    with torch.inference_mode():
        outputs = model(
            input_ids=ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
            **vision_kwargs,
        )
    logits = getattr(outputs, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ShardContractError("prompt-only prefill did not return rank-3 logits")
    backend = basin.HFCacheBackend(
        model=model, cache=outputs.past_key_values, rope_deltas=rope_deltas
    )
    return basin.PrefillResult(
        backend=backend,
        prefill_logits=logits[0, -1, :].detach().to(device="cpu", dtype=torch.float32),
        prefill_length=int(ids.shape[1]),
    )


class HFCensusBackend(_PrefillTracker):
    """The real production seam.

    Holds one long-lived HF model/session plus this image's materialized native
    inputs (true ``pixel_values``/``image_grid_thw``), and builds a fresh
    ``DynamicCache`` per prefill through ``prefill_context``.  Both the cached
    and uncached paths pass explicit ``position_ids``; this class never calls
    ``model.generate()`` and must not be shared across threads.
    """

    def __init__(
        self,
        *,
        model: Any,
        native_prompt_inputs: Mapping[str, Any],
        prompt_token_ids: Sequence[int],
        identity: Mapping[str, Any],
        expected_layer_count: int | None = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._native = native_prompt_inputs
        self._prompt_token_ids = [int(v) for v in prompt_token_ids]
        self._identity = dict(identity)
        self._expected_layer_count = (
            int(_basin().EXPECTED_LIVE_QWEN_DECODER_LAYER_COUNT)
            if expected_layer_count is None
            else expected_layer_count
        )

    @property
    def expected_layer_count(self) -> int | None:
        return self._expected_layer_count

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._identity

    def _assert_prompt_prefix(self, tokens: Sequence[int]) -> list[int]:
        """Element-wise, never count-only: the literal prompt must match exactly.

        A prefix *equal* to the prompt is legitimate: the root context's
        observed prefix has no generated rows at all.
        """

        literal = [int(v) for v in tokens]
        prompt = self._prompt_token_ids
        if len(literal) < len(prompt):
            raise ShardContractError(
                "a census prefix must contain at least the executed production prompt"
            )
        if literal[: len(prompt)] != prompt:
            raise ShardContractError(
                "census prefix does not begin with this image's executed prompt token ids; "
                "refusing to prefill a divergent multimodal prefix"
            )
        return literal

    def prefill(self, token_ids: Sequence[int]) -> GroupPrefill:
        basin = _basin()
        literal = self._assert_prompt_prefix(token_ids)
        history = literal[len(self._prompt_token_ids) :]
        if history:
            result = basin.prefill_context(
                self._model,
                native_prompt_inputs=self._native,
                generated_history_token_ids=history,
            )
        else:
            # The root context is the executed prompt itself; ``prefill_context``
            # requires at least one generated-history token, so the prompt-only
            # root gets its own prefill on exactly the same seams.
            result = prefill_prompt_only(self._model, native_prompt_inputs=self._native)
        if int(result.prefill_length) != len(literal):
            raise ShardContractError(
                f"prefill consumed {int(result.prefill_length)} tokens but the literal prefix "
                f"has {len(literal)}"
            )
        return self._track(
            GroupPrefill(
                owner=self,
                cache_backend=result.backend,
                root_logits=result.prefill_logits,
                prefill_length=int(result.prefill_length),
                root_token_ids=literal,
            )
        )

    def release(self, prefill: GroupPrefill) -> None:
        backend = prefill.cache_backend
        super().release(prefill)
        # Drop the cache tensors before the next group's prefill allocates.
        if backend is not None:
            backend.cache = None

    @contextlib.contextmanager
    def batched_branch(self, prefill: GroupPrefill, width: int):
        root = prefill.cache_backend
        if root is None:
            raise ShardContractError("cannot open a batched branch on a released prefill")
        entry_length = int(root.cache_length)
        entry_shapes = _cache_layer_shapes(root.cache)
        # The lanes run on a *separate*, group-scoped cache object.  The root is
        # never expanded in place, so batch-1 root semantics are structurally
        # preserved rather than restored after the fact.
        branch = BatchedCacheBranch(
            model=self._model,
            cache=expand_cache_for_lanes(root.cache, int(width)),
            rope_deltas=root.rope_deltas,
            width=int(width),
        )
        try:
            yield branch
        finally:
            branch.release()
        if int(root.cache_length) != entry_length:
            raise ShardContractError(
                "a batched branch changed the group root cache length; candidates would "
                "be contaminated"
            )
        if _cache_layer_shapes(root.cache) != entry_shapes:
            raise ShardContractError(
                "a batched branch changed the group root cache batch/sequence shape; "
                "the root is no longer the admitted batch-1 root"
            )

    def full_reforward(self, token_ids: Sequence[int]) -> Any:
        """Uncached reforward of the **complete** literal prefix.

        ``_build_full_reforward_closure`` consumes the whole sequence -- prompt
        tokens included -- because it re-passes the image's vision kwargs and
        re-derives positions from ``get_rope_index``.  Stripping the prompt here
        would silently reforward a different sequence than the cache path.
        """

        basin = _basin()
        literal = self._assert_prompt_prefix(token_ids)
        closure = basin._build_full_reforward_closure(  # noqa: SLF001
            self._model, native_prompt_inputs=self._native
        )
        return closure(literal)


def assert_repetition_penalty_stratum(value: Any, *, label: str) -> float:
    """Fail closed unless the effective stratum is this unit's rp1.0.

    The unit scores one stratum.  An rp1.10 config would pollute the session's
    runtime identity and make the capture unjoinable to the frozen native
    rollout stratum, even though the census itself reads raw logits.
    """

    try:
        stratum = float(value)
    except (TypeError, ValueError) as exc:
        raise ShardContractError(
            f"{label}: repetition penalty {value!r} is not numeric"
        ) from exc
    expected = float(planner.NATIVE_REPETITION_PENALTY_STRATUM)
    if stratum != expected:
        raise ShardContractError(
            f"{label}: effective repetition-penalty stratum is {stratum}, but this unit "
            f"scores only {expected}; use the rp1p0 infer config"
        )
    return stratum


@dataclass(frozen=True)
class HFSessionSpec:
    """Everything needed to open one image's real session, resolved on CPU."""

    image_id: str
    infer_config: Path
    launch: Any
    request: Any
    image_grid_thw: tuple[int, int, int]
    planned_prompt_token_ids: list[int]
    planned_executed_media_sha256: str
    repetition_penalty_stratum: float


def build_hf_session_spec(
    plan: PlanBundle, image_id: str, *, infer_config: Path
) -> HFSessionSpec:
    """Resolve the production request for one image without opening a model.

    Everything here is the existing, verified production path: resolved infer
    config -> ``assemble_frontend`` -> ``plan_image_batch`` ->
    ``build_prompt_record``.  This module never re-implements prompt
    construction or vision preprocessing.  Kept separate from
    :func:`open_hf_backend` so the resolution step is independently testable.
    """

    from src.config.fingerprint import sha256_json as config_sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import DecodeRequest, GenerationPolicy
    from src.inference.image_plan import plan_image_batch
    from src.inference.pipeline import _processor_config, _template_config
    from src.inference.prompt import build_prompt_record
    from src.inference.runtime import assemble_frontend

    if image_id not in plan.images:
        raise ShardContractError(f"image {image_id!r} is not in the plan's image registry")

    resolved = load_infer_config(Path(infer_config).expanduser().resolve(strict=True))
    config = resolved.config
    if config.backend.type != "hf":
        raise ShardContractError("the census scorer requires backend.type: hf")
    assert_repetition_penalty_stratum(
        config.generation.repetition_penalty, label=f"infer config {infer_config}"
    )

    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=config_sha256_json(config.generation.model_dump(mode="json")),
    )
    raw_rows = load_raw_examples(Path(config.data.input_jsonl).expanduser().resolve(strict=True))
    raw = None
    for row in raw_rows:
        source_metadata = row.metadata.get("source")
        if isinstance(source_metadata, Mapping) and str(
            source_metadata.get("image_id")
        ) == image_id:
            raw = row
            break
    if raw is None:
        raise ShardContractError(f"image {image_id!r} is absent from the production source JSONL")

    image_plan = plan_image_batch(
        [raw],
        components=frontend.qwen,
        processor_config=_processor_config(config),
        row_indices=[0],
    ).rows[0]
    prompt_record = build_prompt_record(
        raw,
        _template_config(config),
        processor=frontend.qwen.processor,
        row_index=0,
        merged_visual_tokens=image_plan.merged_visual_tokens,
        object_order_seed=config.template.object_order_seed,
    )
    grid = tuple(int(v) for v in image_plan.expected_image_grid_thw)
    request = DecodeRequest(
        request_id=f"owner-accessibility-census:{image_id}",
        chat_text=prompt_record.chat_text,
        input_prompt_token_ids=tuple(prompt_record.input_prompt_token_ids),
        expected_executed_prompt_token_ids=tuple(prompt_record.expected_executed_prompt_token_ids),
        image_path=image_plan.image_path,
        declared_image_width=image_plan.declared_width,
        declared_image_height=image_plan.declared_height,
        decoded_image_width=image_plan.decoded_width,
        decoded_image_height=image_plan.decoded_height,
        image_sha256=image_plan.image_content_sha256,
        expected_image_grid_thw=(grid[0], grid[1], grid[2]),
        logical_transform_id=image_plan.logical_transform_id,
        generation_policy=GenerationPolicy(
            max_new_tokens=1,
            repetition_penalty=planner.NATIVE_REPETITION_PENALTY_STRATUM,
            temperature=0.0,
            top_p=1.0,
            include_raw_model_logprob=True,
        ),
    )
    return HFSessionSpec(
        image_id=image_id,
        infer_config=Path(infer_config),
        launch=frontend.launch,
        request=request,
        image_grid_thw=(grid[0], grid[1], grid[2]),
        planned_prompt_token_ids=[int(v) for v in plan.images[image_id]["prompt_token_ids"]],
        planned_executed_media_sha256=str(plan.images[image_id]["executed_media_sha256"]),
        repetition_penalty_stratum=float(planner.NATIVE_REPETITION_PENALTY_STRATUM),
    )


@contextlib.contextmanager
def open_hf_backend(spec: HFSessionSpec, *, session_opener: Any = None):
    """Open exactly one immutable model/session for one image shard.

    ``session_opener`` defaults to the production ``open_backend_session``; it
    is an explicit parameter so the wiring is monkeypatch-testable without a
    GPU.  The session is opened once, verified against the plan's frozen prompt
    and media digests, and reused for every group of this image.
    """

    from src.inference.hf_backend import HFBackendSession

    if session_opener is None:
        from src.inference.backend import open_backend_session as session_opener  # noqa: PLW0127

    # The effective *request* stratum, re-asserted independently of the config
    # gate in :func:`build_hf_session_spec`.
    assert_repetition_penalty_stratum(
        spec.request.generation_policy.repetition_penalty,
        label=f"decode request for image {spec.image_id}",
    )
    assert_repetition_penalty_stratum(
        spec.repetition_penalty_stratum, label=f"session spec for image {spec.image_id}"
    )

    with session_opener(spec.launch) as opened:
        if not isinstance(opened, HFBackendSession):
            raise ShardContractError("HF launch opened an unexpected backend session")
        model = opened._model  # noqa: SLF001
        if model is None:
            raise ShardContractError("HF backend session did not expose its opened model")
        model.eval()
        (
            native_inputs,
            executed_prompt_ids,
            _observed_grids,
            executed_media_sha256,
        ) = opened._materialize_native_inputs((spec.request,))  # noqa: SLF001

        executed = [int(v) for v in executed_prompt_ids[0]]
        if executed != spec.planned_prompt_token_ids:
            raise ShardContractError(
                f"image {spec.image_id!r} materialized prompt tokens differ from the plan's "
                "frozen native prompt; refusing to score a divergent prompt"
            )
        observed_media = (
            str(executed_media_sha256[0])
            if isinstance(executed_media_sha256, (list, tuple))
            else str(executed_media_sha256)
        )
        if (
            spec.planned_executed_media_sha256
            and observed_media
            and spec.planned_executed_media_sha256 != observed_media
        ):
            raise ShardContractError(
                f"image {spec.image_id!r} executed media digest differs from the frozen rollout"
            )

        receipt = opened.receipt.to_artifact_dict()
        identity = {
            "backend": "hf",
            "infer_config": str(spec.infer_config),
            "model_identity": receipt.get("model_identity"),
            "tokenizer_identity": receipt.get("tokenizer_identity"),
            "adapter_identity": receipt.get("adapter_identity"),
            "executed_media_sha256": observed_media,
            "executed_prompt_token_count": len(executed),
            "image_grid_thw": list(spec.image_grid_thw),
            "repetition_penalty_stratum": float(spec.repetition_penalty_stratum),
            "session_scope": "image_shard",
            "uses_model_generate": False,
            "is_real_model": True,
            "usable_as_evidence": True,
        }
        yield HFCensusBackend(
            model=model,
            native_prompt_inputs=native_inputs,
            prompt_token_ids=executed,
            identity=identity,
        )


# ---------------------------------------------------------------------------
# Shard execution
# ---------------------------------------------------------------------------


@dataclass
class ShardResult:
    receipt: dict[str, Any]
    scores: list[dict[str, Any]] = field(default_factory=list)
    x1: list[dict[str, Any]] = field(default_factory=list)
    proposals: list[dict[str, Any]] = field(default_factory=list)
    free_decodes: list[dict[str, Any]] = field(default_factory=list)


def run_shard(
    plan: PlanBundle,
    *,
    image_id: str,
    backend: CensusBackend,
    output_dir: Path | None,
    top_k: int = DEFAULT_TOP_K,
    query_group_ids: Sequence[str] | None = None,
    max_query_groups: int | None = None,
    candidate_batch_size: int = DEFAULT_CANDIDATE_BATCH_SIZE,
    capture_behavior_sidecars: bool = True,
) -> ShardResult:
    """Score one image shard: decision phase first, then the behavior phase.

    Nothing is written until every phase has completed, so a failed shard can
    never leave a cached primary row behind.

    ``capture_behavior_sidecars`` is ``True`` for this census and every existing
    caller.  A score-only successor unit that reuses this scorer -- and needs no
    free-decode behavior record -- passes ``False``; the decision phases are
    untouched, the terminal generation phase is simply not opened, and the
    receipt records that the sidecars were intentionally disabled rather than
    lost.
    """

    if image_id not in plan.shards:
        raise ShardContractError(f"image {image_id!r} is not a planned shard")
    rules = plan.capture_rules
    admission_rules = rules["admission"]
    scalar_rules = admission_rules["scalar_reference"]

    planned = [row for row in plan.image_query_groups(image_id) if row["status"] == "admitted"]
    groups = list(planned)
    if query_group_ids is not None:
        wanted = {str(v) for v in query_group_ids}
        unknown = wanted - {str(row["query_group_id"]) for row in groups}
        if unknown:
            raise ShardContractError(
                f"requested query groups are not admitted in this shard: {sorted(unknown)!r}"
            )
        groups = [row for row in groups if str(row["query_group_id"]) in wanted]
    if max_query_groups is not None:
        groups = groups[: int(max_query_groups)]
    if not groups:
        raise ShardContractError(f"image {image_id!r} has no admitted query group to score")
    complete = len(groups) == len(planned)

    items = [resolve_work_item(plan, str(row["query_group_id"])) for row in groups]
    bank_tokens = {
        tuple(int(v) for v in candidate["coord_token_ids"]): str(candidate["candidate_id"])
        for candidate in plan.candidates.values()
        if str(candidate["image_id"]) == image_id
    }

    guard = PhaseGuard()
    started = time.time()
    score_rows: list[dict[str, Any]] = []
    x1_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    free_rows: list[dict[str, Any]] = []
    admissions: dict[str, dict[str, Any]] = {}
    competitions: list[dict[str, Any]] = []

    # ---- Phase 0: one target-blind repeat-noise receipt for this image.
    epsilon_probe = items[0]
    epsilon_receipt = run_image_epsilon_receipt(
        backend,
        probe_token_ids=epsilon_probe.query_prefix_token_ids,
        probe_label=f"first_admitted_query_prefix:{epsilon_probe.query_group_id}",
        repeat_count=int(scalar_rules["repeat_count"]),
        guard=guard,
    )
    if not epsilon_receipt["finite"] or not epsilon_receipt["argmax_stable"]:
        raise ShardContractError(
            "the image repeat-noise receipt is not finite/argmax-stable; refusing to score"
        )

    if int(candidate_batch_size) < 1:
        raise ShardContractError("candidate batch size must be at least one")

    # ---- Phase 0b: if candidates are batched, prove on this live model that
    # the batched lanes reproduce the single-branch numbers at the real lane
    # width, before any decision-bearing row is written.
    batched_path_parity: dict[str, Any] | None = None
    if int(candidate_batch_size) > 1:
        batched_path_parity = run_batched_path_parity(
            backend, items[0], candidate_batch_size=int(candidate_batch_size)
        )
        if not batched_path_parity["admitted"]:
            raise ShardContractError(
                "the batched candidate path does not reproduce the single-branch scores "
                f"within {batched_path_parity['max_abs_diff_bound']} "
                f"(observed {batched_path_parity['max_abs_diff_observed']}); refusing to "
                "capture -- there is no silent fallback to the single-branch path"
            )

    # ---- Phase 1a: box channel, one exact-prefix admission per group.
    for item in items:
        validate_singleton_group([item])
        admission = run_box_channel_admission(backend, item, guard=guard)
        if admission["admission_receipt_id"] in admissions:
            raise ShardContractError(
                f"duplicate admission receipt {admission['admission_receipt_id']!r}; "
                "two groups resolved to the same exact prefix identity"
            )
        admissions[admission["admission_receipt_id"]] = admission
        if not admission["admitted"]:
            raise ShardContractError(
                f"query group {item.query_group_id!r} failed its own exact-prefix cache "
                "parity admission; bulk scoring in this group is refused"
            )
        rows, diagnostic, competition = score_query_group(
            backend,
            plan,
            item,
            guard=guard,
            admission=admission,
            top_k=top_k,
            scalar_modulus=int(scalar_rules["modulus"]),
            capture_full_x1=True,
            candidate_batch_size=int(candidate_batch_size),
        )
        score_rows.extend(rows)
        x1_rows.append(diagnostic)
        competitions.append({"query_group_id": item.query_group_id, **competition})
        if backend.live_prefill_count:
            raise ShardContractError(
                f"query group {item.query_group_id!r} left a cache backend live"
            )

    shape_coverage = assert_suffix_shape_coverage(admissions, items)

    # ---- Phase 1b: proposal channels.  The boundary gate is admitted on the
    # observed prefix; every forced category routing path is admitted on its
    # own exact path, so no category inherits another's admission.
    boundary_admissions: dict[str, dict[str, Any]] = {}
    route_admissions: dict[str, dict[str, dict[str, Any]]] = {}
    for context_id in sorted({item.context_id for item in items}):
        boundary = run_proposal_boundary_admission(
            backend, plan, context_id=context_id, guard=guard
        )
        if not boundary["admitted"]:
            raise ShardContractError(
                f"context {context_id!r} failed its proposal boundary-gate admission"
            )
        boundary_admissions[context_id] = boundary
        admissions[boundary["admission_receipt_id"]] = boundary

        per_category: dict[str, dict[str, Any]] = {}
        for category in plan.image_categories(image_id):
            route = run_proposal_path_admission(
                backend, plan, context_id=context_id, category=category, guard=guard
            )
            if not route["admitted"]:
                raise ShardContractError(
                    f"context {context_id!r} category {category['category_query_id']!r} "
                    "failed its own proposal-route admission"
                )
            if route["admission_receipt_id"] in admissions:
                raise ShardContractError(
                    f"duplicate proposal-route receipt {route['admission_receipt_id']!r}"
                )
            per_category[str(category["category_query_id"])] = route
            admissions[route["admission_receipt_id"]] = route
        route_admissions[context_id] = per_category

        proposal_rows.append(
            score_proposal_surface(
                backend,
                plan,
                context_id=context_id,
                guard=guard,
                boundary_admission=boundary,
                route_admissions=per_category,
                top_k=top_k,
            )
        )
        if backend.live_prefill_count:
            raise ShardContractError(f"context {context_id!r} left a cache backend live")

    # ---- Phase 2: terminal behavior phase.  No likelihood score may follow.
    if capture_behavior_sidecars:
        guard.open_generation_phase(backend=backend)
        for item in items:
            free_rows.append(
                free_greedy_box_sidecar(
                    backend,
                    plan,
                    item,
                    guard=guard,
                    admission=admissions[item.admission_receipt_id],
                    bank_tokens=bank_tokens,
                )
            )
        for context_id in sorted(boundary_admissions):
            free_rows.append(
                free_next_row_sidecar(
                    backend,
                    plan,
                    context_id=context_id,
                    guard=guard,
                    admission=boundary_admissions[context_id],
                )
            )
        if backend.live_prefill_count:
            raise ShardContractError("the generation phase left a cache backend live")

    scored_candidate_ids = sorted({str(row["candidate_id"]) for row in score_rows})
    owner_bank = build_owner_bank_accounting(
        plan, image_id, scored_candidate_ids=scored_candidate_ids
    )

    for label, rows in (
        ("score row", score_rows),
        ("x1 row", x1_rows),
        ("proposal row", proposal_rows),
        ("sidecar row", free_rows),
    ):
        for row in rows:
            assert_row_bindings(row, label=label)

    scalar_diffs = [
        float(row["scalar_reference"]["abs_diff_vs_cached"])
        for row in score_rows
        if row.get("scalar_reference") is not None
    ]
    receipt = {
        "schema_version": SHARD_RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "row_contract": P0_ROW_CONTRACT,
        "image_id": image_id,
        "split": str(plan.shards[image_id]["split"]),
        "status": "captured",
        "code": {
            "executed_source_sha256": EXECUTED_SOURCE_SHA256,
            "planner_source_sha256": planner.sha256_file(Path(planner.__file__).resolve()),
            "runtime_seam_source_sha256": planner.sha256_file(
                REPO_ROOT / "scripts/research/score_sorted_owner_basin_landscape.py"
            ),
        },
        "plan": {
            "plan_dir": str(plan.plan_dir),
            "plan_schema_version": PLAN_SCHEMA_VERSION,
            "receipt_content_sha256": plan.receipt_content_sha256,
            "capture_rules_sha256": plan.capture_rules_sha256,
        },
        "backend_identity": dict(backend.identity),
        "score_input_policy": {
            "reads_any_score_artifact": False,
            "pre_p0_scores": "quarantined_never_read_mechanically_unjoinable",
        },
        "capture_completeness": "complete_shard" if complete else "subset_smoke",
        "subset_capture": {
            "is_subset": not complete,
            "planned_admitted_query_group_count": len(planned),
            "executed_query_group_count": len(groups),
            "plan_was_modified": False,
            "usable_as_complete_shard_evidence": complete,
        },
        "granularity": {
            "session_scope": admission_rules["session_scope"],
            "admission_scope": admission_rules["admission_scope"],
            "cache_scope": admission_rules["cache_scope"],
            "admission_key": "exact_prefix_sha256_per_channel",
            "singleton_group_validated_per_scoring_unit": True,
            "admission_inherited_across_prefixes": False,
            "fresh_cache_per_query_group": True,
            "groups_executed_sequentially": True,
            "bulk_scoring_path": (
                "admitted_kv_cache_batched_candidate_lanes"
                if int(candidate_batch_size) > 1
                else "admitted_kv_cache_single_candidate_branch"
            ),
            "batched_full_reforward_role": "parity_diagnostic_only_never_fallback",
            "candidate_batch_size": int(candidate_batch_size),
            "candidate_batching_enabled": int(candidate_batch_size) > 1,
            "candidate_batch_scope": "one_exact_query_group_and_admitted_prefix_only",
            "bytes_identical_to_candidate_batch_size_one": int(candidate_batch_size) == 1,
            "out_of_memory_policy": "fail_loud_quarantine_never_silent_fallback",
        },
        "numerics": {
            "matmul_precision": matmul_precision_state(),
            "batched_path_parity": batched_path_parity,
            "batched_path_max_abs_diff_bound": BATCHED_PATH_MAX_ABS_DIFF,
        },
        "runtime_invariants": {
            "explicit_position_ids_on_every_model_call": True,
            "uses_model_generate": False,
            "concurrent_groups_on_one_model": False,
            "cache_length_asserted_against_prefill_length": True,
            "cache_backend_released_before_next_group": True,
            "explicit_position_ids_shape": "3_by_batch_by_time_from_prefill_rope_deltas",
            "shared_model_rope_deltas_recomputation_used": False,
            "candidate_lane_kv_storage": "per_lane_materialized_never_stride_zero",
            "group_root_cache_never_expanded_in_place": True,
        },
        "phase_order": {
            "decision_scoring_complete_before_generation": True,
            "generation_phase_after_decision_scoring": capture_behavior_sidecars,
            "likelihood_scored_after_generation_phase": False,
            "free_box_max_tokens": FREE_BOX_MAX_TOKENS,
            "free_row_max_tokens": FREE_ROW_MAX_TOKENS,
            "behavior_sidecars_captured": capture_behavior_sidecars,
            "behavior_sidecars_intentionally_disabled": not capture_behavior_sidecars,
            "behavior_sidecar_disable_reason": (
                None
                if capture_behavior_sidecars
                else "score_only_successor_capture_requested_no_free_decode_behavior_record"
            ),
        },
        "admission": {
            "channels": list(planner.ADMISSION_CHANNELS),
            "box_channel_receipt_count": sum(
                1 for row in admissions.values() if row["channel"] == CHANNEL_QUERY_SUFFIX
            ),
            "proposal_boundary_gate_receipt_count": len(boundary_admissions),
            "proposal_route_receipt_count": sum(
                len(row) for row in route_admissions.values()
            ),
            "proposal_route_receipts_are_per_context_and_category": True,
            "all_admitted": all(row["admitted"] for row in admissions.values()),
            "relaxed_selected_logprob_max_abs_diff": RELAXED_SELECTED_LOGPROB_MAX_ABS_DIFF,
            "suffix_shape_coverage": shape_coverage,
            "receipts": [admissions[key] for key in sorted(admissions)],
        },
        # One target-blind repeat/epsilon receipt per image; every exact group
        # additionally carries its own four-depth parity gate above.
        "scalar_admission": {
            "image_epsilon_receipt": epsilon_receipt,
            "selection": scalar_rules["selection"],
            "modulus": int(scalar_rules["modulus"]),
            "repeat_count": int(scalar_rules["repeat_count"]),
            "max_abs_diff_bound": float(scalar_rules["max_abs_diff"]),
            "target_blind": True,
            "spot_check_row_count": len(scalar_diffs),
            "max_scalar_vs_cached_abs_diff": max(scalar_diffs, default=0.0),
            "every_group_has_a_spot_check": all(
                any(
                    row.get("scalar_reference") is not None
                    and row["query_group_id"] == item.query_group_id
                    for row in score_rows
                )
                for item in items
            ),
        },
        "counts": {
            "query_group_count": len(groups),
            "context_count": len(boundary_admissions),
            "localization_score_rows": len(score_rows),
            "x1_diagnostic_rows": len(x1_rows),
            "proposal_surface_rows": len(proposal_rows),
            "free_decode_sidecar_rows": len(free_rows),
            "distinct_scored_physical_candidates": len(scored_candidate_ids),
        },
        "competition": {
            "rank_scope": "image_context_normalized_description",
            "population": "collapsed_unique_physical_candidates_only",
            "sidecars_excluded": True,
            "per_query_group": competitions,
        },
        "owner_bank_accounting": owner_bank,
        "checks": {
            "all_finite": all(
                math.isfinite(float(row["complete_box_logprob_sum"])) for row in score_rows
            ),
            "coordinate_domain_ok": all(
                all(
                    COORD_TOKEN_START <= int(token) <= COORD_TOKEN_END
                    for token in row["coord_token_ids"]
                )
                for row in score_rows
            ),
            "canonical_suffix_verified": True,
            "cross_owner_tuple_collapse_verified": True,
            "every_row_bound_to_an_admission_receipt": True,
        },
        "elapsed_seconds": time.time() - started,
    }

    result = ShardResult(
        receipt=receipt,
        scores=score_rows,
        x1=x1_rows,
        proposals=proposal_rows,
        free_decodes=free_rows,
    )
    if output_dir is not None:
        write_shard(Path(output_dir), result)
    return result


#: Prefix of the sibling staging directory one invocation owns.  Failure
#: cleanup only ever touches a directory matching this pattern *and* holding
#: nothing but this module's own declared outputs.
STAGING_DIR_PREFIX = ".staging-"


def shard_output_files(result: ShardResult) -> dict[str, bytes]:
    """The complete published byte content of one shard, as one indivisible set."""

    return {
        SCORES_NAME: b"".join(canonical_json_bytes(row) + b"\n" for row in result.scores),
        X1_NAME: b"".join(canonical_json_bytes(row) + b"\n" for row in result.x1),
        PROPOSAL_NAME: b"".join(canonical_json_bytes(row) + b"\n" for row in result.proposals),
        FREE_DECODE_NAME: b"".join(
            canonical_json_bytes(row) + b"\n" for row in result.free_decodes
        ),
        RECEIPT_NAME: canonical_json_bytes(result.receipt) + b"\n",
    }


def _write_durable(path: Path, content: bytes) -> None:
    """Write and flush to stable storage before the file is ever published."""

    with open(path, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    """Persist a directory entry, so a publish survives an interruption."""

    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        # Directory fsync is not available on every filesystem; the file
        # contents are already durable, so this is a best-effort hardening.
        pass
    finally:
        os.close(fd)


def _staging_dir_for(output_dir: Path) -> Path:
    return output_dir.parent / f"{output_dir.name}{STAGING_DIR_PREFIX}{os.getpid()}-{uuid4().hex[:12]}"


def _remove_owned_staging(staging: Path, *, expected_names: Sequence[str]) -> None:
    """Delete only this invocation's staging directory, and only if it is ours.

    Refuses to remove anything that is not named as our staging directory or
    that holds a file this module did not write, so an operator's data can
    never be destroyed by a failed capture.
    """

    if not staging.exists():
        return
    if STAGING_DIR_PREFIX not in staging.name:
        raise ShardContractError(
            f"refusing to clean {staging} : it is not a shard staging directory"
        )
    entries = sorted(child.name for child in staging.iterdir())
    foreign = [name for name in entries if name not in set(expected_names)]
    if foreign:
        raise ShardContractError(
            f"refusing to clean staging directory {staging} : it holds unexpected entries "
            f"{foreign!r}"
        )
    for name in entries:
        (staging / name).unlink()
    staging.rmdir()


def inspect_published_shard(output_dir: Path, files: Mapping[str, bytes]) -> dict[str, Any]:
    """Classify an already-existing final output directory against this shard."""

    entries = sorted(child.name for child in output_dir.iterdir())
    missing = sorted(name for name in files if name not in entries)
    unexpected = sorted(name for name in entries if name not in files)
    differing = sorted(
        name
        for name in files
        if name in entries and (output_dir / name).read_bytes() != files[name]
    )
    return {
        "entries": entries,
        "missing": missing,
        "unexpected": unexpected,
        "differing": differing,
        "identical": not (missing or unexpected or differing),
    }


def write_shard(output_dir: Path, result: ShardResult) -> dict[str, Any]:
    """Publish one shard atomically: create-or-identical, never partial.

    Every file is written and fsynced into a sibling staging directory this
    invocation owns, and the shard becomes visible only when the *whole* set is
    durable, via a single directory rename.  An interruption or disk error
    therefore leaves no partial primary evidence at the final path -- only an
    owned staging directory, which this function removes on the way out.

    An existing final directory is never overwritten.  It is either
    byte-identical to this shard (an idempotent re-run, published as a no-op)
    or it is foreign, in which case the publish fails closed and the existing
    directory is left exactly as it was.  A stale ``shard-quarantine.json``
    counts as foreign on purpose: silently replacing it would erase the record
    that this image once failed, which the global stop policy counts.
    """

    output_dir = Path(output_dir)
    files = shard_output_files(result)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if output_dir.exists():
        return _publish_record_for_existing(output_dir, files)

    staging = _staging_dir_for(output_dir)
    if staging.exists():
        raise ShardContractError(f"staging directory {staging} already exists")
    staging.mkdir(parents=True)
    published = False
    try:
        for name in sorted(files):
            _write_durable(staging / name, files[name])
        _fsync_dir(staging)
        try:
            os.rename(staging, output_dir)
        except OSError as exc:
            if output_dir.exists():
                # Another invocation published while we were staging.
                record = _publish_record_for_existing(output_dir, files)
                _remove_owned_staging(staging, expected_names=tuple(files))
                return record
            raise ShardContractError(
                f"could not publish shard output to {output_dir}: {exc}"
            ) from exc
        published = True
    finally:
        if not published:
            _remove_owned_staging(staging, expected_names=tuple(files))

    _fsync_dir(output_dir.parent)
    return {
        "output_dir": str(output_dir),
        "published": True,
        "publish_mode": "atomic_staging_directory_rename",
        "already_present_identical": False,
        "file_names": sorted(files),
        "staging_residue": [],
    }


def _publish_record_for_existing(
    output_dir: Path, files: Mapping[str, bytes]
) -> dict[str, Any]:
    state = inspect_published_shard(output_dir, files)
    if state["identical"]:
        return {
            "output_dir": str(output_dir),
            "published": False,
            "publish_mode": "no_op_identical_rerun",
            "already_present_identical": True,
            "file_names": sorted(files),
            "staging_residue": [],
        }
    raise ShardContractError(
        f"refusing to publish into existing output directory {output_dir}: it is not a "
        "byte-identical capture of this shard "
        f"(missing={state['missing']!r}, differing={state['differing']!r}, "
        f"unexpected={state['unexpected']!r}); "
        "the existing directory is left untouched -- capture into a fresh output directory"
    )


def find_staging_residue(output_dir: Path) -> list[str]:
    """Any staging directory left behind at this output path, from any run."""

    output_dir = Path(output_dir)
    parent = output_dir.parent
    if not parent.is_dir():
        return []
    return sorted(
        child.name
        for child in parent.iterdir()
        if child.is_dir() and child.name.startswith(f"{output_dir.name}{STAGING_DIR_PREFIX}")
    )


def write_quarantine(
    output_dir: Path, *, image_id: str, reason: str, detail: str
) -> dict[str, Any]:
    """Quarantine exactly this image; other shards are unaffected.

    The receipt reports residue *honestly*.  This invocation publishes shard
    outputs atomically, so it never writes partial primary rows itself; but a
    prior run, another tool, or an operator may have left files at this path.
    Those are reported, never deleted, and the shard's evidence stays unusable
    either way.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    residue = sorted(name for name in PRIMARY_OUTPUT_NAMES if (output_dir / name).exists())
    staging_residue = find_staging_residue(output_dir)
    payload = {
        "schema_version": QUARANTINE_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "image_id": image_id,
        "status": "quarantined",
        "reason": reason,
        "detail": detail,
        "scope": "this_image_only",
        "evidence_usable": False,
        # Honest residue accounting: this is what is actually on disk now, not
        # a claim about what this invocation intended.
        "primary_output_residue": residue,
        "primary_rows_written": bool(residue),
        "shard_receipt_present": (output_dir / RECEIPT_NAME).exists(),
        "staging_directory_residue": staging_residue,
        "residue_left_in_place": True,
        "residue_origin": (
            "not_written_by_this_invocation_shard_publish_is_atomic"
            if residue or staging_residue
            else "none"
        ),
        "publish_policy": "atomic_staging_directory_rename_create_or_identical",
        "global_stop_threshold_quarantined_images": 2,
        "executed_source_sha256": EXECUTED_SOURCE_SHA256,
    }
    _write_durable(output_dir / QUARANTINE_NAME, canonical_json_bytes(payload) + b"\n")
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backend", choices=("hf", "fake"), default="fake")
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_INFER_CONFIG)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=DEFAULT_CANDIDATE_BATCH_SIZE,
        help=(
            "independent candidates scored per forward inside one exact query group "
            f"(default {DEFAULT_CANDIDATE_BATCH_SIZE}, which reproduces the single-branch "
            "bytes exactly); above one the batched path is gated by a live-model parity "
            "probe and an out-of-memory error quarantines the shard rather than falling back"
        ),
    )
    parser.add_argument(
        "--validate-contract-only",
        action="store_true",
        help="CPU-only contract check over every planned work item; loads no model",
    )
    parser.add_argument("--dry-run", action="store_true", help="alias for --validate-contract-only")
    parser.add_argument(
        "--query-group-id",
        action="append",
        default=None,
        help="score only these singleton groups (repeatable); subset capture, sealed plan unchanged",
    )
    parser.add_argument(
        "--smoke-query-groups",
        type=int,
        default=None,
        help=(
            "score only the first N admitted groups of this shard, in sealed plan order; "
            "a real-model smoke subset that never modifies the plan and is receipted as "
            "capture_completeness=subset_smoke"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    image_id = str(args.image_id)
    try:
        plan = load_plan(args.plan_dir)
    except ShardContractError as exc:
        print(f"plan error: {exc}", file=sys.stderr)
        return 2

    if args.validate_contract_only or args.dry_run:
        try:
            report = validate_shard_contract(plan, image_id)
        except ShardContractError as exc:
            print(f"contract error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.output_dir is None:
        print("--output-dir is required unless --validate-contract-only", file=sys.stderr)
        return 2

    kwargs = {
        "image_id": image_id,
        "output_dir": args.output_dir,
        "top_k": args.top_k,
        "query_group_ids": args.query_group_id,
        "max_query_groups": args.smoke_query_groups,
        "candidate_batch_size": args.candidate_batch_size,
    }
    try:
        if args.backend == "fake":
            result = run_shard(plan, backend=FakeCensusBackend(), **kwargs)
        else:
            spec = build_hf_session_spec(plan, image_id, infer_config=args.infer_config)
            with open_hf_backend(spec) as backend:
                result = run_shard(plan, backend=backend, **kwargs)
    except Exception as exc:  # noqa: BLE001 - shard-local quarantine is the contract
        payload = write_quarantine(
            args.output_dir,
            image_id=image_id,
            reason=type(exc).__name__,
            detail="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        )
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        print(
            f"shard {image_id} quarantined; other shards are unaffected "
            "(global stop only above two quarantined images)",
            file=sys.stderr,
        )
        return 3

    print(json.dumps(result.receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
