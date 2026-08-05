#!/usr/bin/env python3
"""Immutable secondary downstream-compatibility merge for the sorted
crossing-boundary owner release/realization unit
(``2026-08-03-sorted-crossing-boundary-owner-release-realization``).

Frozen unit::

    research/investigations/qwen3-vl-dense-enumeration/experiments/
    2026-08-03-sorted-crossing-boundary-owner-release-realization/
    {unit.md,tasks.md}  -- "Secondary downstream compatibility"

What this module is
-------------------
It consumes **exactly one successful capture shard per sealed plan image** --
all twelve, no more and no fewer -- re-proves every identity, token digest and
paired arithmetic those shards declare, and republishes their *raw* rows under
one deterministic, self-sealed merged artifact family.

The producer is
``score_sorted_crossing_boundary_owner_release_secondary``; the emitted row,
parity and receipt schemas are what this module binds to.

Everything it re-proves before a single row is copied:

* **file identity** -- every shard directory carries exactly the three secondary
  files, no quarantine file, no admission file, and no unknown artifact;
* **self identity** -- every receipt reconstructs its own
  ``receipt_content_sha256`` from its own declared content;
* **source identity** -- every shard declares one and the same secondary scorer
  digest, both as ``scorer_source_sha256`` and inside
  ``runtime_identity.source_identity``, and one and the same full source
  identity map;
* **plan identity** -- every shard was captured against the frozen plan whose
  manifest self-seals and whose sealed file digests match the bytes on disk;
* **primary-analysis identity** -- every shard, and every row inside it, carries
  the same sealed primary-branch binding, and that binding reconstructs from the
  named analysis directory's own bytes;
* **runtime / admission identity** -- all twelve shards share one runtime and
  one self-sealed smoke admission; and
* **content identity** -- the union of published rows is *exactly* the sealed
  plan's 64 secondary requests, and every row reproduces its own plan request,
  root prefixes, appended clean row ``C``, literal segment spans, per-segment
  sums and modified-minus-baseline deltas.

What this module deliberately is **not**
----------------------------------------
* It reads, assigns and re-derives **no branch**.  The sealed primary analysis
  is bound only by *digest*; its branch labels are never opened, never carried
  into a merged artifact, and never used to select or drop a row.
* It fits **no threshold**, assigns **no compatibility verdict**, and makes no
  retention, recovery or free-rollout claim.  ``unit.md``: these readouts
  "measure local compatibility of exact row sequences".  That boundary travels
  in every merged row and in the merge receipt.
* It repairs nothing.  A withheld, incomplete, duplicated, quarantine-stopped or
  unknown input fails closed; the shards on disk are never modified.

Why the secondary scorer digest is checked for *agreement* rather than pinned
----------------------------------------------------------------------------
The primary merge pins one frozen ``FROZEN_SCORER_SOURCE_SHA256`` constant.  The
secondary producer is still being tightened, so pinning a literal here would
freeze a digest that is known to move.  Instead every shard must agree on one
secondary scorer digest and that digest must reconcile with the shard's own
``runtime_identity.source_identity``; a caller who wants the primary merge's
stronger guarantee passes ``--expect-scorer-source-sha256``.

Outputs (one explicit merged directory, published create-or-identical)::

    secondary-compatibility-rows.jsonl    raw per-request rows, all shards
    secondary-compatibility-parity.jsonl  one raw parity object per shard
    secondary-merge-receipt.json          identities, digests, counters
"""

from __future__ import annotations

import argparse
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

from scripts.research import (  # noqa: E402
    merge_sorted_crossing_boundary_owner_release as primary_merge,
)
from scripts.research import (  # noqa: E402
    prepare_sorted_crossing_boundary_owner_release_realization as plan_builder,
)
from scripts.research import (  # noqa: E402
    score_sorted_crossing_boundary_owner_release as primary,
)
from scripts.research import (  # noqa: E402
    score_sorted_crossing_boundary_owner_release_secondary as secondary,
)

# ---------------------------------------------------------------------------
# 0. Frozen identities
# ---------------------------------------------------------------------------

UNIT_ID = secondary.UNIT_ID
MERGE_SCHEMA_VERSION = "sorted_crossing_boundary_secondary_compatibility_merge.v1"

#: unit.md "Exact state pair": the frozen twelve images, shared with the primary
#: merge so the two families can never cover different image sets.
FROZEN_IMAGE_IDS: tuple[str, ...] = primary_merge.FROZEN_IMAGE_IDS
EXPECTED_SHARD_COUNT = primary_merge.EXPECTED_SHARD_COUNT
FROZEN_PLAN_MANIFEST_SCHEMA_VERSION = primary_merge.FROZEN_PLAN_MANIFEST_SCHEMA_VERSION

#: Module keys inside ``runtime_identity.source_identity``.
SECONDARY_SCORER_SOURCE_IDENTITY_KEY = (
    "scripts.research.score_sorted_crossing_boundary_owner_release_secondary"
)
PRIMARY_SCORER_SOURCE_IDENTITY_KEY = primary_merge.SCORER_SOURCE_IDENTITY_KEY
PRIMARY_ANALYZER_SOURCE_IDENTITY_KEY = (
    "scripts.research.analyze_sorted_crossing_boundary_owner_release"
)
#: Every module whose digest a shard must have sealed for its semantics to be
#: reconstructable.  Taken from the producer's own list so the two cannot drift.
REQUIRED_SOURCE_IDENTITY_KEYS: tuple[str, ...] = tuple(
    sorted(secondary.SOURCE_IDENTITY_MODULES)
)

#: Shard input file names, taken from the producer so the two cannot drift.
SHARD_ROWS_NAME = secondary.ROWS_NAME
SHARD_PARITY_NAME = secondary.PARITY_NAME
SHARD_RECEIPT_NAME = secondary.RECEIPT_NAME
SHARD_QUARANTINE_NAME = secondary.QUARANTINE_NAME
SHARD_ADMISSION_NAME = secondary.ADMISSION_NAME
REQUIRED_SHARD_FILES: tuple[str, ...] = (
    SHARD_ROWS_NAME,
    SHARD_PARITY_NAME,
    SHARD_RECEIPT_NAME,
)

#: Merged output file names.
MERGED_ROWS_NAME = SHARD_ROWS_NAME
MERGED_PARITY_NAME = "secondary-compatibility-parity.jsonl"
MERGE_RECEIPT_NAME = "secondary-merge-receipt.json"
MERGED_OUTPUT_NAMES: tuple[str, ...] = (
    MERGED_ROWS_NAME,
    MERGED_PARITY_NAME,
    MERGE_RECEIPT_NAME,
)

#: Plan files whose bytes the frozen manifest re-proves.
PLAN_FILE_NAMES: tuple[str, ...] = primary_merge.PLAN_FILE_NAMES

PRIMARY_COHORT = plan_builder.PRIMARY_COHORT
TP_REPLAY_CONTROL_COHORT = plan_builder.TP_REPLAY_CONTROL_COHORT

SEGMENTS: tuple[str, ...] = secondary.SEGMENTS
PAIRED_ROOTS: tuple[str, ...] = secondary.PAIRED_ROOTS
ROOT_BASELINE = secondary.ROOT_BASELINE
ROOT_MODIFIED = secondary.ROOT_MODIFIED
SECONDARY_VARIANTS: tuple[str, ...] = secondary.SECONDARY_VARIANTS

#: The registry field that owns each variant's scored native action, and the
#: baseline-context source label the producer stamps beside it.  Keyed by
#: variant so a row can never borrow another variant's sealed row identity.
VARIANT_REGISTRY_ROW_KEY: Mapping[str, str] = {
    secondary.VARIANT_P_PLUS_C_THEN_E: "e_row",
    secondary.VARIANT_P_PLUS_E_PLUS_C_THEN_F: "f_row",
    secondary.VARIANT_BENIGN_SUBSTITUTION: "following_native_action",
}
#: Within those registry blocks, the field naming the scored row's token digest
#: and its native row index.  The two sidecar shapes differ, so both are named.
VARIANT_REGISTRY_DIGEST_KEY: Mapping[str, str] = {
    secondary.VARIANT_P_PLUS_C_THEN_E: "full_row_token_ids_sha256",
    secondary.VARIANT_P_PLUS_E_PLUS_C_THEN_F: "full_row_token_ids_sha256",
    secondary.VARIANT_BENIGN_SUBSTITUTION: "token_ids_sha256",
}

#: The capture policy fields a merged shard must have executed under.
REQUIRED_POLICY: Mapping[str, Any] = {
    "uses_model_generate": False,
    "retokenizes": False,
    "sampling": "not_implemented_secondary_deterministic_teacher_forcing_only",
    "likelihood_channel": secondary.LIKELIHOOD_CHANNEL,
    "primary_branches_sealed_before_this_pass": True,
    "primary_branch_assignment_performed": False,
    "branch_labels_used_to_select_requests": False,
    "executes_every_sealed_secondary_request_of_the_image": True,
    "one_image_session_per_shard": True,
    "fresh_cache_per_logical_root": True,
    "paired_roots_force_identical_target_tokens": True,
    "delta_orientation": "modified_minus_baseline",
    "replay_admission_enforced": True,
    "claim_boundary": secondary.CLAIM_BOUNDARY,
    "final_set_retention_or_free_rollout_claimed": False,
}

#: Row-level policy sentinels: what every published row must still declare.
REQUIRED_ROW_POLICY: Mapping[str, Any] = {
    "row_kind": "secondary_compatibility_row",
    "readout_tier": secondary.SECONDARY_READOUT_TIER,
    "request_family": secondary.REQUEST_FAMILY,
    "likelihood_channel": secondary.LIKELIHOOD_CHANNEL,
    "uses_model_generate": False,
    "sampling": "disabled_secondary_deterministic_teacher_forcing_only",
    "retokenized": False,
    "delta_orientation": "modified_minus_baseline",
    "claim_boundary": secondary.CLAIM_BOUNDARY,
    "replay_admission_enforced": True,
    "baseline_replay_admitted": True,
}


class SecondaryMergeContractError(primary_merge.MergeContractError):
    """A precondition of this unit's immutable secondary merge was not proven."""


def _fail(message: str) -> NoReturn:
    raise SecondaryMergeContractError(message)


# ---------------------------------------------------------------------------
# 1. IO / digest helpers (reuse the producer's canonicalization)
# ---------------------------------------------------------------------------

canonical_json_bytes = secondary.canonical_json_bytes
sha256_bytes = secondary.sha256_bytes
sha256_json = secondary.sha256_json
sha256_file = secondary.sha256_file

#: The digest of the empty appended-token list, i.e. "this root appended
#: nothing".  Derived, never written as a literal.
EMPTY_APPEND_SHA256 = sha256_json([])


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


def assert_self_sealed(payload: Mapping[str, Any], *, digest_key: str, label: str) -> str:
    """A sealed artifact must reconstruct its own digest from its own content."""

    declared = payload.get(digest_key)
    if not isinstance(declared, str) or not declared:
        _fail(f"{label} declares no {digest_key}")
    reconstructed = sha256_json(
        {key: value for key, value in payload.items() if key != digest_key}
    )
    if reconstructed != declared:
        _fail(
            f"{label} does not reconstruct its own {digest_key}; it was edited after it "
            "was sealed"
        )
    return declared


def _token_ids(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(f"{label} is not a token id sequence")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        _fail(f"{label} carries a non-integer token id")


def _floats(value: Any, *, label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(f"{label} is not a numeric sequence")
    values: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            _fail(f"{label} carries a non-numeric entry {item!r}")
        number = float(item)
        if not math.isfinite(number):
            _fail(f"{label} carries the non-finite value {item!r}")
        values.append(number)
    return values


# ---------------------------------------------------------------------------
# 2. The frozen plan and its sealed 64-request secondary census
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenSecondaryPlan:
    """The sealed CPU plan, plus the secondary census and registries it owns."""

    plan_dir: Path
    manifest: dict[str, Any]
    manifest_content_sha256: str
    plan_file_sha256: dict[str, str]
    requests_by_id: dict[str, dict[str, Any]]
    registry_by_owner: dict[str, dict[str, Any]]
    counts: dict[str, Any]

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.requests_by_id))

    @property
    def request_ids_sha256(self) -> str:
        return sha256_json(sorted(self.requests_by_id))

    def requests_for_image(self, image_id: str) -> list[dict[str, Any]]:
        return sorted(
            (
                dict(row)
                for row in self.requests_by_id.values()
                if str(row["image_id"]) == str(image_id)
            ),
            key=lambda row: str(row["request_id"]),
        )


def _assert_plan_request_token_digests(row: Mapping[str, Any]) -> None:
    """Every token payload the plan seals must hash to its own declared digest.

    The request *identity* digest is not recomputed here: its exact field shape
    is the plan builder's contract, and duplicating that shape in a downstream
    merge would create two definitions of one identity.  What is re-proven is
    what this merge actually consumes -- the literal token sequences -- plus the
    plan builder's own ``request_id`` derivation from the sealed identity digest.
    """

    request_id = str(row.get("request_id"))
    identity_digest = str(row.get("identity_digest"))
    if request_id != f"req:{identity_digest[:32]}":
        _fail(
            f"sealed plan request {request_id!r} is not derived from its own identity digest "
            f"{identity_digest!r}"
        )
    prefix = row.get("prefix")
    if not isinstance(prefix, Mapping):
        _fail(f"sealed plan request {request_id!r} carries no prefix binding")
    appended = _token_ids(
        prefix.get("appended_token_ids"), label=f"plan request {request_id!r} appended tokens"
    )
    if sha256_json(appended) != str(prefix.get("appended_token_ids_sha256")):
        _fail(
            f"sealed plan request {request_id!r} appended tokens do not hash to the digest the "
            "plan declares"
        )
    if len(appended) != int(prefix.get("appended_token_count", -1)):
        _fail(
            f"sealed plan request {request_id!r} appended-token count contradicts its own tokens"
        )
    target = row.get("scored_target")
    if not isinstance(target, Mapping):
        _fail(f"sealed plan request {request_id!r} carries no scored target")
    tokens = _token_ids(
        target.get("token_ids"), label=f"plan request {request_id!r} scored target"
    )
    if sha256_json(tokens) != str(target.get("token_ids_sha256")):
        _fail(
            f"sealed plan request {request_id!r} scored-target tokens do not hash to the digest "
            "the plan declares"
        )


def _assert_request_key(row: Mapping[str, Any], *, label: str) -> str:
    """The immutable request key: family|cohort|owner|context|variant."""

    expected = "|".join(
        (
            secondary.REQUEST_FAMILY,
            str(row.get("cohort")),
            str(row.get("gt_owner_id")),
            str(row.get("context_id") or row.get("modified_context_id")),
            str(row.get("variant")),
        )
    )
    observed = str(row.get("request_key"))
    if observed != expected:
        _fail(
            f"{label} declares request key {observed!r}, which does not reconstruct from its own "
            f"sealed identity fields ({expected!r})"
        )
    return observed


def load_frozen_plan(plan_dir: Path) -> FrozenSecondaryPlan:
    """Re-prove the frozen plan, then index its sealed secondary census."""

    frozen = primary_merge.load_frozen_plan(Path(plan_dir))

    cohort_rows = _read_jsonl(
        frozen.plan_dir / plan_builder.COHORT_REGISTRY_NAME, "sealed cohort registry"
    )
    control_rows = _read_jsonl(
        frozen.plan_dir / plan_builder.CONTROL_REGISTRY_NAME, "sealed control registry"
    )
    request_rows = _read_jsonl(
        frozen.plan_dir / plan_builder.REQUEST_PLAN_NAME, "sealed request plan"
    )
    for label, rows, schema in (
        ("cohort registry", cohort_rows, plan_builder.COHORT_SCHEMA_VERSION),
        ("control registry", control_rows, plan_builder.CONTROL_SCHEMA_VERSION),
        ("request plan", request_rows, plan_builder.REQUEST_SCHEMA_VERSION),
    ):
        if not rows:
            _fail(f"the sealed {label} is empty")
        for row in rows:
            if str(row.get("schema_version")) != schema:
                _fail(
                    f"the sealed {label} carries a row of schema {row.get('schema_version')!r}, "
                    f"not {schema!r}"
                )
            if str(row.get("unit_id")) != UNIT_ID:
                _fail(f"the sealed {label} carries a row from another unit")

    registry_by_owner: dict[str, dict[str, Any]] = {}
    for row in (*cohort_rows, *control_rows):
        owner_id = str(row["gt_owner_id"])
        if owner_id in registry_by_owner:
            _fail(f"owner {owner_id!r} is registered twice in the sealed plan registries")
        registry_by_owner[owner_id] = dict(row)

    secondary_rows: list[dict[str, Any]] = []
    requests_by_id: dict[str, dict[str, Any]] = {}
    for row in request_rows:
        if str(row.get("readout_tier")) != secondary.SECONDARY_READOUT_TIER:
            continue
        secondary.assert_secondary_request(row)
        _assert_plan_request_token_digests(row)
        _assert_request_key(row, label=f"sealed plan request {row.get('request_id')!r}")
        request_id = str(row["request_id"])
        if request_id in requests_by_id:
            _fail(f"the sealed plan carries a duplicate secondary request {request_id!r}")
        requests_by_id[request_id] = dict(row)
        secondary_rows.append(dict(row))

    counts = secondary.validate_secondary_plan_counts(
        secondary_rows, manifest=frozen.manifest
    )

    covered_images = sorted({str(row["image_id"]) for row in secondary_rows})
    if covered_images != sorted(FROZEN_IMAGE_IDS):
        _fail(
            f"the sealed secondary census covers images {covered_images!r}, not this unit's "
            f"frozen {sorted(FROZEN_IMAGE_IDS)!r}"
        )
    for row in secondary_rows:
        owner_id = str(row["gt_owner_id"])
        registry_row = registry_by_owner.get(owner_id)
        if registry_row is None:
            _fail(
                f"sealed secondary request {row['request_id']!r} names owner {owner_id!r}, which "
                "is in neither sealed registry"
            )
        if str(registry_row.get("cohort")) != str(row.get("cohort")):
            _fail(
                f"owner {owner_id!r} is registered in cohort {registry_row.get('cohort')!r} but "
                f"its sealed secondary request declares {row.get('cohort')!r}"
            )

    return FrozenSecondaryPlan(
        plan_dir=frozen.plan_dir,
        manifest=frozen.manifest,
        manifest_content_sha256=frozen.manifest_content_sha256,
        plan_file_sha256=dict(frozen.plan_file_sha256),
        requests_by_id=requests_by_id,
        registry_by_owner=registry_by_owner,
        counts=dict(counts),
    )


# ---------------------------------------------------------------------------
# 3. The sealed primary-branch gate, bound by digest only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrimaryAnalysisBinding:
    """The sealed primary analysis, reduced to the identities this merge binds.

    Deliberately *identity only*: no branch label, count or routing decision is
    read out of the analysis directory or carried into any merged artifact.
    ``unit.md`` seals the branch pass before a secondary field exists, and this
    merge keeps that separation in the other direction too.
    """

    analysis_dir: Path
    file_sha256: dict[str, str]
    receipt_content_sha256: str
    analyzer_source_sha256: str
    plan_manifest_content_sha256: str


def load_primary_analysis_binding(
    analysis_dir: Path, *, plan: FrozenSecondaryPlan
) -> PrimaryAnalysisBinding:
    """Re-hash the sealed analysis directory and prove it is this plan's gate."""

    analysis_dir = Path(analysis_dir)
    file_sha256 = secondary._assert_analysis_directory_shape(analysis_dir)  # noqa: SLF001
    receipt = _read_json(
        analysis_dir / secondary.ANALYSIS_RECEIPT_NAME, "primary analysis receipt"
    )
    if str(receipt.get("schema_version")) != secondary.ANALYSIS_RECEIPT_SCHEMA_VERSION:
        _fail(
            f"primary analysis receipt schema {receipt.get('schema_version')!r} is not "
            f"{secondary.ANALYSIS_RECEIPT_SCHEMA_VERSION!r}"
        )
    if str(receipt.get("unit_id")) != UNIT_ID:
        _fail("primary analysis receipt belongs to another unit")
    receipt_digest = assert_self_sealed(
        receipt, digest_key="receipt_content_sha256", label="primary analysis receipt"
    )
    analysis_policy = receipt.get("policy")
    if not isinstance(analysis_policy, Mapping):
        _fail("the sealed primary analysis receipt carries no policy block")
    if analysis_policy.get("secondary_compatibility_read") is not False:
        _fail(
            "the sealed primary analysis does not declare secondary_compatibility_read=false; "
            "its branches may have been decided while reading this very readout"
        )
    merge_receipt_path = Path(str(receipt.get("merged_dir"))) / (
        secondary.ANALYSIS_MERGE_RECEIPT_NAME
    )
    merge_receipt = _read_json(merge_receipt_path, "primary merge receipt")
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
    plan_manifest_sha256 = str(merge_plan.get("manifest_content_sha256"))
    if plan_manifest_sha256 != plan.manifest_content_sha256:
        _fail(
            "the sealed primary analysis was decided over a different CPU plan manifest than the "
            "one this secondary merge binds; refusing to publish secondary rows beside a branch "
            "gate that never saw this cohort"
        )
    return PrimaryAnalysisBinding(
        analysis_dir=analysis_dir,
        file_sha256=dict(sorted(file_sha256.items())),
        receipt_content_sha256=receipt_digest,
        analyzer_source_sha256=str(receipt.get("analyzer_source_sha256")),
        plan_manifest_content_sha256=plan_manifest_sha256,
    )


def assert_shard_analysis_binding(
    binding_block: Mapping[str, Any],
    *,
    declared_sha256: str,
    analysis: PrimaryAnalysisBinding,
    label: str,
) -> str:
    """One shard's sealed analysis binding must reconstruct and match the gate."""

    reconstructed = sha256_json(dict(binding_block))
    if reconstructed != str(declared_sha256):
        _fail(
            f"{label} declares a primary_analysis_binding_sha256 that does not reconstruct from "
            "its own sealed binding block"
        )
    declared_files = binding_block.get("analysis_file_sha256")
    if not isinstance(declared_files, Mapping):
        _fail(f"{label} seals no analysis_file_sha256")
    if dict(sorted((str(k), str(v)) for k, v in declared_files.items())) != analysis.file_sha256:
        _fail(
            f"{label} was captured against a primary analysis whose files do not hash to the "
            f"bytes now at {analysis.analysis_dir}; the branch gate drifted after the capture"
        )
    if str(binding_block.get("receipt_content_sha256")) != analysis.receipt_content_sha256:
        _fail(f"{label} binds a different primary analysis receipt than the supplied gate")
    if str(binding_block.get("analyzer_source_sha256")) != analysis.analyzer_source_sha256:
        _fail(f"{label} binds a different analyzer revision than the supplied gate")
    if str(binding_block.get("plan_manifest_content_sha256")) != (
        analysis.plan_manifest_content_sha256
    ):
        _fail(f"{label} binds a primary analysis of another CPU plan")
    if binding_block.get("secondary_fields_present") is not False:
        _fail(f"{label} binds a primary analysis that already carried secondary fields")
    if binding_block.get("branch_labels_used_to_select_requests") is not False:
        _fail(f"{label} declares branch labels were used to select its requests")
    return reconstructed


# ---------------------------------------------------------------------------
# 4. The inherited smoke admission
# ---------------------------------------------------------------------------


def load_admission(
    admission_path: Path, *, plan: FrozenSecondaryPlan, analysis_binding_sha256: str
) -> dict[str, Any]:
    """The one secondary smoke admission every merged shard must have inherited."""

    admission = _read_json(Path(admission_path), "secondary smoke admission receipt")
    if str(admission.get("schema_version")) != secondary.ADMISSION_SCHEMA_VERSION:
        _fail(
            f"secondary admission schema {admission.get('schema_version')!r} is not "
            f"{secondary.ADMISSION_SCHEMA_VERSION!r}"
        )
    if str(admission.get("unit_id")) != UNIT_ID:
        _fail("secondary admission receipt belongs to another unit")
    assert_self_sealed(
        admission,
        digest_key="admission_content_sha256",
        label="secondary smoke admission receipt",
    )
    if admission.get("plan_manifest_content_sha256") != plan.manifest_content_sha256:
        _fail(
            "the secondary admission was sealed against a different plan manifest; it cannot "
            "admit a capture of this plan"
        )
    if str(admission.get("primary_analysis_binding_sha256")) != str(analysis_binding_sha256):
        _fail(
            "the secondary admission was sealed against a different primary analysis than the "
            "gate this merge binds"
        )
    if primary.runtime_identity_digest(admission) != str(
        admission.get("runtime_identity_sha256")
    ):
        _fail(
            "the secondary admission's runtime_identity_sha256 does not reconstruct from its own "
            "declared identity fields"
        )
    root_backend = admission.get("root_backend")
    if not isinstance(root_backend, Mapping) or any(
        root not in root_backend for root in PAIRED_ROOTS
    ):
        _fail("the secondary admission declares no backend for both paired roots")
    if not bool(admission.get("cache_admitted")) and any(
        str(value) != secondary.UNCACHED_BACKEND for value in root_backend.values()
    ):
        _fail("the secondary admission is internally inconsistent about its root backends")
    smoke_variants = admission.get("smoke_variants") or {}
    missing = sorted(
        variant
        for variant in SECONDARY_VARIANTS
        if int(smoke_variants.get(variant, 0)) < 1
    )
    if missing:
        _fail(
            f"the secondary admission proved no parity for variant(s) {missing!r}; one image "
            "session must carry the whole role matrix"
        )
    return admission


# ---------------------------------------------------------------------------
# 5. One shard: read, then prove
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardInput:
    """One capture shard directory, read verbatim before anything is proven."""

    shard_dir: Path
    receipt: dict[str, Any]
    rows: list[dict[str, Any]]
    parity: dict[str, Any]
    file_sha256: dict[str, str]


@dataclass(frozen=True)
class ValidatedShard:
    """One shard whose every declared identity has been re-proven."""

    shard_dir: Path
    shard_id: str
    session_image_id: str
    receipt: dict[str, Any]
    rows: list[dict[str, Any]]
    parity: dict[str, Any]
    file_sha256: dict[str, str]
    runtime_identity_sha256: str
    admission_content_sha256: str
    scorer_source_sha256: str
    source_identity_sha256: str
    request_ids: tuple[str, ...]
    counts_by_variant: dict[str, int]


def read_shard(shard_dir: Path) -> ShardInput:
    """Read one shard directory, refusing anything that is not a complete capture."""

    shard_dir = Path(shard_dir)
    if not shard_dir.is_dir():
        _fail(f"shard directory {shard_dir} does not exist")
    present = sorted(entry.name for entry in shard_dir.iterdir())
    if SHARD_QUARANTINE_NAME in present:
        _fail(
            f"shard {shard_dir} published {SHARD_QUARANTINE_NAME!r}: it was stopped by the "
            "quarantine rule and carries no secondary evidence"
        )
    if SHARD_ADMISSION_NAME in present:
        _fail(
            f"shard {shard_dir} published {SHARD_ADMISSION_NAME!r}: it is a smoke admission "
            "shard, not a capture shard"
        )
    missing = sorted(set(REQUIRED_SHARD_FILES) - set(present))
    if missing:
        _fail(f"shard {shard_dir} is incomplete; missing {missing!r}")
    unknown = sorted(set(present) - set(REQUIRED_SHARD_FILES))
    if unknown:
        _fail(
            f"shard {shard_dir} carries unknown artifact(s) {unknown!r}; an unrecognised file "
            "inside a sealed shard fails closed"
        )
    return ShardInput(
        shard_dir=shard_dir,
        receipt=_read_json(
            shard_dir / SHARD_RECEIPT_NAME, f"{shard_dir}/{SHARD_RECEIPT_NAME}"
        ),
        rows=_read_jsonl(shard_dir / SHARD_ROWS_NAME, f"{shard_dir}/{SHARD_ROWS_NAME}"),
        parity=_read_json(
            shard_dir / SHARD_PARITY_NAME, f"{shard_dir}/{SHARD_PARITY_NAME}"
        ),
        file_sha256={
            name: sha256_file(shard_dir / name) for name in sorted(REQUIRED_SHARD_FILES)
        },
    )


def _validate_shard_receipt(
    shard: ShardInput,
    *,
    plan: FrozenSecondaryPlan,
    analysis: PrimaryAnalysisBinding,
    expected_scorer_source_sha256: str | None,
) -> dict[str, Any]:
    receipt = shard.receipt
    label = f"shard receipt at {shard.shard_dir}"
    if str(receipt.get("schema_version")) != secondary.RECEIPT_SCHEMA_VERSION:
        _fail(
            f"{label} declares schema {receipt.get('schema_version')!r}, not "
            f"{secondary.RECEIPT_SCHEMA_VERSION!r}"
        )
    if str(receipt.get("unit_id")) != UNIT_ID:
        _fail(f"{label} belongs to another unit")
    if str(receipt.get("mode")) != secondary.MODE_CAPTURE:
        _fail(f"{label} is a {receipt.get('mode')!r} receipt, not a capture receipt")
    assert_self_sealed(receipt, digest_key="receipt_content_sha256", label=label)

    scorer_source = str(receipt.get("scorer_source_sha256"))
    if expected_scorer_source_sha256 is not None and (
        scorer_source != str(expected_scorer_source_sha256)
    ):
        _fail(
            f"{label} was produced by secondary scorer revision {scorer_source!r}, not the "
            f"pinned {expected_scorer_source_sha256!r}"
        )
    source_identity = receipt.get("source_identity")
    if not isinstance(source_identity, Mapping):
        _fail(f"{label} carries no source_identity")
    if str(source_identity.get(SECONDARY_SCORER_SOURCE_IDENTITY_KEY)) != scorer_source:
        _fail(
            f"{label} disagrees with itself about which secondary scorer produced it "
            f"({scorer_source!r} vs "
            f"{source_identity.get(SECONDARY_SCORER_SOURCE_IDENTITY_KEY)!r})"
        )
    unsealed = sorted(set(REQUIRED_SOURCE_IDENTITY_KEYS) - set(source_identity))
    if unsealed:
        _fail(f"{label} seals no source digest for module(s) {unsealed!r}")

    plan_block = receipt.get("plan")
    if not isinstance(plan_block, Mapping):
        _fail(f"{label} carries no plan block")
    if str(plan_block.get("manifest_schema_version")) != (
        FROZEN_PLAN_MANIFEST_SCHEMA_VERSION
    ):
        _fail(f"{label} was captured against a plan of another schema revision")
    if str(plan_block.get("manifest_content_sha256")) != plan.manifest_content_sha256:
        _fail(
            f"{label} was captured against plan manifest "
            f"{plan_block.get('manifest_content_sha256')!r}, not the frozen "
            f"{plan.manifest_content_sha256!r}"
        )
    declared_plan_files = plan_block.get("plan_file_sha256")
    if not isinstance(declared_plan_files, Mapping):
        _fail(f"{label} declares no plan_file_sha256")
    if dict(sorted((str(k), str(v)) for k, v in declared_plan_files.items())) != dict(
        sorted(plan.plan_file_sha256.items())
    ):
        _fail(
            f"{label} declares plan file digests that do not match the frozen plan's own sealed "
            "bytes"
        )

    runtime_identity = receipt.get("runtime_identity")
    if not isinstance(runtime_identity, Mapping):
        _fail(f"{label} carries no runtime_identity")
    if primary.runtime_identity_digest(runtime_identity) != str(
        receipt.get("runtime_identity_sha256")
    ):
        _fail(
            f"{label} declares a runtime_identity_sha256 that does not reconstruct from its own "
            "identity fields; the receipt is internally inconsistent"
        )
    runtime_sources = runtime_identity.get("source_identity")
    if not isinstance(runtime_sources, Mapping):
        _fail(f"{label} carries no runtime_identity.source_identity")
    if str(runtime_sources.get(SECONDARY_SCORER_SOURCE_IDENTITY_KEY)) != scorer_source:
        _fail(
            f"{label} ran under secondary scorer source identity "
            f"{runtime_sources.get(SECONDARY_SCORER_SOURCE_IDENTITY_KEY)!r}, not the revision its "
            "own receipt declares"
        )
    if not bool(runtime_identity.get("is_real_model")):
        _fail(
            f"{label} was captured against a backend that is not a real model; a simulated "
            "capture is never merged into conclusion-bearing evidence"
        )
    if not bool(runtime_identity.get("usable_as_evidence")):
        _fail(f"{label} declares its own runtime is not usable as evidence")

    policy = receipt.get("policy")
    if not isinstance(policy, Mapping):
        _fail(f"{label} carries no policy block")
    for key, expected in sorted(REQUIRED_POLICY.items()):
        if policy.get(key) != expected:
            _fail(
                f"{label} declares policy {key}={policy.get(key)!r}, not the required "
                f"{expected!r}"
            )

    quarantine = receipt.get("quarantine")
    if not isinstance(quarantine, Mapping):
        _fail(f"{label} carries no quarantine ledger")
    if int(quarantine.get("count", -1)) != 0 or list(quarantine.get("entries") or []):
        _fail(
            f"{label} declares {quarantine.get('count')!r} quarantined request(s); any "
            "quarantined request withholds the whole shard's evidence"
        )

    binding_block = receipt.get("primary_analysis")
    if not isinstance(binding_block, Mapping):
        _fail(f"{label} carries no sealed primary_analysis binding")
    assert_shard_analysis_binding(
        binding_block,
        declared_sha256=str(receipt.get("primary_analysis_binding_sha256")),
        analysis=analysis,
        label=label,
    )

    admission_block = receipt.get("admission")
    if not isinstance(admission_block, Mapping):
        _fail(f"{label} carries no inherited admission block")
    admission_digest = admission_block.get("admission_content_sha256")
    if not isinstance(admission_digest, str) or not admission_digest:
        _fail(f"{label} inherited no admission digest")

    return {
        "scorer_source_sha256": scorer_source,
        "source_identity_sha256": sha256_json(dict(source_identity)),
        "runtime_identity_sha256": str(receipt["runtime_identity_sha256"]),
        "admission_content_sha256": str(admission_digest),
    }


def _validate_shard_parity(
    shard: ShardInput, *, shard_id: str, session_image_id: str, admission: Mapping[str, Any]
) -> None:
    parity = shard.parity
    if str(parity.get("schema_version")) != secondary.PARITY_SCHEMA_VERSION:
        _fail(f"the parity file in shard {shard_id} declares an unknown schema")
    if str(parity.get("unit_id")) != UNIT_ID:
        _fail(f"the parity file in shard {shard_id} belongs to another unit")
    if str(parity.get("shard_id")) != shard_id:
        _fail(f"the parity file in shard {shard_id} was published by a different shard")
    if str(parity.get("session_image_id")) != session_image_id:
        _fail(f"the parity file in shard {shard_id} names a different image session")
    inherited = parity.get("inherited_admission")
    if not isinstance(inherited, Mapping):
        _fail(f"the parity file in shard {shard_id} inherits no admission")
    if inherited.get("admission_content_sha256") != shard.receipt["admission"].get(
        "admission_content_sha256"
    ):
        _fail(
            f"shard {shard_id} disagrees with itself about which smoke admission it inherited"
        )
    if inherited.get("admission_content_sha256") != admission.get(
        "admission_content_sha256"
    ):
        _fail(
            f"the parity file in shard {shard_id} inherits an admission other than this merged "
            "run's"
        )
    root_backend = parity.get("root_backend")
    if not isinstance(root_backend, Mapping) or any(
        root not in root_backend for root in PAIRED_ROOTS
    ):
        _fail(f"the parity file in shard {shard_id} declares no backend for both paired roots")


# ---------------------------------------------------------------------------
# 6. One row: re-prove the whole paired readout
# ---------------------------------------------------------------------------


def _assert_root_payload(
    payload: Mapping[str, Any],
    *,
    root: str,
    context_id: str,
    scored_token_ids: Sequence[int],
    segments: secondary.RowSegments,
    appended_sha256: str,
    appended_count: int,
    gt_owner_id: str,
    variant: str,
    label: str,
) -> dict[str, Any]:
    """Re-prove one root's identity, per-token stream and per-segment sums."""

    if str(payload.get("root")) != root:
        _fail(f"{label}: root payload {payload.get('root')!r} is filed under {root!r}")
    if str(payload.get("context_id")) != str(context_id):
        _fail(
            f"{label}: root {root!r} declares context {payload.get('context_id')!r}, not the "
            f"sealed {context_id!r}"
        )
    if str(payload.get("appended_token_ids_sha256")) != str(appended_sha256):
        _fail(
            f"{label}: root {root!r} appended tokens hash to "
            f"{payload.get('appended_token_ids_sha256')!r}, not the sealed {appended_sha256!r}"
        )
    if int(payload.get("appended_token_count", -1)) != int(appended_count):
        _fail(
            f"{label}: root {root!r} appended {payload.get('appended_token_count')!r} tokens, not "
            f"the sealed {appended_count}"
        )
    prefix_count = int(payload.get("executed_prefix_token_count", -1))
    if prefix_count < 1:
        _fail(f"{label}: root {root!r} declares no executed prefix")
    if int(payload.get("root_token_count", -1)) != prefix_count + int(appended_count):
        _fail(
            f"{label}: root {root!r} root_token_count does not equal its own executed prefix "
            "plus its appended tokens"
        )
    scoring_backend = str(payload.get("scoring_backend"))
    if scoring_backend not in (secondary.KV_CACHE_BACKEND, secondary.UNCACHED_BACKEND):
        _fail(f"{label}: root {root!r} declares unknown scoring backend {scoring_backend!r}")
    expected_group = secondary._root_group_id(  # noqa: SLF001
        gt_owner_id=gt_owner_id,
        context_id=str(context_id),
        variant=variant,
        root=root,
        appended_digest=str(appended_sha256),
        scoring_backend=scoring_backend,
    )
    if str(payload.get("context_group_id")) != expected_group:
        _fail(
            f"{label}: root {root!r} declares logical context group "
            f"{payload.get('context_group_id')!r}, which does not reconstruct from its own owner, "
            "context, variant, appended digest and backend"
        )

    tokens = [int(value) for value in scored_token_ids]
    selected = _floats(
        payload.get("selected_logprobs"), label=f"{label} root {root!r} selected logprobs"
    )
    argmax_ids = _token_ids(
        payload.get("argmax_token_ids"), label=f"{label} root {root!r} argmax token ids"
    )
    argmax_logprobs = _floats(
        payload.get("argmax_logprobs"), label=f"{label} root {root!r} argmax logprobs"
    )
    for name, observed in (
        ("selected_logprobs", len(selected)),
        ("argmax_token_ids", len(argmax_ids)),
        ("argmax_logprobs", len(argmax_logprobs)),
    ):
        if observed != len(tokens):
            _fail(
                f"{label}: root {root!r} carries {observed} {name} for {len(tokens)} scored "
                "tokens"
            )
    declared_rank = payload.get("selected_is_argmax")
    if not isinstance(declared_rank, Sequence) or isinstance(declared_rank, (str, bytes)):
        _fail(f"{label}: root {root!r} carries no selected_is_argmax stream")
    expected_rank = [
        int(token) == int(argmax) for token, argmax in zip(tokens, argmax_ids, strict=True)
    ]
    if [bool(value) for value in declared_rank] != expected_rank:
        _fail(
            f"{label}: root {root!r} selected_is_argmax does not reconstruct from its own scored "
            "and argmax token ids"
        )
    if bool(payload.get("argmax_reproduces_complete_row")) != (tokens == argmax_ids):
        _fail(
            f"{label}: root {root!r} argmax_reproduces_complete_row contradicts its own token "
            "streams"
        )
    expected_replay = primary.replay_argmax_through_prefix(
        expected_token_ids=tokens,
        argmax_token_ids=argmax_ids,
        up_to_index=segments.description[1],
    )
    if bool(payload.get("argmax_reproduces_description_path")) != bool(expected_replay):
        _fail(
            f"{label}: root {root!r} argmax_reproduces_description_path contradicts its own token "
            "streams"
        )

    declared_sums = payload.get("segment_sums")
    if not isinstance(declared_sums, Mapping):
        _fail(f"{label}: root {root!r} carries no segment_sums")
    expected_sums = secondary.segment_sums(
        segments, selected, label=f"{label} root {root!r}"
    )
    if {key: dict(value) for key, value in declared_sums.items()} != expected_sums:
        _fail(
            f"{label}: root {root!r} per-segment sums do not reconstruct from its own selected "
            "log probabilities"
        )
    for key in ("request_identity_sha256", "output_identity_sha256"):
        secondary._hex64(payload.get(key), label=f"{label} root {root!r} {key}")  # noqa: SLF001
    return expected_sums


def validate_row(
    row: Mapping[str, Any],
    *,
    plan: FrozenSecondaryPlan,
    shard_id: str,
    session_image_id: str,
    analysis_binding_sha256: str,
) -> str:
    """Re-prove one published secondary row against the sealed plan, or fail."""

    request_id = str(row.get("request_id"))
    label = f"secondary row {request_id!r} in shard {shard_id}"
    if str(row.get("schema_version")) != secondary.SCHEMA_VERSION:
        _fail(f"{label} declares schema {row.get('schema_version')!r}, not the frozen one")
    if str(row.get("unit_id")) != UNIT_ID:
        _fail(f"{label} belongs to another unit")
    if str(row.get("shard_id")) != shard_id:
        _fail(f"{label} was published by a different shard")
    for key, expected in sorted(REQUIRED_ROW_POLICY.items()):
        if row.get(key) != expected:
            _fail(f"{label} declares {key}={row.get(key)!r}, not the required {expected!r}")
    stratum = row.get("repetition_penalty_stratum")
    if isinstance(stratum, bool) or not isinstance(stratum, (int, float)):
        _fail(f"{label} declares no numeric repetition-penalty stratum")
    if float(stratum) != float(secondary.NATIVE_REPETITION_PENALTY_STRATUM):
        _fail(f"{label} was scored under another repetition-penalty stratum")
    if str(row.get("primary_analysis_binding_sha256")) != str(analysis_binding_sha256):
        _fail(f"{label} binds a primary analysis other than this merged run's gate")

    request = plan.requests_by_id.get(request_id)
    if request is None:
        _fail(
            f"{label} is not one of the sealed plan's {len(plan.requests_by_id)} secondary "
            "requests; an unknown row fails closed"
        )
    for key in ("variant", "cohort", "gt_owner_id", "image_id", "request_key"):
        if str(row.get(key)) != str(request.get(key)):
            _fail(
                f"{label} declares {key}={row.get(key)!r} but its sealed plan request declares "
                f"{request.get(key)!r}"
            )
    if bool(row.get("plan_optional")) is not bool(request.get("optional")):
        _fail(f"{label} disagrees with the sealed plan about its optionality")
    if str(row.get("plan_identity_digest")) != str(request.get("identity_digest")):
        _fail(f"{label} declares another request identity than the sealed plan's")

    variant = str(row["variant"])
    if str(row.get("cohort")) != secondary.VARIANT_COHORT[variant]:
        _fail(
            f"{label} attributes variant {variant!r} to cohort {row.get('cohort')!r}, not the "
            f"sealed {secondary.VARIANT_COHORT[variant]!r}"
        )
    if bool(row.get("plan_optional")) is not secondary.VARIANT_IS_PLAN_OPTIONAL[variant]:
        _fail(f"{label} declares an optionality the variant contract forbids")
    image_id = str(row.get("image_id"))
    if image_id != str(session_image_id) or str(row.get("session_image_id")) != str(
        session_image_id
    ):
        _fail(
            f"{label} carries image {image_id!r} inside the {session_image_id!r} session shard; "
            "one image session per shard is a contract"
        )
    _assert_request_key(row, label=label)

    gt_owner_id = str(row["gt_owner_id"])
    registry_row = plan.registry_by_owner.get(gt_owner_id)
    if registry_row is None:
        _fail(f"{label} names owner {gt_owner_id!r}, which is in neither sealed registry")
    if str(registry_row.get("image_id")) != image_id:
        _fail(f"{label} joins an owner registered on another image")

    # --- exact scored target -------------------------------------------------
    target = request["scored_target"]
    tokens = _token_ids(row.get("scored_token_ids"), label=f"{label} scored token ids")
    if tokens != _token_ids(target["token_ids"], label="sealed scored target"):
        _fail(f"{label} scores tokens the sealed plan request does not")
    digest = sha256_json(tokens)
    if digest != str(row.get("scored_token_ids_sha256")) or digest != str(
        target.get("token_ids_sha256")
    ):
        _fail(f"{label} scored-token digest does not reconstruct from its own tokens")
    if int(row.get("scored_token_count", -1)) != len(tokens):
        _fail(f"{label} declares a scored token count that contradicts its own tokens")
    if str(row.get("scored_target_kind")) != str(target.get("kind")):
        _fail(f"{label} declares another scored-target kind than the sealed plan's")
    native_row_index = int(row.get("native_row_index", -1))
    if native_row_index != int(target.get("native_row_index", -2)):
        _fail(f"{label} declares another native row index than the sealed plan's")

    registry_block = registry_row.get(VARIANT_REGISTRY_ROW_KEY[variant])
    if not isinstance(registry_block, Mapping):
        _fail(
            f"{label} at variant {variant!r} has no sealed "
            f"{VARIANT_REGISTRY_ROW_KEY[variant]!r} in its registry row"
        )
    if str(registry_block.get(VARIANT_REGISTRY_DIGEST_KEY[variant])) != digest:
        _fail(
            f"{label} scores a row that is not the owner's sealed "
            f"{VARIANT_REGISTRY_ROW_KEY[variant]!r} token sequence"
        )
    if int(registry_block.get("row_index", -2)) != native_row_index:
        _fail(
            f"{label} scores native row {native_row_index}, not the sealed "
            f"{VARIANT_REGISTRY_ROW_KEY[variant]!r} row index"
        )

    # --- literal segment spans ----------------------------------------------
    segments = secondary.row_segments(tokens)
    declared_segments = row.get("segments")
    if not isinstance(declared_segments, Mapping) or set(declared_segments) != set(SEGMENTS):
        _fail(f"{label} does not carry exactly the three frozen row segments")
    for segment in SEGMENTS:
        start, stop = segments.span(segment)
        block = declared_segments[segment]
        if not isinstance(block, Mapping):
            _fail(f"{label} segment {segment!r} is not an object")
        if int(block.get("token_index_start", -1)) != start or int(
            block.get("token_index_stop", -1)
        ) != stop:
            _fail(
                f"{label} segment {segment!r} declares span "
                f"({block.get('token_index_start')!r}, {block.get('token_index_stop')!r}), not the "
                f"literal ({start}, {stop})"
            )
        if _token_ids(
            block.get("token_ids"), label=f"{label} segment {segment!r}"
        ) != tokens[start:stop]:
            _fail(f"{label} segment {segment!r} tokens are not the row's own literal span")

    # --- paired roots --------------------------------------------------------
    modified_context_id = str(row.get("modified_context_id"))
    if modified_context_id != str(request.get("context_id")):
        _fail(f"{label} appends at a context the sealed plan request does not")
    sealed_baseline, baseline_source = secondary._sealed_baseline_context(  # noqa: SLF001
        registry_row, variant=variant, gt_owner_id=gt_owner_id
    )
    baseline_context_id = str(row.get("baseline_context_id"))
    if baseline_context_id != sealed_baseline:
        _fail(
            f"{label} pairs against baseline context {baseline_context_id!r}, not the sealed "
            f"{sealed_baseline!r}"
        )
    if str(row.get("baseline_context_source")) != baseline_source:
        _fail(
            f"{label} attributes its baseline to {row.get('baseline_context_source')!r}, not the "
            f"sealed {baseline_source!r}"
        )
    successor_context_id = str(row.get("successor_context_id"))
    if successor_context_id == baseline_context_id:
        _fail(f"{label} names its own baseline boundary as that boundary's successor")
    if variant == secondary.VARIANT_BENIGN_SUBSTITUTION:
        if modified_context_id == baseline_context_id:
            _fail(
                f"{label} is a benign substitution but appends at the very boundary it pairs "
                "against; the clean twin must replace the native row between them"
            )
    else:
        if modified_context_id != baseline_context_id:
            _fail(
                f"{label} at variant {variant!r} must append to the same native boundary it "
                f"scores against ({baseline_context_id!r}), not {modified_context_id!r}"
            )
        if str(registry_block.get("post_row_context_id")) != successor_context_id:
            _fail(
                f"{label} names successor {successor_context_id!r}, not the sealed "
                f"{VARIANT_REGISTRY_ROW_KEY[variant]!r} post-row boundary"
            )

    inserted = registry_row.get("inserted_clean_row_c")
    if not isinstance(inserted, Mapping):
        _fail(f"{label} names an owner that seals no inserted clean GT row")
    appended_sha256 = str(request["prefix"]["appended_token_ids_sha256"])
    if appended_sha256 != str(inserted.get("token_ids_sha256")):
        _fail(f"{label} appends a row that is not the owner's sealed clean GT row")
    if str(row.get("inserted_clean_row_c_token_ids_sha256")) != appended_sha256:
        _fail(f"{label} declares another inserted clean row than the sealed plan request's")
    appended_count = int(request["prefix"]["appended_token_count"])
    if int(row.get("inserted_clean_row_c_token_count", -1)) != appended_count:
        _fail(f"{label} declares an inserted-row token count the sealed plan does not")

    roots = row.get("roots")
    if not isinstance(roots, Mapping) or set(roots) != set(PAIRED_ROOTS):
        _fail(f"{label} does not carry exactly the two paired roots")
    sums: dict[str, dict[str, Any]] = {}
    for root, context_id, root_append_sha256, root_append_count in (
        (ROOT_BASELINE, baseline_context_id, EMPTY_APPEND_SHA256, 0),
        (ROOT_MODIFIED, modified_context_id, appended_sha256, appended_count),
    ):
        sums[root] = _assert_root_payload(
            roots[root],
            root=root,
            context_id=context_id,
            scored_token_ids=tokens,
            segments=segments,
            appended_sha256=root_append_sha256,
            appended_count=root_append_count,
            gt_owner_id=gt_owner_id,
            variant=variant,
            label=label,
        )
    baseline_prefix = str(roots[ROOT_BASELINE].get("executed_prefix_token_ids_sha256"))
    modified_prefix = str(roots[ROOT_MODIFIED].get("executed_prefix_token_ids_sha256"))
    if variant == secondary.VARIANT_BENIGN_SUBSTITUTION:
        if baseline_prefix == modified_prefix:
            _fail(
                f"{label} forwards one executed prefix from two different sealed boundaries; the "
                "benign substitution's roots are not the same context"
            )
    elif baseline_prefix != modified_prefix:
        _fail(
            f"{label} forwards two different executed prefixes from one sealed boundary; the "
            "paired roots would differ by more than the inserted clean row"
        )

    # --- modified-minus-baseline arithmetic ----------------------------------
    declared_deltas = row.get("deltas")
    if not isinstance(declared_deltas, Mapping) or set(declared_deltas) != set(SEGMENTS):
        _fail(f"{label} does not carry exactly the three per-segment deltas")
    expected_deltas = secondary.paired_deltas(sums[ROOT_BASELINE], sums[ROOT_MODIFIED])
    if {key: dict(value) for key, value in declared_deltas.items()} != expected_deltas:
        _fail(
            f"{label} per-segment deltas do not reconstruct as modified-minus-baseline over its "
            "own per-segment sums"
        )
    return request_id


def validate_shard(
    shard: ShardInput,
    *,
    plan: FrozenSecondaryPlan,
    analysis: PrimaryAnalysisBinding,
    analysis_binding_sha256: str,
    admission: Mapping[str, Any],
    expected_scorer_source_sha256: str | None,
) -> ValidatedShard:
    """Re-prove every identity one shard declares, then accept its raw rows."""

    identities = _validate_shard_receipt(
        shard,
        plan=plan,
        analysis=analysis,
        expected_scorer_source_sha256=expected_scorer_source_sha256,
    )
    if identities["admission_content_sha256"] != str(
        admission.get("admission_content_sha256")
    ):
        _fail(
            f"shard at {shard.shard_dir} inherited admission "
            f"{identities['admission_content_sha256']!r}, not the merged run's "
            f"{admission.get('admission_content_sha256')!r}"
        )
    if identities["runtime_identity_sha256"] != str(
        admission.get("runtime_identity_sha256")
    ):
        _fail(
            f"shard at {shard.shard_dir} ran under a runtime the smoke admission never admitted; "
            "cached execution is never inherited across a changed runtime"
        )

    shard_id = str(shard.receipt.get("shard_id"))
    executed = shard.receipt.get("executed")
    if not isinstance(executed, Mapping):
        _fail(f"shard receipt at {shard.shard_dir} carries no executed block")
    session_image_id = str(executed.get("session_image_id"))
    if session_image_id not in FROZEN_IMAGE_IDS:
        _fail(
            f"shard {shard_id} scored image {session_image_id!r}, which is outside this unit's "
            "frozen twelve"
        )

    if int(executed.get("row_count", -1)) != len(shard.rows):
        _fail(
            f"shard {shard_id} published {len(shard.rows)} secondary rows but its receipt sealed "
            f"{executed.get('row_count')!r}"
        )
    request_ids: list[str] = []
    for row in shard.rows:
        request_ids.append(
            validate_row(
                row,
                plan=plan,
                shard_id=shard_id,
                session_image_id=session_image_id,
                analysis_binding_sha256=analysis_binding_sha256,
            )
        )
    if len(set(request_ids)) != len(request_ids):
        _fail(f"shard {shard_id} published the same secondary request twice")
    if sha256_json(sorted(request_ids)) != str(executed.get("request_ids_sha256")):
        _fail(
            f"shard {shard_id} published rows whose request ids do not reproduce the digest its "
            "own receipt sealed"
        )
    declared_owners = sorted(str(value) for value in (executed.get("gt_owner_ids") or ()))
    if declared_owners != sorted({str(row["gt_owner_id"]) for row in shard.rows}):
        _fail(
            f"shard {shard_id} published owner ids that do not match the set its receipt sealed"
        )

    expected_requests = plan.requests_for_image(session_image_id)
    expected_ids = sorted(str(row["request_id"]) for row in expected_requests)
    if sorted(request_ids) != expected_ids:
        missing = sorted(set(expected_ids) - set(request_ids))
        extra = sorted(set(request_ids) - set(expected_ids))
        _fail(
            f"shard {shard_id} does not publish exactly image {session_image_id!r}'s sealed "
            f"secondary requests (missing={missing!r}, unexpected={extra!r})"
        )
    observed_counts = secondary.counts_by_variant(shard.rows)
    expected_counts = secondary.counts_by_variant(expected_requests)
    if observed_counts != expected_counts:
        _fail(
            f"shard {shard_id} published {observed_counts!r} secondary rows for image "
            f"{session_image_id!r}, not the sealed {expected_counts!r}"
        )
    sealed_image_counts = shard.receipt.get("secondary_request_counts", {}).get("image")
    if not isinstance(sealed_image_counts, Mapping):
        _fail(f"shard {shard_id} seals no per-image secondary request census")
    if str(sealed_image_counts.get("image_id")) != session_image_id:
        _fail(f"shard {shard_id} seals a per-image census of another image")
    for key in ("expected_by_variant", "observed_by_variant"):
        block = sealed_image_counts.get(key)
        if not isinstance(block, Mapping) or {
            str(name): int(value) for name, value in block.items()
        } != expected_counts:
            _fail(
                f"shard {shard_id} seals {key}={block!r}, not the sealed plan's "
                f"{expected_counts!r}"
            )

    _validate_shard_parity(
        shard,
        shard_id=shard_id,
        session_image_id=session_image_id,
        admission=admission,
    )

    return ValidatedShard(
        shard_dir=shard.shard_dir,
        shard_id=shard_id,
        session_image_id=session_image_id,
        receipt=shard.receipt,
        rows=shard.rows,
        parity=shard.parity,
        file_sha256=shard.file_sha256,
        runtime_identity_sha256=identities["runtime_identity_sha256"],
        admission_content_sha256=identities["admission_content_sha256"],
        scorer_source_sha256=identities["scorer_source_sha256"],
        source_identity_sha256=identities["source_identity_sha256"],
        request_ids=tuple(sorted(request_ids)),
        counts_by_variant=observed_counts,
    )


# ---------------------------------------------------------------------------
# 7. Cross-shard closure
# ---------------------------------------------------------------------------


def _assert_one_shard_per_frozen_image(shards: Sequence[ValidatedShard]) -> None:
    by_image: dict[str, list[str]] = {}
    for shard in shards:
        by_image.setdefault(shard.session_image_id, []).append(shard.shard_id)
    duplicated = sorted(image for image, ids in by_image.items() if len(ids) > 1)
    if duplicated:
        _fail(
            f"image(s) {duplicated!r} were merged from more than one shard; exactly one "
            "successful shard per sealed plan image is a contract"
        )
    missing = sorted(set(FROZEN_IMAGE_IDS) - set(by_image))
    if missing:
        _fail(
            f"no successful shard was supplied for image(s) {missing!r}; a partial merge is "
            "never published"
        )
    if len(shards) != EXPECTED_SHARD_COUNT:
        _fail(f"{len(shards)} shards were supplied for {EXPECTED_SHARD_COUNT} frozen images")
    shard_ids = [shard.shard_id for shard in shards]
    if len(set(shard_ids)) != len(shard_ids):
        _fail("two shards declare the same shard_id; a duplicated capture fails closed")


def _assert_one_identity(shards: Sequence[ValidatedShard]) -> dict[str, str]:
    identities: dict[str, str] = {}
    for label, values in (
        ("runtime identities", {shard.runtime_identity_sha256 for shard in shards}),
        ("smoke admissions", {shard.admission_content_sha256 for shard in shards}),
        ("secondary scorer revisions", {shard.scorer_source_sha256 for shard in shards}),
        ("source identity maps", {shard.source_identity_sha256 for shard in shards}),
        (
            "primary analysis bindings",
            {
                str(shard.receipt.get("primary_analysis_binding_sha256"))
                for shard in shards
            },
        ),
    ):
        if len(values) != 1:
            _fail(
                f"the supplied shards span {len(values)} {label} {sorted(values)!r}; evidence "
                "from mixed identities is never merged into one conclusion"
            )
        identities[label] = sorted(values)[0]
    return identities


def _assert_request_closure(
    shards: Sequence[ValidatedShard], *, plan: FrozenSecondaryPlan
) -> dict[str, Any]:
    """The union of published rows is exactly the sealed 64-request census."""

    request_ids: list[str] = []
    for shard in shards:
        request_ids.extend(shard.request_ids)
    if len(set(request_ids)) != len(request_ids):
        duplicated = sorted({value for value in request_ids if request_ids.count(value) > 1})
        _fail(f"secondary request(s) {duplicated!r} were published by more than one shard")
    expected = list(plan.request_ids)
    if sorted(request_ids) != expected:
        missing = sorted(set(expected) - set(request_ids))
        extra = sorted(set(request_ids) - set(expected))
        _fail(
            f"the merged secondary evidence is not the sealed plan's {len(expected)} requests "
            f"(missing={missing!r}, unexpected={extra!r})"
        )

    observed = {variant: 0 for variant in SECONDARY_VARIANTS}
    for shard in shards:
        for variant, count in shard.counts_by_variant.items():
            observed[variant] += count
    if observed != dict(secondary.EXPECTED_REQUEST_COUNT_BY_VARIANT):
        _fail(
            f"the merged secondary evidence holds {observed!r} rows, not the frozen "
            f"{dict(secondary.EXPECTED_REQUEST_COUNT_BY_VARIANT)!r}"
        )
    return {
        "request_ids_sha256": sha256_json(sorted(request_ids)),
        "request_count": len(request_ids),
        "counts_by_variant": dict(sorted(observed.items())),
    }


def _assert_cohort_closure(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Owner closure per cohort, with the optional ``F`` presence taken from the rows."""

    by_variant: dict[str, list[str]] = {variant: [] for variant in SECONDARY_VARIANTS}
    for row in rows:
        by_variant[str(row["variant"])].append(str(row["gt_owner_id"]))
    for variant, owner_ids in by_variant.items():
        if len(set(owner_ids)) != len(owner_ids):
            _fail(f"variant {variant!r} carries the same owner twice across shards")

    primary_owners = sorted(by_variant[secondary.VARIANT_P_PLUS_C_THEN_E])
    f_owners = sorted(by_variant[secondary.VARIANT_P_PLUS_E_PLUS_C_THEN_F])
    tp_owners = sorted(by_variant[secondary.VARIANT_BENIGN_SUBSTITUTION])
    if len(primary_owners) != primary.PRIMARY_OWNER_COUNT_U:
        _fail(
            f"the merged primary compatibility readout covers {len(primary_owners)} owners, not "
            f"the frozen {primary.PRIMARY_OWNER_COUNT_U}"
        )
    if len(tp_owners) != primary.TP_CALIBRATION_OWNER_COUNT:
        _fail(
            f"the merged benign-substitution reference covers {len(tp_owners)} owners, not the "
            f"frozen {primary.TP_CALIBRATION_OWNER_COUNT}"
        )
    stray = sorted(set(f_owners) - set(primary_owners))
    if stray:
        _fail(
            f"the optional F readout covers owner(s) {stray!r} that carry no primary "
            "compatibility readout"
        )
    overlap = sorted(set(primary_owners) & set(tp_owners))
    if overlap:
        _fail(
            f"owner(s) {overlap!r} appear in both the primary cohort and the TP replay control "
            "cohort; a mismatch is reported, never patched by allowing overlap"
        )
    return {
        "primary_owner_ids": primary_owners,
        "primary_owner_count": len(primary_owners),
        "optional_f_owner_ids": f_owners,
        "optional_f_owner_count": len(f_owners),
        "tp_replay_control_owner_ids": tp_owners,
        "tp_replay_control_owner_count": len(tp_owners),
    }


# ---------------------------------------------------------------------------
# 8. Deterministic merged bytes
# ---------------------------------------------------------------------------


def _row_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("image_id")),
        str(row.get("gt_owner_id")),
        str(row.get("variant")),
        str(row.get("request_id")),
    )


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def merge_shards(
    *,
    shard_dirs: Sequence[Path],
    plan_dir: Path,
    primary_analysis_dir: Path,
    admission_path: Path,
    expect_scorer_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Prove every shard, then build the merged bytes and the merge receipt."""

    plan = load_frozen_plan(plan_dir)
    analysis = load_primary_analysis_binding(primary_analysis_dir, plan=plan)

    seen_dirs: set[Path] = set()
    read: list[ShardInput] = []
    for shard_dir in shard_dirs:
        resolved = Path(shard_dir).resolve()
        if resolved in seen_dirs:
            _fail(f"shard directory {resolved} was supplied twice")
        seen_dirs.add(resolved)
        read.append(read_shard(resolved))
    if not read:
        _fail("no capture shard was supplied")

    # The binding digest is owned by the shards: they sealed the binding block
    # this merge re-proves against the analysis directory's own bytes.
    binding_digests = {
        str(shard.receipt.get("primary_analysis_binding_sha256")) for shard in read
    }
    if len(binding_digests) != 1:
        _fail(
            f"the supplied shards span {len(binding_digests)} primary analysis bindings "
            f"{sorted(binding_digests)!r}"
        )
    analysis_binding_sha256 = sorted(binding_digests)[0]
    admission = load_admission(
        admission_path, plan=plan, analysis_binding_sha256=analysis_binding_sha256
    )

    validated = [
        validate_shard(
            shard,
            plan=plan,
            analysis=analysis,
            analysis_binding_sha256=analysis_binding_sha256,
            admission=admission,
            expected_scorer_source_sha256=expect_scorer_source_sha256,
        )
        for shard in read
    ]
    validated.sort(key=lambda shard: (shard.session_image_id, shard.shard_id))
    _assert_one_shard_per_frozen_image(validated)
    _assert_one_identity(validated)
    request_closure = _assert_request_closure(validated, plan=plan)

    rows = sorted(
        (row for shard in validated for row in shard.rows), key=_row_sort_key
    )
    cohort_closure = _assert_cohort_closure(rows)
    parity_rows = [shard.parity for shard in validated]

    rows_bytes = _jsonl_bytes(rows)
    parity_bytes = _jsonl_bytes(parity_rows)

    receipt: dict[str, Any] = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "merger_source_sha256": sha256_file(Path(__file__).resolve()),
        "secondary_scorer_source_sha256": validated[0].scorer_source_sha256,
        "scorer_source_pinned": expect_scorer_source_sha256 is not None,
        "source_identity": dict(validated[0].receipt["source_identity"]),
        "artifact_determinism": "no_wall_clock_or_host_specific_field_is_sealed",
        "plan": {
            "plan_dir": str(plan.plan_dir),
            "manifest_schema_version": FROZEN_PLAN_MANIFEST_SCHEMA_VERSION,
            "manifest_content_sha256": plan.manifest_content_sha256,
            "builder_source_sha256": primary.builder_source_sha256(plan.manifest),
            "plan_file_sha256": dict(sorted(plan.plan_file_sha256.items())),
            "lineage": plan.manifest.get("lineage"),
            "cohort_counts": plan.manifest.get("cohort_counts"),
            "control_counts": plan.manifest.get("control_counts"),
            "secondary_request_counts": dict(plan.counts),
        },
        # Identity only: no branch label, branch count or routing decision of the
        # sealed primary analysis is read here or carried downstream.
        "primary_analysis": {
            "analysis_dir": str(analysis.analysis_dir),
            "analysis_file_sha256": dict(analysis.file_sha256),
            "receipt_content_sha256": analysis.receipt_content_sha256,
            "analyzer_source_sha256": analysis.analyzer_source_sha256,
            "plan_manifest_content_sha256": analysis.plan_manifest_content_sha256,
            "binding_sha256": analysis_binding_sha256,
            "branch_labels_read_by_this_merge": False,
        },
        "primary_analysis_binding_sha256": analysis_binding_sha256,
        "admission": {
            "path": str(Path(admission_path)),
            "admission_content_sha256": str(admission.get("admission_content_sha256")),
            "smoke_shard_id": admission.get("smoke_shard_id"),
            "smoke_image_id": admission.get("smoke_image_id"),
            "cache_admitted": admission.get("cache_admitted"),
            "root_backend": dict(sorted((admission.get("root_backend") or {}).items())),
        },
        "runtime_identity_sha256": validated[0].runtime_identity_sha256,
        "runtime_identity": dict(validated[0].receipt["runtime_identity"]),
        "secondary_requests": {
            **request_closure,
            "expected_by_variant": dict(secondary.EXPECTED_REQUEST_COUNT_BY_VARIANT),
            "expected_total": secondary.EXPECTED_SECONDARY_REQUEST_COUNT,
            "plan_request_ids_sha256": plan.request_ids_sha256,
        },
        "observed_cohorts": dict(cohort_closure),
        "shards": [
            {
                "shard_id": shard.shard_id,
                "session_image_id": shard.session_image_id,
                "shard_dir": str(shard.shard_dir),
                "file_sha256": dict(sorted(shard.file_sha256.items())),
                "receipt_content_sha256": str(shard.receipt["receipt_content_sha256"]),
                "row_count": len(shard.rows),
                "counts_by_variant": dict(sorted(shard.counts_by_variant.items())),
                "request_ids_sha256": sha256_json(list(shard.request_ids)),
            }
            for shard in validated
        ],
        "policy": {
            "rows_preserved_verbatim": True,
            "merged_readout_tier": secondary.SECONDARY_READOUT_TIER,
            "primary_branch_assignment_performed": False,
            "primary_branch_labels_read": False,
            "quarantine_evidence_merged": False,
            "threshold_fitting": False,
            "compatibility_verdict_assigned": False,
            "claim_boundary": secondary.CLAIM_BOUNDARY,
            "final_set_retention_or_free_rollout_claimed": False,
        },
        "output_file_digests": {
            MERGED_ROWS_NAME: {
                "path": MERGED_ROWS_NAME,
                "byte_size": len(rows_bytes),
                "row_count": len(rows),
                "sha256": sha256_bytes(rows_bytes),
            },
            MERGED_PARITY_NAME: {
                "path": MERGED_PARITY_NAME,
                "byte_size": len(parity_bytes),
                "row_count": len(parity_rows),
                "sha256": sha256_bytes(parity_bytes),
            },
        },
    }
    receipt["receipt_content_sha256"] = sha256_json(receipt)

    return {
        "receipt": receipt,
        "files": {
            MERGED_ROWS_NAME: rows_bytes,
            MERGED_PARITY_NAME: parity_bytes,
            MERGE_RECEIPT_NAME: canonical_json_bytes(receipt) + b"\n",
        },
        "rows": rows,
        "parity_rows": parity_rows,
        "shards": validated,
    }


# ---------------------------------------------------------------------------
# 9. CLI
# ---------------------------------------------------------------------------


def discover_shard_dirs(shard_root: Path) -> list[Path]:
    """Every immediate subdirectory of ``shard_root`` that carries a receipt."""

    shard_root = Path(shard_root)
    if not shard_root.is_dir():
        _fail(f"shard root {shard_root} does not exist")
    found = sorted(
        entry
        for entry in shard_root.iterdir()
        if entry.is_dir() and (entry / SHARD_RECEIPT_NAME).is_file()
    )
    if not found:
        _fail(f"shard root {shard_root} contains no secondary capture shard directory")
    return found


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--shard-root",
        type=Path,
        help="Directory whose immediate subdirectories are the per-image capture shards",
    )
    source.add_argument(
        "--shard",
        type=Path,
        action="append",
        dest="shards",
        help="One explicit capture shard directory; repeat once per sealed plan image",
    )
    parser.add_argument(
        "--plan-dir", type=Path, required=True, help="Frozen CPU plan directory"
    )
    parser.add_argument(
        "--primary-analysis-dir",
        type=Path,
        required=True,
        help="Sealed primary analysis directory whose branch gate every shard bound",
    )
    parser.add_argument(
        "--admission",
        type=Path,
        required=True,
        help="Sealed secondary smoke admission receipt every shard inherited",
    )
    parser.add_argument(
        "--expect-scorer-source-sha256",
        default=None,
        help=(
            "Optional pin: refuse any shard produced by another secondary scorer revision. "
            "Without it the twelve shards must still agree on one revision."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        shard_dirs = (
            discover_shard_dirs(args.shard_root)
            if args.shard_root is not None
            else [Path(value) for value in args.shards]
        )
        result = merge_shards(
            shard_dirs=shard_dirs,
            plan_dir=args.plan_dir,
            primary_analysis_dir=args.primary_analysis_dir,
            admission_path=args.admission,
            expect_scorer_source_sha256=args.expect_scorer_source_sha256,
        )
        published = primary_merge.publish_merge(Path(args.output_dir), result["files"])
    except primary_merge.MergeContractError as exc:
        raise SystemExit(f"secondary merge contract violated: {exc}") from exc
    print(
        json.dumps(
            {
                "merged": published,
                "receipt_content_sha256": result["receipt"]["receipt_content_sha256"],
                "row_count": len(result["rows"]),
                "request_ids_sha256": result["receipt"]["secondary_requests"][
                    "request_ids_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
