#!/usr/bin/env python3
"""Immutable merge for the sorted crossing matched-length neutral-row insertion
control (``2026-08-04-sorted-crossing-matched-length-neutral-row-insertion-control``).

Frozen unit::

    research/investigations/qwen3-vl-dense-enumeration/experiments/
    2026-08-04-sorted-crossing-matched-length-neutral-row-insertion-control/unit.md

What this module is
-------------------
The capture publishes one shard per image.  This pass re-proves every shard
against the sealed CPU plan and folds them into one immutable evidence set:

* **completeness** -- exactly the frozen 54 sealed requests, 21 ``P`` versus
  ``P+N``, 21 ``P`` versus ``P+C`` and 12 benign pairs, each exactly once, one
  shard per covered image, no cross-image or cross-owner leakage;
* **token identity** (``unit.md`` gate 1) -- every row's scored tokens
  reconstruct their sealed digest, both roots forced identical tokens, and the
  scored ``E`` token ids are identical between the ``N`` and ``C`` arms of every
  executed owner;
* **runtime replay** (``unit.md`` gate 2) -- every new-run ``P -> E`` coordinate
  baseline agrees with the sealed selected-logit sum within ``1e-3`` and
  preserves the compared argmax; a mismatch quarantines that owner and more than
  two quarantined owners stops the unit for runtime repair;
* **parity** (``unit.md`` gate 5) -- every shard inherited an admission whose
  cached-versus-uncached smoke passed the frozen ``1e-3`` tolerance, or ran on
  the uncached fallback; that smoke proved all four frozen strata; and every
  shard inherited the *same* smoke image and admission digest, so the whole
  capture stands behind one representative smoke; and
* **quarantine** -- the merge-owned ledger travels with the evidence, and a
  quarantined owner is neither material nor nonmaterial downstream.

What this module deliberately does **not** do
---------------------------------------------
It never computes a relative delta, a materiality, a specificity count or a
route.  Gates 3, 4 and 6 need the same-run benign reference and the frozen
``-1.0`` nat cutoff, and they belong to the analysis pass, which walks all six
gates in ``unit.md``'s fixed order using this receipt's sealed results for gates
1, 2 and 5.

Outputs (one immutable merged directory)::

    neutral-row-control-rows.jsonl
    neutral-row-control-parity.jsonl
    neutral-row-control-merge-receipt.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import (  # noqa: E402
    merge_sorted_crossing_boundary_owner_release_secondary as paired_merge,
)
from scripts.research import prepare_sorted_crossing_neutral_row_control as plan_builder  # noqa: E402
from scripts.research import (  # noqa: E402
    score_sorted_crossing_boundary_owner_release as crossing_scorer,
)
from scripts.research import (  # noqa: E402
    score_sorted_crossing_boundary_owner_release_secondary as paired,
)
from scripts.research import score_sorted_crossing_neutral_row_control as scorer  # noqa: E402

UNIT_ID = plan_builder.UNIT_ID
MERGE_SCHEMA_VERSION = "sorted_crossing_neutral_row_control_merge.v1"
MERGED_PARITY_SCHEMA_VERSION = "sorted_crossing_neutral_row_control_merged_parity.v1"

#: Shard input file names, taken from the producer so the two cannot drift.
SHARD_ROWS_NAME = scorer.ROWS_NAME
SHARD_PARITY_NAME = scorer.PARITY_NAME
SHARD_RECEIPT_NAME = scorer.RECEIPT_NAME
SHARD_QUARANTINE_NAME = scorer.QUARANTINE_NAME
SHARD_ADMISSION_NAME = scorer.ADMISSION_NAME
REQUIRED_SHARD_FILES: tuple[str, ...] = (
    SHARD_ROWS_NAME,
    SHARD_PARITY_NAME,
    SHARD_RECEIPT_NAME,
)

MERGED_ROWS_NAME = SHARD_ROWS_NAME
MERGED_PARITY_NAME = "neutral-row-control-parity.jsonl"
MERGE_RECEIPT_NAME = "neutral-row-control-merge-receipt.json"
MERGED_OUTPUT_NAMES: tuple[str, ...] = (
    MERGED_ROWS_NAME,
    MERGED_PARITY_NAME,
    MERGE_RECEIPT_NAME,
)

ARMS = scorer.ARMS
ARM_NEUTRAL = scorer.ARM_NEUTRAL
ARM_CLEAN_REPLAY = scorer.ARM_CLEAN_REPLAY
ARM_BENIGN = scorer.ARM_BENIGN
PAIRED_CROSSING_ARMS = scorer.PAIRED_CROSSING_ARMS
SEGMENTS = scorer.SEGMENTS
SEGMENT_COORDINATES = scorer.SEGMENT_COORDINATES
PAIRED_ROOTS = scorer.PAIRED_ROOTS
ROOT_BASELINE = scorer.ROOT_BASELINE
ROOT_MODIFIED = scorer.ROOT_MODIFIED

#: unit.md gate 2 tolerances and stop rule.
REPLAY_MAX_SELECTED_LOGIT_ABS_DIFF = plan_builder.REPLAY_MAX_SELECTED_LOGIT_ABS_DIFF
MAX_QUARANTINED_OWNERS = plan_builder.MAX_QUARANTINED_OWNERS

#: unit.md gate 5: the four strata the one representative smoke must have
#: exercised, in the frozen order the producer seals them.  Imported from the
#: scorer so the enforced and the produced vocabulary cannot drift.
REQUIRED_SMOKE_STRATA: tuple[str, ...] = scorer.REQUIRED_SMOKE_STRATA

#: Every module whose digest a shard must have sealed for its semantics to be
#: reconstructable.  Taken from the producer's own list so the two cannot drift.
REQUIRED_SOURCE_IDENTITY_KEYS: tuple[str, ...] = tuple(sorted(scorer.SOURCE_IDENTITY_MODULES))

#: The capture policy every merged shard must have executed under.
REQUIRED_POLICY: Mapping[str, Any] = {
    "uses_model_generate": False,
    "free_decode": False,
    "greedy_coordinate_decode": False,
    "retokenizes": False,
    "sampling": "not_implemented_deterministic_teacher_forcing_only",
    "likelihood_channel": scorer.LIKELIHOOD_CHANNEL,
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
    "replay_admission_enforced": True,
    "claim_boundary": scorer.CLAIM_BOUNDARY,
}

#: Row-level policy sentinels: what every published row must still declare.
REQUIRED_ROW_POLICY: Mapping[str, Any] = {
    "row_kind": "neutral_row_control_row",
    "likelihood_channel": scorer.LIKELIHOOD_CHANNEL,
    "uses_model_generate": False,
    "sampling": "disabled_deterministic_teacher_forcing_only",
    "retokenized": False,
    "delta_orientation": "modified_minus_baseline",
    "primary_segment": SEGMENT_COORDINATES,
    "claim_boundary": scorer.CLAIM_BOUNDARY,
    "replay_admission_enforced": True,
    "baseline_replay_admitted": True,
}

GATE_INPUT_AND_TOKEN_IDENTITY = "input_and_token_identity"
GATE_RUNTIME_REPLAY = "runtime_replay"
GATE_CACHE_PARITY = "cached_versus_uncached_parity"
MERGE_OWNED_GATES: tuple[str, ...] = (
    GATE_INPUT_AND_TOKEN_IDENTITY,
    GATE_RUNTIME_REPLAY,
    GATE_CACHE_PARITY,
)
#: The gates this pass deliberately leaves to the analysis, named so a reader
#: never mistakes a merge receipt for a decided unit.
ANALYSIS_OWNED_GATES: tuple[str, ...] = (
    "same_run_positive_controls",
    "benign_reference_replay",
    "neutral_row_specificity",
)


class NeutralRowMergeContractError(crossing_scorer.CrossingBoundaryContractError):
    """A precondition of this unit's immutable merge was not proven."""


def _fail(message: str) -> NoReturn:
    raise NeutralRowMergeContractError(message)


@contextmanager
def _predecessor_contract(label: str) -> Iterator[None]:
    """Re-raise a predecessor merge failure as this merge's own failure.

    The predecessor merge owns the per-root re-proof this pass reuses, and its
    error type is a different root class.  Surfacing it as
    :class:`NeutralRowMergeContractError` keeps one exception type at the merge
    boundary and keeps the CLI's fail-closed exit path intact.
    """

    try:
        yield
    except paired_merge.primary_merge.MergeContractError as exc:
        raise NeutralRowMergeContractError(f"{label}: {exc}") from exc


canonical_json_bytes = crossing_scorer.canonical_json_bytes
sha256_json = crossing_scorer.sha256_json
sha256_bytes = crossing_scorer.sha256_bytes
sha256_file = crossing_scorer.sha256_file


def _read_json(path: Path, label: str) -> dict[str, Any]:
    return crossing_scorer._read_json(Path(path), label)  # noqa: SLF001


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    return crossing_scorer._read_jsonl(Path(path), label)  # noqa: SLF001


def _token_ids(value: Any, *, label: str) -> list[int]:
    return crossing_scorer._token_ids(value, label=label)  # noqa: SLF001


def assert_self_sealed(payload: Mapping[str, Any], *, digest_key: str, label: str) -> str:
    declared = payload.get(digest_key)
    unsealed = {key: value for key, value in payload.items() if key != digest_key}
    reconstructed = sha256_json(unsealed)
    if reconstructed != declared:
        _fail(f"{label} does not self-seal; it has been edited after it was written")
    return str(reconstructed)


def _finite(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        _fail(f"{label} is not a number ({value!r})")
    if not math.isfinite(number):
        _fail(f"{label} is not finite ({value!r})")
    return number


# ---------------------------------------------------------------------------
# 1. The frozen plan, bound from its directory alone
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenPlan:
    """The sealed CPU plan, re-proven without loading any census surface."""

    plan_dir: Path
    manifest: dict[str, Any]
    manifest_content_sha256: str
    selection_by_owner: dict[str, dict[str, Any]]
    benign_by_owner: dict[str, dict[str, Any]]
    request_rows: list[dict[str, Any]]
    requests_by_id: dict[str, dict[str, Any]]
    plan_file_sha256: dict[str, str]

    @property
    def executed_owner_ids(self) -> list[str]:
        return sorted(
            owner_id
            for owner_id, row in self.selection_by_owner.items()
            if bool(row.get("executed"))
        )

    def image_of(self, gt_owner_id: str) -> str:
        row = self.selection_by_owner.get(gt_owner_id) or self.benign_by_owner.get(gt_owner_id)
        if row is None:
            _fail(f"owner {gt_owner_id!r} is in neither sealed registry")
        return str(row["image_id"])


def load_frozen_plan(plan_dir: Path) -> FrozenPlan:
    """Re-prove the plan manifest, its file digests and its frozen denominators."""

    plan_dir = Path(plan_dir)
    manifest = _read_json(plan_dir / plan_builder.MANIFEST_NAME, "neutral-row plan manifest")
    if str(manifest.get("schema_version")) != plan_builder.MANIFEST_SCHEMA_VERSION:
        _fail(
            f"plan manifest schema {manifest.get('schema_version')!r} is not "
            f"{plan_builder.MANIFEST_SCHEMA_VERSION!r}"
        )
    if str(manifest.get("unit_id")) != UNIT_ID:
        _fail("plan manifest belongs to another unit")
    manifest_seal = assert_self_sealed(
        manifest, digest_key="manifest_content_sha256", label="plan manifest"
    )
    digests = manifest.get("output_file_digests")
    if not isinstance(digests, Mapping):
        _fail("plan manifest carries no output_file_digests")

    plan_file_sha256: dict[str, str] = {}
    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for name, schema in (
        (plan_builder.SELECTION_REGISTRY_NAME, plan_builder.SELECTION_SCHEMA_VERSION),
        (plan_builder.BENIGN_REGISTRY_NAME, plan_builder.BENIGN_SCHEMA_VERSION),
        (plan_builder.REQUEST_PLAN_NAME, plan_builder.REQUEST_SCHEMA_VERSION),
    ):
        entry = digests.get(name)
        if not isinstance(entry, Mapping):
            _fail(f"plan manifest declares no digest for {name}")
        observed = sha256_file(plan_dir / name)
        if observed != str(entry.get("sha256")):
            _fail(f"sealed plan file {name} hashes to {observed}, not the manifest's digest")
        plan_file_sha256[name] = observed
        rows = _read_jsonl(plan_dir / name, name)
        if int(entry.get("row_count", -1)) != len(rows):
            _fail(f"sealed plan file {name} holds {len(rows)} rows, not the manifest's count")
        for row in rows:
            if str(row.get("schema_version")) != schema:
                _fail(f"{name} carries a row with schema {row.get('schema_version')!r}")
            if str(row.get("unit_id")) != UNIT_ID:
                _fail(f"{name} carries a row from another unit")
        rows_by_name[name] = rows

    selection_by_owner: dict[str, dict[str, Any]] = {}
    for row in rows_by_name[plan_builder.SELECTION_REGISTRY_NAME]:
        owner_id = str(row["gt_owner_id"])
        if owner_id in selection_by_owner:
            _fail(f"the sealed selection registry carries duplicate owner {owner_id!r}")
        selection_by_owner[owner_id] = row
    benign_by_owner: dict[str, dict[str, Any]] = {}
    for row in rows_by_name[plan_builder.BENIGN_REGISTRY_NAME]:
        owner_id = str(row["gt_owner_id"])
        if owner_id in benign_by_owner:
            _fail(f"the sealed benign registry carries duplicate owner {owner_id!r}")
        if owner_id in selection_by_owner:
            _fail(
                f"owner {owner_id!r} is both a crossing target and a benign control; the two "
                "registries must be disjoint"
            )
        benign_by_owner[owner_id] = row

    request_rows = rows_by_name[plan_builder.REQUEST_PLAN_NAME]
    requests_by_id: dict[str, dict[str, Any]] = {}
    for row in request_rows:
        request_id = str(row["request_id"])
        if request_id in requests_by_id:
            _fail(f"the sealed request plan carries duplicate request {request_id!r}")
        requests_by_id[request_id] = row

    plan = FrozenPlan(
        plan_dir=plan_dir,
        manifest=manifest,
        manifest_content_sha256=manifest_seal,
        selection_by_owner=selection_by_owner,
        benign_by_owner=benign_by_owner,
        request_rows=request_rows,
        requests_by_id=requests_by_id,
        plan_file_sha256=plan_file_sha256,
    )
    scorer.validate_plan_counts(plan)
    return plan


# ---------------------------------------------------------------------------
# 2. Shard reading and validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatedShard:
    """One published capture shard, proven against the sealed plan."""

    shard_dir: Path
    shard_id: str
    session_image_id: str
    receipt: dict[str, Any]
    receipt_content_sha256: str
    parity: dict[str, Any]
    rows: list[dict[str, Any]]
    file_sha256: dict[str, str]


def read_shard(shard_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Read one shard's three published files and hash them."""

    shard_dir = Path(shard_dir)
    if not shard_dir.is_dir():
        _fail(f"shard directory {shard_dir} does not exist")
    observed = sorted(child.name for child in shard_dir.iterdir())
    if SHARD_QUARANTINE_NAME in observed:
        _fail(
            f"shard {shard_dir} is quarantined and published no evidence; repair the runtime "
            "rather than merging a partial capture"
        )
    if SHARD_ADMISSION_NAME in observed:
        _fail(
            f"shard {shard_dir} is a smoke shard; only capture shards carry mergeable evidence"
        )
    missing = sorted(set(REQUIRED_SHARD_FILES) - set(observed))
    unknown = sorted(set(observed) - set(REQUIRED_SHARD_FILES))
    if missing:
        _fail(f"shard {shard_dir} is incomplete (missing {missing!r})")
    if unknown:
        _fail(f"shard {shard_dir} carries unknown artifact(s) {unknown!r}")
    file_sha256 = {name: sha256_file(shard_dir / name) for name in REQUIRED_SHARD_FILES}
    receipt = _read_json(shard_dir / SHARD_RECEIPT_NAME, f"{shard_dir.name} receipt")
    parity = _read_json(shard_dir / SHARD_PARITY_NAME, f"{shard_dir.name} parity")
    rows = _read_jsonl(shard_dir / SHARD_ROWS_NAME, f"{shard_dir.name} rows")
    return receipt, parity, rows, file_sha256


def _validate_shard_receipt(
    receipt: Mapping[str, Any],
    *,
    plan: FrozenPlan,
    shard_dir: Path,
    file_sha256: Mapping[str, str],
) -> tuple[str, str]:
    if str(receipt.get("schema_version")) != scorer.RECEIPT_SCHEMA_VERSION:
        _fail(f"{shard_dir} receipt schema {receipt.get('schema_version')!r} is not frozen")
    if str(receipt.get("unit_id")) != UNIT_ID:
        _fail(f"{shard_dir} receipt belongs to another unit")
    if str(receipt.get("mode")) != scorer.MODE_CAPTURE:
        _fail(f"{shard_dir} receipt declares mode {receipt.get('mode')!r}, not a capture")
    assert_self_sealed(receipt, digest_key="receipt_content_sha256", label=f"{shard_dir} receipt")

    declared_outputs = receipt.get("output_file_digests")
    if not isinstance(declared_outputs, Mapping):
        _fail(f"{shard_dir} receipt carries no output_file_digests")
    for name in (SHARD_ROWS_NAME, SHARD_PARITY_NAME):
        entry = declared_outputs.get(name)
        if not isinstance(entry, Mapping):
            _fail(f"{shard_dir} receipt declares no digest for {name}")
        if str(entry.get("sha256")) != file_sha256[name]:
            _fail(
                f"{shard_dir}/{name} hashes to {file_sha256[name]}, not the receipt's "
                f"{entry.get('sha256')}"
            )

    sealed_plan = receipt.get("plan")
    if not isinstance(sealed_plan, Mapping):
        _fail(f"{shard_dir} receipt seals no plan block")
    if str(sealed_plan.get("manifest_content_sha256")) != plan.manifest_content_sha256:
        _fail(
            f"{shard_dir} was captured against a different CPU plan manifest; refusing to merge "
            "rows produced under another cohort"
        )
    for name, digest in plan.plan_file_sha256.items():
        declared = (sealed_plan.get("plan_file_sha256") or {}).get(name)
        if str(declared) != digest:
            _fail(f"{shard_dir} sealed a different digest for plan file {name}")

    policy = receipt.get("policy")
    if not isinstance(policy, Mapping):
        _fail(f"{shard_dir} receipt seals no policy block")
    drifted = sorted(key for key, value in REQUIRED_POLICY.items() if policy.get(key) != value)
    if drifted:
        _fail(f"{shard_dir} was captured under a drifted policy: {drifted!r}")

    quarantine = receipt.get("quarantine") or {}
    if int(quarantine.get("count", -1)) != 0:
        _fail(
            f"{shard_dir} declares {quarantine.get('count')!r} quarantined request(s) but "
            "published evidence rows; a quarantined shard withholds its evidence"
        )

    runtime_identity = receipt.get("runtime_identity")
    if not isinstance(runtime_identity, Mapping):
        _fail(f"{shard_dir} receipt seals no runtime identity")
    source_identity = runtime_identity.get("source_identity")
    if not isinstance(source_identity, Mapping):
        _fail(f"{shard_dir} runtime identity seals no source_identity")
    missing_sources = sorted(set(REQUIRED_SOURCE_IDENTITY_KEYS) - set(source_identity))
    if missing_sources:
        _fail(f"{shard_dir} seals no source digest for module(s) {missing_sources!r}")

    executed = receipt.get("executed") or {}
    session_image_id = str(executed.get("session_image_id"))
    if not session_image_id or session_image_id == "None":
        _fail(f"{shard_dir} receipt names no session image")
    return str(receipt["shard_id"]), session_image_id


def _validate_shard_parity(
    parity: Mapping[str, Any], *, shard_dir: Path, shard_id: str, session_image_id: str
) -> None:
    if str(parity.get("schema_version")) != scorer.PARITY_SCHEMA_VERSION:
        _fail(f"{shard_dir} parity schema {parity.get('schema_version')!r} is not frozen")
    if str(parity.get("unit_id")) != UNIT_ID:
        _fail(f"{shard_dir} parity belongs to another unit")
    if str(parity.get("shard_id")) != shard_id or str(
        parity.get("session_image_id")
    ) != session_image_id:
        _fail(f"{shard_dir} parity does not describe its own shard")
    if (
        _finite(
            parity.get("max_selected_logit_abs_diff_threshold"),
            label=f"{shard_dir} parity threshold",
        )
        != REPLAY_MAX_SELECTED_LOGIT_ABS_DIFF
    ):
        _fail(f"{shard_dir} parity declares a threshold other than the frozen 1e-3")
    inherited = parity.get("inherited_admission")
    if not isinstance(inherited, Mapping):
        _fail(
            f"{shard_dir} parity inherits no admission; a capture never assumes cached execution"
        )
    proven = inherited.get("smoke_strata_proven")
    if not isinstance(proven, Sequence) or isinstance(proven, (str, bytes)):
        _fail(
            f"{shard_dir} parity inherits an admission that names no smoke_strata_proven; only a "
            "smoke that recorded every stratum it exercised may admit a capture"
        )
    if [str(value) for value in proven] != list(REQUIRED_SMOKE_STRATA):
        _fail(
            f"{shard_dir} inherited a smoke proving {[str(v) for v in proven]!r}, not the frozen "
            f"{list(REQUIRED_SMOKE_STRATA)!r} in order; unit.md gate 5 requires one real "
            "matched-E, unmatched-E, same-description and different-description smoke"
        )
    for field in ("smoke_image_id", "admission_content_sha256"):
        value = inherited.get(field)
        if not value or str(value) == "None":
            _fail(f"{shard_dir} parity inherits an admission that names no {field}")
    root_backend = parity.get("root_backend") or {}
    for root in PAIRED_ROOTS:
        backend = str(root_backend.get(root))
        if backend not in (crossing_scorer.KV_CACHE_BACKEND, crossing_scorer.UNCACHED_BACKEND):
            _fail(f"{shard_dir} parity declares unknown backend {backend!r} for root {root!r}")
    if not bool(inherited.get("cache_admitted")) and any(
        str(value) != crossing_scorer.UNCACHED_BACKEND for value in root_backend.values()
    ):
        _fail(
            f"{shard_dir} inherited a non-admitted cache but still executed a cached root; the "
            "uncached fallback was not applied"
        )


def validate_row(
    row: Mapping[str, Any],
    *,
    plan: FrozenPlan,
    shard_id: str,
    session_image_id: str,
) -> str:
    """Re-prove one published row against the sealed plan request, or fail."""

    request_id = str(row.get("request_id"))
    label = f"row {request_id!r}"
    if str(row.get("schema_version")) != scorer.SCHEMA_VERSION:
        _fail(f"{label} carries schema {row.get('schema_version')!r}, not the frozen one")
    if str(row.get("unit_id")) != UNIT_ID:
        _fail(f"{label} belongs to another unit")
    if str(row.get("shard_id")) != shard_id:
        _fail(f"{label} is filed under shard {row.get('shard_id')!r}, not {shard_id!r}")
    if str(row.get("session_image_id")) != session_image_id:
        _fail(f"{label} was captured in another image's session")
    drifted = sorted(key for key, value in REQUIRED_ROW_POLICY.items() if row.get(key) != value)
    if drifted:
        _fail(f"{label} declares a drifted row policy: {drifted!r}")

    request = plan.requests_by_id.get(request_id)
    if request is None:
        _fail(f"{label} is not a sealed request of this plan")
    arm = scorer.assert_request(request)
    for key in ("arm", "cohort_role", "gt_owner_id", "image_id", "request_key", "request_family"):
        if str(row.get(key)) != str(request.get(key)):
            _fail(f"{label} declares {key}={row.get(key)!r}, not the sealed {request.get(key)!r}")
    if str(row.get("plan_identity_digest")) != str(request["identity_digest"]):
        _fail(f"{label} does not carry its sealed plan identity digest")
    if str(row.get("plan_manifest_content_sha256")) != plan.manifest_content_sha256:
        _fail(f"{label} was produced against another plan manifest")
    if str(row.get("image_id")) != session_image_id:
        _fail(f"{label} belongs to image {row.get('image_id')!r} but is in another image's shard")

    target = request["scored_target"]
    scored_tokens = _token_ids(row.get("scored_token_ids"), label=f"{label} scored tokens")
    if scored_tokens != _token_ids(target["token_ids"], label=f"{label} sealed target"):
        _fail(f"{label} scored tokens are not the sealed target tokens")
    if sha256_json(scored_tokens) != str(target["token_ids_sha256"]):
        _fail(f"{label} scored tokens do not reconstruct the sealed target digest")
    if str(row.get("scored_token_ids_sha256")) != str(target["token_ids_sha256"]):
        _fail(f"{label} declares a scored-token digest other than the sealed one")
    if int(row.get("scored_token_count", -1)) != len(scored_tokens):
        _fail(f"{label} scored token count disagrees with its own token ids")
    if int(row.get("native_row_index", -1)) != int(target["native_row_index"]):
        _fail(f"{label} scores another native row than the sealed one")
    for key, sealed_key in (
        ("baseline_context_id", "baseline_context_id"),
        ("successor_context_id", "successor_context_id"),
    ):
        if str(row.get(key)) != str(target[sealed_key]):
            _fail(f"{label} declares {key}={row.get(key)!r}, not the sealed {target[sealed_key]!r}")
    if str(row.get("modified_context_id")) != str(request["context_id"]):
        _fail(f"{label} appends at a context the plan does not seal")

    prefix = request["prefix"]
    if str(row.get("appended_row_token_ids_sha256")) != str(prefix["appended_token_ids_sha256"]):
        _fail(f"{label} appended a row other than the sealed one")
    if int(row.get("appended_row_token_count", -1)) != int(prefix["appended_token_count"]):
        _fail(f"{label} appended token count disagrees with the sealed plan")
    if str(row.get("appended_role")) != str(prefix["appended_role"]):
        _fail(f"{label} declares an appended role other than the sealed one")
    if dict(row.get("sealed_reference") or {}) != dict(request["sealed_reference"]):
        _fail(f"{label} carries a sealed reference block the plan does not seal")

    segments = paired.row_segments(scored_tokens)
    declared_segments = row.get("segments")
    if not isinstance(declared_segments, Mapping):
        _fail(f"{label} carries no segments block")
    for segment in SEGMENTS:
        start, stop = segments.span(segment)
        block = declared_segments.get(segment)
        if not isinstance(block, Mapping):
            _fail(f"{label} carries no {segment!r} segment")
        if int(block.get("token_index_start", -1)) != start or int(
            block.get("token_index_stop", -1)
        ) != stop:
            _fail(f"{label} {segment!r} span does not reconstruct from the row grammar")
        if _token_ids(block.get("token_ids"), label=label) != scored_tokens[start:stop]:
            _fail(f"{label} {segment!r} tokens do not reconstruct from its own scored tokens")

    roots = row.get("roots")
    if not isinstance(roots, Mapping) or sorted(roots) != sorted(PAIRED_ROOTS):
        _fail(f"{label} does not carry exactly the two paired roots")
    root_context = {
        ROOT_BASELINE: str(target["baseline_context_id"]),
        ROOT_MODIFIED: str(request["context_id"]),
    }
    root_appended_digest = {
        ROOT_BASELINE: sha256_json([]),
        ROOT_MODIFIED: str(prefix["appended_token_ids_sha256"]),
    }
    root_appended_count = {
        ROOT_BASELINE: 0,
        ROOT_MODIFIED: int(prefix["appended_token_count"]),
    }
    sums: dict[str, dict[str, Any]] = {}
    for root in PAIRED_ROOTS:
        # The predecessor merge owns the per-root re-proof -- identity, per-token
        # streams, rank flags and per-segment sums.  Its ``variant`` is this
        # unit's ``arm``: both are the discriminator the logical context-group id
        # is derived from, so reusing it keeps one implementation of that seam.
        with _predecessor_contract(f"{label} root {root!r}"):
            sums[root] = paired_merge._assert_root_payload(  # noqa: SLF001
                roots[root],
                root=root,
                context_id=root_context[root],
                scored_token_ids=scored_tokens,
                segments=segments,
                appended_sha256=root_appended_digest[root],
                appended_count=root_appended_count[root],
                gt_owner_id=str(request["gt_owner_id"]),
                variant=arm,
                label=label,
            )
    expected_deltas = paired.paired_deltas(sums[ROOT_BASELINE], sums[ROOT_MODIFIED])
    declared_deltas = row.get("deltas")
    if not isinstance(declared_deltas, Mapping):
        _fail(f"{label} carries no deltas block")
    if {key: dict(value) for key, value in declared_deltas.items()} != expected_deltas:
        _fail(f"{label} deltas do not reconstruct from its own per-root segment sums")

    if bool(row.get("baseline_replay_admitted")) is not bool(
        roots[ROOT_BASELINE]["argmax_reproduces_description_path"]
    ):
        _fail(f"{label} baseline replay admission contradicts its own baseline root")
    return request_id


def validate_shard(shard_dir: Path, *, plan: FrozenPlan) -> ValidatedShard:
    """Read and completely re-prove one published shard."""

    receipt, parity, rows, file_sha256 = read_shard(shard_dir)
    shard_id, session_image_id = _validate_shard_receipt(
        receipt, plan=plan, shard_dir=Path(shard_dir), file_sha256=file_sha256
    )
    _validate_shard_parity(
        parity,
        shard_dir=Path(shard_dir),
        shard_id=shard_id,
        session_image_id=session_image_id,
    )
    seen: set[str] = set()
    for row in rows:
        request_id = validate_row(
            row, plan=plan, shard_id=shard_id, session_image_id=session_image_id
        )
        if request_id in seen:
            _fail(f"shard {shard_dir} publishes request {request_id!r} more than once")
        seen.add(request_id)

    expected = sorted(
        str(request["request_id"])
        for request in plan.request_rows
        if str(request["image_id"]) == session_image_id
    )
    if sorted(seen) != expected:
        _fail(
            f"shard {shard_dir} published {len(seen)} of image {session_image_id!r}'s "
            f"{len(expected)} sealed requests; a partial image is never merged"
        )
    executed = receipt.get("executed") or {}
    if int(executed.get("row_count", -1)) != len(rows):
        _fail(f"shard {shard_dir} receipt row_count disagrees with its published rows")
    return ValidatedShard(
        shard_dir=Path(shard_dir),
        shard_id=shard_id,
        session_image_id=session_image_id,
        receipt=receipt,
        receipt_content_sha256=str(receipt["receipt_content_sha256"]),
        parity=parity,
        rows=rows,
        file_sha256=dict(file_sha256),
    )


# ---------------------------------------------------------------------------
# 3. Closure and the merge-owned gates
# ---------------------------------------------------------------------------


def assert_shard_closure(plan: FrozenPlan, shards: Sequence[ValidatedShard]) -> dict[str, Any]:
    """One shard per covered image, one identity, and exactly the frozen 54 rows."""

    covered_images = sorted({str(row["image_id"]) for row in plan.request_rows})
    images = [shard.session_image_id for shard in shards]
    duplicated = sorted({image for image in images if images.count(image) > 1})
    if duplicated:
        _fail(f"image(s) {duplicated!r} are covered by more than one shard")
    if sorted(images) != covered_images:
        _fail(
            f"the shard set covers image(s) {sorted(images)!r}, not the sealed "
            f"{covered_images!r}"
        )
    identities = {
        shard.session_image_id: str(
            (shard.receipt.get("runtime_identity") or {}).get("runtime_identity_sha256")
            or shard.receipt.get("runtime_identity_sha256")
        )
        for shard in shards
    }
    distinct = sorted(set(identities.values()))
    if len(distinct) != 1:
        _fail(
            "the shards were captured under more than one runtime identity "
            f"({identities!r}); a same-run contrast requires one runtime"
        )

    rows = [row for shard in shards for row in shard.rows]
    request_ids = [str(row["request_id"]) for row in rows]
    if len(set(request_ids)) != len(request_ids):
        _fail("the merged evidence carries a duplicated request")
    if sorted(request_ids) != sorted(plan.requests_by_id):
        missing = sorted(set(plan.requests_by_id) - set(request_ids))
        extra = sorted(set(request_ids) - set(plan.requests_by_id))
        _fail(
            f"the merged evidence is not the sealed request census (missing={missing!r}, "
            f"unsealed={extra!r})"
        )
    by_arm = scorer.counts_by_arm(rows)
    if by_arm != dict(scorer.EXPECTED_REQUEST_COUNT_BY_ARM):
        _fail(f"the merged evidence holds {by_arm!r}, not the frozen arm census")

    for row in rows:
        owner_image = plan.image_of(str(row["gt_owner_id"]))
        if owner_image != str(row["image_id"]):
            _fail(
                f"row {row['request_id']!r} attributes owner {row['gt_owner_id']!r} to image "
                f"{row['image_id']!r}, but the sealed registry places it on {owner_image!r}"
            )
    return {
        "shard_count": len(shards),
        "image_ids": covered_images,
        "runtime_identity_sha256": distinct[0],
        "row_count": len(rows),
        "row_count_by_arm": dict(sorted(by_arm.items())),
        "per_image_row_counts": dict(sorted(Counter(images).items())),
    }


def assert_one_smoke_admission(shards: Sequence[ValidatedShard]) -> dict[str, Any]:
    """``unit.md`` gate 5: every shard ran behind one four-strata smoke session.

    Each shard's parity has already proven, per shard, that it inherited the
    exact frozen strata.  What is left is the one-smoke contract itself: a
    second admission -- another image, another session, or a re-run smoke --
    would mean the merged evidence was never admitted by a single proven
    cached-versus-uncached seam, so a divergence stops the merge.
    """

    identities: dict[str, dict[str, list[str]]] = {"smoke image": {}, "admission receipt": {}}
    proven: set[tuple[str, ...]] = set()
    for shard in shards:
        inherited = shard.parity.get("inherited_admission") or {}
        proven.add(tuple(str(value) for value in inherited.get("smoke_strata_proven") or ()))
        for label, field in (
            ("smoke image", "smoke_image_id"),
            ("admission receipt", "admission_content_sha256"),
        ):
            identities[label].setdefault(str(inherited.get(field)), []).append(
                shard.session_image_id
            )
    for label, witnesses in identities.items():
        if len(witnesses) != 1:
            grouped = {key: sorted(images) for key, images in sorted(witnesses.items())}
            _fail(
                f"the shards inherited {len(witnesses)} different {label}s ({grouped!r}); "
                "unit.md admits the whole capture from exactly one representative smoke"
            )
    # Already proven per shard; re-proven here so this receipt fact stands on
    # its own rather than on the caller having validated every shard first.
    if proven != {tuple(REQUIRED_SMOKE_STRATA)}:
        _fail(
            f"the shards inherited smoke(s) proving {sorted(proven)!r}, not one smoke proving "
            f"the frozen {list(REQUIRED_SMOKE_STRATA)!r}"
        )
    return {
        "required_strata": list(REQUIRED_SMOKE_STRATA),
        "strata_proven_by_every_shard": list(next(iter(proven))),
        "smoke_image_id": next(iter(identities["smoke image"])),
        "admission_content_sha256": next(iter(identities["admission receipt"])),
        "admitted_shard_count": len(shards),
        "semantics": (
            "one representative smoke exercised matched-E, unmatched-E, same-description and "
            "different-description inside one image session, and every shard inherited it"
        ),
    }


def evaluate_merge_gates(
    plan: FrozenPlan, rows: Sequence[Mapping[str, Any]], shards: Sequence[ValidatedShard]
) -> dict[str, Any]:
    """``unit.md`` gates 1, 2 and 5, in their fixed order.

    Gate 2 is owner-level: the baseline root of a crossing arm is the unmodified
    native ``P``, so the ``N`` and ``C`` arms of one owner replay the same
    baseline and are checked once, against the sealed predecessor capture.
    """

    token_identity = scorer.assert_scored_token_identity(rows)

    baseline_by_owner: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row["arm"]) not in PAIRED_CROSSING_ARMS:
            continue
        baseline_by_owner.setdefault(str(row["gt_owner_id"]), []).append(row)

    replay: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for gt_owner_id in sorted(baseline_by_owner):
        owner_rows = baseline_by_owner[gt_owner_id]
        reference = dict(owner_rows[0]["sealed_reference"])
        sealed_sum = _finite(
            reference["baseline_coordinate_sum"], label=f"{gt_owner_id} sealed baseline sum"
        )
        observed_sums = {
            _finite(
                row["roots"][ROOT_BASELINE]["segment_sums"][SEGMENT_COORDINATES]["sum"],
                label=f"{gt_owner_id} observed baseline sum",
            )
            for row in owner_rows
        }
        if len(observed_sums) != 1:
            _fail(
                f"owner {gt_owner_id!r} produced different baseline coordinate sums across its "
                "arms; the two arms did not replay one unmodified native root"
            )
        observed_sum = observed_sums.pop()
        abs_diff = abs(observed_sum - sealed_sum)
        argmax_digests = {
            sha256_json(
                _token_ids(
                    row["roots"][ROOT_BASELINE]["argmax_token_ids"],
                    label=f"{gt_owner_id} baseline argmax",
                )
            )
            for row in owner_rows
        }
        argmax_preserved = argmax_digests == {
            str(reference["baseline_argmax_token_ids_sha256"])
        }
        admitted = abs_diff <= REPLAY_MAX_SELECTED_LOGIT_ABS_DIFF and argmax_preserved
        entry = {
            "gt_owner_id": gt_owner_id,
            "image_id": str(owner_rows[0]["image_id"]),
            "cohort_role": str(owner_rows[0]["cohort_role"]),
            "sealed_baseline_coordinate_sum": sealed_sum,
            "observed_baseline_coordinate_sum": observed_sum,
            "abs_diff": abs_diff,
            "tolerance": REPLAY_MAX_SELECTED_LOGIT_ABS_DIFF,
            "compared_argmax_preserved": argmax_preserved,
            "admitted": admitted,
        }
        replay.append(entry)
        if not admitted:
            quarantined.append(
                {
                    **entry,
                    "reason": "baseline_coordinate_replay_disagrees_with_the_sealed_capture",
                }
            )
    if len(quarantined) > MAX_QUARANTINED_OWNERS:
        _fail(
            f"{len(quarantined)} owners failed the runtime replay gate, more than the frozen "
            f"{MAX_QUARANTINED_OWNERS}; unit.md stops the unit for runtime repair"
        )

    parity_statuses = sorted(
        {
            str(
                (shard.parity.get("inherited_admission") or {}).get("cache_admitted")
            )
            for shard in shards
        }
    )
    return {
        "order": list(MERGE_OWNED_GATES),
        "deferred_to_analysis": list(ANALYSIS_OWNED_GATES),
        GATE_INPUT_AND_TOKEN_IDENTITY: {
            "passed": True,
            **token_identity,
            "every_row_reconstructs_its_sealed_target_digest": True,
            "paired_roots_forced_identical_tokens": True,
        },
        GATE_RUNTIME_REPLAY: {
            "passed": True,
            "tolerance": REPLAY_MAX_SELECTED_LOGIT_ABS_DIFF,
            "checked_owner_count": len(replay),
            "max_abs_diff": max((entry["abs_diff"] for entry in replay), default=0.0),
            "quarantined_owner_count": len(quarantined),
            "quarantined_owner_ids": sorted(
                str(entry["gt_owner_id"]) for entry in quarantined
            ),
            "max_quarantined_owners": MAX_QUARANTINED_OWNERS,
            "owner_rows": replay,
            "semantics": (
                "a quarantined owner is neither material nor nonmaterial and therefore "
                "conservatively satisfies no route"
            ),
        },
        GATE_CACHE_PARITY: {
            "passed": True,
            "tolerance": REPLAY_MAX_SELECTED_LOGIT_ABS_DIFF,
            "inherited_cache_admitted": parity_statuses,
            "inherited_smoke": assert_one_smoke_admission(shards),
            "shard_root_backends": {
                shard.session_image_id: dict(shard.parity.get("root_backend") or {})
                for shard in shards
            },
        },
    }


# ---------------------------------------------------------------------------
# 4. Merge
# ---------------------------------------------------------------------------


def _row_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["image_id"]),
        str(row["gt_owner_id"]),
        str(row["arm"]),
        str(row["request_id"]),
    )


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def discover_shard_dirs(shard_root: Path) -> list[Path]:
    shard_root = Path(shard_root)
    if not shard_root.is_dir():
        _fail(f"shard root {shard_root} does not exist")
    dirs = sorted(child for child in shard_root.iterdir() if child.is_dir())
    if not dirs:
        _fail(f"shard root {shard_root} holds no shard directory")
    return dirs


def merge_shards(
    *, plan_dir: Path, shard_dirs: Sequence[Path], output_dir: Path
) -> dict[str, Any]:
    """Validate every shard, fold them into one immutable evidence set."""

    plan = load_frozen_plan(plan_dir)
    shards = [validate_shard(Path(shard_dir), plan=plan) for shard_dir in shard_dirs]
    closure = assert_shard_closure(plan, shards)
    rows = sorted(
        (row for shard in shards for row in shard.rows), key=_row_sort_key
    )
    gates = evaluate_merge_gates(plan, rows, shards)

    parity_rows = sorted(
        (
            {
                "unit_id": UNIT_ID,
                "schema_version": MERGED_PARITY_SCHEMA_VERSION,
                "row_kind": "neutral_row_control_merged_parity_row",
                "shard_dir": str(shard.shard_dir),
                "shard_id": shard.shard_id,
                "session_image_id": shard.session_image_id,
                "shard_receipt_content_sha256": shard.receipt_content_sha256,
                "shard_file_sha256": dict(sorted(shard.file_sha256.items())),
                "parity": shard.parity,
            }
            for shard in shards
        ),
        key=lambda row: str(row["session_image_id"]),
    )

    files = {
        MERGED_ROWS_NAME: _jsonl_bytes(rows),
        MERGED_PARITY_NAME: _jsonl_bytes(parity_rows),
    }
    receipt = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "merger_source_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "scorer_source_sha256": str(
            (shards[0].receipt.get("scorer_source_sha256") or "")
        ),
        "plan": {
            "plan_dir": str(plan.plan_dir),
            "manifest_content_sha256": plan.manifest_content_sha256,
            "plan_file_sha256": dict(sorted(plan.plan_file_sha256.items())),
            "cohort": plan.manifest.get("cohort"),
            "gates": plan.manifest.get("gates"),
            "routes": plan.manifest.get("routes"),
            "materiality": plan.manifest.get("materiality"),
            "non_voting_sensitivities": plan.manifest.get("non_voting_sensitivities"),
            "lineage": plan.manifest.get("lineage"),
        },
        "runtime_identity_sha256": closure["runtime_identity_sha256"],
        "shards": [
            {
                "shard_dir": str(shard.shard_dir),
                "shard_id": shard.shard_id,
                "session_image_id": shard.session_image_id,
                "receipt_content_sha256": shard.receipt_content_sha256,
                "file_sha256": dict(sorted(shard.file_sha256.items())),
                "row_count": len(shard.rows),
            }
            for shard in shards
        ],
        "closure": closure,
        "gates": gates,
        "policy": {
            "merges_only_complete_unquarantined_shards": True,
            "relative_estimand_computed_here": False,
            "materiality_decided_here": False,
            "specificity_gate_decided_here": False,
            "route_decided_here": False,
            "sealed_reference_role": "gate_reference_only_never_the_estimand",
            "same_run_benign_replay_owns_both_relative_estimands": True,
            "claim_boundary": scorer.CLAIM_BOUNDARY,
        },
        "artifact_determinism": "no_wall_clock_or_host_specific_field_is_sealed",
        "output_file_digests": {
            name: {"path": name, "byte_size": len(payload), "sha256": sha256_bytes(payload)}
            for name, payload in sorted(files.items())
        },
    }
    receipt["receipt_content_sha256"] = sha256_json(receipt)
    files[MERGE_RECEIPT_NAME] = canonical_json_bytes(receipt) + b"\n"
    crossing_scorer._publish(Path(output_dir), files)  # noqa: SLF001
    return receipt


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True, type=Path, help="sealed CPU plan directory")
    parser.add_argument(
        "--shard-root",
        type=Path,
        default=None,
        help="directory whose immediate subdirectories are capture shards",
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        action="append",
        default=None,
        help="explicit capture shard directory; repeatable",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="merged output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.shard_dir:
            shard_dirs = [Path(value) for value in args.shard_dir]
        elif args.shard_root is not None:
            shard_dirs = discover_shard_dirs(Path(args.shard_root))
        else:
            _fail("pass --shard-root or at least one --shard-dir")
        receipt = merge_shards(
            plan_dir=Path(args.plan_dir),
            shard_dirs=shard_dirs,
            output_dir=Path(args.output_dir),
        )
    except crossing_scorer.CrossingBoundaryContractError as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        return 1
    closure = receipt["closure"]
    replay = receipt["gates"][GATE_RUNTIME_REPLAY]
    print(
        "merged neutral-row evidence: "
        f"shards={closure['shard_count']} rows={closure['row_count']} "
        f"{closure['row_count_by_arm']} "
        f"quarantined={replay['quarantined_owner_count']} "
        f"receipt={receipt['receipt_content_sha256'][:12]}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
