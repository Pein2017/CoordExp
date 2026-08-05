#!/usr/bin/env python3
"""Immutable primary-only merge for the sorted crossing-boundary owner
release/realization unit
(``2026-08-03-sorted-crossing-boundary-owner-release-realization``).

Frozen unit::

    research/investigations/qwen3-vl-dense-enumeration/experiments/
    2026-08-03-sorted-crossing-boundary-owner-release-realization/{unit.md,tasks.md}

What this module is
-------------------
It consumes **exactly one admitted capture shard per frozen image** -- all
twelve, no more and no fewer -- re-proves every identity the shards declare, and
republishes their *raw* rows and records under one deterministic, self-sealed
merged artifact family.

Everything it verifies before a single row is copied:

* **file identity** -- every shard directory carries exactly the four primary
  files, no quarantine file, and no unknown artifact;
* **self identity** -- every receipt reconstructs its own
  ``receipt_content_sha256`` from its own declared content;
* **source identity** -- every shard declares the frozen scorer digest, both as
  ``scorer_source_sha256`` and inside ``runtime_identity.source_identity``;
* **plan identity** -- every shard was captured against the frozen v3 plan whose
  manifest self-seals and whose sealed file digests match the bytes on disk;
* **runtime identity** -- every shard reconstructs its own
  ``runtime_identity_sha256`` from its own declared identity fields, and all
  twelve shards share one runtime; a mixed runtime is refused, never averaged;
* **admission identity** -- every shard inherited the same self-sealed smoke
  admission, and its parity file agrees with its receipt about which one; and
* **content identity** -- owner records and score rows reproduce the counters
  and request-id digest their own receipt sealed.

What this module deliberately is **not**
----------------------------------------
* It assigns **no branch**.  ``unit.md`` gives branch assignment to the analysis
  pass over merged primary owner records; this merge only proves that every
  record it republishes still carries the capture's
  ``branch_assignment``/``secondary_compatibility`` sentinels.
* It reads and republishes only ``readout_tier == "primary"`` rows.  A shard
  that executed a deferred secondary (``P+C`` / ``P+E+C``) request is refused
  rather than partially merged.
* It repairs nothing.  A withheld, incomplete, duplicated, quarantine-stopped or
  unknown input fails closed; the shards on disk are never modified.

Outputs (one explicit merged directory, published create-or-identical)::

    crossing-boundary-scores.jsonl        raw per-request rows, all shards
    crossing-boundary-owner-records.jsonl raw per-owner records, all shards
    crossing-boundary-parity.jsonl        one raw parity object per shard
    merge-receipt.json                    identities, digests, counters
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import score_sorted_crossing_boundary_owner_release as scorer  # noqa: E402

# ---------------------------------------------------------------------------
# 0. Frozen identities
# ---------------------------------------------------------------------------

UNIT_ID = scorer.UNIT_ID
MERGE_SCHEMA_VERSION = "sorted_crossing_boundary_owner_release_merge.v1"

#: The scorer revision this unit's evidence is frozen against.  A capture from
#: any other revision is a different measurement and is never merged into this
#: family.
FROZEN_SCORER_SOURCE_SHA256 = (
    "51df797e75023053e098c24a581d9f6387bac3b18074aa3cc3a78d0814d82d26"
)
#: The scorer module key inside ``runtime_identity.source_identity``.
SCORER_SOURCE_IDENTITY_KEY = "scripts.research.score_sorted_crossing_boundary_owner_release"
#: The frozen CPU plan revision.
FROZEN_PLAN_MANIFEST_SCHEMA_VERSION = (
    "sorted-crossing-boundary-owner-release-realization-plan.v3"
)

#: unit.md "Exact state pair": the frozen twelve images, and nothing else.
FROZEN_IMAGE_IDS: tuple[str, ...] = (
    "10707",
    "13348",
    "13923",
    "14038",
    "14439",
    "1584",
    "16228",
    "2685",
    "4134",
    "5001",
    "6040",
    "7511",
)
EXPECTED_SHARD_COUNT = len(FROZEN_IMAGE_IDS)

#: Cohort labels the sealed plan emits.  ``u_bound_crossing_primary`` is the
#: only primary denominator; the two control cohorts are carried through the
#: merge but never counted into it.
PRIMARY_COHORT = "u_bound_crossing_primary"
TIMING_CONTROL_COHORT = "disjoint_timing_control"
TP_REPLAY_CONTROL_COHORT = "native_tp_replay_control"
COHORTS: tuple[str, ...] = (PRIMARY_COHORT, TIMING_CONTROL_COHORT, TP_REPLAY_CONTROL_COHORT)

MATCHED_E_STRATUM = "matched_e"
UNMATCHED_E_STRATUM = "unmatched_e"
STRATA: tuple[str, ...] = (MATCHED_E_STRATUM, UNMATCHED_E_STRATUM)

#: unit.md "Exact state pair": the two decision-bearing primary ladders.
PRIMARY_LADDER_VARIANTS: frozenset[str] = frozenset({"at_p", "at_p_plus_e"})

#: The sentinels the capture writes and the merge must still find intact.
BRANCH_ASSIGNMENT_SENTINEL = (
    "not_performed_in_capture_pure_helpers_own_it_in_the_later_analysis"
)
SECONDARY_COMPATIBILITY_SENTINEL = "deferred_until_primary_branches_are_sealed"

#: Shard input file names, taken from the scorer so the two cannot drift.
SHARD_SCORES_NAME = scorer.SCORES_NAME
SHARD_OWNER_RECORDS_NAME = scorer.OWNER_RECORDS_NAME
SHARD_PARITY_NAME = scorer.PARITY_NAME
SHARD_RECEIPT_NAME = scorer.RECEIPT_NAME
SHARD_QUARANTINE_NAME = scorer.QUARANTINE_NAME
SHARD_ADMISSION_NAME = scorer.ADMISSION_NAME
REQUIRED_SHARD_FILES: tuple[str, ...] = (
    SHARD_SCORES_NAME,
    SHARD_OWNER_RECORDS_NAME,
    SHARD_PARITY_NAME,
    SHARD_RECEIPT_NAME,
)

#: Merged output file names.
MERGED_SCORES_NAME = SHARD_SCORES_NAME
MERGED_OWNER_RECORDS_NAME = SHARD_OWNER_RECORDS_NAME
MERGED_PARITY_NAME = "crossing-boundary-parity.jsonl"
MERGE_RECEIPT_NAME = "merge-receipt.json"
MERGED_OUTPUT_NAMES: tuple[str, ...] = (
    MERGED_SCORES_NAME,
    MERGED_OWNER_RECORDS_NAME,
    MERGED_PARITY_NAME,
    MERGE_RECEIPT_NAME,
)

#: Plan files whose bytes this merge re-proves against the sealed manifest.
PLAN_FILE_NAMES: tuple[str, ...] = (
    "cohort-registry.jsonl",
    "control-registry.jsonl",
    "request-plan.jsonl",
)

#: unit.md "Native replay alignment": more than two quarantined primary owners
#: stops the unit before interpretation.
MAX_PRIMARY_QUARANTINES = scorer.MAX_PRIMARY_QUARANTINES

#: The capture policy fields a merged shard must have executed under.
REQUIRED_POLICY: Mapping[str, Any] = {
    "branch_assignment_performed": False,
    "secondary_compatibility_executed": False,
    "uses_model_generate": False,
    "retokenizes": False,
    "sampling": "not_implemented_primary_deterministic_pass_only",
    "likelihood_channel": scorer.LIKELIHOOD_CHANNEL,
    "fresh_cache_per_logical_context_group": True,
    "one_image_session_per_shard": True,
}


class MergeContractError(RuntimeError):
    """A precondition of this unit's immutable primary merge was not proven."""


def _fail(message: str) -> NoReturn:
    raise MergeContractError(message)


# ---------------------------------------------------------------------------
# 1. IO / digest helpers (reuse the scorer's canonicalization so digests agree)
# ---------------------------------------------------------------------------

canonical_json_bytes = scorer.canonical_json_bytes
sha256_bytes = scorer.sha256_bytes
sha256_json = scorer.sha256_json
sha256_file = scorer.sha256_file


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


def _file_entry(path: Path) -> dict[str, Any]:
    payload = Path(path).read_bytes()
    return {
        "path": str(path),
        "byte_size": len(payload),
        "sha256": sha256_bytes(payload),
    }


# ---------------------------------------------------------------------------
# 2. The frozen plan and the inherited smoke admission
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenPlan:
    """The sealed v3 CPU plan, proven from its own manifest and its own bytes."""

    plan_dir: Path
    manifest: dict[str, Any]
    manifest_content_sha256: str
    plan_file_sha256: dict[str, str]

    @property
    def cohort_counts(self) -> Mapping[str, Any]:
        return self.manifest["cohort_counts"]

    @property
    def control_counts(self) -> Mapping[str, Any]:
        return self.manifest["control_counts"]

    @property
    def per_image_owner_counts(self) -> dict[str, int]:
        counts = self.cohort_counts.get("per_image_owner_counts")
        if not isinstance(counts, Mapping):
            _fail("the sealed plan manifest declares no per_image_owner_counts")
        return {str(key): int(value) for key, value in counts.items()}


def load_frozen_plan(plan_dir: Path) -> FrozenPlan:
    """Load the frozen plan and re-prove every digest it declares about itself."""

    plan_dir = Path(plan_dir)
    manifest = _read_json(plan_dir / "manifest.json", "plan manifest")
    if str(manifest.get("schema_version")) != FROZEN_PLAN_MANIFEST_SCHEMA_VERSION:
        _fail(
            f"plan manifest schema {manifest.get('schema_version')!r} is not the frozen "
            f"{FROZEN_PLAN_MANIFEST_SCHEMA_VERSION!r}"
        )
    manifest_digest = assert_self_sealed(
        manifest, digest_key="manifest_content_sha256", label="plan manifest"
    )

    declared = manifest.get("output_file_digests")
    if not isinstance(declared, Mapping):
        _fail("the sealed plan manifest declares no output_file_digests")
    plan_file_sha256: dict[str, str] = {}
    for name in PLAN_FILE_NAMES:
        entry = declared.get(name)
        if not isinstance(entry, Mapping):
            _fail(f"the sealed plan manifest declares no digest for {name!r}")
        path = plan_dir / name
        if not path.is_file():
            _fail(f"sealed plan file {name!r} is missing at {path}")
        observed = sha256_file(path)
        if observed != str(entry.get("sha256")):
            _fail(
                f"sealed plan file {name!r} does not match the digest its own manifest "
                "declares; the frozen plan was modified"
            )
        plan_file_sha256[name] = observed
    unknown = sorted(set(declared) - set(PLAN_FILE_NAMES))
    if unknown:
        _fail(f"the sealed plan manifest declares unknown output file(s) {unknown!r}")

    scorer.validate_primary_cohort_counts(
        u_count=int(manifest["cohort_counts"]["u_bound_crossing_count"]),
        l_count=int(manifest["cohort_counts"]["l_bound_crossing_count"]),
        same_context_ul_count=int(
            manifest["cohort_counts"]["exact_same_context_u_and_l_count"]
        ),
        matched_e_count=int(manifest["cohort_counts"]["matched_e_count"]),
        unmatched_e_count=int(manifest["cohort_counts"]["unmatched_e_count"]),
    )
    scorer.validate_timing_control_count(
        int(manifest["control_counts"]["timing_control_count"])
    )

    plan = FrozenPlan(
        plan_dir=plan_dir,
        manifest=manifest,
        manifest_content_sha256=manifest_digest,
        plan_file_sha256=plan_file_sha256,
    )
    observed_images = tuple(sorted(plan.per_image_owner_counts))
    if observed_images != tuple(sorted(FROZEN_IMAGE_IDS)):
        _fail(
            f"the sealed plan covers images {list(observed_images)!r}, not this unit's frozen "
            f"{list(sorted(FROZEN_IMAGE_IDS))!r}"
        )
    if sum(plan.per_image_owner_counts.values()) != scorer.PRIMARY_OWNER_COUNT_U:
        _fail(
            "the sealed plan's per-image primary owner counts do not sum to the frozen "
            f"{scorer.PRIMARY_OWNER_COUNT_U}"
        )
    return plan


def load_admission(admission_path: Path, *, plan: FrozenPlan) -> dict[str, Any]:
    """The one smoke admission every merged shard must have inherited."""

    admission = _read_json(Path(admission_path), "smoke admission receipt")
    if str(admission.get("schema_version")) != scorer.ADMISSION_SCHEMA_VERSION:
        _fail(
            f"admission receipt schema {admission.get('schema_version')!r} is not "
            f"{scorer.ADMISSION_SCHEMA_VERSION!r}"
        )
    if str(admission.get("unit_id")) != UNIT_ID:
        _fail("admission receipt belongs to another unit")
    assert_self_sealed(
        admission,
        digest_key="admission_content_sha256",
        label="smoke admission receipt",
    )
    if admission.get("plan_manifest_content_sha256") != plan.manifest_content_sha256:
        _fail(
            "the smoke admission was sealed against a different plan manifest; it cannot "
            "admit a capture of this plan"
        )
    reconstructed = scorer.runtime_identity_digest(admission)
    if reconstructed != str(admission.get("runtime_identity_sha256")):
        _fail(
            "the smoke admission's runtime_identity_sha256 does not reconstruct from its own "
            "declared identity fields"
        )
    scorer.validate_smoke_matrix_roles(
        [str(row["role"]) for row in (admission.get("smoke_rows") or ())]
    )
    return admission


# ---------------------------------------------------------------------------
# 3. One shard: read, then prove
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardInput:
    """One capture shard directory, read verbatim before anything is proven."""

    shard_dir: Path
    receipt: dict[str, Any]
    score_rows: list[dict[str, Any]]
    owner_records: list[dict[str, Any]]
    parity: dict[str, Any]
    file_sha256: dict[str, str]


@dataclass(frozen=True)
class ValidatedShard:
    """One shard whose every declared identity has been re-proven."""

    shard_dir: Path
    shard_id: str
    session_image_id: str
    receipt: dict[str, Any]
    score_rows: list[dict[str, Any]]
    owner_records: list[dict[str, Any]]
    parity: dict[str, Any]
    file_sha256: dict[str, str]
    runtime_identity_sha256: str
    admission_content_sha256: str
    primary_owner_ids: tuple[str, ...]
    timing_control_owner_ids: tuple[str, ...]
    tp_replay_control_owner_ids: tuple[str, ...]
    quarantined_primary_owner_ids: tuple[str, ...]


def read_shard(shard_dir: Path) -> ShardInput:
    """Read one shard directory, refusing anything that is not a complete capture."""

    shard_dir = Path(shard_dir)
    if not shard_dir.is_dir():
        _fail(f"shard directory {shard_dir} does not exist")
    present = sorted(entry.name for entry in shard_dir.iterdir())
    if SHARD_QUARANTINE_NAME in present:
        _fail(
            f"shard {shard_dir} published {SHARD_QUARANTINE_NAME!r}: it was stopped by the "
            "quarantine rule and carries no primary evidence"
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
        receipt=_read_json(shard_dir / SHARD_RECEIPT_NAME, f"{shard_dir}/{SHARD_RECEIPT_NAME}"),
        score_rows=_read_jsonl(
            shard_dir / SHARD_SCORES_NAME, f"{shard_dir}/{SHARD_SCORES_NAME}"
        ),
        owner_records=_read_jsonl(
            shard_dir / SHARD_OWNER_RECORDS_NAME, f"{shard_dir}/{SHARD_OWNER_RECORDS_NAME}"
        ),
        parity=_read_json(shard_dir / SHARD_PARITY_NAME, f"{shard_dir}/{SHARD_PARITY_NAME}"),
        file_sha256={
            name: sha256_file(shard_dir / name) for name in sorted(REQUIRED_SHARD_FILES)
        },
    )


def _validate_shard_identity(shard: ShardInput, *, plan: FrozenPlan) -> tuple[str, str]:
    receipt = shard.receipt
    label = f"shard receipt at {shard.shard_dir}"
    if str(receipt.get("schema_version")) != scorer.RECEIPT_SCHEMA_VERSION:
        _fail(
            f"{label} declares schema {receipt.get('schema_version')!r}, not "
            f"{scorer.RECEIPT_SCHEMA_VERSION!r}"
        )
    if str(receipt.get("unit_id")) != UNIT_ID:
        _fail(f"{label} belongs to another unit")
    if str(receipt.get("mode")) != scorer.MODE_CAPTURE:
        _fail(f"{label} is a {receipt.get('mode')!r} receipt, not a capture receipt")
    assert_self_sealed(receipt, digest_key="receipt_content_sha256", label=label)

    if str(receipt.get("scorer_source_sha256")) != FROZEN_SCORER_SOURCE_SHA256:
        _fail(
            f"{label} was produced by scorer revision "
            f"{receipt.get('scorer_source_sha256')!r}, not this unit's frozen "
            f"{FROZEN_SCORER_SOURCE_SHA256!r}"
        )

    plan_block = receipt.get("plan")
    if not isinstance(plan_block, Mapping):
        _fail(f"{label} carries no plan block")
    if str(plan_block.get("manifest_schema_version")) != FROZEN_PLAN_MANIFEST_SCHEMA_VERSION:
        _fail(f"{label} was captured against a plan of another schema revision")
    if plan_block.get("manifest_content_sha256") != plan.manifest_content_sha256:
        _fail(
            f"{label} was captured against plan manifest "
            f"{plan_block.get('manifest_content_sha256')!r}, not the frozen "
            f"{plan.manifest_content_sha256!r}"
        )
    if plan_block.get("builder_source_sha256") != scorer.builder_source_sha256(plan.manifest):
        _fail(f"{label} declares a plan builder digest the frozen manifest does not")
    declared_plan_files = plan_block.get("plan_file_sha256")
    if not isinstance(declared_plan_files, Mapping):
        _fail(f"{label} declares no plan_file_sha256")
    if dict(sorted(declared_plan_files.items())) != dict(sorted(plan.plan_file_sha256.items())):
        _fail(
            f"{label} declares plan file digests that do not match the frozen plan's own "
            "sealed bytes"
        )

    runtime_identity = receipt.get("runtime_identity")
    if not isinstance(runtime_identity, Mapping):
        _fail(f"{label} carries no runtime_identity")
    reconstructed = scorer.runtime_identity_digest(runtime_identity)
    if reconstructed != str(receipt.get("runtime_identity_sha256")):
        _fail(
            f"{label} declares a runtime_identity_sha256 that does not reconstruct from its "
            "own identity fields; the receipt is internally inconsistent"
        )
    source_identity = runtime_identity.get("source_identity")
    if not isinstance(source_identity, Mapping):
        _fail(f"{label} carries no runtime_identity.source_identity")
    if str(source_identity.get(SCORER_SOURCE_IDENTITY_KEY)) != FROZEN_SCORER_SOURCE_SHA256:
        _fail(
            f"{label} ran under scorer source identity "
            f"{source_identity.get(SCORER_SOURCE_IDENTITY_KEY)!r}, not the frozen revision"
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

    denominators = receipt.get("cohort_denominators")
    if not isinstance(denominators, Mapping):
        _fail(f"{label} carries no cohort_denominators")
    scorer.validate_primary_cohort_counts(
        u_count=int(denominators["u_count"]),
        l_count=int(denominators["l_count"]),
        same_context_ul_count=int(denominators["same_context_ul_count"]),
        matched_e_count=int(denominators["matched_e_count"]),
        unmatched_e_count=int(denominators["unmatched_e_count"]),
    )

    quarantine = receipt.get("quarantine")
    if not isinstance(quarantine, Mapping):
        _fail(f"{label} carries no quarantine ledger")
    if bool(quarantine.get("stopped")):
        _fail(f"{label} declares the shard was stopped by the quarantine rule")

    admission_block = receipt.get("admission")
    if not isinstance(admission_block, Mapping):
        _fail(f"{label} carries no inherited admission block")
    admission_digest = admission_block.get("admission_content_sha256")
    if not isinstance(admission_digest, str) or not admission_digest:
        _fail(f"{label} inherited no admission digest")

    return str(receipt["runtime_identity_sha256"]), admission_digest


def _validate_shard_owner_records(
    shard: ShardInput, *, session_image_id: str, shard_id: str
) -> dict[str, tuple[str, ...]]:
    executed = shard.receipt.get("executed")
    if not isinstance(executed, Mapping):
        _fail(f"shard receipt at {shard.shard_dir} carries no executed block")

    declared_ids = [str(value) for value in (executed.get("owner_ids") or ())]
    if len(declared_ids) != len(set(declared_ids)):
        _fail(f"shard {shard_id} declares a duplicate owner id in its executed block")
    if int(executed.get("owner_count", -1)) != len(declared_ids):
        _fail(f"shard {shard_id} declares an owner_count that contradicts its own owner_ids")
    if len(shard.owner_records) != len(declared_ids):
        _fail(
            f"shard {shard_id} published {len(shard.owner_records)} owner records but its "
            f"receipt sealed {len(declared_ids)}"
        )

    by_cohort: dict[str, list[str]] = {cohort: [] for cohort in COHORTS}
    quarantined_primary: list[str] = []
    observed_ids: list[str] = []
    for record in shard.owner_records:
        owner_id = str(record.get("gt_owner_id"))
        observed_ids.append(owner_id)
        if str(record.get("schema_version")) != scorer.OWNER_RECORD_SCHEMA_VERSION:
            _fail(f"owner record {owner_id!r} in shard {shard_id} declares an unknown schema")
        if str(record.get("unit_id")) != UNIT_ID:
            _fail(f"owner record {owner_id!r} in shard {shard_id} belongs to another unit")
        if str(record.get("shard_id")) != shard_id:
            _fail(f"owner record {owner_id!r} was published by a different shard")
        if str(record.get("image_id")) != session_image_id:
            _fail(
                f"owner record {owner_id!r} carries image {record.get('image_id')!r} inside the "
                f"{session_image_id!r} session shard; one image session per shard is a contract"
            )
        if str(record.get("branch_assignment")) != BRANCH_ASSIGNMENT_SENTINEL:
            _fail(
                f"owner record {owner_id!r} carries a branch assignment; branch assignment "
                "belongs to the analysis pass, never to a capture or a merge"
            )
        if str(record.get("secondary_compatibility")) != SECONDARY_COMPATIBILITY_SENTINEL:
            _fail(
                f"owner record {owner_id!r} carries secondary compatibility evidence; it must "
                "stay deferred until every primary branch is sealed"
            )
        cohort = str(record.get("cohort"))
        if cohort not in by_cohort:
            _fail(f"owner record {owner_id!r} declares unknown cohort {cohort!r}")
        by_cohort[cohort].append(owner_id)
        if cohort == PRIMARY_COHORT:
            stratum = str(record.get("stratum"))
            if stratum not in STRATA:
                _fail(
                    f"primary owner record {owner_id!r} declares stratum {stratum!r}; the two "
                    "preregistered native-row strata are the only admissible values"
                )
            variants = set(record.get("ladders") or {})
            if variants != set(PRIMARY_LADDER_VARIANTS):
                _fail(
                    f"primary owner record {owner_id!r} carries ladders {sorted(variants)!r}, "
                    f"not the frozen {sorted(PRIMARY_LADDER_VARIANTS)!r}"
                )
            if bool(record.get("quarantined")):
                quarantined_primary.append(owner_id)

    if sorted(observed_ids) != sorted(declared_ids):
        _fail(
            f"shard {shard_id} published owner ids that do not match the set its receipt "
            "sealed"
        )
    if len(set(observed_ids)) != len(observed_ids):
        _fail(f"shard {shard_id} published a duplicate owner record")

    return {
        "primary": tuple(sorted(by_cohort[PRIMARY_COHORT])),
        "timing_control": tuple(sorted(by_cohort[TIMING_CONTROL_COHORT])),
        "tp_replay_control": tuple(sorted(by_cohort[TP_REPLAY_CONTROL_COHORT])),
        "quarantined_primary": tuple(sorted(quarantined_primary)),
    }


def _validate_shard_score_rows(
    shard: ShardInput, *, session_image_id: str, shard_id: str
) -> None:
    executed = shard.receipt["executed"]
    if int(executed.get("score_row_count", -1)) != len(shard.score_rows):
        _fail(
            f"shard {shard_id} published {len(shard.score_rows)} score rows but its receipt "
            f"sealed {executed.get('score_row_count')!r}"
        )
    request_ids: set[str] = set()
    for row in shard.score_rows:
        if str(row.get("schema_version")) != scorer.SCHEMA_VERSION:
            _fail(f"a score row in shard {shard_id} declares an unknown schema")
        if str(row.get("unit_id")) != UNIT_ID:
            _fail(f"a score row in shard {shard_id} belongs to another unit")
        if str(row.get("shard_id")) != shard_id:
            _fail(f"a score row in shard {shard_id} was published by a different shard")
        if str(row.get("image_id")) != session_image_id:
            _fail(
                f"a score row in shard {shard_id} carries image {row.get('image_id')!r}; one "
                "image session per shard is a contract"
            )
        if str(row.get("readout_tier")) != scorer.PRIMARY_READOUT_TIER:
            _fail(
                f"a score row in shard {shard_id} is tier {row.get('readout_tier')!r}; this "
                "merge is primary-only and never carries a deferred secondary readout"
            )
        if str(row.get("request_family")) == scorer.REQUEST_DOWNSTREAM_COMPATIBILITY:
            _fail(
                f"shard {shard_id} executed a deferred downstream-compatibility request; "
                "secondary evidence is sealed away until every primary branch is assigned"
            )
        request_ids.add(str(row.get("request_id")))
    if sha256_json(sorted(request_ids)) != str(executed.get("request_ids_sha256")):
        _fail(
            f"shard {shard_id} published score rows whose request ids do not reproduce the "
            "digest its own receipt sealed"
        )


def _validate_shard_parity(shard: ShardInput, *, shard_id: str, session_image_id: str) -> None:
    parity = shard.parity
    if str(parity.get("schema_version")) != scorer.PARITY_SCHEMA_VERSION:
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
    receipt_admission = shard.receipt["admission"]
    if inherited.get("admission_content_sha256") != receipt_admission.get(
        "admission_content_sha256"
    ):
        _fail(
            f"shard {shard_id} disagrees with itself about which smoke admission it inherited"
        )


def validate_shard(shard: ShardInput, *, plan: FrozenPlan, admission: Mapping[str, Any]) -> ValidatedShard:
    """Re-prove every identity one shard declares, then accept its raw rows."""

    runtime_identity_sha256, admission_digest = _validate_shard_identity(shard, plan=plan)
    if admission_digest != str(admission.get("admission_content_sha256")):
        _fail(
            f"shard at {shard.shard_dir} inherited admission {admission_digest!r}, not the "
            f"merged run's {admission.get('admission_content_sha256')!r}"
        )
    if str(shard.receipt["runtime_identity_sha256"]) != str(
        admission.get("runtime_identity_sha256")
    ):
        _fail(
            f"shard at {shard.shard_dir} ran under a runtime the smoke admission never "
            "admitted; cached execution and batching are never inherited across a changed "
            "runtime"
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

    cohorts = _validate_shard_owner_records(
        shard, session_image_id=session_image_id, shard_id=shard_id
    )
    _validate_shard_score_rows(shard, session_image_id=session_image_id, shard_id=shard_id)
    _validate_shard_parity(shard, shard_id=shard_id, session_image_id=session_image_id)

    expected_primary = plan.per_image_owner_counts[session_image_id]
    if len(cohorts["primary"]) != expected_primary:
        _fail(
            f"shard {shard_id} carries {len(cohorts['primary'])} primary owners for image "
            f"{session_image_id!r}, not the sealed plan's {expected_primary}"
        )

    return ValidatedShard(
        shard_dir=shard.shard_dir,
        shard_id=shard_id,
        session_image_id=session_image_id,
        receipt=shard.receipt,
        score_rows=shard.score_rows,
        owner_records=shard.owner_records,
        parity=shard.parity,
        file_sha256=shard.file_sha256,
        runtime_identity_sha256=runtime_identity_sha256,
        admission_content_sha256=admission_digest,
        primary_owner_ids=cohorts["primary"],
        timing_control_owner_ids=cohorts["timing_control"],
        tp_replay_control_owner_ids=cohorts["tp_replay_control"],
        quarantined_primary_owner_ids=cohorts["quarantined_primary"],
    )


# ---------------------------------------------------------------------------
# 4. Cross-shard closure
# ---------------------------------------------------------------------------


def _assert_one_shard_per_frozen_image(shards: Sequence[ValidatedShard]) -> None:
    by_image: dict[str, list[str]] = {}
    for shard in shards:
        by_image.setdefault(shard.session_image_id, []).append(shard.shard_id)
    duplicated = sorted(image for image, ids in by_image.items() if len(ids) > 1)
    if duplicated:
        _fail(
            f"image(s) {duplicated!r} were merged from more than one shard; exactly one "
            "admitted shard per frozen image is a contract"
        )
    missing = sorted(set(FROZEN_IMAGE_IDS) - set(by_image))
    if missing:
        _fail(
            f"no admitted shard was supplied for image(s) {missing!r}; a partial merge is "
            "never published"
        )
    if len(shards) != EXPECTED_SHARD_COUNT:
        _fail(
            f"{len(shards)} shards were supplied for {EXPECTED_SHARD_COUNT} frozen images"
        )
    shard_ids = [shard.shard_id for shard in shards]
    if len(set(shard_ids)) != len(shard_ids):
        _fail("two shards declare the same shard_id; a duplicated capture fails closed")


def _assert_one_runtime(shards: Sequence[ValidatedShard]) -> str:
    digests = sorted({shard.runtime_identity_sha256 for shard in shards})
    if len(digests) != 1:
        _fail(
            f"the supplied shards span {len(digests)} runtime identities {digests!r}; evidence "
            "from mixed runtimes is never merged into one conclusion"
        )
    admissions = sorted({shard.admission_content_sha256 for shard in shards})
    if len(admissions) != 1:
        _fail(
            f"the supplied shards inherited {len(admissions)} different smoke admissions "
            f"{admissions!r}"
        )
    denominators = sorted(
        {sha256_json(shard.receipt["cohort_denominators"]) for shard in shards}
    )
    if len(denominators) != 1:
        _fail("the supplied shards disagree about this unit's sealed cohort denominators")
    return digests[0]


def _assert_cohort_closure(shards: Sequence[ValidatedShard], *, plan: FrozenPlan) -> dict[str, Any]:
    primary: list[str] = []
    timing: list[str] = []
    tp_replay: list[str] = []
    quarantined: list[str] = []
    for shard in shards:
        primary.extend(shard.primary_owner_ids)
        timing.extend(shard.timing_control_owner_ids)
        tp_replay.extend(shard.tp_replay_control_owner_ids)
        quarantined.extend(shard.quarantined_primary_owner_ids)

    for label, owner_ids in (
        ("primary", primary),
        ("timing control", timing),
        ("TP replay control", tp_replay),
    ):
        if len(set(owner_ids)) != len(owner_ids):
            _fail(f"the {label} cohort carries a duplicated owner across shards")
    overlap = sorted(set(primary) & (set(timing) | set(tp_replay)))
    if overlap:
        _fail(
            f"owner(s) {overlap!r} appear in both the primary cohort and a control cohort; "
            "a mismatch is reported, never patched by allowing overlap"
        )

    if len(primary) != scorer.PRIMARY_OWNER_COUNT_U:
        _fail(
            f"the merged primary cohort has {len(primary)} owners, not the frozen "
            f"{scorer.PRIMARY_OWNER_COUNT_U}"
        )
    declared_timing = [
        str(value) for value in (plan.control_counts.get("timing_control_owner_ids") or ())
    ]
    if sorted(timing) != sorted(declared_timing):
        _fail("the merged timing-control cohort does not reproduce the sealed control registry")
    declared_tp = [
        str(value) for value in (plan.control_counts.get("tp_replay_control_owner_ids") or ())
    ]
    if sorted(tp_replay) != sorted(declared_tp):
        _fail("the merged TP-replay control cohort does not reproduce the sealed registry")

    if len(quarantined) > MAX_PRIMARY_QUARANTINES:
        _fail(
            f"{len(quarantined)} primary owners are quarantined ({sorted(quarantined)!r}); "
            f"more than {MAX_PRIMARY_QUARANTINES} stops this unit without interpretation, and "
            "the runtime alignment is repaired rather than the thresholds changed"
        )
    return {
        "primary_owner_ids": sorted(primary),
        "timing_control_owner_ids": sorted(timing),
        "tp_replay_control_owner_ids": sorted(tp_replay),
        "quarantined_primary_owner_ids": sorted(quarantined),
    }


def _assert_strata_closure(owner_records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    matched = sorted(
        str(record["gt_owner_id"])
        for record in owner_records
        if str(record.get("cohort")) == PRIMARY_COHORT
        and str(record.get("stratum")) == MATCHED_E_STRATUM
    )
    unmatched = sorted(
        str(record["gt_owner_id"])
        for record in owner_records
        if str(record.get("cohort")) == PRIMARY_COHORT
        and str(record.get("stratum")) == UNMATCHED_E_STRATUM
    )
    return scorer.validate_primary_cohort_counts(
        u_count=len(matched) + len(unmatched),
        l_count=scorer.PRIMARY_OWNER_COUNT_L,
        same_context_ul_count=scorer.PRIMARY_OWNER_COUNT_SAME_CONTEXT_UL,
        matched_e_count=len(matched),
        unmatched_e_count=len(unmatched),
    )


# ---------------------------------------------------------------------------
# 5. Deterministic merged bytes
# ---------------------------------------------------------------------------


def _score_row_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("image_id")),
        str(row.get("request_id")),
        sha256_bytes(canonical_json_bytes(row)),
    )


def _owner_record_sort_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return (str(record.get("image_id")), str(record.get("gt_owner_id")))


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def merge_shards(
    *,
    shard_dirs: Sequence[Path],
    plan_dir: Path,
    admission_path: Path,
) -> dict[str, Any]:
    """Prove every shard, then build the merged bytes and the merge receipt."""

    plan = load_frozen_plan(plan_dir)
    admission = load_admission(admission_path, plan=plan)

    seen_dirs: set[Path] = set()
    validated: list[ValidatedShard] = []
    for shard_dir in shard_dirs:
        resolved = Path(shard_dir).resolve()
        if resolved in seen_dirs:
            _fail(f"shard directory {resolved} was supplied twice")
        seen_dirs.add(resolved)
        validated.append(validate_shard(read_shard(resolved), plan=plan, admission=admission))

    validated.sort(key=lambda shard: (shard.session_image_id, shard.shard_id))
    _assert_one_shard_per_frozen_image(validated)
    runtime_identity_sha256 = _assert_one_runtime(validated)
    cohort_closure = _assert_cohort_closure(validated, plan=plan)

    owner_records = sorted(
        (record for shard in validated for record in shard.owner_records),
        key=_owner_record_sort_key,
    )
    score_rows = sorted(
        (row for shard in validated for row in shard.score_rows),
        key=_score_row_sort_key,
    )
    parity_rows = [shard.parity for shard in validated]
    strata = _assert_strata_closure(owner_records)

    owner_records_bytes = _jsonl_bytes(owner_records)
    score_rows_bytes = _jsonl_bytes(score_rows)
    parity_bytes = _jsonl_bytes(parity_rows)

    receipt: dict[str, Any] = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "merger_source_sha256": sha256_file(Path(__file__).resolve()),
        "scorer_source_sha256": FROZEN_SCORER_SOURCE_SHA256,
        "artifact_determinism": "no_wall_clock_or_host_specific_field_is_sealed",
        "plan": {
            "plan_dir": str(plan.plan_dir),
            "manifest_schema_version": FROZEN_PLAN_MANIFEST_SCHEMA_VERSION,
            "manifest_content_sha256": plan.manifest_content_sha256,
            "builder_source_sha256": scorer.builder_source_sha256(plan.manifest),
            "plan_file_sha256": dict(sorted(plan.plan_file_sha256.items())),
            "lineage": plan.manifest.get("lineage"),
            "support_calibration": plan.manifest.get("support_calibration"),
        },
        "admission": {
            "path": str(Path(admission_path)),
            "admission_content_sha256": str(admission.get("admission_content_sha256")),
            "smoke_shard_id": admission.get("smoke_shard_id"),
            "smoke_image_id": admission.get("smoke_image_id"),
            "cache_admitted": admission.get("cache_admitted"),
            "admitted_batch_size": admission.get("admitted_batch_size"),
        },
        "runtime_identity_sha256": runtime_identity_sha256,
        "runtime_identity": dict(validated[0].receipt["runtime_identity"]),
        "cohort_denominators": dict(validated[0].receipt["cohort_denominators"]),
        "observed_cohorts": {
            **cohort_closure,
            "primary_owner_count": len(cohort_closure["primary_owner_ids"]),
            "timing_control_owner_count": len(cohort_closure["timing_control_owner_ids"]),
            "tp_replay_control_owner_count": len(cohort_closure["tp_replay_control_owner_ids"]),
            "matched_e_count": strata["matched_e_count"],
            "unmatched_e_count": strata["unmatched_e_count"],
            "maximum_primary_quarantines": MAX_PRIMARY_QUARANTINES,
        },
        "shards": [
            {
                "shard_id": shard.shard_id,
                "session_image_id": shard.session_image_id,
                "shard_dir": str(shard.shard_dir),
                "file_sha256": dict(sorted(shard.file_sha256.items())),
                "receipt_content_sha256": str(shard.receipt["receipt_content_sha256"]),
                "owner_count": len(shard.owner_records),
                "score_row_count": len(shard.score_rows),
                "primary_owner_ids": list(shard.primary_owner_ids),
                "quarantined_primary_owner_ids": list(shard.quarantined_primary_owner_ids),
            }
            for shard in validated
        ],
        "policy": {
            "branch_assignment_performed": False,
            "branch_assignment_owner": "the later analysis pass over merged primary records",
            "secondary_compatibility_merged": False,
            "rows_preserved_verbatim": True,
            "merged_readout_tier": scorer.PRIMARY_READOUT_TIER,
        },
        "output_file_digests": {
            MERGED_OWNER_RECORDS_NAME: {
                "path": MERGED_OWNER_RECORDS_NAME,
                "byte_size": len(owner_records_bytes),
                "row_count": len(owner_records),
                "sha256": sha256_bytes(owner_records_bytes),
            },
            MERGED_SCORES_NAME: {
                "path": MERGED_SCORES_NAME,
                "byte_size": len(score_rows_bytes),
                "row_count": len(score_rows),
                "sha256": sha256_bytes(score_rows_bytes),
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
            MERGED_OWNER_RECORDS_NAME: owner_records_bytes,
            MERGED_SCORES_NAME: score_rows_bytes,
            MERGED_PARITY_NAME: parity_bytes,
            MERGE_RECEIPT_NAME: canonical_json_bytes(receipt) + b"\n",
        },
        "owner_records": owner_records,
        "score_rows": score_rows,
        "parity_rows": parity_rows,
        "shards": validated,
    }


# ---------------------------------------------------------------------------
# 6. Publish
# ---------------------------------------------------------------------------


def publish_merge(output_dir: Path, files: Mapping[str, bytes]) -> dict[str, Any]:
    """Create-or-identical publish: a rerun is a no-op, a drift fails closed."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        present = sorted(entry.name for entry in output_dir.iterdir())
        missing = sorted(set(files) - set(present))
        unexpected = sorted(set(present) - set(files))
        differing = sorted(
            name
            for name in files
            if name in present and (output_dir / name).read_bytes() != files[name]
        )
        if not missing and not unexpected and not differing:
            return {
                "output_dir": str(output_dir),
                "published": False,
                "publish_mode": "no_op_identical_rerun",
                "file_names": sorted(files),
            }
        _fail(
            f"refusing to publish into existing merged directory {output_dir}: it is not a "
            f"byte-identical merge (missing={missing!r}, differing={differing!r}, "
            f"unexpected={unexpected!r}); the existing directory is left untouched"
        )
    staging = output_dir.parent / f"{output_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        _fail(f"staging directory {staging} already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True)
    published = False
    try:
        for name in sorted(files):
            (staging / name).write_bytes(files[name])
        os.rename(staging, output_dir)
        published = True
    finally:
        if not published:
            for child in sorted(staging.iterdir()):
                child.unlink()
            staging.rmdir()
    return {
        "output_dir": str(output_dir),
        "published": True,
        "publish_mode": "atomic_staging_directory_rename",
        "file_names": sorted(files),
    }


# ---------------------------------------------------------------------------
# 7. CLI
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
        _fail(f"shard root {shard_root} contains no capture shard directory")
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
        help="One explicit capture shard directory; repeat once per frozen image",
    )
    parser.add_argument(
        "--plan-dir", type=Path, required=True, help="Frozen v3 CPU plan directory"
    )
    parser.add_argument(
        "--admission",
        type=Path,
        required=True,
        help="Sealed smoke admission receipt every shard inherited",
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
            admission_path=args.admission,
        )
        published = publish_merge(Path(args.output_dir), result["files"])
    except MergeContractError as exc:
        raise SystemExit(f"merge contract violated: {exc}") from exc
    print(
        json.dumps(
            {
                "merged": published,
                "receipt_content_sha256": result["receipt"]["receipt_content_sha256"],
                "owner_record_count": len(result["owner_records"]),
                "score_row_count": len(result["score_rows"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
