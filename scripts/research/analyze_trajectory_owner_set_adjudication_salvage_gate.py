#!/usr/bin/env python3
"""Build the frozen train-only Stage-Zero adjudication salvage gate.

This module deliberately keeps the category-capacity search independent from
the current owner assignment.  Production materialization is pinned to the
completed admission census and the exact v2 B16 panel; the pure search and
certificate functions are exposed for independent replay.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import uuid

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.research.analyze_trajectory_owner_set_admission_census as census
from scripts.research.analyze_individual_trajectory_union_support import (
    _normalize_category,
)
from scripts.research.assemble_constant_dose_breadth_state_banks import (
    AssemblyError,
    _candidate_pool,
    _read_json_exact_bytes,
    _split_membership,
    load_v2_b16_panel_adapter,
)
from src.config.fingerprint import sha256_file, sha256_json
from src.inference.backend import token_ids_sha256


SCHEMA_VERSION = "trajectory_owner_set_adjudication_salvage_gate.stage_zero.v1"
SEARCH_SCHEMA_VERSION = f"{SCHEMA_VERSION}.category_capacity_search.v1"
EXPECTED_CENSUS_IMAGE_COUNT = 2_004
EXPECTED_POPULATION_COUNT = 1_622
EXPECTED_CANDIDATES_PER_IMAGE = 17
REQUIRED_ADDITIONAL_ADMISSIONS = 248

CENSUS_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-23-trajectory-owner-set-admission-census/production-v1"
)
FROZEN_IMAGE_CENSUS_PATH = CENSUS_ROOT / "image-census.jsonl"
FROZEN_SUMMARY_PATH = CENSUS_ROOT / "summary.json"
FROZEN_RECEIPT_PATH = CENSUS_ROOT / "receipt.json"
FROZEN_IMAGE_CENSUS_SHA256 = (
    "c375e09df621714da82118a2cdf3d80205fd7429424b1a853f38ec60e68e1805"
)
FROZEN_SUMMARY_SHA256 = (
    "2cc526310e8bca2de11f2d9d7c2e42511a017a882455dd1ebdb671b0c0df4514"
)
FROZEN_RECEIPT_SHA256 = (
    "d322335c5e92f3fd4860b9524b8d709a3d01fe9e1cbc625dfa1d4bf002d12fe2"
)
FROZEN_CANDIDATE_POOL_PATH = census.FROZEN_CANDIDATE_POOL_PATH
FROZEN_CANDIDATE_POOL_SHA256 = (
    "133afcf6659b78d71893e9f79e568dca676515a05c7fe23f3c237de01350b7e2"
)
FROZEN_SPLIT_RECEIPT_PATH = census.FROZEN_SPLIT_RECEIPT_PATH
FROZEN_SPLIT_RECEIPT_SHA256 = census.EXPECTED_SPLIT_RECEIPT_SHA256
FROZEN_TRAIN_CANDIDATE_PATH = census.FROZEN_SPLIT_OUTPUT_PATHS["train_candidate"]
FROZEN_TRAIN_CANDIDATE_SHA256 = census.EXPECTED_TRAIN_CANDIDATE_SHA256
FROZEN_SAMPLED_ROOT = census.FROZEN_SAMPLED_ROOT
FROZEN_SOURCE_ROOT = census.FROZEN_SOURCE_ROOT
FROZEN_SAMPLED_MANIFEST_SET_SHA256 = census.FROZEN_SAMPLED_MANIFEST_SET_SHA256
FROZEN_SOURCE_MANIFEST_SET_SHA256 = census.FROZEN_SOURCE_MANIFEST_SET_SHA256
FROZEN_EXECUTION_MODEL_IDENTITY_SHA256 = (
    census.FROZEN_EXECUTION_MODEL_IDENTITY_SHA256
)
FROZEN_TOKENIZER_IDENTITY_SHA256 = census.FROZEN_TOKENIZER_IDENTITY_SHA256
FROZEN_OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-23-trajectory-owner-set-adjudication-salvage-gate/stage-zero-v1"
)

SOURCE_SNAPSHOT_PATHS = (
    "research/investigations/qwen3-vl-dense-enumeration/experiments/"
    "2026-07-23-trajectory-owner-set-adjudication-salvage-gate/unit.md",
    "scripts/research/analyze_trajectory_owner_set_adjudication_salvage_gate.py",
    "scripts/research/analyze_trajectory_owner_set_admission_census.py",
    "scripts/research/assemble_constant_dose_breadth_state_banks.py",
    "scripts/research/analyze_individual_trajectory_union_support.py",
    "src/config/fingerprint.py",
    "src/inference/backend.py",
)

ACCEPTED_PROJECTION_STATUSES = {"accepted_budget", "accepted_natural_end"}
ACCEPTED_PARSER_STATUSES = {"accepted", "accepted_with_drops"}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssemblyError(f"{label} must be an object")
    return value


def _int_count(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise AssemblyError(f"{label} must be a nonnegative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise AssemblyError(f"{label} must be a nonnegative integer") from exc
    if result < 0:
        raise AssemblyError(f"{label} must be a nonnegative integer")
    return result


def _route_sort_key(candidate_id: str) -> tuple[int, str]:
    return (0 if candidate_id == "source-b16" else 1, candidate_id)


def parser_usable(
    *, projection_status: str, parser_status: str, invalid_before_b16_count: int
) -> bool:
    """Return the exact Stage-Zero parser-usability predicate."""

    return (
        projection_status in ACCEPTED_PROJECTION_STATUSES
        and parser_status in ACCEPTED_PARSER_STATUSES
        and invalid_before_b16_count == 0
    )


def _candidate_state(
    *,
    candidate_id: str,
    route_row: Mapping[str, Any],
    route_evidence: Mapping[str, Any],
    assignment: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract only immutable token, parser, and ordered-category facts."""

    projection = route_row.get(
        "_source_b16_provenance"
        if candidate_id == "source-b16"
        else "_sampled_b16_provenance"
    )
    projection = _mapping(projection, f"{candidate_id}.projection")
    projection_status = str(projection.get("status", ""))
    parser = _mapping(route_evidence.get("parser"), f"{candidate_id}.parser")
    parser_status = str(parser.get("parse_status", ""))
    row_counts = _mapping(
        assignment.get("row_counts", {}), f"{candidate_id}.row_counts"
    )
    invalid_count = max(
        _int_count(
            assignment.get("malformed_row_count", 0),
            f"{candidate_id}.malformed_row_count",
        ),
        _int_count(
            row_counts.get("malformed", 0),
            f"{candidate_id}.row_counts.malformed",
        ),
    )
    raw_rows = assignment.get("row_assignment_receipts")
    if not isinstance(raw_rows, list) or any(
        not isinstance(item, Mapping) for item in raw_rows
    ):
        raise AssemblyError(f"{candidate_id}.row_assignment_receipts must be objects")
    rows = sorted(
        (dict(item) for item in raw_rows),
        key=lambda item: (
            _int_count(item.get("generated_row_index", 0), "generated_row_index"),
            str(item.get("prediction_id", "")),
        ),
    )
    categories = [_normalize_category(item.get("category")) for item in rows]
    if any(not item for item in categories):
        raise AssemblyError(f"{candidate_id} has an empty normalized category")
    projected_ids = route_row.get("generated_token_ids")
    if not isinstance(projected_ids, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in projected_ids
    ):
        raise AssemblyError(f"{candidate_id} lacks projected generated token IDs")
    token_hash = str(route_row.get("generated_token_ids_sha256", ""))
    provenance_hash = str(projection.get("projected_token_ids_sha256", ""))
    provenance_count = _int_count(
        projection.get("projected_token_count"),
        f"{candidate_id}.projected_token_count",
    )
    raw_hash = str(projection.get("raw_generated_token_ids_sha256", ""))
    raw_count = _int_count(
        projection.get("raw_generated_token_count"),
        f"{candidate_id}.raw_generated_token_count",
    )
    if (
        len(token_hash) != 64
        or token_ids_sha256(projected_ids) != token_hash
        or provenance_hash != token_hash
        or provenance_count != len(projected_ids)
        or len(raw_hash) != 64
        or raw_count < provenance_count
    ):
        raise AssemblyError(f"{candidate_id} B16 token provenance differs")
    return {
        "candidate_id": candidate_id,
        "projected_token_sha256": token_hash,
        "projection_status": projection_status,
        "parser_status": parser_status,
        "invalid_before_b16_count": invalid_count,
        "ordered_categories": categories,
        "parser_usable": parser_usable(
            projection_status=projection_status,
            parser_status=parser_status,
            invalid_before_b16_count=invalid_count,
        ),
    }


def collapse_exact_token_states(
    candidate_states: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse exact tokens, failing if parser or category semantics disagree."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for state in candidate_states:
        token_hash = str(state.get("projected_token_sha256", ""))
        if len(token_hash) != 64:
            raise AssemblyError("candidate state lacks a projected-token SHA-256")
        grouped.setdefault(token_hash, []).append(state)
    representatives: list[dict[str, Any]] = []
    for token_hash in sorted(grouped):
        aliases = sorted(
            grouped[token_hash],
            key=lambda item: _route_sort_key(str(item.get("candidate_id", ""))),
        )
        categories = [str(item) for item in aliases[0].get("ordered_categories", [])]
        usable = bool(aliases[0].get("parser_usable"))
        if any(
            [str(value) for value in item.get("ordered_categories", [])] != categories
            or bool(item.get("parser_usable")) != usable
            for item in aliases[1:]
        ):
            raise AssemblyError(
                f"exact-token aliases disagree on category/parser state: {token_hash}"
            )
        counts = Counter(categories)
        representatives.append(
            {
                "representative_candidate_id": str(aliases[0]["candidate_id"]),
                "alias_candidate_ids": [str(item["candidate_id"]) for item in aliases],
                "projected_token_sha256": token_hash,
                "parser_usable": usable,
                "ordered_categories": categories,
                "category_support": sorted(counts),
                "category_multiplicities": dict(sorted(counts.items())),
                "first_category": categories[0] if categories else None,
            }
        )
    return representatives


def _attempt_result(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    k: Mapping[str, Any],
    d: Mapping[str, Any] | None,
) -> tuple[str, dict[str, int]]:
    support_a = set(str(item) for item in a["category_support"])
    support_b = set(str(item) for item in b["category_support"])
    support_k = set(str(item) for item in k["category_support"])
    if support_a != support_b:
        return "higher_support_mismatch", {}
    if not support_k <= support_a:
        return "lower_support_not_subset", {}
    high = [a, b]
    if support_k:
        if d is None:
            return "nonempty_lower_missing_d", {}
        if set(str(item) for item in d["category_support"]) != support_a:
            return "d_support_mismatch", {}
        if d.get("first_category") != k.get("first_category"):
            return "d_first_category_mismatch", {}
        high.append(d)
    upper = {
        category: min(
            _int_count(item["category_multiplicities"][category], category)
            for item in high
        )
        for category in sorted(support_a)
    }
    if a.get("first_category") == b.get("first_category"):
        first = str(a.get("first_category", ""))
        if not first or upper.get(first, 0) < 2:
            return "same_first_category_capacity_lt_two", upper
    if support_k == support_a and max(upper.values(), default=0) < 2:
        return "equal_support_lacks_strict_capacity", upper
    return "valid", upper


def _attempt_identity(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    k: Mapping[str, Any],
    d: Mapping[str, Any] | None,
    result: str,
) -> dict[str, Any]:
    return {
        "a": str(a["projected_token_sha256"]),
        "b": str(b["projected_token_sha256"]),
        "k": str(k["projected_token_sha256"]),
        "d": None if d is None else str(d["projected_token_sha256"]),
        "result": result,
    }


def _witness(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    k: Mapping[str, Any],
    d: Mapping[str, Any] | None,
    upper: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "a_token_sha256": str(a["projected_token_sha256"]),
        "b_token_sha256": str(b["projected_token_sha256"]),
        "k_token_sha256": str(k["projected_token_sha256"]),
        "d_token_sha256": None if d is None else str(d["projected_token_sha256"]),
        "a_candidate_id": str(a["representative_candidate_id"]),
        "b_candidate_id": str(b["representative_candidate_id"]),
        "k_candidate_id": str(k["representative_candidate_id"]),
        "d_candidate_id": (
            None if d is None else str(d["representative_candidate_id"])
        ),
        "higher_category_capacity": dict(sorted(upper.items())),
        "empty_lower_vacuity": not bool(k["category_support"]),
    }


def _search(
    representatives: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    usable = sorted(
        (dict(item) for item in representatives if bool(item.get("parser_usable"))),
        key=lambda item: (
            str(item["projected_token_sha256"]),
            str(item["representative_candidate_id"]),
        ),
    )
    digest = hashlib.sha256()
    rejection_counts: Counter[str] = Counter()
    attempt_count = 0
    for a_index, a in enumerate(usable):
        for b in usable[a_index + 1 :]:
            high_tokens = {
                str(a["projected_token_sha256"]),
                str(b["projected_token_sha256"]),
            }
            for k in usable:
                if str(k["projected_token_sha256"]) in high_tokens:
                    continue
                d_values: Sequence[Mapping[str, Any] | None]
                if k["category_support"]:
                    d_values = [
                        item
                        for item in usable
                        if item["projected_token_sha256"]
                        != k["projected_token_sha256"]
                    ]
                else:
                    d_values = [None]
                for d in d_values:
                    result, upper = _attempt_result(a, b, k, d)
                    attempt = _attempt_identity(a, b, k, d, result)
                    digest.update(
                        json.dumps(
                            attempt,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ).encode("utf-8")
                        + b"\n"
                    )
                    attempt_count += 1
                    if result == "valid":
                        return _witness(a, b, k, d, upper), {
                            "attempt_count_before_canonical_witness": attempt_count,
                            "attempt_prefix_sha256": digest.hexdigest(),
                        }
                    rejection_counts[result] += 1
    certificate = {
        "schema_version": SEARCH_SCHEMA_VERSION,
        "representative_state_sha256": sha256_json(usable),
        "parser_usable_representative_count": len(usable),
        "attempt_count": attempt_count,
        "attempts_sha256": digest.hexdigest(),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "valid_witness_count": 0,
    }
    return None, certificate


def search_category_possibility(
    representatives: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return one canonical witness or a replayable exhaustive certificate."""

    witness, evidence = _search(representatives)
    if witness is not None:
        return {
            "possible": True,
            "canonical_witness": witness,
            "canonical_search_prefix": evidence,
            "impossibility_certificate": None,
        }
    return {
        "possible": False,
        "canonical_witness": None,
        "canonical_search_prefix": None,
        "impossibility_certificate": evidence,
    }


def replay_impossibility_certificate(
    representatives: Sequence[Mapping[str, Any]], certificate: Mapping[str, Any]
) -> bool:
    """Replay and exactly compare a claimed exhaustive impossibility proof."""

    witness, replayed = _search(representatives)
    return witness is None and dict(certificate) == replayed


def _image_category_record(
    image_id: str, adapter: Mapping[str, Any], *, reverse_input: bool
) -> dict[str, Any]:
    result = _mapping(adapter["image_results"][image_id], f"image {image_id}")
    evidence = _mapping(result.get("trajectory_evidence"), "trajectory_evidence")
    budgets = result.get("budgets")
    if not isinstance(budgets, list) or len(budgets) != 1:
        raise AssemblyError(f"image {image_id} lacks one exact budget")
    assignments = _mapping(budgets[0].get("trajectory_assignments"), "assignments")
    route_ids = ["source-b16", *(f"sample-{index:02d}" for index in range(16))]
    if reverse_input:
        route_ids.reverse()
    states: list[dict[str, Any]] = []
    for route_id in route_ids:
        route_row = (
            adapter["source_rows"].get((image_id, 0))
            if route_id == "source-b16"
            else adapter["sampled_rows"].get((image_id, int(route_id[-2:])))
        )
        if not isinstance(route_row, Mapping):
            raise AssemblyError(f"image {image_id} lacks {route_id}")
        states.append(
            _candidate_state(
                candidate_id=route_id,
                route_row=route_row,
                route_evidence=_mapping(evidence.get(route_id), route_id),
                assignment=_mapping(assignments.get(route_id), route_id),
            )
        )
    states.sort(key=lambda item: _route_sort_key(str(item["candidate_id"])))
    representatives = collapse_exact_token_states(states)
    return {
        "schema_version": SCHEMA_VERSION,
        "image_id": image_id,
        "candidate_count": len(states),
        "exact_token_representative_count": len(representatives),
        "exact_token_duplicate_count": len(states) - len(representatives),
        "parser_usable_candidate_count": sum(
            bool(item["parser_usable"]) for item in states
        ),
        "candidate_states": states,
        "exact_token_representatives": representatives,
    }


def analyze_stage_zero(
    adapter: Mapping[str, Any], population_ids: Iterable[str], *, reverse_input: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Analyze a population and return canonical category and possibility rows."""

    ids = sorted((str(item) for item in population_ids), key=int, reverse=reverse_input)
    category_records = [
        _image_category_record(image_id, adapter, reverse_input=reverse_input)
        for image_id in ids
    ]
    category_records.sort(key=lambda item: int(item["image_id"]))
    possibility_records: list[dict[str, Any]] = []
    for category_record in category_records:
        image_id = str(category_record["image_id"])
        search = search_category_possibility(
            category_record["exact_token_representatives"]
        )
        possibility_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "image_id": image_id,
                **search,
            }
        )
    possible_ids = sorted(
        (
            str(item["image_id"])
            for item in possibility_records
            if bool(item["possible"])
        ),
        key=int,
    )
    impossible_ids = sorted(
        (
            str(item["image_id"])
            for item in possibility_records
            if not bool(item["possible"])
        ),
        key=int,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "terminal_status": "completed",
        "population_scope": "train_only_censored_nonpassing_U",
        "population_count": len(category_records),
        "possible_pool_count": len(possible_ids),
        "impossible_pool_count": len(impossible_ids),
        "required_additional_admission_count": REQUIRED_ADDITIONAL_ADMISSIONS,
        "deterministic_frozen_panel_impossibility": (
            len(possible_ids) < REQUIRED_ADDITIONAL_ADMISSIONS
        ),
        "possible_image_ids": possible_ids,
        "impossible_image_ids": impossible_ids,
        "ordered_possible_image_ids_sha256": sha256_json(possible_ids),
        "ordered_impossible_image_ids_sha256": sha256_json(impossible_ids),
        "category_state_sha256": sha256_json(category_records),
        "possibility_census_sha256": sha256_json(possibility_records),
    }
    return category_records, possibility_records, summary


def derive_population(
    census_records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Derive the exact frozen 1,622-image population U."""

    if len(census_records) != EXPECTED_CENSUS_IMAGE_COUNT:
        raise AssemblyError("completed census does not contain 2,004 images")
    image_ids = [str(item.get("image_id", "")) for item in census_records]
    if len(set(image_ids)) != len(image_ids):
        raise AssemblyError("completed census image identifiers are not unique")
    result = sorted(
        (
            str(item["image_id"])
            for item in census_records
            if not bool(item.get("fully_adjudicable"))
            and not bool(
                _mapping(item.get("admission"), "admission").get(
                    "primary_natural_alias_admitted"
                )
            )
        ),
        key=int,
    )
    if len(result) != EXPECTED_POPULATION_COUNT:
        raise AssemblyError("derived population U does not contain exactly 1,622 images")
    return result


def _read_jsonl(
    path: Path, *, expected_sha256: str | None = None
) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise AssemblyError(f"{path} changed before JSONL decoding")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise AssemblyError(f"{path} is not UTF-8 JSONL") from exc
    if any(not line.strip() for line in lines):
        raise AssemblyError(f"{path} contains blank JSONL rows")
    values = [json.loads(line) for line in lines]
    if any(not isinstance(item, Mapping) for item in values):
        raise AssemblyError(f"{path} contains a non-object row")
    return [dict(item) for item in values]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for item in values
        ),
        encoding="utf-8",
    )


def _pool_payload(
    image_ids: Sequence[str], *, pool_role: str
) -> dict[str, Any]:
    if pool_role not in {"possible", "certified_impossible"}:
        raise AssemblyError(f"invalid Stage-Zero pool role: {pool_role}")
    ordered = sorted((str(item) for item in image_ids), key=int)
    return {
        "schema_version": SCHEMA_VERSION,
        "terminal_status": "completed",
        "population_scope": "train_only_censored_nonpassing_U",
        "pool_role": pool_role,
        "count": len(ordered),
        "ordered_image_ids": ordered,
        "ordered_image_ids_sha256": sha256_json(ordered),
    }


def _materialize_source_snapshot(repo: Path, output_root: Path) -> dict[str, Any]:
    snapshot_root = output_root / "source-snapshot"
    loaded = census._repo_loaded_source_paths(repo)
    relative_paths = sorted(set(SOURCE_SNAPSHOT_PATHS) | set(loaded))
    entries: list[dict[str, Any]] = []
    for relative in relative_paths:
        source = (repo / relative).resolve(strict=True)
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        digest = sha256_file(source)
        if sha256_file(destination) != digest:
            raise AssemblyError(f"source snapshot hash mismatch: {relative}")
        entries.append(
            {
                "repo_relative_path": relative,
                "source_sha256": digest,
                "snapshot_relative_path": str(destination.relative_to(output_root)),
                "snapshot_sha256": digest,
            }
        )
    manifest = {
        "schema_version": f"{SCHEMA_VERSION}.source_snapshot.v1",
        "scope": "stage_zero_task_relevant_source",
        "entries": entries,
        "source_set_sha256": sha256_json(
            [[item["repo_relative_path"], item["source_sha256"]] for item in entries]
        ),
    }
    manifest_path = snapshot_root / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "manifest_relative_path": str(manifest_path.relative_to(output_root)),
        "manifest_sha256": sha256_file(manifest_path),
        "source_set_sha256": manifest["source_set_sha256"],
        "snapshotted_repo_paths": relative_paths,
        "entries": entries,
    }


def _discard_staging(path: Path) -> None:
    if not path.exists():
        return
    for item in sorted(path.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        try:
            os.chmod(item, 0o755 if item.is_dir() else 0o644)
        except OSError:
            pass
    try:
        os.chmod(path, 0o755)
    except OSError:
        pass
    shutil.rmtree(path)


def _residue_paths(output_dir: Path) -> list[Path]:
    paths = list(output_dir.parent.glob(f".{output_dir.name}.staging-*"))
    paths.extend(output_dir.parent.glob(f".{output_dir.name}.failed-*"))
    paths.extend(output_dir.parent.glob(f".{output_dir.name}.quarantine-*"))
    return sorted(paths, key=str)


def _assert_immutable_tree(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise AssemblyError(f"immutable root is not a real directory: {root}")
    if (root.stat(follow_symlinks=False).st_mode & 0o777) != 0o555:
        raise AssemblyError(f"immutable root mode differs from 0555: {root}")
    for path in sorted(root.rglob("*"), key=str):
        if path.is_symlink():
            raise AssemblyError(f"immutable tree contains a symlink: {path}")
        mode = path.stat(follow_symlinks=False).st_mode & 0o777
        if path.is_dir():
            if mode != 0o555:
                raise AssemblyError(f"immutable directory mode differs: {path}")
        elif path.is_file():
            if mode != 0o444:
                raise AssemblyError(f"immutable file mode differs: {path}")
        else:
            raise AssemblyError(f"immutable tree contains a special file: {path}")


def _validate_failed_output(
    root: Path,
    expected_receipt: Mapping[str, Any],
    *,
    canonical_output: Path | None = None,
) -> None:
    _assert_immutable_tree(root)
    if sorted(str(path.relative_to(root)) for path in root.rglob("*")) != [
        "receipt.json"
    ]:
        raise AssemblyError("failed output inventory is not exactly receipt.json")
    decoded, _ = _read_json_exact_bytes(root / "receipt.json")
    receipt = _mapping(decoded, "failed receipt")
    if dict(receipt) != dict(expected_receipt):
        raise AssemblyError("failed receipt readback differs from memory")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("terminal_status") != "failed"
        or receipt.get("immutable_output_directory")
        != str(root if canonical_output is None else canonical_output)
        or not isinstance(receipt.get("failed_validation_stage"), str)
        or not receipt.get("failed_validation_stage")
        or not isinstance(receipt.get("failure_class"), str)
        or not isinstance(receipt.get("failure_message"), str)
    ):
        raise AssemblyError("failed receipt terminal identity differs")
    analyzer = _mapping(receipt.get("analyzer"), "failed receipt analyzer")
    analyzer_path = Path(str(analyzer.get("path", ""))).resolve(strict=True)
    if sha256_file(analyzer_path) != analyzer.get("sha256"):
        raise AssemblyError("failed receipt analyzer identity differs")


def _quarantine_invalid_canonical_root(root: Path) -> Path | None:
    if not root.exists():
        return None
    quarantine = root.parent / f".{root.name}.quarantine-{uuid.uuid4().hex}"
    try:
        os.replace(root, quarantine)
    except Exception:
        _discard_staging(root)
        if root.exists():
            raise AssemblyError(f"invalid canonical root could not be removed: {root}")
        return None
    try:
        _discard_staging(quarantine)
    except Exception:
        return quarantine
    return None


def _publish_failed_output(
    output_dir: Path,
    *,
    analyzer: Path,
    failed_validation_stage: str,
    failure: BaseException,
) -> None:
    if output_dir.exists():
        raise AssemblyError(f"refusing to overwrite existing output: {output_dir}")
    failed_staging = output_dir.parent / f".{output_dir.name}.failed-{uuid.uuid4().hex}"
    failed_staging.mkdir()
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "terminal_status": "failed",
        "failed_validation_stage": failed_validation_stage,
        "failure_class": type(failure).__name__,
        "failure_message": str(failure),
        "immutable_output_directory": str(output_dir),
        "analyzer": {"path": str(analyzer), "sha256": sha256_file(analyzer)},
    }
    receipt_path = failed_staging / "receipt.json"
    _write_json(receipt_path, receipt)
    try:
        os.chmod(receipt_path, 0o444)
        os.chmod(failed_staging, 0o555)
        _validate_failed_output(
            failed_staging, receipt, canonical_output=output_dir
        )
        if output_dir.exists():
            raise AssemblyError(
                f"refusing to overwrite existing output: {output_dir}"
            )
        os.replace(failed_staging, output_dir)
        try:
            _validate_failed_output(output_dir, receipt)
        except Exception:
            _quarantine_invalid_canonical_root(output_dir)
            raise
    except Exception:
        if failed_staging.exists():
            try:
                _discard_staging(failed_staging)
            except Exception:
                pass
        raise


def _finalize_staging(
    staging: Path,
    output_dir: Path,
    *,
    analyzer: Path,
    validate_completed_root: Callable[[Path], None],
    validate_pre_rename: Callable[[Path], None] | None = None,
    post_replace_hook: Callable[[Path], None] | None = None,
) -> None:
    renamed = False
    try:
        if output_dir.exists():
            raise AssemblyError(f"final output appeared during staging: {output_dir}")
        files = sorted((item for item in staging.rglob("*") if item.is_file()), key=str)
        directories = sorted(
            (item for item in staging.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        )
        for item in files:
            os.chmod(item, 0o444)
        for item in directories:
            os.chmod(item, 0o555)
        os.chmod(staging, 0o555)
        if any((item.stat().st_mode & 0o777) != 0o444 for item in files):
            raise AssemblyError("staged success files are not immutable")
        if any((item.stat().st_mode & 0o777) != 0o555 for item in directories):
            raise AssemblyError("staged success directories are not immutable")
        if (staging.stat().st_mode & 0o777) != 0o555:
            raise AssemblyError("staged success root is not immutable")
        sibling_staging = sorted(
            staging.parent.glob(f".{output_dir.name}.staging-*")
        )
        if sibling_staging != [staging]:
            raise AssemblyError(
                f"sibling Stage-Zero staging residue appeared: {sibling_staging}"
            )
        if validate_pre_rename is not None:
            validate_pre_rename(staging)
        os.replace(staging, output_dir)
        renamed = True
        if post_replace_hook is not None:
            post_replace_hook(output_dir)
        _assert_immutable_tree(output_dir)
        validate_completed_root(output_dir)
    except Exception as exc:
        cleanup_failure: Exception | None = None
        try:
            if renamed:
                _quarantine_invalid_canonical_root(output_dir)
            else:
                _discard_staging(staging)
        except Exception as cleanup_exc:
            cleanup_failure = cleanup_exc
        _publish_failed_output(
            output_dir,
            analyzer=analyzer,
            failed_validation_stage=(
                "post_rename_validate_completed_root"
                if renamed
                else "finalize_staging"
            ),
            failure=cleanup_failure or exc,
        )
        raise


def _validate_exact_file(path: Path, expected_path: Path, expected_hash: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if resolved != expected_path.expanduser().resolve(strict=True):
        raise AssemblyError(f"input path is not frozen: {path}")
    if hashlib.sha256(resolved.read_bytes()).hexdigest() != expected_hash:
        raise AssemblyError(f"input hash differs from frozen identity: {path}")
    return resolved


def _file_identity(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _validate_panel_manifests_exact(
    root: Path, *, mode: str, expected_pool_ids: Iterable[str]
) -> dict[str, Any]:
    """Read, hash, decode, and bind one complete manifest inventory exactly."""

    root = root.expanduser().resolve(strict=True)
    expected_ids = {str(item) for item in expected_pool_ids}
    manifests = sorted(root.glob("worker-*-of-*/manifest.json"), key=str)
    all_manifests = sorted(root.rglob("manifest.json"), key=str)
    if manifests != all_manifests or len(manifests) != 8:
        raise AssemblyError(f"{mode} exact manifest inventory is not eight workers")
    manifest_receipts: list[dict[str, Any]] = []
    batch_receipts: list[dict[str, Any]] = []
    worker_indices: set[int] = set()
    inventory_image_ids: set[str] = set()
    bound_paths: set[Path] = set()
    for manifest_path in manifests:
        payload, manifest_hash = _read_json_exact_bytes(manifest_path)
        manifest = _mapping(payload, f"manifest {manifest_path}")
        worker_index = _int_count(manifest.get("worker_index"), "worker_index")
        if (
            worker_index in worker_indices
            or manifest.get("worker_count") != 8
            or manifest.get("status")
            not in {"completed", "completed_with_source_b16_ineligible"}
        ):
            raise AssemblyError(f"invalid exact {mode} manifest: {manifest_path}")
        worker_indices.add(worker_index)
        batches = manifest.get("batches")
        if not isinstance(batches, list):
            raise AssemblyError(f"{manifest_path} lacks batches")
        manifest_receipts.append(
            {"path": str(manifest_path), "sha256": manifest_hash}
        )
        for raw_batch in batches:
            batch = _mapping(raw_batch, f"{manifest_path}.batch")
            image_ids = batch.get("image_ids")
            if not isinstance(image_ids, list) or len(image_ids) != 16:
                raise AssemblyError(f"{manifest_path} batch lacks 16 image IDs")
            canonical_ids = [str(item) for item in image_ids]
            if inventory_image_ids & set(canonical_ids):
                raise AssemblyError(f"{mode} exact manifest image inventory overlaps")
            inventory_image_ids.update(canonical_ids)
            artifacts = _mapping(batch.get("artifacts"), "batch.artifacts")
            entry = _mapping(artifacts.get(mode), f"batch.artifacts.{mode}")
            relative_path = entry.get("path")
            expected_hash = entry.get("sha256")
            if not isinstance(relative_path, str) or not isinstance(
                expected_hash, str
            ):
                raise AssemblyError(f"{manifest_path} has an invalid artifact entry")
            artifact_path = (manifest_path.parent / relative_path).resolve(strict=True)
            if artifact_path in bound_paths:
                raise AssemblyError(f"{mode} exact manifest duplicates {artifact_path}")
            _, observed_hash = _read_json_exact_bytes(
                artifact_path, expected_sha256=expected_hash
            )
            bound_paths.add(artifact_path)
            batch_receipts.append(
                {
                    "path": str(artifact_path),
                    "sha256": observed_hash,
                    "worker_index": worker_index,
                    "batch_index": batch.get("batch_index"),
                }
            )
    if worker_indices != set(range(8)):
        raise AssemblyError(f"{mode} exact worker indices are not 0 through 7")
    if len(batch_receipts) != 152:
        raise AssemblyError(f"{mode} exact batch inventory is not 152 files")
    if inventory_image_ids != expected_ids:
        raise AssemblyError(f"{mode} exact image inventory differs from the pool")
    pattern = "sampled-batch-*.json" if mode == "sampled" else "source_b16-batch-*.json"
    discovered = {path.resolve() for path in root.rglob(pattern)}
    if discovered != bound_paths:
        raise AssemblyError(f"{mode} exact manifest/file inventory differs")
    ordered_manifest_entries = [
        {
            "relative_path": str(Path(item["path"]).relative_to(root)),
            "sha256": item["sha256"],
        }
        for item in sorted(manifest_receipts, key=lambda item: item["path"])
    ]
    ordered_inventory = sorted(inventory_image_ids, key=int)
    return {
        "root": str(root),
        "manifest_count": len(manifest_receipts),
        "batch_count": len(batch_receipts),
        "image_inventory_count": len(inventory_image_ids),
        "ordered_image_inventory_sha256": sha256_json(ordered_inventory),
        "manifest_set_sha256": sha256_json(ordered_manifest_entries),
        "manifests": manifest_receipts,
        "batches": sorted(batch_receipts, key=lambda item: item["path"]),
    }


def _validate_adapter_artifact_inventory(
    adapter: Mapping[str, Any],
    *,
    sampled_manifests: Mapping[str, Any],
    source_manifests: Mapping[str, Any],
) -> list[dict[str, str]]:
    census_receipt = _mapping(adapter.get("census"), "adapter.census")
    raw_provenance = census_receipt.get("artifact_provenance")
    if not isinstance(raw_provenance, list):
        raise AssemblyError("adapter artifact provenance is missing")
    observed = sorted(
        (
            {
                "mode": str(_mapping(item, "artifact provenance").get("mode", "")),
                "path": str(_mapping(item, "artifact provenance").get("path", "")),
                "sha256": str(
                    _mapping(item, "artifact provenance").get("sha256", "")
                ),
            }
            for item in raw_provenance
        ),
        key=lambda item: (item["mode"], item["path"]),
    )
    expected = sorted(
        [
            {
                "mode": mode,
                "path": str(_mapping(item, "manifest batch")["path"]),
                "sha256": str(_mapping(item, "manifest batch")["sha256"]),
            }
            for mode, receipt in (
                ("sampled", sampled_manifests),
                ("source_b16", source_manifests),
            )
            for item in _mapping(receipt, f"{mode} manifests")["batches"]
        ],
        key=lambda item: (item["mode"], item["path"]),
    )
    if observed != expected:
        raise AssemblyError("adapter provenance differs from manifest-bound inventory")
    return observed


def _observe_fixed_inputs(
    paths: Mapping[str, Path], expected_hashes: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    if set(paths) != set(expected_hashes):
        raise AssemblyError("fixed input path/hash labels differ")
    observed = {label: _file_identity(paths[label]) for label in sorted(paths)}
    for label, identity in observed.items():
        if identity["sha256"] != expected_hashes[label]:
            raise AssemblyError(f"frozen input changed: {label}")
    return observed


def _revalidate_frozen_external_state(
    *,
    fixed_paths: Mapping[str, Path],
    fixed_hashes: Mapping[str, str],
    sampled_root: Path,
    source_root: Path,
    candidate_pool_ids: Sequence[str],
    expected_sampled_manifests: Mapping[str, Any],
    expected_source_manifests: Mapping[str, Any],
    expected_adapter_artifacts: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    fixed = _observe_fixed_inputs(fixed_paths, fixed_hashes)
    sampled = _validate_panel_manifests_exact(
        sampled_root, mode="sampled", expected_pool_ids=candidate_pool_ids
    )
    source = _validate_panel_manifests_exact(
        source_root, mode="source_b16", expected_pool_ids=candidate_pool_ids
    )
    if (
        sampled != dict(expected_sampled_manifests)
        or source != dict(expected_source_manifests)
    ):
        raise AssemblyError("manifest-bound inventory changed after adapter analysis")
    expected_artifacts = [dict(item) for item in expected_adapter_artifacts]
    reobserved_artifacts = sorted(
        [
            {
                "mode": mode,
                "path": str(_mapping(item, "manifest batch")["path"]),
                "sha256": str(_mapping(item, "manifest batch")["sha256"]),
            }
            for mode, receipt in (("sampled", sampled), ("source_b16", source))
            for item in _mapping(receipt, f"{mode} manifests")["batches"]
        ],
        key=lambda item: (item["mode"], item["path"]),
    )
    if reobserved_artifacts != expected_artifacts:
        raise AssemblyError("reobserved batch inventory differs from adapter provenance")
    return {
        "fixed_files": fixed,
        "sampled_manifest_binding": sampled,
        "source_manifest_binding": source,
        "adapter_artifact_inventory": reobserved_artifacts,
    }


def _validate_readback(
    staging: Path,
    category_records: Sequence[Mapping[str, Any]],
    possibility_records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    if _read_jsonl(staging / "category-state.jsonl") != list(category_records):
        raise AssemblyError("category-state readback differs")
    if _read_jsonl(staging / "possibility-census.jsonl") != list(possibility_records):
        raise AssemblyError("possibility-census readback differs")
    read_summary = json.loads((staging / "summary.json").read_text(encoding="utf-8"))
    if read_summary != dict(summary):
        raise AssemblyError("summary readback differs")
    if len(category_records) != EXPECTED_POPULATION_COUNT:
        raise AssemblyError("category-state count differs from U")
    if len(possibility_records) != EXPECTED_POPULATION_COUNT:
        raise AssemblyError("possibility-census count differs from U")
    for category_record, possibility_record in zip(
        category_records, possibility_records, strict=True
    ):
        if category_record["image_id"] != possibility_record["image_id"]:
            raise AssemblyError("category/possibility image order differs")
        certificate = possibility_record.get("impossibility_certificate")
        if not possibility_record["possible"] and (
            not isinstance(certificate, Mapping)
            or not replay_impossibility_certificate(
                category_record["exact_token_representatives"], certificate
            )
        ):
            raise AssemblyError("impossibility certificate replay failed")


def _verify_source_snapshot(
    repo: Path, output_root: Path, snapshot_receipt: Mapping[str, Any]
) -> None:
    entries = snapshot_receipt.get("entries")
    if not isinstance(entries, list):
        raise AssemblyError("source snapshot receipt lacks entries")
    expected_paths = sorted(
        str(item) for item in snapshot_receipt.get("snapshotted_repo_paths", [])
    )
    observed_paths = [
        str(_mapping(item, "source snapshot entry").get("repo_relative_path", ""))
        for item in entries
    ]
    if observed_paths != expected_paths:
        raise AssemblyError("source snapshot entry ordering or scope drifted")
    if not set(SOURCE_SNAPSHOT_PATHS) <= set(expected_paths):
        raise AssemblyError("source snapshot lost a required direct source")
    current_closure = census._repo_loaded_source_paths(repo)
    absent = sorted(set(current_closure) - set(expected_paths))
    if absent:
        raise AssemblyError(
            f"new repo-local execution source is absent from snapshot: {absent[:5]}"
        )
    source_pairs: list[list[str]] = []
    for raw in entries:
        entry = _mapping(raw, "source snapshot entry")
        relative = str(entry.get("repo_relative_path", ""))
        live = (repo / relative).resolve(strict=True)
        snapshot = (
            output_root / str(entry.get("snapshot_relative_path", ""))
        ).resolve(strict=True)
        try:
            snapshot.relative_to(output_root.resolve(strict=True))
        except ValueError as exc:
            raise AssemblyError("source snapshot path escapes its output root") from exc
        live_hash = sha256_file(live)
        snapshot_hash = sha256_file(snapshot)
        if (
            live_hash != entry.get("source_sha256")
            or snapshot_hash != entry.get("snapshot_sha256")
            or live_hash != snapshot_hash
        ):
            raise AssemblyError(f"source or snapshot drift detected: {relative}")
        source_pairs.append([relative, live_hash])
    if sha256_json(source_pairs) != snapshot_receipt.get("source_set_sha256"):
        raise AssemblyError("source snapshot set hash drifted")
    manifest_path = (
        output_root / str(snapshot_receipt.get("manifest_relative_path", ""))
    ).resolve(strict=True)
    manifest_payload, manifest_hash = _read_json_exact_bytes(manifest_path)
    expected_manifest = {
        "schema_version": f"{SCHEMA_VERSION}.source_snapshot.v1",
        "scope": "stage_zero_task_relevant_source",
        "entries": entries,
        "source_set_sha256": snapshot_receipt.get("source_set_sha256"),
    }
    if (
        manifest_hash != snapshot_receipt.get("manifest_sha256")
        or manifest_payload != expected_manifest
    ):
        raise AssemblyError("source snapshot manifest drifted")


def _output_inventory(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*"), key=str):
        if path.is_symlink():
            raise AssemblyError(f"output inventory contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise AssemblyError(f"output inventory contains a special file: {path}")
        relative = str(path.relative_to(root))
        identity = _file_identity(path)
        result[relative] = {
            "relative_path": relative,
            "sha256": identity["sha256"],
            "size_bytes": identity["size_bytes"],
        }
    return result


def _directory_inventory(root: Path) -> list[str]:
    directories: list[str] = []
    for path in sorted(root.rglob("*"), key=str):
        if path.is_symlink():
            raise AssemblyError(f"output inventory contains a symlink: {path}")
        if path.is_dir():
            directories.append(str(path.relative_to(root)))
        elif not path.is_file():
            raise AssemblyError(f"output inventory contains a special file: {path}")
    return directories


def _require_exact_git_identity(
    repo: Path,
    snapshotted_paths: Sequence[str],
    expected_identity: Mapping[str, Any],
    *,
    boundary: str,
) -> dict[str, Any]:
    observed = _stage_zero_git_identity(repo, snapshotted_paths)
    if observed != dict(expected_identity):
        raise AssemblyError(f"{boundary} Git identity differs")
    return observed


def _stage_zero_git_identity(
    repo: Path, snapshotted_paths: Sequence[str]
) -> dict[str, Any]:
    """Extend the predecessor identity with the exact scoped index diff."""

    resolved_repo = repo.resolve(strict=True)
    scope = sorted(str(item) for item in snapshotted_paths)
    identity = dict(census._git_identity(resolved_repo, scope))
    command = [
        "git",
        "diff",
        "--cached",
        "--binary",
        "HEAD",
        "--",
        *scope,
    ]
    completed = subprocess.run(
        [command[0], "-C", str(resolved_repo), *command[1:]],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    identity.update(
        {
            "scoped_cached_diff_command": command,
            "scoped_cached_diff_exit_code": completed.returncode,
            "scoped_cached_diff_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "scoped_cached_diff_size_bytes": len(completed.stdout),
            "scoped_cached_diff_stderr_sha256": hashlib.sha256(
                completed.stderr
            ).hexdigest(),
        }
    )
    return identity


def _validate_success_receipt_readback(
    *,
    receipt_path: Path,
    expected_receipt: Mapping[str, Any],
    output_root: Path,
    repo: Path,
) -> str:
    decoded, receipt_hash = _read_json_exact_bytes(receipt_path)
    receipt = _mapping(decoded, "success receipt")
    if dict(receipt) != dict(expected_receipt):
        raise AssemblyError("serialized success receipt differs from memory")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("terminal_status") != "completed"
        or receipt.get("population_scope")
        != "train_only_censored_nonpassing_U"
    ):
        raise AssemblyError("success receipt terminal schema/status/scope differs")
    outputs = _mapping(receipt.get("outputs"), "receipt outputs")
    observed_inventory = _output_inventory(output_root)
    observed_directories = _directory_inventory(output_root)
    declared_directories = receipt.get("output_directories")
    if (
        not isinstance(declared_directories, list)
        or any(not isinstance(item, str) for item in declared_directories)
        or declared_directories != observed_directories
    ):
        raise AssemblyError("success output directory inventory is not exact")
    if set(observed_inventory) != set(outputs) | {"receipt.json"}:
        raise AssemblyError("success output root inventory is not exact")
    if observed_inventory["receipt.json"]["sha256"] != receipt_hash:
        raise AssemblyError("success receipt hash changed during readback")
    for relative, raw in outputs.items():
        entry = _mapping(raw, f"output {relative}")
        if entry != observed_inventory.get(relative):
            raise AssemblyError(f"success output identity differs: {relative}")

    categories = _read_jsonl(output_root / "category-state.jsonl")
    possibilities = _read_jsonl(output_root / "possibility-census.jsonl")
    summary_decoded, _ = _read_json_exact_bytes(output_root / "summary.json")
    possible_decoded, _ = _read_json_exact_bytes(output_root / "possible-pool.json")
    impossible_decoded, _ = _read_json_exact_bytes(
        output_root / "impossible-pool.json"
    )
    summary = _mapping(summary_decoded, "summary")
    possible_pool = _mapping(possible_decoded, "possible pool")
    impossible_pool = _mapping(impossible_decoded, "impossible pool")
    _validate_readback(output_root, categories, possibilities, summary)
    category_ids = [str(item["image_id"]) for item in categories]
    possibility_ids = [str(item["image_id"]) for item in possibilities]
    possible_ids = [
        str(item["image_id"]) for item in possibilities if bool(item["possible"])
    ]
    impossible_ids = [
        str(item["image_id"]) for item in possibilities if not bool(item["possible"])
    ]
    if (
        category_ids != possibility_ids
        or possible_pool
        != _pool_payload(possible_ids, pool_role="possible")
        or impossible_pool
        != _pool_payload(impossible_ids, pool_role="certified_impossible")
        or summary.get("population_count") != len(category_ids)
        or summary.get("possible_image_ids") != possible_ids
        or summary.get("impossible_image_ids") != impossible_ids
        or summary.get("possible_pool_count") != len(possible_ids)
        or summary.get("impossible_pool_count") != len(impossible_ids)
        or set(possible_ids) & set(impossible_ids)
        or sorted([*possible_ids, *impossible_ids], key=int) != category_ids
    ):
        raise AssemblyError("serialized Stage-Zero population partition differs")
    population = _mapping(receipt.get("population"), "receipt population")
    if population != {
        "count": len(category_ids),
        "ordered_image_ids_sha256": sha256_json(category_ids),
    }:
        raise AssemblyError("success receipt population binding differs")
    category_hash = sha256_json(categories)
    possibility_hash = sha256_json(possibilities)
    summary_hash = sha256_json(summary)
    if (
        summary.get("category_state_sha256") != category_hash
        or summary.get("possibility_census_sha256") != possibility_hash
        or summary.get("ordered_possible_image_ids_sha256")
        != possible_pool.get("ordered_image_ids_sha256")
        or summary.get("ordered_impossible_image_ids_sha256")
        != impossible_pool.get("ordered_image_ids_sha256")
    ):
        raise AssemblyError("serialized Stage-Zero semantic hashes differ")
    row_counts = _mapping(receipt.get("row_counts"), "receipt row counts")
    if row_counts != {
        "category_state": len(categories),
        "possibility_census": len(possibilities),
        "possible_pool": len(possible_ids),
        "impossible_pool": len(impossible_ids),
    }:
        raise AssemblyError("success receipt row counts do not reconcile")
    determinism = _mapping(receipt.get("determinism"), "receipt determinism")
    if determinism != {
        "forward_category_state_sha256": category_hash,
        "reversed_category_state_sha256": category_hash,
        "forward_possibility_census_sha256": possibility_hash,
        "reversed_possibility_census_sha256": possibility_hash,
        "forward_summary_sha256": summary_hash,
        "reversed_summary_sha256": summary_hash,
    }:
        raise AssemblyError("success receipt determinism hashes do not reconcile")
    if (
        receipt.get("possible_pool_count") != len(possible_ids)
        or receipt.get("ordered_possible_image_ids_sha256")
        != possible_pool.get("ordered_image_ids_sha256")
        or receipt.get("possible_pool_artifact_sha256")
        != observed_inventory["possible-pool.json"]["sha256"]
    ):
        raise AssemblyError("success receipt possible-pool binding differs")
    integrity = _mapping(
        receipt.get("artifact_integrity_checks"), "receipt integrity checks"
    )
    if not integrity or any(value is not True for value in integrity.values()):
        raise AssemblyError("success receipt contains an unproved integrity flag")
    snapshot = _mapping(receipt.get("source_snapshot"), "source snapshot")
    _verify_source_snapshot(repo, output_root, snapshot)
    _require_exact_git_identity(
        repo,
        snapshot.get("snapshotted_repo_paths", []),
        _mapping(receipt.get("git_identity"), "receipt Git identity"),
        boundary="success_receipt_readback",
    )
    return receipt_hash


def _materialize(
    *,
    image_census: Path,
    census_summary: Path,
    census_receipt: Path,
    candidate_pool: Path,
    split_receipt: Path,
    sampled_root: Path,
    source_root: Path,
    output_dir: Path,
) -> None:
    analyzer = Path(__file__).resolve()
    repo = analyzer.parents[2]
    final_output = output_dir.expanduser().resolve(strict=False)
    if final_output != FROZEN_OUTPUT_ROOT.resolve(strict=False):
        raise AssemblyError("output root is not the frozen Stage-Zero path")
    if final_output.exists():
        raise AssemblyError(f"immutable output already exists: {final_output}")
    final_output.parent.mkdir(parents=True, exist_ok=True)
    residue = _residue_paths(final_output)
    if residue:
        raise AssemblyError(f"pre-existing Stage-Zero staging residue: {residue[:3]}")
    staging = final_output.parent / f".{final_output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    stage = "validate_frozen_inputs"
    try:
        image_census = _validate_exact_file(
            image_census, FROZEN_IMAGE_CENSUS_PATH, FROZEN_IMAGE_CENSUS_SHA256
        )
        census_summary = _validate_exact_file(
            census_summary, FROZEN_SUMMARY_PATH, FROZEN_SUMMARY_SHA256
        )
        census_receipt = _validate_exact_file(
            census_receipt, FROZEN_RECEIPT_PATH, FROZEN_RECEIPT_SHA256
        )
        candidate_pool = _validate_exact_file(
            candidate_pool,
            FROZEN_CANDIDATE_POOL_PATH,
            FROZEN_CANDIDATE_POOL_SHA256,
        )
        split_receipt = _validate_exact_file(
            split_receipt, FROZEN_SPLIT_RECEIPT_PATH, FROZEN_SPLIT_RECEIPT_SHA256
        )
        split_decoded, _ = _read_json_exact_bytes(
            split_receipt, expected_sha256=FROZEN_SPLIT_RECEIPT_SHA256
        )
        split_payload = _mapping(split_decoded, "split receipt")
        split_outputs = _mapping(split_payload.get("outputs"), "split receipt outputs")
        split_output_paths: dict[str, Path] = {}
        split_output_hashes: dict[str, str] = {}
        for name in ("train_candidate", "development", "heldout"):
            entry = _mapping(split_outputs.get(name), f"split receipt {name}")
            path = Path(str(entry.get("path", ""))).expanduser().resolve(strict=True)
            digest = str(entry.get("sha256", ""))
            if _file_identity(path)["sha256"] != digest:
                raise AssemblyError(f"split receipt {name} hash does not reproduce")
            split_output_paths[name] = path
            split_output_hashes[name] = digest
        train_path = split_output_paths["train_candidate"]
        if (
            train_path != FROZEN_TRAIN_CANDIDATE_PATH.resolve(strict=True)
            or split_output_hashes["train_candidate"]
            != FROZEN_TRAIN_CANDIDATE_SHA256
        ):
            raise AssemblyError("training-candidate split is not frozen")
        sampled_root = sampled_root.expanduser().resolve(strict=True)
        source_root = source_root.expanduser().resolve(strict=True)
        if sampled_root != FROZEN_SAMPLED_ROOT.resolve(strict=True):
            raise AssemblyError("sampled root is not frozen")
        if source_root != FROZEN_SOURCE_ROOT.resolve(strict=True):
            raise AssemblyError("Source root is not frozen")
        summary_decoded, _ = _read_json_exact_bytes(
            census_summary, expected_sha256=FROZEN_SUMMARY_SHA256
        )
        receipt_decoded, _ = _read_json_exact_bytes(
            census_receipt, expected_sha256=FROZEN_RECEIPT_SHA256
        )
        summary_input = _mapping(summary_decoded, "census summary")
        receipt_input = _mapping(receipt_decoded, "census receipt")
        if (
            summary_input.get("terminal_status") != "completed"
            or receipt_input.get("terminal_status") != "completed"
            or summary_input.get("image_count") != EXPECTED_CENSUS_IMAGE_COUNT
        ):
            raise AssemblyError("completed census inputs do not reconcile")
        records = _read_jsonl(
            image_census, expected_sha256=FROZEN_IMAGE_CENSUS_SHA256
        )
        population = derive_population(records)

        fixed_paths = {
            "candidate_pool": candidate_pool,
            "census_receipt": census_receipt,
            "census_summary": census_summary,
            "development": split_output_paths["development"],
            "heldout": split_output_paths["heldout"],
            "image_census": image_census,
            "split_receipt": split_receipt,
            "train_candidate": train_path,
        }
        fixed_hashes = {
            "candidate_pool": FROZEN_CANDIDATE_POOL_SHA256,
            "census_receipt": FROZEN_RECEIPT_SHA256,
            "census_summary": FROZEN_SUMMARY_SHA256,
            "development": split_output_hashes["development"],
            "heldout": split_output_hashes["heldout"],
            "image_census": FROZEN_IMAGE_CENSUS_SHA256,
            "split_receipt": FROZEN_SPLIT_RECEIPT_SHA256,
            "train_candidate": FROZEN_TRAIN_CANDIDATE_SHA256,
        }
        initial_fixed_inputs = _observe_fixed_inputs(fixed_paths, fixed_hashes)

        stage = "validate_train_only_inventory"
        membership = _split_membership(
            candidate_pool=candidate_pool, split_receipt=split_receipt
        )
        train = set(membership["train_candidate"])
        development = set(membership["development"])
        heldout = set(membership["heldout"])
        if not set(population) <= train or set(population) & (development | heldout):
            raise AssemblyError("population U violates the train-only semantic boundary")
        candidate_pool_ids = list(_candidate_pool(candidate_pool))
        predecessor_sampled_manifests = census._validate_panel_manifests(
            sampled_root, mode="sampled", expected_pool_ids=candidate_pool_ids
        )
        predecessor_source_manifests = census._validate_panel_manifests(
            source_root, mode="source_b16", expected_pool_ids=candidate_pool_ids
        )
        sampled_manifests = _validate_panel_manifests_exact(
            sampled_root, mode="sampled", expected_pool_ids=candidate_pool_ids
        )
        source_manifests = _validate_panel_manifests_exact(
            source_root, mode="source_b16", expected_pool_ids=candidate_pool_ids
        )
        if (
            sampled_manifests != predecessor_sampled_manifests
            or source_manifests != predecessor_source_manifests
        ):
            raise AssemblyError("exact-byte and predecessor manifest checks differ")
        if (
            sampled_manifests["manifest_set_sha256"]
            != FROZEN_SAMPLED_MANIFEST_SET_SHA256
            or source_manifests["manifest_set_sha256"]
            != FROZEN_SOURCE_MANIFEST_SET_SHA256
        ):
            raise AssemblyError("panel manifest-set identity differs")

        stage = "load_exact_b16_adapter"
        adapter = load_v2_b16_panel_adapter(
            sampled_panel_root=sampled_root,
            source_b16_root=source_root,
            candidate_pool=candidate_pool,
            semantic_image_ids=population,
            sampled_artifact_inventory=sampled_manifests["batches"],
            source_artifact_inventory=source_manifests["batches"],
            expected_candidate_pool_sha256=FROZEN_CANDIDATE_POOL_SHA256,
        )
        census._validate_frozen_panel_bindings(
            sampled_root=sampled_root,
            source_root=source_root,
            sampled_manifests=sampled_manifests,
            source_manifests=source_manifests,
            adapter=adapter,
        )
        input_identity = census._input_identity_receipt(
            candidate_pool_ids=candidate_pool_ids,
            sampled_manifests=sampled_manifests,
            source_manifests=source_manifests,
            adapter=adapter,
        )
        adapter_census = _mapping(adapter.get("census"), "adapter.census")
        adapter_artifacts = _validate_adapter_artifact_inventory(
            adapter,
            sampled_manifests=sampled_manifests,
            source_manifests=source_manifests,
        )
        if (
            adapter.get("execution_model_identity_sha256")
            != FROZEN_EXECUTION_MODEL_IDENTITY_SHA256
            or adapter.get("tokenizer_identity_sha256")
            != FROZEN_TOKENIZER_IDENTITY_SHA256
            or set(adapter["image_results"]) != set(population)
            or adapter_census.get("sampled_image_count") != EXPECTED_POPULATION_COUNT
            or adapter_census.get("sampled_trajectory_count")
            != EXPECTED_POPULATION_COUNT * 16
            or adapter_census.get("source_image_count") != EXPECTED_POPULATION_COUNT
            or adapter_census.get("source_accepted_image_count")
            != EXPECTED_POPULATION_COUNT
            or adapter_census.get("source_ineligible_image_count") != 0
            or adapter.get("candidate_pool_sha256")
            != FROZEN_CANDIDATE_POOL_SHA256
        ):
            raise AssemblyError("Stage-Zero adapter identity or population differs")

        stage = "analyze_forward_and_reverse"
        category_records, possibility_records, summary = analyze_stage_zero(
            adapter, population, reverse_input=False
        )
        reversed_category, reversed_possibility, reversed_summary = analyze_stage_zero(
            adapter, population, reverse_input=True
        )
        if (
            category_records != reversed_category
            or possibility_records != reversed_possibility
            or summary != reversed_summary
        ):
            raise AssemblyError("forward/reversed Stage-Zero semantics differ")

        stage = "post_revalidate_frozen_inputs"
        external_input_binding = _revalidate_frozen_external_state(
            fixed_paths=fixed_paths,
            fixed_hashes=fixed_hashes,
            sampled_root=sampled_root,
            source_root=source_root,
            candidate_pool_ids=candidate_pool_ids,
            expected_sampled_manifests=sampled_manifests,
            expected_source_manifests=source_manifests,
            expected_adapter_artifacts=adapter_artifacts,
        )
        if external_input_binding["fixed_files"] != initial_fixed_inputs:
            raise AssemblyError("fixed input identities changed during analysis")

        stage = "write_and_replay_outputs"
        _write_jsonl(staging / "category-state.jsonl", category_records)
        _write_jsonl(staging / "possibility-census.jsonl", possibility_records)
        _write_json(staging / "summary.json", summary)
        possible_pool = _pool_payload(
            summary["possible_image_ids"], pool_role="possible"
        )
        impossible_pool = _pool_payload(
            summary["impossible_image_ids"], pool_role="certified_impossible"
        )
        _write_json(staging / "possible-pool.json", possible_pool)
        _write_json(staging / "impossible-pool.json", impossible_pool)
        snapshot = _materialize_source_snapshot(repo, staging)
        _validate_readback(staging, category_records, possibility_records, summary)
        possible_readback, _ = _read_json_exact_bytes(staging / "possible-pool.json")
        impossible_readback, _ = _read_json_exact_bytes(
            staging / "impossible-pool.json"
        )
        if possible_readback != possible_pool or impossible_readback != impossible_pool:
            raise AssemblyError("Stage-Zero pool readback differs")
        _verify_source_snapshot(repo, staging, snapshot)
        outputs = _output_inventory(staging)
        output_directories = _directory_inventory(staging)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "terminal_status": "completed",
            "population_scope": "train_only_censored_nonpassing_U",
            "analyzer": {"path": str(analyzer), "sha256": sha256_file(analyzer)},
            "inputs": {
                "observed_fixed_files": external_input_binding["fixed_files"],
                "sampled_root": str(sampled_root),
                "source_root": str(source_root),
                "sampled_manifest_binding": external_input_binding[
                    "sampled_manifest_binding"
                ],
                "source_manifest_binding": external_input_binding[
                    "source_manifest_binding"
                ],
                "adapter_artifact_inventory": external_input_binding[
                    "adapter_artifact_inventory"
                ],
                "execution_model_identity_sha256": FROZEN_EXECUTION_MODEL_IDENTITY_SHA256,
                "tokenizer_identity_sha256": FROZEN_TOKENIZER_IDENTITY_SHA256,
                "identity_receipt": input_identity,
            },
            "population": {
                "count": len(population),
                "ordered_image_ids_sha256": sha256_json(population),
            },
            "possible_pool_count": possible_pool["count"],
            "ordered_possible_image_ids_sha256": possible_pool[
                "ordered_image_ids_sha256"
            ],
            "possible_pool_artifact_sha256": outputs["possible-pool.json"]["sha256"],
            "row_counts": {
                "category_state": len(category_records),
                "possibility_census": len(possibility_records),
                "possible_pool": possible_pool["count"],
                "impossible_pool": impossible_pool["count"],
            },
            "determinism": {
                "forward_category_state_sha256": sha256_json(category_records),
                "reversed_category_state_sha256": sha256_json(reversed_category),
                "forward_possibility_census_sha256": sha256_json(
                    possibility_records
                ),
                "reversed_possibility_census_sha256": sha256_json(
                    reversed_possibility
                ),
                "forward_summary_sha256": sha256_json(summary),
                "reversed_summary_sha256": sha256_json(reversed_summary),
            },
            "source_snapshot": snapshot,
            "git_identity": _stage_zero_git_identity(
                repo, snapshot["snapshotted_repo_paths"]
            ),
            "outputs": outputs,
            "output_directories": output_directories,
            "artifact_integrity_checks": {
                "hard_pinned_inputs": True,
                "post_revalidated_inputs": True,
                "manifest_bound_adapter_inventory": True,
                "train_only_semantic_filter": True,
                "exact_b16_adapter": True,
                "exact_token_alias_agreement": True,
                "impossibility_certificates_replayed": True,
                "forward_reverse_semantic_determinism": True,
                "serialized_readback": True,
                "serialized_success_receipt_readback": True,
                "exact_output_root_inventory": True,
                "live_source_snapshot_revalidated": True,
                "no_preexisting_staging_residue": True,
            },
            "publication_contract": {
                "method": "same_parent_staging_then_os_replace",
                "completed_file_mode": "0444",
                "completed_directory_mode": "0555",
                "canonical_post_rename_validation_required": True,
                "invalid_canonical_root_quarantine_required": True,
            },
        }
        _write_json(staging / "receipt.json", receipt)
        stage = "validate_success_receipt"
        _validate_success_receipt_readback(
            receipt_path=staging / "receipt.json",
            expected_receipt=receipt,
            output_root=staging,
            repo=repo,
        )
        stage = "final_post_revalidate_frozen_inputs"
        final_external_binding = _revalidate_frozen_external_state(
            fixed_paths=fixed_paths,
            fixed_hashes=fixed_hashes,
            sampled_root=sampled_root,
            source_root=source_root,
            candidate_pool_ids=candidate_pool_ids,
            expected_sampled_manifests=sampled_manifests,
            expected_source_manifests=source_manifests,
            expected_adapter_artifacts=adapter_artifacts,
        )
        if final_external_binding != external_input_binding:
            raise AssemblyError("frozen external inputs changed before publication")
        _verify_source_snapshot(repo, staging, snapshot)
        expected_git_identity = _mapping(
            receipt.get("git_identity"), "receipt Git identity"
        )
        snapshotted_paths = [
            str(item) for item in snapshot.get("snapshotted_repo_paths", [])
        ]

        def validate_last_pre_rename(_: Path) -> None:
            _require_exact_git_identity(
                repo,
                snapshotted_paths,
                expected_git_identity,
                boundary="last_pre_rename",
            )

        def validate_canonical_completed_root(root: Path) -> None:
            if root != final_output:
                raise AssemblyError("completed root is not the canonical frozen path")
            _assert_immutable_tree(root)
            _validate_success_receipt_readback(
                receipt_path=root / "receipt.json",
                expected_receipt=receipt,
                output_root=root,
                repo=repo,
            )
            post_rename_external_binding = _revalidate_frozen_external_state(
                fixed_paths=fixed_paths,
                fixed_hashes=fixed_hashes,
                sampled_root=sampled_root,
                source_root=source_root,
                candidate_pool_ids=candidate_pool_ids,
                expected_sampled_manifests=sampled_manifests,
                expected_source_manifests=source_manifests,
                expected_adapter_artifacts=adapter_artifacts,
            )
            if post_rename_external_binding != external_input_binding:
                raise AssemblyError("frozen external inputs changed after publication")
            _require_exact_git_identity(
                repo,
                snapshotted_paths,
                expected_git_identity,
                boundary="post_rename",
            )

        stage = "finalize_staging"
        _finalize_staging(
            staging,
            final_output,
            analyzer=analyzer,
            validate_completed_root=validate_canonical_completed_root,
            validate_pre_rename=validate_last_pre_rename,
        )
    except Exception as exc:
        if staging.exists():
            _discard_staging(staging)
        if not final_output.exists():
            _publish_failed_output(
                final_output,
                analyzer=analyzer,
                failed_validation_stage=stage,
                failure=exc,
            )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-census", type=Path, default=FROZEN_IMAGE_CENSUS_PATH)
    parser.add_argument("--census-summary", type=Path, default=FROZEN_SUMMARY_PATH)
    parser.add_argument("--census-receipt", type=Path, default=FROZEN_RECEIPT_PATH)
    parser.add_argument("--candidate-pool", type=Path, default=FROZEN_CANDIDATE_POOL_PATH)
    parser.add_argument("--split-receipt", type=Path, default=FROZEN_SPLIT_RECEIPT_PATH)
    parser.add_argument("--sampled-root", type=Path, default=FROZEN_SAMPLED_ROOT)
    parser.add_argument("--source-root", type=Path, default=FROZEN_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=FROZEN_OUTPUT_ROOT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    _materialize(
        image_census=args.image_census,
        census_summary=args.census_summary,
        census_receipt=args.census_receipt,
        candidate_pool=args.candidate_pool,
        split_receipt=args.split_receipt,
        sampled_root=args.sampled_root,
        source_root=args.source_root,
        output_dir=args.output_root,
    )


if __name__ == "__main__":
    main()
