#!/usr/bin/env python3
"""Select the frozen owner-ledger review sample exactly once.

All scientific inputs are validated before an exclusive sibling journal claim.
The journal then acquires one 64-byte entropy block and can only resume that
same block; it never redraws.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
from typing import Any, Literal
import uuid

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "trajectory_owner_set_adjudication_selection.v1"
JOURNAL_CLAIM_SCHEMA_VERSION = f"{SCHEMA_VERSION}.entropy_journal.claim.v2"
JOURNAL_TERMINAL_SCHEMA_VERSION = f"{SCHEMA_VERSION}.entropy_journal.terminal.v2"
SOURCE_MANIFEST_SCHEMA_VERSION = f"{SCHEMA_VERSION}.frozen_sources.v2"
MEMBER_MANIFEST_SCHEMA_VERSION = f"{SCHEMA_VERSION}.possible_pool_members.v2"
STAGE_ZERO_SCHEMA_VERSION = (
    "trajectory_owner_set_adjudication_salvage_gate.stage_zero.v1"
)
STAGE_ZERO_AUDIT_SCHEMA_VERSION = (
    "trajectory_owner_set_adjudication_salvage_gate.stage_zero_audit.v2"
)
STAGE_ZERO_POPULATION_SCOPE = "train_only_censored_nonpassing_U"
POPULATION_SCOPE = "train_only"
ENTROPY_BYTE_COUNT = 64
ENTROPY_SPACE_SIZE = 1 << 512
SAMPLE_SIZE = 32
LOOK_ONE_SIZE = 16
NULL_SUCCESS_COUNT = 248
PER_LOOK_ALPHA = Fraction(1, 40)
SAMPLE_SIZES = (LOOK_ONE_SIZE, SAMPLE_SIZE)
EXPECTED_STAGE_ZERO_POPULATION_COUNT = 1_622
FINAL_UNIT_SHA256 = "885683a947a000a732bc4ed9afce6384bd694068dedb88f49aae0f600005c687"
CONTRACT_REVIEW_SHA256 = (
    "6fba33f33dcbcfd34632cf023f3a1d7f601f0017f2038ca2b7bf972924f18f0d"
)
STAGE_ZERO_PRODUCER_SHA256 = (
    "578df4b97ae7394fe678540ba13af149f52e4c13ce81d9f72128eb74cdc01ae9"
)
UNIT_PATH = REPOSITORY_ROOT / (
    "research/investigations/qwen3-vl-dense-enumeration/experiments/"
    "2026-07-23-trajectory-owner-set-adjudication-salvage-gate/unit.md"
)
CONTRACT_REVIEW_PATH = UNIT_PATH.with_name("contract-independent-review-v1.json")
STAGE_ZERO_PRODUCER_PATH = (
    REPOSITORY_ROOT
    / "scripts/research/analyze_trajectory_owner_set_adjudication_salvage_gate.py"
)
FROZEN_STAGE_ZERO_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-23-trajectory-owner-set-adjudication-salvage-gate/stage-zero-v1"
)
FROZEN_OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-23-trajectory-owner-set-adjudication-salvage-gate/selection-v1"
)

# Hashes are caller-supplied and revalidated.  In particular, the in-progress
# review assembler hash is not pinned in selector source.
PRODUCTION_NAMED_SOURCE_LABELS = frozenset(
    {
        "reviewer_instruction_packet",
        "ontology_state_artifact",
        "review_adjudication_replay_implementation",
        "outcome_classifier_contract",
        "review_implementation_approval_receipt",
        "census_analyzer",
        "global_matcher",
        "panel_adapter",
        "detection_categories",
        "fingerprint",
        "inference_backend",
        "geometry_semantics",
    }
)
AUTO_SOURCE_LABELS = frozenset(
    {"selector", "unit_contract", "contract_review", "stage_zero_producer"}
)
SEMANTIC_STAGE_OUTPUTS = frozenset(
    {
        "category-state.jsonl",
        "possibility-census.jsonl",
        "possible-pool.json",
        "impossible-pool.json",
        "summary.json",
    }
)
BASE_SELECTION_OUTPUTS = frozenset(
    {
        "selection.json",
        "frozen-source-manifest.json",
        "possible-pool-member-manifest.json",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LABEL = re.compile(r"^[a-z][a-z0-9_]*$")


class SelectionError(RuntimeError):
    """A scientific contract, artifact, or semantic binding is invalid."""


class MechanicalSelectionError(RuntimeError):
    """The same complete journal may be resumed after a mechanical failure."""


@dataclass(frozen=True)
class NamedSource:
    label: str
    path: Path
    expected_sha256: str


@dataclass(frozen=True)
class StageZeroBinding:
    root: Path
    receipt: Mapping[str, Any]
    receipt_sha256: str
    inventory: Mapping[str, Mapping[str, Any]]
    inventory_sha256: str
    possible_image_ids: tuple[str, ...]
    impossible_image_ids: tuple[str, ...]
    ordered_pool_sha256: str
    audit_path: Path
    audit: Mapping[str, Any]
    audit_sha256: str

    def to_record(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "receipt_sha256": self.receipt_sha256,
            "root_inventory": copy.deepcopy(dict(self.inventory)),
            "root_inventory_sha256": self.inventory_sha256,
            "possible_pool_count": len(self.possible_image_ids),
            "impossible_pool_count": len(self.impossible_image_ids),
            "ordered_possible_image_ids_sha256": self.ordered_pool_sha256,
            "audit_path": str(self.audit_path),
            "audit_sha256": self.audit_sha256,
            "audit_receipt": copy.deepcopy(dict(self.audit)),
        }


@dataclass(frozen=True)
class JournalState:
    root: Path
    claim: Mapping[str, Any] | None
    claim_file_sha256: str | None
    entropy: bytes | None
    resumed: bool
    terminal: Mapping[str, Any] | None = None
    persistence_unknown_reason: str | None = None
    semantic_void_reason: str | None = None


@dataclass(frozen=True)
class RandomizationResult:
    permutation_count_M: int
    entropy_integer_R: int
    acceptance_limit_L: int
    accepted: bool
    accepted_initial_rank: int | None
    ordered_sample: tuple[str, ...]
    unranking_trace: tuple[Mapping[str, object], ...]


EntropyFactory = Callable[[int], bytes]
FaultInjector = Callable[[str], None]
MemberManifestProvider = Callable[[], Mapping[str, Any]]
TerminalStatus = Literal[
    "completed", "entropy_rejected", "entropy_persistence_unknown", "void"
]


def _noop_fault(_: str) -> None:
    return


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(_canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelectionError(f"{label} must be a JSON object")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SelectionError(f"{label} is not a canonical SHA-256 digest")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = {str(key) for key in value}
    if observed != expected:
        raise SelectionError(
            f"{label} key inventory differs; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _read_json(path: Path, label: str) -> tuple[Mapping[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        decoded = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectionError(f"cannot decode {label}: {path}: {exc}") from exc
    return _mapping(decoded, label), raw


def _read_jsonl(path: Path, label: str) -> tuple[list[dict[str, Any]], bytes]:
    try:
        raw = path.read_bytes()
        lines = raw.decode("utf-8").splitlines()
        if any(not line.strip() for line in lines):
            raise SelectionError(f"{label} contains blank JSONL rows")
        decoded = [json.loads(line) for line in lines]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectionError(f"cannot decode {label}: {path}: {exc}") from exc
    if any(not isinstance(item, Mapping) for item in decoded):
        raise SelectionError(f"{label} contains a non-object row")
    return [dict(item) for item in decoded], raw


def _canonical_image_id(value: object) -> str:
    if isinstance(value, bool):
        raise SelectionError("boolean is not an image identifier")
    if isinstance(value, int):
        if value < 0:
            raise SelectionError("negative image identifier")
        return str(value)
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.isdecimal()
        or (len(value) > 1 and value.startswith("0"))
    ):
        raise SelectionError(f"non-canonical image identifier: {value!r}")
    return value


def canonicalize_image_ids(values: Sequence[object]) -> tuple[str, ...]:
    result = tuple(_canonical_image_id(value) for value in values)
    if len(result) != len(set(result)):
        raise SelectionError("image identifier inventory contains duplicates")
    return result


def falling_factorial(population_size: int, sample_size: int) -> int:
    """Return the exact ordered without-replacement count P(N, k)."""

    if population_size < 0 or sample_size < 0 or sample_size > population_size:
        raise ValueError("falling-factorial dimensions are invalid")
    return math.prod(range(population_size - sample_size + 1, population_size + 1))


def rejection_sampling_parameters(
    *, entropy_space_size: int, permutation_count: int
) -> tuple[int, int]:
    if entropy_space_size <= 0 or permutation_count <= 0:
        raise ValueError("entropy space and permutation count must be positive")
    quotient = entropy_space_size // permutation_count
    return quotient, quotient * permutation_count


def unrank_ordered_sample(
    ordered_pool: Sequence[object], sample_size: int, rank: int
) -> tuple[tuple[str, ...], tuple[Mapping[str, object], ...]]:
    """Unrank one lexicographic ordered sample and emit its exact trace."""

    pool = canonicalize_image_ids(ordered_pool)
    if list(pool) != sorted(pool, key=int):
        raise SelectionError("ordered pool is not numeric-sorted")
    count = falling_factorial(len(pool), sample_size)
    if rank < 0 or rank >= count:
        raise ValueError("rank lies outside P(N, k)")
    remaining = list(pool)
    current_rank = rank
    selected: list[str] = []
    trace: list[Mapping[str, object]] = []
    for position in range(sample_size):
        remaining_positions = sample_size - position - 1
        suffix_count = falling_factorial(len(remaining) - 1, remaining_positions)
        choice_index, next_rank = divmod(current_rank, suffix_count)
        if choice_index >= len(remaining):
            raise SelectionError("unranking choice escaped remaining pool")
        chosen = remaining.pop(choice_index)
        trace.append(
            {
                "position": position,
                "remaining_item_count_before": len(remaining) + 1,
                "remaining_positions_after": remaining_positions,
                "suffix_count": suffix_count,
                "rank_before": current_rank,
                "choice_index": choice_index,
                "selected_image_id": chosen,
                "rank_after": next_rank,
            }
        )
        selected.append(chosen)
        current_rank = next_rank
    if current_rank != 0:
        raise SelectionError("unranking did not consume rank")
    return tuple(selected), tuple(trace)


def rank_ordered_sample(
    ordered_pool: Sequence[object], ordered_sample: Sequence[object]
) -> int:
    pool = canonicalize_image_ids(ordered_pool)
    sample = canonicalize_image_ids(ordered_sample)
    if list(pool) != sorted(pool, key=int) or not set(sample) <= set(pool):
        raise SelectionError("sample is not a canonical pool subset")
    remaining = list(pool)
    rank = 0
    for position, image_id in enumerate(sample):
        choice_index = remaining.index(image_id)
        suffix_count = falling_factorial(len(remaining) - 1, len(sample) - position - 1)
        rank += choice_index * suffix_count
        remaining.pop(choice_index)
    return rank


def randomization_from_entropy(
    ordered_pool: Sequence[object], entropy: bytes
) -> RandomizationResult:
    """Apply the frozen 64-byte rejection and ordered-unranking protocol."""

    pool = canonicalize_image_ids(ordered_pool)
    if list(pool) != sorted(pool, key=int):
        raise SelectionError("ordered pool is not numeric-sorted")
    if len(pool) < SAMPLE_SIZE:
        raise SelectionError("possible pool contains fewer than 32 images")
    if not isinstance(entropy, bytes) or len(entropy) != ENTROPY_BYTE_COUNT:
        raise SelectionError("entropy block must contain exactly 64 bytes")
    permutation_count = falling_factorial(len(pool), SAMPLE_SIZE)
    entropy_integer = int.from_bytes(entropy, "big", signed=False)
    _, acceptance_limit = rejection_sampling_parameters(
        entropy_space_size=ENTROPY_SPACE_SIZE,
        permutation_count=permutation_count,
    )
    if entropy_integer >= acceptance_limit:
        return RandomizationResult(
            permutation_count,
            entropy_integer,
            acceptance_limit,
            False,
            None,
            (),
            (),
        )
    initial_rank = entropy_integer % permutation_count
    sample, trace = unrank_ordered_sample(pool, SAMPLE_SIZE, initial_rank)
    if rank_ordered_sample(pool, sample) != initial_rank:
        raise SelectionError("ordered-sample inverse rank differs")
    replayed, replay_trace = unrank_ordered_sample(pool, SAMPLE_SIZE, initial_rank)
    if replayed != sample or replay_trace != trace:
        raise SelectionError("ordered-sample deterministic replay differs")
    return RandomizationResult(
        permutation_count,
        entropy_integer,
        acceptance_limit,
        True,
        initial_rank,
        sample,
        trace,
    )


def hypergeometric_lower_tail(
    *, population_size: int, success_count: int, sample_size: int, cutoff: int
) -> Fraction:
    if population_size < 0 or not 0 <= success_count <= population_size:
        raise ValueError("invalid hypergeometric population")
    if not 0 <= sample_size <= population_size:
        raise ValueError("invalid hypergeometric sample size")
    minimum = max(0, sample_size - (population_size - success_count))
    maximum = min(sample_size, success_count, cutoff)
    if maximum < minimum:
        return Fraction(0, 1)
    numerator = sum(
        math.comb(success_count, observed)
        * math.comb(population_size - success_count, sample_size - observed)
        for observed in range(minimum, maximum + 1)
    )
    return Fraction(numerator, math.comb(population_size, sample_size))


def hypergeometric_cutoff(
    *,
    population_size: int,
    success_count: int,
    sample_size: int,
    alpha: Fraction,
) -> tuple[int, Fraction]:
    if not Fraction(0, 1) <= alpha <= Fraction(1, 1):
        raise ValueError("alpha lies outside [0, 1]")
    chosen = -1
    probability = Fraction(0, 1)
    for cutoff in range(-1, sample_size + 1):
        current = hypergeometric_lower_tail(
            population_size=population_size,
            success_count=success_count,
            sample_size=sample_size,
            cutoff=cutoff,
        )
        if current > alpha:
            break
        chosen, probability = cutoff, current
    return chosen, probability


def _fraction_record(value: Fraction) -> dict[str, object]:
    with localcontext() as context:
        context.prec = 80
        decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": format(decimal_value, "f"),
    }


def hypergeometric_design(population_size: int) -> list[dict[str, object]]:
    if population_size < NULL_SUCCESS_COUNT:
        raise SelectionError(
            f"Stage-Zero possible pool N={population_size} is below K=248; "
            "stop without claiming entropy"
        )
    result: list[dict[str, object]] = []
    for sample_size in SAMPLE_SIZES:
        cutoff, probability = hypergeometric_cutoff(
            population_size=population_size,
            success_count=NULL_SUCCESS_COUNT,
            sample_size=sample_size,
            alpha=PER_LOOK_ALPHA,
        )
        result.append(
            {
                "cumulative_sample_size": sample_size,
                "largest_rejection_success_count": cutoff,
                "boundary_lower_tail_probability": _fraction_record(probability),
            }
        )
    return result


def _root_inventory(root: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*"), key=str):
        if path.is_symlink():
            raise SelectionError(f"root contains a symlink: {path}")
        mode = path.stat(follow_symlinks=False).st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise SelectionError(f"root contains a special entry: {path}")
        relative = str(path.relative_to(root))
        raw = path.read_bytes()
        result[relative] = {
            "relative_path": relative,
            "sha256": sha256_bytes(raw),
            "size_bytes": len(raw),
        }
    return result


def _require_immutable_tree(root: Path, label: str) -> Path:
    if root.is_symlink():
        raise SelectionError(f"{label} may not be a symlink")
    try:
        resolved = root.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise SelectionError(f"missing {label}: {root}") from exc
    if not resolved.is_dir():
        raise SelectionError(f"{label} is not a directory")
    for path in [resolved, *sorted(resolved.rglob("*"), key=str)]:
        if path.is_symlink():
            raise SelectionError(f"{label} contains a symlink: {path}")
        mode = path.stat(follow_symlinks=False).st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)) or mode & 0o222:
            raise SelectionError(f"{label} is not an immutable regular tree: {path}")
    return resolved


def _stage_zero_module() -> Any:
    if sha256_file(STAGE_ZERO_PRODUCER_PATH) != STAGE_ZERO_PRODUCER_SHA256:
        raise SelectionError("frozen Stage-Zero producer hash drift")
    import scripts.research.analyze_trajectory_owner_set_adjudication_salvage_gate as stage_zero

    if Path(stage_zero.__file__).resolve() != STAGE_ZERO_PRODUCER_PATH.resolve():
        raise SelectionError("loaded Stage-Zero producer path drift")
    return stage_zero


def _validate_stage_zero_external_inputs(receipt: Mapping[str, Any]) -> None:
    stage_zero = _stage_zero_module()
    inputs = _mapping(receipt.get("inputs"), "Stage-Zero inputs")
    fixed = _mapping(inputs.get("observed_fixed_files"), "fixed files")
    fixed_paths: dict[str, Path] = {}
    fixed_hashes: dict[str, str] = {}
    for label, raw in fixed.items():
        identity = _mapping(raw, f"fixed input {label}")
        fixed_paths[str(label)] = Path(str(identity.get("path", "")))
        fixed_hashes[str(label)] = _require_sha256(
            identity.get("sha256"), f"fixed input {label} sha256"
        )
    candidate = fixed_paths.get("candidate_pool")
    if candidate is None:
        raise SelectionError("Stage-Zero inputs omit candidate_pool")
    candidate_ids = list(stage_zero._candidate_pool(candidate))
    sampled = _mapping(inputs.get("sampled_manifest_binding"), "sampled binding")
    source = _mapping(inputs.get("source_manifest_binding"), "source binding")
    adapter = inputs.get("adapter_artifact_inventory")
    if not isinstance(adapter, list):
        raise SelectionError("Stage-Zero adapter inventory is invalid")
    observed = stage_zero._revalidate_frozen_external_state(
        fixed_paths=fixed_paths,
        fixed_hashes=fixed_hashes,
        sampled_root=Path(str(inputs.get("sampled_root", ""))),
        source_root=Path(str(inputs.get("source_root", ""))),
        candidate_pool_ids=candidate_ids,
        expected_sampled_manifests=sampled,
        expected_source_manifests=source,
        expected_adapter_artifacts=adapter,
    )
    expected = {
        "fixed_files": dict(fixed),
        "sampled_manifest_binding": dict(sampled),
        "source_manifest_binding": dict(source),
        "adapter_artifact_inventory": adapter,
    }
    if observed != expected:
        raise SelectionError("Stage-Zero external-state replay differs")


def validate_stage_zero_root(
    stage_zero_root: Path,
    audit_path: Path,
    *,
    synthetic_test_only: bool = False,
) -> StageZeroBinding:
    """Validate the complete producer root and full replay audit."""

    producer = _stage_zero_module()
    root = _require_immutable_tree(stage_zero_root, "Stage-Zero root")
    receipt_path = root / "receipt.json"
    receipt, receipt_raw = _read_json(receipt_path, "Stage-Zero receipt")
    if (
        receipt.get("schema_version") != STAGE_ZERO_SCHEMA_VERSION
        or receipt.get("terminal_status") != "completed"
        or receipt.get("population_scope") != STAGE_ZERO_POPULATION_SCOPE
    ):
        raise SelectionError("Stage-Zero terminal schema/status/scope differs")
    analyzer = _mapping(receipt.get("analyzer"), "Stage-Zero analyzer")
    if (
        Path(str(analyzer.get("path", ""))).resolve()
        != STAGE_ZERO_PRODUCER_PATH.resolve()
        or analyzer.get("sha256") != STAGE_ZERO_PRODUCER_SHA256
    ):
        raise SelectionError("Stage-Zero producer identity differs")
    try:
        producer._validate_success_receipt_readback(
            receipt_path=receipt_path,
            expected_receipt=receipt,
            output_root=root,
            repo=REPOSITORY_ROOT,
        )
    except Exception as exc:
        raise SelectionError(f"complete Stage-Zero replay failed: {exc}") from exc
    inventory = _root_inventory(root)
    outputs = _mapping(receipt.get("outputs"), "Stage-Zero outputs")
    if set(inventory) != set(outputs) | {"receipt.json"}:
        raise SelectionError("Stage-Zero exact root inventory differs")
    if not SEMANTIC_STAGE_OUTPUTS <= set(outputs):
        raise SelectionError("Stage-Zero semantic outputs are incomplete")
    for relative, raw in outputs.items():
        if _mapping(raw, f"Stage-Zero output {relative}") != inventory.get(relative):
            raise SelectionError(f"Stage-Zero output identity differs: {relative}")
    categories, _ = _read_jsonl(root / "category-state.jsonl", "category state")
    possibilities, _ = _read_jsonl(
        root / "possibility-census.jsonl", "possibility census"
    )
    category_ids = canonicalize_image_ids([row.get("image_id") for row in categories])
    possibility_ids = canonicalize_image_ids(
        [row.get("image_id") for row in possibilities]
    )
    if category_ids != possibility_ids or list(category_ids) != sorted(
        category_ids, key=int
    ):
        raise SelectionError("Stage-Zero category/possibility order differs")
    if (
        not synthetic_test_only
        and len(category_ids) != EXPECTED_STAGE_ZERO_POPULATION_COUNT
    ):
        raise SelectionError("Stage-Zero population is not 1,622")
    for category, possibility in zip(categories, possibilities, strict=True):
        representatives = category.get("exact_token_representatives")
        if not isinstance(representatives, list):
            raise SelectionError("category record lacks exact-token representatives")
        expected = {
            "schema_version": STAGE_ZERO_SCHEMA_VERSION,
            "image_id": str(category["image_id"]),
            **producer.search_category_possibility(representatives),
        }
        if possibility != expected:
            raise SelectionError(
                f"Stage-Zero witness/certificate replay differs: {category['image_id']}"
            )
    possible_ids = tuple(
        str(row["image_id"]) for row in possibilities if bool(row.get("possible"))
    )
    impossible_ids = tuple(
        str(row["image_id"]) for row in possibilities if not bool(row.get("possible"))
    )
    possible_pool, _ = _read_json(root / "possible-pool.json", "possible pool")
    impossible_pool, _ = _read_json(root / "impossible-pool.json", "impossible pool")
    if possible_pool != producer._pool_payload(
        possible_ids, pool_role="possible"
    ) or impossible_pool != producer._pool_payload(
        impossible_ids, pool_role="certified_impossible"
    ):
        raise SelectionError("Stage-Zero pool schemas differ from producer")
    if not synthetic_test_only:
        _validate_stage_zero_external_inputs(receipt)
    raw_audit = audit_path.expanduser()
    if raw_audit.is_symlink():
        raise SelectionError("Stage-Zero audit may not be a symlink")
    audit = raw_audit.resolve(strict=True)
    try:
        audit.relative_to(root)
    except ValueError:
        pass
    else:
        raise SelectionError("independent Stage-Zero audit is inside producer root")
    if audit.stat().st_mode & 0o222:
        raise SelectionError("independent Stage-Zero audit is writable")
    audit_receipt, audit_raw = _read_json(audit, "independent Stage-Zero audit")
    required_audit = {
        "schema_version",
        "terminal_status",
        "population_scope",
        "review_scope",
        "verdict",
        "stage_zero_receipt_sha256",
        "stage_zero_root_inventory_sha256",
        "population_count",
        "possible_pool_count",
        "impossible_pool_count",
        "ordered_possible_image_ids_sha256",
        "ordered_impossible_image_ids_sha256",
        "category_state_sha256",
        "possibility_census_sha256",
        "replayed_possible_witness_count",
        "replayed_impossibility_certificate_count",
        "full_certificate_replay",
    }
    if not required_audit <= set(audit_receipt):
        raise SelectionError("independent Stage-Zero audit is incomplete")
    summary, _ = _read_json(root / "summary.json", "Stage-Zero summary")
    expected_values = {
        "schema_version": STAGE_ZERO_AUDIT_SCHEMA_VERSION,
        "terminal_status": "completed",
        "population_scope": STAGE_ZERO_POPULATION_SCOPE,
        "review_scope": "full_witness_and_certificate_replay",
        "verdict": "approved",
        "stage_zero_receipt_sha256": sha256_bytes(receipt_raw),
        "stage_zero_root_inventory_sha256": sha256_json(inventory),
        "population_count": len(category_ids),
        "possible_pool_count": len(possible_ids),
        "impossible_pool_count": len(impossible_ids),
        "ordered_possible_image_ids_sha256": sha256_json(list(possible_ids)),
        "ordered_impossible_image_ids_sha256": sha256_json(list(impossible_ids)),
        "category_state_sha256": summary.get("category_state_sha256"),
        "possibility_census_sha256": summary.get("possibility_census_sha256"),
        "replayed_possible_witness_count": len(possible_ids),
        "replayed_impossibility_certificate_count": len(impossible_ids),
        "full_certificate_replay": True,
    }
    for field, expected in expected_values.items():
        if audit_receipt.get(field) != expected:
            raise SelectionError(f"independent Stage-Zero audit mismatch: {field}")
    return StageZeroBinding(
        root,
        receipt,
        sha256_bytes(receipt_raw),
        inventory,
        sha256_json(inventory),
        possible_ids,
        impossible_ids,
        sha256_json(list(possible_ids)),
        audit,
        audit_receipt,
        sha256_bytes(audit_raw),
    )


def _source_entry(label: str, path: Path, expected_sha256: str) -> dict[str, object]:
    if not _LABEL.fullmatch(label):
        raise SelectionError(f"invalid source label: {label!r}")
    _require_sha256(expected_sha256, f"source {label}")
    raw_path = path.expanduser()
    if raw_path.is_symlink():
        raise SelectionError(f"source {label} may not be a symlink")
    canonical = raw_path.resolve(strict=True)
    if not canonical.is_file():
        raise SelectionError(f"source {label} is not a regular file")
    payload = canonical.read_bytes()
    observed = sha256_bytes(payload)
    if observed != expected_sha256:
        raise SelectionError(f"source {label} hash differs")
    return {
        "label": label,
        "path": str(canonical),
        "bytes": len(payload),
        "sha256": observed,
        "snapshot_relative_path": f"source-snapshots/{label}",
    }


def build_frozen_source_manifest(
    named_sources: Sequence[NamedSource],
    *,
    expected_named_labels: frozenset[str] = PRODUCTION_NAMED_SOURCE_LABELS,
) -> dict[str, object]:
    """Bind the complete explicit and transitive review source set."""

    labels = [item.label for item in named_sources]
    if len(labels) != len(set(labels)):
        raise SelectionError("named review sources duplicate a label")
    if set(labels) != set(expected_named_labels):
        missing = sorted(set(expected_named_labels) - set(labels))
        extra = sorted(set(labels) - set(expected_named_labels))
        raise SelectionError(
            f"named review source labels differ: missing={missing}, extra={extra}"
        )
    auto_sources = (
        NamedSource("selector", Path(__file__), sha256_file(Path(__file__))),
        NamedSource("unit_contract", UNIT_PATH, FINAL_UNIT_SHA256),
        NamedSource("contract_review", CONTRACT_REVIEW_PATH, CONTRACT_REVIEW_SHA256),
        NamedSource(
            "stage_zero_producer", STAGE_ZERO_PRODUCER_PATH, STAGE_ZERO_PRODUCER_SHA256
        ),
    )
    all_sources = (*auto_sources, *named_sources)
    all_labels = [item.label for item in all_sources]
    if set(all_labels) != set(AUTO_SOURCE_LABELS) | set(expected_named_labels):
        raise SelectionError(
            "frozen source manifest is not the exact required label set"
        )
    entries = sorted(
        (
            _source_entry(item.label, item.path, item.expected_sha256)
            for item in all_sources
        ),
        key=lambda item: str(item["label"]),
    )
    paths = [str(item["path"]) for item in entries]
    if len(paths) != len(set(paths)):
        raise SelectionError("frozen source manifest duplicates a canonical path")
    return {
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "expected_named_labels": sorted(expected_named_labels),
        "ordered_labels": [str(item["label"]) for item in entries],
        "entries": entries,
        "entries_sha256": sha256_json(entries),
    }


def revalidate_frozen_source_manifest(manifest: Mapping[str, Any]) -> None:
    _exact_keys(
        manifest,
        {
            "schema_version",
            "expected_named_labels",
            "ordered_labels",
            "entries",
            "entries_sha256",
        },
        "frozen source manifest",
    )
    if manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise SelectionError("frozen source manifest schema differs")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise SelectionError("frozen source manifest entries are missing")
    expected_keys = {
        "label",
        "path",
        "bytes",
        "sha256",
        "snapshot_relative_path",
    }
    labels: list[str] = []
    paths: list[str] = []
    for raw in entries:
        entry = _mapping(raw, "frozen source entry")
        _exact_keys(entry, expected_keys, "frozen source entry")
        label = str(entry["label"])
        labels.append(label)
        path = Path(str(entry["path"]))
        if path.is_symlink() or path.resolve(strict=True) != path:
            raise SelectionError(f"source {label} is no longer canonical")
        payload = path.read_bytes()
        if len(payload) != entry["bytes"] or sha256_bytes(payload) != entry["sha256"]:
            raise SelectionError(f"source {label} drifted")
        if entry["snapshot_relative_path"] != f"source-snapshots/{label}":
            raise SelectionError(f"source {label} snapshot path differs")
        paths.append(str(path))
    if labels != sorted(labels) or labels != manifest.get("ordered_labels"):
        raise SelectionError("frozen source manifest labels are not canonical")
    expected_named = manifest.get("expected_named_labels")
    if not isinstance(expected_named, list) or expected_named != sorted(expected_named):
        raise SelectionError("frozen source expected labels are not canonical")
    if set(labels) != set(AUTO_SOURCE_LABELS) | set(expected_named):
        raise SelectionError("frozen source manifest label coverage differs")
    if len(labels) != len(set(labels)) or len(paths) != len(set(paths)):
        raise SelectionError("frozen source manifest has duplicate labels or paths")
    if manifest.get("entries_sha256") != sha256_json(entries):
        raise SelectionError("frozen source manifest hash differs")


_MEMBER_KEYS = {
    "image_id",
    "canonical_source_path",
    "source_image_bytes",
    "source_image_sha256",
    "source_image_width",
    "source_image_height",
    "official_owner_record_sha256",
    "route_replay_input_record_sha256",
}


def validate_member_manifest(
    manifest: Mapping[str, Any], possible_image_ids: Sequence[str]
) -> dict[str, Any]:
    """Validate every possible member, hashing bytes before image decoding."""

    _exact_keys(
        manifest,
        {
            "schema_version",
            "population_scope",
            "count",
            "ordered_image_ids",
            "ordered_image_ids_sha256",
            "members",
            "members_sha256",
            "input_bindings",
            "input_bindings_sha256",
        },
        "possible-pool member manifest",
    )
    expected_ids = list(canonicalize_image_ids(possible_image_ids))
    members = manifest.get("members")
    if (
        manifest.get("schema_version") != MEMBER_MANIFEST_SCHEMA_VERSION
        or manifest.get("population_scope") != STAGE_ZERO_POPULATION_SCOPE
        or manifest.get("count") != len(expected_ids)
        or manifest.get("ordered_image_ids") != expected_ids
        or manifest.get("ordered_image_ids_sha256") != sha256_json(expected_ids)
        or not isinstance(members, list)
        or len(members) != len(expected_ids)
    ):
        raise SelectionError("possible-pool member manifest header differs")
    normalized: list[dict[str, Any]] = []
    for expected_id, raw in zip(expected_ids, members, strict=True):
        member = dict(_mapping(raw, f"member {expected_id}"))
        _exact_keys(member, _MEMBER_KEYS, f"member {expected_id}")
        image_id = _canonical_image_id(member.get("image_id"))
        if image_id != expected_id:
            raise SelectionError("member manifest image order differs")
        for field in (
            "source_image_sha256",
            "official_owner_record_sha256",
            "route_replay_input_record_sha256",
        ):
            _require_sha256(member.get(field), f"member {image_id}.{field}")
        path = Path(str(member["canonical_source_path"]))
        if path.is_symlink() or path.resolve(strict=True) != path or not path.is_file():
            raise SelectionError(f"member {image_id} source path is not canonical")
        payload = path.read_bytes()
        if len(payload) != member["source_image_bytes"]:
            raise SelectionError(f"member {image_id} source byte count differs")
        # Deliberately hash the complete file before handing any bytes to Pillow.
        if sha256_bytes(payload) != member["source_image_sha256"]:
            raise SelectionError(f"member {image_id} source hash differs")
        try:
            with Image.open(BytesIO(payload)) as image:
                width, height = image.size
                image.verify()
        except Exception as exc:
            raise SelectionError(
                f"member {image_id} source image is undecodable"
            ) from exc
        if [width, height] != [
            member["source_image_width"],
            member["source_image_height"],
        ]:
            raise SelectionError(f"member {image_id} source dimensions differ")
        normalized.append(member)
    if manifest.get("members_sha256") != sha256_json(normalized):
        raise SelectionError("possible-pool member manifest hash differs")
    bindings = _mapping(manifest.get("input_bindings"), "member input bindings")
    if manifest.get("input_bindings_sha256") != sha256_json(bindings):
        raise SelectionError("member input binding hash differs")
    return copy.deepcopy(dict(manifest))


def build_synthetic_member_manifest(
    possible_image_ids: Sequence[str],
    *,
    image_path: Path,
    official_owner_records: Mapping[str, object] | None = None,
    route_replay_records: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Test-only factory; production always reconstructs from the strict adapter."""

    ids = list(canonicalize_image_ids(possible_image_ids))
    path = image_path.expanduser().resolve(strict=True)
    payload = path.read_bytes()
    image_hash = sha256_bytes(payload)
    with Image.open(BytesIO(payload)) as image:
        width, height = image.size
        image.verify()
    owner_records = official_owner_records or {item: {"image_id": item} for item in ids}
    replay_records = route_replay_records or {
        item: {
            "image_id": item,
            "route_ids": [
                "source-b16",
                *(f"sample-{index:02d}" for index in range(16)),
            ],
        }
        for item in ids
    }
    if set(owner_records) != set(ids) or set(replay_records) != set(ids):
        raise SelectionError("synthetic member records do not cover the exact pool")
    members = [
        {
            "image_id": image_id,
            "canonical_source_path": str(path),
            "source_image_bytes": len(payload),
            "source_image_sha256": image_hash,
            "source_image_width": width,
            "source_image_height": height,
            "official_owner_record_sha256": sha256_json(owner_records[image_id]),
            "route_replay_input_record_sha256": sha256_json(replay_records[image_id]),
        }
        for image_id in ids
    ]
    bindings = {"mode": "synthetic_test_only", "shared_image_sha256": image_hash}
    manifest = {
        "schema_version": MEMBER_MANIFEST_SCHEMA_VERSION,
        "population_scope": STAGE_ZERO_POPULATION_SCOPE,
        "count": len(ids),
        "ordered_image_ids": ids,
        "ordered_image_ids_sha256": sha256_json(ids),
        "members": members,
        "members_sha256": sha256_json(members),
        "input_bindings": bindings,
        "input_bindings_sha256": sha256_json(bindings),
    }
    return validate_member_manifest(manifest, ids)


def build_production_member_manifest(
    stage: StageZeroBinding,
) -> dict[str, object]:
    """Reconstruct all possible members from exact Stage-Zero frozen inputs."""

    producer = _stage_zero_module()
    inputs = _mapping(stage.receipt.get("inputs"), "Stage-Zero inputs")
    sampled_binding = _mapping(
        inputs.get("sampled_manifest_binding"), "sampled binding"
    )
    source_binding = _mapping(inputs.get("source_manifest_binding"), "source binding")
    sampled_batches = sampled_binding.get("batches")
    source_batches = source_binding.get("batches")
    if not isinstance(sampled_batches, list) or not isinstance(source_batches, list):
        raise SelectionError("Stage-Zero receipt lacks strict panel batch inventories")
    candidate_path = producer.FROZEN_CANDIDATE_POOL_PATH.resolve(strict=True)
    candidate_bytes = candidate_path.read_bytes()
    candidate_hash = sha256_bytes(candidate_bytes)
    if candidate_hash != producer.FROZEN_CANDIDATE_POOL_SHA256:
        raise SelectionError("frozen candidate pool hash differs")
    candidate_pool = producer._candidate_pool(candidate_path)
    adapter = producer.load_v2_b16_panel_adapter(
        sampled_panel_root=Path(str(inputs["sampled_root"])),
        source_b16_root=Path(str(inputs["source_root"])),
        candidate_pool=candidate_path,
        semantic_image_ids=stage.possible_image_ids,
        sampled_artifact_inventory=sampled_batches,
        source_artifact_inventory=source_batches,
        expected_candidate_pool_sha256=candidate_hash,
    )
    route_ids = ["source-b16", *(f"sample-{index:02d}" for index in range(16))]
    members: list[dict[str, object]] = []
    for image_id in stage.possible_image_ids:
        candidate = _mapping(candidate_pool[image_id], f"candidate {image_id}")
        images = candidate.get("images")
        if (
            not isinstance(images, list)
            or len(images) != 1
            or not isinstance(images[0], str)
        ):
            raise SelectionError(f"candidate {image_id} lacks one source image")
        source_path = (candidate_path.parent / images[0]).resolve(strict=True)
        if source_path.is_symlink() or not source_path.is_file():
            raise SelectionError(f"candidate {image_id} source path is invalid")
        image_bytes = source_path.read_bytes()
        image_hash = sha256_bytes(image_bytes)
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                width, height = image.size
                image.verify()
        except Exception as exc:
            raise SelectionError(
                f"candidate {image_id} source image is undecodable"
            ) from exc
        result = _mapping(
            adapter["image_results"][image_id], f"image result {image_id}"
        )
        evidence = _mapping(result.get("trajectory_evidence"), "trajectory evidence")
        budgets = result.get("budgets")
        if not isinstance(budgets, list) or len(budgets) != 1:
            raise SelectionError(f"image {image_id} lacks one exact B16 budget")
        assignments = _mapping(budgets[0].get("trajectory_assignments"), "assignments")
        replay_routes: list[dict[str, object]] = []
        for route_id in route_ids:
            route_row = (
                adapter["source_rows"].get((image_id, 0))
                if route_id == "source-b16"
                else adapter["sampled_rows"].get((image_id, int(route_id[-2:])))
            )
            if not isinstance(route_row, Mapping):
                raise SelectionError(f"image {image_id} lacks {route_id}")
            replay_routes.append(
                {
                    "route_id": route_id,
                    "route_row": copy.deepcopy(dict(route_row)),
                    "trajectory_evidence": copy.deepcopy(
                        dict(_mapping(evidence.get(route_id), f"evidence {route_id}"))
                    ),
                    "assignment": copy.deepcopy(
                        dict(
                            _mapping(
                                assignments.get(route_id), f"assignment {route_id}"
                            )
                        )
                    ),
                }
            )
        members.append(
            {
                "image_id": image_id,
                "canonical_source_path": str(source_path),
                "source_image_bytes": len(image_bytes),
                "source_image_sha256": image_hash,
                "source_image_width": width,
                "source_image_height": height,
                "official_owner_record_sha256": sha256_json(result.get("owners")),
                "route_replay_input_record_sha256": sha256_json(
                    {"image_id": image_id, "routes": replay_routes}
                ),
            }
        )
    bindings = {
        "candidate_pool_path": str(candidate_path),
        "candidate_pool_bytes": len(candidate_bytes),
        "candidate_pool_sha256": candidate_hash,
        "sampled_manifest_binding_sha256": sha256_json(sampled_binding),
        "source_manifest_binding_sha256": sha256_json(source_binding),
        "execution_model_identity_sha256": inputs.get(
            "execution_model_identity_sha256"
        ),
        "tokenizer_identity_sha256": inputs.get("tokenizer_identity_sha256"),
        "stage_zero_receipt_sha256": stage.receipt_sha256,
    }
    manifest = {
        "schema_version": MEMBER_MANIFEST_SCHEMA_VERSION,
        "population_scope": STAGE_ZERO_POPULATION_SCOPE,
        "count": len(stage.possible_image_ids),
        "ordered_image_ids": list(stage.possible_image_ids),
        "ordered_image_ids_sha256": stage.ordered_pool_sha256,
        "members": members,
        "members_sha256": sha256_json(members),
        "input_bindings": bindings,
        "input_bindings_sha256": sha256_json(bindings),
    }
    return validate_member_manifest(manifest, stage.possible_image_ids)


def journal_root_for(output_root: Path) -> Path:
    output = output_root.expanduser().absolute()
    return output.with_name(f"{output.name}.entropy-journal-v1")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_write_bytes(
    path: Path, payload: bytes, *, created_fault: str | None, fault: FaultInjector
) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        if created_fault is not None:
            fault(created_fault)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    path.chmod(0o444)
    _fsync_directory(path.parent)


def _exclusive_write_json(
    path: Path, value: object, *, created_fault: str | None, fault: FaultInjector
) -> None:
    _exclusive_write_bytes(
        path, _pretty_json_bytes(value), created_fault=created_fault, fault=fault
    )


def _journal_claim(
    *,
    output_root: Path,
    source_manifest: Mapping[str, Any],
    stage: StageZeroBinding,
    member_manifest: Mapping[str, Any],
) -> dict[str, object]:
    claim = {
        "schema_version": JOURNAL_CLAIM_SCHEMA_VERSION,
        "output_root": str(output_root.expanduser().absolute()),
        "selector": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "unit_contract": {"path": str(UNIT_PATH), "sha256": FINAL_UNIT_SHA256},
        "contract_review": {
            "path": str(CONTRACT_REVIEW_PATH),
            "sha256": CONTRACT_REVIEW_SHA256,
        },
        "frozen_source_manifest": copy.deepcopy(dict(source_manifest)),
        "frozen_source_manifest_sha256": sha256_json(source_manifest),
        "stage_zero": stage.to_record(),
        "stage_zero_binding_sha256": sha256_json(stage.to_record()),
        "population_scope": POPULATION_SCOPE,
        "population_size_N": len(stage.possible_image_ids),
        "ordered_possible_pool_sha256": stage.ordered_pool_sha256,
        "sample_size": SAMPLE_SIZE,
        "member_manifest_sha256": sha256_json(member_manifest),
        "member_records_sha256": member_manifest["members_sha256"],
        "member_input_bindings_sha256": member_manifest["input_bindings_sha256"],
    }
    return claim


def _read_journal(root: Path) -> JournalState:
    """Inspect an existing journal without ever invoking an entropy factory."""

    canonical = root.expanduser().absolute()
    if canonical.is_symlink() or not canonical.is_dir():
        return JournalState(
            canonical,
            None,
            None,
            None,
            True,
            semantic_void_reason="journal root is invalid",
        )
    terminal: Mapping[str, Any] | None = None
    terminal_path = canonical / "terminal.json"
    if terminal_path.exists():
        try:
            terminal_value, _ = _read_json(terminal_path, "journal terminal")
            terminal = terminal_value
        except (OSError, SelectionError) as exc:
            return JournalState(
                canonical,
                None,
                None,
                None,
                True,
                semantic_void_reason=f"journal terminal is invalid: {exc}",
            )
    claim_path = canonical / "claim.json"
    if not claim_path.exists():
        return JournalState(
            canonical,
            None,
            None,
            None,
            True,
            terminal=terminal,
            persistence_unknown_reason="journal claim is absent after root creation",
        )
    try:
        claim, claim_raw = _read_json(claim_path, "journal claim")
    except (OSError, SelectionError) as exc:
        return JournalState(
            canonical,
            None,
            None,
            None,
            True,
            terminal=terminal,
            persistence_unknown_reason=f"journal claim persistence is unknown: {exc}",
        )
    entropy_path = canonical / "entropy.bin"
    entropy_receipt_path = canonical / "entropy-receipt.json"
    if not entropy_path.exists() or not entropy_receipt_path.exists():
        return JournalState(
            canonical,
            claim,
            sha256_bytes(claim_raw),
            None,
            True,
            terminal=terminal,
            persistence_unknown_reason="complete persisted entropy record is absent",
        )
    try:
        entropy = entropy_path.read_bytes()
        entropy_receipt, _ = _read_json(entropy_receipt_path, "entropy receipt")
    except (OSError, SelectionError) as exc:
        return JournalState(
            canonical,
            claim,
            sha256_bytes(claim_raw),
            None,
            True,
            terminal=terminal,
            persistence_unknown_reason=f"persisted entropy readback is unknown: {exc}",
        )
    if len(entropy) != ENTROPY_BYTE_COUNT or entropy_receipt != {
        "byte_count": ENTROPY_BYTE_COUNT,
        "sha256": sha256_bytes(entropy),
    }:
        return JournalState(
            canonical,
            claim,
            sha256_bytes(claim_raw),
            None,
            True,
            terminal=terminal,
            persistence_unknown_reason="persisted entropy bytes or receipt are incomplete",
        )
    allowed = {"claim.json", "entropy.bin", "entropy-receipt.json", "terminal.json"}
    if {item.name for item in canonical.iterdir()} - allowed:
        return JournalState(
            canonical,
            claim,
            sha256_bytes(claim_raw),
            entropy,
            True,
            terminal=terminal,
            semantic_void_reason="journal inventory contains unexpected entries",
        )
    return JournalState(
        canonical,
        claim,
        sha256_bytes(claim_raw),
        entropy,
        True,
        terminal=terminal,
    )


def _claim_and_acquire_entropy(
    root: Path,
    claim: Mapping[str, Any],
    *,
    entropy_factory: EntropyFactory,
    fault: FaultInjector,
) -> JournalState:
    """Exclusively claim once, then acquire and durably bind exactly 64 bytes."""

    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir(mode=0o700)
    except FileExistsError:
        return _read_journal(root)
    _fsync_directory(root.parent)
    fault("claim_root_created")
    claim_path = root / "claim.json"
    _exclusive_write_json(claim_path, claim, created_fault=None, fault=fault)
    fault("claim_file_fsynced")
    observed_claim, claim_raw = _read_json(claim_path, "journal claim")
    if observed_claim != claim:
        raise SelectionError("journal claim readback differs")
    fault("claim_readback_validated")
    entropy = entropy_factory(ENTROPY_BYTE_COUNT)
    fault("entropy_acquired")
    if not isinstance(entropy, bytes) or len(entropy) != ENTROPY_BYTE_COUNT:
        raise SelectionError("entropy factory did not return exactly 64 bytes")
    entropy_path = root / "entropy.bin"
    _exclusive_write_bytes(
        entropy_path, entropy, created_fault="entropy_file_created", fault=fault
    )
    fault("entropy_file_fsynced")
    entropy_receipt = {
        "byte_count": ENTROPY_BYTE_COUNT,
        "sha256": sha256_bytes(entropy),
    }
    _exclusive_write_json(
        root / "entropy-receipt.json",
        entropy_receipt,
        created_fault=None,
        fault=fault,
    )
    observed_entropy = entropy_path.read_bytes()
    observed_receipt, _ = _read_json(root / "entropy-receipt.json", "entropy receipt")
    if observed_entropy != entropy or observed_receipt != entropy_receipt:
        raise SelectionError("entropy persistence readback differs")
    fault("entropy_readback_validated")
    return JournalState(
        root,
        observed_claim,
        sha256_bytes(claim_raw),
        entropy,
        False,
    )


def _record_journal_terminal(
    state: JournalState,
    *,
    status: TerminalStatus,
    claim_sha256: str | None,
    reason: str | None,
) -> dict[str, object]:
    terminal = {
        "schema_version": JOURNAL_TERMINAL_SCHEMA_VERSION,
        "terminal_status": status,
        "claim_sha256": claim_sha256,
        "entropy_sha256": (
            None if state.entropy is None else sha256_bytes(state.entropy)
        ),
        "reason": reason,
    }
    path = state.root / "terminal.json"
    if path.exists():
        observed, _ = _read_json(path, "journal terminal")
        if observed != terminal:
            raise SelectionError("journal terminal marker differs")
    else:
        # A prior interrupted write may have left the directory read-only only
        # after a terminal marker; otherwise make it writable for this one file.
        state.root.chmod(0o700)
        _exclusive_write_json(path, terminal, created_fault=None, fault=_noop_fault)
    for child in state.root.iterdir():
        if child.is_file():
            child.chmod(0o444)
    state.root.chmod(0o555)
    _fsync_directory(state.root.parent)
    observed, _ = _read_json(path, "journal terminal")
    if observed != terminal:
        raise SelectionError("journal terminal readback differs")
    return terminal


def _selection_record(
    *,
    state: JournalState,
    stage: StageZeroBinding,
    source_manifest: Mapping[str, Any],
    member_manifest: Mapping[str, Any],
    randomization: RandomizationResult,
) -> dict[str, object]:
    if not randomization.accepted or state.entropy is None:
        raise SelectionError(
            "cannot build a completed selection without accepted entropy"
        )
    selected = list(randomization.ordered_sample)
    look_one = selected[:LOOK_ONE_SIZE]
    look_two_additional = selected[LOOK_ONE_SIZE:]
    if len(selected) != SAMPLE_SIZE or len(set(selected)) != SAMPLE_SIZE:
        raise SelectionError("accepted sample is not 32 distinct image IDs")
    return {
        "schema_version": SCHEMA_VERSION,
        "terminal_status": "completed",
        "population_scope": POPULATION_SCOPE,
        "population_size_N": len(stage.possible_image_ids),
        "sample_size": SAMPLE_SIZE,
        "ordered_possible_pool_sha256": stage.ordered_pool_sha256,
        "ordered_sample_sha256": sha256_json(selected),
        "look_one_image_ids": look_one,
        "look_two_additional_image_ids": look_two_additional,
        "look_two_cumulative_image_ids": selected,
        "entropy": {
            "byte_count": len(state.entropy),
            "bytes_hex": state.entropy.hex(),
            "sha256": sha256_bytes(state.entropy),
            "permutation_count_M": randomization.permutation_count_M,
            "entropy_integer_R": randomization.entropy_integer_R,
            "acceptance_limit_L": randomization.acceptance_limit_L,
            "accepted": True,
            "accepted_initial_rank": randomization.accepted_initial_rank,
            "unranking_trace": [dict(item) for item in randomization.unranking_trace],
        },
        "hypergeometric_design": hypergeometric_design(len(stage.possible_image_ids)),
        "frozen_source_manifest_sha256": sha256_json(source_manifest),
        "stage_zero_binding_sha256": sha256_json(stage.to_record()),
        "possible_pool_member_manifest_sha256": sha256_json(member_manifest),
        "possible_pool_member_records_sha256": member_manifest["members_sha256"],
        "stage_zero_replay_validation": {
            "complete_category_witness_and_certificate_replay": True,
            "independent_audit_sha256": stage.audit_sha256,
            "stage_zero_root_inventory_sha256": stage.inventory_sha256,
        },
    }


def _selection_receipt(
    *,
    output_root: Path,
    journal: JournalState,
    claim: Mapping[str, Any],
    selection: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    member_manifest: Mapping[str, Any],
    outputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    return {
        "schema_version": f"{SCHEMA_VERSION}.receipt.v2",
        "terminal_status": "completed",
        "output_root": str(output_root),
        "journal_root": str(journal.root),
        "journal_claim_file_sha256": journal.claim_file_sha256,
        "journal_claim_sha256": sha256_json(claim),
        "entropy": copy.deepcopy(dict(selection["entropy"])),
        "randomization": {
            "population_size_N": selection["population_size_N"],
            "sample_size": selection["sample_size"],
            "ordered_possible_pool_sha256": selection["ordered_possible_pool_sha256"],
            "ordered_sample_sha256": selection["ordered_sample_sha256"],
            "look_one_image_ids": copy.deepcopy(selection["look_one_image_ids"]),
            "look_two_additional_image_ids": copy.deepcopy(
                selection["look_two_additional_image_ids"]
            ),
            "look_two_cumulative_image_ids": copy.deepcopy(
                selection["look_two_cumulative_image_ids"]
            ),
            "hypergeometric_design": copy.deepcopy(selection["hypergeometric_design"]),
        },
        "bindings": {
            "selector_sha256": sha256_file(Path(__file__).resolve()),
            "unit_contract_sha256": FINAL_UNIT_SHA256,
            "contract_review_sha256": CONTRACT_REVIEW_SHA256,
            "stage_zero_producer_sha256": STAGE_ZERO_PRODUCER_SHA256,
            "frozen_source_manifest_sha256": sha256_json(source_manifest),
            "stage_zero_binding_sha256": claim["stage_zero_binding_sha256"],
            "stage_zero_receipt_sha256": claim["stage_zero"]["receipt_sha256"],
            "stage_zero_root_inventory_sha256": claim["stage_zero"][
                "root_inventory_sha256"
            ],
            "stage_zero_independent_audit_sha256": claim["stage_zero"]["audit_sha256"],
            "possible_pool_member_manifest_sha256": sha256_json(member_manifest),
            "possible_pool_member_records_sha256": member_manifest["members_sha256"],
            "member_input_bindings_sha256": member_manifest["input_bindings_sha256"],
        },
        "replay_validation": copy.deepcopy(
            dict(selection["stage_zero_replay_validation"])
        ),
        "outputs": copy.deepcopy(dict(outputs)),
        "publication_contract": {
            "method": "same_parent_hidden_staging_then_os_replace",
            "completed_file_mode": "0444",
            "completed_directory_mode": "0555",
            "hidden_until_terminal_readback": True,
            "invalid_post_rename_root_quarantined": True,
            "exact_inventory_hash_and_size_checked": True,
        },
    }


def _freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise SelectionError(f"cannot freeze symlink: {path}")
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
        else:
            raise SelectionError(f"cannot freeze special entry: {path}")
    root.chmod(0o555)


def _make_staging(output_root: Path) -> Path:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(f".{output_root.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(mode=0o700)
    return staging


def _discard_staging(staging: Path) -> None:
    if not staging.exists():
        return
    for path in [staging, *staging.rglob("*")]:
        try:
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o700)
            elif path.exists():
                path.chmod(0o600)
        except OSError:
            pass
    shutil.rmtree(staging)


def _quarantine_output(output_root: Path) -> Path:
    quarantine = output_root.with_name(f"{output_root.name}.invalid-{uuid.uuid4().hex}")
    os.replace(output_root, quarantine)
    _fsync_directory(output_root.parent)
    return quarantine


def _expected_completed_paths(source_manifest: Mapping[str, Any]) -> set[str]:
    entries = source_manifest.get("entries")
    if not isinstance(entries, list):
        raise SelectionError("frozen source manifest lacks entries")
    snapshots = {
        str(_mapping(item, "source entry")["snapshot_relative_path"])
        for item in entries
    }
    return set(BASE_SELECTION_OUTPUTS) | snapshots


def _validate_completed_root(
    root: Path,
    *,
    expected_selection: Mapping[str, Any],
    expected_source_manifest: Mapping[str, Any],
    expected_member_manifest: Mapping[str, Any],
    require_immutable: bool,
) -> dict[str, Any]:
    if require_immutable:
        canonical = _require_immutable_tree(root, "completed selection root")
    else:
        canonical = root.resolve(strict=True)
    selection, _ = _read_json(canonical / "selection.json", "selection")
    source_manifest, _ = _read_json(
        canonical / "frozen-source-manifest.json", "frozen source manifest"
    )
    member_manifest, _ = _read_json(
        canonical / "possible-pool-member-manifest.json", "member manifest"
    )
    if selection != expected_selection:
        raise SelectionError("serialized selection differs")
    if source_manifest != expected_source_manifest:
        raise SelectionError("serialized frozen source manifest differs")
    if member_manifest != expected_member_manifest:
        raise SelectionError("serialized member manifest differs")
    entries = source_manifest.get("entries")
    assert isinstance(entries, list)
    for raw in entries:
        entry = _mapping(raw, "source entry")
        snapshot = canonical / str(entry["snapshot_relative_path"])
        payload = snapshot.read_bytes()
        if len(payload) != entry["bytes"] or sha256_bytes(payload) != entry["sha256"]:
            raise SelectionError(f"source snapshot differs: {entry['label']}")
    inventory = _root_inventory(canonical)
    receipt_entry = inventory.pop("receipt.json", None)
    if receipt_entry is None or set(inventory) != _expected_completed_paths(
        source_manifest
    ):
        raise SelectionError("completed selection inventory differs")
    receipt, _ = _read_json(canonical / "receipt.json", "selection receipt")
    if (
        receipt.get("terminal_status") != "completed"
        or receipt.get("outputs") != inventory
    ):
        raise SelectionError("completed selection receipt output inventory differs")
    return dict(receipt)


def _publish_completed(
    *,
    output_root: Path,
    journal: JournalState,
    claim: Mapping[str, Any],
    selection: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    member_manifest: Mapping[str, Any],
    fault: FaultInjector,
) -> dict[str, Any]:
    staging = _make_staging(output_root)
    renamed = False
    try:
        fault("materialization_started")
        (staging / "source-snapshots").mkdir(mode=0o700)
        (staging / "selection.json").write_bytes(_pretty_json_bytes(selection))
        (staging / "frozen-source-manifest.json").write_bytes(
            _pretty_json_bytes(source_manifest)
        )
        (staging / "possible-pool-member-manifest.json").write_bytes(
            _pretty_json_bytes(member_manifest)
        )
        entries = source_manifest.get("entries")
        assert isinstance(entries, list)
        for raw in entries:
            entry = _mapping(raw, "source entry")
            source_path = Path(str(entry["path"]))
            payload = source_path.read_bytes()
            if (
                len(payload) != entry["bytes"]
                or sha256_bytes(payload) != entry["sha256"]
            ):
                raise SelectionError(f"source {entry['label']} drifted during snapshot")
            (staging / str(entry["snapshot_relative_path"])).write_bytes(payload)
        outputs = _root_inventory(staging)
        receipt = _selection_receipt(
            output_root=output_root,
            journal=journal,
            claim=claim,
            selection=selection,
            source_manifest=source_manifest,
            member_manifest=member_manifest,
            outputs=outputs,
        )
        (staging / "receipt.json").write_bytes(_pretty_json_bytes(receipt))
        _validate_completed_root(
            staging,
            expected_selection=selection,
            expected_source_manifest=source_manifest,
            expected_member_manifest=member_manifest,
            require_immutable=False,
        )
        fault("materialization_readback_validated")
        _freeze_tree(staging)
        _validate_completed_root(
            staging,
            expected_selection=selection,
            expected_source_manifest=source_manifest,
            expected_member_manifest=member_manifest,
            require_immutable=True,
        )
        if output_root.exists():
            raise SelectionError("selection output root appeared before publication")
        os.replace(staging, output_root)
        renamed = True
        _fsync_directory(output_root.parent)
        fault("post_rename")
        result = _validate_completed_root(
            output_root,
            expected_selection=selection,
            expected_source_manifest=source_manifest,
            expected_member_manifest=member_manifest,
            require_immutable=True,
        )
        fault("post_rename_readback_validated")
        return result
    except BaseException:
        if renamed and output_root.exists():
            _quarantine_output(output_root)
        elif staging.exists():
            _discard_staging(staging)
        raise


def _terminal_output_receipt(
    *,
    output_root: Path,
    state: JournalState,
    status: TerminalStatus,
    reason: str | None,
    randomization: RandomizationResult | None,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema_version": f"{SCHEMA_VERSION}.receipt.v2",
        "terminal_status": status,
        "output_root": str(output_root),
        "journal_root": str(state.root),
        "journal_claim_file_sha256": state.claim_file_sha256,
        "selection_published": False,
        "reason": reason,
        "outputs": {},
    }
    if state.entropy is not None:
        receipt["entropy"] = {
            "byte_count": len(state.entropy),
            "bytes_hex": state.entropy.hex(),
            "sha256": sha256_bytes(state.entropy),
        }
    if randomization is not None:
        receipt["randomization"] = {
            "population_size_N": (
                None if state.claim is None else state.claim.get("population_size_N")
            ),
            "sample_size": SAMPLE_SIZE,
            "permutation_count_M": randomization.permutation_count_M,
            "entropy_integer_R": randomization.entropy_integer_R,
            "acceptance_limit_L": randomization.acceptance_limit_L,
            "accepted": randomization.accepted,
            "accepted_initial_rank": randomization.accepted_initial_rank,
            "unranking_trace": [dict(item) for item in randomization.unranking_trace],
        }
    return receipt


def _validate_terminal_output(
    root: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    canonical = _require_immutable_tree(root, "terminal selection root")
    inventory = _root_inventory(canonical)
    if set(inventory) != {"receipt.json"}:
        raise SelectionError("terminal selection root inventory differs")
    receipt, _ = _read_json(canonical / "receipt.json", "terminal receipt")
    if receipt != expected:
        raise SelectionError("terminal selection receipt differs")
    return dict(receipt)


def _publish_terminal_output(
    output_root: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    if output_root.exists():
        try:
            return _validate_terminal_output(output_root, expected)
        except (OSError, SelectionError):
            _quarantine_output(output_root)
    staging = _make_staging(output_root)
    try:
        (staging / "receipt.json").write_bytes(_pretty_json_bytes(expected))
        observed, _ = _read_json(staging / "receipt.json", "terminal receipt")
        if observed != expected or set(_root_inventory(staging)) != {"receipt.json"}:
            raise SelectionError("terminal receipt staging readback differs")
        _freeze_tree(staging)
        os.replace(staging, output_root)
        _fsync_directory(output_root.parent)
        return _validate_terminal_output(output_root, expected)
    except BaseException:
        if staging.exists():
            _discard_staging(staging)
        raise


def _validate_terminal_marker(terminal: Mapping[str, Any]) -> TerminalStatus:
    _exact_keys(
        terminal,
        {
            "schema_version",
            "terminal_status",
            "claim_sha256",
            "entropy_sha256",
            "reason",
        },
        "journal terminal marker",
    )
    status = terminal.get("terminal_status")
    if terminal.get(
        "schema_version"
    ) != JOURNAL_TERMINAL_SCHEMA_VERSION or status not in {
        "completed",
        "entropy_rejected",
        "entropy_persistence_unknown",
        "void",
    }:
        raise SelectionError("journal terminal marker differs")
    return status


def _validate_existing_completed_output(output_root: Path) -> dict[str, Any]:
    canonical = _require_immutable_tree(output_root, "completed selection root")
    selection, _ = _read_json(canonical / "selection.json", "selection")
    source_manifest, _ = _read_json(
        canonical / "frozen-source-manifest.json", "frozen source manifest"
    )
    member_manifest, _ = _read_json(
        canonical / "possible-pool-member-manifest.json", "member manifest"
    )
    return _validate_completed_root(
        canonical,
        expected_selection=selection,
        expected_source_manifest=source_manifest,
        expected_member_manifest=member_manifest,
        require_immutable=True,
    )


def _finish_terminal(
    *,
    output_root: Path,
    state: JournalState,
    status: TerminalStatus,
    reason: str | None,
    randomization: RandomizationResult | None,
) -> dict[str, Any]:
    _record_journal_terminal(
        state,
        status=status,
        claim_sha256=(None if state.claim is None else sha256_json(state.claim)),
        reason=reason,
    )
    expected = _terminal_output_receipt(
        output_root=output_root,
        state=state,
        status=status,
        reason=reason,
        randomization=randomization,
    )
    return _publish_terminal_output(output_root, expected)


def _current_bound_inputs(
    *,
    stage_zero_root: Path,
    stage_zero_audit: Path,
    named_sources: Sequence[NamedSource],
    expected_named_labels: frozenset[str],
    member_manifest_provider: MemberManifestProvider | None,
    synthetic_test_only: bool,
) -> tuple[dict[str, Any], StageZeroBinding, dict[str, Any], MemberManifestProvider]:
    source_manifest = build_frozen_source_manifest(
        named_sources, expected_named_labels=expected_named_labels
    )
    revalidate_frozen_source_manifest(source_manifest)
    stage = validate_stage_zero_root(
        stage_zero_root,
        stage_zero_audit,
        synthetic_test_only=synthetic_test_only,
    )
    hypergeometric_design(len(stage.possible_image_ids))
    if synthetic_test_only:
        if member_manifest_provider is None:
            raise SelectionError(
                "synthetic tests must inject a member manifest provider"
            )
        provider = member_manifest_provider
    else:
        if member_manifest_provider is not None:
            raise SelectionError(
                "production member manifests must be reconstructed by the selector"
            )

        def provider() -> Mapping[str, Any]:
            return build_production_member_manifest(stage)

    member_manifest = validate_member_manifest(provider(), stage.possible_image_ids)
    return source_manifest, stage, member_manifest, provider


def execute_selection(
    *,
    stage_zero_root: Path,
    stage_zero_audit: Path,
    output_root: Path,
    named_sources: Sequence[NamedSource],
    expected_named_labels: frozenset[str] = PRODUCTION_NAMED_SOURCE_LABELS,
    member_manifest_provider: MemberManifestProvider | None = None,
    entropy_factory: EntropyFactory = secrets.token_bytes,
    fault_injector: FaultInjector | None = None,
    synthetic_test_only: bool = False,
) -> dict[str, Any]:
    """Validate, claim, draw once, and publish or resume the exact selection."""

    fault = fault_injector or _noop_fault
    output = output_root.expanduser().absolute()
    journal_root = journal_root_for(output)
    state: JournalState | None = None
    if journal_root.exists():
        state = _read_journal(journal_root)
        if state.terminal is not None:
            status = _validate_terminal_marker(state.terminal)
            if status == "completed":
                if not output.exists():
                    raise MechanicalSelectionError(
                        "completed journal exists without its immutable selection root"
                    )
                return _validate_existing_completed_output(output)
            randomization = None
            if (
                status == "entropy_rejected"
                and state.entropy is not None
                and state.claim is not None
            ):
                pool_size = int(state.claim["population_size_N"])
                placeholder_pool = [str(index) for index in range(pool_size)]
                randomization = randomization_from_entropy(
                    placeholder_pool, state.entropy
                )
                if randomization.accepted:
                    raise SelectionError(
                        "rejected journal terminal has accepted entropy"
                    )
            expected = _terminal_output_receipt(
                output_root=output,
                state=state,
                status=status,
                reason=(
                    None if state.terminal is None else state.terminal.get("reason")
                ),
                randomization=randomization,
            )
            return _publish_terminal_output(output, expected)
        if state.persistence_unknown_reason is not None:
            return _finish_terminal(
                output_root=output,
                state=state,
                status="entropy_persistence_unknown",
                reason=state.persistence_unknown_reason,
                randomization=None,
            )
        if state.semantic_void_reason is not None:
            return _finish_terminal(
                output_root=output,
                state=state,
                status="void",
                reason=state.semantic_void_reason,
                randomization=None,
            )

    try:
        source_manifest, stage, member_manifest, provider = _current_bound_inputs(
            stage_zero_root=stage_zero_root,
            stage_zero_audit=stage_zero_audit,
            named_sources=named_sources,
            expected_named_labels=expected_named_labels,
            member_manifest_provider=member_manifest_provider,
            synthetic_test_only=synthetic_test_only,
        )
    except OSError as exc:
        raise MechanicalSelectionError(
            f"mechanical pre-claim validation failure: {exc}"
        ) from exc
    except SelectionError as exc:
        if state is None or state.entropy is None:
            raise
        return _finish_terminal(
            output_root=output,
            state=state,
            status="void",
            reason=f"post-entropy binding validation failed: {exc}",
            randomization=None,
        )
    claim = _journal_claim(
        output_root=output,
        source_manifest=source_manifest,
        stage=stage,
        member_manifest=member_manifest,
    )
    if state is None:
        if output.exists():
            raise SelectionError(
                "selection output exists without its canonical journal"
            )
        try:
            state = _claim_and_acquire_entropy(
                journal_root,
                claim,
                entropy_factory=entropy_factory,
                fault=fault,
            )
        except OSError as exc:
            raise MechanicalSelectionError(
                f"mechanical journal persistence failure: {exc}"
            ) from exc
        except SelectionError:
            observed = _read_journal(journal_root)
            if observed.persistence_unknown_reason is not None:
                return _finish_terminal(
                    output_root=output,
                    state=observed,
                    status="entropy_persistence_unknown",
                    reason=observed.persistence_unknown_reason,
                    randomization=None,
                )
            raise
    assert state is not None
    if state.persistence_unknown_reason is not None:
        return _finish_terminal(
            output_root=output,
            state=state,
            status="entropy_persistence_unknown",
            reason=state.persistence_unknown_reason,
            randomization=None,
        )
    if state.semantic_void_reason is not None:
        return _finish_terminal(
            output_root=output,
            state=state,
            status="void",
            reason=state.semantic_void_reason,
            randomization=None,
        )
    if state.claim != claim:
        return _finish_terminal(
            output_root=output,
            state=state,
            status="void",
            reason="current pre-entropy bindings differ from the durable journal claim",
            randomization=None,
        )
    if state.entropy is None:
        return _finish_terminal(
            output_root=output,
            state=state,
            status="entropy_persistence_unknown",
            reason="journal does not contain complete readback-valid entropy",
            randomization=None,
        )
    randomization = randomization_from_entropy(stage.possible_image_ids, state.entropy)
    if not randomization.accepted:
        return _finish_terminal(
            output_root=output,
            state=state,
            status="entropy_rejected",
            reason=None,
            randomization=randomization,
        )
    try:
        revalidate_frozen_source_manifest(source_manifest)
        revalidated_stage = validate_stage_zero_root(
            stage_zero_root,
            stage_zero_audit,
            synthetic_test_only=synthetic_test_only,
        )
        if revalidated_stage.to_record() != stage.to_record():
            raise SelectionError("Stage-Zero binding drifted after entropy persistence")
        revalidated_members = validate_member_manifest(
            provider(), stage.possible_image_ids
        )
        if revalidated_members != member_manifest:
            raise SelectionError("possible-pool member manifest drifted after entropy")
        if (
            _journal_claim(
                output_root=output,
                source_manifest=source_manifest,
                stage=revalidated_stage,
                member_manifest=revalidated_members,
            )
            != claim
        ):
            raise SelectionError("durable claim inputs drifted after entropy")
        selection = _selection_record(
            state=state,
            stage=stage,
            source_manifest=source_manifest,
            member_manifest=member_manifest,
            randomization=randomization,
        )
        if output.exists():
            receipt = _validate_completed_root(
                output,
                expected_selection=selection,
                expected_source_manifest=source_manifest,
                expected_member_manifest=member_manifest,
                require_immutable=True,
            )
        else:
            receipt = _publish_completed(
                output_root=output,
                journal=state,
                claim=claim,
                selection=selection,
                source_manifest=source_manifest,
                member_manifest=member_manifest,
                fault=fault,
            )
        _record_journal_terminal(
            state,
            status="completed",
            claim_sha256=sha256_json(claim),
            reason=None,
        )
        return receipt
    except OSError as exc:
        raise MechanicalSelectionError(
            f"mechanical post-entropy materialization failure: {exc}"
        ) from exc
    except SelectionError as exc:
        return _finish_terminal(
            output_root=output,
            state=state,
            status="void",
            reason=f"post-entropy semantic validation failed: {exc}",
            randomization=None,
        )


def _parse_named_source(value: str) -> NamedSource:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected LABEL=PATH=SHA256")
    label, raw_path, expected_sha256 = parts
    if not _LABEL.fullmatch(label) or not _SHA256.fullmatch(expected_sha256):
        raise argparse.ArgumentTypeError("invalid source label or SHA-256")
    return NamedSource(label, Path(raw_path), expected_sha256)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-zero-root", type=Path, default=FROZEN_STAGE_ZERO_ROOT)
    parser.add_argument("--stage-zero-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=FROZEN_OUTPUT_ROOT)
    parser.add_argument(
        "--named-source",
        action="append",
        type=_parse_named_source,
        default=[],
        metavar="LABEL=PATH=SHA256",
        help="repeat for the exact explicit/transitive review source label set",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    receipt = execute_selection(
        stage_zero_root=args.stage_zero_root,
        stage_zero_audit=args.stage_zero_audit,
        output_root=args.output_root,
        named_sources=args.named_source,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
