#!/usr/bin/env python3
"""Task-3 runtime scorer for the sorted-owner-basin-landscape-and-repair unit.

Experiment-local to
``docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/
2026-08-01-sorted-owner-basin-landscape-and-repair/unit.md``.

Scope
-----
This module is the *runtime execution engine* for Task 3. It consumes:

- sealed Task 0/1 ledgers (``owner-ledger.jsonl``, ``prediction-row-
  ledger.jsonl``);
- ``landscape-decision-rules.json`` (schema tokens, per-owner canonical
  description token IDs, frozen foil-set digests, numeric tolerance);
- deterministic candidate/foil rows (``landscape-candidates.jsonl``), each
  either a literal complete-box point candidate or a dense full-vocabulary
  scan request at one coordinate slot;

and emits versioned ``landscape-scores.jsonl`` rows plus an execution
receipt. It executes both the frozen bounded free coordinate tree and the
authoritative pre-score restricted bank. Basin clustering, peak/prominence
math, and calibration remain in the deterministic pure core,
``scripts/research/sorted_owner_basin_landscape.py``. Production execution
fails before scoring when that core or any v2 builder/input binding is absent
or incompatible.

Execution architecture (audit-mandated)
----------------------------------------
A passed cache-admission gate selects the KV-cache path for the entire run.
The default gate retains strict all-vocabulary parity. An explicit relaxed
request may instead admit only one exact context/history group after
coordinate argmax and selected-token logprob stability checks pass at all
four slots and both policy views; the failed strict result remains visible in
the receipt. A failed admission selects a true ``use_cache=False``
full-reforward reference backend for every score row; the two paths are never
mixed. The literal fallback may co-schedule equal-length
same-context sequences in an explicitly admitted GPU batch; a coordinate-
behavior parity probe falls back to scalar execution if batching changes the
reference beyond its frozen tolerance. On the cache path, every request for a given
(image, self-prefix) context shares one prefill through ``<|box_start|>`` and
advances that cache one coordinate token at a time (:class:`BranchCursor`).
``transformers`` ``DynamicCache`` mutates in place, so branch isolation and
order-invariance are load-bearing correctness properties, not incidental --
see the ``_FakeCacheBackend``-driven tests in the paired test file for a
canary that proves a missing crop *would* be caught.

The prefill is the *real* multimodal prefix, not a text-only placeholder:
``run`` obtains ``pixel_values``/``image_grid_thw``/executed prompt ids from
``HFBackendSession._materialize_native_inputs`` (the same exact native
-materialization seam the sibling scorers in this investigation use), never
an independently reconstructed or ``None`` grid; :func:`prefill_context`
fails fast rather than silently prefilling a text-only sequence if that
materialization is missing.

Every log-softmax in this module is taken over the model's complete,
unfiltered vocabulary dimension (never a pre-filtered subset); every raw and
policy view records a :class:`VocabAttestation` so a mis-sized or
accidentally-subset distribution fails fast instead of silently normalizing
over the wrong domain.

Likelihood channels are never mixed: ``raw_model_logprob`` is the unmodified
fp32 lm-head channel; ``auxiliary_policy_scores`` are repetition-penalty
(``1.0`` / ``1.10``) adjusted policy readouts derived from that *same* raw
forward and are explicitly labelled as policy scores, never model
likelihoods.

``run`` treats ``run_cache_parity_gate`` as mandatory: before any candidate
is scored, the cache-branch path's raw logits, repetition-penalty-processor
output, coordinate behavior, and reverse-order isolation are checked against
an independent ``use_cache=False`` reforward on a live model. A skipped gate
aborts; a failed admission selects the full-reforward backend globally.

No GPU launch happens as a side effect of importing or unit-testing this
module. ``run``/``main`` open a real HF backend session and are exercised in
this delivery only as reviewed, source-grounded code -- not as an executed
GPU smoke -- see ``RUNTIME_EXECUTION_STATUS`` below. In particular, the
mandatory parity gate above has not itself been executed against a live
model this session; it is implemented and control-flow-tested against fakes
(see the paired test file), but its real numerical pass/fail is unverified
until a GPU smoke actually runs it.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib import metadata as importlib_metadata
import importlib.util
import json
import math
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NoReturn, Protocol, TypeGuard, cast

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import (  # noqa: E402
    build_sorted_owner_basin_candidates as candidate_builder,
)


# ---------------------------------------------------------------------------
# Schema / contract constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "landscape_scores.v1"
RECEIPT_SCHEMA_VERSION = "landscape_scores_receipt.v1"
DECISION_RULES_SCHEMA_VERSION = "landscape_decision_rules.v1"
CANDIDATE_SCHEMA_VERSION = candidate_builder.SCHEMA_VERSION
V2_RULES_SCHEMA_VERSION = "sorted_owner_basin_landscape_rules.v2"
UNIT_ID = "2026-08-01-sorted-owner-basin-landscape-and-repair"

OUTPUT_JSONL_NAME = "landscape-scores.jsonl"
RECEIPT_NAME = "landscape-scores-receipt.json"

#: The two established decode-policy strata this unit ever mixes into an
#: auxiliary policy view. Fixed by ``unit.md``; not rules-configurable.
REPETITION_PENALTY_STRATA: tuple[float, ...] = (1.0, 1.10)

CANDIDATE_KINDS = frozenset(
    {"target_anchor", "covered_owner", "background", "scan", "part", "whole", "merged"}
)
CANDIDATE_ROLES = frozenset({"target", "registered_foil"})
REVIEW_STATUSES = frozenset({"reviewed", "unreviewed"})
#: Mirrors the landed core v2's ``RegisteredBasinRole.identity_kind`` literal
#: exactly (verified by reading ``sorted_owner_basin_landscape.py``).
#: "registered_geometry" is the owner-neutral identity used only by
#: background/scan foils; target/covered-owner candidates always require
#: "reviewed_physical_owner".
IDENTITY_KINDS = frozenset({"reviewed_physical_owner", "registered_geometry"})
#: Candidate kinds the core restricts to owner-anchored identity (target and
#: covered-owner banks cannot use owner-neutral registered geometry).
OWNER_ANCHORED_CANDIDATE_KINDS = frozenset({"target_anchor", "covered_owner"})
#: Candidate kinds the core allows to (optionally) use owner-neutral
#: registered geometry; every other kind must stay reviewed_physical_owner.
REGISTERED_GEOMETRY_ELIGIBLE_CANDIDATE_KINDS = frozenset({"background", "scan"})
CONTEXT_IDS = frozenset(
    {"P_pre", "P_post", "root", "natural_stop", "reference_scan_position"}
)
#: These contexts are always anchored to a specific sealed Task-1 native row;
#: root/natural_stop are sentinel contexts constructed without one.
CONTEXTS_REQUIRING_SOURCE_ROW = frozenset(
    {"P_pre", "P_post", "reference_scan_position"}
)

REQUEST_KINDS = frozenset({"complete_box", "dense_scan", "free_coordinate_tree_root"})
COORD_SLOTS: tuple[str, ...] = ("x1", "y1", "x2", "y2")

FREE_SEARCH_BUDGET: Mapping[str, Any] = {
    "x1_branch_budget": 64,
    "y1_branch_budget_per_x1": 32,
    "extent_branch_budget_per_anchor": 16,
    "spatial_diversification": {
        "algorithm": "deterministic_farthest_point_xy_anchor_selection",
        "tie_break": "ascending_x1_then_y1",
        "minimum_center_distance_bins": 24,
    },
}

#: Candidate rows must never carry re-tokenizable text; only literal token ids
#: are an admissible self-prefix or row source (see unit.md "Context
#: Construction": decoded text is never re-tokenized to reconstruct a state).
FORBIDDEN_TEXT_KEYS = frozenset(
    {"prefix_text", "generated_text", "prefix_chat_text", "chat_text", "decoded_text"}
)

#: Reported verbatim in the execution receipt: this module has not been run
#: against a live model/GPU in this delivery. Everything above this constant
#: is unit-tested with fakes/mocks; the driver below is source-grounded
#: (Qwen3-VL ``get_rope_index``/continuation contract read from the installed
#: transformers) but unexecuted here.
RUNTIME_EXECUTION_STATUS = "implemented_not_gpu_smoke_tested_this_session"

LANDSCAPE_SURFACES: tuple[str, str] = (
    "canonical_description_free",
    "restricted_gt_target",
)
LIVE_SCORING_PENDING_SEAL_STATUS = "live_model_scoring_completed_artifacts_pending_seal"
LIVE_SCORING_SEALED_STATUS = "live_model_scoring_completed_artifacts_sealed"
TEST_FIXTURE_SCORING_STATUS = "test_fixture_scorer_shaped_output"
NON_C_SMOKE_FREEZE_SCHEMA_VERSION = (
    "sorted_owner_basin_non_c_smoke_freeze_receipt.v1"
)
SENTINEL_STRUCTURAL_STATUS = "sealed_non_c_smoke"
DRAFT_CONTROL_STRUCTURAL_STATUS = "draft_pre_smoke"

DEFAULT_PURE_CORE_PATH = (
    REPO_ROOT / "scripts" / "research" / "sorted_owner_basin_landscape.py"
)
#: Real symbols exported by the landed core v2 (verified by reading
#: ``sorted_owner_basin_landscape.py`` directly -- not guessed). There is no
#: ``recompute_candidate_proposal_digest`` on the real core; that expectation
#: was invented before the core landed and is retired. Production requires
#: every one of these to be present and callable/accessible (see
#: ``require_production_pure_core``); fixture-only unit tests may still probe
#: absent/incompatible cores permissively via ``pure_core_status``.
EXPECTED_PURE_CORE_API: tuple[str, ...] = (
    "validate_rule_mapping",
    "full_vocabulary_id_digest",
    "FullVocabularyAttestation",
    "PolicyRuntimeIdentity",
    "RegisteredBasinRole",
    "CoordinateBin",
    "CoordinateBox",
    "ConditionalY1ScoreReceipt",
    "ConditionalY1CompletenessAttestation",
    "attest_complete_conditional_y1_scores",
)

#: The default cache-vs-reforward numerical parity semantics: exact production
#: callers retain the historical ``torch.allclose(cache, ground_truth,
#: atol=.., rtol=..)`` full-vocabulary gate. A caller may explicitly request
#: approximate admission through the CLI, but that does not alter these
#: thresholds or relabel a failed full-vocabulary comparison as strict parity.
CACHE_PARITY_ATOL = 1e-6
CACHE_PARITY_RTOL = 1e-5
BATCH_REFORWARD_COORD_LOGPROB_MAX_ABS_DIFF = 1e-3
KV_CACHE_SCORING_BACKEND = "kv_cache"
FULL_REFORWARD_SCORING_BACKEND = "full_reforward_uncached"
PARITY_FAILURE_FALLBACK_TRIGGER = "mandatory_cache_parity_failed"
STRICT_CACHE_ADMISSION_MODE = "strict_all_vocab"
RELAXED_CACHE_ADMISSION_MODE = "relaxed_coordinate_behavior"
UNCACHED_CACHE_ADMISSION_MODE = "uncached_fallback"
DECISION_BEARING_SCORE_USE = "decision_bearing"
PROBE_ONLY_SCORE_USE = "probe_only_not_decision_bearing"

#: Expected transformer decoder layer count for the live Qwen3-VL-2B
#: checkpoint this unit targets; used only as a diagnostic/gate on the real
#: backend's cache layer count, never on fakes unless a test explicitly
#: supplies it.
EXPECTED_LIVE_QWEN_DECODER_LAYER_COUNT = 28


class LandscapeScoringError(RuntimeError):
    """A verification gate failed; no partial decision-bearing score is emitted."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(
            f"{message} | context: {json.dumps(context, sort_keys=True, default=str)}"
        )
        self.context = context


def _fail(message: str, **context: Any) -> NoReturn:
    raise LandscapeScoringError(message, **context)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_file_streamed(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _matches_stratum(
    value: object, *, target: float | None = None
) -> TypeGuard[int | float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    value = float(value)
    if target is not None:
        return math.isclose(value, target, abs_tol=1e-9)
    return any(
        math.isclose(value, stratum, abs_tol=1e-9)
        for stratum in REPETITION_PENALTY_STRATA
    )


def _policy_view_key(penalty: float) -> str:
    return f"rp_{float(penalty):.2f}"


def _read_json(path: Path) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        _fail("expected a JSON object", path=str(resolved))
    return payload


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    rows: list[Mapping[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                _fail(
                    "jsonl row is not a JSON object",
                    path=str(resolved),
                    line=line_number,
                )
            rows.append(payload)
    return rows


# ---------------------------------------------------------------------------
# Sealed Task 0/1 ledgers
# ---------------------------------------------------------------------------

OWNER_STATUSES = frozenset({"gt", "aux", "unresolved"})
_OWNER_ID_PREFIX = {"gt": "gt:", "aux": "aux:", "unresolved": "unresolved:"}


@dataclass(frozen=True)
class OwnerLedgerEntry:
    diagnostic_owner_id: str
    status: str
    gt_owner_id: str | None
    image_id: str


def load_owner_ledger(path: Path) -> dict[str, OwnerLedgerEntry]:
    entries: dict[str, OwnerLedgerEntry] = {}
    for row in _read_jsonl(path):
        diagnostic_owner_id = str(row.get("diagnostic_owner_id", ""))
        status = str(row.get("status", ""))
        if status not in OWNER_STATUSES:
            _fail(
                "owner ledger row has an unsupported status",
                diagnostic_owner_id=diagnostic_owner_id,
                status=status,
            )
        if not diagnostic_owner_id.startswith(_OWNER_ID_PREFIX[status]):
            _fail(
                "owner ledger diagnostic_owner_id prefix does not match its declared status",
                diagnostic_owner_id=diagnostic_owner_id,
                status=status,
            )
        if diagnostic_owner_id in entries:
            _fail(
                "duplicate diagnostic_owner_id in owner ledger",
                diagnostic_owner_id=diagnostic_owner_id,
            )
        image_id = str(row.get("image_id", ""))
        if not image_id:
            _fail(
                "owner ledger row requires image_id",
                diagnostic_owner_id=diagnostic_owner_id,
            )
        gt_owner_id = row.get("gt_owner_id")
        entries[diagnostic_owner_id] = OwnerLedgerEntry(
            diagnostic_owner_id=diagnostic_owner_id,
            status=status,
            gt_owner_id=(str(gt_owner_id) if gt_owner_id else None),
            image_id=image_id,
        )
    return entries


@dataclass(frozen=True)
class PredictionRowLedgerEntry:
    pred_row_id: str
    image_id: str
    repetition_penalty: float


def load_prediction_row_ledger(path: Path) -> dict[str, PredictionRowLedgerEntry]:
    entries: dict[str, PredictionRowLedgerEntry] = {}
    for row in _read_jsonl(path):
        pred_row_id = str(row.get("pred_row_id", ""))
        if not pred_row_id:
            _fail("prediction row ledger row requires pred_row_id")
        if pred_row_id in entries:
            _fail(
                "duplicate pred_row_id in prediction row ledger",
                pred_row_id=pred_row_id,
            )
        repetition_penalty = row.get("repetition_penalty")
        if not _matches_stratum(repetition_penalty):
            _fail(
                "prediction row ledger repetition_penalty is not an established stratum",
                pred_row_id=pred_row_id,
                repetition_penalty=repetition_penalty,
            )
        image_id = str(row.get("image_id", ""))
        if not image_id:
            _fail(
                "prediction row ledger row requires image_id", pred_row_id=pred_row_id
            )
        entries[pred_row_id] = PredictionRowLedgerEntry(
            pred_row_id=pred_row_id,
            image_id=image_id,
            repetition_penalty=float(repetition_penalty),
        )
    return entries


# ---------------------------------------------------------------------------
# landscape-decision-rules.json
# ---------------------------------------------------------------------------

_REQUIRED_SCHEMA_TOKEN_KEYS: tuple[str, ...] = (
    "object_ref_start_token_id",
    "object_ref_end_token_id",
    "box_start_token_id",
    "box_end_token_id",
    "coordinate_token_id_start",
    "coordinate_token_id_end_exclusive",
)


@dataclass(frozen=True)
class DecisionRules:
    rules_digest: str
    schema_tokens: Mapping[str, int]
    owner_canonical_description: Mapping[str, tuple[int, ...]]
    foil_set_digests: Mapping[str, str]
    model_vocab_size: int
    numeric_tolerance: float
    contract_version: str = DECISION_RULES_SCHEMA_VERSION
    rules_file_sha256: str | None = None
    materializer_rules: Any | None = None
    token_registry: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    free_search_budget: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    contract_mode: str = "test_fixture"
    structural_status: str = "test_fixture"
    non_c_smoke_freeze_binding: Mapping[str, Any] = dataclasses.field(
        default_factory=dict
    )


def _load_v2_decision_rules(*, payload: Mapping[str, Any], path: Path) -> DecisionRules:
    """Parse the candidate-builder-v2 rule document through its owning parser."""

    try:
        parsed = candidate_builder._parse_rules(payload)  # noqa: SLF001
    except (TypeError, ValueError) as exc:
        _fail(
            "landscape-decision-rules.json is not a complete candidate-builder-v2 contract",
            path=str(path),
            error=str(exc),
        )
    if parsed.free_search_budget != FREE_SEARCH_BUDGET:
        _fail(
            "candidate-builder-v2 free coordinate-tree budget differs from the frozen scorer contract",
            path=str(path),
            observed=parsed.free_search_budget,
            expected=FREE_SEARCH_BUDGET,
        )
    descriptions = {
        owner_id: tuple(int(token_id) for token_id in value["token_ids"])
        for owner_id, value in parsed.canonical_descriptions.items()
    }
    if parsed.landscape.contract_mode == "production" and "semantic_core" not in payload:
        _fail(
            "production decision rules must carry the landed semantic_core digest",
            path=str(path),
        )
    structural_status = str(payload.get("structural_status", ""))
    if parsed.landscape.contract_mode == "production" and structural_status not in {
        DRAFT_CONTROL_STRUCTURAL_STATUS,
        SENTINEL_STRUCTURAL_STATUS,
    }:
        _fail(
            "production decision rules must declare draft-control or sealed-sentinel structural status",
            path=str(path),
            observed=structural_status,
        )
    freeze_binding_value = payload.get("non_c_smoke_freeze_receipt")
    if structural_status == SENTINEL_STRUCTURAL_STATUS:
        if not isinstance(freeze_binding_value, Mapping):
            _fail("sealed sentinel decision rules lack the non-C smoke freeze binding")
        freeze_binding = dict(freeze_binding_value)
    else:
        if freeze_binding_value is not None:
            _fail("draft control decision rules must not bind a non-C smoke freeze receipt")
        freeze_binding = {}
    return DecisionRules(
        rules_digest=parsed.landscape.rule_digest,
        schema_tokens={
            **parsed.token_registry["schema_tokens"],
            "coordinate_token_id_start": parsed.token_registry[
                "coordinate_bin_to_token_id"
            ]["token_id_start"],
            "coordinate_token_id_end_exclusive": parsed.token_registry[
                "coordinate_bin_to_token_id"
            ]["token_id_end_exclusive"],
        },
        owner_canonical_description=descriptions,
        foil_set_digests={parsed.foil_set_id: parsed.foil_set_sha256},
        model_vocab_size=int(parsed.token_registry["model_vocab_size"]),
        numeric_tolerance=CACHE_PARITY_ATOL,
        contract_version=CANDIDATE_SCHEMA_VERSION,
        rules_file_sha256=sha256_file(path),
        materializer_rules=parsed,
        token_registry=parsed.token_registry,
        free_search_budget=parsed.free_search_budget,
        contract_mode=parsed.landscape.contract_mode,
        structural_status=(
            structural_status
            if structural_status
            else "test_fixture"
        ),
        non_c_smoke_freeze_binding=freeze_binding,
    )


def load_decision_rules(path: Path) -> DecisionRules:
    resolved = path.expanduser().resolve(strict=True)
    payload = _read_json(resolved)
    if payload.get("schema_version") == V2_RULES_SCHEMA_VERSION:
        return _load_v2_decision_rules(payload=payload, path=resolved)
    if payload.get("schema_version") != DECISION_RULES_SCHEMA_VERSION:
        _fail(
            "landscape-decision-rules.json has an unexpected schema_version",
            path=str(resolved),
            observed=payload.get("schema_version"),
            expected=DECISION_RULES_SCHEMA_VERSION,
        )
    declared_digest = payload.get("rules_digest")
    if not isinstance(declared_digest, str) or not declared_digest:
        _fail(
            "landscape-decision-rules.json is missing rules_digest", path=str(resolved)
        )
    content_without_digest = {
        key: value for key, value in payload.items() if key != "rules_digest"
    }
    recomputed_digest = sha256_json(content_without_digest)
    if recomputed_digest != declared_digest:
        _fail(
            "landscape-decision-rules.json rules_digest is stale or was tampered with after freeze",
            path=str(resolved),
            declared=declared_digest,
            recomputed=recomputed_digest,
        )

    schema_tokens_raw = payload.get("schema_tokens")
    if not isinstance(schema_tokens_raw, Mapping):
        _fail(
            "landscape-decision-rules.json is missing schema_tokens", path=str(resolved)
        )
    schema_tokens: dict[str, int] = {}
    for key in _REQUIRED_SCHEMA_TOKEN_KEYS:
        value = schema_tokens_raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            _fail(
                f"landscape-decision-rules.json schema_tokens.{key} must be an integer",
                path=str(resolved),
            )
        schema_tokens[key] = int(value)
    if (
        not schema_tokens["coordinate_token_id_start"]
        < schema_tokens["coordinate_token_id_end_exclusive"]
    ):
        _fail(
            "landscape-decision-rules.json coordinate token bounds are not a valid half-open range",
            path=str(resolved),
        )
    wrapper_ids = [schema_tokens[key] for key in _REQUIRED_SCHEMA_TOKEN_KEYS[:4]]
    if len(set(wrapper_ids)) != 4:
        _fail(
            "landscape-decision-rules.json wrapper token ids are not pairwise distinct",
            path=str(resolved),
        )

    owner_map_raw = payload.get("owner_canonical_description")
    if not isinstance(owner_map_raw, Mapping) or not owner_map_raw:
        _fail(
            "landscape-decision-rules.json requires a non-empty owner_canonical_description",
            path=str(resolved),
        )
    owner_map: dict[str, tuple[int, ...]] = {}
    for owner_id, entry in owner_map_raw.items():
        token_ids = entry.get("token_ids") if isinstance(entry, Mapping) else None
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(isinstance(v, bool) or not isinstance(v, int) for v in token_ids)
        ):
            _fail(
                "landscape-decision-rules.json owner_canonical_description entry requires non-empty integer token_ids",
                path=str(resolved),
                owner_id=str(owner_id),
            )
        owner_map[str(owner_id)] = tuple(int(v) for v in token_ids)

    foil_set_digests_raw = payload.get("foil_set_digests")
    if not isinstance(foil_set_digests_raw, Mapping) or not foil_set_digests_raw:
        _fail(
            "landscape-decision-rules.json requires a non-empty foil_set_digests",
            path=str(resolved),
        )
    foil_set_digests = {str(k): str(v) for k, v in foil_set_digests_raw.items()}

    model_vocab_size = payload.get("model_vocab_size")
    if (
        isinstance(model_vocab_size, bool)
        or not isinstance(model_vocab_size, int)
        or model_vocab_size <= 0
    ):
        _fail(
            "landscape-decision-rules.json model_vocab_size must be a positive integer",
            path=str(resolved),
        )

    tolerance = payload.get("numeric_tolerance")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or tolerance <= 0
    ):
        _fail(
            "landscape-decision-rules.json numeric_tolerance must be a positive number",
            path=str(resolved),
        )

    return DecisionRules(
        rules_digest=declared_digest,
        schema_tokens=schema_tokens,
        owner_canonical_description=owner_map,
        foil_set_digests=foil_set_digests,
        model_vocab_size=int(model_vocab_size),
        numeric_tolerance=float(tolerance),
    )


# ---------------------------------------------------------------------------
# Candidate / foil rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateRow:
    candidate_id: str
    diagnostic_owner_id: str
    image_id: str
    context_id: str
    role: str
    physical_owner_hint: str | None
    review_status: str
    candidate_kind: str
    foil_set_id: str
    identity_kind: str
    native_repetition_penalty_stratum: float
    source_pred_row_id: str | None
    prompt_prefix_token_count: int
    prefix_token_ids: tuple[int, ...]
    prefix_token_ids_sha256: str
    proposal_digest: str
    basin_id: str | None
    request_kind: str
    coord_token_ids: tuple[int, int, int, int] | None
    fixed_coord_token_ids: tuple[int, ...]
    scan_slot: str | None
    record_type: str = "legacy_candidate_request"
    gt_owner_id: str | None = None
    image_identity: str | None = None
    coordinate_bin_values: tuple[int, ...] = ()
    complete_conditional_y1: tuple[Mapping[str, Any], ...] = ()
    completeness_plan_digest: str | None = None
    conditional_y1_plan_sha256: str | None = None
    raw_payload: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def _require_int_list(value: Any, *, count: int | None, context: str) -> list[int]:
    if not isinstance(value, list) or (count is not None and len(value) != count):
        _fail(
            f"{context} must be a list of {count if count is not None else 'zero or more'} integers"
        )
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        _fail(f"{context} must contain only integers")
    return [int(item) for item in value]


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{context} must be a mapping")
    return value


def _validate_v2_core_request_attestation(
    payload: Mapping[str, Any], *, rules: DecisionRules, prefix: Sequence[int]
) -> None:
    candidate_id = str(payload.get("candidate_id", payload.get("request_id", "")))
    attestation = payload.get("core_request_attestation")
    if not isinstance(attestation, Mapping):
        _fail(
            "candidate-builder-v2 row lacks core_request_attestation",
            candidate_id=candidate_id,
        )
    attestation_payload = {
        key: value for key, value in attestation.items() if key != "attestation_sha256"
    }
    if attestation.get("attestation_sha256") != sha256_json(attestation_payload):
        _fail(
            "candidate-builder-v2 core_request_attestation digest is stale",
            candidate_id=candidate_id,
        )
    if attestation.get("core_rule_digest") != rules.rules_digest:
        _fail(
            "candidate-builder-v2 row is bound to a different core rule digest",
            candidate_id=candidate_id,
        )
    if attestation.get("landscape_decision_rules_sha256") != rules.rules_file_sha256:
        _fail(
            "candidate-builder-v2 row is bound to a different decision-rules file",
            candidate_id=candidate_id,
        )
    if attestation.get("prefix_token_ids_sha256") != sha256_json(list(prefix)):
        _fail(
            "candidate-builder-v2 request attestation does not bind its root prefix",
            candidate_id=candidate_id,
        )
    expected_attestation_bindings = {
        "coordinate_bin_token_ids_sha256": rules.token_registry[
            "coordinate_bin_to_token_id"
        ]["coordinate_bin_token_ids_sha256"],
        "foil_set_sha256": payload.get("foil_set_sha256"),
        "runtime_vocabulary_receipt_sha256": payload.get(
            "runtime_vocabulary_receipt_sha256"
        ),
        "token_registry_sha256": rules.token_registry["registry_sha256"],
        "vocabulary_attestation": payload.get("vocabulary_attestation"),
    }
    for field, expected in expected_attestation_bindings.items():
        if attestation.get(field) != expected:
            _fail(
                "candidate-builder-v2 request attestation has a stale binding",
                candidate_id=candidate_id,
                field=field,
            )

    request_kind = payload.get("request_kind")
    if request_kind == "complete_box":
        request_payload = {
            "request_kind": "complete_box",
            "coordinate_bin_values": payload.get("coordinate_bin_values"),
            "coord_token_ids": payload.get("coord_token_ids"),
        }
    elif request_kind == "dense_scan":
        request_payload = {
            "request_kind": "dense_scan",
            "fixed_coord_bin_values": payload.get("fixed_coord_bin_values"),
            "fixed_coord_token_ids": payload.get("fixed_coord_token_ids"),
            "scan_slot": payload.get("scan_slot"),
        }
    elif request_kind == "free_coordinate_tree_root":
        request_payload = {
            "request_kind": "free_coordinate_tree_root",
            "scan_slot": "x1",
            "budget": dict(rules.free_search_budget),
            "dynamic_traversal_owner": "runtime_landscape_scorer_free_tree_surface",
        }
    else:
        _fail(
            "candidate-builder-v2 row has an unsupported request kind",
            candidate_id=candidate_id,
        )
    if attestation.get("request_payload_sha256") != sha256_json(request_payload):
        _fail(
            "candidate-builder-v2 request payload digest is stale",
            candidate_id=candidate_id,
        )


def _load_v2_candidate_rows(
    payloads: Sequence[Mapping[str, Any]], *, rules: DecisionRules | None
) -> list[CandidateRow]:
    if rules is None or rules.contract_version != CANDIDATE_SCHEMA_VERSION:
        _fail(
            "candidate-builder-v2 rows require their complete v2 decision-rules contract"
        )
    materializer = rules.materializer_rules
    if materializer is None:
        _fail("candidate-builder-v2 rules lack their parsed materializer contract")
    lo = rules.schema_tokens["coordinate_token_id_start"]
    landscape = materializer.landscape
    coordinate_values = list(
        range(landscape.coordinate_min, landscape.coordinate_max + 1)
    )

    rows: list[CandidateRow] = []
    seen_ids: set[str] = set()
    for payload in payloads:
        record_type = str(payload.get("record_type", ""))
        if record_type not in {
            "free_coordinate_tree_root_request",
            "conditional_y1_score_plan",
            "complete_box_candidate",
        }:
            _fail(
                "candidate-builder-v2 row has an unsupported record_type",
                record_type=record_type,
            )
        candidate_id = str(payload.get("candidate_id", ""))
        if not candidate_id or candidate_id in seen_ids:
            _fail(
                "candidate-builder-v2 candidate_id is empty or duplicated",
                candidate_id=candidate_id,
            )
        seen_ids.add(candidate_id)

        prefix = tuple(
            _require_int_list(
                payload.get("prefix_token_ids"),
                count=None,
                context=f"{candidate_id}.prefix_token_ids",
            )
        )
        root_prefix = tuple(
            _require_int_list(
                payload.get("root_prefix_token_ids"),
                count=None,
                context=f"{candidate_id}.root_prefix_token_ids",
            )
        )
        if not prefix or prefix != root_prefix:
            _fail(
                "candidate-builder-v2 executable prefix must equal the root prefix through box_start",
                candidate_id=candidate_id,
            )
        prefix_sha = sha256_json(list(prefix))
        if (
            payload.get("prefix_token_ids_sha256") != prefix_sha
            or payload.get("root_prefix_token_ids_sha256") != prefix_sha
        ):
            _fail(
                "candidate-builder-v2 root-prefix digest is stale",
                candidate_id=candidate_id,
            )
        if prefix[-1] != rules.schema_tokens["box_start_token_id"]:
            _fail(
                "candidate-builder-v2 root prefix must end exactly at box_start",
                candidate_id=candidate_id,
            )
        canonical = _require_mapping(
            payload.get("canonical_description"),
            context=f"{candidate_id}.canonical_description",
        )
        forced = canonical.get("forced_row_prefix_through_box_start_token_ids")
        if (
            not isinstance(forced, list)
            or tuple(int(value) for value in forced) != prefix[-len(forced) :]
        ):
            _fail(
                "candidate-builder-v2 root prefix does not end in its frozen description wrapper",
                candidate_id=candidate_id,
            )
        if canonical.get("forced_row_prefix_through_box_start_sha256") != sha256_json(
            forced
        ):
            _fail(
                "candidate-builder-v2 description wrapper digest is stale",
                candidate_id=candidate_id,
            )
        if any(
            token_id < 0 or token_id >= rules.model_vocab_size for token_id in prefix
        ):
            _fail(
                "candidate-builder-v2 root prefix token is outside the model vocabulary",
                candidate_id=candidate_id,
            )
        prompt_count = payload.get("prompt_prefix_token_count")
        if (
            isinstance(prompt_count, bool)
            or not isinstance(prompt_count, int)
            or not 0 <= prompt_count <= len(prefix)
        ):
            _fail(
                "candidate-builder-v2 prompt_prefix_token_count is invalid",
                candidate_id=candidate_id,
            )
        if (
            payload.get("core_rule_digest") != rules.rules_digest
            or payload.get("landscape_decision_rules_sha256") != rules.rules_file_sha256
        ):
            _fail(
                "candidate-builder-v2 row is stale against its decision rules",
                candidate_id=candidate_id,
            )
        native_stratum_raw = payload.get("native_repetition_penalty_stratum")
        if (
            isinstance(native_stratum_raw, bool)
            or not isinstance(native_stratum_raw, (int, float))
            or not _matches_stratum(native_stratum_raw)
        ):
            _fail(
                "candidate-builder-v2 row has an unsupported native repetition-penalty stratum",
                candidate_id=candidate_id,
            )
        native_stratum = float(native_stratum_raw)
        foil_set_id = str(payload.get("foil_set_id", ""))
        if rules.foil_set_digests.get(foil_set_id) != payload.get("foil_set_sha256"):
            _fail(
                "candidate-builder-v2 row has a stale foil-set binding",
                candidate_id=candidate_id,
            )
        if payload.get("coordinate_space") != materializer.coordinate_space:
            _fail(
                "candidate-builder-v2 row has a stale coordinate-space binding",
                candidate_id=candidate_id,
            )
        if payload.get("vocabulary_attestation") != materializer.vocabulary_attestation:
            _fail(
                "candidate-builder-v2 row has a stale vocabulary-attestation binding",
                candidate_id=candidate_id,
            )
        if payload.get("coordinate_token_registry_sha256") != rules.token_registry.get(
            "registry_sha256"
        ):
            _fail(
                "candidate-builder-v2 row has a stale coordinate-token registry binding",
                candidate_id=candidate_id,
            )
        expected_coordinate_ids_sha = rules.token_registry[
            "coordinate_bin_to_token_id"
        ]["coordinate_bin_token_ids_sha256"]
        if (
            payload.get("coordinate_bin_token_ids_sha256")
            != expected_coordinate_ids_sha
        ):
            _fail(
                "candidate-builder-v2 row has a stale coordinate-bin token binding",
                candidate_id=candidate_id,
            )
        runtime_vocabulary_receipt = payload.get("runtime_vocabulary_receipt")
        if not isinstance(runtime_vocabulary_receipt, Mapping) or payload.get(
            "runtime_vocabulary_receipt_sha256"
        ) != sha256_json(dict(runtime_vocabulary_receipt)):
            _fail(
                "candidate-builder-v2 runtime vocabulary receipt is missing or stale",
                candidate_id=candidate_id,
            )
        lineage = payload.get("source_review_foreign_key_lineage")
        if not isinstance(lineage, Mapping):
            _fail(
                "candidate-builder-v2 source lineage is missing",
                candidate_id=candidate_id,
            )
        lineage_payload = {
            key: value for key, value in lineage.items() if key != "lineage_sha256"
        }
        if lineage.get("lineage_sha256") != sha256_json(lineage_payload):
            _fail(
                "candidate-builder-v2 source lineage digest is stale",
                candidate_id=candidate_id,
            )
        _validate_v2_core_request_attestation(payload, rules=rules, prefix=prefix)

        coordinate_bins: tuple[int, ...] = ()
        coord_token_ids: tuple[int, int, int, int] | None = None
        fixed_coord_token_ids: tuple[int, ...] = ()
        scan_slot: str | None = None
        request_kind = str(payload.get("request_kind", ""))
        if record_type == "complete_box_candidate":
            bins = _require_int_list(
                payload.get("coordinate_bin_values"),
                count=4,
                context=f"{candidate_id}.coordinate_bin_values",
            )
            tokens = _require_int_list(
                payload.get("coordinate_token_ids"),
                count=4,
                context=f"{candidate_id}.coordinate_token_ids",
            )
            aliases = _require_int_list(
                payload.get("coord_token_ids"),
                count=4,
                context=f"{candidate_id}.coord_token_ids",
            )
            if tokens != aliases or tokens != [lo + value for value in bins]:
                _fail(
                    "candidate-builder-v2 complete-box coordinate registry binding is invalid",
                    candidate_id=candidate_id,
                )
            if payload.get("coord_token_ids_sha256") != sha256_json(tokens):
                _fail(
                    "candidate-builder-v2 complete-box coordinate-token digest is stale",
                    candidate_id=candidate_id,
                )
            if any(value not in coordinate_values for value in bins):
                _fail(
                    "candidate-builder-v2 complete-box coordinate is outside the declared landscape",
                    candidate_id=candidate_id,
                )
            if not (bins[2] > bins[0] and bins[3] > bins[1]):
                _fail(
                    "candidate-builder-v2 complete box is geometrically invalid",
                    candidate_id=candidate_id,
                )
            coordinate_bins = tuple(bins)
            coord_token_ids = (tokens[0], tokens[1], tokens[2], tokens[3])
        elif record_type == "conditional_y1_score_plan":
            if payload.get("fixed_coord_tokens_materialized_in_prefix") is not False:
                _fail(
                    "conditional-y1 fixed x1 must not already be materialized in the root prefix",
                    candidate_id=candidate_id,
                )
            fixed_bins = _require_int_list(
                payload.get("fixed_coord_bin_values"),
                count=1,
                context=f"{candidate_id}.fixed_coord_bin_values",
            )
            fixed_tokens = _require_int_list(
                payload.get("fixed_coord_token_ids"),
                count=1,
                context=f"{candidate_id}.fixed_coord_token_ids",
            )
            if (
                fixed_tokens != [lo + fixed_bins[0]]
                or payload.get("x1") != fixed_bins[0]
                or payload.get("scan_slot") != "y1"
            ):
                _fail(
                    "conditional-y1 plan has an invalid fixed x1 binding",
                    candidate_id=candidate_id,
                )
            expected_after_fixed = [*prefix, *fixed_tokens]
            if (
                payload.get("expected_prefix_after_fixed_token_ids")
                != expected_after_fixed
                or payload.get("expected_prefix_after_fixed_token_ids_sha256")
                != sha256_json(expected_after_fixed)
                or payload.get("expected_prefix_after_fixed_is_attestation_only")
                is not True
            ):
                _fail(
                    "conditional-y1 plan does not attest one exact x1 append",
                    candidate_id=candidate_id,
                )
            entries = payload.get("complete_conditional_y1")
            if not isinstance(entries, list) or len(entries) != len(coordinate_values):
                _fail(
                    "conditional-y1 plan must enumerate every coordinate bin",
                    candidate_id=candidate_id,
                )
            if [
                entry.get("y1") for entry in entries if isinstance(entry, Mapping)
            ] != coordinate_values:
                _fail(
                    "conditional-y1 plan bins must be complete, ordered, and unique",
                    candidate_id=candidate_id,
                )
            if (
                entries[-1].get("can_form_valid_box") is not False
                or entries[-1].get("invalid_box_reason") != "no_later_representable_y2"
            ):
                _fail(
                    "conditional-y1 terminal invalid-box audit is missing",
                    candidate_id=candidate_id,
                )
            plan_payload = {
                "canonical_description_id": canonical["description_id"],
                "context_id": payload.get("context_id"),
                "coordinate_space": payload.get("coordinate_space"),
                "diagnostic_owner_id": payload.get("diagnostic_owner_id"),
                "gt_owner_id": payload.get("gt_owner_id"),
                "prefix_token_ids_sha256": prefix_sha,
                "prompt_prefix_token_count": prompt_count,
                "fixed_coord_bin_values": fixed_bins,
                "fixed_coord_token_ids": fixed_tokens,
                "scan_slot": "y1",
                "source_review_foreign_key_lineage": {
                    key: value
                    for key, value in payload.get(
                        "source_review_foreign_key_lineage", {}
                    ).items()
                    if key != "lineage_sha256"
                },
                "x1": fixed_bins[0],
            }
            expected_plan_sha = sha256_json(
                {
                    **plan_payload,
                    "complete_conditional_y1": entries,
                    "core_rule_digest": rules.rules_digest,
                    "landscape_decision_rules_sha256": rules.rules_file_sha256,
                    "runtime_vocabulary_receipt_sha256": payload.get(
                        "runtime_vocabulary_receipt_sha256"
                    ),
                    "token_registry_sha256": rules.token_registry["registry_sha256"],
                    "vocabulary_attestation": payload.get("vocabulary_attestation"),
                }
            )
            if payload.get("conditional_y1_plan_sha256") != expected_plan_sha:
                _fail("conditional-y1 plan digest is stale", candidate_id=candidate_id)
            fixed_coord_token_ids = tuple(fixed_tokens)
            coordinate_bins = tuple(fixed_bins)
            scan_slot = "y1"
        else:
            if (
                request_kind != "free_coordinate_tree_root"
                or payload.get("scan_slot") != "x1"
            ):
                _fail(
                    "free coordinate-tree root request is malformed",
                    candidate_id=candidate_id,
                )
            if (
                payload.get("fixed_coord_token_ids") != []
                or payload.get("fixed_coord_tokens_materialized_in_prefix") is not False
            ):
                _fail(
                    "free coordinate-tree root must not pre-materialize a coordinate",
                    candidate_id=candidate_id,
                )
            seam = payload.get("dynamic_traversal_seam")
            if (
                not isinstance(seam, Mapping)
                or seam.get("budget") != FREE_SEARCH_BUDGET
            ):
                _fail(
                    "free coordinate-tree root does not bind the frozen branch budget",
                    candidate_id=candidate_id,
                )
            scan_slot = "x1"

        rows.append(
            CandidateRow(
                candidate_id=candidate_id,
                diagnostic_owner_id=str(payload.get("diagnostic_owner_id", "")),
                image_id=str(payload.get("image_id", "")),
                context_id=str(payload.get("context_id", "")),
                role=str(payload.get("role", "surface_root")),
                physical_owner_hint=(
                    str(payload["physical_owner_hint"])
                    if payload.get("physical_owner_hint")
                    else None
                ),
                review_status=str(payload.get("review_status", "")),
                candidate_kind=str(
                    payload.get("candidate_kind", "free_coordinate_tree_root")
                ),
                foil_set_id=foil_set_id,
                identity_kind=str(
                    payload.get(
                        "identity_kind",
                        "reviewed_physical_owner"
                        if record_type == "conditional_y1_score_plan"
                        else "",
                    )
                ),
                native_repetition_penalty_stratum=native_stratum,
                source_pred_row_id=(
                    str(payload["source_pred_row_id"])
                    if payload.get("source_pred_row_id")
                    else None
                ),
                prompt_prefix_token_count=int(prompt_count),
                prefix_token_ids=prefix,
                prefix_token_ids_sha256=prefix_sha,
                proposal_digest=str(
                    payload.get(
                        "proposal_digest",
                        payload["core_request_attestation"]["attestation_sha256"],
                    )
                ),
                basin_id=None,
                request_kind=request_kind,
                coord_token_ids=coord_token_ids,
                fixed_coord_token_ids=fixed_coord_token_ids,
                scan_slot=scan_slot,
                record_type=record_type,
                gt_owner_id=str(payload.get("gt_owner_id", "")),
                image_identity=str(payload.get("image_identity", "")),
                coordinate_bin_values=coordinate_bins,
                complete_conditional_y1=tuple(
                    payload.get("complete_conditional_y1", ())
                ),
                completeness_plan_digest=(
                    str(payload["completeness_plan_digest"])
                    if payload.get("completeness_plan_digest")
                    else None
                ),
                conditional_y1_plan_sha256=(
                    str(payload["conditional_y1_plan_sha256"])
                    if payload.get("conditional_y1_plan_sha256")
                    else None
                ),
                raw_payload=dict(payload),
            )
        )
    return rows


def load_candidate_rows(
    path: Path, *, rules: DecisionRules | None = None
) -> list[CandidateRow]:
    payloads = _read_jsonl(path)
    if any(
        payload.get("schema_version") == CANDIDATE_SCHEMA_VERSION
        for payload in payloads
    ):
        if not all(
            payload.get("schema_version") == CANDIDATE_SCHEMA_VERSION
            for payload in payloads
        ):
            _fail(
                "candidate JSONL mixes candidate-builder-v2 rows with an incompatible schema"
            )
        return _load_v2_candidate_rows(payloads, rules=rules)
    rows: list[CandidateRow] = []
    seen_ids: set[str] = set()
    for payload in payloads:
        forbidden = FORBIDDEN_TEXT_KEYS & set(payload)
        if forbidden:
            _fail(
                "candidate row supplies re-tokenizable text instead of a literal token-id prefix",
                forbidden_keys=sorted(forbidden),
            )
        candidate_id = str(payload.get("candidate_id", ""))
        if not candidate_id:
            _fail("candidate row requires a non-empty candidate_id")
        if candidate_id in seen_ids:
            _fail("duplicate candidate_id", candidate_id=candidate_id)
        seen_ids.add(candidate_id)
        where = {"candidate_id": candidate_id}

        context_id = str(payload.get("context_id", ""))
        if context_id not in CONTEXT_IDS:
            _fail(
                "candidate row has an unsupported context_id",
                context_id=context_id,
                **where,
            )

        role = str(payload.get("role", ""))
        if role not in CANDIDATE_ROLES:
            _fail("candidate row has an unsupported role", role=role, **where)

        review_status = str(payload.get("review_status", ""))
        if review_status not in REVIEW_STATUSES:
            _fail(
                "candidate row has an unsupported review_status",
                review_status=review_status,
                **where,
            )

        candidate_kind = str(payload.get("candidate_kind", ""))
        if candidate_kind not in CANDIDATE_KINDS:
            _fail(
                "candidate row has an unsupported candidate_kind",
                candidate_kind=candidate_kind,
                **where,
            )

        foil_set_id = str(payload.get("foil_set_id", ""))
        if not foil_set_id:
            _fail("candidate row requires foil_set_id", **where)

        identity_kind = str(payload.get("identity_kind", ""))
        if identity_kind not in IDENTITY_KINDS:
            _fail(
                "candidate row has an unsupported identity_kind",
                identity_kind=identity_kind,
                **where,
            )

        stratum = payload.get("native_repetition_penalty_stratum")
        if not _matches_stratum(stratum):
            _fail(
                "candidate row native_repetition_penalty_stratum must be 1.0 or 1.10",
                stratum=stratum,
                **where,
            )

        source_pred_row_id = payload.get("source_pred_row_id")
        if context_id in CONTEXTS_REQUIRING_SOURCE_ROW and not source_pred_row_id:
            _fail(
                "candidate row context requires source_pred_row_id",
                context_id=context_id,
                **where,
            )

        prefix = payload.get("prefix_token_ids")
        if not isinstance(prefix, list) or not prefix:
            _fail(
                "candidate row requires a non-empty literal prefix_token_ids", **where
            )
        prefix_tokens = tuple(
            _require_int_list(
                prefix, count=None, context=f"{candidate_id}.prefix_token_ids"
            )
        )
        declared_prefix_sha = payload.get("prefix_token_ids_sha256")
        recomputed_prefix_sha = sha256_json(list(prefix_tokens))
        if declared_prefix_sha != recomputed_prefix_sha:
            _fail(
                "candidate row prefix_token_ids_sha256 does not match prefix_token_ids; "
                "exact prefix identity cannot be verified",
                declared=declared_prefix_sha,
                recomputed=recomputed_prefix_sha,
                **where,
            )

        prompt_prefix_token_count = payload.get("prompt_prefix_token_count")
        if (
            isinstance(prompt_prefix_token_count, bool)
            or not isinstance(prompt_prefix_token_count, int)
            or not 0 <= prompt_prefix_token_count <= len(prefix_tokens)
        ):
            _fail(
                "candidate row prompt_prefix_token_count must index within prefix_token_ids",
                **where,
            )

        proposal_digest = payload.get("proposal_digest")
        if not isinstance(proposal_digest, str) or not proposal_digest:
            _fail("candidate row requires a non-empty proposal_digest", **where)

        diagnostic_owner_id = str(payload.get("diagnostic_owner_id", ""))
        if not diagnostic_owner_id:
            _fail("candidate row requires diagnostic_owner_id", **where)
        image_id = str(payload.get("image_id", ""))
        if not image_id:
            _fail("candidate row requires image_id", **where)
        physical_owner_hint = payload.get("physical_owner_hint")
        basin_id = payload.get("basin_id")

        request_kind = str(payload.get("request_kind", ""))
        if request_kind not in REQUEST_KINDS:
            _fail(
                "candidate row has an unsupported request_kind",
                request_kind=request_kind,
                **where,
            )

        coord_token_ids: tuple[int, int, int, int] | None = None
        fixed_coord_token_ids: tuple[int, ...] = ()
        scan_slot: str | None = None
        if request_kind == "complete_box":
            coords = _require_int_list(
                payload.get("coord_token_ids"),
                count=4,
                context=f"{candidate_id}.coord_token_ids",
            )
            coord_token_ids = (coords[0], coords[1], coords[2], coords[3])
        else:
            fixed = _require_int_list(
                payload.get("fixed_coord_token_ids", []),
                count=None,
                context=f"{candidate_id}.fixed_coord_token_ids",
            )
            if len(fixed) > 3:
                _fail(
                    "dense-scan candidate fixed_coord_token_ids must have at most three entries",
                    **where,
                )
            fixed_coord_token_ids = tuple(fixed)
            scan_slot = str(payload.get("scan_slot", ""))
            expected_slot = COORD_SLOTS[len(fixed_coord_token_ids)]
            if scan_slot != expected_slot:
                _fail(
                    "dense-scan candidate scan_slot does not follow its fixed_coord_token_ids prefix",
                    scan_slot=scan_slot,
                    expected_slot=expected_slot,
                    **where,
                )

        rows.append(
            CandidateRow(
                candidate_id=candidate_id,
                diagnostic_owner_id=diagnostic_owner_id,
                image_id=image_id,
                context_id=context_id,
                role=role,
                physical_owner_hint=(
                    str(physical_owner_hint) if physical_owner_hint else None
                ),
                review_status=review_status,
                candidate_kind=candidate_kind,
                foil_set_id=foil_set_id,
                identity_kind=identity_kind,
                native_repetition_penalty_stratum=float(stratum),
                source_pred_row_id=(
                    str(source_pred_row_id) if source_pred_row_id else None
                ),
                prompt_prefix_token_count=int(prompt_prefix_token_count),
                prefix_token_ids=prefix_tokens,
                prefix_token_ids_sha256=str(declared_prefix_sha),
                proposal_digest=str(proposal_digest),
                basin_id=(str(basin_id) if basin_id else None),
                request_kind=request_kind,
                coord_token_ids=coord_token_ids,
                fixed_coord_token_ids=fixed_coord_token_ids,
                scan_slot=scan_slot,
            )
        )
    return rows


def load_owner_context_ledger(
    path: Path, *, rules: DecisionRules
) -> dict[tuple[str, str], Any]:
    """Load the exact v2 ledger consumed by the authoritative candidate builder."""

    if rules.materializer_rules is None:
        _fail("owner-context-ledger requires candidate-builder-v2 decision rules")
    try:
        parsed = candidate_builder._parse_ledger_rows(  # noqa: SLF001
            _read_jsonl(path), rules.materializer_rules
        )
    except (TypeError, ValueError) as exc:
        _fail(
            "owner-context ledger is incomplete or incompatible",
            path=str(path),
            error=str(exc),
        )
    return {row.key: row for row in parsed}


def validate_v2_candidate_contract(
    candidates: Sequence[CandidateRow],
    *,
    owner_context_ledger: Mapping[tuple[str, str], Any],
    rules: DecisionRules,
) -> dict[str, Any]:
    """Cross-bind every v2 request to its builder input and require complete surfaces."""

    if rules.materializer_rules is None:
        _fail("v2 candidate validation requires parsed materializer rules")
    by_key: dict[tuple[str, str], list[CandidateRow]] = {}
    for candidate in candidates:
        key = (candidate.diagnostic_owner_id, candidate.context_id)
        ledger = owner_context_ledger.get(key)
        if ledger is None:
            _fail(
                "candidate-builder-v2 row has no owning owner-context ledger row",
                candidate_id=candidate.candidate_id,
            )
        expected_prefix = tuple(
            [
                *ledger.context_tokens["token_ids"],
                *ledger.canonical_description[
                    "forced_row_prefix_through_box_start_token_ids"
                ],
            ]
        )
        if candidate.prefix_token_ids != expected_prefix:
            _fail(
                "candidate-builder-v2 prefix differs from its owner-context ledger",
                candidate_id=candidate.candidate_id,
            )
        if (
            candidate.gt_owner_id != ledger.gt_owner_id
            or candidate.image_id != ledger.image_id
            or candidate.image_identity != ledger.image_identity
            or candidate.prompt_prefix_token_count != ledger.prompt_prefix_token_count
        ):
            _fail(
                "candidate-builder-v2 stable identity differs from its owner-context ledger",
                candidate_id=candidate.candidate_id,
            )
        if (
            candidate.raw_payload.get("canonical_description")
            != ledger.canonical_description
        ):
            _fail(
                "candidate-builder-v2 canonical description differs from its ledger",
                candidate_id=candidate.candidate_id,
            )
        if (
            candidate.raw_payload.get("vocabulary_attestation")
            != ledger.vocabulary_attestation
        ):
            _fail(
                "candidate-builder-v2 vocabulary attestation differs from its ledger",
                candidate_id=candidate.candidate_id,
            )
        by_key.setdefault(key, []).append(candidate)

    for key, ledger in owner_context_ledger.items():
        rows = by_key.get(key)
        if not rows:
            _fail(
                "owner-context ledger row has no candidate-builder-v2 requests",
                owner_context=key,
            )
        free_roots = [
            row
            for row in rows
            if row.record_type == "free_coordinate_tree_root_request"
        ]
        plans = [row for row in rows if row.record_type == "conditional_y1_score_plan"]
        complete_boxes = [
            row for row in rows if row.record_type == "complete_box_candidate"
        ]
        if len(free_roots) != 1:
            _fail(
                "each owner-context requires exactly one free coordinate-tree root",
                owner_context=key,
                observed=len(free_roots),
            )

        expected_x1 = tuple(
            sorted(
                {
                    anchor.x1.value
                    for anchor in candidate_builder.enumerate_target_anchor_pairs(
                        ledger.gt_box, rules.materializer_rules.landscape
                    )
                }
            )
        )
        observed_x1 = tuple(sorted(row.coordinate_bin_values[0] for row in plans))
        if observed_x1 != expected_x1:
            _fail(
                "restricted conditional-y1 plans do not cover the complete declared x1 domain",
                owner_context=key,
                expected=list(expected_x1),
                observed=list(observed_x1),
            )
        if len({row.candidate_id for row in plans}) != len(plans):
            _fail(
                "restricted conditional-y1 plans duplicate an x1 request",
                owner_context=key,
            )

        for plan in plans:
            x1 = plan.coordinate_bin_values[0]
            expected_entries = [
                {
                    "y1": entry.y1.value,
                    "is_target_anchor": entry.is_target_anchor,
                    "can_form_valid_box": entry.can_form_valid_box,
                    "invalid_box_reason": entry.invalid_box_reason,
                }
                for entry in candidate_builder.enumerate_complete_conditional_y1(
                    candidate_builder.CoordinateBin(x1),
                    ledger.gt_box,
                    rules.materializer_rules.landscape,
                )
            ]
            if list(plan.complete_conditional_y1) != expected_entries:
                _fail(
                    "conditional-y1 plan differs from the complete core-declared row",
                    candidate_id=plan.candidate_id,
                )

        completeness_payload = {
            "canonical_description_id": ledger.canonical_description["description_id"],
            "conditional_y1_plan_sha256": [
                next(
                    plan.conditional_y1_plan_sha256
                    for plan in plans
                    if plan.coordinate_bin_values == (x1,)
                )
                for x1 in expected_x1
            ],
            "context_id": ledger.context_id,
            "coordinate_space": ledger.coordinate_space,
            "core_rule_digest": rules.rules_digest,
            "declared_x1_anchor_bins": list(expected_x1),
            "diagnostic_owner_id": ledger.diagnostic_owner_id,
            "gt_box": list(ledger.gt_box.as_tuple()),
            "gt_owner_id": ledger.gt_owner_id,
            "image_identity": ledger.image_identity,
            "landscape_decision_rules_sha256": rules.rules_file_sha256,
            "attestation_kind": rules.contract_mode,
            "canonical_description_text": ledger.canonical_description["text"],
            "canonical_description_token_digest": ledger.canonical_description[
                "token_ids_sha256"
            ],
            "context_token_digest": ledger.context_tokens["token_ids_sha256"],
            "geometry_identity_schema": rules.materializer_rules.landscape.geometry_identity_schema,
            "tokenizer_identity": ledger.vocabulary_attestation[
                "tokenizer_identity_sha256"
            ],
            "model_identity": ledger.vocabulary_attestation["model_identity_sha256"],
            "runtime_identity": ledger.vocabulary_attestation[
                "runtime_identity_sha256"
            ],
            "vocabulary_attestation": ledger.vocabulary_attestation,
        }
        expected_completeness = "restricted-completeness-plan:sha256:" + sha256_json(
            completeness_payload
        )
        if any(
            row.completeness_plan_digest != expected_completeness
            for row in [*plans, *complete_boxes]
        ):
            _fail(
                "restricted candidate surface has a stale completeness-plan digest",
                owner_context=key,
            )
        plan_ids = {row.candidate_id for row in plans}
        for candidate in complete_boxes:
            linked = candidate.raw_payload.get("conditional_y1_plan_id")
            anchor = candidate.raw_payload.get("anchor")
            if anchor is None:
                if linked is not None:
                    _fail(
                        "unanchored complete-box candidate unexpectedly links a conditional-y1 plan",
                        candidate_id=candidate.candidate_id,
                    )
            elif linked not in plan_ids:
                _fail(
                    "anchored complete-box candidate links a missing conditional-y1 plan",
                    candidate_id=candidate.candidate_id,
                )
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "owner_context_count": len(owner_context_ledger),
        "free_root_count": sum(
            row.record_type == "free_coordinate_tree_root_request" for row in candidates
        ),
        "conditional_y1_plan_count": sum(
            row.record_type == "conditional_y1_score_plan" for row in candidates
        ),
        "complete_box_candidate_count": sum(
            row.record_type == "complete_box_candidate" for row in candidates
        ),
        "status": "complete",
    }


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def resolve_owner(
    diagnostic_owner_id: str, owner_ledger: Mapping[str, OwnerLedgerEntry]
) -> OwnerLedgerEntry:
    entry = owner_ledger.get(diagnostic_owner_id)
    if entry is None or entry.status == "unresolved":
        _fail(
            "candidate targets an unresolved or unregistered owner; cannot decide B2, C, or a repair gain",
            diagnostic_owner_id=diagnostic_owner_id,
            found=(entry is not None),
        )
    return entry


def verify_no_policy_mixing(
    candidate: CandidateRow,
    prediction_row_ledger: Mapping[str, PredictionRowLedgerEntry],
) -> None:
    if candidate.source_pred_row_id is None:
        return
    row = prediction_row_ledger.get(candidate.source_pred_row_id)
    if row is None:
        _fail(
            "candidate references a prediction row absent from the sealed Task-1 ledger",
            candidate_id=candidate.candidate_id,
            source_pred_row_id=candidate.source_pred_row_id,
        )
    if not _matches_stratum(
        row.repetition_penalty, target=candidate.native_repetition_penalty_stratum
    ):
        _fail(
            "candidate's declared repetition-penalty stratum does not match the sealed native row that "
            "produced its self-prefix; refuse to pool strata",
            candidate_id=candidate.candidate_id,
            declared=candidate.native_repetition_penalty_stratum,
            ledger=row.repetition_penalty,
        )
    if row.image_id != candidate.image_id:
        _fail(
            "candidate image_id does not match its source prediction row's image_id",
            candidate_id=candidate.candidate_id,
        )


def _coord_token_ids_to_check(candidate: CandidateRow) -> tuple[int, ...]:
    if candidate.request_kind == "complete_box":
        assert candidate.coord_token_ids is not None
        return candidate.coord_token_ids
    return candidate.fixed_coord_token_ids


def verify_coordinate_tokens(candidate: CandidateRow, rules: DecisionRules) -> None:
    lo = rules.schema_tokens["coordinate_token_id_start"]
    hi = rules.schema_tokens["coordinate_token_id_end_exclusive"]
    for token in _coord_token_ids_to_check(candidate):
        if not lo <= token < hi:
            _fail(
                "candidate coordinate token id falls outside the declared coordinate vocabulary",
                candidate_id=candidate.candidate_id,
                token_id=token,
                coordinate_token_id_start=lo,
                coordinate_token_id_end_exclusive=hi,
            )


def verify_identity_kind_matches_role_constraints(candidate: CandidateRow) -> None:
    """Mirror the landed core v2's own ``RegisteredBasinRole`` identity constraint exactly.

    Target and covered-owner candidates always require reviewed physical-
    owner identity; owner-neutral ``registered_geometry`` identity is
    restricted to background/scan foil banks (verified directly against the
    real ``validate_rule_mapping`` constraint logic in
    ``sorted_owner_basin_landscape.py``, not guessed).
    """

    if candidate.identity_kind == "registered_geometry":
        if candidate.candidate_kind in OWNER_ANCHORED_CANDIDATE_KINDS:
            _fail(
                "target and covered-owner candidates cannot use owner-neutral registered_geometry identity",
                candidate_id=candidate.candidate_id,
                candidate_kind=candidate.candidate_kind,
            )
        if candidate.candidate_kind not in REGISTERED_GEOMETRY_ELIGIBLE_CANDIDATE_KINDS:
            _fail(
                "owner-neutral registered_geometry identity is restricted to background and scan foil banks",
                candidate_id=candidate.candidate_id,
                candidate_kind=candidate.candidate_kind,
            )
    elif (
        candidate.candidate_kind in OWNER_ANCHORED_CANDIDATE_KINDS
        and candidate.identity_kind != "reviewed_physical_owner"
    ):
        _fail(
            "target and covered-owner candidates require reviewed_physical_owner identity",
            candidate_id=candidate.candidate_id,
            candidate_kind=candidate.candidate_kind,
            identity_kind=candidate.identity_kind,
        )


class CoordinateTokenDomain(Protocol):
    @property
    def schema_tokens(self) -> Mapping[str, int]: ...


def verify_core_geometry_domain(core: Any, rules: CoordinateTokenDomain) -> None:
    """Cross-check this scorer's coordinate-token vocabulary against the core's real bins 0..999.

    Uses only the core's stable ``COORDINATE_BIN_MIN``/``COORDINATE_BIN_MAX``
    constants and ``GEOMETRY_IDENTITY_SCHEMA`` -- no GT box or the core's own
    (structurally different) rules-mapping payload is required for this
    check, so it never needs to invent data this scorer's ledgers do not
    carry.
    """

    core_min = getattr(core, "COORDINATE_BIN_MIN", None)
    core_max = getattr(core, "COORDINATE_BIN_MAX", None)
    if not isinstance(core_min, int) or not isinstance(core_max, int):
        _fail(
            "pure core does not expose COORDINATE_BIN_MIN/COORDINATE_BIN_MAX",
            core_min=core_min,
            core_max=core_max,
        )
    expected_bin_count = core_max - core_min + 1
    lo = rules.schema_tokens["coordinate_token_id_start"]
    hi = rules.schema_tokens["coordinate_token_id_end_exclusive"]
    observed_bin_count = hi - lo
    if observed_bin_count != expected_bin_count:
        _fail(
            "declared coordinate-token vocabulary width does not match the core's bins "
            f"{core_min}..{core_max}",
            observed_bin_count=observed_bin_count,
            expected_bin_count=expected_bin_count,
        )


def verify_canonical_description(
    candidate: CandidateRow, rules: DecisionRules
) -> tuple[int, ...]:
    owner_key = (
        candidate.gt_owner_id
        if rules.contract_version == CANDIDATE_SCHEMA_VERSION
        else candidate.diagnostic_owner_id
    )
    tokens = rules.owner_canonical_description.get(str(owner_key))
    if tokens is None:
        _fail(
            "landscape-decision-rules.json does not bind a canonical description for this owner",
            candidate_id=candidate.candidate_id,
            diagnostic_owner_id=candidate.diagnostic_owner_id,
        )
    return tokens


def resolve_foil_set_digest(candidate: CandidateRow, rules: DecisionRules) -> str:
    digest = rules.foil_set_digests.get(candidate.foil_set_id)
    if digest is None:
        _fail(
            "candidate references a foil_set_id absent from the frozen decision rules; refuse to score "
            "against an unregistered or drifted foil set",
            candidate_id=candidate.candidate_id,
            foil_set_id=candidate.foil_set_id,
        )
    return digest


def verify_production_prompt_prefix(
    candidate: CandidateRow, reconstructed_prompt_token_ids: Sequence[int]
) -> None:
    observed = tuple(candidate.prefix_token_ids[: candidate.prompt_prefix_token_count])
    expected = tuple(int(v) for v in reconstructed_prompt_token_ids)
    if observed != expected:
        _fail(
            "candidate prefix's prompt-token region does not match the production prompt/image "
            "reconstruction for this image",
            candidate_id=candidate.candidate_id,
            image_id=candidate.image_id,
            observed_count=len(observed),
            expected_count=len(expected),
        )


def derive_generated_history_token_ids(candidate: CandidateRow) -> tuple[int, ...]:
    """The literal, never-retokenized suffix of ``prefix_token_ids`` beyond the production prompt.

    This is exactly what :func:`prefill_context` concatenates onto the real
    backend-materialized prompt tensor; it is never independently
    reconstructed or re-derived from text.
    """

    return tuple(candidate.prefix_token_ids[candidate.prompt_prefix_token_count :])


def _proposal_seed(candidate: CandidateRow) -> dict[str, Any]:
    seed: dict[str, Any] = {
        "candidate_kind": candidate.candidate_kind,
        "diagnostic_owner_id": candidate.diagnostic_owner_id,
        "context_id": candidate.context_id,
        "role": candidate.role,
        "foil_set_id": candidate.foil_set_id,
        "request_kind": candidate.request_kind,
    }
    if candidate.request_kind == "complete_box":
        seed["coord_token_ids"] = list(candidate.coord_token_ids or ())
    else:
        seed["fixed_coord_token_ids"] = list(candidate.fixed_coord_token_ids)
        seed["scan_slot"] = candidate.scan_slot
    return seed


def _load_pure_core(path: Path | None = None) -> Any | None:
    """Load the deterministic pure core if present; never let its import crash this scorer.

    The core is owned by another worker and is out of scope to edit here.
    Registering the module in ``sys.modules`` before ``exec_module`` is
    required for its own dataclasses to resolve correctly (some dataclass
    features, e.g. ``order=True``, introspect ``sys.modules[cls.__module__]``
    during class creation); skipping that step raises inside the core's own
    module body, which must not be allowed to propagate out of an "absent or
    incompatible" adapter seam and take down the whole runtime scorer.
    """

    core_path = path or DEFAULT_PURE_CORE_PATH
    if path is None:
        try:
            from scripts.research import sorted_owner_basin_landscape as core_module

            return core_module
        except Exception:
            return None
    if not core_path.is_file():
        return None
    module_name = "_sorted_owner_basin_landscape_core"
    spec = importlib.util.spec_from_file_location(module_name, core_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        return None
    return module


def pure_core_status(core: Any | None, *, path: Path | None = None) -> dict[str, Any]:
    core_path = path or DEFAULT_PURE_CORE_PATH
    if core is None:
        return {
            "path": str(core_path),
            "present": False,
            "compatible_api": [],
            "note": "pure core absent",
        }
    compatible = [name for name in EXPECTED_PURE_CORE_API if hasattr(core, name)]
    fully = set(compatible) == set(EXPECTED_PURE_CORE_API)
    return {
        "path": str(core_path),
        "present": True,
        "compatible_api": compatible,
        "note": (
            "fully compatible" if fully else "core present but missing expected v2 API"
        ),
    }


def require_production_pure_core(core: Any | None, *, path: Path | None = None) -> Any:
    """Hard production gate: the core must be present and expose the real v2 API.

    Unlike :func:`pure_core_status` (which may be used permissively by
    fixture-only unit tests), this never returns a "degraded but continuing"
    outcome. A missing or incompatible core must never let a production run
    reach scoring or emit a receipt with production runtime status.
    """

    status = pure_core_status(core, path=path)
    if not status["present"] or status["note"] != "fully compatible":
        _fail(
            "production scoring requires the landed pure core v2 with its real API; refusing to "
            "continue with an absent or incompatible core",
            pure_core_status=status,
            expected_api=list(EXPECTED_PURE_CORE_API),
        )
    return core


def verify_candidate_proposal(
    candidate: CandidateRow, *, pure_core: Any | None
) -> dict[str, Any]:
    """Verify a candidate's declared ``proposal_digest`` against its own defining parameters.

    The landed core v2 exposes no ``recompute_candidate_proposal_digest``-
    shaped API (verified by reading the real module: its digest-shaped
    surface is ``complete_box_candidate_id``, which has a different identity
    domain. The candidate-builder-v2 parser separately validates its full
    ``core_request_attestation``; this function verifies the builder's frozen
    request digest seed without inventing a mismatched pure-core call.
    """

    expected_self_digest = sha256_json(_proposal_seed(candidate))
    if expected_self_digest != candidate.proposal_digest:
        _fail(
            "candidate proposal_digest does not match its own declared defining parameters "
            "(construction bug or tamper)",
            candidate_id=candidate.candidate_id,
            declared=candidate.proposal_digest,
            recomputed=expected_self_digest,
        )
    if candidate.record_type != "legacy_candidate_request":
        if pure_core is None:
            _fail(
                "candidate-builder-v2 proposal verification requires the landed pure core",
                candidate_id=candidate.candidate_id,
            )
        # The builder's request digest and its bindings were independently
        # recomputed by _validate_v2_core_request_attestation while loading.
        # Complete boxes additionally carry the pure core's own canonical
        # candidate identifier; reproduce it here from the authoritative
        # bank/source/box/anchor fields rather than declaring that core
        # verification is unavailable.
        if candidate.record_type == "complete_box_candidate":
            payload = candidate.raw_payload
            anchor_payload = payload.get("anchor")
            anchor = None
            if isinstance(anchor_payload, Mapping):
                anchor_x1 = anchor_payload.get("x1")
                anchor_y1 = anchor_payload.get("y1")
                if (
                    isinstance(anchor_x1, bool)
                    or not isinstance(anchor_x1, int)
                    or isinstance(anchor_y1, bool)
                    or not isinstance(anchor_y1, int)
                ):
                    _fail(
                        "complete-box anchor must contain integer x1/y1 bins",
                        candidate_id=candidate.candidate_id,
                    )
                anchor = pure_core.TargetAnchorPair(
                    x1=pure_core.CoordinateBin(anchor_x1),
                    y1=pure_core.CoordinateBin(anchor_y1),
                )
            box = pure_core.CoordinateBox.from_values(*candidate.coordinate_bin_values)
            recomputed_core_id = pure_core.complete_box_candidate_id(
                bank_name=str(payload.get("bank_name", "")),
                source_id=str(payload.get("source_id", "")),
                box=box,
                anchor=anchor,
            )
            if recomputed_core_id != payload.get("complete_box_candidate_id"):
                _fail(
                    "complete-box candidate does not reproduce the pure-core candidate identity",
                    candidate_id=candidate.candidate_id,
                    declared=payload.get("complete_box_candidate_id"),
                    recomputed=recomputed_core_id,
                )
        return {
            "self_consistency": "passed",
            "pure_core_recomputation": "passed",
            "pure_core_present": True,
            "verification_domain": "builder_request_attestation_and_pure_core_v2",
        }
    return {
        "self_consistency": "passed",
        "pure_core_recomputation": "not_available",
        "pure_core_present": pure_core is not None,
        "note": (
            "legacy fixture rows do not carry the v2 core_request_attestation "
            "or pure-core identity bindings"
        ),
    }


# ---------------------------------------------------------------------------
# Full-vocabulary attested scoring primitive
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttestationContext:
    """Run-level identity every full-vocabulary attestation must bind.

    Built once per execution (or once per test) and threaded through every
    scoring call, so per-call sites pass one object instead of four loose
    identity strings.
    """

    expected_vocab_size: int
    tokenizer_identity_digest: str
    model_identity_digest: str
    rule_digest: str
    runtime_receipt_id: str

    def domain_seed(self) -> dict[str, Any]:
        return {
            "vocab_size": int(self.expected_vocab_size),
            "domain": "full_lm_head_logits",
            "filtered": False,
            "tokenizer_identity_digest": self.tokenizer_identity_digest,
            "model_identity_digest": self.model_identity_digest,
            "rule_digest": self.rule_digest,
            "runtime_receipt_id": self.runtime_receipt_id,
        }


def build_attestation_context(
    *,
    expected_vocab_size: int,
    tokenizer_identity: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    rule_digest: str,
    runtime_receipt_id: str,
    frozen_tokenizer_identity_digest: str | None = None,
    frozen_model_identity_digest: str | None = None,
) -> AttestationContext:
    return AttestationContext(
        expected_vocab_size=int(expected_vocab_size),
        tokenizer_identity_digest=frozen_tokenizer_identity_digest
        or sha256_json(dict(tokenizer_identity)),
        model_identity_digest=frozen_model_identity_digest
        or sha256_json(dict(model_identity)),
        rule_digest=str(rule_digest),
        runtime_receipt_id=str(runtime_receipt_id),
    )


def build_core_full_vocabulary_attestation(
    core: Any,
    *,
    expected_vocab_size: int,
    tokenizer_identity_digest: str,
    model_identity_digest: str,
    runtime_rule_digest: str,
) -> Any:
    """Construct the core v2's own ``FullVocabularyAttestation`` for this run's vocabulary.

    Uses the core's real ``full_vocabulary_id_digest``/``FullVocabularyAttestation``
    (verified by reading the landed module) instead of this scorer's cheaper
    ad hoc domain digest; the core's own ``__post_init__`` validates the
    digest matches the contiguous ``0..V-1`` domain. Computed once per run
    (the digest hashes the full vocabulary id range), not per candidate.
    """

    contiguous_digest = core.full_vocabulary_id_digest(int(expected_vocab_size))
    return core.FullVocabularyAttestation(
        expected_vocabulary_size=int(expected_vocab_size),
        contiguous_token_id_digest=contiguous_digest,
        tokenizer_identity=str(tokenizer_identity_digest),
        model_identity=str(model_identity_digest),
        runtime_rule_digest=str(runtime_rule_digest),
    )


@dataclass(frozen=True)
class VocabAttestation:
    vocab_size: int
    domain_digest: str
    filtered: bool
    tokenizer_identity_digest: str
    model_identity_digest: str
    rule_digest: str
    runtime_receipt_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "domain_digest": self.domain_digest,
            "filtered": self.filtered,
            "tokenizer_identity_digest": self.tokenizer_identity_digest,
            "model_identity_digest": self.model_identity_digest,
            "rule_digest": self.rule_digest,
            "runtime_receipt_id": self.runtime_receipt_id,
        }


def attest_full_vocabulary(
    observed_last_dim: int, *, attestation: AttestationContext
) -> VocabAttestation:
    if int(observed_last_dim) != int(attestation.expected_vocab_size):
        _fail(
            "logits vocabulary dimension does not match the attested model vocabulary size; refusing to "
            "score against a filtered or mismatched domain",
            observed=int(observed_last_dim),
            expected=int(attestation.expected_vocab_size),
        )
    digest = sha256_json(attestation.domain_seed())
    return VocabAttestation(
        vocab_size=int(attestation.expected_vocab_size),
        domain_digest=digest,
        filtered=False,
        tokenizer_identity_digest=attestation.tokenizer_identity_digest,
        model_identity_digest=attestation.model_identity_digest,
        rule_digest=attestation.rule_digest,
        runtime_receipt_id=attestation.runtime_receipt_id,
    )


def score_coordinate_position(
    logits_at_position: torch.Tensor,
    *,
    coordinate_token_id_start: int,
    coordinate_token_id_end_exclusive: int,
    attestation: AttestationContext,
    running_context_token_ids: Sequence[int],
    repetition_penalties: Sequence[float] = REPETITION_PENALTY_STRATA,
) -> dict[str, Any]:
    """Raw + repetition-penalty-adjusted coordinate-bin log-probabilities at one position.

    Both the raw channel and every auxiliary policy channel are computed with
    ``log_softmax`` over the complete, unfiltered last dimension of
    ``logits_at_position``; the coordinate-bin range is read out only after
    that full-domain normalization, so no channel is ever renormalized over a
    filtered subset. The returned ``bin_logprobs`` covers
    ``[coordinate_token_id_start, coordinate_token_id_end_exclusive)`` exactly
    once, in bin order. ``attestation`` binds this reading to the tokenizer,
    model, rule, and runtime-receipt identity it was produced under.
    """

    if logits_at_position.ndim != 1:
        raise ValueError(
            "score_coordinate_position expects a single-position logits vector [vocab]"
        )
    vocab_size = int(logits_at_position.shape[-1])
    attested = attest_full_vocabulary(vocab_size, attestation=attestation)
    lo, hi = int(coordinate_token_id_start), int(coordinate_token_id_end_exclusive)
    bin_count = hi - lo
    if bin_count <= 0:
        raise ValueError(
            "coordinate_token_id_end_exclusive must be greater than coordinate_token_id_start"
        )

    def _slice_bins(full_log_probs: torch.Tensor) -> list[float]:
        coord_slice = full_log_probs[lo:hi]
        if int(coord_slice.shape[0]) != bin_count:
            _fail(
                "coordinate-bin slice does not cover every declared bin exactly once",
                expected_bin_count=bin_count,
                observed_bin_count=int(coord_slice.shape[0]),
            )
        return [float(v) for v in coord_slice.detach().to(device="cpu").tolist()]

    raw_full = torch.log_softmax(logits_at_position.to(dtype=torch.float32), dim=-1)
    raw_view = {
        "bin_logprobs": _slice_bins(raw_full),
        "vocab_attestation": attested.to_dict(),
    }

    policy_view: dict[str, Any] = {}
    for penalty in repetition_penalties:
        from transformers import RepetitionPenaltyLogitsProcessor

        processor = RepetitionPenaltyLogitsProcessor(penalty=float(penalty))
        context_ids = torch.tensor(
            [[int(v) for v in running_context_token_ids]], dtype=torch.long
        )
        scores = logits_at_position.to(dtype=torch.float32).clone().unsqueeze(0)
        # transformers annotates this processor with dtype-specific tensor
        # aliases that torch's factories do not preserve statically. The
        # explicit dtypes above establish the runtime contract.
        processed = processor(
            cast(torch.LongTensor, context_ids), cast(torch.FloatTensor, scores)
        )
        policy_full = torch.log_softmax(processed[0], dim=-1)
        policy_view[_policy_view_key(penalty)] = {
            "bin_logprobs": _slice_bins(policy_full),
            "vocab_attestation": attested.to_dict(),
        }

    return {
        "coordinate_bin_count": bin_count,
        "coordinate_token_id_start": lo,
        "coordinate_token_id_end_exclusive": hi,
        "raw": raw_view,
        "auxiliary_policy": policy_view,
    }


def _single_bin_view(
    logits_at_position: torch.Tensor,
    *,
    token_id: int,
    attestation: AttestationContext,
    running_context_token_ids: Sequence[int],
    repetition_penalties: Sequence[float] = REPETITION_PENALTY_STRATA,
) -> dict[str, Any]:
    return score_coordinate_position(
        logits_at_position,
        coordinate_token_id_start=int(token_id),
        coordinate_token_id_end_exclusive=int(token_id) + 1,
        attestation=attestation,
        running_context_token_ids=running_context_token_ids,
        repetition_penalties=repetition_penalties,
    )


def combine_complete_box(views: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Combine four single-bin :func:`score_coordinate_position` views into one box score.

    Preserves each of ``x1``, ``y1``, ``x2``, ``y2`` individually (a scalar
    row score alone is insufficient per unit.md) and sums them into
    ``complete_box_logprob_sum`` -- the autoregressive chain is additive in
    log space. The raw view never depends on which/how many repetition
    penalties were requested.
    """

    missing = [name for name in COORD_SLOTS if name not in views]
    if missing:
        raise ValueError(f"complete-box combination is missing views for: {missing}")

    raw: dict[str, Any] = {
        f"{name}_logprob": views[name]["raw"]["bin_logprobs"][0] for name in COORD_SLOTS
    }
    raw["complete_box_logprob_sum"] = sum(
        raw[f"{name}_logprob"] for name in COORD_SLOTS
    )
    raw["vocab_attestation"] = {
        name: views[name]["raw"]["vocab_attestation"] for name in COORD_SLOTS
    }

    penalty_keys: set[str] = set()
    for name in COORD_SLOTS:
        penalty_keys.update(views[name]["auxiliary_policy"])
    policy: dict[str, Any] = {}
    for key in sorted(penalty_keys):
        per_slot: dict[str, Any] = {}
        for name in COORD_SLOTS:
            entry = views[name]["auxiliary_policy"].get(key)
            if entry is None:
                raise ValueError(
                    f"policy view {key!r} missing for coordinate {name!r}; refusing partial policy mixing"
                )
            per_slot[f"{name}_logprob"] = entry["bin_logprobs"][0]
        per_slot["complete_box_logprob_sum"] = sum(
            per_slot[f"{name}_logprob"] for name in COORD_SLOTS
        )
        per_slot["vocab_attestation"] = {
            name: views[name]["auxiliary_policy"][key]["vocab_attestation"]
            for name in COORD_SLOTS
        }
        policy[key] = per_slot

    return {"raw": raw, "auxiliary_policy": policy}


# ---------------------------------------------------------------------------
# Prefix-KV-cache branching engine
# ---------------------------------------------------------------------------


class CacheBackend(Protocol):
    """Structural contract the branching engine needs from a cache-bearing forward.

    Deliberately narrow so tests can supply a pure-Python fake with no torch
    model dependency; the real implementation (:class:`HFCacheBackend`) wraps
    a ``transformers`` ``DynamicCache``.
    """

    @property
    def cache_length(self) -> int: ...

    @property
    def layer_count(self) -> int | None:
        """Observed number of cache layers, or ``None`` if not applicable/known."""
        ...

    def step(self, token_ids: Sequence[int]) -> torch.Tensor:
        """Append ``token_ids``, forward, and return logits ``[len(token_ids), vocab]``.

        Row ``i`` is the distribution predicting the token *after* the
        ``i``-th newly appended token.
        """
        ...

    def crop(self, length: int) -> None:
        """Truncate the cache back to ``length`` tokens, undoing later steps."""
        ...


class FullReforwardBackend:
    """Logical branching over true ``use_cache=False`` literal reforwards."""

    def __init__(
        self,
        *,
        root_prefix_token_ids: Sequence[int],
        full_reforward: Callable[[Sequence[int]], torch.Tensor],
        batched_full_reforward: Callable[
            [Sequence[Sequence[int]]], torch.Tensor
        ]
        | None = None,
        full_reforward_batch_size: int = 1,
        context_id: str,
        group_id: str,
        progress_every_actual_forwards: int = 250,
    ) -> None:
        self._root_prefix = tuple(int(value) for value in root_prefix_token_ids)
        if not self._root_prefix:
            raise ValueError("full-reforward backend requires a non-empty root prefix")
        self._suffix: list[int] = []
        self._full_reforward = full_reforward
        self._batched_full_reforward = batched_full_reforward
        self._full_reforward_batch_size = int(full_reforward_batch_size)
        if self._full_reforward_batch_size <= 0:
            raise ValueError("full-reforward batch size must be positive")
        if self._full_reforward_batch_size > 1 and batched_full_reforward is None:
            raise ValueError(
                "batched full-reforward closure is required when batch size exceeds one"
            )
        self._context_id = context_id
        self._group_id = group_id
        self._progress_every = int(progress_every_actual_forwards)
        if self._progress_every < 0:
            raise ValueError("progress interval must be non-negative")
        self._started_at = time.monotonic()
        self._logical_requests_by_depth: dict[int, int] = {}
        self._actual_forwards_by_depth: dict[int, int] = {}
        self._memo_hits_by_depth: dict[int, int] = {}
        self._memo: dict[int, dict[tuple[int, ...], torch.Tensor]] = {
            1: {},
            2: {},
        }
        # A prefetched value is an actual literal sequence evaluation whose
        # corresponding logical step has not yet consumed it. Depths 1/2 then
        # become the existing retained memo; depth 3 is deleted immediately
        # after its reserved consumers finish so the historical retention
        # contract stays exactly [0, 1, 2].
        self._prefetched_unconsumed: dict[
            tuple[int, tuple[int, ...]], int
        ] = {}
        self._ephemeral_depth_three: dict[tuple[int, ...], torch.Tensor] = {}
        self._actual_forward_calls = 0
        self._physical_model_forward_calls = 0
        self._physical_forward_batch_histogram: Counter[int] = Counter()
        self._requested_sequence_batch_histogram: Counter[int] = Counter()
        self._padded_model_sequence_evaluations = 0
        self._last_progress_bucket = 0
        self._root_logits = self._forward(self._root_prefix, relative_depth=0)

    @property
    def root_logits(self) -> torch.Tensor:
        return self._root_logits.clone()

    @property
    def cache_length(self) -> int:
        return len(self._root_prefix) + len(self._suffix)

    @property
    def layer_count(self) -> int | None:
        return None

    def _forward(
        self, literal_prefix: Sequence[int], *, relative_depth: int
    ) -> torch.Tensor:
        logits = self._full_reforward(literal_prefix)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
            raise RuntimeError("full reforward backend requires rank-1 logits")
        self._actual_forward_calls += 1
        self._physical_model_forward_calls += 1
        self._physical_forward_batch_histogram[1] += 1
        self._requested_sequence_batch_histogram[1] += 1
        self._actual_forwards_by_depth[relative_depth] = (
            self._actual_forwards_by_depth.get(relative_depth, 0) + 1
        )
        self._report_progress_if_due()
        return logits.detach().to(device="cpu", dtype=torch.float32)

    def _report_progress_if_due(self) -> None:
        if (
            self._progress_every > 0
            and self._actual_forward_calls // self._progress_every
            > self._last_progress_bucket
        ):
            self._last_progress_bucket = (
                self._actual_forward_calls // self._progress_every
            )
            elapsed = max(time.monotonic() - self._started_at, 1e-9)
            rate = self._actual_forward_calls / elapsed
            print(
                "full-reforward progress "
                f"context_id={self._context_id} group_id={self._group_id} "
                f"actual_forwards={self._actual_forward_calls} "
                f"physical_model_forwards={self._physical_model_forward_calls} "
                f"elapsed_seconds={elapsed:.1f} forwards_per_second={rate:.3f}",
                file=sys.stderr,
                flush=True,
            )

    def _forward_batch(
        self,
        literal_prefixes: Sequence[Sequence[int]],
        *,
        relative_depth: int,
    ) -> torch.Tensor:
        if not literal_prefixes:
            raise ValueError("full-reforward batch requires at least one sequence")
        if self._batched_full_reforward is None:
            raise RuntimeError("batched full-reforward closure is unavailable")
        logits = self._batched_full_reforward(literal_prefixes)
        requested = len(literal_prefixes)
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 2
            or int(logits.shape[0]) != requested
        ):
            raise RuntimeError(
                "batched full reforward requires rank-2 logits with one row per sequence"
            )
        self._actual_forward_calls += requested
        self._physical_model_forward_calls += 1
        self._physical_forward_batch_histogram[self._full_reforward_batch_size] += 1
        self._requested_sequence_batch_histogram[requested] += 1
        self._padded_model_sequence_evaluations += (
            self._full_reforward_batch_size - requested
        )
        self._actual_forwards_by_depth[relative_depth] = (
            self._actual_forwards_by_depth.get(relative_depth, 0) + requested
        )
        self._report_progress_if_due()
        return logits.detach().to(device="cpu", dtype=torch.float32)

    def prefetch_suffixes(self, suffixes: Sequence[Sequence[int]]) -> None:
        """Batch literal same-depth suffixes without changing logical traversal.

        The caller must later consume every supplied suffix through ``step``.
        Depths 1/2 retain the historical exact memo semantics. Depth 3 keeps a
        short-lived reservation only, which is released when the corresponding
        candidate step consumes it.
        """

        if self._full_reforward_batch_size <= 1 or not suffixes:
            return
        grouped: dict[int, list[tuple[int, ...]]] = {}
        for raw_suffix in suffixes:
            key = tuple(int(value) for value in raw_suffix)
            depth = len(key)
            if depth not in {1, 2, 3}:
                raise ValueError("prefetch supports only relative depths 1, 2, and 3")
            if self._suffix and key[: len(self._suffix)] != tuple(self._suffix):
                raise RuntimeError(
                    "prefetched suffix does not descend from the active logical branch"
                )
            grouped.setdefault(depth, []).append(key)

        for depth in sorted(grouped):
            occurrence_counts = Counter(grouped[depth])
            evaluation_keys: list[tuple[int, ...]] = []
            if depth in self._memo:
                memo = self._memo[depth]
                for key in occurrence_counts:
                    if key not in memo:
                        evaluation_keys.append(key)
            else:
                # Preserve the uncached reference's old depth-3 accounting:
                # duplicate candidate occurrences remain duplicate literal
                # sequence evaluations, merely co-scheduled in one GPU call.
                for key, count in occurrence_counts.items():
                    if (depth, key) in self._prefetched_unconsumed:
                        raise RuntimeError("depth-3 prefetch reservation was not consumed")
                    evaluation_keys.extend([key] * count)

            for start in range(0, len(evaluation_keys), self._full_reforward_batch_size):
                batch_keys = evaluation_keys[
                    start : start + self._full_reforward_batch_size
                ]
                literal_batch = [
                    [*self._root_prefix, *key] for key in batch_keys
                ]
                batch_logits = self._forward_batch(
                    literal_batch, relative_depth=depth
                )
                for key, logits in zip(batch_keys, batch_logits, strict=True):
                    if depth in self._memo:
                        self._memo[depth][key] = logits
                        self._prefetched_unconsumed[(depth, key)] = 1
                    else:
                        self._ephemeral_depth_three[key] = logits
                        reservation_key = (depth, key)
                        self._prefetched_unconsumed[reservation_key] = (
                            self._prefetched_unconsumed.get(reservation_key, 0) + 1
                        )

    def step(self, token_ids: Sequence[int]) -> torch.Tensor:
        if not token_ids:
            raise ValueError("full-reforward step requires at least one token")
        rows: list[torch.Tensor] = []
        for token_id in token_ids:
            self._suffix.append(int(token_id))
            depth = len(self._suffix)
            self._logical_requests_by_depth[depth] = (
                self._logical_requests_by_depth.get(depth, 0) + 1
            )
            suffix_key = tuple(self._suffix)
            memo = self._memo.get(depth)
            reservation_key = (depth, suffix_key)
            reserved = self._prefetched_unconsumed.get(reservation_key, 0)
            if reserved > 0:
                logits = (
                    memo[suffix_key]
                    if memo is not None
                    else self._ephemeral_depth_three[suffix_key]
                )
                if reserved == 1:
                    self._prefetched_unconsumed.pop(reservation_key)
                    if depth == 3:
                        self._ephemeral_depth_three.pop(suffix_key)
                else:
                    self._prefetched_unconsumed[reservation_key] = reserved - 1
            elif memo is not None and suffix_key in memo:
                self._memo_hits_by_depth[depth] = (
                    self._memo_hits_by_depth.get(depth, 0) + 1
                )
                logits = memo[suffix_key]
            else:
                logits = self._forward(
                    [*self._root_prefix, *self._suffix], relative_depth=depth
                )
                if memo is not None:
                    memo[suffix_key] = logits
            rows.append(logits)
        return torch.stack(rows)

    def crop(self, length: int) -> None:
        relative_length = int(length) - len(self._root_prefix)
        if relative_length < 0 or relative_length > len(self._suffix):
            raise ValueError(
                "full-reforward crop must truncate within the current logical suffix"
            )
        del self._suffix[relative_length:]

    def accounting(self) -> dict[str, Any]:
        if self._prefetched_unconsumed or self._ephemeral_depth_three:
            raise RuntimeError(
                "full-reforward accounting found unconsumed prefetch reservations"
            )
        return {
            "scoring_backend": FULL_REFORWARD_SCORING_BACKEND,
            "context_id": self._context_id,
            "group_id": self._group_id,
            "root_prefix_length": len(self._root_prefix),
            "root_calls": self._actual_forwards_by_depth.get(0, 0),
            "logical_token_step_requests": sum(
                self._logical_requests_by_depth.values()
            ),
            "logical_token_step_requests_by_depth": {
                str(depth): count
                for depth, count in sorted(self._logical_requests_by_depth.items())
            },
            "actual_forward_calls": self._actual_forward_calls,
            "physical_model_forward_calls": self._physical_model_forward_calls,
            "configured_full_reforward_batch_size": self._full_reforward_batch_size,
            "maximum_observed_model_forward_batch_size": max(
                self._physical_forward_batch_histogram, default=0
            ),
            "physical_forward_batch_histogram": {
                str(batch_size): count
                for batch_size, count in sorted(
                    self._physical_forward_batch_histogram.items()
                )
            },
            "requested_sequence_batch_histogram": {
                str(batch_size): count
                for batch_size, count in sorted(
                    self._requested_sequence_batch_histogram.items()
                )
            },
            "padded_model_sequence_evaluations": self._padded_model_sequence_evaluations,
            "actual_forward_calls_by_depth": {
                str(depth): count
                for depth, count in sorted(self._actual_forwards_by_depth.items())
            },
            "memo_hits_by_depth": {
                str(depth): self._memo_hits_by_depth.get(depth, 0)
                for depth in (1, 2)
            },
            "memo_entries_by_depth": {
                "0": 1,
                "1": len(self._memo[1]),
                "2": len(self._memo[2]),
            },
            "retained_relative_depths": [0, 1, 2],
        }


class AccountingCacheBackend:
    """Transparent cache backend wrapper with score-path-only forward counts."""

    def __init__(
        self, backend: CacheBackend, *, context_id: str, group_id: str
    ) -> None:
        self._backend = backend
        self._root_length = backend.cache_length
        self._context_id = context_id
        self._group_id = group_id
        self._logical_requests_by_depth: dict[int, int] = {}
        self._actual_forwards_by_depth: dict[int, int] = {0: 1}
        self._actual_step_forward_calls = 0

    @property
    def cache_length(self) -> int:
        return self._backend.cache_length

    @property
    def layer_count(self) -> int | None:
        return self._backend.layer_count

    def step(self, token_ids: Sequence[int]) -> torch.Tensor:
        if not token_ids:
            raise ValueError("cache step requires at least one token")
        start_depth = self.cache_length - self._root_length
        for offset, _token_id in enumerate(token_ids, start=1):
            depth = start_depth + offset
            self._logical_requests_by_depth[depth] = (
                self._logical_requests_by_depth.get(depth, 0) + 1
            )
        logits = self._backend.step(token_ids)
        final_depth = start_depth + len(token_ids)
        self._actual_step_forward_calls += 1
        self._actual_forwards_by_depth[final_depth] = (
            self._actual_forwards_by_depth.get(final_depth, 0) + 1
        )
        return logits

    def crop(self, length: int) -> None:
        self._backend.crop(length)

    def accounting(self) -> dict[str, Any]:
        return {
            "scoring_backend": KV_CACHE_SCORING_BACKEND,
            "context_id": self._context_id,
            "group_id": self._group_id,
            "root_prefix_length": self._root_length,
            "root_calls": 1,
            "logical_token_step_requests": sum(
                self._logical_requests_by_depth.values()
            ),
            "logical_token_step_requests_by_depth": {
                str(depth): count
                for depth, count in sorted(self._logical_requests_by_depth.items())
            },
            "actual_forward_calls": 1 + self._actual_step_forward_calls,
            "actual_forward_calls_by_depth": {
                str(depth): count
                for depth, count in sorted(self._actual_forwards_by_depth.items())
            },
            "memo_hits_by_depth": {},
            "memo_entries_by_depth": {},
            "retained_relative_depths": [],
        }


class BranchCursor:
    """One DFS branch scope: crop back to the entry cache length on exit.

    ``DynamicCache`` mutates in place, so exploring sibling branches from a
    shared ancestor requires restoring the cache to its pre-branch length
    before the next sibling begins; that restoration is this class's only
    job. Nested ``BranchCursor`` scopes compose (inner exit crops back to the
    inner entry point; outer exit then crops back to the outer entry point).
    """

    def __init__(self, backend: CacheBackend) -> None:
        self._backend = backend
        self._entry_length = backend.cache_length

    def step(self, token_ids: Sequence[int]) -> torch.Tensor:
        return self._backend.step(token_ids)

    def __enter__(self) -> "BranchCursor":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._backend.crop(self._entry_length)


# ---------------------------------------------------------------------------
# Explicit Qwen mrope continuation positions
# ---------------------------------------------------------------------------


def derive_prefill_position_state(
    model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(position_ids[3,1,seq], rope_deltas[1,1])`` from the model's own ``get_rope_index``."""

    owner = getattr(model, "model", model)
    fn = getattr(owner, "get_rope_index", None)
    if not callable(fn):
        raise TypeError(
            "model does not expose get_rope_index for explicit Qwen mrope positions"
        )
    with torch.no_grad():
        rope_index = fn(
            input_ids, image_grid_thw, None, attention_mask=attention_mask
        )
    if not isinstance(rope_index, tuple) or len(rope_index) != 2:
        raise ValueError("get_rope_index must return a two-tensor tuple")
    position_ids, rope_deltas = rope_index
    if not isinstance(position_ids, torch.Tensor) or position_ids.ndim != 3:
        raise ValueError("get_rope_index returned invalid position_ids")
    if not isinstance(rope_deltas, torch.Tensor):
        raise ValueError("get_rope_index returned invalid rope_deltas")
    return position_ids.detach().clone(), rope_deltas.detach().clone()


def continuation_position_ids(
    *, context_length_before_step: int, rope_deltas: torch.Tensor, new_token_count: int
) -> torch.Tensor:
    """Explicit per-step mrope position ids for pure-text continuation after prefill.

    Grounded directly in the installed Qwen3-VL modeling source
    (``modeling_qwen3_vl.py``): once past the vision block, HF computes
    ``delta = cache_position[0] + rope_deltas`` and
    ``position_ids = arange(seq_length) + delta``, replicated identically
    across all three mrope axes. ``context_length_before_step`` plays the
    role of ``cache_position[0]`` for a freshly appended, contiguous chunk of
    ``new_token_count`` tokens. This must not be reconstructed by naively
    incrementing a running counter without ``rope_deltas`` -- doing so
    silently drifts from the model's own prefill-time convention.
    """

    if new_token_count <= 0:
        raise ValueError("continuation requires at least one new token")
    delta = rope_deltas.reshape(-1)[0]
    offsets = (
        torch.arange(
            int(new_token_count), dtype=delta.dtype, device=delta.device
        )
        + int(context_length_before_step)
        + delta
    )
    return offsets.view(1, 1, -1).expand(3, 1, -1).to(dtype=torch.long).clone()


# ---------------------------------------------------------------------------
# Real HF cache backend (not exercised by unit tests; requires a live model)
# ---------------------------------------------------------------------------


@dataclass
class HFCacheBackend:
    """Post-prefill continuation only: never re-passes ``pixel_values``/``image_grid_thw``.

    The prefill call already folded the image into the KV cache; Qwen3-VL's
    own ``prepare_inputs_for_generation`` clears vision inputs once
    ``cache_position[0] != 0`` for exactly this reason. ``model`` must be used
    exclusively for the lifetime of one backend (no concurrent scoring on the
    same model instance), because Qwen3-VL's forward touches a shared,
    mutable ``rope_deltas`` attribute on the model itself when it detects a
    fresh prefill; this backend never reads that shared attribute -- it
    carries its own ``rope_deltas`` captured once at prefill time -- but a
    concurrent caller that lets the model recompute its own could still
    corrupt an interleaved session.
    """

    model: Any
    cache: Any
    rope_deltas: torch.Tensor

    @property
    def cache_length(self) -> int:
        return int(self.cache.get_seq_length())

    @property
    def layer_count(self) -> int | None:
        layers = getattr(self.cache, "layers", None)
        return None if layers is None else len(layers)

    def crop(self, length: int) -> None:
        self.cache.crop(int(length))
        observed = int(self.cache.get_seq_length())
        if observed != int(length):
            _fail(
                "cache crop did not restore the aggregate sequence length",
                expected=int(length),
                observed=observed,
            )
        for layer_index, layer in enumerate(getattr(self.cache, "layers", [])):
            layer_observed = int(layer.get_seq_length())
            if layer_observed != int(length):
                _fail(
                    "cache crop did not restore every layer; a stale layer would silently contaminate "
                    "the next branch",
                    layer_index=layer_index,
                    expected=int(length),
                    observed=layer_observed,
                )

    def step(self, token_ids: Sequence[int]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        ids = torch.tensor(
            [[int(v) for v in token_ids]], dtype=torch.long, device=device
        )
        context_length_before = self.cache_length
        position_ids = continuation_position_ids(
            context_length_before_step=context_length_before,
            rope_deltas=self.rope_deltas,
            new_token_count=int(ids.shape[1]),
        ).to(device=device)
        attention_mask = torch.ones(
            (1, context_length_before + int(ids.shape[1])),
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode():
            outputs = self.model(
                input_ids=ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self.cache,
                use_cache=True,
                return_dict=True,
                logits_to_keep=int(ids.shape[1]),
            )
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise RuntimeError("model forward did not return rank-3 logits")
        self.cache = outputs.past_key_values
        return logits[0].detach().to(device="cpu", dtype=torch.float32)


@dataclass(frozen=True)
class PrefillResult:
    backend: HFCacheBackend
    prefill_logits: torch.Tensor
    prefill_length: int


#: Keys of a ``_materialize_native_inputs`` result that are per-forward
#: identity (position/cache bookkeeping we derive ourselves), not model
#: kwargs. Everything else (``pixel_values``, ``image_grid_thw``,
#: ``video_grid_thw``, ...) is passed through to the model unchanged.
_NATIVE_INPUT_NON_MODEL_KWARGS = frozenset({"input_ids", "attention_mask"})


def _require_native_image_grid_thw(
    native_prompt_inputs: Mapping[str, Any], *, context: str
) -> torch.Tensor:
    image_grid_thw = native_prompt_inputs.get("image_grid_thw")
    if not isinstance(image_grid_thw, torch.Tensor):
        _fail(
            f"{context} requires a real materialized image_grid_thw tensor from the HF backend session's "
            "native input path; a None/placeholder grid cannot derive explicit Qwen mrope positions",
            observed_type=type(image_grid_thw).__name__,
        )
    return image_grid_thw


def _native_vision_kwargs(native_prompt_inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in native_prompt_inputs.items()
        if key not in _NATIVE_INPUT_NON_MODEL_KWARGS
    }


def prefill_context(
    model: Any,
    *,
    native_prompt_inputs: Mapping[str, Any],
    generated_history_token_ids: Sequence[int],
) -> PrefillResult:
    """Prefill exactly once through the real multimodal prefix: the processor-expanded,

    backend-materialized production prompt (with its real ``pixel_values``/
    ``image_grid_thw``, already verified against the HF session's exact
    native-input path) concatenated with the literal, never-retokenized
    generated-history/self-prefix suffix through ``<|box_start|>``.

    ``native_prompt_inputs`` must be the mapping returned by
    ``HFBackendSession._materialize_native_inputs`` for exactly this image
    (its tensors are already on the model's device); this function never
    reconstructs the prompt independently. Vision kwargs
    (``pixel_values``/``image_grid_thw``/...) are consumed only at this
    prefill call; :meth:`HFCacheBackend.step` never re-passes them.
    ``logits_to_keep=1`` keeps the returned logits to the single position
    that matters (predicting the token right after the prefix) without
    paying to materialize a full-vocab row for every prefill position.
    """

    from transformers.cache_utils import DynamicCache

    prompt_input_ids = native_prompt_inputs.get("input_ids")
    if (
        not isinstance(prompt_input_ids, torch.Tensor)
        or prompt_input_ids.ndim != 2
        or int(prompt_input_ids.shape[0]) != 1
    ):
        _fail(
            "native_prompt_inputs.input_ids must be a materialized [1, prompt_len] tensor",
            context="prefill_context",
        )
    history = [int(v) for v in generated_history_token_ids]
    if not history:
        raise ValueError(
            "prefill requires at least one generated-history/self-prefix token beyond the prompt"
        )

    device = prompt_input_ids.device
    suffix = torch.tensor([history], dtype=prompt_input_ids.dtype, device=device)
    ids = torch.cat([prompt_input_ids, suffix], dim=1)
    attention_mask = torch.ones_like(ids)
    image_grid_thw = _require_native_image_grid_thw(
        native_prompt_inputs, context="prefill_context"
    )
    position_ids, rope_deltas = derive_prefill_position_state(
        model,
        input_ids=ids,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
    )
    cache = DynamicCache()
    vision_kwargs = _native_vision_kwargs(native_prompt_inputs)
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
        raise RuntimeError("prefill forward did not return rank-3 logits")
    backend = HFCacheBackend(
        model=model, cache=outputs.past_key_values, rope_deltas=rope_deltas
    )
    prefill_logits = logits[0, -1, :].detach().to(device="cpu", dtype=torch.float32)
    return PrefillResult(
        backend=backend, prefill_logits=prefill_logits, prefill_length=int(ids.shape[1])
    )


# ---------------------------------------------------------------------------
# Mandatory cache-vs-reforward parity gate
# ---------------------------------------------------------------------------


def _branch_single_step_readout(backend: CacheBackend, token_id: int) -> torch.Tensor:
    with BranchCursor(backend) as branch:
        return branch.step([token_id])[-1].clone()


def _build_full_reforward_closure(
    model: Any, *, native_prompt_inputs: Mapping[str, Any]
) -> Any:
    """An independent ``use_cache=False`` oracle: the ground truth the cache-branch path must match.

    Uses the same real, backend-materialized ``image_grid_thw``/vision kwargs
    as :func:`prefill_context` -- never a placeholder or independently
    reconstructed grid -- since every literal token sequence the parity gate
    reforwards still starts with the same processor-expanded, image-bearing
    prompt.
    """

    image_grid_thw = _require_native_image_grid_thw(
        native_prompt_inputs, context="_build_full_reforward_closure"
    )
    vision_kwargs = _native_vision_kwargs(native_prompt_inputs)

    def _call(token_ids: Sequence[int]) -> torch.Tensor:
        device = next(model.parameters()).device
        ids = torch.tensor(
            [[int(v) for v in token_ids]], dtype=torch.long, device=device
        )
        attention_mask = torch.ones_like(ids)
        position_ids, _rope_deltas = derive_prefill_position_state(
            model,
            input_ids=ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
        )
        with torch.inference_mode():
            outputs = model(
                input_ids=ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
                **vision_kwargs,
            )
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise RuntimeError("full reforward did not return rank-3 logits")
        return logits[0, -1, :].detach().to(device="cpu", dtype=torch.float32)

    return _call


def _build_batched_full_reforward_closure(
    model: Any,
    *,
    native_prompt_inputs: Mapping[str, Any],
    maximum_batch_size: int,
) -> Any:
    """Build an equal-length literal ``use_cache=False`` batch oracle.

    ``native_prompt_inputs`` must have been materialized for exactly
    ``maximum_batch_size`` repetitions of the same image and prompt. A final
    short candidate batch is padded by repeating its last literal sequence so
    the vision tensors retain that fixed native batch identity; padded rows are
    discarded before returning and are separately reported by the backend.
    """

    batch_size = int(maximum_batch_size)
    if batch_size <= 1:
        raise ValueError("batched full reforward requires maximum_batch_size > 1")
    prompt_input_ids = native_prompt_inputs.get("input_ids")
    if (
        not isinstance(prompt_input_ids, torch.Tensor)
        or prompt_input_ids.ndim != 2
        or int(prompt_input_ids.shape[0]) != batch_size
    ):
        _fail(
            "batched native_prompt_inputs.input_ids must match the configured batch size",
            configured_batch_size=batch_size,
            observed_shape=None
            if not isinstance(prompt_input_ids, torch.Tensor)
            else list(prompt_input_ids.shape),
        )
    image_grid_thw = _require_native_image_grid_thw(
        native_prompt_inputs, context="_build_batched_full_reforward_closure"
    )
    if int(image_grid_thw.shape[0]) != batch_size:
        _fail(
            "batched materialized image_grid_thw must contain one row per repeated image",
            configured_batch_size=batch_size,
            observed_shape=list(image_grid_thw.shape),
        )
    vision_kwargs = _native_vision_kwargs(native_prompt_inputs)

    def _call(token_id_rows: Sequence[Sequence[int]]) -> torch.Tensor:
        requested = len(token_id_rows)
        if requested <= 0 or requested > batch_size:
            raise ValueError("literal batch size is outside the configured maximum")
        normalized = [tuple(int(value) for value in row) for row in token_id_rows]
        lengths = {len(row) for row in normalized}
        if len(lengths) != 1 or not next(iter(lengths)):
            raise ValueError("literal batch rows must be non-empty and equal length")
        padded = [*normalized]
        padded.extend([normalized[-1]] * (batch_size - requested))
        device = next(model.parameters()).device
        ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.ones_like(ids)
        position_ids, _rope_deltas = derive_prefill_position_state(
            model,
            input_ids=ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
        )
        with torch.inference_mode():
            outputs = model(
                input_ids=ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
                **vision_kwargs,
            )
        logits = getattr(outputs, "logits", None)
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 3
            or int(logits.shape[0]) != batch_size
        ):
            raise RuntimeError("batched full reforward did not return expected rank-3 logits")
        return logits[:requested, -1, :].detach().to(
            device="cpu", dtype=torch.float32
        )

    return _call


def run_cache_parity_gate(
    *,
    backend: CacheBackend,
    prefill_logits: torch.Tensor,
    prefix_token_ids: Sequence[int],
    x1_token_id: int,
    y1_token_id: int,
    x2_token_id: int,
    y2_token_id: int,
    coordinate_token_id_start: int,
    coordinate_token_id_end_exclusive: int,
    full_reforward: Any,
    repetition_penalties: Sequence[float] = REPETITION_PENALTY_STRATA,
    expected_layer_count: int | None = None,
    relaxed_selected_logprob_max_abs_diff: float | None = None,
) -> dict[str, Any]:
    """Admit cache scoring against an independent ``use_cache=False`` reforward.

    ``full_reforward`` is ``Callable[[Sequence[int]], torch.Tensor]``: given a
    complete literal token sequence, it performs an independent forward with
    no cache and returns the raw logits vector ``[vocab]`` at the sequence's
    last position. This is the audit-mandated real-parity check for the
    prefix-KV branching engine -- it must run and pass before any cache-branch
    score from a live model is trusted; it is orthogonal to (and does not
    replace) the fake-backend order-invariance tests, which only prove the
    isolation *logic* is sound, not that a real model's cache honors it.

    By default, parity is required at all four coordinate depths using the
    historical full-vocabulary ``torch.allclose(atol=1e-6, rtol=1e-5)``
    semantics. An explicit ``relaxed_selected_logprob_max_abs_diff`` may
    admit a strict-parity failure only when every raw/RP1.0/RP1.1 view keeps
    the same coordinate-domain argmax and the selected coordinate token's
    full-vocabulary-normalized logprob stays within that bound. Strict
    full-vocabulary results remain separately recorded in either mode.
    """

    relaxed_tolerance: float | None = None
    if relaxed_selected_logprob_max_abs_diff is not None:
        if isinstance(relaxed_selected_logprob_max_abs_diff, bool):
            raise ValueError("relaxed cache admission tolerance must be numeric")
        relaxed_tolerance = float(relaxed_selected_logprob_max_abs_diff)
        if not math.isfinite(relaxed_tolerance) or relaxed_tolerance < 0.0:
            raise ValueError(
                "relaxed cache admission tolerance must be finite and non-negative"
            )

    prefix = [int(v) for v in prefix_token_ids]
    lo = int(coordinate_token_id_start)
    hi = int(coordinate_token_id_end_exclusive)
    selected_tokens = {
        "x1": int(x1_token_id),
        "y1": int(y1_token_id),
        "x2": int(x2_token_id),
        "y2": int(y2_token_id),
    }
    if lo < 0 or hi <= lo:
        raise ValueError("cache admission requires a non-empty coordinate token domain")
    invalid_selected = {
        slot: token_id
        for slot, token_id in selected_tokens.items()
        if token_id < lo or token_id >= hi
    }
    if invalid_selected:
        raise ValueError(
            "cache admission selected coordinate tokens must lie inside the coordinate domain"
        )

    entry_length = backend.cache_length
    steps: list[dict[str, Any]] = []
    comparison_inputs: dict[
        str, tuple[str, torch.Tensor, torch.Tensor, tuple[int, ...]]
    ] = {}

    def _compare(
        *,
        depth: str,
        predicted_slot: str,
        cache_logits: torch.Tensor,
        ground_truth_logits: torch.Tensor,
        running_context_token_ids: Sequence[int],
    ) -> None:
        a = cache_logits.to(torch.float32)
        b = ground_truth_logits.to(torch.float32)
        if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
            _fail(
                "cache admission requires matching rank-1 full-vocabulary logits",
                depth=depth,
                cache_shape=list(a.shape),
                reforward_shape=list(b.shape),
            )
        if hi > int(a.shape[0]):
            _fail(
                "coordinate token domain exceeds the cache-admission vocabulary",
                coordinate_token_id_end_exclusive=hi,
                vocabulary_size=int(a.shape[0]),
            )
        max_abs_diff = float((a - b).abs().max().item())
        within_tolerance = bool(
            torch.allclose(a, b, atol=CACHE_PARITY_ATOL, rtol=CACHE_PARITY_RTOL)
        )
        steps.append(
            {
                "step": f"{depth}_predicts_{predicted_slot}",
                "depth": depth,
                "predicted_slot": predicted_slot,
                "max_abs_diff": max_abs_diff,
                "atol": CACHE_PARITY_ATOL,
                "rtol": CACHE_PARITY_RTOL,
                "within_tolerance": within_tolerance,
            }
        )
        comparison_inputs[predicted_slot] = (
            depth,
            a,
            b,
            tuple(int(value) for value in running_context_token_ids),
        )

    _compare(
        depth="root",
        predicted_slot="x1",
        cache_logits=prefill_logits,
        ground_truth_logits=full_reforward(prefix),
        running_context_token_ids=prefix,
    )

    with BranchCursor(backend) as branch:
        cache_y1 = branch.step([x1_token_id])[-1]
        _compare(
            depth="post_x1",
            predicted_slot="y1",
            cache_logits=cache_y1,
            ground_truth_logits=full_reforward([*prefix, x1_token_id]),
            running_context_token_ids=[*prefix, x1_token_id],
        )

        cache_x2 = branch.step([y1_token_id])[-1]
        ground_truth_x2 = full_reforward([*prefix, x1_token_id, y1_token_id])
        _compare(
            depth="post_y1",
            predicted_slot="x2",
            cache_logits=cache_x2,
            ground_truth_logits=ground_truth_x2,
            running_context_token_ids=[*prefix, x1_token_id, y1_token_id],
        )

        cache_y2 = branch.step([x2_token_id])[-1]
        _compare(
            depth="post_x2",
            predicted_slot="y2",
            cache_logits=cache_y2,
            ground_truth_logits=full_reforward(
                [*prefix, x1_token_id, y1_token_id, x2_token_id]
            ),
            running_context_token_ids=[
                *prefix,
                x1_token_id,
                y1_token_id,
                x2_token_id,
            ],
        )

    length_after_branch_exit = backend.cache_length
    if length_after_branch_exit != entry_length:
        _fail(
            "parity-gate branch did not crop back to its entry length",
            expected=entry_length,
            observed=length_after_branch_exit,
        )

    observed_layer_count = getattr(backend, "layer_count", None)
    layer_count_matches_expected = (
        None
        if observed_layer_count is None or expected_layer_count is None
        else observed_layer_count == expected_layer_count
    )
    cache_layers = {
        "observed_layer_count": observed_layer_count,
        "expected_layer_count": expected_layer_count,
        "layer_count_matches_expected": layer_count_matches_expected,
    }

    forward_first = _branch_single_step_readout(backend, x1_token_id)
    _branch_single_step_readout(
        backend, y1_token_id
    )  # unrelated sibling branch, cropped on exit
    reverse_after_sibling = _branch_single_step_readout(backend, x1_token_id)
    order_diff = float(
        (forward_first.to(torch.float32) - reverse_after_sibling.to(torch.float32))
        .abs()
        .max()
        .item()
    )
    order_invariance = {
        "max_abs_diff": order_diff,
        "atol": CACHE_PARITY_ATOL,
        "rtol": CACHE_PARITY_RTOL,
        "within_tolerance": bool(
            torch.allclose(
                forward_first.to(torch.float32),
                reverse_after_sibling.to(torch.float32),
                atol=CACHE_PARITY_ATOL,
                rtol=CACHE_PARITY_RTOL,
            )
        ),
        "cache_length_restored": backend.cache_length == entry_length,
    }

    from transformers import RepetitionPenaltyLogitsProcessor

    behavior_checks: list[dict[str, Any]] = []
    policy_full_vocab_parity: dict[str, dict[str, Any]] = {}
    for slot in COORD_SLOTS:
        depth, raw_cache, raw_reforward, running_context = comparison_inputs[slot]
        view_scores: list[tuple[str, torch.Tensor, torch.Tensor]] = [
            ("raw", raw_cache, raw_reforward)
        ]
        context_ids = torch.tensor([list(running_context)], dtype=torch.long)
        for penalty in repetition_penalties:
            processor = RepetitionPenaltyLogitsProcessor(penalty=float(penalty))
            cache_processed = processor(
                cast(torch.LongTensor, context_ids),
                cast(torch.FloatTensor, raw_cache.clone().unsqueeze(0)),
            )[0]
            reforward_processed = processor(
                cast(torch.LongTensor, context_ids),
                cast(torch.FloatTensor, raw_reforward.clone().unsqueeze(0)),
            )[0]
            view_key = _policy_view_key(penalty)
            view_scores.append((view_key, cache_processed, reforward_processed))
            policy_full_vocab_parity.setdefault(view_key, {})[slot] = {
                "depth": depth,
                "predicted_slot": slot,
                "max_abs_diff": float(
                    (cache_processed - reforward_processed).abs().max().item()
                ),
                "atol": CACHE_PARITY_ATOL,
                "rtol": CACHE_PARITY_RTOL,
                "within_tolerance": bool(
                    torch.allclose(
                        cache_processed,
                        reforward_processed,
                        atol=CACHE_PARITY_ATOL,
                        rtol=CACHE_PARITY_RTOL,
                    )
                ),
            }

        for view_key, cache_scores, reforward_scores in view_scores:
            cache_logprobs = torch.log_softmax(cache_scores, dim=-1)
            reforward_logprobs = torch.log_softmax(reforward_scores, dim=-1)
            cache_coordinate = cache_logprobs[lo:hi]
            reforward_coordinate = reforward_logprobs[lo:hi]
            finite = bool(
                torch.isfinite(cache_coordinate).all()
                and torch.isfinite(reforward_coordinate).all()
            )
            selected_token_id = selected_tokens[slot]
            cache_argmax = (
                lo + int(torch.argmax(cache_coordinate).item()) if finite else None
            )
            reforward_argmax = (
                lo + int(torch.argmax(reforward_coordinate).item()) if finite else None
            )
            selected_cache = cache_logprobs[selected_token_id]
            selected_reforward = reforward_logprobs[selected_token_id]
            selected_finite = bool(
                torch.isfinite(selected_cache) and torch.isfinite(selected_reforward)
            )
            selected_delta = (
                float((selected_cache - selected_reforward).abs().item())
                if selected_finite
                else None
            )
            coordinate_delta = (
                float((cache_coordinate - reforward_coordinate).abs().max().item())
                if finite
                else None
            )
            strict_selected_stable = bool(
                selected_finite
                and torch.allclose(
                    selected_cache,
                    selected_reforward,
                    atol=CACHE_PARITY_ATOL,
                    rtol=CACHE_PARITY_RTOL,
                )
            )
            relaxed_selected_stable = bool(
                selected_delta is not None
                and relaxed_tolerance is not None
                and selected_delta <= relaxed_tolerance
            )
            behavior_checks.append(
                {
                    "depth": depth,
                    "predicted_slot": slot,
                    "view": view_key,
                    "selected_coordinate_token_id": selected_token_id,
                    "finite": finite and selected_finite,
                    "coordinate_argmax_cache_token_id": cache_argmax,
                    "coordinate_argmax_reforward_token_id": reforward_argmax,
                    "coordinate_argmax_parity": (
                        finite and cache_argmax == reforward_argmax
                    ),
                    "coordinate_logprob_max_abs_diff": coordinate_delta,
                    "selected_coordinate_logprob_cache": (
                        float(selected_cache.item()) if selected_finite else None
                    ),
                    "selected_coordinate_logprob_reforward": (
                        float(selected_reforward.item()) if selected_finite else None
                    ),
                    "selected_coordinate_logprob_abs_diff": selected_delta,
                    "strict_selected_logprob_within_tolerance": strict_selected_stable,
                    "relaxed_selected_logprob_within_tolerance": (
                        relaxed_selected_stable
                    ),
                }
            )

    raw_full_vocab_passed = all(step["within_tolerance"] for step in steps)
    policy_full_vocab_passed = all(
        entry["within_tolerance"]
        for per_view in policy_full_vocab_parity.values()
        for entry in per_view.values()
    )
    all_coordinate_argmax_parity = all(
        entry["coordinate_argmax_parity"] for entry in behavior_checks
    )
    strict_selected_logprob_stability = all(
        entry["strict_selected_logprob_within_tolerance"] for entry in behavior_checks
    )
    relaxed_selected_logprob_stability = bool(
        relaxed_tolerance is not None
        and all(
            entry["relaxed_selected_logprob_within_tolerance"]
            for entry in behavior_checks
        )
    )
    structural_passed = bool(
        order_invariance["within_tolerance"]
        and order_invariance["cache_length_restored"]
        and layer_count_matches_expected is not False
    )
    strict_passed = bool(
        raw_full_vocab_passed
        and policy_full_vocab_passed
        and all_coordinate_argmax_parity
        and strict_selected_logprob_stability
        and structural_passed
    )
    relaxed_passed = bool(
        relaxed_tolerance is not None
        and all_coordinate_argmax_parity
        and relaxed_selected_logprob_stability
        and structural_passed
    )
    requested_mode = (
        RELAXED_CACHE_ADMISSION_MODE
        if relaxed_tolerance is not None
        else STRICT_CACHE_ADMISSION_MODE
    )
    if strict_passed:
        effective_mode = STRICT_CACHE_ADMISSION_MODE
        passed = True
    elif relaxed_passed:
        effective_mode = RELAXED_CACHE_ADMISSION_MODE
        passed = True
    else:
        effective_mode = UNCACHED_CACHE_ADMISSION_MODE
        passed = False

    selected_deltas = [
        float(entry["selected_coordinate_logprob_abs_diff"])
        for entry in behavior_checks
        if entry["selected_coordinate_logprob_abs_diff"] is not None
    ]
    coordinate_deltas = [
        float(entry["coordinate_logprob_max_abs_diff"])
        for entry in behavior_checks
        if entry["coordinate_logprob_max_abs_diff"] is not None
    ]
    admission_policy = {
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "requested_thresholds": {
            "full_vocabulary_atol": CACHE_PARITY_ATOL,
            "full_vocabulary_rtol": CACHE_PARITY_RTOL,
            "selected_coordinate_logprob_max_abs_diff": relaxed_tolerance,
            "coordinate_argmax_requirement": "exact_token_id_parity",
        },
        "effective_thresholds": (
            {
                "full_vocabulary_atol": CACHE_PARITY_ATOL,
                "full_vocabulary_rtol": CACHE_PARITY_RTOL,
                "selected_coordinate_logprob_max_abs_diff": None,
                "coordinate_argmax_requirement": "exact_token_id_parity",
            }
            if effective_mode == STRICT_CACHE_ADMISSION_MODE
            else {
                "full_vocabulary_atol": None,
                "full_vocabulary_rtol": None,
                "selected_coordinate_logprob_max_abs_diff": (
                    relaxed_tolerance
                    if effective_mode == RELAXED_CACHE_ADMISSION_MODE
                    else None
                ),
                "coordinate_argmax_requirement": "exact_token_id_parity",
            }
        ),
        "strict_full_vocabulary_parity_passed": (
            raw_full_vocab_passed and policy_full_vocab_passed
        ),
        "all_coordinate_argmax_parity": all_coordinate_argmax_parity,
        "strict_selected_coordinate_logprob_stability_passed": (
            strict_selected_logprob_stability
        ),
        "relaxed_selected_coordinate_logprob_stability_passed": (
            relaxed_selected_logprob_stability
        ),
        "observed_max_raw_logit_abs_diff": max(
            float(step["max_abs_diff"]) for step in steps
        ),
        "observed_max_policy_logit_abs_diff": max(
            (
                float(entry["max_abs_diff"])
                for per_view in policy_full_vocab_parity.values()
                for entry in per_view.values()
            ),
            default=0.0,
        ),
        "observed_max_coordinate_logprob_abs_diff": (
            max(coordinate_deltas) if coordinate_deltas else None
        ),
        "observed_max_selected_coordinate_logprob_abs_diff": (
            max(selected_deltas) if selected_deltas else None
        ),
    }
    return {
        "status": "passed" if passed else "failed",
        "cache_admission_status": "passed" if passed else "failed",
        "strict_full_vocabulary_parity_status": (
            "passed"
            if raw_full_vocab_passed and policy_full_vocab_passed
            else "failed"
        ),
        "atol": CACHE_PARITY_ATOL,
        "rtol": CACHE_PARITY_RTOL,
        "raw_logit_parity_steps": steps,
        "repetition_penalty_processor_parity": policy_full_vocab_parity,
        "coordinate_behavior_checks": behavior_checks,
        "admission_policy": admission_policy,
        "branch_reverse_order_invariance": order_invariance,
        "cache_layers": cache_layers,
    }


def select_scoring_backend_from_parity(
    parity_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the one scoring backend for the whole run before rows exist."""

    status = parity_gate.get("status")
    if status == "passed":
        policy = parity_gate.get("admission_policy")
        if isinstance(policy, Mapping):
            requested_mode = policy.get("requested_mode")
            effective_mode = policy.get("effective_mode")
            strict_passed = policy.get("strict_full_vocabulary_parity_passed")
            if effective_mode == RELAXED_CACHE_ADMISSION_MODE:
                if (
                    requested_mode != RELAXED_CACHE_ADMISSION_MODE
                    or policy.get("all_coordinate_argmax_parity") is not True
                    or policy.get(
                        "relaxed_selected_coordinate_logprob_stability_passed"
                    )
                    is not True
                ):
                    _fail(
                        "relaxed cache backend selection lacks an explicit request or passed behavior checks",
                        admission_policy=dict(policy),
                    )
            elif (
                effective_mode != STRICT_CACHE_ADMISSION_MODE
                or strict_passed is not True
            ):
                _fail(
                    "passed cache backend selection has an inconsistent effective admission mode",
                    admission_policy=dict(policy),
                )
        return {
            "selected_backend": KV_CACHE_SCORING_BACKEND,
            "cache_enabled": True,
            "use_cache": True,
            "fallback_trigger": None,
        }
    if status == "failed":
        return {
            "selected_backend": FULL_REFORWARD_SCORING_BACKEND,
            "cache_enabled": False,
            "use_cache": False,
            "fallback_trigger": PARITY_FAILURE_FALLBACK_TRIGGER,
        }
    _fail(
        "mandatory cache parity must resolve to passed or failed before scoring",
        observed_status=status,
    )


def build_scoring_backend_admission(
    *,
    parity_gate: Mapping[str, Any],
    selection: Mapping[str, Any],
    score_row_count: int,
    per_context_group_accounting: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal backend selection and reconstructable per-group forward accounting."""

    selected_backend = selection.get("selected_backend")
    expected = select_scoring_backend_from_parity(parity_gate)
    if dict(selection) != expected:
        _fail(
            "scoring backend selection drifted from the mandatory parity result",
            expected=expected,
            observed=dict(selection),
        )
    accounting = [dict(entry) for entry in per_context_group_accounting]
    if not accounting:
        _fail("scoring backend admission requires per-context/group accounting")
    if any(entry.get("scoring_backend") != selected_backend for entry in accounting):
        _fail("scoring backend accounting mixes backend identities")

    aggregate_by_depth: dict[str, dict[str, int]] = {
        "logical_token_step_requests": {},
        "actual_forward_calls": {},
        "memo_hits": {},
        "memo_entries": {},
    }
    aggregate: dict[str, Any] = {
        "context_group_count": len(accounting),
        "root_calls": 0,
        "logical_token_step_requests": 0,
        "actual_forward_calls": 0,
        "memo_hits": 0,
        "memo_entries": 0,
    }
    depth_sources = {
        "logical_token_step_requests": "logical_token_step_requests_by_depth",
        "actual_forward_calls": "actual_forward_calls_by_depth",
        "memo_hits": "memo_hits_by_depth",
        "memo_entries": "memo_entries_by_depth",
    }
    for entry in accounting:
        for scalar in (
            "root_calls",
            "logical_token_step_requests",
            "actual_forward_calls",
        ):
            value = entry.get(scalar)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _fail(
                    "scoring backend accounting scalar must be a non-negative integer",
                    field=scalar,
                    observed=value,
                )
            aggregate[scalar] += value
        for target, source in depth_sources.items():
            depth_counts = entry.get(source)
            if not isinstance(depth_counts, Mapping):
                _fail("scoring backend depth accounting must be a mapping", field=source)
            depth_total = 0
            for depth, value in depth_counts.items():
                if (
                    not isinstance(depth, str)
                    or not depth.isdigit()
                    or isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    _fail("invalid scoring backend depth accounting", field=source)
                aggregate_by_depth[target][depth] = (
                    aggregate_by_depth[target].get(depth, 0) + value
                )
                depth_total += value
            if target in {"memo_hits", "memo_entries"}:
                aggregate[target] += depth_total
    aggregate["by_depth"] = aggregate_by_depth

    row_count = int(score_row_count)
    if row_count < 0:
        raise ValueError("score_row_count must be non-negative")
    cache_rows = row_count if selected_backend == KV_CACHE_SCORING_BACKEND else 0
    uncached_rows = row_count - cache_rows
    cache_policy = (
        dict(parity_gate["admission_policy"])
        if isinstance(parity_gate.get("admission_policy"), Mapping)
        else None
    )
    decision_use = (
        PROBE_ONLY_SCORE_USE
        if cache_policy is not None
        and cache_policy.get("effective_mode") == RELAXED_CACHE_ADMISSION_MODE
        else DECISION_BEARING_SCORE_USE
    )
    admission = {
        "status": "passed",
        "decision_use": decision_use,
        "selected_backend": selected_backend,
        "cache_enabled": selection["cache_enabled"],
        "use_cache": selection["use_cache"],
        "fallback_trigger": selection["fallback_trigger"],
        "parity_status": parity_gate["status"],
        "atol": CACHE_PARITY_ATOL,
        "rtol": CACHE_PARITY_RTOL,
        "all_score_rows_backend": selected_backend,
        "score_row_count": row_count,
        "cache_score_row_count": cache_rows,
        "uncached_score_row_count": uncached_rows,
        "backend_mixing_detected": False,
        "cache_admission_policy": cache_policy,
        "cache_admission_scope": (
            dict(parity_gate["admission_scope"])
            if isinstance(parity_gate.get("admission_scope"), Mapping)
            else None
        ),
        "forward_accounting": {
            "per_context_group": accounting,
            "aggregate": aggregate,
        },
    }
    admission["sha256"] = sha256_json(admission)
    return admission


def execution_architecture_for_admission(
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    selected = admission.get("selected_backend")
    if selected == KV_CACHE_SCORING_BACKEND:
        cache_policy = admission.get("cache_admission_policy")
        effective_mode = (
            cache_policy.get("effective_mode")
            if isinstance(cache_policy, Mapping)
            else STRICT_CACHE_ADMISSION_MODE
        )
        if effective_mode == RELAXED_CACHE_ADMISSION_MODE:
            strategy = (
                "explicit relaxed coordinate-behavior admission passed after strict "
                "full-vocabulary parity failed; prefill once for the single admitted exact "
                "context/history group and branch every score row through a shared "
                "DynamicCache with crop-on-exit isolation"
            )
        else:
            strategy = (
                "strict full-vocabulary cache parity passed; prefill once per context/history "
                "group and branch every score row through a shared DynamicCache with "
                "crop-on-exit isolation"
            )
        return {
            "one_full_reforward_per_candidate": False,
            "strategy": strategy,
            "selected_backend": selected,
            "cache_admission_mode": effective_mode,
        }
    if selected == FULL_REFORWARD_SCORING_BACKEND:
        entries = (
            admission.get("forward_accounting", {}).get("per_context_group", [])
            if isinstance(admission.get("forward_accounting"), Mapping)
            else []
        )
        configured_batch_size = max(
            (
                int(entry.get("configured_full_reforward_batch_size", 1))
                for entry in entries
                if isinstance(entry, Mapping)
            ),
            default=1,
        )
        if configured_batch_size > 1:
            strategy = (
                "mandatory cache parity failed; discard cache logits and score every row through "
                "literal use_cache=False full reforwards, co-scheduling equal-length same-context "
                "sequences in GPU batches while retaining exact depth-1/depth-2 memoization"
            )
        else:
            strategy = (
                "mandatory cache parity failed; discard cache logits and score every row through "
                "literal use_cache=False full reforwards with exact depth-1/depth-2 memoization"
            )
        return {
            "one_full_reforward_per_candidate": False,
            "strategy": strategy,
            "selected_backend": selected,
            "configured_full_reforward_batch_size": configured_batch_size,
        }
    _fail("unknown selected scoring backend in execution architecture", observed=selected)


def _prefetch_backend_suffixes(
    backend: CacheBackend, suffixes: Sequence[Sequence[int]]
) -> None:
    """Use optional literal-reforward batching without widening CacheBackend."""

    prefetch = getattr(backend, "prefetch_suffixes", None)
    if callable(prefetch):
        prefetch(suffixes)


def _candidate_reforward_suffixes(candidate: CandidateRow) -> list[list[int]]:
    """Return the exact relative prefixes a restricted candidate will consume."""

    if candidate.coord_token_ids is not None:
        tokens = [int(value) for value in candidate.coord_token_ids[:3]]
    else:
        tokens = [int(value) for value in candidate.fixed_coord_token_ids]
    return [tokens[:depth] for depth in range(1, len(tokens) + 1)]


def select_mandatory_parity_probe(
    candidates: Sequence[CandidateRow],
) -> CandidateRow:
    """Choose the deterministic complete-box probe required before row scoring."""

    probe = min(
        (
            candidate
            for candidate in candidates
            if candidate.request_kind == "complete_box"
            and candidate.coord_token_ids is not None
        ),
        key=lambda candidate: candidate.candidate_id,
        default=None,
    )
    if probe is None:
        _fail(
            "mandatory cache parity requires a complete_box probe before any score row can be emitted"
        )
    return probe


# ---------------------------------------------------------------------------
# Per-candidate scoring against a prefilled branch
# ---------------------------------------------------------------------------


def score_complete_box_candidate(
    *,
    backend: CacheBackend,
    prefill_logits: torch.Tensor,
    coord_token_ids: Sequence[int],
    attestation: AttestationContext,
    running_context_token_ids: Sequence[int],
    repetition_penalties: Sequence[float] = REPETITION_PENALTY_STRATA,
) -> dict[str, Any]:
    """Score one literal (x1, y1, x2, y2) box with three cache-branch steps, not four forwards.

    ``x1``'s distribution already came for free from the shared prefill (its
    logits predict the token right after ``box_start``); each subsequent
    coordinate's distribution comes for free from the *previous* step's
    returned logits. The branch is cropped back to the prefill point on exit.
    """

    x1, y1, x2, y2 = (int(v) for v in coord_token_ids)
    context = list(running_context_token_ids)
    with BranchCursor(backend) as branch:
        x1_view = _single_bin_view(
            prefill_logits,
            token_id=x1,
            attestation=attestation,
            running_context_token_ids=context,
            repetition_penalties=repetition_penalties,
        )
        step1 = branch.step([x1])[-1]
        y1_view = _single_bin_view(
            step1,
            token_id=y1,
            attestation=attestation,
            running_context_token_ids=[*context, x1],
            repetition_penalties=repetition_penalties,
        )
        step2 = branch.step([y1])[-1]
        x2_view = _single_bin_view(
            step2,
            token_id=x2,
            attestation=attestation,
            running_context_token_ids=[*context, x1, y1],
            repetition_penalties=repetition_penalties,
        )
        step3 = branch.step([x2])[-1]
        y2_view = _single_bin_view(
            step3,
            token_id=y2,
            attestation=attestation,
            running_context_token_ids=[*context, x1, y1, x2],
            repetition_penalties=repetition_penalties,
        )
    return combine_complete_box(
        {"x1": x1_view, "y1": y1_view, "x2": x2_view, "y2": y2_view}
    )


def score_dense_scan_candidate(
    *,
    backend: CacheBackend,
    prefill_logits: torch.Tensor,
    fixed_coord_token_ids: Sequence[int],
    scan_slot: str,
    coordinate_token_id_start: int,
    coordinate_token_id_end_exclusive: int,
    attestation: AttestationContext,
    running_context_token_ids: Sequence[int],
    repetition_penalties: Sequence[float] = REPETITION_PENALTY_STRATA,
) -> dict[str, Any]:
    """Densely score every declared coordinate bin at ``scan_slot`` given fixed earlier coordinates.

    At most three branch steps (one per fixed coordinate) precede the single
    full-vocabulary read; scanning the complete y1 vocabulary at an admitted
    x1 anchor costs exactly one branch step, never one forward per y1
    candidate.
    """

    fixed = [int(v) for v in fixed_coord_token_ids]
    if COORD_SLOTS[len(fixed)] != scan_slot:
        raise ValueError("scan_slot does not follow fixed_coord_token_ids")
    context = list(running_context_token_ids)
    with BranchCursor(backend) as branch:
        logits_at_slot = prefill_logits
        for token in fixed:
            logits_at_slot = branch.step([token])[-1]
            context.append(token)
        return score_coordinate_position(
            logits_at_slot,
            coordinate_token_id_start=coordinate_token_id_start,
            coordinate_token_id_end_exclusive=coordinate_token_id_end_exclusive,
            attestation=attestation,
            running_context_token_ids=context,
            repetition_penalties=repetition_penalties,
        )


def _selected_bin_view(scan: Mapping[str, Any], coordinate_bin: int) -> dict[str, Any]:
    raw = scan["raw"]
    policy = scan["auxiliary_policy"]
    return {
        "raw": {
            "bin_logprobs": [raw["bin_logprobs"][coordinate_bin]],
            "vocab_attestation": raw["vocab_attestation"],
        },
        "auxiliary_policy": {
            key: {
                "bin_logprobs": [value["bin_logprobs"][coordinate_bin]],
                "vocab_attestation": value["vocab_attestation"],
            }
            for key, value in policy.items()
        },
    }


def deterministic_farthest_point_anchors(
    candidates: Sequence[tuple[int, int, float]], *, minimum_distance_bins: float
) -> tuple[list[tuple[int, int, float]], dict[str, Any]]:
    """Apply the lead-frozen global anchor selector with exact deterministic ties."""

    if minimum_distance_bins <= 0:
        raise ValueError("minimum_distance_bins must be positive")
    unique: dict[tuple[int, int], float] = {}
    for x1, y1, score in candidates:
        if not math.isfinite(score):
            raise ValueError("free-tree anchor scores must be finite")
        key = (int(x1), int(y1))
        unique[key] = max(float(score), unique.get(key, -math.inf))
    if not unique:
        return [], {
            "candidate_count": 0,
            "selected_count": 0,
            "stop_reason": "empty_candidate_pool",
        }

    pool = [(x1, y1, score) for (x1, y1), score in unique.items()]
    first = min(pool, key=lambda item: (-item[2], item[0], item[1]))
    selected = [first]
    remaining = [item for item in pool if item[:2] != first[:2]]
    final_best_distance: float | None = None
    stop_reason = "candidate_pool_exhausted"
    while remaining:
        ranked: list[tuple[float, float, int, int, tuple[int, int, float]]] = []
        for item in remaining:
            minimum_distance = min(
                math.hypot(item[0] - chosen[0], item[1] - chosen[1])
                for chosen in selected
            )
            ranked.append((-minimum_distance, -item[2], item[0], item[1], item))
        best = min(ranked)
        final_best_distance = -best[0]
        if final_best_distance < minimum_distance_bins:
            stop_reason = "best_remaining_minimum_distance_below_threshold"
            break
        selected.append(best[-1])
        chosen_xy = best[-1][:2]
        remaining = [item for item in remaining if item[:2] != chosen_xy]
    return selected, {
        "candidate_count": len(pool),
        "selected_count": len(selected),
        "stop_reason": stop_reason,
        "best_remaining_minimum_distance_bins": final_best_distance,
    }


def execute_free_coordinate_tree(
    *,
    request: CandidateRow,
    backend: CacheBackend,
    prefill_logits: torch.Tensor,
    rules: DecisionRules,
    attestation: AttestationContext,
    repetition_penalties: Sequence[float] = REPETITION_PENALTY_STRATA,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute the frozen bounded free tree and return complete boxes plus its receipt."""

    if request.record_type != "free_coordinate_tree_root_request":
        raise ValueError("execute_free_coordinate_tree requires a v2 free root request")
    budget = rules.free_search_budget
    if budget != FREE_SEARCH_BUDGET:
        _fail("free coordinate-tree execution received a non-frozen budget")
    materializer = rules.materializer_rules
    if materializer is None:
        _fail("free coordinate-tree execution requires parsed v2 materializer rules")
    lo = rules.schema_tokens["coordinate_token_id_start"]
    landscape = materializer.landscape
    if landscape.coordinate_min != 0:
        _fail("free coordinate-tree execution requires a zero-based coordinate domain")
    bin_count = landscape.coordinate_max + 1
    hi = lo + bin_count
    if rules.contract_mode == "production" and bin_count != 1000:
        _fail(
            "free coordinate-tree production execution requires coordinate bins 0..999"
        )

    root_scan = score_coordinate_position(
        prefill_logits,
        coordinate_token_id_start=lo,
        coordinate_token_id_end_exclusive=hi,
        attestation=attestation,
        running_context_token_ids=request.prefix_token_ids,
        repetition_penalties=repetition_penalties,
    )
    root_raw = root_scan["raw"]["bin_logprobs"]
    valid_x1 = range(0, bin_count - 1)
    ranked_x1 = sorted(valid_x1, key=lambda value: (-root_raw[value], value))
    selected_x1 = ranked_x1[: int(budget["x1_branch_budget"])]
    _prefetch_backend_suffixes(backend, [[lo + x1] for x1 in selected_x1])

    y1_scans: dict[int, Mapping[str, Any]] = {}
    y1_candidates: list[tuple[int, int, float]] = []
    y1_counts: dict[str, int] = {}
    y1_shortfalls: dict[str, str] = {}
    for x1 in selected_x1:
        x1_token = lo + x1
        with BranchCursor(backend) as branch:
            y1_logits = branch.step([x1_token])[-1]
            y1_scan = score_coordinate_position(
                y1_logits,
                coordinate_token_id_start=lo,
                coordinate_token_id_end_exclusive=hi,
                attestation=attestation,
                running_context_token_ids=[*request.prefix_token_ids, x1_token],
                repetition_penalties=repetition_penalties,
            )
        y1_scans[x1] = y1_scan
        y1_raw = y1_scan["raw"]["bin_logprobs"]
        ranked_y1 = sorted(
            range(0, bin_count - 1),
            key=lambda value: (-(root_raw[x1] + y1_raw[value]), value),
        )
        retained_y1 = ranked_y1[: int(budget["y1_branch_budget_per_x1"])]
        y1_counts[str(x1)] = len(retained_y1)
        if len(retained_y1) < int(budget["y1_branch_budget_per_x1"]):
            y1_shortfalls[str(x1)] = "fewer_valid_y1_bins_than_per_x1_budget"
        y1_candidates.extend(
            (x1, y1, float(root_raw[x1] + y1_raw[y1])) for y1 in retained_y1
        )

    diversification = budget["spatial_diversification"]
    selected_anchors, diversification_receipt = deterministic_farthest_point_anchors(
        y1_candidates,
        minimum_distance_bins=float(diversification["minimum_center_distance_bins"]),
    )
    _prefetch_backend_suffixes(
        backend,
        [[lo + x1, lo + y1] for x1, y1, _joint in selected_anchors],
    )

    rows: list[dict[str, Any]] = []
    extent_counts: dict[str, int] = {}
    extent_shortfalls: dict[str, str] = {}
    for x1, y1, joint_anchor_logprob in selected_anchors:
        x1_token = lo + x1
        y1_token = lo + y1
        with BranchCursor(backend) as anchor_branch:
            anchor_branch.step([x1_token])
            x2_logits = anchor_branch.step([y1_token])[-1]
            x2_scan = score_coordinate_position(
                x2_logits,
                coordinate_token_id_start=lo,
                coordinate_token_id_end_exclusive=hi,
                attestation=attestation,
                running_context_token_ids=[
                    *request.prefix_token_ids,
                    x1_token,
                    y1_token,
                ],
                repetition_penalties=repetition_penalties,
            )
            x2_raw = x2_scan["raw"]["bin_logprobs"]
            valid_x2 = range(x1 + 1, bin_count)
            ranked_x2 = sorted(valid_x2, key=lambda value: (-x2_raw[value], value))
            selected_x2 = ranked_x2[: int(budget["extent_branch_budget_per_anchor"])]
            _prefetch_backend_suffixes(
                backend,
                [
                    [x1_token, y1_token, lo + x2]
                    for x2 in selected_x2
                ],
            )
            anchor_key = f"{x1},{y1}"
            extent_counts[anchor_key] = len(selected_x2)
            if len(selected_x2) < int(budget["extent_branch_budget_per_anchor"]):
                extent_shortfalls[anchor_key] = "fewer_valid_x2_bins_than_extent_budget"
            for x2 in selected_x2:
                x2_token = lo + x2
                with BranchCursor(backend) as extent_branch:
                    y2_logits = extent_branch.step([x2_token])[-1]
                    y2_scan = score_coordinate_position(
                        y2_logits,
                        coordinate_token_id_start=lo,
                        coordinate_token_id_end_exclusive=hi,
                        attestation=attestation,
                        running_context_token_ids=[
                            *request.prefix_token_ids,
                            x1_token,
                            y1_token,
                            x2_token,
                        ],
                        repetition_penalties=repetition_penalties,
                    )
                y2_raw = y2_scan["raw"]["bin_logprobs"]
                valid_y2 = range(y1 + 1, bin_count)
                if not valid_y2:
                    continue
                y2 = min(valid_y2, key=lambda value: (-y2_raw[value], value))
                y2_token = lo + y2
                scored = combine_complete_box(
                    {
                        "x1": _selected_bin_view(root_scan, x1),
                        "y1": _selected_bin_view(y1_scans[x1], y1),
                        "x2": _selected_bin_view(x2_scan, x2),
                        "y2": _selected_bin_view(y2_scan, y2),
                    }
                )
                coordinate_bins = [x1, y1, x2, y2]
                coordinate_tokens = [x1_token, y1_token, x2_token, y2_token]
                free_candidate_id = "free-tree-box:sha256:" + sha256_json(
                    {
                        "request_id": request.candidate_id,
                        "coordinate_bin_values": coordinate_bins,
                    }
                )
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "surface": "free_coordinate_tree",
                        "candidate_id": free_candidate_id,
                        "free_tree_request_id": request.candidate_id,
                        "diagnostic_owner_id": request.diagnostic_owner_id,
                        "gt_owner_id": request.gt_owner_id,
                        "image_id": request.image_id,
                        "context_id": request.context_id,
                        "coordinate_bin_values": coordinate_bins,
                        "coord_token_ids": coordinate_tokens,
                        "raw_model_logprob": scored["raw"],
                        "auxiliary_policy_scores": scored["auxiliary_policy"],
                        "anchor_joint_raw_logprob": joint_anchor_logprob,
                        "bounded_search_null_semantics": "non_evidence_for_absence",
                        "positive_target_support_semantics": "decision_bearing_blocks_C",
                    }
                )

    receipt = {
        "schema_version": "sorted_owner_basin_free_tree_execution.v1",
        "surface": "free_coordinate_tree",
        "request_id": request.candidate_id,
        "status": "executed",
        "selector": {
            "x1": "descending_raw_fp32_logprob_tie_ascending_x1",
            "y1_candidate_pool": "descending_joint_raw_fp32_logprob_tie_ascending_y1_per_x1",
            "anchor_seed": "highest_joint_raw_fp32_logprob_tie_ascending_x1_then_y1",
            "anchor_iteration": "maximize_minimum_euclidean_xy_distance_tie_higher_joint_then_ascending_x1_y1",
            "extent_x2": "descending_raw_fp32_logprob_valid_x2_tie_ascending_x2",
            "extent_y2": "single_highest_raw_fp32_logprob_valid_y2_tie_ascending_y2",
            "exhaustive_x2_y2_pair_ranking": False,
        },
        "budget": dict(budget),
        "counts": {
            "valid_x1_count": bin_count - 1,
            "selected_x1_count": len(selected_x1),
            "y1_candidate_count_by_x1": y1_counts,
            "anchor_candidate_count": len(y1_candidates),
            "selected_anchor_count": len(selected_anchors),
            "extent_count_by_anchor": extent_counts,
            "complete_box_count": len(rows),
        },
        "shortfalls": {
            "x1": None
            if len(selected_x1) == int(budget["x1_branch_budget"])
            else "fewer_valid_x1_bins_than_budget",
            "spatial_diversification": diversification_receipt,
            "y1_by_x1": y1_shortfalls,
            "extents_by_anchor": extent_shortfalls,
        },
        "null_semantics": "bounded_free_search_null_is_non_evidence_for_absence",
        "positive_semantics": "positive_target_support_is_decision_bearing_and_blocks_C",
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    return rows, receipt


def bind_free_surface_score_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    request: CandidateRow,
    owner_ledger: Mapping[str, OwnerLedgerEntry],
    rules: DecisionRules,
) -> list[dict[str, Any]]:
    """Bind dynamically discovered boxes to the consumable v2 row envelope.

    Free-tree geometry has not passed the restricted bank's pre-score physical
    owner adjudication. It is therefore emitted as explicitly unreviewed raw
    evidence; the summarizer retains it but cannot silently promote it to a
    target/foil basin. A later owning review can bind physical identities
    without rerunning the model scores.
    """

    owner = resolve_owner(request.diagnostic_owner_id, owner_ledger)
    common = _common_row_prefix(candidate=request, owner=owner, rules=rules)
    result: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        coordinate_tokens = row.get("coord_token_ids")
        if not isinstance(coordinate_tokens, list) or len(coordinate_tokens) != 4:
            _fail(
                "free-tree complete box lacks four coordinate tokens",
                candidate_id=row.get("candidate_id"),
            )
        expected_id = "free-tree-box:sha256:" + sha256_json(
            {
                "request_id": request.candidate_id,
                "coordinate_bin_values": row.get("coordinate_bin_values"),
            }
        )
        if row.get("candidate_id") != expected_id:
            _fail(
                "free-tree complete box identity does not reconstruct",
                candidate_id=row.get("candidate_id"),
                recomputed=expected_id,
            )
        result.append(
            {
                **common,
                "candidate_id": expected_id,
                "image_identity": request.image_identity
                or str(request.raw_payload.get("image_identity") or ""),
                "landscape_surface": "canonical_description_free",
                "role": "unadjudicated_free_candidate",
                "physical_owner_hint": None,
                "review_status": "unreviewed",
                "candidate_kind": "scan",
                "request_kind": "complete_box",
                "basin_id": None,
                "upstream_adjudication": {
                    **dict(common["upstream_adjudication"]),
                    "generated_candidate_review_status": "unreviewed",
                },
                "coord_token_ids": coordinate_tokens,
                "coord_token_ids_sha256": sha256_json(coordinate_tokens),
                "raw_model_logprob": row["raw_model_logprob"],
                "auxiliary_policy_scores": row["auxiliary_policy_scores"],
                "proposal_verification": {
                    "self_consistency": "passed",
                    "pure_core_recomputation": "passed",
                    "verification_domain": "frozen_free_tree_request_and_coordinate_identity",
                },
                "likelihood_channel_note": _LIKELIHOOD_CHANNEL_NOTE,
                "free_surface_execution": {
                    "free_tree_request_id": request.candidate_id,
                    "anchor_joint_raw_logprob": row["anchor_joint_raw_logprob"],
                    "bounded_search_null_semantics": row[
                        "bounded_search_null_semantics"
                    ],
                    "positive_target_support_semantics": row[
                        "positive_target_support_semantics"
                    ],
                },
            }
        )
    return result


def run_batched_reforward_parity_gate(
    *,
    prefix_token_ids: Sequence[int],
    coordinate_token_ids: Sequence[int],
    coordinate_token_id_start: int,
    coordinate_token_id_end_exclusive: int,
    full_reforward: Callable[[Sequence[int]], torch.Tensor],
    batched_full_reforward: Callable[[Sequence[Sequence[int]]], torch.Tensor],
    requested_batch_size: int,
) -> dict[str, Any]:
    """Admit batching by downstream coordinate behavior, not bitwise identity."""

    if len(coordinate_token_ids) < 3:
        raise ValueError("batched parity probe requires x1, y1, and x2 tokens")
    lo = int(coordinate_token_id_start)
    hi = int(coordinate_token_id_end_exclusive)
    comparisons: list[dict[str, Any]] = []
    for depth in (1, 2, 3):
        literal = [
            *[int(value) for value in prefix_token_ids],
            *[int(value) for value in coordinate_token_ids[:depth]],
        ]
        scalar_logits = full_reforward(literal).to(dtype=torch.float32)
        batch_rows = batched_full_reforward([literal])
        if batch_rows.ndim != 2 or int(batch_rows.shape[0]) != 1:
            raise RuntimeError("batched parity probe returned an invalid logits batch")
        batch_logits = batch_rows[0].to(dtype=torch.float32)
        scalar_coord = torch.log_softmax(scalar_logits, dim=-1)[lo:hi]
        batch_coord = torch.log_softmax(batch_logits, dim=-1)[lo:hi]
        finite = bool(torch.isfinite(scalar_coord).all() and torch.isfinite(batch_coord).all())
        max_abs_diff = (
            float(torch.max(torch.abs(scalar_coord - batch_coord)).item())
            if finite
            else None
        )
        scalar_argmax = int(torch.argmax(scalar_coord).item()) if finite else None
        batch_argmax = int(torch.argmax(batch_coord).item()) if finite else None
        comparisons.append(
            {
                "relative_depth": depth,
                "finite": finite,
                "coordinate_argmax_scalar": scalar_argmax,
                "coordinate_argmax_batched": batch_argmax,
                "coordinate_argmax_identical": scalar_argmax == batch_argmax,
                "coordinate_logprob_max_abs_diff": max_abs_diff,
            }
        )
    passed = all(
        comparison["finite"]
        and comparison["coordinate_argmax_identical"]
        and comparison["coordinate_logprob_max_abs_diff"] is not None
        and comparison["coordinate_logprob_max_abs_diff"]
        <= BATCH_REFORWARD_COORD_LOGPROB_MAX_ABS_DIFF
        for comparison in comparisons
    )
    result = {
        "schema_version": "batched_full_reforward_parity.v1",
        "status": "passed" if passed else "failed_scalar_fallback_required",
        "requested_batch_size": int(requested_batch_size),
        "effective_batch_size": int(requested_batch_size) if passed else 1,
        "coordinate_logprob_max_abs_diff_tolerance": (
            BATCH_REFORWARD_COORD_LOGPROB_MAX_ABS_DIFF
        ),
        "argmax_requirement": "identical_within_coordinate_token_domain",
        "comparisons": comparisons,
    }
    result["sha256"] = sha256_json(result)
    return result


# ---------------------------------------------------------------------------
# TF32 parity pinning
# ---------------------------------------------------------------------------


def pin_fp32_parity_flags() -> dict[str, bool]:
    """Disable both TF32 fast-paths and return the flags actually observed afterward."""

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------

COMPLETE_BOX_ROW_FIELDS: tuple[str, ...] = (
    "schema_version",
    "candidate_id",
    "diagnostic_owner_id",
    "gt_owner_id",
    "owner_status",
    "image_id",
    "image_identity",
    "context_id",
    "landscape_surface",
    "role",
    "physical_owner_hint",
    "review_status",
    "candidate_kind",
    "foil_set_id",
    "foil_set_digest",
    "rule_digest",
    "request_kind",
    "native_repetition_penalty_stratum",
    "basin_id",
    "prefix_token_count",
    "prefix_token_ids_sha256",
    "upstream_adjudication",
    "coord_token_ids",
    "coord_token_ids_sha256",
    "raw_model_logprob",
    "auxiliary_policy_scores",
    "core_candidate",
    "proposal_verification",
    "likelihood_channel_note",
)

DENSE_SCAN_ROW_FIELDS: tuple[str, ...] = (
    "schema_version",
    "candidate_id",
    "diagnostic_owner_id",
    "gt_owner_id",
    "owner_status",
    "image_id",
    "image_identity",
    "context_id",
    "landscape_surface",
    "role",
    "physical_owner_hint",
    "review_status",
    "candidate_kind",
    "foil_set_id",
    "foil_set_digest",
    "rule_digest",
    "request_kind",
    "native_repetition_penalty_stratum",
    "basin_id",
    "prefix_token_count",
    "prefix_token_ids_sha256",
    "upstream_adjudication",
    "fixed_coord_token_ids",
    "scan_slot",
    "raw_bin_scan",
    "auxiliary_policy_bin_scan",
    "proposal_verification",
    "likelihood_channel_note",
)

_LIKELIHOOD_CHANNEL_NOTE = (
    "raw_model_logprob/raw_bin_scan is the unmodified fp32 lm-head channel; "
    "auxiliary_policy_scores/auxiliary_policy_bin_scan are repetition-penalty-adjusted "
    "policy readouts derived from that same raw forward, not model likelihoods."
)


def _common_row_prefix(
    *, candidate: CandidateRow, owner: OwnerLedgerEntry, rules: DecisionRules
) -> dict[str, Any]:
    lineage = candidate.raw_payload.get("source_review_foreign_key_lineage")
    lineage_mapping = dict(lineage) if isinstance(lineage, Mapping) else {}
    global_status = (
        "globally_ambiguous"
        if owner.status == "unresolved" or candidate.gt_owner_id in {None, ""}
        else "clear"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate.candidate_id,
        "diagnostic_owner_id": candidate.diagnostic_owner_id,
        "gt_owner_id": owner.gt_owner_id,
        "owner_status": owner.status,
        "image_id": candidate.image_id,
        "image_identity": candidate.image_identity
        or str(candidate.raw_payload.get("image_identity") or f"image:{candidate.image_id}:legacy_fixture"),
        "context_id": candidate.context_id,
        "landscape_surface": "restricted_gt_target",
        "role": candidate.role,
        "physical_owner_hint": candidate.physical_owner_hint,
        "review_status": candidate.review_status,
        "candidate_kind": candidate.candidate_kind,
        "foil_set_id": candidate.foil_set_id,
        "foil_set_digest": resolve_foil_set_digest(candidate, rules),
        "rule_digest": rules.rules_digest,
        "request_kind": candidate.request_kind,
        "native_repetition_penalty_stratum": candidate.native_repetition_penalty_stratum,
        "basin_id": candidate.basin_id,
        "prefix_token_count": len(candidate.prefix_token_ids),
        "prefix_token_ids_sha256": candidate.prefix_token_ids_sha256,
        "upstream_adjudication": {
            "global_ambiguity_status": global_status,
            "source_review_foreign_key_lineage": lineage_mapping,
        },
    }


def _restricted_core_candidate(candidate: CandidateRow) -> dict[str, Any]:
    """Project a v2 builder row onto the exact pure-core reconstruction fields."""

    payload = candidate.raw_payload
    required = (
        "bank_name",
        "source_id",
        "extent_submode",
        "proposal_measure_id",
        "role_id",
        "identity_kind",
        "identity_id",
        "geometry_identity",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        _fail(
            "complete-box score row lacks authoritative pure-core candidate fields",
            candidate_id=candidate.candidate_id,
            missing=missing,
        )
    return {
        "bank_name": payload["bank_name"],
        "source_id": payload["source_id"],
        "extent_submode": payload["extent_submode"],
        "proposal_measure_id": payload["proposal_measure_id"],
        "coordinate_bins": list(candidate.coordinate_bin_values),
        "role_id": payload["role_id"],
        "identity_kind": payload["identity_kind"],
        "identity_id": payload["identity_id"],
        "geometry_identity": payload["geometry_identity"],
    }


def build_landscape_score_row(
    *,
    candidate: CandidateRow,
    owner_ledger: Mapping[str, OwnerLedgerEntry],
    prediction_row_ledger: Mapping[str, PredictionRowLedgerEntry],
    rules: DecisionRules,
    pure_core: Any | None,
    reconstructed_prompt_token_ids: Sequence[int],
    backend: CacheBackend,
    prefill_logits: torch.Tensor,
    tokenizer_identity: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    runtime_receipt_id: str,
    frozen_tokenizer_identity_digest: str | None = None,
    frozen_model_identity_digest: str | None = None,
) -> dict[str, Any]:
    """Run every fail-fast gate, then score one candidate against a prefilled branch.

    ``tokenizer_identity``/``model_identity``/``runtime_receipt_id`` are
    bound into every full-vocabulary attestation produced for this row (see
    :class:`AttestationContext`), not just the raw vocab size/domain.
    """

    owner = resolve_owner(candidate.diagnostic_owner_id, owner_ledger)
    verify_no_policy_mixing(candidate, prediction_row_ledger)
    verify_coordinate_tokens(candidate, rules)
    verify_identity_kind_matches_role_constraints(candidate)
    verify_canonical_description(candidate, rules)
    verify_production_prompt_prefix(candidate, reconstructed_prompt_token_ids)
    proposal_outcome = verify_candidate_proposal(candidate, pure_core=pure_core)
    attestation = build_attestation_context(
        expected_vocab_size=rules.model_vocab_size,
        tokenizer_identity=tokenizer_identity,
        model_identity=model_identity,
        rule_digest=rules.rules_digest,
        runtime_receipt_id=runtime_receipt_id,
        frozen_tokenizer_identity_digest=frozen_tokenizer_identity_digest,
        frozen_model_identity_digest=frozen_model_identity_digest,
    )

    if candidate.request_kind == "complete_box":
        assert candidate.coord_token_ids is not None
        scored = score_complete_box_candidate(
            backend=backend,
            prefill_logits=prefill_logits,
            coord_token_ids=candidate.coord_token_ids,
            attestation=attestation,
            running_context_token_ids=candidate.prefix_token_ids,
        )
        row = {
            **_common_row_prefix(candidate=candidate, owner=owner, rules=rules),
            "coord_token_ids": list(candidate.coord_token_ids),
            "coord_token_ids_sha256": sha256_json(list(candidate.coord_token_ids)),
            "raw_model_logprob": scored["raw"],
            "auxiliary_policy_scores": scored["auxiliary_policy"],
            "core_candidate": _restricted_core_candidate(candidate)
            if candidate.record_type == "complete_box_candidate"
            else {},
            "proposal_verification": proposal_outcome,
            "likelihood_channel_note": _LIKELIHOOD_CHANNEL_NOTE,
        }
        if tuple(row) != COMPLETE_BOX_ROW_FIELDS:
            _fail(
                "emitted complete-box row field set drifted from the frozen schema",
                observed=list(row),
            )
        return row

    assert candidate.scan_slot is not None
    lo = rules.schema_tokens["coordinate_token_id_start"]
    hi = rules.schema_tokens["coordinate_token_id_end_exclusive"]
    scored = score_dense_scan_candidate(
        backend=backend,
        prefill_logits=prefill_logits,
        fixed_coord_token_ids=candidate.fixed_coord_token_ids,
        scan_slot=candidate.scan_slot,
        coordinate_token_id_start=lo,
        coordinate_token_id_end_exclusive=hi,
        attestation=attestation,
        running_context_token_ids=candidate.prefix_token_ids,
    )
    row = {
        **_common_row_prefix(candidate=candidate, owner=owner, rules=rules),
        "fixed_coord_token_ids": list(candidate.fixed_coord_token_ids),
        "scan_slot": candidate.scan_slot,
        "raw_bin_scan": scored["raw"],
        "auxiliary_policy_bin_scan": scored["auxiliary_policy"],
        "proposal_verification": proposal_outcome,
        "likelihood_channel_note": _LIKELIHOOD_CHANNEL_NOTE,
    }
    if tuple(row) != DENSE_SCAN_ROW_FIELDS:
        _fail(
            "emitted dense-scan row field set drifted from the frozen schema",
            observed=list(row),
        )
    return row


# ---------------------------------------------------------------------------
# Conditional-y1 completeness digest
# ---------------------------------------------------------------------------


def compute_conditional_y1_completeness_digest(
    *,
    context_id: str,
    diagnostic_owner_id: str,
    rule_digest: str,
    admitted_x1_token_ids: Sequence[int],
    y1_bin_count: int,
    dense_scan_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Attest that every declared x1 anchor was scanned over all declared y1 bins.

    Consumed by downstream basin registration/measurement so it can trust the
    y1 landscape for (``context_id``, ``diagnostic_owner_id``) is complete
    without re-deriving coverage from raw rows. Fails fast on a missing,
    duplicated, or foreign-context/owner/rule-digest x1 anchor, or on a row
    whose bin scan does not cover exactly ``y1_bin_count`` bins.
    """

    admitted = sorted(int(v) for v in admitted_x1_token_ids)
    if len(set(admitted)) != len(admitted):
        _fail(
            "admitted_x1_token_ids contains duplicates", admitted_x1_token_ids=admitted
        )

    covered: dict[int, Mapping[str, Any]] = {}
    for row in dense_scan_rows:
        if row.get("request_kind") != "dense_scan" or row.get("scan_slot") != "y1":
            _fail(
                "conditional-y1 completeness digest requires only y1 dense-scan rows",
                row_candidate_id=row.get("candidate_id"),
            )
        if (
            row.get("context_id") != context_id
            or row.get("diagnostic_owner_id") != diagnostic_owner_id
        ):
            _fail(
                "dense-scan row belongs to a different context/owner than this completeness digest",
                row_candidate_id=row.get("candidate_id"),
                context_id=context_id,
                diagnostic_owner_id=diagnostic_owner_id,
            )
        if row.get("rule_digest") != rule_digest:
            _fail(
                "dense-scan row was produced under a different rule_digest",
                row_candidate_id=row.get("candidate_id"),
            )
        fixed = row.get("fixed_coord_token_ids")
        if not isinstance(fixed, list) or len(fixed) != 1:
            _fail(
                "y1 dense-scan row must fix exactly one prior coordinate (x1)",
                row_candidate_id=row.get("candidate_id"),
            )
        x1 = int(fixed[0])
        bins = (row.get("raw_bin_scan") or {}).get("bin_logprobs")
        if not isinstance(bins, list) or len(bins) != int(y1_bin_count):
            _fail(
                "y1 dense-scan row does not cover exactly the declared y1 bin count",
                row_candidate_id=row.get("candidate_id"),
                observed=None if bins is None else len(bins),
                expected=y1_bin_count,
            )
        if x1 in covered:
            _fail("duplicate y1 dense-scan row for the same x1 anchor", x1=x1)
        covered[x1] = row

    missing = sorted(set(admitted) - set(covered))
    extra = sorted(set(covered) - set(admitted))
    if missing or extra:
        _fail(
            "y1 dense-scan rows do not exactly cover every declared x1 anchor",
            missing_x1=missing,
            unexpected_x1=extra,
        )

    seed = {
        "context_id": context_id,
        "diagnostic_owner_id": diagnostic_owner_id,
        "rule_digest": rule_digest,
        "admitted_x1_token_ids": admitted,
        "y1_bin_count": int(y1_bin_count),
    }
    return {
        "context_id": context_id,
        "diagnostic_owner_id": diagnostic_owner_id,
        "rule_digest": rule_digest,
        "admitted_x1_token_ids": admitted,
        "x1_count": len(admitted),
        "y1_bin_count": int(y1_bin_count),
        "status": "complete",
        "completeness_digest": sha256_json(seed),
    }


def build_conditional_y1_completeness_attestations(
    *,
    pure_core: Any,
    candidates: Sequence[CandidateRow],
    scored_rows: Sequence[Mapping[str, Any]],
    owner_context_ledger: Mapping[tuple[str, str], Any],
    rules: DecisionRules,
) -> list[dict[str, Any]]:
    """Build the core's full identity-bound attestation, not a local digest proxy."""

    if rules.materializer_rules is None:
        _fail("full conditional-y1 attestation requires candidate-builder-v2 rules")
    plans = {
        candidate.candidate_id: candidate
        for candidate in candidates
        if candidate.record_type == "conditional_y1_score_plan"
    }
    scored_by_id = {
        str(row.get("candidate_id")): row
        for row in scored_rows
        if row.get("request_kind") == "dense_scan" and row.get("scan_slot") == "y1"
    }
    if set(scored_by_id) != set(plans):
        _fail(
            "scored restricted y1 rows do not exactly cover the authoritative plans",
            missing=sorted(set(plans) - set(scored_by_id)),
            extra=sorted(set(scored_by_id) - set(plans)),
        )

    plan_groups: dict[tuple[str, str], list[CandidateRow]] = {}
    for plan in plans.values():
        plan_groups.setdefault((plan.diagnostic_owner_id, plan.context_id), []).append(
            plan
        )

    results: list[dict[str, Any]] = []
    for key, group in sorted(plan_groups.items()):
        ledger = owner_context_ledger.get(key)
        if ledger is None:
            _fail(
                "conditional-y1 plan group lacks its owner-context ledger",
                owner_context=key,
            )
        scores_by_x1: dict[Any, list[Any]] = {}
        for plan in sorted(group, key=lambda item: item.coordinate_bin_values):
            x1_value = plan.coordinate_bin_values[0]
            row = scored_by_id[plan.candidate_id]
            bins = (row.get("raw_bin_scan") or {}).get("bin_logprobs")
            if not isinstance(bins, list) or len(bins) != len(
                plan.complete_conditional_y1
            ):
                _fail(
                    "conditional-y1 score row is incomplete",
                    candidate_id=plan.candidate_id,
                )
            x1 = pure_core.CoordinateBin(x1_value)
            scores_by_x1[x1] = [
                pure_core.ConditionalY1ScoreReceipt(
                    x1=x1,
                    y1=pure_core.CoordinateBin(int(entry["y1"])),
                    raw_selected_token_logprob=float(logprob),
                    can_form_valid_box=bool(entry["can_form_valid_box"]),
                    invalid_box_reason=entry["invalid_box_reason"],
                )
                for entry, logprob in zip(
                    plan.complete_conditional_y1, bins, strict=True
                )
            ]
        attestation = pure_core.attest_complete_conditional_y1_scores(
            diagnostic_owner_id=ledger.diagnostic_owner_id,
            gt_owner_id=ledger.gt_owner_id,
            image_identity=ledger.image_identity,
            context_id=ledger.context_id,
            canonical_description_text=ledger.canonical_description["text"],
            canonical_description_token_digest=ledger.canonical_description[
                "token_ids_sha256"
            ],
            context_token_digest=ledger.context_tokens["token_ids_sha256"],
            tokenizer_identity=ledger.vocabulary_attestation[
                "tokenizer_identity_sha256"
            ],
            model_identity=ledger.vocabulary_attestation["model_identity_sha256"],
            runtime_identity=ledger.vocabulary_attestation["runtime_identity_sha256"],
            gt_box=ledger.gt_box,
            scores_by_x1=scores_by_x1,
            rules=rules.materializer_rules.landscape,
        )
        result = dataclasses.asdict(attestation)
        result["gt_box"] = list(ledger.gt_box.as_tuple())
        result["landscape_surface"] = "restricted_gt_target"
        result["completeness_plan_digest"] = group[0].completeness_plan_digest
        result["conditional_y1_plan_ids"] = sorted(plan.candidate_id for plan in group)
        result["scored_completeness_receipt_sha256"] = sha256_json(result)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Execution receipt
# ---------------------------------------------------------------------------

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def load_artifact_manifest_digests(
    path: Path, *, required_keys: Sequence[str]
) -> dict[str, str]:
    """Resolve ``{name: sha256_hex}`` for every required source artifact.

    Accepts either a flat top-level mapping (``{"owner_ledger": "<hex>"}``)
    or the nested ``{"owner_ledger": {"sha256": "<hex>"}}``/``{"digest": "<hex>"}``
    form real artifact manifests commonly use. A manifest that cannot bind a
    resolvable sha256 digest for every required key fails fast here -- it
    must never be silently treated as "no manifest asserted", which is what
    a bare ``{k: v for k, v in payload.items() if isinstance(v, str)}`` filter
    does against a manifest whose digests are nested one level down.
    """

    payload = _read_json(path)
    resolved: dict[str, str] = {}
    for key in required_keys:
        value = payload.get(key)
        digest: Any = None
        if isinstance(value, str):
            digest = value
        elif isinstance(value, Mapping):
            digest = value.get("sha256", value.get("digest"))
        if not isinstance(digest, str) or not _SHA256_HEX_RE.match(digest):
            _fail(
                "artifact-manifest.json does not bind a resolvable sha256 digest for a required artifact",
                path=str(path),
                key=key,
                observed=None if value is None else type(value).__name__,
            )
        resolved[key] = digest
    return resolved


def _installed_package_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "not_installed"


def _git_capture(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def build_implementation_provenance() -> dict[str, Any]:
    """Capture code identity without embedding a potentially sensitive diff."""

    relevant = (
        Path("scripts/research/score_sorted_owner_basin_landscape.py"),
        Path("scripts/research/summarize_sorted_owner_basin_landscape.py"),
        Path("scripts/research/sorted_owner_basin_landscape.py"),
        Path("scripts/research/build_sorted_owner_basin_candidates.py"),
    )
    relative_paths = [path.as_posix() for path in relevant]
    status = _git_capture(
        "status", "--short", "--untracked-files=all", "--", *relative_paths
    )
    diff = _git_capture(
        "diff", "--no-ext-diff", "--binary", "HEAD", "--", *relative_paths
    )
    return {
        "git_head": _git_capture("rev-parse", "HEAD").strip(),
        "git_dirty": bool(status),
        "git_dirty_diff_sha256": hashlib.sha256(
            (status + "\n" + diff).encode("utf-8")
        ).hexdigest(),
        "relevant_file_digests": {
            path.as_posix(): sha256_file(REPO_ROOT / path) for path in relevant
        },
    }


def _validate_free_execution_receipts(
    *,
    declared_request_ids: Sequence[str],
    execution_receipts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    declared = list(declared_request_ids)
    if any(not isinstance(request_id, str) or not request_id for request_id in declared):
        _fail("declared free-tree request IDs must be non-empty strings")
    if len(declared) != len(set(declared)):
        _fail("declared free-tree request IDs must be unique")
    validated: list[dict[str, Any]] = []
    executed_ids: list[str] = []
    for receipt in execution_receipts:
        record = dict(receipt)
        request_id = record.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            _fail("free-tree execution receipt lacks a request ID")
        if (
            record.get("schema_version")
            != "sorted_owner_basin_free_tree_execution.v1"
            or record.get("surface") != "free_coordinate_tree"
            or record.get("status") != "executed"
        ):
            _fail(
                "free-tree execution receipt does not attest executed frozen-tree semantics",
                request_id=request_id,
            )
        declared_digest = record.pop("receipt_sha256", None)
        if declared_digest != sha256_json(record):
            _fail("free-tree execution receipt digest is stale", request_id=request_id)
        record["receipt_sha256"] = declared_digest
        executed_ids.append(request_id)
        validated.append(record)
    if len(executed_ids) != len(set(executed_ids)):
        _fail("free-tree execution receipts contain duplicate request IDs")
    if sorted(executed_ids) != sorted(declared):
        _fail(
            "every declared free-tree root must have exactly one execution receipt",
            declared=sorted(declared),
            executed=sorted(executed_ids),
        )
    return sorted(validated, key=lambda record: str(record["request_id"]))


def build_landscape_surface_receipts(
    rows: Sequence[Mapping[str, Any]],
    *,
    free_declared_request_ids: Sequence[str] = (),
    free_execution_receipts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, dict[str, Any]]:
    validated_free_receipts = _validate_free_execution_receipts(
        declared_request_ids=free_declared_request_ids,
        execution_receipts=free_execution_receipts,
    )
    receipts: dict[str, dict[str, Any]] = {}
    for surface in LANDSCAPE_SURFACES:
        surface_rows = sorted(
            (dict(row) for row in rows if row.get("landscape_surface") == surface),
            key=lambda row: str(row.get("candidate_id")),
        )
        status = "passed" if surface_rows else "not_scored_test_fixture"
        if surface == "canonical_description_free" and free_declared_request_ids:
            status = (
                "passed"
                if surface_rows
                else "executed_bounded_null_non_evidence"
            )
            observed_counts: dict[str, int] = {}
            for row in surface_rows:
                execution = row.get("free_surface_execution")
                if not isinstance(execution, Mapping):
                    _fail("free-surface score row lacks its free-tree execution binding")
                request_id = execution.get("free_tree_request_id")
                if not isinstance(request_id, str):
                    _fail("free-surface score row lacks a free-tree request ID")
                observed_counts[request_id] = observed_counts.get(request_id, 0) + 1
            expected_counts: dict[str, int] = {}
            for execution_receipt in validated_free_receipts:
                counts = execution_receipt.get("counts")
                if not isinstance(counts, Mapping):
                    _fail("free-tree execution receipt lacks counts")
                count = counts.get("complete_box_count")
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    _fail("free-tree execution receipt complete-box count is invalid")
                expected_counts[str(execution_receipt["request_id"])] = count
            if observed_counts != {
                request_id: count
                for request_id, count in expected_counts.items()
                if count > 0
            }:
                _fail(
                    "free-surface rows do not match per-request execution counts",
                    expected=expected_counts,
                    observed=observed_counts,
                )
        receipts[surface] = {
            "status": status,
            "row_count": len(surface_rows),
            "score_rows_sha256": sha256_json(surface_rows),
        }
        if surface == "canonical_description_free":
            receipts[surface].update(
                {
                    "declared_request_ids": sorted(free_declared_request_ids),
                    "executed_request_ids": [
                        str(record["request_id"])
                        for record in validated_free_receipts
                    ],
                    "execution_receipts": validated_free_receipts,
                    "execution_receipts_sha256": sha256_json(
                        validated_free_receipts
                    ),
                }
            )
    return receipts


def seal_execution_receipt(
    receipt: Mapping[str, Any], *, scores_path: Path, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Bind an already-written score JSONL and its exact surface projections."""

    sealed = dict(receipt)
    sealed["output_artifacts"] = {
        "landscape_scores": {
            "path": str(scores_path),
            "sha256": sha256_file(scores_path),
            "row_count": len(rows),
        }
    }
    free_declared_request_ids: Sequence[str] = ()
    free_execution_receipts: Sequence[Mapping[str, Any]] = ()
    free_surface_mapping: Mapping[str, Any] | None = None
    restricted_surface_mapping: Mapping[str, Any] | None = None
    surfaces = sealed.get("surfaces")
    if surfaces is not None:
        if not isinstance(surfaces, Mapping):
            _fail("receipt.surfaces must be a mapping")
        free_surface = surfaces.get("free_coordinate_tree")
        if not isinstance(free_surface, Mapping):
            _fail("receipt.surfaces must bind the free coordinate-tree surface")
        free_surface_mapping = free_surface
        restricted_surface = surfaces.get("restricted_candidate_bank")
        if isinstance(restricted_surface, Mapping):
            restricted_surface_mapping = restricted_surface
        elif sealed.get("runtime_execution_status") == LIVE_SCORING_PENDING_SEAL_STATUS:
            _fail("live receipt.surfaces must bind the restricted candidate-bank surface")
        declared = free_surface.get("declared_request_ids")
        executions = free_surface.get("receipts")
        if (
            isinstance(declared, (str, bytes))
            or not isinstance(declared, Sequence)
            or isinstance(executions, (str, bytes))
            or not isinstance(executions, Sequence)
        ):
            _fail("free coordinate-tree surface must declare request IDs and receipts")
        declared_request_ids: list[str] = []
        for request_id in declared:
            if not isinstance(request_id, str):
                _fail("free coordinate-tree declared request ID must be a string")
            declared_request_ids.append(request_id)
        free_declared_request_ids = declared_request_ids
        free_execution_receipts = []
        for execution in executions:
            if not isinstance(execution, Mapping):
                _fail("free coordinate-tree execution receipt must be a mapping")
            free_execution_receipts.append(execution)
    elif sealed.get("runtime_execution_status") == LIVE_SCORING_PENDING_SEAL_STATUS:
        _fail("live scoring receipt lacks declared free-tree execution state")
    sealed["landscape_surface_receipts"] = build_landscape_surface_receipts(
        rows,
        free_declared_request_ids=free_declared_request_ids,
        free_execution_receipts=free_execution_receipts,
    )
    free_authority = sealed["landscape_surface_receipts"][
        "canonical_description_free"
    ]
    restricted_authority = sealed["landscape_surface_receipts"][
        "restricted_gt_target"
    ]
    if isinstance(surfaces, Mapping) and free_surface_mapping is not None:
        free_surface_record = dict(free_surface_mapping)
        free_surface_record.update(
            status=free_authority["status"],
            row_count=free_authority["row_count"],
        )
        updated_surfaces = {
            **dict(surfaces),
            "free_coordinate_tree": free_surface_record,
        }
        if restricted_surface_mapping is not None:
            restricted_surface_record = dict(restricted_surface_mapping)
            restricted_surface_record.update(
                status=restricted_authority["status"],
                row_count=restricted_authority["row_count"],
            )
            updated_surfaces["restricted_candidate_bank"] = (
                restricted_surface_record
            )
        sealed["surfaces"] = {
            **updated_surfaces,
        }
        sealed["free_surface_status"] = free_authority["status"]
        if restricted_surface_mapping is not None:
            sealed["restricted_surface_status"] = restricted_authority["status"]
    if sealed.get("runtime_execution_status") == LIVE_SCORING_PENDING_SEAL_STATUS:
        sealed["runtime_execution_status"] = LIVE_SCORING_SEALED_STATUS
    return sealed


def build_execution_receipt(
    *,
    command: Sequence[str],
    source_files: Mapping[str, Path],
    manifest_digests: Mapping[str, str] | None,
    rules: DecisionRules,
    rules_path: Path,
    backend_receipt: Mapping[str, Any],
    candidate_count: int,
    owners_covered: int,
    pure_core_status_dict: Mapping[str, Any],
    environment: Mapping[str, Any],
    runtime_execution_status: str = TEST_FIXTURE_SCORING_STATUS,
) -> dict[str, Any]:
    source_digests: dict[str, Any] = {}
    for name, path in source_files.items():
        actual = sha256_file(path)
        expected = None if manifest_digests is None else manifest_digests.get(name)
        if expected is not None and expected != actual:
            _fail(
                f"{name} sha256 does not match the recorded artifact-manifest digest; refusing to score "
                "against a stale or substituted source",
                name=name,
                path=str(path),
                expected=expected,
                actual=actual,
            )
        source_digests[name] = {
            "path": str(path),
            "sha256": actual,
            "manifest_expected": expected,
            "match": (expected is None or expected == actual),
        }
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "generated_at_unix": time.time(),
        "command": list(command),
        "runtime_execution_status": runtime_execution_status,
        "source_digests": source_digests,
        "decision_rules": {
            "path": str(rules_path),
            "file_sha256": sha256_file(rules_path),
            "core_rule_digest": rules.rules_digest,
        },
        "model_identity": backend_receipt.get("model_identity"),
        "tokenizer_identity": backend_receipt.get("tokenizer_identity"),
        "backend_session": dict(backend_receipt),
        "environment": dict(environment),
        "command_environment": {
            "cwd": str(Path.cwd()),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "transformers_version": _installed_package_version("transformers"),
        },
        "implementation_provenance": build_implementation_provenance(),
        "numeric_reproduction_tolerance": rules.numeric_tolerance,
        "pure_core": dict(pure_core_status_dict),
        "candidate_count": int(candidate_count),
        "owners_covered": int(owners_covered),
        "likelihood_channels": {
            "raw": "fp32_log_softmax_over_the_complete_unfiltered_lm_head_vocabulary",
            "auxiliary_policy": [
                _policy_view_key(p) for p in REPETITION_PENALTY_STRATA
            ],
            "policy_is_not_a_model_likelihood": True,
            "subset_normalization_forbidden": True,
        },
        "execution_architecture": {
            "one_full_reforward_per_candidate": False,
            "strategy": (
                "prefill once through (image, self-prefix, description, box_start); branch per candidate "
                "with BranchCursor over a shared DynamicCache; crop back to the branch entry length on exit"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Driver (not exercised by unit tests; requires a live model -- no GPU launch
# happens as part of this delivery)
# ---------------------------------------------------------------------------


def select_candidate_contexts(
    candidates: Sequence[CandidateRow], include_context_ids: Sequence[str] | None
) -> tuple[list[CandidateRow], dict[str, Any]]:
    """Apply an exact context allowlist after full input-contract validation."""

    available = sorted({candidate.context_id for candidate in candidates})
    requested = list(include_context_ids or ())
    if len(requested) != len(set(requested)):
        duplicates = sorted(
            context_id for context_id in set(requested) if requested.count(context_id) > 1
        )
        _fail("--include-context-id contains duplicates", duplicates=duplicates)
    unknown = sorted(set(requested) - set(available))
    if unknown:
        _fail(
            "--include-context-id names context IDs absent from the fully validated candidate input",
            unknown=unknown,
            available=available,
        )
    included = sorted(requested) if requested else available
    selected = [candidate for candidate in candidates if candidate.context_id in included]
    selection_payload = {
        "mode": "explicit_allowlist" if requested else "all_validated_contexts",
        "available_context_ids": available,
        "included_context_ids": included,
        "excluded_context_ids": sorted(set(available) - set(included)),
        "input_candidate_count": len(candidates),
        "selected_candidate_count": len(selected),
        "selected_diagnostic_owner_ids": sorted(
            {candidate.diagnostic_owner_id for candidate in selected}
        ),
    }
    return selected, {
        **selection_payload,
        "selection_sha256": sha256_json(selection_payload),
    }


def validate_cache_admission_scope(
    candidates: Sequence[CandidateRow],
    *,
    relaxed_selected_logprob_max_abs_diff: float | None,
) -> dict[str, Any]:
    """Keep opt-in approximate admission local to one exact cache identity.

    The strict all-vocabulary gate retains its historical one-probe/global-run
    behavior. Approximate admission is intentionally narrower: one scorer
    invocation may contain only one exact image/context/literal-prefix/history
    group, so a behavior check from one cache cannot silently authorize a
    different context's cache.
    """

    requested_mode = (
        RELAXED_CACHE_ADMISSION_MODE
        if relaxed_selected_logprob_max_abs_diff is not None
        else STRICT_CACHE_ADMISSION_MODE
    )
    identities: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        identity = {
            "image_id": candidate.image_id,
            "context_id": candidate.context_id,
            "prefix_token_ids_sha256": sha256_json(list(candidate.prefix_token_ids)),
            "generated_history_token_ids_sha256": sha256_json(
                list(derive_generated_history_token_ids(candidate))
            ),
        }
        identities[sha256_json(identity)] = identity
    ordered = [identities[digest] for digest in sorted(identities)]
    if requested_mode == RELAXED_CACHE_ADMISSION_MODE and len(ordered) != 1:
        _fail(
            "relaxed cache admission requires exactly one exact image/context/history group per scorer invocation",
            observed_group_count=len(ordered),
            groups=ordered,
        )
    scope = {
        "requested_mode": requested_mode,
        "group_count": len(ordered),
        "relaxed_mode_group_requirement": "exactly_one",
        "requirement_satisfied": (
            requested_mode == STRICT_CACHE_ADMISSION_MODE or len(ordered) == 1
        ),
        "groups": ordered,
    }
    return {**scope, "scope_sha256": sha256_json(scope)}


def validate_phase_a_freeze_binding(
    *,
    rules: DecisionRules,
    rules_path: Path,
    candidates_path: Path,
    candidate_count: int,
    candidate_receipt_path: Path | None,
    freeze_receipt_path: Path | None,
) -> dict[str, Any]:
    """Admit sentinel scoring only after consuming the exact Phase-A freeze."""

    if rules.structural_status != SENTINEL_STRUCTURAL_STATUS:
        if freeze_receipt_path is not None:
            _fail(
                "draft control scoring must not consume a hidden sentinel freeze dependency",
                structural_status=rules.structural_status,
            )
        return {
            "status": "not_applicable_non_sentinel",
            "structural_status": rules.structural_status,
            "sentinel_dependency_consumed": False,
        }
    if freeze_receipt_path is None or candidate_receipt_path is None:
        _fail(
            "sealed sentinel scoring requires --non-c-smoke-freeze-receipt and --candidate-receipt"
        )

    resolved_rules_path = rules_path.expanduser().resolve(strict=True)
    resolved_candidates_path = candidates_path.expanduser().resolve(strict=True)
    resolved_candidate_receipt_path = candidate_receipt_path.expanduser().resolve(
        strict=True
    )
    resolved_freeze_path = freeze_receipt_path.expanduser().resolve(strict=True)
    rules_file_sha256 = sha256_file(resolved_rules_path)
    candidates_file_sha256 = sha256_file(resolved_candidates_path)
    freeze_file_sha256 = sha256_file(resolved_freeze_path)
    if rules.rules_file_sha256 != rules_file_sha256:
        _fail("loaded sentinel rules do not match their current outer file SHA")

    binding = _identity_mapping(
        rules.non_c_smoke_freeze_binding,
        "sentinel rules non-C smoke freeze binding",
    )
    bound_freeze_sha256 = _identity_sha256(
        binding.get("sha256"), "sentinel rules freeze receipt SHA"
    )
    bound_control_sha256 = _identity_sha256(
        binding.get("control_decision_rules_sha256"),
        "sentinel rules control outer SHA",
    )
    bound_semantic_core_sha256 = _identity_sha256(
        binding.get("semantic_core_sha256"),
        "sentinel rules semantic-core SHA",
    )
    if freeze_file_sha256 != bound_freeze_sha256:
        _fail(
            "Phase-A freeze receipt file SHA differs from the sentinel rule binding",
            expected=bound_freeze_sha256,
            observed=freeze_file_sha256,
        )
    if bound_semantic_core_sha256 != rules.rules_digest:
        _fail("sentinel rule freeze binding uses the wrong shared semantic core")

    freeze = _read_json(resolved_freeze_path)
    if freeze.get("schema_version") != NON_C_SMOKE_FREEZE_SCHEMA_VERSION:
        _fail("Phase-A non-C freeze receipt schema_version is not recognized")
    if freeze.get("status") != "passed":
        _fail("Phase-A non-C freeze receipt status is not passed")
    if freeze.get("c_outcomes_read") is not False:
        _fail("Phase-A non-C freeze receipt is not C-blind")
    if freeze.get("independent_reconstruction") != "passed":
        _fail("Phase-A non-C freeze lacks independent reconstruction")
    if (
        _identity_sha256(
            freeze.get("control_decision_rules_sha256"),
            "Phase-A freeze control outer SHA",
        )
        != bound_control_sha256
    ):
        _fail("Phase-A freeze receipt is bound to the wrong control outer SHA")
    if (
        _identity_sha256(
            freeze.get("semantic_core_sha256"),
            "Phase-A freeze semantic-core SHA",
        )
        != rules.rules_digest
    ):
        _fail("Phase-A freeze receipt is bound to the wrong shared semantic core")
    for field in (
        "control_score_artifact_sha256",
        "control_score_receipt_sha256",
        "control_summary_sha256",
        "control_summary_receipt_sha256",
        "calibration_receipt_sha256",
    ):
        _identity_sha256(freeze.get(field), f"Phase-A freeze {field}")
    gates = _identity_mapping(freeze.get("gates"), "Phase-A freeze gates")
    required_non_backend_gates = {
        "representative_positive_control",
        "b2_reviewed_pair",
        "free_surface_executed_raw_only",
    }
    backend_gate_names = {
        "mandatory_cache_parity",
        "scoring_backend_admission",
    }
    expected_gate_names = required_non_backend_gates | backend_gate_names
    observed_gate_names = set(gates)
    if observed_gate_names != expected_gate_names:
        if not backend_gate_names <= observed_gate_names:
            _fail(
                "Phase-A freeze scoring backend admission is missing mandatory state"
            )
        _fail(
            "Phase-A freeze gate names drifted from the sealed non-C smoke contract",
            missing=sorted(expected_gate_names - observed_gate_names),
            unexpected=sorted(observed_gate_names - expected_gate_names),
        )
    if any(gates.get(name) != "passed" for name in required_non_backend_gates):
        _fail("Phase-A freeze receipt lacks a passed non-C smoke gate")
    parity_backend_state = (
        gates.get("mandatory_cache_parity"),
        gates.get("scoring_backend_admission"),
    )
    admitted_backend_states = {
        ("passed", "cache_parity_passed"),
        ("failed", "cache_parity_failed_uncached_reference_used"),
    }
    if parity_backend_state not in admitted_backend_states:
        _fail(
            "Phase-A freeze has an inconsistent scoring backend admission",
            mandatory_cache_parity=parity_backend_state[0],
            scoring_backend_admission=parity_backend_state[1],
        )

    candidate_receipt = _read_json(resolved_candidate_receipt_path)
    if candidate_receipt.get("schema_version") != candidate_builder.SCHEMA_VERSION:
        _fail("candidate receipt schema_version is not recognized")
    candidate_inputs = _identity_mapping(
        candidate_receipt.get("inputs"), "candidate receipt inputs"
    )
    if candidate_inputs.get("landscape_decision_rules_sha256") != rules_file_sha256:
        _fail("candidate receipt does not bind the sentinel decision-rule outer SHA")
    if candidate_inputs.get("core_rule_digest") != rules.rules_digest:
        _fail("candidate receipt does not bind the shared semantic-core digest")
    candidate_output = _identity_mapping(
        candidate_receipt.get("output_jsonl"), "candidate receipt output JSONL"
    )
    if (
        candidate_output.get("sha256") != candidates_file_sha256
        or candidate_output.get("row_count") != candidate_count
    ):
        _fail("candidate receipt does not bind the exact candidate JSONL")

    admission = {
        "status": "passed_post_phase_a_freeze",
        "structural_status": SENTINEL_STRUCTURAL_STATUS,
        "sentinel_dependency_consumed": True,
        "freeze_receipt": {
            "path": str(resolved_freeze_path),
            "file_sha256": freeze_file_sha256,
            "schema_version": freeze["schema_version"],
            "c_outcomes_read": False,
            "control_decision_rules_sha256": bound_control_sha256,
            "semantic_core_sha256": rules.rules_digest,
            "mandatory_cache_parity": parity_backend_state[0],
            "scoring_backend_admission": parity_backend_state[1],
        },
        "sentinel_rules": {
            "path": str(resolved_rules_path),
            "file_sha256": rules_file_sha256,
            "semantic_core_sha256": rules.rules_digest,
            "bound_freeze_receipt_sha256": bound_freeze_sha256,
        },
        "candidate_receipt": {
            "path": str(resolved_candidate_receipt_path),
            "file_sha256": sha256_file(resolved_candidate_receipt_path),
            "candidate_jsonl_sha256": candidates_file_sha256,
            "sentinel_rules_sha256": rules_file_sha256,
            "semantic_core_sha256": rules.rules_digest,
        },
    }
    return {**admission, "admission_sha256": sha256_json(admission)}


def phase_a_provenance_sources(
    admission: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if admission.get("sentinel_dependency_consumed") is not True:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for output_name, admission_key in (
        ("non_c_smoke_freeze_receipt", "freeze_receipt"),
        ("candidate_receipt", "candidate_receipt"),
    ):
        record = _identity_mapping(
            admission.get(admission_key), f"Phase-A admission {admission_key}"
        )
        path_value = record.get("path")
        if not isinstance(path_value, str) or not path_value:
            _fail(f"Phase-A admission {admission_key} lacks its source path")
        source_path = Path(path_value).expanduser().resolve(strict=True)
        admitted_sha256 = _identity_sha256(
            record.get("file_sha256"), f"Phase-A admission {admission_key} SHA"
        )
        if sha256_file(source_path) != admitted_sha256:
            _fail(
                "Phase-A admission source changed after validation",
                source=admission_key,
            )
        result[output_name] = {
            "path": str(source_path),
            "sha256": admitted_sha256,
            "manifest_expected": None,
            "match": True,
            "binding_owner": "phase_a_freeze_admission",
        }
    return result


def _identity_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a mapping")
    return value


def _identity_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_HEX_RE.fullmatch(value):
        _fail(f"{label} must be a lowercase SHA-256 digest", observed=value)
    return value


def _frozen_source_panel_digest(identity: Mapping[str, Any]) -> str:
    direct = identity.get("source_panel_sha256")
    if isinstance(direct, str):
        return _identity_sha256(direct, "runtime identity source_panel_sha256")
    sources = _identity_mapping(identity.get("sources"), "runtime identity.sources")
    for key in ("source_panel", "source_jsonl", "input_panel"):
        entry = sources.get(key)
        if isinstance(entry, Mapping):
            digest = entry.get("sha256", entry.get("file_sha256"))
            if isinstance(digest, str):
                return _identity_sha256(digest, f"runtime identity source {key}")
    task0 = sources.get("task0_execution_receipt")
    task0_binding = _identity_mapping(task0, "runtime Task-0 receipt binding")
    task0_path = Path(str(task0_binding.get("path", ""))).expanduser().resolve(
        strict=True
    )
    expected_task0_file = _identity_sha256(
        task0_binding.get("file_sha256"), "runtime Task-0 receipt file_sha256"
    )
    if sha256_file_streamed(task0_path) != expected_task0_file:
        _fail("runtime Task-0 execution receipt file digest drifted")
    task0_receipt = _read_json(task0_path)
    expected_content = _identity_sha256(
        task0_binding.get("content_sha256"), "runtime Task-0 receipt content_sha256"
    )
    task0_content = {
        key: value
        for key, value in task0_receipt.items()
        if key != "execution_receipt_content_sha256"
    }
    if sha256_json(task0_content) != expected_content:
        _fail("runtime Task-0 execution receipt content digest drifted")
    inputs = _identity_mapping(task0_receipt.get("inputs"), "Task-0 inputs")
    bound_files = inputs.get("bound_files")
    if not isinstance(bound_files, Sequence) or isinstance(bound_files, (str, bytes)):
        _fail("Task-0 inputs.bound_files must be a sequence")
    panel_records = [
        _identity_mapping(record, "Task-0 bound file")
        for record in bound_files
        if isinstance(record, Mapping)
        and "panel" in str(record.get("role", "")).lower()
    ]
    if len(panel_records) != 1:
        _fail(
            "Task-0 execution receipt must bind exactly one frozen source panel",
            observed=len(panel_records),
        )
    panel = panel_records[0]
    panel_path = Path(str(panel.get("path", ""))).expanduser().resolve(strict=True)
    if panel_path.stat().st_size != panel.get("bytes"):
        _fail("frozen source panel byte count drifted")
    panel_digest = _identity_sha256(panel.get("sha256"), "frozen source panel sha256")
    if sha256_file_streamed(panel_path) != panel_digest:
        _fail("frozen source panel content digest drifted")
    return panel_digest


def validate_runtime_identity_preload(
    *,
    identity_path: Path,
    resolved_infer_fingerprint: str,
    source_jsonl: Path,
    candidates: Sequence[CandidateRow],
    rules: DecisionRules,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Validate frozen files and top identities before opening any backend."""

    resolved_path = identity_path.expanduser().resolve(strict=True)
    identity = _read_json(resolved_path)
    content = {key: value for key, value in identity.items() if key != "receipt_digest"}
    if identity.get("schema_version") != "sorted-owner-basin-runtime-identity.v1":
        _fail("runtime identity schema_version is not recognized")
    if identity.get("status") != "frozen" or identity.get("receipt_digest") != sha256_json(
        content
    ):
        _fail("runtime identity is not frozen or its receipt digest is stale")

    tokenizer = _identity_mapping(identity.get("tokenizer"), "runtime tokenizer")
    model = _identity_mapping(identity.get("model"), "runtime model")
    runtime = _identity_mapping(identity.get("runtime"), "runtime projection")
    tokenizer_digest = _identity_sha256(
        tokenizer.get("identity_sha256"), "frozen tokenizer identity"
    )
    model_digest = _identity_sha256(
        model.get("identity_sha256"), "frozen model identity"
    )
    runtime_digest = _identity_sha256(
        runtime.get("identity_sha256"), "frozen runtime identity"
    )
    for label, owner, digest in (
        ("tokenizer", tokenizer, tokenizer_digest),
        ("model", model, model_digest),
        ("runtime", runtime, runtime_digest),
    ):
        source = _identity_mapping(owner.get("identity_source"), f"{label} identity source")
        if sha256_json(dict(source)) != digest:
            _fail(f"frozen {label} top identity digest is stale")
    runtime_source = _identity_mapping(
        runtime.get("identity_source"), "runtime identity source"
    )
    resolved_fingerprints = _identity_mapping(
        runtime_source.get("resolved_config_fingerprints"),
        "runtime resolved config fingerprints",
    )
    if resolved_fingerprints.get("infer_config") != resolved_infer_fingerprint:
        _fail(
            "resolved infer-config fingerprint differs from the frozen runtime identity",
            expected=resolved_fingerprints.get("infer_config"),
            observed=resolved_infer_fingerprint,
        )

    component_records: dict[str, Mapping[str, Any]] = {}
    for owner in (tokenizer, model):
        source = _identity_mapping(owner.get("identity_source"), "identity source")
        raw_records = source.get("component_files")
        if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
            _fail("frozen runtime component_files must be a sequence")
        for raw_record in raw_records:
            record = _identity_mapping(raw_record, "runtime component file")
            path_value = str(record.get("path", ""))
            if not path_value:
                _fail("runtime component file lacks a path")
            component_records[path_value] = record
    for path_value, record in component_records.items():
        component_path = Path(path_value).expanduser().resolve(strict=True)
        if component_path.stat().st_size != record.get("bytes"):
            _fail("runtime component byte count drifted", path=str(component_path))
        expected = _identity_sha256(record.get("sha256"), "component file sha256")
        if sha256_file_streamed(component_path) != expected:
            _fail("runtime component content digest drifted", path=str(component_path))

    coordinate = _identity_mapping(
        identity.get("coordinate_vocabulary"), "runtime coordinate vocabulary"
    )
    if (
        identity.get("model_vocab_size") != rules.model_vocab_size
        or coordinate.get("token_id_start")
        != rules.schema_tokens["coordinate_token_id_start"]
        or coordinate.get("token_id_end_exclusive")
        != rules.schema_tokens["coordinate_token_id_end_exclusive"]
    ):
        _fail("runtime identity coordinate/model vocabulary is incompatible with scorer rules")

    top_attestations = {
        (
            candidate.raw_payload.get("vocabulary_attestation", {}).get(
                "tokenizer_identity_sha256"
            ),
            candidate.raw_payload.get("vocabulary_attestation", {}).get(
                "model_identity_sha256"
            ),
            candidate.raw_payload.get("vocabulary_attestation", {}).get(
                "runtime_identity_sha256"
            ),
        )
        for candidate in candidates
    }
    if top_attestations != {(tokenizer_digest, model_digest, runtime_digest)}:
        _fail(
            "candidate/ledger top identities differ from the frozen runtime identity",
            observed=sorted(str(item) for item in top_attestations),
        )
    source_digest = sha256_file_streamed(source_jsonl.expanduser().resolve(strict=True))
    if source_digest != _frozen_source_panel_digest(identity):
        _fail("source panel digest differs from the frozen runtime identity")

    admission = {
        "status": "passed",
        "identity_path": str(resolved_path),
        "identity_file_sha256": sha256_file(resolved_path),
        "identity_receipt_digest": identity["receipt_digest"],
        "resolved_infer_fingerprint": resolved_infer_fingerprint,
        "source_panel_sha256": source_digest,
        "tokenizer_identity_sha256": tokenizer_digest,
        "model_identity_sha256": model_digest,
        "runtime_identity_sha256": runtime_digest,
        "component_file_count": len(component_records),
    }
    admission["admission_sha256"] = sha256_json(admission)
    return identity, admission


def validate_runtime_identity_postload(
    *,
    frozen_identity: Mapping[str, Any],
    backend_receipt: Mapping[str, Any],
    resolved_infer_fingerprint: str,
) -> dict[str, Any]:
    """Reconstruct and exactly match the frozen runtime identity after open."""

    tokenizer = _identity_mapping(frozen_identity.get("tokenizer"), "runtime tokenizer")
    model = _identity_mapping(frozen_identity.get("model"), "runtime model")
    runtime = _identity_mapping(frozen_identity.get("runtime"), "runtime projection")
    expected_runtime = _identity_mapping(
        runtime.get("identity_source"), "runtime identity source"
    )
    _identity_mapping(model.get("identity_source"), "model identity source")
    _identity_mapping(tokenizer.get("identity_source"), "tokenizer identity source")
    expected_runtime_identity_sha256 = _identity_sha256(
        runtime.get("identity_sha256"), "runtime identity_sha256"
    )
    if sha256_json(expected_runtime) != expected_runtime_identity_sha256:
        _fail("frozen runtime identity source does not match its identity_sha256")
    expected_settings = _identity_mapping(
        expected_runtime.get("effective_settings"), "frozen effective settings"
    )
    observed_settings = _identity_mapping(
        backend_receipt.get("effective_settings"), "observed effective settings"
    )
    observed_options = _identity_mapping(
        observed_settings.get("backend_options"), "observed backend options"
    )
    observed_hf_options = _identity_mapping(
        observed_options.get("hf"), "observed HF options"
    )
    observed_dtype = _identity_mapping(
        observed_settings.get("observed_model_dtype"), "observed model dtype"
    )
    if observed_dtype.get("parameter_dtype_names") != ["torch.float32"]:
        _fail(
            "live backend model parameters are not uniformly FP32",
            observed=observed_dtype,
        )
    observed_attention = observed_settings.get(
        "observed_attn_implementation",
        observed_hf_options.get("attn_implementation"),
    )
    if observed_attention != "sdpa":
        _fail(
            "live backend attention implementation is not SDPA",
            observed=observed_attention,
        )
    separately_observed_setting_keys = {
        "observed_model_dtype",
        "observed_attn_implementation",
    }
    unexpected_frozen_observed_keys = sorted(
        separately_observed_setting_keys.intersection(expected_settings)
    )
    if unexpected_frozen_observed_keys:
        _fail(
            "frozen effective settings contain separately observed runtime-only fields",
            fields=unexpected_frozen_observed_keys,
        )
    observed_identity_settings = {
        key: observed_settings.get(key) for key in sorted(expected_settings)
    }
    observed_processor = _identity_mapping(
        backend_receipt.get("processor_identity"), "observed processor identity"
    )
    expected_resolved = _identity_mapping(
        expected_runtime.get("resolved_config_fingerprints"),
        "frozen resolved config fingerprints",
    )
    if set(expected_resolved) != {"infer_config"}:
        _fail(
            "frozen runtime identity must bind exactly the resolved infer config fingerprint",
            fields=sorted(expected_resolved),
        )
    observed_runtime_source = {
        "backend": backend_receipt.get("backend"),
        "backend_mode": backend_receipt.get("backend_mode"),
        "backend_version": backend_receipt.get("backend_version"),
        "effective_settings": observed_identity_settings,
        "generation_config_fingerprint": expected_runtime.get(
            "generation_config_fingerprint"
        ),
        "likelihood_semantics": backend_receipt.get("likelihood_semantics"),
        "precision": "float32",
        "processor_identity": dict(observed_processor),
        "processor_identity_fingerprint": sha256_json(observed_processor),
        "resolved_config_fingerprints": {
            "infer_config": resolved_infer_fingerprint
        },
        "response_family": backend_receipt.get("response_family"),
    }
    observed_runtime_source["generation_config_fingerprint"] = backend_receipt.get(
        "generation_config_fingerprint"
    )
    observed_runtime_identity_sha256 = sha256_json(observed_runtime_source)
    if expected_runtime != observed_runtime_source:
        differing = sorted(
            key
            for key in set(expected_runtime).union(observed_runtime_source)
            if expected_runtime.get(key) != observed_runtime_source.get(key)
        )
        _fail(
            "live backend differs from the exact frozen runtime identity source",
            differing_fields=differing,
        )
    if observed_runtime_identity_sha256 != expected_runtime_identity_sha256:
        _fail(
            "live backend runtime identity hash differs from the frozen runtime identity hash",
            expected=expected_runtime_identity_sha256,
            observed=observed_runtime_identity_sha256,
        )
    return {
        "status": "passed",
        "projection_sha256": expected_runtime_identity_sha256,
        "runtime_identity_sha256": expected_runtime_identity_sha256,
        "observed_runtime_identity_sha256": observed_runtime_identity_sha256,
        "expected_projection": dict(expected_runtime),
        "observed_projection": observed_runtime_source,
        "output_location_excluded_from_projection": True,
    }


def validate_live_config_binding(
    *, config: Any, source_jsonl: Path
) -> dict[str, Any]:
    """Require the resolved live config to already name the admitted FP32 run."""

    source_path = source_jsonl.expanduser().resolve(strict=True)
    try:
        configured_source = Path(config.data.input_jsonl).expanduser().resolve(
            strict=True
        )
        configured_dtype = config.model.dtype
    except (AttributeError, TypeError) as exc:
        _fail("resolved infer config lacks data.input_jsonl or model.dtype", error=str(exc))
    if configured_source != source_path:
        _fail(
            "resolved infer config data.input_jsonl differs from --source-jsonl",
            configured=str(configured_source),
            requested=str(source_path),
        )
    if configured_dtype != "fp32":
        _fail(
            "resolved infer config model.dtype must already be fp32",
            observed=configured_dtype,
        )
    return {
        "status": "passed",
        "source_jsonl": str(source_path),
        "model_dtype": configured_dtype,
        "binding_sha256": sha256_json(
            {"source_jsonl": str(source_path), "model_dtype": configured_dtype}
        ),
    }


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _finite_nonnegative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("value must be a finite non-negative float")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-context-ledger", type=Path, default=None)
    parser.add_argument("--owner-ledger", type=Path, default=None)
    parser.add_argument("--prediction-row-ledger", type=Path, default=None)
    parser.add_argument("--decision-rules", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--candidate-receipt",
        type=Path,
        default=None,
        help="candidate-builder receipt; required for sealed sentinel scoring",
    )
    parser.add_argument(
        "--non-c-smoke-freeze-receipt",
        type=Path,
        default=None,
        help="Phase-A non-C freeze receipt; required only for sealed sentinel scoring",
    )
    parser.add_argument("--artifact-manifest", type=Path, default=None)
    parser.add_argument("--runtime-identity", type=Path, default=None)
    parser.add_argument("--infer-config", type=Path, default=None)
    parser.add_argument("--source-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--include-context-id",
        action="append",
        default=None,
        help=(
            "repeatable exact context-ID allowlist applied only after the complete candidate "
            "input has passed validation"
        ),
    )
    parser.add_argument(
        "--validate-contract-only",
        action="store_true",
        help="validate the complete CPU-side v2 contract without loading a model",
    )
    parser.add_argument(
        "--full-reforward-batch-size",
        type=_positive_int,
        default=1,
        help=(
            "equal-length same-context GPU batch size for the literal use_cache=False "
            "fallback; 1 preserves the scalar reference path"
        ),
    )
    parser.add_argument(
        "--cache-admission-max-selected-logprob-diff",
        type=_finite_nonnegative_float,
        default=None,
        help=(
            "explicitly request approximate KV-cache admission with this maximum absolute "
            "selected-coordinate logprob delta; omission preserves strict full-vocabulary "
            "allclose admission, and either mode falls back to uncached scoring on failure"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Wire the full contract to a live HF backend session in FP32.

    Routes every image through ``HFBackendSession._materialize_native_inputs``
    -- the same exact native-materialization seam the sibling scorers in this
    investigation use -- to obtain the real processor-expanded executed
    prompt ids, ``pixel_values``, ``image_grid_thw``, attention mask, and
    (via :func:`derive_prefill_position_state`) explicit Qwen positions and
    rope delta. ``plan_image_batch``/``build_prompt_record`` only build the
    request's expected metadata (image path, expected token ids); they never
    independently stand in for the materialized native inputs actually fed to
    the model. Every candidate then advances that shared, real multimodal
    prefill through the prefix-KV branching engine above, grouped by their
    exact literal generated-history suffix so each shared context is
    prefilled exactly once.
    """

    rules = load_decision_rules(args.decision_rules)
    candidates = load_candidate_rows(args.candidates, rules=rules)
    pure_core = _load_pure_core()
    core_status = pure_core_status(pure_core)
    # Production hard gate: an absent or incompatible core must never let this
    # run reach scoring or emit a receipt with production runtime status.
    pure_core = require_production_pure_core(pure_core)
    verify_core_geometry_domain(pure_core, rules)

    owner_context_ledger: dict[tuple[str, str], Any] = {}
    contract_validation: dict[str, Any]
    if rules.contract_version == CANDIDATE_SCHEMA_VERSION:
        if args.owner_context_ledger is None:
            _fail("candidate-builder-v2 execution requires --owner-context-ledger")
        owner_context_ledger = load_owner_context_ledger(
            args.owner_context_ledger, rules=rules
        )
        contract_validation = validate_v2_candidate_contract(
            candidates, owner_context_ledger=owner_context_ledger, rules=rules
        )
        owner_ledger = {
            ledger.diagnostic_owner_id: OwnerLedgerEntry(
                diagnostic_owner_id=ledger.diagnostic_owner_id,
                status="gt",
                gt_owner_id=ledger.gt_owner_id,
                image_id=ledger.image_id,
            )
            for ledger in owner_context_ledger.values()
        }
        prediction_row_ledger = {
            ledger.source_pred_row_id: PredictionRowLedgerEntry(
                pred_row_id=ledger.source_pred_row_id,
                image_id=ledger.image_id,
                repetition_penalty=ledger.native_repetition_penalty_stratum,
            )
            for ledger in owner_context_ledger.values()
            if ledger.source_pred_row_id is not None
        }
        source_files = {
            "owner_context_ledger": args.owner_context_ledger,
            "decision_rules": args.decision_rules,
            "candidates": args.candidates,
        }

    else:
        if args.owner_ledger is None or args.prediction_row_ledger is None:
            _fail(
                "legacy fixture contract requires --owner-ledger and --prediction-row-ledger"
            )
        owner_ledger = load_owner_ledger(args.owner_ledger)
        prediction_row_ledger = load_prediction_row_ledger(args.prediction_row_ledger)
        contract_validation = {
            "schema_version": DECISION_RULES_SCHEMA_VERSION,
            "status": "legacy_fixture_only",
            "candidate_count": len(candidates),
        }
        source_files = {
            "owner_ledger": args.owner_ledger,
            "prediction_row_ledger": args.prediction_row_ledger,
            "decision_rules": args.decision_rules,
            "candidates": args.candidates,
        }

    phase_a_freeze_admission = validate_phase_a_freeze_binding(
        rules=rules,
        rules_path=args.decision_rules,
        candidates_path=args.candidates,
        candidate_count=len(candidates),
        candidate_receipt_path=getattr(args, "candidate_receipt", None),
        freeze_receipt_path=getattr(args, "non_c_smoke_freeze_receipt", None),
    )

    candidates, context_selection = select_candidate_contexts(
        candidates, getattr(args, "include_context_id", None)
    )
    cache_admission_scope = validate_cache_admission_scope(
        candidates,
        relaxed_selected_logprob_max_abs_diff=getattr(
            args, "cache_admission_max_selected_logprob_diff", None
        ),
    )
    selected_diagnostic_owner_ids = sorted(
        {candidate.diagnostic_owner_id for candidate in candidates}
    )
    selected_free_request_ids = sorted(
        candidate.candidate_id
        for candidate in candidates
        if candidate.record_type == "free_coordinate_tree_root_request"
    )
    if rules.contract_mode == "production" and not selected_free_request_ids:
        _fail("production scoring requires at least one declared free-tree root")

    manifest_digests: Mapping[str, str] | None = None
    if args.artifact_manifest is not None:
        manifest_digests = load_artifact_manifest_digests(
            args.artifact_manifest, required_keys=list(source_files)
        )

    if args.validate_contract_only:
        source_digests = {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in source_files.items()
        }
        source_digests.update(phase_a_provenance_sources(phase_a_freeze_admission))
        return {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "runtime_execution_status": "contract_validated_no_model_loaded",
            "candidate_count": len(candidates),
            "owners_covered": len(selected_diagnostic_owner_ids),
            "contract_validation": contract_validation,
            "context_selection": context_selection,
            "cache_admission_scope": cache_admission_scope,
            "phase_a_freeze_admission": phase_a_freeze_admission,
            "source_digests": source_digests,
            "pure_core": core_status,
            "free_surface_status": "declared_validated_not_executed_contract_only",
            "restricted_surface_status": "declared_validated_not_executed_contract_only",
        }

    pin_fp32_parity_flags()
    if (
        args.infer_config is None
        or args.source_jsonl is None
        or args.output_dir is None
        or args.runtime_identity is None
    ):
        _fail(
            "live scoring requires --infer-config, --source-jsonl, --output-dir, and --runtime-identity"
        )

    output_dir: Path = args.output_dir.expanduser().resolve()
    jsonl_path = output_dir / OUTPUT_JSONL_NAME
    receipt_path = output_dir / RECEIPT_NAME
    existing_outputs = [
        str(path) for path in (jsonl_path, receipt_path) if path.exists()
    ]
    if existing_outputs and not args.force:
        _fail(
            "refusing to overwrite existing landscape-score artifacts; pass --force",
            paths=existing_outputs,
        )

    import torch as _torch

    if not _torch.cuda.is_available():
        _fail("CUDA is required to execute the runtime scorer")

    from src.config.fingerprint import sha256_json as config_sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import (
        DecodeRequest,
        GenerationPolicy,
        open_backend_session,
    )
    from src.inference.hf_backend import HFBackendSession
    from src.inference.image_plan import plan_image_batch
    from src.inference.pipeline import _processor_config, _template_config
    from src.inference.prompt import build_prompt_record
    from src.inference.runtime import assemble_frontend

    resolved = load_infer_config(args.infer_config.expanduser().resolve(strict=True))
    config = resolved.config
    live_config_admission = validate_live_config_binding(
        config=config,
        source_jsonl=args.source_jsonl,
    )
    frozen_runtime_identity, runtime_preload_admission = (
        validate_runtime_identity_preload(
            identity_path=args.runtime_identity,
            resolved_infer_fingerprint=resolved.fingerprint,
            source_jsonl=args.source_jsonl,
            candidates=candidates,
            rules=rules,
        )
    )
    if config.backend.type != "hf":
        raise ValueError("the runtime scorer requires backend.type: hf")

    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=config_sha256_json(
            config.generation.model_dump(mode="json")
        ),
    )
    raw_rows = load_raw_examples(args.source_jsonl.expanduser().resolve(strict=True))
    raw_by_image = {}
    for row in raw_rows:
        source_metadata = row.metadata.get("source")
        if not isinstance(source_metadata, Mapping):
            _fail("production source row metadata.source must be a mapping")
        raw_by_image[str(source_metadata.get("image_id"))] = row
    template = _template_config(config)

    candidates_by_image: dict[str, list[CandidateRow]] = {}
    for candidate in candidates:
        candidates_by_image.setdefault(candidate.image_id, []).append(candidate)

    parity_probe_candidate = select_mandatory_parity_probe(candidates)

    runtime_receipt_id = sha256_json(
        {
            "command": list(sys.argv),
            "owner_context_ledger": str(args.owner_context_ledger),
            "owner_ledger": str(args.owner_ledger),
            "prediction_row_ledger": str(args.prediction_row_ledger),
            "decision_rules": str(args.decision_rules),
            "candidates": str(args.candidates),
            "runtime_identity": str(args.runtime_identity),
            "started_at_unix": time.time(),
        }
    )

    all_rows: list[dict[str, Any]] = []
    restricted_rows: list[dict[str, Any]] = []
    free_surface_receipts: list[dict[str, Any]] = []
    owners_seen: set[str] = set()
    backend_receipt: dict[str, Any] = {}
    mandatory_parity_gate_result: dict[str, Any] | None = None
    batched_reforward_admission: dict[str, Any] | None = (
        {
            "schema_version": "batched_full_reforward_parity.v1",
            "status": "not_requested",
            "requested_batch_size": 1,
            "effective_batch_size": 1,
        }
        if args.full_reforward_batch_size == 1
        else None
    )
    scoring_backend_selection: dict[str, Any] | None = None
    scoring_backend_accounting: list[dict[str, Any]] = []
    runtime_postload_admission: dict[str, Any] = {}

    with open_backend_session(frontend.launch) as opened:
        if not isinstance(opened, HFBackendSession):
            raise RuntimeError("HF launch opened an unexpected backend session")
        backend_receipt = opened.receipt.to_artifact_dict()
        runtime_postload_admission = validate_runtime_identity_postload(
            frozen_identity=frozen_runtime_identity,
            backend_receipt=backend_receipt,
            resolved_infer_fingerprint=resolved.fingerprint,
        )
        tokenizer_identity = backend_receipt.get("tokenizer_identity") or {}
        model_identity = backend_receipt.get("model_identity") or {}
        model = opened._model  # noqa: SLF001
        if model is None:
            _fail("HF backend session did not expose its opened model")
        model.eval()

        # Computed once (hashes the full vocabulary id range); the core's own
        # __post_init__ validates the digest against the contiguous 0..V-1 domain.
        core_full_vocabulary_attestation = build_core_full_vocabulary_attestation(
            pure_core,
            expected_vocab_size=rules.model_vocab_size,
            tokenizer_identity_digest=runtime_preload_admission[
                "tokenizer_identity_sha256"
            ],
            model_identity_digest=runtime_preload_admission["model_identity_sha256"],
            runtime_rule_digest=rules.rules_digest,
        )

        ordered_image_groups = sorted(
            candidates_by_image.items(),
            key=lambda item: (
                item[0] != parity_probe_candidate.image_id,
                item[0],
            ),
        )
        for image_id, image_candidates in ordered_image_groups:
            raw = raw_by_image.get(image_id)
            if raw is None:
                _fail(
                    "candidate image_id is absent from the production source JSONL",
                    image_id=image_id,
                )
            image_plan = plan_image_batch(
                [raw],
                components=frontend.qwen,
                processor_config=_processor_config(config),
                row_indices=[0],
            ).rows[0]
            prompt_record = build_prompt_record(
                raw,
                template,
                processor=frontend.qwen.processor,
                row_index=0,
                merged_visual_tokens=image_plan.merged_visual_tokens,
            )
            expected_prompt_token_ids = tuple(
                int(v) for v in prompt_record.prompt_token_ids
            )
            image_grid_thw = tuple(int(v) for v in image_plan.expected_image_grid_thw)
            if len(image_grid_thw) != 3:
                _fail(
                    "planned image_grid_thw must contain exactly three dimensions",
                    image_id=image_id,
                    observed=image_grid_thw,
                )
            expected_image_grid_thw = (
                image_grid_thw[0],
                image_grid_thw[1],
                image_grid_thw[2],
            )
            request = DecodeRequest(
                request_id=f"landscape-score:{image_id}",
                chat_text=prompt_record.chat_text,
                input_prompt_token_ids=tuple(prompt_record.input_prompt_token_ids),
                expected_executed_prompt_token_ids=tuple(
                    prompt_record.expected_executed_prompt_token_ids
                ),
                image_path=image_plan.image_path,
                declared_image_width=image_plan.declared_width,
                declared_image_height=image_plan.declared_height,
                decoded_image_width=image_plan.decoded_width,
                decoded_image_height=image_plan.decoded_height,
                image_sha256=image_plan.image_content_sha256,
                expected_image_grid_thw=expected_image_grid_thw,
                logical_transform_id=image_plan.logical_transform_id,
                generation_policy=GenerationPolicy(
                    max_new_tokens=1,
                    repetition_penalty=1.0,
                    temperature=0.0,
                    top_p=1.0,
                    include_raw_model_logprob=True,
                ),
            )
            # The exact native-materialization seam: real processor invocation,
            # real pixel_values/image_grid_thw, tensors already moved to the
            # model's device. This -- not plan_image_batch/build_prompt_record
            # -- is what actually feeds the model.
            (
                native_inputs,
                executed_prompt_ids,
                _observed_grids,
                _executed_media_sha256,
            ) = opened._materialize_native_inputs((request,))  # noqa: SLF001
            if tuple(executed_prompt_ids[0]) != expected_prompt_token_ids:
                _fail(
                    "materialized executed prompt tokens differ from the expected production prompt "
                    "reconstruction; refusing to score against a divergent prompt",
                    image_id=image_id,
                )
            reconstructed_prompt_token_ids = list(executed_prompt_ids[0])
            batched_native_inputs: Mapping[str, Any] | None = None
            batched_full_reforward: Callable[
                [Sequence[Sequence[int]]], torch.Tensor
            ] | None = None

            grouped_by_history: dict[
                tuple[str, tuple[int, ...]], list[CandidateRow]
            ] = {}
            for candidate in image_candidates:
                grouped_by_history.setdefault(
                    (
                        candidate.context_id,
                        derive_generated_history_token_ids(candidate),
                    ),
                    [],
                ).append(candidate)

            ordered_history_groups = sorted(
                grouped_by_history.items(),
                key=lambda item: (
                    parity_probe_candidate not in item[1],
                    item[0][0],
                    item[0][1],
                ),
            )
            for (
                context_id,
                generated_history_token_ids,
            ), group in ordered_history_groups:
                group_prefix_token_ids = group[0].prefix_token_ids
                if any(
                    candidate.prefix_token_ids != group_prefix_token_ids
                    for candidate in group
                ):
                    _fail(
                        "one context/history group contains divergent literal prefixes",
                        context_id=context_id,
                    )
                for candidate in group:
                    verify_production_prompt_prefix(
                        candidate, reconstructed_prompt_token_ids
                    )
                group_id = sha256_json(
                    {
                        "image_id": image_id,
                        "context_id": context_id,
                        "prefix_token_ids": list(group_prefix_token_ids),
                    }
                )
                full_reforward = _build_full_reforward_closure(
                    model, native_prompt_inputs=native_inputs
                )
                parity_prefill: PrefillResult | None = None

                if mandatory_parity_gate_result is None:
                    if parity_probe_candidate not in group:
                        _fail(
                            "mandatory cache parity probe was not ordered before score rows"
                        )
                    first_coord_token_ids = parity_probe_candidate.coord_token_ids
                    if first_coord_token_ids is None:
                        _fail(
                            "selected parity-gate candidate lacks coordinate tokens",
                            candidate_id=parity_probe_candidate.candidate_id,
                        )
                    parity_prefill = prefill_context(
                        model,
                        native_prompt_inputs=native_inputs,
                        generated_history_token_ids=generated_history_token_ids,
                    )
                    x1, y1, x2, y2 = first_coord_token_ids
                    mandatory_parity_gate_result = run_cache_parity_gate(
                        backend=parity_prefill.backend,
                        prefill_logits=parity_prefill.prefill_logits,
                        prefix_token_ids=parity_probe_candidate.prefix_token_ids,
                        x1_token_id=x1,
                        y1_token_id=y1,
                        x2_token_id=x2,
                        y2_token_id=y2,
                        coordinate_token_id_start=rules.schema_tokens[
                            "coordinate_token_id_start"
                        ],
                        coordinate_token_id_end_exclusive=rules.schema_tokens[
                            "coordinate_token_id_end_exclusive"
                        ],
                        full_reforward=full_reforward,
                        expected_layer_count=EXPECTED_LIVE_QWEN_DECODER_LAYER_COUNT,
                        relaxed_selected_logprob_max_abs_diff=getattr(
                            args,
                            "cache_admission_max_selected_logprob_diff",
                            None,
                        ),
                    )
                    mandatory_parity_gate_result["admission_scope"] = (
                        cache_admission_scope
                    )
                    scoring_backend_selection = select_scoring_backend_from_parity(
                        mandatory_parity_gate_result
                    )

                if scoring_backend_selection is None:
                    _fail("scoring backend was not selected before candidate scoring")
                selected_backend = scoring_backend_selection["selected_backend"]
                if selected_backend == KV_CACHE_SCORING_BACKEND:
                    cache_prefill = parity_prefill or prefill_context(
                        model,
                        native_prompt_inputs=native_inputs,
                        generated_history_token_ids=generated_history_token_ids,
                    )
                    scoring_backend: CacheBackend = AccountingCacheBackend(
                        cache_prefill.backend,
                        context_id=context_id,
                        group_id=group_id,
                    )
                    scoring_root_logits = cache_prefill.prefill_logits
                elif selected_backend == FULL_REFORWARD_SCORING_BACKEND:
                    if (
                        args.full_reforward_batch_size > 1
                        and (
                            batched_reforward_admission is None
                            or batched_reforward_admission["status"] == "passed"
                        )
                    ):
                        if batched_native_inputs is None:
                            (
                                materialized_batch,
                                batch_executed_prompt_ids,
                                _batch_observed_grids,
                                _batch_executed_media_sha256,
                            ) = opened._materialize_native_inputs(  # noqa: SLF001
                                tuple(
                                    request
                                    for _ in range(args.full_reforward_batch_size)
                                )
                            )
                            if any(
                                tuple(row) != expected_prompt_token_ids
                                for row in batch_executed_prompt_ids
                            ):
                                _fail(
                                    "batched materialization changed the executed production prompt",
                                    image_id=image_id,
                                )
                            batched_native_inputs = materialized_batch
                            batched_full_reforward = (
                                _build_batched_full_reforward_closure(
                                    model,
                                    native_prompt_inputs=batched_native_inputs,
                                    maximum_batch_size=args.full_reforward_batch_size,
                                )
                            )
                        if batched_reforward_admission is None:
                            if batched_full_reforward is None:
                                _fail(
                                    "batched full reforward closure was not constructed before admission"
                                )
                            first_coord_token_ids = (
                                parity_probe_candidate.coord_token_ids
                            )
                            if first_coord_token_ids is None:
                                _fail(
                                    "batch parity candidate lacks coordinate tokens",
                                    candidate_id=parity_probe_candidate.candidate_id,
                                )
                            batched_reforward_admission = (
                                run_batched_reforward_parity_gate(
                                    prefix_token_ids=group_prefix_token_ids,
                                    coordinate_token_ids=first_coord_token_ids,
                                    coordinate_token_id_start=rules.schema_tokens[
                                        "coordinate_token_id_start"
                                    ],
                                    coordinate_token_id_end_exclusive=rules.schema_tokens[
                                        "coordinate_token_id_end_exclusive"
                                    ],
                                    full_reforward=full_reforward,
                                    batched_full_reforward=batched_full_reforward,
                                    requested_batch_size=args.full_reforward_batch_size,
                                )
                            )
                            if batched_reforward_admission["status"] != "passed":
                                batched_full_reforward = None
                    effective_reforward_batch_size = (
                        args.full_reforward_batch_size
                        if batched_reforward_admission is not None
                        and batched_reforward_admission["status"] == "passed"
                        else 1
                    )
                    uncached_backend = FullReforwardBackend(
                        root_prefix_token_ids=group_prefix_token_ids,
                        full_reforward=full_reforward,
                        batched_full_reforward=batched_full_reforward,
                        full_reforward_batch_size=effective_reforward_batch_size,
                        context_id=context_id,
                        group_id=group_id,
                    )
                    scoring_backend = uncached_backend
                    scoring_root_logits = uncached_backend.root_logits
                else:
                    _fail(
                        "unknown selected scoring backend",
                        observed_backend=selected_backend,
                    )

                prefetched_through_index = -1
                for candidate_index, candidate in enumerate(group):
                    if (
                        candidate.record_type != "free_coordinate_tree_root_request"
                        and candidate_index > prefetched_through_index
                    ):
                        candidate_chunk: list[CandidateRow] = []
                        for future in group[
                            candidate_index : candidate_index
                            + args.full_reforward_batch_size
                        ]:
                            if future.record_type == "free_coordinate_tree_root_request":
                                break
                            candidate_chunk.append(future)
                        _prefetch_backend_suffixes(
                            scoring_backend,
                            [
                                suffix
                                for chunk_candidate in candidate_chunk
                                for suffix in _candidate_reforward_suffixes(
                                    chunk_candidate
                                )
                            ],
                        )
                        prefetched_through_index = (
                            candidate_index + len(candidate_chunk) - 1
                        )
                    if candidate.record_type == "free_coordinate_tree_root_request":
                        free_attestation = build_attestation_context(
                            expected_vocab_size=rules.model_vocab_size,
                            tokenizer_identity=tokenizer_identity,
                            model_identity=model_identity,
                            rule_digest=rules.rules_digest,
                            runtime_receipt_id=runtime_receipt_id,
                            frozen_tokenizer_identity_digest=runtime_preload_admission[
                                "tokenizer_identity_sha256"
                            ],
                            frozen_model_identity_digest=runtime_preload_admission[
                                "model_identity_sha256"
                            ],
                        )
                        free_rows, free_receipt = execute_free_coordinate_tree(
                            request=candidate,
                            backend=scoring_backend,
                            prefill_logits=scoring_root_logits,
                            rules=rules,
                            attestation=free_attestation,
                        )
                        free_rows = bind_free_surface_score_rows(
                            rows=free_rows,
                            request=candidate,
                            owner_ledger=owner_ledger,
                            rules=rules,
                        )
                        owners_seen.add(candidate.diagnostic_owner_id)
                        all_rows.extend(free_rows)
                        free_surface_receipts.append(free_receipt)
                        continue
                    row = build_landscape_score_row(
                        candidate=candidate,
                        owner_ledger=owner_ledger,
                        prediction_row_ledger=prediction_row_ledger,
                        rules=rules,
                        pure_core=pure_core,
                        reconstructed_prompt_token_ids=reconstructed_prompt_token_ids,
                        backend=scoring_backend,
                        prefill_logits=scoring_root_logits,
                        tokenizer_identity=tokenizer_identity,
                        model_identity=model_identity,
                        runtime_receipt_id=runtime_receipt_id,
                        frozen_tokenizer_identity_digest=runtime_preload_admission[
                            "tokenizer_identity_sha256"
                        ],
                        frozen_model_identity_digest=runtime_preload_admission[
                            "model_identity_sha256"
                        ],
                    )
                    owners_seen.add(candidate.diagnostic_owner_id)
                    all_rows.append(row)
                    restricted_rows.append(row)

                accounting_method = getattr(scoring_backend, "accounting", None)
                if not callable(accounting_method):
                    _fail("selected scoring backend does not expose accounting")
                accounting_value = accounting_method()
                if not isinstance(accounting_value, Mapping):
                    _fail("selected scoring backend accounting must be a mapping")
                scoring_backend_accounting.append(dict(accounting_value))

    if mandatory_parity_gate_result is None or scoring_backend_selection is None:
        _fail("mandatory cache parity was not executed before score production")
    if batched_reforward_admission is None:
        if scoring_backend_selection["selected_backend"] == KV_CACHE_SCORING_BACKEND:
            batched_reforward_admission = {
                "schema_version": "batched_full_reforward_parity.v1",
                "status": "not_applicable_cache_backend_selected",
                "requested_batch_size": args.full_reforward_batch_size,
                "effective_batch_size": 1,
            }
        else:
            _fail("requested batched full reforward was not admitted before scoring")
    scoring_backend_admission = build_scoring_backend_admission(
        parity_gate=mandatory_parity_gate_result,
        selection=scoring_backend_selection,
        score_row_count=len(all_rows),
        per_context_group_accounting=scoring_backend_accounting,
    )

    # Conditional-y1 completeness digests: one per (context_id, diagnostic_owner_id)
    # group that actually produced y1 dense-scan rows. Downstream basin
    # registration/measurement consumes these digests instead of re-deriving
    # x1-anchor coverage from raw rows. Grouping is keyed by whatever x1
    # anchors were actually candidates for that owner/context; a group that is
    # not exactly covered fails fast inside the helper.
    y1_bin_count = (
        rules.schema_tokens["coordinate_token_id_end_exclusive"]
        - rules.schema_tokens["coordinate_token_id_start"]
    )
    y1_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in all_rows:
        if row.get("request_kind") == "dense_scan" and row.get("scan_slot") == "y1":
            key = (row["context_id"], row["diagnostic_owner_id"])
            y1_groups.setdefault(key, []).append(row)
    if rules.contract_version == CANDIDATE_SCHEMA_VERSION:
        conditional_y1_completeness = build_conditional_y1_completeness_attestations(
            pure_core=pure_core,
            candidates=candidates,
            scored_rows=restricted_rows,
            owner_context_ledger=owner_context_ledger,
            rules=rules,
        )
    else:
        conditional_y1_completeness = [
            compute_conditional_y1_completeness_digest(
                context_id=context_id,
                diagnostic_owner_id=diagnostic_owner_id,
                rule_digest=rules.rules_digest,
                admitted_x1_token_ids=[
                    int(row["fixed_coord_token_ids"][0]) for row in rows
                ],
                y1_bin_count=y1_bin_count,
                dense_scan_rows=rows,
            )
            for (context_id, diagnostic_owner_id), rows in sorted(y1_groups.items())
        ]

    environment = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "transformers_version": _installed_package_version("transformers"),
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "tf32": pin_fp32_parity_flags(),
    }
    receipt = build_execution_receipt(
        command=sys.argv,
        source_files=source_files,
        manifest_digests=manifest_digests,
        rules=rules,
        rules_path=args.decision_rules,
        backend_receipt=backend_receipt,
        candidate_count=len(all_rows),
        owners_covered=len(owners_seen),
        pure_core_status_dict=core_status,
        environment=environment,
        runtime_execution_status=LIVE_SCORING_PENDING_SEAL_STATUS,
    )
    receipt["source_digests"].update(
        phase_a_provenance_sources(phase_a_freeze_admission)
    )
    receipt["runtime_receipt_id"] = runtime_receipt_id
    receipt["runtime_identity_admission"] = {
        "config": live_config_admission,
        "preload": runtime_preload_admission,
        "postload": runtime_postload_admission,
    }
    receipt["mandatory_cache_parity_gate"] = mandatory_parity_gate_result
    receipt["batched_full_reforward_admission"] = batched_reforward_admission
    receipt["scoring_backend_admission"] = scoring_backend_admission
    receipt["execution_architecture"] = execution_architecture_for_admission(
        scoring_backend_admission
    )
    receipt["conditional_y1_attestations"] = conditional_y1_completeness
    receipt["conditional_y1_completeness_contract"] = (
        "full_core_v2_ConditionalY1CompletenessAttestation"
        if rules.contract_version == CANDIDATE_SCHEMA_VERSION
        else "legacy_fixture_local_digest"
    )
    receipt["core_v2_full_vocabulary_attestation"] = dataclasses.asdict(
        core_full_vocabulary_attestation
    )
    receipt["contract_validation"] = contract_validation
    receipt["context_selection"] = context_selection
    receipt["phase_a_freeze_admission"] = phase_a_freeze_admission
    receipt["surfaces"] = {
        "free_coordinate_tree": {
            "declared_request_ids": selected_free_request_ids,
            "receipts": free_surface_receipts,
            "null_semantics": "bounded_free_search_null_is_non_evidence_for_absence",
        },
        "restricted_candidate_bank": {
            "conditional_y1_attestation_count": len(conditional_y1_completeness),
            "membership": "authoritative_candidate_builder_v2_pre_score_frozen",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_temp = output_dir / f".{OUTPUT_JSONL_NAME}.tmp"
    receipt_temp = output_dir / f".{RECEIPT_NAME}.tmp"
    try:
        with jsonl_temp.open("w", encoding="utf-8") as handle:
            for row in all_rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n"
                )
        receipt = seal_execution_receipt(
            receipt,
            scores_path=jsonl_temp,
            rows=all_rows,
        )
        receipt["output_artifacts"]["landscape_scores"]["path"] = str(jsonl_path)
        receipt_temp.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        jsonl_temp.replace(jsonl_path)
        receipt_temp.replace(receipt_path)
    finally:
        jsonl_temp.unlink(missing_ok=True)
        receipt_temp.unlink(missing_ok=True)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run(args)
    print(
        json.dumps(
            {"rows": receipt["candidate_count"], "owners": receipt["owners_covered"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
