#!/usr/bin/env python3
"""Successor-local, generic score-shard merge for the sorted false-negative
mechanism decomposition unit (``2026-08-02-sorted-false-negative-mechanism-
decomposition``).

This is a CPU-only artifact merger. It does not score, reinterpret, or select
candidates, and it never loads a model, tokenizer, or GPU. It joins an
arbitrary, non-fixed number of per-rank raw score shards -- each produced by
one invocation of the successor's own scorer,
``score_sorted_fn_fixed_budget.py`` -- against the successor's own frozen
``owner-context-ledger.jsonl``, ``landscape-decision-rules.json``,
``fixed-budget-candidates.jsonl``, planner receipt
(``prepare_sorted_fn_successor_inputs.py`` output), and FN mechanism registry
(``build_sorted_fn_mechanism_registry.py`` output).

Predecessor scorer artifacts are never accepted
----------------------------------------------
``score_sorted_owner_basin_landscape.py`` (the predecessor's closed
candidate-builder-v2 production scorer) cannot score the fixed-budget lattice
at all -- it requires a complete conditional-x1 domain plus a declared
free-coordinate-tree root, which the fixed-budget lattice has neither of (see
``score_sorted_fn_fixed_budget.py``'s module docstring). This merger
therefore requires every shard to carry the successor scorer's own
``schema_version``/``unit_id`` and explicitly rejects any shard whose
receipt or rows carry the predecessor's own module constants -- that is
either stale/foreign provenance or an attempt to impersonate a shard this
scorer never produced.

This merger *does* reuse the predecessor's low-level scoring primitives
(``FullReforwardBackend``, ``score_complete_box_candidate``, the batched-
reforward parity gate, and the backend-selection/admission constants) --
``score_sorted_fn_fixed_budget.py`` imports them unmodified. That reuse is
recorded on every live-sealed shard receipt under
``implementation_provenance.relevant_file_digests``, keyed by the exact
predecessor primitives file path. This merger validates that digest is
identical across every shard and republishes it, verbatim, under this
document's own ``imported_predecessor_primitive_provenance`` -- a distinct,
exact code/helper-digest surface, never conflated with scorer identity or
this document's own ``unit_id`` ownership claim.

Unlike the predecessor's ``merge_sorted_owner_basin_landscape_shards.py``,
this merger does not assume a fixed four-context world: it admits any
non-empty set of shards covering any number of ledger context IDs ("logical
roles"). The authoritative candidate identity space for this unit is the
score-independent fixed-budget L0/L1/scalar_smoke lattice in
``fixed-budget-candidates.jsonl``, keyed by ``(owner_context_id,
candidate_id)``; every merged row is cross-validated field-by-field against
its declared fixed-budget candidate (rung, region, population, control
membership, coordinate tokens, source digest) so a stale or substituted
candidate definition can never silently enter the merged surface.

Two ledger rows ("logical roles") may legitimately share one exact
self-prefix token digest (``owner-context-ledger.jsonl``'s
``execution_dedup_key``) -- for example, two owners' ``root`` contexts on the
same image, prior to any generated row. A shard producer may reuse one
forward pass's logits for two such roles rather than re-running the model.
This merger accepts that only when the reused rows are truly identical
executions: for any coordinate-token sequence that is scored under more than
one context sharing an ``execution_dedup_key``, every observed raw score for
that exact sequence must agree within the frozen ``numeric_tolerance``. Every
logical role (context ID) still receives its own distinct row entries; dedup
never collapses two roles into one.

Separately, two *different* shards may cover the *same* context (a resharded
or retried run). Per-shard context sets must either be pairwise disjoint, or
-- for any shared context -- every one of that context's rows must be
byte-identical (canonical JSON) across the overlapping shards; the merge then
keeps exactly one copy. Any other overlap (partial coverage, or differing
content for the same candidate) is a hard failure.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import score_sorted_owner_basin_landscape as scorer  # noqa: E402
from scripts.research.build_sorted_owner_basin_candidates import (  # noqa: E402
    OWNER_CONTEXT_LEDGER_SCHEMA_VERSION,
)
from scripts.research.build_sorted_fn_mechanism_registry import (  # noqa: E402
    SCHEMA_VERSION as REGISTRY_SCHEMA_VERSION,
    UNIT_ID,
)
from scripts.research.prepare_sorted_fn_successor_inputs import (  # noqa: E402
    FIXED_BUDGET_SCHEMA_VERSION,
    MECHANISM_DECISION_RULES_SCHEMA_VERSION,
    RECEIPT_SCHEMA_VERSION as PLANNER_RECEIPT_SCHEMA_VERSION,
)
from scripts.research.prepare_sorted_fn_successor_inputs import (  # noqa: E402
    _STALE_FIXED_BUDGET_SCHEMA_VERSIONS as STALE_FIXED_BUDGET_SCHEMA_VERSIONS,
    NO_SAME_DESCRIPTION_NON_OVERLAPPING_OWNER,
)
# Region/rung vocabularies are pure, stable domain constants already frozen
# by the successor's own CPU-only reanalysis consumer; reused read-only
# rather than silently re-declared and risking drift. ``_VALID_POPULATIONS``
# is declared locally instead of imported: that sibling module still only
# admits {"target", "decoy"} and is not owned by this change, but v2
# fixed-budget candidates legitimately add a third "reference" population
# (see build_reference_family_boxes/other_owner_reference upstream).
from scripts.research.reanalyze_sorted_fn_fixed_budget_controls import (  # noqa: E402
    _VALID_REGIONS,
    _VALID_RUNGS,
)

_VALID_POPULATIONS = frozenset({"target", "decoy", "reference"})


MERGE_SCHEMA_VERSION = "sorted_fn_successor_score_shard_merge.v2"
MERGED_SCORE_NAME = "fn-successor-merged-scores.jsonl"
MERGED_RECEIPT_NAME = "fn-successor-merged-scores-receipt.json"

# ``score_sorted_fn_fixed_budget.py`` imports several private helpers from
# *this* module (planner/registry/ledger/fixed-budget loaders, canonical
# JSON/digest helpers) at its own module top level, so importing it back
# here to read its constants would be a circular import. These four literals
# are its own frozen module constants, duplicated read-only (not derived);
# ``test_merge_sorted_fn_successor_score_shards.py`` asserts they stay in
# sync with the real module.
SUCCESSOR_SCORE_ROW_SCHEMA_VERSION = "sorted_fn_fixed_budget_scores.v1"
SUCCESSOR_SCORE_RECEIPT_SCHEMA_VERSION = "sorted_fn_fixed_budget_scores_receipt.v1"
SUCCESSOR_LIVE_SCORING_SEALED_STATUS = "successor_fixed_budget_live_scoring_completed_artifacts_sealed"
SUCCESSOR_OUTPUT_JSONL_NAME = "fn-fixed-budget-scores.jsonl"
SUCCESSOR_RECEIPT_NAME = "fn-fixed-budget-scores-receipt.json"
#: This successor unit's own scorer never claims the predecessor's unit ID.
SUCCESSOR_SCORER_UNIT_ID = UNIT_ID

# The predecessor's own module constants -- imported only to positively
# detect and reject impersonation/foreign-provenance shards, never accepted
# as a valid shard identity below.
PREDECESSOR_SCORE_ROW_SCHEMA_VERSION = scorer.SCHEMA_VERSION
PREDECESSOR_SCORE_RECEIPT_SCHEMA_VERSION = scorer.RECEIPT_SCHEMA_VERSION
PREDECESSOR_SCORER_UNIT_ID = scorer.UNIT_ID

#: The exact predecessor low-level primitives file whose functions this
#: successor scorer reuses unmodified; kept as a separate code-digest surface.
PREDECESSOR_PRIMITIVES_FILE = "scripts/research/score_sorted_owner_basin_landscape.py"

_FIXED_BUDGET_IDENTITY_FIELDS = (
    "rung",
    "region",
    "population",
    "is_control",
    "control_kind",
    "matched_control_group",
    "family_id",
    "iou_to_target",
    "source_digest",
    "candidate_neighborhood_id",
    "candidate_neighborhood_member",
    "mechanism_decision_rules_sha256",
    "other_owner_gt_owner_id",
    "other_owner_selection_trace",
    "exact_gt_singleton_member",
    "exact_gt_singleton_id",
)
#: Only present (non-null) on ``population == "reference"`` rows; a row of
#: any other population must never carry a fabricated value for these.
_REFERENCE_ONLY_FIELDS = ("other_owner_gt_owner_id", "other_owner_selection_trace")
#: F3 (the collision-statistic exact-GT-box score): exactly one
#: ``near_gt_micro`` target row per (owner_context_id, rung) is marked
#: ``exact_gt_singleton_member=True``, discovered by the planner scanning
#: actual geometry -- never assumed to be a fixed local index (see
#: prepare_sorted_fn_successor_inputs.build_fixed_budget_candidates_for_context).
_EXACT_GT_SINGLETON_ID_PREFIX = "exact-gt:"
_REQUIRED_SOURCE_DIGEST_KEYS = (
    "registry",
    "owner_context_ledger",
    "decision_rules",
    "mechanism_decision_rules",
    "fixed_budget_candidates",
    "planner_receipt",
    "runtime_identity",
)
_RAW_LOGPROB_SLOTS = ("x1_logprob", "y1_logprob", "x2_logprob", "y2_logprob")
_AUXILIARY_POLICY_VIEWS = (scorer._policy_view_key(1.0), scorer._policy_view_key(1.10))  # noqa: SLF001
_BATCH_ADMISSION_STATUSES = frozenset({"not_requested", "passed", "failed_scalar_fallback_required"})
_ADMISSION_IDENTITY_KEYS = (
    "parity_status",
    "cache_admission_policy",
    "selected_backend",
    "cache_enabled",
    "use_cache",
    "atol",
    "rtol",
    "all_score_rows_backend",
    "backend_mixing_detected",
    "fallback_trigger",
)
NATIVE_REPETITION_PENALTY_STRATUM = 1.0
_NUMERIC_ROUND_TRIP_TOLERANCE = 1e-6

#: ``run_sorted_fn_successor_behavior.validate_live_runtime_identity``'s own
#: sealed postload status/check-key contract. That function raises before
#: ever returning if any check is not literally ``True``, so a genuinely
#: live-sealed receipt can only ever carry this exact status with every
#: check passing -- this merger still validates both explicitly (fail-closed
#: against a malformed or hand-crafted receipt, never trusting the producer).
_POSTLOAD_STATUS = "passed_before_generation"
_POSTLOAD_CHECK_KEYS = frozenset(
    {
        "model_identity",
        "model_identity_fingerprint",
        "tokenizer_identity",
        "runtime_identity",
        "effective_settings_extra_fields",
        "observed_attn_implementation",
        "observed_model_dtype",
        "processor_identity_fingerprint",
        "resolved_config_fingerprint",
        "generation_config_fingerprint",
        "precision",
    }
)


class MergeContractError(RuntimeError):
    """A precondition for a conclusion-bearing successor shard merge failed."""


def _fail(message: str) -> NoReturn:
    raise MergeContractError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a JSON object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(f"{label} must be a JSON array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty string")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        _fail(f"{label} must be a finite number")
    return float(value)


_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256_hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_HEX_PATTERN.match(value):
        _fail(f"{label} must be a lowercase 64-character hex sha256 digest")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {path} as JSON: {exc}")
    return dict(_mapping(value, str(path)))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.resolve(strict=True).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    _fail(f"{path}:{line_number} is not valid JSON: {exc}")
                rows.append(dict(_mapping(value, f"{path}:{line_number}")))
    except OSError as exc:
        _fail(f"cannot read {path}: {exc}")
    return rows


def _write_create_or_identical(path: Path, encoded: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
        return True
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != encoded:
            _fail(f"{path} already exists with different content; refusing to overwrite")
        return False


# ---------------------------------------------------------------------------
# Successor-owned upstream artifacts
#
# NOTE: score_sorted_fn_fixed_budget.py imports every function in this
# section directly (``merger._load_planner_receipt`` etc). Keep names and
# signatures stable; the reverse dependency is deliberate reuse, not
# incidental coupling (see that module's own docstring).
# ---------------------------------------------------------------------------


def _load_planner_receipt(path: Path) -> dict[str, Any]:
    receipt = _read_json(path)
    if receipt.get("schema_version") != PLANNER_RECEIPT_SCHEMA_VERSION:
        _fail(f"planner receipt.schema_version must be {PLANNER_RECEIPT_SCHEMA_VERSION!r}")
    if receipt.get("unit_id") != UNIT_ID:
        _fail("planner receipt.unit_id is not this successor unit's own unit_id")
    content = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    if sha256_json(content) != receipt.get("receipt_digest"):
        _fail("planner receipt_digest does not reconstruct from its own content; stale or tampered")
    return receipt


def _bind_planner_output(
    *, receipt: Mapping[str, Any], output_key: str, path: Path, label: str
) -> Mapping[str, Any]:
    outputs = _mapping(receipt.get("outputs"), "planner receipt.outputs")
    entry = _mapping(outputs.get(output_key), f"planner receipt.outputs.{output_key}")
    if entry.get("sha256") != sha256_file(path):
        _fail(f"{label} sha256 does not match the planner receipt's declared digest; stale input")
    return entry


def _bind_planner_source(*, receipt: Mapping[str, Any], source_key: str, path: Path, label: str) -> None:
    sources = _mapping(receipt.get("sources"), "planner receipt.sources")
    entry = _mapping(sources.get(source_key), f"planner receipt.sources.{source_key}")
    if entry.get("sha256") != sha256_file(path):
        _fail(f"{label} sha256 does not match the planner receipt's declared digest; stale input")


def _load_registry(path: Path, *, planner_receipt: Mapping[str, Any]) -> dict[str, Any]:
    _bind_planner_source(receipt=planner_receipt, source_key="registry", path=path, label="FN mechanism registry")
    registry = _read_json(path)
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        _fail(f"FN mechanism registry.schema_version must be {REGISTRY_SCHEMA_VERSION!r}")
    if registry.get("unit_id") != UNIT_ID:
        _fail("FN mechanism registry.unit_id is not this successor unit's own unit_id")
    return registry


def _load_ledger(path: Path, *, planner_receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entry = _bind_planner_output(
        receipt=planner_receipt, output_key="owner_context_ledger", path=path, label="owner-context-ledger.jsonl"
    )
    rows = _read_jsonl(path)
    if entry.get("row_count") != len(rows):
        _fail("owner-context-ledger.jsonl row_count does not match the planner receipt")
    if planner_receipt.get("context_count") != len(rows):
        _fail("owner-context-ledger.jsonl row count does not match the planner receipt's context_count")
    by_context: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("schema_version") != OWNER_CONTEXT_LEDGER_SCHEMA_VERSION:
            _fail(f"ledger row has an unexpected schema_version: {row.get('schema_version')!r}")
        context_id = _string(row.get("context_id"), "ledger row.context_id")
        if context_id in by_context:
            _fail(f"owner-context-ledger.jsonl has a duplicate context_id: {context_id}")
        _string(row.get("gt_owner_id"), f"ledger row {context_id}.gt_owner_id")
        _string(row.get("diagnostic_owner_id"), f"ledger row {context_id}.diagnostic_owner_id")
        context_tokens = _mapping(row.get("context_tokens"), f"ledger row {context_id}.context_tokens")
        _string(context_tokens.get("token_ids_sha256"), f"ledger row {context_id}.context_tokens.token_ids_sha256")
        if row.get("execution_dedup_key") != context_tokens.get("token_ids_sha256"):
            _fail(f"ledger row {context_id} execution_dedup_key does not match its own context_tokens digest")
        stratum = row.get("native_repetition_penalty_stratum")
        if stratum != NATIVE_REPETITION_PENALTY_STRATUM:
            _fail(f"ledger row {context_id} native_repetition_penalty_stratum must be {NATIVE_REPETITION_PENALTY_STRATUM!r}")
        # "Reference selection/status provenance" at the context level: the
        # planner freezes, per context, whether a same-normalized-description,
        # zero-overlap owner was bound (``bound:<gt_owner_id>``) before any
        # fixed-budget candidate is generated.
        status = _string(row.get("other_owner_reference_status"), f"ledger row {context_id}.other_owner_reference_status")
        if status != NO_SAME_DESCRIPTION_NON_OVERLAPPING_OWNER and not status.startswith("bound:"):
            _fail(f"ledger row {context_id}.other_owner_reference_status has an unrecognized value: {status!r}")
        by_context[context_id] = row
    return by_context


def _validate_reference_selection_bindings(
    *, ledger_by_context: Mapping[str, Mapping[str, Any]], fixed_budget_by_key: Mapping[tuple[str, str], Mapping[str, Any]]
) -> None:
    """Cross-check each context's frozen reference-selection status against its candidates.

    "no_same_description_non_overlapping_owner" contexts must generate zero
    ``population == "reference"`` candidates. A "bound:<gt_owner_id>" context
    must generate at least one (the frozen ``near_other_micro`` lattice), all
    naming that exact owner -- never a different or absent one.
    """

    reference_owner_by_context: dict[str, set[str]] = {}
    for (context_id, _candidate_id), row in fixed_budget_by_key.items():
        if row.get("population") == "reference":
            reference_owner_by_context.setdefault(context_id, set()).add(str(row.get("other_owner_gt_owner_id")))

    for context_id, ledger_row in ledger_by_context.items():
        status = str(ledger_row["other_owner_reference_status"])
        observed_owners = reference_owner_by_context.get(context_id, set())
        if status == NO_SAME_DESCRIPTION_NON_OVERLAPPING_OWNER:
            if observed_owners:
                _fail(
                    f"context {context_id!r} is frozen as {NO_SAME_DESCRIPTION_NON_OVERLAPPING_OWNER!r} but declares "
                    f"population=reference candidates for owner(s) {sorted(observed_owners)}"
                )
        else:
            expected_owner = status.split("bound:", 1)[1]
            if observed_owners != {expected_owner}:
                _fail(
                    f"context {context_id!r} is frozen as bound to owner {expected_owner!r} but its reference "
                    f"candidates name owner(s) {sorted(observed_owners)}"
                )


def _expected_runtime_identity(ledger_by_context: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Derive the one expected model/tokenizer identity from the ledger itself.

    Every ledger row carries its own ``vocabulary_attestation``; this unit's
    ledger is single-checkpoint, so every row must agree. This is the
    "explicit expected runtime/scorer identity" this merger holds shards to,
    rather than merely checking shards agree with *each other*.
    """

    identities = set()
    for context_id, row in ledger_by_context.items():
        attestation = _mapping(row.get("vocabulary_attestation"), f"ledger row {context_id}.vocabulary_attestation")
        model_identity = _string(attestation.get("model_identity_sha256"), f"ledger row {context_id} model_identity_sha256")
        tokenizer_identity = _string(
            attestation.get("tokenizer_identity_sha256"), f"ledger row {context_id} tokenizer_identity_sha256"
        )
        identities.add((model_identity, tokenizer_identity))
    if len(identities) != 1:
        _fail("owner-context-ledger.jsonl does not declare exactly one model/tokenizer identity")
    model_identity, tokenizer_identity = next(iter(identities))
    return {"model_identity": model_identity, "tokenizer_identity": tokenizer_identity}


def _load_mechanism_decision_rules(
    path: Path, *, planner_receipt: Mapping[str, Any], landscape_rules_path: Path, registry_document: Mapping[str, Any]
) -> dict[str, Any]:
    """Load, self-verify, and parent-bind the successor-owned mechanism rules.

    ``mechanism-decision-rules.json`` is a distinct artifact from the
    execution ``landscape-decision-rules.json``: it freezes the *analysis*
    policy (candidate generator, proposal weights, geometry regions,
    calibration, matched-control strata, sampling escalation -- see
    unit.md's "Frozen quantitative decision functional"), never the token/
    coordinate/model execution contract the scorer runs against. It is
    child-to-parent bound to the execution rules and to the FN mechanism
    registry via ``upstream_digests``, and self-verifies via ``self_digest``
    (a JSON-content digest, distinct from the raw file's sha256).
    """

    entry = _bind_planner_output(
        receipt=planner_receipt,
        output_key="mechanism_decision_rules",
        path=path,
        label="mechanism-decision-rules.json",
    )
    document = _read_json(path)
    if document.get("schema_version") != MECHANISM_DECISION_RULES_SCHEMA_VERSION:
        _fail(f"mechanism-decision-rules.json schema_version must be {MECHANISM_DECISION_RULES_SCHEMA_VERSION!r}")
    if document.get("unit_id") != UNIT_ID:
        _fail("mechanism-decision-rules.json.unit_id is not this successor unit's own unit_id")
    content = {key: value for key, value in document.items() if key != "self_digest"}
    if sha256_json(content) != document.get("self_digest"):
        _fail("mechanism-decision-rules.json self_digest does not reconstruct from its own content; stale or tampered")

    upstream = _mapping(document.get("upstream_digests"), "mechanism-decision-rules.json.upstream_digests")
    # "Mechanism parent execution digest": mechanism rules are frozen on top
    # of one exact execution (landscape) rules file; any drift there voids
    # every downstream calibration/eligibility receipt bound to this document.
    if upstream.get("execution_landscape_decision_rules_sha256") != sha256_file(landscape_rules_path):
        _fail(
            "mechanism-decision-rules.json.upstream_digests.execution_landscape_decision_rules_sha256 does not "
            "match the current landscape-decision-rules.json; stale mechanism-rules/execution-rules parent binding"
        )
    if upstream.get("fn_mechanism_registry_sha256") != registry_document.get("registry_digest"):
        _fail(
            "mechanism-decision-rules.json.upstream_digests.fn_mechanism_registry_sha256 does not match the "
            "current FN mechanism registry's own registry_digest; stale mechanism-rules/registry binding"
        )
    # F3 mechanism-rule binding: the declared candidate-row field convention
    # for the exact-GT-box singleton statistic must match the literal field
    # names this merger (and the scorer/attestor) actually validate.
    singleton_section = _mapping(document.get("exact_gt_singleton"), "mechanism-decision-rules.json.exact_gt_singleton")
    if singleton_section.get("member_field") != "exact_gt_singleton_member" or singleton_section.get("id_field") != "exact_gt_singleton_id":
        _fail("mechanism-decision-rules.json.exact_gt_singleton declares an unrecognized field-name binding (F3)")
    _ = entry  # already validated by _bind_planner_output; kept for readability
    return document


def _load_fixed_budget_candidates(
    path: Path, *, planner_receipt: Mapping[str, Any], ledger_by_context: Mapping[str, Mapping[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    entry = _bind_planner_output(
        receipt=planner_receipt,
        output_key="fixed_budget_candidates",
        path=path,
        label="fixed-budget-candidates.jsonl",
    )
    rows = _read_jsonl(path)
    if entry.get("row_count") != len(rows):
        _fail("fixed-budget-candidates.jsonl row_count does not match the planner receipt")
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        owner_context_id = _string(row.get("owner_context_id"), "fixed-budget row.owner_context_id")
        if owner_context_id not in ledger_by_context:
            _fail(f"fixed-budget-candidates.jsonl names an unknown context: {owner_context_id}")
        candidate_id = _string(row.get("candidate_id"), "fixed-budget row.candidate_id")
        key = (owner_context_id, candidate_id)
        if key in by_key:
            _fail(f"fixed-budget-candidates.jsonl has a duplicate candidate: {key}")
        schema_version = row.get("schema_version")
        if schema_version in STALE_FIXED_BUDGET_SCHEMA_VERSIONS:
            _fail(f"fixed-budget row {candidate_id} carries a stale v1 fixed-budget-candidates schema_version; v1 fallbacks are not accepted")
        if schema_version != FIXED_BUDGET_SCHEMA_VERSION:
            _fail(f"fixed-budget row {candidate_id} has an unrecognized schema_version: {schema_version!r}")
        rung = _string(row.get("rung"), f"fixed-budget row {candidate_id}.rung")
        if rung not in _VALID_RUNGS:
            _fail(f"fixed-budget row {candidate_id} has an invalid rung: {rung!r}")
        region = _string(row.get("region"), f"fixed-budget row {candidate_id}.region")
        if region not in _VALID_REGIONS:
            _fail(f"fixed-budget row {candidate_id} has an invalid region: {region!r}")
        population = _string(row.get("population"), f"fixed-budget row {candidate_id}.population")
        if population not in _VALID_POPULATIONS:
            _fail(f"fixed-budget row {candidate_id} has an invalid population: {population!r}")
        coord_token_ids = [
            int(v) for v in _sequence(row.get("coord_token_ids"), f"fixed-budget row {candidate_id}.coord_token_ids")
        ]
        _string(
            row.get("mechanism_decision_rules_sha256"),
            f"fixed-budget row {candidate_id}.mechanism_decision_rules_sha256",
        )
        member = row.get("candidate_neighborhood_member")
        if not isinstance(member, bool):
            _fail(f"fixed-budget row {candidate_id}.candidate_neighborhood_member must be a boolean")
        neighborhood_id = row.get("candidate_neighborhood_id")
        if member:
            _string(neighborhood_id, f"fixed-budget row {candidate_id}.candidate_neighborhood_id")
        elif neighborhood_id is not None:
            _fail(f"fixed-budget row {candidate_id} declares candidate_neighborhood_id without candidate_neighborhood_member")
        if population == "reference":
            if region != "other_owner":
                _fail(f"fixed-budget row {candidate_id} has population=reference but region={region!r}, not other_owner")
            _string(row.get("other_owner_gt_owner_id"), f"fixed-budget row {candidate_id}.other_owner_gt_owner_id")
            trace = _mapping(
                row.get("other_owner_selection_trace"), f"fixed-budget row {candidate_id}.other_owner_selection_trace"
            )
            if not trace:
                _fail(f"fixed-budget row {candidate_id}.other_owner_selection_trace must be non-empty")
        else:
            for field_name in _REFERENCE_ONLY_FIELDS:
                if row.get(field_name) is not None:
                    _fail(f"fixed-budget row {candidate_id} declares {field_name!r} but population={population!r}, not reference")
        # F3: exactly one near_gt_micro target row per (owner_context_id,
        # rung) is the exact-GT-box singleton; never assumed at a fixed
        # local index -- the planner discovers it by scanning geometry.
        singleton_member = row.get("exact_gt_singleton_member")
        if not isinstance(singleton_member, bool):
            _fail(f"fixed-budget row {candidate_id}.exact_gt_singleton_member must be a boolean")
        singleton_id = row.get("exact_gt_singleton_id")
        if singleton_member:
            _string(singleton_id, f"fixed-budget row {candidate_id}.exact_gt_singleton_id")
            if not singleton_id.startswith(_EXACT_GT_SINGLETON_ID_PREFIX):
                _fail(f"fixed-budget row {candidate_id}.exact_gt_singleton_id has an unrecognized form: {singleton_id!r}")
            if population != "target" or row.get("family_id") != "near_gt_micro":
                _fail(f"fixed-budget row {candidate_id} is exact_gt_singleton_member but is not a near_gt_micro target row")
        elif singleton_id is not None:
            _fail(f"fixed-budget row {candidate_id} declares exact_gt_singleton_id without exact_gt_singleton_member")
        by_key[key] = {**row, "coord_token_ids": coord_token_ids}

    singleton_groups: dict[tuple[str, str], list[str]] = {}
    for (owner_context_id, candidate_id), row in by_key.items():
        if row.get("population") == "target" and row.get("family_id") == "near_gt_micro" and row.get("exact_gt_singleton_member"):
            singleton_groups.setdefault((owner_context_id, row["rung"]), []).append(candidate_id)
    for (owner_context_id, rung), members in singleton_groups.items():
        if len(members) != 1:
            _fail(
                f"context {owner_context_id!r} rung {rung!r} must have exactly one exact_gt_singleton_member "
                f"near_gt_micro target row; found {len(members)}"
            )
    near_gt_micro_groups = {
        (row["owner_context_id"], row["rung"])
        for row in by_key.values()
        if row.get("population") == "target" and row.get("family_id") == "near_gt_micro"
    }
    missing_singleton = near_gt_micro_groups - set(singleton_groups)
    if missing_singleton:
        _fail(
            "fixed-budget-candidates.jsonl has near_gt_micro target rows with no exact_gt_singleton_member "
            f"row for context/rung group(s): {sorted(missing_singleton)[:5]}"
        )
    return by_key


def _validate_fixed_budget_mechanism_binding(
    fixed_budget_by_key: Mapping[tuple[str, str], Mapping[str, Any]], *, mechanism_decision_rules_sha256: str
) -> None:
    """Every fixed-budget candidate must bind to the exact loaded mechanism rules.

    Each row's ``mechanism_decision_rules_sha256`` is the *raw file* sha256
    of ``mechanism-decision-rules.json`` (the planner's own
    ``sha256_file(mechanism_decision_rules_path)``) -- distinct from the
    document's internal ``self_digest`` (a JSON-content digest excluding
    that very key). Both are validated, for different purposes: the file
    sha256 here proves this row was generated against the exact artifact on
    disk; ``self_digest`` (checked in :func:`_load_mechanism_decision_rules`)
    proves the artifact's own content was never tampered with after freeze.
    """

    for key, row in fixed_budget_by_key.items():
        if row.get("mechanism_decision_rules_sha256") != mechanism_decision_rules_sha256:
            _fail(f"fixed-budget candidate {key!r} mechanism_decision_rules_sha256 does not match the loaded mechanism-decision-rules.json")


# ---------------------------------------------------------------------------
# Shard validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardInput:
    scores_path: Path
    receipt_path: Path


@dataclass
class ValidatedShard:
    scores_path: Path
    receipt_path: Path
    receipt: dict[str, Any]
    rows: list[dict[str, Any]]
    identity_projection: dict[str, Any] = field(repr=False)
    selected_rungs: frozenset[str] = field(default_factory=frozenset)


def _reject_predecessor_provenance(*, schema_version: Any, unit_id: Any, label: str) -> None:
    if schema_version == PREDECESSOR_SCORE_RECEIPT_SCHEMA_VERSION or schema_version == PREDECESSOR_SCORE_ROW_SCHEMA_VERSION:
        _fail(f"{label} carries the predecessor scorer's schema_version; predecessor scorer artifacts are not accepted here")
    if unit_id == PREDECESSOR_SCORER_UNIT_ID:
        _fail(f"{label} carries the predecessor scorer's unit_id; predecessor scorer artifacts are not accepted here")


def _validate_implementation_provenance(receipt: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    provenance = _mapping(receipt.get("implementation_provenance"), f"{label}.implementation_provenance")
    digests = _mapping(provenance.get("relevant_file_digests"), f"{label}.implementation_provenance.relevant_file_digests")
    predecessor_digest = digests.get(PREDECESSOR_PRIMITIVES_FILE)
    if not isinstance(predecessor_digest, str) or len(predecessor_digest) != 64:
        _fail(
            f"{label} implementation_provenance does not carry a digest for the reused predecessor "
            f"primitives file {PREDECESSOR_PRIMITIVES_FILE!r}"
        )
    if not isinstance(provenance.get("git_head"), str) or not provenance.get("git_head"):
        _fail(f"{label} implementation_provenance.git_head must be a non-empty string")
    if not isinstance(provenance.get("git_dirty"), bool):
        _fail(f"{label} implementation_provenance.git_dirty must be a boolean")
    if not isinstance(provenance.get("git_dirty_diff_sha256"), str) or not provenance.get("git_dirty_diff_sha256"):
        _fail(f"{label} implementation_provenance.git_dirty_diff_sha256 must be a non-empty string")
    return dict(provenance)


def _validate_runtime_identity_admission(receipt: Mapping[str, Any], *, expected_identity: Mapping[str, str], label: str) -> None:
    """Validate the two-stage runtime identity admission -- two distinct digest domains.

    ``preload`` (``score_sorted_fn_fixed_budget.validate_runtime_identity_
    against_static_contract``) is a pure, pre-model-load JSON/digest
    comparison of ``runtime-identity.json`` against the owner-context
    ledger's own ``vocabulary_attestation`` sha256 domain -- ``expected_
    identity`` here.

    ``postload`` (``run_sorted_fn_successor_behavior.validate_live_runtime_
    identity``) is a separate, later comparison of the *actually opened* HF
    session against the frozen ``runtime-identity.json`` document's own raw
    identity dicts -- a different digest domain entirely (that function
    raises before ever returning unless every check already passed, so a
    genuinely live-sealed receipt can only carry ``status ==
    'passed_before_generation'`` with every check ``True``; this merger still
    validates both explicitly rather than trusting the producer). It never
    carries a ledger-domain digest and must never be compared against
    ``expected_identity``.
    """

    admission = _mapping(receipt.get("runtime_identity_admission"), f"{label}.runtime_identity_admission")
    preload = _mapping(admission.get("preload"), f"{label}.runtime_identity_admission.preload")
    postload = _mapping(admission.get("postload"), f"{label}.runtime_identity_admission.postload")

    if preload.get("status") != "passed":
        _fail(f"{label} runtime_identity_admission.preload.status must be 'passed'")
    if preload.get("model_identity_sha256") != expected_identity["model_identity"]:
        _fail(f"{label} runtime_identity_admission.preload.model_identity_sha256 does not match the ledger's declared identity")
    if preload.get("tokenizer_identity_sha256") != expected_identity["tokenizer_identity"]:
        _fail(f"{label} runtime_identity_admission.preload.tokenizer_identity_sha256 does not match the ledger's declared identity")

    if postload.get("status") != _POSTLOAD_STATUS:
        _fail(f"{label} runtime_identity_admission.postload.status must be {_POSTLOAD_STATUS!r}")
    checks = _mapping(postload.get("checks"), f"{label}.runtime_identity_admission.postload.checks")
    if set(checks) != _POSTLOAD_CHECK_KEYS:
        _fail(f"{label} runtime_identity_admission.postload.checks does not carry exactly the current check-key set")
    failed_checks = sorted(key for key, passed in checks.items() if passed is not True)
    if failed_checks:
        _fail(f"{label} runtime_identity_admission.postload.checks has non-passing check(s): {failed_checks}")

    observed_projection = _sha256_hex(
        postload.get("observed_projection_sha256"), f"{label} runtime_identity_admission.postload.observed_projection_sha256"
    )
    frozen_projection = _sha256_hex(
        postload.get("frozen_projection_sha256"), f"{label} runtime_identity_admission.postload.frozen_projection_sha256"
    )
    if observed_projection != frozen_projection:
        _fail(f"{label} runtime_identity_admission.postload.observed_projection_sha256 does not match its own frozen_projection_sha256")


def _validate_admission_self_consistency(admission: Mapping[str, Any], *, label: str) -> None:
    without_digest = {key: value for key, value in admission.items() if key != "sha256"}
    if sha256_json(without_digest) != admission.get("sha256"):
        _fail(f"{label} scoring_backend_admission.sha256 does not reconstruct; stale or tampered")
    # This scorer never opens a KV-cache path: the decision channel is always
    # raw fp32, literal use_cache=False, full-prefix reforward.
    expected = {
        "parity_status": "failed",
        "cache_admission_policy": None,
        "selected_backend": scorer.FULL_REFORWARD_SCORING_BACKEND,
        "cache_enabled": False,
        "use_cache": False,
        "atol": scorer.CACHE_PARITY_ATOL,
        "rtol": scorer.CACHE_PARITY_RTOL,
        "all_score_rows_backend": scorer.FULL_REFORWARD_SCORING_BACKEND,
        "backend_mixing_detected": False,
        "fallback_trigger": scorer.PARITY_FAILURE_FALLBACK_TRIGGER,
    }
    for key, expected_value in expected.items():
        if admission.get(key) != expected_value:
            _fail(
                f"{label} scoring_backend_admission.{key} is not the successor scorer's frozen raw fp32 "
                f"uncached full-reforward decision channel; observed {admission.get(key)!r}"
            )


def _validate_batched_reforward_admission(admission: Mapping[str, Any], *, label: str) -> None:
    batched = _mapping(admission.get("batched_reforward_admission"), f"{label}.scoring_backend_admission.batched_reforward_admission")
    if batched.get("schema_version") != "batched_full_reforward_parity.v1":
        _fail(f"{label} batched_reforward_admission.schema_version is unrecognized")
    status = batched.get("status")
    if status not in _BATCH_ADMISSION_STATUSES:
        _fail(f"{label} batched_reforward_admission.status is unrecognized: {status!r}")
    requested = batched.get("requested_batch_size")
    effective = batched.get("effective_batch_size")
    if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
        _fail(f"{label} batched_reforward_admission.requested_batch_size must be a positive integer")
    if isinstance(effective, bool) or not isinstance(effective, int) or effective <= 0:
        _fail(f"{label} batched_reforward_admission.effective_batch_size must be a positive integer")
    if status == "not_requested":
        if requested != 1 or effective != 1:
            _fail(f"{label} batched_reforward_admission status not_requested requires requested==effective==1")
    elif status == "passed":
        if effective != requested:
            _fail(f"{label} batched_reforward_admission status passed requires effective_batch_size == requested_batch_size")
    else:  # failed_scalar_fallback_required
        if effective != 1 or requested <= 1:
            _fail(f"{label} batched_reforward_admission failed-fallback requires requested_batch_size > 1 and effective_batch_size == 1")


def _validate_per_context_accounting(admission: Mapping[str, Any], *, observed_context_ids: set[str], label: str) -> None:
    entries = _sequence(admission.get("per_context_accounting"), f"{label}.scoring_backend_admission.per_context_accounting")
    if not entries:
        _fail(f"{label} scoring_backend_admission.per_context_accounting must be non-empty")
    for untyped in entries:
        entry = _mapping(untyped, f"{label} per_context_accounting entry")
        if entry.get("scoring_backend") != scorer.FULL_REFORWARD_SCORING_BACKEND:
            _fail(f"{label} per_context_accounting entry uses a foreign scoring_backend")
        context_id = entry.get("context_id")
        if context_id not in observed_context_ids:
            _fail(f"{label} per_context_accounting references a context this shard did not score: {context_id!r}")


def _identity_projection(receipt: Mapping[str, Any], *, implementation_provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Project conclusion-critical scorer/runtime identity shared by every shard.

    Deliberately excludes ``batched_reforward_admission``/
    ``per_context_accounting`` (legitimately shard-scoped: different shards
    may cover different contexts or request different batch sizes without
    that being a "mixed backend" concern -- the decision channel itself is
    always the same raw fp32 uncached full reforward, checked separately).
    """

    admission = _mapping(receipt.get("scoring_backend_admission"), "receipt.scoring_backend_admission")
    admission_shape = {key: admission.get(key) for key in _ADMISSION_IDENTITY_KEYS}
    return {
        "schema_version": receipt.get("schema_version"),
        "unit_id": receipt.get("unit_id"),
        "model_identity": receipt.get("model_identity"),
        "tokenizer_identity": receipt.get("tokenizer_identity"),
        "backend_session": receipt.get("backend_session"),
        "environment": receipt.get("environment"),
        "likelihood_channels": receipt.get("likelihood_channels"),
        "scoring_backend_admission_shape": admission_shape,
        "predecessor_primitives_file_sha256": implementation_provenance["relevant_file_digests"].get(PREDECESSOR_PRIMITIVES_FILE),
        "implementation_git_head": implementation_provenance.get("git_head"),
        "implementation_git_dirty": implementation_provenance.get("git_dirty"),
        "implementation_git_dirty_diff_sha256": implementation_provenance.get("git_dirty_diff_sha256"),
    }


def _validate_row(
    row: Mapping[str, Any],
    *,
    label: str,
    rules: Any,
    ledger_by_context: Mapping[str, Mapping[str, Any]],
    fixed_budget_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[str, str]:
    _reject_predecessor_provenance(schema_version=row.get("schema_version"), unit_id=row.get("unit_id"), label=label)
    if row.get("schema_version") != SUCCESSOR_SCORE_ROW_SCHEMA_VERSION:
        _fail(f"{label} has a stale or foreign score-row schema_version")
    if row.get("unit_id") != SUCCESSOR_SCORER_UNIT_ID:
        _fail(f"{label} row unit_id is not this successor scorer's own unit_id")
    if "predecessor_candidate_id" in row:
        _fail(f"{label} carries predecessor_candidate_id; every selected row must be a fresh live score, never a reused value")

    context_id = _string(row.get("context_id"), f"{label}.context_id")
    candidate_id = _string(row.get("candidate_id"), f"{label}.candidate_id")
    if row.get("owner_context_id") != context_id:
        _fail(f"{label} owner_context_id does not match its own context_id")
    ledger_row = ledger_by_context.get(context_id)
    if ledger_row is None:
        _fail(f"{label} names an unknown context: {context_id}")
    key = (context_id, candidate_id)
    candidate = fixed_budget_by_key.get(key)
    if candidate is None:
        _fail(f"{label} is not a declared fixed-budget candidate: {key}")

    if row.get("gt_owner_id") != ledger_row.get("gt_owner_id"):
        _fail(f"{label} names the wrong owner for context {context_id!r}")
    if row.get("diagnostic_owner_id") != ledger_row.get("diagnostic_owner_id"):
        _fail(f"{label} names the wrong diagnostic owner for context {context_id!r}")
    if row.get("image_id") != ledger_row.get("image_id"):
        _fail(f"{label} names the wrong image_id for context {context_id!r}")
    # "Current prefix/context digest": the row must be bound to the exact
    # ledger prefix it was scored against, never a stale one.
    if row.get("ledger_context_tokens_sha256") != ledger_row["context_tokens"]["token_ids_sha256"]:
        _fail(f"{label} ledger_context_tokens_sha256 does not match the current ledger's context prefix digest")
    if row.get("native_repetition_penalty_stratum") != NATIVE_REPETITION_PENALTY_STRATUM:
        _fail(f"{label} native_repetition_penalty_stratum must be {NATIVE_REPETITION_PENALTY_STRATUM!r}")
    if row.get("rule_digest") != rules.rules_digest:
        _fail(f"{label} rule_digest does not match the current decision rules")

    # Exact fixed-budget candidate join: every retained identity field must
    # match the declared candidate bit-for-bit, never a substituted one.
    for identity_field in _FIXED_BUDGET_IDENTITY_FIELDS:
        if row.get(identity_field) != candidate.get(identity_field):
            _fail(f"{label}.{identity_field} does not match its declared fixed-budget candidate")
    if [int(v) for v in _sequence(row.get("coord_token_ids"), f"{label}.coord_token_ids")] != candidate["coord_token_ids"]:
        _fail(f"{label}.coord_token_ids does not match its declared fixed-budget candidate")

    population = row.get("population")
    if population not in _VALID_POPULATIONS:
        _fail(f"{label}.population must be one of {sorted(_VALID_POPULATIONS)}")
    if population == "reference":
        if row.get("region") != "other_owner":
            _fail(f"{label} has population=reference but region!=other_owner")
        _string(row.get("other_owner_gt_owner_id"), f"{label}.other_owner_gt_owner_id")
        if not _mapping(row.get("other_owner_selection_trace"), f"{label}.other_owner_selection_trace"):
            _fail(f"{label}.other_owner_selection_trace must be a non-empty object")
    else:
        for field_name in _REFERENCE_ONLY_FIELDS:
            if row.get(field_name) is not None:
                _fail(f"{label} declares {field_name!r} but population={population!r}, not reference")
    member = row.get("candidate_neighborhood_member")
    if not isinstance(member, bool):
        _fail(f"{label}.candidate_neighborhood_member must be a boolean")
    if member and not _string(row.get("candidate_neighborhood_id"), f"{label}.candidate_neighborhood_id"):
        _fail(f"{label} is a candidate-neighborhood member but declares no candidate_neighborhood_id")
    if not member and row.get("candidate_neighborhood_id") is not None:
        _fail(f"{label} declares candidate_neighborhood_id without candidate_neighborhood_member")

    singleton_member = row.get("exact_gt_singleton_member")
    if not isinstance(singleton_member, bool):
        _fail(f"{label}.exact_gt_singleton_member must be a boolean")
    singleton_id = row.get("exact_gt_singleton_id")
    if singleton_member:
        if not _string(singleton_id, f"{label}.exact_gt_singleton_id").startswith(_EXACT_GT_SINGLETON_ID_PREFIX):
            _fail(f"{label}.exact_gt_singleton_id has an unrecognized form")
    elif singleton_id is not None:
        _fail(f"{label} declares exact_gt_singleton_id without exact_gt_singleton_member")

    raw = _mapping(row.get("raw_model_logprob"), f"{label}.raw_model_logprob")
    slot_values = {slot: _finite_number(raw.get(slot), f"{label}.raw_model_logprob.{slot}") for slot in _RAW_LOGPROB_SLOTS}
    complete = _finite_number(raw.get("complete_box_logprob_sum"), f"{label}.raw_model_logprob.complete_box_logprob_sum")
    if abs(complete - sum(slot_values.values())) > _NUMERIC_ROUND_TRIP_TOLERANCE:
        _fail(f"{label}.raw_model_logprob.complete_box_logprob_sum is not the sum of its own four coordinate slots")

    policy = _mapping(row.get("auxiliary_policy_scores"), f"{label}.auxiliary_policy_scores")
    if set(policy) != set(_AUXILIARY_POLICY_VIEWS):
        _fail(f"{label}.auxiliary_policy_scores must carry exactly the rp=1.0 primary and rp=1.10 auxiliary views")
    for view_key, view in policy.items():
        view_mapping = _mapping(view, f"{label}.auxiliary_policy_scores.{view_key}")
        view_slots = {slot: _finite_number(view_mapping.get(slot), f"{label}.auxiliary_policy_scores.{view_key}.{slot}") for slot in _RAW_LOGPROB_SLOTS}
        view_complete = _finite_number(view_mapping.get("complete_box_logprob_sum"), f"{label}.auxiliary_policy_scores.{view_key}.complete_box_logprob_sum")
        if abs(view_complete - sum(view_slots.values())) > _NUMERIC_ROUND_TRIP_TOLERANCE:
            _fail(f"{label}.auxiliary_policy_scores.{view_key}.complete_box_logprob_sum is not the sum of its own four coordinate slots")

    return context_id, candidate_id


def _validate_shard_selection(
    receipt: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
    observed_context_ids: set[str],
    label: str,
) -> frozenset[str]:
    """Bind and field-validate the scorer's fail-closed ``--include-rung`` selection.

    ``score_sorted_fn_fixed_budget.py`` never scores an implicit all-rungs
    domain: every sealed shard receipt freezes exactly which rungs and
    contexts were selected, self-verified via ``selection.selection_sha256``
    and bound to the JSONL via
    ``output_artifacts.fixed_budget_scores.selection_sha256``. This merger
    never trusts that declaration blindly: it re-derives the same selection
    from the shard's own rows and requires an exact match, so a shard cannot
    silently score (or omit) a different rung than its receipt claims.
    """

    selection = _mapping(receipt.get("selection"), f"{label}.selection")
    content = {key: value for key, value in selection.items() if key != "selection_sha256"}
    if sha256_json(content) != selection.get("selection_sha256"):
        _fail(f"{label}.selection.selection_sha256 does not reconstruct from its own content; stale or tampered")

    raw_rungs = _sequence(selection.get("selected_rungs"), f"{label}.selection.selected_rungs")
    selected_rungs = [_string(value, f"{label}.selection.selected_rungs[]") for value in raw_rungs]
    if not selected_rungs:
        _fail(f"{label}.selection.selected_rungs must be non-empty; implicit all-rungs scoring is not accepted")
    if len(selected_rungs) != len(set(selected_rungs)):
        _fail(f"{label}.selection.selected_rungs contains duplicates")
    invalid_rungs = sorted(set(selected_rungs) - set(_VALID_RUNGS))
    if invalid_rungs:
        _fail(f"{label}.selection.selected_rungs names invalid rung(s): {invalid_rungs}")

    observed_rungs = {str(row["rung"]) for row in rows}
    if set(selected_rungs) != observed_rungs:
        _fail(
            f"{label}.selection.selected_rungs {sorted(selected_rungs)} does not exactly equal the rungs "
            f"actually observed in this shard's own rows {sorted(observed_rungs)}"
        )

    raw_selected_contexts = _sequence(selection.get("selected_context_ids"), f"{label}.selection.selected_context_ids")
    if {str(value) for value in raw_selected_contexts} != observed_context_ids:
        _fail(
            f"{label}.selection.selected_context_ids does not exactly equal the contexts actually observed "
            "in this shard's own rows"
        )

    observed_keyset = sorted([[str(row["context_id"]), str(row["candidate_id"])] for row in rows])
    if selection.get("candidate_keyset_sha256") != sha256_json(observed_keyset):
        _fail(f"{label}.selection.candidate_keyset_sha256 does not match this shard's own observed candidate keys")
    if selection.get("candidate_count") != len(rows):
        _fail(f"{label}.selection.candidate_count does not match this shard's own observed row count")

    output_artifacts = _mapping(receipt.get("output_artifacts"), f"{label}.output_artifacts")
    scores_entry = _mapping(output_artifacts.get("fixed_budget_scores"), f"{label}.output_artifacts.fixed_budget_scores")
    if scores_entry.get("selection_sha256") != selection.get("selection_sha256"):
        _fail(f"{label}.output_artifacts.fixed_budget_scores.selection_sha256 does not match this shard's own selection")

    return frozenset(selected_rungs)


def _validate_shard(
    shard: ShardInput,
    *,
    rules: Any,
    rules_path: Path,
    expected_identity: Mapping[str, str],
    ledger_by_context: Mapping[str, Mapping[str, Any]],
    fixed_budget_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    source_paths: Mapping[str, Path],
) -> ValidatedShard:
    scores_path = shard.scores_path.expanduser().resolve(strict=True)
    receipt_path = shard.receipt_path.expanduser().resolve(strict=True)
    receipt = _read_json(receipt_path)
    rows = _read_jsonl(scores_path)
    label_prefix = f"shard {receipt_path}"

    _reject_predecessor_provenance(schema_version=receipt.get("schema_version"), unit_id=receipt.get("unit_id"), label=label_prefix)
    if receipt.get("schema_version") != SUCCESSOR_SCORE_RECEIPT_SCHEMA_VERSION:
        _fail(f"{label_prefix} has a stale or foreign scorer receipt schema_version")
    if receipt.get("unit_id") != SUCCESSOR_SCORER_UNIT_ID:
        _fail(f"{label_prefix} unit_id does not match the successor scorer's own module constant")
    if receipt.get("runtime_execution_status") != SUCCESSOR_LIVE_SCORING_SEALED_STATUS:
        _fail(f"{label_prefix} is not a sealed live successor-scorer output")

    decision_rules = _mapping(receipt.get("decision_rules"), f"{label_prefix}.decision_rules")
    if decision_rules.get("file_sha256") != sha256_file(rules_path):
        _fail(f"{label_prefix} was scored against a stale landscape-decision-rules.json")
    if decision_rules.get("core_rule_digest") != rules.rules_digest:
        _fail(f"{label_prefix} core_rule_digest does not match the current decision rules")

    source_digests = _mapping(receipt.get("source_digests"), f"{label_prefix}.source_digests")
    missing_keys = [key for key in _REQUIRED_SOURCE_DIGEST_KEYS if key not in source_digests]
    if missing_keys:
        _fail(f"{label_prefix}.source_digests is missing required keys: {missing_keys}")
    for key, path in source_paths.items():
        entry = _mapping(source_digests.get(key), f"{label_prefix}.source_digests.{key}")
        if entry.get("sha256") != sha256_file(path):
            _fail(f"{label_prefix}.source_digests.{key} does not match the exact planner receipt/registry/ledger/fixed-budget input on disk")
    runtime_identity_entry = _mapping(source_digests.get("runtime_identity"), f"{label_prefix}.source_digests.runtime_identity")
    _string(runtime_identity_entry.get("sha256"), f"{label_prefix}.source_digests.runtime_identity.sha256")

    _validate_runtime_identity_admission(receipt, expected_identity=expected_identity, label=label_prefix)

    # Top-level model_identity/tokenizer_identity are the executed identities
    # (score_sorted_fn_fixed_budget.py sets them straight from the opened
    # backend_session receipt); bind them to that same backend_session
    # record rather than to any ledger/preload sha256 digest -- those are a
    # distinct domain, already checked above via preload alone.
    model_identity = _mapping(receipt.get("model_identity"), f"{label_prefix}.model_identity")
    tokenizer_identity = _mapping(receipt.get("tokenizer_identity"), f"{label_prefix}.tokenizer_identity")
    backend_session = _mapping(receipt.get("backend_session"), f"{label_prefix}.backend_session")
    if backend_session.get("model_identity") != dict(model_identity):
        _fail(f"{label_prefix}.backend_session.model_identity does not match the receipt's top-level model_identity")
    if backend_session.get("tokenizer_identity") != dict(tokenizer_identity):
        _fail(f"{label_prefix}.backend_session.tokenizer_identity does not match the receipt's top-level tokenizer_identity")

    channels = _mapping(receipt.get("likelihood_channels"), f"{label_prefix}.likelihood_channels")
    raw_channel = channels.get("raw")
    if not isinstance(raw_channel, str) or "fp32" not in raw_channel:
        _fail(f"{label_prefix}.likelihood_channels.raw does not declare an fp32 raw channel")
    if set(_sequence(channels.get("auxiliary_policy"), f"{label_prefix}.likelihood_channels.auxiliary_policy")) != set(_AUXILIARY_POLICY_VIEWS):
        _fail(f"{label_prefix}.likelihood_channels.auxiliary_policy must declare exactly the rp=1.0/rp=1.10 views")
    if channels.get("policy_is_not_a_model_likelihood") is not True or channels.get("subset_normalization_forbidden") is not True:
        _fail(f"{label_prefix}.likelihood_channels does not attest the auxiliary-policy/raw separation")

    admission = _mapping(receipt.get("scoring_backend_admission"), f"{label_prefix}.scoring_backend_admission")
    _validate_admission_self_consistency(admission, label=label_prefix)
    _validate_batched_reforward_admission(admission, label=label_prefix)

    implementation_provenance = _validate_implementation_provenance(receipt, label=label_prefix)

    if not rows:
        _fail(f"{label_prefix} has an empty score JSONL")
    observed_context_ids: set[str] = set()
    for index, row in enumerate(rows):
        context_id, _candidate_id = _validate_row(
            row,
            label=f"{label_prefix} row[{index}]",
            rules=rules,
            ledger_by_context=ledger_by_context,
            fixed_budget_by_key=fixed_budget_by_key,
        )
        observed_context_ids.add(context_id)
    _validate_per_context_accounting(admission, observed_context_ids=observed_context_ids, label=label_prefix)
    selected_rungs = _validate_shard_selection(
        receipt, rows=rows, observed_context_ids=observed_context_ids, label=label_prefix
    )

    return ValidatedShard(
        scores_path=scores_path,
        receipt_path=receipt_path,
        receipt=receipt,
        rows=rows,
        identity_projection=_identity_projection(receipt, implementation_provenance=implementation_provenance),
        selected_rungs=selected_rungs,
    )


def _reconcile_shards(
    *,
    validated: Sequence[ValidatedShard],
    fixed_budget_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Per-shard context disjointness, or an exact byte-identical dedup policy.

    Shards covering disjoint context sets are simply unioned. Two shards
    sharing a context must agree on every one of that context's rows
    byte-for-byte (canonical JSON); the merge then keeps exactly one copy.
    Any partial overlap or disagreement is a hard failure.
    """

    shard_context_ids = [sorted({str(row["context_id"]) for row in shard.rows}) for shard in validated]
    row_by_shard_and_key = [
        {(str(row["context_id"]), str(row["candidate_id"])): row for row in shard.rows} for shard in validated
    ]

    observed: dict[tuple[str, str], dict[str, Any]] = {}
    for shard_index, shard in enumerate(validated):
        for key, row in row_by_shard_and_key[shard_index].items():
            existing = observed.get(key)
            if existing is None:
                observed[key] = row
                continue
            if canonical_json(existing) != canonical_json(row):
                _fail(
                    f"shards disagree on identical candidate {key}: overlapping shards must be "
                    "context-disjoint or byte-identical for every shared candidate"
                )
    # Overlapping shards (any two shards that both cover the same context) must
    # declare an identical rung selection. Without this, two individually
    # rung-pure shards -- each correctly reporting selected_rungs == its own
    # observed row rungs -- could jointly deposit a silently mixed rung
    # composition into one shared context that neither shard's own receipt
    # ever declared.
    context_rung_domain: dict[str, frozenset[str]] = {}
    for shard_index, shard in enumerate(validated):
        for context_id in shard_context_ids[shard_index]:
            existing_rungs = context_rung_domain.get(context_id)
            if existing_rungs is None:
                context_rung_domain[context_id] = shard.selected_rungs
            elif existing_rungs != shard.selected_rungs:
                _fail(
                    f"shards covering context {context_id!r} disagree on their selected rungs: "
                    f"{sorted(existing_rungs)} vs {sorted(shard.selected_rungs)}; a silent mixed-rung "
                    "merge for a shared context is refused"
                )

    for a_index in range(len(validated)):
        for b_index in range(a_index + 1, len(validated)):
            shared_contexts = set(shard_context_ids[a_index]) & set(shard_context_ids[b_index])
            for context_id in shared_contexts:
                a_keys = {key for key in row_by_shard_and_key[a_index] if key[0] == context_id}
                b_keys = {key for key in row_by_shard_and_key[b_index] if key[0] == context_id}
                if a_keys != b_keys:
                    _fail(
                        f"shards {validated[a_index].receipt_path} and {validated[b_index].receipt_path} "
                        f"partially overlap on context {context_id!r}; a shared context must be fully "
                        "duplicated (exact dedup policy), never partially covered by two shards"
                    )

    # Selection-aware completeness: a shard is only required to be complete
    # against the exact (context, rung) domain its own receipt selected, not
    # against every rung the fixed-budget lattice ever declared for that
    # context -- a shard that intentionally scores only one rung must not be
    # rejected against the full multi-rung candidate count for its contexts.
    observed_contexts = sorted({key[0] for key in observed})
    expected_keys: set[tuple[str, str]] = set()
    for context_id in observed_contexts:
        rung_domain = context_rung_domain[context_id]
        context_expected = {
            key for key in fixed_budget_by_key if key[0] == context_id and fixed_budget_by_key[key]["rung"] in rung_domain
        }
        if not context_expected:
            _fail(f"context {context_id!r} has no declared fixed-budget candidates for its selected rung(s) {sorted(rung_domain)}")
        expected_keys |= context_expected

    observed_keys = set(observed)
    missing = sorted(expected_keys - observed_keys)
    extra = sorted(observed_keys - expected_keys)
    if missing or extra:
        _fail(
            "merged score rows do not exactly join the declared fixed-budget candidate set for the "
            f"selected contexts; missing={missing[:8]} extra={extra[:8]}"
        )

    rows = [observed[key] for key in sorted(observed)]
    return rows, observed_contexts


def _validate_execution_dedup(
    *,
    rows: Sequence[Mapping[str, Any]],
    ledger_by_context: Mapping[str, Mapping[str, Any]],
    numeric_tolerance: float,
) -> dict[str, int]:
    """Rows scored for two roles sharing one exact prefix must agree exactly."""

    groups: dict[tuple[str, tuple[int, ...]], list[tuple[str, float]]] = {}
    for row in rows:
        context_id = str(row["context_id"])
        dedup_key = ledger_by_context[context_id]["execution_dedup_key"]
        coord_token_ids = tuple(int(v) for v in row["coord_token_ids"])
        value = float(row["raw_model_logprob"]["complete_box_logprob_sum"])
        groups.setdefault((dedup_key, coord_token_ids), []).append((context_id, value))

    groups_with_multiple_roles = 0
    for (dedup_key, _coord), members in groups.items():
        distinct_contexts = {context_id for context_id, _value in members}
        if len(distinct_contexts) < 2:
            continue
        groups_with_multiple_roles += 1
        values = [value for _context_id, value in members]
        spread = max(values) - min(values)
        if spread > numeric_tolerance:
            _fail(
                "execution-deduplicated rows disagree beyond the frozen numeric tolerance for one "
                f"identical (prefix digest, coordinate token sequence) pair: {sorted(distinct_contexts)}"
            )
    return {"groups_checked": len(groups), "groups_with_multiple_roles": groups_with_multiple_roles}


def merge_shards(
    *,
    shards: Sequence[ShardInput],
    owner_context_ledger: Path,
    decision_rules: Path,
    mechanism_decision_rules: Path,
    fixed_budget_candidates: Path,
    planner_receipt: Path,
    fn_mechanism_registry: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if not shards:
        _fail("at least one score/receipt shard pair is required")

    ledger_path = owner_context_ledger.expanduser().resolve(strict=True)
    rules_path = decision_rules.expanduser().resolve(strict=True)
    mechanism_rules_path = mechanism_decision_rules.expanduser().resolve(strict=True)
    fixed_budget_path = fixed_budget_candidates.expanduser().resolve(strict=True)
    planner_receipt_path = planner_receipt.expanduser().resolve(strict=True)
    registry_path = fn_mechanism_registry.expanduser().resolve(strict=True)

    planner = _load_planner_receipt(planner_receipt_path)
    registry_document = _load_registry(registry_path, planner_receipt=planner)
    ledger_by_context = _load_ledger(ledger_path, planner_receipt=planner)
    rules = scorer.load_decision_rules(rules_path)
    mechanism_rules = _load_mechanism_decision_rules(
        mechanism_rules_path, planner_receipt=planner, landscape_rules_path=rules_path, registry_document=registry_document
    )
    fixed_budget_by_key = _load_fixed_budget_candidates(
        fixed_budget_path, planner_receipt=planner, ledger_by_context=ledger_by_context
    )
    _validate_fixed_budget_mechanism_binding(
        fixed_budget_by_key, mechanism_decision_rules_sha256=sha256_file(mechanism_rules_path)
    )
    _validate_reference_selection_bindings(ledger_by_context=ledger_by_context, fixed_budget_by_key=fixed_budget_by_key)
    expected_identity = _expected_runtime_identity(ledger_by_context)
    source_paths = {
        "registry": registry_path,
        "owner_context_ledger": ledger_path,
        "decision_rules": rules_path,
        "mechanism_decision_rules": mechanism_rules_path,
        "fixed_budget_candidates": fixed_budget_path,
        "planner_receipt": planner_receipt_path,
    }

    validated = [
        _validate_shard(
            shard,
            rules=rules,
            rules_path=rules_path,
            expected_identity=expected_identity,
            ledger_by_context=ledger_by_context,
            fixed_budget_by_key=fixed_budget_by_key,
            source_paths=source_paths,
        )
        for shard in shards
    ]

    identity_digests = {sha256_json(shard.identity_projection) for shard in validated}
    if len(identity_digests) != 1:
        _fail("shards differ in conclusion-critical scorer/model/tokenizer/backend/implementation identity")

    predecessor_primitives_sha256 = validated[0].identity_projection["predecessor_primitives_file_sha256"]

    merged_selected_rungs = sorted({rung for shard in validated for rung in shard.selected_rungs})

    rows, selected_context_ids = _reconcile_shards(validated=validated, fixed_budget_by_key=fixed_budget_by_key)
    dedup_accounting = _validate_execution_dedup(
        rows=rows,
        ledger_by_context=ledger_by_context,
        numeric_tolerance=rules.numeric_tolerance,
    )

    output_dir = output_dir.expanduser().resolve()
    scores_path = output_dir / MERGED_SCORE_NAME
    receipt_path = output_dir / MERGED_RECEIPT_NAME

    scores_bytes = b"".join(canonical_json(row).encode("utf-8") + b"\n" for row in rows)
    _write_create_or_identical(scores_path, scores_bytes)

    shard_lineage = [
        {
            "scores": {"path": str(shard.scores_path), "sha256": sha256_file(shard.scores_path), "row_count": len(shard.rows)},
            "receipt": {"path": str(shard.receipt_path), "sha256": sha256_file(shard.receipt_path)},
            "context_ids_covered": sorted({str(row["context_id"]) for row in shard.rows}),
        }
        for shard in validated
    ]

    receipt_content = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "generic_arbitrary_role_merger": True,
        "decision_channel": {
            "name": "raw_model_logprob.complete_box_logprob_sum",
            "description": (
                "raw, unmodified fp32 lm-head log-softmax at each of the four selected coordinate tokens "
                "under a literal use_cache=False full-prefix reforward, summed; this is the sole "
                "decision-bearing channel"
            ),
            "primary_repetition_penalty_stratum": NATIVE_REPETITION_PENALTY_STRATUM,
            "auxiliary_policy_views": sorted(_AUXILIARY_POLICY_VIEWS),
            "auxiliary_policy_is_not_a_model_likelihood": True,
        },
        "successor_scorer_provenance": {
            "row_schema_version": SUCCESSOR_SCORE_ROW_SCHEMA_VERSION,
            "receipt_schema_version": SUCCESSOR_SCORE_RECEIPT_SCHEMA_VERSION,
            "unit_id": SUCCESSOR_SCORER_UNIT_ID,
            "note": "every merged shard was produced by score_sorted_fn_fixed_budget.py; predecessor scorer artifacts are refused",
        },
        "imported_predecessor_primitive_provenance": {
            "predecessor_primitives_file": PREDECESSOR_PRIMITIVES_FILE,
            "predecessor_primitives_file_sha256": predecessor_primitives_sha256,
            "note": (
                "exact code/helper digest of the low-level scoring primitives this successor scorer "
                "imports unmodified from the predecessor module; a code-reuse record only, never a "
                "claim about which unit owns this artifact or which scorer produced these rows"
            ),
        },
        "source_digests": {
            "owner_context_ledger": {"path": str(ledger_path), "sha256": sha256_file(ledger_path)},
            "decision_rules": {"path": str(rules_path), "sha256": sha256_file(rules_path), "core_rule_digest": rules.rules_digest},
            "mechanism_decision_rules": {
                "path": str(mechanism_rules_path),
                "sha256": sha256_file(mechanism_rules_path),
                "self_digest": mechanism_rules["self_digest"],
                "parent_execution_rules_sha256": mechanism_rules["upstream_digests"]["execution_landscape_decision_rules_sha256"],
            },
            "fixed_budget_candidates": {"path": str(fixed_budget_path), "sha256": sha256_file(fixed_budget_path)},
            "fn_mechanism_registry": {
                "path": str(registry_path),
                "sha256": sha256_file(registry_path),
                "registry_digest": registry_document.get("registry_digest"),
            },
        },
        "planner_receipt": {
            "path": str(planner_receipt_path),
            "sha256": sha256_file(planner_receipt_path),
            "receipt_digest": planner.get("receipt_digest"),
        },
        "expected_runtime_identity": {
            **expected_identity,
            "derived_from": "owner_context_ledger.jsonl vocabulary_attestation (single-checkpoint invariant)",
        },
        "native_repetition_penalty_stratum": NATIVE_REPETITION_PENALTY_STRATUM,
        "context_selection": {
            "selected_context_ids": selected_context_ids,
            "candidate_count": len(rows),
        },
        "selected_rungs": merged_selected_rungs,
        "execution_dedup": {**dedup_accounting, "status": "consistent"},
        "identity_projection_sha256": next(iter(identity_digests)),
        "shards": shard_lineage,
        "candidate_count": len(rows),
        "owners_covered": len({str(row["gt_owner_id"]) for row in rows}),
    }
    receipt_content["output_artifacts"] = {
        "merged_scores": {"path": str(scores_path), "sha256": sha256_file(scores_path), "row_count": len(rows)}
    }
    receipt_bytes = canonical_json(receipt_content).encode("utf-8") + b"\n"
    _write_create_or_identical(receipt_path, receipt_bytes)
    return receipt_content


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard",
        action="append",
        nargs=2,
        metavar=("SCORES", "RECEIPT"),
        default=[],
        help="repeatable score JSONL and its sealed successor-scorer receipt; at least one is required",
    )
    parser.add_argument(
        "--shard-dir",
        action="append",
        type=Path,
        default=[],
        help=f"alternative repeated shard directory containing {SUCCESSOR_OUTPUT_JSONL_NAME!r}/{SUCCESSOR_RECEIPT_NAME!r}",
    )
    parser.add_argument("--owner-context-ledger", type=Path, required=True)
    parser.add_argument("--decision-rules", type=Path, required=True)
    parser.add_argument("--mechanism-decision-rules", type=Path, required=True)
    parser.add_argument("--fixed-budget-candidates", type=Path, required=True)
    parser.add_argument("--planner-receipt", type=Path, required=True)
    parser.add_argument("--fn-mechanism-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shard_inputs = [ShardInput(Path(scores), Path(receipt)) for scores, receipt in args.shard]
    shard_inputs.extend(
        ShardInput(directory / SUCCESSOR_OUTPUT_JSONL_NAME, directory / SUCCESSOR_RECEIPT_NAME)
        for directory in args.shard_dir
    )
    receipt = merge_shards(
        shards=shard_inputs,
        owner_context_ledger=args.owner_context_ledger,
        decision_rules=args.decision_rules,
        mechanism_decision_rules=args.mechanism_decision_rules,
        fixed_budget_candidates=args.fixed_budget_candidates,
        planner_receipt=args.planner_receipt,
        fn_mechanism_registry=args.fn_mechanism_registry,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "rows": receipt["candidate_count"],
                "owners": receipt["owners_covered"],
                "contexts": len(receipt["context_selection"]["selected_context_ids"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
