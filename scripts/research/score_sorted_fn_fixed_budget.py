#!/usr/bin/env python3
"""Successor-local restricted-complete-box scorer for the fixed-budget lattice.

Experiment-local to ``research/investigations/qwen3-vl-dense-enumeration/
experiments/2026-08-02-sorted-false-negative-mechanism-decomposition/unit.md``.

Closes the live-GPU seam between ``prepare_sorted_fn_successor_inputs.py``'s
``fixed-budget-candidates.jsonl`` (a score-independent, score-free ``L0``/
``L1``/``scalar_smoke`` box lattice) and the successor's own arbitrary-role
merger (``merge_sorted_fn_successor_score_shards.py``).

The unchanged predecessor production scorer
(``score_sorted_owner_basin_landscape.py``) requires the closed
candidate-builder-v2 contract: a complete conditional-``x1`` domain plus at
least one declared free-coordinate-tree root per run. The fixed-budget
lattice is neither -- it is a finite, score-independent set of literal
complete boxes with no free-tree root -- so it cannot be fed into that
production contract, and that contract is never relaxed or edited to accept
it (see unit.md, "Execution outline" item 2 and "Alignment and
preservation"). This module is therefore a distinct, successor-owned scorer
for exactly this lattice; it never claims the predecessor's unit ID or
schema versions.

Reuse discipline
-----------------
This module never reconstructs the multimodal materialization, the raw
per-coordinate log-softmax math, or the literal ``use_cache=False``
full-reforward branching engine: all of that is imported, unmodified, from
``score_sorted_owner_basin_landscape.py`` (the same native-materialization
seam via ``HFBackendSession._materialize_native_inputs``, the same
:class:`~scripts.research.score_sorted_owner_basin_landscape.FullReforwardBackend`,
:func:`~scripts.research.score_sorted_owner_basin_landscape.score_complete_box_candidate`,
:func:`~scripts.research.score_sorted_owner_basin_landscape.combine_complete_box`, and
the same :func:`~scripts.research.score_sorted_owner_basin_landscape.run_batched_reforward_parity_gate`).
Unlike the predecessor, this scorer never opens a KV-cache path at all --
every score row is a literal, uncached, full-prefix reforward -- so none of
the predecessor's cache-branch/parity-vs-cache machinery is imported.

Static-contract loading (planner receipt binding, FN registry/ledger
cross-checks, fixed-budget candidate joins) reuses the already-implemented,
already-tested private loaders in ``merge_sorted_fn_successor_score_shards.py``
rather than re-deriving the same digests and joins a second way.

Decision channel
-----------------
For each fixed-budget candidate, the decision-bearing channel is the raw,
unmodified fp32 lm-head log-softmax at each of the four selected coordinate
tokens (``x1``, ``y1``, ``x2``, ``y2``) under a literal ``use_cache=False``
full-prefix reforward, summed into ``raw_model_logprob.complete_box_logprob_sum``.
Repetition-penalty ``1.0``/``1.10`` policy views are derived from the exact
same raw logits and recorded under ``auxiliary_policy_scores``; they are
never a model likelihood and are never mixed into the raw channel.

No GPU launch happens as a side effect of importing or unit-testing this
module (the heavy ``torch``/``transformers``/``src.inference.*`` imports used
only by the live path are deferred into :func:`run`). ``--validate-contract-only``
performs every CPU-only join/digest/identity check below without loading a
model.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import score_sorted_owner_basin_landscape as scorer  # noqa: E402
from scripts.research import merge_sorted_fn_successor_score_shards as merger  # noqa: E402
from scripts.research import prepare_sorted_fn_successor_inputs as planner_mod  # noqa: E402
from scripts.research import run_sorted_fn_successor_behavior as behavior_runner  # noqa: E402
from scripts.research.build_sorted_fn_mechanism_registry import (  # noqa: E402
    UNIT_ID as REGISTRY_UNIT_ID,
)
from scripts.research.sorted_owner_basin_landscape import (  # noqa: E402
    semantic_core_payload as _validate_semantic_core_payload,
)

# Thin, well-tested reuse from the merger: file IO, canonical digesting, and
# the exact static-contract loaders it already uses to bind the same five
# upstream files (planner receipt, FN registry, ledger, fixed-budget
# candidates). Reused verbatim rather than re-derived a second way.
_mapping = merger._mapping
_sequence = merger._sequence
_string = merger._string
_read_json = merger._read_json
_read_jsonl = merger._read_jsonl
_write_create_or_identical = merger._write_create_or_identical
canonical_json = merger.canonical_json
sha256_json = merger.sha256_json
sha256_file = merger.sha256_file
_load_planner_receipt = merger._load_planner_receipt
_bind_planner_output = merger._bind_planner_output
_bind_planner_source = merger._bind_planner_source
_load_registry = merger._load_registry
_load_ledger = merger._load_ledger
_load_mechanism_decision_rules = merger._load_mechanism_decision_rules
_validate_fixed_budget_mechanism_binding = (
    merger._validate_fixed_budget_mechanism_binding
)
_validate_reference_selection_bindings = merger._validate_reference_selection_bindings
_load_fixed_budget_candidates = merger._load_fixed_budget_candidates
_expected_runtime_identity = merger._expected_runtime_identity
NATIVE_REPETITION_PENALTY_STRATUM = merger.NATIVE_REPETITION_PENALTY_STRATUM

# ---------------------------------------------------------------------------
# Schema / contract constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "sorted_fn_fixed_budget_scores.v1"
RECEIPT_SCHEMA_VERSION = "sorted_fn_fixed_budget_scores_receipt.v1"
#: This successor's own unit; never the predecessor scorer's unit ID.
UNIT_ID = REGISTRY_UNIT_ID
if UNIT_ID != "2026-08-02-sorted-false-negative-mechanism-decomposition":
    raise RuntimeError(
        "unexpected FN mechanism registry unit_id; refusing to impersonate a stale unit"
    )

OUTPUT_JSONL_NAME = "fn-fixed-budget-scores.jsonl"
RECEIPT_NAME = "fn-fixed-budget-scores-receipt.json"

CONTRACT_VALIDATED_STATUS = "contract_validated_no_model_loaded"
LIVE_SCORING_PENDING_SEAL_STATUS = (
    "successor_fixed_budget_live_scoring_completed_artifacts_pending_seal"
)
LIVE_SCORING_SEALED_STATUS = (
    "successor_fixed_budget_live_scoring_completed_artifacts_sealed"
)

_RUNTIME_IDENTITY_SCHEMA_VERSION = "sorted-owner-basin-runtime-identity.v1"

_LIKELIHOOD_CHANNEL_NOTE = (
    "raw_model_logprob is the unmodified fp32 lm-head channel; auxiliary_policy_scores are "
    "repetition-penalty-adjusted (1.0/1.10) policy readouts derived from that same raw forward, "
    "never a model likelihood. This scorer never opens a KV-cache path: every row is a literal, "
    "uncached, use_cache=False full-prefix reforward."
)


class FixedBudgetScoringError(RuntimeError):
    """Raised before scoring/output when a precondition for this scorer fails."""


def _fail(message: str, **context: Any) -> NoReturn:
    if context:
        message = f"{message}: {json.dumps(context, ensure_ascii=False, sort_keys=True, default=str)}"
    raise FixedBudgetScoringError(message)


def _resolved_file(path: str | Path, label: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        _fail(f"{label} does not exist", path=str(path))
        raise exc  # unreachable, appeases type checkers
    if not resolved.is_file():
        _fail(f"{label} must be a regular file", path=str(resolved))
    return resolved


def _finite(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _fail(
            f"{label} is non-finite; refusing to emit a non-finite score",
            observed=value,
        )
    return float(value)


# ---------------------------------------------------------------------------
# Runtime identity (static, CPU-only cross-check against ledger + rules)
# ---------------------------------------------------------------------------


def validate_runtime_identity_against_static_contract(
    identity: Mapping[str, Any],
    *,
    rules: "scorer.DecisionRules",
    expected_identity: Mapping[str, str],
) -> dict[str, Any]:
    """Cross-check ``runtime-identity.json`` against the ledger's declared identity and the rules.

    Every check here is a pure JSON/digest comparison; nothing opens a model
    or touches the filesystem beyond the already-loaded documents, so this is
    safe to run under ``--validate-contract-only``.
    """

    tokenizer = scorer._identity_mapping(
        identity.get("tokenizer"), "runtime identity.tokenizer"
    )
    model = scorer._identity_mapping(identity.get("model"), "runtime identity.model")
    tokenizer_digest = scorer._identity_sha256(
        tokenizer.get("identity_sha256"), "runtime identity.tokenizer.identity_sha256"
    )
    model_digest = scorer._identity_sha256(
        model.get("identity_sha256"), "runtime identity.model.identity_sha256"
    )
    if tokenizer_digest != expected_identity["tokenizer_identity"]:
        _fail(
            "runtime-identity.json tokenizer identity does not match the owner-context ledger's declared identity",
            runtime_identity=tokenizer_digest,
            ledger_identity=expected_identity["tokenizer_identity"],
        )
    if model_digest != expected_identity["model_identity"]:
        _fail(
            "runtime-identity.json model identity does not match the owner-context ledger's declared identity",
            runtime_identity=model_digest,
            ledger_identity=expected_identity["model_identity"],
        )
    coordinate = scorer._identity_mapping(
        identity.get("coordinate_vocabulary"), "runtime identity.coordinate_vocabulary"
    )
    if (
        identity.get("model_vocab_size") != rules.model_vocab_size
        or coordinate.get("token_id_start")
        != rules.schema_tokens["coordinate_token_id_start"]
        or coordinate.get("token_id_end_exclusive")
        != rules.schema_tokens["coordinate_token_id_end_exclusive"]
    ):
        _fail(
            "runtime-identity.json coordinate/model vocabulary is incompatible with landscape-decision-rules.json",
            identity_model_vocab_size=identity.get("model_vocab_size"),
            rules_model_vocab_size=rules.model_vocab_size,
        )
    return {
        "status": "passed",
        "tokenizer_identity_sha256": tokenizer_digest,
        "model_identity_sha256": model_digest,
    }


# ---------------------------------------------------------------------------
# Static contract: registry + ledger + rules + fixed-budget candidates
# ---------------------------------------------------------------------------


@dataclass
class ValidatedContract:
    planner_receipt: dict[str, Any]
    registry: dict[str, Any]
    ledger_by_context: dict[str, dict[str, Any]]
    rules: Any
    rules_path: Path
    mechanism_decision_rules: dict[str, Any]
    mechanism_decision_rules_path: Path
    fixed_budget_by_key: dict[tuple[str, str], dict[str, Any]]
    expected_identity: dict[str, str]
    runtime_identity_doc: dict[str, Any]
    runtime_identity_receipt: dict[str, Any]
    runtime_identity_admission: dict[str, Any]
    live_config_admission: dict[str, Any]
    context_role_index: dict[str, dict[str, Any]]
    source_paths: dict[str, Path] = field(default_factory=dict)


def _context_role_index(
    *, registry: Mapping[str, Any], ledger_by_context: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Reject unknown/stale contexts: every ledger context must bind a live registry role.

    Reuses the registry's own role-iteration helper
    (``prepare_sorted_fn_successor_inputs._iter_all_roles``) rather than
    re-walking ``smoke.roles``/``null_pair_envelope.pairs`` a second way.
    """

    role_by_id = {
        str(role["role_id"]): role for role in planner_mod._iter_all_roles(registry)
    }
    index: dict[str, dict[str, Any]] = {}
    for context_id, ledger_row in ledger_by_context.items():
        provenance = _mapping(
            ledger_row.get("context_provenance"),
            f"ledger row {context_id}.context_provenance",
        )
        registry_id = _string(
            provenance.get("registry_id"),
            f"ledger row {context_id}.context_provenance.registry_id",
        )
        role = role_by_id.get(registry_id)
        if role is None:
            _fail(
                f"context {context_id!r} references registry role {registry_id!r}, which is absent from the "
                "supplied FN mechanism registry; unknown or stale context"
            )
        if str(role.get("gt_owner_id")) != str(ledger_row.get("gt_owner_id")):
            _fail(
                f"context {context_id!r} owner {ledger_row.get('gt_owner_id')!r} does not match its registry "
                f"role's owner {role.get('gt_owner_id')!r}; stale binding"
            )
        index[context_id] = role
    return index


def _validate_fixed_budget_row_shape(
    *, key: tuple[str, str], row: Mapping[str, Any], rules: Any
) -> None:
    context_id, candidate_id = key
    lo = rules.schema_tokens["coordinate_token_id_start"]
    hi = rules.schema_tokens["coordinate_token_id_end_exclusive"]
    coord = row.get("coord_token_ids")
    if (
        not isinstance(coord, Sequence)
        or isinstance(coord, (str, bytes))
        or len(coord) != 4
    ):
        _fail(
            f"fixed-budget candidate {key!r} must declare exactly four coord_token_ids"
        )
    for token_id in coord:
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not lo <= token_id < hi
        ):
            _fail(
                f"fixed-budget candidate {key!r} has a coord token outside the registered coordinate domain",
                token_id=token_id,
                domain=[lo, hi],
            )
    for required in (
        "source_digest",
        "family_id",
        "matched_control_group",
        "rung",
        "region",
        "population",
    ):
        value = row.get(required)
        if not isinstance(value, str) or not value:
            _fail(
                f"fixed-budget candidate {key!r} is missing retained metadata field {required!r}"
            )
    if not isinstance(row.get("is_control"), bool):
        _fail(f"fixed-budget candidate {key!r}.is_control must be a boolean")
    control_kind = row.get("control_kind")
    if row["is_control"] and (not isinstance(control_kind, str) or not control_kind):
        _fail(
            f"fixed-budget candidate {key!r} is_control=true but declares no control_kind"
        )


def load_and_validate_contract(
    *,
    registry: str | Path,
    owner_context_ledger: str | Path,
    decision_rules: str | Path,
    mechanism_decision_rules: str | Path,
    fixed_budget_candidates: str | Path,
    planner_receipt: str | Path,
    runtime_identity: str | Path,
    infer_config: str | Path,
    source_jsonl: str | Path,
) -> ValidatedContract:
    registry_path = _resolved_file(registry, "FN mechanism registry")
    ledger_path = _resolved_file(owner_context_ledger, "owner-context-ledger.jsonl")
    rules_path = _resolved_file(decision_rules, "landscape-decision-rules.json")
    mechanism_rules_path = _resolved_file(
        mechanism_decision_rules, "mechanism-decision-rules.json"
    )
    fixed_budget_path = _resolved_file(
        fixed_budget_candidates, "fixed-budget-candidates.jsonl"
    )
    planner_receipt_path = _resolved_file(planner_receipt, "planner receipt")
    runtime_identity_path = _resolved_file(runtime_identity, "runtime-identity.json")
    infer_config_path = _resolved_file(infer_config, "infer config")
    source_jsonl_path = _resolved_file(source_jsonl, "source JSONL")

    planner = _load_planner_receipt(planner_receipt_path)
    try:
        _bind_planner_source(
            receipt=planner,
            source_key="panel",
            path=source_jsonl_path,
            label="source JSONL / planner source panel",
        )
    except merger.MergeContractError as exc:
        _fail(str(exc))
    registry_document = _load_registry(registry_path, planner_receipt=planner)
    ledger_by_context = _load_ledger(ledger_path, planner_receipt=planner)
    _bind_planner_output(
        receipt=planner,
        output_key="landscape_decision_rules",
        path=rules_path,
        label="landscape-decision-rules.json",
    )

    rules_raw = _read_json(rules_path)
    # A landed semantic_core is only required/authored for production-mode
    # rules documents (mirrors prepare_sorted_fn_successor_inputs._reseal_
    # semantic_core, which likewise skips test_fixture-mode templates that
    # never carry one); when present, it must round-trip through the
    # canonical predecessor validator unchanged.
    if "semantic_core" in rules_raw:
        try:
            _validate_semantic_core_payload(rules_raw)
        except ValueError as exc:
            _fail(
                f"landscape-decision-rules.json semantic_core failed round-trip verification: {exc}"
            )
    rules = scorer.load_decision_rules(rules_path)
    mechanism_rules = _load_mechanism_decision_rules(
        mechanism_rules_path,
        planner_receipt=planner,
        landscape_rules_path=rules_path,
        registry_document=registry_document,
    )

    fixed_budget_by_key = _load_fixed_budget_candidates(
        fixed_budget_path, planner_receipt=planner, ledger_by_context=ledger_by_context
    )
    _validate_fixed_budget_mechanism_binding(
        fixed_budget_by_key,
        mechanism_decision_rules_sha256=sha256_file(mechanism_rules_path),
    )
    _validate_reference_selection_bindings(
        ledger_by_context=ledger_by_context, fixed_budget_by_key=fixed_budget_by_key
    )
    # Reject score-derived candidate inputs: read the raw rows independently
    # of the merger's typed join and scan every one for score/logit/logprob
    # fields before any candidate is admitted.
    for index, raw_row in enumerate(_read_jsonl(fixed_budget_path)):
        try:
            planner_mod._reject_score_derived_fields(
                raw_row, label=f"fixed-budget candidate row {index}"
            )
        except planner_mod.SuccessorInputPlanError as exc:
            _fail(str(exc))
    for key, row in fixed_budget_by_key.items():
        _validate_fixed_budget_row_shape(key=key, row=row, rules=rules)

    expected_identity = _expected_runtime_identity(ledger_by_context)
    context_role_index = _context_role_index(
        registry=registry_document, ledger_by_context=ledger_by_context
    )

    # Reuse the successor behavior runner's production identity owner.  It
    # validates the frozen receipt, re-hashes its Task-0/manifest sources and
    # rebuilds the complete identity document byte-for-byte; a local subset
    # interpretation here would be weaker and is deliberately forbidden.
    try:
        runtime_identity_doc, runtime_identity_receipt = (
            behavior_runner.validate_runtime_identity(runtime_identity_path)
        )
        live_config_admission = behavior_runner.validate_static_live_config_binding(
            infer_config_path=infer_config_path,
            source_jsonl_path=source_jsonl_path,
            runtime_identity=runtime_identity_doc,
            runtime_receipt=runtime_identity_receipt,
        )
    except (behavior_runner.BehaviorContractError, OSError, ValueError) as exc:
        _fail(str(exc))
    live_config_admission = {
        **live_config_admission,
        "infer_config": {
            "path": str(infer_config_path),
            "sha256": sha256_file(infer_config_path),
        },
    }
    runtime_identity_admission = validate_runtime_identity_against_static_contract(
        runtime_identity_doc, rules=rules, expected_identity=expected_identity
    )

    return ValidatedContract(
        planner_receipt=planner,
        registry=registry_document,
        ledger_by_context=ledger_by_context,
        rules=rules,
        rules_path=rules_path,
        mechanism_decision_rules=mechanism_rules,
        mechanism_decision_rules_path=mechanism_rules_path,
        fixed_budget_by_key=fixed_budget_by_key,
        expected_identity=expected_identity,
        runtime_identity_doc=runtime_identity_doc,
        runtime_identity_receipt=runtime_identity_receipt,
        runtime_identity_admission=runtime_identity_admission,
        live_config_admission=live_config_admission,
        context_role_index=context_role_index,
        source_paths={
            "registry": registry_path,
            "owner_context_ledger": ledger_path,
            "decision_rules": rules_path,
            "mechanism_decision_rules": mechanism_rules_path,
            "fixed_budget_candidates": fixed_budget_path,
            "planner_receipt": planner_receipt_path,
            "runtime_identity": runtime_identity_path,
            "infer_config": infer_config_path,
            "source_jsonl": source_jsonl_path,
        },
    )


def select_contexts(
    contract: ValidatedContract, include_context_ids: Sequence[str] | None
) -> tuple[list[str], dict[str, Any]]:
    available = sorted(contract.ledger_by_context)
    requested = list(include_context_ids or ())
    if len(requested) != len(set(requested)):
        _fail("--include-context-id contains duplicates", requested=requested)
    unknown = sorted(set(requested) - set(available))
    if unknown:
        _fail(
            "--include-context-id names context IDs absent from the validated ledger",
            unknown=unknown,
            available=available,
        )
    included = sorted(requested) if requested else available
    selection = {
        "mode": "explicit_allowlist" if requested else "all_validated_contexts",
        "available_context_ids": available,
        "included_context_ids": included,
        "excluded_context_ids": sorted(set(available) - set(included)),
    }
    return included, {**selection, "selection_sha256": sha256_json(selection)}


def select_rungs(
    contract: ValidatedContract,
    *,
    included_context_ids: Sequence[str],
    include_rungs: Sequence[str] | None,
) -> tuple[list[str], dict[str, Any]]:
    """Select score rungs only after the complete candidate/rules contract passes."""

    rung_quotas = _mapping(
        contract.mechanism_decision_rules.get("rung_quotas"),
        "mechanism-decision-rules.json.rung_quotas",
    )
    declared_rule_rungs = sorted(
        _string(value, "mechanism-decision-rules.json rung") for value in rung_quotas
    )
    candidate_rungs = {
        _string(row.get("rung"), "fixed-budget candidate.rung")
        for row in contract.fixed_budget_by_key.values()
    }
    undeclared_candidate_rungs = sorted(candidate_rungs - set(declared_rule_rungs))
    if undeclared_candidate_rungs:
        _fail(
            "fixed-budget candidates contain rungs absent from mechanism decision rules",
            undeclared=undeclared_candidate_rungs,
        )

    selected_context_set = set(included_context_ids)
    available_for_contexts = sorted(
        {
            str(row["rung"])
            for (context_id, _candidate_id), row in contract.fixed_budget_by_key.items()
            if context_id in selected_context_set
        }
    )
    if include_rungs is None or not include_rungs:
        _fail(
            "at least one explicit --include-rung is required; implicit all-rungs scoring is forbidden",
            available=available_for_contexts,
        )
    requested = list(include_rungs)
    if len(requested) != len(set(requested)):
        _fail("--include-rung contains duplicates", requested=requested)
    unknown = sorted(set(requested) - set(declared_rule_rungs))
    if unknown:
        _fail(
            "--include-rung names rungs absent from mechanism decision rules",
            unknown=unknown,
            declared_rule_rungs=declared_rule_rungs,
        )
    if "scalar_smoke" in requested and set(requested) != {"scalar_smoke"}:
        _fail(
            "--include-rung scalar_smoke cannot be combined with any other rung; scalar runs must be independently attestable",
            requested=requested,
        )
    unavailable = sorted(set(requested) - set(available_for_contexts))
    if unavailable:
        _fail(
            "--include-rung names rungs with no candidates in the selected contexts",
            unavailable=unavailable,
            available=available_for_contexts,
        )

    included = sorted(requested)
    included_set = set(included)
    selected_items = [
        (key, row)
        for key, row in contract.fixed_budget_by_key.items()
        if key[0] in selected_context_set and row["rung"] in included_set
    ]
    if not selected_items:
        _fail(
            "context/rung selection is empty; refusing to produce an unscored artifact",
            included_context_ids=sorted(selected_context_set),
            included_rungs=included,
        )
    selected_context_ids = sorted(
        {str(row["owner_context_id"]) for _key, row in selected_items}
    )
    selected_counts_by_rung = {
        rung: sum(row["rung"] == rung for _key, row in selected_items)
        for rung in included
    }
    selected_candidate_keys = sorted([list(key) for key, _row in selected_items])
    selection = {
        "mode": "explicit_allowlist",
        "declared_rule_rungs": declared_rule_rungs,
        "available_candidate_rungs": sorted(candidate_rungs),
        "available_rungs_for_selected_contexts": available_for_contexts,
        "included_rungs": included,
        "excluded_rungs_for_selected_contexts": sorted(
            set(available_for_contexts) - included_set
        ),
        "selected_context_ids": selected_context_ids,
        "contexts_without_selected_candidates": sorted(
            selected_context_set - set(selected_context_ids)
        ),
        "selected_candidate_count": len(selected_items),
        "selected_candidate_counts_by_rung": selected_counts_by_rung,
        "selected_candidate_keyset_sha256": sha256_json(selected_candidate_keys),
    }
    return included, {**selection, "selection_sha256": sha256_json(selection)}


def build_execution_selection(
    *, context_selection: Mapping[str, Any], rung_selection: Mapping[str, Any]
) -> dict[str, Any]:
    content = {
        "context_selection_sha256": context_selection["selection_sha256"],
        "rung_selection_sha256": rung_selection["selection_sha256"],
        "requested_context_ids": list(context_selection["included_context_ids"]),
        "selected_context_ids": list(rung_selection["selected_context_ids"]),
        "selected_rungs": list(rung_selection["included_rungs"]),
        "candidate_count": rung_selection["selected_candidate_count"],
        "candidate_keyset_sha256": rung_selection["selected_candidate_keyset_sha256"],
    }
    return {**content, "selection_sha256": sha256_json(content)}


# ---------------------------------------------------------------------------
# Score row assembly
# ---------------------------------------------------------------------------


def _candidate_probe_suffixes(coord_token_ids: Sequence[int]) -> list[list[int]]:
    """Relative suffixes ``[x1]``, ``[x1,y1]``, ``[x1,y1,x2]`` a candidate will consume.

    Mirrors ``score_sorted_owner_basin_landscape._candidate_reforward_suffixes``:
    the root (depth 0) forward already yields the x1 distribution, so only
    the three subsequent literal continuations need prefetching.
    """

    tokens = [int(value) for value in coord_token_ids[:3]]
    return [tokens[:depth] for depth in range(1, len(tokens) + 1)]


def build_score_row(
    *,
    context_id: str,
    candidate: Mapping[str, Any],
    ledger_row: Mapping[str, Any],
    rules: Any,
    combined: Mapping[str, Any],
) -> dict[str, Any]:
    raw = dict(combined["raw"])
    policy = dict(combined["auxiliary_policy"])
    _finite(
        raw.get("complete_box_logprob_sum"),
        label=f"{candidate['candidate_id']}.raw_model_logprob.complete_box_logprob_sum",
    )
    for slot in scorer.COORD_SLOTS:
        _finite(
            raw.get(f"{slot}_logprob"),
            label=f"{candidate['candidate_id']}.raw_model_logprob.{slot}_logprob",
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "candidate_id": candidate["candidate_id"],
        "context_id": context_id,
        "owner_context_id": context_id,
        "diagnostic_owner_id": ledger_row["diagnostic_owner_id"],
        "gt_owner_id": ledger_row["gt_owner_id"],
        "image_id": ledger_row["image_id"],
        "rung": candidate["rung"],
        "region": candidate["region"],
        "population": candidate["population"],
        "is_control": bool(candidate.get("is_control", False)),
        "control_kind": candidate.get("control_kind"),
        "matched_control_group": candidate["matched_control_group"],
        "family_id": candidate["family_id"],
        "iou_to_target": candidate.get("iou_to_target"),
        "source_digest": candidate["source_digest"],
        "coord_token_ids": [int(v) for v in candidate["coord_token_ids"]],
        "candidate_neighborhood_member": bool(
            candidate.get("candidate_neighborhood_member", False)
        ),
        "candidate_neighborhood_id": candidate.get("candidate_neighborhood_id"),
        "mechanism_decision_rules_sha256": candidate["mechanism_decision_rules_sha256"],
        "other_owner_gt_owner_id": candidate.get("other_owner_gt_owner_id"),
        "other_owner_selection_trace": candidate.get("other_owner_selection_trace"),
        "exact_gt_singleton_member": bool(
            candidate.get("exact_gt_singleton_member", False)
        ),
        "exact_gt_singleton_id": candidate.get("exact_gt_singleton_id"),
        "native_repetition_penalty_stratum": NATIVE_REPETITION_PENALTY_STRATUM,
        "rule_digest": rules.rules_digest,
        "ledger_context_tokens_sha256": ledger_row["context_tokens"][
            "token_ids_sha256"
        ],
        "raw_model_logprob": raw,
        "auxiliary_policy_scores": policy,
        "likelihood_channel_note": _LIKELIHOOD_CHANNEL_NOTE,
    }


# ---------------------------------------------------------------------------
# Implementation identity (dirty-diff receipt binding; no live-diff content)
# ---------------------------------------------------------------------------


def _git_capture(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    return completed.stdout


def build_implementation_provenance() -> dict[str, Any]:
    relevant = (
        Path("scripts/research/score_sorted_fn_fixed_budget.py"),
        Path("scripts/research/score_sorted_owner_basin_landscape.py"),
        Path("scripts/research/merge_sorted_fn_successor_score_shards.py"),
        Path("scripts/research/prepare_sorted_fn_successor_inputs.py"),
    )
    relative_paths = [path.as_posix() for path in relevant]
    status = _git_capture(
        "status", "--short", "--untracked-files=all", "--", *relative_paths
    )
    diff = _git_capture(
        "diff", "--no-ext-diff", "--binary", "HEAD", "--", *relative_paths
    )
    import hashlib

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


# ---------------------------------------------------------------------------
# Live image admission
# ---------------------------------------------------------------------------


_LOWERCASE_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_RAW_IMAGE_CONTENT_HASH_DOMAIN = "raw_file_bytes_sha256"
_EXECUTED_MEDIA_HASH_DOMAIN = "coordexp_rgb8_pixels_v1_sha256"


def validate_materialized_image_identity(
    *,
    image_id: str,
    ledger_rows: Sequence[Mapping[str, Any]],
    source_panel_sha256: str,
    planned_image_content_sha256: str,
    executed_media_sha256: Sequence[str],
    expected_materialization_count: int,
    expected_executed_media_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind native RGB materialization to the ledger's project image identity.

    The owner-context ledger does not define image identity as a filesystem
    path.  Its exact project semantics are
    ``sha256_json({"image_id": ..., "panel_sha256": ...})``.  The frozen
    panel/source digest establishes that ledger identity.  The planned digest
    identifies raw encoded file bytes, while the executed digest identifies
    canonical transformed RGB pixels.  Those domains intentionally differ:
    the HF backend binds them by verifying the raw digest before decoding and
    hashing the exact RGB object passed to the processor.
    """

    expected_ledger_identity = sha256_json(
        {"image_id": str(image_id), "panel_sha256": source_panel_sha256}
    )
    observed_ledger_identities = {
        _string(row.get("image_identity"), "owner-context ledger image_identity")
        for row in ledger_rows
    }
    if observed_ledger_identities != {expected_ledger_identity}:
        _fail(
            "owner-context ledger image_identity differs from the frozen source-panel identity",
            image_id=image_id,
            observed=sorted(observed_ledger_identities),
            expected=expected_ledger_identity,
        )
    if (
        not isinstance(planned_image_content_sha256, str)
        or _LOWERCASE_SHA256_PATTERN.fullmatch(planned_image_content_sha256) is None
    ):
        _fail(
            "planned raw image content identity must be a lowercase SHA-256 digest",
            image_id=image_id,
            observed=planned_image_content_sha256,
        )
    if (
        isinstance(expected_materialization_count, bool)
        or not isinstance(expected_materialization_count, int)
        or expected_materialization_count <= 0
    ):
        _fail(
            "expected materialization count must be a positive integer",
            image_id=image_id,
            observed=expected_materialization_count,
        )
    executed = list(executed_media_sha256)
    if not executed:
        _fail(
            "executed RGB identity is missing from native materialization",
            image_id=image_id,
        )
    if len(executed) != expected_materialization_count:
        _fail(
            "executed RGB materialization count differs from the required count",
            image_id=image_id,
            observed=len(executed),
            expected=expected_materialization_count,
        )
    invalid_executed = [
        value
        for value in executed
        if not isinstance(value, str)
        or _LOWERCASE_SHA256_PATTERN.fullmatch(value) is None
    ]
    if invalid_executed:
        _fail(
            "executed RGB identity must be a lowercase SHA-256 digest",
            image_id=image_id,
            observed=invalid_executed,
        )
    canonical_executed_media_sha256 = executed[0]
    if any(value != canonical_executed_media_sha256 for value in executed):
        _fail(
            "materialized executed RGB identities are not unanimous",
            image_id=image_id,
            observed=executed,
        )
    if expected_executed_media_sha256 is not None and (
        not isinstance(expected_executed_media_sha256, str)
        or _LOWERCASE_SHA256_PATTERN.fullmatch(expected_executed_media_sha256) is None
    ):
        _fail(
            "expected executed RGB identity must be a lowercase SHA-256 digest",
            image_id=image_id,
            observed=expected_executed_media_sha256,
        )
    if (
        expected_executed_media_sha256 is not None
        and canonical_executed_media_sha256 != expected_executed_media_sha256
    ):
        _fail(
            "repeated materialization changed canonical executed RGB identity",
            image_id=image_id,
            observed=executed,
            expected=expected_executed_media_sha256,
        )
    return {
        "status": "passed_before_coordinate_score",
        "image_id": str(image_id),
        "ledger_image_identity": expected_ledger_identity,
        "source_panel_sha256": source_panel_sha256,
        "planned_image_content_sha256": planned_image_content_sha256,
        "planned_image_content_hash_domain": _RAW_IMAGE_CONTENT_HASH_DOMAIN,
        "executed_media_sha256": executed,
        "canonical_executed_media_sha256": canonical_executed_media_sha256,
        "executed_media_hash_domain": _EXECUTED_MEDIA_HASH_DOMAIN,
        "materialization_count": len(executed),
        "expected_materialization_count": expected_materialization_count,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.infer_config is None or args.source_jsonl is None:
        _fail(
            "--infer-config and --source-jsonl are required for contract and live admission"
        )
    contract = load_and_validate_contract(
        registry=args.fn_mechanism_registry,
        owner_context_ledger=args.owner_context_ledger,
        decision_rules=args.decision_rules,
        mechanism_decision_rules=args.mechanism_decision_rules,
        fixed_budget_candidates=args.fixed_budget_candidates,
        planner_receipt=args.planner_receipt,
        runtime_identity=args.runtime_identity,
        infer_config=args.infer_config,
        source_jsonl=args.source_jsonl,
    )
    included_context_ids, context_selection = select_contexts(
        contract, args.include_context_id
    )
    included_rungs, rung_selection = select_rungs(
        contract,
        included_context_ids=included_context_ids,
        include_rungs=args.include_rung,
    )
    execution_selection = build_execution_selection(
        context_selection=context_selection, rung_selection=rung_selection
    )

    if args.validate_contract_only:
        source_digests = {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in contract.source_paths.items()
        }
        return {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "runtime_execution_status": CONTRACT_VALIDATED_STATUS,
            "candidate_count": rung_selection["selected_candidate_count"],
            "selection": execution_selection,
            "context_selection": context_selection,
            "rung_selection": rung_selection,
            "source_digests": source_digests,
            "live_config_admission": contract.live_config_admission,
            "runtime_identity_admission": {
                "frozen_source_rebuild": contract.runtime_identity_receipt,
                "preload": contract.runtime_identity_admission,
            },
            "decision_rules": {
                "path": str(contract.rules_path),
                "file_sha256": sha256_file(contract.rules_path),
                "core_rule_digest": contract.rules.rules_digest,
            },
            "mechanism_decision_rules": {
                "path": str(contract.mechanism_decision_rules_path),
                "file_sha256": sha256_file(contract.mechanism_decision_rules_path),
                "self_digest": contract.mechanism_decision_rules["self_digest"],
                "parent_execution_rules_sha256": contract.mechanism_decision_rules[
                    "upstream_digests"
                ]["execution_landscape_decision_rules_sha256"],
            },
        }

    scorer.pin_fp32_parity_flags()
    if args.output_dir is None:
        _fail("live scoring requires --output-dir")

    output_dir: Path = args.output_dir.expanduser().resolve()
    jsonl_path = output_dir / OUTPUT_JSONL_NAME
    receipt_path = output_dir / RECEIPT_NAME
    _admit_output_dir(
        jsonl_path=jsonl_path, receipt_path=receipt_path, force=args.force
    )

    import torch as _torch

    if not _torch.cuda.is_available():
        _fail("CUDA is required for live scoring; pass an explicit --cuda-device")
    if not args.cuda_device:
        _fail("live scoring requires an explicit --cuda-device")

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

    resolved = load_infer_config(contract.source_paths["infer_config"])
    config = resolved.config
    live_config_admission = contract.live_config_admission
    generation_config_fingerprint = config_sha256_json(
        config.generation.model_dump(mode="json")
    )
    if (
        sha256_file(contract.source_paths["infer_config"])
        != live_config_admission["infer_config"]["sha256"]
    ):
        _fail("infer config changed after CPU live-binding admission")
    if (
        resolved.fingerprint != live_config_admission["resolved_config_fingerprint"]
        or generation_config_fingerprint
        != live_config_admission["generation_config_fingerprint"]
    ):
        _fail(
            "resolved infer/generation config changed after CPU live-binding admission"
        )
    if (
        sha256_file(contract.source_paths["source_jsonl"])
        != live_config_admission["source_jsonl"]["sha256"]
    ):
        _fail("source JSONL changed after CPU live-binding admission")

    frontend = assemble_frontend(
        config, generation_config_fingerprint=generation_config_fingerprint
    )
    raw_rows = load_raw_examples(contract.source_paths["source_jsonl"])
    raw_by_image: dict[str, Any] = {}
    for row in raw_rows:
        source_metadata = row.metadata.get("source")
        if not isinstance(source_metadata, Mapping):
            _fail("production source row metadata.source must be a mapping")
        image_id = str(source_metadata.get("image_id"))
        if not image_id or image_id == "None" or image_id in raw_by_image:
            _fail(
                "production source JSONL image ids must be present and unique",
                image_id=image_id,
            )
        raw_by_image[image_id] = row
    template = _template_config(config)

    contexts_by_image: dict[str, list[str]] = {}
    for context_id in rung_selection["selected_context_ids"]:
        image_id = str(contract.ledger_by_context[context_id]["image_id"])
        contexts_by_image.setdefault(image_id, []).append(context_id)

    runtime_receipt_id = sha256_json(
        {
            "command": list(sys.argv),
            "owner_context_ledger": str(contract.source_paths["owner_context_ledger"]),
            "decision_rules": str(contract.source_paths["decision_rules"]),
            "fixed_budget_candidates": str(
                contract.source_paths["fixed_budget_candidates"]
            ),
            "planner_receipt": str(contract.source_paths["planner_receipt"]),
            "runtime_identity": str(contract.source_paths["runtime_identity"]),
            "runtime_identity_receipt_digest": contract.runtime_identity_doc[
                "receipt_digest"
            ],
            "infer_config": live_config_admission["infer_config"],
            "source_jsonl": live_config_admission["source_jsonl"],
            "selection": execution_selection,
            "started_at_unix": time.time(),
        }
    )

    all_rows: list[dict[str, Any]] = []
    owners_seen: set[str] = set()
    backend_receipt: dict[str, Any] = {}
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
    per_context_accounting: list[dict[str, Any]] = []
    runtime_postload_admission: dict[str, Any] = {}
    per_image_materialization_admission: list[dict[str, Any]] = []

    with open_backend_session(frontend.launch) as opened:
        if not isinstance(opened, HFBackendSession):
            raise RuntimeError("HF launch opened an unexpected backend session")
        backend_receipt = opened.receipt.to_artifact_dict()
        try:
            runtime_postload_admission = behavior_runner.validate_live_runtime_identity(
                observed_receipt=backend_receipt,
                frozen_identity=contract.runtime_identity_doc,
                resolved_config_fingerprint=live_config_admission[
                    "resolved_config_fingerprint"
                ],
                generation_config_fingerprint=live_config_admission[
                    "generation_config_fingerprint"
                ],
            )
        except behavior_runner.BehaviorContractError as exc:
            _fail(str(exc))
        tokenizer_identity = backend_receipt.get("tokenizer_identity") or {}
        model_identity = backend_receipt.get("model_identity") or {}
        model = opened._model  # noqa: SLF001
        if model is None:
            _fail("HF backend session did not expose its opened model")
        model.eval()

        for image_id in sorted(contexts_by_image):
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
            request = DecodeRequest(
                request_id=f"fn-fixed-budget-score:{image_id}",
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
                expected_image_grid_thw=(
                    image_grid_thw[0],
                    image_grid_thw[1],
                    image_grid_thw[2],
                ),
                logical_transform_id=image_plan.logical_transform_id,
                generation_policy=GenerationPolicy(
                    max_new_tokens=1,
                    repetition_penalty=1.0,
                    temperature=0.0,
                    top_p=1.0,
                    include_raw_model_logprob=True,
                ),
            )
            (
                native_inputs,
                executed_prompt_ids,
                _observed_grids,
                executed_media_sha256,
            ) = opened._materialize_native_inputs((request,))  # noqa: SLF001
            if tuple(executed_prompt_ids[0]) != expected_prompt_token_ids:
                _fail(
                    "materialized executed prompt tokens differ from the expected production prompt reconstruction",
                    image_id=image_id,
                )
            image_ledger_rows = [
                contract.ledger_by_context[context_id]
                for context_id in contexts_by_image[image_id]
            ]
            materialization_admission = validate_materialized_image_identity(
                image_id=image_id,
                ledger_rows=image_ledger_rows,
                source_panel_sha256=live_config_admission["source_jsonl"]["sha256"],
                planned_image_content_sha256=image_plan.image_content_sha256,
                executed_media_sha256=executed_media_sha256,
                expected_materialization_count=1,
            )
            per_image_materialization_admission.append(materialization_admission)

            batched_native_inputs: Mapping[str, Any] | None = None
            batched_full_reforward: Callable[[Sequence[Sequence[int]]], Any] | None = (
                None
            )

            for context_id in sorted(contexts_by_image[image_id]):
                ledger_row = contract.ledger_by_context[context_id]
                root_prefix_token_ids = [
                    int(v) for v in ledger_row["context_tokens"]["token_ids"]
                ]
                candidates = sorted(
                    (
                        row
                        for key, row in contract.fixed_budget_by_key.items()
                        if key[0] == context_id and row["rung"] in included_rungs
                    ),
                    key=lambda row: row["candidate_id"],
                )
                if not candidates:
                    _fail(
                        f"context {context_id!r} has no declared fixed-budget candidates"
                    )

                full_reforward = scorer._build_full_reforward_closure(
                    model, native_prompt_inputs=native_inputs
                )  # noqa: SLF001

                if args.full_reforward_batch_size > 1 and (
                    batched_reforward_admission is None
                    or batched_reforward_admission["status"] == "passed"
                ):
                    if batched_native_inputs is None:
                        (
                            materialized_batch,
                            batch_executed_prompt_ids,
                            _batch_grids,
                            batch_media_sha256,
                        ) = opened._materialize_native_inputs(  # noqa: SLF001
                            tuple(
                                request for _ in range(args.full_reforward_batch_size)
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
                        validate_materialized_image_identity(
                            image_id=image_id,
                            ledger_rows=image_ledger_rows,
                            source_panel_sha256=live_config_admission["source_jsonl"][
                                "sha256"
                            ],
                            planned_image_content_sha256=image_plan.image_content_sha256,
                            executed_media_sha256=batch_media_sha256,
                            expected_materialization_count=args.full_reforward_batch_size,
                            expected_executed_media_sha256=materialization_admission[
                                "canonical_executed_media_sha256"
                            ],
                        )
                        batched_native_inputs = materialized_batch
                        batched_full_reforward = (
                            scorer._build_batched_full_reforward_closure(  # noqa: SLF001
                                model,
                                native_prompt_inputs=batched_native_inputs,
                                maximum_batch_size=args.full_reforward_batch_size,
                            )
                        )
                    if batched_reforward_admission is None:
                        probe = candidates[0]
                        batched_reforward_admission = scorer.run_batched_reforward_parity_gate(
                            prefix_token_ids=root_prefix_token_ids,
                            coordinate_token_ids=probe["coord_token_ids"],
                            coordinate_token_id_start=contract.rules.schema_tokens[
                                "coordinate_token_id_start"
                            ],
                            coordinate_token_id_end_exclusive=contract.rules.schema_tokens[
                                "coordinate_token_id_end_exclusive"
                            ],
                            full_reforward=full_reforward,
                            batched_full_reforward=batched_full_reforward,
                            requested_batch_size=args.full_reforward_batch_size,
                        )
                        batched_reforward_admission["probe_context_id"] = context_id
                        batched_reforward_admission["probe_candidate_id"] = probe[
                            "candidate_id"
                        ]
                        if batched_reforward_admission["status"] != "passed":
                            batched_full_reforward = None

                effective_batch_size = (
                    args.full_reforward_batch_size
                    if batched_reforward_admission is not None
                    and batched_reforward_admission["status"] == "passed"
                    else 1
                )
                backend = scorer.FullReforwardBackend(
                    root_prefix_token_ids=root_prefix_token_ids,
                    full_reforward=full_reforward,
                    batched_full_reforward=batched_full_reforward,
                    full_reforward_batch_size=effective_batch_size,
                    context_id=context_id,
                    group_id=sha256_json(
                        {
                            "image_id": image_id,
                            "context_id": context_id,
                            "prefix_token_ids": root_prefix_token_ids,
                        }
                    ),
                )
                attestation = scorer.build_attestation_context(
                    expected_vocab_size=contract.rules.model_vocab_size,
                    tokenizer_identity=tokenizer_identity,
                    model_identity=model_identity,
                    rule_digest=contract.rules.rules_digest,
                    runtime_receipt_id=runtime_receipt_id,
                    frozen_tokenizer_identity_digest=contract.runtime_identity_admission[
                        "tokenizer_identity_sha256"
                    ],
                    frozen_model_identity_digest=contract.runtime_identity_admission[
                        "model_identity_sha256"
                    ],
                )

                for start in range(0, len(candidates), effective_batch_size):
                    chunk = candidates[start : start + effective_batch_size]
                    scorer._prefetch_backend_suffixes(  # noqa: SLF001
                        backend,
                        [
                            suffix
                            for candidate in chunk
                            for suffix in _candidate_probe_suffixes(
                                candidate["coord_token_ids"]
                            )
                        ],
                    )
                    for candidate in chunk:
                        combined = scorer.score_complete_box_candidate(
                            backend=backend,
                            prefill_logits=backend.root_logits,
                            coord_token_ids=candidate["coord_token_ids"],
                            attestation=attestation,
                            running_context_token_ids=root_prefix_token_ids,
                        )
                        row = build_score_row(
                            context_id=context_id,
                            candidate=candidate,
                            ledger_row=ledger_row,
                            rules=contract.rules,
                            combined=combined,
                        )
                        owners_seen.add(row["gt_owner_id"])
                        all_rows.append(row)

                per_context_accounting.append(backend.accounting())

    observed_candidate_keys = sorted(
        [[str(row["context_id"]), str(row["candidate_id"])] for row in all_rows]
    )
    if (
        len(all_rows) != execution_selection["candidate_count"]
        or sha256_json(observed_candidate_keys)
        != execution_selection["candidate_keyset_sha256"]
        or sorted({str(row["rung"]) for row in all_rows})
        != execution_selection["selected_rungs"]
        or sorted({str(row["context_id"]) for row in all_rows})
        != execution_selection["selected_context_ids"]
    ):
        _fail(
            "live score rows differ from the exact admitted context/rung candidate domain",
            observed_count=len(all_rows),
            expected_selection=execution_selection,
        )

    if batched_reforward_admission is None:
        _fail("requested batched full reforward was not admitted before scoring")

    admission_content = {
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
        "cache_path_note": (
            "this successor scorer never attempts the KV-cache path; parity_status/cache_admission_policy are "
            "honestly recorded in the predecessor's own 'failed'/None vocabulary so "
            "score_sorted_owner_basin_landscape.select_scoring_backend_from_parity reconstructs full_reforward_uncached"
        ),
        "batched_reforward_admission": batched_reforward_admission,
        "per_context_accounting": per_context_accounting,
    }
    scoring_backend_admission = {
        **admission_content,
        "sha256": sha256_json(admission_content),
    }

    environment = {
        "python_version": sys.version,
        "torch_version": _torch.__version__,
        "transformers_version": scorer._installed_package_version("transformers"),  # noqa: SLF001
        "cuda_available": _torch.cuda.is_available(),
        "device_name": _torch.cuda.get_device_name(0)
        if _torch.cuda.is_available()
        else None,
        "tf32": scorer.pin_fp32_parity_flags(),
    }
    receipt_content = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "generated_at_unix": time.time(),
        "command": list(sys.argv),
        "runtime_execution_status": LIVE_SCORING_PENDING_SEAL_STATUS,
        "runtime_receipt_id": runtime_receipt_id,
        "source_digests": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in contract.source_paths.items()
        },
        "decision_rules": {
            "path": str(contract.rules_path),
            "file_sha256": sha256_file(contract.rules_path),
            "core_rule_digest": contract.rules.rules_digest,
        },
        "mechanism_decision_rules": {
            "path": str(contract.mechanism_decision_rules_path),
            "file_sha256": sha256_file(contract.mechanism_decision_rules_path),
            "self_digest": contract.mechanism_decision_rules["self_digest"],
            "parent_execution_rules_sha256": contract.mechanism_decision_rules[
                "upstream_digests"
            ]["execution_landscape_decision_rules_sha256"],
        },
        "model_identity": model_identity,
        "tokenizer_identity": tokenizer_identity,
        "backend_session": dict(backend_receipt),
        "environment": environment,
        "live_config_admission": live_config_admission,
        "runtime_identity_admission": {
            "frozen_source_rebuild": contract.runtime_identity_receipt,
            "preload": contract.runtime_identity_admission,
            "postload": runtime_postload_admission,
        },
        "materialized_image_identity_admission": per_image_materialization_admission,
        "implementation_provenance": build_implementation_provenance(),
        "numeric_reproduction_tolerance": contract.rules.numeric_tolerance,
        "selection": execution_selection,
        "context_selection": context_selection,
        "rung_selection": rung_selection,
        "candidate_count": len(all_rows),
        "owners_covered": len(owners_seen),
        "likelihood_channels": {
            "raw": "fp32_log_softmax_over_the_complete_unfiltered_lm_head_vocabulary",
            "auxiliary_policy": [
                scorer._policy_view_key(p) for p in scorer.REPETITION_PENALTY_STRATA
            ],  # noqa: SLF001
            "policy_is_not_a_model_likelihood": True,
            "subset_normalization_forbidden": True,
        },
        "scoring_backend_admission": scoring_backend_admission,
        "execution_architecture": {
            "kv_cache_ever_used": False,
            "strategy": (
                "one scalar (root, depth-0) reforward per context yields the x1 distribution; y1/x2/y2 "
                "come from three subsequent literal use_cache=False full-prefix reforwards, optionally "
                "co-scheduled in an admitted equal-length same-context GPU batch (--full-reforward-batch-size)"
            ),
        },
        "not_reused": {
            "free_coordinate_tree": "this fixed lattice has no free-tree root; the predecessor's free-search surface is never invoked",
            "conditional_x1_domain": "the predecessor's complete conditional-x1 production contract is not required or relaxed",
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
        receipt_content["runtime_execution_status"] = LIVE_SCORING_SEALED_STATUS
        receipt_content["output_artifacts"] = {
            "fixed_budget_scores": {
                "path": str(jsonl_path),
                "sha256": sha256_file(jsonl_temp),
                "row_count": len(all_rows),
                "selection_sha256": execution_selection["selection_sha256"],
            }
        }
        receipt_temp.write_text(
            json.dumps(receipt_content, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        jsonl_temp.replace(jsonl_path)
        receipt_temp.replace(receipt_path)
    finally:
        jsonl_temp.unlink(missing_ok=True)
        receipt_temp.unlink(missing_ok=True)
    return receipt_content


def _admit_output_dir(*, jsonl_path: Path, receipt_path: Path, force: bool) -> None:
    """Create-or-identical for a live GPU run: never silently overwrite a sealed artifact.

    A GPU rerun is not guaranteed bit-identical, so this does not attempt a
    byte-identical comparison the way the CPU-only planner/merger artifacts
    do. Instead: an output dir holding a genuinely sealed prior receipt bound
    to the exact jsonl on disk is always refused (``--force`` included). An
    output dir holding only partial/incomplete remnants (e.g. a stray temp
    file, or a receipt not naming this exact jsonl) requires ``--force`` to
    clear before a fresh run may write into it. An empty dir needs neither.
    """

    jsonl_exists = jsonl_path.is_file()
    receipt_exists = receipt_path.is_file()
    if not jsonl_exists and not receipt_exists:
        return
    if jsonl_exists and receipt_exists:
        try:
            receipt_document = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            receipt_document = {}
        artifact = receipt_document.get("output_artifacts", {}).get(
            "fixed_budget_scores", {}
        )
        sealed = (
            receipt_document.get("schema_version") == RECEIPT_SCHEMA_VERSION
            and receipt_document.get("runtime_execution_status")
            == LIVE_SCORING_SEALED_STATUS
            and artifact.get("sha256") == sha256_file(jsonl_path)
        )
        if sealed:
            _fail(
                "refusing to overwrite a sealed fixed-budget score artifact in --output-dir; choose a new output "
                "dir instead of --force"
            )
    if not force:
        _fail(
            "--output-dir contains incomplete/partial prior artifacts; pass --force to clear them and retry"
        )
    jsonl_path.unlink(missing_ok=True)
    receipt_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-receipt", type=Path, required=True)
    parser.add_argument("--fn-mechanism-registry", type=Path, required=True)
    parser.add_argument("--owner-context-ledger", type=Path, required=True)
    parser.add_argument("--decision-rules", type=Path, required=True)
    parser.add_argument("--mechanism-decision-rules", type=Path, required=True)
    parser.add_argument("--fixed-budget-candidates", type=Path, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--include-context-id",
        action="append",
        default=None,
        help="repeatable exact context-ID allowlist applied only after the complete static contract has passed validation",
    )
    parser.add_argument(
        "--include-rung",
        action="append",
        default=None,
        help="required repeatable exact rung allowlist applied only after the complete static contract has passed validation; omission is rejected",
    )
    parser.add_argument(
        "--validate-contract-only",
        action="store_true",
        help="validate the CPU-side contract without loading a model",
    )
    parser.add_argument(
        "--full-reforward-batch-size",
        type=_positive_int,
        default=1,
        help="equal-length same-context GPU batch size for literal use_cache=False reforwards; 1 is the scalar reference",
    )
    parser.add_argument(
        "--cuda-device",
        type=str,
        default=None,
        help="explicit CUDA device (e.g. cuda:0); required for live scoring",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="clear an incomplete/partial prior output dir before writing; never overwrites a sealed artifact",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cuda_device and not args.validate_contract_only:
        import os

        os.environ.setdefault(
            "CUDA_VISIBLE_DEVICES", args.cuda_device.replace("cuda:", "")
        )
    receipt = run(args)
    print(
        json.dumps(
            {
                "rows": receipt.get("candidate_count"),
                "status": receipt["runtime_execution_status"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
