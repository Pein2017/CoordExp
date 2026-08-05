#!/usr/bin/env python3
"""Discovery-only provenance salvage/adjudication for the sorted all-person
route landscape unit's already-completed score shards.

Background (2026-08-03 code-identity race)
--------------------------------------------
``score_sorted_all_person_route_landscape.py`` built its shard receipt's
``code`` block by re-reading ``Path(__file__)`` at the *end* of the job.
Six long-running context captures were still executing while that file was
concurrently edited (to add ``--include-request-id``/merge plumbing); each
job's receipt therefore recorded whatever bytes happened to be on disk when
it finished, not the bytes that actually executed the run. The scorer has
since been fixed (receipt schema v2: an import-time
``executed_source_sha256`` fingerprint, captured once and immune to this
race -- see that module's docstring). This module exists only to examine
the six *pre-fix* shards honestly, not to re-legitimize them as if they had
passed a normal merge.

What this module is not
------------------------
It is not a relaxed merge. ``merge_sorted_all_person_route_landscape.py``'s
strict code-hash uniformity requirement is untouched and still rejects
these six shards outright -- correctly. This module never emits anything
resembling a normal merge "passed" disposition; every receipt it produces
is unambiguously scoped as discovery-only evidence (``claim_scope =
"descriptive_discovery_only"``), and it refuses to run at all unless the
supplied shards actually exhibit the documented heterogeneity (see
``--acknowledge-discovery-only-salvage`` below) -- it cannot be used to
launder an already-clean, already-uniform shard set through a weaker path.

What this module preserves as evidence
----------------------------------------
* every shard's declared code identity (whatever fields its receipt
  schema version happens to carry) and receipt schema version, listed
  per-shard, never collapsed to a single "the" value;
* every shard's score-JSONL digest;
* exact plan/request/context/kind coverage over the *declared* selection
  (reusing the merge module's own strict reconciliation: overlapping
  shards must still be disjoint or byte-identical; the union must still
  exactly equal the plan's own request domain for that selection -- a
  genuine content disagreement between shards, a coverage gap, or a
  foreign row is still a hard failure here, exactly as in strict merge);
* every shard's batch-parity admission, unmodified (a bare failed
  admission with no recorded fallback is still a hard failure here);
* the frozen numerical-repeat epsilon, recomputed from any in-plan
  ``numerical_repeat`` rows the primary shards happen to cover, *and*
  separately from any auxiliary independent-process repeat shards
  supplied via ``--independent-repeat-shard-dir`` -- these two sources are
  never conflated (see "Numerical-repeat evidence: in-process memo vs
  independent process" below); and
* the plan's own residual disclosure (teacher-forced chosen-token prefix
  parity remains ``unverified_by_this_merge``, unchanged) plus, when the
  covered contexts allow it, the CPU-only ``token_execution_parity``
  attestation (literal prefix-arithmetic proof, never a numerical
  chosen-token parity claim).

Numerical-repeat evidence: in-process memo vs. independent process
----------------------------------------------------------------------
The original eight ``self-due-gt17`` numerical repeats were all scored
inside *one* process against *one*
``score_sorted_owner_basin_landscape.FullReforwardBackend`` instance.  That
backend persistently memoizes the literal depth-1 (``[x1]``) and depth-2
(``[x1, y1]``) continuations for its entire lifetime.  Because every one of
the eight repeats requests the exact same literal candidate, the ``y1`` and
``x2`` coordinate distributions are computed by the model exactly once and
then reused verbatim for repeats 2 through 8; only ``x1`` (free, from the
shared root prefill) and ``y2`` (depth-3, never persistently memoized) are
genuinely re-evaluated per repeat.  A ``delta == 0`` finding from that
single-process run is therefore real independent evidence only for the
``y2`` slot -- it is not, by itself, cross-process numerical-stability
evidence for the complete four-coordinate sum.

``--independent-repeat-shard-dir`` accepts a separate, stronger evidence
source: shard directories each produced by an entirely independent
process/model load scoring the identical logical request. These share no
backend, no memo, and no process state with each other or with the
original in-process repeat -- their agreement is unconfounded across all
four coordinate slots. This module reports the two sources side by side
under distinctly named fields and never merges them into one number.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import attest_sorted_all_person_route_landscape as attest_module  # noqa: E402
from scripts.research import build_sorted_all_person_route_landscape as planner  # noqa: E402
from scripts.research import merge_sorted_all_person_route_landscape as merge_module  # noqa: E402
from scripts.research import score_sorted_all_person_route_landscape as score_module  # noqa: E402

ADJUDICATION_SCHEMA_VERSION = "sorted-all-person-route-landscape-provenance-adjudication.v1"
ADJUDICATION_NAME = "route-landscape-provenance-adjudication.json"
UNIT_ID = planner.UNIT_ID

#: The scorer's shard-receipt schema versions this adjudicator can read.
#: v1 is the pre-fix, end-of-job-file-hash schema (the race victim); v2 is
#: the current import-time-fingerprint schema. Any other value is an
#: unknown mismatch and is refused.
LEGACY_RECEIPT_SCHEMA_VERSION_V1 = "sorted-all-person-route-landscape-score-receipt.v1"
KNOWN_RECEIPT_SCHEMA_VERSIONS = frozenset(
    {LEGACY_RECEIPT_SCHEMA_VERSION_V1, score_module.RECEIPT_SCHEMA_VERSION}
)

#: The unit's own frozen numerical-repeat tolerance floor (unit.md);
#: independent of the scorer's looser batch-vs-scalar coordinate gate
#: (``BATCH_REFORWARD_COORD_LOGPROB_MAX_ABS_DIFF = 1e-3``).
UNIT_NUMERICAL_EPSILON_FLOOR = merge_module.EPSILON_FLOOR

MEMO_REUSE_CAVEAT = (
    "the original in-process eight-repeat measurement shares one FullReforwardBackend across all "
    "eight repeat_index rows; because that backend persistently memoizes the literal depth-1 ([x1]) "
    "and depth-2 ([x1, y1]) continuations for its lifetime, and every repeat requests the identical "
    "literal candidate, the y1 and x2 coordinate distributions are computed once and reused verbatim "
    "across repeats -- only x1 (shared free root-prefill logits) and y2 (ephemeral depth-3) are "
    "genuinely re-evaluated per repeat. A delta==0 in-process finding is therefore real independent "
    "evidence only for the y2 slot, not for the complete four-coordinate sum. See "
    "independent_process_numerical_repeat_admission for unconfounded, cross-process evidence."
)

canonical_json_bytes = score_module.canonical_json_bytes
sha256_json = score_module.sha256_json
sha256_file = score_module.sha256_file
_read_json = score_module._read_json  # noqa: SLF001
_read_jsonl = score_module._read_jsonl  # noqa: SLF001
_write_create_or_identical = score_module._write_create_or_identical  # noqa: SLF001
_mapping = merge_module._mapping  # noqa: SLF001
_sequence = merge_module._sequence  # noqa: SLF001


class ProvenanceAdjudicationError(RuntimeError):
    """Raised when an unknown (undocumented) mismatch or precondition failure is found."""

    def __init__(self, message: str, **context: Any) -> None:
        if context:
            message = f"{message} | context: {json.dumps(context, sort_keys=True, default=str)}"
        super().__init__(message)


def _fail(message: str, **context: Any) -> NoReturn:
    raise ProvenanceAdjudicationError(message, **context)


# ---------------------------------------------------------------------------
# Lenient shard loading (accepts v1 OR v2 receipt schema; strict on everything else)
# ---------------------------------------------------------------------------


def _lenient_load_shard(
    shard: merge_module.ShardInput, *, expected_plan_content_sha256: str
) -> merge_module.ValidatedShard:
    scores_path = shard.scores_path.expanduser().resolve(strict=True)
    receipt_path = shard.receipt_path.expanduser().resolve(strict=True)
    receipt = _read_json(receipt_path, f"shard receipt {receipt_path}")

    if receipt.get("unit_id") != UNIT_ID:
        _fail("shard receipt.unit_id is not this unit's own unit_id", receipt_path=str(receipt_path))
    if receipt.get("row_schema_version") != score_module.SCHEMA_VERSION:
        _fail(
            "shard receipt.row_schema_version is not this scorer's own row schema; unknown mismatch",
            receipt_path=str(receipt_path),
            observed=receipt.get("row_schema_version"),
        )
    receipt_schema_version = receipt.get("schema_version")
    if receipt_schema_version not in KNOWN_RECEIPT_SCHEMA_VERSIONS:
        _fail(
            "shard receipt.schema_version is not a recognized scorer receipt schema (v1 pre-fix or "
            "v2 current); unknown mismatch, refusing to adjudicate",
            receipt_path=str(receipt_path),
            observed=receipt_schema_version,
            known=sorted(KNOWN_RECEIPT_SCHEMA_VERSIONS),
        )

    plan_entry = _mapping(receipt.get("plan"), f"shard receipt {receipt_path}.plan")
    if plan_entry.get("receipt_content_sha256") != expected_plan_content_sha256:
        _fail(
            "shard was scored against a different plan (plan.receipt_content_sha256 mismatch); "
            "unknown mismatch, never tolerated even in salvage mode",
            receipt_path=str(receipt_path),
            observed=plan_entry.get("receipt_content_sha256"),
            expected=expected_plan_content_sha256,
        )

    row_ids = _sequence(receipt.get("row_ids"), f"shard receipt {receipt_path}.row_ids")
    declared_row_ids = sorted(str(v) for v in row_ids)
    if len(declared_row_ids) != len(set(declared_row_ids)):
        _fail("shard receipt.row_ids contains duplicates", receipt_path=str(receipt_path))

    rows = _read_jsonl(scores_path, f"shard scores {scores_path}")
    rows_by_request_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("schema_version") != score_module.SCHEMA_VERSION:
            _fail(
                "shard score row does not carry this scorer's own row schema_version; unknown mismatch",
                scores_path=str(scores_path),
            )
        if row.get("unit_id") != UNIT_ID:
            _fail("shard score row does not carry this unit's own unit_id", scores_path=str(scores_path))
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            _fail("shard score row has a missing/invalid request_id", scores_path=str(scores_path))
        if request_id in rows_by_request_id:
            _fail(
                "shard scores file has a duplicate request_id within one shard",
                scores_path=str(scores_path),
                request_id=request_id,
            )
        rows_by_request_id[request_id] = row

    if sorted(rows_by_request_id) != declared_row_ids:
        _fail(
            "shard scores file content does not match the exact request-id set the shard receipt "
            "declared",
            scores_path=str(scores_path),
            receipt_path=str(receipt_path),
        )
    counts = _mapping(receipt.get("counts"), f"shard receipt {receipt_path}.counts")
    if counts.get("rows") != len(rows_by_request_id):
        _fail(
            "shard receipt.counts.rows does not match the actual row count on disk",
            receipt_path=str(receipt_path),
        )

    return merge_module.ValidatedShard(
        scores_path=scores_path,
        receipt_path=receipt_path,
        receipt=receipt,
        rows_by_request_id=rows_by_request_id,
    )


def _code_identity_of(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Every code-identity field the shard's receipt schema happens to carry.

    v1 receipts only ever had ``path``/``sha256``; v2 receipts additionally
    carry ``executed_source_sha256``/``receipt_time_file_sha256``/
    ``source_drift_detected``. Never assume a field is present.
    """

    code = _mapping(receipt.get("code"), "shard receipt.code")
    return {
        "schema_version": receipt.get("schema_version"),
        "path": code.get("path"),
        "sha256": code.get("sha256"),
        "executed_source_sha256": code.get("executed_source_sha256"),
        "receipt_time_file_sha256": code.get("receipt_time_file_sha256"),
        "source_drift_detected": code.get("source_drift_detected"),
    }


# ---------------------------------------------------------------------------
# Batch-vs-scalar drift evidence
# ---------------------------------------------------------------------------


def _observed_batch_scalar_max_abs_diffs(validated: Sequence[merge_module.ValidatedShard]) -> list[float]:
    observed: list[float] = []
    for shard in validated:
        admission = _mapping(
            shard.receipt.get("scoring_backend_admission"),
            f"shard {shard.receipt_path}.scoring_backend_admission",
        )
        for entry in _sequence(
            admission.get("per_context_accounting", []),
            f"shard {shard.receipt_path}.scoring_backend_admission.per_context_accounting",
        ):
            entry = _mapping(entry, "per_context_accounting entry")
            batch_admission = entry.get("batch_admission")
            if not isinstance(batch_admission, Mapping):
                continue
            for comparison in batch_admission.get("comparisons") or ():
                if not isinstance(comparison, Mapping):
                    continue
                value = comparison.get("coordinate_logprob_max_abs_diff")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    observed.append(float(value))
    return observed


# ---------------------------------------------------------------------------
# Independent-process numerical-repeat evidence (auxiliary, out-of-plan-coverage)
# ---------------------------------------------------------------------------


def _load_independent_repeat_row(shard_dir: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    scores_path = shard_dir / score_module.OUTPUT_JSONL_NAME
    receipt_path = shard_dir / score_module.OUTPUT_RECEIPT_NAME
    receipt = _read_json(receipt_path, f"independent repeat shard receipt {receipt_path}")
    if receipt.get("unit_id") != UNIT_ID:
        _fail("independent repeat shard receipt.unit_id is not this unit's own", receipt_path=str(receipt_path))
    if receipt.get("row_schema_version") != score_module.SCHEMA_VERSION:
        _fail(
            "independent repeat shard receipt.row_schema_version is not this scorer's own row schema",
            receipt_path=str(receipt_path),
        )
    if receipt.get("schema_version") not in KNOWN_RECEIPT_SCHEMA_VERSIONS:
        _fail(
            "independent repeat shard receipt.schema_version is not recognized",
            receipt_path=str(receipt_path),
            observed=receipt.get("schema_version"),
        )
    expected_receipt_content_sha256 = sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    if receipt.get("receipt_content_sha256") != expected_receipt_content_sha256:
        _fail(
            "independent repeat shard receipt_content_sha256 does not match its content",
            receipt_path=str(receipt_path),
        )
    rows = _read_jsonl(scores_path, f"independent repeat shard scores {scores_path}")
    if len(rows) != 1:
        _fail(
            "an --independent-repeat-shard-dir must contain exactly one score row (one request per "
            "independent process)",
            scores_path=str(scores_path),
            observed_row_count=len(rows),
        )
    row = rows[0]
    if row.get("schema_version") != score_module.SCHEMA_VERSION or row.get("unit_id") != UNIT_ID:
        _fail("independent repeat shard row does not carry this scorer's own schema/unit identity", scores_path=str(scores_path))
    raw = _mapping(row.get("raw_model_logprob"), f"{scores_path} row.raw_model_logprob")
    value = raw.get("complete_box_logprob_sum")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        _fail(
            "independent repeat shard row has a missing/non-finite "
            "raw_model_logprob.complete_box_logprob_sum",
            scores_path=str(scores_path),
        )
    request_id = row.get("request_id")
    if receipt.get("row_ids") != [request_id]:
        _fail(
            "independent repeat shard receipt.row_ids does not exactly name its one score row",
            receipt_path=str(receipt_path),
            observed=receipt.get("row_ids"),
            expected=[request_id],
        )
    counts = _mapping(receipt.get("counts"), f"independent repeat shard receipt {receipt_path}.counts")
    if counts.get("rows") != 1:
        _fail(
            "independent repeat shard receipt.counts.rows is not exactly one",
            receipt_path=str(receipt_path),
            observed=counts.get("rows"),
        )
    return row, receipt


def _independent_repeat_measurement_identity(
    row: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    coord_values = _sequence(row.get("coord_token_ids"), "independent repeat row.coord_token_ids")
    if (
        len(coord_values) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in coord_values)
    ):
        _fail(
            "independent repeat row.coord_token_ids must contain exactly four integer token IDs",
            observed=coord_values,
        )
    coord_token_ids = [int(value) for value in coord_values]
    if row.get("coord_token_ids_sha256") != sha256_json(coord_token_ids):
        _fail("independent repeat row.coord_token_ids_sha256 does not match coord_token_ids")

    source_identity = _mapping(receipt.get("source_identity"), "independent repeat receipt.source_identity")
    plan_identity = _mapping(receipt.get("plan"), "independent repeat receipt.plan")
    code_identity = _mapping(receipt.get("code"), "independent repeat receipt.code")
    decision_channel = _mapping(
        receipt.get("decision_channel"), "independent repeat receipt.decision_channel"
    )
    backend_admission = _mapping(
        receipt.get("scoring_backend_admission"),
        "independent repeat receipt.scoring_backend_admission",
    )
    environment = _mapping(receipt.get("environment"), "independent repeat receipt.environment")

    if decision_channel.get("name") != score_module.DECISION_BEARING_CHANNEL:
        _fail(
            "independent repeat receipt does not use the frozen decision-bearing channel",
            observed=decision_channel.get("name"),
            expected=score_module.DECISION_BEARING_CHANNEL,
        )
    if backend_admission.get("selected_backend") != score_module.scorer.FULL_REFORWARD_SCORING_BACKEND:
        _fail(
            "independent repeat receipt does not use the frozen full-reforward scoring backend",
            observed=backend_admission.get("selected_backend"),
        )
    if backend_admission.get("use_cache") is not False or backend_admission.get("cache_enabled") is not False:
        _fail("independent repeat receipt scoring backend is not explicitly uncached")

    raw = _mapping(row.get("raw_model_logprob"), "independent repeat row.raw_model_logprob")
    vocab_attestation = _mapping(
        raw.get("vocab_attestation"), "independent repeat row.raw_model_logprob.vocab_attestation"
    )
    expected_slots = tuple(score_module.scorer.COORD_SLOTS)
    if set(vocab_attestation) != set(expected_slots):
        _fail(
            "independent repeat row vocab_attestation does not cover exactly the coordinate slots",
            observed=sorted(vocab_attestation),
            expected=list(expected_slots),
        )

    stable_vocab_attestation: dict[str, dict[str, Any]] = {}
    runtime_receipt_ids: set[str] = set()
    domain_digests: set[str] = set()
    for slot in expected_slots:
        slot_attestation = _mapping(vocab_attestation[slot], f"vocab_attestation.{slot}")
        runtime_receipt_id = slot_attestation.get("runtime_receipt_id")
        domain_digest = slot_attestation.get("domain_digest")
        if not isinstance(runtime_receipt_id, str) or not runtime_receipt_id:
            _fail("independent repeat row has a missing runtime_receipt_id", slot=slot)
        if not isinstance(domain_digest, str) or not domain_digest:
            _fail("independent repeat row has a missing domain_digest", slot=slot)
        runtime_receipt_ids.add(runtime_receipt_id)
        domain_digests.add(domain_digest)
        stable_vocab_attestation[slot] = {
            "filtered": slot_attestation.get("filtered"),
            "model_identity_digest": slot_attestation.get("model_identity_digest"),
            "rule_digest": slot_attestation.get("rule_digest"),
            "tokenizer_identity_digest": slot_attestation.get("tokenizer_identity_digest"),
            "vocab_size": slot_attestation.get("vocab_size"),
        }
    if len(runtime_receipt_ids) != 1 or len(domain_digests) != 1:
        _fail("independent repeat row coordinate slots do not share one runtime/domain attestation")

    for slot, slot_attestation in stable_vocab_attestation.items():
        if slot_attestation["filtered"] is not False:
            _fail("independent repeat coordinate score is not raw/unfiltered", slot=slot)
        if slot_attestation["model_identity_digest"] != source_identity.get("model_identity_sha256"):
            _fail("independent repeat row model identity is not bound to its receipt", slot=slot)
        if slot_attestation["tokenizer_identity_digest"] != source_identity.get("tokenizer_identity_sha256"):
            _fail("independent repeat row tokenizer identity is not bound to its receipt", slot=slot)
        if slot_attestation["rule_digest"] != plan_identity.get("receipt_content_sha256"):
            _fail("independent repeat row scoring rule is not bound to its plan receipt", slot=slot)

    full_prefix_token_ids_sha256 = row.get("full_prefix_token_ids_sha256")
    if not isinstance(full_prefix_token_ids_sha256, str) or not full_prefix_token_ids_sha256:
        _fail("independent repeat row is missing full_prefix_token_ids_sha256")
    executed_code_sha256 = code_identity.get("executed_source_sha256") or code_identity.get("sha256")
    if not isinstance(executed_code_sha256, str) or not executed_code_sha256:
        _fail("independent repeat receipt is missing executed scoring-code identity")

    return {
        "logical_measurement": {
            "context_id": row.get("context_id"),
            "candidate_id": row.get("candidate_id"),
            "sidecar_id": row.get("sidecar_id"),
            "coord_token_ids": coord_token_ids,
            "coord_token_ids_sha256": row.get("coord_token_ids_sha256"),
            "full_prefix_token_ids_sha256": full_prefix_token_ids_sha256,
            "decision_bearing_channel": row.get("decision_bearing_channel"),
            "native_repetition_penalty_stratum": row.get("native_repetition_penalty_stratum"),
            "primary_role": row.get("primary_role"),
            "excluded_from_primary_ranks": row.get("excluded_from_primary_ranks"),
            "stable_vocab_attestation": stable_vocab_attestation,
        },
        "source_model_plan_scoring": {
            "receipt_schema_version": receipt.get("schema_version"),
            "executed_code_sha256": executed_code_sha256,
            "plan_receipt_content_sha256": plan_identity.get("receipt_content_sha256"),
            "source_identity": source_identity,
            "decision_channel": decision_channel,
            "scoring_backend_admission": backend_admission,
            "environment": environment,
        },
    }


def _independent_process_numerical_repeat_admission(shard_dirs: Sequence[Path]) -> dict[str, Any]:
    if not shard_dirs:
        return {"status": "not_supplied"}
    loaded = [_load_independent_repeat_row(shard_dir) for shard_dir in shard_dirs]
    rows = [row for row, _receipt in loaded]
    expected_repeat_indices = list(range(merge_module.NUMERICAL_REPEAT_COUNT))
    expected_request_ids = [
        f"numerical-repeat:{merge_module.NUMERICAL_REPEAT_CONTEXT_ID}:{repeat_index}"
        for repeat_index in expected_repeat_indices
    ]
    observed_repeat_indices: list[int] = []
    observed_request_ids: list[str] = []
    for row, receipt in loaded:
        if row.get("request_kind") != "numerical_repeat":
            _fail(
                "--independent-repeat-shard-dir row has a non-repeat request_kind",
                observed=row.get("request_kind"),
            )
        repeat_index = row.get("repeat_index")
        if isinstance(repeat_index, bool) or not isinstance(repeat_index, int):
            _fail("independent repeat row.repeat_index is not an integer", observed=repeat_index)
        if repeat_index not in expected_repeat_indices:
            _fail(
                "independent repeat row is outside the frozen repeat_index domain",
                observed=repeat_index,
                expected=expected_repeat_indices,
            )
        request_id = row.get("request_id")
        expected_request_id = expected_request_ids[repeat_index]
        if request_id != expected_request_id:
            _fail(
                "independent repeat row.request_id does not match its frozen repeat ordinal",
                observed=request_id,
                expected=expected_request_id,
            )
        if row.get("context_id") != merge_module.NUMERICAL_REPEAT_CONTEXT_ID:
            _fail(
                "independent repeat row is not bound to the frozen repeat context",
                observed=row.get("context_id"),
                expected=merge_module.NUMERICAL_REPEAT_CONTEXT_ID,
            )

        selection = _mapping(receipt.get("selection"), "independent repeat receipt.selection")
        expected_selection_fields = {
            "included_context_ids": [merge_module.NUMERICAL_REPEAT_CONTEXT_ID],
            "included_request_kinds": ["numerical_repeat"],
            "shard_index": repeat_index,
            "shard_count": merge_module.NUMERICAL_REPEAT_COUNT,
            "filtered_request_count": merge_module.NUMERICAL_REPEAT_COUNT,
            "shard_request_count": 1,
            "shard_request_ids": [request_id],
        }
        for field, expected in expected_selection_fields.items():
            if selection.get(field) != expected:
                _fail(
                    "independent repeat shard selection is not the expected one-of-eight plan shard",
                    field=field,
                    observed=selection.get(field),
                    expected=expected,
                )
        expected_selection_sha256 = sha256_json(
            {key: value for key, value in selection.items() if key != "selection_sha256"}
        )
        if selection.get("selection_sha256") != expected_selection_sha256:
            _fail("independent repeat shard selection_sha256 does not match its content")
        observed_repeat_indices.append(repeat_index)
        observed_request_ids.append(request_id)

    if sorted(observed_repeat_indices) != expected_repeat_indices:
        _fail(
            "--independent-repeat-shard-dir rows do not cover exactly the frozen repeat_index domain; "
            "duplicate, missing, or foreign repeat ordinals are not admissible",
            observed=sorted(observed_repeat_indices),
            expected=expected_repeat_indices,
        )
    if sorted(observed_request_ids) != sorted(expected_request_ids):
        _fail(
            "--independent-repeat-shard-dir rows do not cover exactly the expected repeat request IDs",
            observed=sorted(observed_request_ids),
            expected=sorted(expected_request_ids),
        )

    identities = [
        _independent_repeat_measurement_identity(row, receipt) for row, receipt in loaded
    ]
    logical_measurement_hashes = {
        sha256_json(identity["logical_measurement"]) for identity in identities
    }
    if len(logical_measurement_hashes) != 1:
        _fail(
            "--independent-repeat-shard-dir rows do not share one logical measurement identity "
            "(context/candidate/coordinate tokens/prefix/scoring attestation)",
        )
    source_model_plan_scoring_hashes = {
        sha256_json(identity["source_model_plan_scoring"]) for identity in identities
    }
    if len(source_model_plan_scoring_hashes) != 1:
        _fail(
            "--independent-repeat-shard-dir rows do not share one source/model/plan/scoring identity"
        )

    values = [float(row["raw_model_logprob"]["complete_box_logprob_sum"]) for row in rows]
    median = statistics.median(values)
    delta = max(abs(value - median) for value in values)
    epsilon = max(UNIT_NUMERICAL_EPSILON_FLOOR, 2.0 * delta)
    return {
        "status": "computed",
        "evidence_class": "independent_process",
        "note": (
            "each row was produced by a wholly separate process/model load scoring the identical "
            "logical request; no shared backend, memo, or process state -- unconfounded across every "
            "coordinate slot, unlike the in-process repeat (see memo_reuse_caveat)"
        ),
        "request_ids": expected_request_ids,
        "repeat_indices": expected_repeat_indices,
        "context_id": merge_module.NUMERICAL_REPEAT_CONTEXT_ID,
        "candidate_id": rows[0]["candidate_id"],
        "shard_count": len(rows),
        "values": values,
        "median": median,
        "delta": delta,
        "epsilon": epsilon,
        "epsilon_floor": UNIT_NUMERICAL_EPSILON_FLOOR,
        "formula": "epsilon = max(1e-6, 2 * max_i|r_i - median(r)|)",
    }


# ---------------------------------------------------------------------------
# Adjudication driver
# ---------------------------------------------------------------------------


def adjudicate(
    *,
    plan_dir: str | Path,
    shards: Sequence[merge_module.ShardInput],
    include_context_ids: Sequence[str] | None,
    include_request_kinds: Sequence[str] | None,
    independent_repeat_shard_dirs: Sequence[Path],
    output_dir: str | Path,
    acknowledge_discovery_only_salvage: bool,
    force: bool = False,
) -> dict[str, Any]:
    if not acknowledge_discovery_only_salvage:
        _fail(
            "this is a discovery-only salvage/adjudication path for a known code-identity race, not a "
            "substitute for strict merge; pass --acknowledge-discovery-only-salvage to proceed"
        )
    if not shards:
        _fail("at least one score/receipt shard pair is required")

    plan = score_module.load_plan(plan_dir)
    expected_requests, expected_selection = score_module.select_requests(
        plan, include_context_ids=include_context_ids, include_request_kinds=include_request_kinds
    )
    expected_request_ids = [str(row["request_id"]) for row in expected_requests]
    expected_plan_content_sha256 = str(plan.receipt.get("receipt_content_sha256"))

    validated = [
        _lenient_load_shard(shard, expected_plan_content_sha256=expected_plan_content_sha256) for shard in shards
    ]

    code_identities = [_code_identity_of(shard.receipt) for shard in validated]
    observed_code_sha256 = sorted({str(identity["sha256"]) for identity in code_identities})
    observed_receipt_schema_versions = sorted({str(identity["schema_version"]) for identity in code_identities})
    if len(observed_code_sha256) <= 1 and len(observed_receipt_schema_versions) <= 1:
        _fail(
            "supplied shards are already code-uniform and receipt-schema-current; this discovery-only "
            "salvage path exists specifically to adjudicate the documented heterogeneous-code-hash "
            "race and refuses to run over an already-clean shard set -- use strict "
            "merge_sorted_all_person_route_landscape.py instead",
            observed_code_sha256=observed_code_sha256,
            observed_receipt_schema_versions=observed_receipt_schema_versions,
        )

    # source_identity divergence is a genuinely different, more serious
    # problem than the documented code-hash race (it would mean shards were
    # scored against different checkpoints/tokenizers/configs) and is never
    # tolerated here -- an unknown mismatch, hard failure.
    source_identities = {sha256_json(dict(shard.receipt.get("source_identity") or {})) for shard in validated}
    if len(source_identities) != 1:
        _fail(
            "shards do not share one uniform source_identity (model/tokenizer/infer-config/source-"
            "jsonl); this is not the documented code-hash race and is never tolerated, even in "
            "discovery-only salvage"
        )

    # Reuse the merge module's own strict reconciliation: content
    # disagreement, partial overlap, coverage gaps, and foreign rows remain
    # hard failures identical to strict merge. Only shard-receipt code-hash
    # and receipt-schema-version heterogeneity are tolerated here.
    observed, dedup_accounting = merge_module._reconcile_shards(validated)  # noqa: SLF001
    merge_module._validate_exact_coverage(observed, expected_request_ids=expected_request_ids)  # noqa: SLF001
    rows = [observed[request_id] for request_id in sorted(observed)]
    covered_context_ids = sorted({str(row["context_id"]) for row in rows})

    batch_parity_receipts = merge_module._collect_batch_parity_receipts(validated)  # noqa: SLF001
    observed_batch_scalar_diffs = _observed_batch_scalar_max_abs_diffs(validated)
    numerical_repeat_admission = merge_module._numerical_repeat_admission(rows)  # noqa: SLF001
    independent_admission = _independent_process_numerical_repeat_admission(independent_repeat_shard_dirs)
    residual = merge_module._residual_disclosures(plan, covered_context_ids=covered_context_ids)  # noqa: SLF001
    try:
        token_execution_attestation = attest_module.token_execution_parity_attestation(
            plan, context_ids=covered_context_ids
        )
    except attest_module.RunAttestationError as exc:
        token_execution_attestation = {"status": "failed", "error": str(exc)}

    counts_by_kind: dict[str, int] = {}
    for row in rows:
        kind = str(row["request_kind"])
        counts_by_kind[kind] = counts_by_kind.get(kind, 0) + 1

    output_dir = Path(output_dir).expanduser().resolve()
    scores_path = output_dir / "salvaged-route-landscape-scores.jsonl"
    receipt_path = output_dir / ADJUDICATION_NAME
    scores_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    scores_status = _write_create_or_identical(scores_path, scores_bytes, force=force)

    shard_lineage = [
        {
            "scores": {
                "path": str(shard.scores_path),
                "sha256": sha256_file(shard.scores_path),
                "row_count": len(shard.rows_by_request_id),
            },
            "receipt": {"path": str(shard.receipt_path), "sha256": sha256_file(shard.receipt_path)},
            "context_ids_covered": sorted({str(row["context_id"]) for row in shard.rows_by_request_id.values()}),
            "code_identity": identity,
        }
        for shard, identity in zip(validated, code_identities, strict=True)
    ]

    receipt: dict[str, Any] = {
        "schema_version": ADJUDICATION_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        # --- required disposition fields (never a normal strict-merge pass) ---
        "claim_scope": "descriptive_discovery_only",
        "provenance_disposition": "post_capture_hash_race_adjudicated",
        "uniform_executed_code": "unproven",
        "scalar_batch_admission": "unresolved",
        "causal_confirmation": False,
        "strict_merge_pass": False,
        # --- rationale, bound to observed evidence ---
        "rationale": {
            "summary": (
                "score_sorted_all_person_route_landscape.py built shard receipts' code.sha256 from a "
                "live end-of-job re-read of Path(__file__); these shards' scoring jobs were still "
                "running while that file was concurrently edited (--include-request-id/merge plumbing), "
                "so their recorded code hash does not reliably identify the code that executed the run. "
                "The scorer has since been fixed (receipt schema v2: an import-time "
                "executed_source_sha256 fingerprint). This adjudication preserves and lists every "
                "observed heterogeneous value rather than erasing or averaging over the mismatch."
            ),
            "observed_code_sha256": observed_code_sha256,
            "observed_receipt_schema_versions": observed_receipt_schema_versions,
            "memo_reuse_caveat": MEMO_REUSE_CAVEAT,
            "batch_scalar_tolerance_note": (
                "the scorer's own batch-admission gate uses BATCH_REFORWARD_COORD_LOGPROB_MAX_ABS_DIFF "
                "= 1e-3; this unit's own frozen numerical-repeat epsilon is typically 1e-6. A passed "
                "scorer batch gate is not proof a batched row meets the unit's tighter standard."
            ),
        },
        "plan": {
            "plan_dir": str(plan.plan_dir),
            "receipt_path": str(plan.receipt_path),
            "receipt_content_sha256": expected_plan_content_sha256,
        },
        "selection": expected_selection,
        "source_identity_sha256": next(iter(source_identities)),
        "dedup_accounting": dedup_accounting,
        "shards": shard_lineage,
        "covered_context_ids": covered_context_ids,
        "counts": {"rows": len(rows), "rows_by_request_kind": counts_by_kind},
        "batch_parity_receipts": batch_parity_receipts,
        "observed_batch_scalar_max_abs_diffs": {
            "count": len(observed_batch_scalar_diffs),
            "min": min(observed_batch_scalar_diffs) if observed_batch_scalar_diffs else None,
            "max": max(observed_batch_scalar_diffs) if observed_batch_scalar_diffs else None,
            "unit_numerical_epsilon_floor": UNIT_NUMERICAL_EPSILON_FLOOR,
            "exceeds_unit_epsilon": (
                max(observed_batch_scalar_diffs) > UNIT_NUMERICAL_EPSILON_FLOOR
                if observed_batch_scalar_diffs
                else None
            ),
        },
        "numerical_repeat_admission": numerical_repeat_admission,
        "independent_process_numerical_repeat_admission": independent_admission,
        "residual_disclosures": residual,
        "token_execution_parity_attestation": token_execution_attestation,
    }
    receipt["output_artifacts"] = {
        "salvaged_scores": {"path": str(scores_path), "sha256": sha256_file(scores_path), "row_count": len(rows)}
    }
    receipt["receipt_content_sha256"] = sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    receipt_bytes = canonical_json_bytes(receipt) + b"\n"
    receipt_status = _write_create_or_identical(receipt_path, receipt_bytes, force=force)

    return {
        **receipt,
        "output_artifacts": {
            **receipt["output_artifacts"],
            "salvaged_scores": {**receipt["output_artifacts"]["salvaged_scores"], "status": scores_status},
            "adjudication_receipt": {"path": str(receipt_path), "status": receipt_status},
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument(
        "--shard",
        action="append",
        nargs=2,
        metavar=("SCORES", "RECEIPT"),
        default=[],
        help="repeatable score JSONL and its (possibly pre-fix v1) shard receipt; at least one is required",
    )
    parser.add_argument(
        "--shard-dir",
        action="append",
        default=[],
        type=Path,
        help=f"repeatable directory containing {score_module.OUTPUT_JSONL_NAME!r} and {score_module.OUTPUT_RECEIPT_NAME!r}",
    )
    parser.add_argument(
        "--independent-repeat-shard-dir",
        action="append",
        default=[],
        type=Path,
        help=(
            "repeatable directory (one independent process/model load each) each containing exactly "
            "one score row of the identical logical repeated request; auxiliary evidence, never part "
            "of exact plan coverage"
        ),
    )
    parser.add_argument("--include-context-id", action="append", default=None)
    parser.add_argument("--include-request-kind", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--acknowledge-discovery-only-salvage",
        action="store_true",
        help="required opt-in: this is a discovery-only adjudication, never a substitute for strict merge",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def _resolve_shards(args: argparse.Namespace) -> list[merge_module.ShardInput]:
    shards = [
        merge_module.ShardInput(scores_path=Path(scores), receipt_path=Path(receipt))
        for scores, receipt in args.shard
    ]
    for shard_dir in args.shard_dir:
        shard_dir = Path(shard_dir)
        shards.append(
            merge_module.ShardInput(
                scores_path=shard_dir / score_module.OUTPUT_JSONL_NAME,
                receipt_path=shard_dir / score_module.OUTPUT_RECEIPT_NAME,
            )
        )
    return shards


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shards = _resolve_shards(args)
    receipt = adjudicate(
        plan_dir=args.plan_dir,
        shards=shards,
        include_context_ids=args.include_context_id,
        include_request_kinds=args.include_request_kind,
        independent_repeat_shard_dirs=list(args.independent_repeat_shard_dir),
        output_dir=args.output_dir,
        acknowledge_discovery_only_salvage=args.acknowledge_discovery_only_salvage,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "claim_scope": receipt["claim_scope"],
                "provenance_disposition": receipt["provenance_disposition"],
                "counts": receipt["counts"],
                "output_artifacts": receipt["output_artifacts"],
            },
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
