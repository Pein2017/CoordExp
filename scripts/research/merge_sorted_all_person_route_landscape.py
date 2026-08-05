#!/usr/bin/env python3
"""Merge raw pre-penalty score shards for the sorted all-person route
landscape unit (``2026-08-03-sorted-all-person-owner-relative-route-
landscape``).

This is a CPU-only artifact merger. It never loads a model, tokenizer, or
GPU; it only joins an arbitrary, non-fixed number of per-run raw score
shards -- each produced by one invocation of
``score_sorted_all_person_route_landscape.py`` -- against the frozen plan
(``build_sorted_all_person_route_landscape.py`` output) those shards were
scored against.

Every shard must bind the exact same plan (``plan.receipt_content_sha256``);
shards binding a different or foreign plan are refused outright. Two shards
may legitimately cover the same request only when their rows are
byte-identical (canonical JSON); any other overlap (partial coverage, or
differing content for the same request) is a hard failure. The merged
surface must exactly equal the plan's own scoring-requests domain, filtered
by the same ``--include-context-id``/``--include-request-kind`` selection
this merge was asked to cover -- no missing request, no duplicate, no
foreign request id.

This merger also republishes, unmodified, the per-context batch admission
(scalar-vs-batch coordinate parity) each shard recorded, and independently
recomputes the frozen numerical-repeat tolerance
(``epsilon = max(1e-6, 2 * max_i|r_i - median(r)|)``) from any merged
``numerical_repeat`` rows -- never trusting a shard's self-reported epsilon,
because no shard computes one.

Raw-capture validity (every declared request scored, digests bind, backend
identity uniform) is a structural/mechanical property this merger *can*
establish CPU-only. Teacher-forced chosen-token prefix parity is not: the
plan's own ``contexts.jsonl`` rows declare
``teacher_forced_chosen_token_parity.status = "required_not_cpu_verifiable"``
and ``claimed_pass = false`` for every non-root context. This merger
republishes that residual disclosure verbatim per covered context and never
promotes it to "passed" -- raw capture succeeding is not evidence that the
donor-derived prefixes were exactly teacher-forced.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import sys
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_all_person_route_landscape as planner  # noqa: E402
from scripts.research import score_sorted_all_person_route_landscape as score_module  # noqa: E402

MERGE_SCHEMA_VERSION = "sorted-all-person-route-landscape-merge.v1"
UNIT_ID = planner.UNIT_ID
SCORE_ROW_SCHEMA_VERSION = score_module.SCHEMA_VERSION
SCORE_RECEIPT_SCHEMA_VERSION = score_module.RECEIPT_SCHEMA_VERSION

MERGED_SCORES_NAME = "merged-route-landscape-scores.jsonl"
MERGED_RECEIPT_NAME = "merged-route-landscape-receipt.json"

#: Frozen numerical-repeat tolerance formula (unit.md "Numerical tolerance
#: and ranks"): a floor of 1e-6, or twice the observed max deviation from
#: the repeat median, whichever is larger.
EPSILON_FLOOR = 1e-6
NUMERICAL_REPEAT_CONTEXT_ID = "self-due-gt17"
NUMERICAL_REPEAT_COUNT = 8

canonical_json_bytes = score_module.canonical_json_bytes
sha256_json = score_module.sha256_json
sha256_file = score_module.sha256_file
_read_json = score_module._read_json  # noqa: SLF001
_read_jsonl = score_module._read_jsonl  # noqa: SLF001
_write_create_or_identical = score_module._write_create_or_identical  # noqa: SLF001


class MergeContractError(RuntimeError):
    """Raised before/while merging when a shard or partition invariant fails."""

    def __init__(self, message: str, **context: Any) -> None:
        if context:
            message = f"{message} | context: {json.dumps(context, sort_keys=True, default=str)}"
        super().__init__(message)


def _fail(message: str, **context: Any) -> NoReturn:
    raise MergeContractError(message, **context)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a JSON object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(f"{label} must be a JSON array")
    return value


# ---------------------------------------------------------------------------
# Shard loading + validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardInput:
    scores_path: Path
    receipt_path: Path


@dataclass(frozen=True)
class ValidatedShard:
    scores_path: Path
    receipt_path: Path
    receipt: Mapping[str, Any]
    rows_by_request_id: dict[str, Mapping[str, Any]]


def _validate_shard(shard: ShardInput, *, expected_plan_content_sha256: str) -> ValidatedShard:
    scores_path = shard.scores_path.expanduser().resolve(strict=True)
    receipt_path = shard.receipt_path.expanduser().resolve(strict=True)
    receipt = _read_json(receipt_path, f"shard receipt {receipt_path}")

    if receipt.get("schema_version") != SCORE_RECEIPT_SCHEMA_VERSION:
        _fail(
            "shard receipt.schema_version is not this unit's own successor score-receipt schema",
            receipt_path=str(receipt_path),
            observed=receipt.get("schema_version"),
            expected=SCORE_RECEIPT_SCHEMA_VERSION,
        )
    if receipt.get("unit_id") != UNIT_ID:
        _fail(
            "shard receipt.unit_id is not this unit's own unit_id",
            receipt_path=str(receipt_path),
            observed=receipt.get("unit_id"),
        )
    if receipt.get("row_schema_version") != SCORE_ROW_SCHEMA_VERSION:
        _fail(
            "shard receipt.row_schema_version is not this scorer's own row schema",
            receipt_path=str(receipt_path),
        )

    plan_entry = _mapping(receipt.get("plan"), f"shard receipt {receipt_path}.plan")
    plan_content_sha256 = plan_entry.get("receipt_content_sha256")
    if plan_content_sha256 != expected_plan_content_sha256:
        _fail(
            "shard was scored against a different plan (plan.receipt_content_sha256 mismatch); "
            "refusing to merge shards bound to different plans",
            receipt_path=str(receipt_path),
            observed=plan_content_sha256,
            expected=expected_plan_content_sha256,
        )

    row_ids = _sequence(receipt.get("row_ids"), f"shard receipt {receipt_path}.row_ids")
    declared_row_ids = sorted(str(v) for v in row_ids)
    if len(declared_row_ids) != len(set(declared_row_ids)):
        _fail("shard receipt.row_ids contains duplicates", receipt_path=str(receipt_path))

    rows = _read_jsonl(scores_path, f"shard scores {scores_path}")
    rows_by_request_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("schema_version") != SCORE_ROW_SCHEMA_VERSION:
            _fail("shard score row does not carry this scorer's own row schema_version", scores_path=str(scores_path))
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
            "shard scores file content does not match the exact request-id set the shard receipt declared",
            scores_path=str(scores_path),
            receipt_path=str(receipt_path),
        )
    counts = _mapping(receipt.get("counts"), f"shard receipt {receipt_path}.counts")
    if counts.get("rows") != len(rows_by_request_id):
        _fail(
            "shard receipt.counts.rows does not match the actual row count on disk",
            receipt_path=str(receipt_path),
        )

    return ValidatedShard(
        scores_path=scores_path,
        receipt_path=receipt_path,
        receipt=receipt,
        rows_by_request_id=rows_by_request_id,
    )


# ---------------------------------------------------------------------------
# Reconciliation: exact partition, no duplicate/missing IDs
# ---------------------------------------------------------------------------


def _reconcile_shards(
    validated: Sequence[ValidatedShard],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    observed: dict[str, Mapping[str, Any]] = {}
    per_shard_ids: list[set[str]] = [set(shard.rows_by_request_id) for shard in validated]

    for shard_index, shard in enumerate(validated):
        for request_id, row in shard.rows_by_request_id.items():
            existing = observed.get(request_id)
            if existing is None:
                observed[request_id] = row
                continue
            if canonical_json_bytes(existing) != canonical_json_bytes(row):
                _fail(
                    "shards disagree on an identical request_id: overlapping shards must be "
                    "disjoint or byte-identical for every shared request",
                    request_id=request_id,
                )

    for a_index in range(len(validated)):
        for b_index in range(a_index + 1, len(validated)):
            shared = per_shard_ids[a_index] & per_shard_ids[b_index]
            if not shared:
                continue
            disagreeing = [
                request_id
                for request_id in shared
                if canonical_json_bytes(validated[a_index].rows_by_request_id[request_id])
                != canonical_json_bytes(validated[b_index].rows_by_request_id[request_id])
            ]
            if disagreeing:
                _fail(
                    "overlapping shards disagree on shared request_id rows",
                    shard_a=str(validated[a_index].receipt_path),
                    shard_b=str(validated[b_index].receipt_path),
                    disagreeing_request_ids=sorted(disagreeing)[:8],
                )

    dedup_accounting = {
        "shard_count": len(validated),
        "union_request_count": len(observed),
        "per_shard_request_counts": [len(ids) for ids in per_shard_ids],
    }
    return observed, dedup_accounting


def _validate_exact_coverage(
    observed: Mapping[str, Mapping[str, Any]], *, expected_request_ids: Sequence[str]
) -> None:
    expected = set(expected_request_ids)
    observed_ids = set(observed)
    missing = sorted(expected - observed_ids)
    extra = sorted(observed_ids - expected)
    if missing or extra:
        _fail(
            "merged score rows do not exactly partition the expected scoring-request domain for the "
            "requested context(s)/kind(s); no gap or foreign row is admitted",
            missing_count=len(missing),
            missing_sample=missing[:8],
            extra_count=len(extra),
            extra_sample=extra[:8],
        )


# ---------------------------------------------------------------------------
# Batch parity + numerical-repeat epsilon
# ---------------------------------------------------------------------------


def _collect_batch_parity_receipts(validated: Sequence[ValidatedShard]) -> list[dict[str, Any]]:
    """Republish every shard's per-context batch admission unmodified.

    A ``status == "failed_scalar_fallback_required"`` admission is only ever
    accepted here if the shard also recorded an explicit ``fallback`` key
    (``score_sorted_all_person_route_landscape.run``'s
    ``--on-batch-parity-failure=scalar_fallback`` path) -- a bare failure
    with no recorded fallback means that scorer run should itself have
    aborted before writing a shard, so seeing one here is treated as a
    contract violation, not silently accepted.
    """

    entries: list[dict[str, Any]] = []
    for shard in validated:
        admission = _mapping(
            shard.receipt.get("scoring_backend_admission"), f"shard {shard.receipt_path}.scoring_backend_admission"
        )
        per_context = _sequence(
            admission.get("per_context_accounting"),
            f"shard {shard.receipt_path}.scoring_backend_admission.per_context_accounting",
        )
        for entry in per_context:
            entry = _mapping(entry, "per_context_accounting entry")
            batch_admission = entry.get("batch_admission")
            if isinstance(batch_admission, Mapping):
                status = batch_admission.get("status")
                if status not in ("passed", "not_requested") and "fallback" not in batch_admission:
                    _fail(
                        "shard recorded a failed batch-parity admission with no explicit recorded fallback; "
                        "a bare failed admission must never reach a sealed shard",
                        receipt_path=str(shard.receipt_path),
                        context_id=entry.get("context_id"),
                        batch_admission=dict(batch_admission),
                    )
            entries.append(
                {
                    "shard_receipt_path": str(shard.receipt_path),
                    "context_id": entry.get("context_id"),
                    "batch_admission": batch_admission,
                    "batchable_row_count": entry.get("batchable_row_count"),
                    "scalar_only_row_count": entry.get("scalar_only_row_count"),
                }
            )
    return entries


def _numerical_repeat_admission(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Independently recompute the frozen repeat-derived epsilon; never trust a shard's claim."""

    repeats = [row for row in rows if row.get("request_kind") == "numerical_repeat"]
    if not repeats:
        return {"status": "not_included_in_this_merge"}

    context_ids = {str(row.get("context_id")) for row in repeats}
    candidate_ids = {str(row.get("candidate_id")) for row in repeats}
    if context_ids != {NUMERICAL_REPEAT_CONTEXT_ID}:
        _fail(
            "merged numerical_repeat rows are not all bound to the frozen repeat context",
            observed=sorted(context_ids),
            expected=NUMERICAL_REPEAT_CONTEXT_ID,
        )
    if len(candidate_ids) != 1:
        _fail("merged numerical_repeat rows do not all repeat the same candidate", observed=sorted(candidate_ids))

    repeat_indices = sorted(int(row["repeat_index"]) for row in repeats)
    if repeat_indices != list(range(NUMERICAL_REPEAT_COUNT)):
        _fail(
            "merged numerical_repeat rows do not cover exactly the frozen repeat_index domain",
            observed=repeat_indices,
            expected=list(range(NUMERICAL_REPEAT_COUNT)),
        )

    values = [float(row["raw_model_logprob"]["complete_box_logprob_sum"]) for row in repeats]
    median = statistics.median(values)
    delta = max(abs(value - median) for value in values)
    epsilon = max(EPSILON_FLOOR, 2.0 * delta)
    return {
        "status": "computed",
        "context_id": NUMERICAL_REPEAT_CONTEXT_ID,
        "candidate_id": next(iter(candidate_ids)),
        "repeat_count": len(repeats),
        "values": values,
        "median": median,
        "delta": delta,
        "epsilon": epsilon,
        "epsilon_floor": EPSILON_FLOOR,
        "formula": "epsilon = max(1e-6, 2 * max_i|r_i - median(r)|)",
    }


def _residual_disclosures(plan: score_module.PlanBundle, *, covered_context_ids: Sequence[str]) -> dict[str, Any]:
    per_context: dict[str, Any] = {}
    for context_id in sorted(set(covered_context_ids)):
        context = plan.contexts_by_id.get(context_id)
        if context is None:
            continue
        parity = context.get("teacher_forced_chosen_token_parity")
        per_context[context_id] = dict(parity) if isinstance(parity, Mapping) else parity
    return {
        "status": "unverified_by_this_merge",
        "note": (
            "raw-capture validity (every declared request scored, digests bind, backend identity "
            "uniform) is a mechanical property this CPU-only merge establishes directly. It is NOT "
            "evidence that the donor-derived generated prefixes were exactly teacher-forced with "
            "chosen-token parity -- that check is declared not_cpu_verifiable by the plan itself and "
            "remains outstanding for every covered non-root context."
        ),
        "per_context_teacher_forced_chosen_token_parity": per_context,
    }


# ---------------------------------------------------------------------------
# Merge driver
# ---------------------------------------------------------------------------


def merge_shards(
    *,
    plan_dir: str | Path,
    shards: Sequence[ShardInput],
    include_context_ids: Sequence[str] | None,
    include_request_kinds: Sequence[str] | None,
    output_dir: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    if not shards:
        _fail("at least one score/receipt shard pair is required")

    plan = score_module.load_plan(plan_dir)
    expected_requests, expected_selection = score_module.select_requests(
        plan,
        include_context_ids=include_context_ids,
        include_request_kinds=include_request_kinds,
    )
    expected_request_ids = [str(row["request_id"]) for row in expected_requests]
    expected_plan_content_sha256 = str(plan.receipt.get("receipt_content_sha256"))

    validated = [
        _validate_shard(shard, expected_plan_content_sha256=expected_plan_content_sha256) for shard in shards
    ]

    observed, dedup_accounting = _reconcile_shards(validated)
    _validate_exact_coverage(observed, expected_request_ids=expected_request_ids)

    rows = [observed[request_id] for request_id in sorted(observed)]
    covered_context_ids = sorted({str(row["context_id"]) for row in rows})

    counts_by_kind: dict[str, int] = {}
    for row in rows:
        kind = str(row["request_kind"])
        counts_by_kind[kind] = counts_by_kind.get(kind, 0) + 1

    source_identities = {sha256_json(dict(shard.receipt.get("source_identity") or {})) for shard in validated}
    if len(source_identities) != 1:
        _fail(
            "merged shards do not share one uniform source_identity (model/tokenizer/infer-config/"
            "source-jsonl); refusing to merge shards scored under different runtime identities"
        )
    code_identities = {
        _mapping(shard.receipt.get("code"), "shard receipt.code").get("sha256") for shard in validated
    }
    if len(code_identities) != 1:
        _fail(
            "merged shards were produced by different scorer code (code.sha256 differs); refusing to "
            "merge shards that may embody different scoring logic"
        )

    batch_parity_receipts = _collect_batch_parity_receipts(validated)
    numerical_repeat_admission = _numerical_repeat_admission(rows)
    residual = _residual_disclosures(plan, covered_context_ids=covered_context_ids)

    output_dir = Path(output_dir).expanduser().resolve()
    scores_path = output_dir / MERGED_SCORES_NAME
    receipt_path = output_dir / MERGED_RECEIPT_NAME
    scores_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    scores_status = _write_create_or_identical(scores_path, scores_bytes, force=force)

    shard_lineage = [
        {
            "scores": {"path": str(shard.scores_path), "sha256": sha256_file(shard.scores_path), "row_count": len(shard.rows_by_request_id)},
            "receipt": {"path": str(shard.receipt_path), "sha256": sha256_file(shard.receipt_path)},
            "context_ids_covered": sorted({str(row["context_id"]) for row in shard.rows_by_request_id.values()}),
            "code_sha256": _mapping(shard.receipt.get("code"), "shard.code").get("sha256"),
        }
        for shard in validated
    ]

    receipt: dict[str, Any] = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "plan": {
            "plan_dir": str(plan.plan_dir),
            "receipt_path": str(plan.receipt_path),
            "receipt_sha256": sha256_file(plan.receipt_path),
            "receipt_content_sha256": expected_plan_content_sha256,
        },
        "selection": expected_selection,
        "decision_channel": {
            "name": score_module.DECISION_BEARING_CHANNEL,
            "native_repetition_penalty_stratum": score_module.NATIVE_REPETITION_PENALTY_STRATUM,
        },
        "source_identity_sha256": next(iter(source_identities)),
        "scorer_code_sha256": next(iter(code_identities)),
        "dedup_accounting": dedup_accounting,
        "shards": shard_lineage,
        "covered_context_ids": covered_context_ids,
        "counts": {
            "rows": len(rows),
            "rows_by_request_kind": counts_by_kind,
            "primary_role_rows": sum(1 for row in rows if row.get("primary_role")),
            "excluded_from_primary_ranks_rows": sum(1 for row in rows if row.get("excluded_from_primary_ranks")),
        },
        "batch_parity_receipts": batch_parity_receipts,
        "numerical_repeat_admission": numerical_repeat_admission,
        "raw_capture_validity": {
            "status": "passed",
            "note": (
                "every request in the requested selection was scored exactly once, every shard's "
                "declared row set matches its on-disk content, and every shard shares one uniform "
                "runtime/code identity; this is a mechanical, non-scientific disposition"
            ),
        },
        "residual_disclosures": residual,
    }
    receipt["output_artifacts"] = {
        "merged_scores": {"path": str(scores_path), "sha256": sha256_file(scores_path), "row_count": len(rows)}
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
            "merged_scores": {**receipt["output_artifacts"]["merged_scores"], "status": scores_status},
            "merged_receipt": {"path": str(receipt_path), "status": receipt_status},
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
        help="repeatable score JSONL and its sealed scorer receipt; at least one is required",
    )
    parser.add_argument(
        "--shard-dir",
        action="append",
        default=[],
        type=Path,
        help=(
            f"repeatable directory containing {score_module.OUTPUT_JSONL_NAME!r} and "
            f"{score_module.OUTPUT_RECEIPT_NAME!r} (one score_sorted_all_person_route_landscape.py "
            "--output-dir)"
        ),
    )
    parser.add_argument("--include-context-id", action="append", default=None)
    parser.add_argument("--include-request-kind", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def _resolve_shards(args: argparse.Namespace) -> list[ShardInput]:
    shards = [ShardInput(scores_path=Path(scores), receipt_path=Path(receipt)) for scores, receipt in args.shard]
    for shard_dir in args.shard_dir:
        shard_dir = Path(shard_dir)
        shards.append(
            ShardInput(
                scores_path=shard_dir / score_module.OUTPUT_JSONL_NAME,
                receipt_path=shard_dir / score_module.OUTPUT_RECEIPT_NAME,
            )
        )
    return shards


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shards = _resolve_shards(args)
    receipt = merge_shards(
        plan_dir=args.plan_dir,
        shards=shards,
        include_context_ids=args.include_context_id,
        include_request_kinds=args.include_request_kind,
        output_dir=args.output_dir,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "counts": receipt["counts"],
                "covered_context_ids": receipt["covered_context_ids"],
                "output_artifacts": receipt["output_artifacts"],
            },
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
