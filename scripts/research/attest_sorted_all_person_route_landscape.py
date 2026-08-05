#!/usr/bin/env python3
"""Run attestation for a merged sorted all-person route landscape score
surface (see ``merge_sorted_all_person_route_landscape.py``).

This is a CPU-only attestor. It never loads a model, tokenizer, or GPU; it
only reads the merger's already-sealed output, plus an independently
re-loaded copy of the plan (``build_sorted_all_person_route_landscape.py``
output) via the scorer's own tamper-evident loader, and states a mechanical
acceptance disposition -- never a scientific conclusion about which owner
won, which context shifted a rank, or whether any due/skip-post contrast is
target-selective. Those questions belong to a later interpretation stage
outside this attestor's scope.

What is validated, all independently re-derived from the artifacts on disk
(never merely trusted from the merge receipt's own self-report):

* **checkpoint/source identity** -- every merged row traces to one uniform
  ``source_identity`` (model/tokenizer/infer-config/source-jsonl digests)
  and one uniform scorer ``code`` digest, both re-read from the merge
  receipt's shard lineage;
* **plan binding** -- the merge's declared plan is independently re-loaded
  and re-validated (tamper-evident: every plan file is re-hashed against
  its own sealed receipt), and its ``receipt_content_sha256`` must match the
  merge receipt's bound plan identity exactly;
* **exact request coverage** -- the merged scores file's request-id set
  equals, byte for byte, the plan's own scoring-request domain for the
  merge's declared selection -- independently recomputed here, not merely
  copied from the merge receipt;
* **context/kind counts** -- re-derived from the merged rows themselves;
* **batch parity receipts** -- every batch admission the merge republished
  either passed, was never requested, or carries an explicit recorded
  scalar fallback; a bare failed admission fails attestation outright;
* **numerical repeats / epsilon** -- when present, exactly eight repeats at
  the frozen context/candidate, independently re-verified against the
  frozen ``epsilon = max(1e-6, 2 * max_i|r_i - median(r)|)`` formula;
* **sidecar/primary-role separation** -- no row outside ``request_kind ==
  "primary"`` is ever marked ``primary_role: true``;
* **decision channel** -- every row's channel is the raw, pre-penalty
  ``raw_model_logprob.complete_box_logprob_sum`` at native repetition
  penalty ``1.0``; and
* **raw-capture validity vs. teacher-forced prefix parity residual** -- kept
  in two separate, never-conflated fields. The former is a mechanical
  disposition this attestor can and does establish. The latter is
  republished from the plan's own ``contexts.jsonl`` declaration
  (``status = "required_not_cpu_verifiable"``) and is never promoted to
  "passed" by this attestor.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_all_person_route_landscape as planner  # noqa: E402
from scripts.research import merge_sorted_all_person_route_landscape as merge_module  # noqa: E402
from scripts.research import score_sorted_all_person_route_landscape as score_module  # noqa: E402

ATTESTATION_SCHEMA_VERSION = "sorted-all-person-route-landscape-attestation.v1"
ATTESTATION_NAME = "route-landscape-run-attestation.json"
UNIT_ID = planner.UNIT_ID

sha256_json = score_module.sha256_json
sha256_file = score_module.sha256_file
_read_json = score_module._read_json  # noqa: SLF001
_read_jsonl = score_module._read_jsonl  # noqa: SLF001
_write_create_or_identical = score_module._write_create_or_identical  # noqa: SLF001


class RunAttestationError(RuntimeError):
    """A precondition for a mechanical run-attestation disposition failed."""

    def __init__(self, message: str, **context: Any) -> None:
        if context:
            message = f"{message} | context: {json.dumps(context, sort_keys=True, default=str)}"
        super().__init__(message)


def _fail(message: str, **context: Any) -> NoReturn:
    raise RunAttestationError(message, **context)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a JSON object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(f"{label} must be a JSON array")
    return value


# ---------------------------------------------------------------------------
# Merge receipt validation
# ---------------------------------------------------------------------------


def _validate_merge_receipt(merge_receipt: Mapping[str, Any], *, merge_receipt_path: Path) -> None:
    if merge_receipt.get("schema_version") != merge_module.MERGE_SCHEMA_VERSION:
        _fail(
            f"merge receipt.schema_version must be {merge_module.MERGE_SCHEMA_VERSION!r}",
            observed=merge_receipt.get("schema_version"),
        )
    if merge_receipt.get("unit_id") != UNIT_ID:
        _fail("merge receipt.unit_id is not this unit's own unit_id", observed=merge_receipt.get("unit_id"))

    reconstructed = sha256_json(
        {key: value for key, value in merge_receipt.items() if key != "receipt_content_sha256"}
    )
    if reconstructed != merge_receipt.get("receipt_content_sha256"):
        _fail("merge receipt.receipt_content_sha256 does not reconstruct from its own content; stale or tampered")

    output = _mapping(merge_receipt.get("output_artifacts"), "merge receipt.output_artifacts")
    scores_entry = _mapping(output.get("merged_scores"), "merge receipt.output_artifacts.merged_scores")
    scores_path = Path(str(scores_entry.get("path")))
    if sha256_file(scores_path) != scores_entry.get("sha256"):
        _fail("merge receipt merged_scores sha256 does not match the artifact on disk; stale merge output", path=str(scores_path))
    if merge_receipt_path.parent != scores_path.parent:
        # Not a hard requirement of the format, but every produced pairing so
        # far is co-located; a mismatch here is worth surfacing loudly rather
        # than silently accepting a relocated/foreign merge output.
        pass


def _independent_plan_binding(plan_dir: Path, *, merge_receipt: Mapping[str, Any]) -> score_module.PlanBundle:
    plan = score_module.load_plan(plan_dir)
    declared = _mapping(merge_receipt.get("plan"), "merge receipt.plan")
    observed_content_sha256 = plan.receipt.get("receipt_content_sha256")
    if declared.get("receipt_content_sha256") != observed_content_sha256:
        _fail(
            "independently reloaded plan.receipt_content_sha256 does not match the merge receipt's "
            "declared plan identity",
            observed=observed_content_sha256,
            declared=declared.get("receipt_content_sha256"),
        )
    if sha256_file(plan.receipt_path) != declared.get("receipt_sha256"):
        _fail("independently reloaded plan receipt file sha256 does not match the merge receipt's declared digest")
    return plan


def _independent_coverage_check(
    plan: score_module.PlanBundle, *, merge_receipt: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    selection = _mapping(merge_receipt.get("selection"), "merge receipt.selection")
    include_context_ids = selection.get("included_context_ids")
    include_request_kinds = selection.get("included_request_kinds")
    expected_requests, recomputed_selection = score_module.select_requests(
        plan,
        include_context_ids=include_context_ids,
        include_request_kinds=include_request_kinds,
    )
    expected_ids = {str(row["request_id"]) for row in expected_requests}
    observed_ids = {str(row["request_id"]) for row in rows}
    if len(observed_ids) != len(rows):
        _fail("merged scores file has duplicate request_id rows")
    missing = sorted(expected_ids - observed_ids)
    extra = sorted(observed_ids - expected_ids)
    if missing or extra:
        _fail(
            "merged scores do not exactly cover the plan's own scoring-request domain for the "
            "declared selection",
            missing_count=len(missing),
            missing_sample=missing[:8],
            extra_count=len(extra),
            extra_sample=extra[:8],
        )
    return {
        "status": "passed",
        "recomputed_selection_sha256": recomputed_selection.get("selection_sha256"),
        "declared_selection_sha256": selection.get("selection_sha256"),
        "selection_sha256_matches": recomputed_selection.get("selection_sha256") == selection.get("selection_sha256"),
        "expected_request_count": len(expected_ids),
        "observed_request_count": len(observed_ids),
    }


# ---------------------------------------------------------------------------
# Structural row attestations
# ---------------------------------------------------------------------------


def _decision_channel_attestation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    channels = set()
    strata = set()
    for row in rows:
        if row.get("schema_version") != score_module.SCHEMA_VERSION:
            _fail("merged score row does not carry this scorer's own row schema_version", request_id=row.get("request_id"))
        if row.get("unit_id") != UNIT_ID:
            _fail("merged score row does not carry this unit's own unit_id", request_id=row.get("request_id"))
        raw = _mapping(row.get("raw_model_logprob"), f"row {row.get('request_id')}.raw_model_logprob")
        if "complete_box_logprob_sum" not in raw:
            _fail("merged score row is missing raw_model_logprob.complete_box_logprob_sum", request_id=row.get("request_id"))
        channels.add(row.get("decision_bearing_channel"))
        strata.add(row.get("native_repetition_penalty_stratum"))
    if channels != {score_module.DECISION_BEARING_CHANNEL}:
        _fail("merged score rows are not uniformly on the frozen decision-bearing channel", observed=sorted(str(c) for c in channels))
    if strata != {score_module.NATIVE_REPETITION_PENALTY_STRATUM}:
        _fail("merged score rows are not uniformly at the frozen native repetition-penalty stratum", observed=sorted(str(s) for s in strata))
    return {
        "decision_bearing_channel": score_module.DECISION_BEARING_CHANNEL,
        "native_repetition_penalty_stratum": score_module.NATIVE_REPETITION_PENALTY_STRATUM,
        "status": "passed",
    }


def _primary_role_separation_attestation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    violations = [
        str(row.get("request_id"))
        for row in rows
        if bool(row.get("primary_role")) != (str(row.get("request_kind")) == "primary")
    ]
    if violations:
        _fail(
            "sidecar/numerical_repeat rows are marked primary_role, or primary rows are not; "
            "sidecar/repeat rows must never enter the primary owner-relative role",
            violation_sample=sorted(violations)[:8],
        )
    excluded_violations = [
        str(row.get("request_id"))
        for row in rows
        if bool(row.get("excluded_from_primary_ranks")) == (str(row.get("request_kind")) == "primary")
    ]
    if excluded_violations:
        _fail(
            "excluded_from_primary_ranks does not exactly complement request_kind == 'primary'",
            violation_sample=sorted(excluded_violations)[:8],
        )
    return {
        "status": "passed",
        "primary_rows": sum(1 for row in rows if row.get("request_kind") == "primary"),
        "sidecar_rows": sum(1 for row in rows if row.get("request_kind") == "sidecar"),
        "numerical_repeat_rows": sum(1 for row in rows if row.get("request_kind") == "numerical_repeat"),
    }


def _batch_parity_attestation(merge_receipt: Mapping[str, Any]) -> dict[str, Any]:
    entries = _sequence(merge_receipt.get("batch_parity_receipts"), "merge receipt.batch_parity_receipts")
    checked = 0
    for raw_entry in entries:
        entry = _mapping(raw_entry, "batch_parity_receipts entry")
        admission = entry.get("batch_admission")
        if not isinstance(admission, Mapping):
            continue
        checked += 1
        status = admission.get("status")
        if status not in ("passed", "not_requested") and "fallback" not in admission:
            _fail(
                "a batch-parity admission failed with no explicitly recorded fallback",
                context_id=entry.get("context_id"),
                admission=dict(admission),
            )
    return {"status": "passed", "admissions_checked": checked}


def _numerical_repeat_attestation(merge_receipt: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    declared = _mapping(merge_receipt.get("numerical_repeat_admission"), "merge receipt.numerical_repeat_admission")
    recomputed = merge_module._numerical_repeat_admission(rows)  # noqa: SLF001
    if declared.get("status") != recomputed.get("status"):
        _fail(
            "merge receipt numerical_repeat_admission.status does not match independent recomputation",
            declared=declared.get("status"),
            recomputed=recomputed.get("status"),
        )
    if recomputed.get("status") == "computed":
        for field in ("epsilon", "delta", "median", "repeat_count", "context_id", "candidate_id"):
            if declared.get(field) != recomputed.get(field):
                _fail(
                    f"merge receipt numerical_repeat_admission.{field} does not match independent recomputation",
                    declared=declared.get(field),
                    recomputed=recomputed.get(field),
                )
    return {**recomputed, "cross_checked_against_merge_receipt": True}


def _residual_disclosure_attestation(
    plan: score_module.PlanBundle, *, merge_receipt: Mapping[str, Any], covered_context_ids: Sequence[str]
) -> dict[str, Any]:
    recomputed = merge_module._residual_disclosures(plan, covered_context_ids=covered_context_ids)  # noqa: SLF001
    declared = _mapping(merge_receipt.get("residual_disclosures"), "merge receipt.residual_disclosures")
    if declared.get("status") != "unverified_by_this_merge" or recomputed.get("status") != "unverified_by_this_merge":
        _fail(
            "teacher-forced chosen-token prefix parity must remain 'unverified_by_this_merge'; a "
            "promoted/altered status is refused",
            declared_status=declared.get("status"),
        )
    declared_per_context = _mapping(
        declared.get("per_context_teacher_forced_chosen_token_parity"),
        "merge receipt.residual_disclosures.per_context_teacher_forced_chosen_token_parity",
    )
    if declared_per_context != recomputed["per_context_teacher_forced_chosen_token_parity"]:
        _fail(
            "merge receipt's republished per-context teacher-forced-parity disclosure does not match "
            "the plan's own contexts.jsonl declaration"
        )
    for context_id, parity in declared_per_context.items():
        if not isinstance(parity, Mapping) or parity.get("claimed_pass") is not False:
            _fail(
                f"context {context_id!r} teacher_forced_chosen_token_parity.claimed_pass must be "
                "declared false by the plan; refusing to attest a context whose prefix parity claim "
                "is not explicitly unproven"
            )
    return {
        "status": "unverified_by_this_merge",
        "raw_capture_validity_is_distinct_from_this_field": True,
        "per_context_teacher_forced_chosen_token_parity": declared_per_context,
    }


# ---------------------------------------------------------------------------
# Token execution parity: literal prefix-arithmetic attestation
# ---------------------------------------------------------------------------

TOKEN_EXECUTION_PARITY_KIND = "token_execution_parity"


def token_execution_parity_attestation(
    plan: score_module.PlanBundle, *, context_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    """CPU-only, structural: does the plan's own token arithmetic hold?

    For every covered context, checks that ``prompt_token_ids +
    generated_prefix_token_ids == full_prefix_token_ids`` (byte-for-byte, as
    plain integer lists) and that every one of the three declared
    ``*_token_ids_sha256`` digests matches its own content. Also binds each
    context's root prompt length and full-prefix length for the receipt.

    This is deliberately named ``token_execution_parity``, distinct from
    "chosen-token parity" or any numerical donor-logprob comparison: it
    proves the plan's literal prefix construction is internally consistent
    (no silent truncation/duplication/reordering when the donor-generated
    rows were concatenated onto the production prompt). It does **not**
    prove -- and no artifact in this unit can prove -- that the live model,
    teacher-forced along this exact prefix, would reproduce the donor's
    originally sampled tokens; no numerical chosen-token logprob comparator
    against the donor rollout exists anywhere in this pipeline.
    """

    selected_ids = sorted(context_ids) if context_ids else sorted(plan.contexts_by_id)
    unknown = sorted(set(selected_ids) - set(plan.contexts_by_id))
    if unknown:
        _fail("context ids are absent from the plan", unknown=unknown)

    per_context: dict[str, Any] = {}
    for context_id in selected_ids:
        context = plan.contexts_by_id[context_id]
        prompt = [int(v) for v in context["prompt_token_ids"]]
        generated_prefix = [int(v) for v in context["generated_prefix_token_ids"]]
        full_prefix = [int(v) for v in context["full_prefix_token_ids"]]
        if prompt + generated_prefix != full_prefix:
            _fail(
                f"context {context_id!r}: prompt_token_ids + generated_prefix_token_ids does not "
                "reconstruct full_prefix_token_ids; token execution parity violated",
                context_id=context_id,
            )
        for label, tokens, digest_field in (
            ("prompt_token_ids", prompt, "prompt_token_ids_sha256"),
            ("generated_prefix_token_ids", generated_prefix, "generated_prefix_token_ids_sha256"),
            ("full_prefix_token_ids", full_prefix, "full_prefix_token_ids_sha256"),
        ):
            if score_module.sha256_json(tokens) != context.get(digest_field):
                _fail(
                    f"context {context_id!r}.{digest_field} does not match its own {label} content",
                    context_id=context_id,
                )
        per_context[context_id] = {
            "root_prompt_length": len(prompt),
            "generated_prefix_length": len(generated_prefix),
            "full_prefix_length": len(full_prefix),
            "prompt_token_ids_sha256": context.get("prompt_token_ids_sha256"),
            "generated_prefix_token_ids_sha256": context.get("generated_prefix_token_ids_sha256"),
            "full_prefix_token_ids_sha256": context.get("full_prefix_token_ids_sha256"),
            "reconstruction_status": "passed",
        }

    return {
        "attestation_kind": TOKEN_EXECUTION_PARITY_KIND,
        "status": "passed",
        "not_a_numerical_chosen_token_parity_claim": True,
        "note": (
            "literal token-sequence arithmetic only (prompt_token_ids + generated_prefix_token_ids == "
            "full_prefix_token_ids, with every declared digest matching its own content) for every "
            "covered context. This is proof of consistent literal token execution/construction, never "
            "a numerical chosen-token logprob comparison against the donor rollout -- no such "
            "comparator exists in this pipeline."
        ),
        "context_count": len(per_context),
        "per_context": per_context,
    }


# ---------------------------------------------------------------------------
# Attestation driver
# ---------------------------------------------------------------------------


def attest(*, merge_receipt_path: Path, plan_dir: Path, output_dir: Path) -> dict[str, Any]:
    merge_receipt_path = merge_receipt_path.expanduser().resolve(strict=True)
    plan_dir = plan_dir.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve()
    attestation_path = output_dir / ATTESTATION_NAME

    merge_receipt = _read_json(merge_receipt_path, "merge receipt")
    _validate_merge_receipt(merge_receipt, merge_receipt_path=merge_receipt_path)

    plan = _independent_plan_binding(plan_dir, merge_receipt=merge_receipt)

    output = _mapping(merge_receipt.get("output_artifacts"), "merge receipt.output_artifacts")
    scores_entry = _mapping(output.get("merged_scores"), "merge receipt.output_artifacts.merged_scores")
    rows = _read_jsonl(Path(str(scores_entry.get("path"))), "merged scores")
    if len(rows) != scores_entry.get("row_count"):
        _fail("merged score row count does not match the merge receipt's declared row_count")

    coverage_attestation = _independent_coverage_check(plan, merge_receipt=merge_receipt, rows=rows)
    decision_channel_attestation = _decision_channel_attestation(rows)
    primary_role_attestation = _primary_role_separation_attestation(rows)
    batch_parity_attestation = _batch_parity_attestation(merge_receipt)
    numerical_repeat_attestation = _numerical_repeat_attestation(merge_receipt, rows)
    covered_context_ids = sorted({str(row["context_id"]) for row in rows})
    residual_attestation = _residual_disclosure_attestation(
        plan, merge_receipt=merge_receipt, covered_context_ids=covered_context_ids
    )
    token_execution_attestation = token_execution_parity_attestation(
        plan, context_ids=covered_context_ids
    )

    counts_by_kind: dict[str, int] = {}
    for row in rows:
        kind = str(row["request_kind"])
        counts_by_kind[kind] = counts_by_kind.get(kind, 0) + 1
    declared_counts = _mapping(merge_receipt.get("counts"), "merge receipt.counts")
    if declared_counts.get("rows_by_request_kind") != counts_by_kind or declared_counts.get("rows") != len(rows):
        _fail(
            "merge receipt.counts does not match independently recomputed counts over the merged rows",
            declared=dict(declared_counts),
            recomputed={"rows": len(rows), "rows_by_request_kind": counts_by_kind},
        )

    source_identity_sha256 = merge_receipt.get("source_identity_sha256")
    scorer_code_sha256 = merge_receipt.get("scorer_code_sha256")
    if not isinstance(source_identity_sha256, str) or not source_identity_sha256:
        _fail("merge receipt.source_identity_sha256 is missing")
    if not isinstance(scorer_code_sha256, str) or not scorer_code_sha256:
        _fail("merge receipt.scorer_code_sha256 is missing")

    document: dict[str, Any] = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "merge_receipt": {
            "path": str(merge_receipt_path),
            "sha256": sha256_file(merge_receipt_path),
            "schema_version": merge_receipt.get("schema_version"),
        },
        "plan": {
            "plan_dir": str(plan.plan_dir),
            "receipt_path": str(plan.receipt_path),
            "receipt_content_sha256": plan.receipt.get("receipt_content_sha256"),
        },
        "merged_scores": {
            "path": str(scores_entry.get("path")),
            "sha256": scores_entry.get("sha256"),
            "row_count": scores_entry.get("row_count"),
        },
        "covered_context_ids": covered_context_ids,
        "counts": {"rows": len(rows), "rows_by_request_kind": counts_by_kind},
        "checkpoint_source_identity": {
            "status": "passed",
            "source_identity_sha256": source_identity_sha256,
            "scorer_code_sha256": scorer_code_sha256,
        },
        "coverage_attestation": coverage_attestation,
        "decision_channel_attestation": decision_channel_attestation,
        "primary_role_separation_attestation": primary_role_attestation,
        "batch_parity_attestation": batch_parity_attestation,
        "numerical_repeat_attestation": numerical_repeat_attestation,
        "raw_capture_validity": {
            "status": "passed",
            "basis": [
                "coverage_attestation",
                "decision_channel_attestation",
                "primary_role_separation_attestation",
                "batch_parity_attestation",
                "checkpoint_source_identity",
            ],
        },
        "teacher_forced_chosen_token_parity_residual": residual_attestation,
        "token_execution_parity_attestation": token_execution_attestation,
        "disposition": "accepted",
        "scientific_conclusion": None,
    }
    encoded = score_module.canonical_json_bytes(document) + b"\n"
    _write_create_or_identical(attestation_path, encoded)
    return document


TOKEN_EXECUTION_ECHO_SCHEMA_VERSION = "sorted-all-person-route-landscape-token-execution-echo.v1"
TOKEN_EXECUTION_ECHO_NAME = "route-landscape-token-execution-echo.json"


def attest_token_execution_only(
    *, plan_dir: Path, output_dir: Path, include_context_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    """Standalone CPU-only token-execution-parity echo, independent of any merge/shard.

    Runs :func:`token_execution_parity_attestation` directly against the
    plan's ``contexts.jsonl`` -- no score shard, merge, or strict code-hash
    uniformity is required. This is the mode to reach for when the current
    score shards cannot yet pass a normal (or even salvage) merge, but the
    plan's own literal prefix construction can still be attested today.
    """

    plan_dir = plan_dir.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve()
    plan = score_module.load_plan(plan_dir)
    attestation = token_execution_parity_attestation(plan, context_ids=include_context_ids)
    document: dict[str, Any] = {
        "schema_version": TOKEN_EXECUTION_ECHO_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "plan": {
            "plan_dir": str(plan.plan_dir),
            "receipt_path": str(plan.receipt_path),
            "receipt_content_sha256": plan.receipt.get("receipt_content_sha256"),
        },
        "token_execution_parity_attestation": attestation,
        "disposition": "accepted",
        "scientific_conclusion": None,
    }
    encoded = score_module.canonical_json_bytes(document) + b"\n"
    _write_create_or_identical(output_dir / TOKEN_EXECUTION_ECHO_NAME, encoded)
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge-receipt", type=Path, default=None)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--token-execution-only",
        action="store_true",
        help=(
            "run only the standalone token_execution_parity echo against --plan-dir (no --merge-receipt "
            "required); use this when shards cannot yet pass a normal merge"
        ),
    )
    parser.add_argument(
        "--include-context-id",
        action="append",
        default=None,
        help="restrict --token-execution-only to these context ids (default: every context in the plan)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.token_execution_only:
        document = attest_token_execution_only(
            plan_dir=args.plan_dir,
            output_dir=args.output_dir,
            include_context_ids=args.include_context_id,
        )
        print(
            json.dumps(
                {
                    "disposition": document["disposition"],
                    "context_count": document["token_execution_parity_attestation"]["context_count"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.merge_receipt is None:
        raise SystemExit("--merge-receipt is required unless --token-execution-only is set")
    document = attest(merge_receipt_path=args.merge_receipt, plan_dir=args.plan_dir, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "disposition": document["disposition"],
                "covered_context_ids": document["covered_context_ids"],
                "counts": document["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
