#!/usr/bin/env python3
"""Successor-local run attestation for a merged sorted false-negative
mechanism decomposition score shard (see
``merge_sorted_fn_successor_score_shards.py``).

This is a CPU-only attestor. It never loads a model, tokenizer, or GPU; it
only reads the merger's already-sealed output plus the same
``landscape-decision-rules.json`` it was scored against, and states a
mechanical acceptance disposition -- never a scientific conclusion.

It exists to make explicit and checkable, in one immutable receipt, per
``unit.md``:

1. **scalar smoke vs scale** -- which acceptance mode this run is entering
   under (``--run-mode {scalar_smoke,scale}``), stated by the caller and
   never inferred. ``scalar_smoke`` additionally requires every merged
   shard's batched-reforward admission to declare an effective batch size of
   exactly ``1`` -- per unit.md, "cache reuse, reduced precision, or batched
   reforward cannot decide the smoke." A run that fails this is rejected
   outright, never softened into a held disposition;
2. **the successor scorer's own decision channel and schema** -- the
   decision-bearing channel is ``raw_model_logprob.complete_box_logprob_sum``
   from the unmodified fp32 lm-head channel under a literal ``use_cache=
   False`` full-prefix reforward, never the repetition-penalty
   ``auxiliary_policy_scores`` view; every merged row's native decode
   stratum is exactly ``1.0``; and the row/receipt schema versions named in
   the attestation are ``score_sorted_fn_fixed_budget.py``'s own
   (``SUCCESSOR_SCORE_ROW_SCHEMA_VERSION``/``SUCCESSOR_SCORE_RECEIPT_SCHEMA_
   VERSION``), never the predecessor scorer's;
3. **fail foreign/predecessor unit IDs** -- the merge receipt's own
   ``unit_id`` must be this successor's, and any merge receipt that carries
   the predecessor scorer's unit ID (impersonation, or a merge receipt built
   from predecessor scorer shards) is rejected outright. The predecessor's
   low-level *primitive-code* digest this scorer imports unmodified is
   surfaced separately, verbatim, and is never treated as a unit-ownership
   claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.build_sorted_fn_mechanism_registry import UNIT_ID  # noqa: E402
from scripts.research.prepare_sorted_fn_successor_inputs import (  # noqa: E402
    MECHANISM_DECISION_RULES_SCHEMA_VERSION,
)
from scripts.research.merge_sorted_fn_successor_score_shards import (  # noqa: E402
    MERGE_SCHEMA_VERSION,
    NATIVE_REPETITION_PENALTY_STRATUM,
    PREDECESSOR_PRIMITIVES_FILE,
    PREDECESSOR_SCORER_UNIT_ID,
    SUCCESSOR_SCORE_RECEIPT_SCHEMA_VERSION,
    SUCCESSOR_SCORE_ROW_SCHEMA_VERSION,
    SUCCESSOR_SCORER_UNIT_ID,
)

#: The frozen scalar_smoke lattice quotas primary population counts must
#: never drift from (see mechanism-decision-rules.json rung_quotas.
#: scalar_smoke); reference rows are optional (0 when no same-description,
#: zero-overlap owner exists in the image, else exactly 7).
SCALAR_SMOKE_TARGET_COUNT = 30
SCALAR_SMOKE_DECOY_COUNT = 30
SCALAR_SMOKE_REFERENCE_COUNTS = frozenset({0, 7})


ATTESTATION_SCHEMA_VERSION = "sorted_fn_successor_score_run_attestation.v2"
ATTESTATION_NAME = "fn-successor-run-attestation.json"
RUN_MODES = frozenset({"scalar_smoke", "scale"})
DECISION_BEARING_CHANNEL = "raw_model_logprob.complete_box_logprob_sum"


class RunAttestationError(RuntimeError):
    """A precondition for a mechanical run-attestation disposition failed."""


def _fail(message: str) -> NoReturn:
    raise RunAttestationError(message)


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {path} as JSON: {exc}")
    return dict(_mapping(value, str(path)))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.resolve(strict=True).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                _fail(f"{path}:{line_number} is not valid JSON: {exc}")
            rows.append(dict(_mapping(value, f"{path}:{line_number}")))
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


def _validate_merge_receipt(merge_receipt: Mapping[str, Any]) -> None:
    if merge_receipt.get("schema_version") != MERGE_SCHEMA_VERSION:
        _fail(f"merge receipt.schema_version must be {MERGE_SCHEMA_VERSION!r}")
    if merge_receipt.get("unit_id") == PREDECESSOR_SCORER_UNIT_ID:
        _fail("merge receipt.unit_id carries the predecessor scorer's unit_id; refusing to attest a foreign/predecessor artifact")
    if merge_receipt.get("unit_id") != UNIT_ID:
        _fail(
            "merge receipt does not carry this successor unit's own unit_id; refusing to attest a run "
            "that may be misattributed to a foreign unit"
        )
    scorer_provenance = _mapping(merge_receipt.get("successor_scorer_provenance"), "merge receipt.successor_scorer_provenance")
    if scorer_provenance.get("unit_id") != SUCCESSOR_SCORER_UNIT_ID:
        _fail("merge receipt successor_scorer_provenance.unit_id does not name the successor scorer's own unit_id")
    if scorer_provenance.get("unit_id") == PREDECESSOR_SCORER_UNIT_ID:
        _fail("merge receipt successor_scorer_provenance.unit_id carries the predecessor scorer's unit_id")
    if scorer_provenance.get("row_schema_version") != SUCCESSOR_SCORE_ROW_SCHEMA_VERSION:
        _fail("merge receipt successor_scorer_provenance.row_schema_version does not name the successor scorer's own schema")
    if scorer_provenance.get("receipt_schema_version") != SUCCESSOR_SCORE_RECEIPT_SCHEMA_VERSION:
        _fail("merge receipt successor_scorer_provenance.receipt_schema_version does not name the successor scorer's own schema")

    primitive_provenance = _mapping(
        merge_receipt.get("imported_predecessor_primitive_provenance"), "merge receipt.imported_predecessor_primitive_provenance"
    )
    if primitive_provenance.get("predecessor_primitives_file") != PREDECESSOR_PRIMITIVES_FILE:
        _fail("merge receipt imported_predecessor_primitive_provenance.predecessor_primitives_file is not the expected predecessor primitives file")
    digest = primitive_provenance.get("predecessor_primitives_file_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        _fail("merge receipt imported_predecessor_primitive_provenance.predecessor_primitives_file_sha256 must be a sha256 digest")

    output = _mapping(merge_receipt.get("output_artifacts"), "merge receipt.output_artifacts")
    scores_entry = _mapping(output.get("merged_scores"), "merge receipt.output_artifacts.merged_scores")
    scores_path = Path(str(scores_entry.get("path")))
    if sha256_file(scores_path) != scores_entry.get("sha256"):
        _fail("merge receipt merged_scores sha256 does not match the artifact on disk; stale merge output")


def _validate_mechanism_decision_rules(
    *, merge_receipt: Mapping[str, Any], mechanism_decision_rules_path: Path, decision_rules_path: Path
) -> dict[str, Any]:
    """Bind and validate the successor-owned mechanism-decision-rules.json.

    Distinct from the execution ``landscape-decision-rules.json`` bound
    separately below: this document freezes analysis policy (candidate
    generator, proposal weights, geometry regions, calibration) and is
    child-to-parent bound to the execution rules via ``upstream_digests``.
    Binding to the merge receipt's declared source digest transitively
    trusts the merger's own (already-exercised) join against the real
    planner receipt; this attestor independently re-verifies the mechanism
    document's own self-digest and its parent-execution binding.
    """

    source_digests = _mapping(merge_receipt.get("source_digests"), "merge receipt.source_digests")
    entry = _mapping(source_digests.get("mechanism_decision_rules"), "merge receipt.source_digests.mechanism_decision_rules")
    if entry.get("sha256") != sha256_file(mechanism_decision_rules_path):
        _fail("supplied mechanism-decision-rules.json does not match the merge receipt's bound digest")

    document = _read_json(mechanism_decision_rules_path)
    if document.get("schema_version") != MECHANISM_DECISION_RULES_SCHEMA_VERSION:
        _fail(f"mechanism-decision-rules.json.schema_version must be {MECHANISM_DECISION_RULES_SCHEMA_VERSION!r}")
    if document.get("unit_id") != UNIT_ID:
        _fail("mechanism-decision-rules.json.unit_id is not this successor unit's own unit_id")
    content = {key: value for key, value in document.items() if key != "self_digest"}
    if sha256_json(content) != document.get("self_digest"):
        _fail("mechanism-decision-rules.json self_digest does not reconstruct from its own content; stale or tampered")

    upstream = _mapping(document.get("upstream_digests"), "mechanism-decision-rules.json.upstream_digests")
    if upstream.get("execution_landscape_decision_rules_sha256") != sha256_file(decision_rules_path):
        _fail(
            "mechanism-decision-rules.json parent execution digest does not match the supplied "
            "landscape-decision-rules.json; stale mechanism-rules/execution-rules parent binding"
        )
    registry_entry = _mapping(source_digests.get("fn_mechanism_registry"), "merge receipt.source_digests.fn_mechanism_registry")
    if upstream.get("fn_mechanism_registry_sha256") != registry_entry.get("registry_digest"):
        _fail("mechanism-decision-rules.json registry binding does not match the merge receipt's bound FN mechanism registry_digest")
    # F3 mechanism-rule binding: the document's declared field-name
    # convention for the collision exact-GT-box statistic must match the
    # literal fields this attestor (and the merger) actually validate.
    singleton_section = _mapping(document.get("exact_gt_singleton"), "mechanism-decision-rules.json.exact_gt_singleton")
    if singleton_section.get("member_field") != "exact_gt_singleton_member" or singleton_section.get("id_field") != "exact_gt_singleton_id":
        _fail("mechanism-decision-rules.json.exact_gt_singleton declares an unrecognized field-name binding (F3)")
    return document


def _load_shard_receipts(merge_receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    shards = [_mapping(shard, "merge receipt shard entry") for shard in _sequence(merge_receipt.get("shards"), "merge receipt.shards")]
    if not shards:
        _fail("merge receipt has no shard lineage to attest")
    receipts: list[dict[str, Any]] = []
    for shard in shards:
        receipt_entry = _mapping(shard.get("receipt"), "merge receipt shard.receipt")
        receipt_path = Path(str(receipt_entry.get("path")))
        receipt = _read_json(receipt_path)
        if sha256_file(receipt_path) != receipt_entry.get("sha256"):
            _fail(f"shard receipt {receipt_path} does not match the digest bound by the merge receipt")
        receipts.append(receipt)
    return receipts


def _decision_channel_attestation(
    merge_receipt: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], shard_receipts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    identity_projection_sha256 = merge_receipt.get("identity_projection_sha256")
    if not isinstance(identity_projection_sha256, str) or not identity_projection_sha256:
        _fail("merge receipt is missing identity_projection_sha256")
    raw_channels = {
        _mapping(receipt.get("likelihood_channels"), "shard.likelihood_channels").get("raw") for receipt in shard_receipts
    }
    if len(raw_channels) != 1:
        _fail("shard likelihood_channels.raw is not uniform across shards")
    raw_channel = next(iter(raw_channels))
    if not isinstance(raw_channel, str) or "fp32" not in raw_channel:
        _fail("shard likelihood_channels.raw does not declare an fp32 raw channel")
    strata = set()
    for row in rows:
        if row.get("schema_version") != SUCCESSOR_SCORE_ROW_SCHEMA_VERSION:
            _fail("merged score row does not carry the successor scorer's own row schema_version")
        if row.get("unit_id") != SUCCESSOR_SCORER_UNIT_ID:
            _fail("merged score row does not carry the successor scorer's own unit_id")
        raw = _mapping(row.get("raw_model_logprob"), "merged score row.raw_model_logprob")
        if "complete_box_logprob_sum" not in raw:
            _fail("merged score row is missing raw_model_logprob.complete_box_logprob_sum")
        strata.add(row.get("native_repetition_penalty_stratum"))
    if strata != {NATIVE_REPETITION_PENALTY_STRATUM}:
        _fail(f"merged score rows are not uniformly at repetition-penalty stratum {NATIVE_REPETITION_PENALTY_STRATUM!r}")
    return {
        "decision_bearing_channel": DECISION_BEARING_CHANNEL,
        "channel_note": (
            "raw_model_logprob.complete_box_logprob_sum is the unmodified fp32 lm-head channel, computed "
            "under a literal use_cache=False full-prefix reforward, and is the only decision-bearing "
            "likelihood; auxiliary_policy_scores (rp=1.0/rp=1.10) are labelled repetition-penalty policy "
            "views and are never treated as a model likelihood"
        ),
        "raw_channel_declaration": raw_channel,
        "repetition_penalty_stratum": {"value": NATIVE_REPETITION_PENALTY_STRATUM, "status": "passed"},
        "successor_scorer_schema": {
            "row_schema_version": SUCCESSOR_SCORE_ROW_SCHEMA_VERSION,
            "receipt_schema_version": SUCCESSOR_SCORE_RECEIPT_SCHEMA_VERSION,
        },
    }


def _validate_mechanism_row_bindings(
    rows: Sequence[Mapping[str, Any]], *, mechanism_decision_rules_sha256: str
) -> dict[str, Any]:
    """Every merged row's parent mechanism-rules digest, field-validated.

    Defense in depth: the merger already enforces this per candidate join;
    this attestor independently re-derives it over the final merged surface,
    plus validates the reference/neighborhood field shapes the task
    requires this attestor to name explicitly. ``mechanism_decision_rules_
    sha256`` here is the raw-file sha256 the planner embeds on every row
    (``sha256_file(mechanism_decision_rules_path)``), distinct from the
    document's own content ``self_digest`` (validated separately).
    """

    population_counts: dict[str, int] = {}
    for row in rows:
        if row.get("mechanism_decision_rules_sha256") != mechanism_decision_rules_sha256:
            _fail("merged score row mechanism_decision_rules_sha256 does not match the bound mechanism-decision-rules.json")
        population = row.get("population")
        if population not in {"target", "decoy", "reference"}:
            _fail(f"merged score row.population must be target/decoy/reference, observed {population!r}")
        population_counts[str(population)] = population_counts.get(str(population), 0) + 1
        if population == "reference":
            if row.get("region") != "other_owner":
                _fail("merged score row has population=reference but region!=other_owner")
            if not isinstance(row.get("other_owner_gt_owner_id"), str) or not row.get("other_owner_gt_owner_id"):
                _fail("merged score row has population=reference but no other_owner_gt_owner_id")
            if not _mapping(row.get("other_owner_selection_trace"), "merged score row.other_owner_selection_trace"):
                _fail("merged score row has population=reference but an empty other_owner_selection_trace")
        member = row.get("candidate_neighborhood_member")
        if not isinstance(member, bool):
            _fail("merged score row.candidate_neighborhood_member must be a boolean")
        if member and not row.get("candidate_neighborhood_id"):
            _fail("merged score row is a candidate-neighborhood member but declares no candidate_neighborhood_id")
        singleton_member = row.get("exact_gt_singleton_member")
        if not isinstance(singleton_member, bool):
            _fail("merged score row.exact_gt_singleton_member must be a boolean")
        if singleton_member and not str(row.get("exact_gt_singleton_id") or "").startswith("exact-gt:"):
            _fail("merged score row is exact_gt_singleton_member but declares no recognizable exact_gt_singleton_id")

    singleton_groups: dict[tuple[str, str], int] = {}
    for row in rows:
        if row.get("population") == "target" and row.get("family_id") == "near_gt_micro" and row.get("exact_gt_singleton_member"):
            key = (str(row.get("context_id")), str(row.get("rung")))
            singleton_groups[key] = singleton_groups.get(key, 0) + 1
    drifted = {key: count for key, count in singleton_groups.items() if count != 1}
    if drifted:
        _fail(f"merged score rows have context/rung group(s) with != 1 exact_gt_singleton_member row: {drifted}")
    return population_counts


def _validate_run_mode_rung_binding(
    merge_receipt: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, run_mode: str
) -> list[str]:
    """Fail-closed bidirectional binding between ``--run-mode`` and the selected rungs.

    ``run_mode == "scalar_smoke"`` is only ever entered when the merge
    receipt's own ``selected_rungs`` is exactly ``["scalar_smoke"]`` and
    every merged row's ``rung`` agrees; any other rung present -- even
    alongside ``scalar_smoke`` rows -- is rejected outright, never silently
    filtered down to the scalar subset. Symmetrically, ``run_mode ==
    "scale"`` forbids ``scalar_smoke`` from appearing anywhere in the
    selection or the rows: scalar-smoke acceptance (its frozen 30/30/{0,7}
    population lattice and required effective-batch-size-1 backend gate)
    must only ever be entered through ``--run-mode scalar_smoke``, never
    bypassed by attesting scalar_smoke-selected rows under ``--run-mode
    scale`` (which never enforces those gates).
    """

    raw_selected_rungs = _sequence(merge_receipt.get("selected_rungs"), "merge receipt.selected_rungs")
    selected_rungs = sorted({str(value) for value in raw_selected_rungs})
    if not selected_rungs:
        _fail("merge receipt.selected_rungs must be non-empty")
    observed_rungs = sorted({str(row.get("rung")) for row in rows})
    if selected_rungs != observed_rungs:
        _fail(
            "merge receipt.selected_rungs does not exactly equal the rungs actually present in the merged "
            f"score rows; declared={selected_rungs} observed={observed_rungs}"
        )

    is_scalar_smoke_only = selected_rungs == ["scalar_smoke"]
    if run_mode == "scalar_smoke":
        if not is_scalar_smoke_only:
            _fail(
                "run_mode=scalar_smoke requires the merge receipt's selected_rungs to be exactly "
                f"['scalar_smoke']; observed {selected_rungs}"
            )
    elif "scalar_smoke" in selected_rungs:
        _fail(
            f"run_mode={run_mode!r} forbids any scalar_smoke rung selection or row; scalar_smoke "
            "acceptance (population/batch gates) must be entered only via --run-mode scalar_smoke, "
            "never bypassed by attesting scalar_smoke-selected rows under a non-scalar run_mode"
        )
    return selected_rungs


def _validate_scalar_smoke_populations(rows: Sequence[Mapping[str, Any]], *, run_mode: str) -> dict[str, Any] | None:
    """Per scalar_smoke context: 30 target + 30 decoy unchanged, 0 or 7 reference.

    Only enforced when attesting under ``run_mode == "scalar_smoke"``; a
    scale-mode run may legitimately use other rungs.
    """

    scalar_rows = [row for row in rows if row.get("rung") == "scalar_smoke"]
    if run_mode != "scalar_smoke":
        return None
    if not scalar_rows:
        _fail("scalar_smoke acceptance requires at least one rung=scalar_smoke row")
    by_context: dict[str, dict[str, int]] = {}
    for row in scalar_rows:
        context_id = str(row.get("context_id"))
        counts = by_context.setdefault(context_id, {"target": 0, "decoy": 0, "reference": 0})
        population = str(row.get("population"))
        if population not in counts:
            _fail(f"scalar_smoke row at context {context_id!r} has an unrecognized population: {population!r}")
        counts[population] += 1
    for context_id, counts in by_context.items():
        if counts["target"] != SCALAR_SMOKE_TARGET_COUNT or counts["decoy"] != SCALAR_SMOKE_DECOY_COUNT:
            _fail(
                f"scalar_smoke context {context_id!r} primary population counts drifted from the frozen "
                f"{SCALAR_SMOKE_TARGET_COUNT}/{SCALAR_SMOKE_DECOY_COUNT} target/decoy lattice; observed {counts}"
            )
        if counts["reference"] not in SCALAR_SMOKE_REFERENCE_COUNTS:
            _fail(
                f"scalar_smoke context {context_id!r} reference row count must be one of "
                f"{sorted(SCALAR_SMOKE_REFERENCE_COUNTS)}; observed {counts['reference']}"
            )
    return {
        "status": "passed",
        "target_count": SCALAR_SMOKE_TARGET_COUNT,
        "decoy_count": SCALAR_SMOKE_DECOY_COUNT,
        "allowed_reference_counts": sorted(SCALAR_SMOKE_REFERENCE_COUNTS),
        "contexts_checked": sorted(by_context),
        "per_context_counts": by_context,
    }


def _scalar_acceptance_backend(*, run_mode: str, shard_receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    admissions = [_mapping(r.get("scoring_backend_admission"), "shard.scoring_backend_admission") for r in shard_receipts]
    selected_backends = {a.get("selected_backend") for a in admissions}
    cache_enabled = {a.get("cache_enabled") for a in admissions}
    use_cache = {a.get("use_cache") for a in admissions}
    atol = {a.get("atol") for a in admissions}
    rtol = {a.get("rtol") for a in admissions}
    if len(selected_backends) != 1 or len(cache_enabled) != 1 or len(use_cache) != 1:
        _fail("shard scoring backends are not uniform; the merge should already have rejected this")

    batched = [
        _mapping(a.get("batched_reforward_admission"), "shard.scoring_backend_admission.batched_reforward_admission")
        for a in admissions
    ]
    effective_batch_sizes = sorted({int(b["effective_batch_size"]) for b in batched})
    requested_batch_sizes = sorted({int(b["requested_batch_size"]) for b in batched})
    batch_statuses = sorted({str(b["status"]) for b in batched})

    backend = {
        "selected_backend": next(iter(selected_backends)),
        "cache_enabled": next(iter(cache_enabled)),
        "use_cache": next(iter(use_cache)),
        "atol": next(iter(atol)) if len(atol) == 1 else sorted(atol),
        "rtol": next(iter(rtol)) if len(rtol) == 1 else sorted(rtol),
        "effective_batch_sizes": effective_batch_sizes,
        "requested_batch_sizes": requested_batch_sizes,
        "batched_reforward_statuses": batch_statuses,
    }
    if run_mode != "scalar_smoke":
        return {**backend, "required": False, "status": "not_applicable_scale_mode"}

    if effective_batch_sizes != [1]:
        _fail(
            "scalar_smoke acceptance requires every merged shard to declare an effective batch size of 1 "
            f"(scalar, uncached, full fp32 reforward); observed effective_batch_sizes={effective_batch_sizes}"
        )
    return {**backend, "required": True, "status": "passed"}


def attest(
    *,
    merge_receipt_path: Path,
    decision_rules_path: Path,
    mechanism_decision_rules_path: Path,
    run_mode: str,
    output_dir: Path,
) -> dict[str, Any]:
    if run_mode not in RUN_MODES:
        _fail(f"--run-mode must be one of {sorted(RUN_MODES)}")
    merge_receipt_path = merge_receipt_path.expanduser().resolve(strict=True)
    decision_rules_path = decision_rules_path.expanduser().resolve(strict=True)
    mechanism_decision_rules_path = mechanism_decision_rules_path.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve()
    attestation_path = output_dir / ATTESTATION_NAME

    merge_receipt = _read_json(merge_receipt_path)
    _validate_merge_receipt(merge_receipt)

    decision_rules_entry = _mapping(
        merge_receipt.get("source_digests", {}).get("decision_rules"), "merge receipt.source_digests.decision_rules"
    )
    if decision_rules_entry.get("sha256") != sha256_file(decision_rules_path):
        _fail("supplied landscape-decision-rules.json does not match the merge receipt's bound digest")
    mechanism_rules_document = _validate_mechanism_decision_rules(
        merge_receipt=merge_receipt,
        mechanism_decision_rules_path=mechanism_decision_rules_path,
        decision_rules_path=decision_rules_path,
    )

    output = _mapping(merge_receipt.get("output_artifacts"), "merge receipt.output_artifacts")
    scores_entry = _mapping(output.get("merged_scores"), "merge receipt.output_artifacts.merged_scores")
    rows = _read_jsonl(Path(str(scores_entry.get("path"))))
    if len(rows) != scores_entry.get("row_count"):
        _fail("merged score row count does not match the merge receipt's declared row_count")

    selected_rungs = _validate_run_mode_rung_binding(merge_receipt, rows, run_mode=run_mode)

    shard_receipts = _load_shard_receipts(merge_receipt)
    channel_attestation = _decision_channel_attestation(merge_receipt, rows, shard_receipts)
    backend_attestation = _scalar_acceptance_backend(run_mode=run_mode, shard_receipts=shard_receipts)
    population_counts = _validate_mechanism_row_bindings(
        rows, mechanism_decision_rules_sha256=sha256_file(mechanism_decision_rules_path)
    )
    scalar_smoke_populations = _validate_scalar_smoke_populations(rows, run_mode=run_mode)

    document = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "run_mode": run_mode,
        "selected_rungs": selected_rungs,
        "merge_receipt": {
            "path": str(merge_receipt_path),
            "sha256": sha256_file(merge_receipt_path),
            "schema_version": merge_receipt.get("schema_version"),
        },
        "merged_scores": {
            "path": str(scores_entry.get("path")),
            "sha256": scores_entry.get("sha256"),
            "row_count": scores_entry.get("row_count"),
        },
        "decision_rules_sha256": decision_rules_entry.get("sha256"),
        "mechanism_decision_rules_attestation": {
            "path": str(mechanism_decision_rules_path),
            "sha256": sha256_file(mechanism_decision_rules_path),
            "schema_version": mechanism_rules_document.get("schema_version"),
            "self_digest": mechanism_rules_document["self_digest"],
            "parent_execution_rules_sha256": mechanism_rules_document["upstream_digests"][
                "execution_landscape_decision_rules_sha256"
            ],
            "fn_mechanism_registry_sha256": mechanism_rules_document["upstream_digests"]["fn_mechanism_registry_sha256"],
            "population_counts": dict(sorted(population_counts.items())),
            "exact_gt_singleton_binding": dict(mechanism_rules_document["exact_gt_singleton"]),
        },
        "scalar_smoke_population_admission": scalar_smoke_populations,
        "decision_channel_attestation": channel_attestation,
        "scalar_acceptance_backend": backend_attestation,
        "successor_scorer_provenance": dict(_mapping(merge_receipt.get("successor_scorer_provenance"), "merge receipt.successor_scorer_provenance")),
        "imported_predecessor_primitive_provenance": {
            **_mapping(merge_receipt.get("imported_predecessor_primitive_provenance"), "merge receipt.imported_predecessor_primitive_provenance"),
            "is_not_a_unit_ownership_claim": True,
        },
        "disposition": "accepted",
        "scientific_conclusion": None,
    }
    encoded = canonical_json(document).encode("utf-8") + b"\n"
    _write_create_or_identical(attestation_path, encoded)
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge-receipt", type=Path, required=True)
    parser.add_argument("--decision-rules", type=Path, required=True)
    parser.add_argument("--mechanism-decision-rules", type=Path, required=True)
    parser.add_argument("--run-mode", choices=sorted(RUN_MODES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document = attest(
        merge_receipt_path=args.merge_receipt,
        decision_rules_path=args.decision_rules,
        mechanism_decision_rules_path=args.mechanism_decision_rules,
        run_mode=args.run_mode,
        output_dir=args.output_dir,
    )
    print(json.dumps({"disposition": document["disposition"], "run_mode": document["run_mode"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
