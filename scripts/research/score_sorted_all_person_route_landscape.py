#!/usr/bin/env python3
"""Score raw pre-penalty complete-box likelihoods for the sorted all-person
route landscape unit (``2026-08-03-sorted-all-person-owner-relative-route-
landscape``).

Consumes the immutable CPU-only plan produced by
``build_sorted_all_person_route_landscape.py`` (``scoring-requests.jsonl``
joined against ``contexts.jsonl``/``primary-candidates.jsonl``/
``sidecars.jsonl``) and emits one raw, unmodified fp32 lm-head log-likelihood
per request under a literal ``use_cache=False`` full-prefix reforward -- the
sole decision-bearing channel this unit scores
(``raw_model_logprob.complete_box_logprob_sum``). No repetition-penalty
auxiliary policy view is computed; this scorer is pre-penalty only.

This module reuses the predecessor's low-level scoring primitives
unmodified: ``score_sorted_owner_basin_landscape.FullReforwardBackend``,
``.score_complete_box_candidate``, ``.build_attestation_context``, and the
uncached full-reforward closures. It never opens the KV-cache branching
path -- every request is scored via a literal, uncached, full fp32
reforward per candidate. Batched execution (``--full-reforward-batch-size``
> 1) is admitted per context only after
``score_sorted_owner_basin_landscape.run_batched_reforward_parity_gate``
passes that context's own coordinate-argmax/logprob probe within its fixed
``BATCH_REFORWARD_COORD_LOGPROB_MAX_ABS_DIFF`` tolerance (``1e-3``); a
failed probe either aborts the run or falls back to scalar, always
explicitly recorded, per ``--on-batch-parity-failure``. That tolerance is
looser than this unit's own frozen numerical-repeat epsilon (typically
``1e-6``, derived from eight scalar repeats); a passed batch admission is
evidence of *this scorer's* coordinate-behavior gate, not proof a batched
row meets the unit's tighter numerical standard -- see
``adjudicate_sorted_all_person_route_landscape_provenance.py`` for a
concrete case where batched rows showed a max abs diff of
``6.8e-5``-``2.2e-4``, comfortably inside the scorer's ``1e-3`` gate but
outside the unit's ``1e-6`` epsilon.

CLI selection (``--include-context-id``, ``--include-request-kind``,
``--include-request-id``, ``--shard-index``/``--shard-count``) lets one run
cover a narrow slice (for example, the eight ``self-due-gt17`` numerical
repeats, one context's full candidate bank, or a hand-picked set of exact
request ids for posthoc scalar rescoring of specific rank-critical rows) so
the representative smoke can proceed in stages. The
output directory is create-or-identical, exactly like the planner: an
existing shard may only be reused when every emitted byte is identical.

Shard-receipt code identity (schema v2)
----------------------------------------
Every shard receipt's ``code`` block binds three fields, never just one
live file hash:

* ``executed_source_sha256`` -- this module's own source bytes, hashed
  exactly once at *import time* (module-level constant
  :data:`EXECUTED_SOURCE_SHA256`). This is the fingerprint of the code that
  actually executed the run, immune to concurrent edits to this file on
  disk while the process is still running.
* ``receipt_time_file_sha256`` -- a live re-read of ``Path(__file__)`` at
  receipt-build time (end of job); purely diagnostic, never used for
  identity/admission decisions.
* ``source_drift_detected`` -- ``receipt_time_file_sha256 !=
  executed_source_sha256``; true only when the file on disk changed after
  this process imported it. A true value does not itself invalidate the
  run (the run executed the *imported* bytes regardless), but it is a
  loud, structural signal worth surfacing.
* ``sha256`` -- retained for backward-compatible callers; always equal to
  ``executed_source_sha256`` (never the stale end-of-job read that caused
  the 2026-08-03 code-identity race across six context captures).

``RECEIPT_SCHEMA_VERSION`` was bumped ("...v1" -> "...v2") for this
semantic change: a v1 receipt's ``code.sha256`` cannot be trusted as an
import-time fingerprint and must never be treated as equivalent to a v2
receipt's. Strict merge (``merge_sorted_all_person_route_landscape.py``)
requires v2 uniformly; the six pre-fix v1 shards are only ever examined
through the explicit, opt-in, discovery-only salvage path in
``adjudicate_sorted_all_person_route_landscape_provenance.py``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_all_person_route_landscape as planner  # noqa: E402
from scripts.research import score_sorted_owner_basin_landscape as scorer  # noqa: E402

SCHEMA_VERSION = "sorted-all-person-route-landscape-score.v1"
#: v2: shard-receipt ``code`` binds an import-time source fingerprint
#: (``executed_source_sha256``) instead of a live end-of-job file hash; see
#: the module docstring "Shard-receipt code identity (schema v2)".
RECEIPT_SCHEMA_VERSION = "sorted-all-person-route-landscape-score-receipt.v2"
UNIT_ID = planner.UNIT_ID
PLAN_SCHEMA_VERSION = planner.SCHEMA_VERSION

OUTPUT_JSONL_NAME = "route-landscape-scores.jsonl"
OUTPUT_RECEIPT_NAME = "route-landscape-scores-receipt.json"

#: The sole decision-bearing channel this unit ever scores; see unit.md
#: "Decision score: raw pre-penalty complete-box log likelihood".
DECISION_BEARING_CHANNEL = "raw_model_logprob.complete_box_logprob_sum"
NATIVE_REPETITION_PENALTY_STRATUM = 1.0

REQUEST_KINDS = frozenset({"primary", "sidecar", "numerical_repeat"})
#: Only "primary" rows enter primary owner-relative ranks; sidecars are a
#: score-independent finite-bank diagnostic and numerical repeats are a
#: variance probe, per unit.md.
PRIMARY_ROLE_REQUEST_KINDS = frozenset({"primary"})

PLAN_FILE_NAMES: tuple[str, ...] = (
    "owner-ledger.jsonl",
    "primary-candidates.jsonl",
    "contexts.jsonl",
    "sidecars.jsonl",
    "sampling-seeds.jsonl",
    "scoring-requests.jsonl",
)

canonical_json_bytes = planner.canonical_json_bytes
sha256_json = planner.sha256_json
sha256_file = planner.sha256_file

#: The exact source bytes of *this* module, hashed exactly once, at import
#: time. This is the "executed source" fingerprint every shard receipt binds
#: -- never a live re-read of ``Path(__file__)`` at receipt-build time.
#:
#: Incident this guards against: a long-running scoring process imports this
#: module once at process start; if the file on disk is concurrently edited
#: while that process is still scoring (e.g. a sibling development change),
#: a receipt built from a live end-of-job ``sha256_file(Path(__file__))``
#: read captures the *edited* bytes, not the bytes that actually executed --
#: silently misattributing the run's code identity. Six 2026-08-03 sorted-
#: all-person-route-landscape context captures were affected by exactly this
#: race (concurrent request-id/merge-plumbing edits landed while those jobs
#: were still running); see
#: ``adjudicate_sorted_all_person_route_landscape_provenance.py`` for the
#: discovery-only salvage path over those six pre-fix shards.
EXECUTED_SOURCE_SHA256 = sha256_file(Path(__file__).resolve())


class ScoreRunError(RuntimeError):
    """Raised before scoring/output when a precondition for this scorer fails."""

    def __init__(self, message: str, **context: Any) -> None:
        if context:
            message = f"{message} | context: {json.dumps(context, sort_keys=True, default=str)}"
        super().__init__(message)


def _fail(message: str, **context: Any) -> NoReturn:
    raise ScoreRunError(message, **context)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a JSON object")
    return value


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ScoreRunError(f"{label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ScoreRunError(f"{label} is not valid JSON: {path}") from exc
    return _mapping(raw, label)


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ScoreRunError(f"{label} is missing: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            _fail(f"{label} line {line_number} is blank")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScoreRunError(f"{label} line {line_number} is not JSON") from exc
        rows.append(dict(_mapping(raw, f"{label} line {line_number}")))
    if not rows:
        _fail(f"{label} must contain at least one row")
    return rows


def _write_create_or_identical(path: Path, content: bytes, *, force: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == content:
            return "identical_existing_output"
        if not force:
            _fail(
                f"{path} already exists with different (or non-file) content; refusing to overwrite "
                "without --force"
            )
        temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temp.write_bytes(content)
        os.replace(temp, path)
        return "overwritten_forced"
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_bytes(content)
    os.replace(temp, path)
    return "created"


def _normalized_executed_media_sha256(value: Any, *, expected_count: int) -> tuple[str, ...]:
    """Strictly normalize ``_materialize_native_inputs``'s per-request digest sequence.

    ``HFBackendSession._materialize_native_inputs`` always returns one digest
    per requested row (a tuple), even for a single-request call. A prior
    version of this scorer compared that sequence directly against the
    frozen scalar ``EXECUTED_MEDIA_SHA256``, which can never match a
    length-1 tuple; this normalizer requires the exact expected row count,
    requires every entry to be identical (this unit is single-image), and
    only then exposes the shared scalar digest -- never silently taking an
    arbitrary element of a mismatched-length or non-uniform sequence.
    """

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(
            "materialized executed_media_sha256 must be a per-request sequence, not a bare scalar",
            observed=value,
        )
    normalized = tuple(str(item) for item in value)
    if len(normalized) != expected_count:
        _fail(
            "materialized executed_media_sha256 length does not match the requested batch size",
            expected_count=expected_count,
            observed_count=len(normalized),
        )
    if len(set(normalized)) != 1:
        _fail(
            "materialized executed_media_sha256 is not uniform across a single-image batch",
            observed=normalized,
        )
    return normalized


# ---------------------------------------------------------------------------
# Plan loading (tamper-evident: every consumed file is re-hashed against the
# planner's own sealed receipt before any row is trusted)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanBundle:
    plan_dir: Path
    receipt_path: Path
    receipt: Mapping[str, Any]
    contexts_by_id: dict[str, Mapping[str, Any]]
    candidates_by_id: dict[str, Mapping[str, Any]]
    sidecars_by_id: dict[str, Mapping[str, Any]]
    requests_by_id: dict[str, Mapping[str, Any]]
    file_paths: dict[str, Path]


def load_plan(plan_dir: str | Path) -> PlanBundle:
    resolved_dir = Path(plan_dir).expanduser().resolve(strict=True)
    receipt_path = resolved_dir / "receipt.json"
    receipt = _read_json(receipt_path, "plan receipt.json")
    if receipt.get("schema_version") != PLAN_SCHEMA_VERSION:
        _fail(
            "plan receipt.schema_version does not match this scorer's expected planner schema",
            observed=receipt.get("schema_version"),
            expected=PLAN_SCHEMA_VERSION,
        )
    if receipt.get("unit_id") != UNIT_ID:
        _fail(
            "plan receipt.unit_id does not match this unit's own unit_id",
            observed=receipt.get("unit_id"),
            expected=UNIT_ID,
        )
    reconstructed = sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    if reconstructed != receipt.get("receipt_content_sha256"):
        _fail(
            "plan receipt.receipt_content_sha256 does not reconstruct from its own content; "
            "stale or tampered plan receipt"
        )

    output_digests = _mapping(
        receipt.get("output_file_digests"), "plan receipt.output_file_digests"
    )
    file_paths: dict[str, Path] = {}
    for name in PLAN_FILE_NAMES:
        path = resolved_dir / name
        if not path.is_file():
            _fail(f"plan directory is missing declared output file {name!r}: {path}")
        expected_digest = output_digests.get(name)
        if not isinstance(expected_digest, str) or not expected_digest:
            _fail(f"plan receipt does not declare a digest for {name!r}")
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            _fail(
                f"plan file {name!r} does not match the digest sealed in the plan receipt; "
                "tampered or stale plan artifact",
                path=str(path),
                expected=expected_digest,
                observed=actual_digest,
            )
        file_paths[name] = path

    contexts_rows = _read_jsonl(file_paths["contexts.jsonl"], "contexts.jsonl")
    candidates_rows = _read_jsonl(file_paths["primary-candidates.jsonl"], "primary-candidates.jsonl")
    sidecars_rows = _read_jsonl(file_paths["sidecars.jsonl"], "sidecars.jsonl")
    requests_rows = _read_jsonl(file_paths["scoring-requests.jsonl"], "scoring-requests.jsonl")

    contexts_by_id = {str(row["context_id"]): row for row in contexts_rows}
    candidates_by_id = {str(row["candidate_id"]): row for row in candidates_rows}
    sidecars_by_id = {str(row["sidecar_id"]): row for row in sidecars_rows}
    requests_by_id: dict[str, Mapping[str, Any]] = {}
    for row in requests_rows:
        request_id = str(row.get("request_id"))
        if not request_id or request_id in requests_by_id:
            _fail("scoring-requests.jsonl has a missing or duplicate request_id", request_id=request_id)
        if row.get("request_kind") not in REQUEST_KINDS:
            _fail(
                "scoring-requests.jsonl row has an unrecognized request_kind",
                request_id=request_id,
                request_kind=row.get("request_kind"),
            )
        requests_by_id[request_id] = row

    return PlanBundle(
        plan_dir=resolved_dir,
        receipt_path=receipt_path,
        receipt=receipt,
        contexts_by_id=contexts_by_id,
        candidates_by_id=candidates_by_id,
        sidecars_by_id=sidecars_by_id,
        requests_by_id=requests_by_id,
        file_paths=file_paths,
    )


# ---------------------------------------------------------------------------
# Selection: contexts / request kinds / shard-index-and-count
# ---------------------------------------------------------------------------


def _load_request_id_file(path: Path) -> list[str]:
    """One request id per line, or one JSONL object with a ``request_id`` field per line.

    Fails closed on any blank line and on any duplicate id within the file
    (a separate check against duplicates across ``--include-request-id`` and
    this file happens at the call site, before either reaches
    :func:`select_requests`).
    """

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ScoreRunError(f"--include-request-id-file does not exist: {path}") from exc
    ids: list[str] = []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            _fail(f"--include-request-id-file line {line_number} is blank", path=str(path))
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScoreRunError(
                    f"--include-request-id-file line {line_number} looks like JSON but failed to parse"
                ) from exc
            if not isinstance(parsed, Mapping) or not isinstance(parsed.get("request_id"), str) or not parsed["request_id"]:
                _fail(
                    f"--include-request-id-file line {line_number} is a JSON object with no non-empty "
                    "string request_id field",
                    path=str(path),
                )
            ids.append(parsed["request_id"])
        else:
            ids.append(line)
    if not ids:
        _fail("--include-request-id-file contains no request ids", path=str(path))
    if len(ids) != len(set(ids)):
        duplicates = sorted({request_id for request_id in ids if ids.count(request_id) > 1})
        _fail(
            "--include-request-id-file contains duplicate request ids",
            path=str(path),
            duplicates=duplicates[:8],
        )
    return ids


def select_requests(
    plan: PlanBundle,
    *,
    include_context_ids: Sequence[str] | None = None,
    include_request_kinds: Sequence[str] | None = None,
    include_request_ids: Sequence[str] | None = None,
    request_id_file: tuple[str, str] | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    if shard_count < 1:
        _fail("--shard-count must be a positive integer", shard_count=shard_count)
    if not 0 <= shard_index < shard_count:
        _fail(
            "--shard-index must be in [0, shard_count)",
            shard_index=shard_index,
            shard_count=shard_count,
        )

    available_context_ids = sorted(plan.contexts_by_id)
    requested_context_ids = list(include_context_ids or ())
    if len(requested_context_ids) != len(set(requested_context_ids)):
        _fail("--include-context-id contains duplicates", requested=requested_context_ids)
    unknown_contexts = sorted(set(requested_context_ids) - set(available_context_ids))
    if unknown_contexts:
        _fail(
            "--include-context-id names context IDs absent from the plan",
            unknown=unknown_contexts,
            available=available_context_ids,
        )
    context_ids = sorted(requested_context_ids) if requested_context_ids else available_context_ids

    requested_kinds = list(include_request_kinds or ())
    if len(requested_kinds) != len(set(requested_kinds)):
        _fail("--include-request-kind contains duplicates", requested=requested_kinds)
    unknown_kinds = sorted(set(requested_kinds) - REQUEST_KINDS)
    if unknown_kinds:
        _fail(
            "--include-request-kind names kinds absent from the plan's request-kind domain",
            unknown=unknown_kinds,
            available=sorted(REQUEST_KINDS),
        )
    request_kinds = sorted(requested_kinds) if requested_kinds else sorted(REQUEST_KINDS)

    # Strict request-id selector: validated against the *full* plan domain
    # before any context/kind filtering, so a typo'd or foreign id always
    # fails loudly here rather than silently vanishing into an empty (or
    # unexpectedly non-empty) context/kind-filtered result.
    requested_request_ids = list(include_request_ids or ())
    if len(requested_request_ids) != len(set(requested_request_ids)):
        _fail("--include-request-id contains duplicates", requested=requested_request_ids)
    unknown_request_ids = sorted(set(requested_request_ids) - set(plan.requests_by_id))
    if unknown_request_ids:
        _fail(
            "--include-request-id names request IDs absent from the plan's full scoring-request domain",
            unknown=unknown_request_ids[:8],
            unknown_count=len(unknown_request_ids),
        )
    request_id_set = set(requested_request_ids) if requested_request_ids else None

    context_id_set = set(context_ids)
    request_kind_set = set(request_kinds)
    filtered = sorted(
        (
            row
            for row in plan.requests_by_id.values()
            if str(row["context_id"]) in context_id_set
            and str(row["request_kind"]) in request_kind_set
            and (request_id_set is None or str(row["request_id"]) in request_id_set)
        ),
        key=lambda row: str(row["request_id"]),
    )
    if request_id_set is not None:
        excluded_by_other_filters = sorted(request_id_set - {str(row["request_id"]) for row in filtered})
        if excluded_by_other_filters:
            _fail(
                "--include-request-id names request IDs that exist in the plan but are excluded by "
                "--include-context-id/--include-request-kind; refusing to silently drop them",
                excluded=excluded_by_other_filters[:8],
                excluded_count=len(excluded_by_other_filters),
            )
    if not filtered:
        _fail(
            "selection is empty: no scoring requests match the requested context(s)/kind(s)/id(s)",
            context_ids=context_ids,
            request_kinds=request_kinds,
            request_ids=sorted(request_id_set) if request_id_set else None,
        )

    sharded = [row for index, row in enumerate(filtered) if index % shard_count == shard_index]
    if not sharded:
        _fail(
            "shard selection is empty for this shard-index; reduce --shard-count or pick a "
            "different --shard-index",
            shard_index=shard_index,
            shard_count=shard_count,
            filtered_count=len(filtered),
        )

    selection = {
        "included_context_ids": context_ids,
        "included_request_kinds": request_kinds,
        "included_request_ids": sorted(requested_request_ids) if requested_request_ids else None,
        "included_request_id_file": (
            {"path": request_id_file[0], "sha256": request_id_file[1]} if request_id_file else None
        ),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "filtered_request_count": len(filtered),
        "shard_request_count": len(sharded),
        "shard_request_ids": [str(row["request_id"]) for row in sharded],
    }
    selection["selection_sha256"] = sha256_json(selection)
    return sharded, selection


def _resolve_request(
    plan: PlanBundle, request: Mapping[str, Any]
) -> tuple[Mapping[str, Any], list[int]]:
    context_id = str(request["context_id"])
    context = plan.contexts_by_id.get(context_id)
    if context is None:
        _fail("scoring request references a context absent from the plan", context_id=context_id)

    kind = str(request["request_kind"])
    if kind in ("primary", "numerical_repeat"):
        candidate_id = str(request.get("candidate_id"))
        candidate = plan.candidates_by_id.get(candidate_id)
        if candidate is None:
            _fail(
                "scoring request references a candidate absent from the plan",
                candidate_id=candidate_id,
            )
        coord_token_ids = [int(v) for v in candidate["coord_token_ids"]]
    elif kind == "sidecar":
        sidecar_id = str(request.get("sidecar_id"))
        sidecar = plan.sidecars_by_id.get(sidecar_id)
        if sidecar is None:
            _fail("scoring request references a sidecar absent from the plan", sidecar_id=sidecar_id)
        coord_token_ids = [int(v) for v in sidecar["coord_token_ids"]]
    else:  # pragma: no cover - guarded by load_plan()/select_requests()
        _fail("unrecognized request_kind", request_kind=kind)

    if len(coord_token_ids) != 4:
        _fail(
            "resolved candidate/sidecar does not have exactly four coordinate tokens",
            request_id=request.get("request_id"),
        )
    return context, coord_token_ids


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        _fail(f"{label} is non-finite; refusing to emit a non-finite score", observed=value)
    return float(value)


def build_score_row(
    *,
    request: Mapping[str, Any],
    context: Mapping[str, Any],
    coord_token_ids: Sequence[int],
    combined: Mapping[str, Any],
) -> dict[str, Any]:
    raw = dict(combined["raw"])
    for slot in scorer.COORD_SLOTS:
        _finite(raw.get(f"{slot}_logprob"), label=f"{request['request_id']}.raw_model_logprob.{slot}_logprob")
    _finite(
        raw.get("complete_box_logprob_sum"),
        label=f"{request['request_id']}.raw_model_logprob.complete_box_logprob_sum",
    )
    request_kind = str(request["request_kind"])
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "request_id": str(request["request_id"]),
        "request_kind": request_kind,
        "context_id": str(context["context_id"]),
        "candidate_id": request.get("candidate_id"),
        "sidecar_id": request.get("sidecar_id"),
        "repeat_index": request.get("repeat_index"),
        "coord_token_ids": [int(v) for v in coord_token_ids],
        "coord_token_ids_sha256": sha256_json([int(v) for v in coord_token_ids]),
        "full_prefix_token_ids_sha256": context["full_prefix_token_ids_sha256"],
        "native_repetition_penalty_stratum": NATIVE_REPETITION_PENALTY_STRATUM,
        "raw_model_logprob": raw,
        "decision_bearing_channel": DECISION_BEARING_CHANNEL,
        "primary_role": request_kind in PRIMARY_ROLE_REQUEST_KINDS,
        "excluded_from_primary_ranks": request_kind not in PRIMARY_ROLE_REQUEST_KINDS,
    }


def _coord_probe_suffixes(coord_token_ids: Sequence[int]) -> list[list[int]]:
    """Relative suffixes ``[x1]``, ``[x1,y1]``, ``[x1,y1,x2]`` a candidate will consume.

    Mirrors ``score_sorted_owner_basin_landscape._candidate_reforward_suffixes``
    / ``score_sorted_fn_fixed_budget._candidate_probe_suffixes``: the root
    (depth 0) forward already yields the x1 distribution, so only the three
    subsequent literal continuations need prefetching.
    """

    tokens = [int(value) for value in coord_token_ids[:3]]
    return [tokens[:depth] for depth in range(1, len(tokens) + 1)]


OpenScalarBackend = Callable[[str, Sequence[int]], "scorer.FullReforwardBackend"]
#: Returns ``(backend, batch_admission)``; ``batch_admission`` always carries
#: at least ``status``/``requested_batch_size``/``effective_batch_size`` --
#: never a silent choice between scalar and batched.
OpenBatchableBackend = Callable[
    [str, Sequence[int]], tuple["scorer.FullReforwardBackend", dict[str, Any]]
]


def score_selected_requests(
    plan: PlanBundle,
    requests: Sequence[Mapping[str, Any]],
    *,
    open_scalar_backend: OpenScalarBackend,
    open_batchable_backend: OpenBatchableBackend,
    attestation: "scorer.AttestationContext",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score every request per context, honoring an explicit per-context batch admission.

    ``primary``/``sidecar`` rows ("batchable") are scored through whatever
    backend ``open_batchable_backend`` returns for that context (scalar, or
    batched only after the caller's own mandatory parity gate admits it).
    ``numerical_repeat`` rows are always scored through a dedicated
    ``open_scalar_backend`` backend, regardless of any batch admission for
    that context's other rows -- the eight-repeat numerical-noise probe must
    never be silently batched.
    """

    by_context: dict[str, list[Mapping[str, Any]]] = {}
    for request in requests:
        by_context.setdefault(str(request["context_id"]), []).append(request)

    rows: list[dict[str, Any]] = []
    per_context_accounting: list[dict[str, Any]] = []
    for context_id in sorted(by_context):
        context_requests = sorted(by_context[context_id], key=lambda row: str(row["request_id"]))
        context = plan.contexts_by_id[context_id]
        full_prefix_token_ids = [int(v) for v in context["full_prefix_token_ids"]]
        batchable = [row for row in context_requests if row["request_kind"] != "numerical_repeat"]
        scalar_only = [row for row in context_requests if row["request_kind"] == "numerical_repeat"]
        context_accounting: dict[str, Any] = {"context_id": context_id}

        if batchable:
            backend, batch_admission = open_batchable_backend(context_id, full_prefix_token_ids)
            effective_batch_size = int(batch_admission.get("effective_batch_size", 1))
            if effective_batch_size < 1:
                _fail(
                    "batch admission declared a non-positive effective_batch_size",
                    context_id=context_id,
                    batch_admission=batch_admission,
                )
            for start in range(0, len(batchable), effective_batch_size):
                chunk = batchable[start : start + effective_batch_size]
                resolved_chunk = [_resolve_request(plan, row) for row in chunk]
                scorer._prefetch_backend_suffixes(  # noqa: SLF001
                    backend,
                    [
                        suffix
                        for _ctx, coord_token_ids in resolved_chunk
                        for suffix in _coord_probe_suffixes(coord_token_ids)
                    ],
                )
                for request, (resolved_context, coord_token_ids) in zip(chunk, resolved_chunk, strict=True):
                    combined = scorer.score_complete_box_candidate(
                        backend=backend,
                        prefill_logits=backend.root_logits,
                        coord_token_ids=coord_token_ids,
                        attestation=attestation,
                        running_context_token_ids=full_prefix_token_ids,
                        repetition_penalties=(),
                    )
                    rows.append(
                        build_score_row(
                            request=request,
                            context=resolved_context,
                            coord_token_ids=coord_token_ids,
                            combined=combined,
                        )
                    )
            context_accounting["batch_admission"] = batch_admission
            context_accounting["batchable_backend_accounting"] = backend.accounting()
            context_accounting["batchable_row_count"] = len(batchable)

        if scalar_only:
            scalar_backend = open_scalar_backend(context_id, full_prefix_token_ids)
            for request in scalar_only:
                resolved_context, coord_token_ids = _resolve_request(plan, request)
                combined = scorer.score_complete_box_candidate(
                    backend=scalar_backend,
                    prefill_logits=scalar_backend.root_logits,
                    coord_token_ids=coord_token_ids,
                    attestation=attestation,
                    running_context_token_ids=full_prefix_token_ids,
                    repetition_penalties=(),
                )
                rows.append(
                    build_score_row(
                        request=request,
                        context=resolved_context,
                        coord_token_ids=coord_token_ids,
                        combined=combined,
                    )
                )
            context_accounting["scalar_only_backend_accounting"] = scalar_backend.accounting()
            context_accounting["scalar_only_row_count"] = len(scalar_only)

        per_context_accounting.append(context_accounting)

    return rows, {"contexts_scored": sorted(by_context), "per_context_accounting": per_context_accounting}


# ---------------------------------------------------------------------------
# Shard receipt + create-or-identical output
# ---------------------------------------------------------------------------


def build_shard_receipt(
    *,
    plan: PlanBundle,
    selection: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    backend_admission: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    code_path = Path(__file__).resolve()
    receipt_time_file_sha256 = sha256_file(code_path)
    counts_by_kind: dict[str, int] = {}
    for row in rows:
        kind = str(row["request_kind"])
        counts_by_kind[kind] = counts_by_kind.get(kind, 0) + 1
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "row_schema_version": SCHEMA_VERSION,
        "code": {
            "path": str(code_path),
            # Import-time fingerprint: the code that actually executed this
            # run, immune to concurrent edits to this file while the
            # process was running. This is the identity merge/attest treat
            # as authoritative.
            "executed_source_sha256": EXECUTED_SOURCE_SHA256,
            # Diagnostic-only live re-read at receipt-build time (end of
            # job); never used for identity/admission decisions.
            "receipt_time_file_sha256": receipt_time_file_sha256,
            "source_drift_detected": receipt_time_file_sha256 != EXECUTED_SOURCE_SHA256,
            # Back-compat alias, always bound to the correct (import-time)
            # value -- never the stale end-of-job read.
            "sha256": EXECUTED_SOURCE_SHA256,
        },
        "plan": {
            "plan_dir": str(plan.plan_dir),
            "receipt_path": str(plan.receipt_path),
            "receipt_sha256": sha256_file(plan.receipt_path),
            "receipt_content_sha256": plan.receipt.get("receipt_content_sha256"),
        },
        "source_identity": dict(source_identity),
        "selection": dict(selection),
        "decision_channel": {
            "name": DECISION_BEARING_CHANNEL,
            "native_repetition_penalty_stratum": NATIVE_REPETITION_PENALTY_STRATUM,
            "note": (
                "raw, unmodified fp32 lm-head log-softmax under a literal use_cache=False full-prefix "
                "reforward, summed over the four chosen coordinate tokens; no repetition-penalty "
                "auxiliary policy view is computed by this scorer"
            ),
        },
        "scoring_backend_admission": dict(backend_admission),
        "environment": dict(environment),
        "counts": {
            "rows": len(rows),
            "rows_by_request_kind": counts_by_kind,
            "primary_role_rows": sum(1 for row in rows if row.get("primary_role")),
            "excluded_from_primary_ranks_rows": sum(
                1 for row in rows if row.get("excluded_from_primary_ranks")
            ),
        },
        "row_ids": sorted(str(row["request_id"]) for row in rows),
    }
    receipt["receipt_content_sha256"] = sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    return receipt


def write_shard(
    output_dir: str | Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    receipt: Mapping[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """Create-or-identical shard write.

    By default an existing shard at ``output_dir`` may only be reused when
    every emitted byte is identical (a safe, idempotent retry of the exact
    same selection). ``force=True`` is the explicit, documented escape hatch
    for replacing a stale/partial shard from a prior failed run -- it never
    triggers silently.
    """

    resolved_dir = Path(output_dir).expanduser().resolve()
    scores_path = resolved_dir / OUTPUT_JSONL_NAME
    receipt_path = resolved_dir / OUTPUT_RECEIPT_NAME
    ordered_rows = sorted(rows, key=lambda row: str(row["request_id"]))
    scores_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in ordered_rows)
    scores_status = _write_create_or_identical(scores_path, scores_bytes, force=force)
    receipt_bytes = canonical_json_bytes(receipt) + b"\n"
    receipt_status = _write_create_or_identical(receipt_path, receipt_bytes, force=force)
    return {
        "scores_path": str(scores_path),
        "scores_status": scores_status,
        "scores_sha256": sha256_file(scores_path),
        "receipt_path": str(receipt_path),
        "receipt_status": receipt_status,
        "receipt_sha256": sha256_file(receipt_path),
    }


# ---------------------------------------------------------------------------
# CLI / live execution
# ---------------------------------------------------------------------------


#: Batch-vs-scalar coordinate parity is admitted only via
#: ``scorer.run_batched_reforward_parity_gate``'s own frozen tolerance
#: (``BATCH_REFORWARD_COORD_LOGPROB_MAX_ABS_DIFF``); this scorer never
#: accepts a caller-supplied epsilon for that admission decision.
ON_BATCH_PARITY_FAILURE_CHOICES = frozenset({"fail", "scalar_fallback"})


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = load_plan(args.plan_dir)

    combined_request_ids = list(args.include_request_id or [])
    request_id_file_binding: tuple[str, str] | None = None
    if args.include_request_id_file is not None:
        file_path = args.include_request_id_file.expanduser().resolve(strict=True)
        file_request_ids = _load_request_id_file(file_path)
        overlap = set(combined_request_ids) & set(file_request_ids)
        if overlap:
            _fail(
                "--include-request-id and --include-request-id-file name overlapping request ids",
                overlap=sorted(overlap)[:8],
            )
        combined_request_ids = combined_request_ids + file_request_ids
        request_id_file_binding = (str(file_path), sha256_file(file_path))

    selected, selection = select_requests(
        plan,
        include_context_ids=args.include_context_id,
        include_request_kinds=args.include_request_kind,
        include_request_ids=combined_request_ids or None,
        request_id_file=request_id_file_binding,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )

    if args.full_reforward_batch_size < 1:
        _fail(
            "--full-reforward-batch-size must be a positive integer",
            requested_batch_size=args.full_reforward_batch_size,
        )
    if args.on_batch_parity_failure not in ON_BATCH_PARITY_FAILURE_CHOICES:
        _fail(
            "--on-batch-parity-failure must be one of the recognized choices",
            observed=args.on_batch_parity_failure,
            choices=sorted(ON_BATCH_PARITY_FAILURE_CHOICES),
        )

    if args.validate_contract_only:
        return {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "runtime_execution_status": "contract_validated_no_gpu",
            "plan_receipt_content_sha256": plan.receipt.get("receipt_content_sha256"),
            "selection": selection,
        }

    if args.output_dir is None:
        _fail("live scoring requires --output-dir")
    if args.infer_config is None or args.source_jsonl is None:
        _fail("live scoring requires --infer-config and --source-jsonl")

    import torch as _torch

    if not _torch.cuda.is_available():
        _fail("CUDA is required for live scoring; pass an explicit --cuda-device")
    if not args.cuda_device:
        _fail("live scoring requires an explicit --cuda-device")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device.replace("cuda:", "")

    scorer.pin_fp32_parity_flags()

    from src.config.fingerprint import sha256_json as config_sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import DecodeRequest, GenerationPolicy, open_backend_session
    from src.inference.hf_backend import HFBackendSession
    from src.inference.image_plan import plan_image_batch
    from src.inference.pipeline import _processor_config, _template_config
    from src.inference.prompt import build_prompt_record
    from src.inference.runtime import assemble_frontend

    resolved = load_infer_config(args.infer_config)
    config = resolved.config
    generation_config_fingerprint = config_sha256_json(config.generation.model_dump(mode="json"))
    frontend = assemble_frontend(config, generation_config_fingerprint=generation_config_fingerprint)

    raw_rows = load_raw_examples(args.source_jsonl)
    raw = next(
        (
            row
            for row in raw_rows
            if str(_mapping(row.metadata.get("source"), "source row metadata.source").get("image_id"))
            == planner.IMAGE_ID
        ),
        None,
    )
    if raw is None:
        _fail(f"--source-jsonl does not contain a row for image_id={planner.IMAGE_ID!r}")
    template = _template_config(config)

    runtime_receipt_id = sha256_json(
        {"command": list(sys.argv), "selection": selection, "started_at_unix": time.time()}
    )

    rows: list[dict[str, Any]] = []
    backend_admission: dict[str, Any] = {}
    environment: dict[str, Any] = {}
    source_identity: dict[str, Any] = {}

    with open_backend_session(frontend.launch) as opened:
        if not isinstance(opened, HFBackendSession):
            raise RuntimeError("HF launch opened an unexpected backend session")
        backend_receipt = opened.receipt.to_artifact_dict()
        tokenizer_identity = backend_receipt.get("tokenizer_identity") or {}
        model_identity = backend_receipt.get("model_identity") or {}
        model = opened._model  # noqa: SLF001
        if model is None:
            _fail("HF backend session did not expose its opened model")
        model.eval()

        image_plan = plan_image_batch(
            [raw], components=frontend.qwen, processor_config=_processor_config(config), row_indices=[0]
        ).rows[0]
        prompt_record = build_prompt_record(
            raw,
            template,
            processor=frontend.qwen.processor,
            row_index=0,
            merged_visual_tokens=image_plan.merged_visual_tokens,
        )
        expected_prompt_token_ids = tuple(int(v) for v in prompt_record.prompt_token_ids)
        image_grid_thw = tuple(int(v) for v in image_plan.expected_image_grid_thw)
        if len(image_grid_thw) != 3:
            _fail("planned image_grid_thw must contain exactly three dimensions", observed=image_grid_thw)

        decode_request = DecodeRequest(
            request_id=f"route-landscape-score:{planner.IMAGE_ID}",
            chat_text=prompt_record.chat_text,
            input_prompt_token_ids=tuple(prompt_record.input_prompt_token_ids),
            expected_executed_prompt_token_ids=tuple(prompt_record.expected_executed_prompt_token_ids),
            image_path=image_plan.image_path,
            declared_image_width=image_plan.declared_width,
            declared_image_height=image_plan.declared_height,
            decoded_image_width=image_plan.decoded_width,
            decoded_image_height=image_plan.decoded_height,
            image_sha256=image_plan.image_content_sha256,
            expected_image_grid_thw=(image_grid_thw[0], image_grid_thw[1], image_grid_thw[2]),
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
            raw_executed_media_sha256,
        ) = opened._materialize_native_inputs((decode_request,))  # noqa: SLF001
        if tuple(executed_prompt_ids[0]) != expected_prompt_token_ids:
            _fail("materialized executed prompt tokens differ from the expected production prompt reconstruction")
        (executed_media_sha256,) = _normalized_executed_media_sha256(
            raw_executed_media_sha256, expected_count=1
        )
        if executed_media_sha256 != planner.EXECUTED_MEDIA_SHA256:
            _fail(
                "live executed media digest differs from the unit's frozen executed_media_sha256",
                observed=executed_media_sha256,
                expected=planner.EXECUTED_MEDIA_SHA256,
            )

        for context_id in {str(row["context_id"]) for row in selected}:
            context = plan.contexts_by_id[context_id]
            declared_prompt = tuple(int(v) for v in context["prompt_token_ids"])
            if declared_prompt != expected_prompt_token_ids:
                _fail(
                    f"plan context {context_id!r} prompt token ids do not match the live materialized "
                    "production prompt"
                )

        full_reforward = scorer._build_full_reforward_closure(model, native_prompt_inputs=native_inputs)  # noqa: SLF001

        def _open_scalar_backend(
            context_id: str, full_prefix_token_ids: Sequence[int]
        ) -> "scorer.FullReforwardBackend":
            return scorer.FullReforwardBackend(
                root_prefix_token_ids=full_prefix_token_ids,
                full_reforward=full_reforward,
                context_id=context_id,
                group_id=sha256_json(
                    {
                        "image_id": planner.IMAGE_ID,
                        "context_id": context_id,
                        "prefix_token_ids": list(full_prefix_token_ids),
                    }
                ),
            )

        # Batched materialization/closure are built at most once, lazily, and
        # reused across every context: this unit is single-image, so N
        # identical copies of the same decode request are sufficient for
        # every context's batched probe and chunked scoring.
        requested_batch_size = int(args.full_reforward_batch_size)
        batched_full_reforward: Callable[[Sequence[Sequence[int]]], Any] | None = None
        if requested_batch_size > 1:
            (
                batched_native_inputs,
                batch_executed_prompt_ids,
                _batch_grids,
                raw_batch_media_sha256,
            ) = opened._materialize_native_inputs(  # noqa: SLF001
                tuple(decode_request for _ in range(requested_batch_size))
            )
            if any(tuple(row) != expected_prompt_token_ids for row in batch_executed_prompt_ids):
                _fail("batched materialization changed the executed production prompt")
            (batch_media_sha256,) = set(
                _normalized_executed_media_sha256(
                    raw_batch_media_sha256, expected_count=requested_batch_size
                )
            )
            if batch_media_sha256 != executed_media_sha256:
                _fail(
                    "batched materialization changed the canonical executed RGB identity",
                    observed=batch_media_sha256,
                    expected=executed_media_sha256,
                )
            batched_full_reforward = scorer._build_batched_full_reforward_closure(  # noqa: SLF001
                model,
                native_prompt_inputs=batched_native_inputs,
                maximum_batch_size=requested_batch_size,
            )

        def _open_batchable_backend(
            context_id: str, full_prefix_token_ids: Sequence[int]
        ) -> tuple["scorer.FullReforwardBackend", dict[str, Any]]:
            if requested_batch_size <= 1 or batched_full_reforward is None:
                admission = {
                    "schema_version": "batched_full_reforward_parity.v1",
                    "status": "not_requested",
                    "requested_batch_size": 1,
                    "effective_batch_size": 1,
                }
                return _open_scalar_backend(context_id, full_prefix_token_ids), admission

            context_requests = [row for row in selected if str(row["context_id"]) == context_id]
            probe_request = min(
                (row for row in context_requests if row["request_kind"] != "numerical_repeat"),
                key=lambda row: str(row["request_id"]),
            )
            _probe_context, probe_coord_token_ids = _resolve_request(plan, probe_request)
            admission = scorer.run_batched_reforward_parity_gate(
                prefix_token_ids=full_prefix_token_ids,
                coordinate_token_ids=probe_coord_token_ids,
                coordinate_token_id_start=planner.COORD_TOKEN_START,
                coordinate_token_id_end_exclusive=planner.COORD_TOKEN_END + 1,
                full_reforward=full_reforward,
                batched_full_reforward=batched_full_reforward,
                requested_batch_size=requested_batch_size,
            )
            admission = {**admission, "probe_context_id": context_id, "probe_request_id": probe_request["request_id"]}
            if admission["status"] == "passed":
                backend = scorer.FullReforwardBackend(
                    root_prefix_token_ids=full_prefix_token_ids,
                    full_reforward=full_reforward,
                    batched_full_reforward=batched_full_reforward,
                    full_reforward_batch_size=requested_batch_size,
                    context_id=context_id,
                    group_id=sha256_json(
                        {
                            "image_id": planner.IMAGE_ID,
                            "context_id": context_id,
                            "prefix_token_ids": list(full_prefix_token_ids),
                            "batch_size": requested_batch_size,
                        }
                    ),
                )
                return backend, admission
            if args.on_batch_parity_failure == "fail":
                _fail(
                    "batched full-reforward parity failed and --on-batch-parity-failure=fail; "
                    "refusing to silently fall back to scalar",
                    context_id=context_id,
                    admission=admission,
                )
            # Explicit, recorded scalar fallback -- never silent.
            fallback_admission = {**admission, "fallback": "scalar_explicit"}
            return _open_scalar_backend(context_id, full_prefix_token_ids), fallback_admission

        source_identity = {
            "model_identity_sha256": sha256_json(dict(model_identity)),
            "tokenizer_identity_sha256": sha256_json(dict(tokenizer_identity)),
            "executed_media_sha256": executed_media_sha256,
            "infer_config": {"path": str(args.infer_config), "sha256": sha256_file(args.infer_config)},
            "source_jsonl": {"path": str(args.source_jsonl), "sha256": sha256_file(args.source_jsonl)},
        }
        attestation = scorer.build_attestation_context(
            expected_vocab_size=int(model.get_output_embeddings().weight.shape[0]),
            tokenizer_identity=tokenizer_identity,
            model_identity=model_identity,
            rule_digest=str(plan.receipt.get("receipt_content_sha256", "")),
            runtime_receipt_id=runtime_receipt_id,
        )
        rows, scoring_meta = score_selected_requests(
            plan,
            selected,
            open_scalar_backend=_open_scalar_backend,
            open_batchable_backend=_open_batchable_backend,
            attestation=attestation,
        )
        backend_admission = {
            "selected_backend": scorer.FULL_REFORWARD_SCORING_BACKEND,
            "cache_enabled": False,
            "use_cache": False,
            "requested_batch_size": requested_batch_size,
            "on_batch_parity_failure": args.on_batch_parity_failure,
            **scoring_meta,
        }
        environment = {
            "python_version": sys.version,
            "torch_version": _torch.__version__,
            "transformers_version": scorer._installed_package_version("transformers"),  # noqa: SLF001
            "cuda_available": _torch.cuda.is_available(),
            "device_name": _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else None,
            "tf32": scorer.pin_fp32_parity_flags(),
        }

    receipt = build_shard_receipt(
        plan=plan,
        selection=selection,
        rows=rows,
        backend_admission=backend_admission,
        source_identity=source_identity,
        environment=environment,
    )
    outcome = write_shard(args.output_dir, rows=rows, receipt=receipt, force=args.force)
    return {"selection": selection, "counts": {"rows": len(rows)}, **outcome}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--include-context-id", action="append", default=None)
    parser.add_argument(
        "--include-request-id",
        action="append",
        default=None,
        help=(
            "repeatable exact request_id selector (e.g. for scalar posthoc rescoring of specific "
            "rank-critical rows); every id must exist in the plan's full request domain and must "
            "also survive any --include-context-id/--include-request-kind filter, or the run fails "
            "closed rather than silently dropping it"
        ),
    )
    parser.add_argument(
        "--include-request-id-file",
        type=Path,
        default=None,
        help=(
            "file with one exact request_id per line (or one JSONL object with a request_id field per "
            "line), for a deterministic scalar-confirmation manifest without hundreds of repeated "
            "--include-request-id flags; combined with --include-request-id (duplicates/overlap fail "
            "closed) and validated against the plan the same way; the file's own path/sha256 is bound "
            "into selection.included_request_id_file"
        ),
    )
    parser.add_argument(
        "--include-request-kind", action="append", default=None, choices=sorted(REQUEST_KINDS)
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--full-reforward-batch-size", type=int, default=1)
    parser.add_argument(
        "--on-batch-parity-failure",
        choices=sorted(ON_BATCH_PARITY_FAILURE_CHOICES),
        default="fail",
        help=(
            "when --full-reforward-batch-size > 1 and a context's batched-vs-scalar coordinate "
            "parity probe fails: 'fail' aborts the run (default); 'scalar_fallback' explicitly "
            "records the failed admission and re-scores that context scalar. Never silent."
        ),
    )
    parser.add_argument("--infer-config", type=Path, default=None)
    parser.add_argument("--source-jsonl", type=Path, default=None)
    parser.add_argument("--cuda-device", type=str, default=None)
    parser.add_argument("--validate-contract-only", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing, non-identical shard at --output-dir (safe retry of a failed/partial run)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
