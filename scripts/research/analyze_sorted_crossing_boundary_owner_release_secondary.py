#!/usr/bin/env python3
"""Secondary downstream-compatibility analysis for the sorted crossing-boundary
owner release/realization unit
(``2026-08-03-sorted-crossing-boundary-owner-release-realization``).

Frozen unit::

    research/investigations/qwen3-vl-dense-enumeration/experiments/
    2026-08-03-sorted-crossing-boundary-owner-release-realization/
    {unit.md,tasks.md}  -- "Secondary downstream compatibility"

What this module does
---------------------
It reads **only** the sealed secondary merge published by
``merge_sorted_crossing_boundary_owner_release_secondary``, plus the *digest
binding* of the finalized primary analysis that gated the capture, and renders
the readout ``unit.md`` asks for:

* one owner row per readout owner, preserving **every** per-segment exact
  paired delta (``description``, ``coordinates``, ``complete_row``) for the
  ``P+C -> E`` readout and, where the plan materialised it, the optional
  ``P+E+C -> F`` late catch-up;
* the 26 U-bound crossing owners joined by ``gt_owner_id`` / ``image_id`` /
  ``cohort``, and the 12 native-TP replay controls kept **separately**;
* per variant and segment: finite coverage and positive / zero / negative sign
  counts -- discrete counts over owners -- beside each image's own raw values
  and its within-image median; and
* the benign-substitution readouts, one per image, as an explicit *per-image*
  empirical reference a crossing owner is read beside **within its own image**.

The one aggregation rule
------------------------
``unit.md`` "Secondary downstream compatibility": these deltas are "interpreted
only relative to that reference distribution and never pooled across images as
raw values".  This module takes that literally.  Aggregating over owners is
allowed **only** for discrete counts -- coverage denominators and sign counts.
Raw deltas stay indexed by image.

There is therefore no cross-image median, minimum, maximum, quantile or ordered
raw-value array anywhere in this artifact family, for the primary variants or
for the reference controls.  Every owner-level delta and every per-image list is
published, so a reader who wants a pooled number can compute one -- and owns
that step, rather than finding it pre-computed and reading it as a result.

What this module never does
---------------------------
``unit.md`` puts every branch decision in the sealed primary pass.  So this
analyzer:

* reads no branch label.  The sealed primary analysis is opened only for its
  file digests and artifact schema versions; its owner rows are hashed, never
  interpreted, and nothing derived from them enters an output;
* fits no cutoff, assigns no compatibility class, and emits no promotion
  criterion, routing outcome or successor;
* makes no retention, eventual-recovery, natural-stop or free-rollout claim.

Those boundaries are enforced, not merely stated: every emitted payload is run
through :func:`assert_no_decision_surface`, which fails closed if any key at any
depth names a branch, a cutoff, a verdict or a routing outcome -- in the merged
input as much as in the analysis output.

One row per request in, one row per owner out
---------------------------------------------
The merged input holds one row per executed request; this module emits one row
per owner carrying that owner's readouts side by side.  That is a
transformation, not a verbatim copy, and it is declared as one in
``row_mapping`` / the receipt policy.  The invariant that matters is checked
rather than asserted: :func:`build_row_mapping` fails closed unless the owner
join carries through exactly as many readouts as it consumed, and every
per-segment sum, delta, token-mean and sign is copied unchanged.

Outputs (one explicit analysis directory, published create-or-identical)::

    secondary-owner-rows.jsonl      one row per readout owner, per cohort
    secondary-summary.json          denominators, coverage, signs, per-image values
    secondary-report.md             the same content, rendered
    secondary-analysis-receipt.json input/output digests, self-sealed
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import (  # noqa: E402
    analyze_sorted_crossing_boundary_owner_release as primary_analyzer,
)
from scripts.research import (  # noqa: E402
    merge_sorted_crossing_boundary_owner_release as primary_merge,
)
from scripts.research import (  # noqa: E402
    merge_sorted_crossing_boundary_owner_release_secondary as secondary_merge,
)
from scripts.research import (  # noqa: E402
    score_sorted_crossing_boundary_owner_release as primary,
)
from scripts.research import (  # noqa: E402
    score_sorted_crossing_boundary_owner_release_secondary as secondary,
)

# ---------------------------------------------------------------------------
# 0. Frozen identities
# ---------------------------------------------------------------------------

UNIT_ID = secondary.UNIT_ID

OWNER_ROW_SCHEMA_VERSION = "sorted-crossing-boundary-secondary-owner-row.v1"
SUMMARY_SCHEMA_VERSION = "sorted-crossing-boundary-secondary-summary.v1"
RECEIPT_SCHEMA_VERSION = "sorted-crossing-boundary-secondary-analysis-receipt.v1"

OWNER_ROWS_NAME = "secondary-owner-rows.jsonl"
SUMMARY_NAME = "secondary-summary.json"
REPORT_MD_NAME = "secondary-report.md"
RECEIPT_NAME = "secondary-analysis-receipt.json"

MERGED_ROWS_NAME = secondary_merge.MERGED_ROWS_NAME
MERGED_PARITY_NAME = secondary_merge.MERGED_PARITY_NAME
MERGE_RECEIPT_NAME = secondary_merge.MERGE_RECEIPT_NAME
MERGED_REQUIRED_FILES: tuple[str, ...] = (
    MERGED_ROWS_NAME,
    MERGED_PARITY_NAME,
    MERGE_RECEIPT_NAME,
)

SEGMENTS: tuple[str, ...] = secondary.SEGMENTS
SECONDARY_VARIANTS: tuple[str, ...] = secondary.SECONDARY_VARIANTS
PRIMARY_COHORT = secondary_merge.PRIMARY_COHORT
TP_REPLAY_CONTROL_COHORT = secondary_merge.TP_REPLAY_CONTROL_COHORT

#: The two variants that read out on a primary crossing owner, in report order.
PRIMARY_VARIANTS: tuple[str, ...] = (
    secondary.VARIANT_P_PLUS_C_THEN_E,
    secondary.VARIANT_P_PLUS_E_PLUS_C_THEN_F,
)
REFERENCE_VARIANT = secondary.VARIANT_BENIGN_SUBSTITUTION

SIGN_POSITIVE = "positive"
SIGN_ZERO = "zero"
SIGN_NEGATIVE = "negative"
SIGN_LABELS: tuple[str, ...] = (SIGN_POSITIVE, SIGN_ZERO, SIGN_NEGATIVE)

#: The v2 primary artifact revision this readout may sit beside.  A secondary
#: readout is only interpretable next to the branch registry that gated it, so
#: any other analysis revision fails closed instead of being reconciled.
PRIMARY_RECEIPT_SCHEMA_VERSION = primary_analyzer.RECEIPT_SCHEMA_VERSION
PRIMARY_REPORT_SCHEMA_VERSION = primary_analyzer.REPORT_SCHEMA_VERSION
PRIMARY_ANALYSIS_REQUIRED_FILES: tuple[str, ...] = secondary.ANALYSIS_REQUIRED_FILES

#: No key of any payload this module reads or writes may name a decision
#: surface.  ``unit.md`` keeps branch assignment, routing and every promotion
#: criterion in the sealed primary pass; this readout is descriptive only.
FORBIDDEN_KEY_FRAGMENTS: tuple[str, ...] = (
    "branch",
    "threshold",
    "cutoff",
    "verdict",
    "decision",
    "routing",
    "successor_route",
    "retention",
    "recovery",
    "rollout",
    "launch",
)

#: No key of any payload this module *emits* may name a pooled location or
#: range statistic over raw deltas.  ``unit.md`` forbids pooling them across
#: images as raw values, so the prohibition is enforced rather than reviewed.
POOLED_STATISTIC_KEY_FRAGMENTS: tuple[str, ...] = (
    "median",
    "mean",
    "quantile",
    "quartile",
    "percentile",
    "minimum",
    "maximum",
    "stdev",
    "variance",
    "iqr",
)

#: Short pooled-statistic spellings that are too generic to match as substrings
#: (``q1`` would hit an unrelated ``seq1``), so they are blocked by exact name.
POOLED_STATISTIC_KEY_NAMES: frozenset[str] = frozenset(
    {"q1", "q3", "quartiles", "min", "max", "range", "spread"}
)

#: The complete list of statistic keys whose scope is a single owner or a single
#: image by construction, and which are therefore not pooled quantities:
#:
#: * ``delta_token_mean`` -- one owner's own delta divided by its own token
#:   count, a field the producer sealed and this module copies unchanged; and
#: * ``within_image_median`` -- computed over one image's owners only.
WITHIN_SCOPE_STATISTIC_KEYS: frozenset[str] = frozenset(
    {"delta_token_mean", "within_image_median"}
)

#: ``unit.md`` "Secondary downstream compatibility": secondary deltas are
#: "never pooled across images as raw values".  Stamped beside every block that
#: aggregates over owners, so a reader can see the boundary at the point of use
#: rather than having to find it in a policy footer.
NO_CROSS_IMAGE_POOLING = (
    "aggregated over owners as discrete counts only; raw deltas are reported per image and "
    "never pooled across images into one location, range or quantile"
)

#: ``unit.md`` "Secondary downstream compatibility" and "Not claimed", written
#: verbatim into every output so no reader has to reconstruct the boundary.
NOT_CLAIMED: tuple[str, ...] = (
    "no compatibility class, cutoff or promotion criterion is derived from these deltas",
    "no primary branch is read, recomputed or rerouted here",
    "no final-set retention, eventual recovery, natural stop or free-rollout result",
    "no causal effect of inserting the exact clean GT row for the target owner",
    "the exact GT row is an oracle intervention; its local compatibility does not imply the "
    "model could naturally generate it",
    "raw log probabilities are never pooled across images; every reported quantity is a "
    "within-owner, within-target paired difference on byte-identical forced tokens",
    "no single cross-image effect size is reported: there is no pooled median, minimum, "
    "maximum or quantile of these deltas anywhere in this artifact family",
)

OPERATIONAL_DEFINITIONS: Mapping[str, str] = {
    "paired_delta": (
        "modified-minus-baseline sum of selected-token log probabilities over one literal "
        "segment of one exact native row, both roots forcing byte-identical tokens"
    ),
    "segments": (
        "description is the row through <|box_start|>, coordinates is the frozen four-token "
        "coordinate slot, complete_row is everything through <|box_end|>"
    ),
    "finite_coverage": (
        "the share of owners whose segment delta is a finite number; the capture and merge "
        "already fail closed on a non-finite value, so this is an invariant readout"
    ),
    "sign_counts": (
        "counts of strictly positive, exactly zero and strictly negative paired deltas; the "
        "sign is the producer's own sealed field, recomputed here from the same delta"
    ),
    "within_image_median": (
        "median of one image's own within-owner paired deltas; an image-local location "
        "statistic computed only over that image, never a pooled cross-image raw value"
    ),
    "benign_substitution_reference": (
        "the native-TP replay controls, one per image, each replacing a natively emitted "
        "true-positive row with its exact clean GT twin; reported per image as the empirical "
        "reference a crossing owner's delta is read beside within its own image, never as a "
        "pooled distribution and never as a fitted criterion"
    ),
    "no_pooled_effect_size": (
        "no cross-image median, minimum, maximum, quantile or ordered raw-value array is "
        "emitted; owner-level deltas and per-image lists are published so a reader who wants "
        "one pools it themselves and owns that step"
    ),
    "optional_f_presence": (
        "taken from the merged rows the sealed plan materialised; an owner without a sealed F "
        "row simply carries no late catch-up readout"
    ),
}


class SecondaryAnalysisContractError(RuntimeError):
    """A precondition of this unit's secondary compatibility analysis failed."""


def _fail(message: str) -> NoReturn:
    raise SecondaryAnalysisContractError(message)


canonical_json_bytes = secondary.canonical_json_bytes
sha256_bytes = secondary.sha256_bytes
sha256_json = secondary.sha256_json
sha256_file = secondary.sha256_file


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not Path(path).is_file():
        _fail(f"{label} is missing at {path}")
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{label} is unreadable at {path}: {exc}")
    if not isinstance(value, Mapping):
        _fail(f"{label} at {path} is not a JSON object")
    return dict(value)


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        _fail(f"{label} is missing at {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"{label} line {number} is not valid JSON: {exc}")
        if not isinstance(value, Mapping):
            _fail(f"{label} line {number} is not a JSON object")
        rows.append(dict(value))
    return rows


def assert_self_sealed(payload: Mapping[str, Any], *, digest_key: str, label: str) -> str:
    """A sealed artifact must reconstruct its own digest from its own content.

    Re-stated here rather than borrowed from the merge module so every failure
    this analyzer reports carries its own error type.
    """

    declared = payload.get(digest_key)
    if not isinstance(declared, str) or not declared:
        _fail(f"{label} declares no {digest_key}")
    reconstructed = sha256_json(
        {key: value for key, value in payload.items() if key != digest_key}
    )
    if reconstructed != declared:
        _fail(
            f"{label} does not reconstruct its own {digest_key}; it was edited after it "
            "was sealed"
        )
    return declared


def assert_no_decision_surface(payload: Any, *, label: str) -> None:
    """Refuse any key, at any depth, that names a decision surface.

    Applied to the merged input and to every emitted payload.  A branch label
    arriving in a secondary row would mean the sealed primary pass leaked into
    this readout; a branch, cutoff or routing key leaving this module would mean
    the readout had quietly become the decision ``unit.md`` reserves for the
    primary pass.
    """

    if isinstance(payload, Mapping):
        offending = sorted(
            str(key)
            for key in payload
            if any(fragment in str(key).lower() for fragment in FORBIDDEN_KEY_FRAGMENTS)
        )
        if offending:
            _fail(
                f"{label} carries decision-surface key(s) {offending!r}; this readout is "
                "descriptive and never assigns, reroutes or gates anything"
            )
        for value in payload.values():
            assert_no_decision_surface(value, label=label)
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for value in payload:
            assert_no_decision_surface(value, label=label)


def assert_no_pooled_raw_summary(payload: Any, *, label: str) -> None:
    """Refuse any key that would publish a pooled cross-image raw statistic.

    ``unit.md`` forbids pooling these deltas across images as raw values, so the
    prohibition is enforced on what this module *emits* rather than left to
    review.  Adding a ``median_delta``, ``q1`` or ``maximum`` back to a summary
    block fails closed here instead of shipping.

    :data:`WITHIN_SCOPE_STATISTIC_KEYS` is the complete, deliberately short list
    of exceptions: each names a statistic whose scope is a single owner or a
    single image by construction, never a pool of two images.
    """

    if isinstance(payload, Mapping):
        offending = sorted(
            str(key)
            for key in payload
            if str(key) not in WITHIN_SCOPE_STATISTIC_KEYS
            and (
                str(key).lower() in POOLED_STATISTIC_KEY_NAMES
                or any(
                    fragment in str(key).lower()
                    for fragment in POOLED_STATISTIC_KEY_FRAGMENTS
                )
            )
        )
        if offending:
            _fail(
                f"{label} carries pooled-statistic key(s) {offending!r}; raw deltas are "
                "reported per image and never pooled across images into one location, range "
                "or quantile"
            )
        for value in payload.values():
            assert_no_pooled_raw_summary(value, label=label)
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for value in payload:
            assert_no_pooled_raw_summary(value, label=label)


def assert_emitted_payload(payload: Any, *, label: str) -> None:
    """Both emission guards, so neither can be applied without the other."""

    assert_no_decision_surface(payload, label=label)
    assert_no_pooled_raw_summary(payload, label=label)


# ---------------------------------------------------------------------------
# 1. The sealed secondary merge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergedSecondary:
    """The sealed merged secondary evidence, re-proven from its own bytes."""

    merged_dir: Path
    receipt: dict[str, Any]
    rows: list[dict[str, Any]]
    file_sha256: dict[str, str]

    @property
    def receipt_content_sha256(self) -> str:
        return str(self.receipt["receipt_content_sha256"])

    @property
    def primary_analysis_binding_sha256(self) -> str:
        return str(self.receipt["primary_analysis_binding_sha256"])


def load_merged_secondary(merged_dir: Path) -> MergedSecondary:
    """Re-prove the merged directory's shape, seal, digests and counters."""

    merged_dir = Path(merged_dir)
    if not merged_dir.is_dir():
        _fail(f"merged secondary directory {merged_dir} does not exist")
    present = sorted(entry.name for entry in merged_dir.iterdir())
    missing = sorted(set(MERGED_REQUIRED_FILES) - set(present))
    unknown = sorted(set(present) - set(MERGED_REQUIRED_FILES))
    if missing:
        _fail(f"merged secondary directory {merged_dir} is incomplete; missing {missing!r}")
    if unknown:
        _fail(
            f"merged secondary directory {merged_dir} carries unknown artifact(s) {unknown!r}; "
            "an unrecognised file inside a sealed directory fails closed"
        )
    file_sha256 = {
        name: sha256_file(merged_dir / name) for name in sorted(MERGED_REQUIRED_FILES)
    }

    receipt = _read_json(merged_dir / MERGE_RECEIPT_NAME, "secondary merge receipt")
    if str(receipt.get("schema_version")) != secondary_merge.MERGE_SCHEMA_VERSION:
        _fail(
            f"secondary merge receipt schema {receipt.get('schema_version')!r} is not "
            f"{secondary_merge.MERGE_SCHEMA_VERSION!r}"
        )
    if str(receipt.get("unit_id")) != UNIT_ID:
        _fail("secondary merge receipt belongs to another unit")
    assert_self_sealed(
        receipt, digest_key="receipt_content_sha256", label="secondary merge receipt"
    )

    declared_outputs = receipt.get("output_file_digests")
    if not isinstance(declared_outputs, Mapping):
        _fail("secondary merge receipt carries no output_file_digests")
    for name in (MERGED_ROWS_NAME, MERGED_PARITY_NAME):
        entry = declared_outputs.get(name)
        if not isinstance(entry, Mapping):
            _fail(f"secondary merge receipt declares no digest for {name}")
        if str(entry.get("sha256")) != file_sha256[name]:
            _fail(
                f"merged file {name} hashes to {file_sha256[name]}, not the merge receipt's "
                f"{entry.get('sha256')}; the merged evidence drifted after it was sealed"
            )
        observed_size = (merged_dir / name).stat().st_size
        if int(entry.get("byte_size", -1)) != observed_size:
            _fail(
                f"merged file {name} is {observed_size} bytes, not the merge receipt's "
                f"{entry.get('byte_size')}"
            )

    rows = _read_jsonl(merged_dir / MERGED_ROWS_NAME, "merged secondary rows")
    if len(rows) != int(declared_outputs[MERGED_ROWS_NAME].get("row_count", -1)):
        _fail(
            f"merged secondary rows hold {len(rows)} rows, not the merge receipt's "
            f"{declared_outputs[MERGED_ROWS_NAME].get('row_count')!r}"
        )
    assert_no_decision_surface(rows, label="merged secondary rows")

    census = receipt.get("secondary_requests")
    if not isinstance(census, Mapping):
        _fail("secondary merge receipt seals no secondary_requests census")
    request_ids = sorted(str(row.get("request_id")) for row in rows)
    if len(set(request_ids)) != len(request_ids):
        _fail("the merged secondary rows carry a duplicated request id")
    if sha256_json(request_ids) != str(census.get("request_ids_sha256")):
        _fail(
            "the merged secondary rows do not reproduce the exact request-id set digest their "
            "own merge receipt sealed"
        )
    observed = secondary.counts_by_variant(rows)
    declared_counts = census.get("counts_by_variant")
    if not isinstance(declared_counts, Mapping) or {
        str(key): int(value) for key, value in declared_counts.items()
    } != observed:
        _fail(
            f"the merged secondary rows hold {observed!r} rows, not the census "
            f"{declared_counts!r} their own merge receipt sealed"
        )
    return MergedSecondary(
        merged_dir=merged_dir,
        receipt=receipt,
        rows=rows,
        file_sha256=file_sha256,
    )


# ---------------------------------------------------------------------------
# 2. The finalized primary analysis, bound by digest only
# ---------------------------------------------------------------------------


def _schema_version_of(path: Path, label: str) -> str:
    """Read *only* a JSON artifact's declared schema version.

    The sealed primary analysis is a gate, not an input: its bytes are already
    pinned by digest, and this reads the one field that makes the pinned
    revision legible to a human reader.  Nothing else in the payload is kept.
    """

    return str(_read_json(path, label).get("schema_version"))


def bind_primary_analysis(
    analysis_dir: Path, *, merged: MergedSecondary
) -> dict[str, Any]:
    """Prove the supplied analysis directory is the exact v2 gate the merge bound."""

    analysis_dir = Path(analysis_dir)
    file_sha256 = secondary._assert_analysis_directory_shape(analysis_dir)  # noqa: SLF001
    sealed = merged.receipt.get("primary_analysis")
    if not isinstance(sealed, Mapping):
        _fail("the secondary merge receipt seals no primary_analysis binding")
    declared_files = sealed.get("analysis_file_sha256")
    if not isinstance(declared_files, Mapping):
        _fail("the secondary merge receipt seals no primary analysis file digests")
    if {str(key): str(value) for key, value in declared_files.items()} != dict(
        sorted(file_sha256.items())
    ):
        _fail(
            f"the primary analysis at {analysis_dir} does not hash to the gate the secondary "
            "merge bound; the branch registry drifted after the secondary capture"
        )

    receipt_schema = _schema_version_of(
        analysis_dir / secondary.ANALYSIS_RECEIPT_NAME, "primary analysis receipt"
    )
    if receipt_schema != PRIMARY_RECEIPT_SCHEMA_VERSION:
        _fail(
            f"the bound primary analysis receipt is schema {receipt_schema!r}, not "
            f"{PRIMARY_RECEIPT_SCHEMA_VERSION!r}"
        )
    report_schema = _schema_version_of(
        analysis_dir / secondary.ANALYSIS_REPORT_JSON_NAME, "primary analysis report"
    )
    if report_schema != PRIMARY_REPORT_SCHEMA_VERSION:
        _fail(
            f"the bound primary analysis report is schema {report_schema!r}, not the finalized "
            f"{PRIMARY_REPORT_SCHEMA_VERSION!r}"
        )
    return {
        "analysis_dir": str(analysis_dir),
        "analysis_file_sha256": dict(sorted(file_sha256.items())),
        "receipt_schema_version": receipt_schema,
        "report_schema_version": report_schema,
        "receipt_content_sha256": str(sealed.get("receipt_content_sha256")),
        "analyzer_source_sha256": str(sealed.get("analyzer_source_sha256")),
        "binding_sha256": merged.primary_analysis_binding_sha256,
        "read_scope": (
            "artifact digests and schema versions only; no owner row, label or routing field of "
            "the sealed primary pass is opened, carried or re-derived here"
        ),
    }


# ---------------------------------------------------------------------------
# 3. Owner rows
# ---------------------------------------------------------------------------


def _delta_block(row: Mapping[str, Any], segment: str, *, label: str) -> dict[str, Any]:
    deltas = row.get("deltas")
    if not isinstance(deltas, Mapping):
        _fail(f"{label} carries no per-segment deltas")
    block = deltas.get(segment)
    if not isinstance(block, Mapping):
        _fail(f"{label} carries no {segment!r} delta")
    values: dict[str, Any] = {}
    for key in ("baseline_sum", "modified_sum", "delta", "delta_token_mean"):
        value = block.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail(f"{label} segment {segment!r} carries a non-numeric {key}")
        values[key] = float(value)
    finite = all(math.isfinite(value) for value in values.values())
    sign = primary._margin_sign(values["delta"])  # noqa: SLF001
    declared_sign = block.get("sign")
    if isinstance(declared_sign, bool) or not isinstance(declared_sign, int):
        _fail(f"{label} segment {segment!r} declares no integer delta sign")
    if int(declared_sign) != int(sign):
        _fail(
            f"{label} segment {segment!r} declares sign {block.get('sign')!r}, which does not "
            "reconstruct from its own delta"
        )
    return {
        **values,
        "token_count": int(block.get("token_count", -1)),
        "sign": int(sign),
        "sign_label": sign_label(int(sign)),
        "finite": bool(finite),
    }


def sign_label(sign: int) -> str:
    if sign > 0:
        return SIGN_POSITIVE
    if sign < 0:
        return SIGN_NEGATIVE
    return SIGN_ZERO


def build_readout(row: Mapping[str, Any]) -> dict[str, Any]:
    """One variant's readout for one owner, preserving every exact delta."""

    label = f"merged secondary row {row.get('request_id')!r}"
    return {
        "request_id": str(row["request_id"]),
        "request_key": str(row["request_key"]),
        "variant": str(row["variant"]),
        "plan_optional": bool(row["plan_optional"]),
        "scored_target_kind": str(row["scored_target_kind"]),
        "scored_token_count": int(row["scored_token_count"]),
        "native_row_index": int(row["native_row_index"]),
        "baseline_context_id": str(row["baseline_context_id"]),
        "modified_context_id": str(row["modified_context_id"]),
        "successor_context_id": str(row["successor_context_id"]),
        "baseline_context_source": str(row["baseline_context_source"]),
        "inserted_clean_row_c_token_count": int(row["inserted_clean_row_c_token_count"]),
        "segments": {
            segment: _delta_block(row, segment, label=label) for segment in SEGMENTS
        },
    }


def build_owner_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Join the readouts onto owners, primary cohort and TP controls separately."""

    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        cohort = str(row.get("cohort"))
        variant = str(row.get("variant"))
        if variant not in SECONDARY_VARIANTS:
            _fail(f"merged secondary row {row.get('request_id')!r} declares unknown variant")
        if cohort != secondary.VARIANT_COHORT[variant]:
            _fail(
                f"merged secondary row {row.get('request_id')!r} attributes variant {variant!r} "
                f"to cohort {cohort!r}, not the sealed {secondary.VARIANT_COHORT[variant]!r}"
            )
        key = (cohort, str(row.get("image_id")), str(row.get("gt_owner_id")))
        by_variant = grouped.setdefault(key, {})
        if variant in by_variant:
            _fail(
                f"owner {key[2]!r} on image {key[1]!r} carries two {variant!r} readouts; the "
                "join is not one-to-one"
            )
        by_variant[variant] = build_readout(row)

    owner_rows: list[dict[str, Any]] = []
    for (cohort, image_id, gt_owner_id), by_variant in sorted(grouped.items()):
        if cohort == PRIMARY_COHORT:
            expected_required = secondary.VARIANT_P_PLUS_C_THEN_E
        elif cohort == TP_REPLAY_CONTROL_COHORT:
            expected_required = REFERENCE_VARIANT
        else:
            _fail(f"merged secondary rows declare unknown cohort {cohort!r}")
        if expected_required not in by_variant:
            _fail(
                f"owner {gt_owner_id!r} in cohort {cohort!r} carries no {expected_required!r} "
                "readout; a partial join is never reported"
            )
        stray = sorted(
            variant
            for variant in by_variant
            if secondary.VARIANT_COHORT[variant] != cohort
        )
        if stray:
            _fail(f"owner {gt_owner_id!r} carries readout(s) {stray!r} from another cohort")
        owner_rows.append(
            {
                "schema_version": OWNER_ROW_SCHEMA_VERSION,
                "row_kind": "secondary_owner_row",
                "unit_id": UNIT_ID,
                "cohort": cohort,
                "image_id": image_id,
                "gt_owner_id": gt_owner_id,
                "join_key": f"{cohort}|{image_id}|{gt_owner_id}",
                "readout_variants": sorted(by_variant),
                "optional_f_present": (
                    secondary.VARIANT_P_PLUS_E_PLUS_C_THEN_F in by_variant
                ),
                "readouts": {
                    variant: by_variant[variant] for variant in sorted(by_variant)
                },
                "claim_boundary": secondary.CLAIM_BOUNDARY,
            }
        )
    return owner_rows


def assert_cohort_denominators(owner_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The frozen 26 crossing owners and 12 TP controls, kept disjoint."""

    primary_owners = sorted(
        str(row["gt_owner_id"])
        for row in owner_rows
        if str(row["cohort"]) == PRIMARY_COHORT
    )
    control_owners = sorted(
        str(row["gt_owner_id"])
        for row in owner_rows
        if str(row["cohort"]) == TP_REPLAY_CONTROL_COHORT
    )
    if len(primary_owners) != primary.PRIMARY_OWNER_COUNT_U:
        _fail(
            f"the merged secondary evidence covers {len(primary_owners)} crossing owners, not "
            f"the frozen {primary.PRIMARY_OWNER_COUNT_U}"
        )
    if len(control_owners) != primary.TP_CALIBRATION_OWNER_COUNT:
        _fail(
            f"the merged secondary evidence covers {len(control_owners)} TP replay controls, "
            f"not the frozen {primary.TP_CALIBRATION_OWNER_COUNT}"
        )
    overlap = sorted(set(primary_owners) & set(control_owners))
    if overlap:
        _fail(
            f"owner(s) {overlap!r} appear in both the crossing cohort and the TP replay control "
            "cohort; the reference distribution must stay disjoint from what it references"
        )
    optional_f = sorted(
        str(row["gt_owner_id"])
        for row in owner_rows
        if str(row["cohort"]) == PRIMARY_COHORT and bool(row["optional_f_present"])
    )
    return {
        "crossing_owner_ids": primary_owners,
        "crossing_owner_count": len(primary_owners),
        "tp_replay_control_owner_ids": control_owners,
        "tp_replay_control_owner_count": len(control_owners),
        "optional_f_owner_ids": optional_f,
        "optional_f_owner_count": len(optional_f),
        "optional_f_presence_source": OPERATIONAL_DEFINITIONS["optional_f_presence"],
    }


# ---------------------------------------------------------------------------
# 4. Descriptive summaries
# ---------------------------------------------------------------------------


def _within_image_median(values: Sequence[float]) -> float | None:
    """Median of one image's own within-owner paired deltas.

    The only location statistic this module computes.  Its input is always a
    single image's deltas, so it never mixes raw values from two images.
    """

    return statistics.median(values) if values else None


def _segment_readouts(
    entries: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Coverage, sign counts and per-image values for one variant × segment.

    ``entries`` are ``(image_id, segment block)`` pairs, one per owner.

    ``unit.md``: secondary deltas are "interpreted only relative to that
    reference distribution and never pooled across images as raw values".  So
    what is reported here is deliberately limited to

    * discrete counts over owners -- coverage denominators and sign counts,
      which are not raw values; and
    * per-image raw values with a *within-image* median.

    No cross-image median, minimum, maximum, quantile or ordered raw-value array
    is produced.  A reader who wants one has every owner-level delta in
    ``secondary-owner-rows.jsonl`` and every per-image list here, and takes
    responsibility for that pooling themselves.
    """

    finite = [
        (image_id, block) for image_id, block in entries if bool(block.get("finite"))
    ]
    sign_counts = {label: 0 for label in SIGN_LABELS}
    for _, block in finite:
        sign_counts[str(block["sign_label"])] += 1
    per_image: dict[str, list[float]] = {}
    for image_id, block in finite:
        per_image.setdefault(image_id, []).append(float(block["delta"]))
    return {
        "owner_count": len(entries),
        "finite_count": len(finite),
        "nonfinite_count": len(entries) - len(finite),
        "finite_coverage": (len(finite) / len(entries)) if entries else None,
        "sign_counts": dict(sign_counts),
        "image_count": len(per_image),
        "pooling": NO_CROSS_IMAGE_POOLING,
        "per_image": {
            image_id: {
                "owner_count": len(image_values),
                "values": list(image_values),
                "within_image_median": _within_image_median(image_values),
            }
            for image_id, image_values in sorted(per_image.items())
        },
    }


def build_variant_summary(owner_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per variant and segment, over the owners that carry that variant."""

    summary: dict[str, Any] = {}
    for variant in SECONDARY_VARIANTS:
        entries = [
            (str(row["image_id"]), row["readouts"][variant])
            for row in owner_rows
            if variant in row["readouts"]
        ]
        summary[variant] = {
            "cohort": secondary.VARIANT_COHORT[variant],
            "plan_optional": secondary.VARIANT_IS_PLAN_OPTIONAL[variant],
            "owner_count": len(entries),
            "image_count": len({image_id for image_id, _ in entries}),
            "segments": {
                segment: _segment_readouts(
                    [(image_id, readout["segments"][segment]) for image_id, readout in entries]
                )
                for segment in SEGMENTS
            },
        }
    return summary


def build_reference_distribution(owner_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The benign-substitution readouts, as a *per-image* empirical reference.

    ``unit.md`` calls these the reference distribution the primary deltas are
    read beside, and in the same breath forbids pooling them across images as
    raw values.  Both hold at once only if the reference stays image-indexed:
    each image contributes its own control value, and a primary owner's delta is
    compared with the control of *its own* image.  No pooled quantile, minimum,
    maximum, median or ordered cross-image raw array is emitted.
    """

    readouts = [
        (str(row["image_id"]), row["readouts"][REFERENCE_VARIANT])
        for row in owner_rows
        if REFERENCE_VARIANT in row["readouts"]
    ]
    return {
        "variant": REFERENCE_VARIANT,
        "cohort": TP_REPLAY_CONTROL_COHORT,
        "owner_count": len(readouts),
        "image_count": len({image_id for image_id, _ in readouts}),
        "role": OPERATIONAL_DEFINITIONS["benign_substitution_reference"],
        "pooling": NO_CROSS_IMAGE_POOLING,
        "comparison_scope": (
            "read a crossing owner's delta beside the control value of that owner's own image"
        ),
        "segments": {
            segment: {
                "finite_count": sum(
                    1
                    for _, readout in readouts
                    if bool(readout["segments"][segment]["finite"])
                ),
                "owner_count": len(readouts),
                "sign_counts": {
                    label: sum(
                        1
                        for _, readout in readouts
                        if str(readout["segments"][segment]["sign_label"]) == label
                    )
                    for label in SIGN_LABELS
                },
                "per_image": {
                    image_id: {
                        "gt_owner_id": str(readout["request_key"].split("|")[2]),
                        "delta": float(readout["segments"][segment]["delta"]),
                        "sign": int(readout["segments"][segment]["sign"]),
                        "finite": bool(readout["segments"][segment]["finite"]),
                    }
                    for image_id, readout in sorted(readouts, key=lambda item: item[0])
                },
            }
            for segment in SEGMENTS
        },
    }


def build_row_mapping(
    merged_rows: Sequence[Mapping[str, Any]],
    owner_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """State plainly that readout rows were reshaped onto owners, and how.

    The merged input holds one row per executed *request*; this analyzer emits
    one row per *owner*, carrying that owner's readouts side by side.  That is a
    transformation, not a verbatim copy, so it is declared as one -- together
    with the invariant that actually matters, which is that no exact per-segment
    delta was recomputed, rounded or dropped on the way.
    """

    carried = sum(len(row["readouts"]) for row in owner_rows)
    if carried != len(merged_rows):
        _fail(
            f"{len(merged_rows)} merged readout rows produced {carried} owner readouts; the "
            "owner join neither drops nor duplicates a readout"
        )
    return {
        "input_readout_row_count": len(merged_rows),
        "owner_row_count": len(owner_rows),
        "owner_readout_count": carried,
        "input_rows_transformed_to_owner_rows": True,
        "per_segment_deltas_preserved_exactly": True,
        "transformation": (
            "one row per executed request was regrouped into one row per owner; every "
            "per-segment baseline sum, modified sum, delta, token-mean and sign is carried "
            "through unchanged, and no readout is dropped or duplicated"
        ),
    }


def build_summary(
    *,
    merged: MergedSecondary,
    analysis_binding: Mapping[str, Any],
    owner_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The complete machine-readable secondary readout."""

    denominators = assert_cohort_denominators(owner_rows)
    census = merged.receipt["secondary_requests"]
    summary = {
        "row_mapping": build_row_mapping(merged.rows, owner_rows),
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "merged_dir": str(merged.merged_dir),
        "merge_receipt_content_sha256": merged.receipt_content_sha256,
        "merge_input_file_sha256": dict(sorted(merged.file_sha256.items())),
        "primary_analysis": dict(analysis_binding),
        "runtime_identity_sha256": str(merged.receipt.get("runtime_identity_sha256")),
        "secondary_scorer_source_sha256": str(
            merged.receipt.get("secondary_scorer_source_sha256")
        ),
        "request_census": {
            "request_count": int(census["request_count"]),
            "request_ids_sha256": str(census["request_ids_sha256"]),
            "counts_by_variant": dict(sorted(census["counts_by_variant"].items())),
        },
        "denominators": denominators,
        "per_variant": build_variant_summary(owner_rows),
        "benign_substitution_reference": build_reference_distribution(owner_rows),
        "operational_definitions": dict(OPERATIONAL_DEFINITIONS),
        "claim_boundary": secondary.CLAIM_BOUNDARY,
        "not_claimed": list(NOT_CLAIMED),
    }
    assert_emitted_payload(summary, label="secondary summary")
    return summary


# ---------------------------------------------------------------------------
# 5. Markdown rendering
# ---------------------------------------------------------------------------


def _fmt(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render the same content ``secondary-summary.json`` carries."""

    lines: list[str] = [
        "# Sorted crossing-boundary secondary downstream compatibility",
        "",
        f"Unit: `{summary['unit_id']}`",
        "",
        f"- merged secondary evidence: `{summary['merged_dir']}`",
        f"- merge receipt: `{summary['merge_receipt_content_sha256']}`",
        f"- primary analysis gate (digest binding only): "
        f"`{summary['primary_analysis']['binding_sha256']}`",
        f"- runtime identity: `{summary['runtime_identity_sha256']}`",
        f"- requests: {summary['request_census']['request_count']} "
        f"(`{summary['request_census']['request_ids_sha256']}`)",
        "",
        "## Claim boundary",
        "",
        f"{summary['claim_boundary']}.",
        "",
    ]
    for statement in summary["not_claimed"]:
        lines.append(f"- {statement}")
    lines.extend(
        [
            "",
            "## How to read this",
            "",
            "Each number below is one owner's *own* paired difference between two roots that "
            "force byte-identical tokens. There is no pooled cross-image effect size here, "
            "by construction: only discrete counts are aggregated over owners, and raw deltas "
            "stay indexed by image. Compare a crossing owner with the benign-substitution "
            "control of that owner's own image, never with a summary of all twelve.",
            "",
            "## Denominators",
            "",
        ]
    )
    denominators = summary["denominators"]
    lines.extend(
        [
            f"- readout rows consumed: {summary['row_mapping']['input_readout_row_count']}",
            f"- owner rows emitted: {summary['row_mapping']['owner_row_count']}",
            f"- crossing owners: {denominators['crossing_owner_count']}",
            f"- owners with the optional late catch-up (`F`): "
            f"{denominators['optional_f_owner_count']}",
            f"- native-TP replay controls: "
            f"{denominators['tp_replay_control_owner_count']}",
            "",
        ]
    )

    for variant in SECONDARY_VARIANTS:
        block = summary["per_variant"][variant]
        lines.extend(
            [
                f"## `{variant}`",
                "",
                f"Cohort `{block['cohort']}`; {block['owner_count']} owner(s) over "
                f"{block['image_count']} image(s); plan-optional: {block['plan_optional']}.",
                "",
                "Counts only -- these columns aggregate owners, not raw values.",
                "",
                "| segment | finite | coverage | + | 0 | - |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for segment in SEGMENTS:
            stats = block["segments"][segment]
            signs = stats["sign_counts"]
            lines.append(
                f"| {segment} | {stats['finite_count']}/{stats['owner_count']} | "
                f"{_fmt(stats['finite_coverage'], digits=3)} | {signs[SIGN_POSITIVE]} | "
                f"{signs[SIGN_ZERO]} | {signs[SIGN_NEGATIVE]} |"
            )
        lines.extend(
            [
                "",
                "Within-image medians -- each computed over one image's own owners only, and "
                "not comparable across images as raw values:",
                "",
            ]
        )
        for segment in SEGMENTS:
            per_image = block["segments"][segment]["per_image"]
            rendered = ", ".join(
                f"{image_id}={_fmt(values['within_image_median'])} "
                f"(n={values['owner_count']})"
                for image_id, values in per_image.items()
            )
            lines.append(f"- `{segment}`: {rendered or 'n/a'}")
        lines.append("")

    reference = summary["benign_substitution_reference"]
    lines.extend(
        [
            "## Benign-substitution reference, by image",
            "",
            reference["role"] + ".",
            "",
            reference["comparison_scope"].capitalize() + ".",
            "",
            "| segment | finite | + | 0 | - |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for segment in SEGMENTS:
        stats = reference["segments"][segment]
        signs = stats["sign_counts"]
        lines.append(
            f"| {segment} | {stats['finite_count']}/{stats['owner_count']} | "
            f"{signs[SIGN_POSITIVE]} | {signs[SIGN_ZERO]} | {signs[SIGN_NEGATIVE]} |"
        )
    lines.append("")
    for segment in SEGMENTS:
        per_image = reference["segments"][segment]["per_image"]
        rendered = ", ".join(
            f"{image_id}={_fmt(entry['delta'])}" for image_id, entry in per_image.items()
        )
        lines.append(f"- `{segment}` control value by image: {rendered or 'n/a'}")
    lines.extend(["", "## Operational definitions", ""])
    for name, text in sorted(summary["operational_definitions"].items()):
        lines.append(f"- **{name}**: {text}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Run and publish
# ---------------------------------------------------------------------------


def run_analysis(merged_dir: Path, primary_analysis_dir: Path) -> dict[str, Any]:
    """Load the sealed merge and its gate, then build every output payload."""

    merged = load_merged_secondary(merged_dir)
    analysis_binding = bind_primary_analysis(primary_analysis_dir, merged=merged)
    owner_rows = build_owner_rows(merged.rows)
    assert_emitted_payload(owner_rows, label="secondary owner rows")
    summary = build_summary(
        merged=merged, analysis_binding=analysis_binding, owner_rows=owner_rows
    )
    return {
        "summary": summary,
        "owner_rows": owner_rows,
        "input_file_sha256": dict(sorted(merged.file_sha256.items())),
        "merged_dir": str(merged.merged_dir),
        "merge_receipt_content_sha256": merged.receipt_content_sha256,
        "primary_analysis": analysis_binding,
    }


def build_output_files(result: Mapping[str, Any]) -> dict[str, bytes]:
    """The deterministic, self-sealed secondary analysis byte content."""

    summary = result["summary"]
    owner_rows = result["owner_rows"]

    summary_json = (
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    report_md = render_markdown(summary).encode("utf-8")
    owner_rows_bytes = b"".join(
        canonical_json_bytes(row) + b"\n" for row in owner_rows
    )

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "analyzer_source_sha256": sha256_file(Path(__file__).resolve()),
        "merger_source_sha256": sha256_file(Path(secondary_merge.__file__).resolve()),
        "merged_dir": result["merged_dir"],
        "merge_receipt_content_sha256": result["merge_receipt_content_sha256"],
        "primary_analysis": dict(result["primary_analysis"]),
        "runtime_identity_sha256": summary["runtime_identity_sha256"],
        "secondary_scorer_source_sha256": summary["secondary_scorer_source_sha256"],
        "input_file_sha256": dict(sorted(result["input_file_sha256"].items())),
        "artifact_determinism": "no_wall_clock_or_host_specific_field_is_sealed",
        "request_census": dict(summary["request_census"]),
        "row_mapping": dict(summary["row_mapping"]),
        "denominators": {
            key: value
            for key, value in summary["denominators"].items()
            if key.endswith("_count")
        },
        "policy": {
            "inputs": (
                "the sealed secondary merge and the finalized primary analysis digest binding, "
                "and nothing else"
            ),
            "primary_analysis_read_scope": result["primary_analysis"]["read_scope"],
            "input_rows_transformed_to_owner_rows": True,
            "per_segment_deltas_preserved_exactly": True,
            "row_transformation": summary["row_mapping"]["transformation"],
            "reported_quantities": OPERATIONAL_DEFINITIONS["paired_delta"],
            "cross_image_pooling": NO_CROSS_IMAGE_POOLING,
            "claim_boundary": secondary.CLAIM_BOUNDARY,
            "not_claimed": list(NOT_CLAIMED),
        },
        "output_file_digests": {
            OWNER_ROWS_NAME: {
                "path": OWNER_ROWS_NAME,
                "byte_size": len(owner_rows_bytes),
                "row_count": len(owner_rows),
                "sha256": sha256_bytes(owner_rows_bytes),
            },
            SUMMARY_NAME: {
                "path": SUMMARY_NAME,
                "byte_size": len(summary_json),
                "sha256": sha256_bytes(summary_json),
            },
            REPORT_MD_NAME: {
                "path": REPORT_MD_NAME,
                "byte_size": len(report_md),
                "sha256": sha256_bytes(report_md),
            },
        },
    }
    assert_emitted_payload(receipt, label="secondary analysis receipt")
    receipt["receipt_content_sha256"] = sha256_json(receipt)

    return {
        OWNER_ROWS_NAME: owner_rows_bytes,
        SUMMARY_NAME: summary_json,
        REPORT_MD_NAME: report_md,
        RECEIPT_NAME: canonical_json_bytes(receipt) + b"\n",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merged-dir",
        type=Path,
        required=True,
        help="Sealed secondary merge directory published by the secondary merge module",
    )
    parser.add_argument(
        "--primary-analysis-dir",
        type=Path,
        required=True,
        help="The finalized primary analysis directory the merge bound, as a digest gate",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_analysis(Path(args.merged_dir), Path(args.primary_analysis_dir))
        files = build_output_files(result)
        published = primary_merge.publish_merge(Path(args.output_dir), files)
    except (
        SecondaryAnalysisContractError,
        primary_merge.MergeContractError,
        primary.CrossingBoundaryContractError,
    ) as exc:
        raise SystemExit(f"secondary analysis contract violated: {exc}") from exc
    print(
        json.dumps(
            {
                "analysis": published,
                "crossing_owner_count": result["summary"]["denominators"][
                    "crossing_owner_count"
                ],
                "optional_f_owner_count": result["summary"]["denominators"][
                    "optional_f_owner_count"
                ],
                "tp_replay_control_owner_count": result["summary"]["denominators"][
                    "tp_replay_control_owner_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
