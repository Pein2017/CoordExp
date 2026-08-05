#!/usr/bin/env python3
"""Visualization for the sorted crossing-boundary owner release/realization
unit (``2026-08-03-sorted-crossing-boundary-owner-release-realization``).

Frozen unit::

    research/investigations/qwen3-vl-dense-enumeration/experiments/
    2026-08-03-sorted-crossing-boundary-owner-release-realization/{unit.md,tasks.md}

This module reads **only** the analysis directory published by
``analyze_sorted_crossing_boundary_owner_release.py``: ``owner-rows.jsonl``,
``report.json`` and ``receipt.json`` (``report.md``, when the receipt seals
it, is digest-checked but never parsed).  It never opens a predecessor
capture/merge artifact, never recomputes a rank, margin, support disposition,
greedy status or branch, and fails closed if any analysis file does not match
the exact digest the analysis receipt sealed, or if the analysis directory
carries a file the receipt does not know about.

The owner-matrix's ``e_owner_identity_relation`` column is the one cell that
is *not* read from an owner row's own ``at_p``/``at_p_plus_e``/``displacement``
fields: it is the analyzer's own sealed identity comparison, published at
``report.json``'s ``conclusion_fragility.greedy_displacer_identity.pairs``,
between the ``P+E`` coordinate-only greedy displacer and crossing ``E``'s own
strict-match owner from the sealed plan cohort registry.  This module never
recomputes that comparison from the raw displacer/E owner IDs -- doing so
would silently drop the plan-registry cross-check the analyzer performs -- it
only renders the analyzer's own ``disposition`` field, and fails closed if the
``pairs`` mapping is duplicated, names an owner outside the 26 primary owners,
disagrees with owner-rows.jsonl's own ``displacement.greedy_displaced_owner_id``,
or does not exactly cover the greedy-displaced primary owners.

Two rendered products
----------------------
``owner-matrix.png``
    One row per primary owner (frozen count ``26``), ordered by image, the
    ``P`` boundary's due-boundary index, then owner ID.  Columns are
    categorical/ordinal read-outs only -- branch, matched/unmatched ``E``,
    same/different description, ``P``/``P+E`` U support, ``P``/``P+E``
    within-owner target rank, ``P``/``P+E`` target-minus-competitor margin
    **sign only**, ``P``/``P+E`` coordinate-greedy match status, and the
    displacement-derived ``E`` owner identity relation where available.  No
    cell ever encodes a raw or cross-image log probability.
``paired-owner-deltas.png``
    Two within-owner ``P`` -> ``P+E`` delta panels -- target rank delta and
    target-minus-competitor margin delta -- each with an explicit zero
    reference line and points colored/faceted only by the frozen discrete
    ``primary_branch``.  Never a cross-image raw likelihood plot and never a
    color scale implying a calibrated probability.

Two sealed JSON products
-------------------------
``visual-spec.json``
    The exact, pure column/panel definitions, ordering rule, color
    semantics, input digests, rendered dimensions and the ordered owner-ID
    list -- everything needed to reproduce the two PNGs deterministically.
``visual-manifest.json``
    Schema/unit identity, this script's own source sha256, and the exact
    byte size + sha256 of every input consumed and every output emitted
    (including itself, self-sealed the same way every sibling receipt in
    this unit's family is).

All four outputs are published together, atomically, via the merge module's
``publish_merge``: a byte-identical rerun is a no-op, and any drift from an
existing non-identical directory fails closed rather than overwriting it.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import io
import json
from pathlib import Path
import re
import sys
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import analyze_sorted_crossing_boundary_owner_release as analyzer  # noqa: E402
from scripts.research import merge_sorted_crossing_boundary_owner_release as merge  # noqa: E402
from scripts.research import score_sorted_crossing_boundary_owner_release as scorer  # noqa: E402

UNIT_ID = analyzer.UNIT_ID
VISUAL_SCHEMA_VERSION = "sorted-crossing-boundary-owner-release-visual.v1"
MANIFEST_SCHEMA_VERSION = "sorted-crossing-boundary-owner-release-visual-manifest.v1"

OWNER_MATRIX_NAME = "owner-matrix.png"
PAIRED_OWNER_DELTAS_NAME = "paired-owner-deltas.png"
VISUAL_SPEC_NAME = "visual-spec.json"
MANIFEST_NAME = "visual-manifest.json"

PRIMARY_ROW_KIND = "crossing_boundary_primary_owner_row"
CONTROL_ROW_KIND = "crossing_boundary_control_owner_row"
KNOWN_ROW_KINDS = frozenset((PRIMARY_ROW_KIND, CONTROL_ROW_KIND))

EXPECTED_PRIMARY_OWNER_COUNT = 26

KNOWN_BRANCHES = frozenset(
    ("displaced", "release_lost", "realization_fail", "ambiguous")
)
NOT_CLASSIFIED_BRANCH = "not_classified"

KNOWN_STRATA = frozenset((merge.MATCHED_E_STRATUM, merge.UNMATCHED_E_STRATUM))
KNOWN_CONTROL_COHORTS = frozenset(
    (merge.TIMING_CONTROL_COHORT, merge.TP_REPLAY_CONTROL_COHORT)
)

_CONTEXT_ID_PATTERN = re.compile(r"^(?P<image_id>.+):boundary-(?P<index>\d+)$")

_MISSING = object()


class VisualContractError(RuntimeError):
    """A precondition for rendering this unit's figures was not proven."""


def _fail(message: str) -> NoReturn:
    raise VisualContractError(message)


sha256_bytes = analyzer.sha256_bytes
sha256_json = analyzer.sha256_json
canonical_json_bytes = analyzer.canonical_json_bytes

# ---------------------------------------------------------------------------
# Discrete, categorical colors.  Every color below is a fixed label for a
# discrete category (branch, disposition, status, sign); none encodes a
# continuous or cross-image raw log probability.
# ---------------------------------------------------------------------------

COLOR_GREEN = (76, 175, 80)
COLOR_RED = (229, 115, 115)
COLOR_YELLOW = (255, 213, 79)
COLOR_ORANGE = (255, 183, 77)
COLOR_PURPLE = (149, 117, 205)
COLOR_BLUE = (100, 181, 246)
COLOR_GREY = (189, 189, 189)
COLOR_DARK_GREY = (117, 117, 117)
COLOR_BLACK = (33, 33, 33)

BRANCH_COLORS: Mapping[str, tuple[int, int, int]] = {
    "displaced": COLOR_RED,
    "release_lost": COLOR_ORANGE,
    "realization_fail": COLOR_PURPLE,
    "ambiguous": COLOR_YELLOW,
    NOT_CLASSIFIED_BRANCH: COLOR_DARK_GREY,
}

STRATUM_COLORS: Mapping[str, tuple[int, int, int]] = {
    merge.MATCHED_E_STRATUM: COLOR_BLUE,
    merge.UNMATCHED_E_STRATUM: COLOR_GREY,
}

DESCRIPTION_COLORS: Mapping[str, tuple[int, int, int]] = {
    "same": COLOR_BLUE,
    "different": COLOR_GREY,
}

SUPPORT_COLORS: Mapping[str, tuple[int, int, int]] = {
    scorer.SUPPORT_SUPPORTED: COLOR_GREEN,
    scorer.SUPPORT_UNSUPPORTED: COLOR_RED,
    scorer.SUPPORT_AMBIGUOUS_TIE: COLOR_YELLOW,
    scorer.SUPPORT_CALIBRATION_UNAVAILABLE: COLOR_GREY,
}

GREEDY_COLORS: Mapping[str, tuple[int, int, int]] = {
    scorer.GREEDY_TARGET_MATCH: COLOR_GREEN,
    scorer.GREEDY_OTHER_OWNER_MATCH: COLOR_ORANGE,
    scorer.GREEDY_UNMATCHED: COLOR_RED,
    scorer.GREEDY_AMBIGUOUS: COLOR_YELLOW,
    scorer.GREEDY_MALFORMED: COLOR_BLACK,
}

MARGIN_SIGN_COLORS: Mapping[str, tuple[int, int, int]] = {
    "positive": COLOR_GREEN,
    "negative": COLOR_RED,
    "zero": COLOR_GREY,
    "null": COLOR_DARK_GREY,
}

#: The analyzer's own sealed disposition values for
#: ``conclusion_fragility.greedy_displacer_identity.pairs`` -- never recomputed
#: here, only rendered.
DISPLACER_EQUALS_E = analyzer.DISPLACER_EQUALS_E
DISPLACER_NOT_E = analyzer.DISPLACER_NOT_E
DISPLACER_NOT_DETERMINABLE = analyzer.DISPLACER_E_NOT_DETERMINABLE
NO_DISPLACER_PAIR = "no_displacer_pair"
KNOWN_DISPLACER_DISPOSITIONS = frozenset(
    (DISPLACER_EQUALS_E, DISPLACER_NOT_E, DISPLACER_NOT_DETERMINABLE)
)

DISPLACER_DISPOSITION_SHORT_LABELS: Mapping[str, str] = {
    DISPLACER_EQUALS_E: "=E",
    DISPLACER_NOT_E: "!=E",
    DISPLACER_NOT_DETERMINABLE: "E?",
    NO_DISPLACER_PAIR: "n/a",
}

E_OWNER_RELATION_COLORS: Mapping[str, tuple[int, int, int]] = {
    DISPLACER_EQUALS_E: COLOR_ORANGE,
    DISPLACER_NOT_E: COLOR_BLUE,
    DISPLACER_NOT_DETERMINABLE: COLOR_GREY,
    NO_DISPLACER_PAIR: COLOR_DARK_GREY,
}

OWNER_MATRIX_COLUMNS: tuple[str, ...] = (
    "branch",
    "e_stratum",
    "description",
    "support_u_at_p",
    "support_u_at_p_plus_e",
    "target_rank_at_p",
    "target_rank_at_p_plus_e",
    "margin_sign_at_p",
    "margin_sign_at_p_plus_e",
    "greedy_status_at_p",
    "greedy_status_at_p_plus_e",
    "e_owner_identity_relation",
)

OWNER_MATRIX_COLUMN_DEFINITIONS: Mapping[str, str] = {
    "branch": (
        "primary_branch assigned by the analyzer's frozen truth table "
        "(displaced / release_lost / realization_fail / ambiguous), or "
        "'not_classified' when replay was not admitted for this owner"
    ),
    "e_stratum": "matched_e (E strict-matches a physical owner) vs unmatched_e",
    "description": "same_description_as_e vs different description",
    "support_u_at_p": "U-bound forced-D_C target-local coordinate support disposition at P",
    "support_u_at_p_plus_e": "U-bound forced-D_C target-local coordinate support disposition at P+E",
    "target_rank_at_p": (
        "within-owner ordinal target rank among the scored candidate family at P; "
        "never a cross-image raw likelihood"
    ),
    "target_rank_at_p_plus_e": (
        "within-owner ordinal target rank among the scored candidate family at P+E; "
        "never a cross-image raw likelihood"
    ),
    "margin_sign_at_p": (
        "sign only (positive/negative/zero/null) of the target-minus-competitor margin at P"
    ),
    "margin_sign_at_p_plus_e": (
        "sign only (positive/negative/zero/null) of the target-minus-competitor margin at P+E"
    ),
    "greedy_status_at_p": "coordinate-only greedy decode match status at P",
    "greedy_status_at_p_plus_e": "coordinate-only greedy decode match status at P+E",
    "e_owner_identity_relation": (
        "the analyzer's own sealed identity comparison from report.json's "
        "conclusion_fragility.greedy_displacer_identity.pairs, between this owner's P+E "
        "coordinate-only greedy displacer and crossing E's own strict-match owner from the "
        "sealed plan cohort registry: '=E' (displacer_is_crossing_e_owner), '!=E' "
        "(displacer_is_not_crossing_e_owner), 'E?' (crossing_e_owner_not_determinable), or "
        "'n/a' when this owner's P+E branch was never assigned greedy_displaced and so has no "
        "pair; the trailing 'd:<id> e:<id>' shows the displacer and crossing-E owner suffixes "
        "verbatim from that pair, with 'e:null' whenever crossing E is unmatched -- an "
        "unmatched E's owner is never inferred. This is an identity comparison only, never "
        "causal evidence that emitting E produced the displacement"
    ),
}

SHORT_COLUMN_HEADERS: Mapping[str, str] = {
    "branch": "branch",
    "e_stratum": "E stratum",
    "description": "descr.",
    "support_u_at_p": "supp@P",
    "support_u_at_p_plus_e": "supp@P+E",
    "target_rank_at_p": "rank@P",
    "target_rank_at_p_plus_e": "rank@P+E",
    "margin_sign_at_p": "sign@P",
    "margin_sign_at_p_plus_e": "sign@P+E",
    "greedy_status_at_p": "greedy@P",
    "greedy_status_at_p_plus_e": "greedy@P+E",
    "e_owner_identity_relation": "E owner rel.",
}

BRANCH_ORDER: tuple[str, ...] = (
    "displaced",
    "release_lost",
    "realization_fail",
    "ambiguous",
    NOT_CLASSIFIED_BRANCH,
)

PAIRED_DELTA_PANELS: tuple[str, ...] = (
    "target_rank_delta",
    "target_minus_competitor_margin_delta",
)

# ---------------------------------------------------------------------------
# Layout constants shared by the pure spec builders and the PIL renderers, so
# a spec's declared "dimensions" always exactly matches what gets rendered.
# ---------------------------------------------------------------------------

MATRIX_ROW_HEIGHT = 18
MATRIX_HEADER_HEIGHT = 40
MATRIX_LABEL_WIDTH = 230
MATRIX_COLUMN_WIDTH = 96

DELTA_PANEL_WIDTH = 520
DELTA_PANEL_HEIGHT = 360
DELTA_MARGIN = 24
DELTA_TOP_GAP = 40
DELTA_AXIS_GAP = 40
DELTA_GAP = 40


def _owner_matrix_dimensions(*, num_columns: int, num_rows: int) -> dict[str, int]:
    return {
        "width": MATRIX_LABEL_WIDTH + MATRIX_COLUMN_WIDTH * num_columns,
        "height": MATRIX_HEADER_HEIGHT + MATRIX_ROW_HEIGHT * max(1, num_rows),
        "row_height": MATRIX_ROW_HEIGHT,
        "header_height": MATRIX_HEADER_HEIGHT,
        "label_width": MATRIX_LABEL_WIDTH,
        "column_width": MATRIX_COLUMN_WIDTH,
    }


def _paired_deltas_dimensions() -> dict[str, int]:
    return {
        "width": DELTA_MARGIN * 2 + DELTA_PANEL_WIDTH * 2 + DELTA_GAP,
        "height": DELTA_MARGIN + DELTA_TOP_GAP + DELTA_PANEL_HEIGHT + DELTA_AXIS_GAP,
        "panel_width": DELTA_PANEL_WIDTH,
        "panel_height": DELTA_PANEL_HEIGHT,
        "margin": DELTA_MARGIN,
        "top_gap": DELTA_TOP_GAP,
        "axis_gap": DELTA_AXIS_GAP,
        "gap": DELTA_GAP,
    }


# ---------------------------------------------------------------------------
# 1. Load and digest-verify the analysis directory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Artifacts:
    analysis_dir: Path
    receipt: dict[str, Any]
    report: dict[str, Any]
    primary_rows: list[dict[str, Any]]
    control_rows: list[dict[str, Any]]
    input_file_digests: dict[str, dict[str, Any]]
    #: report.json's conclusion_fragility.greedy_displacer_identity.pairs, keyed by
    #: gt_owner_id and validated against owner-rows.jsonl; see
    #: ``_load_greedy_displacer_identity``.
    displacer_identity_by_owner: dict[str, dict[str, Any]]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"{label} is missing at {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{label} is unreadable at {path}: {exc}")
    if not isinstance(value, Mapping):
        _fail(f"{label} at {path} is not a JSON object")
    return dict(value)


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        _fail(f"{label} is missing at {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
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


def _require(row: Mapping[str, Any], key: str, *, owner_id: str) -> Any:
    value = row.get(key, _MISSING)
    if value is _MISSING:
        _fail(f"owner row {owner_id!r} is missing required field {key!r}")
    return value


def _require_nested(row: Mapping[str, Any], *keys: str, owner_id: str) -> Any:
    value: Any = row
    path: list[str] = []
    for key in keys:
        path.append(key)
        if not isinstance(value, Mapping) or key not in value:
            _fail(f"owner row {owner_id!r} is missing required field {'.'.join(path)!r}")
        value = value[key]
    return value


def load_artifacts(analysis_dir: Path) -> Artifacts:
    """Load, schema-check and digest-verify the analyzer's published family.

    Reads ``receipt.json`` first, re-proves its own self-seal, then checks
    every file the receipt declares in ``output_file_digests`` against the
    exact bytes on disk, and fails closed on any file the receipt does not
    know about.  Only after every declared digest matches are
    ``owner-rows.jsonl`` and ``report.json`` parsed and schema-checked.
    """

    analysis_dir = Path(analysis_dir)
    if not analysis_dir.is_dir():
        _fail(f"analysis directory is missing at {analysis_dir}")

    receipt = _read_json(analysis_dir / analyzer.RECEIPT_NAME, "receipt.json")
    if str(receipt.get("schema_version")) != analyzer.RECEIPT_SCHEMA_VERSION:
        _fail(
            f"receipt.json schema {receipt.get('schema_version')!r} is not "
            f"{analyzer.RECEIPT_SCHEMA_VERSION!r}"
        )
    if str(receipt.get("unit_id")) != UNIT_ID:
        _fail("receipt.json belongs to another unit")
    reconstructed = sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    if reconstructed != receipt.get("receipt_content_sha256"):
        _fail("receipt.json does not reconstruct its own digest; it was edited after sealing")

    declared = receipt.get("output_file_digests")
    if not isinstance(declared, Mapping) or not declared:
        _fail("receipt.json declares no output_file_digests")

    input_file_digests: dict[str, dict[str, Any]] = {}
    for name, entry in sorted(declared.items()):
        path = analysis_dir / str(name)
        if not path.is_file():
            _fail(f"analysis file {name!r} is missing at {path}")
        payload = path.read_bytes()
        observed_sha256 = sha256_bytes(payload)
        expected_sha256 = str((entry or {}).get("sha256"))
        if observed_sha256 != expected_sha256:
            _fail(
                f"analysis file {name!r} does not match the digest sealed in receipt.json; "
                "tampered or stale"
            )
        expected_size = (entry or {}).get("byte_size")
        if expected_size is not None and len(payload) != int(expected_size):
            _fail(f"analysis file {name!r} byte size does not match the digest sealed in receipt.json")
        input_file_digests[str(name)] = {"byte_size": len(payload), "sha256": observed_sha256}

    receipt_payload = (analysis_dir / analyzer.RECEIPT_NAME).read_bytes()
    input_file_digests[analyzer.RECEIPT_NAME] = {
        "byte_size": len(receipt_payload),
        "sha256": sha256_bytes(receipt_payload),
    }

    expected_files = set(declared) | {analyzer.RECEIPT_NAME}
    present_files = {entry.name for entry in analysis_dir.iterdir() if entry.is_file()}
    unknown_files = sorted(present_files - expected_files)
    if unknown_files:
        _fail(
            f"analysis directory {analysis_dir} carries file(s) {unknown_files!r} that receipt.json "
            "never sealed; the analyzer's output contract is closed"
        )

    report = json.loads((analysis_dir / analyzer.REPORT_JSON_NAME).read_bytes().decode("utf-8"))
    if str(report.get("schema_version")) != analyzer.REPORT_SCHEMA_VERSION:
        _fail(
            f"report.json schema {report.get('schema_version')!r} is not "
            f"{analyzer.REPORT_SCHEMA_VERSION!r}"
        )
    if str(report.get("unit_id")) != UNIT_ID:
        _fail("report.json belongs to another unit")

    owner_rows = _read_jsonl(analysis_dir / analyzer.OWNER_ROWS_NAME, "owner-rows.jsonl")
    declared_row_count = int(
        (declared.get(analyzer.OWNER_ROWS_NAME) or {}).get("row_count", -1)
    )
    if len(owner_rows) != declared_row_count:
        _fail(
            f"owner-rows.jsonl carries {len(owner_rows)} rows, not the {declared_row_count} "
            "receipt.json sealed"
        )

    primary_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for row in owner_rows:
        owner_id = str(row.get("gt_owner_id"))
        if str(_require(row, "schema_version", owner_id=owner_id)) != analyzer.OWNER_ROW_SCHEMA_VERSION:
            _fail(f"owner row {owner_id!r} carries an unexpected schema_version")
        if str(_require(row, "unit_id", owner_id=owner_id)) != UNIT_ID:
            _fail(f"owner row {owner_id!r} belongs to another unit")
        row_kind = str(_require(row, "row_kind", owner_id=owner_id))
        if row_kind not in KNOWN_ROW_KINDS:
            _fail(f"owner row {owner_id!r} carries unknown row_kind {row_kind!r}")
        if row_kind == PRIMARY_ROW_KIND:
            primary_rows.append(row)
        else:
            control_rows.append(row)

    _validate_primary_rows(primary_rows)
    _validate_control_rows(control_rows)

    reported_denominator = _require_nested(
        report, "primary_cohort", "denominator", owner_id="<report.json>"
    )
    if int(reported_denominator) != len(primary_rows):
        _fail(
            f"owner-rows.jsonl carries {len(primary_rows)} primary owners, but report.json's "
            f"primary_cohort.denominator is {reported_denominator}"
        )

    displacer_identity_by_owner = _load_greedy_displacer_identity(report, primary_rows)

    return Artifacts(
        analysis_dir=analysis_dir,
        receipt=receipt,
        report=report,
        primary_rows=primary_rows,
        control_rows=control_rows,
        input_file_digests=input_file_digests,
        displacer_identity_by_owner=displacer_identity_by_owner,
    )


def _validate_primary_rows(primary_rows: Sequence[Mapping[str, Any]]) -> None:
    if len(primary_rows) != EXPECTED_PRIMARY_OWNER_COUNT:
        _fail(
            f"owner-rows.jsonl carries {len(primary_rows)} primary owners, not the frozen "
            f"{EXPECTED_PRIMARY_OWNER_COUNT}"
        )
    seen_owner_ids: set[str] = set()
    for row in primary_rows:
        owner_id = str(_require(row, "gt_owner_id", owner_id="<unknown>"))
        if owner_id in seen_owner_ids:
            _fail(f"owner {owner_id!r} is duplicated among the primary cohort")
        seen_owner_ids.add(owner_id)

        if str(row.get("cohort")) != merge.PRIMARY_COHORT:
            _fail(f"primary owner {owner_id!r} does not declare the primary cohort")
        _require(row, "image_id", owner_id=owner_id)
        stratum = str(_require(row, "stratum", owner_id=owner_id))
        if stratum not in KNOWN_STRATA:
            _fail(f"owner {owner_id!r} carries unknown stratum {stratum!r}")
        if not isinstance(_require(row, "same_description_as_e", owner_id=owner_id), bool):
            _fail(f"owner {owner_id!r} carries a non-boolean same_description_as_e")

        branch = row.get("primary_branch", _MISSING)
        if branch is _MISSING:
            _fail(f"owner {owner_id!r} is missing required field 'primary_branch'")
        if branch is not None and str(branch) not in KNOWN_BRANCHES:
            _fail(f"owner {owner_id!r} carries unknown primary_branch {branch!r}")

        displacement = row.get("displacement", _MISSING)
        if displacement is _MISSING:
            _fail(f"owner {owner_id!r} is missing required field 'displacement'")
        if displacement is not None and not isinstance(displacement, Mapping):
            _fail(f"owner {owner_id!r} carries a non-object 'displacement'")

        for boundary_key in ("at_p", "at_p_plus_e"):
            boundary = _require(row, boundary_key, owner_id=owner_id)
            if not isinstance(boundary, Mapping):
                _fail(f"owner {owner_id!r}.{boundary_key} is not an object")
            context_id = _require_nested(row, boundary_key, "context_id", owner_id=owner_id)
            _parse_boundary_index(str(context_id), owner_id=owner_id)
            support = str(
                _require_nested(row, boundary_key, "support_disposition_u", owner_id=owner_id)
            )
            if support not in scorer.SUPPORT_DISPOSITIONS:
                _fail(f"owner {owner_id!r}.{boundary_key} carries unknown support disposition {support!r}")
            greedy_status = _require_nested(row, boundary_key, "greedy_status", owner_id=owner_id)
            if greedy_status is not None and str(greedy_status) not in scorer.GREEDY_STATUSES:
                _fail(f"owner {owner_id!r}.{boundary_key} carries unknown greedy_status {greedy_status!r}")
            # target_rank is legitimately null (missing_required_fields owners); presence of the
            # key itself is what is required here, not a non-null value.
            _require_nested(row, boundary_key, "target_rank", owner_id=owner_id)

        transitions = _require(row, "paired_transitions", owner_id=owner_id)
        if not isinstance(transitions, Mapping):
            _fail(f"owner {owner_id!r}.paired_transitions is not an object")
        for metric in ("target_rank", "target_minus_competitor_margin"):
            _require_nested(row, "paired_transitions", metric, "delta", owner_id=owner_id)
        for sign_key in ("sign_at_p", "sign_at_p_plus_e"):
            sign = str(
                _require_nested(
                    row, "paired_transitions", "target_minus_competitor_margin", sign_key,
                    owner_id=owner_id,
                )
            )
            if sign not in MARGIN_SIGN_COLORS:
                _fail(f"owner {owner_id!r} carries unknown margin sign {sign!r}")


def _validate_control_rows(control_rows: Sequence[Mapping[str, Any]]) -> None:
    for row in control_rows:
        owner_id = str(_require(row, "gt_owner_id", owner_id="<unknown>"))
        cohort = str(row.get("cohort"))
        if cohort not in KNOWN_CONTROL_COHORTS:
            _fail(f"control owner {owner_id!r} carries unknown cohort {cohort!r}")


def _load_greedy_displacer_identity(
    report: Mapping[str, Any], primary_rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """The analyzer's sealed identity comparison, keyed by owner and cross-checked.

    Consumes only ``report.json``'s
    ``conclusion_fragility.greedy_displacer_identity.pairs`` -- already digest-
    verified as part of report.json's own bytes -- and never recomputes a
    disposition from raw displacer/E owner IDs. Fails closed on a duplicated
    owner, an owner outside the 26 primary owners, a ``denominator`` that
    disagrees with the pair count, a ``displacing_owner_id`` that disagrees
    with owner-rows.jsonl's own ``displacement.greedy_displaced_owner_id``
    (a misaligned mapping), a non-null ``crossing_e_owner_id`` on an unmatched
    E (an unmatched E's owner must stay null, never inferred), or a pair
    coverage that does not exactly match the greedy_displaced primary owners.
    """

    fragility = report.get("conclusion_fragility")
    if not isinstance(fragility, Mapping):
        _fail("report.json is missing the required 'conclusion_fragility' block")
    identity = fragility.get("greedy_displacer_identity")
    if not isinstance(identity, Mapping):
        _fail("report.json is missing conclusion_fragility.greedy_displacer_identity")
    pairs = identity.get("pairs")
    if not isinstance(pairs, list):
        _fail("conclusion_fragility.greedy_displacer_identity.pairs is not a list")
    declared_denominator = identity.get("denominator")
    if declared_denominator is None or int(declared_denominator) != len(pairs):
        _fail(
            "conclusion_fragility.greedy_displacer_identity.denominator "
            f"({declared_denominator!r}) does not match its own pairs count ({len(pairs)})"
        )

    primary_by_owner = {str(row["gt_owner_id"]): row for row in primary_rows}
    expected_owner_ids = {
        owner_id
        for owner_id, row in primary_by_owner.items()
        if bool((row.get("displacement") or {}).get("greedy_displaced"))
    }

    mapping: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        if not isinstance(pair, Mapping):
            _fail("a greedy_displacer_identity pair is not an object")
        owner_id = str(pair.get("gt_owner_id"))
        if owner_id in mapping:
            _fail(f"owner {owner_id!r} is duplicated in greedy_displacer_identity.pairs")
        if owner_id not in primary_by_owner:
            _fail(
                f"greedy_displacer_identity.pairs names owner {owner_id!r}, which is not one "
                "of the 26 primary owners"
            )
        disposition = pair.get("disposition")
        if disposition not in KNOWN_DISPLACER_DISPOSITIONS:
            _fail(f"owner {owner_id!r} carries unknown displacer disposition {disposition!r}")
        if (
            pair.get("crossing_e_strict_match_status") == "unmatched"
            and pair.get("crossing_e_owner_id") is not None
        ):
            _fail(
                f"owner {owner_id!r} declares an unmatched crossing E but a non-null "
                "crossing_e_owner_id; unmatched E must remain null, never inferred"
            )

        row = primary_by_owner[owner_id]
        row_displacing_owner_id = (row.get("displacement") or {}).get("greedy_displaced_owner_id")
        pair_displacing_owner_id = pair.get("displacing_owner_id")
        if row_displacing_owner_id != pair_displacing_owner_id:
            _fail(
                f"owner {owner_id!r} carries a displacing_owner_id in "
                "greedy_displacer_identity.pairs that disagrees with owner-rows.jsonl's own "
                "displacement.greedy_displaced_owner_id; the mapping is misaligned"
            )
        mapping[owner_id] = dict(pair)

    missing = sorted(expected_owner_ids - set(mapping))
    extra = sorted(set(mapping) - expected_owner_ids)
    if missing or extra:
        _fail(
            "greedy_displacer_identity.pairs does not exactly cover the greedy_displaced "
            f"primary owners (missing={missing!r}, extra={extra!r})"
        )
    return mapping


def _parse_boundary_index(context_id: str, *, owner_id: str) -> int:
    match = _CONTEXT_ID_PATTERN.match(context_id)
    if match is None:
        _fail(f"owner {owner_id!r} carries a malformed context_id {context_id!r}")
    return int(match.group("index"))


# ---------------------------------------------------------------------------
# 2. Pure spec builders (no PIL import; fully testable)
# ---------------------------------------------------------------------------


def _owner_sort_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    owner_id = str(row["gt_owner_id"])
    boundary_index = _parse_boundary_index(str(row["at_p"]["context_id"]), owner_id=owner_id)
    return (str(row["image_id"]), boundary_index, owner_id)


def _cell(text: Any, color: tuple[int, int, int] | None) -> dict[str, Any]:
    return {"text": str(text), "color": None if color is None else list(color)}


def _branch_cell(branch: str | None) -> dict[str, Any]:
    key = branch if branch is not None else NOT_CLASSIFIED_BRANCH
    return _cell(key, BRANCH_COLORS.get(key, COLOR_GREY))


def _stratum_cell(stratum: str) -> dict[str, Any]:
    return _cell(stratum, STRATUM_COLORS.get(stratum, COLOR_GREY))


def _description_cell(same_description: bool) -> dict[str, Any]:
    key = "same" if same_description else "different"
    return _cell(key, DESCRIPTION_COLORS[key])


def _support_cell(disposition: str) -> dict[str, Any]:
    return _cell(disposition, SUPPORT_COLORS.get(disposition, COLOR_GREY))


def _rank_cell(rank: int | None) -> dict[str, Any]:
    return _cell("n/a" if rank is None else rank, None)


def _margin_sign_cell(sign: str) -> dict[str, Any]:
    return _cell(sign, MARGIN_SIGN_COLORS.get(sign, COLOR_GREY))


def _greedy_cell(status: str | None) -> dict[str, Any]:
    return _cell("n/a" if status is None else status, GREEDY_COLORS.get(status, COLOR_GREY))


def _short_owner_suffix(owner_id: str, *, image_id: str) -> str:
    """Drop the redundant ``gt:<image_id>:`` prefix already shown by the row label."""

    prefix = f"gt:{image_id}:"
    return owner_id[len(prefix):] if owner_id.startswith(prefix) else owner_id


def _e_owner_identity_relation_cell(
    row: Mapping[str, Any], displacer_identity_by_owner: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Render the analyzer's own sealed disposition; never recompute it here."""

    owner_id = str(row["gt_owner_id"])
    image_id = str(row["image_id"])
    pair = displacer_identity_by_owner.get(owner_id)
    if pair is None:
        return _cell(
            DISPLACER_DISPOSITION_SHORT_LABELS[NO_DISPLACER_PAIR],
            E_OWNER_RELATION_COLORS[NO_DISPLACER_PAIR],
        )
    disposition = str(pair["disposition"])
    label = DISPLACER_DISPOSITION_SHORT_LABELS.get(disposition, disposition)
    displacer_id = pair.get("displacing_owner_id")
    crossing_e_id = pair.get("crossing_e_owner_id")
    displacer_text = "null" if displacer_id is None else _short_owner_suffix(str(displacer_id), image_id=image_id)
    e_text = "null" if crossing_e_id is None else _short_owner_suffix(str(crossing_e_id), image_id=image_id)
    text = f"{label} d:{displacer_text} e:{e_text}"
    color = E_OWNER_RELATION_COLORS.get(disposition, COLOR_GREY)
    return _cell(text, color)


def _owner_matrix_row_cells(
    row: Mapping[str, Any], displacer_identity_by_owner: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    at_p = row["at_p"]
    at_p_plus_e = row["at_p_plus_e"]
    margin = row["paired_transitions"]["target_minus_competitor_margin"]
    return {
        "branch": _branch_cell(row["primary_branch"]),
        "e_stratum": _stratum_cell(str(row["stratum"])),
        "description": _description_cell(bool(row["same_description_as_e"])),
        "support_u_at_p": _support_cell(str(at_p["support_disposition_u"])),
        "support_u_at_p_plus_e": _support_cell(str(at_p_plus_e["support_disposition_u"])),
        "target_rank_at_p": _rank_cell(at_p["target_rank"]),
        "target_rank_at_p_plus_e": _rank_cell(at_p_plus_e["target_rank"]),
        "margin_sign_at_p": _margin_sign_cell(str(margin["sign_at_p"])),
        "margin_sign_at_p_plus_e": _margin_sign_cell(str(margin["sign_at_p_plus_e"])),
        "greedy_status_at_p": _greedy_cell(at_p["greedy_status"]),
        "greedy_status_at_p_plus_e": _greedy_cell(at_p_plus_e["greedy_status"]),
        "e_owner_identity_relation": _e_owner_identity_relation_cell(
            row, displacer_identity_by_owner
        ),
    }


def build_owner_matrix_spec(
    primary_rows: Sequence[Mapping[str, Any]],
    displacer_identity_by_owner: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """One row per primary owner, ordered by image, due-boundary index, owner ID."""

    # Each owner's due-boundary index is parsed once here and reused for both
    # sorting and the row payload, rather than re-parsing the same context_id
    # a second time.
    due_boundary_index_by_owner = {
        str(row["gt_owner_id"]): _parse_boundary_index(
            str(row["at_p"]["context_id"]), owner_id=str(row["gt_owner_id"])
        )
        for row in primary_rows
    }
    ordered = sorted(
        primary_rows,
        key=lambda row: (
            str(row["image_id"]),
            due_boundary_index_by_owner[str(row["gt_owner_id"])],
            str(row["gt_owner_id"]),
        ),
    )
    rows: list[dict[str, Any]] = []
    for row in ordered:
        owner_id = str(row["gt_owner_id"])
        rows.append(
            {
                "gt_owner_id": owner_id,
                "image_id": str(row["image_id"]),
                "due_boundary_index": due_boundary_index_by_owner[owner_id],
                "context_id_at_p": str(row["at_p"]["context_id"]),
                "context_id_at_p_plus_e": str(row["at_p_plus_e"]["context_id"]),
                "cells": _owner_matrix_row_cells(row, displacer_identity_by_owner),
            }
        )
    return {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "product": "owner_matrix",
        "unit_id": UNIT_ID,
        "cohort": merge.PRIMARY_COHORT,
        "columns": list(OWNER_MATRIX_COLUMNS),
        "column_headers": dict(SHORT_COLUMN_HEADERS),
        "column_definitions": dict(OWNER_MATRIX_COLUMN_DEFINITIONS),
        "ordering": {
            "sort_keys": ["image_id", "due_boundary_index_at_p", "gt_owner_id"],
            "definition": (
                "due_boundary_index_at_p is the integer boundary index parsed from "
                "at_p.context_id ('<image_id>:boundary-<NNN>'); by the predecessor's own "
                "boundary convention P's boundary is always exactly one less than P+E's "
                "boundary, so this operational reading of unit.md's 'due boundary/index' "
                "orders identically whichever of the two paired boundaries is used"
            ),
        },
        "row_count": len(rows),
        "rows": rows,
        "color_semantics": {
            "branch": {key: list(value) for key, value in BRANCH_COLORS.items()},
            "e_stratum": {key: list(value) for key, value in STRATUM_COLORS.items()},
            "description": {key: list(value) for key, value in DESCRIPTION_COLORS.items()},
            "support_disposition": {key: list(value) for key, value in SUPPORT_COLORS.items()},
            "margin_sign": {key: list(value) for key, value in MARGIN_SIGN_COLORS.items()},
            "greedy_status": {key: list(value) for key, value in GREEDY_COLORS.items()},
            "e_owner_identity_relation": {
                key: list(value) for key, value in E_OWNER_RELATION_COLORS.items()
            },
            "e_owner_identity_relation_labels": dict(DISPLACER_DISPOSITION_SHORT_LABELS),
            "note": (
                "every color above is a fixed discrete category label; no cell ever encodes a "
                "raw or cross-image log probability, and the margin columns carry sign only, "
                "never magnitude"
            ),
        },
        "dimensions": _owner_matrix_dimensions(
            num_columns=len(OWNER_MATRIX_COLUMNS), num_rows=len(rows)
        ),
    }


def build_paired_owner_deltas_spec(primary_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Paired within-owner P -> P+E delta points, faceted only by primary_branch."""

    ordered = sorted(primary_rows, key=_owner_sort_key)
    panel_field = {
        "target_rank_delta": "target_rank",
        "target_minus_competitor_margin_delta": "target_minus_competitor_margin",
    }
    panels: dict[str, Any] = {}
    for panel_name, transition_key in panel_field.items():
        points: list[dict[str, Any]] = []
        for row in ordered:
            branch = row["primary_branch"] or NOT_CLASSIFIED_BRANCH
            delta = row["paired_transitions"][transition_key]["delta"]
            points.append(
                {
                    "gt_owner_id": str(row["gt_owner_id"]),
                    "image_id": str(row["image_id"]),
                    "branch": branch,
                    "value": None if delta is None else float(delta),
                    "determinable": delta is not None,
                }
            )
        panels[panel_name] = {
            "title": {
                "target_rank_delta": "target rank delta (P+E - P), within-owner ordinal only",
                "target_minus_competitor_margin_delta": (
                    "target-minus-competitor margin delta (P+E - P)"
                ),
            }[panel_name],
            "value_definition": (
                f"paired_transitions.{transition_key}.delta as computed by the analyzer; "
                "never recomputed here"
            ),
            "points": points,
        }
    return {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "product": "paired_owner_deltas",
        "unit_id": UNIT_ID,
        "cohort": merge.PRIMARY_COHORT,
        "zero_reference": 0.0,
        "facet_by": "primary_branch",
        "branch_order": list(BRANCH_ORDER),
        "panel_order": list(PAIRED_DELTA_PANELS),
        "panels": panels,
        "color_semantics": {
            "branch": {key: list(value) for key, value in BRANCH_COLORS.items()},
            "note": (
                "color/facet encodes only the frozen discrete primary_branch; never a raw or "
                "cross-image log probability, and never a color scale implying a calibrated "
                "probability"
            ),
        },
        "excludes": [
            "cross_image_raw_likelihood_plots",
            "calibrated_probability_color_semantics",
        ],
        "dimensions": _paired_deltas_dimensions(),
    }


# ---------------------------------------------------------------------------
# 3. Rendering (PIL import deferred so specs stay importable without it)
# ---------------------------------------------------------------------------


def render_owner_matrix_png(spec: Mapping[str, Any]) -> bytes:
    from PIL import Image, ImageDraw

    columns = list(spec["columns"])
    headers = spec["column_headers"]
    rows = spec["rows"]
    dims = spec["dimensions"]
    label_width = int(dims["label_width"])
    column_width = int(dims["column_width"])
    row_height = int(dims["row_height"])
    header_height = int(dims["header_height"])

    image = Image.new("RGB", (int(dims["width"]), int(dims["height"])), (18, 18, 20))
    draw = ImageDraw.Draw(image)
    draw.text(
        (4, 2),
        f"owner_matrix ({spec['cohort']}, n={spec['row_count']})",
        fill=(230, 230, 230),
    )
    for column_index, column_name in enumerate(columns):
        x = label_width + column_index * column_width
        header_text = str(headers.get(column_name, column_name))
        draw.text((x + 2, header_height - 16), header_text[:14], fill=(190, 190, 200))

    for row_index, row in enumerate(rows):
        y = header_height + row_index * row_height
        label = f"{row['image_id']}/b{int(row['due_boundary_index']):03d}/{row['gt_owner_id']}"
        draw.text((4, y + 2), label[:40], fill=(220, 220, 225))
        for column_index, column_name in enumerate(columns):
            cell = row["cells"][column_name]
            x = label_width + column_index * column_width
            color = tuple(cell["color"]) if cell.get("color") else (60, 60, 64)
            draw.rectangle([x, y, x + column_width - 2, y + row_height - 2], fill=color)
            draw.text((x + 3, y + 2), str(cell["text"])[:13], fill=(15, 15, 15))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_paired_owner_deltas_png(spec: Mapping[str, Any]) -> bytes:
    from PIL import Image, ImageDraw

    dims = spec["dimensions"]
    margin = int(dims["margin"])
    panel_width = int(dims["panel_width"])
    panel_height = int(dims["panel_height"])
    top_gap = int(dims["top_gap"])
    gap = int(dims["gap"])
    branch_order = list(spec["branch_order"])
    colors = spec["color_semantics"]["branch"]

    image = Image.new("RGB", (int(dims["width"]), int(dims["height"])), (18, 18, 20))
    draw = ImageDraw.Draw(image)

    for panel_index, panel_name in enumerate(spec["panel_order"]):
        panel = spec["panels"][panel_name]
        x0 = margin + panel_index * (panel_width + gap)
        y0 = margin + top_gap
        draw.text((x0, 4), str(panel["title"])[:70], fill=(230, 230, 230))
        draw.rectangle([x0, y0, x0 + panel_width, y0 + panel_height], outline=(90, 90, 96))

        points = list(panel["points"])
        determinable_values = [p["value"] for p in points if p["determinable"]]
        max_abs = max((abs(v) for v in determinable_values), default=1.0) or 1.0
        zero_y = y0 + panel_height / 2
        draw.line([(x0, zero_y), (x0 + panel_width, zero_y)], fill=(140, 140, 150), width=1)
        draw.text((x0 + panel_width + 4, zero_y - 6), "0", fill=(140, 140, 150))

        band_width = panel_width / max(1, len(branch_order))
        points_by_branch: dict[str, list[dict[str, Any]]] = {branch: [] for branch in branch_order}
        for point in points:
            points_by_branch.setdefault(point["branch"], []).append(point)

        for band_index, branch in enumerate(branch_order):
            band_points = points_by_branch.get(branch, [])
            band_x0 = x0 + band_index * band_width
            color = tuple(colors.get(branch, (150, 150, 150)))
            count = max(1, len(band_points))
            for slot, point in enumerate(band_points):
                px = band_x0 + (slot + 0.5) / count * band_width
                draw.text((band_x0 + 2, y0 + panel_height + 4), branch[:11], fill=(190, 190, 200))
                if not point["determinable"]:
                    continue
                value = float(point["value"])
                py = zero_y - (value / max_abs) * (panel_height / 2 - 8)
                radius = 3
                draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=color)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# 4. Combined visual-spec.json and self-sealed visual-manifest.json
# ---------------------------------------------------------------------------


def build_visual_spec(
    *,
    owner_matrix_spec: Mapping[str, Any],
    paired_owner_deltas_spec: Mapping[str, Any],
    input_file_digests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    owner_ids = [str(row["gt_owner_id"]) for row in owner_matrix_spec["rows"]]
    return {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "owner_ids": owner_ids,
        "owner_count": len(owner_ids),
        "input_digests": {name: dict(entry) for name, entry in sorted(input_file_digests.items())},
        "owner_matrix": dict(owner_matrix_spec),
        "paired_owner_deltas": dict(paired_owner_deltas_spec),
    }


def build_visual_manifest(
    *,
    analysis_dir: Path,
    output_dir: Path,
    input_file_digests: Mapping[str, Mapping[str, Any]],
    output_files: Mapping[str, bytes],
    owner_count: int,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "visualizer_source_sha256": sha256_bytes(Path(__file__).resolve().read_bytes()),
        "analysis_dir": str(analysis_dir),
        "output_dir": str(output_dir),
        "owner_count": owner_count,
        "input_file_digests": {
            name: dict(entry) for name, entry in sorted(input_file_digests.items())
        },
        "output_file_digests": {
            name: {"byte_size": len(payload), "sha256": sha256_bytes(payload)}
            for name, payload in sorted(output_files.items())
        },
        "publish_contract": "atomic_staging_directory_rename_or_byte_identical_no_op",
    }
    manifest["manifest_content_sha256"] = sha256_json(manifest)
    return manifest


# ---------------------------------------------------------------------------
# 5. Orchestration and CLI
# ---------------------------------------------------------------------------


def build_output_files(artifacts: Artifacts, *, output_dir: Path) -> dict[str, bytes]:
    owner_matrix_spec = build_owner_matrix_spec(
        artifacts.primary_rows, artifacts.displacer_identity_by_owner
    )
    paired_owner_deltas_spec = build_paired_owner_deltas_spec(artifacts.primary_rows)

    owner_matrix_png = render_owner_matrix_png(owner_matrix_spec)
    paired_owner_deltas_png = render_paired_owner_deltas_png(paired_owner_deltas_spec)

    visual_spec = build_visual_spec(
        owner_matrix_spec=owner_matrix_spec,
        paired_owner_deltas_spec=paired_owner_deltas_spec,
        input_file_digests=artifacts.input_file_digests,
    )
    visual_spec_bytes = (
        json.dumps(visual_spec, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    output_files = {
        OWNER_MATRIX_NAME: owner_matrix_png,
        PAIRED_OWNER_DELTAS_NAME: paired_owner_deltas_png,
        VISUAL_SPEC_NAME: visual_spec_bytes,
    }
    manifest = build_visual_manifest(
        analysis_dir=artifacts.analysis_dir,
        output_dir=output_dir,
        input_file_digests=artifacts.input_file_digests,
        output_files=output_files,
        owner_count=len(artifacts.primary_rows),
    )
    manifest_bytes = canonical_json_bytes(manifest) + b"\n"
    output_files[MANIFEST_NAME] = manifest_bytes
    return output_files


def run_visualization(*, analysis_dir: Path, output_dir: Path) -> dict[str, Any]:
    artifacts = load_artifacts(analysis_dir)
    files = build_output_files(artifacts, output_dir=output_dir)
    published = merge.publish_merge(output_dir, files)
    return {"published": published, "owner_count": len(artifacts.primary_rows)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Analysis directory published by analyze_sorted_crossing_boundary_owner_release.py",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_visualization(
            analysis_dir=Path(args.analysis_dir), output_dir=Path(args.output_dir)
        )
    except (VisualContractError, merge.MergeContractError) as exc:
        raise SystemExit(f"visualization contract violated: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
