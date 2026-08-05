"""Focused tests for the sorted crossing-boundary owner release/realization
visualizer.

Rather than driving the (concurrently evolving) real capture/merge/analyze
pipeline, these tests build a minimal, hand-constructed 26-row analysis
directory that matches exactly the owner-row/report/receipt contract the
visualizer consumes -- schema versions and constants are read live from the
analyzer module so the fixture tracks it, but no analyzer internal logic is
re-derived or duplicated here.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import analyze_sorted_crossing_boundary_owner_release as analyzer  # noqa: E402
from scripts.research import merge_sorted_crossing_boundary_owner_release as merge  # noqa: E402
from scripts.research import visualize_sorted_crossing_boundary_owner_release as viz  # noqa: E402


def _sign(value: float | None) -> str:
    if value is None:
        return "null"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _boundary(
    *,
    context_id: str,
    support: str,
    greedy_status: str | None,
    rank: int | None,
    margin: float | None,
) -> dict:
    return {
        "context_id": context_id,
        "boundary_label": "P",
        "support_disposition_u": support,
        "support_disposition_l": support,
        "greedy_status": greedy_status,
        "greedy_owner_match": None,
        "greedy_nonunique_match": False,
        "target_rank": rank,
        "best_competitor_owner_id": None,
        "target_minus_competitor_margin": margin,
    }


def _owner_row(
    *,
    gt_owner_id: str,
    image_id: str,
    boundary_index: int,
    stratum: str,
    same_description: bool,
    branch: str | None,
    rank_p: int | None,
    rank_p_plus_e: int | None,
    margin_p: float | None,
    margin_p_plus_e: float | None,
    support_p: str = "supported",
    support_p_plus_e: str = "supported",
    greedy_p: str | None = "target_match",
    greedy_p_plus_e: str | None = "target_match",
    greedy_displaced_owner_id: str | None = None,
    likelihood_displaced_owner_id: str | None = None,
    quarantined: bool = False,
    replay_admitted: bool = True,
) -> dict:
    context_p = f"{image_id}:boundary-{boundary_index:03d}"
    context_p_plus_e = f"{image_id}:boundary-{boundary_index + 1:03d}"
    displacement = None
    if greedy_displaced_owner_id or likelihood_displaced_owner_id:
        displacement = {
            "likelihood_displaced": likelihood_displaced_owner_id is not None,
            "likelihood_displaced_owner_id": likelihood_displaced_owner_id,
            "greedy_displaced": greedy_displaced_owner_id is not None,
            "greedy_displaced_owner_id": greedy_displaced_owner_id,
            "decoding_contradicted": False,
        }
    return {
        "schema_version": analyzer.OWNER_ROW_SCHEMA_VERSION,
        "unit_id": analyzer.UNIT_ID,
        "row_kind": viz.PRIMARY_ROW_KIND,
        "gt_owner_id": gt_owner_id,
        "image_id": image_id,
        "cohort": merge.PRIMARY_COHORT,
        "stratum": stratum,
        "same_description_as_e": same_description,
        "primary_branch": branch,
        "displacement": displacement,
        "at_p": _boundary(
            context_id=context_p, support=support_p, greedy_status=greedy_p,
            rank=rank_p, margin=margin_p,
        ),
        "at_p_plus_e": _boundary(
            context_id=context_p_plus_e, support=support_p_plus_e, greedy_status=greedy_p_plus_e,
            rank=rank_p_plus_e, margin=margin_p_plus_e,
        ),
        "paired_transitions": {
            "target_rank": {
                "delta": None if rank_p is None or rank_p_plus_e is None else rank_p_plus_e - rank_p,
            },
            "target_minus_competitor_margin": {
                "delta": (
                    None
                    if margin_p is None or margin_p_plus_e is None
                    else margin_p_plus_e - margin_p
                ),
                "sign_at_p": _sign(margin_p),
                "sign_at_p_plus_e": _sign(margin_p_plus_e),
            },
        },
        "quarantined": quarantined,
        "replay_admitted": replay_admitted,
    }


def _control_row(*, gt_owner_id: str, image_id: str, cohort: str) -> dict:
    return {
        "schema_version": analyzer.OWNER_ROW_SCHEMA_VERSION,
        "unit_id": analyzer.UNIT_ID,
        "row_kind": viz.CONTROL_ROW_KIND,
        "gt_owner_id": gt_owner_id,
        "image_id": image_id,
        "cohort": cohort,
        "in_primary_denominator": False,
    }


BRANCH_CYCLE = ("displaced", "release_lost", "realization_fail", "ambiguous")


def build_primary_rows() -> list[dict]:
    rows: list[dict] = []
    for k in range(25):
        image_id = f"900{1 + (k % 3)}"
        stratum = merge.MATCHED_E_STRATUM if k % 2 == 0 else merge.UNMATCHED_E_STRATUM
        same_description = k % 3 == 0
        branch = BRANCH_CYCLE[k % len(BRANCH_CYCLE)]
        rank_p = 1 + (k % 3)
        rank_p_plus_e = 1 + ((k + 1) % 3)
        margin_p = round(((k % 5) - 2) * 0.7, 4)
        margin_p_plus_e = round(((k % 7) - 3) * 0.5, 4)
        greedy_displaced_owner_id = None
        likelihood_displaced_owner_id = None
        if branch == "displaced":
            if k % 8 == 0:
                greedy_displaced_owner_id = f"gt:{image_id}:other"
                likelihood_displaced_owner_id = f"gt:{image_id}:other"
            elif k % 8 == 4:
                greedy_displaced_owner_id = f"gt:{image_id}:g-owner"
                likelihood_displaced_owner_id = f"gt:{image_id}:l-owner"
        rows.append(
            _owner_row(
                gt_owner_id=f"gt:{image_id}:{k}",
                image_id=image_id,
                boundary_index=k,
                stratum=stratum,
                same_description=same_description,
                branch=branch,
                rank_p=rank_p,
                rank_p_plus_e=rank_p_plus_e,
                margin_p=margin_p,
                margin_p_plus_e=margin_p_plus_e,
                greedy_displaced_owner_id=greedy_displaced_owner_id,
                likelihood_displaced_owner_id=likelihood_displaced_owner_id,
            )
        )
    # A 26th owner whose replay was never admitted: not_classified branch,
    # exercising the None-branch / quarantine path end to end.
    rows.append(
        _owner_row(
            gt_owner_id="gt:9002:99",
            image_id="9002",
            boundary_index=99,
            stratum=merge.UNMATCHED_E_STRATUM,
            same_description=False,
            branch=None,
            rank_p=None,
            rank_p_plus_e=None,
            margin_p=None,
            margin_p_plus_e=None,
            support_p="calibration_unavailable",
            support_p_plus_e="calibration_unavailable",
            greedy_p=None,
            greedy_p_plus_e=None,
            quarantined=True,
            replay_admitted=False,
        )
    )
    assert len(rows) == 26
    return rows


def build_control_rows() -> list[dict]:
    return [
        _control_row(gt_owner_id="gt:9001:timing", image_id="9001", cohort=merge.TIMING_CONTROL_COHORT),
        _control_row(gt_owner_id="gt:9002:tp", image_id="9002", cohort=merge.TP_REPLAY_CONTROL_COHORT),
    ]


# The seven owners build_primary_rows() gives a non-null greedy_displaced_owner_id
# (k in 0, 4, 8, 12, 16, 20, 24), each deliberately covering one of the sealed
# dispositions: an exact identity match, an ordinary mismatch, an unmatched-E
# owner (crossing_e_owner_id must stay null), and a not-determinable owner.
DISPLACER_PAIR_OVERRIDES: dict[str, dict] = {
    "gt:9001:0": {
        "crossing_e_owner_id": "gt:9001:other",
        "crossing_e_strict_match_status": "matched",
        "disposition": viz.DISPLACER_EQUALS_E,
    },
    "gt:9002:4": {
        "crossing_e_owner_id": "gt:9002:someone-else",
        "crossing_e_strict_match_status": "matched",
        "disposition": viz.DISPLACER_NOT_E,
    },
    "gt:9003:8": {
        "crossing_e_owner_id": None,
        "crossing_e_strict_match_status": "unmatched",
        "disposition": viz.DISPLACER_NOT_E,
    },
    "gt:9001:12": {
        "crossing_e_owner_id": None,
        "crossing_e_strict_match_status": None,
        "disposition": viz.DISPLACER_NOT_DETERMINABLE,
    },
    "gt:9002:16": {
        "crossing_e_owner_id": "gt:9002:different",
        "crossing_e_strict_match_status": "matched",
        "disposition": viz.DISPLACER_NOT_E,
    },
    "gt:9003:20": {
        "crossing_e_owner_id": "gt:9003:xyz",
        "crossing_e_strict_match_status": "matched",
        "disposition": viz.DISPLACER_NOT_E,
    },
    "gt:9001:24": {
        "crossing_e_owner_id": "gt:9001:zzz",
        "crossing_e_strict_match_status": "matched",
        "disposition": viz.DISPLACER_NOT_E,
    },
}


def build_displacer_pairs(primary_rows: list[dict]) -> list[dict]:
    """The sealed report.v2 greedy_displacer_identity.pairs matching the fixture.

    ``displacing_owner_id`` is always read back from the owner row's own
    ``displacement.greedy_displaced_owner_id`` (never independently typed), so
    the happy-path fixture cannot itself drift into a misaligned mapping.
    """

    pairs = []
    for row in primary_rows:
        displacement = row.get("displacement") or {}
        if not displacement.get("greedy_displaced"):
            continue
        owner_id = row["gt_owner_id"]
        override = DISPLACER_PAIR_OVERRIDES[owner_id]
        pairs.append(
            {
                "gt_owner_id": owner_id,
                "stratum": row["stratum"],
                "primary_branch": row.get("primary_branch"),
                "interpretable": True,
                "displacing_owner_id": displacement.get("greedy_displaced_owner_id"),
                "crossing_e_owner_id": override["crossing_e_owner_id"],
                "crossing_e_strict_match_status": override["crossing_e_strict_match_status"],
                "disposition": override["disposition"],
            }
        )
    return sorted(pairs, key=lambda pair: pair["gt_owner_id"])


def build_conclusion_fragility(primary_rows: list[dict]) -> dict:
    pairs = build_displacer_pairs(primary_rows)
    return {
        "greedy_displacer_identity": {
            "denominator": len(pairs),
            "owner_ids": [pair["gt_owner_id"] for pair in pairs],
            "pairs": pairs,
        },
    }


#: Passed as ``conclusion_fragility`` to omit the block from report.json
#: entirely (distinct from ``None``, which means "build the default fixture").
OMIT_CONCLUSION_FRAGILITY = object()


def write_analysis_dir(
    analysis_dir: Path,
    *,
    primary_rows: list[dict],
    control_rows: list[dict],
    conclusion_fragility: dict | None = None,
) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    owner_rows_bytes = b"".join(
        analyzer.canonical_json_bytes(row) + b"\n" for row in (*primary_rows, *control_rows)
    )
    report = {
        "schema_version": analyzer.REPORT_SCHEMA_VERSION,
        "unit_id": analyzer.UNIT_ID,
        "primary_cohort": {"denominator": len(primary_rows)},
    }
    if conclusion_fragility is not OMIT_CONCLUSION_FRAGILITY:
        report["conclusion_fragility"] = (
            build_conclusion_fragility(primary_rows)
            if conclusion_fragility is None
            else conclusion_fragility
        )
    report_bytes = analyzer.canonical_json_bytes(report) + b"\n"
    report_md_bytes = b"# fixture report\n"

    (analysis_dir / analyzer.OWNER_ROWS_NAME).write_bytes(owner_rows_bytes)
    (analysis_dir / analyzer.REPORT_JSON_NAME).write_bytes(report_bytes)
    (analysis_dir / analyzer.REPORT_MD_NAME).write_bytes(report_md_bytes)

    receipt = {
        "schema_version": analyzer.RECEIPT_SCHEMA_VERSION,
        "unit_id": analyzer.UNIT_ID,
        "output_file_digests": {
            analyzer.OWNER_ROWS_NAME: {
                "path": analyzer.OWNER_ROWS_NAME,
                "byte_size": len(owner_rows_bytes),
                "row_count": len(primary_rows) + len(control_rows),
                "sha256": analyzer.sha256_bytes(owner_rows_bytes),
            },
            analyzer.REPORT_JSON_NAME: {
                "path": analyzer.REPORT_JSON_NAME,
                "byte_size": len(report_bytes),
                "sha256": analyzer.sha256_bytes(report_bytes),
            },
            analyzer.REPORT_MD_NAME: {
                "path": analyzer.REPORT_MD_NAME,
                "byte_size": len(report_md_bytes),
                "sha256": analyzer.sha256_bytes(report_md_bytes),
            },
        },
    }
    receipt["receipt_content_sha256"] = analyzer.sha256_json(receipt)
    (analysis_dir / analyzer.RECEIPT_NAME).write_text(json.dumps(receipt), encoding="utf-8")


@pytest.fixture()
def analysis_dir(tmp_path: Path) -> Path:
    out = tmp_path / "analysis"
    write_analysis_dir(out, primary_rows=build_primary_rows(), control_rows=build_control_rows())
    return out


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def test_load_artifacts_accepts_a_minimal_valid_analysis_directory(analysis_dir: Path) -> None:
    artifacts = viz.load_artifacts(analysis_dir)
    assert len(artifacts.primary_rows) == 26
    assert len(artifacts.control_rows) == 2
    assert set(artifacts.input_file_digests) == {
        analyzer.OWNER_ROWS_NAME, analyzer.REPORT_JSON_NAME, analyzer.REPORT_MD_NAME, analyzer.RECEIPT_NAME,
    }


def test_load_artifacts_fails_closed_on_wrong_unit_id(tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    write_analysis_dir(out, primary_rows=build_primary_rows(), control_rows=build_control_rows())
    receipt = json.loads((out / analyzer.RECEIPT_NAME).read_text())
    receipt["unit_id"] = "some-other-unit"
    receipt["receipt_content_sha256"] = analyzer.sha256_json(
        {k: v for k, v in receipt.items() if k != "receipt_content_sha256"}
    )
    (out / analyzer.RECEIPT_NAME).write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(viz.VisualContractError, match="another unit"):
        viz.load_artifacts(out)


def test_load_artifacts_fails_closed_on_receipt_self_digest_tamper(analysis_dir: Path) -> None:
    receipt_path = analysis_dir / analyzer.RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())
    receipt["extra_field"] = "tampered"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(viz.VisualContractError, match="does not reconstruct"):
        viz.load_artifacts(analysis_dir)


def test_load_artifacts_fails_closed_on_tampered_owner_rows(analysis_dir: Path) -> None:
    path = analysis_dir / analyzer.OWNER_ROWS_NAME
    path.write_bytes(path.read_bytes() + b'{"tampered": true}\n')
    with pytest.raises(viz.VisualContractError, match="digest"):
        viz.load_artifacts(analysis_dir)


def test_load_artifacts_fails_closed_on_unknown_file_in_analysis_dir(analysis_dir: Path) -> None:
    (analysis_dir / "stray-unsealed-file.txt").write_text("nope", encoding="utf-8")
    with pytest.raises(viz.VisualContractError, match="never sealed"):
        viz.load_artifacts(analysis_dir)


def test_load_artifacts_fails_closed_on_owner_count_not_26(tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    rows = build_primary_rows()[:-1]
    write_analysis_dir(out, primary_rows=rows, control_rows=build_control_rows())
    with pytest.raises(viz.VisualContractError, match="not the frozen 26"):
        viz.load_artifacts(out)


def test_load_artifacts_fails_closed_on_duplicate_owner(tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    rows = build_primary_rows()[:-1]
    duplicate = dict(rows[0])
    rows.append(duplicate)
    write_analysis_dir(out, primary_rows=rows, control_rows=build_control_rows())
    with pytest.raises(viz.VisualContractError, match="duplicated"):
        viz.load_artifacts(out)


def test_load_artifacts_fails_closed_on_missing_required_field(tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    rows = build_primary_rows()
    del rows[0]["primary_branch"]
    write_analysis_dir(out, primary_rows=rows, control_rows=build_control_rows())
    with pytest.raises(viz.VisualContractError, match="missing required field"):
        viz.load_artifacts(out)


def test_load_artifacts_fails_closed_on_unknown_branch(tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    rows = build_primary_rows()
    rows[0]["primary_branch"] = "not_a_real_branch"
    write_analysis_dir(out, primary_rows=rows, control_rows=build_control_rows())
    with pytest.raises(viz.VisualContractError, match="unknown primary_branch"):
        viz.load_artifacts(out)


def test_load_artifacts_fails_closed_on_unknown_support_disposition(tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    rows = build_primary_rows()
    rows[0]["at_p"]["support_disposition_u"] = "bogus_disposition"
    write_analysis_dir(out, primary_rows=rows, control_rows=build_control_rows())
    with pytest.raises(viz.VisualContractError, match="unknown support disposition"):
        viz.load_artifacts(out)


def test_load_artifacts_fails_closed_on_unknown_greedy_status(tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    rows = build_primary_rows()
    rows[0]["at_p_plus_e"]["greedy_status"] = "bogus_status"
    write_analysis_dir(out, primary_rows=rows, control_rows=build_control_rows())
    with pytest.raises(viz.VisualContractError, match="unknown greedy_status"):
        viz.load_artifacts(out)


def test_load_artifacts_fails_closed_on_unknown_stratum(tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    rows = build_primary_rows()
    rows[0]["stratum"] = "not_a_real_stratum"
    write_analysis_dir(out, primary_rows=rows, control_rows=build_control_rows())
    with pytest.raises(viz.VisualContractError, match="unknown stratum"):
        viz.load_artifacts(out)


def test_load_artifacts_fails_closed_on_malformed_context_id(tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    rows = build_primary_rows()
    rows[0]["at_p"]["context_id"] = "not-a-boundary-id"
    write_analysis_dir(out, primary_rows=rows, control_rows=build_control_rows())
    with pytest.raises(viz.VisualContractError, match="malformed context_id"):
        viz.load_artifacts(out)


def test_load_artifacts_fails_closed_on_denominator_mismatch(tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    write_analysis_dir(out, primary_rows=build_primary_rows(), control_rows=build_control_rows())
    report_path = out / analyzer.REPORT_JSON_NAME
    payload = json.loads(report_path.read_bytes())
    payload["primary_cohort"]["denominator"] = 25
    new_bytes = analyzer.canonical_json_bytes(payload) + b"\n"
    report_path.write_bytes(new_bytes)
    receipt_path = out / analyzer.RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())
    receipt["output_file_digests"][analyzer.REPORT_JSON_NAME] = {
        "path": analyzer.REPORT_JSON_NAME,
        "byte_size": len(new_bytes),
        "sha256": analyzer.sha256_bytes(new_bytes),
    }
    receipt["receipt_content_sha256"] = analyzer.sha256_json(
        {k: v for k, v in receipt.items() if k != "receipt_content_sha256"}
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(viz.VisualContractError, match="primary_cohort.denominator"):
        viz.load_artifacts(out)


# ---------------------------------------------------------------------------
# report.v2 conclusion_fragility.greedy_displacer_identity mapping
# ---------------------------------------------------------------------------


def test_load_artifacts_fails_closed_on_missing_conclusion_fragility(tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    write_analysis_dir(
        out, primary_rows=build_primary_rows(), control_rows=build_control_rows(),
        conclusion_fragility=OMIT_CONCLUSION_FRAGILITY,
    )
    with pytest.raises(viz.VisualContractError, match="conclusion_fragility"):
        viz.load_artifacts(out)


def test_load_artifacts_fails_closed_on_missing_greedy_displacer_identity(tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    write_analysis_dir(
        out, primary_rows=build_primary_rows(), control_rows=build_control_rows(),
        conclusion_fragility={},
    )
    with pytest.raises(viz.VisualContractError, match="greedy_displacer_identity"):
        viz.load_artifacts(out)


def test_load_artifacts_fails_closed_on_displacer_identity_denominator_mismatch(
    tmp_path: Path,
) -> None:
    out = tmp_path / "analysis"
    primary_rows = build_primary_rows()
    fragility = build_conclusion_fragility(primary_rows)
    fragility["greedy_displacer_identity"]["denominator"] = 999
    write_analysis_dir(
        out, primary_rows=primary_rows, control_rows=build_control_rows(),
        conclusion_fragility=fragility,
    )
    with pytest.raises(viz.VisualContractError, match="does not match its own pairs count"):
        viz.load_artifacts(out)


def test_load_artifacts_fails_closed_on_duplicate_displacer_pair(tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    primary_rows = build_primary_rows()
    fragility = build_conclusion_fragility(primary_rows)
    pairs = fragility["greedy_displacer_identity"]["pairs"]
    pairs.append(dict(pairs[0]))
    fragility["greedy_displacer_identity"]["denominator"] = len(pairs)
    write_analysis_dir(
        out, primary_rows=primary_rows, control_rows=build_control_rows(),
        conclusion_fragility=fragility,
    )
    with pytest.raises(viz.VisualContractError, match="duplicated in greedy_displacer_identity"):
        viz.load_artifacts(out)


def test_load_artifacts_fails_closed_on_unknown_owner_in_displacer_pairs(tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    primary_rows = build_primary_rows()
    fragility = build_conclusion_fragility(primary_rows)
    pairs = fragility["greedy_displacer_identity"]["pairs"]
    rogue = dict(pairs[0])
    rogue["gt_owner_id"] = "gt:9999:not-a-real-owner"
    pairs.append(rogue)
    fragility["greedy_displacer_identity"]["denominator"] = len(pairs)
    write_analysis_dir(
        out, primary_rows=primary_rows, control_rows=build_control_rows(),
        conclusion_fragility=fragility,
    )
    with pytest.raises(viz.VisualContractError, match="not one of the 26 primary owners"):
        viz.load_artifacts(out)


def test_load_artifacts_fails_closed_on_misaligned_displacer_pair(tmp_path: Path) -> None:
    """A pair whose displacing_owner_id disagrees with owner-rows.jsonl is rejected."""

    out = tmp_path / "analysis"
    primary_rows = build_primary_rows()
    fragility = build_conclusion_fragility(primary_rows)
    for pair in fragility["greedy_displacer_identity"]["pairs"]:
        if pair["gt_owner_id"] == "gt:9001:0":
            pair["displacing_owner_id"] = "gt:9001:someone-the-owner-row-never-named"
    write_analysis_dir(
        out, primary_rows=primary_rows, control_rows=build_control_rows(),
        conclusion_fragility=fragility,
    )
    with pytest.raises(viz.VisualContractError, match="the mapping is misaligned"):
        viz.load_artifacts(out)


def test_load_artifacts_fails_closed_on_missing_pair_for_greedy_displaced_owner(
    tmp_path: Path,
) -> None:
    out = tmp_path / "analysis"
    primary_rows = build_primary_rows()
    fragility = build_conclusion_fragility(primary_rows)
    pairs = [
        pair
        for pair in fragility["greedy_displacer_identity"]["pairs"]
        if pair["gt_owner_id"] != "gt:9001:0"
    ]
    fragility["greedy_displacer_identity"]["pairs"] = pairs
    fragility["greedy_displacer_identity"]["denominator"] = len(pairs)
    write_analysis_dir(
        out, primary_rows=primary_rows, control_rows=build_control_rows(),
        conclusion_fragility=fragility,
    )
    with pytest.raises(viz.VisualContractError, match="does not exactly cover"):
        viz.load_artifacts(out)


def test_load_artifacts_fails_closed_on_extra_pair_for_non_displaced_owner(tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    primary_rows = build_primary_rows()
    fragility = build_conclusion_fragility(primary_rows)
    pairs = fragility["greedy_displacer_identity"]["pairs"]
    pairs.append(
        {
            "gt_owner_id": "gt:9002:1",  # release_lost, never greedy_displaced
            "stratum": merge.UNMATCHED_E_STRATUM,
            "primary_branch": "release_lost",
            "interpretable": True,
            "displacing_owner_id": None,
            "crossing_e_owner_id": None,
            "crossing_e_strict_match_status": None,
            "disposition": viz.DISPLACER_NOT_DETERMINABLE,
        }
    )
    fragility["greedy_displacer_identity"]["denominator"] = len(pairs)
    write_analysis_dir(
        out, primary_rows=primary_rows, control_rows=build_control_rows(),
        conclusion_fragility=fragility,
    )
    with pytest.raises(viz.VisualContractError, match="does not exactly cover"):
        viz.load_artifacts(out)


def test_load_artifacts_fails_closed_on_unmatched_e_with_non_null_owner_id(tmp_path: Path) -> None:
    """Unmatched E must remain null, never inferred."""

    out = tmp_path / "analysis"
    primary_rows = build_primary_rows()
    fragility = build_conclusion_fragility(primary_rows)
    for pair in fragility["greedy_displacer_identity"]["pairs"]:
        if pair["gt_owner_id"] == "gt:9003:8":
            assert pair["crossing_e_strict_match_status"] == "unmatched"
            pair["crossing_e_owner_id"] = "gt:9003:inferred-not-allowed"
    write_analysis_dir(
        out, primary_rows=primary_rows, control_rows=build_control_rows(),
        conclusion_fragility=fragility,
    )
    with pytest.raises(viz.VisualContractError, match="never inferred"):
        viz.load_artifacts(out)


def test_load_artifacts_fails_closed_on_unknown_displacer_disposition(tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    primary_rows = build_primary_rows()
    fragility = build_conclusion_fragility(primary_rows)
    fragility["greedy_displacer_identity"]["pairs"][0]["disposition"] = "not_a_real_disposition"
    write_analysis_dir(
        out, primary_rows=primary_rows, control_rows=build_control_rows(),
        conclusion_fragility=fragility,
    )
    with pytest.raises(viz.VisualContractError, match="unknown displacer disposition"):
        viz.load_artifacts(out)


# ---------------------------------------------------------------------------
# Pure spec builders
# ---------------------------------------------------------------------------


def test_owner_matrix_spec_has_26_rows_ordered_deterministically(analysis_dir: Path) -> None:
    artifacts = viz.load_artifacts(analysis_dir)
    spec = viz.build_owner_matrix_spec(artifacts.primary_rows, artifacts.displacer_identity_by_owner)
    assert spec["row_count"] == 26
    keys = [(row["image_id"], row["due_boundary_index"], row["gt_owner_id"]) for row in spec["rows"]]
    assert keys == sorted(keys)
    # Rebuilding from the same rows in reverse order must not change the result.
    again = viz.build_owner_matrix_spec(
        list(reversed(artifacts.primary_rows)), artifacts.displacer_identity_by_owner
    )
    assert [row["gt_owner_id"] for row in again["rows"]] == [row["gt_owner_id"] for row in spec["rows"]]


def test_owner_matrix_spec_not_classified_branch_for_unadmitted_replay(analysis_dir: Path) -> None:
    artifacts = viz.load_artifacts(analysis_dir)
    spec = viz.build_owner_matrix_spec(artifacts.primary_rows, artifacts.displacer_identity_by_owner)
    by_owner = {row["gt_owner_id"]: row for row in spec["rows"]}
    cell = by_owner["gt:9002:99"]["cells"]["branch"]
    assert cell["text"] == viz.NOT_CLASSIFIED_BRANCH
    assert tuple(cell["color"]) == viz.COLOR_DARK_GREY


def test_owner_matrix_e_owner_identity_relation_reflects_sealed_report_disposition(
    analysis_dir: Path,
) -> None:
    """The cell renders report.json's own disposition, never a recomputed guess."""

    artifacts = viz.load_artifacts(analysis_dir)
    spec = viz.build_owner_matrix_spec(artifacts.primary_rows, artifacts.displacer_identity_by_owner)
    by_owner = {row["gt_owner_id"]: row for row in spec["rows"]}

    # gt:9001:0 is sealed as an exact identity match (displacer == crossing E).
    equals_cell = by_owner["gt:9001:0"]["cells"]["e_owner_identity_relation"]
    assert equals_cell["text"] == "=E d:other e:other"
    assert tuple(equals_cell["color"]) == viz.COLOR_ORANGE

    # gt:9002:4 is sealed as an ordinary mismatch with a determinate crossing E.
    not_equal_cell = by_owner["gt:9002:4"]["cells"]["e_owner_identity_relation"]
    assert not_equal_cell["text"] == "!=E d:g-owner e:someone-else"
    assert tuple(not_equal_cell["color"]) == viz.COLOR_BLUE

    # gt:9003:8's crossing E is unmatched: crossing_e_owner_id stays null, never
    # inferred from the displacer or any other field.
    unmatched_e_cell = by_owner["gt:9003:8"]["cells"]["e_owner_identity_relation"]
    assert unmatched_e_cell["text"] == "!=E d:other e:null"

    # gt:9001:12 is sealed as not determinable (plan identity unresolved).
    not_determinable_cell = by_owner["gt:9001:12"]["cells"]["e_owner_identity_relation"]
    assert not_determinable_cell["text"] == "E? d:g-owner e:null"
    assert tuple(not_determinable_cell["color"]) == viz.COLOR_GREY

    # A release_lost owner (k=1) was never assigned greedy_displaced and so has
    # no pair at all.
    na_cell = by_owner["gt:9002:1"]["cells"]["e_owner_identity_relation"]
    assert na_cell["text"] == "n/a"
    assert tuple(na_cell["color"]) == viz.COLOR_DARK_GREY


def test_owner_matrix_e_owner_identity_relation_is_driven_by_report_not_by_displacement(
    tmp_path: Path,
) -> None:
    """Changing only report.json's disposition changes the rendered cell.

    The owner row's own ``displacement`` fields are untouched, proving the
    matrix consumes the analyzer's sealed comparison rather than recomputing
    one from the raw displacer/likelihood owner IDs.
    """

    out = tmp_path / "analysis"
    primary_rows = build_primary_rows()
    fragility = build_conclusion_fragility(primary_rows)
    for pair in fragility["greedy_displacer_identity"]["pairs"]:
        if pair["gt_owner_id"] == "gt:9002:4":
            pair["disposition"] = viz.DISPLACER_EQUALS_E
            pair["crossing_e_owner_id"] = pair["displacing_owner_id"]
            pair["crossing_e_strict_match_status"] = "matched"
    write_analysis_dir(
        out, primary_rows=primary_rows, control_rows=build_control_rows(),
        conclusion_fragility=fragility,
    )
    artifacts = viz.load_artifacts(out)
    spec = viz.build_owner_matrix_spec(artifacts.primary_rows, artifacts.displacer_identity_by_owner)
    by_owner = {row["gt_owner_id"]: row for row in spec["rows"]}
    cell = by_owner["gt:9002:4"]["cells"]["e_owner_identity_relation"]
    assert cell["text"] == "=E d:g-owner e:g-owner"
    assert tuple(cell["color"]) == viz.COLOR_ORANGE


def test_owner_matrix_spec_colors_are_all_discrete_and_known(analysis_dir: Path) -> None:
    artifacts = viz.load_artifacts(analysis_dir)
    spec = viz.build_owner_matrix_spec(artifacts.primary_rows, artifacts.displacer_identity_by_owner)
    allowed_colors = {
        None,
        *viz.BRANCH_COLORS.values(),
        *viz.STRATUM_COLORS.values(),
        *viz.DESCRIPTION_COLORS.values(),
        *viz.SUPPORT_COLORS.values(),
        *viz.MARGIN_SIGN_COLORS.values(),
        *viz.GREEDY_COLORS.values(),
        *viz.E_OWNER_RELATION_COLORS.values(),
    }
    for row in spec["rows"]:
        for cell in row["cells"].values():
            color = cell["color"]
            assert (None if color is None else tuple(color)) in allowed_colors


def test_paired_owner_deltas_spec_matches_paired_transitions(analysis_dir: Path) -> None:
    artifacts = viz.load_artifacts(analysis_dir)
    spec = viz.build_paired_owner_deltas_spec(artifacts.primary_rows)
    by_owner = {row["gt_owner_id"]: row for row in artifacts.primary_rows}

    rank_points = {p["gt_owner_id"]: p for p in spec["panels"]["target_rank_delta"]["points"]}
    margin_points = {
        p["gt_owner_id"]: p for p in spec["panels"]["target_minus_competitor_margin_delta"]["points"]
    }
    assert set(rank_points) == set(by_owner)
    for owner_id, row in by_owner.items():
        expected_rank_delta = row["paired_transitions"]["target_rank"]["delta"]
        assert rank_points[owner_id]["value"] == expected_rank_delta
        assert rank_points[owner_id]["determinable"] == (expected_rank_delta is not None)
        expected_margin_delta = row["paired_transitions"]["target_minus_competitor_margin"]["delta"]
        assert margin_points[owner_id]["value"] == expected_margin_delta

    # The unadmitted-replay owner has no branch and must facet under not_classified.
    assert rank_points["gt:9002:99"]["branch"] == viz.NOT_CLASSIFIED_BRANCH
    assert rank_points["gt:9002:99"]["determinable"] is False


def test_paired_owner_deltas_spec_never_names_raw_likelihood(analysis_dir: Path) -> None:
    artifacts = viz.load_artifacts(analysis_dir)
    spec = viz.build_paired_owner_deltas_spec(artifacts.primary_rows)
    assert "cross_image_raw_likelihood_plots" in spec["excludes"]
    assert spec["facet_by"] == "primary_branch"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_owner_matrix_png_is_a_valid_png_sized_from_spec(analysis_dir: Path) -> None:
    from PIL import Image

    artifacts = viz.load_artifacts(analysis_dir)
    spec = viz.build_owner_matrix_spec(artifacts.primary_rows, artifacts.displacer_identity_by_owner)
    payload = viz.render_owner_matrix_png(spec)
    with Image.open(__import__("io").BytesIO(payload)) as image:
        assert image.format == "PNG"
        assert image.size == (spec["dimensions"]["width"], spec["dimensions"]["height"])


def test_render_paired_owner_deltas_png_is_a_valid_png_sized_from_spec(analysis_dir: Path) -> None:
    from PIL import Image

    artifacts = viz.load_artifacts(analysis_dir)
    spec = viz.build_paired_owner_deltas_spec(artifacts.primary_rows)
    payload = viz.render_paired_owner_deltas_png(spec)
    with Image.open(__import__("io").BytesIO(payload)) as image:
        assert image.format == "PNG"
        assert image.size == (spec["dimensions"]["width"], spec["dimensions"]["height"])


def test_rendering_is_byte_deterministic_across_repeated_calls(analysis_dir: Path) -> None:
    artifacts = viz.load_artifacts(analysis_dir)
    matrix_spec = viz.build_owner_matrix_spec(artifacts.primary_rows, artifacts.displacer_identity_by_owner)
    deltas_spec = viz.build_paired_owner_deltas_spec(artifacts.primary_rows)
    assert viz.render_owner_matrix_png(matrix_spec) == viz.render_owner_matrix_png(matrix_spec)
    assert viz.render_paired_owner_deltas_png(deltas_spec) == viz.render_paired_owner_deltas_png(deltas_spec)


# ---------------------------------------------------------------------------
# End-to-end CLI / manifest
# ---------------------------------------------------------------------------


def test_main_publishes_all_four_sealed_outputs(analysis_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "viz"
    exit_code = viz.main(["--analysis-dir", str(analysis_dir), "--output-dir", str(output_dir)])
    assert exit_code == 0

    for name in (viz.OWNER_MATRIX_NAME, viz.PAIRED_OWNER_DELTAS_NAME, viz.VISUAL_SPEC_NAME, viz.MANIFEST_NAME):
        assert (output_dir / name).is_file()

    visual_spec = json.loads((output_dir / viz.VISUAL_SPEC_NAME).read_text())
    assert visual_spec["owner_count"] == 26
    assert len(visual_spec["owner_ids"]) == 26
    assert set(visual_spec["input_digests"]) == {
        analyzer.OWNER_ROWS_NAME, analyzer.REPORT_JSON_NAME, analyzer.REPORT_MD_NAME, analyzer.RECEIPT_NAME,
    }

    manifest = json.loads((output_dir / viz.MANIFEST_NAME).read_text())
    assert manifest["owner_count"] == 26
    reconstructed = analyzer.sha256_json(
        {k: v for k, v in manifest.items() if k != "manifest_content_sha256"}
    )
    assert reconstructed == manifest["manifest_content_sha256"]

    # The manifest self-seals via manifest_content_sha256 (checked above); it
    # cannot also declare a byte digest of itself in output_file_digests, so
    # that section only covers the three other emitted files.
    for name in (viz.OWNER_MATRIX_NAME, viz.PAIRED_OWNER_DELTAS_NAME, viz.VISUAL_SPEC_NAME):
        entry = manifest["output_file_digests"][name]
        on_disk = (output_dir / name).read_bytes()
        assert entry["sha256"] == analyzer.sha256_bytes(on_disk)
        assert entry["byte_size"] == len(on_disk)
    assert viz.MANIFEST_NAME not in manifest["output_file_digests"]


def test_main_rerun_is_a_byte_identical_no_op(analysis_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "viz"
    first = viz.run_visualization(analysis_dir=analysis_dir, output_dir=output_dir)
    assert first["published"]["published"] is True
    second = viz.run_visualization(analysis_dir=analysis_dir, output_dir=output_dir)
    assert second["published"]["published"] is False
    assert second["published"]["publish_mode"] == "no_op_identical_rerun"


def test_main_fails_closed_and_writes_nothing_on_a_broken_analysis_dir(tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    rows = build_primary_rows()[:-1]
    write_analysis_dir(out, primary_rows=rows, control_rows=build_control_rows())
    output_dir = tmp_path / "viz"
    with pytest.raises(SystemExit, match="not the frozen 26"):
        viz.main(["--analysis-dir", str(out), "--output-dir", str(output_dir)])
    assert not output_dir.exists()
