"""Contracts for the CPU-only sorted false-negative fixed-budget reanalysis."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts.research.reanalyze_sorted_fn_fixed_budget_controls import (
    CANDIDATE_ROW_SCHEMA_VERSION,
    ReanalysisError,
    _match_candidates,
    canonical_json_bytes,
    load_score_index,
    reanalyze_sorted_fn_fixed_budget_controls,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))
            handle.write(b"\n")


def _candidate(
    *,
    owner_context_id: str,
    rung: str,
    candidate_id: str,
    box_tokens: list[int],
    population: str,
    region: str,
    predecessor_candidate_id: str | None = None,
    raw_row_identity: str | None = None,
    source_digest: str = "digest-a",
    is_control: bool = False,
    control_kind: str | None = None,
    matched_control_group: str = "group-default",
    family_id: str = "family-a",
    iou_to_target: float = 0.0,
    use_real_schema: bool = False,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": CANDIDATE_ROW_SCHEMA_VERSION,
        "owner_context_id": owner_context_id,
        "rung": rung,
        "candidate_id": candidate_id,
        "raw_row_identity": raw_row_identity,
        "source_digest": source_digest,
        "region": region,
        "population": population,
        "is_control": is_control,
        "control_kind": control_kind,
        "matched_control_group": matched_control_group,
        "family_id": family_id,
        "iou_to_target": iou_to_target,
    }
    if predecessor_candidate_id is not None:
        row["predecessor_candidate_id"] = predecessor_candidate_id
    if use_real_schema:
        row["coord_token_ids"] = box_tokens
    else:
        row["box_tokens"] = box_tokens
    return row


def _score_row(
    *,
    candidate_id: str,
    box_tokens: list[int],
    raw_row_identity: str | None,
    source_digest: str,
    score: float,
    use_real_schema: bool = False,
    request_kind: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_id": candidate_id,
        "raw_row_identity": raw_row_identity,
        "source_digest": source_digest,
    }
    if use_real_schema:
        row["coord_token_ids"] = box_tokens
        row["raw_model_logprob"] = {"complete_box_logprob_sum": score}
    else:
        row["box_tokens"] = box_tokens
        row["raw_score"] = score
    if request_kind is not None:
        row["request_kind"] = request_kind
    return row


def _l0_owner_context(
    owner_context_id: str,
    *,
    is_control: bool = False,
    control_kind: str | None = None,
    matched_control_group: str = "group-default",
    target_count: int = 24,
    family_id: str = "family-a",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A minimal admissible L0 rung: equal-count target/decoy population, all
    matched to old score rows, with one clear target-strict peak."""

    candidates: list[dict[str, Any]] = []
    old_scores: list[dict[str, Any]] = []
    for index in range(target_count):
        region = "target_strict" if index == 0 else "target_halo"
        iou = 0.9 if index == 0 else 0.1
        row = _candidate(
            owner_context_id=owner_context_id,
            rung="L0",
            candidate_id=f"{owner_context_id}:t:{index}",
            box_tokens=[index, index + 1, index + 2, index + 3],
            population="target",
            region=region,
            iou_to_target=iou,
            is_control=is_control,
            control_kind=control_kind,
            matched_control_group=matched_control_group,
            family_id=family_id,
        )
        candidates.append(row)
        old_scores.append(
            _score_row(
                candidate_id=row["candidate_id"],
                box_tokens=row["box_tokens"],
                raw_row_identity=None,
                source_digest=row["source_digest"],
                score=10.0 if index == 0 else 1.0,
            )
        )
    for index in range(target_count):
        row = _candidate(
            owner_context_id=owner_context_id,
            rung="L0",
            candidate_id=f"{owner_context_id}:d:{index}",
            box_tokens=[1000 + index, 0, 0, 0],
            population="decoy",
            region="background",
            is_control=is_control,
            control_kind=control_kind,
            matched_control_group=matched_control_group,
            family_id=family_id,
        )
        candidates.append(row)
        old_scores.append(
            _score_row(
                candidate_id=row["candidate_id"],
                box_tokens=row["box_tokens"],
                raw_row_identity=None,
                source_digest=row["source_digest"],
                score=0.5,
            )
        )
    return candidates, old_scores


def _write_fixture(
    tmp_path: Path, candidates: list[dict[str, Any]], old_scores: list[dict[str, Any]], *, name: str = "candidates"
) -> tuple[Path, Path]:
    candidates_path = tmp_path / f"{name}.jsonl"
    old_scores_path = tmp_path / f"{name}-old-scores.jsonl"
    _write_jsonl(candidates_path, candidates)
    _write_jsonl(old_scores_path, old_scores)
    return candidates_path, old_scores_path


# ---------------------------------------------------------------------------
# Preserved core behavior
# ---------------------------------------------------------------------------


def test_l0_happy_path_reuses_scores(tmp_path: Path) -> None:
    candidates, old_scores = _l0_owner_context(
        "owner:strict-positive", is_control=True, control_kind="strict_positive"
    )
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    document = reanalyze_sorted_fn_fixed_budget_controls(
        fixed_budget_candidates=candidates_path,
        old_score_rows=old_scores_path,
        output=tmp_path / "report.json",
    )
    report = document["owner_contexts"]["owner:strict-positive"]["L0"]
    assert report["scored_count"] == report["candidate_count"]
    assert report["unmatched_candidate_ids"] == []
    # No dense reference was supplied: never silently "preserved".
    assert report["dense_reference_validation"]["status"] == "no_dense_reference"
    for threshold in ("0.4", "0.5", "0.6"):
        assert report["per_threshold"][threshold]["target_peak_candidate_id"] == "owner:strict-positive:t:0"


def test_unmatched_candidate_never_reuses_a_near_identity_score(tmp_path: Path) -> None:
    candidates, old_scores = _l0_owner_context("owner:x")
    for row in old_scores:
        if row["candidate_id"] == "owner:x:t:0":
            row["box_tokens"] = [row["box_tokens"][0], row["box_tokens"][1], 999, 999]
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    document = reanalyze_sorted_fn_fixed_budget_controls(
        fixed_budget_candidates=candidates_path,
        old_score_rows=old_scores_path,
        output=tmp_path / "report.json",
    )
    report = document["owner_contexts"]["owner:x"]["L0"]
    assert "owner:x:t:0" in report["unmatched_candidate_ids"]
    assert "owner:x:t:0" in report["needs_rescore"]
    assert report["per_threshold"]["0.5"]["target_peak_candidate_id"] != "owner:x:t:0"


def test_unequal_target_decoy_counts_are_rejected(tmp_path: Path) -> None:
    candidates, old_scores = _l0_owner_context("owner:x")
    candidates.pop()
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    with pytest.raises(ReanalysisError, match="unequal target"):
        reanalyze_sorted_fn_fixed_budget_controls(
            fixed_budget_candidates=candidates_path,
            old_score_rows=old_scores_path,
            output=tmp_path / "report.json",
        )


def test_strict_positive_strict_rescue_and_loose_only_controls_stay_separate(tmp_path: Path) -> None:
    strict_positive, strict_positive_scores = _l0_owner_context(
        "owner:strict-positive", is_control=True, control_kind="strict_positive"
    )
    loose_only, loose_only_scores = _l0_owner_context(
        "owner:loose-only", is_control=True, control_kind="loose_only"
    )
    strict_rescue_candidates = [
        _candidate(
            owner_context_id="owner:strict-rescue",
            rung="scalar_smoke",
            candidate_id="owner:strict-rescue:t:0",
            box_tokens=[7, 7, 7, 7],
            population="target",
            region="target_strict",
            iou_to_target=0.6,
            is_control=True,
            control_kind="strict_rescue",
        ),
        _candidate(
            owner_context_id="owner:strict-rescue",
            rung="scalar_smoke",
            candidate_id="owner:strict-rescue:d:0",
            box_tokens=[8, 8, 8, 8],
            population="decoy",
            region="background",
            is_control=True,
            control_kind="strict_rescue",
        ),
    ]
    candidates = [*strict_positive, *loose_only, *strict_rescue_candidates]
    old_scores = [*strict_positive_scores, *loose_only_scores]
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    document = reanalyze_sorted_fn_fixed_budget_controls(
        fixed_budget_candidates=candidates_path,
        old_score_rows=old_scores_path,
        output=tmp_path / "report.json",
    )
    owner_contexts = document["owner_contexts"]
    assert set(owner_contexts) == {"owner:strict-positive", "owner:loose-only", "owner:strict-rescue"}
    assert set(owner_contexts["owner:strict-positive"]) == {"L0"}
    assert set(owner_contexts["owner:loose-only"]) == {"L0"}
    assert set(owner_contexts["owner:strict-rescue"]) == {"scalar_smoke"}
    rescue_report = owner_contexts["owner:strict-rescue"]["scalar_smoke"]
    assert rescue_report["scored_count"] == 0
    assert rescue_report["unmatched_candidate_ids"] == ["owner:strict-rescue:d:0", "owner:strict-rescue:t:0"]
    assert rescue_report["dense_reference_validation"]["status"] == "not_applicable"


def test_strict_rescue_rejects_a_stray_old_score_match(tmp_path: Path) -> None:
    candidates = [
        _candidate(
            owner_context_id="owner:strict-rescue",
            rung="scalar_smoke",
            candidate_id="owner:strict-rescue:t:0",
            box_tokens=[7, 7, 7, 7],
            population="target",
            region="target_strict",
            is_control=True,
            control_kind="strict_rescue",
        ),
        _candidate(
            owner_context_id="owner:strict-rescue",
            rung="scalar_smoke",
            candidate_id="owner:strict-rescue:d:0",
            box_tokens=[8, 8, 8, 8],
            population="decoy",
            region="background",
            is_control=True,
            control_kind="strict_rescue",
        ),
    ]
    old_scores = [
        _score_row(
            candidate_id="owner:strict-rescue:t:0",
            box_tokens=[7, 7, 7, 7],
            raw_row_identity=None,
            source_digest="digest-a",
            score=5.0,
        )
    ]
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    with pytest.raises(ReanalysisError, match="no dense predecessor score rows are expected"):
        reanalyze_sorted_fn_fixed_budget_controls(
            fixed_budget_candidates=candidates_path,
            old_score_rows=old_scores_path,
            output=tmp_path / "report.json",
        )


@pytest.mark.parametrize("target_count,should_raise", [(23, True), (24, False), (40, False), (41, True)])
def test_l0_target_count_band_is_24_to_40(tmp_path: Path, target_count: int, should_raise: bool) -> None:
    candidates, old_scores = _l0_owner_context("owner:x", target_count=target_count)
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    if should_raise:
        with pytest.raises(ReanalysisError, match="frozen core band is 24-40"):
            reanalyze_sorted_fn_fixed_budget_controls(
                fixed_budget_candidates=candidates_path,
                old_score_rows=old_scores_path,
                output=tmp_path / "report.json",
            )
    else:
        document = reanalyze_sorted_fn_fixed_budget_controls(
            fixed_budget_candidates=candidates_path,
            old_score_rows=old_scores_path,
            output=tmp_path / "report.json",
        )
        assert "owner:x" in document["owner_contexts"]


def _l1_owner_context(
    owner_context_id: str,
    *,
    strict_region_count: int = 64,
    matched_control_group: str = "group-default",
    is_control: bool = False,
    control_kind: str | None = None,
    scored_target_strict_ids: tuple[str, ...] | None = None,
    strict_score: float = 1.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    old_scores: list[dict[str, Any]] = []
    scored_target_strict_ids = scored_target_strict_ids or ()
    for index in range(256):
        region = "target_strict" if index < strict_region_count else "target_halo"
        row = _candidate(
            owner_context_id=owner_context_id,
            rung="L1",
            candidate_id=f"{owner_context_id}:t:{index}",
            box_tokens=[index, 0, 0, 0],
            population="target",
            region=region,
            iou_to_target=0.9 if region == "target_strict" else 0.1,
            is_control=is_control,
            control_kind=control_kind,
            matched_control_group=matched_control_group,
        )
        candidates.append(row)
        if not scored_target_strict_ids or row["candidate_id"] in scored_target_strict_ids:
            old_scores.append(
                _score_row(
                    candidate_id=row["candidate_id"],
                    box_tokens=row["box_tokens"],
                    raw_row_identity=None,
                    source_digest=row["source_digest"],
                    score=strict_score if region == "target_strict" else 0.1,
                )
            )
    for index in range(256):
        row = _candidate(
            owner_context_id=owner_context_id,
            rung="L1",
            candidate_id=f"{owner_context_id}:d:{index}",
            box_tokens=[2000 + index, 0, 0, 0],
            population="decoy",
            region="background",
            is_control=is_control,
            control_kind=control_kind,
            matched_control_group=matched_control_group,
        )
        candidates.append(row)
        old_scores.append(
            _score_row(
                candidate_id=row["candidate_id"],
                box_tokens=row["box_tokens"],
                raw_row_identity=None,
                source_digest=row["source_digest"],
                score=0.1,
            )
        )
    return candidates, old_scores


def test_l1_requires_exactly_256_targets_with_at_least_64_strict_region(tmp_path: Path) -> None:
    candidates, old_scores = _l1_owner_context("owner:x", strict_region_count=63)
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    with pytest.raises(ReanalysisError, match="at least 64"):
        reanalyze_sorted_fn_fixed_budget_controls(
            fixed_budget_candidates=candidates_path,
            old_score_rows=old_scores_path,
            output=tmp_path / "report.json",
        )

    ok_candidates, ok_old_scores = _l1_owner_context("owner:y", strict_region_count=64)
    candidates_path, old_scores_path = _write_fixture(tmp_path, ok_candidates, ok_old_scores, name="y")
    document = reanalyze_sorted_fn_fixed_budget_controls(
        fixed_budget_candidates=candidates_path,
        old_score_rows=old_scores_path,
        output=tmp_path / "report-2.json",
    )
    assert "owner:y" in document["owner_contexts"]


def test_l1_wrong_target_count_is_rejected(tmp_path: Path) -> None:
    candidates, old_scores = _l1_owner_context("owner:x")
    candidates.pop(0)
    candidates.pop(-1)
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    with pytest.raises(ReanalysisError, match="exactly 256"):
        reanalyze_sorted_fn_fixed_budget_controls(
            fixed_budget_candidates=candidates_path,
            old_score_rows=old_scores_path,
            output=tmp_path / "report.json",
        )


def _l2_owner_context(
    owner_context_id: str, *, matched_control_group: str = "group-default"
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    old_scores: list[dict[str, Any]] = []
    for index in range(1024):
        row = _candidate(
            owner_context_id=owner_context_id,
            rung="L2",
            candidate_id=f"{owner_context_id}:t:{index}",
            box_tokens=[index, 0, 0, 0],
            population="target",
            region="target_strict" if index == 0 else "target_halo",
            iou_to_target=0.9 if index == 0 else 0.1,
            matched_control_group=matched_control_group,
        )
        candidates.append(row)
        old_scores.append(
            _score_row(
                candidate_id=row["candidate_id"],
                box_tokens=row["box_tokens"],
                raw_row_identity=None,
                source_digest=row["source_digest"],
                score=1.0,
            )
        )
    for index in range(1024):
        row = _candidate(
            owner_context_id=owner_context_id,
            rung="L2",
            candidate_id=f"{owner_context_id}:d:{index}",
            box_tokens=[4000 + index, 0, 0, 0],
            population="decoy",
            region="background",
            matched_control_group=matched_control_group,
        )
        candidates.append(row)
        old_scores.append(
            _score_row(
                candidate_id=row["candidate_id"],
                box_tokens=row["box_tokens"],
                raw_row_identity=None,
                source_digest=row["source_digest"],
                score=0.1,
            )
        )
    return candidates, old_scores


def test_l2_without_escalation_trigger_is_rejected(tmp_path: Path) -> None:
    candidates, old_scores = _l2_owner_context("owner:x")
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    with pytest.raises(ReanalysisError, match="without a valid escalation trigger"):
        reanalyze_sorted_fn_fixed_budget_controls(
            fixed_budget_candidates=candidates_path,
            old_score_rows=old_scores_path,
            output=tmp_path / "report.json",
        )


def test_l2_is_admitted_with_a_predeclared_near_miss(tmp_path: Path) -> None:
    candidates, old_scores = _l2_owner_context("owner:x")
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    document = reanalyze_sorted_fn_fixed_budget_controls(
        fixed_budget_candidates=candidates_path,
        old_score_rows=old_scores_path,
        near_miss_declarations=["owner:x"],
        output=tmp_path / "report.json",
    )
    assert "L2" in document["owner_contexts"]["owner:x"]


def test_leakage_fields_are_rejected(tmp_path: Path) -> None:
    candidates, old_scores = _l0_owner_context("owner:x")
    candidates[0]["raw_score"] = 3.0
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    with pytest.raises(ReanalysisError, match="score-derived field"):
        reanalyze_sorted_fn_fixed_budget_controls(
            fixed_budget_candidates=candidates_path,
            old_score_rows=old_scores_path,
            output=tmp_path / "report.json",
        )


def test_ambiguous_duplicate_old_score_identity_is_rejected(tmp_path: Path) -> None:
    candidates, old_scores = _l0_owner_context("owner:x")
    old_scores.append(dict(old_scores[0]))
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    with pytest.raises(ReanalysisError, match="ambiguous duplicate identity key"):
        reanalyze_sorted_fn_fixed_budget_controls(
            fixed_budget_candidates=candidates_path,
            old_score_rows=old_scores_path,
            output=tmp_path / "report.json",
        )


def test_output_is_idempotent_and_deterministic(tmp_path: Path) -> None:
    candidates, old_scores = _l0_owner_context("owner:x")
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    output = tmp_path / "report.json"
    first = reanalyze_sorted_fn_fixed_budget_controls(
        fixed_budget_candidates=candidates_path, old_score_rows=old_scores_path, output=output
    )
    second = reanalyze_sorted_fn_fixed_budget_controls(
        fixed_budget_candidates=candidates_path, old_score_rows=old_scores_path, output=output
    )
    assert first["report_digest"] == second["report_digest"]


def test_cli_writes_report(tmp_path: Path) -> None:
    candidates, old_scores = _l0_owner_context("owner:x")
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.research.reanalyze_sorted_fn_fixed_budget_controls",
            "--fixed-budget-candidates",
            str(candidates_path),
            "--old-score-rows",
            str(old_scores_path),
            "--output",
            str(output),
        ],
        cwd="/data/CoordExp/.worktrees/research-probes",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["owner_context_count"] == 1
    assert output.is_file()


# ---------------------------------------------------------------------------
# P0-5: family_id multiset mirroring (not merely description overlap)
# ---------------------------------------------------------------------------


def test_decoy_population_must_mirror_target_family_id_multiset(tmp_path: Path) -> None:
    candidates, old_scores = _l0_owner_context("owner:x")
    # Break exact multiset equality while keeping every candidate's
    # description-equivalent family value present in *some* target row (the
    # old, too-weak "subset" check would have accepted this).
    candidates[24]["family_id"] = "family-b"  # first decoy row
    for row in candidates[:24]:
        if row["candidate_id"] == "owner:x:t:1":
            row["family_id"] = "family-b"
            break
    candidates[25]["family_id"] = "family-b"  # a second decoy row, now 2 decoys of family-b vs 1 target
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    with pytest.raises(ReanalysisError, match="family-mirrored"):
        reanalyze_sorted_fn_fixed_budget_controls(
            fixed_budget_candidates=candidates_path,
            old_score_rows=old_scores_path,
            output=tmp_path / "report.json",
        )


def test_decoy_population_family_id_multiset_match_is_admitted(tmp_path: Path) -> None:
    candidates, old_scores = _l0_owner_context("owner:x")
    for row in candidates[:24]:
        if row["candidate_id"] == "owner:x:t:1":
            row["family_id"] = "family-b"
            break
    candidates[24]["family_id"] = "family-b"  # exactly one matching decoy too
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    document = reanalyze_sorted_fn_fixed_budget_controls(
        fixed_budget_candidates=candidates_path,
        old_score_rows=old_scores_path,
        output=tmp_path / "report.json",
    )
    assert "owner:x" in document["owner_contexts"]


# ---------------------------------------------------------------------------
# P0-6: L2 escalation is matched_control_group-local
# ---------------------------------------------------------------------------


def _failing_l1_control(
    owner_context_id: str, group: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """An L1 strict_positive control whose dense reference disagrees sharply
    with the fixed-budget subset peak, producing a genuine "failed" dense
    reference validation (not merely "insufficient_subset_evidence")."""

    candidates, old_scores = _l1_owner_context(
        owner_context_id,
        strict_region_count=64,
        matched_control_group=group,
        is_control=True,
        control_kind="strict_positive",
        scored_target_strict_ids=(f"{owner_context_id}:t:0",),
        strict_score=1.0,
    )
    dense_candidates = [
        _candidate(
            owner_context_id=owner_context_id,
            rung="dense",
            candidate_id=f"{owner_context_id}:dense:t:0",
            box_tokens=[9000],
            population="target",
            region="target_strict",
            iou_to_target=0.9,
            is_control=True,
            control_kind="strict_positive",
            matched_control_group=group,
        ),
        _candidate(
            owner_context_id=owner_context_id,
            rung="dense",
            candidate_id=f"{owner_context_id}:dense:d:0",
            box_tokens=[9001],
            population="decoy",
            region="background",
            is_control=True,
            control_kind="strict_positive",
            matched_control_group=group,
        ),
    ]
    dense_scores = [
        _score_row(
            candidate_id=f"{owner_context_id}:dense:t:0",
            box_tokens=[9000],
            raw_row_identity=None,
            source_digest="digest-a",
            score=10.0,  # far from the subset's matched peak of 1.0
        ),
        _score_row(
            candidate_id=f"{owner_context_id}:dense:d:0",
            box_tokens=[9001],
            raw_row_identity=None,
            source_digest="digest-a",
            score=0.5,
        ),
    ]
    return candidates, old_scores, dense_candidates, dense_scores


def test_l1_failure_authorizes_l2_only_within_the_same_matched_control_group(tmp_path: Path) -> None:
    control_candidates, control_old_scores, dense_candidates, dense_scores = _failing_l1_control(
        "owner:control", "group-A"
    )
    target_a_candidates, target_a_scores = _l2_owner_context("owner:target-a", matched_control_group="group-A")
    target_b_candidates, target_b_scores = _l2_owner_context("owner:target-b", matched_control_group="group-B")

    candidates = [*control_candidates, *target_a_candidates, *target_b_candidates]
    old_scores = [*control_old_scores, *target_a_scores, *target_b_scores]
    candidates_path = tmp_path / "candidates.jsonl"
    old_scores_path = tmp_path / "old-scores.jsonl"
    dense_candidates_path = tmp_path / "dense-candidates.jsonl"
    dense_scores_path = tmp_path / "dense-scores.jsonl"
    _write_jsonl(candidates_path, candidates)
    _write_jsonl(old_scores_path, old_scores)
    _write_jsonl(dense_candidates_path, dense_candidates)
    _write_jsonl(dense_scores_path, dense_scores)

    with pytest.raises(ReanalysisError, match="owner:target-b"):
        reanalyze_sorted_fn_fixed_budget_controls(
            fixed_budget_candidates=candidates_path,
            old_score_rows=old_scores_path,
            dense_candidates=dense_candidates_path,
            dense_score_rows=dense_scores_path,
            output=tmp_path / "report.json",
        )

    # With group-B's target removed, group-A's L2 target is admitted because
    # its own group's L1 control failed.
    candidates_a_only = [*control_candidates, *target_a_candidates]
    old_scores_a_only = [*control_old_scores, *target_a_scores]
    candidates_a_path = tmp_path / "candidates-a.jsonl"
    old_scores_a_path = tmp_path / "old-scores-a.jsonl"
    _write_jsonl(candidates_a_path, candidates_a_only)
    _write_jsonl(old_scores_a_path, old_scores_a_only)
    document = reanalyze_sorted_fn_fixed_budget_controls(
        fixed_budget_candidates=candidates_a_path,
        old_score_rows=old_scores_a_path,
        dense_candidates=dense_candidates_path,
        dense_score_rows=dense_scores_path,
        output=tmp_path / "report-a-only.json",
    )
    assert document["owner_contexts"]["owner:control"]["L1"]["dense_reference_validation"]["status"] == "failed"
    assert document["l1_stratum_failure"] == {"group-A": True}
    assert "L2" in document["owner_contexts"]["owner:target-a"]


# ---------------------------------------------------------------------------
# P0-7: dense-vs-subset preservation (replaces the old same-subset-peak check)
# ---------------------------------------------------------------------------


def test_dense_reference_preserved_when_subset_peak_and_rank_agree(tmp_path: Path) -> None:
    candidates, old_scores = _l0_owner_context(
        "owner:dense-happy", is_control=True, control_kind="strict_positive"
    )
    dense_candidates = [
        _candidate(
            owner_context_id="owner:dense-happy",
            rung="dense",
            candidate_id="owner:dense-happy:dense:t:0",
            box_tokens=[0, 1, 2, 3],  # identical identity to the matched subset candidate
            population="target",
            region="target_strict",
            iou_to_target=0.9,
            is_control=True,
            control_kind="strict_positive",
        ),
        _candidate(
            owner_context_id="owner:dense-happy",
            rung="dense",
            candidate_id="owner:dense-happy:dense:d:0",
            box_tokens=[1000, 0, 0, 0],
            population="decoy",
            region="background",
            is_control=True,
            control_kind="strict_positive",
        ),
    ]
    dense_scores = [
        _score_row(
            candidate_id="owner:dense-happy:dense:t:0",
            box_tokens=[0, 1, 2, 3],
            raw_row_identity=None,
            source_digest="digest-a",
            score=10.0,  # matches the subset's own matched peak score
        ),
        _score_row(
            candidate_id="owner:dense-happy:dense:d:0",
            box_tokens=[1000, 0, 0, 0],
            raw_row_identity=None,
            source_digest="digest-a",
            score=0.5,
        ),
    ]
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    dense_candidates_path = tmp_path / "dense-candidates.jsonl"
    dense_scores_path = tmp_path / "dense-scores.jsonl"
    _write_jsonl(dense_candidates_path, dense_candidates)
    _write_jsonl(dense_scores_path, dense_scores)
    document = reanalyze_sorted_fn_fixed_budget_controls(
        fixed_budget_candidates=candidates_path,
        old_score_rows=old_scores_path,
        dense_candidates=dense_candidates_path,
        dense_score_rows=dense_scores_path,
        output=tmp_path / "report.json",
    )
    validation = document["owner_contexts"]["owner:dense-happy"]["L0"]["dense_reference_validation"]
    assert validation["status"] == "preserved"
    for entry in validation["per_threshold"].values():
        assert entry["peak_preserved"] is True
        assert entry["rank_side_preserved"] is True
        assert entry["usable_status_preserved"] is True


def test_dense_reference_fails_when_subset_peak_diverges_beyond_tolerance(tmp_path: Path) -> None:
    candidates, old_scores = _l0_owner_context(
        "owner:dense-diverge", is_control=True, control_kind="strict_positive"
    )
    dense_candidates = [
        _candidate(
            owner_context_id="owner:dense-diverge",
            rung="dense",
            candidate_id="owner:dense-diverge:dense:t:0",
            box_tokens=[500, 500, 500, 500],  # a distinct identity, unmatched in the subset
            population="target",
            region="target_strict",
            iou_to_target=0.9,
            is_control=True,
            control_kind="strict_positive",
        ),
        _candidate(
            owner_context_id="owner:dense-diverge",
            rung="dense",
            candidate_id="owner:dense-diverge:dense:d:0",
            box_tokens=[1000, 0, 0, 0],
            population="decoy",
            region="background",
            is_control=True,
            control_kind="strict_positive",
        ),
    ]
    dense_scores = [
        _score_row(
            candidate_id="owner:dense-diverge:dense:t:0",
            box_tokens=[500, 500, 500, 500],
            raw_row_identity=None,
            source_digest="digest-a",
            score=50.0,
        ),
        _score_row(
            candidate_id="owner:dense-diverge:dense:d:0",
            box_tokens=[1000, 0, 0, 0],
            raw_row_identity=None,
            source_digest="digest-a",
            score=0.5,
        ),
    ]
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    dense_candidates_path = tmp_path / "dense-candidates.jsonl"
    dense_scores_path = tmp_path / "dense-scores.jsonl"
    _write_jsonl(dense_candidates_path, dense_candidates)
    _write_jsonl(dense_scores_path, dense_scores)
    document = reanalyze_sorted_fn_fixed_budget_controls(
        fixed_budget_candidates=candidates_path,
        old_score_rows=old_scores_path,
        dense_candidates=dense_candidates_path,
        dense_score_rows=dense_scores_path,
        output=tmp_path / "report.json",
    )
    validation = document["owner_contexts"]["owner:dense-diverge"]["L0"]["dense_reference_validation"]
    assert validation["status"] == "failed"
    assert validation["per_threshold"]["0.5"]["peak_preserved"] is False


def test_dense_reference_status_is_no_dense_reference_when_not_supplied(tmp_path: Path) -> None:
    candidates, old_scores = _l0_owner_context(
        "owner:no-dense", is_control=True, control_kind="loose_only"
    )
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    document = reanalyze_sorted_fn_fixed_budget_controls(
        fixed_budget_candidates=candidates_path,
        old_score_rows=old_scores_path,
        output=tmp_path / "report.json",
    )
    validation = document["owner_contexts"]["owner:no-dense"]["L0"]["dense_reference_validation"]
    assert validation["status"] == "no_dense_reference"


# ---------------------------------------------------------------------------
# B2 sign must be integrated into dense_reference_validation *before* status
# is finalized and before l1_stratum_failure/L2 escalation are computed --
# not patched in afterward where a disagreement could leave "preserved"
# standing. Each test here would have passed under the earlier
# post-hoc-patch implementation (status stayed "preserved", L2 was
# incorrectly rejected) but fails/behaves differently now.
# ---------------------------------------------------------------------------


def _b2_pair_with_own_three_checks_preserved(
    *, before_id: str, after_id: str, group: str, before_subset_peak: float, after_subset_peak: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """An L1 b2_pair_before/after group whose own peak/rank/usable checks at
    L1 independently agree (dense after-peak close to subset after-peak,
    same rank side, same usable status) -- so, absent B2 sign integration,
    the after-context's status would be "preserved" regardless of what the
    paired-decline sign actually says."""

    before_candidates, before_scores = _l1_owner_context(
        before_id,
        strict_region_count=64,
        matched_control_group=group,
        is_control=True,
        control_kind="b2_pair_before",
        scored_target_strict_ids=(f"{before_id}:t:0",),
        strict_score=before_subset_peak,
    )
    after_candidates, after_scores = _l1_owner_context(
        after_id,
        strict_region_count=64,
        matched_control_group=group,
        is_control=True,
        control_kind="b2_pair_after",
        scored_target_strict_ids=(f"{after_id}:t:0",),
        strict_score=after_subset_peak,
    )
    return before_candidates, before_scores, after_candidates, after_scores


def test_b2_sign_disagreement_flips_status_to_failed_and_authorizes_l2(tmp_path: Path) -> None:
    before_id, after_id, group = "owner:b2-before", "owner:b2-after", "group-b2"
    before_candidates, before_scores, after_candidates, after_scores = (
        _b2_pair_with_own_three_checks_preserved(
            before_id=before_id,
            after_id=after_id,
            group=group,
            before_subset_peak=5.0,
            after_subset_peak=3.0,
        )
    )
    target_candidates, target_scores = _l2_owner_context("owner:b2-target", matched_control_group=group)
    candidates = [*before_candidates, *after_candidates, *target_candidates]
    old_scores = [*before_scores, *after_scores, *target_scores]

    # Dense after-peak (3.2) sits within tolerance of the subset after-peak
    # (3.0), and dense before-peak (2.0) is lower -- so dense declines
    # *positively* (after > before) while the subset declines
    # *non-positively* (3.0 < 5.0): the paired-change signs disagree.
    dense_candidates = [
        _candidate(
            owner_context_id=before_id, rung="dense", candidate_id=f"{before_id}:dense:t:0",
            box_tokens=[9100], population="target", region="target_strict", iou_to_target=0.9,
            is_control=True, control_kind="b2_pair_before", matched_control_group=group,
        ),
        _candidate(
            owner_context_id=before_id, rung="dense", candidate_id=f"{before_id}:dense:d:0",
            box_tokens=[9101], population="decoy", region="background",
            is_control=True, control_kind="b2_pair_before", matched_control_group=group,
        ),
        _candidate(
            owner_context_id=after_id, rung="dense", candidate_id=f"{after_id}:dense:t:0",
            box_tokens=[9200], population="target", region="target_strict", iou_to_target=0.9,
            is_control=True, control_kind="b2_pair_after", matched_control_group=group,
        ),
        _candidate(
            owner_context_id=after_id, rung="dense", candidate_id=f"{after_id}:dense:d:0",
            box_tokens=[9201], population="decoy", region="background",
            is_control=True, control_kind="b2_pair_after", matched_control_group=group,
        ),
    ]
    dense_scores = [
        _score_row(candidate_id=f"{before_id}:dense:t:0", box_tokens=[9100], raw_row_identity=None, source_digest="digest-a", score=2.0),
        _score_row(candidate_id=f"{before_id}:dense:d:0", box_tokens=[9101], raw_row_identity=None, source_digest="digest-a", score=0.1),
        _score_row(candidate_id=f"{after_id}:dense:t:0", box_tokens=[9200], raw_row_identity=None, source_digest="digest-a", score=3.2),
        _score_row(candidate_id=f"{after_id}:dense:d:0", box_tokens=[9201], raw_row_identity=None, source_digest="digest-a", score=0.5),
    ]

    candidates_path = tmp_path / "candidates.jsonl"
    old_scores_path = tmp_path / "old-scores.jsonl"
    dense_candidates_path = tmp_path / "dense-candidates.jsonl"
    dense_scores_path = tmp_path / "dense-scores.jsonl"
    _write_jsonl(candidates_path, candidates)
    _write_jsonl(old_scores_path, old_scores)
    _write_jsonl(dense_candidates_path, dense_candidates)
    _write_jsonl(dense_scores_path, dense_scores)

    document = reanalyze_sorted_fn_fixed_budget_controls(
        fixed_budget_candidates=candidates_path,
        old_score_rows=old_scores_path,
        dense_candidates=dense_candidates_path,
        dense_score_rows=dense_scores_path,
        output=tmp_path / "report.json",
    )

    after_validation = document["owner_contexts"][after_id]["L1"]["dense_reference_validation"]
    threshold_entry = after_validation["per_threshold"]["0.5"]
    # The after-context's own three checks agree in isolation...
    assert threshold_entry["peak_preserved"] is True
    assert threshold_entry["rank_side_preserved"] is True
    assert threshold_entry["usable_status_preserved"] is True
    # ...but the integrated b2 sign disagrees, and that must be
    # conclusion-bearing: status is "failed", not "preserved".
    assert threshold_entry["b2_sign_preserved"] is False
    assert after_validation["status"] == "failed"
    # l1_stratum_failure must reflect this *before* L2 escalation checks run:
    # a same-group L2 target is admitted without a near-miss declaration.
    assert document["l1_stratum_failure"] == {group: True}
    assert "L2" in document["owner_contexts"]["owner:b2-target"]


def test_b2_sign_unresolvable_downgrades_preserved_to_insufficient_subset_evidence(
    tmp_path: Path,
) -> None:
    before_id, after_id, group = "owner:b2-before-2", "owner:b2-after-2", "group-b2-2"
    before_candidates, _before_scores, after_candidates, after_scores = (
        _b2_pair_with_own_three_checks_preserved(
            before_id=before_id,
            after_id=after_id,
            group=group,
            before_subset_peak=5.0,
            after_subset_peak=3.0,
        )
    )
    candidates = [*before_candidates, *after_candidates]
    # No old-score rows for the before-context: irrelevant to this test, and
    # its subset peak is not read for the b2 sign check on the dense side.
    old_scores = [*after_scores]

    # Dense reference exists only for the after-context; the before-context
    # has none, so the dense decline sign cannot be computed at all.
    dense_candidates = [
        _candidate(
            owner_context_id=after_id, rung="dense", candidate_id=f"{after_id}:dense:t:0",
            box_tokens=[9200], population="target", region="target_strict", iou_to_target=0.9,
            is_control=True, control_kind="b2_pair_after", matched_control_group=group,
        ),
        _candidate(
            owner_context_id=after_id, rung="dense", candidate_id=f"{after_id}:dense:d:0",
            box_tokens=[9201], population="decoy", region="background",
            is_control=True, control_kind="b2_pair_after", matched_control_group=group,
        ),
    ]
    dense_scores = [
        _score_row(candidate_id=f"{after_id}:dense:t:0", box_tokens=[9200], raw_row_identity=None, source_digest="digest-a", score=3.2),
        _score_row(candidate_id=f"{after_id}:dense:d:0", box_tokens=[9201], raw_row_identity=None, source_digest="digest-a", score=0.5),
    ]

    candidates_path = tmp_path / "candidates.jsonl"
    old_scores_path = tmp_path / "old-scores.jsonl"
    dense_candidates_path = tmp_path / "dense-candidates.jsonl"
    dense_scores_path = tmp_path / "dense-scores.jsonl"
    _write_jsonl(candidates_path, candidates)
    _write_jsonl(old_scores_path, old_scores)
    _write_jsonl(dense_candidates_path, dense_candidates)
    _write_jsonl(dense_scores_path, dense_scores)

    document = reanalyze_sorted_fn_fixed_budget_controls(
        fixed_budget_candidates=candidates_path,
        old_score_rows=old_scores_path,
        dense_candidates=dense_candidates_path,
        dense_score_rows=dense_scores_path,
        output=tmp_path / "report.json",
    )

    after_validation = document["owner_contexts"][after_id]["L1"]["dense_reference_validation"]
    threshold_entry = after_validation["per_threshold"]["0.5"]
    assert threshold_entry["peak_preserved"] is True
    assert threshold_entry["rank_side_preserved"] is True
    assert threshold_entry["usable_status_preserved"] is True
    assert threshold_entry["b2_sign_preserved"] is None
    # An unresolvable b2 sign must not leave the threshold "preserved".
    assert threshold_entry["overall_preserved"] is None
    assert after_validation["status"] == "insufficient_subset_evidence"
    assert document["l1_stratum_failure"] == {}


# ---------------------------------------------------------------------------
# P0-8: real predecessor schema (coord_token_ids / raw_model_logprob) and
# predecessor_candidate_id binding without requiring equal successor IDs
# ---------------------------------------------------------------------------


def test_real_predecessor_schema_and_predecessor_candidate_id_binding(tmp_path: Path) -> None:
    candidates, old_scores = _l0_owner_context("owner:real-schema")
    target_0 = next(row for row in candidates if row["candidate_id"] == "owner:real-schema:t:0")
    # The successor owns its own candidate_id, but binds the predecessor's
    # real-schema score row (coord_token_ids + raw_model_logprob) explicitly
    # via predecessor_candidate_id, whose own candidate_id differs.
    target_0["predecessor_candidate_id"] = "free-tree-box:sha256:abc123"
    target_0["coord_token_ids"] = target_0.pop("box_tokens")
    old_scores = [row for row in old_scores if row["candidate_id"] != "owner:real-schema:t:0"]
    old_scores.append(
        _score_row(
            candidate_id="free-tree-box:sha256:abc123",  # the predecessor's own ID, not the successor's
            box_tokens=target_0["coord_token_ids"],
            raw_row_identity=None,
            source_digest=target_0["source_digest"],
            score=42.0,
            use_real_schema=True,
            request_kind="complete_box",
        )
    )
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    document = reanalyze_sorted_fn_fixed_budget_controls(
        fixed_budget_candidates=candidates_path,
        old_score_rows=old_scores_path,
        output=tmp_path / "report.json",
    )
    report = document["owner_contexts"]["owner:real-schema"]["L0"]
    assert "owner:real-schema:t:0" not in report["unmatched_candidate_ids"]
    assert report["per_threshold"]["0.5"]["target_peak"] == 42.0


def test_real_predecessor_schema_candidate_id_mismatch_without_binding_is_unmatched(tmp_path: Path) -> None:
    # Same setup but *without* predecessor_candidate_id: the successor's own
    # candidate_id does not equal the predecessor's stored candidate_id, so
    # the candidate is correctly left unmatched (never nearest-matched).
    candidates, old_scores = _l0_owner_context("owner:real-schema-2")
    target_0 = next(row for row in candidates if row["candidate_id"] == "owner:real-schema-2:t:0")
    target_0["coord_token_ids"] = target_0.pop("box_tokens")
    old_scores = [row for row in old_scores if row["candidate_id"] != "owner:real-schema-2:t:0"]
    old_scores.append(
        _score_row(
            candidate_id="free-tree-box:sha256:abc123",
            box_tokens=target_0["coord_token_ids"],
            raw_row_identity=None,
            source_digest=target_0["source_digest"],
            score=42.0,
            use_real_schema=True,
            request_kind="complete_box",
        )
    )
    candidates_path, old_scores_path = _write_fixture(tmp_path, candidates, old_scores)
    document = reanalyze_sorted_fn_fixed_budget_controls(
        fixed_budget_candidates=candidates_path,
        old_score_rows=old_scores_path,
        output=tmp_path / "report.json",
    )
    report = document["owner_contexts"]["owner:real-schema-2"]["L0"]
    assert "owner:real-schema-2:t:0" in report["unmatched_candidate_ids"]


# ---------------------------------------------------------------------------
# Predecessor merged landscape-scores.jsonl compatibility: complete free-tree-
# box score rows and conditional raw_bin_scan/dense_scan rows are mixed in
# one artifact; complete-box rows never carry raw_row_identity/source_digest.
# load_score_index must admit exactly the former by explicit request_kind,
# leave source_digest genuinely optional (never fabricated), and still fail
# closed on any row shape it cannot positively explain.
# ---------------------------------------------------------------------------


def _real_complete_box_score_row(
    *, candidate_id: str, coord_token_ids: list[int], score: float
) -> dict[str, Any]:
    """A row shaped like a real predecessor complete free-tree-box score row:
    no raw_row_identity, no source_digest, request_kind=complete_box."""

    return {
        "request_kind": "complete_box",
        "landscape_surface": "canonical_description_free",
        "candidate_id": candidate_id,
        "coord_token_ids": coord_token_ids,
        "raw_model_logprob": {"complete_box_logprob_sum": score},
    }


def _real_dense_scan_row(*, candidate_id: str) -> dict[str, Any]:
    """A row shaped like a real predecessor conditional raw_bin_scan row: a
    different, deliberate request kind with neither coord_token_ids nor
    raw_model_logprob -- must be filtered, not treated as malformed."""

    return {
        "request_kind": "dense_scan",
        "landscape_surface": "restricted_gt_target",
        "candidate_id": candidate_id,
        "candidate_kind": "target_anchor",
        "fixed_coord_token_ids": [151670, 151670],
        "raw_bin_scan": {"rp_1.00": {"bin_logprobs": [-1.0, -2.0]}},
    }


def test_load_score_index_filters_recognized_non_complete_box_rows(tmp_path: Path) -> None:
    path = tmp_path / "merged-landscape-scores.jsonl"
    _write_jsonl(
        path,
        [
            _real_dense_scan_row(candidate_id="conditional-y1-plan:sha256:aaa"),
            _real_complete_box_score_row(
                candidate_id="free-tree-box:sha256:bbb", coord_token_ids=[1, 2, 3, 4], score=7.5
            ),
        ],
    )
    index, filter_report = load_score_index(path, label="old score rows")
    assert filter_report == {
        "total_rows": 2,
        "complete_box_rows": 1,
        "filtered_non_complete_box_rows": 1,
        "filtered_request_kinds": {"dense_scan": 1},
    }
    assert len(index) == 1
    ((candidate_id, box_tokens, raw_row_identity, source_digest), score), = index.items()
    assert candidate_id == "free-tree-box:sha256:bbb"
    assert box_tokens == (1, 2, 3, 4)
    assert raw_row_identity is None
    # Never fabricated: genuinely absent from the real row, recorded as None.
    assert source_digest is None
    assert score == 7.5


def test_load_score_index_no_path_returns_empty_filter_report(tmp_path: Path) -> None:
    index, filter_report = load_score_index(None, label="dense score rows")
    assert index == {}
    assert filter_report == {
        "total_rows": 0,
        "complete_box_rows": 0,
        "filtered_non_complete_box_rows": 0,
        "filtered_request_kinds": {},
    }


def test_load_score_index_unlabeled_row_with_no_recognizable_shape_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    # No coord_token_ids/raw_model_logprob or box_tokens/raw_score, and no
    # request_kind explaining why: an unexpected shape, not a known
    # non-complete-box row kind. Must fail rather than be silently dropped.
    _write_jsonl(path, [{"candidate_id": "mystery:sha256:ccc"}])
    with pytest.raises(
        ReanalysisError,
        match="has no request_kind and is not the narrow unlabeled test-alias complete-box shape",
    ):
        load_score_index(path, label="old score rows")


def test_load_score_index_unlabeled_row_with_real_complete_box_shape_is_rejected(tmp_path: Path) -> None:
    """The original gap: eligibility must never be inferred from shape
    alone. An unlabeled row that happens to carry real complete-box fields
    (coord_token_ids + raw_model_logprob.complete_box_logprob_sum) is not
    the narrow test-alias exception and must still be rejected, not
    silently admitted as if it were an explicit complete_box row."""

    path = tmp_path / "scores.jsonl"
    row = {
        "candidate_id": "free-tree-box:sha256:unlabeled",
        "coord_token_ids": [1, 2, 3, 4],
        "raw_model_logprob": {"complete_box_logprob_sum": 5.0},
    }
    assert "request_kind" not in row
    _write_jsonl(path, [row])
    with pytest.raises(
        ReanalysisError,
        match="has no request_kind and is not the narrow unlabeled test-alias complete-box shape",
    ):
        load_score_index(path, label="old score rows")


def test_load_score_index_dense_scan_row_carrying_complete_box_fields_is_rejected(tmp_path: Path) -> None:
    """Regression class 1: a row explicitly labeled request_kind="dense_scan"
    must not be admitted as a scored complete-box row merely because it
    happens to also carry coord_token_ids/raw_model_logprob fields -- a
    declared kind that contradicts its own shape is a corrupt/ambiguous
    payload and is rejected, never silently scored or silently filtered."""

    path = tmp_path / "scores.jsonl"
    contradictory_row = {
        **_real_dense_scan_row(candidate_id="conditional-y1-plan:sha256:contradictory"),
        "coord_token_ids": [1, 2, 3, 4],
        "raw_model_logprob": {"complete_box_logprob_sum": 5.0},
    }
    _write_jsonl(path, [contradictory_row])
    with pytest.raises(
        ReanalysisError,
        match=r"declares request_kind='dense_scan' but carries complete-box-shaped fields",
    ):
        load_score_index(path, label="old score rows")


def test_load_score_index_unrecognized_request_kind_is_rejected(tmp_path: Path) -> None:
    """Regression class 2: only "complete_box" and "dense_scan" are the
    closed vocabulary. Any other declared request_kind -- a typo, a future
    predecessor row kind not yet supported here, or garbage -- must be
    rejected rather than silently treated as "some other filterable kind"."""

    path = tmp_path / "scores.jsonl"
    _write_jsonl(
        path,
        [
            {
                "candidate_id": "conditional-y1-plan:sha256:mystery-kind",
                "request_kind": "some_future_request_kind",
            }
        ],
    )
    with pytest.raises(
        ReanalysisError,
        match=r"has unrecognized request_kind 'some_future_request_kind'",
    ):
        load_score_index(path, label="old score rows")


def test_load_score_index_explicit_complete_box_lacking_required_shape_is_rejected(
    tmp_path: Path,
) -> None:
    """Regression class 3: a row explicitly labeled request_kind="complete_box"
    but missing the required complete-box identity/score fields must be
    rejected, not silently filtered as if it were some other row kind."""

    path = tmp_path / "scores.jsonl"
    _write_jsonl(
        path,
        [
            {
                "candidate_id": "free-tree-box:sha256:incomplete",
                "request_kind": "complete_box",
                # Missing coord_token_ids and raw_model_logprob entirely.
            }
        ],
    )
    with pytest.raises(
        ReanalysisError,
        match=r"declares request_kind='complete_box' but lacks the required complete-box shape",
    ):
        load_score_index(path, label="old score rows")


def test_load_score_index_never_fabricates_source_digest_to_force_a_match(tmp_path: Path) -> None:
    """A candidate that declares a real source_digest must NOT be silently
    matched against a predecessor row that has none -- that would be exactly
    the kind of identity weakening this fix must not introduce."""

    candidates = [
        _candidate(
            owner_context_id="owner:no-fabricate",
            rung="L0",
            candidate_id="owner:no-fabricate:t:0",
            box_tokens=[1, 2, 3, 4],
            population="target",
            region="target_strict",
            iou_to_target=0.9,
            source_digest="real-source-digest",
            use_real_schema=True,
        ),
    ]
    old_scores_path = tmp_path / "old-scores.jsonl"
    _write_jsonl(
        old_scores_path,
        [
            _real_complete_box_score_row(
                candidate_id="owner:no-fabricate:t:0", coord_token_ids=[1, 2, 3, 4], score=99.0
            )
        ],
    )
    candidates_path = tmp_path / "candidates.jsonl"
    _write_jsonl(candidates_path, candidates)
    index, _report = load_score_index(old_scores_path, label="old score rows")
    reused, unmatched = _match_candidates(
        [
            {
                "candidate_id": "owner:no-fabricate:t:0",
                "predecessor_candidate_id": None,
                "box_identity": (1, 2, 3, 4),
                "raw_row_identity": None,
                "source_digest": "real-source-digest",
            }
        ],
        index,
        reuse_disabled=False,
    )
    assert reused == {}
    assert unmatched == ["owner:no-fabricate:t:0"]


def test_load_score_index_ambiguous_duplicate_among_none_source_digest_rows_is_detected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scores.jsonl"
    _write_jsonl(
        path,
        [
            _real_complete_box_score_row(
                candidate_id="free-tree-box:sha256:dup", coord_token_ids=[9, 9, 9, 9], score=1.0
            ),
            _real_complete_box_score_row(
                candidate_id="free-tree-box:sha256:dup", coord_token_ids=[9, 9, 9, 9], score=2.0
            ),
        ],
    )
    with pytest.raises(ReanalysisError, match="ambiguous duplicate identity key"):
        load_score_index(path, label="old score rows")


def test_end_to_end_reuse_against_merged_predecessor_style_old_score_rows(tmp_path: Path) -> None:
    """A fixed-budget candidate that binds predecessor_candidate_id and
    declares no source_digest (matching the real predecessor row's own
    absence of one) is reused from a merged old-score-rows file mixing
    dense_scan and complete_box rows, producing a sealed report."""

    candidates, _unused_old_scores = _l0_owner_context("owner:merged-reuse")
    target_0 = next(row for row in candidates if row["candidate_id"] == "owner:merged-reuse:t:0")
    target_0["coord_token_ids"] = target_0.pop("box_tokens")
    target_0["predecessor_candidate_id"] = "free-tree-box:sha256:real-predecessor-row"
    del target_0["source_digest"]

    old_scores_path = tmp_path / "merged-old-scores.jsonl"
    _write_jsonl(
        old_scores_path,
        [
            _real_dense_scan_row(candidate_id="conditional-y1-plan:sha256:unrelated"),
            _real_complete_box_score_row(
                candidate_id="free-tree-box:sha256:real-predecessor-row",
                coord_token_ids=target_0["coord_token_ids"],
                score=42.0,
            ),
        ],
    )
    candidates_path = tmp_path / "candidates.jsonl"
    _write_jsonl(candidates_path, candidates)

    document = reanalyze_sorted_fn_fixed_budget_controls(
        fixed_budget_candidates=candidates_path,
        old_score_rows=old_scores_path,
        output=tmp_path / "report.json",
    )
    report = document["owner_contexts"]["owner:merged-reuse"]["L0"]
    assert "owner:merged-reuse:t:0" not in report["unmatched_candidate_ids"]
    assert report["per_threshold"]["0.5"]["target_peak"] == 42.0
    assert document["sources"]["old_score_rows_filter_report"] == {
        "total_rows": 2,
        "complete_box_rows": 1,
        "filtered_non_complete_box_rows": 1,
        "filtered_request_kinds": {"dense_scan": 1},
    }
