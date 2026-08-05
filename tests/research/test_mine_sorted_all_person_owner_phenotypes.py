from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/research/mine_sorted_all_person_owner_phenotypes.py"
)
SPEC = importlib.util.spec_from_file_location("mine_owner_phenotypes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
miner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(miner)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    )


def _context(score: float) -> dict[str, object]:
    return {
        "exact_anchor": {
            "candidate_id": "exact",
            "score": score - 0.5,
            "competition_rank": 2,
            "tied_owner_count": 1,
        },
        "neighborhood": {
            "score": score,
            "competition_rank": 1,
            "margin": 0.25,
            "margin_sign": "positive",
            "point_outcome": "winner",
        },
        "peak": {
            "winning_candidate_score": score,
            "winning_candidate_ids": ["candidate"],
            "winning_transforms": ["translate_left"],
            "concentration": {
                "candidate_count": 8,
                "effective_candidate_count": 2.0,
                "normalized_entropy": 0.4,
                "top_candidate_probability": 0.6,
                "top1_minus_top2_score": 0.5,
            },
        },
        "realized_sidecars": {
            "entries": [],
            "finite_bank_undercoverage": False,
            "maximum_gap": None,
        },
        "ambiguity_bounds": {
            "status": "invariant",
            "outcome_invariant": True,
            "invariant_outcome": "winner",
            "best_rank": 1,
            "worst_rank": 1,
            "lower_margin": 0.25,
            "upper_margin": 0.25,
        },
    }


def _analyzer_rows(*, overlay_delta: float = 0.0) -> list[dict[str, object]]:
    rows = []
    for annotation_index in range(2, 43):
        owner_id = f"gt:7511:{annotation_index}"
        contexts = {
            context_id: _context(-float(annotation_index) - offset + overlay_delta)
            for offset, context_id in enumerate(miner.CONTEXT_IDS)
        }
        rows.append(
            {
                "schema_version": miner.ANALYZER_OWNER_SCHEMA,
                "unit_id": miner.UNIT_ID,
                "owner_id": owner_id,
                "original_annotation_index": annotation_index,
                "contexts": contexts,
                "geometry_and_neighbor_context": {
                    "overlap_other_owner_count": annotation_index % 3,
                    "strict_iou_0p5_other_owner_count": 0,
                    "sum_other_owner_iou": annotation_index / 100.0,
                    "max_other_owner_iou": annotation_index / 200.0,
                    "nearest_other_center_distance_pixels": float(annotation_index),
                    "nearest_other_center_distance_over_gt_diagonal": 1.0,
                    "neighbor_center_count_within_1x_gt_diagonal": annotation_index % 2,
                    "neighbor_center_count_within_2x_gt_diagonal": annotation_index % 4,
                },
            }
        )
    return rows


def _materialize_sources(root: Path) -> tuple[Path, Path, Path, Path]:
    analysis_dir = root / "analysis"
    census_path = analysis_dir / miner.OWNER_CENSUS_NAME
    _write_jsonl(census_path, _analyzer_rows())
    score_inputs = []
    for context_offset, context_id in enumerate(miner.CONTEXT_IDS):
        score_path = root / "scores" / context_id / "route-landscape-scores.jsonl"
        score_receipt_path = root / "scores" / context_id / "receipt.json"
        score_rows = []
        for annotation_index in range(2, 43):
            owner_id = f"gt:7511:{annotation_index}"
            for candidate_index in range(9):
                score_rows.append(
                    {
                        "candidate_id": (
                            f"primary:{owner_id}:{candidate_index:02d}:candidate_{candidate_index}"
                        ),
                        "context_id": context_id,
                        "request_kind": "primary",
                        "primary_role": True,
                        "raw_model_logprob": {
                            "complete_box_logprob_sum": (
                                -100.0
                                + annotation_index
                                + context_offset
                                + candidate_index / 10.0
                            )
                        },
                    }
                )
        _write_jsonl(score_path, score_rows)
        _write_json(score_receipt_path, {"context_id": context_id})
        score_inputs.append(
            {
                "score_path": str(score_path),
                "score_sha256": miner.sha256_file(score_path),
                "receipt_path": str(score_receipt_path),
                "receipt_sha256": miner.sha256_file(score_receipt_path),
            }
        )
    analysis_receipt = {
        "schema_version": miner.ANALYZER_RECEIPT_SCHEMA,
        "unit_id": miner.UNIT_ID,
        "outputs": {
            miner.OWNER_CENSUS_NAME: {
                "sha256": miner.sha256_file(census_path),
                "rows": 41,
            }
        },
        "score_inputs": score_inputs,
    }
    _write_json(analysis_dir / miner.ANALYZER_RECEIPT_NAME, analysis_receipt)

    owner_ledger_path = root / "task0" / "owner-ledger.jsonl"
    behavior_ledger_path = root / "task0" / "owner-trajectory-matrix.jsonl"
    cohort_ledger_path = root / "cohorts" / "cohort-assignments.jsonl"
    owner_rows = []
    behavior_rows = []
    cohort_rows = []
    for annotation_index in range(2, 43):
        owner_id = f"gt:7511:{annotation_index}"
        owner_rows.append(
            {
                "gt_owner_id": owner_id,
                "image_id": "7511",
                "original_annotation_index": annotation_index,
                "bbox_xyxy": [annotation_index, annotation_index * 2, annotation_index + 5, annotation_index * 2 + 7],
            }
        )
        greedy = annotation_index in {2, 22, 32}
        rescued = annotation_index == 17
        behavior_rows.append(
            {
                "gt_owner_id": owner_id,
                "policy_stratum": "primary_rp_1.00",
                "decode_mode": "greedy",
                "seed": 0,
                "strict_match_presence": greedy,
            }
        )
        behavior_rows.extend(
            {
                "gt_owner_id": owner_id,
                "policy_stratum": "primary_rp_1.00",
                "decode_mode": "sampled",
                "seed": seed,
                "strict_match_presence": rescued,
            }
            for seed in range(21001, 21017)
        )
        cohort_rows.append(
            {
                "gt_owner_id": owner_id,
                "image_id": "7511",
                "cohort": "existing_sealed_status",
                "primary_strict_support": {
                    "rp1_00_greedy": greedy,
                    "strict_rescued": rescued,
                },
                "natural_spatial_support": {
                    "admits_loose_only_b1": annotation_index % 5 == 0,
                    "no_free_spatial_support": annotation_index % 7 == 0,
                },
                "null_status": {"registered_sampling": "evaluated"},
            }
        )
    _write_jsonl(owner_ledger_path, owner_rows)
    _write_jsonl(behavior_ledger_path, behavior_rows)
    _write_jsonl(cohort_ledger_path, cohort_rows)
    return analysis_dir, owner_ledger_path, behavior_ledger_path, cohort_ledger_path


def test_stable_bank_statistics_handles_large_negative_scores() -> None:
    stats = miner.stable_bank_statistics([-10_000.0, -10_001.0, -10_002.0])

    assert stats["candidate_count"] == 3
    assert stats["logsumexp"] == pytest.approx(-9_999.592394035555)
    assert stats["logmeanexp"] == pytest.approx(stats["logsumexp"] - math.log(3))
    assert 1.0 < stats["effective_candidate_count"] < 3.0
    assert 0.0 < stats["top_candidate_probability"] < 1.0


def test_behavior_false_is_no_positive_evidence_not_absence() -> None:
    behavior = miner.build_behavior_status(
        {
            "cohort": "no_free_spatial_support",
            "primary_strict_support": {
                "rp1_00_greedy": False,
                "strict_rescued": None,
            },
            "natural_spatial_support": {
                "admits_loose_only_b1": False,
                "no_free_spatial_support": True,
            },
            "null_status": {},
        },
        {
            "greedy_strict_positive": False,
            "strict_rescued": None,
            "sampled_strict_positive_count": 0,
            "sampled_trajectory_count": 16,
        },
    )

    assert behavior["greedy"]["status"] == "no_positive_evidence"
    assert behavior["strict_rescued"]["status"] == "missing"
    assert behavior["no_free"]["status"] == "no_free_positive"
    assert "never establishes visual absence" in behavior["interpretation_boundary"]


def test_realistic_full_join_overlay_and_deterministic_outputs(tmp_path: Path) -> None:
    analysis_dir, owner_ledger, behavior_ledger, cohort_ledger = _materialize_sources(tmp_path)
    rows, summary, sources = miner.mine(
        analysis_dir, owner_ledger, behavior_ledger, cohort_ledger
    )

    assert len(rows) == 41
    assert rows[0]["owner_id"] == "gt:7511:2"
    assert rows[0]["gt_geometry"]["sorted_index_zero_based"] == 0
    assert rows[0]["contexts"]["root"]["candidate_bank"]["candidate_count"] == 9
    assert rows[0]["contexts"]["root"]["candidate_bank"]["logmeanexp"] < rows[0]["contexts"]["root"]["candidate_bank"]["maximum_score"]
    assert sum(row["scorer_to_behavior_agreement"] is not None for row in rows) == 4
    assert summary["clustering_performed"] is False
    assert summary["hard_phenotype_labels_emitted"] is False
    assert summary["correlations"]

    output_dir = tmp_path / "output"
    first = miner.write_outputs(output_dir, rows, summary, sources)
    second = miner.write_outputs(output_dir, rows, summary, sources)
    assert first["feature_status"] == "created"
    assert second["feature_status"] == "identical_existing_output"
    receipt = json.loads((output_dir / miner.RECEIPT_NAME).read_text())
    reconstructed = miner.sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    assert receipt["receipt_content_sha256"] == reconstructed

    overlay_dir = tmp_path / "overlay"
    _write_jsonl(overlay_dir / miner.OWNER_CENSUS_NAME, _analyzer_rows(overlay_delta=10.0))
    overlay_rows, _, _ = miner.mine(
        analysis_dir,
        owner_ledger,
        behavior_ledger,
        cohort_ledger,
        overlay_dir,
    )
    assert (
        overlay_rows[0]["contexts"]["root"]["exact_anchor"]["score"]
        == rows[0]["contexts"]["root"]["exact_anchor"]["score"] + 10.0
    )
    assert (
        overlay_rows[0]["contexts"]["root"]["candidate_bank"]["logsumexp"]
        == rows[0]["contexts"]["root"]["candidate_bank"]["logsumexp"]
    )
    assert (
        overlay_rows[0]["contexts"]["root"]["analyzer_value_source"]
        == "scalar_overlay_analyzer_output"
    )


def test_mine_rejects_tampered_sealed_score(tmp_path: Path) -> None:
    analysis_dir, owner_ledger, behavior_ledger, cohort_ledger = _materialize_sources(tmp_path)
    score_path = tmp_path / "scores" / "root" / "route-landscape-scores.jsonl"
    score_path.write_text(score_path.read_text() + "{}\n")

    with pytest.raises(miner.MiningContractError, match="score input digest mismatch"):
        miner.mine(analysis_dir, owner_ledger, behavior_ledger, cohort_ledger)
