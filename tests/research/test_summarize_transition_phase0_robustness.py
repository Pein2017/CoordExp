from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/research/summarize_transition_phase0_robustness.py"
SPEC = importlib.util.spec_from_file_location("phase0_robustness", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _prediction(*, right: int = 100, close_step: int = 4) -> dict:
    return {
        "description": "person",
        "bbox": [0, 0, right, 100],
        "object_span_id": f"span-{close_step}",
        "pred_score_source": {"generated_step_indices": [0, close_step]},
    }


def _row(row_id: str, predictions: list[dict], stop: str) -> dict:
    return {
        "row_id": row_id,
        "image_width": 1000,
        "image_height": 1000,
        "decode_stop_reason": stop,
        "parse_status": "accepted",
        "gt": [{"object_id": f"owner-{row_id}", "description": "person", "bbox": [0, 0, 100, 100]}],
        "pred": predictions,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def paired_artifacts(tmp_path: Path) -> dict[str, Path]:
    # a: treatment gains at 0.50 but not 0.75; b: treatment loses; c: retains
    # and adds a strict duplicate; d: missed by both.
    source = [
        _row("a", [], "im_end"),
        _row("b", [_prediction(close_step=3)], "im_end"),
        _row("c", [_prediction(close_step=12)], "length"),
        _row("d", [], "length"),
    ]
    treatment = [
        _row("a", [_prediction(right=60, close_step=2)], "length"),
        _row("b", [], "im_end"),
        _row("c", [_prediction(close_step=12), _prediction(close_step=13)], "im_end"),
        _row("d", [], "length"),
    ]
    paths = {
        "source_raw": tmp_path / "source.raw.jsonl",
        "treatment_raw": tmp_path / "treatment.raw.jsonl",
        "source_scored": tmp_path / "source.scored.jsonl",
        "treatment_scored": tmp_path / "treatment.scored.jsonl",
        "source_trace": tmp_path / "source.trace.jsonl",
        "treatment_trace": tmp_path / "treatment.trace.jsonl",
        "owner": tmp_path / "owner-comparison.json",
    }
    _write_jsonl(paths["source_raw"], source)
    _write_jsonl(paths["treatment_raw"], treatment)
    _write_jsonl(paths["source_scored"], source)
    _write_jsonl(paths["treatment_scored"], treatment)
    _write_jsonl(paths["source_trace"], [{"row_id": "a", "step": 0}, {"row_id": "a", "step": 1}])
    _write_jsonl(paths["treatment_trace"], [{"row_id": "a", "step": 0}, {"row_id": "a", "step": 1}])
    paths["owner"].write_text(json.dumps({"owner_comparison": {"synthetic": True}}), encoding="utf-8")
    return paths


def _summarize(paths: dict[str, Path]) -> dict:
    return MODULE.summarize(
        source_raw=paths["source_raw"], treatment_raw=paths["treatment_raw"],
        source_scored=paths["source_scored"], treatment_scored=paths["treatment_scored"],
        source_trace=paths["source_trace"], treatment_trace=paths["treatment_trace"],
        owner_comparisons=[paths["owner"]], cutoffs=[10, 20],
    )


def test_symmetric_tail_trim_is_declared_and_symmetric() -> None:
    summary = MODULE.symmetric_trimmed_mean([-100, 1, 2, 3, 4, 5, 6, 7, 8, 100], 0.10)
    assert summary == {"proportion_each_tail": 0.10, "trim_each_tail_count": 1, "retained_count": 8, "mean": 4.5}


def test_paired_natural_stop_subset_excludes_any_length_stop(paired_artifacts: dict[str, Path]) -> None:
    result = _summarize(paired_artifacts)
    assert result["full_cohort"]["primary"]["image_count"] == 4
    subset = result["paired_natural_stop_subset"]
    assert subset["selection"] == {"source_natural_stop_count": 2, "treatment_natural_stop_count": 2, "paired_natural_stop_count": 1}
    assert subset["primary"]["image_count"] == 1


def test_cutoff_uses_complete_spans_closed_before_cutoff(paired_artifacts: dict[str, Path]) -> None:
    curve = _summarize(paired_artifacts)["fixed_token_cutoff_curve"]
    by_cutoff = {row["token_cutoff_exclusive"]: row for row in curve}
    # At 10, only a treatment's closing step 2 and b source's step 3 count.
    assert by_cutoff[10]["source"]["prediction_count"] == 1
    assert by_cutoff[10]["treatment"]["prediction_count"] == 1
    # c is admitted only once its closing steps 12 and 13 are complete.
    assert by_cutoff[20]["source"]["prediction_count"] == 2
    assert by_cutoff[20]["treatment"]["prediction_count"] == 3


def test_threshold_sensitivity_and_zero_safe_rates(paired_artifacts: dict[str, Path]) -> None:
    result = _summarize(paired_artifacts)
    sensitivity = result["full_cohort"]["threshold_sensitivity"]
    assert sensitivity["0.50"]["gained_owner_count"] == 1
    assert sensitivity["0.50"]["retained_owner_count"] == 1
    assert sensitivity["0.50"]["lost_owner_count"] == 1
    assert sensitivity["0.50"]["missed_by_both_owner_count"] == 1
    assert sensitivity["0.75"]["gained_owner_count"] == 0
    assert sensitivity["0.50"]["treatment"]["strict_duplicate_count"] == 1
    assert sensitivity["0.50"]["source"]["owner_yield"] > 0
    assert result["full_cohort"]["influence"]["category_concentration"]["gained"] == {"person": 1}


def test_zero_prediction_denominators_are_safe() -> None:
    row = {
        "gt_owner_count": 1, "source_owner_count": 0, "treatment_owner_count": 0,
        "gained_owner_count": 0, "retained_owner_count": 0, "lost_owner_count": 0,
        "missed_by_both_owner_count": 1, "net_owner_delta": 0,
        "source_prediction_count": 0, "treatment_prediction_count": 0,
        "source_strict_duplicate_count": 0, "treatment_strict_duplicate_count": 0,
    }
    aggregate = MODULE._aggregate([row])
    assert aggregate["source"]["owner_yield"] == 0.0
    assert aggregate["treatment"]["strict_duplicate_rate"] == 0.0


def test_output_is_deterministic(paired_artifacts: dict[str, Path], tmp_path: Path) -> None:
    first = _summarize(paired_artifacts)
    second = _summarize(paired_artifacts)
    assert first == second
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    MODULE.write_outputs(first, out_a)
    MODULE.write_outputs(second, out_b)
    for name in ("summary.json", "per_image.tsv", "token_cutoff.tsv", "threshold_sensitivity.tsv"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()
