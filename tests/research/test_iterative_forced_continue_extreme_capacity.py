from __future__ import annotations

import importlib
from pathlib import Path

analyze = importlib.import_module(
    "scripts.research.analyze_iterative_forced_continue_extreme_capacity"
)
runner = importlib.import_module(
    "scripts.research.run_iterative_forced_continue_extreme_capacity"
)


class _Session:
    @staticmethod
    def _im_end_token_id() -> int:
        return 99


def _complete_row(token_ids: list[int], description: str, bbox: list[float]) -> dict:
    return {
        "status": "success",
        "raw_generated_token_ids": token_ids,
        "row_stop": {"stop_reason": "complete_row"},
        "parsed_predictions": [{"description": description, "bbox": bbox}],
    }


def _terminal(token_ids: list[int]) -> dict:
    return {
        "status": "success",
        "raw_generated_token_ids": token_ids,
        "row_stop": {"stop_reason": "terminal"},
        "parsed_predictions": [],
    }


def _owners() -> list[dict]:
    return [
        {"owner_id": "1:a", "category": "a", "bbox": (0.0, 0.0, 10.0, 10.0)},
        {"owner_id": "1:b", "category": "b", "bbox": (20.0, 20.0, 30.0, 30.0)},
    ]


def test_replaces_each_clean_terminal_without_counting_stop(monkeypatch) -> None:
    natural_rows = iter(
        [
            _complete_row([10, 11], "a", [0.0, 0.0, 10.0, 10.0]),
            _terminal([99]),
        ]
    )
    forced_rows = iter(
        [
            _complete_row(
                [runner.OBJECT_REF_START, 20],
                "b",
                [20.0, 20.0, 30.0, 30.0],
            )
        ]
    )
    monkeypatch.setattr(runner, "_generate_row", lambda **_: next(natural_rows))
    monkeypatch.setattr(
        runner,
        "_generate_after_forced_partial_row",
        lambda **_: next(forced_rows),
    )
    monkeypatch.setattr(
        runner,
        "_probe_complete_boundary",
        lambda **_: {"counted_in_completion_budget": False, "would_naturally_stop": True},
    )

    case = runner._run_case(
        session=_Session(),
        native_inputs={},
        tokenizer=None,
        image_id="1",
        owners=_owners(),
        image_width=100,
        image_height=100,
        repetition_penalty=1.0,
        max_new_tokens=16,
        malformed_limit=2,
    )

    assert case["terminal_reason"] == "all_gt_matched"
    assert case["native_snapshot"]["coverage"] == 1
    assert case["final_snapshot"]["coverage"] == 2
    assert case["force_count"] == 1
    assert case["forces"][0]["marginal_owner_gain"] == 1
    assert case["executed_completion_token_ids"] == [
        10,
        11,
        runner.OBJECT_REF_START,
        20,
    ]
    assert 99 not in case["executed_completion_token_ids"]
    assert case["post_complete_boundary_probe"]["would_naturally_stop"] is True


def test_nonclean_terminal_is_not_repaired(monkeypatch) -> None:
    natural_rows = iter(
        [
            _complete_row([10], "a", [0.0, 0.0, 10.0, 10.0]),
            _terminal([88, 99]),
        ]
    )
    monkeypatch.setattr(runner, "_generate_row", lambda **_: next(natural_rows))
    monkeypatch.setattr(
        runner,
        "_generate_after_forced_partial_row",
        lambda **_: (_ for _ in ()).throw(AssertionError("must not force malformed terminal")),
    )

    case = runner._run_case(
        session=_Session(),
        native_inputs={},
        tokenizer=None,
        image_id="1",
        owners=_owners(),
        image_width=100,
        image_height=100,
        repetition_penalty=1.1,
        max_new_tokens=16,
        malformed_limit=2,
    )

    assert case["force_count"] == 0
    assert case["executed_completion_token_ids"] == [10, 88, 99]
    assert case["terminal_reason"] == "malformed_or_incomplete_natural_row:terminal"
    assert case["final_snapshot"]["coverage"] == 1


def test_summary_reports_force_productivity_and_curves(tmp_path: Path) -> None:
    case = {
        "image_id": "1",
        "native_boundary_observed": True,
        "native_snapshot": {
            "coverage": 1,
            "gt_owner_count": 2,
            "prediction_count": 1,
            "false_positive_count": 0,
            "matched_owner_ids": ["1:a"],
        },
        "final_snapshot": {
            "coverage": 2,
            "gt_owner_count": 2,
            "prediction_count": 3,
            "false_positive_count": 1,
            "matched_owner_ids": ["1:a", "1:b"],
        },
        "forces": [
            {"marginal_owner_gain": 0},
            {"marginal_owner_gain": 1},
        ],
        "executed_completion_token_count": 300,
        "complete_row_count": 3,
        "terminal_reason": "all_gt_matched",
        "post_complete_boundary_probe": {"would_naturally_stop": False},
        "coverage_curve": [
            {"executed_token_count": 0, "force_count": 0, "coverage": 0},
            {"executed_token_count": 100, "force_count": 0, "coverage": 1},
            {"executed_token_count": 200, "force_count": 1, "coverage": 1},
            {"executed_token_count": 300, "force_count": 2, "coverage": 2},
        ],
    }
    payload = {
        "schema_version": runner.SCHEMA_VERSION,
        "config": {
            "infer_config": "/tmp/config.yaml",
            "infer_config_sha256": "abc",
            "repetition_penalty": 1.0,
        },
        "case_count": 1,
        "cases": [case],
    }
    result = analyze._summarize_arm(payload, source=tmp_path / "run.json")

    assert result["aggregate"]["coverage_gain"] == 1
    assert result["aggregate"]["force_productivity"] == 0.5
    assert result["aggregate"]["owners_per_force"] == 0.5
    assert result["aggregate"]["final_f1"] == 0.8
    assert result["equal_token_budget_curve"][0]["coverage"] == 1
    assert result["equal_token_budget_curve"][1]["coverage"] == 1
    assert result["equal_force_count_curve"][0]["coverage"] == 1
    assert result["equal_force_count_curve"][1]["coverage"] == 1
    assert result["equal_force_count_curve"][2]["coverage"] == 2
