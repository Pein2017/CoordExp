from __future__ import annotations

import importlib
import json
from pathlib import Path

import torch

from scripts.research.run_local_branch_causal_value import hash_prefix_token_ids
from src.templates.renderer import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    OBJECT_REF_END_TOKEN,
    OBJECT_REF_START_TOKEN,
)


runner = importlib.import_module(
    "scripts.research.run_iterative_forced_continue_exact_native"
)
analyzer = importlib.import_module(
    "scripts.research.analyze_iterative_forced_continue_extreme_capacity"
)


class _Session:
    @staticmethod
    def _im_end_token_id() -> int:
        return 99


def _owners(count: int = 2) -> list[dict]:
    values = [
        {"owner_id": "1:a", "category": "a", "bbox": (0.0, 0.0, 10.0, 10.0)},
        {"owner_id": "1:b", "category": "b", "bbox": (20.0, 20.0, 30.0, 30.0)},
        {"owner_id": "1:c", "category": "c", "bbox": (40.0, 40.0, 50.0, 50.0)},
    ]
    return values[:count]


def _prediction(description: str, bbox: list[float]) -> dict:
    return {"description": description, "bbox": bbox}


def _segment(
    token_ids: list[int],
    predictions: list[dict],
    *,
    stop_reason: str = "im_end",
    clean: bool = True,
) -> dict:
    complete_rows = []
    if clean and token_ids:
        complete_rows.append(
            {
                "raw_generated_token_ids": token_ids,
                "parsed_predictions": predictions,
            }
        )
    return {
        "status": "success",
        "stop_reason": stop_reason,
        "raw_generated_token_ids": token_ids,
        "raw_generated_token_ids_sha256": hash_prefix_token_ids(token_ids),
        "raw_generated_text": "",
        "segment_is_clean_complete_rows": clean,
        "complete_rows": complete_rows,
        "parse_evidence": {},
        "parsed_predictions": predictions,
    }


def _forced_row(token_ids: list[int], description: str, bbox: list[float]) -> dict:
    return {
        "status": "success",
        "raw_generated_token_ids": token_ids,
        "row_stop": {"stop_reason": "complete_row"},
        "parsed_predictions": [_prediction(description, bbox)],
    }


def test_natural_segment_with_empty_prefix_preserves_native_prompt(monkeypatch) -> None:
    observed: dict = {}

    def fake_sample_one(**kwargs):
        observed.update(kwargs)
        return [], "", "im_end"

    monkeypatch.setattr(runner, "_sample_one", fake_sample_one)
    native_inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    result = runner._generate_natural_segment(
        session=_Session(),
        native_inputs=native_inputs,
        prefix_token_ids=[],
        tokenizer=None,
        image_width=100,
        image_height=100,
        repetition_penalty=1.0,
        max_new_tokens=3084,
        row_index=0,
    )

    assert observed["prompt_width"] == 3
    assert torch.equal(observed["native_inputs"]["input_ids"], native_inputs["input_ids"])
    assert torch.equal(
        observed["native_inputs"]["attention_mask"], native_inputs["attention_mask"]
    )
    assert result["stop_reason"] == "im_end"
    assert result["raw_generated_token_ids"] == []


def test_replays_native_segment_before_one_forced_row(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_natural(**kwargs):
        calls.append({**kwargs, "prefix_token_ids": list(kwargs["prefix_token_ids"])})
        return _segment(
            [10, 11],
            [_prediction("a", [0.0, 0.0, 10.0, 10.0])],
        )

    monkeypatch.setattr(runner, "_generate_natural_segment", fake_natural)
    monkeypatch.setattr(
        runner,
        "_generate_after_forced_partial_row",
        lambda **_: _forced_row(
            [runner.OBJECT_REF_START, 20],
            "b",
            [20.0, 20.0, 30.0, 30.0],
        ),
    )
    monkeypatch.setattr(
        runner,
        "_probe_complete_boundary",
        lambda **_: {"counted_in_completion_budget": False, "would_naturally_stop": True},
    )

    case = runner._run_case_exact_native(
        session=_Session(),
        native_inputs={},
        tokenizer=None,
        image_id="1",
        owners=_owners(2),
        image_width=100,
        image_height=100,
        repetition_penalty=1.0,
        max_new_tokens=16,
        malformed_limit=2,
    )

    assert calls[0]["prefix_token_ids"] == []
    assert calls[0]["max_new_tokens"] == 16
    assert case["native_snapshot"]["coverage"] == 1
    assert case["native_snapshot"]["generated_token_ids_sha256"] == hash_prefix_token_ids(
        [10, 11]
    )
    assert case["final_snapshot"]["coverage"] == 2
    assert case["executed_completion_token_ids"] == [
        10,
        11,
        runner.OBJECT_REF_START,
        20,
    ]
    assert case["force_count"] == 1
    assert case["forces"][0]["marginal_owner_gain"] == 1
    assert case["forces"][0]["interval_owner_gain"] == 1
    assert case["terminal_reason"] == "all_gt_matched"


def test_release_after_force_is_one_uninterrupted_natural_segment(monkeypatch) -> None:
    calls: list[dict] = []
    segments = iter(
        [
            _segment([10], [_prediction("a", [0.0, 0.0, 10.0, 10.0])]),
            _segment([30, 31], [_prediction("c", [40.0, 40.0, 50.0, 50.0])]),
        ]
    )

    def fake_natural(**kwargs):
        calls.append({**kwargs, "prefix_token_ids": list(kwargs["prefix_token_ids"])})
        return next(segments)

    monkeypatch.setattr(runner, "_generate_natural_segment", fake_natural)
    monkeypatch.setattr(
        runner,
        "_generate_after_forced_partial_row",
        lambda **_: _forced_row(
            [runner.OBJECT_REF_START, 20],
            "b",
            [20.0, 20.0, 30.0, 30.0],
        ),
    )

    case = runner._run_case_exact_native(
        session=_Session(),
        native_inputs={},
        tokenizer=None,
        image_id="1",
        owners=_owners(3),
        image_width=100,
        image_height=100,
        repetition_penalty=1.0,
        max_new_tokens=16,
        malformed_limit=2,
    )

    assert calls[0]["prefix_token_ids"] == []
    assert calls[1]["prefix_token_ids"] == [10, runner.OBJECT_REF_START, 20]
    assert calls[1]["max_new_tokens"] == 13
    assert case["final_snapshot"]["coverage"] == 3
    assert case["force_count"] == 1
    assert case["forces"][0]["marginal_owner_gain"] == 1
    assert case["forces"][0]["interval_owner_gain"] == 2
    assert case["forces"][0]["interval_termination"] == "natural_segment:im_end"
    assert case["post_complete_boundary_probe"] == {
        "counted_in_completion_budget": False,
        "would_naturally_stop": True,
        "evidence": "observed exact natural segment terminal",
    }


def test_one_remaining_budget_executes_exactly_one_forced_opener(monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "_generate_natural_segment",
        lambda **_: _segment([], []),
    )
    monkeypatch.setattr(
        runner,
        "_generate_after_forced_partial_row",
        lambda **_: (_ for _ in ()).throw(AssertionError("tail generation must not run")),
    )

    case = runner._run_case_exact_native(
        session=_Session(),
        native_inputs={},
        tokenizer=None,
        image_id="1",
        owners=_owners(1),
        image_width=100,
        image_height=100,
        repetition_penalty=1.0,
        max_new_tokens=1,
        malformed_limit=2,
    )

    assert case["executed_completion_token_ids"] == [runner.OBJECT_REF_START]
    assert case["executed_completion_token_count"] == 1
    assert case["force_count"] == 1
    assert case["terminal_reason"] == "token_budget_after_forced_opener"


def test_malformed_terminal_does_not_count_trimmed_im_end(monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "_generate_natural_segment",
        lambda **_: _segment([88], [], clean=False),
    )
    monkeypatch.setattr(
        runner,
        "_generate_after_forced_partial_row",
        lambda **_: (_ for _ in ()).throw(AssertionError("malformed terminal must not force")),
    )

    case = runner._run_case_exact_native(
        session=_Session(),
        native_inputs={},
        tokenizer=None,
        image_id="1",
        owners=_owners(1),
        image_width=100,
        image_height=100,
        repetition_penalty=1.1,
        max_new_tokens=16,
        malformed_limit=2,
    )

    assert case["executed_completion_token_ids"] == [88]
    assert 99 not in case["executed_completion_token_ids"]
    assert case["force_count"] == 0
    assert case["terminal_reason"] == "malformed_or_incomplete_natural_segment:im_end"


def test_clean_segment_requires_only_complete_rows_and_whitespace() -> None:
    row = (
        f"{OBJECT_REF_START_TOKEN}a{OBJECT_REF_END_TOKEN}{BOX_START_TOKEN}"
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
        f"{BOX_END_TOKEN}"
    )
    assert runner._segment_is_clean_complete_rows("")
    assert runner._segment_is_clean_complete_rows(f"\n{row}\n{row}\n")
    assert not runner._segment_is_clean_complete_rows(f"junk{row}")
    assert not runner._segment_is_clean_complete_rows(row[:-1])


def test_native_reference_loader_enforces_decode_identity(tmp_path: Path) -> None:
    path = tmp_path / "native.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "current_seeded_sampled_rollouts.v1",
                "config": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "repetition_penalty": 1.0,
                    "max_new_tokens": 3084,
                },
                "rollouts": [
                    {"image_id": 1584, "generated_token_ids_sha256": "abc"},
                    {"image_id": 1754, "generated_token_ids_sha256": "def"},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert runner._load_native_reference(
        path,
        repetition_penalty=1.0,
        max_new_tokens=3084,
    ) == {"1584": "abc", "1754": "def"}


def test_native_reference_loader_rejects_wrong_repetition_penalty(tmp_path: Path) -> None:
    path = tmp_path / "native.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "current_seeded_sampled_rollouts.v1",
                "config": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "repetition_penalty": 1.0,
                    "max_new_tokens": 3084,
                },
                "rollouts": [
                    {"image_id": 1584, "generated_token_ids_sha256": "abc"}
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        runner._load_native_reference(
            path,
            repetition_penalty=1.1,
            max_new_tokens=3084,
        )
    except ValueError as exc:
        assert "repetition_penalty" in str(exc)
    else:
        raise AssertionError("wrong repetition penalty must fail closed")


def _summary_payload(*, schema_version: str) -> dict:
    snapshot = {
        "coverage": 1,
        "gt_owner_count": 1,
        "prediction_count": 1,
        "false_positive_count": 0,
        "matched_owner_ids": ["1:a"],
    }
    return {
        "schema_version": schema_version,
        "config": {
            "infer_config": "/tmp/config.yaml",
            "infer_config_sha256": "abc",
            "repetition_penalty": 1.0,
        },
        "case_count": 1,
        "cases": [
            {
                "image_id": "1",
                "native_boundary_observed": True,
                "native_snapshot": {
                    **snapshot,
                    "generated_token_ids_sha256": "native-sha",
                },
                "native_replay": {"status": "exact_match"},
                "final_snapshot": snapshot,
                "force_count": 0,
                "forces": [],
                "executed_completion_token_count": 2,
                "executed_completion_token_ids": [10, 11],
                "complete_row_count": 1,
                "terminal_reason": "all_gt_matched",
                "post_complete_boundary_probe": {"would_naturally_stop": True},
                "coverage_curve": [
                    {"executed_token_count": 0, "force_count": 0, "coverage": 0},
                    {"executed_token_count": 2, "force_count": 0, "coverage": 1},
                ],
            }
        ],
    }


def test_summary_accepts_exact_native_receipt_and_preserves_replay_sha(
    tmp_path: Path,
) -> None:
    path = tmp_path / "exact.json"
    path.write_text(json.dumps(_summary_payload(schema_version=runner.SCHEMA_VERSION)))

    result = analyzer.summarize([str(path)])

    assert result["run_schema_version"] == runner.SCHEMA_VERSION
    assert result["arms"][0]["aggregate"]["exact_native_replay_cases"] == 1
    assert result["arms"][0]["cases"][0]["native_generated_token_ids_sha256"] == (
        "native-sha"
    )


def test_summary_refuses_to_mix_rowwise_and_exact_native_schemas(tmp_path: Path) -> None:
    exact = tmp_path / "exact.json"
    rowwise = tmp_path / "rowwise.json"
    exact.write_text(json.dumps(_summary_payload(schema_version=runner.SCHEMA_VERSION)))
    rowwise_payload = _summary_payload(
        schema_version="iterative_forced_continue_extreme_capacity.v1"
    )
    rowwise_payload["config"]["infer_config_sha256"] = "def"
    rowwise.write_text(json.dumps(rowwise_payload))

    try:
        analyzer.summarize([str(exact), str(rowwise)])
    except ValueError as exc:
        assert "mix" in str(exc)
    else:
        raise AssertionError("rowwise and exact-native artifacts must not share one summary")
