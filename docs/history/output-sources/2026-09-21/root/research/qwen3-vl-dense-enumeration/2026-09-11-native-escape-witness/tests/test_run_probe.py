from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


RUNNER = Path(__file__).parents[1] / "run_probe.py"
SPEC = importlib.util.spec_from_file_location("native_escape_witness_runner", RUNNER)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def golden() -> dict:
    return {
        "example_id": "case",
        "gt": [],
        "image_height": 1000,
        "image_path": "/tmp/image.jpg",
        "image_width": 1000,
        "row_id": "case",
        "row_index": 0,
    }


def test_real_parser_keeps_complete_invalid_free_row_out_of_forced_credit() -> None:
    from probes.source_rweak_row_cross.run import native_record

    forced = (
        "<|object_ref_start|>cake<|object_ref_end|><|box_start|>"
        "<|coord_100|><|coord_100|><|coord_200|><|coord_200|><|box_end|>"
    )
    invalid_free = (
        "<|object_ref_start|>knife<|object_ref_end|><|box_start|>"
        "<|coord_300|><|coord_300|><|coord_300|><|coord_400|><|box_end|>"
    )
    valid_free = (
        "<|object_ref_start|>cup<|object_ref_end|><|box_start|>"
        "<|coord_400|><|coord_400|><|coord_500|><|coord_500|><|box_end|>"
    )
    parsed = native_record(forced + invalid_free + valid_free,
                           {"row_id": "case"}, golden(), "length")
    partition = runner.partition_parser_rows(parsed, prefix_row_count=1)
    assert partition["counts"] == {
        "all_free_rows_or_fragments": 2,
        "valid_complete_free_rows": 1,
        "invalid_complete_free_rows": 1,
        "incomplete_free_fragments": 0,
    }
    assert partition["complete_invalid_free_rows"][0]["generated_order"] == 1
    credit = runner.credit_identity(
        {"kind": "h_plus_c", "candidate": {"candidate_id": "c0"}},
        partition,
    )
    assert credit["forced_candidate_row_count"] == 1
    assert credit["forced_candidate_in_free_counts"] is False
    assert credit["credited_free_generated_orders"] == [2]
    assert credit["natural_owner_recovery_eligible"] is False


def test_record_consumer_binds_h_plus_literal_c_plus_fresh_suffix() -> None:
    candidate = {"candidate_id": "c0"}
    case = {
        "case_id": "351017",
        "expected_h_only_first512_ids": [],
    }
    job = {
        "job_id": "h_plus_c0",
        "kind": "h_plus_c",
        "prefix_ids": [10, 11, 12, 13],
        "budget": 2,
        "candidate": candidate,
    }
    record = {
        "schema": runner.RECORD_SCHEMA,
        "packet_sha256": "a" * 64,
        "case_id": "351017",
        "job_id": "h_plus_c0",
        "prefix_ids": [10, 11, 12, 13],
        "free_ids": [20, runner.EOS],
        "action_ids": [10, 11, 12, 13, 20, runner.EOS],
        "stop_reason": "im_end",
        "credit_identity": {
            "forced_candidate_row_count": 1,
            "forced_candidate_in_free_counts": False,
        },
    }
    runner.validate_record(record, case=case, job=job, packet_sha256="a" * 64)
    corrupted = dict(record, action_ids=[10, 11, 12, 20, runner.EOS])
    with pytest.raises(ValueError, match="prefix plus free"):
        runner.validate_record(corrupted, case=case, job=job,
                               packet_sha256="a" * 64)


def test_candidate_consumer_rejects_non_original_or_late_source() -> None:
    ids = [1, 2, 3]
    raw = (
        "<|object_ref_start|>cake<|object_ref_end|><|box_start|>"
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|><|box_end|>"
    )
    candidate = {
        "candidate_id": "c0",
        "c_ids": ids,
        "c_ids_sha256": runner.digest_json(ids),
        "geometry_valid": True,
        "source_job_id": "early_original_translated",
        "source_image_condition": "original",
        "source_free_complete_order": 0,
        "max_class_blind_iou_to_h": 0.0,
        "visual_admission": "root_accepted",
        "raw_span_text": raw,
        "raw_span_sha256": __import__("hashlib").sha256(raw.encode()).hexdigest(),
    }
    assert runner.validate_complete_row(candidate, case_id="351017")["c_ids"] == ids
    for field, bad in (
        ("source_job_id", "late_original_translated"),
        ("source_image_condition", "donor"),
    ):
        changed = json.loads(json.dumps(candidate))
        changed[field] = bad
        with pytest.raises(ValueError):
            runner.validate_complete_row(changed, case_id="351017")


def test_smoke_is_complete_351017_case_and_is_reused_as_final_rank() -> None:
    case = {
        "case_id": "351017",
        "h_ids": [1, 2],
        "candidates": [
            {"candidate_id": "c01", "c_ids": [3]},
            {"candidate_id": "c02", "c_ids": [4]},
        ],
    }
    assert [job["job_id"] for job in runner.selected_jobs(case, smoke=True)] == [
        "natural_anchor",
        "h_only",
        "h_plus_c01",
        "h_plus_c02",
    ]


def test_batch_identity_consumes_real_native_batch_media_sha256() -> None:
    import torch
    from src.qwen.native import NativeBatch

    batch = NativeBatch(
        inputs={
            "input_ids": torch.tensor([[10, 11]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "image_grid_thw": torch.tensor([[1, 2, 3]], dtype=torch.long),
        },
        request_ids=("case",),
        media_sha256=("a" * 64,),
    )
    identity = runner.batch_identity(batch)
    assert identity["value"] == {
        "request_ids": ["case"],
        "prompt_token_ids": [[10, 11]],
        "image_grids": [[1, 2, 3]],
        "media_sha256": ["a" * 64],
    }
