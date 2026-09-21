from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "endpoint_eval.py"
SPEC = importlib.util.spec_from_file_location("positive_branch_endpoint_eval", MODULE_PATH)
assert SPEC and SPEC.loader
endpoint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(endpoint)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def adapter_receipt(tmp_path: Path, arm: str, payload: bytes) -> dict:
    adapter = tmp_path / arm / "adapter"
    adapter.mkdir(parents=True)
    model = adapter / "adapter_model.safetensors"
    model.write_bytes(payload)
    config = adapter / "adapter_config.json"
    config.write_text("{}\n")
    files = [
        {"relative_path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size_bytes": p.stat().st_size}
        for p in (config, model)
    ]
    return {
        "schema": "repeat_recovery_train.receipt.v1",
        "status": "completed",
        "scientific_status": "candidate",
        "arm": arm,
        "mode": "full",
        "updates": 32,
        "stop_reason": "fixed_32_updates",
        "saved_adapter": {"root": str(adapter), "fingerprint": digest(files), "files": files,
                          "file_count": 2},
        "composition": {"base_model_path": "/base"},
    }


def test_arbitrary_adapter_selection_is_receipt_bound(tmp_path):
    a = adapter_receipt(tmp_path, "A", b"arm-a")
    b = adapter_receipt(tmp_path, "B", b"arm-b")
    selected_a = endpoint.validate_adapter_export(a, arm="A", inspect_payload=lambda *_: a["saved_adapter"])
    selected_b = endpoint.validate_adapter_export(b, arm="B", inspect_payload=lambda *_: b["saved_adapter"])
    assert selected_a["root"] != selected_b["root"]
    assert selected_a["fingerprint"] == a["saved_adapter"]["fingerprint"]
    assert selected_b["fingerprint"] == b["saved_adapter"]["fingerprint"]
    a["saved_adapter"]["files"][1]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="adapter payload changed"):
        endpoint.validate_adapter_export(a, arm="A", inspect_payload=lambda *_: a["saved_adapter"])


def test_conditional_partition_keeps_dropped_row_and_excludes_forced_credit():
    h = [10, 11]
    c = [12, 13]
    job = endpoint.ConditionalJob(
        arm="A", rank=1, case_id="1", candidate_id="1-c01", kind="h_plus_c",
        prefix_ids=h + c, prefix_row_count=2, forced_candidate={"candidate_id": "1-c01"},
        budget=8,
    )
    parsed = {
        "pred": [
            {"generated_order": 0, "bbox": [0, 0, 2, 2]},
            {"generated_order": 1, "bbox": [3, 3, 5, 5]},
            {"generated_order": 2, "bbox": [6, 6, 8, 8]},
        ],
        "dropped_predictions": [{
            "generated_order": 3, "reason": "geometry_invalid",
            "raw_span_text": "bad<|box_end|>",
        }],
    }
    record = endpoint.conditional_artifact(job, free_ids=list(range(20, 28)), stop_reason="length", parsed=parsed)
    assert record["parser_partition"]["counts"] == {
        "all_free_rows_or_fragments": 2,
        "valid_complete_free_rows": 1,
        "invalid_complete_free_rows": 1,
        "incomplete_free_fragments": 0,
    }
    assert record["credit_identity"]["forced_candidate_row_count"] == 1
    assert record["credit_identity"]["forced_candidate_in_free_counts"] is False
    dropped_miscredit = copy.deepcopy(record)
    dropped_miscredit["parser_partition"]["counts"]["invalid_complete_free_rows"] = 0
    with pytest.raises(ValueError, match="parser partition differs"):
        endpoint.validate_conditional_artifact(dropped_miscredit, job)
    forced_miscredit = copy.deepcopy(record)
    forced_miscredit["credit_identity"]["forced_candidate_in_free_counts"] = True
    with pytest.raises(ValueError, match="credit identity differs"):
        endpoint.validate_conditional_artifact(forced_miscredit, job)


def test_runtime_arm_overrides_packet_arm_and_rejects_swapped_record():
    packet = endpoint.prepare_packet_payload()
    job = endpoint.runtime_conditional_jobs(packet, arm="B", rank=0)[0]
    assert job.arm == "B"
    parsed = {
        "pred": [{"generated_order": i, "bbox": [i, i, i + 1, i + 1]}
                 for i in range(job.prefix_row_count)],
        "dropped_predictions": [],
    }
    record = endpoint.conditional_artifact(
        job,
        free_ids=list(range(100, 100 + job.budget)),
        stop_reason="length",
        parsed=parsed,
    )
    endpoint.validate_conditional_artifact(record, job)
    assert record["arm"] == "B"
    swapped = copy.deepcopy(record)
    swapped["arm"] = "A"
    with pytest.raises(ValueError, match="runtime arm differs"):
        endpoint.validate_conditional_artifact(swapped, job)


def test_exact_merged_384_identity_rejects_missing_row():
    shards = [[f"id-{rank}-{i}" for i in range(48)] for rank in range(8)]
    records = [{"example_id": eid, "shard": rank} for rank, ids in enumerate(shards) for eid in ids]
    endpoint.validate_exact_natural_population(records, shards)
    with pytest.raises(ValueError, match="exact384"):
        endpoint.validate_exact_natural_population(records[:-1], shards)


def test_balanced_conditional_plan_and_bounds():
    cases = [
        {"case_id": "351017", "candidate_id": "351017-c01", "h_ids": list(range(19)), "c_ids": list(range(11))},
        {"case_id": "417044", "candidate_id": "417044-c01", "h_ids": list(range(79)), "c_ids": list(range(10))},
        {"case_id": "477415", "candidate_id": "477415-c02", "h_ids": list(range(63)), "c_ids": list(range(9))},
    ]
    jobs = endpoint.build_conditional_jobs(cases)
    assert [job.rank for job in jobs] == list(range(6))
    assert [job.budget for job in jobs] == [3065, 3054, 3005, 2995, 3021, 3012]
    bounds = endpoint.rank_bounds(jobs)
    assert [row["max_calls"] for row in bounds] == [49] * 6 + [48, 48]
    assert [row["max_generated_tokens"] for row in bounds] == [
        151097, 151086, 151037, 151027, 151053, 151044, 148032, 148032,
    ]


def test_complete_terminal_over_cuda_contract_is_rejected():
    bound = endpoint.rank_bounds([])[7]
    terminal = {
        "model_loads": 1,
        "natural_continuations": 48,
        "conditional_continuations": 0,
        "continuations": 48,
        "image_forwards": 48,
        "new_tokens": 148032,
        "model_forwards": 148032,
        "elapsed_seconds": 1200,
        "peak_cuda_allocated_bytes": 13 * 1024**3,
        "peak_cuda_reserved_bytes": 13 * 1024**3,
        "peak_rss_bytes": 8 * 1024**3,
    }
    with pytest.raises(ValueError, match="CUDA bound failure"):
        endpoint.validate_terminal_resources(terminal, bound)
    terminal["peak_cuda_allocated_bytes"] = 12 * 1024**3
    terminal["peak_cuda_reserved_bytes"] = 12 * 1024**3
    endpoint.validate_terminal_resources(terminal, bound)


def test_pairing_preserves_directional_owner_gains_and_losses():
    a = [{"example_id": "x", "score": {"50": {"owners": ["1", "2"]}, "60": {"owners": []}, "80": {"owners": ["9"]}}}]
    b = [{"example_id": "x", "score": {"50": {"owners": ["2", "3"]}, "60": {"owners": []}, "80": {"owners": []}}}]
    paired = endpoint.pair_owner_changes(a, b)
    assert paired[0]["B_vs_A"]["50"] == {"gained": ["3"], "lost": ["1"], "retained": ["2"]}
    assert paired[0]["B_vs_A"]["80"] == {"gained": [], "lost": ["9"], "retained": []}


def test_panel_partition_keeps_train256_and_dev128_disjoint():
    rows = [{"example_id": f"t-{i}", "split": "remaining192"} for i in range(256)]
    rows += [{"example_id": f"d-{i}", "split": "dev128"} for i in range(128)]
    train = endpoint.panel_ids("train256", rows)
    dev = endpoint.panel_ids("dev128", rows)
    assert len(train) == 256 and len(dev) == 128 and train.isdisjoint(dev)
    assert endpoint.panel_ids("union384", rows) == train | dev


def test_real_packet_preparation_binds_frozen_inputs():
    packet = endpoint.prepare_packet_payload()
    assert packet["schema"] == "positive_branch_vs_repeat_event.endpoint_packet.v1"
    assert len(packet["eval_records"]) == 384
    assert list(map(len, packet["eval_shards"])) == [48] * 8
    assert packet["sources"]["geometric_inputs"]["sha256"] == endpoint.GEOM_SHA256
    assert packet["sources"]["input_manifest"]["sha256"] == endpoint.INPUT_MANIFEST_SHA256
    assert [job["rank"] for job in packet["conditional_jobs"]] == list(range(6))
