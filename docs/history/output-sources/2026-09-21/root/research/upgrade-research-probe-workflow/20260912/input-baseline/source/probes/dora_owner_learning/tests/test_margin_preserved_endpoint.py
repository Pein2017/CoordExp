from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from probes.dora_owner_learning import margin_preserved_endpoint as endpoint


def test_real_retained_baselines_have_exact_identity_order_and_diagnostics() -> None:
    old_packet = endpoint.load_old_packet()
    summary = endpoint.validate_retained_baselines(old_packet)

    assert summary == {
        "natural_rows": 384,
        "arm": "A",
        "split_counts": {"reference56": 56, "train256": 256, "dev128": 128},
        "positive_A_tp50": {"351017": 12, "417044": 11, "477415": 16},
        "diagnostic_39654": {
            "split": "dev128",
            "tp50": 0,
            "strict_repeats": 194,
            "parser_drops": 144,
            "cap": 1,
        },
    }


def test_retained_baseline_rejects_mislabel_order_and_geometry() -> None:
    packet = endpoint.load_old_packet()
    rows = endpoint.load_json(endpoint.RETAINED_A_CONSUMER)

    wrong_arm = copy.deepcopy(rows)
    wrong_arm[0]["arm"] = "C"
    with pytest.raises(ValueError, match="retained A arm"):
        endpoint.validate_natural_rows(wrong_arm, packet, arm="A", retained=True)

    wrong_order = copy.deepcopy(rows)
    wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
    with pytest.raises(ValueError, match="ordered shard"):
        endpoint.validate_natural_rows(wrong_order, packet, arm="A", retained=True)

    wrong_geometry = copy.deepcopy(rows)
    wrong_geometry[0]["observed_image_grid_thw"][0] += 1
    with pytest.raises(ValueError, match="prompt/media/grid"):
        endpoint.validate_natural_rows(wrong_geometry, packet, arm="A", retained=True)

    wrong_score = copy.deepcopy(rows)
    wrong_score[0]["score"]["50"]["tp"] += 1
    with pytest.raises(ValueError, match="score"):
        endpoint.validate_natural_rows(wrong_score, packet, arm="A", retained=True)


def test_c_label_is_not_a_caller_override() -> None:
    packet = endpoint.load_old_packet()
    rows = copy.deepcopy(endpoint.load_json(endpoint.RETAINED_A_CONSUMER))
    for row in rows:
        row["arm"] = "C"
        row["schema"] = endpoint.NATURAL_SCHEMA
    endpoint.validate_natural_rows(rows, packet, arm="C", retained=False)

    rows[0]["arm"] = "A"
    with pytest.raises(ValueError, match="C arm"):
        endpoint.validate_natural_rows(rows, packet, arm="C", retained=False)
    with pytest.raises(ValueError, match="only C"):
        endpoint.validate_c_training_metadata({"arm": "A"}, {})


def test_rank_and_global_resource_contract_is_exact() -> None:
    packet = endpoint.packet_base(endpoint.load_old_packet())
    bounds = packet["resource_bounds"]

    assert len(packet["eval_shards"]) == endpoint.WORLD_SIZE == 8
    assert [len(x) for x in packet["eval_shards"]] == [48] * 8
    assert [x["max_calls"] for x in bounds["ranks"]] == [49] * 6 + [48, 48]
    assert sum(x["natural_calls"] + x["conditional_calls"] for x in bounds["ranks"]) == 390
    assert sum(x["max_model_loads"] for x in bounds["ranks"]) == 8
    assert bounds["global_max_generated_tokens"] == 120_000
    assert all(x["max_generated_tokens"] == x["max_model_forwards"] == 15_000 for x in bounds["ranks"])
    assert sum(x["max_generated_tokens"] for x in bounds["ranks"]) == 120_000
    assert all(x["max_worker_seconds"] == 1500 for x in bounds["ranks"])
    assert all(x["max_cuda_bytes"] == x["max_rss_bytes"] == 24 * 1024**3 for x in bounds["ranks"])


def test_live_forward_hook_withholds_the_next_forward_at_worker_cap() -> None:
    counters = {"model": endpoint.MAX_GENERATED_TOKENS_PER_WORKER - 1, "image": 0}
    module = torch.nn.Identity()
    handle = module.register_forward_pre_hook(endpoint.forward_budget_hook(counters))
    try:
        assert module(torch.tensor([1])).item() == 1
        assert counters["model"] == endpoint.MAX_GENERATED_TOKENS_PER_WORKER
        with pytest.raises(endpoint.ForwardBudgetExceeded, match="before next forward"):
            module(torch.tensor([2]))
        assert counters["model"] == endpoint.MAX_GENERATED_TOKENS_PER_WORKER
    finally:
        handle.remove()


def test_package_import_verifies_real_smoke_but_endpoint_rejects_it() -> None:
    smoke_root = endpoint.OUTPUT_ROOT / "smoke-weight10"
    receipt = endpoint.verify_training_receipt(smoke_root / "receipt.json")
    cold = endpoint.load_json(smoke_root / "cold-check.json")
    assert receipt["arm"] == "C" and receipt["mode"] == "smoke"
    with pytest.raises(ValueError, match="actual C32 full receipt"):
        endpoint.validate_c_training_metadata(receipt, cold)


def test_owner_change_and_burden_reports_joint_thresholds() -> None:
    before = {"50": {"owners": ["a", "b"]}, "60": {"owners": ["a"]}, "80": {"owners": []}}
    after = {"50": {"owners": ["b", "c"]}, "60": {"owners": []}, "80": {"owners": ["d"]}}
    counts = endpoint.owner_change_counts([before], [after])
    assert counts == {
        "50": {"gained": 1, "lost": 1, "retained": 1},
        "60": {"gained": 0, "lost": 1, "retained": 0},
        "80": {"gained": 1, "lost": 0, "retained": 0},
    }

    a = endpoint.load_json(endpoint.RETAINED_A_CONSUMER)
    burden = endpoint.burden(a)
    assert burden["row_starts"] == burden["parsed_predictions"] + burden["parser_drops"]
    assert set(burden["parser_drop_reasons"]) == {"geometry_invalid", "malformed_object_span"}
    assert burden["eos"] + burden["caps"] == 384


def test_conditional6_keeps_forced_rows_out_of_free_credit() -> None:
    packet = endpoint.packet_base(endpoint.load_old_packet())
    rows = copy.deepcopy(endpoint.load_json(endpoint.RETAINED_A_CONDITIONAL))
    for row in rows:
        row["schema"] = endpoint.CONDITIONAL_SCHEMA
        row["arm"] = "C"
    endpoint.validate_conditional_rows(rows, packet)
    assert [row["kind"] for row in rows] == [
        "h_only", "h_plus_c", "h_only", "h_plus_c", "h_only", "h_plus_c"
    ]
    assert [row["credit_identity"]["forced_candidate_row_count"] for row in rows] == [0, 1, 0, 1, 0, 1]
    assert all(not row["credit_identity"]["forced_candidate_in_free_counts"] for row in rows)

    rows[1]["credit_identity"]["forced_candidate_in_free_counts"] = True
    with pytest.raises(ValueError, match="credit identity"):
        endpoint.validate_conditional_rows(rows, packet)


def test_pending_is_honest_and_final_packet_requires_actual_c(tmp_path: Path) -> None:
    pending = endpoint.pending_payload()
    assert pending["status"] == "pending_actual_C32_receipt_and_cold_check"
    assert pending["verified"]["retained_A_consumer"]["sha256"] == endpoint.RETAINED_A_CONSUMER_SHA256
    assert pending["unresolved"] == ["actual_C32_training_receipt", "actual_C32_cold_check"]
    assert "candidate_adapter" not in pending

    out = tmp_path / "packet.json"
    with pytest.raises((ValueError, FileNotFoundError)):
        endpoint.prepare_final_packet(
            tmp_path / "absent-receipt.json", tmp_path / "absent-cold.json", out
        )
    assert not out.exists()


def test_finalizer_rejects_incomplete_outer_exit_set(tmp_path: Path) -> None:
    launch = tmp_path / "launcher"
    launch.mkdir()
    (launch / "shard-0-outer-exit.json").write_text(
        json.dumps({"schema": endpoint.OUTER_EXIT_SCHEMA, "shard": 0, "exit_code": 0})
    )
    with pytest.raises(ValueError, match="outer exits"):
        endpoint.validate_outer_exits(launch)
