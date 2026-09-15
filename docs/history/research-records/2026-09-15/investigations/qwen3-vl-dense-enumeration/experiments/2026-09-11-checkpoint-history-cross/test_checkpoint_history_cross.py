from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


prepare = load("prepare")
runner = load("run_cross")


@pytest.fixture(scope="module")
def packet():
    return prepare.build_packet()


def test_real_frozen_cross_has_exact_offsets_and_finite_bounds(packet):
    assert [row["image_id"] for row in packet["images"]] == [39654, 351017, 417044, 477415]
    assert packet["limits"] == {
        "workers": 8, "cells": 16, "calls": 16, "model_loads": 8,
        "image_forwards": 16, "global_generated_tokens_max": 49082,
        "worker_generated_tokens_max": 6150, "seconds_per_worker": 1500,
        "cuda_allocated_bytes_max": 12 * 1024**3,
        "cuda_reserved_bytes_max": 12 * 1024**3,
        "rss_bytes_max": 16 * 1024**3, "training_steps": 0,
    }
    assert len(packet["workers"]) == 8
    assert sum(worker["max_generated_tokens"] for worker in packet["workers"]) == 49082
    for image in packet["images"]:
        stable = image["histories"]["stable50"]
        positive = image["histories"]["positive32"]
        assert prepare.first_difference(stable["full_action_ids"], positive["full_action_ids"]) == image["offsets"]["first_difference"]
        assert stable["h_ids"] == positive["h_ids"]
        assert stable["history_ids"][:image["offsets"]["h_end"]] == stable["h_ids"]
        assert positive["history_ids"][:image["offsets"]["h_end"]] == positive["h_ids"]
        assert prepare.split_complete_rows(stable["branch_row_ids"]) == [stable["branch_row_ids"]]
        assert prepare.split_complete_rows(positive["branch_row_ids"]) == [positive["branch_row_ids"]]


def test_real_adapter_override_closes_raw_source_config_trap(packet):
    raw = packet["raw_config_adapter_path_is_source_trap"]
    assert raw not in {packet["checkpoints"][name]["adapter"]["root"] for name in prepare.CHECKPOINTS}
    effective = packet["frontend_cpu_preflight"]["effective_checkpoints"]
    assert effective["stable50"]["adapter_path"] == packet["checkpoints"]["stable50"]["adapter"]["root"]
    assert effective["positive32"]["adapter_path"] == packet["checkpoints"]["positive32"]["adapter"]["root"]
    assert {value["dtype"] for value in effective.values()} == {"fp32"}
    assert {value["attention"] for value in effective.values()} == {"sdpa"}
    assert all("signature" in item and item["signature"] for item in packet["frontend_cpu_preflight"]["callables"].values())


def test_saved_on_diagonal_suffixes_traverse_actual_native_parser(packet):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(packet["model"]["base_model_path"] + "/tokenizer.json")
    for image in packet["images"]:
        for checkpoint in prepare.CHECKPOINTS:
            history = image["histories"][checkpoint]
            analysis = runner.analyze_cell(
                packet, image, checkpoint, history["expected_on_diagonal_free_ids"],
                history["full_stop_reason"], tokenizer)
            assert analysis["action_ids"] == history["full_action_ids"]
            assert analysis["prefix_ids"] == history["history_ids"]
            assert analysis["forced_free_accounting"]["forced_history_in_free_counts"] is False
            assert len(analysis["parser_partition"]["observed_prefix_rows"]) == history["history_complete_row_count"]
            assert analysis["free_token_count"] <= analysis["free_budget"]
            assert analysis["eos"] != analysis["cap"]


def test_packet_is_cpu_candidate_not_launch_grant(packet):
    assert packet["status"] == "candidate_cpu_prepared_no_gpu_grant"
    assert packet["claim_boundary"].startswith("Exposed four-image checkpoint/history cross")
    serialized = json.dumps(packet)
    assert "positive_owner" not in serialized
    assert "guessed_bbox" not in serialized
    assert all(worker["on_diagonal_history"] == worker["checkpoint"] for worker in packet["workers"])
    assert all(worker["cross_history"] != worker["checkpoint"] for worker in packet["workers"])


def test_suffix_gate_rejects_early_eos_and_underfilled_length():
    with pytest.raises(ValueError, match="token corruption"):
        runner.checked_suffix([1, runner.EOS, 2], budget=3, stop="length")
    with pytest.raises(ValueError, match="stop/budget"):
        runner.checked_suffix([1, 2], budget=3, stop="length")
