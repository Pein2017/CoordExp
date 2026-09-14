import copy
import json

import pytest

from probes.owner_successor_scale import paired_evaluation as e


def test_real_preparation_reuses_exact_n16_anchors_and_all_panels():
    packet = e.read(e.ROOT / "packet.json")
    readiness = e.read(e.ROOT / "readiness.json")
    old_completion = e.read(e.OLD_COMPLETION)
    e._validate_prepared(packet, allow_bound=False)
    assert packet["status"] == "trained_endpoints_pending"
    assert packet["denominators"] == {
        "old_exposed640": 640, "confirmation256": 256,
        "combined896": 896, "source_blind32": 32,
    }
    assert packet["anchor"]["old640_rows"] == old_completion["rows"]["scaled_terminal"]
    assert packet["anchor"]["old640_rows"] != old_completion["rows"]["Stable50"]
    assert packet["anchor"]["reuse_only_no_regeneration"] is True
    assert packet["baseline_cpu_consumer"]["images"] == 896
    assert all(values == {"gained": 0, "lost": 0, "retained": sum(
        len(row[threshold]["owners"]) for row in [r["score"] for r in e.read_jsonl(packet["anchor"]["rows"]["path"])]
    )} for threshold, values in packet["baseline_cpu_consumer"]["self_owner_changes"].items())
    assert readiness["status"] == "ready_waiting_for_two_full256_cold_receipts"
    assert "one-update smoke is ineligible" in readiness["missing_boundary"]


def test_wrong_anchor_and_row_omission_or_duplication_fail_closed():
    packet = e.read(e.ROOT / "packet.json")
    rows = e.read_jsonl(packet["anchor"]["rows"]["path"])
    wrong = copy.deepcopy(rows[0])
    wrong["adapter_fingerprint"] = "wrong-anchor"
    with pytest.raises(ValueError, match="wrong-anchor adapter"):
        e._exact_rows([wrong], packet["records"][:1], arm=wrong["arm"],
                      adapter_fingerprint=e.N16_ADAPTER_FINGERPRINT)
    with pytest.raises(ValueError, match="row omission"):
        e._exact_rows(rows[:1], packet["records"][:2], arm=rows[0]["arm"],
                      adapter_fingerprint=e.N16_ADAPTER_FINGERPRINT)
    with pytest.raises(ValueError, match="row duplication"):
        e._exact_rows([rows[0], rows[0]], packet["records"][:2], arm=rows[0]["arm"],
                      adapter_fingerprint=e.N16_ADAPTER_FINGERPRINT)


def test_prompt_and_budget_mismatch_have_teeth():
    generation = copy.deepcopy(e.read(e.ROOT / "packet.json")["generation"])
    generation["max_new_tokens"] = 512
    with pytest.raises(ValueError, match="prompt/budget"):
        e._generation_contract(generation)
    packet = e.read(e.ROOT / "packet.json")
    rows = e.read_jsonl(packet["anchor"]["rows"]["path"])
    record = copy.deepcopy(packet["records"][0])
    record["prompt_token_ids"] = [*record["prompt_token_ids"], 1]
    with pytest.raises(ValueError, match="prompt mismatch"):
        e._exact_rows([rows[0]], [record], arm=rows[0]["arm"],
                      adapter_fingerprint=e.N16_ADAPTER_FINGERPRINT)


def test_training_producer_binding_shape_and_mismatches_fail_closed(tmp_path):
    packet = e.read(e.ROOT / "packet.json")
    cold = tmp_path / "cold.json"
    cold.write_text("{}")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "schema": "owner_successor_scale.training.receipt.v1",
        "status": "technically_completed_cold_pending", "arm": "B",
        "updates": 256, "world_size": 8, "input": packet["training_input"],
    }))
    with pytest.raises(ValueError, match="trained-receipt mismatch"):
        e._verify_endpoint(receipt, cold, arm="A", training_input=packet["training_input"])
    value = json.loads(receipt.read_text())
    value["arm"] = "A"
    wrong_input = tmp_path / "wrong-input.json"
    wrong_input.write_text("{}")
    value["input"] = e._training_binding(wrong_input)
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="trained-receipt input mismatch"):
        e._verify_endpoint(receipt, cold, arm="A", training_input=packet["training_input"])

    bad_hash = e._training_binding(wrong_input)
    bad_hash["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="training binding changed"):
        e._verify_training_binding(bad_hash, "bad hash")

    smoke_root = e.UNIT_ROOT / "training/integrated-smoke-B-v1"
    smoke_receipt_path = smoke_root / "receipt.json"
    smoke_cold_path = smoke_root / "cold-check.json"
    smoke_receipt, smoke_cold = e.read(smoke_receipt_path), e.read(smoke_cold_path)
    assert set(smoke_receipt["input"]) == {"path", "sha256"}
    assert e._verify_training_binding(smoke_receipt["input"], "actual smoke input")
    for name in ("terminals", "update_records", "refresh_records", "final_protection_records"):
        assert all(e._verify_training_binding(item, f"actual smoke {name}")
                   for item in smoke_receipt[name])
    assert e._verify_training_binding(smoke_cold["training_receipt"], "actual smoke cold receipt")
    with pytest.raises(ValueError, match="not a full256 endpoint"):
        e._verify_endpoint(smoke_receipt_path, smoke_cold_path, arm="B",
                           training_input=packet["training_input"])


def test_blind_freeze_is_single_image_source_only_and_unlabeled():
    freeze = e.read(e.ROOT / "blind-review-freeze.json")
    selection = e.read(e.NEW_SELECTION)
    assert freeze["image_ids"] == selection["blind_review_ids"]
    assert len(freeze["items"]) == 32
    assert len({item["image_id"] for item in freeze["items"]}) == 32
    assert all("prediction" not in json.dumps(item).lower() and "physical_labels" not in item
               for item in freeze["items"])
    assert "no prediction" in freeze["boundary"]
