from probes.owner_successor_scale import evaluation as e


def test_n16_anchor_packet_binds_immutable_panel_and_no_rerun_slice():
    packet = e.read(e.ROOT / "packet.json")
    baseline = e.read(e.ROOT / "baseline-packet.json")
    assert packet["schema"] == "native_owner_scale_state.evaluation.packet.v1"
    assert packet["status"] == "cpu_prepared_n16_anchor_no_model_calls"
    assert len(packet["records"]) == 256
    assert packet["n16_anchor"]["adapter_fingerprint"] == e.N16_ADAPTER_FINGERPRINT
    assert packet["n16_anchor"]["adapter_weights_sha256"] == e.N16_ADAPTER_SHA256
    assert baseline["arm"] == "N16-anchor"
    assert baseline["physical_gpus"] == [6, 7]
    assert baseline["phases"]["slice"]["image_ids"] == e.read(e.CONFIRMATION)["image_ids"][:2]


def test_phase_union_is_exact_and_generation_contract_is_frozen():
    packet = e.read(e.ROOT / "baseline-packet.json")
    selection = e.read(e.CONFIRMATION)
    left = packet["phases"]["slice"]["image_ids"]
    right = packet["phases"]["full"]["image_ids"]
    assert len(left) == 2 and len(right) == 254
    assert len(set(left + right)) == 256
    assert set(left + right) == set(selection["image_ids"])
    assert packet["generation"] == {
        "dtype": "fp32",
        "attention": "sdpa",
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 3084,
        "eos_token_id": 151645,
        "natural_prefix_ids": [],
        "forced_credit": False,
    }


def test_cold_anchor_result_is_combined_and_raw_class_inventory_is_separate():
    result = e.read(e.ROOT / "anchor-result.json")
    assert result["status"] == "cold_verified_n16_anchor_256"
    assert result["denominators"] == {"fresh256": 256, "slice": 2, "remaining254": 254}
    assert result["burden"]["eos"] == 256
    assert result["burden"]["caps"] == 0
    assert result["costs"]["image_forwards"] == 256
    assert result["raw_class_inventory"]["raw_prediction_count"] == 1766
    assert "not silently relabeled or filtered" in result["raw_class_inventory"]["policy"]
