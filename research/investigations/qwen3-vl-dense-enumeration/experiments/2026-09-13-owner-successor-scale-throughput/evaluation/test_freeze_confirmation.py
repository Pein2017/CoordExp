import json
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("freeze_confirmation.py")
SPEC = importlib.util.spec_from_file_location("freeze_confirmation", MODULE_PATH)
assert SPEC and SPEC.loader
_unit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_unit)


def test_frozen_confirmation_packet_replays_without_model_calls():
    packet_path = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput/evaluation/confirmation-selection.json"
    )
    result = _unit.replay(packet_path)
    assert result == {
        "status": "passed",
        "source_images": 4952,
        "excluded_source_images": 1563,
        "eligible_source_images": 3389,
        "panel_images": 256,
        "review_images": 32,
        "physical_geometry_records": 256,
        "model_calls": 0,
    }


def test_packet_has_complete_identity_boundary_and_no_disjointness_claim():
    packet_path = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput/evaluation/confirmation-selection.json"
    )
    packet = json.loads(packet_path.read_text())
    assert packet["status"] == "frozen_cpu_no_model_calls"
    assert len(packet["image_ids"]) == 256
    assert len(packet["blind_review_ids"]) == 32
    assert set(packet["blind_review_ids"]) <= set(packet["image_ids"])
    assert len(packet["excluded_image_ids"]) == len(set(packet["excluded_image_ids"]))
    assert set(packet["image_ids"]).isdisjoint(packet["excluded_image_ids"])
    assert len(packet["records"]) == 256
    assert "pretraining/SFT-disjointness" in packet["provenance_boundary"]
    assert "model calls" in packet["claim_boundary"]
