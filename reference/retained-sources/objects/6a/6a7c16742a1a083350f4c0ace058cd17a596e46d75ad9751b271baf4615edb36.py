from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).parent
PACKET = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-14-label-vs-compilation/diagnostic/execution-packet-v2.json"
)
SPEC = importlib.util.spec_from_file_location("label_vs_compilation_runner", HERE / "runner.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_sealed_packet_and_mutated_h_digest_fail_closed() -> None:
    packet = runner.validate_packet(runner._read(PACKET))
    assert len(packet["cells"]) == 8
    assert packet["bounds"]["max_new_tokens"] == 24260
    changed = copy.deepcopy(packet)
    changed["cells"][0]["h"]["token_ids"][0] += 1
    with pytest.raises(ValueError, match="h digest"):
        runner.validate_packet(changed)


def _cell() -> dict:
    c = [10, 11]
    w = [12, 13]
    return {
        "c": {
            "token_ids": c,
            "literal_target": {"description": "donut", "bbox_xyxy_pixels": [0, 0, 10, 10]},
        },
        "w": {
            "token_ids": w,
            "literal_target": {"description": "donut", "bbox_xyxy_pixels": [20, 20, 30, 30]},
        },
        "annotation_owner_obligations": [{"owner_id": "known"}],
    }


def _ledger() -> dict:
    return {
        "free_parsed": {
            "pred": [
                {"description": "donut", "bbox": [0, 0, 10, 10]},
                {"description": "donut", "bbox": [20, 20, 30, 30]},
            ]
        },
        "free_score": {"50": {"owners": ["known", "new"]}},
    }


def test_literal_parity_geometry_and_annotation_owners_are_separate() -> None:
    assessment = runner._assessment(_cell(), [10, 11, 12, 13], _ledger())
    assert assessment["literal_c_prefix_parity"] is True
    assert assessment["scientific_interpretation"] == "eligible_after_free_c_parity"
    assert assessment["c_geometry"]["same_class_iou_gt_0_5"] is True
    assert assessment["literal_w_occurrences"] == [2]
    assert assessment["annotation_owner_obligations"]["retained"] == ["known"]
    assert assessment["annotation_owner_obligations"]["gained_beyond_obligations"] == ["new"]


def test_nonliteral_geometric_c_stops_scientific_interpretation() -> None:
    assessment = runner._assessment(_cell(), [99, 12, 13], _ledger())
    assert assessment["literal_c_prefix_parity"] is False
    assert assessment["c_geometry"]["same_class_iou_gt_0_5"] is True
    assert assessment["scientific_interpretation"].startswith("STOP_cell_free_c")
