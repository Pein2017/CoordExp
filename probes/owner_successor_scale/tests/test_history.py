from __future__ import annotations

import pytest

from probes.owner_successor_scale import history as h


def literal(label: str, token: int, description: str, count: int = 9) -> dict:
    ids = [h.ROW_START, token, h.ROW_END]
    # The production grammar has more fields; tests only exercise packet
    # invariants, so represent a complete row with a harmless fixed shape.
    ids = [h.ROW_START, token, 151647, 151648, 152000, 152001, 152002, 152003, h.ROW_END]
    return {
        "token_ids": ids,
        "token_count": count,
        "description": description,
        "bbox": [0, 0, 10, 10],
        "coord_bins": [0, 0, 10, 10],
        "text": label,
    }


def components() -> dict[str, dict]:
    return {
        "H": literal("h", 1, "person"),
        "n": literal("n", 2, "frisbee"),
        "S": literal("s", 3, "person"),
        "a": literal("a", 4, "cup"),
        "b": literal("b", 5, "cup"),
    }


def test_history_family_has_four_matched_orders_and_common_suffix():
    rows = components()
    family = {name: h._history_record(image={"image_id": 1, "example_id": "x"}, components=rows, label=name) for name in ("H_a_n_S", "H_b_n_S", "H_n_a_S", "H_n_b_S")}
    h._validate_history_family(histories=family, components=rows)
    assert all(item["component_order"][-1] == "S" for item in family.values())
    assert len({item["token_count"] for item in family.values()}) == 1


def test_history_family_rejects_unmatched_candidate_token_count():
    rows = components()
    rows["b"] = literal("b", 5, "cup", count=10)
    with pytest.raises(ValueError, match="candidate token lengths"):
        h._history_record(image={"image_id": 1, "example_id": "x"}, components=rows, label="H_a_n_S")


def test_target_interaction_does_not_credit_supplied_prefix_rows():
    components_ = components()
    prefix = {"pred": [{"description": "cup", "bbox": [0, 0, 10, 10]}]}
    free = {"pred": []}
    observed = h._target_interaction(prefix_parsed=prefix, free_parsed=free, components=components_)
    assert observed["a"]["prefix_forced_ordinals"] == [0]
    assert observed["a"]["free_present"] is False
    assert observed["a"]["free_first_ordinal"] is None
    assert observed["a"]["forced_rows_earn_no_credit"] is True


def test_strict_repeat_uses_native_pixel_iou_and_strict_greater_than_point95():
    parsed = {"pred": [{"bbox": [0, 0, 100, 100]}, {"bbox": [0, 0, 95, 100]}, {"bbox": [0, 0, 94, 100]}]}
    # The exact .95 row is not strict; the next row is.
    assert h._strict_repeat_first(parsed) == 2


def test_job_bound_is_eight_per_arm_for_two_images():
    packet = {
        "cases": [
            {"image_id": 210457, "example_id": "a", "components": {"a": {"token_ids": [1]}, "b": {"token_ids": [2]}}, "histories": {name: {"token_ids": [3]} for name in ("H_a_n_S", "H_b_n_S", "H_n_a_S", "H_n_b_S")}},
            {"image_id": 219546, "example_id": "b", "components": {"a": {"token_ids": [1]}, "b": {"token_ids": [2]}}, "histories": {name: {"token_ids": [3]} for name in ("H_a_n_S", "H_b_n_S", "H_n_a_S", "H_n_b_S")}},
        ]
    }
    assert len(h._history_jobs(packet, 0)) == 4
    assert len(h._history_jobs(packet, 1)) == 4


def test_arm_launch_slots_keep_all_eight_histories_per_arm():
    packet = {
        "cases": [
            {"image_id": image_id, "example_id": str(image_id), "components": {"a": {"token_ids": [1]}, "b": {"token_ids": [2]}}, "histories": {name: {"token_ids": [3]} for name in ("H_a_n_S", "H_b_n_S", "H_n_a_S", "H_n_b_S")}}
            for image_id in (210457, 219546)
        ]
    }
    specs = h._arm_launch_specs()
    assert [(item["arm"], item["physical_gpu"], item["shard"]) for item in specs] == [
        ("Stable50", 6, 0),
        ("N16", 7, 1),
    ]
    assert all(item["physical_gpu"] == h.GPUS[item["shard"]] for item in specs)
    assert all(len(h._all_history_jobs(packet)) == 8 for _ in specs)


def test_runtime_config_only_changes_image_root_path():
    panel = {
        "data": {"input_jsonl": "/wrong/val.coord.jsonl"},
        "template": {"assistant_format": "object_box_closed"},
        "backend": {"type": "hf"},
    }
    runtime = h._runtime_config(panel)
    assert runtime["data"]["input_jsonl"] == str(h.RUNTIME_INPUT_JSONL)
    assert runtime["template"] == panel["template"]
    assert runtime["backend"] == panel["backend"]
