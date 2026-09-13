import copy
import json

import pytest

from probes.native_owner_scale import evaluation as e


def test_candidate_materialization_uses_bound_image_without_mutating_source(tmp_path):
    from src.data.examples import raw_example_from_jsonl_row

    image = tmp_path / "bound.jpg"
    image.write_bytes(b"bound image bytes")
    case = {
        "image_path": str(image),
        "image_plan": {"image_content_sha256": e.file_hash(image)},
        "input_record": {"images": ["../../bound.jpg"], "file_name": "bound.jpg", "image_id": 1, "metadata": {"source": "coco2017", "split": "train"}, "objects": [{"bbox_2d": ["<|coord_0|>", "<|coord_0|>", "<|coord_999|>", "<|coord_999|>"], "desc": "person", "category_id": 1, "category_name": "person", "coco_ann_id": 1}], "width": 32, "height": 32},
    }
    original = copy.deepcopy(case)
    kwargs = {"jsonl_path": tmp_path / "fresh.jsonl", "row_number": 1, "raw_line": "fixture"}
    with pytest.raises(Exception, match="data.image_missing"):
        raw_example_from_jsonl_row(case["input_record"], **kwargs)
    config = {"data": {"input_jsonl": str(tmp_path / "fresh.jsonl")}}
    materialized = e._candidate_materialized_case(case, config)
    raw = raw_example_from_jsonl_row(materialized["input_record"], **kwargs)
    assert str(raw.image.path) == str(image)
    assert case == original
    assert materialized["image_plan"] == case["image_plan"]
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bound image bytes changed"):
        e._candidate_materialized_case(case, config)


def _selection():
    panel = list(range(100, 356))
    blind = panel[:32]
    excluded = list(range(1, 100))
    return {
        "image_ids": panel,
        "image_ids_sha256": e.digest(panel),
        "blind_review_ids": blind,
        "blind_review_ids_sha256": e.digest(blind),
        "excluded_image_ids": excluded,
    }


def test_identity_projection_ignores_metrics_and_selects_by_salted_identity():
    evidence = {
        "image_ids": [1, 2],
        "nested": {"image_id": 3, "tp": 9000},
        "visual": "/x/val2017/000000000004.jpg",
        "example_id": "coco2017_train_000000000005",
        "object_id": 6,
        "token_ids": [7, 8],
    }
    assert e.image_ids(evidence) == {1, 2, 3, 4, 5}
    left = e.select_ids(range(1, 40), e.image_ids(evidence), 10, "test:")
    right = e.select_ids(reversed(range(1, 40)), {5, 4, 3, 2, 1}, 10, "test:")
    assert left == right
    assert not set(left) & {1, 2, 3, 4, 5}
    with pytest.raises(ValueError, match="insufficient"):
        e.select_ids([1, 2], {1}, 2, "test:")


def test_selection_checks_panel_blind_subset_and_false_overlap():
    selection = _selection()
    e.validate_selection(selection, range(1, 1000))

    overlap = copy.deepcopy(selection)
    overlap["excluded_image_ids"] = [*selection["excluded_image_ids"], selection["image_ids"][0]]
    with pytest.raises(ValueError, match="fresh256 overlaps"):
        e.validate_selection(overlap, range(1, 1000))

    blind_overlap = copy.deepcopy(selection)
    blind_overlap["excluded_image_ids"] = [*selection["excluded_image_ids"], selection["blind_review_ids"][0]]
    with pytest.raises(ValueError, match="fresh256 overlaps"):
        e.validate_selection(blind_overlap, range(1, 1000))

    not_subset = copy.deepcopy(selection)
    not_subset["blind_review_ids"] = [*selection["blind_review_ids"][:-1], 999]
    not_subset["blind_review_ids_sha256"] = e.digest(not_subset["blind_review_ids"])
    with pytest.raises(ValueError, match="blind32 must"):
        e.validate_selection(not_subset, range(1, 1000))


def test_declared_lane_manifest_requires_acknowledged_status_and_projects_ids(tmp_path):
    path = tmp_path / "lane.json"
    path.write_text(json.dumps({"status": "acknowledged", "owner": "B", "scenes": [{"image_id": 12}]}))
    manifest = e._declared_manifest(path, "B")
    assert manifest["image_ids"] == [12]
    path.write_text(json.dumps({"status": "running", "image_ids": [12]}))
    with pytest.raises(ValueError, match="not acknowledged"):
        e._declared_manifest(path, "B")


def test_candidate_binding_requires_n16_and_native_numerics():
    base = {
        "label": "scaled_terminal",
        "status": "cold_verified",
        "admitted_packages": 16,
        "adapter": {"root": "/tmp/adapter", "fingerprint": "fp"},
        "max_new_tokens": 3084,
        "dtype": "fp32",
        "attention": "sdpa",
    }
    e.validate_candidate_binding(base)
    too_small = copy.deepcopy(base)
    too_small["admitted_packages"] = 15
    with pytest.raises(ValueError, match="below N16"):
        e.validate_candidate_binding(too_small)
    changed_cap = copy.deepcopy(base)
    changed_cap["max_new_tokens"] = 1024
    with pytest.raises(ValueError, match="cap changed"):
        e.validate_candidate_binding(changed_cap)


def test_bind_candidate_preserves_cpu_packet_and_writes_new_bound_packet(tmp_path):
    packet_path = tmp_path / "packet.json"
    candidate_path = tmp_path / "candidate.json"
    packet = {
        "status": "cpu_prepared_candidate_pending",
        "selection": {"path": "selection.json", "sha256": "selection"},
        "scaled_terminal": {"status": "pending_A_terminal_adapter"},
    }
    candidate = {
        "label": "scaled_terminal",
        "status": "cold_verified",
        "admitted_packages": 16,
        "adapter": {"root": "/tmp/adapter", "fingerprint": "fp"},
        "max_new_tokens": 3084,
        "dtype": "fp32",
        "attention": "sdpa",
    }
    packet_path.write_text(json.dumps(packet))
    candidate_path.write_text(json.dumps(candidate))
    bound = e.bind_candidate(packet_path=packet_path, candidate_path=candidate_path)
    assert json.loads(packet_path.read_text()) == packet
    assert bound["status"] == "ready_for_root_launch_gate"
    assert (tmp_path / "packet-bound.json").exists()


def test_paired_summary_keeps_gain_loss_and_rejects_reordering():
    def row(image_id, owners):
        n = len(owners)
        score = {
            threshold: {
                "tp": n,
                "fp": 0,
                "fn": 2 - n,
                "f1": 2 * n / (n + 2),
                "owners": owners,
            }
            for threshold in ("50", "60", "80")
        }
        return {"image_id": image_id, "example_id": str(image_id), "score": score}

    before = [row(1, ["a"]), row(2, ["c"])]
    after = [row(1, ["a", "b"]), row(2, [])]
    result = e.paired_summary(before, after)
    assert result["owner_counts"]["50"] == {"gained": 1, "lost": 1, "retained": 1}
    assert result["paired_image_bootstrap"]["intervals"]["50"]["delta_tp95"] == [-2, 2]
    with pytest.raises(ValueError, match="unpaired"):
        e.paired_summary(before, after[::-1])


def test_source_blind_queue_excludes_arm_labels_and_binds_literal_paths(tmp_path):
    packet = {
        "selection": {"path": str(tmp_path / "selection.json")},
        "records": [
            {
                "image_id": image_id,
                "example_id": f"coco2017_train_{image_id:012d}",
                "case": {"image_path": f"/data/images/{image_id}.jpg", "image_width": 10, "image_height": 10},
            }
            for image_id in range(7, 39)
        ],
    }
    selection = {"blind_review_ids": list(range(7, 39))}
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    def output_row(arm, image_id, description, bbox):
        return {
            "image_id": image_id,
            "parsed": {"pred": [{"description": description, "bbox": bbox, "bbox_format": "xyxy"}]},
        }
    result = e._proposal_queue(
        {
            "Stable50": [output_row("Stable50", image_id, "cat", [0, 0, 2, 2]) for image_id in range(7, 39)],
            "scaled_terminal": [output_row("scaled_terminal", image_id, "dog", [1, 1, 3, 3]) for image_id in range(7, 39)],
        },
        packet,
        tmp_path / "review",
    )
    queue = [json.loads(line) for line in (tmp_path / "review/blind-review-queue.jsonl").read_text().splitlines()]
    assert len(queue) == 32 and len(queue[0]["proposals"]) == 2
    serialized = json.dumps(queue)
    assert "Stable50" not in serialized and "scaled_terminal" not in serialized
    assert queue[0]["image_path"] == "/data/images/7.jpg"
    assert result["images"] == 32
    assert json.loads((tmp_path / "review/visualization-manifest.json").read_text())["items"][0]["literal_source_canvas_path"] == "/data/images/7.jpg"


def test_baseline_phase_contract_keeps_slice_out_of_remaining_panel(monkeypatch):
    packet = {
        "phases": {
            "slice": {"image_ids": [64523, 395633]},
            "full": {"image_ids": [10, 11, 12]},
        }
    }
    monkeypatch.setattr(e, "validate_baseline_packet", lambda _: None)
    assert e._baseline_phase(packet, "slice")["image_ids"] == [64523, 395633]
    assert not set(e._baseline_phase_ids(packet, "slice")) & set(e._baseline_phase_ids(packet, "full"))


def test_file_binding_rejects_changed_and_unknown_source(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("original\n")
    sealed = e.binding(source)
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="source changed"):
        e._verify_file_binding(sealed, "source")
    with pytest.raises(ValueError, match="source missing"):
        e._verify_file_binding({"path": str(tmp_path / "unknown.jsonl"), "sha256": "x", "size_bytes": 1}, "source")


def test_panel_union_rejects_overlap_or_missing_identity():
    slice_ids = [1, 2]
    remaining = list(range(3, 257))
    panel = list(range(1, 257))
    e._validate_panel_union(slice_ids, remaining, panel)
    with pytest.raises(ValueError, match="overlaps"):
        e._validate_panel_union(slice_ids, [2, *range(3, 256)], panel)
    with pytest.raises(ValueError, match="panel union"):
        e._validate_panel_union(slice_ids, [*range(3, 256), 257], panel)


def test_baseline_launch_records_two_physical_gpu_worker_commands(monkeypatch, tmp_path):
    packet_path = tmp_path / "baseline-packet.json"
    run_root = tmp_path / "slice"
    packet_path.write_text(
        json.dumps(
            {
                "physical_gpus": [4, 5],
                "phases": {"slice": {"image_ids": [64523, 395633], "run_root": str(run_root)}},
            }
        )
    )
    monkeypatch.setattr(e, "_baseline_phase", lambda packet, phase: packet["phases"][phase])
    calls = []

    class Process:
        def wait(self):
            return 0

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr(e.subprocess, "Popen", fake_popen)
    result = e.baseline_launch(packet_path=packet_path, phase="slice")
    assert result["status"] == "completed"
    assert len(calls) == 2
    assert [kwargs["env"]["CUDA_VISIBLE_DEVICES"] for _, kwargs in calls] == ["4", "5"]
    assert all("baseline-worker" in command for command, _ in calls)
    assert json.loads((run_root / "outer-exits.json").read_text()) == [
        {"shard": 0, "exit_code": 0},
        {"shard": 1, "exit_code": 0},
    ]


def _synthetic_panel_packet():
    records = [
        {"image_id": image_id, "example_id": f"example-{image_id}"}
        for image_id in range(640)
    ]
    strata = e._candidate_panel_strata(
        records,
        admitted_ids=range(11),
        reference_ids=range(11, 65),
        fresh_ids=range(384, 640),
    )
    return {
        "schema": "native_owner_scale_state.evaluation.candidate_panel.v2",
        "status": "candidate_adapter_pending",
        "records": records,
        "strata": strata,
        "generation": {"max_new_tokens": 3084, "natural_prefix_ids": [], "dtype": "fp32", "attention": "sdpa"},
    }


def test_lossless_stable_projection_keeps_tokens_stop_metrics_and_identity():
    parsed = {
        "raw_decode_text": "<|object_ref_start|>cat<|object_ref_end|><|im_end|>",
        "decode_stop_reason": "im_end",
        "pred": [{"description": "cat", "bbox": [0, 0, 2, 2], "bbox_format": "xyxy"}],
    }
    source = {
        "example_id": "coco2017_train_000000000001",
        "image_id": 1,
        "split": "reference56",
        "prompt_token_ids": [9, 8, 7],
        "stable_ids": [4, 5, 151645],
        "stable_parsed": parsed,
        "stable_score": {"50": {"tp": 1, "fp": 0, "fn": 0, "f1": 1.0, "owners": ["owner"]}},
        "stable_overlap_counts": {"80": 0, "90": 0, "95": 0},
        "case": {"image_plan": {"executed_media_sha256": "media", "observed_image_grid_thw": [1, 2, 2]}},
    }
    projected = e._project_legacy_stable_row(source, source_packet_sha256="packet", source_adapter_fingerprint="old-fp")
    projected = e._annotate_projection(
        projected,
        source_record=source,
        source_kind="legacy_exposed_eval_record",
        source_file_sha256="packet",
        source_adapter_fingerprint="old-fp",
    )
    assert projected["action_ids"] == source["stable_ids"]
    assert projected["image_id"] == source["image_id"]
    assert projected["stop_reason"] == source["stable_parsed"]["decode_stop_reason"]
    assert projected["text"] == source["stable_parsed"]["raw_decode_text"]
    assert projected["parsed"] == source["stable_parsed"]
    assert projected["score"] == source["stable_score"]
    assert projected["overlap_counts"] == source["stable_overlap_counts"]
    assert projected["source_adapter_fingerprint"] == "old-fp"
    assert projected["projection_exact_fields_sha256"] == e.digest(e._sealed_fields(projected))


def test_candidate_panel_strata_are_exact_and_false_overlap_fails():
    records = [{"image_id": image_id} for image_id in range(640)]
    strata = e._candidate_panel_strata(
        records,
        admitted_ids=range(11),
        reference_ids=range(11, 65),
        fresh_ids=range(384, 640),
    )
    assert {name: value["count"] for name, value in strata.items()} == {
        "train11": 11,
        "reference54": 54,
        "legacy_retention": 319,
        "fresh256": 256,
        "remaining_retention": 575,
    }
    assert set(strata["train11"]["image_ids"]) | set(strata["reference54"]["image_ids"]) | set(strata["remaining_retention"]["image_ids"]) == set(range(640))
    with pytest.raises(ValueError, match="strata overlap"):
        e._candidate_panel_strata(
            records,
            admitted_ids=[0, 384, *range(1, 11)],
            reference_ids=range(11, 65),
            fresh_ids=range(384, 640),
        )


def test_candidate_panel_contract_rejects_dropped_or_duplicate_identity():
    packet = _synthetic_panel_packet()
    e._validate_candidate_panel_packet(packet)
    dropped = copy.deepcopy(packet)
    dropped["records"] = dropped["records"][:-1]
    with pytest.raises(ValueError, match="record denominator"):
        e._validate_candidate_panel_packet(dropped)
    duplicate = copy.deepcopy(packet)
    duplicate["records"][-1] = copy.deepcopy(duplicate["records"][0])
    with pytest.raises(ValueError, match="image identity"):
        e._validate_candidate_panel_packet(duplicate)


def test_blind_freeze_source_map_has_no_arm_or_gt_labels(tmp_path):
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({"blind_review_ids": list(range(32))}))
    records = [
        {
            "image_id": image_id,
            "example_id": f"example-{image_id}",
            "case": {"image_path": f"/images/{image_id}.jpg", "image_width": 10, "image_height": 10},
        }
        for image_id in range(32)
    ]
    freeze, _ = e._write_blind_freeze(output=tmp_path, selection_path=selection_path, fresh_records=records)
    source_map = json.loads((tmp_path / "candidate-blind-review-source-map-v2.json").read_text())
    assert freeze["image_ids"] == list(range(32))
    assert all(set(row) == {"review_id", "image_id", "example_id", "literal_source_canvas_path", "image_width", "image_height"} for row in source_map["rows"])
    assert "Stable50" not in json.dumps(source_map) and "scaled_terminal" not in json.dumps(source_map)


def test_candidate_panel_binding_still_requires_a_real_adapter(tmp_path):
    packet_path = tmp_path / "candidate-panel.json"
    packet_path.write_text(json.dumps(_synthetic_panel_packet()))
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps({"label": "scaled_terminal", "status": "cold_verified", "admitted_packages": 16}))
    with pytest.raises(ValueError, match="candidate adapter identity"):
        e.bind_candidate_panel(packet_path=packet_path, candidate_path=candidate_path)


def test_trusted_target_score_recovers_unlabeled_review_target_without_gt():
    target = {
        "object_id": "417044:review:P6",
        "description": "donut",
        "bbox": [100, 100, 300, 300],
    }
    row = {
        "row_id": "coco2017_train_000000417044",
        "image_width": 1000,
        "image_height": 1000,
        "gt": [],
        "pred": [{"description": "donut", "bbox": [100, 100, 300, 300]}],
        "dropped_prediction_count": 0,
        "action_ids": [1, 2],
        "stop_reason": "im_end",
    }
    scored = e._trusted_target_score(row, [target])
    assert scored["50"]["owners"] == ["417044:review:P6"]
    assert scored["50"]["tp"] == 1 and scored["50"]["fn"] == 0


def test_trusted_target_score_joint_matching_rejects_malformed_group_and_duplicate_credit():
    targets = [
        {"object_id": "owner-a", "description": "donut", "bbox": [100, 100, 300, 300]},
        {"object_id": "owner-b", "description": "donut", "bbox": [600, 600, 800, 800]},
    ]
    row = {
        "row_id": "reviewed-c",
        "image_width": 1000,
        "image_height": 1000,
        "gt": [],
        "pred": [
            {"description": "donut", "bbox": [100, 100, 300, 300]},
            {"description": "donut", "bbox": [100, 100, 300, 300]},
            {"description": "donut", "bbox": [700, 700, 700, 900]},
            {"description": "food", "bbox": [600, 600, 800, 800], "group": ["donut", "donut"]},
        ],
        "dropped_prediction_count": 0,
        "action_ids": [],
        "stop_reason": "im_end",
    }
    scored = e._trusted_target_score(row, targets)
    assert scored["50"]["owners"] == ["owner-a"]
    assert scored["50"]["tp"] == 1 and scored["50"]["fn"] == 1
    assert scored["invalid_predictions"] == 1
    assert "owner-b" not in scored["50"]["owners"]


def test_candidate_launch_contract_freezes_eight_way_budget_and_entrypoints(tmp_path):
    contract = e._candidate_launch_contract(
        packet_path=tmp_path / "candidate-panel-bound.json",
        output_root=tmp_path / "candidate-natural",
    )
    assert contract["physical_gpus"] == list(range(8))
    assert contract["image_counts_per_worker"] == [80] * 8
    assert contract["max_new_tokens_total"] == 640 * e.CAP
    assert contract["max_new_tokens_per_worker"] == 80 * e.CAP
    assert contract["max_model_forwards_per_worker"] == 80 * (e.CAP + 1)
    assert contract["launch_command"][4] == "candidate-launch"
    assert len(contract["worker_commands"]) == 8
    assert all(command[3] == "candidate-worker" for command in contract["worker_commands"])


def test_actual_candidate_endpoint_schema_accepts_cold_fit_shape():
    candidate = {
        "schema": "native_owner_scale_state.evaluation.candidate_endpoint.v1",
        "status": "cold_verified",
        "label": "scaled_terminal",
        "admitted_packages": 16,
        "adapter": {"root": "/tmp/adapter", "fingerprint": "fp"},
        "training_receipt": {"path": "/tmp/receipt.json"},
        "cold_check": {"path": "/tmp/cold-check.json"},
        "training_completion": {"path": "/tmp/training-completion.json"},
        "training_input": {"path": "/tmp/inputs-v2.json"},
        "fit": {"updates": 256, "world_size": 8},
        "max_new_tokens": e.CAP,
        "dtype": "fp32",
        "attention": "sdpa",
        "claim_boundary": "natural rows remain unmeasured until consumer.",
    }
    e.validate_candidate_binding(candidate)
