"""Caller-facing contract checks for review-only COCO22 discovery."""

import copy
import json
import queue
import subprocess
import sys

import pytest
from PIL import Image

from probes.training_set_completion import coco22_acquisition as a
from probes.training_set_completion import review_packets as visuals


NEW = [296894, 196090, 210584, 109707, 260604, 29802, 457861,
       19413, 510122, 116096, 438671]


def test_44_new_image_requests_reuse_independent_sample_seeds():
    rows = a.request_plan(NEW)
    assert len(rows) == 44
    assert [row["temperature"] for row in rows[:4]] == [0.0, 0.1, 0.3, 0.7]
    assert all(row["grouping"] == "one_request_per_generate_call" for row in rows)
    assert len({row["seed"] for row in rows if row["kind"] == "sample"}) == 33
    assert rows[1]["seed"] == a.frozen.sample_seed(NEW[0], 0.1)
    reused = copy.deepcopy(rows)
    reused[2]["seed"] = rows[1]["seed"]
    with pytest.raises(ValueError, match="sample seeds"):
        a.validate_request_plan(reused, NEW)
    with pytest.raises(ValueError, match="request denominator"):
        a.validate_request_plan(rows[:-1], NEW)
    with pytest.raises(ValueError, match="request image cohort"):
        a.request_plan([*NEW[:-1], a.frozen.IMAGE_IDS[0]])


def test_first_greedy_then_all_43_unique_requests_across_eight_shards():
    rows = a.request_plan(NEW)
    manifest = {"requests": rows, "execution": {"first_request_id": rows[0]["request_id"]}}
    first = a._phase_requests(manifest, "first", 0, 1)
    remaining = [row for shard in range(8)
                 for row in a._phase_requests(manifest, "remaining", shard, 8)]
    assert len(first) == 1 and first[0]["kind"] == "greedy"
    assert len(remaining) == 43
    assert set(row["request_id"] for row in first).isdisjoint(row["request_id"] for row in remaining)
    assert {row["request_id"] for row in first + remaining} == {row["request_id"] for row in rows}


def test_result_replay_requires_exact_prompt_media_terminal_and_policy_identity():
    request = a.request_plan(NEW)[0]
    record = {"example_id": "new", "image_id": NEW[0], "prompt_token_ids": [4, 5],
              "case": {"image_plan": {"executed_media_sha256": "media",
                                      "observed_image_grid_thw": [1, 2, 3]}}}
    ids = [7, a.EOS]
    payload = {"schema": f"{a.SCHEMA}.row", "request": dict(request),
               "manifest_sha256": "bound", "example_id": "new", "image_id": NEW[0],
               "empty_assistant_prefix": True, "assistant_token_cap": a.CAP,
               "prompt_token_ids": [4, 5], "prompt_token_ids_sha256": a.digest([4, 5]),
               "generated_token_ids": ids, "generated_token_ids_sha256": a.digest(ids),
               "generated_token_count": 2, "decode_stop_reason": "im_end", "raw_decode_text": "decoded",
               "executed_media_sha256": "media", "observed_image_grid_thw": [1, 2, 3]}
    a.validate_result_payload(payload, request, record, decode=lambda _: "decoded")
    invalid = copy.deepcopy(payload)
    invalid["generated_token_ids"] = [a.EOS, 7]
    invalid["generated_token_ids_sha256"] = a.digest(invalid["generated_token_ids"])
    with pytest.raises(ValueError, match="generated terminal"):
        a.validate_result_payload(invalid, request, record, decode=lambda _: "decoded")
    for key, wrong, reason in (("request", {**request, "seed": 42}, "result request"),
                               ("prompt_token_ids", [4, 6], "result prompt IDs"),
                               ("executed_media_sha256", "other", "result media")):
        invalid = copy.deepcopy(payload)
        invalid[key] = wrong
        with pytest.raises(ValueError, match=reason):
            a.validate_result_payload(invalid, request, record, decode=lambda _: "decoded")


def test_portable_wait_records_actual_child_exit_without_pidfd():
    child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    completions: queue.Queue = queue.Queue()
    a._wait_child(child, completions)
    receipt = completions.get_nowait()
    assert receipt["pid"] == child.pid
    assert receipt["exit_code"] == 7 and receipt["wait_error"] is None


def test_named_tmux_controller_and_eight_disjoint_worker_commands(monkeypatch, tmp_path):
    rows = a.request_plan(NEW)
    manifest = {"requests": rows, "execution": {"first_request_id": rows[0]["request_id"]}}
    monkeypatch.setattr(a, "validate_manifest", lambda _: manifest)
    manifest_path = tmp_path / "manifest.json"
    specs = a.worker_commands(manifest_path=manifest_path, output=tmp_path, phase="remaining")
    assert [entry["physical_gpu"] for entry in specs] == list(range(8))
    assert sum(entry["expected_requests"] for entry in specs) == 43
    assert len({entry["log"] for entry in specs}) == 8
    assert all(entry["command"][0] == sys.executable and
               "probes.training_set_completion.coco22_acquisition" in entry["command"] for entry in specs)
    command = a.tmux_command(manifest_path=manifest_path, output=tmp_path, phase="remaining")
    assert command[:5] == ["tmux", "new-session", "-d", "-s", a.TMUX_REMAINING]
    assert "--phase remaining" in command[5]


def test_review_packets_keep_every_valid_and_parser_dropped_candidate_pending(monkeypatch, tmp_path):
    image_path = tmp_path / "original.png"
    Image.new("RGB", (16, 16), "white").save(image_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    requests = a.request_plan(NEW)
    gt = {"object_id": "1", "description": "person", "bbox": [100, 100, 800, 800],
          "metadata": {"source": {"category_id": 1}}}
    records = [{"image_id": image_id, "image_file": a.binding(image_path),
                "golden": {"gt": [gt], "image_width": 16, "image_height": 16}}
               for image_id in NEW]
    manifest = {"cohort": {"new_image_ids": NEW}, "records": records,
                "requests": requests, "content_sha256": "frozen"}
    monkeypatch.setattr(a, "validate_manifest", lambda _: manifest)
    (tmp_path / "result.json").write_text(json.dumps({"status": "passed", "request_count": 44,
                                                       "manifest": a.binding(manifest_path)}))
    bbox = visuals._pixel_from_bins(gt["bbox"], 16, 16)
    parsed = {"image_path": str(image_path), "image_width": 16, "image_height": 16,
              "pred": [{"description": "person", "bbox": bbox,
                        "coord_bins": gt["bbox"], "generated_order": 0}],
              "dropped_predictions": [{"reason": "incomplete", "generated_order": 1,
                                       "raw_text": "<|object_ref_start|>ghost"}]}
    with (tmp_path / "rows.jsonl").open("w") as stream:
        for request in requests:
            stream.write(json.dumps({"request": request, "image_id": request["image_id"],
                                     "manifest_sha256": "frozen", "parsed": parsed,
                                     "decode_stop_reason": "im_end"}) + "\n")
    gt_index = tmp_path / "gt-review-index.jsonl"
    with gt_index.open("w") as stream:
        for image_id in NEW:
            stream.write(json.dumps({"image_id": image_id, "owner_id": f"gt:{image_id}:0",
                                     "visual": {"original": {"path": str(image_path)}}}) + "\n")
    receipt = a.build_review_packets(manifest_path=manifest_path, output=tmp_path,
                                     gt_index_path=gt_index)
    assert receipt["image_count"] == 11 and receipt["request_count"] == 44
    assert receipt["raw_candidate_count"] == 88 and receipt["crop_count"] == 44
    assert receipt["no_truth_or_teacher_admissions"] is True
    packet = a.read(tmp_path / "review-packets-v1" / f"image-{NEW[0]:012d}" / "packet.json")
    assert packet["status"] == "candidate_pending_native_visual_review"
    assert packet["gt_visual_references"][0]["owner_id"] == f"gt:{NEW[0]}:0"
    assert len(packet["policies"]) == 4
    assert all(policy["proposals"][0]["crop_path"] for policy in packet["policies"])
    assert all(policy["proposals"][1]["status"] == "parser_dropped_raw_visible"
               for policy in packet["policies"])
