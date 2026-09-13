from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from probes.row_feedback import evaluation as subject
from probes.row_feedback import evaluation_run as runner
from probes.row_feedback import runtime as feedback_runtime


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _small_sources(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "original" / "val.coord.jsonl"
    native = tmp_path / "native" / "val.coord.jsonl"
    source.parent.mkdir(parents=True)
    native.parent.mkdir(parents=True)
    rows = []
    for image_id in range(1, 34):
        image = tmp_path / "images" / f"{image_id}.jpg"
        image.parent.mkdir(exist_ok=True)
        image.write_bytes(f"image-{image_id}".encode())
        objects = [
            {
                "bbox_2d": ["<|coord_100|>", "<|coord_100|>", "<|coord_300|>", "<|coord_300|>"],
                "desc": "cat",
                "category_id": 17,
                "category_name": "cat",
                "coco_ann_id": image_id * 10,
            }
        ]
        if image_id == 1:
            objects += [dict(objects[0], coco_ann_id=11), dict(objects[0], coco_ann_id=12)]
        rows.append(
            {
                "images": [str(image)],
                "objects": objects,
                "width": 100,
                "height": 100,
                "image_id": image_id,
                "file_name": image.name,
                "metadata": {},
            }
        )
    payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    source.write_text(payload)
    native.write_text(payload)
    return source, native


def _frozen_exclusion(tmp_path: Path) -> Path:
    path = tmp_path / "excluded.json"
    _write_json(
        path,
        {
            "schema": "native_owner_scale_state.evaluation.selection.v1",
            "status": "frozen_before_new_outputs",
            "image_ids": [33],
            "blind_review_ids": [],
            "excluded_image_ids": [],
        },
    )
    return path


def _runtime(arm: str, text: str = "") -> dict:
    ids = ([1] if text else []) + [151645]
    return {
        "visible_token_ids": ids,
        "text": text + "<|im_end|>",
        "finish_reason": "eos",
        "eos": True,
        "cap": False,
        "visible_generated_tokens": len(ids),
        "internal_slot_count": 0,
        "model_forwards": 1,
        "image_forwards": 1,
        "timing": {"wall_seconds": 1.0},
        "slot_work": {"prefill": 0, "history": 0, "generated": 0, "total": 0},
        "arm": arm,
        "decode_contract": {
            "max_visible_tokens": 3084,
            "eos_token_id": 151645,
            "do_sample": False,
            "temperature": 0,
            "top_p": 1,
            "repetition_penalty": 1,
        },
    }


def _box() -> str:
    return (
        "<|object_ref_start|>cat<|object_ref_end|><|box_start|>"
        "<|coord_100|><|coord_100|><|coord_300|><|coord_300|><|box_end|>"
    )


def test_freeze_is_deterministic_disjoint_and_dense_review_is_preselected(tmp_path: Path) -> None:
    source, native = _small_sources(tmp_path)
    exclusion = _frozen_exclusion(tmp_path)
    first = subject.freeze_selection(
        output=tmp_path / "one",
        source=source,
        native_source=native,
        exclusion_specs=(("test-ledger", exclusion),),
    )
    second = subject.freeze_selection(
        output=tmp_path / "two",
        source=source,
        native_source=native,
        exclusion_specs=(("test-ledger", exclusion),),
    )
    assert first["image_ids"] == second["image_ids"]
    assert 33 not in first["image_ids"]
    assert first["dense_review_ids"][0] == 1
    assert first["model_calls"] == 0
    subject.validate_selection(first, verify_files=True)
    contract = subject.read(tmp_path / "one" / "cost-contract.json")
    assert contract["natural_calls"] == 64
    assert contract["max_visible_tokens_total"] == 64 * 3084
    assert contract["status"] == "awaiting_real_runtime_measurement"
    interface = subject.consumer_contract(tmp_path / "one" / "selection.json")
    assert interface["status"] == "cpu_verified_awaiting_real_outputs"
    assert "max_visible_tokens" in interface["runtime_call"]


def test_exclusion_contract_fails_closed_on_mutable_status(tmp_path: Path) -> None:
    path = _frozen_exclusion(tmp_path)
    value = subject.read(path)
    value["status"] = "running"
    _write_json(path, value)
    with pytest.raises(ValueError, match="unfrozen exclusion"):
        subject.exclusion_inventory((("bad", path),))


def test_runtime_validation_rejects_hidden_slot_and_visible_cap_inconsistency() -> None:
    value = _runtime("F")
    value["internal_slot_count"] = 1
    with pytest.raises(ValueError, match="internal slot count"):
        subject.validate_runtime_result(value, expected_arm="F")
    value = _runtime("F")
    value.update(finish_reason="length", eos=False, cap=True)
    value.update(visible_token_ids=[1], text="x", visible_generated_tokens=1)
    with pytest.raises(ValueError, match="length finish before visible cap"):
        subject.validate_runtime_result(value, expected_arm="F")


def test_sharding_and_prepared_input_fingerprint_are_deterministic() -> None:
    import torch

    ids = list(range(10))
    assert runner.shard_ids(ids, shard_index=1, shard_count=3) == [1, 4, 7]
    left = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)}
    right = {"attention_mask": torch.ones((1, 2), dtype=torch.long), "input_ids": torch.tensor([[1, 2]])}
    assert feedback_runtime.prepared_inputs_sha256(left) == feedback_runtime.prepared_inputs_sha256(right)
    right["input_ids"][0, 1] = 3
    assert feedback_runtime.prepared_inputs_sha256(left) != feedback_runtime.prepared_inputs_sha256(right)

    receipts = {}
    for image_id, prompt in ((465180, [1] * 1232), (226171, [1] * 1320)):
        prompt_sha = subject.digest(prompt)
        runner.add_materialization_receipt(
            receipts,
            image_id=image_id,
            materialized={
                "prompt_ids": prompt,
                "prepared_inputs_sha256": str(image_id).zfill(64),
                "materialization": {
                    "prompt_token_ids_sha256": prompt_sha,
                    "executed_media_sha256": "e" * 64,
                    "observed_image_grid_thw": [1, 2, 2],
                },
            },
        )
    assert receipts["465180"]["prompt_token_count"] == 1232
    assert receipts["226171"]["prompt_token_count"] == 1320


def test_adapter_binding_rechecks_payload_and_fit_receipt(tmp_path: Path, monkeypatch) -> None:
    import src.adapters.dora

    adapter_root = tmp_path / "adapter"
    adapter_root.mkdir()
    weight = adapter_root / "adapter_model.safetensors"
    weight.write_bytes(b"weights-v1")

    def inspect(path):
        root = Path(path).resolve()
        return {
            "root": str(root),
            "fingerprint": subject.file_hash(root / "adapter_model.safetensors"),
            "files": [subject.binding(root / "adapter_model.safetensors")],
        }

    monkeypatch.setattr(src.adapters.dora, "inspect_dora_adapter_payload", inspect)
    adapter = inspect(adapter_root)
    fit, packet = tmp_path / "fit.json", tmp_path / "packet.json"
    _write_json(fit, {"saved_adapter": adapter})
    _write_json(packet, {"dose": 16})
    bound = tmp_path / "binding.json"
    _write_json(
        bound,
        {
            "schema": "row_feedback.endpoint_adapter_binding.v1",
            "status": "root_bound_after_paired_fit",
            "arm": "S",
            "dose": 16,
            "adapter": adapter,
            "fit_receipt": subject.binding(fit),
            "execution_packet": subject.binding(packet),
        },
    )
    assert runner.validate_adapter_binding(bound, arm="S")["adapter"] == adapter
    weight.write_bytes(b"weights-v2")
    with pytest.raises(ValueError, match="payload differs"):
        runner.validate_adapter_binding(bound, arm="S")


def test_consumer_requires_exact_pair_and_reports_gained_lost_retained_and_burden(tmp_path: Path) -> None:
    source, native = _small_sources(tmp_path)
    exclusion = _frozen_exclusion(tmp_path)
    output = tmp_path / "selection"
    selection = subject.freeze_selection(
        output=output,
        source=source,
        native_source=native,
        exclusion_specs=(("test-ledger", exclusion),),
    )
    selection_path = output / "selection.json"
    selection_sha = subject.file_hash(selection_path)
    s_rows, f_rows = [], []
    for index, record in enumerate(selection["records"]):
        s_text = _box() if index in (0, 2) else ""
        f_text = _box() if index in (1, 2) else ""
        if index == 2:
            f_text += _box()
        common = dict(
            selection_sha256=selection_sha,
            record=record,
            prompt_token_ids=[7, 8],
            prepared_inputs_sha256="a" * 64,
            model_identity={"base": "same", "source_embedding": "same"},
        )
        s_rows.append(
            subject.make_output_envelope(
                **common,
                arm="S",
                adapter_identity={"arm": "S"},
                runtime=_runtime("S", s_text),
            )
        )
        f_rows.append(
            subject.make_output_envelope(
                **common,
                arm="F",
                adapter_identity={"arm": "F"},
                runtime=_runtime("F", f_text),
            )
        )
    s_path, f_path = tmp_path / "s.jsonl", tmp_path / "f.jsonl"
    s_path.write_text("".join(json.dumps(row) + "\n" for row in s_rows))
    f_path.write_text("".join(json.dumps(row) + "\n" for row in f_rows))
    shard_dirs = []
    for shard_index in range(2):
        shard_dir = tmp_path / f"s-shard-{shard_index}"
        shard_dir.mkdir()
        shard_rows = s_rows[shard_index::2]
        shard_path = shard_dir / "rows.jsonl"
        shard_path.write_text("".join(json.dumps(row) + "\n" for row in shard_rows))
        subject.publish(shard_dir / "launch.json", {"shard_index": shard_index})
        subject.publish(
            shard_dir / "terminal.json",
            {
                "status": "completed",
                "arm": "S",
                "shard_index": shard_index,
                "shard_count": 2,
                "image_ids": [row["image_id"] for row in shard_rows],
                "selection": subject.binding(selection_path),
                "adapter_binding": {"path": "/fixed/S", "sha256": "s"},
                "rows": subject.binding(shard_path),
                "launch": subject.binding(shard_dir / "launch.json"),
            },
        )
        shard_dirs.append(shard_dir)
    merged = runner.merge_shards(
        selection_path=selection_path,
        arm="S",
        shard_dirs=shard_dirs,
        output=tmp_path / "s-merged",
    )
    assert merged["images"] == 32
    assert [row["image_id"] for row in subject.read_jsonl(tmp_path / "s-merged" / "rows.jsonl")] == selection["image_ids"]
    result = subject.consume_pair(
        selection_path=selection_path,
        s_rows_path=s_path,
        f_rows_path=f_path,
        output=tmp_path / "result.json",
    )
    change = result["owner_changes_F_vs_S"]["50"]
    assert change["gained"] == 1
    assert change["lost"] == 1
    assert change["retained"] == 1
    assert result["aggregates"]["F"]["metrics"]["strict_repeats"] == 1
    assert result["aggregates"]["F"]["metrics"]["prediction_count"] == 3
    assert result["aggregates"]["F"]["runtime"]["visible_generated_tokens"] == 34
    assert result["aggregates"]["F"]["runtime"]["internal_slot_count"] == 0
    estimate = subject.estimate_endpoint_cost(s_rows_path=s_path, f_rows_path=f_path)
    assert estimate["observed"]["S"]["calls"] == 32
    assert estimate["observed"]["F"]["calls"] == 32
    assert estimate["projected_allocated_gpu_hours_at_observed_mean"] == pytest.approx(64 / 3600)
    review = runner.prepare_blind_review(
        selection_path=selection_path,
        s_rows_path=s_path,
        f_rows_path=f_path,
        output=tmp_path / "review",
    )
    queue = subject.read(tmp_path / "review" / "queue.json")
    source_map = subject.read(tmp_path / "review" / "source-map.json")
    assert review["status"] == "ready_for_render"
    assert queue["images"] == 8 and all(row["source_blind"] for row in queue["rows"])
    assert "source_arm" not in json.dumps(queue)
    assert {row["source_arm"] for row in source_map["rows"]} <= {"S", "F"}

    broken = copy.deepcopy(f_rows)
    broken[0]["prompt_token_ids_sha256"] = "b" * 64
    f_path.write_text("".join(json.dumps(row) + "\n" for row in broken))
    with pytest.raises(ValueError, match="paired prompt_token_ids_sha256"):
        subject.consume_pair(
            selection_path=selection_path,
            s_rows_path=s_path,
            f_rows_path=f_path,
            output=tmp_path / "must-not-exist.json",
        )
