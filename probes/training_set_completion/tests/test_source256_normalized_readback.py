from __future__ import annotations

import copy

import pytest

from probes.training_set_completion import source256_normalized_readback as readback
from probes.training_set_completion import training


def test_successor_readback_declares_only_the_two_new_saved_endpoints() -> None:
    assert readback.ENDPOINTS == (("Bnormalized16", 16), ("Bnormalized64", 64))
    with pytest.raises(ValueError, match="new-only"):
        readback._endpoint("B64", 64)


def test_endpoint_command_binds_the_successor_control_receipt(tmp_path) -> None:
    command = readback.endpoint_command(
        control_reuse_path=tmp_path / "control-reuse.json",
        plan_path=tmp_path / "plan.json",
        qualification_path=tmp_path / "qualification.json",
        training_manifest_path=tmp_path / "manifest.json",
        terminal_path=tmp_path / "terminal.json",
        label="Bnormalized64",
        step=64,
        split="train",
        shard=0,
        output=tmp_path / "shard.json",
        device="cuda:7",
    )

    assert command[0]
    assert command[command.index("--control-reuse") + 1].endswith("control-reuse.json")
    assert command[command.index("--label") + 1] == "Bnormalized64"
    assert command[command.index("--device") + 1] == "cuda:7"


def test_generation_identity_rejects_a_row_from_the_wrong_requested_cohort() -> None:
    items = [
        {
            "split": "train",
            "row_index": index,
            "image_id": index,
            "example_id": f"example-{index}",
            "expected_prompt_token_ids": [index, index + 10],
            "expected_media_sha256": "a" * 64,
            "expected_grid": [1, 2, 3],
        }
        for index in range(4)
    ]
    rows = [
        {
            "split": item["split"],
            "row_index": item["row_index"],
            "image_id": item["image_id"],
            "example_id": item["example_id"],
            "prompt_token_ids": item["expected_prompt_token_ids"],
            "prompt_token_ids_sha256": training.digest(item["expected_prompt_token_ids"]),
            "generated_token_ids": [readback.predecessor_readback.EOS],
            "generated_token_ids_sha256": training.digest([readback.predecessor_readback.EOS]),
            "decode_stop_reason": "im_end",
            "raw_decode_text": "x",
            "executed_media_sha256": item["expected_media_sha256"],
            "observed_image_grid_thw": item["expected_grid"],
            "batch_index": 0,
            "actual_batch_size": 4,
        }
        for item in items
    ]
    generation = {
        "status": "completed",
        "configured_batch_size": 4,
        "request_count": 4,
        "generated_tokens": 4,
        "batches": [{"batch_index": 0, "actual_batch_size": 4, "generated_tokens": 4}],
        "rows": rows,
    }

    readback._validate_generation_identity(generation=generation, items=items)

    bad = copy.deepcopy(generation)
    bad["rows"][0]["prompt_token_ids"] = [999]
    with pytest.raises(ValueError, match="generation row identity"):
        readback._validate_generation_identity(generation=bad, items=items)


def test_qualification_checkpoint_command_is_one_bound_batch4_step2(tmp_path) -> None:
    command = readback.qualification_checkpoint_command(
        control_reuse_path=tmp_path / "control-reuse.json",
        plan_path=tmp_path / "plan.json",
        qualification_path=tmp_path / "batch4-qualification.json",
        training_manifest_path=tmp_path / "qualification-manifest.json",
        terminal_path=tmp_path / "qualification-terminal.json",
        output=tmp_path / "qualification-checkpoint-batch4.json",
        device="cuda:3",
    )

    assert command[command.index("--qualification") + 1].endswith("batch4-qualification.json")
    assert command[command.index("--training-manifest") + 1].endswith("qualification-manifest.json")
    assert command[command.index("--terminal") + 1].endswith("qualification-terminal.json")
    assert command[command.index("--device") + 1] == "cuda:3"
    assert "qualification-checkpoint-worker" in command
