import json
from pathlib import Path

import pytest

from probes.training_set_completion import coco227_readback as readback
from src.qwen.native import NativeBatch, padded_histories


class Tokenizer:
    def decode(self, ids, **_kwargs):
        return ",".join(map(str, ids))


def _route(image_id: int, prompt: list[int]) -> dict:
    return {
        "image_id": image_id,
        "route_id": f"route-{image_id}",
        "example_id": f"example-{image_id}",
        "prompt_token_ids": prompt,
        "image_identity": {
            "executed_media_sha256": "a" * 64,
            "observed_image_grid_thw": [1, 2, image_id],
        },
        "case": {
            "row_id": f"example-{image_id}",
            "row_index": image_id,
            "image_path": f"/images/{image_id}.jpg",
            "image_width": 1000,
            "image_height": 1000,
            "input_record": {"objects": []},
        },
    }


def _bound_file(tmp_path: Path, name: str) -> dict:
    path = tmp_path / name
    path.write_text("{}\n")
    return readback.binding(path)


def _bound_json(tmp_path: Path, name: str, value: dict) -> dict:
    path = tmp_path / name
    path.write_bytes(readback.canonical(value))
    return readback.binding(path)


def _row(
    tmp_path: Path,
    route: dict,
    manifest: dict,
    terminal: dict,
    adapter: dict,
    *,
    ids: list[int] | None = None,
    stop: str = "im_end",
    batch_size: int = 2,
) -> dict:
    ids = ids or [7, readback.EOS]
    model_receipt = _bound_json(
        tmp_path,
        f"model-{route['image_id']}.json",
        {
            "schema": f"{readback.SCHEMA}.model_receipt",
            "checkpoint_adapter": adapter,
            "training_manifest": manifest,
            "training_terminal": terminal,
            "trial": None,
        },
    )
    raw_batch_receipt = _bound_json(
        tmp_path,
        f"batch-{route['image_id']}.json",
        {
            "schema": f"{readback.SCHEMA}.raw_batch",
            "configured_batch_size": batch_size,
            "actual_batch_size": 2,
            "image_ids": [route["image_id"], route["image_id"] + 100],
            "rows": [
                {
                    "image_id": route["image_id"],
                    "generated_token_ids": ids,
                    "generated_token_ids_sha256": readback.digest(ids),
                    "decode_stop_reason": stop,
                }
            ],
        },
    )
    return {
        "schema": f"{readback.SCHEMA}.row",
        "source_kind": "scientific_checkpoint_readback",
        "arm": "S",
        "step": 8,
        "checkpoint_step": 8,
        "route_id": route["route_id"],
        "example_id": route["example_id"],
        "image_id": route["image_id"],
        "request_id": route["case"]["row_id"],
        "policy": dict(readback.POLICY),
        "empty_assistant_prefix": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_new_tokens": readback.CAP,
        "batch_size": batch_size,
        "batch_index": 0,
        "batch_position": 0,
        "actual_batch_size": 2,
        "prompt_token_ids": route["prompt_token_ids"],
        "prompt_token_ids_sha256": readback.digest(route["prompt_token_ids"]),
        "generated_token_ids": ids,
        "generated_token_ids_sha256": readback.digest(ids),
        "raw_decode_text": Tokenizer().decode(ids),
        "decode_stop_reason": stop,
        "executed_media_sha256": route["image_identity"]["executed_media_sha256"],
        "observed_image_grid_thw": route["image_identity"]["observed_image_grid_thw"],
        "checkpoint_adapter": adapter,
        "adapter": adapter,
        "model_receipt": model_receipt,
        "raw_batch_receipt": raw_batch_receipt,
        "training_manifest": manifest,
        "training_terminal": terminal,
        "trial": None,
    }


def test_left_padding_and_last_partial_batch_preserve_request_rows():
    ids, mask = padded_histories([[1, 2, 3], [4]], pad_token_id=0)
    native = NativeBatch(
        {"input_ids": ids, "attention_mask": mask},
        ("long", "short"),
    )
    assert native.prompt_token_ids == ((1, 2, 3), (4,))
    assert readback.batches(list(range(5)), 2) == [[0, 1], [2, 3], [4]]


def test_per_request_terminal_validation_accepts_mixed_eos_and_length():
    readback._validate_stop([4, readback.EOS], "im_end")
    readback._validate_stop([4] * readback.CAP, "length")
    with pytest.raises(ValueError, match="EOS/cap"):
        readback._validate_stop([4, readback.EOS, 5], "im_end")
    with pytest.raises(ValueError, match="EOS/cap"):
        readback._validate_stop([4], "length")


def test_resume_validates_retained_row_and_returns_only_missing(tmp_path):
    routes = [_route(1, [1, 2]), _route(2, [3])]
    manifest = _bound_file(tmp_path, "manifest.json")
    terminal = _bound_file(tmp_path, "terminal.json")
    adapter = {"root": "/adapter", "fingerprint": "adapter-one"}
    root = tmp_path / "endpoint"
    saved = _row(tmp_path, routes[0], manifest, terminal, adapter)
    readback.publish(readback._row_path(root, 1), saved)
    retained, missing = readback.pending_routes(
        row_root=root,
        routes=routes,
        arm="S",
        step=8,
        batch_size=2,
        adapter=adapter,
        training_manifest=manifest,
        training_terminal=terminal,
        tokenizer=Tokenizer(),
        source_kind="scientific_checkpoint_readback",
    )
    assert [row["image_id"] for row in retained] == [1]
    assert [route["image_id"] for route in missing] == [2]

    changed = json.loads(readback._row_path(root, 1).read_text())
    changed["prompt_token_ids"] = [99]
    readback._row_path(root, 1).write_bytes(readback.canonical(changed))
    with pytest.raises(ValueError, match="prompt IDs"):
        readback.pending_routes(
            row_root=root,
            routes=routes,
            arm="S",
            step=8,
            batch_size=2,
            adapter=adapter,
            training_manifest=manifest,
            training_terminal=terminal,
            tokenizer=Tokenizer(),
            source_kind="scientific_checkpoint_readback",
        )


def test_rows_match_the_scoped_evaluator_endpoint_surface(tmp_path):
    from probes.training_set_completion import coco227_evaluation

    routes = [_route(image_id, [image_id]) for image_id in range(1, 12)]
    manifest = _bound_file(tmp_path, "manifest.json")
    terminal = _bound_file(tmp_path, "terminal.json")
    adapter = {"root": "/adapter", "fingerprint": "adapter-one"}
    rows = [
        _row(tmp_path, route, manifest, terminal, adapter) for route in routes
    ]
    coco227_evaluation._validate_saved_rows(
        rows,
        routes={route["image_id"]: route for route in routes},
        arm="S",
        step=8,
    )


def _parsed_row(description: str, bins: tuple[int, int, int, int], token: int) -> dict:
    text = (
        f"<|object_ref_start|>{description}<|object_ref_end|><|box_start|>"
        + "".join(f"<|coord_{value}|>" for value in bins)
        + "<|box_end|><|im_end|>"
    )
    return {
        "generated_token_ids": [token, readback.EOS],
        "raw_decode_text": text,
        "decode_stop_reason": "im_end",
    }


def test_detection_consistency_allows_one_bin_change_only_when_readout_is_same():
    route = _route(1, [1])
    reference = _parsed_row("person", (100, 100, 200, 200), 10)
    shifted = _parsed_row("person", (101, 100, 200, 200), 11)
    result = readback.detection_consistency(reference, shifted, route)
    assert result["parity"] is True
    assert result["exact_token_identity"] is False
    assert result["maximum_coordinate_bin_delta"] == 1

    wrong_description = _parsed_row("chair", (101, 100, 200, 200), 12)
    assert readback.detection_consistency(reference, wrong_description, route)["parity"] is False
