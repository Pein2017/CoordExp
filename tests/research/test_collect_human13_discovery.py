from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.research.collect_human13_discovery as adapter
from scripts.research.build_human13_k_union_manifest import (
    EXPECTED_IMAGE_IDENTITIES,
    EXPECTED_K_SEEDS,
    ImageInput,
    RequestIdentity,
    TrajectoryInput,
    build_manifest,
    default_binding,
    load_frozen_panel,
)


ROW_PARTS = (
    "<|object_ref_start|>",
    "cat",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|coord_100|>",
    "<|coord_200|>",
    "<|coord_300|>",
    "<|coord_400|>",
    "<|box_end|>",
)
ROW_TEXT = "".join(ROW_PARTS)


def _trace(parts: tuple[str, ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "step_index": index,
            "token_id": 1000 + index,
            "token_text": text,
            "is_stop": text == "<|im_end|>",
            "is_pad": False,
        }
        for index, text in enumerate(parts)
    )


def _result(*, repeated: bool = False, stop: bool = True) -> SimpleNamespace:
    body = ROW_PARTS * (2 if repeated else 1)
    parts = body + (("<|im_end|>",) if stop else ())
    return SimpleNamespace(
        request_id="request",
        backend="hf",
        backend_mode="generate",
        response_family="transformers",
        generated_token_ids=tuple(1000 + index for index in range(len(parts))),
        parser_text="".join(body),
        raw_generated_text="".join(parts),
        stop_reason="im_end" if stop else "length",
        token_trace=_trace(parts),
        executed_media_sha256="a" * 64,
    )


def _request(*, mode: str, seed: int | None = None) -> RequestIdentity:
    if mode == "source":
        return RequestIdentity(
            backend="hf",
            backend_version="test-hf",
            mode="source_greedy",
            n=1,
            seed=None,
            physical_batch_index=0,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            max_new_tokens=3084,
        )
    assert seed is not None
    return RequestIdentity(
        backend="vllm",
        backend_version="test-vllm",
        mode="k_sampled",
        n=1,
        seed=seed,
        physical_batch_index=(seed - 21001) // 4,
        temperature=0.4,
        top_p=0.95,
        repetition_penalty=1.10,
        max_new_tokens=512,
    )


def _empty_trajectory(
    *, image_id: int, mode: str, seed: int | None = None
) -> TrajectoryInput:
    request = _request(mode=mode, seed=seed)
    trajectory_id = (
        f"human13:{image_id}:source"
        if mode == "source"
        else f"human13:{image_id}:k16:{seed}"
    )
    return TrajectoryInput(
        trajectory_id=trajectory_id,
        request=request,
        token_ids=(999,),
        terminal_token_index=0,
        stop_reason="im_end",
        parser_status="empty",
        rows=(),
    )


def _record(*, frozen: object, mode: str, seed: int | None = None) -> dict[str, object]:
    image_id = int(getattr(frozen, "image_id"))
    trajectory = _empty_trajectory(image_id=image_id, mode=mode, seed=seed)
    binding = default_binding()
    return {
        "schema_version": adapter.RECORD_SCHEMA_VERSION,
        "mode": mode,
        "image_id": image_id,
        "panel_row_sha256": str(getattr(frozen, "panel_row_sha256")),
        "image_sha256": str(getattr(frozen, "image_sha256")),
        "prompt_identity": {
            "prompt_policy_fingerprint": binding.surface.prompt_policy_fingerprint,
            "prompt_token_ids_sha256": "b" * 64,
            "chat_text_sha256": "c" * 64,
        },
        "execution_model_identity": {
            "source_identity": asdict(binding.source),
            "mode": "dynamic_hf" if mode == "source" else "materialized_vllm",
        },
        "session_identity": {
            "backend": "hf" if mode == "source" else "vllm",
            "backend_version": "test",
            "backend_mode": "test",
            "response_family": "test",
            "session_identity_sha256": "d" * 64,
        },
        "runtime_counters": {
            "physical_batch_index": trajectory.request.physical_batch_index,
            "physical_batch_size": 1 if mode == "source" else 4,
            "request_id": trajectory.trajectory_id,
            "generated_token_count": 1,
            "batch_elapsed_seconds": 0.01,
        },
        "token_trace": [
            {
                "step_index": 0,
                "token_id": 999,
                "token_text": "<|im_end|>",
                "is_stop": True,
                "is_pad": False,
            }
        ],
        "parse": {
            "parser_id": "compact-object-box-closed-v1",
            "parser_policy": "compact_object_box_closed_only",
            "parse_status": "empty",
            "dropped_predictions": [],
        },
        "trajectory": asdict(trajectory),
    }


def _write_artifact(root: Path, *, mode: str, records: list[dict[str, object]]) -> None:
    root.mkdir(parents=True)
    payload = b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for record in records
    )
    (root / adapter.RECORDS_NAME).write_bytes(payload)
    config_path = (
        adapter.SOURCE_CONFIG_PATH if mode == "source" else adapter.K_CONFIG_PATH
    )
    receipt = {
        "schema_version": adapter.COLLECTION_SCHEMA_VERSION,
        "status": "completed",
        "mode": mode,
        "binding": asdict(default_binding()),
        "config_identity": {
            "path": str(config_path),
            "fingerprint": adapter.load_exact_config(mode, config_path).fingerprint,
        },
        "session_identity": {
            "backend": "hf" if mode == "source" else "vllm",
            "session_identity_sha256": "d" * 64,
        },
        "runtime_counters": {
            "image_count": 13,
            "physical_batch_count": 13 if mode == "source" else 52,
            "request_count": 13 if mode == "source" else 208,
            "generated_token_count": 13 if mode == "source" else 208,
            "wall_seconds": 1.0,
            "retry_count": 0,
            "resume_count": 0,
        },
        "records_file": adapter.RECORDS_NAME,
        "records_sha256": hashlib.sha256(payload).hexdigest(),
        "record_count": len(records),
    }
    (root / adapter.RECEIPT_NAME).write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_dry_run_is_zero_action_and_freezes_exact_source_and_k_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter,
        "_execute_runtime",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run entered runtime")
        ),
    )

    plan = adapter.dry_run_plan()

    assert plan["status"] == "plan_only"
    assert plan["actions"] == {
        "model_imports": 0,
        "model_loads": 0,
        "engine_opens": 0,
        "gpu_allocations": 0,
        "artifact_writes": 0,
    }
    assert plan["source"] == {
        "backend": "hf",
        "image_count": 13,
        "physical_batch_count": 13,
        "physical_batch_size": 1,
        "request_count": 13,
        "n": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 3084,
    }
    assert plan["k"] == {
        "backend": "vllm",
        "image_count": 13,
        "physical_batch_count": 52,
        "physical_batch_size": 4,
        "request_count": 208,
        "n": 1,
        "seeds": list(range(21001, 21017)),
        "temperature": 0.4,
        "top_p": 0.95,
        "repetition_penalty": 1.10,
        "max_new_tokens": 512,
    }


def test_exact_infer_configs_bind_source_and_k_runtime_scaffolds() -> None:
    source = adapter.load_exact_config("source", adapter.SOURCE_CONFIG_PATH)
    sampled = adapter.load_exact_config("k", adapter.K_CONFIG_PATH)

    assert source.config.backend.type == "hf"
    assert source.config.generation.batch_size == 1
    assert source.config.debug.smoke is True
    assert source.config.generation.max_new_tokens == 3084
    assert sampled.config.backend.type == "vllm"
    assert sampled.config.generation.batch_size == 4
    assert sampled.config.generation.max_new_tokens == 512
    assert sampled.config.generation.repetition_penalty == pytest.approx(1.10)
    with pytest.raises(ValueError, match="K discovery config"):
        adapter.validate_exact_config("k", source)


def test_absolute_char_spans_align_repeated_rows_but_text_only_lookup_fails_closed() -> (
    None
):
    result = _result(repeated=True)
    second_start = len(ROW_TEXT)

    assert adapter.align_char_span_to_token_span(
        parser_text=result.parser_text,
        token_trace=result.token_trace,
        span_text=ROW_TEXT,
        char_start=second_start,
        char_end=second_start + len(ROW_TEXT),
    ) == (len(ROW_PARTS), 2 * len(ROW_PARTS))
    with pytest.raises(ValueError, match="ambiguous"):
        adapter.align_char_span_to_token_span(
            parser_text=result.parser_text,
            token_trace=result.token_trace,
            span_text=ROW_TEXT,
            char_start=None,
            char_end=None,
        )


def test_projection_preserves_terminal_rows_and_final_coordinate_token() -> None:
    result = _result()

    projected, parse = adapter.trajectory_input_from_decode_result(
        image_id=1584,
        trajectory_id="human13:1584:source",
        request=_request(mode="source"),
        result=result,
        image_width=1000,
        image_height=1000,
    )

    assert projected.token_ids == result.generated_token_ids
    assert projected.terminal_token_index == len(result.generated_token_ids) - 1
    assert projected.stop_reason == "im_end"
    assert projected.parser_status == "accepted"
    assert len(projected.rows) == 1
    row = projected.rows[0]
    assert row.token_start == 0
    assert row.token_end == len(ROW_PARTS)
    assert row.final_coordinate_token_index == 7
    assert row.category == "cat"
    assert row.parser_status == "complete"
    assert parse["dropped_predictions"] == []


def test_projection_rejects_bad_terminal_parser_or_token_alignment() -> None:
    wrong_stop = _result(stop=False)
    wrong_stop.stop_reason = "im_end"
    with pytest.raises(ValueError, match="terminal"):
        adapter.trajectory_input_from_decode_result(
            image_id=1584,
            trajectory_id="bad-stop",
            request=_request(mode="source"),
            result=wrong_stop,
            image_width=1000,
            image_height=1000,
        )

    wrong_trace = _result()
    wrong_trace.token_trace = tuple(
        {**item, "token_text": "dog" if item["step_index"] == 1 else item["token_text"]}
        for item in wrong_trace.token_trace
    )
    with pytest.raises(ValueError, match="token trace"):
        adapter.trajectory_input_from_decode_result(
            image_id=1584,
            trajectory_id="bad-trace",
            request=_request(mode="source"),
            result=wrong_trace,
            image_width=1000,
            image_height=1000,
        )

    bad_parser = _result()
    bad_parser.parser_text = "{not compact rows}"
    bad_parser.raw_generated_text = bad_parser.parser_text + "<|im_end|>"
    bad_parser.token_trace = _trace((bad_parser.parser_text, "<|im_end|>"))
    bad_parser.generated_token_ids = (1000, 1001)
    with pytest.raises(ValueError, match="parser status"):
        adapter.trajectory_input_from_decode_result(
            image_id=1584,
            trajectory_id="bad-parser",
            request=_request(mode="source"),
            result=bad_parser,
            image_width=1000,
            image_height=1000,
        )


def test_dispatch_is_thirteen_source_batch1_calls_and_fifty_two_k_four_calls() -> None:
    base_requests = {image_id: object() for image_id, _ in EXPECTED_IMAGE_IDENTITIES}
    source_calls: list[tuple[int, int]] = []
    k_calls: list[tuple[int, int, tuple[int, ...]]] = []

    source = adapter.dispatch_source_batches(
        base_requests,
        lambda image_id, requests: source_calls.append((image_id, len(requests)))
        or (f"source:{image_id}",),
    )
    sampled = adapter.dispatch_k_batches(
        base_requests,
        lambda batch, base_request: k_calls.append(
            (batch.image_id, len(batch.requests), tuple(r.seed for r in batch.requests))
        )
        or tuple(f"k:{request.seed}" for request in batch.requests),
    )

    assert len(source) == 13
    assert source_calls == [(image_id, 1) for image_id, _ in EXPECTED_IMAGE_IDENTITIES]
    assert len(sampled) == 208
    assert len(k_calls) == 52
    assert all(size == 4 for _, size, _ in k_calls)
    assert [seeds for _, _, seeds in k_calls[:4]] == [
        (21001, 21002, 21003, 21004),
        (21005, 21006, 21007, 21008),
        (21009, 21010, 21011, 21012),
        (21013, 21014, 21015, 21016),
    ]


def test_atomic_execution_never_overwrites_and_publishes_terminal_failure(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "source"

    with pytest.raises(RuntimeError, match="boom"):
        adapter.execute_atomically(
            output_root=output_root,
            mode="source",
            operation=lambda staging: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    failure = json.loads((output_root / adapter.RECEIPT_NAME).read_text())
    assert failure["status"] == "failed"
    assert failure["mode"] == "source"
    assert failure["error"]["type"] == "RuntimeError"

    with pytest.raises(FileExistsError, match="overwrite"):
        adapter.execute_atomically(
            output_root=output_root,
            mode="source",
            operation=lambda staging: None,
        )


def test_converter_requires_completed_exact_coverage_and_admits_canonical_manifest(
    tmp_path: Path,
) -> None:
    frozen = load_frozen_panel()
    source_records = [_record(frozen=row, mode="source") for row in frozen]
    k_records = [
        _record(frozen=row, mode="k", seed=seed)
        for row in frozen
        for seed in EXPECTED_K_SEEDS
    ]
    source_root = tmp_path / "source"
    k_root = tmp_path / "k"
    _write_artifact(source_root, mode="source", records=source_records)
    _write_artifact(k_root, mode="k", records=k_records)

    images = adapter.artifacts_to_image_inputs(
        source_root=source_root,
        k_root=k_root,
    )
    manifest = build_manifest(binding=default_binding(), images=images)

    assert len(images) == 13
    assert all(isinstance(image, ImageInput) for image in images)
    assert all(len(image.sampled) == 16 for image in images)
    assert manifest.full_panel is True
    assert len(manifest.images) == 13


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("failed_receipt", "completed"),
        ("bad_digest", "SHA-256"),
        ("missing_seed", "sixteen"),
        ("duplicate_seed", "sixteen"),
        ("bad_image_hash", "image-content"),
        ("bad_request", "frozen recipe"),
        ("bad_parser", "parser status"),
        ("bad_terminal_index", "terminal token index"),
        ("bad_record_runtime", "runtime request identity"),
        ("bad_session_receipt", "session identity"),
        ("bad_runtime_receipt", "runtime counters"),
        ("bad_config_receipt", "config identity"),
    ],
)
def test_converter_fails_closed_on_bad_receipt_result_or_identity(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    frozen = load_frozen_panel()
    source_records = [_record(frozen=row, mode="source") for row in frozen]
    k_records = [
        _record(frozen=row, mode="k", seed=seed)
        for row in frozen
        for seed in EXPECTED_K_SEEDS
    ]
    if mutation == "missing_seed":
        k_records.pop()
    elif mutation == "duplicate_seed":
        k_records[-1] = dict(k_records[-2])
    elif mutation == "bad_image_hash":
        source_records[0]["image_sha256"] = "0" * 64
    elif mutation == "bad_request":
        trajectory = dict(k_records[0]["trajectory"])
        request = dict(trajectory["request"])
        request["temperature"] = 0.5
        trajectory["request"] = request
        k_records[0]["trajectory"] = trajectory
    elif mutation == "bad_parser":
        parse = dict(source_records[0]["parse"])
        parse["parse_status"] = "invented"
        source_records[0]["parse"] = parse
    elif mutation == "bad_terminal_index":
        trajectory = dict(source_records[0]["trajectory"])
        trajectory["terminal_token_index"] = None
        source_records[0]["trajectory"] = trajectory
    elif mutation == "bad_record_runtime":
        runtime = dict(source_records[0]["runtime_counters"])
        runtime["request_id"] = "wrong"
        source_records[0]["runtime_counters"] = runtime

    source_root = tmp_path / "source"
    k_root = tmp_path / "k"
    _write_artifact(source_root, mode="source", records=source_records)
    _write_artifact(k_root, mode="k", records=k_records)
    if mutation == "failed_receipt":
        receipt = json.loads((source_root / adapter.RECEIPT_NAME).read_text())
        receipt["status"] = "failed"
        (source_root / adapter.RECEIPT_NAME).write_text(json.dumps(receipt) + "\n")
    elif mutation == "bad_digest":
        with (k_root / adapter.RECORDS_NAME).open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
    elif mutation in {
        "bad_session_receipt",
        "bad_runtime_receipt",
        "bad_config_receipt",
    }:
        receipt = json.loads((source_root / adapter.RECEIPT_NAME).read_text())
        if mutation == "bad_session_receipt":
            receipt["session_identity"]["session_identity_sha256"] = "f" * 64
        elif mutation == "bad_runtime_receipt":
            receipt["runtime_counters"]["request_count"] = 12
        else:
            receipt["config_identity"]["fingerprint"] = "not-a-digest"
        (source_root / adapter.RECEIPT_NAME).write_text(json.dumps(receipt) + "\n")

    with pytest.raises(ValueError, match=message):
        adapter.artifacts_to_image_inputs(
            source_root=source_root,
            k_root=k_root,
        )
