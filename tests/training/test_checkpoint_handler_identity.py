from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.training.pipeline import _checkpoint_handler


def test_checkpoint_handler_passes_qwen_sha_identity_to_writer() -> None:
    calls: list[dict[str, Any]] = []
    writer = SimpleNamespace(write_checkpoint=lambda **kwargs: calls.append(kwargs))
    schedule = SimpleNamespace(
        resolved_max_steps=5,
        runtime_batch=SimpleNamespace(to_artifact_dict=lambda: {"effective_batch_size": 1}),
        events={},
    )
    event = SimpleNamespace(
        scheduled_event=SimpleNamespace(
            planned_step_id=5,
            trigger_reasons=("checkpoint.final",),
        ),
        step_result=SimpleNamespace(
            finite_status="finite",
            optimizer_update_status="applied",
            to_artifact_dict=lambda: {"loss_bundle": {"loss": 1.0}},
        ),
    )

    handler = _checkpoint_handler(
        writer,
        model=object(),
        runtime=None,
        adapter_receipt={"status": "validated"},
        special_token_result={"status": "installed"},
        trainable_surface={"trainable": True},
        processor_identity={"name": "qwen-test"},
        resolved_config_fingerprint="config-fingerprint",
        schedule=schedule,
        base_model_path=Path("/models/qwen-base"),
        base_config_sha256="base-config-sha",
        tokenizer_sha256="tokenizer-sha",
    )

    handler(event)

    assert calls[0]["base_model_path"] == Path("/models/qwen-base")
    assert calls[0]["base_config_sha256"] == "base-config-sha"
    assert calls[0]["tokenizer_sha256"] == "tokenizer-sha"
