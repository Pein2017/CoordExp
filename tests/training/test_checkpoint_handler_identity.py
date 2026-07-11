from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.training.pipeline import _checkpoint_handler, _final_handler
from src.training.supervised_trainer import CompletedStepObservation


def _observation(step: int = 5) -> CompletedStepObservation:
    return CompletedStepObservation(
        planned_step_id=step,
        micro_step_count=1,
        loss_bundle_artifact={"metrics": {"acc_top1": 0.5}},
        optimizer_update_status="applied",
        finite_status="finite",
    )


def _handlers() -> tuple[list[dict[str, Any]], Any, Any]:
    calls: list[dict[str, Any]] = []
    writer = SimpleNamespace(write_checkpoint=lambda **kwargs: calls.append(kwargs))
    accelerator = SimpleNamespace(is_main_process=True)
    runtime = SimpleNamespace(accelerator=accelerator)
    schedule = SimpleNamespace(resolved_max_steps=5)
    committed: set[int] = set()
    checkpoint = _checkpoint_handler(
        writer,
        model=object(),
        runtime=runtime,
        adapter_name="default",
        special_token_result=None,
        schedule=schedule,
        base_model_path=Path("/models/qwen-base"),
        base_config_sha256="base-config-sha",
        tokenizer_sha256="tokenizer-sha",
        writer=None,
        eval_by_step={},
        committed_steps=committed,
        lifecycle={},
    )
    return calls, checkpoint, _final_handler(
        checkpoint_handler=checkpoint, committed_steps=committed
    )


def test_checkpoint_handler_passes_loader_identity_and_all_rank_runtime() -> None:
    calls, checkpoint, _ = _handlers()
    checkpoint(SimpleNamespace(planned_step_id=5), _observation())
    assert calls[0]["base_model_path"] == Path("/models/qwen-base")
    assert calls[0]["base_config_sha256"] == "base-config-sha"
    assert calls[0]["tokenizer_sha256"] == "tokenizer-sha"
    assert calls[0]["adapter_name"] == "default"
    assert calls[0]["is_final"] is True


def test_final_only_cadence_saves_inference_loadable_final_checkpoint() -> None:
    calls, _, final = _handlers()
    final(SimpleNamespace(planned_step_id=5), _observation())
    assert len(calls) == 1 and calls[0]["is_final"] is True


def test_same_step_checkpoint_and_final_are_deduplicated() -> None:
    calls, checkpoint, final = _handlers()
    event = SimpleNamespace(planned_step_id=5)
    checkpoint(event, _observation())
    final(event, _observation())
    assert len(calls) == 1
