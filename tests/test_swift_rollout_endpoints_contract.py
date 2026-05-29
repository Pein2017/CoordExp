from fastapi import FastAPI
from types import SimpleNamespace

from src.infer.backend_sync import (
    COORDEXP_WORKER_EXTENSION_CLS,
    CoordExpWeightSyncWorkerExtension,
    apply_coord_row_patch_for_rollout_server,
    apply_coord_row_patch_for_vllm_client,
)


def test_swift_rollout_server_registers_expected_endpoints() -> None:
    # CPU-only contract test against the local ms-swift install.
    apply_coord_row_patch_for_rollout_server()
    try:
        from swift.pipelines.infer.rollout import SwiftRolloutDeploy
    except ImportError:
        from swift.llm.infer.rollout import SwiftRolloutDeploy

    deploy = SwiftRolloutDeploy.__new__(SwiftRolloutDeploy)
    deploy.app = FastAPI()

    deploy._register_rl_rollout_app()

    paths = {getattr(r, "path", None) for r in deploy.app.routes}

    # Launcher health/readiness polling.
    assert "/health/" in paths
    assert "/get_world_size/" in paths

    # Inference endpoint used for rollouts.
    assert "/infer/" in paths

    # Adapter sync / communicator surface used by VLLMClient.
    assert "/init_communicator/" in paths
    assert "/close_communicator/" in paths
    assert "/update_adapter_flattened_param/" in paths
    assert "/update_adapter_param/" in paths
    assert "/update_token_row_offsets/" in paths


def test_coordexp_patch_adds_vllm_client_token_row_method() -> None:
    VLLMClient = apply_coord_row_patch_for_vllm_client()

    assert callable(getattr(VLLMClient, "update_token_row_offsets", None))


def test_coordexp_patch_selects_coordexp_worker_extension(monkeypatch) -> None:
    apply_coord_row_patch_for_rollout_server()
    from swift.pipelines.infer import rollout as rollout_mod

    captured = {}

    def fake_engine(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(rollout_mod, "GRPOVllmEngine", fake_engine)

    args = SimpleNamespace(
        model="model",
        model_type="qwen3_vl",
        model_revision=None,
        torch_dtype="bfloat16",
        vllm_use_async_engine=False,
        vllm_max_lora_rank=16,
        infer_backend="vllm",
        vllm_enable_lora=True,
        vllm_data_parallel_size=1,
        get_vllm_engine_kwargs=lambda: {"engine_kwargs": {"load_format": "auto"}},
    )

    rollout_mod.SwiftRolloutDeploy.get_infer_engine(args, template="qwen3_vl")

    assert captured["engine_kwargs"]["worker_extension_cls"] == COORDEXP_WORKER_EXTENSION_CLS
    assert callable(getattr(CoordExpWeightSyncWorkerExtension, "update_token_row_offsets"))
