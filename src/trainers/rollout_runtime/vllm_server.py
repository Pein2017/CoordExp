from __future__ import annotations

import threading
from typing import Any


def ensure_vllm_server_client(
    *,
    owner: Any,
    logger: Any,
) -> Any:
    if owner._vllm_server_client is not None:
        return owner._vllm_server_client

    lock = getattr(owner, "_vllm_server_client_lock", None)
    if lock is None:
        lock = threading.Lock()
        setattr(owner, "_vllm_server_client_lock", lock)

    with lock:
        if owner._vllm_server_client is not None:
            return owner._vllm_server_client

        rank = 0
        world_size = 1
        try:
            import torch.distributed as dist
        except (TypeError, ValueError):
            dist = None  # type: ignore[assignment]

        if dist is not None and dist.is_available() and dist.is_initialized():
            try:
                rank = int(dist.get_rank())
                world_size = int(dist.get_world_size())
            except (TypeError, ValueError):
                rank = 0
                world_size = 1

        servers = owner._vllm_server_specs()
        timeout_s, _infer_timeout_s = owner._vllm_server_timeouts()

        try:
            from swift.trainers.rlhf_trainer.vllm_client import VLLMClient
        except (ImportError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "vLLM server mode requires ms-swift's VLLMClient (and vLLM + pynccl). "
                "Install/enable vLLM in the ms env, or switch to vllm.mode=colocate or rollout_backend=hf."
            ) from exc

        base_urls = [str(s["base_url"]) for s in servers]
        group_ports = [int(s["group_port"]) for s in servers]

        try:
            client = VLLMClient(
                base_urls=base_urls,
                group_ports=group_ports,
                connection_timeout=float(timeout_s),
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Failed to connect to vLLM rollout server(s). "
                "Check rollout_matching.vllm.server (base_url/group_port) and ensure /health/ is reachable."
            ) from exc

        if int(world_size) > 1 and int(rank) == 0:
            logger.info(
                "vLLM server client created for multi-process learner; communicator init deferred (rank0-only). world_size=%s",
                int(world_size),
            )

        try:
            info = client.get_engine_type()
            logger.info("vLLM rollout server engine_type: %s", info)
        except (TypeError, ValueError):
            raise

        owner._vllm_server_client = client
        return client
