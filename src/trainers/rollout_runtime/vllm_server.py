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


def ensure_vllm_server_communicator_rank0(
    *,
    owner: Any,
    client: Any,
) -> None:
    if bool(getattr(owner, "_vllm_server_comm_inited", False)):
        return

    rank = 0
    try:
        import torch.distributed as dist
    except (TypeError, ValueError):
        dist = None  # type: ignore[assignment]

    if dist is not None and dist.is_available() and dist.is_initialized():
        rank = int(dist.get_rank())

    if int(rank) != 0:
        raise RuntimeError(
            "vLLM server communicator init must be rank0-only under DDP. "
            f"Got rank={int(rank)}."
        )

    try:
        client.init_communicator(device=0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Failed to initialize NCCL communicator with vLLM rollout server(s). "
            "Mitigations: verify group_port reachability, set NCCL env, or increase vllm.server.timeout_s."
        ) from exc

    setattr(owner, "_vllm_server_comm_inited", True)


def shutdown_vllm_server_client(
    *,
    owner: Any,
    logger: Any,
    close_communicator: bool = True,
    close_sessions: bool = True,
) -> None:
    lock = getattr(owner, "_vllm_server_client_lock", None)
    if lock is None:
        lock = threading.Lock()
        setattr(owner, "_vllm_server_client_lock", lock)

    with lock:
        client = getattr(owner, "_vllm_server_client", None)
        if client is None:
            setattr(owner, "_vllm_server_comm_inited", False)
            return

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

        if bool(close_communicator):
            should_close_comm = int(world_size) <= 1 or int(rank) == 0
            if should_close_comm:
                try:
                    close_fn = getattr(client, "close_communicator", None)
                    if callable(close_fn):
                        close_fn()
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "Failed to close vLLM server communicator during shutdown: %r",
                        exc,
                    )

        if bool(close_sessions):
            try:
                sessions = getattr(client, "sessions", None)
                if isinstance(sessions, list):
                    for sess in sessions:
                        try:
                            close_fn = getattr(sess, "close", None)
                            if callable(close_fn):
                                close_fn()
                        except (TypeError, ValueError):
                            raise
            except (TypeError, ValueError):
                raise

        owner._vllm_server_client = None
        setattr(owner, "_vllm_server_comm_inited", False)


def sync_vllm_server_rollout_model_if_needed(
    *,
    owner: Any,
) -> None:
    step = int(getattr(getattr(owner, "state", None), "global_step", 0) or 0)

    rank = 0
    world_size = 1
    try:
        import torch
        import torch.distributed as dist
    except (TypeError, ValueError):
        dist = None  # type: ignore[assignment]
        torch = None  # type: ignore[assignment]

    if dist is not None and dist.is_available() and dist.is_initialized():
        rank = int(dist.get_rank())
        world_size = int(dist.get_world_size())

    last = int(getattr(owner, "_vllm_server_last_synced_step", -1))
    need_sync = int(step != last)

    if (
        dist is not None
        and dist.is_available()
        and dist.is_initialized()
        and int(world_size) > 1
    ):
        try:
            backend = str(dist.get_backend()).lower()
        except (TypeError, ValueError):
            backend = ""

        reduce_device = torch.device("cpu")
        if backend == "nccl" and torch is not None and torch.cuda.is_available():
            reduce_device = owner.model.device

        flag = torch.tensor([need_sync], device=reduce_device, dtype=torch.int32)
        dist.broadcast(flag, src=0)
        need_sync = int(flag.item())

    if need_sync == 0:
        return

    eff_mode = owner._effective_vllm_server_sync_mode()
    if eff_mode != "full":
        raise ValueError(
            "rollout_matching.vllm.sync.mode must be 'full' in this stack "
            "(adapter/auto sync modes are unsupported)."
        )

    if (
        dist is None
        or (not dist.is_available())
        or (not dist.is_initialized())
        or int(world_size) == 1
    ):
        client = owner._ensure_vllm_server_client()
        if not bool(getattr(owner, "_vllm_server_comm_inited", False)):
            try:
                client.init_communicator(device=0)
                setattr(owner, "_vllm_server_comm_inited", True)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Failed to initialize NCCL communicator with vLLM rollout server(s). "
                    "Mitigations: verify group_port reachability, set NCCL env, or increase vllm.server.timeout_s."
                ) from exc

        owner._sync_vllm_server_full_weights(client)
        owner._vllm_server_last_synced_step = step
        return

    assert dist is not None and dist.is_initialized()

    sync_failed = 0
    sync_err_msg = ""
    if int(rank) == 0:
        try:
            client = owner._ensure_vllm_server_client()
            owner._ensure_vllm_server_communicator_rank0(client)
            owner._sync_vllm_server_full_weights(client)
        except Exception as exc:
            sync_failed = 1
            sync_err_msg = f"{exc.__class__.__name__}: {exc}"

    try:
        try:
            backend = str(dist.get_backend()).lower()
        except Exception:
            backend = ""

        reduce_device = torch.device("cpu")
        if backend == "nccl" and torch is not None and torch.cuda.is_available():
            reduce_device = owner.model.device
    except Exception:
        reduce_device = torch.device("cpu")

    flag = torch.tensor([int(sync_failed)], device=reduce_device, dtype=torch.int32)
    dist.broadcast(flag, src=0)
    sync_failed = int(flag.item())

    msg_list = [sync_err_msg] if int(rank) == 0 else [""]
    try:
        dist.broadcast_object_list(msg_list, src=0, device=reduce_device)
    except TypeError:
        dist.broadcast_object_list(msg_list, src=0)
    sync_err_msg = str(msg_list[0])

    if int(sync_failed) != 0:
        raise RuntimeError(
            "vLLM server full weight sync failed on rank0 under DDP; aborting all ranks to avoid deadlocks. "
            f"Error: {sync_err_msg}"
        )

    owner._vllm_server_last_synced_step = step
