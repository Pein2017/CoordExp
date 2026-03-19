from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


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


def vllm_server_update_state_dict(
    *,
    client: Any,
    state_dict: dict[str, Any],
) -> None:
    try:
        from swift.trainers.rlhf_trainer.utils import FlattenedTensorBucket
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "FlattenedTensorBucket is required for vLLM server sync"
        ) from exc

    bucket_size_mb = int(os.environ.get("SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE", 512))
    bucket_size_bytes = int(bucket_size_mb) * 1024 * 1024

    bucket: list[tuple[str, torch.Tensor]] = []
    bucket_bytes = 0

    def _flush_bucket() -> None:
        nonlocal bucket, bucket_bytes
        if not bucket:
            return
        b = FlattenedTensorBucket(named_tensors=bucket)
        client.update_flattened_params(b.get_metadata(), b.get_flattened_tensor())
        bucket = []
        bucket_bytes = 0

    for name, t in state_dict.items():
        if t is None or not isinstance(t, torch.Tensor):
            continue
        if t.numel() == 0:
            continue
        ten = t.detach()
        nbytes = int(ten.numel() * ten.element_size())
        if bucket and bucket_size_bytes > 0 and bucket_bytes + nbytes > bucket_size_bytes:
            _flush_bucket()
        bucket.append((str(name), ten))
        bucket_bytes += nbytes

    _flush_bucket()


def sync_vllm_server_full_weights(
    *,
    owner: Any,
    client: Any,
    logger: Any,
) -> None:
    from contextlib import nullcontext

    try:
        from accelerate.utils import is_peft_model
    except (TypeError, ValueError):
        is_peft_model = None  # type: ignore[assignment]

    is_peft = bool(is_peft_model(owner.model)) if is_peft_model is not None else False

    merge_cm = nullcontext()
    unmerge_cm = nullcontext()
    if is_peft:
        try:
            from swift.trainers.rlhf_trainer.utils import (
                patch_lora_merge,
                patch_lora_unmerge,
            )

            merge_cm = patch_lora_merge(owner.model)
            unmerge_cm = patch_lora_unmerge(owner.model)
        except (TypeError, ValueError):
            merge_cm = nullcontext()
            unmerge_cm = nullcontext()

    from swift.trainers.rlhf_trainer.utils import get_gather_if_zero3_context

    params = [p for _, p in owner.model.named_parameters()]
    gather_if_zero3 = get_gather_if_zero3_context(owner)

    with gather_if_zero3(params), merge_cm, torch.no_grad():
        merged = False
        try:
            if is_peft:
                try:
                    owner.model.merge_adapter()
                    merged = True
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "vLLM server full sync requires merging adapter weights from the training model. "
                        "Mitigations: ensure PEFT supports merge_adapter/unmerge_adapter (required for vLLM full sync in this stack), or switch rollout_matching.rollout_backend=hf."
                    ) from exc

            state_dict = owner.model.state_dict()
            if is_peft:
                prefix_removed = {
                    k.removeprefix("base_model.model."): v
                    for k, v in state_dict.items()
                }
                state_dict = {
                    k.replace(".base_layer", ""): v for k, v in prefix_removed.items()
                }
                prefix = getattr(owner.model, "prefix", None)
                if isinstance(prefix, str) and prefix:
                    state_dict = {
                        k: v for k, v in state_dict.items() if prefix not in k
                    }
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
                state_dict = {
                    k: v for k, v in state_dict.items() if "lora_" not in k
                }

            owner._vllm_server_update_state_dict(client, state_dict)
        finally:
            if is_peft and merged:
                with unmerge_cm:
                    owner.model.unmerge_adapter()

    try:
        client.reset_prefix_cache()
    except (TypeError, ValueError) as exc:
        logger.warning(
            "Failed to reset vLLM server prefix cache after full sync: %s", exc
        )


def infer_on_vllm_server_slice(
    *,
    owner: Any,
    logger: Any,
    client: Any,
    servers: Sequence[Mapping[str, Any]],
    infer_requests: Sequence[Any],
    base_request_config_dict: Mapping[str, Any],
    effective_seed_base: int,
    infer_timeout_s: Optional[float],
    with_logprobs: bool,
    decode_mode: str,
    results: List[Any],
    server_idx: int,
    start: int,
    end: int,
) -> None:
    if start >= end:
        return
    base_url = str(servers[server_idx]["base_url"]).rstrip("/")

    req_cfg = dict(base_request_config_dict)
    req_cfg["seed"] = int(
        owner._normalize_rollout_seed_int32(int(effective_seed_base + int(start)))
    )

    payload = {
        "infer_requests": infer_requests[start:end],
        "request_config": req_cfg,
        "metrics": None,
        "template": None,
        "use_tqdm": None,
        "adapter_request": None,
    }

    url = f"{base_url}/infer/"
    session = client.sessions[server_idx]
    req_timeout: Optional[Tuple[float, float]]
    if infer_timeout_s is None:
        req_timeout = None
    else:
        req_timeout_s = float(infer_timeout_s)
        req_timeout = (min(10.0, req_timeout_s), req_timeout_s)

    import requests

    request_errors: Tuple[type[BaseException], ...] = (
        requests.exceptions.RequestException,
        TypeError,
        ValueError,
    )
    try:
        with owner._vllm_server_infer_guard():
            resp = session.post(url, json=payload, timeout=req_timeout)
    except request_errors as exc:
        try:
            client.sessions[server_idx] = requests.Session()
            session = client.sessions[server_idx]
            with owner._vllm_server_infer_guard():
                resp = session.post(url, json=payload, timeout=req_timeout)
        except request_errors as exc2:
            if int(end - start) > 1:
                mid = int((start + end) // 2)
                logger.warning(
                    "vLLM server infer request failed; splitting batch: url=%s start=%s end=%s mid=%s exc=%r",
                    url,
                    int(start),
                    int(end),
                    int(mid),
                    exc2,
                )
                infer_on_vllm_server_slice(
                    owner=owner,
                    logger=logger,
                    client=client,
                    servers=servers,
                    infer_requests=infer_requests,
                    base_request_config_dict=base_request_config_dict,
                    effective_seed_base=int(effective_seed_base),
                    infer_timeout_s=infer_timeout_s,
                    with_logprobs=bool(with_logprobs),
                    decode_mode=str(decode_mode),
                    results=results,
                    server_idx=int(server_idx),
                    start=int(start),
                    end=int(mid),
                )
                infer_on_vllm_server_slice(
                    owner=owner,
                    logger=logger,
                    client=client,
                    servers=servers,
                    infer_requests=infer_requests,
                    base_request_config_dict=base_request_config_dict,
                    effective_seed_base=int(effective_seed_base),
                    infer_timeout_s=infer_timeout_s,
                    with_logprobs=bool(with_logprobs),
                    decode_mode=str(decode_mode),
                    results=results,
                    server_idx=int(server_idx),
                    start=int(mid),
                    end=int(end),
                )
                return

            raise RuntimeError(
                "vLLM server infer request failed after retry: "
                f"url={url} timeout={req_timeout!r} first_exc={exc!r} retry_exc={exc2!r}"
            ) from exc2

    if int(getattr(resp, "status_code", 0) or 0) != 200:
        raise RuntimeError(
            f"vLLM server infer failed: url={url} status={getattr(resp, 'status_code', None)} body={getattr(resp, 'text', '')}"
        )

    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError("vLLM server returned non-list JSON")
    if len(data) != int(end - start):
        raise RuntimeError(
            "vLLM server returned unexpected number of outputs: "
            f"expected={int(end - start)} got={len(data)}"
        )

    for j, raw_out in enumerate(data):
        idx = int(start + j)
        if with_logprobs:
            (
                token_ids,
                text,
                prompt_ids,
                token_logprobs,
                generated_token_text,
            ) = owner._parse_vllm_server_output_traced(raw_out, tokenizer=owner.tokenizer)
            results[idx] = (
                token_ids,
                text,
                decode_mode,
                prompt_ids,
                token_logprobs,
                generated_token_text,
            )
        else:
            token_ids, text, prompt_ids = owner._parse_vllm_server_output(raw_out)
            prompt_ids = owner._strip_left_padding_token_ids(
                prompt_ids,
                pad_token_id=getattr(owner.tokenizer, "pad_token_id", None),
            )
            results[idx] = (token_ids, text, decode_mode, prompt_ids)
