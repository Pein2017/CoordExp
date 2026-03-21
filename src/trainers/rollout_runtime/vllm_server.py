from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class PreparedVLLMServerRollout:
    decode_mode: str
    global_step: int
    infer_requests: List[Dict[str, Any]]
    servers: List[Dict[str, Any]]
    client: Any
    infer_timeout_s: Optional[float]
    base_request_config_dict: Dict[str, Any]
    effective_seed_base: int
    rollout_seed_base: int
    request_index_offset: int
    decode_batch_size_cap: int
    per_rank_chunk: int
    learner_world_size: int
    learner_rank: int
    server_world_sizes: List[int]
    per_server_rank_caps: List[int]
    round_cap_total: int
    seed_plan: List[Dict[str, Any]]


def build_vllm_server_infer_requests(
    *,
    samples: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Build JSON-serializable ms-swift RolloutInferRequest-compatible dicts."""

    infer_requests: List[Dict[str, Any]] = []
    for sample in samples:
        msgs = sample.get("messages")
        if not isinstance(msgs, list):
            raise ValueError("rollout-matching samples must contain messages (list)")
        try:
            msgs_json = json.loads(json.dumps(msgs))
        except Exception as exc:
            raise ValueError(
                "vLLM server mode requires JSON-serializable messages. "
                "Ensure images are passed as strings (path/url/base64), not PIL objects."
            ) from exc

        req: Dict[str, Any] = {"messages": msgs_json}

        images_raw = sample.get("images", None)
        if images_raw is None:
            image = sample.get("image", None)
            if isinstance(image, str) and image:
                images_raw = [image]
        if images_raw is not None:
            if isinstance(images_raw, str):
                images = [images_raw]
            elif isinstance(images_raw, (list, tuple)):
                images = list(images_raw)
            else:
                raise ValueError(
                    "vLLM server mode expects sample['images'] to be a string or list of strings"
                )
            if not all(isinstance(x, str) for x in images):
                raise ValueError(
                    "vLLM server mode expects all image entries to be strings (path/url/base64)"
                )
            req["images"] = images

        infer_requests.append(req)

    return infer_requests


def prepare_vllm_server_rollout(
    *,
    owner: Any,
    logger: Any,
    samples: Sequence[Mapping[str, Any]],
    request_index_offset: int,
    with_logprobs: bool,
    decode_override: Optional[Mapping[str, Any]],
    per_server_rank_request_caps_fn: Any,
    allocate_weighted_counts_with_caps_fn: Any,
) -> PreparedVLLMServerRollout:
    """Resolve server rollout config, capacity, and reproducibility metadata."""

    decode_request = owner._resolve_rollout_decode_request(
        decode_override=decode_override
    )
    decode_mode = str(decode_request.decode_mode)
    if decode_mode == "beam":
        raise ValueError(
            "vLLM server rollout backend does not support decode_mode=beam; "
            "use greedy or sampling overrides instead"
        )

    if not bool(getattr(owner, "_stage2_skip_vllm_server_sync", False)):
        owner._sync_vllm_server_rollout_model_if_needed()

    max_new_tokens = int(decode_request.max_new_tokens)
    temperature = float(decode_request.temperature)
    top_p = float(decode_request.top_p)
    top_k = int(decode_request.top_k)
    repetition_penalty = float(decode_request.repetition_penalty)

    if with_logprobs and float(temperature) > 0.0:
        raise ValueError(
            "eval-step confidence scoring requires decoding.temperature=0.0 "
            f"(greedy), got {float(temperature)}"
        )

    try:
        from swift.llm import RequestConfig
    except (ImportError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "swift.llm.RequestConfig is required for vLLM server rollouts"
        ) from exc

    base_request_kwargs = owner._rollout_vllm_request_config_kwargs(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    if with_logprobs:
        base_request_kwargs["logprobs"] = True
    base_request_config = RequestConfig(**base_request_kwargs)
    base_request_config_dict = asdict(base_request_config)

    global_step = int(getattr(getattr(owner, "state", None), "global_step", 0) or 0)
    rollout_seed_base = int(owner._derive_rollout_seed_base(global_step=global_step))
    request_index_offset_i = max(0, int(request_index_offset))
    effective_seed_base = int(rollout_seed_base + request_index_offset_i)

    infer_requests = build_vllm_server_infer_requests(samples=samples)

    servers = [dict(server) for server in owner._vllm_server_specs()]
    if not servers:
        raise ValueError("vLLM server mode requires a non-empty server list")

    _timeout_s, infer_timeout_s = owner._vllm_server_timeouts()

    client = owner._ensure_vllm_server_client()

    server_world_sizes = [int(x) for x in owner._vllm_server_world_sizes()]
    if len(server_world_sizes) != int(len(servers)):
        raise RuntimeError(
            "vLLM server world_size discovery returned unexpected length: "
            f"servers={int(len(servers))} world_sizes={server_world_sizes}"
        )

    learner_world = 1
    learner_rank = 0
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            learner_world = int(dist.get_world_size())
            learner_rank = int(dist.get_rank())
    except (TypeError, ValueError):
        learner_world = 1
        learner_rank = 0
    learner_world = max(1, int(learner_world))
    learner_rank = max(0, int(learner_rank))

    rollout_context = owner._current_rollout_context()
    decode_batch_size_cap = int(owner._decode_batch_size(context=rollout_context))
    per_rank_chunk = int(
        owner._rollout_decode_batch_size_per_rank(
            rollout_context=rollout_context,
        )
    )

    per_server_rank_caps = [
        int(x)
        for x in per_server_rank_request_caps_fn(
            per_rank_chunk_size=int(per_rank_chunk),
            server_world_sizes=server_world_sizes,
            learner_world_size=int(learner_world),
            learner_rank=int(learner_rank),
        )
    ]
    round_cap_total = int(sum(per_server_rank_caps))
    if int(round_cap_total) != int(per_rank_chunk):
        raise RuntimeError(
            "internal per-rank rollout cap mismatch: "
            f"per_rank_chunk={int(per_rank_chunk)} round_cap_total={int(round_cap_total)} "
            f"learner_rank={int(learner_rank)} learner_world_size={int(learner_world)} "
            f"server_world_sizes={server_world_sizes}"
        )

    seed_plan = build_vllm_server_seed_plan(
        owner=owner,
        servers=servers,
        infer_requests=infer_requests,
        effective_seed_base=int(effective_seed_base),
        per_server_rank_caps=per_server_rank_caps,
        round_cap_total=int(round_cap_total),
        allocate_weighted_counts_with_caps_fn=allocate_weighted_counts_with_caps_fn,
    )

    if global_step != int(getattr(owner, "_vllm_server_last_logged_step", -1)):
        logger.info(
            "vLLM server rollout metadata: servers=%s sync_mode=%s request_n=%s rollout_seed_base=%s request_index_offset=%s effective_seed_base=%s decode_batch_size_cap=%s per_rank_chunk=%s learner_world_size=%s learner_rank=%s server_world_sizes=%s per_server_rank_caps=%s round_cap_total=%s seed_plan=%s",
            servers,
            owner._effective_vllm_server_sync_mode(),
            int(len(infer_requests)),
            int(rollout_seed_base),
            int(request_index_offset_i),
            int(effective_seed_base),
            int(decode_batch_size_cap),
            int(per_rank_chunk),
            int(learner_world),
            int(learner_rank),
            server_world_sizes,
            per_server_rank_caps,
            int(round_cap_total),
            seed_plan,
        )
        owner._vllm_server_last_logged_step = int(global_step)

    return PreparedVLLMServerRollout(
        decode_mode=str(decode_mode),
        global_step=int(global_step),
        infer_requests=infer_requests,
        servers=servers,
        client=client,
        infer_timeout_s=infer_timeout_s,
        base_request_config_dict=base_request_config_dict,
        effective_seed_base=int(effective_seed_base),
        rollout_seed_base=int(rollout_seed_base),
        request_index_offset=int(request_index_offset_i),
        decode_batch_size_cap=int(decode_batch_size_cap),
        per_rank_chunk=int(per_rank_chunk),
        learner_world_size=int(learner_world),
        learner_rank=int(learner_rank),
        server_world_sizes=server_world_sizes,
        per_server_rank_caps=per_server_rank_caps,
        round_cap_total=int(round_cap_total),
        seed_plan=seed_plan,
    )


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


def dispatch_vllm_server_rounds(
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
    per_server_rank_caps: Sequence[int],
    round_cap_total: int,
    allocate_weighted_counts_with_caps_fn: Any,
) -> List[Any]:
    from concurrent.futures import ThreadPoolExecutor

    results: List[Any] = [None] * len(infer_requests)

    cursor = 0
    while cursor < int(len(infer_requests)):
        remaining = int(len(infer_requests) - cursor)
        round_budget = int(min(remaining, max(1, int(round_cap_total))))
        counts = allocate_weighted_counts_with_caps_fn(
            int(round_budget), list(int(x) for x in per_server_rank_caps)
        )

        round_slices: List[Tuple[int, int, int]] = []
        offset = int(cursor)
        for i, cnt in enumerate(counts):
            if int(cnt) <= 0:
                continue
            start = int(offset)
            end = int(offset + int(cnt))
            round_slices.append((int(i), int(start), int(end)))
            offset = int(end)

        if not round_slices:
            raise RuntimeError(
                "vLLM server rollout produced an empty dispatch round under non-empty workload: "
                f"cursor={int(cursor)} remaining={int(remaining)} per_server_rank_caps={list(int(x) for x in per_server_rank_caps)}"
            )

        with ThreadPoolExecutor(max_workers=int(len(round_slices))) as ex:
            futs = [
                ex.submit(
                    infer_on_vllm_server_slice,
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
                    server_idx=int(i),
                    start=int(start),
                    end=int(end),
                )
                for i, start, end in round_slices
            ]
            for f in futs:
                f.result()

        cursor = int(cursor + round_budget)

    out: List[Any] = []
    for r in results:
        if r is None:
            raise RuntimeError("vLLM server failed to produce outputs for all requests")
        out.append(r)

    return out


def build_vllm_server_seed_plan(
    *,
    owner: Any,
    servers: Sequence[Mapping[str, Any]],
    infer_requests: Sequence[Any],
    effective_seed_base: int,
    per_server_rank_caps: Sequence[int],
    round_cap_total: int,
    allocate_weighted_counts_with_caps_fn: Any,
) -> List[Dict[str, Any]]:
    seed_plan: List[Dict[str, Any]] = []
    if int(len(infer_requests)) <= 0 or int(round_cap_total) <= 0:
        return seed_plan

    cursor = 0
    round_idx = 0
    while cursor < int(len(infer_requests)):
        remaining = int(len(infer_requests) - cursor)
        round_budget = int(min(remaining, int(round_cap_total)))
        counts = allocate_weighted_counts_with_caps_fn(
            int(round_budget), list(int(x) for x in per_server_rank_caps)
        )
        offset = int(cursor)
        for i, cnt in enumerate(counts):
            if int(cnt) <= 0:
                continue
            start = int(offset)
            end = int(offset + int(cnt))
            seed_plan.append(
                {
                    "round": int(round_idx),
                    "server_idx": int(i),
                    "base_url": str(servers[i].get("base_url", "")),
                    "start": int(start),
                    "end": int(end),
                    "cap_for_rank": int(per_server_rank_caps[i]),
                    "seed": int(
                        owner._normalize_rollout_seed_int32(
                            int(effective_seed_base + int(start))
                        )
                    ),
                }
            )
            offset = int(end)
        cursor = int(cursor + round_budget)
        round_idx = int(round_idx + 1)

    return seed_plan
