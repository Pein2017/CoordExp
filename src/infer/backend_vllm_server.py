from __future__ import annotations

"""ms-swift vLLM server rollout helpers owned by the inference runtime."""

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from src.infer.backend import (
    build_swift_request_config_from_decode_request,
    normalize_vllm_trace_response,
)
from src.infer.runtime import (
    build_decode_request_from_rollout_owner,
    current_rollout_context_from_owner,
    effective_rollout_backend_from_owner,
    rollout_decode_batch_size_from_owner,
    vllm_mode_from_rollout_owner,
)
from src.tokens.row_offsets import CoordOffsetAdapter

from src.infer.backend_sync import (
    COORDEXP_WORKER_EXTENSION_CLS,
    apply_coord_row_patch_for_vllm_client,
)


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


def _instance_override(owner: Any, name: str) -> Any:
    try:
        raw = getattr(owner, "__dict__", {}).get(name)
    except (AttributeError, TypeError):
        return None
    return raw if callable(raw) else None


def vllm_server_cfg(owner: Any) -> Mapping[str, Any]:
    override = _instance_override(owner, "_vllm_server_cfg")
    if override is not None:
        value = override()
        if not isinstance(value, Mapping):
            raise ValueError("rollout_matching.vllm.server must be a mapping")
        return value

    vcfg_raw = owner._cfg("vllm", {}) or {}
    if not isinstance(vcfg_raw, Mapping):
        raise ValueError("rollout_matching.vllm must be a mapping")
    scfg_raw = vcfg_raw.get("server", {}) or {}
    if not isinstance(scfg_raw, Mapping):
        raise ValueError("rollout_matching.vllm.server must be a mapping")
    return scfg_raw


def vllm_server_specs(owner: Any) -> List[Dict[str, Any]]:
    """Normalize server list config to a list of {base_url, group_port} dicts."""

    override = _instance_override(owner, "_vllm_server_specs")
    if override is not None:
        return [dict(server) for server in override()]

    scfg = vllm_server_cfg(owner)

    if "base_url" in scfg or "group_port" in scfg:
        raise ValueError(
            "Legacy rollout server config has been removed: "
            "rollout_matching.vllm.server.base_url/group_port. "
            "Use rollout_matching.vllm.server.servers[] (list of {base_url, group_port})."
        )

    servers_raw = scfg.get("servers", None)
    if not isinstance(servers_raw, list) or not servers_raw:
        raise ValueError("rollout_matching.vllm.server.servers must be a non-empty list")

    out: List[Dict[str, Any]] = []
    for i, server_raw in enumerate(servers_raw):
        if not isinstance(server_raw, Mapping):
            raise ValueError(
                "rollout_matching.vllm.server.servers[%d] must be a mapping"
                % int(i)
            )

        base_url = server_raw.get("base_url")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError(
                "rollout_matching.vllm.server.servers[%d].base_url must be a non-empty string"
                % int(i)
            )

        group_port_entry_raw = server_raw.get("group_port")
        try:
            group_port_entry = int(group_port_entry_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "rollout_matching.vllm.server.servers[%d].group_port must be an int"
                % int(i)
            ) from exc
        if group_port_entry <= 0:
            raise ValueError(
                "rollout_matching.vllm.server.servers[%d].group_port must be > 0"
                % int(i)
            )

        out.append(
            {
                "base_url": base_url.strip().rstrip("/"),
                "group_port": int(group_port_entry),
            }
        )

    return out


def vllm_server_timeouts(
    *,
    owner: Any,
    logger: Any,
) -> Tuple[float, Optional[float]]:
    override = _instance_override(owner, "_vllm_server_timeouts")
    if override is not None:
        timeout_s, infer_timeout_s = override()
        return float(timeout_s), (
            float(infer_timeout_s) if infer_timeout_s is not None else None
        )

    scfg = vllm_server_cfg(owner)

    timeout_raw = scfg.get("timeout_s", None)
    if timeout_raw is None:
        timeout_s = 240.0
    else:
        try:
            timeout_s = float(timeout_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "rollout_matching.vllm.server.timeout_s must be a float/int"
            ) from exc
    if timeout_s <= 0:
        raise ValueError("rollout_matching.vllm.server.timeout_s must be > 0")

    allow_infinite_infer_timeout = bool(
        scfg.get("allow_infinite_infer_timeout", False)
    )

    infer_timeout_raw = scfg.get("infer_timeout_s", None)
    if infer_timeout_raw is None:
        infer_timeout_s: Optional[float]
        if allow_infinite_infer_timeout:
            infer_timeout_s = None
        else:
            infer_timeout_s = float(timeout_s)
    else:
        try:
            infer_timeout_s = float(infer_timeout_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "rollout_matching.vllm.server.infer_timeout_s must be null or a float/int"
            ) from exc
        if infer_timeout_s <= 0:
            if allow_infinite_infer_timeout:
                infer_timeout_s = None
            else:
                raise ValueError(
                    "rollout_matching.vllm.server.infer_timeout_s must be > 0 unless "
                    "rollout_matching.vllm.server.allow_infinite_infer_timeout=true"
                )

    if infer_timeout_s is None and allow_infinite_infer_timeout:
        warned = bool(getattr(owner, "_vllm_server_infinite_timeout_warned", False))
        if not warned:
            logger.warning(
                "vLLM server infer timeout is unbounded because "
                "rollout_matching.vllm.server.allow_infinite_infer_timeout=true"
            )
            setattr(owner, "_vllm_server_infinite_timeout_warned", True)

    return float(timeout_s), (
        float(infer_timeout_s) if infer_timeout_s is not None else None
    )


def vllm_server_world_sizes(
    *,
    owner: Any,
    logger: Any,
) -> List[int]:
    """Return cached vLLM server data-parallel world sizes (one per server)."""

    override = _instance_override(owner, "_vllm_server_world_sizes")
    if override is not None:
        return [int(x) for x in override()]

    cached = getattr(owner, "_vllm_server_cached_world_sizes", None)
    if (
        isinstance(cached, list)
        and cached
        and all(isinstance(x, int) and x > 0 for x in cached)
    ):
        return list(int(x) for x in cached)

    servers = vllm_server_specs(owner)
    timeout_s, _infer_timeout_s = vllm_server_timeouts(owner=owner, logger=logger)

    import urllib.request as _urllib

    opener = _urllib.build_opener(_urllib.ProxyHandler({}))
    out: List[int] = []
    for server in servers:
        base_url = str(server["base_url"]).rstrip("/")
        url = f"{base_url}/get_world_size/"
        req = _urllib.Request(url, method="GET")
        with opener.open(req, timeout=float(timeout_s)) as resp:
            code = int(resp.getcode())
            body = resp.read()
        if code != 200:
            raise RuntimeError(
                f"vLLM rollout server /get_world_size/ returned HTTP {code}: {url}"
            )
        try:
            data = json.loads(body.decode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"vLLM rollout server /get_world_size/ returned non-JSON payload: {url}"
            ) from exc
        try:
            ws = int(data.get("world_size", 1)) if isinstance(data, dict) else 1
        except (TypeError, ValueError):
            ws = 1
        out.append(max(1, int(ws)))

    setattr(owner, "_vllm_server_cached_world_sizes", list(out))
    logger.info("vLLM rollout server world_size(s): %s", out)
    return list(out)


def effective_vllm_server_sync_mode(owner: Any) -> str:
    override = _instance_override(owner, "_effective_vllm_server_sync_mode")
    if override is not None:
        return str(override()).strip().lower()

    vcfg_raw = owner._cfg("vllm", {}) or {}
    if not isinstance(vcfg_raw, Mapping):
        raise ValueError("rollout_matching.vllm must be a mapping")
    sync_raw = vcfg_raw.get("sync", {}) or {}
    if not isinstance(sync_raw, Mapping):
        raise ValueError("rollout_matching.vllm.sync must be a mapping")

    mode = str(sync_raw.get("mode", "full") or "full").strip().lower()
    if mode != "adapter":
        raise ValueError(
            "CoordExp vLLM rollouts require official adapter sync: set "
            "rollout_matching.vllm.sync.mode=adapter."
        )
    return mode


def rollout_decode_batch_size_per_rank(
    *,
    owner: Any,
    rollout_backend: Optional[str] = None,
    rollout_context: str = "train",
    logger: Any,
) -> int:
    """Derived rollout request chunk size per learner rank."""

    context_norm = str(rollout_context).strip().lower()
    if context_norm not in {"train", "eval"}:
        raise ValueError("rollout_context must be one of {'train', 'eval'}")

    cap = int(
        rollout_decode_batch_size_from_owner(
            owner,
            context=("eval" if context_norm == "eval" else "train"),
        )
    )
    backend = (
        rollout_backend
        if rollout_backend is not None
        else effective_rollout_backend_from_owner(owner, context=context_norm)
    )
    if backend != "vllm":
        return max(1, int(cap))

    mode = str(vllm_mode_from_rollout_owner(owner)).strip().lower()
    if mode != "server":
        return max(1, int(cap))

    server_world_sizes = vllm_server_world_sizes(owner=owner, logger=logger)
    rollout_world = int(sum(int(x) for x in server_world_sizes))
    if rollout_world <= 0:
        rollout_world = 1

    learner_world = 1
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            learner_world = int(dist.get_world_size())
    except (TypeError, ValueError):
        learner_world = 1

    if learner_world <= 0:
        learner_world = 1

    if int(cap) * int(rollout_world) < int(learner_world):
        cap_key = (
            "rollout_matching.eval_decode_batch_size"
            if context_norm == "eval"
            else "rollout_matching.rollout_decode_batch_size"
        )
        raise ValueError(
            "rollout decode batch-size cap is infeasible for the current topology: "
            f"context={context_norm} {cap_key}={cap} rollout_world_size={rollout_world} learner_world_size={learner_world}. "
            "Increase rollout server DP world size, reduce learner world size, or increase the context-specific decode batch size."
        )

    per_rank = max(1, int(int(cap) * int(rollout_world) // int(learner_world)))

    meta = (
        int(cap),
        int(learner_world),
        tuple(int(x) for x in server_world_sizes),
        int(per_rank),
        str(context_norm),
    )
    if meta != getattr(owner, "_last_logged_rollout_decode_chunk_meta", None):
        logger.info(
            "Rollout decode batching (vLLM server): context=%s decode_batch_size_cap=%s learner_world_size=%s "
            "rollout_server_world_sizes=%s rollout_world_size=%s per_rank_chunk=%s total_chunk_across_ranks=%s",
            str(context_norm),
            int(cap),
            int(learner_world),
            list(int(x) for x in server_world_sizes),
            int(rollout_world),
            int(per_rank),
            int(per_rank) * int(learner_world),
        )
        setattr(owner, "_last_logged_rollout_decode_chunk_meta", meta)

    return int(per_rank)


def contiguous_chunk_slices(n: int, num_chunks: int) -> List[Tuple[int, int]]:
    """Deterministically slice ``range(n)`` into contiguous chunks."""

    if num_chunks <= 0:
        raise ValueError("num_chunks must be > 0")
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return [(0, 0) for _ in range(int(num_chunks))]

    chunk_size = int((n + num_chunks - 1) // num_chunks)
    out: List[Tuple[int, int]] = []
    for i in range(int(num_chunks)):
        start = min(int(i * chunk_size), int(n))
        end = min(int((i + 1) * chunk_size), int(n))
        if end < start:
            end = start
        out.append((start, end))
    return out


def contiguous_weighted_chunk_slices(
    n: int, weights: Sequence[int]
) -> List[Tuple[int, int]]:
    """Deterministically slice ``range(n)`` into weighted contiguous chunks."""

    if n < 0:
        raise ValueError("n must be >= 0")
    if not isinstance(weights, (list, tuple)) or not weights:
        raise ValueError("weights must be a non-empty list")

    ws: List[int] = []
    for i, raw_weight in enumerate(weights):
        try:
            weight = int(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"weights[{int(i)}] must be an int") from exc
        if weight < 0:
            raise ValueError(f"weights[{int(i)}] must be >= 0")
        ws.append(int(weight))

    total = int(sum(ws))
    if total <= 0:
        return contiguous_chunk_slices(int(n), int(len(ws)))
    if n == 0:
        return [(0, 0) for _ in range(int(len(ws)))]

    base_counts: List[int] = [int((int(n) * int(w)) // total) for w in ws]
    remainder = int(n) - int(sum(base_counts))
    if remainder < 0:
        remainder = 0

    frac_rank: List[Tuple[int, int]] = [
        (int((int(n) * int(w)) % total), int(i)) for i, w in enumerate(ws)
    ]
    frac_rank.sort(key=lambda item: (-int(item[0]), int(item[1])))
    for k in range(int(remainder)):
        _frac, idx = frac_rank[int(k % len(frac_rank))]
        base_counts[int(idx)] += 1

    out: List[Tuple[int, int]] = []
    start = 0
    for count in base_counts:
        end = int(start + int(count))
        out.append((int(start), int(end)))
        start = end

    if out and int(out[-1][1]) != int(n):
        raise RuntimeError(
            "weighted chunking produced invalid slices: "
            f"n={int(n)} weights={ws} slices={out}"
        )
    return out


def per_server_rank_request_caps(
    *,
    per_rank_chunk_size: int,
    server_world_sizes: Sequence[int],
    learner_world_size: int,
    learner_rank: int,
) -> List[int]:
    """Compute strict per-server request caps for one learner rank."""

    chunk = int(max(0, int(per_rank_chunk_size)))
    world = int(max(1, int(learner_world_size)))
    rank = int(max(0, int(learner_rank)))
    if world > 0:
        rank = min(rank, world - 1)

    weights = [int(max(1, int(x))) for x in server_world_sizes]
    if not weights:
        return []
    if chunk == 0:
        return [0 for _ in weights]

    global_budget = int(world * chunk)
    server_slices = contiguous_weighted_chunk_slices(int(global_budget), weights)
    rank_start = int(rank * chunk)
    rank_end = int(rank_start + chunk)

    out: List[int] = []
    for start, end in server_slices:
        overlap = max(0, min(int(rank_end), int(end)) - max(int(rank_start), int(start)))
        out.append(int(overlap))

    if int(sum(out)) != int(chunk):
        raise RuntimeError(
            "invalid per-server rank cap allocation: "
            f"chunk={int(chunk)} world={int(world)} rank={int(rank)} "
            f"server_world_sizes={weights} caps={out}"
        )
    return out


def allocate_weighted_counts_with_caps(n: int, caps: Sequence[int]) -> List[int]:
    """Allocate ``n`` contiguous requests across servers with strict caps."""

    n_i = int(n)
    if n_i < 0:
        raise ValueError("n must be >= 0")
    caps_i: List[int] = []
    for idx, raw_cap in enumerate(caps):
        try:
            cap = int(raw_cap)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"caps[{int(idx)}] must be an int") from exc
        if cap < 0:
            raise ValueError(f"caps[{int(idx)}] must be >= 0")
        caps_i.append(int(cap))

    total_cap = int(sum(caps_i))
    if n_i > total_cap:
        raise ValueError(
            "requested batch exceeds strict per-server cap budget: "
            f"n={n_i} total_cap={total_cap} caps={caps_i}"
        )
    if n_i == 0 or total_cap == 0:
        return [0 for _ in caps_i]

    positive = [(int(i), int(c)) for i, c in enumerate(caps_i) if int(c) > 0]
    if not positive:
        return [0 for _ in caps_i]

    pos_idx = [int(i) for i, _cap in positive]
    pos_caps = [int(cap) for _i, cap in positive]
    chunks = contiguous_weighted_chunk_slices(int(n_i), pos_caps)
    out = [0 for _ in caps_i]
    for local_i, (start, end) in enumerate(chunks):
        idx = int(pos_idx[local_i])
        count = int(end - start)
        if count < 0 or count > int(caps_i[idx]):
            raise RuntimeError(
                "invalid weighted capped allocation: "
                f"idx={idx} cnt={count} cap={caps_i[idx]} n={n_i} caps={caps_i}"
            )
        out[idx] = int(count)

    if int(sum(out)) != int(n_i):
        raise RuntimeError(
            "invalid weighted capped allocation total: "
            f"sum={int(sum(out))} n={int(n_i)} caps={caps_i}"
        )
    return out


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
    per_server_rank_request_caps_fn: Any | None = None,
    allocate_weighted_counts_with_caps_fn: Any | None = None,
) -> PreparedVLLMServerRollout:
    """Resolve server rollout config, capacity, and reproducibility metadata."""

    decode_request = build_decode_request_from_rollout_owner(
        owner,
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

    temperature = float(decode_request.temperature)

    if with_logprobs and float(temperature) > 0.0:
        raise ValueError(
            "eval-step confidence scoring requires decoding.temperature=0.0 "
            f"(greedy), got {float(temperature)}"
        )

    base_request_config = build_swift_request_config_from_decode_request(
        decode_request,
        trace_logprobs=bool(with_logprobs),
    )
    base_request_config_dict = asdict(base_request_config)

    global_step = int(getattr(getattr(owner, "state", None), "global_step", 0) or 0)
    rollout_seed_base = int(owner._derive_rollout_seed_base(global_step=global_step))
    request_index_offset_i = max(0, int(request_index_offset))
    effective_seed_base = int(rollout_seed_base + request_index_offset_i)

    infer_requests = build_vllm_server_infer_requests(samples=samples)

    servers = [dict(server) for server in vllm_server_specs(owner)]
    if not servers:
        raise ValueError("vLLM server mode requires a non-empty server list")

    _timeout_s, infer_timeout_s = vllm_server_timeouts(owner=owner, logger=logger)

    client = owner._ensure_vllm_server_client()

    server_world_sizes = [
        int(x) for x in vllm_server_world_sizes(owner=owner, logger=logger)
    ]
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

    rollout_context = current_rollout_context_from_owner(owner)
    decode_batch_size_cap = int(
        rollout_decode_batch_size_from_owner(owner, context=rollout_context)
    )
    per_rank_chunk = int(
        rollout_decode_batch_size_per_rank(
            owner=owner,
            rollout_context=rollout_context,
            logger=logger,
        )
    )

    caps_fn = per_server_rank_request_caps_fn or per_server_rank_request_caps
    alloc_fn = allocate_weighted_counts_with_caps_fn or allocate_weighted_counts_with_caps

    per_server_rank_caps = [
        int(x)
        for x in caps_fn(
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
        allocate_weighted_counts_with_caps_fn=alloc_fn,
    )

    if global_step != int(getattr(owner, "_vllm_server_last_logged_step", -1)):
        logger.info(
            "vLLM server rollout metadata: servers=%s sync_mode=%s request_n=%s rollout_seed_base=%s request_index_offset=%s effective_seed_base=%s decode_batch_size_cap=%s per_rank_chunk=%s learner_world_size=%s learner_rank=%s server_world_sizes=%s per_server_rank_caps=%s round_cap_total=%s seed_plan=%s",
            servers,
            effective_vllm_server_sync_mode(owner),
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

        servers = vllm_server_specs(owner)
        timeout_s, _infer_timeout_s = vllm_server_timeouts(
            owner=owner,
            logger=logger,
        )

        try:
            VLLMClient = apply_coord_row_patch_for_vllm_client()
        except Exception as exc:
            raise RuntimeError(
                "vLLM server mode requires ms-swift's VLLMClient (and vLLM + pynccl). "
                "Install/enable vLLM in the ms env, or switch to rollout_backend=hf."
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

    eff_mode = effective_vllm_server_sync_mode(owner)
    if eff_mode != "adapter":
        raise ValueError(
            "CoordExp vLLM rollouts now require official adapter sync: "
            "set rollout_matching.vllm.sync.mode=adapter and "
            "rollout_matching.vllm.enable_lora=true."
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

        owner._sync_vllm_server_adapter(client)
        owner._vllm_server_last_synced_step = step
        return

    assert dist is not None and dist.is_initialized()

    sync_failed = 0
    sync_err_msg = ""
    if int(rank) == 0:
        try:
            client = owner._ensure_vllm_server_client()
            owner._ensure_vllm_server_communicator_rank0(client)
            owner._sync_vllm_server_adapter(client)
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
            "vLLM server adapter sync failed on rank0 under DDP; aborting all ranks to avoid deadlocks. "
            f"Error: {sync_err_msg}"
        )

    owner._vllm_server_last_synced_step = step


def _import_swift_rollout_utils() -> Any:
    try:
        from swift.rlhf_trainers import utils as rollout_utils
    except (ImportError, TypeError, ValueError):
        try:
            from swift.trainers.rlhf_trainer import utils as rollout_utils
        except (ImportError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Adapter-only vLLM sync requires ms-swift rollout utility helpers."
            ) from exc
    return rollout_utils


def _peft_config_to_dict(peft_config: Any) -> dict[str, Any]:
    if hasattr(peft_config, "to_dict"):
        return dict(peft_config.to_dict())
    if hasattr(peft_config, "model_dump"):
        return dict(peft_config.model_dump())
    if hasattr(peft_config, "dict"):
        return dict(peft_config.dict())
    if isinstance(peft_config, Mapping):
        return dict(peft_config)
    raise RuntimeError(
        "Adapter-only vLLM sync requires a serializable PEFT LoRA config."
    )


def _vllm_adapter_peft_config(peft_config: Any) -> tuple[dict[str, Any], tuple[str, ...]]:
    payload = _peft_config_to_dict(peft_config)
    modules_raw = payload.get("modules_to_save") or []
    modules_to_save = (
        tuple(str(item) for item in modules_raw)
        if isinstance(modules_raw, (list, tuple, set))
        else ()
    )
    if modules_to_save:
        # vLLM's in-memory adapter endpoint only consumes LoRA tensors. Extra
        # PEFT modules such as CoordExp's coord_offset_adapter stay on the
        # learner/checkpoint path and must not be declared to the vLLM LoRA loader.
        payload["modules_to_save"] = None
    return payload, modules_to_save


def _filter_vllm_adapter_lora_tensors(
    lora_params: "OrderedDict[str, torch.Tensor]",
) -> tuple["OrderedDict[str, torch.Tensor]", tuple[str, ...]]:
    kept: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    dropped: list[str] = []
    for name, tensor in lora_params.items():
        name_s = str(name)
        if "modules_to_save." in name_s or "coord_offset_adapter" in name_s:
            dropped.append(name_s)
            continue
        if "lora_" in name_s or "lora_magnitude_vector" in name_s:
            kept[name_s] = tensor
            continue
        dropped.append(name_s)
    return kept, tuple(dropped)


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _tensor_payload_digest(tensor: torch.Tensor) -> dict[str, Any]:
    tensor_cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes(
        {
            "dtype": str(tensor_cpu.dtype),
            "shape": [int(dim) for dim in tensor_cpu.shape],
            "numel": int(tensor_cpu.numel()),
        }
    ))
    try:
        raw = tensor_cpu.numpy().tobytes()
    except TypeError:
        raw = tensor_cpu.to(dtype=torch.float32).numpy().tobytes()
    digest.update(raw)
    return {
        "dtype": str(tensor_cpu.dtype),
        "shape": [int(dim) for dim in tensor_cpu.shape],
        "numel": int(tensor_cpu.numel()),
        "digest": f"sha256:{digest.hexdigest()}",
    }


def _named_tensor_payload_digest(
    named_tensors: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    tensor_payloads = {
        str(name): _tensor_payload_digest(tensor)
        for name, tensor in named_tensors.items()
    }
    digest = hashlib.sha256(_canonical_json_bytes(tensor_payloads)).hexdigest()
    return {
        "tensor_count": int(len(tensor_payloads)),
        "tensor_names": sorted(tensor_payloads.keys()),
        "tensors": tensor_payloads,
        "digest": f"sha256:{digest}",
    }


def _build_vllm_adapter_sync_provenance(
    *,
    sync_mode: str,
    sync_step: int | None = None,
    sync_frequency: str = "per_global_step",
    lora_params: Mapping[str, torch.Tensor],
    vllm_peft_config: Mapping[str, Any],
    dropped_modules_to_save: Sequence[str],
    dropped_param_names: Sequence[str],
    coord_ids: torch.Tensor | None,
    embed_offset: torch.Tensor | None,
    head_offset: torch.Tensor | None,
    tie_head: bool,
    coord_row_status: str,
    worker_verified: bool,
    worker_verified_status: str,
    rank_symmetric_failure: bool,
) -> dict[str, Any]:
    coord_tensors: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    if coord_ids is not None:
        coord_tensors["coord_ids"] = coord_ids
    if embed_offset is not None:
        coord_tensors["embed_offset"] = embed_offset
    if head_offset is not None:
        coord_tensors["head_offset"] = head_offset
    coord_digest = (
        _named_tensor_payload_digest(coord_tensors)
        if coord_tensors
        else {
            "tensor_count": 0,
            "tensor_names": [],
            "tensors": {},
            "digest": None,
        }
    )
    return {
        "schema_version": "coordexp_vllm_adapter_sync_v1",
        "sync_policy": {
            "mode": str(sync_mode),
            "frequency": str(sync_frequency),
            "global_step": int(sync_step) if sync_step is not None else None,
            "adapter_endpoint": "update_adapter_flattened_param",
            "coord_row_endpoint": "update_token_row_offsets",
        },
        "server_identity": {
            "sync_schema": "coordexp_vllm_adapter_sync_v1",
            "coord_row_api": "coordexp_token_row_offsets_v1",
            "client_patch": "coordexp_vllm_client_token_row_offsets_v1",
            "worker_extension_cls": COORDEXP_WORKER_EXTENSION_CLS,
        },
        "lora": {
            **_named_tensor_payload_digest(lora_params),
            "peft_config": dict(vllm_peft_config),
            "dropped_modules_to_save": [str(item) for item in dropped_modules_to_save],
            "dropped_param_names": [str(item) for item in dropped_param_names],
        },
        "coord_rows": {
            **coord_digest,
            "status": str(coord_row_status),
            "tie_head": bool(tie_head),
        },
        "requested": {
            "adapter_update": True,
            "coord_row_update": str(coord_row_status) == "requested",
        },
        "worker_verified": {
            "verified": bool(worker_verified),
            "status": str(worker_verified_status),
        },
        "rank_symmetric_failure": bool(rank_symmetric_failure),
    }


_UNRESOLVED_ACTIVE_ADAPTER = object()


def _find_active_coord_offset_adapter(model: Any) -> CoordOffsetAdapter | None:
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return None

    direct: CoordOffsetAdapter | None = None
    for name, module in named_modules():
        wrapped = _active_modules_to_save_coord_adapter(module)
        if wrapped is _UNRESOLVED_ACTIVE_ADAPTER:
            return None
        if isinstance(wrapped, CoordOffsetAdapter):
            return wrapped
        if (
            direct is None
            and isinstance(module, CoordOffsetAdapter)
            and not _is_wrapper_internal_module_name(name)
        ):
            direct = module
    return direct


def _active_modules_to_save_coord_adapter(
    module: Any,
) -> CoordOffsetAdapter | object | None:
    modules_to_save = getattr(module, "modules_to_save", None)
    if not _looks_like_module_mapping(modules_to_save):
        return None

    active_adapters = _active_adapter_names(module)
    if active_adapters is not None:
        for name in active_adapters:
            if name in modules_to_save and isinstance(
                modules_to_save[name], CoordOffsetAdapter
            ):
                return modules_to_save[name]
        return _UNRESOLVED_ACTIVE_ADAPTER

    for candidate in modules_to_save.values():
        if isinstance(candidate, CoordOffsetAdapter):
            return candidate
    return None


def _active_adapter_names(module: Any) -> list[str] | None:
    if hasattr(module, "active_adapters"):
        active_adapters = getattr(module, "active_adapters")
    elif hasattr(module, "active_adapter"):
        active_adapters = getattr(module, "active_adapter")
    else:
        return None

    if isinstance(active_adapters, str):
        return [active_adapters]
    return list(active_adapters or [])


def _is_wrapper_internal_module_name(name: str) -> bool:
    return any(
        component in {"original_module", "modules_to_save"}
        for component in str(name).split(".")
    )


def _looks_like_module_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) or (
        hasattr(value, "__contains__")
        and hasattr(value, "__getitem__")
        and hasattr(value, "values")
    )


def _validate_coord_offset_adapter_for_vllm_sync(
    adapter: CoordOffsetAdapter,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, bool]:
    coord_ids = getattr(adapter, "coord_ids", None)
    embed_offset = getattr(adapter, "embed_offset", None)
    head_offset = getattr(adapter, "head_offset", None)
    tie_head = bool(getattr(adapter, "tie_head", True))

    if not torch.is_tensor(coord_ids) or coord_ids.ndim != 1 or coord_ids.numel() == 0:
        raise RuntimeError(
            "coord_offset_adapter vLLM sync requires non-empty 1D coord_ids."
        )
    if (
        not torch.is_tensor(embed_offset)
        or embed_offset.ndim != 2
        or embed_offset.size(0) != coord_ids.numel()
    ):
        raise RuntimeError(
            "coord_offset_adapter vLLM sync requires embed_offset with shape "
            "[num_coord_ids, hidden]."
        )
    if not tie_head and (
        not torch.is_tensor(head_offset)
        or head_offset.ndim != 2
        or head_offset.size(0) != coord_ids.numel()
    ):
        raise RuntimeError(
            "untied coord_offset_adapter vLLM sync requires head_offset with "
            "shape [num_coord_ids, hidden]."
        )
    return (
        coord_ids.detach().to(dtype=torch.long).contiguous(),
        embed_offset.detach().contiguous(),
        head_offset.detach().contiguous() if torch.is_tensor(head_offset) else None,
        tie_head,
    )


def _sync_vllm_server_coord_offset_adapter(
    *,
    owner: Any,
    client: Any,
    logger: Any,
    dropped_modules_to_save: tuple[str, ...],
    dropped_param_names: tuple[str, ...],
) -> dict[str, Any]:
    has_declared_coord = "coord_offset_adapter" in set(dropped_modules_to_save) or any(
        "coord_offset_adapter" in name for name in dropped_param_names
    )
    adapter = _find_active_coord_offset_adapter(owner.model)
    if adapter is None:
        if has_declared_coord:
            raise RuntimeError(
                "vLLM adapter sync detected coord_offset_adapter in the PEFT "
                "payload, but could not find an active CoordOffsetAdapter on "
                "the learner model. Refusing to run rollouts with missing "
                "token-row offsets."
            )
        return {
            "status": "not_present",
            "coord_ids": None,
            "embed_offset": None,
            "head_offset": None,
            "tie_head": True,
        }

    update_fn = getattr(client, "update_token_row_offsets", None)
    if not callable(update_fn):
        raise RuntimeError(
            "vLLM adapter sync requires ms-swift VLLMClient.update_token_row_offsets "
            "when the learner has coord_offset_adapter. Update /data/ms-swift or "
            "disable vLLM server rollouts for this checkpoint."
        )

    coord_ids, embed_offset, head_offset, tie_head = (
        _validate_coord_offset_adapter_for_vllm_sync(adapter)
    )
    update_fn(
        coord_ids.to(device=embed_offset.device),
        embed_offset,
        head_offset=head_offset,
        tie_head=tie_head,
    )
    logger.info(
        "vLLM adapter sync updated coord_offset_adapter token rows: rows=%s "
        "tie_head=%s embed_shape=%s head_shape=%s",
        int(coord_ids.numel()),
        bool(tie_head),
        tuple(embed_offset.shape),
        tuple(head_offset.shape) if head_offset is not None else None,
    )
    return {
        "status": "requested",
        "coord_ids": coord_ids,
        "embed_offset": embed_offset,
        "head_offset": head_offset,
        "tie_head": bool(tie_head),
    }


def sync_vllm_server_adapter(
    *,
    owner: Any,
    client: Any,
    logger: Any,
) -> None:
    try:
        from accelerate.utils import is_peft_model
    except (ImportError, TypeError, ValueError):
        is_peft_model = None  # type: ignore[assignment]

    is_peft = bool(is_peft_model(owner.model)) if is_peft_model is not None else False
    if not is_peft:
        raise RuntimeError(
            "Adapter-only vLLM sync requires a PEFT/Swift LoRA-wrapped learner model."
        )

    try:
        from peft.utils.save_and_load import get_peft_model_state_dict
    except (ImportError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Adapter-only vLLM sync requires peft.utils.save_and_load.get_peft_model_state_dict."
        ) from exc

    rollout_utils = _import_swift_rollout_utils()
    gather_if_zero3 = rollout_utils.get_gather_if_zero3_context(owner)
    patch_lora_merge = rollout_utils.patch_lora_merge
    patch_lora_unmerge = rollout_utils.patch_lora_unmerge
    FlattenedTensorBucket = rollout_utils.FlattenedTensorBucket

    peft_config = getattr(owner.model, "peft_config", {}).get("default", None)
    if peft_config is None:
        raise RuntimeError(
            "Adapter-only vLLM sync could not find owner.model.peft_config['default']."
        )

    params = [p for _, p in owner.model.named_parameters()]

    with gather_if_zero3(params), patch_lora_merge(owner.model), torch.no_grad():
        merged = False
        try:
            try:
                owner.model.merge_adapter()
                merged = True
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Adapter-only vLLM sync requires PEFT merge_adapter/unmerge_adapter "
                    "to extract the current LoRA tensors for ms-swift's in-memory adapter update."
                ) from exc

            named_state = OrderedDict(owner.model.named_parameters())
            for name, buffer in owner.model.named_buffers():
                named_state.setdefault(str(name), buffer)
            lora_params = get_peft_model_state_dict(owner.model, named_state)
            lora_params = OrderedDict(
                (
                    str(name),
                    param.full_tensor().detach()
                    if hasattr(param, "full_tensor")
                    else param.detach(),
                )
                for name, param in lora_params.items()
                if torch.is_tensor(param)
            )
        finally:
            if merged:
                with patch_lora_unmerge(owner.model):
                    owner.model.unmerge_adapter()

    if not lora_params:
        raise RuntimeError(
            "Adapter-only vLLM sync collected no LoRA tensors. "
            "Check tuner.train_type/lora_rank/target_modules and PEFT wrapping."
        )
    lora_params, dropped_param_names = _filter_vllm_adapter_lora_tensors(lora_params)
    vllm_peft_config, dropped_modules_to_save = _vllm_adapter_peft_config(peft_config)
    if not lora_params:
        raise RuntimeError(
            "Adapter-only vLLM sync has no vLLM-compatible LoRA tensors after "
            f"filtering unsupported modules_to_save tensors: {list(dropped_param_names)}"
        )
    if dropped_param_names or dropped_modules_to_save:
        logger.info(
            "vLLM adapter sync filtered unsupported PEFT modules_to_save payload: "
            "modules_to_save=%s dropped_tensors=%s",
            list(dropped_modules_to_save),
            list(dropped_param_names),
        )

    bucket = FlattenedTensorBucket(named_tensors=list(lora_params.items()))
    client.update_adapter_flattened_param(
        vllm_peft_config,
        bucket.get_metadata(),
        bucket.get_flattened_tensor(),
    )
    coord_row_provenance = _sync_vllm_server_coord_offset_adapter(
        owner=owner,
        client=client,
        logger=logger,
        dropped_modules_to_save=dropped_modules_to_save,
        dropped_param_names=dropped_param_names,
    )
    sync_step = int(getattr(getattr(owner, "state", None), "global_step", 0) or 0)
    sync_provenance = _build_vllm_adapter_sync_provenance(
        sync_mode="adapter",
        sync_step=sync_step,
        sync_frequency="per_global_step",
        lora_params=lora_params,
        vllm_peft_config=vllm_peft_config,
        dropped_modules_to_save=dropped_modules_to_save,
        dropped_param_names=dropped_param_names,
        coord_ids=coord_row_provenance.get("coord_ids"),
        embed_offset=coord_row_provenance.get("embed_offset"),
        head_offset=coord_row_provenance.get("head_offset"),
        tie_head=bool(coord_row_provenance.get("tie_head", True)),
        coord_row_status=str(coord_row_provenance.get("status", "unknown")),
        worker_verified=False,
        worker_verified_status="unavailable_fire_and_forget",
        rank_symmetric_failure=True,
    )
    owner._vllm_server_last_backend_sync_identity = sync_provenance
    owner._vllm_server_last_sync_provenance = sync_provenance
    logger.info(
        "synced vLLM LoRA adapter via official ms-swift endpoint: tensors=%s bytes=%s",
        int(len(lora_params)),
        int(bucket.get_flattened_tensor().numel()),
    )

    try:
        client.reset_prefix_cache()
        reset_mm_cache = getattr(client, "reset_mm_cache", None)
        if callable(reset_mm_cache):
            reset_mm_cache()
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.warning(
            "Failed to reset vLLM server caches after adapter sync: %s", exc
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
        result = normalize_vllm_trace_response(
            raw_out,
            trace_logprobs=bool(with_logprobs),
            backend_mode="ms-swift",
            tokenizer=owner.tokenizer,
        )
        token_ids = [int(t) for t in (result.generated_token_ids or [])]
        text = str(result.text or "")
        prompt_ids = [int(t) for t in (result.prompt_token_ids or [])]
        if with_logprobs:
            results[idx] = (
                token_ids,
                text,
                decode_mode,
                prompt_ids,
                [float(t) for t in (result.generated_logprobs or [])],
                [str(t) for t in (result.generated_tokens or [])],
            )
        else:
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


def rollout_many_vllm_server(
    *,
    owner: Any,
    logger: Any,
    samples: Sequence[Mapping[str, Any]],
    debug_samples: Optional[Sequence[Mapping[str, Any]]] = None,
    request_index_offset: int = 0,
    with_logprobs: bool = False,
    decode_override: Optional[Mapping[str, Any]] = None,
    per_server_rank_request_caps_fn: Any | None = None,
    allocate_weighted_counts_with_caps_fn: Any | None = None,
) -> List[Any]:
    """vLLM server rollout backend entrypoint owned by the inference runtime."""

    if int(len(samples)) == 0:
        return []

    caps_fn = per_server_rank_request_caps_fn or per_server_rank_request_caps
    alloc_fn = allocate_weighted_counts_with_caps_fn or allocate_weighted_counts_with_caps

    prepared = prepare_vllm_server_rollout(
        owner=owner,
        logger=logger,
        samples=samples,
        request_index_offset=int(request_index_offset),
        with_logprobs=bool(with_logprobs),
        decode_override=decode_override,
        per_server_rank_request_caps_fn=caps_fn,
        allocate_weighted_counts_with_caps_fn=alloc_fn,
    )

    out = dispatch_vllm_server_rounds(
        owner=owner,
        logger=logger,
        client=prepared.client,
        servers=prepared.servers,
        infer_requests=prepared.infer_requests,
        base_request_config_dict=prepared.base_request_config_dict,
        effective_seed_base=int(prepared.effective_seed_base),
        infer_timeout_s=prepared.infer_timeout_s,
        with_logprobs=bool(with_logprobs),
        decode_mode=str(prepared.decode_mode),
        per_server_rank_caps=prepared.per_server_rank_caps,
        round_cap_total=int(prepared.round_cap_total),
        allocate_weighted_counts_with_caps_fn=alloc_fn,
    )
    if len(out) != len(samples):
        raise RuntimeError("vLLM server returned unexpected number of outputs")
    dump_fn = getattr(owner, "_maybe_debug_dump_vllm_server_rollouts", None)
    if callable(dump_fn):
        dump_fn(
            global_step=prepared.global_step,
            seed_base=prepared.effective_seed_base,
            infer_requests=prepared.infer_requests,
            outputs=out,
            samples=debug_samples if debug_samples is not None else samples,
        )
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
