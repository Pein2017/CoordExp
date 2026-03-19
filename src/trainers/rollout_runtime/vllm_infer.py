from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


def vllm_infer_tp_group(
    *,
    owner: Any,
    infer_requests: List[Dict[str, Any]],
    request_config: Any,
) -> List[Any]:
    engine = owner._ensure_vllm_engine()
    tp = int(owner._vllm_tp_size)

    vcfg = owner._cfg("vllm", None)
    infer_batch_size: Optional[int] = None
    if isinstance(vcfg, Mapping):
        raw = vcfg.get("infer_batch_size", None)
        if raw is not None:
            try:
                infer_batch_size = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "rollout_matching.vllm.infer_batch_size must be an int"
                ) from exc
            if infer_batch_size <= 0:
                infer_batch_size = None

    def _infer_batched(reqs: List[Dict[str, Any]]) -> List[Any]:
        if not reqs:
            return []
        if infer_batch_size is None or infer_batch_size >= len(reqs):
            return engine.infer(reqs, request_config=request_config, use_tqdm=False)
        outs: List[Any] = []
        for i in range(0, len(reqs), infer_batch_size):
            outs.extend(
                engine.infer(
                    reqs[i : i + infer_batch_size],
                    request_config=request_config,
                    use_tqdm=False,
                )
            )
        return outs

    if tp <= 1:
        return _infer_batched(infer_requests)

    import torch.distributed as dist

    group = owner._vllm_tp_group
    local_rank = int(dist.get_rank(group=group))
    local_len = int(len(infer_requests))
    all_lens: List[int] = [0 for _ in range(tp)]
    dist.all_gather_object(all_lens, local_len, group=group)
    start_idx = sum(int(x) for x in all_lens[:local_rank])
    end_idx = start_idx + local_len

    gathered: List[List[Dict[str, Any]]] = [[] for _ in range(tp)]
    dist.all_gather_object(gathered, infer_requests, group=group)
    flat: List[Dict[str, Any]] = [x for sub in gathered for x in sub]

    outs = _infer_batched(flat)
    return outs[start_idx:end_idx]
