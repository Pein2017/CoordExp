from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from pydantic import BaseModel

try:
    from swift.pipelines.infer.rollout import (
        WeightSyncWorkerExtension as _SwiftWeightSyncWorkerExtension,
    )
except Exception:
    _SwiftWeightSyncWorkerExtension = object


COORDEXP_WORKER_EXTENSION_CLS = (
    "src.infer.backend_sync.CoordExpWeightSyncWorkerExtension"
)


class _TensorRowOffsetMetadata(BaseModel):
    name: str | None = None
    dtype: str
    shape: tuple[int, ...]
    numel: int


class _UpdateTokenRowOffsetsRequest(BaseModel):
    coord_ids: _TensorRowOffsetMetadata
    embed_offset: _TensorRowOffsetMetadata
    head_offset: _TensorRowOffsetMetadata | None = None
    tie_head: bool = True
    embed_key: str = "language_model.model.embed_tokens.weight"
    head_key: str = "language_model.lm_head.weight"


class CoordExpWeightSyncWorkerExtension(_SwiftWeightSyncWorkerExtension):
    _token_row_offset_base_rows = None

    def update_token_row_offsets(
        self,
        coord_ids_metadata: dict[str, Any],
        embed_offset_metadata: dict[str, Any],
        head_offset_metadata: dict[str, Any] | None,
        tie_head: bool = True,
        embed_key: str = "language_model.model.embed_tokens.weight",
        head_key: str = "language_model.lm_head.weight",
    ) -> None:
        return _worker_update_token_row_offsets(
            self,
            coord_ids_metadata,
            embed_offset_metadata,
            head_offset_metadata,
            tie_head,
            embed_key,
            head_key,
        )

    def _apply_token_row_offsets_to_param(
        self,
        key: str,
        param: torch.nn.Parameter,
        coord_ids: torch.Tensor,
        offset: torch.Tensor,
    ) -> None:
        return _worker_apply_token_row_offsets_to_param(
            self, key, param, coord_ids, offset
        )

    def _find_module_for_param(self, param_key: str) -> Any:
        return _worker_find_module_for_param(self, param_key)


def _dump_metadata(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "dict"):
        return dict(value.dict())
    return dict(value)


def _coordexp_llm_worker(args: Any, data_parallel_rank: int, master_port: int, connection: Any) -> Any:
    apply_coord_row_patch_for_rollout_server()
    from swift.pipelines.infer.rollout import llm_worker

    return llm_worker(args, data_parallel_rank, master_port, connection)


def _coordexp_llm_worker_entry(args: Any, data_parallel_rank: int, master_port: int, connection: Any) -> Any:
    apply_coord_row_patch_for_rollout_server()
    from swift.pipelines.infer.rollout import llm_worker_entry

    return llm_worker_entry(args, data_parallel_rank, master_port, connection)


def apply_coord_row_patch_for_vllm_client() -> type:
    """Install CoordExp's token-row offset sender on ms-swift's VLLMClient."""
    try:
        from swift.rlhf_trainers.vllm_client import VLLMClient
    except (ImportError, TypeError, ValueError):
        from swift.trainers.rlhf_trainer.vllm_client import VLLMClient

    if getattr(VLLMClient, "_coordexp_token_row_offsets_patch", False):
        return VLLMClient

    def update_token_row_offsets(self, coord_ids, embed_offset, head_offset=None, tie_head: bool = True):
        from swift.utils import get_torch_device, synchronize

        errors = [None] * self.num_servers
        named_tensors = [("coord_ids", coord_ids), ("embed_offset", embed_offset)]
        if head_offset is not None:
            named_tensors.append(("head_offset", head_offset))

        metadata = {
            name: {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": tuple(tensor.shape),
                "numel": int(tensor.numel()),
            }
            for name, tensor in named_tensors
        }

        def _update_single_server(i: int) -> None:
            try:
                response = self.sessions[i].post(
                    f"{self.base_urls[i]}/update_token_row_offsets/",
                    json={
                        "coord_ids": metadata["coord_ids"],
                        "embed_offset": metadata["embed_offset"],
                        "head_offset": metadata.get("head_offset"),
                        "tie_head": bool(tie_head),
                    },
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Server {i} update token row offsets failed: {response.text}"
                    )

                for _, tensor in named_tensors:
                    synchronize()
                    self.pynccl_comms[i].broadcast(
                        tensor,
                        src=self.pynccl_comms[i].rank,
                        stream=getattr(get_torch_device(), "current_stream", lambda: None)(),
                    )
                synchronize()
                self.pynccl_comms[i].group.barrier()
            except Exception as exc:  # mirrors ms-swift's VLLMClient error aggregation style
                errors[i] = exc

        with ThreadPoolExecutor(max_workers=self.num_servers) as executor:
            futures = [executor.submit(_update_single_server, i) for i in range(self.num_servers)]
            for future in futures:
                future.result()

        all_errors = [exc for exc in errors if exc is not None]
        if all_errors:
            raise RuntimeError(f"Multiple errors on update_token_row_offsets: {all_errors}")

    VLLMClient.update_token_row_offsets = update_token_row_offsets
    VLLMClient._coordexp_token_row_offsets_patch = True
    return VLLMClient


def apply_coord_row_patch_for_rollout_server() -> None:
    """Install CoordExp's rollout-server endpoint and worker row patch locally."""
    from swift.pipelines.infer import rollout as rollout_mod

    worker_cls = rollout_mod.WeightSyncWorkerExtension
    deploy_cls = rollout_mod.SwiftRolloutDeploy

    if not getattr(worker_cls, "_coordexp_token_row_offsets_patch", False):
        worker_cls._token_row_offset_base_rows = None
        worker_cls.update_token_row_offsets = _worker_update_token_row_offsets
        worker_cls._apply_token_row_offsets_to_param = _worker_apply_token_row_offsets_to_param
        worker_cls._find_module_for_param = _worker_find_module_for_param
        worker_cls._coordexp_token_row_offsets_patch = True

    if not getattr(deploy_cls, "_coordexp_token_row_offsets_patch", False):
        original_register = deploy_cls._register_rl_rollout_app

        async def update_token_row_offsets(self, request: _UpdateTokenRowOffsetsRequest):
            coord_ids_metadata = _dump_metadata(request.coord_ids)
            embed_offset_metadata = _dump_metadata(request.embed_offset)
            head_offset_metadata = (
                None
                if request.head_offset is None
                else _dump_metadata(request.head_offset)
            )
            kwargs = {
                "method": "update_token_row_offsets",
                "args": (
                    coord_ids_metadata,
                    embed_offset_metadata,
                    head_offset_metadata,
                    bool(request.tie_head),
                    request.embed_key,
                    request.head_key,
                ),
            }
            for connection in self.connections:
                connection.send(
                    {"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs}
                )
            return {"message": "Request received, updating token row offsets"}

        def _register_rl_rollout_app(self):
            original_register(self)
            paths = {getattr(route, "path", None) for route in self.app.routes}
            if "/update_token_row_offsets/" not in paths:
                self.app.post("/update_token_row_offsets/")(self.update_token_row_offsets)

        def _start_data_parallel_workers(self):
            for data_parallel_rank in range(self.num_connections):
                parent_conn, child_conn = rollout_mod.Pipe()
                worker_func = (
                    _coordexp_llm_worker_entry
                    if self.use_async_engine
                    else _coordexp_llm_worker
                )
                process = rollout_mod.Process(
                    target=worker_func,
                    args=(self.args, data_parallel_rank, self.master_port, child_conn),
                )
                process.start()
                self.connections.append(parent_conn)
                self.processes.append(process)

        @staticmethod
        def get_infer_engine(args, template=None, **kwargs):
            kwargs.update({
                "model_id_or_path": args.model,
                "model_type": args.model_type,
                "revision": args.model_revision,
                "torch_dtype": args.torch_dtype,
                "template": template,
                "use_async_engine": args.vllm_use_async_engine,
                "max_lora_rank": args.vllm_max_lora_rank,
            })
            infer_backend = kwargs.pop("infer_backend", None) or args.infer_backend
            if infer_backend != "vllm":
                rollout_mod.logger.info(
                    "Currently, rollout only supports the vLLM backend. Set vLLM backend"
                )
            kwargs.update(args.get_vllm_engine_kwargs())
            kwargs.update({"enable_lora": args.vllm_enable_lora})
            kwargs["logprobs_mode"] = (
                "processed_logprobs"
                if rollout_mod.check_vllm_version_ge("0.10.2")
                else None
            )

            engine_kwargs = kwargs.get("engine_kwargs", {})
            engine_kwargs.update({"worker_extension_cls": COORDEXP_WORKER_EXTENSION_CLS})

            load_format = engine_kwargs.pop("load_format", "auto")
            kwargs["load_format"] = load_format

            if args.vllm_use_async_engine and args.vllm_data_parallel_size > 1:
                engine_kwargs["data_parallel_size"] = args.vllm_data_parallel_size
            kwargs["engine_kwargs"] = engine_kwargs

            return rollout_mod.GRPOVllmEngine(**kwargs)

        deploy_cls.update_token_row_offsets = update_token_row_offsets
        deploy_cls._register_rl_rollout_app = _register_rl_rollout_app
        deploy_cls._start_data_parallel_workers = _start_data_parallel_workers
        deploy_cls.get_infer_engine = get_infer_engine
        deploy_cls._coordexp_token_row_offsets_patch = True


def _worker_update_token_row_offsets(
    self,
    coord_ids_metadata: dict[str, Any],
    embed_offset_metadata: dict[str, Any],
    head_offset_metadata: dict[str, Any] | None,
    tie_head: bool = True,
    embed_key: str = "language_model.model.embed_tokens.weight",
    head_key: str = "language_model.lm_head.weight",
) -> None:
    from swift.utils import get_torch_device, synchronize

    if self.communicator is None:
        raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")

    def _recv_tensor(metadata: dict[str, Any]) -> torch.Tensor:
        dtype = getattr(torch, str(metadata["dtype"]).split(".")[-1])
        tensor = torch.empty(
            tuple(metadata["shape"]), dtype=dtype, device=self.communicator.device
        )
        self.communicator.broadcast(
            tensor,
            src=self.client_rank,
            stream=getattr(get_torch_device(), "current_stream", lambda: None)(),
        )
        synchronize()
        return tensor

    coord_ids = _recv_tensor(coord_ids_metadata).to(dtype=torch.long)
    embed_offset = _recv_tensor(embed_offset_metadata)
    head_offset = None if head_offset_metadata is None else _recv_tensor(head_offset_metadata)
    self.communicator.group.barrier()

    if coord_ids.ndim != 1 or coord_ids.numel() == 0:
        raise RuntimeError("coord_ids must be a non-empty 1D tensor for token row offset sync")
    if embed_offset.ndim != 2 or embed_offset.shape[0] != coord_ids.numel():
        raise RuntimeError("embed_offset must be [num_coord_ids, hidden] for token row offset sync")
    if not tie_head and head_offset is None:
        raise RuntimeError("untied token row offset sync requires head_offset")
    if head_offset is not None and (
        head_offset.ndim != 2 or head_offset.shape[0] != coord_ids.numel()
    ):
        raise RuntimeError("head_offset must be [num_coord_ids, hidden] for token row offset sync")

    named_parameters = dict(self.model_runner.model.named_parameters())
    embed_param = named_parameters.get(embed_key)
    if embed_param is None:
        raise RuntimeError(f"token row offset sync could not find embedding parameter: {embed_key}")

    head_param = named_parameters.get(head_key)
    if head_param is None:
        head_param = embed_param if bool(tie_head) else None
    if head_param is None:
        raise RuntimeError(f"token row offset sync could not find lm_head parameter: {head_key}")

    if self._token_row_offset_base_rows is None:
        self._token_row_offset_base_rows = {}
    self._apply_token_row_offsets_to_param(embed_key, embed_param, coord_ids, embed_offset)

    head_delta = embed_offset if bool(tie_head) else head_offset
    if head_param is not embed_param or not bool(tie_head):
        self._apply_token_row_offsets_to_param(head_key, head_param, coord_ids, head_delta)


def _worker_apply_token_row_offsets_to_param(
    self,
    key: str,
    param: torch.nn.Parameter,
    coord_ids: torch.Tensor,
    offset: torch.Tensor,
) -> None:
    if offset is None:
        raise RuntimeError(f"token row offset sync received no offset for {key}")
    if param.ndim != 2:
        raise RuntimeError(f"token row offset target must be 2D, got {key} shape={tuple(param.shape)}")
    if offset.shape[1] != param.shape[1]:
        raise RuntimeError(
            f"token row offset hidden size mismatch for {key}: "
            f"offset={offset.shape[1]} target={param.shape[1]}"
        )

    module = self._find_module_for_param(key)
    shard_indices = getattr(module, "shard_indices", None)
    if shard_indices is None:
        local_mask = (coord_ids >= 0) & (coord_ids < param.shape[0])
        local_indices = coord_ids[local_mask]
        local_offsets = offset[local_mask]
    else:
        start = int(shard_indices.org_vocab_start_index)
        end = int(shard_indices.org_vocab_end_index)
        local_mask = (coord_ids >= start) & (coord_ids < end)
        local_indices = coord_ids[local_mask] - start
        local_offsets = offset[local_mask]

    if local_indices.numel() == 0:
        return
    if torch.any(local_indices < 0) or torch.any(local_indices >= param.shape[0]):
        raise RuntimeError(f"token row offset local row index out of bounds for {key}")

    local_indices = local_indices.to(device=param.device, dtype=torch.long)
    local_offsets = local_offsets.to(device=param.device, dtype=param.dtype)

    if self._token_row_offset_base_rows is None:
        self._token_row_offset_base_rows = {}
    base_cache = self._token_row_offset_base_rows.setdefault(key, {})
    cpu_indices = [int(idx) for idx in local_indices.detach().cpu().tolist()]
    for row_idx in cpu_indices:
        if row_idx not in base_cache:
            base_cache[row_idx] = param.data[row_idx].detach().clone()

    base_rows = torch.stack([base_cache[row_idx] for row_idx in cpu_indices]).to(
        device=param.device, dtype=param.dtype
    )
    param.data[local_indices] = base_rows + local_offsets


def _worker_find_module_for_param(self, param_key: str) -> Any:
    suffix = ".weight"
    module_name = param_key[: -len(suffix)] if param_key.endswith(suffix) else param_key
    for name, module in self.model_runner.model.named_modules():
        if name == module_name:
            return module
    return None
