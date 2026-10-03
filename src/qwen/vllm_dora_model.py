"""One-adapter, TP=1 Qwen3-VL DoRA execution for vLLM 0.29.

Register ``src.qwen.vllm_dora_model.CoordExpDoRAQwen3VLForConditionalGeneration``
through vLLM's ``ModelRegistry``. The PEFT payload remains unmerged.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

from src.adapters.dora import inspect_dora_adapter_payload, normalize_dora_state_key
from src.qwen.untied_embeddings import inspect_special_token_embedding_delta_payload
from src.qwen.tokens import DEFAULT_COORDINATE_TOKENS
from src.qwen.coordinate_policy import coordinate_norm_values, scale_coordinate_logits
from src.qwen.vllm_trace import assert_trace_idle, capture_raw_logits


_KINDS = frozenset({"lora_A", "lora_B", "lora_magnitude_vector"})
_COORDINATE_TOKENS = frozenset(DEFAULT_COORDINATE_TOKENS)
_SUFFIXES = (
    (".lora_A.weight", "lora_A"),
    (".lora_B.weight", "lora_B"),
    (".lora_magnitude_vector.weight", "lora_magnitude_vector"),
    (".lora_magnitude_vector", "lora_magnitude_vector"),
)


def _target_tensors(tensors: Mapping[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    targets: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in tensors.items():
        normalized = normalize_dora_state_key(key, adapter_name="default")
        match = next(((normalized[: -len(suffix)], kind) for suffix, kind in _SUFFIXES if normalized.endswith(suffix)), None)
        if match is None or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"unsupported DoRA tensor: {key}")
        target, kind = match
        if target.startswith("language_model."):
            target = "model." + target
        if not target.startswith("model.language_model."):
            raise ValueError(f"DoRA target is outside language tower: {target}")
        parts = targets.setdefault(target, {})
        if kind in parts:
            raise ValueError(f"duplicate DoRA tensor: {key}")
        parts[kind] = tensor
    if not targets or any(set(parts) != _KINDS for parts in targets.values()):
        raise ValueError("DoRA targets must each contain A, B, and magnitude")
    return targets


def _validate_declared_targets(declared: set[str], targets: set[str]) -> None:
    # PEFT may save target_modules as shared suffixes or exact module names.
    observed = {target.rsplit(".", 1)[-1] for target in targets} if all("." not in item for item in declared) else targets
    if not declared or declared != observed:
        raise ValueError("DoRA config targets differ from tensor targets")


class _DoRALinear(nn.Module):
    """Add PEFT's unmerged DoRA correction to a vLLM packed linear."""

    def __init__(
        self,
        base: nn.Module,
        targets: tuple[str, ...],
        widths: tuple[int, ...],
        tensors: Mapping[str, Mapping[str, torch.Tensor]],
        scaling: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.targets = targets
        self.widths = widths
        self.scaling = scaling
        if base.weight.ndim != 2 or sum(widths) != base.weight.shape[0]:
            raise ValueError("packed DoRA output slices disagree with base weight")
        for index, (target, width) in enumerate(zip(targets, widths, strict=True)):
            parts = tensors.get(target)
            if parts is None:
                raise ValueError(f"missing DoRA target: {target}")
            a, b, m = (parts[kind] for kind in ("lora_A", "lora_B", "lora_magnitude_vector"))
            if a.ndim != 2 or b.ndim != 2 or m.ndim != 1 or a.shape[1] != base.weight.shape[1] or b.shape != (width, a.shape[0]) or m.shape != (width,):
                raise ValueError(f"invalid DoRA shape: {target}")
            if a.dtype != b.dtype or a.dtype not in (torch.bfloat16, torch.float32) or m.dtype != torch.float32:
                raise ValueError(f"unsupported DoRA dtype: {target}")
            for kind, value in (("a", a), ("b", b), ("m", m)):
                self.register_buffer(f"{kind}_{index}", value.detach().to(base.weight.device).clone(), persistent=False)
            self.register_buffer(f"scale_{index}", torch.empty_like(m, device=base.weight.device), persistent=False)
        self._recompute_scales()

    @torch.no_grad()
    def _recompute_scales(self) -> None:
        for index, scale in enumerate(self._calculated_scales(None)):
            getattr(self, f"scale_{index}").copy_(scale)

    @torch.no_grad()
    def _calculated_scales(
        self, replacement: Mapping[str, Mapping[str, torch.Tensor]] | None
    ) -> list[torch.Tensor]:
        # PEFT forms B(A(I)) before its row norm. Compute once per refresh.
        scales = []
        for index, weight in enumerate(self.base.weight.split(self.widths, dim=0)):
            if replacement is None:
                a, b, m = (getattr(self, f"{kind}_{index}") for kind in ("a", "b", "m"))
            else:
                parts = replacement[self.targets[index]]
                a, b, m = (
                    parts[kind].detach().to(weight.device)
                    for kind in ("lora_A", "lora_B", "lora_magnitude_vector")
                )
            eye = torch.eye(a.shape[1], dtype=a.dtype, device=a.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=a.is_cuda):
                lora_weight = F.linear(F.linear(eye, a), b).T.to(a.dtype)
            norm = torch.linalg.vector_norm(weight.to(a.dtype) + self.scaling * lora_weight, dim=1).to(a.dtype)
            if not torch.isfinite(norm).all() or (norm == 0).any():
                raise ValueError(f"invalid DoRA weight norm: {self.targets[index]}")
            scale = m / norm
            if not torch.isfinite(scale).all():
                raise ValueError(f"invalid DoRA scale: {self.targets[index]}")
            scales.append(scale)
        return scales

    def forward(self, x: torch.Tensor):
        base_result = self.base(x)
        output, returned_bias = base_result if isinstance(base_result, tuple) else (base_result, None)
        if output.shape[-1] != sum(self.widths):
            raise ValueError("DoRA base output width changed")
        corrections = []
        for index, base_slice in enumerate(output.split(self.widths, dim=-1)):
            a, b, scale = (getattr(self, f"{kind}_{index}") for kind in ("a", "b", "scale"))
            x_lora = x.to(a.dtype)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                lora_result = F.linear(F.linear(x_lora, a), b)
            bias = self.base.bias
            if bias is not None and not self.base.skip_bias_add:
                base_slice = base_slice - bias.split(self.widths)[index]
            correction = (scale - 1) * base_slice + scale * lora_result * self.scaling
            corrections.append(correction)
        result = output + torch.cat(corrections, dim=-1)
        result = result.to(output.dtype)
        return (result, returned_bias) if isinstance(base_result, tuple) else result


class _SelectedInput(nn.Module):
    def __init__(self, base: nn.Module, rows: torch.Tensor, delta: torch.Tensor) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("delta", delta, persistent=False)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.base(input_ids)
        rows = self.rows[input_ids]
        selected = rows >= 0
        addition = self.delta.to(output.dtype)[rows.clamp_min(0)]
        return output + addition * selected.unsqueeze(-1).to(output.dtype)


class CoordExpDoRAQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Fixed language-only DoRA with in-place refresh and untied token deltas."""

    def __init__(self, *, vllm_config, prefix: str = "model") -> None:
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("CoordExp DoRA requires TP=1 and PP=1")
        if vllm_config.quant_config is not None or vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("CoordExp DoRA requires nonquantized BF16 base weights")
        if vllm_config.lora_config is not None:
            raise ValueError("CoordExp DoRA cannot use vLLM LoRA manager")
        cfg = getattr(vllm_config.model_config.hf_config, "coordexp_dora", None)
        if not isinstance(cfg, dict) or set(cfg) != {"adapter_path", "embedding_path", "identity"}:
            raise ValueError("hf_config.coordexp_dora requires adapter_path, embedding_path, identity")
        if any(not isinstance(value, str) or not value for value in cfg.values()):
            raise ValueError("CoordExp DoRA paths and identity must be nonempty strings")
        if not vllm_config.model_config.hf_config.tie_word_embeddings:
            raise ValueError("CoordExp DoRA requires tied frozen base embeddings")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        processor = self.language_model.logits_processor
        if (
            processor.scale != 1.0
            or processor.soft_cap is not None
            or processor.logits_as_input
            or processor.head_dtype not in (None, torch.bfloat16)
        ):
            raise ValueError("CoordExp DoRA requires unmodified BF16 language logits")
        self._coordexp_config = cfg
        self.coordexp_dora_identity: str | None = None
        self._coordexp_linears: tuple[_DoRALinear, ...] = ()
        self._coordexp_coordinate_token_ids: tuple[int, ...] = ()

    def load_weights(self, weights) -> set[str]:
        if self.coordexp_dora_identity is not None:
            raise ValueError("CoordExp DoRA base weights may be loaded only once")
        loaded = super().load_weights(weights)
        cfg = self._coordexp_config
        model_path = self.model_config.model
        adapter = inspect_dora_adapter_payload(cfg["adapter_path"], expected_base_model_path=model_path)
        embedding = inspect_special_token_embedding_delta_payload(cfg["embedding_path"], expected_base_model_path=model_path)
        semantic = embedding["semantic_identity"]
        if semantic["tie_word_embeddings"] is not False:
            raise ValueError("CoordExp DoRA requires independent input/output deltas")
        adapter_config = json.loads((Path(adapter["root"]) / "adapter_config.json").read_text())
        unsupported = (
            "modules_to_save", "use_rslora", "use_qalora", "lora_bias",
            "fan_in_fan_out", "rank_pattern", "alpha_pattern", "target_parameters",
            "layers_to_transform", "layers_pattern", "layer_replication",
            "exclude_modules", "trainable_token_indices", "loftq_config",
            "corda_config", "eva_config", "megatron_config",
        )
        if adapter_config.get("bias") != "none" or adapter_config.get("lora_dropout") != 0 or any(adapter_config.get(key) for key in unsupported):
            raise ValueError("unsupported DoRA adapter options")
        rank = adapter["semantic_identity"]["r"]
        scaling = adapter["semantic_identity"]["lora_alpha"] / rank
        adapter_tensors: dict[str, torch.Tensor] = {}
        for path in sorted(Path(adapter["root"]).glob("adapter_model*.safetensors")):
            for key, value in load_file(str(path), device="cpu").items():
                if key in adapter_tensors:
                    raise ValueError(f"duplicate adapter key: {key}")
                adapter_tensors[key] = value
        targets = _target_tensors(adapter_tensors)
        declared = set(adapter["semantic_identity"]["target_modules"])
        _validate_declared_targets(declared, set(targets))
        installed: list[_DoRALinear] = []
        used: set[str] = set()
        for layer_index, layer in enumerate(self.language_model.model.layers):
            stem = f"model.language_model.layers.{layer_index}."
            for owner, name, suffixes, widths in (
                (layer.self_attn, "qkv_proj", ("q_proj", "k_proj", "v_proj"), (layer.self_attn.q_size, layer.self_attn.kv_size, layer.self_attn.kv_size)),
                (layer.self_attn, "o_proj", ("o_proj",), None),
                (layer.mlp, "gate_up_proj", ("gate_proj", "up_proj"), None),
                (layer.mlp, "down_proj", ("down_proj",), None),
            ):
                base = getattr(owner, name)
                if widths is None:
                    widths = (base.weight.shape[0] // len(suffixes),) * len(suffixes)
                parent = "self_attn" if owner is layer.self_attn else "mlp"
                names = tuple(stem + parent + "." + suffix for suffix in suffixes)
                present = set(names) & set(targets)
                if not present:
                    continue
                if present != set(names):
                    raise ValueError(f"partial packed DoRA target: {names}")
                wrapped = _DoRALinear(base, names, widths, targets, scaling)
                setattr(owner, name, wrapped)
                installed.append(wrapped)
                used.update(names)
        if used != set(targets) or not installed:
            raise ValueError(f"unmapped DoRA targets: {sorted(set(targets) - used)}")
        self._coordexp_linears = tuple(installed)
        delta_path = Path(embedding["root"]) / "special_token_embeddings.safetensors"
        delta_tensors = load_file(str(delta_path), device="cpu")
        token_ids = semantic["token_ids"]
        self._coordexp_coordinate_token_ids = tuple(
            int(token_id)
            for token, token_id in zip(semantic["token_strings"], token_ids, strict=True)
            if token in _COORDINATE_TOKENS
        )
        base_embedding = self.language_model.model.embed_tokens.weight
        head = self.language_model.lm_head.weight
        if base_embedding.data_ptr() != head.data_ptr() or max(token_ids) >= base_embedding.shape[0]:
            raise ValueError("tied base embedding identity or token range differs")
        self.register_buffer("coordexp_token_ids", torch.tensor(token_ids, dtype=torch.long, device=base_embedding.device), persistent=False)
        lookup = torch.full((base_embedding.shape[0],), -1, dtype=torch.long, device=base_embedding.device)
        lookup[self.coordexp_token_ids] = torch.arange(len(token_ids), device=lookup.device)
        self.register_buffer("coordexp_token_rows", lookup, persistent=False)
        for key in ("input_embed_delta", "output_embed_delta"):
            value = delta_tensors[key]
            if value.shape != (len(token_ids), base_embedding.shape[1]) or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"invalid selected-token delta: {key}")
            self.register_buffer("coordexp_" + key, value.to(base_embedding.device).clone(), persistent=False)
        self.language_model.model.embed_tokens = _SelectedInput(
            self.language_model.model.embed_tokens,
            self.coordexp_token_rows,
            self.coordexp_input_embed_delta,
        )
        self.coordexp_dora_identity = cfg["identity"]
        return loaded

    @torch.no_grad()
    def refresh_coordexp_dora(
        self,
        adapter_tensors: Mapping[str, torch.Tensor],
        embedding_tensors: Mapping[str, torch.Tensor],
        *,
        identity: str,
    ) -> str:
        assert_trace_idle(self)
        if self.coordexp_dora_identity is None or not isinstance(identity, str) or not identity:
            raise ValueError("CoordExp DoRA is uninitialized or snapshot identity is empty")
        targets = _target_tensors(adapter_tensors)
        expected = {target for layer in self._coordexp_linears for target in layer.targets}
        if set(targets) != expected or set(embedding_tensors) != {"input_embed_delta", "output_embed_delta"}:
            raise ValueError("refresh payload keys differ from installed DoRA schema")
        copies: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self._coordexp_linears:
            for index, target in enumerate(layer.targets):
                for source_kind, buffer_kind in (("lora_A", "a"), ("lora_B", "b"), ("lora_magnitude_vector", "m")):
                    source = targets[target][source_kind]
                    destination = getattr(layer, f"{buffer_kind}_{index}")
                    if not isinstance(source, torch.Tensor) or source.shape != destination.shape or source.dtype != destination.dtype or not torch.isfinite(source).all():
                        raise ValueError(f"invalid refresh tensor: {target}.{source_kind}")
                    copies.append((destination, source))
        for key in ("input_embed_delta", "output_embed_delta"):
            source = embedding_tensors[key]
            destination = getattr(self, "coordexp_" + key)
            if not isinstance(source, torch.Tensor) or source.shape != destination.shape or source.dtype != destination.dtype or not torch.isfinite(source).all():
                raise ValueError(f"invalid refresh tensor: {key}")
            copies.append((destination, source))
        refreshed_coordinate_norms = None
        if getattr(self, "_coordinate_output_norm_mode", "off") == "median":
            refreshed_coordinate_norms = self._coordinate_output_norm_values(
                self.coordexp_coordinate_output_norm_ids,
                output_delta=embedding_tensors["output_embed_delta"],
            )
        scale_copies = [
            (getattr(layer, f"scale_{index}"), value)
            for layer in self._coordexp_linears
            for index, value in enumerate(layer._calculated_scales(targets))
        ]
        for destination, source in copies:
            destination.copy_(source.detach().to(destination.device))
        for destination, source in scale_copies:
            destination.copy_(source)
        if refreshed_coordinate_norms is not None:
            self.coordexp_coordinate_output_norm_factors.copy_(
                refreshed_coordinate_norms["factors"]
            )
            self._store_coordinate_output_norm_stats(refreshed_coordinate_norms)
        elif getattr(self, "_coordinate_output_norm_configured", False):
            self._coordinate_output_norm_stats = None
        self.coordexp_dora_identity = identity
        return identity

    @staticmethod
    def _coordinate_output_norm_ids(token_ids) -> tuple[int, ...]:
        if isinstance(token_ids, torch.Tensor):
            if token_ids.ndim != 1:
                raise ValueError("coordinate token IDs must be one-dimensional")
            values = token_ids.detach().cpu().tolist()
        else:
            try:
                values = list(token_ids)
            except TypeError as exc:
                raise ValueError("coordinate token IDs must be a sequence") from exc
        if len(values) != 1000 or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise ValueError("coordinate token IDs must contain 1,000 integer IDs")
        result = tuple(values)
        if len(set(result)) != 1000 or any(value < 0 for value in result):
            raise ValueError("coordinate token IDs must be 1,000 unique non-negative IDs")
        return result

    @torch.no_grad()
    def _coordinate_output_norm_values(
        self,
        token_ids: torch.Tensor,
        *,
        output_delta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | float]:
        head = self.language_model.lm_head
        weight = getattr(head, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.dtype != torch.bfloat16:
            raise ValueError("coordinate output norms require a BF16 language head")
        if getattr(head, "bias", None) is not None:
            raise ValueError("coordinate output norms do not support an output-head bias")
        delta = (
            getattr(self, "coordexp_output_embed_delta")
            if output_delta is None
            else output_delta
        )
        if (
            not isinstance(delta, torch.Tensor)
            or delta.ndim != 2
            or delta.shape[1] != weight.shape[1]
            or delta.dtype != torch.float32
            or not torch.isfinite(delta).all()
        ):
            raise ValueError("coordinate output norm delta must be finite FP32 rows")
        ids = token_ids.to(device=weight.device, dtype=torch.long)
        if ids.numel() != 1000 or int(ids.min()) < 0 or int(ids.max()) >= weight.shape[0]:
            raise ValueError("coordinate output norm IDs are outside the language head")
        selected_rows = self.coordexp_token_rows.index_select(0, ids.to(self.coordexp_token_rows.device))
        if (
            selected_rows.shape != ids.shape
            or (selected_rows < 0).any()
            or int(selected_rows.max()) >= delta.shape[0]
        ):
            raise ValueError("coordinate IDs are missing selected output delta rows")
        values = coordinate_norm_values(weight, delta, ids, selected_rows)
        factors, norms, median = (values[key] for key in ("factors", "norms", "median"))
        return {
            "factors": factors,
            "norm_min": float(norms.min().item()),
            "norm_max": float(norms.max().item()),
            "median_norm": float(median.item()),
            "factor_min": float(factors.min().item()),
            "factor_max": float(factors.max().item()),
        }

    def _store_coordinate_output_norm_stats(self, values: Mapping[str, torch.Tensor | float]) -> None:
        self._coordinate_output_norm_stats = {
            key: float(value) for key, value in values.items() if key != "factors"
        }

    @torch.no_grad()
    def configure_coordinate_output_norm(
        self,
        mode: str,
        token_ids,
        *,
        identity: str,
    ) -> dict[str, object]:
        if mode not in ("off", "median"):
            raise ValueError("coordinate output norm mode must be 'off' or 'median'")
        if not isinstance(identity, str) or not identity or identity != self.coordexp_dora_identity:
            raise ValueError("coordinate output norm identity differs from installed DoRA snapshot")
        ids = self._coordinate_output_norm_ids(token_ids)
        expected_ids = getattr(self, "_coordexp_coordinate_token_ids", ())
        selected_ids = getattr(self, "coordexp_token_ids", None)
        if (
            len(expected_ids) != 1000
            or ids != tuple(expected_ids)
            or not isinstance(selected_ids, torch.Tensor)
            or not set(ids).issubset(set(int(value) for value in selected_ids.detach().cpu().tolist()))
        ):
            raise ValueError("coordinate IDs differ from selected output coordinate rows")
        head = self.language_model.lm_head
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=head.weight.device)
        values = self._coordinate_output_norm_values(ids_tensor) if mode == "median" else None
        id_buffer = getattr(self, "coordexp_coordinate_output_norm_ids", None)
        if id_buffer is None:
            self.register_buffer("coordexp_coordinate_output_norm_ids", ids_tensor, persistent=False)
        else:
            id_buffer.copy_(ids_tensor)
        if mode == "median":
            assert values is not None
            factor_buffer = getattr(self, "coordexp_coordinate_output_norm_factors", None)
            if factor_buffer is None:
                self.register_buffer(
                    "coordexp_coordinate_output_norm_factors",
                    values["factors"],
                    persistent=False,
                )
            else:
                factor_buffer.copy_(values["factors"])
            self._store_coordinate_output_norm_stats(values)
        else:
            self._coordinate_output_norm_stats = None
        self._coordinate_output_norm_mode = mode
        self._coordinate_output_norm_configured = True
        self._coordinate_output_norm_calls = 0
        self._coordinate_output_norm_positions = 0
        self._coordinate_output_norm_first_call = None
        return self.coordinate_output_norm_receipt()

    def coordinate_output_norm_receipt(self) -> dict[str, object]:
        stats = getattr(self, "_coordinate_output_norm_stats", None) or {}
        ids = getattr(self, "coordexp_coordinate_output_norm_ids", None)
        first_call = getattr(self, "_coordinate_output_norm_first_call", None)
        if first_call is not None:
            first_call = dict(first_call)
            for key in ("changed_coordinates", "max_abs_difference"):
                value = first_call[key]
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().item()
                if key == "max_abs_difference" and not math.isfinite(float(value)):
                    value = None
                first_call[key] = int(value) if key == "changed_coordinates" else value
        return {
            "mode": getattr(self, "_coordinate_output_norm_mode", "off"),
            "identity": getattr(self, "coordexp_dora_identity", None),
            "coordinate_ids": None if ids is None else ids.detach().cpu().tolist(),
            "coordinate_tokens": 0 if ids is None else int(ids.numel()),
            "calls": int(getattr(self, "_coordinate_output_norm_calls", 0)),
            "coordinate_positions_processed": int(getattr(self, "_coordinate_output_norm_positions", 0)),
            "norm_min": stats.get("norm_min"),
            "norm_max": stats.get("norm_max"),
            "median_norm": stats.get("median_norm"),
            "factor_min": stats.get("factor_min"),
            "factor_max": stats.get("factor_max"),
            "first_call": first_call,
        }

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        if logits is None:
            return None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=hidden_states.is_cuda):
            correction = hidden_states.to(self.coordexp_output_embed_delta.dtype) @ self.coordexp_output_embed_delta.T
        correction = correction.to(logits.dtype)
        index = self.coordexp_token_ids.view(*([1] * (correction.ndim - 1)), -1).expand_as(correction)
        logits.scatter_add_(-1, index, correction)
        capture_raw_logits(self, logits)
        if getattr(self, "_coordinate_output_norm_configured", False):
            self._coordinate_output_norm_calls += 1
            self._coordinate_output_norm_positions += logits.numel() // logits.shape[-1]
            if self._coordinate_output_norm_mode == "median":
                norm_ids = self.coordexp_coordinate_output_norm_ids.to(logits.device)
                before = (logits.index_select(-1, norm_ids)
                          if self._coordinate_output_norm_first_call is None else None)
                scale_coordinate_logits(logits, norm_ids,
                    self.coordexp_coordinate_output_norm_factors, inplace=True)
                if self._coordinate_output_norm_first_call is None:
                    assert before is not None
                    scaled = logits.index_select(-1, norm_ids)
                    delta = (scaled.to(torch.float64) - before.to(torch.float64)).abs()
                    self._coordinate_output_norm_first_call = {
                        "native_dtype": str(logits.dtype),
                        "scaling_active": True,
                        "changed_coordinates": torch.count_nonzero(scaled != before),
                        "max_abs_difference": delta.max() if delta.numel() else torch.zeros((), device=logits.device),
                        "non_coordinate_unchanged": True,
                    }
            elif self._coordinate_output_norm_first_call is None:
                self._coordinate_output_norm_first_call = {
                    "native_dtype": str(logits.dtype),
                    "scaling_active": False,
                    "changed_coordinates": 0,
                    "max_abs_difference": 0.0,
                    "non_coordinate_unchanged": True,
                }
        return logits
