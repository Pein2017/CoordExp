from __future__ import annotations

import contextlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch

from src.metrics.reporter import warn_once

from ..teacher_forcing.contracts import PipelineModuleSpec, PipelineResult

_MONITOR_ATTR = "_coordexp_loss_gradient_monitor"

_AUTO_EXCLUDE_TOKENS = (
    "vision",
    "visual",
    "image",
    "patch_embed",
    "vision_tower",
    "embed_tokens",
    "embeddings",
)
_AUTO_NORM_TOKENS = (
    "layernorm",
    "layer_norm",
    ".norm",
    "ln_f",
    "ln_",
)

_BBOX_TERM_STATE_KEYS = {
    "bbox_smoothl1": "bbox_smoothl1_contrib",
    "bbox_ciou": "bbox_ciou_contrib",
}
_COORD_TERM_STATE_KEYS = {
    "coord_token_ce": "coord_token_ce_contrib",
    "coord_soft_ce": "coord_soft_ce_contrib",
    "coord_w1": "coord_w1_contrib",
    "coord_el1": "coord_el1_contrib",
    "coord_ehuber": "coord_ehuber_contrib",
    "coord_entropy": "coord_entropy_contrib",
    "coord_gate": "coord_gate_contrib",
}


def loss_gradient_monitor_enabled(trainer: Any) -> bool:
    cfg = getattr(trainer, "loss_gradient_monitor_cfg", None)
    return isinstance(cfg, Mapping) and bool(cfg.get("enabled", False))


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_float(cfg: Any, key: str, default: float) -> float:
    raw = _cfg_get(cfg, key, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _as_scalar_tensor(value: Any) -> Optional[torch.Tensor]:
    if not isinstance(value, torch.Tensor):
        return None
    if value.numel() != 1:
        return None
    return value


def _parse_objective_specs(
    objective_specs: Optional[Sequence[Mapping[str, Any]]],
) -> Dict[str, PipelineModuleSpec]:
    specs: Dict[str, PipelineModuleSpec] = {}
    for raw in list(objective_specs or []):
        if not isinstance(raw, Mapping):
            continue
        parsed = PipelineModuleSpec.from_mapping(raw)
        if not parsed.name:
            continue
        specs.setdefault(parsed.name, parsed)
    return specs


def build_stage1_coord_monitor_terms(
    *,
    result: Any,
    cfg: Any,
) -> Dict[str, torch.Tensor]:
    terms: Dict[str, torch.Tensor] = {}
    candidates = (
        ("S1/coord_soft_ce", "softce_contrib", "soft_ce_weight"),
        ("S1/coord_w1", "w1_contrib", "w1_weight"),
        ("S1/coord_ce", "ce_contrib", "ce_weight"),
        ("S1/coord_gate", "gate_contrib", "gate_weight"),
    )
    for name, attr_name, weight_key in candidates:
        if float(_cfg_float(cfg, weight_key, 0.0)) == 0.0:
            continue
        tensor = _as_scalar_tensor(getattr(result, attr_name, None))
        if tensor is None:
            continue
        terms[name] = tensor
    return terms


def _collect_weighted_coord_terms_from_state(
    *,
    state: Mapping[str, Any],
    coord_provenance: str,
    bbox_module_weight: float,
    coord_module_weight: float,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}

    if float(bbox_module_weight) != 0.0:
        for atom_name, state_key in _BBOX_TERM_STATE_KEYS.items():
            tensor = _as_scalar_tensor(state.get(state_key))
            if tensor is None:
                continue
            out[f"{coord_provenance}/{atom_name}"] = tensor * float(bbox_module_weight)

    if float(coord_module_weight) != 0.0:
        for atom_name, state_key in _COORD_TERM_STATE_KEYS.items():
            tensor = _as_scalar_tensor(state.get(state_key))
            if tensor is None:
                continue
            out[f"{coord_provenance}/{atom_name}"] = tensor * float(coord_module_weight)

    return out


def build_stage2_coord_monitor_terms_from_pipeline(
    *,
    pipeline_result: PipelineResult,
    objective_specs: Optional[Sequence[Mapping[str, Any]]],
    coord_provenance: str,
) -> Dict[str, torch.Tensor]:
    specs = _parse_objective_specs(objective_specs)
    bbox_spec = specs.get("bbox_geo")
    coord_spec = specs.get("coord_reg")
    bbox_module_weight = float(bbox_spec.weight) if bbox_spec is not None else 0.0
    coord_module_weight = float(coord_spec.weight) if coord_spec is not None else 0.0
    return _collect_weighted_coord_terms_from_state(
        state=dict(pipeline_result.state or {}),
        coord_provenance=str(coord_provenance),
        bbox_module_weight=float(bbox_module_weight),
        coord_module_weight=float(coord_module_weight),
    )


def build_stage2_two_channel_coord_monitor_terms(
    *,
    channel: str,
    pipeline_result: PipelineResult,
    objective_specs: Optional[Sequence[Mapping[str, Any]]],
    bbox_module_weight: float,
    coord_module_weight: float,
    a1_bbox_state: Optional[Mapping[str, Any]] = None,
    a1_coord_state: Optional[Mapping[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    s = str(channel).strip().upper()
    if s == "A":
        terms = build_stage2_coord_monitor_terms_from_pipeline(
            pipeline_result=pipeline_result,
            objective_specs=objective_specs,
            coord_provenance="A2_coord",
        )
        a1_state: Dict[str, Any] = {}
        if isinstance(a1_bbox_state, Mapping):
            a1_state.update(dict(a1_bbox_state))
        if isinstance(a1_coord_state, Mapping):
            a1_state.update(dict(a1_coord_state))
        if a1_state:
            terms.update(
                _collect_weighted_coord_terms_from_state(
                    state=a1_state,
                    coord_provenance="A1_coord",
                    bbox_module_weight=float(bbox_module_weight),
                    coord_module_weight=float(coord_module_weight),
                )
            )
        return terms

    return build_stage2_coord_monitor_terms_from_pipeline(
        pipeline_result=pipeline_result,
        objective_specs=objective_specs,
        coord_provenance="B_coord",
    )


@dataclass
class LossGradientMonitor:
    trainer: Any
    cfg: Mapping[str, Any]
    interval_steps: int
    ema_beta: float
    require_sync_gradients: bool
    param_strategy: str
    max_params: int
    max_numel: int
    include_pattern: Optional[re.Pattern[str]] = None
    exclude_pattern: Optional[re.Pattern[str]] = None
    _ema_state: Dict[str, float] = field(default_factory=dict)
    _selected_params: Optional[tuple[tuple[str, torch.nn.Parameter], ...]] = None
    _shared_param_count: int = 0
    _shared_param_numel: int = 0

    @classmethod
    def from_cfg(cls, *, trainer: Any, cfg: Mapping[str, Any]) -> "LossGradientMonitor":
        try:
            interval_steps = int(cfg.get("interval_steps", 50) or 50)
        except (TypeError, ValueError) as exc:
            raise ValueError("loss_gradient_monitor.interval_steps must be a positive int") from exc
        if interval_steps <= 0:
            raise ValueError("loss_gradient_monitor.interval_steps must be a positive int")

        try:
            ema_beta = float(cfg.get("ema_beta", 0.98) or 0.98)
        except (TypeError, ValueError) as exc:
            raise ValueError("loss_gradient_monitor.ema_beta must be a float in [0, 1)") from exc
        if not (0.0 <= float(ema_beta) < 1.0):
            raise ValueError("loss_gradient_monitor.ema_beta must be a float in [0, 1)")

        coord_only = cfg.get("coord_only", True)
        if bool(coord_only) is not True:
            raise ValueError("loss_gradient_monitor.coord_only=false is not supported")

        granularity = str(cfg.get("granularity", "atomic") or "atomic").strip().lower()
        if granularity != "atomic":
            raise ValueError("loss_gradient_monitor.granularity must be 'atomic'")

        param_block = cfg.get("param_block", {})
        if param_block is None:
            param_block = {}
        if not isinstance(param_block, Mapping):
            raise ValueError("loss_gradient_monitor.param_block must be a mapping when provided")

        strategy = str(
            param_block.get("strategy", "auto_last_lm_layernorm") or "auto_last_lm_layernorm"
        ).strip()
        if strategy not in {"auto_last_lm_layernorm", "regex"}:
            raise ValueError(
                "loss_gradient_monitor.param_block.strategy must be one of "
                "{'auto_last_lm_layernorm', 'regex'}"
            )

        try:
            max_params = int(param_block.get("max_params", 64) or 64)
        except (TypeError, ValueError) as exc:
            raise ValueError("loss_gradient_monitor.param_block.max_params must be an int > 0") from exc
        if max_params <= 0:
            raise ValueError("loss_gradient_monitor.param_block.max_params must be an int > 0")

        try:
            max_numel = int(param_block.get("max_numel", 200000) or 200000)
        except (TypeError, ValueError) as exc:
            raise ValueError("loss_gradient_monitor.param_block.max_numel must be an int > 0") from exc
        if max_numel <= 0:
            raise ValueError("loss_gradient_monitor.param_block.max_numel must be an int > 0")

        include_pattern = None
        exclude_pattern = None
        if strategy == "regex":
            include_raw = param_block.get("include")
            if not isinstance(include_raw, str) or not include_raw.strip():
                raise ValueError(
                    "loss_gradient_monitor.param_block.include must be a non-empty regex when strategy=regex"
                )
            try:
                include_pattern = re.compile(include_raw)
            except re.error as exc:
                raise ValueError(
                    "loss_gradient_monitor.param_block.include is not a valid regex"
                ) from exc

            exclude_raw = param_block.get("exclude")
            if isinstance(exclude_raw, str) and exclude_raw.strip():
                try:
                    exclude_pattern = re.compile(exclude_raw)
                except re.error as exc:
                    raise ValueError(
                        "loss_gradient_monitor.param_block.exclude is not a valid regex"
                    ) from exc

        return cls(
            trainer=trainer,
            cfg=dict(cfg),
            interval_steps=int(interval_steps),
            ema_beta=float(ema_beta),
            require_sync_gradients=bool(cfg.get("require_sync_gradients", True)),
            param_strategy=str(strategy),
            max_params=int(max_params),
            max_numel=int(max_numel),
            include_pattern=include_pattern,
            exclude_pattern=exclude_pattern,
        )

    def should_run(self) -> bool:
        step = self._optimizer_step()
        if step <= 0:
            return False
        if int(step) % int(self.interval_steps) != 0:
            return False
        if self.require_sync_gradients and not self._sync_gradients_now():
            return False
        return True

    def measure(
        self,
        *,
        model: Any,
        loss_terms: Mapping[str, torch.Tensor],
    ) -> Dict[str, float]:
        if not self.should_run():
            return {}

        terms: Dict[str, torch.Tensor] = {}
        for name, tensor in loss_terms.items():
            scalar = _as_scalar_tensor(tensor)
            if scalar is None:
                continue
            if not bool(getattr(scalar, "requires_grad", False)):
                continue
            terms[str(name)] = scalar

        if not terms:
            return {}

        params = self._resolve_probe_parameters(model)
        param_refs = [param for _, param in params]
        if not param_refs:
            return {}

        metrics: Dict[str, float] = {}
        vectors: Dict[str, torch.Tensor] = {}
        start_s = time.perf_counter()

        with self._extra_grad_no_sync(model):
            for term_name, term in terms.items():
                metrics[f"gradmon/loss_raw/{term_name}"] = float(term.detach().float().cpu().item())
                metrics[f"gradmon/loss_ema_norm/{term_name}"] = float(
                    self._loss_ema_norm(term_name, term)
                )
                gradients = torch.autograd.grad(
                    term,
                    param_refs,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
                vectors[term_name] = self._flatten_gradients(
                    params=param_refs,
                    gradients=gradients,
                )

        norm_by_term: Dict[str, float] = {}
        valid_pair_terms = []
        total_vector = None
        eps = 1e-8

        for term_name, vector in vectors.items():
            norm_value = float(vector.norm(p=2).detach().cpu().item())
            norm_by_term[term_name] = float(norm_value)
            metrics[f"gradmon/grad_norm/{term_name}"] = float(norm_value)
            if total_vector is None:
                total_vector = vector.clone()
            else:
                total_vector = total_vector + vector
            if norm_value > eps:
                valid_pair_terms.append(term_name)

        if total_vector is None:
            return {}

        total_norm = float(total_vector.norm(p=2).detach().cpu().item())
        neg_cos_to_total = 0
        valid_total_terms = 0
        for term_name, vector in vectors.items():
            term_norm = float(norm_by_term.get(term_name, 0.0))
            if term_norm <= eps or total_norm <= eps:
                cos_to_total = 0.0
            else:
                cos_to_total = float(
                    torch.dot(vector, total_vector).detach().cpu().item()
                    / (float(term_norm) * float(total_norm) + eps)
                )
                valid_total_terms += 1
                if cos_to_total < 0.0:
                    neg_cos_to_total += 1
            metrics[f"gradmon/cos_to_total/{term_name}"] = float(cos_to_total)

        neg_pairs = 0
        valid_pairs = 0
        for i, left_name in enumerate(valid_pair_terms):
            left_vec = vectors[left_name]
            left_norm = float(norm_by_term[left_name])
            for right_name in valid_pair_terms[i + 1 :]:
                right_vec = vectors[right_name]
                right_norm = float(norm_by_term[right_name])
                cos_ij = float(
                    torch.dot(left_vec, right_vec).detach().cpu().item()
                    / (float(left_norm) * float(right_norm) + eps)
                )
                valid_pairs += 1
                if cos_ij < 0.0:
                    neg_pairs += 1

        norm_values = list(norm_by_term.values())
        if norm_values:
            norm_tensor = torch.tensor(norm_values, dtype=torch.float32)
            median_norm = float(norm_tensor.median().item())
            max_norm = float(norm_tensor.max().item())
            ratio = float(max_norm / max(median_norm, eps))
        else:
            ratio = 0.0

        neg_pair_frac = float(neg_pairs / valid_pairs) if valid_pairs > 0 else 0.0
        neg_cos_to_total_frac = (
            float(neg_cos_to_total / valid_total_terms) if valid_total_terms > 0 else 0.0
        )

        metrics["gradmon/grad_norm_ratio_max_over_median"] = float(ratio)
        metrics["gradmon/neg_cosine_pair_frac"] = float(neg_pair_frac)
        metrics["gradmon/neg_cosine_pair_pct"] = float(100.0 * neg_pair_frac)
        metrics["gradmon/neg_cos_to_total_frac"] = float(neg_cos_to_total_frac)
        metrics["gradmon/num_terms"] = float(len(terms))
        metrics["gradmon/shared_param_count"] = float(self._shared_param_count)
        metrics["gradmon/shared_param_numel"] = float(self._shared_param_numel)
        metrics["time/gradmon_s"] = float(time.perf_counter() - start_s)
        return metrics

    def _optimizer_step(self) -> int:
        state = getattr(self.trainer, "state", None)
        raw = getattr(state, "global_step", 0)
        try:
            return int(raw) + 1
        except (TypeError, ValueError):
            return 1

    def _sync_gradients_now(self) -> bool:
        override = getattr(self.trainer, "_loss_gradient_monitor_sync_gradients", None)
        if isinstance(override, bool):
            return bool(override)

        accelerator = getattr(self.trainer, "accelerator", None)
        sync_gradients = getattr(accelerator, "sync_gradients", None)
        if isinstance(sync_gradients, bool):
            return bool(sync_gradients)
        return True

    def _loss_ema_norm(self, term_name: str, term: torch.Tensor) -> float:
        raw = float(term.detach().float().cpu().item())
        prev = self._ema_state.get(term_name)
        current_abs = abs(float(raw))
        ema = current_abs if prev is None else (
            float(self.ema_beta) * float(prev) + (1.0 - float(self.ema_beta)) * current_abs
        )
        self._ema_state[term_name] = float(ema)
        return float(raw / (float(ema) + 1e-8))

    def _resolve_probe_parameters(
        self,
        model: Any,
    ) -> tuple[tuple[str, torch.nn.Parameter], ...]:
        if self._selected_params is not None:
            return self._selected_params

        selected = self._select_parameters(model)
        if not selected:
            raise RuntimeError("loss_gradient_monitor selected no shared probe parameters")

        self._selected_params = tuple(selected)
        self._shared_param_count = len(selected)
        self._shared_param_numel = int(sum(int(param.numel()) for _, param in selected))
        return self._selected_params

    def _select_parameters(
        self,
        model: Any,
    ) -> list[tuple[str, torch.nn.Parameter]]:
        named: list[tuple[str, torch.nn.Parameter]] = []
        seen: set[int] = set()
        for name, param in model.named_parameters():
            if not isinstance(param, torch.nn.Parameter):
                continue
            if not bool(param.requires_grad):
                continue
            ident = id(param)
            if ident in seen:
                continue
            seen.add(ident)
            named.append((str(name), param))

        if not named:
            return []

        if self.param_strategy == "regex":
            selected = []
            for name, param in named:
                if self.include_pattern is None or self.include_pattern.search(name) is None:
                    continue
                if self.exclude_pattern is not None and self.exclude_pattern.search(name):
                    continue
                selected.append((name, param))
            return self._apply_caps(selected)

        return self._apply_caps(self._select_auto_last_norm(named))

    def _select_auto_last_norm(
        self,
        named: Iterable[tuple[str, torch.nn.Parameter]],
    ) -> list[tuple[str, torch.nn.Parameter]]:
        named_list = list(named)
        prefix = None
        for name, _param in reversed(named_list):
            lname = name.lower()
            if any(token in lname for token in _AUTO_EXCLUDE_TOKENS):
                continue
            if not any(token in lname for token in _AUTO_NORM_TOKENS):
                continue
            prefix = name.rsplit(".", 1)[0]
            break

        if prefix is not None:
            selected = [
                (name, param)
                for name, param in named_list
                if name == prefix or name.startswith(f"{prefix}.")
            ]
            if selected:
                return selected

        fallback: list[tuple[str, torch.nn.Parameter]] = []
        for name, param in reversed(named_list):
            lname = name.lower()
            if any(token in lname for token in _AUTO_EXCLUDE_TOKENS):
                continue
            fallback.insert(0, (name, param))
            if len(fallback) >= 2:
                break
        return fallback

    def _apply_caps(
        self,
        selected: Iterable[tuple[str, torch.nn.Parameter]],
    ) -> list[tuple[str, torch.nn.Parameter]]:
        out: list[tuple[str, torch.nn.Parameter]] = []
        total_numel = 0
        for name, param in selected:
            if len(out) >= int(self.max_params):
                break
            next_numel = int(total_numel + int(param.numel()))
            if next_numel > int(self.max_numel):
                continue
            out.append((name, param))
            total_numel = next_numel
            if len(out) >= int(self.max_params) or total_numel >= int(self.max_numel):
                break
        return out

    def _extra_grad_no_sync(self, model: Any):
        accelerator = getattr(self.trainer, "accelerator", None)
        if accelerator is not None and hasattr(accelerator, "no_sync"):
            return accelerator.no_sync(model)

        no_sync = getattr(model, "no_sync", None)
        if callable(no_sync):
            return no_sync()
        return contextlib.nullcontext()

    @staticmethod
    def _flatten_gradients(
        *,
        params: Sequence[torch.nn.Parameter],
        gradients: Sequence[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        chunks = []
        for param, grad in zip(params, gradients):
            if grad is None:
                chunk = torch.zeros(
                    int(param.numel()),
                    device=param.device,
                    dtype=torch.float32,
                )
            else:
                chunk = grad.detach().to(dtype=torch.float32).reshape(-1)
            chunks.append(chunk)
        return torch.cat(chunks, dim=0)


def get_loss_gradient_monitor(trainer: Any) -> Optional[LossGradientMonitor]:
    cached = getattr(trainer, _MONITOR_ATTR, None)
    if cached is False:
        return None
    if isinstance(cached, LossGradientMonitor):
        return cached

    cfg = getattr(trainer, "loss_gradient_monitor_cfg", None)
    if not isinstance(cfg, Mapping) or not bool(cfg.get("enabled", False)):
        return None

    try:
        monitor = LossGradientMonitor.from_cfg(trainer=trainer, cfg=cfg)
    except Exception as exc:
        warn_once(
            trainer,
            key="loss_gradient_monitor_init_failed",
            message=(
                "LossGradientMonitor disabled after invalid configuration: "
                f"{type(exc).__name__}: {exc}"
            ),
            exc_info=True,
        )
        setattr(trainer, _MONITOR_ATTR, False)
        return None

    setattr(trainer, _MONITOR_ATTR, monitor)
    return monitor


__all__ = [
    "LossGradientMonitor",
    "build_stage1_coord_monitor_terms",
    "build_stage2_coord_monitor_terms_from_pipeline",
    "build_stage2_two_channel_coord_monitor_terms",
    "get_loss_gradient_monitor",
    "loss_gradient_monitor_enabled",
]
