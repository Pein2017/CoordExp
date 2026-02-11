from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _env_true(name: str) -> bool:
    value = str(os.environ.get(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_imports() -> None:
    repo_root = _resolve_repo_root()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_imports()

from src.common.repeat_terminate import (  # noqa: E402
    ForceEosOnRepeatSequenceGuard,
    RepeatTerminateConfig,
    encode_object_key_prefix,
    parse_repeat_terminate_config,
    should_trigger_repeat_terminate,
)

_ENV_ENABLE = "COORDEXP_ENABLE_VLLM_REPEAT_TERMINATE_INJECTION"
_ENV_CFG_JSON = "COORDEXP_VLLM_REPEAT_TERMINATE_CONFIG_JSON"
_ENV_CFG_JSON_PATH = "COORDEXP_VLLM_REPEAT_TERMINATE_CONFIG_JSON_PATH"
_PATCHED_SENTINEL = "_coordexp_repeat_terminate_patch_applied"


def _load_repeat_cfg_raw() -> Mapping[str, Any]:
    cfg_path = str(os.environ.get(_ENV_CFG_JSON_PATH, "") or "").strip()
    if cfg_path:
        path = Path(cfg_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, Mapping):
            raise RuntimeError(
                f"{_ENV_CFG_JSON_PATH} must point to a JSON object; got {type(data).__name__}."
            )
        return data

    cfg_json = str(os.environ.get(_ENV_CFG_JSON, "") or "").strip()
    if not cfg_json:
        return {}

    data = json.loads(cfg_json)
    if not isinstance(data, Mapping):
        raise RuntimeError(
            f"{_ENV_CFG_JSON} must decode to a JSON object; got {type(data).__name__}."
        )
    return data


def _as_plain_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        out = obj.model_dump()
        if isinstance(out, dict):
            return dict(out)
    if hasattr(obj, "dict") and callable(obj.dict):
        out = obj.dict()
        if isinstance(out, dict):
            return dict(out)
    raise TypeError(f"Cannot convert {type(obj).__name__} to dict")


def _extract_generated_token_ids(output_obj: Any) -> List[int]:
    response = getattr(output_obj, "response", output_obj)

    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            ch0 = choices[0]
            if isinstance(ch0, Mapping):
                token_ids = ch0.get("token_ids")
                if isinstance(token_ids, list):
                    return [int(x) for x in token_ids]
        return []

    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        return []

    ch0 = choices[0]
    token_ids = getattr(ch0, "token_ids", None)
    if not isinstance(token_ids, list):
        return []

    return [int(x) for x in token_ids]


def _patch_vllm_engine(
    *,
    cfg: RepeatTerminateConfig,
) -> None:
    from swift.llm.infer.infer_engine.grpo_vllm_engine import GRPOVllmEngine
    from swift.llm.infer.infer_engine.vllm_engine import VllmEngine

    original_prepare_generation_config = VllmEngine._prepare_generation_config

    def _patched_prepare_generation_config(self, request_config):
        sampling_params = original_prepare_generation_config(self, request_config)
        if not cfg.enabled:
            return sampling_params

        tokenizer = getattr(getattr(self, "default_template", None), "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "repeat_terminate.enabled=true but vLLM tokenizer is unavailable in server plugin"
            )

        eos_id = int(getattr(tokenizer, "eos_token_id", -1) or -1)
        if eos_id < 0:
            raise RuntimeError(
                "repeat_terminate.enabled=true but tokenizer has no eos_token_id"
            )

        obj_prefix_ids = (
            encode_object_key_prefix(tokenizer)
            if cfg.max_object_keys is not None
            else None
        )

        processor = ForceEosOnRepeatSequenceGuard(
            eos_token_id=eos_id,
            cfg=cfg,
            object_key_prefix_token_ids=obj_prefix_ids,
        )

        existing = getattr(sampling_params, "logits_processors", None)
        if existing is None:
            sampling_params.logits_processors = [processor]
        elif isinstance(existing, list):
            existing.append(processor)
            sampling_params.logits_processors = existing
        else:
            sampling_params.logits_processors = list(existing) + [processor]

        processors = getattr(self, "_coordexp_repeat_processors", None)
        if not isinstance(processors, list):
            processors = []
        processors.append(processor)
        setattr(self, "_coordexp_repeat_processors", processors)

        return sampling_params

    VllmEngine._prepare_generation_config = _patched_prepare_generation_config

    original_infer = GRPOVllmEngine.infer

    def _patched_infer(self, *args, **kwargs):
        setattr(self, "_coordexp_repeat_processors", [])
        outputs = original_infer(self, *args, **kwargs)
        if not cfg.enabled:
            return outputs

        tokenizer = getattr(getattr(self, "default_template", None), "tokenizer", None)
        obj_prefix_ids = (
            encode_object_key_prefix(tokenizer)
            if tokenizer is not None and cfg.max_object_keys is not None
            else None
        )

        processors = getattr(self, "_coordexp_repeat_processors", None)
        if not isinstance(processors, list):
            processors = []

        for idx, output in enumerate(outputs):
            triggered = 0
            if idx < len(processors):
                triggered = int(1 if bool(getattr(processors[idx], "triggered", False)) else 0)
            if not triggered:
                generated_ids = _extract_generated_token_ids(output)
                triggered = int(
                    1
                    if should_trigger_repeat_terminate(
                        generated_token_ids=generated_ids,
                        cfg=cfg,
                        object_key_prefix_token_ids=obj_prefix_ids,
                    )
                    else 0
                )

            rollout_infos = getattr(output, "rollout_infos", None)
            if isinstance(rollout_infos, Mapping):
                new_infos = dict(rollout_infos)
            else:
                new_infos = {}
            new_infos["repeat_terminate_triggered"] = int(triggered)
            setattr(output, "rollout_infos", new_infos)

        return outputs

    GRPOVllmEngine.infer = _patched_infer


def _patch_rollout_infer_response() -> None:
    from swift.llm.infer.rollout import SwiftRolloutDeploy

    original_infer = SwiftRolloutDeploy.infer

    async def _patched_infer(self, *args, **kwargs):
        outputs = await original_infer(self, *args, **kwargs)
        wrapped: List[Dict[str, Any]] = []

        for output in outputs:
            if isinstance(output, Mapping):
                response = output.get("response", output)
                rollout_infos = output.get("rollout_infos")
            else:
                response = getattr(output, "response", output)
                rollout_infos = getattr(output, "rollout_infos", None)

            repeat_triggered = 0
            if isinstance(rollout_infos, Mapping):
                try:
                    repeat_triggered = int(
                        rollout_infos.get("repeat_terminate_triggered", 0) or 0
                    )
                except Exception:
                    repeat_triggered = 0

            response_payload = _as_plain_dict(response)
            wrapped.append(
                {
                    "response": response_payload,
                    "coordexp": {
                        "repeat_terminate_triggered": int(1 if repeat_triggered else 0)
                    },
                }
            )

        return wrapped

    SwiftRolloutDeploy.infer = _patched_infer


def _activate_repeat_terminate_patch() -> None:
    if not _env_true(_ENV_ENABLE):
        return

    cfg = parse_repeat_terminate_config(_load_repeat_cfg_raw())
    if not cfg.enabled:
        return

    _patch_vllm_engine(cfg=cfg)
    _patch_rollout_infer_response()


def _ensure_patched_once() -> None:
    module = sys.modules[__name__]
    if bool(getattr(module, _PATCHED_SENTINEL, False)):
        return
    _activate_repeat_terminate_patch()
    setattr(module, _PATCHED_SENTINEL, True)


_ensure_patched_once()
