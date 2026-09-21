"""Rebuild native requests from already-bound research cases without replanning them.

This adapter is intentionally narrower than :func:`src.inference.inputs.plan_examples`.
The caller already owns the frozen image plan and input row.  We only rebuild the
prompt/request objects against the current processor and fail if the bound prompt
width or image identity no longer agrees.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.config.models import TemplateConfig
from src.data.examples import raw_example_from_jsonl_row
from src.inference.prompt import build_prompt_record
from src.qwen.native import NativeRequest


def build_bound_native_requests(
    qwen: Any,
    config: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> tuple[list[NativeRequest], list[dict[str, Any]]]:
    """Build native requests while treating each case's image plan as authority."""

    template = config["template"]
    template_config = TemplateConfig(
        **{
            key: template[key]
            for key in (
                "object_field_order",
                "object_ordering",
                "assistant_format",
                "prompt",
            )
        }
    )
    requests: list[NativeRequest] = []
    metadata: list[dict[str, Any]] = []
    for case in cases:
        raw = raw_example_from_jsonl_row(
            case["input_record"],
            jsonl_path=Path(config["data"]["input_jsonl"]),
            row_number=int(case["row_index"]) + 1,
            raw_line=json.dumps(case["input_record"]),
        )
        if str(raw.example_id) != case["row_id"]:
            raise ValueError("case input row identity differs")
        plan = case["image_plan"]
        prompt = build_prompt_record(
            raw,
            template_config,
            processor=qwen.processor,
            row_index=int(case["row_index"]),
            merged_visual_tokens=int(plan["merged_visual_tokens"]),
            object_order_seed=template.get("object_order_seed"),
        )
        if len(prompt.expected_executed_prompt_token_ids) != plan["backend_prompt_token_count"]:
            raise ValueError("historical prompt width differs")
        requests.append(
            NativeRequest(
                case["row_id"],
                prompt.chat_text,
                case["image_path"],
                expected_token_ids=tuple(prompt.expected_executed_prompt_token_ids),
                expected_image_grid=tuple(plan["observed_image_grid_thw"]),
                expected_image_size=(case["image_width"], case["image_height"]),
                image_sha256=plan["image_content_sha256"],
                logical_transform=plan["logical_transform_id"],
            )
        )
        metadata.append(prompt.to_artifact_dict())
    return requests, metadata
