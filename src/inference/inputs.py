"""Plan reusable generation prefixes and optional annotated targets, without weights."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.common.errors import EncodingContractError
from src.config.inference import InferConfig
from src.config.models import ProcessorConfig, TemplateConfig, TemplatePromptConfig
from src.data import RawExample
from src.inference.image_plan import ImagePlanRow, _row_from_encoding
from src.inference.prompt import PromptRecord, _prompt_from_rendered
from src.qwen.encoding import EncodedExample, encode_rendered_example
from src.qwen.images import plan_qwen_image
from src.qwen.native import NativeRequest
from src.qwen.runtime_loading import QwenComponents
from src.templates import RenderedExample, render_example


@dataclass(frozen=True)
class PlannedExample:
    rendered: RenderedExample
    image: ImagePlanRow
    prompt: PromptRecord
    request: NativeRequest
    target: EncodedExample | None


def processor_config(config: InferConfig) -> ProcessorConfig:
    return ProcessorConfig(
        do_resize=config.model.processor.do_resize,
        max_raw_pixels=1_000_000_000,
        max_merged_visual_tokens=1_000_000,
    )


def template_config(config: InferConfig) -> TemplateConfig:
    return TemplateConfig(
        object_field_order=config.template.object_field_order,
        object_ordering=config.template.object_ordering,
        assistant_format=config.template.assistant_format,
        prompt=TemplatePromptConfig(
            system=config.template.prompt.system, user=config.template.prompt.user,
        ),
    )


def plan_examples(
    rows: Sequence[RawExample],
    *,
    config: InferConfig,
    components: QwenComponents,
    row_indices: Sequence[int] | None = None,
    target_max_length: int | None = None,
) -> tuple[PlannedExample, ...]:
    """Render and plan each row once; callers explicitly materialize native requests.

    Targets include physical supervision spans in their full encoded sequence.
    They are separate from generation prefixes and do not select a loss.
    """
    rows = tuple(rows)
    if not rows or len({row.example_id for row in rows}) != len(rows):
        raise ValueError("input planning requires nonempty uniquely identified rows")
    indices = tuple(range(len(rows))) if row_indices is None else tuple(row_indices)
    if len(indices) != len(rows) or any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices
    ):
        raise ValueError("row indices must be nonnegative integers matching the selected rows")
    if target_max_length is not None and (
        isinstance(target_max_length, bool) or not isinstance(target_max_length, int)
        or target_max_length <= 0
    ):
        raise ValueError("annotated targets require a positive integer maximum length")
    template = template_config(config)
    processor = processor_config(config)
    planned = []
    for raw, index in zip(rows, indices, strict=True):
        rendered = render_example(raw, template, object_order_seed=config.template.object_order_seed)
        image_encoding = plan_qwen_image(raw, components=components, processor_config=processor)
        image = _row_from_encoding(index, image_encoding)
        prompt = _prompt_from_rendered(
            rendered, template, processor=components.processor, row_index=index,
            merged_visual_tokens=image.merged_visual_tokens,
        )
        request = NativeRequest(
            request_id=str(raw.example_id), chat_text=prompt.chat_text, image=image.image_path,
            expected_token_ids=tuple(prompt.expected_executed_prompt_token_ids),
            expected_image_grid=tuple(image.expected_image_grid_thw),
            expected_image_size=(image.decoded_width, image.decoded_height),
            image_sha256=image.image_content_sha256, logical_transform=image.logical_transform_id,
        )
        target = None
        if target_max_length is not None:
            target = encode_rendered_example(
                raw, rendered, components=components, processor_config=processor,
                global_max_length=target_max_length, materialize_image_pixels=False,
                _image_encoding=image_encoding,
            )
            start = target.supervised_token_spans[0].physical_token_start
            if tuple(target.input_ids[:start]) != request.expected_token_ids:
                raise EncodingContractError(
                    "annotated target prefix differs from the generation prompt",
                    code="inference.target_prompt_mismatch",
                    context={"example_id": raw.example_id},
                )
        planned.append(PlannedExample(rendered, image, prompt, request, target))
    return tuple(planned)
