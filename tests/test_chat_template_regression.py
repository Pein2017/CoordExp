from __future__ import annotations

import json
from pathlib import Path

import pytest


def _load_jsonl_record(path: Path, index: int = 0) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"Index {index} out of range for {path}")


@pytest.fixture(scope="module")
def _qwen3vl_processor():
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = repo_root / "model_cache/models/Qwen/Qwen3-VL-4B-Instruct-coordexp"
    if not model_dir.exists():
        pytest.skip(f"missing local model processor under {model_dir}")

    from transformers import AutoProcessor  # type: ignore

    return AutoProcessor.from_pretrained(str(model_dir))


@pytest.mark.parametrize("object_field_order", ["desc_first", "geometry_first"])
def test_chat_template_renders_coordjson_and_tokenizes(
    _qwen3vl_processor,
    object_field_order: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    jsonl_path = repo_root / "public_data/coco/rescale_32_768_bbox_max60/val.coord.jsonl"
    if not jsonl_path.exists():
        pytest.skip(f"missing JSONL fixture under {jsonl_path}")

    record = _load_jsonl_record(jsonl_path, 0)

    from src.config.prompts import get_template_prompts
    from src.coord_tokens.codec import get_coord_token_ids
    from src.datasets.builders import JSONLinesBuilder
    from src.utils.coordjson_transpiler import coordjson_to_strict_json_with_meta

    system_prompt, user_prompt = get_template_prompts(
        ordering="sorted",
        coord_mode="coord_tokens",
        prompt_variant="coco_80",
        object_field_order=object_field_order,
    )

    builder = JSONLinesBuilder(
        user_prompt=user_prompt,
        emit_norm="norm1000",
        coord_tokens_enabled=True,
        object_field_order=object_field_order,
    )
    merged = builder.build(record)

    assistant_text = str(merged["messages"][1]["content"][0]["text"])
    assert assistant_text.startswith('{"objects": [')
    assert "<|coord_" in assistant_text
    assert '"<|coord_' not in assistant_text

    if object_field_order == "desc_first":
        first = assistant_text.find('{"desc":')
        assert first >= 0
        bbox_pos = assistant_text.find('"bbox_2d":', first)
        assert bbox_pos >= 0
        assert first < bbox_pos
    else:
        first = assistant_text.find('{"bbox_2d":')
        assert first >= 0
        desc_pos = assistant_text.find('"desc":', first)
        assert desc_pos >= 0
        assert first < desc_pos

    strict_text, meta = coordjson_to_strict_json_with_meta(
        assistant_text,
        mode="salvage",
        object_field_order=object_field_order,
    )
    assert bool(meta.parse_failed) is False

    payload = json.loads(strict_text)
    assert isinstance(payload, dict)
    objects = payload.get("objects")
    assert isinstance(objects, list)
    if objects:
        obj0 = objects[0]
        assert isinstance(obj0, dict)
        bbox = obj0.get("bbox_2d")
        if bbox is not None:
            assert isinstance(bbox, list)
            assert all(isinstance(x, int) for x in bbox)

    messages_sys = [{"role": "system", "content": system_prompt}, *merged["messages"]]
    chat_text = _qwen3vl_processor.apply_chat_template(
        messages_sys, tokenize=False, add_generation_prompt=False
    )
    assert "<|im_start|>system" in chat_text
    assert "<|im_start|>user" in chat_text
    assert "<|im_start|>assistant" in chat_text
    assert "<|vision_start|><|image_pad|><|vision_end|>" in chat_text

    # Stage-2 rollout style prompt: system + user, then a generation prompt that
    # opens an assistant turn without closing it via <|im_end|>.
    messages_gen = [
        {"role": "system", "content": system_prompt},
        merged["messages"][0],
    ]
    chat_text_gen = _qwen3vl_processor.apply_chat_template(
        messages_gen, tokenize=False, add_generation_prompt=True
    )
    last_assistant = chat_text_gen.rfind("<|im_start|>assistant")
    assert last_assistant >= 0
    assert "<|im_end|>" not in chat_text_gen[last_assistant:]

    tokenized = _qwen3vl_processor.tokenizer(chat_text, add_special_tokens=False)
    input_ids = tokenized["input_ids"]
    assert isinstance(input_ids, list)
    assert input_ids
    im_start_id = _qwen3vl_processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    assert int(input_ids[0]) == int(im_start_id)

    # Tokenizer contract: coord tokens are present and map to distinct ids.
    coord_ids = get_coord_token_ids(_qwen3vl_processor.tokenizer, validate=True)
    assert len(coord_ids) == 1000
