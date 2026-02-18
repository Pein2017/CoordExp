#!/usr/bin/env python
"""
Inspect how a JSONL sample plus the CoordExp prompts are rendered by the Qwen3-VL
chat template and tokenizer.

Example:
    python scripts/tools/inspect_chat_template.py \\
        --jsonl public_data/lvis/rescale_32_768_poly_20/train.jsonl \\
        --index 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoProcessor  # type: ignore

from src.config.prompts import get_template_prompts
from src.datasets.builders import JSONLinesBuilder
from src.utils.coordjson_transpiler import coordjson_to_strict_json_with_meta


def load_record(path: Path, index: int) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"Index {index} out of range for {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        type=Path,
        required=True,
        help="Path to a CoordExp JSONL sample",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based line index to load from the JSONL (default: 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model_cache/Qwen3-VL-8B-Instruct-coordexp",
        help="Model/processor path for chat template rendering",
    )
    parser.add_argument(
        "--token-head",
        type=int,
        default=80,
        help="How many token ids to print from the start (default: 80)",
    )
    parser.add_argument(
        "--object-field-order",
        type=str,
        default="desc_first",
        choices=["desc_first", "geometry_first"],
        help="Per-object key order for assistant payload serialization.",
    )
    parser.add_argument(
        "--object-ordering",
        type=str,
        default="sorted",
        choices=["sorted", "random"],
        help="Object instance ordering instruction used when building prompts.",
    )
    parser.add_argument(
        "--prompt-variant",
        type=str,
        default=None,
        help="Optional dense prompt variant key (for example: coco_80).",
    )
    args = parser.parse_args()

    record = load_record(args.jsonl, args.index)

    system_prompt, user_prompt = get_template_prompts(
        ordering=args.object_ordering,
        coord_mode="coord_tokens",
        prompt_variant=args.prompt_variant,
        object_field_order=args.object_field_order,
    )

    builder = JSONLinesBuilder(
        user_prompt=user_prompt,
        emit_norm="norm1000",
        coord_tokens_enabled=True,
        object_field_order=args.object_field_order,
    )
    merged = builder.build(record)
    messages = merged["messages"]
    messages_sys = [{"role": "system", "content": system_prompt}, *messages]

    # Render text with the model's native (jinja) chat template
    processor = AutoProcessor.from_pretrained(args.model)
    chat_text = processor.apply_chat_template(
        messages_sys, tokenize=False, add_generation_prompt=False
    )
    tokenized = processor.tokenizer(chat_text, add_special_tokens=False)

    assistant_text = str(messages[1]["content"][0]["text"])
    strict_text, strict_meta = coordjson_to_strict_json_with_meta(
        assistant_text,
        mode="salvage",
        object_field_order=args.object_field_order,
    )

    print("=== USER TEXT ===")
    for c in messages[0]["content"]:
        if isinstance(c, dict) and "text" in c:
            print(c["text"])
    print("\n=== ASSISTANT COORDJSON TEXT ===")
    print(assistant_text)
    print("\n=== ASSISTANT STRICT JSON VIEW (transpiled) ===")
    print(strict_text)
    print("\n=== TRANSPILE META ===")
    print(
        json.dumps(
            {
                "parse_failed": bool(strict_meta.parse_failed),
                "truncated": bool(strict_meta.truncated),
                "dropped_invalid_records": int(strict_meta.dropped_invalid_records),
                "dropped_invalid_by_reason": dict(strict_meta.dropped_invalid_by_reason),
                "dropped_incomplete_tail": int(strict_meta.dropped_incomplete_tail),
            },
            ensure_ascii=False,
            separators=(", ", ": "),
        )
    )
    print("\n=== CHAT TEMPLATE (exact text fed to tokenizer) ===")
    print(chat_text)
    print("\n=== TOKEN IDS (head) ===")
    head = tokenized["input_ids"][: args.token_head]
    print(head)
    print("\n=== DECODED HEAD ===")
    print(processor.tokenizer.decode(head, skip_special_tokens=False))


if __name__ == "__main__":
    main()
