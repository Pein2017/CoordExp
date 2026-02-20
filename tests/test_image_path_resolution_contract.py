from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from src.config.schema import CoordTokensConfig
from src.datasets.dense_caption import BaseCaptionDataset


class _FakeTemplate:
    max_pixels = 786432
    system = None

    def encode(self, merged: Dict[str, Any], return_length: bool = True) -> Dict[str, Any]:
        return {"input_ids": [0], "labels": [0], "length": 1}


def _write_img(path: Path, *, size: int = 32) -> None:
    img = Image.new("RGB", (size, size), color=(128, 128, 128))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def test_jsonl_image_paths_resolve_relative_to_jsonl_dir_not_root_image_dir(
    tmp_path: Path, monkeypatch
) -> None:
    # Arrange: JSONL stores image as a relative path (contract).
    img_path = tmp_path / "images" / "img_0.png"
    _write_img(img_path)

    jsonl_path = tmp_path / "train.jsonl"
    record = {
        "images": ["images/img_0.png"],
        "width": 32,
        "height": 32,
        "objects": [
            {
                "desc": "obj",
                "bbox_2d": ["<|coord_0|>", "<|coord_0|>", "<|coord_1|>", "<|coord_1|>"],
            }
        ],
    }
    jsonl_path.write_text(json.dumps(record, ensure_ascii=True) + "\n", encoding="utf-8")

    # If ROOT_IMAGE_DIR were incorrectly used by the learner load path, this would break.
    monkeypatch.setenv("ROOT_IMAGE_DIR", str(tmp_path / "wrong_root"))

    ds = BaseCaptionDataset.from_jsonl(
        str(jsonl_path),
        template=_FakeTemplate(),
        user_prompt="prompt",
        emit_norm="none",
        json_format="standard",
        coord_tokens=CoordTokensConfig(enabled=True, skip_bbox_norm=True),
    )

    sample = ds[0]
    messages = sample["messages"]
    assert messages and messages[0]["role"] == "user"
    user_contents = messages[0]["content"]
    assert user_contents and user_contents[0]["type"] == "image"

    resolved = user_contents[0]["image"]
    assert isinstance(resolved, str)
    assert os.path.isabs(resolved)
    assert Path(resolved).resolve() == img_path.resolve()

