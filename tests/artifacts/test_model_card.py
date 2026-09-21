import hashlib
import json

import pytest

from src.artifacts.model_card import package_model_card


def test_generated_card_preserves_exact_utf8_and_newlines(tmp_path):
    raw = "# 模型\r\n原始文本\r\n".encode()
    path = tmp_path / "README.md"
    path.write_bytes(raw)
    result = package_model_card(tmp_path)
    payload = json.loads(result.read_text())
    assert payload["content_utf8"].encode() == raw
    assert payload["sha256"] == hashlib.sha256(raw).hexdigest()
    assert not path.exists()
    assert package_model_card(tmp_path) is None


def test_conflict_retains_source_card(tmp_path):
    (tmp_path / "README.md").write_bytes(b"source")
    (tmp_path / "model_card.json").write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        package_model_card(tmp_path)
    assert (tmp_path / "README.md").read_bytes() == b"source"
    assert (tmp_path / "model_card.json").read_bytes() == b"existing"
