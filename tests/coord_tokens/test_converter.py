import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONVERTER_PATH = ROOT / "public_data/scripts/convert_to_coord_tokens.py"
_converter_spec = importlib.util.spec_from_file_location(
    "public_data_convert_to_coord_tokens",
    CONVERTER_PATH,
)
assert _converter_spec is not None and _converter_spec.loader is not None
_converter = importlib.util.module_from_spec(_converter_spec)
_converter_spec.loader.exec_module(_converter)
convert_list = _converter.convert_list


def test_convert_list_rounds_like_ms_swift():
    out = convert_list([0, 0, 2, 2], width=4, height=4)
    assert out == ["<|coord_0|>", "<|coord_0|>", "<|coord_666|>", "<|coord_666|>"]


def test_convert_list_rejects_1000():
    with pytest.raises(AssertionError):
        convert_list([4, 4, 4, 4], width=4, height=4)
