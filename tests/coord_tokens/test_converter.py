import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.convert_to_coord_tokens import convert_list  # noqa: E402


def test_convert_list_rounds_like_ms_swift():
    out = convert_list([0, 0, 2, 2], width=4, height=4)
    assert out == ["<|coord_0|>", "<|coord_0|>", "<|coord_500|>", "<|coord_500|>"]


def test_convert_list_rejects_1000():
    with pytest.raises(AssertionError):
        convert_list([4, 4, 4, 4], width=4, height=4)
