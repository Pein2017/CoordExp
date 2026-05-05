from __future__ import annotations

from src.coord_tokens.offset_adapter import CoordOffsetAdapter
from src.tokens.row_offsets import (
    TokenRowOffsetAdapter,
    install_coord_offset_adapter,
    install_token_row_offset_adapter,
)


def test_row_offset_module_preserves_coord_offset_compatibility_names():
    assert TokenRowOffsetAdapter is CoordOffsetAdapter
    assert install_token_row_offset_adapter is install_coord_offset_adapter
