#!/usr/bin/env python3
"""
Shim to keep backward compatibility.
Delegates to public_data/scripts/convert_to_coord_tokens.py.
"""

from typing import List, Sequence

from public_data.scripts.convert_to_coord_tokens import (
    main,
    normalize_list,
)
from src.coord_tokens.codec import int_to_token


def convert_list(values: Sequence, *, width: float, height: float) -> List[str]:
    """Convert a geometry list into coord tokens (ms-swift rounding compatible).

    Unit tests import this helper from `scripts.convert_to_coord_tokens`, so keep it
    here even though the CLI implementation lives under `public_data/scripts/`.
    """

    ints = normalize_list(values, width=width, height=height, assume_normalized=False)
    return [int_to_token(v) for v in ints]

if __name__ == "__main__":
    main()
