#!/usr/bin/env python
"""DEPRECATED: moved to scripts/_legacy/tools/check_mm_alignment.py

This file is a thin forwarder kept for backward compatibility.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    legacy = Path(__file__).resolve().parent / "_legacy/tools/check_mm_alignment.py"
    print(f"[DEPRECATED] scripts/check_mm_alignment.py moved to {legacy}", file=sys.stderr)
    runpy.run_path(str(legacy), run_name="__main__")


if __name__ == "__main__":
    main()
