#!/usr/bin/env python
"""DEPRECATED: moved to scripts/_legacy/tools/sample_jsonl.py

This file is a thin forwarder kept for backward compatibility.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    legacy = Path(__file__).resolve().parent / "_legacy/tools/sample_jsonl.py"
    print(f"[DEPRECATED] scripts/sample_jsonl.py moved to {legacy}", file=sys.stderr)
    runpy.run_path(str(legacy), run_name="__main__")


if __name__ == "__main__":
    main()
