#!/usr/bin/env python3
"""CLI for current knowledge and optional historical exposure checks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.tools.research_knowledge import main

if __name__ == '__main__':
    raise SystemExit(main())
