#!/usr/bin/env python3
"""Run the workspace Codex usage ledger without installing it globally."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    ledger_root = Path(
        os.environ.get(
            "CODEX_USAGE_LEDGER_ROOT", "/data/CoordExp/codex-usage-ledger"
        )
    ).resolve()
    package_root = ledger_root / "codex_usage_ledger"
    if not package_root.is_dir():
        print(f"ledger package not found: {package_root}", file=sys.stderr)
        return 2
    if not os.environ.get("CODEX_HOME"):
        print("set CODEX_HOME before running the ledger", file=sys.stderr)
        return 2
    sys.path.insert(0, str(ledger_root))
    from codex_usage_ledger.cli import main as ledger_main

    return ledger_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
