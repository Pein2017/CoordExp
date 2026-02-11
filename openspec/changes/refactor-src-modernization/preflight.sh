#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

PYTHONPATH=. conda run -n ms python -m py_compile src/infer/engine.py
PYTHONPATH=. conda run -n ms python -m pytest --collect-only -q tests/test_unified_infer_pipeline.py

