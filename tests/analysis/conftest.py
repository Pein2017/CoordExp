from __future__ import annotations

from pathlib import Path

import pytest

from spatial_scope_history_fixtures import (
    install_execution_evidence_test_root,
    reset_execution_evidence_test_root,
)


@pytest.fixture(autouse=True)
def _execution_evidence_test_root(tmp_path: Path):
    token = install_execution_evidence_test_root(tmp_path)
    try:
        yield
    finally:
        reset_execution_evidence_test_root(token)
