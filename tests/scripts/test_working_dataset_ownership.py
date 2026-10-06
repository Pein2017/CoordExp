"""The dataset authority must be usable without importing either application."""

import os
import subprocess
import sys
from pathlib import Path


def test_working_dataset_import_does_not_load_application_adapters():
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            """
import importlib.abc
import sys

class RejectApplications(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('src.coco_refinement', 'src.label_studio_coco_refinement')):
            raise AssertionError('dataset authority imported application: ' + fullname)

sys.meta_path.insert(0, RejectApplications())
from src.working_dataset.store import WorkingDatasetStore
from src.working_dataset.codec import canonical_json, semantic_hash
from src.working_dataset.models import WorkingRow
assert WorkingDatasetStore.__module__ == 'src.working_dataset.store'
assert canonical_json({'b': 2, 'a': 1}) == '{"a":1,"b":2}'
assert len(semantic_hash([])) == 64
""",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
