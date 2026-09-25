"""Imported mechanics are the public owner callables, not producer wrappers."""
import importlib

import pytest

from src.eval.native_rows import native_detection_record
from src.inference.bound_requests import build_bound_native_requests


@pytest.mark.parametrize("name", [
    "geometric_dedup_eval", "entrance_ce_eval", "selective_preservation_eval",
    "selective_preservation_dense_eval", "selective_preservation_wide_eval",
    "selective_preservation_strong_eval", "selective_preservation_seven_eval",
    "selective_preservation_stable_eval",
])
def test_evaluation_imports_exact_existing_mechanics(name):
    consumer = importlib.import_module(f"probes.dora_owner_learning.{name}")
    assert consumer.native_record is native_detection_record
    assert consumer.build_requests is build_bound_native_requests
