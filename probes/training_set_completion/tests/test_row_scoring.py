"""Frozen saved-output characterization, not a model-quality experiment."""

import gzip
import json
from pathlib import Path

import pytest

from probes.training_set_completion.row_scoring import score


FIXTURE = Path(__file__).parent / "fixtures" / "row_scoring_v1.json.gz"
CASES = json.loads(gzip.decompress(FIXTURE.read_bytes()))["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_retained_scoring_is_exactly_preserved(case):
    actual = score(case["raw"], case["case"], case["bank"])
    # JSON normalization is the observable saved-result boundary (tuples/lists).
    assert json.loads(json.dumps(actual)) == case["expected"]
