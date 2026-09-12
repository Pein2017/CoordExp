import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from probes.dora_owner_learning import inspect as inspection
from src.common.errors import DataContractError


@pytest.fixture
def rows():
    return tuple(SimpleNamespace(example_id=name) for name in ("a", "b", "c"))


def test_selection_preserves_requested_order_and_shared_membership(rows):
    selected = inspection.select_examples(rows, example_ids=("c", "a"))
    assert selected == (rows[2], rows[0])
    assert selected[0] is rows[2] and selected[1] is rows[0]
    assert inspection.select_examples(rows, count=2, seed=7) == (rows[1], rows[0])
    assert inspection.select_examples(rows, count=2, seed=7) == inspection.select_examples(rows, count=2, seed=7)


@pytest.mark.parametrize("kwargs", [
    {"example_ids": ("missing",)},
    {"example_ids": ("a", "a")},
    {"example_ids": ()},
    {"example_ids": "a"},
    {"example_ids": ("a",), "count": 1},
    {"example_ids": ("a",), "seed": 1},
    {"count": 4, "seed": 1},
    {"count": 0, "seed": 1},
    {"count": True, "seed": 1},
    {"count": 1},
    {"count": 1, "seed": True},
])
def test_selection_rejects_denominator_changes(rows, kwargs):
    with pytest.raises(ValueError):
        inspection.select_examples(rows, **kwargs)


def test_selection_rejects_duplicate_source_ids(rows):
    with pytest.raises(ValueError, match="source example IDs"):
        inspection.select_examples((rows[0], rows[0]), count=1, seed=0)


def test_bad_source_fails_before_processor_loading(tmp_path, monkeypatch):
    source = tmp_path / "rows.jsonl"
    source.write_text("{broken\n")

    def forbidden(*args, **kwargs):
        pytest.fail("invalid selection must not load the processor")

    monkeypatch.setattr(inspection, "assemble_frontend", forbidden)
    with pytest.raises(DataContractError):
        inspection.inspect_examples(input_path=source, count=1, seed=0)


def test_source_drift_fails_before_processor_loading(tmp_path, monkeypatch):
    source = tmp_path / "rows.jsonl"
    source.write_text("first source")

    def changed(path):
        path.write_text("different source")
        return ()

    monkeypatch.setattr(inspection, "load_raw_examples", changed)
    monkeypatch.setattr(inspection, "assemble_frontend", lambda *a, **k: pytest.fail("loaded processor"))
    with pytest.raises(ValueError, match="source changed"):
        inspection.inspect_examples(input_path=source, count=1, seed=0)


def test_real_inspection_cli_publishes_generation_and_target_views(tmp_path, monkeypatch):
    source = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/examples.jsonl")
    output = tmp_path / "inspection.json"
    monkeypatch.setattr(sys, "argv", [
        "inspect", "--input", str(source), "--count", "2", "--seed", "7",
        "--target-max-length", "12000", "--output", str(output),
    ])
    monkeypatch.setattr("src.qwen.runtime_loading._load_model_from_options",
                        lambda *a, **k: pytest.fail("inspection loaded model weights"))
    assert inspection.main() == 0
    result = json.loads(output.read_text())
    assert result["scope"] == "input-inspection-only"
    assert result["model_weights_loaded"] is False
    assert [row["example_id"] for row in result["rows"]] == result["selection"]["example_ids"]
    for row in result["rows"]:
        assert len(row["native_image_grid"]) == 3
        spans = row["target"]["supervised_token_spans"]
        start = spans[0]["physical_token_start"]
        assert row["target"]["input_ids"][:start] == row["native_prompt_token_ids"]
        assert spans[-1]["token_type"] == "eos"
        assert row["target"]["ignored_token_spans"][0]["physical_token_start"] == spans[-1]["physical_token_end"]
