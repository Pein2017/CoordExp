from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "run_complete_candidate_row_scoring.py"
)
_SPEC = importlib.util.spec_from_file_location("complete_candidate_row_scoring", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _row(description: list[int], start: int = 0) -> list[int]:
    return [
        _MODULE.OBJECT_REF_START,
        *description,
        _MODULE.OBJECT_REF_END,
        _MODULE.BOX_START,
        *[_MODULE.COORDINATE_TOKEN_START + start + index for index in range(4)],
        _MODULE.BOX_END,
    ]


def _logits(row: list[int], boundary_length: int, vocab: int | None = None) -> torch.Tensor:
    vocab = vocab or (_MODULE.COORDINATE_TOKEN_END_EXCLUSIVE + 8)
    values = torch.zeros((boundary_length + len(row), vocab), dtype=torch.float32)
    for index, token in enumerate(row):
        values[boundary_length + index - 1, token] = 2.0
    return values


def test_literal_prefix_and_candidate_hashes_are_validated() -> None:
    prefix = [11, 12]
    row = _row([31, 32], start=4)
    manifest = {
        "schema_version": _MODULE.MANIFEST_SCHEMA_VERSION,
        "unit_id": "test",
        "images": [
            {
                "image_id": "image-a",
                "boundaries": [
                    {
                        "boundary_id": "b0",
                        "prefix": {"token_ids": prefix, "token_ids_sha256": _MODULE.sha256_json(prefix)},
                        "candidates": [
                            {
                                "candidate_id": "row-a",
                                "row": {"token_ids": row, "token_ids_sha256": _MODULE.sha256_json(row)},
                                "owner": "object-a",
                                "category": "person",
                                "role": "remaining",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    normalized = _MODULE.validate_manifest(manifest, manifest_path=Path("/tmp/test-manifest.json"))
    assert normalized["images"][0]["boundaries"][0]["prefix_token_ids"] == prefix
    assert normalized["images"][0]["boundaries"][0]["candidates"][0]["row_token_ids"] == row

    bad = json.loads(json.dumps(manifest))
    bad["images"][0]["boundaries"][0]["candidates"][0]["row"]["token_ids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="literal token hash mismatch"):
        _MODULE.validate_manifest(bad, manifest_path=Path("/tmp/test-manifest.json"))


def test_image_filter_preserves_manifest_order_and_rejects_unknown_ids() -> None:
    manifest = {"images": [{"image_id": "image-a"}, {"image_id": "image-b"}, {"image_id": "image-c"}]}
    selected = _MODULE.select_manifest_images(manifest, ["image-c", "image-a"])
    assert [item["image_id"] for item in selected] == ["image-a", "image-c"]
    assert [item["image_id"] for item in _MODULE.select_manifest_images(manifest)] == ["image-a", "image-b", "image-c"]
    with pytest.raises(ValueError, match=r"requested image_id\(s\) absent"):
        _MODULE.select_manifest_images(manifest, ["missing-image"])


def test_compact_stage2_selector_resolves_and_checks_hash(tmp_path: Path) -> None:
    row = _row([41], start=5)
    artifact = {
        "base_prompt": {"prompt_token_ids": [101, 102]},
        "cases": [
            {
                "case_id": "case-a",
                "arms": {
                    "arm-a": {
                        "runs": [
                            {
                                "comparison_id": "comparison-a",
                                "mode": "greedy",
                                "seed": None,
                                "rows": [{"prefix_token_ids": [7, 8], "raw_generated_token_ids": row}],
                            }
                        ]
                    }
                },
            }
        ],
    }
    artifact_path = tmp_path / "stage2.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    selector = {
        "selector": {
            "source_artifact": str(artifact_path),
            "case_id": "case-a",
            "arm_name": "arm-a",
            "comparison_id": "comparison-a",
            "mode": "greedy",
            "seed": None,
            "row_index": 0,
            "token_field": "row_generated_token_ids",
        },
        "token_ids_sha256": _MODULE.sha256_json(row),
    }
    resolved, info = _MODULE.resolve_token_reference(selector, manifest_path=tmp_path / "manifest.json", context="candidate")
    assert resolved == row
    assert info["kind"] == "stage2_selector"
    assert info["token_field"] == "row_generated_token_ids"


def test_selected_coordinate_ranks_are_computed_from_float32_logits() -> None:
    boundary = 2
    row = _row([31, 32], start=10)
    logits = _logits(row, boundary)
    # At x1, three vocabulary entries are strictly larger than the selected
    # coordinate; ties must not count as higher-ranked tokens.
    x1_position = boundary + 5 - 1
    x1_token = row[5]
    logits[x1_position, x1_token] = 1.0
    logits[x1_position, 12] = 2.0
    logits[x1_position, 13] = 3.0
    logits[x1_position, 14] = 1.0
    ranks = _MODULE.selected_token_ranks(logits, boundary_length=boundary, row_tokens=row)
    assert ranks["row_entry"] == [1]
    assert ranks["description"] == [1, 1]
    assert ranks["x1"] == [3]
    assert ranks["y1"] == [1]
    assert ranks["x2"] == [1]
    assert ranks["y2"] == [1]
    assert ranks["closure"] == [1]
    assert len(ranks["full_row"]) == len(row)


def test_terminal_score_is_separate_from_raw_candidate_row_score() -> None:
    boundary = 3
    row = _row([51], start=8)
    logits = _logits(row, boundary)
    row_score = _MODULE.score_token_logits(logits, boundary_length=boundary, row_tokens=row)
    terminal = _MODULE.terminal_boundary_score(
        logits,
        boundary_length=boundary,
        row_entry_token_id=_MODULE.OBJECT_REF_START,
        terminal_token_id=17,
    )
    assert "row_entry_vs_terminal" not in row_score
    assert "full_row" in row_score
    assert "row_entry_minus_terminal" in terminal
    assert "probability_matrix" not in row_score


def test_raw_score_matches_selected_float32_log_softmax_sum() -> None:
    boundary = 2
    row = _row([61], start=1)
    logits = _logits(row, boundary)
    score = _MODULE.score_token_logits(logits, boundary_length=boundary, row_tokens=row)
    expected = torch.log_softmax(logits.to(dtype=torch.float32), dim=-1)
    expected_sum = sum(float(expected[boundary + index - 1, token]) for index, token in enumerate(row))
    assert score["full_row"]["sum"] == pytest.approx(expected_sum, abs=1e-6)
    assert score["full_row"]["count"] == len(row)
    assert score["token_count"] == len(row)
