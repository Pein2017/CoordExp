from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "research" / "run_next_row_likelihood_change.py"
_SPEC = importlib.util.spec_from_file_location("next_row_likelihood_change", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _row(description: list[int], start: int = 1) -> list[int]:
    return [
        _MODULE.OBJECT_REF_START,
        *description,
        _MODULE.OBJECT_REF_END,
        _MODULE.BOX_START,
        *[_MODULE.COORDINATE_TOKEN_START + start + index for index in range(4)],
        _MODULE.BOX_END,
    ]


def _write_bundle(path: Path, *, prompt: list[int], generated: list[int], image_id: str = "img") -> None:
    path.write_text(
        json.dumps(
            {
                "decode_result": {
                    "prompt_token_ids": prompt,
                    "generated_token_ids": generated,
                },
                "execution_evidence": {"image_id": image_id},
            }
        ),
        encoding="utf-8",
    )


def _native_call_bundle(
    *,
    image_id: str = "img",
    source_image_sha256: str = "a" * 64,
    source_width: int = 10,
    source_height: int = 20,
) -> dict[str, object]:
    model_identity = {"family": "model", "adapter": "adapter"}
    tokenizer_identity = {"family": "tokenizer", "vocab": "v1"}
    attention_implementation = "sdpa"
    model_config_dtype = "bf16"
    return {
        "schema_version": _MODULE.NATIVE_CALL_BUNDLE_SCHEMA_VERSION,
        "image_id": image_id,
        "donor": {
            "source_image_sha256": source_image_sha256,
            "source_image_width": source_width,
            "source_image_height": source_height,
            "donor_lineage": {
                "image_id": image_id,
                "source_image_sha256": source_image_sha256,
                "source_width": source_width,
                "source_height": source_height,
                "model_identity_sha256": _MODULE.sha256_json(model_identity),
                "tokenizer_identity_sha256": _MODULE.sha256_json(tokenizer_identity),
                "attention_implementation": attention_implementation,
            },
        },
        "runtime": {
            "model_identity": model_identity,
            "tokenizer_identity": tokenizer_identity,
            "attention_implementation": attention_implementation,
            "model_config_dtype": model_config_dtype,
        },
        "decode_result": {
            "prompt_token_ids": [11, 12],
            "generated_token_ids": [],
            "model_identity": model_identity,
            "tokenizer_identity": tokenizer_identity,
            "execution_receipt": {
                "schema_version": "decode_execution_receipt.v1",
                "attention_implementation": attention_implementation,
            },
        },
    }


def _manifest(tmp_path: Path, *, prefix_count: int, candidate_span: list[int], candidate_hash: str | None = None) -> tuple[Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    prompt = [11, 12]
    prefix = _row([21])
    candidate = _row([31, 32], start=10)
    bundle_path = tmp_path / "parent.json"
    _write_bundle(bundle_path, prompt=prompt, generated=prefix + candidate)
    manifest = {
        "schema_version": _MODULE.MANIFEST_SCHEMA_VERSION,
        "image_id": "img",
        "parent": {
            "source_bundle": str(bundle_path),
            "source_bundle_sha256": _MODULE.sha256_file(bundle_path),
            "prefix_token_count": prefix_count,
            "prefix_token_ids_sha256": _MODULE.sha256_json(prefix[:prefix_count]),
        },
        "candidate_variants": [
            {
                "candidate_id": "candidate-1",
                "owner": "object-a",
                "category": "person",
                "role": "remaining",
                "source_bundle": str(bundle_path),
                "source_bundle_sha256": _MODULE.sha256_file(bundle_path),
                "source_token_span": candidate_span,
                "token_ids_sha256": candidate_hash or _MODULE.sha256_json(candidate),
                "description": "person",
                "geometry": [1, 2, 3, 4],
                "natural_support": {"source": "test-native", "count": 1, "policy": "greedy"},
            }
        ],
        "donors": [
            {
                "donor_id": "donor-1",
                "candidate_id": "candidate-1",
                "owner": "object-a",
                "category": "person",
                "role": "remaining",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest


def test_validate_manifest_requires_exact_contiguous_parent(tmp_path: Path) -> None:
    prefix = _row([21])
    candidate = _row([31, 32], start=10)
    manifest_path, _ = _manifest(
        tmp_path,
        prefix_count=len(prefix) - 1,
        candidate_span=[len(prefix) - 1, len(prefix) - 1 + len(candidate)],
    )
    with pytest.raises(ValueError, match="parent prefix must end"):
        _MODULE.validate_frozen_manifest(json.loads(manifest_path.read_text()), manifest_path=manifest_path)


def test_validate_manifest_rejects_row_hash_mismatch(tmp_path: Path) -> None:
    prefix = _row([21])
    candidate = _row([31, 32], start=10)
    manifest_path, _ = _manifest(
        tmp_path,
        prefix_count=len(prefix),
        candidate_span=[len(prefix), len(prefix) + len(candidate)],
        candidate_hash="0" * 64,
    )
    with pytest.raises(ValueError, match="token_ids_sha256 mismatch"):
        _MODULE.validate_frozen_manifest(json.loads(manifest_path.read_text()), manifest_path=manifest_path)


def test_validate_manifest_rejects_duplicate_ids_and_donor_metadata_mismatch(tmp_path: Path) -> None:
    prefix = _row([21])
    candidate = _row([31, 32], start=10)
    manifest_path, manifest = _manifest(
        tmp_path,
        prefix_count=len(prefix),
        candidate_span=[len(prefix), len(prefix) + len(candidate)],
    )
    manifest["candidate_variants"].append(dict(manifest["candidate_variants"][0]))
    manifest["candidate_variants"][1]["candidate_id"] = "candidate-1"
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        _MODULE.validate_frozen_manifest(manifest, manifest_path=manifest_path)

    manifest_path, manifest = _manifest(
        tmp_path / "metadata-mismatch",
        prefix_count=len(prefix),
        candidate_span=[len(prefix), len(prefix) + len(candidate)],
    )
    manifest["donors"][0]["category"] = "chair"
    with pytest.raises(ValueError, match="category disagrees"):
        _MODULE.validate_frozen_manifest(manifest, manifest_path=manifest_path)


def test_validate_active_source_lineage_rejects_non_native_bundle() -> None:
    with pytest.raises(ValueError, match="native sibling call-bundle schema"):
        _MODULE.validate_active_source_lineage(
            {"decode_result": {"prompt_token_ids": [1], "generated_token_ids": []}},
            image_id="img",
            image_sha256="abc",
            image_width=10,
            image_height=10,
            model_identity={"base": "x"},
            tokenizer_identity={"name": "x"},
            attention_implementation="sdpa",
            model_config_dtype="bf16",
        )


def test_validate_active_source_lineage_accepts_native_call_bundle_schema() -> None:
    bundle = _native_call_bundle()
    lineage = _MODULE.validate_active_source_lineage(
        bundle,
        image_id="img",
        image_sha256="a" * 64,
        image_width=10,
        image_height=20,
        model_identity={"family": "model", "adapter": "adapter"},
        tokenizer_identity={"family": "tokenizer", "vocab": "v1"},
        attention_implementation="sdpa",
        model_config_dtype="bf16",
    )
    assert lineage["image_id"] == "img"
    assert lineage["model_config_dtype"] == "bf16"
    assert lineage["execution_receipt_schema_version"] == "decode_execution_receipt.v1"


def _controlled_logits(row: list[int], boundary_length: int, shift: float) -> torch.Tensor:
    vocab = _MODULE.COORDINATE_TOKEN_END_EXCLUSIVE + 8
    logits = torch.zeros((boundary_length + len(row), vocab), dtype=torch.float32)
    for index, token in enumerate(row):
        logits[boundary_length + index - 1, token] = float(shift)
    return logits


def test_paired_delta_keeps_raw_signs_for_unequal_row_lengths() -> None:
    boundary = 3
    short = _row([31])
    long = _row([31, 32])
    short_before = _MODULE.score_token_logits(
        _controlled_logits(short, boundary, 0.0), boundary_length=boundary, row_tokens=short
    )
    short_after = _MODULE.score_token_logits(
        _controlled_logits(short, boundary, 2.0), boundary_length=boundary, row_tokens=short
    )
    long_before = _MODULE.score_token_logits(
        _controlled_logits(long, boundary, 0.0), boundary_length=boundary, row_tokens=long
    )
    long_after = _MODULE.score_token_logits(
        _controlled_logits(long, boundary, 2.0), boundary_length=boundary, row_tokens=long
    )
    short_delta = _MODULE.paired_score_delta(short_before, short_after)
    long_delta = _MODULE.paired_score_delta(long_before, long_after)
    assert short_delta["full_row"]["sum_delta"] > 0
    assert long_delta["full_row"]["sum_delta"] > 0
    assert short_delta["token_count"] != long_delta["token_count"]
    assert "probability_matrix" not in short_before
    assert "probability_matrix" not in long_before


def test_emitted_candidate_score_preserves_phase_mappings_for_paired_delta() -> None:
    row = _row([31, 32])
    candidate = {
        "candidate_id": "candidate-1",
        "owner": "object-a",
        "category": "person",
        "role": "remaining",
        "source_bundle": "bundle",
        "source_bundle_sha256": "b" * 64,
        "source_token_span": [0, len(row)],
        "source_prompt_ids_sha256": "p" * 64,
        "source_prompt_matches_base_prompt": True,
        "source_prompt_matches_reconstructed_parent": False,
        "token_ids": row,
        "description": "person",
        "geometry": [1, 2, 3, 4],
        "natural_support": {"source": "test", "count": 1, "policy": "greedy"},
        "token_count": len(row),
    }
    before = _MODULE.emit_candidate_score(
        _MODULE.score_token_logits(
            _controlled_logits(row, boundary_length=2, shift=0.0),
            boundary_length=2,
            row_tokens=row,
        ),
        candidate,
    )
    after = _MODULE.emit_candidate_score(
        _MODULE.score_token_logits(
            _controlled_logits(row, boundary_length=2, shift=2.0),
            boundary_length=2,
            row_tokens=row,
        ),
        candidate,
    )
    assert isinstance(before["description"], dict)
    assert isinstance(before["geometry"], dict)
    assert before["description_text"] == "person"
    assert before["geometry_xyxy"] == [1, 2, 3, 4]
    delta = _MODULE.paired_score_delta(before, after)
    assert delta["description"]["sum_delta"] > 0
    assert delta["geometry"]["sum_delta"] > 0


def test_terminal_margin_is_separate_from_full_row_score() -> None:
    row = _row([31])
    boundary = 2
    logits = _controlled_logits(row, boundary, 0.0)
    terminal_id = 42
    logits[boundary - 1, row[0]] = 2.0
    logits[boundary - 1, terminal_id] = 1.0
    score = _MODULE.score_token_logits(
        logits, boundary_length=boundary, row_tokens=row
    )
    candidate = {
        "candidate_id": "c",
        "owner": "o",
        "category": "person",
        "role": "remaining",
        "source_bundle": "bundle",
        "source_bundle_sha256": "hash",
        "source_token_span": [0, len(row)],
        "source_prompt_ids_sha256": "prompt-hash",
        "source_prompt_matches_base_prompt": True,
        "source_prompt_matches_reconstructed_parent": False,
        "token_ids": row,
        "description": "person",
        "geometry": [1, 2, 3, 4],
        "natural_support": {"source": "test", "count": 1, "policy": "greedy"},
        "token_count": len(row),
    }
    emitted = _MODULE.emit_candidate_score(score, candidate)
    margin = _MODULE.terminal_boundary_score(
        logits, boundary_length=boundary, row_entry_token_id=row[0], terminal_token_id=terminal_id
    )
    assert margin["row_entry_token_id"] == row[0]
    assert margin["terminal_token_id"] == terminal_id
    assert margin["row_entry_minus_terminal"] > 0
    assert "row_entry_vs_terminal" not in emitted
    assert "terminal" not in emitted["full_row"]
    assert margin["row_entry_minus_terminal"] != score["full_row"]["sum"]


def test_owner_logsumexp_exposes_constituents_deterministically() -> None:
    row = _row([31])
    items = [
        {"candidate_id": "b", "owner": "same", "token_ids_sha256": _MODULE.sha256_json(row), "full_row": {"sum": -2.0}},
        {"candidate_id": "a", "owner": "same", "token_ids_sha256": _MODULE.sha256_json(row), "full_row": {"sum": -1.0}},
    ]
    aggregate = _MODULE.aggregate_frozen_owner_scores(items)
    expected = float(torch.logsumexp(torch.tensor([-2.0, -1.0], dtype=torch.float32), dim=0).item())
    assert aggregate["variant_ids"] == ["b", "a"]
    assert aggregate["constituent_sums"] == [-2.0, -1.0]
    assert aggregate["logsumexp_sum"] == expected
    assert aggregate["aggregation"] == "logsumexp_over_frozen_variants"
    assert aggregate["constituents"][0]["candidate_id"] == "b"


def test_owner_aggregate_before_after_exposes_paired_delta() -> None:
    before = {
        "candidate_id": "a",
        "owner": "same",
        "token_ids_sha256": "a",
        "token_count": 9,
        "full_row": {"sum": -3.0},
    }
    after = {**before, "full_row": {"sum": -2.0}}
    before_aggregate = _MODULE.aggregate_frozen_owner_scores([before])
    after_aggregate = _MODULE.aggregate_frozen_owner_scores([after])
    delta = _MODULE.paired_owner_aggregate_delta(before_aggregate, after_aggregate)
    assert delta["sum_delta"] == 1.0
    assert delta["variant_ids_before"] == ["a"]


def test_unmatched_crossover_is_descriptive_only() -> None:
    short = _row([31])
    long = _row([31, 32])
    short_before = _MODULE.score_token_logits(_controlled_logits(short, 2, 0.0), boundary_length=2, row_tokens=short)
    short_after = _MODULE.score_token_logits(_controlled_logits(short, 2, 1.0), boundary_length=2, row_tokens=short)
    long_before = _MODULE.score_token_logits(_controlled_logits(long, 2, 0.0), boundary_length=2, row_tokens=long)
    long_after = _MODULE.score_token_logits(_controlled_logits(long, 2, 1.0), boundary_length=2, row_tokens=long)
    result = _MODULE.object_specific_crossover(
        donor_score=short_after,
        candidate_score=long_after,
        donor_before=short_before,
        candidate_before=long_before,
        donor_category="person",
        candidate_category="person",
        donor_owner="donor-owner",
        candidate_owner="candidate-owner",
    )
    assert result["token_length_matched"] is False
    assert result["conclusion_eligible"] is False
    assert result["comparison_scope"] == "descriptive_only"
    assert "descriptive_only" in result["full_row"]
    assert "token_length_mismatch" in result["ineligibility_reasons"]


def test_crossover_second_pass_handles_donor_ordered_after_first_candidate() -> None:
    boundary = 2
    first = _row([31])
    donor = _row([41])
    candidates = [
        {"candidate_id": "first", "category": "person", "owner": "first-owner"},
        {"candidate_id": "donor", "category": "person", "owner": "donor-owner"},
    ]
    before_scores = {
        "first": _MODULE.score_token_logits(
            _controlled_logits(first, boundary, 0.0), boundary_length=boundary, row_tokens=first
        ),
        "donor": _MODULE.score_token_logits(
            _controlled_logits(donor, boundary, 0.0), boundary_length=boundary, row_tokens=donor
        ),
    }
    after_scores = {
        "first": _MODULE.score_token_logits(
            _controlled_logits(first, boundary, 1.0), boundary_length=boundary, row_tokens=first
        ),
        "donor": _MODULE.score_token_logits(
            _controlled_logits(donor, boundary, 1.0), boundary_length=boundary, row_tokens=donor
        ),
    }
    crossover = _MODULE.compute_object_specific_crossover_scores(
        candidates,
        donor_candidate_id="donor",
        before_scores=before_scores,
        after_scores=after_scores,
    )
    assert list(crossover) == ["first", "donor"]
    assert crossover["first"]["conclusion_eligible"] is True
    assert crossover["donor"]["conclusion_eligible"] is False
    assert "same_physical_owner" in crossover["donor"]["ineligibility_reasons"]


def test_crossover_requires_distinct_physical_owner() -> None:
    boundary = 2
    row = _row([31])
    before = _MODULE.score_token_logits(
        _controlled_logits(row, boundary, 0.0), boundary_length=boundary, row_tokens=row
    )
    after = _MODULE.score_token_logits(
        _controlled_logits(row, boundary, 1.0), boundary_length=boundary, row_tokens=row
    )
    self_score = {**after, "candidate_id": "same-candidate"}
    same_owner_self = _MODULE.object_specific_crossover(
        donor_score=self_score,
        candidate_score=self_score,
        donor_before=before,
        candidate_before=before,
        donor_category="person",
        candidate_category="person",
        donor_owner="owner-a",
        candidate_owner="owner-a",
    )
    donor_score = {**after, "candidate_id": "donor-variant"}
    candidate_score = {**after, "candidate_id": "alternate-variant"}
    same_owner_variant = _MODULE.object_specific_crossover(
        donor_score=donor_score,
        candidate_score=candidate_score,
        donor_before=before,
        candidate_before=before,
        donor_category="person",
        candidate_category="person",
        donor_owner="owner-a",
        candidate_owner="owner-a",
    )
    distinct_owner = _MODULE.object_specific_crossover(
        donor_score=donor_score,
        candidate_score=candidate_score,
        donor_before=before,
        candidate_before=before,
        donor_category="person",
        candidate_category="person",
        donor_owner="owner-a",
        candidate_owner="owner-b",
    )
    assert same_owner_self["conclusion_eligible"] is False
    assert same_owner_self["comparison_scope"] == "descriptive_only"
    assert "same_physical_owner" in same_owner_self["ineligibility_reasons"]
    assert same_owner_variant["conclusion_eligible"] is False
    assert distinct_owner["conclusion_eligible"] is True
    assert distinct_owner["ineligibility_reasons"] == []
