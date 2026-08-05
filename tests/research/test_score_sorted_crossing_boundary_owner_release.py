"""Contracts for the not-yet-implemented crossing-boundary owner
release/realization scorer (``scripts/research/score_sorted_crossing_boundary_
owner_release.py``), authored against the frozen unit:

``research/investigations/qwen3-vl-dense-enumeration/experiments/
2026-08-03-sorted-crossing-boundary-owner-release-realization/unit.md``

This file is failing-first: the production module does not exist yet, so
the whole file is expected to fail at *collection* (``ModuleNotFoundError``)
until it is authored. Every test below states, via a small pure helper or a
frozen dataclass, the exact contract the production module must satisfy so
that authoring it can proceed test-first.

Design discipline mirrored from sibling scorers in this package
(``score_sorted_owner_basin_landscape.py``, ``score_sorted_fn_fixed_budget.py``,
``score_sorted_owner_accessibility_census_shard.py``):

- literal token ids/digests only, never re-tokenizable text
  (``FORBIDDEN_TEXT_KEYS``);
- deterministic, explicit-step scoring/decoding -- never ``model.generate()``,
  never sampling, never an analyzer/classification call, never a GPU or a
  real model/tokenizer. Every fixture below is a plain Python value or an
  injected pure callable standing in for a per-step argmax/logit readout;
- fail-closed on malformed/ambiguous input rather than silent repair; and
- deterministic sha256 digests over canonical JSON for request/output
  identity.

Not covered here (out of this test file's scope; reused unmodified by the
eventual production module rather than re-tested): the physical-owner box
matcher itself (the unit names it as "the predecessor's one-row strict
physical-owner matcher"), the live HF/GPU scoring backend, and the CPU plan
builder that seals the 26-owner cohort registry. Those seams are exercised
as pre-computed inputs (``greedy_status``, ``greedy_owner_match``,
``target_minus_native_margin``, ...) below.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from scripts.research import score_sorted_crossing_boundary_owner_release as sut


# ---------------------------------------------------------------------------
# Shared local helpers (deliberately duplicated per this codebase's
# self-contained-test-file convention; mirrors sha256_json in every sibling
# scorer/test pair).
# ---------------------------------------------------------------------------


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


COORDINATE_DOMAIN = range(151670, 152670)
BOX_END = 151649
OBJECT_REF_START = 151646
OBJECT_REF_END = 151647
BOX_START = 151648


# ---------------------------------------------------------------------------
# 0. Frozen module-level constants (schema identity + the exact cohort/
#    control/threshold facts named by unit.md; regression-proofs against
#    silent drift, including the explicit "14, not the stale 21" fact).
# ---------------------------------------------------------------------------


def test_schema_and_unit_identity_constants() -> None:
    assert sut.SCHEMA_VERSION == "sorted_crossing_boundary_owner_release_scores.v1"
    assert (
        sut.RECEIPT_SCHEMA_VERSION
        == "sorted_crossing_boundary_owner_release_scores_receipt.v1"
    )
    assert (
        sut.UNIT_ID
        == "2026-08-03-sorted-crossing-boundary-owner-release-realization"
    )


def test_coordinate_and_wrapper_token_constants_match_frozen_domain() -> None:
    assert sut.OBJECT_REF_START == OBJECT_REF_START
    assert sut.OBJECT_REF_END == OBJECT_REF_END
    assert sut.BOX_START == BOX_START
    assert sut.BOX_END == BOX_END
    assert sut.COORDINATE_TOKEN_ID_START == COORDINATE_DOMAIN.start
    assert sut.COORDINATE_TOKEN_ID_END_EXCLUSIVE == COORDINATE_DOMAIN.stop


def test_frozen_primary_cohort_denominators() -> None:
    # unit.md "Exact state pair": U=26 primary cohort, L=25 sensitivity,
    # same-context U/L=24 sensitivity, 12/26 matched-E, 14/26 unmatched-E.
    assert sut.PRIMARY_OWNER_COUNT_U == 26
    assert sut.PRIMARY_OWNER_COUNT_L == 25
    assert sut.PRIMARY_OWNER_COUNT_SAME_CONTEXT_UL == 24
    assert sut.MATCHED_E_OWNER_COUNT == 12
    assert sut.UNMATCHED_E_OWNER_COUNT == 14
    assert sut.MATCHED_E_OWNER_COUNT + sut.UNMATCHED_E_OWNER_COUNT == sut.PRIMARY_OWNER_COUNT_U


def test_timing_control_count_is_14_not_the_stale_21() -> None:
    # unit.md "Timing controls": "The audit reconstruction expects about 14
    # owners". A prior scout note said 21; that value must never reappear.
    assert sut.TIMING_CONTROL_OWNER_COUNT == 14
    assert sut.TIMING_CONTROL_OWNER_COUNT != 21


def test_tp_calibration_owner_count_is_twelve() -> None:
    # unit.md "Timing controls": "must yield twelve fixed owners".
    assert sut.TP_CALIBRATION_OWNER_COUNT == 12


def test_max_primary_quarantines_is_two() -> None:
    # unit.md "Native replay alignment": "More than two quarantined primary
    # owners stops the unit before interpretation."
    assert sut.MAX_PRIMARY_QUARANTINES == 2


def test_cache_parity_threshold_is_1e_minus_3() -> None:
    # unit.md "Native replay alignment": "maximum selected-logit absolute
    # difference at most 1e-3".
    assert sut.CACHE_PARITY_MAX_SELECTED_LOGIT_ABS_DIFF == pytest.approx(1e-3)


def test_interpretability_gate_thresholds() -> None:
    # unit.md "Stop rule": ">= 20/26 interpretable, >= 6 matched-E, >= 7
    # unmatched-E".
    assert sut.MIN_INTERPRETABLE_OWNERS == 20
    assert sut.MIN_INTERPRETABLE_MATCHED_E == 6
    assert sut.MIN_INTERPRETABLE_UNMATCHED_E == 7


def test_primary_branch_and_smoke_role_enums() -> None:
    assert sut.PRIMARY_BRANCHES == frozenset(
        {"displaced", "release_lost", "realization_fail", "ambiguous"}
    )
    assert sut.DISPLACED_SUBTAGS == frozenset(
        {"likelihood_displaced", "greedy_displaced"}
    )
    assert sut.REQUIRED_SMOKE_MATRIX_ROLES == frozenset(
        {"matched_e_diff_desc", "unmatched_e_diff_desc", "same_desc"}
    )
    assert sut.OPTIONAL_SMOKE_MATRIX_ROLES == frozenset(
        {"f_compatibility", "tp_calibration"}
    )
    assert not (sut.REQUIRED_SMOKE_MATRIX_ROLES & sut.OPTIONAL_SMOKE_MATRIX_ROLES)


def test_forbidden_text_keys_reject_retokenizable_surfaces() -> None:
    # Mirrors score_sorted_owner_basin_landscape.FORBIDDEN_TEXT_KEYS: decoded
    # text is never re-tokenized to reconstruct a state.
    assert {
        "prefix_text",
        "generated_text",
        "prefix_chat_text",
        "chat_text",
        "decoded_text",
    } <= sut.FORBIDDEN_TEXT_KEYS


# ---------------------------------------------------------------------------
# 1. Registry token capture: consume literal registry tokens/digests; reject
#    retokenization or source mismatch.
# ---------------------------------------------------------------------------


def _write_registry(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "registry.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _registry_row(*, row_id: str = "row:0", context_id: str = "ctx:0", token_ids: list[int] | None = None) -> dict[str, Any]:
    tokens = token_ids if token_ids is not None else [OBJECT_REF_START, 1, 2, OBJECT_REF_END]
    return {
        "row_id": row_id,
        "context_id": context_id,
        "token_ids": tokens,
        "token_ids_sha256": _sha256_json(tokens),
    }


def test_load_registry_token_span_accepts_literal_tokens(tmp_path: Path) -> None:
    row = _registry_row()
    registry_path = _write_registry(tmp_path, [row])
    registry_sha256 = sut.sha256_file(registry_path)
    span = sut.load_registry_token_span(
        row, registry_path=registry_path, registry_sha256=registry_sha256, label="P"
    )
    assert span.row_id == "row:0"
    assert span.context_id == "ctx:0"
    assert span.token_ids == tuple(row["token_ids"])
    assert span.token_ids_sha256 == row["token_ids_sha256"]


def test_load_registry_token_span_rejects_source_mismatch(tmp_path: Path) -> None:
    row = _registry_row()
    registry_path = _write_registry(tmp_path, [row])
    stale_sha256 = "0" * 64
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.load_registry_token_span(
            row, registry_path=registry_path, registry_sha256=stale_sha256, label="P"
        )


@pytest.mark.parametrize(
    "forbidden_key",
    ["prefix_text", "generated_text", "prefix_chat_text", "chat_text", "decoded_text"],
)
def test_load_registry_token_span_rejects_retokenizable_text(
    tmp_path: Path, forbidden_key: str
) -> None:
    row = _registry_row()
    row[forbidden_key] = "some decoded prose"
    registry_path = _write_registry(tmp_path, [row])
    registry_sha256 = sut.sha256_file(registry_path)
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.load_registry_token_span(
            row, registry_path=registry_path, registry_sha256=registry_sha256, label="P"
        )


def test_load_registry_token_span_rejects_tampered_digest(tmp_path: Path) -> None:
    row = _registry_row()
    row["token_ids_sha256"] = "1" * 64
    registry_path = _write_registry(tmp_path, [row])
    registry_sha256 = sut.sha256_file(registry_path)
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.load_registry_token_span(
            row, registry_path=registry_path, registry_sha256=registry_sha256, label="P"
        )


def test_load_registry_token_span_rejects_empty_tokens(tmp_path: Path) -> None:
    row = _registry_row(token_ids=[])
    registry_path = _write_registry(tmp_path, [row])
    registry_sha256 = sut.sha256_file(registry_path)
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.load_registry_token_span(
            row, registry_path=registry_path, registry_sha256=registry_sha256, label="P"
        )


# ---------------------------------------------------------------------------
# 2. Full-row E suffix identity: tokens(P+E) - tokens(P). The sidecar owns
#    only E's coordinate-token subsequence/digest, row id/index, description,
#    strict-match fields, and raw-span digest; never a full-row token
#    sequence.
# ---------------------------------------------------------------------------


def test_derive_e_row_suffix_returns_literal_tail() -> None:
    p_tokens = [1, 2, 3]
    p_plus_e_tokens = [1, 2, 3, 4, 5, 6]
    assert sut.derive_e_row_suffix(
        p_tokens=p_tokens, p_plus_e_tokens=p_plus_e_tokens
    ) == (4, 5, 6)


def test_derive_e_row_suffix_rejects_non_prefix_mismatch() -> None:
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.derive_e_row_suffix(p_tokens=[1, 2, 3], p_plus_e_tokens=[1, 9, 3, 4])


def test_derive_e_row_suffix_rejects_p_plus_e_not_longer() -> None:
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.derive_e_row_suffix(p_tokens=[1, 2, 3], p_plus_e_tokens=[1, 2, 3])


def _e_sidecar_row(
    *,
    row_id: str = "sidecar:E",
    row_index: int = 4,
    coord_token_ids: tuple[int, int, int, int] = (151700, 151701, 151702, 151703),
    strict_match_status: str = "matched",
    raw_span_digest: str = "a" * 64,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "row_index": row_index,
        "coord_token_ids": list(coord_token_ids),
        "coord_token_ids_sha256": _sha256_json(list(coord_token_ids)),
        "strict_match_status": strict_match_status,
        "raw_span_digest": raw_span_digest,
    }


def test_validate_e_row_binding_accepts_matching_coordinate_subsequence() -> None:
    coord = (151700, 151701, 151702, 151703)
    p_tokens = [OBJECT_REF_START, 9, OBJECT_REF_END]
    suffix = [OBJECT_REF_START, 8, OBJECT_REF_END, BOX_START, *coord, BOX_END]
    p_plus_e_tokens = p_tokens + suffix
    sidecar = _e_sidecar_row(coord_token_ids=coord)
    binding = sut.validate_e_row_binding(
        sidecar,
        p_tokens=p_tokens,
        p_plus_e_tokens=p_plus_e_tokens,
        coordinate_domain=COORDINATE_DOMAIN,
    )
    assert binding.coord_token_ids == coord
    assert binding.strict_match_status == "matched"
    assert binding.full_row_suffix_sha256 == _sha256_json(suffix)


def test_validate_e_row_binding_rejects_coordinate_subsequence_not_found_in_suffix() -> None:
    coord = (151700, 151701, 151702, 151703)
    p_tokens = [1, 2]
    p_plus_e_tokens = [1, 2, 3, 4, 5, 6, BOX_END]
    sidecar = _e_sidecar_row(coord_token_ids=coord)
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.validate_e_row_binding(
            sidecar,
            p_tokens=p_tokens,
            p_plus_e_tokens=p_plus_e_tokens,
            coordinate_domain=COORDINATE_DOMAIN,
        )


def test_validate_e_row_binding_rejects_out_of_domain_coordinate() -> None:
    coord = (1, 151701, 151702, 151703)
    p_tokens = [1, 2]
    suffix = [OBJECT_REF_START, OBJECT_REF_END, BOX_START, *coord, BOX_END]
    sidecar = _e_sidecar_row(coord_token_ids=coord)
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.validate_e_row_binding(
            sidecar,
            p_tokens=p_tokens,
            p_plus_e_tokens=p_tokens + suffix,
            coordinate_domain=COORDINATE_DOMAIN,
        )


def test_validate_e_row_binding_rejects_stale_coordinate_digest() -> None:
    coord = (151700, 151701, 151702, 151703)
    p_tokens = [1, 2]
    suffix = [OBJECT_REF_START, OBJECT_REF_END, BOX_START, *coord, BOX_END]
    sidecar = _e_sidecar_row(coord_token_ids=coord)
    sidecar["coord_token_ids_sha256"] = "f" * 64
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.validate_e_row_binding(
            sidecar,
            p_tokens=p_tokens,
            p_plus_e_tokens=p_tokens + suffix,
            coordinate_domain=COORDINATE_DOMAIN,
        )


def test_validate_e_row_binding_rejects_unrecognized_strict_match_status() -> None:
    coord = (151700, 151701, 151702, 151703)
    p_tokens = [1, 2]
    suffix = [OBJECT_REF_START, OBJECT_REF_END, BOX_START, *coord, BOX_END]
    sidecar = _e_sidecar_row(coord_token_ids=coord, strict_match_status="probably")
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.validate_e_row_binding(
            sidecar,
            p_tokens=p_tokens,
            p_plus_e_tokens=p_tokens + suffix,
            coordinate_domain=COORDINATE_DOMAIN,
        )


# ---------------------------------------------------------------------------
# 3. Natural description release ladder: gate margin, first-divergent-token
#    margin, description sum/mean, argmax-follows-target, and the
#    same-description coordinate-only flag.
# ---------------------------------------------------------------------------


def test_score_natural_release_reports_first_divergence_and_margin() -> None:
    target_ids = [10, 11, 12]
    native_ids = [10, 11, 99]
    target_logprobs = [-0.1, -0.2, -0.3]
    native_logprobs = [-0.1, -0.2, -0.5]
    observation = sut.score_natural_release(
        boundary_label="P",
        target_token_ids=target_ids,
        native_token_ids=native_ids,
        target_description_logprobs=target_logprobs,
        native_description_logprobs=native_logprobs,
        target_argmax_ids=[10, 11, 99],
    )
    assert observation.same_description is False
    assert observation.first_divergence_index == 2
    assert observation.target_minus_native_margin == pytest.approx(-0.3 - (-0.5))
    assert observation.description_sum == pytest.approx(sum(target_logprobs))
    assert observation.description_token_count == 3
    assert observation.argmax_follows_target is False


def test_score_natural_release_argmax_follows_target_when_prefix_matches() -> None:
    target_ids = [10, 11, 12]
    observation = sut.score_natural_release(
        boundary_label="P",
        target_token_ids=target_ids,
        native_token_ids=[10, 11, 99],
        target_description_logprobs=[-0.1, -0.2, -0.3],
        native_description_logprobs=[-0.1, -0.2, -0.5],
        target_argmax_ids=[10, 11, 12],
    )
    assert observation.argmax_follows_target is True


def test_score_natural_release_flags_same_description_as_construction_determined() -> None:
    target_ids = [10, 11, 12]
    observation = sut.score_natural_release(
        boundary_label="P_plus_E",
        target_token_ids=target_ids,
        native_token_ids=list(target_ids),
        target_description_logprobs=[-0.1, -0.2, -0.3],
        native_description_logprobs=[-0.1, -0.2, -0.3],
        target_argmax_ids=[10, 11, 12],
    )
    assert observation.same_description is True
    assert observation.first_divergence_index is None
    assert observation.target_minus_native_margin is None


def test_score_natural_release_rejects_mismatched_token_and_logprob_lengths() -> None:
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.score_natural_release(
            boundary_label="P",
            target_token_ids=[10, 11],
            native_token_ids=[10, 11],
            target_description_logprobs=[-0.1],
            native_description_logprobs=[-0.1, -0.2],
            target_argmax_ids=[10, 11],
        )


def test_score_natural_release_rejects_unknown_boundary_label() -> None:
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.score_natural_release(
            boundary_label="P_plus_F",
            target_token_ids=[10],
            native_token_ids=[10],
            target_description_logprobs=[-0.1],
            native_description_logprobs=[-0.1],
            target_argmax_ids=[10],
        )


# ---------------------------------------------------------------------------
# 4. Exact 4-coordinate + box_end greedy grammar. Malformed fails closed,
#    never silently repaired.
# ---------------------------------------------------------------------------


def _valid_box() -> list[int]:
    return [151700, 151701, 151702, 151703, BOX_END]


def test_validate_coordinate_grammar_accepts_exact_four_coordinates_and_box_end() -> None:
    result = sut.validate_coordinate_grammar(
        _valid_box(), coordinate_domain=COORDINATE_DOMAIN, box_end_token_id=BOX_END
    )
    assert result.status == "valid"
    assert result.coord_token_ids == tuple(_valid_box()[:4])
    assert result.malformed_reason is None


@pytest.mark.parametrize(
    "tokens,expected_reason_substring",
    [
        ([151700, 151701, 151702, BOX_END], "arity"),  # only 3 coords
        ([151700, 151701, 151702, 151703, 151704, BOX_END], "arity"),  # 5 coords
        ([151700, 151701, 1, 151703, BOX_END], "domain"),  # out-of-domain token
        ([151700, 151701, 151702, 151703], "box_end"),  # premature termination
        ([151700, 151701, 151702, 151703, BOX_END, BOX_END], "arity"),  # trailing extra token
        ([BOX_END, 151701, 151702, 151703, BOX_END], "domain"),  # box_end used as a coordinate
    ],
)
def test_validate_coordinate_grammar_marks_malformed_without_repair(
    tokens: list[int], expected_reason_substring: str
) -> None:
    result = sut.validate_coordinate_grammar(
        tokens, coordinate_domain=COORDINATE_DOMAIN, box_end_token_id=BOX_END
    )
    assert result.status == "malformed"
    assert expected_reason_substring in result.malformed_reason
    # Fail closed: the malformed result must retain the literal offending
    # tokens, never a truncated/padded "repaired" reconstruction.
    assert result.coord_token_ids == tuple(tokens[:4])


def test_validate_coordinate_grammar_result_is_immutable() -> None:
    result = sut.validate_coordinate_grammar(
        _valid_box(), coordinate_domain=COORDINATE_DOMAIN, box_end_token_id=BOX_END
    )
    with pytest.raises(FrozenInstanceError):
        result.status = "malformed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 5. Deterministic coordinate-only greedy decode: an explicit per-step
#    argmax callable, never model.generate(), never sampling. Fails closed
#    at the first malformed step instead of continuing to decode.
# ---------------------------------------------------------------------------


def test_greedy_decode_coordinate_row_calls_argmax_exactly_once_per_step_when_valid() -> None:
    scripted = [151700, 151701, 151702, 151703, BOX_END]
    calls: list[int] = []

    def argmax_token_fn(step_index: int, tokens_so_far: tuple[int, ...]) -> int:
        calls.append(step_index)
        assert tokens_so_far == tuple(scripted[:step_index])
        return scripted[step_index]

    result = sut.greedy_decode_coordinate_row(
        argmax_token_fn,
        coordinate_domain=COORDINATE_DOMAIN,
        box_end_token_id=BOX_END,
    )
    assert result.status == "valid"
    assert result.coord_token_ids == tuple(scripted[:4])
    assert calls == [0, 1, 2, 3, 4]


def test_greedy_decode_coordinate_row_stops_at_first_malformed_step() -> None:
    scripted = [151700, 151701, 1, 151703, BOX_END]  # step 2 is out of domain
    calls: list[int] = []

    def argmax_token_fn(step_index: int, tokens_so_far: tuple[int, ...]) -> int:
        calls.append(step_index)
        return scripted[step_index]

    result = sut.greedy_decode_coordinate_row(
        argmax_token_fn,
        coordinate_domain=COORDINATE_DOMAIN,
        box_end_token_id=BOX_END,
    )
    assert result.status == "malformed"
    # Fails closed immediately: never probes steps after the offending one.
    assert calls == [0, 1, 2]


def test_greedy_decode_coordinate_row_stops_when_terminal_step_is_not_box_end() -> None:
    scripted = [151700, 151701, 151702, 151703, 151704]

    def argmax_token_fn(step_index: int, tokens_so_far: tuple[int, ...]) -> int:
        return scripted[step_index]

    result = sut.greedy_decode_coordinate_row(
        argmax_token_fn,
        coordinate_domain=COORDINATE_DOMAIN,
        box_end_token_id=BOX_END,
    )
    assert result.status == "malformed"
    assert "box_end" in result.malformed_reason


def test_greedy_decode_coordinate_row_never_samples_or_free_generates() -> None:
    # The fixture-level contract itself is the proof: greedy_decode_coordinate_row
    # accepts only a per-step deterministic callable and a fixed step budget;
    # it has no temperature/top_p/generate/sampler parameter to accept.
    import inspect

    signature = inspect.signature(sut.greedy_decode_coordinate_row)
    forbidden_params = {"temperature", "top_p", "top_k", "num_samples", "generate", "sampler"}
    assert forbidden_params.isdisjoint(signature.parameters)


# ---------------------------------------------------------------------------
# 6. Target-conditioned coordinate candidate scoring: owner rank, best
#    competitor, margin, support disposition. Within-context-only.
# ---------------------------------------------------------------------------


def _candidate(candidate_id: str, owner_id: str, score: float, context_id: str = "ctx:crossing:0") -> "sut.CandidateScore":
    return sut.CandidateScore(
        candidate_id=candidate_id,
        owner_id=owner_id,
        context_id=context_id,
        complete_box_logprob_sum=score,
    )


def test_rank_owner_candidates_reports_target_ranks_first() -> None:
    candidates = [
        _candidate("c0", "gt:target", -1.0),
        _candidate("c1", "gt:other", -3.0),
        _candidate("c2", "gt:other2", -5.0),
    ]
    result = sut.rank_owner_candidates(candidates, target_owner_id="gt:target")
    assert result.target_rank == 1
    assert result.best_competitor_owner_id == "gt:other"
    assert result.target_minus_competitor_margin == pytest.approx(-1.0 - (-3.0))
    # A rank disposition, never a support claim: the census freezes rank and
    # margin as a competition surface that is never a support criterion.
    assert result.family_rank_disposition == "target_ranks_first"
    assert not hasattr(result, "support_disposition")


def test_rank_owner_candidates_reports_target_outranked_when_not_top() -> None:
    candidates = [
        _candidate("c0", "gt:target", -5.0),
        _candidate("c1", "gt:other", -1.0),
    ]
    result = sut.rank_owner_candidates(candidates, target_owner_id="gt:target")
    assert result.target_rank == 2
    assert result.best_competitor_owner_id == "gt:other"
    assert result.target_minus_competitor_margin == pytest.approx(-5.0 - (-1.0))
    assert result.family_rank_disposition == "target_outranked"


def test_rank_owner_candidates_flags_exact_rank_tie() -> None:
    candidates = [
        _candidate("c0", "gt:target", -2.0),
        _candidate("c1", "gt:other", -2.0),
    ]
    result = sut.rank_owner_candidates(candidates, target_owner_id="gt:target")
    assert result.family_rank_disposition == "rank_tie"


def test_rank_owner_candidates_rejects_multiple_context_ids() -> None:
    candidates = [
        _candidate("c0", "gt:target", -1.0, context_id="ctx:0"),
        _candidate("c1", "gt:other", -2.0, context_id="ctx:1"),
    ]
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.rank_owner_candidates(candidates, target_owner_id="gt:target")


def test_rank_owner_candidates_rejects_missing_target_owner() -> None:
    candidates = [_candidate("c0", "gt:other", -1.0)]
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.rank_owner_candidates(candidates, target_owner_id="gt:target")


def test_rank_owner_candidates_rejects_empty_candidate_list() -> None:
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.rank_owner_candidates([], target_owner_id="gt:target")


# ---------------------------------------------------------------------------
# 7. Candidate/greedy owner identity fields: preserve likelihood_displaced
#    vs greedy_displaced disagreement, both sub-tags retained independently.
# ---------------------------------------------------------------------------


def _owner_rank(
    *, target_rank: int = 1, best_competitor_owner_id: str | None = None, margin: float | None = None,
    family_rank_disposition: str = "target_ranks_first",
) -> "sut.OwnerRankResult":
    return sut.OwnerRankResult(
        target_owner_id="gt:target",
        target_rank=target_rank,
        best_competitor_owner_id=best_competitor_owner_id,
        target_minus_competitor_margin=margin,
        family_rank_disposition=family_rank_disposition,
    )


def test_classify_displacement_true_for_negative_likelihood_margin_to_another_owner() -> None:
    displacement = sut.classify_displacement(
        target_owner_id="gt:target",
        owner_rank=_owner_rank(
            target_rank=2,
            best_competitor_owner_id="gt:other",
            margin=-0.5,
            family_rank_disposition="target_outranked",
        ),
        greedy_status="target_match",
        greedy_owner_match="gt:target",
    )
    assert displacement.likelihood_displaced is True
    assert displacement.likelihood_displaced_owner_id == "gt:other"
    assert displacement.greedy_displaced is False
    assert displacement.greedy_displaced_owner_id is None


def test_classify_displacement_true_for_greedy_strict_match_to_another_owner() -> None:
    displacement = sut.classify_displacement(
        target_owner_id="gt:target",
        owner_rank=_owner_rank(),
        greedy_status="other_owner_match",
        greedy_owner_match="gt:other",
    )
    assert displacement.likelihood_displaced is False
    assert displacement.greedy_displaced is True
    assert displacement.greedy_displaced_owner_id == "gt:other"


def test_classify_displacement_retains_disagreement_between_both_subtags() -> None:
    # Candidate/likelihood says displaced by gt:other; the deterministic
    # greedy box nonetheless strict-matches the target. Both tags are
    # retained rather than collapsed into a single disposition.
    displacement = sut.classify_displacement(
        target_owner_id="gt:target",
        owner_rank=_owner_rank(
            target_rank=2, best_competitor_owner_id="gt:other", margin=-0.2, family_rank_disposition="target_outranked"
        ),
        greedy_status="target_match",
        greedy_owner_match="gt:target",
    )
    assert displacement.likelihood_displaced is True
    assert displacement.greedy_displaced is False
    assert displacement.decoding_contradicted is True


def test_classify_displacement_not_decoding_contradicted_when_greedy_also_displaced() -> None:
    displacement = sut.classify_displacement(
        target_owner_id="gt:target",
        owner_rank=_owner_rank(
            target_rank=2, best_competitor_owner_id="gt:other", margin=-0.2, family_rank_disposition="target_outranked"
        ),
        greedy_status="other_owner_match",
        greedy_owner_match="gt:other",
    )
    assert displacement.decoding_contradicted is False


def test_classify_displacement_no_displacement_when_target_wins_and_greedy_matches() -> None:
    displacement = sut.classify_displacement(
        target_owner_id="gt:target",
        owner_rank=_owner_rank(target_rank=1, family_rank_disposition="target_ranks_first"),
        greedy_status="target_match",
        greedy_owner_match="gt:target",
    )
    assert displacement.likelihood_displaced is False
    assert displacement.greedy_displaced is False
    assert displacement.decoding_contradicted is False


def test_classify_displacement_rejects_unknown_greedy_status() -> None:
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.classify_displacement(
            target_owner_id="gt:target",
            owner_rank=_owner_rank(),
            greedy_status="something_else",
            greedy_owner_match=None,
        )


# ---------------------------------------------------------------------------
# 8. Primary classification: exhaustive, order-1-to-4 branch assignment.
# ---------------------------------------------------------------------------


def _displacement(likelihood: bool = False, greedy: bool = False, contradicted: bool = False) -> "sut.DisplacementResult":
    return sut.DisplacementResult(
        likelihood_displaced=likelihood,
        likelihood_displaced_owner_id="gt:other" if likelihood else None,
        greedy_displaced=greedy,
        greedy_displaced_owner_id="gt:other" if greedy else None,
        decoding_contradicted=contradicted,
    )


def test_classify_primary_branch_displaced_takes_precedence_over_release_and_realization() -> None:
    # Every predicate for release_lost and realization_fail is also
    # satisfied; branch 1 (displaced) must still win per the exhaustive
    # order in unit.md "Primary classification".
    result = sut.classify_primary_branch(
        displacement=_displacement(likelihood=True),
        release_observable=True,
        release_margin=-0.4,
        forced_dc_support_disposition="unsupported",
        greedy_status="unmatched",
        tie_or_nonunique=False,
        missing_fields=False,
    )
    assert result.branch == "displaced"


def test_classify_primary_branch_release_lost_requires_negative_margin_and_retained_support() -> None:
    result = sut.classify_primary_branch(
        displacement=_displacement(),
        release_observable=True,
        release_margin=-0.4,
        forced_dc_support_disposition="supported",
        greedy_status="target_match",
        tie_or_nonunique=False,
        missing_fields=False,
    )
    assert result.branch == "release_lost"


def test_classify_primary_branch_release_lost_does_not_apply_without_retained_support() -> None:
    result = sut.classify_primary_branch(
        displacement=_displacement(),
        release_observable=True,
        release_margin=-0.4,
        forced_dc_support_disposition="unsupported",
        greedy_status="unmatched",
        tie_or_nonunique=False,
        missing_fields=False,
    )
    assert result.branch == "realization_fail"


def test_classify_primary_branch_realization_fail_requires_no_unique_other_owner_displacement() -> None:
    result = sut.classify_primary_branch(
        displacement=_displacement(),
        release_observable=True,
        release_margin=0.3,
        forced_dc_support_disposition="unsupported",
        greedy_status="malformed",
        tie_or_nonunique=False,
        missing_fields=False,
    )
    assert result.branch == "realization_fail"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(
            release_observable=True,
            release_margin=-0.1,
            forced_dc_support_disposition="ambiguous_tie",
            greedy_status="unmatched",
            tie_or_nonunique=True,
            missing_fields=False,
        ),
        dict(
            release_observable=True,
            release_margin=None,
            forced_dc_support_disposition="unsupported",
            greedy_status="unmatched",
            tie_or_nonunique=False,
            missing_fields=True,
        ),
        dict(
            release_observable=False,
            release_margin=None,
            forced_dc_support_disposition="supported",
            greedy_status="unmatched",
            tie_or_nonunique=False,
            missing_fields=False,
        ),
    ],
)
def test_classify_primary_branch_ambiguous_covers_ties_and_missing_and_non_observable(
    kwargs: dict[str, Any],
) -> None:
    result = sut.classify_primary_branch(displacement=_displacement(), **kwargs)
    assert result.branch == "ambiguous"


def test_classify_primary_branch_at_p_with_same_description_is_not_decision_bearing() -> None:
    # unit.md "Primary classification": at P, a same-description coordinate
    # readout is construction-determined and "never treated as displacement
    # evidence"; classification is only decision-bearing at P+E.
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.classify_primary_branch(
            displacement=_displacement(),
            release_observable=True,
            release_margin=-0.1,
            forced_dc_support_disposition="supported",
            greedy_status="target_match",
            tie_or_nonunique=False,
            missing_fields=False,
            boundary_label="P",
        )


def test_classify_primary_branch_result_reasons_name_the_deciding_predicate() -> None:
    result = sut.classify_primary_branch(
        displacement=_displacement(greedy=True),
        release_observable=True,
        release_margin=-0.1,
        forced_dc_support_disposition="supported",
        greedy_status="other_owner_match",
        tie_or_nonunique=False,
        missing_fields=False,
    )
    assert result.branch == "displaced"
    assert any("greedy_displaced" in reason for reason in result.reasons)


# ---------------------------------------------------------------------------
# 9. Fresh cache/context isolation: no logical context group is reused
#    across owners.
# ---------------------------------------------------------------------------


def test_assert_fresh_context_per_owner_accepts_unique_group_ids() -> None:
    sut.assert_fresh_context_per_owner(["ctx-group:0", "ctx-group:1", "ctx-group:2"])


def test_assert_fresh_context_per_owner_rejects_reused_group_id() -> None:
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.assert_fresh_context_per_owner(["ctx-group:0", "ctx-group:1", "ctx-group:0"])


def test_assert_fresh_context_per_owner_rejects_empty_sequence() -> None:
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.assert_fresh_context_per_owner([])


# ---------------------------------------------------------------------------
# 10. Cached-versus-uncached parity gate: selected-logit <= 1e-3 plus exact
#     argmax/sign/rank/support/match parity; uncached fallback/quarantine
#     semantics.
# ---------------------------------------------------------------------------


def _streams(
    *, selected: list[float], argmax: list[int], request_ids: tuple[str, ...] = ("req:a",)
) -> dict[str, "sut.SurfaceParityStreams"]:
    """One aligned stream per surface; an absent surface is never agreement."""

    return {
        surface: sut.SurfaceParityStreams(
            request_ids=request_ids,
            selected_logprobs=tuple(selected),
            argmax_token_ids=tuple(argmax),
        )
        for surface in sut.SURFACES
    }


def _parity_inputs(**overrides: Any) -> "sut.ParityCheckInputs":
    base = dict(
        cached_selected_logit=-1.0,
        uncached_selected_logit=-1.0005,
        cached_streams=_streams(
            selected=[-1.0, -2.0, -3.0], argmax=[151700, 151701, 151702]
        ),
        uncached_streams=_streams(
            selected=[-1.0, -2.0005, -3.0], argmax=[151700, 151701, 151702]
        ),
        cached_argmax_token_id=151700,
        uncached_argmax_token_id=151700,
        cached_margin_sign=-1,
        uncached_margin_sign=-1,
        cached_owner_rank=1,
        uncached_owner_rank=1,
        cached_support_disposition="supported",
        uncached_support_disposition="supported",
        cached_owner_match="gt:target",
        uncached_owner_match="gt:target",
        cached_primary_branch="release_lost",
        uncached_primary_branch="release_lost",
    )
    base.update(overrides)
    return sut.ParityCheckInputs(**base)


def test_evaluate_cache_parity_admits_cache_within_threshold_and_full_agreement() -> None:
    result = sut.evaluate_cache_parity(_parity_inputs())
    assert result.status == "cache_admitted"
    assert result.max_selected_logit_abs_diff <= sut.CACHE_PARITY_MAX_SELECTED_LOGIT_ABS_DIFF
    assert result.mismatched_fields == ()


def test_evaluate_cache_parity_falls_back_when_selected_logit_diff_exceeds_threshold() -> None:
    result = sut.evaluate_cache_parity(
        _parity_inputs(uncached_selected_logit=-1.0 - 5e-3)
    )
    assert result.status == "uncached_fallback"
    assert result.max_selected_logit_abs_diff > sut.CACHE_PARITY_MAX_SELECTED_LOGIT_ABS_DIFF


def test_evaluate_cache_parity_falls_back_on_argmax_mismatch_even_within_threshold() -> None:
    result = sut.evaluate_cache_parity(_parity_inputs(uncached_argmax_token_id=151701))
    assert result.status == "uncached_fallback"
    assert "argmax" in result.mismatched_fields


@pytest.mark.parametrize(
    "field,cached_value,uncached_value",
    [
        ("margin_sign", -1, 1),
        ("owner_rank", 1, 2),
        ("support_disposition", "supported", "unsupported"),
        ("owner_match", "gt:target", "gt:other"),
        ("primary_branch", "release_lost", "ambiguous"),
    ],
)
def test_evaluate_cache_parity_falls_back_on_any_compared_field_mismatch(
    field: str, cached_value: Any, uncached_value: Any
) -> None:
    overrides = {f"uncached_{field}": uncached_value}
    result = sut.evaluate_cache_parity(_parity_inputs(**overrides))
    assert result.status == "uncached_fallback"
    assert field in result.mismatched_fields


def test_evaluate_cache_parity_result_is_immutable() -> None:
    result = sut.evaluate_cache_parity(_parity_inputs())
    with pytest.raises(FrozenInstanceError):
        result.status = "cache_admitted"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 11. Native replay through pre-divergence tokens; quarantine ledger;
#     >2 primary quarantines stop.
# ---------------------------------------------------------------------------


def test_replay_argmax_through_prefix_admits_on_exact_match() -> None:
    assert sut.replay_argmax_through_prefix(
        expected_token_ids=[1, 2, 3, 4],
        argmax_token_ids=[1, 2, 3, 9],
        up_to_index=3,
    ) is True


def test_replay_argmax_through_prefix_rejects_on_pre_divergence_mismatch() -> None:
    assert sut.replay_argmax_through_prefix(
        expected_token_ids=[1, 2, 3, 4],
        argmax_token_ids=[1, 9, 3, 4],
        up_to_index=3,
    ) is False


def test_replay_argmax_through_prefix_rejects_length_shortfall() -> None:
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.replay_argmax_through_prefix(
            expected_token_ids=[1, 2], argmax_token_ids=[1, 2], up_to_index=5
        )


def test_quarantine_ledger_starts_empty() -> None:
    ledger = sut.QuarantineLedger(entries=())
    assert ledger.count == 0


def test_apply_native_replay_quarantine_appends_only_on_mismatch() -> None:
    ledger = sut.QuarantineLedger(entries=())
    admitted = sut.apply_native_replay_quarantine(
        argmax_replay_matches=True, owner_id="owner:0", reason="n/a", detail="n/a", ledger=ledger
    )
    assert admitted.count == 0
    quarantined = sut.apply_native_replay_quarantine(
        argmax_replay_matches=False,
        owner_id="owner:1",
        reason="pre_divergence_argmax_mismatch",
        detail="token 2 diverged before the tested boundary",
        ledger=admitted,
    )
    assert quarantined.count == 1
    assert quarantined.entries[0].owner_id == "owner:1"


def test_check_quarantine_stop_false_at_exactly_two() -> None:
    ledger = sut.QuarantineLedger(
        entries=(
            sut.QuarantineEntry(owner_id="owner:0", reason="r", detail="d"),
            sut.QuarantineEntry(owner_id="owner:1", reason="r", detail="d"),
        )
    )
    assert sut.check_quarantine_stop(ledger) is False


def test_check_quarantine_stop_true_above_two() -> None:
    ledger = sut.QuarantineLedger(
        entries=(
            sut.QuarantineEntry(owner_id="owner:0", reason="r", detail="d"),
            sut.QuarantineEntry(owner_id="owner:1", reason="r", detail="d"),
            sut.QuarantineEntry(owner_id="owner:2", reason="r", detail="d"),
        )
    )
    assert sut.check_quarantine_stop(ledger) is True


def test_quarantine_ledger_is_immutable_append_only() -> None:
    ledger = sut.QuarantineLedger(entries=())
    with pytest.raises(FrozenInstanceError):
        ledger.entries = (sut.QuarantineEntry(owner_id="x", reason="r", detail="d"),)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 12. Exact request/output identity and deterministic hashes.
# ---------------------------------------------------------------------------


def test_build_request_identity_is_deterministic() -> None:
    kwargs = dict(
        context_id="ctx:0",
        prefix_token_ids=[1, 2, 3],
        appended_token_ids=[4, 5],
    )
    first = sut.build_request_identity(**kwargs)
    second = sut.build_request_identity(**kwargs)
    assert first == second
    assert first["request_identity_sha256"] == _sha256_json(
        {
            "context_id": "ctx:0",
            "prefix_token_ids": [1, 2, 3],
            "appended_token_ids": [4, 5],
        }
    )


@pytest.mark.parametrize(
    "mutation",
    [
        dict(context_id="ctx:1"),
        dict(prefix_token_ids=[1, 2, 9]),
        dict(appended_token_ids=[4, 6]),
    ],
)
def test_build_request_identity_changes_with_any_input(mutation: dict[str, Any]) -> None:
    base = dict(context_id="ctx:0", prefix_token_ids=[1, 2, 3], appended_token_ids=[4, 5])
    mutated = {**base, **mutation}
    assert sut.build_request_identity(**base)["request_identity_sha256"] != sut.build_request_identity(**mutated)["request_identity_sha256"]


def test_build_output_identity_binds_to_its_request_identity() -> None:
    request = sut.build_request_identity(
        context_id="ctx:0", prefix_token_ids=[1, 2, 3], appended_token_ids=[4, 5]
    )
    output = sut.build_output_identity(
        request_identity_sha256=request["request_identity_sha256"],
        selected_logits=[-0.1, -0.2],
        token_ids=[4, 5],
    )
    assert output["request_identity_sha256"] == request["request_identity_sha256"]
    again = sut.build_output_identity(
        request_identity_sha256=request["request_identity_sha256"],
        selected_logits=[-0.1, -0.2],
        token_ids=[4, 5],
    )
    assert output["output_identity_sha256"] == again["output_identity_sha256"]


def test_build_output_identity_rejects_selected_logits_and_token_length_mismatch() -> None:
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.build_output_identity(
            request_identity_sha256="a" * 64, selected_logits=[-0.1], token_ids=[4, 5]
        )


# ---------------------------------------------------------------------------
# 13. Smoke matrix roles.
# ---------------------------------------------------------------------------


def test_validate_smoke_matrix_roles_accepts_the_three_required_roles() -> None:
    sut.validate_smoke_matrix_roles(
        ["matched_e_diff_desc", "unmatched_e_diff_desc", "same_desc"]
    )


def test_validate_smoke_matrix_roles_accepts_optional_roles_alongside_required() -> None:
    sut.validate_smoke_matrix_roles(
        [
            "matched_e_diff_desc",
            "unmatched_e_diff_desc",
            "same_desc",
            "f_compatibility",
            "tp_calibration",
        ]
    )


def test_validate_smoke_matrix_roles_rejects_missing_required_role() -> None:
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.validate_smoke_matrix_roles(["matched_e_diff_desc", "same_desc"])


def test_validate_smoke_matrix_roles_rejects_unknown_role() -> None:
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.validate_smoke_matrix_roles(
            ["matched_e_diff_desc", "unmatched_e_diff_desc", "same_desc", "not_a_role"]
        )


# ---------------------------------------------------------------------------
# 14. Cohort denominators and timing-control count validation (fail closed
#     rather than silently patched).
# ---------------------------------------------------------------------------


def test_validate_primary_cohort_counts_accepts_the_frozen_denominators() -> None:
    sut.validate_primary_cohort_counts(
        u_count=26, l_count=25, same_context_ul_count=24, matched_e_count=12, unmatched_e_count=14
    )


@pytest.mark.parametrize(
    "overrides",
    [
        dict(u_count=27),
        dict(l_count=26),
        dict(same_context_ul_count=23),
        dict(matched_e_count=11),
        dict(unmatched_e_count=15),
    ],
)
def test_validate_primary_cohort_counts_rejects_any_drifted_denominator(
    overrides: dict[str, int],
) -> None:
    base = dict(u_count=26, l_count=25, same_context_ul_count=24, matched_e_count=12, unmatched_e_count=14)
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.validate_primary_cohort_counts(**{**base, **overrides})


def test_validate_timing_control_count_accepts_fourteen() -> None:
    sut.validate_timing_control_count(14)


def test_validate_timing_control_count_rejects_the_stale_twenty_one() -> None:
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.validate_timing_control_count(21)


def test_validate_timing_control_count_rejects_any_other_drift() -> None:
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.validate_timing_control_count(13)


# ---------------------------------------------------------------------------
# 15. Interpretability gate and two-thirds branch routing, including the
#     decoding-contradicted-likelihood sensitivity check.
# ---------------------------------------------------------------------------


def _owner_records(
    *, branch_counts: dict[str, int], matched_e: int, unmatched_e: int, uninterpretable: int = 0
) -> list["sut.OwnerRecord"]:
    records: list[sut.OwnerRecord] = []
    remaining_matched = matched_e
    index = 0
    for branch, count in branch_counts.items():
        for _ in range(count):
            stratum = "matched_e" if remaining_matched > 0 else "unmatched_e"
            if remaining_matched > 0:
                remaining_matched -= 1
            records.append(
                sut.OwnerRecord(
                    owner_id=f"owner:{index}", stratum=stratum, interpretable=True, branch=branch
                )
            )
            index += 1
    for _ in range(uninterpretable):
        records.append(
            sut.OwnerRecord(owner_id=f"owner:{index}", stratum="unmatched_e", interpretable=False, branch=None)
        )
        index += 1
    return records


def test_evaluate_interpretability_gate_passes_the_frozen_minimums() -> None:
    records = _owner_records(
        branch_counts={"realization_fail": 20}, matched_e=6, unmatched_e=14
    )
    gate = sut.evaluate_interpretability_gate(records)
    assert gate["interpretable_count"] == 20
    assert gate["matched_e_interpretable_count"] == 6
    assert gate["unmatched_e_interpretable_count"] == 14
    assert gate["passed"] is True


def test_evaluate_interpretability_gate_fails_below_total_minimum() -> None:
    records = _owner_records(branch_counts={"realization_fail": 19}, matched_e=6, unmatched_e=13)
    gate = sut.evaluate_interpretability_gate(records)
    assert gate["passed"] is False


def test_evaluate_interpretability_gate_fails_below_matched_e_minimum() -> None:
    records = _owner_records(branch_counts={"realization_fail": 20}, matched_e=5, unmatched_e=15)
    gate = sut.evaluate_interpretability_gate(records)
    assert gate["matched_e_interpretable_count"] == 5
    assert gate["passed"] is False


def test_evaluate_branch_routing_routes_successor_at_two_thirds_majority() -> None:
    # 20 interpretable owners, 14/20 (0.70) in one branch clears 2/3.
    records = _owner_records(
        branch_counts={"realization_fail": 14, "release_lost": 6}, matched_e=6, unmatched_e=14
    )
    routing = sut.evaluate_branch_routing(records)
    assert routing["status"] == "routed"
    assert routing["routed_branch"] == "realization_fail"


def test_evaluate_branch_routing_split_when_no_branch_reaches_two_thirds() -> None:
    records = _owner_records(
        branch_counts={"realization_fail": 10, "release_lost": 10}, matched_e=6, unmatched_e=14
    )
    routing = sut.evaluate_branch_routing(records)
    assert routing["status"] == "split"
    assert routing["routed_branch"] is None


def test_evaluate_branch_routing_split_when_excluding_decoding_contradicted_cell_drops_below_two_thirds() -> None:
    # 14/20 in realization_fail clears 2/3 (0.70); excluding one
    # decoding-contradicted owner from that branch drops it to 13/19
    # (~0.684), which still clears -- so pick counts that do flip below the
    # threshold once excluded: 14/21 with a 1-quarantine style exclusion.
    records = _owner_records(
        branch_counts={"realization_fail": 14, "release_lost": 7}, matched_e=6, unmatched_e=15
    )
    contradicted_owner_id = records[0].owner_id
    assert records[0].branch == "realization_fail"
    routing_without_exclusion = sut.evaluate_branch_routing(records)
    assert routing_without_exclusion["status"] == "routed"
    routing_with_exclusion = sut.evaluate_branch_routing(
        records, decoding_contradicted_owner_ids=[contradicted_owner_id]
    )
    assert routing_with_exclusion["status"] == "split"
    assert routing_with_exclusion["routed_branch"] is None


def test_evaluate_branch_routing_requires_at_least_the_interpretability_gate() -> None:
    records = _owner_records(branch_counts={"realization_fail": 14}, matched_e=6, unmatched_e=0)
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.evaluate_branch_routing(records)


# ---------------------------------------------------------------------------
# 16. No free generate()/sampling/analyzer/GPU surfaces in the tested public
#     contract: every function above accepts only plain values or a single
#     explicit per-step callable. This is asserted, not assumed.
# ---------------------------------------------------------------------------


def test_public_scoring_helpers_never_accept_a_model_tokenizer_or_device_argument() -> None:
    import inspect

    forbidden = {"model", "tokenizer", "device", "cuda_device", "generation_policy"}
    public_callables = [
        sut.load_registry_token_span,
        sut.derive_e_row_suffix,
        sut.validate_e_row_binding,
        sut.score_natural_release,
        sut.validate_coordinate_grammar,
        sut.greedy_decode_coordinate_row,
        sut.rank_owner_candidates,
        sut.classify_displacement,
        sut.classify_primary_branch,
        sut.assert_fresh_context_per_owner,
        sut.evaluate_cache_parity,
        sut.replay_argmax_through_prefix,
        sut.build_request_identity,
        sut.build_output_identity,
        sut.validate_smoke_matrix_roles,
        sut.validate_primary_cohort_counts,
        sut.validate_timing_control_count,
        sut.evaluate_interpretability_gate,
        sut.evaluate_branch_routing,
    ]
    for fn in public_callables:
        signature = inspect.signature(fn)
        assert forbidden.isdisjoint(signature.parameters), fn.__name__


# ---------------------------------------------------------------------------
# 17. Appended regressions (authored after the pure contracts above, against
#     the now-implemented capture path).  Still pure: no torch, no model, no
#     sealed-plan directory and no filesystem beyond ``tmp_path``.
#
#     Each block below pins one defect class that a partially-authored scorer
#     actually exhibited or could silently reintroduce.
# ---------------------------------------------------------------------------


def _cohort_row(
    *, image_id: str, owner_id: str, stratum: str, same_description: bool
) -> dict[str, Any]:
    """One sealed primary-cohort row, reduced to the smoke-role fields."""

    return {
        "image_id": image_id,
        "gt_owner_id": owner_id,
        "same_description_as_e": same_description,
        "e_row": {"stratum": stratum},
    }


MATCHED_E = "matched_e"
UNMATCHED_E = "unmatched_e"


def _three_role_image(image_id: str, *, suffix: str = "") -> list[dict[str, Any]]:
    return [
        _cohort_row(
            image_id=image_id,
            owner_id=f"gt:{image_id}:1{suffix}",
            stratum=MATCHED_E,
            same_description=False,
        ),
        _cohort_row(
            image_id=image_id,
            owner_id=f"gt:{image_id}:2{suffix}",
            stratum=UNMATCHED_E,
            same_description=False,
        ),
        _cohort_row(
            image_id=image_id,
            owner_id=f"gt:{image_id}:3{suffix}",
            stratum=MATCHED_E,
            same_description=True,
        ),
    ]


# --- 17a. The smoke matrix must fit inside ONE image session ---------------


def test_smoke_role_of_reads_same_description_before_the_matched_stratum() -> None:
    # A same-description owner is a coordinate-only observation whatever its
    # E stratum is, so the description test has to come first.
    same_desc_but_matched = _cohort_row(
        image_id="1", owner_id="gt:1:1", stratum=MATCHED_E, same_description=True
    )
    assert sut.smoke_role_of(same_desc_but_matched) == "same_desc"
    assert (
        sut.smoke_role_of(
            _cohort_row(
                image_id="1", owner_id="gt:1:2", stratum=MATCHED_E, same_description=False
            )
        )
        == "matched_e_diff_desc"
    )
    assert (
        sut.smoke_role_of(
            _cohort_row(
                image_id="1", owner_id="gt:1:3", stratum=UNMATCHED_E, same_description=False
            )
        )
        == "unmatched_e_diff_desc"
    )


def test_images_with_all_required_smoke_roles_lists_only_common_image_carriers() -> None:
    rows = [
        *_three_role_image("4134"),
        *_three_role_image("16228"),
        # Spread thin across two images: neither can carry the matrix alone.
        _cohort_row(
            image_id="6040", owner_id="gt:6040:1", stratum=MATCHED_E, same_description=False
        ),
        _cohort_row(
            image_id="7511", owner_id="gt:7511:1", stratum=UNMATCHED_E, same_description=False
        ),
        _cohort_row(
            image_id="7511", owner_id="gt:7511:2", stratum=MATCHED_E, same_description=True
        ),
    ]
    assert sut.images_with_all_required_smoke_roles(rows) == ["4134", "16228"]


def test_select_smoke_owners_picks_one_owner_per_role_from_the_named_image() -> None:
    rows = [*_three_role_image("4134"), *_three_role_image("16228")]
    selected = sut.select_smoke_owners(rows, image_id="16228")

    assert set(selected) == set(sut.REQUIRED_SMOKE_MATRIX_ROLES)
    # Restricting to one image is the correctness requirement: a session holds
    # exactly one image's visual state.
    assert all(owner_id.startswith("gt:16228:") for owner_id in selected.values())


def test_select_smoke_owners_is_deterministic_and_score_blind() -> None:
    rows = [
        _cohort_row(
            image_id="4134", owner_id="gt:4134:9", stratum=MATCHED_E, same_description=False
        ),
        _cohort_row(
            image_id="4134", owner_id="gt:4134:27", stratum=MATCHED_E, same_description=False
        ),
        _cohort_row(
            image_id="4134", owner_id="gt:4134:32", stratum=UNMATCHED_E, same_description=False
        ),
        _cohort_row(
            image_id="4134", owner_id="gt:4134:22", stratum=MATCHED_E, same_description=True
        ),
    ]
    first = sut.select_smoke_owners(rows, image_id="4134")
    second = sut.select_smoke_owners(list(reversed(rows)), image_id="4134")

    assert first == second
    # Lexicographically smallest owner id within the role, never a score.
    assert first["matched_e_diff_desc"] == "gt:4134:27"


def test_select_smoke_owners_fails_closed_and_names_the_eligible_images() -> None:
    rows = [
        *_three_role_image("4134"),
        _cohort_row(
            image_id="6040", owner_id="gt:6040:1", stratum=MATCHED_E, same_description=False
        ),
    ]
    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.select_smoke_owners(rows, image_id="6040")

    message = str(excinfo.value)
    assert "unmatched_e_diff_desc" in message and "same_desc" in message
    # It must route the operator at the eligible image rather than tempting a
    # cross-image smoke.
    assert "'4134'" in message


# --- 17b. Never score a context through another image's backend ------------


class _StubCensus:
    def __init__(self, contexts: dict[str, Any]) -> None:
        self.contexts_by_id = contexts


class _StubInputs:
    def __init__(self, contexts: dict[str, Any], images: dict[str, Any]) -> None:
        self.census = _StubCensus(contexts)
        self.images_by_id = images


def _two_image_plan() -> Any:
    """A plan holding two images whose prompts and prefixes are literal tokens."""

    images = {
        "4134": {"prompt_token_ids": [11, 12, 13]},
        "16228": {"prompt_token_ids": [21, 22, 23]},
    }
    contexts = {
        "ctx:4134": {"image_id": "4134", "generated_prefix_token_ids": [101, 102]},
        "ctx:16228": {"image_id": "16228", "generated_prefix_token_ids": [201, 202]},
    }
    return sut.SealedPlan(
        plan_dir=Path("/nonexistent/plan"),
        manifest={"manifest_content_sha256": "plan-digest"},
        cohort_rows=[],
        control_rows=[],
        request_rows=[],
        plan_file_sha256={},
        inputs=_StubInputs(contexts, images),
        candidates_by_id={},
        owners_by_image={},
    )


def test_executed_prefix_is_the_prompt_then_the_native_self_prefix() -> None:
    plan = _two_image_plan()
    # Scoring the generated prefix alone would drop the multimodal prompt and
    # forward a sequence the native rollout never ran.
    assert plan.executed_prefix_token_ids("ctx:4134") == [11, 12, 13, 101, 102]
    assert plan.context_prefix_token_ids("ctx:4134") == [101, 102]


def test_context_of_another_image_is_refused_by_the_open_session() -> None:
    plan = _two_image_plan()
    plan.assert_context_belongs_to_session_image(
        "ctx:4134", session_image_id="4134", label="smoke"
    )
    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        plan.assert_context_belongs_to_session_image(
            "ctx:16228", session_image_id="4134", label="capture shard"
        )

    message = str(excinfo.value)
    assert "capture shard" in message
    assert "another image's visual state" in message


def test_every_boundary_of_the_smoke_image_is_admitted_by_its_own_session() -> None:
    # The deciding check is the context's sealed image id; the prompt-token
    # cross-check cannot disagree once the ids agree, because the executed
    # prefix is built from the context's own image.
    plan = _two_image_plan()
    plan.contexts_by_id["ctx:4134:b8"] = {
        "image_id": "4134",
        "generated_prefix_token_ids": [103, 104, 105],
    }
    for context_id in ("ctx:4134", "ctx:4134:b8"):
        plan.assert_context_belongs_to_session_image(
            context_id, session_image_id="4134", label="capture shard"
        )


def test_session_image_absent_from_the_registry_fails_closed() -> None:
    plan = _two_image_plan()
    plan.contexts_by_id["ctx:9999"] = {
        "image_id": "9999",
        "generated_prefix_token_ids": [301],
    }
    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        plan.assert_context_belongs_to_session_image(
            "ctx:9999", session_image_id="9999", label="capture shard"
        )
    assert "absent from the image registry" in str(excinfo.value)


def test_unknown_context_id_fails_closed_instead_of_defaulting() -> None:
    plan = _two_image_plan()
    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        plan.assert_context_belongs_to_session_image(
            "ctx:absent", session_image_id="4134", label="capture shard"
        )
    assert "absent from the sealed context registry" in str(excinfo.value)


# --- 17c. The admission receipt must carry every field its digest covers ----


def _runtime_identity(**overrides: Any) -> dict[str, Any]:
    identity = {
        "backend": "hf",
        "model_identity": {"checkpoint": "step-4887", "dtype": "float32"},
        "tokenizer_identity": {"vocab_size": 151936},
        "adapter_identity": {"adapter_name": "default", "rank": 16},
        "numerics": {
            "explicit_position_ids": True,
            "repetition_penalty_stratum": 1.0,
            "matmul_precision": {
                "float32_matmul_precision": "highest",
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "torch_version": "2.9.1+cu128",
            },
            "infer_config_sha256": "config-digest",
        },
        "source_identity": {
            "scripts.research.score_sorted_crossing_boundary_owner_release": "scorer-digest",
            "scripts.research.score_sorted_owner_accessibility_census_shard": "backend-digest",
        },
        # Per-image and lineage fields deliberately outside the digest.
        "session_image_id": "4134",
        "frozen_identity_path": "/somewhere/shard-receipt.json",
    }
    identity.update(overrides)
    return identity


def _sealed_admission(runtime_identity: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    admission = {
        "schema_version": sut.ADMISSION_SCHEMA_VERSION,
        "unit_id": sut.UNIT_ID,
        "smoke_shard_id": "smoke-4134",
        "smoke_image_id": "4134",
        "plan_manifest_content_sha256": "plan-digest",
        "runtime_identity_sha256": sut.runtime_identity_digest(runtime_identity),
        "admission_identity_fields": list(sut.ADMISSION_IDENTITY_FIELDS),
        **sut.admission_identity_payload(runtime_identity),
        "smoke_matrix": sut.validate_smoke_matrix_roles(
            sorted(sut.REQUIRED_SMOKE_MATRIX_ROLES)
        ),
        "smoke_rows": [
            {"role": role, "status": sut.CACHE_ADMITTED}
            for role in sorted(sut.REQUIRED_SMOKE_MATRIX_ROLES)
        ],
        "cache_admitted": True,
        "surface_backend": {surface: sut.KV_CACHE_BACKEND for surface in sut.SURFACES},
        "batch_parity": {"status": "passed", "effective_batch_size": 4},
        "admitted_batch_size": 4,
    }
    admission.update(overrides)
    admission["admission_content_sha256"] = _sha256_json(admission)
    return admission


def test_admission_identity_fields_bind_plan_runtime_and_source_surfaces() -> None:
    # unit.md binds model, tokenizer, adapter, numerics and scorer/backend
    # source; dropping any of them would let a changed runtime inherit a cache
    # admission it never earned.
    assert set(sut.ADMISSION_IDENTITY_FIELDS) == {
        "backend",
        "model_identity",
        "tokenizer_identity",
        "adapter_identity",
        "numerics",
        "source_identity",
    }


def test_admission_identity_payload_carries_every_digested_field_verbatim() -> None:
    # The regression for the defect that made every honest capture fail: the
    # payload was hand-enumerated and omitted ``backend``/``adapter_identity``,
    # so the digest rebuilt from the receipt read them as ``None``.
    runtime_identity = _runtime_identity()
    payload = sut.admission_identity_payload(runtime_identity)

    assert set(payload) == set(sut.ADMISSION_IDENTITY_FIELDS)
    assert all(payload[field] is not None for field in sut.ADMISSION_IDENTITY_FIELDS)
    assert sut.runtime_identity_digest(payload) == sut.runtime_identity_digest(runtime_identity)


def test_admission_validates_against_the_unchanged_runtime_it_was_sealed_under() -> None:
    runtime_identity = _runtime_identity()
    plan = _two_image_plan()

    validated = sut.validate_admission_receipt(
        _sealed_admission(runtime_identity), plan=plan, runtime_identity=runtime_identity
    )
    assert validated["cache_admitted"] is True
    assert validated["admitted_batch_size"] == 4


@pytest.mark.parametrize(
    ("field", "drifted"),
    [
        ("backend", "fake"),
        ("adapter_identity", {"adapter_name": "other", "rank": 32}),
        ("model_identity", {"checkpoint": "step-9999", "dtype": "float32"}),
        ("tokenizer_identity", {"vocab_size": 32000}),
        ("source_identity", {"scripts.research.x": "different-digest"}),
    ],
)
def test_admission_refuses_a_capture_whose_runtime_identity_drifted(
    field: str, drifted: Any
) -> None:
    sealed_under = _runtime_identity()
    admission = _sealed_admission(sealed_under)
    running_now = _runtime_identity(**{field: drifted})

    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.validate_admission_receipt(
            admission, plan=_two_image_plan(), runtime_identity=running_now
        )
    message = str(excinfo.value)
    assert "different runtime identity" in message
    assert field in message


def test_admission_refuses_a_capture_whose_numerics_drifted() -> None:
    # A TF32 flip is a different numerical runtime even at the same checkpoint.
    sealed_under = _runtime_identity()
    admission = _sealed_admission(sealed_under)
    drifted_numerics = json.loads(json.dumps(sealed_under["numerics"]))
    drifted_numerics["matmul_precision"]["cuda_matmul_allow_tf32"] = True

    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.validate_admission_receipt(
            admission,
            plan=_two_image_plan(),
            runtime_identity=_runtime_identity(numerics=drifted_numerics),
        )
    assert "numerics" in str(excinfo.value)


def test_admission_refuses_an_edited_receipt_that_did_not_reseal() -> None:
    runtime_identity = _runtime_identity()
    tampered = _sealed_admission(runtime_identity)
    tampered["cache_admitted"] = False  # edited after the seal was computed

    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.validate_admission_receipt(
            tampered, plan=_two_image_plan(), runtime_identity=runtime_identity
        )
    assert "does not self-seal" in str(excinfo.value)


def test_admission_refuses_a_resealed_receipt_carrying_a_stale_identity_digest() -> None:
    # The sharper tamper: swap in a foreign runtime, keep the *old* digest and
    # re-seal the content hash so the self-seal check passes.  The digest must
    # be recomputed from the receipt's own declared fields, never trusted.
    sealed_under = _runtime_identity()
    stale_digest = sut.runtime_identity_digest(sealed_under)
    foreign = _runtime_identity(backend="fake", model_identity={"checkpoint": "other"})
    resealed = _sealed_admission(foreign, runtime_identity_sha256=stale_digest)

    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.validate_admission_receipt(
            resealed, plan=_two_image_plan(), runtime_identity=foreign
        )
    assert "internally inconsistent" in str(excinfo.value)


def test_admission_refuses_a_receipt_sealed_against_another_plan() -> None:
    runtime_identity = _runtime_identity()
    admission = _sealed_admission(
        runtime_identity, plan_manifest_content_sha256="another-plan-digest"
    )

    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.validate_admission_receipt(
            admission, plan=_two_image_plan(), runtime_identity=runtime_identity
        )
    assert "different plan manifest" in str(excinfo.value)


def test_admission_refuses_a_receipt_missing_a_required_smoke_role() -> None:
    runtime_identity = _runtime_identity()
    admission = _sealed_admission(
        runtime_identity,
        smoke_rows=[{"role": "matched_e_diff_desc", "status": sut.CACHE_ADMITTED}],
    )

    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.validate_admission_receipt(
            admission, plan=_two_image_plan(), runtime_identity=runtime_identity
        )
    assert "missing required role" in str(excinfo.value)


# --- 17d. Predecessor numerics: bind what moves fp32, report what may differ -


PREDECESSOR_MATMUL = {
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": True,
    "float32_matmul_precision": "highest",
    "torch_available": True,
    "torch_version": "2.9.1+cu128",
}


def test_matmul_binding_and_reported_field_split_is_frozen() -> None:
    assert sut.MATMUL_PRECISION_BINDING_FIELDS == (
        "float32_matmul_precision",
        "cuda_matmul_allow_tf32",
        "torch_version",
    )
    assert sut.MATMUL_PRECISION_REPORTED_FIELDS == ("cudnn_allow_tf32",)


def test_extract_frozen_matmul_precision_reads_the_receipt_sibling_numerics_block() -> None:
    # The predecessor census receipt seals matmul state beside
    # ``backend_identity``, not inside it.
    receipt = {
        "backend_identity": {"backend": "hf", "model_identity": {}, "tokenizer_identity": {}},
        "numerics": {"matmul_precision": dict(PREDECESSOR_MATMUL)},
    }
    assert sut.extract_frozen_matmul_precision(receipt) == PREDECESSOR_MATMUL
    assert sut.extract_frozen_matmul_precision({"backend_identity": {}}) is None


def test_compare_matmul_precision_reports_the_expected_cudnn_tf32_difference() -> None:
    # ``pin_fp32_parity_flags`` turns off the cuDNN TF32 path the predecessor
    # left on: strictly more precise, so it is reported, never a blocker.
    observed = {**PREDECESSOR_MATMUL, "cudnn_allow_tf32": False}
    comparison = sut.compare_matmul_precision(PREDECESSOR_MATMUL, observed)

    assert comparison["compared"] is True
    assert comparison["binding_fields_agree"] is True
    assert comparison["reported_field_differences"] == ["cudnn_allow_tf32"]
    assert comparison["reported_fields"]["cudnn_allow_tf32"] == {
        "frozen": True,
        "observed": False,
    }


@pytest.mark.parametrize(
    ("field", "drifted"),
    [
        ("float32_matmul_precision", "high"),
        ("cuda_matmul_allow_tf32", True),
        ("torch_version", "2.4.0+cu121"),
    ],
)
def test_compare_matmul_precision_fails_closed_on_any_binding_field(
    field: str, drifted: Any
) -> None:
    observed = {**PREDECESSOR_MATMUL, "cudnn_allow_tf32": False, field: drifted}
    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.compare_matmul_precision(PREDECESSOR_MATMUL, observed)
    assert field in str(excinfo.value)


def test_compare_matmul_precision_reports_when_the_predecessor_sealed_none() -> None:
    comparison = sut.compare_matmul_precision(None, {"float32_matmul_precision": "highest"})
    assert comparison["compared"] is False
    assert "seals no numerics.matmul_precision" in comparison["reason"]


def test_compare_matmul_precision_refuses_an_unobserved_live_state() -> None:
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.compare_matmul_precision(PREDECESSOR_MATMUL, None)


def _predecessor_receipt(tmp_path: Path, infer_config: Path) -> Path:
    receipt = {
        "backend_identity": {
            "backend": "hf",
            "adapter_identity": None,
            "infer_config": str(infer_config),
            "model_identity": {"checkpoint": "step-4887", "dtype": "float32"},
            "tokenizer_identity": {"vocab_size": 151936},
            "repetition_penalty_stratum": 1.0,
        },
        "numerics": {"matmul_precision": dict(PREDECESSOR_MATMUL)},
    }
    path = tmp_path / "shard-receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_validate_runtime_identity_records_the_bounded_infer_config_caveat(
    tmp_path: Path,
) -> None:
    infer_config = tmp_path / "infer.yaml"
    infer_config.write_text("model: qwen3-vl-2b\n", encoding="utf-8")
    frozen = _predecessor_receipt(tmp_path, infer_config)
    observed = {
        "backend": "hf",
        "model_identity": {"checkpoint": "step-4887", "dtype": "float32"},
        "tokenizer_identity": {"vocab_size": 151936},
    }
    numerics = {
        "infer_config_path": str(infer_config),
        "infer_config_sha256": hashlib.sha256(infer_config.read_bytes()).hexdigest(),
        "matmul_precision": {**PREDECESSOR_MATMUL, "cudnn_allow_tf32": False},
    }

    bound = sut.validate_runtime_identity(observed, frozen, numerics=numerics)

    assert bound["frozen_identity_bound"] is True
    # The predecessor sealed a path only, so the digest is recomputed now and
    # must never be presented as a historical content comparison.
    assert (
        bound["frozen_identity_infer_config_sha256_provenance"]
        == "recomputed_now_from_the_declared_path_not_sealed_by_the_predecessor"
    )
    assert "no content hash" in bound["frozen_identity_infer_config_caveat"]
    assert bound["frozen_identity_matmul_precision"]["compared"] is True
    assert bound["frozen_identity_matmul_precision"]["reported_field_differences"] == [
        "cudnn_allow_tf32"
    ]


def test_validate_runtime_identity_fails_closed_on_predecessor_matmul_drift(
    tmp_path: Path,
) -> None:
    infer_config = tmp_path / "infer.yaml"
    infer_config.write_text("model: qwen3-vl-2b\n", encoding="utf-8")
    frozen = _predecessor_receipt(tmp_path, infer_config)
    observed = {
        "backend": "hf",
        "model_identity": {"checkpoint": "step-4887", "dtype": "float32"},
        "tokenizer_identity": {"vocab_size": 151936},
    }
    numerics = {
        "infer_config_path": str(infer_config),
        "infer_config_sha256": hashlib.sha256(infer_config.read_bytes()).hexdigest(),
        "matmul_precision": {**PREDECESSOR_MATMUL, "float32_matmul_precision": "high"},
    }

    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.validate_runtime_identity(observed, frozen, numerics=numerics)
    assert "float32_matmul_precision" in str(excinfo.value)


def test_validate_runtime_identity_refuses_a_drifted_infer_config_without_overclaiming(
    tmp_path: Path,
) -> None:
    infer_config = tmp_path / "infer.yaml"
    infer_config.write_text("model: qwen3-vl-2b\n", encoding="utf-8")
    frozen = _predecessor_receipt(tmp_path, infer_config)
    observed = {
        "backend": "hf",
        "model_identity": {"checkpoint": "step-4887", "dtype": "float32"},
        "tokenizer_identity": {"vocab_size": 151936},
    }
    numerics = {
        "infer_config_path": str(infer_config),
        "infer_config_sha256": "a-different-config-digest",
        "matmul_precision": {**PREDECESSOR_MATMUL, "cudnn_allow_tf32": False},
    }

    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.validate_runtime_identity(observed, frozen, numerics=numerics)
    message = str(excinfo.value)
    assert "as it stands now" in message
    assert "no content hash" in message


def test_validate_runtime_identity_refuses_a_frozen_identity_from_another_checkpoint(
    tmp_path: Path,
) -> None:
    infer_config = tmp_path / "infer.yaml"
    infer_config.write_text("model: qwen3-vl-2b\n", encoding="utf-8")
    frozen = _predecessor_receipt(tmp_path, infer_config)

    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.validate_runtime_identity(
            {
                "backend": "hf",
                "model_identity": {"checkpoint": "step-9999", "dtype": "float32"},
                "tokenizer_identity": {"vocab_size": 151936},
            },
            frozen,
            numerics={"matmul_precision": dict(PREDECESSOR_MATMUL)},
        )
    assert "model_identity" in str(excinfo.value)


# --- 17e. Logical context groups: one per prefill, not one per request ------


def _group_ids_for_one_boundary(
    *, owner_id: str, context_id: str, variant: str, dc_tokens: list[int], backend: str
) -> list[str]:
    """The group ids one boundary emits, exactly as ``_capture_boundary`` does.

    The release surface scores two request families (target ``D_C`` path and
    native next action) from the *bare* context, and the coordinate surface
    scores three (target-local candidates, competitor candidates, greedy row)
    from the *same* forced-``D_C`` root.  Five requests, two prefills.
    """

    bare = sut._context_group_id(  # noqa: SLF001
        gt_owner_id=owner_id,
        context_id=context_id,
        variant=variant,
        appended_digest=_sha256_json([]),
        scoring_backend=backend,
    )
    forced_dc = sut._context_group_id(  # noqa: SLF001
        gt_owner_id=owner_id,
        context_id=context_id,
        variant=variant,
        appended_digest=_sha256_json(dc_tokens),
        scoring_backend=backend,
    )
    return [bare, forced_dc]


def test_multiple_request_families_share_one_logical_context_group_per_surface() -> None:
    # Regression: keying the group per *request* made a legitimate five-request
    # boundary look like five groups, three of them "reused", and the freshness
    # attestation fired on an honest capture.
    dc_tokens = [OBJECT_REF_START, 900, OBJECT_REF_END, BOX_START]
    groups = _group_ids_for_one_boundary(
        owner_id="gt:16228:11",
        context_id="ctx:16228:b7",
        variant="at_p_plus_e",
        dc_tokens=dc_tokens,
        backend=sut.KV_CACHE_BACKEND,
    )

    assert len(groups) == 2
    assert len(set(groups)) == 2
    sut.assert_fresh_context_per_owner(groups)


def test_both_boundaries_of_one_owner_stay_fresh_across_all_five_families() -> None:
    dc_tokens = [OBJECT_REF_START, 900, OBJECT_REF_END, BOX_START]
    groups = [
        *_group_ids_for_one_boundary(
            owner_id="gt:16228:11",
            context_id="ctx:16228:b7",
            variant="at_p",
            dc_tokens=dc_tokens,
            backend=sut.KV_CACHE_BACKEND,
        ),
        *_group_ids_for_one_boundary(
            owner_id="gt:16228:11",
            context_id="ctx:16228:b8",
            variant="at_p_plus_e",
            dc_tokens=dc_tokens,
            backend=sut.KV_CACHE_BACKEND,
        ),
    ]

    assert len(set(groups)) == 4
    sut.assert_fresh_context_per_owner(groups)


def test_uncached_fallback_is_a_separate_group_not_a_reused_cache() -> None:
    # unit.md requires an uncached re-read of the affected surface; it runs on
    # its own fresh state, so it must not trip the freshness attestation.
    dc_tokens = [OBJECT_REF_START, 900, OBJECT_REF_END, BOX_START]
    cached = _group_ids_for_one_boundary(
        owner_id="gt:16228:11",
        context_id="ctx:16228:b7",
        variant="at_p_plus_e",
        dc_tokens=dc_tokens,
        backend=sut.KV_CACHE_BACKEND,
    )
    uncached = _group_ids_for_one_boundary(
        owner_id="gt:16228:11",
        context_id="ctx:16228:b7",
        variant="at_p_plus_e",
        dc_tokens=dc_tokens,
        backend=sut.UNCACHED_BACKEND,
    )

    assert set(cached).isdisjoint(uncached)
    sut.assert_fresh_context_per_owner([*cached, *uncached])


def test_a_genuinely_reused_group_still_fails_closed() -> None:
    dc_tokens = [OBJECT_REF_START, 900, OBJECT_REF_END, BOX_START]
    groups = _group_ids_for_one_boundary(
        owner_id="gt:16228:11",
        context_id="ctx:16228:b7",
        variant="at_p_plus_e",
        dc_tokens=dc_tokens,
        backend=sut.KV_CACHE_BACKEND,
    )

    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.assert_fresh_context_per_owner([*groups, groups[0]])
    assert "was reused" in str(excinfo.value)


# --- 17f. A smoke seals what it executed, not only what it published --------


def test_smoke_counters_separate_published_evidence_from_executed_work() -> None:
    # Regression: the smoke handed each role a throwaway ``CaptureState``, so
    # the receipt sealed ``logical_context_group_count: 0`` while twelve groups
    # had actually been executed.  Zero published rows is contractual; zero
    # executed groups would have been false.
    state = sut.CaptureState()
    state.context_group_ids.extend(f"group-{index}" for index in range(12))
    state.score_rows.extend({"request_id": f"req-{index}"} for index in range(30))

    counters = sut.smoke_executed_counters(state)

    assert counters["score_row_count"] == 0
    assert counters["executed_unpublished_score_row_count"] == 30
    assert counters["logical_context_group_count"] == 12
    assert counters["logical_context_groups_sha256"] == _sha256_json(
        sorted(state.context_group_ids)
    )


def test_smoke_counters_are_order_independent_for_a_byte_identical_rerun() -> None:
    forward = sut.CaptureState()
    forward.context_group_ids.extend(["b", "a", "c"])
    reversed_state = sut.CaptureState()
    reversed_state.context_group_ids.extend(["c", "b", "a"])

    assert sut.smoke_executed_counters(forward) == sut.smoke_executed_counters(
        reversed_state
    )


# ---------------------------------------------------------------------------
# 18. Rank is not support (conclusion-changing regression), calibrated
#     target-local support under both ambiguity bounds, and the full-stream
#     cached-versus-uncached parity gate.
# ---------------------------------------------------------------------------


# --- 18a. The two surfaces must never be aliased again ---------------------


def test_owner_rank_result_carries_no_support_field() -> None:
    # The frozen census invariant: "rank and margin are a routing/competition
    # surface and are never a support criterion".  Aliasing them made
    # realization_fail unreachable and inflated release_lost.
    fields = {field.name for field in dataclasses.fields(sut.OwnerRankResult)}
    assert "support_disposition" not in fields
    assert "family_rank_disposition" in fields


def test_rank_and_support_vocabularies_are_disjoint() -> None:
    assert sut.RANK_DISPOSITIONS.isdisjoint(sut.SUPPORT_DISPOSITIONS)
    assert sut.SUPPORT_CALIBRATION_UNAVAILABLE in sut.SUPPORT_DISPOSITIONS


def test_likelihood_displacement_is_independent_of_calibrated_support() -> None:
    # Branch 1 is a competition statement only: a named other owner holding the
    # best identifiable candidate at a strictly negative margin.
    displaced = sut.classify_displacement(
        target_owner_id="gt:target",
        owner_rank=_owner_rank(
            target_rank=2,
            best_competitor_owner_id="gt:other",
            margin=-0.5,
            family_rank_disposition="target_outranked",
        ),
        greedy_status="unmatched",
        greedy_owner_match=None,
    )
    assert displaced.likelihood_displaced is True
    assert displaced.likelihood_displaced_owner_id == "gt:other"


def test_realization_fail_is_reachable_once_support_is_calibrated() -> None:
    # The defect made this branch dead: rank-derived "unsupported" always
    # implied a negative margin, so branch 1 fired first and branch 3 could
    # never be reached.  With calibrated support the target can rank first
    # while its own landscape is flat.
    not_displaced = sut.classify_displacement(
        target_owner_id="gt:target",
        owner_rank=_owner_rank(
            target_rank=1,
            best_competitor_owner_id="gt:other",
            margin=+0.7,
            family_rank_disposition="target_ranks_first",
        ),
        greedy_status="unmatched",
        greedy_owner_match=None,
    )
    assert not_displaced.likelihood_displaced is False

    branch = sut.classify_primary_branch(
        displacement=not_displaced,
        release_observable=True,
        release_margin=-0.4,
        forced_dc_support_disposition=sut.SUPPORT_UNSUPPORTED,
        greedy_status="unmatched",
        tie_or_nonunique=False,
        missing_fields=False,
    )
    assert branch.branch == "realization_fail"


def test_release_lost_is_no_longer_inflated_by_a_target_that_merely_ranks_first() -> None:
    # Same rank-first, greedy-unmatched case as above: under the aliased
    # contract it would have read as calibrated support and landed in
    # release_lost.  Calibrated support decides it instead.
    not_displaced = sut.classify_displacement(
        target_owner_id="gt:target",
        owner_rank=_owner_rank(
            target_rank=1,
            best_competitor_owner_id="gt:other",
            margin=+0.7,
            family_rank_disposition="target_ranks_first",
        ),
        greedy_status="unmatched",
        greedy_owner_match=None,
    )
    supported = sut.classify_primary_branch(
        displacement=not_displaced,
        release_observable=True,
        release_margin=-0.4,
        forced_dc_support_disposition=sut.SUPPORT_SUPPORTED,
        greedy_status="unmatched",
        tie_or_nonunique=False,
        missing_fields=False,
    )
    assert supported.branch == "release_lost"


def test_uncalibrated_support_routes_to_ambiguous_not_to_a_guessed_branch() -> None:
    not_displaced = sut.classify_displacement(
        target_owner_id="gt:target",
        owner_rank=_owner_rank(
            target_rank=1,
            best_competitor_owner_id="gt:other",
            margin=+0.7,
            family_rank_disposition="target_ranks_first",
        ),
        greedy_status="unmatched",
        greedy_owner_match=None,
    )
    branch = sut.classify_primary_branch(
        displacement=not_displaced,
        release_observable=True,
        release_margin=-0.4,
        forced_dc_support_disposition=sut.SUPPORT_CALIBRATION_UNAVAILABLE,
        greedy_status="unmatched",
        tie_or_nonunique=False,
        missing_fields=False,
    )
    assert branch.branch == "ambiguous"


# --- 18b. The census's own support statistics, under both bounds -----------


def _probe(candidate_id: str, score: float, bank_class: str) -> "sut.BankProbe":
    return sut.BankProbe(
        candidate_id=candidate_id,
        complete_box_logprob_sum=score,
        bank_class=bank_class,
    )


def test_bank_probe_classification_matches_the_sealed_partition() -> None:
    assert (
        sut.classify_bank_probe(
            strict_assignment_status="matched",
            strict_assignment_gt_owner_id="gt:target",
            target_owner_id="gt:target",
        )
        == "strict_assigned_self"
    )
    assert (
        sut.classify_bank_probe(
            strict_assignment_status="matched",
            strict_assignment_gt_owner_id="gt:other",
            target_owner_id="gt:target",
        )
        == "other_owner_strict"
    )
    assert (
        sut.classify_bank_probe(
            strict_assignment_status="ambiguous_neutral",
            strict_assignment_gt_owner_id=None,
            target_owner_id="gt:target",
        )
        == "ambiguous_upper"
    )
    assert (
        sut.classify_bank_probe(
            strict_assignment_status="unmatched",
            strict_assignment_gt_owner_id=None,
            target_owner_id="gt:target",
        )
        == "unmatched_generator_local"
    )


def test_bound_membership_is_the_frozen_census_partition() -> None:
    # L counts only strict self-assignments; U adds ambiguous and unmatched
    # probes; a probe that strict-matches another owner is in neither.
    assert sut.SUPPORT_BOUND_MEMBERSHIP["l"] == frozenset({"strict_assigned_self"})
    assert sut.SUPPORT_BOUND_MEMBERSHIP["u"] == frozenset(
        {"strict_assigned_self", "ambiguous_upper", "unmatched_generator_local"}
    )
    # Every census partition is accounted for; a new one cannot silently fall
    # outside both bounds without this failing.
    assert set(sut.SUPPORT_PARTITIONS) >= set().union(
        *sut.SUPPORT_BOUND_MEMBERSHIP.values()
    )
    assert all(
        "other_owner_strict" not in members
        for members in sut.SUPPORT_BOUND_MEMBERSHIP.values()
    )


def test_unique_population_log_posteriors_normalizes_over_the_deduplicated_group() -> None:
    posteriors, size = sut.unique_population_log_posteriors(
        [("cand:a", -1.0), ("cand:b", -2.0), ("cand:a", -1.0)]
    )
    assert size == 2
    assert set(posteriors) == {"cand:a", "cand:b"}
    assert math.isclose(
        math.fsum(math.exp(value) for value in posteriors.values()), 1.0, rel_tol=1e-12
    )


def test_unique_population_refuses_a_duplicate_candidate_scored_differently() -> None:
    # The same physical candidate reached through both families was scored from
    # the same root, so its two readings must agree; collapsing by max would
    # hide a broken request or batching alignment.
    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.unique_population_log_posteriors([("cand:a", -1.0), ("cand:a", -1.5)])
    message = str(excinfo.value)
    assert "cand:a" in message
    assert "same root" in message


def test_unique_population_admits_a_duplicate_inside_float_noise() -> None:
    posteriors, size = sut.unique_population_log_posteriors(
        [("cand:a", -1.0), ("cand:a", -1.0 + 1e-12), ("cand:b", -2.0)]
    )
    assert size == 2
    assert set(posteriors) == {"cand:a", "cand:b"}


def test_unique_population_refuses_an_empty_group() -> None:
    with pytest.raises(sut.CrossingBoundaryContractError):
        sut.unique_population_log_posteriors([])


def _bound_features(bound: str) -> "sut.SupportBoundFeatures":
    probes = [
        _probe("cand:self_best", -1.0, "strict_assigned_self"),
        _probe("cand:self_low", -5.0, "strict_assigned_self"),
        _probe("cand:ambiguous", -0.5, "ambiguous_upper"),
        _probe("cand:unmatched", -6.0, "unmatched_generator_local"),
        _probe("cand:other", -0.1, "other_owner_strict"),
    ]
    posteriors, size = sut.unique_population_log_posteriors(
        [(probe.candidate_id, probe.complete_box_logprob_sum) for probe in probes]
    )
    return sut.compute_support_bound_features(
        probes,
        bound=bound,
        log_posterior_by_key=posteriors,
        unique_population_size=size,
    )


def test_support_features_differ_between_the_u_and_l_bounds() -> None:
    upper = _bound_features("u")
    lower = _bound_features("l")

    # U admits the ambiguous probe, which is this owner's best; L does not.
    assert upper.owner_best_candidate_id == "cand:ambiguous"
    assert lower.owner_best_candidate_id == "cand:self_best"
    assert upper.bank_size == 4
    assert lower.bank_size == 2
    # Neither bound admits ``cand:other``, even though it is the highest-scoring
    # probe in the whole bank: it strict-matches another owner.
    assert upper.owner_best_candidate_id != "cand:other"
    assert lower.owner_best_candidate_id != "cand:other"
    assert upper.peak_lift != lower.peak_lift


def test_local_concentration_is_best_minus_the_median_of_that_bound() -> None:
    lower = _bound_features("l")
    # Bank under L is {-1.0, -5.0}; median -3.0; best -1.0.
    assert lower.bank_median == pytest.approx(-3.0)
    assert lower.local_concentration == pytest.approx(-1.0 - (-3.0))


def test_peak_lift_is_log_posterior_plus_log_unique_population() -> None:
    upper = _bound_features("u")
    posteriors, size = sut.unique_population_log_posteriors(
        [
            ("cand:self_best", -1.0),
            ("cand:self_low", -5.0),
            ("cand:ambiguous", -0.5),
            ("cand:unmatched", -6.0),
            ("cand:other", -0.1),
        ]
    )
    assert upper.peak_lift == pytest.approx(
        posteriors["cand:ambiguous"] + math.log(float(size))
    )


def test_support_features_are_empty_when_a_bound_has_no_probe() -> None:
    posteriors, size = sut.unique_population_log_posteriors([("cand:other", -0.1)])
    features = sut.compute_support_bound_features(
        [_probe("cand:other", -0.1, "other_owner_strict")],
        bound="l",
        log_posterior_by_key=posteriors,
        unique_population_size=size,
    )
    assert features.bank_size == 0
    assert features.peak_lift is None
    assert features.local_concentration is None


def test_support_clears_only_when_both_statistics_beat_threshold_plus_epsilon() -> None:
    features = _bound_features("l")
    peak_lift = float(features.peak_lift)
    local_concentration = float(features.local_concentration)

    supported = sut.evaluate_target_local_support(
        features,
        peak_lift_threshold=peak_lift - 1.0,
        local_concentration_threshold=local_concentration - 1.0,
        calibration_source="test",
    )
    assert supported.support_disposition == sut.SUPPORT_SUPPORTED

    # Exactly at the threshold is *not* support: epsilon must be cleared too.
    at_threshold = sut.evaluate_target_local_support(
        features,
        peak_lift_threshold=peak_lift,
        local_concentration_threshold=local_concentration,
        calibration_source="test",
    )
    assert at_threshold.support_disposition == sut.SUPPORT_UNSUPPORTED


@pytest.mark.parametrize("failing", ["peak_lift", "local_concentration"])
def test_support_requires_both_statistics_not_either(failing: str) -> None:
    features = _bound_features("l")
    thresholds = {
        "peak_lift_threshold": float(features.peak_lift) - 1.0,
        "local_concentration_threshold": float(features.local_concentration) - 1.0,
    }
    thresholds[f"{failing}_threshold"] += 10.0
    result = sut.evaluate_target_local_support(
        features, calibration_source="test", **thresholds
    )
    assert result.support_disposition == sut.SUPPORT_UNSUPPORTED


def test_support_is_unavailable_without_a_bound_calibration() -> None:
    result = sut.evaluate_target_local_support(
        _bound_features("u"),
        peak_lift_threshold=None,
        local_concentration_threshold=None,
        calibration_source="absent_from_the_sealed_plan",
    )
    assert result.support_disposition == sut.SUPPORT_CALIBRATION_UNAVAILABLE


def _plan_with_calibration(**overrides: Any) -> Any:
    """A plan carrying the calibration block the CPU plan builder publishes."""

    block = {
        "support_contract_id": sut.SUPPORT_CALIBRATION_CONTRACT_ID,
        "criterion_id": "local_peak_lift_and_local_concentration_under_both_ambiguity_bounds",
        "derived_here": False,
        "bounds": {"primary_bound": "u", "sensitivity_bound": "l"},
        "rank_and_margin": {
            "rank_is_a_support_input": False,
            "margin_is_a_support_input": False,
        },
        "thresholds": {
            "theta_peak_lift": 2.3884847780085985,
            "theta_local_concentration": 1.63233060836792,
            "epsilon": 0.002,
            "quantile": 0.1,
            "observation_count": 70,
            "calibration_stratum": "pooled_discovery_native_true_positives",
        },
        "source": {
            "calibration_sha256": "9dd6d7646fc55db6155124dc4bbfa46642b32b006d39758bd2e24d1ca97058c5",
            "path": "phases/discovery-sealed/support-calibration.json",
            "run_root": "/data/CoordExp/outputs/research/somewhere",
            "merge_source_sha256": "1b663f7b62b1ffa6ac4424d17f3eaf71c2d3feefded1a1d01cc55087e54efb95",
        },
    }
    for path, value in overrides.items():
        section, _, field = path.partition(".")
        if field:
            block[section] = {**block[section], field: value}
        else:
            block[section] = value
    plan = _two_image_plan()
    return dataclasses.replace(
        plan, manifest={**plan.manifest, sut.SUPPORT_CALIBRATION_KEY: block}
    )


def test_plan_support_calibration_absent_is_reported_not_defaulted() -> None:
    assert sut.read_plan_support_calibration(_two_image_plan()) is None


def test_plan_support_calibration_names_the_census_it_was_decided_against() -> None:
    lineage = sut.read_plan_support_calibration(_plan_with_calibration())

    assert lineage["support_contract_id"] == sut.SUPPORT_CALIBRATION_CONTRACT_ID
    assert lineage["theta_peak_lift"] == pytest.approx(2.3884847780085985)
    assert lineage["theta_local_concentration"] == pytest.approx(1.63233060836792)
    assert lineage["epsilon"] == pytest.approx(0.002)
    assert lineage["quantile"] == 0.1
    assert lineage["observation_count"] == 70
    assert lineage["calibration_sha256"].startswith("9dd6d764")
    # The plan binds a sealed census calibration; it never authors one.
    assert lineage["derived_here"] is False


def test_plan_support_calibration_uses_the_key_the_builder_publishes() -> None:
    # Regression: this reader was keyed on a name the plan builder never
    # publishes, so a plan that *does* carry a calibration read as absent.
    assert sut.SUPPORT_CALIBRATION_KEY == "support_calibration"
    assert sut.SUPPORT_CALIBRATION_REQUIRED_FIELDS == (
        "theta_peak_lift",
        "theta_local_concentration",
    )
    plan = _plan_with_calibration()
    assert sut.SUPPORT_CALIBRATION_KEY in plan.manifest
    assert sut.read_plan_support_calibration(plan) is not None


def test_plan_support_calibration_refuses_a_foreign_contract() -> None:
    plan = _plan_with_calibration(support_contract_id="some-other-contract.v9")
    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.read_plan_support_calibration(plan)
    assert "not proven" in str(excinfo.value)


@pytest.mark.parametrize(
    "flag", ["rank_is_a_support_input", "margin_is_a_support_input"]
)
def test_plan_support_calibration_refuses_a_rank_or_margin_support_input(
    flag: str,
) -> None:
    plan = _plan_with_calibration(**{f"rank_and_margin.{flag}": True})
    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.read_plan_support_calibration(plan)
    assert "frozen census invariant" in str(excinfo.value)


@pytest.mark.parametrize(
    "threshold", ["theta_peak_lift", "theta_local_concentration"]
)
def test_plan_support_calibration_refuses_a_missing_threshold(threshold: str) -> None:
    plan = _plan_with_calibration(**{f"thresholds.{threshold}": None})
    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.read_plan_support_calibration(plan)
    assert threshold in str(excinfo.value)


def test_plan_support_calibration_refuses_a_block_without_thresholds() -> None:
    plan = _plan_with_calibration(thresholds={})
    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.read_plan_support_calibration(plan)
    assert "theta_peak_lift" in str(excinfo.value)


# --- 18c. Full-stream cached-versus-uncached parity ------------------------


def test_parity_compares_every_token_not_only_the_first(  # non-first-token drift
) -> None:
    # Drift at greedy step 3 with an unchanged summary scalar and unchanged
    # first token: the compressed comparison admitted this.
    result = sut.evaluate_cache_parity(
        _parity_inputs(
            uncached_streams=_streams(
                selected=[-1.0, -2.0, -3.0 - 5e-3], argmax=[151700, 151701, 151702]
            )
        )
    )
    assert result.status == "uncached_fallback"
    assert "selected_logit" in result.mismatched_fields
    assert f"selected_logit:{sut.SURFACE_COORDINATE}" in result.mismatched_fields
    assert result.max_selected_logit_abs_diff == pytest.approx(5e-3)


def test_parity_compares_every_argmax_not_only_the_first() -> None:
    result = sut.evaluate_cache_parity(
        _parity_inputs(
            uncached_streams=_streams(
                selected=[-1.0, -2.0, -3.0], argmax=[151700, 151701, 999999]
            )
        )
    )
    assert result.status == "uncached_fallback"
    assert "argmax" in result.mismatched_fields


def test_parity_detects_a_release_margin_sign_flip_alone() -> None:
    # Rank, support, owner match and branch all unchanged; only the release
    # margin sign moves.  The compressed comparison missed this entirely.
    result = sut.evaluate_cache_parity(
        _parity_inputs(cached_release_margin_sign=-1, uncached_release_margin_sign=1)
    )
    assert result.status == "uncached_fallback"
    assert "release_margin_sign" in result.mismatched_fields


def test_parity_refuses_misaligned_request_streams() -> None:
    result = sut.evaluate_cache_parity(
        _parity_inputs(
            uncached_streams=_streams(
                selected=[-1.0, -2.0, -3.0],
                argmax=[151700, 151701, 151702],
                request_ids=("req:different",),
            )
        )
    )
    assert result.status == "uncached_fallback"
    assert any(
        name.startswith("stream_alignment:") for name in result.mismatched_fields
    )


def test_parity_refuses_a_length_mismatch() -> None:
    result = sut.evaluate_cache_parity(
        _parity_inputs(
            uncached_streams=_streams(selected=[-1.0, -2.0], argmax=[151700, 151701])
        )
    )
    assert result.status == "uncached_fallback"
    assert any(
        name.startswith("stream_alignment:") for name in result.mismatched_fields
    )


def test_parity_refuses_absent_streams_instead_of_silently_admitting() -> None:
    # If a call site ever stops passing the score rows, the defaults must not
    # admit a cache on the strength of a few summary scalars.
    result = sut.evaluate_cache_parity(
        _parity_inputs(cached_streams={}, uncached_streams={})
    )
    assert result.status == "uncached_fallback"
    for surface in sut.SURFACES:
        assert f"stream_missing:{surface}" in result.mismatched_fields


def test_parity_refuses_aligned_but_empty_streams() -> None:
    empty = _streams(selected=[], argmax=[], request_ids=())
    result = sut.evaluate_cache_parity(
        _parity_inputs(cached_streams=empty, uncached_streams=empty)
    )
    assert result.status == "uncached_fallback"
    for surface in sut.SURFACES:
        assert f"stream_empty:{surface}" in result.mismatched_fields


def test_parity_refuses_a_surface_present_on_only_one_side() -> None:
    one_sided = {
        sut.SURFACE_RELEASE: sut.SurfaceParityStreams(
            request_ids=("req:a",),
            selected_logprobs=(-1.0,),
            argmax_token_ids=(151700,),
        )
    }
    result = sut.evaluate_cache_parity(_parity_inputs(uncached_streams=one_sided))
    assert result.status == "uncached_fallback"
    assert f"stream_missing:{sut.SURFACE_COORDINATE}" in result.mismatched_fields


def test_parity_reports_per_surface_token_counts_for_the_receipt() -> None:
    result = sut.evaluate_cache_parity(_parity_inputs())
    assert result.status == "cache_admitted"
    per_surface = {entry.surface: entry for entry in result.per_surface}
    assert set(per_surface) == set(sut.SURFACES)
    for entry in per_surface.values():
        assert entry.aligned is True
        assert entry.compared_token_count == 3
        assert entry.argmax_parity is True
        assert entry.max_selected_logit_abs_diff <= (
            sut.CACHE_PARITY_MAX_SELECTED_LOGIT_ABS_DIFF
        )


def test_parity_max_is_global_over_every_surface_and_the_summary_scalar() -> None:
    result = sut.evaluate_cache_parity(
        _parity_inputs(
            uncached_streams=_streams(
                selected=[-1.0, -2.0, -3.0 - 4e-4], argmax=[151700, 151701, 151702]
            )
        )
    )
    # 5e-4 from the release summary scalar, 4e-4 from a coordinate token: the
    # reported maximum is the larger of the two, and both stay under the bound.
    assert result.status == "cache_admitted"
    assert result.max_selected_logit_abs_diff == pytest.approx(5e-4)


def test_parity_streams_by_surface_groups_families_and_orders_by_request_id() -> None:
    rows = [
        {
            "request_id": "req:b",
            "request_family": sut.REQUEST_NATIVE_NEXT_ACTION,
            "selected_logprobs": [-2.0],
            "argmax_token_ids": [2],
        },
        {
            "request_id": "req:a",
            "request_family": sut.REQUEST_NATURAL_RELEASE,
            "selected_logprobs": [-1.0],
            "argmax_token_ids": [1],
        },
        {
            "request_id": "req:c",
            "request_family": sut.REQUEST_COORDINATE_GREEDY,
            "selected_logprobs": [-3.0],
            "argmax_token_ids": [3],
        },
    ]
    streams = sut.parity_streams_by_surface(rows)

    assert set(streams) == set(sut.SURFACES)
    release = streams[sut.SURFACE_RELEASE]
    # Request-id order, not execution order, is what makes the two sides
    # comparable position by position.
    assert release.request_ids == ("req:a", "req:b")
    assert release.selected_logprobs == (-1.0, -2.0)
    assert streams[sut.SURFACE_COORDINATE].request_ids == ("req:c",)


def test_parity_streams_are_stable_under_input_ordering() -> None:
    rows = [
        {
            "request_id": f"req:{name}",
            "request_family": family,
            "selected_logprobs": [value],
            "argmax_token_ids": [index],
        }
        for index, (name, family, value) in enumerate(
            [
                ("a", sut.REQUEST_NATURAL_RELEASE, -1.0),
                ("b", sut.REQUEST_NATIVE_NEXT_ACTION, -2.0),
                ("c", sut.REQUEST_COORDINATE_TARGET_LOCAL, -3.0),
                ("d", sut.REQUEST_COORDINATE_COMPETITOR, -4.0),
            ]
        )
    ]
    assert sut.parity_streams_by_surface(rows) == sut.parity_streams_by_surface(
        list(reversed(rows))
    )


# ---------------------------------------------------------------------------
# 19. Consumption of the sealed plan's support contract.  The thresholds and
#     the per-bound membership are the plan builder's to publish and this
#     unit's only to read: nothing here re-derives a quantile, refits a
#     threshold, or reclassifies a probe.
# ---------------------------------------------------------------------------


def _target_local_request(**overrides: Any) -> dict[str, Any]:
    lower = ["cand:self_best", "cand:self_low"]
    upper = [*lower, "cand:ambiguous", "cand:unmatched"]
    calibration = {
        "support_contract_id": sut.SUPPORT_CALIBRATION_CONTRACT_ID,
        "calibration_sha256": "9dd6d7646fc5",
        "theta_peak_lift": 2.3884847780085985,
        "theta_local_concentration": 1.63233060836792,
        "epsilon": 0.002,
        "primary_bound": "u",
        "rank_is_a_support_input": False,
        "margin_is_a_support_input": False,
    }
    calibration.update(overrides.pop("calibration_overrides", {}))
    family = {
        "candidate_ids": upper,
        "support_calibration": calibration,
        "bounds": {
            "l": {
                "candidate_ids": lower,
                "candidate_ids_sha256": _sha256_json(lower),
                "owner_context_path": "ambiguity_excluded_l",
            },
            "u": {
                "candidate_ids": upper,
                "candidate_ids_sha256": _sha256_json(upper),
                "owner_context_path": "ambiguity_included_u",
            },
        },
    }
    family.update(overrides.pop("family_overrides", {}))
    return {"request_family": sut.REQUEST_COORDINATE_TARGET_LOCAL, "candidate_family": family}


def test_request_support_calibration_is_read_from_the_sealed_request() -> None:
    calibration = sut.read_request_support_calibration(_target_local_request())
    assert calibration["peak_lift_threshold"] == pytest.approx(2.3884847780085985)
    assert calibration["local_concentration_threshold"] == pytest.approx(1.63233060836792)
    assert calibration["epsilon"] == pytest.approx(0.002)
    assert "9dd6d7646fc5" in calibration["source"]


def test_request_support_calibration_absent_is_unavailable_not_defaulted() -> None:
    request = _target_local_request(family_overrides={"support_calibration": None})
    assert sut.read_request_support_calibration(request) is None


def test_request_support_calibration_refuses_a_foreign_contract() -> None:
    request = _target_local_request(
        calibration_overrides={"support_contract_id": "some-other-contract.v9"}
    )
    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.read_request_support_calibration(request)
    assert "not proven" in str(excinfo.value)


@pytest.mark.parametrize(
    "flag", ["rank_is_a_support_input", "margin_is_a_support_input"]
)
def test_request_support_calibration_refuses_a_rank_or_margin_support_input(
    flag: str,
) -> None:
    # The frozen census invariant, enforced at the point of consumption: a
    # calibration that made rank or margin a support input would reintroduce
    # exactly the aliasing this unit had to remove.
    request = _target_local_request(calibration_overrides={flag: True})
    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.read_request_support_calibration(request)
    assert "frozen census invariant" in str(excinfo.value)


@pytest.mark.parametrize("threshold", ["theta_peak_lift", "theta_local_concentration"])
def test_request_support_calibration_refuses_a_missing_threshold(threshold: str) -> None:
    request = _target_local_request(calibration_overrides={threshold: None})
    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.read_request_support_calibration(request)
    assert threshold in str(excinfo.value)


def test_sealed_bound_membership_is_read_per_bound_and_digest_verified() -> None:
    request = _target_local_request()
    assert sut.sealed_bound_candidate_ids(request, bound="l") == [
        "cand:self_best",
        "cand:self_low",
    ]
    assert len(sut.sealed_bound_candidate_ids(request, bound="u")) == 4


def test_sealed_bound_membership_refuses_a_tampered_digest() -> None:
    request = _target_local_request()
    request["candidate_family"]["bounds"]["u"]["candidate_ids"].append("cand:smuggled")

    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.sealed_bound_candidate_ids(request, bound="u")
    assert "digest does not reconstruct" in str(excinfo.value)


def test_sealed_bound_membership_refuses_an_absent_bound() -> None:
    request = _target_local_request()
    del request["candidate_family"]["bounds"]["l"]

    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.sealed_bound_candidate_ids(request, bound="l")
    assert "no membership for bound" in str(excinfo.value)


def test_sealed_membership_never_admits_an_other_owner_strict_probe() -> None:
    # The plan publishes those separately; they belong to the collision
    # diagnostic and to neither support bound.
    request = _target_local_request()
    every_member = set(sut.sealed_bound_candidate_ids(request, bound="u")) | set(
        sut.sealed_bound_candidate_ids(request, bound="l")
    )
    assert "cand:other" not in every_member


# ---------------------------------------------------------------------------
# 20. The production HF session seam.  ``build_hf_session_spec`` resolves a
#     session from ``plan.images`` alone; this unit's sealed plan carries the
#     same rows under ``inputs.images_by_id``.  The fake backend never touches
#     this path, so it went unexercised until a live smoke hit it.
# ---------------------------------------------------------------------------


class _CrossingPlanInputsLike:
    """The shape the real crossing ``PlanInputs`` presents: no ``.images``."""

    def __init__(self, images: dict[str, Any] | None = None) -> None:
        self.images_by_id = (
            {
                "4134": {
                    "prompt_token_ids": [11, 12, 13],
                    "executed_media_sha256": "media-digest-4134",
                }
            }
            if images is None
            else images
        )


def _session_plan(images: dict[str, Any] | None = None) -> Any:
    return sut.SealedPlan(
        plan_dir=Path("/nonexistent/plan"),
        manifest={},
        cohort_rows=[],
        control_rows=[],
        request_rows=[],
        plan_file_sha256={},
        inputs=_CrossingPlanInputsLike(images),
        candidates_by_id={},
        owners_by_image={},
    )


def test_census_session_view_exposes_images_by_reference_not_by_copy() -> None:
    # A copy would let the session and the scored prefixes drift apart; the
    # adapter renames the attribute and nothing else.
    plan = _session_plan()
    view = sut.census_session_plan_view(plan)

    assert view.images is plan.inputs.images_by_id
    assert view.images["4134"]["prompt_token_ids"] == [11, 12, 13]
    assert view.images["4134"]["executed_media_sha256"] == "media-digest-4134"


def test_census_session_view_fails_closed_without_an_image_registry() -> None:
    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut.census_session_plan_view(_session_plan(images={}))
    assert "inputs.images_by_id" in str(excinfo.value)


def test_real_build_hf_session_spec_accepts_the_view_and_never_raises_attribute_error(
    tmp_path: Path,
) -> None:
    """The exact seam a live smoke hit, called for real, before any model load.

    Regression: ``_open_backend`` passed ``plan.inputs``, which has no
    ``images``, so the production path died with ``AttributeError`` at
    ``build_hf_session_spec``'s first statement.  Calling the real function with
    an unregistered image id proves the attribute now resolves and that the
    failure is the registry contract, not the seam.  ``load_infer_config`` and
    the model are never reached.
    """

    from scripts.research import score_sorted_owner_accessibility_census_shard as shard

    view = sut.census_session_plan_view(_session_plan())
    with pytest.raises(shard.ShardContractError) as excinfo:
        shard.build_hf_session_spec(
            view, "not-a-registered-image", infer_config=tmp_path / "never-read.yaml"
        )
    assert "is not in the plan's image registry" in str(excinfo.value)


def test_open_backend_hands_the_session_seam_everything_it_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_open_backend`` must satisfy every access the real spec builder makes.

    The spy performs exactly the three reads ``build_hf_session_spec`` performs
    on its plan argument, so a regression that hands over the wrong object
    fails here rather than in a live smoke.  ``open_hf_backend`` is stubbed to
    prove no model is opened.
    """

    from scripts.research import score_sorted_owner_accessibility_census_shard as shard

    infer_config = tmp_path / "infer.yaml"
    infer_config.write_text("backend: {type: hf}\n", encoding="utf-8")
    observed: dict[str, Any] = {}

    def _spy_build(plan_arg: Any, image_id: str, *, infer_config: Path) -> Any:
        observed["member"] = image_id in plan_arg.images
        observed["prompt_token_ids"] = list(
            plan_arg.images[image_id]["prompt_token_ids"]
        )
        observed["executed_media_sha256"] = str(
            plan_arg.images[image_id]["executed_media_sha256"]
        )
        observed["infer_config"] = Path(infer_config)
        return "session-spec"

    def _refuse_open(spec: Any) -> Any:
        observed["opened_spec"] = spec
        return "opened-session"

    monkeypatch.setattr(shard, "build_hf_session_spec", _spy_build)
    monkeypatch.setattr(shard, "open_hf_backend", _refuse_open)

    args = argparse.Namespace(backend="hf", infer_config=infer_config)
    result = sut._open_backend(args, _session_plan(), "4134")  # noqa: SLF001

    assert result == "opened-session"
    assert observed["member"] is True
    # The literal sealed rows, unchanged by the adapter.
    assert observed["prompt_token_ids"] == [11, 12, 13]
    assert observed["executed_media_sha256"] == "media-digest-4134"
    assert observed["infer_config"] == infer_config.resolve()


def test_open_backend_still_refuses_live_scoring_without_an_infer_config() -> None:
    args = argparse.Namespace(backend="hf", infer_config=None)
    with pytest.raises(sut.CrossingBoundaryContractError) as excinfo:
        sut._open_backend(args, _session_plan(), "4134")  # noqa: SLF001
    assert "--infer-config" in str(excinfo.value)


def test_open_backend_fake_path_opens_no_session_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fake path must not touch the production session seam at all, which is
    # precisely why it never caught the missing attribute.
    from scripts.research import score_sorted_owner_accessibility_census_shard as shard

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the fake backend must not build a production session spec")

    monkeypatch.setattr(shard, "build_hf_session_spec", _explode)

    args = argparse.Namespace(backend="fake", infer_config=None)
    with sut._open_backend(args, _session_plan(), "4134") as backend:  # noqa: SLF001
        assert backend.identity["backend"] == "fake"
