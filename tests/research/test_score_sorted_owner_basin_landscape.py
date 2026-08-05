"""Unit tests for the Task-3 sorted-owner-basin-landscape runtime scorer.

Everything here runs on fakes/mocks: no GPU launch, no real HF backend
session, no real ``transformers`` model. The KV-cache branching engine is
exercised through a pure-Python fake cache backend whose deterministic
"logits" are a function of its accumulated token history, which makes
cross-branch cache contamination directly observable (see the
order-invariance tests below).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import runpy
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from scripts.research import score_sorted_owner_basin_landscape as sut


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 64
COORD_START = 10
COORD_END = 30  # 20 coordinate bins, well inside VOCAB_SIZE

TEST_ATTESTATION = sut.build_attestation_context(
    expected_vocab_size=VOCAB_SIZE,
    tokenizer_identity={"tokenizer": "fake-v1"},
    model_identity={"model": "fake-v1"},
    rule_digest="rule-digest-abc",
    runtime_receipt_id="runtime-receipt-1",
)


def _spike_logits(
    vocab_size: int, token_id: int, *, spike: float = 8.0, base: float = 0.0
) -> torch.Tensor:
    values = torch.full((vocab_size,), base, dtype=torch.float32)
    values[token_id] = spike
    return values


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _owner_ledger_row(
    *,
    diagnostic_owner_id: str,
    status: str,
    image_id: str = "image-1",
    gt_owner_id: str | None = None,
) -> dict[str, Any]:
    return {
        "diagnostic_owner_id": diagnostic_owner_id,
        "status": status,
        "image_id": image_id,
        "gt_owner_id": gt_owner_id,
    }


def _pred_row_ledger_row(
    *, pred_row_id: str, image_id: str = "image-1", repetition_penalty: float = 1.0
) -> dict[str, Any]:
    return {
        "pred_row_id": pred_row_id,
        "image_id": image_id,
        "repetition_penalty": repetition_penalty,
    }


def _decision_rules_payload(
    *,
    owner_canonical_description: dict[str, Any] | None = None,
    foil_set_digests: dict[str, str] | None = None,
    model_vocab_size: int = VOCAB_SIZE,
) -> dict[str, Any]:
    content = {
        "schema_version": sut.DECISION_RULES_SCHEMA_VERSION,
        "schema_tokens": {
            "object_ref_start_token_id": 0,
            "object_ref_end_token_id": 1,
            "box_start_token_id": 2,
            "box_end_token_id": 3,
            "coordinate_token_id_start": COORD_START,
            "coordinate_token_id_end_exclusive": COORD_END,
        },
        "owner_canonical_description": owner_canonical_description
        or {"gt:image-1:0": {"token_ids": [40]}},
        "foil_set_digests": foil_set_digests or {"foils-v1": "deadbeef"},
        "model_vocab_size": model_vocab_size,
        "numeric_tolerance": 1e-6,
    }
    digest = sut.sha256_json(content)
    return {**content, "rules_digest": digest}


def _write_decision_rules(path: Path, **kwargs: Any) -> dict[str, Any]:
    payload = _decision_rules_payload(**kwargs)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _candidate_payload(
    *,
    candidate_id: str = "cand-1",
    request_kind: str = "complete_box",
    prefix_token_ids: list[int] | None = None,
    prompt_prefix_token_count: int | None = None,
    coord_token_ids: list[int] | None = None,
    fixed_coord_token_ids: list[int] | None = None,
    scan_slot: str | None = None,
    context_id: str = "root",
    diagnostic_owner_id: str = "gt:image-1:0",
    image_id: str = "image-1",
    role: str = "target",
    review_status: str = "reviewed",
    candidate_kind: str = "target_anchor",
    foil_set_id: str = "foils-v1",
    identity_kind: str = "reviewed_physical_owner",
    native_repetition_penalty_stratum: float = 1.0,
    source_pred_row_id: str | None = None,
    physical_owner_hint: str | None = None,
    basin_id: str | None = None,
) -> dict[str, Any]:
    prefix = prefix_token_ids if prefix_token_ids is not None else [100, 101, 102]
    prompt_prefix_token_count = (
        len(prefix) if prompt_prefix_token_count is None else prompt_prefix_token_count
    )
    payload: dict[str, Any] = {
        "candidate_id": candidate_id,
        "diagnostic_owner_id": diagnostic_owner_id,
        "image_id": image_id,
        "context_id": context_id,
        "role": role,
        "review_status": review_status,
        "candidate_kind": candidate_kind,
        "foil_set_id": foil_set_id,
        "identity_kind": identity_kind,
        "native_repetition_penalty_stratum": native_repetition_penalty_stratum,
        "source_pred_row_id": source_pred_row_id,
        "prompt_prefix_token_count": prompt_prefix_token_count,
        "prefix_token_ids": prefix,
        "prefix_token_ids_sha256": sut.sha256_json(list(prefix)),
        "physical_owner_hint": physical_owner_hint,
        "basin_id": basin_id,
        "request_kind": request_kind,
    }
    if request_kind == "complete_box":
        coords = coord_token_ids if coord_token_ids is not None else [11, 12, 13, 14]
        payload["coord_token_ids"] = coords
        seed = {
            "candidate_kind": candidate_kind,
            "diagnostic_owner_id": diagnostic_owner_id,
            "context_id": context_id,
            "role": role,
            "foil_set_id": foil_set_id,
            "request_kind": request_kind,
            "coord_token_ids": coords,
        }
    else:
        fixed = fixed_coord_token_ids if fixed_coord_token_ids is not None else []
        slot = scan_slot if scan_slot is not None else sut.COORD_SLOTS[len(fixed)]
        payload["fixed_coord_token_ids"] = fixed
        payload["scan_slot"] = slot
        seed = {
            "candidate_kind": candidate_kind,
            "diagnostic_owner_id": diagnostic_owner_id,
            "context_id": context_id,
            "role": role,
            "foil_set_id": foil_set_id,
            "request_kind": request_kind,
            "fixed_coord_token_ids": fixed,
            "scan_slot": slot,
        }
    payload["proposal_digest"] = sut.sha256_json(seed)
    return payload


def _candidate_row(**kwargs: Any) -> sut.CandidateRow:
    tmp_dir_value = kwargs.pop("_tmp_dir", None)
    if not tmp_dir_value:
        raise AssertionError("test helper misuse: _candidate_row requires _tmp_dir")
    tmp_dir = Path(tmp_dir_value)
    payload = _candidate_payload(**kwargs)
    path = tmp_dir / f"{payload['candidate_id']}.jsonl"
    _write_jsonl(path, [payload])
    return sut.load_candidate_rows(path)[0]


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def test_sha256_json_is_order_independent_over_keys() -> None:
    assert sut.sha256_json({"a": 1, "b": 2}) == sut.sha256_json({"b": 2, "a": 1})


def test_sha256_file_matches_manual_digest(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello")
    import hashlib

    assert sut.sha256_file(path) == hashlib.sha256(b"hello").hexdigest()


# ---------------------------------------------------------------------------
# Owner ledger
# ---------------------------------------------------------------------------


def test_load_owner_ledger_accepts_valid_rows(tmp_path: Path) -> None:
    path = tmp_path / "owner-ledger.jsonl"
    _write_jsonl(
        path,
        [
            _owner_ledger_row(
                diagnostic_owner_id="gt:image-1:0",
                status="gt",
                gt_owner_id="gt:image-1:0",
            )
        ],
    )
    ledger = sut.load_owner_ledger(path)
    assert ledger["gt:image-1:0"].status == "gt"


def test_load_owner_ledger_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "owner-ledger.jsonl"
    row = _owner_ledger_row(diagnostic_owner_id="gt:image-1:0", status="gt")
    _write_jsonl(path, [row, row])
    with pytest.raises(sut.LandscapeScoringError, match="duplicate"):
        sut.load_owner_ledger(path)


def test_load_owner_ledger_rejects_status_prefix_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "owner-ledger.jsonl"
    _write_jsonl(
        path, [_owner_ledger_row(diagnostic_owner_id="aux:image-1:0", status="gt")]
    )
    with pytest.raises(sut.LandscapeScoringError, match="prefix"):
        sut.load_owner_ledger(path)


# ---------------------------------------------------------------------------
# Prediction row ledger
# ---------------------------------------------------------------------------


def test_load_prediction_row_ledger_accepts_valid_rows(tmp_path: Path) -> None:
    path = tmp_path / "prediction-row-ledger.jsonl"
    _write_jsonl(
        path, [_pred_row_ledger_row(pred_row_id="pred:1", repetition_penalty=1.10)]
    )
    ledger = sut.load_prediction_row_ledger(path)
    assert ledger["pred:1"].repetition_penalty == pytest.approx(1.10)


def test_load_prediction_row_ledger_rejects_bad_stratum(tmp_path: Path) -> None:
    path = tmp_path / "prediction-row-ledger.jsonl"
    _write_jsonl(
        path, [_pred_row_ledger_row(pred_row_id="pred:1", repetition_penalty=1.05)]
    )
    with pytest.raises(sut.LandscapeScoringError, match="established stratum"):
        sut.load_prediction_row_ledger(path)


def test_load_prediction_row_ledger_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "prediction-row-ledger.jsonl"
    row = _pred_row_ledger_row(pred_row_id="pred:1")
    _write_jsonl(path, [row, row])
    with pytest.raises(sut.LandscapeScoringError, match="duplicate"):
        sut.load_prediction_row_ledger(path)


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------


def test_load_decision_rules_accepts_valid_and_matching_digest(tmp_path: Path) -> None:
    path = tmp_path / "landscape-decision-rules.json"
    _write_decision_rules(path)
    rules = sut.load_decision_rules(path)
    assert rules.model_vocab_size == VOCAB_SIZE
    assert rules.owner_canonical_description["gt:image-1:0"] == (40,)


def test_load_decision_rules_rejects_stale_digest(tmp_path: Path) -> None:
    path = tmp_path / "landscape-decision-rules.json"
    payload = _decision_rules_payload()
    payload["rules_digest"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(sut.LandscapeScoringError, match="stale"):
        sut.load_decision_rules(path)


def test_load_decision_rules_rejects_invalid_coordinate_bounds(tmp_path: Path) -> None:
    path = tmp_path / "landscape-decision-rules.json"
    payload = _decision_rules_payload()
    payload["schema_tokens"]["coordinate_token_id_end_exclusive"] = payload[
        "schema_tokens"
    ]["coordinate_token_id_start"]
    payload["rules_digest"] = sut.sha256_json(
        {k: v for k, v in payload.items() if k != "rules_digest"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(sut.LandscapeScoringError, match="half-open range"):
        sut.load_decision_rules(path)


def test_load_decision_rules_requires_foil_set_digests(tmp_path: Path) -> None:
    path = tmp_path / "landscape-decision-rules.json"
    payload = _decision_rules_payload()
    payload["foil_set_digests"] = {}
    payload["rules_digest"] = sut.sha256_json(
        {k: v for k, v in payload.items() if k != "rules_digest"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(sut.LandscapeScoringError, match="foil_set_digests"):
        sut.load_decision_rules(path)


# ---------------------------------------------------------------------------
# Candidate rows
# ---------------------------------------------------------------------------


def test_load_candidate_rows_accepts_valid_complete_box(tmp_path: Path) -> None:
    payload = _candidate_payload(candidate_id="cand-1")
    path = tmp_path / "candidates.jsonl"
    _write_jsonl(path, [payload])
    rows = sut.load_candidate_rows(path)
    assert rows[0].coord_token_ids == (11, 12, 13, 14)
    assert rows[0].request_kind == "complete_box"


def test_load_candidate_rows_accepts_valid_dense_scan(tmp_path: Path) -> None:
    payload = _candidate_payload(
        candidate_id="cand-2",
        request_kind="dense_scan",
        fixed_coord_token_ids=[11],
        scan_slot="y1",
    )
    path = tmp_path / "candidates.jsonl"
    _write_jsonl(path, [payload])
    rows = sut.load_candidate_rows(path)
    assert rows[0].fixed_coord_token_ids == (11,)
    assert rows[0].scan_slot == "y1"


def test_load_candidate_rows_rejects_retokenizable_text_fields(tmp_path: Path) -> None:
    payload = _candidate_payload(candidate_id="cand-3")
    payload["generated_text"] = "some decoded text"
    path = tmp_path / "candidates.jsonl"
    _write_jsonl(path, [payload])
    with pytest.raises(sut.LandscapeScoringError, match="re-tokenizable"):
        sut.load_candidate_rows(path)


def test_load_candidate_rows_rejects_prefix_hash_mismatch(tmp_path: Path) -> None:
    payload = _candidate_payload(candidate_id="cand-4")
    payload["prefix_token_ids_sha256"] = "0" * 64
    path = tmp_path / "candidates.jsonl"
    _write_jsonl(path, [payload])
    with pytest.raises(sut.LandscapeScoringError, match="exact prefix identity"):
        sut.load_candidate_rows(path)


def test_load_candidate_rows_rejects_context_missing_source_row(tmp_path: Path) -> None:
    payload = _candidate_payload(
        candidate_id="cand-5", context_id="P_pre", source_pred_row_id=None
    )
    path = tmp_path / "candidates.jsonl"
    _write_jsonl(path, [payload])
    with pytest.raises(sut.LandscapeScoringError, match="source_pred_row_id"):
        sut.load_candidate_rows(path)


def test_load_candidate_rows_rejects_scan_slot_not_following_fixed_coords(
    tmp_path: Path,
) -> None:
    payload = _candidate_payload(
        candidate_id="cand-6",
        request_kind="dense_scan",
        fixed_coord_token_ids=[11],
        scan_slot="x2",
    )
    path = tmp_path / "candidates.jsonl"
    _write_jsonl(path, [payload])
    with pytest.raises(sut.LandscapeScoringError, match="scan_slot"):
        sut.load_candidate_rows(path)


@pytest.mark.parametrize(
    "field_name,bad_value,message",
    [
        ("context_id", "not-a-context", "context_id"),
        ("role", "observer", "role"),
        ("review_status", "maybe", "review_status"),
        ("candidate_kind", "ghost", "candidate_kind"),
        ("native_repetition_penalty_stratum", 1.05, "stratum"),
    ],
)
def test_load_candidate_rows_rejects_unsupported_enum_values(
    tmp_path: Path, field_name: str, bad_value: Any, message: str
) -> None:
    payload = _candidate_payload(candidate_id="cand-7")
    payload[field_name] = bad_value
    path = tmp_path / "candidates.jsonl"
    _write_jsonl(path, [payload])
    with pytest.raises(sut.LandscapeScoringError, match=message):
        sut.load_candidate_rows(path)


# ---------------------------------------------------------------------------
# Owner / policy / description / foil-set gates
# ---------------------------------------------------------------------------


def test_resolve_owner_rejects_unresolved_and_missing(tmp_path: Path) -> None:
    ledger = {
        "unresolved:image-1:0": sut.OwnerLedgerEntry(
            diagnostic_owner_id="unresolved:image-1:0",
            status="unresolved",
            gt_owner_id=None,
            image_id="image-1",
        )
    }
    with pytest.raises(sut.LandscapeScoringError, match="unresolved"):
        sut.resolve_owner("unresolved:image-1:0", ledger)
    with pytest.raises(sut.LandscapeScoringError, match="unresolved"):
        sut.resolve_owner("gt:image-1:9", ledger)


def test_resolve_owner_accepts_gt_and_aux() -> None:
    ledger = {
        "gt:image-1:0": sut.OwnerLedgerEntry(
            diagnostic_owner_id="gt:image-1:0",
            status="gt",
            gt_owner_id="gt:image-1:0",
            image_id="image-1",
        ),
        "aux:image-1:0": sut.OwnerLedgerEntry(
            diagnostic_owner_id="aux:image-1:0",
            status="aux",
            gt_owner_id=None,
            image_id="image-1",
        ),
    }
    assert sut.resolve_owner("gt:image-1:0", ledger).status == "gt"
    assert sut.resolve_owner("aux:image-1:0", ledger).status == "aux"


def test_verify_no_policy_mixing_rejects_stratum_mismatch(tmp_path: Path) -> None:
    candidate = _candidate_row(
        _tmp_dir=tmp_path,
        candidate_id="cand-8",
        context_id="P_pre",
        source_pred_row_id="pred:1",
        native_repetition_penalty_stratum=1.0,
    )
    ledger = {
        "pred:1": sut.PredictionRowLedgerEntry(
            pred_row_id="pred:1", image_id="image-1", repetition_penalty=1.10
        )
    }
    with pytest.raises(sut.LandscapeScoringError, match="refuse to pool strata"):
        sut.verify_no_policy_mixing(candidate, ledger)


def test_verify_no_policy_mixing_rejects_missing_source_row(tmp_path: Path) -> None:
    candidate = _candidate_row(
        _tmp_dir=tmp_path,
        candidate_id="cand-9",
        context_id="P_pre",
        source_pred_row_id="pred:missing",
    )
    with pytest.raises(
        sut.LandscapeScoringError, match="absent from the sealed Task-1 ledger"
    ):
        sut.verify_no_policy_mixing(candidate, {})


def test_verify_no_policy_mixing_passes_for_sentinel_contexts_without_source_row(
    tmp_path: Path,
) -> None:
    candidate = _candidate_row(
        _tmp_dir=tmp_path,
        candidate_id="cand-10",
        context_id="root",
        source_pred_row_id=None,
    )
    sut.verify_no_policy_mixing(candidate, {})  # must not raise


def test_verify_coordinate_tokens_rejects_out_of_range(tmp_path: Path) -> None:
    rules = sut.load_decision_rules(_write_and_return_rules_path(tmp_path))
    candidate = _candidate_row(
        _tmp_dir=tmp_path,
        candidate_id="cand-11",
        coord_token_ids=[COORD_END, COORD_START, COORD_START, COORD_START],
    )
    with pytest.raises(
        sut.LandscapeScoringError, match="outside the declared coordinate vocabulary"
    ):
        sut.verify_coordinate_tokens(candidate, rules)


def test_verify_coordinate_tokens_accepts_in_range(tmp_path: Path) -> None:
    rules = sut.load_decision_rules(_write_and_return_rules_path(tmp_path))
    candidate = _candidate_row(
        _tmp_dir=tmp_path,
        candidate_id="cand-12",
        coord_token_ids=[
            COORD_START,
            COORD_START + 1,
            COORD_START + 2,
            COORD_START + 3,
        ],
    )
    sut.verify_coordinate_tokens(candidate, rules)  # must not raise


def test_verify_canonical_description_rejects_owner_missing_from_rules(
    tmp_path: Path,
) -> None:
    rules = sut.load_decision_rules(_write_and_return_rules_path(tmp_path))
    candidate = _candidate_row(
        _tmp_dir=tmp_path, candidate_id="cand-13", diagnostic_owner_id="gt:image-1:99"
    )
    with pytest.raises(
        sut.LandscapeScoringError, match="does not bind a canonical description"
    ):
        sut.verify_canonical_description(candidate, rules)


def test_resolve_foil_set_digest_rejects_unknown_foil_set(tmp_path: Path) -> None:
    rules = sut.load_decision_rules(_write_and_return_rules_path(tmp_path))
    candidate = _candidate_row(
        _tmp_dir=tmp_path, candidate_id="cand-14", foil_set_id="unregistered-set"
    )
    with pytest.raises(
        sut.LandscapeScoringError, match="unregistered or drifted foil set"
    ):
        sut.resolve_foil_set_digest(candidate, rules)


def test_resolve_foil_set_digest_returns_frozen_value(tmp_path: Path) -> None:
    rules = sut.load_decision_rules(_write_and_return_rules_path(tmp_path))
    candidate = _candidate_row(
        _tmp_dir=tmp_path, candidate_id="cand-15", foil_set_id="foils-v1"
    )
    assert sut.resolve_foil_set_digest(candidate, rules) == "deadbeef"


def test_verify_production_prompt_prefix_rejects_mismatch(tmp_path: Path) -> None:
    candidate = _candidate_row(
        _tmp_dir=tmp_path,
        candidate_id="cand-16",
        prefix_token_ids=[100, 101, 102],
        prompt_prefix_token_count=2,
    )
    with pytest.raises(
        sut.LandscapeScoringError, match="production prompt/image reconstruction"
    ):
        sut.verify_production_prompt_prefix(candidate, [100, 999])


def test_verify_production_prompt_prefix_accepts_match(tmp_path: Path) -> None:
    candidate = _candidate_row(
        _tmp_dir=tmp_path,
        candidate_id="cand-17",
        prefix_token_ids=[100, 101, 102],
        prompt_prefix_token_count=2,
    )
    sut.verify_production_prompt_prefix(candidate, [100, 101])  # must not raise


def _write_and_return_rules_path(tmp_path: Path) -> Path:
    path = tmp_path / "landscape-decision-rules.json"
    _write_decision_rules(path)
    return path


# ---------------------------------------------------------------------------
# Candidate proposal / pure-core adapter
# ---------------------------------------------------------------------------


def test_verify_candidate_proposal_rejects_self_hash_tamper(tmp_path: Path) -> None:
    candidate = _candidate_row(_tmp_dir=tmp_path, candidate_id="cand-18")
    tampered = replace(candidate, proposal_digest="0" * 64)
    with pytest.raises(
        sut.LandscapeScoringError,
        match="does not match its own declared defining parameters",
    ):
        sut.verify_candidate_proposal(tampered, pure_core=None)


def test_verify_candidate_proposal_reports_absent_core(tmp_path: Path) -> None:
    candidate = _candidate_row(_tmp_dir=tmp_path, candidate_id="cand-19")
    outcome = sut.verify_candidate_proposal(candidate, pure_core=None)
    assert outcome["self_consistency"] == "passed"
    assert outcome["pure_core_recomputation"] == "not_available"
    assert outcome["pure_core_present"] is False


def test_verify_candidate_proposal_never_invents_a_recomputation_call_even_with_a_real_shaped_core(
    tmp_path: Path,
) -> None:
    """The landed core v2 has no digest-recomputation API compatible with this scorer's schema;

    verify_candidate_proposal must stay self-consistency-only and say so, even when handed an
    object that looks like a plausible core (has unrelated attributes), rather than guessing at
    a call that was never verified against the real contract.
    """

    candidate = _candidate_row(_tmp_dir=tmp_path, candidate_id="cand-20")

    class _PlausibleLookingCore:
        def validate_rule_mapping(
            self, value: Any
        ) -> Any:  # pragma: no cover - never called
            raise AssertionError(
                "verify_candidate_proposal must not call core functions"
            )

    outcome = sut.verify_candidate_proposal(
        candidate, pure_core=_PlausibleLookingCore()
    )
    assert outcome["pure_core_recomputation"] == "not_available"
    assert outcome["pure_core_present"] is True
    assert "core_request_attestation" in outcome["note"]


def _write_v2_shaped_fake_core(path: Path, *, complete: bool = True) -> None:
    lines = [
        "from dataclasses import dataclass",
        "def validate_rule_mapping(value):",
        "    return value",
        "def full_vocabulary_id_digest(vocabulary_size):",
        "    return 'x' * 64",
    ]
    if complete:
        lines += [
            "@dataclass(frozen=True)",
            "class FullVocabularyAttestation:",
            "    expected_vocabulary_size: int",
            "    contiguous_token_id_digest: str",
            "    tokenizer_identity: str",
            "    model_identity: str",
            "    runtime_rule_digest: str",
            "@dataclass(frozen=True)",
            "class PolicyRuntimeIdentity:",
            "    tokenizer_identity: str",
            "    model_identity: str",
            "    runtime_rule_digest: str",
            "@dataclass(frozen=True, order=True)",
            "class RegisteredBasinRole:",
            "    role_id: str",
            "@dataclass(frozen=True, order=True)",
            "class CoordinateBin:",
            "    value: int",
            "@dataclass(frozen=True)",
            "class CoordinateBox:",
            "    x1: CoordinateBin",
            "    y1: CoordinateBin",
            "    x2: CoordinateBin",
            "    y2: CoordinateBin",
            "@dataclass(frozen=True)",
            "class ConditionalY1ScoreReceipt:",
            "    x1: CoordinateBin",
            "    y1: CoordinateBin",
            "    raw_selected_token_logprob: float",
            "    can_form_valid_box: bool",
            "    invalid_box_reason: str | None",
            "@dataclass(frozen=True)",
            "class ConditionalY1CompletenessAttestation:",
            "    completeness_digest: str",
            "def attest_complete_conditional_y1_scores(**kwargs):",
            "    return ConditionalY1CompletenessAttestation('x' * 64)",
            "COORDINATE_BIN_MIN = 0",
            "COORDINATE_BIN_MAX = 999",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_pure_core_returns_none_when_absent(tmp_path: Path) -> None:
    assert sut._load_pure_core(tmp_path / "does-not-exist.py") is None
    status = sut.pure_core_status(None, path=tmp_path / "does-not-exist.py")
    assert status["present"] is False


def test_load_pure_core_loads_present_module_with_the_real_v2_api(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fake_core_v2.py"
    _write_v2_shaped_fake_core(path, complete=True)
    core = sut._load_pure_core(path)
    assert core is not None
    status = sut.pure_core_status(core, path=path)
    assert status["present"] is True
    assert set(status["compatible_api"]) == set(sut.EXPECTED_PURE_CORE_API)
    assert status["note"] == "fully compatible"


def test_load_pure_core_registers_module_before_exec_for_order_true_dataclasses(
    tmp_path: Path,
) -> None:
    """Regression: a real landed core using ``@dataclass(order=True)`` previously crashed the

    whole CLI, because ``exec_module`` ran before the module was registered
    in ``sys.modules`` -- ``order=True`` introspects
    ``sys.modules[cls.__module__]`` during class creation and raised
    ``AttributeError: 'NoneType' object has no attribute '__dict__'``.
    """

    path = tmp_path / "fake_core_with_ordered_dataclass.py"
    path.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True, order=True)\n"
        "class Ordered:\n"
        "    value: int\n"
        "def recompute_candidate_proposal_digest(**kwargs):\n"
        "    return 'x'\n",
        encoding="utf-8",
    )
    core = sut._load_pure_core(path)
    assert core is not None
    assert core.Ordered(1) < core.Ordered(2)


def test_load_pure_core_returns_none_instead_of_propagating_when_exec_raises(
    tmp_path: Path,
) -> None:
    """A pure core that is syntactically valid Python but raises during import must be

    treated as absent/incompatible, never crash the runtime scorer.
    """

    path = tmp_path / "fake_core_that_raises.py"
    path.write_text(
        "raise RuntimeError('deliberately broken core')\n", encoding="utf-8"
    )
    assert sut._load_pure_core(path) is None
    import sys as _sys

    assert "_sorted_owner_basin_landscape_core" not in _sys.modules


# ---------------------------------------------------------------------------
# Production-required pure core v2 gate, geometry-domain cross-check,
# identity_kind role constraints, and core full-vocabulary attestation
# ---------------------------------------------------------------------------


def test_require_production_pure_core_fails_fast_on_absent_core(tmp_path: Path) -> None:
    with pytest.raises(sut.LandscapeScoringError, match="absent or incompatible core"):
        sut.require_production_pure_core(None, path=tmp_path / "absent.py")


def test_require_production_pure_core_fails_fast_on_incompatible_core(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incomplete_core.py"
    _write_v2_shaped_fake_core(path, complete=False)
    core = sut._load_pure_core(path)
    assert core is not None
    with pytest.raises(sut.LandscapeScoringError, match="absent or incompatible core"):
        sut.require_production_pure_core(core, path=path)


def test_require_production_pure_core_returns_the_core_when_fully_compatible(
    tmp_path: Path,
) -> None:
    path = tmp_path / "complete_core.py"
    _write_v2_shaped_fake_core(path, complete=True)
    core = sut._load_pure_core(path)
    assert sut.require_production_pure_core(core, path=path) is core


def test_verify_core_geometry_domain_accepts_matching_1000_bin_width(
    tmp_path: Path,
) -> None:
    path = tmp_path / "core.py"
    _write_v2_shaped_fake_core(path, complete=True)
    core = sut._load_pure_core(path)
    assert core is not None

    class _MatchingRules:
        schema_tokens = {
            "coordinate_token_id_start": 0,
            "coordinate_token_id_end_exclusive": 1000,
        }

    class _MismatchedRules:
        schema_tokens = {
            "coordinate_token_id_start": 10,
            "coordinate_token_id_end_exclusive": 30,
        }  # width 20, not 1000

    sut.verify_core_geometry_domain(core, _MatchingRules())  # must not raise
    with pytest.raises(
        sut.LandscapeScoringError, match="does not match the core's bins"
    ):
        sut.verify_core_geometry_domain(core, _MismatchedRules())


def test_verify_core_geometry_domain_rejects_core_missing_bin_constants() -> None:
    class _EmptyCore:
        pass

    class _Rules:
        schema_tokens = {
            "coordinate_token_id_start": 0,
            "coordinate_token_id_end_exclusive": 1000,
        }

    with pytest.raises(sut.LandscapeScoringError, match="COORDINATE_BIN_MIN"):
        sut.verify_core_geometry_domain(_EmptyCore(), _Rules())


@pytest.mark.parametrize(
    "candidate_kind,identity_kind,should_raise",
    [
        ("target_anchor", "reviewed_physical_owner", False),
        ("covered_owner", "reviewed_physical_owner", False),
        ("target_anchor", "registered_geometry", True),
        ("covered_owner", "registered_geometry", True),
        ("background", "registered_geometry", False),
        ("scan", "registered_geometry", False),
        ("background", "reviewed_physical_owner", False),
        ("part", "registered_geometry", True),
    ],
)
def test_verify_identity_kind_matches_role_constraints(
    tmp_path: Path, candidate_kind: str, identity_kind: str, should_raise: bool
) -> None:
    candidate = _candidate_row(
        _tmp_dir=tmp_path,
        candidate_id=f"cand-idk-{candidate_kind}-{identity_kind}",
        candidate_kind=candidate_kind,
        identity_kind=identity_kind,
    )
    if should_raise:
        with pytest.raises(sut.LandscapeScoringError):
            sut.verify_identity_kind_matches_role_constraints(candidate)
    else:
        sut.verify_identity_kind_matches_role_constraints(candidate)  # must not raise


def test_build_core_full_vocabulary_attestation_matches_the_real_core_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "core.py"
    _write_v2_shaped_fake_core(path, complete=True)
    core = sut._load_pure_core(path)
    assert core is not None
    attestation = sut.build_core_full_vocabulary_attestation(
        core,
        expected_vocab_size=64,
        tokenizer_identity_digest="tok-digest",
        model_identity_digest="model-digest",
        runtime_rule_digest="rule-digest",
    )
    assert attestation.expected_vocabulary_size == 64
    assert attestation.contiguous_token_id_digest == core.full_vocabulary_id_digest(64)
    assert attestation.tokenizer_identity == "tok-digest"


# ---------------------------------------------------------------------------
# Full-vocabulary attestation and coordinate-position scoring
# ---------------------------------------------------------------------------


def test_attest_full_vocabulary_rejects_mismatched_size() -> None:
    context = sut.build_attestation_context(
        expected_vocab_size=20,
        tokenizer_identity={},
        model_identity={},
        rule_digest="r",
        runtime_receipt_id="run",
    )
    with pytest.raises(
        sut.LandscapeScoringError, match="filtered or mismatched domain"
    ):
        sut.attest_full_vocabulary(10, attestation=context)


def test_attest_full_vocabulary_accepts_match() -> None:
    attestation = sut.attest_full_vocabulary(VOCAB_SIZE, attestation=TEST_ATTESTATION)
    assert attestation.vocab_size == VOCAB_SIZE
    assert attestation.filtered is False


def test_attest_full_vocabulary_binds_tokenizer_model_rule_and_runtime_identity() -> (
    None
):
    attestation = sut.attest_full_vocabulary(VOCAB_SIZE, attestation=TEST_ATTESTATION)
    assert attestation.tokenizer_identity_digest == sut.sha256_json(
        {"tokenizer": "fake-v1"}
    )
    assert attestation.model_identity_digest == sut.sha256_json({"model": "fake-v1"})
    assert attestation.rule_digest == "rule-digest-abc"
    assert attestation.runtime_receipt_id == "runtime-receipt-1"


def test_attest_full_vocabulary_digest_changes_when_identity_context_changes() -> None:
    other = sut.build_attestation_context(
        expected_vocab_size=VOCAB_SIZE,
        tokenizer_identity={"tokenizer": "different"},
        model_identity={"model": "fake-v1"},
        rule_digest="rule-digest-abc",
        runtime_receipt_id="runtime-receipt-1",
    )
    a = sut.attest_full_vocabulary(VOCAB_SIZE, attestation=TEST_ATTESTATION)
    b = sut.attest_full_vocabulary(VOCAB_SIZE, attestation=other)
    assert a.domain_digest != b.domain_digest


def test_score_coordinate_position_covers_every_declared_bin_exactly_once() -> None:
    token_id = COORD_START + 5
    logits = _spike_logits(VOCAB_SIZE, token_id)
    result = sut.score_coordinate_position(
        logits,
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        attestation=TEST_ATTESTATION,
        running_context_token_ids=[1, 2, 3],
        repetition_penalties=(1.0,),
    )
    bins = result["raw"]["bin_logprobs"]
    assert len(bins) == COORD_END - COORD_START
    assert bins.index(max(bins)) == token_id - COORD_START


def test_score_coordinate_position_rejects_subset_vocab() -> None:
    logits = torch.zeros(VOCAB_SIZE - 1, dtype=torch.float32)
    with pytest.raises(
        sut.LandscapeScoringError, match="filtered or mismatched domain"
    ):
        sut.score_coordinate_position(
            logits,
            coordinate_token_id_start=COORD_START,
            coordinate_token_id_end_exclusive=COORD_END,
            attestation=TEST_ATTESTATION,
            running_context_token_ids=[],
        )


def test_score_coordinate_position_raw_is_unchanged_by_which_rp_views_are_requested() -> (
    None
):
    token_id = COORD_START + 2
    logits = _spike_logits(VOCAB_SIZE, token_id)
    only_default = sut.score_coordinate_position(
        logits,
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        attestation=TEST_ATTESTATION,
        running_context_token_ids=[token_id],
        repetition_penalties=(1.0,),
    )
    both = sut.score_coordinate_position(
        logits,
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        attestation=TEST_ATTESTATION,
        running_context_token_ids=[token_id],
        repetition_penalties=(1.0, 1.10),
    )
    assert only_default["raw"] == both["raw"]


def test_score_coordinate_position_rp_1_00_equals_raw_exactly() -> None:
    token_id = COORD_START + 3
    logits = _spike_logits(VOCAB_SIZE, token_id)
    result = sut.score_coordinate_position(
        logits,
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        attestation=TEST_ATTESTATION,
        running_context_token_ids=[token_id],
        repetition_penalties=(1.0,),
    )
    assert result["raw"]["bin_logprobs"] == pytest.approx(
        result["auxiliary_policy"]["rp_1.00"]["bin_logprobs"]
    )


def test_score_coordinate_position_rp_1_10_penalizes_a_context_repeated_token() -> None:
    token_id = COORD_START + 4
    logits = _spike_logits(VOCAB_SIZE, token_id)
    result = sut.score_coordinate_position(
        logits,
        coordinate_token_id_start=token_id,
        coordinate_token_id_end_exclusive=token_id + 1,
        attestation=TEST_ATTESTATION,
        running_context_token_ids=[token_id],
        repetition_penalties=(1.0, 1.10),
    )
    raw_value = result["raw"]["bin_logprobs"][0]
    penalized_value = result["auxiliary_policy"]["rp_1.10"]["bin_logprobs"][0]
    assert penalized_value < raw_value


def test_score_coordinate_position_is_deterministic_across_calls() -> None:
    token_id = COORD_START + 1
    logits = _spike_logits(VOCAB_SIZE, token_id)
    first = sut.score_coordinate_position(
        logits,
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        attestation=TEST_ATTESTATION,
        running_context_token_ids=[token_id],
        repetition_penalties=(1.0, 1.10),
    )
    second = sut.score_coordinate_position(
        logits,
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        attestation=TEST_ATTESTATION,
        running_context_token_ids=[token_id],
        repetition_penalties=(1.0, 1.10),
    )
    assert first == second


# ---------------------------------------------------------------------------
# combine_complete_box: retains all four coordinate factors
# ---------------------------------------------------------------------------


def _single_view(token_id: int, *, context: list[int]) -> dict[str, Any]:
    logits = _spike_logits(VOCAB_SIZE, token_id)
    return sut._single_bin_view(
        logits,
        token_id=token_id,
        attestation=TEST_ATTESTATION,
        running_context_token_ids=context,
        repetition_penalties=(1.0, 1.10),
    )


def test_combine_complete_box_retains_all_four_coordinate_factors_and_sums_correctly() -> (
    None
):
    x1, y1, x2, y2 = COORD_START, COORD_START + 1, COORD_START + 2, COORD_START + 3
    views = {
        "x1": _single_view(x1, context=[]),
        "y1": _single_view(y1, context=[x1]),
        "x2": _single_view(x2, context=[x1, y1]),
        "y2": _single_view(y2, context=[x1, y1, x2]),
    }
    combined = sut.combine_complete_box(views)
    raw = combined["raw"]
    assert set(f"{n}_logprob" for n in sut.COORD_SLOTS) <= set(raw)
    expected_sum = sum(raw[f"{n}_logprob"] for n in sut.COORD_SLOTS)
    assert raw["complete_box_logprob_sum"] == pytest.approx(expected_sum)
    for penalty_key in ("rp_1.00", "rp_1.10"):
        policy = combined["auxiliary_policy"][penalty_key]
        assert set(f"{n}_logprob" for n in sut.COORD_SLOTS) <= set(policy)


def test_combine_complete_box_rejects_missing_slot() -> None:
    views = {"x1": _single_view(COORD_START, context=[])}
    with pytest.raises(ValueError, match="missing views"):
        sut.combine_complete_box(views)


# ---------------------------------------------------------------------------
# TF32 pinning
# ---------------------------------------------------------------------------


def test_pin_fp32_parity_flags_disables_both_flags() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    result = sut.pin_fp32_parity_flags()
    assert result == {"matmul_allow_tf32": False, "cudnn_allow_tf32": False}
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


# ---------------------------------------------------------------------------
# Explicit Qwen mrope continuation arithmetic
# ---------------------------------------------------------------------------


def test_continuation_position_ids_matches_grounded_qwen3_vl_formula() -> None:
    rope_deltas = torch.tensor([[5]], dtype=torch.long)
    positions = sut.continuation_position_ids(
        context_length_before_step=10, rope_deltas=rope_deltas, new_token_count=3
    )
    assert positions.shape == (3, 1, 3)
    expected = [15, 16, 17]
    for axis in range(3):
        assert positions[axis, 0].tolist() == expected


def test_continuation_position_ids_preserves_non_cpu_rope_device() -> None:
    rope_deltas = torch.empty((1, 1), dtype=torch.long, device="meta")
    positions = sut.continuation_position_ids(
        context_length_before_step=10,
        rope_deltas=rope_deltas,
        new_token_count=3,
    )
    assert positions.device == rope_deltas.device
    assert positions.shape == (3, 1, 3)


def test_continuation_position_ids_rejects_zero_new_tokens() -> None:
    rope_deltas = torch.tensor([[0]], dtype=torch.long)
    with pytest.raises(ValueError, match="at least one new token"):
        sut.continuation_position_ids(
            context_length_before_step=5, rope_deltas=rope_deltas, new_token_count=0
        )


# ---------------------------------------------------------------------------
# Prefix-KV-cache branch engine: order invariance and crop isolation
# ---------------------------------------------------------------------------


def _deterministic_logits_row(
    context: tuple[int, ...], vocab_size: int
) -> torch.Tensor:
    import zlib

    seed = zlib.crc32(json.dumps(list(context)).encode("utf-8")) & 0xFFFFFFFF
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(vocab_size, generator=generator)


class _FakeCacheBackend:
    """Deterministic function of accumulated token history; crop can be disabled to prove contamination."""

    def __init__(
        self,
        prefix_tokens: list[int],
        vocab_size: int,
        *,
        crop_is_buggy: bool = False,
        layer_count: int | None = None,
    ) -> None:
        self._tokens: list[int] = list(prefix_tokens)
        self._vocab_size = vocab_size
        self._crop_is_buggy = crop_is_buggy
        self._layer_count = layer_count

    @property
    def cache_length(self) -> int:
        return len(self._tokens)

    @property
    def layer_count(self) -> int | None:
        return self._layer_count

    def crop(self, length: int) -> None:
        if self._crop_is_buggy:
            return
        del self._tokens[length:]

    def step(self, token_ids: Sequence[int]) -> torch.Tensor:
        self._tokens.extend(int(v) for v in token_ids)
        start = len(self._tokens) - len(token_ids)
        rows = [
            _deterministic_logits_row(
                tuple(self._tokens[: position + 1]), self._vocab_size
            )
            for position in range(start, len(self._tokens))
        ]
        return torch.stack(rows)


def _full_reforward_backend(
    prefix_tokens: list[int], *, vocab_size: int = VOCAB_SIZE
) -> tuple[sut.FullReforwardBackend, list[tuple[int, ...]]]:
    calls: list[tuple[int, ...]] = []

    def full_reforward(token_ids: Sequence[int]) -> torch.Tensor:
        literal = tuple(int(value) for value in token_ids)
        calls.append(literal)
        return _deterministic_logits_row(literal, vocab_size)

    return (
        sut.FullReforwardBackend(
            root_prefix_token_ids=prefix_tokens,
            full_reforward=full_reforward,
            context_id="fixture-context",
            group_id="fixture-group",
            progress_every_actual_forwards=0,
        ),
        calls,
    )


def test_full_reforward_backend_uses_literal_prefix_and_strict_logical_crop() -> None:
    prefix = [11, 12]
    backend, calls = _full_reforward_backend(prefix)
    assert calls == [(11, 12)]
    assert torch.equal(
        backend.root_logits,
        _deterministic_logits_row((11, 12), VOCAB_SIZE),
    )
    rows = backend.step([21, 22])
    assert calls[-2:] == [(11, 12, 21), (11, 12, 21, 22)]
    assert rows.shape == (2, VOCAB_SIZE)
    backend.crop(len(prefix) + 1)
    assert backend.cache_length == len(prefix) + 1
    branch = backend.step([23])[-1]
    assert calls[-1] == (11, 12, 21, 23)
    assert torch.equal(
        branch,
        _deterministic_logits_row((11, 12, 21, 23), VOCAB_SIZE),
    )
    with pytest.raises(ValueError, match="logical suffix"):
        backend.crop(len(prefix) - 1)


def test_full_reforward_backend_memoizes_only_exact_depths_one_and_two() -> None:
    prefix = [31]
    backend, calls = _full_reforward_backend(prefix)
    backend.step([41, 42, 43])
    assert len(calls) == 4
    backend.crop(len(prefix))
    backend.step([41, 42])
    assert len(calls) == 4
    accounting = backend.accounting()
    assert accounting["memo_hits_by_depth"] == {"1": 1, "2": 1}
    assert accounting["memo_entries_by_depth"] == {"0": 1, "1": 1, "2": 1}
    backend.step([43])
    assert len(calls) == 5
    backend.crop(len(prefix) + 2)
    backend.step([43])
    assert len(calls) == 6
    assert backend.accounting()["memo_entries_by_depth"] == {
        "0": 1,
        "1": 1,
        "2": 1,
    }


def test_full_reforward_backend_branch_order_is_literal_and_invariant() -> None:
    prefix = [51, 52]
    backend, _calls = _full_reforward_backend(prefix)
    first = sut._branch_single_step_readout(backend, 61)  # noqa: SLF001
    sut._branch_single_step_readout(backend, 62)  # noqa: SLF001
    repeated = sut._branch_single_step_readout(backend, 61)  # noqa: SLF001
    assert torch.equal(first, repeated)
    assert backend.cache_length == len(prefix)


def test_full_reforward_backend_batches_literal_reservations_with_exact_accounting() -> None:
    prefix = [31]
    scalar_calls: list[tuple[int, ...]] = []
    batch_calls: list[tuple[tuple[int, ...], ...]] = []

    def scalar(token_ids: Sequence[int]) -> torch.Tensor:
        literal = tuple(int(value) for value in token_ids)
        scalar_calls.append(literal)
        return _deterministic_logits_row(literal, VOCAB_SIZE)

    def batched(token_rows: Sequence[Sequence[int]]) -> torch.Tensor:
        literals = tuple(tuple(int(value) for value in row) for row in token_rows)
        batch_calls.append(literals)
        return torch.stack(
            [_deterministic_logits_row(row, VOCAB_SIZE) for row in literals]
        )

    backend = sut.FullReforwardBackend(
        root_prefix_token_ids=prefix,
        full_reforward=scalar,
        batched_full_reforward=batched,
        full_reforward_batch_size=4,
        context_id="fixture-context",
        group_id="fixture-group",
        progress_every_actual_forwards=0,
    )
    backend.prefetch_suffixes(
        [
            [41],
            [42],
            [41],
            [41, 51],
            [42, 52],
            [41, 51, 61],
            [42, 52, 62],
            [41, 51, 61],
        ]
    )
    observed: list[torch.Tensor] = []
    for suffix in ([41, 51, 61], [42, 52, 62], [41, 51, 61]):
        with sut.BranchCursor(backend) as branch:
            observed.append(branch.step(suffix)[-1])

    assert scalar_calls == [(31,)]
    assert [len(batch) for batch in batch_calls] == [2, 2, 3]
    assert torch.equal(
        observed[0], _deterministic_logits_row((31, 41, 51, 61), VOCAB_SIZE)
    )
    assert torch.equal(observed[0], observed[2])
    accounting = backend.accounting()
    assert accounting["logical_token_step_requests_by_depth"] == {
        "1": 3,
        "2": 3,
        "3": 3,
    }
    assert accounting["actual_forward_calls_by_depth"] == {
        "0": 1,
        "1": 2,
        "2": 2,
        "3": 3,
    }
    assert accounting["memo_hits_by_depth"] == {"1": 1, "2": 1}
    assert accounting["physical_model_forward_calls"] == 4
    assert accounting["physical_forward_batch_histogram"] == {
        "1": 1,
        "4": 3,
    }
    assert accounting["requested_sequence_batch_histogram"] == {
        "1": 1,
        "2": 2,
        "3": 1,
    }
    assert accounting["padded_model_sequence_evaluations"] == 5


def test_batched_full_reforward_closure_pads_only_the_native_batch() -> None:
    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.observed_input_ids: torch.Tensor | None = None

        def get_rope_index(
            self,
            input_ids: torch.Tensor,
            image_grid_thw: torch.Tensor,
            _video_grid_thw: None,
            *,
            attention_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert image_grid_thw.shape == (4, 3)
            assert attention_mask.shape == input_ids.shape
            return (
                torch.zeros(
                    (3, *input_ids.shape), dtype=torch.long, device=input_ids.device
                ),
                torch.zeros((input_ids.shape[0], 1), device=input_ids.device),
            )

        def forward(self, *, input_ids: torch.Tensor, **kwargs: Any) -> Any:
            assert kwargs["use_cache"] is False
            assert kwargs["image_grid_thw"].shape == (4, 3)
            self.observed_input_ids = input_ids.detach().clone()
            rows = torch.stack(
                [
                    _deterministic_logits_row(tuple(row.tolist()), VOCAB_SIZE)
                    for row in input_ids
                ]
            )
            return SimpleNamespace(logits=rows[:, None, :])

    model = FakeModel()
    closure = sut._build_batched_full_reforward_closure(  # noqa: SLF001
        model,
        native_prompt_inputs={
            "input_ids": torch.ones((4, 2), dtype=torch.long),
            "attention_mask": torch.ones((4, 2), dtype=torch.long),
            "image_grid_thw": torch.ones((4, 3), dtype=torch.long),
            "pixel_values": torch.ones((4, 1)),
        },
        maximum_batch_size=4,
    )
    result = closure([[1, 2, 3], [4, 5, 6]])
    assert result.shape == (2, VOCAB_SIZE)
    assert model.observed_input_ids is not None
    assert model.observed_input_ids.tolist() == [
        [1, 2, 3],
        [4, 5, 6],
        [4, 5, 6],
        [4, 5, 6],
    ]


def test_full_reforward_batch_size_cli_requires_positive_integer() -> None:
    parser = sut.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--decision-rules",
                "rules.json",
                "--candidates",
                "candidates.jsonl",
                "--full-reforward-batch-size",
                "0",
            ]
        )
    parsed = parser.parse_args(
        [
            "--decision-rules",
            "rules.json",
            "--candidates",
            "candidates.jsonl",
            "--full-reforward-batch-size",
            "16",
        ]
    )
    assert parsed.full_reforward_batch_size == 16


def test_cache_admission_tolerance_cli_is_explicit_and_nonnegative() -> None:
    parser = sut.build_parser()
    base = [
        "--decision-rules",
        "rules.json",
        "--candidates",
        "candidates.jsonl",
    ]
    assert parser.parse_args(base).cache_admission_max_selected_logprob_diff is None
    parsed = parser.parse_args(
        [*base, "--cache-admission-max-selected-logprob-diff", "0.0002"]
    )
    assert parsed.cache_admission_max_selected_logprob_diff == pytest.approx(2e-4)
    for invalid in ("-0.1", "inf", "nan"):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [*base, "--cache-admission-max-selected-logprob-diff", invalid]
            )


def test_batched_reforward_parity_uses_coordinate_behavior_and_falls_back_on_drift() -> None:
    prefix = [1, 2]
    coords = [10, 11, 12, 13]

    def scalar(token_ids: Sequence[int]) -> torch.Tensor:
        return _deterministic_logits_row(tuple(token_ids), VOCAB_SIZE)

    def matching(token_rows: Sequence[Sequence[int]]) -> torch.Tensor:
        return torch.stack([scalar(row) for row in token_rows])

    passed = sut.run_batched_reforward_parity_gate(
        prefix_token_ids=prefix,
        coordinate_token_ids=coords,
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        full_reforward=scalar,
        batched_full_reforward=matching,
        requested_batch_size=16,
    )
    assert passed["status"] == "passed"
    assert passed["effective_batch_size"] == 16

    def drifted(token_rows: Sequence[Sequence[int]]) -> torch.Tensor:
        rows = matching(token_rows)
        rows[:, COORD_START] += 5.0
        return rows

    failed = sut.run_batched_reforward_parity_gate(
        prefix_token_ids=prefix,
        coordinate_token_ids=coords,
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        full_reforward=scalar,
        batched_full_reforward=drifted,
        requested_batch_size=16,
    )
    assert failed["status"] == "failed_scalar_fallback_required"
    assert failed["effective_batch_size"] == 1


def test_parity_selects_one_global_backend_and_fallback_admission_has_no_cache_rows() -> None:
    passed = {
        "status": "passed",
        "atol": sut.CACHE_PARITY_ATOL,
        "rtol": sut.CACHE_PARITY_RTOL,
    }
    assert sut.select_scoring_backend_from_parity(passed) == {
        "selected_backend": sut.KV_CACHE_SCORING_BACKEND,
        "cache_enabled": True,
        "use_cache": True,
        "fallback_trigger": None,
    }
    failed = {**passed, "status": "failed"}
    selection = sut.select_scoring_backend_from_parity(failed)
    backend, _calls = _full_reforward_backend([1, 2])
    backend.step([3, 4, 5])
    admission = sut.build_scoring_backend_admission(
        parity_gate=failed,
        selection=selection,
        score_row_count=7,
        per_context_group_accounting=[backend.accounting()],
    )
    assert admission["selected_backend"] == sut.FULL_REFORWARD_SCORING_BACKEND
    assert admission["cache_enabled"] is False
    assert admission["use_cache"] is False
    assert admission["fallback_trigger"] == sut.PARITY_FAILURE_FALLBACK_TRIGGER
    assert admission["cache_score_row_count"] == 0
    assert admission["uncached_score_row_count"] == 7
    assert admission["forward_accounting"]["aggregate"]["actual_forward_calls"] == 4


def test_mandatory_parity_probe_is_deterministic_and_missing_probe_fails_before_rows(
    tmp_path: Path,
) -> None:
    later = _candidate_row(_tmp_dir=tmp_path, candidate_id="probe-z")
    earlier = _candidate_row(_tmp_dir=tmp_path, candidate_id="probe-a")
    assert sut.select_mandatory_parity_probe([later, earlier]) == earlier
    dense = _candidate_row(
        _tmp_dir=tmp_path,
        candidate_id="dense-only",
        request_kind="dense_scan",
        fixed_coord_token_ids=[],
        scan_slot="x1",
    )
    with pytest.raises(
        sut.LandscapeScoringError, match="before any score row can be emitted"
    ):
        sut.select_mandatory_parity_probe([dense])


@pytest.mark.parametrize(
    ("scan_slot", "fixed"),
    [
        ("x1", []),
        ("y1", [COORD_START]),
        ("x2", [COORD_START, COORD_START + 1]),
        ("y2", [COORD_START, COORD_START + 1, COORD_START + 2]),
    ],
)
def test_full_reforward_matches_cache_dense_and_policy_payloads(
    scan_slot: str, fixed: list[int]
) -> None:
    prefix = [7, 8, 9]
    cache = _FakeCacheBackend(prefix, vocab_size=VOCAB_SIZE)
    uncached, _calls = _full_reforward_backend(prefix)
    cache_payload = sut.score_dense_scan_candidate(
        backend=cache,
        prefill_logits=_deterministic_logits_row(tuple(prefix), VOCAB_SIZE),
        fixed_coord_token_ids=fixed,
        scan_slot=scan_slot,
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        attestation=TEST_ATTESTATION,
        running_context_token_ids=prefix,
    )
    uncached_payload = sut.score_dense_scan_candidate(
        backend=uncached,
        prefill_logits=uncached.root_logits,
        fixed_coord_token_ids=fixed,
        scan_slot=scan_slot,
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        attestation=TEST_ATTESTATION,
        running_context_token_ids=prefix,
    )
    assert uncached_payload == cache_payload
    if scan_slot == "y1":
        def _conditional_row(payload: Mapping[str, Any]) -> dict[str, Any]:
            return {
                "request_kind": "dense_scan",
                "scan_slot": "y1",
                "context_id": "fixture-context",
                "diagnostic_owner_id": "gt:image-1:0",
                "rule_digest": "fixture-rule",
                "fixed_coord_token_ids": fixed,
                "raw_bin_scan": payload["raw"],
                "candidate_id": "fixture-y1",
            }

        cache_digest = sut.compute_conditional_y1_completeness_digest(
            context_id="fixture-context",
            diagnostic_owner_id="gt:image-1:0",
            rule_digest="fixture-rule",
            admitted_x1_token_ids=fixed,
            y1_bin_count=COORD_END - COORD_START,
            dense_scan_rows=[_conditional_row(cache_payload)],
        )
        uncached_digest = sut.compute_conditional_y1_completeness_digest(
            context_id="fixture-context",
            diagnostic_owner_id="gt:image-1:0",
            rule_digest="fixture-rule",
            admitted_x1_token_ids=fixed,
            y1_bin_count=COORD_END - COORD_START,
            dense_scan_rows=[_conditional_row(uncached_payload)],
        )
        assert uncached_digest == cache_digest


def test_full_reforward_matches_cache_complete_box_and_policy_payloads() -> None:
    prefix = [17, 18]
    coords = (COORD_START, COORD_START + 1, COORD_START + 2, COORD_START + 3)
    cache = _FakeCacheBackend(prefix, vocab_size=VOCAB_SIZE)
    uncached, _calls = _full_reforward_backend(prefix)
    kwargs = {
        "coord_token_ids": coords,
        "attestation": TEST_ATTESTATION,
        "running_context_token_ids": prefix,
    }
    cache_payload = sut.score_complete_box_candidate(
        backend=cache,
        prefill_logits=_deterministic_logits_row(tuple(prefix), VOCAB_SIZE),
        **kwargs,
    )
    uncached_payload = sut.score_complete_box_candidate(
        backend=uncached,
        prefill_logits=uncached.root_logits,
        **kwargs,
    )
    assert uncached_payload == cache_payload


def _explore_three_branches(
    backend: _FakeCacheBackend, order: list[int]
) -> dict[int, float]:
    results: dict[int, float] = {}
    for token in order:
        with sut.BranchCursor(backend) as branch:
            logits = branch.step([token])[-1]
            results[token] = float(logits[0].item())
    return results


def test_branch_cursor_crop_makes_exploration_order_invariant() -> None:
    forward_backend = _FakeCacheBackend([1, 2, 3], vocab_size=16)
    reverse_backend = _FakeCacheBackend([1, 2, 3], vocab_size=16)
    forward_result = _explore_three_branches(forward_backend, [10, 20, 30])
    reverse_result = _explore_three_branches(reverse_backend, [30, 20, 10])
    assert forward_result == reverse_result
    assert forward_backend.cache_length == 3
    assert reverse_backend.cache_length == 3


def test_missing_crop_would_be_caught_by_the_order_invariance_check() -> None:
    """Canary: a backend that fails to crop produces order-dependent results.

    This proves the order-invariance assertion above has teeth -- it is not
    vacuously true for any backend.
    """

    buggy_forward = _FakeCacheBackend([1, 2, 3], vocab_size=16, crop_is_buggy=True)
    buggy_reverse = _FakeCacheBackend([1, 2, 3], vocab_size=16, crop_is_buggy=True)
    forward_result = _explore_three_branches(buggy_forward, [10, 20, 30])
    reverse_result = _explore_three_branches(buggy_reverse, [30, 20, 10])
    assert forward_result != reverse_result


def test_nested_branch_cursors_crop_back_to_each_entry_length() -> None:
    backend = _FakeCacheBackend([1, 2, 3], vocab_size=16)
    with sut.BranchCursor(backend) as outer:
        outer.step([100])
        assert backend.cache_length == 4
        with sut.BranchCursor(backend) as inner:
            inner.step([200])
            inner.step([300])
            assert backend.cache_length == 6
        assert backend.cache_length == 4  # inner exit cropped back to its entry point
    assert backend.cache_length == 3  # outer exit cropped back to the original prefix


# ---------------------------------------------------------------------------
# score_complete_box_candidate / score_dense_scan_candidate over the fake backend
# ---------------------------------------------------------------------------


class _ScriptedCacheBackend:
    """Replays a fixed sequence of per-step logits and records step() calls."""

    def __init__(self, scripted_logits: list[torch.Tensor]) -> None:
        self._scripted = scripted_logits
        self._calls: list[list[int]] = []
        self._length = 0

    @property
    def cache_length(self) -> int:
        return self._length

    @property
    def layer_count(self) -> int | None:
        return None

    def crop(self, length: int) -> None:
        self._length = length

    def step(self, token_ids: Sequence[int]) -> torch.Tensor:
        self._calls.append(list(token_ids))
        self._length += len(token_ids)
        return self._scripted[len(self._calls) - 1].unsqueeze(0)


def test_score_complete_box_candidate_uses_three_branch_steps_and_crops_back() -> None:
    x1, y1, x2, y2 = COORD_START, COORD_START + 1, COORD_START + 2, COORD_START + 3
    prefill_logits = _spike_logits(VOCAB_SIZE, x1)
    backend = _ScriptedCacheBackend(
        [
            _spike_logits(VOCAB_SIZE, y1),
            _spike_logits(VOCAB_SIZE, x2),
            _spike_logits(VOCAB_SIZE, y2),
        ]
    )
    result = sut.score_complete_box_candidate(
        backend=backend,
        prefill_logits=prefill_logits,
        coord_token_ids=(x1, y1, x2, y2),
        attestation=TEST_ATTESTATION,
        running_context_token_ids=[1, 2, 3],
    )
    assert len(backend._calls) == 3
    assert backend._calls == [[x1], [y1], [x2]]
    assert backend.cache_length == 0  # branch cropped back to entry (0) on exit
    raw = result["raw"]
    for name, token in zip(sut.COORD_SLOTS, (x1, y1, x2, y2)):
        assert raw[f"{name}_logprob"] == pytest.approx(
            0.0, abs=0.05
        )  # spike token dominates its 64-way softmax


def test_score_dense_scan_candidate_scans_full_vocabulary_at_admitted_x1() -> None:
    x1 = COORD_START
    prefill_logits = _spike_logits(VOCAB_SIZE, x1)
    y1_target = COORD_START + 7
    backend = _ScriptedCacheBackend([_spike_logits(VOCAB_SIZE, y1_target)])
    result = sut.score_dense_scan_candidate(
        backend=backend,
        prefill_logits=prefill_logits,
        fixed_coord_token_ids=[x1],
        scan_slot="y1",
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        attestation=TEST_ATTESTATION,
        running_context_token_ids=[1, 2, 3],
    )
    assert (
        len(backend._calls) == 1
    )  # exactly one branch step: append x1, read the full y1 vocab for free
    bins = result["raw"]["bin_logprobs"]
    assert len(bins) == COORD_END - COORD_START
    assert bins.index(max(bins)) == y1_target - COORD_START
    assert backend.cache_length == 0


def test_score_dense_scan_candidate_scans_x1_directly_from_prefill_with_zero_steps() -> (
    None
):
    x1_target = COORD_START + 2
    prefill_logits = _spike_logits(VOCAB_SIZE, x1_target)
    backend = _ScriptedCacheBackend([])
    result = sut.score_dense_scan_candidate(
        backend=backend,
        prefill_logits=prefill_logits,
        fixed_coord_token_ids=[],
        scan_slot="x1",
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        attestation=TEST_ATTESTATION,
        running_context_token_ids=[1, 2, 3],
    )
    assert len(backend._calls) == 0
    bins = result["raw"]["bin_logprobs"]
    assert bins.index(max(bins)) == x1_target - COORD_START


# ---------------------------------------------------------------------------
# build_landscape_score_row: full gate + scoring integration
# ---------------------------------------------------------------------------


class _RaisingBackend:
    """Fails the test if .step is ever called -- proves fail-fast gates run before any forward."""

    @property
    def cache_length(self) -> int:
        return 0

    @property
    def layer_count(self) -> int | None:
        return None

    def crop(self, length: int) -> None:
        return

    def step(
        self, token_ids: Sequence[int]
    ) -> torch.Tensor:  # pragma: no cover - must never run
        raise AssertionError(
            "backend.step must not be called when an earlier gate should have failed fast"
        )


def test_build_landscape_score_row_fails_fast_on_unresolved_owner_before_any_forward(
    tmp_path: Path,
) -> None:
    rules = sut.load_decision_rules(_write_and_return_rules_path(tmp_path))
    owner_ledger = {
        "unresolved:image-1:0": sut.OwnerLedgerEntry(
            diagnostic_owner_id="unresolved:image-1:0",
            status="unresolved",
            gt_owner_id=None,
            image_id="image-1",
        )
    }
    candidate = _candidate_row(
        _tmp_dir=tmp_path,
        candidate_id="cand-22",
        diagnostic_owner_id="unresolved:image-1:0",
    )
    with pytest.raises(sut.LandscapeScoringError, match="unresolved"):
        sut.build_landscape_score_row(
            candidate=candidate,
            owner_ledger=owner_ledger,
            prediction_row_ledger={},
            rules=rules,
            pure_core=None,
            reconstructed_prompt_token_ids=[100, 101],
            backend=_RaisingBackend(),
            prefill_logits=torch.zeros(VOCAB_SIZE),
            tokenizer_identity={"tokenizer": "fake-v1"},
            model_identity={"model": "fake-v1"},
            runtime_receipt_id="runtime-receipt-1",
        )


def test_build_landscape_score_row_emits_frozen_complete_box_schema_with_separated_channels(
    tmp_path: Path,
) -> None:
    rules = sut.load_decision_rules(_write_and_return_rules_path(tmp_path))
    owner_ledger = {
        "gt:image-1:0": sut.OwnerLedgerEntry(
            diagnostic_owner_id="gt:image-1:0",
            status="gt",
            gt_owner_id="gt:image-1:0",
            image_id="image-1",
        )
    }
    x1, y1, x2, y2 = COORD_START, COORD_START + 1, COORD_START + 2, COORD_START + 3
    candidate = _candidate_row(
        _tmp_dir=tmp_path,
        candidate_id="cand-23",
        coord_token_ids=[x1, y1, x2, y2],
        prefix_token_ids=[1, 2],
        prompt_prefix_token_count=2,
    )
    backend = _ScriptedCacheBackend(
        [
            _spike_logits(VOCAB_SIZE, y1),
            _spike_logits(VOCAB_SIZE, x2),
            _spike_logits(VOCAB_SIZE, y2),
        ]
    )
    row = sut.build_landscape_score_row(
        candidate=candidate,
        owner_ledger=owner_ledger,
        prediction_row_ledger={},
        rules=rules,
        pure_core=None,
        reconstructed_prompt_token_ids=[1, 2],
        backend=backend,
        prefill_logits=_spike_logits(VOCAB_SIZE, x1),
        tokenizer_identity={"tokenizer": "fake-v1"},
        model_identity={"model": "fake-v1"},
        runtime_receipt_id="runtime-receipt-1",
    )
    assert tuple(row) == sut.COMPLETE_BOX_ROW_FIELDS
    assert row["gt_owner_id"] == "gt:image-1:0"
    assert row["foil_set_digest"] == "deadbeef"
    assert row["rule_digest"] == rules.rules_digest
    assert (
        set(row["raw_model_logprob"]) & set(row["auxiliary_policy_scores"]) == set()
    )  # never mixed as sibling keys


def test_build_landscape_score_row_emits_frozen_dense_scan_schema(
    tmp_path: Path,
) -> None:
    rules = sut.load_decision_rules(_write_and_return_rules_path(tmp_path))
    owner_ledger = {
        "gt:image-1:0": sut.OwnerLedgerEntry(
            diagnostic_owner_id="gt:image-1:0",
            status="gt",
            gt_owner_id="gt:image-1:0",
            image_id="image-1",
        )
    }
    x1 = COORD_START
    candidate = _candidate_row(
        _tmp_dir=tmp_path,
        candidate_id="cand-24",
        request_kind="dense_scan",
        fixed_coord_token_ids=[x1],
        scan_slot="y1",
        prefix_token_ids=[1, 2],
        prompt_prefix_token_count=2,
    )
    backend = _ScriptedCacheBackend([_spike_logits(VOCAB_SIZE, COORD_START + 6)])
    row = sut.build_landscape_score_row(
        candidate=candidate,
        owner_ledger=owner_ledger,
        prediction_row_ledger={},
        rules=rules,
        pure_core=None,
        reconstructed_prompt_token_ids=[1, 2],
        backend=backend,
        prefill_logits=_spike_logits(VOCAB_SIZE, x1),
        tokenizer_identity={"tokenizer": "fake-v1"},
        model_identity={"model": "fake-v1"},
        runtime_receipt_id="runtime-receipt-1",
    )
    assert tuple(row) == sut.DENSE_SCAN_ROW_FIELDS
    assert len(row["raw_bin_scan"]["bin_logprobs"]) == COORD_END - COORD_START


# ---------------------------------------------------------------------------
# Execution receipt
# ---------------------------------------------------------------------------


def test_build_execution_receipt_reports_source_and_rule_digests_identity_environment_command_tolerance(
    tmp_path: Path,
) -> None:
    rules_path = _write_and_return_rules_path(tmp_path)
    rules = sut.load_decision_rules(rules_path)
    owner_path = tmp_path / "owner-ledger.jsonl"
    _write_jsonl(
        owner_path, [_owner_ledger_row(diagnostic_owner_id="gt:image-1:0", status="gt")]
    )
    receipt = sut.build_execution_receipt(
        command=["score.py", "--foo", "bar"],
        source_files={"owner_ledger": owner_path},
        manifest_digests=None,
        rules=rules,
        rules_path=rules_path,
        backend_receipt={
            "model_identity": {"base": "m"},
            "tokenizer_identity": {"vocab": 1},
        },
        candidate_count=3,
        owners_covered=2,
        pure_core_status_dict={"present": False},
        environment={"tf32": {"matmul_allow_tf32": False, "cudnn_allow_tf32": False}},
    )
    assert receipt["source_digests"]["owner_ledger"]["match"] is True
    assert receipt["model_identity"] == {"base": "m"}
    assert receipt["numeric_reproduction_tolerance"] == pytest.approx(1e-6)
    assert receipt["command"] == ["score.py", "--foo", "bar"]
    assert (
        receipt["execution_architecture"]["one_full_reforward_per_candidate"] is False
    )
    assert receipt["environment"]["tf32"]["matmul_allow_tf32"] is False


def test_build_execution_receipt_rejects_manifest_digest_mismatch(
    tmp_path: Path,
) -> None:
    rules_path = _write_and_return_rules_path(tmp_path)
    rules = sut.load_decision_rules(rules_path)
    owner_path = tmp_path / "owner-ledger.jsonl"
    _write_jsonl(
        owner_path, [_owner_ledger_row(diagnostic_owner_id="gt:image-1:0", status="gt")]
    )
    with pytest.raises(sut.LandscapeScoringError, match="stale or substituted source"):
        sut.build_execution_receipt(
            command=[],
            source_files={"owner_ledger": owner_path},
            manifest_digests={"owner_ledger": "0" * 64},
            rules=rules,
            rules_path=rules_path,
            backend_receipt={},
            candidate_count=0,
            owners_covered=0,
            pure_core_status_dict={},
            environment={},
        )


# ---------------------------------------------------------------------------
# Conditional-y1 completeness digest
# ---------------------------------------------------------------------------


def _y1_dense_scan_row(
    *,
    x1: int,
    context_id: str = "root",
    diagnostic_owner_id: str = "gt:image-1:0",
    rule_digest: str = "rule-digest-abc",
    bin_count: int = 20,
) -> dict[str, Any]:
    return {
        "request_kind": "dense_scan",
        "scan_slot": "y1",
        "context_id": context_id,
        "diagnostic_owner_id": diagnostic_owner_id,
        "rule_digest": rule_digest,
        "fixed_coord_token_ids": [x1],
        "raw_bin_scan": {"bin_logprobs": [0.0] * bin_count},
        "candidate_id": f"scan-x1-{x1}",
    }


def test_conditional_y1_completeness_digest_accepts_exact_coverage() -> None:
    rows = [
        _y1_dense_scan_row(x1=10),
        _y1_dense_scan_row(x1=11),
        _y1_dense_scan_row(x1=12),
    ]
    result = sut.compute_conditional_y1_completeness_digest(
        context_id="root",
        diagnostic_owner_id="gt:image-1:0",
        rule_digest="rule-digest-abc",
        admitted_x1_token_ids=[10, 11, 12],
        y1_bin_count=20,
        dense_scan_rows=rows,
    )
    assert result["status"] == "complete"
    assert result["x1_count"] == 3
    assert result["admitted_x1_token_ids"] == [10, 11, 12]


def test_conditional_y1_completeness_digest_is_stable_and_sensitive() -> None:
    rows = [_y1_dense_scan_row(x1=10), _y1_dense_scan_row(x1=11)]
    same_again = [_y1_dense_scan_row(x1=10), _y1_dense_scan_row(x1=11)]
    a = sut.compute_conditional_y1_completeness_digest(
        context_id="root",
        diagnostic_owner_id="gt:image-1:0",
        rule_digest="rule-digest-abc",
        admitted_x1_token_ids=[10, 11],
        y1_bin_count=20,
        dense_scan_rows=rows,
    )
    b = sut.compute_conditional_y1_completeness_digest(
        context_id="root",
        diagnostic_owner_id="gt:image-1:0",
        rule_digest="rule-digest-abc",
        admitted_x1_token_ids=[10, 11],
        y1_bin_count=20,
        dense_scan_rows=same_again,
    )
    assert a["completeness_digest"] == b["completeness_digest"]
    c = sut.compute_conditional_y1_completeness_digest(
        context_id="root",
        diagnostic_owner_id="gt:image-1:0",
        rule_digest="rule-digest-abc",
        admitted_x1_token_ids=[10, 11, 12],
        y1_bin_count=20,
        dense_scan_rows=[*rows, _y1_dense_scan_row(x1=12)],
    )
    assert c["completeness_digest"] != a["completeness_digest"]


def test_conditional_y1_completeness_digest_rejects_missing_x1_anchor() -> None:
    rows = [_y1_dense_scan_row(x1=10)]
    with pytest.raises(
        sut.LandscapeScoringError, match="do not exactly cover every declared x1 anchor"
    ):
        sut.compute_conditional_y1_completeness_digest(
            context_id="root",
            diagnostic_owner_id="gt:image-1:0",
            rule_digest="rule-digest-abc",
            admitted_x1_token_ids=[10, 11],
            y1_bin_count=20,
            dense_scan_rows=rows,
        )


def test_conditional_y1_completeness_digest_rejects_unexpected_x1_anchor() -> None:
    rows = [_y1_dense_scan_row(x1=10), _y1_dense_scan_row(x1=99)]
    with pytest.raises(
        sut.LandscapeScoringError, match="do not exactly cover every declared x1 anchor"
    ):
        sut.compute_conditional_y1_completeness_digest(
            context_id="root",
            diagnostic_owner_id="gt:image-1:0",
            rule_digest="rule-digest-abc",
            admitted_x1_token_ids=[10],
            y1_bin_count=20,
            dense_scan_rows=rows,
        )


def test_conditional_y1_completeness_digest_rejects_incomplete_bin_scan() -> None:
    rows = [_y1_dense_scan_row(x1=10, bin_count=19)]
    with pytest.raises(
        sut.LandscapeScoringError,
        match="does not cover exactly the declared y1 bin count",
    ):
        sut.compute_conditional_y1_completeness_digest(
            context_id="root",
            diagnostic_owner_id="gt:image-1:0",
            rule_digest="rule-digest-abc",
            admitted_x1_token_ids=[10],
            y1_bin_count=20,
            dense_scan_rows=rows,
        )


def test_conditional_y1_completeness_digest_rejects_foreign_context_or_rule_digest() -> (
    None
):
    rows = [_y1_dense_scan_row(x1=10, context_id="P_pre")]
    with pytest.raises(sut.LandscapeScoringError, match="different context/owner"):
        sut.compute_conditional_y1_completeness_digest(
            context_id="root",
            diagnostic_owner_id="gt:image-1:0",
            rule_digest="rule-digest-abc",
            admitted_x1_token_ids=[10],
            y1_bin_count=20,
            dense_scan_rows=rows,
        )
    rows_bad_rule = [_y1_dense_scan_row(x1=10, rule_digest="stale-digest")]
    with pytest.raises(sut.LandscapeScoringError, match="different rule_digest"):
        sut.compute_conditional_y1_completeness_digest(
            context_id="root",
            diagnostic_owner_id="gt:image-1:0",
            rule_digest="rule-digest-abc",
            admitted_x1_token_ids=[10],
            y1_bin_count=20,
            dense_scan_rows=rows_bad_rule,
        )


# ---------------------------------------------------------------------------
# Mandatory cache-vs-reforward parity gate (audit-mandated)
# ---------------------------------------------------------------------------


def test_run_cache_parity_gate_passes_when_cache_path_matches_the_reforward_oracle() -> (
    None
):
    prefix = [1, 2, 3]
    x1, y1, x2, y2 = 10, 11, 12, 13
    vocab = VOCAB_SIZE

    def oracle(tokens: list[int]) -> torch.Tensor:
        return _deterministic_logits_row(tuple(tokens), vocab)

    backend = _FakeCacheBackend(list(prefix), vocab_size=vocab)
    prefill_logits = oracle(prefix)
    result = sut.run_cache_parity_gate(
        backend=backend,
        prefill_logits=prefill_logits,
        prefix_token_ids=prefix,
        x1_token_id=x1,
        y1_token_id=y1,
        x2_token_id=x2,
        y2_token_id=y2,
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        full_reforward=oracle,
    )
    assert result["status"] == "passed"
    assert result["strict_full_vocabulary_parity_status"] == "passed"
    assert result["admission_policy"]["requested_mode"] == (
        sut.STRICT_CACHE_ADMISSION_MODE
    )
    assert result["admission_policy"]["effective_mode"] == (
        sut.STRICT_CACHE_ADMISSION_MODE
    )
    assert result["admission_policy"]["requested_thresholds"][
        "selected_coordinate_logprob_max_abs_diff"
    ] is None
    assert all(step["within_tolerance"] for step in result["raw_logit_parity_steps"])
    assert [step["step"] for step in result["raw_logit_parity_steps"]] == [
        "root_predicts_x1",
        "post_x1_predicts_y1",
        "post_y1_predicts_x2",
        "post_x2_predicts_y2",
    ]
    assert result["atol"] == sut.CACHE_PARITY_ATOL
    assert result["rtol"] == sut.CACHE_PARITY_RTOL
    assert result["branch_reverse_order_invariance"]["cache_length_restored"] is True
    assert backend.cache_length == len(prefix)


def test_run_cache_parity_gate_fails_when_cache_path_disagrees_with_the_reforward_oracle() -> (
    None
):
    prefix = [1, 2, 3]
    x1, y1, x2, y2 = 10, 11, 12, 13
    vocab = VOCAB_SIZE

    def drifting_oracle(tokens: list[int]) -> torch.Tensor:
        base = _deterministic_logits_row(tuple(tokens), vocab)
        if len(tokens) == len(prefix) + 1:  # perturb exactly the y1-reforward step
            return base + 5.0
        return base

    backend = _FakeCacheBackend(list(prefix), vocab_size=vocab)
    prefill_logits = drifting_oracle(prefix)
    result = sut.run_cache_parity_gate(
        backend=backend,
        prefill_logits=prefill_logits,
        prefix_token_ids=prefix,
        x1_token_id=x1,
        y1_token_id=y1,
        x2_token_id=x2,
        y2_token_id=y2,
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        full_reforward=drifting_oracle,
    )
    assert result["status"] == "failed"
    y1_step = next(
        step
        for step in result["raw_logit_parity_steps"]
        if step["step"] == "post_x1_predicts_y1"
    )
    assert y1_step["within_tolerance"] is False
    assert y1_step["depth"] == "post_x1" and y1_step["predicted_slot"] == "y1"
    # the unperturbed post-x2 depth (added by this fix) still independently passes
    y2_step = next(
        step
        for step in result["raw_logit_parity_steps"]
        if step["step"] == "post_x2_predicts_y2"
    )
    assert y2_step["within_tolerance"] is True


def test_relaxed_cache_admission_requires_all_coordinate_behavior_checks_and_records_receipts() -> (
    None
):
    prefix = [1, 2, 3]
    x1, y1, x2, y2 = 10, 11, 12, 13

    def shifted_oracle(tokens: list[int]) -> torch.Tensor:
        values = _deterministic_logits_row(tuple(tokens), VOCAB_SIZE)
        if len(tokens) == len(prefix) + 1:
            # Perturb a low-probability, non-coordinate, non-context token.
            # This breaks strict all-vocabulary parity without changing the
            # coordinate decision or its selected-token probability materially.
            values = values.clone()
            tail_start = COORD_END
            perturbed = tail_start + int(torch.argmin(values[tail_start:]).item())
            values[perturbed] += 0.01
        return values

    backend = _FakeCacheBackend(list(prefix), vocab_size=VOCAB_SIZE)
    result = sut.run_cache_parity_gate(
        backend=backend,
        prefill_logits=shifted_oracle(prefix),
        prefix_token_ids=prefix,
        x1_token_id=x1,
        y1_token_id=y1,
        x2_token_id=x2,
        y2_token_id=y2,
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        full_reforward=shifted_oracle,
        relaxed_selected_logprob_max_abs_diff=1e-4,
    )

    policy = result["admission_policy"]
    assert result["status"] == "passed"
    assert result["strict_full_vocabulary_parity_status"] == "failed"
    assert policy["requested_mode"] == sut.RELAXED_CACHE_ADMISSION_MODE
    assert policy["effective_mode"] == sut.RELAXED_CACHE_ADMISSION_MODE
    assert policy["strict_full_vocabulary_parity_passed"] is False
    assert policy["all_coordinate_argmax_parity"] is True
    assert policy["requested_thresholds"][
        "selected_coordinate_logprob_max_abs_diff"
    ] == pytest.approx(1e-4)
    assert policy["effective_thresholds"][
        "selected_coordinate_logprob_max_abs_diff"
    ] == pytest.approx(1e-4)
    assert policy["observed_max_raw_logit_abs_diff"] == pytest.approx(0.01)
    assert policy["observed_max_selected_coordinate_logprob_abs_diff"] <= 1e-4
    checks = result["coordinate_behavior_checks"]
    assert {(entry["predicted_slot"], entry["view"]) for entry in checks} == {
        (slot, view)
        for slot in sut.COORD_SLOTS
        for view in ("raw", "rp_1.00", "rp_1.10")
    }
    assert all(entry["coordinate_argmax_parity"] for entry in checks)

    result["admission_scope"] = {"group_count": 1, "scope_sha256": "fixture"}
    selection = sut.select_scoring_backend_from_parity(result)
    accounted = sut.AccountingCacheBackend(
        _FakeCacheBackend(list(prefix), vocab_size=VOCAB_SIZE),
        context_id="root",
        group_id="group-1",
    )
    admission = sut.build_scoring_backend_admission(
        parity_gate=result,
        selection=selection,
        score_row_count=3,
        per_context_group_accounting=[accounted.accounting()],
    )
    assert admission["selected_backend"] == sut.KV_CACHE_SCORING_BACKEND
    assert admission["decision_use"] == sut.PROBE_ONLY_SCORE_USE
    assert admission["cache_admission_policy"]["effective_mode"] == (
        sut.RELAXED_CACHE_ADMISSION_MODE
    )
    assert admission["cache_admission_scope"]["group_count"] == 1
    architecture = sut.execution_architecture_for_admission(admission)
    assert architecture["cache_admission_mode"] == sut.RELAXED_CACHE_ADMISSION_MODE
    assert "strict full-vocabulary parity failed" in architecture["strategy"]
    forged_implicit_relaxation = {
        **result,
        "admission_policy": {
            **policy,
            "requested_mode": sut.STRICT_CACHE_ADMISSION_MODE,
        },
    }
    with pytest.raises(sut.LandscapeScoringError, match="explicit request"):
        sut.select_scoring_backend_from_parity(forged_implicit_relaxation)


def test_relaxed_cache_admission_falls_back_when_selected_logprob_or_argmax_drifts() -> (
    None
):
    prefix = [1, 2, 3]
    x1, y1, x2, y2 = 10, 11, 12, 13

    def normalization_drift(tokens: list[int]) -> torch.Tensor:
        values = _deterministic_logits_row(tuple(tokens), VOCAB_SIZE)
        if len(tokens) == len(prefix) + 1:
            values = values.clone()
            values[0] += 1.0
        return values

    threshold_failed = sut.run_cache_parity_gate(
        backend=_FakeCacheBackend(list(prefix), vocab_size=VOCAB_SIZE),
        prefill_logits=normalization_drift(prefix),
        prefix_token_ids=prefix,
        x1_token_id=x1,
        y1_token_id=y1,
        x2_token_id=x2,
        y2_token_id=y2,
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        full_reforward=normalization_drift,
        relaxed_selected_logprob_max_abs_diff=0.0,
    )
    assert threshold_failed["status"] == "failed"
    assert threshold_failed["admission_policy"]["all_coordinate_argmax_parity"]
    assert threshold_failed["admission_policy"]["effective_mode"] == (
        sut.UNCACHED_CACHE_ADMISSION_MODE
    )

    def argmax_drift(tokens: list[int]) -> torch.Tensor:
        values = _deterministic_logits_row(tuple(tokens), VOCAB_SIZE)
        if len(tokens) == len(prefix) + 1:
            values = values.clone()
            cache_argmax = COORD_START + int(
                torch.argmax(values[COORD_START:COORD_END]).item()
            )
            replacement = (
                COORD_START if cache_argmax != COORD_START else COORD_START + 1
            )
            values[replacement] = values[cache_argmax] + 10.0
        return values

    argmax_failed = sut.run_cache_parity_gate(
        backend=_FakeCacheBackend(list(prefix), vocab_size=VOCAB_SIZE),
        prefill_logits=argmax_drift(prefix),
        prefix_token_ids=prefix,
        x1_token_id=x1,
        y1_token_id=y1,
        x2_token_id=x2,
        y2_token_id=y2,
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        full_reforward=argmax_drift,
        relaxed_selected_logprob_max_abs_diff=100.0,
    )
    assert argmax_failed["status"] == "failed"
    assert argmax_failed["admission_policy"]["all_coordinate_argmax_parity"] is False
    assert (
        sut.select_scoring_backend_from_parity(argmax_failed)["selected_backend"]
        == sut.FULL_REFORWARD_SCORING_BACKEND
    )


def test_run_cache_parity_gate_rejects_a_backend_whose_crop_does_not_restore_cache_length() -> (
    None
):
    """A backend that fails to crop is caught immediately at branch-exit, before any
    order-invariance probe runs -- the gate never trusts a cache it cannot prove is restored."""

    prefix = [1, 2, 3]
    x1, y1, x2, y2 = 10, 11, 12, 13
    vocab = VOCAB_SIZE

    def oracle(tokens: list[int]) -> torch.Tensor:
        return _deterministic_logits_row(tuple(tokens), vocab)

    buggy_backend = _FakeCacheBackend(
        list(prefix), vocab_size=vocab, crop_is_buggy=True
    )
    prefill_logits = oracle(prefix)
    with pytest.raises(
        sut.LandscapeScoringError, match="did not crop back to its entry length"
    ):
        sut.run_cache_parity_gate(
            backend=buggy_backend,
            prefill_logits=prefill_logits,
            prefix_token_ids=prefix,
            x1_token_id=x1,
            y1_token_id=y1,
            x2_token_id=x2,
            y2_token_id=y2,
            coordinate_token_id_start=COORD_START,
            coordinate_token_id_end_exclusive=COORD_END,
            full_reforward=oracle,
        )


def test_run_cache_parity_gate_passes_via_rtol_despite_absolute_diff_exceeding_atol() -> (
    None
):
    """Exact unit semantics: torch.allclose(atol=1e-6, rtol=1e-5), not a bare max_abs_diff<=1e-6 check.

    Constructs a magnitude-10 logit with a 5e-5 absolute difference -- more
    than 50x atol alone -- that still satisfies atol + rtol*|b| = 1e-6 +
    1e-5*10 = 1.01e-4, so a correct allclose-based gate passes while the
    recorded max_abs_diff still honestly exceeds atol.
    """

    prefix = [1, 2, 3]
    x1, y1, x2, y2 = 10, 11, 12, 13
    vocab = VOCAB_SIZE
    bump = 5e-5
    assert bump > sut.CACHE_PARITY_ATOL  # the point: bigger than atol alone

    def full_reforward(tokens: list[int]) -> torch.Tensor:
        values = _deterministic_logits_row(tuple(tokens), vocab)
        if list(tokens) == prefix:
            # Only the root reading is pushed to a magnitude-10 baseline; every
            # other depth stays the untouched deterministic function, so it
            # matches the fake backend's own step() output exactly (diff=0).
            values = values.clone()
            values[0] = 10.0
        return values

    backend = _FakeCacheBackend(list(prefix), vocab_size=vocab)
    ground_truth_root = full_reforward(prefix)
    prefill_logits = ground_truth_root.clone()
    prefill_logits[0] = ground_truth_root[0] + bump  # ~10.00005 vs ~10.0

    result = sut.run_cache_parity_gate(
        backend=backend,
        prefill_logits=prefill_logits,
        prefix_token_ids=prefix,
        x1_token_id=x1,
        y1_token_id=y1,
        x2_token_id=x2,
        y2_token_id=y2,
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        full_reforward=full_reforward,
    )
    root_step = next(
        step
        for step in result["raw_logit_parity_steps"]
        if step["step"] == "root_predicts_x1"
    )
    assert root_step["max_abs_diff"] == pytest.approx(
        bump, abs=2e-6
    )  # float32 rounding of the injected bump
    assert root_step["max_abs_diff"] > sut.CACHE_PARITY_ATOL
    assert root_step["within_tolerance"] is True
    assert result["status"] == "passed"


def test_run_cache_parity_gate_records_and_gates_on_observed_layer_count() -> None:
    prefix = [1, 2, 3]
    x1, y1, x2, y2 = 10, 11, 12, 13
    vocab = VOCAB_SIZE

    def oracle(tokens: list[int]) -> torch.Tensor:
        return _deterministic_logits_row(tuple(tokens), vocab)

    matching_backend = _FakeCacheBackend(list(prefix), vocab_size=vocab, layer_count=28)
    matching_result = sut.run_cache_parity_gate(
        backend=matching_backend,
        prefill_logits=oracle(prefix),
        prefix_token_ids=prefix,
        x1_token_id=x1,
        y1_token_id=y1,
        x2_token_id=x2,
        y2_token_id=y2,
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        full_reforward=oracle,
        expected_layer_count=sut.EXPECTED_LIVE_QWEN_DECODER_LAYER_COUNT,
    )
    assert matching_result["cache_layers"] == {
        "observed_layer_count": 28,
        "expected_layer_count": 28,
        "layer_count_matches_expected": True,
    }
    assert matching_result["status"] == "passed"

    mismatched_backend = _FakeCacheBackend(
        list(prefix), vocab_size=vocab, layer_count=27
    )
    mismatched_result = sut.run_cache_parity_gate(
        backend=mismatched_backend,
        prefill_logits=oracle(prefix),
        prefix_token_ids=prefix,
        x1_token_id=x1,
        y1_token_id=y1,
        x2_token_id=x2,
        y2_token_id=y2,
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        full_reforward=oracle,
        expected_layer_count=sut.EXPECTED_LIVE_QWEN_DECODER_LAYER_COUNT,
    )
    assert mismatched_result["cache_layers"]["layer_count_matches_expected"] is False
    assert mismatched_result["status"] == "failed"


def test_run_cache_parity_gate_layer_count_is_non_gating_when_unknown() -> None:
    prefix = [1, 2, 3]
    x1, y1, x2, y2 = 10, 11, 12, 13
    vocab = VOCAB_SIZE

    def oracle(tokens: list[int]) -> torch.Tensor:
        return _deterministic_logits_row(tuple(tokens), vocab)

    backend = _FakeCacheBackend(
        list(prefix), vocab_size=vocab
    )  # layer_count=None by default
    result = sut.run_cache_parity_gate(
        backend=backend,
        prefill_logits=oracle(prefix),
        prefix_token_ids=prefix,
        x1_token_id=x1,
        y1_token_id=y1,
        x2_token_id=x2,
        y2_token_id=y2,
        coordinate_token_id_start=COORD_START,
        coordinate_token_id_end_exclusive=COORD_END,
        full_reforward=oracle,
    )
    assert result["cache_layers"] == {
        "observed_layer_count": None,
        "expected_layer_count": None,
        "layer_count_matches_expected": None,
    }
    assert result["status"] == "passed"


# ---------------------------------------------------------------------------
# P0 regression: real native materialization must feed prefill, never a
# None/placeholder image_grid_thw or an independently reconstructed prompt.
# ---------------------------------------------------------------------------


class _FakeOutput:
    def __init__(self, *, logits: torch.Tensor, past_key_values: Any) -> None:
        self.logits = logits
        self.past_key_values = past_key_values


class _FakeQwenModel:
    """Minimal stand-in for the Qwen3-VL model: records every call's kwargs."""

    def __init__(self, vocab_size: int) -> None:
        self._param = torch.zeros(1)
        self.vocab_size = vocab_size
        self.calls: list[dict[str, Any]] = []

    def parameters(self):
        yield self._param

    class _Owner:
        def get_rope_index(
            self, input_ids, image_grid_thw, video_grid_thw, attention_mask=None
        ):
            seq_len = int(input_ids.shape[1])
            position_ids = (
                torch.arange(seq_len, dtype=torch.long)
                .view(1, 1, -1)
                .expand(3, 1, -1)
                .clone()
            )
            rope_deltas = torch.zeros((1, 1), dtype=torch.long)
            return position_ids, rope_deltas

    @property
    def model(self):
        return _FakeQwenModel._Owner()

    def __call__(self, **kwargs: Any) -> _FakeOutput:
        self.calls.append(kwargs)
        seq_len = int(kwargs["input_ids"].shape[1])
        return _FakeOutput(
            logits=torch.zeros((1, seq_len, self.vocab_size), dtype=torch.float32),
            past_key_values=kwargs.get("past_key_values"),
        )


def _fake_native_inputs(*, prompt_ids: list[int], vocab_size: int) -> dict[str, Any]:
    return {
        "input_ids": torch.tensor([prompt_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        "pixel_values": torch.arange(8, dtype=torch.float32).reshape(4, 2),
    }


def test_prefill_context_rejects_missing_image_grid_thw_with_a_clear_error() -> None:
    """Regression for the P0: a None/placeholder grid must fail fast, not crash inside `.to()`."""

    model = _FakeQwenModel(vocab_size=VOCAB_SIZE)
    native_inputs = {
        **_fake_native_inputs(prompt_ids=[1, 2, 3], vocab_size=VOCAB_SIZE),
        "image_grid_thw": None,
    }
    with pytest.raises(
        sut.LandscapeScoringError, match="real materialized image_grid_thw"
    ):
        sut.prefill_context(
            model,
            native_prompt_inputs=native_inputs,
            generated_history_token_ids=[20, 21],
        )
    assert model.calls == []  # never even reaches the forward call


def test_prefill_context_concatenates_prompt_and_history_and_passes_pixels_through() -> (
    None
):
    model = _FakeQwenModel(vocab_size=VOCAB_SIZE)
    native_inputs = _fake_native_inputs(prompt_ids=[1, 2, 3], vocab_size=VOCAB_SIZE)
    result = sut.prefill_context(
        model, native_prompt_inputs=native_inputs, generated_history_token_ids=[20, 21]
    )
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["input_ids"].tolist() == [[1, 2, 3, 20, 21]]
    assert torch.equal(call["pixel_values"], native_inputs["pixel_values"])
    assert torch.equal(call["image_grid_thw"], native_inputs["image_grid_thw"])
    assert call["logits_to_keep"] == 1
    assert call["use_cache"] is True
    assert result.prefill_length == 5
    assert result.prefill_logits.shape[0] == VOCAB_SIZE


def test_prefill_context_rejects_empty_generated_history() -> None:
    model = _FakeQwenModel(vocab_size=VOCAB_SIZE)
    native_inputs = _fake_native_inputs(prompt_ids=[1, 2, 3], vocab_size=VOCAB_SIZE)
    with pytest.raises(ValueError, match="at least one"):
        sut.prefill_context(
            model, native_prompt_inputs=native_inputs, generated_history_token_ids=[]
        )


def test_full_reforward_closure_rejects_missing_image_grid_thw() -> None:
    model = _FakeQwenModel(vocab_size=VOCAB_SIZE)
    native_inputs = {
        **_fake_native_inputs(prompt_ids=[1, 2, 3], vocab_size=VOCAB_SIZE),
        "image_grid_thw": None,
    }
    with pytest.raises(
        sut.LandscapeScoringError, match="real materialized image_grid_thw"
    ):
        sut._build_full_reforward_closure(model, native_prompt_inputs=native_inputs)


def test_full_reforward_closure_uses_no_cache_and_passes_pixels_through() -> None:
    model = _FakeQwenModel(vocab_size=VOCAB_SIZE)
    native_inputs = _fake_native_inputs(prompt_ids=[1, 2, 3], vocab_size=VOCAB_SIZE)
    closure = sut._build_full_reforward_closure(
        model, native_prompt_inputs=native_inputs
    )
    logits = closure([1, 2, 3, 20])
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["use_cache"] is False
    assert torch.equal(call["pixel_values"], native_inputs["pixel_values"])
    assert call["input_ids"].tolist() == [[1, 2, 3, 20]]
    assert logits.shape[0] == VOCAB_SIZE


class _FakeHFBackendSessionForMaterialization:
    """Mimics ``HFBackendSession._materialize_native_inputs``'s return contract."""

    def __init__(
        self,
        *,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> None:
        self._input_ids = input_ids
        self._pixel_values = pixel_values
        self._image_grid_thw = image_grid_thw

    def _materialize_native_inputs(self, requests: Any):
        native_inputs = {
            "input_ids": self._input_ids,
            "attention_mask": torch.ones_like(self._input_ids),
            "pixel_values": self._pixel_values,
            "image_grid_thw": self._image_grid_thw,
        }
        executed_prompt_ids = (tuple(int(v) for v in self._input_ids[0].tolist()),)
        observed_grids = (tuple(int(v) for v in self._image_grid_thw[0].tolist()),)
        executed_media_sha256 = ("fake-media-sha256",)
        return native_inputs, executed_prompt_ids, observed_grids, executed_media_sha256


def test_materialized_native_inputs_regression_would_fail_on_a_none_image_grid_thw() -> (
    None
):
    """Integration-style regression for the P0.

    Chains a fake ``HFBackendSession``-shaped materialization call into the
    real ``prefill_context`` consumer, proving the wiring feeds real
    pixel_values/image_grid_thw through to the model, and that substituting
    the historical ``image_grid_thw=None`` placeholder at this exact seam
    fails fast with a clear error instead of crashing inside ``.to()`` or
    silently prefilling a text-only sequence.
    """

    prompt_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    pixel_values = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
    session = _FakeHFBackendSessionForMaterialization(
        input_ids=prompt_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw
    )
    native_inputs, executed_prompt_ids, _observed, _media = (
        session._materialize_native_inputs(("fake-request",))
    )
    assert tuple(executed_prompt_ids[0]) == (1, 2, 3)

    model = _FakeQwenModel(vocab_size=VOCAB_SIZE)
    result = sut.prefill_context(
        model, native_prompt_inputs=native_inputs, generated_history_token_ids=[20, 21]
    )
    assert result.prefill_length == 5
    call = model.calls[0]
    assert torch.equal(call["pixel_values"], pixel_values)
    assert torch.equal(call["image_grid_thw"], image_grid_thw)

    broken_native_inputs = {**native_inputs, "image_grid_thw": None}
    with pytest.raises(
        sut.LandscapeScoringError, match="real materialized image_grid_thw"
    ):
        sut.prefill_context(
            model,
            native_prompt_inputs=broken_native_inputs,
            generated_history_token_ids=[20, 21],
        )


# ---------------------------------------------------------------------------
# Candidate prompt-prefix <-> generated-history binding
# ---------------------------------------------------------------------------


def test_derive_generated_history_token_ids_matches_prefix_minus_prompt_prefix(
    tmp_path: Path,
) -> None:
    candidate = _candidate_row(
        _tmp_dir=tmp_path,
        candidate_id="cand-hist",
        prefix_token_ids=[1, 2, 3, 20, 21],
        prompt_prefix_token_count=3,
    )
    assert sut.derive_generated_history_token_ids(candidate) == (20, 21)


def test_candidate_prefix_binds_consistently_to_prompt_and_generated_history(
    tmp_path: Path,
) -> None:
    reconstructed_prompt = [1, 2, 3]
    history = (20, 21)
    candidate = _candidate_row(
        _tmp_dir=tmp_path,
        candidate_id="cand-bind",
        prefix_token_ids=[*reconstructed_prompt, *history],
        prompt_prefix_token_count=len(reconstructed_prompt),
    )
    sut.verify_production_prompt_prefix(
        candidate, reconstructed_prompt
    )  # must not raise
    assert sut.derive_generated_history_token_ids(candidate) == history


def test_candidate_prefix_binding_fails_when_prompt_region_diverges(
    tmp_path: Path,
) -> None:
    candidate = _candidate_row(
        _tmp_dir=tmp_path,
        candidate_id="cand-bind-2",
        prefix_token_ids=[1, 2, 3, 20, 21],
        prompt_prefix_token_count=3,
    )
    with pytest.raises(
        sut.LandscapeScoringError, match="production prompt/image reconstruction"
    ):
        sut.verify_production_prompt_prefix(candidate, [1, 2, 999])


# ---------------------------------------------------------------------------
# Artifact-manifest digest resolution (flat and nested forms)
# ---------------------------------------------------------------------------


def test_load_artifact_manifest_digests_resolves_flat_and_nested_forms(
    tmp_path: Path,
) -> None:
    digest_a = "a" * 64
    digest_b = "b" * 64
    manifest_path = tmp_path / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps({"owner_ledger": digest_a, "decision_rules": {"sha256": digest_b}}),
        encoding="utf-8",
    )
    resolved = sut.load_artifact_manifest_digests(
        manifest_path, required_keys=["owner_ledger", "decision_rules"]
    )
    assert resolved == {"owner_ledger": digest_a, "decision_rules": digest_b}


def test_load_artifact_manifest_digests_resolves_nested_digest_alias_key(
    tmp_path: Path,
) -> None:
    digest = "c" * 64
    manifest_path = tmp_path / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps({"candidates": {"digest": digest}}), encoding="utf-8"
    )
    resolved = sut.load_artifact_manifest_digests(
        manifest_path, required_keys=["candidates"]
    )
    assert resolved == {"candidates": digest}


def test_load_artifact_manifest_digests_fails_fast_on_missing_key(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "artifact-manifest.json"
    manifest_path.write_text(json.dumps({"owner_ledger": "a" * 64}), encoding="utf-8")
    with pytest.raises(
        sut.LandscapeScoringError, match="does not bind a resolvable sha256 digest"
    ):
        sut.load_artifact_manifest_digests(
            manifest_path, required_keys=["owner_ledger", "prediction_row_ledger"]
        )


def test_load_artifact_manifest_digests_fails_fast_on_unresolvable_nested_value(
    tmp_path: Path,
) -> None:
    """The historical bug: a nested manifest without a recognized digest key would have been
    silently treated as 'no digest asserted' by a flat `isinstance(v, str)` filter."""

    manifest_path = tmp_path / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps({"owner_ledger": {"path": "/somewhere"}}), encoding="utf-8"
    )
    with pytest.raises(
        sut.LandscapeScoringError, match="does not bind a resolvable sha256 digest"
    ):
        sut.load_artifact_manifest_digests(
            manifest_path, required_keys=["owner_ledger"]
        )


# ---------------------------------------------------------------------------
# Authoritative candidate-builder-v2 production integration
# ---------------------------------------------------------------------------


def _materialize_v2_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    builder_tests = runpy.run_path(
        str(Path(__file__).with_name("test_build_sorted_owner_basin_candidates.py"))
    )
    candidates_path, _receipt_path, _receipt = builder_tests["_materialize"](tmp_path)
    return (
        tmp_path / "landscape-decision-rules.json",
        tmp_path / "owner-context-ledger.jsonl",
        candidates_path,
    )


def _runtime_identity_fixture(
    tmp_path: Path, *, rules: sut.DecisionRules, candidate: sut.CandidateRow
) -> tuple[Path, Path, sut.CandidateRow, dict[str, Any]]:
    component = tmp_path / "component.bin"
    component.write_bytes(b"component")
    component_record = {
        "path": str(component),
        "bytes": component.stat().st_size,
        "sha256": sut.sha256_file(component),
    }
    tokenizer_payload = {"tokenizer_vocab_size": rules.model_vocab_size}
    model_payload = {"family": "fixture-model"}
    processor_payload = {"processor_class": "fixture-processor"}
    tokenizer_source = {
        "component_files": [component_record],
        "path": str(tmp_path),
        "tokenizer_identity": tokenizer_payload,
    }
    model_source = {
        "component_files": [component_record],
        "model_identity": model_payload,
        "model_identity_fingerprint": sut.sha256_json(model_payload),
    }
    infer_fingerprint = "d" * 64
    generation_fingerprint = "e" * 64
    likelihood = {
        "policy": "fp32_log_softmax_after_active_generation_processors",
        "raw": "fp32_log_softmax_unmodified_lm_head_logits",
        "score_owned_channel": "policy_logprob",
    }
    runtime_source = {
        "backend": "hf",
        "backend_mode": "generate",
        "backend_version": "fixture",
        "effective_settings": {
            "backend_options": {"hf": {"attn_implementation": "sdpa"}},
            "batch_size": 4,
            "device": "cuda",
            "output_scores": True,
        },
        "generation_config_fingerprint": generation_fingerprint,
        "likelihood_semantics": likelihood,
        "precision": "float32",
        "processor_identity": processor_payload,
        "processor_identity_fingerprint": sut.sha256_json(processor_payload),
        "resolved_config_fingerprints": {"infer_config": infer_fingerprint},
        "response_family": "hf",
    }
    source_jsonl = tmp_path / "source.jsonl"
    source_jsonl.write_text("{}\n", encoding="utf-8")
    content = {
        "schema_version": "sorted-owner-basin-runtime-identity.v1",
        "status": "frozen",
        "sources": {
            "source_panel": {
                "file_sha256": sut.sha256_file(source_jsonl),
                "path": str(source_jsonl),
            }
        },
        "tokenizer": {
            "path": str(tmp_path),
            "identity_sha256": sut.sha256_json(tokenizer_source),
            "identity_source": tokenizer_source,
        },
        "model": {
            "identity_sha256": sut.sha256_json(model_source),
            "identity_source": model_source,
        },
        "runtime": {
            "identity_sha256": sut.sha256_json(runtime_source),
            "identity_source": runtime_source,
        },
        "coordinate_vocabulary": {
            "coordinate_min": 0,
            "coordinate_max": 9,
            "token_id_start": rules.schema_tokens["coordinate_token_id_start"],
            "token_id_end_exclusive": rules.schema_tokens[
                "coordinate_token_id_end_exclusive"
            ],
        },
        "model_vocab_size": rules.model_vocab_size,
        "schema_tokens": dict(rules.schema_tokens),
    }
    identity = {**content, "receipt_digest": sut.sha256_json(content)}
    identity_path = tmp_path / "runtime-identity.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    vocabulary: dict[str, str] = {
        "tokenizer_identity_sha256": sut.sha256_json(tokenizer_source),
        "model_identity_sha256": sut.sha256_json(model_source),
        "runtime_identity_sha256": sut.sha256_json(runtime_source),
    }
    admitted_candidate = replace(
        candidate,
        raw_payload={**dict(candidate.raw_payload), "vocabulary_attestation": vocabulary},
    )
    backend_receipt = {
        "backend": "hf",
        "backend_mode": "generate",
        "response_family": "hf",
        "backend_version": "fixture",
        "model_identity": model_payload,
        "tokenizer_identity": tokenizer_payload,
        "processor_identity": processor_payload,
        "generation_config_fingerprint": generation_fingerprint,
        "effective_settings": {
            "backend_options": {"hf": {"attn_implementation": "sdpa"}},
            "batch_size": 4,
            "device": "cuda",
            "output_scores": True,
            "observed_model_dtype": {
                "parameter_dtype_counts": {"torch.float32": 10},
                "parameter_dtype_names": ["torch.float32"],
            },
            "observed_attn_implementation": "sdpa",
        },
        "likelihood_semantics": likelihood,
    }
    return identity_path, source_jsonl, admitted_candidate, backend_receipt


def _phase_a_freeze_fixture(tmp_path: Path) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    semantic_core_sha256 = "a" * 64
    control_rules_sha256 = "b" * 64
    freeze: dict[str, Any] = {
        "schema_version": sut.NON_C_SMOKE_FREEZE_SCHEMA_VERSION,
        "status": "passed",
        "control_decision_rules_sha256": control_rules_sha256,
        "semantic_core_sha256": semantic_core_sha256,
        "control_score_artifact_sha256": "c" * 64,
        "control_score_receipt_sha256": "d" * 64,
        "control_summary_sha256": "e" * 64,
        "control_summary_receipt_sha256": "f" * 64,
        "calibration_receipt_sha256": "1" * 64,
        "independent_reconstruction": "passed",
        "gates": {
            "representative_positive_control": "passed",
            "mandatory_cache_parity": "passed",
            "scoring_backend_admission": "cache_parity_passed",
            "b2_reviewed_pair": "passed",
            "free_surface_executed_raw_only": "passed",
        },
        "c_outcomes_read": False,
        "scientific_conclusion": None,
    }
    freeze_path = tmp_path / "non-c-smoke-freeze-receipt.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    binding = {
        "sha256": sut.sha256_file(freeze_path),
        "control_decision_rules_sha256": control_rules_sha256,
        "semantic_core_sha256": semantic_core_sha256,
    }
    rule_document = {
        "structural_status": sut.SENTINEL_STRUCTURAL_STATUS,
        "non_c_smoke_freeze_receipt": binding,
    }
    rules_path = tmp_path / "sentinel-rules.json"
    rules_path.write_text(json.dumps(rule_document), encoding="utf-8")
    candidates_path = tmp_path / "sentinel-candidates.jsonl"
    candidates_path.write_text("{}\n", encoding="utf-8")
    candidate_receipt = {
        "schema_version": sut.candidate_builder.SCHEMA_VERSION,
        "inputs": {
            "landscape_decision_rules_sha256": sut.sha256_file(rules_path),
            "core_rule_digest": semantic_core_sha256,
        },
        "output_jsonl": {
            "sha256": sut.sha256_file(candidates_path),
            "row_count": 1,
        },
    }
    candidate_receipt_path = tmp_path / "sentinel-candidates-receipt.json"
    candidate_receipt_path.write_text(
        json.dumps(candidate_receipt), encoding="utf-8"
    )
    rules = sut.DecisionRules(
        rules_digest=semantic_core_sha256,
        schema_tokens={},
        owner_canonical_description={},
        foil_set_digests={},
        model_vocab_size=1,
        numeric_tolerance=1e-6,
        rules_file_sha256=sut.sha256_file(rules_path),
        contract_mode="production",
        structural_status=sut.SENTINEL_STRUCTURAL_STATUS,
        non_c_smoke_freeze_binding=binding,
    )
    return {
        "freeze": freeze,
        "freeze_path": freeze_path,
        "binding": binding,
        "rules": rules,
        "rules_path": rules_path,
        "rule_document": rule_document,
        "candidates_path": candidates_path,
        "candidate_receipt": candidate_receipt,
        "candidate_receipt_path": candidate_receipt_path,
    }


def _reseal_phase_a_fixture(
    fixture: dict[str, Any], *, binding: dict[str, Any]
) -> sut.DecisionRules:
    rule_document = {
        **fixture["rule_document"],
        "non_c_smoke_freeze_receipt": binding,
    }
    rules_path = fixture["rules_path"]
    rules_path.write_text(json.dumps(rule_document), encoding="utf-8")
    candidate_receipt = dict(fixture["candidate_receipt"])
    candidate_receipt["inputs"] = {
        **candidate_receipt["inputs"],
        "landscape_decision_rules_sha256": sut.sha256_file(rules_path),
    }
    fixture["candidate_receipt_path"].write_text(
        json.dumps(candidate_receipt), encoding="utf-8"
    )
    return replace(
        fixture["rules"],
        rules_file_sha256=sut.sha256_file(rules_path),
        non_c_smoke_freeze_binding=binding,
    )


def _validate_phase_a_fixture(
    fixture: dict[str, Any],
    *,
    rules: sut.DecisionRules | None = None,
    freeze_path: Path | None = None,
    candidate_receipt_path: Path | None = None,
) -> dict[str, Any]:
    return sut.validate_phase_a_freeze_binding(
        rules=fixture["rules"] if rules is None else rules,
        rules_path=fixture["rules_path"],
        candidates_path=fixture["candidates_path"],
        candidate_count=1,
        candidate_receipt_path=(
            fixture["candidate_receipt_path"]
            if candidate_receipt_path is None
            else candidate_receipt_path
        ),
        freeze_receipt_path=(
            fixture["freeze_path"] if freeze_path is None else freeze_path
        ),
    )


def test_v2_parser_and_ledger_gate_accept_authoritative_builder_fixture(
    tmp_path: Path,
) -> None:
    rules_path, ledger_path, candidates_path = _materialize_v2_fixture(tmp_path)
    rules = sut.load_decision_rules(rules_path)
    candidates = sut.load_candidate_rows(candidates_path, rules=rules)
    ledger = sut.load_owner_context_ledger(ledger_path, rules=rules)
    receipt = sut.validate_v2_candidate_contract(
        candidates, owner_context_ledger=ledger, rules=rules
    )
    assert rules.contract_version == sut.CANDIDATE_SCHEMA_VERSION
    assert receipt == {
        "schema_version": sut.CANDIDATE_SCHEMA_VERSION,
        "owner_context_count": 1,
        "free_root_count": 1,
        "conditional_y1_plan_count": 3,
        "complete_box_candidate_count": 12,
        "status": "complete",
    }
    complete = next(
        candidate
        for candidate in candidates
        if candidate.record_type == "complete_box_candidate"
    )
    outcome = sut.verify_candidate_proposal(
        complete,
        pure_core=sut.require_production_pure_core(sut._load_pure_core()),
    )
    assert outcome["pure_core_recomputation"] == "passed"


def test_runtime_identity_two_stage_admission_accepts_exact_projection(
    tmp_path: Path,
) -> None:
    rules_path, _ledger_path, candidates_path = _materialize_v2_fixture(tmp_path)
    rules = sut.load_decision_rules(rules_path)
    candidate = sut.load_candidate_rows(candidates_path, rules=rules)[0]
    identity_path, source_path, candidate, backend_receipt = _runtime_identity_fixture(
        tmp_path,
        rules=rules,
        candidate=candidate,
    )
    identity, preload = sut.validate_runtime_identity_preload(
        identity_path=identity_path,
        resolved_infer_fingerprint="d" * 64,
        source_jsonl=source_path,
        candidates=[candidate],
        rules=rules,
    )
    postload = sut.validate_runtime_identity_postload(
        frozen_identity=identity,
        backend_receipt=backend_receipt,
        resolved_infer_fingerprint="d" * 64,
    )
    assert preload["status"] == "passed"
    assert preload["component_file_count"] == 1
    assert postload["status"] == "passed"
    assert postload["expected_projection"] == postload["observed_projection"]


def test_sealed_sentinel_consumes_exact_phase_a_freeze_and_candidate_receipt(
    tmp_path: Path,
) -> None:
    fixture = _phase_a_freeze_fixture(tmp_path)
    admission = _validate_phase_a_fixture(fixture)
    assert admission["status"] == "passed_post_phase_a_freeze"
    assert admission["freeze_receipt"]["c_outcomes_read"] is False
    assert admission["candidate_receipt"]["sentinel_rules_sha256"] == sut.sha256_file(
        fixture["rules_path"]
    )
    assert len(admission["admission_sha256"]) == 64


def _rewrite_phase_a_freeze_gates(
    fixture: dict[str, Any], gates: dict[str, Any]
) -> sut.DecisionRules:
    freeze = {**fixture["freeze"], "gates": gates}
    fixture["freeze_path"].write_text(json.dumps(freeze), encoding="utf-8")
    binding = {
        **fixture["binding"],
        "sha256": sut.sha256_file(fixture["freeze_path"]),
    }
    return _reseal_phase_a_fixture(fixture, binding=binding)


def test_sealed_sentinel_accepts_failed_parity_with_admitted_uncached_freeze(
    tmp_path: Path,
) -> None:
    fixture = _phase_a_freeze_fixture(tmp_path)
    gates = {
        **fixture["freeze"]["gates"],
        "mandatory_cache_parity": "failed",
        "scoring_backend_admission": (
            "cache_parity_failed_uncached_reference_used"
        ),
    }
    rules = _rewrite_phase_a_freeze_gates(fixture, gates)
    admission = _validate_phase_a_fixture(fixture, rules=rules)
    assert admission["status"] == "passed_post_phase_a_freeze"
    assert admission["freeze_receipt"]["mandatory_cache_parity"] == "failed"
    assert (
        admission["freeze_receipt"]["scoring_backend_admission"]
        == "cache_parity_failed_uncached_reference_used"
    )


@pytest.mark.parametrize(
    ("mandatory_cache_parity", "scoring_backend_admission"),
    [
        ("failed", "cache_parity_passed"),
        ("passed", "cache_parity_failed_uncached_reference_used"),
        ("unknown", "cache_parity_passed"),
        ("failed", None),
    ],
)
def test_sealed_sentinel_rejects_inconsistent_or_unknown_backend_freeze_gate(
    tmp_path: Path,
    mandatory_cache_parity: str,
    scoring_backend_admission: str | None,
) -> None:
    fixture = _phase_a_freeze_fixture(tmp_path)
    gates = {
        **fixture["freeze"]["gates"],
        "mandatory_cache_parity": mandatory_cache_parity,
    }
    if scoring_backend_admission is None:
        gates.pop("scoring_backend_admission")
    else:
        gates["scoring_backend_admission"] = scoring_backend_admission
    rules = _rewrite_phase_a_freeze_gates(fixture, gates)
    with pytest.raises(
        sut.LandscapeScoringError, match="scoring backend admission"
    ):
        _validate_phase_a_fixture(fixture, rules=rules)


def test_sealed_sentinel_rejects_other_freeze_gate_drift(tmp_path: Path) -> None:
    fixture = _phase_a_freeze_fixture(tmp_path)
    gates = {
        **fixture["freeze"]["gates"],
        "b2_reviewed_pair": "failed",
    }
    rules = _rewrite_phase_a_freeze_gates(fixture, gates)
    with pytest.raises(sut.LandscapeScoringError, match="passed non-C smoke gate"):
        _validate_phase_a_fixture(fixture, rules=rules)


@pytest.mark.parametrize("missing", ["freeze", "candidate_receipt"])
def test_sealed_sentinel_rejects_missing_phase_a_inputs(
    tmp_path: Path, missing: str
) -> None:
    fixture = _phase_a_freeze_fixture(tmp_path)
    with pytest.raises(sut.LandscapeScoringError, match="requires"):
        sut.validate_phase_a_freeze_binding(
            rules=fixture["rules"],
            rules_path=fixture["rules_path"],
            candidates_path=fixture["candidates_path"],
            candidate_count=1,
            candidate_receipt_path=(
                None
                if missing == "candidate_receipt"
                else fixture["candidate_receipt_path"]
            ),
            freeze_receipt_path=(
                None if missing == "freeze" else fixture["freeze_path"]
            ),
        )


def test_sealed_sentinel_rejects_stale_freeze_binding(tmp_path: Path) -> None:
    fixture = _phase_a_freeze_fixture(tmp_path)
    stale_binding = {**fixture["binding"], "sha256": "0" * 64}
    rules = _reseal_phase_a_fixture(fixture, binding=stale_binding)
    with pytest.raises(sut.LandscapeScoringError, match="file SHA differs"):
        _validate_phase_a_fixture(fixture, rules=rules)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("semantic_core_sha256", "wrong shared semantic core"),
        ("control_decision_rules_sha256", "wrong control outer SHA"),
    ],
)
def test_sealed_sentinel_rejects_wrong_core_or_control_binding(
    tmp_path: Path, field: str, message: str
) -> None:
    fixture = _phase_a_freeze_fixture(tmp_path)
    freeze = {**fixture["freeze"], field: "9" * 64}
    fixture["freeze_path"].write_text(json.dumps(freeze), encoding="utf-8")
    binding = {
        **fixture["binding"],
        "sha256": sut.sha256_file(fixture["freeze_path"]),
    }
    rules = _reseal_phase_a_fixture(fixture, binding=binding)
    with pytest.raises(sut.LandscapeScoringError, match=message):
        _validate_phase_a_fixture(fixture, rules=rules)


def test_sealed_sentinel_rejects_tampered_freeze_bytes_and_candidate_receipt(
    tmp_path: Path,
) -> None:
    fixture = _phase_a_freeze_fixture(tmp_path)
    fixture["freeze_path"].write_text(
        fixture["freeze_path"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(sut.LandscapeScoringError, match="file SHA differs"):
        _validate_phase_a_fixture(fixture)

    fixture = _phase_a_freeze_fixture(tmp_path / "candidate")
    candidate_receipt = dict(fixture["candidate_receipt"])
    candidate_receipt["output_jsonl"] = {
        **candidate_receipt["output_jsonl"],
        "sha256": "8" * 64,
    }
    fixture["candidate_receipt_path"].write_text(
        json.dumps(candidate_receipt), encoding="utf-8"
    )
    with pytest.raises(sut.LandscapeScoringError, match="exact candidate JSONL"):
        _validate_phase_a_fixture(fixture)


def test_draft_control_has_no_hidden_sentinel_freeze_dependency(
    tmp_path: Path,
) -> None:
    fixture = _phase_a_freeze_fixture(tmp_path)
    draft_rules = replace(
        fixture["rules"],
        structural_status=sut.DRAFT_CONTROL_STRUCTURAL_STATUS,
        non_c_smoke_freeze_binding={},
    )
    admission = sut.validate_phase_a_freeze_binding(
        rules=draft_rules,
        rules_path=fixture["rules_path"],
        candidates_path=fixture["candidates_path"],
        candidate_count=1,
        candidate_receipt_path=None,
        freeze_receipt_path=None,
    )
    assert admission["sentinel_dependency_consumed"] is False
    with pytest.raises(sut.LandscapeScoringError, match="hidden sentinel"):
        sut.validate_phase_a_freeze_binding(
            rules=draft_rules,
            rules_path=fixture["rules_path"],
            candidates_path=fixture["candidates_path"],
            candidate_count=1,
            candidate_receipt_path=None,
            freeze_receipt_path=fixture["freeze_path"],
        )


def test_runtime_identity_preload_rejects_config_and_component_drift(
    tmp_path: Path,
) -> None:
    rules_path, _ledger_path, candidates_path = _materialize_v2_fixture(tmp_path)
    rules = sut.load_decision_rules(rules_path)
    candidate = sut.load_candidate_rows(candidates_path, rules=rules)[0]
    identity_path, source_path, candidate, _ = _runtime_identity_fixture(
        tmp_path,
        rules=rules,
        candidate=candidate,
    )
    with pytest.raises(sut.LandscapeScoringError, match="infer-config fingerprint"):
        sut.validate_runtime_identity_preload(
            identity_path=identity_path,
            resolved_infer_fingerprint="0" * 64,
            source_jsonl=source_path,
            candidates=[candidate],
            rules=rules,
        )
    (tmp_path / "component.bin").write_bytes(b"drift")
    with pytest.raises(sut.LandscapeScoringError, match="component byte count drifted"):
        sut.validate_runtime_identity_preload(
            identity_path=identity_path,
            resolved_infer_fingerprint="d" * 64,
            source_jsonl=source_path,
            candidates=[candidate],
            rules=rules,
        )


def test_runtime_identity_postload_rejects_non_fp32_or_non_sdpa(
    tmp_path: Path,
) -> None:
    rules_path, _ledger_path, candidates_path = _materialize_v2_fixture(tmp_path)
    rules = sut.load_decision_rules(rules_path)
    candidate = sut.load_candidate_rows(candidates_path, rules=rules)[0]
    identity_path, _source_path, _candidate, backend_receipt = (
        _runtime_identity_fixture(tmp_path, rules=rules, candidate=candidate)
    )
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    bad_dtype = {
        **backend_receipt,
        "effective_settings": {
            **backend_receipt["effective_settings"],
            "observed_model_dtype": {
                "parameter_dtype_counts": {"torch.bfloat16": 10},
                "parameter_dtype_names": ["torch.bfloat16"],
            },
        },
    }
    with pytest.raises(sut.LandscapeScoringError, match="not uniformly FP32"):
        sut.validate_runtime_identity_postload(
            frozen_identity=identity,
            backend_receipt=bad_dtype,
            resolved_infer_fingerprint="d" * 64,
        )
    bad_attention = {
        **backend_receipt,
        "effective_settings": {
            **backend_receipt["effective_settings"],
            "observed_attn_implementation": "flash_attention_2",
        },
    }
    with pytest.raises(sut.LandscapeScoringError, match="not SDPA"):
        sut.validate_runtime_identity_postload(
            frozen_identity=identity,
            backend_receipt=bad_attention,
            resolved_infer_fingerprint="d" * 64,
        )


@pytest.mark.parametrize(
    ("field", "drifted"),
    [
        ("backend_mode", "score"),
        ("backend_version", "other"),
        ("response_family", "other"),
        ("generation_config_fingerprint", "0" * 64),
        ("likelihood_semantics", {"score_owned_channel": "raw_model_logprob"}),
        ("processor_identity", {"processor_class": "drifted"}),
    ],
)
def test_runtime_identity_postload_rejects_exact_identity_source_drift(
    tmp_path: Path, field: str, drifted: Any
) -> None:
    rules_path, _ledger_path, candidates_path = _materialize_v2_fixture(tmp_path)
    rules = sut.load_decision_rules(rules_path)
    candidate = sut.load_candidate_rows(candidates_path, rules=rules)[0]
    identity_path, _source_path, _candidate, backend_receipt = (
        _runtime_identity_fixture(tmp_path, rules=rules, candidate=candidate)
    )
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    with pytest.raises(sut.LandscapeScoringError, match="exact frozen runtime"):
        sut.validate_runtime_identity_postload(
            frozen_identity=identity,
            backend_receipt={**backend_receipt, field: drifted},
            resolved_infer_fingerprint="d" * 64,
        )


def test_runtime_identity_postload_rejects_effective_setting_and_config_drift(
    tmp_path: Path,
) -> None:
    rules_path, _ledger_path, candidates_path = _materialize_v2_fixture(tmp_path)
    rules = sut.load_decision_rules(rules_path)
    candidate = sut.load_candidate_rows(candidates_path, rules=rules)[0]
    identity_path, _source_path, _candidate, backend_receipt = (
        _runtime_identity_fixture(tmp_path, rules=rules, candidate=candidate)
    )
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    drifted_settings = {
        **backend_receipt,
        "effective_settings": {
            **backend_receipt["effective_settings"],
            "batch_size": 8,
        },
    }
    with pytest.raises(sut.LandscapeScoringError, match="exact frozen runtime"):
        sut.validate_runtime_identity_postload(
            frozen_identity=identity,
            backend_receipt=drifted_settings,
            resolved_infer_fingerprint="d" * 64,
        )
    with pytest.raises(sut.LandscapeScoringError, match="exact frozen runtime"):
        sut.validate_runtime_identity_postload(
            frozen_identity=identity,
            backend_receipt=backend_receipt,
            resolved_infer_fingerprint="1" * 64,
        )


def test_live_config_binding_requires_exact_source_and_predeclared_fp32(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    other = tmp_path / "other.jsonl"
    other.write_text("{}\n", encoding="utf-8")
    config = SimpleNamespace(
        data=SimpleNamespace(input_jsonl=str(source)),
        model=SimpleNamespace(dtype="fp32"),
    )
    assert sut.validate_live_config_binding(
        config=config, source_jsonl=source
    )["status"] == "passed"
    with pytest.raises(sut.LandscapeScoringError, match="data.input_jsonl"):
        sut.validate_live_config_binding(config=config, source_jsonl=other)
    with pytest.raises(sut.LandscapeScoringError, match="already be fp32"):
        sut.validate_live_config_binding(
            config=SimpleNamespace(
                data=SimpleNamespace(input_jsonl=str(source)),
                model=SimpleNamespace(dtype="bf16"),
            ),
            source_jsonl=source,
        )


def test_runtime_identity_derives_and_rehashes_transitive_task0_panel(
    tmp_path: Path,
) -> None:
    panel = tmp_path / "panel.jsonl"
    panel.write_text("{}\n", encoding="utf-8")
    task0_content = {
        "schema_version": "sorted-owner-basin-task0-execution-receipt.v2",
        "inputs": {
            "bound_files": [
                {
                    "role": "frozen_panel",
                    "path": str(panel),
                    "bytes": panel.stat().st_size,
                    "sha256": sut.sha256_file(panel),
                }
            ]
        },
    }
    task0 = {
        **task0_content,
        "execution_receipt_content_sha256": sut.sha256_json(task0_content),
    }
    task0_path = tmp_path / "task0.json"
    task0_path.write_text(json.dumps(task0), encoding="utf-8")
    identity = {
        "sources": {
            "task0_execution_receipt": {
                "path": str(task0_path),
                "file_sha256": sut.sha256_file(task0_path),
                "content_sha256": sut.sha256_json(task0_content),
            }
        }
    }
    assert sut._frozen_source_panel_digest(identity) == sut.sha256_file(panel)
    panel.write_text('{"drift": true}\n', encoding="utf-8")
    with pytest.raises(sut.LandscapeScoringError, match="byte count drifted"):
        sut._frozen_source_panel_digest(identity)


def test_v2_parser_rejects_fixed_coordinate_double_append_in_root_prefix(
    tmp_path: Path,
) -> None:
    rules_path, _ledger_path, candidates_path = _materialize_v2_fixture(tmp_path)
    rules = sut.load_decision_rules(rules_path)
    rows = [json.loads(line) for line in candidates_path.read_text().splitlines()]
    plan = next(
        row for row in rows if row["record_type"] == "conditional_y1_score_plan"
    )
    plan["prefix_token_ids"].append(plan["fixed_coord_token_ids"][0])
    plan["root_prefix_token_ids"] = list(plan["prefix_token_ids"])
    digest = sut.sha256_json(plan["prefix_token_ids"])
    plan["prefix_token_ids_sha256"] = digest
    plan["root_prefix_token_ids_sha256"] = digest
    tampered = tmp_path / "tampered-candidates.jsonl"
    tampered.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(sut.LandscapeScoringError, match="end exactly at box_start"):
        sut.load_candidate_rows(tampered, rules=rules)


def test_full_conditional_y1_attestation_uses_core_and_builder_input_fields(
    tmp_path: Path,
) -> None:
    rules_path, ledger_path, candidates_path = _materialize_v2_fixture(tmp_path)
    rules = sut.load_decision_rules(rules_path)
    candidates = sut.load_candidate_rows(candidates_path, rules=rules)
    ledger = sut.load_owner_context_ledger(ledger_path, rules=rules)
    plans = [
        candidate
        for candidate in candidates
        if candidate.record_type == "conditional_y1_score_plan"
    ]
    scored_rows = [
        {
            "candidate_id": plan.candidate_id,
            "request_kind": "dense_scan",
            "scan_slot": "y1",
            "raw_bin_scan": {
                "bin_logprobs": [-float(index + 1) for index in range(10)]
            },
        }
        for plan in plans
    ]
    core = sut.require_production_pure_core(sut._load_pure_core())
    attestations = sut.build_conditional_y1_completeness_attestations(
        pure_core=core,
        candidates=candidates,
        scored_rows=scored_rows,
        owner_context_ledger=ledger,
        rules=rules,
    )
    assert len(attestations) == 1
    assert attestations[0]["attestation_kind"] == "test_fixture"
    assert attestations[0]["gt_box"] == [2, 3, 5, 6]
    assert attestations[0]["declared_x1_bins"] == (2, 3, 4)
    assert len(attestations[0]["per_x1_receipt_digests"]) == 3
    assert len(attestations[0]["completeness_digest"]) == 64


def test_deterministic_farthest_point_selector_follows_frozen_ties_and_stop() -> None:
    selected, receipt = sut.deterministic_farthest_point_anchors(
        [
            (0, 0, 10.0),
            (30, 0, 9.0),
            (0, 30, 9.0),
            (30, 30, 8.0),
            (1, 1, 7.0),
        ],
        minimum_distance_bins=24,
    )
    assert selected == [
        (0, 0, 10.0),
        (30, 30, 8.0),
        (0, 30, 9.0),
        (30, 0, 9.0),
    ]
    assert receipt["stop_reason"] == ("best_remaining_minimum_distance_below_threshold")
    assert receipt["selected_count"] == 4


def test_free_tree_executes_declared_surface_and_restores_root_cache(
    tmp_path: Path,
) -> None:
    rules_path, _ledger_path, candidates_path = _materialize_v2_fixture(tmp_path)
    rules = sut.load_decision_rules(rules_path)
    candidates = sut.load_candidate_rows(candidates_path, rules=rules)
    request = next(
        candidate
        for candidate in candidates
        if candidate.record_type == "free_coordinate_tree_root_request"
    )
    backend = _FakeCacheBackend(
        list(request.prefix_token_ids), vocab_size=rules.model_vocab_size
    )
    root_length = backend.cache_length
    prefill_logits = _deterministic_logits_row(
        request.prefix_token_ids, rules.model_vocab_size
    )
    attestation = sut.build_attestation_context(
        expected_vocab_size=rules.model_vocab_size,
        tokenizer_identity={"fixture": "tokenizer"},
        model_identity={"fixture": "model"},
        rule_digest=rules.rules_digest,
        runtime_receipt_id="fixture-runtime",
    )
    rows, receipt = sut.execute_free_coordinate_tree(
        request=request,
        backend=backend,
        prefill_logits=prefill_logits,
        rules=rules,
        attestation=attestation,
    )
    uncached, _calls = _full_reforward_backend(
        list(request.prefix_token_ids), vocab_size=rules.model_vocab_size
    )
    uncached_rows, uncached_receipt = sut.execute_free_coordinate_tree(
        request=request,
        backend=uncached,
        prefill_logits=uncached.root_logits,
        rules=rules,
        attestation=attestation,
    )
    batched = sut.FullReforwardBackend(
        root_prefix_token_ids=list(request.prefix_token_ids),
        full_reforward=lambda token_ids: _deterministic_logits_row(
            tuple(token_ids), rules.model_vocab_size
        ),
        batched_full_reforward=lambda token_rows: torch.stack(
            [
                _deterministic_logits_row(tuple(row), rules.model_vocab_size)
                for row in token_rows
            ]
        ),
        full_reforward_batch_size=4,
        context_id="fixture-context",
        group_id="fixture-group",
        progress_every_actual_forwards=0,
    )
    batched_rows, batched_receipt = sut.execute_free_coordinate_tree(
        request=request,
        backend=batched,
        prefill_logits=batched.root_logits,
        rules=rules,
        attestation=attestation,
    )
    assert uncached_rows == rows
    assert uncached_receipt == receipt
    assert batched_rows == rows
    assert batched_receipt == receipt
    batch_accounting = batched.accounting()
    assert (
        batch_accounting["physical_model_forward_calls"]
        < batch_accounting["actual_forward_calls"]
    )
    assert backend.cache_length == root_length
    assert receipt["schema_version"] == "sorted_owner_basin_free_tree_execution.v1"
    assert receipt["counts"]["selected_x1_count"] == 9
    assert receipt["counts"]["selected_anchor_count"] == 1
    assert receipt["counts"]["complete_box_count"] == len(rows)
    assert rows
    assert all(row["surface"] == "free_coordinate_tree" for row in rows)
    assert all(
        row["bounded_search_null_semantics"] == "non_evidence_for_absence"
        for row in rows
    )


def _bound_free_surface_row(
    tmp_path: Path,
) -> tuple[sut.CandidateRow, dict[str, Any]]:
    rules_path, _ledger_path, candidates_path = _materialize_v2_fixture(tmp_path)
    rules = sut.load_decision_rules(rules_path)
    request = next(
        candidate
        for candidate in sut.load_candidate_rows(candidates_path, rules=rules)
        if candidate.record_type == "free_coordinate_tree_root_request"
    )
    raw = {
        "candidate_id": "free-tree-box:sha256:"
        + sut.sha256_json(
            {
                "request_id": request.candidate_id,
                "coordinate_bin_values": [0, 0, 1, 1],
            }
        ),
        "coordinate_bin_values": [0, 0, 1, 1],
        "coord_token_ids": [100, 100, 101, 101],
        "raw_model_logprob": {},
        "auxiliary_policy_scores": {},
        "anchor_joint_raw_logprob": -1.0,
        "bounded_search_null_semantics": "non_evidence_for_absence",
        "positive_target_support_semantics": "decision_bearing_blocks_C",
    }
    owner = sut.OwnerLedgerEntry(
        diagnostic_owner_id=request.diagnostic_owner_id,
        status="gt",
        gt_owner_id=request.gt_owner_id,
        image_id=request.image_id,
    )
    [row] = sut.bind_free_surface_score_rows(
        rows=[raw],
        request=request,
        owner_ledger={request.diagnostic_owner_id: owner},
        rules=rules,
    )
    return request, row


def test_free_tree_rows_are_bound_to_free_surface_as_unreviewed_raw_only(
    tmp_path: Path,
) -> None:
    _request, row = _bound_free_surface_row(tmp_path)
    assert row["landscape_surface"] == "canonical_description_free"
    assert row["review_status"] == "unreviewed"
    assert row["proposal_verification"]["pure_core_recomputation"] == "passed"


def _free_execution_receipt(request_id: str, *, complete_box_count: int) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": "sorted_owner_basin_free_tree_execution.v1",
        "surface": "free_coordinate_tree",
        "request_id": request_id,
        "status": "executed",
        "counts": {"complete_box_count": complete_box_count},
    }
    receipt["receipt_sha256"] = sut.sha256_json(receipt)
    return receipt


def _live_surface_receipt(
    request_id: str, *, execution_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "runtime_execution_status": sut.LIVE_SCORING_PENDING_SEAL_STATUS,
        "surfaces": {
            "free_coordinate_tree": {
                "status": "executed",
                "declared_request_ids": [request_id],
                "receipts": [execution_receipt],
                "row_count": 0,
            },
            "restricted_candidate_bank": {
                "status": "executed",
                "row_count": 1,
            },
        },
        "free_surface_status": "executed",
        "restricted_surface_status": "executed",
    }


def test_live_seal_projects_nonempty_bound_free_and_restricted_surface_views_from_authority(
    tmp_path: Path,
) -> None:
    request, free_row = _bound_free_surface_row(tmp_path / "bound")
    restricted_row = {
        "candidate_id": "restricted-fixture",
        "landscape_surface": "restricted_gt_target",
    }
    rows = [free_row, restricted_row]
    scores_path = tmp_path / "landscape-scores.jsonl"
    _write_jsonl(scores_path, rows)
    execution = _free_execution_receipt(
        request.candidate_id, complete_box_count=1
    )
    sealed = sut.seal_execution_receipt(
        _live_surface_receipt(
            request.candidate_id, execution_receipt=execution
        ),
        scores_path=scores_path,
        rows=rows,
    )
    free_authority = sealed["landscape_surface_receipts"][
        "canonical_description_free"
    ]
    free_view = sealed["surfaces"]["free_coordinate_tree"]
    assert free_authority["row_count"] == 1
    assert free_view["row_count"] == free_authority["row_count"]
    assert free_view["status"] == free_authority["status"] == "passed"
    assert sealed["free_surface_status"] == free_authority["status"]
    restricted_authority = sealed["landscape_surface_receipts"][
        "restricted_gt_target"
    ]
    restricted_view = sealed["surfaces"]["restricted_candidate_bank"]
    assert restricted_view["row_count"] == restricted_authority["row_count"] == 1
    assert restricted_view["status"] == restricted_authority["status"] == "passed"
    assert sealed["restricted_surface_status"] == restricted_authority["status"]


def test_live_seal_preserves_bounded_null_free_surface_across_both_receipt_views(
    tmp_path: Path,
) -> None:
    request, _free_row = _bound_free_surface_row(tmp_path / "bound")
    restricted_row = {
        "candidate_id": "restricted-fixture",
        "landscape_surface": "restricted_gt_target",
    }
    rows = [restricted_row]
    scores_path = tmp_path / "landscape-scores.jsonl"
    _write_jsonl(scores_path, rows)
    execution = _free_execution_receipt(
        request.candidate_id, complete_box_count=0
    )
    sealed = sut.seal_execution_receipt(
        _live_surface_receipt(
            request.candidate_id, execution_receipt=execution
        ),
        scores_path=scores_path,
        rows=rows,
    )
    free_authority = sealed["landscape_surface_receipts"][
        "canonical_description_free"
    ]
    free_view = sealed["surfaces"]["free_coordinate_tree"]
    assert free_view["row_count"] == free_authority["row_count"] == 0
    assert (
        free_view["status"]
        == free_authority["status"]
        == "executed_bounded_null_non_evidence"
    )
    assert sealed["free_surface_status"] == free_authority["status"]


def test_context_allowlist_is_exact_repeatable_and_digested(tmp_path: Path) -> None:
    candidates = [
        _candidate_row(_tmp_dir=tmp_path, candidate_id="a", context_id="root"),
        _candidate_row(
            _tmp_dir=tmp_path, candidate_id="b", context_id="natural_stop"
        ),
    ]
    selected, receipt = sut.select_candidate_contexts(candidates, ["root"])
    assert [candidate.candidate_id for candidate in selected] == ["a"]
    assert receipt["mode"] == "explicit_allowlist"
    assert receipt["excluded_context_ids"] == ["natural_stop"]
    assert receipt["selected_diagnostic_owner_ids"] == [
        selected[0].diagnostic_owner_id
    ]
    assert len(receipt["selection_sha256"]) == 64
    with pytest.raises(sut.LandscapeScoringError, match="duplicates"):
        sut.select_candidate_contexts(candidates, ["root", "root"])
    with pytest.raises(sut.LandscapeScoringError, match="absent"):
        sut.select_candidate_contexts(candidates, ["unknown"])


def test_relaxed_cache_admission_scope_requires_one_exact_context_history_group(
    tmp_path: Path,
) -> None:
    one = _candidate_row(_tmp_dir=tmp_path, candidate_id="scope-a")
    one_scope = sut.validate_cache_admission_scope(
        [one], relaxed_selected_logprob_max_abs_diff=2e-4
    )
    assert one_scope["requested_mode"] == sut.RELAXED_CACHE_ADMISSION_MODE
    assert one_scope["group_count"] == 1
    assert one_scope["requirement_satisfied"] is True
    assert len(one_scope["scope_sha256"]) == 64

    different_prefix = _candidate_row(
        _tmp_dir=tmp_path,
        candidate_id="scope-b",
        prefix_token_ids=[100, 101, 103],
    )
    strict_scope = sut.validate_cache_admission_scope(
        [one, different_prefix], relaxed_selected_logprob_max_abs_diff=None
    )
    assert strict_scope["requested_mode"] == sut.STRICT_CACHE_ADMISSION_MODE
    assert strict_scope["group_count"] == 2
    with pytest.raises(
        sut.LandscapeScoringError,
        match="exactly one exact image/context/history group",
    ):
        sut.validate_cache_admission_scope(
            [one, different_prefix],
            relaxed_selected_logprob_max_abs_diff=2e-4,
        )


def test_contract_only_owner_count_uses_selected_diagnostic_owners(
    tmp_path: Path,
) -> None:
    rules_path, ledger_path, candidates_path = _materialize_v2_fixture(tmp_path)
    all_candidates = sut.load_candidate_rows(
        candidates_path,
        rules=sut.load_decision_rules(rules_path),
    )
    selected_context_id = all_candidates[0].context_id
    selected_candidates = [
        candidate
        for candidate in all_candidates
        if candidate.context_id == selected_context_id
    ]
    expected_owners = {
        candidate.diagnostic_owner_id for candidate in selected_candidates
    }
    args = sut.build_parser().parse_args(
        [
            "--owner-context-ledger",
            str(ledger_path),
            "--decision-rules",
            str(rules_path),
            "--candidates",
            str(candidates_path),
            "--include-context-id",
            selected_context_id,
            "--validate-contract-only",
        ]
    )
    receipt = sut.run(args)
    assert receipt["owners_covered"] == len(expected_owners)
    assert receipt["context_selection"]["selected_diagnostic_owner_ids"] == sorted(
        expected_owners
    )
