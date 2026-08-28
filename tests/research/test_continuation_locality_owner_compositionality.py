from __future__ import annotations

import copy
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from scripts.research import run_exact_prefix_owner_compositionality as owner_probe
from scripts.research.materialize_continuation_locality_owner_compositionality import (
    SCHEMA_VERSION,
    _row_prefixes,
)
from src.inference.backend import token_ids_sha256


def _complete_row(description: list[int]) -> list[int]:
    return [151646, *description, 151647, 151648, 151670, 151671, 151672, 151673, 151649]


def test_description_force_keeps_the_full_multi_token_description() -> None:
    row = _complete_row([41, 42, 43])

    assert owner_probe._description_force(row) == [151646, 41, 42, 43, 151647]


def test_complete_row_prefixes_preserve_every_literal_boundary() -> None:
    first = _complete_row([51])
    second = _complete_row([52, 53])

    assert _row_prefixes(first + second) == [first, first + second]


def test_owner_manifest_validates_composite_owner_ids_and_literal_hashes() -> None:
    prompt = [101, 102]
    prefix = _complete_row([61])
    target_row = _complete_row([62, 63])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "owner_cases": [
            {
                "case_id": "case-a",
                "base_prompt_token_ids": prompt,
                "base_prompt_token_ids_sha256": token_ids_sha256(prompt),
                "prefix_token_ids": prefix,
                "prefix_token_ids_sha256": token_ids_sha256(prefix),
                "target": {
                    "owner_id": "225458:1234",
                    "category": "bottle",
                    "row_token_ids": target_row,
                    "row_token_ids_sha256": token_ids_sha256(target_row),
                },
                "secondary_targets": [],
            }
        ],
    }

    cases = owner_probe._validated_cases(manifest)
    assert cases[0]["target"]["owner_id"] == "225458:1234"

    corrupted = copy.deepcopy(manifest)
    corrupted["owner_cases"][0]["target"]["row_token_ids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="row hash mismatch"):
        owner_probe._validated_cases(corrupted)


def test_owner_state_score_preserves_existing_reduction_from_token_evidence() -> None:
    from scripts.research.run_complete_candidate_row_scoring import _score_candidate
    from scripts.research.run_next_row_likelihood_change import terminal_boundary_score
    from src.inference.hf_backend import HFChosenTokenEvidence

    base_prompt = [101, 102]
    prefix = [201, 202]
    row = _complete_row([41])
    boundary_length = len(base_prompt) + len(prefix)
    vocab_size = 152_000
    logits = torch.full(
        (boundary_length + len(row) - 1, vocab_size),
        -4.0,
        dtype=torch.float32,
    )
    logits[:, 0] = 3.0
    logits[:, 1] = 2.0
    for index, token_id in enumerate(row):
        logits[boundary_length + index - 1, token_id] = 1.0 + index / 10.0
    terminal_id = 31
    logits[boundary_length - 1, terminal_id] = 0.5

    class EvidenceSession:
        def __init__(self) -> None:
            self.calls: list[tuple[int, ...]] = []

        def extend_exact_history(
            self,
            history: Any,
            token_ids: Sequence[int],
        ) -> object:
            values = tuple(int(value) for value in token_ids)
            conditioning = (*history.conditioning_token_ids, *values)
            return SimpleNamespace(
                conditioning_token_ids=conditioning,
                conditioning_token_ids_sha256=token_ids_sha256(conditioning),
            )

        def teacher_forced_evidence(
            self,
            history: Any,
            continuation_token_ids: Sequence[int],
        ) -> tuple[HFChosenTokenEvidence, ...]:
            continuation = tuple(int(value) for value in continuation_token_ids)
            self.calls.append(continuation)
            boundary = len(history.conditioning_token_ids)
            evidence = []
            for index, token_id in enumerate(continuation):
                values = logits[boundary + index - 1].float()
                selected = values[token_id]
                evidence.append(
                    HFChosenTokenEvidence(
                        token_id=token_id,
                        raw_model_logprob=float(
                            torch.log_softmax(values, dim=-1)[token_id].item()
                        ),
                        candidate_vocab_rank=int((values > selected).sum().item()) + 1,
                    )
                )
            return tuple(evidence)

    session = EvidenceSession()
    base_history = SimpleNamespace(
        conditioning_token_ids=tuple(base_prompt),
        conditioning_token_ids_sha256=token_ids_sha256(base_prompt),
    )
    target = {
        "owner_id": "225458:1234",
        "category": "bottle",
        "row_token_ids": row,
    }

    observed = owner_probe._score_state_target(
        session=session,
        base_history=base_history,
        prefix=prefix,
        target=target,
        terminal_id=terminal_id,
    )
    expected_terminal = terminal_boundary_score(
        logits,
        boundary_length=boundary_length,
        row_entry_token_id=row[0],
        terminal_token_id=terminal_id,
    )
    expected_candidate = _score_candidate(
        logits,
        boundary_length=boundary_length,
        row_tokens=row,
        metadata={
            "candidate_id": "owner-225458:1234",
            "owner": "225458:1234",
            "category": "bottle",
            "role": "verified_uncovered_owner",
            "covered": False,
        },
    )

    assert session.calls == [(terminal_id,), tuple(row)]
    assert observed == {
        "prefix_token_ids_sha256": token_ids_sha256(prefix),
        "actual_prompt_plus_prefix_token_ids_sha256": token_ids_sha256(
            [*base_prompt, *prefix]
        ),
        "terminal_boundary": expected_terminal,
        "candidate_score": expected_candidate,
    }
