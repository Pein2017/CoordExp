"""Frozen owner-wise witness and qualification dose-mechanics semantics.

| # | Frozen semantic | Owner | Wrong alternative | Test |
| - | --------------- | ----- | ----------------- | ---- |
| 1 | One constraint per (trusted owner, Source RP membership) | `WitnessMeasurement.witness_sites` | one witness per owner across both surfaces | `test_owner_on_both_surfaces_yields_two_constraints` |
| 2 | Legacy-M is audit-only and carries no Jacobian | `freeze_witness_bank` | legacy-M enters the constraints | `test_legacy_m_owner_stays_audit_only_without_jacobian` |
| 3 | Eligible tokens are exactly `[token_start, token_end)` | site selection | terminal/inter-row tokens leak in | `test_only_the_half_open_parser_span_is_eligible` |
| 4 | Weakest margin, ties by index then token id | site selection | first or last span token | `test_minimum_margin_ties_break_by_index_then_token_id` |
| 5 | Sign-aware RP, no temperature, full vocabulary | `processed_logits` | temperature division or top-k truncation | `test_processed_logits_are_sign_aware_without_temperature` |
| 6 | Margin must be finite and Source-greedy | site selection | a negative margin is admitted | `test_non_greedy_or_non_finite_margin_fails_closed` |
| 7 | Frozen `(y, v*)` float64 Jacobian over the frozen layout | `freeze_witness_bank` | re-picking the competitor per call | `test_frozen_jacobian_matches_frozen_competitor_pair` |
| 8 | Realized probe re-maxes the competitor | `margin_values` | reusing the frozen competitor | `test_realized_probe_remaxes_the_competitor` |
| 9 | `jvp_fd_max_abs_error = max abs(J.delta - fd)` | `jvp_finite_difference_error` | mean, or a small-step finite difference | `test_jvp_finite_difference_error_is_the_max_abs_gap` |
| 10 | Dose sites are the deduped compiler/witness union without legacy-M | `dose_sites` | witness-only or legacy-M included | `test_dose_sites_are_the_deduped_union_without_legacy_m` |
| 11 | Even-cardinality median is the mean of the middle two | `median` | lower middle value | `test_even_cardinality_median_is_the_middle_mean` |
| 12 | Teacher-forced greedy change over both surfaces, ties smallest id | `teacher_forced_greedy_change_count` | free-running decode comparison | `test_teacher_forced_greedy_change_count_uses_both_surfaces` |
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
import torch

from scripts.research import human13_rp_crossover_witness as witness
from scripts.research.human13_adamw_proposal_preservation import (
    LEGACY_M_OWNER_CLASS,
    TRUSTED_OWNER_CLASS,
    WITNESS_FIRST_ORDER_TOLERANCE,
    WitnessBinding,
    jacobian_sha256,
)


VOCAB = 6
PROMPT = (0, 1)


def _binding() -> WitnessBinding:
    return WitnessBinding(
        unit_id="2026-08-14-human13-k-trajectory-rp-crossover-screen",
        source_checkpoint_sha256="a" * 64,
        manifest_sha256="b" * 64,
        frozen_before_acquisition=True,
    )


class FakeSurface:
    """A differentiable CPU stand-in for the HF fp32/SDPA batch-one surface."""

    def __init__(self, rows: dict[tuple[int, float, int], Sequence[float]]) -> None:
        self._rows = {
            key: torch.tensor(value, dtype=torch.float32) for key, value in rows.items()
        }
        self.scale = torch.nn.Parameter(torch.ones(VOCAB, dtype=torch.float32))
        self.shift = torch.nn.Parameter(torch.zeros(VOCAB, dtype=torch.float32))

    def named_trainable_parameters(self) -> Sequence[tuple[str, torch.nn.Parameter]]:
        return (("dora.scale", self.scale), ("dora.shift", self.shift))

    def raw_logit_rows(
        self, decode: witness.SealedSourceDecode, token_indices: Sequence[int]
    ) -> torch.Tensor:
        selected = []
        for index in token_indices:
            key = (decode.image_id, decode.repetition_penalty, int(index))
            selected.append(self._rows[key] * self.scale + self.shift)
        return torch.stack(selected)


def _decode(
    *,
    image_id: int = 1584,
    rp: float = 1.0,
    generated: Sequence[int] = (2, 3, 4),
    rows: Sequence[tuple[str, str, int, int]] = (
        ("owner-a", TRUSTED_OWNER_CLASS, 0, 2),
    ),
    compiler_token_indices: Sequence[int] = (),
) -> witness.SealedSourceDecode:
    return witness.SealedSourceDecode(
        image_id=image_id,
        repetition_penalty=rp,
        prompt_token_ids=PROMPT,
        generated_token_ids=tuple(generated),
        owner_rows=tuple(
            witness.SealedOwnerRow(
                owner_id=owner_id,
                owner_class=owner_class,
                token_start=start,
                token_end=end,
            )
            for owner_id, owner_class, start, end in rows
        ),
        compiler_token_indices=tuple(compiler_token_indices),
    )


def _flat_rows(values: dict[tuple[int, float, int], Sequence[float]]) -> FakeSurface:
    return FakeSurface(values)


def _greedy_rows(
    image_id: int, rp: float, generated: Sequence[int], margins: Sequence[float]
):
    """Rows whose argmax is the sealed token with the requested margin."""

    rows: dict[tuple[int, float, int], Sequence[float]] = {}
    for index, (token, margin) in enumerate(zip(generated, margins, strict=True)):
        row = [0.0] * VOCAB
        row[token] = margin
        rows[(image_id, rp, index)] = row
    return rows


def test_owner_on_both_surfaces_yields_two_constraints() -> None:
    generated = (2, 3, 4)
    rows = {
        **_greedy_rows(1584, 1.0, generated, (1.5, 0.5, 2.0)),
        **_greedy_rows(1584, 1.10, generated, (2.5, 0.75, 3.0)),
    }
    surface = _flat_rows(rows)
    measurement = witness.WitnessMeasurement(
        decodes=(
            _decode(rp=1.0, generated=generated),
            _decode(rp=1.10, generated=generated),
        ),
        surface=surface,
    )

    keys = [site.canonical_key for site in measurement.witness_sites]
    assert keys == [
        "u_intersect_s_1.0|1584|owner-a",
        "u_intersect_s_1.10|1584|owner-a",
    ]


def test_legacy_m_owner_stays_audit_only_without_jacobian() -> None:
    generated = (2, 3, 4)
    surface = _flat_rows(_greedy_rows(1584, 1.0, generated, (1.5, 0.5, 2.0)))
    measurement = witness.WitnessMeasurement(
        decodes=(
            _decode(
                generated=generated,
                rows=(
                    ("owner-a", TRUSTED_OWNER_CLASS, 0, 2),
                    ("owner-m", LEGACY_M_OWNER_CLASS, 2, 3),
                ),
            ),
        ),
        surface=surface,
    )
    bank = measurement.freeze_witness_bank(binding=_binding())

    assert [item.owner_id for item in bank.constraints] == ["owner-a"]
    assert [item.owner_id for item in bank.audit_only] == ["owner-m"]
    assert bank.audit_only[0].jacobian_sha256 is None


def test_only_the_half_open_parser_span_is_eligible() -> None:
    generated = (2, 3, 4)
    # index 2 has the smallest margin but lies outside [0, 2)
    surface = _flat_rows(_greedy_rows(1584, 1.0, generated, (1.5, 0.5, 0.1)))
    measurement = witness.WitnessMeasurement(
        decodes=(_decode(generated=generated),), surface=surface
    )

    site = measurement.witness_sites[0]
    assert site.token_index == 1
    assert site.chosen_token_id == 3


def test_minimum_margin_ties_break_by_index_then_token_id() -> None:
    generated = (2, 3, 4)
    rows = _greedy_rows(1584, 1.0, generated, (0.5, 0.5, 2.0))
    # index 0 competitor tie between token 1 and token 5
    rows[(1584, 1.0, 0)] = [0.0, 0.0, 0.5, -1.0, -1.0, 0.0]
    surface = _flat_rows(rows)
    measurement = witness.WitnessMeasurement(
        decodes=(_decode(generated=generated),), surface=surface
    )

    site = measurement.witness_sites[0]
    assert (site.token_index, site.chosen_token_id, site.competitor_token_id) == (
        0,
        2,
        0,
    )


def test_processed_logits_are_sign_aware_without_temperature() -> None:
    raw = torch.tensor([-2.0, 1.0, 4.0, 0.0], dtype=torch.float32)
    processed = witness.processed_logits(
        raw, history_token_ids=(0, 1, 1), repetition_penalty=1.10
    )

    assert processed.shape == raw.shape
    assert math.isclose(float(processed[0]), -2.0 * 1.10, rel_tol=1e-6)
    assert math.isclose(float(processed[1]), 1.0 / 1.10, rel_tol=1e-6)
    # untouched tokens keep their exact raw value: no temperature, no normalization
    assert float(processed[2]) == 4.0
    assert float(processed[3]) == 0.0


def test_non_greedy_or_non_finite_margin_fails_closed() -> None:
    generated = (2, 3, 4)
    rows = _greedy_rows(1584, 1.0, generated, (1.5, 0.5, 2.0))
    rows[(1584, 1.0, 1)] = [
        0.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        0.0,
    ]  # sealed token is not the argmax
    surface = _flat_rows(rows)

    with pytest.raises(witness.WitnessMeasurementError, match="Source-greedy"):
        witness.WitnessMeasurement(
            decodes=(_decode(generated=generated),), surface=surface
        ).witness_sites


def test_frozen_jacobian_matches_frozen_competitor_pair() -> None:
    generated = (2, 3, 4)
    surface = _flat_rows(_greedy_rows(1584, 1.0, generated, (1.5, 0.5, 2.0)))
    measurement = witness.WitnessMeasurement(
        decodes=(_decode(generated=generated),), surface=surface
    )
    site = measurement.witness_sites[0]
    bank = measurement.freeze_witness_bank(binding=_binding())

    assert bank.layout.total_numel == 2 * VOCAB
    (_index, frozen, jacobian) = next(iter(bank.stream_constraints()))
    assert frozen.canonical_key == site.canonical_key
    assert jacobian.dtype == torch.float64
    assert frozen.jacobian_sha256 == jacobian_sha256(jacobian)
    # d(z[y] - z[v*])/d shift is +1 at y and -1 at v*; scale rows carry the raw values
    shift = jacobian[VOCAB:]
    assert float(shift[site.chosen_token_id]) == pytest.approx(1.0)
    assert float(shift[site.competitor_token_id]) == pytest.approx(-1.0)
    assert float(shift.abs().sum()) == pytest.approx(2.0)


def test_realized_probe_remaxes_the_competitor() -> None:
    generated = (2, 3, 4)
    surface = _flat_rows(_greedy_rows(1584, 1.0, generated, (1.5, 0.5, 2.0)))
    measurement = witness.WitnessMeasurement(
        decodes=(_decode(generated=generated),), surface=surface
    )
    site = measurement.witness_sites[0]
    assert measurement.margin_values()[site.canonical_key] == pytest.approx(0.5)

    with torch.no_grad():
        # lift a token that was not the frozen competitor above it
        surface.shift[(site.competitor_token_id + 1) % VOCAB] += 1.0
    remaxed = measurement.margin_values()[site.canonical_key]
    assert remaxed == pytest.approx(-0.5)


def test_jvp_finite_difference_error_is_the_max_abs_gap() -> None:
    generated = (2, 3, 4)
    surface = _flat_rows(_greedy_rows(1584, 1.0, generated, (1.5, 0.5, 2.0)))
    measurement = witness.WitnessMeasurement(
        decodes=(_decode(generated=generated),), surface=surface
    )
    bank = measurement.freeze_witness_bank(binding=_binding())
    site = measurement.witness_sites[0]

    delta = torch.zeros(bank.layout.total_numel, dtype=torch.float64)
    delta[VOCAB + site.chosen_token_id] = 0.25
    realized = {site.canonical_key: site.margin_value + 0.25 + 3e-5}

    error = witness.jvp_finite_difference_error(
        bank, applied_delta=delta, realized=realized
    )
    assert error == pytest.approx(3e-5, abs=1e-9)
    assert witness.JVP_FD_TOLERANCE == WITNESS_FIRST_ORDER_TOLERANCE


def test_dose_sites_are_the_deduped_union_without_legacy_m() -> None:
    generated = (2, 3, 4)
    rows = {
        **_greedy_rows(1584, 1.0, generated, (1.5, 0.5, 2.0)),
        **_greedy_rows(1584, 1.10, generated, (2.5, 0.75, 3.0)),
    }
    surface = _flat_rows(rows)
    measurement = witness.WitnessMeasurement(
        decodes=(
            _decode(
                rp=1.0,
                generated=generated,
                rows=(
                    ("owner-a", TRUSTED_OWNER_CLASS, 0, 2),
                    ("owner-m", LEGACY_M_OWNER_CLASS, 2, 3),
                ),
                # index 1 duplicates the witness site; index 2 is compiler-only
                compiler_token_indices=(1, 2),
            ),
            _decode(rp=1.10, generated=generated),
        ),
        surface=surface,
    )

    assert measurement.dose_sites == (
        (1584, 1.0, 1),
        (1584, 1.0, 2),
        (1584, 1.10, 1),
    )
    assert set(measurement.dose_site_margins()) == {
        "1584|1.0|1",
        "1584|1.0|2",
        "1584|1.10|1",
    }


def test_even_cardinality_median_is_the_middle_mean() -> None:
    assert witness.median((1.0, 2.0, 5.0, 6.0)) == pytest.approx(3.5)
    assert witness.median((4.0, 1.0, 3.0)) == pytest.approx(3.0)
    with pytest.raises(witness.WitnessMeasurementError):
        witness.median(())


def test_teacher_forced_greedy_change_count_uses_both_surfaces() -> None:
    generated = (2, 3, 4)
    rows = {
        **_greedy_rows(1584, 1.0, generated, (1.5, 0.5, 2.0)),
        **_greedy_rows(1584, 1.10, generated, (2.5, 0.75, 3.0)),
    }
    surface = _flat_rows(rows)
    measurement = witness.WitnessMeasurement(
        decodes=(
            _decode(rp=1.0, generated=generated),
            _decode(rp=1.10, generated=generated),
        ),
        surface=surface,
    )
    assert measurement.teacher_forced_greedy_change_count() == 0

    with torch.no_grad():
        surface.shift[5] += 10.0
    assert measurement.teacher_forced_greedy_change_count() == 6
