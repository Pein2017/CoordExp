from __future__ import annotations

import pytest

from scripts.research.run_sampling_calibration import execute_frozen_protocol
from src.analysis.spatial_scope_history.calibration import CALIBRATION_TEMPERATURES


def _requests() -> dict[float, tuple[tuple[str, ...], ...]]:
    return {
        temperature: tuple(
            tuple(
                f"{temperature:.1f}-{batch_index}-{request_index}"
                for request_index in range(4)
            )
            for batch_index in range(12)
        )
        for temperature in CALIBRATION_TEMPERATURES
    }


def test_protocol_uses_physical_batch_four_and_stops_after_first_pass() -> None:
    calls: list[tuple[float, str, int, tuple[str, ...]]] = []

    def execute(
        temperature: float,
        panel_kind: str,
        batch_index: int,
        requests: tuple[str, ...],
    ) -> tuple[str, ...]:
        calls.append((temperature, panel_kind, batch_index, requests))
        return requests

    terminal = execute_frozen_protocol(
        requests_by_temperature=_requests(),
        execute_batch=execute,
        initial_gate=lambda temperature, rows: temperature == 0.4,
    )

    assert len(terminal) == 192
    assert len(calls) == 48
    assert [call[0] for call in calls[:12]] == [0.2] * 12
    assert [call[0] for call in calls[12:]] == [0.4] * 36
    assert all(call[1] == "initial" for call in calls[:24])
    assert [call[1] for call in calls[24:36]] == ["exact_replay"] * 12
    assert [call[1] for call in calls[36:48]] == ["reversed_order"] * 12
    assert all(call[0] != 0.6 for call in calls)
    for _, panel_kind, _, requests in calls[24:]:
        if panel_kind == "reversed_order":
            assert requests[0].endswith("3")


def test_protocol_fails_closed_when_no_temperature_passes() -> None:
    calls: list[tuple[float, str, int]] = []

    def execute(
        temperature: float,
        panel_kind: str,
        batch_index: int,
        requests: tuple[str, ...],
    ) -> tuple[str, ...]:
        calls.append((temperature, panel_kind, batch_index))
        return requests

    with pytest.raises(RuntimeError, match="no sampled calibration temperature"):
        execute_frozen_protocol(
            requests_by_temperature=_requests(),
            execute_batch=execute,
            initial_gate=lambda temperature, rows: False,
        )
    assert len(calls) == 36
    assert all(panel_kind == "initial" for _, panel_kind, _ in calls)


def test_protocol_uses_exact_240_call_ceiling_when_last_candidate_passes() -> None:
    terminal = execute_frozen_protocol(
        requests_by_temperature=_requests(),
        execute_batch=lambda temperature, panel_kind, batch_index, requests: requests,
        initial_gate=lambda temperature, rows: temperature == 0.6,
    )

    assert len(terminal) == 240
