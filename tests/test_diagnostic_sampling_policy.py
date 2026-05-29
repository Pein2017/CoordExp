from __future__ import annotations

from src.training.observability import ObservabilityService


class _LargeRepr:
    """Object with an intentionally oversized representation for diagnostics."""

    def __repr__(self) -> str:
        return "R" * 64


def test_diagnostic_profile_off_emits_no_payloads() -> None:
    service = ObservabilityService(diagnostic_profile="off")

    assert (
        service.diagnostic_event(
            "rollout_sample",
            {
                "loss": 1.0,
                "tokens": [1, 2, 3],
            },
        )
        is None
    )


def test_standard_diagnostic_profile_keeps_small_bounded_simple_summary() -> None:
    service = ObservabilityService(
        diagnostic_profile="standard",
        max_diagnostic_entries=2,
        max_diagnostic_items=3,
    )

    event = service.diagnostic_event(
        "rollout_sample",
        {
            "loss": 1.0,
            "ok": True,
            "tokens": [1, 2, 3, 4, 5],
            "nested": {"x": 1},
        },
    )

    assert event is not None
    assert event.profile == "standard"
    assert event.payload == {"loss": 1.0, "ok": True}
    assert event.truncated is True


def test_debug_diagnostic_profile_keeps_richer_but_bounded_payload() -> None:
    service = ObservabilityService(
        diagnostic_profile="debug",
        max_diagnostic_entries=3,
        max_diagnostic_items=2,
    )

    event = service.diagnostic_event(
        "rollout_sample",
        {
            "loss": 1.0,
            "tokens": [1, 2, 3, 4],
            "nested": {"a": 1, "b": [2, 3, 4], "c": 5},
            "extra": "dropped",
        },
    )

    assert event is not None
    assert event.profile == "debug"
    assert event.payload == {
        "loss": 1.0,
        "tokens": [1, 2],
        "nested": {"a": 1, "b": [2, 3]},
    }
    assert event.truncated is True


def test_debug_diagnostic_profile_has_global_depth_bound() -> None:
    service = ObservabilityService(
        diagnostic_profile="debug",
        max_diagnostic_entries=1,
        max_diagnostic_items=1,
        max_diagnostic_depth=2,
    )

    event = service.diagnostic_event(
        "deep_payload",
        {"root": {"child": {"grandchild": {"too_deep": [1, 2, 3]}}}},
    )

    assert event is not None
    assert event.payload == {"root": {"child": "<truncated>"}}
    assert event.truncated is True


def test_debug_diagnostic_profile_marks_cycles() -> None:
    service = ObservabilityService(
        diagnostic_profile="debug",
        max_diagnostic_entries=1,
        max_diagnostic_items=2,
        max_diagnostic_depth=4,
    )
    cyclic: list[object] = []
    cyclic.append(cyclic)

    event = service.diagnostic_event("cyclic_payload", {"cycle": cyclic})

    assert event is not None
    assert event.payload == {"cycle": ["<cycle>"]}
    assert event.truncated is True


def test_standard_diagnostic_profile_truncates_large_scalar_strings() -> None:
    service = ObservabilityService(
        diagnostic_profile="standard",
        max_diagnostic_scalar_chars=8,
    )

    event = service.diagnostic_event("large_text", {"prompt": "0123456789abcdef"})

    assert event is not None
    assert event.payload == {"prompt": "01234567..."}
    assert event.truncated is True


def test_debug_diagnostic_profile_truncates_large_scalar_strings_and_reprs() -> None:
    service = ObservabilityService(
        diagnostic_profile="debug",
        max_diagnostic_scalar_chars=8,
    )

    event = service.diagnostic_event(
        "large_debug",
        {
            "text": "abcdefghijklmnop",
            "object": _LargeRepr(),
        },
    )

    assert event is not None
    assert event.payload == {
        "text": "abcdefgh...",
        "object": "RRRRRRRR...",
    }
    assert event.truncated is True
