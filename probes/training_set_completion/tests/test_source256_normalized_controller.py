from __future__ import annotations

import json

import pytest

from probes.training_set_completion import source256_normalized_controller as controller
from probes.training_set_completion import training


def test_controller_argv_is_a_durable_explicit_packet_release_invocation(tmp_path) -> None:
    command = controller.controller_argv(
        packet_path=tmp_path / "packet.json",
        release_path=tmp_path / "release.json",
        output=tmp_path / "runtime",
    )

    assert command[command.index("--packet") + 1].endswith("packet.json")
    assert command[command.index("--release") + 1].endswith("release.json")
    assert command[command.index("--output") + 1].endswith("runtime")


def test_release_rejects_a_packet_other_than_the_exact_held_packet(tmp_path) -> None:
    packet = tmp_path / "packet.json"
    packet.write_text("{}")
    qualification = tmp_path / "qualification.json"
    qualification.write_text("{}")
    training_release = tmp_path / "training-release.json"
    training_release.write_text("{}")
    release = tmp_path / "release.json"
    release.write_text(json.dumps({
        "schema": f"{controller.SCHEMA}.main_release",
        "status": "released",
        "unit_id": "2026-09-16-source256-completion-ce-normalization",
        "packet": {**training.binding(packet), "sha256": "0" * 64},
        "actual_entry_qualification": training.binding(qualification),
        "training_release": training.binding(training_release),
    }))

    with pytest.raises(ValueError, match="explicit accepted"):
        controller._validate_release(
            release_path=release,
            packet_path=packet,
            packet={"sources": {"actual_entry_qualification": training.binding(qualification)}},
        )


def test_release_requires_a_separate_bound_training_release(tmp_path) -> None:
    packet = tmp_path / "packet.json"
    packet.write_text("{}")
    qualification = tmp_path / "qualification.json"
    qualification.write_text("{}")
    release = tmp_path / "release.json"
    release.write_text(json.dumps({
        "schema": f"{controller.SCHEMA}.main_release",
        "status": "released",
        "packet": training.binding(packet),
        "actual_entry_qualification": training.binding(qualification),
    }))

    with pytest.raises(ValueError, match="training release"):
        controller._validate_release(
            release_path=release,
            packet_path=packet,
            packet={"sources": {"actual_entry_qualification": training.binding(qualification)}},
        )
