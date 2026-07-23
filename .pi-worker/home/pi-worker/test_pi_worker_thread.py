"""Focused contracts for the reusable Pi RPC thread supervisor."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import uuid

import pytest

from pi_worker_thread import (
    PiThreadError,
    PiThreadRuntime,
    PiThreadSpec,
    build_pi_rpc_command,
    iter_lf_records,
    translate_command,
)


def _thread(**overrides: object) -> PiThreadSpec:
    value: dict[str, object] = {
        "thread_id": "pi-thread-test",
        "provider": "openai-codex",
        "model": "gpt-5.6-luna",
        "reasoning": "medium",
        "auto_compaction": True,
        "auto_retry": True,
        "event_stream": "lifecycle",
    }
    value.update(overrides)
    return PiThreadSpec.from_mapping(value)


def _runtime(tmp_path: Path) -> PiThreadRuntime:
    for relative in ("workspace", "opt/node/bin", "opt/pi"):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    (tmp_path / "opt/node/bin/node").write_text("node", encoding="utf-8")
    (tmp_path / "opt/pi/cli.js").write_text("pi", encoding="utf-8")
    return PiThreadRuntime.from_mapping(
        {
            "sandbox_root": str(tmp_path),
            "workspace": "/workspace",
            "node_path": "/opt/node/bin/node",
            "pi_cli_path": "/opt/pi/cli.js",
            "home": "/state/home",
            "agent_dir": "/state/agent",
            "session_dir": "/state/sessions",
            "proxy_url": "http://127.0.0.1:9090",
            "userspec": "65534:65534",
        }
    )


def test_thread_contract_rejects_max_reasoning() -> None:
    with pytest.raises(PiThreadError, match="max is intentionally unsupported"):
        _thread(reasoning="max")


def test_rpc_command_is_persistent_and_contains_no_codex_dependency(tmp_path: Path) -> None:
    session_id = str(uuid.uuid4())
    command = build_pi_rpc_command(_thread(), _runtime(tmp_path), session_id=session_id)
    rendered = " ".join(command)

    assert command[0] == "/usr/sbin/chroot"
    assert "/bin/bash" in command
    assert "--mode rpc" in rendered
    assert f"--session-id {session_id}" in rendered
    assert "--no-session" not in rendered
    assert "codex" not in rendered.lower().replace("openai-codex", "")


def test_stable_commands_translate_to_pi_rpc_without_exposing_session_switch() -> None:
    assert translate_command({"request_id": "turn-1", "type": "prompt", "message": "inspect"}) == {
        "id": "turn-1",
        "type": "prompt",
        "message": "inspect",
    }
    assert translate_command({"id": "stats-1", "type": "get_stats"}) == {
        "id": "stats-1",
        "type": "get_session_stats",
    }
    assert translate_command({"type": "follow_up", "message": "then summarize"}) == {
        "type": "follow_up",
        "message": "then summarize",
    }
    with pytest.raises(PiThreadError, match="unsupported command"):
        translate_command({"type": "switch_session", "sessionPath": "/other"})


def test_request_id_retains_id_alias_and_rejects_ambiguity() -> None:
    assert translate_command({"id": "legacy-1", "type": "get_state"}) == {
        "id": "legacy-1",
        "type": "get_state",
    }
    with pytest.raises(PiThreadError, match="must match"):
        translate_command(
            {
                "request_id": "canonical-1",
                "id": "different-1",
                "type": "get_state",
            }
        )


def test_prompt_requires_explicit_streaming_behavior_when_requested() -> None:
    assert translate_command(
        {
            "type": "prompt",
            "message": "change direction",
            "streaming_behavior": "steer",
        }
    )["streamingBehavior"] == "steer"
    with pytest.raises(PiThreadError, match="steer or followUp"):
        translate_command(
            {
                "type": "prompt",
                "message": "change direction",
                "streaming_behavior": "immediate",
            }
        )


def test_strict_lf_framing_preserves_unicode_line_separators() -> None:
    stream = BytesIO(b'{"message":"left\xe2\x80\xa8right"}\n{"type":"status"}\r\n')
    assert list(iter_lf_records(stream, chunk_size=5)) == [
        b'{"message":"left\xe2\x80\xa8right"}',
        b'{"type":"status"}',
    ]


def test_runtime_rejects_nonlocal_or_credentialed_proxy(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    assert runtime.proxy_url == "http://127.0.0.1:9090"
    with pytest.raises(PiThreadError, match="explicit local proxy"):
        PiThreadRuntime.from_mapping(
            {
                "sandbox_root": str(tmp_path),
                "workspace": "/workspace",
                "node_path": "/opt/node/bin/node",
                "pi_cli_path": "/opt/pi/cli.js",
                "home": "/state/home",
                "agent_dir": "/state/agent",
                "session_dir": "/state/sessions",
                "proxy_url": "https://proxy.example:9090",
            }
        )
