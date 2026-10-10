"""Read-only resolution of explicit Pi peer-session routes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

MAX_STATE_BYTES = 8 * 1024 * 1024
_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


class BindingFailure(ValueError):
    """A safe validation failure in an explicit peer binding."""


@dataclass(frozen=True)
class ParticipantBinding:
    alias: str | None
    session_id: str
    cwd: Path


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BindingFailure("participant state contains a duplicate JSON key")
        result[key] = value
    return result


def _canonical_session_id(value: object) -> str:
    if not isinstance(value, str):
        raise BindingFailure("participant session_id must be a canonical UUID")
    try:
        normalized = str(UUID(value))
    except (ValueError, AttributeError) as exc:
        raise BindingFailure("participant session_id must be a canonical UUID") from exc
    if normalized != value:
        raise BindingFailure("participant session_id must be a canonical UUID")
    return normalized


def _physical_cwd(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise BindingFailure("participant cwd must be an absolute existing directory")
    path = Path(value)
    if not path.is_absolute():
        raise BindingFailure("participant cwd must be an absolute existing directory")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BindingFailure("participant cwd must be an absolute existing directory") from exc
    if not resolved.is_dir():
        raise BindingFailure("participant cwd must be an absolute existing directory")
    return resolved


def _read_state(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_STATE_BYTES + 1)
    except OSError as exc:
        raise BindingFailure(f"cannot read participant state ({type(exc).__name__})") from None
    if len(raw) > MAX_STATE_BYTES:
        raise BindingFailure("participant state exceeds the size limit")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except BindingFailure:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BindingFailure("participant state is not valid UTF-8 JSON") from None
    if not isinstance(document, dict):
        raise BindingFailure("participant state must be a JSON object")
    return document


def _load_bindings(path: Path) -> dict[str, ParticipantBinding]:
    document = _read_state(path)
    transport = document.get("pi_peer_transport")
    if not isinstance(transport, dict):
        raise BindingFailure("participant state has no pi_peer_transport object")
    if set(transport) != {"version", "participants"}:
        raise BindingFailure("pi_peer_transport has unknown or missing fields")
    version = transport.get("version")
    if type(version) is not int or version != 1:
        raise BindingFailure("pi_peer_transport version must be 1")
    raw_participants = transport.get("participants")
    if not isinstance(raw_participants, dict):
        raise BindingFailure("pi_peer_transport participants must be an object")

    bindings: dict[str, ParticipantBinding] = {}
    seen_sessions: set[str] = set()
    for alias, raw_binding in raw_participants.items():
        if not isinstance(alias, str) or _ALIAS.fullmatch(alias) is None:
            raise BindingFailure("participant aliases must be simple alphanumeric atoms")
        if not isinstance(raw_binding, dict) or set(raw_binding) != {"session_id", "cwd"}:
            raise BindingFailure("participant entries must contain only session_id and cwd")
        session_id = _canonical_session_id(raw_binding.get("session_id"))
        if session_id in seen_sessions:
            raise BindingFailure("participant state contains duplicate session IDs")
        seen_sessions.add(session_id)
        cwd = _physical_cwd(raw_binding.get("cwd"))
        bindings[alias] = ParticipantBinding(alias=alias, session_id=session_id, cwd=cwd)
    return bindings


def resolve_aliases(
    state_path: Path, sender_alias: str, recipient_alias: str
) -> tuple[ParticipantBinding, ParticipantBinding]:
    """Read one selected state snapshot and resolve both aliases exactly once."""

    bindings = _load_bindings(state_path)
    try:
        sender = bindings[sender_alias]
    except KeyError as exc:
        raise BindingFailure("selected sender alias is absent from participant state") from exc
    try:
        recipient = bindings[recipient_alias]
    except KeyError as exc:
        raise BindingFailure("selected recipient alias is absent from participant state") from exc
    if sender.session_id == recipient.session_id:
        raise BindingFailure("sender and recipient must be different sessions")
    return sender, recipient


def resolve_explicit(
    sender_id: str, sender_cwd: Path | str, recipient_id: str, recipient_cwd: Path | str
) -> tuple[ParticipantBinding, ParticipantBinding]:
    """Resolve exact IDs and their explicitly supplied physical directories."""

    sender = ParticipantBinding(
        alias=None, session_id=_canonical_session_id(sender_id), cwd=_physical_cwd(str(sender_cwd))
    )
    recipient = ParticipantBinding(
        alias=None, session_id=_canonical_session_id(recipient_id), cwd=_physical_cwd(str(recipient_cwd))
    )
    if sender.session_id == recipient.session_id:
        raise BindingFailure("sender and recipient must be different sessions")
    return sender, recipient
