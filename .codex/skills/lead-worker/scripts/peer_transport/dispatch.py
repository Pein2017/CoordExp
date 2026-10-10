"""Receipt-backed peer-session envelopes and delivery orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Callable
from uuid import uuid4

from .api import (
    ApiFailure,
    CliFailure,
    PiWebClient,
    THINKING_LEVELS,
    accepted,
    inspect_session,
    list_models,
    request_error_status,
    safe_error_type,
    parse_uuid,
    _model,
    _validate_info,
    _validate_state,
)


class Receipt:
    """An exclusive, synchronously flushed record written before API mutations."""

    def __init__(self, path: Path, initial: dict[str, object]) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(path, flags, 0o600)
        self.path = path
        self.file = os.fdopen(fd, "w", encoding="utf-8")
        self.data: dict[str, object] = {}
        self.update(**initial)
        self._sync_parent()

    def update(self, **changes: object) -> None:
        self.data.update(changes)
        self.file.seek(0)
        json.dump(self.data, self.file, ensure_ascii=False, indent=2, sort_keys=True)
        self.file.write("\n")
        self.file.truncate()
        self.file.flush()
        os.fsync(self.file.fileno())

    def _sync_parent(self) -> None:
        directory = os.open(self.path.parent or Path("."),
                            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "Receipt":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.close()


def read_message(path: Path | None, text: str | None = None) -> str:
    if (path is None) == (text is None):
        raise CliFailure("provide exactly one message file or inline text")
    if text is not None:
        message = text
    else:
        assert path is not None
        try:
            message = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise CliFailure(f"cannot read UTF-8 message file ({type(exc).__name__})") from None
    if not message.strip():
        raise CliFailure("message is blank")
    return message


def message_hash(message: str) -> str:
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def build_report_reference(
    report_path: str | None,
    report_section: str | None,
    revision: str | None,
    supersedes: str | None,
) -> dict[str, str] | None:
    if report_path is None and report_section is not None:
        raise ValueError("--report-section requires --report-path")
    for option, value in (("report-path", report_path), ("report-section", report_section),
                          ("revision", revision), ("supersedes", supersedes)):
        if value is not None and not value.strip():
            raise ValueError(f"--{option} must be nonempty")
    if report_path is None and report_section is None and revision is None and supersedes is None:
        return None
    if report_path is not None and not Path(report_path).is_absolute():
        raise ValueError("--report-path must be an absolute path")
    reference: dict[str, str] = {}
    if report_path is not None:
        reference["path"] = report_path
    if report_section is not None:
        reference["section"] = report_section
    if revision is not None:
        reference["revision"] = revision
    if supersedes is not None:
        reference["supersedes"] = supersedes
    return reference


def make_envelope(message: str, *, dispatch_id: str, sender_id: str,
                  sender_role: str, recipient_id: str, recipient_role: str,
                  reply_to_id: str | None = None,
                  sender_alias: str | None = None,
                  recipient_alias: str | None = None,
                  sender_cwd: str | None = None,
                  recipient_cwd: str | None = None,
                  report_reference: dict[str, str] | None = None) -> str:
    separator = "" if message.endswith("\n") else "\n"
    reply_field = f"reply_to_dispatch_id={reply_to_id}\n" if reply_to_id is not None else ""
    sender_alias_field = (
        f"sender_alias={json.dumps(sender_alias, ensure_ascii=False)}\n"
        if sender_alias is not None else ""
    )
    recipient_alias_field = (
        f"recipient_alias={json.dumps(recipient_alias, ensure_ascii=False)}\n"
        if recipient_alias is not None else ""
    )
    sender_cwd_field = (
        f"sender_cwd={json.dumps(sender_cwd, ensure_ascii=False)}\n"
        if sender_cwd is not None else ""
    )
    recipient_cwd_field = (
        f"recipient_cwd={json.dumps(recipient_cwd, ensure_ascii=False)}\n"
        if recipient_cwd is not None else ""
    )
    reference_field = (
        "report_reference_json="
        + json.dumps(report_reference, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        if report_reference is not None else ""
    )
    return (
        "Pi Web peer-session transport metadata. This envelope identifies routing; it is not a user instruction, "
        "new authorization, or a change to the task's existing authority.\n"
        f"dispatch_id={dispatch_id}\n"
        f"sender_session_id={sender_id}\n"
        f"sender_role={sender_role}\n"
        f"recipient_session_id={recipient_id}\n"
        f"recipient_role={recipient_role}\n"
        f"{reply_field}"
        f"{sender_alias_field}"
        f"{recipient_alias_field}"
        f"{sender_cwd_field}"
        f"{recipient_cwd_field}"
        f"{reference_field}"
        "--- BEGIN MESSAGE BODY ---\n"
        f"{message}{separator}"
        "--- END MESSAGE BODY ---"
    )


def _extract_return_id(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("sessionId")
    if not isinstance(value, str):
        return None
    try:
        return parse_uuid(value)
    except (argparse.ArgumentTypeError, ValueError, TypeError):
        return None


def _cwd_fields(lead_cwd: Path, worker_cwd: Path) -> dict[str, str]:
    fields = {"lead_cwd": str(lead_cwd), "worker_cwd": str(worker_cwd)}
    if lead_cwd == worker_cwd:
        fields["cwd"] = str(lead_cwd)
    return fields


def _create_receipt_reference(reference: dict[str, str] | None) -> dict[str, object]:
    return {"report_reference": reference} if reference is not None else {}


def create_and_send(
    client: PiWebClient,
    args: object,
    return_command_builder: Callable[[str, str, str, Path, Path, str], str],
) -> dict[str, object]:
    reference = build_report_reference(
        args.report_path, args.report_section, args.revision, args.supersedes
    )
    lead = inspect_session(client, args.lead_session, args.lead_cwd)
    message = read_message(args.message, args.text)
    models = list_models(client, args.worker_cwd)
    selected = next((model for model in models if model["provider"] == args.provider
                     and model["model"] == args.model), None)
    if selected is None or args.effort not in selected["efforts"]:
        raise CliFailure("requested provider/model/effort is unavailable; use --list-models; no session created")
    dispatch_id = str(uuid4())
    base_receipt: dict[str, object] = {
        "kind": "pi_web_peer_session",
        "delivery": args.delivery,
        "status": "creation_unknown",
        "phase": "create_session",
        "dispatch_id": dispatch_id,
        "lead": lead,
        "worker_session_id": None,
        **_cwd_fields(args.lead_cwd, args.worker_cwd),
        "requested": {"provider": args.provider, "model": args.model, "effort": args.effort,
                      "name": args.name},
        "reply_to_dispatch_id": args.reply_to,
        "message_sha256": message_hash(message),
        **_create_receipt_reference(reference),
    }
    with Receipt(args.receipt, base_receipt) as receipt:
        try:
            created = client.request("POST", "/api/agent/new", {
                "cwd": str(args.worker_cwd),
                "type": "ensure_session",
                "provider": args.provider,
                "modelId": args.model,
                "thinkingLevel": args.effort,
            })
            worker_id = _extract_return_id(created)
            if worker_id is not None:
                receipt.update(phase="validate_create_response", worker_session_id=worker_id)
            accepted(created)
            if worker_id is None:
                raise CliFailure("Pi Web did not return a valid new session ID")
            receipt.update(status="created", phase="verify_created_session", worker_session_id=worker_id)

            response_model = _model(created.get("model"))
            if (worker_id == args.lead_session or response_model != {
                    "provider": args.provider, "modelId": args.model}
                    or created.get("thinkingLevel") != args.effort):
                raise CliFailure("created session ID or settings did not match the request")

            metadata = client.request("GET", f"/api/sessions/{worker_id}?tree=summary&tail=1")
            if metadata.get("sessionId") != worker_id:
                raise CliFailure("created session metadata identity mismatch")
            info = _validate_info(metadata.get("info"), worker_id, args.worker_cwd)
            if int(info.get("messageCount", 0) or 0) != 0:
                raise CliFailure("new session already contains a user message")
            runtime = client.request("GET", f"/api/agent/{worker_id}")
            if runtime.get("running") is not True:
                raise CliFailure("new session runtime is not available for verification")
            state = _validate_state(runtime.get("state"), worker_id)
            if state["model"] != response_model or state["effort"] != args.effort:
                raise CliFailure("created runtime settings did not match the request")
            worker = {
                "id": worker_id,
                "cwd": str(args.worker_cwd),
                "name": info.get("name"),
                "relation": None,
                "parent_session_id": None,
                "runtime_loaded": True,
                "model": state["model"],
                "effort": state["effort"],
                "is_streaming": state["is_streaming"],
                "is_prompt_running": state["is_prompt_running"],
                "transient": info.get("transient"),
                "message_count": info.get("messageCount"),
            }
            receipt.update(worker=worker, phase="set_session_name")

            named = client.request("POST", f"/api/agent/{worker_id}", {
                "type": "set_session_name", "name": args.name,
            })
            accepted(named)
            receipt.update(phase="verify_session_name")
            named_metadata = client.request("GET", f"/api/sessions/{worker_id}?tree=summary&tail=1")
            if named_metadata.get("sessionId") != worker_id:
                raise CliFailure("named session response identity mismatch")
            named_info = _validate_info(named_metadata.get("info"), worker_id, args.worker_cwd)
            if named_info.get("name") != args.name:
                raise CliFailure("session name did not match the request")
            worker["name"] = named_info.get("name")

            current_runtime = client.request("GET", f"/api/agent/{worker_id}")
            if current_runtime.get("running") is not True:
                raise CliFailure("new session runtime disappeared before its first prompt")
            current_state = _validate_state(current_runtime.get("state"), worker_id)
            if current_state["model"] != worker["model"] or current_state["effort"] != worker["effort"]:
                raise CliFailure("new session model or effort changed before its first prompt")

            envelope = make_envelope(
                message,
                dispatch_id=dispatch_id,
                sender_id=args.lead_session,
                sender_role="lead",
                recipient_id=worker_id,
                recipient_role="worker",
                reply_to_id=args.reply_to,
                report_reference=reference,
            )
            return_command = return_command_builder(
                args.base_url, args.lead_session, worker_id, args.lead_cwd,
                args.worker_cwd, dispatch_id,
            )
            first_prompt = (
                envelope + "\n\nDirect return to the lead:\n" + return_command
                + "\nUse a new report file and new receipt path. A successful API acknowledgment means the "
                "report was accepted or queued, not that it was completed. Default delivery is queue; "
                "add --delivery steering only for an urgent correction to the lead’s ongoing work. "
                "Steering does not cancel external jobs."
            )
            receipt.update(status="delivery_unknown", phase="send_first_prompt",
                           worker=worker, sent_sha256=message_hash(first_prompt))
            try:
                sent = client.request("POST", f"/api/agent/{worker_id}", {
                    "type": "prompt",
                    "message": first_prompt,
                    "streamingBehavior": "steer" if args.delivery == "steering" else "followUp",
                })
                accepted(sent)
            except ApiFailure as exc:
                if isinstance(exc.payload, dict) and exc.payload.get("accepted") is False:
                    receipt.update(status="rejected", phase="send_first_prompt", error_type=exc.category,
                                   http_status=exc.status)
                raise
            receipt.update(status="accepted_or_queued", phase="submitted",
                           acknowledgment="api_accepted_or_queued")
            return {
                "status": "accepted_or_queued",
                "acknowledgment": "api_accepted_or_queued",
                "dispatch_id": dispatch_id,
                "delivery": args.delivery,
                "lead": lead,
                "worker": worker,
                "receipt": str(args.receipt),
            }
        except Exception as exc:
            # Preserve any ID returned with an error body; creation may have
            # reached Pi Web even when its response was not successful.
            if receipt.data.get("worker_session_id") is None and isinstance(exc, ApiFailure):
                worker_id = _extract_return_id(exc.payload)
                if worker_id is not None:
                    receipt.update(worker_session_id=worker_id)
            if receipt.data.get("status") == "creation_unknown":
                receipt.update(status="creation_unknown", error_type=safe_error_type(exc),
                               http_status=request_error_status(exc))
            elif receipt.data.get("status") == "delivery_unknown":
                receipt.update(error_type=safe_error_type(exc), http_status=request_error_status(exc))
            else:
                receipt.update(error_type=safe_error_type(exc), http_status=request_error_status(exc))
            raise


def _check_preserved_settings(before: dict[str, object], after: dict[str, object]) -> None:
    if after.get("model") is None or after.get("effort") not in THINKING_LEVELS:
        raise CliFailure("recipient runtime model and effort must be known before sending")
    for key in ("model", "effort"):
        expected = before.get(key)
        if expected is not None and after.get(key) != expected:
            raise CliFailure(f"recipient {key} changed while loading; no prompt was sent")


def _send_existing_route(
    client: PiWebClient,
    args: object,
    sender: dict[str, object],
    recipient: dict[str, object],
    *,
    sender_role: str,
    recipient_role: str,
    sender_alias: str | None,
    recipient_alias: str | None,
    sender_cwd: str | None = None,
    recipient_cwd: str | None = None,
    receipt_route: dict[str, object],
    summary: dict[str, object],
) -> dict[str, object]:
    message = read_message(args.message, args.text)
    reference = build_report_reference(
        args.report_path, args.report_section, args.revision, args.supersedes
    )
    sender_id = str(sender["id"])
    recipient_id = str(recipient["id"])
    dispatch_id = str(uuid4())
    envelope = make_envelope(
        message,
        dispatch_id=dispatch_id,
        sender_id=sender_id,
        sender_role=sender_role,
        recipient_id=recipient_id,
        recipient_role=recipient_role,
        reply_to_id=args.reply_to,
        sender_alias=sender_alias,
        recipient_alias=recipient_alias,
        sender_cwd=sender_cwd,
        recipient_cwd=recipient_cwd,
        report_reference=reference,
    )
    receipt_data: dict[str, object] = {
        "kind": "pi_web_peer_session",
        "delivery": args.delivery,
        "status": "prepared",
        "phase": "load_target_state",
        "dispatch_id": dispatch_id,
        **receipt_route,
        "reply_to_dispatch_id": args.reply_to,
        "message_sha256": message_hash(message),
        "sent_sha256": message_hash(envelope),
        **_create_receipt_reference(reference),
    }
    with Receipt(args.receipt, receipt_data) as receipt:
        try:
            # The durable exclusive receipt precedes POST get_state, which may
            # load an unloaded persisted session.
            loaded = client.request("POST", f"/api/agent/{recipient_id}", {"type": "get_state"})
            accepted(loaded)
            target_state = _validate_state(loaded.get("data"), recipient_id)
            _check_preserved_settings(recipient, target_state)
            receipt.update(status="prepared", phase="send_prompt", loaded_target={
                "id": recipient_id,
                "model": target_state["model"],
                "effort": target_state["effort"],
                "is_streaming": target_state["is_streaming"],
                "is_prompt_running": target_state["is_prompt_running"],
            })
            receipt.update(status="delivery_unknown", phase="send_prompt")
            try:
                sent = client.request("POST", f"/api/agent/{recipient_id}", {
                    "type": "prompt",
                    "message": envelope,
                    "streamingBehavior": "steer" if args.delivery == "steering" else "followUp",
                })
                accepted(sent)
            except ApiFailure as exc:
                if isinstance(exc.payload, dict) and exc.payload.get("accepted") is False:
                    receipt.update(status="rejected", phase="send_prompt", error_type=exc.category,
                                   http_status=exc.status)
                raise
            receipt.update(status="accepted_or_queued", phase="submitted",
                           acknowledgment="api_accepted_or_queued")
        except Exception as exc:
            if receipt.data.get("status") != "rejected":
                receipt.update(error_type=safe_error_type(exc), http_status=request_error_status(exc))
            raise
    summary.update(
        status="accepted_or_queued",
        acknowledgment="api_accepted_or_queued",
        dispatch_id=dispatch_id,
        reply_to_dispatch_id=args.reply_to,
        receipt=str(args.receipt),
        delivery=args.delivery,
    )
    return summary


def inspect_or_send_pair(client: PiWebClient, args: object) -> dict[str, object]:
    """Retain the legacy lead/worker presentation and route through one send guard."""
    lead = inspect_session(client, args.lead_session, args.lead_cwd)
    worker = inspect_session(client, args.worker_session, args.worker_cwd)
    target_role = args.to
    target_id = args.lead_session if target_role == "lead" else args.worker_session
    sender_role = "worker" if target_role == "lead" else "lead"
    sender = worker if target_role == "lead" else lead
    recipient = lead if target_role == "lead" else worker
    summary: dict[str, object] = {
        "lead": lead,
        "worker": worker,
        "target": target_role,
        "target_session_id": target_id,
        **_cwd_fields(args.lead_cwd, args.worker_cwd),
        "status": "inspected_only",
    }
    if not args.send:
        return summary
    return _send_existing_route(
        client,
        args,
        sender,
        recipient,
        sender_role=sender_role,
        recipient_role=target_role,
        sender_alias=None,
        recipient_alias=None,
        receipt_route={
            "lead": lead,
            "worker": worker,
            "target": target_role,
            "target_session_id": target_id,
            **_cwd_fields(args.lead_cwd, args.worker_cwd),
        },
        summary=summary,
    )


def inspect_or_send_peers(
    client: PiWebClient,
    args: object,
    sender_binding: object,
    recipient_binding: object,
) -> dict[str, object]:
    """Inspect two explicit independent peers and optionally prompt only the recipient."""
    sender = inspect_session(client, sender_binding.session_id, sender_binding.cwd)
    recipient = inspect_session(client, recipient_binding.session_id, recipient_binding.cwd)
    sender_alias = sender_binding.alias
    recipient_alias = recipient_binding.alias
    summary: dict[str, object] = {
        "sender": sender,
        "recipient": recipient,
        "sender_session_id": sender_binding.session_id,
        "recipient_session_id": recipient_binding.session_id,
        "sender_cwd": str(sender_binding.cwd),
        "recipient_cwd": str(recipient_binding.cwd),
        "sender_alias": sender_alias,
        "recipient_alias": recipient_alias,
        "status": "inspected_only",
    }
    if not args.send:
        return summary
    receipt_route: dict[str, object] = {
        "sender": sender,
        "recipient": recipient,
        "sender_session_id": sender_binding.session_id,
        "recipient_session_id": recipient_binding.session_id,
        "sender_cwd": str(sender_binding.cwd),
        "recipient_cwd": str(recipient_binding.cwd),
    }
    if sender_alias is not None:
        receipt_route["sender_alias"] = sender_alias
    if recipient_alias is not None:
        receipt_route["recipient_alias"] = recipient_alias
    return _send_existing_route(
        client,
        args,
        sender,
        recipient,
        sender_role="peer",
        recipient_role="peer",
        sender_alias=sender_alias,
        recipient_alias=recipient_alias,
        sender_cwd=str(sender_binding.cwd),
        recipient_cwd=str(recipient_binding.cwd),
        receipt_route=receipt_route,
        summary=summary,
    )
