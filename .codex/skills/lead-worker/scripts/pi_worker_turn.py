#!/usr/bin/env python3
"""Create, inspect, or message a normal peer session through Pi Web's local API."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from uuid import uuid4

# Direct execution sets sys.path[0] to this directory; runpy.run_path from an
# unrelated cwd does not. Keep the retained script path a self-contained public CLI.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from peer_transport.api import (  # noqa: E402
    ApiFailure,
    CliFailure,
    PiWebClient,
    THINKING_LEVELS,
    error_text,
    list_models,
    normalize_base_url,
    parse_uuid,
)
from peer_transport.bindings import (  # noqa: E402
    BindingFailure,
    resolve_aliases,
    resolve_explicit,
)
from peer_transport.dispatch import (  # noqa: E402
    build_report_reference,
    create_and_send,
    inspect_or_send_pair,
    inspect_or_send_peers,
)


def _resolve_cwd(parser: argparse.ArgumentParser, option: str, path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        parser.error(f"{option} must be an existing directory ({type(exc).__name__})")
    if not resolved.is_dir():
        parser.error(f"{option} must be an existing directory")
    return resolved


def _selected_once(parser: argparse.ArgumentParser, option: str,
                   values: list[str] | None) -> str:
    if values is None or len(values) != 1:
        parser.error(f"{option} must be provided exactly once")
    return values[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:12345", type=normalize_base_url,
                        help="Pi Web loopback endpoint (default: http://127.0.0.1:12345)")
    parser.add_argument("--lead-session", type=parse_uuid)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--worker-session", type=parse_uuid)
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--list-models", action="store_true",
                      help="list current provider/model/effort choices without creating a session")
    parser.add_argument("--cwd", type=Path,
                        help="common expected cwd for both legacy lead/worker sessions")
    parser.add_argument("--lead-cwd", type=Path, help="expected physical cwd of the lead session")
    parser.add_argument("--worker-cwd", type=Path,
                        help="expected physical cwd of the worker session; used for creation/model lookup")
    parser.add_argument("--to", choices=("worker", "lead"))
    parser.add_argument("--sender", action="append", help="one exact peer session ID or selected task alias")
    parser.add_argument("--recipient", action="append", help="one exact peer session ID or selected task alias")
    parser.add_argument("--sender-cwd", type=Path, help="required physical cwd for an explicit sender ID")
    parser.add_argument("--recipient-cwd", type=Path,
                        help="required physical cwd for an explicit recipient ID")
    parser.add_argument("--participants-state", type=Path,
                        help="explicit state file containing read-only pi_peer_transport version-1 bindings")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--delivery", choices=("queue", "steering"),
                        help="send mode: queue (default) or steering at the next agent boundary")
    parser.add_argument("--message", type=Path,
                        help="UTF-8 message file body (use --text for short inline messages)")
    parser.add_argument("--text", help="inline message text for a short peer message")
    parser.add_argument("--reply-to", type=parse_uuid, help="dispatch UUID this message replies to")
    parser.add_argument("--report-path", help="literal absolute report pointer; contents are never read")
    parser.add_argument("--report-section", help="optional section label for --report-path")
    parser.add_argument("--revision", help="brief/report revision label")
    parser.add_argument("--supersedes", help="brief/report supersession note")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--effort", help="Pi Web thinking level: off, minimal, low, medium, high, xhigh, or max")
    args = parser.parse_args(argv)

    peer_requested = any(value is not None for value in (
        args.sender, args.recipient, args.sender_cwd, args.recipient_cwd, args.participants_state,
    ))
    legacy_selector = any((args.worker_session is not None, args.create, args.list_models))
    if peer_requested:
        sender_value = _selected_once(parser, "--sender", args.sender)
        recipient_value = _selected_once(parser, "--recipient", args.recipient)
        if legacy_selector or args.lead_session is not None:
            parser.error("peer routing cannot be combined with legacy lead/worker, create, or model-list modes")
        if args.cwd is not None or args.lead_cwd is not None or args.worker_cwd is not None:
            parser.error("peer routing cannot be combined with legacy cwd options")
        if args.to is not None:
            parser.error("peer routing uses --sender and --recipient, not --to")
        if any(value is not None for value in (args.name, args.provider, args.model, args.effort)):
            parser.error("peer routing cannot create or change session settings")
        try:
            if args.participants_state is not None:
                if args.sender_cwd is not None or args.recipient_cwd is not None:
                    parser.error("--participants-state supplies peer cwds; do not combine explicit cwd options")
                args.peer_sender, args.peer_recipient = resolve_aliases(
                    args.participants_state, sender_value, recipient_value
                )
            else:
                if args.sender_cwd is None or args.recipient_cwd is None:
                    parser.error("explicit peer IDs require both --sender-cwd and --recipient-cwd")
                args.peer_sender, args.peer_recipient = resolve_explicit(
                    sender_value, args.sender_cwd, recipient_value, args.recipient_cwd
                )
        except BindingFailure as exc:
            parser.error(str(exc))
        args.peer_mode = True
    else:
        if not legacy_selector:
            parser.error("select a legacy mode or provide one --sender and one --recipient")
        if args.sender_cwd is not None or args.recipient_cwd is not None:
            parser.error("--sender-cwd and --recipient-cwd require peer routing")
        if args.participants_state is not None:
            parser.error("--participants-state requires --sender and --recipient")
        if args.cwd is not None:
            if args.lead_cwd is not None or args.worker_cwd is not None:
                parser.error("--cwd cannot be combined with --lead-cwd or --worker-cwd")
            args.cwd = _resolve_cwd(parser, "--cwd", args.cwd)
            args.lead_cwd = args.cwd
            args.worker_cwd = args.cwd
        elif args.list_models:
            if args.lead_cwd is not None or args.worker_cwd is None:
                parser.error("--list-models needs --cwd or --worker-cwd only")
            args.worker_cwd = _resolve_cwd(parser, "--worker-cwd", args.worker_cwd)
        else:
            if args.lead_cwd is None or args.worker_cwd is None:
                parser.error("provide --cwd or both --lead-cwd and --worker-cwd")
            args.lead_cwd = _resolve_cwd(parser, "--lead-cwd", args.lead_cwd)
            args.worker_cwd = _resolve_cwd(parser, "--worker-cwd", args.worker_cwd)
        if not args.list_models and args.lead_session is None:
            parser.error("--lead-session is required for create, inspect, and send")
        if args.worker_session is not None and args.worker_session == args.lead_session:
            parser.error("lead and worker sessions must be different UUIDs")
        if args.list_models and (args.lead_session is not None or args.send or args.to == "lead"):
            parser.error("--list-models needs only --base-url and --cwd or --worker-cwd")
        if args.create:
            if args.to == "lead" or not args.send:
                parser.error("--create requires --send and --to worker")
            create_only = (args.name, args.provider, args.model, args.effort)
            if not all(isinstance(value, str) and value.strip() for value in create_only):
                parser.error("--create requires nonempty --name, --provider, --model, and --effort")
            if args.effort not in THINKING_LEVELS:
                parser.error("--effort must be a supported Pi Web thinking level")
        elif any(value is not None for value in (args.name, args.provider, args.model, args.effort)):
            parser.error("--name, --provider, --model, and --effort require --create")
        args.to = args.to or "worker"
        args.peer_mode = False

    if args.send:
        if (args.message is None) == (args.text is None) or args.receipt is None:
            parser.error("--send requires exactly one of --message or --text, and a new --receipt path")
    elif (args.message is not None or args.text is not None or args.reply_to is not None
          or args.receipt is not None):
        parser.error("--message, --text, --reply-to, and --receipt require --send")
    if args.delivery is not None and not args.send:
        parser.error("--delivery requires --send")
    reference_args = (args.report_path, args.report_section, args.revision, args.supersedes)
    if any(value is not None for value in reference_args) and not args.send:
        parser.error("report-reference flags require --send")
    try:
        build_report_reference(*reference_args)
    except ValueError as exc:
        parser.error(str(exc))
    args.delivery = args.delivery or "queue"
    return args


def _command_for_return(
    base_url: str, lead_id: str, worker_id: str, lead_cwd: Path, worker_cwd: Path,
    reply_to_id: str,
) -> str:
    script = Path(__file__).resolve()
    command = [
        sys.executable, str(script), "--base-url", base_url,
        "--lead-session", lead_id, "--worker-session", worker_id,
    ]
    if lead_cwd == worker_cwd:
        command.extend(["--cwd", str(lead_cwd)])
    else:
        command.extend(["--lead-cwd", str(lead_cwd), "--worker-cwd", str(worker_cwd)])
    command.extend([
        "--to", "lead", "--send", "--message", "REPORT_FILE", "--reply-to", reply_to_id,
        "--receipt", "NEW_RECEIPT_FILE",
    ])
    return shlex.join(command)


def operate(args: argparse.Namespace) -> dict[str, object]:
    client = PiWebClient(args.base_url)
    if args.peer_mode:
        return inspect_or_send_peers(client, args, args.peer_sender, args.peer_recipient)
    if args.list_models:
        return {
            "status": "listed_only",
            "cwd": str(args.worker_cwd),
            "worker_cwd": str(args.worker_cwd),
            "models": list_models(client, args.worker_cwd),
        }
    if args.create:
        return create_and_send(client, args, _command_for_return)
    return inspect_or_send_pair(client, args)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = operate(args)
    except FileExistsError:
        print("pi_worker_turn: receipt path already exists; reconcile it before another send", file=sys.stderr)
        return 1
    except BindingFailure as exc:
        print(f"pi_worker_turn: {exc}", file=sys.stderr)
        return 1
    except (CliFailure, ApiFailure, OSError, ValueError, TypeError) as exc:
        print(f"pi_worker_turn: {error_text(exc)}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
